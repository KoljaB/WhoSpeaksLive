"""UI-independent launcher controller shared by desktop and command front ends.

The controller owns validated profile snapshots, diagnostics, operation state,
process ownership, cancellation, and event publication.  Presentation layers
may call the blocking commands from their own worker mechanism, but must never
mutate the controller's process handles or state snapshots directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import http.client
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cli_diagnostics import DoctorReport, check_local_provider_syntax, run_doctor
from .cli_installation import installer_backend_available, normalize_installer_backend
from .planning import (
    InstallPlan,
    LaunchPlan,
    build_launch_plan,
    install_plan_for_target,
    profile_for_install,
)
from .profiles import (
    PROVIDER_PRESETS,
    Profile,
    load_profile,
    profile_with_provider_preset,
    save_profile,
)
from .service_processes import (
    start_service_process,
    terminate_service_processes,
    wait_for_service_health,
)
from .tui_state import ServerState, ServerSupervisor, SetupCoordinator
from window.language_config import normalize_language_code
from window.realtime_preview_backends import (
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
)


class EventKind(str, enum.Enum):
    SNAPSHOT = "snapshot"
    OPERATION = "operation"
    REPORT = "report"
    PROFILE = "profile"
    SERVICE = "service"
    LOG = "log"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class LauncherEvent:
    kind: EventKind
    message: str = ""
    payload: object | None = None
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True)
class ServiceSnapshot:
    kind: str
    status: str
    ownership: str
    address: str


@dataclasses.dataclass(frozen=True)
class LauncherSnapshot:
    profile: Profile
    report: DoctorReport
    operation: object
    services: tuple[ServiceSnapshot, ...]
    logs: tuple[str, ...]


Listener = Callable[[LauncherEvent], None]


class _LaunchCancelled(RuntimeError):
    pass


class ProfileValidationError(ValueError):
    """A profile validation failure tied to one editable launcher field."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


class LauncherController:
    """Coordinate all mutable launcher behavior without depending on a UI toolkit."""

    SERVICE_KINDS = ("live", "reports", "translation", "macos_asr", "macos_embeddings")

    def __init__(
        self,
        profile: Profile | None = None,
        *,
        doctor_runner: Callable[..., DoctorReport] = run_doctor,
        profile_saver: Callable[[Profile], Path] = save_profile,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        remote_backend_probe: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._logs: list[str] = []
        self._doctor_runner = doctor_runner
        self._profile_saver = profile_saver
        self._popen_factory = popen_factory
        self._remote_backend_probe = remote_backend_probe
        self._clock = clock
        self.profile = profile or load_profile()
        self.report = DoctorReport(self.profile.mode, [])
        self.coordinator = SetupCoordinator(clock=clock)
        self.servers = ServerSupervisor()
        self.install_process: subprocess.Popen[str] | None = None
        self._last_probe_at = 0.0
        self._launch_spawning = False

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, kind: EventKind, message: str = "", payload: object | None = None) -> None:
        event = LauncherEvent(kind, message, payload)
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(event)

    def _append_log(self, message: str) -> None:
        if not message:
            return
        stamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"{stamp}  {message}"
        with self._lock:
            self._logs.append(line)
            if len(self._logs) > 10_000:
                del self._logs[:1_000]
        self._emit(EventKind.LOG, line, line)

    @property
    def snapshot(self) -> LauncherSnapshot:
        with self._lock:
            services = tuple(
                ServiceSnapshot(
                    kind,
                    self.servers.state(kind).status,
                    self.servers.state(kind).ownership,
                    self.service_address(kind),
                )
                for kind in self.SERVICE_KINDS
            )
            return LauncherSnapshot(
                profile=self.profile,
                report=self.report,
                operation=self.coordinator.snapshot.operation,
                services=services,
                logs=tuple(self._logs),
            )

    def service_address(self, kind: str) -> str:
        if kind == "macos_asr":
            if self.profile.mode == "local":
                return f"In process · {self.profile.model}"
            return self.profile.remote_asr_url
        if kind == "macos_embeddings":
            if self.profile.mode == "local":
                preset = self.profile.provider_preset.replace("_", " ").title()
                return f"In process · {preset} preset"
            return self.profile.remote_embeddings_url
        return {
            "live": f"{self.profile.host}:{self.profile.port}",
            "reports": f"{self.profile.host}:{self.profile.reports_port}",
            "translation": f"{self.profile.host}:{self.profile.translation_port}",
        }.get(kind, "")

    @staticmethod
    def _coerce_port(name: str, value: object) -> int:
        try:
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise ProfileValidationError(
                name, f"{name.replace('_', ' ').title()} must be an integer."
            ) from exc
        if not 1 <= port <= 65535:
            raise ProfileValidationError(
                name, f"{name.replace('_', ' ').title()} must be between 1 and 65535."
            )
        return port

    def validate_profile_updates(self, updates: Mapping[str, object]) -> Profile:
        values = dict(updates)
        profile_fields = {field.name for field in dataclasses.fields(Profile)}
        unknown = sorted(set(values) - profile_fields)
        if unknown:
            raise ProfileValidationError(
                unknown[0], f"Unknown profile setting: {', '.join(unknown)}"
            )
        for field in ("port", "reports_port", "translation_port"):
            if field in values:
                values[field] = self._coerce_port(field, values[field])
        if "translation_max_targets" in values:
            try:
                targets = int(values["translation_max_targets"])
            except (TypeError, ValueError) as exc:
                raise ProfileValidationError(
                    "translation_max_targets",
                    "Maximum translation targets must be an integer.",
                ) from exc
            if not 1 <= targets <= 16:
                raise ProfileValidationError(
                    "translation_max_targets",
                    "Maximum translation targets must be between 1 and 16.",
                )
            values["translation_max_targets"] = targets
        if "host" in values and not str(values["host"]).strip():
            raise ProfileValidationError("host", "Browser host cannot be empty.")
        enum_fields = {
            "mode": {"local", "remote", "server"},
            "deployment_target": {"", "macos"},
            "asr_backend": {"local", "remote"},
            "embeddings_backend": {"local", "remote"},
            "provider_preset": {"custom", *PROVIDER_PRESETS},
            "vad_backend": {"rms", "silero"},
            "report_llm_provider": {
                "llama_cpp",
                "ollama",
                "lm_studio",
                "openai_compatible",
                "openai",
                "openrouter",
            },
            "translation_provider": {
                "sidecar",
                "transformers",
                "reports_llm",
                "openai_compatible",
                "deepl",
                "google_cloud",
                "azure_translator",
                "libretranslate",
            },
            "translation_model_profile": {
                "translate-gemma-4b",
                "nllb-200-600m",
                "madlad-400-3b",
            },
            "translation_device": {"auto", "cuda", "cpu"},
            "device": {"auto", "cuda", "cpu"},
        }
        for field, allowed in enum_fields.items():
            if field not in values:
                continue
            selected = str(values[field])
            if selected not in allowed:
                raise ProfileValidationError(
                    field,
                    f"Unsupported {field.replace('_', ' ')} {selected!r}. "
                    f"Choose one of: {', '.join(sorted(allowed)) or '(blank)'}."
                )
        for field in ("language", "report_language"):
            if field not in values or (field == "report_language" and not str(values[field]).strip()):
                continue
            try:
                normalized = normalize_language_code(str(values[field]))
            except ValueError as exc:
                raise ProfileValidationError(
                    field,
                    f"Unsupported {field.replace('_', ' ')}: {values[field]!r}.",
                ) from exc
            values[field] = normalized
        if "realtime_preview_engine" in values:
            try:
                engine = normalize_preview_engine(values["realtime_preview_engine"])
            except ValueError as exc:
                raise ProfileValidationError("realtime_preview_engine", str(exc)) from exc
            if engine == "mock":
                raise ProfileValidationError(
                    "realtime_preview_engine",
                    "The mock realtime engine is for tests and cannot be saved by the launcher.",
                )
            values["realtime_preview_engine"] = engine
            preset_value = values.get(
                "realtime_preview_model_preset",
                self.profile.realtime_preview_model_preset,
            )
            if engine != "off":
                try:
                    values["realtime_preview_model_preset"] = normalize_preview_model_preset(
                        engine,
                        preset_value,
                    )
                except (ValueError, argparse.ArgumentTypeError) as exc:
                    raise ProfileValidationError(
                        "realtime_preview_model_preset",
                        f"The selected live model is not compatible with {engine}.",
                    ) from exc
            else:
                values["realtime_preview_model_preset"] = ""
        requested_mode = str(values.get("mode", self.profile.mode))
        for backend_field in ("asr_backend", "embeddings_backend"):
            if requested_mode in {"local", "remote"} and backend_field in values:
                if str(values[backend_field]) != requested_mode:
                    raise ProfileValidationError(
                        "mode",
                        f"{backend_field.replace('_', ' ').title()} must be {requested_mode} "
                        f"for the selected deployment."
                    )
        if "translation_target_languages" in values:
            source_language = str(values.get("language", self.profile.language))
            targets: list[str] = []
            for raw_target in re.split(r"[,;\s]+", str(values["translation_target_languages"] or "")):
                if not raw_target:
                    continue
                try:
                    target = normalize_language_code(raw_target)
                except ValueError as exc:
                    raise ProfileValidationError(
                        "translation_target_languages",
                        f"Unsupported translation target: {raw_target!r}.",
                    ) from exc
                if target == source_language:
                    raise ProfileValidationError(
                        "translation_target_languages",
                        "A translation target must differ from the live language.",
                    )
                if target not in targets:
                    targets.append(target)
            maximum = int(values.get("translation_max_targets", self.profile.translation_max_targets))
            if len(targets) > maximum:
                raise ProfileValidationError(
                    "translation_target_languages",
                    f"Choose at most {maximum} translation target languages."
                )
            values["translation_target_languages"] = ",".join(targets)
        reports_enabled = bool(values.get("reports_enabled", self.profile.reports_enabled))
        translation_enabled = bool(
            values.get("translation_enabled", self.profile.translation_enabled)
        )
        translation_provider = str(
            values.get("translation_provider", self.profile.translation_provider)
        )
        needs_report_llm = reports_enabled or (
            translation_enabled and translation_provider == "reports_llm"
        )
        if needs_report_llm and not str(
            values.get("report_llm_model", self.profile.report_llm_model)
        ).strip():
            raise ProfileValidationError(
                "report_llm_model",
                "Choose the Meeting Intelligence model exposed by the selected provider.",
            )
        if translation_enabled and translation_provider == "openai_compatible":
            if not str(
                values.get("translation_base_url", self.profile.translation_base_url)
            ).strip():
                raise ProfileValidationError(
                    "translation_base_url",
                    "Enter the OpenAI-compatible translation endpoint.",
                )
            if not str(
                values.get("translation_model", self.profile.translation_model)
            ).strip():
                raise ProfileValidationError(
                    "translation_model",
                    "Enter the model ID exposed by the translation endpoint.",
                )
        relevant_key_fields = set()
        if reports_enabled and (
            str(values.get("text_embedding_base_url", self.profile.text_embedding_base_url)).strip()
            or str(values.get("text_embedding_model", self.profile.text_embedding_model)).strip()
        ):
            relevant_key_fields.add("text_embedding_api_key_env")
        if translation_enabled and translation_provider not in {
            "sidecar",
            "transformers",
            "reports_llm",
        }:
            relevant_key_fields.add("translation_api_key_env")
        for field in relevant_key_fields:
            if field not in values or not str(values[field]).strip():
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(values[field]).strip()) is None:
                raise ProfileValidationError(
                    field,
                    f"{field.replace('_', ' ').title()} must be an environment-variable name, not a secret value."
                )
        deployment_target = str(
            values.get("deployment_target", self.profile.deployment_target)
        )
        relevant_url_fields: set[str] = set()
        required_url_fields: set[str] = set()
        if requested_mode == "remote" and deployment_target != "macos":
            relevant_url_fields.update(("remote_asr_url", "remote_embeddings_url"))
            required_url_fields.update(("remote_asr_url", "remote_embeddings_url"))
        if needs_report_llm:
            relevant_url_fields.add("report_llm_base_url")
        if reports_enabled and (
            str(values.get("text_embedding_base_url", self.profile.text_embedding_base_url)).strip()
            or str(values.get("text_embedding_model", self.profile.text_embedding_model)).strip()
        ):
            relevant_url_fields.add("text_embedding_base_url")
        if translation_enabled and translation_provider not in {
            "sidecar",
            "transformers",
            "reports_llm",
        }:
            relevant_url_fields.add("translation_base_url")
        for field in relevant_url_fields:
            effective_value = str(
                values.get(field, getattr(self.profile, field, ""))
            ).strip()
            if not effective_value:
                if field in required_url_fields:
                    raise ProfileValidationError(
                        field,
                        f"{field.replace('_', ' ').title()} is required for remote deployment.",
                    )
                continue
            parsed = urlsplit(effective_value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ProfileValidationError(
                    field,
                    f"{field.replace('_', ' ').title()} must be a complete http:// or https:// URL.",
                )
        active_ports: list[tuple[str, int]] = []
        if requested_mode != "server":
            active_ports.append(("port", int(values.get("port", self.profile.port))))
            if reports_enabled:
                active_ports.append(
                    ("reports_port", int(values.get("reports_port", self.profile.reports_port)))
                )
            if translation_enabled and translation_provider == "sidecar":
                active_ports.append(
                    (
                        "translation_port",
                        int(values.get("translation_port", self.profile.translation_port)),
                    )
                )
        used_ports: dict[int, str] = {}
        for field, port in active_ports:
            other = used_ports.get(port)
            if other is not None:
                raise ProfileValidationError(
                    field,
                    f"{field.replace('_', ' ').title()} conflicts with {other.replace('_', ' ')} ({port}).",
                )
            used_ports[port] = field
        if "advanced_args" in values:
            try:
                shlex.split(str(values["advanced_args"] or ""))
            except ValueError as exc:
                raise ProfileValidationError(
                    "advanced_args", f"Advanced launch arguments are invalid: {exc}"
                ) from exc
        current = self.profile
        preset = values.pop("provider_preset", None)
        if preset is not None:
            current = profile_with_provider_preset(current, str(preset))
        candidate = current.with_updates(**values)
        if candidate.mode != "server":
            if candidate.mode == "local" and not candidate.model.strip():
                raise ProfileValidationError("model", "Choose a final ASR model.")
            if candidate.mode == "local" and not candidate.compute_type.strip():
                raise ProfileValidationError("compute_type", "Choose or enter a compute type.")
            if candidate.provider_preset == "custom":
                for field in ("embedding_provider", "live_speaker_embedding_provider"):
                    result = check_local_provider_syntax(
                        str(getattr(candidate, field)),
                        required=(field == "embedding_provider" or candidate.live_speaker_assignment),
                    )
                    if result.status == "fail":
                        raise ProfileValidationError(field, result.detail)
        text_embedding_base = candidate.text_embedding_base_url.strip()
        text_embedding_model = candidate.text_embedding_model.strip()
        if candidate.reports_enabled and bool(text_embedding_base) != bool(text_embedding_model):
            missing = "text_embedding_model" if text_embedding_base else "text_embedding_base_url"
            raise ProfileValidationError(
                missing,
                "Semantic search needs both a text embedding base URL and model, or neither.",
            )
        executable_fields = []
        if candidate.mode == "local" and candidate.embedding_python.strip():
            executable_fields.append("embedding_python")
        if candidate.realtime_preview_engine == "kroko_onnx" and candidate.realtime_preview_python.strip():
            executable_fields.append("realtime_preview_python")
        if (
            candidate.translation_enabled
            and candidate.translation_provider == "sidecar"
            and candidate.translation_python.strip()
        ):
            executable_fields.append("translation_python")
        for field in executable_fields:
            executable = str(getattr(candidate, field)).strip()
            if not Path(executable).expanduser().is_file() and shutil.which(executable) is None:
                raise ProfileValidationError(
                    field,
                    f"The configured Python executable does not exist: {executable}",
                )
        compatibility_error = preview_language_error(
            candidate.realtime_preview_engine,
            candidate.language,
        )
        if compatibility_error and candidate.mode != "server":
            raise ProfileValidationError("language", compatibility_error)
        return candidate

    def update_profile(self, updates: Mapping[str, object], *, persist: bool = True) -> Profile:
        candidate = self.validate_profile_updates(updates)
        path: Path | None = None
        if persist:
            path = self._profile_saver(candidate)
        with self._lock:
            self.profile = candidate
        detail = f"Saved launch profile: {path}" if path is not None else "Updated launch profile"
        self._append_log(detail)
        self._emit(EventKind.PROFILE, detail, candidate)
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return candidate

    def run_diagnostics(self, *, deep: bool = False) -> DoctorReport:
        title = "Running complete diagnostics" if deep else "Checking system readiness"
        self.coordinator.start_operation("doctor", title, "Inspecting installed components")
        self._emit(EventKind.OPERATION, title, self.coordinator.snapshot.operation)
        self._append_log("Starting complete diagnostics…" if deep else "Starting readiness check…")
        try:
            report = self._doctor_runner(self.profile, self.profile.mode, deep=deep)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.coordinator.finish_operation("error", "Diagnostics failed", detail)
            self._append_log(f"Diagnostics failed: {detail}")
            self._emit(EventKind.ERROR, detail, exc)
            self._emit(EventKind.OPERATION, detail, self.coordinator.snapshot.operation)
            raise
        with self._lock:
            self.report = report
        statuses = {check.status for check in report.checks}
        if "fail" in statuses:
            status, result = "error", "Readiness check found required fixes"
        elif "warn" in statuses:
            status, result = "warning", "Readiness check found warnings"
        else:
            status, result = "success", "Readiness check completed"
        counts = {name: sum(check.status == name for check in report.checks) for name in ("ok", "warn", "fail", "skip")}
        detail = f"{counts['ok']} passed, {counts['warn']} warnings, {counts['fail']} failed, {counts['skip']} skipped"
        self.coordinator.finish_operation(status, result, detail)
        self._append_log(detail)
        self._emit(EventKind.REPORT, detail, report)
        self._emit(EventKind.OPERATION, detail, self.coordinator.snapshot.operation)
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return report

    def install_plan(
        self,
        target: str,
        *,
        realtime_preview_engine: str,
        realtime_preview_model_preset: str = "",
        translation_model_profile: str = "off",
    ) -> InstallPlan:
        return install_plan_for_target(
            target,
            realtime_preview_engine=realtime_preview_engine,
            realtime_preview_model_preset=realtime_preview_model_preset,
            translation_model_profile=translation_model_profile,
        )

    def configure_for_install(
        self,
        plan: InstallPlan,
        *,
        language: str,
        live_speaker_assignment: bool,
        persist: bool = True,
    ) -> Profile:
        """Persist exactly the complete profile represented by an install plan."""

        configured = profile_for_install(self.profile, plan)
        return self.update_profile(
            {
                **configured.as_dict(),
                "language": language,
                "live_speaker_assignment": live_speaker_assignment,
            },
            persist=persist,
        )

    @staticmethod
    def preferred_installer() -> str:
        """Prefer uv when it is usable, with pip as the safe fallback."""

        return "uv" if installer_backend_available("uv") else "pip"

    @staticmethod
    def install_command(
        plan: InstallPlan,
        *,
        installer: str = "pip",
        model_dir: str = "",
    ) -> list[str]:
        selected_installer = normalize_installer_backend(installer)
        command = [
            sys.executable,
            "-m",
            "whospeaks_cli",
            "install",
            "--target",
            plan.target,
            "--installer",
            selected_installer,
            "--yes",
        ]
        if plan.target != "server":
            command.extend(["--realtime-preview-engine", plan.realtime_preview_engine])
            if plan.realtime_preview_model_preset:
                command.extend(["--realtime-preview-model-preset", plan.realtime_preview_model_preset])
            if plan.realtime_preview_engine == "sherpa_onnx" and model_dir.strip():
                command.extend(["--realtime-preview-model-dir", model_dir.strip()])
        command.extend(["--translation-model-profile", plan.translation_model_profile])
        return command

    def install(self, command: Iterable[str], *, title: str) -> int:
        if self.coordinator.snapshot.operation.name:
            raise RuntimeError("Another launcher operation is already running.")
        concrete = [str(part) for part in command]
        self.coordinator.start_operation("install", f"Install: {title}", "Starting installer")
        self._emit(EventKind.OPERATION, title, self.coordinator.snapshot.operation)
        self._append_log(f"> {' '.join(concrete)}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            process = self._popen_factory(concrete, **kwargs)
            self.install_process = process
            if self.coordinator.snapshot.operation.cancel_requested and process.poll() is None:
                terminate_service_processes([process])
            if process.stdout is not None:
                for raw in process.stdout:
                    line = raw.rstrip()
                    self._append_log(line)
                    self.coordinator.update_progress(step=self.install_step_for_line(line), latest=line)
                    self._emit(EventKind.OPERATION, line, self.coordinator.snapshot.operation)
            return_code = int(process.wait())
        except Exception as exc:
            return_code = 1
            self._append_log(f"Installer failed: {type(exc).__name__}: {exc}")
            self._emit(EventKind.ERROR, str(exc), exc)
        finally:
            self.install_process = None
        cancelled = self.coordinator.snapshot.operation.cancel_requested
        if cancelled:
            self.coordinator.finish_operation("warning", "Installation cancelled", "The running installer was stopped.")
        elif return_code == 0:
            self.coordinator.finish_operation("success", "Installation completed", "Packages were installed.")
        else:
            self.coordinator.finish_operation(
                "error",
                "Installation failed",
                f"Installer stopped with exit code {return_code}.",
            )
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        return return_code

    @staticmethod
    def install_step_for_line(line: str) -> str:
        lowered = line.lower()
        if any(token in lowered for token in ("pytorch", "torch", "torchaudio", "cuda")):
            return "Installing PyTorch runtime"
        if any(token in lowered for token in ("sherpa", "nemotron", "model download", "model archive")):
            return "Preparing Nemotron realtime ASR"
        if any(token in lowered for token in ("kroko", "docker", "cmake", "native runtime")):
            return "Preparing Kroko realtime ASR"
        if any(token in lowered for token in ("collecting ", "downloading ", "building wheel", "pip install", "uv pip")):
            return "Installing Python packages"
        if any(token in lowered for token in ("saved ", "configuration", "profile")):
            return "Saving configuration"
        if any(token in lowered for token in ("check", "doctor", "readiness")):
            return "Checking installed components"
        return "Running installer"

    def cancel_operation(self) -> bool:
        operation_name = self.coordinator.snapshot.operation.name
        if operation_name not in {"install", "launch"}:
            return False
        self.coordinator.request_cancel()
        self.coordinator.update_progress(
            step=f"Cancelling {operation_name}",
            latest="Stopping processes started by this operation…",
        )
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        if operation_name == "install":
            process = self.install_process
            if process is not None and process.poll() is None:
                terminate_service_processes([process])
        else:
            launch_was_still_spawning = self._launch_spawning
            self.stop_owned_services()
            if not launch_was_still_spawning:
                self.coordinator.finish_operation(
                    "warning",
                    "Launch cancelled",
                    "Services started during this attempt were stopped.",
                )
                self._append_log("Launch cancelled; stopped services from this attempt.")
                self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
                self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return True

    @staticmethod
    def _port_accepting(host: str, port: int, *, timeout: float = 0.08) -> bool:
        probe_host = str(host or "127.0.0.1").strip()
        if probe_host in {"0.0.0.0", "::", "[::]"}:
            probe_host = "127.0.0.1"
        try:
            with socket.create_connection((probe_host, int(port)), timeout=timeout):
                return True
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _http_service_ready(
        host: str,
        port: int,
        *,
        path: str = "/health",
        timeout: float = 0.3,
    ) -> bool:
        """Return true only when the service answers its application health check."""

        probe_host = str(host or "127.0.0.1").strip()
        if probe_host in {"0.0.0.0", "::", "[::]"}:
            probe_host = "127.0.0.1"
        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection(probe_host, int(port), timeout=timeout)
            connection.request("GET", path, headers={"Connection": "close"})
            response = connection.getresponse()
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not bool(payload.get("ok", True)):
                return False
            if "ready" in payload and not bool(payload.get("ready")):
                return False
            readiness = str(payload.get("readiness") or "ready").lower()
            return readiness == "ready"
        except (OSError, TypeError, ValueError, json.JSONDecodeError, http.client.HTTPException):
            return False
        finally:
            if connection is not None:
                connection.close()

    def _service_ready(self, kind: str, port: int) -> bool:
        # The inexpensive socket check avoids waiting on HTTP timeouts for
        # services that have not bound their port yet.  It is not itself a
        # readiness signal: the health response below is authoritative.
        if not self._port_accepting(self.profile.host, port):
            return False
        return self._http_service_ready(self.profile.host, port)

    @staticmethod
    def _remote_backend_available(base_url: str, *, timeout: float = 0.5) -> bool:
        """Probe a configured remote backend without requiring its model to be preloaded."""

        raw_url = str(base_url or "").strip()
        if not raw_url:
            return False
        try:
            parsed = urlsplit(raw_url if "://" in raw_url else f"http://{raw_url}")
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError:
            return False
        if not host or parsed.scheme not in {"http", "https"}:
            return False
        base_path = parsed.path.rstrip("/")
        health_path = base_path if base_path.endswith("/health") else f"{base_path}/health"
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection: http.client.HTTPConnection | None = None
        try:
            connection = connection_type(host, port, timeout=timeout)
            connection.request("GET", health_path or "/health", headers={"Connection": "close"})
            response = connection.getresponse()
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return isinstance(payload, dict) and payload.get("ok") is not False
        except (OSError, TypeError, ValueError, json.JSONDecodeError, http.client.HTTPException):
            return False
        finally:
            if connection is not None:
                connection.close()

    def _refresh_remote_backends(self) -> None:
        if self.profile.mode == "local":
            live_state = self.servers.state("live")
            for kind in ("macos_asr", "macos_embeddings"):
                transition = self.servers.mirror_component(kind, live_state)
                if transition.previous != transition.current:
                    self._emit(EventKind.SERVICE, kind, transition)
            return
        if self.profile.mode != "remote":
            for kind in ("macos_asr", "macos_embeddings"):
                if not self.servers.process_is_running(kind):
                    self.servers.clear(kind)
            return
        endpoints = {
            "macos_asr": self.profile.remote_asr_url,
            "macos_embeddings": self.profile.remote_embeddings_url,
        }
        for kind, base_url in endpoints.items():
            probe = self._remote_backend_probe or self._remote_backend_available
            transition = self.servers.observe_backend(
                kind,
                available=probe(base_url),
            )
            if transition.previous != transition.current or transition.exit_code is not None:
                self._emit(EventKind.SERVICE, kind, transition)
                label = {
                    "macos_asr": "Final ASR backend",
                    "macos_embeddings": "Speaker embeddings backend",
                }[kind]
                if transition.current.status == "running":
                    self._append_log(f"{label} is available at {base_url}")
                elif transition.current.status == "unavailable":
                    self._append_log(f"{label} is unavailable at {base_url}")
                if transition.exit_code is not None:
                    self._append_log(f"{kind} exited with code {transition.exit_code}")

    def _spawn_detached(self, kind: str, command: Iterable[str]) -> subprocess.Popen[str]:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        concrete = [str(part) for part in command]
        try:
            process = self._popen_factory(concrete, **kwargs)
        except Exception:
            self.servers.fail_start(kind)
            self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
            raise
        self.servers.begin(kind, process)
        self._append_log(f"Started {kind}: {' '.join(concrete)}")
        self._stream_service_output(kind, process)
        self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
        return process

    def _stream_service_output(self, kind: str, process: subprocess.Popen[str]) -> None:
        stream = getattr(process, "stdout", None)
        if stream is None:
            return

        def read_output() -> None:
            try:
                for raw in stream:
                    line = str(raw).rstrip()
                    if line:
                        self._append_log(f"{kind}: {line}")
                        if kind == "live" and self.coordinator.snapshot.operation.name == "launch":
                            lowered = line.lower()
                            if "preparing asr, embeddings, and vad" in lowered:
                                step = "Warming speech models"
                            elif "startup model warmup complete" in lowered:
                                step = "Starting the live web server"
                            elif "serving growing-window" in lowered:
                                step = "Checking the live window"
                            else:
                                step = ""
                            if step:
                                self.coordinator.update_progress(step=step, latest=line)
                                self._emit(EventKind.OPERATION, line, self.coordinator.snapshot.operation)
            except (OSError, ValueError) as exc:
                self._append_log(f"Could not read {kind} output: {type(exc).__name__}: {exc}")

        threading.Thread(
            target=read_output,
            name=f"whospeaks-{kind}-output",
            daemon=True,
        ).start()

    def _raise_if_launch_cancelled(self) -> None:
        operation = self.coordinator.snapshot.operation
        if operation.name == "launch" and operation.cancel_requested:
            raise _LaunchCancelled("Launch cancelled by the user.")

    def _guard_external_port(self, kind: str, port: int) -> None:
        if self.servers.process_is_running(kind):
            return
        if self._port_accepting(self.profile.host, port):
            self.servers.observe(kind, listening=True, probe_due=True)
            self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
            label = {"reports": "Meeting Intelligence", "translation": "Translation"}.get(kind, kind.title())
            raise RuntimeError(f"{label} port {port} is already owned by another process.")

    def launch(self) -> LaunchPlan:
        """Start the saved launch group; call from a presentation-layer worker."""

        if self.coordinator.snapshot.operation.name:
            raise RuntimeError("Another launcher operation is already running.")
        if self.profile.mode == "server":
            raise RuntimeError(
                "A server profile exposes two independent service commands and does not launch the browser controller."
            )
        self.validate_profile_updates(self.profile.as_dict())
        plan = build_launch_plan(self.profile)
        self.coordinator.start_operation("launch", "Starting WhoSpeaks", "Checking service ports")
        self._launch_spawning = True
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        started: list[object] = []
        started_kinds: list[str] = []
        optional_errors: list[str] = []
        try:
            if self.profile.mode == "remote" and self.profile.deployment_target != "macos":
                self._refresh_remote_backends()
                unavailable_backends = [
                    label
                    for kind, label in (
                        ("macos_asr", "Final ASR"),
                        ("macos_embeddings", "speaker embeddings"),
                    )
                    if self.servers.state(kind).status != "running"
                ]
                if unavailable_backends:
                    raise RuntimeError(
                        "Required remote backends are unavailable: "
                        + ", ".join(unavailable_backends)
                        + ". Start them or correct their URLs in Settings."
                    )
            if plan.reports:
                self._guard_external_port("reports", self.profile.reports_port)
            if plan.translation:
                self._guard_external_port("translation", self.profile.translation_port)
            self._guard_external_port("live", self.profile.port)
            self._raise_if_launch_cancelled()
            for index, spec in enumerate(plan.services):
                kind = "macos_asr" if index == 0 else "macos_embeddings"
                self.coordinator.update_progress(step=f"Starting {spec.name}")
                self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
                self._raise_if_launch_cancelled()
                process = start_service_process(spec)
                self.servers.begin(kind, process)
                started.append(process)
                started_kinds.append(kind)
                self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
                wait_for_service_health(spec, process)
                self._raise_if_launch_cancelled()
                self.servers.observe(kind, listening=True, probe_due=True)
                self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
            for kind, command in (("reports", plan.reports), ("translation", plan.translation)):
                if not command:
                    continue
                self._raise_if_launch_cancelled()
                try:
                    process = self._spawn_detached(kind, command)
                except Exception as exc:
                    detail = f"{kind.replace('_', ' ').title()} failed: {type(exc).__name__}: {exc}"
                    optional_errors.append(detail)
                    self._append_log(detail)
                    self._emit(EventKind.ERROR, detail, exc)
                else:
                    started.append(process)
                    started_kinds.append(kind)
            self._raise_if_launch_cancelled()
            process = self._spawn_detached("live", plan.live)
            started.append(process)
            started_kinds.append("live")
            self._raise_if_launch_cancelled()
        except _LaunchCancelled:
            self._launch_spawning = False
            terminate_service_processes(started)
            for kind in started_kinds:
                self.servers.clear(kind)
                self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
            self.coordinator.finish_operation(
                "warning",
                "Launch cancelled",
                "Services started during this attempt were stopped.",
            )
            self._append_log("Launch cancelled; stopped services from this attempt.")
            self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
            self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
            return plan
        except Exception as exc:
            self._launch_spawning = False
            terminate_service_processes(started)
            for kind in started_kinds:
                self.servers.clear(kind)
                self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
            self.coordinator.finish_operation("error", "WhoSpeaks did not start", str(exc))
            self._append_log(f"Launch failed: {type(exc).__name__}: {exc}")
            self._emit(EventKind.ERROR, str(exc), exc)
            self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
            self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
            raise
        self._launch_spawning = False
        self.coordinator.update_progress(
            step="Warming speech models",
            latest=(
                "Live capture is warming up; one optional service already failed."
                if optional_errors
                else "The Live window will become available after its speech models and web server are ready."
            ),
        )
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        self.refresh_services(force=True)
        return plan

    def retry_service(self, kind: str) -> object:
        """Retry one failed optional service without disturbing healthy services."""

        if kind not in {"live", "reports", "translation"}:
            raise ValueError(f"{kind!r} is not an independently retryable service.")
        if self.coordinator.snapshot.operation.name:
            raise RuntimeError("Another launcher operation is already running.")
        self.validate_profile_updates(self.profile.as_dict())
        plan = build_launch_plan(self.profile)
        command = {
            "live": plan.live,
            "reports": plan.reports,
            "translation": plan.translation,
        }[kind]
        if not command:
            raise RuntimeError(f"{kind.replace('_', ' ').title()} is disabled in the launch profile.")
        port = {
            "live": self.profile.port,
            "reports": self.profile.reports_port,
            "translation": self.profile.translation_port,
        }[kind]
        label = {
            "live": "Live window",
            "reports": "Meeting Intelligence",
            "translation": "Translation",
        }[kind]
        self.coordinator.start_operation("retry", f"Retrying {label}", "Checking service port")
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        try:
            self._guard_external_port(kind, port)
            process = self._spawn_detached(kind, command)
        except Exception as exc:
            self.servers.fail_start(kind)
            self.coordinator.finish_operation("error", f"{label} retry failed", str(exc))
            self._append_log(f"{label} retry failed: {type(exc).__name__}: {exc}")
            self._emit(EventKind.ERROR, str(exc), exc)
            self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
            self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
            raise
        self.coordinator.finish_operation(
            "success",
            f"{label} is starting",
            "Readiness will update when the service port begins accepting connections.",
        )
        self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return process

    def refresh_services(self, *, force: bool = False) -> tuple[ServerState, ...]:
        now = self._clock()
        if not force and now - self._last_probe_at < 0.75:
            return tuple(self.servers.state(kind) for kind in self.SERVICE_KINDS)
        self._last_probe_at = now
        ports = {
            "live": self.profile.port,
            "reports": self.profile.reports_port,
            "translation": self.profile.translation_port,
        }
        for kind, port in ports.items():
            if kind == "translation" and not (
                self.profile.translation_enabled and self.profile.translation_provider == "sidecar"
            ) and not self.servers.process_is_running(kind):
                listening = False
            else:
                listening = self._service_ready(kind, port)
            transition = self.servers.observe(kind, listening=listening, probe_due=True)
            if transition.previous != transition.current or transition.exit_code is not None:
                self._emit(EventKind.SERVICE, kind, transition)
                if transition.exit_code is not None:
                    self._append_log(f"{kind} exited with code {transition.exit_code}")
        self._refresh_remote_backends()
        operation = self.coordinator.snapshot.operation
        if operation.name == "launch":
            relevant_kinds = ["live"]
            if self.profile.mode == "remote":
                relevant_kinds = ["macos_asr", "macos_embeddings", *relevant_kinds]
            elif self.profile.mode == "local":
                relevant_kinds.extend(("macos_asr", "macos_embeddings"))
            if self.profile.reports_enabled:
                relevant_kinds.append("reports")
            if self.profile.translation_enabled and self.profile.translation_provider == "sidecar":
                relevant_kinds.append("translation")
            relevant = [self.servers.state(kind) for kind in relevant_kinds]
            if relevant and not any(state.status == "starting" for state in relevant):
                ready_count = sum(state.status == "running" for state in relevant)
                failed_count = len(relevant) - ready_count
                if failed_count == 0:
                    self.coordinator.finish_operation(
                        "success",
                        "WhoSpeaks is ready",
                        "All selected services passed their application health checks.",
                    )
                elif ready_count:
                    self.coordinator.finish_operation(
                        "warning",
                        "WhoSpeaks started with an issue",
                        f"{ready_count} of {len(relevant)} selected services became ready.",
                    )
                else:
                    self.coordinator.finish_operation(
                        "error",
                        "WhoSpeaks did not start",
                        "No selected service became application-ready.",
                    )
                self._emit(EventKind.OPERATION, payload=self.coordinator.snapshot.operation)
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)
        return tuple(self.servers.state(kind) for kind in self.SERVICE_KINDS)

    def stop_owned_services(self) -> None:
        owned: list[object] = []
        owned_kinds: list[str] = []
        for kind in ("live", "reports", "translation", "macos_embeddings", "macos_asr"):
            if self.servers.state(kind).ownership != "app":
                continue
            process = self.servers.process(kind)
            if process is not None and self.servers.return_code(process) is None:
                owned.append(process)
                owned_kinds.append(kind)
        terminate_service_processes(owned)
        for kind in owned_kinds:
            self.servers.clear(kind)
            self._emit(EventKind.SERVICE, kind, self.servers.state(kind))
        if self.profile.mode == "local":
            self._refresh_remote_backends()
        self._append_log("Stopped services started by this launcher.")
        self._emit(EventKind.SNAPSHOT, payload=self.snapshot)

    def shutdown(self) -> None:
        self.cancel_operation()
        self.stop_owned_services()


def create_launcher_controller(profile: Profile | None = None) -> LauncherController:
    """Shared factory used by desktop and command entry points."""

    return LauncherController(profile)
