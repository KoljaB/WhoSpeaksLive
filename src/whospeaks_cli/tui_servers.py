"""Server lifecycle mixin for the WhoSpeaks setup TUI."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Any

from textual.widgets import Static

from . import main as backend
from .tui_state import PendingAction


class ServerLifecycleMixin:
    @staticmethod
    def _process_return_code(process: object | None) -> int | None:
        if process is None:
            return None
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return None
        return poll()

    def _process_is_running(self, process: object | None) -> bool:
        return process is not None and self._process_return_code(process) is None

    def _render_server_state(self, selector: str, label: str, state: str) -> None:
        matches = list(self.query(selector))
        if not matches:
            return
        widget = matches[0]
        for state_class in ("running", "starting", "failed"):
            widget.remove_class(state_class)
        if state in {"running", "starting", "failed"}:
            widget.add_class(state)
        widget.update(f"{label}: {state}")

    def _render_server_states(self) -> None:
        self._render_server_state(
            "#live-server-state", "Live", self._servers.state("live").display_status
        )
        self._render_server_state(
            "#reports-server-state", "Meeting Intelligence", self._servers.state("reports").display_status
        )
        self._render_server_state(
            "#translation-server-state", "Translation", self._servers.state("translation").display_status
        )

    def _refresh_server_states(self) -> None:
        if not list(self.query("#live-server-state")):
            return
        changed = False
        now = time.monotonic()
        probe_due = now - self.last_server_probe_at >= 0.75
        listening: dict[str, bool] = {}
        if probe_due:
            self.last_server_probe_at = now
            translation_should_probe = (
                self._process_is_running(self.translation_server_process)
                or (
                    self.profile.translation_enabled
                    and self.profile.translation_provider == "sidecar"
                )
            )
            listening = {
                "live": self._server_port_accepting(self.profile.host, self.profile.port),
                "reports": self._server_port_accepting(self.profile.host, self.profile.reports_port),
                "translation": (
                    translation_should_probe
                    and self._server_port_accepting(self.profile.host, self.profile.translation_port)
                ),
            }
        if self.profile.deployment_target == "macos" and probe_due:
            specs = backend.build_macos_service_specs(self.profile)
            listening["macos_asr"] = self._service_health_ready(specs[0])
            listening["macos_embeddings"] = self._service_health_ready(specs[1])
            for kind, spec in zip(("macos_asr", "macos_embeddings"), specs):
                started_at = self._managed_service_started_at.get(kind)
                if (
                    started_at is not None
                    and not listening[kind]
                    and now - started_at >= spec.readiness_timeout
                ):
                    process = self._servers.process(kind)
                    if process is not None:
                        backend.terminate_service_processes([process])
                    self._servers.fail_start(kind)
                    self._managed_service_started_at.pop(kind, None)
                    self._append_log(f"{spec.name} did not become healthy within {spec.readiness_timeout:g}s.")
                    pending = (
                        PendingAction.START_MACOS_EMBEDDINGS
                        if kind == "macos_asr"
                        else PendingAction.LAUNCH_AFTER_MACOS_SERVICES
                    )
                    self._coordinator.take_pending_action(pending)
                    self._set_feedback(
                        "error",
                        f"{spec.name} failed health checks",
                        f"The browser controller was not started. Check {spec.health_url} and service logs.",
                    )
        transitions = []
        for kind in ("live", "reports", "translation", "macos_asr", "macos_embeddings"):
            transition = self._servers.observe(
                kind,
                listening=bool(listening.get(kind, False)),
                probe_due=probe_due,
            )
            transitions.append(transition)
            if transition.exit_code is not None:
                self._append_log(
                    f"{self._server_label(kind)} exited with code {transition.exit_code}."
                )
            if transition.previous != transition.current:
                changed = True
        if changed:
            self._render_server_states()
            self._sync_action_buttons()
        translation = next(item for item in transitions if item.kind == "translation")
        macos_asr = next(item for item in transitions if item.kind == "macos_asr")
        macos_embeddings = next(item for item in transitions if item.kind == "macos_embeddings")
        if translation.became_app_ready:
            self._set_feedback(
                "success",
                "Translation server ready",
                f"Translation API on http://{self.profile.host}:{self.profile.translation_port}/",
            )
        if translation.became_app_ready and self._coordinator.take_pending_action(
            PendingAction.LAUNCH_LIVE_AFTER_TRANSLATION
        ):
            self._start_live_server()
        elif translation.app_failed and self._coordinator.take_pending_action(
            PendingAction.LAUNCH_LIVE_AFTER_TRANSLATION
        ):
            self._set_feedback(
                "error",
                "Translation warm-up failed",
                "The live server was not started. Check the translation server window for the model-loading error.",
            )
        if macos_asr.became_app_ready and self._coordinator.take_pending_action(
            PendingAction.START_MACOS_EMBEDDINGS
        ):
            self._managed_service_started_at.pop("macos_asr", None)
            self._start_macos_service("macos_embeddings")
        elif macos_asr.app_failed and self._coordinator.take_pending_action(
            PendingAction.START_MACOS_EMBEDDINGS
        ):
            self._set_feedback("error", "MLX ASR failed", "The browser controller was not started.")
        if macos_embeddings.became_app_ready and self._coordinator.take_pending_action(
            PendingAction.LAUNCH_AFTER_MACOS_SERVICES
        ):
            self._managed_service_started_at.pop("macos_embeddings", None)
            self._start_configured_services_and_live()
        elif macos_embeddings.app_failed and self._coordinator.take_pending_action(
            PendingAction.LAUNCH_AFTER_MACOS_SERVICES
        ):
            self._set_feedback("error", "Embeddings service failed", "The browser controller was not started.")

    @staticmethod
    def _server_port_accepting(host: str, port: int) -> bool:
        probe_host = str(host or "127.0.0.1").strip()
        if probe_host in {"0.0.0.0", "::", "[::]"}:
            probe_host = "127.0.0.1"
        try:
            with socket.create_connection((probe_host, int(port)), timeout=0.08):
                return True
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _new_server_console_kwargs() -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_CONSOLE}
        return {"start_new_session": True}

    @staticmethod
    def _service_health_ready(spec: backend.ServiceProcessSpec) -> bool:
        return backend.service_health_ready(spec)

    def _start_server_process(
        self,
        kind: str,
        command: list[str],
        *,
        cwd: str | None = None,
        env_additions: dict[str, str] | None = None,
    ) -> bool:
        server_state = self._servers.state(kind)
        if self._servers.process_is_running(kind) or server_state.status == "running":
            label = self._server_label(kind)
            detail = "is already running" if server_state.ownership == "app" else "port is used by another process"
            self.notify(f"{label} {detail}", severity="warning")
            return False
        label = self._server_label(kind)
        try:
            kwargs = self._new_server_console_kwargs()
            if cwd:
                kwargs["cwd"] = cwd
            if env_additions:
                env = dict(os.environ)
                env.update(env_additions)
                kwargs["env"] = env
            process = self.popen_factory(command, **kwargs)
        except OSError as exc:
            self._servers.fail_start(kind)
            self._render_server_states()
            self._append_log(f"Could not start {label.lower()}: {exc}")
            self._set_feedback("error", f"{label} failed to start", str(exc))
            return False
        self._servers.begin(kind, process)
        self._render_server_states()
        self._sync_action_buttons()
        self._append_log(f"Started {label.lower()}: {backend.format_command(command)}")
        return True

    def _start_macos_service(self, kind: str) -> bool:
        index = 0 if kind == "macos_asr" else 1
        spec = backend.build_macos_service_specs(self.profile)[index]
        started = self._start_server_process(
            kind,
            list(spec.command),
            cwd=spec.cwd,
            env_additions=dict(spec.env),
        )
        if started:
            self._managed_service_started_at[kind] = time.monotonic()
            if kind == "macos_embeddings":
                self._coordinator.set_pending_action(PendingAction.LAUNCH_AFTER_MACOS_SERVICES)
        else:
            self._coordinator.clear_pending_action()
        return started

    @staticmethod
    def _server_label(kind: str) -> str:
        return {
            "live": "Live server",
            "reports": "Meeting Intelligence",
            "translation": "Translation server",
            "macos_asr": "MLX ASR service",
            "macos_embeddings": "MPS embeddings service",
        }.get(kind, f"{kind.title()} server")
