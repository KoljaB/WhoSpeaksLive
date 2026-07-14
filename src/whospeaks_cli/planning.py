"""Pure install and launch planning for WhoSpeaks front ends."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import platform
import shlex
import shutil
import sys
from pathlib import Path
from importlib import resources
from typing import Any

from window.language_config import normalize_language_code
from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
)

from .profiles import (
    DEFAULT_MACOS_ASR_URL,
    DEFAULT_REMOTE_ASR_URL,
    DEFAULT_REMOTE_EMBEDDINGS_URL,
    Profile,
    normalize_mode,
    profile_with_provider_preset,
)


COMPLETE_EXTRA = "complete"
LOCAL_EXTRA = "complete,preview"
CONTROLLER_EXTRA = "controller"
PREVIEW_EXTRA = "preview"
SERVER_EXTRA = "server"
INSTALL_TARGET_CHOICES = ("local", "macos", "core", "server")
TRANSLATION_INSTALL_PROFILE_CHOICES = ("off", "nllb-200-600m", "translate-gemma-4b", "madlad-400-3b")


@dataclasses.dataclass(frozen=True)
class InstallPlan:
    target: str
    title: str
    mode: str
    extra: str
    install_kroko: bool
    summary: str
    realtime_preview_engine: str = "off"
    realtime_preview_model_preset: str = ""
    translation_model_profile: str = "off"


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    """Detached commands required to launch the current profile."""

    live: tuple[str, ...]
    reports: tuple[str, ...] | None = None
    translation: tuple[str, ...] | None = None
    services: tuple[ServiceProcessSpec, ...] = ()


@dataclasses.dataclass(frozen=True)
class ServiceProcessSpec:
    name: str
    command: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    health_url: str
    readiness_timeout: float
    expected_health: tuple[tuple[str, str], ...] = ()


def require_apple_silicon_macos() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise SystemExit(
            "The macos target requires an Apple Silicon Mac (Darwin arm64/aarch64). "
            "Intel Macs are not supported; use the core target with external services instead."
        )


def default_macos_runtime_root() -> Path:
    override = os.environ.get("WHOSPEAKS_MACOS_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "WhoSpeaks" / "macos"
    root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return root / "whospeaks" / "macos"


def service_resource_path(*parts: str) -> Path:
    relative = Path("remote_servers", *parts)
    try:
        resource = Path(os.fspath(resources.files("remote_servers").joinpath(*parts)))
        if resource.is_file():
            return resource
    except (ModuleNotFoundError, NotADirectoryError, TypeError):
        pass
    try:
        import remote_servers

        for package_root in remote_servers.__path__:
            resource = Path(package_root, *parts)
            if resource.is_file():
                return resource
    except (ImportError, TypeError):
        pass
    distribution = importlib.metadata.distribution("whospeaks")
    for entry in distribution.files or ():
        if Path(str(entry)).as_posix().endswith(relative.as_posix()):
            resource = Path(distribution.locate_file(entry))
            if resource.is_file():
                return resource
    raise FileNotFoundError(f"Packaged service resource is missing: {relative}")


def health_payload_matches(spec: ServiceProcessSpec, payload: dict[str, Any] | None) -> bool:
    if not payload or payload.get("ok") is False:
        return False
    return all(payload.get(key) == value for key, value in spec.expected_health)


def normalize_install_target(value: str | None) -> str:
    normalized = str(value or "local").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = {
        "all": "local",
        "full": "local",
        "full_local": "local",
        "local_all_in_one": "local",
        "controller": "core",
        "controller_remote": "core",
        "remote": "core",
        "browser": "core",
        "core_controller": "core",
        "gpu": "server",
        "gpu_server": "server",
        "services": "server",
        "asr_embeddings_server": "server",
        "asr_server": "server",
        "embeddings_server": "server",
    }.get(normalized, normalized)
    if normalized not in INSTALL_TARGET_CHOICES:
        raise SystemExit(
            "Unknown install target {0!r}. Choose one of: {1}.".format(
                value,
                ", ".join(INSTALL_TARGET_CHOICES),
            )
        )
    return normalized


def install_plan_for_target(
    target: str,
    install_kroko: bool = False,
    *,
    realtime_preview_engine: str | None = None,
    realtime_preview_model_preset: str | None = None,
    translation_model_profile: str = "off",
) -> InstallPlan:
    selected = normalize_install_target(target)
    if selected == "macos":
        require_apple_silicon_macos()
    translation_profile = str(translation_model_profile or "off").strip().lower()
    if translation_profile not in TRANSLATION_INSTALL_PROFILE_CHOICES:
        raise SystemExit(
            f"Unknown translation model profile {translation_model_profile!r}. Choose one of: "
            f"{', '.join(TRANSLATION_INSTALL_PROFILE_CHOICES)}."
        )
    if selected == "server":
        engine = "off"
    elif realtime_preview_engine is None:
        engine = "kroko_onnx" if install_kroko else "off"
    else:
        try:
            engine = normalize_preview_engine(realtime_preview_engine)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if engine in {"kroko_onnx", "sherpa_onnx"}:
        default_preset = get_preview_backend_spec(engine).default_preset or ""
        try:
            preset = normalize_preview_model_preset(engine, realtime_preview_model_preset or default_preset)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        preset = ""
    kroko_selected = engine == "kroko_onnx"
    preview_selected = engine in {"kroko_onnx", "sherpa_onnx"}
    if selected == "local":
        return InstallPlan(
            target=selected,
            title="Full local installation",
            mode="local",
            extra=LOCAL_EXTRA if kroko_selected else COMPLETE_EXTRA,
            install_kroko=kroko_selected,
            summary="Browser controller, local final ASR, and local speaker embeddings on this machine.",
            realtime_preview_engine=engine,
            realtime_preview_model_preset=preset,
            translation_model_profile=translation_profile,
        )
    if selected in {"core", "macos"}:
        return InstallPlan(
            target=selected,
            title=(
                "Apple Silicon managed local services"
                if selected == "macos"
                else "Core/controller for remote ASR and embeddings servers"
            ),
            mode="remote",
            extra=f"{CONTROLLER_EXTRA},{PREVIEW_EXTRA}" if preview_selected else CONTROLLER_EXTRA,
            install_kroko=kroko_selected,
            summary=(
                "Browser controller with managed localhost MLX ASR and MPS embeddings on this Apple Silicon Mac."
                if selected == "macos"
                else "Browser controller on this machine with final ASR and embeddings served over HTTP."
            ),
            realtime_preview_engine=engine,
            realtime_preview_model_preset=preset,
            translation_model_profile=translation_profile,
        )
    return InstallPlan(
        target=selected,
        title="ASR and embeddings server packages",
        mode="server",
        extra=SERVER_EXTRA,
        install_kroko=False,
        summary="Service-side dependencies for the remote faster-whisper ASR and embeddings endpoints.",
        realtime_preview_engine="off",
        translation_model_profile=translation_profile,
    )


def profile_for_mode(profile: Profile, mode: str) -> Profile:
    """Return a configured profile without mutating the input snapshot."""

    selected = normalize_mode(mode)
    deployment_updates: dict[str, object] = {"deployment_target": ""}
    if profile.deployment_target == "macos":
        deployment_updates.update(
            remote_asr_url=DEFAULT_REMOTE_ASR_URL,
            remote_embeddings_url=DEFAULT_REMOTE_EMBEDDINGS_URL,
        )
    if selected == "local":
        base = profile_with_provider_preset(profile, "smoke")
        return base.with_updates(
            mode="local",
            asr_backend="local",
            embeddings_backend="local",
            device="auto",
            vad_backend="rms",
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_model_preset="nemotron-3.5-560ms-int8",
            realtime_preview_model_dir="",
            **deployment_updates,
        )
    if selected == "remote":
        base = profile_with_provider_preset(profile, "smoke")
        return base.with_updates(
            mode="remote",
            asr_backend="remote",
            embeddings_backend="remote",
            device="auto",
            vad_backend="rms",
            realtime_preview_engine="off",
            realtime_preview_model_preset="",
            realtime_preview_model_dir="",
            **deployment_updates,
        )
    return profile.with_updates(
        mode="server",
        asr_backend="remote",
        embeddings_backend="remote",
        realtime_preview_engine="off",
        realtime_preview_model_preset="",
        realtime_preview_model_dir="",
        **deployment_updates,
    )


def profile_for_install(profile: Profile, plan: InstallPlan) -> Profile:
    """Return the complete persisted profile implied by an install plan."""

    configured = profile_for_mode(profile, plan.mode)
    updates: dict[str, object] = {
        "translation_enabled": plan.translation_model_profile != "off",
        "deployment_target": "macos" if plan.target == "macos" else "",
    }
    if plan.target == "macos":
        updates.update(
            remote_asr_url=DEFAULT_MACOS_ASR_URL,
            remote_embeddings_url=DEFAULT_REMOTE_EMBEDDINGS_URL,
        )
    if plan.target in {"local", "macos", "core"}:
        updates.update(
            realtime_preview_engine=plan.realtime_preview_engine,
            realtime_preview_model_preset=plan.realtime_preview_model_preset,
        )
        if plan.realtime_preview_engine != "sherpa_onnx":
            updates["realtime_preview_model_dir"] = ""
    else:
        updates.update(
            realtime_preview_engine="off",
            realtime_preview_model_preset="",
            realtime_preview_model_dir="",
        )
    if plan.translation_model_profile != "off":
        updates.update(
            translation_provider="sidecar",
            translation_model_profile=plan.translation_model_profile,
        )
    return configured.with_updates(**updates)


def build_launch_command(profile: Profile, extra_args: str = "") -> list[str]:
    executable = shutil.which("whospeaks-window")
    command = [executable] if executable else [sys.executable, "-m", "window.youtube_window_diarize_gui"]
    command.extend([
        "--host", str(profile.host),
        "--port", str(int(profile.port)),
        "--language", str(profile.language),
        "--model", str(profile.model),
        "--device", str(profile.device),
        "--compute-type", str(profile.compute_type),
        "--asr-backend", str(profile.asr_backend),
        "--embeddings-backend", str(profile.embeddings_backend),
        "--embedding-provider", str(profile.embedding_provider),
        "--live-speaker-embedding-provider", str(profile.live_speaker_embedding_provider),
        "--vad-backend", str(profile.vad_backend),
        "--realtime-preview-engine", str(profile.realtime_preview_engine or "off"),
    ])
    command.append("--live-speaker-assignment" if profile.live_speaker_assignment else "--no-live-speaker-assignment")
    if profile.reports_enabled:
        connect_host = "127.0.0.1" if str(profile.host) in {"0.0.0.0", "::", "[::]"} else str(profile.host)
        command.extend(["--meeting-intelligence-url", f"http://{connect_host}:{int(profile.reports_port)}"])
    if profile.embeddings_backend == "local":
        command.extend(["--embedding-python", str(profile.embedding_python or sys.executable)])
    preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
    if preview_engine in {"kroko_onnx", "sherpa_onnx"} and profile.realtime_preview_model_preset:
        command.extend(["--realtime-preview-model-preset", str(profile.realtime_preview_model_preset)])
    if preview_engine == "sherpa_onnx" and profile.realtime_preview_model_dir:
        command.extend(["--realtime-preview-model-dir", str(profile.realtime_preview_model_dir)])
    if preview_engine == "sherpa_onnx":
        command.extend(["--realtime-preview-python", sys.executable])
    elif profile.realtime_preview_python or preview_engine not in {"off", "none", "false", "0"}:
        command.extend(["--realtime-preview-python", str(profile.realtime_preview_python or sys.executable)])
    if profile.asr_backend == "remote":
        command.extend(["--remote-asr-url", str(profile.remote_asr_url)])
    if profile.embeddings_backend == "remote":
        command.extend(["--remote-embeddings-url", str(profile.remote_embeddings_url)])
    translation_provider = profile.translation_provider if profile.translation_enabled else "off"
    translation_base_url = str(profile.translation_base_url or "").strip()
    translation_model = str(profile.translation_model or "").strip()
    translation_api_key_env = str(profile.translation_api_key_env or "").strip()
    if translation_provider == "sidecar":
        translation_base_url = translation_base_url or f"http://127.0.0.1:{int(profile.translation_port)}"
    elif translation_provider == "reports_llm":
        defaults = {
            "llama_cpp": ("http://127.0.0.1:8081/v1", "local", ""),
            "ollama": ("http://127.0.0.1:11434/v1", "gemma3", ""),
            "lm_studio": ("http://127.0.0.1:1234/v1", "local-model", ""),
            "openai_compatible": ("http://127.0.0.1:8000/v1", "local-model", ""),
            "openai": ("https://api.openai.com/v1", "", "OPENAI_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1", "", "OPENROUTER_API_KEY"),
        }
        default_url, default_model, translation_api_key_env = defaults[profile.report_llm_provider]
        translation_provider = "openai_compatible"
        translation_base_url = translation_base_url or profile.report_llm_base_url or default_url
        translation_model = translation_model or profile.report_llm_model or default_model
    else:
        translation_api_key_env = translation_api_key_env or {
            "openai_compatible": "OPENAI_API_KEY",
            "deepl": "DEEPL_API_KEY",
            "google_cloud": "GOOGLE_TRANSLATE_API_KEY",
            "azure_translator": "AZURE_TRANSLATOR_KEY",
            "libretranslate": "LIBRETRANSLATE_API_KEY",
        }.get(translation_provider, "")
    command.extend([
        "--translation-provider", translation_provider,
        "--translation-max-targets", str(int(profile.translation_max_targets)),
        "--translation-model-profile", str(profile.translation_model_profile),
        "--translation-device", str(profile.translation_device),
    ])
    if profile.translation_browser_preferred:
        command.append("--translation-browser-preferred")
    if translation_base_url:
        command.extend(["--translation-base-url", translation_base_url])
    if translation_model:
        command.extend(["--translation-model", translation_model])
    if translation_api_key_env:
        command.extend(["--translation-api-key-env", translation_api_key_env])
    if profile.translation_region:
        command.extend(["--translation-region", str(profile.translation_region)])
    for target in str(profile.translation_target_languages or "").split(","):
        if target:
            command.extend(["--translation-target-language", target])
    advanced = " ".join(item for item in [profile.advanced_args, extra_args] if item)
    if advanced:
        command.extend(shlex.split(advanced))
    return command


def build_reports_command(
    profile: Profile,
    *,
    port: int = 8798,
    report_language: str = "",
    llm_provider: str = "llama_cpp",
    llm_base_url: str = "",
    llm_model: str = "",
    auto_generate: bool = True,
) -> list[str]:
    executable = shutil.which("whospeaks-meeting-intelligence")
    command = [executable] if executable else [sys.executable, "-m", "window.meeting_intelligence_server"]
    command.extend([
        "--host", str(profile.host),
        "--port", str(int(port)),
        "--report-language", normalize_language_code(report_language or profile.language),
        "--llm-provider", str(llm_provider),
    ])
    if llm_base_url:
        command.extend(["--llm-base-url", str(llm_base_url)])
    if llm_model:
        command.extend(["--llm-model", str(llm_model)])
    if profile.text_embedding_base_url:
        command.extend(["--text-embedding-base-url", str(profile.text_embedding_base_url)])
    if profile.text_embedding_model:
        command.extend(["--text-embedding-model", str(profile.text_embedding_model)])
    if profile.text_embedding_api_key_env:
        command.extend(["--text-embedding-api-key-env", str(profile.text_embedding_api_key_env)])
    if auto_generate:
        command.append("--auto-generate")
    return command


def build_translation_command(profile: Profile) -> list[str]:
    executable = shutil.which("whospeaks-translation-server") if not profile.translation_python else None
    command = [executable] if executable else [str(profile.translation_python or sys.executable), "-m", "window.translation_server"]
    command.extend([
        "--host", str(profile.host),
        "--port", str(int(profile.translation_port)),
        "--model-profile", str(profile.translation_model_profile),
        "--device", str(profile.translation_device),
    ])
    if profile.translation_model:
        command.extend(["--model", str(profile.translation_model)])
    return command


def build_launch_plan(profile: Profile, extra_args: str = "") -> LaunchPlan:
    """Capture all launch commands from one immutable profile snapshot."""

    reports = None
    if profile.reports_enabled:
        reports = tuple(build_reports_command(
            profile,
            port=profile.reports_port,
            report_language=profile.report_language,
            llm_provider=profile.report_llm_provider,
            llm_base_url=profile.report_llm_base_url,
            llm_model=profile.report_llm_model,
            auto_generate=profile.report_auto_generate,
        ))
    translation = None
    if profile.translation_enabled and profile.translation_provider == "sidecar":
        translation = tuple(build_translation_command(profile))
    return LaunchPlan(
        live=tuple(build_launch_command(profile, extra_args)),
        reports=reports,
        translation=translation,
        services=build_macos_service_specs(profile),
    )


def build_macos_service_specs(
    profile: Profile,
    runtime_root: Path | None = None,
) -> tuple[ServiceProcessSpec, ...]:
    if profile.deployment_target != "macos":
        return ()
    root = (runtime_root or default_macos_runtime_root()).expanduser().resolve()
    asr_python = root / "mlx-asr" / "bin" / "python"
    embeddings_python = root / "embeddings" / "bin" / "python"
    return (
        ServiceProcessSpec(
            name="MLX ASR",
            command=(str(asr_python), "-m", "remote_servers.launcher", "mlx-asr"),
            cwd=str(root),
            env=(("ASR_HOST", "127.0.0.1"), ("ASR_PORT", "8651")),
            health_url=f"{profile.remote_asr_url.rstrip('/')}/health",
            readiness_timeout=300.0,
            expected_health=(("service", "mlx-whisper-asr"),),
        ),
        ServiceProcessSpec(
            name="MPS embeddings",
            command=(str(embeddings_python), "-m", "remote_servers.launcher", "embeddings"),
            cwd=str(root),
            env=(
                ("EMBEDDINGS_HOST", "127.0.0.1"),
                ("EMBEDDINGS_PORT", "8660"),
                ("EMBEDDINGS_DEVICE", "auto"),
                ("PYTORCH_ENABLE_MPS_FALLBACK", "1"),
            ),
            health_url=f"{profile.remote_embeddings_url.rstrip('/')}/health",
            readiness_timeout=180.0,
            expected_health=(("service", "voice-embeddings-server"),),
        ),
    )
