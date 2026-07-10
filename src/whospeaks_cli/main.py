"""Lightweight setup, doctor, and launcher CLI for WhoSpeaks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

from . import __version__
from window.language_config import SUPPORTED_LANGUAGE_CONFIGS, get_language_config, normalize_language_code
from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
)
from window.sherpa_onnx_models import (
    default_sherpa_onnx_model_dir,
    missing_sherpa_onnx_model_files,
)


DEFAULT_REMOTE_ASR_URL = "http://127.0.0.1:8650"
DEFAULT_REMOTE_EMBEDDINGS_URL = "http://127.0.0.1:8660"
SMOKE_PROVIDER = "speechbrain_ecapa"
SINGLE_ESPNET_PROVIDER = "espnet_ecapa_wavlm_joint"
PUBLIC_PROVIDER = "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+speechbrain_resnet=0.38+resemblyzer=0.12"
PROMOTED_PUBLIC_PROVIDER = "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
FAST_LIVE_PROVIDER = "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"
COMPLETE_EXTRA = "complete"
LOCAL_EXTRA = "complete,preview"
CONTROLLER_EXTRA = "controller"
PREVIEW_EXTRA = "preview"
SERVER_EXTRA = "server"
PACKAGE_NAME = "whospeaks"
KROKO_INSTALL_MODULE = "RealtimeSTT.install_kroko"
KROKO_PREVIEW_VENV_ENV = "WHOSPEAKS_KROKO_PREVIEW_VENV"
TESTPYPI_SIMPLE_URL = "https://test.pypi.org/simple/"
PIP_INDEX_URL_ENV = "WHOSPEAKS_PIP_INDEX_URL"
PIP_EXTRA_INDEX_URL_ENV = "WHOSPEAKS_PIP_EXTRA_INDEX_URL"
PIP_FIND_LINKS_ENV = "WHOSPEAKS_PIP_FIND_LINKS"
TORCH_INSTALL_POLICY_ENV = "WHOSPEAKS_TORCH_INSTALL"
PYTORCH_CUDA_BUILD_ENV = "WHOSPEAKS_PYTORCH_CUDA_BUILD"
PYTORCH_CUDA_INDEX_URL_ENV = "WHOSPEAKS_PYTORCH_CUDA_INDEX_URL"
PYTORCH_CPU_INDEX_URL_ENV = "WHOSPEAKS_PYTORCH_CPU_INDEX_URL"
DEFAULT_PYTORCH_CUDA_BUILD = "cu128"
PYTORCH_CUDA_INDEX_URLS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGE_SPECS = ("torch>=2.2", "torchaudio>=2.2")
KROKO_LANGUAGE_MENU_CODES = ("en", "de", "es", "fr", "it", "nl", "pt", "sv", "tr", "he")
INSTALL_TARGET_CHOICES = ("local", "core", "server")
TORCH_INSTALL_POLICY_CHOICES = ("auto", "cuda", "cpu", "skip")
EDITABLE_PROFILE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("mode", "Profile mode", "local, remote, or server. Mode also aligns the ASR and embeddings backends."),
    ("language", "Language", "Shared by final ASR, realtime preview model selection, and sentence splitting."),
    ("provider_preset", "Provider preset", "Named final/live speaker embedding stack, or custom."),
    ("embedding_provider", "Final provider", "Exact provider string used for committed speaker assignment."),
    ("live_speaker_embedding_provider", "Live provider", "Exact provider string used for live speaker feedback."),
    ("embedding_python", "Embedding helper Python", "Optional Python executable for local speaker-embedding helper subprocesses."),
    ("realtime_preview_engine", "Realtime text engine", "Use sherpa_onnx for Nemotron, kroko_onnx for Kroko/Banafo, or off."),
    ("realtime_preview_model_preset", "Realtime model preset", "Nemotron: 560ms stable or 160ms low-latency. Kroko: a Kroko model preset."),
    ("realtime_preview_model_dir", "Nemotron model folder", "Optional explicit folder for the unpacked sherpa-onnx/Nemotron model."),
    ("realtime_preview_python", "Realtime preview Python", "Optional Python executable for the realtime worker. Nemotron uses the current environment by default."),
    ("asr_backend", "ASR backend", "local or remote."),
    ("embeddings_backend", "Embeddings backend", "local or remote."),
    ("remote_asr_url", "Remote ASR URL", "Base URL for a remote faster-whisper service."),
    ("remote_embeddings_url", "Remote embeddings URL", "Base URL for a remote voice embeddings service."),
    ("model", "ASR model", "Final ASR model name, for example large-v2."),
    ("device", "Device", "auto, cuda, or cpu."),
    ("compute_type", "Compute type", "faster-whisper compute type, for example float16 or int8."),
    ("vad_backend", "VAD backend", "Voice activity detector used for sentence-window finalization."),
    ("host", "Browser host", "Interface for the browser UI server."),
    ("port", "Browser port", "Port for the browser UI server."),
    ("advanced_args", "Advanced args", "Extra whospeaks-window flags appended after the saved profile."),
)


STATUS_ORDER = {"ok": 0, "skip": 1, "warn": 2, "fail": 3}
STATUS_LABEL = {
    "ok": "OK",
    "skip": "SKIP",
    "warn": "WARN",
    "fail": "FAIL",
}


@dataclasses.dataclass(frozen=True)
class ProviderPreset:
    id: str
    name: str
    summary: str
    details: str
    embedding_provider: str
    live_speaker_embedding_provider: str
    requirements: str = ""
    score_note: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "smoke": ProviderPreset(
        id="smoke",
        name="First start",
        summary="Simple setup check. Fastest to try, not the highest-accuracy setting.",
        details="Uses the SpeechBrain ECAPA provider for both final sentence assignment and live speaker feedback.",
        embedding_provider=SMOKE_PROVIDER,
        live_speaker_embedding_provider=SMOKE_PROVIDER,
        score_note="Baseline smoke setting. Use it to prove installation and media flow before comparing quality.",
    ),
    "single_espnet": ProviderPreset(
        id="single_espnet",
        name="Single ESPnet",
        summary="One ESPnet embedding provider for both final and live speaker assignment.",
        details=(
            "Useful when validating one provider in isolation. It does not use the weighted multi-provider stacks."
        ),
        embedding_provider=SINGLE_ESPNET_PROVIDER,
        live_speaker_embedding_provider=SINGLE_ESPNET_PROVIDER,
        score_note="Single-provider option. Keep score claims separate from the mixed-provider stacks.",
    ),
    "smoke_fast_live": ProviderPreset(
        id="smoke_fast_live",
        name="Smoke final + fast live",
        summary="Keeps the simple final provider and uses the faster live speaker stack.",
        details=(
            "Final assignment stays on SpeechBrain ECAPA. Live feedback uses the pyannote/wespeaker ONNX stack "
            "recommended for responsive live speaker tags."
        ),
        embedding_provider=SMOKE_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Useful when final quality is not the test target and live feedback latency is.",
    ),
    "public_quality": ProviderPreset(
        id="public_quality",
        name="Public high quality",
        summary="Public multi-provider final stack plus fast live speaker feedback.",
        details=(
            "Uses the documented public stack with ESPnet, WeSpeaker CAM++, SpeechBrain ResNet, and Resemblyzer. "
            "All providers are available through the public setup path."
        ),
        embedding_provider=PUBLIC_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Public quality candidate for reproducible comparisons.",
    ),
    "promoted_public": ProviderPreset(
        id="promoted_public",
        name="Promoted public stack",
        summary="Current promoted public final stack plus fast live speaker feedback.",
        details=(
            "Matches the current whospeaks-window default final provider stack. Keep it separate from "
            "public_quality until validation confirms which public stack scores higher for the target data."
        ),
        embedding_provider=PROMOTED_PUBLIC_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Current promoted public default. Keep this and public_quality visible until validation decides the winner.",
    ),
}
PROVIDER_PRESET_CHOICES = tuple(PROVIDER_PRESETS.keys()) + ("custom",)


@dataclasses.dataclass
class Profile:
    mode: str = "local"
    host: str = "127.0.0.1"
    port: int = 8796
    language: str = "en"
    model: str = "large-v2"
    device: str = "auto"
    compute_type: str = "float16"
    asr_backend: str = "local"
    embeddings_backend: str = "local"
    provider_preset: str = "smoke"
    remote_asr_url: str = DEFAULT_REMOTE_ASR_URL
    remote_embeddings_url: str = DEFAULT_REMOTE_EMBEDDINGS_URL
    embedding_provider: str = SMOKE_PROVIDER
    live_speaker_embedding_provider: str = SMOKE_PROVIDER
    embedding_python: str = ""
    vad_backend: str = "rms"
    realtime_preview_engine: str = "sherpa_onnx"
    realtime_preview_model_preset: str = "nemotron-3.5-560ms-int8"
    realtime_preview_model_dir: str = ""
    realtime_preview_python: str = ""
    advanced_args: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Profile":
        allowed = {field.name for field in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {key: item for key, item in value.items() if key in allowed}
        profile = cls(**kwargs)
        profile.port = int(profile.port)
        profile.mode = normalize_mode(profile.mode)
        try:
            profile.language = normalize_language_code(profile.language)
        except ValueError:
            profile.language = "en"
        profile.provider_preset = infer_provider_preset_id(
            profile.provider_preset,
            profile.embedding_provider,
            profile.live_speaker_embedding_provider,
        )
        if profile.mode == "remote":
            profile.asr_backend = "remote"
            profile.embeddings_backend = "remote"
        elif profile.mode == "local":
            profile.asr_backend = "local"
            profile.embeddings_backend = "local"
        try:
            profile.realtime_preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
        except ValueError:
            profile.realtime_preview_engine = "off"
        if profile.realtime_preview_engine in {"kroko_onnx", "sherpa_onnx"}:
            default_preset = get_preview_backend_spec(profile.realtime_preview_engine).default_preset or ""
            try:
                profile.realtime_preview_model_preset = normalize_preview_model_preset(
                    profile.realtime_preview_engine,
                    profile.realtime_preview_model_preset or default_preset,
                )
            except (ValueError, argparse.ArgumentTypeError):
                profile.realtime_preview_model_preset = default_preset
        else:
            profile.realtime_preview_model_preset = ""
        if profile.realtime_preview_engine != "sherpa_onnx":
            profile.realtime_preview_model_dir = ""
        return profile

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def normalize_provider_preset_id(value: str | None) -> str:
    normalized = str(value or "custom").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "first_start": "smoke",
        "speechbrain": "smoke",
        "smoke_provider": "smoke",
        "espnet": "single_espnet",
        "single_espnet_provider": "single_espnet",
        "fast_live": "smoke_fast_live",
        "public": "public_quality",
        "public_high_quality": "public_quality",
        "high_quality": "public_quality",
        "promoted": "promoted_public",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in PROVIDER_PRESETS or normalized == "custom":
        return normalized
    return "custom"


def infer_provider_preset_id(current: str | None, embedding_provider: str, live_provider: str) -> str:
    normalized = normalize_provider_preset_id(current)
    if normalized in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[normalized]
        if (
            str(embedding_provider) == preset.embedding_provider
            and str(live_provider) == preset.live_speaker_embedding_provider
        ):
            return normalized
    for preset_id, preset in PROVIDER_PRESETS.items():
        if (
            str(embedding_provider) == preset.embedding_provider
            and str(live_provider) == preset.live_speaker_embedding_provider
        ):
            return preset_id
    return "custom"


def apply_provider_preset(profile: Profile, preset_id: str) -> Profile:
    normalized = normalize_provider_preset_id(preset_id)
    if normalized == "custom":
        profile.provider_preset = "custom"
        return profile
    preset = PROVIDER_PRESETS[normalized]
    profile.provider_preset = preset.id
    profile.embedding_provider = preset.embedding_provider
    profile.live_speaker_embedding_provider = preset.live_speaker_embedding_provider
    return profile


def selected_provider_preset(profile: Profile) -> ProviderPreset | None:
    preset_id = infer_provider_preset_id(
        profile.provider_preset,
        profile.embedding_provider,
        profile.live_speaker_embedding_provider,
    )
    profile.provider_preset = preset_id
    return PROVIDER_PRESETS.get(preset_id)


def color_enabled() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ


def style_text(text: str, code: str) -> str:
    if not color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def primary_text(text: str) -> str:
    return style_text(text, "97")


def detail_text(text: str) -> str:
    return style_text(text, "37")


def label_text(text: str) -> str:
    return style_text(text, "96")


def wrap_styled_lines(
    text: str,
    *,
    width: int = 72,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
    style: Any = detail_text,
) -> list[str]:
    follow = initial_indent if subsequent_indent is None else subsequent_indent
    wrapped = textwrap.wrap(
        str(text),
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=follow,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [initial_indent.rstrip()]
    return [style(line) for line in wrapped]


def print_wrapped(
    text: str,
    *,
    width: int = 72,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
    style: Any = detail_text,
) -> None:
    for line in wrap_styled_lines(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        style=style,
    ):
        print(line)


@dataclasses.dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    remediation: str = ""

    def is_problem(self) -> bool:
        return self.status in {"warn", "fail"}


@dataclasses.dataclass
class DoctorReport:
    mode: str
    checks: list[CheckResult]

    @property
    def worst_status(self) -> str:
        if not self.checks:
            return "skip"
        return max(self.checks, key=lambda item: STATUS_ORDER[item.status]).status

    @property
    def has_failures(self) -> bool:
        return any(item.status == "fail" for item in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(item.status == "warn" for item in self.checks)


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


@dataclasses.dataclass(frozen=True)
class TorchInstallSelection:
    mode: str
    index_url: str
    reason: str
    build: str = ""

    @property
    def should_install(self) -> bool:
        return self.mode in {"cuda", "cpu"}


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "local").strip().lower().replace("-", "_")
    aliases = {
        "all_in_one": "local",
        "full_local": "local",
        "controller_remote": "remote",
        "gpu_server": "server",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "local", "remote", "server"}:
        return "local"
    return value


def config_path() -> Path:
    override = os.environ.get("WHOSPEAKS_CONFIG")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return root / "WhoSpeaks" / "config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "whospeaks" / "config.json"


def local_config_path() -> Path:
    return Path.cwd() / ".whospeaks" / "config.json"


def config_read_candidates() -> list[Path]:
    if os.environ.get("WHOSPEAKS_CONFIG"):
        return [config_path()]
    primary = config_path()
    fallback = local_config_path()
    return [primary] if primary == fallback else [primary, fallback]


def load_profile(path: Path | None = None) -> Profile:
    for selected in ([path] if path is not None else config_read_candidates()):
        if selected is None:
            continue
        try:
            data = json.loads(selected.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if isinstance(data, dict):
            return Profile.from_mapping(data)
    return Profile()


def save_profile(profile: Profile, path: Path | None = None) -> Path:
    selected = path or config_path()
    try:
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return selected
    except OSError:
        if path is not None or os.environ.get("WHOSPEAKS_CONFIG"):
            raise
    fallback = local_config_path()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fallback


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def subprocess_pythonpath_entries() -> list[str]:
    entries: list[str] = []
    module_root = Path(__file__).resolve().parents[1]
    for candidate in (module_root, module_root / "vendor", Path.cwd() / "src", Path.cwd() / "vendor"):
        try:
            path = candidate.resolve()
        except OSError:
            path = candidate
        if path.exists():
            rendered = str(path)
            if rendered not in entries:
                entries.append(rendered)
    return entries


def check_python_imports(
    name: str,
    python_exe: str,
    modules: list[tuple[str, str]],
    required: bool,
) -> CheckResult:
    executable = Path(str(python_exe or "")).expanduser()
    if not str(python_exe or "").strip():
        return CheckResult(name, "skip", "No separate Python executable is configured.")
    if not executable.is_file():
        return CheckResult(
            name,
            "fail" if required else "warn",
            f"{executable} does not exist.",
            "Set realtime_preview_python to a Python environment that has kroko_onnx installed.",
        )
    script = (
        "import importlib.util, json; "
        f"mods={json.dumps([module for module, _label in modules])}; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}, sort_keys=True))"
    )
    env = dict(os.environ)
    entries = subprocess_pythonpath_entries()
    if env.get("PYTHONPATH"):
        entries.extend(item for item in env["PYTHONPATH"].split(os.pathsep) if item)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    try:
        completed = subprocess.run(
            [str(executable), "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
            env=env,
        )
    except Exception as exc:
        return CheckResult(
            name,
            "fail" if required else "warn",
            f"{type(exc).__name__}: {exc}",
            "Check the realtime preview Python path and environment.",
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        first = detail[-1] if detail else f"exit {completed.returncode}"
        return CheckResult(
            name,
            "fail" if required else "warn",
            first,
            "Install the missing preview runtime packages into that Python environment.",
        )
    try:
        payload = json.loads((completed.stdout or "{}").strip().splitlines()[-1])
    except Exception:
        payload = {}
    missing = [
        label
        for module, label in modules
        if not bool(payload.get(module))
    ]
    if missing:
        return CheckResult(
            name,
            "fail" if required else "warn",
            "Missing in preview Python: " + ", ".join(missing),
            "Install those packages into the realtime preview Python environment.",
        )
    return CheckResult(name, "ok", f"{executable} can import the realtime preview runtime.")


def installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_import_group(name: str, modules: list[tuple[str, str]], required: bool) -> CheckResult:
    missing = [label for module, label in modules if not module_available(module)]
    if not missing:
        return CheckResult(name, "ok", "All required Python modules are importable.")
    status = "fail" if required else "warn"
    return CheckResult(
        name,
        status,
        "Missing: " + ", ".join(missing),
        "Open `whospeaks` and use Install / repair on the Setup tab.",
    )


def command_version(command: str, args: list[str], timeout_seconds: float = 5.0) -> tuple[bool, str]:
    executable = shutil.which(command)
    if not executable:
        return False, f"{command} was not found on PATH."
    try:
        completed = subprocess.run(
            [executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return False, f"{command} exists at {executable}, but probing it failed: {type(exc).__name__}: {exc}"
    output = (completed.stdout or completed.stderr or "").splitlines()
    first_line = output[0].strip() if output else executable
    if completed.returncode != 0:
        return False, f"{command} returned {completed.returncode}: {first_line}"
    return True, first_line


def check_port(host: str, port: int) -> CheckResult:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
    except OSError as exc:
        return CheckResult(
            "Browser UI port",
            "fail",
            f"{host}:{port} is not available: {exc}",
            "Choose a different port in the starter CLI or stop the process using this port.",
        )
    return CheckResult("Browser UI port", "ok", f"{host}:{port} is available.")


def read_json_url(url: str, timeout_seconds: float = 3.0) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        return False, f"HTTP {exc.code}: {detail[:240]}", None
    except URLError as exc:
        return False, f"Connection failed: {exc.reason}", None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None
    if not raw:
        return True, "Empty response body.", {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"Non-JSON response: {raw[:240]}", None
    if not isinstance(payload, dict):
        return False, "Response JSON was not an object.", None
    return True, "JSON response received.", payload


def check_remote_health(name: str, base_url: str, required: bool) -> CheckResult:
    ok, detail, payload = read_json_url(f"{base_url.rstrip('/')}/health")
    if not ok:
        return CheckResult(
            name,
            "fail" if required else "warn",
            f"{base_url}/health: {detail}",
            "Start the service, update the URL, or switch to local mode.",
        )
    if payload and payload.get("ok") is False:
        return CheckResult(
            name,
            "fail" if required else "warn",
            f"{base_url}/health returned ok=false: {payload}",
            "Check service logs for model loading, CUDA, or auth errors.",
        )
    identity = "ok"
    if payload:
        identity = str(payload.get("model") or payload.get("service") or payload.get("status") or "ok")
    return CheckResult(name, "ok", f"{base_url} is reachable ({identity}).")


def check_remote_providers(base_url: str, required: bool) -> CheckResult:
    ok, detail, payload = read_json_url(f"{base_url.rstrip('/')}/providers")
    if not ok:
        return CheckResult(
            "Remote embeddings providers",
            "fail" if required else "warn",
            f"{base_url}/providers: {detail}",
            "Start the embeddings service or switch to local mode.",
        )
    providers = payload.get("providers") if payload else None
    if not isinstance(providers, list) or not providers:
        return CheckResult(
            "Remote embeddings providers",
            "fail" if required else "warn",
            "The service did not report a provider list.",
            "Check that the embeddings server is the WhoSpeaks voice-embeddings-server.",
        )
    provider_ids = [
        str(item.get("id"))
        for item in providers[:5]
        if isinstance(item, dict) and item.get("id")
    ]
    suffix = "..." if len(providers) > 5 else ""
    return CheckResult(
        "Remote embeddings providers",
        "ok",
        f"{len(providers)} providers reported: {', '.join(provider_ids)}{suffix}",
    )


def detect_torch_cuda() -> CheckResult:
    if not module_available("torch"):
        return CheckResult(
            "CUDA visibility",
            "warn",
            "torch is not importable, so CUDA availability cannot be checked.",
            "Open `whospeaks` and use Install / repair on the Setup tab before local GPU checks.",
        )
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        cuda_entrypoint_available = hasattr(getattr(torch, "_C", None), "_cuda_setDevice")
        if cuda_available and cuda_entrypoint_available:
            return CheckResult("CUDA visibility", "ok", f"torch sees CUDA device count={torch.cuda.device_count()}.")
        return CheckResult(
            "CUDA visibility",
            "warn",
            "torch is installed but CUDA is not usable by PyTorch.",
            "The launcher uses device=auto by default and will fall back to CPU for PyTorch embeddings. Install a CUDA-enabled torch build for GPU embeddings.",
        )
    except Exception as exc:
        return CheckResult(
            "CUDA visibility",
            "warn",
            f"torch import failed while checking CUDA: {type(exc).__name__}: {exc}",
            "Check the PyTorch install and NVIDIA driver compatibility.",
        )


def runtime_cache_dir() -> Path:
    env_path = os.environ.get("WHOSPEAKS_CACHE_DIR")
    if env_path:
        return Path(env_path).expanduser()
    try:
        from paths import CACHE_DIR  # type: ignore

        return Path(CACHE_DIR)
    except Exception:
        return Path.cwd() / "runtime" / "cache"


def check_faster_whisper_cache(model: str, required: bool) -> CheckResult:
    cache = runtime_cache_dir() / "faster-whisper"
    model_name = str(model or "large-v2")
    expected = cache / f"models--Systran--faster-whisper-{model_name}"
    if expected.exists():
        return CheckResult("faster-whisper model cache", "ok", f"Found {expected}.")
    if cache.exists():
        matches = [path for path in cache.glob(f"**/*{model_name}*") if path.exists()]
        if matches:
            return CheckResult("faster-whisper model cache", "ok", f"Found cached model path {matches[0]}.")
        status = "warn" if required else "skip"
        return CheckResult(
            "faster-whisper model cache",
            status,
            f"No cached {model_name} model was found under {cache}.",
            "First local ASR startup may need to download the model; use --download-root or WHOSPEAKS_CACHE_DIR to point at an existing cache.",
        )
    status = "warn" if required else "skip"
    return CheckResult(
        "faster-whisper model cache",
        status,
        f"{cache} does not exist yet.",
        "This is normal before first local ASR startup; the model cache will be created when models download.",
    )


def check_embedding_cache(required: bool) -> CheckResult:
    cache = runtime_cache_dir()
    roots = [
        cache / "speechbrain",
        cache / "huggingface" / "hub",
        cache / "espnet_model_zoo",
        cache / "modelscope",
        cache / "wespeaker",
    ]
    present = [path for path in roots if path.exists()]
    if present:
        names = ", ".join(path.name for path in present[:4])
        suffix = "..." if len(present) > 4 else ""
        return CheckResult("Embedding model caches", "ok", f"Found cache roots: {names}{suffix}")
    status = "warn" if required else "skip"
    return CheckResult(
        "Embedding model caches",
        status,
        f"No known embedding model cache roots found under {cache}.",
        "First provider load may download models. Gated providers also need HF_TOKEN and accepted model terms.",
    )


def post_json_url(url: str, timeout_seconds: float = 30.0) -> tuple[bool, str, dict[str, Any] | None]:
    request = Request(url, data=b"", method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        return False, f"HTTP {exc.code}: {detail[:240]}", None
    except URLError as exc:
        return False, f"Connection failed: {exc.reason}", None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None
    if not raw:
        return True, "Empty response body.", {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"Non-JSON response: {raw[:240]}", None
    if not isinstance(payload, dict):
        return False, "Response JSON was not an object.", None
    return True, "JSON response received.", payload


def check_remote_provider_load(base_url: str, provider: str, device: str, required: bool) -> CheckResult:
    query = urlencode({"provider": provider, "device": device or "auto"})
    url = f"{base_url.rstrip('/')}/load?{query}"
    ok, detail, payload = post_json_url(url, timeout_seconds=60.0)
    if not ok:
        return CheckResult(
            "Remote provider load",
            "fail" if required else "warn",
            f"{provider}: {detail}",
            "Use the smoke provider first, check server CUDA/auth logs, or switch providers.",
        )
    if payload and payload.get("ok") is False:
        return CheckResult(
            "Remote provider load",
            "fail" if required else "warn",
            f"{provider}: {payload}",
            "Check server CUDA/auth/model-cache logs.",
        )
    elapsed = payload.get("elapsed_seconds") if payload else None
    if elapsed is not None:
        return CheckResult("Remote provider load", "ok", f"{provider} loaded in {float(elapsed):.2f}s.")
    return CheckResult("Remote provider load", "ok", f"{provider} loaded.")


def check_local_provider_syntax(provider: str, required: bool) -> CheckResult:
    try:
        from embeddings.embedding_providers import parse_embedding_provider_stack_specs

        specs = parse_embedding_provider_stack_specs(provider)
    except Exception as exc:
        return CheckResult(
            "Embedding provider syntax",
            "fail" if required else "warn",
            f"{type(exc).__name__}: {exc}",
            "Fix the provider string or choose the smoke provider.",
        )
    names = ", ".join(name for name, _weight in specs)
    return CheckResult("Embedding provider syntax", "ok", f"Provider stack parses as: {names}")


def run_doctor(profile: Profile, mode: str = "auto", deep: bool = False) -> DoctorReport:
    selected_mode = normalize_mode(mode)
    if selected_mode == "auto":
        selected_mode = normalize_mode(profile.mode)
    remote_required = selected_mode == "remote"
    local_required = selected_mode == "local"
    server_required = selected_mode == "server"
    checks: list[CheckResult] = []

    if sys.version_info >= (3, 11):
        checks.append(CheckResult("Python", "ok", f"{platform.python_implementation()} {platform.python_version()}"))
    else:
        checks.append(CheckResult("Python", "fail", "Python 3.11 or newer is required."))

    installed_version = installed_distribution_version(PACKAGE_NAME) or __version__
    checks.append(CheckResult("WhoSpeaks package", "ok", f"{PACKAGE_NAME} {installed_version}"))

    ffmpeg_ok, ffmpeg_detail = command_version("ffmpeg", ["-version"])
    checks.append(CheckResult(
        "ffmpeg",
        "ok" if ffmpeg_ok else ("fail" if local_required or remote_required else "warn"),
        ffmpeg_detail,
        "Install ffmpeg and open a new shell so PATH is refreshed.",
    ))

    checks.append(check_import_group(
        "Controller Python modules",
        [
            ("av", "av"),
            ("numpy", "numpy"),
            ("soundfile", "soundfile"),
            ("librosa", "librosa"),
            ("nltk", "nltk"),
            ("yt_dlp", "yt-dlp"),
            ("onnxruntime", "onnxruntime"),
        ],
        required=local_required or remote_required,
    ))

    checks.append(check_import_group(
        "Local ASR modules",
        [("faster_whisper", "faster-whisper")],
        required=local_required,
    ))
    checks.append(check_faster_whisper_cache(profile.model, required=local_required))

    checks.append(check_import_group(
        "Local embedding modules",
        [
            ("torch", "torch"),
            ("torchaudio", "torchaudio"),
            ("speechbrain", "speechbrain"),
            ("pyannote.audio", "pyannote.audio"),
            ("resemblyzer", "resemblyzer"),
        ],
        required=local_required,
    ))
    checks.append(check_embedding_cache(required=local_required))
    if local_required and deep:
        checks.append(check_local_provider_syntax(profile.embedding_provider, required=True))
    elif local_required:
        checks.append(CheckResult(
            "Embedding provider syntax",
            "skip",
            "Use `whospeaks doctor --mode local --deep` to parse the selected provider stack.",
        ))

    if local_required:
        checks.append(detect_torch_cuda())
    else:
        checks.append(CheckResult("CUDA visibility", "skip", "CUDA is only checked for local all-in-one mode."))

    checks.append(check_import_group(
        "Server Python modules",
        [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("faster_whisper", "faster-whisper"),
        ],
        required=server_required,
    ))

    if remote_required:
        checks.append(check_remote_health("Remote ASR health", profile.remote_asr_url, required=True))
        checks.append(check_remote_health("Remote embeddings health", profile.remote_embeddings_url, required=True))
        checks.append(check_remote_providers(profile.remote_embeddings_url, required=True))
        if deep:
            checks.append(check_remote_provider_load(
                profile.remote_embeddings_url,
                profile.embedding_provider or SMOKE_PROVIDER,
                "auto",
                required=True,
            ))
        else:
            checks.append(CheckResult(
                "Remote provider load",
                "skip",
                "Use `whospeaks doctor --mode remote --deep` to POST /load for the selected provider.",
            ))
    else:
        checks.append(CheckResult("Remote ASR health", "skip", "Remote ASR is not required in this profile."))
        checks.append(CheckResult("Remote embeddings health", "skip", "Remote embeddings are not required in this profile."))
        checks.append(CheckResult("Remote embeddings providers", "skip", "Remote embeddings are not required in this profile."))
        checks.append(CheckResult("Remote provider load", "skip", "Remote embeddings are not required in this profile."))

    preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
    if preview_engine in {"off", "mock"}:
        checks.append(CheckResult("Realtime preview", "skip", f"Preview engine is {preview_engine}."))
    elif preview_engine == "sherpa_onnx":
        try:
            preset = normalize_preview_model_preset("sherpa_onnx", profile.realtime_preview_model_preset)
            model_dir = Path(profile.realtime_preview_model_dir).expanduser() if profile.realtime_preview_model_dir else default_sherpa_onnx_model_dir(preset)
            missing_model_files = missing_sherpa_onnx_model_files(model_dir)
        except (OSError, ValueError) as exc:
            checks.append(CheckResult("Nemotron model folder", "warn", str(exc)))
        else:
            if model_dir.exists() and missing_model_files:
                checks.append(CheckResult(
                    "Nemotron model folder",
                    "warn",
                    f"{model_dir} is incomplete: missing {', '.join(missing_model_files)}.",
                    "Remove the incomplete folder, or select a complete unpacked Nemotron model folder.",
                ))
            elif missing_model_files:
                checks.append(CheckResult(
                    "Nemotron model folder",
                    "skip",
                    f"{preset} will download to {model_dir} on first realtime-preview start.",
                ))
            else:
                checks.append(CheckResult("Nemotron model folder", "ok", f"{preset} is ready at {model_dir}."))
        if profile.realtime_preview_python:
            checks.append(check_python_imports(
                "Nemotron preview Python",
                profile.realtime_preview_python,
                [
                    ("workers.sherpa_onnx_realtime_preview_worker", "WhoSpeaks Nemotron worker"),
                    ("sherpa_onnx", "sherpa-onnx"),
                    ("numpy", "numpy"),
                ],
                required=False,
            ))
        elif module_available("sherpa_onnx"):
            checks.append(CheckResult("Nemotron sherpa-onnx runtime", "ok", "sherpa_onnx is importable."))
        else:
            checks.append(CheckResult(
                "Nemotron sherpa-onnx runtime",
                "warn",
                "sherpa_onnx is not importable, so Nemotron live text cannot start yet.",
                "Run the WhoSpeaks installer with Nemotron selected, or install sherpa-onnx and sherpa-onnx-bin in this Python environment.",
            ))
    else:
        checks.append(check_import_group(
            "Realtime preview",
            [("RealtimeSTT", "RealtimeSTT")],
            required=False,
        ))
        if profile.realtime_preview_python:
            checks.append(check_python_imports(
                "Realtime preview Python",
                profile.realtime_preview_python,
                [
                    ("workers.kroko_realtime_preview_worker", "WhoSpeaks preview worker"),
                    ("RealtimeSTT", "RealtimeSTT"),
                    ("kroko_onnx", "kroko_onnx"),
                    ("numpy", "numpy"),
                ],
                required=False,
            ))
        elif module_available("kroko_onnx"):
            checks.append(CheckResult("Kroko ONNX runtime", "ok", "kroko_onnx is importable."))
        else:
            checks.append(CheckResult(
                "Kroko ONNX runtime",
                "warn",
                "kroko_onnx is not importable, so live Kroko text cannot start yet.",
                "Install a kroko_onnx wheel for this Python, or run `python -m RealtimeSTT.install_kroko --build` where that build path is supported.",
            ))

    checks.append(check_port(profile.host, profile.port))
    checks.append(CheckResult("Launch profile", "ok", f"{selected_mode} profile; command can be printed with whospeaks launch --print."))
    return DoctorReport(selected_mode, checks)


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    return {
        "mode": report.mode,
        "worst_status": report.worst_status,
        "checks": [dataclasses.asdict(check) for check in report.checks],
    }


def print_report(report: DoctorReport) -> None:
    print(f"WhoSpeaks doctor report ({report.mode})")
    print("=" * 72)
    width = max([len(check.name) for check in report.checks] + [10])
    for check in report.checks:
        label = STATUS_LABEL[check.status]
        print(f"{label:<5} {check.name:<{width}} {check.detail}")
        if check.remediation and check.status in {"warn", "fail"}:
            print(f"{'':<6}{'':<{width}} Fix: {check.remediation}")
    print("=" * 72)
    if report.has_failures:
        print("Result: action needed before this profile is ready.")
    elif report.has_warnings:
        print("Result: usable with warnings.")
    else:
        print("Result: ready.")


def package_extra_spec(extra: str) -> str:
    version = installed_distribution_version(PACKAGE_NAME)
    if version:
        return f"{PACKAGE_NAME}[{extra}]=={version}"
    return f"{PACKAGE_NAME}[{extra}]"


def normalize_install_target(value: str | None) -> str:
    normalized = str(value or "local").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
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
    }
    normalized = aliases.get(normalized, normalized)
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
) -> InstallPlan:
    selected = normalize_install_target(target)
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
        )
    if selected == "core":
        return InstallPlan(
            target=selected,
            title="Core/controller for remote ASR and embeddings servers",
            mode="remote",
            extra=f"{CONTROLLER_EXTRA},{PREVIEW_EXTRA}" if preview_selected else CONTROLLER_EXTRA,
            install_kroko=kroko_selected,
            summary="Browser controller on this machine with final ASR and embeddings served over HTTP.",
            realtime_preview_engine=engine,
            realtime_preview_model_preset=preset,
        )
    return InstallPlan(
        target=selected,
        title="ASR and embeddings server packages",
        mode="server",
        extra=SERVER_EXTRA,
        install_kroko=False,
        summary="Service-side dependencies for the remote faster-whisper ASR and embeddings endpoints.",
        realtime_preview_engine="off",
    )


def configure_profile_for_install(profile: Profile, plan: InstallPlan) -> Profile:
    configure_profile_for_mode(profile, plan.mode)
    if plan.target in {"local", "core"}:
        profile.realtime_preview_engine = plan.realtime_preview_engine
        profile.realtime_preview_model_preset = plan.realtime_preview_model_preset
        if plan.realtime_preview_engine != "sherpa_onnx":
            profile.realtime_preview_model_dir = ""
    else:
        profile.realtime_preview_engine = "off"
        profile.realtime_preview_model_preset = ""
        profile.realtime_preview_model_dir = ""
    return profile


def version_is_prerelease(version: str | None) -> bool:
    normalized = str(version or "").lower()
    return any(marker in normalized for marker in ("a", "b", "rc", "dev"))


def pip_index_args_for_installed_package() -> list[str]:
    args: list[str] = []
    index_url = os.environ.get(PIP_INDEX_URL_ENV, "").strip()
    extra_index_url = os.environ.get(PIP_EXTRA_INDEX_URL_ENV, "").strip()
    find_links = os.environ.get(PIP_FIND_LINKS_ENV, "").strip()
    if index_url:
        args.extend(["--index-url", index_url])
    if extra_index_url:
        args.extend(["--extra-index-url", extra_index_url])
    elif version_is_prerelease(installed_distribution_version(PACKAGE_NAME)):
        args.extend(["--extra-index-url", TESTPYPI_SIMPLE_URL])
    if find_links:
        args.extend(["--find-links", find_links])
    return args


def build_install_command(extra: str = LOCAL_EXTRA) -> list[str]:
    extra = str(extra or LOCAL_EXTRA).strip()
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        *pip_index_args_for_installed_package(),
        package_extra_spec(extra),
    ]


def normalize_torch_install_policy(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "gpu": "cuda",
        "nvidia": "cuda",
        "cu118": "cuda",
        "cu126": "cuda",
        "cu128": "cuda",
        "none": "skip",
        "off": "skip",
        "no": "skip",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TORCH_INSTALL_POLICY_CHOICES:
        raise SystemExit(
            "Unknown Torch install policy {0!r}. Choose one of: {1}.".format(
                value,
                ", ".join(TORCH_INSTALL_POLICY_CHOICES),
            )
        )
    return normalized


def normalize_pytorch_cuda_build(value: str | None) -> str:
    normalized = str(value or DEFAULT_PYTORCH_CUDA_BUILD).strip().lower().replace(".", "")
    aliases = {
        "cuda": DEFAULT_PYTORCH_CUDA_BUILD,
        "gpu": DEFAULT_PYTORCH_CUDA_BUILD,
        "12": DEFAULT_PYTORCH_CUDA_BUILD,
        "128": "cu128",
        "12_8": "cu128",
        "12-8": "cu128",
        "126": "cu126",
        "12_6": "cu126",
        "12-6": "cu126",
        "118": "cu118",
        "11_8": "cu118",
        "11-8": "cu118",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PYTORCH_CUDA_INDEX_URLS:
        raise SystemExit(
            "Unknown PyTorch CUDA build {0!r}. Choose one of: {1}.".format(
                value,
                ", ".join(PYTORCH_CUDA_INDEX_URLS),
            )
        )
    return normalized


def extra_needs_torch(extra: str) -> bool:
    tokens = {item.strip() for item in str(extra or "").split(",") if item.strip()}
    return bool(tokens & {"all", "complete", "local", "server"})


def detect_nvidia_cuda() -> tuple[bool, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi was not found; assuming CPU PyTorch."
    command = [
        nvidia_smi,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return False, f"nvidia-smi could not be executed ({type(exc).__name__}: {exc}); assuming CPU PyTorch."
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            return False, f"nvidia-smi returned {completed.returncode}: {detail}; assuming CPU PyTorch."
        return False, f"nvidia-smi returned {completed.returncode}; assuming CPU PyTorch."
    first_line = next((line.strip() for line in (completed.stdout or "").splitlines() if line.strip()), "")
    if not first_line:
        return False, "nvidia-smi did not report an NVIDIA GPU; assuming CPU PyTorch."
    return True, f"NVIDIA GPU/driver detected: {first_line}."


def select_torch_install(policy: str | None = None) -> TorchInstallSelection:
    selected_policy = normalize_torch_install_policy(
        policy if policy is not None else os.environ.get(TORCH_INSTALL_POLICY_ENV, "auto")
    )
    if selected_policy == "skip":
        return TorchInstallSelection("skip", "", "Torch preinstall disabled.")
    cuda_build = normalize_pytorch_cuda_build(os.environ.get(PYTORCH_CUDA_BUILD_ENV, DEFAULT_PYTORCH_CUDA_BUILD))
    cuda_index = os.environ.get(PYTORCH_CUDA_INDEX_URL_ENV, "").strip() or PYTORCH_CUDA_INDEX_URLS[cuda_build]
    if selected_policy == "cuda":
        return TorchInstallSelection("cuda", cuda_index, f"CUDA PyTorch forced with {cuda_build}.", cuda_build)
    if selected_policy == "cpu":
        cpu_index = os.environ.get(PYTORCH_CPU_INDEX_URL_ENV, "").strip() or (
            "" if platform.system() == "Darwin" else PYTORCH_CPU_INDEX_URL
        )
        return TorchInstallSelection("cpu", cpu_index, "CPU PyTorch forced.")
    has_cuda, detail = detect_nvidia_cuda()
    if has_cuda and platform.system() in {"Windows", "Linux"}:
        return TorchInstallSelection("cuda", cuda_index, f"{detail} Installing PyTorch {cuda_build}.", cuda_build)
    cpu_index = os.environ.get(PYTORCH_CPU_INDEX_URL_ENV, "").strip() or (
        "" if platform.system() == "Darwin" else PYTORCH_CPU_INDEX_URL
    )
    return TorchInstallSelection("cpu", cpu_index, detail)


def build_torch_install_command(policy: str | None = None) -> tuple[list[str], TorchInstallSelection]:
    selection = select_torch_install(policy)
    if not selection.should_install:
        return [], selection
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    ]
    if selection.index_url:
        command.extend(["--index-url", selection.index_url])
    command.extend(TORCH_PACKAGE_SPECS)
    return command, selection


def report_torch_runtime(selection: TorchInstallSelection) -> int:
    if not selection.should_install:
        return 0
    script = (
        "import json, torch; "
        "print(json.dumps({"
        "'version': getattr(torch, '__version__', ''), "
        "'cuda_runtime': getattr(getattr(torch, 'version', None), 'cuda', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        print(f"PyTorch verification failed: {detail}")
        return int(completed.returncode)
    detail = (completed.stdout or "").strip().splitlines()[-1]
    print(f"PyTorch verification: {detail}")
    if selection.mode == "cuda":
        try:
            payload = json.loads(detail)
        except Exception:
            payload = {}
        if not payload.get("cuda_available"):
            print("Warning: CUDA PyTorch installed, but torch.cuda.is_available() is false. The launcher can still fall back to CPU.")
    return 0


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        rendered: list[str] = []
        for arg in command:
            if any(token in arg for token in ("[", "]", "<", ">", "&", "|")):
                rendered.append('"' + arg.replace('"', '\\"') + '"')
            else:
                rendered.append(subprocess.list2cmdline([arg]))
        return " ".join(rendered)
    return shlex.join(command)


def recommended_install_extra(profile: Profile, report: DoctorReport) -> str | None:
    if report.mode == "server":
        return "server"
    if report.mode == "local":
        for check in report.checks:
            if check.status == "fail" and check.name in {
                "Controller Python modules",
                "Local ASR modules",
                "Local embedding modules",
            }:
                return LOCAL_EXTRA
        if any(check.status == "warn" and check.name == "CUDA visibility" for check in report.checks):
            return LOCAL_EXTRA
        if any(check.status == "warn" and check.name == "Realtime preview" for check in report.checks):
            return PREVIEW_EXTRA
    if report.mode == "remote":
        for check in report.checks:
            if check.status == "fail" and check.name == "Controller Python modules":
                return CONTROLLER_EXTRA
        if any(check.status == "warn" and check.name == "Realtime preview" for check in report.checks):
            return PREVIEW_EXTRA
    return None


def install_extra(
    extra: str,
    assume_yes: bool = False,
    dry_run: bool = False,
    *,
    torch_policy: str | None = None,
) -> int:
    command = build_install_command(extra)
    torch_command: list[str] = []
    torch_selection = TorchInstallSelection("skip", "", "Torch is not needed for this dependency set.")
    if extra_needs_torch(extra):
        torch_command, torch_selection = build_torch_install_command(torch_policy)
        print("PyTorch install selection:")
        print(f"  {torch_selection.reason}")
        if torch_command:
            print("PyTorch install command:")
            print(f"  {format_command(torch_command)}")
        else:
            print("PyTorch install command:")
            print("  skipped")
    print("WhoSpeaks package install command:")
    print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Run these install commands now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Install skipped.")
            return 0
    if torch_command:
        completed = subprocess.run(torch_command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
        code = report_torch_runtime(torch_selection)
        if code:
            return code
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def print_install_plan(plan: InstallPlan, profile: Profile) -> None:
    print("WhoSpeaks install plan")
    print("=" * 72)
    print(f"Target: {plan.title}")
    print(f"Profile: {plan.mode}")
    print_wrapped(plan.summary, initial_indent="Summary: ", subsequent_indent="         ", style=detail_text)
    if plan.realtime_preview_engine == "sherpa_onnx":
        preset = plan.realtime_preview_model_preset
        latency_note = "higher stability" if "560ms" in preset else "lower latency"
        print(f"Realtime text: Nemotron 3.5 ({preset}; {latency_note}).")
        print("               sherpa-onnx packages install with WhoSpeaks; the verified model downloads on first launch.")
    elif plan.install_kroko:
        print("Realtime text: enabled; Kroko native runtime will be offered after Python packages.")
    elif plan.target in {"local", "core"}:
        print("Realtime text: disabled for this install. Run the installer again and choose Kroko to try native live text.")
    else:
        print("Realtime text: not part of the server package install.")
    print(f"Language: {language_summary(profile.language)}")
    print(f"Internal dependency set: {plan.extra}")
    print("Underlying pip command:")
    print(f"  {format_command(build_install_command(plan.extra))}")
    print("=" * 72)


def prompt_install_target() -> str:
    print(textwrap.dedent(
        """
        What do you want to install?
          1. Full local installation
          2. Core/controller for remote ASR and embeddings servers
          3. ASR and embeddings server packages
        """
    ).strip())
    while True:
        choice = read_input("> ", "1").strip().lower()
        if choice in {"1", "local", "full", "full local"}:
            return "local"
        if choice in {"2", "core", "controller", "remote"}:
            return "core"
        if choice in {"3", "server", "gpu", "services"}:
            return "server"
        print("Choose 1, 2, or 3.")


def prompt_realtime_preview(target: str) -> tuple[str, str]:
    if normalize_install_target(target) not in {"local", "core"}:
        return "off", ""
    print()
    print("Realtime preview text")
    print_wrapped(
        "1. Nemotron 3.5 (recommended on Windows): installs sherpa-onnx packages normally and downloads a verified model on first use. "
        "2. Nemotron 3.5 low latency: 160ms model, less robust. "
        "3. Kroko/Banafo: native setup that can need Python 3.12 and Docker Desktop on Windows. "
        "4. Disable realtime text. Final ASR and speaker diarization work independently.",
        initial_indent="",
        subsequent_indent="",
        style=detail_text,
    )
    while True:
        answer = read_input("Choose realtime preview [1/2/3/4] ", "1").strip().lower()
        if answer in {"1", "nemotron", "nemotron-560", "560"}:
            return "sherpa_onnx", "nemotron-3.5-560ms-int8"
        if answer in {"2", "nemotron-160", "160"}:
            return "sherpa_onnx", "nemotron-3.5-160ms-int8"
        if answer in {"3", "kroko", "banafo"}:
            return "kroko_onnx", "community-64l"
        if answer in {"4", "off", "none", "no"}:
            return "off", ""
        print("Choose 1, 2, 3, or 4.")


def prompt_kroko_install(target: str) -> bool:
    """Legacy prompt helper retained for classic callers and integrations."""

    engine, _preset = prompt_realtime_preview(target)
    return engine == "kroko_onnx"


def confirm_install_start(assume_yes: bool, dry_run: bool) -> bool:
    if assume_yes or dry_run:
        return True
    if not sys.stdin.isatty():
        print("Use --yes to run the installer non-interactively.")
        return False
    answer = read_input("Start this install now? [Y/n] ", "y").strip().lower()
    return answer in {"", "y", "yes"}


def preview_engine_is_enabled(profile: Profile) -> bool:
    return normalize_preview_engine(profile.realtime_preview_engine) not in {"off", "mock"}


def preview_engine_uses_kroko(profile: Profile) -> bool:
    return normalize_preview_engine(profile.realtime_preview_engine) == "kroko_onnx"


def validate_realtime_preview_language(profile: Profile) -> None:
    error = preview_language_error(profile.realtime_preview_engine, profile.language)
    if error:
        raise SystemExit(error)


def build_kroko_install_command(
    python_executable: str | Path = sys.executable,
    *,
    variant: str = "free",
    work_dir: str | Path | None = None,
) -> list[str]:
    command = [str(python_executable), "-m", KROKO_INSTALL_MODULE, "--build", "--variant", str(variant)]
    if work_dir:
        command.extend(["--work-dir", str(work_dir)])
    return command


def default_kroko_preview_venv_path() -> Path:
    override = os.environ.get(KROKO_PREVIEW_VENV_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or config_path().parent)
        return root / "WhoSpeaks" / "kroko-preview-py312"
    root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return root / "whospeaks" / "kroko-preview-py312"


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def query_python_command_info(command: list[str]) -> dict[str, Any] | None:
    script = (
        "import json, platform, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'version': list(sys.version_info[:3]), "
        "'bits': 64 if sys.maxsize > 2**32 else 32, "
        "'machine': platform.machine()"
        "}, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [*command, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads((completed.stdout or "{}").strip().splitlines()[-1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def windows_python312_command() -> list[str] | None:
    launcher = shutil.which("py")
    candidates: list[list[str]] = []
    if launcher:
        candidates.append([launcher, "-3.12"])
    for path in (
        Path("C:/Python/Python312/python.exe"),
        Path("C:/Python312/python.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe",
    ):
        if path.is_file():
            candidates.append([str(path)])
    for name in ("python3.12", "python"):
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])
    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        info = query_python_command_info(command)
        if not info:
            continue
        version = info.get("version") or []
        bits = int(info.get("bits") or 0)
        if len(version) >= 2 and version[0] == 3 and version[1] == 12 and bits == 64:
            return command
    return None


def report_suggests_kroko_install(profile: Profile, report: DoctorReport) -> bool:
    if normalize_mode(profile.mode) != "local":
        return False
    if not preview_engine_uses_kroko(profile):
        return False
    relevant_names = {"Realtime preview", "Realtime preview Python", "Kroko ONNX runtime"}
    for check in report.checks:
        if check.name in relevant_names and check.status in {"warn", "fail"}:
            return True
    return False


def run_command_sequence(commands: list[list[str]]) -> int:
    for command in commands:
        print(f"+ {format_command(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def install_kroko_in_python(
    python_executable: str | Path,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
) -> int:
    command = build_kroko_install_command(python_executable, variant=variant, work_dir=work_dir)
    print("Kroko native runtime install command:")
    print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Build/install Kroko native runtime now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Kroko native runtime install skipped.")
            return 0
    code = run_command_sequence([command])
    if code and soft_fail:
        print("Kroko native runtime install did not complete. Final ASR can still run without realtime preview text.")
        return 0
    return code


def install_kroko_sidecar(
    profile: Profile,
    python312_command: list[str],
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
) -> int:
    venv_dir = default_kroko_preview_venv_path()
    preview_python = venv_python_path(venv_dir)
    commands = [
        [*python312_command, "-m", "venv", str(venv_dir)],
        [str(preview_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [
            str(preview_python),
            "-m",
            "pip",
            "install",
            *pip_index_args_for_installed_package(),
            package_extra_spec(PREVIEW_EXTRA),
        ],
        build_kroko_install_command(preview_python, variant=variant, work_dir=work_dir),
    ]
    config_command = [
        sys.executable,
        "-m",
        "whospeaks_cli",
        "config",
        "--realtime-preview-python",
        str(preview_python),
    ]
    print("Kroko native runtime setup will use a Python 3.12 realtime-preview sidecar:")
    for command in commands:
        print(f"  {format_command(command)}")
    print(f"  {format_command(config_command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Create sidecar and build/install Kroko now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Kroko native runtime install skipped.")
            return 0
    try:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Could not create sidecar parent directory: {exc}")
        return 0 if soft_fail else 1
    code = run_command_sequence(commands)
    if code:
        if soft_fail:
            print("Kroko sidecar setup did not complete. Final ASR can still run without realtime preview text.")
            return 0
        return code
    profile.realtime_preview_python = str(preview_python)
    save_path = save_profile(profile)
    print(f"Saved realtime preview Python to {save_path}.")
    return 0


def install_kroko_runtime(
    profile: Profile,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
) -> int:
    if not preview_engine_uses_kroko(profile):
        print("Kroko native runtime install skipped because realtime preview is disabled.")
        return 0
    if profile.realtime_preview_python:
        return install_kroko_in_python(
            profile.realtime_preview_python,
            assume_yes=assume_yes,
            dry_run=dry_run,
            variant=variant,
            work_dir=work_dir,
            soft_fail=soft_fail,
        )
    if os.name == "nt" and sys.version_info[:2] != (3, 12):
        python312 = windows_python312_command()
        if python312:
            return install_kroko_sidecar(
                profile,
                python312,
                assume_yes=assume_yes,
                dry_run=dry_run,
                variant=variant,
                work_dir=work_dir,
                soft_fail=soft_fail,
            )
        print(
            "Kroko realtime preview on Windows currently needs CPython 3.12 x64 for the native build path. "
            "Install Python 3.12 x64 or set realtime_preview_python to a prepared Python 3.12 environment, "
            "then open `whospeaks`, enable Kroko on the Setup tab, and install again."
        )
        return 0 if (soft_fail or dry_run) else 1
    return install_kroko_in_python(
        sys.executable,
        assume_yes=assume_yes,
        dry_run=dry_run,
        variant=variant,
        work_dir=work_dir,
        soft_fail=soft_fail,
    )


def install_extra_and_maybe_kroko(
    profile: Profile,
    extra: str,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    install_kroko: bool = True,
    kroko_assume_yes: bool | None = None,
    torch_policy: str | None = None,
) -> int:
    code = install_extra(extra, assume_yes=assume_yes, dry_run=dry_run, torch_policy=torch_policy)
    if code:
        return code
    preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
    if preview_engine == "sherpa_onnx":
        preset = profile.realtime_preview_model_preset or "nemotron-3.5-560ms-int8"
        print(
            f"Nemotron realtime preview is configured with {preset}. "
            "The verified model downloads automatically on first preview start."
        )
        return 0
    if not install_kroko:
        return 0
    if preview_engine != "kroko_onnx":
        return 0
    report = run_doctor(profile)
    if not report_suggests_kroko_install(profile, report):
        return 0
    print("Kroko native runtime is required for realtime preview text.")
    return install_kroko_runtime(
        profile,
        assume_yes=assume_yes if kroko_assume_yes is None else kroko_assume_yes,
        dry_run=dry_run,
        soft_fail=True,
    )


def configure_profile_for_mode(profile: Profile, mode: str) -> Profile:
    selected = normalize_mode(mode)
    profile.mode = selected
    if selected == "local":
        profile.asr_backend = "local"
        profile.embeddings_backend = "local"
        profile.device = "auto"
        apply_provider_preset(profile, "smoke")
        profile.vad_backend = "rms"
        profile.realtime_preview_engine = "sherpa_onnx"
        profile.realtime_preview_model_preset = "nemotron-3.5-560ms-int8"
        profile.realtime_preview_model_dir = ""
    elif selected == "remote":
        profile.asr_backend = "remote"
        profile.embeddings_backend = "remote"
        profile.device = "auto"
        apply_provider_preset(profile, "smoke")
        profile.vad_backend = "rms"
        profile.realtime_preview_engine = "off"
        profile.realtime_preview_model_preset = ""
        profile.realtime_preview_model_dir = ""
    elif selected == "server":
        profile.asr_backend = "remote"
        profile.embeddings_backend = "remote"
        profile.realtime_preview_engine = "off"
        profile.realtime_preview_model_preset = ""
        profile.realtime_preview_model_dir = ""
    return profile


def build_launch_command(profile: Profile, extra_args: str = "") -> list[str]:
    executable = shutil.which("whospeaks-window")
    if executable:
        command = [executable]
    else:
        command = [sys.executable, "-m", "window.youtube_window_diarize_gui"]
    command.extend([
        "--host",
        str(profile.host),
        "--port",
        str(int(profile.port)),
        "--language",
        str(profile.language),
        "--model",
        str(profile.model),
        "--device",
        str(profile.device),
        "--compute-type",
        str(profile.compute_type),
        "--asr-backend",
        str(profile.asr_backend),
        "--embeddings-backend",
        str(profile.embeddings_backend),
        "--embedding-provider",
        str(profile.embedding_provider),
        "--live-speaker-embedding-provider",
        str(profile.live_speaker_embedding_provider),
        "--vad-backend",
        str(profile.vad_backend),
        "--realtime-preview-engine",
        str(profile.realtime_preview_engine or "off"),
    ])
    if profile.embeddings_backend == "local":
        command.extend(["--embedding-python", str(profile.embedding_python or sys.executable)])
    preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
    if preview_engine in {"kroko_onnx", "sherpa_onnx"} and profile.realtime_preview_model_preset:
        command.extend(["--realtime-preview-model-preset", str(profile.realtime_preview_model_preset)])
    if preview_engine == "sherpa_onnx" and profile.realtime_preview_model_dir:
        command.extend(["--realtime-preview-model-dir", str(profile.realtime_preview_model_dir)])
    if profile.realtime_preview_python or preview_engine not in {"off", "none", "false", "0"}:
        command.extend(["--realtime-preview-python", str(profile.realtime_preview_python or sys.executable)])
    if profile.asr_backend == "remote":
        command.extend(["--remote-asr-url", str(profile.remote_asr_url)])
    if profile.embeddings_backend == "remote":
        command.extend(["--remote-embeddings-url", str(profile.remote_embeddings_url)])
    advanced = " ".join(item for item in [profile.advanced_args, extra_args] if item)
    if advanced:
        command.extend(shlex.split(advanced))
    return command


def build_server_launch_lines() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    asr_dir = root / "vendor" / "remote_servers" / "faster-whisper-asr"
    embeddings_dir = root / "vendor" / "remote_servers" / "voice-embeddings-server"

    def line(directory: Path, app: str, port: int) -> str:
        command = format_command([sys.executable, "-m", "uvicorn", app, "--host", "0.0.0.0", "--port", str(port)])
        if directory.is_dir():
            if os.name == "nt":
                return f'cd /d "{directory}" && {command}'
            return f"cd {shlex.quote(str(directory))} && {command}"
        return command

    return [
        line(asr_dir, "asr_server:app", 8650),
        line(embeddings_dir, "embeddings_server:app", 8660),
    ]


def print_profile(profile: Profile) -> None:
    print("Current starter profile")
    print("=" * 72)
    print(f"Saved config: {config_path()}")
    print("-" * 72)
    for key, label, _help_text in EDITABLE_PROFILE_FIELDS:
        print(f"{label:<28} {getattr(profile, key)}")
    print("=" * 72)
    print_launch_command(profile)


def print_provider_summary(profile: Profile, indent: str = "") -> None:
    preset = selected_provider_preset(profile)
    if preset is None:
        print(f"{indent}Provider preset: {primary_text('Custom')}")
        print(f"{indent}  {label_text('Simple:')} {primary_text('Manual provider strings. The launcher will use the exact values below.')}")
    else:
        print(f"{indent}Provider preset: {primary_text(preset.name)} ({preset.id})")
        print(f"{indent}  {label_text('Simple:')} {primary_text(preset.summary)}")
        print_wrapped(
            f"Deep: {preset.details}",
            initial_indent=f"{indent}  ",
            subsequent_indent=f"{indent}        ",
            style=detail_text,
        )
        if preset.score_note:
            print_wrapped(
                f"Validation note: {preset.score_note}",
                initial_indent=f"{indent}  ",
                subsequent_indent=f"{indent}        ",
                style=detail_text,
            )
        if preset.requirements:
            print_wrapped(
                f"Requirement: {preset.requirements}",
                initial_indent=f"{indent}  ",
                subsequent_indent=f"{indent}        ",
                style=detail_text,
            )
    print_wrapped(
        "Exact final provider: " + str(profile.embedding_provider),
        initial_indent=f"{indent}  ",
        subsequent_indent=f"{indent}        ",
        style=detail_text,
    )
    print_wrapped(
        "Exact live provider:  " + str(profile.live_speaker_embedding_provider),
        initial_indent=f"{indent}  ",
        subsequent_indent=f"{indent}        ",
        style=detail_text,
    )


def shorten_value(value: Any, width: int = 58) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def profile_field_metadata() -> dict[str, tuple[str, str]]:
    return {key: (label, help_text) for key, label, help_text in EDITABLE_PROFILE_FIELDS}


def profile_field_label(key: str) -> str:
    return profile_field_metadata().get(key, (key.replace("_", " ").title(), ""))[0]


def profile_field_help(key: str) -> str:
    return profile_field_metadata().get(key, ("", ""))[1]


def print_launch_command(profile: Profile, extra_args: str = "") -> None:
    if profile.mode == "server":
        print("Server profile service commands:")
        for line in build_server_launch_lines():
            print(f"  {line}")
        return
    print("Launch command:")
    command = format_command(build_launch_command(profile, extra_args))
    print_wrapped(command, width=100, initial_indent="  ", subsequent_indent="  ", style=primary_text)


def full_profile_editor_text(profile: Profile) -> str:
    lines = [
        "All Saved Profile Fields",
        "=" * 72,
        "Choose one field to edit. Press Enter at a prompt to keep the current value.",
        "-" * 72,
    ]
    for index, (key, label, help_text) in enumerate(EDITABLE_PROFILE_FIELDS, start=1):
        value = shorten_value(getattr(profile, key), 46)
        lines.append(f"{index:>2}. {label:<25} {value}")
        lines.append(detail_text(f"    {help_text}"))
    lines.append("b. Back")
    return "\n".join(lines)


def profile_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(Profile)}


def coerce_profile_value(profile: Profile, key: str, value: str) -> Any:
    current = getattr(profile, key)
    if key == "language":
        return normalize_language_code(value)
    if isinstance(current, int):
        return int(value)
    return value


def apply_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    explicit_provider_preset: str | None = None
    fields = profile_field_names()
    for raw_key, raw_value in updates:
        key = str(raw_key).strip().replace("-", "_")
        value = str(raw_value)
        if key not in fields:
            allowed = ", ".join(sorted(fields))
            raise SystemExit(f"Unknown profile field {key!r}. Known fields: {allowed}.")
        if key == "mode":
            configure_profile_for_mode(profile, value)
            continue
        if key == "provider_preset":
            explicit_provider_preset = value
            continue
        try:
            setattr(profile, key, coerce_profile_value(profile, key, value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    profile = Profile.from_mapping(profile.as_dict())
    if explicit_provider_preset is not None:
        apply_provider_preset(profile, explicit_provider_preset)
    else:
        profile.provider_preset = infer_provider_preset_id(
            profile.provider_preset,
            profile.embedding_provider,
            profile.live_speaker_embedding_provider,
        )
    return profile


def update_profile_in_place(profile: Profile, updated: Profile) -> Profile:
    profile.__dict__.update(updated.__dict__)
    return profile


def save_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    updated = apply_profile_updates(profile, updates)
    update_profile_in_place(profile, updated)
    save_path = save_profile(profile)
    changed = ", ".join(f"{key}={getattr(profile, str(key).replace('-', '_'))}" for key, _value in updates)
    print(f"Saved {changed} to {save_path}.")
    return profile


def try_save_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    try:
        return save_profile_updates(profile, updates)
    except SystemExit as exc:
        print(str(exc))
        return profile


def language_summary(language: str) -> str:
    try:
        config = get_language_config(language)
    except ValueError:
        return str(language)
    preview_support = []
    if config.kroko_code:
        preview_support.append(f"Kroko {config.kroko_code}")
    if preview_language_error("sherpa_onnx", config.code) is None:
        preview_support.append("Nemotron")
    preview = ", ".join(preview_support) or "no realtime preview"
    return f"{config.display_name} ({config.code}, {preview})"


def profile_summary_lines(profile: Profile) -> list[str]:
    preset = selected_provider_preset(profile)
    provider = f"{preset.name} ({preset.id})" if preset is not None else "Custom"
    return [
        f"Mode: {profile.mode}    ASR: {profile.asr_backend}    Embeddings: {profile.embeddings_backend}",
        f"Language: {language_summary(profile.language)}",
        f"Provider: {provider}",
        f"Final provider: {profile.embedding_provider}",
        f"Live provider:  {profile.live_speaker_embedding_provider}",
        f"Embedding Python: {profile.embedding_python or 'auto/current'}",
        f"ASR model/device: {profile.model} / {profile.device} / {profile.compute_type}",
        f"Realtime text: {profile.realtime_preview_engine} / {profile.realtime_preview_model_preset or 'default'}",
        f"Realtime model folder: {profile.realtime_preview_model_dir or 'automatic'}",
        f"Realtime Python: {profile.realtime_preview_python or 'auto/default'}",
        f"Browser: {profile.host}:{profile.port}",
    ]


def report_status_counts(report: DoctorReport) -> dict[str, int]:
    return {
        status: sum(1 for check in report.checks if check.status == status)
        for status in ("fail", "warn", "skip", "ok")
    }


def report_readiness_line(report: DoctorReport) -> str:
    counts = report_status_counts(report)
    if counts["fail"]:
        state = f"Action needed: {counts['fail']} failed check"
        if counts["fail"] != 1:
            state += "s"
    elif counts["warn"]:
        state = f"Usable with {counts['warn']} warning"
        if counts["warn"] != 1:
            state += "s"
    else:
        state = "Ready: no failed or warning checks"
    if counts["skip"]:
        state += f"; {counts['skip']} skipped"
    return state


def problem_checks(report: DoctorReport) -> list[CheckResult]:
    return [check for check in report.checks if check.is_problem()]


def render_dashboard(profile: Profile, report: DoctorReport) -> None:
    print()
    print("WhoSpeaks")
    print("=" * 72)
    print(f"Profile: {primary_text(profile.mode)}  ASR: {profile.asr_backend}  Embeddings: {profile.embeddings_backend}")
    print(f"Browser: {profile.host}:{profile.port}  Language: {language_summary(profile.language)}")
    if profile.embeddings_backend == "local":
        print(f"Embedding Python: {profile.embedding_python or 'auto/current'}")
    print(f"Realtime text: {profile.realtime_preview_engine}  Python: {profile.realtime_preview_python or 'auto/default'}")
    print_provider_summary(profile)
    if profile.embeddings_backend == "remote":
        print(f"Embeddings URL: {profile.remote_embeddings_url}")
    if profile.asr_backend == "remote":
        print(f"ASR URL: {profile.remote_asr_url}")
    print("-" * 72)
    print(f"Readiness: {report_readiness_line(report)}")
    problems = problem_checks(report)
    if problems:
        width = max([len(check.name) for check in problems] + [10])
        for check in problems:
            print(f"{STATUS_LABEL[check.status]:<5} {check.name:<{width}} {check.detail}")
            if check.remediation:
                print_wrapped(
                    "Fix: " + check.remediation,
                    initial_indent=f"{'':<6}{'':<{width}} ",
                    subsequent_indent=f"{'':<6}{'':<{width}} ",
                    style=detail_text,
                )
        print(detail_text("Run doctor for the complete component list."))
    else:
        print(primary_text("No actionable setup problems detected by the quick doctor pass."))
    print("-" * 72)
    print("Direct controls: language, realtime text, providers, backend URLs, ASR runtime, browser port.")
    print("Validation controls: doctor, install recommendation, print exact launch command, launch browser UI.")
    print("=" * 72)


def read_input(prompt: str, default: str = "") -> str:
    try:
        return input(prompt)
    except EOFError:
        print()
        return default


def prompt_value(label: str, current: Any) -> str:
    value = read_input(f"{label} [{current}]: ").strip()
    return value if value else str(current)


def edit_profile(profile: Profile) -> Profile:
    while True:
        print()
        print(full_profile_editor_text(profile))
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return profile
        try:
            selected = int(choice)
        except ValueError:
            print("Choose one field number or b.")
            continue
        fields = list(EDITABLE_PROFILE_FIELDS)
        if not 1 <= selected <= len(fields):
            print("Choose one of the listed field numbers.")
            continue
        key, label, help_text = fields[selected - 1]
        print_wrapped(help_text, initial_indent="  ", subsequent_indent="  ", style=detail_text)
        value = prompt_value(label, getattr(profile, key))
        try_save_profile_updates(profile, [(key, value)])
    return profile


def language_menu(profile: Profile) -> None:
    while True:
        print()
        print("Language And Realtime Text")
        print("=" * 72)
        print(f"Current: {language_summary(profile.language)}")
        print(f"Realtime preview engine: {profile.realtime_preview_engine}")
        print(f"Realtime preview Python: {profile.realtime_preview_python or 'auto/default'}")
        print("-" * 72)
        for index, code in enumerate(KROKO_LANGUAGE_MENU_CODES, start=1):
            config = SUPPORTED_LANGUAGE_CONFIGS[code]
            marker = " *" if profile.language == code else ""
            print(f"{index}. {config.display_name:<12} {config.code:<3} Kroko {config.kroko_code}{marker}")
        print("c. Custom language code")
        print("o. Turn realtime text off")
        print("e. Realtime text engine")
        print("p. Realtime preview Python")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice in {"o", "off"}:
            try_save_profile_updates(profile, [("realtime_preview_engine", "off")])
            continue
        if choice in {"e", "engine"}:
            try_save_profile_updates(
                profile,
                [("realtime_preview_engine", prompt_value("Realtime preview engine", profile.realtime_preview_engine))],
            )
            continue
        if choice in {"p", "python"}:
            try_save_profile_updates(
                profile,
                [("realtime_preview_python", prompt_value("Realtime preview Python path", profile.realtime_preview_python))],
            )
            continue
        if choice in {"c", "custom"}:
            value = prompt_value("Language code or name", profile.language)
            try:
                normalized = normalize_language_code(value)
            except ValueError as exc:
                print(str(exc))
                continue
            updates: list[tuple[str, Any]] = [("language", normalized)]
            if not get_language_config(normalized).kroko_code and profile.realtime_preview_engine != "off":
                answer = read_input("This language has no Kroko live-text model. Turn realtime text off? [Y/n] ", "y").strip().lower()
                if answer not in {"n", "no"}:
                    updates.append(("realtime_preview_engine", "off"))
            try_save_profile_updates(profile, updates)
            continue
        try:
            selected = int(choice)
        except ValueError:
            print("Choose a language number, c, o, or b.")
            continue
        if not 1 <= selected <= len(KROKO_LANGUAGE_MENU_CODES):
            print("Choose one of the listed language numbers.")
            continue
        code = KROKO_LANGUAGE_MENU_CODES[selected - 1]
        updates = [("language", code)]
        if profile.realtime_preview_engine in {"", "off", "none", "false"}:
            answer = read_input("Enable Kroko realtime text for this language? [Y/n] ", "y").strip().lower()
            if answer not in {"n", "no"}:
                updates.append(("realtime_preview_engine", "kroko_onnx"))
        try_save_profile_updates(profile, updates)


def backend_menu(profile: Profile) -> None:
    while True:
        print()
        print("Backends And URLs")
        print("=" * 72)
        print(f"Mode: {profile.mode}")
        print(f"ASR: {profile.asr_backend}    {profile.remote_asr_url}")
        print(f"Embeddings: {profile.embeddings_backend}    {profile.remote_embeddings_url}")
        print(f"Embedding helper Python: {profile.embedding_python or 'auto/current'}")
        print("-" * 72)
        print("1. Full local ASR and embeddings")
        print("2. Controller with remote ASR and embeddings")
        print("3. Remote ASR URL")
        print("4. Remote embeddings URL")
        print("5. Custom ASR backend")
        print("6. Custom embeddings backend")
        print("7. Embedding helper Python")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            profile.mode = "local"
            profile.asr_backend = "local"
            profile.embeddings_backend = "local"
            profile.device = "auto"
            update_profile_in_place(profile, Profile.from_mapping(profile.as_dict()))
            save_profile(profile)
            print("Saved local backend profile.")
        elif choice == "2":
            profile.mode = "remote"
            profile.asr_backend = "remote"
            profile.embeddings_backend = "remote"
            profile.device = "auto"
            update_profile_in_place(profile, Profile.from_mapping(profile.as_dict()))
            save_profile(profile)
            print("Saved remote backend profile.")
        elif choice == "3":
            try_save_profile_updates(profile, [("remote_asr_url", prompt_value("Remote ASR URL", profile.remote_asr_url))])
        elif choice == "4":
            try_save_profile_updates(
                profile,
                [("remote_embeddings_url", prompt_value("Remote embeddings URL", profile.remote_embeddings_url))],
            )
        elif choice == "5":
            try_save_profile_updates(profile, [("asr_backend", prompt_value("ASR backend local/remote", profile.asr_backend))])
        elif choice == "6":
            try_save_profile_updates(
                profile,
                [("embeddings_backend", prompt_value("Embeddings backend local/remote", profile.embeddings_backend))],
            )
        elif choice == "7":
            try_save_profile_updates(
                profile,
                [("embedding_python", prompt_value("Embedding helper Python path", profile.embedding_python))],
            )
        else:
            print("Choose one of the listed options.")


def asr_runtime_menu(profile: Profile) -> None:
    while True:
        print()
        print("ASR Runtime")
        print("=" * 72)
        print(f"Model: {profile.model}")
        print(f"Device: {profile.device}")
        print(f"Compute type: {profile.compute_type}")
        print("-" * 72)
        print("1. ASR model")
        print("2. Device")
        print("3. Compute type")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            try_save_profile_updates(profile, [("model", prompt_value("ASR model", profile.model))])
        elif choice == "2":
            try_save_profile_updates(profile, [("device", prompt_value("Device auto/cuda/cpu", profile.device))])
        elif choice == "3":
            try_save_profile_updates(profile, [("compute_type", prompt_value("Compute type", profile.compute_type))])
        else:
            print("Choose one of the listed options.")


def browser_menu(profile: Profile) -> None:
    while True:
        print()
        print("Browser UI")
        print("=" * 72)
        print(f"Host: {profile.host}")
        print(f"Port: {profile.port}")
        print("-" * 72)
        print("1. Host")
        print("2. Port")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            try_save_profile_updates(profile, [("host", prompt_value("Host", profile.host))])
        elif choice == "2":
            try_save_profile_updates(profile, [("port", prompt_value("Port", profile.port))])
        else:
            print("Choose one of the listed options.")


def configuration_menu_text(profile: Profile) -> str:
    summary = "\n".join(f"  {line}" for line in profile_summary_lines(profile))
    return textwrap.dedent(
        f"""
        Configure WhoSpeaks
        ============================================================================
        {summary}
        ----------------------------------------------------------------------------
          1. Language and realtime text
          2. Speaker provider quality
          3. Backends and remote URLs
          4. ASR model, device, and compute
          5. Browser host and port
          6. Advanced launch arguments
          7. All saved profile fields
          b. Back
        """
    ).strip()


def configuration_menu(profile: Profile) -> int | None:
    while True:
        print()
        print(configuration_menu_text(profile))
        choice = read_input("> ", "b").strip().lower()
        if choice == "1":
            language_menu(profile)
        elif choice == "2":
            provider_preset_menu(profile)
        elif choice == "3":
            backend_menu(profile)
        elif choice == "4":
            asr_runtime_menu(profile)
        elif choice == "5":
            browser_menu(profile)
        elif choice == "6":
            try_save_profile_updates(profile, [("advanced_args", prompt_value("Advanced whospeaks-window args", profile.advanced_args))])
        elif choice == "7":
            updated = edit_profile(profile)
            update_profile_in_place(profile, Profile.from_mapping(updated.as_dict()))
            save_profile(profile)
        elif choice in {"b", "back", "q", "quit"}:
            return None
        else:
            print("Choose one of the listed options.")


def provider_preset_menu(profile: Profile) -> None:
    while True:
        selected_provider_preset(profile)
        print()
        print("Provider Presets")
        print("=" * 72)
        print_provider_summary(profile)
        print("-" * 72)
        for index, preset in enumerate(PROVIDER_PRESETS.values(), start=1):
            marker = " *" if profile.provider_preset == preset.id else ""
            print(f"{index}. {primary_text(preset.name)} ({preset.id}){marker}")
            print(f"   {label_text('Simple:')} {primary_text(preset.summary)}")
            print_wrapped(
                f"Deep: {preset.details}",
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
            if preset.score_note:
                print_wrapped(
                    f"Validation note: {preset.score_note}",
                    initial_indent="   ",
                    subsequent_indent="         ",
                    style=detail_text,
                )
            if preset.requirements:
                print_wrapped(
                    f"Requirement: {preset.requirements}",
                    initial_indent="   ",
                    subsequent_indent="         ",
                    style=detail_text,
                )
            print_wrapped(
                "Exact final: " + preset.embedding_provider,
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
            print_wrapped(
                "Exact live:  " + preset.live_speaker_embedding_provider,
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
        print("c. Custom provider strings")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice in {"c", "custom"}:
            profile.embedding_provider = prompt_value("Final embedding provider", profile.embedding_provider)
            profile.live_speaker_embedding_provider = prompt_value(
                "Live embedding provider",
                profile.live_speaker_embedding_provider,
            )
            profile.provider_preset = infer_provider_preset_id(
                "custom",
                profile.embedding_provider,
                profile.live_speaker_embedding_provider,
            )
            save_path = save_profile(profile)
            print(f"Saved provider preset {profile.provider_preset} to {save_path}.")
            continue
        try:
            selected = int(choice)
        except ValueError:
            print("Choose a preset number, c, or b.")
            continue
        presets = list(PROVIDER_PRESETS.values())
        if not 1 <= selected <= len(presets):
            print("Choose one of the listed preset numbers.")
            continue
        preset = presets[selected - 1]
        apply_provider_preset(profile, preset.id)
        save_path = save_profile(profile)
        print(f"Saved provider preset {preset.name} ({preset.id}) to {save_path}.")


def select_profile_interactively(profile: Profile, mode: str) -> int | None:
    configure_profile_for_mode(profile, mode)
    save_path = save_profile(profile)
    label = {
        "local": "full local installation",
        "remote": "controller + remote GPU services",
        "server": "GPU server",
    }.get(profile.mode, profile.mode)
    print(f"Selected {label} profile and saved it to {save_path}.")
    report = run_doctor(profile)
    extra = recommended_install_extra(profile, report)
    if extra is None:
        print("No Python package install action is missing for this profile.")
        return None
    print(f"Next installer action: {format_command(build_install_command(extra))}")
    answer = read_input("Install the required Python packages now? [y/N] ", "n").strip().lower()
    if answer in {"y", "yes"}:
        return install_extra_and_maybe_kroko(profile, extra, assume_yes=True, kroko_assume_yes=False)
    print("Install skipped. Choose the install action later to run it.")
    return None


def install_missing_group_interactively(profile: Profile, report: DoctorReport | None = None) -> int | None:
    current_report = report or run_doctor(profile)
    extra = recommended_install_extra(profile, current_report)
    if extra is None:
        print("No Python package install action is missing for this profile.")
        return None
    return install_extra_and_maybe_kroko(profile, extra)


def install_components_interactively(profile: Profile) -> int | None:
    target = prompt_install_target()
    preview_engine, preview_preset = prompt_realtime_preview(target)
    plan = install_plan_for_target(
        target,
        realtime_preview_engine=preview_engine,
        realtime_preview_model_preset=preview_preset,
    )
    configure_profile_for_install(profile, plan)
    validate_realtime_preview_language(profile)
    save_path = save_profile(profile)
    print(f"Saved {profile.mode} profile to {save_path}")
    print_install_plan(plan, profile)
    if not confirm_install_start(False, False):
        print("Install skipped.")
        return None
    return install_extra_and_maybe_kroko(
        profile,
        plan.extra,
        assume_yes=True,
        install_kroko=plan.install_kroko,
        kroko_assume_yes=True if plan.install_kroko else False,
    )


def advanced_setup_menu(profile: Profile) -> int | None:
    while True:
        print(textwrap.dedent(
            """
            Advanced setup
              1. Controller + remote GPU services profile
              2. This machine as a GPU server profile
              3. Edit profile settings
              4. Print exact launch command
              b. Back
            """
        ).strip())
        choice = read_input("> ", "b").strip().lower()
        if choice == "1":
            code = select_profile_interactively(profile, "remote")
            if code:
                return code
        elif choice == "2":
            code = select_profile_interactively(profile, "server")
            if code:
                return code
        elif choice == "3":
            edit_profile(profile)
            save_profile(profile)
        elif choice == "4":
            print_launch_command(profile)
        elif choice in {"b", "back", "q", "quit"}:
            return None


def main_menu_text() -> str:
    return textwrap.dedent(
        """
        Actions
          1. Install or repair WhoSpeaks
          2. Launch browser UI
          3. Doctor / complete diagnostics
          4. Language and realtime text
          5. Speaker provider quality
          6. Backends and remote URLs
          7. ASR model, device, and compute
          8. Browser host and port
          9. All configuration fields
          p. Print exact launch command
          r. Remote/server profiles
          q. Quit
        """
    ).strip()


def launch_profile(profile: Profile) -> int:
    if profile.mode == "server":
        print("Start each server command in its own shell:")
        for line in build_server_launch_lines():
            print(f"  {line}")
        return 0
    command = build_launch_command(profile)
    print(format_command(command))
    return int(subprocess.run(command, check=False).returncode)


def interactive_dashboard(profile: Profile) -> int:
    while True:
        report = run_doctor(profile)
        render_dashboard(profile, report)
        extra = recommended_install_extra(profile, report)
        if extra:
            print(f"Recommended package action: {format_command(build_install_command(extra))}")
        print(main_menu_text())
        choice = read_input("> ", "q").strip().lower()
        if choice in {"1", "i", "install", "s", "setup"}:
            code = install_components_interactively(profile)
            if code:
                return code
        elif choice == "2":
            return launch_profile(profile)
        elif choice == "3":
            print_report(run_doctor(profile, deep=True))
        elif choice == "4":
            language_menu(profile)
        elif choice == "5":
            provider_preset_menu(profile)
        elif choice == "6":
            backend_menu(profile)
        elif choice == "7":
            asr_runtime_menu(profile)
        elif choice == "8":
            browser_menu(profile)
        elif choice == "9":
            configuration_menu(profile)
        elif choice in {"p", "print"}:
            print_launch_command(profile)
        elif choice in {"r", "remote", "server"}:
            code = advanced_setup_menu(profile)
            if code:
                return code
        elif choice in {"q", "quit", "exit"}:
            return 0


def run_textual_dashboard(profile: Profile) -> int:
    try:
        from .tui import run_setup_app
    except ImportError as exc:
        if not str(exc.name or "").startswith("textual"):
            raise
        print("Textual is unavailable; opening the classic terminal interface.")
        return interactive_dashboard(profile)

    result = run_setup_app(profile)
    if result == "launch":
        return launch_profile(load_profile())
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    if args.with_kroko and args.without_kroko:
        raise SystemExit("Choose either --with-kroko or --without-kroko, not both.")

    if args.target:
        target = normalize_install_target(args.target)
    elif sys.stdin.isatty():
        target = prompt_install_target()
    else:
        target = "local"
        print("No install target was supplied; defaulting to full local installation.")

    requested_engine = str(getattr(args, "realtime_preview_engine", "") or "").strip()
    requested_preset = str(getattr(args, "realtime_preview_model_preset", "") or "").strip()
    requested_model_dir = getattr(args, "realtime_preview_model_dir", None)
    if requested_engine and (args.with_kroko or args.without_kroko):
        raise SystemExit("Choose --realtime-preview-engine or the legacy Kroko switches, not both.")
    if requested_preset and not requested_engine:
        raise SystemExit("--realtime-preview-model-preset requires --realtime-preview-engine.")
    if requested_model_dir is not None and requested_engine not in {"sherpa_onnx", "sherpa-onnx", "nemotron", "sherpa"}:
        raise SystemExit("--realtime-preview-model-dir is only valid with --realtime-preview-engine sherpa_onnx.")

    if target == "server":
        preview_engine, preview_preset = "off", ""
        if requested_engine or args.with_kroko:
            print("Realtime preview text is not installed on server-only targets.")
    elif requested_engine:
        preview_engine, preview_preset = requested_engine, requested_preset
    elif args.with_kroko:
        preview_engine, preview_preset = "kroko_onnx", "community-64l"
    elif args.without_kroko:
        preview_engine, preview_preset = "off", ""
    elif sys.stdin.isatty() and not args.yes:
        preview_engine, preview_preset = prompt_realtime_preview(target)
    else:
        preview_engine, preview_preset = "off", ""
        print("Realtime preview text is not selected. Pass --realtime-preview-engine sherpa_onnx to include Nemotron.")

    plan = install_plan_for_target(
        target,
        realtime_preview_engine=preview_engine,
        realtime_preview_model_preset=preview_preset,
    )
    profile = load_profile()
    configure_profile_for_install(profile, plan)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.provider_preset:
        apply_provider_preset(profile, args.provider_preset)
    if requested_model_dir is not None:
        profile.realtime_preview_model_dir = str(requested_model_dir)
    validate_realtime_preview_language(profile)

    if args.dry_run:
        print(f"Dry run: would save {profile.mode} profile to {config_path()}")
    else:
        save_path = save_profile(profile)
        print(f"Saved {profile.mode} profile to {save_path}")

    print_install_plan(plan, profile)
    if not confirm_install_start(args.yes, args.dry_run):
        print("Install skipped.")
        return 0

    code = install_extra_and_maybe_kroko(
        profile,
        plan.extra,
        assume_yes=True,
        dry_run=args.dry_run,
        install_kroko=plan.install_kroko,
        kroko_assume_yes=True if plan.install_kroko else False,
        torch_policy=getattr(args, "torch", None),
    )
    if code:
        return code

    report = run_doctor(profile, profile.mode, deep=args.deep)
    print_report(report)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    profile = load_profile()
    if args.mode:
        configure_profile_for_mode(profile, args.mode)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.remote_asr_url:
        profile.remote_asr_url = args.remote_asr_url
    if args.remote_embeddings_url:
        profile.remote_embeddings_url = args.remote_embeddings_url
    if args.port is not None:
        profile.port = args.port
    report = run_doctor(profile, args.mode or "auto", deep=args.deep)
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        print_report(report)
    if args.fix:
        extra = recommended_install_extra(profile, report)
        if extra:
            code = install_extra(
                extra,
                assume_yes=args.yes,
                dry_run=args.dry_run,
                torch_policy=getattr(args, "torch", None),
            )
            if code:
                return code
        else:
            print("No Python package install action was recommended for the current failures.")
    return 1 if args.strict and report.has_failures else 0


def cmd_setup(args: argparse.Namespace) -> int:
    profile = load_profile()
    configure_profile_for_mode(profile, args.mode)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.provider_preset:
        apply_provider_preset(profile, args.provider_preset)
    if getattr(args, "realtime_preview_engine", ""):
        profile.realtime_preview_engine = normalize_preview_engine(args.realtime_preview_engine)
        if profile.realtime_preview_engine in {"kroko_onnx", "sherpa_onnx"}:
            default_preset = get_preview_backend_spec(profile.realtime_preview_engine).default_preset or ""
            try:
                profile.realtime_preview_model_preset = normalize_preview_model_preset(
                    profile.realtime_preview_engine,
                    profile.realtime_preview_model_preset or default_preset,
                )
            except (ValueError, argparse.ArgumentTypeError):
                profile.realtime_preview_model_preset = default_preset
        else:
            profile.realtime_preview_model_preset = ""
        if profile.realtime_preview_engine != "sherpa_onnx":
            profile.realtime_preview_model_dir = ""
    if getattr(args, "realtime_preview_model_preset", ""):
        profile.realtime_preview_model_preset = normalize_preview_model_preset(
            profile.realtime_preview_engine,
            args.realtime_preview_model_preset,
        )
    if getattr(args, "realtime_preview_model_dir", None) is not None:
        if profile.realtime_preview_engine != "sherpa_onnx":
            raise SystemExit("--realtime-preview-model-dir is only valid with sherpa_onnx.")
        profile.realtime_preview_model_dir = str(args.realtime_preview_model_dir)
    validate_realtime_preview_language(profile)
    if args.dry_run:
        save_path = config_path()
        print(f"Dry run: would save {profile.mode} profile to {save_path}")
    else:
        save_path = save_profile(profile)
        print(f"Saved {profile.mode} profile to {save_path}")
    report = run_doctor(profile, profile.mode, deep=args.deep)
    print_report(report)
    if args.install:
        extra = recommended_install_extra(profile, report)
        if profile.mode == "local":
            extra = install_plan_for_target(
                "local",
                realtime_preview_engine=profile.realtime_preview_engine,
                realtime_preview_model_preset=profile.realtime_preview_model_preset,
            ).extra
        if extra is None and profile.mode == "server":
            extra = "server"
        if extra is None and profile.mode == "remote":
            extra = "controller"
        if extra is not None:
            return install_extra_and_maybe_kroko(
                profile,
                extra,
                assume_yes=args.yes,
                dry_run=args.dry_run,
                install_kroko=preview_engine_uses_kroko(profile) and not args.skip_kroko,
                torch_policy=getattr(args, "torch", None),
            )
    print("Launch command:")
    print(f"  {format_command(build_launch_command(profile))}")
    return 0


def cmd_install_kroko(args: argparse.Namespace) -> int:
    profile = load_profile()
    if args.python:
        profile.realtime_preview_python = args.python
    if args.engine:
        profile.realtime_preview_engine = args.engine
    return install_kroko_runtime(
        profile,
        assume_yes=args.yes,
        dry_run=args.dry_run,
        variant=args.variant,
        work_dir=args.work_dir,
        soft_fail=False,
    )


def cmd_launch(args: argparse.Namespace) -> int:
    profile = load_profile()
    updates: list[tuple[str, Any]] = []
    if args.language:
        updates.append(("language", args.language))
    if args.port is not None:
        updates.append(("port", args.port))
    if updates:
        profile = apply_profile_updates(profile, updates)
    if args.provider_preset:
        apply_provider_preset(profile, args.provider_preset)
    if profile.mode == "server":
        print("Server profile service commands:")
        for line in build_server_launch_lines():
            print(line)
        if not args.print_only and not args.dry_run:
            print("Start each command in a separate shell so both services stay running.")
        return 0
    command = build_launch_command(profile, args.extra_args or "")
    print(format_command(command))
    if args.print_only or args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def cmd_config(args: argparse.Namespace) -> int:
    profile = Profile() if args.reset else load_profile()
    updates: list[tuple[str, Any]] = []
    direct_fields = (
        "mode",
        "host",
        "port",
        "language",
        "model",
        "device",
        "compute_type",
        "asr_backend",
        "embeddings_backend",
        "provider_preset",
        "remote_asr_url",
        "remote_embeddings_url",
        "embedding_provider",
        "live_speaker_embedding_provider",
        "embedding_python",
        "vad_backend",
        "realtime_preview_engine",
        "realtime_preview_model_preset",
        "realtime_preview_model_dir",
        "realtime_preview_python",
        "advanced_args",
    )
    for field_name in direct_fields:
        value = getattr(args, field_name, None)
        if value is not None and value != "":
            updates.append((field_name, value))
    if args.set:
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"Invalid --set value {item!r}; use name=value.")
            key, value = item.split("=", 1)
            updates.append((key, value))
    if updates:
        profile = apply_profile_updates(profile, updates)
    if args.edit:
        profile = edit_profile(profile)
        profile = Profile.from_mapping(profile.as_dict())
    if args.reset or updates or args.edit:
        save_profile(profile)
    if args.json:
        print(json.dumps(profile.as_dict(), indent=2, sort_keys=True))
    else:
        print_profile(profile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whospeaks",
        description="WhoSpeaks setup, doctor, and launcher.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print the dashboard once and exit instead of opening the interactive starter CLI.",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Open the classic numbered interface instead of the Textual setup application.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Run setup and component checks.")
    doctor.add_argument("--mode", choices=("auto", "local", "remote", "server"), default="auto")
    doctor.add_argument("--language", default="", help="Temporarily check a language profile without saving it.")
    doctor.add_argument("--remote-asr-url", default="")
    doctor.add_argument("--remote-embeddings-url", default="")
    doctor.add_argument("--port", type=int, default=None)
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks such as remote /load.")
    doctor.add_argument("--strict", action="store_true", help="Return non-zero when required checks fail.")
    doctor.add_argument("--fix", action="store_true", help="Offer the recommended pip install action after checks.")
    doctor.add_argument("--yes", action="store_true", help="Do not prompt before running the pip install action.")
    doctor.add_argument("--dry-run", action="store_true", help="Print installer commands without running them.")
    doctor.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy for --fix: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    doctor.set_defaults(func=cmd_doctor)

    install = subparsers.add_parser(
        "install",
        help="Guided installer for full local, core/controller, or server packages.",
    )
    install.add_argument(
        "--target",
        choices=INSTALL_TARGET_CHOICES,
        default="",
        help="Install target: local, core, or server. Omit for an interactive choice.",
    )
    install.add_argument("--language", default="", help="Save the setup profile with this language code, for example de.")
    install.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    kroko_group = install.add_mutually_exclusive_group()
    kroko_group.add_argument(
        "--with-kroko",
        action="store_true",
        help="Enable realtime preview text and run the native Kroko setup after Python packages.",
    )
    kroko_group.add_argument(
        "--without-kroko",
        action="store_true",
        help="Disable realtime preview text for this install.",
    )
    install.add_argument(
        "--realtime-preview-engine",
        default="",
        help="Realtime text engine: sherpa_onnx (Nemotron), kroko_onnx, or off.",
    )
    install.add_argument(
        "--realtime-preview-model-preset",
        default="",
        help="Realtime model preset, for example nemotron-3.5-560ms-int8.",
    )
    install.add_argument(
        "--realtime-preview-model-dir",
        type=Path,
        default=None,
        help="Optional directory containing an unpacked Nemotron model.",
    )
    install.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks after installation.")
    install.add_argument("--yes", action="store_true", help="Do not prompt before running installer actions.")
    install.add_argument("--dry-run", action="store_true", help="Print installer actions without running them.")
    install.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    install.set_defaults(func=cmd_install)

    setup = subparsers.add_parser("setup", help="Choose a setup mode and optionally install dependencies.")
    setup.add_argument("--mode", choices=("local", "remote", "server"), default="local")
    setup.add_argument("--language", default="", help="Save the setup profile with this language code, for example de.")
    setup.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    setup.add_argument("--install", action="store_true", help="Run the recommended package installer for this mode.")
    setup.add_argument(
        "--skip-kroko",
        action="store_true",
        help="Do not offer/build the native Kroko realtime preview runtime after installing extras.",
    )
    setup.add_argument(
        "--realtime-preview-engine",
        default="",
        help="Realtime text engine: sherpa_onnx (Nemotron), kroko_onnx, or off.",
    )
    setup.add_argument(
        "--realtime-preview-model-preset",
        default="",
        help="Realtime model preset, for example nemotron-3.5-560ms-int8.",
    )
    setup.add_argument(
        "--realtime-preview-model-dir",
        type=Path,
        default=None,
        help="Optional directory containing an unpacked Nemotron model.",
    )
    setup.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks during setup.")
    setup.add_argument("--yes", action="store_true", help="Do not prompt before running installer actions.")
    setup.add_argument("--dry-run", action="store_true", help="Print installer actions without running them.")
    setup.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    setup.set_defaults(func=cmd_setup)

    install_kroko = subparsers.add_parser(
        "install-kroko",
        help="Build/install the native Kroko realtime preview runtime.",
    )
    install_kroko.add_argument("--python", default="", help="Python executable that should receive kroko_onnx.")
    install_kroko.add_argument(
        "--engine",
        default="kroko_onnx",
        help="Realtime preview engine to enable while installing. Use off to skip.",
    )
    install_kroko.add_argument("--variant", choices=("free", "pro"), default="free")
    install_kroko.add_argument("--work-dir", type=Path, default=None)
    install_kroko.add_argument("--yes", action="store_true", help="Do not prompt before building/installing Kroko.")
    install_kroko.add_argument("--dry-run", action="store_true", help="Print Kroko installer commands without running them.")
    install_kroko.set_defaults(func=cmd_install_kroko)

    launch = subparsers.add_parser("launch", help="Print or run the current whospeaks-window launch command.")
    launch.add_argument("--print", dest="print_only", action="store_true", help="Print the launch command and exit.")
    launch.add_argument("--dry-run", action="store_true", help="Alias for --print.")
    launch.add_argument("--language", default="", help="Temporarily override the saved language for this launch.")
    launch.add_argument("--port", type=int, default=None, help="Temporarily override the saved browser UI port.")
    launch.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    launch.add_argument("--extra-args", default="", help="Additional whospeaks-window arguments appended to the profile.")
    launch.set_defaults(func=cmd_launch)

    config = subparsers.add_parser("config", help="Show or update the saved starter profile.")
    config.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Set any saved profile field.")
    config.add_argument("--reset", action="store_true", help="Reset the saved profile before applying other changes.")
    config.add_argument("--edit", action="store_true", help="Open the interactive full profile editor.")
    config.add_argument("--json", action="store_true", help="Print the profile as JSON.")
    config.add_argument("--mode", choices=("local", "remote", "server"), default=None)
    config.add_argument("--host", default=None)
    config.add_argument("--port", type=int, default=None)
    config.add_argument("--language", default=None, help="Set the saved language code, for example de.")
    config.add_argument("--model", default=None)
    config.add_argument("--device", default=None)
    config.add_argument("--compute-type", dest="compute_type", default=None)
    config.add_argument("--asr-backend", dest="asr_backend", choices=("local", "remote"), default=None)
    config.add_argument("--embeddings-backend", dest="embeddings_backend", choices=("local", "remote"), default=None)
    config.add_argument("--provider-preset", dest="provider_preset", choices=PROVIDER_PRESET_CHOICES, default=None)
    config.add_argument("--remote-asr-url", dest="remote_asr_url", default=None)
    config.add_argument("--remote-embeddings-url", dest="remote_embeddings_url", default=None)
    config.add_argument("--embedding-provider", dest="embedding_provider", default=None)
    config.add_argument("--live-speaker-embedding-provider", dest="live_speaker_embedding_provider", default=None)
    config.add_argument("--embedding-python", dest="embedding_python", default=None)
    config.add_argument("--vad-backend", dest="vad_backend", default=None)
    config.add_argument("--realtime-preview-engine", dest="realtime_preview_engine", default=None)
    config.add_argument("--realtime-preview-model-preset", dest="realtime_preview_model_preset", default=None)
    config.add_argument("--realtime-preview-model-dir", dest="realtime_preview_model_dir", type=Path, default=None)
    config.add_argument("--realtime-preview-python", dest="realtime_preview_python", default=None)
    config.add_argument("--advanced-args", dest="advanced_args", default=None)
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return int(args.func(args))
    profile = load_profile()
    if args.no_interactive or not sys.stdin.isatty():
        report = run_doctor(profile)
        render_dashboard(profile, report)
        print("Run `whospeaks` in an interactive terminal to open the setup application.")
        print("For automation, run `whospeaks install --target local --without-kroko --yes`.")
        print("Run `whospeaks launch --print` to see the exact browser command.")
        return 0
    if args.classic:
        return interactive_dashboard(profile)
    return run_textual_dashboard(profile)
