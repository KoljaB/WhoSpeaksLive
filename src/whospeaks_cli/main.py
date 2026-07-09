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
PACKAGE_NAME = "whospeaks"
KROKO_LANGUAGE_MENU_CODES = ("en", "de", "es", "fr", "it", "nl", "pt", "sv", "tr", "he")
EDITABLE_PROFILE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("mode", "Profile mode", "local, remote, or server. Mode also aligns the ASR and embeddings backends."),
    ("language", "Language", "Shared by final ASR, realtime preview model selection, and sentence splitting."),
    ("provider_preset", "Provider preset", "Named final/live speaker embedding stack, or custom."),
    ("embedding_provider", "Final provider", "Exact provider string used for committed speaker assignment."),
    ("live_speaker_embedding_provider", "Live provider", "Exact provider string used for live speaker feedback."),
    ("embedding_python", "Embedding helper Python", "Optional Python executable for local speaker-embedding helper subprocesses."),
    ("realtime_preview_engine", "Realtime text engine", "Use kroko_onnx for live text, or off to disable preview text."),
    ("realtime_preview_python", "Realtime preview Python", "Optional Python executable for the Kroko/RealtimeSTT worker."),
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
    realtime_preview_engine: str = "kroko_onnx"
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
        "Run the setup installer from this CLI or install the matching whospeaks extra.",
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
            "Install the local or complete dependency set before local GPU checks.",
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

    preview_engine = str(profile.realtime_preview_engine or "off").strip().lower().replace("-", "_")
    if preview_engine in {"off", "mock"}:
        checks.append(CheckResult("Realtime preview", "skip", f"Preview engine is {preview_engine}."))
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


def build_install_command(extra: str = LOCAL_EXTRA) -> list[str]:
    extra = str(extra or LOCAL_EXTRA).strip()
    return [sys.executable, "-m", "pip", "install", f"{PACKAGE_NAME}[{extra}]"]


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        rendered: list[str] = []
        for arg in command:
            if "[" in arg or "]" in arg:
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


def install_extra(extra: str, assume_yes: bool = False, dry_run: bool = False) -> int:
    command = build_install_command(extra)
    print("Install command:")
    print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Run this command now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Install skipped.")
            return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def configure_profile_for_mode(profile: Profile, mode: str) -> Profile:
    selected = normalize_mode(mode)
    profile.mode = selected
    if selected == "local":
        profile.asr_backend = "local"
        profile.embeddings_backend = "local"
        profile.device = "auto"
        apply_provider_preset(profile, "smoke")
        profile.vad_backend = "rms"
        profile.realtime_preview_engine = "kroko_onnx"
    elif selected == "remote":
        profile.asr_backend = "remote"
        profile.embeddings_backend = "remote"
        profile.device = "auto"
        apply_provider_preset(profile, "smoke")
        profile.vad_backend = "rms"
        profile.realtime_preview_engine = "off"
    elif selected == "server":
        profile.asr_backend = "remote"
        profile.embeddings_backend = "remote"
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
    preview_engine = str(profile.realtime_preview_engine or "off").strip().lower().replace("-", "_")
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
    preview = f"Kroko {config.kroko_code}" if config.kroko_code else "no Kroko preview"
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
        f"Realtime text: {profile.realtime_preview_engine}",
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
        print("No pip-installable dependency group is missing for this profile.")
        return None
    print(f"Next installer action: {format_command(build_install_command(extra))}")
    answer = read_input("Install that dependency group now? [y/N] ", "n").strip().lower()
    if answer in {"y", "yes"}:
        return install_extra(extra, assume_yes=True)
    print("Install skipped. Choose action 2 later to run it.")
    return None


def install_missing_group_interactively(profile: Profile, report: DoctorReport | None = None) -> int | None:
    current_report = report or run_doctor(profile)
    extra = recommended_install_extra(profile, current_report)
    if extra is None:
        print("No pip-installable dependency group is missing for this profile.")
        return None
    return install_extra(extra)


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
          1. Launch browser UI
          2. Doctor / complete diagnostics
          3. Install recommended dependency group
          4. Language and realtime text
          5. Speaker provider quality
          6. Backends and remote URLs
          7. ASR model, device, and compute
          8. Browser host and port
          9. All configuration fields
          p. Print exact launch command
          s. First-time full local setup
          r. Remote/server profiles
          q. Quit
        """
    ).strip()


def interactive_dashboard(profile: Profile) -> int:
    while True:
        report = run_doctor(profile)
        render_dashboard(profile, report)
        extra = recommended_install_extra(profile, report)
        if extra:
            print(f"Recommended installer action: {format_command(build_install_command(extra))}")
        print(main_menu_text())
        choice = read_input("> ", "q").strip().lower()
        if choice == "1":
            if profile.mode == "server":
                print("Start each server command in its own shell:")
                for line in build_server_launch_lines():
                    print(f"  {line}")
                return 0
            command = build_launch_command(profile)
            print(format_command(command))
            return subprocess.run(command, check=False).returncode
        elif choice == "2":
            print_report(run_doctor(profile, deep=True))
        elif choice == "3":
            code = install_missing_group_interactively(profile, report)
            if code:
                return code
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
        elif choice in {"s", "setup"}:
            code = select_profile_interactively(profile, "local")
            if code:
                return code
        elif choice in {"r", "remote", "server"}:
            code = advanced_setup_menu(profile)
            if code:
                return code
        elif choice in {"q", "quit", "exit"}:
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
            code = install_extra(extra, assume_yes=args.yes, dry_run=args.dry_run)
            if code:
                return code
        else:
            print("No pip-installable dependency group was recommended for the current failures.")
    return 1 if args.strict and report.has_failures else 0


def cmd_setup(args: argparse.Namespace) -> int:
    profile = load_profile()
    configure_profile_for_mode(profile, args.mode)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.provider_preset:
        apply_provider_preset(profile, args.provider_preset)
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
        if extra is None and profile.mode == "local":
            extra = LOCAL_EXTRA
        if extra is None and profile.mode == "server":
            extra = "server"
        if extra is None and profile.mode == "remote":
            extra = "controller"
        if extra is not None:
            return install_extra(extra, assume_yes=args.yes, dry_run=args.dry_run)
    print("Launch command:")
    print(f"  {format_command(build_launch_command(profile))}")
    return 0


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
    doctor.set_defaults(func=cmd_doctor)

    setup = subparsers.add_parser("setup", help="Choose a setup mode and optionally install dependencies.")
    setup.add_argument("--mode", choices=("local", "remote", "server"), default="local")
    setup.add_argument("--language", default="", help="Save the setup profile with this language code, for example de.")
    setup.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    setup.add_argument("--install", action="store_true", help="Run the recommended pip extra installer.")
    setup.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks during setup.")
    setup.add_argument("--yes", action="store_true", help="Do not prompt before running the pip install action.")
    setup.add_argument("--dry-run", action="store_true", help="Print installer commands without running them.")
    setup.set_defaults(func=cmd_setup)

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
        print("Run `whospeaks setup --mode local --install` for a full local setup.")
        print("Run `whospeaks launch --print` to see the exact browser command.")
        return 0
    return interactive_dashboard(profile)
