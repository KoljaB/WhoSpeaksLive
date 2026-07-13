"""System diagnostics and reports for the WhoSpeaks CLI."""

from __future__ import annotations

import dataclasses
from contextlib import closing
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from paths import RUNTIME_DIR

from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
)
from window.sherpa_onnx_models import (
    default_sherpa_onnx_model_dir,
    missing_sherpa_onnx_model_files,
)

from . import __version__
from .cli_console import print_wrapped
from .profiles import Profile, SMOKE_PROVIDER, normalize_mode
from .runtime_constants import PACKAGE_NAME, STATUS_LABEL, STATUS_ORDER


def _facade_callable(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("whospeaks_cli.main")
    return getattr(facade, name, fallback) if facade is not None else fallback


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
class TorchInstallSelection:
    mode: str
    index_url: str
    reason: str
    build: str = ""

    @property
    def should_install(self) -> bool:
        return self.mode in {"cuda", "cpu"}


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


def check_sherpa_onnx_runtime() -> CheckResult:
    """Verify that this WhoSpeaks Python has the complete Nemotron runtime API."""

    try:
        sherpa_onnx = importlib.import_module("sherpa_onnx")
    except Exception as exc:
        return CheckResult(
            "Nemotron sherpa-onnx runtime",
            "warn",
            f"{sys.executable} cannot import sherpa_onnx: {type(exc).__name__}: {exc}",
            "Install sherpa-onnx and sherpa-onnx-bin into the current WhoSpeaks Python environment.",
        )
    if getattr(sherpa_onnx, "OnlineRecognizer", None) is None:
        return CheckResult(
            "Nemotron sherpa-onnx runtime",
            "warn",
            (
                f"{sys.executable} can import sherpa_onnx, but "
                "sherpa_onnx.OnlineRecognizer is missing."
            ),
            (
                "Replace the incomplete or Kroko-specific sherpa-onnx build in the current WhoSpeaks "
                "Python environment with sherpa-onnx and sherpa-onnx-bin."
            ),
        )
    return CheckResult(
        "Nemotron sherpa-onnx runtime",
        "ok",
        f"{sys.executable} provides sherpa_onnx.OnlineRecognizer.",
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


def check_meeting_intelligence_entrypoint(required: bool) -> CheckResult:
    try:
        module = importlib.import_module("window.meeting_intelligence_server")
    except Exception as exc:
        return CheckResult(
            "Meeting Intelligence modules",
            "fail" if required else "warn",
            f"Cannot import the service: {type(exc).__name__}: {exc}",
            "Install `whospeaks[intelligence]` in this Python environment.",
        )
    if not callable(getattr(module, "main", None)):
        return CheckResult(
            "Meeting Intelligence entry point",
            "fail" if required else "warn",
            "window.meeting_intelligence_server.main is missing.",
            "Reinstall the WhoSpeaks package.",
        )
    executable = shutil.which("whospeaks-meeting-intelligence")
    detail = executable or "packaged module entry point is callable"
    return CheckResult("Meeting Intelligence entry point", "ok", detail)


def check_meeting_intelligence_llm(profile: Profile, *, deep: bool) -> CheckResult:
    defaults = {
        "llama_cpp": ("http://127.0.0.1:8081/v1", "local", ""),
        "ollama": ("http://127.0.0.1:11434/v1", "gemma3", ""),
        "lm_studio": ("http://127.0.0.1:1234/v1", "local-model", ""),
        "openai_compatible": ("http://127.0.0.1:8000/v1", "local-model", ""),
        "openai": ("https://api.openai.com/v1", "gpt-5.6-luna", "OPENAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "google/gemma-3-12b-it", "OPENROUTER_API_KEY"),
    }
    base_url, model, key_env = defaults.get(profile.report_llm_provider, defaults["llama_cpp"])
    base_url = str(profile.report_llm_base_url or base_url).rstrip("/")
    model = str(profile.report_llm_model or model)
    if key_env and not (os.getenv(key_env) or os.getenv("WHOSPEAKS_MI_LLM_API_KEY")):
        return CheckResult(
            "Meeting Intelligence LLM",
            "fail",
            f"{profile.report_llm_provider}:{model} requires {key_env}.",
            f"Set {key_env} in the Meeting Intelligence environment or .env file.",
        )
    if not deep:
        return CheckResult("Meeting Intelligence LLM", "ok", f"Configured as {profile.report_llm_provider}:{model} at {base_url}.")
    headers = {"Accept": "application/json"}
    key = os.getenv(key_env, "") if key_env else os.getenv("WHOSPEAKS_MI_LLM_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with urlopen(Request(f"{base_url}/models", headers=headers), timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return CheckResult(
            "Meeting Intelligence LLM",
            "fail",
            f"{base_url}/models failed: {type(exc).__name__}: {exc}",
            "Start the configured LLM server or correct its base URL and credentials.",
        )
    models = payload.get("data") if isinstance(payload, dict) else None
    return CheckResult("Meeting Intelligence LLM", "ok", f"Endpoint is ready; configured model is {model} ({len(models or [])} listed).")


def check_text_embedding_provider(profile: Profile, *, deep: bool) -> CheckResult:
    base_url = str(profile.text_embedding_base_url or "").strip().rstrip("/")
    model = str(profile.text_embedding_model or "").strip()
    key_env = str(profile.text_embedding_api_key_env or "").strip()
    if not base_url or not model:
        return CheckResult(
            "Text embedding endpoint",
            "warn",
            "Not configured; short single-session chat works, but long and cross-session chat do not.",
            "Set the text embedding URL and model in the Meeting Intelligence tab.",
        )
    if key_env and not os.getenv(key_env):
        return CheckResult(
            "Text embedding endpoint",
            "fail",
            f"Required API-key environment variable {key_env} is missing.",
            f"Set {key_env} before launching Meeting Intelligence.",
        )
    if not deep:
        return CheckResult("Text embedding endpoint", "ok", f"Configured model {model} at {base_url}.")
    url = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key_env:
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    request = Request(
        url,
        data=json.dumps({"model": model, "input": ["WhoSpeaks embedding diagnostic"]}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=15.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vector = payload["data"][0]["embedding"]
        dimensions = len(vector)
        if dimensions < 1:
            raise ValueError("empty embedding vector")
    except Exception as exc:
        return CheckResult(
            "Text embedding endpoint",
            "fail",
            f"Embedding probe failed: {type(exc).__name__}: {exc}",
            "Verify that the endpoint implements the OpenAI-compatible /embeddings API and supports the configured model.",
        )
    return CheckResult("Text embedding endpoint", "ok", f"Endpoint is ready and returned {dimensions}-dimensional vectors.")


def check_meeting_index_writable() -> CheckResult:
    directory = RUNTIME_DIR
    temporary = ""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".meeting-index-doctor-", suffix=".sqlite3", dir=directory)
        os.close(handle)
        with closing(sqlite3.connect(temporary)) as connection:
            with connection:
                connection.execute("CREATE TABLE writable_probe (value INTEGER)")
                connection.execute("INSERT INTO writable_probe VALUES (1)")
    except Exception as exc:
        return CheckResult(
            "Meeting index SQLite",
            "fail",
            f"{directory} is not writable: {type(exc).__name__}: {exc}",
            "Choose a writable WHOSPEAKS_RUNTIME_DIR or fix permissions on the runtime directory.",
        )
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
    return CheckResult("Meeting index SQLite", "ok", f"SQLite writes succeed in {directory}.")


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

    if profile.reports_enabled:
        checks.append(check_meeting_intelligence_entrypoint(required=True))
        checks.append(check_meeting_intelligence_llm(profile, deep=deep))
        checks.append(check_text_embedding_provider(profile, deep=deep))
        checks.append(check_meeting_index_writable())
        checks.append(check_port(profile.host, profile.reports_port))
    else:
        checks.append(CheckResult("Meeting Intelligence", "skip", "Reports + Ask is disabled in this profile."))

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
        checks.append(_facade_callable("check_sherpa_onnx_runtime", check_sherpa_onnx_runtime)())
    else:
        checks.append(check_import_group(
            "Realtime preview",
            [("RealtimeSTT", "RealtimeSTT")],
            required=False,
        ))
        if profile.realtime_preview_python:
            checks.append(_facade_callable("check_python_imports", check_python_imports)(
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
