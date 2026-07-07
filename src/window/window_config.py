"""Configuration, defaults, and speaker-library helpers for the window diarization GUI."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from paths import (
    CACHE_DIR,
    CUNK_CANONICAL,
    KROKO_MODEL_DIR,
    PROJECT_ROOT,
    SPEAKER_LIBRARY_DIR,
    VENDOR_DIR,
    VENVS_DIR,
    WINDOW_OUTPUT_DIR,
    WINDOW_VALIDATION_OUTPUT,
)
from window.language_config import default_language_code, kroko_preview_model_name

ROOT = PROJECT_ROOT
os.environ.setdefault("NLTK_DATA", str(CACHE_DIR / "nltk"))
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from window.speaker_color_allocation import SpeakerColorAllocator

def _safe_console_text(text: object) -> str:
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _console_print(text: object) -> None:
    print(_safe_console_text(text), flush=True)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        _console_print(f"Ignoring invalid {name}={raw!r}; using {default}.")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    _console_print(f"Ignoring invalid {name}={raw!r}; using {default}.")
    return default


DEFAULT_OUTPUT_DIR = WINDOW_OUTPUT_DIR
DEFAULT_SPEAKER_LIBRARY_DIR = SPEAKER_LIBRARY_DIR
DEFAULT_VALIDATION_OUTPUT = WINDOW_VALIDATION_OUTPUT
DEFAULT_CUNK_CANONICAL = CUNK_CANONICAL
SPEAKER_COLORS = SpeakerColorAllocator(max_colors=16, allow_reuse=True).palette()
DEFAULT_REALTIMESTT_ROOT = Path(os.environ.get("REALTIMESTT_ROOT", str(VENDOR_DIR)))
DEFAULT_LANGUAGE = default_language_code()
DEFAULT_KROKO_PREVIEW_PYTHON = Path(os.environ.get(
    "WHOSPEAKS_KROKO_PREVIEW_PYTHON",
    str(VENVS_DIR / "kroko-install-test" / "Scripts" / "python.exe"),
))
DEFAULT_KROKO_COMMUNITY_64L_PREVIEW_MODEL = kroko_preview_model_name("en", "community-64l")
DEFAULT_KROKO_PRO_16L_PREVIEW_MODEL = kroko_preview_model_name("en", "pro-16l")
KROKO_PREVIEW_MODEL_REPO = os.environ.get("WHOSPEAKS_KROKO_PREVIEW_MODEL_REPO", "Banafo/Kroko-ASR")
KROKO_PREVIEW_MODEL_PRESETS = {
    "community-64l": DEFAULT_KROKO_COMMUNITY_64L_PREVIEW_MODEL,
    "pro-16l": DEFAULT_KROKO_PRO_16L_PREVIEW_MODEL,
}
KROKO_PREVIEW_MODEL_PRESET_ALIASES = {
    "64l": "community-64l",
    "64-l": "community-64l",
    "community": "community-64l",
    "community64": "community-64l",
    "community64l": "community-64l",
    "community-64": "community-64l",
    "community-64-l": "community-64l",
    "16": "pro-16l",
    "16l": "pro-16l",
    "16-l": "pro-16l",
    "16pro": "pro-16l",
    "pro": "pro-16l",
    "pro16": "pro-16l",
    "pro16l": "pro-16l",
    "pro-16": "pro-16l",
    "pro-16-l": "pro-16l",
}
_raw_default_kroko_preview_model_preset = (
    os.environ.get("WHOSPEAKS_KROKO_PREVIEW_MODEL_PRESET", "community-64l")
    .strip()
    .lower()
    .replace("_", "-")
)
DEFAULT_KROKO_PREVIEW_MODEL_PRESET = KROKO_PREVIEW_MODEL_PRESET_ALIASES.get(
    _raw_default_kroko_preview_model_preset,
    _raw_default_kroko_preview_model_preset,
)
if DEFAULT_KROKO_PREVIEW_MODEL_PRESET not in KROKO_PREVIEW_MODEL_PRESETS:
    _console_print(
        "Ignoring invalid WHOSPEAKS_KROKO_PREVIEW_MODEL_PRESET="
        f"{os.environ.get('WHOSPEAKS_KROKO_PREVIEW_MODEL_PRESET')!r}; using community-64l."
    )
    DEFAULT_KROKO_PREVIEW_MODEL_PRESET = "community-64l"
try:
    DEFAULT_KROKO_PREVIEW_MODEL = kroko_preview_model_name(
        DEFAULT_LANGUAGE,
        DEFAULT_KROKO_PREVIEW_MODEL_PRESET,
    )
except ValueError as exc:
    _console_print(f"{exc} Using community-64l for en.")
    DEFAULT_KROKO_PREVIEW_MODEL_PRESET = "community-64l"
    DEFAULT_KROKO_PREVIEW_MODEL = kroko_preview_model_name("en", "community-64l")
KROKO_PREVIEW_MODEL_STARTUP_TIMEOUT_SECONDS = {
    "community-64l": 12.0,
    "pro-16l": 45.0,
}
DEFAULT_KROKO_PREVIEW_MODEL_PATH = Path(os.environ.get(
    "WHOSPEAKS_KROKO_PREVIEW_MODEL_PATH",
    str(KROKO_MODEL_DIR / DEFAULT_KROKO_PREVIEW_MODEL),
))
DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD = _env_bool("WHOSPEAKS_KROKO_PREVIEW_AUTO_DOWNLOAD", True)
DEFAULT_KROKO_PREVIEW_MODEL_DIRS = (
    KROKO_MODEL_DIR,
    DEFAULT_REALTIMESTT_ROOT / "test-model-cache" / "kroko-onnx",
    DEFAULT_REALTIMESTT_ROOT / "RealtimeSTT" / "test-model-cache" / "kroko-onnx",
    PROJECT_ROOT.parent / "STT" / "RealtimeSTT" / "RealtimeSTT" / "test-model-cache" / "kroko-onnx",
)
DEFAULT_FAST_WHISPER_CACHE = CACHE_DIR / "faster-whisper"
DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS = _env_float(
    "WHOSPEAKS_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS",
    600.0,
)
SILERO_VAD_SAMPLE_RATE = 16000
SILERO_VAD_CHUNK_SAMPLES = 512
KROKO_PREVIEW_FRAME_SECONDS = 0.02
DEFAULT_KROKO_16L_CHUNK_SECONDS = 16 * KROKO_PREVIEW_FRAME_SECONDS
DEFAULT_REMOTE_ASR_URL = os.environ.get("WHOSPEAKS_REMOTE_ASR_URL", "http://127.0.0.1:8650")
DEFAULT_REMOTE_EMBEDDINGS_URL = os.environ.get("WHOSPEAKS_REMOTE_EMBEDDINGS_URL", "http://127.0.0.1:8660")
DEFAULT_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS = _env_float(
    "WHOSPEAKS_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS",
    600.0,
)
DEFAULT_WINDOW_EMBEDDING_PROVIDER = (
    "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
)
NEW_SPEAKER_SENSITIVITY_FIELDS = (
    "new_speaker_threshold",
    "duplicate_profile_similarity",
    "min_new_speaker_seconds",
    "late_new_speaker_min_seconds",
    "min_new_speaker_words",
    "new_speaker_confirmation_count",
    "new_speaker_confirmation_similarity",
)
NEW_SPEAKER_SENSITIVITY_PRESETS: dict[int, dict[str, Any]] = {
    1: {
        "label": "Very conservative",
        "new_speaker_threshold": 0.64,
        "duplicate_profile_similarity": 0.30,
        "min_new_speaker_seconds": 2.4,
        "late_new_speaker_min_seconds": 4.2,
        "min_new_speaker_words": 5,
        "new_speaker_confirmation_count": 2,
        "new_speaker_confirmation_similarity": 0.56,
    },
    2: {
        "label": "Conservative",
        "new_speaker_threshold": 0.50,
        "duplicate_profile_similarity": 0.35,
        "min_new_speaker_seconds": 2.0,
        "late_new_speaker_min_seconds": 3.8,
        "min_new_speaker_words": 4,
        "new_speaker_confirmation_count": 1,
        "new_speaker_confirmation_similarity": 0.53,
    },
    3: {
        "label": "Balanced",
        "new_speaker_threshold": 0.4309,
        "duplicate_profile_similarity": 0.4247,
        "min_new_speaker_seconds": 2.0358,
        "late_new_speaker_min_seconds": 3.1604,
        "min_new_speaker_words": 3,
        "new_speaker_confirmation_count": 1,
        "new_speaker_confirmation_similarity": 0.5801,
    },
    4: {
        "label": "Sensitive",
        "new_speaker_threshold": 0.32,
        "duplicate_profile_similarity": 0.45,
        "min_new_speaker_seconds": 1.3,
        "late_new_speaker_min_seconds": 3.0,
        "min_new_speaker_words": 2,
        "new_speaker_confirmation_count": 1,
        "new_speaker_confirmation_similarity": 0.48,
    },
    5: {
        "label": "Very sensitive",
        "new_speaker_threshold": 0.26,
        "duplicate_profile_similarity": 0.50,
        "min_new_speaker_seconds": 1.0,
        "late_new_speaker_min_seconds": 2.5,
        "min_new_speaker_words": 2,
        "new_speaker_confirmation_count": 1,
        "new_speaker_confirmation_similarity": 0.45,
    },
}


def default_silero_vad_model_path() -> Path | None:
    env_path = os.environ.get("WHOSPEAKS_SILERO_VAD_ONNX_MODEL_PATH")
    if env_path:
        return Path(env_path)

    site_package_roots = [
        Path(sys.prefix) / "Lib" / "site-packages",
        ROOT / ".venv-voice-embeddings" / "Lib" / "site-packages",
        VENVS_DIR / "kroko-install-test" / "Lib" / "site-packages",
        DEFAULT_REALTIMESTT_ROOT / ".venvs" / "kroko-install-test" / "Lib" / "site-packages",
        DEFAULT_REALTIMESTT_ROOT / ".venvs" / "install-matrix" / "default-faster-whisper" / "Lib" / "site-packages",
        DEFAULT_REALTIMESTT_ROOT / ".venvs" / "install-matrix" / "all" / "Lib" / "site-packages",
    ]
    for filename in ("silero_vad_op18_ifless.onnx", "silero_vad.onnx"):
        for root in site_package_roots:
            candidate = root / "silero_vad" / "data" / filename
            if candidate.exists():
                return candidate
    return None


def default_silero_vad_backend(model_path: Path | None) -> str:
    if model_path is not None and model_path.name == "silero_vad_op18_ifless.onnx":
        return "raw_onnx_ifless"
    if model_path is not None and model_path.name == "silero_vad.onnx":
        return "raw_onnx"
    return "auto"


def normalize_new_speaker_sensitivity(level: Any) -> int:
    try:
        value = int(level)
    except (TypeError, ValueError):
        value = 3
    return min(5, max(1, value))


def apply_new_speaker_sensitivity(args: argparse.Namespace, level: Any) -> dict[str, Any]:
    normalized = normalize_new_speaker_sensitivity(level)
    preset = NEW_SPEAKER_SENSITIVITY_PRESETS[normalized]
    for key in NEW_SPEAKER_SENSITIVITY_FIELDS:
        setattr(args, key, preset[key])
    args.new_speaker_sensitivity = normalized
    args.new_speaker_sensitivity_label = preset["label"]
    return preset


def new_speaker_sensitivity_config(selected: Any) -> dict[str, Any]:
    level = normalize_new_speaker_sensitivity(selected)
    return {
        "selected": level,
        "presets": [
            {
                "level": preset_level,
                "label": str(preset["label"]),
                "settings": {
                    key: preset[key]
                    for key in NEW_SPEAKER_SENSITIVITY_FIELDS
                },
            }
            for preset_level, preset in sorted(NEW_SPEAKER_SENSITIVITY_PRESETS.items())
        ],
    }


def safe_library_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._ -]+", "", str(name or "").strip())
    value = re.sub(r"\s+", "_", value).strip("._- ")
    if not value:
        raise ValueError("Speaker group name must not be empty.")
    return value[:80]


def safe_reference_filename(name: str) -> str:
    stem = Path(str(name or "reference.wav")).stem
    suffix = Path(str(name or "reference.wav")).suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".webm"}:
        suffix = ".wav"
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "", stem).strip("._- ") or "reference"
    return f"{safe_stem[:80]}{suffix}"


def speaker_group_dir(root: Path, name: str) -> Path:
    return root / safe_library_name(name)


def list_speaker_groups(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    groups = [
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "manifest.json").is_file()
    ]
    return sorted(groups, key=str.lower)

PRESET_YOUTUBE_VIDEOS = [
    {
        "title": "Philomena Cunk - multiple speakers",
        "url": "https://www.youtube.com/watch?v=JWS-qfR6K3w",
    },
    {
        "title": "Coin Toss scene",
        "url": "https://www.youtube.com/watch?v=ZY0DG8rUnCA",
    },
    {
        "title": "Margin Call meeting scene",
        "url": "https://www.youtube.com/watch?v=acbnyagl8jo",
    },
    {
        "title": 'Mark Hamill - "I am your father" secret',
        "url": "https://www.youtube.com/watch?v=oFBuCp19L7M",
    },
    {
        "title": "Gordon tries to make Pad Thai",
        "url": "https://www.youtube.com/watch?v=DsyfYJ5Ou3g",
    },
    {
        "title": "Simon Pegg - Benedict Cumberbatch truth",
        "url": "https://www.youtube.com/watch?v=20v1OxUXcQY",
    },
    {
        "title": "Elon Musk and China's richest man",
        "url": "https://www.youtube.com/watch?v=aHGd6LqAVzw",
    },
    {
        "title": "Louis Theroux - interview with drug dealer",
        "url": "https://www.youtube.com/watch?v=1NBVQB-Srpw",
    },
    {
        "title": "Conan & Norm cook with Gordon Ramsay",
        "url": "https://www.youtube.com/watch?v=KdOXM3I_5hk",
    },
    {
        "title": "Blake Lively interview",
        "url": "https://www.youtube.com/watch?v=F2-2RBi1qzY",
    },
    {
        "title": "True Confessions - Kate McKinnon and John Cena",
        "url": "https://www.youtube.com/watch?v=gj7BRMuB-n4",
    },
    {
        "title": "True Confessions - Billie Eilish and Colin Quinn",
        "url": "https://www.youtube.com/watch?v=mWABb5Dy9BQ",
    },
    {
        "title": "True Confessions - Matthew McConaughey and Hugh Grant",
        "url": "https://www.youtube.com/watch?v=WNZn37Uc700",
    },
    {
        "title": "Substitute Teacher - Key & Peele",
        "url": "https://www.youtube.com/watch?v=Dd7FixvoKBw",
    },
    {
        "title": "Barbie Instagram - SNL",
        "url": "https://www.youtube.com/watch?v=blcKeLDDzSM",
    },
]


def normalize_kroko_preview_model_preset(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    preset = KROKO_PREVIEW_MODEL_PRESET_ALIASES.get(normalized, normalized)
    if preset not in KROKO_PREVIEW_MODEL_PRESETS:
        allowed = ", ".join(KROKO_PREVIEW_MODEL_PRESETS)
        raise argparse.ArgumentTypeError(f"invalid Kroko preview model preset {value!r}; choose one of: {allowed}")
    return preset


def default_kroko_preview_startup_timeout_seconds(preset: Any) -> float:
    normalized = str(preset or "").strip().lower().replace("_", "-")
    return KROKO_PREVIEW_MODEL_STARTUP_TIMEOUT_SECONDS.get(normalized, 12.0)


def kroko_preview_model_search_dirs() -> list[Path]:
    env_dirs = [
        Path(value).expanduser()
        for value in os.environ.get("WHOSPEAKS_KROKO_PREVIEW_MODEL_DIR", "").split(os.pathsep)
        if value.strip()
    ]
    dirs: list[Path] = []
    seen: set[str] = set()
    for directory in [*env_dirs, *DEFAULT_KROKO_PREVIEW_MODEL_DIRS]:
        key = str(directory).lower()
        if key in seen:
            continue
        dirs.append(directory)
        seen.add(key)
    return dirs


def default_kroko_preview_model_path(model: Any = None, *, use_env: bool = True) -> Path | None:
    if use_env and "WHOSPEAKS_KROKO_PREVIEW_MODEL_PATH" in os.environ:
        return DEFAULT_KROKO_PREVIEW_MODEL_PATH.expanduser()
    selected_model = str(model or DEFAULT_KROKO_PREVIEW_MODEL)
    selected_path = Path(selected_model).expanduser()
    if selected_path.is_file():
        return selected_path
    for model_dir in kroko_preview_model_search_dirs():
        candidate = model_dir / Path(selected_model).name
        if candidate.is_file():
            return candidate
    return None


def is_public_kroko_community_model(model: Any) -> bool:
    name = Path(str(model or "")).name
    return bool(re.fullmatch(r"Kroko-[A-Z]{2}-Community-64-L-Streaming-\d{3}\.data", name))


def download_kroko_preview_model(
    model: Any,
    *,
    target_dir: Path | None = None,
    repo_id: str = KROKO_PREVIEW_MODEL_REPO,
) -> Path:
    filename = Path(str(model or "")).name
    if not is_public_kroko_community_model(filename):
        raise RuntimeError(
            f"Automatic Kroko download only supports public Community models, got {model!r}."
        )
    target_root = Path(target_dir or KROKO_MODEL_DIR).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / filename
    if target.is_file():
        return target

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-download Kroko preview models."
        ) from exc

    downloaded = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_root),
        local_dir_use_symlinks=False,
    ))
    if downloaded.resolve() != target.resolve() and downloaded.is_file():
        shutil.copy2(downloaded, target)
    if not target.is_file():
        raise RuntimeError(f"Kroko preview model download did not produce {target}")
    return target


def default_faster_whisper_download_root() -> Path | None:
    env_path = os.environ.get("WHOSPEAKS_FAST_WHISPER_DOWNLOAD_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_FAST_WHISPER_CACHE if DEFAULT_FAST_WHISPER_CACHE.exists() else None


from window.window_gui_html import HTML  # noqa: E402



