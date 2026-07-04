"""Central filesystem paths for source, fixtures, and mutable runtime data."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = Path(os.environ.get("WHOSPEAKS_PROJECT_ROOT", SRC_ROOT.parent)).resolve()
VENDOR_DIR = PROJECT_ROOT / "vendor"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


RUNTIME_DIR = _path_from_env("WHOSPEAKS_RUNTIME_DIR", PROJECT_ROOT / "runtime")
CACHE_DIR = _path_from_env("WHOSPEAKS_CACHE_DIR", RUNTIME_DIR / "cache")
MODEL_DIR = _path_from_env("WHOSPEAKS_MODEL_DIR", RUNTIME_DIR / "models")
MODEL_HUB_DIR = MODEL_DIR / "hub"
KROKO_MODEL_DIR = MODEL_DIR / "kroko-onnx"
MEDIA_DIR = RUNTIME_DIR / "media"
LOCAL_FILEFEED_MEDIA_DIR = MEDIA_DIR / "local-filefeed"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
WINDOW_OUTPUT_DIR = OUTPUTS_DIR / "window-diarize"
WINDOW_VALIDATION_OUTPUT = OUTPUTS_DIR / "window-diarize-validation" / "latest.json"
REALTIME_VALIDATION_OUTPUT_DIR = OUTPUTS_DIR / "realtime-speakerdiarize-validation"
SPEAKER_LIBRARY_DIR = _path_from_env("WHOSPEAKS_SPEAKER_LIBRARY_DIR", RUNTIME_DIR / "speakers")

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
CUNK_FIXTURE_DIR = FIXTURES_DIR / "cunk"
CUNK_CANONICAL = CUNK_FIXTURE_DIR / "cunk_on_earth_clip.canonical_diarization.json"

MAIN_VENV = PROJECT_ROOT / ".venv"
EMBEDDING_VENV = PROJECT_ROOT / ".venv-voice-embeddings"
VENVS_DIR = PROJECT_ROOT / ".venvs"


def ensure_runtime_dirs() -> None:
    """Create the standard mutable directories used by local runs."""

    for directory in (
        RUNTIME_DIR,
        CACHE_DIR,
        MODEL_DIR,
        MODEL_HUB_DIR,
        KROKO_MODEL_DIR,
        LOCAL_FILEFEED_MEDIA_DIR,
        OUTPUTS_DIR,
        WINDOW_OUTPUT_DIR,
        REALTIME_VALIDATION_OUTPUT_DIR,
        SPEAKER_LIBRARY_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
