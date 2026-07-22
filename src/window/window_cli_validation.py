"""Browser-synced growing-window diarization experiment.

No RealtimeSTT is used here. The backend periodically transcribes the current
audio window with faster-whisper large-v2, emits confirmed complete sentences,
and clusters one embedding per emitted sentence.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, quote, unquote, urlparse


def _configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def _safe_console_text(text: object) -> str:
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _console_print(text: object) -> None:
    print(_safe_console_text(text), flush=True)


_configure_console_output()

if __name__ == "__main__":
    _console_print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting youtube_window_diarize_gui.py; importing dependencies.",
    )

import numpy as np

from paths import CACHE_DIR, PROJECT_ROOT, VENDOR_DIR

ROOT = PROJECT_ROOT
os.environ.setdefault("NLTK_DATA", str(CACHE_DIR / "nltk"))
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from realtime.realtime_speakerdiarize import (  # noqa: E402
    EmbeddingSubprocessClient,
    default_embedding_python,
    json_dumps,
    load_audio_file,
    pad_audio,
    trim_silence,
    write_wav,
)
from speakers.speaker_embedding_cluster import SpeakerMemory  # noqa: E402
from window.speaker_color_allocation import SpeakerColorAllocator  # noqa: E402
from stream2sentence import generate_sentences, init_tokenizer  # noqa: E402
from replay.youtube_local_filefeed_replay import (  # noqa: E402
    DEFAULT_URL,
    DEFAULT_WORK_DIR,
)
from window.window_domain import (  # noqa: E402
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    EmbeddingSentenceJob,
    MappedWord,
    MediaFiles,
    PendingUnknownSentence,
    SentencePart,
    TimedWord,
    VadWindowState,
    WindowTranscript,
)
from window.window_media import (  # noqa: E402
    media_cache_status,
    resolve_browser_stream_id,
    resolve_media,
    resolve_media_url,
)
from window.window_remote_asr import RemoteWindowAsrClient  # noqa: E402
from window.session_store import DEFAULT_SESSION_DIR, SessionStore  # noqa: E402
from window.session_lease import SessionLease, SessionLeaseError, SessionLeaseStateMachine  # noqa: E402
from window.session_persistence import SessionPersistenceCoordinator  # noqa: E402
from window.media_manager import MediaManager  # noqa: E402
from window.live_translation import LiveTranslationCoordinator  # noqa: E402


from window.window_config import (  # noqa: E402
    DEFAULT_CUNK_CANONICAL,
    DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS,
    DEFAULT_FAST_WHISPER_CACHE,
    DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD,
    DEFAULT_KROKO_PREVIEW_MODEL_PRESET,
    DEFAULT_KROKO_PREVIEW_PYTHON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REALTIMESTT_ROOT,
    DEFAULT_REMOTE_ASR_URL,
    DEFAULT_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS,
    DEFAULT_REMOTE_EMBEDDINGS_URL,
    DEFAULT_SPEAKER_LIBRARY_DIR,
    DEFAULT_VALIDATION_OUTPUT,
    DEFAULT_WINDOW_EMBEDDING_PROVIDER,
    NEW_SPEAKER_SENSITIVITY_PRESETS,
    PRESET_YOUTUBE_VIDEOS,
    SPEAKER_COLORS,
    apply_new_speaker_sensitivity,
    default_faster_whisper_download_root,
    default_kroko_preview_model_path,
    default_kroko_preview_startup_timeout_seconds,
    default_silero_vad_backend,
    default_silero_vad_model_path,
    new_speaker_sensitivity_config,
)
from window.language_config import (  # noqa: E402
    default_language_code,
    default_sentence_language,
    default_sentence_tokenizer,
    get_language_config,
    infer_language_from_kroko_model_name,
    language_arg,
    language_flag_country_code,
    sentence_tokenizer_arg,
)
from window.realtime_preview_backends import (  # noqa: E402
    apply_preview_timing_defaults,
    default_preview_model,
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
)
from window.sherpa_onnx_models import (  # noqa: E402
    DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET,
    default_sherpa_onnx_model_dir,
    sherpa_onnx_model_preset,
)
from window.window_diarizer import StartSessionRequest, WindowDiarizer  # noqa: E402
from window.window_events import EventBus, RecordingEventBus  # noqa: E402
from window.web_assets import (  # noqa: E402
    read_web_asset,
    render_live_index,
    web_asset_content_type,
)
from window.public_events import PublicEventNormalizer  # noqa: E402
from window.browser_live_speaker_scoring import (  # noqa: E402
    DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
    BrowserLiveObservationRecorder,
)

AUDIO_UPLOAD_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}




def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keep-segment-audio", action="store_true")
    parser.add_argument("--validate-window-replay", action="store_true")
    parser.add_argument("--validation-canonical", type=Path, default=DEFAULT_CUNK_CANONICAL)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--validation-trace-output", type=Path, default=None)
    parser.add_argument(
        "--live-speaker-world-tape-output",
        type=Path,
        default=None,
        help=(
            "Campaign directory for a production World Tape. Each GUI launch creates "
            "a unique crash-resilient child directory containing ordered events and arrays."
        ),
    )
    parser.add_argument("--validation-replay-speed", type=float, default=1.0)
    parser.add_argument("--validation-update-interval-seconds", type=float, default=0.1)
    parser.add_argument("--validation-final-wait-seconds", type=float, default=90.0)
    parser.add_argument("--validation-match-mode", choices=("auto", "timestamp", "text"), default="auto")
    parser.add_argument(
        "--browser-live-observation-output",
        type=Path,
        default=None,
        help="When set, the browser samples the rendered live-speaker DOM state and writes a strict browser-observed score JSON here.",
    )
    parser.add_argument(
        "--browser-live-e2e-candidate-artifact",
        type=Path,
        default=None,
        help=(
            "Candidate JSON to hash-bind into a real GUI E2E observation. "
            "A recording without this binding remains diagnostic and cannot be promoted."
        ),
    )
    parser.add_argument(
        "--browser-live-observation-interval-seconds",
        type=float,
        default=DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS,
        help="Seconds between browser DOM live-speaker samples.",
    )
    parser.add_argument(
        "--browser-live-observation-max-sample-gap-seconds",
        type=float,
        default=DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
        help="Maximum playback span represented by one browser DOM sample interval.",
    )
    parser.add_argument(
        "--browser-live-observation-flicker-gap-seconds",
        type=float,
        default=DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
        help="Minimum in-turn live-speaker gap counted as visible flicker.",
    )
    parser.add_argument(
        "--exit-after-browser-live-observation",
        action="store_true",
        help=(
            "Shut down the GUI server after the browser submits its final live-observation "
            "batch. Intended for one-run World-Tape campaigns so the tape is sealed cleanly."
        ),
    )
    parser.add_argument(
        "--validation-keep-preview",
        action="store_true",
        help="Keep realtime preview enabled during validation. Final sentence metrics usually do not need this.",
    )
