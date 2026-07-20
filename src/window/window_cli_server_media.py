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




def add_server_media_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Directory used for durable saved WhoSpeaks Live sessions.",
    )
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--video-file", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--max-audio-upload-mb",
        type=float,
        default=2048.0,
        help="Maximum browser audio-file upload size in MiB.",
    )
    parser.add_argument("--yt-dlp", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument(
        "--meeting-intelligence-url",
        default="",
        help="Base URL of the optional Meeting Intelligence service used for Reports and Ask.",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--demo-seat-lease",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require one browser tab to take the public demo seat before controlling a shared run.",
    )
    parser.add_argument(
        "--session-lease-idle-timeout-seconds",
        type=float,
        default=120.0,
        help="Release an acquired demo seat after this many seconds if no run has started.",
    )
    parser.add_argument(
        "--session-lease-heartbeat-timeout-seconds",
        type=float,
        default=45.0,
        help="Release and stop an active demo seat if the owner tab stops sending heartbeats.",
    )
    parser.add_argument(
        "--session-lease-completed-release-delay-seconds",
        type=float,
        default=10.0,
        help="Seconds to keep a completed one-seat demo session before releasing it for the next user.",
    )
    parser.add_argument(
        "--session-lease-max-run-seconds",
        type=float,
        default=900.0,
        help="Hard maximum owner runtime for one public demo seat.",
    )
    parser.add_argument(
        "--asr-backend",
        choices=("local", "remote", "cpu"),
        default="local",
        help="Final growing-window ASR: local faster-whisper, remote service, or CPU streaming ASR.",
    )
    parser.add_argument(
        "--remote-asr-url",
        default=DEFAULT_REMOTE_ASR_URL,
        help="Base URL of the remote faster-whisper large-v2 ASR server.",
    )
    parser.add_argument(
        "--remote-asr-timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for each remote ASR request.",
    )
    parser.add_argument(
        "--speech-enhancement-url",
        default="http://127.0.0.1:8651",
        help="Base URL of the optional UniSE speech-enhancement service.",
    )
    parser.add_argument(
        "--speech-enhancement-timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for each speech-enhancement request.",
    )
    parser.add_argument(
        "--enhance-asr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enhance only final ASR windows; VAD, preview, and the raw timeline remain unchanged.",
    )
    parser.add_argument(
        "--enhance-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enhance only final sentence audio before embedding; live speaker probes remain unchanged.",
    )
