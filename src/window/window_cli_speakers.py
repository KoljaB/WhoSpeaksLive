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




def add_embedding_speaker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-provider", default=DEFAULT_WINDOW_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-python", type=Path, default=default_embedding_python())
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument(
        "--live-speaker-embedding-provider",
        default="pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50",
        help="Provider or weighted provider stack used only for fast live speaker assignment. Empty uses --embedding-provider.",
    )
    parser.add_argument(
        "--live-speaker-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable live speaker highlighting/scoring during realtime preview. "
            "Use --no-live-speaker-assignment to keep live text preview without live speaker scoring."
        ),
    )
    parser.add_argument(
        "--embeddings-backend",
        "--embedding-backend",
        "-embeddings-backend",
        choices=("local", "remote"),
        default="local",
        help="Speaker embedding backend. Use remote to send embedding requests to the Linux GPU server.",
    )
    parser.add_argument(
        "--remote-embeddings-url",
        "--remote-embedding-url",
        default=DEFAULT_REMOTE_EMBEDDINGS_URL,
        help="Base URL of the remote voice embeddings server.",
    )
    parser.add_argument(
        "--remote-embeddings-timeout-seconds",
        "--remote-embedding-timeout-seconds",
        type=float,
        default=DEFAULT_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS,
        help="HTTP timeout for remote embedding health, load, and embed requests.",
    )
    parser.add_argument(
        "--remote-embeddings-device",
        "--remote-embedding-device",
        default="auto",
        help="Device query parameter sent to the remote embeddings server.",
    )
    parser.add_argument(
        "--embedding-helper-response-timeout-seconds",
        type=float,
        default=DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS,
        help=(
            "Maximum time to wait for an embedding helper response. First startup of the "
            "default high-quality stacked provider may need several minutes while models "
            "download and initialize."
        ),
    )
    parser.add_argument(
        "--speaker-library-dir",
        type=Path,
        default=DEFAULT_SPEAKER_LIBRARY_DIR,
        help="Directory for saved speaker groups and uploaded reference audio.",
    )
    parser.add_argument(
        "--new-speaker-sensitivity",
        type=int,
        choices=range(1, 6),
        default=None,
        metavar="{1,2,3,4,5}",
        help="Optional five-step new-speaker spawning sensitivity preset. Position 3 matches the tuned defaults.",
    )
    parser.add_argument("--same-speaker-similarity", type=float, default=0.43)
    parser.add_argument("--similarity-temperature", type=float, default=0.061)
    parser.add_argument("--speaker-softmax-temperature", type=float, default=0.0557)
    parser.add_argument("--new-speaker-threshold", type=float, default=0.4309)
    parser.add_argument("--duplicate-profile-similarity", type=float, default=0.4247)
    parser.add_argument("--unknown-short-threshold", type=float, default=0.287)
    parser.add_argument("--min-first-speaker-seconds", type=float, default=1.8373)
    parser.add_argument(
        "--first-speaker-immediate-min-seconds",
        type=float,
        default=4.0,
        help=(
            "Create the first speaker immediately only from a sentence at least this long. "
            "Shorter eligible sentences remain provisional until a similar sentence confirms them."
        ),
    )
    parser.add_argument("--min-new-speaker-seconds", type=float, default=2.0358)
    parser.add_argument("--late-new-speaker-min-seconds", type=float, default=3.1604)
    parser.add_argument("--max-speakers", type=int, default=12)
    parser.add_argument("--min-margin", type=float, default=0.0372)
    parser.add_argument("--margin-temperature", type=float, default=0.0361)
    parser.add_argument("--update-unknown-max", type=float, default=0.4289)
    parser.add_argument(
        "--new-speaker-confirmation-count",
        type=int,
        default=1,
        help="Number of mutually similar far-away sentence embeddings required before creating a new speaker.",
    )
    parser.add_argument(
        "--new-speaker-confirmation-similarity",
        type=float,
        default=0.5801,
        help="Minimum cosine similarity between pending new-speaker candidates before creating a speaker.",
    )
    parser.add_argument("--max-pending-new-speakers", type=int, default=6)
    parser.add_argument(
        "--known-speaker-min-similarity",
        type=float,
        default=0.5563,
        help="When non-negative, existing speakers below this top similarity are treated as gray-zone UNKNOWN instead of confidently assigned.",
    )
    parser.add_argument(
        "--known-speaker-gray-zone-min-unknown-probability",
        type=float,
        default=0.064,
        help="Minimum unknown probability required before --known-speaker-min-similarity defers an assignment to UNKNOWN.",
    )
    parser.add_argument(
        "--profile-update-min-similarity",
        type=float,
        default=0.5011,
        help="When non-negative, update existing speaker centroids only if top similarity is at least this value.",
    )
    parser.add_argument(
        "--profile-update-min-margin",
        type=float,
        default=0.0037,
        help="When non-negative, update existing speaker centroids only if top-vs-runner-up margin is at least this value.",
    )
    parser.add_argument(
        "--low-similarity-unknown-floor-similarity",
        type=float,
        default=0.56,
        help="When non-negative, raise unknown probability for known-speaker comparisons below this top similarity.",
    )
    parser.add_argument(
        "--low-similarity-unknown-floor-probability",
        type=float,
        default=0.1885,
        help="Unknown probability floor used with --low-similarity-unknown-floor-similarity.",
    )
    parser.add_argument(
        "--gray-zone-promote-max-similarity",
        type=float,
        default=0.55,
        help="Maximum candidate-vs-known centroid similarity allowed before a gray-zone pending voice can become a new speaker.",
    )
    parser.add_argument(
        "--min-new-speaker-words",
        type=int,
        default=3,
        help="Minimum content words required for a sentence to create or confirm a new speaker profile.",
    )
    parser.add_argument(
        "--retro-reassign-min-similarity",
        type=float,
        default=0.02,
        help="Minimum cosine similarity for assigning an earlier UNKNOWN sentence to an existing speaker.",
    )
    parser.add_argument(
        "--retro-reassign-min-margin",
        type=float,
        default=0.0,
        help="Minimum top-vs-runner-up similarity gap for retro UNKNOWN reassignment when multiple speakers exist.",
    )
    parser.add_argument(
        "--speaker-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prototype-based live refinement. Stable mode only fills UNKNOWN rows later.",
    )
    parser.add_argument(
        "--speaker-refinement-unknown-tentative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow prototype refinement to show tentative speaker hints on UNKNOWN transcript rows.",
    )
    parser.add_argument(
        "--speaker-refinement-unknown-commit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow later evidence to commit UNKNOWN transcript rows to a known or newly confirmed speaker.",
    )
    parser.add_argument(
        "--allow-speaker-reassignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow prototype refinement to change already committed non-UNKNOWN speaker labels.",
    )
    parser.add_argument("--speaker-refinement-max-per-profile", type=int, default=32)
    parser.add_argument("--speaker-refinement-min-duration", type=float, default=0.15)
    parser.add_argument("--speaker-refinement-max-unknown", type=float, default=1.0)
    parser.add_argument("--speaker-refinement-top-k", type=int, default=12)
    parser.add_argument("--speaker-refinement-centroid-blend", type=float, default=0.555)
    parser.add_argument("--speaker-refinement-unknown-min-similarity", type=float, default=0.20)
    parser.add_argument("--speaker-refinement-unknown-min-margin", type=float, default=0.0)
    parser.add_argument("--speaker-refinement-known-max-duration", type=float, default=8.0)
    parser.add_argument("--speaker-refinement-known-min-similarity", type=float, default=-0.039)
    parser.add_argument("--speaker-refinement-known-min-delta", type=float, default=0.04)
    parser.add_argument(
        "--delayed-multirow-clustering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Split a polluted speaker profile only after multiple uncertain rows across time "
            "jointly support a stable second voice cluster."
        ),
    )
    parser.add_argument("--delayed-clustering-core-max-unknown", type=float, default=0.50)
    parser.add_argument("--delayed-clustering-core-min-duration", type=float, default=0.80)
    parser.add_argument("--delayed-clustering-min-core-rows", type=int, default=4)
    parser.add_argument("--delayed-clustering-min-core-duration", type=float, default=8.0)
    parser.add_argument("--delayed-clustering-candidate-min-unknown", type=float, default=0.50)
    parser.add_argument("--delayed-clustering-candidate-min-duration", type=float, default=0.35)
    parser.add_argument("--delayed-clustering-candidate-max-core-similarity", type=float, default=0.45)
    parser.add_argument("--delayed-clustering-candidate-min-similarity", type=float, default=0.20)
    parser.add_argument("--delayed-clustering-candidate-min-gain", type=float, default=0.02)
    parser.add_argument("--delayed-clustering-min-candidate-rows", type=int, default=4)
    parser.add_argument("--delayed-clustering-min-candidate-duration", type=float, default=8.0)
    parser.add_argument("--delayed-clustering-min-candidate-span", type=float, default=12.0)
    parser.add_argument("--delayed-clustering-min-candidate-time-groups", type=int, default=2)
    parser.add_argument("--delayed-clustering-time-group-gap", type=float, default=8.0)
    parser.add_argument("--delayed-clustering-min-average-gain", type=float, default=0.22)
    parser.add_argument("--delayed-clustering-min-leave-one-out-similarity", type=float, default=0.16)
    parser.add_argument("--delayed-clustering-max-core-centroid-similarity", type=float, default=0.58)
    parser.add_argument("--delayed-clustering-max-new-speakers", type=int, default=2)
    parser.add_argument(
        "--speaker-refinement-final-passes",
        type=int,
        default=1,
        help="Bounded extra speaker refinement passes after the final sentence is committed.",
    )
    parser.add_argument(
        "--speaker-refinement-small-island-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge a tiny one-off speaker island when the same speaker appears immediately before and after it.",
    )
    parser.add_argument("--speaker-refinement-small-island-max-duration", type=float, default=5.0)
    parser.add_argument("--speaker-refinement-small-island-max-segments", type=int, default=3)
    parser.add_argument(
        "--speaker-refinement-tiny-fragmented-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge a very small fragmented speaker profile into its dominant neighboring speaker at finalization.",
    )
    parser.add_argument("--speaker-refinement-tiny-fragmented-max-duration", type=float, default=6.0)
    parser.add_argument("--speaker-refinement-tiny-fragmented-max-segments", type=int, default=8)
    parser.add_argument("--speaker-refinement-tiny-fragmented-min-islands", type=int, default=2)
    parser.add_argument("--speaker-refinement-tiny-fragmented-max-islands", type=int, default=3)
    parser.add_argument("--speaker-refinement-tiny-fragmented-min-neighbor-share", type=float, default=0.5)
    parser.add_argument(
        "--speaker-refinement-terminal-outro-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge a singleton terminal promotional outro back to the stable opening speaker.",
    )
    parser.add_argument("--speaker-refinement-terminal-outro-max-duration", type=float, default=12.0)
    parser.add_argument("--speaker-refinement-terminal-outro-lookback-segments", type=int, default=2)
    parser.add_argument("--speaker-refinement-terminal-outro-min-target-duration", type=float, default=5.0)
    parser.add_argument(
        "--speaker-refinement-unknown-same-speaker-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill a short UNKNOWN island only when it is flanked by the same speaker on both sides.",
    )
    parser.add_argument("--speaker-refinement-unknown-same-speaker-max-duration", type=float, default=3.0)
    parser.add_argument("--speaker-refinement-unknown-same-speaker-max-segments", type=int, default=1)
    parser.add_argument(
        "--speaker-refinement-unknown-previous-speaker-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill a short non-embedding UNKNOWN tail only when it is contiguous with the previous speaker and separated from the next speaker by a pause.",
    )
    parser.add_argument("--speaker-refinement-unknown-previous-speaker-max-duration", type=float, default=0.75)
    parser.add_argument("--speaker-refinement-unknown-previous-speaker-max-segments", type=int, default=1)
    parser.add_argument("--speaker-refinement-unknown-previous-speaker-max-previous-gap", type=float, default=0.35)
    parser.add_argument("--speaker-refinement-unknown-previous-speaker-min-next-gap", type=float, default=0.3)
    parser.add_argument(
        "--speaker-refinement-unknown-next-speaker-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill a short non-embedding UNKNOWN head only when it is separated from the previous speaker by a pause and contiguous with the next speaker.",
    )
    parser.add_argument("--speaker-refinement-unknown-next-speaker-max-duration", type=float, default=1.75)
    parser.add_argument("--speaker-refinement-unknown-next-speaker-max-segments", type=int, default=1)
    parser.add_argument("--speaker-refinement-unknown-next-speaker-max-next-gap", type=float, default=0.05)
    parser.add_argument("--speaker-refinement-unknown-next-speaker-min-previous-gap", type=float, default=0.15)
    parser.add_argument(
        "--speaker-refinement-long-low-confidence-retro-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split a long, very low-confidence retro assignment into a new final speaker.",
    )
    parser.add_argument("--speaker-refinement-long-low-confidence-retro-min-duration", type=float, default=4.0)
    parser.add_argument("--speaker-refinement-long-low-confidence-retro-max-similarity", type=float, default=0.06)
    parser.add_argument("--speaker-refinement-long-low-confidence-retro-max-margin", type=float, default=0.04)
    parser.add_argument("--speaker-refinement-long-low-confidence-retro-max-splits", type=int, default=1)
