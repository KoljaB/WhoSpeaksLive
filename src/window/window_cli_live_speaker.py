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




def add_preview_live_speaker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-embed-seconds", type=float, default=0.5)
    parser.add_argument(
        "--min-speech-audio-ratio",
        type=float,
        default=0.0,
        help="Minimum sum(word durations) / sentence audio duration required before embedding a sentence.",
    )
    parser.add_argument(
        "--realtime-preview-engine",
        default="kroko_onnx",
        help="Realtime preview engine: kroko_onnx, sherpa_onnx (Nemotron 3.5), mock, or off.",
    )
    parser.add_argument(
        "--realtime-preview-model",
        default=None,
        help="Backend model name or named preset. For sherpa_onnx use a Nemotron preset.",
    )
    parser.add_argument(
        "--realtime-preview-model-preset",
        default=None,
        help=(
            "Named model preset. Kroko: community-64l or pro-16l. "
            "Nemotron: nemotron-3.5-160ms-int8 or nemotron-3.5-560ms-int8."
        ),
    )
    parser.add_argument("--realtime-preview-model-path", type=Path, default=None)
    parser.add_argument("--realtime-preview-model-dir", type=Path, default=None)
    parser.add_argument(
        "--realtime-preview-auto-download",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD,
        help="Download a missing supported preview model before starting preview.",
    )
    parser.add_argument("--realtime-preview-download-root", type=Path, default=None)
    parser.add_argument("--realtime-preview-python", type=Path, default=None)
    parser.add_argument("--realtime-preview-realtimestt-root", type=Path, default=DEFAULT_REALTIMESTT_ROOT)
    parser.add_argument("--realtime-preview-provider", default="cpu")
    parser.add_argument("--realtime-preview-num-threads", type=int, default=2)
    parser.add_argument(
        "--realtime-preview-startup-timeout-seconds",
        type=float,
        default=None,
        help="Maximum time to wait for the realtime preview engine before disabling preview.",
    )
    parser.add_argument(
        "--realtime-preview-request-timeout-seconds",
        type=float,
        default=5.0,
        help="Maximum time to wait for one realtime preview decode request.",
    )
    parser.add_argument("--realtime-preview-interval-seconds", type=float, default=None)
    parser.add_argument("--realtime-preview-min-audio-seconds", type=float, default=None)
    parser.add_argument("--realtime-preview-min-advance-seconds", type=float, default=None)
    parser.add_argument(
        "--realtime-preview-feed-chunk-seconds",
        type=float,
        default=None,
        help="Audio seconds fed to Kroko per streaming accept call. By default this is inferred from the Kroko model name.",
    )
    parser.add_argument(
        "--realtime-preview-vad-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start Kroko preview only after VAD speech onset and reset it after sustained non-speech.",
    )
    parser.add_argument(
        "--realtime-preview-vad-gate-pre-padding-seconds",
        type=float,
        default=0.35,
        help="Buffered audio kept before VAD speech onset when starting Kroko preview.",
    )
    parser.add_argument(
        "--realtime-preview-vad-gate-post-padding-seconds",
        type=float,
        default=0.35,
        help="Audio kept after VAD speech end before resetting Kroko preview.",
    )
    parser.add_argument(
        "--realtime-preview-vad-gate-close-silence-seconds",
        type=float,
        default=1.1,
        help="Sustained VAD non-speech required before closing and resetting a Kroko preview session.",
    )
    parser.add_argument(
        "--realtime-preview-reset-overlap-seconds",
        type=float,
        default=0.15,
        help="Audio pre-roll kept before the committed sentence boundary when resetting preview after final sentence commits.",
    )
    parser.add_argument(
        "--realtime-preview-diarize-min-audio-seconds",
        type=float,
        default=1.5,
        help="Minimum live unresolved audio duration before scoring it against known speakers.",
    )
    parser.add_argument(
        "--realtime-preview-diarize-min-advance-seconds",
        type=float,
        default=0.75,
        help="Minimum live playback advance before recomputing the live speaker embedding.",
    )
    parser.add_argument(
        "--realtime-preview-diarize-min-similarity",
        type=float,
        default=0.45,
        help="Minimum cosine similarity for assigning a live preview row to an existing speaker.",
    )
    parser.add_argument(
        "--realtime-preview-diarize-min-margin",
        type=float,
        default=0.08,
        help="Minimum top-vs-runner-up margin for assigning a live preview row when multiple speakers exist.",
    )
    parser.add_argument(
        "--realtime-preview-diarize-min-known-probability",
        type=float,
        default=0.5,
        help="Minimum known-speaker probability before the live row label switches from Unknown to a speaker.",
    )
    parser.add_argument(
        "--live-speaker-embedding-min-interval-seconds",
        type=float,
        default=0.75,
        help="Minimum wall-clock spacing between live speaker embedding requests from preview/probe paths.",
    )
    parser.add_argument(
        "--live-speaker-embedding-target-utilization",
        type=float,
        default=0.25,
        help="Target fraction of wall time live speaker embeddings may occupy; use 1.0 to disable latency backoff.",
    )
    parser.add_argument(
        "--live-speaker-verify-on-change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the full embedding stack to confirm visible live speaker changes proposed by the fast provider.",
    )
    parser.add_argument(
        "--live-speaker-verify-min-interval-seconds",
        type=float,
        default=2.0,
        help="Minimum wall-clock spacing between full-stack live speaker change verification requests.",
    )
    parser.add_argument(
        "--live-speaker-ema-window-seconds",
        type=float,
        default=1.0,
        help="Wall-clock window used for smoothing live speaker probabilities.",
    )
    parser.add_argument(
        "--live-speaker-ema-count",
        type=int,
        default=1,
        help="Maximum number of recent live speaker probability snapshots blended by EMA.",
    )
    parser.add_argument(
        "--live-speaker-ema-alpha",
        type=float,
        default=0.55,
        help="EMA weight for the newest live speaker probability snapshot.",
    )
    parser.add_argument(
        "--live-speaker-acquire-count",
        type=int,
        default=1,
        help="Consecutive known-speaker probes required before initially showing a live speaker.",
    )
    parser.add_argument(
        "--live-speaker-switch-count",
        type=int,
        default=1,
        help="Consecutive probes for a different speaker required before switching the live label.",
    )
    parser.add_argument(
        "--live-speaker-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, score the last live audio window against known speakers for fallback speaker highlighting.",
    )
    parser.add_argument(
        "--live-speaker-probe-interval-seconds",
        type=float,
        default=0.75,
        help="Seconds between fallback live-speaker probes.",
    )
    parser.add_argument(
        "--live-speaker-probe-release-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Seconds between cheap silence-release checks while a speaker is visible; "
            "0 reuses the embedding-probe interval."
        ),
    )
    parser.add_argument(
        "--live-speaker-probe-attack-interval-seconds",
        type=float,
        default=0.0,
        help="Optional faster probe interval while acquiring a speaker or resolving UNKNOWN; 0 disables.",
    )
    parser.add_argument(
        "--live-speaker-probe-window-seconds",
        type=float,
        default=1.0,
        help="Recent audio window scored by the fallback live-speaker probe.",
    )
    parser.add_argument(
        "--live-speaker-probe-context-window-seconds",
        type=float,
        default=0.0,
        help="Optional longer live-audio context window blended with the fast probe; 0 disables.",
    )
    parser.add_argument(
        "--live-speaker-probe-context-weight",
        type=float,
        default=0.0,
        help="Weight in [0,1] assigned to the optional longer live-speaker context embedding.",
    )
    parser.add_argument(
        "--live-speaker-tracker",
        choices=("classic", "bayes"),
        default="classic",
        help=(
            "Causal identity tracker. 'classic' blends the two embeddings before scoring; "
            "'bayes' keeps them independent and filters speaker state probabilistically."
        ),
    )
    parser.add_argument(
        "--live-speaker-open-set-tracklets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the output-only two-probe temporary identity overlay. "
            "It reuses the existing 0.7/1.5-second live embeddings and never "
            "changes final speaker memory."
        ),
    )
    parser.add_argument(
        "--live-speaker-open-set-preprofile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow open-set tracklets to run before the first final sentence "
            "speaker profile becomes available."
        ),
    )
    parser.add_argument(
        "--live-speaker-open-set-tracklet-preset",
        choices=(
            "short_history_hybrid_v1",
            "short_history_hybrid_v2_profile_contradiction",
        ),
        default="short_history_hybrid_v1",
        help="Versioned threshold preset for the open-set tracklet overlay.",
    )
    parser.add_argument("--live-speaker-bayes-temperature", type=float, default=0.10)
    parser.add_argument("--live-speaker-bayes-unknown-bias", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-profile-count-threshold", type=int, default=0)
    parser.add_argument("--live-speaker-bayes-low-profile-unknown-bias", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-high-profile-unknown-bias", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-profile-count-bias-slope", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-stay-probability", type=float, default=0.50)
    parser.add_argument("--live-speaker-bayes-prior-strength", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-evidence-strength", type=float, default=1.0)
    parser.add_argument("--live-speaker-bayes-switch-probability-margin", type=float, default=0.0)
    parser.add_argument(
        "--live-speaker-bayes-provisional-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Discover a causal unnamed speaker from unmatched live embeddings before a final sentence profile exists.",
    )
    parser.add_argument("--live-speaker-bayes-provisional-creation-count", type=int, default=2)
    parser.add_argument("--live-speaker-bayes-provisional-later-creation-count", type=int, default=0)
    parser.add_argument("--live-speaker-bayes-provisional-later-creation-profile-threshold", type=int, default=0)
    parser.add_argument("--live-speaker-bayes-provisional-creation-similarity-ceiling", type=float, default=0.20)
    parser.add_argument(
        "--live-speaker-bayes-provisional-boundary-creation-similarity-ceiling",
        type=float,
        default=-1.0,
        help="Optional relaxed new-speaker ceiling used only after a causal voice discontinuity.",
    )
    parser.add_argument(
        "--live-speaker-bayes-provisional-boundary-continuity",
        type=float,
        default=-1.0,
        help="Maximum incumbent-history similarity that marks a new-speaker boundary.",
    )
    parser.add_argument("--live-speaker-bayes-provisional-max-finalized-profiles", type=int, default=-1)
    parser.add_argument("--live-speaker-bayes-provisional-merge-min-similarity", type=float, default=0.25)
    parser.add_argument("--live-speaker-bayes-provisional-update-alpha", type=float, default=0.0)
    parser.add_argument(
        "--live-speaker-bayes-provisional-update-continuity",
        type=float,
        default=-1.0,
        help="Minimum short-term incumbent continuity required before adapting a provisional centroid.",
    )
    parser.add_argument(
        "--live-speaker-bayes-provisional-update-history-size",
        type=int,
        default=1,
        help="Recent confirmed short-window embeddings averaged into each provisional update target.",
    )
    parser.add_argument("--live-speaker-bayes-provisional-max-active-count", type=int, default=0)
    parser.add_argument("--live-speaker-bayes-provisional-pool-overflow-update-alpha", type=float, default=0.0)
    parser.add_argument("--live-speaker-bayes-provisional-scale-agreement", type=float, default=-1.0)
    parser.add_argument("--live-speaker-bayes-provisional-assignment-scale-agreement", type=float, default=-1.0)
    parser.add_argument("--live-speaker-bayes-incumbent-hold-scale-agreement", type=float, default=-1.0)
    parser.add_argument(
        "--live-speaker-bayes-incumbent-continuity",
        type=float,
        default=-1.0,
        help="Hold the incumbent through uncertain speech when its short-window history remains this similar.",
    )
    parser.add_argument(
        "--live-speaker-bayes-incumbent-continuity-history-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--live-speaker-bayes-incumbent-continuity-update-on-hold",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--live-speaker-bayes-boundary-short-only-continuity",
        type=float,
        default=-1.0,
        help="At a detected boundary, ignore the slower context window for this identity decision.",
    )
    parser.add_argument(
        "--live-speaker-bayes-boundary-residual-incumbent-alpha",
        type=float,
        default=0.0,
        help="At a detected boundary, subtract this fraction of the recent incumbent voice anchor from the short embedding.",
    )
    parser.add_argument(
        "--live-speaker-bayes-short-long-crossover-min-margin",
        type=float,
        default=-1.0,
        help="Enable causal short/long crossover switching at this short-window identity margin.",
    )
    parser.add_argument(
        "--live-speaker-bayes-short-long-crossover-min-similarity",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--live-speaker-bayes-short-long-crossover-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--live-speaker-bayes-short-long-differential-candidate-gain",
        type=float,
        default=-2.0,
        help="Optional minimum short-minus-long evidence for a crossover candidate.",
    )
    parser.add_argument(
        "--live-speaker-bayes-short-long-differential-incumbent-loss",
        type=float,
        default=-2.0,
        help="Optional minimum long-minus-short evidence decay for the incumbent.",
    )
    parser.add_argument("--live-speaker-bayes-provisional-temporal-consistency", type=float, default=-1.0)
    parser.add_argument(
        "--live-speaker-probe-hold-seconds",
        type=float,
        default=1.0,
        help="Seconds the browser keeps a fallback live-speaker highlight after a matching probe.",
    )
    parser.add_argument(
        "--live-speaker-probe-min-advance-seconds",
        type=float,
        default=0.75,
        help="Minimum playback advance before rescoring the fallback live-speaker probe window.",
    )
    parser.add_argument(
        "--live-speaker-probe-attack-min-advance-seconds",
        type=float,
        default=0.0,
        help="Optional faster minimum playback advance during attack cadence; 0 uses the attack interval.",
    )
    parser.add_argument(
        "--live-speaker-probe-min-speech-seconds",
        type=float,
        default=0.15,
        help="Minimum RMS-gated speech inside the probe window before embedding it.",
    )
    parser.add_argument(
        "--live-speaker-probe-speech-backend",
        choices=("rms", "vad"),
        default="rms",
        help="Speech gate used by live-speaker probe windows. 'vad' reuses the configured VAD backend.",
    )
    parser.add_argument(
        "--live-speaker-probe-silero-speech-threshold",
        type=float,
        default=-1.0,
        help="Silero threshold for live-speaker acquisition; negative reuses --vad-silero-speech-threshold.",
    )
    parser.add_argument(
        "--live-speaker-probe-vad-min-speech-seconds",
        type=float,
        default=-1.0,
        help="Minimum VAD speech for live-speaker acquisition; negative reuses --vad-min-speech-seconds.",
    )
    parser.add_argument(
        "--live-speaker-probe-release-silero-speech-threshold",
        type=float,
        default=-1.0,
        help="Independent Silero threshold for live-speaker release; negative reuses the acquisition threshold.",
    )
    parser.add_argument(
        "--live-speaker-probe-release-vad-min-speech-seconds",
        type=float,
        default=-1.0,
        help="Independent VAD speech duration for release; negative reuses the acquisition duration.",
    )
    parser.add_argument(
        "--live-speaker-probe-fast-release-window-seconds",
        type=float,
        default=0.0,
        help="Optional shorter VAD-only window for an additional conservative fast-silence release path.",
    )
    parser.add_argument(
        "--live-speaker-probe-fast-release-silero-speech-threshold",
        type=float,
        default=-1.0,
        help="Silero threshold for the optional fast release path.",
    )
    parser.add_argument(
        "--live-speaker-probe-fast-release-vad-min-speech-seconds",
        type=float,
        default=-1.0,
        help="Minimum detected speech duration for the optional fast release path.",
    )
    parser.add_argument(
        "--live-speaker-probe-clear-on-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the fallback live speaker when the recent audio window has no RMS-gated speech.",
    )
    parser.add_argument(
        "--live-speaker-clear-on-vad-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clear the fallback live speaker when the main VAD finalizes a sentence window after trailing silence.",
    )
    parser.add_argument(
        "--live-speaker-probe-clear-window-seconds",
        type=float,
        default=1.0,
        help="Recent audio duration checked for silence before clearing the fallback live speaker.",
    )
    parser.add_argument(
        "--live-speaker-probe-clear-silence-count",
        type=int,
        default=1,
        help="Clear the fallback live speaker after this many consecutive silent clear windows.",
    )
    parser.add_argument(
        "--live-speaker-probe-clear-unknown-count",
        type=int,
        default=2,
        help="Clear the fallback live speaker after this many consecutive speech probes score as UNKNOWN; use 0 to disable.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-clear-debounce-seconds",
        type=float,
        default=0.0,
        help="Delay UNKNOWN fallback-live-speaker clear events in the browser by this many seconds; 0 clears immediately.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-keepalive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the current fallback live speaker highlighted during pre-clear UNKNOWN probes.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-release-smoothing",
        choices=("none", "sma", "ema"),
        default="none",
        help="Smooth current-speaker versus UNKNOWN evidence before releasing the fallback live speaker.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-release-count",
        type=int,
        default=3,
        help="Number of recent UNKNOWN release samples used by SMA/EMA release smoothing.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-release-ema-alpha",
        type=float,
        default=0.5,
        help="EMA weight for the newest UNKNOWN release sample.",
    )
    parser.add_argument(
        "--live-speaker-probe-unknown-release-margin",
        type=float,
        default=0.0,
        help="Tolerance added to the current speaker probability before UNKNOWN release wins.",
    )
    parser.add_argument(
        "--live-speaker-provisional-new-speaker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit a temporary live speaker id for speech that does not yet match any known speaker.",
    )
    parser.add_argument(
        "--live-speaker-provisional-min-audio-seconds",
        type=float,
        default=1.0,
        help="Minimum live probe audio duration before creating a provisional live speaker.",
    )
    parser.add_argument(
        "--live-speaker-provisional-min-unknown-probability",
        type=float,
        default=0.5,
        help="Minimum unknown probability required before creating a provisional live speaker.",
    )
    parser.add_argument(
        "--live-speaker-weak-profile-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a stricter, similarity-based live assignment for very young known-speaker profiles.",
    )
    parser.add_argument(
        "--live-speaker-weak-profile-max-speech-seconds",
        type=float,
        default=2.5,
        help="Maximum accumulated profile speech seconds considered weak for live-speaker assist.",
    )
    parser.add_argument(
        "--live-speaker-weak-profile-min-similarity",
        type=float,
        default=0.40,
        help="Minimum top similarity for weak-profile live-speaker assist.",
    )
    parser.add_argument(
        "--live-speaker-weak-profile-min-margin",
        type=float,
        default=0.12,
        help="Minimum top-vs-runner-up margin for weak-profile live-speaker assist.",
    )
    parser.add_argument(
        "--live-speaker-weak-profile-max-unknown-probability",
        type=float,
        default=0.55,
        help="Maximum UNKNOWN probability allowed for weak-profile live-speaker assist.",
    )
    parser.add_argument(
        "--section-gap-new-speaker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a long media gap plus moderate similarity to create a new section speaker.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-min-gap-seconds",
        type=float,
        default=60.0,
        help="Minimum media-time gap since the matched speaker last ended before section-gap splitting.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-min-prior-speech-seconds",
        type=float,
        default=8.0,
        help="Minimum existing profile speech seconds required before section-gap splitting can clone it.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-min-duration-seconds",
        type=float,
        default=5.0,
        help="Minimum current sentence duration for section-gap new-speaker splitting.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-min-similarity",
        type=float,
        default=0.35,
        help="Minimum similarity to an old speaker for section-gap new-speaker splitting.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-max-similarity",
        type=float,
        default=0.58,
        help="Maximum similarity to an old speaker before section-gap splitting treats it as the same speaker.",
    )
    parser.add_argument(
        "--section-gap-new-speaker-min-margin",
        type=float,
        default=0.08,
        help="Minimum top-vs-runner-up margin for section-gap new-speaker splitting.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create a new speaker when a recent UNKNOWN sentence pairs with a longer weak existing-speaker match.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-max-gap-seconds",
        type=float,
        default=4.0,
        help="Maximum gap between a pending UNKNOWN sentence and the current sentence for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-min-unknown-duration-seconds",
        type=float,
        default=0.2,
        help="Minimum duration of the pending UNKNOWN sentence used for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-min-current-duration-seconds",
        type=float,
        default=2.5,
        help="Minimum current sentence duration for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-min-pair-similarity",
        type=float,
        default=0.45,
        help="Minimum embedding similarity between UNKNOWN and current sentence for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-max-existing-similarity",
        type=float,
        default=0.55,
        help="Maximum similarity to an existing speaker before pair-based new-speaker creation is suppressed.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-max-existing-margin",
        type=float,
        default=0.20,
        help="Maximum existing-speaker margin allowed for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--unknown-pair-new-speaker-min-unknown-probability",
        type=float,
        default=0.10,
        help="Minimum UNKNOWN probability on the current sentence for pair-based new-speaker creation.",
    )
    parser.add_argument(
        "--live-speaker-raw-change-snap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow strong unsmoothed live probabilities to switch away from the active speaker before EMA catches up.",
    )
    parser.add_argument(
        "--live-speaker-raw-change-min-probability",
        type=float,
        default=0.7,
        help="Minimum raw known-speaker probability required for a live speaker-change snap.",
    )
    parser.add_argument(
        "--live-speaker-raw-change-min-margin",
        type=float,
        default=0.25,
        help="Minimum raw probability lead over the active speaker required for a live speaker-change snap.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let fresh final sentence assignments seed the visible live speaker when no stronger live tag is active.",
    )
    parser.add_argument(
        "--live-speaker-highlight-transcript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow realtime transcript rows to drive the speaker-list live highlight when no fallback live-speaker probe is active.",
    )
    parser.add_argument(
        "--live-speaker-highlight-transcript-max-lag-seconds",
        type=float,
        default=-1.0,
        help="Maximum playback lag after a realtime transcript row end for that row to drive the speaker-list live highlight; negative disables the limit.",
    )
    parser.add_argument(
        "--live-speaker-highlight-transcript-override-min-probability",
        type=float,
        default=1.1,
        help="Minimum raw realtime transcript speaker probability needed to override an active fallback live-speaker highlight; values above 1 disable override.",
    )
    parser.add_argument(
        "--live-speaker-highlight-transcript-override-min-margin",
        type=float,
        default=0.0,
        help="Minimum raw probability lead over UNKNOWN needed for transcript speaker highlight override.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow fresh final sentence assignments to replace the current fallback live speaker.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-max-lag-seconds",
        type=float,
        default=1.25,
        help="Maximum playback lag after a final sentence end for emitting a live-speaker sentence hint.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-new-speaker-max-lag-seconds",
        type=float,
        default=1.25,
        help="Maximum playback lag for a newly created speaker's first live-speaker sentence hint.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-new-speaker-hold-seconds",
        type=float,
        default=-1.0,
        help="Optional hold duration for newly created speaker sentence hints; negative uses the normal hint hold.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-new-speaker-max-top-similarity",
        type=float,
        default=1.0,
        help="Only emit delayed new-speaker sentence hints when the new profile's top existing-speaker similarity is at or below this value.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-hold-seconds",
        type=float,
        default=0.3,
        help="Browser hold duration for live-speaker sentence hints.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-hold-through-sentence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When a final sentence is assigned before playback has passed its end, keep its live hint through that end plus the hint hold.",
    )
    parser.add_argument(
        "--live-speaker-sentence-hint-min-duration-seconds",
        type=float,
        default=0.0,
        help="Minimum final sentence duration required before it may emit a live-speaker sentence hint.",
    )
    parser.add_argument(
        "--realtime-preview-engine-options-json",
        default="",
        help="Extra JSON object merged into the RealtimeSTT Kroko engine options.",
    )
