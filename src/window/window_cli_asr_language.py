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




def add_asr_language_arguments(parser: argparse.ArgumentParser, *, default_vad_model_path: Path) -> None:
    parser.add_argument("--model", default="large-v2")
    parser.add_argument(
        "--language",
        type=language_arg,
        default=default_language_code(),
        help="Realtime language for final ASR, Kroko preview model selection, and sentence splitting.",
    )
    parser.add_argument(
        "--translation-provider",
        choices=(
            "off",
            "sidecar",
            "transformers",
            "deepl",
            "google_cloud",
            "azure_translator",
            "libretranslate",
            "openai_compatible",
            "mock",
        ),
        default=os.environ.get("WHOSPEAKS_TRANSLATION_PROVIDER", "off"),
        help=(
            "Optional sentence translation backend. 'sidecar' is recommended for local models so its "
            "dependencies and GPU memory stay isolated from transcription."
        ),
    )
    parser.add_argument(
        "--translation-target-language",
        action="append",
        type=language_arg,
        default=[],
        help="Initial translation target language. Repeat for simultaneous targets; targets can also be changed in the browser.",
    )
    parser.add_argument(
        "--translation-browser-preferred",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer Chrome's on-device Translator API and use the selected server provider as fallback.",
    )
    parser.add_argument(
        "--translation-max-targets",
        type=int,
        default=4,
        help="Maximum simultaneous target languages accepted from the browser.",
    )
    parser.add_argument(
        "--translation-base-url",
        default=os.environ.get("WHOSPEAKS_TRANSLATION_BASE_URL", ""),
        help="Optional provider endpoint override; defaults are used for managed translation APIs.",
    )
    parser.add_argument(
        "--translation-model-profile",
        choices=("translate-gemma-4b", "nllb-200-600m", "madlad-400-3b"),
        default="translate-gemma-4b",
        help="Local translation model profile. TranslateGemma is the recommended quality-first default.",
    )
    parser.add_argument("--translation-model", default="", help="Optional model id override for the selected provider/profile.")
    parser.add_argument("--translation-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--translation-dtype", default="auto")
    parser.add_argument(
        "--translation-api-key-env",
        default="",
        help="Environment variable containing the provider secret; blank uses the provider-specific default.",
    )
    parser.add_argument(
        "--translation-region",
        default="",
        help="Optional provider region, currently used by Azure Translator.",
    )
    parser.add_argument(
        "--translation-timeout-seconds",
        type=float,
        default=600.0,
        help="Provider request timeout. The long default allows a sidecar's first model download/load to finish.",
    )
    parser.add_argument("--translation-queue-size", type=int, default=256)
    parser.add_argument("--translation-context-sentences", type=int, default=2)
    parser.add_argument(
        "--sentence-tokenizer",
        type=sentence_tokenizer_arg,
        default=None,
        help="Sentence tokenizer for stream2sentence. Defaults to the language-specific realtime choice.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--download-root", type=Path, default=default_faster_whisper_download_root())
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--cpu-alignment-model",
        choices=("tiny", "base"),
        default="base",
        help="Whisper model used only to align the fixed CPU streaming-ASR transcript. Base is the quality default.",
    )
    parser.add_argument(
        "--cpu-alignment-threads",
        type=int,
        default=2,
        help="CPU threads reserved for final forced alignment; two keeps desktop load bounded.",
    )
    parser.add_argument("--cpu-alignment-compute-type", default="int8")
    parser.add_argument(
        "--cpu-alignment-min-probability",
        type=float,
        default=0.15,
        help="Reject forced alignment below this mean token probability and use native timestamp fallback.",
    )
    parser.add_argument(
        "--fast-asr-batch-size",
        type=int,
        default=16,
        help="Maximum faster-whisper batch size used for offline fast processing.",
    )
    parser.add_argument(
        "--fast-embedding-queue-size",
        type=int,
        default=24,
        help="Maximum number of sentence-audio chunks retained while fast embeddings catch up.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.7,
        help="Fixed delay between transcription passes, also used as cooldown after a successful sentence split. 0 runs continuously with no overlap.",
    )
    parser.add_argument(
        "--min-playback-advance-seconds",
        type=float,
        default=0.75,
        help="Minimum browser playback-time advance required before starting the next pass.",
    )
    parser.add_argument("--min-window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--unstable-tail-seconds",
        type=float,
        default=1.35,
        help="Minimum seconds after a candidate sentence's last word before committing a punctuation-ending sentence.",
    )
    parser.add_argument(
        "--vad-sentence-splitting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local VAD to force-finalize a window after trailing silence.",
    )
    parser.add_argument(
        "--vad-backend",
        choices=("silero", "rms"),
        default="silero",
        help="VAD backend for sentence-window finalization.",
    )
    parser.add_argument(
        "--vad-silero-backend",
        choices=("auto", "raw_onnx_ifless", "raw_onnx", "official_onnx", "pytorch_cpu"),
        default=default_silero_vad_backend(default_vad_model_path),
        help="Silero implementation used when --vad-backend silero is active.",
    )
    parser.add_argument(
        "--vad-silero-onnx-model-path",
        type=Path,
        default=default_vad_model_path,
        help="Path to a Silero ONNX model file. Defaults to the local RealtimeSTT model cache when available.",
    )
    parser.add_argument(
        "--vad-silero-onnx-threads",
        type=int,
        default=2,
        help="CPU threads used by the raw ONNX Silero VAD session.",
    )
    parser.add_argument(
        "--vad-silero-speech-threshold",
        type=float,
        default=0.5,
        help="Silero speech probability required to mark a 512-sample chunk as speech.",
    )
    parser.add_argument(
        "--vad-silence-seconds",
        type=float,
        default=1.1,
        help="Trailing silence required before VAD forces the current window to finalize.",
    )
    parser.add_argument(
        "--vad-final-window-post-silence-seconds",
        type=float,
        default=0.75,
        help="On a VAD split, transcribe the previous final window only this far after VAD speech end.",
    )
    parser.add_argument(
        "--vad-next-window-start-silence-seconds",
        type=float,
        default=0.7,
        help="On a VAD split, advance the next window start to at least this far after VAD speech end.",
    )
    parser.add_argument(
        "--vad-speech-rms-threshold",
        type=float,
        default=0.003,
        help="RMS threshold used by --vad-backend rms or by the RMS fallback.",
    )
    parser.add_argument(
        "--vad-frame-seconds",
        type=float,
        default=0.03,
        help="Frame size used by the local energy VAD.",
    )
    parser.add_argument(
        "--vad-merge-gap-seconds",
        type=float,
        default=0.18,
        help="Short silence gaps below this length are merged into surrounding speech.",
    )
    parser.add_argument(
        "--vad-min-speech-seconds",
        type=float,
        default=0.25,
        help="Minimum detected speech in a window before VAD can trigger a split.",
    )
    parser.add_argument(
        "--vad-gate-secondary-backend",
        choices=("off", "webrtc"),
        default="webrtc",
        help="Realtime-safe secondary VAD required to confirm ASR/preview speech gates.",
    )
    parser.add_argument(
        "--vad-gate-webrtc-mode",
        type=int,
        default=3,
        help="WebRTC VAD aggressiveness for ASR/preview gate confirmation (0-3).",
    )
    parser.add_argument(
        "--vad-gate-min-consensus-seconds",
        type=float,
        default=0.12,
        help="Minimum secondary-VAD overlap required to accept a primary VAD speech span.",
    )
    parser.add_argument(
        "--vad-gate-min-consensus-ratio",
        type=float,
        default=0.05,
        help="Minimum secondary-VAD overlap ratio required to accept a primary VAD speech span.",
    )
    parser.add_argument(
        "--asr-vad-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before final ASR, trim leading/trailing non-speech and transcribe one padded "
            "speech-bounded clip instead of the full music/silence-containing window."
        ),
    )
    parser.add_argument(
        "--asr-vad-gate-pre-padding-seconds",
        type=float,
        default=0.20,
        help="Audio kept before each VAD speech island sent to final ASR.",
    )
    parser.add_argument(
        "--asr-vad-gate-post-padding-seconds",
        type=float,
        default=0.35,
        help="Audio kept after each VAD speech island sent to final ASR.",
    )
    parser.add_argument(
        "--asr-vad-gate-merge-gap-seconds",
        type=float,
        default=0.85,
        help="When internal gap cutting is enabled, merge padded ASR speech islands separated by at most this many seconds.",
    )
    parser.add_argument(
        "--asr-vad-gate-min-clip-seconds",
        type=float,
        default=0.20,
        help="Drop padded ASR speech clips shorter than this duration.",
    )
    parser.add_argument(
        "--asr-vad-gate-cut-internal-gaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: cut long non-speech gaps inside a final ASR window. Disabled by default to avoid splitting sentences.",
    )
    parser.add_argument(
        "--asr-no-speech-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop ASR segments whose Whisper no_speech_prob is above the configured threshold.",
    )
    parser.add_argument(
        "--asr-no-speech-prob-threshold",
        type=float,
        default=0.65,
        help="Whisper no_speech_prob threshold above which ASR segment words are discarded.",
    )
    parser.add_argument(
        "--asr-no-speech-hard-threshold",
        type=float,
        default=0.85,
        help="Whisper no_speech_prob threshold above which even very short ASR segments are discarded.",
    )
    parser.add_argument(
        "--asr-no-speech-keep-short-max-words",
        type=int,
        default=2,
        help="Keep ASR segments at or above the no_speech_prob threshold when they have at most this many words and stay below the hard threshold.",
    )
    parser.add_argument(
        "--asr-no-speech-keep-short-max-seconds",
        type=float,
        default=0.45,
        help="Keep ASR segments at or above the no_speech_prob threshold when they are at most this long and stay below the hard threshold.",
    )
    parser.add_argument(
        "--sentence-boundary-pre-padding-seconds",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
        help="Audio kept before the next word when cutting between two consecutive completed sentences.",
    )
    parser.add_argument(
        "--sentence-boundary-post-padding-seconds",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
        help="Audio kept after the last word when cutting between two consecutive completed sentences.",
    )
    parser.add_argument(
        "--sentence-boundary-gap-ratio",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
        help="For tight word gaps, fraction of the gap assigned to the previous sentence.",
    )
    parser.add_argument(
        "--final-flush-epsilon-seconds",
        type=float,
        default=0.5,
        help="Treat playback as ended when browser time is within this many seconds of audio duration.",
    )
    parser.add_argument(
        "--start-warmup-stale-seconds",
        type=float,
        default=10.0,
        help="Refresh ASR and embedding warmups on Start when the previous runtime warmup is older than this. Use 0 to always refresh.",
    )
    parser.add_argument(
        "--startup-warmup-before-url",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm ASR, embeddings, and VAD before printing/serving the browser URL.",
    )
