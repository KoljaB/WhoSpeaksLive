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




def _absolute_path_preserving_symlinks(path: Path) -> Path:
    """Return an absolute path without dereferencing venv executable symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))



def _argv_has_option(argv: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(item == option or item.startswith(option_prefix) for item in argv)



from window.window_cli_server_media import add_server_media_arguments
from window.window_cli_asr_language import add_asr_language_arguments
from window.window_cli_speakers import add_embedding_speaker_arguments
from window.window_cli_live_speaker import add_preview_live_speaker_arguments
from window.window_cli_validation import add_validation_arguments
from window.window_runtime_config import WindowConfig


def parse_args(argv: list[str] | None = None) -> WindowConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    preview_model_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model")
    preview_model_path_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model-path")
    preview_model_dir_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model-dir")
    preview_model_preset_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model-preset")
    language_was_explicit = _argv_has_option(raw_argv, "--language")
    language_was_from_env = bool(os.environ.get("WHOSPEAKS_LANGUAGE") or os.environ.get("WHOSPEAKS_ASR_LANGUAGE"))
    default_vad_model_path = default_silero_vad_model_path()
    parser = argparse.ArgumentParser(description="Growing-window faster-whisper speaker diarization GUI.")
    add_server_media_arguments(parser)
    add_asr_language_arguments(parser, default_vad_model_path=default_vad_model_path)
    add_embedding_speaker_arguments(parser)
    add_preview_live_speaker_arguments(parser)
    add_validation_arguments(parser)
    args = parser.parse_args(raw_argv)
    if not language_was_explicit and not language_was_from_env:
        inferred_language = infer_language_from_kroko_model_name(args.realtime_preview_model)
        if inferred_language is not None:
            args.language = inferred_language
    args.sentence_tokenizer = default_sentence_tokenizer(args.language, args.sentence_tokenizer)
    args.sentence_language = default_sentence_language(args.language)
    args.realtime_preview_language = args.language
    if not 1 <= int(args.translation_max_targets) <= 16:
        parser.error("--translation-max-targets must be between 1 and 16.")
    translation_targets: list[str] = []
    for target in args.translation_target_language:
        if target != args.language and target not in translation_targets:
            translation_targets.append(target)
    if len(translation_targets) > int(args.translation_max_targets):
        parser.error("Initial translation targets exceed --translation-max-targets.")
    args.translation_target_language = translation_targets
    args.translation_queue_size = max(1, int(args.translation_queue_size))
    args.translation_context_sentences = max(0, int(args.translation_context_sentences))
    args.translation_timeout_seconds = max(1.0, float(args.translation_timeout_seconds))
    try:
        args.realtime_preview_engine = normalize_preview_engine(args.realtime_preview_engine)
        language_error = preview_language_error(args.realtime_preview_engine, args.language)
        if language_error:
            parser.error(language_error)
        if preview_model_path_was_explicit and preview_model_dir_was_explicit:
            parser.error("--realtime-preview-model-path and --realtime-preview-model-dir cannot be used together.")
        if args.realtime_preview_engine == "kroko_onnx":
            if preview_model_dir_was_explicit:
                parser.error("--realtime-preview-model-dir is only valid for sherpa_onnx realtime preview.")
            requested_preset = args.realtime_preview_model_preset or DEFAULT_KROKO_PREVIEW_MODEL_PRESET
            args.realtime_preview_model_preset = normalize_preview_model_preset("kroko_onnx", requested_preset)
            if args.realtime_preview_model is None:
                args.realtime_preview_model = default_preview_model(
                    "kroko_onnx", args.language, args.realtime_preview_model_preset
                )
            else:
                args.realtime_preview_model_preset = "custom"
            args.realtime_preview_model_dir = None
            if args.realtime_preview_startup_timeout_seconds is None:
                args.realtime_preview_startup_timeout_seconds = default_kroko_preview_startup_timeout_seconds(
                    args.realtime_preview_model_preset
                )
        elif args.realtime_preview_engine == "sherpa_onnx":
            if preview_model_path_was_explicit:
                parser.error("--realtime-preview-model-path is only valid for Kroko; use --realtime-preview-model-dir.")
            requested_preset = args.realtime_preview_model or args.realtime_preview_model_preset or DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET
            args.realtime_preview_model_preset = normalize_preview_model_preset("sherpa_onnx", requested_preset)
            args.realtime_preview_model = args.realtime_preview_model_preset
            if args.realtime_preview_startup_timeout_seconds is None:
                args.realtime_preview_startup_timeout_seconds = sherpa_onnx_model_preset(
                    args.realtime_preview_model_preset
                ).startup_timeout_seconds
        else:
            args.realtime_preview_model = ""
            args.realtime_preview_model_preset = ""
            args.realtime_preview_model_path = None
            args.realtime_preview_model_dir = None
            if args.realtime_preview_startup_timeout_seconds is None:
                args.realtime_preview_startup_timeout_seconds = 12.0
    except ValueError as exc:
        parser.error(str(exc))
    args.work_dir = args.work_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.session_dir = args.session_dir.resolve()
    args.embedding_python = _absolute_path_preserving_symlinks(args.embedding_python)
    args.speaker_library_dir = args.speaker_library_dir.resolve()
    args.validation_canonical = args.validation_canonical.resolve()
    args.validation_output = args.validation_output.resolve()
    if args.validation_trace_output is not None:
        args.validation_trace_output = args.validation_trace_output.resolve()
    if args.browser_live_observation_output is not None:
        args.browser_live_observation_output = args.browser_live_observation_output.resolve()
    if args.download_root is not None:
        args.download_root = args.download_root.resolve()
    if args.realtime_preview_engine == "kroko_onnx" and args.realtime_preview_model_path is None and args.realtime_preview_model:
        use_env_model_path = not (
            preview_model_was_explicit
            or preview_model_path_was_explicit
            or preview_model_preset_was_explicit
        )
        args.realtime_preview_model_path = default_kroko_preview_model_path(
            args.realtime_preview_model,
            use_env=use_env_model_path,
        )
    if args.realtime_preview_model_path is not None:
        args.realtime_preview_model_path = args.realtime_preview_model_path.resolve()
    if args.realtime_preview_engine == "sherpa_onnx":
        args.realtime_preview_model_dir = (
            args.realtime_preview_model_dir or default_sherpa_onnx_model_dir(args.realtime_preview_model_preset)
        ).resolve()
    if args.realtime_preview_download_root is not None:
        args.realtime_preview_download_root = args.realtime_preview_download_root.resolve()
    if args.realtime_preview_python is None:
        args.realtime_preview_python = (
            DEFAULT_KROKO_PREVIEW_PYTHON if args.realtime_preview_engine == "kroko_onnx" else Path(sys.executable)
        )
    args.realtime_preview_python = _absolute_path_preserving_symlinks(args.realtime_preview_python)
    if args.realtime_preview_realtimestt_root is not None:
        args.realtime_preview_realtimestt_root = args.realtime_preview_realtimestt_root.resolve()
    if args.vad_silero_onnx_model_path is not None:
        args.vad_silero_onnx_model_path = args.vad_silero_onnx_model_path.resolve()
    apply_preview_timing_defaults(args)
    if args.new_speaker_sensitivity is not None:
        apply_new_speaker_sensitivity(args, args.new_speaker_sensitivity)
    else:
        args.new_speaker_sensitivity = 3
        args.new_speaker_sensitivity_label = NEW_SPEAKER_SENSITIVITY_PRESETS[3]["label"]
    if not args.live_speaker_assignment:
        args.live_speaker_probe = False
        args.live_speaker_sentence_hint = False
        args.live_speaker_highlight_transcript = False
        args.live_speaker_verify_on_change = False
        args.live_speaker_raw_change_snap = False
        args.live_speaker_provisional_new_speaker = False
        args.live_speaker_weak_profile_assist = False
    return WindowConfig.from_namespace(args)
