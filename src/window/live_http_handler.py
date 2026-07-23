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


def _sanitize_upload_filename(filename: str) -> str:
    name = Path(unquote(str(filename or ""))).name.strip()
    if not name:
        name = "audio.wav"
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem or "audio"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    if not safe_stem:
        safe_stem = "audio"
    return f"{safe_stem[:96]}{suffix}"


def _audio_upload_extension(filename: str, content_type: str = "") -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in AUDIO_UPLOAD_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension(str(content_type or "").split(";", 1)[0].strip())
    if guessed and guessed.lower() in AUDIO_UPLOAD_EXTENSIONS:
        return guessed.lower()
    allowed = ", ".join(sorted(AUDIO_UPLOAD_EXTENSIONS))
    raise RuntimeError(f"Unsupported audio file type. Use one of: {allowed}.")



class Handler(BaseHTTPRequestHandler):
    server: "WindowServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            html = render_live_index(self._bootstrap_payload())
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/health":
            self._send_json({"ok": True, "ready": True, "service": "live-window"})
        elif path == "/events":
            self._serve_events()
        elif path == "/api/events":
            self._serve_public_events(parsed)
        elif path == "/media/video":
            self._serve_file(self.server.current_media().video_file)
        elif path == "/media/audio":
            self._serve_file(self.server.current_media().audio_file)
        elif re.fullmatch(r"/assets/flags/4x3/[a-z]{2}\.svg", path):
            flag_path = Path(__file__).resolve().parent / "assets" / path.removeprefix("/assets/")
            if flag_path.is_file():
                self._serve_file(flag_path)
            else:
                self.send_error(404)
        elif path == "/api/speakers":
            self._send_json({"ok": True, "speaker_state": self.server.controller.speaker_state()})
        elif path == "/api/people":
            self._send_json({"ok": True, "speaker_state": self.server.controller.speaker_state()})
        elif path == "/api/translation/status":
            self._send_json({"ok": True, "translation": self.server.translation.public_config()})
        elif path == "/api/meeting-intelligence/status":
            self._send_json(self.server.meeting_intelligence_status())
        elif path == "/api/meeting-intelligence/chat/job":
            query = parse_qs(parsed.query)
            self._send_json(self.server.meeting_chat_job(str((query.get("job_id") or [""])[0])))
        elif path == "/api/bootstrap":
            self._send_json({"ok": True, **self._bootstrap_payload()})
        elif path == "/api/live-observation-bindings":
            self._send_json(
                {
                    "ok": True,
                    **self.server.final_transcript_dom_snapshot_bindings(),
                }
            )
        elif re.fullmatch(r"/assets/web/live/[a-z][a-z0-9_]*\.(?:css|js)", path):
            name = path.removeprefix("/assets/web/")
            try:
                payload = read_web_asset(name)
            except FileNotFoundError:
                self.send_error(404)
            else:
                self._send_bytes(payload, web_asset_content_type(name))
        elif path == "/api/sessions":
            query = parse_qs(parsed.query)
            filter_mode = str((query.get("filter") or ["active"])[0])
            search = str((query.get("q") or [""])[0])
            self._send_json(self.server.list_saved_sessions(filter_mode, search))
        elif path == "/api/session/status":
            query = parse_qs(parsed.query)
            client_id = str((query.get("client_id") or [""])[0])
            self._send_json({"ok": True, "session": self.server.session_status(client_id)})
        elif path == "/api/meeting-intelligence/report":
            query = parse_qs(parsed.query)
            self._send_json(self.server.meeting_intelligence_report(str((query.get("session_id") or [""])[0])))
        else:
            self.send_error(404)

    def _bootstrap_payload(self) -> dict[str, Any]:
        media = self.server.current_media()
        language = get_language_config(getattr(self.server.args, "language", "en"))
        flag_country = language_flag_country_code(language.code)
        return {
            "source": media.url,
            "preset_videos": PRESET_YOUTUBE_VIDEOS,
            "speaker_colors": SPEAKER_COLORS,
            "language": {
                "code": language.code,
                "name": language.display_name,
                "flag_url": f"/assets/flags/4x3/{flag_country}.svg",
            },
            "translation": self.server.translation.public_config(),
            "meeting_intelligence": {
                "enabled": bool(self.server.meeting_intelligence_url),
            },
            "new_speaker_sensitivity": new_speaker_sensitivity_config(
                getattr(self.server.args, "new_speaker_sensitivity", 3)
            ),
            "speaker_refinement": self.server.controller.speaker_refinement_settings(),
            "live_speaker": {
                "assignment_enabled": bool(getattr(self.server.args, "live_speaker_assignment", True)),
                "unknown_clear_debounce_seconds": max(
                    0.0,
                    float(getattr(self.server.args, "live_speaker_probe_unknown_clear_debounce_seconds", 0.0)),
                ),
                "browser_observation_enabled": self.server.browser_live_observation_enabled,
                "final_transcript_dom_snapshot_required": (
                    self.server.final_transcript_dom_snapshot_required
                ),
                "browser_observation_interval_seconds": max(
                    0.02,
                    float(getattr(
                        self.server.args,
                        "browser_live_observation_interval_seconds",
                        DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS,
                    )),
                ),
                "highlight_transcript": bool(
                    getattr(self.server.args, "live_speaker_highlight_transcript", True)
                ),
                "transcript_highlight_max_lag_seconds": float(
                    getattr(self.server.args, "live_speaker_highlight_transcript_max_lag_seconds", -1.0)
                ),
                "transcript_override_min_probability": float(
                    getattr(self.server.args, "live_speaker_highlight_transcript_override_min_probability", 1.1)
                ),
                "transcript_override_min_margin": float(
                    getattr(self.server.args, "live_speaker_highlight_transcript_override_min_margin", 0.0)
                ),
                "session_lease_enabled": self.server.session_lease_enabled,
            },
            "speaker_library": self.server.controller.initial_speaker_state(),
        }

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/load-audio-file":
                self._handle_audio_file_upload()
                return
            payload = self._read_json_body()
            if path == "/api/translation/configure":
                self._require_session(payload)
                self._send_json({"ok": True, "translation": self.server.translation.configure(payload)})
            elif path == "/api/translation/browser-result":
                self._require_session(payload)
                self._send_json({
                    "ok": True,
                    "translation": self.server.translation.accept_browser_result(payload),
                })
            elif path == "/api/translation/browser-fallback":
                self._require_session(payload)
                self._send_json({
                    "ok": True,
                    "translation": self.server.translation.request_browser_fallback(payload),
                })
            elif path == "/api/sessions/create":
                self._send_json(self.server.create_saved_session(payload))
            elif path == "/api/sessions/open":
                self._send_json(self.server.open_saved_session(str(payload.get("session_id") or "")))
            elif path == "/api/sessions/rename":
                self._send_json(self.server.rename_saved_session(
                    str(payload.get("session_id") or ""),
                    str(payload.get("title") or ""),
                ))
            elif path == "/api/sessions/archive":
                self._send_json(self.server.archive_saved_session(str(payload.get("session_id") or "")))
            elif path == "/api/sessions/restore":
                self._send_json(self.server.restore_saved_session(str(payload.get("session_id") or "")))
            elif path == "/api/sessions/delete":
                self._send_json(self.server.delete_saved_session(str(payload.get("session_id") or "")))
            elif path == "/api/sessions/speakers/rename":
                self._send_json(self.server.rename_saved_session_speaker(
                    str(payload.get("session_id") or ""),
                    str(payload.get("speaker_id") or ""),
                    str(payload.get("name") or ""),
                ))
            elif path == "/api/sessions/people/link":
                self._send_json(self.server.link_saved_session_person(
                    str(payload.get("session_id") or ""),
                    str(payload.get("speaker_id") or ""),
                    person_id=str(payload.get("person_id") or ""),
                    person_name=str(payload.get("person_name") or ""),
                    expected_updated_at=str(payload.get("expected_updated_at") or ""),
                ))
            elif path == "/api/sessions/people/unlink":
                self._send_json(self.server.unlink_saved_session_person(
                    str(payload.get("session_id") or ""),
                    str(payload.get("speaker_id") or ""),
                ))
            elif path == "/api/sessions/corrections/reassign":
                raw_indexes = payload.get("indexes")
                if not isinstance(raw_indexes, list):
                    raise ValueError("indexes must be a list.")
                self._send_json(self.server.reassign_saved_session_rows(
                    str(payload.get("session_id") or ""),
                    [int(index) for index in raw_indexes],
                    str(payload.get("speaker_id") or ""),
                ))
            elif path == "/api/sessions/corrections/mark-correct":
                raw_indexes = payload.get("indexes")
                if not isinstance(raw_indexes, list):
                    raise ValueError("indexes must be a list.")
                self._send_json(self.server.mark_saved_session_rows_correct(
                    str(payload.get("session_id") or ""),
                    [int(index) for index in raw_indexes],
                ))
            elif path == "/api/meeting-intelligence/report":
                self._send_json(self.server.meeting_intelligence_report(str(payload.get("session_id") or "")))
            elif path == "/api/meeting-intelligence/generate":
                self._send_json(self.server.generate_meeting_intelligence(str(payload.get("session_id") or "")))
            elif path == "/api/meeting-intelligence/update-object":
                self._send_json(self.server.update_meeting_intelligence_object(
                    str(payload.get("session_id") or ""),
                    str(payload.get("object_id") or ""),
                    status=str(payload.get("status") or "") or None,
                    title=str(payload.get("title")) if "title" in payload else None,
                    body=str(payload.get("body")) if "body" in payload else None,
                ))
            elif path == "/api/meeting-intelligence/chat/scope":
                self._send_json(self.server.meeting_chat_scope(_string_list(payload.get("session_ids"))))
            elif path == "/api/meeting-intelligence/chat/ask-async":
                self._send_json(self.server.start_meeting_chat(
                    _string_list(payload.get("session_ids")),
                    str(payload.get("question") or ""),
                ))
            elif path == "/api/meeting-intelligence/chat/clear":
                self._send_json(self.server.clear_meeting_chat(_string_list(payload.get("session_ids"))))
            elif path == "/api/session/acquire":
                self._send_json(self.server.acquire_session(str(payload.get("client_id") or "")))
            elif path == "/api/session/heartbeat":
                self._send_json(self.server.heartbeat_session(
                    str(payload.get("session_token") or ""),
                    str(payload.get("client_id") or ""),
                ))
            elif path == "/api/session/release":
                self._send_json(self.server.release_session(
                    str(payload.get("session_token") or ""),
                    str(payload.get("reason") or "released"),
                    str(payload.get("client_id") or ""),
                ))
            elif path == "/api/start":
                session_token = self._require_session(payload)
                self.server.bus.emit("status", {"message": "Browser Start request received."})
                speaker_state = self.server.controller.start(StartSessionRequest(
                    session_id=str(payload.get("session_id") or ""),
                    source_title=str(payload.get("source_title") or ""),
                    processing_mode=str(payload.get("processing_mode") or "playback"),
                ))
                self.server.translation.begin_session(self.server.controller.current_session_id())
                self.server.mark_session_running(session_token)
                saved_session = self.server._save_current_session(status_label="Started", write_audio=False)
                self._send_json({
                    "ok": True,
                    "speaker_state": speaker_state,
                    "saved_session": saved_session,
                    "session": self.server.session_status(str(payload.get("client_id") or "")),
                })
            elif path == "/api/stop":
                session_token = self._require_session(payload)
                self.server.controller.stop()
                self._send_json(self.server.release_session(
                    session_token,
                    "stopped",
                    str(payload.get("client_id") or ""),
                ))
            elif path == "/api/load-url":
                self._require_session(payload)
                url = str(payload.get("url", "")).strip()
                if not url:
                    raise RuntimeError("Missing YouTube URL.")
                cache_only = bool(payload.get("cache_only", False)) or bool(getattr(self.server.args, "skip_download", False))
                media = self.server.load_media_url(url, skip_download=cache_only)
                self._send_json({
                    "ok": True,
                    "url": media.url,
                    "video_id": media.video_id,
                    "audio_file": str(media.audio_file),
                    "video_file": str(media.video_file),
                    "version": self.server.media_version,
                    "speaker_state": self.server.controller.speaker_state(),
                    "session": self.server.session_status(str(payload.get("client_id") or "")),
                })
            elif path == "/api/browser-stream":
                self._require_session(payload)
                url = str(payload.get("url", "")).strip()
                if not url:
                    raise RuntimeError("Missing YouTube URL.")
                media = self.server.start_browser_stream_url(url)
                self._send_json({
                    "ok": True,
                    "url": media.url,
                    "video_id": media.video_id,
                    "browser_stream": True,
                    "version": self.server.media_version,
                    "speaker_state": self.server.controller.speaker_state(),
                    "session": self.server.session_status(str(payload.get("client_id") or "")),
                })
            elif path == "/api/audio-chunk":
                self._require_session(payload)
                audio_b64 = str(payload.get("audio_b64", ""))
                sample_rate = int(payload.get("sample_rate", 16000))
                if not audio_b64:
                    raise RuntimeError("Missing audio chunk.")
                raw = base64.b64decode(audio_b64)
                if len(raw) % 4:
                    raise RuntimeError("Invalid float32 audio chunk length.")
                audio_chunk = np.frombuffer(raw, dtype=np.float32).copy()
                duration = self.server.controller.append_stream_audio(audio_chunk, sample_rate)
                self._send_json({"ok": True, "duration": duration})
            elif path == "/api/playback":
                self._require_session(payload)
                self.server.controller.set_playback_time(float(payload.get("seconds", 0.0)))
                self._send_json({"ok": True})
            elif path == "/api/live-observation":
                self._require_session(payload)
                count = self.server.record_browser_live_observation(
                    payload.get("samples", []),
                    payload.get("batch_sequence"),
                )
                self._send_json({"ok": True, "sample_count": count})
            elif path == "/api/live-observation-finish":
                self._require_session(payload)
                count = self.server.record_browser_live_observation(
                    payload.get("samples", []),
                    payload.get("batch_sequence"),
                )
                summary = self.server.finish_browser_live_observation(
                    str(payload.get("reason") or "done"),
                    payload.get("final_transcript_dom_snapshot"),
                )
                self._send_json({"ok": True, "sample_count": count, "summary": summary})
                if bool(getattr(self.server.args, "exit_after_browser_live_observation", False)):
                    threading.Thread(
                        target=self.server.shutdown,
                        name="world-tape-browser-finish-shutdown",
                        daemon=True,
                    ).start()
            elif path == "/api/settings":
                self._require_session(payload)
                response: dict[str, Any] = {"ok": True}
                if "new_speaker_sensitivity" in payload:
                    response["new_speaker_sensitivity"] = self.server.controller.set_new_speaker_sensitivity(
                        payload.get("new_speaker_sensitivity", getattr(self.server.args, "new_speaker_sensitivity", 3))
                    )
                speaker_refinement_keys = {
                    "speaker_refinement_unknown_tentative",
                    "speaker_refinement_unknown_commit",
                    "allow_speaker_reassignment",
                }
                speaker_refinement_updates = {
                    key: payload.get(key)
                    for key in speaker_refinement_keys
                    if key in payload
                }
                if speaker_refinement_updates:
                    response["speaker_refinement"] = self.server.controller.set_speaker_refinement_settings(
                        speaker_refinement_updates
                    )
                else:
                    response["speaker_refinement"] = self.server.controller.speaker_refinement_settings()
                response["session"] = self.server.session_status(str(payload.get("client_id") or ""))
                self._send_json(response)
            elif path == "/api/speakers/rename":
                self._require_session(payload)
                state = self.server.controller.rename_speaker(
                    str(payload.get("speaker_id", "")),
                    str(payload.get("name", "")),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/corrections/reassign":
                self._require_session(payload)
                raw_indexes = payload.get("indexes")
                if isinstance(raw_indexes, list):
                    result = self.server.controller.reassign_sentences(
                        [int(index) for index in raw_indexes],
                        str(payload.get("speaker_id") or ""),
                        update_memory=bool(payload.get("update_memory", True)),
                    )
                else:
                    result = self.server.controller.reassign_sentence(
                        int(payload.get("index")),
                        str(payload.get("speaker_id") or ""),
                        update_memory=bool(payload.get("update_memory", True)),
                    )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/corrections/mark-correct":
                self._require_session(payload)
                raw_indexes = payload.get("indexes")
                if isinstance(raw_indexes, list):
                    result = self.server.controller.mark_sentences_correct([int(index) for index in raw_indexes])
                else:
                    result = self.server.controller.mark_sentence_correct(int(payload.get("index")))
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/corrections/undo":
                self._require_session(payload)
                result = self.server.controller.undo_last_correction()
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/merge":
                self._require_session(payload)
                expected_source_count = payload.get("expected_source_sentence_count")
                expected_target_count = payload.get("expected_target_sentence_count")
                result = self.server.controller.merge_speakers(
                    str(payload.get("source_speaker_id") or ""),
                    str(payload.get("target_speaker_id") or ""),
                    update_memory=bool(payload.get("update_memory", True)),
                    expected_source_sentence_count=(
                        int(expected_source_count)
                        if expected_source_count is not None
                        else None
                    ),
                    expected_target_sentence_count=(
                        int(expected_target_count)
                        if expected_target_count is not None
                        else None
                    ),
                )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/delete":
                self._require_session(payload)
                expected_sentence_count = payload.get("expected_sentence_count")
                result = self.server.controller.delete_speaker(
                    str(payload.get("speaker_id") or ""),
                    update_memory=bool(payload.get("update_memory", True)),
                    expected_sentence_count=(
                        int(expected_sentence_count)
                        if expected_sentence_count is not None
                        else None
                    ),
                )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/remove-empty":
                self._require_session(payload)
                raw_speaker_ids = payload.get("speaker_ids") or []
                if isinstance(raw_speaker_ids, str):
                    raw_speaker_ids = [raw_speaker_ids]
                if not isinstance(raw_speaker_ids, list):
                    raise ValueError("speaker_ids must be a list.")
                result = self.server.controller.remove_empty_speakers(
                    [str(speaker_id) for speaker_id in raw_speaker_ids[:100]]
                )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/split":
                self._require_session(payload)
                raw_indexes = payload.get("sentence_indices")
                if not isinstance(raw_indexes, list):
                    raw_indexes = [payload.get("index")]
                result = self.server.controller.split_speaker(
                    str(payload.get("speaker_id") or ""),
                    [int(index) for index in raw_indexes],
                    name=str(payload.get("name") or ""),
                    update_memory=bool(payload.get("update_memory", True)),
                )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/clear":
                self._require_session(payload)
                state = self.server.controller.clear_speakers()
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/save":
                self._require_session(payload)
                state = self.server.controller.save_speaker_group(str(payload.get("name", "")))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/save-corrected":
                self._require_session(payload)
                state = self.server.controller.save_speaker_group(str(payload.get("name", "")))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/load":
                self._require_session(payload)
                state = self.server.controller.load_speaker_group(str(payload.get("name", "")))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/export":
                self._require_session(payload)
                group = self.server.controller.export_speaker_group_file(str(payload.get("name", "")))
                self._send_json({
                    "ok": True,
                    "group": group,
                    "speaker_state": self.server.controller.speaker_state(),
                    "session": self.server.session_status(str(payload.get("client_id") or "")),
                })
            elif path == "/api/speakers/import":
                self._require_session(payload)
                group = payload.get("group")
                state = self.server.controller.import_speaker_group_file(group if isinstance(group, dict) else {})
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/reference":
                self._require_session(payload)
                state = self.server.controller.add_manual_voice_sample(
                    str(payload.get("person_id") or ""),
                    str(payload.get("filename", "reference.wav")),
                    str(payload.get("audio_b64", "")),
                    label=str(payload.get("label") or ""),
                    source_type=str(payload.get("source_type") or "manual_upload"),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/create":
                self._require_session(payload)
                state = self.server.controller.create_person(str(payload.get("name") or ""))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/rename":
                self._require_session(payload)
                state = self.server.controller.rename_person(
                    str(payload.get("person_id") or ""),
                    str(payload.get("name") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/remember":
                self._require_session(payload)
                state = self.server.controller.remember_speaker_as_person(
                    str(payload.get("speaker_id") or ""),
                    str(payload.get("name") or ""),
                    str(payload.get("person_id") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/confirm":
                self._require_session(payload)
                state = self.server.controller.confirm_speaker_person(
                    str(payload.get("speaker_id") or ""),
                    str(payload.get("person_id") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/reject":
                self._require_session(payload)
                state = self.server.controller.reject_speaker_person(
                    str(payload.get("speaker_id") or ""),
                    str(payload.get("person_id") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/unlink":
                self._require_session(payload)
                state = self.server.controller.unlink_speaker_person(
                    str(payload.get("speaker_id") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/recognition":
                self._require_session(payload)
                state = self.server.controller.set_person_recognition(
                    str(payload.get("person_id") or ""),
                    bool(payload.get("enabled", True)),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/policy":
                self._require_session(payload)
                updates = payload.get("recognition_policy")
                if not isinstance(updates, dict):
                    raise ValueError("recognition_policy must be an object.")
                state = self.server.controller.set_person_recognition_policy(
                    str(payload.get("person_id") or ""),
                    updates,
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/sample/add":
                self._require_session(payload)
                state = self.server.controller.add_manual_voice_sample(
                    str(payload.get("person_id") or ""),
                    str(payload.get("filename") or "voice-sample.wav"),
                    str(payload.get("audio_b64") or ""),
                    label=str(payload.get("label") or ""),
                    source_type=str(payload.get("source_type") or "manual_upload"),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/sample/state":
                self._require_session(payload)
                state = self.server.controller.set_voice_sample_enabled(
                    str(payload.get("person_id") or ""),
                    str(payload.get("sample_id") or ""),
                    bool(payload.get("enabled", True)),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/sample/label":
                self._require_session(payload)
                state = self.server.controller.label_voice_sample(
                    str(payload.get("person_id") or ""),
                    str(payload.get("sample_id") or ""),
                    str(payload.get("label") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/sample/delete":
                self._require_session(payload)
                state = self.server.delete_person_voice_sample(
                    str(payload.get("person_id") or ""),
                    str(payload.get("sample_id") or ""),
                )
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/expected":
                self._require_session(payload)
                raw_ids = payload.get("person_ids")
                if raw_ids is not None and not isinstance(raw_ids, list):
                    raise ValueError("person_ids must be a list or null.")
                state = self.server.controller.set_expected_people(raw_ids)
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/forget-voice":
                self._require_session(payload)
                state = self.server.forget_person_voice(str(payload.get("person_id") or ""))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/people/delete":
                self._require_session(payload)
                state = self.server.delete_person(str(payload.get("person_id") or ""))
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            else:
                self.send_error(404)
        except SessionLeaseError as exc:
            self._send_json({"error": str(exc), "session": exc.session}, status=exc.status)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_audio_file_upload(self) -> None:
        self._require_session({})
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        filename = self.headers.get("X-Whospeaks-Filename") or "audio.wav"
        media, display_name, byte_count = self.server.load_audio_upload(
            filename=filename,
            content_type=self.headers.get("Content-Type") or "",
            source=self.rfile,
            length=length,
        )
        self._send_json({
            "ok": True,
            "url": media.url,
            "video_id": media.video_id,
            "audio_file": str(media.audio_file),
            "video_file": str(media.video_file),
            "display_name": display_name,
            "size_bytes": byte_count,
            "version": self.server.media_version,
            "speaker_state": self.server.controller.speaker_state(),
            "session": self.server.session_status(self.headers.get("X-Whospeaks-Client") or ""),
        })

    def _require_session(self, payload: dict[str, Any]) -> str:
        token = str(payload.get("session_token") or self.headers.get("X-Whospeaks-Session") or "")
        client_id = str(payload.get("client_id") or self.headers.get("X-Whospeaks-Client") or "")
        self.server.require_session(token, client_id)
        return token

    def _serve_events(self) -> None:
        subscriber = self.server.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                try:
                    event, payload = subscriber.get(timeout=10)
                    message = f"event: {event}\ndata: {payload}\n\n"
                except queue.Empty:
                    message = ": heartbeat\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self.server.bus.unsubscribe(subscriber)

    def _serve_public_events(self, parsed: Any) -> None:
        query = parse_qs(parsed.query)
        include_snapshot = str((query.get("snapshot") or ["1"])[0]).strip().lower() not in {"0", "false", "no"}
        normalizer = PublicEventNormalizer(session_id=self.server.public_event_session_id)
        subscriber = self.server.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            if include_snapshot:
                for envelope in normalizer.speaker_snapshot(self.server.controller.speaker_state()):
                    self._write_public_event(envelope)
            while True:
                try:
                    event, payload_json = subscriber.get(timeout=10)
                    payload = json.loads(payload_json)
                    for envelope in normalizer.normalize(event, payload):
                        self._write_public_event(envelope)
                    continue
                except queue.Empty:
                    message = ": heartbeat\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self.server.bus.unsubscribe(subscriber)

    def _write_public_event(self, envelope: dict[str, Any]) -> None:
        event_type = str(envelope.get("type") or "message")
        event_id = str(envelope.get("id") or "")
        message = f"id: {event_id}\nevent: {event_type}\ndata: {json_dumps(envelope)}\n\n"
        self.wfile.write(message.encode("utf-8"))
        self.wfile.flush()

    def _serve_file(self, path: Path) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        byte_range = parse_range_header(self.headers.get("Range"), size)
        if byte_range is None:
            start, end = 0, size - 1
            self.send_response(200)
        else:
            start, end = byte_range
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        length = end - start + 1
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(json_dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)




def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def parse_range_header(header: str | None, file_size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    value = header[len("bytes="):].split(",", 1)[0].strip()
    if "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    if start < 0 or start >= file_size or end < start:
        return None
    return start, min(end, file_size - 1)
