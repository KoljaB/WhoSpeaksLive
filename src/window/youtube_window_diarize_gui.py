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
    normalize_kroko_preview_model_preset,
)
from window.language_config import (  # noqa: E402
    default_language_code,
    default_sentence_language,
    default_sentence_tokenizer,
    infer_language_from_kroko_model_name,
    is_kroko_preview_language,
    kroko_preview_model_name,
    language_arg,
    sentence_tokenizer_arg,
)
from window.window_diarizer import WindowDiarizer  # noqa: E402
from window.window_events import EventBus, RecordingEventBus  # noqa: E402
from window.window_gui_html import HTML  # noqa: E402
from window.window_preview import infer_kroko_preview_chunk_seconds  # noqa: E402
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

class SessionLeaseError(RuntimeError):
    def __init__(self, message: str, session: dict[str, Any], status: int = 409) -> None:
        super().__init__(message)
        self.session = session
        self.status = status


class SessionLease:
    def __init__(
        self,
        idle_timeout_seconds: float = 120.0,
        heartbeat_timeout_seconds: float = 45.0,
        completed_release_delay_seconds: float = 10.0,
        max_run_seconds: float = 900.0,
    ) -> None:
        self.idle_timeout_seconds = max(1.0, float(idle_timeout_seconds))
        self.heartbeat_timeout_seconds = max(5.0, float(heartbeat_timeout_seconds))
        self.completed_release_delay_seconds = max(0.0, float(completed_release_delay_seconds))
        self.max_run_seconds = max(30.0, float(max_run_seconds))
        self._lock = threading.Lock()
        self._token = ""
        self._client_id = ""
        self._created_at = 0.0
        self._last_seen_at = 0.0
        self._run_started_at: float | None = None
        self._completed_at: float | None = None
        self._last_release_reason = ""
        self._last_release_at = 0.0
        self._waiting_clients: dict[str, float] = {}

    def _prune_waiters_locked(self, now: float) -> None:
        stale_before = now - 120.0
        self._waiting_clients = {
            client_id: seen_at
            for client_id, seen_at in self._waiting_clients.items()
            if seen_at >= stale_before
        }

    def _state_locked(self, now: float, client_id: str = "") -> dict[str, Any]:
        self._prune_waiters_locked(now)
        active = bool(self._token)
        is_owner = active and bool(client_id) and client_id == self._client_id
        expires_in: float | None = None
        heartbeat_expires_in: float | None = None
        idle_expires_in: float | None = None
        completed_expires_in: float | None = None
        hard_expires_in: float | None = None
        release_reason = ""
        if active:
            heartbeat_expires_in = self.heartbeat_timeout_seconds - (now - self._last_seen_at)
            if self._completed_at is not None:
                completed_expires_in = self.completed_release_delay_seconds - (now - self._completed_at)
                expires_in = completed_expires_in
                release_reason = "completed"
            elif self._run_started_at is None:
                idle_expires_in = self.idle_timeout_seconds - (now - self._created_at)
                expires_in = min(idle_expires_in, heartbeat_expires_in)
                release_reason = "idle"
            else:
                hard_expires_in = self.max_run_seconds - (now - self._run_started_at)
                expires_in = min(hard_expires_in, heartbeat_expires_in)
                release_reason = "timeout"
        return {
            "active": active,
            "is_owner": is_owner,
            "running": active and self._run_started_at is not None and self._completed_at is None,
            "completed": active and self._completed_at is not None,
            "client_id": self._client_id if active else "",
            "waiter_count": len(self._waiting_clients),
            "expires_in_seconds": round(max(0.0, expires_in), 1) if expires_in is not None else None,
            "heartbeat_expires_in_seconds": (
                round(max(0.0, heartbeat_expires_in), 1) if heartbeat_expires_in is not None else None
            ),
            "idle_expires_in_seconds": round(max(0.0, idle_expires_in), 1) if idle_expires_in is not None else None,
            "completed_expires_in_seconds": (
                round(max(0.0, completed_expires_in), 1) if completed_expires_in is not None else None
            ),
            "hard_expires_in_seconds": round(max(0.0, hard_expires_in), 1) if hard_expires_in is not None else None,
            "release_reason": release_reason,
            "last_release_reason": self._last_release_reason,
            "last_release_at": round(self._last_release_at, 3) if self._last_release_at else None,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "completed_release_delay_seconds": self.completed_release_delay_seconds,
            "max_run_seconds": self.max_run_seconds,
        }

    def _expired_reason_locked(self, now: float) -> str:
        if not self._token:
            return ""
        if now - self._last_seen_at > self.heartbeat_timeout_seconds:
            return "heartbeat timeout"
        if self._completed_at is not None and now - self._completed_at > self.completed_release_delay_seconds:
            return "completed"
        if self._run_started_at is None and now - self._created_at > self.idle_timeout_seconds:
            return "idle timeout"
        if self._run_started_at is not None and now - self._run_started_at > self.max_run_seconds:
            return "time limit"
        return ""

    def expire_if_needed(self) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            reason = self._expired_reason_locked(now)
            if not reason:
                return None
            was_running = self._run_started_at is not None and self._completed_at is None
            self._release_locked(now, reason)
            return {"reason": reason, "was_running": was_running}

    def _release_locked(self, now: float, reason: str) -> None:
        self._token = ""
        self._client_id = ""
        self._created_at = 0.0
        self._last_seen_at = 0.0
        self._run_started_at = None
        self._completed_at = None
        self._last_release_reason = reason
        self._last_release_at = now

    def status(self, client_id: str = "") -> dict[str, Any]:
        self.expire_if_needed()
        now = time.monotonic()
        with self._lock:
            return self._state_locked(now, client_id)

    def acquire(self, client_id: str) -> dict[str, Any]:
        self.expire_if_needed()
        now = time.monotonic()
        normalized_client_id = str(client_id or "").strip()[:120]
        if not normalized_client_id:
            normalized_client_id = uuid.uuid4().hex
        with self._lock:
            if self._token and self._client_id != normalized_client_id:
                self._waiting_clients[normalized_client_id] = now
                return {
                    "ok": False,
                    "acquired": False,
                    "session": self._state_locked(now, normalized_client_id),
                }
            if not self._token:
                self._token = uuid.uuid4().hex
                self._client_id = normalized_client_id
                self._created_at = now
                self._run_started_at = None
                self._completed_at = None
            self._last_seen_at = now
            self._waiting_clients.pop(normalized_client_id, None)
            return {
                "ok": True,
                "acquired": True,
                "session_token": self._token,
                "session": self._state_locked(now, normalized_client_id),
            }

    def authorize(self, token: str, client_id: str = "") -> dict[str, Any]:
        expired = self.expire_if_needed()
        now = time.monotonic()
        normalized_token = str(token or "").strip()
        normalized_client_id = str(client_id or "").strip()[:120]
        with self._lock:
            if not self._token:
                raise SessionLeaseError(
                    "Take the demo seat first.",
                    self._state_locked(now, normalized_client_id),
                )
            if normalized_token != self._token:
                if normalized_client_id:
                    self._waiting_clients[normalized_client_id] = now
                message = "Session in use. Watching live; controls are disabled until the seat is free."
                if expired:
                    message = "The previous session expired; try taking the seat again."
                raise SessionLeaseError(message, self._state_locked(now, normalized_client_id))
            self._last_seen_at = now
            if normalized_client_id:
                self._client_id = normalized_client_id
                self._waiting_clients.pop(normalized_client_id, None)
            return self._state_locked(now, normalized_client_id)

    def heartbeat(self, token: str, client_id: str = "") -> dict[str, Any]:
        self.authorize(token, client_id)
        return {"ok": True, "session": self.status(client_id)}

    def release(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, Any]:
        self.expire_if_needed()
        now = time.monotonic()
        normalized_token = str(token or "").strip()
        normalized_client_id = str(client_id or "").strip()[:120]
        released = False
        with self._lock:
            if self._token and normalized_token == self._token:
                self._release_locked(now, reason or "released")
                released = True
            elif self._token and normalized_client_id and normalized_client_id != self._client_id:
                self._waiting_clients[normalized_client_id] = now
            return {"ok": True, "released": released, "session": self._state_locked(now, normalized_client_id)}

    def mark_running(self, token: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._token and str(token or "").strip() == self._token:
                self._run_started_at = now
                self._completed_at = None
                self._last_seen_at = now

    def mark_completed(self, token: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if not self._token or str(token or "").strip() != self._token:
                return False
            self._completed_at = now
            return True

    def is_active_token(self, token: str) -> bool:
        with self._lock:
            return bool(self._token) and str(token or "").strip() == self._token


class Handler(BaseHTTPRequestHandler):
    server: "WindowServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            media = self.server.current_media()
            speaker_state = self.server.controller.initial_speaker_state()
            html = (
                HTML
                .replace("__SOURCE_JSON__", json_dumps(media.url))
                .replace("__PRESET_VIDEOS__", json_dumps(PRESET_YOUTUBE_VIDEOS))
                .replace("__SPEAKER_COLORS__", json_dumps(SPEAKER_COLORS))
                .replace(
                    "__NEW_SPEAKER_SENSITIVITY_JSON__",
                    json_dumps(new_speaker_sensitivity_config(getattr(self.server.args, "new_speaker_sensitivity", 3))),
                )
                .replace("__SPEAKER_REFINEMENT_JSON__", json_dumps(self.server.controller.speaker_refinement_settings()))
                .replace(
                    "__LIVE_SPEAKER_JSON__",
                    json_dumps({
                        "assignment_enabled": bool(getattr(self.server.args, "live_speaker_assignment", True)),
                        "unknown_clear_debounce_seconds": max(
                            0.0,
                            float(getattr(self.server.args, "live_speaker_probe_unknown_clear_debounce_seconds", 0.0)),
                        ),
                        "browser_observation_enabled": self.server.browser_live_observation_enabled,
                        "browser_observation_interval_seconds": max(
                            0.02,
                            float(getattr(self.server.args, "browser_live_observation_interval_seconds", DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS)),
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
                    }),
                )
                .replace("__SPEAKER_LIBRARY_JSON__", json_dumps(speaker_state))
            )
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/events":
            self._serve_events()
        elif path == "/api/events":
            self._serve_public_events(parsed)
        elif path == "/media/video":
            self._serve_file(self.server.current_media().video_file)
        elif path == "/media/audio":
            self._serve_file(self.server.current_media().audio_file)
        elif path == "/api/speakers":
            self._send_json({"ok": True, "speaker_state": self.server.controller.speaker_state()})
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

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/load-audio-file":
                self._handle_audio_file_upload()
                return
            payload = self._read_json_body()
            if path == "/api/sessions/create":
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
                self.server.controller.set_session_source_title(str(payload.get("source_title") or ""))
                self.server.controller.set_next_session_id(str(payload.get("session_id") or ""))
                speaker_state = self.server.controller.start()
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
                count = self.server.record_browser_live_observation(payload.get("samples", []))
                self._send_json({"ok": True, "sample_count": count})
            elif path == "/api/live-observation-finish":
                self._require_session(payload)
                count = self.server.record_browser_live_observation(payload.get("samples", []))
                summary = self.server.finish_browser_live_observation(str(payload.get("reason") or "done"))
                self._send_json({"ok": True, "sample_count": count, "summary": summary})
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
                result = self.server.controller.merge_speakers(
                    str(payload.get("source_speaker_id") or ""),
                    str(payload.get("target_speaker_id") or ""),
                    update_memory=bool(payload.get("update_memory", True)),
                )
                result.update({"ok": True, "session": self.server.session_status(str(payload.get("client_id") or ""))})
                self._send_json(result)
            elif path == "/api/speakers/delete":
                self._require_session(payload)
                result = self.server.controller.delete_speaker(
                    str(payload.get("speaker_id") or ""),
                    update_memory=bool(payload.get("update_memory", True)),
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
                state = self.server.controller.add_reference_speaker(
                    str(payload.get("name", "")),
                    str(payload.get("filename", "reference.wav")),
                    str(payload.get("audio_b64", "")),
                )
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


def _absolute_path_preserving_symlinks(path: Path) -> Path:
    """Return an absolute path without dereferencing venv executable symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


class WindowServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace, media: MediaFiles, bus: EventBus, controller: WindowDiarizer) -> None:
        super().__init__(address, Handler)
        self.args = args
        self.media = media
        self.media_version = int(time.time() * 1000)
        self.bus = bus
        self.controller = controller
        self.session_store = SessionStore(Path(getattr(args, "session_dir", DEFAULT_SESSION_DIR)))
        self._session_save_lock = threading.Lock()
        self._session_save_timer: threading.Timer | None = None
        self.public_event_session_id = uuid.uuid4().hex
        self.session_lease = SessionLease(
            idle_timeout_seconds=getattr(args, "session_lease_idle_timeout_seconds", 120.0),
            heartbeat_timeout_seconds=getattr(args, "session_lease_heartbeat_timeout_seconds", 45.0),
            completed_release_delay_seconds=getattr(args, "session_lease_completed_release_delay_seconds", 10.0),
            max_run_seconds=getattr(args, "session_lease_max_run_seconds", 900.0),
        )
        self._media_lock = threading.Lock()
        self._session_monitor_lock = threading.Lock()
        self._session_monitor_token = ""
        self.browser_live_recorder = (
            BrowserLiveObservationRecorder(
                output_path=args.browser_live_observation_output,
                canonical_path=args.validation_canonical,
                max_sample_gap_seconds=args.browser_live_observation_max_sample_gap_seconds,
                flicker_gap_seconds=args.browser_live_observation_flicker_gap_seconds,
            )
            if args.browser_live_observation_output is not None
            else None
        )
        self.bus.add_listener(self._record_session_event)

    def _record_session_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "sentence":
            if payload.get("pending") or payload.get("realtime") or payload.get("provisional_assignment"):
                return
            self._schedule_session_autosave()
        elif event == "speakers":
            self._schedule_session_autosave()
        elif event == "done":
            self._cancel_session_autosave()
            self._save_current_session(status_label="Saved", write_audio=True)

    def _cancel_session_autosave(self) -> None:
        with self._session_save_lock:
            if self._session_save_timer is not None:
                self._session_save_timer.cancel()
                self._session_save_timer = None

    def _schedule_session_autosave(self) -> None:
        with self._session_save_lock:
            if self._session_save_timer is not None:
                self._session_save_timer.cancel()
            timer = threading.Timer(1.0, self._run_session_autosave)
            timer.daemon = True
            self._session_save_timer = timer
            timer.start()

    def _run_session_autosave(self) -> None:
        with self._session_save_lock:
            self._session_save_timer = None
        self._save_current_session(status_label="Autosaved", write_audio=False)

    def _save_current_session(self, *, status_label: str, write_audio: bool) -> dict[str, Any] | None:
        snapshot = self.controller.session_snapshot()
        if not snapshot.get("id"):
            return None
        return self.session_store.save_snapshot(
            snapshot,
            status_label=status_label,
            write_audio=write_audio,
            audio_writer=self.controller.write_session_audio,
        )

    def list_saved_sessions(self, filter_mode: str = "active", query: str = "") -> dict[str, Any]:
        return {
            "ok": True,
            "sessions": self.session_store.list_sessions(filter_mode, query),
            "filter": filter_mode,
        }

    def create_saved_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        summary = self.session_store.create_session(
            source=source,
            title=str(payload.get("title") or ""),
            session_id=str(payload.get("session_id") or ""),
            status_label=str(payload.get("status_label") or "New"),
        )
        return {"ok": True, "session": summary}

    def open_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.open_session(session_id)}

    def rename_saved_session(self, session_id: str, title: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.rename_session(session_id, title)}

    def archive_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.archive_session(session_id)}

    def restore_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.restore_session(session_id)}

    def delete_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.delete_session(session_id)}

    def rename_saved_session_speaker(self, session_id: str, speaker_id: str, name: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.rename_speaker(session_id, speaker_id, name)}

    def _meeting_intelligence_session_id(self, session_id: str) -> str:
        requested = str(session_id or "").strip()
        snapshot = self.controller.session_snapshot()
        current_id = str(snapshot.get("id") or "").strip()
        if requested:
            if current_id and requested == current_id:
                self._save_current_session(status_label="Saved", write_audio=False)
            return requested
        if current_id:
            self._save_current_session(status_label="Saved", write_audio=False)
            return current_id
        raise ValueError("Choose or create a saved session first.")

    def meeting_intelligence_report(self, session_id: str) -> dict[str, Any]:
        resolved_session_id = self._meeting_intelligence_session_id(session_id)
        return {
            "ok": True,
            "session_id": resolved_session_id,
            "meeting_intelligence": self.session_store.meeting_intelligence(resolved_session_id),
        }

    def generate_meeting_intelligence(self, session_id: str) -> dict[str, Any]:
        resolved_session_id = self._meeting_intelligence_session_id(session_id)
        return {
            "ok": True,
            "session_id": resolved_session_id,
            "meeting_intelligence": self.session_store.generate_meeting_intelligence(resolved_session_id),
        }

    def update_meeting_intelligence_object(
        self,
        session_id: str,
        object_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> dict[str, Any]:
        resolved_session_id = self._meeting_intelligence_session_id(session_id)
        return {
            "ok": True,
            "session_id": resolved_session_id,
            "meeting_intelligence": self.session_store.update_meeting_intelligence_object(
                resolved_session_id,
                object_id,
                status=status,
                title=title,
                body=body,
            ),
        }

    @property
    def session_lease_enabled(self) -> bool:
        return bool(getattr(self.args, "demo_seat_lease", False))

    def _disabled_session_state(self, client_id: str = "") -> dict[str, Any]:
        return {
            "enabled": False,
            "active": False,
            "is_owner": True,
            "running": self.controller.is_running(),
            "completed": False,
            "client_id": str(client_id or "")[:120],
        }

    def _enforce_session_timeouts(self) -> None:
        if not self.session_lease_enabled:
            return
        expired = self.session_lease.expire_if_needed()
        if not expired:
            return
        reason = str(expired.get("reason") or "expired")
        self.bus.emit("status", {"message": f"Demo seat released ({reason})."})
        if expired.get("was_running"):
            self.controller.stop()

    def session_status(self, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return self._disabled_session_state(client_id)
        self._enforce_session_timeouts()
        return self.session_lease.status(client_id)

    def acquire_session(self, client_id: str) -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {
                "ok": True,
                "acquired": True,
                "session_token": "",
                "session": self._disabled_session_state(client_id),
            }
        self._enforce_session_timeouts()
        result = self.session_lease.acquire(client_id)
        if result.get("acquired"):
            self.bus.emit("status", {"message": "Demo seat acquired."})
        return result

    def require_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return self._disabled_session_state(client_id)
        self._enforce_session_timeouts()
        return self.session_lease.authorize(token, client_id)

    def heartbeat_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {"ok": True, "session": self._disabled_session_state(client_id)}
        self._enforce_session_timeouts()
        return self.session_lease.heartbeat(token, client_id)

    def release_session(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {"ok": True, "released": False, "session": self._disabled_session_state(client_id)}
        was_running = self.session_lease.is_active_token(token) and self.controller.is_running()
        result = self.session_lease.release(token, reason, client_id)
        if result.get("released"):
            self.bus.emit("status", {"message": f"Demo seat released ({reason})."})
            if was_running:
                self.controller.stop()
        return result

    def mark_session_running(self, token: str) -> None:
        if not self.session_lease_enabled:
            return
        self.session_lease.mark_running(token)
        self._start_session_completion_monitor(token)

    def _start_session_completion_monitor(self, token: str) -> None:
        with self._session_monitor_lock:
            self._session_monitor_token = token

        def monitor() -> None:
            time.sleep(0.5)
            while self.session_lease.is_active_token(token):
                self._enforce_session_timeouts()
                if not self.session_lease.is_active_token(token):
                    return
                if not self.controller.is_running():
                    break
                time.sleep(1.0)
            if not self.session_lease.is_active_token(token):
                return
            if self.session_lease.mark_completed(token):
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            "Run finished; demo seat will release in "
                            f"{self.session_lease.completed_release_delay_seconds:.0f}s."
                        )
                    },
                )
                time.sleep(self.session_lease.completed_release_delay_seconds)
                if self.session_lease.is_active_token(token):
                    self.release_session(token, "completed")

        thread = threading.Thread(target=monitor, name="SessionLeaseMonitor", daemon=True)
        thread.start()

    @property
    def browser_live_observation_enabled(self) -> bool:
        return self.browser_live_recorder is not None

    def record_browser_live_observation(self, samples: Any) -> int:
        if self.browser_live_recorder is None:
            return 0
        if not isinstance(samples, list):
            samples = []
        return self.browser_live_recorder.record(samples)

    def finish_browser_live_observation(self, reason: str = "done") -> dict[str, Any]:
        if self.browser_live_recorder is None:
            return {}
        summary = self.browser_live_recorder.finish(reason=reason)
        self.bus.emit("status", {
            "message": (
                "Browser live-speaker observation score "
                f"{summary.get('strict_browser_live_score', 0.0):.3f} written to "
                f"{self.browser_live_recorder.output_path}"
            ),
        })
        return summary

    def current_media(self) -> MediaFiles:
        with self._media_lock:
            return self.media

    def load_media_url(self, url: str, skip_download: bool = False) -> MediaFiles:
        self.bus.emit("status", {"message": f"Loading media for {url}"})
        if not skip_download:
            video_id, has_cached_audio, has_cached_video = media_cache_status(self.args, url)
            if not has_cached_audio or not has_cached_video:
                missing = []
                if not has_cached_audio:
                    missing.append("audio")
                if not has_cached_video:
                    missing.append("video")
                self.bus.emit("status", {
                    "message": f"Media cache miss for {video_id}; downloading missing {' and '.join(missing)}.",
                })
            else:
                self.bus.emit("status", {"message": f"Media cache hit for {video_id}."})
        media = resolve_media_url(self.args, url, skip_download=skip_download)
        with self._media_lock:
            self.media = media
            self.media_version += 1
        self.controller.set_media(media)
        self.bus.emit("status", {"message": f"Loaded {media.video_id}."})
        return media

    def load_audio_upload(
        self,
        *,
        filename: str,
        content_type: str,
        source: BinaryIO,
        length: int,
    ) -> tuple[MediaFiles, str, int]:
        if length <= 0:
            raise RuntimeError("Uploaded audio file is empty.")
        max_bytes = max(1, int(float(getattr(self.args, "max_audio_upload_mb", 2048)) * 1024 * 1024))
        if length > max_bytes:
            raise RuntimeError(
                f"Audio file is too large ({length / (1024 * 1024):.1f} MB). "
                f"Maximum upload size is {max_bytes / (1024 * 1024):.0f} MB."
            )
        display_name = _sanitize_upload_filename(filename)
        suffix = _audio_upload_extension(display_name, content_type)
        video_id = f"local-audio-{uuid.uuid4().hex[:12]}"
        upload_dir = self.args.work_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_file = (upload_dir / f"{video_id}{suffix}").resolve()
        temp_file = audio_file.with_suffix(audio_file.suffix + ".tmp")
        written = 0
        try:
            with temp_file.open("wb") as handle:
                remaining = length
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
                    if written > max_bytes:
                        raise RuntimeError(
                            f"Audio file is too large. Maximum upload size is {max_bytes / (1024 * 1024):.0f} MB."
                        )
            if written != length:
                raise RuntimeError("Audio upload ended before the full file was received.")
            temp_file.replace(audio_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
        url = f"local-audio://{video_id}/{quote(display_name)}"
        media = MediaFiles(url, video_id, audio_file, audio_file)
        self.bus.emit("status", {"message": f"Loading uploaded audio file {display_name}."})
        self.controller.set_media(media)
        with self._media_lock:
            self.media = media
            self.media_version += 1
        self.bus.emit("status", {"message": f"Loaded uploaded audio file {display_name}."})
        return media, display_name, written

    def start_browser_stream_url(self, url: str) -> MediaFiles:
        self.bus.emit("status", {"message": f"Preparing browser audio stream for {url}"})
        media = self.controller.set_browser_stream(url)
        with self._media_lock:
            self.media = media
            self.media_version += 1
        parsed = urlparse(url)
        if parsed.scheme == "microphone":
            instruction = "press Start and allow microphone access."
        elif parsed.scheme == "mixed-audio":
            instruction = "press Start, share a tab or window with audio, and allow microphone access."
        else:
            instruction = "press Start and share a tab or window with audio."
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Browser audio stream ready for {media.video_id}; {instruction}"
                )
            },
        )
        return media


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


def build_window_validation_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest_by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "sentence":
            continue
        payload = record.get("payload") or {}
        if payload.get("pending") or payload.get("realtime") or payload.get("provisional_assignment"):
            continue
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        latest_by_index[index] = dict(payload)

    analysis_records: list[dict[str, Any]] = []
    final_payloads: list[dict[str, Any]] = []
    for index in sorted(latest_by_index):
        payload = dict(latest_by_index[index])
        start = float(payload.get("start") or 0.0)
        end = float(payload.get("end") or start)
        payload["video_start_seconds"] = start
        payload["video_end_seconds"] = end
        payload["duration_seconds"] = max(0.0, end - start)
        final_payloads.append(payload)
        analysis_records.append({"time": time.time(), "event": "final", "payload": payload})
        analysis_records.append({"time": time.time(), "event": "sentence", "payload": payload})
    return analysis_records, final_payloads


def ratio_summary(final_payloads: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ratios = [
        float(payload["speech_audio_ratio"])
        for payload in final_payloads
        if payload.get("speech_audio_ratio") is not None
    ]
    if not ratios:
        return {"count": 0, "below_threshold": 0}
    return {
        "count": len(ratios),
        "below_threshold": sum(1 for ratio in ratios if ratio < threshold),
        "min": round(min(ratios), 4),
        "max": round(max(ratios), 4),
        "mean": round(sum(ratios) / len(ratios), 4),
    }


def run_window_replay_validation(args: argparse.Namespace) -> int:
    from realtime.realtime_speakerdiarize import analyze_trace_against_canonical, read_canonical_segments

    media = resolve_media(args)
    if not args.validation_keep_preview:
        args.realtime_preview_engine = "off"
    bus = RecordingEventBus()
    controller = WindowDiarizer(args, media, bus)
    reached_end_at: float | None = None
    started = time.monotonic()
    try:
        controller.start()
        replay_started = time.monotonic()
        bus.emit(
            "validation_replay_start",
            {
                "replay_speed": args.validation_replay_speed,
                "duration_seconds": round(float(controller.duration), 4),
            },
        )
        last_report = -1
        while not bus.done.is_set():
            elapsed = time.monotonic() - replay_started
            playback_seconds = min(controller.duration, elapsed * max(0.01, args.validation_replay_speed))
            # Validation replay owns the synthetic media clock; allow accelerated
            # runs to advance faster than the browser/live wall-clock clamp.
            controller.set_playback_time(
                playback_seconds,
                reset=bool(args.validation_replay_speed > 1.0),
            )
            report_second = int(playback_seconds // 15) * 15
            if report_second != last_report and report_second > 0:
                last_report = report_second
                print(f"Replay playback={playback_seconds:.1f}s/{controller.duration:.1f}s", flush=True)
            if playback_seconds >= controller.duration:
                if reached_end_at is None:
                    reached_end_at = time.monotonic()
                elif time.monotonic() - reached_end_at >= args.validation_final_wait_seconds:
                    print("Timed out waiting for final window flush.", flush=True)
                    break
            bus.done.wait(max(0.02, args.validation_update_interval_seconds))
    finally:
        controller.shutdown()

    elapsed = time.monotonic() - started
    analysis_records, final_payloads = build_window_validation_records(bus.records)
    canonical_segments = read_canonical_segments(args.validation_canonical)
    summary = analyze_trace_against_canonical(
        analysis_records,
        canonical_segments,
        match_mode=args.validation_match_mode,
    )
    summary.update({
        "system": "youtube_window_diarize_gui",
        "media": {
            "url": media.url,
            "video_id": media.video_id,
            "audio_file": str(media.audio_file),
            "video_file": str(media.video_file),
            "duration_sec": round(float(controller.duration), 4),
        },
        "canonical": str(args.validation_canonical),
        "elapsed_seconds": round(elapsed, 4),
        "validation_replay_speed": args.validation_replay_speed,
        "validation_keep_preview": args.validation_keep_preview,
        "embedding_provider": args.embedding_provider,
        "embeddings_backend": args.embeddings_backend,
        "embedding_device": args.embedding_device,
        "embedding_python": str(args.embedding_python),
        "remote_embeddings_url": args.remote_embeddings_url,
        "clustering_args": {
            "same_speaker_similarity": args.same_speaker_similarity,
            "similarity_temperature": args.similarity_temperature,
            "speaker_softmax_temperature": args.speaker_softmax_temperature,
            "new_speaker_threshold": args.new_speaker_threshold,
            "duplicate_profile_similarity": args.duplicate_profile_similarity,
            "unknown_short_threshold": args.unknown_short_threshold,
            "min_first_speaker_seconds": args.min_first_speaker_seconds,
            "min_new_speaker_seconds": args.min_new_speaker_seconds,
            "late_new_speaker_min_seconds": args.late_new_speaker_min_seconds,
            "max_speakers": args.max_speakers,
            "min_margin": args.min_margin,
            "margin_temperature": args.margin_temperature,
            "update_unknown_max": args.update_unknown_max,
            "new_speaker_confirmation_count": args.new_speaker_confirmation_count,
            "new_speaker_confirmation_similarity": args.new_speaker_confirmation_similarity,
            "max_pending_new_speakers": args.max_pending_new_speakers,
            "known_speaker_min_similarity": args.known_speaker_min_similarity,
            "known_speaker_gray_zone_min_unknown_probability": (
                args.known_speaker_gray_zone_min_unknown_probability
            ),
            "profile_update_min_similarity": args.profile_update_min_similarity,
            "profile_update_min_margin": args.profile_update_min_margin,
            "low_similarity_unknown_floor_similarity": args.low_similarity_unknown_floor_similarity,
            "low_similarity_unknown_floor_probability": args.low_similarity_unknown_floor_probability,
            "gray_zone_promote_max_similarity": args.gray_zone_promote_max_similarity,
            "min_new_speaker_words": args.min_new_speaker_words,
            "min_speech_audio_ratio": args.min_speech_audio_ratio,
            "retro_reassign_min_similarity": args.retro_reassign_min_similarity,
            "retro_reassign_min_margin": args.retro_reassign_min_margin,
            "speaker_refinement": args.speaker_refinement,
            "speaker_refinement_unknown_tentative": args.speaker_refinement_unknown_tentative,
            "speaker_refinement_unknown_commit": args.speaker_refinement_unknown_commit,
            "allow_speaker_reassignment": args.allow_speaker_reassignment,
            "speaker_refinement_max_per_profile": args.speaker_refinement_max_per_profile,
            "speaker_refinement_min_duration": args.speaker_refinement_min_duration,
            "speaker_refinement_max_unknown": args.speaker_refinement_max_unknown,
            "speaker_refinement_top_k": args.speaker_refinement_top_k,
            "speaker_refinement_centroid_blend": args.speaker_refinement_centroid_blend,
            "speaker_refinement_unknown_min_similarity": args.speaker_refinement_unknown_min_similarity,
            "speaker_refinement_unknown_min_margin": args.speaker_refinement_unknown_min_margin,
            "speaker_refinement_known_max_duration": args.speaker_refinement_known_max_duration,
            "speaker_refinement_known_min_similarity": args.speaker_refinement_known_min_similarity,
            "speaker_refinement_known_min_delta": args.speaker_refinement_known_min_delta,
            "speaker_refinement_final_passes": args.speaker_refinement_final_passes,
            "speaker_refinement_small_island_merge": args.speaker_refinement_small_island_merge,
            "speaker_refinement_small_island_max_duration": args.speaker_refinement_small_island_max_duration,
            "speaker_refinement_small_island_max_segments": args.speaker_refinement_small_island_max_segments,
            "speaker_refinement_tiny_fragmented_merge": args.speaker_refinement_tiny_fragmented_merge,
            "speaker_refinement_tiny_fragmented_max_duration": args.speaker_refinement_tiny_fragmented_max_duration,
            "speaker_refinement_tiny_fragmented_max_segments": args.speaker_refinement_tiny_fragmented_max_segments,
            "speaker_refinement_tiny_fragmented_min_islands": args.speaker_refinement_tiny_fragmented_min_islands,
            "speaker_refinement_tiny_fragmented_max_islands": args.speaker_refinement_tiny_fragmented_max_islands,
            "speaker_refinement_tiny_fragmented_min_neighbor_share": (
                args.speaker_refinement_tiny_fragmented_min_neighbor_share
            ),
            "speaker_refinement_terminal_outro_merge": args.speaker_refinement_terminal_outro_merge,
            "speaker_refinement_terminal_outro_max_duration": args.speaker_refinement_terminal_outro_max_duration,
            "speaker_refinement_terminal_outro_lookback_segments": (
                args.speaker_refinement_terminal_outro_lookback_segments
            ),
            "speaker_refinement_terminal_outro_min_target_duration": (
                args.speaker_refinement_terminal_outro_min_target_duration
            ),
            "speaker_refinement_unknown_same_speaker_fill": (
                args.speaker_refinement_unknown_same_speaker_fill
            ),
            "speaker_refinement_unknown_same_speaker_max_duration": (
                args.speaker_refinement_unknown_same_speaker_max_duration
            ),
            "speaker_refinement_unknown_same_speaker_max_segments": (
                args.speaker_refinement_unknown_same_speaker_max_segments
            ),
            "speaker_refinement_unknown_previous_speaker_fill": (
                args.speaker_refinement_unknown_previous_speaker_fill
            ),
            "speaker_refinement_unknown_previous_speaker_max_duration": (
                args.speaker_refinement_unknown_previous_speaker_max_duration
            ),
            "speaker_refinement_unknown_previous_speaker_max_segments": (
                args.speaker_refinement_unknown_previous_speaker_max_segments
            ),
            "speaker_refinement_unknown_previous_speaker_max_previous_gap": (
                args.speaker_refinement_unknown_previous_speaker_max_previous_gap
            ),
            "speaker_refinement_unknown_previous_speaker_min_next_gap": (
                args.speaker_refinement_unknown_previous_speaker_min_next_gap
            ),
            "speaker_refinement_unknown_next_speaker_fill": (
                args.speaker_refinement_unknown_next_speaker_fill
            ),
            "speaker_refinement_unknown_next_speaker_max_duration": (
                args.speaker_refinement_unknown_next_speaker_max_duration
            ),
            "speaker_refinement_unknown_next_speaker_max_segments": (
                args.speaker_refinement_unknown_next_speaker_max_segments
            ),
            "speaker_refinement_unknown_next_speaker_max_next_gap": (
                args.speaker_refinement_unknown_next_speaker_max_next_gap
            ),
            "speaker_refinement_unknown_next_speaker_min_previous_gap": (
                args.speaker_refinement_unknown_next_speaker_min_previous_gap
            ),
            "speaker_refinement_long_low_confidence_retro_split": (
                args.speaker_refinement_long_low_confidence_retro_split
            ),
            "speaker_refinement_long_low_confidence_retro_min_duration": (
                args.speaker_refinement_long_low_confidence_retro_min_duration
            ),
            "speaker_refinement_long_low_confidence_retro_max_similarity": (
                args.speaker_refinement_long_low_confidence_retro_max_similarity
            ),
            "speaker_refinement_long_low_confidence_retro_max_margin": (
                args.speaker_refinement_long_low_confidence_retro_max_margin
            ),
            "speaker_refinement_long_low_confidence_retro_max_splits": (
                args.speaker_refinement_long_low_confidence_retro_max_splits
            ),
            "new_speaker_sensitivity": getattr(args, "new_speaker_sensitivity", 3),
            "new_speaker_sensitivity_label": getattr(args, "new_speaker_sensitivity_label", "Balanced"),
            "vad_sentence_splitting": args.vad_sentence_splitting,
            "vad_backend": args.vad_backend,
            "vad_silero_backend": args.vad_silero_backend,
            "vad_silero_onnx_model_path": str(args.vad_silero_onnx_model_path) if args.vad_silero_onnx_model_path is not None else None,
            "vad_silero_onnx_threads": args.vad_silero_onnx_threads,
            "vad_silero_speech_threshold": args.vad_silero_speech_threshold,
            "vad_silence_seconds": args.vad_silence_seconds,
            "vad_final_window_post_silence_seconds": args.vad_final_window_post_silence_seconds,
            "vad_next_window_start_silence_seconds": args.vad_next_window_start_silence_seconds,
            "vad_speech_rms_threshold": args.vad_speech_rms_threshold,
            "vad_frame_seconds": args.vad_frame_seconds,
            "vad_merge_gap_seconds": args.vad_merge_gap_seconds,
            "vad_min_speech_seconds": args.vad_min_speech_seconds,
            "vad_gate_secondary_backend": args.vad_gate_secondary_backend,
            "vad_gate_webrtc_mode": args.vad_gate_webrtc_mode,
            "vad_gate_min_consensus_seconds": args.vad_gate_min_consensus_seconds,
            "vad_gate_min_consensus_ratio": args.vad_gate_min_consensus_ratio,
            "asr_no_speech_filter": args.asr_no_speech_filter,
            "asr_no_speech_prob_threshold": args.asr_no_speech_prob_threshold,
            "asr_no_speech_hard_threshold": args.asr_no_speech_hard_threshold,
            "asr_no_speech_keep_short_max_words": args.asr_no_speech_keep_short_max_words,
            "asr_no_speech_keep_short_max_seconds": args.asr_no_speech_keep_short_max_seconds,
            "live_speaker_assignment": args.live_speaker_assignment,
            "live_speaker_embedding_provider": args.live_speaker_embedding_provider,
            "live_speaker_embedding_min_interval_seconds": args.live_speaker_embedding_min_interval_seconds,
            "live_speaker_embedding_target_utilization": args.live_speaker_embedding_target_utilization,
            "live_speaker_verify_on_change": args.live_speaker_verify_on_change,
            "live_speaker_verify_min_interval_seconds": args.live_speaker_verify_min_interval_seconds,
            "live_speaker_ema_window_seconds": args.live_speaker_ema_window_seconds,
            "live_speaker_ema_count": args.live_speaker_ema_count,
            "live_speaker_ema_alpha": args.live_speaker_ema_alpha,
            "live_speaker_probe_interval_seconds": args.live_speaker_probe_interval_seconds,
            "live_speaker_probe_attack_interval_seconds": args.live_speaker_probe_attack_interval_seconds,
            "live_speaker_probe_window_seconds": args.live_speaker_probe_window_seconds,
            "live_speaker_probe_hold_seconds": args.live_speaker_probe_hold_seconds,
            "live_speaker_probe_min_advance_seconds": args.live_speaker_probe_min_advance_seconds,
            "live_speaker_probe_attack_min_advance_seconds": args.live_speaker_probe_attack_min_advance_seconds,
            "live_speaker_probe_clear_silence_count": args.live_speaker_probe_clear_silence_count,
            "live_speaker_probe_clear_unknown_count": args.live_speaker_probe_clear_unknown_count,
            "live_speaker_probe_unknown_clear_debounce_seconds": args.live_speaker_probe_unknown_clear_debounce_seconds,
            "live_speaker_probe_unknown_keepalive": args.live_speaker_probe_unknown_keepalive,
            "live_speaker_probe_unknown_release_smoothing": args.live_speaker_probe_unknown_release_smoothing,
            "live_speaker_probe_unknown_release_count": args.live_speaker_probe_unknown_release_count,
            "live_speaker_probe_unknown_release_ema_alpha": args.live_speaker_probe_unknown_release_ema_alpha,
            "live_speaker_probe_unknown_release_margin": args.live_speaker_probe_unknown_release_margin,
            "live_speaker_weak_profile_assist": args.live_speaker_weak_profile_assist,
            "live_speaker_weak_profile_max_speech_seconds": args.live_speaker_weak_profile_max_speech_seconds,
            "live_speaker_weak_profile_min_similarity": args.live_speaker_weak_profile_min_similarity,
            "live_speaker_weak_profile_min_margin": args.live_speaker_weak_profile_min_margin,
            "live_speaker_weak_profile_max_unknown_probability": (
                args.live_speaker_weak_profile_max_unknown_probability
            ),
            "section_gap_new_speaker": args.section_gap_new_speaker,
            "section_gap_new_speaker_min_gap_seconds": args.section_gap_new_speaker_min_gap_seconds,
            "section_gap_new_speaker_min_prior_speech_seconds": (
                args.section_gap_new_speaker_min_prior_speech_seconds
            ),
            "section_gap_new_speaker_min_duration_seconds": (
                args.section_gap_new_speaker_min_duration_seconds
            ),
            "section_gap_new_speaker_min_similarity": args.section_gap_new_speaker_min_similarity,
            "section_gap_new_speaker_max_similarity": args.section_gap_new_speaker_max_similarity,
            "section_gap_new_speaker_min_margin": args.section_gap_new_speaker_min_margin,
            "unknown_pair_new_speaker": args.unknown_pair_new_speaker,
            "unknown_pair_new_speaker_max_gap_seconds": args.unknown_pair_new_speaker_max_gap_seconds,
            "unknown_pair_new_speaker_min_unknown_duration_seconds": (
                args.unknown_pair_new_speaker_min_unknown_duration_seconds
            ),
            "unknown_pair_new_speaker_min_current_duration_seconds": (
                args.unknown_pair_new_speaker_min_current_duration_seconds
            ),
            "unknown_pair_new_speaker_min_pair_similarity": args.unknown_pair_new_speaker_min_pair_similarity,
            "unknown_pair_new_speaker_max_existing_similarity": (
                args.unknown_pair_new_speaker_max_existing_similarity
            ),
            "unknown_pair_new_speaker_max_existing_margin": args.unknown_pair_new_speaker_max_existing_margin,
            "unknown_pair_new_speaker_min_unknown_probability": (
                args.unknown_pair_new_speaker_min_unknown_probability
            ),
            "live_speaker_raw_change_snap": args.live_speaker_raw_change_snap,
            "live_speaker_raw_change_min_probability": args.live_speaker_raw_change_min_probability,
            "live_speaker_raw_change_min_margin": args.live_speaker_raw_change_min_margin,
            "live_speaker_sentence_hint": args.live_speaker_sentence_hint,
            "live_speaker_sentence_hint_max_lag_seconds": args.live_speaker_sentence_hint_max_lag_seconds,
            "live_speaker_sentence_hint_new_speaker_max_lag_seconds": args.live_speaker_sentence_hint_new_speaker_max_lag_seconds,
            "live_speaker_sentence_hint_new_speaker_hold_seconds": args.live_speaker_sentence_hint_new_speaker_hold_seconds,
            "live_speaker_sentence_hint_new_speaker_max_top_similarity": args.live_speaker_sentence_hint_new_speaker_max_top_similarity,
            "live_speaker_sentence_hint_hold_seconds": args.live_speaker_sentence_hint_hold_seconds,
        },
        "min_speech_audio_ratio": args.min_speech_audio_ratio,
        "speech_audio_ratio": ratio_summary(final_payloads, args.min_speech_audio_ratio),
        "unknown_permanent_segments": sum(1 for payload in final_payloads if payload.get("unknown_permanent")),
        "created_speaker_segments": sum(1 for payload in final_payloads if payload.get("created_speaker")),
        "raw_event_counts": dict(Counter(str(record.get("event")) for record in bus.records)),
        "final_payloads": final_payloads,
    })

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.validation_trace_output is not None:
        args.validation_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_trace_output.write_text(
            "\n".join(json_dumps(record) for record in bus.records) + "\n",
            encoding="utf-8",
        )

    print(f"Window validation output: {args.validation_output}", flush=True)
    if args.validation_trace_output is not None:
        print(f"Window validation trace: {args.validation_trace_output}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Final segments: {summary['final_segments']}", flush=True)
    print(f"Resolved segments: {summary['resolved_segments']}", flush=True)
    print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}", flush=True)
    print(f"Text recall/precision by LCS: {summary['text_recall']:.3f} / {summary['text_precision']:.3f}", flush=True)
    print(f"Assigned counts: {summary['assigned_counts']}", flush=True)
    print(f"Profile map: {summary['profile_map']}", flush=True)
    print(f"Unknown segments: {summary['unknown_segments']}", flush=True)
    print(f"Unknown permanent segments: {summary['unknown_permanent_segments']}", flush=True)
    print(f"Created speaker segments: {summary['created_speaker_segments']}", flush=True)
    print(f"Speech/audio ratio: {summary['speech_audio_ratio']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0


def _argv_has_option(argv: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(item == option or item.startswith(option_prefix) for item in argv)


def parse_args() -> argparse.Namespace:
    raw_argv = sys.argv[1:]
    preview_model_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model")
    preview_model_path_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model-path")
    preview_model_preset_was_explicit = _argv_has_option(raw_argv, "--realtime-preview-model-preset")
    language_was_explicit = _argv_has_option(raw_argv, "--language")
    language_was_from_env = bool(os.environ.get("WHOSPEAKS_LANGUAGE") or os.environ.get("WHOSPEAKS_ASR_LANGUAGE"))
    default_vad_model_path = default_silero_vad_model_path()
    parser = argparse.ArgumentParser(description="Growing-window faster-whisper speaker diarization GUI.")
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
        choices=("local", "remote"),
        default="local",
        help="ASR backend for final growing-window transcription.",
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
    parser.add_argument("--model", default="large-v2")
    parser.add_argument(
        "--language",
        type=language_arg,
        default=default_language_code(),
        help="Realtime language for final ASR, Kroko preview model selection, and sentence splitting.",
    )
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
    parser.add_argument("--embedding-provider", default=DEFAULT_WINDOW_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-python", type=Path, default=default_embedding_python())
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument(
        "--live-speaker-embedding-provider",
        default="jungjee_rawnet3",
        help="Single provider used only for fast live speaker assignment. Empty uses --embedding-provider.",
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
        help="Realtime preview engine: kroko_onnx, mock, or off.",
    )
    parser.add_argument(
        "--realtime-preview-model",
        default=None,
        help="Kroko/Banafo model name for replace-only realtime preview text. Overrides --realtime-preview-model-preset.",
    )
    parser.add_argument(
        "--realtime-preview-model-preset",
        type=normalize_kroko_preview_model_preset,
        default=DEFAULT_KROKO_PREVIEW_MODEL_PRESET,
        metavar="{community-64l,pro-16l}",
        help="Named Kroko preview model preset. Use pro-16l for Kroko-EN-Pro-16-L-Streaming-001.data.",
    )
    parser.add_argument("--realtime-preview-model-path", type=Path, default=None)
    parser.add_argument(
        "--realtime-preview-auto-download",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD,
        help="Download missing public Kroko Community preview models from Hugging Face before starting preview.",
    )
    parser.add_argument("--realtime-preview-download-root", type=Path, default=None)
    parser.add_argument("--realtime-preview-python", type=Path, default=DEFAULT_KROKO_PREVIEW_PYTHON)
    parser.add_argument("--realtime-preview-realtimestt-root", type=Path, default=DEFAULT_REALTIMESTT_ROOT)
    parser.add_argument("--realtime-preview-provider", default="cpu")
    parser.add_argument("--realtime-preview-num-threads", type=int, default=2)
    parser.add_argument(
        "--realtime-preview-startup-timeout-seconds",
        type=float,
        default=None,
        help="Maximum time to wait for the realtime preview engine before disabling preview. Defaults to 45s for pro-16l and 12s otherwise.",
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
    parser.add_argument("--keep-segment-audio", action="store_true")
    parser.add_argument("--validate-window-replay", action="store_true")
    parser.add_argument("--validation-canonical", type=Path, default=DEFAULT_CUNK_CANONICAL)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--validation-trace-output", type=Path, default=None)
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
        "--validation-keep-preview",
        action="store_true",
        help="Keep realtime preview enabled during validation. Final sentence metrics usually do not need this.",
    )
    args = parser.parse_args()
    if not language_was_explicit and not language_was_from_env:
        inferred_language = infer_language_from_kroko_model_name(args.realtime_preview_model)
        if inferred_language is not None:
            args.language = inferred_language
    args.sentence_tokenizer = default_sentence_tokenizer(args.language, args.sentence_tokenizer)
    args.sentence_language = default_sentence_language(args.language)
    args.realtime_preview_language = args.language
    preview_engine = str(args.realtime_preview_engine or "off").strip().lower().replace("-", "_")
    preview_uses_kroko = preview_engine not in {"off", "mock"}
    language_has_kroko_preview = is_kroko_preview_language(args.language)
    if preview_uses_kroko and not language_has_kroko_preview:
        parser.error(
            f"{args.language!r} is supported for final ASR and sentence splitting, but not for Kroko realtime "
            "preview; use --realtime-preview-engine off or choose a Kroko preview language."
        )
    if args.realtime_preview_model is None:
        if language_has_kroko_preview:
            try:
                args.realtime_preview_model = kroko_preview_model_name(
                    args.language,
                    args.realtime_preview_model_preset,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            args.realtime_preview_model = ""
    else:
        args.realtime_preview_model_preset = "custom"
    if args.realtime_preview_startup_timeout_seconds is None:
        args.realtime_preview_startup_timeout_seconds = default_kroko_preview_startup_timeout_seconds(
            args.realtime_preview_model_preset
        )
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
    if args.realtime_preview_model_path is None and args.realtime_preview_model:
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
    if args.realtime_preview_download_root is not None:
        args.realtime_preview_download_root = args.realtime_preview_download_root.resolve()
    if args.realtime_preview_python is not None:
        args.realtime_preview_python = _absolute_path_preserving_symlinks(args.realtime_preview_python)
    if args.realtime_preview_realtimestt_root is not None:
        args.realtime_preview_realtimestt_root = args.realtime_preview_realtimestt_root.resolve()
    if args.vad_silero_onnx_model_path is not None:
        args.vad_silero_onnx_model_path = args.vad_silero_onnx_model_path.resolve()
    preview_chunk_seconds = infer_kroko_preview_chunk_seconds(args.realtime_preview_model_path or args.realtime_preview_model)
    if args.realtime_preview_interval_seconds is None:
        args.realtime_preview_interval_seconds = preview_chunk_seconds
    if args.realtime_preview_min_audio_seconds is None:
        args.realtime_preview_min_audio_seconds = preview_chunk_seconds
    if args.realtime_preview_min_advance_seconds is None:
        args.realtime_preview_min_advance_seconds = preview_chunk_seconds
    if args.realtime_preview_feed_chunk_seconds is None:
        args.realtime_preview_feed_chunk_seconds = preview_chunk_seconds
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
    return args


def main() -> int:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing command line.", flush=True)
    args = parse_args()
    preview_model_display = (
        args.realtime_preview_model_path.name
        if args.realtime_preview_model_path is not None
        else args.realtime_preview_model
    )
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Startup config: "
        f"url={args.url} port={args.port} asr_backend={args.asr_backend} "
        f"language={args.language} sentence_tokenizer={args.sentence_tokenizer}/{args.sentence_language} "
        f"embeddings_backend={args.embeddings_backend} "
        f"embedding_provider={args.embedding_provider} "
        f"embedding_timeout={args.embedding_helper_response_timeout_seconds:.0f}s "
        f"realtime_preview={args.realtime_preview_engine} "
        f"preview_model={args.realtime_preview_model_preset}:{preview_model_display}.",
        flush=True,
    )
    if args.validate_window_replay:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running validation replay.", flush=True)
        return run_window_replay_validation(args)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Resolving media "
        f"({'cache only' if args.skip_download else 'download allowed'}).",
        flush=True,
    )
    media = resolve_media(args)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Media ready: "
        f"video_id={media.video_id} audio={media.audio_file.name} video={media.video_file.name}.",
        flush=True,
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating controller and HTTP server.", flush=True)
    bus = EventBus()
    controller = WindowDiarizer(args, media, bus)
    server: WindowServer | None = None
    try:
        server = WindowServer((args.host, args.port), args, media, bus, controller)
        if args.startup_warmup_before_url:
            controller.prepare_before_browser_release()
        else:
            bus.emit("status", {"message": "Startup model warmup skipped; models will warm before playback."})
        page_url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
        print(f"Serving growing-window diarization GUI at {page_url}", flush=True)
        print(f"Video: {media.video_file}", flush=True)
        print(f"Audio: {media.audio_file}", flush=True)
        if args.startup_warmup_before_url:
            print("Ready. Open the URL and click Start; core models are already warm.", flush=True)
        else:
            print("Ready. Open the URL and click Start; core models will warm before playback.", flush=True)
        if not args.no_browser:
            webbrowser.open(page_url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.", flush=True)
    finally:
        controller.shutdown()
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
