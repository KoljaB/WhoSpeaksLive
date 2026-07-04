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
from typing import Any
from urllib.parse import parse_qs, urlparse


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


from window.window_config import (  # noqa: E402
    DEFAULT_CUNK_CANONICAL,
    DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS,
    DEFAULT_FAST_WHISPER_CACHE,
    DEFAULT_KROKO_PREVIEW_MODEL,
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
    default_silero_vad_backend,
    default_silero_vad_model_path,
    new_speaker_sensitivity_config,
)
from window.window_diarizer import WindowDiarizer  # noqa: E402
from window.window_events import EventBus, RecordingEventBus  # noqa: E402
from window.window_gui_html import HTML  # noqa: E402
from window.window_preview import infer_kroko_preview_chunk_seconds  # noqa: E402
from window.browser_live_speaker_scoring import (  # noqa: E402
    DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
    BrowserLiveObservationRecorder,
)

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
                        "unknown_clear_debounce_seconds": max(
                            0.0,
                            float(getattr(self.server.args, "live_speaker_probe_unknown_clear_debounce_seconds", 0.0)),
                        ),
                        "browser_observation_enabled": self.server.browser_live_observation_enabled,
                        "browser_observation_interval_seconds": max(
                            0.02,
                            float(getattr(self.server.args, "browser_live_observation_interval_seconds", DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS)),
                        ),
                    }),
                )
                .replace("__SPEAKER_LIBRARY_JSON__", json_dumps(speaker_state))
            )
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/events":
            self._serve_events()
        elif path == "/media/video":
            self._serve_file(self.server.current_media().video_file)
        elif path == "/media/audio":
            self._serve_file(self.server.current_media().audio_file)
        elif path == "/api/speakers":
            self._send_json({"ok": True, "speaker_state": self.server.controller.speaker_state()})
        elif path == "/api/session/status":
            query = parse_qs(parsed.query)
            client_id = str((query.get("client_id") or [""])[0])
            self._send_json({"ok": True, "session": self.server.session_status(client_id)})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            payload = self._read_json_body()
            if path == "/api/session/acquire":
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
                speaker_state = self.server.controller.start()
                self.server.mark_session_running(session_token)
                self._send_json({
                    "ok": True,
                    "speaker_state": speaker_state,
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
                if "allow_speaker_reassignment" in payload:
                    response["speaker_refinement"] = self.server.controller.set_allow_speaker_reassignment(
                        payload.get("allow_speaker_reassignment")
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
            elif path == "/api/speakers/clear":
                self._require_session(payload)
                state = self.server.controller.clear_speakers()
                self._send_json({"ok": True, "speaker_state": state, "session": self.server.session_status(str(payload.get("client_id") or ""))})
            elif path == "/api/speakers/save":
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


class WindowServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace, media: MediaFiles, bus: EventBus, controller: WindowDiarizer) -> None:
        super().__init__(address, Handler)
        self.args = args
        self.media = media
        self.media_version = int(time.time() * 1000)
        self.bus = bus
        self.controller = controller
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

    def _enforce_session_timeouts(self) -> None:
        expired = self.session_lease.expire_if_needed()
        if not expired:
            return
        reason = str(expired.get("reason") or "expired")
        self.bus.emit("status", {"message": f"Demo seat released ({reason})."})
        if expired.get("was_running"):
            self.controller.stop()

    def session_status(self, client_id: str = "") -> dict[str, Any]:
        self._enforce_session_timeouts()
        return self.session_lease.status(client_id)

    def acquire_session(self, client_id: str) -> dict[str, Any]:
        self._enforce_session_timeouts()
        result = self.session_lease.acquire(client_id)
        if result.get("acquired"):
            self.bus.emit("status", {"message": "Demo seat acquired."})
        return result

    def require_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        self._enforce_session_timeouts()
        return self.session_lease.authorize(token, client_id)

    def heartbeat_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        self._enforce_session_timeouts()
        return self.session_lease.heartbeat(token, client_id)

    def release_session(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, Any]:
        was_running = self.session_lease.is_active_token(token) and self.controller.is_running()
        result = self.session_lease.release(token, reason, client_id)
        if result.get("released"):
            self.bus.emit("status", {"message": f"Demo seat released ({reason})."})
            if was_running:
                self.controller.stop()
        return result

    def mark_session_running(self, token: str) -> None:
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

    def start_browser_stream_url(self, url: str) -> MediaFiles:
        self.bus.emit("status", {"message": f"Preparing browser audio stream for {url}"})
        media = self.controller.set_browser_stream(url)
        with self._media_lock:
            self.media = media
            self.media_version += 1
        parsed = urlparse(url)
        if parsed.scheme == "microphone":
            instruction = "press Start and allow microphone access."
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
        if payload.get("pending") or payload.get("realtime"):
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
            controller.set_playback_time(playback_seconds)
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
            "min_new_speaker_words": args.min_new_speaker_words,
            "min_speech_audio_ratio": args.min_speech_audio_ratio,
            "retro_reassign_min_similarity": args.retro_reassign_min_similarity,
            "retro_reassign_min_margin": args.retro_reassign_min_margin,
            "speaker_refinement": args.speaker_refinement,
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


def parse_args() -> argparse.Namespace:
    default_vad_model_path = default_silero_vad_model_path()
    parser = argparse.ArgumentParser(description="Growing-window faster-whisper speaker diarization GUI.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--video-file", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--yt-dlp", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--no-browser", action="store_true")
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
    parser.add_argument("--same-speaker-similarity", type=float, default=0.37)
    parser.add_argument("--similarity-temperature", type=float, default=0.0648)
    parser.add_argument("--speaker-softmax-temperature", type=float, default=0.0443)
    parser.add_argument("--new-speaker-threshold", type=float, default=0.38)
    parser.add_argument("--duplicate-profile-similarity", type=float, default=0.4)
    parser.add_argument("--unknown-short-threshold", type=float, default=0.3225)
    parser.add_argument("--min-first-speaker-seconds", type=float, default=1.3098)
    parser.add_argument("--min-new-speaker-seconds", type=float, default=1.6)
    parser.add_argument("--late-new-speaker-min-seconds", type=float, default=3.4127)
    parser.add_argument("--max-speakers", type=int, default=12)
    parser.add_argument("--min-margin", type=float, default=0.0386)
    parser.add_argument("--margin-temperature", type=float, default=0.03)
    parser.add_argument("--update-unknown-max", type=float, default=0.61)
    parser.add_argument(
        "--new-speaker-confirmation-count",
        type=int,
        default=1,
        help="Number of mutually similar far-away sentence embeddings required before creating a new speaker.",
    )
    parser.add_argument(
        "--new-speaker-confirmation-similarity",
        type=float,
        default=0.5149,
        help="Minimum cosine similarity between pending new-speaker candidates before creating a speaker.",
    )
    parser.add_argument("--max-pending-new-speakers", type=int, default=6)
    parser.add_argument(
        "--min-new-speaker-words",
        type=int,
        default=3,
        help="Minimum content words required for a sentence to create or confirm a new speaker profile.",
    )
    parser.add_argument(
        "--retro-reassign-min-similarity",
        type=float,
        default=0.05,
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
        "--allow-speaker-reassignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow prototype refinement to change already assigned speaker labels.",
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
    parser.add_argument("--speaker-refinement-known-min-delta", type=float, default=0.108)
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
        default=DEFAULT_KROKO_PREVIEW_MODEL,
        help="Kroko/Banafo model name for replace-only realtime preview text.",
    )
    parser.add_argument("--realtime-preview-model-path", type=Path, default=default_kroko_preview_model_path())
    parser.add_argument("--realtime-preview-download-root", type=Path, default=None)
    parser.add_argument("--realtime-preview-python", type=Path, default=DEFAULT_KROKO_PREVIEW_PYTHON)
    parser.add_argument("--realtime-preview-realtimestt-root", type=Path, default=DEFAULT_REALTIMESTT_ROOT)
    parser.add_argument("--realtime-preview-provider", default="cpu")
    parser.add_argument("--realtime-preview-num-threads", type=int, default=2)
    parser.add_argument(
        "--realtime-preview-startup-timeout-seconds",
        type=float,
        default=12.0,
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
        "--realtime-preview-reset-overlap-seconds",
        type=float,
        default=0.25,
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
        default=0.2,
        help="Minimum wall-clock spacing between live speaker embedding requests from preview/probe paths.",
    )
    parser.add_argument(
        "--live-speaker-embedding-target-utilization",
        type=float,
        default=1.0,
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
        default=0.2,
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
        default=0.2,
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
    args.work_dir = args.work_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.embedding_python = args.embedding_python.resolve()
    args.speaker_library_dir = args.speaker_library_dir.resolve()
    args.validation_canonical = args.validation_canonical.resolve()
    args.validation_output = args.validation_output.resolve()
    if args.validation_trace_output is not None:
        args.validation_trace_output = args.validation_trace_output.resolve()
    if args.browser_live_observation_output is not None:
        args.browser_live_observation_output = args.browser_live_observation_output.resolve()
    if args.download_root is not None:
        args.download_root = args.download_root.resolve()
    if args.realtime_preview_model_path is not None:
        args.realtime_preview_model_path = args.realtime_preview_model_path.resolve()
    if args.realtime_preview_download_root is not None:
        args.realtime_preview_download_root = args.realtime_preview_download_root.resolve()
    if args.realtime_preview_python is not None:
        args.realtime_preview_python = args.realtime_preview_python.resolve()
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
    return args


def main() -> int:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing command line.", flush=True)
    args = parse_args()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Startup config: "
        f"url={args.url} port={args.port} asr_backend={args.asr_backend} "
        f"embeddings_backend={args.embeddings_backend} "
        f"embedding_provider={args.embedding_provider} "
        f"embedding_timeout={args.embedding_helper_response_timeout_seconds:.0f}s "
        f"realtime_preview={args.realtime_preview_engine}.",
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
