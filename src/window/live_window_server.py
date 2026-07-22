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
import socket
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
import urllib.error
import urllib.request


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
from window.session_lease_coordinator import SessionLeaseCoordinator  # noqa: E402
from window.session_persistence import SessionPersistenceCoordinator  # noqa: E402
from window.saved_person_identity import SavedPersonIdentityService  # noqa: E402
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
from window.live_speaker_e2e_contract import (  # noqa: E402
    build_real_gui_e2e_attestation,
    seal_real_gui_e2e_attestation,
)
from window.live_speaker_world_tape import LiveSpeakerWorldTapeRecorder  # noqa: E402

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


from window.live_http_handler import Handler

def _absolute_path_preserving_symlinks(path: Path) -> Path:
    """Return an absolute path without dereferencing venv executable symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


class LiveWindowApplication:
    """Own live application collaborators independently from HTTP transport."""

    def __init__(self, args: Any, media: MediaFiles, bus: EventBus, controller: WindowDiarizer) -> None:
        self.args = args
        self.media_manager = MediaManager(media)
        self.bus = bus
        self.controller = controller
        self._browser_live_e2e_start_attestation = (
            build_real_gui_e2e_attestation(root=ROOT, args=args, media=media)
            if getattr(args, "browser_live_observation_output", None) is not None
            else None
        )
        world_tape_root = getattr(args, "live_speaker_world_tape_output", None)
        self.world_tape_recorder = (
            LiveSpeakerWorldTapeRecorder(Path(world_tape_root), args=args, media=media)
            if world_tape_root is not None
            else None
        )
        if self.world_tape_recorder is not None:
            self.bus.add_listener(self.world_tape_recorder.record_public)
            self.bus.add_internal_listener(self.world_tape_recorder.record_internal)
            self._bind_world_tape_media(media)
            _console_print(
                "Live-speaker World Tape recording to "
                f"{self.world_tape_recorder.output_dir}"
            )
        self.session_store = SessionStore(Path(getattr(args, "session_dir", DEFAULT_SESSION_DIR)))
        person_library = getattr(self.controller, "person_library", None)
        self.saved_person_identity = (
            SavedPersonIdentityService(self.session_store, person_library)
            if person_library is not None
            else None
        )
        self.translation = LiveTranslationCoordinator(args, bus)
        translation_error = str(self.translation.public_config(refresh_status=False).get("error") or "")
        if translation_error:
            self.bus.emit("status", {"message": f"Translation is unavailable: {translation_error}"})
        self.public_event_session_id = uuid.uuid4().hex
        self.session_lease = SessionLeaseStateMachine(
            idle_timeout_seconds=getattr(args, "session_lease_idle_timeout_seconds", 120.0),
            heartbeat_timeout_seconds=getattr(args, "session_lease_heartbeat_timeout_seconds", 45.0),
            completed_release_delay_seconds=getattr(args, "session_lease_completed_release_delay_seconds", 10.0),
            max_run_seconds=getattr(args, "session_lease_max_run_seconds", 900.0),
        )
        self.session_lease_coordinator = SessionLeaseCoordinator(
            self.session_lease,
            controller_is_running=self.controller.is_running,
            controller_stop=self.controller.stop,
            publish_status=lambda message: self.bus.emit("status", {"message": message}),
        )
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
        self.persistence = SessionPersistenceCoordinator(
            bus=self.bus,
            store=self.session_store,
            session_snapshot=self.controller.session_snapshot,
            session_id=self.controller.current_session_id,
            write_audio=self.controller.write_session_audio,
            translation_snapshot=self.translation.snapshot,
            handle_sentence_translation=self.translation.handle_sentence,
        )
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def meeting_intelligence_url(self) -> str:
        return str(getattr(self.args, "meeting_intelligence_url", "") or "").strip().rstrip("/")

    def meeting_intelligence_status(self) -> dict[str, Any]:
        if not self.meeting_intelligence_url:
            return {"ok": True, "enabled": False, "ready": False, "error": "Meeting Intelligence is disabled."}
        try:
            config = self._meeting_intelligence_request("GET", "/api/config")
        except Exception as exc:
            return {"ok": True, "enabled": True, "ready": False, "error": str(exc)}
        return {"ok": True, "enabled": True, "ready": True, "config": config.get("config") or {}}

    def meeting_chat_scope(self, session_ids: list[str]) -> dict[str, Any]:
        ids, provisional = self._meeting_chat_session_ids(session_ids)
        result = self._meeting_intelligence_request("POST", "/api/chat/scope", {"session_ids": ids})
        result.update({"ok": True, "provisional": provisional})
        return result

    def start_meeting_chat(self, session_ids: list[str], question: str) -> dict[str, Any]:
        ids, provisional = self._meeting_chat_session_ids(session_ids)
        result = self._meeting_intelligence_request("POST", "/api/chat/ask-async", {
            "session_ids": ids,
            "question": str(question or ""),
            "provisional": provisional,
        })
        result.update({"ok": True, "provisional": provisional})
        return result

    def meeting_chat_job(self, job_id: str) -> dict[str, Any]:
        encoded = quote(str(job_id or "").strip(), safe="")
        result = self._meeting_intelligence_request("GET", f"/api/chat/job?job_id={encoded}")
        result["ok"] = True
        return result

    def clear_meeting_chat(self, session_ids: list[str]) -> dict[str, Any]:
        ids, _provisional = self._meeting_chat_session_ids(session_ids)
        result = self._meeting_intelligence_request("POST", "/api/chat/clear", {"session_ids": ids})
        result["ok"] = True
        return result

    def _meeting_chat_session_ids(self, session_ids: list[str]) -> tuple[list[str], bool]:
        ids = sorted({str(value or "").strip() for value in session_ids if str(value or "").strip()})
        snapshot = self.controller.session_snapshot()
        current_id = str(snapshot.get("id") or "").strip()
        if not ids:
            ids = [self._meeting_intelligence_session_id("")]
        elif current_id and ids == [current_id]:
            self._save_current_session(status_label="Autosaved", write_audio=False)
        provisional = bool(current_id and ids == [current_id] and self.controller.is_running())
        return ids, provisional

    def _meeting_intelligence_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.meeting_intelligence_url:
            raise RuntimeError("Enable Meeting Intelligence in the launcher to use Ask.")
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.meeting_intelligence_url}{path}",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Meeting Intelligence request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Meeting Intelligence is unavailable: {exc.reason}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Meeting Intelligence returned an invalid response.")
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result

    def _record_session_event(self, event: str, payload: dict[str, Any]) -> None:
        self.persistence.handle_event(event, payload)

    def _cancel_session_autosave(self) -> None:
        self.persistence.cancel()

    def _schedule_session_autosave(self) -> None:
        self.persistence.schedule()

    def _run_session_autosave(self) -> None:
        self.persistence.save(status_label="Autosaved", write_audio=False)

    def _save_current_session(self, *, status_label: str, write_audio: bool) -> dict[str, Any] | None:
        return self.persistence.save(status_label=status_label, write_audio=write_audio)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.session_lease_coordinator.close()
        finally:
            try:
                self.translation.shutdown()
            finally:
                try:
                    self.persistence.close(flush=True)
                finally:
                    if self.world_tape_recorder is not None:
                        self.bus.remove_listener(self.world_tape_recorder.record_public)
                        self.bus.remove_internal_listener(self.world_tape_recorder.record_internal)
                        summary = self.world_tape_recorder.close(reason="application_close")
                        _console_print(
                            "Live-speaker World Tape finalized at "
                            f"{summary.get('output_dir')} "
                            f"({summary.get('event_count', 0)} events)."
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
        session = self.session_store.open_session(session_id)
        if self.saved_person_identity is not None:
            session = self.saved_person_identity.decorate_session(session)
        return {"ok": True, "session": session}

    def rename_saved_session(self, session_id: str, title: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.rename_session(session_id, title)}

    def archive_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.archive_session(session_id)}

    def restore_saved_session(self, session_id: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.restore_session(session_id)}

    def delete_saved_session(self, session_id: str) -> dict[str, Any]:
        removed_samples = 0
        if self.saved_person_identity is not None:
            removed_samples = self.saved_person_identity.remove_session_samples(session_id)
        try:
            deleted = self.session_store.delete_session(session_id)
        except Exception:
            if removed_samples and self.saved_person_identity is not None:
                self.saved_person_identity.recompute_linked_samples(session_id)
            raise
        return {"ok": True, "session": deleted, "removed_person_voice_samples": removed_samples}

    def delete_person_voice_sample(self, person_id: str, sample_id: str) -> dict[str, Any]:
        return self.controller.delete_voice_sample(person_id, sample_id)

    def forget_person_voice(self, person_id: str) -> dict[str, Any]:
        if self.saved_person_identity is not None:
            self.saved_person_identity.unlink_person_everywhere(person_id)
        return self.controller.forget_person_voice(person_id)

    def delete_person(self, person_id: str) -> dict[str, Any]:
        if self.saved_person_identity is not None:
            self.saved_person_identity.unlink_person_everywhere(person_id)
        return self.controller.delete_person(person_id)

    def rename_saved_session_speaker(self, session_id: str, speaker_id: str, name: str) -> dict[str, Any]:
        return {"ok": True, "session": self.session_store.rename_speaker(session_id, speaker_id, name)}

    def reassign_saved_session_rows(
        self,
        session_id: str,
        indexes: list[int],
        speaker_id: str,
    ) -> dict[str, Any]:
        self.session_store.reassign_rows(session_id, indexes, speaker_id)
        if self.saved_person_identity is not None:
            self.saved_person_identity.recompute_linked_samples(session_id)
        return self.open_saved_session(session_id)

    def mark_saved_session_rows_correct(self, session_id: str, indexes: list[int]) -> dict[str, Any]:
        self.session_store.mark_rows_correct(session_id, indexes)
        if self.saved_person_identity is not None:
            self.saved_person_identity.recompute_linked_samples(session_id)
        return self.open_saved_session(session_id)

    def link_saved_session_person(
        self,
        session_id: str,
        speaker_id: str,
        *,
        person_id: str = "",
        person_name: str = "",
        expected_updated_at: str = "",
    ) -> dict[str, Any]:
        if self.saved_person_identity is None:
            raise RuntimeError("People library is unavailable.")
        return {
            "ok": True,
            "session": self.saved_person_identity.link(
                session_id,
                speaker_id,
                person_id=person_id,
                person_name=person_name,
                expected_updated_at=expected_updated_at,
            ),
            "speaker_state": self.controller.speaker_state(),
        }

    def unlink_saved_session_person(self, session_id: str, speaker_id: str) -> dict[str, Any]:
        if self.saved_person_identity is None:
            raise RuntimeError("People library is unavailable.")
        return {
            "ok": True,
            "session": self.saved_person_identity.unlink(session_id, speaker_id),
            "speaker_state": self.controller.speaker_state(),
        }

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
        self.session_lease_coordinator.enforce_timeouts()

    def session_status(self, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return self._disabled_session_state(client_id)
        return self.session_lease_coordinator.status(client_id)

    def acquire_session(self, client_id: str) -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {
                "ok": True,
                "acquired": True,
                "session_token": "",
                "session": self._disabled_session_state(client_id),
            }
        return self.session_lease_coordinator.acquire(client_id)

    def require_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return self._disabled_session_state(client_id)
        return self.session_lease_coordinator.authorize(token, client_id)

    def heartbeat_session(self, token: str, client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {"ok": True, "session": self._disabled_session_state(client_id)}
        return self.session_lease_coordinator.heartbeat(token, client_id)

    def release_session(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, Any]:
        if not self.session_lease_enabled:
            return {"ok": True, "released": False, "session": self._disabled_session_state(client_id)}
        return self.session_lease_coordinator.release(token, reason, client_id)

    def mark_session_running(self, token: str) -> None:
        if not self.session_lease_enabled:
            return
        self.session_lease_coordinator.mark_running(token)

    def _start_session_completion_monitor(self, token: str) -> None:
        self.session_lease_coordinator.mark_running(token)

    @property
    def browser_live_observation_enabled(self) -> bool:
        return self.browser_live_recorder is not None or self.world_tape_recorder is not None

    def record_browser_live_observation(
        self,
        samples: Any,
        batch_sequence: int | None = None,
    ) -> int:
        if not isinstance(samples, list):
            samples = []
        counts = [0]
        if self.browser_live_recorder is not None:
            counts.append(self.browser_live_recorder.record(samples))
        if self.world_tape_recorder is not None:
            counts.append(
                self.world_tape_recorder.record_browser_samples(
                    samples,
                    batch_sequence=batch_sequence,
                )
            )
        return max(counts)

    def finish_browser_live_observation(self, reason: str = "done") -> dict[str, Any]:
        if self.browser_live_recorder is None and self.world_tape_recorder is None:
            return {}
        seal_world_tape = bool(
            getattr(self.args, "exit_after_browser_live_observation", False)
        )
        world_tape = None
        if self.world_tape_recorder is not None:
            if seal_world_tape:
                # The browser POSTs are serialized, so this closes over the
                # final DOM batch and produces the immutable hashes that are
                # bound into the E2E attestation below.
                world_tape = self.world_tape_recorder.close(reason=f"browser:{reason}")
            else:
                world_tape = self.world_tape_recorder.checkpoint(
                    reason=f"browser:{reason}"
                )
        finished_attestation = build_real_gui_e2e_attestation(
            root=ROOT,
            args=self.args,
            media=self.current_media(),
        )
        started_attestation = self._browser_live_e2e_start_attestation
        if started_attestation is None:
            # This path is diagnostic-only; v2 promotion validation rejects a
            # missing independent startup snapshot.
            started_attestation = finished_attestation
        attestation = seal_real_gui_e2e_attestation(
            started_attestation,
            finished_attestation,
        )
        if world_tape is not None:
            attestation["world_tape"] = world_tape
        if self.browser_live_recorder is not None:
            summary = self.browser_live_recorder.finish(
                reason=reason,
                attestation=attestation,
            )
            self.bus.emit("status", {
                "message": (
                    "Browser live-speaker observation score "
                    f"{summary.get('strict_browser_live_score', 0.0):.3f} written to "
                    f"{self.browser_live_recorder.output_path}"
                ),
            })
        else:
            summary = {"world_tape": world_tape or {}, "reason": reason}
            self.bus.emit("status", {
                "message": (
                    "Live-speaker World Tape checkpoint written to "
                    f"{(world_tape or {}).get('output_dir', '')}"
                )
            })
        return summary

    def current_media(self) -> MediaFiles:
        return self.media_manager.snapshot().media

    def _bind_world_tape_media(self, media: MediaFiles) -> None:
        recorder = self.world_tape_recorder
        if recorder is None:
            return
        recorder.update_media(media)
        audio = np.asarray(getattr(self.controller, "audio", []), dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(self.controller, "sample_rate", 0) or 0)
        if audio.size > 0 and sample_rate > 0:
            recorder.record_decoded_audio(audio, sample_rate)

    @property
    def media_version(self) -> int:
        return self.media_manager.snapshot().version

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
        self.media_manager.replace(media, self.controller.set_media)
        self._bind_world_tape_media(media)
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
        self.media_manager.replace(media, self.controller.set_media)
        self._bind_world_tape_media(media)
        self.bus.emit("status", {"message": f"Loaded uploaded audio file {display_name}."})
        return media, display_name, written

    def start_browser_stream_url(self, url: str) -> MediaFiles:
        self.bus.emit("status", {"message": f"Preparing browser audio stream for {url}"})
        snapshot = self.media_manager.transition(lambda: self.controller.set_browser_stream(url))
        media = snapshot.media
        self._bind_world_tape_media(media)
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


class WindowServer(ThreadingHTTPServer):
    """HTTP transport that delegates application behavior to one owner."""

    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        args: Any,
        media: MediaFiles,
        bus: EventBus,
        controller: WindowDiarizer,
        *,
        application: LiveWindowApplication | None = None,
    ) -> None:
        self.application = application or LiveWindowApplication(args, media, bus, controller)
        self._transport_close_lock = threading.Lock()
        self._transport_closed = False
        super().__init__(address, Handler)

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __getattr__(self, name: str) -> Any:
        application = self.__dict__.get("application")
        if application is None:
            raise AttributeError(name)
        return getattr(application, name)

    def server_close(self) -> None:
        with self._transport_close_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
        try:
            self.application.close()
        finally:
            super().server_close()
