"""Serialized autosave ownership for live window sessions."""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from window.session_store import SessionStore
from window.window_events import EventBus


class SessionPersistenceCoordinator:
    """Own debounce, save serialization, event subscription, and final flush."""

    def __init__(
        self,
        *,
        bus: EventBus,
        store: SessionStore,
        session_snapshot: Callable[[], dict[str, Any]],
        session_id: Callable[[], str],
        write_audio: Callable[[Any], bool],
        translation_snapshot: Callable[[], list[dict[str, Any]]],
        handle_sentence_translation: Callable[[Mapping[str, Any], str], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        self._bus = bus
        self._store = store
        self._session_snapshot = session_snapshot
        self._session_id = session_id
        self._write_audio = write_audio
        self._translation_snapshot = translation_snapshot
        self._handle_sentence_translation = handle_sentence_translation
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        self._timer_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._closed = False
        self._bus.add_listener(self.handle_event)

    def handle_event(self, event: str, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        if event == "sentence":
            self._handle_sentence_translation(payload, self._session_id())
            if payload.get("pending") or payload.get("realtime") or payload.get("provisional_assignment"):
                return
            self.schedule()
        elif event == "translation":
            if str(payload.get("status") or "") in {"complete", "error"}:
                self.schedule()
        elif event == "speakers":
            self.schedule()
        elif event == "done":
            self.cancel()
            self.save(status_label="Saved", write_audio=True)

    def schedule(self) -> None:
        with self._timer_lock:
            if self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._debounce_seconds, self._run_autosave)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def cancel(self) -> None:
        with self._timer_lock:
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def save(self, *, status_label: str, write_audio: bool) -> dict[str, Any] | None:
        with self._save_lock:
            if self._closed and status_label == "Autosaved":
                return None
            snapshot = self._session_snapshot()
            if not snapshot.get("id"):
                return None
            snapshot["translations"] = self._translation_snapshot()
            return self._store.save_snapshot(
                snapshot,
                status_label=status_label,
                write_audio=write_audio,
                audio_writer=self._write_audio,
            )

    def close(self, *, flush: bool = True) -> None:
        with self._timer_lock:
            if self._closed:
                return
            self._closed = True
        self.cancel()
        self._bus.remove_listener(self.handle_event)
        if flush:
            self.save(status_label="Saved", write_audio=False)

    def _run_autosave(self) -> None:
        with self._timer_lock:
            self._timer = None
            if self._closed:
                return
        self.save(status_label="Autosaved", write_audio=False)
