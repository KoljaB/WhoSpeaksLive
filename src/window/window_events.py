"""Server-sent-event fanout helpers for the window diarization GUI."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable

from common.audio_utils import json_dumps
from window.window_config import _console_print

class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[tuple[str, str]]] = []
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self._internal_listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def add_listener(self, listener: Callable[[str, dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[str, dict[str, Any]], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def add_internal_listener(
        self,
        listener: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Observe private diagnostics without publishing them through SSE."""

        with self._lock:
            self._internal_listeners.append(listener)

    def remove_internal_listener(
        self,
        listener: Callable[[str, dict[str, Any]], None],
    ) -> None:
        with self._lock:
            if listener in self._internal_listeners:
                self._internal_listeners.remove(listener)

    def has_internal_listeners(self) -> bool:
        with self._lock:
            return bool(self._internal_listeners)

    def emit_internal(self, event: str, payload: dict[str, Any]) -> None:
        """Send a recorder-only event that never reaches browser clients."""

        with self._lock:
            listeners = list(self._internal_listeners)
        for listener in listeners:
            try:
                listener(event, payload)
            except Exception as exc:
                _console_print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Internal event listener failed: {type(exc).__name__}: {exc}"
                )

    def subscribe(self) -> queue.Queue[tuple[str, str]]:
        subscriber: queue.Queue[tuple[str, str]] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[tuple[str, str]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event in {"status", "error", "done"}:
            message = str(payload.get("message") or payload.get("error") or event)
            _console_print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        line = json_dumps(payload)
        with self._lock:
            subscribers = list(self._subscribers)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event, payload)
            except Exception as exc:
                _console_print(f"[{datetime.now().strftime('%H:%M:%S')}] Event listener failed: {type(exc).__name__}: {exc}")
        for subscriber in subscribers:
            subscriber.put((event, line))


class RecordingEventBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.done = threading.Event()
        self._records_lock = threading.Lock()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        with self._records_lock:
            self.records.append({
                "time": time.time(),
                "event": event,
                "payload": json.loads(json_dumps(payload)),
            })
        if event == "done":
            self.done.set()
        super().emit(event, payload)


