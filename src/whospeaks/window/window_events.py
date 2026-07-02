"""Server-sent-event fanout helpers for the window diarization GUI."""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any

from whospeaks.common.audio_utils import json_dumps
from whospeaks.window.window_config import _console_print

class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[tuple[str, str]]] = []
        self._lock = threading.Lock()

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
        for subscriber in subscribers:
            subscriber.put((event, line))


class RecordingEventBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.done = threading.Event()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.records.append({
            "time": time.time(),
            "event": event,
            "payload": json.loads(json_dumps(payload)),
        })
        if event == "done":
            self.done.set()
        super().emit(event, payload)


