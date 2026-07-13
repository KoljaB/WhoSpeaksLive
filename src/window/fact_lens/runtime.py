"""Owned thread lifecycle for the Fact Lens sidecar runtime."""

from __future__ import annotations

import threading
from typing import Any

from window.fact_lens.domain import FactLensStore


class FactLensRuntime:
    def __init__(
        self,
        *,
        state: FactLensStore,
        stop_event: threading.Event,
        reader: threading.Thread,
        worker: Any | None,
    ) -> None:
        self.state = state
        self.stop_event = stop_event
        self.reader = reader
        self.worker = worker
        self._lock = threading.Lock()
        self._started = False
        self._reader_started = False
        self._closed = False

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("fact lens runtime is closed")
            if self._started:
                return
            self._started = True
            if self.worker is not None:
                self.worker.start()
            self.reader.start()
            self._reader_started = True

    def close(self, *, reader_timeout: float = 35.0, worker_timeout: float = 30.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            reader_started = self._reader_started
        self.stop_event.set()
        failures: list[str] = []
        if reader_started:
            self.reader.join(timeout=max(0.0, float(reader_timeout)))
            if self.reader.is_alive():
                failures.append("source reader did not stop")
        if self.worker is not None:
            try:
                self.worker.stop(timeout=max(0.0, float(worker_timeout)))
            except Exception as exc:
                failures.append(str(exc))
        try:
            self.state.close()
        except Exception as exc:
            failures.append(str(exc))
        if failures:
            raise RuntimeError("fact lens shutdown failed: " + "; ".join(failures))


__all__ = ["FactLensRuntime"]
