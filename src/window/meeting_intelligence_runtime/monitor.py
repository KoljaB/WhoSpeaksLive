"""Owned lifecycle for automatic saved-session report generation."""

from __future__ import annotations

import threading
from typing import Callable


class AutoGenerationTracker:
    """Own the baseline and claims used by concurrent saved-session scans."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._baseline_complete = False
        self._seen: set[str] = set()

    def claim_new_saved_sessions(self, summaries: list[dict[str, object]]) -> list[str]:
        saved = {
            str(summary.get("id") or "").strip()
            for summary in summaries
            if str(summary.get("status_label") or "").casefold() == "saved"
            and bool(summary.get("has_transcript"))
            and str(summary.get("id") or "").strip()
        }
        with self._lock:
            if not self._baseline_complete:
                self._seen.update(saved)
                self._baseline_complete = True
                return []
            claimed = sorted(saved - self._seen)
            self._seen.update(claimed)
            return claimed

    def release(self, session_id: str) -> None:
        with self._lock:
            self._seen.discard(str(session_id))


class AutoGenerationMonitor:
    def __init__(self, scan: Callable[[], object], *, interval_seconds: float) -> None:
        self._scan = scan
        self._interval = max(1.0, float(interval_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="meeting-intelligence-auto-generate", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        thread.join(timeout=self._interval + 5.0)
        if thread.is_alive():
            raise RuntimeError("meeting intelligence auto-generation monitor did not stop")
        with self._lock:
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan()
            except Exception as exc:
                print(f"Meeting intelligence auto-generation scan failed: {exc}", flush=True)
            self._stop.wait(self._interval)


__all__ = ["AutoGenerationMonitor", "AutoGenerationTracker"]
