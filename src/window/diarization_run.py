"""Explicit lifecycle ownership for one window-diarization run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import uuid


class DiarizationRunState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class DiarizationRun:
    """All replaceable synchronization resources belonging to one run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stop_event: threading.Event = field(default_factory=threading.Event)
    state: DiarizationRunState = DiarizationRunState.STARTING
    main_thread: threading.Thread | None = None
    preview_thread: threading.Thread | None = None
    live_probe_thread: threading.Thread | None = None
    done_emitted: bool = False
    failure: str = ""
    processing_mode: str = "playback"

    def threads(self) -> tuple[threading.Thread, ...]:
        return tuple(
            thread
            for thread in (self.main_thread, self.preview_thread, self.live_probe_thread)
            if thread is not None
        )

    def request_stop(self) -> None:
        if self.state not in {DiarizationRunState.IDLE, DiarizationRunState.FAILED}:
            self.state = DiarizationRunState.STOPPING
        self.stop_event.set()

    def mark_running(self) -> None:
        self.state = DiarizationRunState.RUNNING

    def mark_idle(self) -> None:
        self.state = DiarizationRunState.IDLE

    def mark_failed(self, detail: str) -> None:
        self.failure = str(detail)
        self.state = DiarizationRunState.FAILED
        self.stop_event.set()
