"""Owned lifecycle for audio supplied to realtime capture by a producer."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from realtime.realtime_capture import RealtimeCapture

class ExternalFeedState(str, Enum):
    """Lifecycle states for a recorder fed by an external audio producer."""

    CREATED = "created"
    ACTIVE = "active"
    FINISHING = "finishing"
    CLOSED = "closed"

class ExternalAudioFeed:
    """Own one externally-fed recorder session and all of its resources.

    Callers feed audio and media-time updates through this object instead of
    reaching into :class:`RealtimeCapture`'s recorder, locks, or worker state.
    ``finish`` and ``close`` are idempotent, and leaving the context always
    releases the recorder and joins its final-transcript consumer.
    """

    def __init__(
        self,
        capture: "RealtimeCapture",
        *,
        session_id: str,
        media_id: str,
    ) -> None:
        self._capture = capture
        self.session_id = str(session_id)
        self.media_id = str(media_id)
        self._state = ExternalFeedState.CREATED
        self._state_lock = threading.Lock()
        self._io_lock = threading.RLock()
        self._closed = threading.Event()
        self._stop_event = threading.Event()
        self._recorder: Any = None
        self._consumer_thread: threading.Thread | None = None
        self._media_time_seconds = 0.0

    @property
    def state(self) -> ExternalFeedState:
        with self._state_lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state is ExternalFeedState.CLOSED

    @property
    def media_time_seconds(self) -> float:
        with self._state_lock:
            return self._media_time_seconds

    def __enter__(self) -> "ExternalAudioFeed":
        with self._state_lock:
            if self._state is not ExternalFeedState.CREATED:
                raise RuntimeError("External audio feed contexts cannot be reused.")
        self._capture._activate_external_feed(self)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.finish(emit_status=False)

    def _activate(self) -> None:
        recorder = self._capture._create_recorder(self.session_id, None)
        try:
            recorder.start()
            consumer = threading.Thread(
                target=self._capture._consume_final_text,
                args=(self.session_id, recorder, self._stop_event),
                name="RealtimeExternalFeedFinalConsumer",
                daemon=True,
            )
            with self._state_lock:
                if self._state is not ExternalFeedState.CREATED:
                    raise RuntimeError("External audio feed activation was cancelled.")
                self._recorder = recorder
                self._consumer_thread = consumer
                self._state = ExternalFeedState.ACTIVE
            consumer.start()
        except Exception:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass
            raise

    def feed_audio(
        self,
        audio: np.ndarray,
        *,
        original_sample_rate: int,
        media_end_seconds: float | None = None,
    ) -> None:
        with self._io_lock:
            with self._state_lock:
                if self._state is not ExternalFeedState.ACTIVE:
                    raise RuntimeError("External audio feed is not active.")
                recorder = self._recorder
            recorder.feed_audio(audio, original_sample_rate=original_sample_rate)
            if media_end_seconds is not None:
                self.update_media_time(media_end_seconds)

    def update_media_time(self, current_time_seconds: float) -> None:
        with self._io_lock:
            with self._state_lock:
                if self._state is not ExternalFeedState.ACTIVE:
                    raise RuntimeError("External audio feed is not active.")
            accepted_time = self._capture._set_video_time(
                self.session_id,
                current_time_seconds,
            )
            if accepted_time is None:
                raise ValueError("Media time must be a finite non-negative number.")
            with self._state_lock:
                if self._state is ExternalFeedState.ACTIVE:
                    self._media_time_seconds = accepted_time

    def report_status(self, message: str) -> None:
        """Publish a status event associated with this feed's session."""

        self._capture._status(self.session_id, str(message))

    def finish(
        self,
        *,
        transcript_drain_seconds: float = 0.0,
        embedding_drain_seconds: float = 0.0,
        emit_status: bool = True,
    ) -> None:
        wait_for_owner = False
        with self._state_lock:
            if self._state is ExternalFeedState.CLOSED:
                return
            if self._state is ExternalFeedState.FINISHING:
                wait_for_owner = True
            else:
                self._state = ExternalFeedState.FINISHING
                recorder = self._recorder
                consumer = self._consumer_thread
        if wait_for_owner:
            self._closed.wait()
            return

        try:
            if emit_status:
                self._capture._status(
                    self.session_id,
                    "External audio feed complete; draining final transcripts.",
                )
            drain_seconds = max(0.0, float(transcript_drain_seconds))
            if drain_seconds:
                time.sleep(drain_seconds)
            self._stop_event.set()
            with self._io_lock:
                if recorder is not None:
                    try:
                        recorder.stop()
                    except Exception:
                        pass
                    try:
                        recorder.shutdown()
                    except Exception:
                        pass
            if consumer is not None and consumer is not threading.current_thread():
                consumer.join(timeout=3.0)
            self._capture._drain_embedding_jobs(
                max(0.0, float(embedding_drain_seconds))
            )
        finally:
            self._capture._release_external_feed(self)
            with self._state_lock:
                self._recorder = None
                self._consumer_thread = None
                self._state = ExternalFeedState.CLOSED
            self._closed.set()

    close = finish
