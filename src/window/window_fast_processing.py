"""Offline processing path for fully available media files."""

from __future__ import annotations

import threading
import time
from typing import Any

from window.window_text import text_ends_sentence


class WindowFastProcessingMixin:
    """Transcribe a complete file without coupling work to browser playback."""

    def _run_fast_processing(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or self._stop
        model = self._model
        if model is None:
            self.bus.emit("status", {"message": "No ASR backend loaded."})
            return
        if self._streaming_audio:
            raise RuntimeError("Fast processing is available only for fully loaded media files.")

        preview_engine = str(getattr(self.args, "realtime_preview_engine", "off") or "off").lower()
        asr_backend = str(getattr(self.args, "asr_backend", "local") or "local").lower().replace("-", "_")
        if (
            bool(getattr(self.args, "asr_independent_verification", True))
            and asr_backend != "cpu"
            and preview_engine not in {"off", "mock"}
            and getattr(self, "_preview_transcriber", None) is None
        ):
            self.bus.emit(
                "status",
                {"message": "Loading independent ASR acoustic verifier for fast processing."},
            )
            self._load_realtime_preview()

        duration = float(self.duration)
        batch_size = max(1, int(getattr(self.args, "fast_asr_batch_size", 16)))
        started = time.monotonic()
        completed = False
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Fast processing started for {duration:.2f}s of media "
                    f"(batched ASR size {batch_size})."
                )
            },
        )
        try:
            transcript = self._transcribe_window(
                model,
                0.0,
                duration,
                final_flush=True,
                previous_text_ended_sentence=True,
                batched=True,
                batch_size=batch_size,
            )
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Fast ASR completed in {time.monotonic() - started:.2f}s; "
                        f"segments={transcript.segment_count} words={transcript.word_count} "
                        f"sentences={len(transcript.sentences)}."
                    )
                },
            )

            total = len(transcript.sentences)
            for index, sentence in enumerate(transcript.sentences):
                if stop_event.is_set():
                    break
                if not self._wait_for_fast_embedding_capacity(stop_event):
                    break
                self._emit_sentence(index, sentence, 0.0, duration)
                self._last_final_sentence_ended_strong = text_ends_sentence(sentence.text)
                self._final_sentence_count = index + 1
                if index == 0 or (index + 1) % 25 == 0 or index + 1 == total:
                    self.bus.emit(
                        "status",
                        {"message": f"Fast processing queued {index + 1}/{total} sentence embeddings."},
                    )

            completed = not stop_event.is_set() and self._wait_for_fast_embeddings(stop_event, total)
        finally:
            self._pause_realtime_preview()
            if stop_event.is_set():
                jobs = getattr(self, "_embedding_jobs", None)
                if jobs is not None:
                    self._cancel_pending_embedding_jobs(jobs)
            self._revisit_unknown_sentences()
            self._finalize_speaker_refinement()
            self._drain_live_memory_update_jobs()

        if completed:
            self.bus.emit(
                "status",
                {"message": f"Fast processing completed in {time.monotonic() - started:.2f}s."},
            )

    def _wait_for_fast_embedding_capacity(self, stop_event: threading.Event) -> bool:
        jobs = getattr(self, "_embedding_jobs", None)
        if jobs is None:
            return not stop_event.is_set()
        limit = max(1, int(getattr(self.args, "fast_embedding_queue_size", 24)))
        while int(getattr(jobs, "unfinished_tasks", 0)) >= limit:
            if stop_event.wait(0.05):
                return False
            worker = getattr(self, "_embedding_thread", None)
            if worker is not None and not worker.is_alive():
                raise RuntimeError("Speaker embedding worker stopped during fast processing.")
        return True
    def _wait_for_fast_embeddings(self, stop_event: threading.Event, total: int) -> bool:
        jobs = getattr(self, "_embedding_jobs", None)
        if jobs is None:
            return not stop_event.is_set()
        next_status_at = 0.0
        while int(getattr(jobs, "unfinished_tasks", 0)) > 0:
            if stop_event.wait(0.05):
                return False
            worker = getattr(self, "_embedding_thread", None)
            if worker is not None and not worker.is_alive():
                raise RuntimeError("Speaker embedding worker stopped before its queue was drained.")
            now = time.monotonic()
            if now >= next_status_at:
                remaining = int(getattr(jobs, "unfinished_tasks", 0))
                self.bus.emit(
                    "status",
                    {"message": f"Fast processing speaker embeddings: {max(0, total - remaining)}/{total}."},
                )
                next_status_at = now + 2.0
        return True
