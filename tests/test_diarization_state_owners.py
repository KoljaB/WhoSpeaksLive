from __future__ import annotations

import argparse
from types import SimpleNamespace
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from window.audio_timeline import AudioTimeline
from window.diarization_config import DiarizationConfig
from window.diarization_run import DiarizationRun, DiarizationRunState
from window.diarization_session import DiarizationSession
from window.window_diarizer import StartSessionRequest, WindowDiarizer
from window.window_diarizer_live_scoring import WindowLiveScoringMixin
from window.window_diarizer_transcription import WindowTranscriptionMixin
from window.window_domain import EmbeddingSentenceJob, LiveSpeakerMemoryUpdateJob, MediaFiles


class _Bus:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, payload: dict[str, object]) -> None:
        self.records.append((event, payload))

class _TranscriptionHarness(WindowTranscriptionMixin):
    def __init__(self, run: DiarizationRun) -> None:
        self._active_run = run
        self._speaker_generation = 1
        self.bus = _Bus()
        self.embedding_called = False

    def _embed_audio_chunk(self, *_args: object) -> np.ndarray:
        self.embedding_called = True
        return np.array([1.0], dtype=np.float32)


class _LiveScoringHarness(WindowLiveScoringMixin):
    def __init__(self, run: DiarizationRun) -> None:
        self._active_run = run
        self._speaker_generation = 1
        self._speaker_label_generations = {"S1": 2}


class _RunLifecycleHarness:
    def __init__(self, run: DiarizationRun, *, failure: Exception | None = None) -> None:
        self._active_run = run
        self._lifecycle_lock = threading.Lock()
        self._preview_transcriber = None
        self._preview_transcriber_owned = False
        self.dependencies = SimpleNamespace(monotonic=time.monotonic)
        self.bus = _Bus()
        self.failure = failure
        self.embedding_stops = 0
        self.live_stops = 0
        self.final_memory_snapshots = 0

    def _run(self, _stop_event: threading.Event) -> None:
        if self.failure is not None:
            raise self.failure

    def _stop_embedding_worker(self) -> None:
        self.embedding_stops += 1

    def _stop_live_memory_update_worker(self) -> None:
        self.live_stops += 1

    def emit_authoritative_final_speaker_memory_state(self) -> None:
        self.final_memory_snapshots += 1
        self.bus.records.append(
            (
                "internal:speaker_memory_state",
                {"authoritative_final": True},
            )
        )


class AudioTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.media = MediaFiles(
            url="file://test",
            video_id="test",
            audio_file=Path("audio.wav"),
            video_file=Path("video.mp4"),
        )

    def test_file_replacement_is_one_coherent_revision(self) -> None:
        timeline = AudioTimeline(
            self.media,
            audio_loader=lambda _path: (np.array([0.1, 0.2, 0.3], dtype=np.float32), 3),
        )

        replacement = MediaFiles("file://next", "next", Path("next.wav"), Path("next.mp4"))
        snapshot = timeline.replace_file(
            replacement,
            audio_loader=lambda _path: (np.array([0.5, 0.25], dtype=np.float32), 2),
        )

        self.assertEqual(snapshot.media, replacement)
        self.assertEqual(snapshot.sample_rate, 2)
        self.assertEqual(snapshot.duration, 1.0)
        self.assertEqual(snapshot.revision, 1)
        self.assertFalse(snapshot.audio.flags.writeable)

    def test_stream_append_and_cross_chunk_window_share_one_owner(self) -> None:
        timeline = AudioTimeline(
            self.media,
            audio_loader=lambda _path: (np.zeros(1, dtype=np.float32), 4),
        )
        timeline.begin_stream("https://example.test/watch?v=abc")
        timeline.append(np.array([0.1, 0.2], dtype=np.float32), 16000)
        timeline.append(np.array([0.3, 0.4], dtype=np.float32), 16000)

        audio, sample_rate = timeline.window(1 / 16000, 3 / 16000)

        self.assertEqual(sample_rate, 16000)
        np.testing.assert_allclose(audio, [0.2, 0.3])


class DiarizationConfigTests(unittest.TestCase):
    def test_namespace_is_detached_and_updates_are_copy_on_write(self) -> None:
        namespace = argparse.Namespace(language="en", threshold=0.4)
        config = DiarizationConfig.from_namespace(namespace)
        namespace.language = "de"

        updated = config.with_updates(threshold=0.6)

        self.assertEqual(config.language, "en")
        self.assertEqual(config.threshold, 0.4)
        self.assertEqual(updated.threshold, 0.6)


class DiarizationRunTests(unittest.TestCase):
    def test_stop_event_belongs_to_run_and_state_transitions_are_explicit(self) -> None:
        run = DiarizationRun(run_id="run-1", stop_event=threading.Event())
        captured = run.stop_event

        run.mark_running()
        run.request_stop()

        self.assertIs(captured, run.stop_event)
        self.assertTrue(captured.is_set())
        self.assertEqual(run.state, DiarizationRunState.STOPPING)
        run.mark_idle()
        self.assertEqual(run.state, DiarizationRunState.IDLE)

    def test_start_request_normalizes_and_validates_identity(self) -> None:
        request = StartSessionRequest("session-1", "  Weekly   sync  ", "FAST")
        self.assertEqual(request.source_title, "Weekly sync")
        self.assertEqual(request.processing_mode, "fast")
        with self.assertRaises(ValueError):
            StartSessionRequest("not/a/session")
        with self.assertRaisesRegex(ValueError, "processing_mode"):
            StartSessionRequest(processing_mode="turbo")

    def test_embedding_completion_from_previous_run_is_discarded_before_model_work(self) -> None:
        harness = _TranscriptionHarness(DiarizationRun(run_id="new-run"))
        job = EmbeddingSentenceJob(
            index=1,
            base_payload={},
            text="hello",
            audio=np.ones(32, dtype=np.float32),
            sample_rate=16000,
            duration_seconds=0.5,
            speaker_generation=1,
            run_id="old-run",
        )

        harness._process_sentence_embedding(job)

        self.assertFalse(harness.embedding_called)
        self.assertIn("stale diarization run", harness.bus.records[-1][1]["message"])

    def test_live_profile_job_requires_run_and_speaker_generations(self) -> None:
        harness = _LiveScoringHarness(DiarizationRun(run_id="new-run"))
        current = LiveSpeakerMemoryUpdateJob(
            speaker_id="S1",
            audio=np.ones(16, dtype=np.float32),
            sample_rate=16000,
            duration_seconds=0.25,
            speaker_generation=1,
            speaker_label_generation=2,
            run_id="new-run",
        )

        self.assertTrue(harness._live_memory_update_job_is_current(current))
        self.assertFalse(
            harness._live_memory_update_job_is_current(
                LiveSpeakerMemoryUpdateJob(**{**current.__dict__, "run_id": "old-run"})
            )
        )

    def test_natural_completion_joins_auxiliary_worker_and_emits_one_done(self) -> None:
        run = DiarizationRun(run_id="run")
        auxiliary_finished = threading.Event()

        def auxiliary() -> None:
            run.stop_event.wait()
            auxiliary_finished.set()

        run.preview_thread = threading.Thread(target=auxiliary)
        run.preview_thread.start()
        harness = _RunLifecycleHarness(run)

        WindowDiarizer._run_main_worker(harness, run)

        self.assertTrue(auxiliary_finished.is_set())
        self.assertFalse(run.preview_thread.is_alive())
        self.assertEqual(run.state, DiarizationRunState.IDLE)
        self.assertIsNone(harness._active_run)
        self.assertEqual(
            [event for event, _payload in harness.bus.records],
            ["internal:speaker_memory_state", "done"],
        )
        self.assertEqual(harness.final_memory_snapshots, 1)
        self.assertEqual((harness.embedding_stops, harness.live_stops), (1, 1))

    def test_failed_main_worker_keeps_failed_run_until_explicit_cleanup(self) -> None:
        run = DiarizationRun(run_id="run")
        harness = _RunLifecycleHarness(run, failure=RuntimeError("boom"))

        WindowDiarizer._run_main_worker(harness, run)

        self.assertEqual(run.state, DiarizationRunState.FAILED)
        self.assertIn("boom", run.failure)
        self.assertIs(harness._active_run, run)
        self.assertEqual([event for event, _payload in harness.bus.records], ["done"])
        self.assertEqual(harness.final_memory_snapshots, 0)


class DiarizationSessionTests(unittest.TestCase):
    def test_mutating_transaction_advances_version_once(self) -> None:
        session = DiarizationSession()
        before = session.version()

        with session.transaction(mutate=True) as captured:
            self.assertEqual(captured, before)
            self.assertTrue(session.is_current(before))

        self.assertFalse(session.is_current(before))
        self.assertEqual(session.version().value, before.value + 1)


if __name__ == "__main__":
    unittest.main()
