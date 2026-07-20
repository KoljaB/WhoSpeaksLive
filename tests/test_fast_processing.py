from __future__ import annotations

import argparse
import threading
import unittest
from unittest import mock

import numpy as np

from tests.window_diarizer_support import make_window_diarizer
from window.diarization_run import DiarizationRun
from window.window_domain import EmbeddingSentenceJob


class FastProcessingTests(unittest.TestCase):
    def test_complete_media_is_transcribed_once_with_batched_asr(self) -> None:
        diarizer = make_window_diarizer(audio=np.zeros(320, dtype=np.float32), sample_rate=160)
        sentences = [
            argparse.Namespace(text="First sentence."),
            argparse.Namespace(text="Second sentence."),
        ]
        diarizer._model = object()
        diarizer._transcribe_window = mock.Mock(
            return_value=argparse.Namespace(sentences=sentences, segment_count=2, word_count=4)
        )
        diarizer._emit_sentence = mock.Mock()
        diarizer._pause_realtime_preview = mock.Mock()
        diarizer._revisit_unknown_sentences = mock.Mock()
        diarizer._finalize_speaker_refinement = mock.Mock()
        diarizer._drain_live_memory_update_jobs = mock.Mock()
        diarizer._embedding_jobs = None

        diarizer._run_fast_processing(threading.Event())

        diarizer._transcribe_window.assert_called_once_with(
            diarizer._model,
            0.0,
            diarizer.duration,
            final_flush=True,
            previous_text_ended_sentence=True,
            batched=True,
            batch_size=16,
        )
        self.assertEqual(diarizer._emit_sentence.call_count, 2)
        diarizer._finalize_speaker_refinement.assert_called_once_with()

    def test_fast_embedding_job_defers_live_updates_and_per_sentence_refinement(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._active_run = DiarizationRun(run_id="fast-run", processing_mode="fast")
        diarizer._speaker_generation = 3
        diarizer._embed_audio_chunk = mock.Mock(return_value=np.array([1.0, 0.0], dtype=np.float32))
        diarizer._apply_sentence_embedding_decision = mock.Mock()
        job = EmbeddingSentenceJob(
            index=2,
            base_payload={"start": 1.0, "end": 2.0},
            text="A complete sentence.",
            audio=np.ones(160, dtype=np.float32),
            sample_rate=160,
            duration_seconds=1.0,
            speaker_generation=3,
            run_id="fast-run",
        )

        diarizer._process_sentence_embedding(job)

        kwargs = diarizer._apply_sentence_embedding_decision.call_args.kwargs
        self.assertIsNone(kwargs["live_memory_audio"])
        self.assertIsNone(kwargs["live_memory_sample_rate"])
        self.assertFalse(kwargs["run_speaker_refinement"])

    def test_fast_processing_rejects_streaming_audio(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._streaming_audio = True
        diarizer._model = object()

        with self.assertRaisesRegex(RuntimeError, "fully loaded media"):
            diarizer._run_fast_processing(threading.Event())


if __name__ == "__main__":
    unittest.main()
