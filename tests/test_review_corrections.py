from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import types
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

stream2sentence = types.ModuleType("stream2sentence")
stream2sentence.generate_sentences = lambda *args, **kwargs: []
stream2sentence.init_tokenizer = lambda *args, **kwargs: None
sys.modules.setdefault("stream2sentence", stream2sentence)

from speakers.speaker_embedding_cluster import SpeakerMemory
from window.review_flags import annotate_review
from window.session_store import SessionStore
from window.window_diarizer import WindowDiarizer
from window.window_events import RecordingEventBus


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _fake_diarizer() -> WindowDiarizer:
    diarizer = WindowDiarizer.__new__(WindowDiarizer)
    diarizer.args = SimpleNamespace(embedding_provider="test-provider", live_speaker_ema_count=3)
    diarizer.bus = RecordingEventBus()
    diarizer.media = SimpleNamespace(
        url="",
        video_id="",
        video_file=Path("video.mp4"),
        audio_file=Path("audio.wav"),
    )
    diarizer.duration = 4.1
    diarizer.sample_rate = 16000
    diarizer._audio_lock = threading.Lock()
    diarizer._streaming_audio = False
    diarizer._stream_audio_samples = 0
    diarizer._session_id = "test-session"
    diarizer._session_started_at = "2026-07-09T00:00:00+02:00"
    diarizer._session_source_title = "Review test"
    diarizer.speaker_library_dir = Path(tempfile.gettempdir()) / "whospeaks-test-speakers"
    diarizer.memory = SpeakerMemory(min_first_speaker_seconds=0.1)
    diarizer.memory.add_profile(_unit([1.0, 0.0]), duration_seconds=2.0, sentence_count=1)
    diarizer.memory.add_profile(_unit([0.0, 1.0]), duration_seconds=2.0, sentence_count=1)
    diarizer._speaker_lock = threading.Lock()
    diarizer._speaker_group_name = ""
    diarizer._speaker_metadata = {
        "S1": {"name": "Alice", "source": "detected", "locked": False, "reference_audio": ""},
        "S2": {"name": "Bob", "source": "detected", "locked": False, "reference_audio": ""},
    }
    diarizer._seed_profiles = []
    diarizer._seed_live_profiles = []
    diarizer._sentence_refinement_lock = threading.Lock()
    diarizer._sentence_refinement_records = {
        1: {
            "index": 1,
            "base_payload": {
                "index": 1,
                "text": "Alpha sentence.",
                "start": 0.0,
                "end": 2.0,
                "audio_length_seconds": 2.0,
            },
            "embedding": _unit([1.0, 0.0]),
            "duration_seconds": 2.0,
            "assigned_speaker": "S1",
            "created_speaker": False,
            "probabilities": {"speaker1": 0.8, "speaker2": 0.2, "unknown": 0.0},
            "similarities": {"S1": 1.0, "S2": 0.0},
            "unknown_probability": 0.0,
            "top_similarity": 1.0,
            "margin": 1.0,
            "quality": 1.0,
            "assignment_source": "embedding",
        },
        2: {
            "index": 2,
            "base_payload": {
                "index": 2,
                "text": "Beta sentence.",
                "start": 2.1,
                "end": 4.1,
                "audio_length_seconds": 2.0,
            },
            "embedding": _unit([0.0, 1.0]),
            "duration_seconds": 2.0,
            "assigned_speaker": "S2",
            "created_speaker": False,
            "probabilities": {"speaker1": 0.2, "speaker2": 0.8, "unknown": 0.0},
            "similarities": {"S1": 0.0, "S2": 1.0},
            "unknown_probability": 0.0,
            "top_similarity": 1.0,
            "margin": 1.0,
            "quality": 1.0,
            "assignment_source": "embedding",
        },
    }
    diarizer._correction_history = []
    diarizer._sentence_refinement_run_lock = threading.Lock()
    diarizer._speaker_last_media_end = {}
    diarizer._embedding_jobs = None
    diarizer._live_memory_update_jobs = None
    diarizer._live_memory_update_lock = threading.Lock()
    diarizer._live_embedding_separate = False
    diarizer.live_memory = diarizer.memory
    diarizer._live_probability_history = deque(maxlen=3)
    diarizer._live_speaker_verify_next_at = 0.0
    diarizer._speaker_generation = 0
    diarizer._unknown_lock = threading.Lock()
    diarizer._unknown_sentences = []
    diarizer._recent_unknown_pair_candidates = deque(maxlen=24)
    return diarizer


class ReviewFlagTests(unittest.TestCase):
    def test_review_flags_low_margin_short_and_live_conflict(self) -> None:
        review = annotate_review({
            "assigned_speaker": "S1",
            "margin": 0.02,
            "top_similarity": 0.42,
            "unknown_probability": 0.62,
            "audio_length_seconds": 0.35,
            "live_speaker_id": "S2",
        })

        self.assertTrue(review["needs_review"])
        self.assertIn("low margin", review["reasons"])
        self.assertIn("short audio", review["reasons"])
        self.assertIn("conflicting live/final evidence", review["reasons"])

    def test_user_confirmation_resolves_review(self) -> None:
        review = annotate_review({
            "assigned_speaker": "S1",
            "margin": 0.0,
            "correction": {"status": "user_confirmed"},
        })

        self.assertFalse(review["needs_review"])
        self.assertEqual(review["reasons"], [])


class CorrectionControllerTests(unittest.TestCase):
    def test_session_snapshot_serializes_embeddings_after_review_refactor(self) -> None:
        diarizer = _fake_diarizer()

        diarizer.reassign_sentence(1, "S2")
        snapshot = diarizer.session_snapshot()

        self.assertEqual(snapshot["id"], "test-session")
        self.assertEqual(snapshot["transcript_rows"][0]["assigned_speaker"], "S2")
        self.assertEqual(snapshot["transcript_rows"][0]["correction"]["status"], "user_corrected")
        self.assertEqual(snapshot["embedding_records"][0]["index"], 1)
        self.assertEqual(snapshot["embedding_records"][0]["assigned_speaker"], "S2")

    def test_reassign_sentence_updates_row_and_can_undo(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.reassign_sentence(1, "S2")

        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(result["rows"][0]["correction"]["status"], "user_corrected")
        self.assertFalse(result["rows"][0]["review"]["needs_review"])

        undo = diarizer.undo_last_correction()

        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S1")
        self.assertEqual(undo["rows"][0]["assigned_speaker"], "S1")

    def test_merge_speakers_reassigns_source_rows_and_removes_source_profile(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.merge_speakers("S1", "S2")

        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertNotIn("S1", labels)
        self.assertIn("S2", labels)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")

    def test_split_speaker_moves_sentence_to_new_profile(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.split_speaker("S1", [1])

        new_speaker = result["new_speaker_id"]
        self.assertEqual(new_speaker, "S3")
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertIn("S3", labels)


class SessionReviewMetadataTests(unittest.TestCase):
    def test_session_store_preserves_correction_and_adds_review_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot({
                "id": "review-test",
                "source": {"started_at": "2026-07-08T12:00:00"},
                "transcript_rows": [
                    {
                        "index": 1,
                        "start": 0.0,
                        "end": 0.3,
                        "text": "Hi.",
                        "assigned_speaker": "S1",
                        "margin": 0.01,
                        "top_similarity": 0.4,
                        "unknown_probability": 0.7,
                    },
                    {
                        "index": 2,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Corrected.",
                        "assigned_speaker": "S2",
                        "margin": 0.01,
                        "correction": {"status": "user_corrected", "corrected_speaker": "S2"},
                    },
                ],
                "speaker_state": {"speakers": []},
            })

            opened = store.open_session("review-test")
            rows = opened["transcript_rows"]

            self.assertTrue(rows[0]["review"]["needs_review"])
            self.assertIn("short audio", rows[0]["review"]["reasons"])
            self.assertFalse(rows[1]["review"]["needs_review"])
            self.assertEqual(rows[1]["correction"]["corrected_speaker"], "S2")


if __name__ == "__main__":
    unittest.main()
