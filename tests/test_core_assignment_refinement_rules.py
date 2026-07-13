from __future__ import annotations

import argparse
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_events import RecordingEventBus



from tests.window_diarizer_support import make_window_diarizer


class AssignmentRefinementRuleTests(unittest.TestCase):
    def base_sentence_payload(self) -> dict[str, object]:
        return {
            "index": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "same speaker evidence",
            "speech_audio_ratio": 1.0,
        }

    def unknown_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "assigned_speaker": None,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "assignment_source": "embedding",
        }

    def confirmed_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "retro_reassigned": True,
            "revision_from": "S3",
            "revision_to": "S6",
            "assigned_speaker": "S6",
            "probabilities": {"unknown": 0.0, "speaker6": 1.0},
            "similarities": {"S6": 0.82},
            "unknown_probability": 0.0,
            "assignment_source": "retro",
        }

    def canonical(self) -> list[dict[str, object]]:
        return [
            {
                "speaker": "canonical_speaker",
                "start": 0.0,
                "end": 3.0,
                "text": "same speaker evidence",
            }
        ]

    def test_unknown_same_speaker_fill_assigns_short_flanked_unknown(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_same_speaker_fill=True,
            speaker_refinement_unknown_same_speaker_max_duration=3.0,
            speaker_refinement_unknown_same_speaker_max_segments=1,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str | None, duration: float = 1.0) -> dict[str, object]:
            assigned = speaker if speaker is not None else "UNKNOWN"
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": assigned,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": "embedding" if speaker is not None else None,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 2.0),
            1: record(1, None, 0.8),
            2: record(2, "S2", 2.0),
        }

        self.assertEqual(diarizer._fill_unknown_same_speaker_islands(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_same_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S2")
        self.assertEqual(payload["assignment_source"], "unknown_same_speaker_island_fill")

    def test_unknown_previous_speaker_fill_assigns_short_tail_before_pause(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_previous_speaker_fill=True,
            speaker_refinement_unknown_previous_speaker_max_duration=0.6,
            speaker_refinement_unknown_previous_speaker_max_segments=1,
            speaker_refinement_unknown_previous_speaker_max_previous_gap=0.05,
            speaker_refinement_unknown_previous_speaker_min_next_gap=0.15,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 0.0, 2.0),
            1: record(1, None, 2.0, 2.25, "non_embedding_candidate"),
            2: record(2, "S4", 2.7, 4.0),
        }

        self.assertEqual(diarizer._fill_unknown_previous_speaker_tails(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_previous_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S2")
        self.assertEqual(payload["assignment_source"], "unknown_previous_speaker_tail_fill")

    def test_unknown_previous_speaker_fill_updates_scan_for_chained_tails(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_previous_speaker_fill=True,
            speaker_refinement_unknown_previous_speaker_max_duration=0.75,
            speaker_refinement_unknown_previous_speaker_max_segments=1,
            speaker_refinement_unknown_previous_speaker_max_previous_gap=0.35,
            speaker_refinement_unknown_previous_speaker_min_next_gap=0.3,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 0.0, 2.0),
            1: record(1, None, 2.0, 2.25, "non_embedding_candidate"),
            2: record(2, None, 2.58, 2.97, "non_embedding_candidate"),
            3: record(3, "S4", 3.45, 5.0),
        }

        self.assertEqual(diarizer._fill_unknown_previous_speaker_tails(), 2)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S2")
        self.assertEqual(diarizer.bus.records[-1]["payload"]["revision_to"], "S2")

    def test_unknown_next_speaker_fill_assigns_short_head_after_pause(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_next_speaker_fill=True,
            speaker_refinement_unknown_next_speaker_max_duration=0.75,
            speaker_refinement_unknown_next_speaker_max_segments=1,
            speaker_refinement_unknown_next_speaker_max_next_gap=0.05,
            speaker_refinement_unknown_next_speaker_min_previous_gap=0.15,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", 0.0, 2.0),
            1: record(1, None, 2.7, 3.35, "non_embedding_candidate"),
            2: record(2, "S3", 3.35, 5.0),
        }

        self.assertEqual(diarizer._fill_unknown_next_speaker_heads(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_next_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S3")
        self.assertEqual(payload["assignment_source"], "unknown_next_speaker_head_fill")

    def test_long_low_confidence_retro_split_creates_final_speaker(self) -> None:
        from speakers.speaker_embedding_cluster import SpeakerMemory

        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_long_low_confidence_retro_split=True,
            speaker_refinement_long_low_confidence_retro_min_duration=4.0,
            speaker_refinement_long_low_confidence_retro_max_similarity=0.06,
            speaker_refinement_long_low_confidence_retro_max_margin=0.04,
            speaker_refinement_long_low_confidence_retro_max_splits=1,
        )
        diarizer.bus = RecordingEventBus()
        diarizer.memory = SpeakerMemory()
        diarizer.memory.upsert_profile("S1", np.array([1.0, 0.0, 0.0], dtype=np.float32), duration_seconds=8.0)
        diarizer.memory.upsert_profile("S2", np.array([0.0, 1.0, 0.0], dtype=np.float32), duration_seconds=8.0)
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        diarizer._sentence_refinement_records = {
            0: {
                "index": 0,
                "base_payload": {
                    "index": 0,
                    "text": "uncertain long segment",
                    "start": 0.0,
                    "end": 12.0,
                },
                "embedding": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "duration_seconds": 12.0,
                "assigned_speaker": "S1",
                "created_speaker": False,
                "probabilities": {"unknown": 0.0, "speaker1": 1.0},
                "similarities": {"S1": 0.044, "S2": 0.038},
                "unknown_probability": 0.0,
                "top_similarity": 0.044,
                "margin": 0.006,
                "quality": 1.0,
                "assignment_source": "retro",
            }
        }

        self.assertEqual(diarizer._split_long_low_confidence_retro_assignments(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[0]["assigned_speaker"], "S3")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["long_low_confidence_retro_split"])
        self.assertEqual(payload["long_low_confidence_retro_split_from"], "S1")
        self.assertEqual(payload["assignment_source"], "long_low_confidence_retro_split")

    def test_speaker_refinement_settings_update_split_switches(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=True,
            speaker_refinement_unknown_commit=True,
            allow_speaker_reassignment=True,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._revisit_unknown_sentences = mock.Mock()
        diarizer._refine_speaker_assignments = mock.Mock()

        result = diarizer.set_speaker_refinement_settings({
            "speaker_refinement_unknown_tentative": False,
            "speaker_refinement_unknown_commit": False,
            "allow_speaker_reassignment": False,
        })

        self.assertEqual(
            result,
            {
                "enabled": True,
                "unknown_tentative": False,
                "unknown_commit": False,
                "allow_reassignment": False,
            },
        )
        diarizer._revisit_unknown_sentences.assert_not_called()
        diarizer._refine_speaker_assignments.assert_not_called()

        result = diarizer.set_speaker_refinement_settings({
            "speaker_refinement_unknown_tentative": True,
            "speaker_refinement_unknown_commit": True,
            "allow_speaker_reassignment": False,
        })

        self.assertTrue(result["unknown_tentative"])
        self.assertTrue(result["unknown_commit"])
        diarizer._revisit_unknown_sentences.assert_called_once()
        diarizer._refine_speaker_assignments.assert_called_once()

    def test_unknown_commit_switch_blocks_retro_unknown_revisit(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(speaker_refinement_unknown_commit=False)
        diarizer.memory = mock.Mock()

        diarizer._revisit_unknown_sentences()

        diarizer.memory.score_existing.assert_not_called()

    def test_tentative_unknown_switch_blocks_prototype_unknown_hints_only(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=False,
            allow_speaker_reassignment=True,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._sentence_refinement_run_lock = threading.Lock()
        diarizer._sentence_refinement_lock = threading.Lock()
        diarizer._sentence_refinement_records = {
            1: {"index": 1},
            2: {"index": 2},
        }
        diarizer._apply_prototype_revision = mock.Mock(return_value=True)
        unknown_revision = argparse.Namespace(
            index=1,
            previous_speaker=None,
            assigned_speaker="S2",
        )
        known_revision = argparse.Namespace(
            index=2,
            previous_speaker="S1",
            assigned_speaker="S2",
        )

        diarizer._assignment_engine = mock.Mock()
        diarizer._assignment_engine.plan_refinement.return_value = argparse.Namespace(
            revisions=(unknown_revision, known_revision),
        )

        diarizer._refine_speaker_assignments()

        diarizer._apply_prototype_revision.assert_called_once_with(known_revision)
        diarizer._assignment_engine.plan_refinement.assert_called_once()

    def test_prototype_unknown_revision_is_tentative_not_committed(self) -> None:
        from window.youtube_window_diarize_gui import build_window_validation_records

        diarizer = make_window_diarizer()
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()
        diarizer._sentence_refinement_records = {
            1: {
                "index": 1,
                "base_payload": self.base_sentence_payload(),
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "duration_seconds": 3.0,
                "assigned_speaker": None,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": "embedding",
            }
        }
        revision = argparse.Namespace(
            index=1,
            previous_speaker=None,
            assigned_speaker="S3",
            prototype_score=0.62,
            prototype_margin=0.21,
            prototype_delta=1.62,
            prototype_scores={"S3": 0.62, "S1": 0.41},
            assignment_source="prototype_unknown_assign",
        )

        self.assertTrue(diarizer._apply_prototype_revision(revision))

        committed = diarizer._sentence_refinement_records[1]
        self.assertIsNone(committed["assigned_speaker"])
        self.assertEqual(committed["provisional_assigned_speaker"], "S3")

        tentative_payload = diarizer.bus.records[-1]["payload"]
        self.assertEqual(tentative_payload["assigned_speaker"], "S3")
        self.assertTrue(tentative_payload["provisional_assignment"])

        records = [
            {"time": 1.0, "event": "sentence", "payload": self.unknown_sentence_payload()},
            *diarizer.bus.records,
        ]
        _analysis_records, final_payloads = build_window_validation_records(records)

        self.assertEqual(len(final_payloads), 1)
        self.assertIsNone(final_payloads[0]["assigned_speaker"])


if __name__ == "__main__":
    unittest.main()
