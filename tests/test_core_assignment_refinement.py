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


class AssignmentRefinementTests(unittest.TestCase):
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

    def test_small_island_refinement_merges_oneoff_flanked_speaker(self) -> None:
        from window.window_validation_replay import replay_cached_window_diarizer
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()
        args = args.with_updates(
            min_embed_seconds=0.0,
            min_first_speaker_seconds=0.1,
            first_speaker_immediate_min_seconds=0.1,
            min_new_speaker_seconds=0.1,
            late_new_speaker_min_seconds=0.1,
            min_new_speaker_words=3,
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=False,
            speaker_refinement_unknown_commit=False,
            allow_speaker_reassignment=False,
            speaker_refinement_small_island_merge=True,
            speaker_refinement_small_island_max_duration=5.0,
            speaker_refinement_small_island_max_segments=3,
        )

        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "alpha beta anchor",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 2.2,
                "end": 4.2,
                "text": "gamma delta anchor",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 4.4,
                "end": 5.4,
                "text": "brief different island",
                "spoken_word_seconds": 1.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 3,
                "start": 5.6,
                "end": 7.6,
                "text": "gamma delta returns",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.99, 0.01], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {
            payload["index"]: payload
            for payload in replay.final_payloads
        }

        self.assertEqual(final_by_index[2]["assigned_speaker"], "S2")
        self.assertTrue(final_by_index[2].get("small_island_merged"))
        self.assertEqual(final_by_index[2].get("small_island_merged_from"), "S3")
        self.assertEqual(final_by_index[2].get("assignment_source"), "small_island_merge")

    def test_speaker_refinement_split_switches_default_on(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()

        self.assertTrue(args.speaker_refinement_unknown_tentative)
        self.assertTrue(args.speaker_refinement_unknown_commit)
        self.assertTrue(args.allow_speaker_reassignment)
        self.assertEqual(args.speaker_refinement_known_min_delta, 0.04)
        self.assertEqual(args.speaker_refinement_final_passes, 1)
        self.assertTrue(args.speaker_refinement_small_island_merge)
        self.assertTrue(args.speaker_refinement_tiny_fragmented_merge)
        self.assertEqual(args.speaker_refinement_tiny_fragmented_max_islands, 3)
        self.assertTrue(args.speaker_refinement_terminal_outro_merge)
        self.assertEqual(args.speaker_refinement_terminal_outro_max_duration, 12.0)
        self.assertTrue(args.speaker_refinement_unknown_same_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_same_speaker_max_duration, 3.0)
        self.assertEqual(args.speaker_refinement_unknown_same_speaker_max_segments, 1)
        self.assertTrue(args.speaker_refinement_unknown_previous_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_duration, 0.75)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_segments, 1)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_previous_gap, 0.35)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_min_next_gap, 0.3)
        self.assertTrue(args.speaker_refinement_unknown_next_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_duration, 1.75)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_segments, 1)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_next_gap, 0.05)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_min_previous_gap, 0.15)
        self.assertTrue(args.speaker_refinement_long_low_confidence_retro_split)
        self.assertEqual(args.speaker_refinement_long_low_confidence_retro_max_similarity, 0.06)

    def test_final_speaker_refinement_runs_configured_passes(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_final_passes=2,
            speaker_refinement_tiny_fragmented_merge=False,
            speaker_refinement_terminal_outro_merge=False,
            speaker_refinement_long_low_confidence_retro_split=False,
            speaker_refinement_unknown_same_speaker_fill=False,
            speaker_refinement_unknown_previous_speaker_fill=False,
            speaker_refinement_unknown_next_speaker_fill=False,
        )
        calls = 0

        def refine() -> None:
            nonlocal calls
            calls += 1

        diarizer._refine_speaker_assignments = refine

        diarizer._finalize_speaker_refinement()

        self.assertEqual(calls, 2)

    def test_tiny_fragmented_refinement_merges_dominant_neighbor_profile(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_tiny_fragmented_merge=True,
            speaker_refinement_tiny_fragmented_max_duration=6.0,
            speaker_refinement_tiny_fragmented_max_segments=8,
            speaker_refinement_tiny_fragmented_min_islands=2,
            speaker_refinement_tiny_fragmented_max_islands=3,
            speaker_refinement_tiny_fragmented_min_neighbor_share=0.5,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, duration: float = 1.0) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.6, "S2": 0.4},
                "unknown_probability": 0.0,
                "top_similarity": 0.6,
                "margin": 0.2,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 2.0),
            1: record(1, "S3"),
            2: record(2, "S2", 2.0),
            3: record(3, "S2", 2.0),
            4: record(4, "S3"),
            5: record(5, "S2", 2.0),
        }

        self.assertEqual(diarizer._merge_tiny_fragmented_speaker_profiles(), 2)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(diarizer._sentence_refinement_records[4]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["tiny_fragmented_profile_merged"])
        self.assertEqual(payload["tiny_fragmented_profile_merged_from"], "S3")
        self.assertEqual(payload["assignment_source"], "tiny_fragmented_profile_merge")

    def test_tiny_fragmented_refinement_skips_too_many_islands(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_tiny_fragmented_merge=True,
            speaker_refinement_tiny_fragmented_max_duration=6.0,
            speaker_refinement_tiny_fragmented_max_segments=8,
            speaker_refinement_tiny_fragmented_min_islands=2,
            speaker_refinement_tiny_fragmented_max_islands=3,
            speaker_refinement_tiny_fragmented_min_neighbor_share=0.5,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, duration: float = 0.5) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.6, "S1": 0.4},
                "unknown_probability": 0.0,
                "top_similarity": 0.6,
                "margin": 0.2,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", 1.0),
            1: record(1, "S3"),
            2: record(2, "S1", 1.0),
            3: record(3, "S3"),
            4: record(4, "S1", 1.0),
            5: record(5, "S3"),
            6: record(6, "S1", 1.0),
            7: record(7, "S3"),
            8: record(8, "S1", 1.0),
        }

        self.assertEqual(diarizer._merge_tiny_fragmented_speaker_profiles(), 0)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[3]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[5]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[7]["assigned_speaker"], "S3")

    def test_terminal_promotional_outro_merges_to_opening_speaker(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_terminal_outro_merge=True,
            speaker_refinement_terminal_outro_max_duration=12.0,
            speaker_refinement_terminal_outro_lookback_segments=2,
            speaker_refinement_terminal_outro_min_target_duration=5.0,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, text: str, duration: float = 2.0) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": text,
                    "start": float(index * 10),
                    "end": float(index * 10) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.7},
                "unknown_probability": 0.0,
                "top_similarity": 0.7,
                "margin": 0.5,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", "Opening narration from the host.", 6.0),
            1: record(1, "S2", "Main guest answer.", 5.0),
            2: record(2, "S3", "Be sure to like and subscribe on YouTube.", 7.0),
        }

        self.assertEqual(diarizer._merge_terminal_promotional_outro(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S1")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["terminal_promotional_outro_merged"])
        self.assertEqual(payload["terminal_promotional_outro_merged_from"], "S3")
        self.assertEqual(payload["assignment_source"], "terminal_promotional_outro_merge")


if __name__ == "__main__":
    unittest.main()
