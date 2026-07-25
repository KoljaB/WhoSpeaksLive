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

from speakers.speaker_embedding_cluster import SpeakerDecision
from window.window_domain import PendingUnknownSentence
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

    def test_retro_reassignment_keeps_weak_match_unknown_with_one_profile(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_commit=True,
            retro_reassign_min_similarity=0.02,
            retro_reassign_min_margin=0.0,
            update_unknown_max=0.4289,
            same_speaker_similarity=0.45,
        )
        diarizer._unknown_sentences = [
            PendingUnknownSentence(
                index=0,
                base_payload=self.base_sentence_payload(),
                embedding=np.array([0.0, 1.0], dtype=np.float32),
                duration_seconds=1.95,
            )
        ]
        diarizer.memory = mock.Mock()
        diarizer.memory.profile_count.return_value = 1
        diarizer.memory.score_existing.return_value = SpeakerDecision(
            assigned_speaker="S1",
            created_speaker=False,
            probabilities={"unknown": 0.9967, "speaker1": 0.0033},
            similarities={"S1": 0.0903},
            unknown_probability=0.9967,
            top_similarity=0.0903,
            margin=1.0,
            quality=0.6977,
            assignment_source="retro",
        )
        diarizer._remove_unknown_sentence = mock.Mock(return_value=True)
        diarizer._revisit_unknown_sentences()

        diarizer._remove_unknown_sentence.assert_not_called()

    def test_retro_reassignment_still_commits_confident_known_voice(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_commit=True,
            retro_reassign_min_similarity=0.02,
            retro_reassign_min_margin=0.0,
            update_unknown_max=0.4289,
            same_speaker_similarity=0.45,
        )
        diarizer._unknown_sentences = [
            PendingUnknownSentence(
                index=0,
                base_payload=self.base_sentence_payload(),
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                duration_seconds=1.95,
            )
        ]
        diarizer.memory = mock.Mock()
        diarizer.memory.profile_count.return_value = 1
        diarizer.memory.score_existing.return_value = SpeakerDecision(
            assigned_speaker="S1",
            created_speaker=False,
            probabilities={"unknown": 0.92, "speaker1": 0.08},
            similarities={"S1": 0.72},
            unknown_probability=0.92,
            top_similarity=0.72,
            margin=1.0,
            quality=0.6977,
            assignment_source="retro",
        )
        diarizer._remove_unknown_sentence = mock.Mock(return_value=False)

        diarizer._revisit_unknown_sentences()

        diarizer._remove_unknown_sentence.assert_called_once_with(0)

    def test_retro_reassignment_keeps_relaxed_recovery_with_multiple_profiles(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_commit=True,
            retro_reassign_min_similarity=0.02,
            retro_reassign_min_margin=0.0,
            same_speaker_similarity=0.45,
            known_speaker_min_similarity=0.5563,
            min_new_speaker_seconds=1.8,
        )
        diarizer._unknown_sentences = [
            PendingUnknownSentence(
                index=0,
                base_payload=self.base_sentence_payload(),
                embedding=np.array([0.0, 1.0], dtype=np.float32),
                duration_seconds=1.95,
            )
        ]
        diarizer.memory = mock.Mock()
        diarizer.memory.profile_count.return_value = 2
        diarizer.memory.score_existing.return_value = SpeakerDecision(
            assigned_speaker="S2",
            created_speaker=False,
            probabilities={"unknown": 0.9967, "speaker2": 0.0033},
            similarities={"S1": 0.05, "S2": 0.09},
            unknown_probability=0.9967,
            top_similarity=0.09,
            margin=0.04,
            quality=0.6977,
            assignment_source="retro",
        )
        diarizer._remove_unknown_sentence = mock.Mock(return_value=False)

        diarizer._revisit_unknown_sentences()

        diarizer._remove_unknown_sentence.assert_called_once_with(0)

    def test_short_unknown_requires_strong_match_with_multiple_profiles(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_commit=True,
            retro_reassign_min_similarity=0.0,
            retro_reassign_min_margin=0.0,
            same_speaker_similarity=0.34,
            known_speaker_min_similarity=0.5563,
            min_new_speaker_seconds=1.8358,
        )
        diarizer._unknown_sentences = [
            PendingUnknownSentence(
                index=0,
                base_payload={
                    **self.base_sentence_payload(),
                    "spoken_word_seconds": 0.46,
                },
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                duration_seconds=1.276,
            )
        ]
        diarizer.memory = mock.Mock()
        diarizer.memory.profile_count.return_value = 3
        diarizer.memory.score_existing.return_value = SpeakerDecision(
            assigned_speaker=None,
            created_speaker=False,
            probabilities={"unknown": 0.46, "speaker1": 0.54},
            similarities={"S1": 0.4372},
            unknown_probability=0.46,
            top_similarity=0.4372,
            margin=0.14,
            quality=0.25,
            assignment_source="retro",
        )
        diarizer._remove_unknown_sentence = mock.Mock(return_value=True)
        diarizer._sentence_refinement_records = {
            99: {
                "index": 99,
                "assignment_source": "short_distinct_new_speaker",
            }
        }

        diarizer._revisit_unknown_sentences()

        diarizer._remove_unknown_sentence.assert_not_called()
        self.assertEqual(
            diarizer.memory.score_existing.call_args.kwargs["min_similarity"],
            0.5563,
        )

    def test_short_unknown_same_speaker_fill_requires_voice_support(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_same_speaker_fill=True,
            speaker_refinement_unknown_same_speaker_max_duration=3.0,
            speaker_refinement_unknown_same_speaker_max_segments=1,
            min_new_speaker_seconds=1.8358,
            same_speaker_similarity=0.34,
            known_speaker_min_similarity=0.5563,
        )
        diarizer._sentence_refinement_records = {
            0: {
                "index": 0,
                "assigned_speaker": "S1",
                "duration_seconds": 2.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "similarities": {"S1": 1.0},
            },
            1: {
                "index": 1,
                "assigned_speaker": None,
                "duration_seconds": 1.276,
                "base_payload": {"spoken_word_seconds": 0.46},
                "embedding": np.array([0.44, 0.90], dtype=np.float32),
                "similarities": {"S1": 0.4372},
            },
            2: {
                "index": 2,
                "assigned_speaker": "S1",
                "duration_seconds": 2.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "similarities": {"S1": 1.0},
            },
            3: {
                "index": 3,
                "assigned_speaker": "S4",
                "duration_seconds": 2.0,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "similarities": {"S4": 1.0},
                "assignment_source": "prototype_reassign",
                "short_distinct_origin": True,
            },
        }
        diarizer._apply_unknown_same_speaker_fill = mock.Mock(return_value=1)

        applied = diarizer._fill_unknown_same_speaker_islands()

        self.assertEqual(applied, 0)
        diarizer._apply_unknown_same_speaker_fill.assert_not_called()

    def test_short_distinct_new_speaker_requires_every_evidence_gate(self) -> None:
        base_args = {
            "short_distinct_new_speaker_min_spoken_seconds": 1.4,
            "short_distinct_new_speaker_min_words": 6,
            "short_distinct_new_speaker_min_unknown_probability": 0.90,
            "short_distinct_new_speaker_max_similarity": 0.20,
            "short_distinct_new_speaker_max_margin": 0.05,
            "min_new_speaker_seconds": 1.8358,
            "new_speaker_confirmation_count": 1,
            "max_speakers": 10,
        }
        base_payload = {
            "spoken_word_seconds": 1.52,
            "new_speaker_anchor_words": 7,
        }

        def make_diarizer(
            *,
            args_update: dict[str, object] | None = None,
            payload_update: dict[str, object] | None = None,
            unknown_probability: float = 0.9844,
            top_similarity: float = 0.1243,
            margin: float = 0.0155,
            profile_count: int = 3,
        ):
            diarizer = make_window_diarizer()
            args = {**base_args, **(args_update or {})}
            payload = {**base_payload, **(payload_update or {})}
            diarizer.args = argparse.Namespace(**args)
            diarizer.memory = mock.Mock()
            diarizer.memory.profile_count.return_value = profile_count
            diarizer.memory.export_profiles.return_value = [
                {"label": f"S{index}"} for index in range(1, profile_count + 1)
            ]
            diarizer.memory.score_existing.return_value = SpeakerDecision(
                assigned_speaker=None,
                created_speaker=False,
                probabilities={"unknown": unknown_probability},
                similarities={"S1": top_similarity, "S2": top_similarity - margin},
                unknown_probability=unknown_probability,
                top_similarity=top_similarity,
                margin=margin,
                quality=0.82,
                assignment_source="retro",
            )
            return diarizer, payload

        diarizer, payload = make_diarizer()
        decision = diarizer._short_distinct_new_speaker_decision(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            1.77,
            payload,
            allow_new_speaker=True,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.assigned_speaker, "S4")
        self.assertTrue(decision.created_speaker)
        self.assertEqual(decision.assignment_source, "short_distinct_new_speaker")
        diarizer.memory.upsert_profile.assert_called_once()

        rejected_cases = [
            ("disabled", {"short_distinct_new_speaker_min_spoken_seconds": -1.0}, {}, 0.9844, 0.1243, 0.0155, 3),
            ("spoken", {}, {"spoken_word_seconds": 1.39}, 0.9844, 0.1243, 0.0155, 3),
            ("words", {}, {"new_speaker_anchor_words": 5}, 0.9844, 0.1243, 0.0155, 3),
            ("unknown", {}, {}, 0.8999, 0.1243, 0.0155, 3),
            ("similarity", {}, {}, 0.9844, 0.2001, 0.0155, 3),
            ("margin", {}, {}, 0.9844, 0.1243, 0.0501, 3),
            ("confirmation", {"new_speaker_confirmation_count": 2}, {}, 0.9844, 0.1243, 0.0155, 3),
            ("capacity", {}, {}, 0.9844, 0.1243, 0.0155, 10),
        ]
        for name, args_update, payload_update, unknown, similarity, test_margin, count in rejected_cases:
            with self.subTest(name=name):
                diarizer, payload = make_diarizer(
                    args_update=args_update,
                    payload_update=payload_update,
                    unknown_probability=unknown,
                    top_similarity=similarity,
                    margin=test_margin,
                    profile_count=count,
                )
                decision = diarizer._short_distinct_new_speaker_decision(
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    1.77,
                    payload,
                    allow_new_speaker=True,
                )
                self.assertIsNone(decision)
                diarizer.memory.upsert_profile.assert_not_called()

    def test_ordinary_creation_keeps_protected_origin(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            short_distinct_new_speaker_min_spoken_seconds=1.5,
            short_distinct_new_speaker_max_similarity=0.17,
        )
        diarizer._sentence_refinement_records = {
            18: {
                "index": 18,
                "base_payload": {"start": 67.37, "end": 70.31},
                "assigned_speaker": "S1",
            }
        }
        base_payload = {
            **self.base_sentence_payload(),
            "start": 86.03,
            "end": 87.94,
        }
        diarizer._record_sentence_assignment(
            19,
            base_payload,
            np.array([0.0, 1.0], dtype=np.float32),
            1.87,
            {
                "assigned_speaker": "S4",
                "created_speaker": True,
                "top_similarity": 0.55,
                "assignment_source": "embedding",
            },
        )

        record = diarizer._sentence_refinement_records[19]
        self.assertTrue(record["short_distinct_origin"])
        self.assertTrue(diarizer._has_short_distinct_speaker_record())

    def test_legacy_small_island_refinement_still_merges_oneoff_speaker(self) -> None:
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

    def test_non_speech_annotation_cannot_seed_a_duplicate_speaker_profile(self) -> None:
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
            min_new_speaker_words=1,
            new_speaker_confirmation_count=1,
            speaker_refinement=False,
        )
        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "Host anchor sentence.",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 4.0,
                "end": 7.0,
                "text": "Guest anchor sentence.",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 8.0,
                "end": 16.0,
                "text": "AND APPLAUSE No.",
                "spoken_word_seconds": 8.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 3,
                "start": 17.0,
                "end": 27.0,
                "text": "That was so good!",
                "spoken_word_seconds": 10.0,
                "speech_audio_ratio": 1.0,
                "asr_review": {
                    "needs_review": True,
                    "reasons": [
                        "independent ASR heard speech but disagreed with the retained text"
                    ],
                },
            },
            {
                "index": 4,
                "start": 20.0,
                "end": 30.0,
                "text": "Clean third speaker answer.",
                "spoken_word_seconds": 10.0,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {payload["index"]: payload for payload in replay.final_payloads}

        self.assertFalse(final_by_index[2]["created_speaker"])
        self.assertFalse(final_by_index[3]["created_speaker"])
        self.assertEqual(final_by_index[4]["assigned_speaker"], "S3")
        self.assertTrue(final_by_index[4]["created_speaker"])
        self.assertEqual(len(replay.final_profiles), 3)

    def test_exact_repeated_phrase_cannot_seed_a_speaker_profile(self) -> None:
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
            min_new_speaker_words=1,
            new_speaker_confirmation_count=1,
            speaker_refinement=False,
        )
        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "Host anchor sentence.",
                "spoken_word_seconds": 3.0,
            },
            {
                "index": 1,
                "start": 80.98,
                "end": 82.95,
                "text": "I didn't want to cry, I didn't want to cry.",
                "spoken_word_seconds": 1.97,
            },
            {
                "index": 2,
                "start": 83.70,
                "end": 86.53,
                "text": "I had to come back, I had to come back.",
                "spoken_word_seconds": 2.83,
            },
            {
                "index": 3,
                "start": 100.0,
                "end": 106.0,
                "text": "Clean guest answer.",
                "spoken_word_seconds": 6.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {payload["index"]: payload for payload in replay.final_payloads}

        self.assertFalse(final_by_index[1]["created_speaker"])
        self.assertFalse(final_by_index[2]["created_speaker"])
        self.assertTrue(final_by_index[3]["created_speaker"])
        self.assertEqual(len(replay.final_profiles), 2)

    def test_short_common_repetition_can_still_seed_a_speaker_profile(self) -> None:
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
            min_new_speaker_words=1,
            new_speaker_confirmation_count=1,
            speaker_refinement=False,
        )
        replay = replay_cached_window_diarizer(
            [
                {"index": 0, "start": 0.0, "end": 3.0, "text": "Host anchor.", "spoken_word_seconds": 3.0},
                {"index": 1, "start": 4.0, "end": 6.0, "text": "Thank you. Thank you.", "spoken_word_seconds": 2.0},
            ],
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
            args,
            defer_speaker_refinement=False,
        )

        self.assertTrue(replay.final_payloads[1]["created_speaker"])
        self.assertEqual(len(replay.final_profiles), 2)

    def test_late_short_fourth_speaker_stays_pending_without_confirmation(self) -> None:
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
            min_new_speaker_words=1,
            new_speaker_confirmation_count=1,
            single_sentence_new_speaker_min_seconds=5.0,
            speaker_refinement=False,
        )
        sentences = [
            {"index": 0, "start": 0.0, "end": 6.0, "text": "First speaker.", "spoken_word_seconds": 6.0},
            {"index": 1, "start": 10.0, "end": 16.0, "text": "Second speaker.", "spoken_word_seconds": 6.0},
            {"index": 2, "start": 20.0, "end": 26.0, "text": "Third speaker.", "spoken_word_seconds": 6.0},
            {
                "index": 3,
                "start": 200.0,
                "end": 203.31,
                "text": "That was funny!",
                "spoken_word_seconds": 3.22,
            },
        ]
        embeddings = [
            np.eye(4, dtype=np.float32)[index]
            for index in range(4)
        ]

        replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {payload["index"]: payload for payload in replay.final_payloads}

        self.assertEqual(len(replay.final_profiles), 3)
        self.assertIsNone(final_by_index[3]["assigned_speaker"])
        self.assertFalse(final_by_index[3]["created_speaker"])
        self.assertEqual(final_by_index[3]["assignment_source"], "new_speaker_pending")

    def test_short_distinct_origin_is_not_erased_by_topology_cleanup(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_small_island_merge=True,
            speaker_refinement_small_island_max_duration=5.0,
            speaker_refinement_small_island_max_segments=3,
        )
        diarizer._sentence_refinement_records = {
            0: {
                "index": 0,
                "assigned_speaker": "S1",
                "duration_seconds": 2.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "similarities": {"S1": 1.0},
                "assignment_source": "embedding",
            },
            1: {
                "index": 1,
                "assigned_speaker": "S4",
                "duration_seconds": 1.77,
                "embedding": np.array([0.12, 0.99], dtype=np.float32),
                "similarities": {"S1": 0.70},
                "assignment_source": "prototype_reassign",
                "short_distinct_origin": True,
            },
            2: {
                "index": 2,
                "assigned_speaker": "S1",
                "duration_seconds": 2.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "similarities": {"S1": 1.0},
                "assignment_source": "embedding",
            },
        }
        diarizer._apply_small_island_merge = mock.Mock(return_value=1)

        applied = diarizer._merge_small_speaker_islands()

        self.assertEqual(applied, 0)
        diarizer._apply_small_island_merge.assert_not_called()

    def test_evidence_supported_small_island_can_still_merge(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            speaker_refinement_small_island_merge=True,
            speaker_refinement_small_island_max_duration=5.0,
            speaker_refinement_small_island_max_segments=3,
            speaker_refinement_unknown_min_similarity=0.20,
        )
        diarizer._sentence_refinement_records = {
            0: {
                "index": 0,
                "assigned_speaker": "S2",
                "duration_seconds": 2.0,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "similarities": {"S2": 1.0},
            },
            1: {
                "index": 1,
                "assigned_speaker": "S3",
                "duration_seconds": 1.0,
                "embedding": np.array([0.95, 0.30], dtype=np.float32),
                "similarities": {"S2": 0.30},
            },
            2: {
                "index": 2,
                "assigned_speaker": "S2",
                "duration_seconds": 2.0,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "similarities": {"S2": 1.0},
            },
        }
        diarizer._apply_small_island_merge = mock.Mock(return_value=1)

        applied = diarizer._merge_small_speaker_islands()

        self.assertEqual(applied, 1)
        diarizer._apply_small_island_merge.assert_called_once_with(
            [1],
            "S3",
            "S2",
            1.0,
        )

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
