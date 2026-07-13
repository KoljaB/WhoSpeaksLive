from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import types
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
from window.window_speaker_refinement import (
    SpeakerRefinementConfig,
    build_speaker_prototypes,
    find_speaker_prototype_revisions,
)
from tests.window_diarizer_support import make_window_diarizer


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _fake_diarizer() -> WindowDiarizer:
    diarizer = make_window_diarizer(
        audio=np.zeros(int(4.1 * 16_000), dtype=np.float32),
        sample_rate=16_000,
    )
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
        self.assertEqual(result["rows"][0]["correction"]["rejected_speakers"], ["S1"])
        self.assertFalse(result["rows"][0]["review"]["needs_review"])

        undo = diarizer.undo_last_correction()

        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S1")
        self.assertEqual(undo["rows"][0]["assigned_speaker"], "S1")

    def test_reassign_sentence_does_not_keep_current_target_rejected(self) -> None:
        diarizer = _fake_diarizer()

        diarizer.reassign_sentence(1, "S2")
        result = diarizer.reassign_sentence(1, "S1")

        correction = result["rows"][0]["correction"]
        self.assertEqual(correction["corrected_speaker"], "S1")
        self.assertEqual(correction["rejected_speakers"], ["S2"])

    def test_bulk_reassign_sentences_updates_rows_with_single_undo(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.reassign_sentences([1, 2], "S1")

        self.assertEqual([row["assigned_speaker"] for row in result["rows"]], ["S1", "S1"])
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S1")

        undo = diarizer.undo_last_correction()

        restored = {row["index"]: row["assigned_speaker"] for row in undo["rows"]}
        self.assertEqual(restored[1], "S1")
        self.assertEqual(restored[2], "S2")

    def test_bulk_mark_sentences_correct_updates_rows_with_single_undo(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.mark_sentences_correct([1, 2])

        self.assertEqual([row["correction"]["status"] for row in result["rows"]], ["user_confirmed", "user_confirmed"])
        self.assertEqual([row["correction"]["corrected_speaker"] for row in result["rows"]], ["S1", "S2"])
        self.assertEqual([row["correction"]["updates_memory"] for row in result["rows"]], [True, True])

        undo = diarizer.undo_last_correction()

        self.assertNotIn("correction", diarizer._sentence_refinement_records[1])
        self.assertNotIn("correction", diarizer._sentence_refinement_records[2])
        self.assertEqual({row["index"] for row in undo["rows"]}, {1, 2})

    def test_mark_correct_refreshes_speaker_memory_from_confirmed_records(self) -> None:
        diarizer = _fake_diarizer()
        confirmed_embedding = _unit([0.6, 0.8])
        diarizer._sentence_refinement_records[1]["embedding"] = confirmed_embedding

        diarizer.mark_sentence_correct(1)

        profiles = {
            profile["label"]: np.asarray(profile["centroid"], dtype=np.float32)
            for profile in diarizer.memory.export_profiles()
        }
        self.assertTrue(np.allclose(profiles["S1"], confirmed_embedding))

    def test_merge_speakers_reassigns_source_rows_and_removes_source_profile(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.merge_speakers("S1", "S2")

        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertNotIn("S1", labels)
        self.assertIn("S2", labels)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")

    def test_delete_speaker_moves_rows_to_unknown_and_removes_profile(self) -> None:
        diarizer = _fake_diarizer()
        diarizer._speaker_last_media_end["S1"] = 2.0

        result = diarizer.delete_speaker("S1")

        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertNotIn("S1", labels)
        self.assertIn("S2", labels)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertIsNone(row["assigned_speaker"])
        self.assertEqual(row["probabilities"], {"unknown": 1.0})
        self.assertEqual(row["unknown_probability"], 1.0)
        self.assertEqual(row["assignment_source"], "user_deleted_speaker")
        self.assertEqual(row["correction"]["status"], "speaker_deleted")
        self.assertEqual(row["correction"]["deleted_speaker"], "S1")
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], None)
        self.assertNotIn("S1", {profile["label"] for profile in diarizer.memory.export_profiles()})
        self.assertNotIn("S1", diarizer._speaker_last_media_end)

        undo = diarizer.undo_last_correction()

        restored = {speaker["id"] for speaker in undo["speaker_state"]["speakers"]}
        self.assertIn("S1", restored)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S1")
        self.assertEqual(diarizer._speaker_last_media_end["S1"], 2.0)

    def test_delete_empty_speaker_removes_profile_and_is_undoable(self) -> None:
        diarizer = _fake_diarizer()
        empty_label = diarizer.memory.add_profile(_unit([1.0, 1.0]), duration_seconds=0.1, sentence_count=1)
        self.assertEqual(empty_label, "S3")
        diarizer._speaker_metadata[empty_label] = {
            "name": "Empty",
            "source": "detected",
            "locked": False,
            "reference_audio": "",
        }

        result = diarizer.delete_speaker(empty_label)

        self.assertEqual(result["rows"], [])
        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertNotIn(empty_label, labels)

        undo = diarizer.undo_last_correction()

        restored = {speaker["id"] for speaker in undo["speaker_state"]["speakers"]}
        self.assertIn(empty_label, restored)

    def test_delete_speaker_clears_marked_correct_confirmation_on_rows(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.mark_sentence_correct(1)

        result = diarizer.delete_speaker("S1")

        row = result["rows"][0]
        self.assertIsNone(row["assigned_speaker"])
        self.assertEqual(row["correction"]["status"], "speaker_deleted")
        self.assertNotEqual(row["correction"]["status"], "user_confirmed")

    def test_deleted_speaker_rows_are_not_prototype_assigned_from_unknown(self) -> None:
        config = SpeakerRefinementConfig(
            prototype_min_duration=0.0,
            unknown_min_similarity=-1.0,
            unknown_min_margin=0.0,
        )
        rows = [
            {
                "index": 1,
                "assigned_speaker": None,
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 1.0,
                "unknown_probability": 1.0,
            },
            {
                "index": 2,
                "assigned_speaker": "S1",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
            {
                "index": 3,
                "assigned_speaker": "S2",
                "embedding": _unit([0.0, 1.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
        ]

        baseline = find_speaker_prototype_revisions(rows, config)
        self.assertEqual([(revision.index, revision.assigned_speaker) for revision in baseline], [(1, "S1")])

        rows[0]["correction"] = {
            "status": "speaker_deleted",
            "action": "delete_speaker",
            "deleted_speaker": "S3",
            "corrected_speaker": None,
        }

        revisions = find_speaker_prototype_revisions(rows, config)

        self.assertEqual(revisions, [])

    def test_stale_prototype_unknown_revision_cannot_change_deleted_speaker_row(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.delete_speaker("S1")
        revision = SimpleNamespace(
            index=1,
            previous_speaker=None,
            assigned_speaker="S2",
            prototype_score=1.0,
            prototype_margin=1.0,
            prototype_scores={"S2": 1.0},
            assignment_source="prototype_unknown_assign",
        )

        applied = diarizer._apply_prototype_revision(revision)

        self.assertFalse(applied)
        self.assertIsNone(diarizer._sentence_refinement_records[1]["assigned_speaker"])

    def test_split_speaker_moves_sentence_to_new_profile(self) -> None:
        diarizer = _fake_diarizer()

        result = diarizer.split_speaker("S1", [1])

        new_speaker = result["new_speaker_id"]
        self.assertEqual(new_speaker, "S3")
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        labels = {speaker["id"] for speaker in result["speaker_state"]["speakers"]}
        self.assertIn("S3", labels)

    def test_prototype_reassignment_skips_user_rejected_speaker(self) -> None:
        config = SpeakerRefinementConfig(
            prototype_min_duration=0.0,
            known_min_similarity=-1.0,
            known_min_delta=0.0,
        )
        rows = [
            {
                "index": 1,
                "assigned_speaker": "S2",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 1.0,
                "unknown_probability": 0.0,
                "top_similarity": 1.0,
                "margin": 1.0,
            },
            {
                "index": 2,
                "assigned_speaker": "S1",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
            {
                "index": 3,
                "assigned_speaker": "S2",
                "embedding": _unit([0.0, 1.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
        ]

        baseline = find_speaker_prototype_revisions(rows, config, allow_known_reassignment=True)
        self.assertEqual([(revision.index, revision.assigned_speaker) for revision in baseline], [(1, "S1")])

        rows[0]["correction"] = {
            "status": "user_corrected",
            "action": "reassign",
            "previous_speaker": "S1",
            "corrected_speaker": "S2",
            "rejected_speakers": ["S1"],
        }

        revisions = find_speaker_prototype_revisions(rows, config, allow_known_reassignment=True)

        self.assertEqual(revisions, [])

    def test_stale_prototype_revision_cannot_apply_user_rejected_speaker(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.reassign_sentence(1, "S2")
        revision = SimpleNamespace(
            index=1,
            previous_speaker="S2",
            assigned_speaker="S1",
        )

        applied = diarizer._apply_prototype_revision(revision)

        self.assertFalse(applied)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")

    def test_known_prototype_reassignment_deletes_source_when_last_sentence_moves(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.args.speaker_refinement = True
        diarizer.args.speaker_refinement_unknown_tentative = False
        diarizer.args.allow_speaker_reassignment = True
        diarizer.args.speaker_refinement_small_island_merge = False
        diarizer._speaker_last_media_end["S1"] = 2.0
        diarizer._seed_profiles = [{"label": "S2"}]
        revision = SimpleNamespace(
            index=1,
            previous_speaker="S1",
            assigned_speaker="S2",
            prototype_score=0.9,
            prototype_margin=0.8,
            prototype_delta=0.8,
            prototype_scores={"S1": 0.1, "S2": 0.9},
            assignment_source="prototype_reassign",
        )

        diarizer._assignment_engine = mock.Mock()
        diarizer._assignment_engine.plan_refinement.return_value = SimpleNamespace(
            revisions=(revision,),
        )
        diarizer._refine_speaker_assignments()

        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        profiles = {
            profile["label"]: profile
            for profile in diarizer.memory.export_profiles()
        }
        self.assertNotIn("S1", profiles)
        self.assertEqual(profiles["S2"]["sentence_count"], 1)
        self.assertEqual(profiles["S2"]["speech_seconds"], 2.0)
        self.assertTrue(np.allclose(profiles["S2"]["centroid"], _unit([0.0, 1.0])))
        self.assertNotIn("S1", diarizer._speaker_metadata)
        self.assertNotIn("S1", diarizer._speaker_last_media_end)
        speaker_events = [
            record["payload"]
            for record in diarizer.bus.records
            if record["event"] == "speakers"
        ]
        self.assertTrue(speaker_events)
        self.assertNotIn("S1", {speaker["id"] for speaker in speaker_events[-1]["speakers"]})

    def test_known_prototype_reassignment_keeps_persistent_empty_source(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.args.speaker_refinement = True
        diarizer.args.speaker_refinement_unknown_tentative = False
        diarizer.args.allow_speaker_reassignment = True
        diarizer.args.speaker_refinement_small_island_merge = False
        diarizer._speaker_metadata["S1"].update({
            "source": "reference",
            "locked": True,
            "reference_audio": "alice.wav",
        })
        revision = SimpleNamespace(
            index=1,
            previous_speaker="S1",
            assigned_speaker="S2",
            prototype_score=0.9,
            prototype_margin=0.8,
            prototype_delta=0.8,
            prototype_scores={"S1": 0.1, "S2": 0.9},
            assignment_source="prototype_reassign",
        )

        diarizer._assignment_engine = mock.Mock()
        diarizer._assignment_engine.plan_refinement.return_value = SimpleNamespace(
            revisions=(revision,),
        )
        diarizer._refine_speaker_assignments()

        self.assertIn("S1", {profile["label"] for profile in diarizer.memory.export_profiles()})
        self.assertIn("S1", diarizer._speaker_metadata)

    def test_empty_reassignment_cleanup_preserves_live_profiles_and_rejects_stale_update(self) -> None:
        diarizer = _fake_diarizer()
        diarizer._live_embedding_separate = True
        diarizer.live_memory = SpeakerMemory(min_first_speaker_seconds=0.1)
        diarizer.live_memory.upsert_profile("S1", _unit([1.0, 0.0]), duration_seconds=2.0)
        diarizer.live_memory.upsert_profile("S2", _unit([0.0, 1.0]), duration_seconds=2.0)
        diarizer._sentence_refinement_records[1]["assigned_speaker"] = "S2"
        diarizer._sentence_refinement_records[1]["provisional_assigned_speaker"] = "S1"

        removed = diarizer._remove_empty_reassigned_speaker_profiles({"S1"})

        self.assertEqual(removed, ["S1"])
        self.assertEqual(
            {profile["label"] for profile in diarizer.live_memory.export_profiles()},
            {"S2"},
        )
        live_s2 = diarizer.live_memory.export_profiles()[0]
        self.assertEqual(live_s2["sentence_count"], 1)
        self.assertTrue(np.allclose(live_s2["centroid"], _unit([0.0, 1.0])))

        diarizer.memory.upsert_profile("S1", _unit([1.0, 0.0]), duration_seconds=0.1)
        diarizer._embed_live_audio_chunk = mock.Mock(return_value=_unit([1.0, 0.0]))
        diarizer._process_live_speaker_memory_update(SimpleNamespace(
            speaker_id="S1",
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            duration_seconds=0.1,
            suffix=".live-profile.wav",
            speaker_generation=diarizer._speaker_generation,
            run_id="",
        ))

        diarizer._embed_live_audio_chunk.assert_not_called()
        self.assertEqual(
            {profile["label"] for profile in diarizer.live_memory.export_profiles()},
            {"S2"},
        )

    def test_empty_reassignment_cleanup_keeps_preloaded_detected_speaker(self) -> None:
        diarizer = _fake_diarizer()
        diarizer._seed_profiles = [{"label": "S1"}]
        diarizer._sentence_refinement_records[1]["assigned_speaker"] = "S2"

        removed = diarizer._remove_empty_reassigned_speaker_profiles({"S1"})

        self.assertEqual(removed, [])
        self.assertIn("S1", {profile["label"] for profile in diarizer.memory.export_profiles()})

    def test_known_prototype_reassignment_cleans_up_after_complete_batch(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.args.speaker_refinement = True
        diarizer.args.speaker_refinement_unknown_tentative = False
        diarizer.args.allow_speaker_reassignment = True
        diarizer.args.speaker_refinement_small_island_merge = False
        revisions = [
            SimpleNamespace(
                index=1,
                previous_speaker="S1",
                assigned_speaker="S2",
                prototype_score=0.9,
                prototype_margin=0.8,
                prototype_delta=0.8,
                prototype_scores={"S1": 0.1, "S2": 0.9},
                assignment_source="prototype_reassign",
            ),
            SimpleNamespace(
                index=2,
                previous_speaker="S2",
                assigned_speaker="S1",
                prototype_score=0.9,
                prototype_margin=0.8,
                prototype_delta=0.8,
                prototype_scores={"S1": 0.9, "S2": 0.1},
                assignment_source="prototype_reassign",
            ),
        ]

        diarizer._assignment_engine = mock.Mock()
        diarizer._assignment_engine.plan_refinement.return_value = SimpleNamespace(
            revisions=tuple(revisions),
        )
        diarizer._refine_speaker_assignments()

        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S1")
        self.assertEqual(
            {profile["label"] for profile in diarizer.memory.export_profiles()},
            {"S1", "S2"},
        )

    def test_final_review_deletes_empty_detected_profile(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.args.speaker_refinement = False
        diarizer.memory.upsert_profile("S5", _unit([1.0, 1.0]), duration_seconds=0.1)
        diarizer._speaker_metadata["S5"] = {
            "name": "",
            "source": "detected",
            "locked": False,
            "reference_audio": "",
        }
        diarizer._speaker_last_media_end["S5"] = 4.1

        diarizer._finalize_speaker_refinement()

        self.assertNotIn("S5", {profile["label"] for profile in diarizer.memory.export_profiles()})
        self.assertNotIn("S5", diarizer._speaker_metadata)
        self.assertNotIn("S5", diarizer._speaker_last_media_end)
        speaker_events = [
            record["payload"]
            for record in diarizer.bus.records
            if record["event"] == "speakers"
        ]
        self.assertTrue(speaker_events)
        self.assertNotIn("S5", {speaker["id"] for speaker in speaker_events[-1]["speakers"]})
        status_messages = [
            str(record["payload"].get("message") or "")
            for record in diarizer.bus.records
            if record["event"] == "status"
        ]
        self.assertTrue(any("Deleted empty speaker S5 after final review." in message for message in status_messages))

    def test_mark_correct_blocks_known_prototype_reassignment(self) -> None:
        config = SpeakerRefinementConfig(
            prototype_min_duration=0.0,
            known_min_similarity=-1.0,
            known_min_delta=0.0,
        )
        rows = [
            {
                "index": 1,
                "assigned_speaker": "S2",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 1.0,
                "unknown_probability": 0.0,
            },
            {
                "index": 2,
                "assigned_speaker": "S1",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
            {
                "index": 3,
                "assigned_speaker": "S2",
                "embedding": _unit([0.0, 1.0]),
                "duration_seconds": 2.0,
                "unknown_probability": 0.0,
            },
        ]

        baseline = find_speaker_prototype_revisions(rows, config, allow_known_reassignment=True)
        self.assertEqual([(revision.index, revision.assigned_speaker) for revision in baseline], [(1, "S1")])

        rows[0]["correction"] = {
            "status": "user_confirmed",
            "action": "mark_correct",
            "corrected_speaker": "S2",
        }

        revisions = find_speaker_prototype_revisions(rows, config, allow_known_reassignment=True)

        self.assertEqual(revisions, [])

    def test_stale_prototype_revision_cannot_change_marked_correct_row(self) -> None:
        diarizer = _fake_diarizer()
        diarizer.mark_sentence_correct(1)
        revision = SimpleNamespace(
            index=1,
            previous_speaker="S1",
            assigned_speaker="S2",
        )

        applied = diarizer._apply_prototype_revision(revision)

        self.assertFalse(applied)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S1")

    def test_mark_correct_rows_are_preferred_as_prototype_evidence(self) -> None:
        config = SpeakerRefinementConfig(
            max_per_profile=1,
            prototype_min_duration=0.0,
        )
        rows = [
            {
                "index": 1,
                "assigned_speaker": "S1",
                "embedding": _unit([0.0, 1.0]),
                "duration_seconds": 100.0,
                "unknown_probability": 0.0,
            },
            {
                "index": 2,
                "assigned_speaker": "S1",
                "embedding": _unit([1.0, 0.0]),
                "duration_seconds": 0.1,
                "unknown_probability": 0.0,
                "correction": {
                    "status": "user_confirmed",
                    "action": "mark_correct",
                    "corrected_speaker": "S1",
                },
            },
        ]

        prototypes = build_speaker_prototypes(rows, config)

        self.assertTrue(np.allclose(prototypes["S1"][0], _unit([1.0, 0.0])))


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
