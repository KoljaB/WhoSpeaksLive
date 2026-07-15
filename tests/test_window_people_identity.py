from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.window_diarizer_support import make_window_diarizer


def _add_meeting_speaker(controller, centroid, *, seconds: float = 8.0, count: int = 3) -> str:
    return controller.memory.add_profile(
        centroid,
        duration_seconds=seconds,
        sentence_count=count,
    )


def _add_learning_record(
    controller,
    speaker_id: str,
    index: int,
    embedding,
    *,
    duration: float = 6.0,
) -> None:
    start = float(index * 10)
    with controller._sentence_refinement_lock:
        controller._sentence_refinement_records[index] = {
            "index": index,
            "base_payload": {
                "start": start,
                "end": start + duration,
                "speech_audio_ratio": 1.0,
            },
            "embedding": np.asarray(embedding, dtype=np.float32),
            "duration_seconds": duration,
            "assigned_speaker": speaker_id,
            "quality": 1.0,
            "unknown_probability": 0.05,
            "assignment_source": "embedding",
        }


class WindowPeopleIdentityTests(unittest.TestCase):
    def test_manual_sample_is_immutable_and_learning_policy_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_window_diarizer(speaker_library_dir=Path(tmp))
            controller._session_id = "meeting-1"
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            controller.rename_speaker(label, "Alice")
            person_id = controller.remember_speaker_as_person(label)["people"][0]["id"]
            manual = controller.person_library.add_manual_sample(
                person_id, [0.9, 0.1], embedding_provider="mock",
                raw_audio=b"manual-immutable", filename="manual.wav",
            )
            manual_before = controller.person_library.get(person_id)["voice_samples"]
            manual_before = next(item for item in manual_before if item["id"] == manual["id"])

            for index in range(1, 5):
                _add_learning_record(controller, label, index, [1.0, 0.01 * index])
            controller._maybe_checkpoint_confirmed_people(review_assignments=True)
            after = controller.person_library.get(person_id)
            manual_after = next(item for item in after["voice_samples"] if item["id"] == manual["id"])
            self.assertEqual(manual_after, manual_before)
            self.assertEqual(len([item for item in after["voice_samples"] if item["kind"] == "meeting_template"]), 1)

            controller.person_library.set_recognition_policy(
                person_id,
                {"learn_from_confirmed_meetings": False, "meeting_samples": False},
            )
            profile_version = controller.person_library.get(person_id)["profile_version"]
            _add_learning_record(controller, label, 9, [1.0, 0.0], duration=30.0)
            controller._maybe_checkpoint_confirmed_people(review_assignments=True)
            self.assertEqual(controller.person_library.get(person_id)["profile_version"], profile_version)
            self.assertTrue(controller.person_library.get(person_id)["recognition_policy"]["manual_samples"])

    def test_expected_people_are_session_scoped_and_unknown_remains_possible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = make_window_diarizer(speaker_library_dir=root)
            alice = seed.person_library.create_person("Alice")
            bob = seed.person_library.create_person("Bob")
            seed.person_library.add_meeting_sample(alice["id"], [1.0, 0.0], embedding_provider="mock", session_id="a")
            seed.person_library.add_meeting_sample(bob["id"], [0.0, 1.0], embedding_provider="mock", session_id="b")

            controller = make_window_diarizer(speaker_library_dir=root)
            controller.set_expected_people([bob["id"]])
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            state = controller.speaker_state()
            self.assertTrue(state["expected_people_filter_active"])
            self.assertEqual(state["expected_person_ids"], [bob["id"]])
            self.assertEqual(state["speakers"][0]["identity_status"], "unidentified")
            self.assertTrue(controller.person_library.get(alice["id"])["recognition_enabled"])
            controller._reset_runtime_session_state(emit=False)
            self.assertFalse(controller.speaker_state()["expected_people_filter_active"])

    def test_returning_person_is_suggested_then_confirmation_learns_new_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_window_diarizer(speaker_library_dir=root)
            first._session_id = "meeting-1"
            first._session_source_title = "First meeting"
            label = _add_meeting_speaker(first, [1.0, 0.0])
            first._ensure_speaker_metadata(label)
            first.rename_speaker(label, "Alice")

            remembered = first.remember_speaker_as_person(label)
            alice = remembered["people"][0]
            self.assertEqual(remembered["speakers"][0]["identity_status"], "confirmed")
            self.assertEqual(alice["template_count"], 1)

            second = make_window_diarizer(speaker_library_dir=root)
            second._session_id = "meeting-2"
            second._session_source_title = "Different microphone"
            returning_label = _add_meeting_speaker(second, [0.98, 0.02])
            second._ensure_speaker_metadata(returning_label)

            suggested = second.speaker_state()
            speaker = suggested["speakers"][0]
            self.assertEqual(speaker["identity_status"], "suggested")
            self.assertEqual(speaker["display_name"], "Likely Alice")
            self.assertEqual(speaker["name"], "")

            confirmed = second.confirm_speaker_person(returning_label, alice["id"])
            self.assertEqual(confirmed["speakers"][0]["display_name"], "Alice")
            self.assertEqual(confirmed["speakers"][0]["identity_status"], "confirmed")
            self.assertEqual(confirmed["people"][0]["template_count"], 2)

            profile_version = confirmed["people"][0]["profile_version"]
            for index, embedding in enumerate((
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, -0.02],
                [0.97, 0.03],
                [0.45, 0.89],
            ), 1):
                _add_learning_record(second, returning_label, index, embedding)
            second._maybe_checkpoint_confirmed_people(review_assignments=True)

            learned = second.person_library.get(alice["id"])
            assert learned is not None
            meeting_template = next(
                item for item in learned["templates"]
                if item["session_id"] == "meeting-2"
            )
            self.assertEqual(meeting_template["confirmation"], "automatic_checkpoint")
            self.assertEqual(meeting_template["sentence_count"], 4)
            self.assertEqual(meeting_template["outlier_count"], 1)
            self.assertGreater(meeting_template["cohesion"], 0.95)
            self.assertEqual(len(learned["templates"]), 2)
            self.assertGreater(learned["profile_version"], profile_version)


    def test_finalization_commits_small_amount_of_valid_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_window_diarizer(speaker_library_dir=Path(tmp))
            controller._session_id = "meeting-1"
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            controller.rename_speaker(label, "Alice")
            remembered = controller.remember_speaker_as_person(label)
            person_id = remembered["people"][0]["id"]
            original_version = remembered["people"][0]["profile_version"]

            for index, embedding in enumerate(([1.0, 0.0], [0.99, 0.01], [0.98, -0.02]), 1):
                _add_learning_record(controller, label, index, embedding, duration=3.0)
            controller._maybe_checkpoint_confirmed_people(review_assignments=True)
            self.assertEqual(controller.person_library.get(person_id)["profile_version"], original_version)

            controller.consolidate_confirmed_people()
            person = controller.person_library.get(person_id)
            assert person is not None
            self.assertEqual(person["profile_version"], original_version + 1)
            self.assertEqual(person["templates"][0]["confirmation"], "automatic_final")


    def test_final_sentence_flow_checkpoints_without_a_user_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_window_diarizer(speaker_library_dir=Path(tmp))
            controller._session_id = "meeting-1"
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            controller.rename_speaker(label, "Alice")
            person_id = controller.remember_speaker_as_person(label)["people"][0]["id"]

            for index in range(1, 5):
                start = float(index * 10)
                controller._apply_sentence_embedding_decision(
                    index=index,
                    base_payload={
                        "index": index,
                        "text": f"Learning sentence {index}.",
                        "start": start,
                        "end": start + 6.0,
                        "speech_audio_ratio": 1.0,
                    },
                    text=f"Learning sentence {index}.",
                    embedding=np.asarray([1.0, 0.01 * index], dtype=np.float32),
                    duration_seconds=6.0,
                    emit_status=False,
                    run_speaker_refinement=False,
                )

            person = controller.person_library.get(person_id)
            assert person is not None
            self.assertEqual(person["templates"][0]["confirmation"], "automatic_checkpoint")
            self.assertEqual(person["templates"][0]["sentence_count"], 4)


    def test_reassignment_that_invalidates_checkpoint_removes_session_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_window_diarizer(speaker_library_dir=Path(tmp))
            controller._session_id = "meeting-1"
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            controller.rename_speaker(label, "Alice")
            person_id = controller.remember_speaker_as_person(label)["people"][0]["id"]

            for index in range(1, 5):
                _add_learning_record(controller, label, index, [1.0, 0.01 * index])
            controller._maybe_checkpoint_confirmed_people(review_assignments=True)
            self.assertEqual(controller.person_library.public_state()[0]["template_count"], 1)

            with controller._sentence_refinement_lock:
                for record in controller._sentence_refinement_records.values():
                    record["assigned_speaker"] = None
                    record["correction"] = {
                        "status": "user_corrected",
                        "corrected_speaker": None,
                    }
            controller._maybe_checkpoint_confirmed_people(review_assignments=True)
            person = controller.person_library.get(person_id)
            assert person is not None
            self.assertEqual(person["templates"], [])


    def test_rejecting_confirmed_identity_rolls_back_current_meeting_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_window_diarizer(speaker_library_dir=Path(tmp))
            controller._session_id = "meeting-1"
            label = _add_meeting_speaker(controller, [1.0, 0.0])
            controller._ensure_speaker_metadata(label)
            controller.rename_speaker(label, "Alice")
            remembered = controller.remember_speaker_as_person(label)
            person_id = remembered["people"][0]["id"]

            rejected = controller.reject_speaker_person(label, person_id)
            self.assertEqual(rejected["speakers"][0]["identity_status"], "unidentified")
            self.assertEqual(rejected["speakers"][0]["display_name"], "Speaker 1")
            self.assertEqual(rejected["people"][0]["template_count"], 0)


    def test_disabled_person_is_not_suggested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_window_diarizer(speaker_library_dir=root)
            first._session_id = "meeting-1"
            label = _add_meeting_speaker(first, [1.0, 0.0])
            first._ensure_speaker_metadata(label)
            first.rename_speaker(label, "Alice")
            state = first.remember_speaker_as_person(label)
            person_id = state["people"][0]["id"]
            first.set_person_recognition(person_id, False)

            second = make_window_diarizer(speaker_library_dir=root)
            other_label = _add_meeting_speaker(second, [1.0, 0.0])
            second._ensure_speaker_metadata(other_label)
            self.assertEqual(second.speaker_state()["speakers"][0]["identity_status"], "unidentified")


if __name__ == "__main__":
    unittest.main()
