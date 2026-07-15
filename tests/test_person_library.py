from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from speakers.person_library import PersonLibrary


class PersonLibraryTests(unittest.TestCase):
    def test_v1_migration_is_lossless_and_backed_up_before_first_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "people.json"
            path.write_text(json.dumps({
                "format": "whospeaks-people",
                "version": 1,
                "updated_at": "2026-01-01T00:00:00Z",
                "people": [{
                    "id": "alice-id",
                    "name": "Alice",
                    "recognition_enabled": False,
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "profile_version": 7,
                    "templates": [{
                        "id": "template-id",
                        "centroid": [1.0, 0.0],
                        "embedding_provider": "mock-v1",
                        "session_id": "meeting-1",
                        "source_title": "Old meeting",
                        "sentence_count": 4,
                        "speech_seconds": 12.5,
                        "quality": 0.8,
                        "cohesion": 0.91,
                        "outlier_count": 2,
                        "confirmation": "automatic_final",
                        "confirmed_at": "2026-01-01T00:00:00Z",
                        "anchor": True,
                    }],
                }],
            }), encoding="utf-8")

            library = PersonLibrary(path)
            person = library.get("alice-id")
            assert person is not None
            self.assertEqual(person["profile_version"], 7)
            self.assertEqual(person["voice_samples"][0]["kind"], "meeting_template")
            self.assertEqual(person["voice_samples"][0]["representations"][0]["centroid"], [1.0, 0.0])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)

            library.set_recognition_policy("alice-id", {"meeting_samples": False})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 2)
            backups = list(Path(tmp).glob("people.v1.*.bak.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8"))["version"], 1)

    def test_v1_atomic_migration_failure_keeps_original_and_backup_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "people.json"
            legacy = {"format": "whospeaks-people", "version": 1, "people": []}
            path.write_text(json.dumps(legacy), encoding="utf-8")
            library = PersonLibrary(path)
            with mock.patch("speakers.person_library.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(ValueError, "atomically write"):
                    library.create_person("Alice")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)
            backups = list(Path(tmp).glob("people.v1.*.bak.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8"))["version"], 1)

    def test_duplicate_names_are_distinct_and_addressed_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            first = library.create_person("Alex")
            second = library.create_person("Alex")
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual([person["name"] for person in library.public_state()], ["Alex", "Alex"])
            library.rename_person(second["id"], "Alexandra")
            self.assertEqual(library.get(first["id"])["name"], "Alex")
            self.assertEqual(library.get(second["id"])["name"], "Alexandra")

    def test_expected_and_recognition_choices_persist_and_new_people_default_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "people.json"
            library = PersonLibrary(path)
            alice = library.create_person("Alice")
            bob = library.create_person("Bob")

            self.assertEqual(library.expected_person_ids(), set())
            self.assertFalse(library.get(alice["id"])["expected"])
            self.assertFalse(library.get(bob["id"])["expected"])

            library.set_expected_people([alice["id"]])
            library.set_recognition_enabled(alice["id"], False)

            reloaded = PersonLibrary(path)
            self.assertEqual(reloaded.expected_person_ids(), {alice["id"]})
            people = {person["id"]: person for person in reloaded.public_state()}
            self.assertTrue(people[alice["id"]]["expected"])
            self.assertFalse(people[alice["id"]]["recognition_enabled"])
            self.assertFalse(people[bob["id"]]["expected"])

            charlie = reloaded.create_person("Charlie")
            self.assertFalse(charlie["expected"])
            self.assertEqual(reloaded.expected_person_ids(), {alice["id"]})

    def test_multiple_manual_samples_can_be_disabled_and_deleted_with_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = PersonLibrary(root / "people.json")
            alice = library.create_person("Alice")
            first = library.add_manual_sample(
                alice["id"], [1.0, 0.0], embedding_provider="mock",
                raw_audio=b"first-audio", filename="headset.wav", label="Headset",
            )
            second = library.add_manual_sample(
                alice["id"], [0.8, 0.6], embedding_provider="mock",
                raw_audio=b"second-audio", filename="phone.wav", label="Telephone",
            )
            raw_path = root / "voice-samples" / alice["id"] / f"{first['id']}.wav"
            self.assertTrue(raw_path.is_file())
            library.set_sample_state(alice["id"], first["id"], False)
            public = library.public_state()[0]
            self.assertEqual(public["manual_sample_count"], 2)
            self.assertEqual(public["voice_samples"][0]["state"], "disabled")
            serialized = json.dumps(public)
            self.assertNotIn("centroid", serialized)
            self.assertNotIn(str(root), serialized)
            library.delete_sample(alice["id"], first["id"])
            self.assertFalse(raw_path.exists())
            self.assertEqual(library.public_state()[0]["voice_sample_count"], 1)
            self.assertEqual(library.public_state()[0]["voice_samples"][0]["id"], second["id"])

    def test_source_policies_provider_dimension_and_expected_people_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_person("Alice")
            bob = library.create_person("Bob")
            library.add_manual_sample(
                alice["id"], [1.0, 0.0], embedding_provider="mock",
                raw_audio=b"alice-manual", filename="alice.wav",
            )
            library.add_meeting_sample(
                alice["id"], [0.0, 1.0], embedding_provider="mock",
                session_id="meeting-a",
            )
            library.add_meeting_sample(
                bob["id"], [1.0, 0.0], embedding_provider="other-provider",
                session_id="meeting-b",
            )
            self.assertEqual(library.match(
                [1.0, 0.0], embedding_provider="mock", min_similarity=0.8, min_margin=0.0,
            ).person_id, alice["id"])
            library.set_recognition_policy(alice["id"], {"manual_samples": False, "meeting_samples": True})
            self.assertIsNone(library.match(
                [1.0, 0.0], embedding_provider="mock", min_similarity=0.8, min_margin=0.0,
            ))
            self.assertEqual(library.match(
                [0.0, 1.0], embedding_provider="mock", min_similarity=0.8, min_margin=0.0,
            ).person_id, alice["id"])
            library.set_recognition_policy(alice["id"], {"meeting_samples": False})
            self.assertIsNone(library.match(
                [0.0, 1.0], embedding_provider="mock", min_similarity=0.1, min_margin=0.0,
            ))
            library.set_recognition_policy(alice["id"], {"manual_samples": True})
            self.assertIsNone(library.match(
                [1.0, 0.0], embedding_provider="mock", min_similarity=0.8, min_margin=0.0,
                expected_person_ids={bob["id"]},
            ))
            self.assertIsNone(library.match(
                [1.0, 0.0, 0.0], embedding_provider="mock", min_similarity=0.1, min_margin=0.0,
            ))

    def test_duplicate_gallery_does_not_inflate_confidence_and_corroboration_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_person("Alice")
            base = [0.8, 0.6]
            library.add_meeting_sample(alice["id"], base, embedding_provider="mock", session_id="m1")
            initial = library.match(base, embedding_provider="mock", min_similarity=-1.0, min_margin=0.0)
            assert initial is not None
            for index in range(2, 7):
                library.add_meeting_sample(alice["id"], base, embedding_provider="mock", session_id=f"m{index}")
            duplicate = library.match(base, embedding_provider="mock", min_similarity=-1.0, min_margin=0.0)
            assert duplicate is not None
            self.assertEqual(duplicate.similarity, initial.similarity)
            self.assertEqual(duplicate.template_count, 1)

            library.add_meeting_sample(alice["id"], [0.6, 0.8], embedding_provider="mock", session_id="diverse")
            supported = library.match(base, embedding_provider="mock", min_similarity=-1.0, min_margin=0.0)
            assert supported is not None
            self.assertLessEqual(supported.similarity - initial.similarity, 0.0251)
            self.assertEqual(supported.template_count, 2)

    def test_competitor_margin_and_anchor_loss_keep_unknown_and_quarantine_dependents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_person("Alice")
            bob = library.create_person("Bob")
            anchor = library.add_manual_sample(
                alice["id"], [1.0, 0.0], embedding_provider="mock",
                raw_audio=b"anchor", filename="anchor.wav",
            )
            library.add_meeting_sample(
                alice["id"], [0.98, 0.02], embedding_provider="mock", session_id="derived",
                confirmation="automatic_checkpoint", anchor_sample_ids=[anchor["id"]],
            )
            library.add_meeting_sample(bob["id"], [0.99, 0.01], embedding_provider="mock", session_id="bob")
            self.assertIsNone(library.match(
                [1.0, 0.0], embedding_provider="mock", min_similarity=0.8, min_margin=0.04,
            ))
            library.set_sample_state(alice["id"], anchor["id"], False)
            alice_raw = library.get(alice["id"])
            assert alice_raw is not None
            derived = next(sample for sample in alice_raw["voice_samples"] if sample["kind"] == "meeting_template")
            self.assertEqual(derived["state"], "quarantined")

    def test_confirmed_templates_are_session_scoped_persistent_and_matchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "people.json"
            library = PersonLibrary(path)
            alice = library.create_or_get("Alice")

            library.add_confirmed_template(
                alice["id"],
                [1.0, 0.0],
                embedding_provider="mock",
                session_id="meeting-1",
                sentence_count=3,
                speech_seconds=8.0,
            )
    # Reconfirming the same person in the same meeting replaces the staged
    # template instead of inflating the gallery.
            library.add_confirmed_template(
                alice["id"],
                [0.99, 0.01],
                embedding_provider="mock",
                session_id="meeting-1",
                sentence_count=5,
                speech_seconds=14.0,
                cohesion=0.94,
                outlier_count=2,
            )
            library.add_confirmed_template(
                alice["id"],
                [0.8, 0.6],
                embedding_provider="mock",
                session_id="meeting-2-new-mic",
                sentence_count=4,
                speech_seconds=10.0,
            )

            reloaded = PersonLibrary(path)
            public = reloaded.public_state()
            self.assertEqual(public[0]["name"], "Alice")
            self.assertEqual(public[0]["template_count"], 2)
            self.assertNotIn("centroid", public[0])
            raw = reloaded.get(alice["id"])
            assert raw is not None
            first_template = next(item for item in raw["templates"] if item["session_id"] == "meeting-1")
            self.assertEqual(first_template["cohesion"], 0.94)
            self.assertEqual(first_template["outlier_count"], 2)

            match = reloaded.match(
                np.array([0.79, 0.61], dtype=np.float32),
                embedding_provider="mock",
                min_similarity=0.8,
                min_margin=0.04,
            )
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(match.person_id, alice["id"])


    def test_roster_and_forget_voice_control_recognition_without_deleting_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_or_get("Alice")
            library.add_confirmed_template(
                alice["id"],
                [1.0, 0.0],
                embedding_provider="mock",
                session_id="meeting-1",
            )

            library.set_recognition_enabled(alice["id"], False)
            self.assertIsNone(library.match(
                [1.0, 0.0],
                embedding_provider="mock",
                min_similarity=0.5,
                min_margin=0.0,
            ))
            self.assertIsNotNone(library.match(
                [1.0, 0.0],
                embedding_provider="mock",
                min_similarity=0.5,
                min_margin=0.0,
                include_disabled=True,
            ))

            library.set_recognition_enabled(alice["id"], True)
            self.assertIsNotNone(library.match(
                [1.0, 0.0],
                embedding_provider="mock",
                min_similarity=0.5,
                min_margin=0.0,
            ))

            library.forget_voice(alice["id"])
            person = library.public_state()[0]
            self.assertEqual(person["name"], "Alice")
            self.assertFalse(person["recognition_ready"])
            self.assertEqual(person["template_count"], 0)


    def test_removing_an_anchor_promotes_the_oldest_remaining_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_or_get("Alice")
            for session_id, centroid in (("meeting-1", [1.0, 0.0]), ("meeting-2", [0.9, 0.1])):
                library.add_confirmed_template(
                    alice["id"],
                    centroid,
                    embedding_provider="mock",
                    session_id=session_id,
                )

            self.assertTrue(library.remove_session_template(
                alice["id"],
                embedding_provider="mock",
                session_id="meeting-1",
            ))
            person = library.get(alice["id"])
            assert person is not None
            self.assertEqual(len(person["templates"]), 1)
            self.assertTrue(person["templates"][0]["anchor"])


if __name__ == "__main__":
    unittest.main()
