from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.session_store import SessionStore
from window.saved_person_identity import SavedPersonIdentityService
from speakers.person_library import PersonLibrary


class SessionStoreTests(unittest.TestCase):
    def test_saved_speaker_person_link_is_session_specific_persistent_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            alice = people.create_person("Alice")

            linked = service.link(
                "20260707-test-session",
                "S1",
                person_id=alice["id"],
            )
            saved_speaker = linked["speaker_state"]["speakers"][0]
            self.assertEqual(saved_speaker["person_id"], alice["id"])
            self.assertEqual(saved_speaker["identity_status"], "confirmed")
            self.assertTrue(saved_speaker["future_recognition"]["available"])
            self.assertEqual(people.public_state()[0]["meeting_sample_count"], 1)

            service.link("20260707-test-session", "S1", person_id=alice["id"])
            self.assertEqual(people.public_state()[0]["voice_sample_count"], 1)
            reopened = service.decorate_session(store.open_session("20260707-test-session"))
            self.assertEqual(reopened["speaker_state"]["speakers"][0]["person_id"], alice["id"])
            self.assertFalse((root / "sessions" / "20260707-test-session" / ".person-identity-transaction.json").exists())

    def test_saved_correction_recomputes_only_that_session_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            alice = people.create_person("Alice")
            people.add_meeting_sample(
                alice["id"], [0.8, 0.6], embedding_provider="test-provider",
                session_id="independent-meeting",
            )
            service.link("20260707-test-session", "S1", person_id=alice["id"])

            store.reassign_rows("20260707-test-session", [1], "S2")
            service.recompute_linked_samples("20260707-test-session")
            raw = people.get(alice["id"])
            assert raw is not None
            sessions = {
                sample["source"]["session_id"]
                for sample in raw["voice_samples"]
                if sample["kind"] == "meeting_template"
            }
            self.assertEqual(sessions, {"independent-meeting"})
            self.assertEqual(
                store.open_session("20260707-test-session")["speaker_state"]["speakers"][0]["person_id"],
                alice["id"],
            )

    def test_saved_enrollment_reports_stable_reason_when_embeddings_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            snapshot = self.sample_snapshot()
            snapshot["embedding_records"] = []
            store.save_snapshot(snapshot, status_label="Saved")
            availability = service.availability("20260707-test-session", "S1")
            self.assertFalse(availability["available"])
            self.assertEqual(availability["reason"], "missing_embeddings")
            self.assertIn("no compatible stored voice evidence", availability["explanation"])

    def test_saved_link_retry_repairs_interrupted_second_write_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            alice = people.create_person("Alice")
            original_write = store._write_json
            failed = False

            def fail_speaker_write_once(path, payload):
                nonlocal failed
                if Path(path).name == "speakers.json" and not failed:
                    failed = True
                    raise OSError("simulated second-write failure")
                return original_write(path, payload)

            with mock.patch.object(store, "_write_json", side_effect=fail_speaker_write_once):
                with self.assertRaisesRegex(OSError, "second-write"):
                    service.link("20260707-test-session", "S1", person_id=alice["id"])
            intent = root / "sessions" / "20260707-test-session" / ".person-identity-transaction.json"
            self.assertTrue(intent.is_file())
            self.assertEqual(people.public_state()[0]["voice_sample_count"], 1)

            service.link("20260707-test-session", "S1", person_id=alice["id"])
            self.assertFalse(intent.exists())
            self.assertEqual(people.public_state()[0]["voice_sample_count"], 1)
            self.assertEqual(store.open_session("20260707-test-session")["speaker_state"]["speakers"][0]["person_id"], alice["id"])

    def test_saved_link_retry_with_new_person_does_not_leave_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            original_write = store._write_json
            failed = False

            def fail_speaker_write_once(path, payload):
                nonlocal failed
                if Path(path).name == "speakers.json" and not failed:
                    failed = True
                    raise OSError("simulated second-write failure")
                return original_write(path, payload)

            with mock.patch.object(store, "_write_json", side_effect=fail_speaker_write_once):
                with self.assertRaisesRegex(OSError, "second-write"):
                    service.link("20260707-test-session", "S1", person_name="Alice")

            recovered = service.link("20260707-test-session", "S1", person_name="Alice")
            self.assertEqual([person["name"] for person in recovered["speaker_state"]["people"]], ["Alice"])
            self.assertEqual(len(people.public_state()), 1)
            self.assertEqual(people.public_state()[0]["voice_sample_count"], 1)

    def test_deleted_saved_meeting_sample_is_not_recreated_by_correction_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            alice = people.create_person("Alice")
            people.set_recognition_enabled(alice["id"], False)
            service.link("20260707-test-session", "S1", person_id=alice["id"])
            self.assertFalse(people.get(alice["id"])["recognition_enabled"])
            sample_id = people.public_state()[0]["voice_samples"][0]["id"]

            people.delete_sample(alice["id"], sample_id)
            service.recompute_linked_samples("20260707-test-session")

            self.assertEqual(people.public_state()[0]["voice_sample_count"], 0)
            self.assertFalse(people.get(alice["id"])["recognition_enabled"])

    def test_forget_cleanup_unlinks_saved_history_and_session_cleanup_removes_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            people = PersonLibrary(root / "people" / "people.json")
            service = SavedPersonIdentityService(store, people)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")
            alice = people.create_person("Alice")
            service.link("20260707-test-session", "S1", person_id=alice["id"])

            self.assertEqual(service.unlink_person_everywhere(alice["id"]), 1)
            saved = store.open_session("20260707-test-session")
            self.assertNotIn("person_id", saved["speaker_state"]["speakers"][0])
            self.assertEqual(service.remove_session_samples("20260707-test-session"), 1)
            self.assertEqual(people.public_state()[0]["voice_sample_count"], 0)

    def sample_snapshot(self) -> dict[str, object]:
        return {
            "id": "20260707-test-session",
            "created_at": "2026-07-07T20:31:00",
            "duration_seconds": 4.2,
            "source": {
                "url": "https://example.test/watch?v=demo",
                "video_id": "demo",
                "started_at": "2026-07-07T20:31:00",
                "audio_path": str(ROOT / "runtime" / "media" / "demo.wav"),
                "video_path": str(ROOT / "runtime" / "media" / "demo.mp4"),
                "capture_mode": "youtube",
            },
            "transcript_rows": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello there.",
                    "assigned_speaker": "S1",
                    "speaker_name": "Alice",
                    "probabilities": {"unknown": 0.0, "speaker1": 1.0},
                },
                {
                    "index": 2,
                    "start": 2.1,
                    "end": 4.2,
                    "text": "Good morning.",
                    "assigned_speaker": "S2",
                    "speaker_name": "Bob",
                    "probabilities": {"unknown": 0.0, "speaker2": 1.0},
                },
            ],
            "speaker_state": {
                "group_name": "",
                "groups": [],
                "embedding_provider": "test-provider",
                "speakers": [
                    {"id": "S1", "name": "Alice", "display_name": "Alice", "sentence_count": 1, "speech_seconds": 2.0},
                    {"id": "S2", "name": "Bob", "display_name": "Bob", "sentence_count": 1, "speech_seconds": 2.1},
                ],
            },
            "speaker_profiles": [
                {
                    "label": "S1",
                    "name": "Alice",
                    "centroid": [1.0, 0.0],
                    "centroid_encoding": "float32-base64-le",
                    "centroid_b64": "",
                    "centroid_length": 2,
                }
            ],
            "embedding_records": [
                {"index": 1, "duration_seconds": 2.0, "assigned_speaker": "S1", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
                {"index": 2, "duration_seconds": 2.1, "assigned_speaker": "S2", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
            ],
            "embedding_provider": "test-provider",
        }

    def test_save_list_open_and_archive_restore_delete_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))

            summary = store.save_snapshot(self.sample_snapshot(), status_label="Autosaved")

            self.assertEqual(summary["id"], "20260707-test-session")
            self.assertEqual(summary["title"], "YouTube demo - Jul 7 20:31")
            self.assertEqual(summary["started_at"], "2026-07-07T20:31:00")
            self.assertTrue(summary["ended_at"])
            self.assertEqual(summary["transcript_rows"], 2)
            self.assertTrue(summary["has_transcript"])
            self.assertTrue(summary["has_embeddings"])

            opened = store.open_session("20260707-test-session")
            self.assertEqual(len(opened["transcript_rows"]), 2)
            self.assertEqual(opened["speaker_state"]["speakers"][0]["display_name"], "Alice")
            self.assertEqual(opened["embedding_count"], 2)
            public_json = json.dumps(opened)
            self.assertNotIn(str(ROOT), public_json)
            self.assertNotIn('"centroid"', public_json)
            self.assertNotIn('"centroid_b64"', public_json)

            self.assertEqual(len(store.list_sessions("active")), 1)
            archived = store.archive_session("20260707-test-session")
            self.assertTrue(archived["archived"])
            self.assertEqual(store.list_sessions("active"), [])
            self.assertEqual(len(store.list_sessions("archived")), 1)
            restored = store.restore_session("20260707-test-session")
            self.assertFalse(restored["archived"])

            renamed = store.rename_session("20260707-test-session", "Product Review")
            self.assertEqual(renamed["title"], "Product Review")
            self.assertEqual(store.list_sessions("active", "Product")[0]["title"], "Product Review")

            deleted = store.delete_session("20260707-test-session")
            self.assertTrue(deleted["deleted"])
            self.assertEqual(store.list_sessions("all"), [])

    def test_saved_speaker_rename_updates_speaker_state_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")

            opened = store.rename_speaker("20260707-test-session", "S1", "Anna")

            speakers = opened["speaker_state"]["speakers"]
            self.assertEqual(speakers[0]["name"], "Anna")
            self.assertEqual(speakers[0]["display_name"], "Anna")
            self.assertEqual(opened["transcript_rows"][0]["speaker_name"], "Anna")
            manifest = json.loads((Path(directory) / "20260707-test-session" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("Anna", manifest["speaker_names"])

    def test_saved_transcript_rows_can_be_reassigned_and_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)
            store.save_snapshot(self.sample_snapshot(), status_label="Saved")

            reassigned = store.reassign_rows("20260707-test-session", [1], "S2")

            first = reassigned["transcript_rows"][0]
            self.assertEqual(first["assigned_speaker"], "S2")
            self.assertEqual(first["speaker_name"], "Bob")
            self.assertEqual(first["assignment_source"], "user_correction")
            self.assertEqual(first["correction"]["status"], "user_corrected")
            self.assertFalse(first["correction"]["updates_memory"])
            speakers = {speaker["id"]: speaker for speaker in reassigned["speaker_state"]["speakers"]}
            self.assertEqual(speakers["S1"]["sentence_count"], 0)
            self.assertEqual(speakers["S2"]["sentence_count"], 2)
            embeddings = json.loads((root / "20260707-test-session" / "embeddings.json").read_text(encoding="utf-8"))
            self.assertEqual(embeddings["records"][0]["assigned_speaker"], "S2")

            confirmed = store.mark_rows_correct("20260707-test-session", [2])

            self.assertEqual(confirmed["transcript_rows"][1]["correction"]["status"], "user_confirmed")
            with self.assertRaisesRegex(ValueError, "Unknown transcript row"):
                store.reassign_rows("20260707-test-session", [999], "S1")

    def test_create_empty_session_then_fill_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))

            summary = store.create_session(
                session_id="draft-session",
                status_label="New",
                source={
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "capture_mode": "youtube",
                    "started_at": "2026-07-07T20:31:00",
                },
            )

            self.assertEqual(summary["id"], "draft-session")
            self.assertEqual(summary["title"], "YouTube abc123 - Jul 7 20:31")
            self.assertEqual(summary["status_label"], "New")
            self.assertFalse(summary["has_transcript"])
            self.assertEqual(len(store.list_sessions("active")), 1)

            snapshot = self.sample_snapshot()
            snapshot["id"] = "draft-session"
            filled = store.save_snapshot(snapshot, status_label="Saved")

            self.assertEqual(filled["id"], "draft-session")
            self.assertEqual(filled["transcript_rows"], 2)
            self.assertTrue(filled["has_transcript"])

    def test_stream_audio_writer_creates_managed_audio_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            snapshot = self.sample_snapshot()
            snapshot["source"] = {
                "url": "microphone://local",
                "video_id": "microphone",
                "streaming_audio": True,
                "audio_sample_rate": 16000,
                "capture_mode": "microphone",
                "title": "Microphone recording",
            }

            def write_audio(path: Path) -> bool:
                path.write_bytes(b"RIFF-test")
                return True

            summary = store.save_snapshot(snapshot, status_label="Saved", write_audio=True, audio_writer=write_audio)

            opened = store.open_session("20260707-test-session")
            self.assertTrue(summary["has_audio"])
            self.assertEqual(opened["manifest"]["audio"]["kind"], "managed_wav")
            self.assertTrue((Path(directory) / "20260707-test-session" / "audio.wav").is_file())


if __name__ == "__main__":
    unittest.main()
