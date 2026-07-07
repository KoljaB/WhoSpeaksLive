from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
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
