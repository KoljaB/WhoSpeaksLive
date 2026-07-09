from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence import (
    generate_meeting_report,
    mark_report_stale_if_needed,
    update_report_object,
)
from window.session_store import SessionStore


def sample_rows() -> list[dict[str, object]]:
    return [
        {
            "index": 1,
            "start": 0.0,
            "end": 2.0,
            "text": "We decided the beta launch moves to August.",
            "assigned_speaker": "S1",
            "speaker_name": "Alice",
        },
        {
            "index": 2,
            "start": 2.1,
            "end": 4.2,
            "text": "I will follow up with the API team by Tuesday.",
            "assigned_speaker": "S2",
            "speaker_name": "Bob",
        },
        {
            "index": 3,
            "start": 4.3,
            "end": 5.0,
            "text": "What is still blocking the partner onboarding?",
            "assigned_speaker": "S1",
            "speaker_name": "Alice",
        },
    ]


def sample_speaker_state() -> dict[str, object]:
    return {
        "speakers": [
            {"id": "S1", "name": "Alice", "display_name": "Alice"},
            {"id": "S2", "name": "Bob", "display_name": "Bob"},
        ]
    }


def sample_snapshot() -> dict[str, object]:
    return {
        "id": "meeting-intelligence-test",
        "created_at": "2026-07-09T12:00:00",
        "duration_seconds": 5.0,
        "source": {"title": "Meeting Intelligence Test", "started_at": "2026-07-09T12:00:00"},
        "transcript_rows": sample_rows(),
        "speaker_state": sample_speaker_state(),
        "speaker_profiles": [],
        "embedding_records": [],
    }


class MeetingIntelligenceCoreTests(unittest.TestCase):
    def test_report_generation_creates_evidence_grounded_objects(self) -> None:
        report = generate_meeting_report(
            session_id="core-test",
            transcript_rows=sample_rows(),
            speaker_state=sample_speaker_state(),
            generated_at="2026-07-09T10:00:00+00:00",
        )

        self.assertEqual(report["status"], "draft")
        self.assertIn("summary", {obj["type"] for obj in report["objects"]})
        self.assertIn("decision", {obj["type"] for obj in report["objects"]})
        self.assertIn("action_item", {obj["type"] for obj in report["objects"]})
        transcript_indexes = {row["index"] for row in sample_rows()}
        for obj in report["objects"]:
            self.assertEqual(obj["status"], "draft")
            self.assertTrue(obj["evidence_spans"])
            for span in obj["evidence_spans"]:
                self.assertTrue(span["row_ids"])
                for ref in span["rows"]:
                    self.assertIn(ref["index"], transcript_indexes)

    def test_report_marks_evidence_object_stale_after_speaker_change(self) -> None:
        report = generate_meeting_report(
            session_id="core-test",
            transcript_rows=sample_rows(),
            speaker_state=sample_speaker_state(),
            generated_at="2026-07-09T10:00:00+00:00",
        )
        changed_rows = [dict(row) for row in sample_rows()]
        changed_rows[1]["speaker_name"] = "Robert"

        stale, changed = mark_report_stale_if_needed(
            report,
            transcript_rows=changed_rows,
            speaker_state={
                "speakers": [
                    {"id": "S1", "name": "Alice", "display_name": "Alice"},
                    {"id": "S2", "name": "Robert", "display_name": "Robert"},
                ]
            },
            updated_at="2026-07-09T10:05:00+00:00",
        )

        self.assertTrue(changed)
        self.assertEqual(stale["status"], "stale")
        stale_objects = [obj for obj in stale["objects"] if obj["status"] == "stale"]
        self.assertTrue(stale_objects)
        self.assertGreaterEqual(stale["quality"]["stale_objects_count"], 1)

    def test_object_status_update_keeps_report_reviewable(self) -> None:
        report = generate_meeting_report(
            session_id="core-test",
            transcript_rows=sample_rows(),
            speaker_state=sample_speaker_state(),
            generated_at="2026-07-09T10:00:00+00:00",
        )
        decision = next(obj for obj in report["objects"] if obj["type"] == "decision")

        updated = update_report_object(report, object_id=decision["id"], status="accepted")

        updated_decision = next(obj for obj in updated["objects"] if obj["id"] == decision["id"])
        self.assertEqual(updated_decision["status"], "accepted")
        self.assertEqual(updated["status"], "partially_reviewed")


class MeetingIntelligenceSessionStoreTests(unittest.TestCase):
    def test_generated_report_persists_and_status_updates_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot(sample_snapshot(), status_label="Saved")

            generated = store.generate_meeting_intelligence("meeting-intelligence-test")
            report = generated["report"]
            action = next(obj for obj in report["objects"] if obj["type"] == "action_item")

            updated = store.update_meeting_intelligence_object(
                "meeting-intelligence-test",
                action["id"],
                status="rejected",
            )
            reopened = store.open_session("meeting-intelligence-test")

            self.assertTrue(reopened["meeting_intelligence"]["available"])
            reopened_report = reopened["meeting_intelligence"]["report"]
            reopened_action = next(obj for obj in reopened_report["objects"] if obj["id"] == action["id"])
            self.assertEqual(updated["report"]["status"], "partially_reviewed")
            self.assertEqual(reopened_action["status"], "rejected")
            report_file = Path(directory) / "meeting-intelligence-test" / "meeting_intelligence.json"
            self.assertTrue(report_file.is_file())

    def test_old_session_without_meeting_intelligence_block_opens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot(sample_snapshot(), status_label="Saved")
            report_file = Path(directory) / "meeting-intelligence-test" / "meeting_intelligence.json"
            report_file.unlink()

            opened = store.open_session("meeting-intelligence-test")

            self.assertIn("meeting_intelligence", opened)
            self.assertFalse(opened["meeting_intelligence"]["available"])

    def test_saved_speaker_rename_marks_existing_report_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot(sample_snapshot(), status_label="Saved")
            store.generate_meeting_intelligence("meeting-intelligence-test")

            opened = store.rename_speaker("meeting-intelligence-test", "S2", "Robert")

            report = opened["meeting_intelligence"]["report"]
            self.assertEqual(report["status"], "stale")
            self.assertGreaterEqual(report["quality"]["stale_objects_count"], 1)

    def test_meeting_intelligence_json_is_human_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save_snapshot(sample_snapshot(), status_label="Saved")
            store.generate_meeting_intelligence("meeting-intelligence-test")
            report_file = Path(directory) / "meeting-intelligence-test" / "meeting_intelligence.json"

            parsed = json.loads(report_file.read_text(encoding="utf-8"))

            self.assertEqual(parsed["version"], 1)
            self.assertEqual(parsed["report"]["schema_version"], "meeting_intelligence_v1")


if __name__ == "__main__":
    unittest.main()
