from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence_server import (
    DEMO_SESSION_ID,
    PAGE_HTML,
    MeetingIntelligenceServerConfig,
    MeetingIntelligenceService,
    parse_timecode,
    parse_whospeakslive_transcript,
    speaker_id_from_name,
)


DEMO_TEXT = """[00:00.7 - 00:02.8] Speaker 1: Thanks for coming to today's monthly meeting.
[00:04.4 - 00:06.7] Speaker 2: We decided the beta launch moves to August.
[00:08.0 - 00:10.5] Speaker 1: Alice will follow up with the API team by Tuesday.
"""


class MeetingIntelligenceServerTests(unittest.TestCase):
    def test_demo_transcript_parser_preserves_times_speakers_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript_path = Path(directory) / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")

            rows = parse_whospeakslive_transcript(transcript_path)

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["row_id"], "demo_row_0001")
            self.assertEqual(rows[0]["start"], 0.7)
            self.assertEqual(rows[1]["assigned_speaker"], "S2")
            self.assertEqual(rows[2]["speaker_name"], "Speaker 1")

    def test_service_lists_demo_session_and_caches_mock_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            config = MeetingIntelligenceServerConfig(
                session_dir=root / "sessions",
                cache_dir=root / "reports",
                demo_transcript=transcript_path,
                mock_llm=True,
                max_segment_rows=12,
            )
            service = MeetingIntelligenceService(config)

            sessions = service.list_sessions()
            generated = service.generate_report(DEMO_SESSION_ID)
            cached = service.get_report(DEMO_SESSION_ID)

            self.assertEqual(sessions[0]["id"], DEMO_SESSION_ID)
            self.assertTrue(generated["available"])
            self.assertEqual(generated["report"]["provider"], "mock_meeting_llm")
            self.assertTrue(cached["available"])
            self.assertTrue(any((root / "reports").iterdir()))

    def test_async_generation_job_exposes_progress_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            service = MeetingIntelligenceService(
                MeetingIntelligenceServerConfig(
                    session_dir=root / "sessions",
                    cache_dir=root / "reports",
                    demo_transcript=transcript_path,
                    mock_llm=True,
                    max_segment_rows=12,
                )
            )

            started = service.start_generate_report(DEMO_SESSION_ID)
            deadline = time.monotonic() + 5.0
            job = started
            while time.monotonic() < deadline:
                job = service.get_generation_job(started["job_id"])
                if job["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.02)

            self.assertEqual(job["status"], "succeeded")
            self.assertEqual(job["percent"], 100)
            self.assertTrue(job["events"])
            self.assertTrue(service.get_report(DEMO_SESSION_ID)["available"])

    def test_delete_report_removes_cached_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            service = MeetingIntelligenceService(
                MeetingIntelligenceServerConfig(
                    session_dir=root / "sessions",
                    cache_dir=root / "reports",
                    demo_transcript=transcript_path,
                    mock_llm=True,
                    max_segment_rows=12,
                )
            )

            generated = service.generate_report(DEMO_SESSION_ID)
            deleted = service.delete_report(DEMO_SESSION_ID)
            after_delete = service.get_report(DEMO_SESSION_ID)

            self.assertTrue(generated["available"])
            self.assertTrue(deleted["deleted"])
            self.assertFalse(after_delete["available"])
            self.assertFalse(after_delete["stale"])
            self.assertEqual(len(after_delete["transcript_rows"]), 3)

    def test_helpers_and_page_contract_are_stable(self) -> None:
        self.assertEqual(parse_timecode("01:02.5"), 62.5)
        self.assertEqual(parse_timecode("01:02:03.5"), 3723.5)
        self.assertEqual(speaker_id_from_name("Speaker 12"), "S12")
        self.assertIn("Summary", PAGE_HTML)
        self.assertIn("Action items", PAGE_HTML)
        self.assertIn("/api/generate-async", PAGE_HTML)
        self.assertIn("/api/delete-report", PAGE_HTML)
        self.assertIn("deleteReportBtn", PAGE_HTML)
        self.assertIn("deleteReportLabel", PAGE_HTML)
        self.assertIn("progressPanel", PAGE_HTML)
        self.assertIn("data-evidence-id", PAGE_HTML)
        self.assertIn("openEvidenceInTranscript", PAGE_HTML)
        self.assertIn("transcriptRowAliases", PAGE_HTML)
        self.assertIn("data-row-aliases", PAGE_HTML)
        self.assertIn("transcript-row", PAGE_HTML)
        self.assertIn("evidence-hit", PAGE_HTML)


if __name__ == "__main__":
    unittest.main()
