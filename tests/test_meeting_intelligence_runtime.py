from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence_pipeline import default_llm_config, stable_hash
from window.meeting_intelligence_runtime import (
    AutoGenerationMonitor,
    GenerationJobManager,
    GenerationQueueFullError,
    ReportCache,
    ReportGenerationRequest,
)
from window.meeting_intelligence_runtime.models import immutable_json
from window.report_templates import STANDARD_TEMPLATE_ID


def request(session_id: str) -> ReportGenerationRequest:
    return ReportGenerationRequest(
        session_id=session_id,
        template_id=STANDARD_TEMPLATE_ID,
        title=session_id,
        transcript_revision_id="revision",
        report_language="en",
        transcript_rows_json=immutable_json([{"index": 1, "text": session_id}]),
        speaker_state_json=immutable_json({"speakers": []}),
        summary_json=immutable_json({"id": session_id}),
        template_json=immutable_json({"template_id": STANDARD_TEMPLATE_ID}),
        llm_config=default_llm_config(),
        mock_llm=True,
        max_segment_rows=12,
    )


class MeetingIntelligenceRuntimeTests(unittest.TestCase):
    def test_generation_manager_is_bounded_and_deduplicates_active_key(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def runner(item: ReportGenerationRequest, progress: object) -> dict[str, object]:
            entered.set()
            release.wait(timeout=5)
            return {"report": {"sections": {}, "evidence_index": []}}

        manager = GenerationJobManager(runner, max_queue_size=1)
        try:
            first = manager.submit(request("first"))
            self.assertTrue(entered.wait(timeout=2))
            duplicate = manager.submit(request("first"))
            second = manager.submit(request("second"))
            with self.assertRaises(GenerationQueueFullError):
                manager.submit(request("third"))
            self.assertEqual(first["job_id"], duplicate["job_id"])
            self.assertNotEqual(first["job_id"], second["job_id"])
        finally:
            release.set()
            manager.close()

    def test_report_cache_atomic_write_and_legacy_read_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ReportCache(Path(directory), hash_fn=stable_hash)
            report = {"schema_version": "meeting_intelligence_report_v2", "summary": "captured"}
            cache.write("session", STANDARD_TEMPLATE_ID, report)
            loaded = cache.read("session", STANDARD_TEMPLATE_ID, legacy_template_id=STANDARD_TEMPLATE_ID)

            self.assertEqual(loaded, report)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            self.assertTrue(cache.delete("session", STANDARD_TEMPLATE_ID, legacy_template_id=STANDARD_TEMPLATE_ID))

    def test_auto_generation_monitor_has_idempotent_joined_shutdown(self) -> None:
        called = threading.Event()
        monitor = AutoGenerationMonitor(called.set, interval_seconds=1)
        monitor.start()
        self.assertTrue(called.wait(timeout=2))
        started = time.monotonic()
        monitor.close()
        monitor.close()
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
