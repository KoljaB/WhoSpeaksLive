from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence_server import (
    DEMO_SESSION_ID,
    PAGE_HTML,
    MeetingIntelligenceServerConfig,
    MeetingIntelligenceService,
    config_from_args,
    build_arg_parser,
    default_llm_config,
    extract_model_ids,
    load_env_file,
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

    def test_spanish_report_language_is_public_and_cache_aware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            english = MeetingIntelligenceService(MeetingIntelligenceServerConfig(
                session_dir=root / "sessions", cache_dir=root / "reports", demo_transcript=transcript_path, mock_llm=True,
            ))
            english.generate_report(DEMO_SESSION_ID)
            spanish = MeetingIntelligenceService(MeetingIntelligenceServerConfig(
                session_dir=root / "sessions", cache_dir=root / "reports", demo_transcript=transcript_path,
                mock_llm=True, report_language="es",
            ))

            self.assertEqual(spanish.public_config()["report_language"], "es")
            self.assertFalse(spanish.get_report(DEMO_SESSION_ID)["available"])
            generated = spanish.generate_report(DEMO_SESSION_ID)
            self.assertEqual(generated["report"]["report_language"], "es")
            self.assertIn("Borrador de resumen ejecutivo", generated["report"]["summary"])

    def test_report_language_accepts_german_and_regional_aliases(self) -> None:
        args = build_arg_parser().parse_args(["--report-language", "de-AT"])
        config = config_from_args(args)
        service = MeetingIntelligenceService(config)

        self.assertEqual(service.public_config()["report_language"], "de")

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

    def test_auto_generation_queues_saved_session_without_browser_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            session_dir = root / "sessions"
            service = MeetingIntelligenceService(
                MeetingIntelligenceServerConfig(
                    session_dir=session_dir,
                    cache_dir=root / "reports",
                    demo_transcript=transcript_path,
                    mock_llm=True,
                    auto_generate=True,
                    max_segment_rows=12,
                )
            )
            self.assertEqual(service.auto_generate_ready_sessions(), [])
            saved = service.store.create_session(session_id="weekly-executive", status_label="Saved")
            service.store.save_snapshot({
                "id": saved["id"],
                "transcript_rows": parse_whospeakslive_transcript(transcript_path),
                "speaker_state": {"speakers": []},
            }, status_label="Saved")

            queued = service.auto_generate_ready_sessions()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not service.get_report(saved["id"])["available"]:
                time.sleep(0.02)

            self.assertEqual(len(queued), 1)
            self.assertTrue(service.get_report(saved["id"])["available"])

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

    def test_load_env_file_sets_missing_values_without_overwriting_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY='from-file'\nWHOSPEAKS_MI_LLM_MODEL=from-env-file\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "already-set"}, clear=True):
                loaded = load_env_file(env_path)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already-set")
                self.assertEqual(os.environ["WHOSPEAKS_MI_LLM_MODEL"], "from-env-file")

    def test_service_can_switch_runtime_provider_and_reports_key_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = MeetingIntelligenceService(
                MeetingIntelligenceServerConfig(
                    session_dir=root / "sessions",
                    cache_dir=root / "reports",
                    llm_config=default_llm_config("llama_cpp", model="local-before"),
                )
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                updated = service.update_llm_config({
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                    "base_url": "https://api.openai.com/v1",
                })

            self.assertEqual(updated["provider"], "openai")
            self.assertEqual(updated["model"], "gpt-5.6-terra")
            self.assertEqual(updated["expected_report_provider"], "openai:gpt-5.6-terra")
            self.assertEqual(updated["api_key_env_var"], "OPENAI_API_KEY")
            self.assertFalse(updated["api_key_configured"])
            self.assertTrue(any(provider["id"] == "openai" for provider in updated["providers"]))

    def test_model_list_filter_keeps_text_models_and_prefers_cheaper_names(self) -> None:
        models = extract_model_ids({
            "data": [
                {"id": "gpt-5.6-luna"},
                {"id": "text-embedding-3-small"},
                {"id": "gpt-4.1-mini"},
                {"id": "gpt-4.1-nano"},
                {"id": "gpt-4o-audio-preview"},
                {"id": "dall-e-3"},
            ]
        })

        self.assertEqual(models[:2], ["gpt-4.1-nano", "gpt-4.1-mini"])
        self.assertIn("gpt-5.6-luna", models)
        self.assertNotIn("text-embedding-3-small", models)

    def test_openai_generation_fails_fast_without_server_side_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            service = MeetingIntelligenceService(
                MeetingIntelligenceServerConfig(
                    session_dir=root / "sessions",
                    cache_dir=root / "reports",
                    demo_transcript=transcript_path,
                    llm_config=default_llm_config("openai", api_key="", model="gpt-5.6-luna"),
                    max_segment_rows=12,
                )
            )

            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                service.generate_report(DEMO_SESSION_ID)

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
        self.assertIn("/api/llm-config", PAGE_HTML)
        self.assertIn("/api/llm-models", PAGE_HTML)
        self.assertIn("llmProviderSelect", PAGE_HTML)
        self.assertIn("llmModelSelect", PAGE_HTML)
        self.assertIn("llmModelInput", PAGE_HTML)
        self.assertIn("loadModelsBtn", PAGE_HTML)
        self.assertIn("applyProviderConfig", PAGE_HTML)


if __name__ == "__main__":
    unittest.main()
