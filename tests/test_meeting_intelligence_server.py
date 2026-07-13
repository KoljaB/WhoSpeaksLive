from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence_server import (
    DEMO_SESSION_ID,
    MeetingIntelligenceServerConfig,
    MeetingIntelligenceService,
    config_from_args,
    build_arg_parser,
    default_llm_config,
    extract_model_ids,
    load_env_file,
    make_handler,
    parse_timecode,
    parse_whospeakslive_transcript,
    speaker_id_from_name,
)
from window.meeting_intelligence_pipeline import MockMeetingLLMClient
from window.report_templates import STANDARD_TEMPLATE_ID
from window.web_assets import read_web_text


DEMO_TEXT = """[00:00.7 - 00:02.8] Speaker 1: Thanks for coming to today's monthly meeting.
[00:04.4 - 00:06.7] Speaker 2: We decided the beta launch moves to August.
[00:08.0 - 00:10.5] Speaker 1: Alice will follow up with the API team by Tuesday.
"""


GERMAN_WORKS_COUNCIL_TEMPLATE_ID = "builtin.german-works-council"
ENGLISH_PODCAST_TEMPLATE_ID = "builtin.english-podcast-production"


def demo_service(root: Path, **overrides: object) -> MeetingIntelligenceService:
    transcript_path = root / "demo.txt"
    transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
    values: dict[str, object] = {
        "session_dir": root / "sessions",
        "cache_dir": root / "reports",
        "template_dir": root / "templates",
        "demo_transcript": transcript_path,
        "mock_llm": True,
        "max_segment_rows": 12,
    }
    values.update(overrides)
    return MeetingIntelligenceService(MeetingIntelligenceServerConfig(**values))


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

    def test_service_lists_exactly_eleven_inspectable_predefined_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory))

            templates = service.list_report_templates()
            predefined = [template for template in templates if template.get("builtin")]

            self.assertEqual(len(predefined), 11)
            self.assertEqual(len({template["template_id"] for template in predefined}), 11)
            self.assertIn(STANDARD_TEMPLATE_ID, {template["template_id"] for template in predefined})
            self.assertTrue(all(template["source_kind"] == "predefined" for template in predefined))
            self.assertTrue(all(template["read_only"] is True for template in predefined))

    def test_service_inspects_predefined_template_with_builder_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory))

            template = service.get_report_template(GERMAN_WORKS_COUNCIL_TEMPLATE_ID)

            self.assertEqual(template["template_id"], GERMAN_WORKS_COUNCIL_TEMPLATE_ID)
            self.assertEqual(template["language_mode"], "de")
            self.assertEqual(template["privacy_policy"], "local_only")
            self.assertTrue(template["builtin"])
            self.assertTrue(template["read_only"])
            self.assertTrue(template["revision_hash"])
            self.assertTrue(template["sections"])
            self.assertTrue(all("output_fields" in section for section in template["sections"]))

    def test_custom_template_clone_save_version_and_delete_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory))

            cloned = service.clone_report_template(STANDARD_TEMPLATE_ID, "My reusable report")
            self.assertFalse(cloned["builtin"])
            self.assertEqual(cloned["version"], 1)
            self.assertEqual(cloned["source_kind"], "custom")
            self.assertFalse(cloned["read_only"])

            edited = dict(cloned)
            edited.pop("source_kind", None)
            edited.pop("read_only", None)
            edited["description"] = "Updated reusable report definition."
            saved = service.save_report_template({"template": edited})

            self.assertEqual(saved["template_id"], cloned["template_id"])
            self.assertEqual(saved["version"], 2)
            self.assertNotEqual(saved["revision_hash"], cloned["revision_hash"])
            self.assertEqual(
                service.get_report_template(cloned["template_id"])["description"],
                "Updated reusable report definition.",
            )
            self.assertTrue(service.delete_report_template(cloned["template_id"]))
            with self.assertRaisesRegex(ValueError, "Unknown report template"):
                service.get_report_template(cloned["template_id"])

    def test_builder_can_save_a_new_template_without_preallocating_an_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory))
            saved = service.save_report_template({"template": {
                "schema_version": "report_template_v1",
                "name": "Safety review",
                "description": "Created through the report builder.",
                "version": 1,
                "builtin": False,
                "language_mode": "inherit",
                "privacy_policy": "inherit",
                "sections": [{
                    "key": "urgent_items",
                    "title": "Urgent items",
                    "objective": "Find urgent items grounded in transcript evidence.",
                    "max_items": 5,
                    "evidence_required": True,
                    "render_kind": "table",
                    "sort_order": "severity",
                    "output_fields": [],
                }],
            }})

            self.assertEqual(saved["template_id"], "custom.safety-review")
            self.assertFalse(saved["read_only"])

    def test_generation_with_fixed_language_preset_records_template_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory), report_language="en")

            generated = service.generate_report(
                DEMO_SESSION_ID,
                template_id=GERMAN_WORKS_COUNCIL_TEMPLATE_ID,
            )
            report = generated["report"]

            self.assertEqual(generated["report_language"], "de")
            self.assertEqual(report["report_language"], "de")
            self.assertEqual(report["template_id"], GERMAN_WORKS_COUNCIL_TEMPLATE_ID)
            self.assertEqual(report["template_revision"], generated["template"]["revision_hash"])
            self.assertEqual(report["report_template"]["template_id"], GERMAN_WORKS_COUNCIL_TEMPLATE_ID)
            self.assertEqual(report["report_template"]["language_mode"], "de")
            self.assertEqual(
                report["pipeline"]["section_passes"],
                [section["key"] for section in generated["template"]["sections"]],
            )

    def test_reports_for_multiple_templates_share_session_without_cache_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = demo_service(root)

            standard = service.generate_report(DEMO_SESSION_ID, template_id=STANDARD_TEMPLATE_ID)
            podcast = service.generate_report(DEMO_SESSION_ID, template_id=ENGLISH_PODCAST_TEMPLATE_ID)

            self.assertTrue(service.get_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)["available"])
            self.assertTrue(service.get_report(DEMO_SESSION_ID, ENGLISH_PODCAST_TEMPLATE_ID)["available"])
            self.assertNotEqual(standard["report"]["report_id"], podcast["report"]["report_id"])
            self.assertNotEqual(standard["report"]["template_id"], podcast["report"]["template_id"])
            cache_files = list((root / "reports").glob("*.json"))
            self.assertEqual(len(cache_files), 2)
            session = service.list_sessions()[0]
            self.assertEqual(
                set(session["report_template_ids"]),
                {STANDARD_TEMPLATE_ID, ENGLISH_PODCAST_TEMPLATE_ID},
            )

    def test_local_only_template_blocks_public_remote_providers(self) -> None:
        for provider, key_name in (("openai", "OPENAI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                service = demo_service(
                    Path(directory),
                    mock_llm=False,
                    llm_config=default_llm_config(provider, api_key="test-key"),
                )

                with self.assertRaisesRegex(ValueError, "local-only"):
                    service.generate_report(
                        DEMO_SESSION_ID,
                        template_id=GERMAN_WORKS_COUNCIL_TEMPLATE_ID,
                    )
                self.assertEqual(service.public_config()["api_key_env_var"], key_name)

    def test_async_jobs_include_template_id_and_dedupe_per_session_template_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entered = threading.Event()
            release = threading.Event()

            class BlockingMockMeetingLLMClient(MockMeetingLLMClient):
                def chat_json(self, **kwargs: object) -> dict[str, object]:
                    entered.set()
                    if not release.wait(timeout=5.0):
                        raise RuntimeError("test generation was not released")
                    return super().chat_json(**kwargs)

            service = demo_service(root)
            service.client_factory = BlockingMockMeetingLLMClient

            standard = service.start_generate_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)
            self.assertTrue(entered.wait(timeout=2.0))
            duplicate = service.start_generate_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)
            podcast = service.start_generate_report(DEMO_SESSION_ID, ENGLISH_PODCAST_TEMPLATE_ID)

            self.assertEqual(duplicate["job_id"], standard["job_id"])
            self.assertEqual(standard["template_id"], STANDARD_TEMPLATE_ID)
            self.assertNotEqual(podcast["job_id"], standard["job_id"])
            self.assertEqual(podcast["template_id"], ENGLISH_PODCAST_TEMPLATE_ID)

            release.set()
            deadline = time.monotonic() + 5.0
            jobs = [standard, podcast]
            while time.monotonic() < deadline:
                jobs = [service.get_generation_job(job["job_id"]) for job in jobs]
                if all(job["status"] in {"succeeded", "failed"} for job in jobs):
                    break
                time.sleep(0.02)
            self.assertEqual([job["status"] for job in jobs], ["succeeded", "succeeded"])

    def test_queued_generation_uses_captured_template_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entered = threading.Event()
            release = threading.Event()

            class BlockingClient(MockMeetingLLMClient):
                def chat_json(self, **kwargs: object) -> dict[str, object]:
                    entered.set()
                    if not release.wait(timeout=5):
                        raise RuntimeError("generation was not released")
                    return super().chat_json(**kwargs)

            service = demo_service(Path(directory))
            service.client_factory = BlockingClient
            try:
                blocker = service.start_generate_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)
                self.assertTrue(entered.wait(timeout=2))
                original = service.clone_report_template(STANDARD_TEMPLATE_ID, "Captured template")
                queued = service.start_generate_report(DEMO_SESSION_ID, original["template_id"])

                edited = dict(original)
                edited.pop("source_kind", None)
                edited.pop("read_only", None)
                edited["description"] = "Edited after the generation request was queued."
                current = service.save_report_template({"template": edited})
                release.set()

                deadline = time.monotonic() + 5
                jobs = [blocker, queued]
                while time.monotonic() < deadline:
                    jobs = [service.get_generation_job(job["job_id"]) for job in jobs]
                    if all(job["status"] in {"succeeded", "failed"} for job in jobs):
                        break
                    time.sleep(0.02)

                cached = service._read_cached_report(DEMO_SESSION_ID, original["template_id"])
                self.assertEqual([job["status"] for job in jobs], ["succeeded", "succeeded"])
                self.assertEqual(cached["template_revision"], original["revision_hash"])
                self.assertNotEqual(cached["template_revision"], current["revision_hash"])
                self.assertTrue(service.get_report(DEMO_SESSION_ID, original["template_id"])["stale"])
            finally:
                release.set()
                service.close()

    def test_service_lists_demo_session_and_caches_mock_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "demo.txt"
            transcript_path.write_text(DEMO_TEXT, encoding="utf-8")
            config = MeetingIntelligenceServerConfig(
                session_dir=root / "sessions",
                cache_dir=root / "reports",
                template_dir=root / "templates",
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
                session_dir=root / "sessions", cache_dir=root / "reports", template_dir=root / "templates",
                demo_transcript=transcript_path, mock_llm=True,
            ))
            english.generate_report(DEMO_SESSION_ID)
            spanish = MeetingIntelligenceService(MeetingIntelligenceServerConfig(
                session_dir=root / "sessions", cache_dir=root / "reports", template_dir=root / "templates",
                demo_transcript=transcript_path,
                mock_llm=True, report_language="es",
            ))

            self.assertEqual(spanish.public_config()["report_language"], "es")
            self.assertFalse(spanish.get_report(DEMO_SESSION_ID)["available"])
            generated = spanish.generate_report(DEMO_SESSION_ID)
            self.assertEqual(generated["report"]["report_language"], "es")
            self.assertIn("Borrador de resumen ejecutivo", generated["report"]["summary"])

    def test_report_language_accepts_german_and_regional_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = build_arg_parser().parse_args([
                "--report-language", "de-AT", "--template-dir", str(Path(directory) / "templates"),
            ])
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
                    template_dir=root / "templates",
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
                    template_dir=root / "templates",
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
                    template_dir=root / "templates",
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
                    template_dir=root / "templates",
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
                    template_dir=root / "templates",
                    demo_transcript=transcript_path,
                    llm_config=default_llm_config("openai", api_key="", model="gpt-5.6-luna"),
                    max_segment_rows=12,
                )
            )

            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                service.generate_report(DEMO_SESSION_ID)

    def test_helpers_and_page_contract_are_stable(self) -> None:
        page_contract = "\n".join((
            read_web_text("reports/index.html"),
            read_web_text("reports/report_builder.js"),
            read_web_text("reports/app.js"),
            read_web_text("reports/styles-base.css"),
            read_web_text("reports/styles-components.css"),
        ))
        self.assertEqual(parse_timecode("01:02.5"), 62.5)
        self.assertEqual(parse_timecode("01:02:03.5"), 3723.5)
        self.assertEqual(speaker_id_from_name("Speaker 12"), "S12")
        self.assertIn("Report template", page_contract)
        self.assertIn("Report builder", page_contract)
        self.assertIn("templateSelect", page_contract)
        self.assertIn("templateBuilder", page_contract)
        self.assertIn("data-section-prop", page_contract)
        self.assertIn("data-field-prop", page_contract)
        self.assertIn("render_kind", page_contract)
        self.assertIn("evidence_required", page_contract)
        self.assertIn("reportTabs", page_contract)
        self.assertNotIn("const tabs =", page_contract)
        self.assertIn("/api/templates", page_contract)
        self.assertIn("/api/templates/save", page_contract)
        self.assertIn("/api/templates/delete", page_contract)
        self.assertIn("/api/generate-async", page_contract)
        self.assertIn("/api/delete-report", page_contract)
        self.assertIn("deleteReportBtn", page_contract)
        self.assertIn("deleteReportLabel", page_contract)
        self.assertIn("progressPanel", page_contract)
        self.assertIn("progressOverlay", page_contract)
        self.assertIn('role="dialog"', page_contract)
        self.assertIn("closeProgressFooterBtn", page_contract)
        self.assertIn("progress-modal-open", page_contract)
        self.assertIn("data-evidence-id", page_contract)
        self.assertIn("openEvidenceInTranscript", page_contract)
        self.assertIn("transcriptRowAliases", page_contract)
        self.assertIn("data-row-aliases", page_contract)
        self.assertIn("transcript-row", page_contract)
        self.assertIn("evidence-hit", page_contract)
        self.assertIn("/api/llm-config", page_contract)
        self.assertIn("/api/llm-models", page_contract)
        self.assertIn("llmProviderSelect", page_contract)
        self.assertIn("llmModelSelect", page_contract)
        self.assertIn("llmModelInput", page_contract)
        self.assertIn("loadModelsBtn", page_contract)
        self.assertIn("applyProviderConfig", page_contract)

    def test_report_server_serves_only_packaged_module_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = demo_service(Path(directory))
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                with urllib.request.urlopen(
                    f"http://{host}:{port}/assets/web/reports/app.js", timeout=2
                ) as response:
                    body = response.read()
                    content_type = response.headers.get_content_type()
                self.assertTrue(body.startswith(b"import "))
                self.assertEqual(content_type, "text/javascript")
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(f"http://{host}:{port}/assets/web/../../pyproject.toml", timeout=2)
                self.assertEqual(rejected.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                service.close()


if __name__ == "__main__":
    unittest.main()
