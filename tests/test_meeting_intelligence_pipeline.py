from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.meeting_intelligence_pipeline import (
    PIPELINE_SCHEMA_VERSION,
    MockMeetingLLMClient,
    MultiPassMeetingIntelligencePipeline,
    OpenAICompatibleMeetingClient,
    default_llm_config,
    evidence_index_schema,
    clean_text,
    openai_strict_schema,
    parse_openai_chat_json,
    normalize_report_language,
    section_schema,
    sanitize_report_output,
)


def sample_rows(count: int = 30) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    topics = {
        1: "Agenda review and apologies.",
        13: "Moving on to finance, we decided to keep the budget unchanged.",
        25: "Next item is training, Alice will follow up by Tuesday.",
    }
    for index in range(1, count + 1):
        text = topics.get(index, f"Discussion detail {index}.")
        rows.append({
            "index": index,
            "row_id": f"row_{index}",
            "start": float(index),
            "end": float(index) + 0.5,
            "text": text,
            "assigned_speaker": "S1" if index % 2 else "S2",
            "speaker_name": "Alice" if index % 2 else "Bob",
        })
    return rows


def custom_report_template(*, evidence_required: bool = True, max_items: int = 2) -> dict[str, object]:
    return {
        "schema_version": "report_template_v1",
        "template_id": "custom.safety-review",
        "name": "Safety review",
        "description": "A focused reusable report.",
        "version": 3,
        "builtin": False,
        "language_mode": "inherit",
        "privacy_policy": "inherit",
        "sections": [
            {
                "key": "urgent_safety",
                "title": "Urgent safety issues",
                "objective": "Find safety hazards that require immediate intervention.",
                "max_items": max_items,
                "evidence_required": evidence_required,
                "render_kind": "table",
                "sort_order": "severity",
                "output_fields": [
                    {
                        "key": "severity",
                        "label": "Severity",
                        "type": "enum",
                        "description": "Operational severity.",
                        "options": ["Critical", "Routine"],
                    },
                    {
                        "key": "equipment",
                        "label": "Equipment",
                        "type": "text",
                        "description": "Affected equipment.",
                        "options": [],
                    },
                ],
            }
        ],
    }


class CustomTemplateClient:
    name = "custom_template_test_client"

    def __init__(self, *, include_section_evidence: bool = True) -> None:
        self.include_section_evidence = include_section_evidence
        self.calls: list[str] = []
        self.payloads: list[dict[str, object]] = []

    def chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        system_prompt: str,
        user_payload: dict[str, object],
        max_tokens: int,
    ) -> dict[str, object]:
        self.calls.append(schema_name)
        self.payloads.append(user_payload)
        if schema_name == "meeting_evidence_index":
            rows = user_payload["transcript_rows"]
            assert isinstance(rows, list)
            return {
                "schema_version": "meeting_evidence_index_v1",
                "items": [
                    {
                        "id": "EV-SAFETY-001",
                        "title": "Guard missing",
                        "summary": "A machine guard was reported missing.",
                        "row_ids": [str(rows[0]["row_id"])],
                        "section_keys": ["urgent_safety", "invented_section", "urgent_safety"],
                        "support_type": "direct",
                        "confidence": "High",
                    }
                ],
            }
        evidence_ids = ["EV-SAFETY-001"] if self.include_section_evidence else ["EV-NOT-REAL"]
        item = {
            "id": "SAFE-001",
            "title": "Stop the press",
            "body": "The missing guard requires intervention.",
            "status": "urgent",
            "owner": "",
            "due": "",
            "confidence": "High",
            "evidence_ids": evidence_ids,
            "attributes": [
                {"key": "severity", "value": "Critical"},
                {"key": "equipment", "value": "Press 4"},
                {"key": "unconfigured", "value": "discard me"},
            ],
            "grounding_status": "grounded",
            "metadata": {},
        }
        return {
            "schema_version": "meeting_section_v1",
            "section": "urgent_safety",
            "summary": "One urgent safety issue.",
            "items": [item, {**item, "id": "SAFE-002", "title": "Second item"}],
        }


class CoverageRepairClient:
    name = "coverage_repair_test_client"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.payloads: list[dict[str, object]] = []

    def chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        system_prompt: str,
        user_payload: dict[str, object],
        max_tokens: int,
    ) -> dict[str, object]:
        self.calls.append(schema_name)
        self.payloads.append(user_payload)
        if schema_name == "meeting_evidence_index":
            rows = user_payload["transcript_rows"]
            assert isinstance(rows, list)
            if not user_payload.get("coverage_repair"):
                return {
                    "schema_version": "meeting_evidence_index_v1",
                    "items": [{
                        "id": "EV-TOPIC",
                        "title": "Fabric discussion",
                        "summary": "Fabrics were discussed.",
                        "row_ids": [str(rows[0]["row_id"])],
                        "section_keys": ["topics"],
                        "support_type": "direct",
                        "confidence": "High",
                    }],
                }
            requested_keys = [
                str(item["key"])
                for item in user_payload["report_sections"]
                if isinstance(item, dict)
            ]
            return {
                "schema_version": "meeting_evidence_index_v1",
                "items": [{
                    "id": "EV-DECISION-ACTION",
                    "title": "Fabric selected",
                    "summary": "Cotton was selected and ordering was assigned.",
                    "row_ids": [str(rows[-1]["row_id"])],
                    "section_keys": requested_keys,
                    "support_type": "direct",
                    "confidence": "High",
                }],
            }
        evidence = user_payload["evidence_index"]
        assert isinstance(evidence, list) and evidence
        evidence_id = str(evidence[0]["id"])
        section = str(user_payload["section"])
        return {
            "schema_version": "meeting_section_v1",
            "section": section,
            "summary": "Grounded result.",
            "items": [{
                "id": f"ITEM-{section}",
                "title": section.title(),
                "body": "A grounded report item.",
                "status": "confirmed",
                "owner": "",
                "due": "",
                "confidence": "High",
                "evidence_ids": [evidence_id],
                "attributes": [],
                "grounding_status": "grounded",
                "metadata": {},
            }],
        }


class CitationRepairClient(CustomTemplateClient):
    def chat_json(self, **kwargs: object) -> dict[str, object]:
        result = super().chat_json(**kwargs)
        user_payload = kwargs["user_payload"]
        if (
            kwargs["schema_name"] == "meeting_section"
            and isinstance(user_payload, dict)
            and user_payload.get("citation_repair")
        ):
            result["items"][0]["evidence_ids"] = ["EV-SAFETY-001"]
        return result


class MeetingIntelligencePipelineTests(unittest.TestCase):
    def test_report_language_uses_the_shared_language_configuration(self) -> None:
        self.assertEqual(normalize_report_language("es"), ("es", "Spanish"))
        self.assertEqual(normalize_report_language("Spanish"), ("es", "Spanish"))
        self.assertEqual(normalize_report_language("de"), ("de", "German"))
        self.assertEqual(normalize_report_language("de-AT"), ("de", "German"))
        with self.assertRaises(ValueError):
            normalize_report_language("not-a-language")

    def test_default_config_supports_openai_compatible_providers(self) -> None:
        self.assertEqual(default_llm_config("llama-cpp").base_url, "http://127.0.0.1:8081/v1")
        self.assertEqual(default_llm_config("ollama").base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(default_llm_config("lm_studio").base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(default_llm_config("openai").base_url, "https://api.openai.com/v1")
        self.assertEqual(default_llm_config("openai").model, "gpt-5.6-luna")
        self.assertEqual(default_llm_config("openrouter").base_url, "https://openrouter.ai/api/v1")
        with self.assertRaises(ValueError):
            default_llm_config("unsupported")

    def test_openai_compatible_payload_uses_structured_json_contract(self) -> None:
        config = default_llm_config(
            "llama_cpp",
            base_url="http://llm.test/v1",
            model="gemma-12b-q6",
            enable_thinking=False,
        )
        client = OpenAICompatibleMeetingClient(config)

        payload = client._build_payload(
            schema_name="meeting_section",
            schema={"type": "object"},
            system_prompt="Return JSON only.",
            user_payload={"section": "summary"},
            max_tokens=512,
        )

        self.assertEqual(payload["model"], "gemma-12b-q6")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("json_schema", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_section_schema_is_openai_strict_compatible(self) -> None:
        schema = section_schema()
        item_schema = schema["properties"]["items"]["items"]
        metadata_schema = item_schema["properties"]["metadata"]

        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(item_schema["additionalProperties"])
        self.assertFalse(metadata_schema["additionalProperties"])
        self.assertEqual(metadata_schema["properties"], {})
        self.assertIn("metadata", item_schema["required"])
        attributes_schema = item_schema["properties"]["attributes"]["items"]
        self.assertFalse(attributes_schema["additionalProperties"])
        self.assertEqual(attributes_schema["required"], ["key", "value"])
        self.assertIn("grounding_status", item_schema["required"])

    def test_openai_payload_omits_llama_cpp_specific_fields(self) -> None:
        config = default_llm_config(
            "openai",
            model="gpt-4.1-nano",
            api_key="test-key",
        )
        client = OpenAICompatibleMeetingClient(config)

        payload = client._build_payload(
            schema_name="meeting_section",
            schema=section_schema(),
            system_prompt="Return JSON only.",
            user_payload={"section": "summary"},
            max_tokens=512,
        )

        self.assertEqual(payload["model"], "gpt-4.1-nano")
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("json_schema", payload)
        self.assertEqual(payload["response_format"]["json_schema"]["strict"], True)

    def test_openai_payload_normalizes_future_schemas_to_strict_shape(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {"source": {"type": "string"}},
                },
            },
            "required": ["title"],
        }
        config = default_llm_config("openai", model="gpt-4.1-nano", api_key="test-key")
        client = OpenAICompatibleMeetingClient(config)

        payload = client._build_payload(
            schema_name="future_schema",
            schema=schema,
            system_prompt="Return JSON only.",
            user_payload={},
            max_tokens=128,
        )

        strict_schema = payload["response_format"]["json_schema"]["schema"]
        self.assertFalse(strict_schema["additionalProperties"])
        self.assertEqual(strict_schema["required"], ["title", "metadata"])
        self.assertFalse(strict_schema["properties"]["metadata"]["additionalProperties"])
        self.assertEqual(strict_schema["properties"]["metadata"]["required"], ["source"])
        self.assertNotIn("json_schema", payload)

    def test_openai_strict_schema_does_not_mutate_original_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"metadata": {"type": "object", "additionalProperties": True}},
        }

        strict_schema = openai_strict_schema(schema)

        self.assertTrue(schema["properties"]["metadata"]["additionalProperties"])
        self.assertFalse(strict_schema["properties"]["metadata"]["additionalProperties"])

    def test_multi_pass_pipeline_chunks_transcript_then_generates_sections(self) -> None:
        client = MockMeetingLLMClient()
        progress_events: list[dict[str, object]] = []
        pipeline = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=12,
            section_types=("executive_summary", "decisions", "action_items"),
            progress_callback=progress_events.append,
        )

        report = pipeline.generate(
            session_id="pipeline-test",
            transcript_rows=sample_rows(30),
            speaker_state={"speakers": [{"id": "S1", "name": "Alice"}, {"id": "S2", "name": "Bob"}]},
            title="Pipeline test",
        )

        self.assertEqual(report["schema_version"], PIPELINE_SCHEMA_VERSION)
        self.assertGreaterEqual(report["pipeline"]["segments"], 2)
        self.assertEqual(client.calls.count("meeting_evidence_index"), report["pipeline"]["segments"])
        self.assertEqual(client.calls.count("meeting_section"), 3)
        self.assertIn("decisions", report["sections"])
        self.assertTrue(report["evidence_index"])
        self.assertFalse(report["quality"]["local_first"])
        self.assertEqual(progress_events[-1]["stage"], "completed")
        self.assertEqual(progress_events[-1]["percent"], 100)
        self.assertIn("evidence", {event["stage"] for event in progress_events})
        self.assertIn("section", {event["stage"] for event in progress_events})

    def test_omitted_template_loads_the_inspectable_standard_preset(self) -> None:
        client = MockMeetingLLMClient()
        report = MultiPassMeetingIntelligencePipeline(client).generate(
            session_id="standard-template",
            transcript_rows=sample_rows(2),
        )

        self.assertEqual(report["template_id"], "builtin.standard-meeting")
        self.assertTrue(report["report_template"]["builtin"])
        self.assertEqual(
            report["pipeline"]["section_passes"],
            [section["key"] for section in report["report_template"]["sections"]],
        )
        evidence_payload = next(
            payload
            for call, payload in zip(client.calls, client.payloads)
            if call == "meeting_evidence_index"
        )
        self.assertEqual(evidence_payload["max_items"], 24)
        self.assertEqual(evidence_index_schema(max_items=24)["properties"]["items"]["maxItems"], 24)

    def test_multi_pass_pipeline_requests_and_records_spanish_report_content(self) -> None:
        client = MockMeetingLLMClient()
        pipeline = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=12,
            section_types=("executive_summary",),
            report_language="es",
        )

        report = pipeline.generate(session_id="spanish-test", transcript_rows=sample_rows(12))

        self.assertEqual(report["report_language"], "es")
        self.assertIn("Borrador de resumen ejecutivo", report["summary"])
        self.assertTrue(client.payloads)
        self.assertTrue(all(payload["report_language"] == "es" for payload in client.payloads))

    def test_multi_pass_pipeline_propagates_german_to_every_llm_pass(self) -> None:
        client = MockMeetingLLMClient()
        report = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=12,
            section_types=("executive_summary", "decisions"),
            report_language="de",
        ).generate(session_id="german-test", transcript_rows=sample_rows(12))

        self.assertEqual(report["report_language"], "de")
        self.assertTrue(all(payload["report_language"] == "de" for payload in client.payloads))

    def test_custom_template_drives_every_pass_and_preserves_configured_attributes(self) -> None:
        client = CustomTemplateClient()
        template = custom_report_template(max_items=1)
        pipeline = MultiPassMeetingIntelligencePipeline(client, report_template=template)

        report = pipeline.generate(session_id="custom-template", transcript_rows=sample_rows(4))

        evidence_payload = next(
            payload
            for call, payload in zip(client.calls, client.payloads)
            if call == "meeting_evidence_index"
        )
        section_payload = next(
            payload
            for call, payload in zip(client.calls, client.payloads)
            if call == "meeting_section"
        )
        objective = "Find safety hazards that require immediate intervention."
        self.assertEqual(evidence_payload["report_sections"][0]["objective"], objective)
        self.assertEqual(section_payload["section_definition"]["objective"], objective)
        self.assertEqual(section_payload["evidence_index"][0]["id"], "EV-SAFETY-001")
        self.assertEqual(section_payload["global_context"]["template_id"], "custom.safety-review")
        self.assertEqual(report["pipeline"]["section_passes"], ["urgent_safety"])
        self.assertEqual(report["template_id"], "custom.safety-review")
        self.assertEqual(report["template_revision"], report["report_template"]["revision_hash"])
        self.assertEqual(report["report_template"]["version"], 3)
        self.assertEqual(report["evidence_index"][0]["section_keys"], ["urgent_safety"])
        section = report["sections"]["urgent_safety"]
        self.assertEqual(section["definition"], report["report_template"]["sections"][0])
        self.assertEqual(len(section["items"]), 1)
        self.assertEqual(
            section["items"][0]["attributes"],
            [
                {"key": "severity", "value": "Critical"},
                {"key": "equipment", "value": "Press 4"},
            ],
        )
        self.assertEqual(section["items"][0]["grounding_status"], "grounded")

    def test_required_evidence_item_without_a_valid_anchor_is_omitted(self) -> None:
        client = CustomTemplateClient(include_section_evidence=False)
        report = MultiPassMeetingIntelligencePipeline(
            client,
            report_template=custom_report_template(evidence_required=True),
        ).generate(session_id="missing-evidence", transcript_rows=sample_rows(2))

        self.assertEqual(report["sections"]["urgent_safety"]["items"], [])
        self.assertEqual(report["quality"]["evidence_gaps"], [])

    def test_missing_section_evidence_is_searched_again_before_sections_are_generated(self) -> None:
        template = {
            "schema_version": "report_template_v1",
            "template_id": "custom.coverage-repair",
            "name": "Coverage repair",
            "description": "Test template.",
            "version": 1,
            "builtin": False,
            "language_mode": "inherit",
            "privacy_policy": "inherit",
            "sections": [
                {"key": key, "title": key.title(), "objective": objective, "max_items": 3,
                 "evidence_required": True, "render_kind": "cards", "sort_order": "chronological",
                 "output_fields": []}
                for key, objective in (
                    ("topics", "Find discussed topics."),
                    ("decisions", "Find decisions."),
                    ("actions", "Find actions and commitments."),
                )
            ],
        }
        client = CoverageRepairClient()

        report = MultiPassMeetingIntelligencePipeline(
            client,
            report_template=template,
        ).generate(session_id="coverage-repair", transcript_rows=sample_rows(3))

        evidence_payloads = [
            payload
            for call, payload in zip(client.calls, client.payloads)
            if call == "meeting_evidence_index"
        ]
        self.assertEqual(len(evidence_payloads), 2)
        self.assertFalse(evidence_payloads[0]["coverage_repair"])
        self.assertTrue(evidence_payloads[1]["coverage_repair"])
        self.assertEqual(
            [item["key"] for item in evidence_payloads[1]["report_sections"]],
            ["decisions", "actions"],
        )
        self.assertEqual(report["pipeline"]["evidence_coverage_repair"], ["decisions", "actions"])
        self.assertEqual(len(report["sections"]["decisions"]["items"]), 1)
        self.assertEqual(len(report["sections"]["actions"]["items"]), 1)

    def test_section_with_an_invalid_citation_is_retried_with_allowed_evidence_ids(self) -> None:
        client = CitationRepairClient(include_section_evidence=False)

        report = MultiPassMeetingIntelligencePipeline(
            client,
            report_template=custom_report_template(evidence_required=True),
        ).generate(session_id="citation-repair", transcript_rows=sample_rows(2))

        section_payloads = [
            payload
            for call, payload in zip(client.calls, client.payloads)
            if call == "meeting_section"
        ]
        self.assertEqual(len(section_payloads), 2)
        self.assertTrue(section_payloads[1]["citation_repair"])
        self.assertEqual(
            report["sections"]["urgent_safety"]["items"][0]["evidence_ids"],
            ["EV-SAFETY-001"],
        )

    def test_optional_evidence_item_can_remain_without_an_anchor(self) -> None:
        client = CustomTemplateClient(include_section_evidence=False)
        report = MultiPassMeetingIntelligencePipeline(
            client,
            report_template=custom_report_template(evidence_required=False),
        ).generate(session_id="optional-evidence", transcript_rows=sample_rows(2))

        item = report["sections"]["urgent_safety"]["items"][0]
        self.assertEqual(item["evidence_ids"], [])
        self.assertEqual(item["grounding_status"], "not_required")

    def test_mangled_model_unicode_is_repaired(self) -> None:
        self.assertEqual(clean_text("Beschl\x00fcsse und sp\x00e4ter"), "Beschlüsse und später")

    def test_cached_report_sanitizer_repairs_text_and_removes_uncited_items(self) -> None:
        report = {
            "evidence_index": [{"id": "EV-1"}],
            "sections": {
                "decisions": {
                    "definition": {"evidence_required": True},
                    "items": [
                        {"title": "Beschl\x00fcsse", "evidence_ids": [], "grounding_status": "missing_required_evidence"},
                        {"title": "Best\x00e4tigt", "evidence_ids": ["EV-1"], "grounding_status": "missing_required_evidence"},
                    ],
                }
            },
            "quality": {"evidence_gaps": ["decisions[0]"]},
        }

        sanitized = sanitize_report_output(report)

        self.assertEqual([item["title"] for item in sanitized["sections"]["decisions"]["items"]], ["Bestätigt"])
        self.assertEqual(sanitized["sections"]["decisions"]["items"][0]["grounding_status"], "grounded")
        self.assertEqual(sanitized["quality"]["evidence_gaps"], [])

    def test_template_language_overrides_the_inherited_report_language(self) -> None:
        template = custom_report_template()
        template["language_mode"] = "de"
        client = CustomTemplateClient()

        report = MultiPassMeetingIntelligencePipeline(
            client,
            report_template=template,
            report_language="en",
        ).generate(session_id="template-language", transcript_rows=sample_rows(1))

        self.assertEqual(report["report_language"], "de")
        self.assertTrue(all(payload["report_language"] == "de" for payload in client.payloads))

    def test_openai_json_parser_accepts_fenced_content(self) -> None:
        payload = parse_openai_chat_json({
            "choices": [
                {"message": {"content": "```json\n{\"schema_version\":\"meeting_section_v1\",\"items\":[]}\n```"}}
            ]
        })

        self.assertEqual(payload["schema_version"], "meeting_section_v1")


if __name__ == "__main__":
    unittest.main()
