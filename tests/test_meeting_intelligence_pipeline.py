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
    openai_strict_schema,
    parse_openai_chat_json,
    section_schema,
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


class MeetingIntelligencePipelineTests(unittest.TestCase):
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

    def test_openai_json_parser_accepts_fenced_content(self) -> None:
        payload = parse_openai_chat_json({
            "choices": [
                {"message": {"content": "```json\n{\"schema_version\":\"meeting_section_v1\",\"items\":[]}\n```"}}
            ]
        })

        self.assertEqual(payload["schema_version"], "meeting_section_v1")


if __name__ == "__main__":
    unittest.main()
