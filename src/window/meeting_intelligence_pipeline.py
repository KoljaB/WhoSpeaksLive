"""Multi-pass LLM meeting intelligence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Iterable, Protocol
import unicodedata
import urllib.error
import urllib.request

from window.language_config import get_language_config
from window.meeting_intelligence import TranscriptRow, normalize_transcript_rows, transcript_revision_id
from window.report_templates import (
    STANDARD_TEMPLATE_ID,
    get_builtin_report_template,
    validate_report_template,
)


PIPELINE_SCHEMA_VERSION = "meeting_intelligence_report_v2"
DEFAULT_SECTION_TYPES = (
    "speaker_map",
    "executive_summary",
    "structured_brief",
    "decisions",
    "action_items",
    "open_questions",
    "risks",
    "discussion_threads",
    "disagreements",
    "deadlines",
    "speaker_participation",
    "ask_this_meeting",
)


def normalize_report_language(value: str | None) -> tuple[str, str]:
    """Return the project's canonical language code and name for report prompts."""

    config = get_language_config(value or "en")
    return config.code, config.display_name


@dataclass(frozen=True)
class MeetingLLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 900.0
    max_tokens: int = 4096
    section_max_tokens: int = 4096
    temperature: float = 0.0
    schema_mode: str = "both"
    client_name: str = "whospeaks-meeting-intelligence"
    lane: str = "marvin"
    enable_thinking: bool = False


def default_llm_config(provider: str = "llama_cpp", **overrides: Any) -> MeetingLLMConfig:
    normalized = str(provider or "llama_cpp").strip().lower().replace("-", "_")
    defaults: dict[str, dict[str, Any]] = {
        "llama_cpp": {
            "base_url": os.environ.get("WHOSPEAKS_MI_LLM_BASE_URL", "http://127.0.0.1:8081/v1"),
            "model": os.environ.get("WHOSPEAKS_MI_LLM_MODEL", "local"),
            "schema_mode": "both",
        },
        "ollama": {
            "base_url": os.environ.get("WHOSPEAKS_MI_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
            "model": os.environ.get("WHOSPEAKS_MI_LLM_MODEL", "gemma3"),
            "schema_mode": "response_format",
        },
        "lm_studio": {
            "base_url": os.environ.get("WHOSPEAKS_MI_LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
            "model": os.environ.get("WHOSPEAKS_MI_LLM_MODEL", "local-model"),
            "schema_mode": "response_format",
        },
        "openai": {
            "base_url": os.environ.get("WHOSPEAKS_MI_LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": os.environ.get("WHOSPEAKS_MI_LLM_MODEL", "gpt-5.6-luna"),
            "schema_mode": "response_format",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
        "openrouter": {
            "base_url": os.environ.get("WHOSPEAKS_MI_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            "model": os.environ.get("WHOSPEAKS_MI_LLM_MODEL", "google/gemma-3-12b-it"),
            "schema_mode": "response_format",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        },
    }
    if normalized not in defaults:
        raise ValueError(f"Unsupported meeting LLM provider: {provider}")
    raw = dict(defaults[normalized])
    raw.update(overrides)
    return MeetingLLMConfig(
        provider=normalized,
        base_url=str(raw.get("base_url") or "").rstrip("/"),
        model=str(raw.get("model") or ""),
        api_key=str(raw.get("api_key") or os.environ.get("WHOSPEAKS_MI_LLM_API_KEY", "")),
        timeout_seconds=float(raw.get("timeout_seconds", 900.0)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        section_max_tokens=int(raw.get("section_max_tokens", raw.get("max_tokens", 4096))),
        temperature=float(raw.get("temperature", 0.0)),
        schema_mode=str(raw.get("schema_mode") or "both"),
        client_name=str(raw.get("client_name") or "whospeaks-meeting-intelligence"),
        lane=str(raw.get("lane") or "marvin"),
        enable_thinking=bool(raw.get("enable_thinking", False)),
    )


class StructuredChatClient(Protocol):
    name: str

    def chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        """Return a JSON object for one structured chat pass."""


class OpenAICompatibleMeetingClient:
    name = "openai_compatible"

    def __init__(self, config: MeetingLLMConfig) -> None:
        self.config = config
        self.name = f"{config.provider}:{config.model}"

    def chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        last_error: RuntimeError | None = None
        for attempt in range(2):
            payload = self._build_payload(
                schema_name=schema_name,
                schema=schema,
                system_prompt=system_prompt,
                user_payload=user_payload,
                max_tokens=max_tokens if attempt == 0 else max(max_tokens, min(max_tokens * 2, max_tokens + 2048)),
            )
            if attempt:
                payload["messages"].append({
                    "role": "user",
                    "content": (
                        "The previous response was invalid or truncated JSON. "
                        "Retry with fewer items and return only one complete JSON object matching the schema."
                    ),
                })
            headers = {
                "Content-Type": "application/json",
                "X-LLM-Client": self.config.client_name,
                "X-LLM-Lane": self.config.lane,
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            request = urllib.request.Request(
                f"{self.config.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = read_http_error_detail(exc)
                raise RuntimeError(f"Meeting LLM request failed: HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Meeting LLM request failed: {exc}") from exc
            try:
                result = parse_openai_chat_json(response_data)
            except RuntimeError as exc:
                last_error = exc
                if "meeting_llm_invalid_json" in str(exc) and attempt == 0:
                    continue
                raise
            result.setdefault("_request_elapsed_seconds", time.perf_counter() - started)
            return result
        raise last_error or RuntimeError("meeting_llm_invalid_json")

    def _build_payload(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }
        if self.config.provider in {"llama_cpp", "ollama", "lm_studio"}:
            payload["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
        if self.config.schema_mode in {"json_schema", "both"}:
            payload["json_schema"] = schema
        if self.config.schema_mode in {"response_format", "both"}:
            response_schema = openai_strict_schema(schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        return payload


class MockMeetingLLMClient:
    name = "mock_meeting_llm"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    def chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls.append(schema_name)
        self.payloads.append(dict(user_payload))
        report_language, _report_language_label = normalize_report_language(user_payload.get("report_language"))
        if schema_name == "meeting_evidence_index":
            rows = user_payload.get("transcript_rows") or []
            report_sections = user_payload.get("report_sections") or []
            title = "Señal de reunión simulada" if report_language == "es" else "Mock meeting signal"
            summary = (
                "La transcripción contiene una decisión, tarea, pregunta o riesgo."
                if report_language == "es"
                else "The transcript contains a decision, task, question, or risk."
            )
            return {
                "schema_version": "meeting_evidence_index_v1",
                "items": [
                    {
                        "id": "EV-MOCK-001",
                        "title": title,
                        "summary": summary,
                        "row_ids": [str(item.get("row_id")) for item in rows[:3] if isinstance(item, dict)],
                        "section_keys": [
                            str(item.get("key"))
                            for item in report_sections
                            if isinstance(item, dict) and str(item.get("key") or "").strip()
                        ],
                        "support_type": "direct",
                        "confidence": "High",
                    }
                ],
            }
        section = str(user_payload.get("section") or "summary")
        return mock_section_payload(section, report_language=report_language)


def mock_section_payload(section: str, *, report_language: str = "en") -> dict[str, Any]:
    spanish = report_language == "es"
    if section == "executive_summary":
        body = (
            "Borrador de resumen ejecutivo basado en la transcripción seleccionada."
            if spanish
            else "Draft executive summary from the selected transcript."
        )
        return {
            "schema_version": "meeting_section_v1",
            "section": section,
            "summary": body,
            "items": [
                common_item("SUMMARY-001", "Resumen ejecutivo" if spanish else "Executive summary", body, "EV-MOCK-001")
            ],
        }
    if section == "structured_brief":
        return {
            "schema_version": "meeting_section_v1",
            "section": section,
            "summary": "Structured brief",
            "items": [
                common_item("BRIEF-001", "Main topics", "Main topics, outcomes, and unresolved areas are available for review.", "EV-MOCK-001")
            ],
        }
    labels = {
        "speaker_map": ("SPK-001", "Speaker map", "Speakers and roles inferred from transcript evidence."),
        "decisions": ("DEC-001", "Decision candidate", "A decision candidate was extracted from transcript evidence."),
        "action_items": ("ACT-001", "Action item candidate", "An action item candidate was extracted and needs review."),
        "open_questions": ("Q-001", "Open question", "An unresolved question remains after the meeting."),
        "risks": ("RISK-001", "Risk", "A possible risk or blocker was discussed."),
        "discussion_threads": ("THREAD-001", "Discussion thread", "The meeting moved through a coherent discussion thread."),
        "disagreements": ("DIS-001", "Disagreement", "The transcript contains a disagreement or unresolved tradeoff."),
        "deadlines": ("DATE-001", "Deadline", "A deadline or timing window was mentioned."),
        "quotes": ("QUOTE-001", "Quote", "A useful transcript quote is available."),
        "speaker_participation": ("PART-001", "Speaker participation", "Speaker contribution summary."),
        "ask_this_meeting": ("QA-001", "What happened?", "Grounded answer based on cited evidence."),
    }
    item_id, title, body = labels.get(section, ("ITEM-001", section.replace("_", " ").title(), "Draft item."))
    return {
        "schema_version": "meeting_section_v1",
        "section": section,
        "summary": body,
        "items": [common_item(item_id, title, body, "EV-MOCK-001")],
    }


def common_item(item_id: str, title: str, body: str, evidence_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "body": body,
        "status": "draft",
        "owner": "",
        "due": "",
        "confidence": "Medium",
        "evidence_ids": [evidence_id],
        "attributes": [],
        "grounding_status": "grounded",
        "metadata": {},
    }


def parse_openai_chat_json(response_data: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("meeting_llm_missing_chat_content") from exc
    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    if not isinstance(content, str):
        raise RuntimeError("meeting_llm_chat_content_not_string")
    try:
        payload = json.loads(strip_json_fences(content))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"meeting_llm_invalid_json: {content[:800]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("meeting_llm_json_not_object")
    return payload


def strip_json_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def read_http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    if not body:
        return str(exc.reason or exc)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:1200]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        param = str(error.get("param") or "").strip()
        parts = [message, f"code={code}" if code else "", f"param={param}" if param else ""]
        return " / ".join(part for part in parts if part)[:1200]
    return body[:1200]


def openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy adjusted for OpenAI strict structured outputs."""
    return _openai_strict_node(schema)


def _openai_strict_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_openai_strict_node(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {key: _openai_strict_node(value) for key, value in node.items()}
    for keyword in ("anyOf", "oneOf", "allOf"):
        if isinstance(result.get(keyword), list):
            result[keyword] = [_openai_strict_node(item) for item in result[keyword]]
    node_type = result.get("type")
    is_object = node_type == "object" or (
        isinstance(node_type, list) and "object" in node_type
    )
    if is_object:
        properties = result.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        properties = {
            str(key): _openai_strict_node(value)
            for key, value in properties.items()
            if isinstance(value, dict)
        }
        result["properties"] = properties
        result["additionalProperties"] = False
        required = [str(value) for value in result.get("required") or []]
        for key in properties:
            if key not in required:
                required.append(key)
        result["required"] = required
    if "items" in result:
        result["items"] = _openai_strict_node(result["items"])
    return result


def resolve_pipeline_report_template(
    *,
    report_template: dict[str, Any] | None,
    section_types: Iterable[str] | None,
) -> dict[str, Any]:
    """Resolve every pipeline run to one normalized report-template snapshot."""

    if report_template is not None and section_types is not None:
        raise ValueError("Pass either report_template or legacy section_types, not both")
    if report_template is not None:
        return validate_report_template(
            report_template,
            allow_builtin=bool(report_template.get("builtin")),
        )
    if section_types is None:
        template = get_builtin_report_template(STANDARD_TEMPLATE_ID)
        if not isinstance(template, dict):
            raise RuntimeError(f"Built-in report template is unavailable: {STANDARD_TEMPLATE_ID}")
        return json.loads(json.dumps(template))

    keys: list[str] = []
    seen: set[str] = set()
    for value in section_types:
        key = str(value or "").strip()
        if not key or key in seen:
            continue
        keys.append(key)
        seen.add(key)
    legacy_template: dict[str, Any] = {
        "schema_version": "report_template_v1",
        "template_id": "custom.legacy",
        "name": "Legacy section selection",
        "description": "Compatibility template synthesized from section_types.",
        "version": 1,
        "builtin": False,
        "language_mode": "inherit",
        "privacy_policy": "inherit",
        "sections": [
            legacy_section_definition(key, position=index)
            for index, key in enumerate(keys)
        ],
    }
    if legacy_template["sections"]:
        return validate_report_template(legacy_template)
    # Older callers could explicitly request no section passes. The template
    # validator intentionally rejects empty user templates, so retain that
    # narrow compatibility case as a complete, provenance-bearing snapshot.
    legacy_template["revision_hash"] = stable_hash(legacy_template, length=16)
    return legacy_template


def legacy_section_definition(section: str, *, position: int) -> dict[str, Any]:
    """Build a template definition only for the deprecated section_types API."""

    policies = section_policy(section)
    objective = " ".join(policies[:-1] or policies)
    return {
        "key": section,
        "title": section.replace("_", " ").title(),
        "objective": objective or "Summarize only what the evidence supports.",
        "max_items": section_max_items(section),
        "evidence_required": True,
        "render_kind": "cards",
        "sort_order": "relevance",
        "output_fields": [],
    }


def compact_section_definitions(definitions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the section goals needed by each evidence-extraction pass."""

    result: list[dict[str, Any]] = []
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        result.append({
            "key": definition.get("key"),
            "title": definition.get("title"),
            "objective": definition.get("objective"),
            "max_items": definition.get("max_items"),
            "evidence_required": definition.get("evidence_required"),
            "render_kind": definition.get("render_kind"),
            "sort_order": definition.get("sort_order"),
            "output_fields": json.loads(json.dumps(definition.get("output_fields") or [])),
        })
    return result


class MultiPassMeetingIntelligencePipeline:
    def __init__(
        self,
        client: StructuredChatClient,
        *,
        max_segment_rows: int = 80,
        section_types: Iterable[str] | None = None,
        report_template: dict[str, Any] | None = None,
        evidence_max_tokens: int = 4096,
        section_max_tokens: int = 4096,
        report_language: str = "en",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.max_segment_rows = max(12, int(max_segment_rows))
        self.report_template = resolve_pipeline_report_template(
            report_template=report_template,
            section_types=section_types,
        )
        self.section_definitions = tuple(
            dict(section)
            for section in self.report_template.get("sections") or []
            if isinstance(section, dict)
        )
        self.section_types = tuple(str(section["key"]) for section in self.section_definitions)
        self.evidence_max_tokens = int(evidence_max_tokens)
        self.section_max_tokens = int(section_max_tokens)
        template_language = str(self.report_template.get("language_mode") or "inherit")
        effective_report_language = report_language if template_language == "inherit" else template_language
        self.report_language, self.report_language_label = normalize_report_language(effective_report_language)
        self.progress_callback = progress_callback

    def generate(
        self,
        *,
        session_id: str,
        transcript_rows: Iterable[dict[str, Any]],
        speaker_state: dict[str, Any] | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        rows = normalize_transcript_rows(transcript_rows)
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        revision_id = transcript_revision_id([row_to_source_dict(row) for row in rows], speaker_state or {})
        self._emit_progress(
            stage="prepare",
            message="Preparing transcript",
            detail=f"{len(rows)} transcript rows",
            completed_steps=0,
            total_steps=1,
        )
        segments = segment_transcript(rows, max_segment_rows=self.max_segment_rows)
        total_steps = max(1, len(segments) + len(self.section_types) + 2)
        completed_steps = 1
        self._emit_progress(
            stage="segment",
            message="Transcript split into segments",
            detail=f"{len(segments)} segments, {len(self.section_types)} report sections",
            completed_steps=completed_steps,
            total_steps=total_steps,
            current=0,
            total=len(segments),
        )
        evidence_items: list[dict[str, Any]] = []
        for index, segment in enumerate(segments, start=1):
            self._emit_progress(
                stage="evidence",
                message=f"Extracting evidence from segment {index} of {len(segments)}",
                detail=str(segment.get("title") or segment.get("id") or ""),
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=index,
                total=len(segments),
            )
            evidence_items.extend(self._extract_evidence(segment))
            completed_steps += 1
            self._emit_progress(
                stage="evidence",
                message=f"Evidence segment {index} complete",
                detail=f"{len(evidence_items)} evidence anchors so far",
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=index,
                total=len(segments),
            )
        evidence_items = normalize_evidence_items(
            evidence_items,
            rows,
            valid_section_keys=set(self.section_types),
            allow_fallback=False,
        )
        repair_definitions = [
            definition
            for definition in self.section_definitions
            if bool(definition.get("evidence_required", True))
            and not evidence_for_section(evidence_items, str(definition["key"]))
        ]
        repaired_section_keys = [str(definition["key"]) for definition in repair_definitions]
        if repair_definitions and segments:
            self._emit_progress(
                stage="evidence",
                message="Repairing evidence coverage",
                detail=f"Searching again for {len(repair_definitions)} uncovered report sections",
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=len(segments),
                total=len(segments),
            )
            repair_items: list[dict[str, Any]] = []
            for segment in segments:
                repair_items.extend(
                    self._extract_evidence(
                        segment,
                        section_definitions=repair_definitions,
                        coverage_repair=True,
                    )
                )
            evidence_items = normalize_evidence_items(
                [*evidence_items, *repair_items],
                rows,
                valid_section_keys=set(self.section_types),
            )
        self._emit_progress(
            stage="evidence",
            message="Evidence index normalized",
            detail=f"{len(evidence_items)} evidence anchors",
            completed_steps=completed_steps,
            total_steps=total_steps,
            current=len(segments),
            total=len(segments),
        )
        sections: dict[str, Any] = {}
        global_context = {
            "session_id": str(session_id or ""),
            "report_title": title or str(self.report_template.get("name") or ""),
            "template_id": str(self.report_template.get("template_id") or ""),
            "template_name": str(self.report_template.get("name") or ""),
            "template_description": str(self.report_template.get("description") or ""),
            "template_version": self.report_template.get("version"),
            "template_revision": str(self.report_template.get("revision_hash") or ""),
            "speaker_state": speaker_state or {},
            "transcript_outline": transcript_outline(rows),
        }
        for index, definition in enumerate(self.section_definitions, start=1):
            section = str(definition["key"])
            self._emit_progress(
                stage="section",
                message=f"Generating section {index} of {len(self.section_types)}",
                detail=str(definition.get("title") or section.replace("_", " ")),
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=index,
                total=len(self.section_types),
            )
            sections[section] = self._extract_section(
                definition,
                evidence_items,
                rows,
                global_context=global_context,
            )
            completed_steps += 1
            self._emit_progress(
                stage="section",
                message=f"Section complete: {section.replace('_', ' ')}",
                detail=f"{len(sections[section].get('items') or [])} items",
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=index,
                total=len(self.section_types),
            )
        self._emit_progress(
            stage="finalize",
            message="Assembling final report",
            detail="Writing sections and evidence index",
            completed_steps=completed_steps,
            total_steps=total_steps,
        )
        report = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "report_id": f"mir2_{stable_hash({'session_id': session_id, 'revision': revision_id, 'template_revision': self.report_template.get('revision_hash'), 'generated_at': generated_at}, length=16)}",
            "session_id": str(session_id or ""),
            "title": title or str(self.report_template.get("name") or "Meeting intelligence report"),
            "generated_at": generated_at,
            "updated_at": generated_at,
            "transcript_revision_id": revision_id,
            "provider": getattr(self.client, "name", "structured_chat_client"),
            "report_language": self.report_language,
            "template_id": str(self.report_template.get("template_id") or ""),
            "template_revision": str(self.report_template.get("revision_hash") or ""),
            "report_template": json.loads(json.dumps(self.report_template)),
            "pipeline": {
                "mode": "multi_pass",
                "segments": len(segments),
                "section_passes": list(self.section_types),
                "evidence_coverage_repair": repaired_section_keys,
            },
            "summary": section_summary(sections.get("executive_summary")),
            "evidence_index": evidence_items,
            "sections": sections,
            "quality": {
                "needs_human_review": True,
                "evidence_gaps": find_evidence_gaps(sections),
                "local_first": False,
            },
        }
        self._emit_progress(
            stage="completed",
            message="Report generated",
            detail=f"{len(evidence_items)} evidence anchors, {len(sections)} sections",
            completed_steps=total_steps,
            total_steps=total_steps,
            current=total_steps,
            total=total_steps,
        )
        return report

    def _emit_progress(self, **event: Any) -> None:
        if self.progress_callback is None:
            return
        completed = int(event.get("completed_steps") or 0)
        total = max(1, int(event.get("total_steps") or 1))
        payload = dict(event)
        payload["percent"] = max(0, min(100, int(round((completed / total) * 100))))
        payload["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.progress_callback(payload)

    def _extract_evidence(
        self,
        segment: dict[str, Any],
        *,
        section_definitions: Iterable[dict[str, Any]] | None = None,
        coverage_repair: bool = False,
    ) -> list[dict[str, Any]]:
        definitions = list(section_definitions or self.section_definitions)
        report_sections = compact_section_definitions(definitions)
        max_items = evidence_max_items_for_sections(len(report_sections))
        instructions = [
            "Create compact evidence anchors for this transcript segment.",
            "Return no more evidence anchors than max_items.",
            "Use row_ids exactly as supplied.",
            "Target the supplied report section objectives and output fields.",
            "Assign every anchor to each applicable section by using exact section keys in section_keys.",
            f"Write all generated titles and summaries in {self.report_language_label}.",
        ]
        if coverage_repair:
            instructions.extend([
                "This is a coverage-repair pass: search carefully for support for every supplied report section.",
                "Recognize implicit decisions, commitments, actions, owners, deadlines, disagreements, and outcomes even when no formal keyword is used.",
                "Use only the supplied section keys; one anchor may support multiple supplied sections.",
                "Return an empty list only when this segment genuinely contains no support for any supplied section.",
            ])
        payload = {
            "schema_version": "meeting_evidence_index_v1",
            "segment": {key: segment[key] for key in ("id", "title", "start", "end")},
            "max_items": max_items,
            "instructions": instructions,
            "coverage_repair": coverage_repair,
            "report_language": self.report_language,
            "report_sections": report_sections,
            "transcript_rows": [row_to_prompt_dict(row) for row in segment["rows"]],
        }
        result = self.client.chat_json(
            schema_name="meeting_evidence_index",
            schema=evidence_index_schema(max_items=max_items),
            system_prompt=evidence_system_prompt(),
            user_payload=payload,
            max_tokens=self.evidence_max_tokens,
        )
        return result.get("items") if isinstance(result.get("items"), list) else []

    def _extract_section(
        self,
        definition: dict[str, Any],
        evidence_items: list[dict[str, Any]],
        rows: list[TranscriptRow],
        *,
        global_context: dict[str, Any],
    ) -> dict[str, Any]:
        section = str(definition["key"])
        max_items = definition_max_items(definition)
        relevant_evidence = evidence_for_section(evidence_items, section)
        evidence_required = bool(definition.get("evidence_required", True))
        if evidence_required and not relevant_evidence:
            return normalize_section(
                section,
                {
                    "schema_version": "meeting_section_v1",
                    "section": section,
                    "summary": "",
                    "items": [],
                },
                set(),
                definition=definition,
            )
        payload = {
            "schema_version": "meeting_section_v1",
            "section": section,
            "section_definition": json.loads(json.dumps(definition)),
            "max_items": max_items,
            "section_policy": [str(definition.get("objective") or "")],
            "evidence_index": compact_evidence(relevant_evidence),
            "global_context": global_context,
            "transcript_outline": transcript_outline(rows),
            "output_contract": [
                "Return concise JSON only.",
                "Use no more items than max_items.",
                "Keep each body under 45 words.",
                "Use only evidence_ids from evidence_index.",
                "When section_definition.evidence_required is true, every item must cite at least one valid evidence_id. Omit claims that cannot be cited.",
                "Return custom attributes only for keys declared in section_definition.output_fields.",
                "Set grounding_status to grounded only when the cited evidence directly supports the item.",
                f"Write every generated title, summary, body, status, owner, and due value in {self.report_language_label}; preserve quoted transcript text as-is.",
            ],
            "report_language": self.report_language,
        }
        result = self.client.chat_json(
            schema_name="meeting_section",
            schema=section_schema(max_items=max_items),
            system_prompt=section_system_prompt(section),
            user_payload=payload,
            max_tokens=self.section_max_tokens,
        )
        normalized = normalize_section(
            section,
            result,
            {item["id"] for item in relevant_evidence},
            definition=definition,
        )
        if evidence_required and result.get("items") and not normalized.get("items"):
            retry_payload = json.loads(json.dumps(payload))
            retry_payload["citation_repair"] = True
            retry_payload["output_contract"].extend([
                "The previous response contained claims but no usable evidence citation.",
                "Re-evaluate each claim and cite the exact matching IDs from evidence_index.",
                "Do not drop a supported claim merely because its previous evidence ID was invalid.",
            ])
            retry_result = self.client.chat_json(
                schema_name="meeting_section",
                schema=section_schema(max_items=max_items),
                system_prompt=section_system_prompt(section),
                user_payload=retry_payload,
                max_tokens=self.section_max_tokens,
            )
            normalized = normalize_section(
                section,
                retry_result,
                {item["id"] for item in relevant_evidence},
                definition=definition,
            )
        return normalized


def segment_transcript(rows: list[TranscriptRow], *, max_segment_rows: int = 80) -> list[dict[str, Any]]:
    if not rows:
        return []
    segments: list[dict[str, Any]] = []
    current: list[TranscriptRow] = []
    for row in rows:
        starts_new_topic = bool(re.search(
            r"\b(next item|moving on|agenda|parking|morale|finance|financial|training|cleanliness|risk|decision|issue)\b",
            row.text,
            flags=re.IGNORECASE,
        ))
        if current and (len(current) >= max_segment_rows or (len(current) >= 16 and starts_new_topic)):
            segments.append(segment_from_rows(current, len(segments) + 1))
            current = []
        current.append(row)
    if current:
        segments.append(segment_from_rows(current, len(segments) + 1))
    return segments


def segment_from_rows(rows: list[TranscriptRow], index: int) -> dict[str, Any]:
    title = title_from_rows(rows) or f"Segment {index}"
    return {
        "id": f"SEG-{index:03d}",
        "title": title,
        "start": format_seconds(rows[0].start),
        "end": format_seconds(rows[-1].end),
        "rows": rows,
    }


from window.meeting_pipeline_support import (
    title_from_rows,
    evidence_system_prompt,
    section_system_prompt,
    section_policy,
    section_max_items,
    evidence_max_items_for_sections,
    evidence_index_schema,
    section_schema,
    definition_max_items,
    output_field_keys,
    normalize_attributes,
    normalize_section_keys,
    evidence_for_section,
    normalize_evidence_items,
    normalize_section,
    compact_evidence,
    transcript_outline,
    row_to_source_dict,
    row_to_prompt_dict,
    section_summary,
    find_evidence_gaps,
    format_seconds,
    normalize_id,
    clean_text,
    repair_mangled_unicode_text,
    sanitize_report_output,
    stable_hash,
)
