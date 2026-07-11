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
import urllib.error
import urllib.request

from window.language_config import get_language_config
from window.meeting_intelligence import TranscriptRow, normalize_transcript_rows, transcript_revision_id


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


class MultiPassMeetingIntelligencePipeline:
    def __init__(
        self,
        client: StructuredChatClient,
        *,
        max_segment_rows: int = 80,
        section_types: Iterable[str] = DEFAULT_SECTION_TYPES,
        evidence_max_tokens: int = 4096,
        section_max_tokens: int = 4096,
        report_language: str = "en",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.max_segment_rows = max(12, int(max_segment_rows))
        self.section_types = tuple(section_types)
        self.evidence_max_tokens = int(evidence_max_tokens)
        self.section_max_tokens = int(section_max_tokens)
        self.report_language, self.report_language_label = normalize_report_language(report_language)
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
        evidence_items = normalize_evidence_items(evidence_items, rows)
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
        for index, section in enumerate(self.section_types, start=1):
            self._emit_progress(
                stage="section",
                message=f"Generating section {index} of {len(self.section_types)}",
                detail=section.replace("_", " "),
                completed_steps=completed_steps,
                total_steps=total_steps,
                current=index,
                total=len(self.section_types),
            )
            sections[section] = self._extract_section(section, evidence_items, rows)
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
            "report_id": f"mir2_{stable_hash({'session_id': session_id, 'revision': revision_id, 'generated_at': generated_at}, length=16)}",
            "session_id": str(session_id or ""),
            "title": title or "Meeting intelligence report",
            "generated_at": generated_at,
            "updated_at": generated_at,
            "transcript_revision_id": revision_id,
            "provider": getattr(self.client, "name", "structured_chat_client"),
            "report_language": self.report_language,
            "pipeline": {
                "mode": "multi_pass",
                "segments": len(segments),
                "section_passes": list(self.section_types),
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

    def _extract_evidence(self, segment: dict[str, Any]) -> list[dict[str, Any]]:
        payload = {
            "schema_version": "meeting_evidence_index_v1",
            "segment": {key: segment[key] for key in ("id", "title", "start", "end")},
            "instructions": [
                "Create compact evidence anchors for this transcript segment.",
                "Use row_ids exactly as supplied.",
                "Prefer items that can support later decisions, actions, risks, open questions, disagreements, deadlines, speaker identity, and summary claims.",
                f"Write all generated titles and summaries in {self.report_language_label}.",
            ],
            "report_language": self.report_language,
            "transcript_rows": [row_to_prompt_dict(row) for row in segment["rows"]],
        }
        result = self.client.chat_json(
            schema_name="meeting_evidence_index",
            schema=evidence_index_schema(),
            system_prompt=evidence_system_prompt(),
            user_payload=payload,
            max_tokens=self.evidence_max_tokens,
        )
        return result.get("items") if isinstance(result.get("items"), list) else []

    def _extract_section(
        self,
        section: str,
        evidence_items: list[dict[str, Any]],
        rows: list[TranscriptRow],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "meeting_section_v1",
            "section": section,
            "max_items": section_max_items(section),
            "section_policy": section_policy(section),
            "evidence_index": compact_evidence(evidence_items),
            "transcript_outline": transcript_outline(rows),
            "output_contract": [
                "Return concise JSON only.",
                "Use no more items than max_items.",
                "Keep each body under 45 words.",
                "Use only evidence_ids from evidence_index.",
                f"Write every generated title, summary, body, status, owner, and due value in {self.report_language_label}; preserve quoted transcript text as-is.",
            ],
            "report_language": self.report_language,
        }
        result = self.client.chat_json(
            schema_name="meeting_section",
            schema=section_schema(),
            system_prompt=section_system_prompt(section),
            user_payload=payload,
            max_tokens=self.section_max_tokens,
        )
        return normalize_section(section, result, {item["id"] for item in evidence_items})


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


def title_from_rows(rows: list[TranscriptRow]) -> str:
    text = " ".join(row.text for row in rows[:8]).casefold()
    for label, pattern in (
        ("Parking", r"parking|car park"),
        ("Staff morale", r"morale|sickness|appraisal"),
        ("IT and systems", r"\bit\b|power|electricity|mainframe"),
        ("Training", r"training|software"),
        ("Finance", r"financial|balance|revenue|black|red"),
        ("Cleanliness", r"cleanliness|kitchen|shower|dishwasher"),
    ):
        if re.search(pattern, text):
            return label
    return ""


def evidence_system_prompt() -> str:
    return (
        "You build evidence anchors from a speaker-labeled meeting transcript segment. "
        "Return JSON only. Evidence is an auditable support span with row_ids, not a conclusion. "
        "Do not invent row_ids or facts not present in the segment. Follow the requested report language exactly."
    )


def section_system_prompt(section: str) -> str:
    return (
        "You extract one section of a post-meeting intelligence report from evidence anchors. "
        "Return JSON only. Cite evidence_ids on every material item. Preserve uncertainty: "
        "separate decided, suggested, unclear, and unresolved. Do not invent owners or dates. "
        "Be concise: no more than the requested max_items, short titles, and short bodies. "
        f"Current section: {section}. Follow the requested report language exactly."
    )


def section_policy(section: str) -> list[str]:
    policies = {
        "speaker_map": [
            "Map speakers to names or roles only when transcript evidence supports it.",
            "If identity is unclear, keep the speaker label and describe role evidence.",
        ],
        "executive_summary": [
            "Write a short executive summary of the whole meeting.",
            "Prefer concrete outcomes, unresolved issues, and operational context.",
        ],
        "structured_brief": [
            "Create a compact topic brief with outcomes and unresolved follow-up.",
        ],
        "decisions": [
            "Extract final decisions only.",
            "Use status decided/proposed/needs_review when finality is ambiguous.",
        ],
        "action_items": [
            "Extract confirmed and candidate follow-ups.",
            "Owner and due date must be evidence-grounded or explicitly marked unclear.",
        ],
        "open_questions": ["Extract unresolved questions and unclear ownership/date issues."],
        "risks": ["Extract risks, blockers, and unresolved operational hazards."],
        "discussion_threads": ["Extract the main topic threads in meeting order."],
        "disagreements": ["Extract disagreements, positions, and final status if any."],
        "deadlines": ["Extract explicit deadlines, future meetings, and timing windows."],
        "speaker_participation": ["Summarize concrete speaker contributions without judging performance."],
        "ask_this_meeting": ["Create grounded Q&A answers that cite evidence ids."],
    }
    return [
        *policies.get(section, ["Summarize only what the evidence supports."]),
        f"Return at most {section_max_items(section)} items.",
    ]


def section_max_items(section: str) -> int:
    return {
        "executive_summary": 3,
        "structured_brief": 5,
        "speaker_map": 8,
        "speaker_participation": 8,
        "discussion_threads": 8,
        "ask_this_meeting": 6,
    }.get(section, 6)


def evidence_index_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "row_ids": {"type": "array", "items": {"type": "string"}},
            "support_type": {"type": "string"},
            "confidence": {"type": "string"},
        },
        "required": ["id", "title", "summary", "row_ids", "support_type", "confidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "items": {"type": "array", "items": item, "maxItems": 8},
        },
        "required": ["schema_version", "items"],
    }


def section_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "status": {"type": "string"},
            "owner": {"type": "string"},
            "due": {"type": "string"},
            "confidence": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        "required": ["id", "title", "body", "status", "owner", "due", "confidence", "evidence_ids", "metadata"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "section": {"type": "string"},
            "summary": {"type": "string"},
            "items": {"type": "array", "items": item, "maxItems": 8},
        },
        "required": ["schema_version", "section", "summary", "items"],
    }


def normalize_evidence_items(items: Iterable[dict[str, Any]], rows: list[TranscriptRow]) -> list[dict[str, Any]]:
    row_by_id = {row.row_id: row for row in rows}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        row_ids = [str(value) for value in item.get("row_ids") or [] if str(value) in row_by_id]
        if not row_ids:
            continue
        evidence_id = normalize_id(item.get("id"), prefix="EV", fallback=f"EV-{position:03d}")
        if evidence_id in seen:
            evidence_id = f"{evidence_id}-{position:02d}"
        selected_rows = [row_by_id[row_id] for row_id in row_ids]
        normalized.append({
            "id": evidence_id,
            "title": clean_text(item.get("title"), limit=120),
            "summary": clean_text(item.get("summary"), limit=600),
            "row_ids": row_ids,
            "start": format_seconds(min(row.start for row in selected_rows)),
            "end": format_seconds(max(row.end for row in selected_rows)),
            "speakers": sorted({row.speaker_name for row in selected_rows}),
            "quote_excerpt": clean_text(" ".join(row.text for row in selected_rows[:4]), limit=600),
            "support_type": clean_text(item.get("support_type"), limit=40) or "direct",
            "confidence": clean_text(item.get("confidence"), limit=40) or "Medium",
        })
        seen.add(evidence_id)
    if not normalized and rows:
        selected = rows[: min(8, len(rows))]
        normalized.append({
            "id": "EV-FALLBACK-001",
            "title": "Transcript context",
            "summary": "Fallback evidence span from the transcript.",
            "row_ids": [row.row_id for row in selected],
            "start": format_seconds(selected[0].start),
            "end": format_seconds(selected[-1].end),
            "speakers": sorted({row.speaker_name for row in selected}),
            "quote_excerpt": clean_text(" ".join(row.text for row in selected[:4]), limit=600),
            "support_type": "context",
            "confidence": "Low",
        })
    return normalized


def normalize_section(section: str, result: dict[str, Any], valid_evidence_ids: set[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for raw in result.get("items") or []:
        if not isinstance(raw, dict):
            continue
        title = clean_text(raw.get("title"), limit=180)
        body = clean_text(raw.get("body"), limit=1600)
        key = f"{title.casefold()}|{body.casefold()}"
        if not title and not body:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        evidence_ids = [
            str(value)
            for value in raw.get("evidence_ids") or []
            if str(value) in valid_evidence_ids
        ]
        items.append({
            "id": normalize_id(raw.get("id"), prefix=section.upper()[:5], fallback=f"{section.upper()}-{len(items) + 1:03d}"),
            "title": title or body[:80] or section.replace("_", " ").title(),
            "body": body,
            "status": clean_text(raw.get("status"), limit=60) or "draft",
            "owner": clean_text(raw.get("owner"), limit=120),
            "due": clean_text(raw.get("due"), limit=120),
            "confidence": clean_text(raw.get("confidence"), limit=60) or "Medium",
            "evidence_ids": evidence_ids,
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        })
        if len(items) >= section_max_items(section):
            break
    return {
        "section": section,
        "summary": clean_text(result.get("summary"), limit=1600),
        "items": items,
    }


def compact_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "time": f"{item.get('start')}-{item.get('end')}",
            "speakers": item.get("speakers"),
            "quote_excerpt": item.get("quote_excerpt"),
        }
        for item in items
    ]


def transcript_outline(rows: list[TranscriptRow], *, max_rows: int = 80) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        selected = rows
    else:
        step = max(1, len(rows) // max_rows)
        selected = rows[::step][:max_rows]
    return [
        {
            "row_id": row.row_id,
            "time": f"{format_seconds(row.start)}-{format_seconds(row.end)}",
            "speaker": row.speaker_name,
            "text": row.text,
        }
        for row in selected
    ]


def row_to_source_dict(row: TranscriptRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "index": row.index,
        "start": row.start,
        "end": row.end,
        "text": row.text,
        "assigned_speaker": row.speaker_id,
        "speaker_name": row.speaker_name,
    }


def row_to_prompt_dict(row: TranscriptRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "time": f"{format_seconds(row.start)}-{format_seconds(row.end)}",
        "speaker": row.speaker_name,
        "text": row.text,
    }


def section_summary(section: Any) -> str:
    if isinstance(section, dict):
        summary = clean_text(section.get("summary"), limit=1600)
        if summary:
            return summary
        items = section.get("items") if isinstance(section.get("items"), list) else []
        if items:
            return clean_text(items[0].get("body") or items[0].get("title"), limit=1600)
    return ""


def find_evidence_gaps(sections: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            continue
        for index, item in enumerate(section.get("items") or []):
            if isinstance(item, dict) and not item.get("evidence_ids"):
                gaps.append(f"{section_name}[{index}]")
    return gaps


def format_seconds(value: float) -> str:
    total = max(0, int(round(float(value or 0.0))))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def normalize_id(value: Any, *, prefix: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    if not text:
        return fallback
    if not text.upper().startswith(prefix.upper()):
        return f"{prefix}-{text}"[:80]
    return text[:80]


def clean_text(value: Any, *, limit: int = 400) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def stable_hash(value: Any, *, length: int = 12) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
