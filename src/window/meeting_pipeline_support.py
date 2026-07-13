"""Schemas, prompts, normalization, and sanitization for meeting reports."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable
import unicodedata

from window.meeting_intelligence import TranscriptRow


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
        "If evidence is required, omit every item that lacks a directly supporting evidence anchor. "
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


def evidence_max_items_for_sections(section_count: int) -> int:
    return max(8, min(32, 2 * max(0, int(section_count))))


def evidence_index_schema(*, max_items: int = 8) -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "row_ids": {"type": "array", "items": {"type": "string"}},
            "section_keys": {"type": "array", "items": {"type": "string"}},
            "support_type": {"type": "string"},
            "confidence": {"type": "string"},
        },
        "required": ["id", "title", "summary", "row_ids", "section_keys", "support_type", "confidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "items": {"type": "array", "items": item, "maxItems": max(1, int(max_items))},
        },
        "required": ["schema_version", "items"],
    }


def section_schema(*, max_items: int = 8) -> dict[str, Any]:
    attribute = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    }
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
            "attributes": {"type": "array", "items": attribute},
            "grounding_status": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        "required": [
            "id",
            "title",
            "body",
            "status",
            "owner",
            "due",
            "confidence",
            "evidence_ids",
            "attributes",
            "grounding_status",
            "metadata",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "section": {"type": "string"},
            "summary": {"type": "string"},
            "items": {"type": "array", "items": item, "maxItems": max(1, int(max_items))},
        },
        "required": ["schema_version", "section", "summary", "items"],
    }


def definition_max_items(definition: dict[str, Any]) -> int:
    try:
        return max(1, min(50, int(definition.get("max_items", 8))))
    except (TypeError, ValueError):
        return 8


def output_field_keys(definition: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in definition.get("output_fields") or []:
        if isinstance(field, dict):
            key = str(field.get("key") or "").strip()
        else:
            # Tolerate early templates that used a compact list of keys.
            key = str(field or "").strip()
        if key:
            keys.add(key)
    return keys


def normalize_attributes(value: Any, allowed_keys: set[str]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return normalized
    for raw in value:
        if not isinstance(raw, dict):
            continue
        key = clean_text(raw.get("key"), limit=80)
        if key not in allowed_keys or key in seen:
            continue
        normalized.append({
            "key": key,
            "value": clean_text(raw.get("value"), limit=800),
        })
        seen.add(key)
    return normalized


def normalize_section_keys(value: Any, valid_section_keys: set[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value if isinstance(value, list) else []:
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        if valid_section_keys is not None and key not in valid_section_keys:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def evidence_for_section(items: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
    """Select section-targeted anchors while retaining unclassified global context."""

    return [
        item
        for item in items
        if not item.get("section_keys") or section in item.get("section_keys", [])
    ]


def normalize_evidence_items(
    items: Iterable[dict[str, Any]],
    rows: list[TranscriptRow],
    *,
    valid_section_keys: set[str] | None = None,
    allow_fallback: bool = True,
) -> list[dict[str, Any]]:
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
        section_keys = normalize_section_keys(item.get("section_keys"), valid_section_keys)
        normalized.append({
            "id": evidence_id,
            "title": clean_text(item.get("title"), limit=120),
            "summary": clean_text(item.get("summary"), limit=600),
            "row_ids": row_ids,
            "section_keys": section_keys,
            "start": format_seconds(min(row.start for row in selected_rows)),
            "end": format_seconds(max(row.end for row in selected_rows)),
            "speakers": sorted({row.speaker_name for row in selected_rows}),
            "quote_excerpt": clean_text(" ".join(row.text for row in selected_rows[:4]), limit=600),
            "support_type": clean_text(item.get("support_type"), limit=40) or "direct",
            "confidence": clean_text(item.get("confidence"), limit=40) or "Medium",
        })
        seen.add(evidence_id)
    if allow_fallback and not normalized and rows:
        selected = rows[: min(8, len(rows))]
        normalized.append({
            "id": "EV-FALLBACK-001",
            "title": "Transcript context",
            "summary": "Fallback evidence span from the transcript.",
            "row_ids": [row.row_id for row in selected],
            "section_keys": sorted(valid_section_keys or set()),
            "start": format_seconds(selected[0].start),
            "end": format_seconds(selected[-1].end),
            "speakers": sorted({row.speaker_name for row in selected}),
            "quote_excerpt": clean_text(" ".join(row.text for row in selected[:4]), limit=600),
            "support_type": "context",
            "confidence": "Low",
        })
    return normalized


def normalize_section(
    section: str,
    result: dict[str, Any],
    valid_evidence_ids: set[str],
    *,
    definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_definition = (
        json.loads(json.dumps(definition))
        if isinstance(definition, dict)
        else legacy_section_definition(section, position=0)
    )
    max_items = definition_max_items(normalized_definition)
    evidence_required = bool(normalized_definition.get("evidence_required", True))
    allowed_attribute_keys = output_field_keys(normalized_definition)
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
        if evidence_required and not evidence_ids:
            continue
        attributes = normalize_attributes(raw.get("attributes"), allowed_attribute_keys)
        if evidence_ids:
            grounding_status = "grounded"
        else:
            grounding_status = "not_required"
        items.append({
            "id": normalize_id(raw.get("id"), prefix=section.upper()[:5], fallback=f"{section.upper()}-{len(items) + 1:03d}"),
            "title": title or body[:80] or section.replace("_", " ").title(),
            "body": body,
            "status": clean_text(raw.get("status"), limit=60) or "draft",
            "owner": clean_text(raw.get("owner"), limit=120),
            "due": clean_text(raw.get("due"), limit=120),
            "confidence": clean_text(raw.get("confidence"), limit=60) or "Medium",
            "evidence_ids": evidence_ids,
            "attributes": attributes,
            "grounding_status": grounding_status,
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        })
        if len(items) >= max_items:
            break
    return {
        "section": section,
        "definition": normalized_definition,
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
            "section_keys": item.get("section_keys") or [],
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
    text = repair_mangled_unicode_text(str(value or ""))
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def repair_mangled_unicode_text(value: str) -> str:
    """Repair model output that encoded ``ü`` as a NUL plus ``fc``."""

    repaired = re.sub(
        r"\x00([0-9A-Fa-f]{2})",
        lambda match: chr(int(match.group(1), 16)),
        str(value or ""),
    )
    repaired = "".join(
        character
        for character in repaired
        if character in "\n\r\t" or ord(character) >= 32
    )
    return unicodedata.normalize("NFC", repaired)


def sanitize_report_output(report: dict[str, Any]) -> dict[str, Any]:
    """Repair cached text artifacts and remove uncited evidence-required items."""

    def repair(value: Any) -> Any:
        if isinstance(value, str):
            return repair_mangled_unicode_text(value)
        if isinstance(value, list):
            return [repair(item) for item in value]
        if isinstance(value, dict):
            return {key: repair(item) for key, item in value.items()}
        return value

    normalized = repair(report)
    sections = normalized.get("sections") if isinstance(normalized, dict) else None
    if not isinstance(sections, dict):
        return normalized
    valid_evidence_ids = {
        str(item.get("id"))
        for item in normalized.get("evidence_index") or []
        if isinstance(item, dict) and item.get("id")
    }
    for section in sections.values():
        if not isinstance(section, dict):
            continue
        definition = section.get("definition") if isinstance(section.get("definition"), dict) else {}
        evidence_required = bool(definition.get("evidence_required", True))
        items = section.get("items") if isinstance(section.get("items"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            item["evidence_ids"] = [
                str(evidence_id)
                for evidence_id in item.get("evidence_ids") or []
                if str(evidence_id) in valid_evidence_ids
            ]
        if evidence_required:
            section["items"] = [
                item
                for item in items
                if isinstance(item, dict) and bool(item.get("evidence_ids"))
            ]
        for item in section.get("items") or []:
            if isinstance(item, dict):
                item["grounding_status"] = "grounded" if item.get("evidence_ids") else "not_required"
    quality = normalized.get("quality")
    if isinstance(quality, dict):
        quality["evidence_gaps"] = find_evidence_gaps(sections)
    return normalized


def stable_hash(value: Any, *, length: int = 12) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

def legacy_section_definition(section: str, *, position: int) -> dict[str, Any]:
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


__all__ = [name for name in globals() if not name.startswith("_")]
