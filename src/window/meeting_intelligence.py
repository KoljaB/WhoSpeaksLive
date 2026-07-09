"""Evidence-grounded meeting intelligence reports for saved WhoSpeaks sessions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Protocol

from window.fact_lens_sidecar import evidence_matches_transcript


REPORT_SCHEMA_VERSION = "meeting_intelligence_v1"
OBJECT_STATUSES = {"draft", "accepted", "rejected", "edited", "stale", "needs_review"}
REPORT_STATUSES = {"draft", "reviewed", "partially_reviewed", "stale"}


@dataclass(frozen=True)
class TranscriptRow:
    row_id: str
    index: int
    start: float
    end: float
    text: str
    speaker_id: str
    speaker_name: str
    row_revision_id: str


class MeetingIntelligenceProvider(Protocol):
    name: str

    def extract_objects(
        self,
        rows: list[TranscriptRow],
        *,
        report_id: str,
        transcript_revision_id: str,
        generated_at: str,
    ) -> list[dict[str, Any]]:
        """Return draft meeting objects grounded in the supplied transcript rows."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_transcript_rows(rows: Iterable[dict[str, Any]]) -> list[TranscriptRow]:
    normalized: list[TranscriptRow] = []
    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        text = " ".join(str(raw.get("text") or "").strip().split())
        if not text:
            continue
        try:
            index = int(raw.get("index") or position)
        except (TypeError, ValueError):
            index = position
        row_id = str(raw.get("row_id") or raw.get("id") or f"row_{index}").strip()
        if not row_id:
            row_id = f"row_{index}"
        try:
            start = float(raw.get("start") or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(raw.get("end") or start)
        except (TypeError, ValueError):
            end = start
        speaker_id = str(raw.get("assigned_speaker") or raw.get("speaker_id") or "").strip()
        speaker_name = str(raw.get("speaker_name") or raw.get("display_speaker") or "").strip()
        if not speaker_name:
            speaker_name = _speaker_display_name(speaker_id)
        revision_payload = {
            "row_id": row_id,
            "index": index,
            "start": round(start, 4),
            "end": round(end, 4),
            "text": text,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "correction": raw.get("correction") if isinstance(raw.get("correction"), dict) else {},
        }
        normalized.append(
            TranscriptRow(
                row_id=row_id,
                index=index,
                start=start,
                end=end,
                text=text,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                row_revision_id=f"rr_{_stable_hash(revision_payload, length=16)}",
            )
        )
    return normalized


def transcript_revision_id(rows: Iterable[dict[str, Any]], speaker_state: dict[str, Any] | None = None) -> str:
    normalized_rows = normalize_transcript_rows(rows)
    return _transcript_revision_id_from_normalized(normalized_rows, speaker_state or {})


def _transcript_revision_id_from_normalized(
    normalized_rows: list[TranscriptRow],
    speaker_state: dict[str, Any] | None = None,
) -> str:
    speakers = []
    if isinstance(speaker_state, dict):
        for speaker in speaker_state.get("speakers") or []:
            if not isinstance(speaker, dict):
                continue
            speakers.append({
                "id": str(speaker.get("id") or ""),
                "name": str(speaker.get("name") or speaker.get("display_name") or ""),
                "display_name": str(speaker.get("display_name") or speaker.get("name") or ""),
            })
    payload = {
        "rows": [
            {
                "row_id": row.row_id,
                "row_revision_id": row.row_revision_id,
                "speaker_id": row.speaker_id,
                "speaker_name": row.speaker_name,
            }
            for row in normalized_rows
        ],
        "speakers": sorted(speakers, key=lambda item: item["id"]),
    }
    return f"tr_{_stable_hash(payload, length=20)}"


def generate_meeting_report(
    *,
    session_id: str,
    transcript_rows: Iterable[dict[str, Any]],
    speaker_state: dict[str, Any] | None = None,
    provider: MeetingIntelligenceProvider | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = normalize_transcript_rows(transcript_rows)
    generated_at = generated_at or utc_now_iso()
    revision_id = _transcript_revision_id_from_normalized(rows, speaker_state or {})
    report_id = f"mir_{_stable_hash({'session_id': session_id, 'revision_id': revision_id, 'generated_at': generated_at}, length=16)}"
    active_provider = provider or RuleBasedMeetingIntelligenceProvider()
    warnings = [
        "local_rule_based_extractor",
        "no_external_llm_used",
    ]
    if not rows:
        warnings.append("no_transcript_rows")
    objects = active_provider.extract_objects(
        rows,
        report_id=report_id,
        transcript_revision_id=revision_id,
        generated_at=generated_at,
    )
    summary_text = build_summary_text(objects, rows)
    summary_object = build_summary_object(
        summary_text,
        rows=rows,
        objects=objects,
        report_id=report_id,
        transcript_revision_id=revision_id,
        generated_at=generated_at,
        extractor=active_provider.name,
    )
    objects = [summary_object, *objects]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "session_id": str(session_id or ""),
        "generated_at": generated_at,
        "updated_at": generated_at,
        "transcript_revision_id": revision_id,
        "status": "draft",
        "summary": summary_text,
        "objects": objects,
        "quality": {
            "needs_human_review": True,
            "stale_objects_count": 0,
            "extractor": active_provider.name,
            "local_first": True,
        },
        "warnings": warnings,
    }
    return refresh_report_review_status(report)


def mark_report_stale_if_needed(
    report: dict[str, Any] | None,
    *,
    transcript_rows: Iterable[dict[str, Any]],
    speaker_state: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(report, dict) or not report:
        return report, False
    current_revision = transcript_revision_id(transcript_rows, speaker_state or {})
    if str(report.get("transcript_revision_id") or "") == current_revision:
        return report, False

    updated_at = updated_at or utc_now_iso()
    next_report = copy.deepcopy(report)
    next_report["status"] = "stale"
    next_report["updated_at"] = updated_at
    next_report["current_transcript_revision_id"] = current_revision
    warnings = list(next_report.get("warnings") or [])
    if "transcript_or_speaker_changed_after_generation" not in warnings:
        warnings.append("transcript_or_speaker_changed_after_generation")
    next_report["warnings"] = warnings

    current_rows = {row.row_id: row for row in normalize_transcript_rows(transcript_rows)}
    stale_count = 0
    for obj in next_report.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if _object_evidence_changed(obj, current_rows):
            if str(obj.get("status") or "") != "stale":
                obj["previous_status"] = str(obj.get("status") or "draft")
            obj["status"] = "stale"
            obj["stale_reason"] = "Transcript text, timing, or speaker attribution changed for evidence rows."
            obj["updated_at"] = updated_at
            stale_count += 1
    quality = dict(next_report.get("quality") or {})
    quality["needs_human_review"] = True
    quality["stale_objects_count"] = stale_count
    next_report["quality"] = quality
    return next_report, True


def update_report_object(
    report: dict[str, Any],
    *,
    object_id: str,
    status: str | None = None,
    title: str | None = None,
    body: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        raise ValueError("Meeting intelligence report is missing.")
    object_id = str(object_id or "").strip()
    if not object_id:
        raise ValueError("Meeting intelligence object id is required.")
    clean_status = str(status or "").strip()
    if clean_status and clean_status not in OBJECT_STATUSES:
        raise ValueError(f"Unsupported meeting intelligence status: {clean_status}")

    updated_at = updated_at or utc_now_iso()
    next_report = copy.deepcopy(report)
    for obj in next_report.get("objects") or []:
        if not isinstance(obj, dict) or str(obj.get("id") or "") != object_id:
            continue
        if title is not None:
            obj["title"] = _clean_text(title, limit=180)
        if body is not None:
            obj["body"] = _clean_text(body, limit=1200)
        if clean_status:
            obj["status"] = clean_status
        elif title is not None or body is not None:
            obj["status"] = "edited"
        obj["updated_at"] = updated_at
        next_report["updated_at"] = updated_at
        if obj.get("type") == "summary":
            next_report["summary"] = str(obj.get("body") or next_report.get("summary") or "")
        return refresh_report_review_status(next_report)
    raise ValueError(f"Unknown meeting intelligence object: {object_id}")


def refresh_report_review_status(report: dict[str, Any]) -> dict[str, Any]:
    objects = [obj for obj in (report.get("objects") or []) if isinstance(obj, dict)]
    statuses = {str(obj.get("status") or "draft") for obj in objects}
    if "stale" in statuses or str(report.get("status") or "") == "stale":
        report["status"] = "stale"
    elif objects and statuses.issubset({"accepted", "rejected", "edited"}):
        report["status"] = "reviewed"
    elif statuses.intersection({"accepted", "rejected", "edited"}):
        report["status"] = "partially_reviewed"
    else:
        report["status"] = "draft"
    quality = dict(report.get("quality") or {})
    quality["needs_human_review"] = report["status"] != "reviewed"
    quality["stale_objects_count"] = sum(1 for obj in objects if obj.get("status") == "stale")
    report["quality"] = quality
    return report


class RuleBasedMeetingIntelligenceProvider:
    name = "local_rule_based_v1"

    _decision_pattern = re.compile(
        r"\b(decided|decision|agreed|agreement|final|beschlossen|beschluss|entschieden|entscheidung|machen wir so)\b",
        re.IGNORECASE,
    )
    _action_pattern = re.compile(
        r"\b(action|todo|follow up|i will|i'll|we need to|can you|please|owner|deadline|task|"
        r"aufgabe|bitte|ich mache|ich uebernehme|wir muessen|frist|bis)\b",
        re.IGNORECASE,
    )
    _question_pattern = re.compile(
        r"\?|\b(open question|unclear|clarify|who will|what is|what are|how do|"
        r"offen|unklar|klaeren|frage)\b",
        re.IGNORECASE,
    )
    _risk_pattern = re.compile(
        r"\b(risk|blocker|blocked|blocking|problem|issue|concern|warning|delay|"
        r"risiko|blocker|blockiert|problem|kritisch|verzoegerung)\b",
        re.IGNORECASE,
    )
    _claim_pattern = re.compile(
        r"\b(is|are|has|have|was|were|ist|sind|hat|haben|war|waren)\b.*\b(\d+|percent|prozent|million|millionen|eur|usd)\b",
        re.IGNORECASE,
    )

    def extract_objects(
        self,
        rows: list[TranscriptRow],
        *,
        report_id: str,
        transcript_revision_id: str,
        generated_at: str,
    ) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        per_type_counts = {"decision": 0, "action_item": 0, "open_question": 0, "risk": 0, "claim": 0}
        for row in rows:
            if self._decision_pattern.search(row.text) and per_type_counts["decision"] < 8:
                per_type_counts["decision"] += 1
                objects.append(self._object_from_row(
                    row,
                    report_id=report_id,
                    object_type="decision",
                    type_count=per_type_counts["decision"],
                    transcript_revision_id=transcript_revision_id,
                    generated_at=generated_at,
                    confidence=0.68,
                    confidence_reason="Decision cue found in the transcript row; review is required before treating it as final.",
                    title_prefix="Decision candidate",
                    support_type="direct",
                    extra={"decision_state": "needs_review"},
                ))
            if self._action_pattern.search(row.text) and per_type_counts["action_item"] < 8:
                per_type_counts["action_item"] += 1
                owner = self._owner_from_action(row)
                extra: dict[str, Any] = {
                    "action_state": "needs_review",
                    "owner": owner,
                    "due": self._due_hint(row.text),
                }
                objects.append(self._object_from_row(
                    row,
                    report_id=report_id,
                    object_type="action_item",
                    type_count=per_type_counts["action_item"],
                    transcript_revision_id=transcript_revision_id,
                    generated_at=generated_at,
                    confidence=0.61 if owner else 0.53,
                    confidence_reason="Action cue found; owner and deadline stay reviewable unless explicitly supported.",
                    title_prefix="Action item candidate",
                    support_type="direct",
                    extra=extra,
                ))
            if self._question_pattern.search(row.text) and per_type_counts["open_question"] < 6:
                per_type_counts["open_question"] += 1
                objects.append(self._object_from_row(
                    row,
                    report_id=report_id,
                    object_type="open_question",
                    type_count=per_type_counts["open_question"],
                    transcript_revision_id=transcript_revision_id,
                    generated_at=generated_at,
                    confidence=0.66 if "?" in row.text else 0.52,
                    confidence_reason="Question or uncertainty cue found in the transcript row.",
                    title_prefix="Open question",
                    support_type="direct" if "?" in row.text else "context",
                    extra={"blocking": False},
                ))
            if self._risk_pattern.search(row.text) and per_type_counts["risk"] < 6:
                per_type_counts["risk"] += 1
                objects.append(self._object_from_row(
                    row,
                    report_id=report_id,
                    object_type="risk",
                    type_count=per_type_counts["risk"],
                    transcript_revision_id=transcript_revision_id,
                    generated_at=generated_at,
                    confidence=0.6,
                    confidence_reason="Risk or blocker cue found; impact and ownership require human review.",
                    title_prefix="Risk or blocker",
                    support_type="direct",
                    extra={"impact": "needs_review", "likelihood": "needs_review"},
                ))
            if (
                self._claim_pattern.search(row.text)
                and evidence_matches_transcript(row.text, row.text)
                and per_type_counts["claim"] < 4
            ):
                per_type_counts["claim"] += 1
                objects.append(self._object_from_row(
                    row,
                    report_id=report_id,
                    object_type="claim",
                    type_count=per_type_counts["claim"],
                    transcript_revision_id=transcript_revision_id,
                    generated_at=generated_at,
                    confidence=0.58,
                    confidence_reason="Fact-Lens-compatible factual cue found; no external verification was run.",
                    title_prefix="Checkable claim",
                    support_type="direct",
                    extra={"verification_status": "unverified"},
                ))
        return objects

    def _object_from_row(
        self,
        row: TranscriptRow,
        *,
        report_id: str,
        object_type: str,
        type_count: int,
        transcript_revision_id: str,
        generated_at: str,
        confidence: float,
        confidence_reason: str,
        title_prefix: str,
        support_type: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence = evidence_span(row, support_type=support_type)
        object_id = f"mi_{object_type}_{_stable_hash({'report_id': report_id, 'row_id': row.row_id, 'type': object_type, 'n': type_count}, length=12)}"
        body = row.text
        payload = {
            "id": object_id,
            "type": object_type,
            "title": _title_from_text(body, fallback=f"{title_prefix} {type_count}"),
            "body": body,
            "status": "draft",
            "confidence": round(float(confidence), 2),
            "confidence_reason": confidence_reason,
            "evidence_spans": [evidence],
            "created_at": generated_at,
            "updated_at": generated_at,
            "derived_from": {
                "transcript_revision_id": transcript_revision_id,
                "row_refs": evidence["rows"],
                "extractor": self.name,
            },
        }
        payload.update(extra or {})
        return payload

    @staticmethod
    def _owner_from_action(row: TranscriptRow) -> dict[str, Any] | None:
        lowered = row.text.casefold()
        if re.search(r"\b(i will|i'll|ich mache|ich uebernehme)\b", lowered):
            return {
                "speaker_id": row.speaker_id,
                "speaker_name": row.speaker_name,
                "confidence": 0.74,
                "confidence_reason": "First-person commitment in the speaker's own row.",
            }
        return None

    @staticmethod
    def _due_hint(text: str) -> dict[str, Any] | None:
        match = re.search(r"\b(by|until|bis)\s+([^,.!?;]{2,40})", text, flags=re.IGNORECASE)
        if not match:
            return None
        return {
            "raw_text": match.group(0).strip(),
            "confidence": 0.45,
            "confidence_reason": "Relative or textual due cue detected; not resolved to a calendar date.",
        }


def build_summary_text(objects: list[dict[str, Any]], rows: list[TranscriptRow]) -> str:
    primary = [obj for obj in objects if obj.get("type") in {"decision", "action_item", "risk", "open_question"}]
    if primary:
        snippets = [str(obj.get("body") or "").strip() for obj in primary[:3]]
        return "Draft meeting signals: " + " ".join(snippets)
    if rows:
        speaker_count = len({row.speaker_id or row.speaker_name for row in rows if row.speaker_id or row.speaker_name})
        return (
            f"No explicit decision, action item, open question, or risk candidate was found in "
            f"{len(rows)} transcript row{'s' if len(rows) != 1 else ''}"
            f"{f' across {speaker_count} speaker(s)' if speaker_count else ''}."
        )
    return "No transcript rows are available for meeting intelligence yet."


def build_summary_object(
    summary_text: str,
    *,
    rows: list[TranscriptRow],
    objects: list[dict[str, Any]],
    report_id: str,
    transcript_revision_id: str,
    generated_at: str,
    extractor: str,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for obj in objects:
        for span in obj.get("evidence_spans") or []:
            row_ids = span.get("row_ids") if isinstance(span, dict) else []
            if not row_ids:
                continue
            first_row_id = str(row_ids[0])
            if first_row_id in seen_rows:
                continue
            evidence.append(copy.deepcopy(span))
            seen_rows.add(first_row_id)
            break
        if len(evidence) >= 4:
            break
    if not evidence and rows:
        for row in rows[:2]:
            evidence.append(evidence_span(row, support_type="context"))
    return {
        "id": f"mi_summary_{_stable_hash({'report_id': report_id, 'summary': summary_text}, length=12)}",
        "type": "summary",
        "title": "Executive summary",
        "body": summary_text,
        "status": "draft",
        "confidence": 0.56 if evidence else 0.0,
        "confidence_reason": "Summary is synthesized from extracted draft objects and their transcript evidence.",
        "evidence_spans": evidence,
        "created_at": generated_at,
        "updated_at": generated_at,
        "derived_from": {
            "transcript_revision_id": transcript_revision_id,
            "row_refs": [row_ref for span in evidence for row_ref in span.get("rows", [])],
            "extractor": extractor,
        },
    }


def evidence_span(row: TranscriptRow, *, support_type: str = "direct") -> dict[str, Any]:
    row_ref = {
        "row_id": row.row_id,
        "index": row.index,
        "row_revision_id": row.row_revision_id,
        "speaker_id": row.speaker_id,
        "speaker_name": row.speaker_name,
    }
    return {
        "id": f"ev_{_stable_hash({'row_id': row.row_id, 'revision': row.row_revision_id, 'support_type': support_type}, length=12)}",
        "row_ids": [row.row_id],
        "rows": [row_ref],
        "start": row.start,
        "end": row.end,
        "speaker_id": row.speaker_id,
        "speaker_name": row.speaker_name,
        "quote_excerpt": _clean_text(row.text, limit=260),
        "support_type": support_type if support_type in {"direct", "context", "weak"} else "direct",
    }


def _object_evidence_changed(obj: dict[str, Any], current_rows: dict[str, TranscriptRow]) -> bool:
    spans = obj.get("evidence_spans") if isinstance(obj.get("evidence_spans"), list) else []
    if not spans:
        return False
    checked = False
    for span in spans:
        if not isinstance(span, dict):
            continue
        for ref in span.get("rows") or []:
            if not isinstance(ref, dict):
                continue
            checked = True
            row_id = str(ref.get("row_id") or "")
            current = current_rows.get(row_id)
            if current is None:
                return True
            if str(ref.get("row_revision_id") or "") != current.row_revision_id:
                return True
            if str(ref.get("speaker_id") or "") != current.speaker_id:
                return True
            if str(ref.get("speaker_name") or "") != current.speaker_name:
                return True
    return False if checked else False


def _row_to_source_dict(row: TranscriptRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "index": row.index,
        "start": row.start,
        "end": row.end,
        "text": row.text,
        "assigned_speaker": row.speaker_id,
        "speaker_name": row.speaker_name,
    }


def _stable_hash(value: Any, *, length: int = 12) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _speaker_display_name(speaker_id: str) -> str:
    match = re.fullmatch(r"S(\d+)", str(speaker_id or ""))
    if match:
        return f"Speaker {int(match.group(1))}"
    return str(speaker_id or "Unknown")


def _clean_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _title_from_text(text: str, *, fallback: str) -> str:
    clean = _clean_text(text, limit=100)
    if not clean:
        return fallback
    if len(clean) < 96:
        return clean
    return clean[:96].rstrip(" ,.;:") + "..."
