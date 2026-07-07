"""Sidecar dashboard for live LLM-based claim extraction from WhoSpeaksLive."""

from __future__ import annotations

import argparse
import html
import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


DEFAULT_SOURCE_URL = "http://127.0.0.1:8796"
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8890
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LLM_MODEL = "local"
SCHEMA_VERSION = "claim_triage_v1"
CLAIM_STATUSES = {"queued", "checking", "ignored", "needs_context", "unverified", "supported", "contradicted", "mixed"}
CLASSIFICATIONS = {"ignore", "checkable_claim", "needs_context"}
PRIORITIES = {"low", "medium", "high"}


CLAIM_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "sentence_id", "classification", "claims", "rationale"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "sentence_id": {"type": "string"},
        "classification": {"enum": sorted(CLASSIFICATIONS)},
        "rationale": {"type": "string", "maxLength": 240},
        "claims": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "evidence", "priority", "rationale"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "evidence": {"type": "string", "maxLength": 240},
                    "priority": {"enum": sorted(PRIORITIES)},
                    "rationale": {"type": "string", "maxLength": 240},
                },
            },
        },
    },
}


def wall_now() -> float:
    return time.time()


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE))


def evidence_matches_transcript(evidence: str, transcript: str) -> bool:
    evidence_norm = normalize_text(evidence)
    transcript_norm = normalize_text(transcript)
    if not evidence_norm or not transcript_norm:
        return False
    if evidence_norm in transcript_norm:
        return True

    evidence_tokens = evidence_norm.split()
    transcript_tokens = transcript_norm.split()
    if len(evidence_tokens) < 3 or not transcript_tokens:
        return False

    low = max(1, len(evidence_tokens) - 2)
    high = min(len(transcript_tokens), len(evidence_tokens) + 2)
    evidence_token_set = set(evidence_tokens)
    for size in range(low, high + 1):
        for index in range(0, len(transcript_tokens) - size + 1):
            window_tokens = transcript_tokens[index : index + size]
            window_text = " ".join(window_tokens)
            ratio = SequenceMatcher(None, evidence_norm, window_text).ratio()
            overlap = len(evidence_token_set.intersection(window_tokens)) / len(evidence_token_set)
            if ratio >= 0.82 or overlap >= 0.75:
                return True
    return False


def strip_json_fences(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class ClaimExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ServerSentEvent:
    event: str
    data: str
    event_id: str = ""


def parse_sse_lines(lines: Iterable[str | bytes]) -> Iterator[ServerSentEvent]:
    event_name = "message"
    event_id = ""
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield ServerSentEvent(event_name, "\n".join(data_lines), event_id)
            event_name = "message"
            event_id = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        yield ServerSentEvent(event_name, "\n".join(data_lines), event_id)


def iter_url_sse(url: str, *, timeout: float | None = None) -> Iterator[ServerSentEvent]:
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        yield from parse_sse_lines(response)


@dataclass(frozen=True)
class TranscriptSentence:
    id: str
    text: str
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    event_time: float | None = None
    received_at: float = field(default_factory=wall_now)

    @classmethod
    def from_public_event(cls, event: dict[str, Any]) -> "TranscriptSentence":
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("missing public event payload")
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("empty transcript text")
        sentence_id = str(payload.get("id") or payload.get("index") or event.get("id") or "").strip()
        if not sentence_id:
            sentence_id = str(abs(hash((text, payload.get("start"), payload.get("end")))))
        return cls(
            id=sentence_id,
            text=text,
            speaker=_optional_str(payload.get("speaker") or payload.get("speaker_id")),
            start=_optional_float(payload.get("start")),
            end=_optional_float(payload.get("end")),
            event_time=_optional_float(event.get("time")),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedClaim:
    claim: str
    evidence: str
    priority: str = "medium"
    rationale: str = ""


@dataclass
class ExtractionResult:
    sentence_id: str
    classification: str
    rationale: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    rejected_claims: list[str] = field(default_factory=list)


@dataclass
class ClaimCard:
    id: str
    sentence_id: str
    speaker: str | None
    transcript_start: float | None
    transcript_end: float | None
    transcript_text: str
    claim: str
    evidence: str = ""
    classification: str = "needs_context"
    status: str = "queued"
    verdict: str = "unverified"
    rationale: str = ""
    priority: str = "medium"
    sources: list[dict[str, str]] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=wall_now)
    updated_at: float = field(default_factory=wall_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def placeholder_card_id(sentence_id: str) -> str:
    return f"sentence:{sentence_id}"


def claim_card_id(sentence_id: str, index: int) -> str:
    return f"claim:{sentence_id}:{index}"


def coalesce_sentences(sentences: Iterable[TranscriptSentence]) -> list[TranscriptSentence]:
    by_id: OrderedDict[str, TranscriptSentence] = OrderedDict()
    for sentence in sentences:
        if sentence.id in by_id:
            del by_id[sentence.id]
        by_id[sentence.id] = sentence
    return list(by_id.values())


class SidecarState:
    def __init__(self, *, max_sentences: int = 80, max_cards: int = 80) -> None:
        self.max_sentences = max_sentences
        self.max_cards = max_cards
        self.started_at = wall_now()
        self._lock = threading.RLock()
        self._subscribers: list[queue.Queue[str]] = []
        self._sentences: OrderedDict[str, TranscriptSentence] = OrderedDict()
        self._cards: OrderedDict[str, ClaimCard] = OrderedDict()
        self._revision = 0
        self._source_status = "idle"
        self._llm_status = "idle"
        self._last_error = ""
        self._stats: dict[str, int] = {
            "sentences_seen": 0,
            "sentences_queued": 0,
            "llm_requests": 0,
            "llm_failures": 0,
            "claims_accepted": 0,
            "claims_rejected": 0,
            "queue_drops": 0,
        }

    def subscribe(self) -> queue.Queue[str]:
        subscriber: queue.Queue[str] = queue.Queue(maxsize=8)
        with self._lock:
            self._subscribers.append(subscriber)
            subscriber.put_nowait(json.dumps(self._snapshot_locked(), ensure_ascii=True))
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[str]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def record_queue_drop(self) -> None:
        with self._lock:
            self._stats["queue_drops"] += 1
            self._touch_locked()

    def set_source_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._source_status = status
            if error:
                self._last_error = error
            self._touch_locked()

    def set_llm_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._llm_status = status
            if error:
                self._last_error = error
            self._touch_locked()

    def record_sentence(self, sentence: TranscriptSentence, *, queue_claim_extraction: bool = True) -> bool:
        with self._lock:
            previous = self._sentences.get(sentence.id)
            should_queue = previous is None or previous.text != sentence.text or previous.speaker != sentence.speaker
            if previous is not None:
                del self._sentences[sentence.id]
            self._sentences[sentence.id] = sentence
            self._trim_sentences_locked()
            self._stats["sentences_seen"] += 1
            should_queue_claim = should_queue and queue_claim_extraction
            if should_queue_claim:
                self._stats["sentences_queued"] += 1
                self._cards[placeholder_card_id(sentence.id)] = ClaimCard(
                    id=placeholder_card_id(sentence.id),
                    sentence_id=sentence.id,
                    speaker=sentence.speaker,
                    transcript_start=sentence.start,
                    transcript_end=sentence.end,
                    transcript_text=sentence.text,
                    claim=sentence.text,
                    status="queued",
                    rationale="Waiting for claim extraction.",
                )
                self._trim_cards_locked()
            self._touch_locked()
            return should_queue_claim

    def mark_checking(self, sentence: TranscriptSentence) -> None:
        with self._lock:
            card = self._cards.get(placeholder_card_id(sentence.id))
            if card is None:
                card = ClaimCard(
                    id=placeholder_card_id(sentence.id),
                    sentence_id=sentence.id,
                    speaker=sentence.speaker,
                    transcript_start=sentence.start,
                    transcript_end=sentence.end,
                    transcript_text=sentence.text,
                    claim=sentence.text,
                )
                self._cards[card.id] = card
            card.status = "checking"
            card.updated_at = wall_now()
            card.rationale = "Checking whether this final sentence contains an atomic claim."
            self._llm_status = "checking"
            self._touch_locked()

    def apply_extraction(self, sentence: TranscriptSentence, result: ExtractionResult) -> None:
        with self._lock:
            self._stats["claims_rejected"] += len(result.rejected_claims)
            self._cards.pop(placeholder_card_id(sentence.id), None)

            if not result.claims:
                status = "ignored" if result.classification == "ignore" else "needs_context"
                rationale = result.rationale or ("Ignored by claim triage." if status == "ignored" else "Needs more transcript context.")
                if result.rejected_claims:
                    rationale = f"{rationale} Rejected extraction: {'; '.join(result.rejected_claims[:2])}."
                self._cards[placeholder_card_id(sentence.id)] = ClaimCard(
                    id=placeholder_card_id(sentence.id),
                    sentence_id=sentence.id,
                    speaker=sentence.speaker,
                    transcript_start=sentence.start,
                    transcript_end=sentence.end,
                    transcript_text=sentence.text,
                    claim=sentence.text,
                    evidence=sentence.text if status == "ignored" else "",
                    classification=result.classification,
                    status=status,
                    rationale=rationale,
                )
            else:
                for index, claim in enumerate(result.claims, start=1):
                    self._cards[claim_card_id(sentence.id, index)] = ClaimCard(
                        id=claim_card_id(sentence.id, index),
                        sentence_id=sentence.id,
                        speaker=sentence.speaker,
                        transcript_start=sentence.start,
                        transcript_end=sentence.end,
                        transcript_text=sentence.text,
                        claim=claim.claim,
                        evidence=claim.evidence,
                        classification="checkable_claim",
                        status="unverified",
                        verdict="unverified",
                        priority=claim.priority,
                        rationale=claim.rationale or "Extracted from transcript; source verification has not run.",
                    )
                self._stats["claims_accepted"] += len(result.claims)

            self._llm_status = "idle"
            self._trim_cards_locked()
            self._touch_locked()

    def apply_extraction_failure(self, sentence: TranscriptSentence, error: str) -> None:
        with self._lock:
            self._stats["llm_failures"] += 1
            self._last_error = error
            card = self._cards.get(placeholder_card_id(sentence.id))
            if card is None:
                card = ClaimCard(
                    id=placeholder_card_id(sentence.id),
                    sentence_id=sentence.id,
                    speaker=sentence.speaker,
                    transcript_start=sentence.start,
                    transcript_end=sentence.end,
                    transcript_text=sentence.text,
                    claim=sentence.text,
                )
                self._cards[card.id] = card
            card.status = "needs_context"
            card.classification = "needs_context"
            card.rationale = "LLM extraction failed; no fact verdict was produced."
            card.error = error
            card.updated_at = wall_now()
            self._llm_status = "error"
            self._touch_locked()

    def recent_sentences(self, limit: int) -> list[TranscriptSentence]:
        with self._lock:
            return list(self._sentences.values())[-limit:]

    def record_llm_request(self) -> None:
        with self._lock:
            self._stats["llm_requests"] += 1
            self._touch_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "revision": self._revision,
            "started_at": self.started_at,
            "source_status": self._source_status,
            "llm_status": self._llm_status,
            "last_error": self._last_error,
            "stats": dict(self._stats),
            "sentences": [sentence.to_dict() for sentence in self._sentences.values()],
            "cards": [card.to_dict() for card in self._cards.values()],
        }

    def _trim_sentences_locked(self) -> None:
        while len(self._sentences) > self.max_sentences:
            self._sentences.popitem(last=False)

    def _trim_cards_locked(self) -> None:
        while len(self._cards) > self.max_cards:
            self._cards.popitem(last=False)

    def _touch_locked(self) -> None:
        self._revision += 1
        if not self._subscribers:
            return
        payload = json.dumps(self._snapshot_locked(), ensure_ascii=True)
        for subscriber in list(self._subscribers):
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(payload)
                except queue.Empty:
                    pass


def parse_openai_chat_json(response_data: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClaimExtractionError("missing_chat_content") from exc

    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    if not isinstance(content, str):
        raise ClaimExtractionError("chat_content_not_string")

    try:
        payload = json.loads(strip_json_fences(content))
    except json.JSONDecodeError as exc:
        raise ClaimExtractionError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise ClaimExtractionError("json_not_object")
    return payload


def validate_extraction_payload(payload: dict[str, Any], sentence: TranscriptSentence) -> ExtractionResult:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ClaimExtractionError("invalid_schema_version")
    sentence_id = str(payload.get("sentence_id") or "")
    if sentence_id and sentence_id != sentence.id:
        raise ClaimExtractionError("sentence_id_mismatch")

    classification = str(payload.get("classification") or "")
    if classification not in CLASSIFICATIONS:
        raise ClaimExtractionError("invalid_classification")

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise ClaimExtractionError("claims_not_list")

    accepted: list[ExtractedClaim] = []
    rejected: list[str] = []
    accepted_keys: set[tuple[str, str]] = set()
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, dict):
            rejected.append(f"claim_{index}:not_object")
            continue
        claim_text = str(raw_claim.get("claim") or "").strip()
        evidence = str(raw_claim.get("evidence") or "").strip()
        priority = str(raw_claim.get("priority") or "medium")
        rationale = str(raw_claim.get("rationale") or "").strip()
        if priority not in PRIORITIES:
            priority = "medium"
        if not claim_text:
            rejected.append(f"claim_{index}:empty_claim")
            continue
        if not evidence:
            rejected.append(f"claim_{index}:missing_evidence")
            continue
        if not evidence_matches_transcript(evidence, sentence.text):
            rejected.append(f"claim_{index}:evidence_mismatch")
            continue
        claim_key = (normalize_text(claim_text), normalize_text(evidence))
        if claim_key in accepted_keys:
            rejected.append(f"claim_{index}:duplicate_claim")
            continue
        accepted_keys.add(claim_key)
        accepted.append(ExtractedClaim(claim=claim_text, evidence=evidence, priority=priority, rationale=rationale))

    if classification != "checkable_claim":
        accepted = []
    classification_for_result = classification
    if classification == "checkable_claim" and not accepted:
        classification_for_result = "needs_context"

    return ExtractionResult(
        sentence_id=sentence.id,
        classification=classification_for_result,
        rationale=str(payload.get("rationale") or "").strip(),
        claims=accepted,
        rejected_claims=rejected,
    )


class OpenAIClaimClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model: str = DEFAULT_LLM_MODEL,
        client_name: str = "whospeaks-fact-lens",
        lane: str = "shared",
        timeout: float = 12.0,
        max_tokens: int = 768,
        schema_mode: str = "both",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client_name = client_name
        self.lane = lane
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.schema_mode = schema_mode

    def extract(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        payload = self.build_payload(sentence, context)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-LLM-Client": self.client_name,
                "X-LLM-Lane": self.lane,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return parse_openai_chat_json(response_data)

    def build_payload(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        prompt_context = {
            "sentence": sentence.to_prompt_dict(),
            "recent_final_sentences": [item.to_prompt_dict() for item in context],
            "verdict_policy": "Do not verify or label true/false here. Extract checkable claims only; unresolved claims stay unverified.",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": build_claim_extraction_prompt()},
                {"role": "user", "content": json.dumps(prompt_context, ensure_ascii=True)},
            ],
        }
        if self.schema_mode in {"json_schema", "both"}:
            payload["json_schema"] = CLAIM_EXTRACTION_SCHEMA
        if self.schema_mode in {"response_format", "both"}:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "who_speaks_claim_triage",
                    "strict": True,
                    "schema": CLAIM_EXTRACTION_SCHEMA,
                },
            }
        return payload


class MockClaimClient:
    def extract(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sentence_id": sentence.id,
            "classification": "checkable_claim",
            "rationale": "Synthetic offline extraction.",
            "claims": [
                {
                    "claim": sentence.text,
                    "evidence": sentence.text,
                    "priority": "medium",
                    "rationale": "Mock mode uses the full synthetic sentence as evidence.",
                }
            ],
        }


def build_claim_extraction_prompt() -> str:
    return (
        "You classify one finalized speech transcript sentence for live fact-check preparation. "
        "Return JSON only. Use classification ignore for non-factual remarks, pure opinions, "
        "questions, backchannels, or fragments that should not be checked. Use checkable_claim "
        "for externally verifiable factual statements. Use needs_context when the sentence may "
        "contain a claim but cannot stand alone. Extract only atomic claims that were explicitly "
        "said in the sentence. Copy evidence from the sentence exactly or near-exactly; do not "
        "invent evidence or combine separate claims. Do not fact-check, search, or assign true or "
        "false labels. Return at most three claims and never repeat the same claim. If no atomic "
        "checkable claim is present, return an empty claims array."
    )


class ClaimExtractionWorker:
    def __init__(
        self,
        *,
        state: SidecarState,
        client: Any,
        work_queue: "queue.Queue[TranscriptSentence]",
        stop_event: threading.Event,
        debounce_seconds: float = 0.35,
        context_size: int = 8,
    ) -> None:
        self.state = state
        self.client = client
        self.work_queue = work_queue
        self.stop_event = stop_event
        self.debounce_seconds = debounce_seconds
        self.context_size = context_size
        self.thread = threading.Thread(target=self._run, name="fact-lens-llm-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, sentence: TranscriptSentence) -> None:
        try:
            self.work_queue.put_nowait(sentence)
        except queue.Full:
            try:
                self.work_queue.get_nowait()
            except queue.Empty:
                pass
            self.state.record_queue_drop()
            self.work_queue.put_nowait(sentence)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                first = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            batch = [first]
            if self.debounce_seconds > 0:
                self.stop_event.wait(self.debounce_seconds)
            while True:
                try:
                    batch.append(self.work_queue.get_nowait())
                except queue.Empty:
                    break

            for sentence in coalesce_sentences(batch):
                if self.stop_event.is_set():
                    return
                self._process(sentence)

    def _process(self, sentence: TranscriptSentence) -> None:
        self.state.mark_checking(sentence)
        self.state.record_llm_request()
        context = self.state.recent_sentences(self.context_size)
        try:
            payload = self.client.extract(sentence, context)
            result = validate_extraction_payload(payload, sentence)
        except Exception as exc:
            self.state.apply_extraction_failure(sentence, str(exc))
            return
        self.state.apply_extraction(sentence, result)


def run_whospeaks_reader(
    *,
    source_url: str,
    state: SidecarState,
    worker: ClaimExtractionWorker | None,
    stop_event: threading.Event,
    reconnect_seconds: float,
    timeout: float | None,
) -> None:
    url = f"{source_url.rstrip('/')}/api/events?snapshot=0"
    while not stop_event.is_set():
        try:
            state.set_source_status("connecting")
            request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                state.set_source_status("connected")
                for sse_event in parse_sse_lines(response):
                    if stop_event.is_set():
                        return
                    if sse_event.event != "transcript.final":
                        continue
                    event = json.loads(sse_event.data)
                    if not isinstance(event, dict):
                        continue
                    sentence = TranscriptSentence.from_public_event(event)
                    if state.record_sentence(sentence, queue_claim_extraction=worker is not None) and worker is not None:
                        worker.submit(sentence)
        except Exception as exc:
            state.set_source_status("reconnecting", f"SSE connection failed: {exc}")
            stop_event.wait(reconnect_seconds)


def run_offline_demo(
    *,
    state: SidecarState,
    worker: ClaimExtractionWorker | None,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    examples = [
        ("demo-berlin", "Berlin has more than three million residents.", "Demo Speaker", 0.0, 4.0),
        ("demo-meeting", "The meeting starts at 14:30 in the main conference room.", "Demo Speaker", 4.5, 8.0),
        ("demo-release", "The first fact lens prototype reads transcript final events from WhoSpeaksLive.", "Demo Speaker", 8.5, 13.0),
    ]
    state.set_source_status("offline_demo")
    counter = 0
    while not stop_event.is_set():
        base_id, text, speaker, start, end = examples[counter % len(examples)]
        cycle = counter // len(examples)
        offset = cycle * 15.0
        sentence = TranscriptSentence(
            id=f"{base_id}-{cycle}",
            text=text,
            speaker=speaker,
            start=start + offset,
            end=end + offset,
        )
        if state.record_sentence(sentence, queue_claim_extraction=worker is not None) and worker is not None:
            worker.submit(sentence)
        counter += 1
        stop_event.wait(interval_seconds)


def make_handler(state: SidecarState, *, quiet: bool = False) -> type[BaseHTTPRequestHandler]:
    class FactLensHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_dashboard_html())
            elif parsed.path == "/api/state":
                self._send_json(state.snapshot())
            elif parsed.path == "/api/health":
                snapshot = state.snapshot()
                self._send_json(
                    {
                        "ok": True,
                        "source_status": snapshot["source_status"],
                        "llm_status": snapshot["llm_status"],
                        "revision": snapshot["revision"],
                    }
                )
            elif parsed.path == "/events":
                self._serve_events()
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            if not quiet:
                super().log_message(format, *args)

        def _serve_events(self) -> None:
            subscriber = state.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    try:
                        payload = subscriber.get(timeout=15)
                        message = f"event: state\ndata: {payload}\n\n"
                    except queue.Empty:
                        message = ": heartbeat\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            finally:
                state.unsubscribe(subscriber)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return FactLensHandler


def render_dashboard_html() -> str:
    title = "WhoSpeaksLive Fact Lens"
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #17202a;
      --muted: #5c6670;
      --queued: #6b7280;
      --checking: #2563eb;
      --ok: #14804a;
      --bad: #b42318;
      --warn: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(280px, 0.8fr) minmax(320px, 1.4fr);
      gap: 14px;
      padding: 14px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{
      min-width: 0;
    }}
    .strip {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #fff;
      color: var(--muted);
      white-space: nowrap;
    }}
    .panel-title {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
    }}
    .list {{
      display: grid;
      gap: 8px;
    }}
    .sentence, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }}
    .sentence {{
      color: var(--muted);
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .claim {{
      font-size: 15px;
      font-weight: 600;
      overflow-wrap: anywhere;
    }}
    .evidence, .rationale, .error {{
      margin-top: 6px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .error {{ color: var(--bad); }}
    .card {{
      border-left: 5px solid var(--queued);
    }}
    .status-checking {{ border-left-color: var(--checking); }}
    .status-supported {{ border-left-color: var(--ok); }}
    .status-contradicted {{ border-left-color: var(--bad); }}
    .status-mixed, .status-needs_context {{ border-left-color: var(--warn); }}
    .status-unverified {{ border-left-color: var(--checking); }}
    .status-ignored, .status-queued {{ border-left-color: var(--queued); }}
    .empty {{
      padding: 24px;
      color: var(--muted);
      background: var(--panel);
      border: 1px dashed var(--line);
      border-radius: 8px;
    }}
    a {{ color: var(--checking); }}
    @media (max-width: 820px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .strip {{ justify-content: flex-start; }}
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="strip" id="status-strip"></div>
  </header>
  <main>
    <section>
      <h2 class="panel-title">Recent Transcript</h2>
      <div class="list" id="sentences"></div>
    </section>
    <section>
      <h2 class="panel-title">Claims</h2>
      <div class="list" id="cards"></div>
    </section>
  </main>
  <script>
    const statusStrip = document.getElementById('status-strip');
    const sentencesNode = document.getElementById('sentences');
    const cardsNode = document.getElementById('cards');

    function text(value) {{
      return value === null || value === undefined || value === '' ? 'unknown' : String(value);
    }}

    function timeRange(item) {{
      if (typeof item.start !== 'number' && typeof item.transcript_start !== 'number') return '';
      const start = item.start ?? item.transcript_start;
      const end = item.end ?? item.transcript_end;
      if (typeof end === 'number') return `${{start.toFixed(1)}}-${{end.toFixed(1)}}s`;
      return `${{start.toFixed(1)}}s`;
    }}

    function pill(label, value) {{
      const node = document.createElement('span');
      node.className = 'pill';
      node.textContent = `${{label}}: ${{value}}`;
      return node;
    }}

    function render(state) {{
      statusStrip.replaceChildren(
        pill('source', state.source_status),
        pill('llm', state.llm_status),
        pill('claims', state.stats.claims_accepted),
        pill('rejected', state.stats.claims_rejected)
      );

      const recent = state.sentences.slice(-10).reverse();
      if (!recent.length) {{
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'No final transcript sentences yet.';
        sentencesNode.replaceChildren(empty);
      }} else {{
        sentencesNode.replaceChildren(...recent.map(renderSentence));
      }}

      const cards = state.cards.slice().reverse();
      if (!cards.length) {{
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'No claim cards yet.';
        cardsNode.replaceChildren(empty);
      }} else {{
        cardsNode.replaceChildren(...cards.map(renderCard));
      }}
    }}

    function renderSentence(sentence) {{
      const node = document.createElement('article');
      node.className = 'sentence';
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = `${{text(sentence.speaker)}}  ${{timeRange(sentence)}}`;
      const body = document.createElement('div');
      body.textContent = sentence.text;
      node.append(meta, body);
      return node;
    }}

    function renderCard(card) {{
      const node = document.createElement('article');
      node.className = `card status-${{card.status}}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = `${{card.status}}  ${{text(card.speaker)}}  ${{timeRange(card)}}  ${{card.priority}}`;
      const claim = document.createElement('div');
      claim.className = 'claim';
      claim.textContent = card.claim;
      node.append(meta, claim);
      if (card.evidence) {{
        const evidence = document.createElement('div');
        evidence.className = 'evidence';
        evidence.textContent = `Evidence: ${{card.evidence}}`;
        node.append(evidence);
      }}
      if (card.rationale) {{
        const rationale = document.createElement('div');
        rationale.className = 'rationale';
        rationale.textContent = card.rationale;
        node.append(rationale);
      }}
      if (card.error) {{
        const error = document.createElement('div');
        error.className = 'error';
        error.textContent = card.error;
        node.append(error);
      }}
      for (const source of card.sources || []) {{
        const link = document.createElement('a');
        link.href = source.url;
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.textContent = source.title || source.url;
        node.append(link);
      }}
      return node;
    }}

    fetch('/api/state').then(response => response.json()).then(render);
    const stream = new EventSource('/events');
    stream.addEventListener('state', event => render(JSON.parse(event.data)));
  </script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live claim extraction sidecar for WhoSpeaksLive transcript.final events.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM claim extraction. Disabled by default; without this, the sidecar only displays final transcript sentences.",
    )
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-client", default="whospeaks-fact-lens")
    parser.add_argument("--llm-lane", default="shared")
    parser.add_argument("--llm-timeout", type=float, default=12.0)
    parser.add_argument("--llm-max-tokens", type=int, default=768)
    parser.add_argument("--schema-mode", choices=["json_schema", "response_format", "both"], default="both")
    parser.add_argument("--debounce-seconds", type=float, default=0.35)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--max-sentences", type=int, default=80)
    parser.add_argument("--max-cards", type=int, default=80)
    parser.add_argument("--sse-timeout", type=float, default=30.0)
    parser.add_argument("--reconnect-seconds", type=float, default=2.0)
    parser.add_argument("--offline-demo", action="store_true")
    parser.add_argument("--offline-interval-seconds", type=float, default=2.0)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stop_event = threading.Event()
    state = SidecarState(max_sentences=args.max_sentences, max_cards=args.max_cards)
    worker: ClaimExtractionWorker | None = None
    extraction_enabled = bool(args.enable_llm or args.mock_llm)
    if extraction_enabled:
        work_queue: queue.Queue[TranscriptSentence] = queue.Queue(maxsize=args.queue_size)
        client = (
            MockClaimClient()
            if args.mock_llm
            else OpenAIClaimClient(
                base_url=args.llm_base_url,
                model=args.llm_model,
                client_name=args.llm_client,
                lane=args.llm_lane,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                schema_mode=args.schema_mode,
            )
        )
        worker = ClaimExtractionWorker(
            state=state,
            client=client,
            work_queue=work_queue,
            stop_event=stop_event,
            debounce_seconds=args.debounce_seconds,
            context_size=args.context_size,
        )
        worker.start()
    else:
        state.set_llm_status("disabled")

    if args.offline_demo:
        reader = threading.Thread(
            target=run_offline_demo,
            kwargs={
                "state": state,
                "worker": worker,
                "stop_event": stop_event,
                "interval_seconds": args.offline_interval_seconds,
            },
            name="fact-lens-offline-demo",
            daemon=True,
        )
    else:
        reader = threading.Thread(
            target=run_whospeaks_reader,
            kwargs={
                "source_url": args.source_url,
                "state": state,
                "worker": worker,
                "stop_event": stop_event,
                "reconnect_seconds": args.reconnect_seconds,
                "timeout": args.sse_timeout,
            },
            name="fact-lens-whospeaks-reader",
            daemon=True,
        )
    reader.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, quiet=args.quiet))
    print(f"Fact Lens dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
