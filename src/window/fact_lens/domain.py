"""Fact Lens domain models and the single mutable state owner."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
import json
import queue
import threading
import time
from typing import Any


def wall_now() -> float:
    return time.time()


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


@dataclass(frozen=True)
class SentenceRevisionToken:
    """Identity of one exact source revision submitted for extraction."""

    sentence_id: str
    source_revision: int


@dataclass(frozen=True)
class ExtractionJob:
    sentence: TranscriptSentence
    token: SentenceRevisionToken


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


class FactLensStore:
    def __init__(self, *, max_sentences: int = 80, max_cards: int = 80) -> None:
        self.max_sentences = max_sentences
        self.max_cards = max_cards
        self.started_at = wall_now()
        self._lock = threading.RLock()
        self._sentences: OrderedDict[str, TranscriptSentence] = OrderedDict()
        self._sentence_revisions: dict[str, int] = {}
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
        self._publisher = SnapshotPublisher(self.snapshot)

    def subscribe(self) -> queue.Queue[str]:
        return self._publisher.subscribe()

    def unsubscribe(self, subscriber: queue.Queue[str]) -> None:
        self._publisher.unsubscribe(subscriber)

    def close(self) -> None:
        self._publisher.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def record_queue_drop(self, job: ExtractionJob | None = None) -> None:
        with self._lock:
            self._stats["queue_drops"] += 1
            if job is not None and self._token_is_current_locked(job.token):
                self._remove_sentence_cards_locked(job.token.sentence_id)
                self._cards[placeholder_card_id(job.token.sentence_id)] = ClaimCard(
                    id=placeholder_card_id(job.token.sentence_id),
                    sentence_id=job.token.sentence_id,
                    speaker=job.sentence.speaker,
                    transcript_start=job.sentence.start,
                    transcript_end=job.sentence.end,
                    transcript_text=job.sentence.text,
                    claim=job.sentence.text,
                    status="needs_context",
                    classification="needs_context",
                    rationale="Claim extraction was dropped because the bounded queue reached capacity.",
                    error="claim extraction queue capacity exceeded",
                )
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

    def record_sentence(
        self,
        sentence: TranscriptSentence,
        *,
        queue_claim_extraction: bool = True,
    ) -> SentenceRevisionToken | None:
        with self._lock:
            previous = self._sentences.get(sentence.id)
            should_queue = previous is None or (
                previous.text,
                previous.speaker,
                previous.start,
                previous.end,
                previous.event_time,
            ) != (
                sentence.text,
                sentence.speaker,
                sentence.start,
                sentence.end,
                sentence.event_time,
            )
            if should_queue:
                self._sentence_revisions[sentence.id] = self._sentence_revisions.get(sentence.id, 0) + 1
                # A replacement invalidates every prior result even when claim
                # extraction is currently disabled.
                self._remove_sentence_cards_locked(sentence.id)
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
            if not should_queue_claim:
                return None
            return SentenceRevisionToken(sentence.id, self._sentence_revisions[sentence.id])

    def current_token(self, sentence_id: str) -> SentenceRevisionToken | None:
        with self._lock:
            revision = self._sentence_revisions.get(str(sentence_id))
            if revision is None:
                return None
            return SentenceRevisionToken(str(sentence_id), revision)

    def mark_checking(self, sentence: TranscriptSentence, token: SentenceRevisionToken | None = None) -> bool:
        with self._lock:
            token = token or self.current_token(sentence.id)
            if token is None or not self._token_is_current_locked(token):
                return False
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
            return True

    def apply_extraction(
        self,
        sentence: TranscriptSentence,
        result: ExtractionResult,
        token: SentenceRevisionToken | None = None,
    ) -> bool:
        with self._lock:
            token = token or self.current_token(sentence.id)
            if token is None or not self._token_is_current_locked(token):
                return False
            self._stats["claims_rejected"] += len(result.rejected_claims)
            self._remove_sentence_cards_locked(sentence.id)

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
            return True

    def apply_extraction_failure(
        self,
        sentence: TranscriptSentence,
        error: str,
        token: SentenceRevisionToken | None = None,
    ) -> bool:
        with self._lock:
            token = token or self.current_token(sentence.id)
            if token is None or not self._token_is_current_locked(token):
                return False
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
            return True

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
            sentence_id, _sentence = self._sentences.popitem(last=False)
            self._sentence_revisions.pop(sentence_id, None)
            self._remove_sentence_cards_locked(sentence_id)

    def _trim_cards_locked(self) -> None:
        while len(self._cards) > self.max_cards:
            self._cards.popitem(last=False)

    def _remove_sentence_cards_locked(self, sentence_id: str) -> None:
        for card_id, card in tuple(self._cards.items()):
            if card.sentence_id == sentence_id:
                self._cards.pop(card_id, None)

    def _token_is_current_locked(self, token: SentenceRevisionToken) -> bool:
        return self._sentence_revisions.get(token.sentence_id) == token.source_revision

    def _touch_locked(self) -> None:
        self._revision += 1
        # The publisher obtains a detached snapshot and serializes only after
        # this store transaction releases its lock.
        self._publisher.notify(self._revision)


class SnapshotPublisher:
    """Latest-value publisher that never serializes while the store is locked."""

    _STOP = object()

    def __init__(self, snapshot_factory: Any) -> None:
        self._snapshot_factory = snapshot_factory
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[str]] = []
        self._notifications: queue.Queue[int | object] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._last_published_revision = -1

    def subscribe(self) -> queue.Queue[str]:
        subscriber: queue.Queue[str] = queue.Queue(maxsize=8)
        initial = self._snapshot_factory()
        initial_payload = json.dumps(initial, ensure_ascii=True)
        with self._lock:
            if self._closed:
                raise RuntimeError("snapshot publisher is closed")
            self._subscribers.append(subscriber)
            # Queue the initial value before the publisher can deliver a newer
            # revision, preserving per-subscriber monotonic ordering.
            subscriber.put_nowait(initial_payload)
            self._start_locked()
        latest = self._snapshot_factory()
        if int(latest.get("revision") or 0) > int(initial.get("revision") or 0):
            self.notify(int(latest["revision"]))
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[str]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def notify(self, revision: int) -> None:
        with self._lock:
            if self._closed or self._thread is None:
                return
        try:
            self._notifications.put_nowait(int(revision))
        except queue.Full:
            try:
                self._notifications.get_nowait()
            except queue.Empty:
                pass
            try:
                self._notifications.put_nowait(int(revision))
            except queue.Full:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        while True:
            try:
                self._notifications.get_nowait()
            except queue.Empty:
                break
        self._notifications.put(self._STOP)
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("snapshot publisher did not stop")

    def _start_locked(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="fact-lens-snapshot-publisher", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            notification = self._notifications.get()
            if notification is self._STOP:
                return
            snapshot = self._snapshot_factory()
            revision = int(snapshot.get("revision") or 0)
            if revision <= self._last_published_revision:
                continue
            payload = json.dumps(snapshot, ensure_ascii=True)
            self._last_published_revision = revision
            with self._lock:
                subscribers = tuple(self._subscribers)
            for subscriber in subscribers:
                try:
                    subscriber.put_nowait(payload)
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        subscriber.put_nowait(payload)
                    except queue.Full:
                        pass


class SidecarState(FactLensStore):
    """Compatibility façade retaining the original public class name."""

__all__ = [
    "ClaimCard",
    "ExtractedClaim",
    "ExtractionJob",
    "ExtractionResult",
    "FactLensStore",
    "SentenceRevisionToken",
    "SidecarState",
    "SnapshotPublisher",
    "TranscriptSentence",
    "claim_card_id",
    "coalesce_sentences",
    "placeholder_card_id",
]
