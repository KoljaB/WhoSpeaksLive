"""Stable public event normalization for window diarization automation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import time


UNKNOWN_SPEAKER = "UNKNOWN"


def _speaker_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.upper() == UNKNOWN_SPEAKER:
        return None
    return text


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sentence_id(payload: dict[str, Any]) -> str:
    value = payload.get("index")
    index = _optional_int(value)
    return str(index) if index is not None else str(value or "")


def _speaker_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    speakers = payload.get("speakers")
    if not isinstance(speakers, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in speakers:
        if not isinstance(item, dict):
            continue
        speaker_id = str(item.get("id") or "").strip()
        if not speaker_id:
            continue
        by_id[speaker_id] = dict(item)
    return by_id


def _speaker_public_payload(speaker: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker_id": str(speaker.get("id") or ""),
        "name": str(speaker.get("name") or ""),
        "display_name": str(speaker.get("display_name") or speaker.get("id") or ""),
        "source": str(speaker.get("source") or ""),
        "locked": bool(speaker.get("locked")),
        "sentence_count": int(speaker.get("sentence_count") or 0),
        "speech_seconds": _optional_float(speaker.get("speech_seconds")) or 0.0,
        "reference_audio": str(speaker.get("reference_audio") or ""),
        "raw": dict(speaker),
    }


class PublicEventNormalizer:
    """Convert raw GUI bus events into stable automation events.

    The raw stream is optimized for the browser and may send the same sentence
    row more than once as speaker evidence improves. This normalizer keeps the
    latest row state by sentence index so consumers can subscribe to semantic
    events such as ``transcript.final_unknown`` or
    ``transcript.speaker_revised``.
    """

    schema = "whospeaks.events.v1"

    def __init__(self, session_id: str = "") -> None:
        self.session_id = str(session_id or "")
        self._sequence = 0
        self._sentences: dict[str, dict[str, Any]] = {}
        self._speakers: dict[str, dict[str, Any]] | None = None
        self._last_live_speaker: str | None = None

    def normalize(self, raw_event: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        event = str(raw_event or "")
        if event == "sentence":
            return self._normalize_sentence(payload)
        if event == "speakers":
            return self._normalize_speakers(payload)
        if event == "live_speaker":
            return self._normalize_live_speaker(payload)
        if event == "status":
            return [self._envelope("system.status", {"message": str(payload.get("message") or ""), "raw": dict(payload)}, event)]
        if event == "error":
            return [self._envelope("system.error", {"error": str(payload.get("error") or payload.get("message") or ""), "raw": dict(payload)}, event)]
        if event == "done":
            return [self._envelope("session.stopped", {"message": str(payload.get("message") or ""), "raw": dict(payload)}, event)]
        return []

    def speaker_snapshot(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        speakers = _speaker_map(payload)
        self._speakers = speakers
        return [
            self._envelope(
                "speaker.snapshot",
                {
                    "group_name": str(payload.get("group_name") or ""),
                    "embedding_provider": str(payload.get("embedding_provider") or ""),
                    "speakers": [_speaker_public_payload(item) for item in speakers.values()],
                    "raw": dict(payload),
                },
                "speakers",
            )
        ]

    def _normalize_sentence(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        sentence = self._sentence_public_payload(payload)
        sentence_id = str(sentence["id"])
        previous = self._sentences.get(sentence_id)
        self._sentences[sentence_id] = sentence

        if sentence["pending"]:
            return [self._envelope("transcript.pending", sentence, "sentence")]

        events: list[dict[str, Any]] = []
        speaker = sentence["speaker"]
        previous_speaker = None if previous is None else previous.get("speaker")
        previous_was_final = bool(previous) and not bool(previous.get("pending"))

        if not previous_was_final:
            events.append(self._envelope("transcript.final", sentence, "sentence"))
            if speaker is None:
                events.append(self._envelope("transcript.final_unknown", sentence, "sentence"))
            else:
                events.append(self._envelope("transcript.speaker_assigned", sentence, "sentence"))

        if previous_was_final and previous_speaker != speaker:
            revision_payload = {
                **sentence,
                "previous_speaker": previous_speaker,
                "new_speaker": speaker,
            }
            events.append(self._envelope("transcript.speaker_revised", revision_payload, "sentence"))
            if speaker is None:
                events.append(self._envelope("transcript.speaker_cleared", revision_payload, "sentence"))
            elif previous_speaker is None:
                events.append(self._envelope("transcript.speaker_assigned", revision_payload, "sentence"))
        elif previous_was_final and sentence["revision"]:
            events.append(self._envelope("transcript.updated", sentence, "sentence"))

        return events

    def _sentence_public_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        speaker = _speaker_or_none(payload.get("assigned_speaker"))
        start = _optional_float(payload.get("start"))
        end = _optional_float(payload.get("end"))
        duration = _optional_float(payload.get("audio_length_seconds"))
        if duration is None and start is not None and end is not None:
            duration = max(0.0, end - start)
        return {
            "id": _sentence_id(payload),
            "index": _optional_int(payload.get("index")),
            "text": str(payload.get("text") or ""),
            "speaker": speaker,
            "speaker_id": speaker,
            "pending": bool(payload.get("pending")),
            "revision": bool(payload.get("revision")),
            "revision_from": _speaker_or_none(payload.get("revision_from")),
            "revision_to": _speaker_or_none(payload.get("revision_to")),
            "start": start,
            "end": end,
            "duration_seconds": duration,
            "created_speaker": bool(payload.get("created_speaker")),
            "unknown_probability": _optional_float(payload.get("unknown_probability")),
            "top_similarity": _optional_float(payload.get("top_similarity")),
            "margin": _optional_float(payload.get("margin")),
            "assignment_source": str(payload.get("assignment_source") or ""),
            "raw": dict(payload),
        }

    def _normalize_speakers(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        current = _speaker_map(payload)
        previous = self._speakers
        self._speakers = current

        state_payload = {
            "group_name": str(payload.get("group_name") or ""),
            "embedding_provider": str(payload.get("embedding_provider") or ""),
            "speakers": [_speaker_public_payload(item) for item in current.values()],
            "raw": dict(payload),
        }
        events: list[dict[str, Any]] = []

        if previous is None:
            for speaker in current.values():
                events.append(self._envelope("speaker.created", _speaker_public_payload(speaker), "speakers"))
            events.append(self._envelope("speaker.state_changed", state_payload, "speakers"))
            return events

        for speaker_id in sorted(set(current) - set(previous)):
            events.append(self._envelope("speaker.created", _speaker_public_payload(current[speaker_id]), "speakers"))
        for speaker_id in sorted(set(previous) - set(current)):
            events.append(self._envelope("speaker.removed", _speaker_public_payload(previous[speaker_id]), "speakers"))
        for speaker_id in sorted(set(current) & set(previous)):
            old = previous[speaker_id]
            new = current[speaker_id]
            if str(old.get("name") or "") != str(new.get("name") or ""):
                events.append(
                    self._envelope(
                        "speaker.renamed",
                        {
                            **_speaker_public_payload(new),
                            "previous_name": str(old.get("name") or ""),
                            "new_name": str(new.get("name") or ""),
                        },
                        "speakers",
                    )
                )
            elif old != new:
                events.append(self._envelope("speaker.updated", _speaker_public_payload(new), "speakers"))

        if previous != current:
            events.append(self._envelope("speaker.state_changed", state_payload, "speakers"))
        return events

    def _normalize_live_speaker(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        speaker = _speaker_or_none(payload.get("speaker_id") or payload.get("assigned_speaker"))
        public_payload = {
            "speaker": speaker,
            "speaker_id": speaker,
            "text": str(payload.get("text") or ""),
            "start": _optional_float(payload.get("start")),
            "end": _optional_float(payload.get("end")),
            "assignment_source": str(payload.get("assignment_source") or ""),
            "raw": dict(payload),
        }
        if speaker == self._last_live_speaker:
            return [self._envelope("live_speaker.updated", public_payload, "live_speaker")]
        previous = self._last_live_speaker
        self._last_live_speaker = speaker
        return [
            self._envelope(
                "live_speaker.changed",
                {
                    **public_payload,
                    "previous_speaker": previous,
                    "new_speaker": speaker,
                },
                "live_speaker",
            )
        ]

    def _envelope(self, event_type: str, payload: dict[str, Any], source_event: str) -> dict[str, Any]:
        self._sequence += 1
        return {
            "id": self._sequence,
            "time": time.time(),
            "schema": self.schema,
            "type": event_type,
            "source_event": source_event,
            "session_id": self.session_id,
            "payload": payload,
        }


def normalize_public_events(
    raw_events: Iterable[tuple[str, dict[str, Any]]],
    *,
    session_id: str = "",
) -> list[dict[str, Any]]:
    normalizer = PublicEventNormalizer(session_id=session_id)
    normalized: list[dict[str, Any]] = []
    for event, payload in raw_events:
        normalized.extend(normalizer.normalize(event, payload))
    return normalized
