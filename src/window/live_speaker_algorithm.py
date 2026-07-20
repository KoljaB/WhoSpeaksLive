"""Deterministic causal live-speaker algorithm shared by live and cached replay.

The embedding source is deliberately outside this module.  Production may pass a
fresh embedding while an optimizer passes the vector stored for the same
right-aligned media tick.  Everything after that boundary is identical and uses
media time only; wall-clock scheduling is never part of the decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable

import numpy as np

from speakers.speaker_embedding_cluster import SpeakerMemory, normalize_vector


ALGORITHM_ID = "causal_live_speaker_v2_explicit_probe_release_events"


@dataclass(frozen=True)
class LiveSpeakerAlgorithmConfig:
    min_similarity: float = 0.45
    min_margin: float = 0.05
    min_known_probability: float = 0.50
    ema_count: int = 3
    ema_alpha: float = 0.55
    acquire_count: int = 1
    switch_count: int = 1
    unknown_release_count: int = 2
    silence_release_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "ema_count",
            "acquire_count",
            "switch_count",
            "unknown_release_count",
            "silence_release_count",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= float(self.min_known_probability) <= 1.0:
            raise ValueError("min_known_probability must be in [0, 1]")


@dataclass(frozen=True)
class SpeakerProfileEvent:
    """A complete profile snapshot becoming visible at a causal media time."""

    available_at: float
    speaker_id: str
    centroid: np.ndarray
    speech_seconds: float = 0.0
    sentence_count: int = 1
    generation: int = 0
    sentence_start: float | None = None
    sentence_end: float | None = None

    def __post_init__(self) -> None:
        if float(self.available_at) < 0.0:
            raise ValueError("available_at must be non-negative")
        if not str(self.speaker_id).strip():
            raise ValueError("speaker_id must not be empty")
        if (self.sentence_start is None) != (self.sentence_end is None):
            raise ValueError("sentence_start and sentence_end must be provided together")
        if self.sentence_start is not None:
            if float(self.sentence_start) < 0.0 or float(self.sentence_end) < float(self.sentence_start):
                raise ValueError("invalid sentence bounds")
        object.__setattr__(self, "centroid", normalize_vector(self.centroid))


@dataclass(frozen=True)
class LiveSpeakerStep:
    media_time: float
    speech: bool
    embedding: np.ndarray | None
    duration_seconds: float
    probe_scheduled: bool = True
    release_signal: bool = False
    embedding_latency_seconds: float | None = None
    skipped_reason: str = ""


@dataclass(frozen=True)
class LiveSpeakerDecision:
    media_time: float
    visible_speaker: str | None
    action: str
    reason: str
    candidate_speaker: str | None
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    similarities: dict[str, float]
    profile_count: int
    profile_generations: dict[str, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def trace_record(self) -> dict[str, Any]:
        algorithm_id = (
            self.diagnostics.get("algorithm_id")
            or self.diagnostics.get("multiscale_algorithm_id")
            or ALGORITHM_ID
        )
        return {
            "algorithm_id": str(algorithm_id),
            "media_time": round(float(self.media_time), 6),
            "visible_speaker": self.visible_speaker,
            "action": self.action,
            "reason": self.reason,
            "candidate_speaker": self.candidate_speaker,
            "probabilities": dict(self.probabilities),
            "raw_probabilities": dict(self.raw_probabilities),
            "similarities": dict(self.similarities),
            "profile_count": int(self.profile_count),
            "profile_generations": dict(self.profile_generations),
            "diagnostics": dict(self.diagnostics),
        }


def _speaker_id_from_probability_key(key: str) -> str | None:
    value = str(key or "").strip()
    lower = value.lower()
    if lower.startswith("speaker") and lower[7:].isdigit():
        index = int(lower[7:])
        return f"S{index}" if index > 0 else None
    if value.upper().startswith("S") and value[1:].isdigit():
        return f"S{int(value[1:])}"
    return None


class CausalLiveSpeakerAlgorithm:
    """One chronological state machine for both fresh and cached embeddings."""

    def __init__(
        self,
        config: LiveSpeakerAlgorithmConfig | None = None,
        profile_events: Iterable[SpeakerProfileEvent] = (),
    ) -> None:
        self.config = config or LiveSpeakerAlgorithmConfig()
        self._memory = SpeakerMemory(
            same_speaker_similarity=self.config.min_similarity,
            min_margin=self.config.min_margin,
        )
        self._events = sorted(
            list(profile_events),
            key=lambda item: (float(item.available_at), int(item.generation), str(item.speaker_id)),
        )
        self._next_event = 0
        self._generations: dict[str, int] = {}
        self._profile_snapshots: dict[str, SpeakerProfileEvent] = {}
        self._visible: str | None = None
        self._last_media_time = -1.0
        self._probability_history: deque[dict[str, float]] = deque(maxlen=self.config.ema_count)
        self._pending_candidate: str | None = None
        self._pending_count = 0
        self._unknown_count = 0
        self._silence_count = 0

    @property
    def visible_speaker(self) -> str | None:
        return self._visible

    def sync_profiles(self, profiles: Iterable[dict[str, Any]]) -> list[str]:
        """Synchronize production memory snapshots without resetting temporal state."""

        normalized: list[dict[str, Any]] = []
        changed: list[str] = []
        live_labels: set[str] = set()
        fingerprints = getattr(self, "_profile_fingerprints", None)
        if not isinstance(fingerprints, dict):
            fingerprints = {}
            self._profile_fingerprints = fingerprints
        for raw in profiles:
            label = str(raw.get("label") or "").strip()
            if not label:
                continue
            centroid = normalize_vector(np.asarray(raw["centroid"], dtype=np.float32))
            sentence_count = max(1, int(raw.get("sentence_count") or 1))
            speech_seconds = max(0.0, float(raw.get("speech_seconds") or 0.0))
            fingerprint = hashlib.sha256()
            fingerprint.update(np.ascontiguousarray(centroid).tobytes())
            fingerprint.update(str(sentence_count).encode("ascii"))
            fingerprint.update(repr(speech_seconds).encode("ascii"))
            digest = fingerprint.hexdigest()
            live_labels.add(label)
            if fingerprints.get(label) != digest:
                fingerprints[label] = digest
                self._generations[label] = int(self._generations.get(label, 0)) + 1
                changed.append(label)
            normalized.append({
                "label": label,
                "centroid": centroid,
                "speech_seconds": speech_seconds,
                "sentence_count": sentence_count,
            })
        for stale in set(fingerprints) - live_labels:
            fingerprints.pop(stale, None)
            self._generations.pop(stale, None)
            changed.append(stale)
        self._memory.replace_profiles(normalized)
        return sorted(changed)

    def _apply_profile_events(self, media_time: float) -> list[str]:
        applied: list[str] = []
        while self._next_event < len(self._events):
            event = self._events[self._next_event]
            if float(event.available_at) > media_time + 1e-9:
                break
            previous = self._generations.get(event.speaker_id, -1)
            if int(event.generation) >= previous:
                self._profile_snapshots[event.speaker_id] = event
                self._generations[event.speaker_id] = int(event.generation)
                self._memory.replace_profiles([
                    {
                        "label": profile.speaker_id,
                        "centroid": profile.centroid,
                        "speech_seconds": profile.speech_seconds,
                        "sentence_count": profile.sentence_count,
                    }
                    for profile in sorted(
                        self._profile_snapshots.values(), key=lambda item: item.speaker_id
                    )
                ])
                applied.append(event.speaker_id)
            self._next_event += 1
        return applied

    def _ema(self, raw: dict[str, float]) -> dict[str, float]:
        clean = {str(key): max(0.0, float(value)) for key, value in raw.items()}
        self._probability_history.append(clean)
        keys = sorted({key for item in self._probability_history for key in item})
        ema = {key: float(self._probability_history[0].get(key, 0.0)) for key in keys}
        for item in list(self._probability_history)[1:]:
            for key in keys:
                ema[key] = self.config.ema_alpha * float(item.get(key, 0.0)) + (
                    1.0 - self.config.ema_alpha
                ) * ema[key]
        total = sum(ema.values())
        if total > 0.0:
            ema = {key: value / total for key, value in ema.items()}
        return ema

    def _candidate(
        self,
        probabilities: dict[str, float],
        fallback_speaker: str | None = None,
    ) -> str | None:
        speakers = [
            (_speaker_id_from_probability_key(key), float(value))
            for key, value in probabilities.items()
            if _speaker_id_from_probability_key(key) is not None
        ]
        if not speakers:
            return fallback_speaker
        speaker, probability = max(speakers, key=lambda item: item[1])
        unknown = float(probabilities.get("unknown", 0.0))
        if probability <= unknown or probability < self.config.min_known_probability:
            return fallback_speaker
        return speaker

    def _transition_candidate(self, candidate: str) -> tuple[str, str]:
        if candidate == self._visible:
            self._pending_candidate = None
            self._pending_count = 0
            return "hold", "confirmed"
        if candidate == self._pending_candidate:
            self._pending_count += 1
        else:
            self._pending_candidate = candidate
            self._pending_count = 1
        required = self.config.switch_count if self._visible else self.config.acquire_count
        if self._pending_count < required:
            return "hold" if self._visible else "none", "candidate_debounce"
        action = "switch" if self._visible else "acquire"
        self._visible = candidate
        self._pending_candidate = None
        self._pending_count = 0
        self._probability_history.clear()
        return action, "known_candidate"

    def step(self, item: LiveSpeakerStep) -> LiveSpeakerDecision:
        media_time = float(item.media_time)
        if media_time + 1e-9 < self._last_media_time:
            raise ValueError("Live speaker steps must be processed in media-time order")
        self._last_media_time = media_time
        applied = self._apply_profile_events(media_time)
        raw: dict[str, float] = {"unknown": 1.0}
        probabilities: dict[str, float] = dict(raw)
        similarities: dict[str, float] = {}
        candidate: str | None = None

        if not item.probe_scheduled:
            if item.embedding is not None:
                raise ValueError("A non-probe tick may not carry an embedding")
            if item.release_signal:
                self._pending_candidate = None
                self._pending_count = 0
                self._silence_count += 1
                self._unknown_count = 0
                if self._visible and self._silence_count >= self.config.silence_release_count:
                    self._visible = None
                    self._probability_history.clear()
                    action, reason = "clear", "release_gate"
                else:
                    action, reason = (
                        ("hold", "release_debounce") if self._visible else ("none", "release_gate")
                    )
            else:
                action, reason = (
                    ("hold", "non_probe_tick") if self._visible else ("none", "non_probe_tick")
                )
        elif item.release_signal:
            self._silence_count += 1
            self._unknown_count = 0
            self._pending_candidate = None
            self._pending_count = 0
            self._probability_history.clear()
            if self._visible and self._silence_count >= self.config.silence_release_count:
                self._visible = None
                action, reason = "clear", "release_gate"
            else:
                action, reason = (
                    ("hold", "release_debounce") if self._visible else ("none", "release_gate")
                )
        elif not item.speech:
            self._silence_count = 0
            self._unknown_count = 0
            self._pending_candidate = None
            self._pending_count = 0
            action, reason = (
                ("hold", "probe_gate_silence") if self._visible else ("none", "probe_gate_silence")
            )
        elif item.embedding is None or self._memory.profile_count() <= 0:
            self._silence_count = 0
            self._unknown_count += 1
            if self._visible and self._unknown_count >= self.config.unknown_release_count:
                self._visible = None
                self._probability_history.clear()
                action, reason = "clear", "unknown"
            else:
                action, reason = ("hold", "unknown_debounce") if self._visible else ("none", "unknown")
        else:
            self._silence_count = 0
            decision = self._memory.score_existing(
                item.embedding,
                item.duration_seconds,
                min_similarity=self.config.min_similarity,
                min_margin=self.config.min_margin,
            )
            raw = dict(decision.probabilities)
            similarities = {str(key): float(value) for key, value in decision.similarities.items()}
            probabilities = self._ema(raw)
            candidate = self._candidate(probabilities, decision.assigned_speaker)
            if candidate is None:
                self._unknown_count += 1
                self._pending_candidate = None
                self._pending_count = 0
                if self._visible and self._unknown_count >= self.config.unknown_release_count:
                    self._visible = None
                    self._probability_history.clear()
                    action, reason = "clear", "unknown"
                else:
                    action, reason = ("hold", "unknown_debounce") if self._visible else ("none", "unknown")
            else:
                self._unknown_count = 0
                action, reason = self._transition_candidate(candidate)

        return LiveSpeakerDecision(
            media_time=media_time,
            visible_speaker=self._visible,
            action=action,
            reason=reason,
            candidate_speaker=candidate,
            probabilities={str(key): float(value) for key, value in probabilities.items()},
            raw_probabilities={str(key): float(value) for key, value in raw.items()},
            similarities=similarities,
            profile_count=self._memory.profile_count(),
            profile_generations=dict(self._generations),
            diagnostics={
                "profile_events_applied": applied,
                "unknown_count": self._unknown_count,
                "silence_count": self._silence_count,
                "pending_candidate": self._pending_candidate,
                "pending_count": self._pending_count,
                "embedding_latency_seconds": item.embedding_latency_seconds,
                "skipped_reason": item.skipped_reason,
                "probe_scheduled": bool(item.probe_scheduled),
                "release_signal": bool(item.release_signal),
            },
        )


def compare_decision_traces(
    live: Iterable[LiveSpeakerDecision],
    replay: Iterable[LiveSpeakerDecision],
) -> dict[str, Any]:
    """Strict parity report; vectors may differ, decisions at equal ticks may not."""

    left = list(live)
    right = list(replay)
    count = min(len(left), len(right))
    mismatches: list[dict[str, Any]] = []
    for index in range(count):
        a, b = left[index], right[index]
        fields = {
            "media_time": abs(a.media_time - b.media_time) <= 1e-6,
            "visible_speaker": a.visible_speaker == b.visible_speaker,
            "action": a.action == b.action,
            "reason": a.reason == b.reason,
            "candidate_speaker": a.candidate_speaker == b.candidate_speaker,
        }
        if not all(fields.values()):
            mismatches.append({
                "index": index,
                "live": a.trace_record(),
                "replay": b.trace_record(),
                "matching_fields": fields,
            })
    exact = len(left) == len(right) and not mismatches
    left_algorithm = left[0].trace_record()["algorithm_id"] if left else ALGORITHM_ID
    right_algorithm = right[0].trace_record()["algorithm_id"] if right else ALGORITHM_ID
    return {
        "algorithm_id": (
            left_algorithm
            if left_algorithm == right_algorithm
            else f"{left_algorithm}|{right_algorithm}"
        ),
        "exact_match": exact,
        "live_steps": len(left),
        "replay_steps": len(right),
        "compared_steps": count,
        "mismatch_count": len(mismatches) + abs(len(left) - len(right)),
        "decision_match_ratio": (count - len(mismatches)) / count if count else float(len(left) == len(right)),
        "first_mismatches": mismatches[:20],
    }
