"""Immutable records shared by offline validation workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from speakers.realtime_speaker_memory import SpeakerDecision


@dataclass(frozen=True)
class ValidationDecision:
    """Detached, read-only view of a mutable live-memory decision."""

    assigned_speaker: str | None
    created_speaker: bool
    probabilities: Mapping[str, float]
    similarities: Mapping[str, float]
    unknown_probability: float
    top_similarity: float | None
    margin: float | None
    quality: float
    assignment_source: str

    @classmethod
    def from_decision(
        cls,
        decision: SpeakerDecision | "ValidationDecision",
    ) -> "ValidationDecision":
        if isinstance(decision, cls):
            return decision
        return cls(
            assigned_speaker=decision.assigned_speaker,
            created_speaker=bool(decision.created_speaker),
            probabilities=MappingProxyType(dict(decision.probabilities)),
            similarities=MappingProxyType(dict(decision.similarities)),
            unknown_probability=float(decision.unknown_probability),
            top_similarity=decision.top_similarity,
            margin=decision.margin,
            quality=float(decision.quality),
            assignment_source=str(decision.assignment_source),
        )


@dataclass(frozen=True)
class ValidationItem:
    """One validation input plus its current immutable speaker decision."""

    session_id: str
    index: int
    text: str
    duration_seconds: float
    embedding: np.ndarray
    decision: ValidationDecision | SpeakerDecision
    row_fields: Mapping[str, Any]
    reassigned: bool = False

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        embedding.flags.writeable = False
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(
            self,
            "decision",
            ValidationDecision.from_decision(self.decision),
        )
        object.__setattr__(
            self,
            "row_fields",
            MappingProxyType(deepcopy(dict(self.row_fields))),
        )

    def with_decision(self, decision: SpeakerDecision) -> "ValidationItem":
        return replace(self, decision=decision, reassigned=True)

    def to_row(self) -> dict[str, Any]:
        decision = self.decision
        return {
            **deepcopy(dict(self.row_fields)),
            "assigned_speaker": decision.assigned_speaker,
            "created_speaker": decision.created_speaker,
            "reassigned": self.reassigned,
            "probabilities": dict(decision.probabilities),
            "similarities": dict(decision.similarities),
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
        }
