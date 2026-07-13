"""Pure planning boundary for speaker-assignment refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from window.diarization_session import SessionVersion
from window.window_speaker_refinement import (
    SpeakerRefinementConfig,
    find_speaker_prototype_revisions,
)


@dataclass(frozen=True)
class AssignmentRequest:
    records: tuple[dict[str, Any], ...]
    config: SpeakerRefinementConfig
    allow_known_reassignment: bool
    expected_version: SessionVersion


@dataclass(frozen=True)
class AssignmentEffects:
    revisions: tuple[Any, ...]
    expected_version: SessionVersion


class SpeakerAssignmentEngine:
    """Plan revisions without owning controller state or emitting events."""

    def plan_refinement(self, request: AssignmentRequest) -> AssignmentEffects:
        revisions: Sequence[Any] = find_speaker_prototype_revisions(
            list(request.records),
            request.config,
            allow_known_reassignment=request.allow_known_reassignment,
        )
        return AssignmentEffects(tuple(revisions), request.expected_version)
