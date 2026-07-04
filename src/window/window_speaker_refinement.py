"""Prototype-based speaker assignment refinement for the window diarizer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpeakerRefinementConfig:
    max_per_profile: int = 32
    prototype_min_duration: float = 0.15
    prototype_max_unknown: float = 1.0
    top_k: int = 12
    centroid_blend: float = 0.555
    unknown_min_similarity: float = 0.20
    unknown_min_margin: float = 0.0
    known_max_duration: float = 8.0
    known_min_similarity: float = -0.039
    known_min_delta: float = 0.108


@dataclass(frozen=True)
class SpeakerPrototypeRevision:
    index: int
    previous_speaker: str | None
    assigned_speaker: str
    prototype_score: float
    prototype_margin: float
    prototype_delta: float
    prototype_scores: dict[str, float]
    assignment_source: str


def speaker_label(row: dict[str, Any]) -> str | None:
    value = row.get("assigned_speaker")
    return str(value) if value else None


def normalize_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Empty embedding vector.")
    return (vector / norm).astype(np.float32)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_duration(row: dict[str, Any]) -> float:
    value = row.get("duration_seconds", row.get("audio_length_seconds", 0.0))
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _row_embedding(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("embedding")
    if value is None:
        return None
    try:
        return normalize_vector(value)
    except ValueError:
        return None


def build_speaker_prototypes(rows: list[dict[str, Any]], config: SpeakerRefinementConfig) -> dict[str, np.ndarray]:
    candidates: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for row in rows:
        label = speaker_label(row)
        if not label:
            continue
        duration = _row_duration(row)
        if duration < config.prototype_min_duration:
            continue
        unknown_probability = _optional_float(row.get("unknown_probability"))
        if unknown_probability is not None and unknown_probability > config.prototype_max_unknown:
            continue
        embedding = _row_embedding(row)
        if embedding is None:
            continue
        quality = duration
        margin = _optional_float(row.get("margin"))
        if margin is not None:
            quality += max(0.0, margin) * 4.0
        top_similarity = _optional_float(row.get("top_similarity"))
        if top_similarity is not None:
            quality += max(0.0, top_similarity)
        candidates[label].append((quality, embedding))

    prototypes: dict[str, np.ndarray] = {}
    max_per_profile = max(1, int(config.max_per_profile))
    for label, items in candidates.items():
        items.sort(key=lambda item: item[0], reverse=True)
        selected = [embedding for _, embedding in items[:max_per_profile]]
        if selected:
            prototypes[label] = np.stack(selected).astype(np.float32)
    return prototypes


def prototype_score(vector: np.ndarray, prototype_matrix: np.ndarray, top_k: int, centroid_blend: float) -> float:
    similarities = prototype_matrix @ vector
    if similarities.size == 0:
        return -1.0
    top_count = min(max(1, int(top_k)), int(similarities.size))
    top_scores = np.partition(similarities, -top_count)[-top_count:]
    top_mean = float(np.mean(top_scores))
    centroid = normalize_vector(np.mean(prototype_matrix, axis=0))
    centroid_score = float(np.dot(centroid, vector))
    blend = max(0.0, min(1.0, float(centroid_blend)))
    return (1.0 - blend) * top_mean + blend * centroid_score


def _score_against_prototypes(
    embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
    config: SpeakerRefinementConfig,
) -> dict[str, float]:
    vector = normalize_vector(embedding)
    return {
        label: prototype_score(vector, matrix, config.top_k, config.centroid_blend)
        for label, matrix in prototypes.items()
    }


def find_speaker_prototype_revisions(
    rows: list[dict[str, Any]],
    config: SpeakerRefinementConfig,
    allow_known_reassignment: bool = False,
) -> list[SpeakerPrototypeRevision]:
    prototypes = build_speaker_prototypes(rows, config)
    if len(prototypes) <= 1:
        return []

    revisions: list[SpeakerPrototypeRevision] = []
    for row in rows:
        embedding = _row_embedding(row)
        if embedding is None:
            continue
        current = speaker_label(row)
        duration = _row_duration(row)
        if current and not allow_known_reassignment:
            continue
        if current and duration > config.known_max_duration:
            continue

        scores = _score_against_prototypes(embedding, prototypes, config)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - runner_up_score if len(ranked) > 1 else 1.0
        current_score = scores.get(current, -1.0) if current else -1.0
        delta = best_score - current_score
        if best_label == current:
            continue
        if current is None:
            if best_score < config.unknown_min_similarity or margin < config.unknown_min_margin:
                continue
            source = "prototype_unknown_assign"
        else:
            if best_score < config.known_min_similarity or delta < config.known_min_delta:
                continue
            source = "prototype_reassign"

        revisions.append(
            SpeakerPrototypeRevision(
                index=int(row["index"]),
                previous_speaker=current,
                assigned_speaker=best_label,
                prototype_score=round(float(best_score), 4),
                prototype_margin=round(float(margin), 4),
                prototype_delta=round(float(delta), 4),
                prototype_scores={label: round(float(score), 4) for label, score in scores.items()},
                assignment_source=source,
            )
        )
    return revisions
