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
    known_min_delta: float = 0.04


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


@dataclass(frozen=True)
class DelayedClusteringConfig:
    """Evidence gates for splitting a polluted online profile after more rows arrive."""

    core_max_unknown: float = 0.50
    core_min_duration: float = 0.80
    min_core_rows: int = 4
    min_core_duration: float = 8.0
    candidate_min_unknown: float = 0.50
    candidate_min_duration: float = 0.35
    candidate_max_core_similarity: float = 0.45
    candidate_min_similarity: float = 0.20
    candidate_min_gain: float = 0.02
    min_candidate_rows: int = 4
    min_candidate_duration: float = 8.0
    min_candidate_span: float = 12.0
    min_candidate_time_groups: int = 2
    time_group_gap: float = 8.0
    min_average_gain: float = 0.22
    min_leave_one_out_similarity: float = 0.16
    max_core_centroid_similarity: float = 0.58
    max_iterations: int = 8


@dataclass(frozen=True)
class DelayedSpeakerSplit:
    previous_speaker: str
    indexes: tuple[int, ...]
    centroid: np.ndarray
    speech_seconds: float
    average_gain: float
    leave_one_out_similarity: float
    core_similarity: float
    time_groups: int


def speaker_label(row: dict[str, Any]) -> str | None:
    value = row.get("assigned_speaker")
    return str(value) if value else None


def _clean_rejected_speaker_label(value: Any) -> str:
    label = str(value or "").strip()
    return "" if not label or label.upper() == "UNKNOWN" else label


def rejected_speaker_labels(row: dict[str, Any]) -> set[str]:
    correction = row.get("correction")
    if not isinstance(correction, dict):
        return set()

    rejected: set[str] = set()
    values = correction.get("rejected_speakers")
    if isinstance(values, str):
        label = _clean_rejected_speaker_label(values)
        if label:
            rejected.add(label)
    elif isinstance(values, (list, tuple, set)):
        for value in values:
            label = _clean_rejected_speaker_label(value)
            if label:
                rejected.add(label)

    corrected = _clean_rejected_speaker_label(correction.get("corrected_speaker"))
    if correction.get("status") == "user_corrected":
        for key in ("previous_speaker", "original_speaker"):
            label = _clean_rejected_speaker_label(correction.get(key))
            if label and label != corrected:
                rejected.add(label)

    if corrected:
        rejected.discard(corrected)
    return rejected


def user_confirmed_speaker_label(row: dict[str, Any]) -> str | None:
    correction = row.get("correction")
    if not isinstance(correction, dict) or correction.get("status") != "user_confirmed":
        return None
    current = speaker_label(row)
    if not current:
        return None
    corrected = _clean_rejected_speaker_label(correction.get("corrected_speaker"))
    if corrected and corrected != current:
        return None
    return current


def user_deleted_speaker_label(row: dict[str, Any]) -> str | None:
    correction = row.get("correction")
    if not isinstance(correction, dict) or correction.get("status") != "speaker_deleted":
        return None
    label = _clean_rejected_speaker_label(correction.get("deleted_speaker"))
    return label or None


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
        if user_confirmed_speaker_label(row) == label:
            quality += 1_000_000.0
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
        if current is None and user_deleted_speaker_label(row):
            continue
        if current and user_confirmed_speaker_label(row) == current:
            continue
        if current and not allow_known_reassignment:
            continue
        if current and duration > config.known_max_duration:
            continue

        scores = _score_against_prototypes(embedding, prototypes, config)
        rejected = rejected_speaker_labels(row)
        if rejected:
            scores = {
                label: score
                for label, score in scores.items()
                if label not in rejected
            }
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


def _weighted_centroid(rows: list[dict[str, Any]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for row in rows:
        embedding = _row_embedding(row)
        if embedding is None:
            continue
        duration = _row_duration(row)
        vectors.append(embedding)
        weights.append(float(np.sqrt(max(0.20, min(2.25, duration)))))
    if not vectors:
        raise ValueError("Cannot build a centroid without embeddings.")
    return normalize_vector(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)))


def _trusted_core_row(row: dict[str, Any], config: DelayedClusteringConfig) -> bool:
    source = str(row.get("assignment_source") or "").strip().lower()
    if source not in {"embedding", "section_gap_new_speaker"}:
        return False
    unknown = _optional_float(row.get("unknown_probability"))
    return (
        _row_duration(row) >= config.core_min_duration
        and (unknown is None or unknown <= config.core_max_unknown)
        and _row_embedding(row) is not None
    )


def _uncertain_row(row: dict[str, Any], label: str, config: DelayedClusteringConfig) -> bool:
    if speaker_label(row) != label:
        return False
    if user_confirmed_speaker_label(row) or rejected_speaker_labels(row):
        return False
    if _row_duration(row) < config.candidate_min_duration or _row_embedding(row) is None:
        return False
    source = str(row.get("assignment_source") or "").strip().lower()
    unknown = _optional_float(row.get("unknown_probability"))
    return source not in {"embedding", "section_gap_new_speaker"} or (
        unknown is not None and unknown >= config.candidate_min_unknown
    )


def _time_group_count(rows: list[dict[str, Any]], gap: float) -> int:
    ordered = sorted(rows, key=lambda row: float((row.get("base_payload") or row).get("start") or 0.0))
    groups = 0
    previous_end: float | None = None
    for row in ordered:
        payload = row.get("base_payload") or row
        start = float(payload.get("start") or 0.0)
        end = float(payload.get("end") or start)
        if previous_end is None or start - previous_end > gap:
            groups += 1
        previous_end = max(end, previous_end if previous_end is not None else end)
    return groups


def _leave_one_out_similarity(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return -1.0
    scores: list[float] = []
    for position, row in enumerate(rows):
        others = rows[:position] + rows[position + 1 :]
        vector = _row_embedding(row)
        if vector is not None and others:
            scores.append(float(np.dot(vector, _weighted_centroid(others))))
    return float(np.mean(scores)) if scores else -1.0


def find_delayed_speaker_splits(
    rows: list[dict[str, Any]],
    config: DelayedClusteringConfig,
) -> list[DelayedSpeakerSplit]:
    """Find live-safe profile splits using only rows observed so far.

    A split is proposed only for a speaker with a sizeable trusted core and a
    repeated, temporally separated set of uncertain rows that fits a second
    centroid materially better than every trusted speaker core.
    """

    labels = sorted({label for row in rows if (label := speaker_label(row))})
    cores: dict[str, list[dict[str, Any]]] = {}
    core_centroids: dict[str, np.ndarray] = {}
    for label in labels:
        core = [row for row in rows if speaker_label(row) == label and _trusted_core_row(row, config)]
        if len(core) < config.min_core_rows:
            continue
        if sum(_row_duration(row) for row in core) < config.min_core_duration:
            continue
        cores[label] = core
        core_centroids[label] = _weighted_centroid(core)

    if not core_centroids:
        return []

    splits: list[DelayedSpeakerSplit] = []
    for label, core in cores.items():
        source_centroid = core_centroids[label]
        pool = [row for row in rows if _uncertain_row(row, label, config)]
        seed = [
            row
            for row in pool
            if float(np.dot(_row_embedding(row), source_centroid))
            <= config.candidate_max_core_similarity
        ]
        if len(seed) < config.min_candidate_rows:
            continue

        candidate = list(seed)
        for _ in range(max(1, int(config.max_iterations))):
            centroid = _weighted_centroid(candidate)
            revised: list[dict[str, Any]] = []
            for row in pool:
                vector = _row_embedding(row)
                if vector is None:
                    continue
                candidate_similarity = float(np.dot(vector, centroid))
                best_core_similarity = max(
                    float(np.dot(vector, stable_centroid))
                    for stable_centroid in core_centroids.values()
                )
                if (
                    candidate_similarity >= config.candidate_min_similarity
                    and candidate_similarity - best_core_similarity >= config.candidate_min_gain
                ):
                    revised.append(row)
            if {int(row["index"]) for row in revised} == {int(row["index"]) for row in candidate}:
                break
            candidate = revised
            if len(candidate) < config.min_candidate_rows:
                break

        if len(candidate) < config.min_candidate_rows:
            continue
        speech_seconds = sum(_row_duration(row) for row in candidate)
        if speech_seconds < config.min_candidate_duration:
            continue
        starts = [float((row.get("base_payload") or row).get("start") or 0.0) for row in candidate]
        ends = [float((row.get("base_payload") or row).get("end") or start) for row, start in zip(candidate, starts)]
        if max(ends) - min(starts) < config.min_candidate_span:
            continue
        time_groups = _time_group_count(candidate, config.time_group_gap)
        if time_groups < config.min_candidate_time_groups:
            continue

        centroid = _weighted_centroid(candidate)
        core_similarity = max(
            float(np.dot(centroid, stable_centroid))
            for stable_centroid in core_centroids.values()
        )
        if core_similarity > config.max_core_centroid_similarity:
            continue
        gains = []
        for row in candidate:
            vector = _row_embedding(row)
            if vector is None:
                continue
            best_core = max(float(np.dot(vector, item)) for item in core_centroids.values())
            gains.append(float(np.dot(vector, centroid)) - best_core)
        average_gain = float(np.mean(gains)) if gains else -1.0
        if average_gain < config.min_average_gain:
            continue
        leave_one_out = _leave_one_out_similarity(candidate)
        if leave_one_out < config.min_leave_one_out_similarity:
            continue

        splits.append(
            DelayedSpeakerSplit(
                previous_speaker=label,
                indexes=tuple(sorted(int(row["index"]) for row in candidate)),
                centroid=centroid,
                speech_seconds=float(speech_seconds),
                average_gain=average_gain,
                leave_one_out_similarity=leave_one_out,
                core_similarity=core_similarity,
                time_groups=time_groups,
            )
        )
    return splits
