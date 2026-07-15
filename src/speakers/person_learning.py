"""Robust, confirmation-gated evidence aggregation for one meeting speaker."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from speakers.speaker_embedding_cluster import cosine_similarity, normalize_vector


@dataclass(frozen=True)
class PersonLearningPolicy:
    min_sample_seconds: float = 0.8
    min_sample_quality: float = 0.45
    min_sentences: int = 3
    min_speech_seconds: float = 6.0
    overlap_tolerance_seconds: float = 0.12
    seed_similarity: float = 0.41
    competing_speaker_margin: float = 0.04
    max_unknown_probability: float = 0.55
    min_speech_audio_ratio: float = 0.0
    min_cohesion: float = 0.50


@dataclass(frozen=True)
class PersonLearningCandidate:
    centroid: np.ndarray
    sentence_count: int
    speech_seconds: float
    cohesion: float
    seed_similarity: float
    outlier_count: int
    record_indexes: frozenset[int]
    user_trusted_indexes: frozenset[int]


def _record_interval(record: Mapping[str, Any]) -> tuple[float, float] | None:
    base = record.get("base_payload") if isinstance(record.get("base_payload"), dict) else {}
    try:
        start = float(base.get("start"))
        end = float(base.get("end"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def _user_trusted(record: Mapping[str, Any], speaker_id: str) -> bool:
    correction = record.get("correction") if isinstance(record.get("correction"), dict) else {}
    status = str(correction.get("status") or "")
    corrected = str(correction.get("corrected_speaker") or record.get("assigned_speaker") or "")
    return status in {"user_corrected", "user_confirmed"} and corrected == speaker_id


def _overlaps(
    interval: tuple[float, float] | None,
    others: Iterable[tuple[float, float]],
    tolerance_seconds: float,
) -> bool:
    if interval is None:
        return False
    start, end = interval
    return any(
        min(end, other_end) - max(start, other_start) > tolerance_seconds
        for other_start, other_end in others
    )


def _weighted_centroid(
    samples: list[tuple[np.ndarray, float, float, int, bool]],
) -> np.ndarray:
    weighted = np.zeros_like(samples[0][0], dtype=np.float64)
    total_weight = 0.0
    for embedding, duration, quality, _index, _trusted in samples:
        # Very long turns contain useful evidence, but no single turn should
        # dominate a person's meeting template.
        weight = min(6.0, max(0.1, duration)) * (0.5 + 0.5 * quality)
        weighted += embedding.astype(np.float64) * weight
        total_weight += weight
    return normalize_vector(weighted / max(total_weight, 0.0001))


def build_person_learning_candidate(
    records: Iterable[Mapping[str, Any]],
    speaker_profiles: Mapping[str, Any],
    *,
    speaker_id: str,
    seed_centroid: Any,
    policy: PersonLearningPolicy,
) -> PersonLearningCandidate | None:
    """Aggregate independently acceptable sentence embeddings into one template.

    The seed is frozen when the user confirms the person. Automatic assignments
    must agree with it and remain separated from other meeting speakers. User
    corrections may bypass assignment-confidence gates, but never basic audio
    quality, overlap, or robust-outlier filtering.
    """

    record_list = [dict(record) for record in records]
    seed = normalize_vector(seed_centroid)
    profiles = {
        str(label): normalize_vector(centroid)
        for label, centroid in speaker_profiles.items()
        if str(label)
    }
    other_intervals = [
        interval
        for record in record_list
        if str(record.get("assigned_speaker") or "") not in {"", speaker_id}
        if (interval := _record_interval(record)) is not None
    ]
    samples: list[tuple[np.ndarray, float, float, int, bool]] = []
    for record in record_list:
        if str(record.get("assigned_speaker") or "") != speaker_id:
            continue
        try:
            embedding = normalize_vector(record.get("embedding"))
            duration = max(0.0, float(record.get("duration_seconds") or 0.0))
            quality = max(0.0, min(1.0, float(record.get("quality") or 0.0)))
            index = int(record.get("index"))
        except (TypeError, ValueError):
            continue
        if embedding.shape != seed.shape:
            continue
        if duration < policy.min_sample_seconds or quality < policy.min_sample_quality:
            continue
        base = record.get("base_payload") if isinstance(record.get("base_payload"), dict) else {}
        try:
            speech_ratio = float(base.get("speech_audio_ratio", 1.0))
        except (TypeError, ValueError):
            speech_ratio = 0.0
        if speech_ratio < policy.min_speech_audio_ratio:
            continue
        if _overlaps(_record_interval(record), other_intervals, policy.overlap_tolerance_seconds):
            continue
        trusted = _user_trusted(record, speaker_id)
        seed_similarity = cosine_similarity(embedding, seed)
        if not trusted:
            try:
                unknown_probability = float(record.get("unknown_probability") or 0.0)
            except (TypeError, ValueError):
                unknown_probability = 1.0
            if (
                unknown_probability > policy.max_unknown_probability
                or seed_similarity < policy.seed_similarity
            ):
                continue
            other_scores = [
                cosine_similarity(embedding, centroid)
                for label, centroid in profiles.items()
                if label != speaker_id and centroid.shape == embedding.shape
            ]
            if (
                other_scores
                and seed_similarity - max(other_scores) < policy.competing_speaker_margin
            ):
                continue
        samples.append((embedding, duration, quality, index, trusted))

    if len(samples) < policy.min_sentences:
        return None
    preliminary = _weighted_centroid(samples)
    similarities = np.asarray(
        [cosine_similarity(embedding, preliminary) for embedding, *_rest in samples],
        dtype=np.float64,
    )
    median = float(np.median(similarities))
    mad = float(np.median(np.abs(similarities - median)))
    cutoff = median - max(0.08, 3.0 * mad)
    retained = [sample for sample, similarity in zip(samples, similarities) if similarity >= cutoff]
    if len(retained) < policy.min_sentences:
        return None
    speech_seconds = sum(sample[1] for sample in retained)
    if speech_seconds < policy.min_speech_seconds:
        return None
    centroid = _weighted_centroid(retained)
    final_similarities = [cosine_similarity(sample[0], centroid) for sample in retained]
    final_weights = [min(6.0, sample[1]) * (0.5 + 0.5 * sample[2]) for sample in retained]
    cohesion = float(np.average(final_similarities, weights=final_weights))
    seed_similarity = cosine_similarity(centroid, seed)
    if cohesion < policy.min_cohesion or seed_similarity < policy.seed_similarity:
        return None
    return PersonLearningCandidate(
        centroid=centroid,
        sentence_count=len(retained),
        speech_seconds=float(speech_seconds),
        cohesion=cohesion,
        seed_similarity=seed_similarity,
        outlier_count=len(samples) - len(retained),
        record_indexes=frozenset(sample[3] for sample in retained),
        user_trusted_indexes=frozenset(sample[3] for sample in retained if sample[4]),
    )
