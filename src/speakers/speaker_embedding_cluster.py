"""Embedding-only online speaker clustering.

This module is intentionally independent from transcription. Callers pass one
sentence embedding at a time and receive a stable speaker decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def speaker_label_index(label: str) -> int | None:
    value = str(label or "").strip().upper()
    if not value.startswith("S") or not value[1:].isdigit():
        return None
    index = int(value[1:])
    return index if index > 0 else None


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Embedding shape mismatch: {left.shape} vs {right.shape}")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp((value - max_value) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class SpeakerProfile:
    index: int
    centroid: np.ndarray
    sentence_count: int
    speech_seconds: float
    created_at: float
    last_seen_at: float

    @property
    def label(self) -> str:
        return f"S{self.index}"

    def update(self, embedding: np.ndarray, duration_seconds: float, weight: float) -> None:
        weight = clamp01(weight)
        centroid = (self.centroid * (1.0 - weight)) + (embedding * weight)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            self.centroid = (centroid / norm).astype(np.float32)
        self.sentence_count += 1
        self.speech_seconds += max(0.0, duration_seconds)
        self.last_seen_at = time.time()


@dataclass
class NewSpeakerCandidate:
    centroid: np.ndarray
    sentence_count: int
    speech_seconds: float
    created_at: float
    last_seen_at: float

    def update(self, embedding: np.ndarray, duration_seconds: float) -> None:
        weight = 1.0 / float(self.sentence_count + 1)
        centroid = (self.centroid * (1.0 - weight)) + (embedding * weight)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            self.centroid = (centroid / norm).astype(np.float32)
        self.sentence_count += 1
        self.speech_seconds += max(0.0, duration_seconds)
        self.last_seen_at = time.time()


@dataclass
class SpeakerDecision:
    assigned_speaker: str | None
    created_speaker: bool
    probabilities: dict[str, float]
    similarities: dict[str, float]
    unknown_probability: float
    top_similarity: float | None
    margin: float | None
    quality: float
    assignment_source: str = "embedding"


class SpeakerMemory:
    """Stable incremental centroid clustering over sentence embeddings."""

    def __init__(
        self,
        same_speaker_similarity: float = 0.45,
        similarity_temperature: float = 0.07,
        speaker_softmax_temperature: float = 0.075,
        new_speaker_threshold: float = 0.58,
        duplicate_profile_similarity: float = 0.40,
        unknown_short_threshold: float = 0.86,
        min_first_speaker_seconds: float = 1.2,
        min_new_speaker_seconds: float = 2.0,
        late_new_speaker_min_seconds: float = 3.5,
        max_speakers: int = 10,
        min_margin: float = 0.05,
        margin_temperature: float = 0.035,
        update_unknown_max: float = 0.55,
        new_speaker_confirmation_count: int = 1,
        new_speaker_confirmation_similarity: float = 0.52,
        max_pending_new_speakers: int = 6,
    ) -> None:
        self.same_speaker_similarity = same_speaker_similarity
        self.similarity_temperature = similarity_temperature
        self.speaker_softmax_temperature = speaker_softmax_temperature
        self.new_speaker_threshold = new_speaker_threshold
        self.duplicate_profile_similarity = duplicate_profile_similarity
        self.unknown_short_threshold = unknown_short_threshold
        self.min_first_speaker_seconds = min_first_speaker_seconds
        self.min_new_speaker_seconds = min_new_speaker_seconds
        self.late_new_speaker_min_seconds = late_new_speaker_min_seconds
        self.max_speakers = max_speakers
        self.min_margin = min_margin
        self.margin_temperature = margin_temperature
        self.update_unknown_max = update_unknown_max
        self.new_speaker_confirmation_count = max(1, int(new_speaker_confirmation_count))
        self.new_speaker_confirmation_similarity = new_speaker_confirmation_similarity
        self.max_pending_new_speakers = max(1, int(max_pending_new_speakers))
        self._profiles: list[SpeakerProfile] = []
        self._new_speaker_candidates: list[NewSpeakerCandidate] = []
        self.locked_labels: set[str] = set()
        self._lock = threading.Lock()

    def profile_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    def add_profile(
        self,
        embedding: np.ndarray,
        duration_seconds: float = 0.0,
        sentence_count: int = 1,
        locked: bool = False,
    ) -> str:
        embedding = normalize_vector(embedding)
        with self._lock:
            profile = self._create_profile_locked(embedding, duration_seconds, sentence_count=sentence_count)
            if locked:
                self.locked_labels.add(profile.label)
            return profile.label

    def upsert_profile(
        self,
        label: str,
        embedding: np.ndarray,
        duration_seconds: float = 0.0,
        sentence_count: int = 1,
        locked: bool = False,
    ) -> str:
        embedding = normalize_vector(embedding)
        index = speaker_label_index(label)
        if index is None:
            raise ValueError(f"Invalid speaker label: {label!r}")
        with self._lock:
            for profile in self._profiles:
                if profile.index != index:
                    continue
                weight = min(0.35, 1.0 / max(1.0, float(profile.sentence_count + 1) ** 0.35))
                profile.update(embedding, duration_seconds, weight)
                if locked:
                    self.locked_labels.add(profile.label)
                return profile.label
            now = time.time()
            profile = SpeakerProfile(
                index=index,
                centroid=embedding.astype(np.float32),
                sentence_count=max(1, int(sentence_count)),
                speech_seconds=max(0.0, duration_seconds),
                created_at=now,
                last_seen_at=now,
            )
            self._profiles.append(profile)
            self._profiles.sort(key=lambda item: item.index)
            if locked:
                self.locked_labels.add(profile.label)
            return profile.label

    def replace_profiles(self, profiles: list[dict[str, Any]]) -> None:
        with self._lock:
            self._profiles = []
            self._new_speaker_candidates = []
            self.locked_labels = set()
            for item in profiles:
                embedding = normalize_vector(item["centroid"])
                profile = self._create_profile_locked(
                    embedding,
                    float(item.get("speech_seconds") or 0.0),
                    sentence_count=max(1, int(item.get("sentence_count") or 1)),
                )
                if bool(item.get("locked")):
                    self.locked_labels.add(profile.label)

    def export_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "label": profile.label,
                    "index": profile.index,
                    "centroid": profile.centroid.astype(float).tolist(),
                    "sentence_count": profile.sentence_count,
                    "speech_seconds": profile.speech_seconds,
                    "created_at": profile.created_at,
                    "last_seen_at": profile.last_seen_at,
                    "locked": profile.label in self.locked_labels,
                }
                for profile in self._profiles
            ]

    def classify(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        allow_new_speaker: bool = True,
    ) -> SpeakerDecision:
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)
        with self._lock:
            if not self._profiles:
                if not allow_new_speaker or duration_seconds < self.min_first_speaker_seconds:
                    return SpeakerDecision(None, False, {"unknown": 1.0}, {}, 1.0, None, None, quality)
                profile = self._create_profile_locked(embedding, duration_seconds)
                return self._created_profile_decision(profile, quality)
            return self._score_locked(embedding, duration_seconds, quality, allow_new_speaker)

    def score_existing(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        min_similarity: float | None = None,
        min_margin: float | None = None,
    ) -> SpeakerDecision:
        """Score against current speakers without creating or updating profiles."""
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)
        with self._lock:
            if not self._profiles:
                return SpeakerDecision(None, False, {"unknown": 1.0}, {}, 1.0, None, None, quality)
            return self._score_existing_locked(
                embedding,
                quality,
                self.same_speaker_similarity if min_similarity is None else min_similarity,
                self.min_margin if min_margin is None else min_margin,
            )

    def _score_locked(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        quality: float,
        allow_new_speaker: bool,
    ) -> SpeakerDecision:
        similarities = [cosine_similarity(embedding, profile.centroid) for profile in self._profiles]
        order = sorted(range(len(similarities)), key=lambda index: similarities[index], reverse=True)
        top_index = order[0]
        top_similarity = similarities[top_index]
        second_similarity = similarities[order[1]] if len(order) > 1 else -1.0
        margin = top_similarity - second_similarity if len(order) > 1 else 1.0

        same_probability = sigmoid((top_similarity - self.same_speaker_similarity) / self.similarity_temperature)
        margin_probability = 1.0
        if len(self._profiles) > 1:
            margin_probability = sigmoid((margin - self.min_margin) / self.margin_temperature)
        maturity = min(1.0, 0.45 + 0.55 * (self._profiles[top_index].speech_seconds / 8.0))
        known_mass = clamp01(same_probability * margin_probability * maturity * (0.55 + 0.45 * quality))
        unknown_probability = 1.0 - known_mass

        probabilities = {"unknown": unknown_probability}
        similarities_by_label = {}
        for profile, similarity, probability in zip(
            self._profiles,
            similarities,
            softmax(similarities, self.speaker_softmax_temperature),
        ):
            probabilities[f"speaker{profile.index}"] = known_mass * probability
            similarities_by_label[profile.label] = similarity

        single_profile_weak_short = (
            len(self._profiles) == 1
            and duration_seconds < self.min_new_speaker_seconds
            and top_similarity < max(
                self.same_speaker_similarity + 0.12,
                self.duplicate_profile_similarity + 0.08,
            )
        )
        created = False
        assigned: SpeakerProfile | None
        if (
            allow_new_speaker
            and self._should_create_new_profile(
                unknown_probability,
                top_similarity,
                margin,
                duration_seconds,
            )
        ):
            assigned = self._create_or_stage_new_profile_locked(embedding, duration_seconds)
            created = assigned is not None
        elif (
            single_profile_weak_short
            or (
                unknown_probability >= self.unknown_short_threshold
                and duration_seconds < self.min_new_speaker_seconds
            )
        ):
            assigned = None
        else:
            assigned = self._profiles[top_index]
            if (
                assigned.label not in self.locked_labels
                and unknown_probability <= self.update_unknown_max
                and quality >= 0.35
            ):
                weight = min(0.28, 0.08 + 0.18 * quality)
                weight /= max(1.0, float(assigned.sentence_count) ** 0.35)
                assigned.update(embedding, duration_seconds, weight)

        if created and assigned is not None:
            return self._created_profile_decision(
                assigned,
                quality,
                similarities_by_label,
                top_similarity,
                margin,
            )

        return SpeakerDecision(
            assigned_speaker=None if assigned is None else assigned.label,
            created_speaker=created,
            probabilities={key: round(float(value), 4) for key, value in probabilities.items()},
            similarities={key: round(float(value), 4) for key, value in similarities_by_label.items()},
            unknown_probability=round(float(unknown_probability), 4),
            top_similarity=round(float(top_similarity), 4),
            margin=round(float(margin), 4),
            quality=round(float(quality), 4),
        )

    def _score_existing_locked(
        self,
        embedding: np.ndarray,
        quality: float,
        min_similarity: float,
        min_margin: float,
    ) -> SpeakerDecision:
        similarities = [cosine_similarity(embedding, profile.centroid) for profile in self._profiles]
        order = sorted(range(len(similarities)), key=lambda index: similarities[index], reverse=True)
        top_index = order[0]
        top_similarity = similarities[top_index]
        second_similarity = similarities[order[1]] if len(order) > 1 else -1.0
        margin = top_similarity - second_similarity if len(order) > 1 else 1.0

        same_probability = sigmoid((top_similarity - self.same_speaker_similarity) / self.similarity_temperature)
        margin_probability = 1.0
        if len(self._profiles) > 1:
            margin_probability = sigmoid((margin - self.min_margin) / self.margin_temperature)
        maturity = min(1.0, 0.45 + 0.55 * (self._profiles[top_index].speech_seconds / 8.0))
        known_mass = clamp01(same_probability * margin_probability * maturity * (0.55 + 0.45 * quality))
        unknown_probability = 1.0 - known_mass

        probabilities = {"unknown": unknown_probability}
        similarities_by_label = {}
        for profile, similarity, probability in zip(
            self._profiles,
            similarities,
            softmax(similarities, self.speaker_softmax_temperature),
        ):
            probabilities[f"speaker{profile.index}"] = known_mass * probability
            similarities_by_label[profile.label] = similarity

        passes_similarity = top_similarity >= min_similarity
        passes_margin = len(self._profiles) == 1 or margin >= min_margin
        assigned = self._profiles[top_index] if passes_similarity and passes_margin else None

        return SpeakerDecision(
            assigned_speaker=None if assigned is None else assigned.label,
            created_speaker=False,
            probabilities={key: round(float(value), 4) for key, value in probabilities.items()},
            similarities={key: round(float(value), 4) for key, value in similarities_by_label.items()},
            unknown_probability=round(float(unknown_probability), 4),
            top_similarity=round(float(top_similarity), 4),
            margin=round(float(margin), 4),
            quality=round(float(quality), 4),
            assignment_source="retro",
        )

    @staticmethod
    def _created_profile_decision(
        profile: SpeakerProfile,
        quality: float,
        similarities: dict[str, float] | None = None,
        top_similarity: float | None = None,
        margin: float | None = None,
    ) -> SpeakerDecision:
        speaker_key = f"speaker{profile.index}"
        similarity_values = dict(similarities or {})
        if not similarity_values:
            similarity_values[profile.label] = 1.0
        return SpeakerDecision(
            assigned_speaker=profile.label,
            created_speaker=True,
            probabilities={"unknown": 0.0, speaker_key: 1.0},
            similarities={key: round(float(value), 4) for key, value in similarity_values.items()},
            unknown_probability=0.0,
            top_similarity=round(float(1.0 if top_similarity is None else top_similarity), 4),
            margin=round(float(1.0 if margin is None else margin), 4),
            quality=round(float(quality), 4),
        )

    def _should_create_new_profile(
        self,
        unknown_probability: float,
        top_similarity: float,
        margin: float,
        duration_seconds: float,
    ) -> bool:
        if duration_seconds < self.min_new_speaker_seconds:
            return False
        if len(self._profiles) >= self.max_speakers:
            return False
        long_low_margin_distinct = (
            duration_seconds >= self.late_new_speaker_min_seconds
            and unknown_probability >= 0.30
            and top_similarity < self.duplicate_profile_similarity + 0.05
            and margin < max(self.min_margin, 0.10)
        )
        if unknown_probability < self.new_speaker_threshold and not long_low_margin_distinct:
            return False
        clearly_distinct = top_similarity < self.duplicate_profile_similarity
        ambiguously_distinct = (
            unknown_probability >= max(self.new_speaker_threshold, 0.8)
            and top_similarity < self.duplicate_profile_similarity + 0.04
            and margin < max(self.min_margin, 0.08)
        )
        long_ambiguously_distinct = (
            duration_seconds >= self.late_new_speaker_min_seconds
            and unknown_probability >= self.new_speaker_threshold
            and top_similarity < self.duplicate_profile_similarity + 0.04
            and margin < max(self.min_margin, 0.08)
        )
        long_weakly_distinct = (
            duration_seconds >= self.late_new_speaker_min_seconds
            and unknown_probability >= 0.25
            and top_similarity < self.duplicate_profile_similarity
            and margin < max(self.min_margin, 0.12)
        )
        if not (
            clearly_distinct
            or ambiguously_distinct
            or long_ambiguously_distinct
            or long_weakly_distinct
            or long_low_margin_distinct
        ):
            return False
        if (
            len(self._profiles) >= 4
            and duration_seconds < self.late_new_speaker_min_seconds
            and top_similarity >= max(0.25, self.duplicate_profile_similarity - 0.15)
        ):
            return False
        return True

    def _create_or_stage_new_profile_locked(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
    ) -> SpeakerProfile | None:
        if self.new_speaker_confirmation_count <= 1:
            return self._create_profile_locked(embedding, duration_seconds)

        candidate = self._best_new_speaker_candidate_locked(embedding)
        if candidate is None:
            self._add_new_speaker_candidate_locked(embedding, duration_seconds)
            return None

        candidate.update(embedding, duration_seconds)
        if (
            candidate.sentence_count >= self.new_speaker_confirmation_count
            and candidate.speech_seconds >= self.min_new_speaker_seconds
        ):
            self._new_speaker_candidates = [
                item for item in self._new_speaker_candidates
                if item is not candidate
            ]
            return self._create_profile_locked(
                candidate.centroid,
                candidate.speech_seconds,
                sentence_count=candidate.sentence_count,
            )
        return None

    def _best_new_speaker_candidate_locked(self, embedding: np.ndarray) -> NewSpeakerCandidate | None:
        best_candidate = None
        best_similarity = -1.0
        for candidate in self._new_speaker_candidates:
            similarity = cosine_similarity(embedding, candidate.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate
        if best_candidate is None or best_similarity < self.new_speaker_confirmation_similarity:
            return None
        return best_candidate

    def _add_new_speaker_candidate_locked(self, embedding: np.ndarray, duration_seconds: float) -> None:
        now = time.time()
        self._new_speaker_candidates.append(
            NewSpeakerCandidate(
                centroid=embedding.astype(np.float32),
                sentence_count=1,
                speech_seconds=max(0.0, duration_seconds),
                created_at=now,
                last_seen_at=now,
            )
        )
        if len(self._new_speaker_candidates) > self.max_pending_new_speakers:
            self._new_speaker_candidates.sort(
                key=lambda candidate: (
                    candidate.sentence_count,
                    candidate.speech_seconds,
                    candidate.last_seen_at,
                )
            )
            del self._new_speaker_candidates[0]

    def _create_profile_locked(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        sentence_count: int = 1,
    ) -> SpeakerProfile:
        now = time.time()
        profile = SpeakerProfile(
            index=len(self._profiles) + 1,
            centroid=embedding.astype(np.float32),
            sentence_count=sentence_count,
            speech_seconds=max(0.0, duration_seconds),
            created_at=now,
            last_seen_at=now,
        )
        self._profiles.append(profile)
        return profile

    @staticmethod
    def _duration_quality(duration_seconds: float) -> float:
        return max(0.25, min(1.0, (duration_seconds - 0.35) / (2.2 - 0.35)))
