"""Embedding-backed speaker profile memory for realtime diarization."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from common.audio_utils import (
    clamp01,
    cosine_similarity,
    normalize_vector,
    sigmoid,
    softmax,
)


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
    """Stable incremental speaker memory based only on voice embeddings."""

    def __init__(
        self,
        same_speaker_similarity: float,
        similarity_temperature: float,
        speaker_softmax_temperature: float,
        new_speaker_threshold: float,
        duplicate_profile_similarity: float,
        unknown_short_threshold: float,
        min_first_speaker_seconds: float,
        min_new_speaker_seconds: float,
        late_new_speaker_min_seconds: float,
        max_speakers: int,
        min_margin: float,
        margin_temperature: float,
        update_unknown_max: float,
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
        self._profiles: list[SpeakerProfile] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._profiles = []

    def classify(self, embedding: np.ndarray, duration_seconds: float) -> SpeakerDecision:
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)

        with self._lock:
            if not self._profiles:
                if duration_seconds < self.min_first_speaker_seconds:
                    return SpeakerDecision(
                        assigned_speaker=None,
                        created_speaker=False,
                        probabilities={"unknown": 1.0},
                        similarities={},
                        unknown_probability=1.0,
                        top_similarity=None,
                        margin=None,
                        quality=quality,
                    )
                profile = self._create_profile_locked(embedding, duration_seconds)
                return self._created_profile_decision(profile, quality)

            return self._score_locked(
                embedding=embedding,
                duration_seconds=duration_seconds,
                quality=quality,
                allow_create=True,
                allow_update=True,
                force_assignment=False,
            )

    def score_existing(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        force_assignment: bool = False,
    ) -> SpeakerDecision:
        embedding = normalize_vector(embedding)
        quality = self._duration_quality(duration_seconds)

        with self._lock:
            if not self._profiles:
                return SpeakerDecision(
                    assigned_speaker=None,
                    created_speaker=False,
                    probabilities={"unknown": 1.0},
                    similarities={},
                    unknown_probability=1.0,
                    top_similarity=None,
                    margin=None,
                    quality=quality,
                )
            return self._score_locked(
                embedding=embedding,
                duration_seconds=duration_seconds,
                quality=quality,
                allow_create=False,
                allow_update=False,
                force_assignment=force_assignment,
            )

    def _score_locked(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        quality: float,
        allow_create: bool,
        allow_update: bool,
        force_assignment: bool,
    ) -> SpeakerDecision:
        similarities = [
            cosine_similarity(embedding, profile.centroid)
            for profile in self._profiles
        ]
        order = sorted(range(len(similarities)), key=lambda index: similarities[index], reverse=True)
        top_index = order[0]
        top_similarity = similarities[top_index]
        second_similarity = similarities[order[1]] if len(order) > 1 else -1.0
        margin = top_similarity - second_similarity if len(order) > 1 else 1.0

        same_probability = sigmoid(
            (top_similarity - self.same_speaker_similarity)
            / self.similarity_temperature
        )
        margin_probability = 1.0
        if len(self._profiles) > 1:
            margin_probability = sigmoid(
                (margin - self.min_margin) / self.margin_temperature
            )
        maturity = min(1.0, 0.45 + 0.55 * (self._profiles[top_index].speech_seconds / 8.0))
        quality_factor = 0.55 + 0.45 * quality
        known_mass = clamp01(
            same_probability
            * margin_probability
            * maturity
            * quality_factor
        )
        unknown_probability = 1.0 - known_mass
        speaker_distribution = softmax(similarities, self.speaker_softmax_temperature)

        probabilities = {"unknown": unknown_probability}
        similarities_by_label = {}
        for profile, similarity, speaker_probability in zip(
            self._profiles,
            similarities,
            speaker_distribution,
        ):
            key = f"speaker{profile.index}"
            probabilities[key] = known_mass * speaker_probability
            similarities_by_label[profile.label] = similarity

        created_speaker = False
        assigned_profile: SpeakerProfile | None = None
        create_as_new = self._should_create_new_profile_locked(
            unknown_probability=unknown_probability,
            top_similarity=top_similarity,
            margin=margin,
            duration_seconds=duration_seconds,
        )
        if allow_create and create_as_new:
            assigned_profile = self._create_profile_locked(embedding, duration_seconds)
            created_speaker = True
        elif (
            not force_assignment
            and unknown_probability >= self.unknown_short_threshold
            and duration_seconds < self.min_new_speaker_seconds
        ):
            assigned_profile = None
        else:
            assigned_profile = self._profiles[top_index]
            if allow_update and unknown_probability <= self.update_unknown_max and quality >= 0.35:
                update_weight = min(0.28, 0.08 + 0.18 * quality)
                update_weight /= max(1.0, float(assigned_profile.sentence_count) ** 0.35)
                assigned_profile.update(embedding, duration_seconds, update_weight)

        if created_speaker and assigned_profile is not None:
            return self._created_profile_decision(
                assigned_profile,
                quality,
                similarities_by_label,
                top_similarity,
                margin,
            )

        return SpeakerDecision(
            assigned_speaker=None if assigned_profile is None else assigned_profile.label,
            created_speaker=created_speaker,
            probabilities={key: round(float(value), 4) for key, value in probabilities.items()},
            similarities={key: round(float(value), 4) for key, value in similarities_by_label.items()},
            unknown_probability=round(float(unknown_probability), 4),
            top_similarity=round(float(top_similarity), 4),
            margin=round(float(margin), 4),
            quality=round(float(quality), 4),
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

    def _should_create_new_profile_locked(
        self,
        unknown_probability: float,
        top_similarity: float,
        margin: float,
        duration_seconds: float,
    ) -> bool:
        if unknown_probability < self.new_speaker_threshold:
            return False
        if duration_seconds < self.min_new_speaker_seconds:
            return False
        if len(self._profiles) >= self.max_speakers:
            return False

        clearly_distinct = top_similarity < self.duplicate_profile_similarity
        ambiguously_distinct = (
            unknown_probability >= max(self.new_speaker_threshold, 0.8)
            and top_similarity < self.duplicate_profile_similarity + 0.04
            and margin < max(self.min_margin, 0.08)
        )
        if not (clearly_distinct or ambiguously_distinct):
            return False

        many_profiles_exist = len(self._profiles) >= 4
        short_late_candidate = duration_seconds < self.late_new_speaker_min_seconds
        has_some_existing_match = top_similarity >= max(0.25, self.duplicate_profile_similarity - 0.15)
        if many_profiles_exist and short_late_candidate and has_some_existing_match:
            return False

        return True

    def _create_profile_locked(self, embedding: np.ndarray, duration_seconds: float) -> SpeakerProfile:
        now = time.time()
        profile = SpeakerProfile(
            index=len(self._profiles) + 1,
            centroid=embedding.astype(np.float32),
            sentence_count=1,
            speech_seconds=max(0.0, duration_seconds),
            created_at=now,
            last_seen_at=now,
        )
        self._profiles.append(profile)
        return profile

    @staticmethod
    def _duration_quality(duration_seconds: float) -> float:
        # Embeddings from tiny utterances are noisy. Treat 2.2s+ as full quality.
        return max(0.25, min(1.0, (duration_seconds - 0.35) / (2.2 - 0.35)))
