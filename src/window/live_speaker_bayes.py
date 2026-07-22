"""Causal two-window Bayesian speaker-state filter.

Each live probe is an observation rather than a final decision.  A compact
Hidden Markov Model (HMM) combines the new similarity evidence with the prior
probability that the currently active speaker continues.  This provides
history without fixed majority-vote windows and lets strong change evidence
override persistence immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np

from speakers.speaker_embedding_cluster import normalize_vector
from window.live_speaker_algorithm import LiveSpeakerDecision, SpeakerProfileEvent
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_replay import CachedLiveWindowBlock


BAYES_ALGORITHM_ID = "causal_two_window_bayesian_state_filter_v1"


@dataclass(frozen=True)
class BayesSpeakerTrackerConfig:
    scale_windows: tuple[float, ...] = ()
    scale_weights: tuple[float, ...] = ()
    scale_confidence_power: float = 0.0
    scale_confidence_floor: float = 0.02
    min_similarity: float = 0.25
    min_margin: float = 0.0
    similarity_temperature: float = 0.075
    unknown_bias: float = 0.0
    profile_count_bias_threshold: int = 0
    low_profile_unknown_bias: float = 0.0
    high_profile_unknown_bias: float = 0.0
    profile_count_unknown_bias_slope: float = 0.0
    profile_history_size: int = 1
    profile_history_max_weight: float = 0.0
    profile_maturity_logit_strength: float = 0.0
    profile_maturity_pseudoseconds: float = 1.0
    profile_cohort_mean_strength: float = 0.0
    profile_cohort_max_strength: float = 0.0
    enable_provisional_profiles: bool = False
    provisional_creation_count: int = 2
    provisional_later_creation_count: int = 0
    provisional_later_creation_profile_threshold: int = 0
    provisional_creation_similarity_ceiling: float = 0.20
    provisional_boundary_creation_similarity_ceiling: float = -1.0
    provisional_boundary_continuity_max_similarity: float = -1.0
    boundary_short_only_max_continuity: float = -1.0
    boundary_residual_incumbent_alpha: float = 0.0
    provisional_creation_max_finalized_profiles: int = -1
    provisional_merge_min_similarity: float = 0.25
    provisional_merge_recency_weight: float = 0.0
    provisional_merge_recency_seconds: float = 1.0
    provisional_update_alpha: float = 0.0
    provisional_update_continuity_min_similarity: float = -1.0
    provisional_update_history_size: int = 1
    provisional_prototype_bank_size: int = 1
    provisional_prototype_weight: float = 0.0
    provisional_prototype_update_min_similarity: float = -1.0
    provisional_reactivation_min_similarity: float = -2.0
    provisional_expiry_seconds: float = 0.0
    provisional_max_active_count: int = 0
    provisional_pool_overflow_strategy: str = "recent"
    provisional_pool_overflow_update_alpha: float = 0.0
    provisional_pool_overflow_prototype_bank_size: int = 0
    provisional_pool_overflow_prototype_weight: float = 0.0
    provisional_scale_agreement_min_similarity: float = -1.0
    provisional_assignment_scale_agreement_min_similarity: float = -1.0
    incumbent_hold_scale_agreement_min_similarity: float = -1.0
    incumbent_continuity_min_similarity: float = -1.0
    incumbent_continuity_history_size: int = 3
    incumbent_continuity_update_on_hold: bool = False
    provisional_temporal_consistency_min_similarity: float = -1.0
    short_long_crossover_min_margin: float = -1.0
    short_long_crossover_min_similarity: float = -1.0
    short_long_crossover_count: int = 1
    short_long_differential_candidate_gain: float = -2.0
    short_long_differential_incumbent_loss: float = -2.0
    stay_probability: float = 0.80
    prior_strength: float = 1.0
    evidence_strength: float = 1.0
    min_known_probability: float = 0.45
    switch_probability_margin: float = 0.0
    unknown_release_count: int = 2
    silence_release_count: int = 2

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BayesSpeakerTrackerConfig":
        """Load current settings while tolerating keys from archived experiments."""

        supported = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in supported})

    def __post_init__(self) -> None:
        if self.scale_windows and len(self.scale_windows) != len(self.scale_weights):
            raise ValueError("scale window and weight counts must match")
        if float(self.scale_confidence_power) < 0.0:
            raise ValueError("scale_confidence_power must be non-negative")
        if float(self.scale_confidence_floor) <= 0.0:
            raise ValueError("scale_confidence_floor must be positive")
        if float(self.similarity_temperature) <= 0.0:
            raise ValueError("similarity_temperature must be positive")
        if not 0.0 <= float(self.stay_probability) <= 1.0:
            raise ValueError("stay_probability must be in [0, 1]")
        if float(self.prior_strength) < 0.0 or float(self.evidence_strength) <= 0.0:
            raise ValueError("Bayesian strengths are invalid")
        if not 0.0 <= float(self.min_known_probability) <= 1.0:
            raise ValueError("min_known_probability must be in [0, 1]")
        if int(self.unknown_release_count) < 1 or int(self.silence_release_count) < 1:
            raise ValueError("release counts must be positive")
        if int(self.profile_count_bias_threshold) < 0:
            raise ValueError("profile_count_bias_threshold must be non-negative")
        if int(self.profile_history_size) < 1:
            raise ValueError("profile_history_size must be positive")
        if not 0.0 <= float(self.profile_history_max_weight) <= 1.0:
            raise ValueError("profile_history_max_weight must be in [0, 1]")
        if float(self.profile_maturity_pseudoseconds) <= 0.0:
            raise ValueError("profile_maturity_pseudoseconds must be positive")
        if int(self.provisional_creation_count) < 1:
            raise ValueError("provisional_creation_count must be positive")
        if int(self.provisional_later_creation_count) < 0:
            raise ValueError("provisional_later_creation_count must be non-negative")
        if int(self.provisional_later_creation_profile_threshold) < 0:
            raise ValueError("provisional_later_creation_profile_threshold must be non-negative")
        if int(self.provisional_creation_max_finalized_profiles) < -1:
            raise ValueError("provisional_creation_max_finalized_profiles must be -1 or non-negative")
        if not -1.0 <= float(self.provisional_creation_similarity_ceiling) <= 1.0:
            raise ValueError("provisional_creation_similarity_ceiling must be in [-1, 1]")
        if not -1.0 <= float(self.provisional_boundary_creation_similarity_ceiling) <= 1.0:
            raise ValueError("provisional_boundary_creation_similarity_ceiling must be in [-1, 1]")
        if not -1.0 <= float(self.provisional_boundary_continuity_max_similarity) <= 1.0:
            raise ValueError("provisional_boundary_continuity_max_similarity must be in [-1, 1]")
        if not -1.0 <= float(self.boundary_short_only_max_continuity) <= 1.0:
            raise ValueError("boundary_short_only_max_continuity must be in [-1, 1]")
        if not 0.0 <= float(self.boundary_residual_incumbent_alpha) <= 1.5:
            raise ValueError("boundary_residual_incumbent_alpha must be in [0, 1.5]")
        if not -1.0 <= float(self.provisional_merge_min_similarity) <= 1.0:
            raise ValueError("provisional_merge_min_similarity must be in [-1, 1]")
        if float(self.provisional_merge_recency_weight) < 0.0:
            raise ValueError("provisional_merge_recency_weight must be non-negative")
        if float(self.provisional_merge_recency_seconds) <= 0.0:
            raise ValueError("provisional_merge_recency_seconds must be positive")
        if not 0.0 <= float(self.provisional_update_alpha) <= 1.0:
            raise ValueError("provisional_update_alpha must be in [0, 1]")
        if not -1.0 <= float(self.provisional_update_continuity_min_similarity) <= 1.0:
            raise ValueError("provisional_update_continuity_min_similarity must be in [-1, 1]")
        if int(self.provisional_update_history_size) < 1:
            raise ValueError("provisional_update_history_size must be positive")
        if int(self.provisional_prototype_bank_size) < 1:
            raise ValueError("provisional_prototype_bank_size must be positive")
        if not 0.0 <= float(self.provisional_prototype_weight) <= 1.0:
            raise ValueError("provisional_prototype_weight must be in [0, 1]")
        if not -1.0 <= float(self.provisional_prototype_update_min_similarity) <= 1.0:
            raise ValueError("provisional_prototype_update_min_similarity must be in [-1, 1]")
        if not -2.0 <= float(self.provisional_reactivation_min_similarity) <= 1.0:
            raise ValueError("provisional_reactivation_min_similarity must be in [-2, 1]")
        if float(self.provisional_expiry_seconds) < 0.0:
            raise ValueError("provisional_expiry_seconds must be non-negative")
        if int(self.provisional_max_active_count) < 0:
            raise ValueError("provisional_max_active_count must be non-negative")
        if self.provisional_pool_overflow_strategy not in {"recent", "closest", "visible"}:
            raise ValueError("provisional_pool_overflow_strategy is invalid")
        if not 0.0 <= float(self.provisional_pool_overflow_update_alpha) <= 1.0:
            raise ValueError("provisional_pool_overflow_update_alpha must be in [0, 1]")
        if int(self.provisional_pool_overflow_prototype_bank_size) < 0:
            raise ValueError("provisional_pool_overflow_prototype_bank_size must be non-negative")
        if not 0.0 <= float(self.provisional_pool_overflow_prototype_weight) <= 1.0:
            raise ValueError("provisional_pool_overflow_prototype_weight must be in [0, 1]")
        if not -1.0 <= float(self.provisional_scale_agreement_min_similarity) <= 1.0:
            raise ValueError("provisional_scale_agreement_min_similarity must be in [-1, 1]")
        if not -1.0 <= float(self.provisional_assignment_scale_agreement_min_similarity) <= 1.0:
            raise ValueError("provisional_assignment_scale_agreement_min_similarity must be in [-1, 1]")
        if not -1.0 <= float(self.incumbent_hold_scale_agreement_min_similarity) <= 1.0:
            raise ValueError("incumbent_hold_scale_agreement_min_similarity must be in [-1, 1]")
        if not -1.0 <= float(self.incumbent_continuity_min_similarity) <= 1.0:
            raise ValueError("incumbent_continuity_min_similarity must be in [-1, 1]")
        if int(self.incumbent_continuity_history_size) < 1:
            raise ValueError("incumbent_continuity_history_size must be positive")
        if not -1.0 <= float(self.provisional_temporal_consistency_min_similarity) <= 1.0:
            raise ValueError("provisional_temporal_consistency_min_similarity must be in [-1, 1]")
        if float(self.short_long_crossover_min_margin) < -1.0:
            raise ValueError("short_long_crossover_min_margin must be -1 or non-negative")
        if not -1.0 <= float(self.short_long_crossover_min_similarity) <= 1.0:
            raise ValueError("short_long_crossover_min_similarity must be in [-1, 1]")
        if int(self.short_long_crossover_count) < 1:
            raise ValueError("short_long_crossover_count must be positive")
        if not -2.0 <= float(self.short_long_differential_candidate_gain) <= 2.0:
            raise ValueError("short_long_differential_candidate_gain must be in [-2, 2]")
        if not -2.0 <= float(self.short_long_differential_incumbent_loss) <= 2.0:
            raise ValueError("short_long_differential_incumbent_loss must be in [-2, 2]")


@dataclass
class _Profile:
    centroid: np.ndarray
    generation: int
    history: tuple[np.ndarray, ...] = ()
    sentence_count: int = 1
    speech_seconds: float = 0.0
    provisional: bool = False
    last_seen: float = -1.0
    live_prototypes: tuple[np.ndarray, ...] = ()
    overflow_prototypes: tuple[np.ndarray, ...] = ()


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {"unknown": 1.0}
    peak = max(logits.values())
    masses = {
        key: math.exp(max(-60.0, min(60.0, float(value) - peak)))
        for key, value in logits.items()
    }
    total = sum(masses.values())
    return {key: value / total for key, value in masses.items()}


class CausalBayesSpeakerTracker:
    """Filter speaker identity as a causal latent state over live probes."""

    def __init__(
        self,
        config: BayesSpeakerTrackerConfig | None = None,
        profile_events: Iterable[SpeakerProfileEvent] = (),
    ) -> None:
        self.config = config or BayesSpeakerTrackerConfig()
        self._events = sorted(
            list(profile_events),
            key=lambda item: (float(item.available_at), int(item.generation), str(item.speaker_id)),
        )
        self._next_event = 0
        self._profiles: dict[str, _Profile] = {}
        self._profile_fingerprints: dict[str, str] = {}
        self._posterior: dict[str, float] = {"unknown": 1.0}
        self._visible: str | None = None
        self._last_media_time = -1.0
        self._unknown_count = 0
        self._silence_count = 0
        self._profile_aliases: dict[str, str] = {}
        self._next_provisional_id = 1
        self._pending_provisional_embeddings: list[np.ndarray] = []
        self._crossover_label: str | None = None
        self._crossover_count = 0
        self._incumbent_history: dict[str, tuple[np.ndarray, ...]] = {}

    @property
    def visible_speaker(self) -> str | None:
        return self._visible

    def _remember_incumbent(self, label: str, item: MultiScaleStep) -> None:
        if not item.evidences:
            return
        shortest = min(item.evidences, key=lambda evidence: float(evidence.window_seconds))
        size = int(self.config.incumbent_continuity_history_size)
        history = self._incumbent_history.get(label, ())
        self._incumbent_history[label] = (*history, shortest.embedding)[-size:]

    def _identity_continuity(
        self,
        label: str | None,
        item: MultiScaleStep,
    ) -> float | None:
        if label is None or not item.evidences:
            return None
        history = self._incumbent_history.get(label, ())
        if not history:
            return None
        shortest = min(item.evidences, key=lambda evidence: float(evidence.window_seconds))
        anchor = normalize_vector(np.mean(history, axis=0))
        short_similarity = float(np.dot(shortest.embedding, anchor))
        return short_similarity

    def _incumbent_continuity(self, item: MultiScaleStep) -> float | None:
        threshold = float(self.config.incumbent_continuity_min_similarity)
        if threshold < -0.999999:
            return None
        return self._identity_continuity(self._visible, item)

    def _boundary_residualized_item(
        self,
        item: MultiScaleStep,
    ) -> tuple[MultiScaleStep, float | None]:
        """Remove a bounded incumbent component from the shortest boundary window.

        A short window immediately after a turn boundary can contain both the outgoing
        and incoming voices.  The recent incumbent history is a causal estimate of the
        outgoing component, so subtracting part of it can expose the incoming identity
        without requesting another embedding window.
        """

        alpha = float(self.config.boundary_residual_incumbent_alpha)
        label = self._visible
        if alpha <= 0.0 or label is None or not item.evidences:
            return item, None
        history = self._incumbent_history.get(label, ())
        if not history:
            return item, None
        shortest = min(item.evidences, key=lambda evidence: float(evidence.window_seconds))
        anchor = normalize_vector(np.mean(history, axis=0))
        residual = normalize_vector(shortest.embedding - alpha * anchor)
        evidences = tuple(
            MultiScaleEvidence(evidence.window_seconds, residual)
            if evidence is shortest else evidence
            for evidence in item.evidences
        )
        return replace(item, evidences=evidences), float(np.dot(residual, anchor))

    def _resolved_profile_label(
        self,
        external_label: str,
        centroid: np.ndarray,
    ) -> str:
        aliased = self._profile_aliases.get(external_label)
        if aliased is not None:
            return aliased
        if not self.config.enable_provisional_profiles:
            return external_label
        recency_weight = float(self.config.provisional_merge_recency_weight)
        recency_seconds = float(self.config.provisional_merge_recency_seconds)
        candidates = []
        for label, profile in self._profiles.items():
            if not profile.provisional:
                continue
            similarity = float(np.dot(centroid, profile.centroid))
            age = max(0.0, float(self._last_media_time) - float(profile.last_seen))
            freshness = math.exp(-age / recency_seconds) if profile.last_seen >= 0.0 else 0.0
            candidates.append((similarity + recency_weight * freshness, similarity, label))
        if candidates:
            _, similarity, label = max(candidates, key=lambda row: (row[0], row[1], row[2]))
            if similarity >= float(self.config.provisional_merge_min_similarity):
                self._profile_aliases[external_label] = label
                return label
        return external_label

    def _consider_provisional_profile(
        self,
        item: MultiScaleStep,
        similarities: dict[str, float],
    ) -> str | None:
        if not self.config.enable_provisional_profiles or not item.evidences:
            self._pending_provisional_embeddings.clear()
            return None
        finalized_limit = int(self.config.provisional_creation_max_finalized_profiles)
        finalized_count = sum(not profile.provisional for profile in self._profiles.values())
        if finalized_limit >= 0 and finalized_count > finalized_limit:
            self._pending_provisional_embeddings.clear()
            return None
        creation_ceiling = float(self.config.provisional_creation_similarity_ceiling)
        boundary_ceiling = float(
            self.config.provisional_boundary_creation_similarity_ceiling
        )
        boundary_max_continuity = float(
            self.config.provisional_boundary_continuity_max_similarity
        )
        boundary_continuity = self._incumbent_continuity(item)
        boundary_creation = (
            boundary_ceiling >= -0.999999
            and boundary_max_continuity >= -0.999999
            and boundary_continuity is not None
            and boundary_continuity <= boundary_max_continuity
        )
        if boundary_creation:
            creation_ceiling = max(creation_ceiling, boundary_ceiling)
        if max(similarities.values(), default=-1.0) >= creation_ceiling:
            self._pending_provisional_embeddings.clear()
            return None
        shortest = min(item.evidences, key=lambda evidence: float(evidence.window_seconds))
        if len(item.evidences) > 1:
            longest = max(item.evidences, key=lambda evidence: float(evidence.window_seconds))
            if float(np.dot(shortest.embedding, longest.embedding)) < float(
                self.config.provisional_scale_agreement_min_similarity
            ):
                self._pending_provisional_embeddings.clear()
                return None
        if self._pending_provisional_embeddings:
            pending_centroid = normalize_vector(np.mean(self._pending_provisional_embeddings, axis=0))
            if float(np.dot(shortest.embedding, pending_centroid)) < float(
                self.config.provisional_temporal_consistency_min_similarity
            ):
                self._pending_provisional_embeddings.clear()
        self._pending_provisional_embeddings.append(shortest.embedding)
        later_count = int(self.config.provisional_later_creation_count)
        later_threshold = int(self.config.provisional_later_creation_profile_threshold)
        needed = (
            later_count
            if finalized_count > later_threshold and later_count > 0
            else int(self.config.provisional_creation_count)
        )
        self._pending_provisional_embeddings = self._pending_provisional_embeddings[-needed:]
        if len(self._pending_provisional_embeddings) < needed:
            return None
        centroid = normalize_vector(np.mean(self._pending_provisional_embeddings, axis=0))
        reactivation_threshold = float(self.config.provisional_reactivation_min_similarity)
        if reactivation_threshold >= -1.0:
            reactivation_candidates = [
                (float(np.dot(centroid, profile.centroid)), label)
                for label, profile in self._profiles.items()
                if profile.provisional
            ]
            if reactivation_candidates:
                similarity, label = max(
                    reactivation_candidates, key=lambda pair: (pair[0], pair[1])
                )
                if similarity >= reactivation_threshold:
                    profile = self._profiles[label]
                    profile.last_seen = float(item.media_time)
                    self._posterior = {
                        state: (1.0 if state == label else 1e-9)
                        for state in ["unknown", *sorted(self._profiles)]
                    }
                    total = sum(self._posterior.values())
                    self._posterior = {
                        state: value / total for state, value in self._posterior.items()
                    }
                    self._pending_provisional_embeddings.clear()
                    return label
        max_active = int(self.config.provisional_max_active_count)
        active_provisionals = [
            (float(profile.last_seen), label)
            for label, profile in self._profiles.items()
            if profile.provisional
        ]
        if max_active > 0 and len(active_provisionals) >= max_active:
            strategy = self.config.provisional_pool_overflow_strategy
            if strategy == "visible" and self._visible in self._profiles and self._profiles[self._visible].provisional:
                label = str(self._visible)
            elif strategy == "closest":
                label = max(
                    (label for _last_seen, label in active_provisionals),
                    key=lambda candidate: (
                        float(np.dot(centroid, self._profiles[candidate].centroid)), candidate
                    ),
                )
            else:
                _, label = max(active_provisionals, key=lambda pair: (pair[0], pair[1]))
            profile = self._profiles[label]
            bank_size = int(self.config.provisional_pool_overflow_prototype_bank_size)
            if bank_size > 0:
                profile.overflow_prototypes = (*profile.overflow_prototypes, centroid)[-bank_size:]
            alpha = float(self.config.provisional_pool_overflow_update_alpha)
            if alpha > 0.0:
                profile.centroid = normalize_vector(
                    (1.0 - alpha) * profile.centroid + alpha * centroid
                )
            profile.last_seen = float(item.media_time)
            self._posterior = {
                state: (1.0 if state == label else 1e-9)
                for state in ["unknown", *sorted(self._profiles)]
            }
            total = sum(self._posterior.values())
            self._posterior = {
                state: value / total for state, value in self._posterior.items()
            }
            self._pending_provisional_embeddings.clear()
            return label
        while True:
            label = f"provisional_{self._next_provisional_id}"
            self._next_provisional_id += 1
            if label not in self._profiles:
                break
        self._profiles[label] = _Profile(
            centroid=centroid,
            generation=0,
            history=(centroid,),
            sentence_count=0,
            speech_seconds=float(shortest.window_seconds),
            provisional=True,
            last_seen=float(item.media_time),
            live_prototypes=(centroid,),
        )
        self._posterior = {
            state: (1.0 if state == label else 1e-9)
            for state in ["unknown", *sorted(self._profiles)]
        }
        total = sum(self._posterior.values())
        self._posterior = {state: value / total for state, value in self._posterior.items()}
        self._pending_provisional_embeddings.clear()
        return label

    def _expire_provisional_profiles(self, media_time: float) -> list[str]:
        expiry = float(self.config.provisional_expiry_seconds)
        if expiry <= 0.0:
            return []
        expired = [
            label
            for label, profile in self._profiles.items()
            if profile.provisional
            and profile.last_seen >= 0.0
            and media_time - float(profile.last_seen) >= expiry
        ]
        if not expired:
            return []
        for label in expired:
            self._profiles.pop(label, None)
            self._profile_fingerprints.pop(label, None)
            self._incumbent_history.pop(label, None)
            if self._visible == label:
                self._visible = None
        states = ["unknown", *sorted(self._profiles)]
        floor = 1e-9
        posterior = {
            state: max(floor, float(self._posterior.get(state, floor))) for state in states
        }
        total = sum(posterior.values())
        self._posterior = {state: value / total for state, value in posterior.items()}
        return sorted(expired)

    def _update_provisional_profile(
        self,
        label: str,
        item: MultiScaleStep,
        *,
        continuity_similarity: float | None = None,
    ) -> None:
        profile = self._profiles.get(label)
        alpha = float(self.config.provisional_update_alpha)
        if profile is None or not profile.provisional or not item.evidences:
            return
        profile.last_seen = float(item.media_time)
        continuity_threshold = float(
            self.config.provisional_update_continuity_min_similarity
        )
        if continuity_threshold > -0.999999 and (
            continuity_similarity is None or continuity_similarity < continuity_threshold
        ):
            return
        shortest = min(item.evidences, key=lambda evidence: float(evidence.window_seconds))
        if float(np.dot(shortest.embedding, profile.centroid)) >= float(
            self.config.provisional_prototype_update_min_similarity
        ):
            size = int(self.config.provisional_prototype_bank_size)
            profile.live_prototypes = (*profile.live_prototypes, shortest.embedding)[-size:]
        if alpha <= 0.0:
            return
        update_size = int(self.config.provisional_update_history_size)
        history = self._incumbent_history.get(label, ())
        recent_history = history[-(update_size - 1):] if update_size > 1 else ()
        target_embeddings = (*recent_history, shortest.embedding)
        update_target = normalize_vector(np.mean(target_embeddings, axis=0))
        profile.centroid = normalize_vector(
            (1.0 - alpha) * profile.centroid + alpha * update_target
        )
        profile.history = (*profile.history, profile.centroid)[-int(self.config.profile_history_size):]
        profile.speech_seconds += float(shortest.window_seconds)

    def sync_profiles(self, profiles: Iterable[dict[str, Any]]) -> list[str]:
        """Synchronize production profile snapshots without resetting belief."""

        changed: list[str] = []
        live_labels: set[str] = set()
        for raw in profiles:
            label = str(raw.get("label") or "").strip()
            if not label:
                continue
            centroid = normalize_vector(np.asarray(raw["centroid"], dtype=np.float32))
            label = self._resolved_profile_label(label, centroid)
            sentence_count = max(1, int(raw.get("sentence_count") or 1))
            speech_seconds = max(0.0, float(raw.get("speech_seconds") or 0.0))
            fingerprint_builder = hashlib.sha256(np.ascontiguousarray(centroid).tobytes())
            fingerprint_builder.update(str(sentence_count).encode("ascii"))
            fingerprint_builder.update(repr(speech_seconds).encode("ascii"))
            fingerprint = fingerprint_builder.hexdigest()
            live_labels.add(label)
            if self._profile_fingerprints.get(label) != fingerprint:
                old = self._profiles.get(label)
                generation = int(old.generation if old is not None else 0) + 1
                history = tuple(old.history if old is not None else ()) + (centroid,)
                history = history[-int(self.config.profile_history_size):]
                self._profiles[label] = _Profile(
                    centroid=centroid,
                    generation=generation,
                    history=history,
                    sentence_count=sentence_count,
                    speech_seconds=speech_seconds,
                    provisional=False,
                    last_seen=float(old.last_seen if old is not None else -1.0),
                    live_prototypes=tuple(old.live_prototypes if old is not None else ()),
                    overflow_prototypes=tuple(old.overflow_prototypes if old is not None else ()),
                )
                self._profile_fingerprints[label] = fingerprint
                changed.append(label)
        for label in set(self._profiles) - live_labels:
            if self.config.enable_provisional_profiles and self._profiles[label].provisional:
                continue
            self._profiles.pop(label, None)
            self._profile_fingerprints.pop(label, None)
            changed.append(label)
            if self._visible == label:
                self._visible = None
        states = ["unknown", *sorted(self._profiles)]
        floor = 1e-9
        posterior = {state: max(floor, float(self._posterior.get(state, floor))) for state in states}
        total = sum(posterior.values())
        self._posterior = {state: value / total for state, value in posterior.items()}
        return sorted(changed)

    def _apply_profile_events(self, media_time: float) -> list[str]:
        applied: list[str] = []
        while self._next_event < len(self._events):
            event = self._events[self._next_event]
            if float(event.available_at) > media_time + 1e-9:
                break
            external_label = str(event.speaker_id)
            event_centroid = normalize_vector(event.centroid)
            label = self._resolved_profile_label(external_label, event_centroid)
            old = self._profiles.get(label)
            if old is None or int(event.generation) >= int(old.generation):
                centroid = event_centroid
                history = tuple(old.history if old is not None else ()) + (centroid,)
                history = history[-int(self.config.profile_history_size):]
                self._profiles[label] = _Profile(
                    centroid=centroid,
                    generation=int(event.generation),
                    history=history,
                    sentence_count=max(1, int(event.sentence_count)),
                    speech_seconds=max(0.0, float(event.speech_seconds)),
                    provisional=False,
                    last_seen=float(old.last_seen if old is not None else -1.0),
                    live_prototypes=tuple(old.live_prototypes if old is not None else ()),
                    overflow_prototypes=tuple(old.overflow_prototypes if old is not None else ()),
                )
                applied.append(label)
            self._next_event += 1
        if applied:
            states = ["unknown", *sorted(self._profiles)]
            floor = 1e-9
            normalized = {state: max(floor, float(self._posterior.get(state, floor))) for state in states}
            total = sum(normalized.values())
            self._posterior = {state: value / total for state, value in normalized.items()}
        return applied

    def _fused_similarities(
        self,
        evidences: Sequence[MultiScaleEvidence],
        *,
        short_only: bool = False,
    ) -> dict[str, float]:
        if not evidences or not self._profiles:
            return {}
        configured = {
            round(float(window), 6): float(weight)
            for window, weight in zip(self.config.scale_windows, self.config.scale_weights)
        }
        shortest_window = min(float(evidence.window_seconds) for evidence in evidences)
        base_weights = [
            (
                0.0
                if short_only and float(evidence.window_seconds) > shortest_window + 1e-9
                else configured.get(round(float(evidence.window_seconds), 6), 1.0)
            )
            for evidence in evidences
        ]
        history_weight = float(self.config.profile_history_max_weight)
        scale_similarities: list[dict[str, float]] = []
        for evidence in evidences:
            scores: dict[str, float] = {}
            for label, profile in self._profiles.items():
                scores[label] = self._profile_evidence_similarity(
                    evidence.embedding, profile, history_weight
                )
            scale_similarities.append(scores)

        confidence_power = float(self.config.scale_confidence_power)
        if confidence_power > 0.0:
            confidence_floor = float(self.config.scale_confidence_floor)
            weights: list[float] = []
            for base_weight, scores in zip(base_weights, scale_similarities):
                ranked = sorted(scores.values(), reverse=True)
                separation = (
                    ranked[0] - ranked[1]
                    if len(ranked) > 1
                    else max(0.0, ranked[0] - float(self.config.min_similarity))
                )
                reliability = confidence_floor + max(0.0, separation)
                weights.append(max(0.0, base_weight) * reliability ** confidence_power)
        else:
            weights = [max(0.0, value) for value in base_weights]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0 / len(evidences)] * len(evidences)
        else:
            weights = [value / total for value in weights]

        result: dict[str, float] = {}
        for label, profile in self._profiles.items():
            result[label] = float(sum(
                weight * scores[label]
                for weight, scores in zip(weights, scale_similarities)
            ))
        mean_strength = float(self.config.profile_cohort_mean_strength)
        max_strength = float(self.config.profile_cohort_max_strength)
        if (mean_strength != 0.0 or max_strength != 0.0) and len(self._profiles) > 1:
            labels = sorted(self._profiles)
            for label in labels:
                cohort = [
                    float(np.dot(self._profiles[label].centroid, self._profiles[other].centroid))
                    for other in labels
                    if other != label
                ]
                result[label] -= mean_strength * float(sum(cohort) / len(cohort))
                result[label] -= max_strength * max(cohort)
        return result

    def _profile_evidence_similarity(
        self,
        embedding: np.ndarray,
        profile: _Profile,
        history_weight: float,
    ) -> float:
        history = profile.history or (profile.centroid,)
        current = float(np.dot(embedding, profile.centroid))
        historical_max = max(float(np.dot(embedding, item)) for item in history)
        score = (1.0 - history_weight) * current + history_weight * historical_max
        prototype_weight = float(self.config.provisional_prototype_weight)
        if prototype_weight > 0.0 and profile.live_prototypes:
            prototype_max = max(
                float(np.dot(embedding, prototype)) for prototype in profile.live_prototypes
            )
            score = (1.0 - prototype_weight) * score + prototype_weight * max(score, prototype_max)
        overflow_weight = float(self.config.provisional_pool_overflow_prototype_weight)
        if overflow_weight > 0.0 and profile.overflow_prototypes:
            overflow_max = max(
                float(np.dot(embedding, prototype)) for prototype in profile.overflow_prototypes
            )
            score = (1.0 - overflow_weight) * score + overflow_weight * max(score, overflow_max)
        return score

    def _single_scale_similarities(self, evidence: MultiScaleEvidence) -> dict[str, float]:
        history_weight = float(self.config.profile_history_max_weight)
        scores: dict[str, float] = {}
        for label, profile in self._profiles.items():
            scores[label] = self._profile_evidence_similarity(
                evidence.embedding, profile, history_weight
            )
        return scores

    def _short_long_crossover_candidate(
        self,
        evidences: Sequence[MultiScaleEvidence],
        previous: str | None,
    ) -> tuple[str | None, dict[str, Any]]:
        threshold = float(self.config.short_long_crossover_min_margin)
        if threshold < 0.0 or previous is None or len(evidences) != 2:
            self._crossover_label = None
            self._crossover_count = 0
            return None, {}
        ordered = sorted(evidences, key=lambda evidence: float(evidence.window_seconds))
        short_scores = self._single_scale_similarities(ordered[0])
        long_scores = self._single_scale_similarities(ordered[1])
        short_ranked = sorted(short_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        long_ranked = sorted(long_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        short_label, short_similarity = short_ranked[0] if short_ranked else (None, -1.0)
        long_label, long_similarity = long_ranked[0] if long_ranked else (None, -1.0)
        short_runner = short_ranked[1][1] if len(short_ranked) > 1 else -1.0
        short_margin = float(short_similarity) - float(short_runner)
        min_similarity = float(self.config.short_long_crossover_min_similarity)
        if min_similarity < -0.999999:
            min_similarity = float(self.config.min_similarity)
        candidate_gain = (
            float(short_similarity) - float(long_scores.get(short_label, -1.0))
            if short_label is not None else -2.0
        )
        incumbent_loss = float(long_scores.get(previous, -1.0)) - float(
            short_scores.get(previous, -1.0)
        )
        differential_candidate_gain = float(
            self.config.short_long_differential_candidate_gain
        )
        differential_incumbent_loss = float(
            self.config.short_long_differential_incumbent_loss
        )
        base_qualifies = (
            short_label is not None
            and short_label != previous
            and float(short_similarity) >= min_similarity
            and short_margin >= threshold
        )
        if differential_candidate_gain > -1.999999:
            qualifies = (
                base_qualifies
                and candidate_gain >= differential_candidate_gain
                and incumbent_loss >= differential_incumbent_loss
            )
        else:
            qualifies = base_qualifies and long_label == previous
        if qualifies:
            if self._crossover_label == short_label:
                self._crossover_count += 1
            else:
                self._crossover_label = short_label
                self._crossover_count = 1
        else:
            self._crossover_label = None
            self._crossover_count = 0
        candidate = (
            short_label
            if qualifies and self._crossover_count >= int(self.config.short_long_crossover_count)
            else None
        )
        return candidate, {
            "short_label": short_label,
            "long_label": long_label,
            "short_similarity": float(short_similarity),
            "long_similarity": float(long_similarity),
            "short_margin": short_margin,
            "candidate_gain": candidate_gain,
            "incumbent_loss": incumbent_loss,
            "count": self._crossover_count,
            "qualified": bool(qualifies),
        }

    def _predict(self, states: Sequence[str]) -> dict[str, float]:
        count = len(states)
        if count <= 1:
            return {states[0]: 1.0}
        stay = float(self.config.stay_probability)
        prior = {state: max(0.0, float(self._posterior.get(state, 0.0))) for state in states}
        total = sum(prior.values())
        if total <= 0.0:
            prior = {state: 1.0 / count for state in states}
        else:
            prior = {state: value / total for state, value in prior.items()}
        return {
            state: stay * prior[state] + (1.0 - stay) * (1.0 - prior[state]) / (count - 1)
            for state in states
        }

    def _observe(self, similarities: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        temperature = float(self.config.similarity_temperature)
        top_similarity = max(similarities.values(), default=-1.0)
        profile_count = max(1, len(self._profiles))
        unknown_bias = float(self.config.unknown_bias)
        if int(self.config.profile_count_bias_threshold) > 0:
            unknown_bias = (
                float(self.config.low_profile_unknown_bias)
                if profile_count <= int(self.config.profile_count_bias_threshold)
                else float(self.config.high_profile_unknown_bias)
            )
        unknown_bias += float(self.config.profile_count_unknown_bias_slope) * math.log(profile_count)
        emission_logits = {
            label: (float(value) - float(self.config.min_similarity)) / temperature
            for label, value in similarities.items()
        }
        maturity_strength = float(self.config.profile_maturity_logit_strength)
        if maturity_strength != 0.0 and emission_logits:
            pseudo = float(self.config.profile_maturity_pseudoseconds)
            maturity = {
                label: math.log(pseudo + max(0.0, float(profile.speech_seconds)))
                for label, profile in self._profiles.items()
            }
            peak_maturity = max(maturity.values())
            for label in emission_logits:
                emission_logits[label] += maturity_strength * (maturity[label] - peak_maturity)
        emission_logits["unknown"] = (
            (float(self.config.min_similarity) - top_similarity) / temperature
            + unknown_bias
        )
        raw = _softmax(emission_logits)
        states = ["unknown", *sorted(self._profiles)]
        predicted = self._predict(states)
        posterior_logits = {
            state: (
                float(self.config.evidence_strength) * math.log(max(1e-12, raw.get(state, 1e-12)))
                + float(self.config.prior_strength) * math.log(max(1e-12, predicted.get(state, 1e-12)))
            )
            for state in states
        }
        posterior = _softmax(posterior_logits)
        self._posterior = posterior
        return raw, posterior

    def step(self, item: MultiScaleStep) -> LiveSpeakerDecision:
        media_time = float(item.media_time)
        if media_time + 1e-9 < self._last_media_time:
            raise ValueError("Bayesian live-speaker steps must be chronological")
        self._last_media_time = media_time
        applied = self._apply_profile_events(media_time)
        expired_profiles = self._expire_provisional_profiles(media_time)
        previous = self._visible
        similarities: dict[str, float] = {}
        raw: dict[str, float] = {"unknown": 1.0}
        posterior = dict(self._posterior)
        candidate: str | None = None
        crossover_diagnostics: dict[str, Any] = {}
        incumbent_continuity: float | None = None
        boundary_short_only = False
        boundary_residual_continuity: float | None = None
        if (
            not item.probe_scheduled
            or item.release_signal
            or not item.speech
            or len(item.evidences) != 2
        ):
            self._crossover_label = None
            self._crossover_count = 0
        if not item.probe_scheduled:
            self._pending_provisional_embeddings.clear()
            if item.evidences:
                raise ValueError("A non-probe Bayesian tick may not carry evidence")
            if item.release_signal:
                self._silence_count += 1
                self._unknown_count = 0
                if self._visible and self._silence_count >= int(self.config.silence_release_count):
                    self._visible = None
                    self._posterior = {"unknown": 1.0}
                    action, reason = "clear", "release_gate"
                else:
                    action, reason = ("hold", "release_debounce") if self._visible else ("none", "release_gate")
            else:
                action, reason = ("hold", "non_probe_tick") if self._visible else ("none", "non_probe_tick")
        elif item.release_signal:
            self._pending_provisional_embeddings.clear()
            self._silence_count += 1
            self._unknown_count = 0
            if self._visible and self._silence_count >= int(self.config.silence_release_count):
                self._visible = None
                self._posterior = {"unknown": 1.0}
                action, reason = "clear", "release_gate"
            else:
                action, reason = ("hold", "release_debounce") if self._visible else ("none", "release_gate")
        elif not item.speech:
            self._pending_provisional_embeddings.clear()
            self._silence_count = 0
            self._unknown_count = 0
            action, reason = ("hold", "probe_gate_silence") if self._visible else ("none", "probe_gate_silence")
        elif not item.evidences or not self._profiles:
            self._silence_count = 0
            self._unknown_count += 1
            provisional = self._consider_provisional_profile(item, similarities)
            if provisional is not None:
                candidate = provisional
                self._unknown_count = 0
                self._visible = provisional
                self._remember_incumbent(provisional, item)
                posterior = dict(self._posterior)
                action, reason = ("switch", "provisional_acquire") if previous else ("acquire", "provisional_acquire")
            elif self._visible and self._unknown_count >= int(self.config.unknown_release_count):
                self._visible = None
                self._posterior = {"unknown": 1.0}
                action, reason = "clear", "unknown"
            else:
                action, reason = ("hold", "unknown_debounce") if self._visible else ("none", "unknown")
        else:
            self._silence_count = 0
            incumbent_continuity = self._incumbent_continuity(item)
            boundary_short_only_threshold = float(
                self.config.boundary_short_only_max_continuity
            )
            boundary_short_only = (
                boundary_short_only_threshold >= -0.999999
                and incumbent_continuity is not None
                and incumbent_continuity <= boundary_short_only_threshold
            )
            decision_item = item
            if boundary_short_only:
                decision_item, boundary_residual_continuity = (
                    self._boundary_residualized_item(item)
                )
            similarities = self._fused_similarities(
                decision_item.evidences,
                short_only=boundary_short_only,
            )
            raw, posterior = self._observe(similarities)
            ranked = sorted(
                ((label, float(probability)) for label, probability in posterior.items() if label != "unknown"),
                key=lambda pair: (-pair[1], pair[0]),
            )
            top_label, top_probability = ranked[0] if ranked else (None, 0.0)
            runner_probability = ranked[1][1] if len(ranked) > 1 else 0.0
            ordered_similarities = sorted(similarities.values(), reverse=True)
            similarity = similarities.get(top_label, -1.0) if top_label else -1.0
            similarity_margin = (
                ordered_similarities[0] - ordered_similarities[1]
                if len(ordered_similarities) > 1 else 1.0
            )
            scale_agreement = (
                float(np.dot(item.evidences[0].embedding, item.evidences[1].embedding))
                if len(item.evidences) == 2 else 1.0
            )
            top_profile = self._profiles.get(top_label) if top_label is not None else None
            provisional_assignment_allowed = (
                top_profile is None
                or not top_profile.provisional
                or scale_agreement >= float(
                    self.config.provisional_assignment_scale_agreement_min_similarity
                )
            )
            incumbent_hold_allowed = (
                top_label is None
                or top_label != previous
                or scale_agreement >= float(
                    self.config.incumbent_hold_scale_agreement_min_similarity
                )
            )
            candidate = top_label if (
                top_label is not None
                and provisional_assignment_allowed
                and incumbent_hold_allowed
                and top_probability >= float(self.config.min_known_probability)
                and top_probability >= float(posterior.get("unknown", 0.0))
                and top_probability - runner_probability >= float(self.config.switch_probability_margin)
                and similarity >= float(self.config.min_similarity)
                and similarity_margin >= float(self.config.min_margin)
            ) else None
            crossover_candidate, crossover_diagnostics = self._short_long_crossover_candidate(
                decision_item.evidences, previous
            )
            if crossover_candidate is not None:
                candidate = crossover_candidate
                self._posterior = {
                    state: (1.0 if state == candidate else 1e-9)
                    for state in ["unknown", *sorted(self._profiles)]
                }
                total = sum(self._posterior.values())
                self._posterior = {state: value / total for state, value in self._posterior.items()}
                posterior = dict(self._posterior)
            if candidate is None:
                incumbent_continuity = self._incumbent_continuity(item)
                continuity_hold = (
                    incumbent_continuity is not None
                    and incumbent_continuity >= float(
                        self.config.incumbent_continuity_min_similarity
                    )
                )
                if continuity_hold and self._visible is not None:
                    candidate = self._visible
                    self._unknown_count = 0
                    if self.config.incumbent_continuity_update_on_hold:
                        self._remember_incumbent(self._visible, item)
                    action, reason = "hold", "incumbent_continuity"
                else:
                    self._unknown_count += 1
                    provisional = self._consider_provisional_profile(decision_item, similarities)
                if not continuity_hold and provisional is not None:
                    candidate = provisional
                    self._unknown_count = 0
                    self._visible = provisional
                    self._remember_incumbent(provisional, decision_item)
                    posterior = dict(self._posterior)
                    if previous is None:
                        action, reason = "acquire", "provisional_acquire"
                    elif previous != provisional:
                        action, reason = "switch", "provisional_acquire"
                    else:
                        action, reason = "hold", "provisional_acquire"
                elif not continuity_hold and self._visible and self._unknown_count >= int(self.config.unknown_release_count):
                    self._visible = None
                    action, reason = "clear", "bayes_unknown"
                elif not continuity_hold:
                    action, reason = ("hold", "bayes_unknown_debounce") if self._visible else ("none", "bayes_unknown")
            else:
                self._pending_provisional_embeddings.clear()
                self._unknown_count = 0
                self._visible = candidate
                adaptation_continuity = (
                    self._incumbent_continuity(item)
                    if candidate == previous else None
                )
                adaptation_item = item if candidate == previous else decision_item
                self._update_provisional_profile(
                    candidate,
                    adaptation_item,
                    continuity_similarity=adaptation_continuity,
                )
                self._remember_incumbent(candidate, adaptation_item)
                if previous is None:
                    action, reason = "acquire", "bayes_posterior"
                elif previous != candidate:
                    action, reason = "switch", "bayes_posterior"
                else:
                    action, reason = "hold", "bayes_posterior"

        return LiveSpeakerDecision(
            media_time=media_time,
            visible_speaker=self._visible,
            action=action,
            reason=reason,
            candidate_speaker=candidate,
            probabilities={str(key): float(value) for key, value in posterior.items()},
            raw_probabilities={str(key): float(value) for key, value in raw.items()},
            similarities={str(key): float(value) for key, value in similarities.items()},
            profile_count=len(self._profiles),
            profile_generations={label: profile.generation for label, profile in self._profiles.items()},
            diagnostics={
                "algorithm_id": BAYES_ALGORITHM_ID,
                "profile_events_applied": applied,
                "provisional_profiles_expired": expired_profiles,
                "unknown_count": self._unknown_count,
                "silence_count": self._silence_count,
                "probe_scheduled": bool(item.probe_scheduled),
                "release_signal": bool(item.release_signal),
                "scale_windows": [float(evidence.window_seconds) for evidence in item.evidences],
                "scale_agreement": (
                    float(np.dot(item.evidences[0].embedding, item.evidences[1].embedding))
                    if len(item.evidences) == 2 else None
                ),
                "incumbent_hold_allowed": (
                    None if not item.probe_scheduled else (
                        candidate is not None
                        or not self._visible
                        or len(item.evidences) < 2
                        or float(np.dot(item.evidences[0].embedding, item.evidences[1].embedding))
                        >= float(self.config.incumbent_hold_scale_agreement_min_similarity)
                    )
                ),
                "incumbent_continuity": incumbent_continuity,
                "boundary_short_only": boundary_short_only,
                "boundary_residual_incumbent_alpha": float(
                    self.config.boundary_residual_incumbent_alpha
                ),
                "boundary_residual_continuity": boundary_residual_continuity,
                "profile_aliases": dict(self._profile_aliases),
                "short_long_crossover": crossover_diagnostics,
                "provisional_profiles": sorted(
                    label for label, profile in self._profiles.items() if profile.provisional
                ),
                "skipped_reason": item.skipped_reason,
            },
        )


def replay_cached_bayes_windows(
    blocks: Sequence[CachedLiveWindowBlock],
    profile_events: Iterable[SpeakerProfileEvent],
    speech_mask: Sequence[bool] | np.ndarray,
    probe_mask: Sequence[bool] | np.ndarray,
    release_mask: Sequence[bool] | np.ndarray,
    *,
    config: BayesSpeakerTrackerConfig | None = None,
    attack_probe_interval_seconds: float = 0.0,
) -> list[LiveSpeakerDecision]:
    if not blocks or len(blocks) > 2:
        raise ValueError("Bayesian replay requires one or two cached windows")
    ordered = sorted(blocks, key=lambda block: float(block.window_seconds))
    reference = ordered[0]
    for block in ordered[1:]:
        if block.video_id != reference.video_id or not np.array_equal(block.media_times, reference.media_times):
            raise ValueError("Bayesian replay blocks must share video and timeline")
    rows = int(reference.media_times.shape[0])
    speech = np.asarray(speech_mask, dtype=bool).reshape(-1)
    probes = np.asarray(probe_mask, dtype=bool).reshape(-1)
    releases = np.asarray(release_mask, dtype=bool).reshape(-1)
    if any(values.shape[0] != rows for values in (speech, probes, releases)):
        raise ValueError("Bayesian replay masks must match the cached timeline")
    tracker = CausalBayesSpeakerTracker(config=config, profile_events=profile_events)
    decisions: list[LiveSpeakerDecision] = []
    attack_interval = max(0.0, float(attack_probe_interval_seconds))
    last_probe_time = -math.inf
    for index, media_time in enumerate(reference.media_times):
        release_signal = bool(releases[index])
        scheduled = bool(probes[index]) and not release_signal
        if (
            not scheduled
            and not release_signal
            and attack_interval > 0.0
            and tracker.visible_speaker is None
            and float(media_time) + 1e-9 >= last_probe_time + attack_interval
        ):
            scheduled = True
        if scheduled:
            last_probe_time = float(media_time)
        evidences = tuple(
            MultiScaleEvidence(float(block.window_seconds), block.embeddings[index])
            for block in ordered
            if scheduled and bool(block.valid[index])
        )
        decisions.append(tracker.step(MultiScaleStep(
            media_time=float(media_time),
            speech=bool(speech[index]),
            evidences=evidences,
            probe_scheduled=scheduled,
            release_signal=release_signal,
            skipped_reason=("" if evidences else "not_a_scheduled_probe" if not scheduled else "cached_embeddings_invalid"),
        )))
    return decisions
