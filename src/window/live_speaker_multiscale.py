"""Experimental causal multi-scale live-speaker tracking.

Unlike the production dual-window path, this module never averages embeddings from
different durations.  It treats every window as an independent sensor, scores all
sensors against the same causal profiles, and fuses speaker evidence afterwards.
It also supports a short-vs-long crossover detector, bounded similarity history,
and provisional online profiles which can later absorb final sentence profiles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from speakers.speaker_embedding_cluster import normalize_vector
from window.live_speaker_algorithm import LiveSpeakerDecision, SpeakerProfileEvent
from window.live_speaker_replay import CachedLiveWindowBlock


MULTISCALE_ALGORITHM_ID = "causal_multiscale_similarity_tracker_v6"

TRACKING_OFF = "OFF"
TRACKING_STABLE = "STABLE"
TRACKING_TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class MultiScaleTrackerConfig:
    scale_windows: tuple[float, ...] = ()
    scale_weights: tuple[float, ...] = ()
    min_similarity: float = 0.35
    min_margin: float = 0.08
    acquire_scale_agreement: int = 1
    min_scale_agreement: int = 2
    enable_consensus: bool = True
    enable_crossover: bool = True
    enable_history: bool = True
    crossover_short_advantage: float = 0.05
    crossover_scale_gap: float = 0.12
    crossover_required: int = 2
    consensus_advantage: float = 0.04
    history_size: int = 3
    history_required: int = 2
    history_advantage: float = 0.04
    history_short_weight: float = 0.5
    history_statistic: str = "mean"
    history_max_gap_seconds: float = 1.25
    enable_transition_abstention: bool = False
    transition_short_advantage: float = 0.03
    transition_scale_gap: float = 0.05
    transition_clear_required: int = 1
    transition_min_valid_scales: int = 2
    transition_incumbent_max_similarity: float = 0.35
    transition_incumbent_drop: float = 0.12
    transition_incumbent_history_size: int = 3
    transition_incumbent_clear_required: int = 1
    transition_fast_scale_count: int = 1
    transition_slow_scale_count: int = 1
    transition_min_similarity: float = 0.30
    transition_min_margin: float = 0.05
    transition_acquire_history_size: int = 3
    transition_acquire_required: int = 2
    transition_revert_required: int = 1
    transition_min_off_probes: int = 1
    transition_timeout_seconds: float = 3.0
    enable_transition_embedding_change: bool = False
    transition_embedding_history_size: int = 5
    transition_embedding_min_history: int = 2
    transition_embedding_max_similarity: float = 0.55
    transition_embedding_drop: float = 0.20
    transition_embedding_clear_required: int = 1
    enable_transition_speech_gate: bool = False
    transition_speech_gate_clear_required: int = 1
    enable_duration_matched_profiles: bool = False
    duration_profile_score_weight: float = 0.5
    duration_profile_update_alpha: float = 0.25
    duration_profile_min_windows: int = 2
    duration_profile_max_windows_per_sentence: int = 16
    duration_profile_min_cohesion: float = 0.45
    duration_profile_guard_seconds: float = 0.05
    duration_profile_buffer_seconds: float = 120.0
    unknown_release_count: int = 2
    silence_release_count: int = 2
    enable_online_profiles: bool = False
    provisional_first_immediate: bool = True
    provisional_max_existing_similarity: float = 0.28
    provisional_confirm_similarity: float = 0.52
    provisional_confirm_count: int = 2
    provisional_scale_consistency: float = 0.60
    provisional_update_alpha: float = 0.08
    official_merge_similarity: float = 0.42
    official_merge_weight: float = 0.80
    trusted_profile_min_sentence_count: int = 1
    trusted_profile_min_speech_seconds: float = 0.0
    max_profiles: int = 12

    def __post_init__(self) -> None:
        if int(self.acquire_scale_agreement) < 1 or int(self.min_scale_agreement) < 1:
            raise ValueError("scale agreement counts must be positive")
        if int(self.history_size) < 1 or int(self.history_required) < 1:
            raise ValueError("history counts must be positive")
        if int(self.history_required) > int(self.history_size):
            raise ValueError("history_required may not exceed history_size")
        if int(self.crossover_required) < 1 or int(self.crossover_required) > int(self.history_size):
            raise ValueError("crossover_required must be within history_size")
        if int(self.transition_clear_required) < 1 or int(self.transition_min_valid_scales) < 2:
            raise ValueError("transition abstention counts are invalid")
        if int(self.transition_incumbent_history_size) < 1 or int(self.transition_incumbent_clear_required) < 1:
            raise ValueError("transition incumbent-history counts are invalid")
        if int(self.transition_fast_scale_count) < 1 or int(self.transition_slow_scale_count) < 1:
            raise ValueError("transition scale-bank counts must be positive")
        if int(self.transition_acquire_history_size) < 1 or int(self.transition_acquire_required) < 1:
            raise ValueError("transition acquisition-history counts are invalid")
        if int(self.transition_acquire_required) > int(self.transition_acquire_history_size):
            raise ValueError("transition_acquire_required may not exceed its history size")
        if int(self.transition_revert_required) < 1 or int(self.transition_min_off_probes) < 1:
            raise ValueError("transition state counts must be positive")
        if float(self.transition_timeout_seconds) <= 0.0:
            raise ValueError("transition_timeout_seconds must be positive")
        if int(self.transition_embedding_history_size) < 1 or int(self.transition_embedding_min_history) < 1:
            raise ValueError("transition embedding-history counts must be positive")
        if int(self.transition_embedding_min_history) > int(self.transition_embedding_history_size):
            raise ValueError("transition embedding minimum may not exceed its history size")
        if int(self.transition_embedding_clear_required) < 1:
            raise ValueError("transition_embedding_clear_required must be positive")
        if int(self.transition_speech_gate_clear_required) < 1:
            raise ValueError("transition_speech_gate_clear_required must be positive")
        if int(self.duration_profile_min_windows) < 1:
            raise ValueError("duration_profile_min_windows must be positive")
        if int(self.duration_profile_max_windows_per_sentence) < int(self.duration_profile_min_windows):
            raise ValueError("duration profile maximum must cover the minimum")
        if float(self.duration_profile_guard_seconds) < 0.0 or float(self.duration_profile_buffer_seconds) <= 0.0:
            raise ValueError("duration profile timing values are invalid")
        if float(self.history_max_gap_seconds) <= 0.0:
            raise ValueError("history_max_gap_seconds must be positive")
        if not 0.0 <= float(self.history_short_weight) <= 1.0:
            raise ValueError("history_short_weight must be in [0, 1]")
        if self.history_statistic not in {"mean", "median"}:
            raise ValueError("history_statistic must be 'mean' or 'median'")
        if self.scale_windows and len(self.scale_windows) != len(self.scale_weights):
            raise ValueError("scale_windows and scale_weights must have equal lengths")
        if int(self.unknown_release_count) < 1 or int(self.silence_release_count) < 1:
            raise ValueError("release counts must be positive")
        if int(self.provisional_confirm_count) < 1 or int(self.max_profiles) < 1:
            raise ValueError("profile counts must be positive")
        if int(self.trusted_profile_min_sentence_count) < 1:
            raise ValueError("trusted_profile_min_sentence_count must be positive")
        if float(self.trusted_profile_min_speech_seconds) < 0.0:
            raise ValueError("trusted_profile_min_speech_seconds must be non-negative")
        for name in (
            "provisional_update_alpha",
            "official_merge_weight",
            "duration_profile_score_weight",
            "duration_profile_update_alpha",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class MultiScaleEvidence:
    window_seconds: float
    embedding: np.ndarray

    def __post_init__(self) -> None:
        if float(self.window_seconds) <= 0.0:
            raise ValueError("window_seconds must be positive")
        object.__setattr__(self, "embedding", normalize_vector(self.embedding))


@dataclass(frozen=True)
class MultiScaleStep:
    media_time: float
    speech: bool
    evidences: tuple[MultiScaleEvidence, ...] = ()
    probe_scheduled: bool = True
    release_signal: bool = False
    skipped_reason: str = ""


@dataclass
class _Profile:
    label: str
    centroid: np.ndarray
    provisional: bool
    sentence_count: int = 0
    speech_seconds: float = 0.0
    trusted: bool = True
    scale_centroids: dict[float, np.ndarray] = field(default_factory=dict)
    scale_sentence_counts: dict[float, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _BufferedEvidence:
    media_time: float
    window_seconds: float
    embedding: np.ndarray


class CausalMultiScaleSpeakerTracker:
    """Causal speaker tracker over separate short/medium/long similarities."""

    def __init__(
        self,
        config: MultiScaleTrackerConfig | None = None,
        profile_events: Iterable[SpeakerProfileEvent] = (),
    ) -> None:
        self.config = config or MultiScaleTrackerConfig()
        self._events = sorted(
            list(profile_events),
            key=lambda item: (float(item.available_at), int(item.generation), str(item.speaker_id)),
        )
        self._next_event = 0
        self._profiles: dict[str, _Profile] = {}
        self._aliases: dict[str, str] = {}
        self._generations: dict[str, int] = {}
        self._visible: str | None = None
        self._last_media_time = -1.0
        self._unknown_count = 0
        self._silence_count = 0
        self._challenger: str | None = None
        self._short_advantages: deque[float] = deque(maxlen=self.config.history_size)
        self._long_advantages: deque[float] = deque(maxlen=self.config.history_size)
        self._fused_advantages: deque[float] = deque(maxlen=self.config.history_size)
        self._challenger_last_media_time: float | None = None
        self._pending_vector: np.ndarray | None = None
        self._pending_count = 0
        self._pending_last_media_time: float | None = None
        self._next_provisional_index = 1
        self._evidence_buffer: deque[_BufferedEvidence] = deque()
        self._incumbent_history_speaker: str | None = None
        self._incumbent_short_similarities: deque[float] = deque(
            maxlen=self.config.transition_incumbent_history_size
        )
        self._incumbent_rejection_count = 0
        self._incumbent_last_media_time: float | None = None
        self._tracking_state = TRACKING_OFF
        self._transition_incumbent: str | None = None
        self._transition_candidate: str | None = None
        self._transition_candidate_votes: deque[str | None] = deque(
            maxlen=self.config.transition_acquire_history_size
        )
        self._transition_revert_count = 0
        self._transition_started_at: float | None = None
        self._transition_last_probe_at: float | None = None
        self._transition_off_probes = 0
        self._transition_entry_kind = ""
        self._transition_entry_candidate: str | None = None
        self._transition_entry_count = 0
        self._transition_entry_last_at: float | None = None
        self._stable_fast_history_speaker: str | None = None
        self._stable_fast_embeddings: deque[np.ndarray] = deque(
            maxlen=self.config.transition_embedding_history_size
        )
        self._stable_fast_last_at: float | None = None
        self._embedding_change_count = 0
        self._speech_gate_false_probe_count = 0

    @property
    def visible_speaker(self) -> str | None:
        return self._visible

    @property
    def tracking_state(self) -> str:
        return self._tracking_state

    def _new_provisional_label(self) -> str:
        while True:
            label = f"P{self._next_provisional_index}"
            self._next_provisional_index += 1
            if label not in self._profiles:
                return label

    def _merge_vectors(self, left: np.ndarray, right: np.ndarray, right_weight: float) -> np.ndarray:
        weight = max(0.0, min(1.0, float(right_weight)))
        return normalize_vector((1.0 - weight) * left + weight * right)

    def _profile_event_is_mature(self, event: SpeakerProfileEvent) -> bool:
        return (
            int(event.sentence_count)
            >= int(self.config.trusted_profile_min_sentence_count)
            and float(event.speech_seconds)
            >= float(self.config.trusted_profile_min_speech_seconds)
        )

    def _apply_profile_events(self, media_time: float) -> list[str]:
        applied: list[str] = []
        while self._next_event < len(self._events):
            event = self._events[self._next_event]
            if float(event.available_at) > media_time + 1e-9:
                break
            source_label = str(event.speaker_id)
            previous_generation = self._generations.get(source_label, -1)
            if int(event.generation) > previous_generation:
                self._generations[source_label] = int(event.generation)
                first_official_profile = not any(
                    not item.provisional for item in self._profiles.values()
                )
                target = self._aliases.get(source_label)
                if target is None and source_label in self._profiles:
                    target = source_label
                if target is None:
                    provisional = [item for item in self._profiles.values() if item.provisional]
                    if provisional:
                        best = max(
                            provisional,
                            key=lambda item: float(np.dot(item.centroid, event.centroid)),
                        )
                        similarity = float(np.dot(best.centroid, event.centroid))
                        if similarity >= float(self.config.official_merge_similarity):
                            target = best.label
                            self._aliases[source_label] = target
                if target is None:
                    target = source_label
                    self._profiles[target] = _Profile(
                        label=target,
                        centroid=normalize_vector(event.centroid),
                        provisional=False,
                        sentence_count=int(event.sentence_count),
                        speech_seconds=float(event.speech_seconds),
                        trusted=(
                            first_official_profile
                            or self._profile_event_is_mature(event)
                        ),
                    )
                    self._aliases[source_label] = target
                elif target in self._profiles:
                    profile = self._profiles[target]
                    if profile.provisional:
                        profile.centroid = self._merge_vectors(
                            profile.centroid,
                            event.centroid,
                            self.config.official_merge_weight,
                        )
                    else:
                        # Profile events are complete generation snapshots, not deltas.
                        profile.centroid = normalize_vector(event.centroid)
                    profile.provisional = False
                    profile.sentence_count = max(
                        int(profile.sentence_count), int(event.sentence_count)
                    )
                    profile.speech_seconds = max(
                        float(profile.speech_seconds), float(event.speech_seconds)
                    )
                    if first_official_profile or self._profile_event_is_mature(event):
                        profile.trusted = True
                self._seed_duration_profiles(self._profiles[target], event)
                applied.append(f"{source_label}->{target}")
            self._next_event += 1
        return applied

    def _weights(self, evidences: Sequence[MultiScaleEvidence]) -> np.ndarray:
        count = len(evidences)
        configured = tuple(float(value) for value in self.config.scale_weights)
        if configured:
            if self.config.scale_windows:
                by_window = {
                    round(float(window), 6): weight
                    for window, weight in zip(self.config.scale_windows, configured)
                }
                try:
                    values = np.asarray(
                        [by_window[round(float(item.window_seconds), 6)] for item in evidences],
                        dtype=np.float64,
                    )
                except KeyError as error:
                    raise ValueError(
                        f"no configured weight for evidence window {float(error.args[0]):g}"
                    ) from error
            elif len(configured) == count:
                values = np.asarray(configured, dtype=np.float64)
            else:
                raise ValueError(
                    "partial scale evidence requires scale_windows alongside scale_weights"
                )
        else:
            values = np.ones(count, dtype=np.float64)
        values = np.maximum(values, 0.0)
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("at least one scale weight must be positive")
        return values / total

    def _similarities(self, evidence: MultiScaleEvidence) -> dict[str, float]:
        values: dict[str, float] = {}
        scale_key = round(float(evidence.window_seconds), 3)
        weight = float(self.config.duration_profile_score_weight)
        for label, profile in self._profiles.items():
            if not profile.trusted:
                continue
            generic = float(np.dot(evidence.embedding, profile.centroid))
            matched = profile.scale_centroids.get(scale_key)
            if self.config.enable_duration_matched_profiles and matched is not None:
                scale_similarity = float(np.dot(evidence.embedding, matched))
                values[label] = (1.0 - weight) * generic + weight * scale_similarity
            else:
                values[label] = generic
        return values

    def _buffer_evidences(self, media_time: float, item: MultiScaleStep) -> None:
        cutoff = media_time - float(self.config.duration_profile_buffer_seconds)
        while self._evidence_buffer and self._evidence_buffer[0].media_time < cutoff - 1e-9:
            self._evidence_buffer.popleft()
        if not item.probe_scheduled or item.release_signal or not item.speech:
            return
        for evidence in item.evidences:
            self._evidence_buffer.append(_BufferedEvidence(
                media_time=media_time,
                window_seconds=float(evidence.window_seconds),
                embedding=evidence.embedding.copy(),
            ))

    def _robust_sentence_centroid(self, values: Sequence[np.ndarray]) -> np.ndarray | None:
        minimum = int(self.config.duration_profile_min_windows)
        if len(values) < minimum:
            return None
        maximum = int(self.config.duration_profile_max_windows_per_sentence)
        selected = list(values)
        if len(selected) > maximum:
            indices = np.linspace(0, len(selected) - 1, num=maximum, dtype=int)
            selected = [selected[int(index)] for index in indices]
        matrix = np.stack(selected).astype(np.float64, copy=False)
        similarities = matrix @ matrix.T
        medoid_index = int(np.argmax(np.mean(similarities, axis=1)))
        keep = similarities[medoid_index] >= float(self.config.duration_profile_min_cohesion)
        if int(np.count_nonzero(keep)) < minimum:
            return None
        try:
            return normalize_vector(np.mean(matrix[keep], axis=0))
        except ValueError:
            return None

    def _seed_duration_profiles(self, profile: _Profile, event: SpeakerProfileEvent) -> None:
        if not self.config.enable_duration_matched_profiles:
            return
        if event.sentence_start is None or event.sentence_end is None:
            return
        start = float(event.sentence_start) + float(self.config.duration_profile_guard_seconds)
        end = float(event.sentence_end) - float(self.config.duration_profile_guard_seconds)
        if end <= start:
            return
        by_scale: dict[float, list[np.ndarray]] = {}
        for buffered in self._evidence_buffer:
            if buffered.media_time > end + 1e-9:
                continue
            if buffered.media_time - buffered.window_seconds < start - 1e-9:
                continue
            key = round(float(buffered.window_seconds), 3)
            by_scale.setdefault(key, []).append(buffered.embedding)
        for key, values in by_scale.items():
            sentence_centroid = self._robust_sentence_centroid(values)
            if sentence_centroid is None:
                continue
            previous = profile.scale_centroids.get(key)
            if previous is None:
                profile.scale_centroids[key] = sentence_centroid
                profile.scale_sentence_counts[key] = 1
            else:
                profile.scale_centroids[key] = self._merge_vectors(
                    previous,
                    sentence_centroid,
                    self.config.duration_profile_update_alpha,
                )
                profile.scale_sentence_counts[key] = profile.scale_sentence_counts.get(key, 1) + 1

    @staticmethod
    def _top(similarities: dict[str, float]) -> tuple[str | None, float, float]:
        if not similarities:
            return None, -1.0, -1.0
        ordered = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        label, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else -1.0
        return label, float(top), float(top - second if len(ordered) > 1 else 1.0)

    def _probabilities(self, fused: dict[str, float]) -> dict[str, float]:
        if not fused:
            return {"unknown": 1.0}
        logits = {label: (value - self.config.min_similarity) / 0.075 for label, value in fused.items()}
        logits["unknown"] = 0.0
        peak = max(logits.values())
        masses = {key: math.exp(max(-40.0, min(40.0, value - peak))) for key, value in logits.items()}
        total = sum(masses.values())
        return {key: value / total for key, value in masses.items()}

    def _reset_challenger(self) -> None:
        self._challenger = None
        self._short_advantages.clear()
        self._long_advantages.clear()
        self._fused_advantages.clear()
        self._challenger_last_media_time = None

    def _reset_pending_profile(self) -> None:
        self._pending_vector = None
        self._pending_count = 0
        self._pending_last_media_time = None

    def _reset_incumbent_history(self) -> None:
        self._incumbent_history_speaker = None
        self._incumbent_short_similarities.clear()
        self._incumbent_rejection_count = 0
        self._incumbent_last_media_time = None

    def _reset_transition_entry(self) -> None:
        self._transition_entry_candidate = None
        self._transition_entry_count = 0
        self._transition_entry_last_at = None

    def _reset_stable_fast_history(self) -> None:
        self._stable_fast_history_speaker = None
        self._stable_fast_embeddings.clear()
        self._stable_fast_last_at = None
        self._embedding_change_count = 0

    def _reset_speech_gate_history(self) -> None:
        self._speech_gate_false_probe_count = 0

    def _seed_stable_fast_history(self, speaker: str, embedding: np.ndarray) -> None:
        if not self.config.enable_transition_embedding_change:
            return
        if self._stable_fast_history_speaker != speaker:
            self._reset_stable_fast_history()
            self._stable_fast_history_speaker = speaker
        self._stable_fast_embeddings.append(normalize_vector(embedding))
        self._stable_fast_last_at = self._last_media_time
        self._embedding_change_count = 0

    def _embedding_change_signal(
        self,
        speaker: str,
        embedding: np.ndarray,
    ) -> tuple[bool, float, float, float]:
        if not (
            self.config.enable_transition_abstention
            and self.config.enable_transition_embedding_change
        ):
            return False, -1.0, -1.0, -1.0
        if (
            self._stable_fast_history_speaker != speaker
            or (
                self._stable_fast_last_at is not None
                and self._last_media_time - self._stable_fast_last_at
                > float(self.config.history_max_gap_seconds) + 1e-9
            )
        ):
            self._reset_stable_fast_history()
            self._stable_fast_history_speaker = speaker
            return False, -1.0, -1.0, -1.0
        if len(self._stable_fast_embeddings) < int(self.config.transition_embedding_min_history):
            return False, -1.0, -1.0, -1.0
        matrix = np.stack(tuple(self._stable_fast_embeddings)).astype(np.float64, copy=False)
        try:
            reference = normalize_vector(np.mean(matrix, axis=0))
        except ValueError:
            self._reset_stable_fast_history()
            self._stable_fast_history_speaker = speaker
            return False, -1.0, -1.0, -1.0
        history_similarities = matrix @ reference
        history_similarity = float(np.median(history_similarities))
        current_similarity = float(np.dot(normalize_vector(embedding), reference))
        drop = history_similarity - current_similarity
        signal = (
            current_similarity <= float(self.config.transition_embedding_max_similarity)
            and drop >= float(self.config.transition_embedding_drop)
        )
        self._embedding_change_count = self._embedding_change_count + 1 if signal else 0
        return (
            signal and self._embedding_change_count
            >= int(self.config.transition_embedding_clear_required),
            current_similarity,
            history_similarity,
            drop,
        )

    def _reset_transition_evidence(self) -> None:
        self._transition_candidate = None
        self._transition_candidate_votes.clear()
        self._transition_revert_count = 0
        self._transition_last_probe_at = None

    def _clear_transition_context(self, *, reset_entry: bool = True) -> None:
        self._transition_incumbent = None
        self._transition_started_at = None
        self._transition_off_probes = 0
        self._transition_entry_kind = ""
        self._reset_transition_evidence()
        if reset_entry:
            self._reset_transition_entry()

    def _set_tracking_off(self) -> None:
        self._tracking_state = TRACKING_OFF
        self._visible = None
        self._clear_transition_context()
        self._reset_stable_fast_history()
        self._reset_speech_gate_history()

    def _set_tracking_stable(self, speaker: str, *, reset_entry: bool = True) -> None:
        if (
            self._stable_fast_history_speaker is not None
            and self._stable_fast_history_speaker != speaker
        ):
            self._reset_stable_fast_history()
        self._tracking_state = TRACKING_STABLE
        self._visible = speaker
        self._clear_transition_context(reset_entry=reset_entry)
        self._reset_speech_gate_history()

    def _enter_transition(
        self,
        incumbent: str,
        *,
        candidate: str | None,
        kind: str,
    ) -> None:
        self._tracking_state = TRACKING_TRANSITION
        self._visible = None
        self._transition_incumbent = incumbent
        self._transition_started_at = self._last_media_time
        self._transition_off_probes = 0
        self._transition_entry_kind = kind
        self._reset_speech_gate_history()
        self._reset_transition_evidence()
        if candidate is not None:
            self._transition_candidate = candidate
            self._transition_candidate_votes.append(candidate)
            self._transition_last_probe_at = self._last_media_time
        self._reset_transition_entry()
        self._reset_challenger()

    def _transition_timed_out(self) -> bool:
        return (
            self._tracking_state == TRACKING_TRANSITION
            and self._transition_started_at is not None
            and self._last_media_time - self._transition_started_at
            > float(self.config.transition_timeout_seconds) + 1e-9
        )

    def _record_transition_entry(self, candidate: str | None, signal: bool) -> bool:
        gap = (
            self._transition_entry_last_at is not None
            and self._last_media_time - self._transition_entry_last_at
            > float(self.config.history_max_gap_seconds) + 1e-9
        )
        if not signal or candidate is None:
            self._reset_transition_entry()
            return False
        if gap or self._transition_entry_candidate != candidate:
            self._transition_entry_candidate = candidate
            self._transition_entry_count = 0
        self._transition_entry_count += 1
        self._transition_entry_last_at = self._last_media_time
        return self._transition_entry_count >= int(self.config.transition_clear_required)

    @staticmethod
    def _bank_similarities(
        scale_similarities: Sequence[dict[str, float]],
        scale_weights: Sequence[float],
        indices: Sequence[int],
    ) -> dict[str, float]:
        if not indices:
            return {}
        labels = sorted({label for index in indices for label in scale_similarities[index]})
        weights = np.asarray([float(scale_weights[index]) for index in indices], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(len(indices), dtype=np.float64) / len(indices)
        else:
            weights /= total
        return {
            label: float(sum(
                weights[offset] * scale_similarities[index].get(label, -1.0)
                for offset, index in enumerate(indices)
            ))
            for label in labels
        }

    def _transition_banks(
        self,
        scale_similarities: Sequence[dict[str, float]],
        scale_weights: Sequence[float],
    ) -> tuple[dict[str, float], dict[str, float], list[int], list[int]]:
        count = len(scale_similarities)
        fast_count = min(count, int(self.config.transition_fast_scale_count))
        slow_count = min(count, int(self.config.transition_slow_scale_count))
        fast_indices = list(range(fast_count))
        slow_indices = list(range(count - slow_count, count))
        return (
            self._bank_similarities(scale_similarities, scale_weights, fast_indices),
            self._bank_similarities(scale_similarities, scale_weights, slow_indices),
            fast_indices,
            slow_indices,
        )

    def _advance_transition(
        self,
        fast_embedding: np.ndarray,
        scale_similarities: Sequence[dict[str, float]],
        scale_tops: Sequence[tuple[str | None, float, float]],
        weights: Sequence[float],
        fused: dict[str, float],
        fused_label: str | None,
        fused_similarity: float,
        fused_margin: float,
        diagnostics: dict[str, Any],
    ) -> tuple[str | None, str, dict[str, float], dict[str, Any]]:
        incumbent = self._transition_incumbent
        if incumbent is None or self._transition_timed_out():
            diagnostics["transition_timeout"] = True
            self._set_tracking_off()
            return None, "transition_timeout", fused, diagnostics

        if (
            self._transition_last_probe_at is not None
            and self._last_media_time - self._transition_last_probe_at
            > float(self.config.history_max_gap_seconds) + 1e-9
        ):
            self._reset_transition_evidence()

        fast, slow, fast_indices, slow_indices = self._transition_banks(
            scale_similarities, weights
        )
        fast_label, fast_similarity, fast_margin = self._top(fast)
        slow_label, slow_similarity, _slow_margin = self._top(slow)
        minimum_similarity = float(self.config.transition_min_similarity)
        minimum_margin = float(self.config.transition_min_margin)
        valid_support = {
            label: sum(
                1
                for top_label, similarity, _margin in scale_tops
                if top_label == label and similarity >= minimum_similarity
            )
            for label in self._profiles
        }
        fast_advantage = (
            float(fast.get(fast_label, -1.0) - fast.get(incumbent, -1.0))
            if fast_label is not None else -1.0
        )
        strong_challenger = (
            fast_label is not None
            and fast_label != incumbent
            and fast_similarity >= minimum_similarity
            and fast_margin >= minimum_margin
            and fast_advantage >= float(self.config.transition_short_advantage)
        )
        self._transition_off_probes += 1
        if strong_challenger:
            if self._transition_candidate != fast_label:
                self._transition_candidate = fast_label
                self._transition_candidate_votes.clear()
            self._transition_candidate_votes.append(fast_label)
        else:
            self._transition_candidate_votes.append(None)
        self._transition_last_probe_at = self._last_media_time

        candidate = self._transition_candidate
        candidate_votes = (
            sum(1 for value in self._transition_candidate_votes if value == candidate)
            if candidate is not None else 0
        )
        candidate_multiscale = (
            candidate is not None
            and fused_label == candidate
            and fused_similarity >= minimum_similarity
            and fused_margin >= minimum_margin
            and valid_support.get(candidate, 0)
            >= int(self.config.transition_min_valid_scales)
        )
        trusted_candidate = (
            strong_challenger
            and candidate == fast_label
            and (
                candidate_votes >= int(self.config.transition_acquire_required)
                or candidate_multiscale
            )
            and self._transition_off_probes >= int(self.config.transition_min_off_probes)
        )
        incumbent_trusted = (
            fast_label == incumbent
            and fast_similarity >= minimum_similarity
            and fast_margin >= minimum_margin
            and fused_label == incumbent
            and fused_similarity >= minimum_similarity
            and fused_margin >= minimum_margin
            and valid_support.get(incumbent, 0)
            >= int(self.config.transition_min_valid_scales)
        )
        if incumbent_trusted:
            self._transition_revert_count += 1
        else:
            self._transition_revert_count = 0

        diagnostics.update({
            "tracking_state": TRACKING_TRANSITION,
            "transition_incumbent": incumbent,
            "transition_candidate": candidate,
            "transition_entry_kind": self._transition_entry_kind,
            "transition_age_seconds": float(
                self._last_media_time
                - (
                    self._transition_started_at
                    if self._transition_started_at is not None
                    else self._last_media_time
                )
            ),
            "transition_off_probes": self._transition_off_probes,
            "transition_candidate_votes": candidate_votes,
            "transition_revert_count": self._transition_revert_count,
            "transition_fast_indices": fast_indices,
            "transition_slow_indices": slow_indices,
            "transition_fast_label": fast_label,
            "transition_slow_label": slow_label,
            "transition_fast_similarity": fast_similarity,
            "transition_slow_similarity": slow_similarity,
            "transition_fast_advantage": fast_advantage,
            "transition_candidate_multiscale": candidate_multiscale,
            "transition_strong_challenger": strong_challenger,
            "transition_incumbent_trusted": incumbent_trusted,
        })
        if (
            incumbent_trusted
            and self._transition_revert_count >= int(self.config.transition_revert_required)
        ):
            self._set_tracking_stable(incumbent)
            self._seed_stable_fast_history(incumbent, fast_embedding)
            return incumbent, "transition_false_alarm_revert", fused, diagnostics
        if trusted_candidate and candidate is not None:
            self._set_tracking_stable(candidate)
            self._seed_stable_fast_history(candidate, fast_embedding)
            return candidate, (
                "transition_multiscale_acquire"
                if candidate_multiscale else "transition_history_acquire"
            ), fused, diagnostics
        return None, "transition_wait", fused, diagnostics

    def _incumbent_rejection(
        self,
        current: str,
        short_similarity: float,
        long_similarity: float,
    ) -> tuple[bool, float, float]:
        if (
            self._incumbent_history_speaker != current
            or (
                self._incumbent_last_media_time is not None
                and self._last_media_time - self._incumbent_last_media_time
                > float(self.config.history_max_gap_seconds) + 1e-9
            )
        ):
            self._reset_incumbent_history()
            self._incumbent_history_speaker = current
        historical = (
            float(np.median(np.asarray(self._incumbent_short_similarities, dtype=np.float64)))
            if self._incumbent_short_similarities
            else -1.0
        )
        reference = max(float(long_similarity), historical)
        drop = reference - float(short_similarity)
        signal = (
            self.config.enable_transition_abstention
            and float(short_similarity) <= float(self.config.transition_incumbent_max_similarity)
            and drop >= float(self.config.transition_incumbent_drop)
        )
        if signal:
            self._incumbent_rejection_count += 1
        else:
            self._incumbent_rejection_count = 0
            self._incumbent_short_similarities.append(float(short_similarity))
        self._incumbent_last_media_time = self._last_media_time
        confirmed = signal and self._incumbent_rejection_count >= int(
            self.config.transition_incumbent_clear_required
        )
        return confirmed, reference, drop

    def _scale_consistent_for_new_profile(self, evidences: Sequence[MultiScaleEvidence]) -> bool:
        if len(evidences) < 2:
            return False
        short = evidences[0].embedding
        comparisons = [float(np.dot(short, item.embedding)) for item in evidences[1:]]
        return max(comparisons, default=-1.0) >= float(self.config.provisional_scale_consistency)

    def _history_statistic(self, values: Sequence[float]) -> float:
        if not values:
            return -1.0
        if self.config.history_statistic == "median":
            return float(np.median(np.asarray(values, dtype=np.float64)))
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def _create_or_stage_provisional(
        self,
        evidences: Sequence[MultiScaleEvidence],
        best_existing_similarity: float,
    ) -> str | None:
        if not self.config.enable_online_profiles or not evidences:
            return None
        if len(self._profiles) >= int(self.config.max_profiles):
            return None
        if self._profiles and best_existing_similarity > float(
            self.config.provisional_max_existing_similarity
        ):
            self._reset_pending_profile()
            return None
        vector = evidences[0].embedding
        immediate = (
            (not self._profiles and self.config.provisional_first_immediate)
            or self._scale_consistent_for_new_profile(evidences)
        )
        if immediate:
            count = int(self.config.provisional_confirm_count)
            centroid = vector
        elif (
            self._pending_vector is None
            or self._pending_last_media_time is None
            or self._last_media_time - self._pending_last_media_time
            > float(self.config.history_max_gap_seconds) + 1e-9
        ):
            self._pending_vector = vector.copy()
            self._pending_count = 1
            self._pending_last_media_time = self._last_media_time
            return None
        else:
            similarity = float(np.dot(self._pending_vector, vector))
            if similarity < float(self.config.provisional_confirm_similarity):
                self._pending_vector = vector.copy()
                self._pending_count = 1
                self._pending_last_media_time = self._last_media_time
                return None
            self._pending_vector = self._merge_vectors(self._pending_vector, vector, 0.5)
            self._pending_count += 1
            self._pending_last_media_time = self._last_media_time
            count = self._pending_count
            centroid = self._pending_vector
        if count < int(self.config.provisional_confirm_count):
            return None
        label = self._new_provisional_label()
        self._profiles[label] = _Profile(label, normalize_vector(centroid), True)
        self._reset_pending_profile()
        return label

    def _update_provisional(self, label: str, evidence: MultiScaleEvidence) -> None:
        profile = self._profiles.get(label)
        if profile is None or not profile.provisional:
            return
        profile.centroid = self._merge_vectors(
            profile.centroid,
            evidence.embedding,
            self.config.provisional_update_alpha,
        )

    def _known_step(
        self,
        evidences: Sequence[MultiScaleEvidence],
    ) -> tuple[str | None, str, dict[str, float], dict[str, Any]]:
        evidences = tuple(sorted(evidences, key=lambda item: float(item.window_seconds)))
        windows = [round(float(item.window_seconds), 6) for item in evidences]
        if len(windows) != len(set(windows)):
            raise ValueError("multi-scale evidence window lengths must be unique")
        scale_similarities = [self._similarities(item) for item in evidences]
        weights = self._weights(evidences)
        labels = sorted(self._profiles)
        fused = {
            label: float(sum(weights[index] * values.get(label, -1.0) for index, values in enumerate(scale_similarities)))
            for label in labels
        }
        scale_tops = [self._top(values) for values in scale_similarities]
        fused_label, fused_similarity, fused_margin = self._top(fused)
        top_labels = [label for label, similarity, _margin in scale_tops if label is not None and similarity >= self.config.min_similarity]
        agreement = top_labels.count(fused_label) if fused_label is not None else 0
        fused_valid = (
            fused_label is not None
            and fused_similarity >= float(self.config.min_similarity)
            and fused_margin >= float(self.config.min_margin)
        )
        diagnostics: dict[str, Any] = {
            "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
            "scale_windows": [float(item.window_seconds) for item in evidences],
            "scale_weights": [float(value) for value in weights],
            "scale_top_labels": [item[0] for item in scale_tops],
            "scale_top_similarities": [float(item[1]) for item in scale_tops],
            "fused_similarity": fused_similarity,
            "fused_margin": fused_margin,
            "scale_agreement": agreement,
        }

        if (
            self.config.enable_transition_abstention
            and self._tracking_state == TRACKING_TRANSITION
        ):
            return self._advance_transition(
                evidences[0].embedding,
                scale_similarities,
                scale_tops,
                weights,
                fused,
                fused_label,
                fused_similarity,
                fused_margin,
                diagnostics,
            )

        if self._visible is None:
            if fused_valid and agreement >= int(self.config.acquire_scale_agreement):
                self._reset_challenger()
                self._seed_stable_fast_history(fused_label, evidences[0].embedding)
                return fused_label, "multiscale_acquire", fused, diagnostics
            created = self._create_or_stage_provisional(evidences, fused_similarity)
            return created, "provisional_acquire" if created else "multiscale_unknown", fused, diagnostics

        current = self._visible
        short_values = scale_similarities[0] if scale_similarities else {}
        long_values = scale_similarities[-1] if scale_similarities else {}
        short_label = scale_tops[0][0] if scale_tops else None
        fast_values, slow_values, fast_indices, slow_indices = self._transition_banks(
            scale_similarities, weights
        )
        fast_label, fast_similarity, fast_margin = self._top(fast_values)
        slow_label, _slow_similarity, _slow_margin = self._top(slow_values)
        transition_candidate = (
            fast_label if fast_label is not None and fast_label != current else None
        )
        transition_fast_advantage = (
            float(fast_values.get(transition_candidate, -1.0) - fast_values.get(current, -1.0))
            if transition_candidate is not None else -1.0
        )
        transition_slow_advantage = (
            float(slow_values.get(transition_candidate, -1.0) - slow_values.get(current, -1.0))
            if transition_candidate is not None else -1.0
        )
        transition_valid_scales = sum(
            1
            for _label, similarity, _margin in scale_tops
            if similarity >= float(self.config.transition_min_similarity)
        )
        known_short_challenger = (
            transition_candidate is not None
            and fast_similarity >= float(self.config.transition_min_similarity)
            and fast_margin >= float(self.config.transition_min_margin)
            and transition_fast_advantage >= float(self.config.transition_short_advantage)
        )
        transition_known_signal = (
            self.config.enable_transition_abstention
            and transition_valid_scales >= int(self.config.transition_min_valid_scales)
            and known_short_challenger
            and (
                slow_label == current
                or transition_fast_advantage - transition_slow_advantage
                >= float(self.config.transition_scale_gap)
            )
        )
        (
            embedding_change,
            embedding_change_similarity,
            embedding_history_similarity,
            embedding_similarity_drop,
        ) = self._embedding_change_signal(current, evidences[0].embedding)
        incumbent_rejection, incumbent_reference, incumbent_drop = self._incumbent_rejection(
            current,
            fast_values.get(current, -1.0),
            slow_values.get(current, -1.0),
        )
        if known_short_challenger:
            incumbent_rejection = False
            self._incumbent_rejection_count = 0
        diagnostics.update({
            "incumbent_short_similarity": float(fast_values.get(current, -1.0)),
            "incumbent_long_similarity": float(slow_values.get(current, -1.0)),
            "incumbent_reference_similarity": incumbent_reference,
            "incumbent_similarity_drop": incumbent_drop,
            "incumbent_rejection_count": self._incumbent_rejection_count,
            "transition_incumbent_rejection": incumbent_rejection,
            "transition_fast_indices": fast_indices,
            "transition_slow_indices": slow_indices,
            "transition_fast_label": fast_label,
            "transition_slow_label": slow_label,
            "transition_fast_similarity": fast_similarity,
            "transition_fast_margin": fast_margin,
            "transition_fast_advantage": transition_fast_advantage,
            "transition_slow_advantage": transition_slow_advantage,
            "transition_known_signal": transition_known_signal,
            "transition_embedding_change": embedding_change,
            "transition_embedding_similarity": embedding_change_similarity,
            "transition_embedding_history_similarity": embedding_history_similarity,
            "transition_embedding_similarity_drop": embedding_similarity_drop,
            "transition_embedding_change_count": self._embedding_change_count,
        })
        if embedding_change:
            self._enter_transition(
                current,
                candidate=transition_candidate if known_short_challenger else None,
                kind="embedding_change",
            )
            diagnostics.update({
                "tracking_state": TRACKING_TRANSITION,
                "transition_incumbent": current,
                "transition_candidate": (
                    transition_candidate if known_short_challenger else None
                ),
                "transition_entry_kind": "embedding_change",
                "transition_abstention": True,
            })
            return None, "transition_embedding_change", fused, diagnostics
        if incumbent_rejection:
            self._reset_incumbent_history()
            self._enter_transition(
                current,
                candidate=None,
                kind="incumbent_rejection",
            )
            diagnostics.update({
                "tracking_state": TRACKING_TRANSITION,
                "transition_incumbent": current,
                "transition_candidate": None,
                "transition_entry_kind": "incumbent_rejection",
                "transition_abstention": True,
            })
            return None, "transition_incumbent_rejection", fused, diagnostics
        if self._record_transition_entry(transition_candidate, transition_known_signal):
            self._enter_transition(
                current,
                candidate=transition_candidate,
                kind="known_crossover",
            )
            diagnostics.update({
                "tracking_state": TRACKING_TRANSITION,
                "transition_incumbent": current,
                "transition_candidate": transition_candidate,
                "transition_entry_kind": "known_crossover",
                "transition_candidate_votes": 1,
                "transition_abstention": True,
            })
            return None, "transition_abstain", fused, diagnostics
        if not fused_valid and not top_labels:
            self._reset_challenger()
            return None, "multiscale_unknown", fused, diagnostics
        candidate = short_label if short_label and short_label != current else fused_label
        if candidate is None or candidate == current:
            if not fused_valid:
                self._reset_challenger()
                return None, "multiscale_unknown", fused, diagnostics
            self._reset_challenger()
            if current in short_values and short_values[current] >= self.config.min_similarity:
                self._update_provisional(current, evidences[0])
                self._seed_stable_fast_history(current, evidences[0].embedding)
            return current, "multiscale_confirmed", fused, diagnostics

        short_advantage = float(short_values.get(candidate, -1.0) - short_values.get(current, -1.0))
        long_advantage = float(long_values.get(candidate, -1.0) - long_values.get(current, -1.0))
        fused_advantage = float(fused.get(candidate, -1.0) - fused.get(current, -1.0))
        history_gap = (
            self._challenger_last_media_time is not None
            and self._last_media_time - self._challenger_last_media_time
            > float(self.config.history_max_gap_seconds) + 1e-9
        )
        if self._challenger != candidate or history_gap:
            self._challenger = candidate
            self._short_advantages.clear()
            self._long_advantages.clear()
            self._fused_advantages.clear()
        self._short_advantages.append(short_advantage)
        self._long_advantages.append(long_advantage)
        self._fused_advantages.append(fused_advantage)
        self._challenger_last_media_time = self._last_media_time
        recent_count = min(int(self.config.history_required), len(self._short_advantages))
        recent_short = list(self._short_advantages)[-recent_count:]
        recent_fused = list(self._fused_advantages)[-recent_count:]
        short_history_advantage = self._history_statistic(recent_short)
        fused_history_advantage = self._history_statistic(recent_fused)
        history_short_weight = float(self.config.history_short_weight)
        history_advantage = (
            history_short_weight * short_history_advantage
            + (1.0 - history_short_weight) * fused_history_advantage
        )
        consensus_count = top_labels.count(candidate)
        consensus = (
            self.config.enable_consensus
            and consensus_count >= int(self.config.min_scale_agreement)
            and fused_advantage >= float(self.config.consensus_advantage)
        )
        crossover_count = min(int(self.config.crossover_required), len(self._short_advantages))
        short_moving_advantage = (
            float(np.mean(list(self._short_advantages)[-crossover_count:]))
            if crossover_count else -1.0
        )
        long_moving_advantage = (
            float(np.mean(list(self._long_advantages)[-crossover_count:]))
            if crossover_count else -1.0
        )
        crossover = (
            self.config.enable_crossover
            and short_values.get(candidate, -1.0) >= float(self.config.min_similarity)
            and len(self._short_advantages) >= int(self.config.crossover_required)
            and short_moving_advantage >= float(self.config.crossover_short_advantage)
            and (short_moving_advantage - long_moving_advantage)
            >= float(self.config.crossover_scale_gap)
        )
        history = (
            self.config.enable_history
            and short_values.get(candidate, -1.0) >= float(self.config.min_similarity)
            and len(self._short_advantages) >= int(self.config.history_required)
            and history_advantage >= float(self.config.history_advantage)
        )
        direct = not (
            self.config.enable_consensus or self.config.enable_crossover or self.config.enable_history
        ) and fused_valid and fused_label == candidate
        diagnostics.update({
            "challenger": candidate,
            "short_advantage": short_advantage,
            "long_advantage": long_advantage,
            "fused_advantage": fused_advantage,
            "short_moving_advantage": short_moving_advantage,
            "long_moving_advantage": long_moving_advantage,
            "crossover_gap": short_moving_advantage - long_moving_advantage,
            "crossover_count": crossover_count,
            "history_gap_reset": history_gap,
            "history_advantage": history_advantage,
            "short_history_advantage": short_history_advantage,
            "fused_history_advantage": fused_history_advantage,
            "history_short_weight": history_short_weight,
            "history_statistic": self.config.history_statistic,
            "history_count": len(self._short_advantages),
            "challenger_scale_agreement": consensus_count,
            "switch_consensus": consensus,
            "switch_crossover": crossover,
            "switch_history": history,
            "transition_scale_disagreement": len(set(top_labels)) > 1,
            "transition_abstention": False,
        })
        if consensus or crossover or history or direct:
            reason = (
                "multiscale_consensus" if consensus else
                "scale_crossover" if crossover else
                "similarity_history" if history else
                "fused_direct"
            )
            self._reset_challenger()
            self._update_provisional(candidate, evidences[0])
            self._seed_stable_fast_history(candidate, evidences[0].embedding)
            return candidate, reason, fused, diagnostics

        if not (
            self.config.enable_consensus
            or self.config.enable_crossover
            or self.config.enable_history
        ) and not fused_valid:
            self._reset_challenger()
            return None, "multiscale_unknown", fused, diagnostics

        best_existing = max(fused.values(), default=-1.0)
        created = self._create_or_stage_provisional(evidences, best_existing)
        if created is not None:
            self._reset_challenger()
            self._seed_stable_fast_history(created, evidences[0].embedding)
            return created, "provisional_switch", fused, diagnostics
        if fast_label == current and fast_similarity >= float(self.config.min_similarity):
            self._seed_stable_fast_history(current, evidences[0].embedding)
        return current, "multiscale_hold", fused, diagnostics

    def step(self, item: MultiScaleStep) -> LiveSpeakerDecision:
        media_time = float(item.media_time)
        if media_time + 1e-9 < self._last_media_time:
            raise ValueError("multi-scale steps must be chronological")
        if not item.probe_scheduled and item.evidences:
            raise ValueError("a non-probe tick may not carry multi-scale evidence")
        self._last_media_time = media_time
        tracking_state_before = self._tracking_state
        applied = self._apply_profile_events(media_time)
        if applied:
            self._reset_challenger()
            self._reset_pending_profile()
            self._reset_incumbent_history()
            if self._tracking_state == TRACKING_TRANSITION:
                self._reset_transition_evidence()
        self._buffer_evidences(media_time, item)
        action = "none"
        reason = "multiscale_unknown"
        candidate: str | None = None
        fused: dict[str, float] = {}
        extra: dict[str, Any] = {}

        if not item.probe_scheduled:
            if item.release_signal:
                self._reset_challenger()
                self._reset_pending_profile()
                self._reset_incumbent_history()
                self._reset_speech_gate_history()
                self._silence_count += 1
                self._unknown_count = 0
                was_transition = self._tracking_state == TRACKING_TRANSITION
                if was_transition:
                    self._set_tracking_off()
                    action, reason = "none", "release_gate"
                elif self._visible and self._silence_count >= self.config.silence_release_count:
                    self._set_tracking_off()
                    self._reset_challenger()
                    action, reason = "clear", "release_gate"
                else:
                    action, reason = ("hold", "release_debounce") if self._visible else ("none", "release_gate")
            elif (
                not item.speech
                and self.config.enable_transition_abstention
                and self.config.enable_transition_speech_gate
                and self._tracking_state == TRACKING_STABLE
                and self._visible is not None
            ):
                self._reset_challenger()
                self._reset_pending_profile()
                self._reset_incumbent_history()
                self._speech_gate_false_probe_count += 1
                if self._speech_gate_false_probe_count >= int(
                    self.config.transition_speech_gate_clear_required
                ):
                    incumbent = self._visible
                    self._enter_transition(
                        incumbent,
                        candidate=None,
                        kind="speech_gate",
                    )
                    action, reason = "clear", "transition_speech_gate"
                else:
                    action, reason = "hold", "transition_speech_gate_debounce"
            elif self._tracking_state == TRACKING_TRANSITION:
                if self._transition_timed_out():
                    self._set_tracking_off()
                    action, reason = "none", "transition_timeout"
                else:
                    action, reason = "none", "transition_wait_non_probe"
            else:
                self._reset_speech_gate_history()
                action, reason = ("hold", "non_probe_tick") if self._visible else ("none", "non_probe_tick")
        elif item.release_signal:
            self._silence_count += 1
            self._unknown_count = 0
            self._reset_challenger()
            self._reset_pending_profile()
            self._reset_incumbent_history()
            self._reset_speech_gate_history()
            was_transition = self._tracking_state == TRACKING_TRANSITION
            if was_transition:
                self._set_tracking_off()
                action, reason = "none", "release_gate"
            elif self._visible and self._silence_count >= self.config.silence_release_count:
                self._set_tracking_off()
                action, reason = "clear", "release_gate"
            else:
                action, reason = ("hold", "release_debounce") if self._visible else ("none", "release_gate")
        elif not item.speech:
            self._reset_challenger()
            self._reset_pending_profile()
            self._reset_incumbent_history()
            self._silence_count = 0
            self._unknown_count = 0
            if self._tracking_state == TRACKING_TRANSITION:
                if self._transition_timed_out():
                    self._set_tracking_off()
                    action, reason = "none", "transition_timeout"
                else:
                    action, reason = "none", "transition_wait_silence"
            elif (
                self.config.enable_transition_abstention
                and self.config.enable_transition_speech_gate
                and self._tracking_state == TRACKING_STABLE
                and self._visible is not None
            ):
                self._speech_gate_false_probe_count += 1
                if self._speech_gate_false_probe_count >= int(
                    self.config.transition_speech_gate_clear_required
                ):
                    incumbent = self._visible
                    self._enter_transition(
                        incumbent,
                        candidate=None,
                        kind="speech_gate",
                    )
                    action, reason = "clear", "transition_speech_gate"
                else:
                    action, reason = "hold", "transition_speech_gate_debounce"
            else:
                self._reset_speech_gate_history()
                action, reason = ("hold", "probe_gate_silence") if self._visible else ("none", "probe_gate_silence")
        elif not item.evidences:
            self._reset_speech_gate_history()
            self._reset_challenger()
            self._reset_pending_profile()
            self._reset_incumbent_history()
            self._silence_count = 0
            if self._tracking_state == TRACKING_TRANSITION:
                if self._transition_timed_out():
                    self._set_tracking_off()
                    action, reason = "none", "transition_timeout"
                else:
                    action, reason = "none", "transition_wait_no_evidence"
            else:
                self._unknown_count += 1
            if self._visible and self._unknown_count >= self.config.unknown_release_count:
                self._set_tracking_off()
                self._reset_challenger()
                action, reason = "clear", "unknown"
            elif self._tracking_state != TRACKING_TRANSITION:
                action, reason = ("hold", "unknown_debounce") if self._visible else ("none", "unknown")
        else:
            self._reset_speech_gate_history()
            self._silence_count = 0
            previous = self._visible
            candidate, reason, fused, extra = self._known_step(item.evidences)
            if candidate is None:
                if reason.startswith("transition_"):
                    self._unknown_count = 0
                    action = (
                        "clear"
                        if previous is not None and reason in {
                            "transition_abstain",
                            "transition_incumbent_rejection",
                            "transition_embedding_change",
                        }
                        else "none"
                    )
                else:
                    self._unknown_count += 1
                if (
                    not reason.startswith("transition_")
                    and previous
                    and self._unknown_count >= self.config.unknown_release_count
                ):
                    self._set_tracking_off()
                    action, reason = "clear", "unknown"
                elif not reason.startswith("transition_"):
                    action = "hold" if previous else "none"
                    reason = "unknown_debounce" if previous else "unknown"
            else:
                self._unknown_count = 0
                self._set_tracking_stable(candidate, reset_entry=previous != candidate)
                action = "acquire" if previous is None else "switch" if previous != candidate else "hold"
                if previous != candidate:
                    self._reset_incumbent_history()

        probabilities = self._probabilities(fused)
        return LiveSpeakerDecision(
            media_time=media_time,
            visible_speaker=self._visible,
            action=action,
            reason=reason,
            candidate_speaker=candidate,
            probabilities=probabilities,
            raw_probabilities=probabilities,
            similarities=fused,
            profile_count=len(self._profiles),
            profile_generations=dict(self._generations),
            diagnostics={
                "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
                "profile_events_applied": applied,
                "profile_aliases": dict(self._aliases),
                "provisional_profiles": sorted(
                    label for label, profile in self._profiles.items() if profile.provisional
                ),
                "untrusted_profiles": sorted(
                    label for label, profile in self._profiles.items() if not profile.trusted
                ),
                "duration_profile_count": sum(
                    len(profile.scale_centroids) for profile in self._profiles.values()
                ),
                **extra,
                "unknown_count": self._unknown_count,
                "silence_count": self._silence_count,
                "tracking_state_before": tracking_state_before,
                "tracking_state": self._tracking_state,
                "transition_incumbent": self._transition_incumbent,
                "transition_candidate": self._transition_candidate,
                "transition_entry_kind": self._transition_entry_kind,
                "transition_off_probes": self._transition_off_probes,
                "transition_speech_gate_false_probe_count": self._speech_gate_false_probe_count,
                "probe_scheduled": bool(item.probe_scheduled),
                "release_signal": bool(item.release_signal),
                "skipped_reason": item.skipped_reason,
            },
        )


def replay_cached_multiscale_windows(
    blocks: Sequence[CachedLiveWindowBlock],
    profile_events: Iterable[SpeakerProfileEvent],
    speech_mask: Sequence[bool] | np.ndarray,
    probe_mask: Sequence[bool] | np.ndarray,
    release_mask: Sequence[bool] | np.ndarray,
    *,
    config: MultiScaleTrackerConfig | None = None,
) -> list[LiveSpeakerDecision]:
    if not blocks:
        raise ValueError("at least one cached window block is required")
    ordered = sorted(blocks, key=lambda item: float(item.window_seconds))
    reference = ordered[0]
    for block in ordered[1:]:
        if block.video_id != reference.video_id:
            raise ValueError("multi-scale blocks must describe the same video")
        if block.sample_rate != reference.sample_rate or not np.array_equal(
            block.media_times, reference.media_times
        ):
            raise ValueError("multi-scale blocks must share the exact timeline")
        if block.embeddings.shape[1] != reference.embeddings.shape[1]:
            raise ValueError("multi-scale blocks must have the same embedding dimension")
    rows = int(reference.media_times.shape[0])
    speech = np.asarray(speech_mask, dtype=bool).reshape(-1)
    probes = np.asarray(probe_mask, dtype=bool).reshape(-1)
    releases = np.asarray(release_mask, dtype=bool).reshape(-1)
    if any(values.shape[0] != rows for values in (speech, probes, releases)):
        raise ValueError("all multi-scale masks must have one value per timeline tick")
    effective_config = config
    if effective_config is not None and effective_config.scale_weights and not effective_config.scale_windows:
        effective_config = replace(
            effective_config,
            scale_windows=tuple(float(block.window_seconds) for block in ordered),
        )
    tracker = CausalMultiScaleSpeakerTracker(
        config=effective_config,
        profile_events=profile_events,
    )
    results: list[LiveSpeakerDecision] = []
    for index, media_time in enumerate(reference.media_times):
        scheduled = bool(probes[index])
        evidences = tuple(
            MultiScaleEvidence(float(block.window_seconds), block.embeddings[index])
            for block in ordered
            if scheduled and bool(block.valid[index])
        )
        results.append(tracker.step(MultiScaleStep(
            media_time=float(media_time),
            speech=bool(speech[index]),
            evidences=evidences,
            probe_scheduled=scheduled,
            release_signal=bool(releases[index]),
            skipped_reason=(
                "" if evidences else
                "not_a_scheduled_probe" if not scheduled else
                "cached_embeddings_invalid"
            ),
        )))
    return results
