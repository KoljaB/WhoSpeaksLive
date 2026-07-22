"""Replay-only causal segmental/open-set decoder experiments.

This module intentionally stays outside ``src``.  It consumes the authentic
World Tapes and their already-recorded 0.7/1.5 s SpeechBrain vectors, performs
no model inference, and cannot emit promotion evidence.

The practical decoder is deliberately different from the existing Bayes
tracker and the novelty-only tracklet wrapper: every speech probe participates
in an online, non-parametric speaker state.  A bounded exemplar bank models
each state, a causal segment-change test controls transitions, and final
profiles are attached to an already-visible state only when they become
available.  Temporary states never enter final diarization memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from analyze_live_speaker_open_set_tracklets import (
    _PreparedTape,
    _cosine,
    _evaluate_baseline,
    _evaluate_variant as _evaluate_tracklet_variant,
    _identity_error_diagnostics,
    _load_base_config,
    _mapped_values,
    _prepare_tape,
    _profile_vectors,
    _public_probability_key,
    _replay_with_tracklet_actions,
    _unit,
    TrackletConfig,
)
from window.live_speaker_probe_scoring import read_canonical_segments


CONTRACT_ID = "whospeaks.live_world_tape.segmental_dp_diagnostic.v1"


@dataclass(frozen=True)
class SegmentalConfig:
    name: str = "segmental_exemplar_v1"
    max_identities: int = 12
    max_short_exemplars: int = 10
    max_long_exemplars: int = 6
    prototype_min_gap_seconds: float = 0.75
    centroid_alpha: float = 0.12
    top_k: int = 3
    long_weight: float = 0.15
    active_long_weight: float = 0.25
    stay_min: float = 0.28
    acquire_min: float = 0.34
    switch_min: float = 0.34
    switch_margin: float = 0.04
    instant_switch_min: float = 0.48
    instant_switch_margin: float = 0.12
    new_ceiling: float = 0.28
    pending_similarity: float = 0.30
    pending_max_gap_seconds: float = 1.20
    confirm_new_count: int = 2
    confirm_switch_count: int = 2
    update_min: float = 0.42
    stable_update_count: int = 2
    profile_merge_min: float = 0.35
    profile_merge_mode: str = "sentence_overlap"
    profile_merge_overlap_min: float = 0.20
    profile_merge_interval_padding_seconds: float = 0.35
    profile_merge_min_activity_points: int = 1
    exclusive_profile_merge: bool = True
    clear_on_pending_new: bool = True
    clear_on_pending_switch: bool = False
    create_first_immediately: bool = True
    base_vote_bonus: float = 0.0


@dataclass
class _Identity:
    label: str
    short_centroid: np.ndarray
    long_centroid: np.ndarray | None
    created_media_time: float
    last_media_time: float
    short_exemplars: list[np.ndarray] = field(default_factory=list)
    long_exemplars: list[np.ndarray] = field(default_factory=list)
    profile_exemplars: list[np.ndarray] = field(default_factory=list)
    final_labels: set[str] = field(default_factory=set)
    activity_times: list[float] = field(default_factory=list)
    stable_count: int = 1

    @classmethod
    def from_probe(
        cls,
        label: str,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
    ) -> "_Identity":
        return cls(
            label=label,
            short_centroid=short.copy(),
            long_centroid=None if long is None else long.copy(),
            created_media_time=media_time,
            last_media_time=media_time,
            short_exemplars=[short.copy()],
            long_exemplars=[] if long is None else [long.copy()],
            activity_times=[media_time],
        )

    @classmethod
    def from_profile(
        cls, label: str, vector: np.ndarray, media_time: float, final_label: str
    ) -> "_Identity":
        item = cls.from_probe(label, vector, None, media_time)
        item.final_labels.add(final_label)
        return item

    @staticmethod
    def _bank_score(value: np.ndarray | None, bank: list[np.ndarray], centroid: np.ndarray | None, top_k: int) -> float:
        if value is None:
            return -1.0
        scores = [_cosine(value, item) for item in bank]
        if centroid is not None:
            scores.append(_cosine(value, centroid))
        if not scores:
            return -1.0
        scores.sort(reverse=True)
        count = min(max(1, int(top_k)), len(scores))
        # A robust best-exemplar score: the best match carries most of the
        # attack signal, while the top-k mean suppresses accidental spikes.
        return 0.65 * scores[0] + 0.35 * mean(scores[:count])

    def short_score(self, value: np.ndarray | None, top_k: int) -> float:
        bank = self.short_exemplars + self.profile_exemplars
        return self._bank_score(value, bank, self.short_centroid, top_k)

    def long_score(self, value: np.ndarray | None, top_k: int) -> float:
        return self._bank_score(value, self.long_exemplars, self.long_centroid, top_k)

    def profile_score(self, value: np.ndarray | None, top_k: int) -> float:
        return max(self.short_score(value, top_k), self.long_score(value, top_k))

    def attach_profile(self, final_label: str, value: np.ndarray) -> None:
        self.final_labels.add(final_label)
        self.profile_exemplars.append(value.copy())
        del self.profile_exemplars[: max(0, len(self.profile_exemplars) - 8)]

    def update(
        self,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
        config: SegmentalConfig,
    ) -> None:
        alpha = max(0.0, min(1.0, float(config.centroid_alpha)))
        short_centroid = _unit((1.0 - alpha) * self.short_centroid + alpha * short)
        if short_centroid is not None:
            self.short_centroid = short_centroid
        if long is not None:
            if self.long_centroid is None:
                self.long_centroid = long.copy()
            else:
                long_centroid = _unit((1.0 - alpha) * self.long_centroid + alpha * long)
                if long_centroid is not None:
                    self.long_centroid = long_centroid
        if media_time - self.last_media_time >= config.prototype_min_gap_seconds:
            self.short_exemplars.append(short.copy())
            del self.short_exemplars[: max(0, len(self.short_exemplars) - config.max_short_exemplars)]
            if long is not None and self.stable_count >= config.stable_update_count:
                self.long_exemplars.append(long.copy())
                del self.long_exemplars[: max(0, len(self.long_exemplars) - config.max_long_exemplars)]
        self.last_media_time = media_time
        self.stable_count += 1

    def mark_active(self, media_time: float) -> None:
        if not self.activity_times or media_time > self.activity_times[-1] + 1e-6:
            self.activity_times.append(media_time)
            # Enough for every sentence interval while keeping state bounded.
            del self.activity_times[: max(0, len(self.activity_times) - 256)]


@lru_cache(maxsize=32)
def _profile_sentence_intervals(tape_dir: str) -> dict[str, dict[str, float]]:
    """Read the causal sentence interval exposed by profile publication."""

    result: dict[str, dict[str, float]] = {}
    path = Path(tape_dir) / "events.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("event") or "") != "live_speaker_profile_snapshot":
                continue
            payload = dict(record.get("payload") or {})
            if str(payload.get("profile_embedding_provider") or "") != "speechbrain_resnet":
                continue
            speaker_id = str(payload.get("speaker_id") or "")
            if not speaker_id or speaker_id in result:
                continue
            result[speaker_id] = {
                "available_at": float(payload.get("available_at") or 0.0),
                "sentence_start": float(payload.get("sentence_start") or 0.0),
                "sentence_end": float(payload.get("sentence_end") or 0.0),
            }
    return result


@dataclass
class _Pending:
    target: str
    short: np.ndarray
    long: np.ndarray | None
    count: int
    last_media_time: float


def _canonical_at(segments: list[dict[str, Any]], media_time: float) -> str:
    for row in segments:
        if float(row["start"]) <= media_time < float(row["end"]):
            return str(row["speaker"])
    return ""


def _emit_action(
    prepared: _PreparedTape,
    step: Any,
    chosen: str,
    reason: str,
    active_public: str,
) -> tuple[tuple[float, int, str, dict[str, Any]] | None, str]:
    payload = step.payload
    dedicated = bool(str(payload.get("probe_id") or ""))
    release = bool(payload.get("release_signal"))
    if not dedicated:
        return None, active_public
    media_time = float(payload.get("media_time") or 0.0)
    duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
    base = step.recorded_public_payload
    if chosen and not release:
        trace = step.base_trace
        probabilities = _mapped_values(
            dict(trace.get("probabilities") or {}), step.public_aliases, probability_keys=True
        )
        raw_probabilities = _mapped_values(
            dict(trace.get("raw_probabilities") or {}), step.public_aliases, probability_keys=True
        )
        similarities = _mapped_values(
            dict(trace.get("similarities") or {}), step.public_aliases
        )
        live_payload = {
            **base,
            "step_id": int(payload.get("step_id") or 0),
            "assigned_speaker": chosen,
            "speaker_id": chosen,
            "probabilities": probabilities,
            "raw_probabilities": raw_probabilities,
            "similarities": similarities,
            "unknown_probability": float(probabilities.get("unknown", 1.0)),
            "live_speaker_core_action": "show",
            "live_speaker_core_reason": reason,
            "live": True,
            "fallback": True,
            "start": base.get("start", round(max(0.0, media_time - duration), 4)),
            "end": base.get("end", round(media_time, 4)),
            "audio_length_seconds": base.get("audio_length_seconds", round(duration, 4)),
            "hold_seconds": round(prepared.hold_seconds, 4),
            "assignment_source": "offline_segmental_dp_diagnostic",
            "segmental_probability_key": _public_probability_key(chosen),
        }
        return (step.wall_seconds, step.sequence, "live_speaker", live_payload), chosen
    if active_public:
        clear_payload = {
            **base,
            "step_id": int(payload.get("step_id") or 0),
            "speaker_id": active_public,
            "assigned_speaker": None,
            "live": False,
            "fallback": True,
            "start": base.get("start", round(max(0.0, media_time - duration), 4)),
            "end": base.get("end", round(media_time, 4)),
            "reason": "silence" if release else reason,
            "assignment_source": "offline_segmental_dp_diagnostic",
        }
        return (step.wall_seconds, step.sequence, "live_speaker_clear", clear_payload), ""
    return None, ""


def _identity_scores(
    identities: list[_Identity],
    short: np.ndarray,
    long: np.ndarray | None,
    config: SegmentalConfig,
    active_label: str,
    base_public: str,
) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for item in identities:
        short_score = item.short_score(short, config.top_k)
        long_score = item.long_score(long, config.top_k)
        weight = config.active_long_weight if item.label == active_label else config.long_weight
        combined = (1.0 - weight) * short_score + weight * (
            long_score if long is not None and long_score > -0.99 else short_score
        )
        if base_public and base_public in item.final_labels:
            combined += config.base_vote_bonus
        result[item.label] = (combined, short_score, long_score)
    return result


def _segmental_projection(
    prepared: _PreparedTape, config: SegmentalConfig
) -> tuple[list[tuple[float, int, str, dict[str, Any]]], dict[str, Any]]:
    identities: list[_Identity] = []
    final_to_identity: dict[str, str] = {}
    known_final_labels: set[str] = set()
    next_identity = 1
    active_label = ""
    active_public = ""
    pending: _Pending | None = None
    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    stats: dict[str, Any] = {
        "identities_created": 0,
        "profile_only_identities": 0,
        "profile_merges": 0,
        "pending_new": 0,
        "pending_switch": 0,
        "confirmed_new": 0,
        "confirmed_switch": 0,
        "instant_switches": 0,
        "updates": 0,
        "final_aliases": final_to_identity,
        "overlap_guided_profile_merges": 0,
        "cosine_only_profile_merges": 0,
    }

    def by_label(label: str) -> _Identity | None:
        return next((item for item in identities if item.label == label), None)

    publication_intervals = _profile_sentence_intervals(str(prepared.tape_dir))

    for step in prepared.steps:
        payload = step.payload
        profiles = _profile_vectors(payload)
        added = sorted(set(profiles) - known_final_labels)
        known_final_labels = set(profiles)

        # A new profile is attached only at the first step where the authentic
        # World Tape exposes it.  Matching is one-to-one by default.
        for final_label in added:
            profile = profiles[final_label]
            candidates = [
                item
                for item in identities
                if not config.exclusive_profile_merge or not item.final_labels
            ]
            best = None
            similarity = -1.0
            merge_floor = config.profile_merge_min
            overlap_guided = False
            interval = publication_intervals.get(final_label)
            if config.profile_merge_mode == "sentence_overlap" and interval is not None:
                left = interval["sentence_start"] - config.profile_merge_interval_padding_seconds
                right = interval["sentence_end"] + config.profile_merge_interval_padding_seconds
                activity_ranked = []
                for item in candidates:
                    activity = sum(left <= value <= right for value in item.activity_times)
                    activity_ranked.append(
                        (activity, item.profile_score(profile, config.top_k), item)
                    )
                activity_ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
                if (
                    activity_ranked
                    and activity_ranked[0][0] >= config.profile_merge_min_activity_points
                ):
                    _count, similarity, best = activity_ranked[0]
                    merge_floor = config.profile_merge_overlap_min
                    overlap_guided = True
            if best is None:
                best = max(
                    candidates,
                    key=lambda item: item.profile_score(profile, config.top_k),
                    default=None,
                )
                similarity = best.profile_score(profile, config.top_k) if best is not None else -1.0
            final_public = step.public_aliases.get(final_label, final_label)
            if best is not None and similarity >= merge_floor:
                best.attach_profile(final_label, profile)
                final_to_identity[final_label] = best.label
                actions.append(
                    (
                        step.wall_seconds,
                        max(0, step.sequence - 1),
                        "tracklet_alias",
                        {
                            "old_label": final_public,
                            "new_label": best.label,
                            "similarity": similarity,
                            "reason": "segmental_profile_attach",
                        },
                    )
                )
                stats["profile_merges"] += 1
                stats[
                    "overlap_guided_profile_merges"
                    if overlap_guided
                    else "cosine_only_profile_merges"
                ] += 1
            elif len(identities) < config.max_identities:
                item = _Identity.from_profile(final_public, profile, float(payload.get("media_time") or 0.0), final_label)
                identities.append(item)
                final_to_identity[final_label] = item.label
                stats["profile_only_identities"] += 1

        short = _unit(payload.get("embedding"))
        long = _unit(payload.get("context_embedding"))
        media_time = float(payload.get("media_time") or 0.0)
        dedicated = bool(str(payload.get("probe_id") or ""))
        release = bool(payload.get("release_signal"))
        speech = bool(payload.get("speech")) and short is not None
        base_internal = str(step.base_trace.get("visible_speaker") or "")
        base_public = final_to_identity.get(
            base_internal, step.public_aliases.get(base_internal, base_internal)
        )

        chosen = active_label
        reason = "segmental_hold"
        if not dedicated or release or not speech:
            chosen = ""
            active_label = ""
            pending = None
            reason = "segmental_release"
        else:
            scores = _identity_scores(
                identities, short, long, config, active_label, base_public
            )
            ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
            top_label = ranked[0][0] if ranked else ""
            top_score = ranked[0][1][0] if ranked else -1.0
            current_score = scores.get(active_label, (-1.0, -1.0, -1.0))[0]
            alternative = next((item for item in ranked if item[0] != active_label), None)
            alternative_label = alternative[0] if alternative else ""
            alternative_score = alternative[1][0] if alternative else -1.0

            if not identities and config.create_first_immediately:
                item = _Identity.from_probe(f"T{next_identity}", short, long, media_time)
                next_identity += 1
                identities.append(item)
                chosen = item.label
                active_label = item.label
                pending = None
                stats["identities_created"] += 1
                reason = "segmental_first_identity"
            else:
                target = ""
                target_count = 0
                clear_pending = False
                if active_label:
                    if (
                        alternative_label
                        and alternative_score >= config.instant_switch_min
                        and alternative_score - current_score >= config.instant_switch_margin
                    ):
                        chosen = alternative_label
                        active_label = alternative_label
                        pending = None
                        stats["instant_switches"] += 1
                        reason = "segmental_instant_switch"
                    elif (
                        alternative_label
                        and alternative_score >= config.switch_min
                        and alternative_score - current_score >= config.switch_margin
                    ):
                        target = alternative_label
                        target_count = config.confirm_switch_count
                        clear_pending = config.clear_on_pending_switch
                    elif top_score < config.new_ceiling and current_score < config.stay_min:
                        target = "__new__"
                        target_count = config.confirm_new_count
                        clear_pending = config.clear_on_pending_new
                    elif current_score >= config.stay_min:
                        chosen = active_label
                        pending = None
                        reason = "segmental_stay"
                    elif top_label and top_score >= config.acquire_min:
                        chosen = top_label
                        active_label = top_label
                        pending = None
                        reason = "segmental_reacquire"
                    else:
                        target = "__new__"
                        target_count = config.confirm_new_count
                        clear_pending = config.clear_on_pending_new
                elif top_label and top_score >= config.acquire_min:
                    chosen = top_label
                    active_label = top_label
                    pending = None
                    reason = "segmental_acquire"
                else:
                    target = "__new__"
                    target_count = config.confirm_new_count
                    clear_pending = config.clear_on_pending_new

                if target:
                    consistent = bool(
                        pending
                        and pending.target == target
                        and media_time - pending.last_media_time <= config.pending_max_gap_seconds
                        and _cosine(pending.short, short) >= config.pending_similarity
                    )
                    if consistent:
                        pending.count += 1
                        merged = _unit(0.5 * pending.short + 0.5 * short)
                        if merged is not None:
                            pending.short = merged
                        if long is not None:
                            pending.long = long.copy() if pending.long is None else _unit(0.5 * pending.long + 0.5 * long)
                        pending.last_media_time = media_time
                    else:
                        pending = _Pending(target, short.copy(), None if long is None else long.copy(), 1, media_time)
                        stats["pending_new" if target == "__new__" else "pending_switch"] += 1
                    if pending.count >= target_count:
                        if target == "__new__" and len(identities) < config.max_identities:
                            item = _Identity.from_probe(f"T{next_identity}", pending.short, pending.long, media_time)
                            next_identity += 1
                            identities.append(item)
                            chosen = item.label
                            active_label = item.label
                            stats["identities_created"] += 1
                            stats["confirmed_new"] += 1
                            reason = "segmental_new_confirmed"
                        elif target != "__new__":
                            chosen = target
                            active_label = target
                            stats["confirmed_switch"] += 1
                            reason = "segmental_switch_confirmed"
                        pending = None
                    elif clear_pending:
                        chosen = ""
                        reason = "segmental_pending_unknown"

            selected = by_label(chosen)
            if selected is not None:
                selected.mark_active(media_time)
                selected_score = selected.short_score(short, config.top_k)
                if selected.stable_count <= 1 or selected_score >= config.update_min:
                    selected.update(short, long, media_time, config)
                    stats["updates"] += 1
                active_label = chosen

        action, active_public = _emit_action(
            prepared, step, chosen, reason, active_public
        )
        if action is not None:
            actions.append(action)

    stats["identity_count"] = len(identities)
    stats["unclaimed_identity_count"] = sum(not item.final_labels for item in identities)
    stats["final_aliases"] = dict(final_to_identity)
    return actions, stats


def _oracle_projection(
    prepared: _PreparedTape,
    *,
    future_centroid: bool = False,
    short_weight: float = 0.85,
) -> tuple[list[tuple[float, int, str, dict[str, Any]]], dict[str, Any]]:
    segments = read_canonical_segments(prepared.canonical_path)
    oracle_labels = {speaker: f"O{index + 1}" for index, speaker in enumerate(sorted({str(row['speaker']) for row in segments}))}
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    if future_centroid:
        by_speaker_short: dict[str, list[np.ndarray]] = {}
        by_speaker_long: dict[str, list[np.ndarray]] = {}
        for step in prepared.steps:
            t = float(step.payload.get("media_time") or 0.0)
            label = _canonical_at(segments, t)
            short = _unit(step.payload.get("embedding"))
            long = _unit(step.payload.get("context_embedding"))
            if label and short is not None:
                by_speaker_short.setdefault(label, []).append(short)
                if long is not None:
                    by_speaker_long.setdefault(label, []).append(long)
        for label, values in by_speaker_short.items():
            short = _unit(np.mean(values, axis=0))
            longs = by_speaker_long.get(label) or []
            centroids[label] = (short, _unit(np.mean(longs, axis=0)) if longs else None)

    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    active = ""
    for step in prepared.steps:
        payload = step.payload
        t = float(payload.get("media_time") or 0.0)
        truth = _canonical_at(segments, t)
        chosen = oracle_labels.get(truth, "")
        reason = "oracle_truth_at_probe"
        if future_centroid and bool(payload.get("speech")):
            short = _unit(payload.get("embedding"))
            long = _unit(payload.get("context_embedding"))
            ranked: list[tuple[float, str]] = []
            for label, (short_center, long_center) in centroids.items():
                ss = _cosine(short, short_center)
                ls = _cosine(long, long_center)
                score = short_weight * ss + (1.0 - short_weight) * (ls if long is not None and long_center is not None else ss)
                ranked.append((score, label))
            ranked.sort(reverse=True)
            chosen = oracle_labels.get(ranked[0][1], "") if ranked else ""
            reason = "invalid_future_centroid_representation_oracle"
        if not bool(payload.get("speech")) or bool(payload.get("release_signal")):
            chosen = ""
        action, active = _emit_action(prepared, step, chosen, reason, active)
        if action is not None:
            actions.append(action)
    return actions, {"oracle_labels": oracle_labels, "future_centroid": future_centroid}


def _evaluate(
    prepared_tapes: list[_PreparedTape],
    *,
    config: SegmentalConfig | None = None,
    oracle: str = "",
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for prepared in prepared_tapes:
        if oracle == "truth":
            actions, stats = _oracle_projection(prepared)
        elif oracle == "future_centroid":
            actions, stats = _oracle_projection(prepared, future_centroid=True)
        elif config is not None:
            actions, stats = _segmental_projection(prepared, config)
        else:
            raise ValueError("config or oracle required")
        score = _replay_with_tracklet_actions(prepared, actions)
        runs.append(
            {
                "video_id": prepared.video_id,
                "run_id": prepared.run_id,
                "score": float(score["strict_browser_live_score"]),
                "correct_live_speaker_coverage": float(score["correct_live_speaker_coverage"]),
                "wrong_live_speech_ratio": float(score["wrong_live_speech_ratio"]),
                "missing_live_speech_ratio": float(score["missing_live_speech_ratio"]),
                "outside_speech_live_ratio": float(score["outside_speech_live_ratio"]),
                "correct_live_precision_during_speech": float(score["correct_live_precision_during_speech"]),
                "identity_errors": _identity_error_diagnostics(score),
                "decoder_stats": stats,
            }
        )
    by_video: dict[str, list[float]] = {}
    for item in runs:
        by_video.setdefault(item["video_id"], []).append(item["score"])
    per_video = {key: mean(values) for key, values in sorted(by_video.items())}
    return {
        "name": oracle or (config.name if config is not None else ""),
        "config": None if config is None else asdict(config),
        "macro_score": mean(per_video.values()),
        "per_video": per_video,
        "runs": runs,
    }


def _candidate_grid() -> list[SegmentalConfig]:
    result: list[SegmentalConfig] = []
    # A compact, hypothesis-driven grid.  Full reducer scoring is expensive,
    # so only the difficult videos screen the grid before full-seven reruns.
    for new_ceiling, acquire in ((0.22, 0.32), (0.28, 0.34), (0.34, 0.36)):
        for switch_min, switch_margin in ((0.30, 0.02), (0.34, 0.04), (0.38, 0.06)):
            for top_k in (1, 3):
                result.append(
                    SegmentalConfig(
                        name=f"seg_n{new_ceiling:.2f}_a{acquire:.2f}_s{switch_min:.2f}_m{switch_margin:.2f}_k{top_k}",
                        new_ceiling=new_ceiling,
                        acquire_min=acquire,
                        switch_min=switch_min,
                        switch_margin=switch_margin,
                        top_k=top_k,
                    )
                )
    return result


def _macro_loss_components(result: dict[str, Any]) -> dict[str, float]:
    """Return the additive strict-score loss, macro-averaged by video."""

    by_video: dict[str, list[dict[str, float]]] = {}
    for run in result.get("runs") or []:
        missing = float(run["missing_live_speech_ratio"])
        wrong = float(run["wrong_live_speech_ratio"])
        outside = float(run["outside_speech_live_ratio"])
        score = float(run["score"])
        flicker_loss = max(
            0.0,
            1.0 - score - missing - 2.0 * wrong - 0.25 * outside,
        )
        by_video.setdefault(str(run["video_id"]), []).append(
            {
                "missing_loss": missing,
                "wrong_loss": 2.0 * wrong,
                "outside_loss": 0.25 * outside,
                "flicker_loss": flicker_loss,
            }
        )
    per_video: dict[str, dict[str, float]] = {}
    for video_id, rows in sorted(by_video.items()):
        per_video[video_id] = {
            key: mean(row[key] for row in rows)
            for key in rows[0]
        }
    macro = {
        key: mean(row[key] for row in per_video.values())
        for key in next(iter(per_video.values()))
    }
    macro["total_loss"] = sum(macro.values())
    return {"macro": macro, "per_video": per_video}


def _final_rebased_hybrid_report(
    prepared: list[_PreparedTape],
    base_config: dict[str, Any],
    base_artifact: Path,
) -> dict[str, Any]:
    common = dict(
        novelty_short_ceiling=0.20,
        novelty_long_ceiling=0.25,
        pending_short_min=0.30,
        pending_long_min=0.25,
        reuse_short_min=0.40,
        reuse_long_min=0.30,
        require_long_for_confirmation=False,
    )
    tracklet_config = TrackletConfig(
        name="short_history_tracklet_rebased", **common
    )
    hybrid_config = TrackletConfig(
        name="dual_scale_weak_reactivation_rebased",
        relaxed_reuse_short_min=0.25,
        relaxed_reuse_long_min=0.45,
        relaxed_reuse_known_advantage_margin=0.0,
        **common,
    )
    bayes = _evaluate_baseline(prepared, base_config)
    tracklet = _evaluate_tracklet_variant(prepared, tracklet_config)
    hybrid = _evaluate_tracklet_variant(prepared, hybrid_config)
    truth_oracle = _evaluate(prepared, oracle="truth")
    representation_oracle = _evaluate(prepared, oracle="future_centroid")
    for item in (bayes, tracklet, hybrid, truth_oracle, representation_oracle):
        item["loss_components"] = _macro_loss_components(item)
    gap_components = {
        key: (
            tracklet["loss_components"]["macro"][key]
            - representation_oracle["loss_components"]["macro"][key]
        )
        for key in ("missing_loss", "wrong_loss", "outside_loss", "flicker_loss")
    }
    gap_components["score_gap"] = (
        representation_oracle["macro_score"] - tracklet["macro_score"]
    )
    return {
        "contract_id": CONTRACT_ID,
        "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "optimization_eligible": False,
        "future_leakage_in_practical_candidate": False,
        "provider": "speechbrain_resnet",
        "windows_seconds": [0.7, 1.5],
        "model_inference_performed": False,
        "base_artifact": str(base_artifact.resolve()),
        "base_candidate_id": "wt-d2dc9dd7aa0da36786dba22b",
        "bayes_nominee": bayes,
        "short_history_tracklet": tracklet,
        "dual_scale_weak_reactivation_hybrid": hybrid,
        "exact_probe_timing_truth_oracle": truth_oracle,
        "invalid_future_centroid_representation_oracle": representation_oracle,
        "tracklet_to_representation_oracle_gap_by_loss": gap_components,
        "deltas": {
            "tracklet_vs_bayes": tracklet["macro_score"] - bayes["macro_score"],
            "hybrid_vs_tracklet": hybrid["macro_score"] - tracklet["macro_score"],
            "hybrid_vs_bayes": hybrid["macro_score"] - bayes["macro_score"],
            "truth_oracle_vs_hybrid": truth_oracle["macro_score"] - hybrid["macro_score"],
        },
        "integration_implications": {
            "ordinary_reuse": "unchanged short-history tracklet rule at short >= 0.40",
            "weak_band_reuse": "allow short >= 0.25 only when the separate 1.5-second tracklet history is >= 0.45",
            "known_advantage_margin": 0.0,
            "profile_merge": "keep max-cosine causal publication merge; sentence-interval recency replacement was rejected",
            "compute": "no extra window, provider, model inference, or cadence",
            "promotion": "must pass immutable interleaved authentic visible-Chrome wall-clock 1x 3+3 GUI gate",
        },
        "rejected_ablations": {
            "exclusive_profile_merge_only": 0.7818112857142857,
            "sentence_interval_recency_merge": 0.7714695238095238,
            "sentence_interval_recency_plus_exclusive": 0.7723347142857143,
            "wholesale_segmental_exemplar_decoder": 0.7323469523809524,
            "multi_prototype_bank_reason": "false reactivation of old exemplars, especially on onHU/S_o3",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", default="runtime/optimization/live_speaker_world_tapes_20260721")
    parser.add_argument("--base-artifact", default="runtime/optimization/live_speaker_night_20260722/e2e_hold250_v3/candidate_hold250_v3.json")
    parser.add_argument("--output", default="runtime/optimization/live_speaker_night_20260722/segmental_dp_report.json")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--final-rebased-hybrid", action="store_true")
    args = parser.parse_args()

    root = Path(args.campaign_root).resolve()
    parity = json.loads((root / "baseline_parity_report.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(Path(args.base_artifact).resolve())
    prepared = [_prepare_tape(run, base_config) for run in parity.get("runs") or []]
    if args.final_rebased_hybrid:
        report = _final_rebased_hybrid_report(
            prepared, base_config, Path(args.base_artifact)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "bayes": report["bayes_nominee"]["macro_score"],
                    "tracklet": report["short_history_tracklet"]["macro_score"],
                    "hybrid": report["dual_scale_weak_reactivation_hybrid"]["macro_score"],
                    "truth_oracle": report["exact_probe_timing_truth_oracle"]["macro_score"],
                    "representation_oracle": report["invalid_future_centroid_representation_oracle"]["macro_score"],
                    "deltas": report["deltas"],
                    "gap": report["tracklet_to_representation_oracle_gap_by_loss"],
                    "output": str(output.resolve()),
                },
                indent=2,
            )
        )
        return 0
    baseline = _evaluate_baseline(prepared, base_config)
    truth_oracle = _evaluate(prepared, oracle="truth")
    representation_oracle = _evaluate(prepared, oracle="future_centroid")

    if args.screen:
        screen_ids = {"20v1OxUXcQY", "JWS-qfR6K3w", "pD4IdQTmneI"}
        screen_tapes = [item for item in prepared if item.video_id in screen_ids]
        screened = [_evaluate(screen_tapes, config=item) for item in _candidate_grid()]
        screened.sort(key=lambda item: item["macro_score"], reverse=True)
        finalists = [
            _evaluate(prepared, config=SegmentalConfig(**item["config"]))
            for item in screened[:4]
        ]
    else:
        screened = []
        finalists = [_evaluate(prepared, config=SegmentalConfig())]
    finalists.sort(key=lambda item: item["macro_score"], reverse=True)
    report = {
        "contract_id": CONTRACT_ID,
        "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "optimization_eligible": False,
        "provider": "speechbrain_resnet",
        "windows_seconds": [0.7, 1.5],
        "model_inference_performed": False,
        "future_leakage_in_practical_candidates": False,
        "baseline": baseline,
        "oracle_ceiling": truth_oracle,
        "invalid_future_centroid_representation_oracle": representation_oracle,
        "screen": screened,
        "finalists": finalists,
        "best": finalists[0],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline["macro_score"],
        "oracle": truth_oracle["macro_score"],
        "representation_oracle": representation_oracle["macro_score"],
        "best": finalists[0]["macro_score"],
        "per_video": finalists[0]["per_video"],
        "output": str(output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
