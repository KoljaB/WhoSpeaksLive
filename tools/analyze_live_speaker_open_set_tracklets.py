"""Offline-only open-set tracklet experiments over authentic World Tapes.

This file deliberately lives outside ``src``.  It reuses the immutable 0.7/1.5 s
SpeechBrain vectors, recorded profile availability, public-event timing, browser
reducer, DOM sample clock, and strict browser score.  It performs no model
inference and is never production-promotion evidence.

The experimental state machine keeps a novel voice hidden for one probe, creates
a temporary identity only after a second self-consistent probe, and causally
aliases a newly published final profile to the temporary identity when their
embeddings agree.  The alias event also reconciles browser reducer state at that
instant; it never rewrites samples from before profile publication.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable

import numpy as np

from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, CausalBayesSpeakerTracker
from window.live_speaker_browser_parity import (
    BrowserLiveSpeakerReducer,
    _cached_browser_tape_inputs,
)
from window.live_speaker_counterfactual import (
    _cached_counterfactual_tape_inputs,
    _config_values,
    _mapped_values,
    _public_probability_key,
    _step,
    evaluate_counterfactual,
)
from window.live_speaker_probe_scoring import read_canonical_segments


CONTRACT_ID = "whospeaks.live_world_tape.open_set_tracklet_diagnostic.v1"
REQUIRED_WINDOWS = (0.7, 1.5)
REQUIRED_PROVIDER = "speechbrain_resnet"


def _unit(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return array / norm


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


def _max_cosine(value: np.ndarray | None, candidates: Iterable[np.ndarray]) -> float:
    return max((_cosine(value, item) for item in candidates), default=-1.0)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrackletConfig:
    name: str = "two_probe_tracklet_merge"
    min_confirmation_probes: int = 2
    novelty_short_ceiling: float = 0.30
    novelty_long_ceiling: float = 0.35
    # Causal short-window attack mode: at a likely turn boundary, let the
    # responsive short window open a two-probe novelty candidate even while
    # the longer context still contains the previous speaker.  Confirmation
    # remains subject to the normal pending self-consistency gate.
    short_attack_novelty: bool = False
    short_attack_novelty_ceiling: float = 0.30
    # A short-attack identity is only a causal bridge until a finalized
    # sentence profile claims it.  Retire unclaimed bridges after this lease;
    # negative disables expiry and preserves the historical behavior.
    short_attack_unclaimed_lease_seconds: float = -1.0
    # Quarantine mode: a short-window novelty attack gets priority over a
    # matching stale/adapted tracklet, preventing that observation from
    # updating and dragging the old identity toward the new voice.
    short_attack_preempts_tracklet_reuse: bool = False
    pending_short_min: float = 0.35
    pending_long_min: float = 0.30
    reuse_short_min: float = 0.40
    reuse_long_min: float = 0.30
    known_advantage_margin: float = 0.02
    active_reuse_short_min: float = 2.0
    active_reuse_long_min: float = 2.0
    active_reuse_max_gap_seconds: float = 1.60
    merge_min_similarity: float = 0.35
    pending_max_gap_seconds: float = 1.20
    tracklet_update_alpha: float = 0.25
    require_long_for_confirmation: bool = True
    enable_temporary_identity: bool = True
    enable_profile_merge: bool = True
    max_tracklets: int = 12
    single_slot_lease: bool = False
    lease_dies_on_silence: bool = False
    require_base_known_for_novelty: bool = False
    minimum_final_profile_count_for_novelty: int = 0
    profile_merge_survivor: str = "temporary"
    # Diagnostic-only ablation: once a final profile claims a temporary
    # identity, do not let a different final profile claim the same identity.
    # The historical prototype allowed this and could collapse two real
    # speakers onto one public key.
    exclusive_profile_merge: bool = False
    profile_merge_sentence_overlap: bool = False
    profile_merge_overlap_min_similarity: float = 0.20
    profile_merge_interval_padding_seconds: float = 0.35
    profile_merge_min_activity_points: int = 1
    multi_prototype_bank: bool = False
    prototype_bank_size: int = 8
    prototype_min_gap_seconds: float = 0.75
    prototype_top_k: int = 3
    prototype_best_weight: float = 0.65
    relaxed_reuse_short_min: float = 2.0
    relaxed_reuse_long_min: float = 2.0
    relaxed_reuse_known_advantage_margin: float = 0.0
    relaxed_reuse_require_final_profile: bool = False
    relaxed_reuse_min_probe_count: int = 0
    relaxed_reuse_max_gap_seconds: float = -1.0


@dataclass
class _Tracklet:
    label: str
    short_centroid: np.ndarray
    long_centroid: np.ndarray | None
    probe_count: int
    created_media_time: float
    last_media_time: float
    final_label: str = ""
    activity_times: list[float] | None = None
    short_exemplars: list[np.ndarray] | None = None
    long_exemplars: list[np.ndarray] | None = None
    multi_prototype_bank: bool = False
    prototype_bank_size: int = 8
    prototype_min_gap_seconds: float = 0.75
    prototype_top_k: int = 3
    prototype_best_weight: float = 0.65
    last_prototype_media_time: float = -1e9
    short_attack_origin: bool = False

    def _prototype_score(
        self,
        value: np.ndarray | None,
        centroid: np.ndarray | None,
        exemplars: list[np.ndarray] | None,
    ) -> float:
        if value is None or centroid is None:
            return -1.0
        if not self.multi_prototype_bank or not exemplars:
            return _cosine(centroid, value)
        scores = [_cosine(value, centroid)] + [
            _cosine(value, item) for item in exemplars
        ]
        scores.sort(reverse=True)
        count = min(max(1, int(self.prototype_top_k)), len(scores))
        best_weight = max(0.0, min(1.0, float(self.prototype_best_weight)))
        return best_weight * scores[0] + (1.0 - best_weight) * mean(scores[:count])

    def short_similarity(self, value: np.ndarray | None) -> float:
        return self._prototype_score(value, self.short_centroid, self.short_exemplars)

    def long_similarity(self, value: np.ndarray | None) -> float:
        if value is None:
            return -1.0
        return self._prototype_score(value, self.long_centroid, self.long_exemplars)

    def profile_similarity(self, value: np.ndarray | None) -> float:
        if value is None:
            return -1.0
        scores = [self.short_similarity(value)]
        if self.long_centroid is not None:
            scores.append(self.long_similarity(value))
        return max(scores)

    def update(
        self,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
        alpha: float,
    ) -> None:
        weight = max(0.0, min(1.0, float(alpha)))
        self.short_centroid = _unit(
            (1.0 - weight) * self.short_centroid + weight * short
        )
        if long is not None:
            if self.long_centroid is None:
                self.long_centroid = long.copy()
            else:
                self.long_centroid = _unit(
                    (1.0 - weight) * self.long_centroid + weight * long
                )
        self.probe_count += 1
        self.last_media_time = media_time
        if self.activity_times is None:
            self.activity_times = []
        if not self.activity_times or media_time > self.activity_times[-1] + 1e-6:
            self.activity_times.append(media_time)
            del self.activity_times[: max(0, len(self.activity_times) - 256)]
        if (
            self.multi_prototype_bank
            and media_time - self.last_prototype_media_time
            >= self.prototype_min_gap_seconds
        ):
            if self.short_exemplars is None:
                self.short_exemplars = []
            self.short_exemplars.append(short.copy())
            del self.short_exemplars[: max(0, len(self.short_exemplars) - self.prototype_bank_size)]
            if long is not None:
                if self.long_exemplars is None:
                    self.long_exemplars = []
                self.long_exemplars.append(long.copy())
                del self.long_exemplars[: max(0, len(self.long_exemplars) - self.prototype_bank_size)]
            self.last_prototype_media_time = media_time


@lru_cache(maxsize=32)
def _profile_sentence_intervals(tape_dir: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with (Path(tape_dir) / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("event") or "") != "live_speaker_profile_snapshot":
                continue
            payload = dict(record.get("payload") or {})
            if str(payload.get("profile_embedding_provider") or "") != REQUIRED_PROVIDER:
                continue
            speaker_id = str(payload.get("speaker_id") or "")
            if not speaker_id or speaker_id in result:
                continue
            result[speaker_id] = {
                "sentence_start": float(payload.get("sentence_start") or 0.0),
                "sentence_end": float(payload.get("sentence_end") or 0.0),
            }
    return result


@dataclass
class _Pending:
    short: np.ndarray
    long: np.ndarray | None
    count: int
    last_media_time: float
    short_attack_origin: bool = False


@dataclass(frozen=True)
class _PreparedStep:
    payload: dict[str, Any]
    base_trace: dict[str, Any]
    wall_seconds: float
    sequence: int
    recorded_public_payload: dict[str, Any]
    public_aliases: dict[str, str]


@dataclass
class _PreparedTape:
    video_id: str
    run_id: str
    tape_dir: Path
    canonical_path: Path
    hold_seconds: float
    steps: tuple[_PreparedStep, ...]
    manifest: dict[str, Any]


def _load_base_config(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    config = dict(artifact.get("algorithm_config") or {})
    windows = tuple(float(item) for item in config.get("scale_windows") or ())
    if windows != REQUIRED_WINDOWS:
        raise ValueError(f"Expected exactly {REQUIRED_WINDOWS}, got {windows}")
    return config


def _assert_tape_contract(manifest: dict[str, Any], tape_dir: Path) -> None:
    runtime = dict(manifest.get("runtime_config") or {})
    provider = str(runtime.get("live_speaker_embedding_provider") or "")
    short = float(runtime.get("live_speaker_probe_window_seconds") or 0.0)
    long = float(runtime.get("live_speaker_probe_context_window_seconds") or 0.0)
    if provider != REQUIRED_PROVIDER:
        raise ValueError(f"{tape_dir}: expected {REQUIRED_PROVIDER}, got {provider}")
    if (short, long) != REQUIRED_WINDOWS:
        raise ValueError(f"{tape_dir}: expected windows {REQUIRED_WINDOWS}, got {(short, long)}")


def _prepare_tape(
    run: dict[str, Any], base_config: dict[str, Any]
) -> _PreparedTape:
    root = Path(run["tape_dir"]).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _assert_tape_contract(manifest, root)
    input_records, recorded_decisions, public_by_step, hold_seconds = (
        _cached_counterfactual_tape_inputs(str(root))
    )
    tracker = CausalBayesSpeakerTracker(
        BayesSpeakerTrackerConfig(
            **_config_values(BayesSpeakerTrackerConfig, base_config)
        )
    )
    aliases: dict[str, str] = {}
    steps: list[_PreparedStep] = []
    for record in input_records:
        payload = dict(record["payload"])
        if str(payload.get("algorithm_type") or "bayes") != "bayes":
            raise ValueError(f"{root}: non-Bayes input is outside this experiment")
        step_id = int(payload.get("step_id") or 0)
        tracker.sync_profiles(list(payload.get("profiles") or []))
        decision = _step(tracker, "bayes", payload)
        trace = decision.trace_record()
        recorded_decision = dict(
            (recorded_decisions.get(step_id) or {}).get("payload") or {}
        )
        recorded_public = public_by_step.get(step_id)
        recorded_public_payload = dict((recorded_public or {}).get("payload") or {})
        recorded_internal = str(recorded_decision.get("visible_speaker") or "")
        recorded_external = str(
            recorded_public_payload.get("assigned_speaker")
            or recorded_public_payload.get("speaker_id")
            or ""
        )
        if recorded_internal and recorded_external and recorded_internal != recorded_external:
            aliases[recorded_internal] = recorded_external
        event_record = recorded_public or recorded_decisions.get(step_id) or record
        steps.append(
            _PreparedStep(
                payload=payload,
                base_trace=trace,
                wall_seconds=float(event_record.get("wall_us") or 0) / 1_000_000.0,
                sequence=int(event_record.get("seq") or record.get("seq") or 0),
                recorded_public_payload=recorded_public_payload,
                public_aliases=dict(aliases),
            )
        )
    return _PreparedTape(
        video_id=str(run["video_id"]),
        run_id=str(run["run_id"]),
        tape_dir=root,
        canonical_path=Path(run["canonical_path"]).resolve(),
        hold_seconds=max(0.0, float(base_config.get("live_speaker_probe_hold_seconds", hold_seconds))),
        steps=tuple(steps),
        manifest=manifest,
    )


def _profile_vectors(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for profile in payload.get("profiles") or []:
        label = str(profile.get("label") or "")
        centroid = _unit(profile.get("centroid"))
        if label and centroid is not None:
            result[label] = centroid
    return result


def _two_scale_pass(
    short_value: float,
    long_value: float,
    short_min: float,
    long_min: float,
    *,
    long_available: bool,
    require_long: bool,
) -> bool:
    if short_value < short_min:
        return False
    if long_available and require_long and long_value < long_min:
        return False
    return True


def _reconcile_reducer_alias(
    reducer: BrowserLiveSpeakerReducer, old_label: str, new_label: str
) -> None:
    """Apply a causal public-identity merge at profile-publication time."""

    if not old_label or not new_label or old_label == new_label:
        return
    for item in reducer.timeline:
        if str(item.get("speaker") or "") == old_label:
            item["speaker"] = new_label
    for row in reducer.rows:
        if row.raw_speaker == old_label:
            row.raw_speaker = new_label
        if row.speaker == old_label:
            row.speaker = new_label
    if reducer.fallback_speaker == old_label:
        reducer.fallback_speaker = new_label
    if reducer.transcript_speaker == old_label:
        reducer.transcript_speaker = new_label
    if reducer.current_speaker == old_label:
        reducer.current_speaker = new_label
    reducer._refresh_rows()


def _map_public_payload(payload: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("assigned_speaker", "speaker_id", "replaces_speaker_id"):
        value = str(result.get(key) or "")
        if value in aliases:
            result[key] = aliases[value]
    return result


def _replay_with_tracklet_actions(
    prepared: _PreparedTape,
    replacement_actions: list[tuple[float, int, str, dict[str, Any]]],
) -> dict[str, Any]:
    """Use the unchanged production-shaped reducer and exact recorded clocks."""

    manifest, cached_actions, recorded_samples = _cached_browser_tape_inputs(
        str(prepared.tape_dir)
    )
    actions = [
        item
        for item in cached_actions
        if item[2] not in {"live_speaker", "live_speaker_clear"}
    ]
    actions.extend(replacement_actions)
    actions.sort(key=lambda item: (item[0], item[1], 0 if item[2] == "tracklet_alias" else 1))
    reducer = BrowserLiveSpeakerReducer(manifest.get("runtime_config") or {})
    aliases: dict[str, str] = {}
    predicted: list[dict[str, Any]] = []
    action_index = 0
    for sample_wall, _sample_sequence, recorded in recorded_samples:
        while action_index < len(actions) and actions[action_index][0] <= sample_wall:
            wall, _sequence, event, raw_payload = actions[action_index]
            if event == "tracklet_alias":
                old_label = str(raw_payload.get("old_label") or "")
                new_label = str(raw_payload.get("new_label") or "")
                aliases[old_label] = new_label
                _reconcile_reducer_alias(reducer, old_label, new_label)
            else:
                reducer.apply(event, _map_public_payload(raw_payload, aliases), wall)
            action_index += 1
        predicted.append(reducer.sample(recorded, sample_wall))
    return score_browser_live_speaker_samples(
        predicted,
        read_canonical_segments(prepared.canonical_path),
    )


def _identity_error_diagnostics(score: dict[str, Any]) -> dict[str, Any]:
    """Separate public-ID fragmentation from cross-speaker identity mixing."""

    dominant_by_canonical: dict[str, list[tuple[float, str]]] = {}
    false_merge_profiles: list[dict[str, Any]] = []
    secondary_overlap_seconds = 0.0
    for row in score.get("overlap_matrix") or []:
        profile = str(row.get("speaker") or "")
        overlaps = [
            (str(label), float(seconds))
            for label, seconds in dict(row.get("canonical_overlaps") or {}).items()
            if float(seconds) > 0.0
        ]
        overlaps.sort(key=lambda item: item[1], reverse=True)
        if not overlaps:
            continue
        dominant_by_canonical.setdefault(overlaps[0][0], []).append((overlaps[0][1], profile))
        secondary = sum(value for _label, value in overlaps[1:])
        secondary_overlap_seconds += secondary
        total = sum(value for _label, value in overlaps)
        if secondary >= 1.0 and secondary / max(0.001, total) >= 0.15:
            false_merge_profiles.append(
                {
                    "profile": profile,
                    "dominant": overlaps[0][0],
                    "secondary_overlap_seconds": round(secondary, 4),
                    "secondary_share": round(secondary / total, 6),
                }
            )
    split_profiles: list[dict[str, Any]] = []
    split_overlap_seconds = 0.0
    for canonical, values in sorted(dominant_by_canonical.items()):
        material = sorted(
            [(seconds, profile) for seconds, profile in values if seconds >= 1.0],
            reverse=True,
        )
        if len(material) <= 1:
            continue
        extras = material[1:]
        split_overlap_seconds += sum(seconds for seconds, _profile in extras)
        split_profiles.append(
            {
                "canonical_speaker": canonical,
                "public_profiles": [profile for _seconds, profile in material],
                "extra_profile_count": len(extras),
                "extra_dominant_overlap_seconds": round(
                    sum(seconds for seconds, _profile in extras), 4
                ),
            }
        )
    return {
        "false_merge_profile_count": len(false_merge_profiles),
        "false_merge_secondary_overlap_seconds": round(secondary_overlap_seconds, 4),
        "false_merge_profiles": false_merge_profiles,
        "split_canonical_speaker_count": len(split_profiles),
        "split_extra_profile_count": sum(
            int(item["extra_profile_count"]) for item in split_profiles
        ),
        "split_extra_dominant_overlap_seconds": round(split_overlap_seconds, 4),
        "split_profiles": split_profiles,
    }


def _best_tracklet(
    tracklets: list[_Tracklet],
    short: np.ndarray | None,
    long: np.ndarray | None,
) -> tuple[_Tracklet | None, float, float]:
    best: tuple[float, _Tracklet, float, float] | None = None
    for item in tracklets:
        short_score = item.short_similarity(short)
        long_score = item.long_similarity(long)
        score = 0.7 * short_score + 0.3 * (long_score if long is not None else short_score)
        if best is None or score > best[0]:
            best = (score, item, short_score, long_score)
    if best is None:
        return None, -1.0, -1.0
    return best[1], best[2], best[3]


def _tracklet_projection(
    prepared: _PreparedTape,
    config: TrackletConfig,
) -> tuple[list[tuple[float, int, str, dict[str, Any]]], dict[str, Any]]:
    tracklets: list[_Tracklet] = []
    pending: _Pending | None = None
    known_labels: set[str] = set()
    final_to_tracklet: dict[str, str] = {}
    final_to_public: dict[str, str] = {}
    tracklet_to_final: dict[str, str] = {}
    active_public = ""
    active_tracklet_label = ""
    next_tracklet_index = 1
    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    stats = {
        "pending_started": 0,
        "pending_confirmed": 0,
        "pending_rejected": 0,
        "tracklets_created": 0,
        "tracklet_reuses": 0,
        "profile_merges": 0,
        "novel_overrides": 0,
        "creation_events": [],
        "overlap_guided_profile_merges": 0,
        "profile_merge_events": [],
        "short_attack_tracklets_expired": 0,
    }
    publication_intervals = _profile_sentence_intervals(str(prepared.tape_dir))

    for step in prepared.steps:
        payload = step.payload
        media_time = float(payload.get("media_time") or 0.0)
        if config.short_attack_unclaimed_lease_seconds >= 0.0:
            retained: list[_Tracklet] = []
            for item in tracklets:
                expired = bool(
                    item.short_attack_origin
                    and not item.final_label
                    and media_time - item.created_media_time
                    > config.short_attack_unclaimed_lease_seconds
                )
                if expired:
                    stats["short_attack_tracklets_expired"] += 1
                    if active_tracklet_label == item.label:
                        active_tracklet_label = ""
                    continue
                retained.append(item)
            tracklets = retained
        profiles = _profile_vectors(payload)
        added_labels = sorted(set(profiles) - known_labels)
        known_labels = set(profiles)

        # A final profile may only merge at the first core step where production
        # memory already exposes it.  No future profile is visible earlier.
        if config.enable_profile_merge:
            resolved_final_labels = set(final_to_tracklet) | set(final_to_public)
            for final_label in sorted(set(profiles) - resolved_final_labels):
                available = [
                    item
                    for item in tracklets
                    if not config.exclusive_profile_merge or not item.final_label
                ]
                if not available:
                    continue
                profile = profiles[final_label]
                best = None
                similarity = -1.0
                merge_floor = config.merge_min_similarity
                overlap_guided = False
                interval = publication_intervals.get(final_label)
                if config.profile_merge_sentence_overlap and interval is not None:
                    left = (
                        interval["sentence_start"]
                        - config.profile_merge_interval_padding_seconds
                    )
                    right = (
                        interval["sentence_end"]
                        + config.profile_merge_interval_padding_seconds
                    )
                    ranked = []
                    for item in available:
                        activity = sum(
                            left <= value <= right
                            for value in (item.activity_times or [])
                        )
                        ranked.append((activity, item.profile_similarity(profile), item))
                    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
                    if (
                        ranked
                        and ranked[0][0] >= config.profile_merge_min_activity_points
                    ):
                        _activity, similarity, best = ranked[0]
                        merge_floor = config.profile_merge_overlap_min_similarity
                        overlap_guided = True
                if best is None:
                    best = max(
                        available,
                        key=lambda item: item.profile_similarity(profile),
                    )
                    similarity = best.profile_similarity(profile)
                if similarity < merge_floor:
                    continue
                best.final_label = final_label
                final_public = step.public_aliases.get(final_label, final_label)
                if config.profile_merge_survivor == "final":
                    survivor = tracklet_to_final.get(best.label, final_public)
                    old_label = best.label if best.label not in tracklet_to_final else final_public
                    tracklet_to_final[best.label] = survivor
                    final_to_public[final_label] = survivor
                    new_label = survivor
                else:
                    final_to_tracklet[final_label] = best.label
                    old_label = final_public
                    new_label = best.label
                actions.append(
                    (
                        step.wall_seconds,
                        max(0, step.sequence - 1),
                        "tracklet_alias",
                        {
                            "old_label": old_label,
                            "new_label": new_label,
                            "similarity": similarity,
                        },
                    )
                )
                stats["profile_merges"] += 1
                stats["overlap_guided_profile_merges"] += int(overlap_guided)
                stats["profile_merge_events"].append(
                    {
                        "media_time": round(media_time, 4),
                        "final_label": final_label,
                        "tracklet_label": best.label,
                        "similarity": round(float(similarity), 6),
                        "short_attack_origin": bool(best.short_attack_origin),
                    }
                )
                if active_public == old_label:
                    active_public = new_label
                if config.single_slot_lease:
                    tracklets.remove(best)

        short = _unit(payload.get("embedding"))
        long = _unit(payload.get("context_embedding"))
        duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
        dedicated = bool(str(payload.get("probe_id") or ""))
        speech = bool(payload.get("speech")) and short is not None
        release = bool(payload.get("release_signal"))

        base_trace = step.base_trace
        base_internal = str(base_trace.get("visible_speaker") or "")
        base_public = step.public_aliases.get(base_internal, base_internal)
        base_public = final_to_public.get(
            base_internal,
            final_to_tracklet.get(base_internal, base_public),
        )
        chosen = base_public
        reason = str(base_trace.get("reason") or "")

        known_short = _max_cosine(short, profiles.values())
        known_long = _max_cosine(long, profiles.values())
        tracklet, track_short, track_long = _best_tracklet(tracklets, short, long)
        active_tracklet = next(
            (item for item in tracklets if item.label == active_tracklet_label), None
        )
        active_short = (
            active_tracklet.short_similarity(short) if active_tracklet is not None else -1.0
        )
        active_long = (
            active_tracklet.long_similarity(long) if active_tracklet is not None else -1.0
        )
        active_tracklet_pass = bool(
            active_tracklet
            and media_time - active_tracklet.last_media_time
            <= config.active_reuse_max_gap_seconds
            and _two_scale_pass(
                active_short,
                active_long,
                config.active_reuse_short_min,
                config.active_reuse_long_min,
                long_available=long is not None,
                require_long=config.require_long_for_confirmation,
            )
        )
        normal_tracklet_pass = bool(
            tracklet
            and _two_scale_pass(
                track_short,
                track_long,
                config.reuse_short_min,
                config.reuse_long_min,
                long_available=long is not None,
                require_long=config.require_long_for_confirmation,
            )
            and (
                not profiles
                or track_short >= known_short + config.known_advantage_margin
            )
        )
        relaxed_tracklet_pass = bool(
            tracklet
            and short is not None
            and long is not None
            and track_short >= config.relaxed_reuse_short_min
            and track_long >= config.relaxed_reuse_long_min
            and (
                not config.relaxed_reuse_require_final_profile
                or bool(tracklet.final_label)
            )
            and tracklet.probe_count >= config.relaxed_reuse_min_probe_count
            and (
                config.relaxed_reuse_max_gap_seconds < 0.0
                or media_time - tracklet.last_media_time
                <= config.relaxed_reuse_max_gap_seconds
            )
            and (
                not profiles
                or track_short
                >= known_short + config.relaxed_reuse_known_advantage_margin
            )
        )
        tracklet_pass = normal_tracklet_pass or relaxed_tracklet_pass
        conventional_novel_evidence = bool(
            not profiles
            or (
                known_short < config.novelty_short_ceiling
                and (
                    long is None
                    or known_long < config.novelty_long_ceiling
                )
            )
        )
        short_attack_novel_evidence = bool(
            config.short_attack_novelty
            and bool(profiles)
            and known_short < config.short_attack_novelty_ceiling
        )
        novel = bool(
            speech
            and (
                not config.require_base_known_for_novelty
                or bool(base_public)
            )
            and len(profiles) >= config.minimum_final_profile_count_for_novelty
            and (conventional_novel_evidence or short_attack_novel_evidence)
        )
        effective_tracklet_pass = bool(
            tracklet_pass
            and not (
                config.short_attack_preempts_tracklet_reuse
                and short_attack_novel_evidence
            )
        )

        if not dedicated or release or not speech:
            pending = None
            active_tracklet_label = ""
            if config.single_slot_lease and config.lease_dies_on_silence:
                tracklets.clear()
        elif active_tracklet_pass:
            chosen = active_tracklet.label
            reason = "open_set_active_tracklet_continuity"
            active_tracklet.update(
                short,
                long,
                media_time,
                config.tracklet_update_alpha,
            )
            pending = None
            stats["tracklet_reuses"] += 1
        elif effective_tracklet_pass:
            chosen = tracklet.label
            reason = "open_set_tracklet_reuse"
            tracklet.update(
                short,
                long,
                media_time,
                config.tracklet_update_alpha,
            )
            pending = None
            stats["tracklet_reuses"] += 1
        elif novel:
            stats["novel_overrides"] += int(bool(base_public))
            consistent = bool(
                pending
                and media_time - pending.last_media_time <= config.pending_max_gap_seconds
                and _two_scale_pass(
                    _cosine(pending.short, short),
                    _cosine(pending.long, long),
                    config.pending_short_min,
                    config.pending_long_min,
                    long_available=pending.long is not None and long is not None,
                    require_long=config.require_long_for_confirmation,
                )
            )
            if consistent:
                pending.count += 1
                pending.short = _unit(0.5 * pending.short + 0.5 * short)
                if long is not None:
                    pending.long = (
                        long.copy()
                        if pending.long is None
                        else _unit(0.5 * pending.long + 0.5 * long)
                    )
                pending.last_media_time = media_time
            else:
                if pending is not None:
                    stats["pending_rejected"] += 1
                pending = _Pending(
                    short.copy(),
                    None if long is None else long.copy(),
                    1,
                    media_time,
                    short_attack_origin=bool(
                        short_attack_novel_evidence
                        and not conventional_novel_evidence
                    ),
                )
                stats["pending_started"] += 1

            if (
                config.enable_temporary_identity
                and pending.count >= config.min_confirmation_probes
            ):
                candidate, candidate_short, candidate_long = _best_tracklet(
                    tracklets, pending.short, pending.long
                )
                reusable = bool(
                    candidate
                    and _two_scale_pass(
                        candidate_short,
                        candidate_long,
                        config.reuse_short_min,
                        config.reuse_long_min,
                        long_available=pending.long is not None,
                        require_long=config.require_long_for_confirmation,
                    )
                )
                if reusable:
                    selected = candidate
                    stats["tracklet_reuses"] += 1
                elif len(tracklets) < config.max_tracklets:
                    selected = _Tracklet(
                        label=f"T{next_tracklet_index}",
                        short_centroid=pending.short.copy(),
                        long_centroid=None if pending.long is None else pending.long.copy(),
                        probe_count=pending.count,
                        created_media_time=media_time,
                        last_media_time=media_time,
                        multi_prototype_bank=config.multi_prototype_bank,
                        prototype_bank_size=config.prototype_bank_size,
                        prototype_min_gap_seconds=config.prototype_min_gap_seconds,
                        prototype_top_k=config.prototype_top_k,
                        prototype_best_weight=config.prototype_best_weight,
                        short_attack_origin=bool(pending.short_attack_origin),
                    )
                    tracklets.append(selected)
                    next_tracklet_index += 1
                    stats["tracklets_created"] += 1
                    stats["creation_events"].append(
                        {
                            "media_time": round(media_time, 4),
                            "new_label": selected.label,
                            "nearest_existing_label": (
                                candidate.label if candidate is not None else ""
                            ),
                            "nearest_short_similarity": round(candidate_short, 6),
                            "nearest_long_similarity": round(candidate_long, 6),
                            "profile_count": len(profiles),
                        }
                    )
                else:
                    selected = None
                if selected is not None:
                    selected.update(short, long, media_time, config.tracklet_update_alpha)
                    chosen = selected.label
                    reason = "open_set_tracklet_confirmed"
                    stats["pending_confirmed"] += 1
                    active_tracklet_label = selected.label
                else:
                    chosen = ""
                    reason = "open_set_tracklet_capacity"
                pending = None
            else:
                chosen = ""
                reason = "open_set_pending_unknown"
        else:
            pending = None

        if chosen:
            chosen = tracklet_to_final.get(chosen, chosen)
            matching_tracklet = next(
                (
                    item
                    for item in tracklets
                    if item.label == chosen
                    or tracklet_to_final.get(item.label) == chosen
                ),
                None,
            )
            active_tracklet_label = (
                matching_tracklet.label if matching_tracklet is not None else ""
            )

        # Match the existing counterfactual public-action contract and timings.
        if dedicated and chosen and not release:
            probabilities = _mapped_values(
                dict(base_trace.get("probabilities") or {}),
                step.public_aliases,
                probability_keys=True,
            )
            raw_probabilities = _mapped_values(
                dict(base_trace.get("raw_probabilities") or {}),
                step.public_aliases,
                probability_keys=True,
            )
            similarities = _mapped_values(
                dict(base_trace.get("similarities") or {}), step.public_aliases
            )
            public_key = _public_probability_key(chosen)
            base = step.recorded_public_payload
            start = round(max(0.0, media_time - duration), 4)
            end = round(media_time, 4)
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
                "start": base.get("start", start),
                "end": base.get("end", end),
                "audio_length_seconds": base.get("audio_length_seconds", round(duration, 4)),
                "hold_seconds": round(prepared.hold_seconds, 4),
                "assignment_source": "offline_open_set_tracklet_diagnostic",
                "tracklet_probability_key": public_key,
                # Research-only causal diagnostics.  These values are already
                # available at this decision and do not affect the reducer.
                "diagnostic_known_short": round(float(known_short), 6),
                "diagnostic_known_long": round(float(known_long), 6),
                "diagnostic_tracklet_label": tracklet.label if tracklet is not None else "",
                "diagnostic_tracklet_short": round(float(track_short), 6),
                "diagnostic_tracklet_long": round(float(track_long), 6),
                "diagnostic_active_tracklet_label": (
                    active_tracklet.label if active_tracklet is not None else ""
                ),
                "diagnostic_active_short": round(float(active_short), 6),
                "diagnostic_active_long": round(float(active_long), 6),
                "diagnostic_normal_tracklet_pass": bool(normal_tracklet_pass),
                "diagnostic_relaxed_tracklet_pass": bool(relaxed_tracklet_pass),
                "diagnostic_novel": bool(novel),
                "diagnostic_short_attack_novel": bool(
                    short_attack_novel_evidence
                ),
            }
            actions.append((step.wall_seconds, step.sequence, "live_speaker", live_payload))
            active_public = chosen
        elif dedicated and active_public and (release or not chosen):
            base = step.recorded_public_payload
            clear_payload = {
                **base,
                "step_id": int(payload.get("step_id") or 0),
                "speaker_id": active_public,
                "assigned_speaker": None,
                "live": False,
                "fallback": True,
                "start": base.get("start", round(max(0.0, media_time - duration), 4)),
                "end": base.get("end", round(media_time, 4)),
                "reason": "silence" if release else "unknown",
                "assignment_source": "offline_open_set_tracklet_diagnostic",
            }
            actions.append((step.wall_seconds, step.sequence, "live_speaker_clear", clear_payload))
            active_public = ""

    stats["tracklet_count"] = int(stats["tracklets_created"])
    stats["active_tracklet_count"] = len(tracklets)
    stats["unmerged_tracklets"] = sum(not bool(item.final_label) for item in tracklets)
    stats["final_aliases"] = dict(final_to_tracklet)
    stats["final_public_aliases"] = dict(final_to_public)
    stats["tracklet_final_aliases"] = dict(tracklet_to_final)
    return actions, stats


def _evaluate_variant(
    prepared_tapes: list[_PreparedTape],
    config: TrackletConfig,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for prepared in prepared_tapes:
        actions, stats = _tracklet_projection(prepared, config)
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
                "tracklet_stats": stats,
                "identity_errors": _identity_error_diagnostics(score),
            }
        )
    by_video: dict[str, list[float]] = {}
    for item in runs:
        by_video.setdefault(str(item["video_id"]), []).append(float(item["score"]))
    per_video = {key: mean(values) for key, values in sorted(by_video.items())}
    return {
        "name": config.name,
        "config": asdict(config),
        "config_sha256": _stable_hash(asdict(config)),
        "macro_score": mean(per_video.values()),
        "per_video": per_video,
        "runs": runs,
    }


def _evaluate_baseline(
    prepared_tapes: list[_PreparedTape], base_config: dict[str, Any]
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for prepared in prepared_tapes:
        result = evaluate_counterfactual(
            prepared.tape_dir, base_config, prepared.canonical_path
        )
        score = result["score"]
        runs.append(
            {
                "video_id": prepared.video_id,
                "run_id": prepared.run_id,
                "score": float(result["strict_browser_live_score"]),
                "correct_live_speaker_coverage": float(score["correct_live_speaker_coverage"]),
                "wrong_live_speech_ratio": float(score["wrong_live_speech_ratio"]),
                "missing_live_speech_ratio": float(score["missing_live_speech_ratio"]),
                "outside_speech_live_ratio": float(score["outside_speech_live_ratio"]),
                "correct_live_precision_during_speech": float(score["correct_live_precision_during_speech"]),
            }
        )
    by_video: dict[str, list[float]] = {}
    for item in runs:
        by_video.setdefault(str(item["video_id"]), []).append(float(item["score"]))
    per_video = {key: mean(values) for key, values in sorted(by_video.items())}
    return {
        "name": "incumbent",
        "algorithm_config": base_config,
        "macro_score": mean(per_video.values()),
        "per_video": per_video,
        "runs": runs,
    }


def _candidate_grid() -> list[TrackletConfig]:
    configs: list[TrackletConfig] = []
    for novelty_short in (0.20, 0.30, 0.40):
        for novelty_long in (0.25, 0.35, 0.45):
            for pending_short in (0.25, 0.40, 0.55):
                for pending_long in (0.20, 0.35, 0.50):
                    for merge_min in (0.20, 0.35, 0.50):
                        configs.append(
                            TrackletConfig(
                                name=(
                                    f"trk_ns{novelty_short:.2f}_nl{novelty_long:.2f}_"
                                    f"ps{pending_short:.2f}_pl{pending_long:.2f}_m{merge_min:.2f}"
                                ),
                                novelty_short_ceiling=novelty_short,
                                novelty_long_ceiling=novelty_long,
                                pending_short_min=pending_short,
                                pending_long_min=pending_long,
                                reuse_short_min=max(0.30, pending_short),
                                reuse_long_min=max(0.20, pending_long),
                                merge_min_similarity=merge_min,
                            )
                        )
    return configs


def _bounded_candidate_grid() -> list[TrackletConfig]:
    """Conservative bank search weighted toward avoiding identity fragmentation."""

    configs: list[TrackletConfig] = []
    novelty_pairs = ((0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35))
    pending_pairs = ((0.30, 0.25), (0.40, 0.35), (0.50, 0.45))
    merge_values = (0.25, 0.35, 0.45)
    for novelty_short, novelty_long in novelty_pairs:
        for pending_short, pending_long in pending_pairs:
            for merge_min in merge_values:
                for max_tracklets in (4, 8):
                    configs.append(
                        TrackletConfig(
                            name=(
                                f"bounded{max_tracklets}_ns{novelty_short:.2f}_"
                                f"nl{novelty_long:.2f}_ps{pending_short:.2f}_"
                                f"pl{pending_long:.2f}_m{merge_min:.2f}"
                            ),
                            novelty_short_ceiling=novelty_short,
                            novelty_long_ceiling=novelty_long,
                            pending_short_min=pending_short,
                            pending_long_min=pending_long,
                            reuse_short_min=0.22,
                            reuse_long_min=0.15,
                            known_advantage_margin=-0.03,
                            merge_min_similarity=merge_min,
                            max_tracklets=max_tracklets,
                        )
                    )
    return configs


def _active_continuity_candidates() -> list[TrackletConfig]:
    configs: list[TrackletConfig] = [TrackletConfig(name="active_continuity_disabled")]
    for short_min, long_min in (
        (0.25, 0.15),
        (0.30, 0.20),
        (0.35, 0.25),
        (0.40, 0.30),
        (0.45, 0.35),
        (0.50, 0.40),
    ):
        for max_gap in (0.8, 1.2):
            configs.append(
                TrackletConfig(
                    name=f"active_s{short_min:.2f}_l{long_min:.2f}_g{max_gap:.1f}",
                    active_reuse_short_min=short_min,
                    active_reuse_long_min=long_min,
                    active_reuse_max_gap_seconds=max_gap,
                )
            )
    return configs


def _change_point_lease_candidates() -> list[TrackletConfig]:
    configs: list[TrackletConfig] = []
    for minimum_profiles in (1, 2, 3):
        for dies_on_silence in (True, False):
            configs.append(
                TrackletConfig(
                    name=(
                        f"change_gate_single_slot_p{minimum_profiles}_"
                        f"{'dies' if dies_on_silence else 'persists'}"
                    ),
                    max_tracklets=1,
                    single_slot_lease=True,
                    lease_dies_on_silence=dies_on_silence,
                    require_base_known_for_novelty=True,
                    minimum_final_profile_count_for_novelty=minimum_profiles,
                )
            )
    for minimum_profiles in (1, 2, 3):
        configs.append(
            TrackletConfig(
                name=f"change_gate_bounded4_p{minimum_profiles}",
                max_tracklets=4,
                require_base_known_for_novelty=True,
                minimum_final_profile_count_for_novelty=minimum_profiles,
            )
        )
    return configs


def _global_bank_candidates() -> list[TrackletConfig]:
    universe: list[TrackletConfig] = []
    for novelty_short, novelty_long in (
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
    ):
        for pending_short, pending_long in (
            (0.30, 0.25),
            (0.40, 0.35),
            (0.50, 0.45),
        ):
            for reuse_short, reuse_long in (
                (0.35, 0.25),
                (0.40, 0.30),
                (0.45, 0.35),
                (0.50, 0.40),
            ):
                for max_tracklets in (6, 12):
                    universe.append(
                        TrackletConfig(
                            name=(
                                f"global_ns{novelty_short:.2f}_nl{novelty_long:.2f}_"
                                f"ps{pending_short:.2f}_pl{pending_long:.2f}_"
                                f"rs{reuse_short:.2f}_rl{reuse_long:.2f}_"
                                f"k{max_tracklets}"
                            ),
                            novelty_short_ceiling=novelty_short,
                            novelty_long_ceiling=novelty_long,
                            pending_short_min=pending_short,
                            pending_long_min=pending_long,
                            reuse_short_min=reuse_short,
                            reuse_long_min=reuse_long,
                            max_tracklets=max_tracklets,
                        )
                    )
    rng = random.Random(20260722)
    selected = rng.sample(universe, 35)
    selected.append(TrackletConfig(name="global_default_reference"))
    return selected


def _ablations(best: TrackletConfig) -> list[TrackletConfig]:
    values = asdict(best)
    values.pop("name", None)
    return [
        TrackletConfig(name="ablation_one_probe", **{**values, "min_confirmation_probes": 1}),
        TrackletConfig(name="ablation_two_probe_no_merge", **{**values, "enable_profile_merge": False}),
        TrackletConfig(name="ablation_two_probe_no_temp", **{**values, "enable_temporary_identity": False}),
        TrackletConfig(name="ablation_short_history_only", **{**values, "require_long_for_confirmation": False}),
        TrackletConfig(name="ablation_full_two_scale_merge", **values),
    ]


def run_analysis(
    campaign_root: Path,
    base_artifact: Path,
    *,
    output: Path,
    top_full_candidates: int = 18,
    quick: bool = False,
    quick_single_slot: bool = False,
    quick_forward_alias: bool = False,
    focused_bounded: bool = False,
    active_sweep: bool = False,
    change_point_sweep: bool = False,
    global_sweep: bool = False,
) -> dict[str, Any]:
    root = campaign_root.resolve()
    parity = json.loads((root / "baseline_parity_report.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(base_artifact.resolve())
    runs = list(parity.get("runs") or [])
    video_ids = sorted({str(item.get("video_id") or "") for item in runs})
    if len(video_ids) != 7:
        raise ValueError(f"Expected seven videos, got {video_ids}")
    prepared_tapes = [_prepare_tape(item, base_config) for item in runs]
    baseline = _evaluate_baseline(prepared_tapes, base_config)

    if quick or quick_single_slot or quick_forward_alias:
        quick_config = (
            TrackletConfig(
                name="single_slot_two_probe_novelty_lease",
                max_tracklets=1,
                single_slot_lease=True,
                lease_dies_on_silence=True,
            )
            if quick_single_slot
            else (
                TrackletConfig(
                    name="hold250_short_history_final_card_survivor",
                    novelty_short_ceiling=0.2,
                    novelty_long_ceiling=0.25,
                    pending_short_min=0.3,
                    pending_long_min=0.25,
                    reuse_short_min=0.4,
                    reuse_long_min=0.3,
                    require_long_for_confirmation=False,
                    profile_merge_survivor="final",
                )
                if quick_forward_alias
                else TrackletConfig(name="first_two_probe_tracklet_merge")
            )
        )
        first = _evaluate_variant(prepared_tapes, quick_config)
        baseline_score = float(baseline["macro_score"])
        first["delta_vs_incumbent"] = float(first["macro_score"]) - baseline_score
        first["per_video_delta_vs_incumbent"] = {
            video_id: float(first["per_video"][video_id])
            - float(baseline["per_video"][video_id])
            for video_id in video_ids
        }
        report = {
            "contract_id": CONTRACT_ID,
            "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "production_promotion_eligible": False,
            "optimization_eligible": False,
            "campaign_root": str(root),
            "base_artifact": str(base_artifact.resolve()),
            "provider": REQUIRED_PROVIDER,
            "windows_seconds": list(REQUIRED_WINDOWS),
            "model_inference_performed": False,
            "video_ids": video_ids,
            "run_count": len(prepared_tapes),
            "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
            "baseline": baseline,
            "best": first,
            "quick_first_comparison": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    if focused_bounded:
        screen_ids = {"JWS-qfR6K3w", "20v1OxUXcQY", "onHUfyRP1BE"}
        screen_tapes = [item for item in prepared_tapes if item.video_id in screen_ids]
        screen: list[dict[str, Any]] = [
            _evaluate_variant(screen_tapes, config)
            for config in _bounded_candidate_grid()
        ]
        screen.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        finalists = [
            _evaluate_variant(prepared_tapes, TrackletConfig(**item["config"]))
            for item in screen[: max(1, int(top_full_candidates))]
        ]
        finalists.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        baseline_score = float(baseline["macro_score"])
        for item in finalists:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        best_config = TrackletConfig(**finalists[0]["config"])
        ablations = [_evaluate_variant(prepared_tapes, item) for item in _ablations(best_config)]
        for item in ablations:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        ablations.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        report = {
            "contract_id": CONTRACT_ID,
            "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "production_promotion_eligible": False,
            "optimization_eligible": False,
            "campaign_root": str(root),
            "base_artifact": str(base_artifact.resolve()),
            "provider": REQUIRED_PROVIDER,
            "windows_seconds": list(REQUIRED_WINDOWS),
            "model_inference_performed": False,
            "video_ids": video_ids,
            "run_count": len(prepared_tapes),
            "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
            "baseline": baseline,
            "bounded_screen": {
                "video_ids": sorted(screen_ids),
                "candidate_count": len(screen),
                "top": screen[:25],
            },
            "full_seven_video_finalists": finalists,
            "ablations": ablations,
            "best": finalists[0],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    if active_sweep:
        variants = [
            _evaluate_variant(prepared_tapes, config)
            for config in _active_continuity_candidates()
        ]
        variants.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        baseline_score = float(baseline["macro_score"])
        for item in variants:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        report = {
            "contract_id": CONTRACT_ID,
            "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "production_promotion_eligible": False,
            "optimization_eligible": False,
            "campaign_root": str(root),
            "base_artifact": str(base_artifact.resolve()),
            "provider": REQUIRED_PROVIDER,
            "windows_seconds": list(REQUIRED_WINDOWS),
            "model_inference_performed": False,
            "video_ids": video_ids,
            "run_count": len(prepared_tapes),
            "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
            "baseline": baseline,
            "variants": variants,
            "best": variants[0],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    if change_point_sweep:
        variants = [
            _evaluate_variant(prepared_tapes, config)
            for config in _change_point_lease_candidates()
        ]
        variants.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        baseline_score = float(baseline["macro_score"])
        for item in variants:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        report = {
            "contract_id": CONTRACT_ID,
            "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "production_promotion_eligible": False,
            "optimization_eligible": False,
            "campaign_root": str(root),
            "base_artifact": str(base_artifact.resolve()),
            "provider": REQUIRED_PROVIDER,
            "windows_seconds": list(REQUIRED_WINDOWS),
            "model_inference_performed": False,
            "video_ids": video_ids,
            "run_count": len(prepared_tapes),
            "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
            "baseline": baseline,
            "variants": variants,
            "best": variants[0],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    if global_sweep:
        variants = [
            _evaluate_variant(prepared_tapes, config)
            for config in _global_bank_candidates()
        ]
        variants.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        baseline_score = float(baseline["macro_score"])
        for item in variants:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        best_config = TrackletConfig(**variants[0]["config"])
        ablations = [_evaluate_variant(prepared_tapes, item) for item in _ablations(best_config)]
        for item in ablations:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
        ablations.sort(key=lambda item: float(item["macro_score"]), reverse=True)
        report = {
            "contract_id": CONTRACT_ID,
            "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "production_promotion_eligible": False,
            "optimization_eligible": False,
            "campaign_root": str(root),
            "base_artifact": str(base_artifact.resolve()),
            "provider": REQUIRED_PROVIDER,
            "windows_seconds": list(REQUIRED_WINDOWS),
            "model_inference_performed": False,
            "video_ids": video_ids,
            "run_count": len(prepared_tapes),
            "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
            "baseline": baseline,
            "variants": variants,
            "ablations": ablations,
            "best": variants[0],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    stress_ids = {"JWS-qfR6K3w", "20v1OxUXcQY"}
    stress_tapes = [item for item in prepared_tapes if item.video_id in stress_ids]
    coarse: list[dict[str, Any]] = []
    for config in _candidate_grid():
        coarse.append(_evaluate_variant(stress_tapes, config))
    coarse.sort(key=lambda item: float(item["macro_score"]), reverse=True)

    finalists: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in coarse:
        config = TrackletConfig(**item["config"])
        key = config.name
        if key in seen:
            continue
        seen.add(key)
        finalists.append(_evaluate_variant(prepared_tapes, config))
        if len(finalists) >= max(1, int(top_full_candidates)):
            break
    finalists.sort(key=lambda item: float(item["macro_score"]), reverse=True)
    best_config = TrackletConfig(**finalists[0]["config"])
    ablations = [_evaluate_variant(prepared_tapes, item) for item in _ablations(best_config)]
    ablations.sort(key=lambda item: float(item["macro_score"]), reverse=True)

    baseline_score = float(baseline["macro_score"])
    for collection in (finalists, ablations):
        for item in collection:
            item["delta_vs_incumbent"] = float(item["macro_score"]) - baseline_score
            item["per_video_delta_vs_incumbent"] = {
                video_id: float(item["per_video"][video_id])
                - float(baseline["per_video"][video_id])
                for video_id in video_ids
            }
    report = {
        "contract_id": CONTRACT_ID,
        "status": "REPLAY_ONLY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "optimization_eligible": False,
        "campaign_root": str(root),
        "base_artifact": str(base_artifact.resolve()),
        "provider": REQUIRED_PROVIDER,
        "windows_seconds": list(REQUIRED_WINDOWS),
        "model_inference_performed": False,
        "video_ids": video_ids,
        "run_count": len(prepared_tapes),
        "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
        "baseline": baseline,
        "coarse_search": {
            "stress_video_ids": sorted(stress_ids),
            "candidate_count": len(coarse),
            "top": coarse[:25],
        },
        "full_seven_video_finalists": finalists,
        "ablations": ablations,
        "best": finalists[0],
        "runtime_plan": {
            "flag": "--live-speaker-open-set-tracklets",
            "default": False,
            "state": "one bounded pending candidate plus bounded temporary tracklet bank",
            "first_probe": "clear to Unknown and retain only causal 0.7/1.5 embeddings",
            "second_probe": "create/reuse temporary identity only after configured two-scale consistency",
            "profile_arrival": "atomically alias final profile and existing browser/transcript identity to temporary card",
            "compute": "no third embedding window or provider; only cosine/state operations",
            "promotion": "implement behind disabled flag, then exact authentic visible-Chrome 1x GUI E2E",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("base_artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-full-candidates", type=int, default=18)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-single-slot", action="store_true")
    parser.add_argument("--quick-forward-alias", action="store_true")
    parser.add_argument("--focused-bounded", action="store_true")
    parser.add_argument("--active-sweep", action="store_true")
    parser.add_argument("--change-point-sweep", action="store_true")
    parser.add_argument("--global-sweep", action="store_true")
    args = parser.parse_args()
    report = run_analysis(
        args.campaign_root,
        args.base_artifact,
        output=args.output,
        top_full_candidates=args.top_full_candidates,
        quick=args.quick,
        quick_single_slot=args.quick_single_slot,
        quick_forward_alias=args.quick_forward_alias,
        focused_bounded=args.focused_bounded,
        active_sweep=args.active_sweep,
        change_point_sweep=args.change_point_sweep,
        global_sweep=args.global_sweep,
    )
    summary = {
        "baseline": round(float(report["baseline"]["macro_score"]), 6),
        "best": round(float(report["best"]["macro_score"]), 6),
        "delta": round(float(report["best"]["delta_vs_incumbent"]), 6),
        "best_name": report["best"]["name"],
        "per_video": {
            key: round(float(value), 6)
            for key, value in report["best"]["per_video"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
