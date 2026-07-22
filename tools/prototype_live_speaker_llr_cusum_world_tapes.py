"""Replay-only prototype for profile-normalized open-set gating and CUSUM quarantine.

This file deliberately lives outside ``src/``.  It consumes immutable World Tapes,
does no model inference, and can only emit replay-only research artifacts.  It uses
the same browser-state replay and strict browser scorer as the existing World-Tape
counterfactual evaluator.

The prototype wraps the current Bayesian tracker.  Its output filter has two causal
parts:

* a profile-specific open-set likelihood gate.  Each profile's probe similarity is
  normalized relative to the most confusable profile centroid currently available;
* a one-sided CUSUM (cumulative-sum change detector).  Repeated evidence that the
  current voice no longer fits the displayed profile first puts the UI into an
  explicit Unknown quarantine, and only then permits a stable known assignment.

Canonical labels are used only after replay for scoring and the diagnostic
``premature_known`` metric.  They never enter the decision path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from statistics import mean
from typing import Any, Iterable

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from window.browser_live_speaker_scoring import (  # noqa: E402
    browser_observed_state_slices,
    score_browser_live_speaker_samples,
)
from window.live_speaker_browser_parity import replay_browser_state  # noqa: E402
from window.live_speaker_counterfactual import (  # noqa: E402
    _algorithm,
    _cached_counterfactual_tape_inputs,
    _mapped_values,
    _step,
)
from window.live_speaker_probe_scoring import (  # noqa: E402
    overlap_seconds,
    read_canonical_segments,
)


CONTRACT_ID = "whospeaks.live_world_tape.llr_cusum_replay_prototype.v1"
STATUS = "REPLAY_ONLY_NOMINEE_REQUIRES_REAL_GUI_VALIDATION"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unit(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass(frozen=True)
class GateConfig:
    mode: str = "cusum"
    short_weight: float = 0.80
    long_weight: float = 0.20
    single_profile_anchor: float = 0.02
    cohort_anchor_weight: float = 1.0
    normalized_accept: float = -0.10
    normalized_scale: float = 0.075
    fused_min_similarity: float = 0.05
    short_min_similarity: float = -0.10
    long_min_similarity: float = -0.10
    min_margin: float = -0.20
    require_scale_top_agreement: bool = False
    adaptive_min_samples: int = 4
    adaptive_sigma: float = 2.25
    adaptive_std_floor: float = 0.035
    adaptive_weight: float = 0.20
    learn_margin: float = 0.04
    learn_alpha: float = 0.16
    cusum_decay: float = 0.86
    cusum_drift: float = 0.12
    temporal_similarity_floor: float = 0.35
    temporal_gain: float = 1.0
    incumbent_drop_gain: float = 0.75
    challenger_gain: float = 0.80
    quarantine_threshold: float = 0.90
    switch_threshold: float = 1.40
    min_profiles_for_quarantine: int = 2
    alert_requires_proposed_change: bool = True
    post_switch_quarantine: bool = False
    quarantine_min_probes: int = 1
    exit_count: int = 1
    exit_normalized_margin: float = 0.025
    strong_exit_llr: float = 1.20
    new_profile_grace_seconds: float = 1.2

    def __post_init__(self) -> None:
        if self.mode not in {"passthrough", "gate", "cusum"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.short_weight < 0.0 or self.long_weight < 0.0:
            raise ValueError("Scale weights must be non-negative")
        if self.short_weight + self.long_weight <= 0.0:
            raise ValueError("At least one scale weight must be positive")
        if self.normalized_scale <= 0.0 or self.adaptive_std_floor <= 0.0:
            raise ValueError("Likelihood scales must be positive")
        if (
            self.adaptive_min_samples < 1
            or self.quarantine_min_probes < 0
            or self.min_profiles_for_quarantine < 0
        ):
            raise ValueError("Invalid count")
        if self.exit_count < 1 or not 0.0 <= self.cusum_decay <= 1.0:
            raise ValueError("Invalid CUSUM settings")


@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0

    def update(self, value: float, alpha: float) -> None:
        value = float(value)
        if self.count == 0:
            self.mean = value
            self.variance = 0.0
        else:
            delta = value - self.mean
            self.mean += float(alpha) * delta
            self.variance = (1.0 - float(alpha)) * (
                self.variance + float(alpha) * delta * delta
            )
        self.count += 1


@dataclass(frozen=True)
class _Evidence:
    label: str
    short: float
    long: float
    fused: float
    anchor: float
    normalized: float
    llr: float
    margin: float
    short_top: str
    long_top: str
    gate_ok: bool


class LLRQuarantineFilter:
    """Causal output filter around the production Bayesian tracker."""

    def __init__(self, base_config: dict[str, Any], gate: GateConfig) -> None:
        self.base = _algorithm("bayes", base_config)
        self.gate = gate
        self.visible: str | None = None
        self.last_time = -1.0
        self.unknown_cusum = 0.0
        self.challenger_cusum: dict[str, float] = {}
        self.quarantine = False
        self.quarantine_probes = 0
        self.exit_label: str | None = None
        self.exit_count = 0
        self.profile_first_seen: dict[str, float] = {}
        self.profile_fingerprints: dict[str, str] = {}
        self.moments: dict[str, _Moments] = {}
        self.last_stable_short: np.ndarray | None = None
        self.last_stable_label: str | None = None

    def _profiles(self, payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], set[str]]:
        current: dict[str, np.ndarray] = {}
        new: set[str] = set()
        media_time = float(payload.get("media_time") or 0.0)
        for raw in payload.get("profiles") or []:
            label = str(raw.get("label") or "")
            if not label or raw.get("centroid") is None:
                continue
            centroid = _unit(raw["centroid"])
            current[label] = centroid
            fingerprint = hashlib.sha256(centroid.tobytes()).hexdigest()
            if label not in self.profile_fingerprints:
                new.add(label)
                self.profile_first_seen[label] = media_time
            elif self.profile_fingerprints[label] != fingerprint:
                # A final sentence changed the centroid.  Old live-score moments are
                # no longer calibrated to this exact profile generation.
                self.moments.pop(label, None)
            self.profile_fingerprints[label] = fingerprint
        return current, new

    def _evidence(
        self,
        payload: dict[str, Any],
        profiles: dict[str, np.ndarray],
    ) -> dict[str, _Evidence]:
        if not profiles or payload.get("embedding") is None:
            return {}
        short_vector = _unit(payload["embedding"])
        context = payload.get("context_embedding")
        long_vector = short_vector if context is None else _unit(context)
        short_scores = {
            label: float(np.dot(short_vector, centroid))
            for label, centroid in profiles.items()
        }
        long_scores = {
            label: float(np.dot(long_vector, centroid))
            for label, centroid in profiles.items()
        }
        total_weight = self.gate.short_weight + self.gate.long_weight
        fused = {
            label: (
                self.gate.short_weight * short_scores[label]
                + self.gate.long_weight * long_scores[label]
            ) / total_weight
            for label in profiles
        }
        short_top = max(short_scores, key=lambda label: (short_scores[label], label))
        long_top = max(long_scores, key=lambda label: (long_scores[label], label))
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        result: dict[str, _Evidence] = {}
        for label, centroid in profiles.items():
            cohort = [
                float(np.dot(centroid, other))
                for other_label, other in profiles.items()
                if other_label != label
            ]
            cohort_anchor = max(cohort) if cohort else self.gate.single_profile_anchor
            anchor = (
                self.gate.cohort_anchor_weight * cohort_anchor
                + (1.0 - self.gate.cohort_anchor_weight)
                * self.gate.single_profile_anchor
            )
            normalized = (fused[label] - anchor) / max(0.10, 1.0 - anchor)
            llr = (normalized - self.gate.normalized_accept) / self.gate.normalized_scale
            moments = self.moments.get(label)
            if moments and moments.count >= self.gate.adaptive_min_samples:
                std = max(self.gate.adaptive_std_floor, math.sqrt(moments.variance))
                adaptive_floor = moments.mean - self.gate.adaptive_sigma * std
                llr += self.gate.adaptive_weight * (fused[label] - adaptive_floor) / std
            margin = fused[label] - runner if label == ranked[0][0] else fused[label] - ranked[0][1]
            gate_ok = (
                fused[label] >= self.gate.fused_min_similarity
                and short_scores[label] >= self.gate.short_min_similarity
                and long_scores[label] >= self.gate.long_min_similarity
                and normalized >= self.gate.normalized_accept
                and margin >= self.gate.min_margin
                and (
                    not self.gate.require_scale_top_agreement
                    or (short_top == label and long_top == label)
                )
            )
            result[label] = _Evidence(
                label=label,
                short=short_scores[label],
                long=long_scores[label],
                fused=fused[label],
                anchor=anchor,
                normalized=normalized,
                llr=llr,
                margin=margin,
                short_top=short_top,
                long_top=long_top,
                gate_ok=gate_ok,
            )
        return result

    def _change_cusums(
        self,
        payload: dict[str, Any],
        evidence: dict[str, _Evidence],
    ) -> tuple[float | None, str | None, float]:
        if not evidence:
            self.unknown_cusum = max(
                0.0,
                self.gate.cusum_decay * self.unknown_cusum + 1.0 - self.gate.cusum_drift,
            )
            return None, None, 0.0
        top = max(evidence.values(), key=lambda item: (item.fused, item.label))
        incumbent = evidence.get(self.visible or "")
        short_vector = _unit(payload["embedding"])
        temporal_similarity = (
            float(np.dot(short_vector, self.last_stable_short))
            if self.last_stable_short is not None
            else 1.0
        )
        temporal_change = max(
            0.0, self.gate.temporal_similarity_floor - temporal_similarity
        ) / max(0.05, 1.0 - self.gate.temporal_similarity_floor)
        incumbent_drop = max(0.0, -(incumbent.llr if incumbent else top.llr))
        unknown_observation = (
            max(0.0, -top.llr)
            + self.gate.incumbent_drop_gain * incumbent_drop
            + self.gate.temporal_gain * temporal_change
        )
        self.unknown_cusum = max(
            0.0,
            self.gate.cusum_decay * self.unknown_cusum
            + unknown_observation
            - self.gate.cusum_drift,
        )
        challenger_label: str | None = None
        challenger_value = 0.0
        if self.visible and top.label != self.visible:
            incumbent_llr = incumbent.llr if incumbent else -2.0
            observation = (
                self.gate.challenger_gain * max(0.0, top.llr - incumbent_llr)
                - self.gate.cusum_drift
            )
            previous = self.challenger_cusum.get(top.label, 0.0)
            challenger_value = max(0.0, self.gate.cusum_decay * previous + observation)
            self.challenger_cusum = {top.label: challenger_value}
            challenger_label = top.label
        else:
            self.challenger_cusum = {
                label: max(0.0, self.gate.cusum_decay * value - self.gate.cusum_drift)
                for label, value in self.challenger_cusum.items()
                if value > 0.0
            }
        return temporal_similarity, challenger_label, challenger_value

    def _learn(
        self,
        payload: dict[str, Any],
        evidence: dict[str, _Evidence],
    ) -> None:
        label = self.visible
        if not label or label not in evidence or payload.get("embedding") is None:
            return
        item = evidence[label]
        if not item.gate_ok or item.margin < self.gate.learn_margin:
            return
        self.moments.setdefault(label, _Moments()).update(
            item.fused, self.gate.learn_alpha
        )
        self.last_stable_short = _unit(payload["embedding"]).copy()
        self.last_stable_label = label

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        media_time = float(payload.get("media_time") or 0.0)
        if media_time + 1e-9 < self.last_time:
            raise ValueError("Prototype steps must be chronological")
        self.last_time = media_time
        profiles, new_profiles = self._profiles(payload)
        self.base.sync_profiles(list(payload.get("profiles") or []))
        base_decision = _step(self.base, "bayes", payload)
        base_trace = base_decision.trace_record()
        proposed = str(base_trace.get("visible_speaker") or "") or None
        previous = self.visible

        if self.gate.mode == "passthrough":
            self.visible = proposed
            return {**base_trace, "visible_speaker": self.visible}

        dedicated_probe = bool(str(payload.get("probe_id") or ""))
        eligible_probe = (
            dedicated_probe
            and bool(payload.get("probe_scheduled"))
            and bool(payload.get("speech"))
            and not bool(payload.get("release_signal"))
            and payload.get("embedding") is not None
        )
        evidence = self._evidence(payload, profiles) if eligible_probe else {}
        proposed_evidence = evidence.get(proposed or "")

        if not eligible_probe:
            # The wrapped tracker remains authoritative for release/non-probe state.
            if proposed is None and previous is not None:
                self.visible = None
                self.quarantine = False
                self.exit_label = None
                self.exit_count = 0
            elif proposed is not None and previous is None and not self.quarantine:
                self.visible = proposed
            result = {**base_trace, "visible_speaker": self.visible}
            result["action"] = (
                "clear" if previous and not self.visible else
                "acquire" if not previous and self.visible else
                "switch" if previous and self.visible and previous != self.visible else
                "hold" if self.visible else "none"
            )
            result["reason"] = "llr_cusum_non_probe"
            return result

        temporal_similarity, challenger_label, challenger_value = self._change_cusums(
            payload, evidence
        )
        top = max(evidence.values(), key=lambda item: (item.fused, item.label)) if evidence else None
        normalized_ok = bool(proposed_evidence and proposed_evidence.gate_ok)

        if self.gate.mode == "gate":
            self.visible = proposed if normalized_ok else None
        else:
            # The incumbent already has an explicit one-probe Unknown release.
            # Never turn that clear into a stale hold; this wrapper only adds
            # quarantine around otherwise-known proposals.
            if proposed is None:
                self.visible = None
                self.quarantine = False
                self.quarantine_probes = 0
                self.exit_label = None
                self.exit_count = 0
            alert = max(self.unknown_cusum, challenger_value)
            proposed_change = bool(
                self.visible is not None
                and proposed is not None
                and proposed != self.visible
            )
            profile_count_allows = (
                len(profiles) >= self.gate.min_profiles_for_quarantine
            )
            alert_context_allows = (
                not self.gate.alert_requires_proposed_change or proposed_change
            )
            post_switch_alert = bool(
                self.gate.post_switch_quarantine
                and proposed_change
                and normalized_ok
            )
            # A single failed open-set gate is noisy around mixed-speaker window
            # boundaries.  It contributes to the continuous CUSUM, but only the
            # accumulated alert may blank an already-visible speaker.
            if (
                proposed is not None
                and self.visible is not None
                and profile_count_allows
                and (
                    post_switch_alert
                    or (
                        alert_context_allows
                        and alert >= self.gate.quarantine_threshold
                    )
                )
            ):
                self.visible = None
                self.quarantine = True
                self.quarantine_probes = 0
                self.exit_label = None
                self.exit_count = 0

            if self.quarantine:
                self.quarantine_probes += 1
                exit_item = evidence.get(proposed or "")
                exit_ok = bool(
                    exit_item
                    and exit_item.gate_ok
                    and exit_item.normalized
                    >= self.gate.normalized_accept + self.gate.exit_normalized_margin
                )
                if exit_ok and proposed == self.exit_label:
                    self.exit_count += 1
                elif exit_ok:
                    self.exit_label = proposed
                    self.exit_count = 1
                else:
                    self.exit_label = None
                    self.exit_count = 0
                new_profile = bool(proposed and proposed in new_profiles)
                newly_published = bool(
                    proposed
                    and media_time - self.profile_first_seen.get(proposed, -1e9)
                    <= self.gate.new_profile_grace_seconds
                )
                strong = bool(exit_item and exit_item.llr >= self.gate.strong_exit_llr)
                can_exit = (
                    self.quarantine_probes >= self.gate.quarantine_min_probes
                    and (
                        self.exit_count >= self.gate.exit_count
                        or (new_profile and newly_published and exit_ok)
                        or (strong and self.exit_count >= 1)
                    )
                )
                if can_exit:
                    self.visible = proposed
                    self.quarantine = False
                    self.unknown_cusum = 0.0
                    self.challenger_cusum.clear()
                    self.exit_label = None
                    self.exit_count = 0
            elif self.visible is None and normalized_ok:
                self.visible = proposed
            elif (
                self.visible is not None
                and proposed is not None
                and proposed != self.visible
                and normalized_ok
                and challenger_label == proposed
                and challenger_value >= self.gate.switch_threshold
            ):
                self.visible = proposed
                self.unknown_cusum = 0.0
                self.challenger_cusum.clear()
            elif self.visible is not None and proposed == self.visible and normalized_ok:
                pass

        self._learn(payload, evidence)
        action = (
            "clear" if previous and not self.visible else
            "acquire" if not previous and self.visible else
            "switch" if previous and self.visible and previous != self.visible else
            "hold" if self.visible else "none"
        )
        return {
            **base_trace,
            "visible_speaker": self.visible,
            "action": action,
            "reason": "profile_normalized_llr_cusum",
            "diagnostics": {
                **dict(base_trace.get("diagnostics") or {}),
                "prototype_contract_id": CONTRACT_ID,
                "unknown_cusum": self.unknown_cusum,
                "challenger_label": challenger_label,
                "challenger_cusum": challenger_value,
                "quarantine": self.quarantine,
                "quarantine_probes": self.quarantine_probes,
                "temporal_similarity": temporal_similarity,
                "top_evidence": asdict(top) if top else None,
            },
        }


def _project(
    tape_dir: Path,
    base_config: dict[str, Any],
    gate_config: GateConfig,
) -> dict[str, Any]:
    root = Path(tape_dir).resolve()
    input_records, recorded_decisions, public_by_step, hold_seconds = (
        _cached_counterfactual_tape_inputs(str(root))
    )
    hold_seconds = max(
        0.0, float(base_config.get("live_speaker_probe_hold_seconds", hold_seconds))
    )
    tracker = LLRQuarantineFilter(base_config, gate_config)
    active_public_speaker = ""
    aliases: dict[str, str] = {}
    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []
    for record in input_records:
        payload = dict(record["payload"])
        step_id = int(payload.get("step_id") or 0)
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

        trace = tracker.step(payload)
        internal = str(trace.get("visible_speaker") or "")
        public = aliases.get(internal, internal)
        probabilities = _mapped_values(
            dict(trace.get("probabilities") or {}), aliases, probability_keys=True
        )
        raw_probabilities = _mapped_values(
            dict(trace.get("raw_probabilities") or {}), aliases, probability_keys=True
        )
        similarities = _mapped_values(dict(trace.get("similarities") or {}), aliases)
        media_time = float(payload.get("media_time") or 0.0)
        duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
        dedicated_probe = bool(str(payload.get("probe_id") or ""))
        release_signal = bool(payload.get("release_signal"))
        start = round(max(0.0, media_time - duration), 4)
        end = round(media_time, 4)
        event_record = recorded_public or recorded_decisions.get(step_id) or record
        event_wall = float(event_record.get("wall_us") or 0) / 1_000_000.0
        event_sequence = int(event_record.get("seq") or record.get("seq") or 0)
        base = recorded_public_payload
        if dedicated_probe and public and not release_signal:
            action_payload = {
                **base,
                "step_id": step_id,
                "assigned_speaker": public,
                "speaker_id": public,
                "internal_speaker_id": internal,
                "replaces_speaker_id": internal if internal != public else None,
                "probabilities": probabilities,
                "raw_probabilities": raw_probabilities,
                "similarities": similarities,
                "unknown_probability": float(probabilities.get("unknown", 1.0)),
                "live_speaker_core_action": trace.get("action"),
                "live_speaker_core_reason": trace.get("reason"),
                "live": True,
                "fallback": True,
                "start": base.get("start", start),
                "end": base.get("end", end),
                "audio_length_seconds": base.get("audio_length_seconds", round(duration, 4)),
                "hold_seconds": round(hold_seconds, 4),
                "assignment_source": "replay_only_profile_normalized_llr_cusum",
            }
            actions.append((event_wall, event_sequence, "live_speaker", action_payload))
            active_public_speaker = public
        elif dedicated_probe and str(trace.get("action") or "") == "clear" and active_public_speaker:
            clear_payload = {
                **base,
                "step_id": step_id,
                "speaker_id": active_public_speaker,
                "assigned_speaker": None,
                "live": False,
                "fallback": True,
                "start": base.get("start", start),
                "end": base.get("end", end),
                "reason": "silence" if release_signal else str(trace.get("reason") or "unknown"),
                "assignment_source": "replay_only_profile_normalized_llr_cusum",
            }
            actions.append((event_wall, event_sequence, "live_speaker_clear", clear_payload))
            active_public_speaker = ""
        decisions.append(
            {
                "step_id": step_id,
                "media_time": media_time,
                "visible_speaker": public,
                "internal_speaker": internal,
                "action": trace.get("action"),
                "reason": trace.get("reason"),
                "diagnostics": trace.get("diagnostics"),
            }
        )
    first_seen = {
        aliases.get(label, label): value
        for label, value in tracker.profile_first_seen.items()
    }
    return {
        "actions": actions,
        "decisions": decisions,
        "profile_first_seen": first_seen,
        "input_step_count": len(input_records),
    }


def _premature_known(
    samples: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    score: dict[str, Any],
    profile_first_seen: dict[str, float],
) -> dict[str, Any]:
    profile_map = dict(score.get("speaker_map") or {})
    canonical_first_available: dict[str, float] = {}
    for public_label, canonical_label in profile_map.items():
        if public_label not in profile_first_seen:
            continue
        value = float(profile_first_seen[public_label])
        canonical_first_available[canonical_label] = min(
            value, canonical_first_available.get(canonical_label, math.inf)
        )
    state_slices = browser_observed_state_slices(samples)
    total = 0.0
    wrong_known = 0.0
    unknown = 0.0
    for segment in canonical:
        speaker = str(segment.get("speaker") or segment.get("speaker_id") or "")
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        cutoff = min(end, canonical_first_available.get(speaker, math.inf))
        if cutoff <= start:
            continue
        total += cutoff - start
        for item in state_slices:
            overlap = overlap_seconds(
                [(start, cutoff)],
                [(float(item["start"]), float(item["end"]))],
            )
            if overlap <= 0.0:
                continue
            visible = str(item.get("speaker") or "")
            if not visible:
                unknown += overlap
            elif profile_map.get(visible) != speaker:
                wrong_known += overlap
    return {
        "pre_profile_speech_seconds": round(total, 6),
        "premature_known_wrong_seconds": round(wrong_known, 6),
        "pre_profile_unknown_seconds": round(unknown, 6),
        "premature_known_wrong_ratio": round(wrong_known / total, 6) if total else 0.0,
    }


def evaluate_run(
    tape_dir: Path,
    canonical_path: Path,
    base_config: dict[str, Any],
    gate_config: GateConfig,
) -> dict[str, Any]:
    projection = _project(tape_dir, base_config, gate_config)
    browser = replay_browser_state(tape_dir, replacement_live_actions=projection["actions"])
    canonical = read_canonical_segments(canonical_path)
    score = score_browser_live_speaker_samples(browser["replayed_samples"], canonical)
    return {
        "score": score,
        "strict_browser_live_score": score["strict_browser_live_score"],
        "projected_live_action_count": len(projection["actions"]),
        "premature_known": _premature_known(
            browser["replayed_samples"], canonical, score, projection["profile_first_seen"]
        ),
    }


def _runs(campaign_root: Path, video_ids: set[str] | None) -> list[dict[str, Any]]:
    report = json.loads(
        (campaign_root / "baseline_parity_report.json").read_text(encoding="utf-8")
    )
    result = [
        item for item in report.get("runs") or []
        if not video_ids or str(item.get("video_id") or "") in video_ids
    ]
    if not result:
        raise ValueError("No World Tape runs selected")
    return result


def evaluate_candidate(
    name: str,
    base_config: dict[str, Any],
    gate: GateConfig,
    runs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        result = evaluate_run(
            Path(run["tape_dir"]),
            Path(run["canonical_path"]),
            base_config,
            gate,
        )
        score = result["score"]
        rows.append(
            {
                "video_id": str(run["video_id"]),
                "run_id": str(run["run_id"]),
                "score": float(result["strict_browser_live_score"]),
                "correct_coverage": float(score["correct_live_speaker_coverage"]),
                "wrong_ratio": float(score["wrong_live_speech_ratio"]),
                "missing_ratio": float(score["missing_live_speech_ratio"]),
                "outside_ratio": float(score["outside_speech_live_ratio"]),
                "precision": float(score["correct_live_precision_during_speech"]),
                "premature_known": result["premature_known"],
            }
        )
    per_video: dict[str, list[float]] = {}
    for row in rows:
        per_video.setdefault(row["video_id"], []).append(row["score"])
    video_scores = {
        video: mean(values) for video, values in sorted(per_video.items())
    }
    premature_values = [
        float(row["premature_known"]["premature_known_wrong_seconds"])
        for row in rows if row["premature_known"] is not None
    ]
    return {
        "name": name,
        "gate_config": asdict(gate),
        "gate_config_sha256": _stable_hash(asdict(gate)),
        "macro_score": mean(video_scores.values()),
        "per_video": video_scores,
        "premature_known_wrong_seconds_mean_run": (
            mean(premature_values) if premature_values else None
        ),
        "runs": rows,
    }


def _candidate_configs(seed: int, count: int) -> list[tuple[str, GateConfig]]:
    candidates: list[tuple[str, GateConfig]] = [
        ("normalized_gate_default", GateConfig(mode="gate")),
        ("llr_cusum_default", GateConfig(mode="cusum")),
        (
            "cusum_no_adaptive",
            replace(GateConfig(mode="cusum"), adaptive_weight=0.0),
        ),
        (
            "cusum_no_temporal",
            replace(GateConfig(mode="cusum"), temporal_gain=0.0),
        ),
        (
            "cusum_no_cohort",
            replace(GateConfig(mode="cusum"), cohort_anchor_weight=0.0),
        ),
    ]
    rng = random.Random(seed)
    for index in range(max(0, count)):
        short_weight = rng.choice([0.65, 0.72, 0.80, 0.88])
        config = GateConfig(
            mode="cusum",
            short_weight=short_weight,
            long_weight=1.0 - short_weight,
            single_profile_anchor=rng.choice([-0.02, 0.02, 0.06, 0.10]),
            cohort_anchor_weight=rng.choice([0.4, 0.7, 1.0]),
            normalized_accept=rng.choice([-0.18, -0.14, -0.10, -0.06, -0.02, 0.02]),
            normalized_scale=rng.choice([0.05, 0.075, 0.10]),
            fused_min_similarity=rng.choice([-0.02, 0.03, 0.08, 0.13, 0.18]),
            short_min_similarity=rng.choice([-0.15, -0.08, 0.0, 0.08]),
            long_min_similarity=rng.choice([-0.15, -0.08, 0.0, 0.08]),
            min_margin=rng.choice([-0.40, -0.25, -0.10, 0.0, 0.02]),
            require_scale_top_agreement=rng.choice([False, False, False, True]),
            adaptive_sigma=rng.choice([1.75, 2.25, 2.75]),
            adaptive_weight=rng.choice([0.0, 0.15, 0.30]),
            cusum_decay=rng.choice([0.70, 0.82, 0.90, 1.0]),
            cusum_drift=rng.choice([0.05, 0.10, 0.20, 0.35]),
            temporal_similarity_floor=rng.choice([0.15, 0.25, 0.35, 0.45, 0.55]),
            temporal_gain=rng.choice([0.0, 0.5, 1.0, 1.5]),
            incumbent_drop_gain=rng.choice([0.25, 0.75, 1.25]),
            challenger_gain=rng.choice([0.4, 0.8, 1.2]),
            quarantine_threshold=rng.choice([0.5, 0.9, 1.4, 2.0, 3.0]),
            switch_threshold=rng.choice([0.8, 1.4, 2.2, 3.5]),
            quarantine_min_probes=rng.choice([1, 2, 3]),
            exit_count=rng.choice([1, 2, 3]),
            exit_normalized_margin=rng.choice([0.0, 0.025, 0.05, 0.075]),
            strong_exit_llr=rng.choice([0.8, 1.2, 1.8, 2.5]),
        )
        candidates.append((f"random_{index + 1:04d}", config))
    return candidates


def _focused_configs() -> list[tuple[str, GateConfig]]:
    """Sixteen deliberately bounded late-switch quarantine experiments."""

    lenient = GateConfig(
        mode="cusum",
        normalized_accept=-0.18,
        fused_min_similarity=-0.02,
        short_min_similarity=-0.15,
        long_min_similarity=-0.15,
        min_margin=-0.40,
        adaptive_weight=0.0,
        temporal_gain=0.0,
        incumbent_drop_gain=0.0,
        strong_exit_llr=999.0,
        quarantine_min_probes=2,
    )
    result: list[tuple[str, GateConfig]] = []
    for minimum_profiles in (2, 3, 4, 5):
        for exit_count in (1, 2):
            result.append(
                (
                    f"post_switch_p{minimum_profiles}_exit{exit_count}",
                    replace(
                        lenient,
                        post_switch_quarantine=True,
                        min_profiles_for_quarantine=minimum_profiles,
                        exit_count=exit_count,
                    ),
                )
            )
    for minimum_profiles in (2, 4):
        for threshold in (0.35, 0.70, 1.20, 2.00):
            result.append(
                (
                    f"selective_cusum_p{minimum_profiles}_h{str(threshold).replace('.', '')}",
                    replace(
                        lenient,
                        normalized_accept=-0.10,
                        post_switch_quarantine=False,
                        min_profiles_for_quarantine=minimum_profiles,
                        quarantine_threshold=threshold,
                        challenger_gain=0.8,
                        cusum_drift=0.10,
                        exit_count=1,
                    ),
                )
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--videos", default="")
    parser.add_argument(
        "--names",
        default="",
        help="Optional comma-separated candidate names after family construction.",
    )
    parser.add_argument("--random-count", type=int, default=0)
    parser.add_argument(
        "--focused",
        action="store_true",
        help="Run only the bounded 16-member selective/post-switch family.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    artifact = json.loads(args.base_artifact.read_text(encoding="utf-8"))
    base_config = dict(artifact.get("algorithm_config") or {})
    videos = {item.strip() for item in args.videos.split(",") if item.strip()}
    runs = _runs(campaign, videos or None)

    incumbent = evaluate_candidate(
        "incumbent",
        base_config,
        GateConfig(mode="passthrough"),
        runs,
    )
    results: list[dict[str, Any]] = [incumbent]
    candidate_configs = (
        _focused_configs()
        if args.focused
        else _candidate_configs(args.seed, args.random_count)
    )
    requested_names = {
        item.strip() for item in args.names.split(",") if item.strip()
    }
    if requested_names:
        candidate_configs = [
            item for item in candidate_configs if item[0] in requested_names
        ]
        missing = requested_names - {name for name, _ in candidate_configs}
        if missing:
            raise ValueError(f"Unknown candidate names: {sorted(missing)}")
    for index, (name, gate) in enumerate(candidate_configs, 1):
        result = evaluate_candidate(name, base_config, gate, runs)
        result["delta_vs_incumbent"] = result["macro_score"] - incumbent["macro_score"]
        results.append(result)
        print(
            f"[{index:04d}] {name}: {result['macro_score']:.6f} "
            f"({result['delta_vs_incumbent']:+.6f})",
            flush=True,
        )
    incumbent["delta_vs_incumbent"] = 0.0
    results.sort(key=lambda item: item["macro_score"], reverse=True)
    report = {
        "contract_id": CONTRACT_ID,
        "status": STATUS,
        "discovery_only": True,
        "production_promotion_eligible": False,
        "warning": (
            "The World-Tape parity report is optimization_eligible=false.  These "
            "results are replay nominees only and cannot replace a champion without "
            "authentic visible-Chrome wall-clock 1x GUI validation."
        ),
        "campaign_root": str(campaign),
        "base_artifact": str(args.base_artifact.resolve()),
        "base_algorithm_config_sha256": _stable_hash(base_config),
        "video_ids": sorted({str(item["video_id"]) for item in runs}),
        "run_count": len(runs),
        "selection_score": "plain macro mean of per-video mean strict_browser_live_score",
        "candidate_count": len(results),
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nTop candidates:")
    for item in results[: max(1, args.top)]:
        print(
            json.dumps(
                {
                    "name": item["name"],
                    "score": round(item["macro_score"], 6),
                    "delta": round(item["delta_vs_incumbent"], 6),
                    "premature_known_wrong_seconds_mean_run": item[
                        "premature_known_wrong_seconds_mean_run"
                    ],
                    "per_video": {
                        key: round(value, 6)
                        for key, value in item["per_video"].items()
                    },
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
