"""Research-only causal identity-history firewall over exact World Tapes.

The prototype starts from the exact exclusive dual-scale tracklet action stream
and adds a bounded, causal identity-consistency layer.  It never consults the
canonical labels while making a decision.  Canonical labels are read only by
the unchanged strict browser scorer after all actions have been produced.

Mechanisms under test are deliberately structural rather than fine parameter
tuning:

* separate short/long histories for every visible public identity;
* confidence-gated prototype updates to prevent cross-speaker drift;
* negative short-scale change evidence while the long scale is still stale;
* uncertainty-aware competition between mature identity histories;
* a bounded two-probe quarantine identity for observations inconsistent with
  every established identity.

This file lives under ``tools`` and cannot emit promotion evidence.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from analyze_live_speaker_open_set_tracklets import (
    TrackletConfig,
    _identity_error_diagnostics,
    _load_base_config,
    _prepare_tape,
    _replay_with_tracklet_actions,
    _tracklet_projection,
    _unit,
)


CONTRACT_ID = "whospeaks.live_world_tape.identity_history_firewall_diagnostic.v1"
SEARCH_VIDEOS = frozenset({"20v1OxUXcQY", "JWS-qfR6K3w", "pD4IdQTmneI"})
VALIDATION_VIDEOS = frozenset(
    {"L-CfFo5aQGU", "S_o3y7CzDUY", "mBeT_AoCXvc", "onHUfyRP1BE"}
)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


@dataclass(frozen=True)
class FirewallConfig:
    name: str
    enabled: bool = True
    mature_count: int = 3
    bank_size: int = 5
    prototype_alpha: float = 0.18
    prototype_min_gap_seconds: float = 0.70
    short_accept_min: float = 0.24
    long_accept_min: float = 0.32
    alternative_margin: float = 0.08
    enable_alternative_reroute: bool = False
    enable_negative_change: bool = False
    negative_short_drop: float = 0.16
    negative_short_max: float = 0.28
    negative_long_stale_min: float = 0.38
    update_short_min: float = 0.34
    update_long_min: float = 0.38
    update_margin: float = 0.02
    require_both_for_update: bool = False
    robust_bank: bool = False
    enable_quarantine: bool = False
    quarantine_confirm_count: int = 2
    quarantine_pending_short_min: float = 0.30
    quarantine_pending_long_min: float = 0.24
    quarantine_pending_max_gap_seconds: float = 1.20
    quarantine_reuse_short_min: float = 0.31
    quarantine_reuse_long_min: float = 0.36
    quarantine_known_advantage_margin: float = 0.04
    max_quarantine_identities: int = 6


@dataclass
class _History:
    label: str
    short_centroid: np.ndarray
    long_centroid: np.ndarray | None
    count: int
    created_time: float
    last_time: float
    bank_size: int
    short_bank: deque[np.ndarray] = field(default_factory=deque)
    long_bank: deque[np.ndarray] = field(default_factory=deque)
    recent_own_short: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    last_prototype_time: float = -1e9

    @classmethod
    def create(
        cls,
        label: str,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
        bank_size: int,
    ) -> "_History":
        item = cls(
            label=label,
            short_centroid=short.copy(),
            long_centroid=None if long is None else long.copy(),
            count=1,
            created_time=media_time,
            last_time=media_time,
            bank_size=bank_size,
        )
        item.short_bank.append(short.copy())
        if long is not None:
            item.long_bank.append(long.copy())
        return item

    def _score_bank(
        self,
        value: np.ndarray | None,
        centroid: np.ndarray | None,
        bank: deque[np.ndarray],
        robust: bool,
    ) -> float:
        if value is None or centroid is None:
            return -1.0
        center_score = _cosine(value, centroid)
        if not robust or not bank:
            return center_score
        bank_scores = sorted((_cosine(value, item) for item in bank), reverse=True)
        count = min(3, len(bank_scores))
        # A consensus score: the centroid remains authoritative, while recent
        # clean exemplars can recover bounded within-speaker variation.  A
        # single old high-similarity exemplar cannot dominate on its own.
        consensus = median(bank_scores[:count])
        return 0.70 * center_score + 0.30 * consensus

    def scores(
        self, short: np.ndarray | None, long: np.ndarray | None, robust: bool
    ) -> tuple[float, float, float]:
        short_score = self._score_bank(
            short, self.short_centroid, self.short_bank, robust
        )
        long_score = self._score_bank(long, self.long_centroid, self.long_bank, robust)
        composite = 0.72 * short_score + 0.28 * (
            long_score if long is not None and self.long_centroid is not None else short_score
        )
        return composite, short_score, long_score

    def update(
        self,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
        config: FirewallConfig,
        own_short_score: float,
    ) -> None:
        alpha = max(0.0, min(1.0, float(config.prototype_alpha)))
        updated = _unit((1.0 - alpha) * self.short_centroid + alpha * short)
        if updated is not None:
            self.short_centroid = updated
        if long is not None:
            if self.long_centroid is None:
                self.long_centroid = long.copy()
            else:
                updated_long = _unit((1.0 - alpha) * self.long_centroid + alpha * long)
                if updated_long is not None:
                    self.long_centroid = updated_long
        self.count += 1
        self.last_time = media_time
        if own_short_score > -0.99:
            self.recent_own_short.append(own_short_score)
        if media_time - self.last_prototype_time >= config.prototype_min_gap_seconds:
            self.short_bank.append(short.copy())
            while len(self.short_bank) > self.bank_size:
                self.short_bank.popleft()
            if long is not None:
                self.long_bank.append(long.copy())
                while len(self.long_bank) > self.bank_size:
                    self.long_bank.popleft()
            self.last_prototype_time = media_time

    def absorb(self, other: "_History", config: FirewallConfig) -> None:
        total = max(1, self.count + other.count)
        weight = other.count / total
        short = _unit((1.0 - weight) * self.short_centroid + weight * other.short_centroid)
        if short is not None:
            self.short_centroid = short
        if other.long_centroid is not None:
            if self.long_centroid is None:
                self.long_centroid = other.long_centroid.copy()
            else:
                long = _unit(
                    (1.0 - weight) * self.long_centroid + weight * other.long_centroid
                )
                if long is not None:
                    self.long_centroid = long
        self.count = total
        self.last_time = max(self.last_time, other.last_time)
        for value in other.short_bank:
            self.short_bank.append(value)
        while len(self.short_bank) > config.bank_size:
            self.short_bank.popleft()
        for value in other.long_bank:
            self.long_bank.append(value)
        while len(self.long_bank) > config.bank_size:
            self.long_bank.popleft()


@dataclass
class _Pending:
    short: np.ndarray
    long: np.ndarray | None
    count: int
    last_time: float


def _preset_configs() -> list[FirewallConfig]:
    return [
        FirewallConfig(name="passthrough", enabled=False),
        FirewallConfig(
            name="dual_history_reject",
            enable_quarantine=False,
        ),
        FirewallConfig(
            name="dual_history_quarantine",
            enable_quarantine=True,
        ),
        FirewallConfig(
            name="uncertainty_competition_quarantine",
            enable_alternative_reroute=True,
            enable_quarantine=True,
        ),
        FirewallConfig(
            name="negative_change_quarantine",
            enable_negative_change=True,
            enable_quarantine=True,
        ),
        FirewallConfig(
            name="competition_change_quarantine",
            enable_alternative_reroute=True,
            enable_negative_change=True,
            enable_quarantine=True,
        ),
        FirewallConfig(
            name="robust_separate_histories",
            enable_alternative_reroute=True,
            enable_negative_change=True,
            enable_quarantine=True,
            robust_bank=True,
            require_both_for_update=True,
        ),
        FirewallConfig(
            name="conservative_robust_firewall",
            mature_count=4,
            short_accept_min=0.20,
            long_accept_min=0.28,
            alternative_margin=0.10,
            enable_alternative_reroute=True,
            enable_negative_change=True,
            negative_short_drop=0.20,
            negative_short_max=0.22,
            negative_long_stale_min=0.42,
            update_short_min=0.40,
            update_long_min=0.44,
            update_margin=0.05,
            require_both_for_update=True,
            robust_bank=True,
            enable_quarantine=True,
            quarantine_reuse_short_min=0.36,
            quarantine_reuse_long_min=0.40,
            quarantine_known_advantage_margin=0.08,
        ),
        FirewallConfig(
            name="short_attack_long_confirm_firewall",
            short_accept_min=0.28,
            long_accept_min=0.26,
            alternative_margin=0.06,
            enable_alternative_reroute=True,
            enable_negative_change=True,
            negative_short_drop=0.14,
            negative_short_max=0.30,
            negative_long_stale_min=0.36,
            update_short_min=0.38,
            update_long_min=0.32,
            update_margin=0.03,
            robust_bank=True,
            enable_quarantine=True,
            quarantine_pending_short_min=0.34,
            quarantine_pending_long_min=0.20,
            quarantine_reuse_short_min=0.34,
            quarantine_reuse_long_min=0.30,
        ),
    ]


def _alias_history(
    histories: dict[str, _History], old_label: str, new_label: str, config: FirewallConfig
) -> None:
    if not old_label or not new_label or old_label == new_label:
        return
    old = histories.pop(old_label, None)
    if old is None:
        return
    current = histories.get(new_label)
    if current is None:
        old.label = new_label
        histories[new_label] = old
    else:
        current.absorb(old, config)


def _clear_from(
    step: Any, active_public: str, reason: str
) -> tuple[float, int, str, dict[str, Any]] | None:
    if not active_public:
        return None
    payload = step.payload
    media_time = float(payload.get("media_time") or 0.0)
    duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
    base = dict(step.recorded_public_payload or {})
    return (
        step.wall_seconds,
        step.sequence,
        "live_speaker_clear",
        {
            **base,
            "step_id": int(payload.get("step_id") or 0),
            "speaker_id": active_public,
            "assigned_speaker": None,
            "live": False,
            "fallback": True,
            "start": base.get("start", round(max(0.0, media_time - duration), 4)),
            "end": base.get("end", round(media_time, 4)),
            "reason": reason,
            "assignment_source": "offline_identity_history_firewall_diagnostic",
        },
    )


def _show_from(
    source: tuple[float, int, str, dict[str, Any]],
    label: str,
    reason: str,
) -> tuple[float, int, str, dict[str, Any]]:
    wall, sequence, _event, raw = source
    payload = dict(raw)
    payload.update(
        {
            "assigned_speaker": label,
            "speaker_id": label,
            "live": True,
            "fallback": True,
            "live_speaker_core_reason": reason,
            "assignment_source": "offline_identity_history_firewall_diagnostic",
        }
    )
    return wall, sequence, "live_speaker", payload


def _firewall_projection(
    prepared: Any,
    base_actions: list[tuple[float, int, str, dict[str, Any]]],
    config: FirewallConfig,
) -> tuple[list[tuple[float, int, str, dict[str, Any]]], dict[str, Any]]:
    if not config.enabled:
        return list(base_actions), {"passthrough": True}

    aliases = sorted(
        [item for item in base_actions if item[2] == "tracklet_alias"],
        key=lambda item: (item[0], item[1]),
    )
    by_step: dict[int, tuple[float, int, str, dict[str, Any]]] = {}
    for item in base_actions:
        if item[2] not in {"live_speaker", "live_speaker_clear"}:
            continue
        step_id = int(dict(item[3]).get("step_id") or 0)
        if step_id:
            by_step[step_id] = item

    output: list[tuple[float, int, str, dict[str, Any]]] = []
    histories: dict[str, _History] = {}
    pending: _Pending | None = None
    active_public = ""
    next_quarantine = 1
    alias_index = 0
    stats: dict[str, Any] = {
        "shown_base": 0,
        "uncertain_rejections": 0,
        "alternative_reroutes": 0,
        "negative_change_points": 0,
        "quarantine_started": 0,
        "quarantine_created": 0,
        "quarantine_reused": 0,
        "gated_updates": 0,
        "rejected_updates": 0,
        "history_count": 0,
    }

    for step in prepared.steps:
        while alias_index < len(aliases) and aliases[alias_index][0] <= step.wall_seconds:
            alias = aliases[alias_index]
            old_label = str(alias[3].get("old_label") or "")
            new_label = str(alias[3].get("new_label") or "")
            _alias_history(histories, old_label, new_label, config)
            if active_public == old_label:
                active_public = new_label
            output.append(alias)
            alias_index += 1

        payload = step.payload
        step_id = int(payload.get("step_id") or 0)
        source = by_step.get(step_id)
        if source is None:
            continue
        event = source[2]
        if event == "live_speaker_clear":
            pending = None
            if active_public:
                clear = _clear_from(step, active_public, str(source[3].get("reason") or "clear"))
                if clear is not None:
                    output.append(clear)
            active_public = ""
            continue

        short = _unit(payload.get("embedding"))
        long = _unit(payload.get("context_embedding"))
        if short is None:
            continue
        media_time = float(payload.get("media_time") or 0.0)
        base_label = str(source[3].get("assigned_speaker") or source[3].get("speaker_id") or "")
        if not base_label:
            continue
        own = histories.get(base_label)
        ranked: list[tuple[float, str, float, float, _History]] = []
        for label, item in histories.items():
            composite, short_score, long_score = item.scores(short, long, config.robust_bank)
            ranked.append((composite, label, short_score, long_score, item))
        ranked.sort(reverse=True, key=lambda item: item[0])
        own_values = own.scores(short, long, config.robust_bank) if own is not None else (-1.0, -1.0, -1.0)
        own_composite, own_short, own_long = own_values
        alternative = next((item for item in ranked if item[1] != base_label), None)
        alternative_wins = bool(
            config.enable_alternative_reroute
            and own is not None
            and own.count >= config.mature_count
            and alternative is not None
            and alternative[4].count >= config.mature_count
            and alternative[0] >= own_composite + config.alternative_margin
            and alternative[2] >= config.short_accept_min
            and (
                long is None
                or alternative[3] >= config.long_accept_min
                or alternative[2] >= config.short_accept_min + 0.12
            )
        )
        negative_change = False
        if (
            config.enable_negative_change
            and own is not None
            and own.count >= config.mature_count
            and len(own.recent_own_short) >= 3
        ):
            expected_short = median(own.recent_own_short)
            negative_change = bool(
                own_short <= expected_short - config.negative_short_drop
                and own_short <= config.negative_short_max
                and (long is None or own_long >= config.negative_long_stale_min)
            )
        dual_failure = bool(
            own is not None
            and own.count >= config.mature_count
            and own_short < config.short_accept_min
            and (long is None or own_long < config.long_accept_min)
        )
        uncertain = dual_failure or negative_change

        chosen = base_label
        reason = "identity_firewall_base"
        if alternative_wins:
            chosen = alternative[1]
            reason = "identity_firewall_alternative"
            stats["alternative_reroutes"] += 1
            pending = None
        elif uncertain:
            if negative_change:
                stats["negative_change_points"] += 1
            quarantine_candidates = [
                item
                for item in ranked
                if item[1].startswith("Q")
                and item[4].count >= config.quarantine_confirm_count
                and item[2] >= config.quarantine_reuse_short_min
                and (long is None or item[3] >= config.quarantine_reuse_long_min)
                and item[0] >= own_composite + config.quarantine_known_advantage_margin
            ]
            if config.enable_quarantine and quarantine_candidates:
                chosen = quarantine_candidates[0][1]
                reason = "identity_firewall_quarantine_reuse"
                stats["quarantine_reused"] += 1
                pending = None
            elif config.enable_quarantine:
                consistent = bool(
                    pending
                    and media_time - pending.last_time
                    <= config.quarantine_pending_max_gap_seconds
                    and _cosine(pending.short, short)
                    >= config.quarantine_pending_short_min
                    and (
                        long is None
                        or pending.long is None
                        or _cosine(pending.long, long)
                        >= config.quarantine_pending_long_min
                    )
                )
                if consistent:
                    pending.count += 1
                    short_center = _unit(0.5 * pending.short + 0.5 * short)
                    if short_center is not None:
                        pending.short = short_center
                    if long is not None:
                        if pending.long is None:
                            pending.long = long.copy()
                        else:
                            long_center = _unit(0.5 * pending.long + 0.5 * long)
                            if long_center is not None:
                                pending.long = long_center
                    pending.last_time = media_time
                else:
                    pending = _Pending(short.copy(), None if long is None else long.copy(), 1, media_time)
                    stats["quarantine_started"] += 1
                quarantine_count = sum(label.startswith("Q") for label in histories)
                if (
                    pending.count >= config.quarantine_confirm_count
                    and quarantine_count < config.max_quarantine_identities
                ):
                    chosen = f"Q{next_quarantine}"
                    next_quarantine += 1
                    histories[chosen] = _History.create(
                        chosen,
                        pending.short,
                        pending.long,
                        media_time,
                        config.bank_size,
                    )
                    reason = "identity_firewall_quarantine_created"
                    stats["quarantine_created"] += 1
                    pending = None
                else:
                    chosen = ""
                    reason = "identity_firewall_uncertain"
                    stats["uncertain_rejections"] += 1
            else:
                chosen = ""
                reason = "identity_firewall_reject"
                stats["uncertain_rejections"] += 1
                pending = None
        else:
            pending = None
            stats["shown_base"] += 1

        selected = histories.get(chosen)
        if chosen and selected is None:
            selected = _History.create(
                chosen, short, long, media_time, config.bank_size
            )
            histories[chosen] = selected
        elif chosen and selected is not None:
            selected_composite, selected_short, selected_long = selected.scores(
                short, long, config.robust_bank
            )
            best_other = max(
                (
                    item.scores(short, long, config.robust_bank)[0]
                    for label, item in histories.items()
                    if label != chosen
                ),
                default=-1.0,
            )
            scale_ok = (
                selected_short >= config.update_short_min
                and selected_long >= config.update_long_min
                if config.require_both_for_update and long is not None
                else selected_short >= config.update_short_min
                or (long is not None and selected_long >= config.update_long_min)
            )
            update_ok = bool(
                not uncertain
                and scale_ok
                and selected_composite >= best_other + config.update_margin
            )
            if update_ok:
                selected.update(short, long, media_time, config, selected_short)
                stats["gated_updates"] += 1
            else:
                if selected_short > -0.99 and not uncertain:
                    selected.recent_own_short.append(selected_short)
                stats["rejected_updates"] += 1

        if chosen:
            output.append(_show_from(source, chosen, reason))
            active_public = chosen
        else:
            clear = _clear_from(step, active_public, reason)
            if clear is not None:
                output.append(clear)
            active_public = ""

    while alias_index < len(aliases):
        output.append(aliases[alias_index])
        alias_index += 1
    output.sort(key=lambda item: (item[0], item[1], 0 if item[2] == "tracklet_alias" else 1))
    stats["history_count"] = len(histories)
    stats["history_labels"] = sorted(histories)
    return output, stats


def _evaluate(
    prepared_tapes: list[Any],
    base_actions: dict[str, list[tuple[float, int, str, dict[str, Any]]]],
    config: FirewallConfig,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for prepared in prepared_tapes:
        actions, stats = _firewall_projection(
            prepared, base_actions[prepared.run_id], config
        )
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
                "firewall_stats": stats,
            }
        )
    per_video_values: dict[str, list[float]] = {}
    for item in runs:
        per_video_values.setdefault(item["video_id"], []).append(item["score"])
    per_video = {
        key: mean(values) for key, values in sorted(per_video_values.items())
    }
    return {
        "name": config.name,
        "config": asdict(config),
        "config_sha256": _stable_hash(asdict(config)),
        "macro_score": mean(per_video.values()),
        "per_video": per_video,
        "runs": runs,
    }


def _subset(result: dict[str, Any], video_ids: set[str] | frozenset[str]) -> float:
    values = [
        float(value)
        for video_id, value in result["per_video"].items()
        if video_id in video_ids
    ]
    return mean(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=Path(
            "runtime/optimization/live_speaker_world_tapes_20260721/baseline_parity_report.json"
        ),
    )
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--exclusive-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.parity_report.read_text(encoding="utf-8-sig"))
    base_config = _load_base_config(args.base_candidate.resolve())
    prepared = [_prepare_tape(run, base_config) for run in parity.get("runs") or []]
    exclusive_data = json.loads(args.exclusive_result.read_text(encoding="utf-8-sig"))
    exclusive_config = TrackletConfig(**exclusive_data["exclusive"]["config"])
    base_actions: dict[str, list[tuple[float, int, str, dict[str, Any]]]] = {}
    for tape in prepared:
        actions, _stats = _tracklet_projection(tape, exclusive_config)
        base_actions[tape.run_id] = actions

    configs = _preset_configs()
    search_tapes = [item for item in prepared if item.video_id in SEARCH_VIDEOS]
    validation_tapes = [item for item in prepared if item.video_id in VALIDATION_VIDEOS]
    search_results = [_evaluate(search_tapes, base_actions, item) for item in configs]
    baseline_search = next(item for item in search_results if item["name"] == "passthrough")
    for item in search_results:
        item["search_score"] = item["macro_score"]
        item["search_delta"] = item["macro_score"] - baseline_search["macro_score"]
        item["hard_floor_delta"] = min(
            float(item["per_video"].get(video_id, -1.0))
            - float(baseline_search["per_video"].get(video_id, -1.0))
            for video_id in ("20v1OxUXcQY", "JWS-qfR6K3w")
        )

    eligible = [
        item
        for item in search_results
        if item["name"] != "passthrough"
        and item["search_delta"] > 0.0
        and item["hard_floor_delta"] >= -0.005
    ]
    eligible.sort(key=lambda item: item["search_score"], reverse=True)
    finalist_names = [item["name"] for item in eligible[:4]]
    # Always carry the three structurally fullest variants to validation when
    # the search split is hostile, so a broad generalizer is not discarded by
    # one difficult-video noise realization.
    for name in (
        "competition_change_quarantine",
        "robust_separate_histories",
        "conservative_robust_firewall",
    ):
        if name not in finalist_names:
            finalist_names.append(name)
    finalist_names = finalist_names[:6]
    by_name = {item.name: item for item in configs}
    validation_results = [
        _evaluate(validation_tapes, base_actions, by_name[name])
        for name in finalist_names
    ]
    baseline_validation = _evaluate(
        validation_tapes, base_actions, by_name["passthrough"]
    )
    search_by_name = {item["name"]: item for item in search_results}
    for item in validation_results:
        item["validation_score"] = item["macro_score"]
        item["validation_delta"] = (
            item["macro_score"] - baseline_validation["macro_score"]
        )
        item["search_delta"] = search_by_name[item["name"]]["search_delta"]
        item["hard_floor_delta"] = search_by_name[item["name"]]["hard_floor_delta"]

    acceptable = [
        item
        for item in validation_results
        if item["search_delta"] > 0.0
        and item["hard_floor_delta"] >= -0.005
        and item["validation_delta"] >= -0.003
    ]
    acceptable.sort(
        key=lambda item: (item["validation_delta"], item["search_delta"]),
        reverse=True,
    )
    selected_name = acceptable[0]["name"] if acceptable else ""
    full_baseline = _evaluate(prepared, base_actions, by_name["passthrough"])
    full_selected = (
        _evaluate(prepared, base_actions, by_name[selected_name])
        if selected_name
        else None
    )
    exact_expected = float(exclusive_data["exclusive"]["macro_score"])
    if abs(full_baseline["macro_score"] - exact_expected) > 1e-9:
        raise RuntimeError(
            f"Exclusive baseline drift: {full_baseline['macro_score']} != {exact_expected}"
        )

    result = {
        "contract_id": CONTRACT_ID,
        "status": "REPLAY_ONLY_CAUSAL_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "optimization_eligible": False,
        "future_or_canonical_labels_used_in_inference": False,
        "model_inference_performed": False,
        "base_candidate_id": str(exclusive_data.get("base_candidate_id") or ""),
        "parent_exclusive_tracklet_config_sha256": exclusive_data["exclusive"].get(
            "config_sha256"
        ),
        "split": {
            "unit": "video",
            "search": sorted(SEARCH_VIDEOS),
            "validation": sorted(VALIDATION_VIDEOS),
            "selection_rule": (
                "positive search delta; 20v1/JWS floor >= -0.005; validation delta >= -0.003; "
                "then maximum validation delta"
            ),
        },
        "baseline": full_baseline,
        "search_results": search_results,
        "validation_baseline": baseline_validation,
        "validation_results": validation_results,
        "selected_name": selected_name,
        "selected_full_result": full_selected,
        "selected_full_delta": (
            None
            if full_selected is None
            else full_selected["macro_score"] - full_baseline["macro_score"]
        ),
        "material_gain_target": 0.01,
        "material_gain_reached": bool(
            full_selected is not None
            and full_selected["macro_score"] - full_baseline["macro_score"] >= 0.01
        ),
        "mechanism_cost": {
            "additional_embeddings": 0,
            "additional_windows": 0,
            "additional_providers": 0,
            "bounded_identity_histories": True,
            "maximum_history_vectors_per_identity_per_scale": max(
                item.bank_size for item in configs
            ),
            "maximum_quarantine_identities": max(
                item.max_quarantine_identities for item in configs
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": full_baseline["macro_score"],
                "selected": selected_name,
                "selected_score": None if full_selected is None else full_selected["macro_score"],
                "selected_delta": result["selected_full_delta"],
                "material": result["material_gain_reached"],
                "search": [
                    {
                        "name": item["name"],
                        "delta": item["search_delta"],
                        "hard_floor": item["hard_floor_delta"],
                    }
                    for item in sorted(search_results, key=lambda item: item["search_score"], reverse=True)
                ],
                "validation": [
                    {"name": item["name"], "delta": item["validation_delta"]}
                    for item in sorted(validation_results, key=lambda item: item["validation_delta"], reverse=True)
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
