"""Causal two-window residual corrections around the legacy live decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Iterator

import numpy as np

from window.live_speaker_algorithm import LiveSpeakerDecision, SpeakerProfileEvent
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep


HYBRID_ALGORITHM_ID = "causal_legacy_residual_hybrid_v1"
PROFILE_QUALITY_META_LEASE_ID = "causal_profile_quality_meta_lease_a005_s035_v1"

PROFILE_QUALITY_META_CONFIG_FIELDS = (
    "enable_profile_quality_meta_lease",
    "profile_quality_meta_fresh_min_age_seconds",
    "profile_quality_meta_fresh_max_age_seconds",
    "profile_quality_meta_fresh_min_speech_seconds",
    "profile_quality_meta_fresh_min_short_margin",
    "profile_quality_meta_fresh_min_long_margin",
    "profile_quality_meta_independent_max_profile_count",
    "profile_quality_meta_switch_min_short_margin",
)


@dataclass(frozen=True)
class HybridSpeakerTrackerConfig:
    enable_young_profile_confirmation: bool = False
    enable_young_profile_lease: bool = False
    enable_short_scale_fast_lease: bool = False
    enable_profile_quality_short_scale_fast_lease: bool = False
    profile_quality_fast_lease_min_sentence_count: int = 2
    profile_quality_fast_lease_min_speech_seconds: float = 3.1
    profile_quality_fast_lease_min_similarity: float = 0.18
    profile_quality_fast_lease_min_margin: float = 0.06
    enable_profile_quality_meta_lease: bool = False
    profile_quality_meta_fresh_min_age_seconds: float = 0.05
    profile_quality_meta_fresh_max_age_seconds: float = 0.8
    profile_quality_meta_fresh_min_speech_seconds: float = 3.8
    profile_quality_meta_fresh_min_short_margin: float = 0.30
    profile_quality_meta_fresh_min_long_margin: float = 0.70
    profile_quality_meta_independent_max_profile_count: int = 8
    profile_quality_meta_switch_min_short_margin: float = 0.35
    young_trusted_min_sentence_count: int = 4
    young_trusted_min_speech_seconds: float = 8.0
    young_min_similarity: float = 0.45
    young_min_margin: float = 0.05
    young_required_consecutive_probes: int = 2
    young_independent_scale_count: int = 1
    young_fast_independent_scale_count: int = 1
    self_echo_guard_seconds: float = 0.0
    enable_boundary_abstention: bool = False
    boundary_min_similarity: float = 0.35
    boundary_min_margin: float = 0.04
    boundary_short_advantage: float = 0.05
    boundary_long_advantage: float = 0.05
    boundary_required_consecutive_probes: int = 2
    history_max_gap_seconds: float = 1.5

    def __post_init__(self) -> None:
        positive = (
            "young_trusted_min_sentence_count",
            "young_required_consecutive_probes",
            "young_independent_scale_count",
            "young_fast_independent_scale_count",
            "boundary_required_consecutive_probes",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.young_independent_scale_count) > 2:
            raise ValueError("young_independent_scale_count may not exceed two")
        if int(self.young_fast_independent_scale_count) > int(self.young_independent_scale_count):
            raise ValueError("fast confirmation may not require more scales than permanent confirmation")
        if float(self.young_trusted_min_speech_seconds) < 0.0:
            raise ValueError("young_trusted_min_speech_seconds must be non-negative")
        if int(self.profile_quality_fast_lease_min_sentence_count) < 0:
            raise ValueError("profile_quality_fast_lease_min_sentence_count must be non-negative")
        if float(self.profile_quality_fast_lease_min_speech_seconds) < 0.0:
            raise ValueError("profile_quality_fast_lease_min_speech_seconds must be non-negative")
        meta_non_negative = (
            "profile_quality_meta_fresh_min_age_seconds",
            "profile_quality_meta_fresh_max_age_seconds",
            "profile_quality_meta_fresh_min_speech_seconds",
            "profile_quality_meta_fresh_min_short_margin",
            "profile_quality_meta_fresh_min_long_margin",
            "profile_quality_meta_switch_min_short_margin",
        )
        for name in meta_non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            float(self.profile_quality_meta_fresh_max_age_seconds)
            < float(self.profile_quality_meta_fresh_min_age_seconds)
        ):
            raise ValueError("profile-quality meta maximum age may not precede minimum age")
        if int(self.profile_quality_meta_independent_max_profile_count) < 1:
            raise ValueError("profile_quality_meta_independent_max_profile_count must be positive")
        if self.enable_profile_quality_meta_lease and not self.enable_young_profile_confirmation:
            raise ValueError("profile-quality meta lease requires the run018 young-profile expert")
        if (
            self.enable_profile_quality_meta_lease
            and self.enable_profile_quality_short_scale_fast_lease
        ):
            raise ValueError(
                "profile-quality meta lease requires the run018 precision expert; "
                "the older profile-quality output lease must remain disabled"
            )
        if float(self.self_echo_guard_seconds) < 0.0:
            raise ValueError("self_echo_guard_seconds must be non-negative")
        if float(self.history_max_gap_seconds) <= 0.0:
            raise ValueError("history_max_gap_seconds must be positive")


@dataclass
class _OfficialProfile:
    centroid: np.ndarray
    generation: int
    sentence_count: int
    speech_seconds: float
    latest_sentence_end: float | None
    available_at: float


def hybrid_config_identity_payload(config: HybridSpeakerTrackerConfig) -> dict[str, Any]:
    """Return an ID payload that preserves hashes of disabled optional rules."""

    payload = asdict(config)
    if not bool(payload.get("enable_short_scale_fast_lease")):
        payload.pop("enable_short_scale_fast_lease", None)
    if not bool(payload.get("enable_profile_quality_short_scale_fast_lease")):
        payload.pop("enable_profile_quality_short_scale_fast_lease", None)
        # The meta rule consumes this raw qvote even though the older rule is
        # disabled, so its four thresholds become identity-bearing under Meta.
        if not bool(payload.get("enable_profile_quality_meta_lease")):
            for name in (
                "profile_quality_fast_lease_min_sentence_count",
                "profile_quality_fast_lease_min_speech_seconds",
                "profile_quality_fast_lease_min_similarity",
                "profile_quality_fast_lease_min_margin",
            ):
                payload.pop(name, None)
    if not bool(payload.get("enable_profile_quality_meta_lease")):
        for name in PROFILE_QUALITY_META_CONFIG_FIELDS:
            payload.pop(name, None)
    return payload


class CausalHybridSpeakerTracker:
    """Use two existing scale embeddings only as residual sensors."""

    def __init__(
        self,
        config: HybridSpeakerTrackerConfig | None = None,
        profile_events: Iterable[SpeakerProfileEvent] = (),
    ) -> None:
        self.config = config or HybridSpeakerTrackerConfig()
        self._events = sorted(
            list(profile_events),
            key=lambda e: (float(e.available_at), int(e.generation), str(e.speaker_id)),
        )
        self._next_event = 0
        self._profiles: dict[str, _OfficialProfile] = {}
        self._first_official: str | None = None
        self._permanently_confirmed: set[str] = set()
        self._young_candidate: str | None = None
        self._young_streak = 0
        self._leased_profile: str | None = None
        self._profile_quality_meta_leased_profile: str | None = None
        self._profile_quality_meta_emitted_visible: str | None = None
        self._boundary_incumbent: str | None = None
        self._boundary_challenger: str | None = None
        self._boundary_streak = 0
        self._boundary_active = False
        self._last_full_probe_time: float | None = None
        self._last_media_time = -1.0
        self._last_baseline_visible: str | None = None
        self._visible: str | None = None
        self._abstaining = False

    @property
    def visible_speaker(self) -> str | None:
        if self.config.enable_profile_quality_meta_lease:
            return self._profile_quality_meta_emitted_visible
        return self._visible

    def _reset_young(self, drop_lease: bool = True) -> None:
        self._young_candidate = None
        self._young_streak = 0
        if drop_lease:
            self._leased_profile = None

    def _reset_profile_quality_meta_lease(self) -> None:
        self._profile_quality_meta_leased_profile = None

    def _reset_boundary(self) -> None:
        self._boundary_incumbent = None
        self._boundary_challenger = None
        self._boundary_streak = 0
        self._boundary_active = False

    def _reset_temporal(self) -> None:
        self._reset_young()
        self._reset_profile_quality_meta_lease()
        self._reset_boundary()
        self._last_full_probe_time = None

    def _apply_profile_events(self, media_time: float) -> list[str]:
        applied: list[str] = []
        while self._next_event < len(self._events):
            event = self._events[self._next_event]
            if float(event.available_at) > media_time + 1e-9:
                break
            label = str(event.speaker_id)
            old = self._profiles.get(label)
            if old is None or int(event.generation) > old.generation:
                if self._first_official is None:
                    self._first_official = label
                self._profiles[label] = _OfficialProfile(
                    centroid=np.asarray(event.centroid, dtype=np.float64),
                    generation=int(event.generation),
                    sentence_count=int(event.sentence_count),
                    speech_seconds=float(event.speech_seconds),
                    latest_sentence_end=None if event.sentence_end is None else float(event.sentence_end),
                    available_at=float(event.available_at),
                )
                applied.append(f"{label}:{int(event.generation)}")
            self._next_event += 1
        if applied:
            self._reset_temporal()
        return applied

    def _is_mature(self, label: str) -> bool:
        if label == self._first_official or label in self._permanently_confirmed:
            return True
        profile = self._profiles.get(label)
        if profile is None:
            return True
        return (
            profile.sentence_count >= int(self.config.young_trusted_min_sentence_count)
            and profile.speech_seconds >= float(self.config.young_trusted_min_speech_seconds)
        )

    def _validate(self, baseline: LiveSpeakerDecision, item: MultiScaleStep) -> None:
        if abs(float(baseline.media_time) - float(item.media_time)) > 1e-6:
            raise ValueError("baseline and multi-scale media times must match")
        if float(item.media_time) + 1e-9 < self._last_media_time:
            raise ValueError("hybrid steps must be chronological")
        if len(item.evidences) > 2:
            raise ValueError("hybrid tracker accepts at most two already-computed windows")
        if len(item.evidences) == 2 and len({round(float(e.window_seconds), 6) for e in item.evidences}) != 2:
            raise ValueError("hybrid tracker requires distinct window durations")
        if not item.probe_scheduled and item.evidences:
            raise ValueError("non-probe hybrid ticks may not carry evidence")

    def _rankings(self, item: MultiScaleStep) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for evidence in sorted(item.evidences, key=lambda e: e.window_seconds):
            scores = {
                label: float(np.dot(evidence.embedding, profile.centroid))
                for label, profile in self._profiles.items()
            }
            ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            similarity = ordered[0][1] if ordered else -1.0
            result.append({
                "window_seconds": float(evidence.window_seconds),
                "top": ordered[0][0] if ordered else None,
                "similarity": similarity,
                "margin": similarity - (ordered[1][1] if len(ordered) > 1 else -1.0),
                "scores": scores,
            })
        return result

    def _young_votes(self, label: str, media_time: float, rankings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        profile = self._profiles[label]
        details: list[dict[str, Any]] = []
        independent_count = valid_count = 0
        for ranking in rankings:
            window = float(ranking["window_seconds"])
            end = profile.latest_sentence_end
            independent = end is None or media_time - window >= end + float(self.config.self_echo_guard_seconds) - 1e-9
            valid = (
                independent
                and ranking["top"] == label
                and float(ranking["similarity"]) >= float(self.config.young_min_similarity)
                and float(ranking["margin"]) >= float(self.config.young_min_margin)
            )
            independent_count += int(independent)
            valid_count += int(valid)
            details.append({
                "window_seconds": window,
                "independent": independent,
                "contaminated": not independent,
                "top": ranking["top"],
                "similarity": float(ranking["similarity"]),
                "margin": float(ranking["margin"]),
                "valid": valid,
            })
        return details, independent_count, valid_count

    def _profile_quality_fast_lease_vote(
        self,
        label: str,
        young_details: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate a provisional lease from the already-computed shortest window."""

        profile = self._profiles[label]
        profile_eligible = (
            profile.sentence_count
            >= int(self.config.profile_quality_fast_lease_min_sentence_count)
            or profile.speech_seconds
            >= float(self.config.profile_quality_fast_lease_min_speech_seconds)
        )
        short = young_details[0] if young_details else None
        vote = bool(
            self.config.enable_young_profile_lease
            and self.config.enable_profile_quality_short_scale_fast_lease
            and profile_eligible
            and short is not None
            and short["independent"]
            and short["top"] == label
            and float(short["similarity"])
            >= float(self.config.profile_quality_fast_lease_min_similarity)
            and float(short["margin"])
            >= float(self.config.profile_quality_fast_lease_min_margin)
        )
        return vote, {
            "enabled": bool(self.config.enable_profile_quality_short_scale_fast_lease),
            "profile_eligible": profile_eligible,
            "profile_sentence_count": profile.sentence_count,
            "profile_speech_seconds": profile.speech_seconds,
            "short_window_seconds": None if short is None else short["window_seconds"],
            "short_independent": bool(short is not None and short["independent"]),
            "short_top_matches_target": bool(short is not None and short["top"] == label),
            "short_similarity": None if short is None else short["similarity"],
            "short_margin": None if short is None else short["margin"],
            "min_sentence_count": int(self.config.profile_quality_fast_lease_min_sentence_count),
            "min_speech_seconds": float(self.config.profile_quality_fast_lease_min_speech_seconds),
            "min_similarity": float(self.config.profile_quality_fast_lease_min_similarity),
            "min_margin": float(self.config.profile_quality_fast_lease_min_margin),
            "vote": vote,
        }

    def _profile_quality_meta_qvote(
        self,
        label: str,
        young_details: list[dict[str, Any]],
    ) -> bool:
        """Evaluate the profile-quality evidence without changing run018 output."""

        if label not in self._profiles or self._is_mature(label):
            return False
        profile = self._profiles[label]
        profile_eligible = (
            profile.sentence_count
            >= int(self.config.profile_quality_fast_lease_min_sentence_count)
            or profile.speech_seconds
            >= float(self.config.profile_quality_fast_lease_min_speech_seconds)
        )
        short = young_details[0] if young_details else None
        return bool(
            self.config.enable_young_profile_lease
            and profile_eligible
            and short is not None
            and short["independent"]
            and short["top"] == label
            and float(short["similarity"])
            >= float(self.config.profile_quality_fast_lease_min_similarity)
            and float(short["margin"])
            >= float(self.config.profile_quality_fast_lease_min_margin)
        )

    @staticmethod
    def _target_margin(ranking: dict[str, Any], label: str) -> float:
        scores = ranking.get("scores") or {}
        target = float(scores.get(label, -1.0))
        runner_up = max(
            (float(value) for key, value in scores.items() if key != label),
            default=-1.0,
        )
        return target - runner_up

    def _apply_profile_quality_meta_lease(
        self,
        baseline: LiveSpeakerDecision,
        precision: LiveSpeakerDecision,
        item: MultiScaleStep,
        *,
        rankings: list[dict[str, Any]],
        young_details: list[dict[str, Any]],
        resets: list[str],
    ) -> LiveSpeakerDecision:
        """Choose the recall expert for at most one probe interval."""

        if not self.config.enable_profile_quality_meta_lease:
            return precision

        previous_lease = self._profile_quality_meta_leased_profile
        target = baseline.visible_speaker
        experts_disagree = target != precision.visible_speaker
        details: dict[str, Any] = {
            "algorithm_id": PROFILE_QUALITY_META_LEASE_ID,
            "enabled": True,
            "probe_scheduled": bool(item.probe_scheduled),
            "recall_visible_speaker": target,
            "precision_visible_speaker": precision.visible_speaker,
            "experts_disagree": experts_disagree,
            "previous_leased_profile": previous_lease,
            "leased_profile": None,
            "branch": None,
            "used": False,
            "profile_age_seconds": None,
            "profile_sentence_count": None,
            "profile_speech_seconds": None,
            "active_profile_count": len(self._profiles),
            "short_top": None,
            "long_top": None,
            "short_target_margin": None,
            "long_target_margin": None,
            "profile_quality_qvote": False,
            "lease_expired_at_probe": False,
            "reset_reasons": list(resets),
            "state_reason": "disabled",
            "thresholds": {
                "fresh_min_age_seconds": float(
                    self.config.profile_quality_meta_fresh_min_age_seconds
                ),
                "fresh_max_age_seconds": float(
                    self.config.profile_quality_meta_fresh_max_age_seconds
                ),
                "fresh_min_speech_seconds": float(
                    self.config.profile_quality_meta_fresh_min_speech_seconds
                ),
                "fresh_min_short_margin": float(
                    self.config.profile_quality_meta_fresh_min_short_margin
                ),
                "fresh_min_long_margin": float(
                    self.config.profile_quality_meta_fresh_min_long_margin
                ),
                "independent_max_profile_count": int(
                    self.config.profile_quality_meta_independent_max_profile_count
                ),
                "switch_min_short_margin": float(
                    self.config.profile_quality_meta_switch_min_short_margin
                ),
            },
        }

        def finish(
            decision: LiveSpeakerDecision,
            *,
            reason: str,
            branch: str | None = None,
            used: bool = False,
        ) -> LiveSpeakerDecision:
            details["state_reason"] = reason
            details["branch"] = branch
            details["used"] = used
            details["leased_profile"] = self._profile_quality_meta_leased_profile
            diagnostics = {
                **dict(precision.diagnostics),
                "profile_quality_meta_algorithm_id": PROFILE_QUALITY_META_LEASE_ID,
                "profile_quality_meta_lease": details,
                "profile_quality_meta_lease_used": used,
                "profile_quality_meta_leased_profile": (
                    self._profile_quality_meta_leased_profile
                ),
            }
            return replace(decision, diagnostics=diagnostics)

        if item.release_signal or "release" in resets:
            self._reset_profile_quality_meta_lease()
            return finish(precision, reason="release")

        if not experts_disagree:
            self._reset_profile_quality_meta_lease()
            return finish(precision, reason="expert_agreement")

        if target is None:
            self._reset_profile_quality_meta_lease()
            return finish(precision, reason="recall_expert_off")

        if not item.probe_scheduled:
            if previous_lease == target:
                return finish(
                    replace(baseline, reason="profile_quality_meta_lease_hold"),
                    reason="lease_hold",
                    branch="hold",
                    used=True,
                )
            self._reset_profile_quality_meta_lease()
            return finish(
                precision,
                reason="target_changed" if previous_lease is not None else "no_active_lease",
            )

        details["lease_expired_at_probe"] = previous_lease is not None
        self._reset_profile_quality_meta_lease()
        if target not in self._profiles:
            return finish(precision, reason="target_profile_missing")
        if len(rankings) != 2:
            return finish(precision, reason="two_scales_required")

        short, long = rankings
        profile = self._profiles[target]
        profile_age = float(item.media_time) - profile.available_at
        short_margin = self._target_margin(short, target)
        long_margin = self._target_margin(long, target)
        both_top_target = short["top"] == target and long["top"] == target
        if not young_details and not self._is_mature(target):
            young_details, _, _ = self._young_votes(
                target, float(item.media_time), rankings
            )
        qvote = self._profile_quality_meta_qvote(target, young_details)
        details.update({
            "profile_age_seconds": profile_age,
            "profile_sentence_count": profile.sentence_count,
            "profile_speech_seconds": profile.speech_seconds,
            "short_top": short["top"],
            "long_top": long["top"],
            "short_target_margin": short_margin,
            "long_target_margin": long_margin,
            "profile_quality_qvote": qvote,
        })

        fresh = bool(
            float(self.config.profile_quality_meta_fresh_min_age_seconds)
            <= profile_age
            <= float(self.config.profile_quality_meta_fresh_max_age_seconds)
            and profile.speech_seconds
            >= float(self.config.profile_quality_meta_fresh_min_speech_seconds)
            and both_top_target
            and short_margin
            >= float(self.config.profile_quality_meta_fresh_min_short_margin)
            and long_margin
            >= float(self.config.profile_quality_meta_fresh_min_long_margin)
        )
        independent = bool(
            qvote
            and (
                (
                    baseline.action != "switch"
                    and both_top_target
                    and len(self._profiles)
                    <= int(self.config.profile_quality_meta_independent_max_profile_count)
                )
                or (
                    baseline.action == "switch"
                    and short_margin
                    >= float(self.config.profile_quality_meta_switch_min_short_margin)
                )
            )
        )
        branch = "fresh" if fresh else "independent" if independent else None
        if branch is None:
            return finish(precision, reason="evidence_rejected")

        self._profile_quality_meta_leased_profile = target
        return finish(
            replace(baseline, reason=f"profile_quality_meta_{branch}_lease"),
            reason="lease_started",
            branch=branch,
            used=True,
        )

    def _normalize_profile_quality_meta_output_action(
        self,
        decision: LiveSpeakerDecision,
    ) -> LiveSpeakerDecision:
        """Express the final meta output relative to what the GUI last received."""

        if not self.config.enable_profile_quality_meta_lease:
            return decision
        previous = self._profile_quality_meta_emitted_visible
        current = decision.visible_speaker
        if previous is None and current is None:
            action = "none"
        elif previous is None:
            action = "acquire"
        elif current is None:
            action = "clear"
        elif previous != current:
            action = "switch"
        else:
            action = "hold"
        self._profile_quality_meta_emitted_visible = current
        diagnostics = {
            **dict(decision.diagnostics),
            "profile_quality_meta_previous_emitted_visible": previous,
            "profile_quality_meta_emitted_visible": current,
            "profile_quality_meta_normalized_action": action,
        }
        return replace(decision, action=action, diagnostics=diagnostics)

    def _update_boundary(
        self,
        baseline: LiveSpeakerDecision,
        item: MultiScaleStep,
        rankings: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        incumbent = baseline.visible_speaker
        if not self.config.enable_boundary_abstention or incumbent is None:
            self._reset_boundary()
            return False, "disabled_or_off"
        if self._boundary_active and incumbent != self._boundary_incumbent:
            self._reset_boundary()
            return False, "baseline_switched"
        if not item.probe_scheduled:
            return self._boundary_active, "active_non_probe" if self._boundary_active else "non_probe"
        if len(rankings) != 2:
            self._reset_boundary()
            return False, "missing_scale"
        short, long = rankings
        if self._boundary_active and short["top"] == long["top"] and short["top"] is not None:
            self._reset_boundary()
            return False, "scales_reconverged"
        challenger = short["top"]
        short_scores, long_scores = short["scores"], long["scores"]
        conflict = (
            self._last_baseline_visible == incumbent
            and challenger is not None
            and challenger != incumbent
            and long["top"] == incumbent
            and float(short["similarity"]) >= float(self.config.boundary_min_similarity)
            and float(long["similarity"]) >= float(self.config.boundary_min_similarity)
            and float(short["margin"]) >= float(self.config.boundary_min_margin)
            and float(long["margin"]) >= float(self.config.boundary_min_margin)
            and short_scores.get(challenger, -1.0) - short_scores.get(incumbent, -1.0)
            >= float(self.config.boundary_short_advantage)
            and long_scores.get(incumbent, -1.0) - long_scores.get(challenger, -1.0)
            >= float(self.config.boundary_long_advantage)
        )
        if conflict:
            same_pair = self._boundary_incumbent == incumbent and self._boundary_challenger == challenger
            self._boundary_streak = self._boundary_streak + 1 if same_pair else 1
            self._boundary_incumbent = incumbent
            self._boundary_challenger = challenger
            if self._boundary_streak >= int(self.config.boundary_required_consecutive_probes):
                self._boundary_active = True
            return self._boundary_active, "short_long_conflict"
        if self._boundary_active:
            return True, "unresolved_boundary"
        self._reset_boundary()
        return False, "no_conflict"

    def _diagnostics(
        self,
        baseline: LiveSpeakerDecision,
        *,
        applied: list[str],
        rankings: list[dict[str, Any]],
        resets: list[str],
        young_details: list[dict[str, Any]],
        intervention: str,
        boundary_reason: str,
        short_scale_fast_lease_used: bool,
        profile_quality_fast_lease_details: dict[str, Any],
        profile_quality_fast_lease_used: bool,
    ) -> dict[str, Any]:
        return {
            **dict(baseline.diagnostics),
            "hybrid_algorithm_id": HYBRID_ALGORITHM_ID,
            "hybrid_intervention": intervention,
            "profile_events_applied": applied,
            "first_official_profile": self._first_official,
            "immature_profiles": sorted(label for label in self._profiles if not self._is_mature(label)),
            "permanently_confirmed_profiles": sorted(self._permanently_confirmed),
            "leased_profile": self._leased_profile,
            "young_candidate": self._young_candidate,
            "young_streak": self._young_streak,
            "young_scale_votes": young_details,
            "young_probe_independent_scale_count": sum(
                int(detail["independent"]) for detail in young_details
            ),
            "young_probe_valid_scale_count": sum(
                int(detail["valid"]) for detail in young_details
            ),
            "young_probe_short_scale_valid": bool(
                young_details and young_details[0]["valid"]
            ),
            "short_scale_fast_lease_used": short_scale_fast_lease_used,
            "profile_quality_fast_lease": profile_quality_fast_lease_details,
            "profile_quality_fast_lease_used": profile_quality_fast_lease_used,
            "independent_windows": [d["window_seconds"] for d in young_details if d["independent"]],
            "contaminated_windows": [d["window_seconds"] for d in young_details if d["contaminated"]],
            "scale_rankings": rankings,
            "boundary_active": self._boundary_active,
            "boundary_incumbent": self._boundary_incumbent,
            "boundary_challenger": self._boundary_challenger,
            "boundary_streak": self._boundary_streak,
            "boundary_state_reason": boundary_reason,
            "history_reset_reasons": resets,
        }

    def step(self, baseline: LiveSpeakerDecision, item: MultiScaleStep) -> LiveSpeakerDecision:
        self._validate(baseline, item)
        media_time = float(item.media_time)
        self._last_media_time = media_time
        if not (
            self.config.enable_young_profile_confirmation
            or self.config.enable_boundary_abstention
            or self.config.enable_profile_quality_meta_lease
        ):
            return baseline

        resets: list[str] = []
        applied = self._apply_profile_events(media_time)
        if applied:
            resets.append("profile_change")
        if (
            self._last_full_probe_time is not None
            and media_time - self._last_full_probe_time > float(self.config.history_max_gap_seconds) + 1e-9
        ):
            self._reset_temporal()
            resets.append("probe_gap")
        if item.release_signal:
            self._reset_temporal()
            resets.append("release")

        rankings = self._rankings(item) if item.probe_scheduled else []
        if item.probe_scheduled and len(rankings) == 2:
            self._last_full_probe_time = media_time

        target = baseline.visible_speaker
        young_details: list[dict[str, Any]] = []
        young_block = False
        young_reason = ""
        short_scale_fast_lease_used = False
        profile_quality_fast_lease_details: dict[str, Any] = {}
        profile_quality_fast_lease_used = False
        if (
            self.config.enable_young_profile_confirmation
            and target is not None
            and target in self._profiles
            and not self._is_mature(target)
        ):
            if item.probe_scheduled:
                young_details, independent_count, valid_count = self._young_votes(target, media_time, rankings)
                all_independent_valid = independent_count == valid_count
                permanent_vote = (
                    independent_count >= int(self.config.young_independent_scale_count)
                    and all_independent_valid
                )
                if permanent_vote:
                    self._young_streak = self._young_streak + 1 if self._young_candidate == target else 1
                    self._young_candidate = target
                    if self._young_streak >= int(self.config.young_required_consecutive_probes):
                        self._permanently_confirmed.add(target)
                        self._leased_profile = None
                        young_reason = "young_profile_confirmed"
                    else:
                        young_block = True
                        young_reason = "young_profile_confirmation_pending"
                else:
                    self._young_candidate = target
                    self._young_streak = 0
                    fast_scale_count = int(self.config.young_fast_independent_scale_count)
                    strict_fast_vote = (
                        self.config.enable_young_profile_lease
                        and independent_count >= fast_scale_count
                        and valid_count >= fast_scale_count
                        and all_independent_valid
                    )
                    short_scale_vote = (
                        self.config.enable_young_profile_lease
                        and self.config.enable_short_scale_fast_lease
                        and fast_scale_count == 1
                        and bool(young_details)
                        and bool(young_details[0]["valid"])
                    )
                    profile_quality_fast_vote, profile_quality_fast_lease_details = (
                        self._profile_quality_fast_lease_vote(target, young_details)
                    )
                    fast_vote = strict_fast_vote or short_scale_vote or profile_quality_fast_vote
                    if fast_vote:
                        self._leased_profile = target
                        short_scale_fast_lease_used = short_scale_vote and not strict_fast_vote
                        profile_quality_fast_lease_used = (
                            profile_quality_fast_vote
                            and not strict_fast_vote
                            and not short_scale_vote
                        )
                        young_reason = "young_profile_lease_renewed"
                    else:
                        self._leased_profile = None
                        young_block = True
                        young_reason = "young_profile_unconfirmed"
            elif self._leased_profile == target and self.config.enable_young_profile_lease:
                young_reason = "young_profile_lease_hold"
            else:
                young_block = True
                young_reason = "young_profile_wait_probe"
        elif self._leased_profile is not None and target != self._leased_profile:
            self._reset_young()
            resets.append("profile_mismatch")

        if young_block:
            self._reset_boundary()
            boundary_block, boundary_reason = False, "young_gate_precedence"
        else:
            boundary_block, boundary_reason = self._update_boundary(baseline, item, rankings)

        if young_block or boundary_block:
            intervention = young_reason if young_block else "boundary_abstention"
            diagnostics = self._diagnostics(
                baseline, applied=applied, rankings=rankings, resets=resets,
                young_details=young_details, intervention=intervention,
                boundary_reason=boundary_reason,
                short_scale_fast_lease_used=short_scale_fast_lease_used,
                profile_quality_fast_lease_details=profile_quality_fast_lease_details,
                profile_quality_fast_lease_used=profile_quality_fast_lease_used,
            )
            result = replace(
                baseline,
                visible_speaker=None,
                action="clear" if self._visible is not None else "none",
                reason=intervention,
                diagnostics=diagnostics,
            )
            self._visible = None
            self._abstaining = True
        else:
            intervention = young_reason or (
                "boundary_recovered" if boundary_reason in {"baseline_switched", "scales_reconverged"} else "none"
            )
            diagnostics = self._diagnostics(
                baseline, applied=applied, rankings=rankings, resets=resets,
                young_details=young_details, intervention=intervention,
                boundary_reason=boundary_reason,
                short_scale_fast_lease_used=short_scale_fast_lease_used,
                profile_quality_fast_lease_details=profile_quality_fast_lease_details,
                profile_quality_fast_lease_used=profile_quality_fast_lease_used,
            )
            result = replace(baseline, diagnostics=diagnostics)
            if self._abstaining and baseline.visible_speaker is not None and baseline.action in {"hold", "none"}:
                result = replace(result, action="acquire", reason="hybrid_recover_baseline")
            self._visible = baseline.visible_speaker
            self._abstaining = False

        result = self._apply_profile_quality_meta_lease(
            baseline,
            result,
            item,
            rankings=rankings,
            young_details=young_details,
            resets=resets,
        )
        result = self._normalize_profile_quality_meta_output_action(result)
        self._last_baseline_visible = baseline.visible_speaker
        return result


def replay_hybrid_decisions(
    baseline_decisions: Iterable[LiveSpeakerDecision],
    multiscale_steps: Iterable[MultiScaleStep],
    profile_events: Iterable[SpeakerProfileEvent] = (),
    config: HybridSpeakerTrackerConfig | None = None,
) -> list[LiveSpeakerDecision]:
    tracker = CausalHybridSpeakerTracker(config=config, profile_events=profile_events)
    decisions: list[LiveSpeakerDecision] = []
    baseline_iterator: Iterator[LiveSpeakerDecision] = iter(baseline_decisions)
    step_iterator: Iterator[MultiScaleStep] = iter(multiscale_steps)
    sentinel = object()
    while True:
        baseline = next(baseline_iterator, sentinel)
        item = next(step_iterator, sentinel)
        if baseline is sentinel and item is sentinel:
            return decisions
        if baseline is sentinel or item is sentinel:
            raise ValueError("baseline decisions and multi-scale steps must have equal lengths")
        decisions.append(tracker.step(baseline, item))


def replay_cached_hybrid_windows(
    baseline_decisions: Iterable[LiveSpeakerDecision],
    multiscale_steps: Iterable[MultiScaleStep],
    profile_events: Iterable[SpeakerProfileEvent] = (),
    config: HybridSpeakerTrackerConfig | None = None,
) -> list[LiveSpeakerDecision]:
    return replay_hybrid_decisions(
        baseline_decisions, multiscale_steps, profile_events=profile_events, config=config
    )
