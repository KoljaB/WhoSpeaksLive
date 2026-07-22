"""Causal output-only temporary identities for the live speaker indicator.

The overlay consumes the two embeddings already computed for the Bayesian live
tracker.  It never inserts temporary identities into Bayesian state or final
speaker memory.  The same state machine is used by the GUI and World-Tape
counterfactual replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


OPEN_SET_TRACKLET_PRESET = "short_history_hybrid_v1"
PROFILE_CONTRADICTION_TRACKLET_PRESET = (
    "short_history_hybrid_v2_profile_contradiction"
)


def _unit(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return np.asarray(array / norm, dtype=np.float32)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


def _unit_or(value: Any, fallback: np.ndarray) -> np.ndarray:
    normalized = _unit(value)
    return fallback if normalized is None else normalized


@dataclass(frozen=True)
class OpenSetTrackletConfig:
    preset: str = OPEN_SET_TRACKLET_PRESET
    confirmation_probes: int = 2
    novelty_short_ceiling: float = 0.20
    novelty_long_ceiling: float = 0.25
    pending_short_min: float = 0.30
    pending_long_min: float = 0.25
    pending_max_gap_seconds: float = 1.20
    reuse_short_min: float = 0.40
    weak_reactivation_short_min: float = 0.25
    weak_reactivation_long_min: float = 0.45
    known_advantage_margin: float = 0.02
    weak_reactivation_known_advantage_margin: float = 0.0
    merge_min_similarity: float = 0.35
    update_alpha: float = 0.25
    max_tracklets: int = 12
    reuse_idle_ttl_seconds: float = 8.0
    temporary_prefix: str = "LIVE_TRACKLET_"
    profile_contradiction_enabled: bool = False
    profile_contradiction_min_profiles: int = 4
    profile_contradiction_short_ceiling: float = 0.36
    profile_contradiction_long_ceiling: float = 0.36

    def __post_init__(self) -> None:
        if self.preset not in {
            OPEN_SET_TRACKLET_PRESET,
            PROFILE_CONTRADICTION_TRACKLET_PRESET,
        }:
            raise ValueError(f"Unsupported open-set tracklet preset: {self.preset}")
        expected_contradiction = self.preset == PROFILE_CONTRADICTION_TRACKLET_PRESET
        if bool(self.profile_contradiction_enabled) != expected_contradiction:
            raise ValueError(
                "Open-set tracklet preset/config mismatch; use "
                "open_set_tracklet_config_for_preset()"
            )
        if int(self.confirmation_probes) < 2:
            raise ValueError("Open-set tracklets require at least two confirmation probes")
        if float(self.pending_max_gap_seconds) <= 0.0:
            raise ValueError("pending_max_gap_seconds must be positive")
        if not 0.0 <= float(self.update_alpha) <= 1.0:
            raise ValueError("update_alpha must be in [0, 1]")
        if int(self.max_tracklets) < 1:
            raise ValueError("max_tracklets must be positive")
        if float(self.reuse_idle_ttl_seconds) < 0.0:
            raise ValueError("reuse_idle_ttl_seconds must be non-negative")
        if int(self.profile_contradiction_min_profiles) < 1:
            raise ValueError("profile_contradiction_min_profiles must be positive")
        for name, value in (
            ("profile_contradiction_short_ceiling", self.profile_contradiction_short_ceiling),
            ("profile_contradiction_long_ceiling", self.profile_contradiction_long_ceiling),
        ):
            if not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def open_set_tracklet_config_for_preset(preset: str) -> OpenSetTrackletConfig:
    """Decode one versioned preset identically for GUI and counterfactual replay."""

    normalized = str(preset or OPEN_SET_TRACKLET_PRESET)
    if normalized == OPEN_SET_TRACKLET_PRESET:
        return OpenSetTrackletConfig(preset=OPEN_SET_TRACKLET_PRESET)
    if normalized == PROFILE_CONTRADICTION_TRACKLET_PRESET:
        return OpenSetTrackletConfig(
            preset=PROFILE_CONTRADICTION_TRACKLET_PRESET,
            profile_contradiction_enabled=True,
            profile_contradiction_min_profiles=4,
            profile_contradiction_short_ceiling=0.36,
            profile_contradiction_long_ceiling=0.36,
        )
    raise ValueError(f"Unsupported open-set tracklet preset: {normalized}")


@dataclass(frozen=True)
class OpenSetTrackletAlias:
    alias_generation: int
    final_internal_speaker_id: str
    surviving_public_speaker_id: str
    merge_similarity: float
    final_speaker: dict[str, Any]
    retired: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "alias_generation": int(self.alias_generation),
            "final_internal_speaker_id": self.final_internal_speaker_id,
            "surviving_public_speaker_id": self.surviving_public_speaker_id,
            "merge_similarity": float(self.merge_similarity),
            "final_speaker": dict(self.final_speaker),
            "retired": bool(self.retired),
        }


@dataclass(frozen=True)
class OpenSetTrackletStep:
    media_time: float
    speech: bool
    probe_scheduled: bool
    release_signal: bool
    short_embedding: np.ndarray | None
    long_embedding: np.ndarray | None
    profiles: tuple[dict[str, Any], ...]
    base_visible_speaker: str | None
    base_action: str
    base_reason: str


@dataclass(frozen=True)
class OpenSetTrackletDecision:
    visible_speaker: str | None
    action: str
    reason: str
    provisional_speaker: bool
    created_speaker: bool
    aliases: tuple[OpenSetTrackletAlias, ...]
    diagnostics: dict[str, Any]


@dataclass
class _PendingNovelVoice:
    short: np.ndarray
    long: np.ndarray | None
    count: int
    last_media_time: float


@dataclass
class _TemporaryTracklet:
    public_id: str
    short_centroid: np.ndarray
    long_centroid: np.ndarray | None
    probe_count: int
    created_media_time: float
    last_media_time: float

    def update(
        self,
        short: np.ndarray,
        long: np.ndarray | None,
        media_time: float,
        alpha: float,
    ) -> None:
        self.short_centroid = _unit_or(
            (1.0 - alpha) * self.short_centroid + alpha * short,
            self.short_centroid,
        )
        if long is not None:
            if self.long_centroid is None:
                self.long_centroid = long.copy()
            else:
                self.long_centroid = _unit_or(
                    (1.0 - alpha) * self.long_centroid + alpha * long,
                    self.long_centroid,
                )
        self.probe_count += 1
        self.last_media_time = float(media_time)


class OpenSetTrackletOverlay:
    """Project a stable temporary public identity over an unchanged base tracker."""

    def __init__(self, config: OpenSetTrackletConfig | None = None) -> None:
        self.config = config or OpenSetTrackletConfig()
        self._pending: _PendingNovelVoice | None = None
        self._contradiction_pending = False
        self._contradiction_active = False
        self._tracklets: list[_TemporaryTracklet] = []
        self._ever_seen_final_labels: set[str] = set()
        self._final_to_public: dict[str, str] = {}
        self._public_to_final: dict[str, str] = {}
        self._visible: str | None = None
        self._next_tracklet_index = 1
        self._alias_generation = 0
        self._stats = {
            "pending_started": 0,
            "pending_confirmed": 0,
            "pending_rejected": 0,
            "tracklets_created": 0,
            "tracklet_reuses": 0,
            "profile_merges": 0,
            "alias_conflicts": 0,
        }

    @property
    def final_to_public(self) -> dict[str, str]:
        return dict(self._final_to_public)

    @property
    def public_to_final(self) -> dict[str, str]:
        return dict(self._public_to_final)

    def identity_snapshot(self) -> tuple[int, dict[str, str], dict[str, str]]:
        return (
            int(self._alias_generation),
            dict(self._final_to_public),
            dict(self._public_to_final),
        )

    def _profile_vectors(
        self, profiles: Iterable[dict[str, Any]]
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        vectors: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for raw in profiles:
            label = str(raw.get("label") or raw.get("id") or "").strip()
            vector = _unit(raw.get("centroid"))
            if not label or vector is None:
                continue
            vectors[label] = vector
            metadata[label] = {
                key: value for key, value in dict(raw).items() if key != "centroid"
            }
            metadata[label].setdefault("id", label)
        return vectors, metadata

    def _best_tracklet(
        self,
        short: np.ndarray | None,
        long: np.ndarray | None,
        media_time: float,
    ) -> tuple[_TemporaryTracklet | None, float, float]:
        best: tuple[float, _TemporaryTracklet, float, float] | None = None
        for item in self._tracklets:
            if (
                float(media_time) - float(item.last_media_time)
                > float(self.config.reuse_idle_ttl_seconds)
            ):
                # Keep the slot and every alias intact, but do not let an
                # inactive prototype independently re-enter either reuse path
                # as a stale identity.  A bound final profile can still reach
                # the same public ID through the authoritative base tracker.
                continue
            short_score = _cosine(item.short_centroid, short)
            long_score = _cosine(item.long_centroid, long)
            rank = 0.7 * short_score + 0.3 * (
                long_score if long is not None else short_score
            )
            if best is None or rank > best[0]:
                best = (rank, item, short_score, long_score)
        if best is None:
            return None, -1.0, -1.0
        return best[1], best[2], best[3]

    def _reuse_pass(
        self,
        short_similarity: float,
        long_similarity: float,
        known_short_similarity: float = -1.0,
        *,
        compare_known: bool = False,
        allow_relaxed: bool = True,
    ) -> bool:
        fast = short_similarity >= float(self.config.reuse_short_min)
        if compare_known:
            fast = fast and short_similarity >= (
                known_short_similarity + float(self.config.known_advantage_margin)
            )
        relaxed = allow_relaxed and (
            short_similarity >= float(self.config.weak_reactivation_short_min)
            and long_similarity >= float(self.config.weak_reactivation_long_min)
        )
        if compare_known:
            relaxed = relaxed and short_similarity >= (
                known_short_similarity
                + float(self.config.weak_reactivation_known_advantage_margin)
            )
        return fast or relaxed

    def _sync_new_profiles(
        self,
        vectors: dict[str, np.ndarray],
        metadata: dict[str, dict[str, Any]],
    ) -> tuple[OpenSetTrackletAlias, ...]:
        aliases: list[OpenSetTrackletAlias] = []
        first_publication_labels = sorted(
            set(vectors) - self._ever_seen_final_labels
        )
        self._ever_seen_final_labels.update(vectors)
        for final_label in sorted(set(self._final_to_public) - set(vectors)):
            public_id = self._final_to_public.pop(final_label)
            self._public_to_final.pop(public_id, None)
            self._alias_generation += 1
            aliases.append(OpenSetTrackletAlias(
                alias_generation=self._alias_generation,
                final_internal_speaker_id=final_label,
                surviving_public_speaker_id=public_id,
                merge_similarity=-1.0,
                final_speaker={"id": final_label},
                retired=True,
            ))
        for final_label in first_publication_labels:
            if final_label in self._final_to_public or not self._tracklets:
                continue
            candidates = [
                item for item in self._tracklets if item.public_id not in self._public_to_final
            ]
            if not candidates:
                self._stats["alias_conflicts"] += 1
                continue
            profile = vectors[final_label]
            scored = [
                (
                    max(
                        _cosine(item.short_centroid, profile),
                        _cosine(item.long_centroid, profile),
                    ),
                    item,
                )
                for item in candidates
            ]
            similarity, tracklet = max(scored, key=lambda row: (row[0], row[1].public_id))
            if similarity < float(self.config.merge_min_similarity):
                continue
            if (
                final_label in self._final_to_public
                or tracklet.public_id in self._public_to_final
            ):
                self._stats["alias_conflicts"] += 1
                continue
            self._final_to_public[final_label] = tracklet.public_id
            self._public_to_final[tracklet.public_id] = final_label
            self._alias_generation += 1
            self._stats["profile_merges"] += 1
            aliases.append(OpenSetTrackletAlias(
                alias_generation=self._alias_generation,
                final_internal_speaker_id=final_label,
                surviving_public_speaker_id=tracklet.public_id,
                merge_similarity=float(similarity),
                final_speaker=dict(metadata.get(final_label) or {"id": final_label}),
            ))
        return tuple(aliases)

    def _decision(
        self,
        chosen: str | None,
        reason: str,
        aliases: tuple[OpenSetTrackletAlias, ...],
        *,
        created: bool = False,
        base_action: str = "",
    ) -> OpenSetTrackletDecision:
        previous = self._visible
        if chosen:
            action = "acquire" if previous is None else ("switch" if previous != chosen else "hold")
        else:
            action = "clear" if previous is not None else ("none" if not base_action else base_action)
        self._visible = chosen
        diagnostics = {
            "preset": self.config.preset,
            "final_to_public": dict(self._final_to_public),
            "public_to_final": dict(self._public_to_final),
            "tracklet_count": len(self._tracklets),
            "pending_count": 0 if self._pending is None else int(self._pending.count),
            **{key: int(value) for key, value in self._stats.items()},
        }
        if self.config.profile_contradiction_enabled:
            diagnostics.update({
                "profile_contradiction_active": bool(self._contradiction_active),
                "profile_contradiction_pending": bool(self._contradiction_pending),
                "profile_contradiction_min_profiles": int(
                    self.config.profile_contradiction_min_profiles
                ),
                "profile_contradiction_short_ceiling": float(
                    self.config.profile_contradiction_short_ceiling
                ),
                "profile_contradiction_long_ceiling": float(
                    self.config.profile_contradiction_long_ceiling
                ),
            })
        return OpenSetTrackletDecision(
            visible_speaker=chosen,
            action=action,
            reason=reason,
            provisional_speaker=bool(chosen and chosen.startswith(self.config.temporary_prefix)),
            created_speaker=bool(created),
            aliases=aliases,
            diagnostics=diagnostics,
        )

    def step(self, item: OpenSetTrackletStep) -> OpenSetTrackletDecision:
        previous_contradiction_pending = bool(self._contradiction_pending)
        self._contradiction_pending = False
        self._contradiction_active = False
        profiles, metadata = self._profile_vectors(item.profiles)
        aliases = self._sync_new_profiles(profiles, metadata)
        base_visible = (
            self._final_to_public.get(str(item.base_visible_speaker), str(item.base_visible_speaker))
            if item.base_visible_speaker else None
        )
        short = _unit(item.short_embedding)
        long = _unit(item.long_embedding)

        if item.release_signal or not item.speech or not item.probe_scheduled or short is None:
            self._pending = None
            return self._decision(
                base_visible,
                str(item.base_reason or "open_set_base"),
                aliases,
                base_action=str(item.base_action or ""),
            )

        known_short = max((_cosine(short, value) for value in profiles.values()), default=-1.0)
        known_long = max((_cosine(long, value) for value in profiles.values()), default=-1.0)
        profile_contradiction = bool(
            self.config.profile_contradiction_enabled
            and len(profiles) >= int(self.config.profile_contradiction_min_profiles)
            and known_short < float(self.config.profile_contradiction_short_ceiling)
            and (
                long is None
                or known_long < float(self.config.profile_contradiction_long_ceiling)
            )
            and (bool(item.base_visible_speaker) or previous_contradiction_pending)
        )
        self._contradiction_active = profile_contradiction
        effective_base_visible = None if profile_contradiction else base_visible
        effective_base_action = (
            "clear" if profile_contradiction else str(item.base_action or "")
        )
        effective_base_reason = (
            "profile_contradiction_quarantine"
            if profile_contradiction
            else str(item.base_reason or "open_set_base")
        )
        novelty_short_ceiling = (
            float(self.config.profile_contradiction_short_ceiling)
            if profile_contradiction
            else float(self.config.novelty_short_ceiling)
        )
        novelty_long_ceiling = (
            float(self.config.profile_contradiction_long_ceiling)
            if profile_contradiction
            else float(self.config.novelty_long_ceiling)
        )
        best, best_short, best_long = self._best_tracklet(
            short, long, item.media_time
        )
        reusable = bool(
            best
            and self._reuse_pass(
                best_short,
                best_long,
                known_short,
                compare_known=bool(profiles),
            )
        )
        novel = bool(
            not profiles
            or (
                known_short < novelty_short_ceiling
                and (long is None or known_long < novelty_long_ceiling)
            )
        )

        if reusable and best is not None:
            best.update(short, long, item.media_time, float(self.config.update_alpha))
            self._pending = None
            self._stats["tracklet_reuses"] += 1
            return self._decision(best.public_id, "open_set_tracklet_reuse", aliases)

        if not novel:
            self._pending = None
            return self._decision(
                effective_base_visible,
                effective_base_reason,
                aliases,
                base_action=effective_base_action,
            )

        consistent = bool(
            self._pending
            and float(item.media_time) - self._pending.last_media_time
            <= float(self.config.pending_max_gap_seconds)
            and _cosine(self._pending.short, short) >= float(self.config.pending_short_min)
        )
        if consistent and self._pending is not None:
            self._pending.count += 1
            self._pending.short = _unit_or(
                0.5 * self._pending.short + 0.5 * short, short.copy()
            )
            if long is not None:
                self._pending.long = (
                    long.copy()
                    if self._pending.long is None
                    else _unit_or(0.5 * self._pending.long + 0.5 * long, long.copy())
                )
            self._pending.last_media_time = float(item.media_time)
        else:
            if self._pending is not None:
                self._stats["pending_rejected"] += 1
            self._pending = _PendingNovelVoice(
                short=short.copy(),
                long=None if long is None else long.copy(),
                count=1,
                last_media_time=float(item.media_time),
            )
            self._stats["pending_started"] += 1

        if self._pending.count < int(self.config.confirmation_probes):
            self._contradiction_pending = profile_contradiction
            return self._decision(None, "open_set_pending_unknown", aliases)

        candidate, candidate_short, candidate_long = self._best_tracklet(
            self._pending.short, self._pending.long, item.media_time
        )
        if candidate is not None and self._reuse_pass(
            candidate_short,
            candidate_long,
            allow_relaxed=False,
        ):
            selected = candidate
            created = False
            self._stats["tracklet_reuses"] += 1
        elif len(self._tracklets) < int(self.config.max_tracklets):
            selected = _TemporaryTracklet(
                public_id=f"{self.config.temporary_prefix}{self._next_tracklet_index}",
                short_centroid=self._pending.short.copy(),
                long_centroid=None if self._pending.long is None else self._pending.long.copy(),
                probe_count=int(self._pending.count),
                created_media_time=float(item.media_time),
                last_media_time=float(item.media_time),
            )
            self._next_tracklet_index += 1
            self._tracklets.append(selected)
            created = True
            self._stats["tracklets_created"] += 1
        else:
            self._pending = None
            return self._decision(None, "open_set_tracklet_capacity", aliases)

        selected.update(short, long, item.media_time, float(self.config.update_alpha))
        self._pending = None
        self._stats["pending_confirmed"] += 1
        return self._decision(
            selected.public_id,
            "open_set_tracklet_confirmed",
            aliases,
            created=created,
        )
