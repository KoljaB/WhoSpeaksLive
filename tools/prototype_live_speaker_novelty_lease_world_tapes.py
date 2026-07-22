"""Replay-only prototype for a causal single-slot novelty lease.

This experiment deliberately lives outside production modules.  It reuses the frozen
0.7/1.5-second SpeechBrain vectors and profile publication events from a World Tape,
then changes only projected live-speaker output actions.  Results are diagnostic replay
nominees and can never promote a production champion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_browser_parity import replay_browser_state
from window.live_speaker_counterfactual import (
    _cached_counterfactual_tape_inputs,
    project_counterfactual_live_actions,
)
from window.live_speaker_probe_scoring import read_canonical_segments


CONTRACT_ID = "whospeaks.live_speaker.novelty_lease_replay_prototype.v1"


@lru_cache(maxsize=32)
def _cached_base_projection(root_key: str, config_json: str) -> dict[str, Any]:
    return project_counterfactual_live_actions(
        Path(root_key), json.loads(config_json)
    )


@lru_cache(maxsize=32)
def _cached_canonical(path_key: str) -> tuple[dict[str, Any], ...]:
    return tuple(read_canonical_segments(Path(path_key)))


@dataclass(frozen=True)
class LeaseConfig:
    min_profiles: int = 3
    change_short_max_similarity: float = 0.50
    novelty_short_max_profile_similarity: float = 0.34
    novelty_long_max_profile_similarity: float = 0.44
    change_short_long_max_similarity: float = 0.62
    activation_streak: int = 1
    prototype_short_weight: float = 0.80
    prototype_update_alpha: float = 0.25
    prototype_update_min_similarity: float = 0.28
    keep_min_similarity: float = 0.24
    death_streak: int = 2
    max_seconds: float = 12.0
    alias_min_similarity: float = 0.48
    public_identity_mode: str = "hidden_until_alias"


def _unit(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return vector / norm


def _cos(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _profiles(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for profile in payload.get("profiles") or []:
        label = str(profile.get("label") or "")
        centroid = _unit(profile.get("centroid"))
        if label and centroid is not None:
            result[label] = centroid
    return result


def _max_profile_similarity(
    vector: np.ndarray | None, profiles: dict[str, np.ndarray]
) -> tuple[str, float]:
    if vector is None or not profiles:
        return "", -1.0
    label, similarity = max(
        ((label, _cos(vector, centroid)) for label, centroid in profiles.items()),
        key=lambda item: item[1],
    )
    return label, similarity


def _action_payload(
    *,
    step_id: int,
    speaker: str,
    payload: dict[str, Any],
    hold_seconds: float,
    source: str,
) -> dict[str, Any]:
    media_time = float(payload.get("media_time") or 0.0)
    duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
    return {
        "step_id": step_id,
        "assigned_speaker": speaker,
        "speaker_id": speaker,
        "internal_speaker_id": speaker,
        "live": True,
        "fallback": True,
        "start": round(max(0.0, media_time - duration), 4),
        "end": round(media_time, 4),
        "audio_length_seconds": round(duration, 4),
        "hold_seconds": round(max(0.0, hold_seconds), 4),
        "assignment_source": source,
    }


def _clear_payload(
    *, step_id: int, speaker: str, payload: dict[str, Any], reason: str
) -> dict[str, Any]:
    media_time = float(payload.get("media_time") or 0.0)
    duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
    return {
        "step_id": step_id,
        "speaker_id": speaker,
        "assigned_speaker": None,
        "live": False,
        "fallback": True,
        "start": round(max(0.0, media_time - duration), 4),
        "end": round(media_time, 4),
        "reason": reason,
        "assignment_source": "counterfactual_novelty_lease",
    }


def project_novelty_lease_actions(
    tape_dir: Path,
    algorithm_config: dict[str, Any],
    lease_config: LeaseConfig,
) -> dict[str, Any]:
    root = Path(tape_dir).resolve()
    base = _cached_base_projection(
        str(root), json.dumps(algorithm_config, sort_keys=True, separators=(",", ":"))
    )
    base_by_step = {
        int(item[3].get("step_id") or 0): item for item in base["actions"]
    }
    base_decisions = {
        int(item.get("step_id") or 0): item for item in base["decisions"]
    }
    inputs, recorded_decisions, public_by_step, recorded_hold = (
        _cached_counterfactual_tape_inputs(str(root))
    )
    hold_seconds = max(
        0.0,
        float(algorithm_config.get("live_speaker_probe_hold_seconds", recorded_hold)),
    )

    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    previous_short: np.ndarray | None = None
    previous_profile_labels: set[str] = set()
    novelty_streak = 0
    lease_active = False
    lease_started = 0.0
    lease_prototype: np.ndarray | None = None
    lease_death_streak = 0
    lease_public_id = "NOVELTY_LEASE_1"
    alias_profile = ""
    permanent_aliases: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    activation_count = 0
    alias_count = 0
    silence_death_count = 0

    for record in inputs:
        payload = dict(record["payload"])
        step_id = int(payload.get("step_id") or 0)
        media_time = float(payload.get("media_time") or 0.0)
        short = _unit(payload.get("embedding"))
        long = _unit(payload.get("context_embedding"))
        profiles = _profiles(payload)
        profile_labels = set(profiles)
        new_labels = sorted(profile_labels - previous_profile_labels)
        short_label, short_max = _max_profile_similarity(short, profiles)
        _long_label, long_max = _max_profile_similarity(long, profiles)
        short_change = _cos(short, previous_short)
        cross_scale = _cos(short, long)
        speech = bool(payload.get("speech")) and not bool(payload.get("release_signal"))

        recorded_public = public_by_step.get(step_id)
        fallback_record = recorded_public or recorded_decisions.get(step_id) or record
        wall = float(fallback_record.get("wall_us") or 0) / 1_000_000.0
        sequence = int(fallback_record.get("seq") or record.get("seq") or 0)

        if lease_active and not alias_profile and lease_prototype is not None:
            matches = [
                (label, _cos(lease_prototype, profiles[label])) for label in new_labels
            ]
            if matches:
                label, similarity = max(matches, key=lambda item: item[1])
                if similarity >= lease_config.alias_min_similarity:
                    alias_profile = label
                    alias_count += 1
                    if lease_config.public_identity_mode == "stable_lease":
                        permanent_aliases[label] = lease_public_id
                    diagnostics.append(
                        {
                            "event": "alias",
                            "step_id": step_id,
                            "media_time": media_time,
                            "profile": label,
                            "similarity": similarity,
                        }
                    )

        if lease_active and not speech:
            actions.append(
                (
                    wall,
                    sequence,
                    "live_speaker_clear",
                    _clear_payload(
                        step_id=step_id,
                        speaker=(
                            alias_profile
                            if lease_config.public_identity_mode in {"final_profile", "hidden_until_alias"}
                            and alias_profile
                            else (
                                ""
                                if lease_config.public_identity_mode == "hidden_until_alias"
                                else lease_public_id
                            )
                        ),
                        payload=payload,
                        reason="silence_novelty_lease_death",
                    ),
                )
            )
            diagnostics.append(
                {"event": "silence_death", "step_id": step_id, "media_time": media_time}
            )
            silence_death_count += 1
            lease_active = False
            lease_prototype = None
            alias_profile = ""
            lease_death_streak = 0
            novelty_streak = 0
        elif lease_active:
            lease_similarity = _cos(short, lease_prototype)
            if lease_similarity < lease_config.keep_min_similarity:
                lease_death_streak += 1
            else:
                lease_death_streak = 0
            expired = media_time - lease_started >= lease_config.max_seconds
            if lease_death_streak >= lease_config.death_streak or expired:
                diagnostics.append(
                    {
                        "event": "voice_death" if not expired else "timeout_death",
                        "step_id": step_id,
                        "media_time": media_time,
                        "lease_similarity": lease_similarity,
                    }
                )
                lease_active = False
                lease_prototype = None
                alias_profile = ""
                lease_death_streak = 0
                novelty_streak = 0
            else:
                if (
                    short is not None
                    and lease_prototype is not None
                    and lease_similarity >= lease_config.prototype_update_min_similarity
                ):
                    blended = _unit(
                        (1.0 - lease_config.prototype_update_alpha) * lease_prototype
                        + lease_config.prototype_update_alpha * short
                    )
                    if blended is not None:
                        lease_prototype = blended
                if lease_config.public_identity_mode == "hidden_until_alias" and not alias_profile:
                    actions.append(
                        (
                            wall,
                            sequence,
                            "live_speaker_clear",
                            _clear_payload(
                                step_id=step_id,
                                speaker="",
                                payload=payload,
                                reason="novelty_lease_unbound",
                            ),
                        )
                    )
                else:
                    public_id = (
                        alias_profile
                        if lease_config.public_identity_mode in {"final_profile", "hidden_until_alias"}
                        and alias_profile
                        else lease_public_id
                    )
                    actions.append(
                        (
                            wall,
                            sequence,
                            "live_speaker",
                            _action_payload(
                                step_id=step_id,
                                speaker=public_id,
                                payload=payload,
                                hold_seconds=hold_seconds,
                                source="counterfactual_novelty_lease",
                            ),
                        )
                    )
                previous_short = short if short is not None else previous_short
                previous_profile_labels = profile_labels
                continue

        if not lease_active and speech and short is not None and long is not None:
            eligible = (
                len(profiles) >= lease_config.min_profiles
                and previous_short is not None
                and short_change <= lease_config.change_short_max_similarity
                and short_max <= lease_config.novelty_short_max_profile_similarity
                and long_max <= lease_config.novelty_long_max_profile_similarity
                and cross_scale <= lease_config.change_short_long_max_similarity
            )
            novelty_streak = novelty_streak + 1 if eligible else 0
            if novelty_streak >= lease_config.activation_streak:
                initial = _unit(
                    lease_config.prototype_short_weight * short
                    + (1.0 - lease_config.prototype_short_weight) * long
                )
                if initial is not None:
                    lease_active = True
                    lease_started = media_time
                    lease_prototype = initial
                    lease_death_streak = 0
                    alias_profile = ""
                    activation_count += 1
                    diagnostics.append(
                        {
                            "event": "activate",
                            "step_id": step_id,
                            "media_time": media_time,
                            "short_change": short_change,
                            "short_max_profile_similarity": short_max,
                            "long_max_profile_similarity": long_max,
                            "short_long_similarity": cross_scale,
                            "baseline_speaker": str(
                                (base_decisions.get(step_id) or {}).get("visible_speaker")
                                or ""
                            ),
                        }
                    )
                    if lease_config.public_identity_mode == "hidden_until_alias":
                        actions.append(
                            (
                                wall,
                                sequence,
                                "live_speaker_clear",
                                _clear_payload(
                                    step_id=step_id,
                                    speaker="",
                                    payload=payload,
                                    reason="novelty_lease_activation_unbound",
                                ),
                            )
                        )
                    else:
                        actions.append(
                            (
                                wall,
                                sequence,
                                "live_speaker",
                                _action_payload(
                                    step_id=step_id,
                                    speaker=lease_public_id,
                                    payload=payload,
                                    hold_seconds=hold_seconds,
                                    source="counterfactual_novelty_lease_activation",
                                ),
                            )
                        )
                    previous_short = short
                    previous_profile_labels = profile_labels
                    continue

        base_action = base_by_step.get(step_id)
        if base_action is not None:
            base_wall, base_sequence, event, raw_base_payload = base_action
            output_payload = dict(raw_base_payload)
            for key in ("assigned_speaker", "speaker_id", "internal_speaker_id"):
                value = str(output_payload.get(key) or "")
                if value in permanent_aliases:
                    output_payload[key] = permanent_aliases[value]
            actions.append((base_wall, base_sequence, event, output_payload))

        previous_short = short if short is not None else previous_short
        previous_profile_labels = profile_labels

    actions.sort(key=lambda item: (item[0], item[1]))
    return {
        "actions": actions,
        "diagnostics": diagnostics,
        "activation_count": activation_count,
        "alias_count": alias_count,
        "silence_death_count": silence_death_count,
        "permanent_aliases": permanent_aliases,
    }


def evaluate_one(
    tape_dir: Path,
    canonical_path: Path,
    algorithm_config: dict[str, Any],
    lease_config: LeaseConfig | None,
) -> dict[str, Any]:
    if lease_config is None:
        projection = _cached_base_projection(
            str(Path(tape_dir).resolve()),
            json.dumps(algorithm_config, sort_keys=True, separators=(",", ":")),
        )
        actions = projection["actions"]
        extras = {
            "activation_count": 0,
            "alias_count": 0,
            "silence_death_count": 0,
            "diagnostics": [],
        }
    else:
        projection = project_novelty_lease_actions(
            tape_dir, algorithm_config, lease_config
        )
        actions = projection["actions"]
        extras = projection
    browser = replay_browser_state(tape_dir, replacement_live_actions=actions)
    score = score_browser_live_speaker_samples(
        browser["replayed_samples"],
        list(_cached_canonical(str(Path(canonical_path).resolve()))),
    )
    return {
        "score": float(score["strict_browser_live_score"]),
        "components": {
            key: score[key]
            for key in (
                "correct_live_speaker_coverage",
                "wrong_live_speech_ratio",
                "missing_live_speech_ratio",
                "outside_speech_live_ratio",
                "correct_live_precision_during_speech",
                "correct_live_speaker_f1",
            )
        },
        "activation_count": int(extras["activation_count"]),
        "alias_count": int(extras["alias_count"]),
        "silence_death_count": int(extras["silence_death_count"]),
        "diagnostics": extras["diagnostics"],
    }


def _candidate_grid() -> Iterable[LeaseConfig]:
    # Coarse, intentionally bounded: the mechanism must earn complexity before a fine sweep.
    for values in itertools.product(
        (2, 3, 4),
        (0.35, 0.45, 0.55),
        (0.24, 0.30, 0.36),
        (0.34, 0.42, 0.50),
        (0.50, 0.62, 0.74),
        (0.18, 0.28, 0.38),
        (8.0, 12.0, 20.0),
        (0.20, 0.30, 0.40, 0.50),
    ):
        yield LeaseConfig(
            min_profiles=values[0],
            change_short_max_similarity=values[1],
            novelty_short_max_profile_similarity=values[2],
            novelty_long_max_profile_similarity=values[3],
            change_short_long_max_similarity=values[4],
            keep_min_similarity=values[5],
            max_seconds=values[6],
            alias_min_similarity=values[7],
        )


def _search_configs(profile: str, max_candidates: int) -> list[LeaseConfig]:
    if profile == "broad":
        universe = list(_candidate_grid())
        if max_candidates >= len(universe):
            return universe
        indices = np.linspace(0, len(universe) - 1, max_candidates, dtype=int)
        return [universe[int(index)] for index in indices]

    center = LeaseConfig(
        min_profiles=4,
        change_short_max_similarity=1.0,
        novelty_short_max_profile_similarity=0.24,
        novelty_long_max_profile_similarity=0.34,
        change_short_long_max_similarity=0.74,
        keep_min_similarity=0.38,
        max_seconds=12.0,
        alias_min_similarity=0.30,
    )
    choices: dict[str, tuple[Any, ...]] = {
        "min_profiles": (3, 4, 5, 6),
        "change_short_max_similarity": (0.45, 0.60, 0.75, 0.90, 1.0),
        "novelty_short_max_profile_similarity": (0.18, 0.21, 0.24, 0.27, 0.30),
        "novelty_long_max_profile_similarity": (0.28, 0.31, 0.34, 0.37, 0.40),
        "change_short_long_max_similarity": (0.62, 0.68, 0.74, 0.80, 0.86),
        "keep_min_similarity": (0.30, 0.34, 0.38, 0.42, 0.46),
        "max_seconds": (8.0, 12.0, 16.0, 20.0),
        "alias_min_similarity": (0.20, 0.25, 0.30, 0.35, 0.40),
        "activation_streak": (1, 2, 3),
    }
    configs = [center]
    base = asdict(center)
    for field, values in choices.items():
        for value in values:
            configs.append(LeaseConfig(**{**base, field: value}))
    rng = np.random.default_rng(20260722)
    while len(configs) < max_candidates:
        values = {
            field: options[int(rng.integers(0, len(options)))]
            for field, options in choices.items()
        }
        configs.append(LeaseConfig(**{**base, **values}))
    unique: dict[str, LeaseConfig] = {}
    for config in configs:
        unique[_hash(asdict(config))] = config
    return list(unique.values())[:max_candidates]


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run(
    campaign_root: Path,
    artifact_path: Path,
    *,
    max_candidates: int,
    search_profile: str,
    output: Path,
) -> dict[str, Any]:
    campaign = Path(campaign_root).resolve()
    parity = json.loads((campaign / "baseline_parity_report.json").read_text("utf-8"))
    artifact = json.loads(Path(artifact_path).read_text("utf-8"))
    algorithm_config = dict(artifact["algorithm_config"])
    runs = list(parity["runs"])

    baseline_runs = []
    for item in runs:
        result = evaluate_one(
            Path(item["tape_dir"]),
            Path(item["canonical_path"]),
            algorithm_config,
            None,
        )
        baseline_runs.append({**item, **result})
    baseline_per_video: dict[str, list[float]] = {}
    for item in baseline_runs:
        baseline_per_video.setdefault(str(item["video_id"]), []).append(item["score"])
    baseline_video_scores = {
        video: mean(scores) for video, scores in baseline_per_video.items()
    }
    baseline_macro = mean(baseline_video_scores.values())

    # Cheap representative screen on the two identity-confusion videos.
    screen_runs = []
    for video_id in ("JWS-qfR6K3w", "20v1OxUXcQY"):
        match = next(item for item in runs if item["video_id"] == video_id)
        screen_runs.append(match)
    screened: list[tuple[float, LeaseConfig]] = []
    seen: set[str] = set()
    search_configs = _search_configs(search_profile, max_candidates)
    for config in search_configs:
        key = _hash(asdict(config))
        if key in seen:
            continue
        seen.add(key)
        scores = [
            evaluate_one(
                Path(item["tape_dir"]),
                Path(item["canonical_path"]),
                algorithm_config,
                config,
            )["score"]
            for item in screen_runs
        ]
        screened.append((mean(scores), config))
    screened.sort(key=lambda item: item[0], reverse=True)

    # Strict seven-video scoring for a small finalist set plus explicit ablations.
    finalists: list[tuple[str, LeaseConfig]] = []
    for index, (_score, config) in enumerate(screened[:12], 1):
        finalists.append((f"screen_finalist_{index:02d}", config))
    if finalists:
        best = finalists[0][1]
        finalists.extend(
            [
                ("ablation_no_change_gate", LeaseConfig(**{**asdict(best), "change_short_max_similarity": 1.0})),
                ("ablation_no_cross_scale_gate", LeaseConfig(**{**asdict(best), "change_short_long_max_similarity": 1.0})),
                ("ablation_no_profile_novelty_gate", LeaseConfig(**{**asdict(best), "novelty_short_max_profile_similarity": 1.0, "novelty_long_max_profile_similarity": 1.0})),
                ("ablation_no_alias", LeaseConfig(**{**asdict(best), "alias_min_similarity": 2.0})),
                ("ablation_visible_stable_lease", LeaseConfig(**{**asdict(best), "public_identity_mode": "stable_lease"})),
                ("ablation_visible_final_profile", LeaseConfig(**{**asdict(best), "public_identity_mode": "final_profile"})),
                ("ablation_short_only_prototype", LeaseConfig(**{**asdict(best), "prototype_short_weight": 1.0})),
                ("ablation_no_prototype_update", LeaseConfig(**{**asdict(best), "prototype_update_alpha": 0.0})),
            ]
        )

    results = []
    for name, config in finalists:
        run_rows = []
        for item in runs:
            evaluated = evaluate_one(
                Path(item["tape_dir"]),
                Path(item["canonical_path"]),
                algorithm_config,
                config,
            )
            run_rows.append(
                {
                    "video_id": item["video_id"],
                    "run_id": item["run_id"],
                    **evaluated,
                }
            )
        per_video: dict[str, list[float]] = {}
        for row in run_rows:
            per_video.setdefault(row["video_id"], []).append(row["score"])
        video_scores = {video: mean(scores) for video, scores in per_video.items()}
        macro = mean(video_scores.values())
        results.append(
            {
                "name": name,
                "config": asdict(config),
                "macro_score": macro,
                "delta_vs_baseline": macro - baseline_macro,
                "per_video": video_scores,
                "per_video_delta": {
                    video: video_scores[video] - baseline_video_scores[video]
                    for video in video_scores
                },
                "activation_count": sum(row["activation_count"] for row in run_rows),
                "alias_count": sum(row["alias_count"] for row in run_rows),
                "silence_death_count": sum(
                    row["silence_death_count"] for row in run_rows
                ),
                "runs": run_rows,
            }
        )
    results.sort(key=lambda item: item["macro_score"], reverse=True)
    report = {
        "contract_id": CONTRACT_ID,
        "status": "REPLAY_ONLY_RESEARCH",
        "production_promotion_eligible": False,
        "live_validated": False,
        "hypothesis": "A causal, change-point-gated single-slot novelty lease can avoid confidently displaying an old speaker for a truly new voice, then atomically bind the lease identity to the first matching newly published final centroid.",
        "inference_changes": "none; frozen existing 0.7/1.5 s speechbrain_resnet vectors only",
        "campaign_root": str(campaign),
        "artifact": str(Path(artifact_path).resolve()),
        "algorithm_config_sha256": _hash(algorithm_config),
        "baseline": {
            "macro_score": baseline_macro,
            "per_video": baseline_video_scores,
            "runs": baseline_runs,
        },
        "screen_candidate_count": len(screened),
        "search_profile": search_profile,
        "strict_finalist_count": len(results),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--search-profile", choices=("broad", "local"), default="broad")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.campaign_root,
        args.artifact,
        max_candidates=max(1, args.max_candidates),
        search_profile=args.search_profile,
        output=args.output,
    )
    print(json.dumps({
        "baseline": report["baseline"]["macro_score"],
        "best": report["results"][0] if report["results"] else None,
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
