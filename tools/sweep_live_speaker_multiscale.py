"""Deterministic cached-data search for causal multi-scale live assignment.

The two development videos are the only inputs used to rank or expand search
candidates.  The 20v video is opened only after the search has stopped and is
used as a promotion gate.  The known JWS holdout is rejected explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from optimize_live_speaker_replay import Dataset, _trace_hash
from window.live_speaker_algorithm import ALGORITHM_ID, LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import (
    SCORER_ID,
    aggregate_video_scores,
    score_live_speaker_decisions,
)
from window.live_speaker_multiscale import (
    MULTISCALE_ALGORITHM_ID,
    MultiScaleTrackerConfig,
    replay_cached_multiscale_windows,
)
from window.live_speaker_replay import replay_cached_live_windows_dual


OPTIMIZER_ID = "causal_live_speaker_multiscale_sweep_v5"
DEVELOPMENT_VIDEOS = frozenset({"Dd7FixvoKBw", "DsyfYJ5Ou3g"})
VALIDATION_VIDEO = "20v1OxUXcQY"
KNOWN_HOLDOUT = "JWS-qfR6K3w"
HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE = 2
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _fresh_live_cost_diagnostics(
    windows: Sequence[float],
    *,
    max_windows_per_probe: int,
    cache_hop_seconds: float,
    production_probe_interval_seconds: float | None,
) -> dict[str, Any]:
    """Describe compute eligibility without treating cache ticks as live probes."""

    window_count = len(tuple(windows))
    within_budget = window_count <= int(max_windows_per_probe)
    reason = None
    if not within_budget:
        reason = (
            f"candidate requires {window_count} freshly computed windows per probe; "
            f"the promotion budget allows at most {int(max_windows_per_probe)}"
        )
    return {
        "fresh_windows_computed_per_probe": window_count,
        "configured_max_fresh_windows_per_probe": int(max_windows_per_probe),
        "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
        "within_fresh_live_window_budget": within_budget,
        "research_only": not within_budget,
        "research_only_reason": reason,
        "cache_hop_seconds": float(cache_hop_seconds),
        "production_probe_interval_seconds": (
            float(production_probe_interval_seconds)
            if production_probe_interval_seconds is not None
            else None
        ),
        "cache_grid_is_live_probe_cadence": False,
        "cache_grid_role": "offline_embedding_lookup_and_scoring_grid",
        "fresh_live_cadence_verified": False,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _provider_spec(spec: dict[str, Any], champion: dict[str, Any]) -> str:
    direct = champion.get("provider")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    weights = spec["baseline"]["provider_weights"]
    return "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in weights.items()
        if float(weight) > 0.0
    )


def _champion_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "short_window_seconds",
        "long_window_seconds",
        "long_weight",
        "algorithm_config",
    }
    if required.issubset(payload):
        return payload
    for key in ("winner", "best", "candidate", "champion"):
        value = payload.get(key)
        if isinstance(value, dict) and required.issubset(value):
            return value
    raise ValueError(
        "Champion JSON must contain short_window_seconds, long_window_seconds, "
        "long_weight, and algorithm_config (directly or under winner/best/candidate)."
    )


def _declared_champion_score(payload: dict[str, Any], candidate: dict[str, Any]) -> Any:
    for source in (candidate, payload):
        for key in (
            "cached_score",
            "fresh_live_score",
            "candidate_score",
            "score",
        ):
            if key in source:
                return source[key]
    return None


def _split_guard(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    search = [str(value) for value in spec["split"]["search"]]
    validation = [str(value) for value in spec["split"]["validation"]]
    if len(search) != 2 or set(search) != DEVELOPMENT_VIDEOS:
        raise ValueError(
            "This runner is intentionally sealed to the two development videos: "
            f"{sorted(DEVELOPMENT_VIDEOS)}; received {search}."
        )
    if validation != [VALIDATION_VIDEO]:
        raise ValueError(
            f"This runner requires validation=[{VALIDATION_VIDEO!r}]; received {validation}."
        )
    if KNOWN_HOLDOUT in search or KNOWN_HOLDOUT in validation:
        raise ValueError(f"The known holdout {KNOWN_HOLDOUT} may not be scored by this runner")
    return search, validation


def _score_dual_baseline(
    dataset: Dataset,
    videos: Sequence[str],
    *,
    short_window: float,
    long_window: float,
    long_weight: float,
    config: LiveSpeakerAlgorithmConfig,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    trace_hashes: dict[str, str] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id)
        decisions = replay_cached_live_windows_dual(
            dataset.block(video_id, short_window),
            dataset.block(video_id, long_window),
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            long_weight=long_weight,
            config=config,
        )
        per_video[video_id] = score_live_speaker_decisions(
            decisions, inputs["canonical"], inputs["profiles"]
        )
        trace_hashes[video_id] = _trace_hash(decisions)
    return {
        "aggregate": aggregate_video_scores(per_video.values()),
        "per_video": per_video,
        "trace_hashes": trace_hashes,
    }


def _score_multiscale(
    dataset: Dataset,
    videos: Sequence[str],
    windows: Sequence[float],
    config: MultiScaleTrackerConfig,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id)
        decisions = replay_cached_multiscale_windows(
            [dataset.block(video_id, window) for window in windows],
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            config=config,
        )
        per_video[video_id] = _compact_video_score(
            score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            )
        )
    return {
        "aggregate": aggregate_video_scores(per_video.values()),
        "per_video": per_video,
    }


def _compact_video_score(score: dict[str, Any]) -> dict[str, Any]:
    """Keep every aggregate metric while omitting repeated per-turn event arrays."""

    compact = {
        key: value
        for key, value in score.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    for key in ("sampled_playback_seconds", "speaker_map"):
        if key in score:
            compact[key] = score[key]
    turn_latency = score.get("turn_latency")
    if isinstance(turn_latency, dict):
        compact["turn_latency"] = {
            key: value for key, value in turn_latency.items() if key != "turns"
        }
    release = score.get("release")
    if isinstance(release, dict):
        compact["release"] = {
            key: value for key, value in release.items() if key != "events"
        }
    availability = score.get("profile_availability")
    if isinstance(availability, dict):
        compact["profile_availability"] = {
            "counts": dict(availability.get("counts") or {})
        }
    return compact


def _candidate_id(windows: Sequence[float], config: MultiScaleTrackerConfig) -> str:
    return _stable_id(
        {
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": MULTISCALE_ALGORITHM_ID,
            "scorer_id": SCORER_ID,
            "windows_seconds": [round(float(value), 3) for value in windows],
            "config": asdict(config),
        }
    )


def _quality(row: dict[str, Any]) -> tuple[float, float, float, float]:
    scores = list(row["search_per_video"].values())
    wrong = [float(item["wrong_live_speech_ratio"]) for item in scores]
    strict = [float(item["strict_browser_live_score"]) for item in scores]
    return (
        float(row["search"]["global_score"]),
        -sum(wrong) / len(wrong),
        -max(wrong),
        min(strict),
    )


def _rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_quality, reverse=True)


def _top_diverse(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Prefer one strong parent per scale set, then fill by search quality."""

    ranked = _rank(rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_windows: set[tuple[float, ...]] = set()
    for row in ranked:
        signature = tuple(float(value) for value in row["windows_seconds"])
        if signature in seen_windows:
            continue
        seen_windows.add(signature)
        selected.append(row)
        selected_ids.add(str(row["candidate_id"]))
        if len(selected) >= limit:
            return selected
    for row in ranked:
        candidate_id = str(row["candidate_id"])
        if candidate_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(candidate_id)
        if len(selected) >= limit:
            break
    return selected


def _windows_from_spec(spec: dict[str, Any]) -> list[tuple[float, ...]]:
    available = {
        round(float(value), 3)
        for value in spec["dense_corpus_expectation"]["window_lengths_seconds"]
    }
    requested = [
        (0.7, 2.8),
        (0.8, 2.4),
        (0.8, 2.8),
        (0.8, 3.0),
        (0.9, 2.8),
        (0.7, 1.2, 2.8),
        (0.8, 1.4, 2.8),
        (0.8, 1.8, 2.8),
        (0.9, 1.5, 2.9),
        (0.7, 0.9, 1.2, 1.8, 2.8),
        (0.8, 1.0, 1.4, 2.0, 2.8),
    ]
    missing = sorted({value for windows in requested for value in windows} - available)
    if missing:
        raise ValueError(f"Dense corpus specification is missing required windows: {missing}")
    return requested


def _scale_weight_profiles(windows: Sequence[float]) -> list[tuple[float, ...]]:
    """Exercise balanced, fast-reacting, and stable-context score fusion."""

    count = len(windows)
    equal = tuple(1.0 / count for _value in windows)
    if count == 2:
        profiles = (equal, (0.75, 0.25), (0.35, 0.65))
    elif count == 3:
        profiles = (equal, (0.60, 0.25, 0.15), (0.20, 0.30, 0.50))
    elif count == 5:
        profiles = (
            equal,
            (0.35, 0.25, 0.18, 0.12, 0.10),
            (0.10, 0.12, 0.18, 0.25, 0.35),
        )
    else:
        raise ValueError(f"no scale-weight profiles defined for {count} windows")
    return list(dict.fromkeys(profiles))


def _validation_finalists(
    rows: Iterable[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    ranked = _rank(rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for family in (
        "score_fusion",
        "consensus",
        "crossover_only",
        "history_only",
        "consensus_crossover_history",
        "transition_abstention",
        "duration_matched_profiles",
        "online_profiles",
        "state_machine",
        "state_machine_embedding_change_point",
        "transition_speech_gate_only",
        "transition_speech_gate_state_machine",
        "known_crossover_transition",
        "profile_maturity_gate",
    ):
        match = next((row for row in ranked if row["ablation_family"] == family), None)
        if match is not None and str(match["candidate_id"]) not in selected_ids:
            selected.append(match)
            selected_ids.add(str(match["candidate_id"]))
    for row in ranked:
        if len(selected) >= limit:
            break
        candidate_id = str(row["candidate_id"])
        if candidate_id not in selected_ids:
            selected.append(row)
            selected_ids.add(candidate_id)
    return selected[:limit]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded multi-scale/history search on Dd+Dsy, then gate finalists "
            "once on 20v without touching JWS."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=3300)
    parser.add_argument("--validation-reserve-seconds", type=int, default=300)
    parser.add_argument("--max-search-candidates", type=int, default=1000)
    parser.add_argument(
        "--max-state-machine-candidates",
        type=int,
        default=150,
        help=(
            "Maximum combined Stage-7 candidates for the explicit state machine "
            "and its embedding-change ablation. The built-in focused grid produces "
            "at most 132 candidates."
        ),
    )
    parser.add_argument(
        "--max-speech-gate-candidates",
        type=int,
        default=80,
        help=(
            "Maximum combined Stage-8 candidates for speech-gate-only and "
            "speech-gate-plus-state-machine ablations. The built-in focused "
            "grid produces at most 60 candidates."
        ),
    )
    parser.add_argument("--top-per-stage", type=int, default=10)
    parser.add_argument("--validation-finalists", type=int, default=18)
    parser.add_argument("--minimum-search-improvement", type=float, default=1e-6)
    parser.add_argument("--validation-score-tolerance", type=float, default=0.005)
    parser.add_argument("--wrong-ratio-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--max-fresh-live-windows-per-probe",
        type=int,
        default=HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
        help=(
            "Promotion-time compute budget for freshly embedded audio windows at one "
            "real live probe. Values above the hard limit of 2 are rejected; cached "
            "candidates using more windows remain research-only."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    if args.max_search_candidates < 1:
        raise ValueError("--max-search-candidates must be positive")
    if not 1 <= args.max_state_machine_candidates <= 180:
        raise ValueError("--max-state-machine-candidates must be in [1, 180]")
    if not 1 <= args.max_speech_gate_candidates <= 100:
        raise ValueError("--max-speech-gate-candidates must be in [1, 100]")
    if args.top_per_stage < 1 or args.validation_finalists < 1:
        raise ValueError("stage and validation finalist counts must be positive")
    if args.max_fresh_live_windows_per_probe < 1:
        raise ValueError("--max-fresh-live-windows-per-probe must be positive")
    if args.max_fresh_live_windows_per_probe > HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE:
        raise ValueError(
            "--max-fresh-live-windows-per-probe cannot exceed the hard real-time "
            f"promotion limit of {HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE}"
        )

    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    reserve = max(0, min(int(args.validation_reserve_seconds), int(args.budget_seconds) - 1))
    search_deadline = deadline - reserve
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_payload = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    search_videos, validation_videos = _split_guard(spec)
    scored_videos = search_videos + validation_videos
    champion = _champion_candidate(champion_payload)
    provider = _provider_spec(spec, champion_payload)
    baseline_short = float(champion["short_window_seconds"])
    baseline_long = float(champion["long_window_seconds"])
    baseline_weight = float(champion["long_weight"])
    baseline_config = LiveSpeakerAlgorithmConfig(**champion["algorithm_config"])
    windows_to_search = _windows_from_spec(spec)
    cache_hop_seconds = float(
        spec.get("dense_corpus_expectation", {}).get("hop_seconds", 0.2)
    )
    raw_probe_interval = spec.get("baseline", {}).get(
        "normal_probe_interval_seconds"
    )
    production_probe_interval_seconds = (
        float(raw_probe_interval) if raw_probe_interval is not None else None
    )
    max_fresh_live_windows_per_probe = int(
        args.max_fresh_live_windows_per_probe
    )

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    validation_path = run_dir / "validation_trials.jsonl"
    if not args.resume and (trials_path.exists() or validation_path.exists()):
        raise FileExistsError(
            f"Run artifacts already exist in {run_dir}; pass --resume or choose a new --run-dir"
        )

    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider)
    run_identity = _stable_id(
        {
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": MULTISCALE_ALGORITHM_ID,
            "baseline_algorithm_id": ALGORITHM_ID,
            "scorer_id": SCORER_ID,
            "spec": spec,
            "champion": champion,
            "provider": provider,
            "corpus_root": str(args.corpus_root.resolve()),
            "input_root": str(args.input_root.resolve()),
            "fresh_live_cost_policy": {
                "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
                "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
                "cache_grid_is_live_probe_cadence": False,
            },
            "state_machine_search_policy": {
                "version": 1,
                "max_candidates": int(args.max_state_machine_candidates),
                "parent_families": [
                    "score_fusion",
                    "consensus",
                    "consensus_crossover_history",
                ],
                "max_parent_windows": max_fresh_live_windows_per_probe,
            },
            "speech_gate_search_policy": {
                "version": 1,
                "max_candidates": int(args.max_speech_gate_candidates),
                "parent_priority": [
                    "consensus_crossover_history",
                    "consensus",
                    "score_fusion",
                ],
                "max_parent_windows": max_fresh_live_windows_per_probe,
            },
        }
    )
    existing_run_path = run_dir / "run.json"
    if args.resume and existing_run_path.is_file():
        existing_run = json.loads(existing_run_path.read_text(encoding="utf-8-sig"))
        if existing_run.get("run_identity") != run_identity:
            raise RuntimeError("Cannot resume: run identity differs from the existing run.json")
    _atomic_json(
        existing_run_path,
        {
            "schema_version": 1,
            "optimizer_id": OPTIMIZER_ID,
            "run_identity": run_identity,
            "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
            "baseline_algorithm_id": ALGORITHM_ID,
            "scorer_id": SCORER_ID,
            "provider": provider,
            "search_videos": search_videos,
            "validation_videos": validation_videos,
            "known_holdout_excluded": [KNOWN_HOLDOUT],
            "sealed_holdout_opened": False,
            "selection_policy": "rank_on_development_then_validation_gate",
            "budget_seconds": int(args.budget_seconds),
            "validation_reserve_seconds": reserve,
            "max_search_candidates": int(args.max_search_candidates),
            "max_state_machine_candidates": int(args.max_state_machine_candidates),
            "max_speech_gate_candidates": int(args.max_speech_gate_candidates),
            "fresh_live_cost_policy": {
                "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
                "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
                "cache_hop_seconds": cache_hop_seconds,
                "production_probe_interval_seconds": production_probe_interval_seconds,
                "cache_grid_is_live_probe_cadence": False,
                "policy": (
                    "more-than-budget windows may be scored only as research; they "
                    "cannot enter fresh-live verification or champion selection"
                ),
            },
        },
    )

    baseline_search_first = _score_dual_baseline(
        dataset,
        search_videos,
        short_window=baseline_short,
        long_window=baseline_long,
        long_weight=baseline_weight,
        config=baseline_config,
    )
    baseline_search_second = _score_dual_baseline(
        dataset,
        search_videos,
        short_window=baseline_short,
        long_window=baseline_long,
        long_weight=baseline_weight,
        config=baseline_config,
    )
    baseline_identical = _stable_json(baseline_search_first) == _stable_json(
        baseline_search_second
    )
    baseline = {
        "status": (
            "SEARCH_REPRODUCED_TWICE__VALIDATION_STILL_SEALED"
            if baseline_identical
            else "MISMATCH"
        ),
        "short_window_seconds": baseline_short,
        "long_window_seconds": baseline_long,
        "long_weight": baseline_weight,
        "algorithm_config": asdict(baseline_config),
        "declared_champion_score": _declared_champion_score(champion_payload, champion),
        "aggregate": None,
        "search": baseline_search_first["aggregate"],
        "validation": None,
        "per_video": baseline_search_first["per_video"],
        "trace_hashes": baseline_search_first["trace_hashes"],
        "fresh_live_cost": _fresh_live_cost_diagnostics(
            (baseline_short, baseline_long),
            max_windows_per_probe=max_fresh_live_windows_per_probe,
            cache_hop_seconds=cache_hop_seconds,
            production_probe_interval_seconds=production_probe_interval_seconds,
        ),
    }
    _atomic_json(run_dir / "baseline_reproduction.json", baseline)
    if not baseline_identical:
        raise RuntimeError("Production dual-window search baseline did not reproduce exactly twice")

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for raw in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                completed[str(row["candidate_id"])] = row
    validated: dict[str, dict[str, Any]] = {}
    if args.resume and validation_path.is_file():
        for raw in validation_path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                validated[str(row["candidate_id"])] = row
    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    def write_progress(phase: str) -> None:
        best = _rank(completed.values())[0] if completed else None
        promotable_rows = [
            row
            for row in completed.values()
            if not bool(row.get("research_only", False))
        ]
        best_promotable = _rank(promotable_rows)[0] if promotable_rows else None
        research_only_count = sum(
            bool(row.get("research_only", False)) for row in completed.values()
        )
        elapsed = time.monotonic() - started
        _atomic_json(
            run_dir / "progress.json",
            {
                "phase": phase,
                "evaluated_search_candidates": len(completed),
                "validated_finalists": len(validated),
                "phase_counts": phase_counts,
                "elapsed_seconds": round(elapsed, 6),
                "budget_seconds": int(args.budget_seconds),
                "candidate_progress_percent": round(
                    min(100.0, 100.0 * len(completed) / args.max_search_candidates), 2
                ),
                "wall_progress_percent": round(
                    min(100.0, 100.0 * elapsed / max(1, args.budget_seconds)), 2
                ),
                "best_search_score": best["search"]["global_score"] if best else None,
                "best_candidate_id": best["candidate_id"] if best else None,
                "best_promotable_search_score": (
                    best_promotable["search"]["global_score"]
                    if best_promotable
                    else None
                ),
                "best_promotable_candidate_id": (
                    best_promotable["candidate_id"] if best_promotable else None
                ),
                "promotable_search_candidates": len(promotable_rows),
                "research_only_search_candidates": research_only_count,
                "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
                "cache_grid_is_live_probe_cadence": False,
                "sealed_holdout_opened": False,
            },
        )

    def evaluate_search(
        windows: Sequence[float],
        config: MultiScaleTrackerConfig,
        phase: str,
        family: str,
        hypothesis: str,
        parent_candidate_id: str | None = None,
    ) -> dict[str, Any] | None:
        # Bind weights to named durations so the core can drop not-yet-causal
        # long windows while retaining the correct weights for available scales.
        bound_windows = tuple(float(value) for value in windows)
        if not config.scale_weights:
            config = replace(
                config,
                scale_windows=bound_windows,
                scale_weights=tuple(1.0 for _value in bound_windows),
            )
        elif not config.scale_windows:
            config = replace(config, scale_windows=bound_windows)
        elif tuple(float(value) for value in config.scale_windows) != bound_windows:
            raise ValueError("config.scale_windows must match candidate windows")
        candidate_id = _candidate_id(windows, config)
        previous = completed.get(candidate_id)
        if previous is not None:
            return previous
        if (
            _STOP
            or time.monotonic() >= search_deadline
            or len(completed) >= int(args.max_search_candidates)
        ):
            return None
        scored = _score_multiscale(dataset, search_videos, windows, config)
        fresh_live_cost = _fresh_live_cost_diagnostics(
            windows,
            max_windows_per_probe=max_fresh_live_windows_per_probe,
            cache_hop_seconds=cache_hop_seconds,
            production_probe_interval_seconds=production_probe_interval_seconds,
        )
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "ablation_family": family,
            "hypothesis": hypothesis,
            "parent_candidate_id": parent_candidate_id,
            "windows_seconds": [round(float(value), 3) for value in windows],
            "algorithm_config": asdict(config),
            "search": scored["aggregate"],
            "search_per_video": scored["per_video"],
            "search_score_delta_vs_production": round(
                float(scored["aggregate"]["global_score"])
                - float(baseline["search"]["global_score"]),
                6,
            ),
            "validation_opened": False,
            "fresh_live_cost": fresh_live_cost,
            "research_only": fresh_live_cost["research_only"],
            "research_only_reason": fresh_live_cost["research_only_reason"],
            "eligible_for_fresh_live_verification": False,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[candidate_id] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        _append_jsonl(trials_path, row)
        write_progress(phase)
        return row

    # Candidate evaluation binds equal weights to the exact named durations.
    # The core can therefore adapt those weights when early long windows are
    # absent under the corpus' full-window-only policy.
    base_multiscale = MultiScaleTrackerConfig(
        scale_weights=(),
        min_similarity=float(baseline_config.min_similarity),
        min_margin=float(baseline_config.min_margin),
        unknown_release_count=int(baseline_config.unknown_release_count),
        silence_release_count=int(baseline_config.silence_release_count),
    )

    # Stage 1: score fusion only.  No consensus, crossover, history, or online
    # profiles are active, so every later family has an exact ablation parent.
    fusion_rows: list[dict[str, Any]] = []
    threshold_pairs = (
        (0.25, 0.00),
        (0.30, 0.00),
        (0.30, 0.03),
        (0.35, 0.03),
        (0.35, 0.05),
        (0.35, 0.08),
        (0.40, 0.05),
        (0.45, 0.08),
    )
    for windows in windows_to_search:
        for scale_weights in _scale_weight_profiles(windows):
            for min_similarity, min_margin in threshold_pairs:
                config = replace(
                    base_multiscale,
                    scale_windows=tuple(float(value) for value in windows),
                    scale_weights=scale_weights,
                    min_similarity=min_similarity,
                    min_margin=min_margin,
                    acquire_scale_agreement=1,
                    enable_consensus=False,
                    enable_crossover=False,
                    enable_history=False,
                    enable_online_profiles=False,
                )
                row = evaluate_search(
                    windows,
                    config,
                    "STAGE_1_SCORE_FUSION",
                    "score_fusion",
                    "Fuse independent per-scale similarities with balanced, fast-heavy, or stable-heavy weights.",
                )
                if row is not None:
                    fusion_rows.append(row)
                if row is None and (
                    _STOP
                    or time.monotonic() >= search_deadline
                    or len(completed) >= args.max_search_candidates
                ):
                    break
            if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
                break
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 2: require agreement across independent scales before switching.
    consensus_rows: list[dict[str, Any]] = []
    for parent in _top_diverse(fusion_rows, int(args.top_per_stage)):
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        max_agreement = min(3, len(windows))
        variants = (
            (2, 0.00, 1),
            (2, 0.03, 1),
            (2, 0.06, 1),
            (2, 0.10, 1),
            (max_agreement, 0.03, 1),
            (max_agreement, 0.06, min(2, len(windows))),
        )
        for agreement, advantage, acquire in variants:
            config = replace(
                source,
                min_scale_agreement=min(agreement, len(windows)),
                acquire_scale_agreement=min(acquire, len(windows)),
                consensus_advantage=advantage,
                enable_consensus=True,
                enable_crossover=False,
                enable_history=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_2_CONSENSUS",
                "consensus",
                "Switch only when multiple independently scored scales agree.",
            )
            if row is not None:
                consensus_rows.append(row)
            if row is None and (
                _STOP
                or time.monotonic() >= search_deadline
                or len(completed) >= args.max_search_candidates
            ):
                break
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 3: exact crossover-only, history-only, and combined ablations.
    temporal_rows: list[dict[str, Any]] = []
    temporal_parents = _top_diverse(
        list(fusion_rows) + list(consensus_rows), int(args.top_per_stage)
    )
    for parent in temporal_parents:
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for short_advantage, scale_gap, crossover_required in (
            (0.03, 0.05, 1),
            (0.03, 0.08, 2),
            (0.05, 0.10, 2),
            (0.05, 0.10, 3),
            (0.08, 0.12, 2),
            (0.10, 0.18, 3),
        ):
            config = replace(
                source,
                enable_consensus=False,
                enable_crossover=True,
                enable_history=False,
                crossover_short_advantage=short_advantage,
                crossover_scale_gap=scale_gap,
                crossover_required=crossover_required,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_3_CROSSOVER",
                "crossover_only",
                "Treat a short-window lead over stale long context as a turn-change signal.",
            )
            if row is not None:
                temporal_rows.append(row)

        for history_size, history_required, advantage, short_weight, statistic in (
            (3, 2, 0.02, 1.0, "mean"),
            (3, 2, 0.02, 0.5, "mean"),
            (3, 2, 0.02, 0.0, "mean"),
            (3, 2, 0.03, 0.5, "median"),
            (3, 3, 0.02, 0.5, "median"),
            (5, 2, 0.05, 0.5, "mean"),
            (5, 3, 0.03, 0.5, "mean"),
            (5, 3, 0.03, 0.0, "median"),
            (5, 4, 0.02, 0.5, "median"),
        ):
            config = replace(
                source,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=True,
                history_size=history_size,
                history_required=history_required,
                history_advantage=advantage,
                history_short_weight=short_weight,
                history_statistic=statistic,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_3_HISTORY",
                "history_only",
                "Require a challenger similarity advantage across three to five probes.",
            )
            if row is not None:
                temporal_rows.append(row)

        for short_advantage, scale_gap, crossover_required, history_size, history_required, history_advantage, short_weight, statistic in (
            (0.03, 0.08, 1, 3, 2, 0.03, 0.5, "mean"),
            (0.05, 0.10, 2, 3, 2, 0.05, 1.0, "mean"),
            (0.05, 0.12, 2, 5, 3, 0.03, 0.5, "mean"),
            (0.05, 0.12, 2, 5, 3, 0.03, 0.5, "median"),
            (0.08, 0.15, 3, 5, 3, 0.05, 0.0, "median"),
        ):
            config = replace(
                source,
                min_scale_agreement=min(2, len(windows)),
                enable_consensus=True,
                enable_crossover=True,
                enable_history=True,
                crossover_short_advantage=short_advantage,
                crossover_scale_gap=scale_gap,
                crossover_required=crossover_required,
                history_size=history_size,
                history_required=history_required,
                history_advantage=history_advantage,
                history_short_weight=short_weight,
                history_statistic=statistic,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_3_COMBINED",
                "consensus_crossover_history",
                "Combine stable multi-scale consensus with fast crossover and bounded history.",
            )
            if row is not None:
                temporal_rows.append(row)
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 4: when fast and slow evidence contradict each other at a likely
    # boundary, show OFF instead of knowingly extending a probably-wrong label.
    transition_rows: list[dict[str, Any]] = []
    transition_parents = _top_diverse(
        list(temporal_rows) + list(consensus_rows), int(args.top_per_stage)
    )
    for parent in transition_parents:
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for short_advantage, scale_gap, clear_required, acquire_agreement, incumbent_max, incumbent_drop, incumbent_clear in (
            (0.03, 0.05, 1, 2, 0.20, 0.10, 1),
            (0.03, 0.05, 1, 2, 0.25, 0.10, 1),
            (0.03, 0.05, 1, 2, 0.30, 0.10, 1),
            (0.03, 0.05, 1, 2, 0.35, 0.10, 1),
            (0.03, 0.05, 1, 2, 0.40, 0.10, 1),
            (0.03, 0.05, 1, 2, 0.35, 0.05, 1),
            (0.03, 0.05, 1, 2, 0.35, 0.15, 1),
            (0.03, 0.05, 1, 2, 0.35, 0.20, 1),
            (0.03, 0.05, 1, 2, 0.35, 0.10, 2),
            (0.05, 0.10, 2, 3, 0.35, 0.12, 1),
        ):
            config = replace(
                source,
                enable_transition_abstention=True,
                transition_short_advantage=short_advantage,
                transition_scale_gap=scale_gap,
                transition_clear_required=clear_required,
                transition_min_valid_scales=2,
                transition_incumbent_max_similarity=incumbent_max,
                transition_incumbent_drop=incumbent_drop,
                transition_incumbent_clear_required=incumbent_clear,
                acquire_scale_agreement=min(acquire_agreement, len(windows)),
                enable_online_profiles=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_4_TRANSITION_ABSTENTION",
                "transition_abstention",
                "Clear to OFF during fast/slow identity disagreement, then reacquire only after scale agreement.",
            )
            if row is not None:
                transition_rows.append(row)
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 5: use only already-computed probes that lie wholly inside a final
    # sentence to create causal, duration-matched identity prototypes.
    duration_rows: list[dict[str, Any]] = []
    duration_parents = _top_diverse(
        list(transition_rows) + list(temporal_rows) + list(consensus_rows),
        min(10, int(args.top_per_stage)),
    )
    duration_variants = (
        (0.25, 0.20, 2, 0.35, 0.05),
        (0.50, 0.25, 2, 0.45, 0.05),
        (0.75, 0.25, 2, 0.45, 0.05),
        (1.00, 0.25, 2, 0.45, 0.05),
        (0.50, 0.10, 1, 0.35, 0.00),
        (0.50, 0.25, 2, 0.60, 0.10),
    )
    for parent in duration_parents:
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for score_weight, update_alpha, min_windows, cohesion, guard in duration_variants:
            config = replace(
                source,
                enable_duration_matched_profiles=True,
                duration_profile_score_weight=score_weight,
                duration_profile_update_alpha=update_alpha,
                duration_profile_min_windows=min_windows,
                duration_profile_min_cohesion=cohesion,
                duration_profile_guard_seconds=guard,
                enable_online_profiles=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_5_DURATION_MATCHED_PROFILES",
                "duration_matched_profiles",
                "Seed guarded per-duration profiles only from past probes inside finalized sentence bounds.",
            )
            if row is not None:
                duration_rows.append(row)
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 6: let strong temporal trackers form trusted provisional profiles
    # before the first finalized-sentence profile exists.
    online_rows: list[dict[str, Any]] = []
    online_parents = _top_diverse(
        list(duration_rows) + list(transition_rows) + list(temporal_rows) + list(consensus_rows) + list(fusion_rows),
        min(8, int(args.top_per_stage)),
    )
    provisional_variants = (
        (0.20, 2, 0.60, 0.42),
        (0.28, 2, 0.60, 0.42),
        (0.35, 2, 0.60, 0.42),
        (0.28, 1, 0.60, 0.42),
        (0.28, 2, 0.50, 0.35),
        (0.35, 2, 0.50, 0.50),
    )
    for parent in online_parents:
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for max_existing, confirm_count, consistency, merge_similarity in provisional_variants:
            config = replace(
                source,
                enable_online_profiles=True,
                provisional_first_immediate=True,
                provisional_max_existing_similarity=max_existing,
                provisional_confirm_count=confirm_count,
                provisional_scale_consistency=consistency,
                official_merge_similarity=merge_similarity,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_6_ONLINE_PROFILES",
                "online_profiles",
                "Create high-confidence causal profiles before finalized sentences arrive.",
            )
            if row is not None:
                online_rows.append(row)
            if row is None and (
                _STOP
                or time.monotonic() >= search_deadline
                or len(completed) >= args.max_search_candidates
            ):
                break
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= args.max_search_candidates:
            break

    # Stage 7a: exercise the explicit OFF/STABLE/TRANSITION state machine
    # without changing embeddings or adding another identity source.  Seeds are
    # deliberately limited to good score-fusion, consensus, and combined
    # temporal parents that already satisfy the <=2 fresh-window budget.
    state_machine_seed_parents: list[dict[str, Any]] = []
    state_machine_seed_ids: set[str] = set()
    for family, source_rows in (
        ("score_fusion", fusion_rows),
        ("consensus", consensus_rows),
        ("consensus_crossover_history", temporal_rows),
    ):
        family_rows = [
            row
            for row in source_rows
            if row["ablation_family"] == family
            and not bool(row.get("research_only", False))
            and len(row["windows_seconds"]) <= max_fresh_live_windows_per_probe
        ]
        for row in _rank(family_rows)[:2]:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in state_machine_seed_ids:
                state_machine_seed_parents.append(row)
                state_machine_seed_ids.add(candidate_id)

    state_machine_rows: list[dict[str, Any]] = []
    state_machine_embedding_rows: list[dict[str, Any]] = []
    state_machine_cap = int(args.max_state_machine_candidates)

    def state_machine_budget_open() -> bool:
        return (
            not _STOP
            and time.monotonic() < search_deadline
            and len(completed) < int(args.max_search_candidates)
            and len(state_machine_rows) + len(state_machine_embedding_rows)
            < state_machine_cap
        )

    state_history_profiles = ((3, 2), (5, 2), (5, 3))
    state_threshold_profiles = ((0.28, 0.03), (0.34, 0.06))
    state_timing_profiles = (
        # At least one complete OFF probe is mandatory before acquisition.
        (1, 1, 2.25),
        (1, 2, 3.00),
    )
    for parent in state_machine_seed_parents:
        if not state_machine_budget_open():
            break
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for history_size, required in state_history_profiles:
            for minimum_similarity, minimum_margin in state_threshold_profiles:
                for minimum_off, revert_required, timeout_seconds in state_timing_profiles:
                    if not state_machine_budget_open():
                        break
                    config = replace(
                        source,
                        enable_transition_abstention=True,
                        enable_transition_embedding_change=False,
                        transition_clear_required=1,
                        transition_min_valid_scales=2,
                        transition_fast_scale_count=1,
                        transition_slow_scale_count=1,
                        transition_min_similarity=minimum_similarity,
                        transition_min_margin=minimum_margin,
                        transition_incumbent_history_size=history_size,
                        transition_acquire_history_size=history_size,
                        transition_acquire_required=required,
                        transition_revert_required=revert_required,
                        transition_min_off_probes=minimum_off,
                        transition_timeout_seconds=timeout_seconds,
                        enable_duration_matched_profiles=False,
                        enable_online_profiles=False,
                    )
                    row = evaluate_search(
                        windows,
                        config,
                        "STAGE_7_STATE_MACHINE",
                        "state_machine",
                        "Use an explicit OFF/STABLE/TRANSITION machine with bounded acquisition, revert, and timeout evidence, without changing embeddings.",
                        parent_candidate_id=str(parent["candidate_id"]),
                    )
                    if row is not None:
                        state_machine_rows.append(row)

    # Stage 7b: exact additive ablation.  Starting from the strongest pure
    # state-machine rows, add only a profile-independent change point measured
    # by self-similarity of the already-required fast-window embedding.
    embedding_change_parents = _top_diverse(state_machine_rows, 6)
    embedding_change_variants = (
        # history, required, maximum self-similarity, similarity drop, repeats
        (3, 2, 0.50, 0.12, 1),
        (3, 2, 0.55, 0.16, 1),
        (3, 2, 0.60, 0.20, 1),
        (3, 3, 0.55, 0.16, 1),
        (3, 3, 0.60, 0.20, 2),
        (5, 2, 0.50, 0.12, 1),
        (5, 2, 0.55, 0.20, 1),
        (5, 3, 0.55, 0.16, 1),
        (5, 3, 0.60, 0.20, 1),
        (5, 3, 0.60, 0.25, 2),
    )
    for parent in embedding_change_parents:
        if not state_machine_budget_open():
            break
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for history_size, required, maximum_similarity, drop, clear_required in embedding_change_variants:
            if not state_machine_budget_open():
                break
            config = replace(
                source,
                enable_transition_abstention=True,
                enable_transition_embedding_change=True,
                transition_embedding_history_size=history_size,
                transition_embedding_min_history=required,
                transition_embedding_max_similarity=maximum_similarity,
                transition_embedding_drop=drop,
                transition_embedding_clear_required=clear_required,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_7_EMBEDDING_CHANGE_POINT",
                "state_machine_embedding_change_point",
                "Add a profile-independent fast-embedding self-similarity change point to the otherwise identical explicit state machine.",
                parent_candidate_id=str(parent["candidate_id"]),
            )
            if row is not None:
                state_machine_embedding_rows.append(row)

    # Stage 8: target the observed dominant failure mode directly.  Most new
    # wrong-speaker time in strong two-window candidates occurs on probe ticks
    # where the causal speech gate is already false.  Keep this as two clean
    # ablations: gate-only entry from parents without transition machinery, and
    # the same gate combined with a small explicit state-machine grid.
    speech_gate_seed_parents: list[dict[str, Any]] = []
    speech_gate_seed_ids: set[str] = set()
    for family, source_rows in (
        ("consensus_crossover_history", temporal_rows),
        ("consensus", consensus_rows),
        ("score_fusion", fusion_rows),
    ):
        family_rows = [
            row
            for row in source_rows
            if row["ablation_family"] == family
            and not bool(row.get("research_only", False))
            and len(row["windows_seconds"]) <= max_fresh_live_windows_per_probe
        ]
        ranked_family = _rank(family_rows)
        if family == "consensus_crossover_history":
            # Preserve score order while giving the known b775-style combined
            # lineage first refusal when it is present in a resumed seed set.
            ranked_family = sorted(
                ranked_family,
                key=lambda row: not str(row["candidate_id"]).startswith("b775"),
            )
        for row in ranked_family[:2]:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in speech_gate_seed_ids:
                speech_gate_seed_parents.append(row)
                speech_gate_seed_ids.add(candidate_id)

    speech_gate_only_rows: list[dict[str, Any]] = []
    speech_gate_state_rows: list[dict[str, Any]] = []
    speech_gate_cap = int(args.max_speech_gate_candidates)

    def speech_gate_budget_open() -> bool:
        return (
            not _STOP
            and time.monotonic() < search_deadline
            and len(completed) < int(args.max_search_candidates)
            and len(speech_gate_only_rows) + len(speech_gate_state_rows)
            < speech_gate_cap
        )

    # Gate-only: the speech gate is the sole transition-entry trigger.  Known
    # profile crossover, incumbent-rejection, duration-profile, online-profile,
    # and embedding self-change triggers are disabled.  Normal transition
    # acquisition remains available after the gate has cleared the display.
    for parent in speech_gate_seed_parents:
        if not speech_gate_budget_open():
            break
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for speech_gate_clear_required in (1, 2):
            if not speech_gate_budget_open():
                break
            config = replace(
                source,
                enable_transition_abstention=True,
                enable_transition_speech_gate=True,
                transition_speech_gate_clear_required=speech_gate_clear_required,
                enable_transition_embedding_change=False,
                transition_clear_required=1_000_000,
                transition_incumbent_max_similarity=-1.0,
                transition_incumbent_drop=2.0,
                transition_incumbent_clear_required=1_000_000,
                transition_min_valid_scales=2,
                transition_fast_scale_count=1,
                transition_slow_scale_count=1,
                transition_min_similarity=0.30,
                transition_min_margin=0.05,
                transition_acquire_history_size=3,
                transition_acquire_required=2,
                transition_revert_required=1,
                transition_min_off_probes=1,
                transition_timeout_seconds=3.0,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_8_SPEECH_GATE_ONLY",
                "transition_speech_gate_only",
                "Enter OFF/TRANSITION only after one or two causal speech-gate-false probes; disable every other optional transition-entry trigger.",
                parent_candidate_id=str(parent["candidate_id"]),
            )
            if row is not None:
                speech_gate_only_rows.append(row)

    # Gate plus targeted state machine: restore known-speaker transition entry
    # and incumbent rejection, but keep profile extensions and embedding
    # self-change disabled so the delta remains attributable.
    speech_gate_state_variants = (
        # gate required, history, acquire required, sim, margin, incumbent max,
        # incumbent drop, revert required, timeout
        (1, 3, 2, 0.28, 0.03, 0.30, 0.10, 1, 2.25),
        (1, 3, 2, 0.34, 0.06, 0.35, 0.12, 2, 3.00),
        (1, 5, 2, 0.28, 0.03, 0.30, 0.15, 1, 2.25),
        (1, 5, 3, 0.34, 0.06, 0.35, 0.15, 2, 3.00),
        (2, 3, 2, 0.28, 0.03, 0.30, 0.10, 1, 2.25),
        (2, 3, 2, 0.34, 0.06, 0.35, 0.12, 2, 3.00),
        (2, 5, 2, 0.28, 0.03, 0.30, 0.15, 1, 2.25),
        (2, 5, 3, 0.34, 0.06, 0.35, 0.15, 2, 3.00),
    )
    for parent in speech_gate_seed_parents:
        if not speech_gate_budget_open():
            break
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for gate_required, history_size, acquire_required, minimum_similarity, minimum_margin, incumbent_maximum, incumbent_drop, revert_required, timeout_seconds in speech_gate_state_variants:
            if not speech_gate_budget_open():
                break
            config = replace(
                source,
                enable_transition_abstention=True,
                enable_transition_speech_gate=True,
                transition_speech_gate_clear_required=gate_required,
                enable_transition_embedding_change=False,
                transition_clear_required=1,
                transition_min_valid_scales=2,
                transition_fast_scale_count=1,
                transition_slow_scale_count=1,
                transition_min_similarity=minimum_similarity,
                transition_min_margin=minimum_margin,
                transition_incumbent_max_similarity=incumbent_maximum,
                transition_incumbent_drop=incumbent_drop,
                transition_incumbent_history_size=history_size,
                transition_incumbent_clear_required=1,
                transition_acquire_history_size=history_size,
                transition_acquire_required=acquire_required,
                transition_revert_required=revert_required,
                transition_min_off_probes=1,
                transition_timeout_seconds=timeout_seconds,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_8_SPEECH_GATE_STATE_MACHINE",
                "transition_speech_gate_state_machine",
                "Combine the causal speech-gate transition with a small, explicit known-speaker state-machine grid while keeping profile and self-similarity triggers disabled.",
                parent_candidate_id=str(parent["candidate_id"]),
            )
            if row is not None:
                speech_gate_state_rows.append(row)

    # Stage 9: use the short/long disagreement only as a change detector.  The
    # short scale may clear to TRANSITION, but it may not directly choose the
    # new identity.  A second causal short-window vote or renewed two-scale
    # agreement is required.  Incumbent rejection, speech-gate liveness,
    # embedding self-change, duration profiles, and online profiles are all
    # disabled so this is an exact test of the MA-crossover/history idea.
    known_crossover_rows: list[dict[str, Any]] = []
    known_crossover_parents = []
    known_crossover_parent_ids: set[str] = set()
    for family, source_rows in (
        ("consensus_crossover_history", temporal_rows),
        ("consensus", consensus_rows),
    ):
        family_rows = [
            row
            for row in source_rows
            if row["ablation_family"] == family
            and not bool(row.get("research_only", False))
            and len(row["windows_seconds"]) <= max_fresh_live_windows_per_probe
        ]
        for row in _rank(family_rows)[:2]:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in known_crossover_parent_ids:
                known_crossover_parents.append(row)
                known_crossover_parent_ids.add(candidate_id)

    known_crossover_variants = (
        # short advantage, fast/slow gap, entry votes, min sim, min margin, timeout
        (0.03, 0.05, 1, 0.25, 0.00, 1.50),
        (0.05, 0.08, 1, 0.30, 0.03, 1.50),
        (0.05, 0.10, 2, 0.25, 0.00, 2.25),
        (0.08, 0.12, 2, 0.30, 0.03, 2.25),
    )
    for parent in known_crossover_parents:
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= int(args.max_search_candidates):
            break
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for short_advantage, scale_gap, entry_required, minimum_similarity, minimum_margin, timeout_seconds in known_crossover_variants:
            config = replace(
                source,
                # Safe two-scale agreement remains a direct switch path; all
                # short-only switching goes through TRANSITION below.
                enable_consensus=True,
                min_scale_agreement=2,
                consensus_advantage=max(0.03, float(source.consensus_advantage)),
                enable_crossover=False,
                enable_history=False,
                enable_transition_abstention=True,
                enable_transition_embedding_change=False,
                enable_transition_speech_gate=False,
                transition_short_advantage=short_advantage,
                transition_scale_gap=scale_gap,
                transition_clear_required=entry_required,
                transition_min_valid_scales=2,
                transition_fast_scale_count=1,
                transition_slow_scale_count=1,
                transition_min_similarity=minimum_similarity,
                transition_min_margin=minimum_margin,
                # Explicitly remove every non-crossover transition trigger.
                transition_incumbent_max_similarity=-1.0,
                transition_incumbent_drop=2.0,
                transition_incumbent_clear_required=1_000_000,
                transition_acquire_history_size=3,
                transition_acquire_required=2,
                transition_revert_required=1,
                transition_min_off_probes=1,
                transition_timeout_seconds=timeout_seconds,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_9_KNOWN_CROSSOVER_TRANSITION",
                "known_crossover_transition",
                "Use fast/slow crossover only to enter OFF, then require a second causal fast vote or renewed two-scale agreement before assigning identity.",
                parent_candidate_id=str(parent["candidate_id"]),
            )
            if row is not None:
                known_crossover_rows.append(row)

    # Stage 10: defer assignment of later, low-evidence profiles while always
    # trusting the first causal official profile.  This targets rare one- or
    # two-sentence clusters that otherwise steal the display from an established
    # speaker.  It changes no embeddings and uses only metadata available in the
    # causal profile event.
    profile_maturity_rows: list[dict[str, Any]] = []
    profile_maturity_parents = _top_diverse(
        list(known_crossover_rows) + list(temporal_rows) + list(consensus_rows),
        6,
    )
    for parent in profile_maturity_parents:
        if _STOP or time.monotonic() >= search_deadline or len(completed) >= int(args.max_search_candidates):
            break
        if bool(parent.get("research_only", False)) or len(parent["windows_seconds"]) > max_fresh_live_windows_per_probe:
            continue
        windows = tuple(float(value) for value in parent["windows_seconds"])
        source = MultiScaleTrackerConfig(**parent["algorithm_config"])
        for minimum_sentences, minimum_speech_seconds in (
            (2, 3.0),
            (2, 5.0),
            (3, 5.0),
            (3, 6.0),
            (4, 8.0),
        ):
            config = replace(
                source,
                trusted_profile_min_sentence_count=minimum_sentences,
                trusted_profile_min_speech_seconds=minimum_speech_seconds,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
                enable_transition_embedding_change=False,
                enable_transition_speech_gate=False,
            )
            row = evaluate_search(
                windows,
                config,
                "STAGE_10_PROFILE_MATURITY_GATE",
                "profile_maturity_gate",
                "Trust the first official profile immediately, but defer later profiles until their causal sentence and speech evidence reaches a minimum.",
                parent_candidate_id=str(parent["candidate_id"]),
            )
            if row is not None:
                profile_maturity_rows.append(row)

    # Search is now closed.  Open 20v first for an exact production-baseline
    # reproduction, then for the deterministically selected finalists.  No
    # validation result may create another search candidate.
    search_ranked = _rank(completed.values())
    promotable_search_ranked = [
        row for row in search_ranked if not bool(row.get("research_only", False))
    ]
    finalists = _validation_finalists(
        promotable_search_ranked, int(args.validation_finalists)
    )
    baseline_validation_first = _score_dual_baseline(
        dataset,
        validation_videos,
        short_window=baseline_short,
        long_window=baseline_long,
        long_weight=baseline_weight,
        config=baseline_config,
    )
    baseline_validation_second = _score_dual_baseline(
        dataset,
        validation_videos,
        short_window=baseline_short,
        long_window=baseline_long,
        long_weight=baseline_weight,
        config=baseline_config,
    )
    if _stable_json(baseline_validation_first) != _stable_json(
        baseline_validation_second
    ):
        raise RuntimeError("Production dual-window validation baseline did not reproduce exactly twice")
    baseline["status"] = "SEARCH_AND_VALIDATION_REPRODUCED_TWICE_IDENTICALLY"
    baseline["validation"] = baseline_validation_first["aggregate"]
    baseline["per_video"] = {
        **baseline["per_video"],
        **baseline_validation_first["per_video"],
    }
    baseline["trace_hashes"] = {
        **baseline["trace_hashes"],
        **baseline_validation_first["trace_hashes"],
    }
    baseline["aggregate"] = aggregate_video_scores(baseline["per_video"].values())
    _atomic_json(run_dir / "baseline_reproduction.json", baseline)
    for candidate in finalists:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in validated:
            continue
        if _STOP or time.monotonic() >= deadline:
            break
        windows = tuple(float(value) for value in candidate["windows_seconds"])
        config = MultiScaleTrackerConfig(**candidate["algorithm_config"])
        scored = _score_multiscale(dataset, validation_videos, windows, config)
        combined_per_video = {
            **candidate["search_per_video"],
            **scored["per_video"],
        }
        combined = aggregate_video_scores(combined_per_video.values())
        per_video_score_delta = {
            video_id: round(
                float(combined_per_video[video_id]["strict_browser_live_score"])
                - float(baseline["per_video"][video_id]["strict_browser_live_score"]),
                6,
            )
            for video_id in scored_videos
        }
        per_video_wrong_delta = {
            video_id: round(
                float(combined_per_video[video_id]["wrong_live_speech_ratio"])
                - float(baseline["per_video"][video_id]["wrong_live_speech_ratio"]),
                6,
            )
            for video_id in scored_videos
        }
        search_improved = (
            float(candidate["search"]["global_score"])
            > float(baseline["search"]["global_score"])
            + float(args.minimum_search_improvement)
        )
        validation_gate = (
            per_video_score_delta[VALIDATION_VIDEO]
            >= -float(args.validation_score_tolerance)
        )
        wrong_gate = max(per_video_wrong_delta.values()) <= float(args.wrong_ratio_tolerance)
        combined_improved = (
            float(combined["global_score"])
            > float(baseline["aggregate"]["global_score"])
        )
        fresh_live_cost = _fresh_live_cost_diagnostics(
            windows,
            max_windows_per_probe=max_fresh_live_windows_per_probe,
            cache_hop_seconds=cache_hop_seconds,
            production_probe_interval_seconds=production_probe_interval_seconds,
        )
        fresh_live_cost_gate = bool(
            fresh_live_cost["within_fresh_live_window_budget"]
        )
        result = {
            "candidate_id": candidate_id,
            "ablation_family": candidate["ablation_family"],
            "windows_seconds": candidate["windows_seconds"],
            "algorithm_config": candidate["algorithm_config"],
            "search": candidate["search"],
            "validation": scored["aggregate"],
            "validation_per_video": scored["per_video"],
            "combined": combined,
            "combined_per_video": combined_per_video,
            "score_delta_vs_production": round(
                float(combined["global_score"])
                - float(baseline["aggregate"]["global_score"]),
                6,
            ),
            "per_video_score_delta_vs_production": per_video_score_delta,
            "per_video_wrong_ratio_delta_vs_production": per_video_wrong_delta,
            "search_improvement_gate_passed": search_improved,
            "validation_gate_passed": validation_gate,
            "wrong_ratio_gate_passed": wrong_gate,
            "combined_improvement_gate_passed": combined_improved,
            "fresh_live_window_cost_gate_passed": fresh_live_cost_gate,
            "fresh_live_cost": fresh_live_cost,
            "research_only": fresh_live_cost["research_only"],
            "research_only_reason": fresh_live_cost["research_only_reason"],
            "eligible_for_fresh_live_verification": (
                fresh_live_cost_gate
                and search_improved
                and validation_gate
                and wrong_gate
                and combined_improved
            ),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        validated[candidate_id] = result
        _append_jsonl(validation_path, result)
        write_progress("VALIDATION_GATE")

    validation_by_search_rank = [
        validated[str(row["candidate_id"])]
        for row in search_ranked
        if str(row["candidate_id"]) in validated
    ]
    eligible = [
        row
        for row in validation_by_search_rank
        if row["eligible_for_fresh_live_verification"]
    ]
    winner = eligible[0] if eligible else None

    best_by_family: dict[str, Any] = {}
    for family in (
        "score_fusion",
        "consensus",
        "crossover_only",
        "history_only",
        "consensus_crossover_history",
        "transition_abstention",
        "duration_matched_profiles",
        "online_profiles",
        "state_machine",
        "state_machine_embedding_change_point",
        "transition_speech_gate_only",
        "transition_speech_gate_state_machine",
        "known_crossover_transition",
        "profile_maturity_gate",
    ):
        row = next((item for item in search_ranked if item["ablation_family"] == family), None)
        if row is not None:
            best_by_family[family] = {
                "candidate_id": row["candidate_id"],
                "windows_seconds": row["windows_seconds"],
                "algorithm_config": row["algorithm_config"],
                "search": row["search"],
                "search_score_delta_vs_production": row[
                    "search_score_delta_vs_production"
                ],
                "fresh_live_cost": row.get("fresh_live_cost"),
                "research_only": bool(row.get("research_only", False)),
                "research_only_reason": row.get("research_only_reason"),
                "validation": validated.get(str(row["candidate_id"])),
            }
    _atomic_json(
        run_dir / "ablation_report.json",
        {
            "schema_version": 1,
            "optimizer_id": OPTIMIZER_ID,
            "selection_used_development_only": True,
            "validation_created_no_new_candidates": True,
            "known_holdout_opened": False,
            "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
            "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
            "cache_grid_is_live_probe_cadence": False,
            "best_by_family": best_by_family,
        },
    )
    state_machine_ranked = _rank(state_machine_rows)
    state_machine_embedding_ranked = _rank(state_machine_embedding_rows)
    embedding_change_deltas: list[dict[str, Any]] = []
    for row in state_machine_embedding_ranked:
        parent_id = str(row.get("parent_candidate_id") or "")
        parent = completed.get(parent_id)
        if parent is None:
            continue
        embedding_change_deltas.append(
            {
                "candidate_id": row["candidate_id"],
                "parent_candidate_id": parent_id,
                "windows_seconds": row["windows_seconds"],
                "search_global_score": row["search"]["global_score"],
                "parent_search_global_score": parent["search"]["global_score"],
                "search_score_delta_vs_same_state_machine": round(
                    float(row["search"]["global_score"])
                    - float(parent["search"]["global_score"]),
                    6,
                ),
            }
        )
    state_machine_report = {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "stage": "STAGE_7_STATE_MACHINE_AND_EMBEDDING_CHANGE_POINT",
        "parent_policy": {
            "families": [
                "score_fusion",
                "consensus",
                "consensus_crossover_history",
            ],
            "parents_per_family": 2,
            "maximum_windows_per_parent": max_fresh_live_windows_per_probe,
            "all_selected_parents_within_fresh_live_budget": all(
                len(row["windows_seconds"]) <= max_fresh_live_windows_per_probe
                and not bool(row.get("research_only", False))
                for row in state_machine_seed_parents
            ),
            "selected": [
                {
                    "candidate_id": row["candidate_id"],
                    "ablation_family": row["ablation_family"],
                    "windows_seconds": row["windows_seconds"],
                    "search_global_score": row["search"]["global_score"],
                }
                for row in state_machine_seed_parents
            ],
        },
        "candidate_cap": state_machine_cap,
        "pure_state_machine_candidates": len(state_machine_rows),
        "embedding_change_point_candidates": len(state_machine_embedding_rows),
        "total_stage_candidates": len(state_machine_rows)
        + len(state_machine_embedding_rows),
        "pure_state_machine_family": "state_machine",
        "embedding_change_point_family": "state_machine_embedding_change_point",
        "best_pure_state_machine": (
            state_machine_ranked[0] if state_machine_ranked else None
        ),
        "best_with_embedding_change_point": (
            state_machine_embedding_ranked[0]
            if state_machine_embedding_ranked
            else None
        ),
        "top_embedding_change_deltas_vs_exact_parent": embedding_change_deltas[:20],
        "fresh_live_cost_policy": {
            "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
            "cache_grid_is_live_probe_cadence": False,
        },
    }
    _atomic_json(run_dir / "state_machine_ablation_report.json", state_machine_report)
    speech_gate_only_ranked = _rank(speech_gate_only_rows)
    speech_gate_state_ranked = _rank(speech_gate_state_rows)

    def speech_gate_parent_deltas(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for row in _rank(rows):
            parent_id = str(row.get("parent_candidate_id") or "")
            parent = completed.get(parent_id)
            if parent is None:
                continue
            values.append(
                {
                    "candidate_id": row["candidate_id"],
                    "parent_candidate_id": parent_id,
                    "ablation_family": row["ablation_family"],
                    "windows_seconds": row["windows_seconds"],
                    "search_global_score": row["search"]["global_score"],
                    "parent_search_global_score": parent["search"]["global_score"],
                    "search_score_delta_vs_parent": round(
                        float(row["search"]["global_score"])
                        - float(parent["search"]["global_score"]),
                        6,
                    ),
                }
            )
        return values

    speech_gate_report = {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "stage": "STAGE_8_SPEECH_GATE_TRANSITION",
        "rationale": {
            "observed_fraction_of_new_wrong_time_on_speech_gate_false": "0.73-0.74",
            "claim_scope": "diagnostic motivation, not an algorithm input",
        },
        "parent_policy": {
            "priority": [
                "consensus_crossover_history",
                "consensus",
                "score_fusion",
            ],
            "parents_per_family": 2,
            "maximum_windows_per_parent": max_fresh_live_windows_per_probe,
            "all_selected_parents_within_fresh_live_budget": all(
                len(row["windows_seconds"]) <= max_fresh_live_windows_per_probe
                and not bool(row.get("research_only", False))
                for row in speech_gate_seed_parents
            ),
            "selected": [
                {
                    "candidate_id": row["candidate_id"],
                    "ablation_family": row["ablation_family"],
                    "windows_seconds": row["windows_seconds"],
                    "search_global_score": row["search"]["global_score"],
                }
                for row in speech_gate_seed_parents
            ],
        },
        "candidate_cap": speech_gate_cap,
        "gate_only_candidates": len(speech_gate_only_rows),
        "gate_plus_state_machine_candidates": len(speech_gate_state_rows),
        "total_stage_candidates": len(speech_gate_only_rows)
        + len(speech_gate_state_rows),
        "gate_only_family": "transition_speech_gate_only",
        "gate_plus_state_machine_family": "transition_speech_gate_state_machine",
        "best_gate_only": (
            speech_gate_only_ranked[0] if speech_gate_only_ranked else None
        ),
        "best_gate_plus_state_machine": (
            speech_gate_state_ranked[0] if speech_gate_state_ranked else None
        ),
        "top_gate_only_deltas_vs_parent": speech_gate_parent_deltas(
            speech_gate_only_rows
        )[:20],
        "top_gate_plus_state_machine_deltas_vs_parent": speech_gate_parent_deltas(
            speech_gate_state_rows
        )[:20],
        "fresh_live_cost_policy": {
            "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
            "cache_grid_is_live_probe_cadence": False,
        },
    }
    _atomic_json(run_dir / "speech_gate_transition_report.json", speech_gate_report)
    best_promotable_search = (
        promotable_search_ranked[0] if promotable_search_ranked else None
    )
    research_only_search_count = len(search_ranked) - len(promotable_search_ranked)
    sweep = {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "provider": provider,
        "search_videos": search_videos,
        "validation_videos": validation_videos,
        "known_holdout_excluded": [KNOWN_HOLDOUT],
        "sealed_holdout_opened": False,
        "baseline": baseline,
        "evaluated_search_candidates": len(completed),
        "validated_finalists": len(validated),
        "phase_counts": phase_counts,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "best_search": search_ranked[0] if search_ranked else None,
        "best_promotable_search": best_promotable_search,
        "promotable_search_candidates": len(promotable_search_ranked),
        "research_only_search_candidates": research_only_search_count,
        "fresh_live_cost_policy": {
            "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
            "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
            "cache_hop_seconds": cache_hop_seconds,
            "production_probe_interval_seconds": production_probe_interval_seconds,
            "cache_grid_is_live_probe_cadence": False,
            "fresh_live_cadence_verified": False,
        },
        "state_machine_stage": {
            "candidate_cap": state_machine_cap,
            "pure_state_machine_candidates": len(state_machine_rows),
            "embedding_change_point_candidates": len(state_machine_embedding_rows),
            "total_candidates": len(state_machine_rows)
            + len(state_machine_embedding_rows),
            "report": "state_machine_ablation_report.json",
        },
        "speech_gate_transition_stage": {
            "candidate_cap": speech_gate_cap,
            "gate_only_candidates": len(speech_gate_only_rows),
            "gate_plus_state_machine_candidates": len(speech_gate_state_rows),
            "total_candidates": len(speech_gate_only_rows)
            + len(speech_gate_state_rows),
            "report": "speech_gate_transition_report.json",
        },
        "top20_search": search_ranked[:20],
        "validation_results_in_search_rank_order": validation_by_search_rank,
        "winner": winner,
    }
    _atomic_json(run_dir / "cached_sweep.json", sweep)
    champion_output = {
        "schema_version": 1,
        "status": (
            "CACHE_CHAMPION_PENDING_FRESH_LIVE"
            if winner is not None
            else "NO_PROMOTABLE_CANDIDATE_PASSED_VALIDATION"
        ),
        "optimizer_id": OPTIMIZER_ID,
        "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
        "provider": provider,
        "previous_production_score": baseline["aggregate"]["global_score"],
        "fresh_live_verified": False,
        "production_ready": False,
        "sealed_holdout_opened": False,
        "fresh_live_cost_policy": {
            "configured_max_fresh_windows_per_probe": max_fresh_live_windows_per_probe,
            "hard_max_fresh_windows_per_probe": HARD_MAX_FRESH_LIVE_WINDOWS_PER_PROBE,
            "cache_grid_is_live_probe_cadence": False,
        },
        "candidate": winner,
    }
    if winner is not None:
        champion_output.update(
            {
                "candidate_id": winner["candidate_id"],
                "cached_score": winner["combined"]["global_score"],
                "cached_score_delta": winner["score_delta_vs_production"],
                "windows_seconds": winner["windows_seconds"],
                "algorithm_config": winner["algorithm_config"],
                "ablation_family": winner["ablation_family"],
                "fresh_live_cost": winner["fresh_live_cost"],
            }
        )
    _atomic_json(run_dir / "champion.json", champion_output)
    write_progress("COMPLETE" if not _STOP else "CONTROLLED_STOP")
    print(
        json.dumps(
            {
                "baseline_score": baseline["aggregate"]["global_score"],
                "baseline_search_score": baseline["search"]["global_score"],
                "evaluated_search_candidates": len(completed),
                "validated_finalists": len(validated),
                "best_search_score": (
                    search_ranked[0]["search"]["global_score"] if search_ranked else None
                ),
                "best_promotable_search_score": (
                    best_promotable_search["search"]["global_score"]
                    if best_promotable_search
                    else None
                ),
                "promotable_search_candidates": len(promotable_search_ranked),
                "research_only_search_candidates": research_only_search_count,
                "max_fresh_live_windows_per_probe": max_fresh_live_windows_per_probe,
                "cache_grid_is_live_probe_cadence": False,
                "winner": winner,
                "sealed_holdout_opened": False,
                "run_dir": str(run_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
