"""Resumable Top-7 search for a causal, two-window similarity tracker.

This campaign deliberately differs from the production dual-window algorithm:
the two embeddings are never averaged.  They remain independent sensors whose
speaker similarities can be fused, compared over time, or used as a causal
change detector.  Every candidate is selected by one scalar only: the macro
mean of ``strict_browser_live_score`` over all seven prepared videos.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (SRC, ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_overnight_top7 import Dataset, evaluate_candidate
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import (
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)
from window.live_speaker_multiscale import (
    MULTISCALE_ALGORITHM_ID,
    MultiScaleTrackerConfig,
    replay_cached_multiscale_windows,
)


OPTIMIZER_ID = "live_speaker_top7_two_window_multiscale_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _compact_video_score(score: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in score.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    for key in ("sampled_playback_seconds", "speaker_map"):
        if key in score:
            compact[key] = score[key]
    for key, omitted in (("turn_latency", "turns"), ("release", "events")):
        value = score.get(key)
        if isinstance(value, dict):
            compact[key] = {name: item for name, item in value.items() if name != omitted}
    availability = score.get("profile_availability")
    if isinstance(availability, dict):
        compact["profile_availability"] = {"counts": dict(availability.get("counts") or {})}
    return compact


def _score_multiscale(
    dataset: Dataset,
    videos: Sequence[str],
    windows: Sequence[float],
    config: MultiScaleTrackerConfig,
) -> dict[str, Any]:
    per_video: dict[str, dict[str, Any]] = {}
    short_window = float(min(windows))
    for video_id in videos:
        inputs = dataset.video_inputs(video_id, short_window)
        decisions = replay_cached_multiscale_windows(
            [dataset.block(video_id, window) for window in windows],
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            config=config,
        )
        per_video[video_id] = _compact_video_score(
            score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"])
        )
    aggregate = aggregate_video_scores_primary_v2(per_video.values())
    if not math.isfinite(float(aggregate["primary_score"])):
        raise RuntimeError("Non-finite primary score")
    return {"aggregate": aggregate, "per_video": per_video}


def _candidate_id(windows: Sequence[float], config: MultiScaleTrackerConfig) -> str:
    return _stable_id(
        {
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": MULTISCALE_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "windows_seconds": [round(float(value), 3) for value in windows],
            "config": asdict(config),
        }
    )


def _rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["aggregate"]["primary_score"]),
            -float(row["aggregate"]["diagnostics"]["mean_wrong_live_speech_ratio"]),
        ),
        reverse=True,
    )


def _top(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return _rank(rows)[:count]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the causal two-window multiscale tracker on all Top-7 videos."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--max-candidates", type=int, default=1600)
    parser.add_argument("--minimum-improvement", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_payload = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    champion = dict(champion_payload["description"])
    videos = [str(value) for value in spec["videos"]]
    provider_spec = str(champion["provider_spec"])
    profile_name = str(champion["profile_name"])
    if provider_spec != "speechbrain_resnet":
        raise ValueError(f"Expected the verified speechbrain_resnet champion, got {provider_spec!r}")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for raw in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                completed[str(row["candidate_id"])] = row
    elif trials_path.exists():
        raise FileExistsError(f"{trials_path} exists; use --resume or a new run directory")

    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider_spec, profile_name)
    production_result = evaluate_candidate(dataset, videos, champion)
    production_score = float(production_result["aggregate"]["primary_score"])
    _atomic_json(
        run_dir / "baseline_reproduction.json",
        {
            "status": "REPRODUCED",
            "description": champion,
            "aggregate": production_result["aggregate"],
            "per_video": production_result["per_video"],
        },
    )
    incumbent: dict[str, Any] | None = None
    if completed:
        incumbent = _rank(completed.values())[0]

    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    def write_state(phase: str, active: str = "") -> None:
        best_score = (
            float(incumbent["aggregate"]["primary_score"])
            if incumbent is not None
            else production_score
        )
        status = "interrupted" if _STOP else "running"
        _atomic_json(
            run_dir / "progress.json",
            {
                "schema_version": 1,
                "optimizer_id": OPTIMIZER_ID,
                "status": status,
                "phase": phase,
                "active": active,
                "completed_candidate_count": len(completed),
                "phase_counts": phase_counts,
                "production_champion_score": production_score,
                "best_multiscale_score": best_score if incumbent is not None else None,
                "best_score_delta": round(best_score - production_score, 6),
                "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "budget_seconds": int(args.budget_seconds),
                "maximum_fresh_windows_per_probe": 2,
            },
        )
        if incumbent is not None:
            _atomic_json(
                run_dir / "champion.json",
                {
                    "status": (
                        "CACHE_MULTISCALE_WINNER_PENDING_PRODUCTION_INTEGRATION"
                        if best_score > production_score + float(args.minimum_improvement)
                        else "RESEARCH_BEST_BELOW_PRODUCTION_CHAMPION"
                    ),
                    "selection_policy": "primary_score_only_no_per_video_vetoes",
                    "production_champion_score": production_score,
                    "candidate_score": best_score,
                    "score_delta": round(best_score - production_score, 6),
                    "candidate_id": incumbent["candidate_id"],
                    "provider_spec": provider_spec,
                    "profile_name": profile_name,
                    "windows_seconds": incumbent["windows_seconds"],
                    "algorithm_config": incumbent["algorithm_config"],
                    "aggregate": incumbent["aggregate"],
                    "per_video": incumbent["per_video"],
                    "fresh_live_verified": False,
                },
            )

    def evaluate(
        windows: Sequence[float],
        config: MultiScaleTrackerConfig,
        phase: str,
        family: str,
        hypothesis: str,
        parent: str | None = None,
    ) -> dict[str, Any] | None:
        nonlocal incumbent
        windows = tuple(round(float(value), 3) for value in windows)
        if len(windows) != 2:
            raise ValueError("Every candidate must use exactly two embedding windows")
        if tuple(config.scale_windows) != windows:
            config = replace(config, scale_windows=windows)
        candidate_id = _candidate_id(windows, config)
        previous = completed.get(candidate_id)
        if previous is not None:
            return previous
        if _STOP or time.monotonic() >= deadline or len(completed) >= int(args.max_candidates):
            return None
        write_state(phase, candidate_id)
        result = _score_multiscale(dataset, videos, windows, config)
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "family": family,
            "hypothesis": hypothesis,
            "parent_candidate_id": parent,
            "provider_spec": provider_spec,
            "profile_name": profile_name,
            "windows_seconds": list(windows),
            "algorithm_config": asdict(config),
            **result,
            "score_delta_vs_production": round(
                float(result["aggregate"]["primary_score"]) - production_score, 6
            ),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[candidate_id] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        _append_jsonl(trials_path, row)
        if incumbent is None or float(row["aggregate"]["primary_score"]) > float(
            incumbent["aggregate"]["primary_score"]
        ) + float(args.minimum_improvement):
            incumbent = row
        write_state(phase, candidate_id)
        return row

    _atomic_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "optimizer_id": OPTIMIZER_ID,
            "multiscale_algorithm_id": MULTISCALE_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "promotion_policy": "maximize_one_top7_macro_score",
            "per_video_scores_are_diagnostics_only": True,
            "maximum_fresh_windows_per_probe": 2,
            "videos": videos,
            "provider_spec": provider_spec,
            "profile_name": profile_name,
            "budget_seconds": int(args.budget_seconds),
        },
    )
    write_state("BASELINE")

    base = MultiScaleTrackerConfig(
        scale_windows=(0.7, 2.9),
        scale_weights=(0.8, 0.2),
        min_similarity=0.30,
        min_margin=0.05,
        acquire_scale_agreement=1,
        min_scale_agreement=2,
        enable_consensus=False,
        enable_crossover=False,
        enable_history=False,
        unknown_release_count=2,
        silence_release_count=2,
    )

    # Stage 1: establish which pair and independent-score fusion are useful.
    fusion_rows: list[dict[str, Any]] = []
    window_pairs = (
        (0.7, 1.5), (0.7, 2.0), (0.7, 2.4), (0.7, 2.8),
        (0.7, 2.9), (0.7, 3.0), (0.8, 2.8), (0.9, 2.9),
    )
    weight_pairs = ((0.90, 0.10), (0.80, 0.20), (0.75, 0.25), (0.65, 0.35), (0.50, 0.50), (0.35, 0.65))
    threshold_pairs = ((0.25, 0.00), (0.28, 0.03), (0.30, 0.00), (0.30, 0.03), (0.30, 0.05), (0.35, 0.03), (0.35, 0.05), (0.40, 0.05))
    for windows in window_pairs:
        for weights in weight_pairs:
            for minimum, margin in threshold_pairs:
                config = replace(
                    base,
                    scale_windows=windows,
                    scale_weights=weights,
                    min_similarity=minimum,
                    min_margin=margin,
                )
                row = evaluate(
                    windows, config, "STAGE_1_INDEPENDENT_FUSION", "independent_similarity_fusion",
                    "Keep both embeddings independent and fuse their per-speaker similarities.",
                )
                if row is not None:
                    fusion_rows.append(row)

    # Stage 2: independent scales must agree before a switch, but acquisition may remain fast.
    consensus_rows: list[dict[str, Any]] = []
    for source_row in _top(fusion_rows, 12):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for agreement, advantage, acquire in ((2, 0.00, 1), (2, 0.02, 1), (2, 0.04, 1), (2, 0.06, 1), (2, 0.10, 1), (2, 0.04, 2)):
            config = replace(
                source,
                enable_consensus=True,
                min_scale_agreement=agreement,
                consensus_advantage=advantage,
                acquire_scale_agreement=acquire,
            )
            row = evaluate(
                windows, config, "STAGE_2_CONSENSUS", "scale_consensus",
                "Switch only when two independent window scales agree on the challenger.",
                source_row["candidate_id"],
            )
            if row is not None:
                consensus_rows.append(row)

    # Stage 3: MA-crossover and short bounded history, each separately and combined.
    temporal_rows: list[dict[str, Any]] = []
    for source_row in _top(fusion_rows + consensus_rows, 12):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for short_advantage, scale_gap, required in ((0.03, 0.05, 1), (0.03, 0.08, 2), (0.05, 0.10, 2), (0.08, 0.12, 2), (0.10, 0.18, 3)):
            config = replace(
                source, enable_crossover=True, enable_history=False,
                crossover_short_advantage=short_advantage,
                crossover_scale_gap=scale_gap, crossover_required=required,
            )
            row = evaluate(
                windows, config, "STAGE_3_CROSSOVER", "crossover",
                "Use a short-over-long similarity crossover as causal evidence of a speaker turn.",
                source_row["candidate_id"],
            )
            if row is not None:
                temporal_rows.append(row)
        for size, required, advantage, short_weight, statistic in (
            (3, 2, 0.02, 1.0, "mean"), (3, 2, 0.02, 0.5, "mean"),
            (3, 2, 0.03, 0.5, "median"), (5, 2, 0.05, 0.5, "mean"),
            (5, 3, 0.03, 0.5, "mean"), (5, 3, 0.03, 0.0, "median"),
        ):
            config = replace(
                source, enable_history=True,
                history_size=size, history_required=required,
                history_advantage=advantage, history_short_weight=short_weight,
                history_statistic=statistic,
            )
            row = evaluate(
                windows, config, "STAGE_3_HISTORY", "bounded_history",
                "Use only the last three to five causal probes to reject one-tick identity noise.",
                source_row["candidate_id"],
            )
            if row is not None:
                temporal_rows.append(row)
        for crossover_required, size, history_required, statistic in ((1, 3, 2, "mean"), (2, 3, 2, "mean"), (2, 5, 3, "mean"), (2, 5, 3, "median")):
            config = replace(
                source,
                enable_consensus=True, min_scale_agreement=2, consensus_advantage=0.03,
                enable_crossover=True, crossover_short_advantage=0.05,
                crossover_scale_gap=0.10, crossover_required=crossover_required,
                enable_history=True, history_size=size, history_required=history_required,
                history_advantage=0.03, history_short_weight=0.5,
                history_statistic=statistic,
            )
            row = evaluate(
                windows, config, "STAGE_3_COMBINED", "consensus_crossover_history",
                "Combine scale agreement, crossover, and a bounded three-to-five-probe history.",
                source_row["candidate_id"],
            )
            if row is not None:
                temporal_rows.append(row)

    # Stage 4: explicitly display OFF while fast and slow evidence contradict at a boundary.
    transition_rows: list[dict[str, Any]] = []
    for source_row in _top(consensus_rows + temporal_rows, 10):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for incumbent_max, incumbent_drop, acquire_required in (
            (0.20, 0.10, 1), (0.25, 0.10, 1), (0.30, 0.10, 1),
            (0.35, 0.05, 1), (0.35, 0.10, 1), (0.35, 0.15, 1),
            (0.40, 0.10, 1), (0.35, 0.10, 2),
        ):
            config = replace(
                source,
                enable_transition_abstention=True,
                transition_short_advantage=0.03,
                transition_scale_gap=0.05,
                transition_clear_required=1,
                transition_min_valid_scales=2,
                transition_incumbent_max_similarity=incumbent_max,
                transition_incumbent_drop=incumbent_drop,
                transition_incumbent_clear_required=1,
                transition_acquire_history_size=3,
                transition_acquire_required=acquire_required,
                transition_min_off_probes=1,
                enable_online_profiles=False,
            )
            row = evaluate(
                windows, config, "STAGE_4_TRANSITION", "transition_abstention",
                "Clear the stale identity at a likely boundary and reacquire from causal evidence.",
                source_row["candidate_id"],
            )
            if row is not None:
                transition_rows.append(row)

    # Stage 5: compare live windows against causal profiles built at matching durations.
    duration_rows: list[dict[str, Any]] = []
    for source_row in _top(temporal_rows + transition_rows, 10):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for weight, alpha, minimum_windows, cohesion, guard in (
            (0.25, 0.20, 2, 0.35, 0.05), (0.50, 0.25, 2, 0.45, 0.05),
            (0.75, 0.25, 2, 0.45, 0.05), (1.00, 0.25, 2, 0.45, 0.05),
            (0.50, 0.10, 1, 0.35, 0.00), (0.50, 0.25, 2, 0.60, 0.10),
        ):
            config = replace(
                source,
                enable_duration_matched_profiles=True,
                duration_profile_score_weight=weight,
                duration_profile_update_alpha=alpha,
                duration_profile_min_windows=minimum_windows,
                duration_profile_min_cohesion=cohesion,
                duration_profile_guard_seconds=guard,
                enable_online_profiles=False,
            )
            row = evaluate(
                windows, config, "STAGE_5_DURATION_PROFILES", "duration_matched_profiles",
                "Compare short and long probes with causal speaker prototypes of the same duration.",
                source_row["candidate_id"],
            )
            if row is not None:
                duration_rows.append(row)

    # Stage 6: an explicit OFF/STABLE/TRANSITION state machine, with optional
    # profile-independent fast-window embedding change detection.
    state_rows: list[dict[str, Any]] = []
    for source_row in _top(fusion_rows + consensus_rows + temporal_rows + transition_rows, 8):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for history_size, required, minimum, margin, revert, timeout in (
            (3, 2, 0.28, 0.03, 1, 2.25), (3, 2, 0.34, 0.06, 2, 3.00),
            (5, 2, 0.28, 0.03, 1, 2.25), (5, 3, 0.34, 0.06, 2, 3.00),
        ):
            config = replace(
                source,
                enable_transition_abstention=True,
                transition_min_valid_scales=2,
                transition_fast_scale_count=1,
                transition_slow_scale_count=1,
                transition_min_similarity=minimum,
                transition_min_margin=margin,
                transition_incumbent_history_size=history_size,
                transition_acquire_history_size=history_size,
                transition_acquire_required=required,
                transition_revert_required=revert,
                transition_min_off_probes=1,
                transition_timeout_seconds=timeout,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate(
                windows, config, "STAGE_6_STATE_MACHINE", "state_machine",
                "Model OFF, stable identity, and uncertain transition as separate causal states.",
                source_row["candidate_id"],
            )
            if row is not None:
                state_rows.append(row)
                for maximum_similarity, drop in ((0.50, 0.12), (0.55, 0.16), (0.60, 0.20)):
                    changed = replace(
                        config,
                        enable_transition_embedding_change=True,
                        transition_embedding_history_size=history_size,
                        transition_embedding_min_history=min(required, history_size),
                        transition_embedding_max_similarity=maximum_similarity,
                        transition_embedding_drop=drop,
                        transition_embedding_clear_required=1,
                    )
                    changed_row = evaluate(
                        windows, changed, "STAGE_6_CHANGE_POINT", "embedding_change_point",
                        "Detect a boundary from fast-window self-similarity before a new profile wins.",
                        row["candidate_id"],
                    )
                    if changed_row is not None:
                        state_rows.append(changed_row)

    # Stage 7: causal speech-gate transitions and later-profile maturity.
    gate_rows: list[dict[str, Any]] = []
    for source_row in _top(temporal_rows + state_rows, 8):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for gate_required, acquire_required in ((1, 1), (1, 2), (2, 1), (2, 2)):
            config = replace(
                source,
                enable_transition_abstention=True,
                enable_transition_speech_gate=True,
                transition_speech_gate_clear_required=gate_required,
                transition_min_valid_scales=2,
                transition_acquire_history_size=3,
                transition_acquire_required=acquire_required,
                transition_min_off_probes=1,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate(
                windows, config, "STAGE_7_SPEECH_GATE", "speech_gate_transition",
                "Use a causal speech-gate drop to release the old identity at a pause boundary.",
                source_row["candidate_id"],
            )
            if row is not None:
                gate_rows.append(row)

    maturity_rows: list[dict[str, Any]] = []
    for source_row in _top(temporal_rows + transition_rows + state_rows + gate_rows, 8):
        windows = tuple(source_row["windows_seconds"])
        source = MultiScaleTrackerConfig(**source_row["algorithm_config"])
        for sentences, speech_seconds in ((2, 3.0), (2, 5.0), (3, 5.0), (3, 6.0), (4, 8.0)):
            config = replace(
                source,
                trusted_profile_min_sentence_count=sentences,
                trusted_profile_min_speech_seconds=speech_seconds,
                enable_duration_matched_profiles=False,
                enable_online_profiles=False,
            )
            row = evaluate(
                windows, config, "STAGE_7_PROFILE_MATURITY", "profile_maturity",
                "Prevent a very immature later profile from stealing an established identity.",
                source_row["candidate_id"],
            )
            if row is not None:
                maturity_rows.append(row)

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    best_score = float(incumbent["aggregate"]["primary_score"]) if incumbent else production_score
    _atomic_json(
        run_dir / "final_report.json",
        {
            "schema_version": 1,
            "optimizer_id": OPTIMIZER_ID,
            "status": progress["status"],
            "production_champion_score": production_score,
            "best_multiscale_score": best_score if incumbent else None,
            "score_delta": round(best_score - production_score, 6),
            "candidate_count": len(completed),
            "winner_requires_production_integration_and_fresh_live_verification": best_score > production_score,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(json.dumps({
        "status": progress["status"],
        "production_champion_score": production_score,
        "best_multiscale_score": best_score if incumbent else None,
        "score_delta": round(best_score - production_score, 6),
        "candidate_count": len(completed),
        "champion_path": str(run_dir / "champion.json"),
    }, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
