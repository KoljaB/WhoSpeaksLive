"""Focused cached sweep for the causal two-window legacy/residual hybrid.

Dd, Dsy, and 20v are development data because results from all three have already
informed the design.  JWS remains sealed and this program refuses to score it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from optimize_live_speaker_replay import Dataset
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_hybrid import (
    HYBRID_ALGORITHM_ID,
    HybridSpeakerTrackerConfig,
    hybrid_config_identity_payload,
    replay_hybrid_decisions,
)
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_replay import replay_cached_live_windows_dual


OPTIMIZER_ID = "causal_live_speaker_hybrid_sweep_v1"
DEVELOPMENT_VIDEOS = ("Dd7FixvoKBw", "DsyfYJ5Ou3g", "20v1OxUXcQY")
SEALED_HOLDOUT = "JWS-qfR6K3w"
SHORT_WINDOW_SECONDS = 0.8
LONG_WINDOW_SECONDS = 2.8
LONG_WEIGHT = 0.25
MAX_FRESH_WINDOWS_PER_PROBE = 2
CACHE_HOP_SECONDS = 0.2
PRODUCTION_PROBE_INTERVAL_SECONDS = 0.75
DEFAULT_PROVIDER = (
    "pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5"
)

BASELINE_CONFIG = LiveSpeakerAlgorithmConfig(
    min_similarity=0.35,
    min_margin=0.08,
    min_known_probability=0.5,
    ema_count=1,
    ema_alpha=0.55,
    acquire_count=1,
    switch_count=1,
    unknown_release_count=2,
    silence_release_count=2,
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_id(family: str, config: HybridSpeakerTrackerConfig) -> str:
    config_payload = hybrid_config_identity_payload(config)
    return hashlib.sha256(
        _stable_json({"family": family, "config": config_payload}).encode("utf-8")
    ).hexdigest()


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


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    turn_latency = {
        key: value for key, value in score["turn_latency"].items() if key != "turns"
    }
    release = {
        key: value for key, value in score["release"].items() if key != "events"
    }
    return {
        "strict_browser_live_score": float(score["strict_browser_live_score"]),
        "wrong_live_speech_ratio": float(score["wrong_live_speech_ratio"]),
        "missing_live_speech_ratio": float(score["missing_live_speech_ratio"]),
        "correct_live_speaker_coverage": float(score["correct_live_speaker_coverage"]),
        "correct_live_precision_during_speech": float(
            score["correct_live_precision_during_speech"]
        ),
        "turn_latency": turn_latency,
        "release": release,
    }


def _build_steps(short: Any, long: Any, inputs: dict[str, Any]) -> list[MultiScaleStep]:
    if short.video_id != long.video_id or not np.array_equal(short.media_times, long.media_times):
        raise ValueError("hybrid cached blocks must share video and timeline")
    steps: list[MultiScaleStep] = []
    for index, media_time in enumerate(short.media_times):
        scheduled = bool(inputs["probes"][index])
        evidences = tuple(
            MultiScaleEvidence(float(block.window_seconds), block.embeddings[index])
            for block in (short, long)
            if scheduled and bool(block.valid[index])
        )
        if len(evidences) > MAX_FRESH_WINDOWS_PER_PROBE:
            raise AssertionError("hybrid sweep attempted more than two windows")
        steps.append(MultiScaleStep(
            media_time=float(media_time),
            speech=bool(inputs["speech"][index]),
            evidences=evidences,
            probe_scheduled=scheduled,
            release_signal=bool(inputs["releases"][index]),
            skipped_reason=(
                "" if evidences else
                "not_a_scheduled_probe" if not scheduled else
                "cached_embeddings_invalid"
            ),
        ))
    return steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--score-tolerance", type=float, default=0.005)
    parser.add_argument("--wrong-ratio-tolerance", type=float, default=0.005)
    parser.add_argument("--boundary-material-improvement", type=float, default=0.005)
    return parser.parse_args()


def _young_grid(*, lease: bool) -> list[HybridSpeakerTrackerConfig]:
    return [
        HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=lease,
            young_trusted_min_sentence_count=4,
            young_trusted_min_speech_seconds=8.0,
            young_min_similarity=similarity,
            young_min_margin=margin,
            young_required_consecutive_probes=required,
            young_independent_scale_count=permanent_scales,
            young_fast_independent_scale_count=1,
        )
        for similarity, margin, permanent_scales, required in itertools.product(
            (0.30, 0.35, 0.40), (0.04, 0.08), (1, 2), (1, 2)
        )
    ]


def _boundary_grid(base: HybridSpeakerTrackerConfig) -> list[HybridSpeakerTrackerConfig]:
    return [
        replace(
            base,
            enable_boundary_abstention=True,
            boundary_min_similarity=similarity,
            boundary_min_margin=margin,
            boundary_short_advantage=short_advantage,
            boundary_long_advantage=long_advantage,
            boundary_required_consecutive_probes=required,
        )
        for similarity, margin, short_advantage, long_advantage, required in itertools.product(
            (0.30, 0.35, 0.40), (0.04, 0.08), (0.04, 0.08), (0.04, 0.08), (1, 2)
        )
    ]


def _fast_latency_eligible(config: HybridSpeakerTrackerConfig) -> bool:
    return bool(
        config.enable_young_profile_confirmation
        and config.enable_young_profile_lease
        and int(config.young_fast_independent_scale_count) == 1
        and int(config.young_independent_scale_count) == 2
        and int(config.young_required_consecutive_probes) == 1
    )


def _fresh_live_cost() -> dict[str, Any]:
    return {
        "fresh_window_requests_per_probe": 2,
        "max_fresh_window_requests_per_probe": MAX_FRESH_WINDOWS_PER_PROBE,
        "within_window_budget": True,
        "provider_component_forwards_per_probe": 4,
        "provider_component_count": 2,
        "cache_hop_seconds": CACHE_HOP_SECONDS,
        "production_probe_interval_seconds": PRODUCTION_PROBE_INTERVAL_SECONDS,
        "cache_grid_is_live_probe_cadence": False,
        "fresh_live_cadence_verified": False,
    }


def _prepare(dataset: Dataset) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for video_id in DEVELOPMENT_VIDEOS:
        if video_id == SEALED_HOLDOUT:
            raise AssertionError("sealed holdout entered the development split")
        inputs = dataset.video_inputs(video_id)
        short = dataset.block(video_id, SHORT_WINDOW_SECONDS)
        long = dataset.block(video_id, LONG_WINDOW_SECONDS)
        baseline = replay_cached_live_windows_dual(
            short,
            long,
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            long_weight=LONG_WEIGHT,
            config=BASELINE_CONFIG,
        )
        prepared[video_id] = {
            "inputs": inputs,
            "baseline": baseline,
            "steps": _build_steps(short, long, inputs),
        }
    return prepared


def _baseline_result(prepared: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_video = {
        video_id: score_live_speaker_decisions(
            value["baseline"],
            value["inputs"]["canonical"],
            value["inputs"]["profiles"],
        )
        for video_id, value in prepared.items()
    }
    return {
        "aggregate": aggregate_video_scores(per_video.values()),
        "per_video": per_video,
    }


def _score_candidate(
    prepared: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    family: str,
    config: HybridSpeakerTrackerConfig,
    *,
    score_tolerance: float,
    wrong_tolerance: float,
) -> dict[str, Any]:
    started = time.monotonic()
    per_video: dict[str, Any] = {}
    for video_id, value in prepared.items():
        decisions = replay_hybrid_decisions(
            value["baseline"],
            value["steps"],
            value["inputs"]["profiles"],
            config=config,
        )
        per_video[video_id] = score_live_speaker_decisions(
            decisions,
            value["inputs"]["canonical"],
            value["inputs"]["profiles"],
        )
    aggregate = aggregate_video_scores(per_video.values())
    score_deltas = {
        video_id: float(score["strict_browser_live_score"])
        - float(baseline["per_video"][video_id]["strict_browser_live_score"])
        for video_id, score in per_video.items()
    }
    wrong_deltas = {
        video_id: float(score["wrong_live_speech_ratio"])
        - float(baseline["per_video"][video_id]["wrong_live_speech_ratio"])
        for video_id, score in per_video.items()
    }
    score_gate = all(value >= -float(score_tolerance) for value in score_deltas.values())
    wrong_gate = all(value <= float(wrong_tolerance) for value in wrong_deltas.values())
    improvement_gate = (
        float(aggregate["global_score"])
        > float(baseline["aggregate"]["global_score"]) + 1e-9
    )
    return {
        "candidate_id": _candidate_id(family, config),
        "family": family,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(config),
        "aggregate": aggregate,
        "per_video": {key: _compact_score(value) for key, value in per_video.items()},
        "score_delta_vs_baseline": float(aggregate["global_score"])
        - float(baseline["aggregate"]["global_score"]),
        "per_video_score_delta_vs_baseline": score_deltas,
        "per_video_wrong_ratio_delta_vs_baseline": wrong_deltas,
        "score_gate_passed": score_gate,
        "wrong_ratio_gate_passed": wrong_gate,
        "combined_improvement_gate_passed": improvement_gate,
        "fast_latency_eligible": _fast_latency_eligible(config),
        "promotion_gates_passed": bool(
            score_gate and wrong_gate and improvement_gate and _fast_latency_eligible(config)
        ),
        "fresh_live_cost": _fresh_live_cost(),
        "elapsed_seconds": time.monotonic() - started,
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    config = row["config"]
    return (
        float(row["aggregate"]["global_score"]),
        float(config["young_min_margin"]),
        -abs(float(config["young_min_similarity"]) - 0.35),
    )


def _best(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(rows)
    return max(values, key=_rank_key) if values else None


def _boundary_materially_better(
    candidate: dict[str, Any],
    simple: dict[str, Any],
    minimum_improvement: float,
) -> bool:
    if (
        float(candidate["aggregate"]["global_score"])
        < float(simple["aggregate"]["global_score"]) + float(minimum_improvement) - 1e-12
    ):
        return False
    for video_id in DEVELOPMENT_VIDEOS:
        candidate_score = candidate["per_video"][video_id]
        simple_score = simple["per_video"][video_id]
        if (
            float(candidate_score["strict_browser_live_score"])
            - float(simple_score["strict_browser_live_score"])
            < -0.002 - 1e-12
        ):
            return False
        if (
            float(candidate_score["wrong_live_speech_ratio"])
            - float(simple_score["wrong_live_speech_ratio"])
            > 0.002 + 1e-12
        ):
            return False
    return True


def main() -> int:
    args = _parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if SEALED_HOLDOUT in DEVELOPMENT_VIDEOS:
        raise AssertionError("JWS must remain sealed")

    started = time.monotonic()
    dataset = Dataset(args.corpus_root, args.input_root, str(args.provider))
    prepared = _prepare(dataset)
    baseline = _baseline_result(prepared)
    baseline_artifact = {
        "algorithm_config": asdict(BASELINE_CONFIG),
        "windows_seconds": [SHORT_WINDOW_SECONDS, LONG_WINDOW_SECONDS],
        "long_weight": LONG_WEIGHT,
        "aggregate": baseline["aggregate"],
        "per_video": {
            key: _compact_score(value) for key, value in baseline["per_video"].items()
        },
        "fresh_live_cost": _fresh_live_cost(),
    }
    _atomic_json(args.run_dir / "baseline.json", baseline_artifact)

    lease_grid = _young_grid(lease=True)
    permanent_grid = _young_grid(lease=False)
    boundary_only_grid = _boundary_grid(HybridSpeakerTrackerConfig())
    total = len(lease_grid) + len(permanent_grid) + 2 * len(boundary_only_grid)
    rows: list[dict[str, Any]] = []
    completed = 0

    def evaluate(family: str, config: HybridSpeakerTrackerConfig) -> dict[str, Any]:
        nonlocal completed
        row = _score_candidate(
            prepared,
            baseline,
            family,
            config,
            score_tolerance=float(args.score_tolerance),
            wrong_tolerance=float(args.wrong_ratio_tolerance),
        )
        rows.append(row)
        _append_jsonl(args.run_dir / "trials.jsonl", row)
        completed += 1
        progress = {
            "completed": completed,
            "total": total,
            "percent": round(100.0 * completed / total, 2),
            "elapsed_seconds": time.monotonic() - started,
            "family": family,
            "candidate_id": row["candidate_id"],
        }
        _atomic_json(args.run_dir / "progress.json", progress)
        print(
            f"[{completed:03d}/{total:03d} {progress['percent']:6.2f}%] "
            f"{family} score={float(row['aggregate']['global_score']):.6f}",
            flush=True,
        )
        return row

    for config in lease_grid:
        evaluate("young_profile_lease", config)
    for config in permanent_grid:
        evaluate("young_profile_permanent", config)
    for config in boundary_only_grid:
        evaluate("boundary_only", config)

    simple = _best(row for row in rows if row["family"] == "young_profile_lease" and row["promotion_gates_passed"])
    combined_base = HybridSpeakerTrackerConfig(
        enable_young_profile_confirmation=True,
        enable_young_profile_lease=True,
        young_trusted_min_sentence_count=4,
        young_trusted_min_speech_seconds=8.0,
        young_min_similarity=0.35,
        young_min_margin=0.08,
        young_required_consecutive_probes=1,
        young_independent_scale_count=2,
        young_fast_independent_scale_count=1,
    )
    if simple is not None:
        combined_base = HybridSpeakerTrackerConfig(**simple["config"])
    for config in _boundary_grid(combined_base):
        evaluate("young_lease_plus_boundary", config)

    best_combined = _best(
        row
        for row in rows
        if row["family"] == "young_lease_plus_boundary" and row["promotion_gates_passed"]
    )
    winner = simple
    boundary_promoted = False
    if simple is not None and best_combined is not None and _boundary_materially_better(
        best_combined, simple, float(args.boundary_material_improvement)
    ):
        winner = best_combined
        boundary_promoted = True

    champion = {
        "status": "development_winner_locked" if winner is not None else "no_winner",
        "winner": winner,
        "sealed_holdout": SEALED_HOLDOUT,
        "sealed_holdout_opened": False,
        "fresh_live_verified": False,
        "production_wired": False,
    }
    _atomic_json(args.run_dir / "champion.json", champion)
    report = {
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "development_videos": list(DEVELOPMENT_VIDEOS),
        "sealed_holdout": SEALED_HOLDOUT,
        "sealed_holdout_opened": False,
        "baseline_global_score": float(baseline["aggregate"]["global_score"]),
        "candidate_count": len(rows),
        "promotion_gate_pass_count": sum(bool(row["promotion_gates_passed"]) for row in rows),
        "best_simple_probation": simple,
        "best_combined_boundary": best_combined,
        "boundary_promoted": boundary_promoted,
        "boundary_occam_rule": {
            "minimum_combined_improvement": float(args.boundary_material_improvement),
            "maximum_per_video_score_regression": 0.002,
            "maximum_per_video_wrong_ratio_increase": 0.002,
        },
        "winner": winner,
        "elapsed_seconds": time.monotonic() - started,
        "fresh_live_cost": _fresh_live_cost(),
    }
    _atomic_json(args.run_dir / "report.json", report)
    _atomic_json(args.run_dir / "run.json", {
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "provider": str(args.provider),
        "development_videos": list(DEVELOPMENT_VIDEOS),
        "validation_policy": "all opened videos are development; JWS is the only sealed holdout",
        "sealed_holdout": SEALED_HOLDOUT,
        "sealed_holdout_opened": False,
        "windows_seconds": [SHORT_WINDOW_SECONDS, LONG_WINDOW_SECONDS],
        "long_weight": LONG_WEIGHT,
        "candidate_count_planned": total,
        "fresh_live_cost": _fresh_live_cost(),
    })
    print(json.dumps({
        "status": champion["status"],
        "baseline": report["baseline_global_score"],
        "winner": None if winner is None else winner["aggregate"]["global_score"],
        "winner_id": None if winner is None else winner["candidate_id"],
        "boundary_promoted": boundary_promoted,
        "sealed_holdout_opened": False,
    }, indent=2))
    return 0 if winner is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
