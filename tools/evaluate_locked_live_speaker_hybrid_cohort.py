"""Evaluate one already-locked two-window hybrid on a fresh video cohort.

This program deliberately has no search space: it reads the exact candidate,
provider stack, window pair, and baseline settings from an existing locked run.
It never writes into that run and rejects the four videos that informed it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for value in (SRC, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_replay import Dataset, _trace_hash
import sweep_live_speaker_hybrid as sweep
from window.live_speaker_benchmark import (
    SCORER_ID,
    aggregate_video_scores,
    score_live_speaker_decisions,
)
from window.live_speaker_hybrid import (
    HYBRID_ALGORITHM_ID,
    HybridSpeakerTrackerConfig,
    replay_hybrid_decisions,
)
from window.live_speaker_replay import replay_cached_live_windows_dual


EVALUATOR_ID = "exact_locked_live_speaker_hybrid_cohort_v1"
_ALLOWED_LOCK_STATES = {
    "development_winner_locked",
    "cached_holdout_passed",
    "cached_holdout_failed",
}


@dataclass(frozen=True)
class LockedCandidate:
    run_dir: Path
    champion_path: Path
    run_path: Path
    status: str
    candidate_id: str
    family: str
    provider: str
    short_window_seconds: float
    long_window_seconds: float
    long_weight: float
    config: HybridSpeakerTrackerConfig
    champion_sha256: str
    run_sha256: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    locked = parser.add_mutually_exclusive_group(required=True)
    locked.add_argument(
        "--locked-run-dir",
        type=Path,
        help="Directory containing the immutable champion.json and run.json.",
    )
    locked.add_argument(
        "--locked-champion",
        type=Path,
        help="Exact champion.json; run.json is read from the same directory.",
    )
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.005)
    parser.add_argument("--wrong-ratio-tolerance", type=float, default=0.005)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _locked_paths(
    locked_run_dir: Path | None,
    locked_champion: Path | None,
) -> tuple[Path, Path, Path]:
    if (locked_run_dir is None) == (locked_champion is None):
        raise ValueError("provide exactly one locked run directory or champion path")
    if locked_champion is not None:
        champion_path = locked_champion.resolve()
        if champion_path.name != "champion.json":
            raise ValueError("--locked-champion must name champion.json")
        run_dir = champion_path.parent
    else:
        assert locked_run_dir is not None
        run_dir = locked_run_dir.resolve()
        champion_path = run_dir / "champion.json"
    return run_dir, champion_path, run_dir / "run.json"


def _load_locked_candidate(
    locked_run_dir: Path | None = None,
    locked_champion: Path | None = None,
) -> LockedCandidate:
    run_dir, champion_path, run_path = _locked_paths(
        locked_run_dir, locked_champion
    )
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))

    status = str(champion.get("status", ""))
    if status not in _ALLOWED_LOCK_STATES:
        raise ValueError(f"champion is not an immutable locked winner: {status!r}")
    winner = champion.get("winner")
    if not isinstance(winner, dict) or not bool(winner.get("promotion_gates_passed")):
        raise ValueError("champion does not contain a development-gated winner")
    if str(run.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("locked run uses a different hybrid algorithm")
    if str(winner.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("locked winner uses a different hybrid algorithm")

    family = str(winner.get("family", ""))
    if not family:
        raise ValueError("locked winner has no candidate family")
    raw_config = winner.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("locked winner has no exact configuration")
    config = HybridSpeakerTrackerConfig(**raw_config)
    expected_candidate_id = sweep._candidate_id(family, config)
    candidate_id = str(winner.get("candidate_id", ""))
    if candidate_id != expected_candidate_id:
        raise ValueError("locked candidate identity does not match its configuration")
    if not sweep._fast_latency_eligible(config):
        raise ValueError("locked candidate violates the fast temporary-lease contract")

    windows = run.get("windows_seconds")
    if not isinstance(windows, list) or len(windows) != 2:
        raise ValueError("locked run must contain exactly two fresh windows")
    normalized_windows = [round(float(value), 6) for value in windows]
    if len(set(normalized_windows)) != 2:
        raise ValueError("locked run windows must be distinct")
    if normalized_windows != [
        round(float(sweep.SHORT_WINDOW_SECONDS), 6),
        round(float(sweep.LONG_WINDOW_SECONDS), 6),
    ]:
        raise ValueError("locked run window pair differs from the scored baseline")
    fresh_cost = run.get("fresh_live_cost")
    if not isinstance(fresh_cost, dict):
        raise ValueError("locked run has no fresh-live cost contract")
    if int(fresh_cost.get("fresh_window_requests_per_probe", 99)) > 2:
        raise ValueError("locked run exceeds the two-window real-time budget")
    if int(fresh_cost.get("max_fresh_window_requests_per_probe", 99)) > 2:
        raise ValueError("locked run permits more than two fresh windows")

    provider = str(run.get("provider", "")).strip()
    if not provider:
        raise ValueError("locked run has no provider stack")
    long_weight = float(run.get("long_weight", sweep.LONG_WEIGHT))
    if abs(long_weight - float(sweep.LONG_WEIGHT)) > 1e-12:
        raise ValueError("locked run long-window weight differs from the baseline")

    return LockedCandidate(
        run_dir=run_dir,
        champion_path=champion_path,
        run_path=run_path,
        status=status,
        candidate_id=candidate_id,
        family=family,
        provider=provider,
        short_window_seconds=float(normalized_windows[0]),
        long_window_seconds=float(normalized_windows[1]),
        long_weight=long_weight,
        config=config,
        champion_sha256=_sha256_file(champion_path),
        run_sha256=_sha256_file(run_path),
    )


def _validate_fresh_video_ids(video_ids: Sequence[str]) -> list[str]:
    normalized = [str(value).strip() for value in video_ids]
    if not normalized or any(not value for value in normalized):
        raise ValueError("at least one non-empty --video-id is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate cohort video IDs are not allowed")
    previously_opened = set(sweep.DEVELOPMENT_VIDEOS) | {sweep.SEALED_HOLDOUT}
    overlap = sorted(previously_opened.intersection(normalized))
    if overlap:
        raise ValueError(
            "fresh cohort contains videos already used for design or holdout: "
            + ", ".join(overlap)
        )
    return normalized


def _assert_output_outside_locked_run(output: Path, locked_run_dir: Path) -> None:
    resolved_output = output.resolve()
    resolved_run = locked_run_dir.resolve()
    if resolved_output == resolved_run or resolved_output.is_relative_to(resolved_run):
        raise ValueError("cohort output must remain outside the immutable locked run")


def _cohort_gates(
    baseline_per_video: dict[str, dict[str, Any]],
    candidate_per_video: dict[str, dict[str, Any]],
    baseline_global_score: float,
    candidate_global_score: float,
    *,
    score_tolerance: float,
    wrong_tolerance: float,
) -> dict[str, Any]:
    if set(baseline_per_video) != set(candidate_per_video):
        raise ValueError("baseline and candidate cohorts differ")
    score_deltas = {
        video_id: float(candidate_per_video[video_id]["strict_browser_live_score"])
        - float(baseline["strict_browser_live_score"])
        for video_id, baseline in baseline_per_video.items()
    }
    wrong_deltas = {
        video_id: float(candidate_per_video[video_id]["wrong_live_speech_ratio"])
        - float(baseline["wrong_live_speech_ratio"])
        for video_id, baseline in baseline_per_video.items()
    }
    aggregate_delta = float(candidate_global_score) - float(baseline_global_score)
    aggregate_score_gate = aggregate_delta >= -float(score_tolerance) - 1e-12
    per_video_score_gate = all(
        value >= -float(score_tolerance) - 1e-12 for value in score_deltas.values()
    )
    per_video_wrong_gate = all(
        value <= float(wrong_tolerance) + 1e-12 for value in wrong_deltas.values()
    )
    return {
        "aggregate_score_delta_vs_baseline": aggregate_delta,
        "per_video_score_delta_vs_baseline": score_deltas,
        "per_video_wrong_ratio_delta_vs_baseline": wrong_deltas,
        "aggregate_score_gate_passed": aggregate_score_gate,
        "per_video_score_gate_passed": per_video_score_gate,
        "per_video_wrong_ratio_gate_passed": per_video_wrong_gate,
        "cohort_gate_passed": bool(
            aggregate_score_gate and per_video_score_gate and per_video_wrong_gate
        ),
    }


def _evaluate(
    dataset: Dataset,
    video_ids: Sequence[str],
    locked: LockedCandidate,
    *,
    score_tolerance: float,
    wrong_tolerance: float,
) -> dict[str, Any]:
    baseline_per_video: dict[str, dict[str, Any]] = {}
    candidate_per_video: dict[str, dict[str, Any]] = {}
    baseline_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for video_id in video_ids:
        inputs = dataset.video_inputs(video_id)
        short = dataset.block(video_id, locked.short_window_seconds)
        long = dataset.block(video_id, locked.long_window_seconds)
        baseline = replay_cached_live_windows_dual(
            short,
            long,
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            long_weight=locked.long_weight,
            config=sweep.BASELINE_CONFIG,
        )
        steps = sweep._build_steps(short, long, inputs)
        if any(len(step.evidences) > 2 for step in steps):
            raise AssertionError("cohort evaluation attempted more than two windows")
        candidate = replay_hybrid_decisions(
            baseline,
            steps,
            inputs["profiles"],
            config=locked.config,
        )
        baseline_per_video[video_id] = score_live_speaker_decisions(
            baseline, inputs["canonical"], inputs["profiles"]
        )
        candidate_per_video[video_id] = score_live_speaker_decisions(
            candidate, inputs["canonical"], inputs["profiles"]
        )
        baseline_hashes[video_id] = _trace_hash(baseline)
        candidate_hashes[video_id] = _trace_hash(candidate)

    baseline_aggregate = aggregate_video_scores(baseline_per_video.values())
    candidate_aggregate = aggregate_video_scores(candidate_per_video.values())
    gates = _cohort_gates(
        baseline_per_video,
        candidate_per_video,
        float(baseline_aggregate["global_score"]),
        float(candidate_aggregate["global_score"]),
        score_tolerance=score_tolerance,
        wrong_tolerance=wrong_tolerance,
    )
    return {
        "baseline": {
            "aggregate": baseline_aggregate,
            "per_video": {
                key: sweep._compact_score(value)
                for key, value in baseline_per_video.items()
            },
            "trace_hashes": baseline_hashes,
        },
        "candidate": {
            "aggregate": candidate_aggregate,
            "per_video": {
                key: sweep._compact_score(value)
                for key, value in candidate_per_video.items()
            },
            "trace_hashes": candidate_hashes,
        },
        "gates": gates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if float(args.score_tolerance) < 0.0 or float(args.wrong_ratio_tolerance) < 0.0:
        raise ValueError("gate tolerances must be non-negative")
    locked = _load_locked_candidate(args.locked_run_dir, args.locked_champion)
    video_ids = _validate_fresh_video_ids(args.video_id)
    _assert_output_outside_locked_run(args.output, locked.run_dir)

    started = time.monotonic()
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(), locked.provider
    )
    result = _evaluate(
        dataset,
        video_ids,
        locked,
        score_tolerance=float(args.score_tolerance),
        wrong_tolerance=float(args.wrong_ratio_tolerance),
    )
    artifact = {
        "evaluator_id": EVALUATOR_ID,
        "evaluation_mode": "exact_locked_config_only",
        "parameter_search_performed": False,
        "post_holdout_tuning_performed": False,
        "fresh_cohort": True,
        "video_ids": list(video_ids),
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "candidate_id": locked.candidate_id,
        "candidate_family": locked.family,
        "config": asdict(locked.config),
        "provider": locked.provider,
        "windows_seconds": [
            locked.short_window_seconds,
            locked.long_window_seconds,
        ],
        "long_weight": locked.long_weight,
        "fresh_live_cost": sweep._fresh_live_cost(),
        "locked_source": {
            "status": locked.status,
            "run_dir": str(locked.run_dir),
            "champion_path": str(locked.champion_path),
            "champion_sha256": locked.champion_sha256,
            "run_path": str(locked.run_path),
            "run_sha256": locked.run_sha256,
        },
        "gate_tolerances": {
            "score": float(args.score_tolerance),
            "wrong_ratio": float(args.wrong_ratio_tolerance),
        },
        **result,
        "elapsed_seconds": time.monotonic() - started,
        "fresh_live_verified": False,
        "production_wired": False,
        "does_not_reopen_or_modify_original_holdout": True,
    }
    _atomic_json(args.output.resolve(), artifact)
    summary = {
        "candidate_id": locked.candidate_id,
        "video_count": len(video_ids),
        "baseline_global_score": artifact["baseline"]["aggregate"]["global_score"],
        "candidate_global_score": artifact["candidate"]["aggregate"]["global_score"],
        **artifact["gates"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if bool(artifact["gates"]["cohort_gate_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
