"""Open the sealed JWS holdout once for an already locked hybrid winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for value in (SRC, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_replay import Dataset, _trace_hash
import sweep_live_speaker_hybrid as sweep
from window.live_speaker_benchmark import score_live_speaker_decisions
from window.live_speaker_hybrid import HybridSpeakerTrackerConfig, replay_hybrid_decisions
from window.live_speaker_replay import replay_cached_live_windows_dual


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--locked-run-dir", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.005)
    parser.add_argument("--wrong-ratio-tolerance", type=float, default=0.005)
    return parser.parse_args()


def _holdout_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    score_tolerance: float,
    wrong_tolerance: float,
) -> dict[str, Any]:
    score_delta = (
        float(candidate["strict_browser_live_score"])
        - float(baseline["strict_browser_live_score"])
    )
    wrong_delta = (
        float(candidate["wrong_live_speech_ratio"])
        - float(baseline["wrong_live_speech_ratio"])
    )
    score_gate = score_delta >= -float(score_tolerance) - 1e-12
    wrong_gate = wrong_delta <= float(wrong_tolerance) + 1e-12
    return {
        "score_delta_vs_baseline": score_delta,
        "wrong_ratio_delta_vs_baseline": wrong_delta,
        "score_gate_passed": score_gate,
        "wrong_ratio_gate_passed": wrong_gate,
        "holdout_passed": bool(score_gate and wrong_gate),
    }


def main() -> int:
    args = _parse_args()
    champion_path = args.locked_run_dir / "champion.json"
    run_path = args.locked_run_dir / "run.json"
    holdout_path = args.locked_run_dir / "holdout.json"
    if holdout_path.exists():
        raise RuntimeError("sealed holdout was already opened for this locked run")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if champion.get("status") != "development_winner_locked":
        raise ValueError("holdout verification requires a locked development winner")
    if champion.get("sealed_holdout") != sweep.SEALED_HOLDOUT:
        raise ValueError("locked run names a different holdout")
    if bool(champion.get("sealed_holdout_opened")):
        raise RuntimeError("sealed holdout is already marked as opened")
    winner = champion.get("winner")
    if not isinstance(winner, dict) or not bool(winner.get("promotion_gates_passed")):
        raise ValueError("locked winner did not pass development promotion gates")
    config = HybridSpeakerTrackerConfig(**winner["config"])
    if not sweep._fast_latency_eligible(config):
        raise ValueError("locked winner violates the fast temporary-lease contract")

    dataset = Dataset(args.corpus_root, args.input_root, str(run["provider"]))
    video_id = sweep.SEALED_HOLDOUT
    inputs = dataset.video_inputs(video_id)
    short = dataset.block(video_id, sweep.SHORT_WINDOW_SECONDS)
    long = dataset.block(video_id, sweep.LONG_WINDOW_SECONDS)
    baseline_decisions = replay_cached_live_windows_dual(
        short,
        long,
        inputs["profiles"],
        inputs["speech"],
        inputs["probes"],
        inputs["releases"],
        long_weight=sweep.LONG_WEIGHT,
        config=sweep.BASELINE_CONFIG,
    )
    hybrid_decisions = replay_hybrid_decisions(
        baseline_decisions,
        sweep._build_steps(short, long, inputs),
        inputs["profiles"],
        config=config,
    )
    baseline_score = score_live_speaker_decisions(
        baseline_decisions, inputs["canonical"], inputs["profiles"]
    )
    candidate_score = score_live_speaker_decisions(
        hybrid_decisions, inputs["canonical"], inputs["profiles"]
    )
    gates = _holdout_gates(
        baseline_score,
        candidate_score,
        score_tolerance=float(args.score_tolerance),
        wrong_tolerance=float(args.wrong_ratio_tolerance),
    )
    artifact = {
        "selection_locked_before_holdout": True,
        "candidate_id": winner["candidate_id"],
        "config": winner["config"],
        "video_id": video_id,
        "baseline": sweep._compact_score(baseline_score),
        "candidate": sweep._compact_score(candidate_score),
        **gates,
        "baseline_trace_hash": _trace_hash(baseline_decisions),
        "candidate_trace_hash": _trace_hash(hybrid_decisions),
        "fresh_live_cost": sweep._fresh_live_cost(),
        "fresh_live_verified": False,
        "production_wired": False,
        "no_post_holdout_tuning_allowed": True,
    }
    sweep._atomic_json(holdout_path, artifact)
    champion["sealed_holdout_opened"] = True
    champion["holdout_passed"] = bool(gates["holdout_passed"])
    champion["holdout_artifact"] = "holdout.json"
    champion["status"] = (
        "cached_holdout_passed" if gates["holdout_passed"] else "cached_holdout_failed"
    )
    sweep._atomic_json(champion_path, champion)
    print(json.dumps({
        "status": champion["status"],
        "candidate_id": winner["candidate_id"],
        "baseline_score": baseline_score["strict_browser_live_score"],
        "candidate_score": candidate_score["strict_browser_live_score"],
        **gates,
    }, indent=2))
    return 0 if gates["holdout_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
