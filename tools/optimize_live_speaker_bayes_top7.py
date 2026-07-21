"""Resumable Top-7 sweep for the causal Bayesian speaker-state filter."""

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
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_overnight_top7 import Dataset, evaluate_candidate
from window.live_speaker_bayes import (
    BAYES_ALGORITHM_ID,
    BayesSpeakerTrackerConfig,
    replay_cached_bayes_windows,
)
from window.live_speaker_benchmark import (
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)


OPTIMIZER_ID = "live_speaker_top7_bayesian_state_filter_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _compact(score: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in score.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    for key in ("sampled_playback_seconds", "speaker_map"):
        if key in score:
            result[key] = score[key]
    for key, omitted in (("turn_latency", "turns"), ("release", "events")):
        if isinstance(score.get(key), dict):
            result[key] = {name: value for name, value in score[key].items() if name != omitted}
    return result


def _rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["aggregate"]["primary_score"]), reverse=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--max-candidates", type=int, default=1400)
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
    provider = str(champion["provider_spec"])
    profile_name = str(champion["profile_name"])
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name)
    baseline = evaluate_candidate(dataset, videos, champion)
    baseline_score = float(baseline["aggregate"]["primary_score"])

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for line in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["candidate_id"])] = row
    elif trials_path.exists():
        raise FileExistsError(f"{trials_path} exists; pass --resume")
    incumbent = _rank(completed.values())[0] if completed else None
    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase_counts[str(row["phase"])] = phase_counts.get(str(row["phase"]), 0) + 1

    def write_state(phase: str, active: str = "") -> None:
        best = float(incumbent["aggregate"]["primary_score"]) if incumbent else baseline_score
        _atomic_json(run_dir / "progress.json", {
            "status": "interrupted" if _STOP else "running",
            "phase": phase,
            "active": active,
            "completed_candidate_count": len(completed),
            "phase_counts": phase_counts,
            "baseline_score": baseline_score,
            "best_bayes_score": best if incumbent else None,
            "score_delta": round(best - baseline_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "maximum_fresh_windows_per_probe": 2,
        })
        if incumbent:
            _atomic_json(run_dir / "champion.json", {
                "status": "CACHE_BAYES_WINNER_PENDING_PRODUCTION_INTEGRATION" if best > baseline_score else "BELOW_BASELINE",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "baseline_score": baseline_score,
                "candidate_score": best,
                "score_delta": round(best - baseline_score, 6),
                **incumbent,
                "fresh_live_verified": False,
            })

    def evaluate(
        windows: Sequence[float],
        config: BayesSpeakerTrackerConfig,
        phase: str,
        hypothesis: str,
        parent: str | None = None,
    ) -> dict[str, Any] | None:
        nonlocal incumbent
        windows = tuple(round(float(value), 3) for value in windows)
        if len(windows) != 2:
            raise ValueError("Bayesian campaign requires exactly two windows")
        config = replace(config, scale_windows=windows)
        candidate_id = _stable_id({
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": BAYES_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "windows": windows,
            "config": asdict(config),
        })
        if candidate_id in completed:
            return completed[candidate_id]
        if _STOP or time.monotonic() >= deadline or len(completed) >= int(args.max_candidates):
            return None
        write_state(phase, candidate_id)
        per_video: dict[str, Any] = {}
        short = min(windows)
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, short)
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        if not math.isfinite(float(aggregate["primary_score"])):
            raise RuntimeError("Non-finite Bayesian score")
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "hypothesis": hypothesis,
            "parent_candidate_id": parent,
            "provider_spec": provider,
            "profile_name": profile_name,
            "windows_seconds": list(windows),
            "algorithm_config": asdict(config),
            "aggregate": aggregate,
            "per_video": per_video,
            "score_delta_vs_baseline": round(float(aggregate["primary_score"]) - baseline_score, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[candidate_id] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        _append_jsonl(trials_path, row)
        if incumbent is None or float(aggregate["primary_score"]) > float(incumbent["aggregate"]["primary_score"]):
            incumbent = row
        write_state(phase, candidate_id)
        return row

    _atomic_json(run_dir / "run.json", {
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": BAYES_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "baseline_description": champion,
        "baseline_score": baseline_score,
        "videos": videos,
        "maximum_fresh_windows_per_probe": 2,
        "selection_policy": "one Top-7 primary score",
    })
    _atomic_json(run_dir / "baseline_reproduction.json", {
        "status": "REPRODUCED",
        "aggregate": baseline["aggregate"],
        "per_video": baseline["per_video"],
    })
    write_state("BASELINE")

    # First isolate the observation model: no temporal prior (prior_strength=0).
    likelihood_rows: list[dict[str, Any]] = []
    window_pairs = ((0.7, 1.5), (0.7, 2.9), (0.8, 2.8), (0.9, 2.9))
    weights = ((0.9, 0.1), (0.8, 0.2), (0.65, 0.35), (0.5, 0.5))
    thresholds = ((0.20, 0.00), (0.25, 0.00), (0.25, 0.03), (0.28, 0.03), (0.30, 0.00), (0.30, 0.03), (0.30, 0.05), (0.35, 0.05))
    for windows in window_pairs:
        for scale_weights in weights:
            for minimum, margin in thresholds:
                for temperature in (0.05, 0.075, 0.10):
                    for known_probability in (0.35, 0.50):
                        config = BayesSpeakerTrackerConfig(
                            scale_windows=windows,
                            scale_weights=scale_weights,
                            min_similarity=minimum,
                            min_margin=margin,
                            similarity_temperature=temperature,
                            stay_probability=0.5,
                            prior_strength=0.0,
                            evidence_strength=1.0,
                            min_known_probability=known_probability,
                            unknown_release_count=2,
                            silence_release_count=2,
                        )
                        row = evaluate(
                            windows, config, "STAGE_1_LIKELIHOOD",
                            "Calibrate independent two-window observation likelihoods without temporal persistence.",
                        )
                        if row:
                            likelihood_rows.append(row)

    # Then add a true Markov prior only to strong observation models.
    temporal_rows: list[dict[str, Any]] = []
    for parent_row in _rank(likelihood_rows)[:12]:
        windows = tuple(parent_row["windows_seconds"])
        source = BayesSpeakerTrackerConfig(**parent_row["algorithm_config"])
        for stay_probability in (0.55, 0.70, 0.80, 0.90, 0.95):
            for prior_strength in (0.50, 1.00, 1.50):
                config = replace(
                    source,
                    stay_probability=stay_probability,
                    prior_strength=prior_strength,
                )
                row = evaluate(
                    windows, config, "STAGE_2_MARKOV_PRIOR",
                    "Use a probabilistic continuation prior that strong change evidence can override immediately.",
                    parent_row["candidate_id"],
                )
                if row:
                    temporal_rows.append(row)

    # Local posterior gates: these change display policy without changing the filter.
    posterior_rows: list[dict[str, Any]] = []
    for parent_row in _rank(temporal_rows + likelihood_rows)[:10]:
        windows = tuple(parent_row["windows_seconds"])
        source = BayesSpeakerTrackerConfig(**parent_row["algorithm_config"])
        for probability_margin in (0.00, 0.03, 0.06, 0.10, 0.15):
            for unknown_bias in (-0.50, 0.00, 0.50, 1.00):
                config = replace(
                    source,
                    switch_probability_margin=probability_margin,
                    unknown_bias=unknown_bias,
                )
                row = evaluate(
                    windows, config, "STAGE_3_POSTERIOR_GATE",
                    "Tune only the UNKNOWN and posterior-separation decisions around the learned temporal belief.",
                    parent_row["candidate_id"],
                )
                if row:
                    posterior_rows.append(row)

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    best = float(incumbent["aggregate"]["primary_score"]) if incumbent else baseline_score
    _atomic_json(run_dir / "final_report.json", {
        "status": progress["status"],
        "baseline_score": baseline_score,
        "champion_score": best if incumbent else None,
        "score_delta": round(best - baseline_score, 6),
        "candidate_count": len(completed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps(progress, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
