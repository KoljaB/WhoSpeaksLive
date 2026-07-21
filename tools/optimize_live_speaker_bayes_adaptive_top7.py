"""Top-7 sweep for causal profile-count-adaptive Bayesian calibration."""

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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BAYES_ALGORITHM_ID, BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import PRIMARY_SCORER_V2_ID, aggregate_video_scores_primary_v2, score_live_speaker_decisions


OPTIMIZER_ID = "live_speaker_top7_bayes_profile_count_adaptation_v1"
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source_champion = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    provider = str(source_champion["provider_spec"])
    profile_name = str(source_champion["profile_name"])
    windows = tuple(float(value) for value in source_champion["windows_seconds"])
    source_config = BayesSpeakerTrackerConfig(**source_champion["algorithm_config"])
    source_score = float(source_champion["candidate_score"])
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name)

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
        best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
        _atomic_json(run_dir / "progress.json", {
            "status": "interrupted" if _STOP else "running",
            "phase": phase,
            "active": active,
            "completed_candidate_count": len(completed),
            "phase_counts": phase_counts,
            "source_champion_score": source_score,
            "best_adaptive_score": best if incumbent else None,
            "score_delta": round(best - source_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "maximum_fresh_windows_per_probe": 2,
        })
        if incumbent:
            _atomic_json(run_dir / "champion.json", {
                "status": "CACHE_ADAPTIVE_BAYES_WINNER_PENDING_FRESH_LIVE" if best > source_score else "BELOW_SOURCE_CHAMPION",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "source_champion_score": source_score,
                "candidate_score": best,
                "score_delta": round(best - source_score, 6),
                "provider_spec": provider,
                "profile_name": profile_name,
                "windows_seconds": list(windows),
                **incumbent,
                "fresh_live_verified": False,
            })

    def evaluate(config: BayesSpeakerTrackerConfig, phase: str, hypothesis: str, parent: str | None = None) -> dict[str, Any] | None:
        nonlocal incumbent
        candidate_id = _stable_id({
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": BAYES_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "windows": windows,
            "config": asdict(config),
        })
        if candidate_id in completed:
            return completed[candidate_id]
        if _STOP or time.monotonic() >= deadline:
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
            raise RuntimeError("Non-finite adaptive Bayesian score")
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "hypothesis": hypothesis,
            "parent_candidate_id": parent,
            "algorithm_config": asdict(config),
            "aggregate": aggregate,
            "per_video": per_video,
            "score_delta_vs_source": round(float(aggregate["primary_score"]) - source_score, 6),
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
        "source_champion": str(args.champion.resolve()),
        "source_champion_score": source_score,
        "videos": videos,
        "maximum_fresh_windows_per_probe": 2,
        "selection_policy": "one Top-7 primary score",
    })
    write_state("SOURCE_CHAMPION")

    adaptive_rows: list[dict[str, Any]] = []
    for threshold in range(1, 9):
        for low_bias in (-1.00, -0.75, -0.50, -0.25, 0.00):
            for high_bias in (-0.25, 0.00, 0.25, 0.50, 0.75):
                row = evaluate(
                    replace(
                        source_config,
                        profile_count_bias_threshold=threshold,
                        low_profile_unknown_bias=low_bias,
                        high_profile_unknown_bias=high_bias,
                        profile_count_unknown_bias_slope=0.0,
                    ),
                    "STAGE_1_PIECEWISE_PROFILE_COUNT",
                    "Use one UNKNOWN calibration below a causal active-profile threshold and another above it.",
                )
                if row:
                    adaptive_rows.append(row)

    slope_rows: list[dict[str, Any]] = []
    for intercept in (-1.00, -0.75, -0.50, -0.25, 0.00, 0.25):
        for slope in (-0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00):
            row = evaluate(
                replace(
                    source_config,
                    unknown_bias=intercept,
                    profile_count_bias_threshold=0,
                    profile_count_unknown_bias_slope=slope,
                ),
                "STAGE_2_LOG_PROFILE_COUNT",
                "Adjust UNKNOWN log-odds smoothly by the logarithm of causal active profile count.",
            )
            if row:
                slope_rows.append(row)

    # Locally recalibrate likelihood temperature and acceptance probability only
    # around the strongest adaptive rules.
    local_rows: list[dict[str, Any]] = []
    for parent_row in _rank(adaptive_rows + slope_rows)[:10]:
        parent_config = BayesSpeakerTrackerConfig(**parent_row["algorithm_config"])
        for temperature in (0.075, 0.0875, 0.10, 0.1125, 0.125):
            for known_probability in (0.40, 0.45, 0.50, 0.55, 0.60):
                row = evaluate(
                    replace(
                        parent_config,
                        similarity_temperature=temperature,
                        min_known_probability=known_probability,
                    ),
                    "STAGE_3_LOCAL_CALIBRATION",
                    "Recalibrate evidence temperature and display probability around an adaptive winner.",
                    parent_row["candidate_id"],
                )
                if row:
                    local_rows.append(row)

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
    _atomic_json(run_dir / "final_report.json", {
        "status": progress["status"],
        "source_champion_score": source_score,
        "champion_score": best if incumbent else None,
        "score_delta": round(best - source_score, 6),
        "candidate_count": len(completed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps(progress, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
