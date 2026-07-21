"""Search causal cohort score-normalization around a verified Bayes champion."""

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BAYES_ALGORITHM_ID, BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import PRIMARY_SCORER_V2_ID, aggregate_video_scores_primary_v2, score_live_speaker_decisions

OPTIMIZER_ID = "live_speaker_top7_bayes_cohort_normalization_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _candidate_id(config: BayesSpeakerTrackerConfig, windows: tuple[float, ...]) -> str:
    payload = json.dumps({
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": BAYES_ALGORITHM_ID,
        "scorer_id": PRIMARY_SCORER_V2_ID,
        "windows": windows,
        "config": asdict(config),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _args() -> argparse.Namespace:
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
    args = _args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, args.budget_seconds)
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    provider = str(source["provider_spec"])
    profile_name = str(source["profile_name"])
    source_score = float(source["candidate_score"])
    source_config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
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
    incumbent = max(completed.values(), key=lambda row: row["aggregate"]["primary_score"], default=None)
    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    def write_state(phase: str, active: str = "") -> None:
        best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
        _atomic(run_dir / "progress.json", {
            "status": "interrupted" if _STOP else "running", "phase": phase, "active": active,
            "completed_candidate_count": len(completed), "phase_counts": phase_counts,
            "source_champion_score": source_score, "best_cohort_score": best if incumbent else None,
            "score_delta": round(best - source_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        if incumbent:
            _atomic(run_dir / "champion.json", {
                "status": "CACHE_COHORT_BAYES_WINNER_PENDING_FRESH_LIVE" if best > source_score else "BELOW_SOURCE_CHAMPION",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "source_champion_score": source_score, "candidate_score": best,
                "score_delta": round(best - source_score, 6), "provider_spec": provider,
                "profile_name": profile_name, "windows_seconds": list(windows),
                **incumbent, "fresh_live_verified": False,
            })

    def evaluate(config: BayesSpeakerTrackerConfig, phase: str, hypothesis: str, parent: str | None = None) -> dict[str, Any] | None:
        nonlocal incumbent
        cid = _candidate_id(config, windows)
        if cid in completed:
            return completed[cid]
        if _STOP or time.monotonic() >= deadline:
            return None
        write_state(phase, cid)
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows], inputs["profiles"],
                inputs["speech"], inputs["probes"], inputs["releases"], config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"]))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        row = {
            "candidate_id": cid, "phase": phase, "hypothesis": hypothesis,
            "parent_candidate_id": parent, "algorithm_config": asdict(config),
            "aggregate": aggregate, "per_video": per_video,
            "score_delta_vs_source": round(float(aggregate["primary_score"]) - source_score, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[cid] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if incumbent is None or aggregate["primary_score"] > incumbent["aggregate"]["primary_score"]:
            incumbent = row
        write_state(phase, cid)
        return row

    _atomic(run_dir / "run.json", {
        "optimizer_id": OPTIMIZER_ID, "algorithm_id": BAYES_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID, "source_champion": str(args.champion.resolve()),
        "source_champion_score": source_score, "videos": videos,
        "maximum_fresh_windows_per_probe": 2, "selection_policy": "one Top-7 primary score",
    })
    write_state("SOURCE_CHAMPION")
    values = (-1.0, -0.75, -0.50, -0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
    mean_rows: list[dict[str, Any]] = []
    max_rows: list[dict[str, Any]] = []
    for value in values:
        row = evaluate(replace(source_config, profile_cohort_mean_strength=value), "STAGE_1_MEAN_COHORT", "Normalize each speaker score by its mean causal profile-cohort similarity.")
        if row:
            mean_rows.append(row)
        row = evaluate(replace(source_config, profile_cohort_max_strength=value), "STAGE_2_MAX_COHORT", "Normalize each speaker score by its closest causal impostor profile.")
        if row:
            max_rows.append(row)
    rank = lambda rows: sorted(rows, key=lambda row: row["aggregate"]["primary_score"], reverse=True)[:5]
    for mean_parent in rank(mean_rows):
        for max_parent in rank(max_rows):
            mean_config = BayesSpeakerTrackerConfig(**mean_parent["algorithm_config"])
            max_config = BayesSpeakerTrackerConfig(**max_parent["algorithm_config"])
            evaluate(replace(mean_config, profile_cohort_max_strength=max_config.profile_cohort_max_strength), "STAGE_3_COMBINED_COHORT", "Combine mean and nearest-impostor causal score normalization.", mean_parent["candidate_id"])
    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress.update(status="interrupted" if _STOP else "complete", phase="INTERRUPTED" if _STOP else "COMPLETE")
    _atomic(run_dir / "progress.json", progress)
    best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
    _atomic(run_dir / "final_report.json", {
        "status": progress["status"], "source_champion_score": source_score,
        "champion_score": best if incumbent else None, "score_delta": round(best - source_score, 6),
        "candidate_count": len(completed), "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps(progress, indent=2))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
