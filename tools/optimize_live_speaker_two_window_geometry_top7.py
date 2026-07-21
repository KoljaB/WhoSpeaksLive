"""Retune exactly two embedding windows after crossover logic changes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import aggregate_video_scores_primary_v2, score_live_speaker_decisions


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    short = min(float(value) for value in source["windows_seconds"])
    base = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    long_windows = (0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0)
    long_weights = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
    variants = list(itertools.product(long_windows, long_weights))
    gate_root = args.gate_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for long_window, long_weight in variants:
        config = replace(
            base,
            scale_windows=(short, long_window),
            scale_weights=(1.0 - long_weight, long_weight),
        )
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, short)
            video_gate = gate_root / video_id
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, short), dataset.block(video_id, long_window)],
                inputs["profiles"],
                np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False),
                np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False),
                np.load(video_gate / "release_gate.u1.npy", allow_pickle=False),
                config=config,
            )
            per_video[video_id] = _compact(
                score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"])
            )
        row = {
            "long_window_seconds": long_window,
            "long_weight": long_weight,
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"],
            "profile_name": source["profile_name"],
            "windows_seconds": [short, long_window],
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video,
        }
        rows.append(row)
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [long_window, long_weight],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_TWO_WINDOW_GEOMETRY_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "total_candidate_count": len(rows), "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
