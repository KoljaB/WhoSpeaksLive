"""Evaluate cheap dense RMS release checks without increasing embedding cadence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dense-gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    source_config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for release_count in range(1, 13):
        config = replace(source_config, silence_release_count=release_count)
        per_video: dict[str, Any] = {}
        release_counts: dict[str, int] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            dense_root = args.dense_gate_root / video_id
            dense_speech = np.load(dense_root / "speech_gate.u1.npy", allow_pickle=False)
            dense_probes = np.load(dense_root / "probe_schedule.u1.npy", allow_pickle=False)
            if not np.array_equal(dense_speech, inputs["speech"]):
                raise RuntimeError(f"Dense speech gate differs for {video_id}")
            if not np.array_equal(dense_probes, inputs["probes"]):
                raise RuntimeError(f"Dense probe schedule differs for {video_id}")
            releases = np.load(dense_root / "release_gate.u1.npy", allow_pickle=False)
            release_counts[video_id] = int(np.count_nonzero(releases))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], inputs["speech"], inputs["probes"], releases, config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        row = {
            "release_every_tick": True,
            "silence_release_count": release_count,
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"], "profile_name": source["profile_name"],
            "windows_seconds": list(windows),
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video, "release_signal_counts": release_counts,
        }
        rows.append(row)
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "best_score": max(float(item["aggregate"]["primary_score"]) for item in rows),
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_DENSE_RELEASE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6),
        **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "best_score": best["aggregate"]["primary_score"],
        "best_silence_release_count": best["silence_release_count"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
