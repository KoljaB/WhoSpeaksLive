"""Tune change-point-gated discovery of previously unseen live speakers."""

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
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--ultrafine", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    base = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    variants = [(-1.0, -1.0)]
    if args.ultrafine:
        ceilings = (0.121, 0.123, 0.125, 0.127, 0.129, 0.131, 0.133)
        continuities = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
    elif args.refine:
        ceilings = (0.115, 0.12, 0.125, 0.13, 0.135)
        continuities = (-0.10, -0.05, 0.0, 0.05)
    else:
        ceilings = (0.11, 0.125, 0.14, 0.155, 0.175)
        continuities = (0.0, 0.10, 0.20, 0.25, 0.30)
    variants.extend(itertools.product(ceilings, continuities))
    gate_root = args.gate_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for boundary_ceiling, boundary_continuity in variants:
        config = replace(
            base,
            provisional_boundary_creation_similarity_ceiling=boundary_ceiling,
            provisional_boundary_continuity_max_similarity=boundary_continuity,
        )
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            video_gate = gate_root / video_id
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
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
            "provisional_boundary_creation_similarity_ceiling": boundary_ceiling,
            "provisional_boundary_continuity_max_similarity": boundary_continuity,
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"],
            "profile_name": source["profile_name"],
            "windows_seconds": list(windows),
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video,
        }
        rows.append(row)
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running",
            "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [boundary_ceiling, boundary_continuity],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_BOUNDARY_DISCOVERY_WINNER_PENDING_FRESH_LIVE",
        "selection_policy": "primary_score_only_no_per_video_vetoes",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        "hypothesis": (
            "Permit moderately less conservative unknown-speaker creation only when "
            "short-window history independently indicates a speaker boundary."
        ),
        **best,
        "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete",
        "completed_candidate_count": len(rows),
        "total_candidate_count": len(rows),
        "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
