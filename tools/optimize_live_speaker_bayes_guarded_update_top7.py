"""Tune continuity-gated provisional centroid adaptation on Top-7."""

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
    parser.add_argument("--history-refine", action="store_true")
    parser.add_argument("--history-fine", action="store_true")
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
    variants = [(0.0, -1.0, 1)]
    if args.history_fine:
        variants.extend(itertools.product(
            tuple(round(0.50 + 0.025 * index, 3) for index in range(13)),
            (-1.0, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20),
            (2, 3, 4),
        ))
    elif args.history_refine:
        variants.extend(itertools.product(
            (0.30, 0.50, 0.65, 0.80, 1.00),
            (0.10, 0.15, 0.20, 0.25, 0.30),
            (1, 2, 3, 4, 5, 6),
        ))
    elif args.refine:
        variants.extend(itertools.product(
            tuple(round(0.30 + 0.05 * index, 2) for index in range(15)),
            tuple(round(0.10 + 0.025 * index, 3) for index in range(10)),
            (1,),
        ))
    else:
        variants.extend(itertools.product(
            (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50),
            (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
            (1,),
        ))
    gate_root = args.gate_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for alpha, threshold, history_size in variants:
        config = replace(
            base,
            provisional_update_alpha=alpha,
            provisional_update_continuity_min_similarity=threshold,
            provisional_update_history_size=history_size,
        )
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            video_gate = gate_root / video_id
            speech = np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False)
            probes = np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False)
            releases = np.load(video_gate / "release_gate.u1.npy", allow_pickle=False)
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], speech, probes, releases, config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        row = {
            "provisional_update_alpha": alpha,
            "provisional_update_continuity_min_similarity": threshold,
            "provisional_update_history_size": history_size,
            "gate_variant": source.get("gate_variant"),
            "vad_frame_seconds": source.get("vad_frame_seconds"),
            "vad_speech_rms_threshold": source.get("vad_speech_rms_threshold"),
            "live_speaker_clear_window_seconds": source.get("live_speaker_clear_window_seconds"),
            "live_speaker_probe_min_speech_seconds": source.get("live_speaker_probe_min_speech_seconds"),
            "release_every_tick": True,
            "silence_release_count": int(config.silence_release_count),
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
            "status": "running", "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [alpha, threshold, history_size],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_GUARDED_UPDATE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        **best,
        "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "total_candidate_count": len(rows), "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
