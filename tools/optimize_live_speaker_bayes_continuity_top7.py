"""Test a causal short-window incumbent continuity memory on Top-7."""

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
    variants = [(-1.0, 3, False)]
    if args.refine:
        variants.extend(itertools.product(
            tuple(round(0.26 + 0.01 * index, 2) for index in range(13)),
            (4, 5, 6, 7, 8, 10),
            (False, True),
        ))
    else:
        variants.extend(itertools.product(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
            (1, 3, 5),
            (False, True),
        ))
    gate_root = args.gate_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for threshold, history_size, update_on_hold in variants:
        config = replace(
            base,
            incumbent_continuity_min_similarity=float(threshold),
            incumbent_continuity_history_size=int(history_size),
            incumbent_continuity_update_on_hold=bool(update_on_hold),
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
            "incumbent_continuity_min_similarity": threshold,
            "incumbent_continuity_history_size": history_size,
            "incumbent_continuity_update_on_hold": update_on_hold,
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
            "active": [threshold, history_size, update_on_hold],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_CONTINUITY_WINNER_PENDING_FRESH_LIVE",
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
