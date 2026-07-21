"""Add a conservative fast-silence path beside the stable long Silero release gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from build_live_speaker_gate_tapes import _atomic_json, _atomic_npy
from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from optimize_live_speaker_silero_gate_top7 import _speech_mask
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
    parser.add_argument("--long-gate-root", type=Path, required=True)
    parser.add_argument("--schedule-gate-root", type=Path)
    parser.add_argument("--short-probability-root", type=Path, required=True)
    parser.add_argument("--output-gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--ultrafine", action="store_true")
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--fixed-minimum", type=float)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    config = BayesSpeakerTrackerConfig.from_mapping(source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    cached: dict[str, dict[str, Any]] = {}
    schedule_gate_root = (
        args.schedule_gate_root.resolve()
        if args.schedule_gate_root is not None
        else args.long_gate_root.resolve()
    )
    for video_id in videos:
        long_gate = args.long_gate_root.resolve() / video_id
        schedule_gate = schedule_gate_root / video_id
        probability_dir = args.short_probability_root.resolve() / video_id
        metadata = json.loads(
            (probability_dir / "probability_tape.json").read_text(encoding="utf-8-sig")
        )
        cached[video_id] = {
            "speech": np.load(schedule_gate / "speech_gate.u1.npy", allow_pickle=False),
            "probes": np.load(schedule_gate / "probe_schedule.u1.npy", allow_pickle=False),
            "long_release": np.load(long_gate / "release_gate.u1.npy", allow_pickle=False),
            "short_probabilities": np.load(
                probability_dir / "release_probabilities.f32.npy", allow_pickle=False
            ),
            "short_window": float(metadata["clear_window_seconds"]),
        }
    variants: list[tuple[float | None, float | None]] = [(None, None)]
    if args.fixed_threshold is not None or args.fixed_minimum is not None:
        if args.fixed_threshold is None or args.fixed_minimum is None:
            raise ValueError("--fixed-threshold and --fixed-minimum must be provided together")
        thresholds = (float(args.fixed_threshold),)
        minima = (float(args.fixed_minimum),)
    elif args.ultrafine:
        thresholds = (0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018)
        minima = (0.120, 0.128, 0.136, 0.144, 0.152, 0.160, 0.168)
    elif args.refine:
        thresholds = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03)
        minima = (0.064, 0.080, 0.096, 0.112, 0.128, 0.144, 0.160)
    else:
        thresholds = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
        minima = (0.032, 0.064, 0.096, 0.128, 0.160)
    variants.extend(itertools.product(thresholds, minima))
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[float | None, float | None], dict[str, np.ndarray]] = {}
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for threshold, minimum in variants:
        per_video: dict[str, Any] = {}
        variant_masks: dict[str, np.ndarray] = {}
        for video_id in videos:
            item = cached[video_id]
            release = np.asarray(item["long_release"], dtype=np.uint8)
            if threshold is not None and minimum is not None:
                short_speech = _speech_mask(
                    item["short_probabilities"],
                    threshold=threshold,
                    minimum_seconds=minimum,
                    window_seconds=item["short_window"],
                    merge_gap_seconds=0.18,
                )
                valid = item["short_probabilities"][:, 0] >= 0.0
                fast_release = np.asarray(valid & (short_speech == 0), dtype=np.uint8)
                release = np.asarray((release != 0) | (fast_release != 0), dtype=np.uint8)
            inputs = dataset.video_inputs(video_id, min(windows))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], item["speech"], item["probes"], release, config=config,
            )
            per_video[video_id] = _compact(
                score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"])
            )
            variant_masks[video_id] = release
        row = {
            "fast_release_silero_speech_threshold": threshold,
            "fast_release_min_speech_seconds": minimum,
            "fast_release_window_seconds": cached[videos[0]]["short_window"],
            "live_speaker_clear_window_seconds": source.get("live_speaker_clear_window_seconds"),
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"],
            "profile_name": source["profile_name"],
            "windows_seconds": list(windows),
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video,
        }
        rows.append(row)
        masks[(threshold, minimum)] = variant_masks
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [threshold, minimum],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    key = (best["fast_release_silero_speech_threshold"], best["fast_release_min_speech_seconds"])
    output_gate = args.output_gate_root.resolve() / "best"
    for video_id, release in masks[key].items():
        source_gate = args.long_gate_root.resolve() / video_id
        target = output_gate / video_id
        _atomic_npy(target / "speech_gate.u1.npy", cached[video_id]["speech"])
        _atomic_npy(target / "probe_schedule.u1.npy", cached[video_id]["probes"])
        _atomic_npy(target / "release_gate.u1.npy", release)
        _atomic_json(target / "gate_tape.json", {
            "tape_id": "production_silero_dual_release_gate_v1",
            "video_id": video_id,
            "long_gate_root": str(source_gate),
            "fast_release_silero_speech_threshold": key[0],
            "fast_release_min_speech_seconds": key[1],
            "fast_release_window_seconds": cached[video_id]["short_window"],
        })
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_DUAL_RELEASE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        "gate_variant": "best", **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "total_candidate_count": len(rows), "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
