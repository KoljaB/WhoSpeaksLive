"""Tune Silero release hysteresis independently from speech acquisition."""

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
    parser.add_argument("--probability-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--schedule-gate-root", type=Path)
    parser.add_argument("--acquisition-threshold", type=float)
    parser.add_argument("--acquisition-minimum", type=float)
    parser.add_argument("--run-dir", type=Path, required=True)
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
    probability_root = args.probability_root.resolve()
    cached: dict[str, dict[str, Any]] = {}
    acquisition_threshold = float(
        args.acquisition_threshold
        if args.acquisition_threshold is not None
        else source["vad_silero_speech_threshold"]
    )
    acquisition_minimum = float(
        args.acquisition_minimum
        if args.acquisition_minimum is not None
        else source["vad_min_speech_seconds"]
    )
    schedule_gate_root = (
        args.schedule_gate_root.resolve()
        if args.schedule_gate_root is not None
        else None
    )
    for video_id in videos:
        root = probability_root / video_id
        metadata = json.loads((root / "probability_tape.json").read_text(encoding="utf-8-sig"))
        probe_probabilities = np.load(root / "probe_probabilities.f32.npy", allow_pickle=False)
        cached[video_id] = {
            "metadata": metadata,
            "speech": _speech_mask(
                probe_probabilities,
                threshold=acquisition_threshold,
                minimum_seconds=acquisition_minimum,
                window_seconds=float(metadata["probe_window_seconds"]),
                merge_gap_seconds=0.18,
            ),
            "release_probabilities": np.load(
                root / "release_probabilities.f32.npy", allow_pickle=False
            ),
            "schedule": np.load(
                (schedule_gate_root / video_id / "probe_schedule.u1.npy")
                if schedule_gate_root is not None
                else (root / "probe_schedule.u1.npy"),
                allow_pickle=False,
            ),
        }
    clear_window_seconds = float(cached[videos[0]]["metadata"]["clear_window_seconds"])
    if any(
        abs(float(cached[video_id]["metadata"]["clear_window_seconds"]) - clear_window_seconds) > 1e-9
        for video_id in videos
    ):
        raise RuntimeError("Silero probability tapes use different clear-window lengths")
    thresholds = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)
    minima = (0.032, 0.064, 0.096, 0.128, 0.160, 0.192, 0.224, 0.250, 0.288, 0.320)
    variants = list(itertools.product(thresholds, minima))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for release_threshold, release_minimum in variants:
        per_video: dict[str, Any] = {}
        variant_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for video_id in videos:
            item = cached[video_id]
            metadata = item["metadata"]
            clear_speech = _speech_mask(
                item["release_probabilities"],
                threshold=release_threshold,
                minimum_seconds=release_minimum,
                window_seconds=float(metadata["clear_window_seconds"]),
                merge_gap_seconds=0.18,
            )
            valid = item["release_probabilities"][:, 0] >= 0.0
            release = np.asarray(valid & (clear_speech == 0), dtype=np.uint8)
            speech = np.asarray(item["speech"], dtype=np.uint8)
            probes = np.asarray(item["schedule"], dtype=np.uint8)
            inputs = dataset.video_inputs(video_id, min(windows))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], speech, probes, release, config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
            variant_masks[video_id] = (speech, probes, release)
        row = {
            "vad_silero_speech_threshold": acquisition_threshold,
            "vad_min_speech_seconds": acquisition_minimum,
            "live_speaker_release_silero_speech_threshold": release_threshold,
            "live_speaker_release_min_speech_seconds": release_minimum,
            "vad_merge_gap_seconds": 0.18,
            "live_speaker_probe_speech_backend": "vad",
            "vad_backend": "silero",
            "live_speaker_clear_window_seconds": clear_window_seconds,
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
        masks[(release_threshold, release_minimum)] = variant_masks
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [release_threshold, release_minimum],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    best_key = (
        float(best["live_speaker_release_silero_speech_threshold"]),
        float(best["live_speaker_release_min_speech_seconds"]),
    )
    best_gate = args.gate_root.resolve() / "best"
    for video_id, (speech, probes, release) in masks[best_key].items():
        target = best_gate / video_id
        _atomic_npy(target / "speech_gate.u1.npy", speech)
        _atomic_npy(target / "probe_schedule.u1.npy", probes)
        _atomic_npy(target / "release_gate.u1.npy", release)
        _atomic_json(target / "gate_tape.json", {
            "tape_id": "production_silero_split_live_gate_tape_v1",
            "video_id": video_id,
            "vad_silero_speech_threshold": acquisition_threshold,
            "vad_min_speech_seconds": acquisition_minimum,
            "release_silero_speech_threshold": best_key[0],
            "release_min_speech_seconds": best_key[1],
        })
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_SILERO_RELEASE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        "gate_variant": "best",
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
