"""Tune production-parity Silero speech and release gates on Top-7."""

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
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import aggregate_video_scores_primary_v2, score_live_speaker_decisions


FRAME_SECONDS = 512 / 16_000


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _speech_mask(
    probabilities: np.ndarray,
    *,
    threshold: float,
    minimum_seconds: float,
    window_seconds: float,
    merge_gap_seconds: float,
) -> np.ndarray:
    result = np.zeros(probabilities.shape[0], dtype=np.uint8)
    max_gap = max(0, int(round(merge_gap_seconds / FRAME_SECONDS)))
    for row_index in np.flatnonzero(probabilities[:, 0] >= 0.0):
        values = probabilities[row_index]
        values = values[values >= 0.0]
        flags = [bool(value >= threshold) for value in values]
        if max_gap > 0 and any(flags):
            index = 0
            while index < len(flags):
                if flags[index]:
                    index += 1
                    continue
                gap_start = index
                while index < len(flags) and not flags[index]:
                    index += 1
                if gap_start > 0 and index < len(flags) and index - gap_start <= max_gap:
                    flags[gap_start:index] = [True] * (index - gap_start)
        speech_seconds = 0.0
        for index, active in enumerate(flags):
            if active:
                speech_seconds += min(FRAME_SECONDS, window_seconds - index * FRAME_SECONDS)
        result[row_index] = int(speech_seconds >= minimum_seconds)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--probability-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    probability_root = args.probability_root.resolve()
    cached: dict[str, dict[str, Any]] = {}
    for video_id in videos:
        root = probability_root / video_id
        metadata = json.loads((root / "probability_tape.json").read_text(encoding="utf-8-sig"))
        cached[video_id] = {
            "metadata": metadata,
            "probe": np.load(root / "probe_probabilities.f32.npy", allow_pickle=False),
            "release": np.load(root / "release_probabilities.f32.npy", allow_pickle=False),
            "schedule": np.load(root / "probe_schedule.u1.npy", allow_pickle=False),
        }
    thresholds = (0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90)
    minima = (0.032, 0.064, 0.096, 0.128, 0.160, 0.192, 0.224, 0.250, 0.288, 0.320)
    variants = list(itertools.product(thresholds, minima))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for threshold, minimum in variants:
        per_video: dict[str, Any] = {}
        variant_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for video_id in videos:
            item = cached[video_id]
            metadata = item["metadata"]
            speech = _speech_mask(
                item["probe"], threshold=threshold, minimum_seconds=minimum,
                window_seconds=float(metadata["probe_window_seconds"]), merge_gap_seconds=0.18,
            )
            clear_speech = _speech_mask(
                item["release"], threshold=threshold, minimum_seconds=minimum,
                window_seconds=float(metadata["clear_window_seconds"]), merge_gap_seconds=0.18,
            )
            probes = np.asarray(item["schedule"], dtype=np.uint8)
            release = np.asarray((item["release"][:, 0] >= 0.0) & (clear_speech == 0), dtype=np.uint8)
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
            "vad_silero_speech_threshold": threshold,
            "vad_min_speech_seconds": minimum,
            "vad_merge_gap_seconds": 0.18,
            "live_speaker_probe_speech_backend": "vad",
            "vad_backend": "silero",
            "live_speaker_clear_window_seconds": float(source["live_speaker_clear_window_seconds"]),
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
        masks[(threshold, minimum)] = variant_masks
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "total_candidate_count": len(variants),
            "best_score": best["aggregate"]["primary_score"],
            "active": [threshold, minimum],
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    best_key = (float(best["vad_silero_speech_threshold"]), float(best["vad_min_speech_seconds"]))
    best_gate = args.gate_root.resolve() / "best"
    for video_id, (speech, probes, release) in masks[best_key].items():
        target = best_gate / video_id
        _atomic_npy(target / "speech_gate.u1.npy", speech)
        _atomic_npy(target / "probe_schedule.u1.npy", probes)
        _atomic_npy(target / "release_gate.u1.npy", release)
        _atomic_json(target / "gate_tape.json", {
            "tape_id": "production_silero_live_gate_tape_v1",
            "video_id": video_id,
            "vad_silero_speech_threshold": best_key[0],
            "vad_min_speech_seconds": best_key[1],
            "vad_merge_gap_seconds": 0.18,
            "release_every_tick": True,
        })
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_SILERO_GATE_WINNER_PENDING_FRESH_LIVE",
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
