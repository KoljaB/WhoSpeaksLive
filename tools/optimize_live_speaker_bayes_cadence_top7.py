"""Evaluate causal probe cadence for a fixed two-window Bayesian champion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from build_live_speaker_gate_tapes import build_video
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import aggregate_video_scores_primary_v2, score_live_speaker_decisions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _probe_schedule(media_times: np.ndarray, first_window: float, cadence: float) -> np.ndarray:
    result = np.zeros(media_times.shape[0], dtype=bool)
    last: float | None = None
    for index, raw_time in enumerate(media_times):
        media_time = float(raw_time)
        if media_time + 1e-9 < first_window:
            continue
        if last is None or media_time - last + 1e-9 >= cadence:
            result[index] = True
            last = media_time
    return result


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
    config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cadence in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00):
        per_video: dict[str, Any] = {}
        exact_source_masks = True
        probe_counts: dict[str, int] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            block = dataset.block(video_id, min(windows))
            if args.gate_root is not None:
                variant_root = args.gate_root.resolve() / f"cadence_{cadence:.2f}"
                source_meta = json.loads(
                    (args.corpus_root / "videos" / video_id / "source.json").read_text(
                        encoding="utf-8-sig"
                    )
                )
                media_root = Path(str(source_meta["audio_path_at_creation"])).resolve().parent
                build_video(
                    args.corpus_root.resolve(), media_root, variant_root, video_id,
                    probe_window_seconds=min(windows),
                    clear_window_seconds=float(source["live_speaker_clear_window_seconds"]),
                    cadence_seconds=cadence,
                    frame_seconds=float(source["vad_frame_seconds"]),
                    threshold=float(source["vad_speech_rms_threshold"]),
                    min_speech_seconds=float(source["live_speaker_probe_min_speech_seconds"]),
                    release_every_tick=True,
                )
                video_gate = variant_root / video_id
                speech = np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False)
                probes = np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False)
                releases = np.load(video_gate / "release_gate.u1.npy", allow_pickle=False)
            else:
                probes = _probe_schedule(block.media_times, min(windows), cadence)
                speech = inputs["speech"]
                releases = inputs["releases"]
            if abs(cadence - 0.75) < 1e-9 and args.gate_root is None:
                exact_source_masks = exact_source_masks and bool(
                    np.array_equal(probes, inputs["probes"])
                )
            probe_counts[video_id] = int(np.count_nonzero(probes))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], speech, probes, releases, config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        row = {
            "probe_interval_seconds": cadence,
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"],
            "profile_name": source["profile_name"],
            "windows_seconds": list(windows),
            "aggregate": aggregate,
            "per_video": per_video,
            "probe_counts": probe_counts,
            "source_mask_exact": exact_source_masks if abs(cadence - 0.75) < 1e-9 else None,
            "gate_variant": f"cadence_{cadence:.2f}" if args.gate_root is not None else None,
        }
        rows.append(row)
        _atomic(args.run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "best_score": max(float(item["aggregate"]["primary_score"]) for item in rows),
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(args.run_dir / "trials.json", rows)
    _atomic(args.run_dir / "champion.json", {
        "status": "CACHE_CADENCE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6),
        **best,
        "fresh_live_verified": False,
    })
    _atomic(args.run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "best_score": best["aggregate"]["primary_score"],
        "best_probe_interval_seconds": best["probe_interval_seconds"],
    })
    print(json.dumps((args.run_dir / "progress.json").read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
