"""Co-optimize the causal RMS speech/release gate around a Bayesian champion."""

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

from build_live_speaker_gate_tapes import build_video
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
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--min-speech-refine", action="store_true")
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
    gate_root = args.gate_root.resolve()
    rows: list[dict[str, Any]] = []
    if args.min_speech_refine:
        thresholds = (float(source["vad_speech_rms_threshold"]),)
        clear_windows = (float(source["live_speaker_clear_window_seconds"]),)
        min_speech_values = (0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.30)
        release_counts = (1, 2, 3, 4, 5, 6)
    elif args.refine:
        thresholds = (0.0014, 0.0016, 0.0018, 0.0020, 0.0022, 0.0024, 0.0026)
        clear_windows = (1.10, 1.30, 1.50, 1.70, 1.90, 2.10)
        min_speech_values = (0.15,)
        release_counts = (1, 2, 3, 4, 5)
    else:
        thresholds = (0.0015, 0.002, 0.003, 0.004, 0.006)
        clear_windows = (0.30, 0.50, 0.70, 1.00, 1.50)
        min_speech_values = (0.15,)
        release_counts = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
    total = len(thresholds) * len(clear_windows) * len(min_speech_values) * len(release_counts)
    completed = 0
    for threshold in thresholds:
        for clear_window in clear_windows:
            for min_speech_seconds in min_speech_values:
                variant = f"rms_{threshold:.4f}_clear_{clear_window:.2f}_speech_{min_speech_seconds:.2f}"
                variant_root = gate_root / variant
                for video_id in videos:
                    source_meta = json.loads(
                        (args.corpus_root / "videos" / video_id / "source.json").read_text(encoding="utf-8-sig")
                    )
                    media_root = Path(str(source_meta["audio_path_at_creation"])).resolve().parent
                    build_video(
                        args.corpus_root.resolve(), media_root, variant_root, video_id,
                        probe_window_seconds=min(windows), clear_window_seconds=clear_window,
                        cadence_seconds=0.75, frame_seconds=0.03, threshold=threshold,
                        min_speech_seconds=min_speech_seconds, release_every_tick=True,
                    )
                for release_count in release_counts:
                    config = replace(source_config, silence_release_count=release_count)
                    per_video: dict[str, Any] = {}
                    gate_counts: dict[str, Any] = {}
                    for video_id in videos:
                        inputs = dataset.video_inputs(video_id, min(windows))
                        video_gate = variant_root / video_id
                        speech = np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False)
                        probes = np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False)
                        releases = np.load(video_gate / "release_gate.u1.npy", allow_pickle=False)
                        gate_counts[video_id] = {
                            "speech_probes": int(np.count_nonzero(speech)),
                            "release_ticks": int(np.count_nonzero(releases)),
                        }
                        decisions = replay_cached_bayes_windows(
                            [dataset.block(video_id, window) for window in windows],
                            inputs["profiles"], speech, probes, releases, config=config,
                        )
                        per_video[video_id] = _compact(score_live_speaker_decisions(
                            decisions, inputs["canonical"], inputs["profiles"]
                        ))
                    row = {
                        "vad_speech_rms_threshold": threshold,
                        "live_speaker_clear_window_seconds": clear_window,
                        "live_speaker_probe_min_speech_seconds": min_speech_seconds,
                        "gate_variant": variant,
                        "release_every_tick": True,
                        "silence_release_count": release_count,
                        "algorithm_config": asdict(config),
                        "provider_spec": source["provider_spec"], "profile_name": source["profile_name"],
                        "windows_seconds": list(windows),
                        "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
                        "per_video": per_video, "gate_counts": gate_counts,
                    }
                    rows.append(row)
                    completed += 1
                    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
                    _atomic(run_dir / "progress.json", {
                        "status": "running", "completed_candidate_count": completed,
                        "total_candidate_count": total, "best_score": best["aggregate"]["primary_score"],
                        "active_variant": variant,
                    })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_RMS_GATE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6),
        **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": completed,
        "total_candidate_count": total, "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
