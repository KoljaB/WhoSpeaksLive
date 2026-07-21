"""Separate causal RMS thresholds for embedding probes and cheap release checks."""

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
    windows = tuple(float(value) for value in source["windows_seconds"])
    source_config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    values = (0.0014, 0.0017, 0.0020, 0.0023, 0.0026, 0.0030, 0.0035)
    release_counts = (1, 2, 3, 4)
    clear_window = float(source["live_speaker_clear_window_seconds"])
    min_speech = float(source["live_speaker_probe_min_speech_seconds"])
    total = len(values) * len(values) * len(release_counts)
    for probe_threshold in values:
        for release_threshold in values:
            variant = f"probe_{probe_threshold:.4f}_release_{release_threshold:.4f}"
            variant_root = args.gate_root.resolve() / variant
            for video_id in videos:
                source_meta = json.loads(
                    (args.corpus_root / "videos" / video_id / "source.json").read_text(encoding="utf-8-sig")
                )
                media_root = Path(str(source_meta["audio_path_at_creation"])).resolve().parent
                build_video(
                    args.corpus_root.resolve(), media_root, variant_root, video_id,
                    probe_window_seconds=min(windows), clear_window_seconds=clear_window,
                    cadence_seconds=0.75, frame_seconds=0.03, threshold=probe_threshold,
                    min_speech_seconds=min_speech, release_every_tick=True,
                    release_threshold=release_threshold,
                )
            for release_count in release_counts:
                config = replace(source_config, silence_release_count=release_count)
                per_video: dict[str, Any] = {}
                for video_id in videos:
                    inputs = dataset.video_inputs(video_id, min(windows))
                    video_gate = variant_root / video_id
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
                    "gate_variant": variant,
                    "vad_speech_rms_threshold": probe_threshold,
                    "live_speaker_release_rms_threshold": release_threshold,
                    "live_speaker_clear_window_seconds": clear_window,
                    "live_speaker_probe_min_speech_seconds": min_speech,
                    "release_every_tick": True,
                    "silence_release_count": release_count,
                    "algorithm_config": asdict(config),
                    "provider_spec": source["provider_spec"], "profile_name": source["profile_name"],
                    "windows_seconds": list(windows),
                    "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
                    "per_video": per_video,
                }
                rows.append(row)
                best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
                _atomic(run_dir / "progress.json", {
                    "status": "running", "completed_candidate_count": len(rows),
                    "total_candidate_count": total,
                    "best_score": best["aggregate"]["primary_score"],
                    "active_variant": variant,
                })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_SPLIT_RMS_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6),
        **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "total_candidate_count": total, "best_score": best["aggregate"]["primary_score"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
