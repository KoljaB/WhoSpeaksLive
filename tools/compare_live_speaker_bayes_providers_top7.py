"""Re-score every cached single provider with the current Bayesian champion."""

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
    config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    providers = list(dict.fromkeys([*spec["providers"], source["provider_spec"]]))
    gate_root = args.gate_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for provider in providers:
        profile_name = str(spec["profile_sets"].get(provider) or provider)
        try:
            dataset = Dataset(
                args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name
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
            rows.append({
                "provider_spec": provider,
                "profile_name": profile_name,
                "windows_seconds": list(windows),
                "algorithm_config": asdict(config),
                "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
                "per_video": per_video,
            })
        except Exception as exc:
            failures.append({"provider_spec": provider, "error": f"{type(exc).__name__}: {exc}"})
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_provider_count": len(rows) + len(failures),
            "total_provider_count": len(providers),
            "best_score": best["aggregate"]["primary_score"],
            "best_provider": best["provider_spec"], "failures": failures,
        })
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_PROVIDER_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(
            float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
        ),
        **best, "failures": failures, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_provider_count": len(rows) + len(failures),
        "total_provider_count": len(providers),
        "best_score": best["aggregate"]["primary_score"],
        "best_provider": best["provider_spec"], "failures": failures,
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
