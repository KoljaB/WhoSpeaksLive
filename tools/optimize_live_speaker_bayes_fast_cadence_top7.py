"""Retune Bayesian state dynamics for a fixed faster two-window cadence."""

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

from optimize_live_speaker_bayes_cadence_top7 import _probe_schedule
from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from optimize_live_speaker_silero_gate_top7 import _speech_mask
from build_live_speaker_gate_tapes import _atomic_json, _atomic_npy
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import aggregate_video_scores_primary_v2, score_live_speaker_decisions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--full-probability-root", type=Path, required=True)
    parser.add_argument("--output-gate-root", type=Path, required=True)
    parser.add_argument("--acquire-threshold", type=float, default=0.3)
    parser.add_argument("--acquire-min-speech-seconds", type=float, default=0.032)
    parser.add_argument("--probe-interval-seconds", type=float, default=0.4)
    parser.add_argument("--passes", type=int, default=3)
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
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(source["provider_spec"]), str(source["profile_name"]),
    )
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    gate_root = args.gate_root.resolve()
    full_probability_root = args.full_probability_root.resolve()
    full_speech: dict[str, np.ndarray] = {}
    for video_id in videos:
        probability_root = full_probability_root / video_id
        metadata = json.loads(
            (probability_root / "probability_tape.json").read_text(encoding="utf-8-sig")
        )
        full_speech[video_id] = _speech_mask(
            np.load(probability_root / "release_probabilities.f32.npy", allow_pickle=False),
            threshold=float(args.acquire_threshold),
            minimum_seconds=float(args.acquire_min_speech_seconds),
            window_seconds=float(metadata["clear_window_seconds"]),
            merge_gap_seconds=0.18,
        )

    def evaluate(config: BayesSpeakerTrackerConfig, phase: str) -> dict[str, Any]:
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(windows))
            block = dataset.block(video_id, min(windows))
            probes = _probe_schedule(
                block.media_times, min(windows), float(args.probe_interval_seconds)
            )
            video_gate = gate_root / video_id
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], full_speech[video_id], probes,
                np.load(video_gate / "release_gate.u1.npy", allow_pickle=False),
                config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        row = {
            "phase": phase,
            "probe_interval_seconds": float(args.probe_interval_seconds),
            "algorithm_config": asdict(config),
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video,
        }
        rows.append(row)
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(run_dir / "progress.json", {
            "status": "running", "completed_candidate_count": len(rows),
            "phase": phase, "best_score": best["aggregate"]["primary_score"],
        })
        return row

    current = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    current_score = float(evaluate(current, "SOURCE_AT_FAST_CADENCE")["aggregate"]["primary_score"])
    axes: list[tuple[str, tuple[Any, ...]]] = [
        ("prior_strength", (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)),
        ("stay_probability", (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98)),
        ("evidence_strength", (0.50, 0.75, 1.0, 1.5, 2.0, 3.0)),
        ("unknown_release_count", (1, 2, 3, 4, 5, 6, 8)),
        ("min_similarity", (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)),
        ("similarity_temperature", (0.05, 0.06, 0.075, 0.0875, 0.10, 0.12, 0.15)),
        ("high_profile_unknown_bias", (-0.25, 0.0, 0.25, 0.50, 0.75, 1.0)),
        ("provisional_scale_agreement_min_similarity", (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)),
        ("provisional_assignment_scale_agreement_min_similarity", (0.45, 0.50, 0.55, 0.60, 0.625, 0.65, 0.70, 0.75)),
    ]
    for pass_index in range(max(0, int(args.passes))):
        improved = False
        for field, values in axes:
            candidates = [evaluate(replace(current, **{field: value}), f"PASS_{pass_index + 1}_{field}") for value in values]
            best = max(candidates, key=lambda item: float(item["aggregate"]["primary_score"]))
            score = float(best["aggregate"]["primary_score"])
            if score > current_score + 1e-9:
                current = BayesSpeakerTrackerConfig(**best["algorithm_config"])
                current_score = score
                improved = True
        if not improved:
            break
    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    output_gate = args.output_gate_root.resolve() / "best"
    for video_id in videos:
        block = dataset.block(video_id, min(windows))
        probes = _probe_schedule(
            block.media_times, min(windows), float(best["probe_interval_seconds"])
        )
        source_gate = gate_root / video_id
        target = output_gate / video_id
        _atomic_npy(target / "speech_gate.u1.npy", full_speech[video_id])
        _atomic_npy(target / "probe_schedule.u1.npy", probes)
        _atomic_npy(
            target / "release_gate.u1.npy",
            np.load(source_gate / "release_gate.u1.npy", allow_pickle=False),
        )
        _atomic_json(target / "gate_tape.json", {
            "tape_id": "production_silero_fast_cadence_dual_release_gate_v1",
            "video_id": video_id,
            "probe_interval_seconds": best["probe_interval_seconds"],
            "source_release_gate": str(source_gate),
        })
    _atomic(run_dir / "trials.json", rows)
    _atomic(run_dir / "champion.json", {
        "status": "CACHE_FAST_CADENCE_WINNER_PENDING_FRESH_LIVE",
        "source_champion_score": source["candidate_score"],
        "candidate_score": best["aggregate"]["primary_score"],
        "score_delta": round(float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6),
        "provider_spec": source["provider_spec"], "profile_name": source["profile_name"],
        "windows_seconds": list(windows), **best, "fresh_live_verified": False,
    })
    _atomic(run_dir / "progress.json", {
        "status": "complete", "completed_candidate_count": len(rows),
        "best_score": best["aggregate"]["primary_score"],
        "source_champion_score": source["candidate_score"],
        "best_probe_interval_seconds": best["probe_interval_seconds"],
    })
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
