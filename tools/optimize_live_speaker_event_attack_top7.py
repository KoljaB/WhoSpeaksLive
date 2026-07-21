"""Test event-driven fast probes only while no live speaker is visible."""

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
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--full-probability-root", type=Path, required=True)
    parser.add_argument("--output-gate-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(),
        args.input_root.resolve(),
        str(source["provider_spec"]),
        str(source["profile_name"]),
    )
    gate_root = args.gate_root.resolve()
    probability_root = args.full_probability_root.resolve()
    cached: dict[str, dict[str, np.ndarray]] = {}
    for video_id in videos:
        probability_dir = probability_root / video_id
        metadata = json.loads(
            (probability_dir / "probability_tape.json").read_text(encoding="utf-8-sig")
        )
        if abs(float(metadata["clear_window_seconds"]) - min(windows)) > 1e-6:
            raise RuntimeError(f"{video_id}: full probability window does not match short embedding")
        probabilities = np.load(
            probability_dir / "release_probabilities.f32.npy", allow_pickle=False
        )
        full_speech = _speech_mask(
            probabilities,
            threshold=float(source["vad_silero_speech_threshold"]),
            minimum_seconds=float(source["vad_min_speech_seconds"]),
            window_seconds=float(metadata["clear_window_seconds"]),
            merge_gap_seconds=float(source.get("vad_merge_gap_seconds") or 0.18),
        )
        video_gate = gate_root / video_id
        cached[video_id] = {
            "speech": np.asarray(full_speech, dtype=np.uint8),
            "probes": np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False),
            "release": np.load(video_gate / "release_gate.u1.npy", allow_pickle=False),
        }

    variants = (0.0, 0.2, 0.4, 0.6)
    rows: list[dict[str, Any]] = []
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for attack_interval in variants:
        per_video: dict[str, Any] = {}
        for video_id in videos:
            item = cached[video_id]
            inputs = dataset.video_inputs(video_id, min(windows))
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"],
                item["speech"],
                item["probes"],
                item["release"],
                config=config,
                attack_probe_interval_seconds=attack_interval,
            )
            per_video[video_id] = _compact(
                score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"])
            )
        row = {
            "live_speaker_probe_attack_interval_seconds": attack_interval,
            "live_speaker_probe_attack_min_advance_seconds": attack_interval,
            "algorithm_config": asdict(config),
            "provider_spec": source["provider_spec"],
            "profile_name": source["profile_name"],
            "windows_seconds": list(windows),
            "aggregate": aggregate_video_scores_primary_v2(per_video.values()),
            "per_video": per_video,
        }
        rows.append(row)
        best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
        _atomic(
            run_dir / "progress.json",
            {
                "status": "running",
                "completed_candidate_count": len(rows),
                "total_candidate_count": len(variants),
                "best_score": best["aggregate"]["primary_score"],
                "active_attack_interval_seconds": attack_interval,
            },
        )

    best = max(rows, key=lambda item: float(item["aggregate"]["primary_score"]))
    output_gate = args.output_gate_root.resolve() / "best"
    for video_id, item in cached.items():
        target = output_gate / video_id
        _atomic_npy(target / "speech_gate.u1.npy", item["speech"])
        _atomic_npy(target / "probe_schedule.u1.npy", item["probes"])
        _atomic_npy(target / "release_gate.u1.npy", item["release"])
        # Fresh verification needs embeddings at every possible attack tick;
        # replay still decides causally which of them is actually consumed.
        _atomic_npy(
            target / "embedding_schedule.u1.npy",
            np.asarray(item["speech"] & (item["release"] == 0), dtype=np.uint8),
        )
        _atomic_json(
            target / "gate_tape.json",
            {
                "tape_id": "production_silero_event_attack_gate_v1",
                "video_id": video_id,
                "attack_interval_seconds": best["live_speaker_probe_attack_interval_seconds"],
            },
        )

    _atomic(run_dir / "trials.json", rows)
    _atomic(
        run_dir / "champion.json",
        {
            "status": "CACHE_EVENT_ATTACK_WINNER_PENDING_FRESH_LIVE",
            "source_champion_score": source["candidate_score"],
            "candidate_score": best["aggregate"]["primary_score"],
            "score_delta": round(
                float(best["aggregate"]["primary_score"]) - float(source["candidate_score"]), 6
            ),
            "gate_variant": "best",
            **best,
            "fresh_live_verified": False,
        },
    )
    _atomic(
        run_dir / "progress.json",
        {
            "status": "complete",
            "completed_candidate_count": len(rows),
            "total_candidate_count": len(rows),
            "best_score": best["aggregate"]["primary_score"],
        },
    )
    print((run_dir / "progress.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
