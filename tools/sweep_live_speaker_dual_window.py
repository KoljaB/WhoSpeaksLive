from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from optimize_live_speaker_replay import Dataset, _stable_id
from window.live_speaker_algorithm import ALGORITHM_ID, LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_replay import replay_cached_live_windows_dual


SWEEP_ID = "causal_dual_window_sweep_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep short-acquisition plus long-context live embedding windows."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = list(dict.fromkeys(spec["split"]["search"] + spec["split"]["validation"]))
    provider_spec = "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in spec["baseline"]["provider_weights"].items()
        if float(weight) > 0.0
    )
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider_spec)
    configs: list[LiveSpeakerAlgorithmConfig] = [
        LiveSpeakerAlgorithmConfig(**spec["baseline"]["algorithm_config"])
    ]
    for row in champion.get("accepted_steps") or []:
        candidate = LiveSpeakerAlgorithmConfig(**row["algorithm_config"])
        if candidate not in configs:
            configs.append(candidate)

    trials: list[dict[str, Any]] = []
    for config in configs:
        for short_window in (0.7, 0.8, 1.0):
            for long_window in (1.5, 2.0, 2.4, 2.8, 3.0):
                for long_weight in (0.25, 0.5, 0.75, 0.9, 1.0):
                    per_video: dict[str, Any] = {}
                    for video_id in videos:
                        inputs = dataset.video_inputs(video_id)
                        decisions = replay_cached_live_windows_dual(
                            dataset.block(video_id, short_window),
                            dataset.block(video_id, long_window),
                            inputs["profiles"],
                            inputs["speech"],
                            inputs["probes"],
                            inputs["releases"],
                            long_weight=long_weight,
                            config=config,
                        )
                        per_video[video_id] = score_live_speaker_decisions(
                            decisions,
                            inputs["canonical"],
                            inputs["profiles"],
                        )
                    aggregate = aggregate_video_scores(per_video.values())
                    trials.append({
                        "candidate_id": _stable_id({
                            "algorithm_id": ALGORITHM_ID,
                            "short_window_seconds": short_window,
                            "long_window_seconds": long_window,
                            "long_weight": long_weight,
                            "algorithm_config": asdict(config),
                        }),
                        "short_window_seconds": short_window,
                        "long_window_seconds": long_window,
                        "long_weight": long_weight,
                        "algorithm_config": asdict(config),
                        "aggregate": aggregate,
                        "per_video": per_video,
                    })
    trials.sort(key=lambda row: float(row["aggregate"]["global_score"]), reverse=True)
    payload = {
        "schema_version": 1,
        "sweep_id": SWEEP_ID,
        "provider": provider_spec,
        "videos": videos,
        "candidate_count": len(trials),
        "best": trials[0],
        "top20": trials[:20],
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "candidate_count": len(trials),
        "best": trials[0],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
