from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import LiveSpeakerScoreConfig, score_live_speaker_decisions
from window.live_speaker_probe_scoring import read_canonical_segments
from embeddings.embedding_providers import parse_embedding_provider_stack_specs
from window.live_speaker_replay import (
    load_cached_live_window_block,
    load_profile_events_jsonl,
    replay_cached_live_windows,
    stack_cached_live_window_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one cached causal live-window block and write its versioned score."
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument("--profile-events", type=Path, required=True)
    parser.add_argument("--speech-mask", type=Path, required=True)
    parser.add_argument("--probe-mask", type=Path, required=True)
    parser.add_argument("--release-mask", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    return parser.parse_args()


def _config(path: Path | None) -> LiveSpeakerAlgorithmConfig:
    if path is None:
        return LiveSpeakerAlgorithmConfig()
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Algorithm config must be a JSON object")
    return LiveSpeakerAlgorithmConfig(**value)


def main() -> int:
    args = parse_args()
    config = _config(args.algorithm_config)
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(args.provider)
        if float(weight) > 0.0
    ]
    if len(provider_specs) == 1:
        block = load_cached_live_window_block(
            args.corpus_root, provider_specs[0][0], args.video_id, args.window_seconds
        )
    else:
        blocks = [
            load_cached_live_window_block(
                args.corpus_root, provider, args.video_id, args.window_seconds
            )
            for provider, _weight in provider_specs
        ]
        block = stack_cached_live_window_blocks(
            blocks,
            [weight for _provider, weight in provider_specs],
            provider=args.provider,
        )
    speech_mask = np.load(args.speech_mask, allow_pickle=False)
    probe_mask = np.load(args.probe_mask, allow_pickle=False)
    release_mask = np.load(args.release_mask, allow_pickle=False)
    profile_events = load_profile_events_jsonl(args.profile_events)
    decisions = replay_cached_live_windows(
        block, profile_events, speech_mask, probe_mask, release_mask, config=config
    )
    score = score_live_speaker_decisions(
        decisions,
        read_canonical_segments(args.canonical),
        profile_events,
        config=LiveSpeakerScoreConfig(),
    )
    payload = {
        "input": {
            "corpus_root": str(args.corpus_root.resolve()),
            "provider": args.provider,
            "video_id": args.video_id,
            "window_seconds": args.window_seconds,
            "profile_events": str(args.profile_events.resolve()),
            "speech_mask": str(args.speech_mask.resolve()),
            "probe_mask": str(args.probe_mask.resolve()),
            "release_mask": str(args.release_mask.resolve()),
            "canonical": str(args.canonical.resolve()),
        },
        "algorithm_config": asdict(config),
        "score": score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(
            "\n".join(json.dumps(item.trace_record(), ensure_ascii=False) for item in decisions)
            + "\n",
            encoding="utf-8",
        )
    print(f"Global video score: {score['strict_browser_live_score']:.6f}")
    print(f"Coverage: {score['correct_live_speaker_coverage']:.6f}")
    print(f"Wrong speech ratio: {score['wrong_live_speech_ratio']:.6f}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
