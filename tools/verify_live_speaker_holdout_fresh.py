from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from embeddings.embedding_providers import (
    RemotePreparedEmbeddingProvider,
    parse_embedding_provider_stack_specs,
)
from verify_live_speaker_candidate_fresh import _atomic_json, _cached_stack, _fresh_block
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig, compare_decision_traces
from window.live_speaker_benchmark import score_live_speaker_decisions
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    load_profile_events_jsonl,
    replay_cached_live_windows,
    replay_cached_live_windows_dual,
)


VERIFIER_ID = "fresh_linux_locked_holdout_verifier_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one locked live-speaker champion once on a fresh holdout."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    if champion.get("status") != "LIVE_VERIFIED_CHAMPION":
        raise RuntimeError("Holdout requires a previously locked LIVE_VERIFIED_CHAMPION")
    video_id = str(args.video_id)
    if video_id not in spec["split"]["sealed_holdout"]:
        raise RuntimeError(f"{video_id} is not the sealed holdout in the frozen spec")
    provider_spec = str(champion["provider"])
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
        if float(weight) > 0.0
    ]
    providers = {
        provider: RemotePreparedEmbeddingProvider(args.provider_endpoint, provider, args.device)
        for provider, _weight in provider_specs
    }
    baseline_window = float(spec["baseline"]["probe_window_seconds"])
    short_window = float(champion["short_window_seconds"])
    long_window = float(champion["long_window_seconds"])
    windows = sorted({baseline_window, short_window, long_window})
    fresh_blocks: dict[float, Any] = {}
    generation_reports: list[dict[str, Any]] = []
    for window in windows:
        block, report = _fresh_block(
            corpus_root=args.corpus_root.resolve(),
            media_root=args.media_root.resolve(),
            input_root=args.input_root.resolve(),
            provider_specs=provider_specs,
            providers=providers,
            provider_spec=provider_spec,
            video_id=video_id,
            window_seconds=window,
        )
        fresh_blocks[window] = block
        generation_reports.append(report)

    root = args.input_root.resolve() / video_id
    profiles = load_profile_events_jsonl(root / "production_stack.profiles.jsonl")
    speech = np.load(root / "speech_gate.u1.npy", allow_pickle=False)
    probes = np.load(root / "probe_schedule.u1.npy", allow_pickle=False)
    releases = np.load(root / "release_gate.u1.npy", allow_pickle=False)
    canonical = read_canonical_segments(root / "canonical_diarization.json")
    baseline_config = LiveSpeakerAlgorithmConfig(**spec["baseline"]["algorithm_config"])
    champion_config = LiveSpeakerAlgorithmConfig(**champion["algorithm_config"])

    baseline_decisions = replay_cached_live_windows(
        fresh_blocks[baseline_window], profiles, speech, probes, releases,
        config=baseline_config,
    )
    champion_decisions = replay_cached_live_windows_dual(
        fresh_blocks[short_window], fresh_blocks[long_window],
        profiles, speech, probes, releases,
        long_weight=float(champion["long_weight"]),
        config=champion_config,
    )
    cached_champion_decisions = replay_cached_live_windows_dual(
        _cached_stack(
            args.corpus_root.resolve(), provider_specs, video_id, short_window, provider_spec
        ),
        _cached_stack(
            args.corpus_root.resolve(), provider_specs, video_id, long_window, provider_spec
        ),
        profiles, speech, probes, releases,
        long_weight=float(champion["long_weight"]),
        config=champion_config,
    )
    baseline_score = score_live_speaker_decisions(baseline_decisions, canonical, profiles)
    champion_score = score_live_speaker_decisions(champion_decisions, canonical, profiles)
    parity = compare_decision_traces(champion_decisions, cached_champion_decisions)
    delta = round(
        float(champion_score["strict_browser_live_score"])
        - float(baseline_score["strict_browser_live_score"]),
        6,
    )
    generation_ok = all(int(row["failure_count"]) == 0 for row in generation_reports)
    passed = generation_ok and delta > 0.0 and float(parity["decision_match_ratio"]) >= 0.99
    payload = {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "status": "HOLDOUT_PASSED" if passed else "HOLDOUT_FAILED",
        "video_id": video_id,
        "champion_was_locked_before_holdout": True,
        "provider": provider_spec,
        "baseline": baseline_score,
        "champion": champion_score,
        "strict_score_delta": delta,
        "cached_decision_parity": parity,
        "generation_reports": generation_reports,
        "passed": passed,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "video_id": video_id,
        "baseline_score": baseline_score["strict_browser_live_score"],
        "champion_score": champion_score["strict_browser_live_score"],
        "strict_score_delta": delta,
        "correct_coverage_baseline": baseline_score["correct_live_speaker_coverage"],
        "correct_coverage_champion": champion_score["correct_live_speaker_coverage"],
        "wrong_speech_ratio_baseline": baseline_score["wrong_live_speech_ratio"],
        "wrong_speech_ratio_champion": champion_score["wrong_live_speech_ratio"],
        "decision_match_ratio": parity["decision_match_ratio"],
        "generation_ok": generation_ok,
    }, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
