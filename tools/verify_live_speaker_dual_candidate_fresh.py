from __future__ import annotations

import argparse
from dataclasses import asdict
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
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    load_profile_events_jsonl,
    replay_cached_live_windows,
    replay_cached_live_windows_dual,
)


VERIFIER_ID = "fresh_linux_dual_window_verifier_v1"


def _inputs(root: Path, video_id: str) -> dict[str, Any]:
    video_root = root / video_id
    return {
        "profiles": load_profile_events_jsonl(video_root / "production_stack.profiles.jsonl"),
        "speech": np.load(video_root / "speech_gate.u1.npy", allow_pickle=False),
        "probes": np.load(video_root / "probe_schedule.u1.npy", allow_pickle=False),
        "releases": np.load(video_root / "release_gate.u1.npy", allow_pickle=False),
        "canonical": read_canonical_segments(video_root / "canonical_diarization.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freshly re-embed and score dual-window live-speaker finalists."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finalist-count", type=int, default=10)
    parser.add_argument("--min-parity", type=float, default=0.99)
    parser.add_argument("--per-video-regression-tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    sweep = json.loads(args.sweep.read_text(encoding="utf-8-sig"))
    videos = list(dict.fromkeys(spec["split"]["search"] + spec["split"]["validation"]))
    provider_spec = "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in spec["baseline"]["provider_weights"].items()
        if float(weight) > 0.0
    )
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
        if float(weight) > 0.0
    ]
    finalists = [dict(row) for row in sweep["top20"][: max(1, args.finalist_count)]]
    baseline_window = float(spec["baseline"]["probe_window_seconds"])
    windows = sorted(
        {baseline_window}
        | {float(row["short_window_seconds"]) for row in finalists}
        | {float(row["long_window_seconds"]) for row in finalists}
    )

    providers = {
        provider: RemotePreparedEmbeddingProvider(args.provider_endpoint, provider, args.device)
        for provider, _weight in provider_specs
    }
    fresh_blocks: dict[tuple[str, float], Any] = {}
    generation_reports: list[dict[str, Any]] = []
    for window in windows:
        for video_id in videos:
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
            fresh_blocks[(video_id, window)] = block
            generation_reports.append(report)

    inputs = {video_id: _inputs(args.input_root.resolve(), video_id) for video_id in videos}
    baseline_config = LiveSpeakerAlgorithmConfig(**spec["baseline"]["algorithm_config"])
    baseline_scores: dict[str, Any] = {}
    for video_id in videos:
        item = inputs[video_id]
        decisions = replay_cached_live_windows(
            fresh_blocks[(video_id, baseline_window)],
            item["profiles"], item["speech"], item["probes"], item["releases"],
            config=baseline_config,
        )
        baseline_scores[video_id] = score_live_speaker_decisions(
            decisions, item["canonical"], item["profiles"]
        )
    baseline_aggregate = aggregate_video_scores(baseline_scores.values())

    candidate_reports: list[dict[str, Any]] = []
    for candidate in finalists:
        short_window = float(candidate["short_window_seconds"])
        long_window = float(candidate["long_window_seconds"])
        long_weight = float(candidate["long_weight"])
        config = LiveSpeakerAlgorithmConfig(**candidate["algorithm_config"])
        fresh_scores: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        for video_id in videos:
            item = inputs[video_id]
            fresh_decisions = replay_cached_live_windows_dual(
                fresh_blocks[(video_id, short_window)],
                fresh_blocks[(video_id, long_window)],
                item["profiles"], item["speech"], item["probes"], item["releases"],
                long_weight=long_weight,
                config=config,
            )
            cached_decisions = replay_cached_live_windows_dual(
                _cached_stack(
                    args.corpus_root.resolve(), provider_specs, video_id, short_window, provider_spec
                ),
                _cached_stack(
                    args.corpus_root.resolve(), provider_specs, video_id, long_window, provider_spec
                ),
                item["profiles"], item["speech"], item["probes"], item["releases"],
                long_weight=long_weight,
                config=config,
            )
            fresh_scores[video_id] = score_live_speaker_decisions(
                fresh_decisions, item["canonical"], item["profiles"]
            )
            parity[video_id] = compare_decision_traces(fresh_decisions, cached_decisions)
        aggregate = aggregate_video_scores(fresh_scores.values())
        per_video_regressions = {
            video_id: round(
                float(fresh_scores[video_id]["strict_browser_live_score"])
                - float(baseline_scores[video_id]["strict_browser_live_score"]),
                6,
            )
            for video_id in videos
        }
        minimum_parity = min(
            float(row["decision_match_ratio"]) for row in parity.values()
        )
        eligible = (
            float(aggregate["global_score"]) > float(baseline_aggregate["global_score"])
            and minimum_parity >= float(args.min_parity)
            and min(per_video_regressions.values()) >= -float(args.per_video_regression_tolerance)
        )
        candidate_reports.append({
            "candidate_id": candidate["candidate_id"],
            "short_window_seconds": short_window,
            "long_window_seconds": long_window,
            "long_weight": long_weight,
            "algorithm_config": asdict(config),
            "cached_score": candidate["aggregate"]["global_score"],
            "fresh_aggregate": aggregate,
            "fresh_per_video": fresh_scores,
            "fresh_score_delta_vs_baseline": round(
                float(aggregate["global_score"]) - float(baseline_aggregate["global_score"]), 6
            ),
            "per_video_score_delta_vs_baseline": per_video_regressions,
            "cached_decision_parity": parity,
            "minimum_cached_decision_match_ratio": minimum_parity,
            "eligible_for_promotion": eligible,
        })

    candidate_reports.sort(
        key=lambda row: float(row["fresh_aggregate"]["global_score"]), reverse=True
    )
    generation_ok = all(int(row["failure_count"]) == 0 for row in generation_reports)
    eligible = [row for row in candidate_reports if row["eligible_for_promotion"]]
    winner = eligible[0] if generation_ok and eligible else None
    payload = {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "provider": provider_spec,
        "videos": videos,
        "windows_freshly_embedded": windows,
        "baseline": {
            "window_seconds": baseline_window,
            "algorithm_config": asdict(baseline_config),
            "aggregate": baseline_aggregate,
            "per_video": baseline_scores,
        },
        "generation_reports": generation_reports,
        "candidate_reports": candidate_reports,
        "winner": winner,
        "passed": winner is not None,
    }
    _atomic_json(args.output.resolve(), payload)
    if winner is not None:
        promotion = {
            "schema_version": 1,
            "status": "LIVE_VERIFIED_CHAMPION",
            "verifier_id": VERIFIER_ID,
            "provider": provider_spec,
            "baseline_score": baseline_aggregate["global_score"],
            "fresh_live_score": winner["fresh_aggregate"]["global_score"],
            "fresh_score_delta": winner["fresh_score_delta_vs_baseline"],
            "short_window_seconds": winner["short_window_seconds"],
            "long_window_seconds": winner["long_window_seconds"],
            "long_weight": winner["long_weight"],
            "algorithm_config": winner["algorithm_config"],
            "minimum_cached_decision_match_ratio": winner[
                "minimum_cached_decision_match_ratio"
            ],
            "verification_path": str(args.output.resolve()),
        }
        _atomic_json(args.promotion_output.resolve(), promotion)
    print(json.dumps({
        "baseline_score": baseline_aggregate["global_score"],
        "generation_ok": generation_ok,
        "winner": None if winner is None else {
            "candidate_id": winner["candidate_id"],
            "fresh_live_score": winner["fresh_aggregate"]["global_score"],
            "fresh_score_delta_vs_baseline": winner["fresh_score_delta_vs_baseline"],
            "short_window_seconds": winner["short_window_seconds"],
            "long_window_seconds": winner["long_window_seconds"],
            "long_weight": winner["long_weight"],
            "algorithm_config": winner["algorithm_config"],
            "per_video_score_delta_vs_baseline": winner[
                "per_video_score_delta_vs_baseline"
            ],
            "minimum_cached_decision_match_ratio": winner[
                "minimum_cached_decision_match_ratio"
            ],
        },
    }, indent=2, ensure_ascii=False))
    return 0 if winner is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
