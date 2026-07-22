"""Fresh-embedding verification for a Top-7 Bayesian cache champion."""

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

from embeddings.embedding_providers import RemotePreparedEmbeddingProvider, parse_embedding_provider_stack_specs
from verify_live_speaker_candidate_fresh import _cached_stack, _fresh_block
from window.live_speaker_algorithm import compare_decision_traces
from window.live_speaker_bayes import BAYES_ALGORITHM_ID, BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import PRIMARY_SCORER_V2_ID, aggregate_video_scores_primary_v2, score_live_speaker_decisions
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import load_profile_events_jsonl


VERIFIER_ID = "live_speaker_top7_bayes_fresh_parity_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--gate-root",
        type=Path,
        help="Optional per-video gate-tape root for a promoted non-default causal gate.",
    )
    parser.add_argument("--baseline-verification", type=Path, required=True)
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-decision-match-ratio", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-champion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_path = args.champion.resolve()
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    baseline_artifact = json.loads(args.baseline_verification.read_text(encoding="utf-8-sig"))
    baseline = baseline_artifact["baseline"]
    baseline_fresh_score = float(baseline["fresh_aggregate"]["primary_score"])
    videos = [str(value) for value in spec["videos"]]
    provider_spec = str(champion["provider_spec"])
    profile_name = str(champion["profile_name"])
    windows = [float(value) for value in champion["windows_seconds"]]
    if len(windows) != 2:
        raise ValueError("Bayesian promotion permits exactly two fresh embedding windows")
    config = BayesSpeakerTrackerConfig.from_mapping(champion["algorithm_config"])
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
        if float(weight) > 0.0
    ]
    providers = {
        provider: RemotePreparedEmbeddingProvider(args.provider_endpoint, provider, args.device)
        for provider, _weight in provider_specs
    }

    fresh_scores: dict[str, Any] = {}
    cached_scores: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    generation_reports: list[dict[str, Any]] = []
    input_root = args.input_root.resolve()
    corpus_root = args.corpus_root.resolve()
    for video_id in videos:
        short = min(windows)
        gate_root = (
            args.gate_root.resolve() / video_id
            if args.gate_root is not None else
            input_root / "gate_sets" / f"{round(short * 1000):04d}ms" / video_id
        )
        profiles = load_profile_events_jsonl(
            input_root / "profiles" / profile_name / video_id / "production_stack.profiles.jsonl"
        )
        canonical = read_canonical_segments(
            input_root / "references" / video_id / "canonical_diarization.json"
        )
        speech = np.load(gate_root / "speech_gate.u1.npy", allow_pickle=False)
        probes = np.load(gate_root / "probe_schedule.u1.npy", allow_pickle=False)
        releases = np.load(gate_root / "release_gate.u1.npy", allow_pickle=False)
        source = json.loads(
            (corpus_root / "videos" / video_id / "source.json").read_text(encoding="utf-8-sig")
        )
        media_root = Path(str(source["audio_path_at_creation"])).resolve().parent
        fresh_blocks = []
        cached_blocks = []
        for window in windows:
            block, report = _fresh_block(
                corpus_root=corpus_root,
                media_root=media_root,
                input_root=gate_root.parent,
                provider_specs=provider_specs,
                providers=providers,
                provider_spec=provider_spec,
                video_id=video_id,
                window_seconds=window,
            )
            fresh_blocks.append(block)
            cached_blocks.append(_cached_stack(corpus_root, provider_specs, video_id, window, provider_spec))
            generation_reports.append(report)
        fresh_decisions = replay_cached_bayes_windows(
            fresh_blocks,
            profiles,
            speech,
            probes,
            releases,
            config=config,
            attack_probe_interval_seconds=float(
                champion.get("live_speaker_probe_attack_interval_seconds") or 0.0
            ),
        )
        cached_decisions = replay_cached_bayes_windows(
            cached_blocks,
            profiles,
            speech,
            probes,
            releases,
            config=config,
            attack_probe_interval_seconds=float(
                champion.get("live_speaker_probe_attack_interval_seconds") or 0.0
            ),
        )
        fresh_scores[video_id] = score_live_speaker_decisions(fresh_decisions, canonical, profiles)
        cached_scores[video_id] = score_live_speaker_decisions(cached_decisions, canonical, profiles)
        parity[video_id] = compare_decision_traces(fresh_decisions, cached_decisions)

    fresh_aggregate = aggregate_video_scores_primary_v2(fresh_scores.values())
    cached_aggregate = aggregate_video_scores_primary_v2(cached_scores.values())
    candidate_fresh_score = float(fresh_aggregate["primary_score"])
    candidate_cached_score = float(cached_aggregate["primary_score"])
    declared_cached_score = float(champion["candidate_score"])
    cache_reproduced = abs(candidate_cached_score - declared_cached_score) <= 1e-6
    all_generation_succeeded = all(int(row["failure_count"]) == 0 for row in generation_reports)
    minimum_match = min(float(row["decision_match_ratio"]) for row in parity.values())
    required_match = max(0.0, min(1.0, float(args.min_decision_match_ratio)))
    passed = bool(
        cache_reproduced
        and all_generation_succeeded
        and minimum_match >= required_match
        and candidate_fresh_score > baseline_fresh_score + 1e-6
    )
    payload = {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "algorithm_id": BAYES_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "promotion_policy": "fresh_primary_score_improves_and_each_video_cache_match_ratio_is_at_least_threshold",
        "videos": videos,
        "provider_spec": provider_spec,
        "windows_freshly_embedded": windows,
        "algorithm_config": asdict(config),
        "gate_root": str(args.gate_root.resolve()) if args.gate_root is not None else None,
        "generation_reports": generation_reports,
        "baseline_fresh_score": baseline_fresh_score,
        "candidate_fresh_score": candidate_fresh_score,
        "fresh_score_delta": round(candidate_fresh_score - baseline_fresh_score, 6),
        "candidate_cached_score": candidate_cached_score,
        "declared_cached_score": declared_cached_score,
        "cache_reproduced": cache_reproduced,
        "minimum_decision_match_ratio_required": required_match,
        "candidate_minimum_decision_match_ratio": minimum_match,
        "all_generation_succeeded": all_generation_succeeded,
        "all_cached_decisions_exact": all(bool(row["exact_match"]) for row in parity.values()),
        "fresh_aggregate": fresh_aggregate,
        "cached_aggregate": cached_aggregate,
        "fresh_per_video": fresh_scores,
        "cached_per_video": cached_scores,
        "parity": parity,
        "passed": passed,
    }
    _atomic_json(args.output.resolve(), payload)
    if args.update_champion:
        champion["fresh_live_verified"] = False
        champion["fresh_embedding_replay_verified"] = passed
        champion["fresh_embedding_replay_verification"] = {
            "verifier_id": VERIFIER_ID,
            "path": str(args.output.resolve()),
            "baseline_score": baseline_fresh_score,
            "candidate_score": candidate_fresh_score,
            "score_delta": payload["fresh_score_delta"],
            "minimum_decision_match_ratio_required": required_match,
            "candidate_minimum_decision_match_ratio": minimum_match,
            "all_cached_decisions_exact": payload["all_cached_decisions_exact"],
        }
        champion["production_promotion_eligible"] = False
        champion["requires_real_gui_live_e2e"] = True
        champion["status"] = (
            "FRESH_EMBEDDING_REPLAY_VERIFIED_AWAITING_REAL_GUI_E2E"
            if passed else "REJECTED_BY_FRESH_EMBEDDING_REPLAY"
        )
        _atomic_json(champion_path, champion)
    print(json.dumps({
        "passed": passed,
        "baseline_fresh_score": baseline_fresh_score,
        "candidate_fresh_score": candidate_fresh_score,
        "fresh_score_delta": payload["fresh_score_delta"],
        "candidate_cached_score": candidate_cached_score,
        "cache_reproduced": cache_reproduced,
        "candidate_minimum_decision_match_ratio": minimum_match,
        "output": str(args.output.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
