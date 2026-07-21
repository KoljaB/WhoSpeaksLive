from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for value in (SRC, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from embeddings.embedding_providers import (
    RemotePreparedEmbeddingProvider,
    parse_embedding_provider_stack_specs,
)
from verify_live_speaker_candidate_fresh import _cached_stack, _fresh_block
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig, compare_decision_traces
from window.live_speaker_benchmark import (
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    load_profile_events_jsonl,
    replay_cached_live_windows,
    replay_cached_live_windows_dual,
)


VERIFIER_ID = "live_speaker_top7_fresh_parity_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _baseline_description(spec: dict[str, Any]) -> dict[str, Any]:
    baseline = spec["baseline"]
    provider_spec = str(baseline["provider_spec"])
    names = [name for name, value in spec["profile_sets"].items() if str(value) == provider_spec]
    if not names:
        raise KeyError(f"No profile set for baseline provider {provider_spec}")
    return {
        "provider_spec": provider_spec,
        "profile_name": str(names[0]),
        "short_window_seconds": float(baseline["short_window_seconds"]),
        "long_window_seconds": float(baseline["long_window_seconds"]),
        "long_weight": float(baseline["long_weight"]),
        "algorithm_config": dict(baseline["algorithm_config"]),
    }


def _inputs(input_root: Path, description: dict[str, Any], video_id: str) -> dict[str, Any]:
    short_window = float(description["short_window_seconds"])
    gate_root = input_root / "gate_sets" / f"{round(short_window * 1000):04d}ms" / video_id
    profiles = load_profile_events_jsonl(
        input_root / "profiles" / str(description["profile_name"]) / video_id
        / "production_stack.profiles.jsonl"
    )
    return {
        "profiles": profiles,
        "canonical": read_canonical_segments(
            input_root / "references" / video_id / "canonical_diarization.json"
        ),
        "speech": np.load(gate_root / "speech_gate.u1.npy", allow_pickle=False),
        "probes": np.load(gate_root / "probe_schedule.u1.npy", allow_pickle=False),
        "releases": np.load(gate_root / "release_gate.u1.npy", allow_pickle=False),
        "gate_root": gate_root.parent,
    }


def verify_description(
    *,
    corpus_root: Path,
    input_root: Path,
    videos: list[str],
    description: dict[str, Any],
    provider_endpoint: str,
    device: str,
) -> dict[str, Any]:
    provider_spec = str(description["provider_spec"])
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
        if float(weight) > 0.0
    ]
    providers = {
        provider: RemotePreparedEmbeddingProvider(provider_endpoint, provider, device)
        for provider, _weight in provider_specs
    }
    short_window = float(description["short_window_seconds"])
    long_value = description.get("long_window_seconds")
    windows = [short_window] + ([] if long_value is None else [float(long_value)])
    fresh_blocks: dict[tuple[str, float], Any] = {}
    generation_reports: list[dict[str, Any]] = []
    prepared_inputs = {
        video_id: _inputs(input_root, description, video_id) for video_id in videos
    }
    for video_id in videos:
        source = json.loads(
            (corpus_root / "videos" / video_id / "source.json").read_text(encoding="utf-8-sig")
        )
        media_root = Path(str(source["audio_path_at_creation"])).resolve().parent
        for window in windows:
            block, report = _fresh_block(
                corpus_root=corpus_root,
                media_root=media_root,
                input_root=prepared_inputs[video_id]["gate_root"],
                provider_specs=provider_specs,
                providers=providers,
                provider_spec=provider_spec,
                video_id=video_id,
                window_seconds=window,
            )
            fresh_blocks[(video_id, window)] = block
            generation_reports.append(report)

    config = LiveSpeakerAlgorithmConfig(**description["algorithm_config"])
    fresh_scores: dict[str, Any] = {}
    cached_scores: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    for video_id in videos:
        inputs = prepared_inputs[video_id]
        cached_short = _cached_stack(
            corpus_root, provider_specs, video_id, short_window, provider_spec
        )
        if long_value is None:
            fresh_decisions = replay_cached_live_windows(
                fresh_blocks[(video_id, short_window)],
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                config=config,
            )
            cached_decisions = replay_cached_live_windows(
                cached_short,
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                config=config,
            )
        else:
            long_window = float(long_value)
            cached_long = _cached_stack(
                corpus_root, provider_specs, video_id, long_window, provider_spec
            )
            fresh_decisions = replay_cached_live_windows_dual(
                fresh_blocks[(video_id, short_window)], fresh_blocks[(video_id, long_window)],
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                long_weight=float(description["long_weight"]), config=config,
            )
            cached_decisions = replay_cached_live_windows_dual(
                cached_short, cached_long,
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                long_weight=float(description["long_weight"]), config=config,
            )
        fresh_scores[video_id] = score_live_speaker_decisions(
            fresh_decisions, inputs["canonical"], inputs["profiles"]
        )
        cached_scores[video_id] = score_live_speaker_decisions(
            cached_decisions, inputs["canonical"], inputs["profiles"]
        )
        parity[video_id] = compare_decision_traces(fresh_decisions, cached_decisions)

    fresh_aggregate = aggregate_video_scores_primary_v2(fresh_scores.values())
    cached_aggregate = aggregate_video_scores_primary_v2(cached_scores.values())
    return {
        "description": description,
        "provider_spec": provider_spec,
        "windows_freshly_embedded": windows,
        "generation_reports": generation_reports,
        "fresh_aggregate": fresh_aggregate,
        "cached_aggregate": cached_aggregate,
        "fresh_per_video": fresh_scores,
        "cached_per_video": cached_scores,
        "parity": parity,
        "all_generation_succeeded": all(
            int(row["failure_count"]) == 0 for row in generation_reports
        ),
        "all_cached_decisions_exact": all(bool(row["exact_match"]) for row in parity.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freshly regenerate and parity-check the top-seven baseline and cache champion."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-champion", action="store_true")
    parser.add_argument(
        "--min-decision-match-ratio",
        type=float,
        default=0.99,
        help=(
            "Minimum fresh-versus-cache visible-decision match ratio per video. "
            "Exact floating-point parity is reported separately."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reassess the existing output without regenerating embeddings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_path = args.champion.resolve()
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    baseline_description = _baseline_description(spec)
    candidate_description = dict(champion["description"])
    output_path = args.output.resolve()
    if args.reuse_existing:
        existing = json.loads(output_path.read_text(encoding="utf-8-sig"))
        baseline = dict(existing["baseline"])
        candidate = dict(existing["candidate"])
    else:
        baseline = verify_description(
            corpus_root=args.corpus_root.resolve(), input_root=args.input_root.resolve(),
            videos=videos, description=baseline_description,
            provider_endpoint=args.provider_endpoint, device=args.device,
        )
        if candidate_description == baseline_description:
            candidate = baseline
        else:
            candidate = verify_description(
                corpus_root=args.corpus_root.resolve(), input_root=args.input_root.resolve(),
                videos=videos, description=candidate_description,
                provider_endpoint=args.provider_endpoint, device=args.device,
            )
    baseline_score = float(baseline["fresh_aggregate"]["primary_score"])
    candidate_score = float(candidate["fresh_aggregate"]["primary_score"])
    baseline_minimum_match = min(
        float(row["decision_match_ratio"]) for row in baseline["parity"].values()
    )
    candidate_minimum_match = min(
        float(row["decision_match_ratio"]) for row in candidate["parity"].values()
    )
    minimum_match_required = max(0.0, min(1.0, float(args.min_decision_match_ratio)))
    passed = (
        candidate_score > baseline_score + 1e-6
        and bool(baseline["all_generation_succeeded"])
        and bool(candidate["all_generation_succeeded"])
        and baseline_minimum_match >= minimum_match_required
        and candidate_minimum_match >= minimum_match_required
    )
    payload = {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "promotion_policy": "fresh_primary_score_improves_and_per_video_cache_match_ratio_passes",
        "minimum_decision_match_ratio_required": minimum_match_required,
        "videos": videos,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_fresh_score": baseline_score,
        "candidate_fresh_score": candidate_score,
        "fresh_score_delta": round(candidate_score - baseline_score, 6),
        "baseline_minimum_decision_match_ratio": baseline_minimum_match,
        "candidate_minimum_decision_match_ratio": candidate_minimum_match,
        "passed": passed,
    }
    _atomic_json(output_path, payload)
    if args.update_champion:
        champion["fresh_live_verified"] = passed
        champion["fresh_live_verification"] = {
            "verifier_id": VERIFIER_ID,
            "path": str(args.output.resolve()),
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": payload["fresh_score_delta"],
            "all_cached_decisions_exact": bool(candidate["all_cached_decisions_exact"]),
            "minimum_decision_match_ratio_required": minimum_match_required,
            "baseline_minimum_decision_match_ratio": baseline_minimum_match,
            "candidate_minimum_decision_match_ratio": candidate_minimum_match,
            "per_video_regressions_are_diagnostics_only": True,
        }
        champion["status"] = "LIVE_VERIFIED_CHAMPION" if passed else "REJECTED_BY_FRESH_LIVE"
        _atomic_json(champion_path, champion)
    print(json.dumps({
        "passed": passed,
        "baseline_fresh_score": baseline_score,
        "candidate_fresh_score": candidate_score,
        "fresh_score_delta": payload["fresh_score_delta"],
        "candidate_all_cached_decisions_exact": candidate["all_cached_decisions_exact"],
        "baseline_minimum_decision_match_ratio": baseline_minimum_match,
        "candidate_minimum_decision_match_ratio": candidate_minimum_match,
        "minimum_decision_match_ratio_required": minimum_match_required,
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
