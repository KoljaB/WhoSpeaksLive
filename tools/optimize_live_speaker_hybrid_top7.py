"""Top-7 single-objective sweep for causal residual corrections.

The verified production champion remains the primary expert.  A residual
tracker may only alter its output for explicitly modeled weak-profile or
speaker-boundary cases, using the same two already-computed embeddings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_overnight_top7 import Dataset, evaluate_candidate
from window.live_speaker_benchmark import (
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)
from window.live_speaker_hybrid import (
    HYBRID_ALGORITHM_ID,
    HybridSpeakerTrackerConfig,
    hybrid_config_identity_payload,
    replay_hybrid_decisions,
)
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_replay import replay_cached_live_windows_dual
from window.live_speaker_algorithm import LiveSpeakerAlgorithmConfig


OPTIMIZER_ID = "live_speaker_top7_residual_hybrid_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _compact(score: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in score.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    for key in ("sampled_playback_seconds", "speaker_map"):
        if key in score:
            result[key] = score[key]
    for key, omitted in (("turn_latency", "turns"), ("release", "events")):
        if isinstance(score.get(key), dict):
            result[key] = {name: value for name, value in score[key].items() if name != omitted}
    return result


def _steps(short: Any, long: Any, inputs: dict[str, Any]) -> list[MultiScaleStep]:
    if not np.array_equal(short.media_times, long.media_times):
        raise ValueError("Hybrid windows must have the same timeline")
    result: list[MultiScaleStep] = []
    for index, media_time in enumerate(short.media_times):
        scheduled = bool(inputs["probes"][index])
        evidences = tuple(
            MultiScaleEvidence(float(block.window_seconds), block.embeddings[index])
            for block in (short, long)
            if scheduled and bool(block.valid[index])
        )
        result.append(MultiScaleStep(
            media_time=float(media_time),
            speech=bool(inputs["speech"][index]),
            evidences=evidences,
            probe_scheduled=scheduled,
            release_signal=bool(inputs["releases"][index]),
            skipped_reason=("" if evidences else "not_a_scheduled_probe" if not scheduled else "cached_embeddings_invalid"),
        ))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_payload = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    champion = dict(champion_payload["description"])
    videos = [str(value) for value in spec["videos"]]
    provider = str(champion["provider_spec"])
    profile_name = str(champion["profile_name"])
    short_window = float(champion["short_window_seconds"])
    long_window = float(champion["long_window_seconds"])
    if long_window <= short_window:
        raise ValueError("Hybrid campaign requires the verified two-window champion")
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name)
    baseline_config = LiveSpeakerAlgorithmConfig(**champion["algorithm_config"])

    prepared: dict[str, dict[str, Any]] = {}
    baseline_per_video: dict[str, Any] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id, short_window)
        short = dataset.block(video_id, short_window)
        long = dataset.block(video_id, long_window)
        decisions = replay_cached_live_windows_dual(
            short, long, inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
            long_weight=float(champion["long_weight"]), config=baseline_config,
        )
        baseline_per_video[video_id] = score_live_speaker_decisions(
            decisions, inputs["canonical"], inputs["profiles"]
        )
        prepared[video_id] = {
            "inputs": inputs,
            "baseline": decisions,
            "steps": _steps(short, long, inputs),
        }
    baseline_aggregate = aggregate_video_scores_primary_v2(baseline_per_video.values())
    baseline_score = float(baseline_aggregate["primary_score"])

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for line in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["candidate_id"])] = row
    elif trials_path.exists():
        raise FileExistsError(f"{trials_path} exists; pass --resume")

    incumbent = max(
        completed.values(),
        key=lambda row: float(row["aggregate"]["primary_score"]),
        default=None,
    )

    def write_state(phase: str, active: str = "") -> None:
        best_score = float(incumbent["aggregate"]["primary_score"]) if incumbent else baseline_score
        payload = {
            "status": "interrupted" if _STOP else "running",
            "phase": phase,
            "active": active,
            "completed_candidate_count": len(completed),
            "baseline_score": baseline_score,
            "best_hybrid_score": best_score if incumbent else None,
            "score_delta": round(best_score - baseline_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        _atomic_json(run_dir / "progress.json", payload)
        if incumbent:
            _atomic_json(run_dir / "champion.json", {
                "status": "CACHE_HYBRID_WINNER_PENDING_PRODUCTION_INTEGRATION" if best_score > baseline_score else "BELOW_BASELINE",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "baseline_score": baseline_score,
                "candidate_score": best_score,
                "score_delta": round(best_score - baseline_score, 6),
                **incumbent,
                "fresh_live_verified": False,
            })

    def evaluate(name: str, family: str, config: HybridSpeakerTrackerConfig) -> dict[str, Any] | None:
        nonlocal incumbent
        candidate_id = _stable_id({
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "name": name,
            "config": hybrid_config_identity_payload(config),
        })
        if candidate_id in completed:
            return completed[candidate_id]
        if _STOP or time.monotonic() >= deadline:
            return None
        write_state(family, candidate_id)
        per_video: dict[str, Any] = {}
        for video_id, value in prepared.items():
            decisions = replay_hybrid_decisions(
                value["baseline"], value["steps"], value["inputs"]["profiles"], config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, value["inputs"]["canonical"], value["inputs"]["profiles"]
            ))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        if not math.isfinite(float(aggregate["primary_score"])):
            raise RuntimeError("Non-finite hybrid score")
        row = {
            "candidate_id": candidate_id,
            "name": name,
            "family": family,
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "config": asdict(config),
            "aggregate": aggregate,
            "per_video": per_video,
            "score_delta_vs_baseline": round(float(aggregate["primary_score"]) - baseline_score, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[candidate_id] = row
        _append_jsonl(trials_path, row)
        if incumbent is None or float(aggregate["primary_score"]) > float(incumbent["aggregate"]["primary_score"]):
            incumbent = row
        write_state(family, candidate_id)
        return row

    _atomic_json(run_dir / "run.json", {
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "baseline_description": champion,
        "baseline_score": baseline_score,
        "videos": videos,
        "maximum_fresh_windows_per_probe": 2,
        "selection_policy": "one Top-7 primary score",
    })
    _atomic_json(run_dir / "baseline_reproduction.json", {
        "status": "REPRODUCED",
        "aggregate": baseline_aggregate,
        "per_video": baseline_per_video,
    })
    write_state("BASELINE")

    control = HybridSpeakerTrackerConfig()
    evaluate("no_residual_intervention", "CONTROL", control)

    young_rows: list[dict[str, Any]] = []
    for similarity, margin, permanent_scales, required, lease in itertools.product(
        (0.25, 0.30, 0.35, 0.40), (0.03, 0.05, 0.08, 0.12), (1, 2), (1, 2), (False, True)
    ):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=lease,
            young_trusted_min_sentence_count=4,
            young_trusted_min_speech_seconds=8.0,
            young_min_similarity=similarity,
            young_min_margin=margin,
            young_required_consecutive_probes=required,
            young_independent_scale_count=permanent_scales,
            young_fast_independent_scale_count=1,
        )
        row = evaluate(
            f"young_s{similarity:g}_m{margin:g}_n{permanent_scales}_r{required}_lease{int(lease)}",
            "YOUNG_PROFILE_EXPERT", config,
        )
        if row:
            young_rows.append(row)

    # Add boundary abstention only to the strongest young-profile lineages.
    parents = sorted(young_rows, key=lambda row: float(row["aggregate"]["primary_score"]), reverse=True)[:8]
    for parent in parents:
        source = HybridSpeakerTrackerConfig(**parent["config"])
        for minimum, margin, short_advantage, long_advantage, required in itertools.product(
            (0.25, 0.30, 0.35), (0.03, 0.06), (0.03, 0.06), (0.03, 0.06), (1, 2)
        ):
            evaluate(
                f"{parent['name']}__boundary_{minimum:g}_{margin:g}_{short_advantage:g}_{long_advantage:g}_{required}",
                "BOUNDARY_ABSTENTION",
                replace(
                    source,
                    enable_boundary_abstention=True,
                    boundary_min_similarity=minimum,
                    boundary_min_margin=margin,
                    boundary_short_advantage=short_advantage,
                    boundary_long_advantage=long_advantage,
                    boundary_required_consecutive_probes=required,
                ),
            )

    # Retest the three strongest previously developed residual policies on the
    # new provider, new geometry, new seven-video objective.
    locked = HybridSpeakerTrackerConfig(
        enable_young_profile_confirmation=True,
        enable_young_profile_lease=True,
        young_trusted_min_sentence_count=4,
        young_trusted_min_speech_seconds=8.0,
        young_min_similarity=0.35,
        young_min_margin=0.08,
        young_required_consecutive_probes=1,
        young_independent_scale_count=2,
        young_fast_independent_scale_count=1,
    )
    evaluate("prior_locked_run018", "PRIOR_POLICY_RETEST", locked)
    evaluate("prior_profile_quality_lease", "PRIOR_POLICY_RETEST", replace(
        locked,
        enable_profile_quality_short_scale_fast_lease=True,
        profile_quality_fast_lease_min_sentence_count=2,
        profile_quality_fast_lease_min_speech_seconds=3.1,
        profile_quality_fast_lease_min_similarity=0.18,
        profile_quality_fast_lease_min_margin=0.06,
    ))
    evaluate("prior_meta_a005_s035", "PRIOR_POLICY_RETEST", replace(
        locked,
        enable_profile_quality_meta_lease=True,
        profile_quality_fast_lease_min_sentence_count=2,
        profile_quality_fast_lease_min_speech_seconds=3.1,
        profile_quality_fast_lease_min_similarity=0.18,
        profile_quality_fast_lease_min_margin=0.06,
        profile_quality_meta_fresh_min_age_seconds=0.05,
        profile_quality_meta_fresh_max_age_seconds=0.8,
        profile_quality_meta_fresh_min_speech_seconds=3.8,
        profile_quality_meta_fresh_min_short_margin=0.30,
        profile_quality_meta_fresh_min_long_margin=0.70,
        profile_quality_meta_independent_max_profile_count=8,
        profile_quality_meta_switch_min_short_margin=0.35,
    ))

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    best_score = float(incumbent["aggregate"]["primary_score"]) if incumbent else baseline_score
    _atomic_json(run_dir / "final_report.json", {
        "status": progress["status"],
        "baseline_score": baseline_score,
        "champion_score": best_score if incumbent else None,
        "score_delta": round(best_score - baseline_score, 6),
        "candidate_count": len(completed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps(progress, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
