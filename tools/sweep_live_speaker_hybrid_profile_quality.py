"""Formal fixed A/B for the profile-quality short-scale fast lease.

The production baseline is evaluated first.  The only two subsequent rows are
the immutable run018 control and one locked profile-quality candidate.  This
program contains no provider, window, threshold, or candidate search space.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid as sweep
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_hybrid import (
    HYBRID_ALGORITHM_ID,
    HybridSpeakerTrackerConfig,
    PROFILE_QUALITY_META_CONFIG_FIELDS,
)


RUNNER_ID = "locked_live_speaker_hybrid_profile_quality_ab_v1"
PROFILE_QUALITY_FAMILY = "profile_quality_short_scale_fast_lease_v1"
PROFILE_QUALITY_NAME = "profile_quality_short_scale_fast_lease"
EVALUATION_ORDER = ("baseline", "locked_run018", PROFILE_QUALITY_NAME)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        action="append",
        required=True,
        metavar="LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2",
    )
    parser.add_argument("--locked-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_id(config: HybridSpeakerTrackerConfig) -> str:
    config_payload = asdict(config)
    if not bool(config_payload.get("enable_profile_quality_meta_lease")):
        for name in PROFILE_QUALITY_META_CONFIG_FIELDS:
            config_payload.pop(name, None)
    payload = {
        "runner_id": RUNNER_ID,
        "source_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "family": PROFILE_QUALITY_FAMILY,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": config_payload,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _fixed_candidates(
    locked_config: HybridSpeakerTrackerConfig,
) -> tuple[round2.Candidate, round2.Candidate]:
    if bool(locked_config.enable_short_scale_fast_lease):
        raise ValueError("run018 must keep the unrestricted short-scale lease disabled")
    if bool(locked_config.enable_profile_quality_short_scale_fast_lease):
        raise ValueError("run018 already enables the profile-quality experiment")
    candidate_config = replace(
        locked_config,
        enable_short_scale_fast_lease=False,
        enable_profile_quality_short_scale_fast_lease=True,
        profile_quality_fast_lease_min_sentence_count=2,
        profile_quality_fast_lease_min_speech_seconds=3.1,
        profile_quality_fast_lease_min_similarity=0.18,
        profile_quality_fast_lease_min_margin=0.06,
    )
    locked_payload = asdict(locked_config)
    candidate_payload = asdict(candidate_config)
    expected_values = {
        "enable_short_scale_fast_lease": False,
        "enable_profile_quality_short_scale_fast_lease": True,
        "profile_quality_fast_lease_min_sentence_count": 2,
        "profile_quality_fast_lease_min_speech_seconds": 3.1,
        "profile_quality_fast_lease_min_similarity": 0.18,
        "profile_quality_fast_lease_min_margin": 0.06,
    }
    for key, value in expected_values.items():
        if candidate_payload.get(key) != value:
            raise AssertionError(f"profile-quality lock mismatch for {key}")
    allowed_changes = set(expected_values) - {"enable_short_scale_fast_lease"}
    changed = {
        key for key, value in locked_payload.items() if candidate_payload.get(key) != value
    }
    if not changed or not changed.issubset(allowed_changes):
        raise AssertionError(f"unexpected profile-quality config delta: {sorted(changed)}")
    return (
        round2.Candidate(
            "locked_run018",
            "locked_run018_control",
            locked_config,
            round2.LOCKED_RUN018_CANDIDATE_ID,
        ),
        round2.Candidate(
            PROFILE_QUALITY_NAME,
            PROFILE_QUALITY_FAMILY,
            candidate_config,
            _candidate_id(candidate_config),
        ),
    )


def _candidate_lock(locked: Any, candidates: Sequence[round2.Candidate]) -> dict[str, Any]:
    payload = {
        "source_candidate_id": locked.candidate_id,
        "source_family": locked.family,
        "source_champion_path": str(locked.champion_path),
        "source_run_path": str(locked.run_path),
        "source_champion_sha256": locked.champion_sha256,
        "source_run_sha256": locked.run_sha256,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "provider": locked.provider,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "candidate_count": len(candidates),
        "search_space": "none",
        "fixed_profile_quality_contract": {
            "eligibility": "sentence_count>=2 OR speech_seconds>=3.1",
            "short_min_similarity": 0.18,
            "short_min_margin": 0.06,
            "enable_short_scale_fast_lease": False,
        },
        "candidates": [
            {
                "name": candidate.name,
                "family": candidate.family,
                "candidate_id": candidate.candidate_id,
                "config": asdict(candidate.config),
            }
            for candidate in candidates
        ],
    }
    return {**payload, "lock_sha256": hashlib.sha256(
        _stable_json(payload).encode("utf-8")
    ).hexdigest()}


def _promotion_decision(
    baseline: dict[str, Any], locked: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    locked_vs_baseline = round2._comparison(locked, baseline, "production_baseline")
    candidate_vs_baseline = round2._comparison(
        candidate, baseline, "production_baseline"
    )
    candidate_vs_locked = round2._comparison(candidate, locked, "locked_run018")
    baseline_gates = round2._comparison_passed(candidate_vs_baseline)
    locked_gates = round2._comparison_passed(candidate_vs_locked)
    eligible = bool(baseline_gates and locked_gates)
    return {
        "winner": PROFILE_QUALITY_NAME if eligible else None,
        "reason": (
            "candidate passes every aggregate and per-video gate versus baseline and run018"
            if eligible
            else "candidate failed at least one mandatory baseline or run018 gate"
        ),
        "candidate_eligible": eligible,
        "baseline_gates_passed": baseline_gates,
        "run018_gates_passed": locked_gates,
        "locked_vs_baseline": locked_vs_baseline,
        "candidate_vs_baseline": candidate_vs_baseline,
        "candidate_vs_locked": candidate_vs_locked,
        "score_tolerance": round2.SCORE_TOLERANCE,
        "wrong_ratio_tolerance": round2.WRONG_RATIO_TOLERANCE,
    }


def _run_metadata(
    sources: Sequence[round2.DatasetSource], candidate_lock: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "initialized",
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "selection_policy": "fixed_profile_quality_candidate_no_search",
        "evaluation_order": list(EVALUATION_ORDER),
        "baseline_must_complete_before_candidates": True,
        "baseline_completed_before_candidate_evaluation": False,
        "candidate_lock": candidate_lock,
        "dataset_sources": [
            {
                "label": source.label,
                "corpus_root": str(source.corpus_root),
                "input_root": str(source.input_root),
                "video_ids": list(source.video_ids),
            }
            for source in sources
        ],
        "video_count": sum(len(source.video_ids) for source in sources),
        "sealed_v3_ids_rejected": sorted(round2.FORBIDDEN_V3_IDS),
        "sealed_v3_opened": False,
        "provider": sweep.DEFAULT_PROVIDER,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "fresh_live_cost": round2._fresh_live_cost(),
    }


def _progress(phase: str, completed_steps: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "completed_steps": completed_steps,
        "total_steps": 3,
        "progress_percent": round(100.0 * completed_steps / 3.0, 2),
        "evaluation_order": list(EVALUATION_ORDER),
        "baseline_first": True,
        "sealed_v3_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # This validates all video IDs, including the v3 seal, before any root is
    # resolved and before Dataset can touch the filesystem.
    sources = round2._validate_sources(args.dataset_source)
    locked = round2._load_run018(args.locked_run_dir)
    candidates = _fixed_candidates(locked.config)
    if len(candidates) != 2:
        raise AssertionError("formal profile-quality A/B must contain exactly two candidates")
    candidate_lock = _candidate_lock(locked, candidates)

    run_dir = args.run_dir.resolve()
    locked_dir = locked.run_dir.resolve()
    if run_dir == locked_dir or run_dir.is_relative_to(locked_dir):
        raise ValueError("output must remain outside the immutable run018 directory")
    artifact_names = (
        "run.json",
        "baseline.json",
        "progress.json",
        "trials.jsonl",
        "report.json",
        "champion.json",
    )
    if any((run_dir / name).exists() for name in artifact_names):
        raise FileExistsError("formal profile-quality artifacts already exist")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    run = _run_metadata(sources, candidate_lock)
    round2._atomic_json(run_dir / "run.json", run)
    round2._atomic_json(run_dir / "progress.json", _progress("INITIALIZED", 0))

    prepared, baseline = round2._prepare_baseline(sources)
    baseline["completed_at_step"] = 1
    baseline["evaluation_order"] = list(EVALUATION_ORDER)
    round2._atomic_json(run_dir / "baseline.json", baseline)
    run["status"] = "baseline_complete"
    run["baseline_completed_before_candidate_evaluation"] = True
    round2._atomic_json(run_dir / "run.json", run)
    round2._atomic_json(run_dir / "progress.json", _progress("BASELINE_COMPLETE", 1))
    print("[33.33%] production baseline complete", flush=True)

    rows: list[dict[str, Any]] = []
    locked_row = round2._score_candidate(prepared, candidates[0])
    locked_row["evaluation_index"] = 1
    locked_row["vs_baseline"] = round2._comparison(
        locked_row, baseline, "production_baseline"
    )
    locked_row["vs_locked"] = round2._comparison(
        locked_row, locked_row, "locked_run018"
    )
    rows.append(locked_row)
    round2._atomic_jsonl(run_dir / "trials.jsonl", rows)
    round2._atomic_json(
        run_dir / "progress.json", _progress("LOCKED_RUN018_COMPLETE", 2)
    )
    print("[66.67%] exact locked run018 control complete", flush=True)

    candidate_row = round2._score_candidate(prepared, candidates[1])
    candidate_row["evaluation_index"] = 2
    candidate_row["vs_baseline"] = round2._comparison(
        candidate_row, baseline, "production_baseline"
    )
    candidate_row["vs_locked"] = round2._comparison(
        candidate_row, locked_row, "locked_run018"
    )
    rows.append(candidate_row)
    decision = _promotion_decision(baseline, locked_row, candidate_row)
    locked_row["promotion_gates_passed"] = False
    candidate_row["promotion_gates_passed"] = bool(decision["candidate_eligible"])
    round2._atomic_jsonl(run_dir / "trials.jsonl", rows)

    winner = candidate_row if decision["candidate_eligible"] else None
    report = {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "complete",
        "candidate_lock": candidate_lock,
        "baseline": baseline,
        "trials": rows,
        "promotion": decision,
        "winner": winner,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "sealed_v3_opened": False,
    }
    champion = {
        "schema_version": 1,
        "status": (
            "profile_quality_cached_winner_requires_fresh_live_verification"
            if winner is not None
            else "profile_quality_candidate_failed_mandatory_gates"
        ),
        "runner_id": RUNNER_ID,
        "candidate_lock_sha256": candidate_lock["lock_sha256"],
        "source_run018_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "winner": winner,
        "promotion": decision,
        "production_defaults_changed": False,
        "fresh_live_verification_required": winner is not None,
        "sealed_v3_opened": False,
    }
    round2._atomic_json(run_dir / "report.json", report)
    round2._atomic_json(run_dir / "champion.json", champion)
    run["status"] = "complete"
    run["winner"] = None if winner is None else winner["name"]
    run["elapsed_seconds"] = round(time.monotonic() - started, 6)
    round2._atomic_json(run_dir / "run.json", run)
    round2._atomic_json(run_dir / "progress.json", _progress("COMPLETE", 3))
    print(f"[100.00%] complete; winner={run['winner']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
