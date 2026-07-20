"""Run the fixed twelve-video A/B for the causal profile-quality meta lease.

The production baseline is completed first, followed by the immutable run018
control and exactly one locked ``a005_s035`` candidate.  The runner has no
search space.  It accepts only the already-opened v1/v2/v3 dataset mapping and
rejects every sealed v4 identifier before constructing or resolving a path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
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
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


RUNNER_ID = "locked_live_speaker_hybrid_profile_quality_meta_ab_v1"
META_NAME = "profile_quality_meta_a005_s035"
META_FAMILY = "profile_quality_meta_lease_a005_s035_v1"
META_POLICY_ID = "causal_profile_quality_meta_lease_a005_s035_v1"
EVALUATION_ORDER = ("baseline", "locked_run018", META_NAME)
EXPECTED_PROVIDER = (
    "pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5"
)
EXPECTED_WINDOWS = (0.8, 2.8)
EXPECTED_LONG_WEIGHT = 0.25

EXPECTED_SOURCE_VIDEO_IDS: dict[str, tuple[str, ...]] = {
    "v1": ("Dd7FixvoKBw", "DsyfYJ5Ou3g", "20v1OxUXcQY", "JWS-qfR6K3w"),
    "v2": (
        "e3h6es6zh1c",
        "1NBVQB-Srpw",
        "F2-2RBi1qzY",
        "vIfGgDnmBXg",
        "ZY0DG8rUnCA",
    ),
    "v3": ("pD4IdQTmneI", "k1tsGGz-Qw0", "aHGd6LqAVzw"),
}
EXPECTED_SOURCE_ORDER = tuple(EXPECTED_SOURCE_VIDEO_IDS)
EXPECTED_VIDEO_IDS = tuple(
    video_id
    for label in EXPECTED_SOURCE_ORDER
    for video_id in EXPECTED_SOURCE_VIDEO_IDS[label]
)
FORBIDDEN_V4_IDS = frozenset({"blcKeLDDzSM", "KdOXM3I_5hk", "acbnyagl8jo"})


@dataclass(frozen=True)
class _RawDatasetSource:
    label: str
    corpus_root: str
    input_root: str
    video_ids: tuple[str, ...]


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


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _split_raw_source(raw: str) -> _RawDatasetSource:
    try:
        label_and_corpus, input_root, raw_ids = raw.split(round2.SOURCE_SEPARATOR, 2)
        label, corpus_root = label_and_corpus.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            "dataset source must be LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2"
        ) from exc
    label = label.strip()
    corpus_root = corpus_root.strip()
    input_root = input_root.strip()
    video_ids = tuple(value.strip() for value in raw_ids.split(",") if value.strip())
    if not label or not corpus_root or not input_root or not video_ids:
        raise ValueError("dataset source label, roots, and video IDs must be non-empty")
    return _RawDatasetSource(label, corpus_root, input_root, video_ids)


def _validate_sources(raw_sources: Sequence[str]) -> list[round2.DatasetSource]:
    """Validate the complete identity mapping before creating any ``Path``."""

    if not raw_sources:
        raise ValueError("exactly three v1/v2/v3 dataset sources are required")
    raw = [_split_raw_source(value) for value in raw_sources]

    # This is intentionally the first cohort rule.  In particular it runs
    # before labels, mappings, Path construction, Path.resolve, or run018 load.
    all_ids = [video_id for source in raw for video_id in source.video_ids]
    forbidden = sorted(FORBIDDEN_V4_IDS.intersection(all_ids))
    if forbidden:
        raise ValueError(
            "sealed v4 video IDs are forbidden before path resolution: "
            + ", ".join(forbidden)
        )

    labels = [source.label for source in raw]
    if len(labels) != len(set(labels)):
        raise ValueError("dataset source labels must be unique")
    if set(labels) != set(EXPECTED_SOURCE_ORDER) or len(raw) != 3:
        raise ValueError("dataset sources must be exactly v1, v2, and v3")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("video IDs may occur in only one dataset source")

    by_label = {source.label: source for source in raw}
    for label, expected in EXPECTED_SOURCE_VIDEO_IDS.items():
        actual = by_label[label].video_ids
        if actual != expected:
            raise ValueError(
                f"dataset source {label} must contain the fixed opened IDs in order; "
                f"expected {expected!r}, found {actual!r}"
            )

    # Path objects are constructed only after every identity and seal check.
    return [
        round2.DatasetSource(
            label,
            Path(by_label[label].corpus_root),
            Path(by_label[label].input_root),
            by_label[label].video_ids,
        )
        for label in EXPECTED_SOURCE_ORDER
    ]


def _dataset_source_lock(sources: Sequence[round2.DatasetSource]) -> dict[str, Any]:
    payload = {
        "mapping_version": "opened_live_replay_v1_v2_v3_12_v1",
        "source_order": list(EXPECTED_SOURCE_ORDER),
        "video_ids": list(EXPECTED_VIDEO_IDS),
        "sources": [
            {
                "label": source.label,
                "corpus_root": str(source.corpus_root),
                "input_root": str(source.input_root),
                "video_ids": list(source.video_ids),
            }
            for source in sources
        ],
    }
    return {**payload, "lock_sha256": _sha256_value(payload)}


def _candidate_id(config: HybridSpeakerTrackerConfig) -> str:
    payload = {
        "runner_id": RUNNER_ID,
        "source_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "family": META_FAMILY,
        "policy_id": META_POLICY_ID,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(config),
    }
    return _sha256_value(payload)


def _fixed_candidates(
    locked_config: HybridSpeakerTrackerConfig,
) -> tuple[round2.Candidate, round2.Candidate]:
    if bool(locked_config.enable_short_scale_fast_lease):
        raise ValueError("run018 must keep the unrestricted short-scale lease disabled")
    if bool(locked_config.enable_profile_quality_short_scale_fast_lease):
        raise ValueError("run018 must keep the old profile-quality output path disabled")
    if bool(locked_config.enable_profile_quality_meta_lease):
        raise ValueError("run018 already enables the profile-quality meta experiment")

    expected_values = {
        "enable_short_scale_fast_lease": False,
        "enable_profile_quality_short_scale_fast_lease": False,
        "enable_profile_quality_meta_lease": True,
        "profile_quality_fast_lease_min_sentence_count": 2,
        "profile_quality_fast_lease_min_speech_seconds": 3.1,
        "profile_quality_fast_lease_min_similarity": 0.18,
        "profile_quality_fast_lease_min_margin": 0.06,
        "profile_quality_meta_fresh_min_age_seconds": 0.05,
        "profile_quality_meta_fresh_max_age_seconds": 0.8,
        "profile_quality_meta_fresh_min_speech_seconds": 3.8,
        "profile_quality_meta_fresh_min_short_margin": 0.30,
        "profile_quality_meta_fresh_min_long_margin": 0.70,
        "profile_quality_meta_independent_max_profile_count": 8,
        "profile_quality_meta_switch_min_short_margin": 0.35,
    }
    candidate_config = replace(locked_config, **expected_values)
    candidate_payload = asdict(candidate_config)
    for key, expected in expected_values.items():
        if candidate_payload.get(key) != expected:
            raise AssertionError(f"profile-quality meta lock mismatch for {key}")
    changed = {
        key
        for key, value in asdict(locked_config).items()
        if candidate_payload.get(key) != value
    }
    if "enable_profile_quality_meta_lease" not in changed:
        raise AssertionError("fixed meta candidate did not differ from run018")
    if not changed.issubset(expected_values):
        raise AssertionError(f"unexpected profile-quality meta config delta: {sorted(changed)}")

    return (
        round2.Candidate(
            "locked_run018",
            "locked_run018_control",
            locked_config,
            round2.LOCKED_RUN018_CANDIDATE_ID,
        ),
        round2.Candidate(
            META_NAME,
            META_FAMILY,
            candidate_config,
            _candidate_id(candidate_config),
        ),
    )


def _candidate_lock(
    locked: Any,
    candidates: Sequence[round2.Candidate],
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    source_config = asdict(locked.config)
    payload = {
        "source_candidate_id": locked.candidate_id,
        "source_family": locked.family,
        "source_champion_path": str(locked.champion_path),
        "source_run_path": str(locked.run_path),
        "source_champion_sha256": locked.champion_sha256,
        "source_run_sha256": locked.run_sha256,
        "source_config": source_config,
        "source_config_sha256": _sha256_value(source_config),
        "dataset_source_lock_sha256": source_lock["lock_sha256"],
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "provider": locked.provider,
        "windows_seconds": list(EXPECTED_WINDOWS),
        "long_weight": EXPECTED_LONG_WEIGHT,
        "candidate_count": len(candidates),
        "search_space": "none",
        "fixed_meta_contract": {
            "policy_id": META_POLICY_ID,
            "candidate_name": META_NAME,
            "fresh_min_age_seconds": 0.05,
            "switch_min_short_margin": 0.35,
            "old_profile_quality_output_path_enabled": False,
            "maximum_fresh_embedding_windows_per_probe": 2,
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
    return {**payload, "lock_sha256": _sha256_value(payload)}


def _assert_exact_result(result: dict[str, Any], label: str) -> None:
    aggregate = result.get("aggregate")
    per_video = result.get("per_video")
    if not isinstance(aggregate, dict) or not isinstance(per_video, dict):
        raise ValueError(f"{label} has no aggregate/per-video score payload")
    if set(per_video) != set(EXPECTED_VIDEO_IDS) or len(per_video) != 12:
        raise ValueError(f"{label} does not cover the exact twelve-video cohort")
    global_score = float(aggregate.get("global_score"))
    if not math.isfinite(global_score):
        raise ValueError(f"{label} global score is not finite")
    for video_id in EXPECTED_VIDEO_IDS:
        row = per_video[video_id]
        strict = float(row["strict_browser_live_score"])
        wrong = float(row["wrong_live_speech_ratio"])
        if not math.isfinite(strict) or not math.isfinite(wrong):
            raise ValueError(f"{label} has non-finite score data for {video_id}")


def _promotion_decision(
    baseline: dict[str, Any],
    locked: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    _assert_exact_result(baseline, "production baseline")
    _assert_exact_result(locked, "run018 control")
    _assert_exact_result(candidate, "meta candidate")
    locked_vs_baseline = round2._comparison(locked, baseline, "production_baseline")
    candidate_vs_baseline = round2._comparison(
        candidate, baseline, "production_baseline"
    )
    candidate_vs_locked = round2._comparison(candidate, locked, "locked_run018")
    baseline_gates = round2._comparison_passed(candidate_vs_baseline)
    locked_gates = round2._comparison_passed(candidate_vs_locked)
    eligible = bool(baseline_gates and locked_gates)
    return {
        "winner": META_NAME if eligible else None,
        "reason": (
            "fixed meta candidate passes aggregate and every per-video score/wrong gate "
            "versus both production baseline and run018"
            if eligible
            else "fixed meta candidate failed at least one mandatory baseline or run018 gate"
        ),
        "candidate_eligible": eligible,
        "baseline_gates_passed": baseline_gates,
        "run018_gates_passed": locked_gates,
        "locked_vs_baseline": locked_vs_baseline,
        "candidate_vs_baseline": candidate_vs_baseline,
        "candidate_vs_locked": candidate_vs_locked,
        "score_tolerance": round2.SCORE_TOLERANCE,
        "wrong_ratio_tolerance": round2.WRONG_RATIO_TOLERANCE,
        "required_video_count": 12,
    }


def _fresh_live_cost() -> dict[str, Any]:
    cost = round2._fresh_live_cost()
    if cost["fresh_window_requests_per_probe"] != 2:
        raise AssertionError("fresh live cost no longer uses exactly two windows")
    if cost["max_fresh_window_requests_per_probe"] != 2:
        raise AssertionError("fresh live cost permits more than two windows")
    if cost["windows_seconds"] != list(EXPECTED_WINDOWS):
        raise AssertionError("fresh live window pair changed")
    if cost["long_weight"] != EXPECTED_LONG_WEIGHT:
        raise AssertionError("fresh live long-window weight changed")
    return cost


def _run_metadata(
    sources: Sequence[round2.DatasetSource],
    source_lock: dict[str, Any],
    candidate_lock: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "initialized",
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "selection_policy": "fixed_profile_quality_meta_a005_s035_no_search",
        "parameter_search_performed": False,
        "evaluation_order": list(EVALUATION_ORDER),
        "baseline_must_complete_before_candidates": True,
        "baseline_completed_before_candidate_evaluation": False,
        "dataset_source_lock": source_lock,
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
        "video_ids": list(EXPECTED_VIDEO_IDS),
        "video_count": 12,
        "sealed_v4_ids_rejected": sorted(FORBIDDEN_V4_IDS),
        "sealed_v4_opened": False,
        "provider": EXPECTED_PROVIDER,
        "windows_seconds": list(EXPECTED_WINDOWS),
        "long_weight": EXPECTED_LONG_WEIGHT,
        "fresh_live_cost": _fresh_live_cost(),
    }


def _progress(phase: str, completed_steps: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "completed_steps": completed_steps,
        "total_steps": 3,
        "progress_percent": round(100.0 * completed_steps / 3.0, 2),
        "evaluation_order": list(EVALUATION_ORDER),
        "baseline_first": True,
        "video_count": 12,
        "sealed_v4_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources = _validate_sources(args.dataset_source)

    # Only opened v1/v2/v3 identities exist beyond this point.
    locked = round2._load_run018(args.locked_run_dir)
    if locked.provider != EXPECTED_PROVIDER or sweep.DEFAULT_PROVIDER != EXPECTED_PROVIDER:
        raise ValueError("run018 provider differs from the exact fixed provider stack")
    candidates = _fixed_candidates(locked.config)
    if len(candidates) != 2:
        raise AssertionError("formal meta A/B must contain exactly control and candidate")
    source_lock = _dataset_source_lock(sources)
    candidate_lock = _candidate_lock(locked, candidates, source_lock)

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
        raise FileExistsError("formal profile-quality meta artifacts already exist")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    run = _run_metadata(sources, source_lock, candidate_lock)
    round2._atomic_json(run_dir / "run.json", run)
    round2._atomic_json(run_dir / "progress.json", _progress("INITIALIZED", 0))
    round2._atomic_jsonl(run_dir / "trials.jsonl", [])

    prepared, baseline = round2._prepare_baseline(sources)
    _assert_exact_result(baseline, "production baseline")
    baseline["completed_at_step"] = 1
    baseline["evaluation_order"] = list(EVALUATION_ORDER)
    round2._atomic_json(run_dir / "baseline.json", baseline)
    run["status"] = "baseline_complete"
    run["baseline_completed_before_candidate_evaluation"] = True
    round2._atomic_json(run_dir / "run.json", run)
    round2._atomic_json(run_dir / "progress.json", _progress("BASELINE_COMPLETE", 1))
    print("[33.33%] exact twelve-video production baseline complete", flush=True)

    rows: list[dict[str, Any]] = []
    locked_row = round2._score_candidate(prepared, candidates[0])
    _assert_exact_result(locked_row, "run018 control")
    locked_row["evaluation_index"] = 1
    locked_row["control_only"] = True
    locked_row["vs_baseline"] = round2._comparison(
        locked_row, baseline, "production_baseline"
    )
    locked_row["vs_locked"] = round2._comparison(
        locked_row, locked_row, "locked_run018"
    )
    locked_row["promotion_gates_passed"] = False
    rows.append(locked_row)
    round2._atomic_jsonl(run_dir / "trials.jsonl", rows)
    round2._atomic_json(
        run_dir / "progress.json", _progress("LOCKED_RUN018_COMPLETE", 2)
    )
    print("[66.67%] exact locked run018 control complete", flush=True)

    candidate_row = round2._score_candidate(prepared, candidates[1])
    _assert_exact_result(candidate_row, "meta candidate")
    candidate_row["evaluation_index"] = 2
    candidate_row["control_only"] = False
    candidate_row["vs_baseline"] = round2._comparison(
        candidate_row, baseline, "production_baseline"
    )
    candidate_row["vs_locked"] = round2._comparison(
        candidate_row, locked_row, "locked_run018"
    )
    decision = _promotion_decision(baseline, locked_row, candidate_row)
    candidate_row["promotion_gates_passed"] = bool(decision["candidate_eligible"])
    rows.append(candidate_row)
    round2._atomic_jsonl(run_dir / "trials.jsonl", rows)

    winner = candidate_row if decision["candidate_eligible"] else None
    report = {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "complete",
        "dataset_source_lock": source_lock,
        "candidate_lock": candidate_lock,
        "baseline": baseline,
        "trials": rows,
        "promotion": decision,
        "winner": winner,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "parameter_search_performed": False,
        "sealed_v4_opened": False,
    }
    champion = {
        "schema_version": 1,
        "status": (
            "profile_quality_meta_cached_winner_requires_fresh_live_verification"
            if winner is not None
            else "profile_quality_meta_candidate_failed_mandatory_gates"
        ),
        "runner_id": RUNNER_ID,
        "dataset_source_lock_sha256": source_lock["lock_sha256"],
        "candidate_lock_sha256": candidate_lock["lock_sha256"],
        "source_run018_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "winner": winner,
        "promotion": decision,
        "production_defaults_changed": False,
        "fresh_live_verification_required": winner is not None,
        "parameter_search_performed": False,
        "sealed_v4_opened": False,
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
