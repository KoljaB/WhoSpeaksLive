"""Formal two-candidate A/B evaluation for the locked live-speaker hybrid.

This runner combines explicitly mapped cached-data sources, reproduces the
production baseline first, and evaluates only the immutable locked candidate
and its one-field partial-fast-scale-lease variant.  It performs no search.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for value in (SRC, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from evaluate_locked_live_speaker_hybrid_cohort import LockedCandidate, _load_locked_candidate
from optimize_live_speaker_replay import Dataset, _trace_hash
import sweep_live_speaker_hybrid as sweep
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_hybrid import (
    HYBRID_ALGORITHM_ID,
    HybridSpeakerTrackerConfig,
    hybrid_config_identity_payload,
    replay_hybrid_decisions,
)
from window.live_speaker_replay import replay_cached_live_windows_dual


RUNNER_ID = "locked_live_speaker_hybrid_round2_ab_v1"
FORBIDDEN_V3_IDS = frozenset({"pD4IdQTmneI", "k1tsGGz-Qw0", "aHGd6LqAVzw"})
SOURCE_SEPARATOR = "::"
SCORE_TOLERANCE = 0.005
WRONG_RATIO_TOLERANCE = 0.005
LOCKED_RUN018_CANDIDATE_ID = "0f58c8894d2c4f0190e30ec072fdad29de485151a2a76803a2abeec96a0e845f"
EVALUATION_ORDER = ("baseline", "locked_run018", "short_scale_fast_lease")


@dataclass(frozen=True)
class DatasetSource:
    label: str
    corpus_root: Path
    input_root: Path
    video_ids: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    config: HybridSpeakerTrackerConfig
    candidate_id: str


@dataclass
class PreparedVideo:
    source_label: str
    inputs: dict[str, Any]
    baseline_decisions: list[Any]
    steps: list[Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_dataset_source(raw: str) -> DatasetSource:
    try:
        label_and_corpus, input_root, raw_ids = raw.split(SOURCE_SEPARATOR, 2)
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
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(f"dataset source {label!r} contains duplicate video IDs")
    forbidden = sorted(FORBIDDEN_V3_IDS.intersection(video_ids))
    if forbidden:
        raise ValueError("sealed v3 video IDs are forbidden: " + ", ".join(forbidden))
    return DatasetSource(label, Path(corpus_root), Path(input_root), video_ids)


def _validate_sources(raw_sources: Sequence[str]) -> list[DatasetSource]:
    if not raw_sources:
        raise ValueError("at least one --dataset-source is required")
    sources = [_parse_dataset_source(raw) for raw in raw_sources]
    labels = [source.label for source in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("dataset source labels must be unique")
    video_ids = [video_id for source in sources for video_id in source.video_ids]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("video IDs may occur in only one dataset source")
    forbidden = sorted(FORBIDDEN_V3_IDS.intersection(video_ids))
    if forbidden:
        raise ValueError("sealed v3 video IDs are forbidden: " + ", ".join(forbidden))
    return sources


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


def _load_run018(path: Path) -> LockedCandidate:
    locked = _load_locked_candidate(locked_run_dir=path)
    if locked.candidate_id != LOCKED_RUN018_CANDIDATE_ID:
        raise ValueError(
            "locked source is not exact run018 candidate " + LOCKED_RUN018_CANDIDATE_ID
        )
    if locked.provider != sweep.DEFAULT_PROVIDER:
        raise ValueError("run018 provider stack differs from the fixed production stack")
    if locked.short_window_seconds != sweep.SHORT_WINDOW_SECONDS:
        raise ValueError("run018 short window is not fixed at 0.8 seconds")
    if locked.long_window_seconds != sweep.LONG_WINDOW_SECONDS:
        raise ValueError("run018 long window is not fixed at 2.8 seconds")
    if locked.long_weight != sweep.LONG_WEIGHT:
        raise ValueError("run018 long weight is not fixed at 0.25")
    return locked


def _candidate_id(name: str, config: HybridSpeakerTrackerConfig) -> str:
    config_payload = hybrid_config_identity_payload(config)
    return _sha256_value(
        {
            "runner_id": RUNNER_ID,
            "source_candidate_id": LOCKED_RUN018_CANDIDATE_ID,
            "name": name,
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "config": config_payload,
        }
    )


def _locked_candidates(config: HybridSpeakerTrackerConfig) -> tuple[Candidate, Candidate]:
    if bool(config.enable_short_scale_fast_lease):
        raise ValueError("run018 control already enables the round-two behavior")
    partial = replace(config, enable_short_scale_fast_lease=True)
    changed = {
        key
        for key, value in asdict(config).items()
        if asdict(partial).get(key) != value
    }
    if changed != {"enable_short_scale_fast_lease"}:
        raise AssertionError(f"round-two variant changed unexpected fields: {sorted(changed)}")
    return (
        Candidate("locked_run018", "locked_run018_control", config, LOCKED_RUN018_CANDIDATE_ID),
        Candidate(
            "short_scale_fast_lease",
            "short_scale_fast_lease_round2",
            partial,
            _candidate_id("short_scale_fast_lease", partial),
        ),
    )


def _fresh_live_cost() -> dict[str, Any]:
    return {
        "fresh_window_requests_per_probe": 2,
        "max_fresh_window_requests_per_probe": 2,
        "within_window_budget": True,
        "provider_component_count": 2,
        "provider_component_forwards_per_probe": 4,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "cache_hop_seconds": 0.2,
        "production_probe_interval_seconds": 0.75,
        "cache_grid_is_live_probe_cadence": False,
        "fresh_live_cadence_verified": False,
    }


def _candidate_lock(locked: LockedCandidate, candidates: Sequence[Candidate]) -> dict[str, Any]:
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
        "candidates": [
            {
                "name": candidate.name,
                "family": candidate.family,
                "candidate_id": candidate.candidate_id,
                "config": asdict(candidate.config),
            }
            for candidate in candidates
        ],
        "candidate_count": len(candidates),
        "search_space": "none",
        "only_allowed_config_delta": "enable_short_scale_fast_lease:false->true",
    }
    return {**payload, "lock_sha256": _sha256_value(payload)}


def _prepare_baseline(
    sources: Sequence[DatasetSource],
) -> tuple[dict[str, PreparedVideo], dict[str, Any]]:
    prepared: dict[str, PreparedVideo] = {}
    full_scores: dict[str, dict[str, Any]] = {}
    trace_hashes: dict[str, str] = {}
    source_for_video: dict[str, str] = {}
    for source in sources:
        dataset = Dataset(source.corpus_root.resolve(), source.input_root.resolve(), sweep.DEFAULT_PROVIDER)
        for video_id in source.video_ids:
            inputs = dataset.video_inputs(video_id)
            short = dataset.block(video_id, sweep.SHORT_WINDOW_SECONDS)
            long = dataset.block(video_id, sweep.LONG_WINDOW_SECONDS)
            decisions = replay_cached_live_windows_dual(
                short,
                long,
                inputs["profiles"],
                inputs["speech"],
                inputs["probes"],
                inputs["releases"],
                long_weight=sweep.LONG_WEIGHT,
                config=sweep.BASELINE_CONFIG,
            )
            steps = sweep._build_steps(short, long, inputs)
            if any(len(step.evidences) > 2 for step in steps):
                raise AssertionError("round-two evaluation exceeded two cached windows")
            prepared[video_id] = PreparedVideo(source.label, inputs, decisions, steps)
            full_scores[video_id] = score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            )
            trace_hashes[video_id] = _trace_hash(decisions)
            source_for_video[video_id] = source.label
    aggregate = aggregate_video_scores(full_scores.values())
    baseline = {
        "status": "baseline_complete_before_candidates",
        "algorithm_id": "production_dual_window_baseline",
        "provider": sweep.DEFAULT_PROVIDER,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "algorithm_config": asdict(sweep.BASELINE_CONFIG),
        "aggregate": aggregate,
        "per_video": {video_id: sweep._compact_score(score) for video_id, score in full_scores.items()},
        "trace_hashes": trace_hashes,
        "source_for_video": source_for_video,
        "fresh_live_cost": _fresh_live_cost(),
    }
    return prepared, baseline


def _score_candidate(
    prepared: dict[str, PreparedVideo], candidate: Candidate
) -> dict[str, Any]:
    started = time.monotonic()
    full_scores: dict[str, dict[str, Any]] = {}
    trace_hashes: dict[str, str] = {}
    for video_id, value in prepared.items():
        decisions = replay_hybrid_decisions(
            value.baseline_decisions,
            value.steps,
            value.inputs["profiles"],
            config=candidate.config,
        )
        full_scores[video_id] = score_live_speaker_decisions(
            decisions, value.inputs["canonical"], value.inputs["profiles"]
        )
        trace_hashes[video_id] = _trace_hash(decisions)
    return {
        "candidate_id": candidate.candidate_id,
        "name": candidate.name,
        "family": candidate.family,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(candidate.config),
        "aggregate": aggregate_video_scores(full_scores.values()),
        "per_video": {video_id: sweep._compact_score(score) for video_id, score in full_scores.items()},
        "trace_hashes": trace_hashes,
        "fresh_live_cost": _fresh_live_cost(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _comparison(candidate: dict[str, Any], reference: dict[str, Any], label: str) -> dict[str, Any]:
    if set(candidate["per_video"]) != set(reference["per_video"]):
        raise ValueError("candidate and reference video cohorts differ")
    score_deltas = {
        video_id: float(candidate["per_video"][video_id]["strict_browser_live_score"])
        - float(reference["per_video"][video_id]["strict_browser_live_score"])
        for video_id in candidate["per_video"]
    }
    wrong_deltas = {
        video_id: float(candidate["per_video"][video_id]["wrong_live_speech_ratio"])
        - float(reference["per_video"][video_id]["wrong_live_speech_ratio"])
        for video_id in candidate["per_video"]
    }
    aggregate_delta = float(candidate["aggregate"]["global_score"]) - float(
        reference["aggregate"]["global_score"]
    )
    return {
        "reference": label,
        "aggregate_score_delta": aggregate_delta,
        "per_video_score_delta": score_deltas,
        "per_video_wrong_ratio_delta": wrong_deltas,
        "aggregate_improvement_passed": aggregate_delta > 1e-12,
        "per_video_score_gate_passed": all(value >= -SCORE_TOLERANCE - 1e-12 for value in score_deltas.values()),
        "per_video_wrong_ratio_gate_passed": all(value <= WRONG_RATIO_TOLERANCE + 1e-12 for value in wrong_deltas.values()),
    }


def _comparison_passed(value: dict[str, Any]) -> bool:
    return bool(
        value["aggregate_improvement_passed"]
        and value["per_video_score_gate_passed"]
        and value["per_video_wrong_ratio_gate_passed"]
    )


def _promotion_decision(
    baseline: dict[str, Any], locked: dict[str, Any], partial: dict[str, Any]
) -> dict[str, Any]:
    locked_vs_baseline = _comparison(locked, baseline, "production_baseline")
    partial_vs_baseline = _comparison(partial, baseline, "production_baseline")
    partial_vs_locked = _comparison(partial, locked, "locked_run018")
    locked_eligible = _comparison_passed(locked_vs_baseline)
    partial_eligible = bool(
        _comparison_passed(partial_vs_baseline)
        and _comparison_passed(partial_vs_locked)
    )
    if partial_eligible:
        winner = "short_scale_fast_lease"
        reason = "partial fast-scale lease improves the locked control and passes every gate"
    elif locked_eligible:
        winner = "locked_run018"
        reason = "round-two variant failed A/B promotion; retain the simpler locked control"
    else:
        winner = None
        reason = "neither hybrid candidate passed the full-cohort baseline gates"
    return {
        "winner": winner,
        "reason": reason,
        "locked_eligible": locked_eligible,
        "partial_eligible": partial_eligible,
        "locked_vs_baseline": locked_vs_baseline,
        "partial_vs_baseline": partial_vs_baseline,
        "partial_vs_locked": partial_vs_locked,
        "score_tolerance": SCORE_TOLERANCE,
        "wrong_ratio_tolerance": WRONG_RATIO_TOLERANCE,
    }


def _run_metadata(
    sources: Sequence[DatasetSource], candidate_lock: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "initialized",
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "selection_policy": "fixed_two_candidate_ab_no_search",
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
        "sealed_v3_ids_rejected": sorted(FORBIDDEN_V3_IDS),
        "sealed_v3_opened": False,
        "provider": sweep.DEFAULT_PROVIDER,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "fresh_live_cost": _fresh_live_cost(),
    }


def _progress(phase: str, completed_steps: int) -> dict[str, Any]:
    total = len(EVALUATION_ORDER)
    return {
        "phase": phase,
        "completed_steps": completed_steps,
        "total_steps": total,
        "progress_percent": round(100.0 * completed_steps / total, 2),
        "evaluation_order": list(EVALUATION_ORDER),
        "baseline_first": True,
        "sealed_v3_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Validate every identifier before resolving or opening any dataset root.
    sources = _validate_sources(args.dataset_source)
    locked = _load_run018(args.locked_run_dir)
    candidates = _locked_candidates(locked.config)
    if len(candidates) != 2:
        raise AssertionError("immutable round-two lock must contain exactly two candidates")
    candidate_lock = _candidate_lock(locked, candidates)

    run_dir = args.run_dir.resolve()
    locked_dir = locked.run_dir.resolve()
    if run_dir == locked_dir or run_dir.is_relative_to(locked_dir):
        raise ValueError("round-two output must remain outside the immutable run018 directory")
    artifact_names = (
        "run.json",
        "baseline.json",
        "progress.json",
        "trials.jsonl",
        "report.json",
        "champion.json",
    )
    if any((run_dir / name).exists() for name in artifact_names):
        raise FileExistsError("round-two artifacts already exist; the formal run is not resumable")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    run = _run_metadata(sources, candidate_lock)
    _atomic_json(run_dir / "run.json", run)
    _atomic_json(run_dir / "progress.json", _progress("INITIALIZED", 0))

    prepared, baseline = _prepare_baseline(sources)
    baseline["completed_at_step"] = 1
    baseline["evaluation_order"] = list(EVALUATION_ORDER)
    _atomic_json(run_dir / "baseline.json", baseline)
    run["status"] = "baseline_complete"
    run["baseline_completed_before_candidate_evaluation"] = True
    _atomic_json(run_dir / "run.json", run)
    _atomic_json(run_dir / "progress.json", _progress("BASELINE_COMPLETE", 1))
    print("[33.33%] production baseline complete", flush=True)

    rows: list[dict[str, Any]] = []
    locked_row = _score_candidate(prepared, candidates[0])
    locked_row["evaluation_index"] = 1
    locked_row["vs_baseline"] = _comparison(locked_row, baseline, "production_baseline")
    locked_row["vs_locked"] = _comparison(locked_row, locked_row, "locked_run018")
    rows.append(locked_row)
    _atomic_jsonl(run_dir / "trials.jsonl", rows)
    _atomic_json(run_dir / "progress.json", _progress("LOCKED_RUN018_COMPLETE", 2))
    print("[66.67%] exact locked run018 candidate complete", flush=True)

    partial_row = _score_candidate(prepared, candidates[1])
    partial_row["evaluation_index"] = 2
    partial_row["vs_baseline"] = _comparison(partial_row, baseline, "production_baseline")
    partial_row["vs_locked"] = _comparison(partial_row, locked_row, "locked_run018")
    rows.append(partial_row)
    decision = _promotion_decision(baseline, locked_row, partial_row)
    locked_row["promotion_gates_passed"] = bool(decision["locked_eligible"])
    partial_row["promotion_gates_passed"] = bool(decision["partial_eligible"])
    _atomic_jsonl(run_dir / "trials.jsonl", rows)

    winner = next(
        (row for row in rows if row["name"] == decision["winner"]), None
    )
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
            "round2_cached_winner_requires_fresh_live_verification"
            if winner is not None
            else "no_round2_candidate_passed"
        ),
        "runner_id": RUNNER_ID,
        "candidate_lock_sha256": candidate_lock["lock_sha256"],
        "source_run018_candidate_id": LOCKED_RUN018_CANDIDATE_ID,
        "winner": winner,
        "promotion": decision,
        "production_defaults_changed": False,
        "fresh_live_verification_required": winner is not None,
        "sealed_v3_opened": False,
    }
    _atomic_json(run_dir / "report.json", report)
    _atomic_json(run_dir / "champion.json", champion)
    run["status"] = "complete"
    run["winner"] = None if winner is None else winner["name"]
    run["elapsed_seconds"] = round(time.monotonic() - started, 6)
    _atomic_json(run_dir / "run.json", run)
    _atomic_json(run_dir / "progress.json", _progress("COMPLETE", 3))
    print(f"[100.00%] complete; winner={run['winner']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
