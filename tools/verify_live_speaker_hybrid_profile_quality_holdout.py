"""Open the three-video profile-quality holdout exactly once.

The verifier has no search knobs.  It accepts only the fixed profile-quality
winner produced by ``sweep_live_speaker_hybrid_profile_quality.py``, the exact
0.8/2.8 second production cache and provider stack, and the three IDs frozen in
``runtime/optimization/live_speaker_round2_holdout_lock.json``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from statistics import mean
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid as sweep
import sweep_live_speaker_hybrid_profile_quality as profile_quality
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_benchmark import aggregate_video_scores
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


VERIFIER_ID = "one_shot_profile_quality_holdout_v1"
DEFAULT_HOLDOUT_LOCK = (
    ROOT / "runtime" / "optimization" / "live_speaker_round2_holdout_lock.json"
)
HOLDOUT_ARTIFACT_NAME = "holdout.json"
EXPECTED_HOLDOUT_ID_SET = frozenset(round2.FORBIDDEN_V3_IDS)
EXPECTED_WINDOWS = (0.8, 2.8)
EXPECTED_LONG_WEIGHT = 0.25
EXPECTED_DEVELOPMENT_VIDEO_COUNT = 9
EXPECTED_HOLDOUT_VIDEO_COUNT = 3
SCORE_TOLERANCE = 0.005
WRONG_RATIO_TOLERANCE = 0.005
INITIAL_CHAMPION_STATUS = "profile_quality_cached_winner_requires_fresh_live_verification"
PASSED_CHAMPION_STATUS = "profile_quality_cached_holdout_passed"
FAILED_CHAMPION_STATUS = "profile_quality_cached_holdout_failed"


@dataclass(frozen=True)
class HoldoutDatasetSource:
    label: str
    corpus_root: Path
    input_root: Path
    video_ids: tuple[str, ...]


@dataclass(frozen=True)
class LockedProfileQualityCandidate:
    run_dir: Path
    run_path: Path
    champion_path: Path
    run: dict[str, Any]
    champion: dict[str, Any]
    winner: dict[str, Any]
    candidate_id: str
    config: HybridSpeakerTrackerConfig
    candidate_lock_sha256: str
    run_sha256: str
    champion_sha256: str


@dataclass(frozen=True)
class HoldoutLock:
    path: Path
    video_ids: tuple[str, ...]
    selection: Any
    sha256: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        action="append",
        required=True,
        metavar="LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2",
    )
    parser.add_argument("--profile-quality-run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_holdout_dataset_source(raw: str) -> HoldoutDatasetSource:
    """Parse source text without touching any referenced path."""

    try:
        label_and_corpus, input_root, raw_ids = raw.split(round2.SOURCE_SEPARATOR, 2)
        label, corpus_root = label_and_corpus.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            "holdout dataset source must be LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2"
        ) from exc
    label = label.strip()
    corpus_root = corpus_root.strip()
    input_root = input_root.strip()
    video_ids = tuple(value.strip() for value in raw_ids.split(",") if value.strip())
    if not label or not corpus_root or not input_root or not video_ids:
        raise ValueError("holdout source label, roots, and video IDs must be non-empty")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(f"holdout source {label!r} contains duplicate video IDs")
    return HoldoutDatasetSource(label, Path(corpus_root), Path(input_root), video_ids)


def _validate_holdout_sources(raw_sources: Sequence[str]) -> list[HoldoutDatasetSource]:
    """Validate the sealed identifiers before any path is opened."""

    if not raw_sources:
        raise ValueError("at least one --dataset-source is required")
    sources = [_parse_holdout_dataset_source(raw) for raw in raw_sources]
    labels = [source.label for source in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("holdout dataset source labels must be unique")
    video_ids = [video_id for source in sources for video_id in source.video_ids]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("holdout video IDs may occur in only one dataset source")
    if len(video_ids) != EXPECTED_HOLDOUT_VIDEO_COUNT:
        raise ValueError("holdout must contain exactly three video IDs")
    if frozenset(video_ids) != EXPECTED_HOLDOUT_ID_SET:
        raise ValueError("holdout IDs differ from the frozen three-video seal")
    return sources


def _comparison_has_nine_passed_videos(value: Any, expected_ids: set[str]) -> bool:
    if not isinstance(value, dict):
        return False
    score_deltas = value.get("per_video_score_delta")
    wrong_deltas = value.get("per_video_wrong_ratio_delta")
    if not isinstance(score_deltas, dict) or not isinstance(wrong_deltas, dict):
        return False
    if set(score_deltas) != expected_ids or set(wrong_deltas) != expected_ids:
        return False
    return bool(
        float(value.get("aggregate_score_delta", 0.0)) > 1e-12
        and bool(value.get("aggregate_improvement_passed"))
        and bool(value.get("per_video_score_gate_passed"))
        and bool(value.get("per_video_wrong_ratio_gate_passed"))
        and all(float(delta) >= -SCORE_TOLERANCE - 1e-12 for delta in score_deltas.values())
        and all(float(delta) <= WRONG_RATIO_TOLERANCE + 1e-12 for delta in wrong_deltas.values())
    )


def _candidate_lock_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    embedded = str(payload.pop("lock_sha256", ""))
    calculated = hashlib.sha256(
        profile_quality._stable_json(payload).encode("utf-8")
    ).hexdigest()
    if not embedded or embedded != calculated:
        raise ValueError("profile-quality candidate lock hash mismatch")
    return calculated


def _validate_candidate_lock(
    value: Any,
    winner: dict[str, Any],
) -> tuple[str, HybridSpeakerTrackerConfig]:
    if not isinstance(value, dict):
        raise ValueError("profile-quality run has no candidate lock")
    lock_hash = _candidate_lock_hash(value)
    if str(value.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-quality candidate lock uses another algorithm")
    if str(value.get("provider", "")) != sweep.DEFAULT_PROVIDER:
        raise ValueError("profile-quality candidate lock uses another provider")
    if [float(item) for item in value.get("windows_seconds", [])] != list(EXPECTED_WINDOWS):
        raise ValueError("profile-quality candidate lock must use 0.8/2.8 second windows")
    if abs(float(value.get("long_weight", -1.0)) - EXPECTED_LONG_WEIGHT) > 1e-12:
        raise ValueError("profile-quality candidate lock has another long-window weight")
    if int(value.get("candidate_count", 0)) != 2 or value.get("search_space") != "none":
        raise ValueError("profile-quality candidate lock is not a fixed two-row A/B")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("profile-quality candidate lock must contain exactly two candidates")
    by_name = {
        str(item.get("name", "")): item for item in candidates if isinstance(item, dict)
    }
    if set(by_name) != {"locked_run018", profile_quality.PROFILE_QUALITY_NAME}:
        raise ValueError("profile-quality candidate lock contains unexpected candidates")
    control_raw = by_name["locked_run018"]
    candidate_raw = by_name[profile_quality.PROFILE_QUALITY_NAME]
    control_config = HybridSpeakerTrackerConfig(**dict(control_raw.get("config") or {}))
    expected_control, expected_candidate = profile_quality._fixed_candidates(control_config)
    expected_rows = {
        expected_control.name: expected_control,
        expected_candidate.name: expected_candidate,
    }
    for name, raw in by_name.items():
        expected = expected_rows[name]
        if str(raw.get("family", "")) != expected.family:
            raise ValueError(f"candidate-lock family mismatch for {name}")
        if str(raw.get("candidate_id", "")) != expected.candidate_id:
            raise ValueError(f"candidate-lock identity mismatch for {name}")
        if dict(raw.get("config") or {}) != asdict(expected.config):
            raise ValueError(f"candidate-lock config mismatch for {name}")
    if str(winner.get("candidate_id", "")) != expected_candidate.candidate_id:
        raise ValueError("winner identity differs from the fixed profile-quality candidate")
    if str(winner.get("name", "")) != expected_candidate.name:
        raise ValueError("winner name differs from the fixed profile-quality candidate")
    if str(winner.get("family", "")) != expected_candidate.family:
        raise ValueError("winner family differs from the fixed profile-quality candidate")
    if dict(winner.get("config") or {}) != asdict(expected_candidate.config):
        raise ValueError("winner config differs from the fixed profile-quality candidate")
    return lock_hash, expected_candidate.config


def _load_locked_profile_quality_candidate(
    run_dir: Path,
) -> LockedProfileQualityCandidate:
    """Open only run/champion and prove that the nine-video choice is frozen."""

    run_path = run_dir / "run.json"
    champion_path = run_dir / "champion.json"
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    if str(run.get("runner_id", "")) != profile_quality.RUNNER_ID:
        raise ValueError("run was not produced by the profile-quality runner")
    if str(champion.get("runner_id", "")) != profile_quality.RUNNER_ID:
        raise ValueError("champion was not produced by the profile-quality runner")
    if str(run.get("status", "")) != "complete":
        raise ValueError("profile-quality run is not complete")
    if str(run.get("selection_policy", "")) != "fixed_profile_quality_candidate_no_search":
        raise ValueError("profile-quality run permits a search space")
    if str(run.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-quality run uses another algorithm")
    if str(run.get("provider", "")) != sweep.DEFAULT_PROVIDER:
        raise ValueError("profile-quality run uses another provider")
    if [float(item) for item in run.get("windows_seconds", [])] != list(EXPECTED_WINDOWS):
        raise ValueError("profile-quality run does not use exactly 0.8/2.8 seconds")
    if abs(float(run.get("long_weight", -1.0)) - EXPECTED_LONG_WEIGHT) > 1e-12:
        raise ValueError("profile-quality run has another long-window weight")
    fresh_cost = run.get("fresh_live_cost")
    if not isinstance(fresh_cost, dict):
        raise ValueError("profile-quality run has no realtime cost contract")
    if int(fresh_cost.get("fresh_window_requests_per_probe", 99)) != 2:
        raise ValueError("profile-quality run must request exactly two fresh windows")
    if int(fresh_cost.get("max_fresh_window_requests_per_probe", 99)) != 2:
        raise ValueError("profile-quality run must cap fresh windows at two")
    if bool(run.get("sealed_v3_opened")) or bool(champion.get("sealed_v3_opened")):
        raise RuntimeError("profile-quality candidate already opened the sealed holdout")
    winner = champion.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("profile-quality champion has no locked winner")
    if str(run.get("winner", "")) != profile_quality.PROFILE_QUALITY_NAME:
        raise ValueError("run does not lock the profile-quality winner")
    if str(winner.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-quality winner uses another algorithm")
    if not bool(winner.get("promotion_gates_passed")):
        raise ValueError("profile-quality winner did not pass mandatory gates")
    lock_hash, config = _validate_candidate_lock(run.get("candidate_lock"), winner)
    if str(champion.get("candidate_lock_sha256", "")) != lock_hash:
        raise ValueError("champion points to another candidate lock")
    promotion = champion.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError("profile-quality champion has no promotion proof")
    if not all(bool(promotion.get(key)) for key in (
        "candidate_eligible", "baseline_gates_passed", "run018_gates_passed"
    )):
        raise ValueError("profile-quality candidate did not pass both mandatory comparisons")
    per_video = winner.get("per_video")
    if not isinstance(per_video, dict) or len(per_video) != EXPECTED_DEVELOPMENT_VIDEO_COUNT:
        raise ValueError("profile-quality winner was not scored on exactly nine videos")
    development_ids = set(str(value) for value in per_video)
    if development_ids.intersection(EXPECTED_HOLDOUT_ID_SET):
        raise ValueError("profile-quality selection data contains sealed holdout IDs")
    run_ids = [
        str(video_id)
        for source in run.get("dataset_sources", [])
        if isinstance(source, dict)
        for video_id in source.get("video_ids", [])
    ]
    if len(run_ids) != EXPECTED_DEVELOPMENT_VIDEO_COUNT or set(run_ids) != development_ids:
        raise ValueError("profile-quality run does not describe the same nine selection videos")
    for key in ("candidate_vs_baseline", "candidate_vs_locked"):
        proof = promotion.get(key)
        if not _comparison_has_nine_passed_videos(proof, development_ids):
            raise ValueError(f"profile-quality nine-video proof failed: {key}")
        if winner.get("vs_baseline" if key.endswith("baseline") else "vs_locked") != proof:
            raise ValueError(f"winner and champion promotion proof differ: {key}")
    allowed_statuses = {
        INITIAL_CHAMPION_STATUS,
        PASSED_CHAMPION_STATUS,
        FAILED_CHAMPION_STATUS,
    }
    if str(champion.get("status", "")) not in allowed_statuses:
        raise ValueError("profile-quality champion has an unexpected status")
    return LockedProfileQualityCandidate(
        run_dir=run_dir,
        run_path=run_path,
        champion_path=champion_path,
        run=run,
        champion=champion,
        winner=winner,
        candidate_id=str(winner["candidate_id"]),
        config=config,
        candidate_lock_sha256=lock_hash,
        run_sha256=_sha256_file(run_path),
        champion_sha256=_sha256_file(champion_path),
    )


def _assert_one_shot(candidate: LockedProfileQualityCandidate) -> Path:
    artifact = candidate.run_dir / HOLDOUT_ARTIFACT_NAME
    champion = candidate.champion
    if artifact.exists():
        raise RuntimeError("profile-quality holdout result already exists; rerun refused")
    if bool(champion.get("holdout_opened")) or bool(champion.get("sealed_v3_opened")):
        raise RuntimeError("profile-quality holdout is already marked as opened")
    if str(champion.get("status", "")) != INITIAL_CHAMPION_STATUS:
        raise RuntimeError("profile-quality champion is no longer awaiting its one-shot holdout")
    return artifact


def _load_holdout_lock(path: Path, requested_ids: Sequence[str]) -> HoldoutLock:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("holdout lock must be a JSON object")
    video_ids = value.get("video_ids")
    if not isinstance(video_ids, list) or any(not isinstance(item, str) for item in video_ids):
        raise ValueError("holdout lock must contain a video_ids array")
    normalized = tuple(item.strip() for item in video_ids)
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("holdout lock video IDs must be unique and non-empty")
    if len(normalized) != EXPECTED_HOLDOUT_VIDEO_COUNT:
        raise ValueError("holdout lock must contain exactly three videos")
    if frozenset(normalized) != EXPECTED_HOLDOUT_ID_SET:
        raise ValueError("holdout lock differs from the frozen three-video seal")
    if tuple(requested_ids) != normalized:
        raise ValueError("dataset-source video order differs from the frozen holdout lock")
    if "selection" not in value or value["selection"] is None:
        raise ValueError("holdout lock has no selection record")
    return HoldoutLock(
        path=path,
        video_ids=normalized,
        selection=value["selection"],
        sha256=_sha256_bytes(raw),
    )


def _aggregate_wrong_ratio(per_video: dict[str, dict[str, Any]]) -> float:
    return float(mean(float(value["wrong_live_speech_ratio"]) for value in per_video.values()))


def _holdout_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    video_ids: Sequence[str],
) -> dict[str, Any]:
    baseline_per_video = baseline.get("per_video")
    candidate_per_video = candidate.get("per_video")
    expected = set(video_ids)
    if not isinstance(baseline_per_video, dict) or set(baseline_per_video) != expected:
        raise ValueError("baseline holdout scores do not cover exactly three videos")
    if not isinstance(candidate_per_video, dict) or set(candidate_per_video) != expected:
        raise ValueError("candidate holdout scores do not cover exactly three videos")
    per_video: dict[str, Any] = {}
    for video_id in video_ids:
        base = baseline_per_video[video_id]
        contender = candidate_per_video[video_id]
        score_delta = float(contender["strict_browser_live_score"]) - float(
            base["strict_browser_live_score"]
        )
        wrong_delta = float(contender["wrong_live_speech_ratio"]) - float(
            base["wrong_live_speech_ratio"]
        )
        score_gate = score_delta >= -SCORE_TOLERANCE - 1e-12
        wrong_gate = wrong_delta <= WRONG_RATIO_TOLERANCE + 1e-12
        per_video[video_id] = {
            "score_delta_vs_baseline": score_delta,
            "wrong_ratio_delta_vs_baseline": wrong_delta,
            "score_gate_passed": score_gate,
            "wrong_ratio_gate_passed": wrong_gate,
            "passed": bool(score_gate and wrong_gate),
        }
    baseline_global = float(baseline["aggregate"]["global_score"])
    candidate_global = float(candidate["aggregate"]["global_score"])
    aggregate_score_delta = candidate_global - baseline_global
    baseline_wrong = _aggregate_wrong_ratio(baseline_per_video)
    candidate_wrong = _aggregate_wrong_ratio(candidate_per_video)
    aggregate_wrong_delta = candidate_wrong - baseline_wrong
    aggregate = {
        "score_delta_vs_baseline": aggregate_score_delta,
        "wrong_ratio_delta_vs_baseline": aggregate_wrong_delta,
        "score_gate_passed": aggregate_score_delta > 1e-12,
        "wrong_ratio_gate_passed": aggregate_wrong_delta <= WRONG_RATIO_TOLERANCE + 1e-12,
    }
    aggregate["passed"] = bool(
        aggregate["score_gate_passed"] and aggregate["wrong_ratio_gate_passed"]
    )
    passed = bool(aggregate["passed"] and all(value["passed"] for value in per_video.values()))
    return {
        "fixed_tolerances": {
            "per_video_score": SCORE_TOLERANCE,
            "per_video_wrong_ratio": WRONG_RATIO_TOLERANCE,
            "aggregate_score_requires_improvement": True,
            "aggregate_wrong_ratio": WRONG_RATIO_TOLERANCE,
        },
        "per_video": per_video,
        "aggregate": aggregate,
        "all_three_individual_gates_passed": all(
            value["passed"] for value in per_video.values()
        ),
        "holdout_passed": passed,
    }


def _evaluate_holdout(
    sources: Sequence[HoldoutDatasetSource],
    candidate: LockedProfileQualityCandidate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # _prepare_baseline completes the baseline for every source/video before
    # this function invokes the single locked candidate.
    prepared, baseline = round2._prepare_baseline(sources)
    contender = round2.Candidate(
        profile_quality.PROFILE_QUALITY_NAME,
        profile_quality.PROFILE_QUALITY_FAMILY,
        candidate.config,
        candidate.candidate_id,
    )
    scored = round2._score_candidate(prepared, contender)
    return baseline, scored


def _holdout_artifact(
    candidate: LockedProfileQualityCandidate,
    lock: HoldoutLock,
    baseline: dict[str, Any],
    scored: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "status": "HOLDOUT_PASSED" if gates["holdout_passed"] else "HOLDOUT_FAILED",
        "selection_locked_before_holdout": True,
        "parameter_search_performed": False,
        "post_holdout_tuning_allowed": False,
        "evaluation_order": ["baseline_all_three", "single_locked_candidate_all_three"],
        "baseline_completed_before_candidate_evaluation": True,
        "video_ids": list(lock.video_ids),
        "holdout_lock": {
            "path": str(lock.path),
            "sha256": lock.sha256,
            "selection": lock.selection,
        },
        "source": {
            "runner_id": profile_quality.RUNNER_ID,
            "run_path": str(candidate.run_path),
            "run_sha256": candidate.run_sha256,
            "champion_path": str(candidate.champion_path),
            "champion_sha256_before_holdout": candidate.champion_sha256,
            "candidate_lock_sha256": candidate.candidate_lock_sha256,
        },
        "candidate": {
            "name": profile_quality.PROFILE_QUALITY_NAME,
            "family": profile_quality.PROFILE_QUALITY_FAMILY,
            "candidate_id": candidate.candidate_id,
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "config": asdict(candidate.config),
        },
        "provider": sweep.DEFAULT_PROVIDER,
        "windows_seconds": list(EXPECTED_WINDOWS),
        "long_weight": EXPECTED_LONG_WEIGHT,
        "fresh_live_cost": round2._fresh_live_cost(),
        "baseline": baseline,
        "candidate_scores": scored,
        "gates": gates,
        "passed": bool(gates["holdout_passed"]),
        "production_wired": False,
        "fresh_live_verified": False,
    }


def _write_json_temporary(path: Path, value: Any) -> Path:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _commit_holdout(
    artifact_path: Path,
    champion_path: Path,
    artifact: dict[str, Any],
    original_champion: dict[str, Any],
) -> dict[str, Any]:
    if artifact_path.exists():
        raise RuntimeError("profile-quality holdout result already exists; rerun refused")
    updated = deepcopy(original_champion)
    passed = bool(artifact["passed"])
    updated.update({
        "status": PASSED_CHAMPION_STATUS if passed else FAILED_CHAMPION_STATUS,
        "holdout_opened": True,
        "sealed_v3_opened": True,
        "holdout_passed": passed,
        "holdout_artifact": artifact_path.name,
        "holdout_verifier_id": VERIFIER_ID,
        "holdout_lock_sha256": artifact["holdout_lock"]["sha256"],
        "fresh_live_verification_required": passed,
        "production_defaults_changed": False,
    })
    artifact_tmp = _write_json_temporary(artifact_path, artifact)
    champion_tmp = _write_json_temporary(champion_path, updated)
    try:
        os.replace(artifact_tmp, artifact_path)
        os.replace(champion_tmp, champion_path)
    finally:
        for temporary in (artifact_tmp, champion_tmp):
            if temporary.exists():
                temporary.unlink()
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Pure string validation is deliberately first.  Invalid IDs cannot cause
    # candidate, lock, corpus, input, or output paths to be opened.
    sources = _validate_holdout_sources(args.dataset_source)
    requested_ids = tuple(
        video_id for source in sources for video_id in source.video_ids
    )

    # Candidate validation is the first filesystem operation.  It proves that
    # selection was frozen and passed every gate on exactly nine opened videos.
    candidate = _load_locked_profile_quality_candidate(args.profile_quality_run_dir)
    artifact_path = _assert_one_shot(candidate)

    # Only after the candidate is valid and one-shot state is clean may the
    # holdout lock and, later, the dataset roots be opened.
    lock = _load_holdout_lock(DEFAULT_HOLDOUT_LOCK, requested_ids)
    baseline, scored = _evaluate_holdout(sources, candidate)
    gates = _holdout_gates(baseline, scored, lock.video_ids)
    artifact = _holdout_artifact(candidate, lock, baseline, scored, gates)
    _commit_holdout(
        artifact_path, candidate.champion_path, artifact, candidate.champion
    )
    print(json.dumps({
        "status": artifact["status"],
        "candidate_id": candidate.candidate_id,
        "video_ids": list(lock.video_ids),
        "baseline_global_score": baseline["aggregate"]["global_score"],
        "candidate_global_score": scored["aggregate"]["global_score"],
        "aggregate_gates": gates["aggregate"],
        "per_video_gates": gates["per_video"],
        "holdout_lock_sha256": lock.sha256,
        "artifact": str(artifact_path),
    }, indent=2, ensure_ascii=False))
    return 0 if gates["holdout_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
