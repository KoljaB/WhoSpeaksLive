"""Open the sealed round-three holdout once for the fixed profile-meta winner.

This verifier exposes no search or tuning controls.  It accepts only the
12-video-gate-passed ``profile_quality_meta_a005_s035`` candidate, evaluates
the production baseline for all three frozen videos before evaluating that one
candidate, and commits the holdout result and champion state as one staged
one-shot transaction.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid as sweep
import sweep_live_speaker_hybrid_profile_quality_meta as meta
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


VERIFIER_ID = "one_shot_profile_quality_meta_round3_holdout_v1"
EXPECTED_RUNNER_ID = "locked_live_speaker_hybrid_profile_quality_meta_ab_v1"
EXPECTED_CANDIDATE_NAME = "profile_quality_meta_a005_s035"
EXPECTED_CANDIDATE_FAMILY = "profile_quality_meta_lease_a005_s035_v1"
EXPECTED_POLICY_ID = "causal_profile_quality_meta_lease_a005_s035_v1"
EXPECTED_SELECTION_POLICY = "fixed_profile_quality_meta_a005_s035_no_search"
DEFAULT_HOLDOUT_LOCK = (
    ROOT / "runtime" / "optimization" / "live_speaker_round3_holdout_lock.json"
)
EXPECTED_HOLDOUT_LOCK_SHA256 = (
    "d2e0331c0d9c77c4f13c1b1b37083874a3f4bbbae74ec246903effe09ab99bfa"
)
HOLDOUT_ARTIFACT_NAME = "holdout.json"
EXPECTED_HOLDOUT_IDS = ("blcKeLDDzSM", "KdOXM3I_5hk", "acbnyagl8jo")
EXPECTED_DEVELOPMENT_IDS = (
    "Dd7FixvoKBw",
    "DsyfYJ5Ou3g",
    "20v1OxUXcQY",
    "JWS-qfR6K3w",
    "e3h6es6zh1c",
    "1NBVQB-Srpw",
    "F2-2RBi1qzY",
    "vIfGgDnmBXg",
    "ZY0DG8rUnCA",
    "pD4IdQTmneI",
    "k1tsGGz-Qw0",
    "aHGd6LqAVzw",
)
EXPECTED_WINDOWS = (0.8, 2.8)
EXPECTED_LONG_WEIGHT = 0.25
EXPECTED_PROVIDER = sweep.DEFAULT_PROVIDER
EXPECTED_DEVELOPMENT_VIDEO_COUNT = 12
EXPECTED_HOLDOUT_VIDEO_COUNT = 3
MAX_FRESH_WINDOWS_PER_PROBE = 2
SCORE_TOLERANCE = 0.005
WRONG_RATIO_TOLERANCE = 0.005
INITIAL_CHAMPION_STATUS = "profile_quality_meta_cached_winner_requires_fresh_live_verification"
PASSED_CHAMPION_STATUS = "profile_quality_meta_cached_holdout_passed"
FAILED_CHAMPION_STATUS = "profile_quality_meta_cached_holdout_failed"


@dataclass(frozen=True)
class HoldoutDatasetSource:
    label: str
    corpus_root: Path
    input_root: Path
    video_ids: tuple[str, ...]


@dataclass(frozen=True)
class LockedMetaCandidate:
    run_dir: Path
    run_path: Path
    champion_path: Path
    run: dict[str, Any]
    champion: dict[str, Any]
    winner: dict[str, Any]
    control_candidate_id: str
    control_config: HybridSpeakerTrackerConfig
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
        metavar="LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2,ID3",
    )
    parser.add_argument("--meta-run-dir", type=Path, required=True)
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
    """Parse source text without resolving or opening either root."""

    try:
        label_and_corpus, input_root, raw_ids = raw.split(round2.SOURCE_SEPARATOR, 2)
        label, corpus_root = label_and_corpus.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            "holdout dataset source must be LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2,ID3"
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
    """Reject any identifier/order mismatch before a referenced path is touched."""

    if not raw_sources:
        raise ValueError("at least one --dataset-source is required")
    sources = [_parse_holdout_dataset_source(raw) for raw in raw_sources]
    labels = [source.label for source in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("holdout dataset source labels must be unique")
    requested = tuple(video_id for source in sources for video_id in source.video_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("holdout video IDs may occur in only one dataset source")
    if len(requested) != EXPECTED_HOLDOUT_VIDEO_COUNT:
        raise ValueError("holdout must contain exactly three video IDs")
    if requested != EXPECTED_HOLDOUT_IDS:
        raise ValueError("holdout IDs or order differ from the frozen round-three seal")
    return sources


def _comparison_has_twelve_passed_videos(
    value: Any, expected_ids: set[str]
) -> bool:
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
        and all(
            float(delta) >= -SCORE_TOLERANCE - 1e-12
            for delta in score_deltas.values()
        )
        and all(
            float(delta) <= WRONG_RATIO_TOLERANCE + 1e-12
            for delta in wrong_deltas.values()
        )
    )


def _candidate_lock_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    embedded = str(payload.pop("lock_sha256", ""))
    calculated = hashlib.sha256(
        meta._stable_json(payload).encode("utf-8")
    ).hexdigest()
    if not embedded or embedded != calculated:
        raise ValueError("profile-meta candidate lock hash mismatch")
    return calculated


def _assert_exact_meta_config(config: HybridSpeakerTrackerConfig) -> None:
    expected = {
        "enable_short_scale_fast_lease": False,
        "enable_profile_quality_short_scale_fast_lease": False,
        "profile_quality_fast_lease_min_sentence_count": 2,
        "profile_quality_fast_lease_min_speech_seconds": 3.1,
        "profile_quality_fast_lease_min_similarity": 0.18,
        "profile_quality_fast_lease_min_margin": 0.06,
        "enable_profile_quality_meta_lease": True,
        "profile_quality_meta_fresh_min_age_seconds": 0.05,
        "profile_quality_meta_fresh_max_age_seconds": 0.8,
        "profile_quality_meta_fresh_min_speech_seconds": 3.8,
        "profile_quality_meta_fresh_min_short_margin": 0.30,
        "profile_quality_meta_fresh_min_long_margin": 0.70,
        "profile_quality_meta_independent_max_profile_count": 8,
        "profile_quality_meta_switch_min_short_margin": 0.35,
    }
    payload = asdict(config)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"fixed profile-meta config mismatch for {key}")


def _fixed_meta_candidates(
    control_config: HybridSpeakerTrackerConfig,
) -> tuple[Any, Any]:
    candidates = tuple(meta._fixed_candidates(control_config))
    if len(candidates) != 2:
        raise ValueError("formal profile-meta run must contain control plus one candidate")
    control, contender = candidates
    if control.name != "locked_run018":
        raise ValueError("formal profile-meta control is not locked run018")
    if contender.name != EXPECTED_CANDIDATE_NAME:
        raise ValueError("formal profile-meta candidate has another name")
    if contender.family != EXPECTED_CANDIDATE_FAMILY:
        raise ValueError("formal profile-meta candidate has another family")
    _assert_exact_meta_config(contender.config)
    return control, contender


def _validate_candidate_lock(
    value: Any, winner: dict[str, Any]
) -> tuple[str, HybridSpeakerTrackerConfig, HybridSpeakerTrackerConfig]:
    if not isinstance(value, dict):
        raise ValueError("profile-meta run has no candidate lock")
    lock_hash = _candidate_lock_hash(value)
    if str(value.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-meta candidate lock uses another algorithm")
    if str(value.get("provider", "")) != EXPECTED_PROVIDER:
        raise ValueError("profile-meta candidate lock uses another provider")
    if tuple(float(item) for item in value.get("windows_seconds", [])) != EXPECTED_WINDOWS:
        raise ValueError("profile-meta candidate lock must use exactly 0.8/2.8 seconds")
    if abs(float(value.get("long_weight", -1.0)) - EXPECTED_LONG_WEIGHT) > 1e-12:
        raise ValueError("profile-meta candidate lock has another long-window weight")
    if int(value.get("candidate_count", 0)) != 2 or value.get("search_space") != "none":
        raise ValueError("profile-meta candidate lock is not a fixed control/candidate A/B")
    contract = value.get("fixed_meta_contract")
    if not isinstance(contract, dict):
        raise ValueError("profile-meta candidate lock has no fixed policy contract")
    if contract.get("policy_id") != EXPECTED_POLICY_ID:
        raise ValueError("profile-meta candidate lock uses another policy")
    if contract.get("candidate_name") != EXPECTED_CANDIDATE_NAME:
        raise ValueError("profile-meta candidate lock names another candidate")
    if float(contract.get("fresh_min_age_seconds", -1.0)) != 0.05:
        raise ValueError("profile-meta candidate lock has another minimum age")
    if float(contract.get("switch_min_short_margin", -1.0)) != 0.35:
        raise ValueError("profile-meta candidate lock has another switch margin")
    if bool(contract.get("old_profile_quality_output_path_enabled", True)):
        raise ValueError("profile-meta candidate lock enables the old output path")
    if int(contract.get("maximum_fresh_embedding_windows_per_probe", 99)) != 2:
        raise ValueError("profile-meta candidate lock permits more than two windows")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("profile-meta candidate lock must contain exactly two rows")
    by_name = {str(row.get("name", "")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != {"locked_run018", EXPECTED_CANDIDATE_NAME}:
        raise ValueError("profile-meta candidate lock contains unexpected rows")
    control_config = HybridSpeakerTrackerConfig(
        **dict(by_name["locked_run018"].get("config") or {})
    )
    expected_control, expected_candidate = _fixed_meta_candidates(control_config)
    source_config = value.get("source_config")
    if not isinstance(source_config, dict) or source_config != asdict(expected_control.config):
        raise ValueError("profile-meta source config differs from locked run018")
    if value.get("source_config_sha256") != meta._sha256_value(source_config):
        raise ValueError("profile-meta source config hash mismatch")
    if value.get("source_candidate_id") != round2.LOCKED_RUN018_CANDIDATE_ID:
        raise ValueError("profile-meta candidate lock points to another source candidate")
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
    if str(winner.get("name", "")) != expected_candidate.name:
        raise ValueError("winner name differs from the fixed profile-meta candidate")
    if str(winner.get("family", "")) != expected_candidate.family:
        raise ValueError("winner family differs from the fixed profile-meta candidate")
    if str(winner.get("candidate_id", "")) != expected_candidate.candidate_id:
        raise ValueError("winner identity differs from the fixed profile-meta candidate")
    if dict(winner.get("config") or {}) != asdict(expected_candidate.config):
        raise ValueError("winner config differs from the fixed profile-meta candidate")
    return lock_hash, expected_control.config, expected_candidate.config


def _validate_realtime_cost(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("profile-meta run has no realtime cost contract")
    exact = int(value.get("fresh_window_requests_per_probe", 99))
    maximum = int(value.get("max_fresh_window_requests_per_probe", 99))
    if exact != MAX_FRESH_WINDOWS_PER_PROBE:
        raise ValueError("profile-meta run must request exactly two fresh windows")
    if maximum != MAX_FRESH_WINDOWS_PER_PROBE:
        raise ValueError("profile-meta run must cap fresh windows at two")
    if tuple(float(item) for item in value.get("windows_seconds", [])) != EXPECTED_WINDOWS:
        raise ValueError("profile-meta realtime cost uses another window pair")
    if abs(float(value.get("long_weight", -1.0)) - EXPECTED_LONG_WEIGHT) > 1e-12:
        raise ValueError("profile-meta realtime cost uses another long-window weight")


def _validate_dataset_source_lock(value: Any) -> str:
    if not isinstance(value, dict):
        raise ValueError("profile-meta run has no dataset source lock")
    payload = dict(value)
    embedded = str(payload.pop("lock_sha256", ""))
    calculated = meta._sha256_value(payload)
    if not embedded or embedded != calculated:
        raise ValueError("profile-meta dataset source lock hash mismatch")
    if tuple(value.get("source_order") or ()) != tuple(meta.EXPECTED_SOURCE_ORDER):
        raise ValueError("profile-meta dataset source lock has another source order")
    if tuple(value.get("video_ids") or ()) != EXPECTED_DEVELOPMENT_IDS:
        raise ValueError("profile-meta dataset source lock has another 12-video cohort")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("profile-meta dataset source lock must contain v1/v2/v3")
    by_label = {
        str(source.get("label", "")): tuple(source.get("video_ids") or ())
        for source in sources
        if isinstance(source, dict)
    }
    if by_label != meta.EXPECTED_SOURCE_VIDEO_IDS:
        raise ValueError("profile-meta dataset source mapping differs from the fixed cohort")
    return calculated


def _load_locked_meta_candidate(run_dir: Path) -> LockedMetaCandidate:
    """Open only formal run/champion artifacts and prove the choice was frozen."""

    run_path = run_dir / "run.json"
    champion_path = run_dir / "champion.json"
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    if str(meta.RUNNER_ID) != EXPECTED_RUNNER_ID:
        raise ValueError("imported profile-meta runner identity is unexpected")
    if meta.META_NAME != EXPECTED_CANDIDATE_NAME:
        raise ValueError("imported profile-meta candidate name is unexpected")
    if meta.META_FAMILY != EXPECTED_CANDIDATE_FAMILY:
        raise ValueError("imported profile-meta candidate family is unexpected")
    if meta.META_POLICY_ID != EXPECTED_POLICY_ID:
        raise ValueError("imported profile-meta policy identity is unexpected")
    if meta.EXPECTED_PROVIDER != EXPECTED_PROVIDER or sweep.DEFAULT_PROVIDER != EXPECTED_PROVIDER:
        raise ValueError("imported profile-meta provider contract is unexpected")
    if str(run.get("runner_id", "")) != EXPECTED_RUNNER_ID:
        raise ValueError("run was not produced by the fixed profile-meta runner")
    if str(champion.get("runner_id", "")) != EXPECTED_RUNNER_ID:
        raise ValueError("champion was not produced by the fixed profile-meta runner")
    if str(run.get("status", "")) != "complete":
        raise ValueError("profile-meta run is not complete")
    if str(run.get("selection_policy", "")) != EXPECTED_SELECTION_POLICY:
        raise ValueError("profile-meta run permits a search space")
    if str(run.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-meta run uses another algorithm")
    if str(run.get("provider", "")) != EXPECTED_PROVIDER:
        raise ValueError("profile-meta run uses another provider")
    if tuple(float(item) for item in run.get("windows_seconds", [])) != EXPECTED_WINDOWS:
        raise ValueError("profile-meta run does not use exactly 0.8/2.8 seconds")
    if abs(float(run.get("long_weight", -1.0)) - EXPECTED_LONG_WEIGHT) > 1e-12:
        raise ValueError("profile-meta run has another long-window weight")
    _validate_realtime_cost(run.get("fresh_live_cost"))
    source_lock_hash = _validate_dataset_source_lock(run.get("dataset_source_lock"))
    if bool(run.get("sealed_v4_opened")) or bool(champion.get("sealed_v4_opened")):
        raise RuntimeError("profile-meta candidate already opened the sealed holdout")
    winner = champion.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("profile-meta champion has no locked winner")
    if str(run.get("winner", "")) != EXPECTED_CANDIDATE_NAME:
        raise ValueError("run does not lock the fixed profile-meta winner")
    if str(winner.get("algorithm_id", "")) != HYBRID_ALGORITHM_ID:
        raise ValueError("profile-meta winner uses another algorithm")
    if not bool(winner.get("promotion_gates_passed")):
        raise ValueError("profile-meta winner did not pass mandatory gates")
    lock_hash, control_config, config = _validate_candidate_lock(
        run.get("candidate_lock"), winner
    )
    if run["candidate_lock"].get("dataset_source_lock_sha256") != source_lock_hash:
        raise ValueError("candidate lock points to another dataset source lock")
    if str(champion.get("candidate_lock_sha256", "")) != lock_hash:
        raise ValueError("champion points to another profile-meta candidate lock")
    if str(champion.get("dataset_source_lock_sha256", "")) != source_lock_hash:
        raise ValueError("champion points to another dataset source lock")
    promotion = champion.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError("profile-meta champion has no promotion proof")
    if not all(
        bool(promotion.get(key))
        for key in ("candidate_eligible", "baseline_gates_passed", "run018_gates_passed")
    ):
        raise ValueError("profile-meta candidate did not pass both mandatory comparisons")
    per_video = winner.get("per_video")
    development = set(EXPECTED_DEVELOPMENT_IDS)
    if not isinstance(per_video, dict) or set(per_video) != development:
        raise ValueError("profile-meta winner was not scored on the exact 12-video cohort")
    if development.intersection(EXPECTED_HOLDOUT_IDS):
        raise AssertionError("development and holdout ID constants overlap")
    run_ids = tuple(
        str(video_id)
        for source in run.get("dataset_sources", [])
        if isinstance(source, dict)
        for video_id in source.get("video_ids", [])
    )
    if run_ids != EXPECTED_DEVELOPMENT_IDS:
        raise ValueError("profile-meta run does not describe the frozen 12-video cohort")
    for proof_key, winner_key in (
        ("candidate_vs_baseline", "vs_baseline"),
        ("candidate_vs_locked", "vs_locked"),
    ):
        proof = promotion.get(proof_key)
        if not _comparison_has_twelve_passed_videos(proof, development):
            raise ValueError(f"profile-meta 12-video proof failed: {proof_key}")
        if winner.get(winner_key) != proof:
            raise ValueError(f"winner and champion promotion proof differ: {proof_key}")
    allowed_statuses = {
        INITIAL_CHAMPION_STATUS,
        PASSED_CHAMPION_STATUS,
        FAILED_CHAMPION_STATUS,
    }
    if str(champion.get("status", "")) not in allowed_statuses:
        raise ValueError("profile-meta champion has an unexpected status")
    return LockedMetaCandidate(
        run_dir=run_dir,
        run_path=run_path,
        champion_path=champion_path,
        run=run,
        champion=champion,
        winner=winner,
        control_candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
        control_config=control_config,
        candidate_id=str(winner["candidate_id"]),
        config=config,
        candidate_lock_sha256=lock_hash,
        run_sha256=_sha256_file(run_path),
        champion_sha256=_sha256_file(champion_path),
    )


def _assert_one_shot(candidate: LockedMetaCandidate) -> Path:
    artifact = candidate.run_dir / HOLDOUT_ARTIFACT_NAME
    champion = candidate.champion
    staged = (artifact.with_name(artifact.name + ".tmp"), candidate.champion_path.with_name(candidate.champion_path.name + ".tmp"))
    if artifact.exists() or any(path.exists() for path in staged):
        raise RuntimeError("profile-meta holdout result or staged result already exists; rerun refused")
    if bool(champion.get("holdout_opened")) or bool(champion.get("sealed_v4_opened")):
        raise RuntimeError("profile-meta holdout is already marked as opened")
    if str(champion.get("status", "")) != INITIAL_CHAMPION_STATUS:
        raise RuntimeError("profile-meta champion is no longer awaiting its one-shot holdout")
    return artifact


def _load_holdout_lock(
    path: Path,
    requested_ids: Sequence[str],
    *,
    expected_sha256: str = EXPECTED_HOLDOUT_LOCK_SHA256,
) -> HoldoutLock:
    raw = path.read_bytes()
    actual_sha256 = _sha256_bytes(raw)
    if actual_sha256 != expected_sha256:
        raise ValueError("round-three holdout lock hash mismatch")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("round-three holdout lock must be a JSON object")
    video_ids = value.get("video_ids")
    if not isinstance(video_ids, list) or any(not isinstance(item, str) for item in video_ids):
        raise ValueError("round-three holdout lock must contain a video_ids array")
    normalized = tuple(item.strip() for item in video_ids)
    if normalized != EXPECTED_HOLDOUT_IDS:
        raise ValueError("round-three holdout lock IDs or order differ from the frozen seal")
    if tuple(requested_ids) != normalized:
        raise ValueError("dataset-source video order differs from the frozen holdout lock")
    if not bool(value.get("locked_before_round3_algorithm_analysis")):
        raise ValueError("holdout lock was not frozen before round-three analysis")
    if not bool(value.get("development_use_forbidden")):
        raise ValueError("holdout lock does not forbid development use")
    if tuple(value.get("already_opened_video_ids") or ()) != EXPECTED_DEVELOPMENT_IDS:
        raise ValueError("holdout lock names another opened development cohort")
    contract = value.get("live_cost_contract")
    if not isinstance(contract, dict):
        raise ValueError("holdout lock has no live-cost contract")
    if int(contract.get("max_fresh_embedding_windows_per_probe", 99)) != 2:
        raise ValueError("holdout lock does not cap fresh windows at two")
    if tuple(float(item) for item in contract.get("window_seconds", [])) != EXPECTED_WINDOWS:
        raise ValueError("holdout lock names another window pair")
    if "selection" not in value or value["selection"] is None:
        raise ValueError("round-three holdout lock has no selection record")
    return HoldoutLock(path, normalized, value["selection"], actual_sha256)


def _aggregate_wrong_ratio(per_video: dict[str, dict[str, Any]]) -> float:
    return float(mean(float(value["wrong_live_speech_ratio"]) for value in per_video.values()))


def _comparison_gates(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    video_ids: Sequence[str],
    *,
    reference_name: str,
    require_aggregate_improvement: bool,
) -> dict[str, Any]:
    expected = set(video_ids)
    baseline_per_video = reference.get("per_video")
    candidate_per_video = candidate.get("per_video")
    if not isinstance(baseline_per_video, dict) or set(baseline_per_video) != expected:
        raise ValueError(f"{reference_name} holdout scores do not cover exactly three videos")
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
        score_passed = score_delta >= -SCORE_TOLERANCE - 1e-12
        wrong_passed = wrong_delta <= WRONG_RATIO_TOLERANCE + 1e-12
        per_video[video_id] = {
            "score_delta_vs_reference": score_delta,
            "wrong_ratio_delta_vs_reference": wrong_delta,
            "score_gate_passed": score_passed,
            "wrong_ratio_gate_passed": wrong_passed,
            "passed": bool(score_passed and wrong_passed),
        }
    baseline_global = float(reference["aggregate"]["global_score"])
    candidate_global = float(candidate["aggregate"]["global_score"])
    score_delta = candidate_global - baseline_global
    wrong_delta = _aggregate_wrong_ratio(candidate_per_video) - _aggregate_wrong_ratio(
        baseline_per_video
    )
    aggregate = {
        "score_delta_vs_reference": score_delta,
        "wrong_ratio_delta_vs_reference": wrong_delta,
        "score_gate_passed": (
            score_delta > 1e-12
            if require_aggregate_improvement
            else score_delta >= -SCORE_TOLERANCE - 1e-12
        ),
        "wrong_ratio_gate_passed": wrong_delta <= WRONG_RATIO_TOLERANCE + 1e-12,
    }
    aggregate["passed"] = bool(
        aggregate["score_gate_passed"] and aggregate["wrong_ratio_gate_passed"]
    )
    all_individual = all(value["passed"] for value in per_video.values())
    return {
        "reference": reference_name,
        "aggregate_score_requires_improvement": require_aggregate_improvement,
        "per_video": per_video,
        "aggregate": aggregate,
        "all_three_individual_gates_passed": all_individual,
        "passed": bool(all_individual and aggregate["passed"]),
    }


def _holdout_gates(
    baseline: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    video_ids: Sequence[str],
) -> dict[str, Any]:
    vs_baseline = _comparison_gates(
        baseline,
        candidate,
        video_ids,
        reference_name="production_baseline",
        require_aggregate_improvement=True,
    )
    vs_run018 = _comparison_gates(
        control,
        candidate,
        video_ids,
        reference_name="locked_run018",
        require_aggregate_improvement=False,
    )
    return {
        "fixed_tolerances": {
            "per_video_score": SCORE_TOLERANCE,
            "per_video_wrong_ratio": WRONG_RATIO_TOLERANCE,
            "aggregate_baseline_score_requires_improvement": True,
            "aggregate_run018_score_non_regression_tolerance": SCORE_TOLERANCE,
            "aggregate_wrong_ratio": WRONG_RATIO_TOLERANCE,
        },
        "vs_baseline": vs_baseline,
        "vs_run018": vs_run018,
        "baseline_gates_passed": bool(vs_baseline["passed"]),
        "run018_gates_passed": bool(vs_run018["passed"]),
        "holdout_passed": bool(vs_baseline["passed"] and vs_run018["passed"]),
    }


def _evaluate_holdout(
    sources: Sequence[HoldoutDatasetSource], candidate: LockedMetaCandidate
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # This call completes all three production-baseline traces and scores before
    # either replay begins.  Both subsequent replays consume the same prepared
    # cached evidence and therefore request no additional embeddings.
    prepared, baseline = round2._prepare_baseline(sources)
    control = round2.Candidate(
        "locked_run018",
        "locked_run018_control",
        candidate.control_config,
        candidate.control_candidate_id,
    )
    control_scored = round2._score_candidate(prepared, control)
    contender = round2.Candidate(
        EXPECTED_CANDIDATE_NAME,
        EXPECTED_CANDIDATE_FAMILY,
        candidate.config,
        candidate.candidate_id,
    )
    scored = round2._score_candidate(prepared, contender)
    return baseline, control_scored, scored


def _holdout_artifact(
    candidate: LockedMetaCandidate,
    lock: HoldoutLock,
    baseline: dict[str, Any],
    control: dict[str, Any],
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
        "evaluation_order": [
            "baseline_all_three",
            "locked_run018_control_all_three",
            "single_locked_meta_candidate_all_three",
        ],
        "baseline_completed_before_candidate_evaluation": True,
        "control_evaluation_count": 1,
        "candidate_evaluation_count": 1,
        "cached_control_replay_additional_embedding_requests": 0,
        "video_ids": list(lock.video_ids),
        "sealed_v4_opened": True,
        "holdout_lock": {
            "path": str(lock.path),
            "sha256": lock.sha256,
            "selection": lock.selection,
        },
        "source": {
            "runner_id": EXPECTED_RUNNER_ID,
            "run_path": str(candidate.run_path),
            "run_sha256": candidate.run_sha256,
            "champion_path": str(candidate.champion_path),
            "champion_sha256_before_holdout": candidate.champion_sha256,
            "candidate_lock_sha256": candidate.candidate_lock_sha256,
        },
        "candidate": {
            "name": EXPECTED_CANDIDATE_NAME,
            "family": EXPECTED_CANDIDATE_FAMILY,
            "candidate_id": candidate.candidate_id,
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "policy_id": EXPECTED_POLICY_ID,
            "config": asdict(candidate.config),
        },
        "provider": EXPECTED_PROVIDER,
        "windows_seconds": list(EXPECTED_WINDOWS),
        "long_weight": EXPECTED_LONG_WEIGHT,
        "fresh_live_cost": round2._fresh_live_cost(),
        "baseline": baseline,
        "locked_run018_control_scores": control,
        "candidate_scores": scored,
        "gates": gates,
        "passed": bool(gates["holdout_passed"]),
        "production_wired": False,
        "fresh_live_verified": False,
    }


def _write_json_temporary(path: Path, value: Any) -> Path:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
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
    *,
    expected_champion_sha256: str | None = None,
) -> dict[str, Any]:
    if artifact_path.exists():
        raise RuntimeError("profile-meta holdout result already exists; rerun refused")
    if expected_champion_sha256 is not None:
        if _sha256_file(champion_path) != expected_champion_sha256:
            raise RuntimeError("profile-meta champion changed after one-shot validation")
    updated = deepcopy(original_champion)
    passed = bool(artifact["passed"])
    updated.update(
        {
            "status": PASSED_CHAMPION_STATUS if passed else FAILED_CHAMPION_STATUS,
            "holdout_opened": True,
            "sealed_v4_opened": True,
            "holdout_passed": passed,
            "holdout_artifact": artifact_path.name,
            "holdout_verifier_id": VERIFIER_ID,
            "holdout_lock_sha256": artifact["holdout_lock"]["sha256"],
            "fresh_live_verification_required": passed,
            "production_defaults_changed": False,
        }
    )
    artifact_tmp: Path | None = None
    champion_tmp: Path | None = None
    try:
        artifact_tmp = _write_json_temporary(artifact_path, artifact)
        champion_tmp = _write_json_temporary(champion_path, updated)
        if artifact_path.exists():
            raise RuntimeError("profile-meta holdout appeared during staged commit")
        os.replace(artifact_tmp, artifact_path)
        os.replace(champion_tmp, champion_path)
    finally:
        for temporary in (artifact_tmp, champion_tmp):
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Pure string validation comes first: a wrong ID/order cannot open the
    # candidate, lock, corpus, inputs, or output path.
    sources = _validate_holdout_sources(args.dataset_source)
    requested_ids = tuple(
        video_id for source in sources for video_id in source.video_ids
    )

    # Candidate proof and one-shot state are validated before the sealed lock or
    # any dataset root is opened.
    candidate = _load_locked_meta_candidate(args.meta_run_dir)
    artifact_path = _assert_one_shot(candidate)
    lock = _load_holdout_lock(DEFAULT_HOLDOUT_LOCK, requested_ids)

    baseline, control, scored = _evaluate_holdout(sources, candidate)
    gates = _holdout_gates(baseline, control, scored, lock.video_ids)
    artifact = _holdout_artifact(
        candidate, lock, baseline, control, scored, gates
    )
    _commit_holdout(
        artifact_path,
        candidate.champion_path,
        artifact,
        candidate.champion,
        expected_champion_sha256=candidate.champion_sha256,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "candidate_id": candidate.candidate_id,
                "video_ids": list(lock.video_ids),
                "baseline_global_score": baseline["aggregate"]["global_score"],
                "run018_global_score": control["aggregate"]["global_score"],
                "candidate_global_score": scored["aggregate"]["global_score"],
                "baseline_gates": gates["vs_baseline"],
                "run018_gates": gates["vs_run018"],
                "holdout_lock_sha256": lock.sha256,
                "artifact": str(artifact_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if gates["holdout_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
