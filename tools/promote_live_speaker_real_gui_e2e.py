"""The only supported promotion path for a production live-speaker champion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_e2e_contract import (  # noqa: E402
    PRODUCTION_CHAMPION_STATUS,
    REAL_GUI_E2E_CONTRACT_ID,
    file_sha256,
    stable_sha256,
    validate_real_gui_e2e_observation,
)


PROMOTER_ID = "real_gui_live_speaker_e2e_promoter_v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fingerprint(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any, ...]:
    attestation = payload.get("attestation")
    values: list[Any] = []
    for key in keys:
        current: Any = attestation
        for part in key.split("."):
            current = current.get(part) if isinstance(current, dict) else None
        values.append(current)
    return tuple(values)


def _score(payload: dict[str, Any]) -> float:
    try:
        return float(payload["summary"]["strict_browser_live_score"])
    except (KeyError, TypeError, ValueError):
        # Observation validation records the precise error.  Returning zero
        # here keeps the promoter on its fail-closed audit path instead of
        # crashing before it can write a rejection artifact.
        return 0.0


def _artifact_expectations(
    label: str,
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the immutable runtime/code identity claimed by an artifact."""
    errors: list[str] = []
    config = record.get("expected_runtime_config")
    config_hash = record.get("expected_runtime_config_sha256")
    source_hash = record.get("expected_source_tree_sha256")

    if not isinstance(config, dict) or not config:
        errors.append(f"{label} artifact: expected_runtime_config is missing or empty")
    elif config_hash != stable_sha256(config):
        errors.append(f"{label} artifact: expected_runtime_config_sha256 does not match config")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash.lower())
    ):
        errors.append(f"{label} artifact: expected_source_tree_sha256 is missing or invalid")

    if errors:
        return None, errors
    return {
        "runtime_config": config,
        "runtime_config_sha256": config_hash,
        "source_tree_sha256": source_hash,
    }, errors


def _validate_artifact_bindings(
    label: str,
    payloads: list[dict[str, Any]],
    expectations: dict[str, Any] | None,
) -> list[str]:
    """Require every run to attest the exact config and source expected by its artifact."""
    if expectations is None:
        return []
    errors: list[str] = []
    for index, payload in enumerate(payloads):
        attestation = payload.get("attestation")
        attestation = attestation if isinstance(attestation, dict) else {}
        if attestation.get("runtime_config") != expectations["runtime_config"]:
            errors.append(
                f"{label} run {index + 1}: effective runtime config does not match artifact"
            )
        if attestation.get("runtime_config_sha256") != expectations["runtime_config_sha256"]:
            errors.append(
                f"{label} run {index + 1}: runtime config hash does not match artifact"
            )
        code = attestation.get("code")
        observed_source_hash = code.get("source_tree_sha256") if isinstance(code, dict) else None
        if observed_source_hash != expectations["source_tree_sha256"]:
            errors.append(
                f"{label} run {index + 1}: source tree hash does not match artifact"
            )
    return errors


def _validate_group(
    label: str,
    paths: list[Path],
    *,
    minimum_runs: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    payloads: list[dict[str, Any]] = []
    if len(paths) < minimum_runs:
        errors.append(f"{label}: requires at least {minimum_runs} complete real GUI runs")
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label} {path}: unreadable evidence: {exc}")
            continue
        payload_errors = validate_real_gui_e2e_observation(payload)
        errors.extend(f"{label} {path}: {message}" for message in payload_errors)
        payloads.append(payload)
    if payloads:
        fingerprint_keys = (
            "runtime_config_sha256",
            "canonical.sha256",
            "media.audio_sha256",
            "media.video_id",
            "code.source_tree_sha256",
        )
        expected = _fingerprint(payloads[0], fingerprint_keys)
        if any(_fingerprint(payload, fingerprint_keys) != expected for payload in payloads[1:]):
            errors.append(f"{label}: runs do not share identical code, config, audio, and reference hashes")
    return payloads, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a live-speaker candidate only after repeated, full-length, real-time "
            "runs through the normal visible Chrome GUI."
        )
    )
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--baseline-observation", type=Path, action="append", required=True)
    parser.add_argument("--candidate-observation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-baseline-runs", type=int, default=3)
    parser.add_argument("--minimum-candidate-runs", type=int, default=3)
    parser.add_argument("--minimum-live-score-improvement", type=float, default=0.0)
    parser.add_argument("--expected-video-id", default="JWS-qfR6K3w")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = args.baseline_artifact.resolve()
    candidate_path = args.candidate_artifact.resolve()
    baseline_record = _read_json(baseline_path)
    candidate_record = _read_json(candidate_path)
    baseline_hash = file_sha256(baseline_path)
    candidate_hash = file_sha256(candidate_path)
    if not baseline_hash:
        raise FileNotFoundError(baseline_path)
    if not candidate_hash:
        raise FileNotFoundError(candidate_path)

    baseline_paths = [path.resolve() for path in args.baseline_observation]
    candidate_paths = [path.resolve() for path in args.candidate_observation]
    all_observation_paths = baseline_paths + candidate_paths
    errors: list[str] = []
    if len(set(all_observation_paths)) != len(all_observation_paths):
        errors.append("observation paths must be unique across baseline and candidate runs")
    if len(baseline_paths) != len(candidate_paths):
        errors.append("baseline and candidate must have equal real-GUI run counts")
    baseline, baseline_errors = _validate_group(
        "baseline", baseline_paths, minimum_runs=max(3, args.minimum_baseline_runs)
    )
    errors.extend(baseline_errors)
    candidate, candidate_errors = _validate_group(
        "candidate", candidate_paths, minimum_runs=max(3, args.minimum_candidate_runs)
    )
    errors.extend(candidate_errors)

    baseline_expectations, artifact_errors = _artifact_expectations(
        "baseline", baseline_record
    )
    errors.extend(artifact_errors)
    candidate_expectations, artifact_errors = _artifact_expectations(
        "candidate", candidate_record
    )
    errors.extend(artifact_errors)
    errors.extend(_validate_artifact_bindings("baseline", baseline, baseline_expectations))
    errors.extend(_validate_artifact_bindings("candidate", candidate, candidate_expectations))

    world_tape_run_ids = [
        str(payload.get("attestation", {}).get("world_tape", {}).get("run_id") or "")
        for payload in baseline + candidate
    ]
    present_run_ids = [run_id for run_id in world_tape_run_ids if run_id]
    if len(set(present_run_ids)) != len(present_run_ids):
        errors.append("live-speaker World Tape run ids must be unique across all runs")

    if baseline and candidate:
        source_keys = ("canonical.sha256", "media.audio_sha256", "media.video_id")
        if _fingerprint(baseline[0], source_keys) != _fingerprint(candidate[0], source_keys):
            errors.append("baseline and candidate did not run against identical source audio and reference")
        observed_video_id = _fingerprint(candidate[0], ("media.video_id",))[0]
        if str(observed_video_id) != str(args.expected_video_id):
            errors.append(
                f"wrong canary video: observed {observed_video_id!r}, expected {args.expected_video_id!r}"
            )
        for index, payload in enumerate(baseline):
            attestation = payload.get("attestation")
            bound_hash = (
                attestation.get("candidate_artifact", {}).get("sha256")
                if isinstance(attestation, dict) else None
            )
            if bound_hash != baseline_hash:
                errors.append(
                    f"baseline run {index + 1}: baseline artifact hash does not match {baseline_path}"
                )
        for index, payload in enumerate(candidate):
            attestation = payload.get("attestation")
            bound_hash = (
                attestation.get("candidate_artifact", {}).get("sha256")
                if isinstance(attestation, dict) else None
            )
            if bound_hash != candidate_hash:
                errors.append(
                    f"candidate run {index + 1}: candidate artifact hash does not match {candidate_path}"
                )

    baseline_scores = [_score(payload) for payload in baseline] if baseline else []
    candidate_scores = [_score(payload) for payload in candidate] if candidate else []
    baseline_median = statistics.median(baseline_scores) if baseline_scores else 0.0
    candidate_median = statistics.median(candidate_scores) if candidate_scores else 0.0
    delta = candidate_median - baseline_median
    required_delta = float(args.minimum_live_score_improvement)
    if baseline_scores and candidate_scores and delta <= required_delta:
        errors.append(
            f"real GUI median score did not improve: {candidate_median:.6f} vs "
            f"{baseline_median:.6f} (required delta > {required_delta:.6f})"
        )
    if candidate_scores and abs(delta) < 0.01 and len(candidate_scores) < 3:
        errors.append("close real-GUI delta below 0.01 requires three candidate runs")

    passed = not errors
    audit = {
        "schema_version": 2,
        "contract_id": REAL_GUI_E2E_CONTRACT_ID,
        "promoter_id": PROMOTER_ID,
        "baseline_artifact": {
            "path": str(baseline_path),
            "sha256": baseline_hash,
        },
        "candidate_artifact": {
            "path": str(candidate_path),
            "sha256": candidate_hash,
        },
        "baseline": {
            "observations": [str(path) for path in baseline_paths],
            "scores": baseline_scores,
            "median_score": round(baseline_median, 6),
        },
        "candidate": {
            "observations": [str(path) for path in candidate_paths],
            "scores": candidate_scores,
            "median_score": round(candidate_median, 6),
        },
        "live_score_delta": round(delta, 6),
        "errors": errors,
    }
    payload = {
        **candidate_record,
        "status": PRODUCTION_CHAMPION_STATUS if passed else "REJECTED_BY_REAL_GUI_LIVE_E2E",
        "fresh_live_verified": passed,
        "production_promotion_eligible": passed,
        "requires_real_gui_live_e2e": not passed,
        "real_gui_live_e2e_promotion": audit,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
