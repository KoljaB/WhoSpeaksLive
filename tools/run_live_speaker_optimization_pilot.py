from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_algorithm import ALGORITHM_ID, LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import SCORER_ID
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import STACKED_CACHE_POLICY_ID, load_profile_events_jsonl

from validate_live_shifting_window_corpus import validate_corpus


RUNNER_ID = "linux_live_speaker_pilot_v1"
_STOP_REQUESTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def stable_candidate_id(
    config: dict[str, Any],
    provider_policy: dict[str, Any],
    split: dict[str, Any],
    input_identity: str,
) -> str:
    return _stable_id({
        "algorithm_id": ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "config": config,
        "provider_policy": provider_policy,
        "split": split,
        "input_identity": input_identity,
    })


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_if(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def _gate(name: str, passed: bool, evidence: Any, required: bool = True) -> dict[str, Any]:
    return {"name": name, "required": required, "passed": bool(passed), "evidence": evidence}


def _existing_files(paths: list[Path]) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    for path in paths:
        row: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            row.update({"exists": True, "size": path.stat().st_size, "sha256": _sha256(path)})
        else:
            row["exists"] = False
        rows.append(row)
    return all(row["exists"] for row in rows), rows


def _audit_canonicals(paths: list[Path]) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                segments = read_canonical_segments(path)
                scorable_count = 0
                zero_duration_count = 0
                hard_invalid_count = 0
                for item in segments:
                    start = float(item["start"])
                    end = float(item["end"])
                    speaker = str(item["speaker"]).strip()
                    if not speaker or not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
                        hard_invalid_count += 1
                    elif end == start:
                        # The versioned scorer deliberately ignores zero-width rows.
                        zero_duration_count += 1
                    else:
                        scorable_count += 1
                valid = scorable_count > 0 and hard_invalid_count == 0
                row.update({
                    "valid": valid, "segment_count": len(segments),
                    "scorable_segment_count": scorable_count,
                    "ignored_zero_duration_count": zero_duration_count,
                    "hard_invalid_count": hard_invalid_count,
                    "sha256": _sha256(path),
                })
            except Exception as exc:
                row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return bool(rows) and all(row.get("valid") is True for row in rows), rows


def _timeline_rows(corpus_root: Path, video_id: str) -> int:
    path = corpus_root / "videos" / video_id / "timeline" / "right_edges.i64.npy"
    return int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])


def _audit_masks(
    paths: list[Path], video_ids: list[str], corpus_root: Path
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path, video_id in zip(paths, video_ids):
        row: dict[str, Any] = {"path": str(path), "video_id": video_id, "exists": path.is_file()}
        if path.is_file():
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                expected_rows = _timeline_rows(corpus_root, video_id)
                values_valid = array.dtype in (np.dtype(bool), np.dtype(np.uint8)) and not np.any(
                    (array != 0) & (array != 1)
                )
                valid = array.ndim == 1 and int(array.shape[0]) == expected_rows and bool(values_valid)
                row.update({
                    "valid": valid, "dtype": str(array.dtype), "rows": int(array.shape[0]),
                    "expected_rows": expected_rows, "true_count": int(np.count_nonzero(array)),
                    "sha256": _sha256(path),
                })
            except Exception as exc:
                row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return bool(rows) and all(row.get("valid") is True for row in rows), rows


def _audit_profiles(
    paths: list[Path], video_ids: list[str], provider_weights: dict[str, float], corpus_root: Path
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    positive_providers = [
        provider for provider, weight in provider_weights.items() if float(weight) > 0.0
    ]
    for path, video_id in zip(paths, video_ids):
        row: dict[str, Any] = {
            "path": str(path), "video_id": video_id,
            "provider_weights": provider_weights, "exists": path.is_file()
        }
        if path.is_file():
            try:
                events = load_profile_events_jsonl(path)
                expected_dim = sum(
                    int(np.load(
                        corpus_root / "providers" / provider / "videos" / video_id
                        / "lengths" / "1000ms" / "embeddings.f32.npy",
                        mmap_mode="r", allow_pickle=False,
                    ).shape[1])
                    for provider in positive_providers
                )
                chronological = all(
                    events[index].available_at <= events[index + 1].available_at + 1e-9
                    for index in range(len(events) - 1)
                )
                valid = bool(events) and chronological and all(
                    int(event.centroid.shape[0]) == expected_dim for event in events
                )
                row.update({
                    "valid": valid, "event_count": len(events), "expected_dimension": expected_dim,
                    "chronological": chronological, "sha256": _sha256(path),
                })
            except Exception as exc:
                row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return bool(rows) and all(row.get("valid") is True for row in rows), rows


def _corpus_metadata_identity(corpus_root: Path, video_ids: list[str]) -> dict[str, Any]:
    files = [corpus_root / "corpus.json", corpus_root / "progress.json"]
    for video_id in video_ids:
        files.extend([
            corpus_root / "videos" / video_id / "source.json",
            corpus_root / "videos" / video_id / "timeline" / "metadata.json",
            corpus_root / "videos" / video_id / "timeline" / "right_edges.i64.npy",
        ])
    present, rows = _existing_files(files)
    payload = [{"path": row["path"], "sha256": row.get("sha256"), "size": row.get("size")} for row in rows]
    return {"complete": present, "files": rows, "identity": _stable_id(payload)}


def _write_notebook(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Live Speaker Linux Pilot — 2026-07-19",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Mode: `{report['mode']}`",
        f"- Highest tier: `{report['highest_promotion_tier']}`",
        f"- Score search ran: `{str(report['score_search_ran']).lower()}`",
        f"- Algorithm: `{report['algorithm_id']}`",
        f"- Scorer: `{report['scorer_id']}`",
        "- Sealed holdout: labels and decision traces unopened; dense-cache integrity only.",
        "",
        "## Readiness gates",
        "",
    ]
    for gate in report["readiness_gates"]:
        lines.append(f"- [{'x' if gate['passed'] else ' '}] `{gate['name']}`")
    lines.extend([
        "",
        "## Outcome",
        "",
        report["summary"],
        "",
        "## Next step",
        "",
        report["next_step"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-boxed Linux-only live-speaker readiness/optimization pilot")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--stop-new-candidates-after-seconds", type=int, default=3300)
    parser.add_argument("--candidate-timeout-seconds", type=int, default=60)
    parser.add_argument("--campaign-monotonic-start", type=float)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_handler)

    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)
    for trace_name in ("baseline.jsonl", "champion.jsonl"):
        (run_dir / "traces" / trace_name).touch(exist_ok=True)

    started_monotonic = args.campaign_monotonic_start or time.monotonic()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    code_paths = {
        "runner": Path(__file__).resolve(),
        "algorithm": ROOT / "src/window/live_speaker_algorithm.py",
        "replay": ROOT / "src/window/live_speaker_replay.py",
        "scorer": ROOT / "src/window/live_speaker_benchmark.py",
        "spec": spec_path,
    }
    code_hashes = {name: _sha256(path) for name, path in code_paths.items()}
    all_videos = list(dict.fromkeys(
        spec["split"]["search"] + spec["split"]["validation"] + spec["split"]["sealed_holdout"]
    ))
    scored_videos = list(dict.fromkeys(spec["split"]["search"] + spec["split"]["validation"]))
    provider_weights = dict(spec["baseline"]["provider_weights"])
    providers = list(provider_weights)
    input_root = ROOT / "runtime/optimization/live_replay_inputs"
    canonical_paths = [input_root / video / "canonical_diarization.json" for video in scored_videos]
    speech_paths = [input_root / video / "speech_gate.u1.npy" for video in scored_videos]
    probe_paths = [input_root / video / "probe_schedule.u1.npy" for video in scored_videos]
    release_paths = [input_root / video / "release_gate.u1.npy" for video in scored_videos]
    profile_paths = [input_root / video / "production_stack.profiles.jsonl" for video in scored_videos]
    production_contract_path = input_root / "shared_core_production_contract.json"
    trace_path = input_root / scored_videos[0] / "production_shared_core_trace.jsonl"
    exact_path = input_root / "recorded_trace_replay_parity.json"
    vector_path = input_root / "fresh_cached_vector_parity.json"
    expected_input_paths = (
        canonical_paths + speech_paths + probe_paths + release_paths + profile_paths
        + [production_contract_path, trace_path, exact_path, vector_path]
    )
    _inputs_complete, input_identity_rows = _existing_files(expected_input_paths)
    input_tape_identity = _stable_id([
        {
            "path": str(Path(row["path"]).relative_to(ROOT)),
            "exists": row["exists"],
            "size": row.get("size"),
            "sha256": row.get("sha256"),
        }
        for row in input_identity_rows
    ])
    corpus_root = (ROOT / spec["execution"]["corpus_root"]).resolve()
    corpus_identity = _corpus_metadata_identity(corpus_root, all_videos)
    run_identity_payload = {
        "runner_id": RUNNER_ID,
        "algorithm_id": ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "spec_hash": code_hashes["spec"],
        "code_hashes": code_hashes,
        "corpus_metadata_identity": corpus_identity["identity"],
        "input_tape_identity": input_tape_identity,
        "split": spec["split"],
    }
    run_identity = _stable_id(run_identity_payload)
    existing = _read_json_if(run_dir / "run.json")
    if existing is not None and existing.get("run_identity") != run_identity:
        print("Refusing resume: run identity changed", file=sys.stderr)
        return 4

    run_record = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "run_identity": run_identity,
        "run_identity_payload": run_identity_payload,
        "started_utc": existing.get("started_utc", started_utc) if existing else started_utc,
        "campaign_monotonic_start": started_monotonic,
        "stop_new_work_monotonic": started_monotonic + args.stop_new_candidates_after_seconds,
        "hard_deadline_monotonic": started_monotonic + args.max_wall_seconds,
        "host_required": spec["execution"]["host"],
        "tests_and_scores_on_linux_only": True,
        "sealed_holdout_policy": "presence_and_dense_cache_integrity_only__no_labels_or_traces",
        "corpus_metadata": corpus_identity,
        "input_tape_identity": input_tape_identity,
        "input_tape_files": input_identity_rows,
    }
    _atomic_json(run_dir / "run.json", run_record)
    _atomic_json(run_dir / "progress.json", {
        "phase": "PREFLIGHT", "mode": "READINESS_ONLY", "percent": 5.0,
        "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
        "current_candidate": None, "current_best": None,
    })

    corpus_audits: dict[str, Any] = {}
    corpus_errors: dict[str, str] = {}
    for video_id in all_videos:
        try:
            corpus_audits[video_id] = validate_corpus(corpus_root, video_id)
        except Exception as exc:  # a readiness report must survive a corrupt block
            corpus_errors[video_id] = f"{type(exc).__name__}: {exc}"
    expected = spec["dense_corpus_expectation"]
    dense_ok = not corpus_errors and len(corpus_audits) == len(all_videos) and all(
        item.get("status") == expected["required_status"]
        and int(item.get("provider_count", 0)) == int(expected["provider_count"])
        and item.get("length_count_per_provider") == [len(expected["window_lengths_seconds"])]
        and int(item.get("partial_file_count", -1)) == 0
        for item in corpus_audits.values()
    )

    canonical_ok, canonical_evidence = _audit_canonicals(canonical_paths)
    speech_ok, speech_evidence = _audit_masks(speech_paths, scored_videos, corpus_root)
    probe_ok, probe_evidence = _audit_masks(probe_paths, scored_videos, corpus_root)
    release_ok, release_evidence = _audit_masks(release_paths, scored_videos, corpus_root)
    profiles_ok, profiles_evidence = _audit_profiles(
        profile_paths, scored_videos, provider_weights, corpus_root
    )

    production_contract = _read_json_if(production_contract_path)
    production_ok = bool(
        isinstance(production_contract, dict)
        and production_contract.get("algorithm_id") == ALGORITHM_ID
        and production_contract.get("all_live_decisions_route_through_shared_core") is True
    )
    trace_ok, trace_evidence = _existing_files([trace_path])
    exact = _read_json_if(exact_path)
    exact_ok = bool(isinstance(exact, dict) and exact.get("exact_match") is True and exact.get("algorithm_id") == ALGORITHM_ID)
    vector = _read_json_if(vector_path)
    vector_ok = bool(isinstance(vector, dict) and vector.get("decision_exact_match") is True and vector.get("sample_count", 0) > 0)
    positive_provider_weights = [float(value) for value in spec["baseline"]["provider_weights"].values() if float(value) > 0.0]
    ensemble_ok = bool(positive_provider_weights)
    baseline_existing = _read_json_if(run_dir / "baseline_reproduction.json")
    baseline_ok = bool(
        isinstance(baseline_existing, dict)
        and baseline_existing.get("status") == "REPRODUCED_TWICE_IDENTICALLY"
        and baseline_existing.get("run_identity") == run_identity
    )

    gates = [
        _gate("dense_corpus_complete_and_valid", dense_ok, {"audits": corpus_audits, "errors": corpus_errors}),
        _gate("canonical_inputs_hash_frozen_search_and_validation", canonical_ok, canonical_evidence),
        _gate("causal_probe_gate_tapes_hash_frozen", speech_ok, speech_evidence),
        _gate("causal_probe_schedule_tapes_hash_frozen", probe_ok, probe_evidence),
        _gate("causal_release_gate_tapes_hash_frozen", release_ok, release_evidence),
        _gate("chronological_provider_profile_tapes_hash_frozen", profiles_ok, profiles_evidence),
        _gate("production_routes_live_decisions_through_shared_core", production_ok, {
            "path": str(production_contract_path), "value": production_contract,
        }),
        _gate("production_shared_core_trace_present", trace_ok, trace_evidence),
        _gate("recorded_trace_cached_replay_exact", exact_ok, {"path": str(exact_path), "value": exact}),
        _gate("sampled_fresh_cached_vector_and_decision_parity", vector_ok, {"path": str(vector_path), "value": vector}),
        _gate("production_baseline_provider_policy_supported", ensemble_ok, {
            "provider_weights": spec["baseline"]["provider_weights"],
            "cached_stack_policy_id": STACKED_CACHE_POLICY_ID,
            "validity_policy": "intersection_of_positive_weight_component_blocks",
        }),
        _gate("production_baseline_reproduced_twice_identically", baseline_ok, baseline_existing),
    ]
    readiness_ok = all(gate["passed"] for gate in gates if gate["required"])

    trial = {
        "trial_kind": "readiness_preflight",
        "trial_id": _stable_id({"run_identity": run_identity, "gates": gates}),
        "parent": None,
        "hypothesis": "All causal inputs and shared-core parity proofs exist before score search.",
        "started_utc": started_utc,
        "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 6),
        "tier": "READINESS_ONLY",
        "decision": "pass" if readiness_ok else "no_go",
        "gates": gates,
    }
    _append_jsonl(run_dir / "trials.jsonl", trial)

    if not readiness_ok:
        baseline_payload = {
            "status": "NOT_RUN_MISSING_READINESS",
            "run_identity": run_identity,
            "missing_gates": [gate["name"] for gate in gates if gate["required"] and not gate["passed"]],
            "baseline_config": spec["baseline"],
        }
        _atomic_json(run_dir / "baseline_reproduction.json", baseline_payload)
        _atomic_json(run_dir / "champion.json", {
            "status": "NO_CHAMPION", "tier": "READINESS_ONLY", "candidate_id": None,
            "reason": "Optimization readiness gate was not met; no score search ran.",
        })
        report = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "run_identity": run_identity,
            "runner_id": RUNNER_ID,
            "algorithm_id": ALGORITHM_ID,
            "scorer_id": SCORER_ID,
            "mode": "READINESS_ONLY",
            "highest_promotion_tier": "READINESS_ONLY",
            "score_search_ran": False,
            "readiness_gates": gates,
            "passed_gates": [gate["name"] for gate in gates if gate["passed"]],
            "failed_gates": [gate["name"] for gate in gates if gate["required"] and not gate["passed"]],
            "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 6),
            "stop_requested": _STOP_REQUESTED,
            "summary": "The dense embedding corpus is usable, but causal tapes and production/shared-core parity evidence are incomplete. No candidate was scored or promoted.",
            "next_step": "Capture real production probe/release gate events and chronological provider profiles, wire production to the shared core, then record and exactly replay one reference trace.",
        }
        _atomic_json(run_dir / "final_report.json", report)
        _atomic_json(run_dir / "progress.json", {
            "phase": "FINALIZED", "mode": "READINESS_ONLY", "percent": 100.0,
            "elapsed_seconds": report["elapsed_seconds"], "current_candidate": None,
            "current_best": None, "exit_code": 3,
        })
        _write_notebook(ROOT / "runtime/optimization/live_speaker_optimization_session_2026-07-19.md", report)
        print(json.dumps({"status": "READINESS_NO_GO", "failed_gates": report["failed_gates"], "run_dir": str(run_dir)}, indent=2))
        return 3

    # This first runner deliberately stops here until the chronological shared-core
    # baseline evaluator supports the frozen multi-provider production policy.
    print("All readiness gates are green, but no candidate implementation is registered.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
