"""Offline provider/window representation-oracle diagnostics.

This research-only tool never invokes an embedding model.  It exports the
authentic World-Tape probe schedule, screens the already materialized dense
shifting-window cache, and feeds selected future-centroid predictions back
through the unchanged browser reducer and strict scorer.

Canonical speaker labels are used both to construct full-video centroids and
to select candidates.  Every result is therefore an intentionally leaky
``INVALID_FUTURE_CENTROID_ORACLE`` diagnostic, never promotion evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


CONTRACT_ID = "whospeaks.live_provider_window_future_centroid_oracle.v1"
TOP7 = (
    "20v1OxUXcQY",
    "JWS-qfR6K3w",
    "L-CfFo5aQGU",
    "S_o3y7CzDUY",
    "mBeT_AoCXvc",
    "onHUfyRP1BE",
    "pD4IdQTmneI",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    result = np.zeros_like(values)
    np.divide(values, norms, out=result, where=np.isfinite(norms) & (norms > 1e-8))
    return result


def _canonical_at(segments: list[dict[str, Any]], media_time: float) -> str:
    for row in segments:
        if float(row["start"]) <= media_time < float(row["end"]):
            return str(row["speaker"])
    return ""


def export_schedule(args: argparse.Namespace) -> int:
    from analyze_live_speaker_open_set_tracklets import (
        _load_base_config,
        _prepare_tape,
        _unit,
    )
    from window.live_speaker_probe_scoring import read_canonical_segments

    parity = _read_json(args.parity_report.resolve())
    base_config = _load_base_config(args.base_artifact.resolve())
    runs: list[dict[str, Any]] = []
    for run in parity.get("runs") or []:
        video_id = str(run.get("video_id") or "")
        if video_id not in TOP7:
            continue
        prepared = _prepare_tape(run, base_config)
        segments = read_canonical_segments(prepared.canonical_path)
        steps: list[dict[str, Any]] = []
        for index, step in enumerate(prepared.steps):
            payload = step.payload
            media_time = float(payload.get("media_time") or 0.0)
            steps.append(
                {
                    "index": index,
                    "media_time": media_time,
                    "truth": _canonical_at(segments, media_time),
                    "speech": bool(payload.get("speech")),
                    "release_signal": bool(payload.get("release_signal")),
                    "dedicated": bool(str(payload.get("probe_id") or "")),
                    "embedding_available": _unit(payload.get("embedding")) is not None,
                    "context_embedding_available": _unit(payload.get("context_embedding"))
                    is not None,
                }
            )
        runs.append(
            {
                "video_id": prepared.video_id,
                "run_id": prepared.run_id,
                "step_count": len(steps),
                "steps": steps,
            }
        )
    result = {
        "contract_id": CONTRACT_ID,
        "status": "INVALID_FUTURE_CENTROID_ORACLE_SCHEDULE",
        "future_label_leakage": True,
        "model_inference_performed": False,
        "source_parity_report": str(args.parity_report.resolve()),
        "source_base_artifact": str(args.base_artifact.resolve()),
        "runs": runs,
    }
    _write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "run_count": len(runs),
                "step_count": sum(item["step_count"] for item in runs),
            },
            indent=2,
        )
    )
    return 0


def _providers(corpus_root: Path) -> list[str]:
    return sorted(
        item.name
        for item in (corpus_root / "providers").iterdir()
        if item.is_dir()
    )


def _lengths(corpus_root: Path, provider: str, video_id: str) -> list[int]:
    root = corpus_root / "providers" / provider / "videos" / video_id / "lengths"
    return sorted(
        int(item.name.removesuffix("ms"))
        for item in root.iterdir()
        if item.is_dir() and item.name.endswith("ms")
    )


def _load_block(
    corpus_root: Path, provider: str, video_id: str, length_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base = (
        corpus_root
        / "providers"
        / provider
        / "videos"
        / video_id
        / "lengths"
        / f"{length_ms:04d}ms"
    )
    embeddings = np.load(base / "embeddings.f32.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(base / "valid.u1.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    right_edges = np.load(
        corpus_root / "videos" / video_id / "timeline" / "right_edges.i64.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = _read_json(base / "metadata.json")
    return embeddings, valid, right_edges, metadata


def _causal_indices(right_edges: np.ndarray, media_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    requested = np.rint(media_times * 16000.0).astype(np.int64)
    indices = np.searchsorted(right_edges, requested, side="right") - 1
    indices = np.clip(indices, 0, len(right_edges) - 1)
    lag = (requested - np.asarray(right_edges[indices], dtype=np.int64)) / 16000.0
    return indices, lag


def _run_vectors(
    embeddings: np.ndarray,
    valid: np.ndarray,
    right_edges: np.ndarray,
    run: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(item["media_time"]) for item in run["steps"]], dtype=np.float64)
    indices, lag = _causal_indices(right_edges, times)
    rows = np.asarray(embeddings[indices], dtype=np.float32)
    row_valid = np.asarray(valid[indices], dtype=bool)
    rows = _unit_rows(rows)
    row_valid &= np.isfinite(rows).all(axis=1) & (np.linalg.norm(rows, axis=1) > 1e-8)
    return rows, row_valid, lag


def _future_centroid_predict(
    run: dict[str, Any],
    short_rows: np.ndarray,
    short_valid: np.ndarray,
    *,
    long_rows: np.ndarray | None = None,
    long_valid: np.ndarray | None = None,
    short_weight: float = 1.0,
) -> tuple[list[str], dict[str, Any]]:
    steps = list(run["steps"])
    labels = sorted({str(item["truth"]) for item in steps if str(item["truth"])})
    short_centroids: list[np.ndarray] = []
    long_centroids: list[np.ndarray | None] = []
    for label in labels:
        mask = np.asarray(
            [
                str(item["truth"]) == label
                and bool(item["embedding_available"])
                and bool(short_valid[index])
                for index, item in enumerate(steps)
            ],
            dtype=bool,
        )
        centroid = _unit_rows(np.mean(short_rows[mask], axis=0, keepdims=True))[0]
        short_centroids.append(centroid)
        if long_rows is None or long_valid is None:
            long_centroids.append(None)
        else:
            long_mask = np.asarray(
                [
                    str(item["truth"]) == label
                    and bool(item["context_embedding_available"])
                    and bool(long_valid[index])
                    for index, item in enumerate(steps)
                ],
                dtype=bool,
            )
            if np.any(long_mask):
                long_centroids.append(
                    _unit_rows(np.mean(long_rows[long_mask], axis=0, keepdims=True))[0]
                )
            else:
                long_centroids.append(None)
    short_matrix = np.stack(short_centroids, axis=0)
    short_scores = short_rows @ short_matrix.T
    scores = short_scores.copy()
    if long_rows is not None and long_valid is not None:
        available = [item is not None for item in long_centroids]
        if any(available):
            long_matrix = np.stack(
                [item if item is not None else short_centroids[index] for index, item in enumerate(long_centroids)],
                axis=0,
            )
            long_scores = long_rows @ long_matrix.T
            weight = max(0.0, min(1.0, float(short_weight)))
            scores = weight * short_scores + (1.0 - weight) * long_scores
    predictions: list[str] = []
    correct = wrong = missing = outside = 0
    for index, item in enumerate(steps):
        truth = str(item["truth"])
        active = bool(item["speech"]) and not bool(item["release_signal"])
        eligible = bool(short_valid[index]) and bool(item["embedding_available"])
        if long_rows is not None and long_valid is not None:
            eligible = eligible and bool(long_valid[index])
        prediction = labels[int(np.argmax(scores[index]))] if active and eligible else ""
        predictions.append(prediction)
        if truth:
            if prediction == truth:
                correct += 1
            elif prediction:
                wrong += 1
            else:
                missing += 1
        elif prediction:
            outside += 1
    denominator = max(1, correct + wrong + missing)
    return predictions, {
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "outside": outside,
        "identity_accuracy": correct / denominator,
        "wrong_ratio": wrong / denominator,
        "missing_ratio": missing / denominator,
        "outside_per_truth_step": outside / denominator,
    }


def _screen_one(
    schedule: dict[str, Any], corpus_root: Path, provider: str, length_ms: int
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    latencies: list[float] = []
    for run in schedule["runs"]:
        video_id = str(run["video_id"])
        embeddings, valid, edges, metadata = _load_block(corpus_root, provider, video_id, length_ms)
        rows, row_valid, lag = _run_vectors(embeddings, valid, edges, run)
        _predictions, metrics = _future_centroid_predict(run, rows, row_valid)
        latencies.append(float(metadata.get("latency_ms_mean") or 0.0))
        runs.append(
            {
                "video_id": video_id,
                "run_id": str(run["run_id"]),
                **metrics,
                "lookup_mean_lag_seconds": float(np.mean(lag)),
                "lookup_max_lag_seconds": float(np.max(lag)),
            }
        )
    per_video: dict[str, list[float]] = {}
    for item in runs:
        per_video.setdefault(item["video_id"], []).append(float(item["identity_accuracy"]))
    video_scores = {key: mean(value) for key, value in sorted(per_video.items())}
    return {
        "candidate_id": f"{provider}@{length_ms:04d}ms",
        "provider": provider,
        "windows_ms": [length_ms],
        "future_label_leakage": True,
        "macro_probe_identity_accuracy": mean(video_scores.values()),
        "per_video_probe_identity_accuracy": video_scores,
        "mean_cached_generation_latency_ms": mean(latencies),
        "run_metrics": runs,
    }


def screen(args: argparse.Namespace) -> int:
    schedule = _read_json(args.schedule.resolve())
    corpus_root = args.corpus_root.resolve()
    providers = _providers(corpus_root)
    common_lengths: set[int] | None = None
    for provider in providers:
        lengths = set(_lengths(corpus_root, provider, TOP7[0]))
        common_lengths = lengths if common_lengths is None else common_lengths & lengths
    lengths = sorted(common_lengths or ())
    candidates: list[dict[str, Any]] = []
    total = len(providers) * len(lengths)
    completed = 0
    for provider in providers:
        for length_ms in lengths:
            candidates.append(_screen_one(schedule, corpus_root, provider, length_ms))
            completed += 1
            if completed % 12 == 0 or completed == total:
                print(f"screen_progress={completed}/{total} ({100.0 * completed / total:.1f}%)", flush=True)
    candidates.sort(key=lambda item: float(item["macro_probe_identity_accuracy"]), reverse=True)
    result = {
        "contract_id": CONTRACT_ID,
        "status": "INVALID_FUTURE_CENTROID_ORACLE_SCREEN",
        "production_promotion_eligible": False,
        "future_label_leakage": True,
        "candidate_selection_leakage": True,
        "model_inference_performed": False,
        "cache_root": str(corpus_root),
        "provider_count": len(providers),
        "window_count": len(lengths),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "top": candidates[:10]}, indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    schedule = _read_json(args.schedule.resolve())
    corpus_root = args.corpus_root.resolve()
    providers_and_windows: list[tuple[str, int, int | None, float]] = []
    for raw in args.candidate:
        parts = raw.split(":")
        if len(parts) not in {2, 4}:
            raise ValueError("Candidate must be PROVIDER:SHORT_MS or PROVIDER:SHORT_MS:LONG_MS:SHORT_WEIGHT")
        providers_and_windows.append(
            (
                parts[0],
                int(parts[1]),
                None if len(parts) == 2 else int(parts[2]),
                1.0 if len(parts) == 2 else float(parts[3]),
            )
        )
    output_candidates: list[dict[str, Any]] = []
    for provider, short_ms, long_ms, short_weight in providers_and_windows:
        run_predictions: list[dict[str, Any]] = []
        latencies: list[float] = []
        for run in schedule["runs"]:
            video_id = str(run["video_id"])
            short_embeddings, short_valid, edges, short_meta = _load_block(
                corpus_root, provider, video_id, short_ms
            )
            short_rows, short_row_valid, short_lag = _run_vectors(
                short_embeddings, short_valid, edges, run
            )
            long_rows = long_row_valid = None
            long_lag = np.asarray([], dtype=np.float64)
            latency = float(short_meta.get("latency_ms_mean") or 0.0)
            if long_ms is not None:
                long_embeddings, long_valid, long_edges, long_meta = _load_block(
                    corpus_root, provider, video_id, long_ms
                )
                long_rows, long_row_valid, long_lag = _run_vectors(
                    long_embeddings, long_valid, long_edges, run
                )
                latency += float(long_meta.get("latency_ms_mean") or 0.0)
            predictions, metrics = _future_centroid_predict(
                run,
                short_rows,
                short_row_valid,
                long_rows=long_rows,
                long_valid=long_row_valid,
                short_weight=short_weight,
            )
            latencies.append(latency)
            run_predictions.append(
                {
                    "video_id": video_id,
                    "run_id": str(run["run_id"]),
                    "predictions": predictions,
                    "proxy_metrics": metrics,
                    "short_lookup_mean_lag_seconds": float(np.mean(short_lag)),
                    "short_lookup_max_lag_seconds": float(np.max(short_lag)),
                    "long_lookup_mean_lag_seconds": float(np.mean(long_lag)) if len(long_lag) else None,
                    "long_lookup_max_lag_seconds": float(np.max(long_lag)) if len(long_lag) else None,
                }
            )
        candidate_id = (
            f"{provider}@{short_ms:04d}ms"
            if long_ms is None
            else f"{provider}@{short_ms:04d}+{long_ms:04d}ms_w{short_weight:.3f}"
        )
        output_candidates.append(
            {
                "candidate_id": candidate_id,
                "provider": provider,
                "windows_ms": [short_ms] if long_ms is None else [short_ms, long_ms],
                "short_weight": short_weight,
                "mean_cached_generation_latency_ms": mean(latencies),
                "runs": run_predictions,
            }
        )
    result = {
        "contract_id": CONTRACT_ID,
        "status": "INVALID_FUTURE_CENTROID_ORACLE_PREDICTIONS",
        "production_promotion_eligible": False,
        "future_label_leakage": True,
        "candidate_selection_leakage": True,
        "model_inference_performed": False,
        "candidates": output_candidates,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "candidates": [item["candidate_id"] for item in output_candidates]}, indent=2))
    return 0


def score(args: argparse.Namespace) -> int:
    from analyze_live_speaker_open_set_tracklets import (
        _identity_error_diagnostics,
        _load_base_config,
        _prepare_tape,
        _replay_with_tracklet_actions,
    )
    from prototype_live_speaker_segmental_dp_world_tapes import _emit_action
    from window.live_speaker_probe_scoring import read_canonical_segments

    parity = _read_json(args.parity_report.resolve())
    base_config = _load_base_config(args.base_artifact.resolve())
    prepared = [
        _prepare_tape(run, base_config)
        for run in parity.get("runs") or []
        if str(run.get("video_id") or "") in TOP7
    ]
    by_run = {(item.video_id, item.run_id): item for item in prepared}
    predictions = _read_json(args.predictions.resolve())
    candidate_reports: list[dict[str, Any]] = []
    for candidate in predictions["candidates"]:
        runs: list[dict[str, Any]] = []
        for prediction_run in candidate["runs"]:
            key = (str(prediction_run["video_id"]), str(prediction_run["run_id"]))
            tape = by_run[key]
            labels = list(prediction_run["predictions"])
            if len(labels) != len(tape.steps):
                raise ValueError(f"{key}: prediction/step mismatch {len(labels)} != {len(tape.steps)}")
            segments = read_canonical_segments(tape.canonical_path)
            oracle_labels = {
                speaker: f"O{index + 1}"
                for index, speaker in enumerate(sorted({str(row["speaker"]) for row in segments}))
            }
            actions: list[tuple[float, int, str, dict[str, Any]]] = []
            active = ""
            for step, prediction in zip(tape.steps, labels):
                chosen = oracle_labels.get(str(prediction), "")
                if not bool(step.payload.get("speech")) or bool(step.payload.get("release_signal")):
                    chosen = ""
                action, active = _emit_action(
                    tape,
                    step,
                    chosen,
                    "invalid_dense_future_centroid_representation_oracle",
                    active,
                )
                if action is not None:
                    actions.append(action)
            result = _replay_with_tracklet_actions(tape, actions)
            runs.append(
                {
                    "video_id": tape.video_id,
                    "run_id": tape.run_id,
                    "score": float(result["strict_browser_live_score"]),
                    "correct_live_speaker_coverage": float(result["correct_live_speaker_coverage"]),
                    "wrong_live_speech_ratio": float(result["wrong_live_speech_ratio"]),
                    "missing_live_speech_ratio": float(result["missing_live_speech_ratio"]),
                    "outside_speech_live_ratio": float(result["outside_speech_live_ratio"]),
                    "correct_live_precision_during_speech": float(result["correct_live_precision_during_speech"]),
                    "identity_errors": _identity_error_diagnostics(result),
                }
            )
        by_video: dict[str, list[float]] = {}
        for item in runs:
            by_video.setdefault(item["video_id"], []).append(float(item["score"]))
        per_video = {key: mean(value) for key, value in sorted(by_video.items())}
        candidate_reports.append(
            {
                "candidate_id": candidate["candidate_id"],
                "provider": candidate["provider"],
                "windows_ms": candidate["windows_ms"],
                "short_weight": candidate["short_weight"],
                "mean_cached_generation_latency_ms": candidate["mean_cached_generation_latency_ms"],
                "macro_score": mean(per_video.values()),
                "per_video": per_video,
                "runs": runs,
            }
        )
    candidate_reports.sort(key=lambda item: float(item["macro_score"]), reverse=True)
    result = {
        "contract_id": CONTRACT_ID,
        "status": "INVALID_FUTURE_CENTROID_REPRESENTATION_ORACLE_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "optimization_eligible": False,
        "future_label_leakage": True,
        "candidate_selection_leakage": True,
        "model_inference_performed": False,
        "reported_world_tape_speechbrain_0700_1500_reference": 0.8703726666666667,
        "candidates": candidate_reports,
        "best": candidate_reports[0],
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "candidates": [{"id": item["candidate_id"], "macro": item["macro_score"], "per_video": item["per_video"]} for item in candidate_reports]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-schedule")
    export_parser.add_argument("--parity-report", type=Path, required=True)
    export_parser.add_argument("--base-artifact", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(func=export_schedule)

    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--schedule", type=Path, required=True)
    screen_parser.add_argument("--corpus-root", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    screen_parser.set_defaults(func=screen)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--schedule", type=Path, required=True)
    predict_parser.add_argument("--corpus-root", type=Path, required=True)
    predict_parser.add_argument("--candidate", action="append", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.set_defaults(func=predict)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--parity-report", type=Path, required=True)
    score_parser.add_argument("--base-artifact", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(func=score)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
