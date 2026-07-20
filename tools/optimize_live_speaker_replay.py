from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embeddings.embedding_providers import parse_embedding_provider_stack_specs
from window.live_speaker_algorithm import ALGORITHM_ID, LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import (
    SCORER_ID,
    aggregate_video_scores,
    score_live_speaker_decisions,
)
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    load_cached_live_window_block,
    load_profile_events_jsonl,
    replay_cached_live_windows,
    stack_cached_live_window_blocks,
)


OPTIMIZER_ID = "causal_live_speaker_optimizer_v1"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _trace_hash(decisions: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(_stable_json(item.trace_record()) for item in decisions).encode("utf-8")
    ).hexdigest()


class Dataset:
    def __init__(self, corpus_root: Path, input_root: Path, provider_spec: str) -> None:
        self.corpus_root = corpus_root
        self.input_root = input_root
        self.provider_spec = provider_spec
        self.provider_specs = [
            (provider, float(weight))
            for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
            if float(weight) > 0.0
        ]
        if not self.provider_specs:
            raise ValueError("Provider stack is empty")
        self._blocks: dict[tuple[str, float], Any] = {}
        self._video_inputs: dict[str, dict[str, Any]] = {}

    def video_inputs(self, video_id: str) -> dict[str, Any]:
        cached = self._video_inputs.get(video_id)
        if cached is not None:
            return cached
        root = self.input_root / video_id
        value = {
            "canonical": read_canonical_segments(root / "canonical_diarization.json"),
            "profiles": load_profile_events_jsonl(root / "production_stack.profiles.jsonl"),
            "speech": np.load(root / "speech_gate.u1.npy", allow_pickle=False),
            "probes": np.load(root / "probe_schedule.u1.npy", allow_pickle=False),
            "releases": np.load(root / "release_gate.u1.npy", allow_pickle=False),
        }
        self._video_inputs[video_id] = value
        return value

    def block(self, video_id: str, window_seconds: float) -> Any:
        key = (video_id, round(float(window_seconds), 3))
        cached = self._blocks.get(key)
        if cached is not None:
            return cached
        blocks = [
            load_cached_live_window_block(
                self.corpus_root,
                provider,
                video_id,
                window_seconds,
            )
            for provider, _weight in self.provider_specs
        ]
        if len(blocks) == 1:
            block = blocks[0]
        else:
            block = stack_cached_live_window_blocks(
                blocks,
                [weight for _provider, weight in self.provider_specs],
                provider=self.provider_spec,
            )
        self._blocks[key] = block
        return block


def evaluate_candidate(
    dataset: Dataset,
    videos: list[str],
    window_seconds: float,
    config: LiveSpeakerAlgorithmConfig,
    *,
    include_traces: bool = False,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    trace_hashes: dict[str, str] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id)
        decisions = replay_cached_live_windows(
            dataset.block(video_id, window_seconds),
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            config=config,
        )
        per_video[video_id] = score_live_speaker_decisions(
            decisions,
            inputs["canonical"],
            inputs["profiles"],
        )
        if include_traces:
            trace_hashes[video_id] = _trace_hash(decisions)
    aggregate = aggregate_video_scores(per_video.values())
    return {
        "window_seconds": round(float(window_seconds), 3),
        "algorithm_config": asdict(config),
        "aggregate": aggregate,
        "per_video": per_video,
        "trace_hashes": trace_hashes,
    }


def _candidate_key(window_seconds: float, config: LiveSpeakerAlgorithmConfig) -> str:
    return _stable_id({
        "algorithm_id": ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "window_seconds": round(float(window_seconds), 3),
        "config": asdict(config),
    })


def _replace(config: LiveSpeakerAlgorithmConfig, **updates: Any) -> LiveSpeakerAlgorithmConfig:
    value = asdict(config)
    value.update(updates)
    return LiveSpeakerAlgorithmConfig(**value)


def coordinate_candidates(config: LiveSpeakerAlgorithmConfig) -> list[LiveSpeakerAlgorithmConfig]:
    axes: dict[str, list[Any]] = {
        "min_similarity": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "min_margin": [0.00, 0.03, 0.05, 0.08, 0.12],
        "min_known_probability": [0.40, 0.45, 0.50, 0.55, 0.60],
        "ema_count": [1, 2, 3, 4, 5],
        "ema_alpha": [0.35, 0.45, 0.55, 0.70, 0.85, 1.0],
        "acquire_count": [1, 2],
        "switch_count": [1, 2, 3],
        "unknown_release_count": [1, 2, 3, 4],
        "silence_release_count": [1, 2, 3],
    }
    candidates: list[LiveSpeakerAlgorithmConfig] = []
    seen: set[str] = set()
    for name, values in axes.items():
        for value in values:
            candidate = _replace(config, **{name: value})
            key = _stable_json(asdict(candidate))
            if key not in seen and candidate != config:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the live baseline, optimize it, and preserve every scored candidate."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=3300)
    parser.add_argument("--minimum-improvement", type=float, default=1e-6)
    parser.add_argument("--max-validation-regression", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    search_videos = list(spec["split"]["search"])
    validation_videos = list(spec["split"]["validation"])
    all_scored = list(dict.fromkeys(search_videos + validation_videos))
    provider_spec = "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in spec["baseline"]["provider_weights"].items()
        if float(weight) > 0.0
    )
    baseline_window = float(spec["baseline"]["probe_window_seconds"])
    baseline_config = LiveSpeakerAlgorithmConfig(**spec["baseline"]["algorithm_config"])
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for raw in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                completed[str(row["candidate_id"])] = row

    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider_spec)
    run_identity = _stable_id({
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "spec": spec,
        "corpus_root": str(args.corpus_root.resolve()),
        "input_root": str(args.input_root.resolve()),
        "provider": provider_spec,
    })
    _atomic_json(run_dir / "run.json", {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "run_identity": run_identity,
        "algorithm_id": ALGORITHM_ID,
        "scorer_id": SCORER_ID,
        "provider": provider_spec,
        "search_videos": search_videos,
        "validation_videos": validation_videos,
        "sealed_holdout_opened": False,
        "budget_seconds": int(args.budget_seconds),
    })

    baseline_first = evaluate_candidate(
        dataset, all_scored, baseline_window, baseline_config, include_traces=True
    )
    baseline_second = evaluate_candidate(
        dataset, all_scored, baseline_window, baseline_config, include_traces=True
    )
    baseline_identical = _stable_json(baseline_first) == _stable_json(baseline_second)
    baseline_payload = {
        "status": "REPRODUCED_TWICE_IDENTICALLY" if baseline_identical else "MISMATCH",
        "first": baseline_first,
        "second": baseline_second,
    }
    _atomic_json(run_dir / "baseline_reproduction.json", baseline_payload)
    if not baseline_identical:
        raise RuntimeError("Baseline did not reproduce exactly twice")

    incumbent = baseline_first
    incumbent_id = _candidate_key(baseline_window, baseline_config)
    accepted: list[dict[str, Any]] = []
    evaluated_count = 0

    def score_one(window: float, config: LiveSpeakerAlgorithmConfig, phase: str) -> dict[str, Any] | None:
        nonlocal evaluated_count
        candidate_id = _candidate_key(window, config)
        if candidate_id in completed:
            return completed[candidate_id]
        if _STOP or time.monotonic() >= deadline:
            return None
        result = evaluate_candidate(dataset, all_scored, window, config)
        search_scores = [result["per_video"][video] for video in search_videos]
        validation_scores = [result["per_video"][video] for video in validation_videos]
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "window_seconds": result["window_seconds"],
            "algorithm_config": result["algorithm_config"],
            "all_scored": result["aggregate"],
            "search": aggregate_video_scores(search_scores),
            "validation": aggregate_video_scores(validation_scores),
            "per_video": result["per_video"],
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        _append_jsonl(trials_path, row)
        completed[candidate_id] = row
        evaluated_count += 1
        _atomic_json(run_dir / "progress.json", {
            "phase": phase,
            "evaluated_count": evaluated_count,
            "elapsed_seconds": row["elapsed_seconds"],
            "best_score": incumbent["aggregate"]["global_score"],
            "best_candidate_id": incumbent_id,
        })
        return row

    # Phase 1: test every cached window length with unchanged production parameters.
    window_rows: list[dict[str, Any]] = []
    for window in spec["dense_corpus_expectation"]["window_lengths_seconds"]:
        row = score_one(float(window), baseline_config, "WINDOW_SCREEN")
        if row is None:
            break
        window_rows.append(row)
    window_rows.sort(key=lambda row: float(row["all_scored"]["global_score"]), reverse=True)

    # Phase 2: coordinate refinement around the strongest windows. Each accepted
    # change is explicit and can be fed to the fresh-LIVE verifier before promotion.
    top_windows = [float(row["window_seconds"]) for row in window_rows[:4]] or [baseline_window]
    current_window = top_windows[0]
    current_config = baseline_config
    for pass_index in range(4):
        if _STOP or time.monotonic() >= deadline:
            break
        pass_best: dict[str, Any] | None = None
        for window in top_windows:
            for config in coordinate_candidates(current_config):
                row = score_one(window, config, f"COORDINATE_PASS_{pass_index + 1}")
                if row is None:
                    break
                if pass_best is None or float(row["all_scored"]["global_score"]) > float(pass_best["all_scored"]["global_score"]):
                    pass_best = row
            if _STOP or time.monotonic() >= deadline:
                break
        if pass_best is None:
            break
        incumbent_score = float(incumbent["aggregate"]["global_score"])
        candidate_score = float(pass_best["all_scored"]["global_score"])
        baseline_validation = aggregate_video_scores(
            [incumbent["per_video"][video] for video in validation_videos]
        )["global_score"]
        candidate_validation = float(pass_best["validation"]["global_score"])
        if (
            candidate_score > incumbent_score + float(args.minimum_improvement)
            and candidate_validation >= float(baseline_validation) - float(args.max_validation_regression)
        ):
            accepted.append({
                "candidate_id": pass_best["candidate_id"],
                "score_before": incumbent_score,
                "score_after": candidate_score,
                "validation_before": baseline_validation,
                "validation_after": candidate_validation,
                "window_seconds": float(pass_best["window_seconds"]),
                "algorithm_config": dict(pass_best["algorithm_config"]),
                "requires_fresh_live_verification": True,
            })
            current_window = float(pass_best["window_seconds"])
            current_config = LiveSpeakerAlgorithmConfig(**pass_best["algorithm_config"])
            incumbent = {
                "window_seconds": current_window,
                "algorithm_config": pass_best["algorithm_config"],
                "aggregate": pass_best["all_scored"],
                "per_video": pass_best["per_video"],
                "trace_hashes": {},
            }
            incumbent_id = str(pass_best["candidate_id"])
        else:
            break

    improved = float(incumbent["aggregate"]["global_score"]) > float(baseline_first["aggregate"]["global_score"])
    champion = {
        "status": "CACHE_CHAMPION_PENDING_FRESH_LIVE" if improved else "NO_IMPROVEMENT",
        "candidate_id": incumbent_id if improved else None,
        "baseline_score": baseline_first["aggregate"]["global_score"],
        "candidate_score": incumbent["aggregate"]["global_score"],
        "score_delta": round(
            float(incumbent["aggregate"]["global_score"])
            - float(baseline_first["aggregate"]["global_score"]),
            6,
        ),
        "window_seconds": incumbent["window_seconds"],
        "algorithm_config": incumbent["algorithm_config"],
        "accepted_steps": accepted,
        "fresh_live_verified": False,
    }
    _atomic_json(run_dir / "champion.json", champion)
    _atomic_json(run_dir / "final_report.json", {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "run_identity": run_identity,
        "baseline_reproduced_twice": True,
        "baseline_score": baseline_first["aggregate"]["global_score"],
        "candidate_score": champion["candidate_score"],
        "score_delta": champion["score_delta"],
        "evaluated_count": evaluated_count,
        "accepted_step_count": len(accepted),
        "fresh_live_verification_required": improved,
        "sealed_holdout_opened": False,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    })
    print(json.dumps(champion, indent=2, ensure_ascii=False))
    return 0 if improved else 2


if __name__ == "__main__":
    raise SystemExit(main())
