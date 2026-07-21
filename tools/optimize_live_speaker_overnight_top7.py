from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
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
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    load_cached_live_window_block,
    load_profile_events_jsonl,
    replay_cached_live_windows,
    replay_cached_live_windows_dual,
    stack_cached_live_window_blocks,
)


OPTIMIZER_ID = "live_speaker_top7_single_objective_overnight_v1"
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


def _profile_name(spec: dict[str, Any], provider_spec: str) -> str:
    matches = [name for name, value in spec["profile_sets"].items() if str(value) == provider_spec]
    if not matches:
        raise KeyError(f"No prepared profile set for provider spec: {provider_spec}")
    return str(matches[0])


class Dataset:
    def __init__(self, corpus_root: Path, input_root: Path, provider_spec: str, profile_name: str) -> None:
        self.corpus_root = corpus_root
        self.input_root = input_root
        self.provider_spec = provider_spec
        self.profile_name = profile_name
        self.provider_specs = [
            (provider, float(weight))
            for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
            if float(weight) > 0.0
        ]
        if not self.provider_specs:
            raise ValueError("Provider stack is empty")
        self._blocks: dict[tuple[str, float], Any] = {}
        self._inputs: dict[tuple[str, float], dict[str, Any]] = {}

    def video_inputs(self, video_id: str, short_window: float) -> dict[str, Any]:
        key = (video_id, round(float(short_window), 3))
        cached = self._inputs.get(key)
        if cached is not None:
            return cached
        gate_root = self.input_root / "gate_sets" / f"{round(short_window * 1000):04d}ms" / video_id
        value = {
            "canonical": read_canonical_segments(
                self.input_root / "references" / video_id / "canonical_diarization.json"
            ),
            "profiles": load_profile_events_jsonl(
                self.input_root / "profiles" / self.profile_name / video_id
                / "production_stack.profiles.jsonl"
            ),
            "speech": np.load(gate_root / "speech_gate.u1.npy", allow_pickle=False),
            "probes": np.load(gate_root / "probe_schedule.u1.npy", allow_pickle=False),
            "releases": np.load(gate_root / "release_gate.u1.npy", allow_pickle=False),
        }
        self._inputs[key] = value
        return value

    def block(self, video_id: str, window_seconds: float) -> Any:
        key = (video_id, round(float(window_seconds), 3))
        cached = self._blocks.get(key)
        if cached is not None:
            return cached
        blocks = [
            load_cached_live_window_block(self.corpus_root, provider, video_id, window_seconds)
            for provider, _weight in self.provider_specs
        ]
        block = blocks[0] if len(blocks) == 1 else stack_cached_live_window_blocks(
            blocks,
            [weight for _provider, weight in self.provider_specs],
            provider=self.provider_spec,
        )
        self._blocks[key] = block
        return block


def _description(
    provider_spec: str,
    profile_name: str,
    short_window: float,
    config: LiveSpeakerAlgorithmConfig,
    *,
    long_window: float | None = None,
    long_weight: float = 0.0,
) -> dict[str, Any]:
    return {
        "provider_spec": provider_spec,
        "profile_name": profile_name,
        "short_window_seconds": round(float(short_window), 3),
        "long_window_seconds": None if long_window is None else round(float(long_window), 3),
        "long_weight": round(float(long_weight), 4) if long_window is not None else 0.0,
        "algorithm_config": asdict(config),
    }


def _candidate_id(description: dict[str, Any]) -> str:
    return _stable_id({
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        **description,
    })


def evaluate_candidate(
    dataset: Dataset,
    videos: Iterable[str],
    description: dict[str, Any],
) -> dict[str, Any]:
    short_window = float(description["short_window_seconds"])
    long_window = description.get("long_window_seconds")
    config = LiveSpeakerAlgorithmConfig(**description["algorithm_config"])
    per_video: dict[str, Any] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id, short_window)
        if long_window is None:
            decisions = replay_cached_live_windows(
                dataset.block(video_id, short_window),
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                config=config,
            )
        else:
            decisions = replay_cached_live_windows_dual(
                dataset.block(video_id, short_window),
                dataset.block(video_id, float(long_window)),
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                long_weight=float(description["long_weight"]),
                config=config,
            )
        per_video[video_id] = score_live_speaker_decisions(
            decisions, inputs["canonical"], inputs["profiles"]
        )
    aggregate = aggregate_video_scores_primary_v2(per_video.values())
    if not math.isfinite(float(aggregate["primary_score"])):
        raise RuntimeError("Non-finite primary score")
    return {"aggregate": aggregate, "per_video": per_video}


def _replace_config(config: LiveSpeakerAlgorithmConfig, **updates: Any) -> LiveSpeakerAlgorithmConfig:
    value = asdict(config)
    value.update(updates)
    return LiveSpeakerAlgorithmConfig(**value)


def coordinate_candidates(config: LiveSpeakerAlgorithmConfig) -> list[LiveSpeakerAlgorithmConfig]:
    axes: dict[str, list[Any]] = {
        "min_similarity": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        "min_margin": [0.00, 0.03, 0.05, 0.08, 0.12, 0.16],
        "min_known_probability": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "ema_count": [1, 2, 3, 4, 5],
        "ema_alpha": [0.35, 0.45, 0.55, 0.70, 0.85, 1.0],
        "acquire_count": [1, 2, 3],
        "switch_count": [1, 2, 3],
        "unknown_release_count": [1, 2, 3, 4],
        "silence_release_count": [1, 2, 3, 4],
    }
    candidates: list[LiveSpeakerAlgorithmConfig] = []
    seen: set[str] = set()
    for name, values in axes.items():
        for value in values:
            candidate = _replace_config(config, **{name: value})
            identity = _stable_json(asdict(candidate))
            if candidate != config and identity not in seen:
                seen.add(identity)
                candidates.append(candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable top-seven live-speaker optimization with one promotion score."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=28_800)
    parser.add_argument("--minimum-improvement", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    input_root = args.input_root.resolve()
    manifest = json.loads((input_root / "preparation_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Preparation manifest is not complete")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for raw in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                row = json.loads(raw)
                completed[str(row["candidate_id"])] = row

    datasets: dict[str, Dataset] = {}

    def dataset_for(provider_spec: str) -> Dataset:
        cached = datasets.get(provider_spec)
        if cached is None:
            cached = Dataset(
                args.corpus_root.resolve(), input_root, provider_spec,
                _profile_name(spec, provider_spec),
            )
            datasets[provider_spec] = cached
        return cached

    def release_dataset(provider_spec: str) -> None:
        """Close provider-local memmaps before moving to the next provider."""

        datasets.pop(provider_spec, None)
        gc.collect()

    baseline_spec = str(spec["baseline"]["provider_spec"])
    baseline_config = LiveSpeakerAlgorithmConfig(**spec["baseline"]["algorithm_config"])
    baseline_description = _description(
        baseline_spec,
        _profile_name(spec, baseline_spec),
        float(spec["baseline"]["short_window_seconds"]),
        baseline_config,
        long_window=float(spec["baseline"]["long_window_seconds"]),
        long_weight=float(spec["baseline"]["long_weight"]),
    )
    first = evaluate_candidate(dataset_for(baseline_spec), videos, baseline_description)
    second = evaluate_candidate(dataset_for(baseline_spec), videos, baseline_description)
    baseline_identical = _stable_json(first) == _stable_json(second)
    _atomic_json(run_dir / "baseline_reproduction.json", {
        "status": "REPRODUCED_TWICE_IDENTICALLY" if baseline_identical else "MISMATCH",
        "description": baseline_description,
        "first": first,
        "second": second,
    })
    if not baseline_identical:
        raise RuntimeError("Champion baseline did not reproduce exactly twice")
    release_dataset(baseline_spec)

    baseline_row = {
        "candidate_id": _candidate_id(baseline_description),
        "phase": "BASELINE",
        **baseline_description,
        **first,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    incumbent = baseline_row
    for row in completed.values():
        if float(row["aggregate"]["primary_score"]) > float(incumbent["aggregate"]["primary_score"]):
            incumbent = row
    queue_path = run_dir / "fresh_verification_queue.json"
    milestones: list[dict[str, Any]] = []
    if args.resume and queue_path.is_file():
        prior_queue = json.loads(queue_path.read_text(encoding="utf-8-sig"))
        milestones = [dict(row) for row in prior_queue.get("milestones") or []]
    last_queued_score = max(
        [float(first["aggregate"]["primary_score"])]
        + [float(row["score"]) for row in milestones]
    )
    evaluated_this_run = 0

    def write_state(phase: str, active: str = "") -> None:
        score = float(incumbent["aggregate"]["primary_score"])
        _atomic_json(run_dir / "progress.json", {
            "schema_version": 1,
            "optimizer_id": OPTIMIZER_ID,
            "status": "interrupted" if _STOP else "running",
            "phase": phase,
            "active": active,
            "completed_candidate_count": len(completed),
            "evaluated_this_run": evaluated_this_run,
            "baseline_score": first["aggregate"]["primary_score"],
            "best_score": score,
            "score_delta": round(score - float(first["aggregate"]["primary_score"]), 6),
            "best_candidate_id": incumbent["candidate_id"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "budget_seconds": int(args.budget_seconds),
        })
        _atomic_json(run_dir / "champion.json", {
            "status": "CACHE_CHAMPION_PENDING_FRESH_LIVE",
            "selection_policy": "primary_score_only_no_per_video_vetoes",
            "baseline_score": first["aggregate"]["primary_score"],
            "candidate_id": incumbent["candidate_id"],
            "candidate_score": score,
            "score_delta": round(score - float(first["aggregate"]["primary_score"]), 6),
            "description": {
                key: incumbent[key] for key in (
                    "provider_spec", "profile_name", "short_window_seconds",
                    "long_window_seconds", "long_weight", "algorithm_config"
                )
            },
            "aggregate": incumbent["aggregate"],
            "per_video": incumbent["per_video"],
            "fresh_live_verified": False,
        })
        _atomic_json(queue_path, {
            "schema_version": 1,
            "policy": "verify meaningful milestones and the final cache champion with fresh embeddings",
            "verifier": "tools/verify_live_speaker_top7_fresh.py",
            "milestones": milestones,
            "final_candidate_id": incumbent["candidate_id"],
        })

    def score_one(description: dict[str, Any], phase: str) -> dict[str, Any] | None:
        nonlocal incumbent, evaluated_this_run, last_queued_score
        candidate_id = _candidate_id(description)
        row = completed.get(candidate_id)
        if row is None:
            if _STOP or time.monotonic() >= deadline:
                return None
            write_state(phase, candidate_id)
            result = evaluate_candidate(dataset_for(str(description["provider_spec"])), videos, description)
            row = {
                "candidate_id": candidate_id,
                "phase": phase,
                **description,
                **result,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            _append_jsonl(trials_path, row)
            completed[candidate_id] = row
            evaluated_this_run += 1
        score = float(row["aggregate"]["primary_score"])
        if score > float(incumbent["aggregate"]["primary_score"]) + float(args.minimum_improvement):
            incumbent = row
            if score >= last_queued_score + float(spec["fresh_live"]["milestone_score_delta"]):
                milestones.append({
                    "candidate_id": candidate_id,
                    "score": score,
                    "phase": phase,
                    "description": description,
                })
                last_queued_score = score
        write_state(phase, candidate_id)
        return row

    _atomic_json(run_dir / "run.json", {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "promotion_policy": "accept_any_deterministic_primary_score_improvement",
        "per_video_scores_are_diagnostics_only": True,
        "maximum_fresh_windows_per_probe": 2,
        "videos": videos,
        "spec": str(args.spec.resolve()),
        "corpus_root": str(args.corpus_root.resolve()),
        "input_root": str(input_root),
        "budget_seconds": int(args.budget_seconds),
        "resume": bool(args.resume),
        "smoke": bool(args.smoke),
    })
    write_state("BASELINE")

    providers = [str(value) for value in spec["providers"]] + [baseline_spec]
    providers = list(dict.fromkeys(providers))
    short_windows = [float(value) for value in spec["search"]["short_windows_seconds"]]
    if args.smoke:
        providers = [baseline_spec, providers[0]]
        short_windows = [0.8]

    atlas_rows: list[dict[str, Any]] = []
    for provider_spec in providers:
        try:
            for short_window in short_windows:
                row = score_one(
                    _description(
                        provider_spec, _profile_name(spec, provider_spec), short_window, baseline_config
                    ),
                    "PROVIDER_SHORT_WINDOW_ATLAS",
                )
                if row is None:
                    break
                atlas_rows.append(row)
        finally:
            release_dataset(provider_spec)
        if _STOP or time.monotonic() >= deadline:
            break

    best_by_provider: dict[str, dict[str, Any]] = {}
    atlas_source = atlas_rows + [
        row for row in completed.values() if row.get("phase") == "PROVIDER_SHORT_WINDOW_ATLAS"
    ]
    for row in atlas_source:
        provider = str(row["provider_spec"])
        old = best_by_provider.get(provider)
        if old is None or float(row["aggregate"]["primary_score"]) > float(old["aggregate"]["primary_score"]):
            best_by_provider[provider] = row
    ranked_providers = sorted(
        best_by_provider,
        key=lambda provider: float(best_by_provider[provider]["aggregate"]["primary_score"]),
        reverse=True,
    )[: int(spec["search"]["dual_provider_finalists"])]
    if baseline_spec not in ranked_providers:
        ranked_providers.append(baseline_spec)

    dual_shorts = [float(value) for value in spec["search"]["dual_short_windows_seconds"]]
    long_windows = [float(value) for value in spec["search"]["long_windows_seconds"]]
    long_weights = [float(value) for value in spec["search"]["long_weights"]]
    if args.smoke:
        ranked_providers = [baseline_spec]
        dual_shorts = [0.8]
        long_windows = [2.8]
        long_weights = [0.25]
    for provider_spec in ranked_providers:
        provider_seed = best_by_provider.get(provider_spec, baseline_row)
        config = LiveSpeakerAlgorithmConfig(**provider_seed["algorithm_config"])
        try:
            for short_window in dual_shorts:
                for long_window in long_windows:
                    if long_window <= short_window:
                        continue
                    for long_weight in long_weights:
                        if score_one(
                            _description(
                                provider_spec, _profile_name(spec, provider_spec), short_window, config,
                                long_window=long_window, long_weight=long_weight,
                            ),
                            "DUAL_WINDOW_COARSE",
                        ) is None:
                            break
                    if _STOP or time.monotonic() >= deadline:
                        break
                if _STOP or time.monotonic() >= deadline:
                    break
        finally:
            release_dataset(provider_spec)
        if _STOP or time.monotonic() >= deadline:
            break

    if not _STOP and time.monotonic() < deadline and not args.smoke:
        seed = incumbent
        long_value = seed.get("long_window_seconds")
        if long_value is not None:
            refined_longs = sorted({
                round(max(0.7, min(3.0, float(long_value) + delta)), 1)
                for delta in (-0.1, 0.0, 0.1)
            })
            refined_weights = sorted({
                round(max(0.05, min(0.95, float(seed["long_weight"]) + delta)), 2)
                for delta in (-0.10, -0.05, 0.0, 0.05, 0.10)
            })
            for long_window in refined_longs:
                if long_window <= float(seed["short_window_seconds"]):
                    continue
                for long_weight in refined_weights:
                    if score_one(
                        _description(
                            str(seed["provider_spec"]), str(seed["profile_name"]),
                            float(seed["short_window_seconds"]),
                            LiveSpeakerAlgorithmConfig(**seed["algorithm_config"]),
                            long_window=long_window, long_weight=long_weight,
                        ),
                        "DUAL_WINDOW_LOCAL_REFINE",
                    ) is None:
                        break

    for pass_index in range(int(spec["search"]["coordinate_passes"])):
        if _STOP or time.monotonic() >= deadline or args.smoke:
            break
        seed = incumbent
        seed_score = float(seed["aggregate"]["primary_score"])
        for config in coordinate_candidates(LiveSpeakerAlgorithmConfig(**seed["algorithm_config"])):
            description = _description(
                str(seed["provider_spec"]), str(seed["profile_name"]),
                float(seed["short_window_seconds"]), config,
                long_window=(
                    None if seed.get("long_window_seconds") is None
                    else float(seed["long_window_seconds"])
                ),
                long_weight=float(seed.get("long_weight") or 0.0),
            )
            if score_one(description, f"COORDINATE_PASS_{pass_index + 1}") is None:
                break
        if float(incumbent["aggregate"]["primary_score"]) <= seed_score + float(args.minimum_improvement):
            break

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    _atomic_json(run_dir / "final_report.json", {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "status": progress["status"],
        "baseline_reproduced_twice": True,
        "baseline_score": first["aggregate"]["primary_score"],
        "champion_score": incumbent["aggregate"]["primary_score"],
        "score_delta": round(
            float(incumbent["aggregate"]["primary_score"])
            - float(first["aggregate"]["primary_score"]), 6
        ),
        "candidate_count": len(completed),
        "evaluated_this_run": evaluated_this_run,
        "fresh_live_verification_required": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps({
        "status": progress["status"],
        "baseline_score": first["aggregate"]["primary_score"],
        "champion_score": incumbent["aggregate"]["primary_score"],
        "score_delta": round(
            float(incumbent["aggregate"]["primary_score"])
            - float(first["aggregate"]["primary_score"]), 6
        ),
        "candidate_id": incumbent["candidate_id"],
        "candidate_count": len(completed),
        "champion_path": str(run_dir / "champion.json"),
    }, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
