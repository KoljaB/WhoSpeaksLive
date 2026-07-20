from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from optimize_live_speaker_replay import Dataset
from window.live_speaker_algorithm import ALGORITHM_ID, LiveSpeakerAlgorithmConfig
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_replay import replay_cached_live_windows_dual


OPTIMIZER_ID = "causal_live_speaker_round2_cached_v1"


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


def _provider_spec(spec: dict[str, Any]) -> str:
    return "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in spec["baseline"]["provider_weights"].items()
        if float(weight) > 0.0
    )


def _window_key(short_window: float, long_window: float, long_weight: float) -> tuple[float, float, float]:
    return (
        round(float(short_window), 3),
        round(float(long_window), 3),
        round(float(long_weight), 3),
    )


def _config_key(config: LiveSpeakerAlgorithmConfig) -> str:
    return _stable_json(asdict(config))


def _candidate_id(
    short_window: float,
    long_window: float,
    long_weight: float,
    config: LiveSpeakerAlgorithmConfig,
) -> str:
    return _stable_id({
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": ALGORITHM_ID,
        "short_window_seconds": round(float(short_window), 3),
        "long_window_seconds": round(float(long_window), 3),
        "long_weight": round(float(long_weight), 3),
        "algorithm_config": asdict(config),
    })


def _score_candidate(
    dataset: Dataset,
    videos: Iterable[str],
    short_window: float,
    long_window: float,
    long_weight: float,
    config: LiveSpeakerAlgorithmConfig,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for video_id in videos:
        inputs = dataset.video_inputs(video_id)
        decisions = replay_cached_live_windows_dual(
            dataset.block(video_id, short_window),
            dataset.block(video_id, long_window),
            inputs["profiles"],
            inputs["speech"],
            inputs["probes"],
            inputs["releases"],
            long_weight=long_weight,
            config=config,
        )
        per_video[video_id] = score_live_speaker_decisions(
            decisions,
            inputs["canonical"],
            inputs["profiles"],
        )
    return {
        "aggregate": aggregate_video_scores(per_video.values()),
        "per_video": per_video,
    }


def _quality(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = list(row["per_video"].values())
    mean_wrong = sum(float(item["wrong_live_speech_ratio"]) for item in metrics) / len(metrics)
    worst_wrong = max(float(item["wrong_live_speech_ratio"]) for item in metrics)
    return (
        float(row["aggregate"]["global_score"]),
        -mean_wrong,
        -worst_wrong,
        min(float(item["strict_browser_live_score"]) for item in metrics),
    )


def _coordinate_configs(config: LiveSpeakerAlgorithmConfig) -> list[LiveSpeakerAlgorithmConfig]:
    axes: dict[str, list[Any]] = {
        "min_similarity": [0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45],
        "min_margin": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14],
        "min_known_probability": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "ema_count": [1, 2, 3],
        "ema_alpha": [0.35, 0.45, 0.55, 0.70, 0.85, 1.0],
        "acquire_count": [1, 2],
        "switch_count": [1, 2, 3],
        "unknown_release_count": [1, 2, 3, 4],
        "silence_release_count": [1, 2, 3],
    }
    found: dict[str, LiveSpeakerAlgorithmConfig] = {}
    for name, values in axes.items():
        for value in values:
            candidate = replace(config, **{name: value})
            if candidate != config:
                found[_config_key(candidate)] = candidate
    return list(found.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-search cached dual-window live assignment around the live-verified champion."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--validation-tolerance", type=float, default=0.005)
    parser.add_argument("--wrong-ratio-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    search_videos = list(spec["split"]["search"])
    validation_videos = list(spec["split"]["validation"])
    videos = list(dict.fromkeys(search_videos + validation_videos))
    provider_spec = _provider_spec(spec)
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider_spec)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"

    base_short = float(champion["short_window_seconds"])
    base_long = float(champion["long_window_seconds"])
    base_weight = float(champion["long_weight"])
    base_config = LiveSpeakerAlgorithmConfig(**champion["algorithm_config"])
    baseline_first = _score_candidate(
        dataset, videos, base_short, base_long, base_weight, base_config
    )
    baseline_second = _score_candidate(
        dataset, videos, base_short, base_long, base_weight, base_config
    )
    if _stable_json(baseline_first) != _stable_json(baseline_second):
        raise RuntimeError("The round-two baseline did not reproduce exactly")
    _atomic_json(run_dir / "baseline_reproduction.json", {
        "status": "REPRODUCED_TWICE_IDENTICALLY",
        "short_window_seconds": base_short,
        "long_window_seconds": base_long,
        "long_weight": base_weight,
        "algorithm_config": asdict(base_config),
        **baseline_first,
    })

    evaluated: dict[str, dict[str, Any]] = {}
    phase_counts: dict[str, int] = {}

    def evaluate(
        short_window: float,
        long_window: float,
        long_weight: float,
        config: LiveSpeakerAlgorithmConfig,
        phase: str,
    ) -> dict[str, Any] | None:
        candidate_id = _candidate_id(
            short_window, long_window, long_weight, config
        )
        if candidate_id in evaluated:
            return evaluated[candidate_id]
        if time.monotonic() >= deadline:
            return None
        scored = _score_candidate(
            dataset,
            videos,
            short_window,
            long_window,
            long_weight,
            config,
        )
        per_video_score_delta = {
            video_id: round(
                float(scored["per_video"][video_id]["strict_browser_live_score"])
                - float(baseline_first["per_video"][video_id]["strict_browser_live_score"]),
                6,
            )
            for video_id in videos
        }
        per_video_wrong_delta = {
            video_id: round(
                float(scored["per_video"][video_id]["wrong_live_speech_ratio"])
                - float(baseline_first["per_video"][video_id]["wrong_live_speech_ratio"]),
                6,
            )
            for video_id in videos
        }
        validation_ok = min(
            per_video_score_delta[video_id] for video_id in validation_videos
        ) >= -float(args.validation_tolerance)
        wrong_ok = max(per_video_wrong_delta.values()) <= float(args.wrong_ratio_tolerance)
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "short_window_seconds": round(float(short_window), 3),
            "long_window_seconds": round(float(long_window), 3),
            "long_weight": round(float(long_weight), 3),
            "algorithm_config": asdict(config),
            **scored,
            "score_delta_vs_champion": round(
                float(scored["aggregate"]["global_score"])
                - float(baseline_first["aggregate"]["global_score"]),
                6,
            ),
            "per_video_score_delta_vs_champion": per_video_score_delta,
            "per_video_wrong_ratio_delta_vs_champion": per_video_wrong_delta,
            "validation_gate_passed": validation_ok,
            "wrong_ratio_gate_passed": wrong_ok,
            "eligible_for_fresh_verification": (
                float(scored["aggregate"]["global_score"])
                > float(baseline_first["aggregate"]["global_score"])
                and validation_ok
                and wrong_ok
            ),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        evaluated[candidate_id] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        _append_jsonl(trials_path, row)
        if len(evaluated) % 25 == 0:
            best = max(evaluated.values(), key=_quality)
            _atomic_json(run_dir / "progress.json", {
                "phase": phase,
                "evaluated_count": len(evaluated),
                "phase_counts": phase_counts,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "budget_seconds": int(args.budget_seconds),
                "progress_percent": round(
                    min(100.0, 100.0 * (time.monotonic() - started) / max(1, args.budget_seconds)),
                    2,
                ),
                "best_score": best["aggregate"]["global_score"],
                "best_candidate_id": best["candidate_id"],
            })
        return row

    evaluate(base_short, base_long, base_weight, base_config, "BASELINE")

    # Stage 1: fine window/weight search with the proven state machine.
    window_rows: list[dict[str, Any]] = []
    for short_window in (0.7, 0.8, 0.9, 1.0):
        for long_window in (2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0):
            if long_window <= short_window:
                continue
            for long_weight in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                row = evaluate(
                    short_window, long_window, long_weight, base_config, "WINDOW_FINE"
                )
                if row is None:
                    break
                window_rows.append(row)
            if time.monotonic() >= deadline:
                break
        if time.monotonic() >= deadline:
            break
    window_rows.sort(key=_quality, reverse=True)
    top_windows: list[tuple[float, float, float]] = []
    for row in window_rows:
        key = _window_key(
            row["short_window_seconds"], row["long_window_seconds"], row["long_weight"]
        )
        if key not in top_windows:
            top_windows.append(key)
        if len(top_windows) >= 12:
            break

    # Stage 2: tune one algorithm parameter at a time on the strongest windows.
    coordinate_rows: list[dict[str, Any]] = []
    for window in top_windows:
        for config in _coordinate_configs(base_config):
            row = evaluate(*window, config, "CONFIG_COORDINATE")
            if row is None:
                break
            coordinate_rows.append(row)
        if time.monotonic() >= deadline:
            break
    coordinate_rows.sort(key=_quality, reverse=True)

    # Stage 3: recombine the best independently discovered windows and configs.
    top_configs: list[LiveSpeakerAlgorithmConfig] = [base_config]
    seen_configs = {_config_key(base_config)}
    for row in coordinate_rows:
        config = LiveSpeakerAlgorithmConfig(**row["algorithm_config"])
        key = _config_key(config)
        if key not in seen_configs:
            seen_configs.add(key)
            top_configs.append(config)
        if len(top_configs) >= 20:
            break
    for window in top_windows:
        for config in top_configs:
            if evaluate(*window, config, "RECOMBINE") is None:
                break
        if time.monotonic() >= deadline:
            break

    # Stage 4: two greedy coordinate passes from the current best. This exposes
    # useful parameter interactions while keeping each accepted change ablatable.
    for pass_index in range(2):
        if time.monotonic() >= deadline:
            break
        incumbent = max(evaluated.values(), key=_quality)
        window = _window_key(
            incumbent["short_window_seconds"],
            incumbent["long_window_seconds"],
            incumbent["long_weight"],
        )
        config = LiveSpeakerAlgorithmConfig(**incumbent["algorithm_config"])
        for candidate in _coordinate_configs(config):
            if evaluate(*window, candidate, f"GREEDY_PASS_{pass_index + 1}") is None:
                break

    ranked = sorted(evaluated.values(), key=_quality, reverse=True)
    eligible = [row for row in ranked if row["eligible_for_fresh_verification"]]
    # Keep finalists diverse so fresh verification does not waste time on tiny
    # threshold variations that share the same window vectors.
    finalists: list[dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for row in eligible:
        config = row["algorithm_config"]
        signature = (
            row["short_window_seconds"],
            row["long_window_seconds"],
            row["long_weight"],
            config["min_similarity"],
            config["min_margin"],
            config["ema_count"],
            config["switch_count"],
            config["unknown_release_count"],
            config["silence_release_count"],
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        finalists.append(row)
        if len(finalists) >= 20:
            break

    payload = {
        "schema_version": 1,
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": ALGORITHM_ID,
        "provider": provider_spec,
        "search_videos": search_videos,
        "validation_videos": validation_videos,
        "known_holdout_excluded_from_search": ["JWS-qfR6K3w"],
        "baseline": {
            "short_window_seconds": base_short,
            "long_window_seconds": base_long,
            "long_weight": base_weight,
            "algorithm_config": asdict(base_config),
            **baseline_first,
        },
        "evaluated_count": len(evaluated),
        "phase_counts": phase_counts,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "eligible_count": len(eligible),
        "best": ranked[0],
        "top20": finalists,
    }
    _atomic_json(run_dir / "cached_sweep.json", payload)
    _atomic_json(run_dir / "progress.json", {
        "phase": "CACHED_SEARCH_COMPLETE",
        "evaluated_count": len(evaluated),
        "phase_counts": phase_counts,
        "elapsed_seconds": payload["elapsed_seconds"],
        "budget_seconds": int(args.budget_seconds),
        "progress_percent": 100.0,
        "best_score": ranked[0]["aggregate"]["global_score"],
        "best_candidate_id": ranked[0]["candidate_id"],
        "eligible_count": len(eligible),
    })
    print(json.dumps({
        "baseline_score": baseline_first["aggregate"]["global_score"],
        "evaluated_count": len(evaluated),
        "eligible_count": len(eligible),
        "best": {
            "score": ranked[0]["aggregate"]["global_score"],
            "score_delta": ranked[0]["score_delta_vs_champion"],
            "short_window_seconds": ranked[0]["short_window_seconds"],
            "long_window_seconds": ranked[0]["long_window_seconds"],
            "long_weight": ranked[0]["long_weight"],
            "algorithm_config": ranked[0]["algorithm_config"],
            "eligible_for_fresh_verification": ranked[0]["eligible_for_fresh_verification"],
            "per_video_score_delta": ranked[0]["per_video_score_delta_vs_champion"],
            "per_video_wrong_delta": ranked[0]["per_video_wrong_ratio_delta_vs_champion"],
        },
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
