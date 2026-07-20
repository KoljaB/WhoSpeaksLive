"""Analyze causal temporal dynamics of the existing 0.8/2.8 s live embeddings.

This is an offline research tool.  It is deliberately restricted to the
already-opened v1-v4 cohorts, requests no embeddings, and does not import or
modify production controller code beyond replaying the existing algorithms.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_replay import Dataset
import sweep_live_speaker_hybrid as sweep
import sweep_live_speaker_hybrid_profile_quality_meta as meta
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_hybrid import replay_hybrid_decisions
from window.live_speaker_replay import replay_cached_live_windows_dual


ANALYZER_ID = "opened_v1_v4_raw_embedding_dynamics_v1"
EXPECTED_PROVIDER = "pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5"
EXPECTED_WINDOWS = (0.8, 2.8)
EXPECTED_SOURCES: dict[str, tuple[str, ...]] = {
    "v1": ("Dd7FixvoKBw", "DsyfYJ5Ou3g", "20v1OxUXcQY", "JWS-qfR6K3w"),
    "v2": ("e3h6es6zh1c", "1NBVQB-Srpw", "F2-2RBi1qzY", "vIfGgDnmBXg", "ZY0DG8rUnCA"),
    "v3": ("pD4IdQTmneI", "k1tsGGz-Qw0", "aHGd6LqAVzw"),
    "v4": ("blcKeLDDzSM", "KdOXM3I_5hk", "acbnyagl8jo"),
}
# These IDs are sealed and must never enter this analysis, even before path construction.
FORBIDDEN_UNOPENED_IDS = frozenset({"bPpcfH_HHH8", "WNZn37Uc700", "oFBuCp19L7M"})


@dataclass(frozen=True)
class DatasetSource:
    label: str
    corpus_root: Path
    input_root: Path
    video_ids: tuple[str, ...]


@dataclass
class PreparedVideo:
    source: str
    inputs: dict[str, Any]
    short: Any
    long: Any
    baseline: list[Any]
    run018: list[Any]
    meta: list[Any]


@dataclass(frozen=True)
class SelectorConfig:
    threshold: float
    statistic: str = "current"
    history_size: int = 1
    history_max_gap_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.statistic not in {"current", "median", "min"}:
            raise ValueError("selector statistic must be current, median, or min")
        if int(self.history_size) not in {1, 2}:
            raise ValueError("selector history must contain one or two scheduled probes")
        if self.statistic == "current" and int(self.history_size) != 1:
            raise ValueError("current selector must use a one-probe history")
        if self.statistic != "current" and int(self.history_size) != 2:
            raise ValueError("median/min selector must use a two-probe history")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        action="append",
        required=True,
        metavar="LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2",
    )
    parser.add_argument("--locked-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-max-gap-seconds", type=float, default=1.5)
    parser.add_argument("--diagnostics-only", action="store_true")
    return parser.parse_args(argv)


def _split_source(raw: str) -> tuple[str, str, str, tuple[str, ...]]:
    try:
        label_and_corpus, input_root, raw_ids = raw.split("::", 2)
        label, corpus_root = label_and_corpus.split("=", 1)
    except ValueError as exc:
        raise ValueError("dataset source must be LABEL=CORPUS_ROOT::INPUT_ROOT::ID1,ID2") from exc
    ids = tuple(value.strip() for value in raw_ids.split(",") if value.strip())
    return label.strip(), corpus_root.strip(), input_root.strip(), ids


def _validate_sources(raw_sources: Sequence[str]) -> list[DatasetSource]:
    raw = [_split_source(value) for value in raw_sources]
    all_ids = [video_id for _label, _corpus, _inputs, ids in raw for video_id in ids]
    forbidden = sorted(FORBIDDEN_UNOPENED_IDS.intersection(all_ids))
    if forbidden:
        raise ValueError("sealed unopened video IDs are forbidden: " + ", ".join(forbidden))
    labels = [value[0] for value in raw]
    if len(raw) != 4 or set(labels) != set(EXPECTED_SOURCES) or len(labels) != len(set(labels)):
        raise ValueError("dataset sources must be exactly v1, v2, v3, and v4")
    by_label = {value[0]: value for value in raw}
    for label, expected in EXPECTED_SOURCES.items():
        if by_label[label][3] != expected:
            raise ValueError(f"{label} IDs must be the exact opened cohort in order")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("video IDs may occur only once")
    # Paths are intentionally constructed only after the cohort seal checks.
    return [
        DatasetSource(label, Path(by_label[label][1]), Path(by_label[label][2]), by_label[label][3])
        for label in EXPECTED_SOURCES
    ]


def _prepare(sources: Sequence[DatasetSource], locked_run_dir: Path) -> dict[str, PreparedVideo]:
    locked = round2._load_run018(locked_run_dir)
    run018_candidate, meta_candidate = meta._fixed_candidates(locked.config)
    prepared: dict[str, PreparedVideo] = {}
    for source in sources:
        dataset = Dataset(source.corpus_root.resolve(), source.input_root.resolve(), EXPECTED_PROVIDER)
        for video_id in source.video_ids:
            inputs = dataset.video_inputs(video_id)
            short = dataset.block(video_id, EXPECTED_WINDOWS[0])
            long = dataset.block(video_id, EXPECTED_WINDOWS[1])
            baseline = replay_cached_live_windows_dual(
                short,
                long,
                inputs["profiles"],
                inputs["speech"],
                inputs["probes"],
                inputs["releases"],
                long_weight=0.25,
                config=sweep.BASELINE_CONFIG,
            )
            steps = sweep._build_steps(short, long, inputs)
            run018 = replay_hybrid_decisions(
                baseline, steps, inputs["profiles"], config=run018_candidate.config
            )
            meta_decisions = replay_hybrid_decisions(
                baseline, steps, inputs["profiles"], config=meta_candidate.config
            )
            prepared[video_id] = PreparedVideo(
                source.label, inputs, short, long, baseline, run018, meta_decisions
            )
    return prepared


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding is not finite and nonzero")
    return value / norm


def _truth_at(canonical: Sequence[dict[str, Any]], media_time: float) -> str | None:
    # The scorer holds each tick until the next 0.2 s observation.  Sampling the
    # midpoint is a compact causal-diagnostic approximation of that slice.
    point = float(media_time) + 0.1
    active = {
        str(segment["speaker"])
        for segment in canonical
        if float(segment["start"]) <= point < float(segment["end"])
    }
    return next(iter(active)) if len(active) == 1 else None


def _is_correct(decision: Any, truth: str | None, speaker_map: dict[str, str]) -> bool:
    if truth is None:
        return decision.visible_speaker is None
    visible = str(decision.visible_speaker or "")
    return bool(visible and speaker_map.get(visible) == truth)


def _quantiles(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": round(float(np.mean(array)), 6),
        "p05": round(float(np.quantile(array, 0.05)), 6),
        "p25": round(float(np.quantile(array, 0.25)), 6),
        "p50": round(float(np.quantile(array, 0.50)), 6),
        "p75": round(float(np.quantile(array, 0.75)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
    }


def _feature_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = ("short_prev_cos", "long_prev_cos", "short_long_cos", "long_minus_short")
    return {name: _quantiles(float(row[name]) for row in rows) for name in names}


def _diagnose(prepared: dict[str, PreparedVideo], history_max_gap_seconds: float) -> dict[str, Any]:
    disagreement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dynamics_by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_video_counts: dict[str, dict[str, int]] = {}
    all_rows: list[dict[str, Any]] = []

    for video_id, value in prepared.items():
        baseline_score = score_live_speaker_decisions(
            value.baseline, value.inputs["canonical"], value.inputs["profiles"]
        )
        run018_score = score_live_speaker_decisions(
            value.run018, value.inputs["canonical"], value.inputs["profiles"]
        )
        baseline_map = dict(baseline_score.get("speaker_map") or {})
        run018_map = dict(run018_score.get("speaker_map") or {})
        previous: tuple[float, np.ndarray, np.ndarray, str | None] | None = None
        counts: dict[str, int] = defaultdict(int)

        for index, media_time in enumerate(value.short.media_times):
            if not bool(value.inputs["probes"][index]):
                continue
            if not (bool(value.short.valid[index]) and bool(value.long.valid[index])):
                continue
            current_short = _unit(value.short.embeddings[index])
            current_long = _unit(value.long.embeddings[index])
            truth = _truth_at(value.inputs["canonical"], float(media_time))
            if previous is not None and float(media_time) - previous[0] <= history_max_gap_seconds + 1e-9:
                previous_time, previous_short, previous_long, previous_truth = previous
                row = {
                    "video_id": video_id,
                    "media_time": float(media_time),
                    "gap_seconds": float(media_time) - previous_time,
                    "short_prev_cos": float(np.dot(current_short, previous_short)),
                    "long_prev_cos": float(np.dot(current_long, previous_long)),
                    "short_long_cos": float(np.dot(current_short, current_long)),
                }
                row["long_minus_short"] = row["long_prev_cos"] - row["short_prev_cos"]
                if truth is None and previous_truth is None:
                    truth_kind = "off"
                elif truth is not None and previous_truth is None:
                    truth_kind = "onset"
                elif truth is None and previous_truth is not None:
                    truth_kind = "offset"
                elif truth == previous_truth:
                    truth_kind = "hold"
                else:
                    truth_kind = "switch"
                dynamics_by_truth[truth_kind].append(row)
                all_rows.append(row)

                baseline_decision = value.baseline[index]
                run018_decision = value.run018[index]
                if baseline_decision.visible_speaker != run018_decision.visible_speaker:
                    baseline_correct = _is_correct(baseline_decision, truth, baseline_map)
                    run018_correct = _is_correct(run018_decision, truth, run018_map)
                    if baseline_correct and run018_correct:
                        category = "both_correct"
                    elif baseline_correct:
                        category = "baseline_only_correct"
                    elif run018_correct:
                        category = "run018_only_correct"
                    else:
                        category = "both_wrong"
                    disagreement[category].append(row)
                    disagreement["baseline_correct" if baseline_correct else "baseline_wrong"].append(row)
                    disagreement["run018_correct" if run018_correct else "run018_wrong"].append(row)
                    counts[category] += 1
            previous = (float(media_time), current_short, current_long, truth)
        per_video_counts[video_id] = dict(counts)

    return {
        "scheduled_dual_probe_pairs_with_history": len(all_rows),
        "truth_dynamics": {
            key: _feature_summary(rows) for key, rows in sorted(dynamics_by_truth.items())
        },
        "baseline_run018_disagreements": {
            key: _feature_summary(rows) for key, rows in sorted(disagreement.items())
        },
        "disagreement_counts_by_video": per_video_counts,
    }


def _compact_scores(prepared: dict[str, PreparedVideo]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("baseline", "run018", "meta"):
        result[name] = {
            video_id: {
                "score": float(score["strict_browser_live_score"]),
                "wrong": float(score["wrong_live_speech_ratio"]),
            }
            for video_id, value in prepared.items()
            for score in [
                score_live_speaker_decisions(
                    getattr(value, name), value.inputs["canonical"], value.inputs["profiles"]
                )
            ]
        }
    return result


def _selector_value(history: deque[float], config: SelectorConfig) -> float:
    values = list(history)[-int(config.history_size):]
    if not values:
        raise ValueError("selector history is empty")
    if config.statistic == "current":
        return float(values[-1])
    if config.statistic == "min":
        return float(min(values))
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _replay_selector(
    value: PreparedVideo,
    config: SelectorConfig,
) -> tuple[list[Any], dict[str, int]]:
    """Select the recall or precision expert from raw fast/slow coherence."""

    results: list[Any] = []
    coherence_history: deque[float] = deque(maxlen=2)
    last_probe_time: float | None = None
    selected_expert = "run018"
    counts: dict[str, int] = defaultdict(int)
    for index, media_time in enumerate(value.short.media_times):
        baseline = value.baseline[index]
        precision = value.run018[index]
        scheduled = bool(value.inputs["probes"][index])
        valid = bool(value.short.valid[index]) and bool(value.long.valid[index])
        if bool(value.inputs["releases"][index]):
            coherence_history.clear()
            last_probe_time = None
            selected_expert = "run018"
        if scheduled and valid:
            if (
                last_probe_time is None
                or float(media_time) - last_probe_time
                > float(config.history_max_gap_seconds) + 1e-9
            ):
                coherence_history.clear()
            short = _unit(value.short.embeddings[index])
            long = _unit(value.long.embeddings[index])
            coherence_history.append(float(np.dot(short, long)))
            last_probe_time = float(media_time)
            if baseline.visible_speaker != precision.visible_speaker:
                selector = _selector_value(coherence_history, config)
                selected_expert = "baseline" if selector >= float(config.threshold) else "run018"
                counts["scheduled_disagreements"] += 1
                counts[f"{selected_expert}_selected"] += 1
        if baseline.visible_speaker == precision.visible_speaker:
            chosen = precision
            counts["agreement_ticks"] += 1
        else:
            chosen = baseline if selected_expert == "baseline" else precision
            counts[f"{selected_expert}_output_ticks"] += 1
        if chosen.visible_speaker != baseline.visible_speaker:
            counts["output_ticks_different_from_baseline"] += 1
        if chosen.visible_speaker != precision.visible_speaker:
            counts["output_ticks_different_from_run018"] += 1
        diagnostics = dict(chosen.diagnostics)
        diagnostics.update({
            "offline_embedding_dynamics_selector": asdict(config),
            "offline_selected_expert": selected_expert,
        })
        results.append(replace(
            chosen,
            action="offline_temporal_select",
            reason=f"raw_short_long_coherence_{selected_expert}",
            diagnostics=diagnostics,
        ))
    return results, dict(counts)


def _score_selector(
    prepared: dict[str, PreparedVideo],
    config: SelectorConfig,
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    interventions: dict[str, dict[str, int]] = {}
    for video_id, value in prepared.items():
        decisions, counts = _replay_selector(value, config)
        scores[video_id] = score_live_speaker_decisions(
            decisions, value.inputs["canonical"], value.inputs["profiles"]
        )
        interventions[video_id] = counts
    return {
        "config": asdict(config),
        "aggregate": aggregate_video_scores(scores.values()),
        "per_video": {
            video_id: {
                "score": float(score["strict_browser_live_score"]),
                "wrong": float(score["wrong_live_speech_ratio"]),
                "missing": float(score["missing_live_speech_ratio"]),
            }
            for video_id, score in scores.items()
        },
        "interventions": interventions,
    }


def _reference_aggregate(reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return aggregate_video_scores(
        {"strict_browser_live_score": float(value["score"])} for value in reference.values()
    )


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for video_id, score in candidate["per_video"].items():
        score_delta = float(score["score"]) - float(reference[video_id]["score"])
        wrong_delta = float(score["wrong"]) - float(reference[video_id]["wrong"])
        rows[video_id] = {
            "score_delta": round(score_delta, 9),
            "wrong_delta": round(wrong_delta, 9),
            "score_gate": score_delta >= -0.005 - 1e-12,
            "wrong_gate": wrong_delta <= 0.005 + 1e-12,
        }
        rows[video_id]["passed"] = rows[video_id]["score_gate"] and rows[video_id]["wrong_gate"]
    reference_aggregate = _reference_aggregate(reference)
    return {
        "global_score_delta": round(
            float(candidate["aggregate"]["global_score"])
            - float(reference_aggregate["global_score"]),
            9,
        ),
        "all_video_gates_passed": all(value["passed"] for value in rows.values()),
        "per_video": rows,
    }


def _selector_trials(
    prepared: dict[str, PreparedVideo],
    references: dict[str, Any],
    history_max_gap_seconds: float,
) -> list[dict[str, Any]]:
    configs = [
        SelectorConfig(threshold, "current", 1, history_max_gap_seconds)
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    ] + [
        SelectorConfig(threshold, statistic, 2, history_max_gap_seconds)
        for statistic in ("median", "min")
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    ]
    rows: list[dict[str, Any]] = []
    for config in configs:
        row = _score_selector(prepared, config)
        row["vs_baseline"] = _comparison(row, references["baseline"])
        row["vs_run018"] = _comparison(row, references["run018"])
        row["all_fifteen_gates_passed"] = bool(
            row["vs_baseline"]["all_video_gates_passed"]
            and row["vs_run018"]["all_video_gates_passed"]
        )
        rows.append(row)
    return rows


def _aggregate_compact(
    per_video: dict[str, dict[str, Any]],
    video_ids: Sequence[str],
) -> dict[str, Any]:
    return aggregate_video_scores(
        {"strict_browser_live_score": float(per_video[video_id]["score"])}
        for video_id in video_ids
    )


def _config_label(config: dict[str, Any]) -> str:
    return f"{config['statistic']}{int(config['history_size'])}_t{float(config['threshold']):.2f}"


def _conditional_loov(
    trials: Sequence[dict[str, Any]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    video_ids = list(references["baseline"])
    folds: list[dict[str, Any]] = []
    for held_out in video_ids:
        train_ids = [video_id for video_id in video_ids if video_id != held_out]
        baseline_aggregate = _aggregate_compact(references["baseline"], train_ids)
        run018_aggregate = _aggregate_compact(references["run018"], train_ids)
        eligible: list[tuple[tuple[float, float, int, float], dict[str, Any], dict[str, Any]]] = []
        for trial in trials:
            all_train_gates = all(
                bool(trial[reference]["per_video"][video_id]["passed"])
                for reference in ("vs_baseline", "vs_run018")
                for video_id in train_ids
            )
            candidate_aggregate = _aggregate_compact(trial["per_video"], train_ids)
            delta_baseline = (
                float(candidate_aggregate["global_score"])
                - float(baseline_aggregate["global_score"])
            )
            delta_run018 = (
                float(candidate_aggregate["global_score"])
                - float(run018_aggregate["global_score"])
            )
            condition = bool(
                all_train_gates
                and delta_baseline > 1e-9
                and delta_run018 > 1e-9
            )
            if condition:
                # Prefer the robust worst-reference delta, then total score.  A
                # single-tick rule wins exact ties over a history rule.
                rank = (
                    min(delta_baseline, delta_run018),
                    float(candidate_aggregate["global_score"]),
                    1 if trial["config"]["statistic"] == "current" else 0,
                    float(trial["config"]["threshold"]),
                )
                eligible.append((rank, trial, {
                    "candidate": candidate_aggregate,
                    "delta_baseline": delta_baseline,
                    "delta_run018": delta_run018,
                }))
        if not eligible:
            folds.append({
                "held_out": held_out,
                "status": "no_training_candidate_met_conditional_promotion",
                "passed": False,
            })
            continue
        _rank, selected, train = max(eligible, key=lambda item: item[0])
        baseline_gate = selected["vs_baseline"]["per_video"][held_out]
        run018_gate = selected["vs_run018"]["per_video"][held_out]
        passed = bool(baseline_gate["passed"] and run018_gate["passed"])
        folds.append({
            "held_out": held_out,
            "status": "selected_on_fourteen",
            "selected": _config_label(selected["config"]),
            "config": selected["config"],
            "eligible_training_candidate_count": len(eligible),
            "training": train,
            "held_out_vs_baseline": baseline_gate,
            "held_out_vs_run018": run018_gate,
            "passed": passed,
        })
    selected_folds = [row for row in folds if row["status"] == "selected_on_fourteen"]
    return {
        "fold_count": len(folds),
        "conditional_selection_count": len(selected_folds),
        "held_out_gate_pass_count": sum(bool(row["passed"]) for row in selected_folds),
        "all_conditional_selections_passed": bool(
            selected_folds and all(bool(row["passed"]) for row in selected_folds)
        ),
        "selection_histogram": {
            label: sum(row.get("selected") == label for row in selected_folds)
            for label in sorted({str(row.get("selected")) for row in selected_folds})
        },
        "folds": folds,
    }


def _full_cohort_winners(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        trial
        for trial in trials
        if trial["all_fifteen_gates_passed"]
        and float(trial["vs_baseline"]["global_score_delta"]) > 1e-9
        and float(trial["vs_run018"]["global_score_delta"]) > 1e-9
    ]
    ordered = sorted(
        eligible,
        key=lambda trial: (
            min(
                float(trial["vs_baseline"]["global_score_delta"]),
                float(trial["vs_run018"]["global_score_delta"]),
            ),
            float(trial["aggregate"]["global_score"]),
            1 if trial["config"]["statistic"] == "current" else 0,
        ),
        reverse=True,
    )
    return [
        {
            "label": _config_label(trial["config"]),
            "config": trial["config"],
            "aggregate": trial["aggregate"],
            "vs_baseline": trial["vs_baseline"],
            "vs_run018": trial["vs_run018"],
            "interventions": trial["interventions"],
        }
        for trial in ordered
    ]


def _coherence_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "current_short_long_cos": _quantiles(
            float(row["current_short_long_cos"]) for row in rows
        ),
        "min_last_two_short_long_cos": _quantiles(
            float(row["min_last_two_short_long_cos"]) for row in rows
        ),
    }


def _threshold_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"total": len(rows)}
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        label = f"ge_{threshold:.2f}"
        result[label] = {
            "current": sum(
                float(row["current_short_long_cos"]) >= threshold for row in rows
            ),
            "min2": sum(
                float(row["min_last_two_short_long_cos"]) >= threshold for row in rows
            ),
        }
    return result


def _fixed_meta_lease_coherence(
    prepared: dict[str, PreparedVideo],
    history_max_gap_seconds: float,
) -> dict[str, Any]:
    """Measure raw coherence at the already-fixed meta leases without proposing a rule."""

    focus_ids = (
        "JWS-qfR6K3w",
        "F2-2RBi1qzY",
        "pD4IdQTmneI",
        "blcKeLDDzSM",
        "KdOXM3I_5hk",
        "acbnyagl8jo",
    )
    all_leases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leases_by_video: dict[str, dict[str, list[dict[str, Any]]]] = {
        video_id: defaultdict(list) for video_id in focus_ids
    }
    blc_disagreements: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for video_id, value in prepared.items():
        baseline_score = score_live_speaker_decisions(
            value.baseline, value.inputs["canonical"], value.inputs["profiles"]
        )
        fixed_map = dict(baseline_score.get("speaker_map") or {})
        history: deque[float] = deque(maxlen=2)
        last_probe_time: float | None = None
        for index, media_time in enumerate(value.short.media_times):
            if bool(value.inputs["releases"][index]):
                history.clear()
                last_probe_time = None
            if not bool(value.inputs["probes"][index]):
                continue
            if not (bool(value.short.valid[index]) and bool(value.long.valid[index])):
                continue
            if (
                last_probe_time is None
                or float(media_time) - last_probe_time
                > history_max_gap_seconds + 1e-9
            ):
                history.clear()
            coherence = float(np.dot(
                _unit(value.short.embeddings[index]),
                _unit(value.long.embeddings[index]),
            ))
            history.append(coherence)
            last_probe_time = float(media_time)
            truth = _truth_at(value.inputs["canonical"], float(media_time))
            row = {
                "video_id": video_id,
                "media_time": round(float(media_time), 6),
                "current_short_long_cos": coherence,
                "min_last_two_short_long_cos": float(min(history)),
            }

            baseline = value.baseline[index]
            precision = value.run018[index]
            fixed_meta = value.meta[index]
            meta_details = dict(
                fixed_meta.diagnostics.get("profile_quality_meta_lease") or {}
            )
            lease_started = bool(
                fixed_meta.diagnostics.get("profile_quality_meta_lease_used")
                and meta_details.get("state_reason") == "lease_started"
            )
            if lease_started:
                meta_correct = _is_correct(fixed_meta, truth, fixed_map)
                run018_correct = _is_correct(precision, truth, fixed_map)
                if meta_correct and not run018_correct:
                    category = "beneficial"
                elif run018_correct and not meta_correct:
                    category = "harmful"
                elif meta_correct:
                    category = "both_correct"
                else:
                    category = "both_wrong"
                lease_row = {
                    **row,
                    "category": category,
                    "branch": meta_details.get("branch"),
                    "recall_visible": baseline.visible_speaker,
                    "run018_visible": precision.visible_speaker,
                    "truth": truth,
                }
                all_leases[category].append(lease_row)
                all_leases["all"].append(lease_row)
                if video_id in leases_by_video:
                    leases_by_video[video_id][category].append(lease_row)
                    leases_by_video[video_id]["all"].append(lease_row)

            if video_id == "blcKeLDDzSM" and baseline.visible_speaker != precision.visible_speaker:
                baseline_correct = _is_correct(baseline, truth, fixed_map)
                run018_correct = _is_correct(precision, truth, fixed_map)
                if baseline_correct and not run018_correct:
                    category = "recall_rescue_opportunity"
                elif run018_correct and not baseline_correct:
                    category = "precision_protection"
                elif baseline_correct:
                    category = "both_correct"
                else:
                    category = "both_wrong"
                blc_disagreements[category].append({**row, "category": category})
                blc_disagreements["all"].append({**row, "category": category})

    def grouped(rows_by_category: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            category: {
                "distribution": _coherence_summary(rows),
                "high_coherence_counts": _threshold_counts(rows),
            }
            for category, rows in sorted(rows_by_category.items())
        }

    return {
        "classification": (
            "tick-local fixed-map effect at scheduled lease start; the production-baseline "
            "speaker map is held fixed for both experts"
        ),
        "aggregate_meta_lease_starts": grouped(all_leases),
        "focus_videos": {
            video_id: grouped(leases_by_video[video_id]) for video_id in focus_ids
        },
        "blc_run018_disagreement_rescue": grouped(blc_disagreements),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources = _validate_sources(args.dataset_source)
    prepared = _prepare(sources, args.locked_run_dir)
    references = _compact_scores(prepared)
    report = {
        "schema_version": 1,
        "analyzer_id": ANALYZER_ID,
        "opened_sources_only": True,
        "source_video_ids": {key: list(value) for key, value in EXPECTED_SOURCES.items()},
        "forbidden_unopened_ids": sorted(FORBIDDEN_UNOPENED_IDS),
        "provider": EXPECTED_PROVIDER,
        "windows_seconds": list(EXPECTED_WINDOWS),
        "fresh_embedding_requests": 0,
        "maximum_live_windows_modeled": 2,
        "history_max_gap_seconds": float(args.history_max_gap_seconds),
        "reference_scores": references,
        "diagnostics": _diagnose(prepared, float(args.history_max_gap_seconds)),
    }
    report["fixed_meta_lease_coherence"] = _fixed_meta_lease_coherence(
        prepared, float(args.history_max_gap_seconds)
    )
    if not args.diagnostics_only:
        report["selector_trials"] = _selector_trials(
            prepared, references, float(args.history_max_gap_seconds)
        )
        report["full_cohort_winners"] = _full_cohort_winners(report["selector_trials"])
        report["conditional_loov"] = _conditional_loov(report["selector_trials"], references)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": "complete",
        "output": str(args.output),
        "video_count": len(prepared),
        "history_rows": report["diagnostics"]["scheduled_dual_probe_pairs_with_history"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
