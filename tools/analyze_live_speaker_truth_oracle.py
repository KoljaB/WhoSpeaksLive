"""Truth-aware Baseline-vs-run018 selector ceiling on opened v1-v4 only."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyze_live_speaker_embedding_dynamics as dynamics
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_probe_scoring import intervals_for_speaker, merge_intervals, overlap_seconds


def _correct(decision: object, truth: str | None, speaker_map: dict[str, str]) -> bool:
    visible = getattr(decision, "visible_speaker", None)
    if truth is None:
        return visible is None
    return bool(visible and speaker_map.get(str(visible)) == truth)


def _comparison(
    candidate: dict[str, dict], reference: dict[str, dict]
) -> dict:
    rows = {}
    for video_id, score in candidate.items():
        score_delta = float(score["strict_browser_live_score"]) - float(
            reference[video_id]["strict_browser_live_score"]
        )
        wrong_delta = float(score["wrong_live_speech_ratio"]) - float(
            reference[video_id]["wrong_live_speech_ratio"]
        )
        rows[video_id] = {
            "score_delta": round(score_delta, 9),
            "wrong_delta": round(wrong_delta, 9),
            "score_gate": score_delta >= -0.005 - 1e-12,
            "wrong_gate": wrong_delta <= 0.005 + 1e-12,
        }
        rows[video_id]["passed"] = bool(
            rows[video_id]["score_gate"] and rows[video_id]["wrong_gate"]
        )
    candidate_aggregate = aggregate_video_scores(candidate.values())
    reference_aggregate = aggregate_video_scores(reference.values())
    return {
        "global_score_delta": round(
            float(candidate_aggregate["global_score"])
            - float(reference_aggregate["global_score"]),
            9,
        ),
        "all_video_gates_passed": all(row["passed"] for row in rows.values()),
        "failed_videos": [key for key, row in rows.items() if not row["passed"]],
        "per_video": rows,
    }


def _interval_utility(
    decisions: list[object],
    start_index: int,
    end_index: int,
    media_times: object,
    speaker_map: dict[str, str],
    canonical_by_speaker: dict[str, list[tuple[float, float]]],
    canonical_speech: list[tuple[float, float]],
) -> dict[str, float]:
    correct = wrong = outside = 0.0
    limit = min(int(end_index), len(decisions), len(media_times) - 1)
    for index in range(int(start_index), limit):
        start = float(media_times[index])
        end = min(float(media_times[index + 1]), start + 0.25)
        if end <= start:
            continue
        visible = str(getattr(decisions[index], "visible_speaker", None) or "")
        if not visible:
            continue
        speech = overlap_seconds([(start, end)], canonical_speech)
        mapped = speaker_map.get(visible)
        matched = (
            overlap_seconds([(start, end)], canonical_by_speaker.get(mapped, []))
            if mapped
            else 0.0
        )
        correct += matched
        wrong += max(0.0, speech - matched)
        outside += max(0.0, end - start - speech)
    return {"correct": correct, "wrong": wrong, "outside": outside}


def _lexicographic_key(utility: dict[str, float]) -> tuple[float, float, float]:
    return (-utility["wrong"], utility["correct"], -utility["outside"])


def _net_key(utility: dict[str, float]) -> float:
    return utility["correct"] - utility["wrong"] - 0.25 * utility["outside"]


def main() -> int:
    sources = [
        dynamics.DatasetSource(
            label,
            ROOT / f"runtime/optimization/live_shifting_windows_{label}",
            ROOT / (
                "runtime/optimization/live_replay_inputs"
                if label == "v1"
                else f"runtime/optimization/live_replay_inputs_{label}"
            ),
            video_ids,
        )
        for label, video_ids in dynamics.EXPECTED_SOURCES.items()
    ]
    prepared = dynamics._prepare(
        sources,
        ROOT / "runtime/optimization/live_speaker_runs/20260720_linux_018_hybrid_locked",
    )
    reference: dict[str, dict[str, dict]] = {"baseline": {}, "run018": {}}
    candidate_scores: dict[str, dict[str, dict]] = {
        "lexicographic": {},
        "correct_minus_wrong": {},
    }
    counts: dict[str, dict[str, dict[str, int]]] = {}
    mapping_disagreements: dict[str, dict] = {}

    for video_id, value in prepared.items():
        baseline_score = score_live_speaker_decisions(
            value.baseline, value.inputs["canonical"], value.inputs["profiles"]
        )
        run018_score = score_live_speaker_decisions(
            value.run018, value.inputs["canonical"], value.inputs["profiles"]
        )
        reference["baseline"][video_id] = baseline_score
        reference["run018"][video_id] = run018_score
        fixed_map = dict(baseline_score.get("speaker_map") or {})
        run018_map = dict(run018_score.get("speaker_map") or {})
        if fixed_map != run018_map:
            mapping_disagreements[video_id] = {
                "baseline": fixed_map,
                "run018": run018_map,
            }

        canonical_by_speaker = intervals_for_speaker(value.inputs["canonical"])
        canonical_speech = merge_intervals([
            interval
            for speaker_intervals in canonical_by_speaker.values()
            for interval in speaker_intervals
        ])
        selected = {
            "lexicographic": "run018",
            "correct_minus_wrong": "run018",
        }
        decisions = {name: [] for name in selected}
        row_counts = {
            name: {
                "scheduled_disagreements": 0,
                "baseline_selected": 0,
                "run018_selected": 0,
            }
            for name in selected
        }
        for index, media_time in enumerate(value.short.media_times):
            baseline = value.baseline[index]
            precision = value.run018[index]
            scheduled = bool(value.inputs["probes"][index])
            valid = bool(value.short.valid[index]) and bool(value.long.valid[index])
            if bool(value.inputs["releases"][index]):
                for name in selected:
                    selected[name] = "run018"
            if scheduled and valid and baseline.visible_speaker != precision.visible_speaker:
                end_index = index + 1
                while end_index < len(value.short.media_times):
                    if bool(value.inputs["probes"][end_index]) or bool(
                        value.inputs["releases"][end_index]
                    ):
                        break
                    end_index += 1
                baseline_utility = _interval_utility(
                    value.baseline,
                    index,
                    end_index,
                    value.short.media_times,
                    fixed_map,
                    canonical_by_speaker,
                    canonical_speech,
                )
                run018_utility = _interval_utility(
                    value.run018,
                    index,
                    end_index,
                    value.short.media_times,
                    fixed_map,
                    canonical_by_speaker,
                    canonical_speech,
                )
                selected["lexicographic"] = (
                    "baseline"
                    if _lexicographic_key(baseline_utility)
                    > _lexicographic_key(run018_utility)
                    else "run018"
                )
                selected["correct_minus_wrong"] = (
                    "baseline"
                    if _net_key(baseline_utility) > _net_key(run018_utility) + 1e-12
                    else "run018"
                )
                for name, expert in selected.items():
                    row_counts[name]["scheduled_disagreements"] += 1
                    row_counts[name][f"{expert}_selected"] += 1
            for name, expert in selected.items():
                chosen = (
                    precision
                    if baseline.visible_speaker == precision.visible_speaker
                    else baseline if expert == "baseline" else precision
                )
                decisions[name].append(
                    replace(
                        chosen,
                        action="offline_interval_truth_oracle",
                        reason=f"interval_truth_oracle_{name}_{expert}",
                    )
                )
        for name, trace in decisions.items():
            candidate_scores[name][video_id] = score_live_speaker_decisions(
                trace, value.inputs["canonical"], value.inputs["profiles"]
            )
        counts[video_id] = row_counts

    report = {
        "schema_version": 1,
        "scope": "opened_v1_v4_only",
        "oracle": (
            "At each scheduled valid expert disagreement integrate each expert's exact "
            "canonical overlap over all scorer slices until the next scheduled probe or "
            "release, choose by either (least wrong, most correct, least outside) or "
            "correct-wrong-0.25*outside, then hold that expert for the interval."
        ),
        "fresh_embedding_requests": 0,
        "windows_seconds": [0.8, 2.8],
        "mapping_disagreements": mapping_disagreements,
        "counts": counts,
        "references": {
            name: {
                "aggregate": aggregate_video_scores(scores.values()),
                "per_video": {
                    video_id: {
                        "score": score["strict_browser_live_score"],
                        "wrong": score["wrong_live_speech_ratio"],
                    }
                    for video_id, score in scores.items()
                },
            }
            for name, scores in reference.items()
        },
        "oracle_results": {},
    }
    for name, scores in candidate_scores.items():
        report["oracle_results"][name] = {
            "aggregate": aggregate_video_scores(scores.values()),
            "per_video": {
                video_id: {
                    "score": score["strict_browser_live_score"],
                    "wrong": score["wrong_live_speech_ratio"],
                }
                for video_id, score in scores.items()
            },
            "vs_baseline": _comparison(scores, reference["baseline"]),
            "vs_run018": _comparison(scores, reference["run018"]),
        }
    output = ROOT / "runtime/optimization/live_speaker_research/learned_selector_v1/oracle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "scheduled_disagreements": sum(
            row["lexicographic"]["scheduled_disagreements"] for row in counts.values()
        ),
        "variants": report["oracle_results"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
