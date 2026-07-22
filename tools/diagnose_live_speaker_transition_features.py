"""Post-hoc feature audit for the exact exclusive-tracklet World-Tape baseline.

Canonical labels are used only after inference to tag observations as correct or
wrong.  The emitted feature rows contain only values that were available at the
causal decision instant.  This is a research diagnostic, never promotion proof.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from analyze_live_speaker_open_set_tracklets import (
    TrackletConfig,
    _load_base_config,
    _prepare_tape,
    _profile_vectors,
    _replay_with_tracklet_actions,
    _tracklet_projection,
    _unit,
)
from window.live_speaker_probe_scoring import read_canonical_segments


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


def _canonical_at(segments: list[dict[str, Any]], time_value: float) -> str:
    hits = [
        item
        for item in segments
        if float(item["start"]) <= time_value < float(item["end"])
    ]
    if not hits:
        return ""
    # Overlap is rare.  Prefer the segment whose midpoint is closest to the
    # observation so the diagnostic remains deterministic.
    return str(
        min(
            hits,
            key=lambda item: abs(
                0.5 * (float(item["start"]) + float(item["end"])) - time_value
            ),
        )["speaker"]
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        "prev_short",
        "prev_long",
        "active_short",
        "active_long",
        "known_short",
        "known_long",
        "known_margin_short",
        "known_margin_long",
        "assigned_short",
        "assigned_long",
        "short_long",
        "profile_count",
        "active_age_steps",
        "seconds_since_release",
        "tracklet_short",
        "tracklet_long",
        "tracklet_known_advantage_short",
        "tracklet_known_advantage_long",
    ]
    result: dict[str, Any] = {"count": len(rows)}
    for key in numeric:
        values = sorted(float(item[key]) for item in rows if item.get(key) is not None)
        if not values:
            continue
        result[key] = {
            "mean": mean(values),
            "p10": values[int(0.10 * (len(values) - 1))],
            "p50": median(values),
            "p90": values[int(0.90 * (len(values) - 1))],
        }
    return result


def _threshold_scan(rows: list[dict[str, Any]], key: str, direction: str) -> list[dict[str, Any]]:
    values = sorted({round(float(item[key]), 3) for item in rows if item.get(key) is not None})
    if len(values) > 80:
        values = [values[int(index * (len(values) - 1) / 79)] for index in range(80)]
    total_wrong = sum(item["outcome"] == "wrong" for item in rows)
    result = []
    for threshold in values:
        selected = [
            item
            for item in rows
            if item.get(key) is not None
            and (
                float(item[key]) <= threshold
                if direction == "low"
                else float(item[key]) >= threshold
            )
        ]
        if len(selected) < 3:
            continue
        wrong = sum(item["outcome"] == "wrong" for item in selected)
        correct = sum(item["outcome"] == "correct" for item in selected)
        result.append(
            {
                "threshold": threshold,
                "selected": len(selected),
                "wrong": wrong,
                "correct": correct,
                "precision_wrong": wrong / max(1, wrong + correct),
                "wrong_recall": wrong / max(1, total_wrong),
                # Clearing a wrong action avoids wrong exposure; clearing a
                # correct action loses correct coverage.  Difference is the
                # useful first-order score proxy before reducer hold effects.
                "net_proxy": wrong - correct,
            }
        )
    return sorted(result, key=lambda item: (item["net_proxy"], item["precision_wrong"]), reverse=True)[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity-report",
        type=Path,
        default=Path("runtime/optimization/live_speaker_world_tapes_20260721/baseline_parity_report.json"),
    )
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--exclusive-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.parity_report.read_text(encoding="utf-8-sig"))
    base_config = _load_base_config(args.base_candidate.resolve())
    prepared = [_prepare_tape(run, base_config) for run in parity.get("runs") or []]
    exclusive_data = json.loads(args.exclusive_result.read_text(encoding="utf-8-sig"))
    config = TrackletConfig(**exclusive_data["exclusive"]["config"])

    all_rows: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for tape in prepared:
        actions, _stats = _tracklet_projection(tape, config)
        score = _replay_with_tracklet_actions(tape, actions)
        speaker_map = {str(k): str(v) for k, v in dict(score.get("speaker_map") or {}).items()}
        canonical = read_canonical_segments(tape.canonical_path)
        by_step = {
            int(dict(item[3]).get("step_id") or 0): item
            for item in actions
            if item[2] in {"live_speaker", "live_speaker_clear"}
            and int(dict(item[3]).get("step_id") or 0)
        }
        prev_short: np.ndarray | None = None
        prev_long: np.ndarray | None = None
        active_label = ""
        active_short_history: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=5))
        active_long_history: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=5))
        active_steps: dict[str, int] = defaultdict(int)
        last_release_time = -1e9
        rows: list[dict[str, Any]] = []
        for step in tape.steps:
            payload = step.payload
            if not str(payload.get("probe_id") or ""):
                continue
            media_time = float(payload.get("media_time") or 0.0)
            if bool(payload.get("release_signal")) or not bool(payload.get("speech")):
                last_release_time = media_time
            short = _unit(payload.get("embedding"))
            long = _unit(payload.get("context_embedding"))
            if short is None:
                continue
            source = by_step.get(int(payload.get("step_id") or 0))
            event = source[2] if source is not None else ""
            assigned = ""
            reason = ""
            if source is not None:
                assigned = str(source[3].get("assigned_speaker") or source[3].get("speaker_id") or "")
                reason = str(source[3].get("live_speaker_core_reason") or source[3].get("reason") or "")
            source_payload = dict(source[3]) if source is not None else {}
            profiles = _profile_vectors(payload)
            short_rank = sorted(((_cosine(short, value), label) for label, value in profiles.items()), reverse=True)
            long_rank = sorted(((_cosine(long, value), label) for label, value in profiles.items()), reverse=True) if long is not None else []
            known_short = short_rank[0][0] if short_rank else -1.0
            known_long = long_rank[0][0] if long_rank else -1.0
            assigned_profile = profiles.get(assigned)
            assigned_short = _cosine(short, assigned_profile)
            assigned_long = _cosine(long, assigned_profile)
            active_short = max((_cosine(short, value) for value in active_short_history.get(active_label, ())), default=-1.0)
            active_long = max((_cosine(long, value) for value in active_long_history.get(active_label, ())), default=-1.0)
            truth = _canonical_at(canonical, media_time - 0.20)
            mapped = speaker_map.get(assigned, "")
            outcome = "outside" if not truth else ("missing" if not assigned else ("correct" if mapped == truth else "wrong"))
            row = {
                "video_id": tape.video_id,
                "run_id": tape.run_id,
                "step_id": int(payload.get("step_id") or 0),
                "media_time": media_time,
                "assigned": assigned,
                "mapped": mapped,
                "canonical": truth,
                "outcome": outcome,
                "reason": reason,
                "base_label_changed": bool(assigned and active_label and assigned != active_label),
                "prev_short": _cosine(short, prev_short),
                "prev_long": _cosine(long, prev_long),
                "active_short": active_short,
                "active_long": active_long,
                "known_short": known_short,
                "known_long": known_long,
                "known_margin_short": known_short - (short_rank[1][0] if len(short_rank) > 1 else -1.0),
                "known_margin_long": known_long - (long_rank[1][0] if len(long_rank) > 1 else -1.0),
                "assigned_short": assigned_short,
                "assigned_long": assigned_long,
                "short_long": _cosine(short, long),
                "profile_count": len(profiles),
                "active_age_steps": active_steps.get(assigned, 0),
                "seconds_since_release": media_time - last_release_time,
                "tracklet_short": float(source_payload.get("diagnostic_tracklet_short", -1.0)),
                "tracklet_long": float(source_payload.get("diagnostic_tracklet_long", -1.0)),
                "tracklet_known_advantage_short": float(source_payload.get("diagnostic_tracklet_short", -1.0)) - float(source_payload.get("diagnostic_known_short", -1.0)),
                "tracklet_known_advantage_long": float(source_payload.get("diagnostic_tracklet_long", -1.0)) - float(source_payload.get("diagnostic_known_long", -1.0)),
                "active_tracklet_short": float(source_payload.get("diagnostic_active_short", -1.0)),
                "active_tracklet_long": float(source_payload.get("diagnostic_active_long", -1.0)),
                "normal_tracklet_pass": bool(source_payload.get("diagnostic_normal_tracklet_pass", False)),
                "relaxed_tracklet_pass": bool(source_payload.get("diagnostic_relaxed_tracklet_pass", False)),
                "novel": bool(source_payload.get("diagnostic_novel", False)),
            }
            rows.append(row)
            if event == "live_speaker" and assigned:
                active_label = assigned
                active_short_history[assigned].append(short.copy())
                if long is not None:
                    active_long_history[assigned].append(long.copy())
                active_steps[assigned] += 1
            elif event == "live_speaker_clear":
                active_label = ""
            prev_short = short.copy()
            prev_long = None if long is None else long.copy()
        all_rows.extend(rows)
        per_run.append(
            {
                "video_id": tape.video_id,
                "run_id": tape.run_id,
                "strict_score": score["strict_browser_live_score"],
                "outcomes": {key: sum(item["outcome"] == key for item in rows) for key in ("correct", "wrong", "missing", "outside")},
            }
        )

    speech_rows = [item for item in all_rows if item["outcome"] in {"correct", "wrong"}]
    by_outcome = {
        key: _summary([item for item in all_rows if item["outcome"] == key])
        for key in ("correct", "wrong", "missing", "outside")
    }
    scans = {}
    for key in ("prev_short", "active_short", "known_short", "known_margin_short", "assigned_short", "prev_long", "active_long", "known_long", "known_margin_long", "assigned_long", "short_long", "tracklet_short", "tracklet_long", "tracklet_known_advantage_short", "tracklet_known_advantage_long", "active_tracklet_short", "active_tracklet_long"):
        scans[f"{key}_low"] = _threshold_scan(speech_rows, key, "low")
        scans[f"{key}_high"] = _threshold_scan(speech_rows, key, "high")
    output = {
        "status": "POST_HOC_DIAGNOSTIC_ONLY_NOT_INFERENCE_OR_PROMOTION_EVIDENCE",
        "canonical_used_only_to_tag_outcome_after_actions": True,
        "exclusive_config": asdict(config),
        "per_run": per_run,
        "by_outcome": by_outcome,
        "threshold_scans": scans,
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "by_outcome": {k: v["count"] for k, v in by_outcome.items()}, "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
