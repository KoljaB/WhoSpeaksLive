"""Describe causal error runs for one cached Bayesian Top-7 champion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import score_live_speaker_decisions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _truth_at(segments: list[dict[str, Any]], media_time: float) -> tuple[str, ...]:
    labels = {
        str(row.get("speaker") or row.get("speaker_id") or row.get("label") or "")
        for row in segments
        if float(row.get("start", 0.0)) <= media_time < float(row.get("end", 0.0))
    }
    labels.discard("")
    return tuple(sorted(labels))


def main() -> int:
    args = _args()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    windows = tuple(float(value) for value in champion["windows_seconds"])
    config = BayesSpeakerTrackerConfig.from_mapping(champion["algorithm_config"])
    dataset = Dataset(
        args.corpus_root.resolve(), args.input_root.resolve(),
        str(champion["provider_spec"]), str(champion["profile_name"]),
    )
    output: dict[str, Any] = {"champion_score": champion["candidate_score"], "videos": {}}
    for video_id in spec["videos"]:
        inputs = dataset.video_inputs(str(video_id), min(windows))
        if args.gate_root is not None:
            video_gate = args.gate_root.resolve() / str(video_id)
            speech = np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False)
            probes = np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False)
            releases = np.load(video_gate / "release_gate.u1.npy", allow_pickle=False)
        else:
            speech, probes, releases = inputs["speech"], inputs["probes"], inputs["releases"]
        decisions = replay_cached_bayes_windows(
            [dataset.block(str(video_id), window) for window in windows],
            inputs["profiles"], speech, probes, releases,
            config=config,
        )
        score = score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"])
        speaker_map = {str(key): str(value) for key, value in score["speaker_map"].items()}
        rows: list[dict[str, Any]] = []
        reason_by_kind: dict[str, Counter[str]] = {}
        for index, decision in enumerate(decisions):
            truth = _truth_at(inputs["canonical"], float(decision.media_time))
            mapped = speaker_map.get(str(decision.visible_speaker or ""), "")
            if truth:
                kind = "correct" if mapped in truth else "wrong" if mapped else "missing"
            else:
                kind = "outside" if decision.visible_speaker else "idle"
            reason_by_kind.setdefault(kind, Counter())[str(decision.reason)] += 1
            delta = (
                max(0.0, float(decisions[index + 1].media_time) - float(decision.media_time))
                if index + 1 < len(decisions) else 0.0
            )
            row = {
                "kind": kind,
                "start": float(decision.media_time),
                "end": float(decision.media_time) + delta,
                "truth": list(truth),
                "visible": decision.visible_speaker,
                "mapped": mapped or None,
                "reason": decision.reason,
                "candidate": decision.candidate_speaker,
                "profile_count": decision.profile_count,
                "release_signal": bool(decision.diagnostics.get("release_signal")),
                "probe_scheduled": bool(decision.diagnostics.get("probe_scheduled")),
                "scale_agreement": decision.diagnostics.get("scale_agreement"),
            }
            if rows and all(rows[-1][key] == row[key] for key in ("kind", "truth", "visible", "mapped")):
                rows[-1]["end"] = row["end"]
            else:
                rows.append(row)
        error_runs = [
            {**row, "duration": round(float(row["end"]) - float(row["start"]), 4)}
            for row in rows if row["kind"] in {"wrong", "missing", "outside"}
        ]
        error_runs.sort(key=lambda row: (-float(row["duration"]), float(row["start"])))
        output["videos"][str(video_id)] = {
            "score": score["strict_browser_live_score"],
            "speaker_map": speaker_map,
            "reason_counts_by_kind": {
                kind: dict(counter.most_common()) for kind, counter in reason_by_kind.items()
            },
            "longest_error_runs": error_runs[:30],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
