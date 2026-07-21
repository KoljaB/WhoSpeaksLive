"""Ablate causal profile history and maturity around the adaptive Bayes champion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BAYES_ALGORITHM_ID, BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import PRIMARY_SCORER_V2_ID, aggregate_video_scores_primary_v2, score_live_speaker_decisions


OPTIMIZER_ID = "live_speaker_top7_bayes_profile_evidence_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["aggregate"]["primary_score"]), reverse=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, int(args.budget_seconds))
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    provider = str(source["provider_spec"])
    profile_name = str(source["profile_name"])
    windows = tuple(float(value) for value in source["windows_seconds"])
    source_config = BayesSpeakerTrackerConfig(**source["algorithm_config"])
    source_score = float(source["candidate_score"])
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and trials_path.is_file():
        for line in trials_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["candidate_id"])] = row
    elif trials_path.exists():
        raise FileExistsError(f"{trials_path} exists; pass --resume")
    incumbent = _rank(completed.values())[0] if completed else None
    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase_counts[str(row["phase"])] = phase_counts.get(str(row["phase"]), 0) + 1

    def write_state(phase: str, active: str = "") -> None:
        best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
        _atomic_json(run_dir / "progress.json", {
            "status": "interrupted" if _STOP else "running",
            "phase": phase,
            "active": active,
            "completed_candidate_count": len(completed),
            "phase_counts": phase_counts,
            "source_champion_score": source_score,
            "best_profile_score": best if incumbent else None,
            "score_delta": round(best - source_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        if incumbent:
            _atomic_json(run_dir / "champion.json", {
                "status": "CACHE_PROFILE_BAYES_WINNER_PENDING_FRESH_LIVE" if best > source_score else "BELOW_SOURCE_CHAMPION",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "source_champion_score": source_score,
                "candidate_score": best,
                "score_delta": round(best - source_score, 6),
                "provider_spec": provider,
                "profile_name": profile_name,
                "windows_seconds": list(windows),
                **incumbent,
                "fresh_live_verified": False,
            })

    def evaluate(config: BayesSpeakerTrackerConfig, phase: str, hypothesis: str, parent: str | None = None) -> dict[str, Any] | None:
        nonlocal incumbent
        candidate_id = _stable_id({
            "optimizer_id": OPTIMIZER_ID,
            "algorithm_id": BAYES_ALGORITHM_ID,
            "primary_scorer_id": PRIMARY_SCORER_V2_ID,
            "windows": windows,
            "config": asdict(config),
        })
        if candidate_id in completed:
            return completed[candidate_id]
        if _STOP or time.monotonic() >= deadline:
            return None
        write_state(phase, candidate_id)
        per_video: dict[str, Any] = {}
        short = min(windows)
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, short)
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in windows],
                inputs["profiles"], inputs["speech"], inputs["probes"], inputs["releases"],
                config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(
                decisions, inputs["canonical"], inputs["profiles"]
            ))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        if not math.isfinite(float(aggregate["primary_score"])):
            raise RuntimeError("Non-finite profile-evidence score")
        row = {
            "candidate_id": candidate_id,
            "phase": phase,
            "hypothesis": hypothesis,
            "parent_candidate_id": parent,
            "algorithm_config": asdict(config),
            "aggregate": aggregate,
            "per_video": per_video,
            "score_delta_vs_source": round(float(aggregate["primary_score"]) - source_score, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        completed[candidate_id] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        _append(trials_path, row)
        if incumbent is None or float(aggregate["primary_score"]) > float(incumbent["aggregate"]["primary_score"]):
            incumbent = row
        write_state(phase, candidate_id)
        return row

    _atomic_json(run_dir / "run.json", {
        "optimizer_id": OPTIMIZER_ID,
        "algorithm_id": BAYES_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID,
        "source_champion": str(args.champion.resolve()),
        "source_champion_score": source_score,
        "videos": videos,
        "maximum_fresh_windows_per_probe": 2,
        "selection_policy": "one Top-7 primary score",
    })
    write_state("SOURCE_CHAMPION")

    history_rows: list[dict[str, Any]] = []
    for size in (2, 3, 4, 5, 6, 8):
        for weight in (0.10, 0.25, 0.50, 0.75, 1.00):
            row = evaluate(
                replace(source_config, profile_history_size=size, profile_history_max_weight=weight),
                "STAGE_1_PROFILE_HISTORY",
                "Match each probe against a bounded history of causal profile snapshots rather than only the latest centroid.",
            )
            if row:
                history_rows.append(row)

    maturity_rows: list[dict[str, Any]] = []
    for strength in (-0.50, -0.25, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00):
        for pseudo in (0.5, 1.0, 2.0, 4.0, 8.0):
            row = evaluate(
                replace(
                    source_config,
                    profile_maturity_logit_strength=strength,
                    profile_maturity_pseudoseconds=pseudo,
                ),
                "STAGE_2_PROFILE_MATURITY",
                "Calibrate each speaker likelihood by causal accumulated profile speech instead of treating one-sentence and mature profiles equally.",
            )
            if row:
                maturity_rows.append(row)

    for history_parent in _rank(history_rows)[:5]:
        history_config = BayesSpeakerTrackerConfig(**history_parent["algorithm_config"])
        for maturity_parent in _rank(maturity_rows)[:5]:
            maturity_config = BayesSpeakerTrackerConfig(**maturity_parent["algorithm_config"])
            evaluate(
                replace(
                    history_config,
                    profile_maturity_logit_strength=maturity_config.profile_maturity_logit_strength,
                    profile_maturity_pseudoseconds=maturity_config.profile_maturity_pseudoseconds,
                ),
                "STAGE_3_COMBINED_PROFILE_EVIDENCE",
                "Combine a bounded centroid history with causal maturity-calibrated speaker likelihoods.",
                history_parent["candidate_id"],
            )

    write_state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress["status"] = "interrupted" if _STOP else "complete"
    progress["phase"] = "INTERRUPTED" if _STOP else "COMPLETE"
    _atomic_json(run_dir / "progress.json", progress)
    best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
    _atomic_json(run_dir / "final_report.json", {
        "status": progress["status"],
        "source_champion_score": source_score,
        "champion_score": best if incumbent else None,
        "score_delta": round(best - source_score, 6),
        "candidate_count": len(completed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(json.dumps(progress, indent=2, ensure_ascii=False))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
