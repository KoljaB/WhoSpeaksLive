from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_browser_parity import replay_browser_state
from window.live_speaker_parity_replay import validate_and_replay_world_tape
from window.live_speaker_probe_scoring import read_canonical_segments


CAMPAIGN_PARITY_CONTRACT_ID = "whospeaks.live_world_tape.campaign_parity.v1"


def _observation_files(campaign_root: Path) -> list[Path]:
    return sorted((campaign_root / "browser_observations").glob("*.json"))


def _small_browser_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"recorded_samples", "replayed_samples"}
    }


def validate_campaign(
    campaign_root: Path,
    *,
    minimum_state_exact_ratio: float = 0.999,
    maximum_score_abs_delta: float = 0.01,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    runs: list[dict[str, Any]] = []
    for observation_path in _observation_files(root):
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        attestation = dict(observation.get("attestation") or {})
        world_tape = dict(attestation.get("world_tape") or {})
        tape_dir = Path(world_tape.get("output_dir") or "")
        canonical_path = Path(
            dict(attestation.get("canonical") or {}).get("path") or ""
        )
        tape_report = validate_and_replay_world_tape(tape_dir)
        browser_report = replay_browser_state(tape_dir)
        replay_score = score_browser_live_speaker_samples(
            browser_report["replayed_samples"],
            read_canonical_segments(canonical_path),
        )
        actual_score = float(
            dict(observation.get("summary") or {}).get("strict_browser_live_score")
            or 0.0
        )
        replayed_score = float(replay_score["strict_browser_live_score"])
        score_delta = replayed_score - actual_score
        state_exact = float(browser_report["current_speaker_exact_ratio"])
        run_valid = bool(
            tape_report.get("validation", {}).get("valid")
            and tape_report.get("server_core_replay", {}).get("exact_match")
            and state_exact >= minimum_state_exact_ratio
            and abs(score_delta) <= maximum_score_abs_delta
        )
        runs.append(
            {
                "video_id": dict(attestation.get("media") or {}).get("video_id"),
                "run_id": world_tape.get("run_id"),
                "tape_dir": str(tape_dir.resolve()),
                "observation_path": str(observation_path.resolve()),
                "canonical_path": str(canonical_path.resolve()),
                "valid": run_valid,
                "event_count": world_tape.get("event_count"),
                "server_core": tape_report.get("server_core_replay"),
                "browser_reducer": _small_browser_report(browser_report),
                "actual_strict_browser_live_score": actual_score,
                "replayed_strict_browser_live_score": replayed_score,
                "score_delta": score_delta,
            }
        )
    state_ratios = [
        float(item["browser_reducer"]["current_speaker_exact_ratio"])
        for item in runs
    ]
    score_deltas = [abs(float(item["score_delta"])) for item in runs]
    diagnostic_thresholds_pass = bool(runs) and all(
        bool(item["valid"]) for item in runs
    )
    # This Python reducer is deliberately a diagnostic scaffold.  The binding
    # optimization contract requires one pure JavaScript reducer shared by the
    # production browser and virtual-clock replay, so it cannot close the gate.
    baseline_parity = False
    return {
        "contract_id": CAMPAIGN_PARITY_CONTRACT_ID,
        "campaign_root": str(root),
        "run_count": len(runs),
        "thresholds": {
            "minimum_state_exact_ratio": minimum_state_exact_ratio,
            "maximum_score_abs_delta": maximum_score_abs_delta,
        },
        "reducer_kind": "python_diagnostic_approximation",
        "diagnostic_thresholds_pass": diagnostic_thresholds_pass,
        "baseline_parity": baseline_parity,
        "minimum_state_exact_ratio": min(state_ratios) if state_ratios else 0.0,
        "maximum_score_abs_delta": max(score_deltas) if score_deltas else None,
        "optimization_eligible": False,
        "eligibility_reason": (
            "The Python browser reducer is diagnostic only. The production browser and "
            "virtual-clock replay must share one pure JavaScript reducer, each Cunk run "
            "must reach 99.9% exact visible-speaker state, and counterfactual score "
            "deltas still require real-GUI predictive validation."
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate server-core, browser-state, and strict-score parity for a "
            "real-GUI World Tape campaign."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--minimum-state-exact-ratio", type=float, default=0.999)
    parser.add_argument("--maximum-score-abs-delta", type=float, default=0.01)
    args = parser.parse_args()
    report = validate_campaign(
        args.campaign_root,
        minimum_state_exact_ratio=args.minimum_state_exact_ratio,
        maximum_score_abs_delta=args.maximum_score_abs_delta,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["diagnostic_thresholds_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
