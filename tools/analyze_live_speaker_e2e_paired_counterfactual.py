"""Compare two tracker artifacts on the exact worlds captured by real GUI runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_counterfactual import evaluate_counterfactual  # noqa: E402


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _counterfactual_config(artifact: dict) -> dict:
    config = dict(artifact.get("algorithm_config") or {})
    runtime = dict(artifact.get("expected_runtime_config") or {})
    # Browser hold is a runtime/reducer parameter rather than a Bayes dataclass
    # field.  Make it explicit for both arms so a tape captured under one arm
    # does not silently lend its recorded hold value to the other arm.
    if "live_speaker_probe_hold_seconds" in runtime:
        config["live_speaker_probe_hold_seconds"] = float(
            runtime["live_speaker_probe_hold_seconds"]
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--observation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_config = _counterfactual_config(_read(args.baseline_artifact))
    candidate_config = _counterfactual_config(_read(args.candidate_artifact))
    rows = []
    for observation_path in args.observation:
        observation = _read(observation_path)
        attestation = dict(observation.get("attestation") or {})
        tape_dir = Path(attestation["world_tape"]["output_dir"])
        canonical_path = Path(attestation["canonical"]["path"])
        baseline = evaluate_counterfactual(tape_dir, baseline_config, canonical_path)
        candidate = evaluate_counterfactual(tape_dir, candidate_config, canonical_path)
        baseline_score = float(baseline["strict_browser_live_score"])
        candidate_score = float(candidate["strict_browser_live_score"])
        rows.append(
            {
                "observation": str(observation_path.resolve()),
                "captured_arm": "candidate" if "\\B_" in str(observation_path.resolve()) else "baseline",
                "real_gui_score": float(observation["summary"]["strict_browser_live_score"]),
                "world_tape_run_id": attestation["world_tape"]["run_id"],
                "baseline_counterfactual_score": baseline_score,
                "candidate_counterfactual_score": candidate_score,
                "paired_counterfactual_delta": candidate_score - baseline_score,
            }
        )

    deltas = [row["paired_counterfactual_delta"] for row in rows]
    report = {
        "contract_id": "whospeaks.real_gui_world_paired_counterfactual.v1",
        "discovery_only": True,
        "production_promotion_eligible": False,
        "baseline_artifact": str(args.baseline_artifact.resolve()),
        "candidate_artifact": str(args.candidate_artifact.resolve()),
        "run_count": len(rows),
        "positive_direction_count": sum(value > 0.0 for value in deltas),
        "mean_paired_counterfactual_delta": mean(deltas),
        "median_paired_counterfactual_delta": median(deltas),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
