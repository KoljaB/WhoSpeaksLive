"""Evaluate two bounded causal turn-attack mechanisms on exact World Tapes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from analyze_live_speaker_open_set_tracklets import (
    TrackletConfig,
    _evaluate_variant,
    _load_base_config,
    _prepare_tape,
)


SEARCH = frozenset({"20v1OxUXcQY", "JWS-qfR6K3w", "pD4IdQTmneI"})
VALIDATION = frozenset({"L-CfFo5aQGU", "S_o3y7CzDUY", "mBeT_AoCXvc", "onHUfyRP1BE"})


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _macro(result: dict[str, Any], videos: frozenset[str]) -> float:
    return mean(float(value) for key, value in result["per_video"].items() if key in videos)


def _decorate(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result["delta"] = float(result["macro_score"]) - float(baseline["macro_score"])
    result["per_video_delta"] = {
        key: float(value) - float(baseline["per_video"][key])
        for key, value in result["per_video"].items()
    }
    result["search_score"] = _macro(result, SEARCH)
    result["search_delta"] = result["search_score"] - _macro(baseline, SEARCH)
    result["validation_score"] = _macro(result, VALIDATION)
    result["validation_delta"] = result["validation_score"] - _macro(baseline, VALIDATION)
    result["problem_floor_delta"] = min(
        result["per_video_delta"]["20v1OxUXcQY"],
        result["per_video_delta"]["JWS-qfR6K3w"],
    )
    return result


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
    tapes = [_prepare_tape(run, base_config) for run in parity.get("runs") or []]
    parent = json.loads(args.exclusive_result.read_text(encoding="utf-8-sig"))
    baseline_config = TrackletConfig(**parent["exclusive"]["config"])
    variants = [
        baseline_config,
        replace(
            baseline_config,
            name="short_attack_quarantine_no_identity_030",
            short_attack_novelty=True,
            short_attack_novelty_ceiling=0.30,
            short_attack_preempts_tracklet_reuse=True,
            enable_temporary_identity=False,
        ),
    ]
    baseline = _evaluate_variant(tapes, variants[0])
    expected = float(parent["exclusive"]["macro_score"])
    if abs(float(baseline["macro_score"]) - expected) > 1e-9:
        raise RuntimeError(f"Baseline drift: {baseline['macro_score']} != {expected}")
    results = [_decorate(_evaluate_variant(tapes, item), baseline) for item in variants[1:]]
    output = {
        "status": "REPLAY_ONLY_CAUSAL_RESEARCH_NOT_PROMOTION_EVIDENCE",
        "production_promotion_eligible": False,
        "model_inference_performed": False,
        "canonical_used_inside_inference": False,
        "mechanism": (
            "0.7-second short window opens a two-probe novelty candidate while "
            "the 1.5-second context is allowed to remain stale; embeddings/provider unchanged"
        ),
        "split": {"search": sorted(SEARCH), "validation": sorted(VALIDATION)},
        "baseline": baseline,
        "results": results,
        "config_sha256": {item.name: _hash(asdict(item)) for item in variants},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline["macro_score"], "results": [{"name": item["name"], "macro": item["macro_score"], "delta": item["delta"], "search_delta": item["search_delta"], "validation_delta": item["validation_delta"], "floor": item["problem_floor_delta"], "per_video_delta": item["per_video_delta"]} for item in results], "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
