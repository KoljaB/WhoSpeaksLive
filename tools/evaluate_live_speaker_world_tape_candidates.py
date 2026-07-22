from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from window.live_speaker_counterfactual import evaluate_counterfactual


CONTRACT_ID = "whospeaks.live_world_tape.discovery_candidates.v1"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_config(candidate: dict[str, Any], workspace: Path) -> dict[str, Any]:
    if isinstance(candidate.get("algorithm_config"), dict):
        config = dict(candidate["algorithm_config"])
    else:
        artifact_path = Path(str(candidate.get("base_artifact") or ""))
        if not artifact_path.is_absolute():
            artifact_path = workspace / artifact_path
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        config = dict(artifact.get("algorithm_config") or {})
    config.update(dict(candidate.get("patch") or {}))
    return config


def evaluate(
    campaign_root: Path,
    candidates_path: Path,
    *,
    video_ids: set[str] | None = None,
) -> dict[str, Any]:
    workspace = Path.cwd().resolve()
    root = Path(campaign_root).resolve()
    parity_report = json.loads(
        (root / "baseline_parity_report.json").read_text(encoding="utf-8")
    )
    candidate_specs = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    if isinstance(candidate_specs, dict):
        candidate_specs = candidate_specs.get("candidates")
    if not isinstance(candidate_specs, list) or not candidate_specs:
        raise ValueError("Candidate file must contain a non-empty candidates list")
    runs = [
        item
        for item in parity_report.get("runs") or []
        if not video_ids or str(item.get("video_id") or "") in video_ids
    ]
    if not runs:
        raise ValueError("No World Tape runs match the requested videos")

    results: list[dict[str, Any]] = []
    for position, spec in enumerate(candidate_specs, 1):
        if not isinstance(spec, dict):
            raise ValueError(f"Candidate {position} is not an object")
        name = str(spec.get("name") or f"candidate_{position}")
        config = _load_config(spec, workspace)
        run_scores: list[dict[str, Any]] = []
        for run in runs:
            replay = evaluate_counterfactual(
                Path(run["tape_dir"]), config, Path(run["canonical_path"])
            )
            run_scores.append(
                {
                    "video_id": run["video_id"],
                    "run_id": run["run_id"],
                    "score": float(replay["strict_browser_live_score"]),
                    "projected_live_action_count": replay[
                        "projected_live_action_count"
                    ],
                    "correct_live_speaker_coverage": float(
                        replay["score"]["correct_live_speaker_coverage"]
                    ),
                    "wrong_live_speech_ratio": float(
                        replay["score"]["wrong_live_speech_ratio"]
                    ),
                    "missing_live_speech_ratio": float(
                        replay["score"]["missing_live_speech_ratio"]
                    ),
                    "outside_speech_live_ratio": float(
                        replay["score"]["outside_speech_live_ratio"]
                    ),
                    "correct_live_precision_during_speech": float(
                        replay["score"]["correct_live_precision_during_speech"]
                    ),
                }
            )
        per_video: dict[str, list[float]] = {}
        for item in run_scores:
            per_video.setdefault(str(item["video_id"]), []).append(float(item["score"]))
        video_scores = {
            video_id: mean(scores) for video_id, scores in sorted(per_video.items())
        }
        results.append(
            {
                "name": name,
                "hypothesis": str(spec.get("hypothesis") or ""),
                "algorithm_config_sha256": _stable_hash(config),
                "algorithm_config": config,
                "macro_score": mean(video_scores.values()),
                "per_video": video_scores,
                "runs": run_scores,
            }
        )
    incumbent = next(
        (item for item in results if item["name"] == "incumbent"), results[0]
    )
    incumbent_score = float(incumbent["macro_score"])
    for item in results:
        item["delta_vs_incumbent"] = float(item["macro_score"]) - incumbent_score
    results.sort(key=lambda item: float(item["macro_score"]), reverse=True)
    return {
        "contract_id": CONTRACT_ID,
        "campaign_root": str(root),
        "candidate_file": str(Path(candidates_path).resolve()),
        "video_ids": sorted({str(item["video_id"]) for item in runs}),
        "run_count": len(runs),
        "candidate_count": len(results),
        "selection_score": "plain macro mean of per-video mean strict browser score",
        "discovery_only": True,
        "production_promotion_eligible": False,
        "candidates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate tracker candidates through authentic World-Tape timing."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--videos", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    videos = {item.strip() for item in args.videos.split(",") if item.strip()}
    report = evaluate(
        args.campaign_root,
        args.candidates,
        video_ids=videos or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "name": item["name"],
                    "score": round(float(item["macro_score"]), 6),
                    "delta": round(float(item["delta_vs_incumbent"]), 6),
                    "per_video": {
                        key: round(float(value), 6)
                        for key, value in item["per_video"].items()
                    },
                }
                for item in report["candidates"]
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
