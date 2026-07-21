"""Search causal online provisional speaker discovery around the Bayes champion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from optimize_live_speaker_bayes_top7 import _compact
from optimize_live_speaker_overnight_top7 import Dataset
from window.live_speaker_bayes import BAYES_ALGORITHM_ID, BayesSpeakerTrackerConfig, replay_cached_bayes_windows
from window.live_speaker_benchmark import PRIMARY_SCORER_V2_ID, aggregate_video_scores_primary_v2, score_live_speaker_decisions

OPTIMIZER_ID = "live_speaker_top7_bayes_provisional_discovery_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _id(config: BayesSpeakerTrackerConfig, windows: tuple[float, ...]) -> str:
    payload = json.dumps({"optimizer": OPTIMIZER_ID, "algorithm": BAYES_ALGORITHM_ID,
                          "scorer": PRIMARY_SCORER_V2_ID, "windows": windows,
                          "config": asdict(config)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--focused-limits", action="store_true")
    parser.add_argument("--finalized-limit-only", action="store_true")
    parser.add_argument("--adaptive-count", action="store_true")
    parser.add_argument("--discovery-agreement", action="store_true")
    parser.add_argument("--merge-refine", action="store_true")
    parser.add_argument("--profile-stage-count", action="store_true")
    parser.add_argument("--champion-retune", action="store_true")
    parser.add_argument("--window-geometry", action="store_true")
    parser.add_argument("--provisional-hold-agreement", action="store_true")
    parser.add_argument("--agreement-matrix-refine", action="store_true")
    parser.add_argument("--incumbent-hold-agreement", action="store_true")
    parser.add_argument("--merge-recency", action="store_true")
    parser.add_argument("--short-long-crossover", action="store_true")
    parser.add_argument("--online-prototype-bank", action="store_true")
    parser.add_argument("--provisional-reactivation", action="store_true")
    parser.add_argument("--provisional-expiry", action="store_true")
    parser.add_argument("--bounded-provisional-pool", action="store_true")
    parser.add_argument("--bounded-provisional-pool-refine", action="store_true")
    parser.add_argument("--bounded-provisional-pool-fine", action="store_true")
    parser.add_argument("--bounded-pool-mixture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    deadline = started + max(1, args.budget_seconds)
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    source = json.loads(args.champion.read_text(encoding="utf-8-sig"))
    videos = [str(value) for value in spec["videos"]]
    windows = tuple(float(value) for value in source["windows_seconds"])
    provider, profile_name = str(source["provider_spec"]), str(source["profile_name"])
    source_score = float(source["candidate_score"])
    source_config = BayesSpeakerTrackerConfig.from_mapping(source["algorithm_config"])
    dataset = Dataset(args.corpus_root.resolve(), args.input_root.resolve(), provider, profile_name)
    gate_root = args.gate_root.resolve() if args.gate_root is not None else None
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
    incumbent = max(completed.values(), key=lambda row: row["aggregate"]["primary_score"], default=None)
    phase_counts: dict[str, int] = {}
    for row in completed.values():
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    def state(phase: str, active: str = "") -> None:
        best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
        _atomic(run_dir / "progress.json", {
            "status": "interrupted" if _STOP else "running", "phase": phase, "active": active,
            "completed_candidate_count": len(completed), "phase_counts": phase_counts,
            "source_champion_score": source_score, "best_provisional_score": best if incumbent else None,
            "score_delta": round(best - source_score, 6),
            "best_candidate_id": incumbent["candidate_id"] if incumbent else None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        if incumbent:
            gate_metadata = {
                key: source[key]
                for key in (
                    "gate_variant", "vad_speech_rms_threshold",
                    "live_speaker_release_rms_threshold",
                    "live_speaker_clear_window_seconds",
                    "live_speaker_probe_min_speech_seconds", "release_every_tick",
                )
                if key in source
            }
            _atomic(run_dir / "champion.json", {
                "status": "CACHE_PROVISIONAL_BAYES_WINNER_PENDING_FRESH_LIVE" if best > source_score else "BELOW_SOURCE_CHAMPION",
                "selection_policy": "primary_score_only_no_per_video_vetoes",
                "source_champion_score": source_score, "candidate_score": best,
                "score_delta": round(best - source_score, 6), "provider_spec": provider,
                "profile_name": profile_name,
                "windows_seconds": list(incumbent.get("windows_seconds") or windows),
                **gate_metadata,
                **incumbent, "fresh_live_verified": False,
            })

    def evaluate(
        config: BayesSpeakerTrackerConfig,
        phase: str,
        hypothesis: str,
        parent: str | None = None,
        candidate_windows: tuple[float, ...] | None = None,
    ) -> dict[str, Any] | None:
        nonlocal incumbent
        chosen_windows = tuple(candidate_windows or windows)
        config = replace(config, scale_windows=chosen_windows)
        cid = _id(config, chosen_windows)
        if cid in completed:
            return completed[cid]
        if _STOP or time.monotonic() >= deadline:
            return None
        state(phase, cid)
        per_video: dict[str, Any] = {}
        for video_id in videos:
            inputs = dataset.video_inputs(video_id, min(chosen_windows))
            if gate_root is not None:
                video_gate = gate_root / video_id
                speech = np.load(video_gate / "speech_gate.u1.npy", allow_pickle=False)
                probes = np.load(video_gate / "probe_schedule.u1.npy", allow_pickle=False)
                releases = np.load(video_gate / "release_gate.u1.npy", allow_pickle=False)
            else:
                speech, probes, releases = inputs["speech"], inputs["probes"], inputs["releases"]
            decisions = replay_cached_bayes_windows(
                [dataset.block(video_id, window) for window in chosen_windows], inputs["profiles"],
                speech, probes, releases, config=config,
            )
            per_video[video_id] = _compact(score_live_speaker_decisions(decisions, inputs["canonical"], inputs["profiles"]))
        aggregate = aggregate_video_scores_primary_v2(per_video.values())
        row = {"candidate_id": cid, "phase": phase, "hypothesis": hypothesis,
               "parent_candidate_id": parent, "algorithm_config": asdict(config),
               "windows_seconds": list(chosen_windows),
               "aggregate": aggregate, "per_video": per_video,
               "score_delta_vs_source": round(float(aggregate["primary_score"]) - source_score, 6),
               "elapsed_seconds": round(time.monotonic() - started, 6)}
        completed[cid] = row
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if incumbent is None or aggregate["primary_score"] > incumbent["aggregate"]["primary_score"]:
            incumbent = row
        state(phase, cid)
        return row

    _atomic(run_dir / "run.json", {"optimizer_id": OPTIMIZER_ID, "algorithm_id": BAYES_ALGORITHM_ID,
        "primary_scorer_id": PRIMARY_SCORER_V2_ID, "source_champion": str(args.champion.resolve()),
        "source_champion_score": source_score, "videos": videos,
        "gate_root": str(gate_root) if gate_root is not None else None,
        "maximum_fresh_windows_per_probe": 2, "selection_policy": "one Top-7 primary score"})
    state("SOURCE_CHAMPION")
    if args.finalized_limit_only:
        for limit in (-1, 0, 1, 2, 3, 4, 5, 6):
            evaluate(
                replace(
                    source_config,
                    provisional_creation_max_finalized_profiles=limit,
                ),
                "STAGE_23_FINALIZED_PROFILE_DISCOVERY_LIMIT",
                "Stop minting unmatched provisional identities once enough finalized causal speaker profiles already exist.",
            )
    elif args.bounded_pool_mixture:
        for size in (1, 2, 3, 5):
            for weight in (0.10, 0.25, 0.50, 0.75, 1.0):
                for alpha in (0.0, 0.10, 0.25, 0.50, 0.75):
                    evaluate(replace(source_config,
                        provisional_pool_overflow_prototype_bank_size=size,
                        provisional_pool_overflow_prototype_weight=weight,
                        provisional_pool_overflow_update_alpha=alpha),
                        "STAGE_22_BOUNDED_POOL_MIXTURE",
                        "Represent a reused full-pool identity as a bounded causal mixture of its original voice and the unmatched overflow cluster instead of collapsing both into one centroid.")
    elif args.bounded_provisional_pool_fine:
        for alpha_index in range(17):
            alpha = round(0.30 + 0.025 * alpha_index, 3)
            for ceiling_index in range(13):
                ceiling = round(-0.025 + 0.025 * ceiling_index, 3)
                evaluate(replace(source_config,
                    provisional_pool_overflow_update_alpha=alpha,
                    provisional_creation_similarity_ceiling=ceiling),
                    "STAGE_21_BOUNDED_POOL_FINE",
                    "Jointly refine cluster adaptation and new-cluster gating after bounded causal clustering produced a verified global gain.")
    elif args.bounded_provisional_pool_refine:
        for count in (3, 4, 5):
            for strategy in ("recent", "closest", "visible"):
                for alpha in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0):
                    evaluate(replace(source_config,
                        provisional_max_active_count=count,
                        provisional_pool_overflow_strategy=strategy,
                        provisional_pool_overflow_update_alpha=alpha),
                        "STAGE_20_BOUNDED_POOL_REFINE",
                        "Refine how a full provisional cluster pool reuses or adapts an existing causal identity instead of fragmenting into another label.")
    elif args.bounded_provisional_pool:
        for count in (1, 2, 3, 4, 5, 6):
            evaluate(replace(source_config, provisional_max_active_count=count),
                "STAGE_19_BOUNDED_PROVISIONAL_POOL",
                "Bound unresolved online voice clusters and reuse the most recent identity until a final sentence profile stabilizes it, preventing early-ID fragmentation.")
    elif args.provisional_expiry:
        for seconds in (0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 60.0):
            evaluate(replace(source_config, provisional_expiry_seconds=seconds),
                "STAGE_18_PROVISIONAL_EXPIRY",
                "Retire unmatched provisional identities after causal inactivity so intro noise and abandoned clusters cannot pollute later assignments.")
    elif args.provisional_reactivation:
        for threshold in (-1.0, -0.75, -0.50, -0.30, -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10):
            evaluate(replace(source_config,
                provisional_reactivation_min_similarity=threshold),
                "STAGE_17_PROVISIONAL_REACTIVATION",
                "Reuse a causally discovered dormant voice cluster before minting another provisional identity for the same returning speaker.")
    elif args.online_prototype_bank:
        for size in (1, 2, 3, 5, 8):
            for weight in (0.10, 0.25, 0.50, 0.75, 1.0):
                for min_similarity in (-0.20, 0.0, 0.10, 0.20, 0.30, 0.40):
                    if size == 1 and min_similarity != -0.20:
                        continue
                    evaluate(replace(source_config,
                        provisional_prototype_bank_size=size,
                        provisional_prototype_weight=weight,
                        provisional_prototype_update_min_similarity=min_similarity),
                        "STAGE_16_ONLINE_PROTOTYPE_BANK",
                        "Retain a bounded bank of causal short-window voice prototypes so live probes are not forced to match only sentence-length final centroids.")
    elif args.short_long_crossover:
        for margin in (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
            for similarity in (0.10, 0.15, 0.175, 0.20, 0.25, 0.30):
                for count in (1, 2, 3):
                    evaluate(replace(source_config,
                        short_long_crossover_min_margin=margin,
                        short_long_crossover_min_similarity=similarity,
                        short_long_crossover_count=count),
                        "STAGE_15_SHORT_LONG_CROSSOVER",
                        "Switch on repeated short-window evidence while the slower context window still identifies the incumbent speaker, analogous to a causal moving-average crossover.")
    elif args.merge_recency:
        for weight in (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
            for seconds in (0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0):
                evaluate(replace(source_config,
                    provisional_merge_recency_weight=weight,
                    provisional_merge_recency_seconds=seconds),
                    "STAGE_14_CAUSAL_MERGE_RECENCY",
                    "Prefer the recently active provisional voice when a finalized sentence profile can plausibly merge with several online identities.")
    elif args.incumbent_hold_agreement:
        for threshold in (-1.0, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.85, 0.90, 0.95):
            for release_count in (1, 2, 3, 4):
                evaluate(replace(source_config,
                    incumbent_hold_scale_agreement_min_similarity=threshold,
                    unknown_release_count=release_count),
                    "STAGE_13_INCUMBENT_CROSSOVER_RELEASE",
                    "Treat a short/long coherence collapse as causal evidence that the incumbent speaker has stopped or changed.")
    elif args.agreement_matrix_refine:
        agreement_values = tuple(round(0.45 + index * 0.025, 3) for index in range(11))
        for creation_threshold in agreement_values:
            for assignment_threshold in agreement_values:
                evaluate(replace(source_config,
                    provisional_scale_agreement_min_similarity=creation_threshold,
                    provisional_assignment_scale_agreement_min_similarity=assignment_threshold),
                    "STAGE_12_AGREEMENT_MATRIX_REFINE",
                    "Jointly refine creation and continuation coherence after both independent mechanisms improved Fresh-LIVE.")
    elif args.provisional_hold_agreement:
        for threshold in (-1.0, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.85, 0.90, 0.95):
            for release_count in (1, 2, 3, 4):
                evaluate(replace(source_config,
                    provisional_assignment_scale_agreement_min_similarity=threshold,
                    unknown_release_count=release_count),
                    "STAGE_11_PROVISIONAL_CONTINUATION_AGREEMENT",
                    "Require coherent short/long evidence while an identity is still provisional, not only when creating it.")
    elif args.window_geometry:
        geometry_rows: list[dict[str, Any]] = []
        for short_window in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            for long_window in (1.3, 1.5, 1.7, 2.0, 2.3, 2.6, 2.9):
                if long_window <= short_window:
                    continue
                row = evaluate(source_config, "STAGE_9_WINDOW_GEOMETRY",
                    "Re-evaluate the two-window geometry after scale agreement became an explicit causal discovery signal.",
                    candidate_windows=(short_window, long_window))
                if row:
                    geometry_rows.append(row)
        for parent in sorted(geometry_rows, key=lambda row: row["aggregate"]["primary_score"], reverse=True)[:6]:
            parent_config = BayesSpeakerTrackerConfig(**parent["algorithm_config"])
            parent_windows = tuple(float(value) for value in parent["windows_seconds"])
            for long_weight in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
                evaluate(replace(parent_config, scale_weights=(1.0 - long_weight, long_weight)),
                    "STAGE_10_WINDOW_WEIGHT_REFINE",
                    "Recalibrate identity evidence weights for the best discovery-window geometries.",
                    str(parent["candidate_id"]), parent_windows)
    elif args.champion_retune:
        current_config = source_config
        current_score = source_score
        coordinates: list[tuple[str, tuple[Any, ...]]] = [
            ("provisional_scale_agreement_min_similarity", tuple(round(0.55 + index * 0.025, 3) for index in range(13))),
            ("min_similarity", (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30)),
            ("similarity_temperature", (0.05, 0.06, 0.07, 0.08, 0.0875, 0.095, 0.105, 0.12, 0.14)),
            ("min_known_probability", (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)),
            ("low_profile_unknown_bias", (-1.50, -1.25, -1.0, -0.75, -0.50)),
            ("high_profile_unknown_bias", (-0.50, -0.25, 0.0, 0.25, 0.50)),
            ("unknown_release_count", (1, 2, 3, 4)),
            ("silence_release_count", (1, 2, 3, 4)),
        ]
        for pass_index in range(3):
            improved = False
            for field_name, values in coordinates:
                rows: list[dict[str, Any]] = []
                for value in values:
                    row = evaluate(replace(current_config, **{field_name: value}),
                        f"STAGE_8_COORDINATE_PASS_{pass_index + 1}",
                        f"Recalibrate {field_name} after provisional speaker discovery changed the decision distribution.")
                    if row:
                        rows.append(row)
                if rows:
                    best = max(rows, key=lambda row: float(row["aggregate"]["primary_score"]))
                    best_score = float(best["aggregate"]["primary_score"])
                    if best_score > current_score + 1e-9:
                        current_config = BayesSpeakerTrackerConfig(**best["algorithm_config"])
                        current_score = best_score
                        improved = True
            if not improved:
                break
    elif args.profile_stage_count:
        for threshold in (0, 1, 2, 3, 4, 5):
            for later_count in (1, 2, 3, 4, 5, 6, 8, 10):
                evaluate(replace(source_config,
                    provisional_later_creation_profile_threshold=threshold,
                    provisional_later_creation_count=later_count),
                    "STAGE_7_PROFILE_STAGE_CONFIRMATION",
                    "Raise new-speaker confirmation only after a chosen number of final profiles already exists.")
    elif args.merge_refine:
        for later_count in (1, 2, 3):
            for ceiling in (-0.10, -0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
                for merge in (-1.0, -0.75, -0.50, -0.30, -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05):
                    evaluate(replace(source_config,
                        provisional_later_creation_count=later_count,
                        provisional_creation_similarity_ceiling=ceiling,
                        provisional_merge_min_similarity=merge),
                        "STAGE_6_MERGE_CONTINUITY_REFINE",
                        "Refine causal provisional-to-final identity continuity after low merge thresholds improved the global score.")
    elif args.discovery_agreement:
        agreement_values = (-1.0, -0.50, -0.20, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
        for scale_agreement in agreement_values:
            for temporal_consistency in agreement_values:
                evaluate(replace(source_config,
                    provisional_scale_agreement_min_similarity=scale_agreement,
                    provisional_temporal_consistency_min_similarity=temporal_consistency),
                    "STAGE_5_DISCOVERY_AGREEMENT",
                    "Create a provisional speaker only when the existing scales and repeated causal probes describe a coherent voice.")
    elif args.adaptive_count:
        for later_count in (1, 2, 3, 4, 5, 6, 8):
            for ceiling in (-0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15):
                for merge in (-0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15):
                    evaluate(replace(source_config, enable_provisional_profiles=True,
                        provisional_creation_count=1,
                        provisional_later_creation_count=later_count,
                        provisional_creation_similarity_ceiling=ceiling,
                        provisional_merge_min_similarity=merge,
                        provisional_update_alpha=0.0),
                        "STAGE_4_ADAPTIVE_DISCOVERY_CONFIRMATION",
                        "Acquire the first provisional speaker immediately but require more repeated evidence for later new speakers.")
    elif args.focused_limits:
        for limit in (-1, 0, 1, 2, 3, 4, 5):
            for count in (1, 2):
                for ceiling in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
                    for merge in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30):
                        evaluate(replace(source_config, enable_provisional_profiles=True,
                            provisional_creation_max_finalized_profiles=limit,
                            provisional_creation_count=count,
                            provisional_creation_similarity_ceiling=ceiling,
                            provisional_merge_min_similarity=merge),
                            "STAGE_3_DISCOVERY_SCOPE_REFINE",
                            "Limit online discovery by finalized-profile count and refine low creation/merge thresholds.")
    else:
        coarse: list[dict[str, Any]] = []
        for count in (1, 2, 3, 4):
            for ceiling in (0.10, 0.20, 0.30, 0.40, 0.50):
                for merge in (0.10, 0.25, 0.40, 0.55, 0.70):
                    row = evaluate(replace(source_config, enable_provisional_profiles=True,
                        provisional_creation_count=count,
                        provisional_creation_similarity_ceiling=ceiling,
                        provisional_merge_min_similarity=merge,
                        provisional_update_alpha=0.0),
                        "STAGE_1_PROVISIONAL_DISCOVERY",
                        "Create a causal unnamed speaker from repeated unmatched live windows and merge it when a final profile arrives.")
                    if row:
                        coarse.append(row)
        for parent in sorted(coarse, key=lambda row: row["aggregate"]["primary_score"], reverse=True)[:10]:
            config = BayesSpeakerTrackerConfig(**parent["algorithm_config"])
            for alpha in (0.02, 0.05, 0.10, 0.20):
                evaluate(replace(config, provisional_update_alpha=alpha), "STAGE_2_PROVISIONAL_ADAPTATION",
                         "Adapt an unmatched provisional centroid slowly from later causal short windows.", parent["candidate_id"])
    state("COMPLETE")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8-sig"))
    progress.update(status="interrupted" if _STOP else "complete", phase="INTERRUPTED" if _STOP else "COMPLETE")
    _atomic(run_dir / "progress.json", progress)
    best = float(incumbent["aggregate"]["primary_score"]) if incumbent else source_score
    _atomic(run_dir / "final_report.json", {"status": progress["status"], "source_champion_score": source_score,
        "champion_score": best if incumbent else None, "score_delta": round(best - source_score, 6),
        "candidate_count": len(completed), "elapsed_seconds": round(time.monotonic() - started, 3)})
    print(json.dumps(progress, indent=2))
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
