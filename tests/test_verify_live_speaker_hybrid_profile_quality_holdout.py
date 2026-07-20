from __future__ import annotations

from dataclasses import asdict
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid as sweep
import sweep_live_speaker_hybrid_profile_quality as profile_quality
import sweep_live_speaker_hybrid_round2 as round2
import verify_live_speaker_hybrid_profile_quality_holdout as verifier
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


DEV_IDS = tuple(f"development_{index}" for index in range(9))
HOLDOUT_IDS = tuple(sorted(verifier.EXPECTED_HOLDOUT_ID_SET))


def score(global_score: float, strict: float, wrong: float, ids=DEV_IDS) -> dict:
    return {
        "aggregate": {"global_score": global_score},
        "per_video": {
            video_id: {
                "strict_browser_live_score": strict,
                "wrong_live_speech_ratio": wrong,
            }
            for video_id in ids
        },
    }


def synthetic_locked_artifacts(root: Path) -> tuple[Path, dict, dict]:
    control_config = HybridSpeakerTrackerConfig(
        enable_young_profile_confirmation=True,
        enable_young_profile_lease=True,
        enable_short_scale_fast_lease=False,
        enable_profile_quality_short_scale_fast_lease=False,
        young_trusted_min_sentence_count=4,
        young_trusted_min_speech_seconds=8.0,
        young_min_similarity=0.35,
        young_min_margin=0.08,
        young_required_consecutive_probes=1,
        young_independent_scale_count=2,
        young_fast_independent_scale_count=1,
    )
    control, candidate = profile_quality._fixed_candidates(control_config)
    frozen_source = SimpleNamespace(
        candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
        family="young_profile_lease",
        champion_path=Path("/synthetic/run018/champion.json"),
        run_path=Path("/synthetic/run018/run.json"),
        champion_sha256="a" * 64,
        run_sha256="b" * 64,
        provider=sweep.DEFAULT_PROVIDER,
    )
    candidate_lock = profile_quality._candidate_lock(
        frozen_source, (control, candidate)
    )
    baseline = score(0.30, 0.30, 0.20)
    locked = {
        "candidate_id": control.candidate_id,
        "name": control.name,
        "family": control.family,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(control.config),
        **score(0.32, 0.32, 0.18),
    }
    winner = {
        "candidate_id": candidate.candidate_id,
        "name": candidate.name,
        "family": candidate.family,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(candidate.config),
        **score(0.34, 0.34, 0.16),
    }
    promotion = profile_quality._promotion_decision(baseline, locked, winner)
    winner["vs_baseline"] = promotion["candidate_vs_baseline"]
    winner["vs_locked"] = promotion["candidate_vs_locked"]
    winner["promotion_gates_passed"] = True
    run = {
        "schema_version": 1,
        "runner_id": profile_quality.RUNNER_ID,
        "status": "complete",
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "selection_policy": "fixed_profile_quality_candidate_no_search",
        "candidate_lock": candidate_lock,
        "dataset_sources": [{
            "label": "opened",
            "corpus_root": "/synthetic/corpus",
            "input_root": "/synthetic/inputs",
            "video_ids": list(DEV_IDS),
        }],
        "video_count": 9,
        "sealed_v3_opened": False,
        "provider": sweep.DEFAULT_PROVIDER,
        "windows_seconds": [0.8, 2.8],
        "long_weight": 0.25,
        "fresh_live_cost": round2._fresh_live_cost(),
        "winner": profile_quality.PROFILE_QUALITY_NAME,
    }
    champion = {
        "schema_version": 1,
        "status": verifier.INITIAL_CHAMPION_STATUS,
        "runner_id": profile_quality.RUNNER_ID,
        "candidate_lock_sha256": candidate_lock["lock_sha256"],
        "source_run018_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "winner": winner,
        "promotion": promotion,
        "production_defaults_changed": False,
        "fresh_live_verification_required": True,
        "sealed_v3_opened": False,
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
    return root, run, champion


def three_video_scores(global_score: float, strict: float, wrong: float) -> dict:
    return score(global_score, strict, wrong, HOLDOUT_IDS)


class IdentifierOrderingTests(unittest.TestCase):
    def test_exact_three_holdout_ids_are_accepted_without_opening_paths(self) -> None:
        raw = "v3=/never/open::/never/open::" + ",".join(HOLDOUT_IDS)
        sources = verifier._validate_holdout_sources([raw])
        self.assertEqual(sources[0].video_ids, HOLDOUT_IDS)

    def test_wrong_id_is_rejected_before_candidate_validation(self) -> None:
        wrong = list(HOLDOUT_IDS)
        wrong[-1] = "not_the_frozen_holdout"
        with patch.object(verifier, "_load_locked_profile_quality_candidate") as load:
            with self.assertRaisesRegex(ValueError, "frozen three-video seal"):
                verifier.main([
                    "--dataset-source",
                    "v3=/never/open::/never/open::" + ",".join(wrong),
                    "--profile-quality-run-dir",
                    "/also/never/open",
                ])
        load.assert_not_called()

    def test_duplicate_across_sources_is_rejected_before_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one dataset source"):
            verifier._validate_holdout_sources([
                f"a=/x::/y::{HOLDOUT_IDS[0]},{HOLDOUT_IDS[1]}",
                f"b=/x2::/y2::{HOLDOUT_IDS[1]},{HOLDOUT_IDS[2]}",
            ])


class CandidateValidationTests(unittest.TestCase):
    def test_exact_nine_video_gate_passed_candidate_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_profile_quality_candidate(root)
        self.assertEqual(locked.candidate_id, profile_quality._candidate_id(locked.config))
        self.assertTrue(locked.config.enable_profile_quality_short_scale_fast_lease)
        self.assertFalse(locked.config.enable_short_scale_fast_lease)
        self.assertEqual(locked.config.profile_quality_fast_lease_min_sentence_count, 2)
        self.assertEqual(locked.config.profile_quality_fast_lease_min_speech_seconds, 3.1)
        self.assertEqual(locked.config.profile_quality_fast_lease_min_similarity, 0.18)
        self.assertEqual(locked.config.profile_quality_fast_lease_min_margin, 0.06)

    def test_candidate_with_tampered_gate_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, champion = synthetic_locked_artifacts(Path(temporary))
            champion["promotion"]["candidate_vs_baseline"][
                "per_video_score_delta"
            ][DEV_IDS[0]] = -0.5
            (root / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nine-video proof failed"):
                verifier._load_locked_profile_quality_candidate(root)

    def test_candidate_with_any_sealed_id_in_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, run, champion = synthetic_locked_artifacts(Path(temporary))
            sealed = HOLDOUT_IDS[0]
            old = DEV_IDS[0]
            run["dataset_sources"][0]["video_ids"][0] = sealed
            winner = champion["winner"]
            winner["per_video"][sealed] = winner["per_video"].pop(old)
            for key in ("candidate_vs_baseline", "candidate_vs_locked"):
                proof = champion["promotion"][key]
                proof["per_video_score_delta"][sealed] = proof[
                    "per_video_score_delta"
                ].pop(old)
                proof["per_video_wrong_ratio_delta"][sealed] = proof[
                    "per_video_wrong_ratio_delta"
                ].pop(old)
            winner["vs_baseline"] = champion["promotion"]["candidate_vs_baseline"]
            winner["vs_locked"] = champion["promotion"]["candidate_vs_locked"]
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (root / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains sealed holdout IDs"):
                verifier._load_locked_profile_quality_candidate(root)


class HoldoutLockTests(unittest.TestCase):
    def test_lock_requires_exact_ids_and_order_and_records_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            raw = json.dumps({
                "video_ids": list(HOLDOUT_IDS),
                "selection": {"policy": "synthetic_fixed"},
            }).encode("utf-8")
            path.write_bytes(raw)
            lock = verifier._load_holdout_lock(path, HOLDOUT_IDS)
            self.assertEqual(lock.video_ids, HOLDOUT_IDS)
            self.assertEqual(lock.sha256, verifier._sha256_bytes(raw))
            with self.assertRaisesRegex(ValueError, "order differs"):
                verifier._load_holdout_lock(path, tuple(reversed(HOLDOUT_IDS)))


class GateTests(unittest.TestCase):
    def test_all_three_and_aggregate_score_wrong_gates_pass(self) -> None:
        baseline = three_video_scores(0.30, 0.30, 0.20)
        candidate = three_video_scores(0.32, 0.31, 0.19)
        gates = verifier._holdout_gates(baseline, candidate, HOLDOUT_IDS)
        self.assertTrue(gates["holdout_passed"])
        self.assertTrue(gates["all_three_individual_gates_passed"])
        self.assertTrue(gates["aggregate"]["passed"])
        self.assertEqual(len(gates["per_video"]), 3)

    def test_one_video_score_regression_fails_even_when_aggregate_improves(self) -> None:
        baseline = three_video_scores(0.30, 0.30, 0.20)
        candidate = three_video_scores(0.32, 0.31, 0.19)
        candidate["per_video"][HOLDOUT_IDS[1]]["strict_browser_live_score"] = 0.29
        gates = verifier._holdout_gates(baseline, candidate, HOLDOUT_IDS)
        self.assertFalse(gates["holdout_passed"])
        self.assertFalse(gates["per_video"][HOLDOUT_IDS[1]]["score_gate_passed"])

    def test_aggregate_wrong_regression_fails(self) -> None:
        baseline = three_video_scores(0.30, 0.30, 0.20)
        candidate = three_video_scores(0.32, 0.31, 0.206)
        gates = verifier._holdout_gates(baseline, candidate, HOLDOUT_IDS)
        self.assertFalse(gates["holdout_passed"])
        self.assertFalse(gates["aggregate"]["wrong_ratio_gate_passed"])


class ExecutionAndCommitTests(unittest.TestCase):
    def test_baseline_for_all_videos_precedes_the_only_candidate(self) -> None:
        order: list[str] = []
        baseline = three_video_scores(0.30, 0.30, 0.20)
        candidate_score = three_video_scores(0.32, 0.31, 0.19)
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_profile_quality_candidate(root)
            with patch.object(
                round2,
                "_prepare_baseline",
                side_effect=lambda _sources: (order.append("baseline") or {}, baseline),
            ), patch.object(
                round2,
                "_score_candidate",
                side_effect=lambda _prepared, _candidate: (
                    order.append("candidate") or candidate_score
                ),
            ) as score_candidate:
                found_baseline, found_candidate = verifier._evaluate_holdout([], locked)
        self.assertEqual(order, ["baseline", "candidate"])
        self.assertIs(found_baseline, baseline)
        self.assertIs(found_candidate, candidate_score)
        score_candidate.assert_called_once()

    def test_atomic_commit_updates_champion_and_refuses_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, champion = synthetic_locked_artifacts(Path(temporary))
            artifact_path = root / verifier.HOLDOUT_ARTIFACT_NAME
            artifact = {
                "passed": True,
                "holdout_lock": {"sha256": "c" * 64},
            }
            updated = verifier._commit_holdout(
                artifact_path, root / "champion.json", artifact, champion
            )
            self.assertTrue(artifact_path.exists())
            self.assertEqual(updated["status"], verifier.PASSED_CHAMPION_STATUS)
            self.assertTrue(updated["holdout_opened"])
            self.assertTrue(updated["sealed_v3_opened"])
            persisted = json.loads((root / "champion.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["holdout_lock_sha256"], "c" * 64)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                verifier._commit_holdout(
                    artifact_path, root / "champion.json", artifact, champion
                )

    def test_one_shot_guard_rejects_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_profile_quality_candidate(root)
            (root / verifier.HOLDOUT_ARTIFACT_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rerun refused"):
                verifier._assert_one_shot(locked)


class CostContractTests(unittest.TestCase):
    def test_verifier_has_no_tuning_arguments_and_exact_live_cost(self) -> None:
        cost = round2._fresh_live_cost()
        self.assertEqual(list(verifier.EXPECTED_WINDOWS), [0.8, 2.8])
        self.assertEqual(verifier.EXPECTED_LONG_WEIGHT, 0.25)
        self.assertEqual(cost["fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["max_fresh_window_requests_per_probe"], 2)
        self.assertEqual(sweep.DEFAULT_PROVIDER, (
            "pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5"
        ))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            verifier._parse_args([
                "--dataset-source", "x=/c::/i::" + ",".join(HOLDOUT_IDS),
                "--profile-quality-run-dir", "/run",
                "--score-tolerance", "1.0",
            ])


if __name__ == "__main__":
    unittest.main()
