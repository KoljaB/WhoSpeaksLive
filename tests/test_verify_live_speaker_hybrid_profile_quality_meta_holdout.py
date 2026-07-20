from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import asdict
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

import sweep_live_speaker_hybrid_profile_quality_meta as meta
import sweep_live_speaker_hybrid_round2 as round2
import verify_live_speaker_hybrid_profile_quality_meta_holdout as verifier
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


DEV_IDS = tuple(meta.EXPECTED_VIDEO_IDS)
HOLDOUT_IDS = verifier.EXPECTED_HOLDOUT_IDS


def run018_config(**overrides: object) -> HybridSpeakerTrackerConfig:
    values: dict[str, object] = {
        "enable_young_profile_confirmation": True,
        "enable_young_profile_lease": True,
        "enable_short_scale_fast_lease": False,
        "enable_profile_quality_short_scale_fast_lease": False,
        "enable_profile_quality_meta_lease": False,
        "young_trusted_min_sentence_count": 4,
        "young_trusted_min_speech_seconds": 8.0,
        "young_min_similarity": 0.35,
        "young_min_margin": 0.08,
        "young_required_consecutive_probes": 1,
        "young_independent_scale_count": 2,
        "young_fast_independent_scale_count": 1,
    }
    values.update(overrides)
    return HybridSpeakerTrackerConfig(**values)


def scores(
    global_score: float,
    strict: float,
    wrong: float,
    video_ids: tuple[str, ...] = DEV_IDS,
) -> dict:
    return {
        "aggregate": {"global_score": global_score},
        "per_video": {
            video_id: {
                "strict_browser_live_score": strict,
                "wrong_live_speech_ratio": wrong,
                "missing_live_speech_ratio": 0.2,
                "correct_live_speaker_coverage": 0.6,
                "correct_live_precision_during_speech": 0.8,
            }
            for video_id in video_ids
        },
    }


def synthetic_sources() -> list:
    raw = [
        f"{label}=/synthetic/corpus/{label}::/synthetic/input/{label}::"
        + ",".join(meta.EXPECTED_SOURCE_VIDEO_IDS[label])
        for label in meta.EXPECTED_SOURCE_ORDER
    ]
    return meta._validate_sources(raw)


def synthetic_locked_artifacts(root: Path) -> tuple[Path, dict, dict]:
    config = run018_config()
    source = SimpleNamespace(
        candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
        family="young_profile_lease",
        champion_path=Path("/synthetic/run018/champion.json"),
        run_path=Path("/synthetic/run018/run.json"),
        run_dir=Path("/synthetic/run018"),
        champion_sha256="a" * 64,
        run_sha256="b" * 64,
        provider=meta.EXPECTED_PROVIDER,
        config=config,
    )
    sources = synthetic_sources()
    source_lock = meta._dataset_source_lock(sources)
    control, contender = meta._fixed_candidates(config)
    candidate_lock = meta._candidate_lock(
        source, (control, contender), source_lock
    )
    baseline = scores(0.30, 0.30, 0.20)
    locked = {
        "name": control.name,
        "family": control.family,
        "candidate_id": control.candidate_id,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(control.config),
        **scores(0.34, 0.34, 0.16),
    }
    winner = {
        "name": contender.name,
        "family": contender.family,
        "candidate_id": contender.candidate_id,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": asdict(contender.config),
        **scores(0.35, 0.35, 0.15),
    }
    promotion = meta._promotion_decision(baseline, locked, winner)
    winner["vs_baseline"] = promotion["candidate_vs_baseline"]
    winner["vs_locked"] = promotion["candidate_vs_locked"]
    winner["promotion_gates_passed"] = True
    run = meta._run_metadata(sources, source_lock, candidate_lock)
    run.update(
        {
            "status": "complete",
            "baseline_completed_before_candidate_evaluation": True,
            "winner": meta.META_NAME,
        }
    )
    champion = {
        "schema_version": 1,
        "status": verifier.INITIAL_CHAMPION_STATUS,
        "runner_id": meta.RUNNER_ID,
        "dataset_source_lock_sha256": source_lock["lock_sha256"],
        "candidate_lock_sha256": candidate_lock["lock_sha256"],
        "source_run018_candidate_id": round2.LOCKED_RUN018_CANDIDATE_ID,
        "winner": winner,
        "promotion": promotion,
        "production_defaults_changed": False,
        "fresh_live_verification_required": True,
        "parameter_search_performed": False,
        "sealed_v4_opened": False,
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
    return root, run, champion


def synthetic_holdout_lock_bytes() -> bytes:
    return json.dumps(
        {
            "locked_before_round3_algorithm_analysis": True,
            "development_use_forbidden": True,
            "video_ids": list(HOLDOUT_IDS),
            "selection": [{"video_id": value} for value in HOLDOUT_IDS],
            "already_opened_video_ids": list(DEV_IDS),
            "live_cost_contract": {
                "max_fresh_embedding_windows_per_probe": 2,
                "window_seconds": [0.8, 2.8],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


class IdentifierOrderingTests(unittest.TestCase):
    def test_exact_three_ids_and_order_are_accepted_without_opening_paths(self) -> None:
        sources = verifier._validate_holdout_sources(
            ["v4=/never/open::/never/open::" + ",".join(HOLDOUT_IDS)]
        )
        self.assertEqual(sources[0].video_ids, HOLDOUT_IDS)

    def test_reordered_or_changed_id_is_rejected_before_candidate_load(self) -> None:
        for ids in (tuple(reversed(HOLDOUT_IDS)), (*HOLDOUT_IDS[:2], "not_frozen")):
            with self.subTest(ids=ids), patch.object(
                verifier, "_load_locked_meta_candidate"
            ) as load:
                with self.assertRaisesRegex(ValueError, "IDs or order"):
                    verifier.main(
                        [
                            "--dataset-source",
                            "v4=/must/not/open::/must/not/open::" + ",".join(ids),
                            "--meta-run-dir",
                            "/also/must/not/open",
                        ]
                    )
            load.assert_not_called()

    def test_duplicate_across_sources_is_rejected_before_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one dataset source"):
            verifier._validate_holdout_sources(
                [
                    f"a=/x::/y::{HOLDOUT_IDS[0]},{HOLDOUT_IDS[1]}",
                    f"b=/x2::/y2::{HOLDOUT_IDS[1]},{HOLDOUT_IDS[2]}",
                ]
            )


class CandidateValidationTests(unittest.TestCase):
    def test_exact_twelve_video_gate_passed_meta_candidate_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_meta_candidate(root)
        self.assertEqual(locked.candidate_id, meta._candidate_id(locked.config))
        self.assertTrue(locked.config.enable_profile_quality_meta_lease)
        self.assertFalse(locked.config.enable_profile_quality_short_scale_fast_lease)
        self.assertEqual(locked.config.profile_quality_meta_fresh_min_age_seconds, 0.05)
        self.assertEqual(locked.config.profile_quality_meta_switch_min_short_margin, 0.35)

    def test_tampered_twelve_video_gate_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, champion = synthetic_locked_artifacts(Path(temporary))
            champion["promotion"]["candidate_vs_baseline"][
                "per_video_score_delta"
            ][DEV_IDS[0]] = -0.5
            (root / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "12-video proof failed"):
                verifier._load_locked_meta_candidate(root)

    def test_noncanonical_development_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, run, _champion = synthetic_locked_artifacts(Path(temporary))
            run["dataset_sources"][0]["video_ids"][0] = HOLDOUT_IDS[0]
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen 12-video cohort"):
                verifier._load_locked_meta_candidate(root)

    def test_provider_window_weight_and_max_two_contracts_are_strict(self) -> None:
        mutations = (
            ("provider", "another", "another provider"),
            ("windows_seconds", [0.8, 1.5, 2.8], "exactly 0.8/2.8"),
            ("long_weight", 0.5, "long-window weight"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root, run, _champion = synthetic_locked_artifacts(Path(temporary))
                run[key] = value
                (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    verifier._load_locked_meta_candidate(root)
        with tempfile.TemporaryDirectory() as temporary:
            root, run, _champion = synthetic_locked_artifacts(Path(temporary))
            run["fresh_live_cost"]["max_fresh_window_requests_per_probe"] = 3
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cap fresh windows at two"):
                verifier._load_locked_meta_candidate(root)


class HoldoutLockTests(unittest.TestCase):
    def test_lock_requires_exact_hash_ids_order_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            raw = synthetic_holdout_lock_bytes()
            path.write_bytes(raw)
            digest = verifier._sha256_bytes(raw)
            lock = verifier._load_holdout_lock(
                path, HOLDOUT_IDS, expected_sha256=digest
            )
            self.assertEqual(lock.video_ids, HOLDOUT_IDS)
            self.assertEqual(lock.sha256, digest)
            with self.assertRaisesRegex(ValueError, "dataset-source video order"):
                verifier._load_holdout_lock(
                    path, tuple(reversed(HOLDOUT_IDS)), expected_sha256=digest
                )

    def test_tampered_lock_hash_is_rejected_before_json_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            raw = synthetic_holdout_lock_bytes()
            path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "lock hash mismatch"):
                verifier._load_holdout_lock(
                    path,
                    HOLDOUT_IDS,
                    expected_sha256=verifier._sha256_bytes(raw),
                )

    def test_production_lock_hash_is_immutable_in_code(self) -> None:
        self.assertEqual(
            verifier.EXPECTED_HOLDOUT_LOCK_SHA256,
            "d2e0331c0d9c77c4f13c1b1b37083874a3f4bbbae74ec246903effe09ab99bfa",
        )


class GateTests(unittest.TestCase):
    def test_all_three_and_aggregate_score_wrong_gates_pass(self) -> None:
        baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
        control = scores(0.31, 0.31, 0.19, HOLDOUT_IDS)
        candidate = scores(0.32, 0.31, 0.19, HOLDOUT_IDS)
        gates = verifier._holdout_gates(baseline, control, candidate, HOLDOUT_IDS)
        self.assertTrue(gates["holdout_passed"])
        self.assertTrue(gates["vs_baseline"]["all_three_individual_gates_passed"])
        self.assertTrue(gates["vs_run018"]["all_three_individual_gates_passed"])
        self.assertTrue(gates["vs_baseline"]["aggregate"]["passed"])
        self.assertTrue(gates["vs_run018"]["aggregate"]["passed"])

    def test_equality_with_run018_is_allowed_when_baseline_improves(self) -> None:
        baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
        control = scores(0.32, 0.32, 0.18, HOLDOUT_IDS)
        candidate = scores(0.32, 0.32, 0.18, HOLDOUT_IDS)
        gates = verifier._holdout_gates(baseline, control, candidate, HOLDOUT_IDS)
        self.assertTrue(gates["holdout_passed"])
        self.assertEqual(
            gates["vs_run018"]["aggregate"]["score_delta_vs_reference"], 0.0
        )

    def test_one_video_score_or_wrong_regression_fails(self) -> None:
        for key, value, gate in (
            ("strict_browser_live_score", 0.29, "score_gate_passed"),
            ("wrong_live_speech_ratio", 0.21, "wrong_ratio_gate_passed"),
        ):
            with self.subTest(key=key):
                baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
                control = scores(0.31, 0.31, 0.19, HOLDOUT_IDS)
                candidate = scores(0.32, 0.31, 0.19, HOLDOUT_IDS)
                candidate["per_video"][HOLDOUT_IDS[1]][key] = value
                gates = verifier._holdout_gates(
                    baseline, control, candidate, HOLDOUT_IDS
                )
                self.assertFalse(gates["holdout_passed"])
                self.assertFalse(
                    gates["vs_baseline"]["per_video"][HOLDOUT_IDS[1]][gate]
                )

    def test_aggregate_score_and_wrong_gates_are_explicit(self) -> None:
        baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
        control = scores(0.31, 0.31, 0.19, HOLDOUT_IDS)
        candidate = scores(0.29, 0.31, 0.206, HOLDOUT_IDS)
        gates = verifier._holdout_gates(baseline, control, candidate, HOLDOUT_IDS)
        self.assertFalse(gates["vs_baseline"]["aggregate"]["score_gate_passed"])
        self.assertFalse(gates["vs_baseline"]["aggregate"]["wrong_ratio_gate_passed"])
        self.assertFalse(gates["holdout_passed"])

    def test_run018_aggregate_and_per_video_non_regression_gates_are_mandatory(self) -> None:
        baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
        control = scores(0.34, 0.34, 0.16, HOLDOUT_IDS)
        candidate = scores(0.333, 0.333, 0.16, HOLDOUT_IDS)
        gates = verifier._holdout_gates(baseline, control, candidate, HOLDOUT_IDS)
        self.assertTrue(gates["vs_baseline"]["passed"])
        self.assertFalse(gates["vs_run018"]["aggregate"]["score_gate_passed"])
        self.assertFalse(gates["vs_run018"]["all_three_individual_gates_passed"])
        self.assertFalse(gates["holdout_passed"])


class ExecutionAndCommitTests(unittest.TestCase):
    def test_baseline_for_all_videos_precedes_exactly_one_candidate(self) -> None:
        order: list[str] = []
        baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
        control_score = scores(0.31, 0.31, 0.19, HOLDOUT_IDS)
        candidate_score = scores(0.32, 0.31, 0.19, HOLDOUT_IDS)
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_meta_candidate(root)
            with patch.object(
                round2,
                "_prepare_baseline",
                side_effect=lambda _sources: (order.append("baseline_all_three") or {}, baseline),
            ), patch.object(
                round2,
                "_score_candidate",
                side_effect=lambda _prepared, contender: (
                    order.append(contender.name)
                    or (
                        control_score
                        if contender.name == "locked_run018"
                        else candidate_score
                    )
                ),
            ) as score_candidate:
                found_baseline, found_control, found_candidate = verifier._evaluate_holdout(
                    [], locked
                )
        self.assertEqual(
            order,
            ["baseline_all_three", "locked_run018", verifier.EXPECTED_CANDIDATE_NAME],
        )
        self.assertIs(found_baseline, baseline)
        self.assertIs(found_control, control_score)
        self.assertIs(found_candidate, candidate_score)
        self.assertEqual(score_candidate.call_count, 2)
        prepared_objects = [call.args[0] for call in score_candidate.call_args_list]
        self.assertIs(prepared_objects[0], prepared_objects[1])

    def test_artifact_counts_control_separately_from_single_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
            locked = verifier._load_locked_meta_candidate(root)
            baseline = scores(0.30, 0.30, 0.20, HOLDOUT_IDS)
            control = scores(0.31, 0.31, 0.19, HOLDOUT_IDS)
            candidate = scores(0.32, 0.32, 0.18, HOLDOUT_IDS)
            gates = verifier._holdout_gates(
                baseline, control, candidate, HOLDOUT_IDS
            )
            lock = verifier.HoldoutLock(
                Path("/synthetic/lock.json"), HOLDOUT_IDS, [], "c" * 64
            )
            artifact = verifier._holdout_artifact(
                locked, lock, baseline, control, candidate, gates
            )
        self.assertEqual(
            artifact["evaluation_order"],
            [
                "baseline_all_three",
                "locked_run018_control_all_three",
                "single_locked_meta_candidate_all_three",
            ],
        )
        self.assertEqual(artifact["control_evaluation_count"], 1)
        self.assertEqual(artifact["candidate_evaluation_count"], 1)
        self.assertEqual(
            artifact["cached_control_replay_additional_embedding_requests"], 0
        )

    def test_staged_commit_updates_both_files_and_refuses_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, champion = synthetic_locked_artifacts(Path(temporary))
            artifact_path = root / verifier.HOLDOUT_ARTIFACT_NAME
            artifact = {
                "passed": True,
                "holdout_lock": {"sha256": "c" * 64},
            }
            champion_hash = verifier._sha256_file(root / "champion.json")
            updated = verifier._commit_holdout(
                artifact_path,
                root / "champion.json",
                artifact,
                champion,
                expected_champion_sha256=champion_hash,
            )
            self.assertTrue(artifact_path.exists())
            self.assertEqual(updated["status"], verifier.PASSED_CHAMPION_STATUS)
            self.assertTrue(updated["holdout_opened"])
            self.assertTrue(updated["sealed_v4_opened"])
            persisted = json.loads((root / "champion.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["holdout_lock_sha256"], "c" * 64)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                verifier._commit_holdout(
                    artifact_path, root / "champion.json", artifact, champion
                )

    def test_one_shot_rejects_existing_or_staged_result(self) -> None:
        for filename in (verifier.HOLDOUT_ARTIFACT_NAME, verifier.HOLDOUT_ARTIFACT_NAME + ".tmp"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root, _run, _champion = synthetic_locked_artifacts(Path(temporary))
                locked = verifier._load_locked_meta_candidate(root)
                (root / filename).write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "result or staged result"):
                    verifier._assert_one_shot(locked)

    def test_commit_rejects_champion_changed_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _run, champion = synthetic_locked_artifacts(Path(temporary))
            champion_path = root / "champion.json"
            old_hash = verifier._sha256_file(champion_path)
            champion_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                verifier._commit_holdout(
                    root / verifier.HOLDOUT_ARTIFACT_NAME,
                    champion_path,
                    {"passed": False, "holdout_lock": {"sha256": "d" * 64}},
                    champion,
                    expected_champion_sha256=old_hash,
                )


class CostAndCliContractTests(unittest.TestCase):
    def test_no_tuning_arguments_and_exact_live_cost(self) -> None:
        self.assertEqual(verifier.EXPECTED_PROVIDER, meta.EXPECTED_PROVIDER)
        self.assertEqual(verifier.EXPECTED_WINDOWS, (0.8, 2.8))
        self.assertEqual(verifier.EXPECTED_LONG_WEIGHT, 0.25)
        verifier._validate_realtime_cost(meta._fresh_live_cost())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            verifier._parse_args(
                [
                    "--dataset-source",
                    "v4=/c::/i::" + ",".join(HOLDOUT_IDS),
                    "--meta-run-dir",
                    "/run",
                    "--score-tolerance",
                    "1.0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
