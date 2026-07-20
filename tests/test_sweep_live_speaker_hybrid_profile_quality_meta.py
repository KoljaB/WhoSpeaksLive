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
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid_profile_quality as old_profile_quality
import sweep_live_speaker_hybrid_profile_quality_meta as meta
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_hybrid import HybridSpeakerTrackerConfig


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


def score_result(
    global_score: float,
    strict: float,
    wrong: float,
    *,
    strict_overrides: dict[str, float] | None = None,
    wrong_overrides: dict[str, float] | None = None,
) -> dict:
    strict_overrides = strict_overrides or {}
    wrong_overrides = wrong_overrides or {}
    return {
        "aggregate": {"global_score": global_score},
        "per_video": {
            video_id: {
                "strict_browser_live_score": strict_overrides.get(video_id, strict),
                "wrong_live_speech_ratio": wrong_overrides.get(video_id, wrong),
            }
            for video_id in meta.EXPECTED_VIDEO_IDS
        },
    }


def source_args() -> list[str]:
    result: list[str] = []
    for label in meta.EXPECTED_SOURCE_ORDER:
        ids = ",".join(meta.EXPECTED_SOURCE_VIDEO_IDS[label])
        result.extend(["--dataset-source", f"{label}=/cache/{label}::/input/{label}::{ids}"])
    return result


def locked_source(run_dir: Path = Path("/immutable/run018")) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
        family="young_profile_lease",
        champion_path=run_dir / "champion.json",
        run_path=run_dir / "run.json",
        run_dir=run_dir,
        champion_sha256="a" * 64,
        run_sha256="b" * 64,
        provider=meta.EXPECTED_PROVIDER,
        config=run018_config(),
    )


class CandidateContractTests(unittest.TestCase):
    def test_exact_control_and_single_meta_candidate(self) -> None:
        control, candidate = meta._fixed_candidates(run018_config())
        self.assertEqual(control.name, "locked_run018")
        self.assertEqual(control.candidate_id, round2.LOCKED_RUN018_CANDIDATE_ID)
        self.assertEqual(candidate.name, "profile_quality_meta_a005_s035")
        self.assertEqual(candidate.family, "profile_quality_meta_lease_a005_s035_v1")
        self.assertEqual(meta.EVALUATION_ORDER, (
            "baseline", "locked_run018", "profile_quality_meta_a005_s035"
        ))
        self.assertNotEqual(candidate.candidate_id, control.candidate_id)

    def test_candidate_is_exact_a005_s035_contract_on_run018(self) -> None:
        control, candidate = meta._fixed_candidates(run018_config(
            profile_quality_fast_lease_min_sentence_count=99,
            profile_quality_fast_lease_min_speech_seconds=99.0,
            profile_quality_fast_lease_min_similarity=0.99,
            profile_quality_fast_lease_min_margin=0.99,
            profile_quality_meta_fresh_min_age_seconds=0.7,
            profile_quality_meta_switch_min_short_margin=0.9,
        ))
        config = candidate.config
        self.assertFalse(config.enable_short_scale_fast_lease)
        self.assertFalse(config.enable_profile_quality_short_scale_fast_lease)
        self.assertTrue(config.enable_profile_quality_meta_lease)
        self.assertEqual(config.profile_quality_fast_lease_min_sentence_count, 2)
        self.assertEqual(config.profile_quality_fast_lease_min_speech_seconds, 3.1)
        self.assertEqual(config.profile_quality_fast_lease_min_similarity, 0.18)
        self.assertEqual(config.profile_quality_fast_lease_min_margin, 0.06)
        self.assertEqual(config.profile_quality_meta_fresh_min_age_seconds, 0.05)
        self.assertEqual(config.profile_quality_meta_fresh_max_age_seconds, 0.8)
        self.assertEqual(config.profile_quality_meta_fresh_min_speech_seconds, 3.8)
        self.assertEqual(config.profile_quality_meta_fresh_min_short_margin, 0.30)
        self.assertEqual(config.profile_quality_meta_fresh_min_long_margin, 0.70)
        self.assertEqual(config.profile_quality_meta_independent_max_profile_count, 8)
        self.assertEqual(config.profile_quality_meta_switch_min_short_margin, 0.35)
        for field in (
            "young_min_similarity",
            "young_min_margin",
            "young_required_consecutive_probes",
            "young_independent_scale_count",
            "young_fast_independent_scale_count",
        ):
            self.assertEqual(getattr(config, field), getattr(control.config, field))

    def test_run018_with_any_experimental_output_path_is_rejected(self) -> None:
        for override, message in (
            ({"enable_short_scale_fast_lease": True}, "unrestricted"),
            ({"enable_profile_quality_short_scale_fast_lease": True}, "old profile-quality"),
            ({"enable_profile_quality_meta_lease": True}, "already enables"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                meta._fixed_candidates(run018_config(**override))

    def test_candidate_id_is_deterministic_and_existing_ids_stay_stable(self) -> None:
        config = run018_config()
        control, candidate = meta._fixed_candidates(config)
        self.assertEqual(meta._candidate_id(candidate.config), candidate.candidate_id)
        self.assertEqual(control.candidate_id, (
            "0f58c8894d2c4f0190e30ec072fdad29de485151a2a76803a2abeec96a0e845f"
        ))
        self.assertEqual(
            round2._locked_candidates(config)[1].candidate_id,
            "c05934b79b3d0dcdac019fae0df6434837b2e0b0147f9cd134ddcda618776c29",
        )
        old_candidate = old_profile_quality._fixed_candidates(config)[1]
        self.assertEqual(
            old_profile_quality._candidate_id(old_candidate.config),
            "bf86349a69b359ad864c1eccadfa1bdabcf469958b12b7bb6cb40b49dde85f1e",
        )


class DatasetSealTests(unittest.TestCase):
    def test_exact_v1_v2_v3_mapping_is_canonicalized(self) -> None:
        raw = []
        for label in reversed(meta.EXPECTED_SOURCE_ORDER):
            ids = ",".join(meta.EXPECTED_SOURCE_VIDEO_IDS[label])
            raw.append(f"{label}=/c/{label}::/i/{label}::{ids}")
        sources = meta._validate_sources(raw)
        self.assertEqual(tuple(source.label for source in sources), meta.EXPECTED_SOURCE_ORDER)
        self.assertEqual(
            tuple(video_id for source in sources for video_id in source.video_ids),
            meta.EXPECTED_VIDEO_IDS,
        )

    def test_v4_is_rejected_before_path_resolution_or_run018_load(self) -> None:
        forbidden = next(iter(meta.FORBIDDEN_V4_IDS))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            round2, "_load_run018"
        ) as load, mock.patch.object(Path, "resolve") as resolve:
            with self.assertRaisesRegex(ValueError, "sealed v4.*before path resolution"):
                meta.main([
                    "--dataset-source", f"v1=/must/not/open::/must/not/open::{forbidden}",
                    "--locked-run-dir", str(Path(directory) / "locked"),
                    "--run-dir", str(Path(directory) / "output"),
                ])
        load.assert_not_called()
        resolve.assert_not_called()

    def test_missing_source_or_changed_mapping_is_rejected(self) -> None:
        ids = ",".join(meta.EXPECTED_SOURCE_VIDEO_IDS["v1"])
        with self.assertRaisesRegex(ValueError, "exactly v1, v2, and v3"):
            meta._validate_sources([f"v1=/c::/i::{ids}"])
        changed = list(meta.EXPECTED_SOURCE_VIDEO_IDS["v3"])
        changed[-1] = "not_opened"
        raw = [
            f"v1=/c1::/i1::{','.join(meta.EXPECTED_SOURCE_VIDEO_IDS['v1'])}",
            f"v2=/c2::/i2::{','.join(meta.EXPECTED_SOURCE_VIDEO_IDS['v2'])}",
            f"v3=/c3::/i3::{','.join(changed)}",
        ]
        with self.assertRaisesRegex(ValueError, "fixed opened IDs"):
            meta._validate_sources(raw)

    def test_no_tuning_arguments_exist(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            meta._parse_args([
                *source_args(),
                "--locked-run-dir", "/locked",
                "--run-dir", "/output",
                "--meta-alpha", "0.1",
            ])


class FrozenMetadataTests(unittest.TestCase):
    def test_source_and_candidate_locks_freeze_hashes_mapping_and_config(self) -> None:
        sources = meta._validate_sources([
            f"{label}=/c/{label}::/i/{label}::{','.join(meta.EXPECTED_SOURCE_VIDEO_IDS[label])}"
            for label in meta.EXPECTED_SOURCE_ORDER
        ])
        source_lock = meta._dataset_source_lock(sources)
        locked = locked_source()
        candidates = meta._fixed_candidates(locked.config)
        candidate_lock = meta._candidate_lock(locked, candidates, source_lock)
        self.assertEqual(len(source_lock["video_ids"]), 12)
        self.assertEqual(len(source_lock["lock_sha256"]), 64)
        self.assertEqual(candidate_lock["source_champion_sha256"], "a" * 64)
        self.assertEqual(candidate_lock["source_run_sha256"], "b" * 64)
        self.assertEqual(candidate_lock["source_candidate_id"], round2.LOCKED_RUN018_CANDIDATE_ID)
        self.assertEqual(candidate_lock["dataset_source_lock_sha256"], source_lock["lock_sha256"])
        self.assertEqual(candidate_lock["candidate_count"], 2)
        self.assertEqual(candidate_lock["search_space"], "none")
        self.assertEqual(candidate_lock["provider"], meta.EXPECTED_PROVIDER)
        self.assertEqual(candidate_lock["windows_seconds"], [0.8, 2.8])
        self.assertEqual(candidate_lock["long_weight"], 0.25)
        self.assertEqual(candidate_lock["fixed_meta_contract"]["policy_id"], meta.META_POLICY_ID)
        self.assertEqual(len(candidate_lock["lock_sha256"]), 64)
        self.assertEqual(
            candidate_lock["source_config_sha256"],
            meta._sha256_value(candidate_lock["source_config"]),
        )

    def test_run_metadata_freezes_real_time_cost_and_seal(self) -> None:
        sources = meta._validate_sources([
            f"{label}=/c/{label}::/i/{label}::{','.join(meta.EXPECTED_SOURCE_VIDEO_IDS[label])}"
            for label in meta.EXPECTED_SOURCE_ORDER
        ])
        source_lock = meta._dataset_source_lock(sources)
        locked = locked_source()
        candidate_lock = meta._candidate_lock(
            locked, meta._fixed_candidates(locked.config), source_lock
        )
        run = meta._run_metadata(sources, source_lock, candidate_lock)
        self.assertEqual(run["video_count"], 12)
        self.assertEqual(run["video_ids"], list(meta.EXPECTED_VIDEO_IDS))
        self.assertTrue(run["baseline_must_complete_before_candidates"])
        self.assertFalse(run["baseline_completed_before_candidate_evaluation"])
        self.assertFalse(run["parameter_search_performed"])
        self.assertFalse(run["sealed_v4_opened"])
        self.assertEqual(run["provider"], meta.EXPECTED_PROVIDER)
        self.assertEqual(run["windows_seconds"], [0.8, 2.8])
        self.assertEqual(run["long_weight"], 0.25)
        cost = run["fresh_live_cost"]
        self.assertEqual(cost["fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["max_fresh_window_requests_per_probe"], 2)


class PromotionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = score_result(0.30, 0.30, 0.20)
        self.control = score_result(0.32, 0.32, 0.18)

    def test_candidate_must_pass_every_gate_against_both_references(self) -> None:
        candidate = score_result(0.34, 0.33, 0.17)
        decision = meta._promotion_decision(self.baseline, self.control, candidate)
        self.assertTrue(decision["baseline_gates_passed"])
        self.assertTrue(decision["run018_gates_passed"])
        self.assertTrue(decision["candidate_eligible"])
        self.assertEqual(decision["winner"], meta.META_NAME)
        self.assertEqual(
            set(decision["candidate_vs_locked"]["per_video_score_delta"]),
            set(meta.EXPECTED_VIDEO_IDS),
        )

    def test_aggregate_must_improve_run018(self) -> None:
        candidate = score_result(0.31, 0.33, 0.17)
        decision = meta._promotion_decision(self.baseline, self.control, candidate)
        self.assertTrue(decision["baseline_gates_passed"])
        self.assertFalse(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])

    def test_one_video_score_regression_against_either_reference_blocks(self) -> None:
        video_id = meta.EXPECTED_VIDEO_IDS[4]
        candidate = score_result(
            0.34, 0.33, 0.17, strict_overrides={video_id: 0.3139}
        )
        decision = meta._promotion_decision(self.baseline, self.control, candidate)
        self.assertFalse(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])

    def test_one_video_score_regression_against_baseline_alone_blocks(self) -> None:
        video_id = meta.EXPECTED_VIDEO_IDS[3]
        control = score_result(
            0.32, 0.32, 0.18, strict_overrides={video_id: 0.28}
        )
        candidate = score_result(
            0.34, 0.33, 0.17, strict_overrides={video_id: 0.2949}
        )
        decision = meta._promotion_decision(self.baseline, control, candidate)
        self.assertFalse(decision["baseline_gates_passed"])
        self.assertTrue(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])

    def test_one_video_wrong_ratio_regression_against_either_reference_blocks(self) -> None:
        video_id = meta.EXPECTED_VIDEO_IDS[8]
        candidate = score_result(
            0.34, 0.33, 0.17, wrong_overrides={video_id: 0.1851}
        )
        decision = meta._promotion_decision(self.baseline, self.control, candidate)
        self.assertFalse(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])

    def test_missing_video_cannot_evade_a_gate(self) -> None:
        candidate = score_result(0.34, 0.33, 0.17)
        candidate["per_video"].pop(meta.EXPECTED_VIDEO_IDS[-1])
        with self.assertRaisesRegex(ValueError, "exact twelve-video cohort"):
            meta._promotion_decision(self.baseline, self.control, candidate)


class SyntheticExecutionTests(unittest.TestCase):
    def test_baseline_then_control_then_only_candidate_and_atomic_artifacts(self) -> None:
        baseline = score_result(0.30, 0.30, 0.20)
        order: list[str] = []

        def prepare(_sources):
            order.append("baseline")
            return {"synthetic": True}, baseline

        def score_candidate(_prepared, candidate):
            order.append(candidate.name)
            result = (
                score_result(0.32, 0.32, 0.18)
                if candidate.name == "locked_run018"
                else score_result(0.34, 0.33, 0.17)
            )
            return {
                **result,
                "name": candidate.name,
                "family": candidate.family,
                "candidate_id": candidate.candidate_id,
                "config": asdict(candidate.config),
            }

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            locked = locked_source(temporary / "locked")
            output = temporary / "output"
            with mock.patch.object(
                round2, "_load_run018", return_value=locked
            ), mock.patch.object(
                round2, "_prepare_baseline", side_effect=prepare
            ), mock.patch.object(
                round2, "_score_candidate", side_effect=score_candidate
            ):
                exit_code = meta.main([
                    *source_args(),
                    "--locked-run-dir", str(temporary / "locked"),
                    "--run-dir", str(output),
                ])

            self.assertEqual(exit_code, 0)
            self.assertEqual(order, list(meta.EVALUATION_ORDER))
            expected_files = {
                "run.json", "baseline.json", "progress.json", "trials.jsonl",
                "report.json", "champion.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            self.assertFalse(any(path.suffix == ".tmp" for path in output.iterdir()))
            trials = [
                json.loads(line)
                for line in (output / "trials.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["name"] for row in trials], ["locked_run018", meta.META_NAME])
            run = json.loads((output / "run.json").read_text())
            champion = json.loads((output / "champion.json").read_text())
            progress = json.loads((output / "progress.json").read_text())
            self.assertTrue(run["baseline_completed_before_candidate_evaluation"])
            self.assertEqual(run["winner"], meta.META_NAME)
            self.assertEqual(champion["winner"]["name"], meta.META_NAME)
            self.assertTrue(champion["promotion"]["candidate_eligible"])
            self.assertFalse(champion["production_defaults_changed"])
            self.assertFalse(champion["parameter_search_performed"])
            self.assertFalse(champion["sealed_v4_opened"])
            self.assertEqual(progress["phase"], "COMPLETE")
            self.assertEqual(progress["progress_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
