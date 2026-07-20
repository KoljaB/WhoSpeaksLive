from __future__ import annotations

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

import sweep_live_speaker_hybrid_profile_quality as profile_quality
import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_hybrid import HybridSpeakerTrackerConfig


def result(global_score: float, videos: dict[str, tuple[float, float]]) -> dict:
    return {
        "aggregate": {"global_score": global_score},
        "per_video": {
            video_id: {
                "strict_browser_live_score": strict,
                "wrong_live_speech_ratio": wrong,
            }
            for video_id, (strict, wrong) in videos.items()
        },
    }


def run018_config(**overrides) -> HybridSpeakerTrackerConfig:
    values = {
        "enable_young_profile_confirmation": True,
        "enable_young_profile_lease": True,
        "enable_short_scale_fast_lease": False,
        "enable_profile_quality_short_scale_fast_lease": False,
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


class CandidateContractTests(unittest.TestCase):
    def test_exactly_two_candidates_and_old_control_id_is_preserved(self) -> None:
        candidates = profile_quality._fixed_candidates(run018_config())
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].candidate_id, round2.LOCKED_RUN018_CANDIDATE_ID)
        self.assertEqual(candidates[0].name, "locked_run018")
        self.assertEqual(candidates[1].name, profile_quality.PROFILE_QUALITY_NAME)
        self.assertNotEqual(candidates[1].candidate_id, candidates[0].candidate_id)

    def test_candidate_is_exact_locked_profile_quality_contract(self) -> None:
        control, candidate = profile_quality._fixed_candidates(
            run018_config(
                profile_quality_fast_lease_min_sentence_count=9,
                profile_quality_fast_lease_min_speech_seconds=9.0,
                profile_quality_fast_lease_min_similarity=0.9,
                profile_quality_fast_lease_min_margin=0.9,
            )
        )
        config = candidate.config
        self.assertFalse(config.enable_short_scale_fast_lease)
        self.assertTrue(config.enable_profile_quality_short_scale_fast_lease)
        self.assertEqual(config.profile_quality_fast_lease_min_sentence_count, 2)
        self.assertEqual(config.profile_quality_fast_lease_min_speech_seconds, 3.1)
        self.assertEqual(config.profile_quality_fast_lease_min_similarity, 0.18)
        self.assertEqual(config.profile_quality_fast_lease_min_margin, 0.06)
        unchanged = {
            "young_min_similarity",
            "young_min_margin",
            "young_required_consecutive_probes",
            "young_independent_scale_count",
            "young_fast_independent_scale_count",
        }
        for field in unchanged:
            self.assertEqual(getattr(config, field), getattr(control.config, field))

    def test_unrestricted_short_lease_in_locked_control_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unrestricted"):
            profile_quality._fixed_candidates(
                run018_config(enable_short_scale_fast_lease=True)
            )

    def test_candidate_id_is_deterministic(self) -> None:
        config = profile_quality._fixed_candidates(run018_config())[1].config
        self.assertEqual(
            profile_quality._candidate_id(config),
            profile_quality._candidate_id(config),
        )
        changed_disabled_meta_threshold = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "profile_quality_meta_fresh_min_age_seconds": 0.44,
        })
        self.assertEqual(
            profile_quality._candidate_id(config),
            profile_quality._candidate_id(changed_disabled_meta_threshold),
        )


class SealAndMetadataTests(unittest.TestCase):
    def test_v3_is_rejected_before_locked_run_or_paths_are_opened(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            round2, "_load_run018"
        ) as loader:
            with self.assertRaisesRegex(ValueError, "sealed v3"):
                profile_quality.main([
                    "--dataset-source",
                    f"v3=/must/not/open::/must/not/open::{next(iter(round2.FORBIDDEN_V3_IDS))}",
                    "--locked-run-dir",
                    str(Path(directory) / "locked"),
                    "--run-dir",
                    str(Path(directory) / "output"),
                ])
            loader.assert_not_called()

    def test_metadata_locks_baseline_first_and_two_window_cost(self) -> None:
        source = round2.DatasetSource("v1", Path("/cache"), Path("/input"), ("one",))
        metadata = profile_quality._run_metadata([source], {"lock_sha256": "x"})
        self.assertEqual(metadata["evaluation_order"], [
            "baseline", "locked_run018", profile_quality.PROFILE_QUALITY_NAME
        ])
        self.assertTrue(metadata["baseline_must_complete_before_candidates"])
        self.assertFalse(metadata["baseline_completed_before_candidate_evaluation"])
        cost = metadata["fresh_live_cost"]
        self.assertEqual(cost["fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["max_fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["windows_seconds"], [0.8, 2.8])
        self.assertEqual(cost["long_weight"], 0.25)
        self.assertFalse(cost["cache_grid_is_live_probe_cadence"])

    def test_candidate_lock_freezes_hashes_provider_and_config(self) -> None:
        candidates = profile_quality._fixed_candidates(run018_config())
        locked = SimpleNamespace(
            candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
            family="young_profile_lease",
            champion_path=Path("/old/champion.json"),
            run_path=Path("/old/run.json"),
            champion_sha256="a" * 64,
            run_sha256="b" * 64,
            provider="pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5",
        )
        lock = profile_quality._candidate_lock(locked, candidates)
        self.assertEqual(lock["candidate_count"], 2)
        self.assertEqual(lock["windows_seconds"], [0.8, 2.8])
        self.assertEqual(lock["long_weight"], 0.25)
        self.assertEqual(lock["source_champion_sha256"], "a" * 64)
        self.assertEqual(len(lock["lock_sha256"]), 64)
        self.assertEqual(lock["fixed_profile_quality_contract"]["eligibility"],
                         "sentence_count>=2 OR speech_seconds>=3.1")

    def test_main_evaluates_baseline_then_exactly_two_candidates_and_writes_contract(self) -> None:
        locked = SimpleNamespace(
            candidate_id=round2.LOCKED_RUN018_CANDIDATE_ID,
            family="young_profile_lease",
            champion_path=Path("/old/champion.json"),
            run_path=Path("/old/run.json"),
            run_dir=Path("/old"),
            champion_sha256="a" * 64,
            run_sha256="b" * 64,
            provider="pyannote_wespeaker_resnet34_lm=1+wespeaker_resnet34_lm_onnx=0.5",
            config=run018_config(),
        )
        baseline = result(0.30, {"a": (0.30, 0.20)})
        order: list[str] = []

        def prepare(_sources):
            order.append("baseline")
            return {}, baseline

        def score_candidate(_prepared, candidate):
            order.append(candidate.name)
            value = result(
                0.31 if candidate.name == "locked_run018" else 0.32,
                {"a": (0.31, 0.18) if candidate.name == "locked_run018" else (0.32, 0.17)},
            )
            return {
                **value,
                "name": candidate.name,
                "family": candidate.family,
                "candidate_id": candidate.candidate_id,
                "config": profile_quality.asdict(candidate.config),
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            round2, "_load_run018", return_value=locked
        ), mock.patch.object(
            round2, "_prepare_baseline", side_effect=prepare
        ), mock.patch.object(
            round2, "_score_candidate", side_effect=score_candidate
        ):
            output = Path(directory) / "output"
            exit_code = profile_quality.main([
                "--dataset-source", "v1=/cache::/input::a",
                "--locked-run-dir", str(Path(directory) / "locked"),
                "--run-dir", str(output),
            ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(order, list(profile_quality.EVALUATION_ORDER))
            self.assertEqual(len((output / "trials.jsonl").read_text().splitlines()), 2)
            run = profile_quality.json.loads((output / "run.json").read_text())
            champion = profile_quality.json.loads((output / "champion.json").read_text())
            self.assertTrue(run["baseline_completed_before_candidate_evaluation"])
            self.assertEqual(run["winner"], profile_quality.PROFILE_QUALITY_NAME)
            self.assertEqual(champion["winner"]["name"], profile_quality.PROFILE_QUALITY_NAME)


class PromotionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = result(0.30, {"a": (0.30, 0.20), "b": (0.40, 0.10)})
        self.locked = result(0.31, {"a": (0.31, 0.18), "b": (0.41, 0.09)})

    def test_candidate_wins_only_when_all_baseline_and_run018_gates_pass(self) -> None:
        candidate = result(0.32, {"a": (0.32, 0.17), "b": (0.42, 0.08)})
        decision = profile_quality._promotion_decision(
            self.baseline, self.locked, candidate
        )
        self.assertTrue(decision["baseline_gates_passed"])
        self.assertTrue(decision["run018_gates_passed"])
        self.assertTrue(decision["candidate_eligible"])
        self.assertEqual(decision["winner"], profile_quality.PROFILE_QUALITY_NAME)

    def test_one_baseline_video_score_regression_blocks_candidate(self) -> None:
        candidate = result(0.32, {"a": (0.294, 0.17), "b": (0.50, 0.08)})
        decision = profile_quality._promotion_decision(
            self.baseline, self.locked, candidate
        )
        self.assertFalse(decision["baseline_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])
        self.assertIsNone(decision["winner"])

    def test_one_run018_wrong_ratio_regression_blocks_candidate(self) -> None:
        # Still better than baseline, but 0.006 worse than run018 on video a.
        candidate = result(0.32, {"a": (0.32, 0.186), "b": (0.42, 0.08)})
        decision = profile_quality._promotion_decision(
            self.baseline, self.locked, candidate
        )
        self.assertTrue(decision["baseline_gates_passed"])
        self.assertFalse(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])

    def test_aggregate_must_improve_both_references(self) -> None:
        candidate = result(0.305, {"a": (0.32, 0.17), "b": (0.42, 0.08)})
        decision = profile_quality._promotion_decision(
            self.baseline, self.locked, candidate
        )
        self.assertTrue(decision["baseline_gates_passed"])
        self.assertFalse(decision["run018_gates_passed"])
        self.assertFalse(decision["candidate_eligible"])


if __name__ == "__main__":
    unittest.main()
