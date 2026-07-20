from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import sweep_live_speaker_hybrid_round2 as round2
from window.live_speaker_hybrid import HybridSpeakerTrackerConfig


def score(global_score: float, videos: dict[str, tuple[float, float]]) -> dict:
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


class DatasetSourceTests(unittest.TestCase):
    def test_source_parser_supports_windows_roots_and_multiple_videos(self) -> None:
        source = round2._parse_dataset_source(
            r"v1=D:\cache::D:\inputs::Dd7FixvoKBw,DsyfYJ5Ou3g"
        )
        self.assertEqual(source.label, "v1")
        self.assertEqual(source.corpus_root, Path(r"D:\cache"))
        self.assertEqual(source.video_ids, ("Dd7FixvoKBw", "DsyfYJ5Ou3g"))

    def test_duplicate_video_across_sources_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one dataset source"):
            round2._validate_sources(
                ["v1=/c1::/i1::same", "v2=/c2::/i2::same"]
            )

    def test_duplicate_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            round2._validate_sources(
                ["v1=/c1::/i1::one", "v1=/c2::/i2::two"]
            )

    def test_each_sealed_v3_video_is_rejected_during_parsing(self) -> None:
        for video_id in sorted(round2.FORBIDDEN_V3_IDS):
            with self.subTest(video_id=video_id), self.assertRaisesRegex(
                ValueError, "sealed v3"
            ):
                round2._parse_dataset_source(f"v3=/never/open::/never/open::{video_id}")


class CandidateLockTests(unittest.TestCase):
    def test_candidate_set_contains_exactly_control_and_one_field_variant(self) -> None:
        control = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_short_scale_fast_lease=False,
            young_min_similarity=0.35,
            young_min_margin=0.08,
            young_required_consecutive_probes=1,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
        )
        candidates = round2._locked_candidates(control)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([value.name for value in candidates], [
            "locked_run018", "short_scale_fast_lease"
        ])
        self.assertFalse(candidates[0].config.enable_short_scale_fast_lease)
        self.assertTrue(candidates[1].config.enable_short_scale_fast_lease)
        left = round2.asdict(candidates[0].config)
        right = round2.asdict(candidates[1].config)
        self.assertEqual(
            {key for key in left if left[key] != right[key]},
            {"enable_short_scale_fast_lease"},
        )

    def test_disabled_profile_quality_fields_do_not_rekey_old_candidate(self) -> None:
        config = HybridSpeakerTrackerConfig(enable_short_scale_fast_lease=True)
        thresholds_changed_while_disabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "profile_quality_fast_lease_min_sentence_count": 99,
            "profile_quality_fast_lease_min_similarity": 0.99,
        })
        self.assertEqual(
            round2._candidate_id("short_scale_fast_lease", config),
            round2._candidate_id(
                "short_scale_fast_lease", thresholds_changed_while_disabled
            ),
        )
        enabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "enable_profile_quality_short_scale_fast_lease": True,
        })
        self.assertNotEqual(
            round2._candidate_id("short_scale_fast_lease", config),
            round2._candidate_id("short_scale_fast_lease", enabled),
        )

    def test_disabled_meta_fields_do_not_rekey_round2_candidate(self) -> None:
        config = HybridSpeakerTrackerConfig(enable_short_scale_fast_lease=True)
        thresholds_changed = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "profile_quality_meta_fresh_min_age_seconds": 0.42,
            "profile_quality_meta_switch_min_short_margin": 0.87,
        })
        self.assertEqual(
            round2._candidate_id("short_scale_fast_lease", config),
            round2._candidate_id("short_scale_fast_lease", thresholds_changed),
        )

    def test_run_metadata_preregisters_baseline_first(self) -> None:
        source = round2.DatasetSource("v1", Path("/c"), Path("/i"), ("one",))
        metadata = round2._run_metadata([source], {"lock_sha256": "abc"})
        self.assertEqual(metadata["evaluation_order"][0], "baseline")
        self.assertTrue(metadata["baseline_must_complete_before_candidates"])
        self.assertFalse(metadata["baseline_completed_before_candidate_evaluation"])
        self.assertEqual(metadata["selection_policy"], "fixed_two_candidate_ab_no_search")

    def test_cost_contract_is_exactly_two_windows(self) -> None:
        cost = round2._fresh_live_cost()
        self.assertEqual(cost["fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["max_fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["windows_seconds"], [0.8, 2.8])
        self.assertEqual(cost["cache_hop_seconds"], 0.2)
        self.assertEqual(cost["production_probe_interval_seconds"], 0.75)
        self.assertFalse(cost["cache_grid_is_live_probe_cadence"])


class PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = score(0.30, {"a": (0.30, 0.20), "b": (0.40, 0.10)})
        self.locked = score(0.32, {"a": (0.32, 0.18), "b": (0.41, 0.09)})

    def test_partial_variant_wins_only_when_it_improves_locked_and_passes_gates(self) -> None:
        partial = score(0.33, {"a": (0.33, 0.17), "b": (0.42, 0.08)})
        decision = round2._promotion_decision(self.baseline, self.locked, partial)
        self.assertTrue(decision["partial_eligible"])
        self.assertEqual(decision["winner"], "short_scale_fast_lease")

    def test_locked_is_retained_when_partial_regresses_one_video(self) -> None:
        partial = score(0.33, {"a": (0.31, 0.18), "b": (0.45, 0.08)})
        decision = round2._promotion_decision(self.baseline, self.locked, partial)
        self.assertFalse(decision["partial_eligible"])
        self.assertTrue(decision["locked_eligible"])
        self.assertEqual(decision["winner"], "locked_run018")


if __name__ == "__main__":
    unittest.main()
