import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sweep_live_speaker_hybrid as sweep
from window.live_speaker_hybrid import HybridSpeakerTrackerConfig


class HybridSweepTests(unittest.TestCase):
    def test_legacy_locked_candidate_id_survives_optional_config_extension(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_trusted_min_sentence_count=4,
            young_trusted_min_speech_seconds=8.0,
            young_min_similarity=0.35,
            young_min_margin=0.08,
            young_required_consecutive_probes=1,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            self_echo_guard_seconds=0.0,
            enable_boundary_abstention=False,
            history_max_gap_seconds=1.5,
        )
        self.assertEqual(
            sweep._candidate_id("young_profile_lease", config),
            "0f58c8894d2c4f0190e30ec072fdad29de485151a2a76803a2abeec96a0e845f",
        )
        short_scale = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "enable_short_scale_fast_lease": True,
        })
        self.assertNotEqual(
            sweep._candidate_id("young_profile_lease", short_scale),
            sweep._candidate_id("young_profile_lease", config),
        )
        profile_quality_disabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "profile_quality_fast_lease_min_similarity": 0.99,
        })
        self.assertEqual(
            sweep._candidate_id("young_profile_lease", profile_quality_disabled),
            sweep._candidate_id("young_profile_lease", config),
        )
        profile_quality_enabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "enable_profile_quality_short_scale_fast_lease": True,
        })
        self.assertNotEqual(
            sweep._candidate_id("young_profile_lease", profile_quality_enabled),
            sweep._candidate_id("young_profile_lease", config),
        )
        meta_threshold_changed_while_disabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "profile_quality_meta_fresh_min_age_seconds": 0.33,
            "profile_quality_meta_switch_min_short_margin": 0.91,
        })
        self.assertEqual(
            sweep._candidate_id("young_profile_lease", meta_threshold_changed_while_disabled),
            sweep._candidate_id("young_profile_lease", config),
        )
        meta_enabled = HybridSpeakerTrackerConfig(**{
            **config.__dict__,
            "enable_profile_quality_meta_lease": True,
        })
        self.assertNotEqual(
            sweep._candidate_id("young_profile_lease", meta_enabled),
            sweep._candidate_id("young_profile_lease", config),
        )
        meta_qvote_threshold_changed = HybridSpeakerTrackerConfig(**{
            **meta_enabled.__dict__,
            "profile_quality_fast_lease_min_similarity": 0.99,
        })
        self.assertNotEqual(
            sweep._candidate_id("young_profile_lease", meta_qvote_threshold_changed),
            sweep._candidate_id("young_profile_lease", meta_enabled),
        )

    def test_split_keeps_jws_sealed(self):
        self.assertNotIn(sweep.SEALED_HOLDOUT, sweep.DEVELOPMENT_VIDEOS)
        self.assertEqual(
            sweep.DEVELOPMENT_VIDEOS,
            ("Dd7FixvoKBw", "DsyfYJ5Ou3g", "20v1OxUXcQY"),
        )

    def test_preregistered_grid_sizes_are_bounded(self):
        self.assertEqual(len(sweep._young_grid(lease=True)), 24)
        self.assertEqual(len(sweep._young_grid(lease=False)), 24)
        self.assertEqual(len(sweep._boundary_grid(HybridSpeakerTrackerConfig())), 48)
        self.assertEqual(24 + 24 + 48 + 48, 144)

    def test_fast_champion_requires_one_probe_lease(self):
        fast = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_fast_independent_scale_count=1,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
        )
        self.assertTrue(sweep._fast_latency_eligible(fast))
        self.assertFalse(sweep._fast_latency_eligible(
            HybridSpeakerTrackerConfig(**{
                **fast.__dict__,
                "young_required_consecutive_probes": 2,
            })
        ))
        self.assertFalse(sweep._fast_latency_eligible(
            HybridSpeakerTrackerConfig(**{
                **fast.__dict__,
                "enable_young_profile_lease": False,
            })
        ))
        self.assertFalse(sweep._fast_latency_eligible(
            HybridSpeakerTrackerConfig(**{
                **fast.__dict__,
                "young_independent_scale_count": 1,
            })
        ))

    def test_cost_contract_distinguishes_windows_and_provider_forwards(self):
        cost = sweep._fresh_live_cost()
        self.assertEqual(cost["fresh_window_requests_per_probe"], 2)
        self.assertEqual(cost["provider_component_forwards_per_probe"], 4)
        self.assertFalse(cost["cache_grid_is_live_probe_cadence"])

    def test_boundary_must_clear_occam_margin(self):
        def row(global_score, score_delta=0.0, wrong_delta=0.0):
            return {
                "aggregate": {"global_score": global_score},
                "per_video": {
                    video: {
                        "strict_browser_live_score": 0.4 + score_delta,
                        "wrong_live_speech_ratio": 0.1 + wrong_delta,
                    }
                    for video in sweep.DEVELOPMENT_VIDEOS
                },
            }

        simple = row(0.34)
        self.assertFalse(sweep._boundary_materially_better(row(0.3449), simple, 0.005))
        self.assertTrue(sweep._boundary_materially_better(row(0.345), simple, 0.005))
        self.assertFalse(sweep._boundary_materially_better(
            row(0.35, score_delta=-0.0021), simple, 0.005
        ))


if __name__ == "__main__":
    unittest.main()
