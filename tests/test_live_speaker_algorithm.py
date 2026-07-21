from __future__ import annotations

import unittest

import numpy as np

from window.live_speaker_algorithm import (
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerStep,
    SpeakerProfileEvent,
    compare_decision_traces,
)
from window.live_speaker_benchmark import (
    PRIMARY_SCORER_V2_ID,
    aggregate_video_scores,
    aggregate_video_scores_primary_v2,
    score_live_speaker_decisions,
)
from window.live_speaker_bayes import (
    BayesSpeakerTrackerConfig,
    CausalBayesSpeakerTracker,
    replay_cached_bayes_windows,
)
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_replay import CachedLiveWindowBlock, replay_cached_live_windows, run_live_embedding_steps


class CausalLiveSpeakerAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(1.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        self.config = LiveSpeakerAlgorithmConfig(
            ema_count=1, min_known_probability=0.2, unknown_release_count=2
        )

    def test_cached_and_fresh_embeddings_have_exact_decision_parity(self) -> None:
        times = np.asarray([0.8, 1.0, 1.2, 1.4, 1.6], dtype=np.float64)
        embeddings = np.asarray([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        ], dtype=np.float32)
        speech = np.ones(times.shape[0], dtype=bool)
        block = CachedLiveWindowBlock(
            provider="fake", video_id="video", window_seconds=1.0,
            media_times=times, embeddings=embeddings, valid=np.ones(5, dtype=bool),
            raw_rms=np.ones(5, dtype=np.float32), sample_rate=16_000,
        )
        cached = replay_cached_live_windows(
            block,
            self.events,
            speech,
            np.ones(5, dtype=bool),
            np.zeros(5, dtype=bool),
            config=self.config,
        )
        fresh = run_live_embedding_steps(
            [LiveSpeakerStep(float(t), True, vector, 1.0) for t, vector in zip(times, embeddings)],
            self.events,
            config=self.config,
        )
        report = compare_decision_traces(fresh, cached)
        self.assertTrue(report["exact_match"])
        self.assertEqual([item.visible_speaker for item in cached], ["S1", "S1", "S2", "S2", "S2"])

    def test_profile_is_not_visible_before_its_causal_event(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config, self.events)
        before = algorithm.step(LiveSpeakerStep(0.8, True, np.asarray([0.0, 1.0]), 1.0))
        after = algorithm.step(LiveSpeakerStep(1.0, True, np.asarray([0.0, 1.0]), 1.0))
        self.assertNotEqual(before.visible_speaker, "S2")
        self.assertEqual(after.visible_speaker, "S2")

    def test_production_profile_sync_preserves_temporal_state(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config)
        changed = algorithm.sync_profiles([
            {"label": "S1", "centroid": [1.0, 0.0], "sentence_count": 1, "speech_seconds": 1.0}
        ])
        acquired = algorithm.step(LiveSpeakerStep(0.2, True, np.asarray([1.0, 0.0]), 1.0))
        unchanged = algorithm.sync_profiles([
            {"label": "S1", "centroid": [1.0, 0.0], "sentence_count": 1, "speech_seconds": 1.0}
        ])
        held = algorithm.step(LiveSpeakerStep(0.4, False, None, 1.0, probe_scheduled=False))
        self.assertEqual(changed, ["S1"])
        self.assertEqual(unchanged, [])
        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertEqual(held.visible_speaker, "S1")

    def test_silence_release_uses_tick_count_not_wall_clock(self) -> None:
        config = LiveSpeakerAlgorithmConfig(
            ema_count=1, min_known_probability=0.2, silence_release_count=2
        )
        algorithm = CausalLiveSpeakerAlgorithm(config, self.events)
        algorithm.step(LiveSpeakerStep(0.2, True, np.asarray([1.0, 0.0]), 1.0))
        first = algorithm.step(LiveSpeakerStep(0.4, False, None, 1.0, release_signal=True))
        second = algorithm.step(LiveSpeakerStep(0.6, False, None, 1.0, release_signal=True))
        self.assertEqual(first.visible_speaker, "S1")
        self.assertIsNone(second.visible_speaker)

    def test_non_probe_tick_holds_without_counting_unknown(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config, self.events)
        acquired = algorithm.step(LiveSpeakerStep(0.2, True, np.asarray([1.0, 0.0]), 1.0))
        held = algorithm.step(
            LiveSpeakerStep(0.4, False, None, 1.0, probe_scheduled=False)
        )
        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertEqual(held.visible_speaker, "S1")
        self.assertEqual(held.reason, "non_probe_tick")
        self.assertEqual(held.diagnostics["unknown_count"], 0)

    def test_failed_scheduled_probe_is_unknown_but_probe_gate_silence_is_not(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config, self.events)
        algorithm.step(LiveSpeakerStep(0.2, True, np.asarray([1.0, 0.0]), 1.0))
        silence = algorithm.step(LiveSpeakerStep(0.4, False, None, 1.0))
        failed = algorithm.step(
            LiveSpeakerStep(0.6, True, None, 1.0, skipped_reason="provider_error")
        )
        self.assertEqual(silence.reason, "probe_gate_silence")
        self.assertEqual(silence.diagnostics["unknown_count"], 0)
        self.assertEqual(failed.reason, "unknown_debounce")
        self.assertEqual(failed.diagnostics["unknown_count"], 1)

    def test_rejects_non_chronological_ticks(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config, self.events)
        algorithm.step(LiveSpeakerStep(1.0, False, None, 1.0))
        with self.assertRaises(ValueError):
            algorithm.step(LiveSpeakerStep(0.8, False, None, 1.0))

    def test_versioned_score_reports_release_and_robust_aggregate(self) -> None:
        algorithm = CausalLiveSpeakerAlgorithm(self.config, self.events[:1])
        decisions = [
            algorithm.step(LiveSpeakerStep(0.2, True, np.asarray([1.0, 0.0]), 1.0)),
            algorithm.step(LiveSpeakerStep(0.4, True, np.asarray([1.0, 0.0]), 1.0)),
            algorithm.step(LiveSpeakerStep(0.6, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(0.8, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(1.0, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(1.2, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(1.4, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(1.6, False, None, 1.0)),
            algorithm.step(LiveSpeakerStep(1.8, True, np.asarray([1.0, 0.0]), 1.0)),
            algorithm.step(LiveSpeakerStep(2.0, True, np.asarray([1.0, 0.0]), 1.0)),
        ]
        canonical = [
            {"speaker": "A", "start": 0.0, "end": 0.5},
            {"speaker": "A", "start": 1.7, "end": 2.1},
        ]
        score = score_live_speaker_decisions(decisions, canonical, self.events[:1])
        self.assertEqual(score["scorer_id"], "causal_live_speaker_score_v1")
        self.assertEqual(score["release"]["eligible_gap_count"], 1)
        aggregate = aggregate_video_scores([score, score])
        self.assertEqual(aggregate["video_count"], 2)
        self.assertGreaterEqual(aggregate["global_score"], 0.0)

    def test_primary_v2_is_plain_macro_mean_without_per_video_vetoes(self) -> None:
        scores = [
            {
                "strict_browser_live_score": 0.2,
                "correct_live_speaker_coverage": 0.4,
                "wrong_live_speech_ratio": 0.1,
                "outside_speech_live_ratio": 0.2,
                "missing_live_speech_ratio": 0.6,
                "canonical_speech_seconds": 10.0,
                "flicker": {"correct_interruption_seconds": 1.0},
            },
            {
                "strict_browser_live_score": 0.8,
                "correct_live_speaker_coverage": 0.9,
                "wrong_live_speech_ratio": 0.0,
                "outside_speech_live_ratio": 0.0,
                "missing_live_speech_ratio": 0.1,
                "canonical_speech_seconds": 30.0,
                "flicker": {"correct_interruption_seconds": 3.0},
            },
        ]
        aggregate = aggregate_video_scores_primary_v2(scores)
        self.assertEqual(aggregate["scorer_id"], PRIMARY_SCORER_V2_ID)
        self.assertEqual(aggregate["primary_score"], 0.5)
        self.assertEqual(aggregate["global_score"], 0.5)
        self.assertEqual(aggregate["diagnostics"]["corpus_flicker_ratio"], 0.1)


class CausalBayesSpeakerTrackerTests(unittest.TestCase):
    def test_voice_boundary_can_unlock_conservative_new_speaker_creation(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.7, 1.5),
                scale_weights=(0.8, 0.2),
                min_similarity=0.2,
                min_known_probability=0.2,
                enable_provisional_profiles=True,
                provisional_creation_count=1,
                provisional_creation_similarity_ceiling=-0.1,
                provisional_boundary_creation_similarity_ceiling=0.1,
                provisional_boundary_continuity_max_similarity=0.1,
                provisional_scale_agreement_min_similarity=0.5,
                incumbent_continuity_min_similarity=0.3,
            ),
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        first = tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))
        boundary = tracker.step(MultiScaleStep(1.2, True, (
            MultiScaleEvidence(0.7, np.asarray([0.0, 1.0])),
            MultiScaleEvidence(1.5, np.asarray([0.0, 1.0])),
        )))

        self.assertEqual(first.visible_speaker, "S1")
        self.assertEqual(boundary.visible_speaker, "provisional_1")
        self.assertEqual(boundary.reason, "provisional_acquire")

    def test_detected_boundary_can_suppress_stale_long_window(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.7, 1.5),
                scale_weights=(0.4, 0.6),
                min_similarity=0.1,
                min_known_probability=0.2,
                similarity_temperature=0.05,
                incumbent_continuity_min_similarity=0.3,
                boundary_short_only_max_continuity=0.1,
            ),
            [
                SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
                SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
            ],
        )
        first = tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))
        boundary = tracker.step(MultiScaleStep(1.2, True, (
            MultiScaleEvidence(0.7, np.asarray([0.0, 1.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))

        self.assertEqual(first.visible_speaker, "S1")
        self.assertEqual(boundary.visible_speaker, "S2")
        self.assertTrue(boundary.diagnostics["boundary_short_only"])

    def test_detected_boundary_can_remove_incumbent_from_mixed_short_window(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.7, 1.5),
                scale_weights=(0.8, 0.2),
                min_similarity=0.1,
                min_known_probability=0.2,
                similarity_temperature=0.05,
                incumbent_continuity_min_similarity=0.9,
                boundary_short_only_max_continuity=0.85,
                boundary_residual_incumbent_alpha=0.8,
            ),
            [
                SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
                SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
            ],
        )
        tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))
        boundary = tracker.step(MultiScaleStep(1.2, True, (
            MultiScaleEvidence(0.7, np.asarray([0.8, 0.6])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))

        self.assertEqual(boundary.visible_speaker, "S2")
        self.assertTrue(boundary.diagnostics["boundary_short_only"])
        self.assertAlmostEqual(
            boundary.diagnostics["boundary_residual_continuity"], 0.0, places=6
        )

    def test_event_attack_probes_early_only_until_a_speaker_is_visible(self) -> None:
        times = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        block = CachedLiveWindowBlock(
            provider="fake",
            video_id="video",
            window_seconds=0.2,
            media_times=times,
            embeddings=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
            valid=np.ones(4, dtype=bool),
            raw_rms=np.ones(4, dtype=np.float32),
            sample_rate=16_000,
        )
        decisions = replay_cached_bayes_windows(
            [block],
            [],
            np.ones(4, dtype=bool),
            np.asarray([False, False, False, True]),
            np.zeros(4, dtype=bool),
            config=BayesSpeakerTrackerConfig(
                scale_windows=(0.2,),
                scale_weights=(1.0,),
                enable_provisional_profiles=True,
                provisional_creation_count=1,
                provisional_creation_similarity_ceiling=1.0,
                provisional_scale_agreement_min_similarity=-1.0,
            ),
            attack_probe_interval_seconds=0.2,
        )

        self.assertEqual(decisions[0].visible_speaker, "provisional_1")
        self.assertTrue(decisions[0].diagnostics["probe_scheduled"])
        self.assertFalse(decisions[1].diagnostics["probe_scheduled"])

    def test_incumbent_hold_can_end_when_short_and_long_windows_diverge(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.7, 1.5), scale_weights=(0.8, 0.2),
                incumbent_hold_scale_agreement_min_similarity=0.7,
                unknown_release_count=1, min_known_probability=0.2,
            ),
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        acquired = tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))
        cleared = tracker.step(MultiScaleStep(1.0, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([0.0, 1.0])),
        )))
        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertIsNone(cleared.visible_speaker)

    def test_provisional_assignment_can_require_continued_scale_agreement(self) -> None:
        tracker = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            scale_windows=(0.7, 1.5),
            scale_weights=(0.8, 0.2),
            enable_provisional_profiles=True,
            provisional_creation_count=1,
            provisional_scale_agreement_min_similarity=0.7,
            provisional_assignment_scale_agreement_min_similarity=0.7,
            unknown_release_count=1,
            min_known_probability=0.2,
        ))
        acquired = tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([1.0, 0.0])),
        )))
        cleared = tracker.step(MultiScaleStep(1.0, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([0.0, 1.0])),
        )))
        self.assertEqual(acquired.visible_speaker, "provisional_1")
        self.assertIsNone(cleared.visible_speaker)
        self.assertEqual(cleared.reason, "bayes_unknown")

    def test_provisional_profile_requires_configured_scale_agreement(self) -> None:
        tracker = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            enable_provisional_profiles=True,
            provisional_creation_count=1,
            provisional_scale_agreement_min_similarity=0.5,
        ))
        decision = tracker.step(MultiScaleStep(0.8, True, (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(1.5, np.asarray([0.0, 1.0])),
        )))
        self.assertIsNone(decision.visible_speaker)
        self.assertEqual(decision.profile_count, 0)

    def test_provisional_profile_is_causally_created_and_later_merged(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                enable_provisional_profiles=True,
                provisional_creation_count=1,
                provisional_creation_similarity_ceiling=0.2,
                provisional_merge_min_similarity=0.5,
                min_known_probability=0.2,
            ),
            [SpeakerProfileEvent(1.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        evidence = (MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),)

        provisional = tracker.step(MultiScaleStep(0.8, True, evidence))
        merged = tracker.step(MultiScaleStep(1.0, True, evidence))

        self.assertEqual(provisional.visible_speaker, "provisional_1")
        self.assertEqual(merged.visible_speaker, "provisional_1")
        self.assertEqual(merged.diagnostics["profile_aliases"], {"S1": "provisional_1"})
        self.assertEqual(merged.diagnostics["provisional_profiles"], [])

    def test_cohort_normalization_penalizes_profile_close_to_other_profiles(self) -> None:
        tracker = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            profile_cohort_max_strength=0.5,
        ))
        tracker.sync_profiles([
            {"label": "S1", "centroid": [1.0, 0.0, 0.0]},
            {"label": "S2", "centroid": [0.99, 0.1, 0.0]},
            {"label": "S3", "centroid": [0.0, 0.0, 1.0]},
        ])
        evidence = (MultiScaleEvidence(0.7, np.asarray([1.0, 0.0, 0.0])),)

        scores = tracker._fused_similarities(evidence)

        self.assertLess(scores["S1"], 1.0)
        self.assertAlmostEqual(scores["S3"], 0.0, places=6)

    def test_confidence_weighting_lets_decisive_scale_override_ambiguous_scale(self) -> None:
        profiles = [
            {"label": "S1", "centroid": [1.0, 0.0]},
            {"label": "S2", "centroid": [0.0, 1.0]},
        ]
        evidences = (
            MultiScaleEvidence(0.7, np.asarray([1.0, 0.8])),
            MultiScaleEvidence(1.5, np.asarray([0.0, 1.0])),
        )
        fixed = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            scale_windows=(0.7, 1.5), scale_weights=(0.9, 0.1)
        ))
        adaptive = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            scale_windows=(0.7, 1.5), scale_weights=(0.9, 0.1),
            scale_confidence_power=1.0, scale_confidence_floor=0.02,
        ))
        fixed.sync_profiles(profiles)
        adaptive.sync_profiles(profiles)

        fixed_scores = fixed._fused_similarities(evidences)
        adaptive_scores = adaptive._fused_similarities(evidences)

        self.assertGreater(fixed_scores["S1"], fixed_scores["S2"])
        self.assertGreater(adaptive_scores["S2"], adaptive_scores["S1"])

    def test_production_profile_sync_preserves_bayesian_state(self) -> None:
        tracker = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            scale_windows=(0.8, 2.8),
            scale_weights=(0.8, 0.2),
            min_similarity=0.2,
            min_known_probability=0.2,
        ))
        profile = [{"label": "S1", "centroid": [1.0, 0.0]}]
        self.assertEqual(tracker.sync_profiles(profile), ["S1"])
        evidence = (
            MultiScaleEvidence(0.8, np.asarray([1.0, 0.0])),
            MultiScaleEvidence(2.8, np.asarray([1.0, 0.0])),
        )
        acquired = tracker.step(MultiScaleStep(0.8, True, evidence))
        self.assertEqual(tracker.sync_profiles(profile), [])
        held = tracker.step(MultiScaleStep(1.0, False, (), probe_scheduled=False))
        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertEqual(held.visible_speaker, "S1")

    def test_strong_new_evidence_overrides_markov_persistence(self) -> None:
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.8, 2.8),
                scale_weights=(0.8, 0.2),
                min_similarity=0.2,
                min_known_probability=0.25,
                similarity_temperature=0.05,
                stay_probability=0.8,
                prior_strength=1.0,
            ),
            events,
        )

        def step(media_time: float, vector: list[float]):
            evidence = tuple(
                MultiScaleEvidence(window, np.asarray(vector, dtype=np.float32))
                for window in (0.8, 2.8)
            )
            return tracker.step(MultiScaleStep(media_time, True, evidence))

        first = step(0.8, [1.0, 0.0])
        changed = step(1.6, [0.0, 1.0])
        self.assertEqual(first.visible_speaker, "S1")
        self.assertEqual(changed.visible_speaker, "S2")
        self.assertEqual(changed.action, "switch")

    def test_profile_remains_causally_unavailable_before_event(self) -> None:
        tracker = CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                scale_windows=(0.8, 2.8),
                scale_weights=(0.8, 0.2),
                min_similarity=0.2,
                min_known_probability=0.2,
            ),
            [
                SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
                SpeakerProfileEvent(1.0, "S2", np.asarray([0.0, 1.0]), generation=1),
            ],
        )
        evidence = (
            MultiScaleEvidence(0.8, np.asarray([0.0, 1.0])),
            MultiScaleEvidence(2.8, np.asarray([0.0, 1.0])),
        )
        before = tracker.step(MultiScaleStep(0.8, True, evidence))
        after = tracker.step(MultiScaleStep(1.0, True, evidence))
        self.assertNotEqual(before.visible_speaker, "S2")
        self.assertEqual(after.visible_speaker, "S2")

    def test_bounded_provisional_pool_reuses_and_adapts_recent_identity(self) -> None:
        tracker = CausalBayesSpeakerTracker(BayesSpeakerTrackerConfig(
            enable_provisional_profiles=True,
            provisional_creation_count=1,
            provisional_creation_similarity_ceiling=0.2,
            provisional_scale_agreement_min_similarity=-1.0,
            provisional_max_active_count=1,
            provisional_pool_overflow_update_alpha=0.5,
            min_similarity=0.3,
            min_known_probability=0.9,
        ))
        first = tracker.step(MultiScaleStep(
            0.8, True, (MultiScaleEvidence(0.7, np.asarray([1.0, 0.0])),)
        ))
        reused = tracker.step(MultiScaleStep(
            1.6, True, (MultiScaleEvidence(0.7, np.asarray([0.0, 1.0])),)
        ))

        self.assertEqual(first.visible_speaker, "provisional_1")
        self.assertEqual(reused.visible_speaker, "provisional_1")
        self.assertEqual(reused.profile_count, 1)
        self.assertEqual(reused.reason, "provisional_acquire")


if __name__ == "__main__":
    unittest.main()
