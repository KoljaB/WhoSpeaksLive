import unittest

import numpy as np

from window.live_speaker_algorithm import LiveSpeakerDecision, SpeakerProfileEvent
from window.live_speaker_hybrid import (
    CausalHybridSpeakerTracker,
    HybridSpeakerTrackerConfig,
    replay_hybrid_decisions,
)
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep


def decision(t, visible, action="hold", reason="legacy"):
    return LiveSpeakerDecision(
        media_time=t,
        visible_speaker=visible,
        action=action,
        reason=reason,
        candidate_speaker=visible,
        probabilities={},
        raw_probabilities={},
        similarities={},
        profile_count=2,
        profile_generations={},
        diagnostics={"legacy_marker": 7},
    )


def evidence(window, vector):
    return MultiScaleEvidence(window, np.asarray(vector, dtype=np.float64))


def probe(t, short, long):
    return MultiScaleStep(
        media_time=t,
        speech=True,
        evidences=(evidence(0.5, short), evidence(1.0, long)),
    )


def meta_probe(t, short, long):
    return MultiScaleStep(
        media_time=t,
        speech=True,
        evidences=(evidence(0.8, short), evidence(2.8, long)),
    )


def meta_config(**overrides):
    values = {
        "enable_young_profile_confirmation": True,
        "enable_young_profile_lease": True,
        "enable_profile_quality_short_scale_fast_lease": False,
        "enable_profile_quality_meta_lease": True,
        "young_min_similarity": 0.35,
        "young_min_margin": 0.08,
        "young_required_consecutive_probes": 1,
        "young_independent_scale_count": 2,
        "young_fast_independent_scale_count": 1,
    }
    values.update(overrides)
    return HybridSpeakerTrackerConfig(**values)


def profiles(
    second_available=0.0,
    second_end=None,
    second_sentence_count=1,
    second_speech_seconds=1.0,
):
    return [
        SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
        SpeakerProfileEvent(
            second_available,
            "S2",
            np.asarray([0.0, 1.0]),
            generation=1,
            sentence_count=second_sentence_count,
            speech_seconds=second_speech_seconds,
            sentence_start=None if second_end is None else max(0.0, second_end - 0.5),
            sentence_end=second_end,
        ),
    ]


class HybridSpeakerTrackerTests(unittest.TestCase):
    def test_disabled_features_return_exact_baseline_object(self):
        baseline = decision(1.0, "S2", action="switch")
        tracker = CausalHybridSpeakerTracker(profile_events=profiles())
        result = tracker.step(baseline, probe(1.0, [0, 1], [0, 1]))
        self.assertIs(result, baseline)
        self.assertEqual(result.action, "switch")

    def test_hard_limit_rejects_a_third_window(self):
        tracker = CausalHybridSpeakerTracker()
        item = MultiScaleStep(
            1.0,
            True,
            (evidence(0.5, [1, 0]), evidence(1.0, [1, 0]), evidence(1.5, [1, 0])),
        )
        with self.assertRaisesRegex(ValueError, "at most two"):
            tracker.step(decision(1.0, "S1"), item)

    def test_first_official_profile_is_never_probation_gated(self):
        config = HybridSpeakerTrackerConfig(enable_young_profile_confirmation=True)
        tracker = CausalHybridSpeakerTracker(config, profiles(second_available=5.0))
        baseline = decision(0.5, "S1", action="acquire")
        result = tracker.step(baseline, probe(0.5, [1, 0], [1, 0]))
        self.assertEqual(result.visible_speaker, "S1")
        self.assertEqual(result.action, "acquire")

    def test_immature_profile_requires_consecutive_two_scale_votes(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=2,
            young_min_similarity=0.4,
            young_min_margin=0.04,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles())
        tracker.step(decision(0.0, "S1", "acquire"), probe(0.0, [1, 0], [1, 0]))
        first = tracker.step(decision(1.0, "S2", "switch"), probe(1.0, [0, 1], [0, 1]))
        second = tracker.step(decision(2.0, "S2"), probe(2.0, [0, 1], [0, 1]))
        self.assertIsNone(first.visible_speaker)
        self.assertEqual(first.action, "clear")
        self.assertEqual(first.reason, "young_profile_confirmation_pending")
        self.assertEqual(second.visible_speaker, "S2")
        self.assertIn("S2", second.diagnostics["permanently_confirmed_profiles"])

    def test_self_echo_window_cannot_confirm_but_fresh_audio_can(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
            self_echo_guard_seconds=0.05,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles(second_available=1.0, second_end=1.0))
        tracker.step(decision(0.5, "S1", "acquire"), probe(0.5, [1, 0], [1, 0]))
        contaminated = tracker.step(decision(1.2, "S2", "switch"), probe(1.2, [0, 1], [0, 1]))
        fresh = tracker.step(decision(2.1, "S2"), probe(2.1, [0, 1], [0, 1]))
        self.assertIsNone(contaminated.visible_speaker)
        self.assertEqual(contaminated.diagnostics["contaminated_windows"], [0.5, 1.0])
        self.assertEqual(fresh.visible_speaker, "S2")
        self.assertEqual(fresh.diagnostics["independent_windows"], [0.5, 1.0])

    def test_fast_lease_holds_nonprobe_then_failed_renewal_clears(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            young_required_consecutive_probes=1,
            self_echo_guard_seconds=0.0,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles(second_available=1.0, second_end=1.0))
        tracker.step(decision(0.5, "S1", "acquire"), probe(0.5, [1, 0], [1, 0]))
        # At 1.6 the short 0.5s window is independent; the 1.0s window still overlaps.
        leased = tracker.step(decision(1.6, "S2", "switch"), probe(1.6, [0, 1], [0, 1]))
        held = tracker.step(
            decision(1.8, "S2"),
            MultiScaleStep(1.8, True, (), probe_scheduled=False),
        )
        failed = tracker.step(decision(2.0, "S2"), probe(2.0, [1, 0], [0, 1]))
        promoted = tracker.step(decision(2.2, "S2"), probe(2.2, [0, 1], [0, 1]))
        self.assertEqual(leased.visible_speaker, "S2")
        self.assertEqual(leased.diagnostics["leased_profile"], "S2")
        self.assertEqual(held.visible_speaker, "S2")
        self.assertEqual(held.diagnostics["hybrid_intervention"], "young_profile_lease_hold")
        self.assertIsNone(failed.visible_speaker)
        self.assertEqual(failed.action, "clear")
        self.assertEqual(promoted.visible_speaker, "S2")
        self.assertIn("S2", promoted.diagnostics["permanently_confirmed_profiles"])

    def test_short_only_fast_lease_is_disabled_by_default(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            young_required_consecutive_probes=1,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_available=0.0, second_end=0.0),
        )
        result = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [0, 1], [1, 0]),
        )
        self.assertIsNone(result.visible_speaker)
        self.assertEqual(result.reason, "young_profile_unconfirmed")
        self.assertEqual(result.diagnostics["young_probe_independent_scale_count"], 2)
        self.assertEqual(result.diagnostics["young_probe_valid_scale_count"], 1)
        self.assertTrue(result.diagnostics["young_probe_short_scale_valid"])
        self.assertFalse(result.diagnostics["short_scale_fast_lease_used"])

    def test_optional_short_scale_can_lease_without_permanent_confirmation(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            young_required_consecutive_probes=1,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_available=0.0, second_end=0.0),
        )
        result = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [0, 1], [1, 0]),
        )
        self.assertEqual(result.visible_speaker, "S2")
        self.assertEqual(result.diagnostics["leased_profile"], "S2")
        self.assertNotIn("S2", result.diagnostics["permanently_confirmed_profiles"])
        self.assertEqual(result.diagnostics["young_streak"], 0)
        self.assertEqual(result.diagnostics["young_probe_independent_scale_count"], 2)
        self.assertEqual(result.diagnostics["young_probe_valid_scale_count"], 1)
        self.assertTrue(result.diagnostics["young_probe_short_scale_valid"])
        self.assertTrue(result.diagnostics["short_scale_fast_lease_used"])

    def test_long_only_vote_cannot_use_short_scale_fast_lease(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            young_required_consecutive_probes=1,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_available=0.0, second_end=0.0),
        )
        result = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [1, 0], [0, 1]),
        )
        self.assertIsNone(result.visible_speaker)
        self.assertEqual(result.diagnostics["young_probe_valid_scale_count"], 1)
        self.assertFalse(result.diagnostics["young_probe_short_scale_valid"])
        self.assertFalse(result.diagnostics["short_scale_fast_lease_used"])

    def test_two_scale_fast_threshold_still_requires_two_valid_scales(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=2,
            young_required_consecutive_probes=1,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_available=0.0, second_end=0.0),
        )
        one_valid = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [0, 1], [1, 0]),
        )
        two_valid = tracker.step(
            decision(2.0, "S2", "switch"),
            probe(2.0, [0, 1], [0, 1]),
        )
        self.assertIsNone(one_valid.visible_speaker)
        self.assertFalse(one_valid.diagnostics["short_scale_fast_lease_used"])
        self.assertEqual(two_valid.visible_speaker, "S2")
        self.assertIn("S2", two_valid.diagnostics["permanently_confirmed_profiles"])

    def test_short_scale_lease_must_renew_on_the_next_probe(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_fast_independent_scale_count=1,
            young_required_consecutive_probes=1,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_available=0.0, second_end=0.0),
        )
        leased = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [0, 1], [1, 0]),
        )
        held = tracker.step(
            decision(1.4, "S2"),
            MultiScaleStep(1.4, True, (), probe_scheduled=False),
        )
        failed_renewal = tracker.step(
            decision(2.0, "S2"),
            probe(2.0, [1, 0], [0, 1]),
        )
        self.assertEqual(leased.visible_speaker, "S2")
        self.assertEqual(held.visible_speaker, "S2")
        self.assertEqual(held.diagnostics["hybrid_intervention"], "young_profile_lease_hold")
        self.assertIsNone(failed_renewal.visible_speaker)
        self.assertIsNone(failed_renewal.diagnostics["leased_profile"])

    def test_profile_quality_fast_lease_is_disabled_by_default(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
            young_min_similarity=0.35,
            young_min_margin=0.08,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_sentence_count=2),
        )
        result = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [-0.98, 0.2], [1.0, 0.0]),
        )
        self.assertIsNone(result.visible_speaker)
        self.assertFalse(result.diagnostics["profile_quality_fast_lease_used"])

    def test_profile_quality_fast_lease_accepts_sentence_or_speech_evidence(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_profile_quality_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
            young_min_similarity=0.35,
            young_min_margin=0.08,
        )
        qualifying_profiles = (
            profiles(second_sentence_count=2, second_speech_seconds=1.0),
            profiles(second_sentence_count=1, second_speech_seconds=3.1),
        )
        for profile_events in qualifying_profiles:
            with self.subTest(profile=profile_events[1]):
                tracker = CausalHybridSpeakerTracker(config, profile_events)
                result = tracker.step(
                    decision(1.2, "S2", "switch"),
                    probe(1.2, [-0.98, 0.2], [1.0, 0.0]),
                )
                self.assertEqual(result.visible_speaker, "S2")
                self.assertEqual(result.diagnostics["leased_profile"], "S2")
                self.assertNotIn("S2", result.diagnostics["permanently_confirmed_profiles"])
                self.assertTrue(result.diagnostics["profile_quality_fast_lease_used"])
                details = result.diagnostics["profile_quality_fast_lease"]
                self.assertTrue(details["profile_eligible"])
                self.assertTrue(details["short_independent"])
                self.assertTrue(details["short_top_matches_target"])
                self.assertTrue(details["vote"])

    def test_profile_quality_fast_lease_rejects_weak_profile_and_long_only_vote(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_profile_quality_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
        )
        weak_profile = CausalHybridSpeakerTracker(config, profiles())
        weak_result = weak_profile.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [-0.98, 0.2], [1.0, 0.0]),
        )
        long_only = CausalHybridSpeakerTracker(
            config,
            profiles(second_sentence_count=2),
        )
        long_result = long_only.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [1.0, 0.0], [-0.98, 0.2]),
        )
        self.assertIsNone(weak_result.visible_speaker)
        self.assertFalse(weak_result.diagnostics["profile_quality_fast_lease"]["profile_eligible"])
        self.assertIsNone(long_result.visible_speaker)
        self.assertFalse(long_result.diagnostics["profile_quality_fast_lease"]["short_top_matches_target"])
        self.assertFalse(long_result.diagnostics["profile_quality_fast_lease_used"])

    def test_profile_quality_lease_does_not_relax_permanent_confirmation(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            enable_profile_quality_short_scale_fast_lease=True,
            young_independent_scale_count=2,
            young_required_consecutive_probes=1,
            young_min_similarity=0.35,
            young_min_margin=0.08,
        )
        tracker = CausalHybridSpeakerTracker(
            config,
            profiles(second_sentence_count=2),
        )
        leased = tracker.step(
            decision(1.2, "S2", "switch"),
            probe(1.2, [-0.98, 0.2], [-0.98, 0.2]),
        )
        confirmed = tracker.step(
            decision(2.0, "S2"),
            probe(2.0, [0.0, 1.0], [0.0, 1.0]),
        )
        self.assertEqual(leased.visible_speaker, "S2")
        self.assertNotIn("S2", leased.diagnostics["permanently_confirmed_profiles"])
        self.assertEqual(leased.diagnostics["young_streak"], 0)
        self.assertEqual(confirmed.visible_speaker, "S2")
        self.assertIn("S2", confirmed.diagnostics["permanently_confirmed_profiles"])

    def test_profile_quality_meta_defaults_freeze_a005_s035_contract(self):
        config = HybridSpeakerTrackerConfig()
        self.assertFalse(config.enable_profile_quality_meta_lease)
        self.assertEqual(config.profile_quality_meta_fresh_min_age_seconds, 0.05)
        self.assertEqual(config.profile_quality_meta_fresh_max_age_seconds, 0.8)
        self.assertEqual(config.profile_quality_meta_fresh_min_speech_seconds, 3.8)
        self.assertEqual(config.profile_quality_meta_fresh_min_short_margin, 0.30)
        self.assertEqual(config.profile_quality_meta_fresh_min_long_margin, 0.70)
        self.assertEqual(config.profile_quality_meta_independent_max_profile_count, 8)
        self.assertEqual(config.profile_quality_meta_switch_min_short_margin, 0.35)

    def test_profile_quality_meta_fresh_lease_holds_only_until_next_probe(self):
        tracker = CausalHybridSpeakerTracker(
            meta_config(),
            profiles(
                second_available=1.0,
                second_end=1.0,
                second_speech_seconds=3.8,
            ),
        )
        leased = tracker.step(
            decision(1.05, "S2", "hold"),
            meta_probe(1.05, [0.0, 0.30], [0.0, 0.70]),
        )
        self.assertEqual(leased.action, "acquire")
        self.assertEqual(tracker.visible_speaker, leased.visible_speaker)
        held = tracker.step(
            decision(1.25, "S2"),
            MultiScaleStep(1.25, True, (), probe_scheduled=False),
        )
        self.assertEqual(held.action, "hold")
        self.assertEqual(tracker.visible_speaker, held.visible_speaker)
        expired = tracker.step(
            decision(1.85, "S2"),
            meta_probe(1.85, [1.0, 0.0], [1.0, 0.0]),
        )
        self.assertEqual(expired.action, "clear")
        self.assertEqual(tracker.visible_speaker, expired.visible_speaker)
        self.assertEqual(leased.visible_speaker, "S2")
        self.assertEqual(leased.reason, "profile_quality_meta_fresh_lease")
        self.assertEqual(
            leased.diagnostics["profile_quality_meta_lease"]["branch"], "fresh"
        )
        self.assertEqual(held.visible_speaker, "S2")
        self.assertEqual(held.reason, "profile_quality_meta_lease_hold")
        self.assertIsNone(expired.visible_speaker)
        self.assertTrue(
            expired.diagnostics["profile_quality_meta_lease"]["lease_expired_at_probe"]
        )
        self.assertEqual(
            expired.diagnostics["profile_quality_meta_lease"]["state_reason"],
            "evidence_rejected",
        )

    def test_profile_quality_meta_independent_non_switch_and_switch_rules(self):
        profile_events = profiles(
            second_available=0.0,
            second_end=0.0,
            second_sentence_count=2,
            second_speech_seconds=1.0,
        )
        non_switch = CausalHybridSpeakerTracker(meta_config(), profile_events)
        non_switch_result = non_switch.step(
            decision(1.2, "S2", "hold"),
            meta_probe(1.2, [-0.98, 0.20], [0.0, 1.0]),
        )
        switch_profiles = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0, 0.0]), generation=1),
            SpeakerProfileEvent(
                0.0,
                "S2",
                np.asarray([0.0, 1.0, 0.0]),
                generation=1,
                sentence_count=2,
                speech_seconds=1.0,
                sentence_start=0.0,
                sentence_end=0.0,
            ),
        ]
        switch = CausalHybridSpeakerTracker(meta_config(), switch_profiles)
        switch_result = switch.step(
            decision(1.2, "S2", "switch"),
            meta_probe(1.2, [-0.15, 0.20, np.sqrt(0.9375)], [1.0, 0.0, 0.0]),
        )
        below_switch_margin = CausalHybridSpeakerTracker(meta_config(), switch_profiles)
        rejected = below_switch_margin.step(
            decision(1.2, "S2", "switch"),
            meta_probe(
                1.2,
                [-0.149, 0.20, np.sqrt(1.0 - 0.149**2 - 0.20**2)],
                [1.0, 0.0, 0.0],
            ),
        )
        self.assertEqual(non_switch_result.visible_speaker, "S2")
        self.assertEqual(
            non_switch_result.diagnostics["profile_quality_meta_lease"]["branch"],
            "independent",
        )
        self.assertEqual(switch_result.visible_speaker, "S2")
        self.assertAlmostEqual(
            switch_result.diagnostics["profile_quality_meta_lease"]["short_target_margin"],
            0.35,
        )
        self.assertIsNone(rejected.visible_speaker)

    def test_profile_quality_meta_non_switch_rejects_crowded_profile_set(self):
        profile_events = profiles(
            second_end=0.0,
            second_sentence_count=2,
        )
        profile_events.extend(
            SpeakerProfileEvent(
                0.0,
                f"S{index}",
                np.asarray([1.0, 0.0]),
                generation=1,
            )
            for index in range(3, 10)
        )
        tracker = CausalHybridSpeakerTracker(meta_config(), profile_events)
        result = tracker.step(
            decision(1.2, "S2", "hold"),
            meta_probe(1.2, [-0.98, 0.20], [0.0, 1.0]),
        )
        self.assertIsNone(result.visible_speaker)
        details = result.diagnostics["profile_quality_meta_lease"]
        self.assertEqual(details["active_profile_count"], 9)
        self.assertTrue(details["profile_quality_qvote"])
        self.assertEqual(details["state_reason"], "evidence_rejected")

    def test_profile_quality_meta_lease_resets_on_agreement_release_target_and_profile(self):
        def leased_tracker(extra_events=()):
            tracker = CausalHybridSpeakerTracker(
                meta_config(),
                [
                    *profiles(
                        second_available=1.0,
                        second_end=1.0,
                        second_speech_seconds=3.8,
                    ),
                    *extra_events,
                ],
            )
            first = tracker.step(
                decision(1.05, "S2", "acquire"),
                meta_probe(1.05, [0.0, 0.30], [0.0, 0.70]),
            )
            self.assertEqual(first.visible_speaker, "S2")
            return tracker

        agreement = leased_tracker().step(
            decision(1.25, "S1", "switch"),
            MultiScaleStep(1.25, True, (), probe_scheduled=False),
        )
        self.assertEqual(agreement.visible_speaker, "S1")
        self.assertEqual(agreement.action, "switch")
        self.assertEqual(agreement.diagnostics["profile_quality_meta_previous_emitted_visible"], "S2")
        self.assertIsNone(agreement.diagnostics["profile_quality_meta_leased_profile"])
        self.assertEqual(
            agreement.diagnostics["profile_quality_meta_lease"]["state_reason"],
            "expert_agreement",
        )

        released = leased_tracker().step(
            decision(1.25, None, "clear"),
            MultiScaleStep(1.25, False, (), probe_scheduled=False, release_signal=True),
        )
        self.assertIsNone(released.diagnostics["profile_quality_meta_leased_profile"])
        self.assertEqual(released.action, "clear")
        self.assertEqual(
            released.diagnostics["profile_quality_meta_lease"]["state_reason"], "release"
        )

        third = SpeakerProfileEvent(
            0.0,
            "S3",
            np.asarray([-1.0, 0.0]),
            generation=1,
            sentence_count=1,
            speech_seconds=1.0,
            sentence_start=0.0,
            sentence_end=0.0,
        )
        changed = leased_tracker((third,)).step(
            decision(1.25, "S3", "switch"),
            MultiScaleStep(1.25, True, (), probe_scheduled=False),
        )
        self.assertIsNone(changed.visible_speaker)
        self.assertEqual(
            changed.diagnostics["profile_quality_meta_lease"]["state_reason"],
            "target_changed",
        )

        refreshed = SpeakerProfileEvent(
            1.2,
            "S2",
            np.asarray([0.0, 1.0]),
            generation=2,
            sentence_count=2,
            speech_seconds=4.0,
            sentence_start=0.7,
            sentence_end=1.2,
        )
        profile_reset = leased_tracker((refreshed,)).step(
            decision(1.25, "S2"),
            MultiScaleStep(1.25, True, (), probe_scheduled=False),
        )
        self.assertIsNone(profile_reset.visible_speaker)
        self.assertIn("profile_change", profile_reset.diagnostics["history_reset_reasons"])
        self.assertIsNone(profile_reset.diagnostics["profile_quality_meta_leased_profile"])

    def test_boundary_abstains_after_debounce_and_recovers_on_reconvergence(self):
        config = HybridSpeakerTrackerConfig(
            enable_boundary_abstention=True,
            boundary_required_consecutive_probes=2,
            boundary_min_similarity=0.2,
            boundary_min_margin=0.0,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles())
        tracker.step(decision(0.0, "S1", "acquire"), probe(0.0, [1, 0], [1, 0]))
        first = tracker.step(decision(0.5, "S1"), probe(0.5, [0, 1], [1, 0]))
        second = tracker.step(decision(1.0, "S1"), probe(1.0, [0, 1], [1, 0]))
        nonprobe = tracker.step(
            decision(1.2, "S1"), MultiScaleStep(1.2, True, (), probe_scheduled=False)
        )
        recovered = tracker.step(decision(1.5, "S1"), probe(1.5, [1, 0], [1, 0]))
        self.assertEqual(first.visible_speaker, "S1")
        self.assertIsNone(second.visible_speaker)
        self.assertEqual(second.reason, "boundary_abstention")
        self.assertIsNone(nonprobe.visible_speaker)
        self.assertEqual(recovered.visible_speaker, "S1")
        self.assertEqual(recovered.action, "acquire")

    def test_boundary_recovers_immediately_when_baseline_switches(self):
        config = HybridSpeakerTrackerConfig(
            enable_boundary_abstention=True,
            boundary_required_consecutive_probes=1,
            boundary_min_similarity=0.2,
            boundary_min_margin=0.0,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles())
        tracker.step(decision(0.0, "S1", "acquire"), probe(0.0, [1, 0], [1, 0]))
        blocked = tracker.step(decision(0.5, "S1"), probe(0.5, [0, 1], [1, 0]))
        switched = tracker.step(decision(1.0, "S2", "switch"), probe(1.0, [0, 1], [1, 0]))
        self.assertIsNone(blocked.visible_speaker)
        self.assertEqual(switched.visible_speaker, "S2")
        self.assertEqual(switched.action, "switch")

    def test_release_resets_lease_and_replay_requires_paired_lengths(self):
        config = HybridSpeakerTrackerConfig(
            enable_young_profile_confirmation=True,
            enable_young_profile_lease=True,
            young_independent_scale_count=2,
        )
        tracker = CausalHybridSpeakerTracker(config, profiles(second_end=0.0))
        tracker.step(decision(1.0, "S2", "switch"), probe(1.0, [0, 1], [0, 1]))
        released = tracker.step(
            decision(1.2, None, "clear"),
            MultiScaleStep(1.2, False, (), probe_scheduled=False, release_signal=True),
        )
        self.assertIsNone(released.diagnostics["leased_profile"])
        self.assertIn("release", released.diagnostics["history_reset_reasons"])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            replay_hybrid_decisions([decision(0.0, None)], [], config=config)


if __name__ == "__main__":
    unittest.main()
