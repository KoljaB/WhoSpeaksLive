from __future__ import annotations

import unittest

import numpy as np

from window.live_speaker_algorithm import SpeakerProfileEvent
from window.live_speaker_multiscale import (
    CausalMultiScaleSpeakerTracker,
    MultiScaleEvidence,
    MultiScaleStep,
    MultiScaleTrackerConfig,
)


def evidence(seconds: float, values: list[float]) -> MultiScaleEvidence:
    return MultiScaleEvidence(seconds, np.asarray(values, dtype=np.float32))


class CausalMultiScaleSpeakerTrackerTests(unittest.TestCase):
    def test_transition_abstention_clears_mixed_boundary_then_acquires_new_speaker(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            acquire_scale_agreement=2,
            enable_consensus=True,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_short_advantage=0.1,
            transition_scale_gap=0.2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        acquired = tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        boundary = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        acquired_from_history = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        switched = tracker.step(MultiScaleStep(
            3.2, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.0, 1.0]))
        ))
        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertIsNone(boundary.visible_speaker)
        self.assertEqual(boundary.action, "clear")
        self.assertEqual(boundary.reason, "transition_abstain")
        self.assertEqual(boundary.diagnostics["tracking_state"], "TRANSITION")
        self.assertEqual(acquired_from_history.visible_speaker, "S2")
        self.assertEqual(acquired_from_history.reason, "transition_history_acquire")
        self.assertEqual(acquired_from_history.diagnostics["tracking_state"], "STABLE")
        self.assertEqual(switched.visible_speaker, "S2")

    def test_transition_abstention_false_alarm_reacquires_previous_speaker(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            acquire_scale_agreement=2,
            enable_consensus=True,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_short_advantage=0.1,
            transition_scale_gap=0.2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        recovered = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        self.assertEqual(recovered.visible_speaker, "S1")
        self.assertEqual(recovered.action, "acquire")
        self.assertEqual(recovered.reason, "transition_false_alarm_revert")
        self.assertEqual(recovered.diagnostics["tracking_state"], "STABLE")

    def test_transition_stays_off_when_strong_challengers_alternate(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_min_similarity=0.2,
            transition_min_margin=0.05,
            transition_short_advantage=0.1,
            transition_scale_gap=0.2,
            transition_acquire_history_size=3,
            transition_acquire_required=2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S3", np.asarray([0.0, 0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0, 0.0]), evidence(2.8, [1.0, 0.0, 0.0]))
        ))
        entered = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0, 0.0]), evidence(2.8, [1.0, 0.0, 0.0]))
        ))
        competing = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 0.0, 1.0]), evidence(2.8, [1.0, 0.0, 0.0]))
        ))
        self.assertIsNone(entered.visible_speaker)
        self.assertIsNone(competing.visible_speaker)
        self.assertEqual(competing.reason, "transition_wait")
        self.assertEqual(competing.diagnostics["tracking_state"], "TRANSITION")

    def test_non_probe_tick_does_not_count_toward_transition_acquisition(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_min_similarity=0.2,
            transition_short_advantage=0.1,
            transition_scale_gap=0.2,
            transition_acquire_required=2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        non_probe = tracker.step(MultiScaleStep(
            2.0, True, (), probe_scheduled=False
        ))
        acquired = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        self.assertIsNone(non_probe.visible_speaker)
        self.assertEqual(non_probe.reason, "transition_wait_non_probe")
        self.assertEqual(non_probe.diagnostics["transition_off_probes"], 0)
        self.assertEqual(acquired.visible_speaker, "S2")

    def test_fast_embedding_change_detects_unknown_voice_without_profile_scores(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.35, 0.65),
            min_similarity=0.25,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_incumbent_max_similarity=-1.0,
            transition_min_similarity=0.25,
            enable_transition_embedding_change=True,
            transition_embedding_history_size=3,
            transition_embedding_min_history=2,
            transition_embedding_max_similarity=0.5,
            transition_embedding_drop=0.2,
        )
        tracker = CausalMultiScaleSpeakerTracker(
            config,
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        changed = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        still_off = tracker.step(MultiScaleStep(
            3.2, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        self.assertIsNone(changed.visible_speaker)
        self.assertEqual(changed.action, "clear")
        self.assertEqual(changed.reason, "transition_embedding_change")
        self.assertEqual(changed.diagnostics["transition_entry_kind"], "embedding_change")
        self.assertIsNone(still_off.visible_speaker)
        self.assertEqual(still_off.diagnostics["tracking_state"], "TRANSITION")

    def test_profile_event_is_not_visible_before_its_available_time_during_transition(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            acquire_scale_agreement=2,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_incumbent_max_similarity=0.3,
            transition_incumbent_drop=0.2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(3.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        before = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        after = tracker.step(MultiScaleStep(
            3.0, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.0, 1.0]))
        ))
        self.assertEqual(before.profile_count, 1)
        self.assertNotEqual(before.visible_speaker, "S2")
        self.assertEqual(after.profile_count, 2)

    def test_transition_timeout_discards_incumbent_context(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_min_similarity=0.2,
            transition_short_advantage=0.1,
            transition_scale_gap=0.2,
            transition_timeout_seconds=1.0,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        expired = tracker.step(MultiScaleStep(
            2.7, True, (), probe_scheduled=False
        ))
        self.assertIsNone(expired.visible_speaker)
        self.assertEqual(expired.reason, "transition_timeout")
        self.assertEqual(expired.diagnostics["tracking_state"], "OFF")
        self.assertIsNone(expired.diagnostics["transition_incumbent"])

    def test_speech_gate_false_clears_then_strong_same_speaker_reverts(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            enable_transition_speech_gate=True,
            transition_speech_gate_clear_required=1,
            transition_min_similarity=0.2,
        )
        tracker = CausalMultiScaleSpeakerTracker(
            config,
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        cleared = tracker.step(MultiScaleStep(
            1.0, False, (), probe_scheduled=False
        ))
        cadence_only = tracker.step(MultiScaleStep(
            1.2, True, (), probe_scheduled=False
        ))
        reverted = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        self.assertIsNone(cleared.visible_speaker)
        self.assertEqual(cleared.action, "clear")
        self.assertEqual(cleared.reason, "transition_speech_gate")
        self.assertEqual(cleared.diagnostics["tracking_state"], "TRANSITION")
        self.assertIsNone(cadence_only.visible_speaker)
        self.assertEqual(cadence_only.reason, "transition_wait_non_probe")
        self.assertEqual(reverted.visible_speaker, "S1")
        self.assertEqual(reverted.reason, "transition_false_alarm_revert")

    def test_speech_gate_transition_stays_off_for_unknown_then_acquires_challenger(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            enable_transition_speech_gate=True,
            transition_min_similarity=0.2,
            transition_min_margin=0.05,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        tracker.step(MultiScaleStep(
            1.0, False, (), probe_scheduled=False
        ))
        ambiguous = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [1.0, 1.0]), evidence(2.8, [1.0, 1.0]))
        ))
        acquired = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.0, 1.0]))
        ))
        self.assertIsNone(ambiguous.visible_speaker)
        self.assertEqual(ambiguous.reason, "transition_wait")
        self.assertEqual(acquired.visible_speaker, "S2")
        self.assertEqual(acquired.reason, "transition_multiscale_acquire")

    def test_speech_gate_disabled_preserves_legacy_hold(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            enable_transition_speech_gate=False,
        )
        tracker = CausalMultiScaleSpeakerTracker(
            config,
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        legacy = tracker.step(MultiScaleStep(
            1.0, False, (), probe_scheduled=False
        ))
        self.assertEqual(legacy.visible_speaker, "S1")
        self.assertEqual(legacy.action, "hold")
        self.assertEqual(legacy.reason, "non_probe_tick")
        self.assertEqual(legacy.diagnostics["tracking_state"], "STABLE")

    def test_non_probe_speech_gate_counts_false_ticks_and_true_tick_resets(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            enable_transition_speech_gate=True,
            transition_speech_gate_clear_required=2,
        )
        tracker = CausalMultiScaleSpeakerTracker(
            config,
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        first_false = tracker.step(MultiScaleStep(
            1.0, False, (), probe_scheduled=False
        ))
        cadence_true = tracker.step(MultiScaleStep(
            1.2, True, (), probe_scheduled=False
        ))
        restarted_false = tracker.step(MultiScaleStep(
            1.4, False, (), probe_scheduled=False
        ))
        second_consecutive_false = tracker.step(MultiScaleStep(
            1.6, False, (), probe_scheduled=False
        ))
        self.assertEqual(first_false.visible_speaker, "S1")
        self.assertEqual(first_false.reason, "transition_speech_gate_debounce")
        self.assertEqual(cadence_true.visible_speaker, "S1")
        self.assertEqual(cadence_true.reason, "non_probe_tick")
        self.assertEqual(restarted_false.visible_speaker, "S1")
        self.assertEqual(restarted_false.reason, "transition_speech_gate_debounce")
        self.assertIsNone(second_consecutive_false.visible_speaker)
        self.assertEqual(second_consecutive_false.reason, "transition_speech_gate")

    def test_later_profile_is_not_assignable_until_causally_mature(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.5, 0.5),
            min_similarity=0.2,
            min_margin=0.0,
            acquire_scale_agreement=1,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            trusted_profile_min_sentence_count=3,
            trusted_profile_min_speech_seconds=5.0,
        )
        events = [
            SpeakerProfileEvent(
                0.0, "S1", np.asarray([1.0, 0.0]),
                speech_seconds=2.0, sentence_count=1, generation=1,
            ),
            SpeakerProfileEvent(
                1.0, "S2", np.asarray([0.0, 1.0]),
                speech_seconds=2.0, sentence_count=1, generation=1,
            ),
            SpeakerProfileEvent(
                2.0, "S2", np.asarray([0.0, 1.0]),
                speech_seconds=6.0, sentence_count=3, generation=2,
            ),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        first = tracker.step(MultiScaleStep(
            0.2, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        immature = tracker.step(MultiScaleStep(
            1.2, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.0, 1.0]))
        ))
        mature = tracker.step(MultiScaleStep(
            2.2, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.0, 1.0]))
        ))
        self.assertEqual(first.visible_speaker, "S1")
        self.assertNotEqual(immature.visible_speaker, "S2")
        self.assertIn("S2", immature.diagnostics["untrusted_profiles"])
        self.assertEqual(mature.visible_speaker, "S2")
        self.assertNotIn("S2", mature.diagnostics["untrusted_profiles"])

    def test_duration_matched_profile_is_seeded_only_when_final_event_becomes_available(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(1.0,),
            min_similarity=-0.1,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
            enable_duration_matched_profiles=True,
            duration_profile_score_weight=1.0,
            duration_profile_min_windows=2,
            duration_profile_min_cohesion=0.0,
            duration_profile_guard_seconds=0.0,
        )
        events = [
            SpeakerProfileEvent(
                0.0,
                "S2",
                np.asarray([-1.0, 0.0]),
                generation=1,
            ),
            SpeakerProfileEvent(
                3.0,
                "S1",
                np.asarray([1.0, 0.0]),
                generation=1,
                sentence_start=0.0,
                sentence_end=2.5,
            ),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        before = tracker.step(MultiScaleStep(1.0, True, (evidence(0.8, [0.0, 1.0]),)))
        tracker.step(MultiScaleStep(2.0, True, (evidence(0.8, [0.0, 1.0]),)))
        after = tracker.step(MultiScaleStep(3.0, True, (evidence(0.8, [0.0, 1.0]),)))
        self.assertNotEqual(before.visible_speaker, "S1")
        self.assertEqual(before.diagnostics["duration_profile_count"], 0)
        self.assertEqual(after.visible_speaker, "S1")
        self.assertEqual(after.diagnostics["duration_profile_count"], 1)

    def test_incumbent_rejection_clears_unknown_new_voice_without_known_challenger(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.35, 0.65),
            min_similarity=0.25,
            min_margin=0.0,
            acquire_scale_agreement=2,
            enable_consensus=True,
            enable_crossover=False,
            enable_history=False,
            enable_transition_abstention=True,
            transition_incumbent_max_similarity=0.3,
            transition_incumbent_drop=0.2,
        )
        tracker = CausalMultiScaleSpeakerTracker(
            config,
            [SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1)],
        )
        tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        rejected = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        still_off = tracker.step(MultiScaleStep(
            2.4, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [1.0, 0.0]))
        ))
        recovered = tracker.step(MultiScaleStep(
            3.2, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        self.assertIsNone(rejected.visible_speaker)
        self.assertEqual(rejected.reason, "transition_incumbent_rejection")
        self.assertIsNone(still_off.visible_speaker)
        self.assertEqual(recovered.visible_speaker, "S1")

    def test_short_long_crossover_switches_before_long_window_catches_up(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.7, 0.3),
            min_similarity=0.2,
            min_margin=0.0,
            min_scale_agreement=2,
            enable_consensus=True,
            enable_crossover=True,
            enable_history=False,
            crossover_short_advantage=0.2,
            crossover_scale_gap=0.5,
            crossover_required=1,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        first = tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0]))
        ))
        changed = tracker.step(MultiScaleStep(
            1.6, True, (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.9, 0.1]))
        ))
        self.assertEqual(first.visible_speaker, "S1")
        self.assertEqual(changed.visible_speaker, "S2")
        self.assertEqual(changed.reason, "scale_crossover")

    def test_history_rejects_one_probe_spike_but_accepts_repeated_evidence(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(1.0,),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=True,
            history_size=5,
            history_required=3,
            history_advantage=0.2,
        )
        events = [
            SpeakerProfileEvent(0.0, "S1", np.asarray([1.0, 0.0]), generation=1),
            SpeakerProfileEvent(0.0, "S2", np.asarray([0.0, 1.0]), generation=1),
        ]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        tracker.step(MultiScaleStep(0.8, True, (evidence(0.8, [1.0, 0.0]),)))
        spike = tracker.step(MultiScaleStep(1.6, True, (evidence(0.8, [0.0, 1.0]),)))
        tracker.step(MultiScaleStep(2.4, True, (evidence(0.8, [0.0, 1.0]),)))
        confirmed = tracker.step(MultiScaleStep(3.2, True, (evidence(0.8, [0.0, 1.0]),)))
        self.assertEqual(spike.visible_speaker, "S1")
        self.assertEqual(confirmed.visible_speaker, "S2")
        self.assertEqual(confirmed.reason, "similarity_history")

    def test_provisional_profile_is_created_then_absorbs_official_profile(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_weights=(0.7, 0.3),
            min_similarity=0.2,
            min_margin=0.0,
            enable_online_profiles=True,
            provisional_confirm_count=1,
            provisional_first_immediate=True,
            official_merge_similarity=0.4,
        )
        events = [SpeakerProfileEvent(2.0, "S1", np.asarray([1.0, 0.0]), generation=1)]
        tracker = CausalMultiScaleSpeakerTracker(config, events)
        provisional = tracker.step(MultiScaleStep(
            0.8, True, (evidence(0.8, [1.0, 0.0]), evidence(1.4, [1.0, 0.0]))
        ))
        merged = tracker.step(MultiScaleStep(
            2.0, True, (evidence(0.8, [1.0, 0.0]), evidence(1.4, [1.0, 0.0]))
        ))
        self.assertEqual(provisional.visible_speaker, "P1")
        self.assertEqual(merged.visible_speaker, "P1")
        self.assertEqual(merged.diagnostics["profile_aliases"]["S1"], "P1")
        self.assertEqual(merged.diagnostics["provisional_profiles"], [])


if __name__ == "__main__":
    unittest.main()
