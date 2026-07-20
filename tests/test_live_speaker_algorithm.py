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
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
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


if __name__ == "__main__":
    unittest.main()
