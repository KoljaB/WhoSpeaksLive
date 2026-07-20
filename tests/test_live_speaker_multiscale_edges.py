from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from window.live_speaker_algorithm import SpeakerProfileEvent
from window.live_speaker_multiscale import (
    MULTISCALE_ALGORITHM_ID,
    CausalMultiScaleSpeakerTracker,
    MultiScaleEvidence,
    MultiScaleStep,
    MultiScaleTrackerConfig,
    replay_cached_multiscale_windows,
)
from window.live_speaker_replay import load_profile_events_jsonl
from window.live_speaker_replay import CachedLiveWindowBlock


def evidence(seconds: float, values: list[float]) -> MultiScaleEvidence:
    return MultiScaleEvidence(seconds, np.asarray(values, dtype=np.float32))


def event(
    available_at: float,
    speaker_id: str,
    values: list[float],
    *,
    generation: int,
) -> SpeakerProfileEvent:
    return SpeakerProfileEvent(
        available_at,
        speaker_id,
        np.asarray(values, dtype=np.float32),
        generation=generation,
    )


class CausalMultiScaleSpeakerTrackerEdgeTests(unittest.TestCase):
    def test_profile_event_loader_preserves_optional_sentence_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiles.jsonl"
            path.write_text(json.dumps({
                "available_at": 3.0,
                "sentence_start": 0.5,
                "sentence_end": 2.75,
                "speaker_id": "S1",
                "profile_generation": 1,
                "centroid": [1.0, 0.0],
            }) + "\n", encoding="utf-8")
            loaded = load_profile_events_jsonl(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sentence_start, 0.5)
        self.assertEqual(loaded[0].sentence_end, 2.75)

    def test_evidence_order_does_not_change_decisions_or_weights(self) -> None:
        config = MultiScaleTrackerConfig(
            scale_windows=(0.8, 2.8),
            scale_weights=(0.7, 0.3),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=True,
            enable_history=False,
            crossover_required=1,
            crossover_short_advantage=0.2,
            crossover_scale_gap=0.5,
        )
        events = [
            event(0.0, "S1", [1.0, 0.0], generation=1),
            event(0.0, "S2", [0.0, 1.0], generation=1),
        ]
        sorted_tracker = CausalMultiScaleSpeakerTracker(config, events)
        reversed_tracker = CausalMultiScaleSpeakerTracker(config, events)

        sorted_first = sorted_tracker.step(MultiScaleStep(
            0.0,
            True,
            (evidence(0.8, [1.0, 0.0]), evidence(2.8, [1.0, 0.0])),
        ))
        reversed_first = reversed_tracker.step(MultiScaleStep(
            0.0,
            True,
            (evidence(2.8, [1.0, 0.0]), evidence(0.8, [1.0, 0.0])),
        ))
        sorted_changed = sorted_tracker.step(MultiScaleStep(
            0.8,
            True,
            (evidence(0.8, [0.0, 1.0]), evidence(2.8, [0.9, 0.1])),
        ))
        reversed_changed = reversed_tracker.step(MultiScaleStep(
            0.8,
            True,
            (evidence(2.8, [0.9, 0.1]), evidence(0.8, [0.0, 1.0])),
        ))

        self.assertEqual(
            (sorted_first.visible_speaker, sorted_first.reason),
            (reversed_first.visible_speaker, reversed_first.reason),
        )
        self.assertEqual(
            (sorted_changed.visible_speaker, sorted_changed.reason),
            (reversed_changed.visible_speaker, reversed_changed.reason),
        )
        self.assertEqual(reversed_changed.diagnostics["scale_windows"], [0.8, 2.8])
        np.testing.assert_allclose(
            reversed_changed.diagnostics["scale_weights"], [0.7, 0.3]
        )
        self.assertEqual(
            reversed_changed.trace_record()["algorithm_id"],
            MULTISCALE_ALGORITHM_ID,
        )

    def test_partial_valid_scales_use_available_weight_and_keep_timeline(self) -> None:
        media_times = np.asarray([0.0, 0.8, 1.6], dtype=np.float64)
        short = CachedLiveWindowBlock(
            provider="test",
            video_id="video",
            window_seconds=0.8,
            media_times=media_times,
            embeddings=np.asarray(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
            valid=np.asarray([True, True, True]),
            raw_rms=np.ones(3, dtype=np.float32),
            sample_rate=16000,
        )
        long = CachedLiveWindowBlock(
            provider="test",
            video_id="video",
            window_seconds=2.8,
            media_times=media_times.copy(),
            embeddings=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32
            ),
            valid=np.asarray([False, True, True]),
            raw_rms=np.ones(3, dtype=np.float32),
            sample_rate=16000,
        )
        config = MultiScaleTrackerConfig(
            scale_weights=(0.7, 0.3),
            min_similarity=0.2,
            min_margin=0.0,
            enable_consensus=False,
            enable_crossover=False,
            enable_history=False,
        )

        decisions = replay_cached_multiscale_windows(
            [long, short],
            [
                event(0.0, "S1", [1.0, 0.0], generation=1),
                event(0.0, "S2", [0.0, 1.0], generation=1),
            ],
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=bool),
            config=config,
        )

        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0].visible_speaker, "S1")
        self.assertEqual(decisions[0].diagnostics["scale_windows"], [0.8])
        self.assertEqual(decisions[0].diagnostics["scale_weights"], [1.0])

    def test_profile_event_is_not_visible_before_its_media_time(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=0.2,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=False,
            ),
            [event(2.0, "S1", [1.0, 0.0], generation=1)],
        )

        before = tracker.step(
            MultiScaleStep(1.0, True, (evidence(0.8, [1.0, 0.0]),))
        )
        at_event = tracker.step(
            MultiScaleStep(2.0, True, (evidence(0.8, [1.0, 0.0]),))
        )

        self.assertIsNone(before.visible_speaker)
        self.assertEqual(before.profile_count, 0)
        self.assertEqual(at_event.visible_speaker, "S1")
        self.assertEqual(at_event.profile_count, 1)

    def test_complete_profile_snapshot_replaces_previous_generation(self) -> None:
        """SpeakerProfileEvent is a snapshot, not an incremental observation."""

        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=-1.0,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=False,
                official_merge_weight=0.8,
            ),
            [
                event(0.0, "S1", [1.0, 0.0], generation=1),
                event(1.0, "S1", [0.0, 1.0], generation=2),
            ],
        )

        tracker.step(MultiScaleStep(0.0, True, (evidence(0.8, [1.0, 0.0]),)))
        updated = tracker.step(
            MultiScaleStep(1.0, True, (evidence(0.8, [0.0, 1.0]),))
        )

        self.assertAlmostEqual(updated.similarities["S1"], 1.0, places=6)

    def test_low_confidence_evidence_releases_visible_speaker(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=0.5,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=False,
                unknown_release_count=2,
            ),
            [
                event(0.0, "S1", [1.0, 0.0, 0.0], generation=1),
                event(0.0, "S2", [0.0, 1.0, 0.0], generation=1),
            ],
        )

        acquired = tracker.step(
            MultiScaleStep(0.0, True, (evidence(0.8, [1.0, 0.0, 0.0]),))
        )
        first_unknown = tracker.step(
            MultiScaleStep(0.8, True, (evidence(0.8, [0.0, 0.0, 1.0]),))
        )
        released = tracker.step(
            MultiScaleStep(1.6, True, (evidence(0.8, [0.0, 0.0, 1.0]),))
        )

        self.assertEqual(acquired.visible_speaker, "S1")
        self.assertIn(first_unknown.reason, {"unknown_debounce", "multiscale_unknown"})
        self.assertIsNone(released.visible_speaker)
        self.assertEqual(released.action, "clear")
        self.assertEqual(released.reason, "unknown")

    def test_no_speech_breaks_challenger_history(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=0.2,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=True,
                history_size=3,
                history_required=2,
                history_advantage=0.2,
            ),
            [
                event(0.0, "S1", [1.0, 0.0], generation=1),
                event(0.0, "S2", [0.0, 1.0], generation=1),
            ],
        )

        tracker.step(MultiScaleStep(0.0, True, (evidence(0.8, [1.0, 0.0]),)))
        tracker.step(MultiScaleStep(0.8, True, (evidence(0.8, [0.0, 1.0]),)))
        tracker.step(MultiScaleStep(1.6, False, ()))
        after_gap = tracker.step(
            MultiScaleStep(2.4, True, (evidence(0.8, [0.0, 1.0]),))
        )

        self.assertEqual(after_gap.visible_speaker, "S1")
        self.assertEqual(after_gap.reason, "multiscale_hold")
        self.assertEqual(after_gap.diagnostics["history_count"], 1)

    def test_non_probe_step_rejects_evidence(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker()

        with self.assertRaisesRegex(ValueError, "non-probe"):
            tracker.step(
                MultiScaleStep(
                    0.0,
                    True,
                    (evidence(0.8, [1.0, 0.0]),),
                    probe_scheduled=False,
                )
            )

    def test_no_speech_breaks_provisional_profile_confirmation(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=0.2,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=False,
                enable_online_profiles=True,
                provisional_first_immediate=False,
                provisional_confirm_count=2,
                provisional_confirm_similarity=0.8,
                provisional_max_existing_similarity=0.2,
            ),
            [event(0.0, "S1", [1.0, 0.0], generation=1)],
        )

        tracker.step(MultiScaleStep(0.0, True, (evidence(0.8, [0.0, 1.0]),)))
        tracker.step(MultiScaleStep(0.8, False, ()))
        after_gap = tracker.step(
            MultiScaleStep(1.6, True, (evidence(0.8, [0.0, 1.0]),))
        )

        self.assertIsNone(after_gap.visible_speaker)
        self.assertEqual(after_gap.profile_count, 1)
        self.assertEqual(after_gap.diagnostics["provisional_profiles"], [])

    def test_provisional_absorption_applies_official_merge_weight_once(self) -> None:
        tracker = CausalMultiScaleSpeakerTracker(
            MultiScaleTrackerConfig(
                scale_weights=(1.0,),
                min_similarity=-1.0,
                min_margin=0.0,
                enable_consensus=False,
                enable_crossover=False,
                enable_history=False,
                enable_online_profiles=True,
                provisional_confirm_count=1,
                provisional_first_immediate=True,
                official_merge_similarity=0.5,
                official_merge_weight=0.5,
            ),
            [event(1.0, "S1", [0.6, 0.8], generation=1)],
        )

        created = tracker.step(
            MultiScaleStep(0.0, True, (evidence(0.8, [1.0, 0.0]),))
        )
        absorbed = tracker.step(
            MultiScaleStep(1.0, True, (evidence(0.8, [1.0, 0.0]),))
        )

        expected = np.asarray([0.8, 0.4], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        self.assertEqual(created.visible_speaker, "P1")
        self.assertAlmostEqual(absorbed.similarities["P1"], float(expected[0]), places=6)


if __name__ == "__main__":
    unittest.main()
