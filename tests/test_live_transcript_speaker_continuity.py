from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_browser_parity import BrowserLiveSpeakerReducer


class LiveTranscriptSpeakerContinuityTests(unittest.TestCase):
    def test_pending_final_sentence_keeps_its_adopted_realtime_speaker(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.apply(
            "realtime",
            {
                "index": "active-1",
                "realtime_generation": 1,
                "start": 10.0,
                "end": 12.0,
                "text": "A sentence still being spoken.",
                "assigned_speaker": "SPEAKER_00",
            },
            now=12.0,
        )

        reducer.apply(
            "sentence",
            {
                "index": "17",
                "start": 10.0,
                "end": 12.0,
                "text": "A sentence still being spoken.",
                "pending": True,
                "assigned_speaker": "UNKNOWN",
            },
            now=12.05,
        )

        self.assertEqual(reducer.transcript_speaker, "SPEAKER_00")
        self.assertEqual(reducer.rows[0].speaker, "SPEAKER_00")

    def test_new_realtime_sentence_starts_with_last_committed_speaker(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.apply(
            "sentence",
            {
                "index": "0",
                "start": 0.0,
                "end": 2.0,
                "assigned_speaker": "SPEAKER_00",
            },
            now=2.0,
        )

        reducer.apply(
            "realtime",
            {
                "index": "1",
                "start": 2.0,
                "end": 2.6,
                "assigned_speaker": "UNKNOWN",
            },
            now=2.6,
        )

        self.assertEqual(reducer.transcript_speaker, "SPEAKER_00")

    def test_tail_evidence_does_not_replace_previous_speaker_before_majority(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            {"speaker": "SPEAKER_01", "start": 0.5, "end": 0.8},
        ]

        speaker = reducer._display_speaker("", 0.0, 0.8, "")

        self.assertEqual(speaker, "SPEAKER_00")

    def test_consistent_new_speaker_can_take_over_after_one_live_window(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 0.7},
        ]

        speaker = reducer._display_speaker("", 0.0, 0.7, "")

        self.assertEqual(speaker, "SPEAKER_01")

    def test_previous_speaker_head_start_does_not_stack_on_real_row_evidence(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.4},
            {"speaker": "SPEAKER_01", "start": 0.4, "end": 1.1},
        ]

        observed, scores = reducer._speaker_time_scores(
            0.0,
            1.1,
            "SPEAKER_00",
        )
        speaker = reducer._display_speaker(
            "",
            0.0,
            1.1,
            "SPEAKER_00",
        )

        self.assertAlmostEqual(observed["SPEAKER_00"], 0.4)
        self.assertAlmostEqual(scores["SPEAKER_00"], 0.4)
        self.assertEqual(speaker, "SPEAKER_01")

    def test_short_row_can_switch_when_challenger_owns_more_than_half_of_it(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            {"speaker": "SPEAKER_01", "start": 0.3639, "end": 1.0639},
        ]

        speaker = reducer._display_speaker(
            "",
            0.0,
            0.8,
            "",
        )

        self.assertEqual(speaker, "SPEAKER_01")

    def test_retired_final_alias_keeps_the_surviving_public_row_assignment(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.apply(
            "live_speaker_identity_alias",
            {
                "alias_generation": 1,
                "final_internal_speaker_id": "S1",
                "surviving_public_speaker_id": "LIVE_TRACKLET_1",
            },
            now=0.0,
        )
        reducer.apply(
            "live_speaker",
            {
                "assigned_speaker": "LIVE_TRACKLET_1",
                "start": 0.0,
                "end": 0.7,
                "hold_seconds": 2.5,
            },
            now=0.7,
        )
        reducer.apply(
            "realtime",
            {
                "index": "rt-1",
                "realtime_generation": 1,
                "start": 0.0,
                "end": 6.0,
                "text": "A long active sentence which has not been finalized yet",
                "assigned_speaker": "UNKNOWN",
            },
            now=0.71,
        )
        self.assertEqual(reducer.transcript_speaker, "LIVE_TRACKLET_1")

        reducer.apply(
            "live_speaker_identity_alias",
            {
                "alias_generation": 2,
                "final_internal_speaker_id": "S1",
                "surviving_public_speaker_id": "LIVE_TRACKLET_1",
                "retired": True,
            },
            now=0.8,
        )

        self.assertEqual(reducer.transcript_speaker, "LIVE_TRACKLET_1")
        self.assertEqual(reducer.rows[0].speaker, "LIVE_TRACKLET_1")
        self.assertEqual(reducer.timeline[0]["speaker"], "LIVE_TRACKLET_1")

    def test_previous_speaker_remains_until_a_challenger_has_a_clear_lead(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"

        speaker = reducer._display_speaker("", 0.0, 8.0, "SPEAKER_00")

        self.assertEqual(speaker, "SPEAKER_00")

    def test_short_sentence_majority_replaces_previous_speaker(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
            {"speaker": "SPEAKER_01", "start": 0.3, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 0.8, "end": 1.5},
            {"speaker": "SPEAKER_01", "start": 1.3, "end": 2.0},
            {"speaker": "SPEAKER_01", "start": 1.8, "end": 2.5},
        ]

        speaker = reducer._display_speaker("", 0.0, 2.5, "")

        self.assertEqual(speaker, "SPEAKER_01")

    def test_end_of_long_sentence_does_not_recolor_the_whole_sentence(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.last_transcript_speaker = "SPEAKER_00"
        reducer.timeline = [
            *[
                {"speaker": "SPEAKER_00", "start": end - 0.7, "end": end}
                for end in (
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    5.5,
                    6.0,
                    6.5,
                    7.0,
                    7.5,
                    8.0,
                )
            ],
            *[
                {"speaker": "SPEAKER_01", "start": end - 0.7, "end": end}
                for end in (8.5, 9.0, 9.5, 10.0)
            ],
        ]

        speaker = reducer._display_speaker("", 0.0, 10.0, "")

        self.assertEqual(speaker, "SPEAKER_00")

    def test_overlapping_analysis_windows_count_each_audio_position_once(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.timeline = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 0.5, "end": 1.5},
        ]

        observed, _ = reducer._speaker_time_scores(0.0, 1.5)

        self.assertAlmostEqual(observed["SPEAKER_00"], 0.5)
        self.assertAlmostEqual(observed["SPEAKER_01"], 1.0)
        self.assertAlmostEqual(sum(observed.values()), 1.5)

    def test_live_window_is_clipped_to_the_active_sentence_timestamps(self) -> None:
        reducer = BrowserLiveSpeakerReducer()
        reducer.timeline = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 2.5},
        ]

        observed, _ = reducer._speaker_time_scores(1.0, 2.0)

        self.assertNotIn("SPEAKER_00", observed)
        self.assertAlmostEqual(observed["SPEAKER_01"], 0.5)


if __name__ == "__main__":
    unittest.main()
