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
