from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))






class RealtimeTraceAnalysisTests(unittest.TestCase):
    def base_sentence_payload(self) -> dict[str, object]:
        return {
            "index": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "same speaker evidence",
            "speech_audio_ratio": 1.0,
        }

    def unknown_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "assigned_speaker": None,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "assignment_source": "embedding",
        }

    def confirmed_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "retro_reassigned": True,
            "revision_from": "S3",
            "revision_to": "S6",
            "assigned_speaker": "S6",
            "probabilities": {"unknown": 0.0, "speaker6": 1.0},
            "similarities": {"S6": 0.82},
            "unknown_probability": 0.0,
            "assignment_source": "retro",
        }

    def canonical(self) -> list[dict[str, object]]:
        return [
            {
                "speaker": "canonical_speaker",
                "start": 0.0,
                "end": 3.0,
                "text": "same speaker evidence",
            }
        ]

    def test_score_reducers_follow_committed_live_state_not_tentative_ui_state(self) -> None:
        from realtime.realtime_speakerdiarize import analyze_trace_against_canonical
        from window.youtube_window_diarize_gui import build_window_validation_records

        tentative_payload = {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "provisional_assignment": True,
            "revision_from": "UNKNOWN",
            "revision_to": "S3",
            "assigned_speaker": "S3",
            "probabilities": {"unknown": 0.45, "speaker3": 0.55},
            "assignment_source": "prototype_unknown_tentative",
        }
        live_records = [
            {"time": 1.0, "event": "sentence", "payload": self.unknown_sentence_payload()},
            {"time": 2.0, "event": "sentence", "payload": tentative_payload},
        ]

        analysis_records, final_payloads = build_window_validation_records(live_records)
        summary = analyze_trace_against_canonical(analysis_records, self.canonical(), match_mode="timestamp")

        self.assertIsNone(final_payloads[0]["assigned_speaker"])
        self.assertIsNone(summary["rows"][0]["assigned_speaker"])
        self.assertEqual(summary["unknown_segments"], 1)
        self.assertEqual(summary["assigned_counts"], {"UNKNOWN": 1})

        confirmed_records = [
            *live_records,
            {"time": 3.0, "event": "sentence", "payload": self.confirmed_sentence_payload()},
        ]
        analysis_records, final_payloads = build_window_validation_records(confirmed_records)
        summary = analyze_trace_against_canonical(analysis_records, self.canonical(), match_mode="timestamp")

        self.assertEqual(final_payloads[0]["assigned_speaker"], "S6")
        self.assertEqual(summary["rows"][0]["assigned_speaker"], "S6")
        self.assertEqual(summary["assigned_counts"], {"S6": 1})
        self.assertEqual(summary["duration_accuracy"], 1.0)

    def test_raw_trace_analysis_ignores_tentative_sentence_events(self) -> None:
        from realtime.realtime_speakerdiarize import analyze_trace_against_canonical

        final_payload = {
            **self.unknown_sentence_payload(),
            "video_start_seconds": 0.0,
            "video_end_seconds": 3.0,
            "duration_seconds": 3.0,
        }
        tentative_payload = {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "provisional_assignment": True,
            "assigned_speaker": "S3",
        }
        raw_records = [
            {"time": 1.0, "event": "final", "payload": final_payload},
            {"time": 1.1, "event": "sentence", "payload": self.unknown_sentence_payload()},
            {"time": 1.2, "event": "sentence", "payload": tentative_payload},
        ]

        summary = analyze_trace_against_canonical(raw_records, self.canonical(), match_mode="timestamp")

        self.assertIsNone(summary["rows"][0]["assigned_speaker"])
        self.assertEqual(summary["unknown_segments"], 1)

        summary = analyze_trace_against_canonical(
            [*raw_records, {"time": 1.3, "event": "sentence", "payload": self.confirmed_sentence_payload()}],
            self.canonical(),
            match_mode="timestamp",
        )

        self.assertEqual(summary["rows"][0]["assigned_speaker"], "S6")
        self.assertEqual(summary["duration_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
