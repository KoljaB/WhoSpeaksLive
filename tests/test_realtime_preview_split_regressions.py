from __future__ import annotations

import argparse
import sys
import threading
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.window_domain import VadWindowState
from window.window_events import RecordingEventBus
from tests.window_diarizer_support import make_window_diarizer


class _CumulativePreviewTranscriber:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_preview(self) -> None:
        self.reset_count += 1

    def accept_preview_audio(self, _audio: np.ndarray, _sample_rate: int) -> str:
        return "opening words survive"


class RealtimePreviewSplitRegressionTests(unittest.TestCase):
    def test_vad_gate_does_not_skip_unseen_audio_after_slow_multi_sentence_commit(self) -> None:
        """A lagging preview generation must inspect audio immediately after its reset boundary."""

        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            realtime_preview_reset_overlap_seconds=0.15,
            realtime_preview_interval_seconds=0.05,
            realtime_preview_min_audio_seconds=0.32,
            realtime_preview_min_advance_seconds=0.32,
            realtime_preview_feed_chunk_seconds=0.32,
            realtime_preview_diarize_min_advance_seconds=0.75,
            realtime_preview_diarize_min_audio_seconds=1.5,
            realtime_preview_vad_gate=True,
            realtime_preview_vad_gate_pre_padding_seconds=0.35,
            realtime_preview_vad_gate_close_silence_seconds=1.1,
            realtime_preview_vad_gate_post_padding_seconds=0.35,
        )
        diarizer.duration = 30.0
        diarizer.bus = RecordingEventBus()
        diarizer._stop = threading.Event()
        diarizer._preview_lock = threading.Lock()
        diarizer._preview_left = 0.0
        diarizer._preview_generation = 0
        diarizer._preview_paused = False
        diarizer._preview_transcriber = _CumulativePreviewTranscriber()
        diarizer.playback_time = lambda: 16.0  # type: ignore[method-assign]
        diarizer._realtime_unknown_speaker_payload = lambda: {}  # type: ignore[method-assign]
        diarizer._live_speaker_assignment_enabled = lambda: False  # type: ignore[method-assign]
        diarizer._audio_window_copy = lambda left, right: (  # type: ignore[method-assign]
            np.zeros(max(1, round((right - left) * 100)), dtype=np.float32),
            100,
        )

        vad_windows: list[tuple[float, float]] = []

        def fake_vad(
            left: float,
            right: float,
            *,
            force: bool = False,
            role: str = "main",
        ) -> VadWindowState:
            del force, role
            vad_windows.append((left, right))
            if len(vad_windows) == 1:
                # End the worker after this iteration while still allowing it to
                # process and emit the result of the current VAD decision.
                diarizer._stop.set()
            if left <= 11.0 < right:
                return VadWindowState(
                    has_speech=True,
                    should_flush=False,
                    speech_start=11.0,
                    speech_end=12.0,
                    speech_seconds=1.0,
                    speech_spans=[(11.0, 12.0)],
                    backend="fake",
                )
            return VadWindowState(False, False, backend="fake")

        diarizer._vad_gate_window_state = fake_vad  # type: ignore[method-assign]

        # A large-v2 pass has committed multiple sentences through 10.15 s.
        # Its configured pre-roll resets the preview generation to 10.00 s,
        # but playback has advanced to 16.00 s while that pass was running.
        diarizer._advance_realtime_preview_after_commit(10.15)
        diarizer._run_realtime_preview()

        clear_event = next(record for record in diarizer.bus.records if record["event"] == "realtime_clear")
        self.assertEqual(clear_event["payload"]["preview_reset_left"], 10.0)
        self.assertTrue(vad_windows)
        self.assertAlmostEqual(vad_windows[0][0], 10.0)
        self.assertGreater(vad_windows[0][1], 11.0)

        realtime_events = [record for record in diarizer.bus.records if record["event"] == "realtime"]
        self.assertEqual(len(realtime_events), 1)
        self.assertEqual(realtime_events[0]["payload"]["text"], "Opening words survive")
        self.assertLessEqual(realtime_events[0]["payload"]["start"], 11.0)


if __name__ == "__main__":
    unittest.main()
