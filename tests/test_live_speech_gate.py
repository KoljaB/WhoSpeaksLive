from __future__ import annotations

import unittest

import numpy as np

from window.live_speech_gate import rms_speech_present


class LiveSpeechGateTests(unittest.TestCase):
    def test_requires_configured_amount_of_active_frames(self) -> None:
        sample_rate = 1000
        audio = np.zeros(300, dtype=np.float32)
        audio[:120] = 0.01
        self.assertFalse(rms_speech_present(
            audio, sample_rate, frame_seconds=0.03, threshold=0.003,
            min_speech_seconds=0.15,
        ))
        audio[:150] = 0.01
        self.assertTrue(rms_speech_present(
            audio, sample_rate, frame_seconds=0.03, threshold=0.003,
            min_speech_seconds=0.15,
        ))

    def test_ignores_a_tail_shorter_than_half_a_frame(self) -> None:
        audio = np.full(14, 0.01, dtype=np.float32)
        self.assertFalse(rms_speech_present(
            audio, 1000, frame_seconds=0.03, threshold=0.003,
            min_speech_seconds=0.001,
        ))


if __name__ == "__main__":
    unittest.main()
