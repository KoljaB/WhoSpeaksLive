from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import audio_utils


class AudioUtilsTests(unittest.TestCase):
    def test_load_audio_file_falls_back_when_soundfile_cannot_decode(self) -> None:
        expected = np.array([0.0, 0.25, -0.25], dtype=np.float32)

        with mock.patch("soundfile.read", side_effect=RuntimeError("unsupported format")):
            with mock.patch.object(
                audio_utils,
                "_load_audio_file_with_av",
                return_value=(expected, 16000),
            ) as av_loader:
                audio, sample_rate = audio_utils.load_audio_file(Path("sample.mp3"))

        av_loader.assert_called_once()
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(np.allclose(audio, expected))


if __name__ == "__main__":
    unittest.main()
