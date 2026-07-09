from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_gui_html import HTML
from window.window_media import resolve_browser_stream_id


class BrowserAudioModeTests(unittest.TestCase):
    def test_mixed_audio_scheme_is_stable_browser_stream_id(self) -> None:
        self.assertEqual(resolve_browser_stream_id("mixed-audio://local"), "mixed-audio")

    def test_combined_computer_audio_and_microphone_mode_is_exposed(self) -> None:
        self.assertIn('<option value="both">Computer audio + microphone</option>', HTML)
        self.assertIn('data-input-mode="both"', HTML)
        self.assertIn('mixed-audio://local', HTML)
        self.assertIn('captureSourceKind === "mixed"', HTML)

    def test_audio_file_mode_is_exposed(self) -> None:
        self.assertIn('<option value="file">Audio file</option>', HTML)
        self.assertIn('data-input-mode="file"', HTML)
        self.assertIn('id="audioFileInput"', HTML)
        self.assertIn('id="fileDropZone"', HTML)
        self.assertIn('/api/load-audio-file', HTML)


if __name__ == "__main__":
    unittest.main()
