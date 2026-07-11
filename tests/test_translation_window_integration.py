from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest import mock
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TranslationWindowIntegrationTests(unittest.TestCase):
    def test_window_parser_accepts_multiple_targets_and_filters_source_language(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        argv = [
            "whospeaks-window",
            "--language", "es",
            "--translation-provider", "mock",
            "--translation-target-language", "en",
            "--translation-target-language", "de",
            "--translation-target-language", "es",
            "--translation-max-targets", "3",
        ]
        with mock.patch.object(sys, "argv", argv):
            parsed = parse_args()
        self.assertEqual(parsed.translation_provider, "mock")
        self.assertEqual(parsed.translation_target_language, ["en", "de"])

    def test_committed_sentence_payload_has_revision_safe_source_hash(self) -> None:
        from window.window_diarizer import WindowDiarizer

        sentence = SimpleNamespace(
            text="Buenos días.",
            start=1.0,
            end=2.0,
            spoken_word_seconds=0.8,
            speech_audio_ratio=0.8,
            next_left=2.0,
            words=[],
            first_word_start=1.0,
            last_word_end=2.0,
            next_word_start=None,
            gap_to_next_word_seconds=None,
            boundary_strategy="punctuation",
            sentence_boundary_pre_padding_seconds=0.1,
            sentence_boundary_post_padding_seconds=0.1,
            sentence_boundary_gap_ratio=0.5,
        )
        payload = WindowDiarizer._base_payload_from_sentence_part(7, sentence, 0.0, 3.0)
        expected = hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()
        self.assertEqual(payload["source_text_hash"], expected)
        self.assertEqual(payload["source_revision"], expected)

    def test_server_wires_translation_config_events_routes_and_persistence(self) -> None:
        server_source = (SRC / "window" / "youtube_window_diarize_gui.py").read_text(encoding="utf-8")
        self.assertIn('replace("__TRANSLATION_JSON__"', server_source)
        self.assertIn('path == "/api/translation/configure"', server_source)
        self.assertIn('path == "/api/translation/status"', server_source)
        self.assertIn('snapshot["translations"] = self.translation.snapshot()', server_source)
        self.assertIn('self.translation.handle_sentence(payload', server_source)


if __name__ == "__main__":
    unittest.main()
