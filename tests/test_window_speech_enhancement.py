from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from window.window_speech_enhancement import SpeechEnhancementClient
from window.window_validation import retranscribe_final_payloads_with_enhancement


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


class SpeechEnhancementClientTests(unittest.TestCase):
    def test_health_requires_ready_json(self) -> None:
        client = SpeechEnhancementClient("http://example.test")
        with patch("window.window_speech_enhancement.urlopen", return_value=_Response(b'{"ok": true}')):
            self.assertTrue(client.health()["ok"])

    def test_enhance_decodes_wav_and_preserves_duration(self) -> None:
        source = np.linspace(-0.25, 0.25, 8000, dtype=np.float32)
        output = io.BytesIO()
        sf.write(output, np.ones(4000, dtype=np.float32) * 0.1, 16000, format="WAV")
        response = _Response(
            output.getvalue(),
            {
                "X-UniSE-Queue-Seconds": "0.01",
                "X-UniSE-Processing-Seconds": "0.02",
            },
        )
        client = SpeechEnhancementClient("http://example.test")
        with patch("window.window_speech_enhancement.urlopen", return_value=response):
            enhanced, sample_rate = client.enhance(source, 8000)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(enhanced), 16000)
        self.assertEqual(client.stats()["request_count"], 1)
        self.assertAlmostEqual(float(client.stats()["queue_seconds"]), 0.01)

    def test_final_asr_retranscription_updates_text_and_hash(self) -> None:
        class _Controller:
            _model = object()

            @staticmethod
            def _audio_window_copy(_start: float, _end: float):
                return np.zeros(1600, dtype=np.float32), 16000

            @staticmethod
            def _transcribe_enhanced_final_audio_text(_model, _audio, _sample_rate):
                return "enhanced words"

        records, payloads, elapsed = retranscribe_final_payloads_with_enhancement(
            _Controller(),
            [{"start": 1.0, "end": 1.1, "text": "raw words"}],
        )

        self.assertEqual(payloads[0]["text"], "enhanced words")
        self.assertEqual(payloads[0]["pre_enhancement_asr_text"], "raw words")
        self.assertTrue(payloads[0]["final_asr_enhanced"])
        self.assertEqual(len(payloads[0]["source_text_hash"]), 64)
        self.assertEqual([record["event"] for record in records], ["final", "sentence"])
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
