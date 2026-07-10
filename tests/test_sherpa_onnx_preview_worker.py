from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SherpaPreviewWorkerTests(unittest.TestCase):
    def test_reset_replaces_only_stream_and_keeps_recognizer_loaded(self) -> None:
        from window.sherpa_onnx_models import REQUIRED_MODEL_FILES
        from workers.sherpa_onnx_realtime_preview_worker import NemotronRecognizer

        class FakeStream:
            def __init__(self) -> None:
                self.options: list[tuple[str, str]] = []
                self.audio: list[tuple[int, np.ndarray]] = []

            def set_option(self, name: str, value: str) -> None:
                self.options.append((name, value))

            def accept_waveform(self, sample_rate: int, audio: np.ndarray) -> None:
                self.audio.append((sample_rate, audio))

        class FakeRecognizer:
            def __init__(self) -> None:
                self.streams: list[FakeStream] = []
                self.decode_count = 0

            def create_stream(self) -> FakeStream:
                stream = FakeStream()
                self.streams.append(stream)
                return stream

            def is_ready(self, _stream: FakeStream) -> bool:
                return self.decode_count == 0

            def decode_stream(self, _stream: FakeStream) -> None:
                self.decode_count += 1

            def get_result(self, _stream: FakeStream) -> object:
                return SimpleNamespace(text="Running text")

        fake_recognizer = FakeRecognizer()
        fake_module = SimpleNamespace(
            OnlineRecognizer=SimpleNamespace(from_transducer=mock.Mock(return_value=fake_recognizer))
        )
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            for name in REQUIRED_MODEL_FILES:
                (model_dir / name).write_bytes(b"model")
            with mock.patch.dict(sys.modules, {"sherpa_onnx": fake_module}):
                session = NemotronRecognizer.load(model_dir, "de", 2, "cpu")
                first_stream = session.stream
                self.assertEqual(first_stream.options, [("language", "de")])
                self.assertEqual(session.accept(np.zeros(320, dtype=np.float32), 16000), "Running text")
                session.reset()

        self.assertIsNot(session.stream, first_stream)
        self.assertEqual(len(fake_recognizer.streams), 2)
        self.assertEqual(fake_module.OnlineRecognizer.from_transducer.call_count, 1)

    def test_decode_request_rejects_non_float32_payload(self) -> None:
        from workers.sherpa_onnx_realtime_preview_worker import decode_request_audio

        with self.assertRaisesRegex(ValueError, "float32"):
            decode_request_audio({"audio_b64": "AA==", "sample_rate": 16000})

    def test_resample_audio_preserves_target_type(self) -> None:
        from workers.sherpa_onnx_realtime_preview_worker import TARGET_SAMPLE_RATE, resample_audio

        samples = resample_audio(np.array([0.0, 1.0], dtype=np.float32), 8000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.size, 4)
        self.assertEqual(TARGET_SAMPLE_RATE, 16000)
