from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from speakers.person_library import PersonLibrary
from speakers.person_sample_ingestion import ingest_manual_voice_sample


def _wav_data_url(seconds: float, *, amplitude: float = 0.2, sample_rate: int = 16000) -> str:
    samples = (np.sin(np.arange(int(seconds * sample_rate)) * 2.0 * np.pi * 220.0 / sample_rate) * amplitude)
    payload = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return "data:audio/wav;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class _EmbeddingClient:
    def __init__(self, values=None) -> None:
        self.values = list(values or [[1.0, 0.0]])
        self.index = 0

    def embed_audio(self, _audio, _sample_rate):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return np.asarray(value, dtype=np.float32)


class PersonSampleIngestionTests(unittest.TestCase):
    def test_long_upload_creates_one_sample_and_rejects_an_outlier_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            person = library.create_person("Alice")
            sample = ingest_manual_voice_sample(
                library,
                _EmbeddingClient([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
                person_id=person["id"],
                embedding_provider="mock",
                filename="headset.wav",
                audio_b64=_wav_data_url(18.0),
                label="Headset",
            )
            self.assertEqual(sample["kind"], "manual_reference")
            self.assertEqual(len(sample["representations"]), 1)
            self.assertEqual(sample["evidence"]["outlier_count"], 1)
            self.assertTrue(sample["raw_audio"]["retained"])
            self.assertEqual(library.public_state()[0]["voice_sample_count"], 1)

    def test_silence_short_decode_nonfinite_and_exact_duplicate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            person = library.create_person("Alice")
            common = {"person_id": person["id"], "embedding_provider": "mock", "filename": "sample.wav"}
            with self.assertRaisesRegex(ValueError, "silent|quiet"):
                ingest_manual_voice_sample(library, _EmbeddingClient(), audio_b64=_wav_data_url(3.0, amplitude=0.0), **common)
            with self.assertRaisesRegex(ValueError, "at least"):
                ingest_manual_voice_sample(library, _EmbeddingClient(), audio_b64=_wav_data_url(0.5), **common)
            with self.assertRaisesRegex(ValueError, "base64"):
                ingest_manual_voice_sample(library, _EmbeddingClient(), audio_b64="%%%", **common)
            with self.assertRaisesRegex(ValueError, "invalid Voice representation"):
                ingest_manual_voice_sample(library, _EmbeddingClient([[float("nan"), 0.0]]), audio_b64=_wav_data_url(3.0), **common)
            payload = _wav_data_url(3.0)
            ingest_manual_voice_sample(library, _EmbeddingClient(), audio_b64=payload, **common)
            with self.assertRaisesRegex(ValueError, "exact Voice sample"):
                ingest_manual_voice_sample(library, _EmbeddingClient(), audio_b64=payload, **common)


if __name__ == "__main__":
    unittest.main()
