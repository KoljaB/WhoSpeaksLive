from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from embeddings.live_shifting_window_corpus import (
    ControlledStop,
    CorpusIdentityError,
    JobConfig,
    build_live_window_job,
)


class _FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return np.asarray(
            [
                float(audio.size) / sample_rate,
                float(np.mean(audio)) if audio.size else 0.0,
                float(np.std(audio)) if audio.size else 0.0,
                1.0,
            ],
            dtype=np.float32,
        )


class LiveShiftingWindowCorpusTests(unittest.TestCase):
    def _fixture(self, root: Path, seconds: float = 5.0) -> tuple[Path, np.ndarray]:
        audio_path = root / "video.audio.mp3"
        audio_path.write_bytes(b"stable-source-audio-identity")
        sample_count = int(seconds * 16_000)
        timeline = np.arange(sample_count, dtype=np.float32) / 16_000.0
        audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)).astype(np.float32)
        audio[sample_count // 2 :] *= 0.4
        return audio_path, audio

    def _loader(self, audio: np.ndarray):
        def load(_path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
            return audio.copy(), sample_rate

        return load

    def test_complete_job_writes_scalable_arrays_and_progress(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, audio = self._fixture(root)
            output = root / "corpus"
            provider = _FakeProvider()
            observed: list[float] = []
            config = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7", "1.0"),
                block_rows=4,
            )

            result = build_live_window_job(
                config,
                provider_factory=lambda _provider, _device: provider,
                audio_loader=self._loader(audio),
                progress_observer=lambda payload: observed.append(float(payload["percent"])),
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_embeddings"], 43)
            self.assertEqual(result["successful_embeddings"], 43)
            self.assertEqual(result["failed_embeddings"], 0)
            self.assertEqual(provider.calls, 43)
            self.assertEqual(observed[-1], 100.0)
            self.assertEqual(observed, sorted(observed))

            job_dir = output / "providers" / "fake_provider" / "videos" / "video"
            progress = __import__("json").loads((job_dir / "progress.json").read_text())
            root_progress = __import__("json").loads((output / "progress.json").read_text())
            self.assertEqual(progress["percent"], 100.0)
            self.assertEqual(root_progress["percent"], 100.0)
            self.assertEqual(root_progress["job_count"], 1)

            for length in ("0700ms", "1000ms"):
                length_dir = job_dir / "lengths" / length
                self.assertTrue((length_dir / "metadata.json").is_file())
                self.assertFalse((length_dir / "embeddings.f32.npy.partial").exists())
                embeddings = np.load(length_dir / "embeddings.f32.npy")
                attempted = np.load(length_dir / "attempted.u1.npy")
                valid = np.load(length_dir / "valid.u1.npy")
                self.assertEqual(embeddings.shape, (25, 4))
                self.assertEqual(int(attempted.sum()), int(valid.sum()))
                self.assertTrue(np.allclose(np.linalg.norm(embeddings[valid.astype(bool)], axis=1), 1.0))
                self.assertTrue(np.all(np.isnan(embeddings[~valid.astype(bool)])))

    def test_interrupted_job_resumes_without_reembedding_completed_ticks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, audio = self._fixture(root)
            output = root / "corpus"
            provider = _FakeProvider()
            paused = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7", "1.0"),
                block_rows=4,
                stop_after_embeddings=7,
            )

            with self.assertRaises(ControlledStop):
                build_live_window_job(
                    paused,
                    provider_factory=lambda _provider, _device: provider,
                    audio_loader=self._loader(audio),
                )
            self.assertEqual(provider.calls, 7)

            resumed = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7", "1.0"),
                block_rows=4,
            )
            result = build_live_window_job(
                resumed,
                provider_factory=lambda _provider, _device: provider,
                audio_loader=self._loader(audio),
            )

            self.assertEqual(result["completed_embeddings"], 43)
            self.assertEqual(result["percent"], 100.0)
            self.assertEqual(provider.calls, 43)

    def test_resume_recovers_arrays_flushed_ahead_of_json_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, audio = self._fixture(root)
            output = root / "corpus"
            provider = _FakeProvider()
            paused = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7", "1.0"),
                block_rows=4,
                stop_after_embeddings=7,
            )
            with self.assertRaises(ControlledStop):
                build_live_window_job(
                    paused,
                    provider_factory=lambda _provider, _device: provider,
                    audio_loader=self._loader(audio),
                )

            length_dir = output / "providers" / "fake_provider" / "videos" / "video" / "lengths" / "0700ms"
            arrays = {
                "embeddings": np.lib.format.open_memmap(length_dir / "embeddings.f32.npy.partial", mode="r+"),
                "attempted": np.lib.format.open_memmap(length_dir / "attempted.u1.npy.partial", mode="r+"),
                "valid": np.lib.format.open_memmap(length_dir / "valid.u1.npy.partial", mode="r+"),
                "raw_rms": np.lib.format.open_memmap(length_dir / "raw_rms.f32.npy.partial", mode="r+"),
                "raw_peak": np.lib.format.open_memmap(length_dir / "raw_peak.f32.npy.partial", mode="r+"),
                "trimmed_samples": np.lib.format.open_memmap(length_dir / "trimmed_samples.i32.npy.partial", mode="r+"),
                "prepared_samples": np.lib.format.open_memmap(length_dir / "prepared_samples.i32.npy.partial", mode="r+"),
                "latency_ms": np.lib.format.open_memmap(length_dir / "latency_ms.f32.npy.partial", mode="r+"),
            }
            for index in (10, 11):
                arrays["embeddings"][index] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                arrays["attempted"][index] = 1
                arrays["valid"][index] = 1
                arrays["raw_rms"][index] = 0.1
                arrays["raw_peak"][index] = 0.2
                arrays["trimmed_samples"][index] = 100
                arrays["prepared_samples"][index] = 8_000
                arrays["latency_ms"][index] = 1.0
            for array in arrays.values():
                array.flush()
            del array
            del arrays

            resumed = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7", "1.0"),
                block_rows=4,
            )
            result = build_live_window_job(
                resumed,
                provider_factory=lambda _provider, _device: provider,
                audio_loader=self._loader(audio),
            )
            self.assertEqual(result["completed_embeddings"], 43)
            self.assertEqual(provider.calls, 41)

    def test_identity_change_is_rejected_instead_of_reusing_stale_arrays(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, audio = self._fixture(root)
            output = root / "corpus"
            provider = _FakeProvider()
            original = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                window_seconds=("0.7",),
            )
            build_live_window_job(
                original,
                provider_factory=lambda _provider, _device: provider,
                audio_loader=self._loader(audio),
            )

            changed = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                hop_seconds="0.25",
                window_seconds=("0.7",),
            )
            with self.assertRaises(CorpusIdentityError):
                build_live_window_job(
                    changed,
                    provider_factory=lambda _provider, _device: provider,
                    audio_loader=self._loader(audio),
                )

    def test_absolute_source_start_preserves_original_tick_phase(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, audio = self._fixture(root, seconds=2.0)
            output = root / "corpus"
            config = JobConfig(
                audio_path=audio_path,
                video_id="video",
                provider="fake_provider",
                output_root=output,
                device="cpu",
                source_start_seconds=0.15,
                window_seconds=("0.7",),
            )

            result = build_live_window_job(
                config,
                provider_factory=lambda _provider, _device: _FakeProvider(),
                audio_loader=self._loader(audio),
            )

            self.assertEqual(result["expected_embeddings"], 6)
            edges = np.load(output / "videos" / "video" / "timeline" / "right_edges.i64.npy")
            self.assertTrue(np.all((edges + 2_400) % 3_200 == 0))


if __name__ == "__main__":
    unittest.main()
