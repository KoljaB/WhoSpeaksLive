from __future__ import annotations

import io
import os
import sys
import tarfile
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


class PreviewBackendTests(unittest.TestCase):
    def test_cpu_mode_reuses_final_asr_source_for_realtime_preview(self) -> None:
        from window.cpu_forced_alignment import CpuHybridTranscriber
        from window.window_diarizer_runtime_audio import WindowRuntimeAudioMixin

        source = mock.Mock()
        runtime = WindowRuntimeAudioMixin()
        runtime.args = SimpleNamespace(
            realtime_preview_engine="kroko_onnx",
            realtime_preview_provider="cpu",
            realtime_preview_model_preset="community-64l",
            realtime_preview_language="de",
            realtime_preview_num_threads=2,
            realtime_preview_model_dir=Path("C:/models/kroko"),
            realtime_preview_model_path=None,
        )
        runtime._model = CpuHybridTranscriber(source, mock.Mock())
        runtime.bus = mock.Mock()
        runtime._ensure_realtime_preview_model = mock.Mock()

        with mock.patch(
            "window.window_diarizer_runtime_audio.create_realtime_preview_transcriber"
        ) as create_transcriber:
            runtime._load_realtime_preview()

        create_transcriber.assert_not_called()
        source.reset_preview.assert_called_once_with()
        self.assertIs(runtime._preview_transcriber, source)
        self.assertFalse(runtime._preview_transcriber_owned)

    def test_cpu_hybrid_uses_forced_alignment_when_health_checks_pass(self) -> None:
        from window.cpu_forced_alignment import AlignmentHealth, CpuHybridTranscriber
        from window.window_preview import FinalRealtimeTranscript, FinalRealtimeWord

        native = FinalRealtimeTranscript("Hello world.", (FinalRealtimeWord("Hello", 0.1, 0.4),))
        aligned = FinalRealtimeTranscript(
            "Hello world.",
            (FinalRealtimeWord("Hello", 0.08, 0.31, 0.9), FinalRealtimeWord(" world.", 0.34, 0.72, 0.8)),
        )
        source = mock.Mock()
        source.transcribe_final.return_value = native
        aligner = mock.Mock()
        aligner.align.return_value = aligned, AlignmentHealth(True, "accepted", 0.85, 0.0)

        hybrid = CpuHybridTranscriber(source, aligner)
        result = hybrid.transcribe_final(np.zeros(16000, dtype=np.float32), 16000)

        self.assertIs(result, aligned)
        self.assertTrue(hybrid.last_health.used_alignment)

    def test_cpu_hybrid_falls_back_to_native_timestamps_after_rejection(self) -> None:
        from window.cpu_forced_alignment import AlignmentHealth, CpuHybridTranscriber
        from window.window_preview import FinalRealtimeTranscript, FinalRealtimeWord

        native = FinalRealtimeTranscript("Wrong words.", (FinalRealtimeWord("Wrong words.", 0.2, 0.8),))
        source = mock.Mock()
        source.transcribe_final.return_value = native
        aligner = mock.Mock()
        aligner.align.return_value = None, AlignmentHealth(False, "alignment confidence below safety threshold")

        hybrid = CpuHybridTranscriber(source, aligner)
        result = hybrid.transcribe_final(np.zeros(16000, dtype=np.float32), 16000)

        self.assertIs(result, native)
        self.assertFalse(hybrid.last_health.used_alignment)

    def test_final_cpu_transcript_preserves_punctuation_and_refines_word_ends(self) -> None:
        from window.window_preview import final_realtime_transcript_from_response

        sample_rate = 16000
        audio = np.zeros(sample_rate, dtype=np.float32)
        audio[int(0.10 * sample_rate):int(0.36 * sample_rate)] = 0.2
        audio[int(0.60 * sample_rate):int(0.86 * sample_rate)] = 0.2
        result = final_realtime_transcript_from_response(
            {
                "text": "Hello world.",
                "words": [
                    {"text": "Hello", "start": 0.10},
                    {"text": "world", "start": 0.60},
                ],
            },
            audio,
            sample_rate,
        )

        self.assertEqual("".join(word.text for word in result.words), "Hello world.")
        self.assertAlmostEqual(result.words[0].start, 0.10)
        self.assertLess(result.words[0].end, 0.45)
        self.assertGreater(result.words[1].end, 0.82)
        self.assertLess(result.words[1].end, 0.90)

    def test_final_cpu_word_end_stops_at_first_sustained_pause(self) -> None:
        from window.window_preview import final_realtime_transcript_from_response

        sample_rate = 16000
        audio = np.zeros(3 * sample_rate, dtype=np.float32)
        audio[int(0.10 * sample_rate):int(0.42 * sample_rate)] = 0.2
        # Later non-speech activity must not stretch the first word across the pause.
        audio[int(1.80 * sample_rate):int(2.10 * sample_rate)] = 0.08
        audio[int(2.50 * sample_rate):int(2.82 * sample_rate)] = 0.2
        result = final_realtime_transcript_from_response(
            {
                "text": "First second.",
                "words": [
                    {"text": "First", "start": 0.10},
                    {"text": "second", "start": 2.50},
                ],
            },
            audio,
            sample_rate,
        )

        self.assertGreater(result.words[0].end, 0.38)
        self.assertLess(result.words[0].end, 0.55)
        self.assertGreater(result.words[1].end, 2.78)

    def test_native_kroko_words_are_normalized_from_started_at(self) -> None:
        from workers.structured_realtime_result import structured_result_payload

        class Result:
            text = "Alles hat ein Ende,"
            tokens = ["▁Alles", "▁hat", "▁ein", "▁Ende"]
            timestamps = [0.28, 0.72, 0.96, 1.28]

            @staticmethod
            def as_json_string() -> str:
                return '{"elements":{"words":[{"value":"Alles","startedAt":0.28},{"value":"hat","startedAt":0.72},{"value":"ein","startedAt":0.96},{"value":"Ende","startedAt":1.28}]}}'

        payload = structured_result_payload(Result())
        self.assertEqual(payload["words"][3], {"text": "Ende", "start": 1.28})

    def test_nemotron_subword_tokens_produce_one_anchor_per_word(self) -> None:
        from window.window_preview import _token_word_starts

        tokens = [" ", " ", "ever", "yo", "ne", " wel", "co", "me", ".", " ", " How", " ", "are"]
        timestamps = [0.64, 0.64, 0.64, 0.72, 0.72, 1.04, 1.20, 1.20, 1.36, 1.36, 1.52, 1.76, 1.76]

        self.assertEqual(_token_word_starts(tokens, timestamps), [0.64, 1.04, 1.52, 1.76])

    def test_final_cpu_asr_timeout_scales_with_audio_duration(self) -> None:
        from window.window_preview import JsonLineSubprocessPreviewTranscriber

        transcriber = object.__new__(JsonLineSubprocessPreviewTranscriber)
        transcriber._send_request = mock.Mock(
            return_value={"text": "hello", "words": [{"text": "hello", "start": 0.1}]}
        )

        transcriber.transcribe_final(np.zeros(30 * 16000, dtype=np.float32), 16000)

        self.assertEqual(transcriber._send_request.call_args.kwargs["timeout_seconds"], 65.0)

    def test_nemotron_engine_aliases_and_language_tiers(self) -> None:
        from window.realtime_preview_backends import (
            normalize_preview_engine,
            preview_language_error,
            preview_language_support,
        )

        self.assertEqual(normalize_preview_engine("nemotron"), "sherpa_onnx")
        self.assertEqual(normalize_preview_engine("sherpa-onnx"), "sherpa_onnx")
        self.assertEqual(preview_language_support("sherpa_onnx", "de").locale, "de-DE")
        self.assertEqual(preview_language_support("sherpa_onnx", "sv").tier, "broad-coverage")
        self.assertIn("not supported", preview_language_error("sherpa_onnx", "he") or "")

    def test_nemotron_model_directory_requires_all_runtime_files(self) -> None:
        from window.sherpa_onnx_models import REQUIRED_MODEL_FILES, validate_sherpa_onnx_model_dir

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "encoder.int8.onnx"):
                validate_sherpa_onnx_model_dir(model_dir)
            for name in REQUIRED_MODEL_FILES:
                (model_dir / name).write_bytes(b"model")
            self.assertEqual(validate_sherpa_onnx_model_dir(model_dir), model_dir.resolve())

    def test_model_archive_rejects_path_traversal(self) -> None:
        from window.sherpa_onnx_models import _safe_archive_members

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "unsafe.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as archive:
                entry = tarfile.TarInfo("../outside.txt")
                entry.size = 0
                archive.addfile(entry, io.BytesIO())
            with tarfile.open(archive_path, "r:bz2") as archive:
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    _safe_archive_members(archive)

    def test_parser_selects_nemotron_defaults_without_changing_kroko_defaults(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["window", "--realtime-preview-engine", "sherpa_onnx", "--language", "de"]):
            args = parse_args()
        self.assertEqual(args.realtime_preview_model_preset, "nemotron-3.5-560ms-int8")
        self.assertEqual(args.realtime_preview_feed_chunk_seconds, 0.16)
        self.assertEqual(args.realtime_preview_interval_seconds, 0.10)
        self.assertEqual(args.realtime_preview_min_audio_seconds, 0.56)
        self.assertIsNotNone(args.realtime_preview_model_dir)
        self.assertIsNone(args.realtime_preview_model_path)

        with mock.patch.object(sys, "argv", ["window", "--realtime-preview-model-preset", "pro-16l"]):
            kroko_args = parse_args()
        self.assertEqual(kroko_args.realtime_preview_model_preset, "pro-16l")
        self.assertEqual(kroko_args.realtime_preview_feed_chunk_seconds, 0.32)

    def test_parser_rejects_model_location_for_wrong_backend(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            ["window", "--realtime-preview-engine", "sherpa_onnx", "--realtime-preview-model-path", "model.data"],
        ):
            with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parse_args()


class SherpaSubprocessClientTests(unittest.TestCase):
    def test_sherpa_client_uses_generic_protocol_and_worker_module(self) -> None:
        from window.window_preview import SherpaOnnxSubprocessPreviewTranscriber

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO('{"ready":true}\n')
                self.stderr = io.StringIO("")
                self.returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                self.returncode = 0
                return 0

            def kill(self) -> None:
                self.returncode = -9

        args = type(
            "Args",
            (),
            {
                "realtime_preview_request_timeout_seconds": 0.2,
                "realtime_preview_startup_timeout_seconds": 0.5,
                "realtime_preview_python": Path(sys.executable),
                "realtime_preview_model_dir": Path("C:/models/nemotron"),
                "realtime_preview_language": "de",
                "language": "de",
                "realtime_preview_num_threads": 2,
            },
        )()
        with mock.patch("window.window_preview.subprocess.Popen", return_value=FakeProcess()) as popen:
            transcriber = SherpaOnnxSubprocessPreviewTranscriber(args)
            transcriber.close()

        command = popen.call_args.args[0]
        self.assertIn("workers.sherpa_onnx_realtime_preview_worker", command)
        self.assertIn("--model-dir", command)
        self.assertIn(str(Path("C:/models/nemotron")), command)
        self.assertIn(str(SRC), str(popen.call_args.kwargs["env"]["PYTHONPATH"]).split(os.pathsep))


class EmbeddingSubprocessClientTests(unittest.TestCase):
    def test_cpu_thread_limits_are_set_before_helper_imports_torch(self) -> None:
        from embeddings.embedding_providers import EmbeddingSubprocessClient

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO()
                self.stderr = io.StringIO()

            def poll(self) -> int | None:
                return 0

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMP_NUM_THREADS", None)
            os.environ.pop("MKL_NUM_THREADS", None)
            with mock.patch("embeddings.embedding_providers.subprocess.Popen", return_value=FakeProcess()) as popen:
                EmbeddingSubprocessClient(Path(sys.executable), "speechbrain_ecapa", "cpu").start()

        helper_env = popen.call_args.kwargs["env"]
        self.assertEqual(helper_env["OMP_NUM_THREADS"], "1")
        self.assertEqual(helper_env["MKL_NUM_THREADS"], "1")

    def test_in_memory_embedding_uses_disposable_system_temp_wav(self) -> None:
        from embeddings.embedding_providers import EmbeddingSubprocessClient

        client = EmbeddingSubprocessClient(Path(sys.executable), "speechbrain_ecapa", "cpu")
        expected = np.asarray([1.0, 0.0], dtype=np.float32)
        seen_path: Path | None = None

        def fake_embed_wav(path: Path) -> np.ndarray:
            nonlocal seen_path
            seen_path = Path(path)
            self.assertTrue(seen_path.exists())
            return expected

        with mock.patch.object(client, "embed_wav", side_effect=fake_embed_wav):
            actual = client.embed_audio(np.zeros(1600, dtype=np.float32), 16000)

        np.testing.assert_array_equal(actual, expected)
        self.assertIsNotNone(seen_path)
        self.assertFalse(seen_path.exists())
