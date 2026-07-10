from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PreviewBackendTests(unittest.TestCase):
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
