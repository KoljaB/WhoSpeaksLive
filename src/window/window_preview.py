"""Realtime preview transcribers for the window diarization GUI."""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.pythonpath import build_pythonpath
from paths import SRC_ROOT, VENDOR_DIR
from window.window_config import DEFAULT_KROKO_16L_CHUNK_SECONDS, KROKO_PREVIEW_FRAME_SECONDS, ROOT


KROKO_KEY_ENV_NAMES = ("REALTIMESTT_KROKO_ONNX_KEY", "KROKO_ONNX_KEY", "KROKO_KEY")
KROKO_REFERRAL_ENV_NAMES = (
    "REALTIMESTT_KROKO_ONNX_REFERRALCODE",
    "KROKO_ONNX_REFERRALCODE",
    "KROKO_REFERRALCODE",
)


def _first_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def add_kroko_license_options(options: dict[str, Any]) -> None:
    options.setdefault("key", _first_env_value(KROKO_KEY_ENV_NAMES))
    options.setdefault("referralcode", _first_env_value(KROKO_REFERRAL_ENV_NAMES))


def preview_worker_pythonpath(existing_pythonpath: str | None = None) -> str:
    return build_pythonpath(
        (SRC_ROOT, ROOT / "src", Path(__file__).resolve().parents[1], VENDOR_DIR),
        existing_pythonpath,
    )


class RealtimePreviewTranscriber:
    def reset_preview(self) -> None:
        return

    def accept_preview_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.transcribe_preview(audio, sample_rate)

    def transcribe_preview(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return


class MockRealtimePreviewTranscriber(RealtimePreviewTranscriber):
    WORDS = (
        "Which was more culturally significant the Renaissance or Single Ladies by Beyonce "
        "they both have their period and they both have their time Beyonce I am rather fond of "
        "but what the Renaissance was trying to do was to reform culture as a whole"
    ).split()

    def __init__(self) -> None:
        self._seconds = 0.0

    def reset_preview(self) -> None:
        self._seconds = 0.0

    def accept_preview_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        self._seconds += max(0.0, len(audio) / float(sample_rate or 16000))
        count = max(1, min(len(self.WORDS), int(self._seconds * 2.2)))
        return " ".join(self.WORDS[:count])

    def transcribe_preview(self, audio: np.ndarray, sample_rate: int) -> str:
        seconds = max(0.0, len(audio) / float(sample_rate or 16000))
        count = max(1, min(len(self.WORDS), int(seconds * 2.2)))
        return " ".join(self.WORDS[:count])


class KrokoRealtimePreviewTranscriber(RealtimePreviewTranscriber):
    """Legacy in-process Kroko path retained for existing direct integrations."""

    def __init__(self, args: argparse.Namespace) -> None:
        from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
        from RealtimeSTT.transcription_engines.factory import create_transcription_engine

        options = {
            "provider": args.realtime_preview_provider,
            "num_threads": args.realtime_preview_num_threads,
            "sample_rate": 16000,
            "language": getattr(args, "realtime_preview_language", getattr(args, "language", "en")),
            "suppress_native_output": True,
        }
        if args.realtime_preview_model_path is not None:
            options["model_path"] = str(args.realtime_preview_model_path)
        add_kroko_license_options(options)
        if args.realtime_preview_engine_options_json:
            extra_options = json.loads(args.realtime_preview_engine_options_json)
            if not isinstance(extra_options, dict):
                raise ValueError("--realtime-preview-engine-options-json must be a JSON object.")
            options.update(extra_options)
        model = str(args.realtime_preview_model_path or args.realtime_preview_model)
        download_root = args.realtime_preview_download_root or args.download_root
        config = TranscriptionEngineConfig(
            model=model,
            download_root=str(download_root) if download_root else None,
            device=args.realtime_preview_provider,
            normalize_audio=False,
            engine_options=options,
        )
        self.engine = create_transcription_engine(args.realtime_preview_engine, config)
        self.session = self.engine.create_streaming_session(
            language=getattr(args, "realtime_preview_language", getattr(args, "language", "en")),
            use_prompt=False,
        )

    def reset_preview(self) -> None:
        self.session.reset()

    def accept_preview_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size <= 0:
            return self.session.get_result().text.strip()
        self.session.accept_audio(audio.astype(np.float32, copy=False), sample_rate=sample_rate)
        self.session.decode()
        return self.session.get_result().text.strip()

    def transcribe_preview(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size <= 0:
            return ""
        self.session.reset()
        self.session.accept_audio(audio.astype(np.float32, copy=False), sample_rate=sample_rate)
        self.session.decode()
        return self.session.get_result().text.strip()

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


class JsonLineSubprocessPreviewTranscriber(RealtimePreviewTranscriber):
    """Common JSON-lines client for isolated native preview engines."""

    worker_module = ""
    worker_label = "Realtime preview"

    def __init__(self, args: argparse.Namespace) -> None:
        self._request_id = 0
        self._stderr_lines: list[str] = []
        self._stdout_messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stdout_closed = threading.Event()
        self._request_timeout_seconds = max(0.1, float(args.realtime_preview_request_timeout_seconds))
        self.process = subprocess.Popen(
            self.build_worker_command(args),
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self.build_worker_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread.start()
        try:
            ready = self._read_message(timeout_seconds=args.realtime_preview_startup_timeout_seconds)
        except Exception:
            self.close()
            raise
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(str(ready.get("error") or f"{self.worker_label} worker did not become ready."))

    def build_worker_command(self, args: argparse.Namespace) -> list[str]:
        raise NotImplementedError

    def build_worker_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = preview_worker_pythonpath(env.get("PYTHONPATH"))
        return env

    def _drain_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    self._stdout_messages.put(message)
        finally:
            self._stdout_closed.set()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            if line := line.strip():
                self._stderr_lines.append(line)
                del self._stderr_lines[:-20]

    def _read_message(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout is not None and timeout <= 0:
                details = "; ".join(self._stderr_lines[-3:])
                raise TimeoutError(f"{self.worker_label} worker did not respond within {timeout_seconds:.1f}s. {details}".strip())
            try:
                return self._stdout_messages.get(timeout=timeout)
            except queue.Empty:
                code = self.process.poll()
                if code is not None or self._stdout_closed.is_set():
                    details = "; ".join(self._stderr_lines[-3:])
                    raise RuntimeError(f"{self.worker_label} worker exited with {code}. {details}".strip())

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.worker_label} worker is not running: exit {self.process.returncode}")
        assert self.process.stdin is not None
        self._request_id += 1
        request_id = self._request_id
        self.process.stdin.write(json.dumps({**payload, "id": request_id}, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            response = self._read_message(timeout_seconds=self._request_timeout_seconds)
            if response.get("id") != request_id:
                continue
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            return response

    def reset_preview(self) -> None:
        self._send_request({"command": "reset"})

    def accept_preview_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        return self._send_audio("accept", audio, sample_rate)

    def transcribe_preview(self, audio: np.ndarray, sample_rate: int) -> str:
        return self._send_audio("transcribe", audio, sample_rate)

    def _send_audio(self, command: str, audio: np.ndarray, sample_rate: int) -> str:
        raw = np.ascontiguousarray(audio.astype(np.float32, copy=False)).tobytes()
        response = self._send_request(
            {
                "command": command,
                "sample_rate": int(sample_rate),
                "audio_b64": base64.b64encode(raw).decode("ascii"),
            }
        )
        return str(response.get("text") or "").strip()

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class KrokoSubprocessPreviewTranscriber(JsonLineSubprocessPreviewTranscriber):
    worker_module = "workers.kroko_realtime_preview_worker"
    worker_label = "Kroko preview"

    def build_worker_command(self, args: argparse.Namespace) -> list[str]:
        command = [
            str(args.realtime_preview_python), "-m", self.worker_module,
            "--engine", str(args.realtime_preview_engine),
            "--model", str(args.realtime_preview_model),
            "--provider", str(args.realtime_preview_provider),
            "--num-threads", str(args.realtime_preview_num_threads),
            "--language", str(getattr(args, "realtime_preview_language", getattr(args, "language", "en"))),
        ]
        if args.realtime_preview_model_path is not None:
            command.extend(["--model-path", str(args.realtime_preview_model_path)])
        download_root = args.realtime_preview_download_root or args.download_root
        if download_root is not None:
            command.extend(["--download-root", str(download_root)])
        if args.realtime_preview_engine_options_json:
            command.extend(["--engine-options-json", args.realtime_preview_engine_options_json])
        if args.realtime_preview_realtimestt_root is not None:
            command.extend(["--realtimestt-root", str(args.realtime_preview_realtimestt_root)])
        return command

    def build_worker_environment(self) -> dict[str, str]:
        env = super().build_worker_environment()
        env["KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT"] = "1"
        return env


class SherpaOnnxSubprocessPreviewTranscriber(JsonLineSubprocessPreviewTranscriber):
    worker_module = "workers.sherpa_onnx_realtime_preview_worker"
    worker_label = "Nemotron preview"

    def build_worker_command(self, args: argparse.Namespace) -> list[str]:
        model_dir = getattr(args, "realtime_preview_model_dir", None)
        if model_dir is None:
            raise ValueError("Nemotron realtime preview requires --realtime-preview-model-dir.")
        return [
            str(args.realtime_preview_python), "-m", self.worker_module,
            "--model-dir", str(model_dir),
            "--language", str(getattr(args, "realtime_preview_language", getattr(args, "language", "en"))),
            "--num-threads", str(args.realtime_preview_num_threads),
        ]


def create_realtime_preview_transcriber(args: argparse.Namespace) -> RealtimePreviewTranscriber:
    from window.realtime_preview_backends import normalize_preview_engine

    engine = normalize_preview_engine(args.realtime_preview_engine)
    if engine == "mock":
        return MockRealtimePreviewTranscriber()
    if engine == "kroko_onnx":
        if args.realtime_preview_python is not None and Path(args.realtime_preview_python).is_file():
            return KrokoSubprocessPreviewTranscriber(args)
        return KrokoRealtimePreviewTranscriber(args)
    if engine == "sherpa_onnx":
        return SherpaOnnxSubprocessPreviewTranscriber(args)
    raise ValueError(f"Unsupported realtime preview engine: {engine}")


def infer_kroko_preview_chunk_seconds(model: Any) -> float:
    match = re.search(r"-(\d+)-[LMS](?:[-_/\\.]|$)", str(model or ""))
    if not match:
        return DEFAULT_KROKO_16L_CHUNK_SECONDS
    return max(KROKO_PREVIEW_FRAME_SECONDS, int(match.group(1)) * KROKO_PREVIEW_FRAME_SECONDS)
