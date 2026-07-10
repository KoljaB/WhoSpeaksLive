"""JSON-lines subprocess worker for Nemotron 3.5 streaming preview text."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from window.sherpa_onnx_models import validate_sherpa_onnx_model_dir


TARGET_SAMPLE_RATE = 16000


def write_message(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu")
    return parser.parse_args()


def resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")
    output_size = max(1, round(audio.size * TARGET_SAMPLE_RATE / sample_rate))
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, num=output_size, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def result_text(result: object) -> str:
    return str(getattr(result, "text", result) or "").strip()


@dataclass
class NemotronRecognizer:
    recognizer: Any
    language: str
    stream: Any

    @classmethod
    def load(cls, model_dir: Path, language: str, num_threads: int, provider: str) -> "NemotronRecognizer":
        if str(provider).strip().lower() != "cpu":
            raise ValueError("Nemotron realtime preview currently supports provider=cpu only.")
        directory = validate_sherpa_onnx_model_dir(model_dir)
        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed in the realtime preview Python environment. "
                "Install sherpa-onnx and sherpa-onnx-bin."
            ) from exc
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(directory / "tokens.txt"),
            encoder=str(directory / "encoder.int8.onnx"),
            decoder=str(directory / "decoder.int8.onnx"),
            joiner=str(directory / "joiner.int8.onnx"),
            num_threads=max(1, int(num_threads)),
            provider="cpu",
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )
        instance = cls(recognizer=recognizer, language=language, stream=None)
        instance.reset()
        return instance

    def reset(self) -> None:
        self.stream = self.recognizer.create_stream()
        self.stream.set_option("language", self.language)

    def accept(self, audio: np.ndarray, sample_rate: int) -> str:
        samples = resample_audio(audio, sample_rate)
        if samples.size:
            self.stream.accept_waveform(TARGET_SAMPLE_RATE, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return result_text(self.recognizer.get_result(self.stream))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.reset()
        return self.accept(audio, sample_rate)


def decode_request_audio(request: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(str(request.get("audio_b64") or ""), validate=True)
    if len(raw) % np.dtype(np.float32).itemsize:
        raise ValueError("audio_b64 does not contain complete float32 samples")
    return np.frombuffer(raw, dtype=np.float32).copy(), int(request.get("sample_rate") or TARGET_SAMPLE_RATE)


def main() -> int:
    args = parse_args()
    try:
        session = NemotronRecognizer.load(args.model_dir, args.language, args.num_threads, args.provider)
    except Exception as exc:
        write_message({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    write_message(
        {
            "ready": True,
            "backend": "sherpa_onnx",
            "model": args.model_dir.name,
            "language": args.language,
        }
    )
    for line in sys.stdin:
        request: Any = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            command = str(request.get("command") or "transcribe").strip().lower()
            if command == "reset":
                session.reset()
                write_message({"id": request_id, "ok": True, "text": ""})
                continue
            if command not in {"accept", "transcribe"}:
                raise ValueError(f"Unknown command: {command}")
            audio, sample_rate = decode_request_audio(request)
            text = session.accept(audio, sample_rate) if command == "accept" else session.transcribe(audio, sample_rate)
            write_message({"id": request_id, "text": text})
        except Exception as exc:
            request_id = request.get("id") if isinstance(request, dict) else None
            write_message({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
