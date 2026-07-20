"""JSON-lines subprocess worker for sherpa-onnx streaming ASR."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from window.sherpa_onnx_models import sherpa_onnx_model_files, validate_sherpa_onnx_model_dir
from workers.structured_realtime_result import structured_result_payload


TARGET_SAMPLE_RATE = 16000


def write_message(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--model-family", choices=("nemotron", "kroko"), default="nemotron")
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
    model_family: str = "nemotron"

    @classmethod
    def load(
        cls,
        model_dir: Path,
        language: str,
        num_threads: int,
        provider: str,
        model_family: str = "nemotron",
    ) -> "NemotronRecognizer":
        if str(provider).strip().lower() != "cpu":
            raise ValueError("Nemotron realtime preview currently supports provider=cpu only.")
        directory = validate_sherpa_onnx_model_dir(model_dir)
        encoder, decoder, joiner, tokens = sherpa_onnx_model_files(directory)
        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed in the realtime preview Python environment. "
                "Install sherpa-onnx and sherpa-onnx-bin."
            ) from exc
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=max(1, int(num_threads)),
            provider="cpu",
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )
        instance = cls(recognizer=recognizer, language=language, stream=None, model_family=model_family)
        instance.reset()
        return instance

    def reset(self) -> None:
        self.stream = self._new_stream()

    def _new_stream(self) -> Any:
        stream = self.recognizer.create_stream()
        if self.model_family == "nemotron":
            stream.set_option("language", self.language)
        return stream

    def accept(self, audio: np.ndarray, sample_rate: int) -> str:
        return result_text(self.accept_result(audio, sample_rate))

    def accept_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        finalize: bool = False,
        stream: Any | None = None,
    ) -> object:
        active_stream = self.stream if stream is None else stream
        samples = resample_audio(audio, sample_rate)
        if samples.size:
            active_stream.accept_waveform(TARGET_SAMPLE_RATE, samples)
        if finalize:
            active_stream.accept_waveform(
                TARGET_SAMPLE_RATE,
                np.zeros(int(1.0 * TARGET_SAMPLE_RATE), dtype=np.float32),
            )
            active_stream.input_finished()
        while self.recognizer.is_ready(active_stream):
            self.recognizer.decode_stream(active_stream)
        get_result_all = getattr(self.recognizer, "get_result_all", None)
        return get_result_all(active_stream) if callable(get_result_all) else self.recognizer.get_result(active_stream)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.reset()
        return self.accept(audio, sample_rate)

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> object:
        # Final sentence decoding must not reset the long-lived preview stream.
        # This lets one loaded recognizer serve both realtime text and final ASR.
        return self.accept_result(
            audio,
            sample_rate,
            finalize=True,
            stream=self._new_stream(),
        )


def decode_request_audio(request: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(str(request.get("audio_b64") or ""), validate=True)
    if len(raw) % np.dtype(np.float32).itemsize:
        raise ValueError("audio_b64 does not contain complete float32 samples")
    return np.frombuffer(raw, dtype=np.float32).copy(), int(request.get("sample_rate") or TARGET_SAMPLE_RATE)


def main() -> int:
    args = parse_args()
    try:
        session = NemotronRecognizer.load(
            args.model_dir,
            args.language,
            args.num_threads,
            args.provider,
            args.model_family,
        )
    except Exception as exc:
        write_message({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    write_message(
        {
            "ready": True,
            "backend": "sherpa_onnx",
            "model_family": args.model_family,
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
            if command not in {"accept", "transcribe", "transcribe_final"}:
                raise ValueError(f"Unknown command: {command}")
            audio, sample_rate = decode_request_audio(request)
            if command == "transcribe_final":
                result = session.transcribe_final(audio, sample_rate)
                write_message({"id": request_id, **structured_result_payload(result)})
            else:
                text = session.accept(audio, sample_rate) if command == "accept" else session.transcribe(audio, sample_rate)
                write_message({"id": request_id, "text": text})
        except Exception as exc:
            request_id = request.get("id") if isinstance(request, dict) else None
            write_message({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
