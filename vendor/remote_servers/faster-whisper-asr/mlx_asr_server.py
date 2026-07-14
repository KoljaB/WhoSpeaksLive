#!/usr/bin/env python3
"""MLX whisper ASR server for Apple Silicon.

Same wire contract as asr_server.py's /health and /transcribe-window (the
only two endpoints the controller uses, see src/window/window_remote_asr.py):
PCM float32/pcm16 body + query params in, faster-whisper-shaped JSON out.

mlx-whisper is greedy-only (no beam search), so beam_size is accepted and
ignored — everything else in the request is honored where mlx_whisper.transcribe
supports it.

Apple Silicon only. Not added to requirements.txt (that file targets the
CUDA/faster-whisper server and mlx-whisper won't install off-arm64); install
it separately into this venv: `uv pip install -p .venv mlx-whisper`.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL_REPO = os.environ.get("ASR_MLX_MODEL", "mlx-community/whisper-large-v3-turbo")
DEFAULT_LANGUAGE = os.environ.get("ASR_LANGUAGE", os.environ.get("WHOSPEAKS_ASR_LANGUAGE", "en"))
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PORT", "8651"))
TARGET_SAMPLE_RATE = 16000

app = FastAPI(title="mlx-whisper ASR", version="1.0.0")
model_loaded_at: float | None = None

NONE_VALUES = {"", "none", "null"}


def value_or_default(values: dict[str, Any], name: str, default: Any) -> Any:
    value = values.get(name, default)
    if isinstance(value, str) and value.strip().lower() in NONE_VALUES:
        return None
    return value


def parse_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name}_must_be_int") from exc


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"{name}_must_be_boolean")


def start_parent_watchdog() -> None:
    # These servers are started by hand (docs/macos-setup.md) and hold multi-GB
    # models; if the launching shell dies they would otherwise run forever as
    # orphans. Exit once reparented. Opt out (nohup-style daemonizing) with
    # WHOSPEAKS_EXIT_WITH_PARENT=0.
    if os.environ.get("WHOSPEAKS_EXIT_WITH_PARENT", "1") in {"0", "false", "False"}:
        return
    parent = os.getppid()
    if parent <= 1:
        return

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


@app.on_event("startup")
def load_model() -> None:
    # mlx_whisper loads/caches the model lazily on first transcribe() call
    # (there is no separate "load model" API), so warm it up here to keep
    # /transcribe-window latency off the model-load cost.
    global model_loaded_at
    start_parent_watchdog()
    start = time.perf_counter()
    warmup = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(warmup, path_or_hf_repo=MODEL_REPO, language=DEFAULT_LANGUAGE)
    model_loaded_at = time.perf_counter() - start


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": model_loaded_at is not None,
        "service": "mlx-whisper-asr",
        "model": MODEL_REPO,
        "device": "mlx",
        "compute_type": "mlx-default",
        "language": DEFAULT_LANGUAGE,
        "model_loaded_seconds": model_loaded_at,
        "routes": ["/health", "/transcribe-window"],
    }


def finite_or_none(value: Any) -> float | None:
    # Greedy decode on very short windows can yield -inf avg_logprob / NaN
    # no_speech_prob; JSONResponse serializes with allow_nan=False and 500s.
    number = float(value)
    return number if math.isfinite(number) else None


def pcm_bytes_to_float32(audio_bytes: bytes, sample_rate: int, encoding: str) -> np.ndarray:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise HTTPException(status_code=400, detail="pcm_sample_rate_must_be_16000")

    normalized = encoding.lower().replace("-", "").replace("_", "")
    if normalized in {"pcm16", "s16le", "int16"}:
        if len(audio_bytes) % 2:
            raise HTTPException(status_code=400, detail="pcm16_payload_length_must_be_even")
        return np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if normalized in {"float32", "f32le"}:
        if len(audio_bytes) % 4:
            raise HTTPException(status_code=400, detail="float32_payload_length_must_be_multiple_of_4")
        return np.frombuffer(audio_bytes, dtype="<f4").astype(np.float32, copy=False)
    raise HTTPException(status_code=400, detail="unsupported_pcm_encoding")


@app.post("/transcribe-window")
async def transcribe_window(request: Request) -> JSONResponse:
    values = dict(request.query_params)
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty_audio_payload")

    decode_started = time.perf_counter()
    sample_rate = parse_int(value_or_default(values, "sample_rate", TARGET_SAMPLE_RATE), "sample_rate")
    encoding = str(value_or_default(values, "encoding", "float32"))
    audio = pcm_bytes_to_float32(audio_bytes, sample_rate, encoding)
    decode_seconds = time.perf_counter() - decode_started

    language = value_or_default(values, "language", DEFAULT_LANGUAGE)
    if isinstance(language, str) and language.strip().lower() in {"auto", "detect"}:
        language = None
    word_timestamps = parse_bool(value_or_default(values, "word_timestamps", True), "word_timestamps")
    # beam_size is accepted for API compatibility but ignored: mlx-whisper is greedy-only.

    started = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        language=language,
        word_timestamps=word_timestamps,
        condition_on_previous_text=False,
    )
    elapsed = time.perf_counter() - started
    # MLX's unified-memory buffer cache is unbounded by default and ratchets up
    # to the largest window ever transcribed, inflating RSS by tens of GB over a
    # long session; drop it after each request so memory stays near model size.
    mx.clear_cache()

    segments = []
    all_words = []
    for segment in result["segments"]:
        words = [
            {
                "start": finite_or_none(w["start"]),
                "end": finite_or_none(w["end"]),
                "word": w["word"],
                "text": w["word"],
                "probability": finite_or_none(w["probability"]),
            }
            for w in segment.get("words", [])
        ]
        all_words.extend(words)
        item = {
            "id": segment["id"],
            "start": finite_or_none(segment["start"]),
            "end": finite_or_none(segment["end"]),
            "text": segment["text"],
            "avg_logprob": finite_or_none(segment["avg_logprob"]),
            "no_speech_prob": finite_or_none(segment["no_speech_prob"]),
        }
        if words:
            item["words"] = words
        segments.append(item)

    return JSONResponse(
        {
            "text": result["text"].strip(),
            "language": language or result.get("language"),
            "detected_language": result.get("language"),
            "language_probability": None,
            "duration": len(audio) / TARGET_SAMPLE_RATE,
            "duration_after_vad": None,
            "elapsed_seconds": elapsed,
            "decode_seconds": decode_seconds,
            "input_mode": f"window_{encoding.lower()}",
            "model": MODEL_REPO,
            "device": "mlx",
            "compute_type": "mlx-default",
            "segments": segments,
            "segment_count": len(segments),
            "words": all_words,
            "word_count": len(all_words),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mlx_asr_server:app", host=HOST, port=PORT, log_level="info")
