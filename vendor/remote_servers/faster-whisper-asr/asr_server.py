#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # faster-whisper < 1.1
    BatchedInferencePipeline = None  # type: ignore[assignment,misc]

MODEL_NAME = os.environ.get("ASR_MODEL", "large-v2")
DEFAULT_LANGUAGE = os.environ.get("ASR_LANGUAGE", os.environ.get("WHOSPEAKS_ASR_LANGUAGE", "en"))
DEVICE = os.environ.get("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PORT", "8650"))
LOCAL_FILES_ONLY = os.environ.get("ASR_LOCAL_FILES_ONLY", "1") not in {"0", "false", "False"}
TARGET_SAMPLE_RATE = 16000

app = FastAPI(title="faster-whisper ASR", version="1.2.0")
model: WhisperModel | None = None
batched_model: BatchedInferencePipeline | None = None
model_loaded_at: float | None = None


def get_model() -> WhisperModel:
    if model is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")
    return model


def get_transcriber(batched: bool) -> Any:
    global batched_model
    loaded = get_model()
    if not batched or BatchedInferencePipeline is None:
        return loaded
    if batched_model is None or batched_model.model is not loaded:
        batched_model = BatchedInferencePipeline(model=loaded)
    return batched_model


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
    global model, model_loaded_at
    start_parent_watchdog()
    start = time.perf_counter()
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model_loaded_at = time.perf_counter() - start


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": model is not None,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": DEFAULT_LANGUAGE,
        "model_loaded_seconds": model_loaded_at,
        "batched_inference": BatchedInferencePipeline is not None,
        "routes": [
            "/transcribe",
            "/transcribe-memory",
            "/transcribe-pcm16",
            "/transcribe-window",
            "/transcribe-file",
            "/v1/audio/transcriptions",
        ],
    }


async def read_audio_bytes(request: Request, file: Optional[UploadFile]) -> tuple[bytes, str, str | None]:
    if file is not None:
        payload = await file.read()
        filename = file.filename or "audio"
        content_type = file.content_type
    else:
        payload = await request.body()
        filename = request.headers.get("x-filename", "audio")
        content_type = request.headers.get("content-type")

    if not payload:
        raise HTTPException(status_code=400, detail="empty_audio_payload")
    return payload, filename, content_type


def suffix_for(filename: str, content_type: str | None) -> str:
    ext = Path(filename).suffix
    if ext:
        return ext
    if content_type == "audio/wav":
        return ".wav"
    if content_type in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if content_type == "audio/ogg":
        return ".ogg"
    if content_type == "audio/webm":
        return ".webm"
    return ".bin"


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
NONE_VALUES = {"", "none", "null"}


async def request_values(request: Request) -> dict[str, Any]:
    values: dict[str, Any] = {}
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        form = await request.form()
        for key, value in form.multi_items():
            if isinstance(value, str):
                values[key] = value
    values.update(dict(request.query_params))
    return values


def value_or_default(values: dict[str, Any], name: str, default: Any) -> Any:
    value = values.get(name, default)
    if isinstance(value, str) and value.strip().lower() in NONE_VALUES:
        return None
    return value


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise HTTPException(status_code=400, detail=f"{name}_must_be_boolean")


def parse_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name}_must_be_int") from exc


def parse_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name}_must_be_float") from exc


def parse_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return parse_float(value, name)


def parse_temperature(value: Any) -> float | list[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="temperature_json_must_be_list")
        return [float(item) for item in parsed]
    if "," in text:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    return float(text)


def parse_int_list(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail=f"{name}_json_must_be_list")
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_json_object(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{name}_must_be_json_object")
    return parsed


def normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"auto", "detect"}:
        return None
    if text.lower() in {"english", "eng"}:
        return "en"
    return text


def transcribe_options(values: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    get = lambda name, default: value_or_default(values, name, defaults.get(name, default))
    return {
        "language": normalize_language(get("language", DEFAULT_LANGUAGE)),
        "task": str(get("task", "transcribe")),
        "beam_size": parse_int(get("beam_size", 1), "beam_size"),
        "best_of": parse_int(get("best_of", 5), "best_of"),
        "patience": parse_float(get("patience", 1), "patience"),
        "length_penalty": parse_float(get("length_penalty", 1), "length_penalty"),
        "repetition_penalty": parse_float(get("repetition_penalty", 1), "repetition_penalty"),
        "no_repeat_ngram_size": parse_int(get("no_repeat_ngram_size", 0), "no_repeat_ngram_size"),
        "temperature": parse_temperature(get("temperature", "0")),
        "compression_ratio_threshold": parse_optional_float(
            get("compression_ratio_threshold", 2.4),
            "compression_ratio_threshold",
        ),
        "log_prob_threshold": parse_optional_float(get("log_prob_threshold", -1.0), "log_prob_threshold"),
        "no_speech_threshold": parse_optional_float(get("no_speech_threshold", 0.6), "no_speech_threshold"),
        "condition_on_previous_text": parse_bool(
            get("condition_on_previous_text", False),
            "condition_on_previous_text",
        ),
        "prompt_reset_on_temperature": parse_float(
            get("prompt_reset_on_temperature", 0.5),
            "prompt_reset_on_temperature",
        ),
        "initial_prompt": get("initial_prompt", None),
        "prefix": get("prefix", None),
        "suppress_blank": parse_bool(get("suppress_blank", True), "suppress_blank"),
        "suppress_tokens": parse_int_list(get("suppress_tokens", "-1"), "suppress_tokens"),
        "without_timestamps": parse_bool(get("without_timestamps", False), "without_timestamps"),
        "max_initial_timestamp": parse_float(get("max_initial_timestamp", 1.0), "max_initial_timestamp"),
        "word_timestamps": parse_bool(get("word_timestamps", False), "word_timestamps"),
        "prepend_punctuations": str(get("prepend_punctuations", "\"'“¿([{-")),
        "append_punctuations": str(get("append_punctuations", "\"'.。,，!！?？:：”)]}、")),
        "multilingual": parse_bool(get("multilingual", False), "multilingual"),
        "vad_filter": parse_bool(get("vad_filter", False), "vad_filter"),
        "vad_parameters": parse_json_object(get("vad_parameters", None), "vad_parameters"),
        "max_new_tokens": None if get("max_new_tokens", None) is None else parse_int(get("max_new_tokens", None), "max_new_tokens"),
        "chunk_length": None if get("chunk_length", None) is None else parse_int(get("chunk_length", None), "chunk_length"),
        "clip_timestamps": get("clip_timestamps", "0"),
        "hallucination_silence_threshold": parse_optional_float(
            get("hallucination_silence_threshold", None),
            "hallucination_silence_threshold",
        ),
        "hotwords": get("hotwords", None),
        "language_detection_threshold": parse_optional_float(
            get("language_detection_threshold", 0.5),
            "language_detection_threshold",
        ),
        "language_detection_segments": parse_int(
            get("language_detection_segments", 1),
            "language_detection_segments",
        ),
    }


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        container = av.open(io.BytesIO(audio_bytes), mode="r")
    except av.FFmpegError as exc:
        raise HTTPException(status_code=400, detail=f"audio_decode_failed: {exc}") from exc

    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise HTTPException(status_code=400, detail="no_audio_stream")

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                array = resampled.to_ndarray()
                if array.ndim > 1:
                    array = array.reshape(-1)
                chunks.append(array.astype(np.float32, copy=False) / 32768.0)

        tail = resampler.resample(None)
        if tail is not None:
            if not isinstance(tail, list):
                tail = [tail]
            for resampled in tail:
                array = resampled.to_ndarray()
                if array.ndim > 1:
                    array = array.reshape(-1)
                chunks.append(array.astype(np.float32, copy=False) / 32768.0)
    finally:
        container.close()

    if not chunks:
        raise HTTPException(status_code=400, detail="no_audio_samples")
    return np.concatenate(chunks)


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


def run_transcription(
    audio: str | np.ndarray,
    *,
    options: dict[str, Any],
    input_mode: str,
    decode_seconds: float | None = None,
    batched: bool = False,
    batch_size: int = 16,
) -> JSONResponse:
    started = time.perf_counter()
    effective_batched = bool(batched and BatchedInferencePipeline is not None)
    transcribe_options = dict(options)
    if effective_batched:
        transcribe_options["batch_size"] = max(1, int(batch_size))
        transcribe_options["vad_filter"] = True
    segments_iter, info = get_transcriber(effective_batched).transcribe(audio, **transcribe_options)

    segments = []
    all_words = []
    text_parts = []
    for segment in segments_iter:
        words = []
        if getattr(segment, "words", None):
            words = [
                {
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "text": w.word,
                    "probability": w.probability,
                }
                for w in segment.words
            ]
            all_words.extend(words)
        item = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
        }
        if words:
            item["words"] = words
        segments.append(item)
        text_parts.append(segment.text)

    elapsed = time.perf_counter() - started
    return JSONResponse(
        {
            "text": "".join(text_parts).strip(),
            "language": options.get("language") or info.language,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "elapsed_seconds": elapsed,
            "decode_seconds": decode_seconds,
            "input_mode": input_mode,
            "model": MODEL_NAME,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "options": options,
            "batched": effective_batched,
            "batch_size": max(1, int(batch_size)) if effective_batched else None,
            "segments": segments,
            "segment_count": len(segments),
            "words": all_words,
            "word_count": len(all_words),
        }
    )


@app.post("/transcribe")
@app.post("/transcribe-memory")
@app.post("/v1/audio/transcriptions")
async def transcribe_memory(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    values = await request_values(request)
    audio_bytes, _filename, _content_type = await read_audio_bytes(request, file)
    decode_started = time.perf_counter()
    audio = decode_audio_bytes(audio_bytes)
    decode_seconds = time.perf_counter() - decode_started
    batched = parse_bool(value_or_default(values, "batched", False), "batched")
    batch_size = parse_int(value_or_default(values, "batch_size", 16), "batch_size")
    return run_transcription(
        audio,
        options=transcribe_options(values),
        input_mode="encoded_memory",
        decode_seconds=decode_seconds,
        batched=batched,
        batch_size=batch_size,
    )


@app.post("/transcribe-pcm16")
async def transcribe_pcm16(
    request: Request,
) -> JSONResponse:
    values = await request_values(request)
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty_audio_payload")
    decode_started = time.perf_counter()
    sample_rate = parse_int(value_or_default(values, "sample_rate", TARGET_SAMPLE_RATE), "sample_rate")
    encoding = str(value_or_default(values, "encoding", "pcm16"))
    audio = pcm_bytes_to_float32(audio_bytes, sample_rate, encoding)
    decode_seconds = time.perf_counter() - decode_started
    batched = parse_bool(value_or_default(values, "batched", False), "batched")
    batch_size = parse_int(value_or_default(values, "batch_size", 16), "batch_size")
    return run_transcription(
        audio,
        options=transcribe_options(values),
        input_mode=encoding.lower(),
        decode_seconds=decode_seconds,
        batched=batched,
        batch_size=batch_size,
    )


@app.post("/transcribe-window")
async def transcribe_window(request: Request) -> JSONResponse:
    values = await request_values(request)
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty_audio_payload")
    decode_started = time.perf_counter()
    sample_rate = parse_int(value_or_default(values, "sample_rate", TARGET_SAMPLE_RATE), "sample_rate")
    encoding = str(value_or_default(values, "encoding", "float32"))
    audio = pcm_bytes_to_float32(audio_bytes, sample_rate, encoding)
    decode_seconds = time.perf_counter() - decode_started
    batched = parse_bool(value_or_default(values, "batched", False), "batched")
    batch_size = parse_int(value_or_default(values, "batch_size", 16), "batch_size")
    return run_transcription(
        audio,
        options=transcribe_options(
            values,
            defaults={
                "beam_size": 5,
                "word_timestamps": True,
                "vad_filter": False,
                "condition_on_previous_text": False,
            },
        ),
        input_mode=f"window_{encoding.lower()}",
        decode_seconds=decode_seconds,
        batched=batched,
        batch_size=batch_size,
    )


@app.post("/transcribe-file")
async def transcribe_file(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    values = await request_values(request)
    audio_bytes, filename, content_type = await read_audio_bytes(request, file)
    suffix = suffix_for(filename, content_type)

    with tempfile.NamedTemporaryFile(prefix="asr_", suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        batched = parse_bool(value_or_default(values, "batched", False), "batched")
        batch_size = parse_int(value_or_default(values, "batch_size", 16), "batch_size")
        return run_transcription(
            tmp.name,
            options=transcribe_options(values),
            input_mode="temp_file",
            decode_seconds=None,
            batched=batched,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asr_server:app", host=HOST, port=PORT, log_level="info")
