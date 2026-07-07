"""Modal deployment for the WhoSpeaks remote ASR backend."""

import os
import time
from typing import Any

import modal

APP_NAME = "whospeaks-live-asr"
ENDPOINT_LABEL = "whospeaks-live-asr"
MODEL_NAME = os.environ.get("WHOSPEAKS_MODAL_ASR_MODEL", "large-v2")
DEFAULT_LANGUAGE = os.environ.get("WHOSPEAKS_ASR_LANGUAGE", os.environ.get("WHOSPEAKS_LANGUAGE", "en"))
CACHE_DIR = "/cache"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.6",
        "faster-whisper==1.2.1",
        "numpy==2.4.6",
    )
)

model_cache = modal.Volume.from_name("whospeaks-faster-whisper-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

_model: Any = None
_loaded_at: float | None = None


def _get_model() -> Any:
    global _model, _loaded_at
    if _model is None:
        from faster_whisper import WhisperModel

        started = time.monotonic()
        _model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
            download_root=CACHE_DIR,
        )
        _loaded_at = time.monotonic() - started
    return _model


@app.function(
    gpu="T4",
    timeout=600,
    startup_timeout=600,
    scaledown_window=900,
    volumes={CACHE_DIR: model_cache},
)
@modal.asgi_app(label=ENDPOINT_LABEL)
def create_asgi_app():
    from fastapi import FastAPI, HTTPException, Query, Request
    import numpy as np

    web = FastAPI(title="WhoSpeaks Remote ASR")

    @web.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": APP_NAME,
            "model": MODEL_NAME,
            "device": "cuda",
            "compute_type": "float16",
            "language": DEFAULT_LANGUAGE,
            "model_loaded": _model is not None,
            "model_load_seconds": _loaded_at,
        }

    @web.post("/transcribe-window")
    async def transcribe_window(
        request: Request,
        sample_rate: int = Query(..., ge=1),
        encoding: str = Query("float32"),
        language: str = Query(DEFAULT_LANGUAGE),
        task: str = Query("transcribe"),
        beam_size: int = Query(5, ge=1),
        word_timestamps: bool = Query(True),
        vad_filter: bool = Query(False),
        condition_on_previous_text: bool = Query(False),
    ) -> dict[str, Any]:
        if encoding != "float32":
            raise HTTPException(status_code=400, detail="Only float32 raw audio is supported.")

        raw_audio = await request.body()
        if not raw_audio:
            return {
                "segments": [],
                "words": [],
                "segment_count": 0,
                "sample_rate": sample_rate,
            }
        if len(raw_audio) % 4:
            raise HTTPException(status_code=400, detail="float32 audio byte length must be divisible by 4.")

        audio = np.frombuffer(raw_audio, dtype=np.float32).copy()
        if audio.size <= 0:
            return {
                "segments": [],
                "words": [],
                "segment_count": 0,
                "sample_rate": sample_rate,
            }

        model = _get_model()
        started = time.monotonic()
        segments_iter, info = model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )

        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        for segment in segments_iter:
            segment_words = []
            for word in getattr(segment, "words", None) or []:
                word_payload = {
                    "word": str(getattr(word, "word", "") or ""),
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "probability": float(getattr(word, "probability", 0.0) or 0.0),
                }
                segment_words.append(word_payload)
                words.append(word_payload)
            segments.append(
                {
                    "id": int(getattr(segment, "id", len(segments)) or 0),
                    "start": float(getattr(segment, "start", 0.0) or 0.0),
                    "end": float(getattr(segment, "end", 0.0) or 0.0),
                    "text": str(getattr(segment, "text", "") or ""),
                    "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                    "compression_ratio": float(getattr(segment, "compression_ratio", 0.0) or 0.0),
                    "words": segment_words,
                }
            )

        return {
            "segments": segments,
            "words": words,
            "segment_count": len(segments),
            "sample_rate": sample_rate,
            "language": getattr(info, "language", language),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration_seconds": float(audio.size) / float(sample_rate),
            "transcribe_seconds": round(time.monotonic() - started, 4),
        }

    return web
