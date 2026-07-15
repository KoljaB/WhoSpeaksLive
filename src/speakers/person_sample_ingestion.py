"""Quality-gated ingestion of one manual audio source into one Voice sample."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from common.audio_utils import load_audio_file, pad_audio, trim_silence, write_wav
from speakers.speaker_embedding_cluster import cosine_similarity, normalize_vector


MIN_MANUAL_SPEECH_SECONDS = 1.5
MANUAL_WINDOW_SECONDS = 8.0
MIN_MANUAL_RMS = 0.0005


def _decode_audio(audio_b64: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    encoded = str(audio_b64 or "").split(",", 1)[-1]
    if not encoded:
        raise ValueError("Voice sample audio is missing.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Voice sample audio is not valid base64 data.") from exc
    if not raw:
        raise ValueError("Voice sample audio is empty.")
    if len(raw) > max_bytes:
        raise ValueError("Voice sample audio is too large.")
    return raw


def _embed_window(client: Any, audio: np.ndarray, sample_rate: int, path: Path) -> np.ndarray:
    if hasattr(client, "embed_audio"):
        value = client.embed_audio(audio, sample_rate)
    else:
        write_wav(path, audio, sample_rate)
        value = client.embed_wav(path)
    try:
        vector = normalize_vector(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("The embedding provider returned an invalid Voice representation.") from exc
    if vector.size <= 0 or not np.all(np.isfinite(vector)):
        raise ValueError("The embedding provider returned an invalid Voice representation.")
    return vector.astype(np.float32)


def ingest_manual_voice_sample(
    library: Any,
    embedding_client: Any,
    *,
    person_id: str,
    embedding_provider: str,
    filename: str,
    audio_b64: str,
    label: str = "",
    source_type: str = "manual_upload",
) -> dict[str, Any]:
    """Decode, quality-check, robustly aggregate, and persist one source."""

    raw = _decode_audio(audio_b64)
    suffix = Path(filename or "sample.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="whospeaks-voice-sample-") as tmp:
        source_path = Path(tmp) / f"source{suffix}"
        source_path.write_bytes(raw)
        try:
            audio, sample_rate = load_audio_file(source_path)
        except Exception as exc:
            raise ValueError(f"Could not decode Voice sample audio: {exc}") from exc
        usable = trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate)
        seconds = len(usable) / float(sample_rate or 16000)
        if seconds < MIN_MANUAL_SPEECH_SECONDS:
            raise ValueError(
                f"Voice sample needs at least {MIN_MANUAL_SPEECH_SECONDS:.1f} seconds of usable speech."
            )
        rms = float(np.sqrt(np.mean(np.square(usable, dtype=np.float64)))) if len(usable) else 0.0
        if not np.isfinite(rms) or rms < MIN_MANUAL_RMS:
            raise ValueError("Voice sample is silent or too quiet to use.")
        clipping_ratio = float(np.mean(np.abs(usable) >= 0.999))
        if clipping_ratio > 0.20:
            raise ValueError("Voice sample is severely clipped; record it again at a lower level.")

        window_size = max(1, int(MANUAL_WINDOW_SECONDS * sample_rate))
        windows = [usable[offset : offset + window_size] for offset in range(0, len(usable), window_size)]
        windows = [window for window in windows if len(window) / sample_rate >= MIN_MANUAL_SPEECH_SECONDS]
        if not windows:
            windows = [usable]
        embeddings = [
            _embed_window(
                embedding_client,
                pad_audio(window, MIN_MANUAL_SPEECH_SECONDS, sample_rate),
                sample_rate,
                Path(tmp) / f"window-{index}.wav",
            )
            for index, window in enumerate(windows)
        ]
        expected_shape = embeddings[0].shape
        embeddings = [embedding for embedding in embeddings if embedding.shape == expected_shape]
        preliminary = normalize_vector(np.mean(np.stack(embeddings), axis=0))
        similarities = np.asarray(
            [cosine_similarity(embedding, preliminary) for embedding in embeddings],
            dtype=np.float64,
        )
        median = float(np.median(similarities))
        cutoff = median - max(0.08, 3.0 * float(np.median(np.abs(similarities - median))))
        retained = [embedding for embedding, score in zip(embeddings, similarities) if score >= cutoff]
        if not retained:
            raise ValueError("Voice sample did not contain a coherent voice.")
        centroid = normalize_vector(np.mean(np.stack(retained), axis=0))
        cohesion = float(np.mean([cosine_similarity(item, centroid) for item in retained]))
        if len(retained) > 1 and cohesion < 0.50:
            raise ValueError("Voice sample is too inconsistent to form reliable recognition evidence.")
        quality = max(0.0, min(1.0, (1.0 - clipping_ratio) * min(1.0, seconds / 8.0) * max(0.5, cohesion)))
        return library.add_manual_sample(
            person_id,
            centroid,
            embedding_provider=embedding_provider,
            raw_audio=raw,
            filename=filename,
            label=label,
            source_type=source_type,
            speech_seconds=seconds,
            sentence_count=len(retained),
            quality=quality,
            cohesion=cohesion,
            outlier_count=len(embeddings) - len(retained),
        )
