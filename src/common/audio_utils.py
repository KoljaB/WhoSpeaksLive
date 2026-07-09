"""Shared audio, numeric, and JSON helpers for diarization scripts."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
INT16_MAX_ABS_VALUE = 32767.0


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def audio_to_float_mono(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)

    if isinstance(value, np.ndarray):
        audio = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) % 4 == 0:
            audio = np.frombuffer(raw, dtype=np.float32)
        elif len(raw) % 2 == 0:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
        else:
            audio = np.empty(0, dtype=np.float32)
    else:
        audio = np.asarray(value)

    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype.kind in {"i", "u"}:
        audio = audio.astype(np.float32) / INT16_MAX_ABS_VALUE
    else:
        audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio.reshape(-1), -1.0, 1.0).astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < int(sample_rate * 0.25):
        return audio

    frame = max(1, int(sample_rate * 0.03))
    hop = max(1, int(sample_rate * 0.01))
    if len(audio) < frame:
        return audio

    rms = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        chunk = audio[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not rms:
        return audio

    peak = max(rms)
    threshold = max(peak * 0.08, 0.003)
    active = [index for index, value in enumerate(rms) if value >= threshold]
    if not active:
        return audio

    pad = int(sample_rate * 0.10)
    start = max(0, active[0] * hop - pad)
    end = min(len(audio), active[-1] * hop + frame + pad)
    return audio[start:end]


def pad_audio(audio: np.ndarray, minimum_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    minimum_samples = int(max(0.0, minimum_seconds) * sample_rate)
    if len(audio) >= minimum_samples:
        return audio
    return np.pad(audio, (0, minimum_samples - len(audio))).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX_ABS_VALUE).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def _audio_frame_to_mono_float(frame: Any) -> np.ndarray:
    array = np.asarray(frame.to_ndarray())
    if array.ndim > 1:
        if array.shape[0] <= array.shape[-1]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    return audio_to_float_mono(array)


def _load_audio_file_with_av(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    import av
    from av.audio.resampler import AudioResampler

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((candidate for candidate in container.streams if candidate.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {path}")
        resampler = AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]
            chunks.extend(_audio_frame_to_mono_float(item) for item in resampled)
        try:
            flushed = resampler.resample(None)
        except Exception:
            flushed = []
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            chunks.extend(_audio_frame_to_mono_float(item) for item in flushed)
    if not chunks:
        raise ValueError(f"No audio samples decoded from {path}")
    return np.concatenate(chunks).astype(np.float32), sample_rate


def _load_audio_file_with_librosa(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    import librosa

    audio, source_rate = librosa.load(str(path), sr=sample_rate, mono=True, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(audio, dtype=np.float32), int(source_rate or sample_rate)


def load_audio_file(path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    import soundfile as sf

    try:
        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as soundfile_error:
        for loader in (_load_audio_file_with_av, _load_audio_file_with_librosa):
            try:
                return loader(path, sample_rate)
            except ModuleNotFoundError:
                continue
            except Exception:
                continue
        raise soundfile_error
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
        except ModuleNotFoundError:
            import torch
            import torchaudio.functional as audio_functional

            tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            audio = audio_functional.resample(
                tensor,
                orig_freq=int(source_rate),
                new_freq=int(sample_rate),
            ).cpu().numpy()
        source_rate = sample_rate
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(audio, dtype=np.float32), source_rate


def normalize_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Embedding provider returned an empty vector.")
    return (vector / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Embedding shape mismatch: {left.shape} vs {right.shape}")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        raise ValueError("Softmax temperature must be greater than zero.")
    max_value = max(values)
    exps = [math.exp((value - max_value) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
