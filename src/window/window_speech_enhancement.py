"""HTTP client for optional final-path speech enhancement."""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


class SpeechEnhancementClient:
    """Send mono float32 audio to the UniSE raw-audio endpoint."""

    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("Speech-enhancement base URL must not be empty.")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._input_seconds = 0.0
        self._http_seconds = 0.0
        self._queue_seconds = 0.0
        self._processing_seconds = 0.0

    def health(self) -> dict[str, Any]:
        try:
            with urlopen(f"{self.base_url}/health", timeout=min(self.timeout_seconds, 10.0)) as response:
                raw = response.read()
        except (HTTPError, URLError, OSError) as exc:
            raise RuntimeError(f"Speech-enhancement health check failed: {exc}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Speech-enhancement health endpoint returned invalid JSON.") from exc
        if not isinstance(payload, dict) or not bool(payload.get("ok")):
            raise RuntimeError(f"Speech-enhancement service is not ready: {payload!r}")
        return payload

    def enhance(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        samples = np.nan_to_num(
            np.asarray(audio, dtype=np.float32).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        input_rate = int(sample_rate)
        if samples.size <= 0:
            return samples, input_rate

        query = urlencode({
            "sample_rate": input_rate,
            "encoding": "float32",
            "channels": 1,
        })
        request = Request(
            f"{self.base_url}/enhance-pcm16?{query}",
            data=np.ascontiguousarray(samples.astype("<f4", copy=False)).tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                headers = response.headers
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Speech enhancement HTTP {exc.code}: {detail[:300]}") from exc
        except (URLError, OSError) as exc:
            raise RuntimeError(f"Speech-enhancement request failed: {exc}") from exc
        http_seconds = time.monotonic() - started

        try:
            enhanced, output_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError("Speech-enhancement response was not a readable WAV file.") from exc
        enhanced = np.asarray(enhanced, dtype=np.float32)
        if enhanced.ndim > 1:
            enhanced = enhanced.mean(axis=1)
        enhanced = np.nan_to_num(enhanced.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        output_rate = int(output_rate)

        # Preserve the original duration so ASR word offsets remain anchored to
        # the raw canonical timeline even when the service resamples to 16 kHz.
        expected_samples = max(1, int(round(samples.size * output_rate / float(input_rate))))
        if enhanced.size > expected_samples:
            enhanced = enhanced[:expected_samples]
        elif enhanced.size < expected_samples:
            enhanced = np.pad(enhanced, (0, expected_samples - enhanced.size))

        input_seconds = samples.size / float(input_rate)
        queue_seconds = self._header_float(headers, "X-UniSE-Queue-Seconds")
        processing_seconds = self._header_float(headers, "X-UniSE-Processing-Seconds")
        with self._stats_lock:
            self._request_count += 1
            self._input_seconds += input_seconds
            self._http_seconds += http_seconds
            self._queue_seconds += queue_seconds
            self._processing_seconds += processing_seconds
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False), output_rate

    def stats(self) -> dict[str, float | int]:
        with self._stats_lock:
            return {
                "request_count": self._request_count,
                "input_seconds": round(self._input_seconds, 6),
                "http_seconds": round(self._http_seconds, 6),
                "queue_seconds": round(self._queue_seconds, 6),
                "processing_seconds": round(self._processing_seconds, 6),
            }

    @staticmethod
    def _header_float(headers: Any, name: str) -> float:
        try:
            return max(0.0, float(headers.get(name, 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0
