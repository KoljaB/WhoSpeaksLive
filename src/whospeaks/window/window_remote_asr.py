"""HTTP client for remote window ASR transcription."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from whospeaks.window.window_domain import TimedWord


def _word_attr(word: Any, name: str, default: Any = None) -> Any:
    if isinstance(word, dict):
        return word.get(name, default)
    return getattr(word, name, default)


class RemoteWindowAsrClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("Remote ASR base URL must not be empty.")
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def health(self) -> dict[str, Any]:
        raw = self._read_url(f"{self.base_url}/health", timeout=min(self.timeout_seconds, 10.0))
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        return data if isinstance(data, dict) else {"value": data}

    def transcribe_window(self, window: np.ndarray, sample_rate: int, beam_size: int) -> tuple[list[TimedWord], int]:
        query = urlencode({
            "sample_rate": int(sample_rate),
            "encoding": "float32",
            "language": "en",
            "task": "transcribe",
            "beam_size": int(beam_size),
            "word_timestamps": "true",
            "vad_filter": "false",
            "condition_on_previous_text": "false",
        })
        url = f"{self.base_url}/transcribe-window?{query}"
        audio_bytes = np.ascontiguousarray(window.astype(np.float32, copy=False)).tobytes()
        request = Request(
            url,
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        raw = self._open_request(request, timeout=self.timeout_seconds)
        try:
            result = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Remote ASR returned non-JSON transcription response.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Remote ASR returned an unexpected transcription response.")
        if result.get("error"):
            raise RuntimeError(f"Remote ASR error: {result['error']}")
        return self._timed_words_from_result(result)

    def _timed_words_from_result(self, result: dict[str, Any]) -> tuple[list[TimedWord], int]:
        raw_segments = result.get("segments")
        raw_words = result.get("words")
        if raw_words is None and isinstance(raw_segments, list):
            raw_words = []
            for segment in raw_segments:
                segment_words = _word_attr(segment, "words", [])
                if isinstance(segment_words, list):
                    raw_words.extend(segment_words)
        if raw_words is None:
            raw_words = []
        if not isinstance(raw_words, list):
            raise RuntimeError("Remote ASR response field 'words' must be a list.")

        words: list[TimedWord] = []
        for word in raw_words:
            text = str(_word_attr(word, "word", _word_attr(word, "text", "")) or "")
            if not text.strip():
                continue
            try:
                start = float(_word_attr(word, "start", 0.0) or 0.0)
                end = float(_word_attr(word, "end", start) or start)
            except (TypeError, ValueError):
                continue
            start_seconds = max(0.0, start)
            end_seconds = max(start_seconds, end)
            words.append(TimedWord(text=text, start=start_seconds, end=end_seconds))
        words.sort(key=lambda item: (item.start, item.end))

        segment_count_value = result.get("segment_count", result.get("segments_count"))
        try:
            segment_count = int(segment_count_value)
        except (TypeError, ValueError):
            segment_count = len(raw_segments) if isinstance(raw_segments, list) else (1 if words else 0)
        return words, segment_count

    def _read_url(self, url: str, timeout: float) -> bytes:
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote ASR HTTP {exc.code}: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote ASR connection failed: {exc.reason}") from exc

    def _open_request(self, request: Request, timeout: float) -> bytes:
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote ASR HTTP {exc.code}: {detail[:300]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Remote ASR connection failed: {exc.reason}") from exc
