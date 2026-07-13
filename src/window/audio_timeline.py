"""Thread-safe ownership of media audio and synchronized playback state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from common.audio_utils import load_audio_file
from window.window_domain import MediaFiles
from window.window_media import resolve_browser_stream_id


AudioLoader = Callable[[Path], tuple[np.ndarray, int]]


@dataclass(frozen=True)
class AudioSnapshot:
    """An immutable description of one coherent audio/playback revision."""

    media: MediaFiles
    audio: np.ndarray
    sample_rate: int
    duration: float
    streaming: bool
    stream_samples: int
    playback_time: float
    playback_clock_started_at: float | None
    revision: int


class AudioTimeline:
    """The sole writer for audio buffers, media identity, and playback time."""

    def __init__(
        self,
        media: MediaFiles,
        *,
        audio_loader: AudioLoader = load_audio_file,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        audio, sample_rate = audio_loader(media.audio_file)
        self._lock = threading.RLock()
        self._monotonic = monotonic
        self._media = media
        self._audio = self._normalize(audio)
        self._sample_rate = int(sample_rate)
        self._streaming = False
        self._stream_chunks: list[np.ndarray] = []
        self._stream_samples = 0
        self._duration = len(self._audio) / float(self._sample_rate)
        self._playback_time = 0.0
        self._playback_clock_started_at: float | None = None
        self._last_playback_jump_warning_at = 0.0
        self._revision = 0

    @property
    def lock(self) -> threading.RLock:
        """Compatibility lock for code being migrated into this owner."""

        return self._lock

    def snapshot(self, *, copy_audio: bool = True) -> AudioSnapshot:
        with self._lock:
            audio = self._combined_audio_locked()
            if copy_audio:
                audio = audio.copy()
            audio.setflags(write=False)
            return AudioSnapshot(
                media=self._media,
                audio=audio,
                sample_rate=self._sample_rate,
                duration=self._duration,
                streaming=self._streaming,
                stream_samples=self._stream_samples,
                playback_time=self._playback_time,
                playback_clock_started_at=self._playback_clock_started_at,
                revision=self._revision,
            )

    def replace_file(self, media: MediaFiles, *, audio_loader: AudioLoader = load_audio_file) -> AudioSnapshot:
        audio, sample_rate = audio_loader(media.audio_file)
        normalized = self._normalize(audio)
        with self._lock:
            self._media = media
            self._audio = normalized
            self._sample_rate = int(sample_rate)
            self._streaming = False
            self._stream_chunks = []
            self._stream_samples = 0
            self._duration = len(normalized) / float(sample_rate)
            self._reset_playback_locked()
            self._revision += 1
        return self.snapshot(copy_audio=False)

    def begin_stream(self, url: str) -> AudioSnapshot:
        with self._lock:
            video_id = resolve_browser_stream_id(url)
            self._media = MediaFiles(url, video_id, self._media.audio_file, self._media.video_file)
            self._audio = np.zeros(0, dtype=np.float32)
            self._sample_rate = 16000
            self._streaming = True
            self._stream_chunks = []
            self._stream_samples = 0
            self._duration = 0.0
            self._reset_playback_locked()
            self._revision += 1
        return self.snapshot(copy_audio=False)

    def append(self, audio: np.ndarray, sample_rate: int) -> float:
        chunk = self._normalize(audio)
        with self._lock:
            if not self._streaming:
                raise RuntimeError("Browser audio stream is not active.")
            if int(sample_rate) != self._sample_rate:
                raise RuntimeError(
                    f"Browser audio sample rate changed from {self._sample_rate} to {sample_rate}."
                )
            if chunk.size:
                self._stream_chunks.append(chunk.copy())
                self._stream_samples += int(chunk.size)
                self._duration = self._stream_samples / float(self._sample_rate)
                self._revision += 1
            duration = self._duration
        self.set_playback_time(duration)
        return duration

    def set_playback_time(self, seconds: float, *, reset: bool = False) -> float:
        with self._lock:
            value = min(self._duration, max(0.0, float(seconds)))
            self._playback_time = value
            if reset:
                self._playback_clock_started_at = None
                self._last_playback_jump_warning_at = 0.0
            self._revision += 1
            return value

    def start_playback_clock(self) -> None:
        with self._lock:
            self._playback_clock_started_at = self._monotonic()
            self._last_playback_jump_warning_at = 0.0

    def stop_playback_clock(self) -> None:
        with self._lock:
            self._playback_clock_started_at = None

    def window(self, left: float, right: float) -> tuple[np.ndarray, int]:
        with self._lock:
            start = max(0, min(self._stream_samples if self._streaming else len(self._audio), int(left * self._sample_rate)))
            end = max(start, min(self._stream_samples if self._streaming else len(self._audio), int(right * self._sample_rate)))
            audio = self._combined_audio_locked()
            return audio[start:end].copy(), self._sample_rate

    def write_stream_audio(self, path: Path, writer: Callable[[Path, np.ndarray, int], None]) -> None:
        snapshot = self.snapshot(copy_audio=True)
        writer(path, snapshot.audio, snapshot.sample_rate)

    def _combined_audio_locked(self) -> np.ndarray:
        if not self._streaming:
            return self._audio
        if not self._stream_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._stream_chunks)

    def _reset_playback_locked(self) -> None:
        self._playback_time = 0.0
        self._playback_clock_started_at = None
        self._last_playback_jump_warning_at = 0.0

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        chunk = np.nan_to_num(chunk, copy=False)
        return np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=True)
