"""Data model for the browser-synced window diarization GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS = 0.06
DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS = 0.09
DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO = 0.6


@dataclass
class MediaFiles:
    url: str
    video_id: str
    audio_file: Path
    video_file: Path


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    probability: float | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    segment_index: int | None = None


@dataclass
class MappedWord:
    word: TimedWord
    text_start: int
    text_end: int


@dataclass
class SentencePart:
    text: str
    start: float
    end: float
    next_left: float
    spoken_word_seconds: float
    speech_audio_ratio: float
    words: list[dict[str, Any]] = field(default_factory=list)
    first_word_start: float | None = None
    last_word_end: float | None = None
    next_word_start: float | None = None
    gap_to_next_word_seconds: float | None = None
    boundary_strategy: str = ""
    sentence_boundary_pre_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS
    sentence_boundary_post_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS
    sentence_boundary_gap_ratio: float = DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO


@dataclass
class WindowTranscript:
    sentences: list[SentencePart]
    word_count: int
    segment_count: int


@dataclass
class VadWindowState:
    has_speech: bool
    should_flush: bool
    speech_start: float | None = None
    speech_end: float | None = None
    speech_seconds: float = 0.0
    trailing_silence_seconds: float = 0.0
    backend: str = ""
    speech_spans: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PendingUnknownSentence:
    index: int
    base_payload: dict[str, Any]
    embedding: np.ndarray
    duration_seconds: float


@dataclass
class EmbeddingSentenceJob:
    index: int
    base_payload: dict[str, Any]
    text: str
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    speaker_generation: int = 0


@dataclass
class LiveSpeakerMemoryUpdateJob:
    speaker_id: str
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    suffix: str = ".live-profile.wav"
    speaker_generation: int = 0
