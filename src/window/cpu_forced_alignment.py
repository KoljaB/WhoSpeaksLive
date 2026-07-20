"""Quality-first CPU final ASR built from streaming text and forced alignment.

Kroko or Nemotron supplies the transcript.  Whisper is not allowed to decode a
different transcript: its cross-attention is used only to place the supplied
text on the audio timeline.  This preserves the streaming recognizer's lower
word error rate while replacing approximate word boundaries with precise ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any

import numpy as np

from window.window_preview import (
    FinalRealtimeTranscript,
    FinalRealtimeWord,
    JsonLineSubprocessPreviewTranscriber,
    _text_word_chunks,
)


TARGET_SAMPLE_RATE = 16000
PREPEND_PUNCTUATION = "\"'“¿([{-"
APPEND_PUNCTUATION = "\"'.。,，!！?？:：”)]}、"


@dataclass(frozen=True)
class AlignmentHealth:
    used_alignment: bool
    reason: str
    mean_probability: float = 0.0
    low_probability_fraction: float = 1.0


def _normalized_word(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum() or character == "'")


def _resample_linear(audio: np.ndarray, source_rate: int) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == TARGET_SAMPLE_RATE or signal.size == 0:
        return signal
    if source_rate <= 0:
        raise ValueError("sample_rate must be positive")
    output_size = max(1, round(signal.size * TARGET_SAMPLE_RATE / source_rate))
    source_positions = np.linspace(0.0, signal.size - 1, output_size)
    return np.interp(source_positions, np.arange(signal.size), signal).astype(np.float32)


class WhisperTextAligner:
    """Align a fixed transcript without running Whisper text generation."""

    def __init__(
        self,
        model_name: str = "base",
        *,
        language: str = "en",
        compute_type: str = "int8",
        cpu_threads: int = 2,
        download_root: str | None = None,
        minimum_mean_probability: float = 0.15,
    ) -> None:
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer

        self.model_name = str(model_name or "base")
        self.language = str(language or "en")
        self.minimum_mean_probability = max(0.0, min(1.0, float(minimum_mean_probability)))
        self._lock = threading.Lock()
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=str(compute_type or "int8"),
            cpu_threads=max(1, int(cpu_threads)),
            num_workers=1,
            download_root=download_root,
        )
        self._tokenizer = Tokenizer(
            self._model.hf_tokenizer,
            self._model.model.is_multilingual,
            task="transcribe",
            language=self.language,
        )

    def align(self, audio: np.ndarray, sample_rate: int, text: str) -> tuple[FinalRealtimeTranscript | None, AlignmentHealth]:
        fixed_text = str(text or "").strip()
        chunks = _text_word_chunks(fixed_text)
        if not chunks:
            return FinalRealtimeTranscript(text="", words=()), AlignmentHealth(True, "empty transcript", 1.0, 0.0)

        from faster_whisper.audio import pad_or_trim
        from faster_whisper.transcribe import merge_punctuations

        signal = _resample_linear(audio, int(sample_rate))
        duration = signal.size / float(TARGET_SAMPLE_RATE)
        try:
            with self._lock:
                features = self._model.feature_extractor(signal)
                num_frames = min(self._model.feature_extractor.nb_max_frames, features.shape[-1] - 1)
                encoded = self._model.encode(pad_or_trim(features[:, :num_frames]))
                tokens = self._tokenizer.encode(fixed_text)
                raw = self._model.find_alignment(
                    self._tokenizer,
                    [tokens],
                    encoded,
                    num_frames,
                    median_filter_width=7,
                )[0]
            merge_punctuations(raw, PREPEND_PUNCTUATION, APPEND_PUNCTUATION)
            aligned = [item for item in raw if str(item.get("word") or "").strip()]
        except Exception as exc:
            return None, AlignmentHealth(False, f"alignment error: {type(exc).__name__}: {exc}")

        source_words = [_normalized_word(chunk) for chunk in chunks]
        aligned_words = [_normalized_word(str(item.get("word") or "")) for item in aligned]
        if aligned_words != source_words:
            return None, AlignmentHealth(
                False,
                f"word mapping mismatch ({len(aligned_words)} aligned, {len(source_words)} source)",
            )

        probabilities = [float(item.get("probability") or 0.0) for item in aligned]
        mean_probability = float(np.mean(probabilities)) if probabilities else 0.0
        low_fraction = sum(value < 0.05 for value in probabilities) / max(1, len(probabilities))
        if mean_probability < self.minimum_mean_probability or low_fraction > 0.50:
            return None, AlignmentHealth(
                False,
                "alignment confidence below safety threshold",
                mean_probability,
                low_fraction,
            )

        words: list[FinalRealtimeWord] = []
        previous_end = 0.0
        for chunk, item, probability in zip(chunks, aligned, probabilities):
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if not all(math.isfinite(value) for value in (start, end)):
                return None, AlignmentHealth(False, "non-finite word timestamp", mean_probability, low_fraction)
            if start < previous_end - 0.08 or end < start or end > duration + 0.20:
                return None, AlignmentHealth(False, "non-monotonic or out-of-range word timestamp", mean_probability, low_fraction)
            start = max(0.0, min(duration, start))
            end = max(start, min(duration, end))
            words.append(FinalRealtimeWord(chunk, start, end, probability=probability))
            previous_end = end
        return (
            FinalRealtimeTranscript(text=fixed_text, words=tuple(words)),
            AlignmentHealth(True, "forced alignment accepted", mean_probability, low_fraction),
        )


class CpuHybridTranscriber:
    """Use CPU streaming ASR for text and Whisper only for final timestamps."""

    def __init__(
        self,
        source: JsonLineSubprocessPreviewTranscriber,
        aligner: WhisperTextAligner,
    ) -> None:
        self.source = source
        self.aligner = aligner
        self.last_health = AlignmentHealth(False, "not run")

    def transcribe_final(self, audio: np.ndarray, sample_rate: int) -> FinalRealtimeTranscript:
        native = self.source.transcribe_final(audio, sample_rate)
        aligned, health = self.aligner.align(audio, sample_rate, native.text)
        self.last_health = health
        return aligned if aligned is not None else native

    def close(self) -> None:
        self.source.close()
