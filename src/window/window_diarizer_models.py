"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import json
import math
import mimetypes
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from stream2sentence import generate_sentences, init_tokenizer

from common.audio_utils import load_audio_file, pad_audio, trim_silence, write_wav
from embeddings.embedding_providers import EmbeddingSubprocessClient, RemoteEmbeddingClient
from speakers.speaker_embedding_cluster import (
    SpeakerDecision,
    SpeakerMemory,
    cosine_similarity,
    normalize_vector,
)
from window.window_config import (
    DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD,
    DEFAULT_REALTIMESTT_ROOT,
    DEFAULT_SPEAKER_LIBRARY_DIR,
    NEW_SPEAKER_SENSITIVITY_FIELDS,
    NEW_SPEAKER_SENSITIVITY_PRESETS,
    SILERO_VAD_CHUNK_SAMPLES,
    SILERO_VAD_SAMPLE_RATE,
    default_silero_vad_backend,
    download_kroko_preview_model,
    list_speaker_groups,
    normalize_new_speaker_sensitivity,
    safe_library_name,
    safe_reference_filename,
    speaker_group_dir,
)
from window.language_config import default_sentence_language, default_sentence_tokenizer
from window.window_domain import (
    EmbeddingSentenceJob,
    LiveSpeakerMemoryUpdateJob,
    MediaFiles,
    PendingUnknownSentence,
    SentencePart,
    TimedWord,
    VadWindowState,
    WindowTranscript,
)
from window.window_events import EventBus
from window.audio_timeline import AudioSnapshot, AudioTimeline
from window.diarization_config import DiarizationConfig
from window.diarization_run import DiarizationRun, DiarizationRunState
from window.diarization_session import DiarizationSession
from window.speaker_assignment_engine import AssignmentRequest, SpeakerAssignmentEngine
from window.window_media import resolve_browser_stream_id
from window.window_preview import (
    RealtimePreviewTranscriber,
    create_realtime_preview_transcriber,
)
from window.realtime_preview_backends import normalize_preview_engine
from window.sherpa_onnx_models import ensure_sherpa_onnx_model, validate_sherpa_onnx_model_dir
from window.review_flags import annotate_review
from window.window_remote_asr import RemoteWindowAsrClient
from window.window_speech_enhancement import SpeechEnhancementClient
from window.window_text import (
    is_embedding_candidate_text,
    round_optional,
    sentence_initial_uppercase_after_strong_boundary,
    split_words_with_stream2sentence,
    text_content_words,
    text_ends_sentence,
    word_attr,
)
from window.window_speaker_refinement import (
    DelayedClusteringConfig,
    SpeakerRefinementConfig,
    find_delayed_speaker_splits,
    find_speaker_prototype_revisions,
    rejected_speaker_labels,
    user_deleted_speaker_label,
    user_confirmed_speaker_label,
)




class WindowModelRuntimeMixin:
    def _load_speech_enhancement(self) -> None:
        if not (
            bool(getattr(self.args, "enhance_asr", False))
            or bool(getattr(self.args, "enhance_embeddings", False))
        ):
            return
        if self._speech_enhancement_client is not None:
            return
        client = SpeechEnhancementClient(
            getattr(self.args, "speech_enhancement_url", "http://127.0.0.1:8651"),
            getattr(self.args, "speech_enhancement_timeout_seconds", 120.0),
        )
        self.bus.emit("status", {"message": f"Checking speech enhancement at {client.base_url}."})
        health = client.health()
        self._speech_enhancement_client = client
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Speech enhancement ready at {client.base_url} "
                    f"(sample_rate={health.get('sample_rate')}, segment_seconds={health.get('segment_seconds')})."
                )
            },
        )

    def _enhance_audio(self, audio: np.ndarray, sample_rate: int, *, path: str) -> tuple[np.ndarray, int]:
        enabled = (
            bool(getattr(self.args, "enhance_asr", False))
            if path == "asr"
            else bool(getattr(self.args, "enhance_embeddings", False))
        )
        if not enabled:
            return audio, sample_rate
        client = self._speech_enhancement_client
        if client is None:
            raise RuntimeError("Speech enhancement was enabled but its client was not initialized.")
        return client.enhance(audio, sample_rate)

    def _load_model(self) -> None:
        with self._model_lock:
            if self._model is not None:
                self.bus.emit("status", {"message": "ASR backend already loaded."})
                return
            asr_backend = str(self.args.asr_backend or "local").strip().lower().replace("-", "_")
            if asr_backend == "remote":
                client = RemoteWindowAsrClient(
                    self.args.remote_asr_url,
                    self.args.remote_asr_timeout_seconds,
                    language=getattr(self.args, "language", "en"),
                )
                self.bus.emit("status", {"message": f"Checking remote ASR server at {client.base_url}."})
                health = client.health()
                health_status = health.get("status") or health.get("model") or health.get("raw") or "ok"
                self._model = client
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Remote faster-whisper large-v2 ASR ready at {client.base_url} for {client.language} "
                            f"(health={health_status})."
                        )
                    },
                )
                return
            self.bus.emit("status", {"message": "Importing faster-whisper."})
            from faster_whisper import WhisperModel

            self.bus.emit("status", {"message": f"Loading faster-whisper {self.args.model} for {getattr(self.args, 'language', 'en')} on {self.args.device} before playback."})
            self._model = WhisperModel(
                self.args.model,
                device=self.args.device,
                compute_type=self.args.compute_type,
                download_root=str(self.args.download_root) if self.args.download_root else None,
            )
            self.bus.emit("status", {"message": "faster-whisper ready; starting synchronized playback."})

    def _transcribe_audio_words(self, model: Any, audio: np.ndarray, sample_rate: int) -> tuple[list[TimedWord], int]:
        if isinstance(model, RemoteWindowAsrClient):
            words, segment_count = model.transcribe_window(audio, sample_rate, self.args.beam_size)
            return self._filter_asr_no_speech_words(words), segment_count

        segments, _info = model.transcribe(
            audio,
            language=getattr(self.args, "language", "en"),
            task="transcribe",
            beam_size=self.args.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        words: list[TimedWord] = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            segment_index = segment_count - 1
            no_speech_prob = self._optional_float(word_attr(segment, "no_speech_prob", None))
            avg_logprob = self._optional_float(word_attr(segment, "avg_logprob", None))
            compression_ratio = self._optional_float(word_attr(segment, "compression_ratio", None))
            for word in getattr(segment, "words", None) or []:
                text = str(word_attr(word, "word", "") or "")
                if not text.strip():
                    continue
                words.append(
                    TimedWord(
                        text,
                        float(word_attr(word, "start", 0.0)),
                        float(word_attr(word, "end", 0.0)),
                        probability=self._optional_float(word_attr(word, "probability", None)),
                        no_speech_prob=no_speech_prob,
                        avg_logprob=avg_logprob,
                        compression_ratio=compression_ratio,
                        segment_index=segment_index,
                    )
                )
        words.sort(key=lambda item: (item.start, item.end))
        return self._filter_asr_no_speech_words(words), segment_count

    def _transcribe_enhanced_final_audio_text(
        self,
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
    ) -> str:
        enhanced, enhanced_rate = self._enhance_audio(audio, sample_rate, path="asr")
        words, _segment_count = self._transcribe_audio_words(model, enhanced, enhanced_rate)
        return "".join(word.text for word in words).strip()

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _filter_asr_no_speech_words(self, words: list[TimedWord]) -> list[TimedWord]:
        if not bool(getattr(self.args, "asr_no_speech_filter", True)):
            return words
        threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_prob_threshold", 0.65))))
        hard_threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_hard_threshold", 0.85))))
        keep_short_max_words = max(0, int(getattr(self.args, "asr_no_speech_keep_short_max_words", 2)))
        keep_short_max_seconds = max(0.0, float(getattr(self.args, "asr_no_speech_keep_short_max_seconds", 0.45)))
        kept: list[TimedWord] = []
        dropped_words = 0
        dropped_segments = 0
        max_dropped_prob = 0.0

        def segment_key(word: TimedWord, fallback_index: int) -> tuple[object, ...]:
            if word.segment_index is not None:
                return ("segment", int(word.segment_index))
            if word.no_speech_prob is not None:
                return (
                    "metadata",
                    float(word.no_speech_prob),
                    word.avg_logprob,
                    word.compression_ratio,
                )
            return ("word", fallback_index)

        groups: list[list[TimedWord]] = []
        current_group: list[TimedWord] = []
        current_key: tuple[object, ...] | None = None
        for index, word in enumerate(words):
            key = segment_key(word, index)
            if current_group and key != current_key:
                groups.append(current_group)
                current_group = []
            current_group.append(word)
            current_key = key
        if current_group:
            groups.append(current_group)

        for group in groups:
            probability_values = [float(word.no_speech_prob) for word in group if word.no_speech_prob is not None]
            probability = max(probability_values) if probability_values else None
            if probability is None:
                kept.extend(group)
                continue
            start = min(float(word.start) for word in group)
            end = max(float(word.end) for word in group)
            duration = max(0.0, end - start)
            is_short_interjection = (
                probability < hard_threshold
                and len(group) <= keep_short_max_words
                and duration <= keep_short_max_seconds
            )
            if probability >= threshold and not is_short_interjection:
                dropped_words += len(group)
                dropped_segments += 1
                max_dropped_prob = max(max_dropped_prob, probability)
                continue
            kept.extend(group)
        bus = getattr(self, "bus", None)
        if dropped_words and bus is not None:
            bus.emit(
                "status",
                {
                    "message": (
                        f"ASR no-speech filter dropped {dropped_words} word(s) from {dropped_segments} segment(s) "
                        f"(max no_speech_prob={max_dropped_prob:.2f}, threshold={threshold:.2f})."
                    )
                },
            )
        return kept

    def _transcribe_window_audio_words(
        self,
        model: Any,
        left: float,
        right: float,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> tuple[list[TimedWord], int]:
        spans = speech_spans if speech_spans is not None else [(left, right)]
        words: list[TimedWord] = []
        segment_count = 0
        for span_left, span_right in spans:
            span_left = max(left, min(right, float(span_left)))
            span_right = max(span_left, min(right, float(span_right)))
            if span_right <= span_left:
                continue
            window, sample_rate = self._audio_window_copy(span_left, span_right)
            if window.size <= 0:
                continue
            relative_words, relative_segment_count = self._transcribe_audio_words(model, window, sample_rate)
            segment_count += relative_segment_count
            for word in relative_words:
                start = span_left + float(word.start)
                end = span_left + float(word.end)
                if end <= span_left or start >= span_right:
                    continue
                words.append(
                    TimedWord(
                        word.text,
                        max(left, min(right, start)),
                        max(left, min(right, end)),
                        probability=word.probability,
                        no_speech_prob=word.no_speech_prob,
                        avg_logprob=word.avg_logprob,
                        compression_ratio=word.compression_ratio,
                        segment_index=word.segment_index,
                    )
                )
        words.sort(key=lambda item: (item.start, item.end))
        return words, segment_count

    def _warm_asr_transcription(self, force: bool = False) -> None:
        if self._asr_probe_warmed and not force:
            self.bus.emit("status", {"message": "ASR warmup transcription already complete."})
            return
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise RuntimeError("ASR backend did not load.")

        sample_rate = int(self.sample_rate)
        probe_samples = max(1, int(sample_rate * 0.75))
        probe, sample_rate = self._audio_window_copy(0.0, probe_samples / float(sample_rate))
        if probe.size < probe_samples:
            padded = np.zeros(probe_samples, dtype=np.float32)
            padded[:probe.size] = probe
            probe = padded

        started = time.monotonic()
        try:
            words, segment_count = self._transcribe_audio_words(self._model, probe, sample_rate)
        except RuntimeError as exc:
            asr_backend = str(getattr(self.args, "asr_backend", "local") or "local").strip().lower().replace("-", "_")
            if asr_backend != "remote":
                raise
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Remote ASR warmup failed after server health check; "
                        f"continuing and retrying during transcription ({exc})."
                    )
                },
            )
            return
        self._asr_probe_warmed = True
        self._asr_probe_warmed_at = time.monotonic()
        self.bus.emit(
            "status",
            {
                "message": (
                    f"ASR warmup transcription complete in {self._asr_probe_warmed_at - started:.2f}s "
                    f"(segments={segment_count}, words={len(words)})."
                )
            },
        )

    def _embed_audio_chunk_with_client(self, client: Any, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        embed_audio = getattr(client, "embed_audio", None)
        if callable(embed_audio) and not self.args.keep_segment_audio:
            return embed_audio(audio, sample_rate)

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="window-diarize-", dir=str(self.args.output_dir), delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            write_wav(wav_path, audio, sample_rate)
            return client.embed_wav(wav_path)
        finally:
            if not self.args.keep_segment_audio:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _embed_audio_chunk(self, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        if np.any(np.asarray(audio, dtype=np.float32)):
            audio, sample_rate = self._enhance_audio(audio, sample_rate, path="embeddings")
        return self._embed_audio_chunk_with_client(self.embedding, audio, sample_rate, suffix)

    def _embed_live_audio_chunk(self, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        return self._embed_audio_chunk_with_client(self.live_embedding, audio, sample_rate, suffix)

    def _warm_embedding(self, force: bool = False) -> None:
        if self._embedding_warmed and not force:
            self.bus.emit("status", {"message": "Speaker embedding model already warm."})
            return
        warmup_label = "Refreshing" if self._embedding_warmed else "Warming"
        self.bus.emit("status", {"message": f"{warmup_label} speaker embedding model before playback."})
        if isinstance(self.embedding, RemoteEmbeddingClient):
            self.bus.emit("status", {"message": f"Checking remote embeddings server at {self.embedding.base_url}."})
            health = self.embedding.health()
            health_status = health.get("status") or health.get("service") or health.get("raw") or "ok"
            self.bus.emit("status", {"message": f"Remote embeddings server ready at {self.embedding.base_url} (health={health_status})."})
        started = time.monotonic()
        self._embed_audio_chunk(np.zeros(int(self.sample_rate * 0.6), dtype=np.float32), self.sample_rate, ".warm.wav")
        if self._live_embedding_separate:
            self._embed_live_audio_chunk(
                np.zeros(int(self.sample_rate * 0.6), dtype=np.float32),
                self.sample_rate,
                ".live-warm.wav",
            )
        self._embedding_warmed = True
        self._embedding_warmed_at = time.monotonic()
        self.bus.emit("status", {"message": f"Speaker embedding model ready in {self._embedding_warmed_at - started:.2f}s."})
