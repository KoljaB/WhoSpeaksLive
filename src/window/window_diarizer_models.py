"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
import copy
from difflib import SequenceMatcher
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
from embeddings.embedding_providers import (
    EmbeddingResult,
    EmbeddingSubprocessClient,
    RemoteEmbeddingClient,
    parse_embedding_provider_stack_specs,
)
from speakers.speaker_embedding_cluster import (
    SpeakerDecision,
    SpeakerMemory,
    cosine_similarity,
    normalize_vector,
)
from window.asr_hallucination_policy import (
    match_asr_hallucination_policy,
    normalize_asr_hallucination_text,
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
    JsonLineSubprocessPreviewTranscriber,
    RealtimePreviewTranscriber,
    create_realtime_preview_transcriber,
)
from window.cpu_forced_alignment import CpuHybridTranscriber, WhisperTextAligner
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


_ASR_NON_SPEECH_ANNOTATION_RE = re.compile(
    r"(?<![A-Za-z])(?:LAUGHTER(?:\s+AND\s+APPLAUSE)?|(?:AND\s+)?APPLAUSE|CHEERING)(?![A-Za-z])"
)
_ASR_BRACKETED_NON_SPEECH_ANNOTATION_RE = re.compile(
    r"[\[(]\s*(?:laughter|applause|cheering)(?:\s+and\s+(?:laughter|applause|cheering))*\s*[\])]",
    re.IGNORECASE,
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
            if asr_backend == "cpu":
                engine = normalize_preview_engine(getattr(self.args, "realtime_preview_engine", "off"))
                if engine in {"off", "mock"}:
                    raise RuntimeError(
                        "CPU final ASR requires --realtime-preview-engine kroko_onnx or sherpa_onnx."
                    )
                self.bus.emit("status", {"message": f"Loading CPU transcript source ({engine})."})
                self._ensure_realtime_preview_model()
                client = create_realtime_preview_transcriber(self.args)
                if not isinstance(client, JsonLineSubprocessPreviewTranscriber):
                    client.close()
                    raise RuntimeError(
                        "CPU final ASR requires a subprocess preview Python so finalized word timestamps are available."
                    )
                alignment_model = str(getattr(self.args, "cpu_alignment_model", "base") or "base")
                alignment_threads = max(1, int(getattr(self.args, "cpu_alignment_threads", 2)))
                self.bus.emit(
                    "status",
                    {"message": f"Loading faster-whisper {alignment_model} CPU forced aligner (x{alignment_threads} threads)."},
                )
                aligner = WhisperTextAligner(
                    alignment_model,
                    language=getattr(self.args, "language", "en"),
                    compute_type=getattr(self.args, "cpu_alignment_compute_type", "int8"),
                    cpu_threads=alignment_threads,
                    download_root=str(self.args.download_root) if self.args.download_root else None,
                    minimum_mean_probability=float(getattr(self.args, "cpu_alignment_min_probability", 0.15)),
                )
                self._model = CpuHybridTranscriber(client, aligner)
                self.bus.emit(
                    "status",
                    {"message": f"Quality CPU ASR ready: {engine} text + {alignment_model} forced alignment."},
                )
                return
            self.bus.emit("status", {"message": "Importing faster-whisper."})
            from faster_whisper import WhisperModel

            self.bus.emit("status", {"message": f"Loading faster-whisper {self.args.model} for {getattr(self.args, 'language', 'en')} on {self.args.device}."})
            self._model = WhisperModel(
                self.args.model,
                device=self.args.device,
                compute_type=self.args.compute_type,
                download_root=str(self.args.download_root) if self.args.download_root else None,
            )
            self.bus.emit("status", {"message": "faster-whisper ready."})

    def _transcribe_audio_words(
        self,
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
        *,
        batched: bool = False,
        batch_size: int = 16,
    ) -> tuple[list[TimedWord], int]:
        if isinstance(model, RemoteWindowAsrClient):
            words, segment_count = model.transcribe_window(
                audio,
                sample_rate,
                self.args.beam_size,
                batched=batched,
                batch_size=batch_size,
            )
            return self._filter_asr_no_speech_words(words), segment_count

        if isinstance(model, (JsonLineSubprocessPreviewTranscriber, CpuHybridTranscriber)):
            transcript = model.transcribe_final(audio, sample_rate)
            words = [
                TimedWord(word.text, word.start, word.end, probability=word.probability, segment_index=0)
                for word in transcript.words
            ]
            if isinstance(model, CpuHybridTranscriber) and not model.last_health.used_alignment:
                self.bus.emit(
                    "status",
                    {"message": f"CPU forced alignment rejected; using native timestamp fallback ({model.last_health.reason})."},
                )
            return self._filter_asr_no_speech_words(words), 1 if transcript.text or words else 0

        transcriber = model
        extra_options: dict[str, Any] = {}
        if batched:
            try:
                from faster_whisper import BatchedInferencePipeline
            except ImportError:
                batched = False
                self.bus.emit(
                    "status",
                    {"message": "Installed faster-whisper has no batched pipeline; using full-file inference."},
                )
            else:
                cached = getattr(self, "_batched_asr_pipeline", None)
                if cached is None or getattr(cached, "model", None) is not model:
                    cached = BatchedInferencePipeline(model=model)
                    self._batched_asr_pipeline = cached
                transcriber = cached
                extra_options = {
                    "batch_size": max(1, int(batch_size)),
                }

        segments, _info = transcriber.transcribe(
            audio,
            language=getattr(self.args, "language", "en"),
            task="transcribe",
            beam_size=self.args.beam_size,
            word_timestamps=True,
            vad_filter=batched,
            condition_on_previous_text=False,
            **extra_options,
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
            return self._filter_asr_non_speech_annotations(
                self._filter_hard_asr_hallucination_words(words)
            )
        threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_prob_threshold", 0.65))))
        hard_threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_hard_threshold", 0.85))))
        keep_short_max_words = max(0, int(getattr(self.args, "asr_no_speech_keep_short_max_words", 2)))
        keep_short_max_seconds = max(0.0, float(getattr(self.args, "asr_no_speech_keep_short_max_seconds", 0.45)))
        flagged_words = 0
        flagged_segments = 0
        max_flagged_prob = 0.0

        for group in self._asr_segment_groups(words):
            probability_values = [float(word.no_speech_prob) for word in group if word.no_speech_prob is not None]
            probability = max(probability_values) if probability_values else None
            if probability is None or probability < threshold:
                continue
            start = min(float(word.start) for word in group)
            end = max(float(word.end) for word in group)
            duration = max(0.0, end - start)
            is_short_interjection = (
                probability < hard_threshold
                and len(group) <= keep_short_max_words
                and duration <= keep_short_max_seconds
            )
            if is_short_interjection:
                continue
            evidence_score = self._asr_segment_evidence_score(group)
            self._mark_asr_words_for_review(
                group,
                reason="conflicting ASR speech evidence",
                details={
                    "no_speech_probability": round(float(probability), 4),
                    "no_speech_threshold": round(float(threshold), 4),
                    "hard_no_speech_threshold": round(float(hard_threshold), 4),
                    "evidence_score": (
                        round(float(evidence_score), 4)
                        if evidence_score is not None
                        else None
                    ),
                },
            )
            flagged_words += len(group)
            flagged_segments += 1
            max_flagged_prob = max(max_flagged_prob, probability)
        bus = getattr(self, "bus", None)
        if flagged_words and bus is not None:
            bus.emit(
                "status",
                {
                    "message": (
                        f"ASR no-speech check retained {flagged_words} word(s) from "
                        f"{flagged_segments} conflicting segment(s) for verification/review "
                        f"(max no_speech_prob={max_flagged_prob:.2f}, threshold={threshold:.2f}); "
                        "no text was discarded on this signal alone."
                    )
                },
            )
        return self._filter_asr_non_speech_annotations(
            self._filter_hard_asr_hallucination_words(words)
        )

    def _filter_hard_asr_hallucination_words(self, words: list[TimedWord]) -> list[TimedWord]:
        """Keep raw text until acoustic verification can make an informed decision."""

        return words

    def _filter_asr_non_speech_annotations(self, words: list[TimedWord]) -> list[TimedWord]:
        """Remove only explicit stage-direction annotations, preserving nearby speech."""

        kept: list[TimedWord] = []
        removed = 0
        for group in self._asr_segment_groups(words):
            rewritten: list[str] = []
            annotation_indexes: set[int] = set()
            for index, word in enumerate(group):
                text = str(word.text or "")
                filtered = _ASR_BRACKETED_NON_SPEECH_ANNOTATION_RE.sub("", text)
                filtered = _ASR_NON_SPEECH_ANNOTATION_RE.sub("", filtered)
                rewritten.append(filtered)
                if filtered != text:
                    annotation_indexes.add(index)

            connector_indexes = {
                index
                for index, word in enumerate(group)
                if normalize_asr_hallucination_text(word.text) == "and"
                and (
                    index - 1 in annotation_indexes
                    or index + 1 in annotation_indexes
                )
                and "".join(character for character in word.text if character.isalpha()).isupper()
            }
            kept_group: list[TimedWord] = []
            for index, word in enumerate(group):
                if index in connector_indexes:
                    removed += 1
                    continue
                filtered = rewritten[index]
                if not filtered.strip():
                    if index in annotation_indexes:
                        removed += 1
                    continue
                if filtered != word.text:
                    word.text = filtered
                    removed += 1
                kept.append(word)
                kept_group.append(word)
            if annotation_indexes and kept_group:
                self._mark_asr_words_for_review(
                    kept_group,
                    reason="non-speech annotation removed before speaker embedding",
                    details={"removed_non_speech_annotation": True},
                )

        bus = getattr(self, "bus", None)
        if removed and bus is not None:
            bus.emit(
                "status",
                {"message": f"Removed {removed} non-speech ASR annotation token(s)."},
            )
        return kept

    @staticmethod
    def _mark_asr_words_for_review(
        words: list[TimedWord],
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        normalized_reason = " ".join(str(reason or "").split())
        if not normalized_reason:
            return
        normalized_details = dict(details or {})
        for word in words:
            current = getattr(word, "asr_review", None)
            review = dict(current) if isinstance(current, dict) else {}
            reasons = [
                str(item)
                for item in review.get("reasons", [])
                if str(item or "").strip()
            ]
            if normalized_reason not in reasons:
                reasons.append(normalized_reason)
            merged_details = (
                dict(review.get("details") or {})
                if isinstance(review.get("details"), dict)
                else {}
            )
            merged_details.update(normalized_details)
            word.asr_review = {
                "needs_review": True,
                "reasons": reasons,
                "details": merged_details,
            }

    def _record_suppressed_asr_candidate(self, candidate: dict[str, Any]) -> None:
        """Persist an auditable review notice for text omitted from the transcript."""

        record = {
            "schema_version": "asr_review_candidate_v1",
            "status": "suppressed",
            "needs_review": True,
            "reason": str(
                candidate.get("reason")
                or "weak primary ASR evidence with no independent speech transcript"
            ),
            "text": " ".join(str(candidate.get("text") or "").split()),
            "start": round(float(candidate.get("start") or 0.0), 4),
            "end": round(float(candidate.get("end") or 0.0), 4),
            "policy_rule": str(candidate.get("policy_rule") or ""),
            "evidence_score": candidate.get("evidence_score"),
            "primary_no_speech_probability": candidate.get("primary_no_speech_probability"),
            "independent_verifier": candidate.get("independent_verifier"),
            "independent_text": candidate.get("independent_text"),
            "independent_context_start": candidate.get("independent_context_start"),
            "independent_context_end": candidate.get("independent_context_end"),
            "independent_latency_ms": candidate.get("independent_latency_ms"),
            "shifted_text_stability": candidate.get("shifted_text_stability"),
            "shifted_evidence_score": candidate.get("shifted_evidence_score"),
            "repetition_count": candidate.get("repetition_count"),
            "independent_repetition_count": candidate.get("independent_repetition_count"),
        }
        key = (
            record["status"],
            record["text"].casefold(),
            record["start"],
            record["end"],
            record["policy_rule"],
        )
        lock = getattr(self, "_asr_review_candidate_lock", None)
        records = getattr(self, "_asr_review_candidates", None)
        if lock is None or not isinstance(records, list):
            return
        with lock:
            existing_keys = {
                (
                    item.get("status"),
                    str(item.get("text") or "").casefold(),
                    item.get("start"),
                    item.get("end"),
                    item.get("policy_rule"),
                )
                for item in records
                if isinstance(item, dict)
            }
            if key not in existing_keys:
                records.append(record)

    def _reconcile_suppressed_asr_candidates(
        self,
        retained_words: list[TimedWord],
        *,
        media_start_seconds: float,
    ) -> int:
        """Remove transient warnings when a later ASR view retains the same text."""

        retained: list[tuple[str, float, float]] = []
        for group in self._asr_segment_groups(retained_words):
            normalized = normalize_asr_hallucination_text(self._asr_group_text(group))
            if not normalized:
                continue
            retained.append(
                (
                    normalized,
                    float(media_start_seconds) + min(float(word.start) for word in group),
                    float(media_start_seconds) + max(float(word.end) for word in group),
                )
            )
        if not retained:
            return 0

        lock = getattr(self, "_asr_review_candidate_lock", None)
        records = getattr(self, "_asr_review_candidates", None)
        if lock is None or not isinstance(records, list):
            return 0

        def reconciled(record: dict[str, Any]) -> bool:
            if record.get("status") != "suppressed":
                return False
            if record.get("policy_rule") in {
                "repeated_short_phrase",
                "implausible_word_timing",
            }:
                return False
            candidate_text = normalize_asr_hallucination_text(str(record.get("text") or ""))
            if not candidate_text:
                return False
            candidate_start = float(record.get("start") or 0.0)
            candidate_end = float(record.get("end") or candidate_start)
            for text, start, end in retained:
                # The later transcript must contain the complete suppressed
                # phrase.  A short fragment such as "Thanks" is not evidence
                # that "Thanks for watching" was ultimately retained.
                same_text = candidate_text == text or candidate_text in text
                if not same_text:
                    continue
                overlaps = min(candidate_end, end) > max(candidate_start, start)
                near_same_time = abs(candidate_start - start) <= 1.0
                if overlaps or near_same_time:
                    return True
            return False

        with lock:
            before = len(records)
            records[:] = [
                record
                for record in records
                if not (isinstance(record, dict) and reconciled(record))
            ]
            return before - len(records)

    @staticmethod
    def _asr_segment_groups(words: list[TimedWord]) -> list[list[TimedWord]]:
        groups: list[list[TimedWord]] = []
        current: list[TimedWord] = []
        current_key: tuple[object, ...] | None = None
        for index, word in enumerate(words):
            key: tuple[object, ...]
            if word.segment_index is not None:
                key = ("segment", int(word.segment_index))
            else:
                key = ("word", index)
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(word)
            current_key = key
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _asr_group_text(words: list[TimedWord]) -> str:
        return "".join(word.text for word in words).strip()

    @staticmethod
    def _asr_segment_evidence_score(group: list[TimedWord]) -> float | None:
        """Fuse independent ASR confidence signals into a conservative 0..1 score."""

        no_speech_values = [
            float(word.no_speech_prob)
            for word in group
            if word.no_speech_prob is not None
        ]
        avg_logprob_values = [
            float(word.avg_logprob)
            for word in group
            if word.avg_logprob is not None
        ]
        word_probabilities = [
            max(1e-4, min(1.0, float(word.probability)))
            for word in group
            if word.probability is not None
        ]
        if not no_speech_values or not avg_logprob_values or not word_probabilities:
            return None

        speech_evidence = max(1e-4, min(1.0, 1.0 - max(no_speech_values)))
        sequence_evidence = max(1e-4, min(1.0, math.exp(min(avg_logprob_values))))
        word_evidence = math.exp(
            sum(math.log(value) for value in word_probabilities) / len(word_probabilities)
        )
        return float(
            math.exp(
                0.35 * math.log(speech_evidence)
                + 0.35 * math.log(sequence_evidence)
                + 0.30 * math.log(max(1e-4, word_evidence))
            )
        )

    @staticmethod
    def _asr_text_stability(first: list[TimedWord], second: list[TimedWord]) -> float:
        def tokens(items: list[TimedWord]) -> list[str]:
            return normalize_asr_hallucination_text(" ".join(word.text for word in items)).split()

        first_tokens = tokens(first)
        second_tokens = tokens(second)
        if not first_tokens or not second_tokens:
            return 0.0
        return float(SequenceMatcher(None, first_tokens, second_tokens).ratio())

    @staticmethod
    def _asr_context_text_similarity(candidate_text: str, context_text: str) -> float:
        """Return the best token-sequence match inside a longer context transcript."""

        candidate = normalize_asr_hallucination_text(candidate_text).split()
        context = normalize_asr_hallucination_text(context_text).split()
        if not candidate or not context:
            return 0.0
        minimum = max(1, len(candidate) - 2)
        maximum = min(len(context), len(candidate) + 2)
        best = 0.0
        for width in range(minimum, maximum + 1):
            for start in range(0, len(context) - width + 1):
                best = max(
                    best,
                    float(SequenceMatcher(None, candidate, context[start : start + width]).ratio()),
                )
        return best

    @staticmethod
    def _asr_sentence_groups(words: list[TimedWord]) -> list[list[TimedWord]]:
        """Group timed words into short sentence-like units before speaker assignment."""

        groups: list[list[TimedWord]] = []
        current: list[TimedWord] = []
        for word in words:
            if (
                current
                and current[-1].segment_index is not None
                and word.segment_index is not None
                and current[-1].segment_index != word.segment_index
            ):
                groups.append(current)
                current = []
            current.append(word)
            if re.search(r'[.!?]["\')\]]*\s*$', str(word.text or "").strip()):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _asr_normalized_phrase_occurrences(phrase: str, context: str) -> int:
        phrase_tokens = normalize_asr_hallucination_text(phrase).split()
        context_tokens = normalize_asr_hallucination_text(context).split()
        if not phrase_tokens or len(phrase_tokens) > len(context_tokens):
            return 0
        width = len(phrase_tokens)
        return sum(
            context_tokens[index : index + width] == phrase_tokens
            for index in range(len(context_tokens) - width + 1)
        )

    def _filter_repeated_asr_hallucinations(
        self,
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
        words: list[TimedWord],
        *,
        media_start_seconds: float,
        media_duration_seconds: float | None,
    ) -> list[TimedWord]:
        """Cross-check runs of three or more identical short sentences once as a unit."""

        units = self._asr_sentence_groups(words)
        runs: list[list[list[TimedWord]]] = []
        index = 0
        while index < len(units):
            normalized = normalize_asr_hallucination_text(self._asr_group_text(units[index]))
            token_count = len(normalized.split())
            end = index + 1
            while (
                normalized
                and 1 <= token_count <= 8
                and end < len(units)
                and normalize_asr_hallucination_text(self._asr_group_text(units[end])) == normalized
                and float(units[end][0].start) - float(units[end - 1][-1].end) <= 2.0
                and float(units[end][-1].end) - float(units[index][0].start) <= 10.0
            ):
                end += 1
            if end - index >= 3:
                runs.append(units[index:end])
            index = max(index + 1, end)

        if not runs:
            return words

        removed_ids: set[int] = set()
        for run in runs:
            run_words = [word for unit in run for word in unit]
            run_text = self._asr_group_text(run[0])
            run_start = min(float(word.start) for word in run_words)
            run_end = max(float(word.end) for word in run_words)
            independent_text, independent_details = self._independent_asr_context_text(
                model,
                audio,
                sample_rate,
                group_start=run_start,
                group_end=run_end,
                media_start_seconds=media_start_seconds,
                media_duration_seconds=media_duration_seconds,
            )
            if independent_text is None:
                self._mark_asr_words_for_review(
                    run_words,
                    reason="repetition-shaped ASR output could not be independently verified",
                    details={
                        "policy_rule": "repeated_short_phrase",
                        "repetition_count": len(run),
                        **independent_details,
                    },
                )
                continue

            independent_count = self._asr_normalized_phrase_occurrences(
                run_text,
                independent_text,
            )
            keep_count = min(len(run), independent_count)
            if keep_count >= len(run):
                continue
            for unit in run[keep_count:]:
                removed_ids.update(id(word) for word in unit)
                candidate_start = min(float(word.start) for word in unit)
                candidate_end = max(float(word.end) for word in unit)
                self._record_suppressed_asr_candidate(
                    {
                        "reason": "repeated short phrase was not supported by independent ASR",
                        "text": self._asr_group_text(unit),
                        "start": round(media_start_seconds + candidate_start, 4),
                        "end": round(media_start_seconds + candidate_end, 4),
                        "policy_rule": "repeated_short_phrase",
                        "repetition_count": len(run),
                        "independent_repetition_count": independent_count,
                        **independent_details,
                    }
                )
            bus = getattr(self, "bus", None)
            if bus is not None:
                bus.emit(
                    "status",
                    {
                        "message": (
                            f"ASR repetition check retained {keep_count} of {len(run)} "
                            f"instance(s) of {run_text!r} after independent verification."
                        )
                    },
                )

        return [word for word in words if id(word) not in removed_ids]

    def _filter_implausible_asr_word_timings(
        self,
        words: list[TimedWord],
        *,
        media_start_seconds: float,
    ) -> list[TimedWord]:
        """Reject sentence fragments whose word timestamps cannot represent speech."""

        removed_ids: set[int] = set()
        for group in self._asr_sentence_groups(words):
            content_words = [
                word
                for word in group
                if normalize_asr_hallucination_text(str(word.text or ""))
            ]
            if not content_words:
                continue
            start = min(float(word.start) for word in content_words)
            end = max(float(word.end) for word in content_words)
            span = max(0.0, end - start)
            token_count = sum(
                len(normalize_asr_hallucination_text(str(word.text or "")).split())
                for word in content_words
            )
            zero_duration_count = sum(
                float(word.end) <= float(word.start)
                for word in content_words
            )
            impossible = (
                span <= 0.0
                or (
                    token_count >= 2
                    and span < 0.04 * token_count
                )
                or (
                    zero_duration_count >= 2
                    and span < 0.25
                )
            )
            if not impossible:
                continue
            removed_ids.update(id(word) for word in group)
            self._record_suppressed_asr_candidate(
                {
                    "reason": "ASR word timestamps were physically implausible",
                    "text": self._asr_group_text(group),
                    "start": round(media_start_seconds + start, 4),
                    "end": round(media_start_seconds + end, 4),
                    "policy_rule": "implausible_word_timing",
                    "word_count": token_count,
                    "zero_duration_word_count": zero_duration_count,
                }
            )

        if removed_ids:
            bus = getattr(self, "bus", None)
            if bus is not None:
                bus.emit(
                    "status",
                    {
                        "message": (
                            "ASR review: suppressed physically implausible "
                            "word timestamp output."
                        )
                    },
                )
        return [word for word in words if id(word) not in removed_ids]

    def _independent_asr_context_text(
        self,
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
        *,
        group_start: float,
        group_end: float,
        media_start_seconds: float,
        media_duration_seconds: float | None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Ask the configured non-Whisper preview model whether contextual speech is present."""

        details: dict[str, Any] = {"independent_verifier": "realtime_preview_asr"}
        if not bool(getattr(self.args, "asr_independent_verification", True)):
            details["independent_unavailable_reason"] = "disabled"
            return None, details
        if isinstance(model, CpuHybridTranscriber):
            details["independent_unavailable_reason"] = "same_model_as_primary"
            return None, details
        transcriber = getattr(self, "_preview_transcriber", None)
        verify = getattr(transcriber, "transcribe_verification", None)
        if not callable(verify):
            details["independent_unavailable_reason"] = "preview_asr_unavailable"
            return None, details
        engine = normalize_preview_engine(getattr(self.args, "realtime_preview_engine", ""))
        if engine in {"off", "mock"}:
            details["independent_unavailable_reason"] = f"non_independent_engine:{engine}"
            return None, details

        radius = max(
            0.0,
            float(getattr(self.args, "asr_independent_verification_context_seconds", 2.5)),
        )
        absolute_start = max(0.0, float(media_start_seconds) + float(group_start) - radius)
        absolute_end = float(media_start_seconds) + float(group_end) + radius
        if media_duration_seconds is not None:
            absolute_end = min(float(media_duration_seconds), absolute_end)
        absolute_end = max(absolute_start, absolute_end)
        details.update(
            {
                "independent_context_start": round(absolute_start, 4),
                "independent_context_end": round(absolute_end, 4),
            }
        )

        verification_audio = np.empty(0, dtype=np.float32)
        verification_rate = int(sample_rate)
        copy_window = getattr(self, "_audio_window_copy", None)
        if callable(copy_window):
            try:
                verification_audio, verification_rate = copy_window(absolute_start, absolute_end)
            except Exception:
                verification_audio = np.empty(0, dtype=np.float32)
        if verification_audio.size <= 0 and sample_rate > 0:
            local_start = max(0.0, absolute_start - float(media_start_seconds))
            local_end = max(local_start, absolute_end - float(media_start_seconds))
            start_sample = max(0, min(audio.size, int(round(local_start * sample_rate))))
            end_sample = max(start_sample, min(audio.size, int(round(local_end * sample_rate))))
            verification_audio = np.ascontiguousarray(audio[start_sample:end_sample])
            verification_rate = int(sample_rate)
        if verification_audio.size <= 0 or verification_rate <= 0:
            details["independent_unavailable_reason"] = "context_audio_unavailable"
            return None, details

        started = time.monotonic()
        try:
            text = " ".join(str(verify(verification_audio, verification_rate) or "").split())
        except Exception as exc:
            details["independent_unavailable_reason"] = f"verification_failed:{type(exc).__name__}"
            return None, details
        details["independent_latency_ms"] = round((time.monotonic() - started) * 1000.0, 2)
        details["independent_text"] = text
        return text, details

    def _verify_low_evidence_asr_words(
        self,
        model: Any,
        audio: np.ndarray,
        sample_rate: int,
        words: list[TimedWord],
        *,
        media_start_seconds: float = 0.0,
        media_duration_seconds: float | None = None,
    ) -> list[TimedWord]:
        """Require cross-model acoustic disagreement before suppressing doubtful text."""

        words = self._filter_implausible_asr_word_timings(
            words,
            media_start_seconds=media_start_seconds,
        )
        if not bool(getattr(self.args, "asr_hallucination_verification", True)):
            return words
        words = self._filter_repeated_asr_hallucinations(
            model,
            audio,
            sample_rate,
            words,
            media_start_seconds=media_start_seconds,
            media_duration_seconds=media_duration_seconds,
        )
        suspicion_threshold = max(
            0.0,
            min(1.0, float(getattr(self.args, "asr_hallucination_suspicion_score", 0.45))),
        )
        independent_evidence_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.args,
                        "asr_independent_verification_max_primary_evidence",
                        0.70,
                    )
                ),
            ),
        )
        independent_no_speech_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.args,
                        "asr_independent_verification_min_no_speech_probability",
                        0.20,
                    )
                ),
            ),
        )
        independent_min_text_similarity = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.args,
                        "asr_independent_verification_min_text_similarity",
                        0.35,
                    )
                ),
            ),
        )
        audio_duration = float(audio.size) / float(sample_rate) if sample_rate > 0 else 0.0
        kept: list[TimedWord] = []
        suppressed: list[dict[str, Any]] = []
        retained_uncertain = 0
        verifier_failures = 0

        for group in self._asr_segment_groups(words):
            group_start = max(0.0, min(float(word.start) for word in group))
            group_end_without_context = min(audio_duration, max(float(word.end) for word in group))
            group_text = self._asr_group_text(group)
            policy = match_asr_hallucination_policy(
                group_text,
                base_suspicion_threshold=suspicion_threshold,
                segment_start_seconds=max(0.0, media_start_seconds + group_start),
                media_duration_seconds=media_duration_seconds,
            )
            required_score = policy.suspicion_threshold if policy is not None else suspicion_threshold
            score = self._asr_segment_evidence_score(group)
            high_risk_policy = policy is not None and policy.risk_score >= 70
            if high_risk_policy and score is None:
                self._mark_asr_words_for_review(
                    group,
                    reason="ASR acoustic evidence unavailable; text retained",
                    details={"policy_rule": policy.rule_id},
                )
                verifier_failures += 1
                kept.extend(group)
                continue
            has_conflicting_evidence = any(
                bool(getattr(word, "asr_review", {}).get("needs_review"))
                for word in group
                if isinstance(getattr(word, "asr_review", None), dict)
            )
            no_speech_values = [
                float(word.no_speech_prob)
                for word in group
                if word.no_speech_prob is not None
            ]
            primary_no_speech = max(no_speech_values) if no_speech_values else None
            primary_acoustic_conflict = (
                score is not None
                and score < independent_evidence_threshold
                and primary_no_speech is not None
                and primary_no_speech >= independent_no_speech_threshold
            )
            policy_requires_check = (
                high_risk_policy
                and score is not None
                and score < required_score
            )
            if not (primary_acoustic_conflict or has_conflicting_evidence or policy_requires_check):
                kept.extend(group)
                continue
            if high_risk_policy and score is not None and score >= required_score:
                kept.extend(group)
                continue

            independent_text, independent_details = self._independent_asr_context_text(
                model,
                audio,
                sample_rate,
                group_start=group_start,
                group_end=group_end_without_context,
                media_start_seconds=media_start_seconds,
                media_duration_seconds=media_duration_seconds,
            )
            decision_details = {
                "policy_rule": policy.rule_id if policy is not None else "generic",
                "evidence_score": round(float(score), 4) if score is not None else None,
                "primary_no_speech_probability": (
                    round(float(primary_no_speech), 4)
                    if primary_no_speech is not None
                    else None
                ),
                **independent_details,
                # Kept for schema compatibility with existing review consumers.
                "shifted_text_stability": None,
                "shifted_evidence_score": None,
            }
            if independent_text is None:
                self._mark_asr_words_for_review(
                    group,
                    reason="independent ASR verification unavailable; text retained",
                    details=decision_details,
                )
                verifier_failures += 1
                kept.extend(group)
                continue

            context_similarity = self._asr_context_text_similarity(group_text, independent_text)
            decision_details["independent_text_similarity"] = round(context_similarity, 4)
            if independent_text:
                if context_similarity < independent_min_text_similarity:
                    token_count = len(normalize_asr_hallucination_text(group_text).split())
                    unsupported_short_fragment = (
                        primary_acoustic_conflict
                        and context_similarity == 0.0
                        and token_count <= 1
                        and score is not None
                        and score < suspicion_threshold
                    )
                    if unsupported_short_fragment:
                        suppressed.append(
                            {
                                **decision_details,
                                "reason": (
                                    "weak short ASR fragment was contradicted "
                                    "by independent ASR"
                                ),
                                "text": group_text,
                                "start": round(float(media_start_seconds + group_start), 4),
                                "end": round(
                                    float(media_start_seconds + group_end_without_context),
                                    4,
                                ),
                            }
                        )
                        continue
                    self._mark_asr_words_for_review(
                        group,
                        reason="independent ASR heard speech but disagreed with the retained text",
                        details=decision_details,
                    )
                    retained_uncertain += 1
                kept.extend(group)
                continue

            if not primary_acoustic_conflict:
                self._mark_asr_words_for_review(
                    group,
                    reason="independent ASR found no contextual text, but primary evidence was not weak enough to suppress",
                    details=decision_details,
                )
                retained_uncertain += 1
                kept.extend(group)
                continue
            suppressed.append(
                {
                    **decision_details,
                    "text": group_text,
                    "start": round(float(media_start_seconds + group_start), 4),
                    "end": round(float(media_start_seconds + group_end_without_context), 4),
                }
            )

        reconciled_count = self._reconcile_suppressed_asr_candidates(
            kept,
            media_start_seconds=media_start_seconds,
        )
        for candidate in suppressed:
            self._record_suppressed_asr_candidate(candidate)
        bus = getattr(self, "bus", None)
        if bus is not None:
            if reconciled_count:
                bus.emit(
                    "status",
                    {
                        "message": (
                            f"ASR review: {reconciled_count} earlier suppression warning(s) "
                            "were cleared because a later acoustic view retained the text."
                        )
                    },
                )
            if suppressed:
                preview = suppressed[0]
                independent_summary = (
                    f"independent similarity={preview.get('independent_text_similarity')}"
                    if preview.get("independent_text")
                    else "independent contextual transcript was empty"
                )
                bus.emit(
                    "status",
                    {
                        "message": (
                            f"ASR review: suppressed {len(suppressed)} independently unconfirmed "
                            f"weak-acoustic text candidate(s); first at {preview['start']:.2f}-"
                            f"{preview['end']:.2f}s was {preview['text']!r} "
                            f"(primary no-speech={preview['primary_no_speech_probability']:.2f}, "
                            f"{independent_summary})."
                        )
                    },
                )
            if retained_uncertain:
                bus.emit(
                    "status",
                    {
                        "message": (
                            f"ASR review retained {retained_uncertain} cross-model disagreement "
                            "segment(s); disagreement is not sufficient evidence to delete text."
                        )
                    },
                )
            if verifier_failures:
                bus.emit(
                    "status",
                    {
                        "message": (
                            f"ASR review retained {verifier_failures} segment(s) because the independent "
                            "verification view was unavailable or failed."
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
        *,
        batched: bool = False,
        batch_size: int = 16,
    ) -> tuple[list[TimedWord], int]:
        spans = speech_spans if speech_spans is not None else [(left, right)]
        if batched and speech_spans is None and left <= 0.0 and right >= float(self.duration):
            snapshot = self._audio_timeline.snapshot(copy_audio=False)
            relative_words, segment_count = self._transcribe_audio_words(
                model,
                snapshot.audio,
                snapshot.sample_rate,
                batched=True,
                batch_size=batch_size,
            )
            relative_words = self._verify_low_evidence_asr_words(
                model,
                snapshot.audio,
                snapshot.sample_rate,
                relative_words,
                media_start_seconds=0.0,
                media_duration_seconds=float(self.duration),
            )
            return relative_words, segment_count
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
            if batched:
                relative_words, relative_segment_count = self._transcribe_audio_words(
                    model,
                    window,
                    sample_rate,
                    batched=True,
                    batch_size=batch_size,
                )
            else:
                relative_words, relative_segment_count = self._transcribe_audio_words(
                    model,
                    window,
                    sample_rate,
                )
            relative_words = self._verify_low_evidence_asr_words(
                model,
                window,
                sample_rate,
                relative_words,
                media_start_seconds=span_left,
                media_duration_seconds=float(self.duration),
            )
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
                        asr_review=dict(word.asr_review or {}),
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

    def _embed_audio_chunk_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        suffix: str,
    ) -> EmbeddingResult:
        """Return the final vector and remote weighted-stack components when available."""

        if isinstance(self.embedding, RemoteEmbeddingClient) and not self.args.keep_segment_audio:
            if np.any(np.asarray(audio, dtype=np.float32)):
                audio, sample_rate = self._enhance_audio(audio, sample_rate, path="embeddings")
            return self.embedding.embed_audio_result(audio, sample_rate)
        return EmbeddingResult(embedding=self._embed_audio_chunk(audio, sample_rate, suffix))

    def _reusable_live_embedding_from_final_result(
        self,
        result: EmbeddingResult,
    ) -> np.ndarray | None:
        """Return a final-stack component only when it exactly matches live PCM semantics."""

        if not getattr(self, "_live_embedding_separate", False):
            return None
        if bool(getattr(self.args, "enhance_embeddings", False)):
            return None
        if bool(getattr(self.args, "keep_segment_audio", False)):
            return None
        if type(self.embedding) is not RemoteEmbeddingClient:
            return None
        if type(self.live_embedding) is not RemoteEmbeddingClient:
            return None
        if self.embedding.base_url != self.live_embedding.base_url:
            return None
        if self.embedding.device != self.live_embedding.device:
            return None
        live_specs = parse_embedding_provider_stack_specs(self.live_embedding.provider)
        if len(live_specs) != 1 or live_specs[0][1] <= 0.0:
            return None
        live_provider = live_specs[0][0]
        final_matches = [
            weight
            for provider, weight in parse_embedding_provider_stack_specs(self.embedding.provider)
            if provider == live_provider and weight > 0.0
        ]
        component_matches = [
            component
            for component in result.components
            if component.provider == live_provider and component.weight > 0.0
        ]
        if len(final_matches) != 1 or len(component_matches) != 1:
            return None
        if not math.isclose(
            float(component_matches[0].weight),
            float(final_matches[0]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            return None
        return component_matches[0].embedding

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
