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
from window.live_profile_tape import emit_live_profile_snapshot




class WindowTranscriptionMixin:
    def _run_realtime_preview(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or self._stop
        transcriber = self._preview_transcriber
        if transcriber is None:
            return
        interval_seconds = max(0.05, float(self.args.realtime_preview_interval_seconds))
        min_audio_seconds = max(0.05, float(self.args.realtime_preview_min_audio_seconds))
        min_advance = max(0.0, float(self.args.realtime_preview_min_advance_seconds))
        feed_chunk_seconds = max(0.02, float(self.args.realtime_preview_feed_chunk_seconds))
        diarize_min_advance = max(0.0, float(self.args.realtime_preview_diarize_min_advance_seconds))
        last_generation = -1
        last_right = 0.0
        last_decode_right = -1.0
        last_diarized_right = -1.0
        last_speaker_payload = self._realtime_unknown_speaker_payload()
        next_at = 0.0
        vad_gate = bool(getattr(self.args, "realtime_preview_vad_gate", True))
        gate_open = not vad_gate
        gate_left = 0.0
        gate_search_left = 0.0
        gate_search_window = max(
            2.5,
            min_audio_seconds
            + max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_pre_padding_seconds", 0.35)))
            + max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_close_silence_seconds", 1.1))),
        )
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Realtime preview started ({interval_seconds:.2f}s interval, "
                    f"min audio {min_audio_seconds:.2f}s, "
                    f"feed chunk {feed_chunk_seconds:.2f}s)."
                )
            },
        )
        while not stop_event.is_set():
            left, right, generation, paused = self._preview_snapshot()
            if paused:
                time.sleep(0.05)
                continue
            if generation != last_generation:
                last_generation = generation
                last_right = left
                last_decode_right = -1.0
                last_diarized_right = -1.0
                last_speaker_payload = self._realtime_unknown_speaker_payload()
                gate_open = not vad_gate
                gate_left = left
                gate_search_left = left
                try:
                    transcriber.reset_preview()
                except Exception as exc:
                    self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    time.sleep(interval_seconds)
                    continue
            active_left = gate_left if gate_open else gate_search_left
            if right - active_left < min_audio_seconds:
                time.sleep(0.05)
                continue
            if vad_gate and not gate_open:
                # Search every audio sample that has not already been proven to be
                # silence.  In particular, do not jump straight to the tail after
                # a slow final-ASR pass: speech may have started while that pass was
                # running, and skipping to the bounded tail cuts its first words.
                search_left = gate_search_left
                vad_state = self._vad_gate_window_state(search_left, right, force=True)
                if not vad_state.has_speech or vad_state.speech_start is None:
                    # Once the complete unseen range is known to contain no speech,
                    # retaining only a bounded tail keeps idle VAD work constant.
                    gate_search_left = max(gate_search_left, right - gate_search_window)
                    time.sleep(0.05)
                    continue
                pre_padding = max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_pre_padding_seconds", 0.35)))
                gate_left = max(gate_search_left, float(vad_state.speech_start) - pre_padding)
                gate_open = True
                last_right = gate_left
                last_decode_right = -1.0
                last_diarized_right = -1.0
                last_speaker_payload = self._realtime_unknown_speaker_payload()
                try:
                    transcriber.reset_preview()
                except Exception as exc:
                    self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    gate_open = False
                    gate_search_left = max(gate_search_left, gate_left)
                    time.sleep(interval_seconds)
                    continue
            feed_limit = right
            close_after_feed = False
            close_search_left = gate_search_left
            if vad_gate and gate_open:
                vad_state = self._vad_gate_window_state(gate_left, right, force=True)
                close_silence = max(
                    0.0,
                    float(getattr(self.args, "realtime_preview_vad_gate_close_silence_seconds", 1.1)),
                )
                if not vad_state.has_speech:
                    close_after_feed = True
                    feed_limit = last_right
                    close_search_left = max(gate_left, right - gate_search_window)
                elif vad_state.trailing_silence_seconds >= close_silence and vad_state.speech_end is not None:
                    post_padding = max(
                        0.0,
                        float(getattr(self.args, "realtime_preview_vad_gate_post_padding_seconds", 0.35)),
                    )
                    speech_end = float(vad_state.speech_end)
                    feed_limit = max(last_right, min(right, speech_end + post_padding))
                    close_after_feed = True
                    close_search_left = max(gate_left, speech_end + post_padding)
            if feed_limit < last_right + feed_chunk_seconds and not close_after_feed:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now < next_at and not close_after_feed:
                time.sleep(min(0.05, next_at - now))
                continue
            try:
                text = ""
                while feed_limit >= last_right + feed_chunk_seconds:
                    feed_right = last_right + feed_chunk_seconds
                    audio, sample_rate = self._audio_window_copy(last_right, feed_right)
                    last_right = feed_right
                    if audio.size <= 0:
                        continue
                    text = " ".join(transcriber.accept_preview_audio(audio, sample_rate).split())
                if close_after_feed:
                    try:
                        transcriber.reset_preview()
                    except Exception as exc:
                        self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    gate_open = False
                    gate_search_left = max(close_search_left, last_right)
                    gate_left = gate_search_left
                    last_right = gate_search_left
                    last_decode_right = -1.0
                    last_diarized_right = -1.0
                    last_speaker_payload = self._realtime_unknown_speaker_payload()
                    self.bus.emit("realtime_clear", {"generation": generation, "reason": "preview_vad_gate_closed"})
                    continue
            except Exception as exc:
                self.bus.emit("status", {"message": f"Realtime preview error: {type(exc).__name__}: {exc}"})
                time.sleep(interval_seconds)
                continue
            preview_right = last_right
            should_emit = last_decode_right < 0.0 or preview_right >= last_decode_right + min_advance
            if not should_emit:
                continue
            last_decode_right = preview_right
            next_at = time.monotonic() + interval_seconds
            if not text or not re.search(r"[A-Za-z0-9]", text):
                continue
            text = self._format_realtime_preview_text(text, gate_left)
            if not self._preview_generation_is_current(generation, left):
                continue
            duration_seconds = max(0.0, preview_right - gate_left)
            if self._live_speaker_assignment_enabled() and duration_seconds >= self.args.realtime_preview_diarize_min_audio_seconds and (
                last_diarized_right < 0.0 or preview_right >= last_diarized_right + diarize_min_advance
            ):
                if self.memory.profile_count() > 0 and self._try_reserve_live_speaker_embedding():
                    audio, _sample_rate = self._audio_window_copy(gate_left, preview_right)
                    last_speaker_payload = self._score_realtime_preview_speaker(audio, duration_seconds)
                    last_diarized_right = preview_right
                    if not self._preview_generation_is_current(generation, left):
                        continue
            self.bus.emit("realtime", {
                "index": f"rt-{generation}",
                "realtime": True,
                "realtime_generation": generation,
                "text": text,
                "start": round(gate_left, 4),
                "end": round(preview_right, 4),
                "audio_length_seconds": round(float(max(0.0, preview_right - gate_left)), 4),
                "pending": False,
                **last_speaker_payload,
            })

    def _run(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or self._stop
        model = self._model
        if model is None:
            self.bus.emit("status", {"message": "No ASR backend loaded."})
            return
        try:
            left = 0.0
            index = 0
            last_transcribed_right = -1.0
            last_vad_flush_right = -1.0
            previous_emitted_sentence_ended_strong = True
            interval_seconds = max(0.0, float(self.args.interval_seconds))
            min_playback_advance = max(0.0, float(self.args.min_playback_advance_seconds))
            final_flush_epsilon = max(0.0, float(self.args.final_flush_epsilon_seconds))
            next_tick = time.monotonic() + interval_seconds if interval_seconds > 0.0 else 0.0
            mode = "continuous" if interval_seconds <= 0.0 else f"{interval_seconds:.2f}s interval"
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Growing-window transcription started ({mode}, "
                        f"min playback advance {min_playback_advance:.2f}s)."
                    )
                },
            )
            while not stop_event.is_set():
                now = time.monotonic()
                duration = self.duration
                if not self._streaming_audio and left >= duration:
                    break
                right = self.playback_time()
                media_final_flush = (not self._streaming_audio) and right >= duration - final_flush_epsilon
                if media_final_flush:
                    right = duration

                vad_state = self._vad_window_state(left, right)
                asr_vad_state = vad_state
                if self._asr_vad_gate_enabled():
                    if getattr(self.args, "vad_sentence_splitting", True):
                        asr_vad_state = self._vad_gate_window_state(
                            left,
                            right,
                            primary_state=vad_state,
                        )
                    else:
                        asr_vad_state = self._vad_gate_window_state(left, right, force=True)
                vad_flush = vad_state.should_flush and not media_final_flush
                if (
                    getattr(self.args, "vad_sentence_splitting", True)
                    and not media_final_flush
                    and not vad_state.has_speech
                ):
                    time.sleep(0.05)
                    continue

                transcript_final_flush = media_final_flush or vad_flush
                if vad_flush and right <= last_vad_flush_right + min_playback_advance:
                    time.sleep(0.05)
                    continue
                if right - left < max(1.0, self.args.min_window_seconds) and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if right <= last_transcribed_right + min_playback_advance and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if interval_seconds > 0.0 and now < next_tick and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if interval_seconds > 0.0:
                    next_tick = now + interval_seconds
                last_transcribed_right = right
                final_note = " final" if media_final_flush else (" vad-final" if vad_flush else "")
                transcribe_right = right
                vad_next_left: float | None = None
                if vad_flush:
                    vad_label = "RMS VAD" if vad_state.backend == "rms" else "Silero VAD"
                    speech_end = float(vad_state.speech_end or right)
                    transcribe_right = max(
                        left,
                        min(
                            right,
                            speech_end + max(0.0, float(self.args.vad_final_window_post_silence_seconds)),
                        ),
                    )
                    vad_next_left = max(
                        left,
                        min(
                            duration,
                            speech_end + max(0.0, float(self.args.vad_next_window_start_silence_seconds)),
                        ),
                    )
                    self.bus.emit(
                        "status",
                        {
                            "message": (
                                f"{vad_label} silence split at {vad_state.speech_end:.2f}s "
                                f"after {vad_state.trailing_silence_seconds:.2f}s silence; "
                                f"final window right={transcribe_right:.2f}s next left={vad_next_left:.2f}s."
                            )
                        },
                    )
                    if bool(getattr(self.args, "live_speaker_clear_on_vad_split", False)):
                        self.bus.emit(
                            "live_speaker_clear",
                            {
                                "live": False,
                                "fallback": True,
                                "start": round(float(speech_end), 4),
                                "end": round(float(right), 4),
                                "reason": "vad_silence_split",
                                "assignment_source": "main_vad_silence_split_clear",
                            },
                        )
                self.bus.emit("status", {"message": f"Transcribing{final_note} window left={left:.2f}s right={transcribe_right:.2f}s."})
                speech_spans: list[tuple[float, float]] | None = None
                if self._asr_vad_gate_enabled():
                    speech_spans = self._asr_vad_gate_spans(left, transcribe_right, asr_vad_state)
                    if not speech_spans:
                        self.bus.emit(
                            "status",
                            {
                                "message": (
                                    f"ASR VAD gate skipped non-speech window "
                                    f"left={left:.2f}s right={transcribe_right:.2f}s."
                                )
                            },
                        )
                        if vad_next_left is not None:
                            left = max(left, vad_next_left)
                            self._advance_realtime_preview_after_commit(left)
                        if media_final_flush:
                            break
                        time.sleep(0.05)
                        continue
                    kept_seconds = sum(max(0.0, span_right - span_left) for span_left, span_right in speech_spans)
                    if len(speech_spans) > 1 or kept_seconds < max(0.0, transcribe_right - left) - 0.05:
                        self.bus.emit(
                            "status",
                            {
                                "message": (
                                    f"ASR VAD gate kept {kept_seconds:.2f}s across "
                                    f"{len(speech_spans)} speech clip(s) from "
                                    f"{transcribe_right - left:.2f}s window."
                                )
                            },
                        )
                transcribe_started = time.monotonic()
                transcript = self._transcribe_window(
                    model,
                    left,
                    transcribe_right,
                    final_flush=transcript_final_flush,
                    previous_text_ended_sentence=previous_emitted_sentence_ended_strong,
                    speech_spans=speech_spans,
                )
                transcribe_seconds = time.monotonic() - transcribe_started
                if vad_flush:
                    last_vad_flush_right = right
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Transcribed {transcribe_right - left:.2f}s window in {transcribe_seconds:.2f}s; "
                            f"segments={transcript.segment_count} words={transcript.word_count} "
                            f"accepted={len(transcript.sentences)}."
                        )
                    },
                )
                for sentence in transcript.sentences:
                    self._emit_sentence(index, sentence, left, transcribe_right)
                    previous_emitted_sentence_ended_strong = text_ends_sentence(sentence.text)
                    self._last_final_sentence_ended_strong = previous_emitted_sentence_ended_strong
                    self._final_sentence_count = int(getattr(self, "_final_sentence_count", 0)) + 1
                    index += 1
                    left = max(left, sentence.next_left)
                if vad_next_left is not None:
                    left = max(left, vad_next_left)
                if transcript.sentences or vad_next_left is not None:
                    self._advance_realtime_preview_after_commit(left)
                    if interval_seconds > 0.0 and not media_final_flush:
                        next_tick = time.monotonic() + interval_seconds
                self.bus.emit("status", {"message": f"Window left={left:.2f}s right={right:.2f}s sentences={len(transcript.sentences)}."})
                if media_final_flush:
                    break
        except Exception as exc:
            self.bus.emit("status", {"message": f"Window diarization error: {type(exc).__name__}: {exc}"})
            raise
        finally:
            self._pause_realtime_preview()
            self._drain_embedding_jobs()
            self._revisit_unknown_sentences()
            self._finalize_speaker_refinement()
            self._drain_live_memory_update_jobs()

    def _transcribe_window(
        self,
        model: Any,
        left: float,
        right: float,
        final_flush: bool = False,
        previous_text_ended_sentence: bool = False,
        speech_spans: list[tuple[float, float]] | None = None,
        *,
        batched: bool = False,
        batch_size: int = 16,
    ) -> WindowTranscript:
        if batched:
            words, segment_count = self._transcribe_window_audio_words(
                model,
                left,
                right,
                speech_spans,
                batched=True,
                batch_size=batch_size,
            )
        else:
            words, segment_count = self._transcribe_window_audio_words(model, left, right, speech_spans)
        words.sort(key=lambda item: (item.start, item.end))
        parts = split_words_with_stream2sentence(
            words,
            left=left,
            right=right,
            unstable_tail_seconds=self.args.unstable_tail_seconds,
            final_flush=final_flush,
            previous_text_ended_sentence=previous_text_ended_sentence,
            boundary_pre_padding_seconds=self.args.sentence_boundary_pre_padding_seconds,
            boundary_post_padding_seconds=self.args.sentence_boundary_post_padding_seconds,
            boundary_gap_ratio=self.args.sentence_boundary_gap_ratio,
            sentence_tokenizer=getattr(self.args, "sentence_tokenizer", "nltk+rule-based"),
            sentence_language=getattr(self.args, "sentence_language", getattr(self.args, "language", "en")),
        )
        return WindowTranscript(parts, len(words), segment_count)

    @staticmethod
    def _base_payload_from_sentence_part(
        index: int,
        sentence: SentencePart,
        window_left: float,
        window_right: float,
    ) -> dict[str, Any]:
        source_text_hash = hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()
        return {
            "index": index,
            "text": sentence.text,
            "source_text_hash": source_text_hash,
            "source_revision": source_text_hash,
            "start": round(sentence.start, 4),
            "end": round(sentence.end, 4),
            "spoken_word_seconds": round(float(sentence.spoken_word_seconds), 4),
            "audio_length_seconds": round(float(max(0.0, sentence.end - sentence.start)), 4),
            "speech_audio_ratio": round(float(sentence.speech_audio_ratio), 4),
            "new_speaker_anchor_words": len(text_content_words(sentence.text)),
            "window_left": round(window_left, 4),
            "window_right": round(window_right, 4),
            "next_left": round(sentence.next_left, 4),
            "words": sentence.words,
            "first_word_start": round_optional(sentence.first_word_start),
            "last_word_end": round_optional(sentence.last_word_end),
            "next_word_start": round_optional(sentence.next_word_start),
            "gap_to_next_word_seconds": round_optional(sentence.gap_to_next_word_seconds),
            "boundary_strategy": sentence.boundary_strategy,
            "sentence_boundary_pre_padding_seconds": round(float(sentence.sentence_boundary_pre_padding_seconds), 4),
            "sentence_boundary_post_padding_seconds": round(float(sentence.sentence_boundary_post_padding_seconds), 4),
            "sentence_boundary_gap_ratio": round(float(sentence.sentence_boundary_gap_ratio), 4),
            "unknown_permanent": False,
        }

    def _emit_sentence(self, index: int, sentence: SentencePart, window_left: float, window_right: float) -> None:
        base_payload = self._base_payload_from_sentence_part(index, sentence, window_left, window_right)
        self.bus.emit("sentence", {
            **base_payload,
            "pending": True,
            "assigned_speaker": None,
            **self._speaker_info_for_payload(None),
            "created_speaker": False,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "top_similarity": None,
            "margin": None,
        })
        if sentence.speech_audio_ratio < self.args.min_speech_audio_ratio:
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Marking sentence {index} permanently unknown: "
                        f"speech/audio ratio {sentence.speech_audio_ratio:.2f} "
                        f"below {self.args.min_speech_audio_ratio:.2f}."
                    )
                },
            )
            self._emit_transcript_sentence({
                **base_payload,
                "pending": False,
                "unknown_permanent": True,
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "quality": None,
                "assignment_source": "unknown_permanent",
            })
            return
        if not is_embedding_candidate_text(sentence.text):
            self.bus.emit("status", {"message": f"Skipping non-speech/vocable sentence {index}: {sentence.text[:72]}"})
            payload = {
                **base_payload,
                "pending": False,
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "assignment_source": "non_embedding_candidate",
            }
            self._emit_transcript_sentence(payload)
            self._record_unknown_refinement_candidate(
                index,
                base_payload,
                max(0.0, sentence.end - sentence.start),
                payload,
            )
            return
        duration_seconds = max(0.0, sentence.end - sentence.start)
        audio, sample_rate = self._audio_window_copy(sentence.start, sentence.end)
        job = EmbeddingSentenceJob(
            index=index,
            base_payload=base_payload,
            text=sentence.text,
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            speaker_generation=self._speaker_generation,
            run_id=str(getattr(getattr(self, "_active_run", None), "run_id", "")),
        )
        jobs = self._embedding_jobs
        if jobs is None:
            self._process_sentence_embedding(job)
            return
        jobs.put(job)
        self.bus.emit("status", {"message": f"Queued speaker embedding for sentence {index}: {sentence.text[:72]}"})

    def _maybe_emit_sentence_live_speaker_hint(
        self,
        sentence_payload: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        if not self._live_speaker_assignment_enabled():
            return
        if not bool(getattr(self.args, "live_speaker_sentence_hint", True)):
            return
        try:
            min_duration = max(
                0.0,
                float(getattr(self.args, "live_speaker_sentence_hint_min_duration_seconds", 0.0)),
            )
        except (TypeError, ValueError):
            min_duration = 0.0
        if duration_seconds < min_duration:
            return
        speaker_id = str(sentence_payload.get("assigned_speaker") or "")
        if not speaker_id or speaker_id == "UNKNOWN":
            return
        try:
            end = float(sentence_payload.get("end") or 0.0)
            playback_time = float(self.playback_time())
        except (TypeError, ValueError):
            return
        try:
            max_lag = max(0.0, float(getattr(self.args, "live_speaker_sentence_hint_max_lag_seconds", 1.25)))
        except (TypeError, ValueError):
            max_lag = 1.25
        created_speaker = bool(sentence_payload.get("created_speaker"))
        if created_speaker:
            try:
                max_lag = max(
                    max_lag,
                    max(0.0, float(getattr(
                        self.args,
                        "live_speaker_sentence_hint_new_speaker_max_lag_seconds",
                        1.25,
                    ))),
                )
            except (TypeError, ValueError):
                max_lag = max(max_lag, 1.25)
            try:
                max_top_similarity = float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_new_speaker_max_top_similarity",
                    1.0,
                ))
            except (TypeError, ValueError):
                max_top_similarity = 1.0
            try:
                top_similarity = float(sentence_payload.get("top_similarity"))
            except (TypeError, ValueError):
                top_similarity = 1.0
            if top_similarity > max_top_similarity:
                return
        lag_seconds = playback_time - end
        if lag_seconds > max_lag:
            return
        try:
            hold_seconds = max(
                0.0,
                float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_hold_seconds",
                    getattr(self.args, "live_speaker_probe_hold_seconds", 1.0),
                )),
            )
        except (TypeError, ValueError):
            hold_seconds = 1.0
        if created_speaker:
            try:
                new_speaker_hold = float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_new_speaker_hold_seconds",
                    -1.0,
                ))
            except (TypeError, ValueError):
                new_speaker_hold = -1.0
            if new_speaker_hold >= 0.0:
                hold_seconds = max(hold_seconds, new_speaker_hold)
        if bool(getattr(self.args, "live_speaker_sentence_hint_hold_through_sentence", False)):
            hold_seconds = max(hold_seconds, max(0.0, end - playback_time) + hold_seconds)
        if hold_seconds <= 0.0:
            return
        self.bus.emit(
            "live_speaker",
            {
                **sentence_payload,
                "speaker_id": speaker_id,
                "live": True,
                "fallback": True,
                "sentence_hint": True,
                "only_if_no_live_speaker": not bool(
                    getattr(self.args, "live_speaker_sentence_hint_override", False)
                ),
                "start": sentence_payload.get("start"),
                "end": sentence_payload.get("end"),
                "audio_length_seconds": round(float(max(0.0, duration_seconds)), 4),
                "hold_seconds": round(float(hold_seconds), 4),
                "playback_time": round(float(playback_time), 4),
                "live_hint_lag_seconds": round(float(lag_seconds), 4),
                "assignment_source": "final_sentence_live_hint",
            },
        )

    def _apply_sentence_embedding_decision(
        self,
        *,
        index: int,
        base_payload: dict[str, Any],
        text: str,
        embedding: np.ndarray,
        duration_seconds: float,
        live_memory_audio: np.ndarray | None = None,
        live_memory_sample_rate: int | None = None,
        live_memory_suffix: str = ".live-sentence.wav",
        speaker_generation: int | None = None,
        emit_status: bool = True,
        elapsed_seconds: float | None = None,
        run_speaker_refinement: bool = True,
    ) -> dict[str, Any]:
        paired_unknown_revision: tuple[PendingUnknownSentence, float] | None = None
        allow_new_speaker = len(text_content_words(text)) >= self.args.min_new_speaker_words
        decision = self._section_gap_new_speaker_decision(
            embedding,
            duration_seconds,
            base_payload,
            allow_new_speaker=allow_new_speaker,
        )
        if decision is None:
            decision = self.memory.classify(
                embedding,
                duration_seconds,
                allow_new_speaker=allow_new_speaker,
            )
            pair_decision = self._unknown_pair_new_speaker_decision(
                embedding,
                duration_seconds,
                base_payload,
                decision,
                allow_new_speaker=allow_new_speaker,
            )
            if pair_decision is not None:
                decision, paired_candidate, pair_similarity = pair_decision
                paired_unknown_revision = (paired_candidate, pair_similarity)
        if emit_status:
            elapsed = 0.0 if elapsed_seconds is None else max(0.0, float(elapsed_seconds))
            self.bus.emit("status", {
                "message": (
                    f"Embedded sentence {index} in {elapsed:.2f}s; "
                    f"speaker={decision.assigned_speaker or 'UNKNOWN'} "
                    f"new={int(bool(decision.created_speaker))} "
                    f"unk={decision.unknown_probability} "
                    f"top={decision.top_similarity} "
                    f"margin={decision.margin}."
                )
            })
        self._ensure_speaker_metadata(decision.assigned_speaker)
        sentence_payload = {
            **base_payload,
            "pending": False,
            "assigned_speaker": decision.assigned_speaker,
            **self._speaker_info_for_payload(decision.assigned_speaker),
            "created_speaker": decision.created_speaker,
            "probabilities": decision.probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            "assignment_source": decision.assignment_source,
        }
        self._record_sentence_assignment(
            index,
            base_payload,
            embedding,
            duration_seconds,
            sentence_payload,
        )
        sentence_payload = self._emit_transcript_sentence(sentence_payload)
        if paired_unknown_revision is not None:
            paired_candidate, pair_similarity = paired_unknown_revision
            self._emit_unknown_pair_revision(paired_candidate, decision, pair_similarity)
        self._maybe_emit_sentence_live_speaker_hint(sentence_payload, duration_seconds)
        if live_memory_audio is not None and live_memory_sample_rate is not None:
            self._update_live_speaker_memory(
                decision.assigned_speaker,
                live_memory_audio,
                live_memory_sample_rate,
                duration_seconds,
                live_memory_suffix,
                speaker_generation=self._speaker_generation if speaker_generation is None else speaker_generation,
                sentence_start=base_payload.get("start"),
                sentence_end=base_payload.get("end"),
            )
        if decision.assigned_speaker is None:
            self._remember_unknown_sentence(index, base_payload, embedding, duration_seconds)
        elif decision.created_speaker:
            self.emit_speaker_state()
            if run_speaker_refinement:
                self._revisit_unknown_sentences()
        elif self._refresh_person_identity_suggestions(self.memory.export_profiles()):
            # An existing Speaker can cross the Person-recognition evidence gate
            # after its first state event. Notify the UI only when the public
            # identity actually changes, rather than after every sentence.
            self.emit_speaker_state()
        if run_speaker_refinement:
            self._refine_speaker_assignments()
        emit_live_profile_snapshot(
            self,
            self.memory,
            decision.assigned_speaker,
            str(getattr(self.args, "embedding_provider", "cached_embedding")),
            source="synchronous_final_sentence_memory_update",
            sentence_start=base_payload.get("start"),
            sentence_end=base_payload.get("end"),
        )
        self._maybe_checkpoint_confirmed_people()
        return sentence_payload

    def _process_sentence_embedding(self, job: EmbeddingSentenceJob) -> None:
        base_payload = job.base_payload
        index = job.index
        active_run = getattr(self, "_active_run", None)
        if job.run_id and (active_run is None or job.run_id != active_run.run_id):
            self.bus.emit("status", {"message": f"Skipped stale diarization run job for sentence {index}."})
            return
        if job.speaker_generation != self._speaker_generation:
            self.bus.emit("status", {"message": f"Skipped stale speaker embedding for sentence {index}."})
            return
        chunk = pad_audio(
            trim_silence(job.audio, job.sample_rate),
            self.args.min_embed_seconds,
            job.sample_rate,
        )
        duration_seconds = job.duration_seconds
        try:
            self.bus.emit("status", {"message": f"Embedding sentence {index}: {job.text[:72]}"})
            embed_started = time.monotonic()
            embedding = self._embed_audio_chunk(chunk, job.sample_rate, ".sentence.wav")
            active_run = getattr(self, "_active_run", None)
            if job.run_id and (active_run is None or job.run_id != active_run.run_id):
                self.bus.emit("status", {"message": f"Discarded stale diarization run job for sentence {index}."})
                return
            if job.speaker_generation != self._speaker_generation:
                self.bus.emit("status", {"message": f"Discarded stale speaker embedding for sentence {index}."})
                return
            active_run = getattr(self, "_active_run", None)
            fast_processing = active_run is not None and active_run.processing_mode == "fast"
            self._apply_sentence_embedding_decision(
                index=index,
                base_payload=base_payload,
                text=job.text,
                embedding=embedding,
                duration_seconds=duration_seconds,
                live_memory_audio=None if fast_processing else chunk,
                live_memory_sample_rate=None if fast_processing else job.sample_rate,
                live_memory_suffix=".live-sentence.wav",
                speaker_generation=job.speaker_generation,
                emit_status=True,
                elapsed_seconds=time.monotonic() - embed_started,
                run_speaker_refinement=not fast_processing,
            )
        except Exception as exc:
            self.bus.emit("status", {"message": f"Embedding failed for sentence {index}: {type(exc).__name__}: {exc}"})
            self._emit_transcript_sentence({
                **base_payload,
                "pending": False,
                "error": f"{type(exc).__name__}: {exc}",
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
            })
            return
