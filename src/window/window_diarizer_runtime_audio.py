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
from window.sherpa_onnx_models import (
    ensure_kroko_sherpa_model,
    ensure_sherpa_onnx_model,
    validate_sherpa_onnx_model_dir,
)
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
from window.live_speech_gate import rms_speech_present
from window.cpu_forced_alignment import CpuHybridTranscriber




class WindowRuntimeAudioMixin:
    def _ensure_realtime_preview_model(self) -> None:
        engine = normalize_preview_engine(getattr(self.args, "realtime_preview_engine", "off"))
        if engine in {"off", "mock"}:
            return
        model_dir = getattr(self.args, "realtime_preview_model_dir", None)
        if engine == "sherpa_onnx" or (engine == "kroko_onnx" and model_dir is not None):
            if model_dir is None:
                raise RuntimeError("Nemotron realtime preview requires a model directory.")
            try:
                validated_model_dir = validate_sherpa_onnx_model_dir(Path(model_dir))
                self._update_config(realtime_preview_model_dir=validated_model_dir)
                return
            except RuntimeError:
                if not bool(getattr(self.args, "realtime_preview_auto_download", True)):
                    raise
            preset = str(getattr(self.args, "realtime_preview_model_preset", "") or "")
            model_label = "Kroko" if engine == "kroko_onnx" else "Nemotron"
            self.bus.emit(
                "status",
                {"message": f"{model_label} preview model {preset} not found locally; downloading verified upstream archive."},
            )
            ready_model_dir = (
                ensure_kroko_sherpa_model(
                    getattr(self.args, "realtime_preview_language", getattr(self.args, "language", "en")),
                    target_dir=Path(model_dir),
                )
                if engine == "kroko_onnx"
                else ensure_sherpa_onnx_model(preset, target_dir=Path(model_dir))
            )
            self._update_config(realtime_preview_model_dir=ready_model_dir)
            self.bus.emit("status", {"message": f"{model_label} preview model ready: {self.args.realtime_preview_model_dir}."})
            return

        model_path = getattr(self.args, "realtime_preview_model_path", None)
        if model_path is not None:
            if Path(model_path).is_file():
                return
            raise RuntimeError(f"Kroko preview model path does not exist: {model_path}")

        if not bool(getattr(self.args, "realtime_preview_auto_download", DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD)):
            return

        model_name = str(getattr(self.args, "realtime_preview_model", "") or "")
        self.bus.emit(
            "status",
            {"message": f"Kroko preview model {model_name} not found locally; downloading from Hugging Face."},
        )
        model_path = download_kroko_preview_model(model_name)
        self._update_config(realtime_preview_model_path=model_path)
        self.bus.emit("status", {"message": f"Kroko preview model ready: {model_path}."})

    def _load_realtime_preview(self) -> None:
        self._preview_transcriber = None
        self._preview_transcriber_owned = False
        engine = normalize_preview_engine(self.args.realtime_preview_engine)
        if engine == "off":
            self.bus.emit("status", {"message": "Realtime preview disabled."})
            return
        started = time.monotonic()
        try:
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Loading realtime preview engine {self.args.realtime_preview_engine} "
                        f"on {self.args.realtime_preview_provider} before playback."
                    )
                },
            )
            if engine != "mock":
                self._ensure_realtime_preview_model()
            model = getattr(self, "_model", None)
            if isinstance(model, CpuHybridTranscriber):
                self._preview_transcriber = model.source
                self._preview_transcriber.reset_preview()
            else:
                self._preview_transcriber = create_realtime_preview_transcriber(self.args)
                self._preview_transcriber_owned = True
            location = getattr(self.args, "realtime_preview_model_dir", None) or getattr(
                self.args, "realtime_preview_model_path", None
            )
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Realtime preview ready in {time.monotonic() - started:.2f}s "
                        f"({engine}, {self.args.realtime_preview_model_preset}, "
                        f"{self.args.realtime_preview_language}, CPU x{self.args.realtime_preview_num_threads}"
                        f"{', ' + str(location) if location else ''})."
                    )
                },
            )
        except Exception as exc:
            self._preview_transcriber = None
            self._preview_transcriber_owned = False
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Realtime preview disabled: {type(exc).__name__}: {exc}"
                    )
                },
            )

    def _reset_realtime_preview_state(self) -> None:
        with self._preview_lock:
            self._preview_left = 0.0
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _pause_realtime_preview(self) -> None:
        with self._preview_lock:
            self._preview_paused = True
            self._preview_generation += 1
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _resume_realtime_preview(self, left: float) -> None:
        with self._preview_lock:
            self._preview_left = max(0.0, min(float(left), self.duration))
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _advance_realtime_preview_after_commit(self, committed_left: float) -> None:
        if self._preview_transcriber is None:
            return
        committed = max(0.0, min(float(committed_left), self.duration))
        overlap = max(0.0, float(self.args.realtime_preview_reset_overlap_seconds))
        reset_left = max(0.0, committed - overlap)
        skipped = max(0.0, reset_left - committed)
        preroll = max(0.0, committed - reset_left)
        with self._preview_lock:
            self._preview_left = reset_left
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit(
            "realtime_clear",
            {
                "generation": generation,
                "committed_audio_time": round(committed, 4),
                "preview_reset_left": round(reset_left, 4),
                "skipped_audio_seconds": round(skipped, 4),
                "preview_preroll_seconds": round(preroll, 4),
            },
        )

    def _preview_snapshot(self) -> tuple[float, float, int, bool]:
        with self._preview_lock:
            return (
                self._preview_left,
                self.playback_time(),
                self._preview_generation,
                self._preview_paused,
            )

    def _preview_generation_is_current(self, generation: int, left: float) -> bool:
        with self._preview_lock:
            return (
                generation == self._preview_generation
                and not self._preview_paused
                and abs(left - self._preview_left) < 0.001
            )

    def _format_realtime_preview_text(self, text: str, left: float) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return ""
        should_uppercase = (
            int(getattr(self, "_final_sentence_count", 0)) <= 0
            or bool(getattr(self, "_last_final_sentence_ended_strong", True))
            or float(left) <= 0.001
        )
        if should_uppercase:
            return sentence_initial_uppercase_after_strong_boundary(normalized)
        return normalized

    def _start_embedding_worker(self) -> None:
        self._stop_embedding_worker()
        self._embedding_jobs = queue.Queue()
        self._embedding_thread = self.dependencies.thread_factory(
            target=self._run_embedding_jobs,
            name="WindowSpeakerEmbedding",
            daemon=True,
        )
        self._embedding_thread.start()

    def _drain_embedding_jobs(self, timeout_seconds: float = 10.0) -> bool:
        jobs = getattr(self, "_embedding_jobs", None)
        if jobs is None:
            return True
        if getattr(jobs, "unfinished_tasks", 0) > 0:
            self.bus.emit("status", {"message": "Draining queued speaker embeddings."})
        drained = self._wait_for_embedding_jobs(jobs, timeout_seconds)
        if not drained:
            self.bus.emit("status", {"message": "Timed out draining queued speaker embeddings; cancelling pending jobs."})
            self._cancel_pending_embedding_jobs(jobs)
        return drained

    def _stop_embedding_worker(self) -> None:
        jobs = self._embedding_jobs
        thread = self._embedding_thread
        if jobs is not None and thread is not None and thread.is_alive():
            jobs.put(None)
            self._wait_for_embedding_jobs(jobs, 5.0)
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.bus.emit("status", {"message": "Speaker embedding worker did not stop before timeout."})
        self._embedding_jobs = None
        self._embedding_thread = None

    def _run_embedding_jobs(self) -> None:
        jobs = self._embedding_jobs
        if jobs is None:
            return
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                self._process_sentence_embedding(job)
            finally:
                jobs.task_done()

    def _start_live_memory_update_worker(self) -> None:
        self._stop_live_memory_update_worker()
        if not self._live_embedding_separate:
            self._live_memory_update_jobs = None
            self._live_memory_update_thread = None
            return
        try:
            queue_size = int(getattr(self.args, "live_speaker_memory_update_queue_size", 64))
        except (TypeError, ValueError):
            queue_size = 64
        self._live_memory_update_jobs = queue.Queue(maxsize=max(1, queue_size))
        self._live_memory_update_thread = self.dependencies.thread_factory(
            target=self._run_live_memory_update_jobs,
            name="LiveSpeakerMemoryUpdate",
            daemon=True,
        )
        self._live_memory_update_thread.start()

    def _drain_live_memory_update_jobs(self, timeout_seconds: float = 10.0) -> bool:
        jobs = getattr(self, "_live_memory_update_jobs", None)
        if jobs is None:
            return True
        if getattr(jobs, "unfinished_tasks", 0) > 0:
            self.bus.emit("status", {"message": "Draining queued live speaker profile updates."})
        drained = self._wait_for_embedding_jobs(jobs, timeout_seconds)
        if not drained:
            self.bus.emit(
                "status",
                {"message": "Timed out draining queued live speaker profile updates; cancelling pending updates."},
            )
            self._cancel_pending_embedding_jobs(jobs)
        return drained

    def _stop_live_memory_update_worker(self) -> None:
        jobs = self._live_memory_update_jobs
        thread = self._live_memory_update_thread
        if jobs is not None and thread is not None and thread.is_alive():
            try:
                jobs.put(None, timeout=1.0)
            except queue.Full:
                self._cancel_pending_embedding_jobs(jobs)
                jobs.put(None)
            self._wait_for_embedding_jobs(jobs, 5.0)
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.bus.emit("status", {"message": "Live speaker profile update worker did not stop before timeout."})
        self._live_memory_update_jobs = None
        self._live_memory_update_thread = None

    def _run_live_memory_update_jobs(self) -> None:
        jobs = self._live_memory_update_jobs
        if jobs is None:
            return
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                self._process_live_speaker_memory_update(job)
            finally:
                jobs.task_done()

    def set_playback_time(self, seconds: float, reset: bool = False) -> None:
        seconds = max(0.0, min(float(seconds), self.duration))
        if not reset:
            seconds = self._clamp_playback_time_to_wall_clock(seconds)
        with self._playback_lock:
            self._playback_time = seconds if reset else max(self._playback_time, seconds)

    def playback_time(self) -> float:
        with self._playback_lock:
            return self._playback_time

    def _clamp_playback_time_to_wall_clock(self, seconds: float) -> float:
        if self._streaming_audio or self._playback_clock_started_at is None:
            return seconds
        max_allowed = min(self.duration, max(0.0, time.monotonic() - self._playback_clock_started_at + 3.0))
        if seconds <= max_allowed + 0.25:
            return seconds
        now = time.monotonic()
        if now >= self._last_playback_jump_warning_at + 5.0:
            self._last_playback_jump_warning_at = now
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Ignored early playback jump to {seconds:.2f}s; "
                        f"limiting live window to {max_allowed:.2f}s."
                    )
                },
            )
        return max_allowed

    def _audio_window_copy(self, left: float, right: float) -> tuple[np.ndarray, int]:
        timeline = getattr(self, "_audio_timeline", None)
        if timeline is not None:
            return timeline.window(left, right)
        with self._audio_lock:
            sample_rate = int(self.sample_rate)
            start_sample = max(0, int(left * sample_rate))
            end_sample = min(self._audio_sample_count_locked(), int(right * sample_rate))
            if end_sample <= start_sample:
                return np.zeros(0, dtype=np.float32), sample_rate
            return self._audio_slice_locked(start_sample, end_sample), sample_rate

    def _audio_sample_count_locked(self) -> int:
        return self._stream_audio_samples if self._streaming_audio else len(self.audio)

    def _audio_slice_locked(self, start_sample: int, end_sample: int) -> np.ndarray:
        if end_sample <= start_sample:
            return np.zeros(0, dtype=np.float32)
        if not self._streaming_audio:
            return np.asarray(self.audio[start_sample:end_sample], dtype=np.float32).copy()

        pieces: list[np.ndarray] = []
        offset = 0
        for chunk in self._stream_audio_chunks:
            next_offset = offset + len(chunk)
            if next_offset <= start_sample:
                offset = next_offset
                continue
            if offset >= end_sample:
                break
            left = max(0, start_sample - offset)
            right = min(len(chunk), end_sample - offset)
            if right > left:
                pieces.append(chunk[left:right])
            offset = next_offset

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        if len(pieces) == 1:
            return np.asarray(pieces[0], dtype=np.float32).copy()
        return np.concatenate(pieces).astype(np.float32, copy=False)

    @staticmethod
    def _wait_for_embedding_jobs(jobs: "queue.Queue[Any]", timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while getattr(jobs, "unfinished_tasks", 0) > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    @staticmethod
    def _cancel_pending_embedding_jobs(jobs: "queue.Queue[Any]") -> None:
        while True:
            try:
                jobs.get_nowait()
            except queue.Empty:
                return
            else:
                jobs.task_done()

    def _load_silero_vad_model(self) -> Any | None:
        if self._vad_model is not None or self._vad_model_error is not None:
            return self._vad_model
        with self._vad_model_lock:
            if self._vad_model is not None or self._vad_model_error is not None:
                return self._vad_model
            try:
                realtime_root = Path(getattr(self.args, "realtime_preview_realtimestt_root", DEFAULT_REALTIMESTT_ROOT))
                if realtime_root.exists() and str(realtime_root) not in sys.path:
                    sys.path.insert(0, str(realtime_root))
                from RealtimeSTT.core.silero_vad import create_silero_vad_model

                model_path = getattr(self.args, "vad_silero_onnx_model_path", None)
                if model_path is not None:
                    model_path = Path(model_path)
                    if not model_path.exists():
                        raise FileNotFoundError(f"Silero ONNX model not found: {model_path}")
                backend = str(getattr(self.args, "vad_silero_backend", "auto") or "auto")
                if backend == "auto" and model_path is not None:
                    backend = default_silero_vad_backend(model_path)
                self._vad_model = create_silero_vad_model(
                    backend=backend,
                    onnx_model_path=str(model_path) if model_path is not None else None,
                    onnx_threads=max(1, int(getattr(self.args, "vad_silero_onnx_threads", 2))),
                    sample_rate=SILERO_VAD_SAMPLE_RATE,
                    chunk_samples=SILERO_VAD_CHUNK_SAMPLES,
                )
                self._vad_model_backend = str(getattr(self._vad_model, "backend", backend))
                loaded_path = getattr(self._vad_model, "model_path", model_path)
                loaded_name = Path(loaded_path).name if loaded_path is not None else "auto"
                self.bus.emit(
                    "status",
                    {"message": f"Silero ONNX VAD ready ({self._vad_model_backend}, {loaded_name})."},
                )
            except Exception as exc:
                self._vad_model_error = str(exc)
                self.bus.emit(
                    "status",
                    {"message": f"Silero ONNX VAD unavailable; falling back to RMS VAD: {exc}"},
                )
        return self._vad_model

    def _resample_for_silero_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if int(sample_rate) == SILERO_VAD_SAMPLE_RATE:
            return np.asarray(audio, dtype=np.float32)
        if audio.size <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32)
        duration = audio.size / float(sample_rate)
        target_size = max(1, int(round(duration * SILERO_VAD_SAMPLE_RATE)))
        source_times = np.arange(audio.size, dtype=np.float64) / float(sample_rate)
        target_times = np.arange(target_size, dtype=np.float64) / float(SILERO_VAD_SAMPLE_RATE)
        return np.interp(target_times, source_times, audio).astype(np.float32)

    def _smooth_vad_flags(self, flags: list[bool], frame_seconds: float) -> None:
        max_gap_frames = max(0, int(round(float(self.args.vad_merge_gap_seconds) / frame_seconds)))
        if max_gap_frames <= 0 or not any(flags):
            return
        index = 0
        while index < len(flags):
            if flags[index]:
                index += 1
                continue
            gap_start = index
            while index < len(flags) and not flags[index]:
                index += 1
            gap_end = index
            if gap_start > 0 and gap_end < len(flags) and gap_end - gap_start <= max_gap_frames:
                for fill_index in range(gap_start, gap_end):
                    flags[fill_index] = True

    def _vad_state_from_flags(
        self,
        left: float,
        right: float,
        audio_size: int,
        sample_rate: int,
        frame_samples: int,
        frame_seconds: float,
        flags: list[bool],
        starts: list[int],
        backend: str,
    ) -> VadWindowState:
        if not flags:
            return VadWindowState(False, False, backend=backend)

        self._smooth_vad_flags(flags, frame_seconds)
        spans: list[tuple[float, float]] = []
        index = 0
        while index < len(flags):
            if not flags[index]:
                index += 1
                continue
            span_start_index = index
            while index < len(flags) and flags[index]:
                index += 1
            span_end_index = index - 1
            span_start = left + (starts[span_start_index] / sample_rate)
            span_end = left + (min(audio_size, starts[span_end_index] + frame_samples) / sample_rate)
            if span_end <= span_start:
                continue
            spans.append((round(float(span_start), 4), round(float(span_end), 4)))
        return self._vad_state_from_spans(left, right, spans, backend=backend)

    def _vad_state_from_spans(
        self,
        left: float,
        right: float,
        spans: list[tuple[float, float]],
        *,
        backend: str,
        min_speech_seconds: float | None = None,
    ) -> VadWindowState:
        spans = [
            (round(max(left, float(start)), 4), round(min(right, float(end)), 4))
            for start, end in sorted(spans)
            if min(right, float(end)) > max(left, float(start))
        ]
        if not spans:
            return VadWindowState(False, False, backend=backend)

        speech_seconds = sum(max(0.0, end - start) for start, end in spans)
        if min_speech_seconds is None:
            min_speech_seconds = float(self.args.vad_min_speech_seconds)
        if speech_seconds < max(0.0, float(min_speech_seconds)):
            return VadWindowState(False, False, backend=backend)

        speech_start = spans[0][0]
        speech_end = spans[-1][1]
        trailing_silence = max(0.0, right - speech_end)
        should_flush = trailing_silence >= max(0.0, float(self.args.vad_silence_seconds))
        return VadWindowState(
            has_speech=True,
            should_flush=should_flush,
            speech_start=round(float(speech_start), 4),
            speech_end=round(float(speech_end), 4),
            speech_seconds=round(float(speech_seconds), 4),
            trailing_silence_seconds=round(float(trailing_silence), 4),
            backend=backend,
            speech_spans=spans,
        )

    def _rms_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        frame_seconds = max(0.01, float(self.args.vad_frame_seconds))
        frame_samples = max(1, int(sample_rate * frame_seconds))
        threshold = max(0.0, float(self.args.vad_speech_rms_threshold))
        flags: list[bool] = []
        starts: list[int] = []
        for start in range(0, audio.size, frame_samples):
            end = min(audio.size, start + frame_samples)
            if end - start < max(1, frame_samples // 2):
                break
            frame = audio[start:end]
            rms_value = float(np.sqrt(np.mean(frame * frame)))
            flags.append(rms_value >= threshold)
            starts.append(start)
        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=audio.size,
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend="rms",
        )

    def _audio_has_rms_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        frame_seconds = max(0.01, float(getattr(self.args, "vad_frame_seconds", 0.03)))
        threshold = max(0.0, float(getattr(self.args, "vad_speech_rms_threshold", 0.003)))
        min_speech_seconds = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "live_speaker_probe_min_speech_seconds",
                    getattr(self.args, "vad_min_speech_seconds", 0.15),
                )
            ),
        )
        return rms_speech_present(
            audio,
            sample_rate,
            frame_seconds=frame_seconds,
            threshold=threshold,
            min_speech_seconds=min_speech_seconds,
        )

    def _audio_has_live_probe_speech(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        backend = str(getattr(self.args, "live_speaker_probe_speech_backend", "rms") or "rms").lower()
        if backend != "vad":
            return self._audio_has_rms_speech(audio, sample_rate)
        if audio.size <= 0 or sample_rate <= 0 or right <= left:
            return False
        if getattr(self.args, "vad_backend", "silero") == "rms":
            return self._rms_vad_window_state(left, right, audio, sample_rate).has_speech
        return self._silero_vad_window_state(left, right, audio, sample_rate).has_speech

    def _silero_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        model = self._load_silero_vad_model()
        if model is None:
            return self._rms_vad_window_state(left, right, audio, sample_rate)

        vad_audio = self._resample_for_silero_vad(audio, sample_rate)
        if vad_audio.size <= 0:
            return VadWindowState(False, False, backend="silero")

        frame_samples = SILERO_VAD_CHUNK_SAMPLES
        frame_seconds = frame_samples / float(SILERO_VAD_SAMPLE_RATE)
        threshold = max(0.0, min(1.0, float(self.args.vad_silero_speech_threshold)))
        flags: list[bool] = []
        starts: list[int] = []
        reset_states = getattr(model, "reset_states", None)
        if callable(reset_states):
            reset_states()
        try:
            for start in range(0, vad_audio.size, frame_samples):
                end = min(vad_audio.size, start + frame_samples)
                if end - start < max(1, frame_samples // 2):
                    break
                chunk = vad_audio[start:end]
                if chunk.size < frame_samples:
                    padded = np.zeros(frame_samples, dtype=np.float32)
                    padded[:chunk.size] = chunk
                    chunk = padded
                probability = float(model(chunk.astype(np.float32, copy=False), SILERO_VAD_SAMPLE_RATE))
                flags.append(probability >= threshold)
                starts.append(start)
        except Exception as exc:
            self._vad_model_error = str(exc)
            self._vad_model = None
            self.bus.emit(
                "status",
                {"message": f"Silero ONNX VAD call failed; falling back to RMS VAD: {exc}"},
            )
            return self._rms_vad_window_state(left, right, audio, sample_rate)

        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=vad_audio.size,
            sample_rate=SILERO_VAD_SAMPLE_RATE,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend=self._vad_model_backend or "silero",
        )

    def _webrtc_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        if getattr(self, "_webrtc_vad_error", None):
            return VadWindowState(False, False, backend="webrtc_unavailable")
        try:
            import webrtcvad  # type: ignore[import-not-found]
        except Exception as exc:
            self._webrtc_vad_error = str(exc)
            self.bus.emit(
                "status",
                {"message": f"WebRTC VAD unavailable; using primary VAD gate only: {exc}"},
            )
            return VadWindowState(False, False, backend="webrtc_unavailable")

        vad_audio = self._resample_for_silero_vad(audio, sample_rate)
        if vad_audio.size <= 0:
            return VadWindowState(False, False, backend="webrtc")

        mode = max(0, min(3, int(getattr(self.args, "vad_gate_webrtc_mode", 3))))
        detector = webrtcvad.Vad(mode)
        frame_samples = int(SILERO_VAD_SAMPLE_RATE * 0.03)
        frame_seconds = frame_samples / float(SILERO_VAD_SAMPLE_RATE)
        flags: list[bool] = []
        starts: list[int] = []
        for start in range(0, vad_audio.size, frame_samples):
            end = min(vad_audio.size, start + frame_samples)
            if end - start < max(1, frame_samples // 2):
                break
            chunk = vad_audio[start:end]
            if chunk.size < frame_samples:
                padded = np.zeros(frame_samples, dtype=np.float32)
                padded[:chunk.size] = chunk
                chunk = padded
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            try:
                flags.append(bool(detector.is_speech(pcm16, SILERO_VAD_SAMPLE_RATE)))
            except Exception as exc:
                self._webrtc_vad_error = str(exc)
                self.bus.emit(
                    "status",
                    {"message": f"WebRTC VAD failed; using primary VAD gate only: {exc}"},
                )
                return VadWindowState(False, False, backend="webrtc_unavailable")
            starts.append(start)

        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=vad_audio.size,
            sample_rate=SILERO_VAD_SAMPLE_RATE,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend=f"webrtc{mode}",
        )

    @staticmethod
    def _spans_overlap_seconds(
        source_spans: list[tuple[float, float]],
        start: float,
        end: float,
    ) -> float:
        overlap = 0.0
        for other_start, other_end in source_spans:
            overlap += max(0.0, min(end, other_end) - max(start, other_start))
        return overlap

    def _vad_gate_secondary_backend(self) -> str:
        return str(getattr(self.args, "vad_gate_secondary_backend", "webrtc") or "off").lower()

    def _vad_gate_evidence_spans(
        self,
        primary_state: VadWindowState,
        secondary_state: VadWindowState | None,
    ) -> list[tuple[float, float]]:
        primary_spans = list(primary_state.speech_spans or [])
        if not primary_spans and primary_state.speech_start is not None and primary_state.speech_end is not None:
            primary_spans = [(float(primary_state.speech_start), float(primary_state.speech_end))]
        if not primary_state.has_speech or not primary_spans:
            return []
        if (
            secondary_state is None
            or self._vad_gate_secondary_backend() == "off"
            or str(secondary_state.backend or "").startswith("webrtc_unavailable")
        ):
            return primary_spans

        secondary_spans = list(secondary_state.speech_spans or [])
        if not secondary_spans and secondary_state.speech_start is not None and secondary_state.speech_end is not None:
            secondary_spans = [(float(secondary_state.speech_start), float(secondary_state.speech_end))]
        if not secondary_state.has_speech or not secondary_spans:
            return []

        min_seconds = max(0.0, float(getattr(self.args, "vad_gate_min_consensus_seconds", 0.12)))
        min_ratio = max(0.0, min(1.0, float(getattr(self.args, "vad_gate_min_consensus_ratio", 0.05))))
        validated_primary: list[tuple[float, float]] = []
        for start, end in primary_spans:
            duration = max(0.0, float(end) - float(start))
            if duration <= 0.0:
                continue
            overlap = self._spans_overlap_seconds(secondary_spans, float(start), float(end))
            required = min(duration, max(min_seconds, duration * min_ratio))
            if overlap >= required:
                validated_primary.append((float(start), float(end)))
        if not validated_primary:
            return []

        evidence: list[tuple[float, float]] = []
        for start, end in secondary_spans:
            if self._spans_overlap_seconds(validated_primary, float(start), float(end)) > 0.0:
                evidence.append((float(start), float(end)))
        return evidence

    def _vad_gate_window_state(
        self,
        left: float,
        right: float,
        *,
        force: bool = False,
        primary_state: VadWindowState | None = None,
    ) -> VadWindowState:
        if primary_state is None:
            primary_state = self._vad_window_state(left, right, force=force)
        if self._vad_gate_secondary_backend() == "off" or not primary_state.has_speech:
            return primary_state
        audio, sample_rate = self._audio_window_copy(left, right)
        if audio.size <= 0 or sample_rate <= 0:
            return VadWindowState(False, False, backend=primary_state.backend)
        secondary_state = self._webrtc_vad_window_state(left, right, audio, sample_rate)
        evidence_spans = self._vad_gate_evidence_spans(primary_state, secondary_state)
        if not evidence_spans:
            return VadWindowState(False, False, backend=f"{primary_state.backend}+{secondary_state.backend}")
        min_speech = max(0.0, float(getattr(self.args, "vad_gate_min_consensus_seconds", 0.12)))
        return self._vad_state_from_spans(
            left,
            right,
            evidence_spans,
            backend=f"{primary_state.backend}+{secondary_state.backend}",
            min_speech_seconds=min_speech,
        )

    def _vad_window_state(self, left: float, right: float, *, force: bool = False) -> VadWindowState:
        if not force and not getattr(self.args, "vad_sentence_splitting", True):
            return VadWindowState(False, False)
        if right <= left:
            return VadWindowState(False, False)

        audio, sample_rate = self._audio_window_copy(left, right)
        if audio.size <= 0 or sample_rate <= 0:
            return VadWindowState(False, False)

        if getattr(self.args, "vad_backend", "silero") == "rms":
            return self._rms_vad_window_state(left, right, audio, sample_rate)
        return self._silero_vad_window_state(left, right, audio, sample_rate)

    def _asr_vad_gate_enabled(self) -> bool:
        return bool(getattr(self.args, "asr_vad_gate", True))

    def _asr_vad_gate_spans(
        self,
        left: float,
        right: float,
        vad_state: VadWindowState,
        secondary_vad_state: VadWindowState | None = None,
    ) -> list[tuple[float, float]]:
        if not self._asr_vad_gate_enabled():
            return [(left, right)]
        if right <= left or not vad_state.has_speech:
            return []

        source_spans = self._vad_gate_evidence_spans(vad_state, secondary_vad_state)
        if not source_spans:
            return []

        pre_padding = max(0.0, float(getattr(self.args, "asr_vad_gate_pre_padding_seconds", 0.20)))
        post_padding = max(0.0, float(getattr(self.args, "asr_vad_gate_post_padding_seconds", 0.35)))
        merge_gap = max(0.0, float(getattr(self.args, "asr_vad_gate_merge_gap_seconds", 0.85)))
        min_clip_seconds = max(0.0, float(getattr(self.args, "asr_vad_gate_min_clip_seconds", 0.20)))
        cut_internal_gaps = bool(getattr(self.args, "asr_vad_gate_cut_internal_gaps", False))
        if not cut_internal_gaps:
            speech_start = min(start for start, _end in source_spans)
            speech_end = max(end for _start, end in source_spans)
            span = (max(left, speech_start - pre_padding), min(right, speech_end + post_padding))
            return [span] if span[1] - span[0] >= min_clip_seconds else []

        spans: list[tuple[float, float]] = []
        for start, end in sorted((float(start), float(end)) for start, end in source_spans):
            padded_start = max(left, start - pre_padding)
            padded_end = min(right, end + post_padding)
            if padded_end - padded_start < min_clip_seconds:
                continue
            if spans and padded_start <= spans[-1][1] + merge_gap:
                spans[-1] = (spans[-1][0], max(spans[-1][1], padded_end))
            else:
                spans.append((padded_start, padded_end))
        return spans

    def _warm_sentence_splitter(self) -> None:
        if self._sentence_splitter_warmed:
            self.bus.emit("status", {"message": "stream2sentence tokenizer already warm."})
            return
        sentence_tokenizer = str(getattr(
            self.args,
            "sentence_tokenizer",
            default_sentence_tokenizer(getattr(self.args, "language", "en")),
        ))
        sentence_language = str(getattr(
            self.args,
            "sentence_language",
            default_sentence_language(getattr(self.args, "language", "en")),
        ))
        self.bus.emit(
            "status",
            {"message": f"Initializing stream2sentence {sentence_tokenizer}/{sentence_language} tokenizer before playback."},
        )
        started = time.monotonic()
        init_tokenizer(sentence_tokenizer, language=sentence_language)
        list(generate_sentences(
            list("A warmup sentence vs. a false split. Another sentence."),
            tokenizer=sentence_tokenizer,
            language=sentence_language,
            auto_context=True,
            minimum_sentence_length=1,
            minimum_first_fragment_length=1,
            context_size=12,
            context_size_look_overhead=64,
        ))
        self._sentence_splitter_warmed = True
        self.bus.emit("status", {"message": f"stream2sentence ready in {time.monotonic() - started:.2f}s."})
