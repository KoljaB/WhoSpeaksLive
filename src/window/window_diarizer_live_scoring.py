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




class WindowLiveScoringMixin:
    def _realtime_unknown_speaker_payload(self) -> dict[str, Any]:
        return {
            "assigned_speaker": None,
            **self._speaker_info_for_payload(None),
            "created_speaker": False,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "top_similarity": None,
            "margin": None,
            "quality": None,
            "assignment_source": "realtime_preview",
        }

    @staticmethod
    def _speaker_id_from_probability_key(key: str) -> str | None:
        value = str(key or "").strip().lower()
        if not value.startswith("speaker") or not value[7:].isdigit():
            return None
        index = int(value[7:])
        return f"S{index}" if index > 0 else None

    def _live_speaker_ema_probabilities(self, probabilities: dict[str, float]) -> dict[str, float]:
        now = time.monotonic()
        window_seconds = max(0.0, float(getattr(self.args, "live_speaker_ema_window_seconds", 1.0)))
        max_count = max(1, int(getattr(self.args, "live_speaker_ema_count", 3)))
        alpha = max(0.05, min(1.0, float(getattr(self.args, "live_speaker_ema_alpha", 0.55))))
        clean = {str(key): float(value) for key, value in (probabilities or {}).items()}
        self._live_probability_history.append((now, clean))
        history = [
            item
            for item in self._live_probability_history
            if window_seconds <= 0.0 or now - item[0] <= window_seconds
        ][-max_count:]
        self._live_probability_history = deque(
            history,
            maxlen=max(1, int(getattr(self.args, "live_speaker_ema_count", 3))),
        )
        if not history:
            return clean
        keys = sorted({key for _time, item in history for key in item})
        ema = {key: float(history[0][1].get(key, 0.0)) for key in keys}
        for _time, item in history[1:]:
            for key in keys:
                ema[key] = alpha * float(item.get(key, 0.0)) + (1.0 - alpha) * ema.get(key, 0.0)
        total = sum(max(0.0, value) for value in ema.values())
        if total > 0.0:
            ema = {key: max(0.0, value) / total for key, value in ema.items()}
        return {key: round(float(value), 4) for key, value in ema.items()}

    def _assign_live_speaker_from_probabilities(
        self,
        probabilities: dict[str, float],
        fallback_speaker: str | None,
    ) -> str | None:
        speaker_items = [
            (self._speaker_id_from_probability_key(key), float(value))
            for key, value in probabilities.items()
            if self._speaker_id_from_probability_key(key) is not None
        ]
        if not speaker_items:
            return fallback_speaker
        speaker, probability = max(speaker_items, key=lambda item: item[1])
        unknown = float(probabilities.get("unknown", 0.0))
        if probability <= unknown:
            return None
        if probability < float(self.args.realtime_preview_diarize_min_known_probability):
            return None
        return speaker

    def _maybe_assign_weak_profile_live_speaker(self, decision: Any) -> tuple[str | None, dict[str, Any]]:
        if not bool(getattr(self.args, "live_speaker_weak_profile_assist", False)):
            return None, {}
        similarities = getattr(decision, "similarities", None)
        if not isinstance(similarities, dict) or not similarities:
            return None, {}
        try:
            max_profile_seconds = max(
                0.0,
                float(getattr(self.args, "live_speaker_weak_profile_max_speech_seconds", 2.5)),
            )
            min_similarity = float(getattr(self.args, "live_speaker_weak_profile_min_similarity", 0.40))
            min_margin = max(
                0.0,
                float(getattr(self.args, "live_speaker_weak_profile_min_margin", 0.12)),
            )
            max_unknown = max(
                0.0,
                float(getattr(self.args, "live_speaker_weak_profile_max_unknown_probability", 0.55)),
            )
        except (TypeError, ValueError):
            return None, {}
        top_speaker, top_similarity = max(
            ((str(speaker), float(value)) for speaker, value in similarities.items()),
            key=lambda item: item[1],
        )
        sorted_similarities = sorted((float(value) for value in similarities.values()), reverse=True)
        runner_up = sorted_similarities[1] if len(sorted_similarities) > 1 else -1.0
        margin = top_similarity - runner_up if len(sorted_similarities) > 1 else 1.0
        try:
            unknown_probability = max(0.0, float(getattr(decision, "unknown_probability", 1.0)))
        except (TypeError, ValueError):
            unknown_probability = 1.0
        if top_similarity < min_similarity or margin < min_margin or unknown_probability > max_unknown:
            return None, {}
        profile_seconds = None
        try:
            profiles = self.live_memory.export_profiles()
        except Exception:
            profiles = []
        for profile in profiles:
            if str(profile.get("label") or "") == top_speaker:
                try:
                    profile_seconds = max(0.0, float(profile.get("speech_seconds") or 0.0))
                except (TypeError, ValueError):
                    profile_seconds = None
                break
        if profile_seconds is None or profile_seconds > max_profile_seconds:
            return None, {}
        return top_speaker, {
            "weak_profile_live_assist": True,
            "weak_profile_speech_seconds": round(float(profile_seconds), 4),
            "weak_profile_min_similarity": round(float(min_similarity), 4),
            "weak_profile_min_margin": round(float(min_margin), 4),
            "weak_profile_max_unknown_probability": round(float(max_unknown), 4),
        }

    def _probability_for_speaker_id(self, probabilities: dict[str, float], speaker_id: str | None) -> float:
        if not speaker_id:
            return 0.0
        for key, value in (probabilities or {}).items():
            if self._speaker_id_from_probability_key(str(key)) == str(speaker_id):
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _maybe_promote_raw_live_speaker_change(
        self,
        active_speaker: str | None,
        speaker_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not active_speaker or not bool(getattr(self.args, "live_speaker_raw_change_snap", True)):
            return speaker_payload
        raw_probabilities = speaker_payload.get("raw_probabilities")
        if not isinstance(raw_probabilities, dict):
            return speaker_payload
        speaker_items = [
            (self._speaker_id_from_probability_key(str(key)), float(value))
            for key, value in raw_probabilities.items()
            if self._speaker_id_from_probability_key(str(key)) is not None
        ]
        if not speaker_items:
            return speaker_payload
        raw_speaker, raw_probability = max(speaker_items, key=lambda item: item[1])
        if not raw_speaker or str(raw_speaker) == str(active_speaker):
            return speaker_payload
        try:
            min_probability = max(
                float(getattr(self.args, "realtime_preview_diarize_min_known_probability", 0.5)),
                float(getattr(self.args, "live_speaker_raw_change_min_probability", 0.62)),
            )
        except (TypeError, ValueError):
            min_probability = 0.62
        try:
            min_margin = max(0.0, float(getattr(self.args, "live_speaker_raw_change_min_margin", 0.18)))
        except (TypeError, ValueError):
            min_margin = 0.18
        active_probability = self._probability_for_speaker_id(raw_probabilities, active_speaker)
        try:
            unknown_probability = max(0.0, float(raw_probabilities.get("unknown", 0.0)))
        except (TypeError, ValueError):
            unknown_probability = 0.0
        if raw_probability < min_probability:
            return speaker_payload
        if raw_probability <= unknown_probability:
            return speaker_payload
        if raw_probability < active_probability + min_margin:
            return speaker_payload
        self._ensure_speaker_metadata(raw_speaker)
        return {
            **speaker_payload,
            "assigned_speaker": raw_speaker,
            **self._speaker_info_for_payload(raw_speaker),
            "probabilities": dict(raw_probabilities),
            "assignment_source": "live_fast_embedding_raw_change_snap",
            "smoothed_assigned_speaker": speaker_payload.get("assigned_speaker"),
            "smoothed_probabilities": speaker_payload.get("probabilities"),
            "raw_change_previous_speaker": active_speaker,
            "raw_change_probability": round(float(raw_probability), 4),
            "raw_change_previous_probability": round(float(active_probability), 4),
            "raw_change_min_probability": round(float(min_probability), 4),
            "raw_change_min_margin": round(float(min_margin), 4),
        }

    def _should_hold_live_speaker_on_unknown(
        self,
        active_speaker: str | None,
        speaker_payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        mode = str(getattr(self.args, "live_speaker_probe_unknown_release_smoothing", "none") or "none").lower()
        if mode not in {"sma", "ema"} or not active_speaker:
            return False, {}
        probabilities = speaker_payload.get("probabilities")
        if not isinstance(probabilities, dict):
            probabilities = {}
        active_probability = self._probability_for_speaker_id(probabilities, active_speaker)
        try:
            unknown_probability = max(0.0, float(probabilities.get("unknown", 0.0)))
        except (TypeError, ValueError):
            unknown_probability = 0.0
        max_count = max(1, int(getattr(self.args, "live_speaker_probe_unknown_release_count", 3)))
        if getattr(self, "_live_unknown_release_speaker", None) != active_speaker:
            self._live_unknown_release_speaker = active_speaker
            self._live_unknown_release_history = deque(maxlen=max_count)
        history = getattr(self, "_live_unknown_release_history", deque(maxlen=max_count))
        if history.maxlen != max_count:
            history = deque(history, maxlen=max_count)
        history.append((active_probability, unknown_probability))
        self._live_unknown_release_history = history
        if mode == "ema":
            alpha = max(0.05, min(1.0, float(getattr(
                self.args,
                "live_speaker_probe_unknown_release_ema_alpha",
                0.5,
            ))))
            active_smoothed, unknown_smoothed = history[0]
            for active_item, unknown_item in list(history)[1:]:
                active_smoothed = alpha * active_item + (1.0 - alpha) * active_smoothed
                unknown_smoothed = alpha * unknown_item + (1.0 - alpha) * unknown_smoothed
        else:
            active_smoothed = sum(item[0] for item in history) / max(1, len(history))
            unknown_smoothed = sum(item[1] for item in history) / max(1, len(history))
        try:
            margin = max(0.0, float(getattr(
                self.args,
                "live_speaker_probe_unknown_release_margin",
                0.0,
            )))
        except (TypeError, ValueError):
            margin = 0.0
        hold = active_smoothed + margin >= unknown_smoothed
        return hold, {
            "release_smoothing": mode,
            "release_smoothing_count": len(history),
            "release_active_probability": round(float(active_probability), 4),
            "release_unknown_probability": round(float(unknown_probability), 4),
            "release_active_smoothed": round(float(active_smoothed), 4),
            "release_unknown_smoothed": round(float(unknown_smoothed), 4),
            "release_margin": round(float(margin), 4),
        }

    def _live_speaker_embedding_lock(self) -> threading.Lock:
        lock = getattr(self, "_live_speaker_embedding_throttle_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._live_speaker_embedding_throttle_lock = lock
        return lock

    def _live_speaker_embedding_min_interval_seconds(self) -> float:
        try:
            value = float(getattr(self.args, "live_speaker_embedding_min_interval_seconds", 0.75))
        except (TypeError, ValueError):
            value = 0.75
        return max(0.05, value)

    def _live_speaker_embedding_target_utilization(self) -> float:
        try:
            value = float(getattr(self.args, "live_speaker_embedding_target_utilization", 0.25))
        except (TypeError, ValueError):
            value = 0.25
        if not math.isfinite(value):
            value = 0.25
        return max(0.05, min(1.0, value))

    def _try_reserve_live_speaker_embedding(self) -> bool:
        now = time.monotonic()
        with self._live_speaker_embedding_lock():
            next_at = float(getattr(self, "_live_speaker_embedding_next_at", 0.0))
            if now < next_at:
                return False
            self._live_speaker_embedding_next_at = now + self._live_speaker_embedding_min_interval_seconds()
            return True

    def _record_live_speaker_embedding_latency(self, elapsed_seconds: float) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        target_utilization = self._live_speaker_embedding_target_utilization()
        min_interval = self._live_speaker_embedding_min_interval_seconds()
        with self._live_speaker_embedding_lock():
            previous = getattr(self, "_live_speaker_embedding_latency_ewma", None)
            if previous is None:
                latency = elapsed
            else:
                latency = (0.75 * float(previous)) + (0.25 * elapsed)
            self._live_speaker_embedding_latency_ewma = latency
            wait_seconds = min_interval
            if target_utilization < 1.0:
                wait_seconds = max(min_interval, latency * ((1.0 / target_utilization) - 1.0))
            self._live_speaker_embedding_next_at = max(
                float(getattr(self, "_live_speaker_embedding_next_at", 0.0)),
                time.monotonic() + wait_seconds,
            )
        self._maybe_emit_live_speaker_embedding_throttle_status(latency, wait_seconds, target_utilization)

    def _maybe_emit_live_speaker_embedding_throttle_status(
        self,
        latency_seconds: float,
        wait_seconds: float,
        target_utilization: float,
    ) -> None:
        if target_utilization >= 1.0:
            return
        if wait_seconds <= self._live_speaker_embedding_min_interval_seconds() + 0.05:
            return
        now = time.monotonic()
        if now - float(getattr(self, "_live_speaker_embedding_last_status_at", 0.0)) < 30.0:
            return
        self._live_speaker_embedding_last_status_at = now
        self.bus.emit(
            "status",
            {
                "message": (
                    "Adaptive live speaker embedding throttle: "
                    f"latency {latency_seconds:.2f}s, next live speaker embedding "
                    f"in at least {wait_seconds:.2f}s (target {target_utilization:.0%})."
                )
            },
        )

    def _live_speaker_change_verification_enabled(self) -> bool:
        return bool(getattr(self.args, "live_speaker_verify_on_change", False)) and bool(
            getattr(self, "_live_embedding_separate", False)
        )

    def _live_speaker_verify_min_interval_seconds(self) -> float:
        try:
            value = float(getattr(self.args, "live_speaker_verify_min_interval_seconds", 2.0))
        except (TypeError, ValueError):
            value = 2.0
        if not math.isfinite(value):
            value = 2.0
        return max(0.0, value)

    def _try_reserve_live_speaker_verification(self) -> bool:
        now = time.monotonic()
        lock = getattr(self, "_live_speaker_verify_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._live_speaker_verify_lock = lock
        with lock:
            next_at = float(getattr(self, "_live_speaker_verify_next_at", 0.0))
            if now < next_at:
                return False
            self._live_speaker_verify_next_at = now + self._live_speaker_verify_min_interval_seconds()
            return True

    def _score_live_speaker_with_full_stack(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
    ) -> dict[str, Any]:
        if duration_seconds < max(0.0, float(self.args.realtime_preview_diarize_min_audio_seconds)):
            return self._realtime_unknown_speaker_payload()
        if self.memory.profile_count() <= 0:
            return self._realtime_unknown_speaker_payload()

        chunk = pad_audio(trim_silence(audio, sample_rate), self.args.min_embed_seconds, sample_rate)
        embedding = self._embed_audio_chunk(chunk, sample_rate, ".live-verify.wav")
        decision = self.memory.score_existing(
            embedding,
            duration_seconds,
            min_similarity=self.args.realtime_preview_diarize_min_similarity,
            min_margin=self.args.realtime_preview_diarize_min_margin,
        )
        probabilities = dict(decision.probabilities)
        assigned_speaker = self._assign_live_speaker_from_probabilities(
            probabilities,
            decision.assigned_speaker,
        )
        self._ensure_speaker_metadata(assigned_speaker)
        return {
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": False,
            "probabilities": probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            "assignment_source": "live_full_stack_change_verify",
        }

    def _verify_live_speaker_change(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
        fast_payload: dict[str, Any],
        active_speaker: str | None,
    ) -> dict[str, Any] | None:
        if not self._live_speaker_change_verification_enabled():
            return fast_payload
        if not self._try_reserve_live_speaker_verification():
            now = time.monotonic()
            if now - float(getattr(self, "_live_speaker_verify_last_status_at", 0.0)) >= 30.0:
                self._live_speaker_verify_last_status_at = now
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            "Skipped full-stack live speaker verification during cooldown; "
                            f"current speaker remains {active_speaker or 'unknown'}."
                        )
                    },
                )
            return None
        try:
            verified_payload = self._score_live_speaker_with_full_stack(audio, sample_rate, duration_seconds)
        except Exception as exc:
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Full-stack live speaker verification failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                },
            )
            return None

        return {
            **verified_payload,
            "fast_assigned_speaker": fast_payload.get("assigned_speaker"),
            "fast_probabilities": fast_payload.get("probabilities"),
            "fast_raw_probabilities": fast_payload.get("raw_probabilities"),
            "verified_live_change": True,
            "previous_live_speaker": active_speaker,
        }

    def _score_realtime_preview_speaker(
        self,
        audio: np.ndarray,
        duration_seconds: float,
        min_audio_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not self._live_speaker_assignment_enabled():
            return self._realtime_unknown_speaker_payload()
        if min_audio_seconds is None:
            min_audio_seconds = float(self.args.realtime_preview_diarize_min_audio_seconds)
        if duration_seconds < max(0.0, float(min_audio_seconds)):
            return self._realtime_unknown_speaker_payload()
        memory = self.live_memory
        if memory.profile_count() <= 0:
            return self._realtime_unknown_speaker_payload()

        chunk = pad_audio(trim_silence(audio, self.sample_rate), self.args.min_embed_seconds, self.sample_rate)
        try:
            embed_started = time.monotonic()
            embedding = self._embed_live_audio_chunk(chunk, self.sample_rate, ".live.wav")
            self._record_live_speaker_embedding_latency(time.monotonic() - embed_started)
            decision = memory.score_existing(
                embedding,
                duration_seconds,
                min_similarity=self.args.realtime_preview_diarize_min_similarity,
                min_margin=self.args.realtime_preview_diarize_min_margin,
            )
        except Exception as exc:
            self.bus.emit("status", {"message": f"Realtime preview speaker scoring error: {type(exc).__name__}: {exc}"})
            return self._realtime_unknown_speaker_payload()

        raw_probabilities = dict(decision.probabilities)
        smoothed_probabilities = self._live_speaker_ema_probabilities(raw_probabilities)
        assigned_speaker = self._assign_live_speaker_from_probabilities(
            smoothed_probabilities,
            decision.assigned_speaker,
        )
        assist_payload: dict[str, Any] = {}
        if not assigned_speaker:
            assigned_speaker, assist_payload = self._maybe_assign_weak_profile_live_speaker(decision)
        self._ensure_speaker_metadata(assigned_speaker)

        return {
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": False,
            "probabilities": smoothed_probabilities,
            "raw_probabilities": raw_probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            **assist_payload,
            "assignment_source": (
                "live_weak_profile_assist"
                if assist_payload
                else (
                    "live_fast_embedding_ema"
                    if self._live_embedding_separate
                    else "realtime_preview_embedding_ema"
                )
            ),
        }

    def _update_live_speaker_memory(
        self,
        speaker_id: str | None,
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
        suffix: str = ".live-profile.wav",
        speaker_generation: int | None = None,
    ) -> None:
        if not getattr(self, "_live_embedding_separate", False) or not speaker_id:
            return
        job = LiveSpeakerMemoryUpdateJob(
            speaker_id=str(speaker_id),
            audio=np.asarray(audio, dtype=np.float32).copy(),
            sample_rate=int(sample_rate),
            duration_seconds=float(duration_seconds),
            suffix=suffix,
            speaker_generation=(
                int(getattr(self, "_speaker_generation", 0))
                if speaker_generation is None
                else int(speaker_generation)
            ),
            speaker_label_generation=int(
                getattr(self, "_speaker_label_generations", {}).get(str(speaker_id), 0)
            ),
            run_id=str(getattr(getattr(self, "_active_run", None), "run_id", "")),
        )
        jobs = getattr(self, "_live_memory_update_jobs", None)
        if jobs is None:
            self._process_live_speaker_memory_update(job)
            return
        try:
            jobs.put_nowait(job)
        except queue.Full:
            try:
                jobs.get_nowait()
            except queue.Empty:
                pass
            else:
                jobs.task_done()
                self.bus.emit(
                    "status",
                    {"message": "Dropped stale queued live speaker profile update because the queue is full."},
                )
            try:
                jobs.put_nowait(job)
            except queue.Full:
                self.bus.emit(
                    "status",
                    {"message": "Skipped live speaker profile update because the queue is still full."},
                )

    def _live_memory_update_job_is_current(self, job: LiveSpeakerMemoryUpdateJob) -> bool:
        active_run = getattr(self, "_active_run", None)
        if job.run_id and (active_run is None or job.run_id != active_run.run_id):
            return False
        if job.speaker_generation != getattr(self, "_speaker_generation", 0):
            return False
        label_generations = getattr(self, "_speaker_label_generations", {})
        return int(getattr(job, "speaker_label_generation", 0)) == int(
            label_generations.get(job.speaker_id, 0)
        )

    def _process_live_speaker_memory_update(self, job: LiveSpeakerMemoryUpdateJob) -> None:
        try:
            if not self._live_memory_update_job_is_current(job):
                return
            if not self._live_update_speaker_exists(job.speaker_id):
                return
            embedding = self._embed_live_audio_chunk(job.audio, job.sample_rate, job.suffix)
            with self._live_memory_update_lock_obj():
                if not self._live_memory_update_job_is_current(job):
                    return
                if not self._live_update_speaker_exists(job.speaker_id):
                    return
                self.live_memory.upsert_profile(
                    job.speaker_id,
                    embedding,
                    duration_seconds=job.duration_seconds,
                    sentence_count=1,
                )
        except Exception as exc:
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Live speaker profile update failed "
                        f"for {job.speaker_id}: {type(exc).__name__}: {exc}"
                    )
                },
            )
