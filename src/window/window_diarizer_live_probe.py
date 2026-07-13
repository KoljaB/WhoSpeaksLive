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




class WindowLiveProbeMixin:
    def _run_live_speaker_probe(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or self._stop
        if not self._live_speaker_assignment_enabled():
            return
        if not bool(getattr(self.args, "live_speaker_probe", True)):
            return
        interval_seconds = max(0.05, float(getattr(self.args, "live_speaker_probe_interval_seconds", 0.4)))
        attack_interval_seconds = max(
            0.0,
            float(getattr(self.args, "live_speaker_probe_attack_interval_seconds", 0.0)),
        )
        window_seconds = max(0.05, float(getattr(self.args, "live_speaker_probe_window_seconds", 1.25)))
        min_advance = max(0.0, float(getattr(self.args, "live_speaker_probe_min_advance_seconds", interval_seconds)))
        attack_min_advance = max(
            0.0,
            float(getattr(self.args, "live_speaker_probe_attack_min_advance_seconds", 0.0)),
        )
        if attack_interval_seconds > 0.0 and attack_min_advance <= 0.0:
            attack_min_advance = attack_interval_seconds
        hold_seconds = max(0.0, float(getattr(self.args, "live_speaker_probe_hold_seconds", 1.5)))
        clear_on_silence = bool(getattr(self.args, "live_speaker_probe_clear_on_silence", True))
        clear_window_seconds = max(
            0.05,
            float(getattr(self.args, "live_speaker_probe_clear_window_seconds", min(window_seconds, 1.0))),
        )
        clear_silence_count = max(1, int(getattr(self.args, "live_speaker_probe_clear_silence_count", 1)))
        clear_unknown_count = max(0, int(getattr(self.args, "live_speaker_probe_clear_unknown_count", 2)))
        unknown_keepalive = bool(getattr(self.args, "live_speaker_probe_unknown_keepalive", False))
        probe_min_audio_seconds = min(
            window_seconds,
            max(0.0, float(getattr(self.args, "realtime_preview_diarize_min_audio_seconds", window_seconds))),
        )
        last_probe_right = -1.0
        active_speaker: str | None = None
        provisional_counter = 0
        consecutive_unknown = 0
        consecutive_silence = 0
        while True:
            attack_mode = attack_interval_seconds > 0.0 and (
                not active_speaker or consecutive_unknown > 0
            )
            wait_seconds = attack_interval_seconds if attack_mode else interval_seconds
            if stop_event.wait(max(0.05, wait_seconds)):
                break
            if self.memory.profile_count() <= 0:
                continue
            right = self.playback_time()
            if right <= 0.0:
                continue
            if active_speaker and clear_on_silence:
                clear_left = max(0.0, right - min(clear_window_seconds, window_seconds))
                clear_audio, clear_sample_rate = self._audio_window_copy(clear_left, right)
                if clear_audio.size > 0 and not self._audio_has_live_probe_speech(
                    clear_left,
                    right,
                    clear_audio,
                    clear_sample_rate,
                ):
                    consecutive_silence += 1
                    if consecutive_silence >= clear_silence_count:
                        self.bus.emit(
                            "live_speaker_clear",
                            {
                                "speaker_id": active_speaker,
                                "live": False,
                                "fallback": True,
                                "start": round(float(clear_left), 4),
                                "end": round(float(right), 4),
                                "reason": "silence",
                                "silence_count": consecutive_silence,
                                "assignment_source": "live_speaker_embedding_probe_clear",
                            },
                        )
                        active_speaker = None
                        consecutive_unknown = 0
                        consecutive_silence = 0
                        self._live_probability_history.clear()
                else:
                    consecutive_silence = 0
            current_min_advance = attack_min_advance if attack_mode else min_advance
            if last_probe_right >= 0.0 and right < last_probe_right + current_min_advance:
                continue
            left = max(0.0, right - window_seconds)
            audio, sample_rate = self._audio_window_copy(left, right)
            duration_seconds = audio.size / float(sample_rate) if sample_rate > 0 else 0.0
            if duration_seconds < probe_min_audio_seconds:
                continue
            last_probe_right = right
            if not self._audio_has_live_probe_speech(left, right, audio, sample_rate):
                continue
            if not self._try_reserve_live_speaker_embedding():
                continue
            speaker_payload = self._score_realtime_preview_speaker(
                audio,
                duration_seconds,
                min_audio_seconds=probe_min_audio_seconds,
            )
            speaker_payload = self._maybe_promote_raw_live_speaker_change(active_speaker, speaker_payload)
            if stop_event.is_set():
                break
            assigned_speaker = speaker_payload.get("assigned_speaker")
            if not assigned_speaker:
                if (
                    bool(getattr(self.args, "live_speaker_provisional_new_speaker", False))
                    and not active_speaker
                    and duration_seconds >= max(
                        0.0,
                        float(getattr(self.args, "live_speaker_provisional_min_audio_seconds", 1.0)),
                    )
                ):
                    probabilities = speaker_payload.get("probabilities")
                    if not isinstance(probabilities, dict):
                        probabilities = {}
                    try:
                        unknown_probability = max(0.0, float(probabilities.get("unknown", 0.0)))
                    except (TypeError, ValueError):
                        unknown_probability = 0.0
                    min_unknown_probability = max(
                        0.0,
                        float(getattr(self.args, "live_speaker_provisional_min_unknown_probability", 0.5)),
                    )
                    if unknown_probability >= min_unknown_probability:
                        provisional_counter += 1
                        active_speaker = f"LIVE_NEW_{provisional_counter}"
                        self._ensure_speaker_metadata(active_speaker, source="live_provisional")
                        consecutive_unknown = 0
                        consecutive_silence = 0
                        self._live_probability_history.clear()
                        self.bus.emit(
                            "live_speaker",
                            {
                                **speaker_payload,
                                "assigned_speaker": active_speaker,
                                **self._speaker_info_for_payload(active_speaker),
                                "created_speaker": True,
                                "speaker_id": active_speaker,
                                "live": True,
                                "fallback": True,
                                "provisional_speaker": True,
                                "start": round(float(left), 4),
                                "end": round(float(right), 4),
                                "audio_length_seconds": round(float(duration_seconds), 4),
                                "hold_seconds": round(float(hold_seconds), 4),
                                "assignment_source": "live_provisional_new_speaker",
                            },
                        )
                        continue
                smoothed_hold, release_payload = self._should_hold_live_speaker_on_unknown(
                    active_speaker,
                    speaker_payload,
                )
                if active_speaker and (unknown_keepalive or smoothed_hold):
                    if smoothed_hold:
                        consecutive_unknown = 0
                    else:
                        consecutive_unknown += 1
                    self.bus.emit(
                        "live_speaker",
                        {
                            **speaker_payload,
                            "assigned_speaker": active_speaker,
                            **self._speaker_info_for_payload(active_speaker),
                            "created_speaker": False,
                            "speaker_id": active_speaker,
                            "live": True,
                            "fallback": True,
                            "start": round(float(left), 4),
                            "end": round(float(right), 4),
                            "audio_length_seconds": round(float(duration_seconds), 4),
                            "hold_seconds": round(float(hold_seconds), 4),
                            **release_payload,
                            "debounced_unknown": True,
                            "assignment_source": (
                                "live_speaker_unknown_release_smoothing"
                                if smoothed_hold
                                else "live_speaker_unknown_keepalive"
                            ),
                        },
                    )
                    if smoothed_hold or clear_unknown_count <= 0 or consecutive_unknown < clear_unknown_count:
                        continue
                else:
                    consecutive_unknown += 1
                if active_speaker and clear_unknown_count and consecutive_unknown >= clear_unknown_count:
                    verification_payload = self._verify_live_speaker_change(
                        audio,
                        sample_rate,
                        duration_seconds,
                        speaker_payload,
                        active_speaker,
                    )
                    if verification_payload is None:
                        continue
                    verified_speaker = verification_payload.get("assigned_speaker")
                    if verified_speaker:
                        active_speaker = str(verified_speaker)
                        consecutive_unknown = 0
                        self.bus.emit(
                            "live_speaker",
                            {
                                **verification_payload,
                                "speaker_id": active_speaker,
                                "live": True,
                                "fallback": True,
                                "start": round(float(left), 4),
                                "end": round(float(right), 4),
                                "audio_length_seconds": round(float(duration_seconds), 4),
                                "hold_seconds": round(float(hold_seconds), 4),
                                "assignment_source": str(
                                    verification_payload.get("assignment_source")
                                    or "live_full_stack_change_verify"
                                ),
                            },
                        )
                        continue
                    self.bus.emit(
                        "live_speaker_clear",
                        {
                            "speaker_id": active_speaker,
                            "live": False,
                            "fallback": True,
                            "start": round(float(left), 4),
                            "end": round(float(right), 4),
                            "reason": "unknown",
                            "unknown_count": consecutive_unknown,
                            "assignment_source": "live_speaker_embedding_probe_clear",
                        },
                    )
                    active_speaker = None
                    self._live_unknown_release_speaker = None
                    self._live_unknown_release_history = deque()
                    self._live_probability_history.clear()
                continue
            if active_speaker and str(assigned_speaker) != str(active_speaker):
                verification_payload = self._verify_live_speaker_change(
                    audio,
                    sample_rate,
                    duration_seconds,
                    speaker_payload,
                    active_speaker,
                )
                if verification_payload is None:
                    continue
                assigned_speaker = verification_payload.get("assigned_speaker")
                if not assigned_speaker:
                    consecutive_unknown += 1
                    continue
                speaker_payload = verification_payload
            active_speaker = str(assigned_speaker)
            consecutive_unknown = 0
            consecutive_silence = 0
            self._live_unknown_release_speaker = None
            self._live_unknown_release_history = deque()
            self.bus.emit(
                "live_speaker",
                {
                    **speaker_payload,
                    "speaker_id": assigned_speaker,
                    "live": True,
                    "fallback": True,
                    "start": round(float(left), 4),
                    "end": round(float(right), 4),
                    "audio_length_seconds": round(float(duration_seconds), 4),
                    "hold_seconds": round(float(hold_seconds), 4),
                    "assignment_source": str(
                        speaker_payload.get("assignment_source")
                        or "live_speaker_embedding_probe"
                    ),
                },
            )
