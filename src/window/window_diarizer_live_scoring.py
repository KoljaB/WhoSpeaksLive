"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass, replace
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
from window.live_speaker_algorithm import (
    ALGORITHM_ID as LIVE_SPEAKER_ALGORITHM_ID,
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerStep,
)
from window.live_speaker_bayes import (
    BayesSpeakerTrackerConfig,
    CausalBayesSpeakerTracker,
)
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_open_set_tracklets import (
    OPEN_SET_TRACKLET_PRESET,
    OpenSetTrackletOverlay,
    OpenSetTrackletStep,
    open_set_tracklet_config_for_preset,
)
from window.live_speaker_replay import blend_live_speaker_embeddings




class WindowLiveScoringMixin:
    def _live_speaker_can_score_without_final_profiles(self) -> bool:
        """Return whether the configured live tracker can establish its own identity."""

        if str(getattr(self.args, "live_speaker_tracker", "classic")) != "bayes":
            return False
        return bool(
            getattr(self.args, "live_speaker_bayes_provisional_profiles", False)
            or (
                getattr(self.args, "live_speaker_open_set_tracklets", False)
                and getattr(self.args, "live_speaker_open_set_preprofile", False)
            )
        )

    def _live_speaker_correlation_run_id(self) -> str:
        active_run = getattr(self, "_active_run", None)
        return str(getattr(active_run, "run_id", "") or "")

    def _shared_live_speaker_lock(self) -> threading.Lock:
        lock = getattr(self, "_shared_live_speaker_state_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._shared_live_speaker_state_lock = lock
        return lock

    def _shared_live_speaker_config(self) -> LiveSpeakerAlgorithmConfig:
        return LiveSpeakerAlgorithmConfig(
            min_similarity=float(getattr(self.args, "realtime_preview_diarize_min_similarity", 0.45)),
            min_margin=float(getattr(self.args, "realtime_preview_diarize_min_margin", 0.08)),
            min_known_probability=float(
                getattr(self.args, "realtime_preview_diarize_min_known_probability", 0.5)
            ),
            ema_count=max(1, int(getattr(self.args, "live_speaker_ema_count", 1))),
            ema_alpha=float(getattr(self.args, "live_speaker_ema_alpha", 0.55)),
            acquire_count=max(1, int(getattr(self.args, "live_speaker_acquire_count", 1))),
            switch_count=max(1, int(getattr(self.args, "live_speaker_switch_count", 1))),
            unknown_release_count=max(
                1, int(getattr(self.args, "live_speaker_probe_clear_unknown_count", 2) or 1)
            ),
            silence_release_count=max(
                1, int(getattr(self.args, "live_speaker_probe_clear_silence_count", 1))
            ),
        )

    def _shared_live_speaker_algorithm(self) -> Any:
        tracker_name = str(getattr(self.args, "live_speaker_tracker", "classic") or "classic")
        if tracker_name == "bayes":
            context_weight = max(
                0.0,
                min(1.0, float(getattr(self.args, "live_speaker_probe_context_weight", 0.0))),
            )
            short_window = max(
                0.01, float(getattr(self.args, "live_speaker_probe_window_seconds", 1.0))
            )
            context_window = max(
                short_window,
                float(getattr(self.args, "live_speaker_probe_context_window_seconds", 0.0)),
            )
            windows = (short_window, context_window) if context_weight > 0.0 and context_window > short_window else (short_window,)
            weights = (1.0 - context_weight, context_weight) if len(windows) == 2 else (1.0,)
            bayes_config = BayesSpeakerTrackerConfig(
                scale_windows=windows,
                scale_weights=weights,
                min_similarity=float(getattr(self.args, "realtime_preview_diarize_min_similarity", 0.45)),
                min_margin=float(getattr(self.args, "realtime_preview_diarize_min_margin", 0.08)),
                similarity_temperature=float(getattr(self.args, "live_speaker_bayes_temperature", 0.10)),
                unknown_bias=float(getattr(self.args, "live_speaker_bayes_unknown_bias", 0.0)),
                profile_count_bias_threshold=max(
                    0, int(getattr(self.args, "live_speaker_bayes_profile_count_threshold", 0))
                ),
                low_profile_unknown_bias=float(
                    getattr(self.args, "live_speaker_bayes_low_profile_unknown_bias", 0.0)
                ),
                high_profile_unknown_bias=float(
                    getattr(self.args, "live_speaker_bayes_high_profile_unknown_bias", 0.0)
                ),
                profile_count_unknown_bias_slope=float(
                    getattr(self.args, "live_speaker_bayes_profile_count_bias_slope", 0.0)
                ),
                enable_provisional_profiles=bool(
                    getattr(self.args, "live_speaker_bayes_provisional_profiles", False)
                ),
                provisional_creation_count=max(1, int(
                    getattr(self.args, "live_speaker_bayes_provisional_creation_count", 2)
                )),
                provisional_later_creation_count=max(0, int(
                    getattr(self.args, "live_speaker_bayes_provisional_later_creation_count", 0)
                )),
                provisional_later_creation_profile_threshold=max(0, int(
                    getattr(self.args, "live_speaker_bayes_provisional_later_creation_profile_threshold", 0)
                )),
                provisional_creation_similarity_ceiling=float(
                    getattr(self.args, "live_speaker_bayes_provisional_creation_similarity_ceiling", 0.20)
                ),
                provisional_boundary_creation_similarity_ceiling=float(getattr(
                    self.args,
                    "live_speaker_bayes_provisional_boundary_creation_similarity_ceiling",
                    -1.0,
                )),
                provisional_boundary_continuity_max_similarity=float(getattr(
                    self.args,
                    "live_speaker_bayes_provisional_boundary_continuity",
                    -1.0,
                )),
                provisional_creation_max_finalized_profiles=int(
                    getattr(self.args, "live_speaker_bayes_provisional_max_finalized_profiles", -1)
                ),
                provisional_merge_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_provisional_merge_min_similarity", 0.25)
                ),
                provisional_update_alpha=float(
                    getattr(self.args, "live_speaker_bayes_provisional_update_alpha", 0.0)
                ),
                provisional_update_continuity_min_similarity=float(
                    getattr(
                        self.args,
                        "live_speaker_bayes_provisional_update_continuity",
                        -1.0,
                    )
                ),
                provisional_update_history_size=max(1, int(
                    getattr(
                        self.args,
                        "live_speaker_bayes_provisional_update_history_size",
                        1,
                    )
                )),
                provisional_max_active_count=max(0, int(
                    getattr(self.args, "live_speaker_bayes_provisional_max_active_count", 0)
                )),
                provisional_pool_overflow_update_alpha=float(
                    getattr(
                        self.args,
                        "live_speaker_bayes_provisional_pool_overflow_update_alpha",
                        0.0,
                    )
                ),
                provisional_scale_agreement_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_provisional_scale_agreement", -1.0)
                ),
                provisional_assignment_scale_agreement_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_provisional_assignment_scale_agreement", -1.0)
                ),
                incumbent_hold_scale_agreement_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_incumbent_hold_scale_agreement", -1.0)
                ),
                incumbent_continuity_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_incumbent_continuity", -1.0)
                ),
                incumbent_continuity_history_size=max(1, int(
                    getattr(
                        self.args,
                        "live_speaker_bayes_incumbent_continuity_history_size",
                        3,
                    )
                )),
                incumbent_continuity_update_on_hold=bool(
                    getattr(
                        self.args,
                        "live_speaker_bayes_incumbent_continuity_update_on_hold",
                        False,
                    )
                ),
                boundary_short_only_max_continuity=float(getattr(
                    self.args,
                    "live_speaker_bayes_boundary_short_only_continuity",
                    -1.0,
                )),
                boundary_residual_incumbent_alpha=float(getattr(
                    self.args,
                    "live_speaker_bayes_boundary_residual_incumbent_alpha",
                    0.0,
                )),
                short_long_crossover_min_margin=float(getattr(
                    self.args,
                    "live_speaker_bayes_short_long_crossover_min_margin",
                    -1.0,
                )),
                short_long_crossover_min_similarity=float(getattr(
                    self.args,
                    "live_speaker_bayes_short_long_crossover_min_similarity",
                    -1.0,
                )),
                short_long_crossover_count=max(1, int(getattr(
                    self.args,
                    "live_speaker_bayes_short_long_crossover_count",
                    1,
                ))),
                short_long_differential_candidate_gain=float(getattr(
                    self.args,
                    "live_speaker_bayes_short_long_differential_candidate_gain",
                    -2.0,
                )),
                short_long_differential_incumbent_loss=float(getattr(
                    self.args,
                    "live_speaker_bayes_short_long_differential_incumbent_loss",
                    -2.0,
                )),
                provisional_temporal_consistency_min_similarity=float(
                    getattr(self.args, "live_speaker_bayes_provisional_temporal_consistency", -1.0)
                ),
                stay_probability=float(getattr(self.args, "live_speaker_bayes_stay_probability", 0.50)),
                prior_strength=float(getattr(self.args, "live_speaker_bayes_prior_strength", 0.0)),
                evidence_strength=float(getattr(self.args, "live_speaker_bayes_evidence_strength", 1.0)),
                min_known_probability=float(
                    getattr(self.args, "realtime_preview_diarize_min_known_probability", 0.5)
                ),
                switch_probability_margin=float(
                    getattr(self.args, "live_speaker_bayes_switch_probability_margin", 0.0)
                ),
                unknown_release_count=max(
                    1, int(getattr(self.args, "live_speaker_probe_clear_unknown_count", 2) or 1)
                ),
                silence_release_count=max(
                    1, int(getattr(self.args, "live_speaker_probe_clear_silence_count", 1))
                ),
            )
            algorithm = getattr(self, "_shared_live_speaker_core", None)
            if not isinstance(algorithm, CausalBayesSpeakerTracker) or algorithm.config != bayes_config:
                algorithm = CausalBayesSpeakerTracker(config=bayes_config)
                self._shared_live_speaker_core = algorithm
            return algorithm
        config = self._shared_live_speaker_config()
        algorithm = getattr(self, "_shared_live_speaker_core", None)
        if not isinstance(algorithm, CausalLiveSpeakerAlgorithm) or algorithm.config != config:
            algorithm = CausalLiveSpeakerAlgorithm(config=config)
            self._shared_live_speaker_core = algorithm
        return algorithm

    def _open_set_tracklet_overlay(self) -> OpenSetTrackletOverlay | None:
        if not bool(getattr(self.args, "live_speaker_open_set_tracklets", False)):
            return None
        tracker = str(getattr(self.args, "live_speaker_tracker", "classic"))
        provider = str(self._current_live_embedding_provider())
        short_window = float(getattr(self.args, "live_speaker_probe_window_seconds", 0.0))
        long_window = float(
            getattr(self.args, "live_speaker_probe_context_window_seconds", 0.0)
        )
        context_weight = float(getattr(self.args, "live_speaker_probe_context_weight", 0.0))
        if tracker != "bayes":
            raise ValueError("Open-set tracklets require --live-speaker-tracker bayes")
        if provider != "speechbrain_resnet":
            raise ValueError("Open-set tracklets require speechbrain_resnet")
        if abs(short_window - 0.7) > 1e-6 or abs(long_window - 1.5) > 1e-6:
            raise ValueError("Open-set tracklets require exactly 0.7/1.5-second windows")
        if context_weight <= 0.0:
            raise ValueError("Open-set tracklets require the existing context embedding")
        if bool(getattr(self.args, "live_speaker_bayes_provisional_profiles", False)):
            raise ValueError(
                "Open-set tracklets cannot run with legacy Bayes provisional profiles"
            )
        preset = str(
            getattr(
                self.args,
                "live_speaker_open_set_tracklet_preset",
                OPEN_SET_TRACKLET_PRESET,
            )
        )
        config = open_set_tracklet_config_for_preset(preset)
        overlay = getattr(self, "_shared_live_speaker_open_set_overlay_state", None)
        if not isinstance(overlay, OpenSetTrackletOverlay) or overlay.config != config:
            overlay = OpenSetTrackletOverlay(config)
            self._shared_live_speaker_open_set_overlay_state = overlay
        return overlay

    def _shared_live_speaker_step(
        self,
        *,
        media_time: float,
        speech: bool,
        embedding: np.ndarray | None,
        duration_seconds: float,
        probe_scheduled: bool,
        context_embedding: np.ndarray | None = None,
        context_duration_seconds: float | None = None,
        release_signal: bool = False,
        embedding_latency_seconds: float | None = None,
        skipped_reason: str = "",
        run_id: str = "",
        probe_id: str = "",
        request_id: str = "",
        correlation_out: dict[str, Any] | None = None,
    ) -> Any:
        correlation = {
            "run_id": str(run_id or self._live_speaker_correlation_run_id()),
            "probe_id": str(probe_id or ""),
            "request_id": str(request_id or ""),
        }
        with self._shared_live_speaker_lock():
            algorithm = self._shared_live_speaker_algorithm()
            profiles = self.live_memory.export_profiles()
            algorithm.sync_profiles(profiles)
            step_id = int(getattr(self, "_live_speaker_world_tape_step_id", 0)) + 1
            self._live_speaker_world_tape_step_id = step_id
            correlation["step_id"] = step_id
            if correlation_out is not None:
                correlation_out.update(correlation)
            algorithm_config = (
                asdict(algorithm.config)
                if is_dataclass(getattr(algorithm, "config", None))
                else {}
            )
            emit_internal = getattr(self.bus, "emit_internal", None)
            if callable(emit_internal):
                emit_internal(
                    "live_speaker_core_input",
                    {
                        **correlation,
                        "media_time": max(0.0, float(media_time)),
                        "speech": bool(speech),
                        "duration_seconds": max(0.0, float(duration_seconds)),
                        "probe_scheduled": bool(probe_scheduled),
                        "release_signal": bool(release_signal),
                        "embedding_latency_seconds": embedding_latency_seconds,
                        "skipped_reason": str(skipped_reason),
                        "algorithm_type": (
                            "bayes"
                            if isinstance(algorithm, CausalBayesSpeakerTracker)
                            else "classic"
                        ),
                        "algorithm_config": algorithm_config,
                        "profiles": profiles,
                        "embedding": (
                            None
                            if embedding is None
                            else np.asarray(embedding, dtype=np.float32).tolist()
                        ),
                        "context_embedding": (
                            None
                            if context_embedding is None
                            else np.asarray(context_embedding, dtype=np.float32).tolist()
                        ),
                        "context_duration_seconds": context_duration_seconds,
                    },
                )
            if isinstance(algorithm, CausalBayesSpeakerTracker):
                evidences: list[MultiScaleEvidence] = []
                if embedding is not None:
                    evidences.append(MultiScaleEvidence(
                        float(algorithm.config.scale_windows[0]),
                        np.asarray(embedding, dtype=np.float32),
                    ))
                configured_context = float(
                    getattr(self.args, "live_speaker_probe_context_window_seconds", 0.0)
                )
                if (
                    context_embedding is not None
                    and context_duration_seconds is not None
                    and float(context_duration_seconds) + 1e-6 >= configured_context
                ):
                    evidences.append(MultiScaleEvidence(
                        float(algorithm.config.scale_windows[-1]),
                        np.asarray(context_embedding, dtype=np.float32),
                    ))
                decision = algorithm.step(MultiScaleStep(
                    media_time=max(0.0, float(media_time)),
                    speech=bool(speech),
                    evidences=tuple(evidences),
                    probe_scheduled=bool(probe_scheduled),
                    release_signal=bool(release_signal),
                    skipped_reason=str(skipped_reason),
                ))
            else:
                decision = algorithm.step(LiveSpeakerStep(
                    media_time=max(0.0, float(media_time)),
                    speech=bool(speech),
                    embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
                    duration_seconds=max(0.0, float(duration_seconds)),
                    probe_scheduled=bool(probe_scheduled),
                    release_signal=bool(release_signal),
                    embedding_latency_seconds=embedding_latency_seconds,
                    skipped_reason=str(skipped_reason),
                ))
            overlay = self._open_set_tracklet_overlay()
            overlay_decision = None
            base_trace_record = decision.trace_record()
            if overlay is not None:
                overlay_decision = overlay.step(OpenSetTrackletStep(
                    media_time=max(0.0, float(media_time)),
                    speech=bool(speech),
                    probe_scheduled=bool(str(probe_id or "")),
                    release_signal=bool(release_signal),
                    short_embedding=(
                        None
                        if embedding is None
                        else np.asarray(embedding, dtype=np.float32)
                    ),
                    long_embedding=(
                        None
                        if context_embedding is None
                        else np.asarray(context_embedding, dtype=np.float32)
                    ),
                    profiles=tuple(dict(profile) for profile in profiles),
                    base_visible_speaker=decision.visible_speaker,
                    base_action=decision.action,
                    base_reason=decision.reason,
                ))
                diagnostics = {
                    **dict(decision.diagnostics),
                    "open_set_tracklet": dict(overlay_decision.diagnostics),
                    "open_set_tracklet_provisional_speaker": bool(
                        overlay_decision.provisional_speaker
                    ),
                    "open_set_tracklet_created_speaker": bool(
                        overlay_decision.created_speaker
                    ),
                    "open_set_tracklet_aliases": [
                        alias.payload() for alias in overlay_decision.aliases
                    ],
                    "base_live_speaker_decision": base_trace_record,
                }
                decision = replace(
                    decision,
                    visible_speaker=overlay_decision.visible_speaker,
                    candidate_speaker=overlay_decision.visible_speaker,
                    action=overlay_decision.action,
                    reason=overlay_decision.reason,
                    diagnostics=diagnostics,
                )
        if overlay_decision is not None:
            emit_internal = getattr(self.bus, "emit_internal", None)
            if callable(emit_internal):
                emit_internal(
                    "live_speaker_open_set_tracklet_decision",
                    {
                        **correlation,
                        "media_time": max(0.0, float(media_time)),
                        "base_decision": base_trace_record,
                        "projected_decision": decision.trace_record(),
                        "aliases": [
                            alias.payload() for alias in overlay_decision.aliases
                        ],
                        "effective_config": overlay.config.snapshot(),
                    },
                )
            for alias in overlay_decision.aliases:
                alias_payload = {
                    **correlation,
                    **alias.payload(),
                    "media_time": max(0.0, float(media_time)),
                    "final_speaker": {
                        **dict(alias.final_speaker),
                        **self._speaker_info_for_payload(
                            alias.final_internal_speaker_id
                        ),
                    },
                    "final_to_public": overlay.final_to_public,
                    "public_to_final": overlay.public_to_final,
                    "assignment_source": "open_set_tracklet_profile_merge",
                }
                self.bus.emit("live_speaker_identity_alias", alias_payload)
        trace_record = decision.trace_record()
        correlated_trace_record = {**trace_record, **correlation}
        self.bus.emit("live_speaker_shared_core_decision", correlated_trace_record)
        emit_internal = getattr(self.bus, "emit_internal", None)
        if callable(emit_internal):
            emit_internal(
                "live_speaker_core_decision",
                correlated_trace_record,
            )
        return decision

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
    def _public_live_speaker_label(
        label: str | None,
        profile_aliases: dict[str, str] | None,
    ) -> str | None:
        """Translate an internal provisional tracker id back to its finalized UI id."""

        if not label:
            return None
        value = str(label)
        for external_label, internal_label in (profile_aliases or {}).items():
            if str(internal_label) == value:
                return str(external_label)
        return value

    @classmethod
    def _public_live_speaker_values(
        cls,
        values: dict[str, float],
        profile_aliases: dict[str, str] | None,
        *,
        probability_keys: bool = False,
    ) -> dict[str, float]:
        public: dict[str, float] = {}
        for raw_label, raw_value in (values or {}).items():
            label = str(raw_label)
            if label != "unknown":
                label = str(cls._public_live_speaker_label(label, profile_aliases) or label)
                if probability_keys and label.startswith("S") and label[1:].isdigit():
                    label = f"speaker{int(label[1:])}"
            public[label] = max(float(raw_value), float(public.get(label, 0.0)))
        return public

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

    def _live_speaker_inference_lock_obj(self) -> threading.Lock:
        lock = getattr(self, "_live_speaker_inference_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._live_speaker_inference_lock = lock
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

    def _try_reserve_live_speaker_embedding(
        self,
        source: str = "unknown",
        *,
        run_id: str = "",
        probe_id: str = "",
        request_id: str = "",
    ) -> bool:
        now = time.monotonic()
        with self._live_speaker_embedding_lock():
            next_at = float(getattr(self, "_live_speaker_embedding_next_at", 0.0))
            admitted = now >= next_at
            if admitted:
                self._live_speaker_embedding_next_at = (
                    now + self._live_speaker_embedding_min_interval_seconds()
                )
            next_after = float(getattr(self, "_live_speaker_embedding_next_at", next_at))
            latency_ewma = getattr(self, "_live_speaker_embedding_latency_ewma", None)
        emit_internal = getattr(self.bus, "emit_internal", None)
        if callable(emit_internal):
            emit_internal(
                "live_speaker_embedding_admission",
                {
                    "run_id": str(run_id or self._live_speaker_correlation_run_id()),
                    "probe_id": str(probe_id or ""),
                    "request_id": str(request_id or ""),
                    "media_time": round(float(self.playback_time()), 6),
                    "source": str(source),
                    "admitted": bool(admitted),
                    "monotonic_request": float(now),
                    "monotonic_next_before": float(next_at),
                    "monotonic_next_after": float(next_after),
                    "min_interval_seconds": self._live_speaker_embedding_min_interval_seconds(),
                    "target_utilization": self._live_speaker_embedding_target_utilization(),
                    "latency_ewma_seconds": (
                        None if latency_ewma is None else float(latency_ewma)
                    ),
                    "skip_reason": "" if admitted else "shared_embedding_throttle",
                },
            )
        return admitted

    def _record_live_speaker_embedding_latency(
        self,
        elapsed_seconds: float,
        *,
        run_id: str = "",
        probe_id: str = "",
        request_id: str = "",
        source: str = "unknown",
    ) -> None:
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
            next_at = float(self._live_speaker_embedding_next_at)
        emit_internal = getattr(self.bus, "emit_internal", None)
        if callable(emit_internal):
            emit_internal(
                "live_speaker_embedding_throttle_update",
                {
                    "run_id": str(run_id or self._live_speaker_correlation_run_id()),
                    "probe_id": str(probe_id or ""),
                    "request_id": str(request_id or ""),
                    "source": str(source),
                    "media_time": round(float(self.playback_time()), 6),
                    "elapsed_seconds": elapsed,
                    "latency_ewma_seconds": float(latency),
                    "wait_seconds": float(wait_seconds),
                    "target_utilization": float(target_utilization),
                    "min_interval_seconds": float(min_interval),
                    "monotonic_next_at": next_at,
                },
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
            "run_id": fast_payload.get("run_id"),
            "probe_id": fast_payload.get("probe_id"),
            "request_id": fast_payload.get("request_id"),
            "step_id": fast_payload.get("step_id"),
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
        *,
        context_audio: np.ndarray | None = None,
        context_duration_seconds: float | None = None,
        context_weight: float = 0.0,
        request_source: str = "unknown",
        request_id: str = "",
        run_id: str = "",
        probe_id: str = "",
        short_window_start: float | None = None,
        short_window_end: float | None = None,
        short_source_start_sample: int | None = None,
        short_source_end_sample: int | None = None,
        context_window_start: float | None = None,
        context_window_end: float | None = None,
        context_source_start_sample: int | None = None,
        context_source_end_sample: int | None = None,
    ) -> dict[str, Any]:
        if not self._live_speaker_assignment_enabled():
            return self._realtime_unknown_speaker_payload()
        if min_audio_seconds is None:
            min_audio_seconds = float(self.args.realtime_preview_diarize_min_audio_seconds)
        if duration_seconds < max(0.0, float(min_audio_seconds)):
            return self._realtime_unknown_speaker_payload()
        memory = self.live_memory
        if (
            memory.profile_count() <= 0
            and not self._live_speaker_can_score_without_final_profiles()
        ):
            return self._realtime_unknown_speaker_payload()

        chunk = pad_audio(trim_silence(audio, self.sample_rate), self.args.min_embed_seconds, self.sample_rate)
        request_id = str(request_id or uuid.uuid4().hex)
        run_id = str(run_id or self._live_speaker_correlation_run_id())
        probe_id = str(probe_id or "")
        request_media_time = float(self.playback_time())
        actual_short_window_end = (
            request_media_time if short_window_end is None else float(short_window_end)
        )
        actual_short_window_start = (
            max(0.0, actual_short_window_end - float(duration_seconds))
            if short_window_start is None
            else max(0.0, float(short_window_start))
        )
        actual_short_start_sample = (
            max(0, int(actual_short_window_start * self.sample_rate))
            if short_source_start_sample is None
            else max(0, int(short_source_start_sample))
        )
        actual_short_end_sample = (
            actual_short_start_sample + int(audio.size)
            if short_source_end_sample is None
            else max(actual_short_start_sample, int(short_source_end_sample))
        )
        actual_context_window_end = (
            None
            if context_audio is None
            else (
                actual_short_window_end
                if context_window_end is None
                else float(context_window_end)
            )
        )
        actual_context_window_start = (
            None
            if context_audio is None
            else (
                max(
                    0.0,
                    float(actual_context_window_end)
                    - float(context_duration_seconds or 0.0),
                )
                if context_window_start is None
                else max(0.0, float(context_window_start))
            )
        )
        actual_context_start_sample = (
            None
            if actual_context_window_start is None
            else (
                max(0, int(actual_context_window_start * self.sample_rate))
                if context_source_start_sample is None
                else max(0, int(context_source_start_sample))
            )
        )
        actual_context_end_sample = (
            None
            if actual_context_start_sample is None or context_audio is None
            else (
                actual_context_start_sample + int(context_audio.size)
                if context_source_end_sample is None
                else max(actual_context_start_sample, int(context_source_end_sample))
            )
        )
        emit_internal = getattr(self.bus, "emit_internal", None)
        recording = callable(emit_internal) and bool(
            getattr(self.bus, "has_internal_listeners", lambda: False)()
        )
        if recording:
            emit_internal(
                "live_speaker_embedding_request_started",
                {
                    "run_id": run_id,
                    "probe_id": probe_id,
                    "request_id": request_id,
                    "source": str(request_source),
                    "source_sample_rate": int(self.sample_rate),
                    "media_time": round(request_media_time, 6),
                    "short_window_start": round(actual_short_window_start, 6),
                    "short_window_end": round(actual_short_window_end, 6),
                    "short_source_start_sample": actual_short_start_sample,
                    "short_source_end_sample": actual_short_end_sample,
                    "short_raw_sample_count": int(audio.size),
                    "short_raw_pcm_sha256": hashlib.sha256(
                        np.ascontiguousarray(audio, dtype=np.float32).tobytes()
                    ).hexdigest(),
                    "short_duration_seconds": float(duration_seconds),
                    "short_prepared_sample_count": int(chunk.size),
                    "short_prepared_pcm_sha256": hashlib.sha256(
                        np.ascontiguousarray(chunk, dtype=np.float32).tobytes()
                    ).hexdigest(),
                    "context_duration_seconds": (
                        None
                        if context_duration_seconds is None
                        else float(context_duration_seconds)
                    ),
                    "context_window_start": (
                        None
                        if actual_context_window_start is None
                        else round(float(actual_context_window_start), 6)
                    ),
                    "context_window_end": (
                        None
                        if actual_context_window_end is None
                        else round(float(actual_context_window_end), 6)
                    ),
                    "context_source_start_sample": actual_context_start_sample,
                    "context_source_end_sample": actual_context_end_sample,
                    "context_raw_sample_count": (
                        None if context_audio is None else int(context_audio.size)
                    ),
                    "context_raw_pcm_sha256": (
                        None
                        if context_audio is None
                        else hashlib.sha256(
                            np.ascontiguousarray(context_audio, dtype=np.float32).tobytes()
                        ).hexdigest()
                    ),
                    "requested_context_weight": max(0.0, min(1.0, float(context_weight))),
                    "provider": self._current_live_embedding_provider(),
                },
            )
        try:
            with self._live_speaker_inference_lock_obj():
                embed_started = time.monotonic()
                short_started = embed_started
                short_embedding = self._embed_live_audio_chunk(
                    chunk,
                    self.sample_rate,
                    ".live.short.wav",
                )
                embedding = short_embedding
                short_finished = time.monotonic()
                context_embedding: np.ndarray | None = None
                context_chunk: np.ndarray | None = None
                context_started: float | None = None
                context_finished: float | None = None
                applied_context_weight = 0.0
                effective_duration = float(duration_seconds)
                requested_context_weight = max(0.0, min(1.0, float(context_weight)))
                if context_audio is not None and requested_context_weight > 0.0:
                    context_chunk = pad_audio(
                        trim_silence(context_audio, self.sample_rate),
                        self.args.min_embed_seconds,
                        self.sample_rate,
                    )
                    context_started = time.monotonic()
                    context_embedding = self._embed_live_audio_chunk(
                        context_chunk, self.sample_rate, ".live.context.wav"
                    )
                    context_finished = time.monotonic()
                    if str(getattr(self.args, "live_speaker_tracker", "classic")) != "bayes":
                        embedding = blend_live_speaker_embeddings(
                            embedding, context_embedding, requested_context_weight
                        )
                    applied_context_weight = requested_context_weight
                    effective_duration = max(
                        effective_duration,
                        float(context_duration_seconds or 0.0),
                    )
            embedding_latency_seconds = time.monotonic() - embed_started
            decision_media_time = float(self.playback_time())
            if recording:
                emit_internal(
                    "live_speaker_embedding_request_completed",
                    {
                        "run_id": run_id,
                        "probe_id": probe_id,
                        "request_id": request_id,
                        "source": str(request_source),
                        "source_sample_rate": int(self.sample_rate),
                        "media_time": round(decision_media_time, 6),
                        "window_media_time": round(actual_short_window_end, 6),
                        "short_window_start": round(actual_short_window_start, 6),
                        "short_window_end": round(actual_short_window_end, 6),
                        "short_source_start_sample": actual_short_start_sample,
                        "short_source_end_sample": actual_short_end_sample,
                        "short_duration_seconds": float(duration_seconds),
                        "context_duration_seconds": (
                            None
                            if context_duration_seconds is None
                            else float(context_duration_seconds)
                        ),
                        "short_latency_seconds": max(0.0, short_finished - short_started),
                        "context_latency_seconds": (
                            None
                            if context_started is None or context_finished is None
                            else max(0.0, context_finished - context_started)
                        ),
                        "total_latency_seconds": embedding_latency_seconds,
                        "requested_context_weight": requested_context_weight,
                        "applied_context_weight": applied_context_weight,
                        "effective_duration_seconds": (
                            float(duration_seconds)
                            if str(getattr(self.args, "live_speaker_tracker", "classic"))
                            == "bayes"
                            else float(effective_duration)
                        ),
                        "tracker_mode": str(
                            getattr(self.args, "live_speaker_tracker", "classic")
                        ),
                        "short_embedding": np.asarray(
                            short_embedding,
                            dtype=np.float32,
                        ).tolist(),
                        "context_embedding": (
                            None
                            if context_embedding is None
                            else np.asarray(context_embedding, dtype=np.float32).tolist()
                        ),
                        "effective_embedding": np.asarray(
                            embedding,
                            dtype=np.float32,
                        ).tolist(),
                        "context_window_start": (
                            None
                            if actual_context_window_start is None
                            else round(float(actual_context_window_start), 6)
                        ),
                        "context_window_end": (
                            None
                            if actual_context_window_end is None
                            else round(float(actual_context_window_end), 6)
                        ),
                        "context_source_start_sample": actual_context_start_sample,
                        "context_source_end_sample": actual_context_end_sample,
                        "context_prepared_sample_count": (
                            None if context_chunk is None else int(context_chunk.size)
                        ),
                        "context_prepared_pcm_sha256": (
                            None
                            if context_chunk is None
                            else hashlib.sha256(
                                np.ascontiguousarray(context_chunk, dtype=np.float32).tobytes()
                            ).hexdigest()
                        ),
                        "provider": self._current_live_embedding_provider(),
                    },
                )
            self._record_live_speaker_embedding_latency(
                embedding_latency_seconds,
                run_id=run_id,
                probe_id=probe_id,
                request_id=request_id,
                source=request_source,
            )
            core_correlation: dict[str, Any] = {}
            core_decision = self._shared_live_speaker_step(
                media_time=decision_media_time,
                speech=True,
                embedding=embedding,
                duration_seconds=(
                    float(duration_seconds)
                    if str(getattr(self.args, "live_speaker_tracker", "classic")) == "bayes"
                    else effective_duration
                ),
                probe_scheduled=True,
                context_embedding=context_embedding,
                context_duration_seconds=context_duration_seconds,
                embedding_latency_seconds=embedding_latency_seconds,
                run_id=run_id,
                probe_id=probe_id,
                request_id=request_id,
                correlation_out=core_correlation,
            )
            record_handoff_evidence = getattr(
                self,
                "_record_sentence_handoff_live_evidence",
                None,
            )
            if callable(record_handoff_evidence):
                record_handoff_evidence(
                    request_source=str(request_source),
                    run_id=run_id,
                    probe_id=probe_id,
                    request_id=request_id,
                    window_start=actual_short_window_start,
                    window_end=actual_short_window_end,
                    source_start_sample=actual_short_start_sample,
                    source_end_sample=actual_short_end_sample,
                    audio=audio,
                    sample_rate=self.sample_rate,
                    embedding=short_embedding,
                    visible_speaker=getattr(core_decision, "visible_speaker", None),
                    similarities=getattr(core_decision, "similarities", None),
                    probabilities=getattr(core_decision, "probabilities", None),
                    profile_generations=getattr(
                        core_decision,
                        "profile_generations",
                        None,
                    ),
                    provider=self._current_live_embedding_provider(),
                )
            if recording:
                emit_internal(
                    "live_speaker_embedding_request_step_bound",
                    {
                        **core_correlation,
                        "source": str(request_source),
                        "window_media_time": round(actual_short_window_end, 6),
                    },
                )
        except Exception as exc:
            if recording:
                emit_internal(
                    "live_speaker_embedding_request_failed",
                    {
                        "run_id": run_id,
                        "probe_id": probe_id,
                        "request_id": request_id,
                        "source": str(request_source),
                        "media_time": round(float(self.playback_time()), 6),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            self.bus.emit("status", {"message": f"Realtime preview speaker scoring error: {type(exc).__name__}: {exc}"})
            return {
                **self._realtime_unknown_speaker_payload(),
                "run_id": run_id,
                "probe_id": probe_id,
                "request_id": request_id,
                "step_id": None,
            }

        diagnostics = getattr(core_decision, "diagnostics", {})
        profile_aliases = (
            dict(diagnostics.get("profile_aliases") or {})
            if isinstance(diagnostics, dict)
            else {}
        )
        internal_assigned_speaker = core_decision.visible_speaker
        assigned_speaker = self._public_live_speaker_label(
            internal_assigned_speaker,
            profile_aliases,
        )
        raw_probabilities = self._public_live_speaker_values(
            dict(core_decision.raw_probabilities),
            profile_aliases,
            probability_keys=True,
        )
        smoothed_probabilities = self._public_live_speaker_values(
            dict(core_decision.probabilities),
            profile_aliases,
            probability_keys=True,
        )
        public_similarities = self._public_live_speaker_values(
            dict(core_decision.similarities),
            profile_aliases,
        )
        provisional = bool(
            assigned_speaker
            and (
                str(assigned_speaker).startswith("provisional_")
                or str(assigned_speaker).startswith("LIVE_TRACKLET_")
                or bool(diagnostics.get("open_set_tracklet_provisional_speaker"))
            )
        )
        self._ensure_speaker_metadata(
            assigned_speaker,
            source="live_provisional" if provisional else "detected",
        )

        known_similarities = list(core_decision.similarities.values())
        ordered_similarities = sorted((float(value) for value in known_similarities), reverse=True)
        top_similarity = ordered_similarities[0] if ordered_similarities else None
        margin = (
            ordered_similarities[0] - ordered_similarities[1]
            if len(ordered_similarities) > 1
            else (1.0 if ordered_similarities else None)
        )

        return {
            "run_id": run_id,
            "probe_id": probe_id,
            "request_id": request_id,
            "step_id": core_correlation.get("step_id"),
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": bool(
                diagnostics.get("open_set_tracklet_created_speaker", False)
            ),
            "provisional_speaker": provisional,
            "internal_speaker_id": internal_assigned_speaker,
            "replaces_speaker_id": (
                internal_assigned_speaker
                if internal_assigned_speaker and assigned_speaker != internal_assigned_speaker
                else None
            ),
            "public_identity_aliases": dict(
                (diagnostics.get("open_set_tracklet") or {}).get(
                    "final_to_public", {}
                )
            ),
            "probabilities": smoothed_probabilities,
            "raw_probabilities": raw_probabilities,
            "similarities": public_similarities,
            "unknown_probability": float(smoothed_probabilities.get("unknown", 1.0)),
            "top_similarity": top_similarity,
            "margin": margin,
            "quality": None,
            "live_speaker_core_action": core_decision.action,
            "live_speaker_core_reason": core_decision.reason,
            "live_speaker_algorithm_id": LIVE_SPEAKER_ALGORITHM_ID,
            "live_speaker_short_window_seconds": round(float(duration_seconds), 4),
            "live_speaker_context_window_seconds": (
                round(float(context_duration_seconds), 4)
                if applied_context_weight > 0.0 and context_duration_seconds is not None
                else None
            ),
            "live_speaker_context_weight": applied_context_weight,
            "assignment_source": "shared_causal_live_speaker_core",
        }

    def _update_live_speaker_memory(
        self,
        speaker_id: str | None,
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
        suffix: str = ".live-profile.wav",
        speaker_generation: int | None = None,
        sentence_start: float | None = None,
        sentence_end: float | None = None,
        precomputed_embedding: np.ndarray | None = None,
        parent_job_id: str = "",
        source_run_id: str = "",
    ) -> None:
        if not getattr(self, "_live_embedding_separate", False) or not speaker_id:
            return
        copied_precomputed_embedding = (
            None
            if precomputed_embedding is None
            else np.asarray(precomputed_embedding, dtype=np.float32).copy()
        )
        job = LiveSpeakerMemoryUpdateJob(
            speaker_id=str(speaker_id),
            audio=(
                np.asarray(audio, dtype=np.float32).copy()
                if copied_precomputed_embedding is None
                else np.empty(0, dtype=np.float32)
            ),
            sample_rate=int(sample_rate),
            duration_seconds=float(duration_seconds),
            job_id=f"live-profile-{uuid.uuid4().hex}",
            parent_job_id=str(parent_job_id or ""),
            suffix=suffix,
            speaker_generation=(
                int(getattr(self, "_speaker_generation", 0))
                if speaker_generation is None
                else int(speaker_generation)
            ),
            speaker_label_generation=int(
                getattr(self, "_speaker_label_generations", {}).get(str(speaker_id), 0)
            ),
            run_id=str(source_run_id or self._live_speaker_correlation_run_id()),
            sentence_start=sentence_start,
            sentence_end=sentence_end,
            precomputed_embedding=copied_precomputed_embedding,
            queued_monotonic=time.monotonic(),
            queued_media_time=float(self.playback_time()),
        )
        jobs = getattr(self, "_live_memory_update_jobs", None)
        self._record_live_profile_queue_stage(
            job,
            "submitted",
            execution_mode="inline" if jobs is None else "worker_queue",
        )
        if jobs is None:
            self._process_live_speaker_memory_update(job)
            return
        try:
            jobs.put_nowait(job)
            self._record_live_profile_queue_stage(
                job,
                "queued",
                queue_size=int(jobs.qsize()),
                queue_capacity=int(getattr(jobs, "maxsize", 0)),
            )
        except queue.Full:
            dropped_job: LiveSpeakerMemoryUpdateJob | None = None
            try:
                candidate = jobs.get_nowait()
                if isinstance(candidate, LiveSpeakerMemoryUpdateJob):
                    dropped_job = candidate
            except queue.Empty:
                pass
            else:
                jobs.task_done()
                if dropped_job is not None:
                    self._record_live_profile_queue_stage(
                        dropped_job,
                        "dropped",
                        reason="queue_full_evicted_oldest",
                    )
                self.bus.emit(
                    "status",
                    {"message": "Dropped stale queued live speaker profile update because the queue is full."},
                )
            try:
                jobs.put_nowait(job)
                self._record_live_profile_queue_stage(
                    job,
                    "queued",
                    reason="queued_after_oldest_eviction",
                    queue_size=int(jobs.qsize()),
                    queue_capacity=int(getattr(jobs, "maxsize", 0)),
                )
            except queue.Full:
                self._record_live_profile_queue_stage(
                    job,
                    "dropped",
                    reason="queue_still_full",
                )
                self.bus.emit(
                    "status",
                    {"message": "Skipped live speaker profile update because the queue is still full."},
                )

    def _record_live_profile_queue_stage(
        self,
        job: LiveSpeakerMemoryUpdateJob,
        stage: str,
        **details: Any,
    ) -> None:
        emit_internal = getattr(self.bus, "emit_internal", None)
        if not callable(emit_internal):
            return
        emit_internal(
            "live_profile_queue_lifecycle",
            {
                "job_id": str(getattr(job, "job_id", "") or ""),
                "run_id": str(getattr(job, "run_id", "") or ""),
                "parent_job_id": str(getattr(job, "parent_job_id", "") or ""),
                "stage": str(stage),
                "speaker_id": str(getattr(job, "speaker_id", "") or ""),
                "speaker_generation": int(getattr(job, "speaker_generation", 0)),
                "speaker_label_generation": int(
                    getattr(job, "speaker_label_generation", 0)
                ),
                "queued_media_time": float(getattr(job, "queued_media_time", 0.0)),
                "media_time": round(float(self.playback_time()), 6),
                "sentence_start": getattr(job, "sentence_start", None),
                "sentence_end": getattr(job, "sentence_end", None),
                **details,
            },
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
        process_started = time.monotonic()
        emit_internal = getattr(self.bus, "emit_internal", None)
        try:
            if not self._live_memory_update_job_is_current(job):
                self._record_live_profile_queue_stage(
                    job,
                    "cancelled",
                    reason="stale_job_before_embedding",
                )
                if callable(emit_internal):
                    emit_internal(
                        "live_profile_embedding_skipped",
                        {
                            "job_id": job.job_id,
                            "run_id": job.run_id,
                            "parent_job_id": job.parent_job_id,
                            "speaker_id": job.speaker_id,
                            "media_time": round(float(self.playback_time()), 6),
                            "reason": "stale_job_before_embedding",
                        },
                    )
                return
            if not self._live_update_speaker_exists(job.speaker_id):
                self._record_live_profile_queue_stage(
                    job,
                    "cancelled",
                    reason="speaker_missing_before_embedding",
                )
                if callable(emit_internal):
                    emit_internal(
                        "live_profile_embedding_skipped",
                        {
                            "job_id": job.job_id,
                            "run_id": job.run_id,
                            "parent_job_id": job.parent_job_id,
                            "speaker_id": job.speaker_id,
                            "media_time": round(float(self.playback_time()), 6),
                            "reason": "speaker_missing_before_embedding",
                        },
                    )
                return
            embedding = job.precomputed_embedding
            used_precomputed = embedding is not None
            self._record_live_profile_queue_stage(
                job,
                "started",
                queue_wait_seconds=max(
                    0.0,
                    process_started - float(getattr(job, "queued_monotonic", process_started)),
                ),
                precomputed=bool(used_precomputed),
            )
            if callable(emit_internal):
                emit_internal(
                    "live_profile_embedding_started",
                    {
                        "job_id": job.job_id,
                        "run_id": job.run_id,
                        "parent_job_id": job.parent_job_id,
                        "speaker_id": job.speaker_id,
                        "media_time": round(float(self.playback_time()), 6),
                        "queued_media_time": float(getattr(job, "queued_media_time", 0.0)),
                        "queue_wait_seconds": max(
                            0.0,
                            process_started - float(
                                getattr(job, "queued_monotonic", process_started)
                            ),
                        ),
                        "sentence_start": job.sentence_start,
                        "sentence_end": job.sentence_end,
                        "duration_seconds": float(job.duration_seconds),
                        "provider": self._current_live_embedding_provider(),
                        "precomputed": bool(used_precomputed),
                    },
                )
            if embedding is None:
                embedding = self._embed_live_audio_chunk(job.audio, job.sample_rate, job.suffix)
            else:
                embedding = np.asarray(embedding, dtype=np.float32)
            embedding_finished = time.monotonic()
            if callable(emit_internal):
                emit_internal(
                    "live_profile_embedding_completed",
                    {
                        "job_id": job.job_id,
                        "run_id": job.run_id,
                        "parent_job_id": job.parent_job_id,
                        "speaker_id": job.speaker_id,
                        "media_time": round(float(self.playback_time()), 6),
                        "sentence_start": job.sentence_start,
                        "sentence_end": job.sentence_end,
                        "duration_seconds": float(job.duration_seconds),
                        "latency_seconds": max(0.0, embedding_finished - process_started),
                        "provider": self._current_live_embedding_provider(),
                        "precomputed": bool(used_precomputed),
                        "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                    },
                )
            with self._live_memory_update_lock_obj():
                if not self._live_memory_update_job_is_current(job):
                    self._record_live_profile_queue_stage(
                        job,
                        "cancelled",
                        reason="stale_job_after_embedding",
                    )
                    if callable(emit_internal):
                        emit_internal(
                            "live_profile_embedding_skipped",
                            {
                                "job_id": job.job_id,
                                "run_id": job.run_id,
                                "parent_job_id": job.parent_job_id,
                                "speaker_id": job.speaker_id,
                                "media_time": round(float(self.playback_time()), 6),
                                "reason": "stale_job_after_embedding",
                            },
                        )
                    return
                if not self._live_update_speaker_exists(job.speaker_id):
                    self._record_live_profile_queue_stage(
                        job,
                        "cancelled",
                        reason="speaker_missing_after_embedding",
                    )
                    if callable(emit_internal):
                        emit_internal(
                            "live_profile_embedding_skipped",
                            {
                                "job_id": job.job_id,
                                "run_id": job.run_id,
                                "parent_job_id": job.parent_job_id,
                                "speaker_id": job.speaker_id,
                                "media_time": round(float(self.playback_time()), 6),
                                "reason": "speaker_missing_after_embedding",
                            },
                        )
                    return
                self.live_memory.upsert_profile(
                    job.speaker_id,
                    embedding,
                    duration_seconds=job.duration_seconds,
                    sentence_count=1,
                )
                emit_live_profile_snapshot(
                    self,
                    self.live_memory,
                    job.speaker_id,
                    self._current_live_embedding_provider(),
                    source="async_live_sentence_reembedding_complete",
                    sentence_start=job.sentence_start,
                    sentence_end=job.sentence_end,
                )
                if callable(emit_internal):
                    emit_internal(
                        "live_profile_memory_replace",
                        {
                            "job_id": job.job_id,
                            "run_id": job.run_id,
                            "parent_job_id": job.parent_job_id,
                            "speaker_id": job.speaker_id,
                            "media_time": round(float(self.playback_time()), 6),
                            "sentence_start": job.sentence_start,
                            "sentence_end": job.sentence_end,
                            "provider": self._current_live_embedding_provider(),
                            "profiles": self.live_memory.export_profiles(),
                        },
                    )
                self._record_live_profile_queue_stage(
                    job,
                    "completed",
                    latency_seconds=max(0.0, time.monotonic() - process_started),
                )
        except Exception as exc:
            self._record_live_profile_queue_stage(
                job,
                "failed",
                reason=type(exc).__name__,
                error=str(exc),
            )
            if callable(emit_internal):
                emit_internal(
                    "live_profile_embedding_failed",
                    {
                        "job_id": job.job_id,
                        "run_id": job.run_id,
                        "parent_job_id": job.parent_job_id,
                        "speaker_id": job.speaker_id,
                        "media_time": round(float(self.playback_time()), 6),
                        "sentence_start": job.sentence_start,
                        "sentence_end": job.sentence_end,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Live speaker profile update failed "
                        f"for {job.speaker_id}: {type(exc).__name__}: {exc}"
                    )
                },
            )
