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




class WindowAssignmentDecisionMixin:
    def _remember_unknown_sentence(
        self,
        index: int,
        base_payload: dict[str, Any],
        embedding: np.ndarray,
        duration_seconds: float,
    ) -> None:
        pending = PendingUnknownSentence(
            index=index,
            base_payload=dict(base_payload),
            embedding=embedding.astype(np.float32, copy=True),
            duration_seconds=duration_seconds,
        )
        with self._unknown_lock:
            self._unknown_sentences = [
                item for item in self._unknown_sentences
                if item.index != index
            ]
            self._unknown_sentences.append(pending)
            self._recent_unknown_pair_queue().append(pending)

    def _remove_unknown_sentence(self, index: int) -> bool:
        with self._unknown_lock:
            old_count = len(self._unknown_sentences)
            self._unknown_sentences = [
                item for item in self._unknown_sentences
                if item.index != index
            ]
            recent_queue = self._recent_unknown_pair_queue()
            if recent_queue:
                recent_queue = deque(
                    (item for item in recent_queue if item.index != index),
                    maxlen=recent_queue.maxlen,
                )
                self._recent_unknown_pair_candidates = recent_queue
            return len(self._unknown_sentences) != old_count

    def _clear_sentence_refinement_records(self) -> None:
        with self._sentence_refinement_lock:
            self._sentence_refinement_records = {}
        self._correction_history = []

    def _speaker_refinement_config(self) -> SpeakerRefinementConfig:
        short_distinct_active = self._has_short_distinct_speaker_record()
        return SpeakerRefinementConfig(
            max_per_profile=int(getattr(self.args, "speaker_refinement_max_per_profile", 32)),
            prototype_min_duration=float(getattr(self.args, "speaker_refinement_min_duration", 0.15)),
            prototype_max_unknown=float(getattr(self.args, "speaker_refinement_max_unknown", 1.0)),
            top_k=int(getattr(self.args, "speaker_refinement_top_k", 12)),
            centroid_blend=float(getattr(self.args, "speaker_refinement_centroid_blend", 0.555)),
            unknown_min_similarity=float(getattr(self.args, "speaker_refinement_unknown_min_similarity", 0.20)),
            unknown_min_margin=float(getattr(self.args, "speaker_refinement_unknown_min_margin", 0.0)),
            unknown_short_max_duration=min(
                1.0 if short_distinct_active else 0.0,
                float(getattr(self.args, "min_new_speaker_seconds", 2.0358)),
            ),
            unknown_short_min_similarity=max(
                float(getattr(self.args, "same_speaker_similarity", 0.45)),
                float(getattr(self.args, "known_speaker_min_similarity", -1.0)),
            ),
            known_max_duration=float(getattr(self.args, "speaker_refinement_known_max_duration", 8.0)),
            known_min_similarity=float(getattr(self.args, "speaker_refinement_known_min_similarity", -0.039)),
            known_min_delta=float(getattr(self.args, "speaker_refinement_known_min_delta", 0.04)),
        )

    def _record_sentence_assignment(
        self,
        index: int,
        base_payload: dict[str, Any],
        embedding: np.ndarray,
        duration_seconds: float,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "index": int(index),
            "base_payload": dict(base_payload),
            "embedding": embedding.astype(np.float32, copy=True),
            "duration_seconds": float(duration_seconds),
            "assigned_speaker": payload.get("assigned_speaker"),
            "created_speaker": bool(payload.get("created_speaker")),
            "probabilities": dict(payload.get("probabilities") or {}),
            "similarities": dict(payload.get("similarities") or {}),
            "unknown_probability": payload.get("unknown_probability"),
            "top_similarity": payload.get("top_similarity"),
            "margin": payload.get("margin"),
            "quality": payload.get("quality"),
            "assignment_source": str(payload.get("assignment_source") or ""),
        }
        session_state = getattr(self, "_session_state", None)
        transaction = session_state.transaction(mutate=True) if session_state is not None else nullcontext()
        with transaction, self._sentence_refinement_lock:
            previous_record = self._sentence_refinement_records.get(int(index)) or {}
            try:
                distinct_gate_enabled = (
                    float(
                        getattr(
                            self.args,
                            "short_distinct_new_speaker_min_spoken_seconds",
                            -1.0,
                        )
                    )
                    >= 0.0
                )
                current_start = float(base_payload.get("start"))
                previous_end = max(
                    (
                        float(
                            (existing.get("base_payload") or {}).get("end")
                        )
                        for existing_index, existing in
                        self._sentence_refinement_records.items()
                        if int(existing_index) != int(index)
                        and (existing.get("base_payload") or {}).get("end")
                        is not None
                    ),
                    default=float("-inf"),
                )
                previous_gap = current_start - previous_end
                protected_new_speaker_creation = (
                    distinct_gate_enabled
                    and bool(payload.get("created_speaker"))
                    and math.isfinite(previous_gap)
                    and previous_gap >= 8.0
                )
            except (TypeError, ValueError):
                protected_new_speaker_creation = False
            record["short_distinct_origin"] = bool(
                previous_record.get("short_distinct_origin")
                or record["assignment_source"] == "short_distinct_new_speaker"
                or protected_new_speaker_creation
            )
            self._sentence_refinement_records[int(index)] = record
            assigned_speaker = str(payload.get("assigned_speaker") or "")
            if assigned_speaker:
                try:
                    end = float(base_payload.get("end"))
                except (TypeError, ValueError):
                    end = float("nan")
                if math.isfinite(end):
                    self._speaker_last_media_end[assigned_speaker] = max(
                        end,
                        float(self._speaker_last_media_end.get(assigned_speaker, 0.0)),
                    )

    def _has_short_distinct_speaker_record(self) -> bool:
        with self._sentence_refinement_lock:
            return any(
                bool(record.get("short_distinct_origin"))
                or str(record.get("assignment_source") or "")
                == "short_distinct_new_speaker"
                for record in self._sentence_refinement_records.values()
            )

    def _clear_weak_short_unknown_provisionals(self) -> int:
        try:
            min_similarity = max(
                float(getattr(self.args, "same_speaker_similarity", 0.45)),
                float(getattr(self.args, "known_speaker_min_similarity", -1.0)),
            )
        except (TypeError, ValueError):
            min_similarity = 0.45
        emitted: list[dict[str, Any]] = []
        with self._sentence_refinement_lock:
            for record in self._sentence_refinement_records.values():
                provisional = str(
                    record.get("provisional_assigned_speaker") or ""
                )
                if (
                    not provisional
                    or record.get("assigned_speaker") is not None
                    or self._record_speech_evidence_duration(record) >= 1.0
                ):
                    continue
                try:
                    top_similarity = float(record.get("top_similarity"))
                except (TypeError, ValueError):
                    top_similarity = -1.0
                if top_similarity >= min_similarity:
                    continue
                record["provisional_assigned_speaker"] = None
                record["provisional_probabilities"] = {}
                record["provisional_similarities"] = {}
                record["provisional_assignment_source"] = ""
                emitted.append({
                    **dict(record["base_payload"]),
                    "pending": False,
                    "revision": True,
                    "short_distinct_reconsideration": True,
                    "revision_from": provisional,
                    "revision_to": "UNKNOWN",
                    "assigned_speaker": None,
                    **self._speaker_info_for_payload(None),
                    "created_speaker": False,
                    "probabilities": dict(record.get("probabilities") or {"unknown": 1.0}),
                    "similarities": dict(record.get("similarities") or {}),
                    "unknown_probability": record.get("unknown_probability"),
                    "top_similarity": record.get("top_similarity"),
                    "margin": record.get("margin"),
                    "quality": record.get("quality"),
                    "assignment_source": "short_distinct_reconsideration",
                })
        for payload in emitted:
            self._emit_transcript_sentence(payload)
        return len(emitted)

    def _record_unknown_refinement_candidate(
        self,
        index: int,
        base_payload: dict[str, Any],
        duration_seconds: float,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "index": int(index),
            "base_payload": dict(base_payload),
            "embedding": None,
            "duration_seconds": float(duration_seconds),
            "assigned_speaker": payload.get("assigned_speaker"),
            "created_speaker": False,
            "probabilities": dict(payload.get("probabilities") or {"unknown": 1.0}),
            "similarities": dict(payload.get("similarities") or {}),
            "unknown_probability": payload.get("unknown_probability", 1.0),
            "top_similarity": payload.get("top_similarity"),
            "margin": payload.get("margin"),
            "quality": payload.get("quality"),
            "assignment_source": str(payload.get("assignment_source") or "unknown"),
        }
        with self._sentence_refinement_lock:
            self._sentence_refinement_records[int(index)] = record

    def _next_detected_speaker_label(self) -> str:
        max_index = 0
        for profile in self.memory.export_profiles():
            label = str(profile.get("label") or "").strip().upper()
            if label.startswith("S") and label[1:].isdigit():
                max_index = max(max_index, int(label[1:]))
        return f"S{max_index + 1}"

    def _section_gap_new_speaker_decision(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        base_payload: dict[str, Any],
        *,
        allow_new_speaker: bool,
    ) -> SpeakerDecision | None:
        if not allow_new_speaker or not bool(getattr(self.args, "section_gap_new_speaker", False)):
            return None
        if self.memory.profile_count() <= 0:
            return None
        try:
            start = float(base_payload.get("start"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(start):
            return None
        try:
            min_gap = max(0.0, float(getattr(self.args, "section_gap_new_speaker_min_gap_seconds", 60.0)))
            min_prior = max(
                0.0,
                float(getattr(self.args, "section_gap_new_speaker_min_prior_speech_seconds", 8.0)),
            )
            min_duration = max(
                0.0,
                float(getattr(self.args, "section_gap_new_speaker_min_duration_seconds", 5.0)),
            )
            min_similarity = float(getattr(self.args, "section_gap_new_speaker_min_similarity", 0.35))
            max_similarity = float(getattr(self.args, "section_gap_new_speaker_max_similarity", 0.58))
            min_margin = max(0.0, float(getattr(self.args, "section_gap_new_speaker_min_margin", 0.08)))
        except (TypeError, ValueError):
            return None
        if duration_seconds < min_duration:
            return None
        preview = self.memory.score_existing(
            embedding,
            duration_seconds,
            min_similarity=-1.0,
            min_margin=0.0,
        )
        similarities = dict(preview.similarities or {})
        if not similarities:
            return None
        top_speaker, top_similarity = max(similarities.items(), key=lambda item: float(item[1]))
        top_speaker = str(top_speaker)
        top_similarity = float(top_similarity)
        ordered = sorted((float(value) for value in similarities.values()), reverse=True)
        runner_up = ordered[1] if len(ordered) > 1 else -1.0
        margin = top_similarity - runner_up if len(ordered) > 1 else 1.0
        if top_similarity < min_similarity or top_similarity >= max_similarity or margin < min_margin:
            return None
        previous_end = self._speaker_last_media_end.get(top_speaker)
        if previous_end is None or start - previous_end < min_gap:
            return None
        prior_seconds = 0.0
        for profile in self.memory.export_profiles():
            if str(profile.get("label") or "") != top_speaker:
                continue
            try:
                prior_seconds = max(0.0, float(profile.get("speech_seconds") or 0.0))
            except (TypeError, ValueError):
                prior_seconds = 0.0
            break
        if prior_seconds < min_prior:
            return None
        label = self._next_detected_speaker_label()
        self.memory.upsert_profile(label, embedding, duration_seconds=duration_seconds, sentence_count=1)
        speaker_key = f"speaker{int(label[1:])}" if label.startswith("S") and label[1:].isdigit() else label
        return SpeakerDecision(
            assigned_speaker=label,
            created_speaker=True,
            probabilities={"unknown": 0.0, speaker_key: 1.0},
            similarities={**similarities, label: 1.0},
            unknown_probability=0.0,
            top_similarity=round(float(top_similarity), 4),
            margin=round(float(margin), 4),
            quality=preview.quality,
            assignment_source="section_gap_new_speaker",
        )

    def _short_distinct_new_speaker_decision(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        base_payload: dict[str, Any],
        *,
        allow_new_speaker: bool,
    ) -> SpeakerDecision | None:
        """Create a nearly-long-enough speaker only from decisive voice evidence."""
        profile_count = self.memory.profile_count()
        if not allow_new_speaker or profile_count <= 0:
            return None
        try:
            min_spoken_seconds = float(
                getattr(
                    self.args,
                    "short_distinct_new_speaker_min_spoken_seconds",
                    -1.0,
                )
            )
            min_words = max(
                1,
                int(getattr(self.args, "short_distinct_new_speaker_min_words", 6)),
            )
            min_unknown_probability = float(
                getattr(
                    self.args,
                    "short_distinct_new_speaker_min_unknown_probability",
                    0.90,
                )
            )
            max_similarity = float(
                getattr(
                    self.args,
                    "short_distinct_new_speaker_max_similarity",
                    0.20,
                )
            )
            max_margin = max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "short_distinct_new_speaker_max_margin",
                        0.05,
                    )
                ),
            )
            ordinary_min_duration = max(
                0.0,
                float(getattr(self.args, "min_new_speaker_seconds", 2.0358)),
            )
            spoken_seconds = max(
                0.0,
                float(base_payload.get("spoken_word_seconds") or 0.0),
            )
            anchor_words = max(
                0,
                int(base_payload.get("new_speaker_anchor_words") or 0),
            )
            confirmation_count = max(
                1,
                int(getattr(self.args, "new_speaker_confirmation_count", 1)),
            )
            max_speakers = max(1, int(getattr(self.args, "max_speakers", 12)))
        except (TypeError, ValueError):
            return None
        if (
            min_spoken_seconds < 0.0
            or ordinary_min_duration <= 0.0
            or duration_seconds >= ordinary_min_duration
            or duration_seconds < max(1.0, ordinary_min_duration * 0.95)
            or spoken_seconds < min_spoken_seconds
            or anchor_words < min_words
            or confirmation_count != 1
            or profile_count >= max_speakers
        ):
            return None

        preview = self.memory.score_existing(
            embedding,
            duration_seconds,
            min_similarity=-1.0,
            min_margin=-1.0,
        )
        try:
            unknown_probability = float(preview.unknown_probability)
            top_similarity = float(preview.top_similarity)
            margin = float(preview.margin)
        except (TypeError, ValueError):
            return None
        if (
            not all(
                math.isfinite(value)
                for value in (unknown_probability, top_similarity, margin)
            )
            or unknown_probability < min_unknown_probability
            or top_similarity > max_similarity
            or profile_count > 1
            and margin > max_margin
        ):
            return None

        label = self._next_detected_speaker_label()
        self.memory.upsert_profile(
            label,
            embedding,
            duration_seconds=duration_seconds,
            sentence_count=1,
        )
        speaker_key = (
            f"speaker{int(label[1:])}"
            if label.startswith("S") and label[1:].isdigit()
            else label
        )
        return SpeakerDecision(
            assigned_speaker=label,
            created_speaker=True,
            probabilities={"unknown": 0.0, speaker_key: 1.0},
            similarities={
                **dict(preview.similarities or {}),
                label: 1.0,
            },
            unknown_probability=0.0,
            top_similarity=round(top_similarity, 4),
            margin=round(margin, 4),
            quality=preview.quality,
            assignment_source="short_distinct_new_speaker",
        )

    def _unknown_pair_new_speaker_decision(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        base_payload: dict[str, Any],
        existing_decision: SpeakerDecision,
        *,
        allow_new_speaker: bool,
    ) -> tuple[SpeakerDecision, PendingUnknownSentence, float] | None:
        if not allow_new_speaker or not bool(getattr(self.args, "unknown_pair_new_speaker", False)):
            return None
        if not bool(getattr(self.args, "speaker_refinement_unknown_commit", True)):
            return None
        if self.memory.profile_count() <= 0:
            return None
        try:
            current_start = float(base_payload.get("start"))
            min_duration = max(
                0.0,
                float(getattr(self.args, "unknown_pair_new_speaker_min_current_duration_seconds", 2.5)),
            )
            max_gap = max(
                0.0,
                float(getattr(self.args, "unknown_pair_new_speaker_max_gap_seconds", 4.0)),
            )
            min_unknown_duration = max(
                0.0,
                float(getattr(self.args, "unknown_pair_new_speaker_min_unknown_duration_seconds", 0.2)),
            )
            min_pair_similarity = float(getattr(self.args, "unknown_pair_new_speaker_min_pair_similarity", 0.45))
            max_existing_similarity = float(
                getattr(self.args, "unknown_pair_new_speaker_max_existing_similarity", 0.55)
            )
            max_existing_margin = max(
                0.0,
                float(getattr(self.args, "unknown_pair_new_speaker_max_existing_margin", 0.20)),
            )
            min_unknown_probability = max(
                0.0,
                float(getattr(self.args, "unknown_pair_new_speaker_min_unknown_probability", 0.10)),
            )
        except (TypeError, ValueError):
            return None
        if not math.isfinite(current_start) or duration_seconds < min_duration:
            return None
        try:
            top_similarity = float(existing_decision.top_similarity)
        except (TypeError, ValueError):
            top_similarity = 1.0
        try:
            margin = float(existing_decision.margin)
        except (TypeError, ValueError):
            margin = 1.0
        try:
            unknown_probability = float(existing_decision.unknown_probability)
        except (TypeError, ValueError):
            unknown_probability = 0.0
        if (
            top_similarity > max_existing_similarity
            or margin > max_existing_margin
            or unknown_probability < min_unknown_probability
        ):
            return None

        with self._unknown_lock:
            candidates = list(self._unknown_sentences)
            recent_pair_candidates = list(self._recent_unknown_pair_queue())
        seen_candidate_indexes = {candidate.index for candidate in candidates}
        candidates.extend(
            candidate
            for candidate in recent_pair_candidates
            if candidate.index not in seen_candidate_indexes
        )
        best_candidate: PendingUnknownSentence | None = None
        best_similarity = -1.0
        best_gap = 0.0
        for candidate in candidates:
            if candidate.duration_seconds < min_unknown_duration:
                continue
            try:
                candidate_end = float(candidate.base_payload.get("end"))
            except (TypeError, ValueError):
                continue
            gap_seconds = current_start - candidate_end
            if gap_seconds < 0.0 or gap_seconds > max_gap:
                continue
            try:
                pair_similarity = cosine_similarity(embedding, candidate.embedding)
            except Exception:
                continue
            if pair_similarity > best_similarity:
                best_candidate = candidate
                best_similarity = float(pair_similarity)
                best_gap = float(gap_seconds)
        if best_candidate is None or best_similarity < min_pair_similarity:
            return None

        label = self._next_detected_speaker_label()
        combined_embedding = embedding + best_candidate.embedding
        self.memory.upsert_profile(
            label,
            combined_embedding,
            duration_seconds=duration_seconds + best_candidate.duration_seconds,
            sentence_count=2,
        )
        self._remove_unknown_sentence(best_candidate.index)
        speaker_key = f"speaker{int(label[1:])}" if label.startswith("S") and label[1:].isdigit() else label
        return (
            SpeakerDecision(
                assigned_speaker=label,
                created_speaker=True,
                probabilities={"unknown": 0.0, speaker_key: 1.0},
                similarities={
                    **dict(existing_decision.similarities or {}),
                    label: round(float(best_similarity), 4),
                },
                unknown_probability=round(float(unknown_probability), 4),
                top_similarity=round(float(top_similarity), 4),
                margin=round(float(margin), 4),
                quality=existing_decision.quality,
                assignment_source="unknown_pair_new_speaker",
            ),
            best_candidate,
            round(float(best_similarity), 4),
        )

    def _emit_unknown_pair_revision(
        self,
        candidate: PendingUnknownSentence,
        decision: SpeakerDecision,
        pair_similarity: float,
    ) -> None:
        if not decision.assigned_speaker:
            return
        with self._sentence_refinement_lock:
            record = self._sentence_refinement_records.get(int(candidate.index)) or {}
            reassigned_from = str(record.get("provisional_assigned_speaker") or "UNKNOWN")
            quality = record.get("quality", decision.quality)
        speaker_key = (
            f"speaker{int(decision.assigned_speaker[1:])}"
            if decision.assigned_speaker.startswith("S") and decision.assigned_speaker[1:].isdigit()
            else decision.assigned_speaker
        )
        similarities = dict(decision.similarities or {})
        similarities[decision.assigned_speaker] = round(float(pair_similarity), 4)
        payload = {
            **candidate.base_payload,
            "pending": False,
            "revision": True,
            "unknown_pair_reassigned": True,
            "revision_from": reassigned_from,
            "revision_to": decision.assigned_speaker,
            "assigned_speaker": decision.assigned_speaker,
            **self._speaker_info_for_payload(decision.assigned_speaker),
            "created_speaker": False,
            "probabilities": {"unknown": 0.0, speaker_key: 1.0},
            "similarities": similarities,
            "unknown_probability": 0.0,
            "top_similarity": round(float(pair_similarity), 4),
            "margin": decision.margin,
            "quality": quality,
            "assignment_source": "unknown_pair_new_speaker",
            "pair_similarity": round(float(pair_similarity), 4),
        }
        self._record_sentence_assignment(
            candidate.index,
            candidate.base_payload,
            candidate.embedding,
            candidate.duration_seconds,
            payload,
        )

    def _delayed_clustering_config(self) -> DelayedClusteringConfig:
        return DelayedClusteringConfig(
            core_max_unknown=float(getattr(self.args, "delayed_clustering_core_max_unknown", 0.50)),
            core_min_duration=float(getattr(self.args, "delayed_clustering_core_min_duration", 0.80)),
            min_core_rows=int(getattr(self.args, "delayed_clustering_min_core_rows", 4)),
            min_core_duration=float(getattr(self.args, "delayed_clustering_min_core_duration", 8.0)),
            candidate_min_unknown=float(
                getattr(self.args, "delayed_clustering_candidate_min_unknown", 0.50)
            ),
            candidate_min_duration=float(
                getattr(self.args, "delayed_clustering_candidate_min_duration", 0.35)
            ),
            candidate_max_core_similarity=float(
                getattr(self.args, "delayed_clustering_candidate_max_core_similarity", 0.45)
            ),
            candidate_min_similarity=float(
                getattr(self.args, "delayed_clustering_candidate_min_similarity", 0.20)
            ),
            candidate_min_gain=float(getattr(self.args, "delayed_clustering_candidate_min_gain", 0.02)),
            min_candidate_rows=int(getattr(self.args, "delayed_clustering_min_candidate_rows", 4)),
            min_candidate_duration=float(
                getattr(self.args, "delayed_clustering_min_candidate_duration", 8.0)
            ),
            min_candidate_anchor_duration=float(
                getattr(
                    self.args,
                    "delayed_clustering_min_candidate_anchor_duration",
                    0.0,
                )
            ),
            min_candidate_span=float(getattr(self.args, "delayed_clustering_min_candidate_span", 12.0)),
            min_candidate_time_groups=int(
                getattr(self.args, "delayed_clustering_min_candidate_time_groups", 2)
            ),
            time_group_gap=float(getattr(self.args, "delayed_clustering_time_group_gap", 8.0)),
            min_average_gain=float(getattr(self.args, "delayed_clustering_min_average_gain", 0.22)),
            min_leave_one_out_similarity=float(
                getattr(self.args, "delayed_clustering_min_leave_one_out_similarity", 0.16)
            ),
            max_core_centroid_similarity=float(
                getattr(self.args, "delayed_clustering_max_core_centroid_similarity", 0.58)
            ),
        )
        self._emit_transcript_sentence(payload)

    @staticmethod
    def _prototype_probabilities(assigned_speaker: str, scores: dict[str, float]) -> dict[str, float]:
        probabilities = {"unknown": 0.0}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return probabilities
        best = max(score for _, score in ordered)
        exps = [
            (label, float(np.exp((score - best) / 0.08)))
            for label, score in ordered
        ]
        total = sum(value for _, value in exps) or 1.0
        for label, value in exps:
            if label.startswith("S") and label[1:].isdigit():
                probabilities[f"speaker{int(label[1:])}"] = round(float(value / total), 4)
        key = f"speaker{int(assigned_speaker[1:])}" if assigned_speaker.startswith("S") and assigned_speaker[1:].isdigit() else ""
        if key:
            probabilities[key] = max(probabilities.get(key, 0.0), 0.0001)
        return probabilities

    def _apply_prototype_revision(self, revision: Any) -> bool:
        is_provisional_unknown = revision.previous_speaker is None
        with self._sentence_refinement_lock:
            record = self._sentence_refinement_records.get(int(revision.index))
            if record is None:
                return False
            if record.get("assigned_speaker") != revision.previous_speaker:
                return False
            if is_provisional_unknown and user_deleted_speaker_label(record):
                return False
            if revision.previous_speaker and user_confirmed_speaker_label(record) == revision.previous_speaker:
                return False
            if revision.assigned_speaker in rejected_speaker_labels(record):
                return False
            previous_provisional = str(record.get("provisional_assigned_speaker") or "")
            if is_provisional_unknown and previous_provisional == revision.assigned_speaker:
                return False
            original_unknown_probability = record.get("unknown_probability")
            probabilities = self._prototype_probabilities(
                revision.assigned_speaker,
                revision.prototype_scores,
            )
            similarities = dict(revision.prototype_scores)
            if is_provisional_unknown:
                record["provisional_assigned_speaker"] = revision.assigned_speaker
                record["provisional_probabilities"] = probabilities
                record["provisional_similarities"] = similarities
                record["provisional_top_similarity"] = revision.prototype_score
                record["provisional_margin"] = revision.prototype_margin
                record["provisional_assignment_source"] = "prototype_unknown_tentative"
            else:
                record["assigned_speaker"] = revision.assigned_speaker
                record["created_speaker"] = False
                record["probabilities"] = probabilities
                record["similarities"] = similarities
                record["unknown_probability"] = 0.0
                record["top_similarity"] = revision.prototype_score
                record["margin"] = revision.prototype_margin
                record["assignment_source"] = revision.assignment_source
            base_payload = dict(record["base_payload"])
            quality = record.get("quality")

        self._ensure_speaker_metadata(revision.assigned_speaker)
        revision_from = previous_provisional or revision.previous_speaker or "UNKNOWN"
        assignment_source = "prototype_unknown_tentative" if is_provisional_unknown else revision.assignment_source
        self._emit_transcript_sentence({
            **base_payload,
            "pending": False,
            "revision": True,
            "prototype_reassigned": True,
            "provisional_assignment": is_provisional_unknown,
            "prototype_reassigned_from": revision_from,
            "revision_from": revision_from,
            "revision_to": revision.assigned_speaker,
            "assigned_speaker": revision.assigned_speaker,
            **self._speaker_info_for_payload(revision.assigned_speaker),
            "created_speaker": False,
            "probabilities": probabilities,
            "similarities": similarities,
            "unknown_probability": original_unknown_probability if is_provisional_unknown else 0.0,
            "top_similarity": revision.prototype_score,
            "margin": revision.prototype_margin,
            "quality": quality,
            "assignment_source": assignment_source,
            "prototype_score": revision.prototype_score,
            "prototype_margin": revision.prototype_margin,
            "prototype_delta": revision.prototype_delta,
        })
        return True

    def _current_speaker_profile_labels(self) -> set[str]:
        memory = getattr(self, "memory", None)
        export_profiles = getattr(memory, "export_profiles", None)
        if not callable(export_profiles):
            return set()
        return {
            str(profile.get("label") or "").strip().upper()
            for profile in export_profiles()
            if str(profile.get("label") or "").strip()
        }

    def _remove_empty_detected_speaker_profiles(self, candidate_labels: set[str]) -> list[str]:
        candidates = {
            str(label or "").strip().upper()
            for label in candidate_labels
            if str(label or "").strip()
        }
        if not candidates:
            return []
        memory = getattr(self, "memory", None)
        if not callable(getattr(memory, "export_profiles", None)):
            return []

        with self._live_memory_update_lock_obj():
            with self._sentence_refinement_lock:
                occupied_labels: set[str] = set()
                for record in self._sentence_refinement_records.values():
                    label = record.get("assigned_speaker")
                    if not label:
                        label = record.get("provisional_assigned_speaker")
                    if label:
                        occupied_labels.add(str(label).strip().upper())

            current_profiles = memory.export_profiles()
            profiles_by_label = {
                str(profile.get("label") or "").strip().upper(): dict(profile)
                for profile in current_profiles
                if str(profile.get("label") or "").strip()
            }
            with self._speaker_lock:
                metadata_by_label = {
                    str(label).strip().upper(): dict(metadata)
                    for label, metadata in self._speaker_metadata.items()
                }
                seed_labels = {
                    str(profile.get("label") or f"S{index}").strip().upper()
                    for index, profile in enumerate(getattr(self, "_seed_profiles", []), 1)
                }

            removable: set[str] = set()
            for label in candidates - occupied_labels:
                profile = profiles_by_label.get(label)
                if profile is None or label in seed_labels:
                    continue
                metadata = metadata_by_label.get(label, {})
                if (
                    bool(profile.get("locked"))
                    or bool(metadata.get("locked"))
                    or str(metadata.get("source") or "detected").strip().lower() == "reference"
                    or bool(metadata.get("reference_audio"))
                    or str(metadata.get("identity_status") or "").strip().lower() == "confirmed"
                    or bool(metadata.get("person_id"))
                ):
                    continue
                removable.add(label)

            if not removable:
                return []

            with self._speaker_lock:
                for label in removable:
                    self._speaker_metadata.pop(label, None)
            for label in removable:
                self._speaker_last_media_end.pop(label, None)
            label_generations = getattr(self, "_speaker_label_generations", None)
            if label_generations is None:
                label_generations = {}
                self._speaker_label_generations = label_generations
            for label in removable:
                label_generations[label] = int(label_generations.get(label, 0)) + 1
            remove_profiles = getattr(memory, "remove_profiles", None)
            if callable(remove_profiles):
                remove_profiles(removable)
            else:
                memory.replace_profiles([
                    profile
                    for profile in current_profiles
                    if str(profile.get("label") or "").strip().upper() not in removable
                ])
            live_memory = getattr(self, "live_memory", None)
            if getattr(self, "_live_embedding_separate", False) and live_memory is not memory:
                remove_live_profiles = getattr(live_memory, "remove_profiles", None)
                if callable(remove_live_profiles):
                    remove_live_profiles(removable)
                else:
                    export_live_profiles = getattr(live_memory, "export_profiles", None)
                    replace_live_profiles = getattr(live_memory, "replace_profiles", None)
                    if callable(export_live_profiles) and callable(replace_live_profiles):
                        replace_live_profiles([
                            profile
                            for profile in export_live_profiles()
                            if str(profile.get("label") or "").strip().upper() not in removable
                        ])
            history = getattr(self, "_live_probability_history", None)
            if history is not None:
                history.clear()
            self._live_speaker_verify_next_at = 0.0
            self._sync_metadata_with_memory()

        self.emit_speaker_state()
        return sorted(removable, key=self._speaker_label_sort_key)

    def _remove_empty_reassigned_speaker_profiles(self, candidate_labels: set[str]) -> list[str]:
        return self._remove_empty_detected_speaker_profiles(candidate_labels)
