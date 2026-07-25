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




class WindowRefinementMixin:
    def _refine_speaker_assignments(self) -> None:
        if not bool(getattr(self.args, "speaker_refinement", True)):
            return
        allow_unknown_tentative = bool(getattr(self.args, "speaker_refinement_unknown_tentative", True))
        allow_known = bool(getattr(self.args, "allow_speaker_reassignment", False))
        allow_small_island_merge = bool(getattr(self.args, "speaker_refinement_small_island_merge", False))
        if (
            not allow_unknown_tentative
            and not allow_known
            and not allow_small_island_merge
        ):
            return
        if not self._sentence_refinement_run_lock.acquire(blocking=False):
            return
        try:
            applied = 0
            known_revisions = 0
            reassigned_source_labels: set[str] = set()
            if allow_unknown_tentative or allow_known:
                session_state = getattr(self, "_session_state", None)
                expected_version = session_state.version() if session_state is not None else None
                with self._sentence_refinement_lock:
                    records = [
                        dict(record)
                        for _, record in sorted(self._sentence_refinement_records.items())
                    ]
                if len(records) >= 2:
                    assignment_engine = getattr(self, "_assignment_engine", None)
                    if assignment_engine is not None and expected_version is not None:
                        effects = assignment_engine.plan_refinement(AssignmentRequest(
                            records=tuple(records),
                            config=self._speaker_refinement_config(),
                            allow_known_reassignment=allow_known,
                            expected_version=expected_version,
                        ))
                        revisions = effects.revisions
                    else:
                        revisions = find_speaker_prototype_revisions(
                            records,
                            self._speaker_refinement_config(),
                            allow_known_reassignment=allow_known,
                        )
                    transaction = (
                        session_state.transaction(mutate=True)
                        if session_state is not None and expected_version is not None
                        else nullcontext()
                    )
                    with transaction:
                        if session_state is not None and not session_state.is_current(expected_version):
                            return
                        for revision in revisions:
                            if revision.previous_speaker is None and not allow_unknown_tentative:
                                continue
                            if not allow_known and revision.previous_speaker is not None:
                                continue
                            if self._apply_prototype_revision(revision):
                                applied += 1
                                if revision.previous_speaker is not None:
                                    known_revisions += 1
                                    reassigned_source_labels.add(str(revision.previous_speaker))
            removed_empty_speakers = self._remove_empty_reassigned_speaker_profiles(
                reassigned_source_labels,
            )
            small_island_merges = self._merge_small_speaker_islands()
            if applied:
                removed_message = ""
                if removed_empty_speakers:
                    noun = "speaker" if len(removed_empty_speakers) == 1 else "speakers"
                    removed_message = (
                        f" Deleted empty {noun} {', '.join(removed_empty_speakers)} after reassignment."
                    )
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Prototype speaker refinement applied {applied} revision(s)"
                            f"{' including ' + str(known_revisions) + ' known-speaker change(s)' if known_revisions else ''}."
                            f"{removed_message}"
                        )
                    },
                )
            if small_island_merges:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement merged {small_island_merges} small one-off speaker segment(s)."
                        )
                    },
                )
        finally:
            self._sentence_refinement_run_lock.release()

    def _apply_delayed_multirow_clustering(self) -> int:
        if not bool(getattr(self.args, "delayed_multirow_clustering", True)):
            return 0
        if not bool(getattr(self.args, "allow_speaker_reassignment", False)):
            return 0
        try:
            max_new_speakers = max(0, int(getattr(self.args, "delayed_clustering_max_new_speakers", 2)))
        except (TypeError, ValueError):
            return 0
        if max_new_speakers <= 0 or self.memory.profile_count() >= int(self.args.max_speakers):
            return 0

        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        proposals = find_delayed_speaker_splits(records, self._delayed_clustering_config())
        if not proposals:
            return 0

        applied = 0
        for proposal in proposals[:max_new_speakers]:
            if self.memory.profile_count() >= int(self.args.max_speakers):
                break
            new_label = self._next_detected_speaker_label()
            with self._sentence_refinement_lock:
                selected = []
                for index in proposal.indexes:
                    record = self._sentence_refinement_records.get(int(index))
                    if record is None or record.get("assigned_speaker") != proposal.previous_speaker:
                        selected = []
                        break
                    if user_confirmed_speaker_label(record) or rejected_speaker_labels(record):
                        selected = []
                        break
                    selected.append(record)
                if not selected:
                    continue

            self.memory.upsert_profile(
                new_label,
                proposal.centroid,
                duration_seconds=proposal.speech_seconds,
                sentence_count=len(selected),
            )
            self._ensure_speaker_metadata(new_label)
            speaker_key = self._speaker_probability_key(new_label)
            emitted: list[dict[str, Any]] = []
            with self._sentence_refinement_lock:
                for record in selected:
                    vector = normalize_vector(record["embedding"])
                    new_similarity = float(np.dot(vector, proposal.centroid))
                    old_similarity = float(
                        (record.get("similarities") or {}).get(proposal.previous_speaker, 0.0) or 0.0
                    )
                    original_unknown = max(
                        0.0,
                        min(1.0, float(record.get("unknown_probability") or 0.0)),
                    )
                    probabilities = {
                        "unknown": round(original_unknown, 4),
                        speaker_key: round(1.0 - original_unknown, 4),
                    }
                    similarities = dict(record.get("similarities") or {})
                    similarities[new_label] = round(new_similarity, 4)
                    record["assigned_speaker"] = new_label
                    record["created_speaker"] = False
                    record["probabilities"] = probabilities
                    record["similarities"] = similarities
                    record["top_similarity"] = round(new_similarity, 4)
                    record["margin"] = round(new_similarity - old_similarity, 4)
                    record["assignment_source"] = "delayed_multirow_split"
                    emitted.append({
                        **dict(record["base_payload"]),
                        "pending": False,
                        "revision": True,
                        "delayed_multirow_split": True,
                        "revision_from": proposal.previous_speaker,
                        "revision_to": new_label,
                        "assigned_speaker": new_label,
                        **self._speaker_info_for_payload(new_label),
                        "created_speaker": False,
                        "probabilities": probabilities,
                        "similarities": similarities,
                        "unknown_probability": round(original_unknown, 4),
                        "top_similarity": round(new_similarity, 4),
                        "margin": round(new_similarity - old_similarity, 4),
                        "quality": record.get("quality"),
                        "assignment_source": "delayed_multirow_split",
                        "delayed_cluster_average_gain": round(proposal.average_gain, 4),
                        "delayed_cluster_leave_one_out_similarity": round(
                            proposal.leave_one_out_similarity, 4
                        ),
                        "delayed_cluster_core_similarity": round(proposal.core_similarity, 4),
                        "delayed_cluster_time_groups": proposal.time_groups,
                    })
            for payload in emitted:
                self._emit_transcript_sentence(payload)
            applied += len(emitted)
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Delayed multi-row clustering split {len(emitted)} sentence(s) "
                        f"from {proposal.previous_speaker} into {new_label} "
                        f"(gain={proposal.average_gain:.3f}, "
                        f"loo={proposal.leave_one_out_similarity:.3f})."
                    )
                },
            )
        if applied:
            self.emit_speaker_state()
        return applied

    def _finalize_speaker_refinement(self) -> None:
        if bool(getattr(self.args, "speaker_refinement", True)):
            try:
                passes = max(0, int(getattr(self.args, "speaker_refinement_final_passes", 1)))
            except (TypeError, ValueError):
                passes = 0
            for _ in range(passes):
                self._refine_speaker_assignments()
            delayed_splits = self._apply_delayed_multirow_clustering()
            if delayed_splits:
                self._refine_speaker_assignments()
            tiny_fragmented_merges = self._merge_tiny_fragmented_speaker_profiles()
            if tiny_fragmented_merges:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement merged {tiny_fragmented_merges} tiny fragmented speaker segment(s)."
                        )
                    },
                )
            terminal_outro_merges = self._merge_terminal_promotional_outro()
            if terminal_outro_merges:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement merged {terminal_outro_merges} terminal promotional outro segment(s)."
                        )
                    },
                )
            long_retro_splits = self._split_long_low_confidence_retro_assignments()
            if long_retro_splits:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement split {long_retro_splits} long low-confidence retro segment(s)."
                        )
                    },
                )
            unknown_same_speaker_fills = self._fill_unknown_same_speaker_islands()
            if unknown_same_speaker_fills:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement filled {unknown_same_speaker_fills} short unknown same-speaker segment(s)."
                        )
                    },
                )
            unknown_previous_speaker_fills = self._fill_unknown_previous_speaker_tails()
            if unknown_previous_speaker_fills:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement filled {unknown_previous_speaker_fills} short unknown previous-speaker segment(s)."
                        )
                    },
                )
            unknown_next_speaker_fills = self._fill_unknown_next_speaker_heads()
            if unknown_next_speaker_fills:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Speaker refinement filled {unknown_next_speaker_fills} short unknown next-speaker segment(s)."
                        )
                    },
                )
        removed_empty_speakers = self._remove_empty_detected_speaker_profiles(
            self._current_speaker_profile_labels(),
        )
        if removed_empty_speakers:
            noun = "speaker" if len(removed_empty_speakers) == 1 else "speakers"
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Deleted empty {noun} {', '.join(removed_empty_speakers)} after final review."
                    )
                },
            )

    def _revisit_unknown_sentences(self) -> None:
        if not bool(getattr(self.args, "speaker_refinement_unknown_commit", True)):
            return
        try:
            strong_match_similarity = float(
                getattr(self.args, "same_speaker_similarity", 0.45)
            )
        except (TypeError, ValueError):
            strong_match_similarity = 0.45
        single_profile = self.memory.profile_count() == 1
        try:
            short_sentence_max_duration = (
                min(
                    1.0,
                    float(getattr(self.args, "min_new_speaker_seconds", 2.0358)),
                )
                if self._has_short_distinct_speaker_record()
                else 0.0
            )
            short_sentence_min_similarity = max(
                strong_match_similarity,
                float(getattr(self.args, "known_speaker_min_similarity", -1.0)),
            )
        except (TypeError, ValueError):
            short_sentence_max_duration = 2.0358
            short_sentence_min_similarity = strong_match_similarity
        with self._unknown_lock:
            candidates = list(self._unknown_sentences)

        for candidate in candidates:
            asr_review = candidate.base_payload.get("asr_review")
            if isinstance(asr_review, dict) and bool(asr_review.get("needs_review")):
                continue
            min_similarity = float(self.args.retro_reassign_min_similarity)
            with self._sentence_refinement_lock:
                candidate_record = (
                    self._sentence_refinement_records.get(int(candidate.index)) or {}
                )
            requires_strong_retro_match = (
                str(candidate_record.get("assignment_source") or "")
                == "new_speaker_pending"
                or (
                    isinstance(asr_review, dict)
                    and bool(asr_review.get("needs_review"))
                )
            )
            if requires_strong_retro_match:
                min_similarity = max(
                    min_similarity,
                    strong_match_similarity,
                    float(
                        getattr(
                            self.args,
                            "new_speaker_confirmation_similarity",
                            strong_match_similarity,
                        )
                    ),
                )
            try:
                speech_evidence_duration = max(
                    0.0,
                    float(
                        candidate.base_payload.get("spoken_word_seconds")
                        or candidate.duration_seconds
                    ),
                )
            except (TypeError, ValueError):
                speech_evidence_duration = candidate.duration_seconds
            if speech_evidence_duration < short_sentence_max_duration:
                min_similarity = max(min_similarity, short_sentence_min_similarity)
            decision = self.memory.score_existing(
                candidate.embedding,
                candidate.duration_seconds,
                min_similarity=min_similarity,
                min_margin=self.args.retro_reassign_min_margin,
            )
            # With one profile there is no genuine runner-up, so the relaxed
            # retro margin is always satisfied. Do not turn an earlier
            # Unknown into that sole speaker unless the voice match itself is
            # strong. Once multiple profiles exist, keep the existing relaxed
            # recovery behavior.
            if (
                not decision.assigned_speaker
                or single_profile
                and (
                    decision.top_similarity is None
                    or decision.top_similarity < strong_match_similarity
                )
            ):
                continue
            with self._sentence_refinement_lock:
                record = self._sentence_refinement_records.get(int(candidate.index)) or {}
                reassigned_from = str(record.get("provisional_assigned_speaker") or "UNKNOWN")
            if not self._remove_unknown_sentence(candidate.index):
                continue
            payload = {
                **candidate.base_payload,
                "pending": False,
                "revision": True,
                "retro_reassigned": True,
                "retro_reassigned_from": reassigned_from,
                "revision_from": reassigned_from,
                "revision_to": decision.assigned_speaker,
                "assigned_speaker": decision.assigned_speaker,
                **self._speaker_info_for_payload(decision.assigned_speaker),
                "created_speaker": False,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
                "quality": decision.quality,
                "assignment_source": decision.assignment_source,
            }
            self._record_sentence_assignment(
                candidate.index,
                candidate.base_payload,
                candidate.embedding,
                candidate.duration_seconds,
                payload,
            )
            self._emit_transcript_sentence(payload)
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Reassigned unknown sentence {candidate.index} "
                        f"to {decision.assigned_speaker} "
                        f"(sim={decision.top_similarity}, margin={decision.margin})."
                    )
                },
            )
