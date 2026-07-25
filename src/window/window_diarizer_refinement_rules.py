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




class WindowRefinementRulesMixin:
    @staticmethod
    def _speaker_probability_key(label: str) -> str:
        if label.startswith("S") and label[1:].isdigit():
            return f"speaker{int(label[1:])}"
        return label

    @staticmethod
    def _record_duration(record: dict[str, Any]) -> float:
        try:
            return max(0.0, float(record.get("duration_seconds") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _record_speech_evidence_duration(cls, record: dict[str, Any]) -> float:
        base_payload = record.get("base_payload")
        if isinstance(base_payload, dict):
            try:
                spoken = float(base_payload.get("spoken_word_seconds") or 0.0)
            except (TypeError, ValueError):
                spoken = 0.0
            if spoken > 0.0:
                return spoken
        return cls._record_duration(record)

    @staticmethod
    def _record_assigned_speaker(record: dict[str, Any]) -> str | None:
        value = record.get("assigned_speaker")
        if not value:
            return None
        label = str(value)
        return label if label.startswith("S") else None

    def _apply_small_island_merge(
        self,
        indexes: list[int],
        from_speaker: str,
        to_speaker: str,
        island_duration: float,
        *,
        assignment_source: str = "small_island_merge",
        marker_field: str = "small_island_merged",
        marker_from_field: str = "small_island_merged_from",
        duration_field: str = "small_island_duration",
    ) -> int:
        if not indexes or from_speaker == to_speaker:
            return 0
        speaker_key = self._speaker_probability_key(to_speaker)
        emitted: list[dict[str, Any]] = []
        with self._sentence_refinement_lock:
            records = []
            for index in indexes:
                record = self._sentence_refinement_records.get(int(index))
                if record is None or record.get("assigned_speaker") != from_speaker:
                    return 0
                records.append(record)
            for record in records:
                record["assigned_speaker"] = to_speaker
                record["created_speaker"] = False
                record["probabilities"] = {"unknown": 0.0, speaker_key: 1.0}
                record["unknown_probability"] = 0.0
                record["assignment_source"] = assignment_source
                similarities = dict(record.get("similarities") or {})
                similarities[to_speaker] = max(float(similarities.get(to_speaker, 0.0) or 0.0), 0.0001)
                record["similarities"] = similarities
                emitted.append({
                    **dict(record["base_payload"]),
                    "pending": False,
                    "revision": True,
                    marker_field: True,
                    marker_from_field: from_speaker,
                    "revision_from": from_speaker,
                    "revision_to": to_speaker,
                    "assigned_speaker": to_speaker,
                    **self._speaker_info_for_payload(to_speaker),
                    "created_speaker": False,
                    "probabilities": dict(record["probabilities"]),
                    "similarities": similarities,
                    "unknown_probability": 0.0,
                    "top_similarity": record.get("top_similarity"),
                    "margin": record.get("margin"),
                    "quality": record.get("quality"),
                    "assignment_source": assignment_source,
                    duration_field: round(float(island_duration), 4),
                })
        for payload in emitted:
            self._emit_transcript_sentence(payload)
        return len(emitted)

    def _merge_small_speaker_islands(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_small_island_merge", False)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_small_island_max_duration", 5.0)),
            )
            max_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_small_island_max_segments", 3)),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 3:
            return 0

        stats: dict[str, dict[str, float | int]] = {}
        islands: list[tuple[int, int, str, list[int], float]] = []
        position = 0
        while position < len(records):
            label = self._record_assigned_speaker(records[position])
            end = position + 1
            while end < len(records) and self._record_assigned_speaker(records[end]) == label:
                end += 1
            if label is not None:
                indexes = [int(record["index"]) for record in records[position:end]]
                duration = sum(self._record_duration(record) for record in records[position:end])
                entry = stats.setdefault(label, {"duration": 0.0, "segments": 0, "islands": 0})
                entry["duration"] = float(entry["duration"]) + duration
                entry["segments"] = int(entry["segments"]) + len(indexes)
                entry["islands"] = int(entry["islands"]) + 1
                islands.append((position, end, label, indexes, duration))
            position = end

        applied = 0
        for start, end, label, indexes, duration in islands:
            entry = stats.get(label) or {}
            if int(entry.get("islands") or 0) != 1:
                continue
            if float(entry.get("duration") or 0.0) > max_duration:
                continue
            if int(entry.get("segments") or 0) > max_segments:
                continue
            previous_speaker = None
            for index in range(start - 1, -1, -1):
                candidate = self._record_assigned_speaker(records[index])
                if candidate and candidate != label:
                    previous_speaker = candidate
                    break
            next_speaker = None
            for index in range(end, len(records)):
                candidate = self._record_assigned_speaker(records[index])
                if candidate and candidate != label:
                    next_speaker = candidate
                    break
            if previous_speaker is None or previous_speaker != next_speaker:
                continue
            island_records = records[start:end]
            if any(
                bool(record.get("short_distinct_origin"))
                or str(record.get("assignment_source") or "")
                == "short_distinct_new_speaker"
                for record in island_records
            ):
                # This row already passed the opt-in acoustic novelty gate.
                # Neighbor topology alone must not immediately erase it.
                continue
            applied += self._apply_small_island_merge(
                indexes,
                label,
                previous_speaker,
                duration,
            )
        return applied

    def _merge_tiny_fragmented_speaker_profiles(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_tiny_fragmented_merge", True)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_tiny_fragmented_max_duration", 6.0)),
            )
            max_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_tiny_fragmented_max_segments", 8)),
            )
            min_islands = max(
                2,
                int(getattr(self.args, "speaker_refinement_tiny_fragmented_min_islands", 2)),
            )
            max_islands = max(
                min_islands,
                int(getattr(self.args, "speaker_refinement_tiny_fragmented_max_islands", 3)),
            )
            min_neighbor_share = max(
                0.0,
                min(1.0, float(getattr(self.args, "speaker_refinement_tiny_fragmented_min_neighbor_share", 0.5))),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 3:
            return 0

        stats: dict[str, dict[str, Any]] = {}
        islands: list[tuple[int, int, str, list[int], float]] = []
        position = 0
        while position < len(records):
            label = self._record_assigned_speaker(records[position])
            end = position + 1
            while end < len(records) and self._record_assigned_speaker(records[end]) == label:
                end += 1
            if label is not None:
                indexes = [int(record["index"]) for record in records[position:end]]
                duration = sum(self._record_duration(record) for record in records[position:end])
                entry = stats.setdefault(label, {"duration": 0.0, "segments": 0, "islands": 0, "indexes": []})
                entry["duration"] = float(entry["duration"]) + duration
                entry["segments"] = int(entry["segments"]) + len(indexes)
                entry["islands"] = int(entry["islands"]) + 1
                entry["indexes"].extend(indexes)
                islands.append((position, end, label, indexes, duration))
            position = end

        candidate_labels = {
            label
            for label, entry in stats.items()
            if float(entry.get("duration") or 0.0) <= max_duration
            and int(entry.get("segments") or 0) <= max_segments
            and int(entry.get("islands") or 0) >= min_islands
            and int(entry.get("islands") or 0) <= max_islands
        }
        applied = 0
        for _start, _end, label, _indexes, _duration in islands:
            entry = stats.get(label) or {}
            if entry.get("seen"):
                continue
            entry["seen"] = True
            duration = float(entry.get("duration") or 0.0)
            segments = int(entry.get("segments") or 0)
            island_count = int(entry.get("islands") or 0)
            if (
                duration > max_duration
                or segments > max_segments
                or island_count < min_islands
                or island_count > max_islands
            ):
                continue
            neighbor_votes: Counter[str] = Counter()
            for start, end, island_label, _island_indexes, _island_duration in islands:
                if island_label != label:
                    continue
                previous_speaker = None
                for index in range(start - 1, -1, -1):
                    candidate = self._record_assigned_speaker(records[index])
                    if candidate and candidate != label:
                        previous_speaker = candidate
                        break
                next_speaker = None
                for index in range(end, len(records)):
                    candidate = self._record_assigned_speaker(records[index])
                    if candidate and candidate != label:
                        next_speaker = candidate
                        break
                if previous_speaker:
                    neighbor_votes[previous_speaker] += 1
                if next_speaker:
                    neighbor_votes[next_speaker] += 1
            if not neighbor_votes:
                continue
            ranked_neighbors = neighbor_votes.most_common()
            if len(ranked_neighbors) > 1 and ranked_neighbors[0][1] == ranked_neighbors[1][1]:
                continue
            target_speaker, target_votes = ranked_neighbors[0]
            if target_votes / max(1, sum(neighbor_votes.values())) < min_neighbor_share:
                continue
            if target_speaker in candidate_labels:
                continue
            applied += self._apply_small_island_merge(
                [int(index) for index in entry.get("indexes") or []],
                label,
                target_speaker,
                duration,
                assignment_source="tiny_fragmented_profile_merge",
                marker_field="tiny_fragmented_profile_merged",
                marker_from_field="tiny_fragmented_profile_merged_from",
                duration_field="tiny_fragmented_profile_duration",
            )
        return applied

    @staticmethod
    def _is_promotional_outro_text(text: str) -> bool:
        words = re.findall(r"[a-z0-9']+", str(text or "").lower())
        if not words:
            return False
        word_set = set(words)
        if "subscribe" in word_set:
            return True
        if "youtube" in word_set and word_set.intersection({"like", "watch", "catch", "channel"}):
            return True
        if "notification" in word_set and "bell" in word_set:
            return True
        return False

    def _merge_terminal_promotional_outro(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_terminal_outro_merge", True)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_terminal_outro_max_duration", 12.0)),
            )
            lookback_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_terminal_outro_lookback_segments", 2)),
            )
            min_target_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_terminal_outro_min_target_duration", 5.0)),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 2:
            return 0

        stats: dict[str, dict[str, Any]] = {}
        for record in records:
            label = self._record_assigned_speaker(record)
            if label is None:
                continue
            entry = stats.setdefault(label, {"duration": 0.0, "segments": 0})
            entry["duration"] = float(entry["duration"]) + self._record_duration(record)
            entry["segments"] = int(entry["segments"]) + 1

        first_speaker = None
        for record in records:
            first_speaker = self._record_assigned_speaker(record)
            if first_speaker:
                break
        if first_speaker is None:
            return 0
        first_stats = stats.get(first_speaker) or {}
        if float(first_stats.get("duration") or 0.0) < min_target_duration:
            return 0

        applied = 0
        start = max(0, len(records) - lookback_segments)
        for record in records[start:]:
            label = self._record_assigned_speaker(record)
            if label is None or label == first_speaker:
                continue
            entry = stats.get(label) or {}
            if int(entry.get("segments") or 0) != 1:
                continue
            duration = self._record_duration(record)
            if duration > max_duration:
                continue
            base_payload = dict(record.get("base_payload") or {})
            text = str(base_payload.get("text") or "")
            if not self._is_promotional_outro_text(text):
                continue
            applied += self._apply_small_island_merge(
                [int(record["index"])],
                label,
                first_speaker,
                duration,
                assignment_source="terminal_promotional_outro_merge",
                marker_field="terminal_promotional_outro_merged",
                marker_from_field="terminal_promotional_outro_merged_from",
                duration_field="terminal_promotional_outro_duration",
            )
        return applied

    def _fill_unknown_same_speaker_islands(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_unknown_same_speaker_fill", True)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_same_speaker_max_duration", 3.0)),
            )
            max_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_unknown_same_speaker_max_segments", 1)),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 3:
            return 0

        applied = 0
        position = 0
        while position < len(records):
            if self._record_assigned_speaker(records[position]) is not None:
                position += 1
                continue
            start = position
            while position < len(records) and self._record_assigned_speaker(records[position]) is None:
                position += 1
            end = position
            if start == 0 or end >= len(records):
                continue
            previous_speaker = self._record_assigned_speaker(records[start - 1])
            next_speaker = self._record_assigned_speaker(records[end])
            if previous_speaker is None or previous_speaker != next_speaker:
                continue
            indexes = [int(record["index"]) for record in records[start:end]]
            duration = sum(self._record_duration(record) for record in records[start:end])
            if len(indexes) > max_segments or duration > max_duration:
                continue
            if any(
                str(record.get("assignment_source") or "")
                == "new_speaker_pending"
                or bool(
                    (
                        (record.get("base_payload") or {}).get("asr_review")
                        or {}
                    ).get("needs_review")
                )
                for record in records[start:end]
            ):
                continue
            try:
                short_distinct_active = any(
                    bool(record.get("short_distinct_origin"))
                    or str(record.get("assignment_source") or "")
                    == "short_distinct_new_speaker"
                    for record in records
                )
                short_sentence_max_duration = min(
                    1.0 if short_distinct_active else 0.0,
                    float(getattr(self.args, "min_new_speaker_seconds", 2.0358)),
                )
                short_sentence_min_similarity = max(
                    float(getattr(self.args, "same_speaker_similarity", 0.45)),
                    float(getattr(self.args, "known_speaker_min_similarity", -1.0)),
                )
            except (TypeError, ValueError):
                short_sentence_max_duration = 2.0358
                short_sentence_min_similarity = 0.45
            target_supported = True
            for record in records[start:end]:
                if (
                    record.get("embedding") is None
                    or self._record_speech_evidence_duration(record)
                    >= short_sentence_max_duration
                ):
                    continue
                similarity = (record.get("similarities") or {}).get(previous_speaker)
                try:
                    target_supported = float(similarity) >= short_sentence_min_similarity
                except (TypeError, ValueError):
                    target_supported = False
                if not target_supported:
                    break
            if not target_supported:
                continue
            applied += self._apply_unknown_same_speaker_fill(
                indexes,
                previous_speaker,
                duration,
            )
        return applied

    def _apply_unknown_same_speaker_fill(
        self,
        indexes: list[int],
        to_speaker: str,
        island_duration: float,
        *,
        assignment_source: str = "unknown_same_speaker_island_fill",
        marker_field: str = "unknown_same_speaker_filled",
        duration_field: str = "unknown_same_speaker_fill_duration",
    ) -> int:
        if not indexes:
            return 0
        speaker_key = self._speaker_probability_key(to_speaker)
        emitted: list[dict[str, Any]] = []
        with self._sentence_refinement_lock:
            records = []
            for index in indexes:
                record = self._sentence_refinement_records.get(int(index))
                if record is None or self._record_assigned_speaker(record) is not None:
                    return 0
                records.append(record)
            for record in records:
                similarities = dict(record.get("similarities") or {})
                similarities[to_speaker] = max(float(similarities.get(to_speaker, 0.0) or 0.0), 0.0001)
                record["assigned_speaker"] = to_speaker
                record["created_speaker"] = False
                record["probabilities"] = {"unknown": 0.0, speaker_key: 1.0}
                record["similarities"] = similarities
                record["unknown_probability"] = 0.0
                record["assignment_source"] = assignment_source
                emitted.append({
                    **dict(record["base_payload"]),
                    "pending": False,
                    "revision": True,
                    marker_field: True,
                    "revision_from": "UNKNOWN",
                    "revision_to": to_speaker,
                    "assigned_speaker": to_speaker,
                    **self._speaker_info_for_payload(to_speaker),
                    "created_speaker": False,
                    "probabilities": dict(record["probabilities"]),
                    "similarities": similarities,
                    "unknown_probability": 0.0,
                    "top_similarity": record.get("top_similarity"),
                    "margin": record.get("margin"),
                    "quality": record.get("quality"),
                    "assignment_source": assignment_source,
                    duration_field: round(float(island_duration), 4),
                })
        for payload in emitted:
            self._emit_transcript_sentence(payload)
        return len(emitted)

    def _fill_unknown_previous_speaker_tails(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_unknown_previous_speaker_fill", True)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_previous_speaker_max_duration", 0.75)),
            )
            max_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_unknown_previous_speaker_max_segments", 1)),
            )
            max_previous_gap = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_previous_speaker_max_previous_gap", 0.35)),
            )
            min_next_gap = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_previous_speaker_min_next_gap", 0.3)),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 3:
            return 0

        applied = 0
        for position, record in enumerate(records):
            if self._record_assigned_speaker(record) is not None:
                continue
            if str(record.get("assignment_source") or "") != "non_embedding_candidate":
                continue
            if position == 0 or position + 1 >= len(records):
                continue
            previous_record = records[position - 1]
            next_record = records[position + 1]
            previous_speaker = self._record_assigned_speaker(previous_record)
            if previous_speaker is None:
                continue
            indexes = [int(record["index"])]
            duration = self._record_duration(record)
            if len(indexes) > max_segments or duration > max_duration:
                continue
            try:
                previous_gap = float(record["base_payload"].get("start")) - float(
                    previous_record["base_payload"].get("end")
                )
                next_gap = float(next_record["base_payload"].get("start")) - float(
                    record["base_payload"].get("end")
                )
            except (TypeError, ValueError):
                continue
            if previous_gap < -0.001 or next_gap < -0.001:
                continue
            if previous_gap > max_previous_gap or next_gap < min_next_gap:
                continue
            fills = self._apply_unknown_same_speaker_fill(
                indexes,
                previous_speaker,
                duration,
                assignment_source="unknown_previous_speaker_tail_fill",
                marker_field="unknown_previous_speaker_filled",
                duration_field="unknown_previous_speaker_fill_duration",
            )
            if fills:
                record["assigned_speaker"] = previous_speaker
                record["assignment_source"] = "unknown_previous_speaker_tail_fill"
                applied += fills
        return applied

    def _fill_unknown_next_speaker_heads(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_unknown_next_speaker_fill", True)):
            return 0
        try:
            max_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_next_speaker_max_duration", 1.75)),
            )
            max_segments = max(
                1,
                int(getattr(self.args, "speaker_refinement_unknown_next_speaker_max_segments", 1)),
            )
            max_next_gap = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_next_speaker_max_next_gap", 0.05)),
            )
            min_previous_gap = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_unknown_next_speaker_min_previous_gap", 0.15)),
            )
        except (TypeError, ValueError):
            return 0
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _, record in sorted(self._sentence_refinement_records.items())
            ]
        if len(records) < 3:
            return 0

        applied = 0
        for position, record in enumerate(records):
            if self._record_assigned_speaker(record) is not None:
                continue
            if str(record.get("assignment_source") or "") != "non_embedding_candidate":
                continue
            if position == 0 or position + 1 >= len(records):
                continue
            previous_record = records[position - 1]
            next_record = records[position + 1]
            next_speaker = self._record_assigned_speaker(next_record)
            if next_speaker is None:
                continue
            if self._record_assigned_speaker(previous_record) == next_speaker:
                continue
            indexes = [int(record["index"])]
            duration = self._record_duration(record)
            if len(indexes) > max_segments or duration > max_duration:
                continue
            try:
                previous_gap = float(record["base_payload"].get("start")) - float(
                    previous_record["base_payload"].get("end")
                )
                next_gap = float(next_record["base_payload"].get("start")) - float(
                    record["base_payload"].get("end")
                )
            except (TypeError, ValueError):
                continue
            if previous_gap < -0.001 or next_gap < -0.001:
                continue
            if previous_gap < min_previous_gap or next_gap > max_next_gap:
                continue
            applied += self._apply_unknown_same_speaker_fill(
                indexes,
                next_speaker,
                duration,
                assignment_source="unknown_next_speaker_head_fill",
                marker_field="unknown_next_speaker_filled",
                duration_field="unknown_next_speaker_fill_duration",
            )
        return applied

    def _split_long_low_confidence_retro_assignments(self) -> int:
        if not bool(getattr(self.args, "speaker_refinement_long_low_confidence_retro_split", True)):
            return 0
        try:
            min_duration = max(
                0.0,
                float(getattr(self.args, "speaker_refinement_long_low_confidence_retro_min_duration", 4.0)),
            )
            max_similarity = float(
                getattr(self.args, "speaker_refinement_long_low_confidence_retro_max_similarity", 0.06)
            )
            max_margin = float(
                getattr(self.args, "speaker_refinement_long_low_confidence_retro_max_margin", 0.04)
            )
            max_splits = max(
                0,
                int(getattr(self.args, "speaker_refinement_long_low_confidence_retro_max_splits", 1)),
            )
        except (TypeError, ValueError):
            return 0
        if max_splits <= 0:
            return 0

        emitted: list[dict[str, Any]] = []
        with self._sentence_refinement_lock:
            records = [
                self._sentence_refinement_records[index]
                for index in sorted(self._sentence_refinement_records)
            ]
            for record in records:
                if len(emitted) >= max_splits:
                    break
                if str(record.get("assignment_source") or "") != "retro":
                    continue
                from_speaker = self._record_assigned_speaker(record)
                if from_speaker is None:
                    continue
                duration = self._record_duration(record)
                if duration < min_duration:
                    continue
                try:
                    top_similarity = float(record.get("top_similarity"))
                    margin = float(record.get("margin"))
                except (TypeError, ValueError):
                    continue
                if top_similarity > max_similarity or margin > max_margin:
                    continue
                embedding = record.get("embedding")
                if embedding is None:
                    continue
                to_speaker = self._next_detected_speaker_label()
                self.memory.upsert_profile(
                    to_speaker,
                    np.asarray(embedding, dtype=np.float32),
                    duration_seconds=duration,
                    sentence_count=1,
                )
                self._ensure_speaker_metadata(to_speaker)
                speaker_key = self._speaker_probability_key(to_speaker)
                record["assigned_speaker"] = to_speaker
                record["created_speaker"] = True
                record["probabilities"] = {"unknown": 0.0, speaker_key: 1.0}
                similarities = dict(record.get("similarities") or {})
                similarities[to_speaker] = 1.0
                record["similarities"] = similarities
                record["unknown_probability"] = 0.0
                record["top_similarity"] = 1.0
                record["margin"] = 1.0
                record["assignment_source"] = "long_low_confidence_retro_split"
                emitted.append({
                    **dict(record["base_payload"]),
                    "pending": False,
                    "revision": True,
                    "long_low_confidence_retro_split": True,
                    "long_low_confidence_retro_split_from": from_speaker,
                    "revision_from": from_speaker,
                    "revision_to": to_speaker,
                    "assigned_speaker": to_speaker,
                    **self._speaker_info_for_payload(to_speaker),
                    "created_speaker": True,
                    "probabilities": dict(record["probabilities"]),
                    "similarities": similarities,
                    "unknown_probability": 0.0,
                    "top_similarity": 1.0,
                    "margin": 1.0,
                    "quality": record.get("quality"),
                    "assignment_source": "long_low_confidence_retro_split",
                    "long_low_confidence_retro_original_similarity": round(top_similarity, 4),
                    "long_low_confidence_retro_original_margin": round(margin, 4),
                    "long_low_confidence_retro_duration": round(duration, 4),
                })
        for payload in emitted:
            self._emit_transcript_sentence(payload)
        return len(emitted)
