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




class WindowSessionViewMixin:
    @staticmethod
    def _speaker_label_sort_key(label: str) -> tuple[int, str]:
        value = str(label or "").strip().upper()
        if value.startswith("S") and value[1:].isdigit():
            return (int(value[1:]), value)
        return (999999, value)

    @staticmethod
    def _normalized_speaker_label(label: Any) -> str | None:
        value = str(label or "").strip().upper()
        if not value or value == "UNKNOWN":
            return None
        if not re.fullmatch(r"S\d+", value):
            raise ValueError("Invalid speaker id.")
        return value

    @staticmethod
    def _embedding_copy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32).reshape(-1).copy()

    def _speaker_review_profiles(self) -> list[dict[str, Any]]:
        memory = getattr(self, "memory", None)
        export_profiles = getattr(memory, "export_profiles", None)
        if not callable(export_profiles):
            return []
        profiles = export_profiles()
        centroids: dict[str, np.ndarray] = {}
        for profile in profiles:
            label = str(profile.get("label") or "")
            if not label:
                continue
            try:
                centroids[label] = np.asarray(profile.get("centroid"), dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
        for profile in profiles:
            label = str(profile.get("label") or "")
            similarities: dict[str, float] = {}
            left = centroids.get(label)
            if left is None:
                continue
            for other_label, right in centroids.items():
                if other_label == label:
                    continue
                try:
                    similarities[other_label] = round(float(cosine_similarity(left, right)), 4)
                except ValueError:
                    continue
            profile["similarities"] = similarities
        return profiles

    def _with_sentence_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        if bool(row.get("pending")) or bool(row.get("realtime")):
            return row
        row["review"] = annotate_review(
            row,
            speaker_profiles=self._speaker_review_profiles(),
        )
        return row

    def _emit_transcript_sentence(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = self._with_sentence_review(payload)
        self.bus.emit("sentence", row)
        return row

    def _record_to_sentence_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        base_payload = dict(record.get("base_payload") or {})
        assigned_speaker = record.get("assigned_speaker")
        row = {
            **base_payload,
            "pending": False,
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": bool(record.get("created_speaker")),
            "probabilities": dict(record.get("probabilities") or {}),
            "similarities": dict(record.get("similarities") or {}),
            "unknown_probability": record.get("unknown_probability"),
            "top_similarity": record.get("top_similarity"),
            "margin": record.get("margin"),
            "quality": record.get("quality"),
            "assignment_source": str(record.get("assignment_source") or ""),
        }
        if "automatic_assigned_speaker" in record:
            row["automatic_assigned_speaker"] = record.get("automatic_assigned_speaker")
        if "automatic_assignment_source" in record:
            row["automatic_assignment_source"] = record.get("automatic_assignment_source")
        correction = record.get("correction")
        if isinstance(correction, dict) and correction:
            row["correction"] = copy.deepcopy(correction)
        return self._with_sentence_review(row)

    def _session_transcript_rows_and_embeddings(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._sentence_refinement_lock:
            records = [
                dict(record)
                for _index, record in sorted(self._sentence_refinement_records.items())
            ]

        rows: list[dict[str, Any]] = []
        embeddings: list[dict[str, Any]] = []
        for record in records:
            row = self._record_to_sentence_payload(record)
            rows.append(row)
            embedding = record.get("embedding")
            if embedding is not None:
                base_payload = dict(record.get("base_payload") or {})
                index_value = record.get("index")
                if index_value is None:
                    index_value = base_payload.get("index", 0)
                embeddings.append({
                    "index": int(index_value or 0),
                    "duration_seconds": float(record.get("duration_seconds") or 0.0),
                    "assigned_speaker": row.get("assigned_speaker"),
                    "embedding": embedding,
                })
        return rows, embeddings

    def _session_speaker_profiles(self) -> list[dict[str, Any]]:
        self._sync_metadata_with_memory()
        profiles = self.memory.export_profiles()
        with self._speaker_lock:
            metadata_by_label = {
                label: dict(metadata)
                for label, metadata in self._speaker_metadata.items()
            }
        serialized: list[dict[str, Any]] = []
        for profile in profiles:
            label = str(profile.get("label") or "")
            metadata = metadata_by_label.get(label, {})
            item = {
                "label": label,
                "name": str(metadata.get("name") or ""),
                "display_name": str(metadata.get("name") or "") or f"Speaker {profile.get('index') or label}",
                "source": str(metadata.get("source") or "detected"),
                "locked": bool(metadata.get("locked") or profile.get("locked")),
                "reference_audio": str(metadata.get("reference_audio") or ""),
                "sentence_count": int(profile.get("sentence_count") or 1),
                "speech_seconds": float(profile.get("speech_seconds") or 0.0),
                "created_at": float(profile.get("created_at") or time.time()),
                "last_seen_at": float(profile.get("last_seen_at") or time.time()),
            }
            item.update(self._centroid_payload(profile["centroid"]))
            serialized.append(item)
        return serialized

    def _session_source_metadata(self) -> dict[str, Any]:
        media = self.media
        with self._audio_lock:
            streaming_audio = bool(self._streaming_audio)
            duration_seconds = float(self.duration)
            sample_rate = int(self.sample_rate)
            stream_samples = int(self._stream_audio_samples)
        source = {
            "url": str(media.url or ""),
            "video_id": str(media.video_id or ""),
            "started_at": str(self._session_started_at or ""),
            "video_path": "" if streaming_audio else str(media.video_file),
            "audio_path": "" if streaming_audio else str(media.audio_file),
            "streaming_audio": streaming_audio,
            "audio_sample_rate": sample_rate,
            "stream_audio_samples": stream_samples,
        }
        if str(media.url or "").startswith("microphone://"):
            source["capture_mode"] = "microphone"
            source["title"] = "Microphone recording"
        elif str(media.url or "").startswith("mixed-audio://"):
            source["capture_mode"] = "mixed"
            source["title"] = "Computer audio + microphone recording"
        elif str(media.url or "").startswith("browser-stream://"):
            source["capture_mode"] = "browser-stream"
            source["title"] = "Browser audio recording"
        elif str(media.url or "").startswith("local-audio://"):
            source["capture_mode"] = "audio-file"
            source["title"] = self._session_source_title or Path(media.audio_file).name
        else:
            source["capture_mode"] = "youtube"
            if self._session_source_title:
                source["title"] = self._session_source_title
        source["duration_seconds"] = round(duration_seconds, 4)
        return source

    def session_snapshot(self) -> dict[str, Any]:
        session_state = getattr(self, "_session_state", None)
        transaction = session_state.transaction() if session_state is not None else nullcontext()
        with transaction:
            rows, embeddings = self._session_transcript_rows_and_embeddings()
            speaker_state = self.speaker_state()
            source = self._session_source_metadata()
            return {
                "id": str(self._session_id or ""),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                # The top-level duration describes how far this session was
                # processed. The complete media length remains in ``source``.
                "duration_seconds": float(self.playback_time()),
                "source": source,
                "transcript_rows": rows,
                "speaker_state": speaker_state,
                "speaker_profiles": self._session_speaker_profiles(),
                "live_speaker_profiles": self._serialized_live_speaker_profiles(
                    self.memory.export_profiles(),
                    portable=True,
                ),
                "embedding_records": embeddings,
                "embedding_provider": str(self.args.embedding_provider),
                "live_embedding_provider": self._current_live_embedding_provider(),
            }

    def current_session_id(self) -> str:
        """Return the active durable-session id without constructing a full snapshot."""

        return str(self._session_id or "")

    def write_session_audio(self, path: Path) -> bool:
        timeline = getattr(self, "_audio_timeline", None)
        if timeline is not None:
            snapshot = timeline.snapshot(copy_audio=True)
            if not snapshot.streaming or snapshot.audio.size <= 0:
                return False
            write_wav(path, snapshot.audio, snapshot.sample_rate)
            return True
        with self._audio_lock:
            if not self._streaming_audio or not self._stream_audio_chunks:
                return False
            audio = np.concatenate([chunk.astype(np.float32, copy=True) for chunk in self._stream_audio_chunks])
            sample_rate = int(self.sample_rate)
        write_wav(path, audio, sample_rate)
        return True
