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


@dataclass(frozen=True)


class WindowSpeakerStateMixin:
    def initial_speaker_state(self) -> dict[str, Any]:
        if self.is_running():
            return self.speaker_state()
        return self._reset_runtime_session_state(emit=False)

    def _sync_metadata_with_memory(self) -> None:
        labels = {profile["label"] for profile in self.memory.export_profiles()}
        with self._speaker_lock:
            self._speaker_metadata = {
                label: metadata
                for label, metadata in self._speaker_metadata.items()
                if label in labels
            }
            for label in sorted(labels, key=lambda value: int(value[1:]) if value.startswith("S") and value[1:].isdigit() else 9999):
                self._speaker_metadata.setdefault(label, {
                    "name": "",
                    "source": "detected",
                    "locked": False,
                    "reference_audio": "",
                })

    def _ensure_speaker_metadata(self, label: str | None, source: str = "detected") -> None:
        if not label:
            return
        with self._speaker_lock:
            metadata = self._speaker_metadata.setdefault(label, {
                "name": "",
                "source": source,
                "locked": False,
                "reference_audio": "",
            })
            if source == "reference":
                metadata["source"] = "reference"

    def _speaker_info_for_payload(self, label: str | None) -> dict[str, Any]:
        if not label:
            return {"speaker_name": None, "speaker_source": None, "speaker_locked": False}
        with self._speaker_lock:
            metadata = dict(self._speaker_metadata.get(label) or {})
        return {
            "speaker_name": str(metadata.get("name") or ""),
            "speaker_source": str(metadata.get("source") or "detected"),
            "speaker_locked": bool(metadata.get("locked")),
        }

    def _live_public_identity_snapshot(
        self,
    ) -> tuple[int, dict[str, str], dict[str, str]]:
        with self._shared_live_speaker_lock():
            overlay = getattr(self, "_shared_live_speaker_open_set_overlay_state", None)
            if overlay is None:
                return 0, {}, {}
            return overlay.identity_snapshot()

    def _speaker_state(self) -> dict[str, Any]:
        profiles = self.memory.export_profiles()
        self._refresh_person_identity_suggestions(profiles)
        with self._speaker_lock:
            metadata_by_label = {
                label: dict(metadata)
                for label, metadata in self._speaker_metadata.items()
            }
            group_name = self._speaker_group_name
        speakers: list[dict[str, Any]] = []
        for profile in profiles:
            label = str(profile["label"])
            metadata = metadata_by_label.get(label, {})
            identity_status = str(metadata.get("identity_status") or "unidentified")
            suggested_name = str(metadata.get("suggested_person_name") or "")
            local_name = str(metadata.get("name") or "")
            if local_name:
                display_name = local_name
            elif identity_status == "suggested" and suggested_name:
                display_name = f"Likely {suggested_name}"
            else:
                display_name = f"Speaker {profile['index']}"
            speakers.append({
                "id": label,
                "name": local_name,
                "display_name": display_name,
                "source": str(metadata.get("source") or "detected"),
                "locked": bool(metadata.get("locked") or profile.get("locked")),
                "sentence_count": int(profile.get("sentence_count") or 0),
                "speech_seconds": round(float(profile.get("speech_seconds") or 0.0), 4),
                # Legacy meeting-local references may retain a local file, but
                # public state never exposes its absolute path.
                "reference_audio": "",
                "reference_audio_retained": bool(metadata.get("reference_audio")),
                "identity_status": identity_status,
                "person_id": str(metadata.get("person_id") or ""),
                "suggested_person_id": str(metadata.get("suggested_person_id") or ""),
                "suggested_person_name": suggested_name,
            })
        first_centroid = profiles[0].get("centroid") if profiles else None
        embedding_length = len(first_centroid) if first_centroid is not None else 0
        alias_generation, final_to_public, public_to_final = (
            self._live_public_identity_snapshot()
        )
        public_speakers: list[dict[str, Any]] = []
        public_seen: set[str] = set()
        for speaker in speakers:
            final_id = str(speaker.get("id") or "")
            public_id = str(final_to_public.get(final_id, final_id))
            if not public_id or public_id in public_seen:
                continue
            public_seen.add(public_id)
            public_speakers.append({
                **speaker,
                "id": public_id,
                "internal_speaker_id": final_id,
                "presentation_aliased": public_id != final_id,
            })
        return {
            "group_name": group_name,
            "groups": list_speaker_groups(self.speaker_library_dir),
            "speakers": speakers,
            "public_speakers": public_speakers,
            "public_identity_aliases": final_to_public,
            "public_identity_reverse_aliases": public_to_final,
            "public_identity_alias_generation": alias_generation,
            "embedding_provider": self.args.embedding_provider,
            "people": self.person_library.public_state(
                embedding_provider=str(self.args.embedding_provider),
                embedding_length=embedding_length,
            ),
            "expected_person_ids": sorted(self._expected_person_ids),
            "expected_people_filter_active": True,
        }

    def emit_speaker_state(self) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        self._maybe_checkpoint_confirmed_people(review_assignments=True)
        state = self._speaker_state()
        self.bus.emit("speakers", state)
        self._emit_speaker_memory_state(state, authoritative_final=False)
        return state

    def _emit_speaker_memory_state(
        self,
        state: dict[str, Any],
        *,
        authoritative_final: bool,
    ) -> None:
        emit_internal = getattr(self.bus, "emit_internal", None)
        if callable(emit_internal):
            emit_internal(
                "speaker_memory_state",
                {
                    "authoritative_final": bool(authoritative_final),
                    "phase": (
                        "post_final_refinement_pre_done"
                        if authoritative_final
                        else "incremental"
                    ),
                    "media_time": round(float(self.playback_time()), 6),
                    "speaker_generation": int(getattr(self, "_speaker_generation", 0)),
                    "final_provider": str(getattr(self.args, "embedding_provider", "")),
                    "live_provider": self._current_live_embedding_provider(),
                    "final_profiles": self.memory.export_profiles(),
                    "live_profiles": self.live_memory.export_profiles(),
                    "public_state": state,
                },
            )

    def emit_authoritative_final_speaker_memory_state(self) -> dict[str, Any]:
        """Record the post-refinement profile state without public side effects."""

        self._sync_metadata_with_memory()
        state = self._speaker_state()
        self._emit_speaker_memory_state(state, authoritative_final=True)
        return state

    def speaker_state(self) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        return self._speaker_state()
