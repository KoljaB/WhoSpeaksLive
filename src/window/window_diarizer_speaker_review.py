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




class WindowSpeakerReviewMixin:
    def rename_speaker(self, speaker_id: str, name: str) -> dict[str, Any]:
        label = str(speaker_id or "").strip()
        if not re.fullmatch(r"S\d+", label):
            raise ValueError("Invalid speaker id.")
        self._sync_metadata_with_memory()
        clean_name = " ".join(str(name or "").strip().split())[:80]
        with self._speaker_lock:
            if label not in self._speaker_metadata:
                raise ValueError(f"Unknown speaker {label}.")
            self._speaker_metadata[label]["name"] = clean_name
        self.bus.emit("status", {"message": f"Renamed {label} to {clean_name or label}."})
        return self.emit_speaker_state()

    def clear_speakers(self) -> dict[str, Any]:
        with self._live_memory_update_lock_obj():
            self._speaker_generation = int(getattr(self, "_speaker_generation", 0)) + 1
            jobs = getattr(self, "_embedding_jobs", None)
            if jobs is not None:
                self._cancel_pending_embedding_jobs(jobs)
            self._cancel_pending_live_memory_update_jobs()
            self.memory = self._new_memory()
            self._reset_live_speaker_memory()
        with self._speaker_lock:
            self._speaker_metadata = {}
            self._seed_profiles = []
            self._seed_live_profiles = []
            self._speaker_group_name = ""
        with self._unknown_lock:
            self._clear_unknown_sentence_state_locked()
        self._clear_sentence_refinement_records()
        self.bus.emit("status", {"message": "Cleared speakers."})
        return self.emit_speaker_state()

    def _speaker_exists(self, label: str) -> bool:
        return any(str(profile.get("label") or "") == label for profile in self.memory.export_profiles())

    def _live_update_speaker_exists(self, label: str) -> bool:
        memory = getattr(self, "memory", None)
        export_profiles = getattr(memory, "export_profiles", None)
        if not callable(export_profiles):
            return True
        return any(str(profile.get("label") or "") == label for profile in export_profiles())

    def _next_speaker_label(self) -> str:
        indexes: set[int] = set()
        for profile in self.memory.export_profiles():
            key = str(profile.get("label") or "").strip().upper()
            if key.startswith("S") and key[1:].isdigit():
                indexes.add(int(key[1:]))
        with self._speaker_lock:
            metadata_labels = list(self._speaker_metadata)
        for label in metadata_labels:
            key = str(label or "").strip().upper()
            if key.startswith("S") and key[1:].isdigit():
                indexes.add(int(key[1:]))
        with self._sentence_refinement_lock:
            for record in self._sentence_refinement_records.values():
                key = str(record.get("assigned_speaker") or "").strip().upper()
                if key.startswith("S") and key[1:].isdigit():
                    indexes.add(int(key[1:]))
        index = 1
        while index in indexes:
            index += 1
        return f"S{index}"

    def _copy_sentence_records(self) -> dict[int, dict[str, Any]]:
        with self._sentence_refinement_lock:
            copied: dict[int, dict[str, Any]] = {}
            for index, record in self._sentence_refinement_records.items():
                next_record = copy.deepcopy(record)
                if record.get("embedding") is not None:
                    next_record["embedding"] = self._embedding_copy(record.get("embedding"))
                copied[int(index)] = next_record
            return copied

    def _push_correction_history(self, action: str) -> None:
        session_state = getattr(self, "_session_state", None)
        transaction = session_state.transaction(mutate=True) if session_state is not None else nullcontext()
        with transaction, self._speaker_lock:
            metadata = copy.deepcopy(self._speaker_metadata)
            group_name = self._speaker_group_name
            seed_profiles = copy.deepcopy(self._seed_profiles)
            seed_live_profiles = copy.deepcopy(getattr(self, "_seed_live_profiles", []))
            speaker_last_media_end = copy.deepcopy(getattr(self, "_speaker_last_media_end", {}))
            self._correction_history.append({
                "action": str(action or "correction"),
                "records": self._copy_sentence_records(),
                "speaker_profiles": copy.deepcopy(self.memory.export_profiles()),
                "speaker_metadata": metadata,
                "speaker_group_name": group_name,
                "seed_profiles": seed_profiles,
                "seed_live_profiles": seed_live_profiles,
                "speaker_last_media_end": speaker_last_media_end,
            })
        if len(self._correction_history) > 50:
            self._correction_history = self._correction_history[-50:]

    def _restore_correction_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._live_memory_update_lock_obj():
            self._speaker_generation = int(getattr(self, "_speaker_generation", 0)) + 1
            jobs = getattr(self, "_embedding_jobs", None)
            if jobs is not None:
                self._cancel_pending_embedding_jobs(jobs)
            self._cancel_pending_live_memory_update_jobs()
            self.memory.replace_profiles(list(snapshot.get("speaker_profiles") or []))
            with self._speaker_lock:
                self._speaker_metadata = copy.deepcopy(snapshot.get("speaker_metadata") or {})
                self._speaker_group_name = str(snapshot.get("speaker_group_name") or "")
                self._seed_profiles = copy.deepcopy(snapshot.get("seed_profiles") or [])
                self._seed_live_profiles = copy.deepcopy(snapshot.get("seed_live_profiles") or [])
                self._speaker_last_media_end = copy.deepcopy(snapshot.get("speaker_last_media_end") or {})
            self._reset_live_speaker_memory()
        with self._sentence_refinement_lock:
            self._sentence_refinement_records = copy.deepcopy(snapshot.get("records") or {})
        with self._unknown_lock:
            self._clear_unknown_sentence_state_locked()

    def _duration_for_record(self, record: dict[str, Any]) -> float:
        try:
            duration = float(record.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0.0:
            return duration
        base = record.get("base_payload") if isinstance(record.get("base_payload"), dict) else {}
        try:
            start = float(base.get("start"))
            end = float(base.get("end"))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, end - start)

    def _rebuild_speaker_memory_from_records(
        self,
        records: dict[int, dict[str, Any]],
        *,
        remove_labels: set[str] | None = None,
    ) -> None:
        removed = {str(label or "").strip().upper() for label in (remove_labels or set()) if label}
        current_profiles = {
            str(profile.get("label") or ""): dict(profile)
            for profile in self.memory.export_profiles()
            if str(profile.get("label") or "")
        }
        embeddings_by_label: dict[str, list[tuple[np.ndarray, float]]] = {}
        for record in records.values():
            label = str(record.get("assigned_speaker") or "").strip().upper()
            if not label or label in removed:
                continue
            embedding = self._embedding_copy(record.get("embedding"))
            if embedding is None:
                continue
            duration = self._duration_for_record(record)
            embeddings_by_label.setdefault(label, []).append((embedding, max(0.05, duration)))

        labels = set(current_profiles) | set(embeddings_by_label)
        labels.difference_update(removed)
        rebuilt: list[dict[str, Any]] = []
        now = time.time()
        for label in sorted(labels, key=self._speaker_label_sort_key):
            samples = embeddings_by_label.get(label) or []
            current = current_profiles.get(label, {})
            if samples:
                weighted = None
                total_weight = 0.0
                for embedding, weight in samples:
                    if weighted is None:
                        weighted = embedding.astype(np.float32) * weight
                    else:
                        weighted = weighted + (embedding.astype(np.float32) * weight)
                    total_weight += weight
                assert weighted is not None
                centroid = weighted / max(total_weight, 0.0001)
                norm = float(np.linalg.norm(centroid))
                if norm > 0.0:
                    centroid = centroid / norm
                sentence_count = len(samples)
                speech_seconds = sum(weight for _embedding, weight in samples)
            else:
                centroid = np.asarray(current.get("centroid"), dtype=np.float32).reshape(-1)
                if centroid.size <= 0:
                    continue
                sentence_count = max(1, int(current.get("sentence_count") or 1))
                speech_seconds = float(current.get("speech_seconds") or 0.0)
            rebuilt.append({
                "label": label,
                "centroid": centroid.astype(float).tolist(),
                "sentence_count": max(1, int(sentence_count)),
                "speech_seconds": float(speech_seconds),
                "created_at": float(current.get("created_at") or now),
                "last_seen_at": now,
                "locked": bool(current.get("locked")),
            })

        self.memory.replace_profiles(rebuilt)
        self._sync_metadata_with_memory()
        self._reset_live_speaker_memory()

    def _set_user_assignment(
        self,
        record: dict[str, Any],
        speaker_id: str | None,
        *,
        action: str,
        updates_memory: bool,
    ) -> None:
        previous_speaker = record.get("assigned_speaker")
        if "automatic_assigned_speaker" not in record:
            record["automatic_assigned_speaker"] = previous_speaker
            record["automatic_assignment_source"] = str(record.get("assignment_source") or "")
        rejected_speakers = rejected_speaker_labels(record)
        previous_label = str(previous_speaker or "").strip()
        if previous_label and previous_label.upper() != "UNKNOWN" and previous_label != speaker_id:
            rejected_speakers.add(previous_label)
        original_label = str(record.get("automatic_assigned_speaker") or "").strip()
        if original_label and original_label.upper() != "UNKNOWN" and original_label != speaker_id:
            rejected_speakers.add(original_label)
        if speaker_id:
            rejected_speakers.discard(speaker_id)
        record["assigned_speaker"] = speaker_id
        record["created_speaker"] = False
        if speaker_id:
            probability_key = self._speaker_probability_key(speaker_id)
            record["probabilities"] = {probability_key: 1.0} if probability_key else {}
            record["unknown_probability"] = 0.0
            record["top_similarity"] = 1.0
            record["margin"] = 1.0
            self._ensure_speaker_metadata(speaker_id)
        else:
            record["probabilities"] = {"unknown": 1.0}
            record["unknown_probability"] = 1.0
            record["top_similarity"] = None
            record["margin"] = None
        record["assignment_source"] = "user_correction"
        record["correction"] = {
            "status": "user_corrected",
            "action": action,
            "original_speaker": record.get("automatic_assigned_speaker"),
            "previous_speaker": previous_speaker,
            "corrected_speaker": speaker_id,
            "rejected_speakers": sorted(rejected_speakers),
            "corrected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "updates_memory": bool(updates_memory),
        }

    def _emit_records(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.bus.emit("sentence", self._record_to_sentence_payload(record))

    def reassign_sentence(
        self,
        index: int,
        speaker_id: str | None,
        *,
        update_memory: bool = True,
    ) -> dict[str, Any]:
        return self.reassign_sentences([index], speaker_id, update_memory=update_memory)

    def reassign_sentences(
        self,
        indices: list[int],
        speaker_id: str | None,
        *,
        update_memory: bool = True,
    ) -> dict[str, Any]:
        target = self._normalized_speaker_label(speaker_id)
        if target and not self._speaker_exists(target):
            raise ValueError(f"Unknown speaker {target}.")
        indexes = sorted({int(index) for index in indices})
        if not indexes:
            raise ValueError("Choose at least one sentence to reassign.")
        with self._sentence_refinement_lock:
            for row_index in indexes:
                if row_index not in self._sentence_refinement_records:
                    raise ValueError(f"Unknown transcript row {row_index}.")
        self._push_correction_history("reassign_sentence" if len(indexes) == 1 else "reassign_sentences")
        changed: list[dict[str, Any]] = []
        with self._live_memory_update_lock_obj():
            with self._sentence_refinement_lock:
                for row_index in indexes:
                    record = self._sentence_refinement_records.get(row_index)
                    if record is None:
                        raise ValueError(f"Unknown transcript row {row_index}.")
                    self._set_user_assignment(
                        record,
                        target,
                        action="reassign",
                        updates_memory=update_memory,
                    )
                    changed.append(copy.deepcopy(record))
                records_copy = copy.deepcopy(self._sentence_refinement_records)
            if update_memory:
                self._rebuild_speaker_memory_from_records(records_copy)
            if target:
                for row_index in indexes:
                    self._remove_unknown_sentence(row_index)
        self._emit_records(changed)
        state = self.emit_speaker_state()
        if len(indexes) == 1:
            message = f"Reassigned sentence {indexes[0]} to {target or 'UNKNOWN'}."
        else:
            message = f"Reassigned {len(indexes)} sentences to {target or 'UNKNOWN'}."
        self.bus.emit("status", {"message": message})
        return {"speaker_state": state, "rows": [self._record_to_sentence_payload(record) for record in changed]}

    def mark_sentence_correct(self, index: int) -> dict[str, Any]:
        return self.mark_sentences_correct([index])

    def mark_sentences_correct(self, indices: list[int]) -> dict[str, Any]:
        indexes = sorted({int(index) for index in indices})
        if not indexes:
            raise ValueError("Choose at least one sentence to mark correct.")
        with self._sentence_refinement_lock:
            for row_index in indexes:
                if row_index not in self._sentence_refinement_records:
                    raise ValueError(f"Unknown transcript row {row_index}.")
        self._push_correction_history("mark_sentence_correct" if len(indexes) == 1 else "mark_sentences_correct")
        changed: list[dict[str, Any]] = []
        with self._live_memory_update_lock_obj():
            with self._sentence_refinement_lock:
                for row_index in indexes:
                    record = self._sentence_refinement_records.get(row_index)
                    if record is None:
                        raise ValueError(f"Unknown transcript row {row_index}.")
                    if "automatic_assigned_speaker" not in record:
                        record["automatic_assigned_speaker"] = record.get("assigned_speaker")
                        record["automatic_assignment_source"] = str(record.get("assignment_source") or "")
                    record["correction"] = {
                        "status": "user_confirmed",
                        "action": "mark_correct",
                        "original_speaker": record.get("automatic_assigned_speaker"),
                        "corrected_speaker": record.get("assigned_speaker"),
                        "corrected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "updates_memory": True,
                    }
                    changed.append(copy.deepcopy(record))
                records_copy = copy.deepcopy(self._sentence_refinement_records)
            self._rebuild_speaker_memory_from_records(records_copy)
        self._emit_records(changed)
        if len(indexes) == 1:
            message = f"Marked sentence {indexes[0]} correct."
        else:
            message = f"Marked {len(indexes)} sentences correct."
        self.bus.emit("status", {"message": message})
        return {"speaker_state": self.speaker_state(), "rows": [self._record_to_sentence_payload(record) for record in changed]}

    def split_speaker(
        self,
        speaker_id: str,
        sentence_indices: list[int],
        *,
        name: str = "",
        update_memory: bool = True,
    ) -> dict[str, Any]:
        source = self._normalized_speaker_label(speaker_id)
        if not source or not self._speaker_exists(source):
            raise ValueError(f"Unknown speaker {speaker_id}.")
        indexes = sorted({int(index) for index in sentence_indices})
        if not indexes:
            raise ValueError("Choose at least one sentence to split.")
        new_label = self._next_speaker_label()
        clean_name = " ".join(str(name or "").strip().split())[:80]
        with self._sentence_refinement_lock:
            for index in indexes:
                record = self._sentence_refinement_records.get(index)
                if record is None:
                    raise ValueError(f"Unknown transcript row {index}.")
                if str(record.get("assigned_speaker") or "").strip().upper() != source:
                    raise ValueError(f"Sentence {index} is not assigned to {source}.")
        self._push_correction_history("split_speaker")
        changed: list[dict[str, Any]] = []
        with self._live_memory_update_lock_obj():
            with self._speaker_lock:
                self._speaker_metadata[new_label] = {
                    "name": clean_name,
                    "source": "user_split",
                    "locked": False,
                    "reference_audio": "",
                }
            with self._sentence_refinement_lock:
                for index in indexes:
                    record = self._sentence_refinement_records.get(index)
                    if record is None:
                        raise ValueError(f"Unknown transcript row {index}.")
                    if str(record.get("assigned_speaker") or "").strip().upper() != source:
                        raise ValueError(f"Sentence {index} is not assigned to {source}.")
                    self._set_user_assignment(
                        record,
                        new_label,
                        action="split",
                        updates_memory=update_memory,
                    )
                    changed.append(copy.deepcopy(record))
                records_copy = copy.deepcopy(self._sentence_refinement_records)
            if update_memory:
                self._rebuild_speaker_memory_from_records(records_copy)
        self._emit_records(changed)
        state = self.emit_speaker_state()
        self.bus.emit(
            "status",
            {"message": f"Created {new_label} from {len(indexes)} sentence(s) previously assigned to {source}."},
        )
        return {
            "speaker_state": state,
            "new_speaker_id": new_label,
            "rows": [self._record_to_sentence_payload(record) for record in changed],
        }

    def merge_speakers(
        self,
        source_speaker_id: str,
        target_speaker_id: str,
        *,
        update_memory: bool = True,
    ) -> dict[str, Any]:
        source = self._normalized_speaker_label(source_speaker_id)
        target = self._normalized_speaker_label(target_speaker_id)
        if not source or not target or source == target:
            raise ValueError("Choose two different speakers to merge.")
        if not self._speaker_exists(source):
            raise ValueError(f"Unknown speaker {source}.")
        if not self._speaker_exists(target):
            raise ValueError(f"Unknown speaker {target}.")
        self._push_correction_history("merge_speakers")
        changed: list[dict[str, Any]] = []
        with self._live_memory_update_lock_obj():
            with self._sentence_refinement_lock:
                for record in self._sentence_refinement_records.values():
                    if str(record.get("assigned_speaker") or "").strip().upper() != source:
                        continue
                    self._set_user_assignment(
                        record,
                        target,
                        action="merge",
                        updates_memory=update_memory,
                    )
                    changed.append(copy.deepcopy(record))
                records_copy = copy.deepcopy(self._sentence_refinement_records)
            with self._speaker_lock:
                self._speaker_metadata.pop(source, None)
            if update_memory:
                self._rebuild_speaker_memory_from_records(records_copy, remove_labels={source})
            else:
                profiles = [profile for profile in self.memory.export_profiles() if profile.get("label") != source]
                self.memory.replace_profiles(profiles)
                self._sync_metadata_with_memory()
                self._reset_live_speaker_memory()
        self._emit_records(changed)
        state = self.emit_speaker_state()
        self.bus.emit("status", {"message": f"Merged {source} into {target}."})
        return {"speaker_state": state, "rows": [self._record_to_sentence_payload(record) for record in changed]}

    def delete_speaker(self, speaker_id: str, *, update_memory: bool = True) -> dict[str, Any]:
        target = self._normalized_speaker_label(speaker_id)
        if not target or not self._speaker_exists(target):
            raise ValueError(f"Unknown speaker {speaker_id}.")
        self._push_correction_history("delete_speaker")
        changed: list[dict[str, Any]] = []
        with self._live_memory_update_lock_obj():
            with self._sentence_refinement_lock:
                for record in self._sentence_refinement_records.values():
                    previous_speaker = str(record.get("assigned_speaker") or "").strip().upper()
                    if previous_speaker != target:
                        continue
                    if "automatic_assigned_speaker" not in record:
                        record["automatic_assigned_speaker"] = record.get("assigned_speaker")
                        record["automatic_assignment_source"] = str(record.get("assignment_source") or "")
                    record["assigned_speaker"] = None
                    record["created_speaker"] = False
                    record["probabilities"] = {"unknown": 1.0}
                    record["similarities"] = {}
                    record["unknown_probability"] = 1.0
                    record["top_similarity"] = None
                    record["margin"] = None
                    record["assignment_source"] = "user_deleted_speaker"
                    record["correction"] = {
                        "status": "speaker_deleted",
                        "action": "delete_speaker",
                        "deleted_speaker": target,
                        "previous_speaker": previous_speaker,
                        "corrected_speaker": None,
                        "corrected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "updates_memory": bool(update_memory),
                    }
                    changed.append(copy.deepcopy(record))
                records_copy = copy.deepcopy(self._sentence_refinement_records)
            with self._speaker_lock:
                self._speaker_metadata.pop(target, None)
            self._speaker_last_media_end.pop(target, None)
            if update_memory:
                self._rebuild_speaker_memory_from_records(records_copy, remove_labels={target})
            else:
                profiles = [profile for profile in self.memory.export_profiles() if profile.get("label") != target]
                self.memory.replace_profiles(profiles)
                self._sync_metadata_with_memory()
                self._reset_live_speaker_memory()
        self._emit_records(changed)
        state = self.emit_speaker_state()
        row_count = len(changed)
        if row_count:
            message = f"Deleted {target} and moved {row_count} sentence{'' if row_count == 1 else 's'} to UNKNOWN."
        else:
            message = f"Deleted empty speaker {target}."
        self.bus.emit("status", {"message": message})
        return {"speaker_state": state, "rows": [self._record_to_sentence_payload(record) for record in changed]}

    def undo_last_correction(self) -> dict[str, Any]:
        if not self._correction_history:
            raise ValueError("No correction to undo.")
        snapshot = self._correction_history.pop()
        self._restore_correction_snapshot(snapshot)
        with self._sentence_refinement_lock:
            records = [copy.deepcopy(record) for _index, record in sorted(self._sentence_refinement_records.items())]
        self._emit_records(records)
        state = self.emit_speaker_state()
        self.bus.emit("status", {"message": f"Undid {snapshot.get('action') or 'correction'}."})
        return {"speaker_state": state, "rows": [self._record_to_sentence_payload(record) for record in records]}
