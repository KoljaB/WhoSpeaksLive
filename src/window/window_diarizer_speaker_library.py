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




class WindowSpeakerLibraryMixin:
    def _speaker_correction_summary(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        corrected_rows = 0
        confirmed_rows = 0
        record_map = getattr(self, "_sentence_refinement_records", {})
        lock = getattr(self, "_sentence_refinement_lock", None)
        if lock is None:
            records = list(record_map.values()) if isinstance(record_map, dict) else []
        else:
            with lock:
                records = list(record_map.values()) if isinstance(record_map, dict) else []
        for record in records:
            correction = record.get("correction")
            if not isinstance(correction, dict) or not correction:
                continue
            action = str(correction.get("action") or "correction")
            counts[action] += 1
            if correction.get("status") == "user_corrected":
                corrected_rows += 1
            if correction.get("status") == "user_confirmed":
                confirmed_rows += 1
        return {
            "corrected_rows": corrected_rows,
            "confirmed_rows": confirmed_rows,
            "actions": dict(sorted(counts.items())),
        }

    def save_speaker_group(self, name: str) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        self._drain_embedding_jobs()
        self._drain_live_memory_update_jobs()
        group_name = safe_library_name(name)
        group_dir = speaker_group_dir(self.speaker_library_dir, group_name)
        references_dir = group_dir / "references"

        profiles = self.memory.export_profiles()
        if not profiles:
            raise ValueError("No speakers to save yet.")
        group_dir.mkdir(parents=True, exist_ok=True)
        references_dir.mkdir(parents=True, exist_ok=True)
        with self._speaker_lock:
            metadata_by_label = {
                label: dict(metadata)
                for label, metadata in self._speaker_metadata.items()
            }

        saved_profiles: list[dict[str, Any]] = []
        for profile in profiles:
            label = str(profile["label"])
            metadata = metadata_by_label.get(label, {})
            reference_audio = str(metadata.get("reference_audio") or "")
            saved_reference = ""
            if reference_audio:
                source_path = Path(reference_audio)
                if source_path.is_file():
                    source_resolved = source_path.resolve()
                    try:
                        source_resolved.relative_to(references_dir.resolve())
                        target = source_resolved
                    except ValueError:
                        target = references_dir / f"{label}_{safe_reference_filename(source_path.name)}"
                    if source_resolved != target.resolve():
                        shutil.copy2(source_path, target)
                    saved_reference = str(target.resolve().relative_to(group_dir.resolve()))
            saved_profiles.append({
                "label": label,
                "name": str(metadata.get("name") or ""),
                "source": str(metadata.get("source") or "detected"),
                "locked": bool(metadata.get("locked") or profile.get("locked")),
                "reference_audio": saved_reference,
                "centroid": profile["centroid"],
                "sentence_count": int(profile.get("sentence_count") or 1),
                "speech_seconds": float(profile.get("speech_seconds") or 0.0),
                "created_at": float(profile.get("created_at") or time.time()),
                "last_seen_at": float(profile.get("last_seen_at") or time.time()),
            })

        saved_live_profiles = self._serialized_live_speaker_profiles(profiles, portable=False)
        manifest = {
            "version": 1,
            "name": group_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_provider": self.args.embedding_provider,
            "embedding_device": self.args.embedding_device,
            "live_embedding_provider": self._current_live_embedding_provider(),
            "correction_summary": self._speaker_correction_summary(),
            "live_speakers": saved_live_profiles,
            "speakers": saved_profiles,
        }
        (group_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._speaker_lock:
            self._speaker_group_name = group_name
            self._seed_profiles = [
                {
                    "label": profile["label"],
                    "centroid": profile["centroid"],
                    "sentence_count": profile["sentence_count"],
                    "speech_seconds": profile["speech_seconds"],
                    "locked": profile["locked"],
                    "metadata": {
                        "name": profile["name"],
                        "source": profile["source"],
                        "locked": profile["locked"],
                        "reference_audio": str((group_dir / profile["reference_audio"]).resolve()) if profile["reference_audio"] else "",
                    },
                }
                for profile in saved_profiles
            ]
            self._seed_live_profiles = [
                {
                    "label": profile["label"],
                    "centroid": profile["centroid"],
                    "sentence_count": profile["sentence_count"],
                    "speech_seconds": profile["speech_seconds"],
                    "locked": profile["locked"],
                }
                for profile in saved_live_profiles
            ]
        self.bus.emit("status", {"message": f"Saved speaker group {group_name}."})
        return self.emit_speaker_state()

    def load_speaker_group(self, name: str) -> dict[str, Any]:
        group_name = safe_library_name(name)
        group_dir = speaker_group_dir(self.speaker_library_dir, group_name)
        manifest_path = group_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Speaker group {group_name} does not exist.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provider = str(manifest.get("embedding_provider") or "")
        if provider and provider != self.args.embedding_provider:
            raise ValueError(
                f"Speaker group uses embedding provider {provider!r}, current provider is {self.args.embedding_provider!r}."
            )
        raw_profiles = manifest.get("speakers")
        if not isinstance(raw_profiles, list):
            raise ValueError("Speaker group manifest has no speaker list.")

        seed_profiles: list[dict[str, Any]] = []
        metadata_by_label: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(raw_profiles, 1):
            if not isinstance(item, dict):
                continue
            label = f"S{index}"
            reference_audio = str(item.get("reference_audio") or "")
            if reference_audio:
                reference_audio = str((group_dir / reference_audio).resolve())
            metadata_by_label[label] = {
                "name": str(item.get("name") or ""),
                "source": str(item.get("source") or "detected"),
                "locked": bool(item.get("locked")),
                "reference_audio": reference_audio,
            }
            seed_profiles.append({
                "label": label,
                "centroid": item["centroid"],
                "sentence_count": max(1, int(item.get("sentence_count") or 1)),
                "speech_seconds": float(item.get("speech_seconds") or 0.0),
                "locked": bool(item.get("locked")),
                "metadata": metadata_by_label[label],
            })

        seed_live_profiles = self._live_seed_profiles_from_manifest(manifest)
        with self._live_memory_update_lock_obj():
            self._speaker_generation = int(getattr(self, "_speaker_generation", 0)) + 1
            jobs = getattr(self, "_embedding_jobs", None)
            if jobs is not None:
                self._cancel_pending_embedding_jobs(jobs)
            self._cancel_pending_live_memory_update_jobs()
            self.memory = self._new_memory()
            with self._speaker_lock:
                self._speaker_group_name = group_name
                self._speaker_metadata = metadata_by_label
                self._seed_profiles = [dict(item) for item in seed_profiles]
                self._seed_live_profiles = [dict(item) for item in seed_live_profiles]
            self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._clear_unknown_sentence_state_locked()
        self._clear_sentence_refinement_records()
        self.bus.emit("status", {"message": f"Loaded speaker group {group_name}."})
        return self.emit_speaker_state()

    @staticmethod
    def _centroid_payload(centroid: Any) -> dict[str, Any]:
        vector = np.asarray(centroid, dtype="<f4")
        return {
            "centroid": vector.astype(float).tolist(),
            "centroid_encoding": "float32-base64-le",
            "centroid_b64": base64.b64encode(vector.tobytes()).decode("ascii"),
            "centroid_length": int(vector.size),
        }

    @staticmethod
    def _centroid_from_payload(item: dict[str, Any]) -> list[float]:
        encoded = str(item.get("centroid_b64") or "")
        if encoded:
            raw = base64.b64decode(encoded)
            vector = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            expected_length = int(item.get("centroid_length") or 0)
            if expected_length and vector.size != expected_length:
                raise ValueError("Speaker centroid length does not match its encoded payload.")
            if vector.size <= 0:
                raise ValueError("Speaker centroid is empty.")
            return vector.astype(float).tolist()
        centroid = item.get("centroid")
        if not isinstance(centroid, list) or not centroid:
            raise ValueError("Speaker profile is missing a centroid.")
        return [float(value) for value in centroid]

    def export_speaker_group_file(self, name: str) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        self._drain_embedding_jobs()
        self._drain_live_memory_update_jobs()
        raw_name = str(name or "").strip() or self._speaker_group_name or "speakers"
        group_name = safe_library_name(raw_name)
        profiles = self.memory.export_profiles()
        if not profiles:
            raise ValueError("No speakers to save yet.")
        with self._speaker_lock:
            metadata_by_label = {
                label: dict(metadata)
                for label, metadata in self._speaker_metadata.items()
            }

        exported_speakers: list[dict[str, Any]] = []
        for profile in profiles:
            label = str(profile["label"])
            metadata = metadata_by_label.get(label, {})
            reference_audio_payload: dict[str, Any] | None = None
            reference_audio = str(metadata.get("reference_audio") or "")
            if reference_audio:
                reference_path = Path(reference_audio)
                if reference_path.is_file():
                    reference_bytes = reference_path.read_bytes()
                    media_type = mimetypes.guess_type(str(reference_path))[0] or "application/octet-stream"
                    reference_audio_payload = {
                        "filename": safe_reference_filename(reference_path.name),
                        "media_type": media_type,
                        "data_url": f"data:{media_type};base64,{base64.b64encode(reference_bytes).decode('ascii')}",
                    }
            exported_speaker = {
                "label": label,
                "name": str(metadata.get("name") or ""),
                "source": str(metadata.get("source") or "detected"),
                "locked": bool(metadata.get("locked") or profile.get("locked")),
                "reference_audio": reference_audio_payload,
                "sentence_count": int(profile.get("sentence_count") or 1),
                "speech_seconds": float(profile.get("speech_seconds") or 0.0),
                "created_at": float(profile.get("created_at") or time.time()),
                "last_seen_at": float(profile.get("last_seen_at") or time.time()),
            }
            exported_speaker.update(self._centroid_payload(profile["centroid"]))
            exported_speakers.append(exported_speaker)

        exported_live_speakers = self._serialized_live_speaker_profiles(profiles, portable=True)
        with self._speaker_lock:
            self._speaker_group_name = group_name
        return {
            "version": 1,
            "format": "whospeaks-speaker-group",
            "name": group_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_provider": self.args.embedding_provider,
            "embedding_device": self.args.embedding_device,
            "live_embedding_provider": self._current_live_embedding_provider(),
            "correction_summary": self._speaker_correction_summary(),
            "live_speakers": exported_live_speakers,
            "speakers": exported_speakers,
        }

    def import_speaker_group_file(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(manifest, dict):
            raise ValueError("Speaker group file is invalid.")
        if str(manifest.get("format") or "") not in {"", "whospeaks-speaker-group"}:
            raise ValueError("Speaker group file is not a WhoSpeaks speaker group.")
        provider = str(manifest.get("embedding_provider") or "")
        if provider and provider != self.args.embedding_provider:
            raise ValueError(
                f"Speaker group uses embedding provider {provider!r}, current provider is {self.args.embedding_provider!r}."
            )
        raw_profiles = manifest.get("speakers")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("Speaker group file has no speaker list.")

        group_name = safe_library_name(str(manifest.get("name") or "imported_speakers"))
        imported_references_dir = self.speaker_library_dir / "_imported_references" / group_name
        seed_profiles: list[dict[str, Any]] = []
        metadata_by_label: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(raw_profiles, 1):
            if not isinstance(item, dict):
                continue
            label = f"S{index}"
            reference_audio = ""
            reference_payload = item.get("reference_audio")
            if isinstance(reference_payload, dict):
                data_url = str(reference_payload.get("data_url") or "")
                if data_url:
                    raw_audio = base64.b64decode(data_url.split(",", 1)[-1])
                    imported_references_dir.mkdir(parents=True, exist_ok=True)
                    filename = safe_reference_filename(str(reference_payload.get("filename") or f"{label}.wav"))
                    reference_path = imported_references_dir / f"{label}_{filename}"
                    reference_path.write_bytes(raw_audio)
                    reference_audio = str(reference_path)
            metadata_by_label[label] = {
                "name": str(item.get("name") or ""),
                "source": str(item.get("source") or "detected"),
                "locked": bool(item.get("locked")),
                "reference_audio": reference_audio,
            }
            seed_profiles.append({
                "label": label,
                "centroid": self._centroid_from_payload(item),
                "sentence_count": max(1, int(item.get("sentence_count") or 1)),
                "speech_seconds": float(item.get("speech_seconds") or 0.0),
                "locked": bool(item.get("locked")),
                "metadata": metadata_by_label[label],
            })
        if not seed_profiles:
            raise ValueError("Speaker group file has no usable speaker profiles.")

        seed_live_profiles = self._live_seed_profiles_from_manifest(manifest)
        with self._live_memory_update_lock_obj():
            self._speaker_generation = int(getattr(self, "_speaker_generation", 0)) + 1
            jobs = getattr(self, "_embedding_jobs", None)
            if jobs is not None:
                self._cancel_pending_embedding_jobs(jobs)
            self._cancel_pending_live_memory_update_jobs()
            self.memory = self._new_memory()
            with self._speaker_lock:
                self._speaker_group_name = group_name
                self._speaker_metadata = metadata_by_label
                self._seed_profiles = [dict(item) for item in seed_profiles]
                self._seed_live_profiles = [dict(item) for item in seed_live_profiles]
            self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._clear_unknown_sentence_state_locked()
        self.bus.emit("status", {"message": f"Imported speaker group {group_name}."})
        return self.emit_speaker_state()

    def add_reference_speaker(self, name: str, filename: str, audio_b64: str) -> dict[str, Any]:
        clean_name = " ".join(str(name or "").strip().split())[:80]
        if not clean_name:
            raise ValueError("Reference speaker name must not be empty.")
        if not audio_b64:
            raise ValueError("Reference audio is missing.")
        raw_audio = base64.b64decode(str(audio_b64).split(",", 1)[-1])
        upload_dir = self.speaker_library_dir / "_uploaded_references"
        upload_dir.mkdir(parents=True, exist_ok=True)
        reference_path = upload_dir / f"{uuid.uuid4().hex}_{safe_reference_filename(filename)}"
        reference_path.write_bytes(raw_audio)

        audio, sample_rate = load_audio_file(reference_path)
        duration_seconds = len(audio) / float(sample_rate or 16000)
        if duration_seconds <= 0.0:
            raise ValueError("Reference audio is empty.")
        embedding = self.embedding.embed_wav(reference_path)
        label = self.memory.add_profile(
            embedding,
            duration_seconds=duration_seconds,
            sentence_count=1,
            locked=True,
        )
        if self._live_embedding_separate:
            self._update_live_speaker_memory(
                label,
                pad_audio(trim_silence(audio, sample_rate), self.args.min_embed_seconds, sample_rate),
                sample_rate,
                duration_seconds,
                ".live-reference.wav",
                speaker_generation=self._speaker_generation,
            )
        with self._speaker_lock:
            self._speaker_metadata[label] = {
                "name": clean_name,
                "source": "reference",
                "locked": True,
                "reference_audio": str(reference_path),
            }
            self._seed_profiles.append({
                "label": label,
                "centroid": embedding.astype(float).tolist(),
                "sentence_count": 1,
                "speech_seconds": duration_seconds,
                "locked": True,
                "metadata": {
                    "name": clean_name,
                    "source": "reference",
                    "locked": True,
                    "reference_audio": str(reference_path),
                },
            })
        self.bus.emit("status", {"message": f"Added reference speaker {clean_name} as {label}."})
        return self.emit_speaker_state()

    def set_new_speaker_sensitivity(self, level: Any) -> dict[str, Any]:
        normalized = normalize_new_speaker_sensitivity(level)
        preset = NEW_SPEAKER_SENSITIVITY_PRESETS[normalized]
        self._update_config(
            new_speaker_sensitivity=normalized,
            new_speaker_sensitivity_label=preset["label"],
            **{key: preset[key] for key in NEW_SPEAKER_SENSITIVITY_FIELDS},
        )
        memory_lock = getattr(self.memory, "_lock", None)
        if memory_lock is not None:
            with memory_lock:
                self._apply_new_speaker_preset_to_memory(preset)
        else:
            self._apply_new_speaker_preset_to_memory(preset)
        payload = {
            "level": normalized,
            "label": str(preset["label"]),
            "settings": {
                key: preset[key]
                for key in NEW_SPEAKER_SENSITIVITY_FIELDS
            },
        }
        self.bus.emit(
            "status",
            {
                "message": (
                    f"New speaker sensitivity set to {normalized}. "
                    f"{preset['label']}."
                )
            },
        )
        return payload

    def speaker_refinement_settings(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.args, "speaker_refinement", True)),
            "unknown_tentative": bool(getattr(self.args, "speaker_refinement_unknown_tentative", True)),
            "unknown_commit": bool(getattr(self.args, "speaker_refinement_unknown_commit", True)),
            "allow_reassignment": bool(getattr(self.args, "allow_speaker_reassignment", False)),
        }

    def set_speaker_refinement_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "speaker_refinement_unknown_tentative",
            "speaker_refinement_unknown_commit",
            "allow_speaker_reassignment",
        }
        self._update_config(**{
            key: bool(value)
            for key, value in updates.items()
            if key in allowed
        })
        settings = self.speaker_refinement_settings()
        self.bus.emit("status", {"message": "Speaker refinement settings updated."})
        if settings["unknown_commit"]:
            self._revisit_unknown_sentences()
        if settings["unknown_tentative"] or settings["allow_reassignment"]:
            self._refine_speaker_assignments()
        return settings

    def set_allow_speaker_reassignment(self, enabled: Any) -> dict[str, Any]:
        value = bool(enabled)
        self._update_config(allow_speaker_reassignment=value)
        self.bus.emit(
            "status",
            {
                "message": (
                    "Later speaker reassignment enabled."
                    if value
                    else "Later speaker reassignment disabled; existing speaker labels stay stable."
                )
            },
        )
        if value:
            self._refine_speaker_assignments()
        return self.speaker_refinement_settings()

    def _apply_new_speaker_preset_to_memory(self, preset: dict[str, Any]) -> None:
        for key in NEW_SPEAKER_SENSITIVITY_FIELDS:
            if hasattr(self.memory, key):
                setattr(self.memory, key, preset[key])
