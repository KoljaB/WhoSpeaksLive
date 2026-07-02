"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
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

from whospeaks.common.audio_utils import load_audio_file, pad_audio, trim_silence, write_wav
from whospeaks.embeddings.embedding_providers import EmbeddingSubprocessClient
from whospeaks.speakers.speaker_embedding_cluster import SpeakerMemory
from whospeaks.window.window_config import (
    DEFAULT_REALTIMESTT_ROOT,
    DEFAULT_SPEAKER_LIBRARY_DIR,
    NEW_SPEAKER_SENSITIVITY_FIELDS,
    SILERO_VAD_CHUNK_SAMPLES,
    SILERO_VAD_SAMPLE_RATE,
    apply_new_speaker_sensitivity,
    default_silero_vad_backend,
    list_speaker_groups,
    normalize_new_speaker_sensitivity,
    safe_library_name,
    safe_reference_filename,
    speaker_group_dir,
)
from whospeaks.window.window_domain import (
    EmbeddingSentenceJob,
    MediaFiles,
    PendingUnknownSentence,
    SentencePart,
    TimedWord,
    VadWindowState,
    WindowTranscript,
)
from whospeaks.window.window_events import EventBus
from whospeaks.window.window_media import resolve_browser_stream_id
from whospeaks.window.window_preview import (
    KrokoRealtimePreviewTranscriber,
    KrokoSubprocessPreviewTranscriber,
    MockRealtimePreviewTranscriber,
    RealtimePreviewTranscriber,
)
from whospeaks.window.window_remote_asr import RemoteWindowAsrClient
from whospeaks.window.window_text import (
    is_embedding_candidate_text,
    round_optional,
    split_words_with_stream2sentence,
    text_content_words,
    word_attr,
)

class WindowDiarizer:
    def __init__(self, args: argparse.Namespace, media: MediaFiles, bus: EventBus) -> None:
        self.args = args
        self.media = media
        self.bus = bus
        self._audio_lock = threading.Lock()
        self._streaming_audio = False
        self.audio, self.sample_rate = load_audio_file(media.audio_file)
        self._stream_audio_chunks: list[np.ndarray] = []
        self._stream_audio_samples = 0
        self.duration = len(self.audio) / float(self.sample_rate)
        self.embedding = EmbeddingSubprocessClient(args.embedding_python, args.embedding_provider, args.embedding_device)
        self.memory = self._new_memory()
        self.speaker_library_dir = Path(getattr(args, "speaker_library_dir", DEFAULT_SPEAKER_LIBRARY_DIR))
        self._speaker_lock = threading.Lock()
        self._speaker_group_name = ""
        self._speaker_metadata: dict[str, dict[str, Any]] = {}
        self._seed_profiles: list[dict[str, Any]] = []
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._session_id = ""
        self._playback_lock = threading.Lock()
        self._playback_time = 0.0
        self._playback_clock_started_at: float | None = None
        self._last_playback_jump_warning_at = 0.0
        self._unknown_lock = threading.Lock()
        self._unknown_sentences: list[PendingUnknownSentence] = []
        self._embedding_jobs: "queue.Queue[EmbeddingSentenceJob | None] | None" = None
        self._embedding_thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._live_probe_thread: threading.Thread | None = None
        self._preview_transcriber: RealtimePreviewTranscriber | None = None
        self._preview_lock = threading.Lock()
        self._preview_left = 0.0
        self._preview_generation = 0
        self._preview_paused = False
        self._vad_model: Any = None
        self._vad_model_backend = ""
        self._vad_model_error: str | None = None
        self._vad_model_lock = threading.Lock()
        self._sentence_splitter_warmed = False
        self._embedding_warmed = False
        self._asr_probe_warmed = False
        self._embedding_warmed_at: float | None = None
        self._asr_probe_warmed_at: float | None = None

    def prepare_before_browser_release(self) -> None:
        self.bus.emit(
            "status",
            {"message": "Preparing ASR, embeddings, and VAD before publishing the browser URL."},
        )
        self._prepare_model_dependencies(include_asr_probe=True)
        self.bus.emit("status", {"message": "Startup model warmup complete; browser GUI can be opened."})

    def start(self) -> None:
        self.bus.emit("status", {"message": "Start requested; preparing models before playback."})
        self.stop()
        refresh_runtime_warmup = self._should_refresh_start_runtime_warmup()
        self._prepare_model_dependencies(
            include_asr_probe=refresh_runtime_warmup,
            force_runtime_warmup=refresh_runtime_warmup,
        )
        self.bus.emit("status", {"message": "Loading realtime preview engine."})
        self._load_realtime_preview()
        self.memory = self._new_memory()
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._unknown_sentences = []
        self._reset_realtime_preview_state()
        self._stop = threading.Event()
        self._session_id = uuid.uuid4().hex
        self.set_playback_time(0.0, reset=True)
        self._playback_clock_started_at = time.monotonic()
        self._last_playback_jump_warning_at = 0.0
        self._start_embedding_worker()
        self._thread = threading.Thread(target=self._run, name="WindowDiarizer", daemon=True)
        self._thread.start()
        self.bus.emit("status", {"message": "Diarization worker started; synchronized playback can begin."})
        if self._preview_transcriber is not None:
            self._preview_thread = threading.Thread(target=self._run_realtime_preview, name="RealtimePreview", daemon=True)
            self._preview_thread.start()
        self._live_probe_thread = threading.Thread(target=self._run_live_speaker_probe, name="LiveSpeakerProbe", daemon=True)
        self._live_probe_thread.start()

    def _prepare_model_dependencies(self, include_asr_probe: bool, force_runtime_warmup: bool = False) -> None:
        self.bus.emit("status", {"message": "Loading transcription model."})
        self._load_model()
        self.bus.emit("status", {"message": "Loading sentence splitter."})
        self._warm_sentence_splitter()
        self.bus.emit("status", {"message": "Loading speaker embedding model."})
        self._warm_embedding(force=force_runtime_warmup)
        if self.args.vad_sentence_splitting and self.args.vad_backend == "silero":
            self.bus.emit("status", {"message": "Loading Silero ONNX VAD."})
            self._load_silero_vad_model()
        if include_asr_probe:
            self._warm_asr_transcription(force=force_runtime_warmup)

    def _should_refresh_start_runtime_warmup(self) -> bool:
        threshold = max(0.0, float(getattr(self.args, "start_warmup_stale_seconds", 10.0)))
        if threshold <= 0.0:
            self.bus.emit("status", {"message": "Refreshing runtime warmup before playback."})
            return True
        warm_times = [
            value
            for value in (self._embedding_warmed_at, self._asr_probe_warmed_at)
            if value is not None
        ]
        if len(warm_times) < 2:
            self.bus.emit("status", {"message": "Runtime warmup incomplete; refreshing before playback."})
            return True
        age = time.monotonic() - min(warm_times)
        if age >= threshold:
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Previous runtime warmup is {age:.1f}s old; "
                        "refreshing ASR and embeddings before playback."
                    )
                },
            )
            return True
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Previous runtime warmup is {age:.1f}s old; "
                    "skipping refresh before playback."
                )
            },
        )
        return False

    def _new_memory(self) -> SpeakerMemory:
        return SpeakerMemory(
            same_speaker_similarity=self.args.same_speaker_similarity,
            similarity_temperature=self.args.similarity_temperature,
            speaker_softmax_temperature=self.args.speaker_softmax_temperature,
            new_speaker_threshold=self.args.new_speaker_threshold,
            duplicate_profile_similarity=self.args.duplicate_profile_similarity,
            unknown_short_threshold=self.args.unknown_short_threshold,
            min_first_speaker_seconds=self.args.min_first_speaker_seconds,
            min_new_speaker_seconds=self.args.min_new_speaker_seconds,
            late_new_speaker_min_seconds=self.args.late_new_speaker_min_seconds,
            max_speakers=self.args.max_speakers,
            min_margin=self.args.min_margin,
            margin_temperature=self.args.margin_temperature,
            update_unknown_max=self.args.update_unknown_max,
            new_speaker_confirmation_count=self.args.new_speaker_confirmation_count,
            new_speaker_confirmation_similarity=self.args.new_speaker_confirmation_similarity,
            max_pending_new_speakers=self.args.max_pending_new_speakers,
        )

    def _rehydrate_seed_profiles(self) -> None:
        with self._speaker_lock:
            seed_profiles = [dict(item) for item in self._seed_profiles]
        seed_metadata = {
            f"S{index}": dict(item.get("metadata") or {})
            for index, item in enumerate(seed_profiles, 1)
            if isinstance(item.get("metadata"), dict)
        }
        if seed_profiles:
            self.memory.replace_profiles(seed_profiles)
            with self._speaker_lock:
                self._speaker_metadata = seed_metadata
        else:
            self.memory.replace_profiles([])
        self._sync_metadata_with_memory()

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

    def _speaker_state(self) -> dict[str, Any]:
        profiles = self.memory.export_profiles()
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
            speakers.append({
                "id": label,
                "name": str(metadata.get("name") or ""),
                "display_name": str(metadata.get("name") or "") or f"Speaker {profile['index']}",
                "source": str(metadata.get("source") or "detected"),
                "locked": bool(metadata.get("locked") or profile.get("locked")),
                "sentence_count": int(profile.get("sentence_count") or 0),
                "speech_seconds": round(float(profile.get("speech_seconds") or 0.0), 4),
                "reference_audio": str(metadata.get("reference_audio") or ""),
            })
        return {
            "group_name": group_name,
            "groups": list_speaker_groups(self.speaker_library_dir),
            "speakers": speakers,
            "embedding_provider": self.args.embedding_provider,
        }

    def emit_speaker_state(self) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        state = self._speaker_state()
        self.bus.emit("speakers", state)
        return state

    def speaker_state(self) -> dict[str, Any]:
        self._sync_metadata_with_memory()
        return self._speaker_state()

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

    def save_speaker_group(self, name: str) -> dict[str, Any]:
        self._sync_metadata_with_memory()
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

        manifest = {
            "version": 1,
            "name": group_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_provider": self.args.embedding_provider,
            "embedding_device": self.args.embedding_device,
            "speakers": saved_profiles,
        }
        (group_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._speaker_lock:
            self._speaker_group_name = group_name
            self._seed_profiles = [
                {
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
                "centroid": item["centroid"],
                "sentence_count": max(1, int(item.get("sentence_count") or 1)),
                "speech_seconds": float(item.get("speech_seconds") or 0.0),
                "locked": bool(item.get("locked")),
                "metadata": metadata_by_label[label],
            })

        self.memory = self._new_memory()
        with self._speaker_lock:
            self._speaker_group_name = group_name
            self._speaker_metadata = metadata_by_label
            self._seed_profiles = [dict(item) for item in seed_profiles]
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._unknown_sentences = []
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

        with self._speaker_lock:
            self._speaker_group_name = group_name
        return {
            "version": 1,
            "format": "whospeaks-speaker-group",
            "name": group_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_provider": self.args.embedding_provider,
            "embedding_device": self.args.embedding_device,
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
                "centroid": self._centroid_from_payload(item),
                "sentence_count": max(1, int(item.get("sentence_count") or 1)),
                "speech_seconds": float(item.get("speech_seconds") or 0.0),
                "locked": bool(item.get("locked")),
                "metadata": metadata_by_label[label],
            })
        if not seed_profiles:
            raise ValueError("Speaker group file has no usable speaker profiles.")

        self.memory = self._new_memory()
        with self._speaker_lock:
            self._speaker_group_name = group_name
            self._speaker_metadata = metadata_by_label
            self._seed_profiles = [dict(item) for item in seed_profiles]
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._unknown_sentences = []
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
        with self._speaker_lock:
            self._speaker_metadata[label] = {
                "name": clean_name,
                "source": "reference",
                "locked": True,
                "reference_audio": str(reference_path),
            }
            self._seed_profiles.append({
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
        preset = apply_new_speaker_sensitivity(self.args, normalized)
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

    def _apply_new_speaker_preset_to_memory(self, preset: dict[str, Any]) -> None:
        for key in NEW_SPEAKER_SENSITIVITY_FIELDS:
            if hasattr(self.memory, key):
                setattr(self.memory, key, preset[key])

    def _load_realtime_preview(self) -> None:
        self._preview_transcriber = None
        engine = str(self.args.realtime_preview_engine or "off").strip().lower().replace("-", "_")
        if engine in {"", "off", "none", "false"}:
            self.bus.emit("status", {"message": "Realtime preview disabled."})
            return
        started = time.monotonic()
        try:
            if engine == "mock":
                self._preview_transcriber = MockRealtimePreviewTranscriber()
                self.bus.emit("status", {"message": "Mock realtime preview ready."})
                return
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Loading realtime preview engine {self.args.realtime_preview_engine} "
                        f"on {self.args.realtime_preview_provider} before playback."
                    )
                },
            )
            if self.args.realtime_preview_python is not None and self.args.realtime_preview_python.is_file():
                self._preview_transcriber = KrokoSubprocessPreviewTranscriber(self.args)
            else:
                self._preview_transcriber = KrokoRealtimePreviewTranscriber(self.args)
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Realtime preview ready in {time.monotonic() - started:.2f}s "
                        f"({self.args.realtime_preview_engine})."
                    )
                },
            )
        except Exception as exc:
            self._preview_transcriber = None
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Realtime preview disabled: {type(exc).__name__}: {exc}"
                    )
                },
            )

    def _reset_realtime_preview_state(self) -> None:
        with self._preview_lock:
            self._preview_left = 0.0
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _pause_realtime_preview(self) -> None:
        with self._preview_lock:
            self._preview_paused = True
            self._preview_generation += 1
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _resume_realtime_preview(self, left: float) -> None:
        with self._preview_lock:
            self._preview_left = max(0.0, min(float(left), self.duration))
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit("realtime_clear", {"generation": generation})

    def _advance_realtime_preview_after_commit(self, committed_left: float) -> None:
        if self._preview_transcriber is None:
            return
        committed = max(0.0, min(float(committed_left), self.duration))
        overlap = max(0.0, float(self.args.realtime_preview_reset_overlap_seconds))
        reset_left = max(0.0, committed - overlap)
        skipped = max(0.0, reset_left - committed)
        preroll = max(0.0, committed - reset_left)
        with self._preview_lock:
            self._preview_left = reset_left
            self._preview_generation += 1
            self._preview_paused = False
            generation = self._preview_generation
        self.bus.emit(
            "realtime_clear",
            {
                "generation": generation,
                "committed_audio_time": round(committed, 4),
                "preview_reset_left": round(reset_left, 4),
                "skipped_audio_seconds": round(skipped, 4),
                "preview_preroll_seconds": round(preroll, 4),
            },
        )

    def _preview_snapshot(self) -> tuple[float, float, int, bool]:
        with self._preview_lock:
            return (
                self._preview_left,
                self.playback_time(),
                self._preview_generation,
                self._preview_paused,
            )

    def _preview_generation_is_current(self, generation: int, left: float) -> bool:
        with self._preview_lock:
            return (
                generation == self._preview_generation
                and not self._preview_paused
                and abs(left - self._preview_left) < 0.001
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
        self._preview_thread = None
        if self._live_probe_thread is not None and self._live_probe_thread.is_alive():
            self._live_probe_thread.join(timeout=2.0)
        self._live_probe_thread = None
        if self._preview_transcriber is not None:
            self._preview_transcriber.close()
        self._preview_transcriber = None
        self._playback_clock_started_at = None
        self._drain_embedding_jobs()
        self._stop_embedding_worker()

    def set_media(self, media: MediaFiles) -> None:
        self.stop()
        self.media = media
        with self._audio_lock:
            self._streaming_audio = False
            self._stream_audio_chunks = []
            self._stream_audio_samples = 0
            self.audio, self.sample_rate = load_audio_file(media.audio_file)
            self.duration = len(self.audio) / float(self.sample_rate)
        self.memory = self._new_memory()
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._unknown_sentences = []
        self.set_playback_time(0.0, reset=True)
        self._reset_realtime_preview_state()

    def set_browser_stream(self, url: str) -> MediaFiles:
        self.stop()
        video_id = resolve_browser_stream_id(url)
        media = MediaFiles(url, video_id, self.media.audio_file, self.media.video_file)
        self.media = media
        with self._audio_lock:
            self._streaming_audio = True
            self.sample_rate = 16000
            self.audio = np.zeros(0, dtype=np.float32)
            self._stream_audio_chunks = []
            self._stream_audio_samples = 0
            self.duration = 0.0
        self.memory = self._new_memory()
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._unknown_sentences = []
        self.set_playback_time(0.0, reset=True)
        self._reset_realtime_preview_state()
        return media

    def append_stream_audio(self, audio: np.ndarray, sample_rate: int) -> float:
        if not self._streaming_audio:
            raise RuntimeError("Browser audio stream is not active.")
        if int(sample_rate) != int(self.sample_rate):
            raise RuntimeError(f"Browser audio sample rate changed from {self.sample_rate} to {sample_rate}.")
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        if chunk.size <= 0:
            return self.duration
        chunk = np.nan_to_num(chunk, copy=False)
        chunk = np.clip(chunk, -1.0, 1.0)
        with self._audio_lock:
            self._stream_audio_chunks.append(chunk.astype(np.float32, copy=True))
            self._stream_audio_samples += int(chunk.size)
            self.duration = self._stream_audio_samples / float(self.sample_rate)
            duration = self.duration
        self.set_playback_time(duration)
        return duration

    def shutdown(self) -> None:
        self.stop()
        self.embedding.shutdown()

    def _start_embedding_worker(self) -> None:
        self._stop_embedding_worker()
        self._embedding_jobs = queue.Queue()
        self._embedding_thread = threading.Thread(
            target=self._run_embedding_jobs,
            name="WindowSpeakerEmbedding",
            daemon=True,
        )
        self._embedding_thread.start()

    def _drain_embedding_jobs(self, timeout_seconds: float = 10.0) -> bool:
        jobs = self._embedding_jobs
        if jobs is None:
            return True
        if getattr(jobs, "unfinished_tasks", 0) > 0:
            self.bus.emit("status", {"message": "Draining queued speaker embeddings."})
        drained = self._wait_for_embedding_jobs(jobs, timeout_seconds)
        if not drained:
            self.bus.emit("status", {"message": "Timed out draining queued speaker embeddings; cancelling pending jobs."})
            self._cancel_pending_embedding_jobs(jobs)
        return drained

    def _stop_embedding_worker(self) -> None:
        jobs = self._embedding_jobs
        thread = self._embedding_thread
        if jobs is not None and thread is not None and thread.is_alive():
            jobs.put(None)
            self._wait_for_embedding_jobs(jobs, 5.0)
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.bus.emit("status", {"message": "Speaker embedding worker did not stop before timeout."})
        self._embedding_jobs = None
        self._embedding_thread = None

    def _run_embedding_jobs(self) -> None:
        jobs = self._embedding_jobs
        if jobs is None:
            return
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                self._process_sentence_embedding(job)
            finally:
                jobs.task_done()

    def set_playback_time(self, seconds: float, reset: bool = False) -> None:
        seconds = max(0.0, min(float(seconds), self.duration))
        if not reset:
            seconds = self._clamp_playback_time_to_wall_clock(seconds)
        with self._playback_lock:
            self._playback_time = seconds if reset else max(self._playback_time, seconds)

    def playback_time(self) -> float:
        with self._playback_lock:
            return self._playback_time

    def _clamp_playback_time_to_wall_clock(self, seconds: float) -> float:
        if self._streaming_audio or self._playback_clock_started_at is None:
            return seconds
        max_allowed = min(self.duration, max(0.0, time.monotonic() - self._playback_clock_started_at + 3.0))
        if seconds <= max_allowed + 0.25:
            return seconds
        now = time.monotonic()
        if now >= self._last_playback_jump_warning_at + 5.0:
            self._last_playback_jump_warning_at = now
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Ignored early playback jump to {seconds:.2f}s; "
                        f"limiting live window to {max_allowed:.2f}s."
                    )
                },
            )
        return max_allowed

    def _audio_window_copy(self, left: float, right: float) -> tuple[np.ndarray, int]:
        with self._audio_lock:
            sample_rate = int(self.sample_rate)
            start_sample = max(0, int(left * sample_rate))
            end_sample = min(self._audio_sample_count_locked(), int(right * sample_rate))
            if end_sample <= start_sample:
                return np.zeros(0, dtype=np.float32), sample_rate
            return self._audio_slice_locked(start_sample, end_sample), sample_rate

    def _audio_sample_count_locked(self) -> int:
        return self._stream_audio_samples if self._streaming_audio else len(self.audio)

    def _audio_slice_locked(self, start_sample: int, end_sample: int) -> np.ndarray:
        if end_sample <= start_sample:
            return np.zeros(0, dtype=np.float32)
        if not self._streaming_audio:
            return np.asarray(self.audio[start_sample:end_sample], dtype=np.float32).copy()

        pieces: list[np.ndarray] = []
        offset = 0
        for chunk in self._stream_audio_chunks:
            next_offset = offset + len(chunk)
            if next_offset <= start_sample:
                offset = next_offset
                continue
            if offset >= end_sample:
                break
            left = max(0, start_sample - offset)
            right = min(len(chunk), end_sample - offset)
            if right > left:
                pieces.append(chunk[left:right])
            offset = next_offset

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        if len(pieces) == 1:
            return np.asarray(pieces[0], dtype=np.float32).copy()
        return np.concatenate(pieces).astype(np.float32, copy=False)

    @staticmethod
    def _wait_for_embedding_jobs(jobs: "queue.Queue[EmbeddingSentenceJob | None]", timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while getattr(jobs, "unfinished_tasks", 0) > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    @staticmethod
    def _cancel_pending_embedding_jobs(jobs: "queue.Queue[EmbeddingSentenceJob | None]") -> None:
        while True:
            try:
                jobs.get_nowait()
            except queue.Empty:
                return
            else:
                jobs.task_done()

    def _load_silero_vad_model(self) -> Any | None:
        if self._vad_model is not None or self._vad_model_error is not None:
            return self._vad_model
        with self._vad_model_lock:
            if self._vad_model is not None or self._vad_model_error is not None:
                return self._vad_model
            try:
                realtime_root = Path(getattr(self.args, "realtime_preview_realtimestt_root", DEFAULT_REALTIMESTT_ROOT))
                if realtime_root.exists() and str(realtime_root) not in sys.path:
                    sys.path.insert(0, str(realtime_root))
                from RealtimeSTT.core.silero_vad import create_silero_vad_model

                model_path = getattr(self.args, "vad_silero_onnx_model_path", None)
                if model_path is not None:
                    model_path = Path(model_path)
                    if not model_path.exists():
                        raise FileNotFoundError(f"Silero ONNX model not found: {model_path}")
                backend = str(getattr(self.args, "vad_silero_backend", "auto") or "auto")
                if backend == "auto" and model_path is not None:
                    backend = default_silero_vad_backend(model_path)
                self._vad_model = create_silero_vad_model(
                    backend=backend,
                    onnx_model_path=str(model_path) if model_path is not None else None,
                    onnx_threads=max(1, int(getattr(self.args, "vad_silero_onnx_threads", 2))),
                    sample_rate=SILERO_VAD_SAMPLE_RATE,
                    chunk_samples=SILERO_VAD_CHUNK_SAMPLES,
                )
                self._vad_model_backend = str(getattr(self._vad_model, "backend", backend))
                loaded_path = getattr(self._vad_model, "model_path", model_path)
                loaded_name = Path(loaded_path).name if loaded_path is not None else "auto"
                self.bus.emit(
                    "status",
                    {"message": f"Silero ONNX VAD ready ({self._vad_model_backend}, {loaded_name})."},
                )
            except Exception as exc:
                self._vad_model_error = str(exc)
                self.bus.emit(
                    "status",
                    {"message": f"Silero ONNX VAD unavailable; falling back to RMS VAD: {exc}"},
                )
        return self._vad_model

    def _resample_for_silero_vad(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if int(sample_rate) == SILERO_VAD_SAMPLE_RATE:
            return np.asarray(audio, dtype=np.float32)
        if audio.size <= 0 or sample_rate <= 0:
            return np.zeros(0, dtype=np.float32)
        duration = audio.size / float(sample_rate)
        target_size = max(1, int(round(duration * SILERO_VAD_SAMPLE_RATE)))
        source_times = np.arange(audio.size, dtype=np.float64) / float(sample_rate)
        target_times = np.arange(target_size, dtype=np.float64) / float(SILERO_VAD_SAMPLE_RATE)
        return np.interp(target_times, source_times, audio).astype(np.float32)

    def _smooth_vad_flags(self, flags: list[bool], frame_seconds: float) -> None:
        max_gap_frames = max(0, int(round(float(self.args.vad_merge_gap_seconds) / frame_seconds)))
        if max_gap_frames <= 0 or not any(flags):
            return
        index = 0
        while index < len(flags):
            if flags[index]:
                index += 1
                continue
            gap_start = index
            while index < len(flags) and not flags[index]:
                index += 1
            gap_end = index
            if gap_start > 0 and gap_end < len(flags) and gap_end - gap_start <= max_gap_frames:
                for fill_index in range(gap_start, gap_end):
                    flags[fill_index] = True

    def _vad_state_from_flags(
        self,
        left: float,
        right: float,
        audio_size: int,
        sample_rate: int,
        frame_samples: int,
        frame_seconds: float,
        flags: list[bool],
        starts: list[int],
        backend: str,
    ) -> VadWindowState:
        if not flags:
            return VadWindowState(False, False, backend=backend)

        self._smooth_vad_flags(flags, frame_seconds)
        speech_indexes = [index for index, is_speech in enumerate(flags) if is_speech]
        if not speech_indexes:
            return VadWindowState(False, False, backend=backend)

        first = speech_indexes[0]
        last = speech_indexes[-1]
        speech_start = left + (starts[first] / sample_rate)
        speech_end = left + (min(audio_size, starts[last] + frame_samples) / sample_rate)
        speech_seconds = sum(frame_seconds for is_speech in flags if is_speech)
        if speech_seconds < max(0.0, float(self.args.vad_min_speech_seconds)):
            return VadWindowState(False, False, backend=backend)

        trailing_silence = max(0.0, right - speech_end)
        should_flush = trailing_silence >= max(0.0, float(self.args.vad_silence_seconds))
        return VadWindowState(
            has_speech=True,
            should_flush=should_flush,
            speech_start=round(float(speech_start), 4),
            speech_end=round(float(speech_end), 4),
            speech_seconds=round(float(speech_seconds), 4),
            trailing_silence_seconds=round(float(trailing_silence), 4),
            backend=backend,
        )

    def _rms_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        frame_seconds = max(0.01, float(self.args.vad_frame_seconds))
        frame_samples = max(1, int(sample_rate * frame_seconds))
        threshold = max(0.0, float(self.args.vad_speech_rms_threshold))
        flags: list[bool] = []
        starts: list[int] = []
        for start in range(0, audio.size, frame_samples):
            end = min(audio.size, start + frame_samples)
            if end - start < max(1, frame_samples // 2):
                break
            frame = audio[start:end]
            rms_value = float(np.sqrt(np.mean(frame * frame)))
            flags.append(rms_value >= threshold)
            starts.append(start)
        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=audio.size,
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend="rms",
        )

    def _audio_has_rms_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        if audio.size <= 0 or sample_rate <= 0:
            return False
        frame_seconds = max(0.01, float(getattr(self.args, "vad_frame_seconds", 0.03)))
        frame_samples = max(1, int(sample_rate * frame_seconds))
        threshold = max(0.0, float(getattr(self.args, "vad_speech_rms_threshold", 0.003)))
        min_speech_seconds = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "live_speaker_probe_min_speech_seconds",
                    getattr(self.args, "vad_min_speech_seconds", 0.15),
                )
            ),
        )
        speech_seconds = 0.0
        for start in range(0, audio.size, frame_samples):
            end = min(audio.size, start + frame_samples)
            if end - start < max(1, frame_samples // 2):
                break
            frame = audio[start:end]
            rms_value = float(np.sqrt(np.mean(frame * frame)))
            if rms_value >= threshold:
                speech_seconds += (end - start) / float(sample_rate)
                if speech_seconds >= min_speech_seconds:
                    return True
        return False

    def _silero_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        model = self._load_silero_vad_model()
        if model is None:
            return self._rms_vad_window_state(left, right, audio, sample_rate)

        vad_audio = self._resample_for_silero_vad(audio, sample_rate)
        if vad_audio.size <= 0:
            return VadWindowState(False, False, backend="silero")

        frame_samples = SILERO_VAD_CHUNK_SAMPLES
        frame_seconds = frame_samples / float(SILERO_VAD_SAMPLE_RATE)
        threshold = max(0.0, min(1.0, float(self.args.vad_silero_speech_threshold)))
        flags: list[bool] = []
        starts: list[int] = []
        reset_states = getattr(model, "reset_states", None)
        if callable(reset_states):
            reset_states()
        try:
            for start in range(0, vad_audio.size, frame_samples):
                end = min(vad_audio.size, start + frame_samples)
                if end - start < max(1, frame_samples // 2):
                    break
                chunk = vad_audio[start:end]
                if chunk.size < frame_samples:
                    padded = np.zeros(frame_samples, dtype=np.float32)
                    padded[:chunk.size] = chunk
                    chunk = padded
                probability = float(model(chunk.astype(np.float32, copy=False), SILERO_VAD_SAMPLE_RATE))
                flags.append(probability >= threshold)
                starts.append(start)
        except Exception as exc:
            self._vad_model_error = str(exc)
            self._vad_model = None
            self.bus.emit(
                "status",
                {"message": f"Silero ONNX VAD call failed; falling back to RMS VAD: {exc}"},
            )
            return self._rms_vad_window_state(left, right, audio, sample_rate)

        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=vad_audio.size,
            sample_rate=SILERO_VAD_SAMPLE_RATE,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend=self._vad_model_backend or "silero",
        )

    def _vad_window_state(self, left: float, right: float) -> VadWindowState:
        if not getattr(self.args, "vad_sentence_splitting", True):
            return VadWindowState(False, False)
        if right <= left:
            return VadWindowState(False, False)

        audio, sample_rate = self._audio_window_copy(left, right)
        if audio.size <= 0 or sample_rate <= 0:
            return VadWindowState(False, False)

        if getattr(self.args, "vad_backend", "silero") == "rms":
            return self._rms_vad_window_state(left, right, audio, sample_rate)
        return self._silero_vad_window_state(left, right, audio, sample_rate)

    def _warm_sentence_splitter(self) -> None:
        if self._sentence_splitter_warmed:
            self.bus.emit("status", {"message": "stream2sentence tokenizer already warm."})
            return
        self.bus.emit("status", {"message": "Initializing stream2sentence tokenizer before playback."})
        started = time.monotonic()
        init_tokenizer("nltk+rule-based", language="en")
        list(generate_sentences(
            list("A warmup sentence vs. a false split. Another sentence."),
            tokenizer="nltk+rule-based",
            language="en",
            auto_context=True,
            minimum_sentence_length=1,
            minimum_first_fragment_length=1,
            context_size=12,
            context_size_look_overhead=64,
        ))
        self._sentence_splitter_warmed = True
        self.bus.emit("status", {"message": f"stream2sentence ready in {time.monotonic() - started:.2f}s."})

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

    def _remove_unknown_sentence(self, index: int) -> bool:
        with self._unknown_lock:
            old_count = len(self._unknown_sentences)
            self._unknown_sentences = [
                item for item in self._unknown_sentences
                if item.index != index
            ]
            return len(self._unknown_sentences) != old_count

    def _revisit_unknown_sentences(self) -> None:
        with self._unknown_lock:
            candidates = list(self._unknown_sentences)

        for candidate in candidates:
            decision = self.memory.score_existing(
                candidate.embedding,
                candidate.duration_seconds,
                min_similarity=self.args.retro_reassign_min_similarity,
                min_margin=self.args.retro_reassign_min_margin,
            )
            if not decision.assigned_speaker:
                continue
            if not self._remove_unknown_sentence(candidate.index):
                continue
            self.bus.emit("sentence", {
                **candidate.base_payload,
                "pending": False,
                "revision": True,
                "retro_reassigned": True,
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
            })
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

    def _load_model(self) -> None:
        with self._model_lock:
            if self._model is not None:
                self.bus.emit("status", {"message": "ASR backend already loaded."})
                return
            asr_backend = str(self.args.asr_backend or "local").strip().lower().replace("-", "_")
            if asr_backend == "remote":
                client = RemoteWindowAsrClient(self.args.remote_asr_url, self.args.remote_asr_timeout_seconds)
                self.bus.emit("status", {"message": f"Checking remote ASR server at {client.base_url}."})
                health = client.health()
                health_status = health.get("status") or health.get("model") or health.get("raw") or "ok"
                self._model = client
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Remote faster-whisper large-v2 ASR ready at {client.base_url} "
                            f"(health={health_status})."
                        )
                    },
                )
                return
            self.bus.emit("status", {"message": "Importing faster-whisper."})
            from faster_whisper import WhisperModel

            self.bus.emit("status", {"message": f"Loading faster-whisper {self.args.model} on {self.args.device} before playback."})
            self._model = WhisperModel(
                self.args.model,
                device=self.args.device,
                compute_type=self.args.compute_type,
                download_root=str(self.args.download_root) if self.args.download_root else None,
            )
            self.bus.emit("status", {"message": "faster-whisper ready; starting synchronized playback."})

    def _transcribe_audio_words(self, model: Any, audio: np.ndarray, sample_rate: int) -> tuple[list[TimedWord], int]:
        if isinstance(model, RemoteWindowAsrClient):
            return model.transcribe_window(audio, sample_rate, self.args.beam_size)

        segments, _info = model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=self.args.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        words: list[TimedWord] = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            for word in getattr(segment, "words", None) or []:
                text = str(word_attr(word, "word", "") or "")
                if not text.strip():
                    continue
                words.append(
                    TimedWord(
                        text,
                        float(word_attr(word, "start", 0.0)),
                        float(word_attr(word, "end", 0.0)),
                    )
                )
        words.sort(key=lambda item: (item.start, item.end))
        return words, segment_count

    def _warm_asr_transcription(self, force: bool = False) -> None:
        if self._asr_probe_warmed and not force:
            self.bus.emit("status", {"message": "ASR warmup transcription already complete."})
            return
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise RuntimeError("ASR backend did not load.")

        sample_rate = int(self.sample_rate)
        probe_samples = max(1, int(sample_rate * 0.75))
        probe, sample_rate = self._audio_window_copy(0.0, probe_samples / float(sample_rate))
        if probe.size < probe_samples:
            padded = np.zeros(probe_samples, dtype=np.float32)
            padded[:probe.size] = probe
            probe = padded

        started = time.monotonic()
        words, segment_count = self._transcribe_audio_words(self._model, probe, sample_rate)
        self._asr_probe_warmed = True
        self._asr_probe_warmed_at = time.monotonic()
        self.bus.emit(
            "status",
            {
                "message": (
                    f"ASR warmup transcription complete in {self._asr_probe_warmed_at - started:.2f}s "
                    f"(segments={segment_count}, words={len(words)})."
                )
            },
        )

    def _warm_embedding(self, force: bool = False) -> None:
        if self._embedding_warmed and not force:
            self.bus.emit("status", {"message": "Speaker embedding model already warm."})
            return
        warmup_label = "Refreshing" if self._embedding_warmed else "Warming"
        self.bus.emit("status", {"message": f"{warmup_label} speaker embedding model before playback."})
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".warm.wav", prefix="window-diarize-", dir=str(self.args.output_dir), delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            write_wav(wav_path, np.zeros(int(self.sample_rate * 0.6), dtype=np.float32), self.sample_rate)
            started = time.monotonic()
            self.embedding.embed_wav(wav_path)
            self._embedding_warmed = True
            self._embedding_warmed_at = time.monotonic()
            self.bus.emit("status", {"message": f"Speaker embedding model ready in {self._embedding_warmed_at - started:.2f}s."})
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

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

    def _score_realtime_preview_speaker(self, audio: np.ndarray, duration_seconds: float) -> dict[str, Any]:
        if duration_seconds < max(0.0, float(self.args.realtime_preview_diarize_min_audio_seconds)):
            return self._realtime_unknown_speaker_payload()
        if self.memory.profile_count() <= 0:
            return self._realtime_unknown_speaker_payload()

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        chunk = pad_audio(trim_silence(audio, self.sample_rate), self.args.min_embed_seconds, self.sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".live.wav", prefix="window-diarize-", dir=str(self.args.output_dir), delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            write_wav(wav_path, chunk, self.sample_rate)
            embedding = self.embedding.embed_wav(wav_path)
            decision = self.memory.score_existing(
                embedding,
                duration_seconds,
                min_similarity=self.args.realtime_preview_diarize_min_similarity,
                min_margin=self.args.realtime_preview_diarize_min_margin,
            )
        except Exception as exc:
            self.bus.emit("status", {"message": f"Realtime preview speaker scoring error: {type(exc).__name__}: {exc}"})
            return self._realtime_unknown_speaker_payload()
        finally:
            if not self.args.keep_segment_audio:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

        assigned_speaker = decision.assigned_speaker
        if assigned_speaker:
            try:
                speaker_probability = float(decision.probabilities.get(f"speaker{int(assigned_speaker[1:])}", 0.0))
            except Exception:
                speaker_probability = 0.0
            if speaker_probability < self.args.realtime_preview_diarize_min_known_probability:
                assigned_speaker = None
        self._ensure_speaker_metadata(assigned_speaker)

        return {
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": False,
            "probabilities": decision.probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            "assignment_source": "realtime_preview_embedding",
        }

    def _run_live_speaker_probe(self) -> None:
        if not bool(getattr(self.args, "live_speaker_probe", True)):
            return
        interval_seconds = max(0.05, float(getattr(self.args, "live_speaker_probe_interval_seconds", 0.5)))
        window_seconds = max(0.05, float(getattr(self.args, "live_speaker_probe_window_seconds", 2.0)))
        min_advance = max(0.0, float(getattr(self.args, "live_speaker_probe_min_advance_seconds", interval_seconds)))
        hold_seconds = max(0.0, float(getattr(self.args, "live_speaker_probe_hold_seconds", 2.0)))
        last_probe_right = -1.0
        while not self._stop.wait(interval_seconds):
            if self.memory.profile_count() <= 0:
                continue
            right = self.playback_time()
            if right <= 0.0:
                continue
            if last_probe_right >= 0.0 and right < last_probe_right + min_advance:
                continue
            left = max(0.0, right - window_seconds)
            audio, sample_rate = self._audio_window_copy(left, right)
            duration_seconds = audio.size / float(sample_rate) if sample_rate > 0 else 0.0
            if duration_seconds < max(0.0, float(self.args.realtime_preview_diarize_min_audio_seconds)):
                continue
            last_probe_right = right
            if not self._audio_has_rms_speech(audio, sample_rate):
                continue
            speaker_payload = self._score_realtime_preview_speaker(audio, duration_seconds)
            if self._stop.is_set():
                break
            assigned_speaker = speaker_payload.get("assigned_speaker")
            if not assigned_speaker:
                continue
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
                    "assignment_source": "last_2s_embedding_probe",
                },
            )

    def _run_realtime_preview(self) -> None:
        transcriber = self._preview_transcriber
        if transcriber is None:
            return
        interval_seconds = max(0.05, float(self.args.realtime_preview_interval_seconds))
        min_audio_seconds = max(0.05, float(self.args.realtime_preview_min_audio_seconds))
        min_advance = max(0.0, float(self.args.realtime_preview_min_advance_seconds))
        feed_chunk_seconds = max(0.02, float(self.args.realtime_preview_feed_chunk_seconds))
        diarize_min_advance = max(0.0, float(self.args.realtime_preview_diarize_min_advance_seconds))
        last_generation = -1
        last_right = 0.0
        last_decode_right = -1.0
        last_diarized_right = -1.0
        last_speaker_payload = self._realtime_unknown_speaker_payload()
        next_at = 0.0
        self.bus.emit(
            "status",
            {
                "message": (
                    f"Realtime preview started ({interval_seconds:.2f}s interval, "
                    f"min audio {min_audio_seconds:.2f}s, "
                    f"feed chunk {feed_chunk_seconds:.2f}s)."
                )
            },
        )
        while not self._stop.is_set():
            left, right, generation, paused = self._preview_snapshot()
            if paused:
                time.sleep(0.05)
                continue
            if generation != last_generation:
                last_generation = generation
                last_right = left
                last_decode_right = -1.0
                last_diarized_right = -1.0
                last_speaker_payload = self._realtime_unknown_speaker_payload()
                try:
                    transcriber.reset_preview()
                except Exception as exc:
                    self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    time.sleep(interval_seconds)
                    continue
            if right - left < min_audio_seconds:
                time.sleep(0.05)
                continue
            if right < last_right + feed_chunk_seconds:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(0.05, next_at - now))
                continue
            try:
                text = ""
                while right >= last_right + feed_chunk_seconds:
                    feed_right = last_right + feed_chunk_seconds
                    audio, sample_rate = self._audio_window_copy(last_right, feed_right)
                    last_right = feed_right
                    if audio.size <= 0:
                        continue
                    text = " ".join(transcriber.accept_preview_audio(audio, sample_rate).split())
            except Exception as exc:
                self.bus.emit("status", {"message": f"Realtime preview error: {type(exc).__name__}: {exc}"})
                time.sleep(interval_seconds)
                continue
            preview_right = last_right
            should_emit = last_decode_right < 0.0 or preview_right >= last_decode_right + min_advance
            if not should_emit:
                continue
            last_decode_right = preview_right
            next_at = time.monotonic() + interval_seconds
            if not text or not re.search(r"[A-Za-z0-9]", text):
                continue
            if not self._preview_generation_is_current(generation, left):
                continue
            duration_seconds = max(0.0, preview_right - left)
            if duration_seconds >= self.args.realtime_preview_diarize_min_audio_seconds and (
                last_diarized_right < 0.0 or preview_right >= last_diarized_right + diarize_min_advance
            ):
                audio, _sample_rate = self._audio_window_copy(left, preview_right)
                last_speaker_payload = self._score_realtime_preview_speaker(audio, duration_seconds)
                last_diarized_right = preview_right
                if not self._preview_generation_is_current(generation, left):
                    continue
            self.bus.emit("realtime", {
                "index": f"rt-{generation}",
                "realtime": True,
                "realtime_generation": generation,
                "text": text,
                "start": round(left, 4),
                "end": round(preview_right, 4),
                "audio_length_seconds": round(float(max(0.0, preview_right - left)), 4),
                "pending": False,
                **last_speaker_payload,
            })

    def _run(self) -> None:
        model = self._model
        if model is None:
            self.bus.emit("status", {"message": "No ASR backend loaded."})
            self.bus.emit("done", {"message": "Window diarization stopped."})
            return
        try:
            left = 0.0
            index = 0
            last_transcribed_right = -1.0
            last_vad_flush_right = -1.0
            interval_seconds = max(0.0, float(self.args.interval_seconds))
            min_playback_advance = max(0.0, float(self.args.min_playback_advance_seconds))
            final_flush_epsilon = max(0.0, float(self.args.final_flush_epsilon_seconds))
            next_tick = time.monotonic() + interval_seconds if interval_seconds > 0.0 else 0.0
            mode = "continuous" if interval_seconds <= 0.0 else f"{interval_seconds:.2f}s interval"
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Growing-window transcription started ({mode}, "
                        f"min playback advance {min_playback_advance:.2f}s)."
                    )
                },
            )
            while not self._stop.is_set():
                now = time.monotonic()
                duration = self.duration
                if not self._streaming_audio and left >= duration:
                    break
                right = self.playback_time()
                media_final_flush = (not self._streaming_audio) and right >= duration - final_flush_epsilon
                if media_final_flush:
                    right = duration

                vad_state = self._vad_window_state(left, right)
                vad_flush = vad_state.should_flush and not media_final_flush
                if (
                    getattr(self.args, "vad_sentence_splitting", True)
                    and not media_final_flush
                    and not vad_state.has_speech
                ):
                    time.sleep(0.05)
                    continue

                transcript_final_flush = media_final_flush or vad_flush
                if vad_flush and right <= last_vad_flush_right + min_playback_advance:
                    time.sleep(0.05)
                    continue
                if right - left < max(1.0, self.args.min_window_seconds) and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if right <= last_transcribed_right + min_playback_advance and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if interval_seconds > 0.0 and now < next_tick and not transcript_final_flush:
                    time.sleep(0.1)
                    continue
                if interval_seconds > 0.0:
                    next_tick = now + interval_seconds
                last_transcribed_right = right
                final_note = " final" if media_final_flush else (" vad-final" if vad_flush else "")
                transcribe_right = right
                vad_next_left: float | None = None
                if vad_flush:
                    vad_label = "RMS VAD" if vad_state.backend == "rms" else "Silero VAD"
                    speech_end = float(vad_state.speech_end or right)
                    transcribe_right = max(
                        left,
                        min(
                            right,
                            speech_end + max(0.0, float(self.args.vad_final_window_post_silence_seconds)),
                        ),
                    )
                    vad_next_left = max(
                        left,
                        min(
                            duration,
                            speech_end + max(0.0, float(self.args.vad_next_window_start_silence_seconds)),
                        ),
                    )
                    self.bus.emit(
                        "status",
                        {
                            "message": (
                                f"{vad_label} silence split at {vad_state.speech_end:.2f}s "
                                f"after {vad_state.trailing_silence_seconds:.2f}s silence; "
                                f"final window right={transcribe_right:.2f}s next left={vad_next_left:.2f}s."
                            )
                        },
                    )
                self.bus.emit("status", {"message": f"Transcribing{final_note} window left={left:.2f}s right={transcribe_right:.2f}s."})
                transcribe_started = time.monotonic()
                transcript = self._transcribe_window(model, left, transcribe_right, final_flush=transcript_final_flush)
                transcribe_seconds = time.monotonic() - transcribe_started
                if vad_flush:
                    last_vad_flush_right = right
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Transcribed {transcribe_right - left:.2f}s window in {transcribe_seconds:.2f}s; "
                            f"segments={transcript.segment_count} words={transcript.word_count} "
                            f"accepted={len(transcript.sentences)}."
                        )
                    },
                )
                for sentence in transcript.sentences:
                    self._emit_sentence(index, sentence, left, transcribe_right)
                    index += 1
                    left = max(left, sentence.next_left)
                if vad_next_left is not None:
                    left = max(left, vad_next_left)
                if transcript.sentences or vad_next_left is not None:
                    self._advance_realtime_preview_after_commit(left)
                self.bus.emit("status", {"message": f"Window left={left:.2f}s right={right:.2f}s sentences={len(transcript.sentences)}."})
                if media_final_flush:
                    break
        except Exception as exc:
            self.bus.emit("status", {"message": f"Window diarization error: {type(exc).__name__}: {exc}"})
        finally:
            self._pause_realtime_preview()
            self._drain_embedding_jobs()
            self._revisit_unknown_sentences()
            self.bus.emit("done", {"message": "Window diarization stopped."})

    def _transcribe_window(self, model: Any, left: float, right: float, final_flush: bool = False) -> WindowTranscript:
        window, sample_rate = self._audio_window_copy(left, right)
        relative_words, segment_count = self._transcribe_audio_words(model, window, sample_rate)
        words = [
            TimedWord(word.text, left + float(word.start), left + float(word.end))
            for word in relative_words
        ]
        words.sort(key=lambda item: (item.start, item.end))
        parts = split_words_with_stream2sentence(
            words,
            left=left,
            right=right,
            unstable_tail_seconds=self.args.unstable_tail_seconds,
            final_flush=final_flush,
            boundary_pre_padding_seconds=self.args.sentence_boundary_pre_padding_seconds,
            boundary_post_padding_seconds=self.args.sentence_boundary_post_padding_seconds,
            boundary_gap_ratio=self.args.sentence_boundary_gap_ratio,
        )
        return WindowTranscript(parts, len(words), segment_count)

    def _emit_sentence(self, index: int, sentence: SentencePart, window_left: float, window_right: float) -> None:
        base_payload = {
            "index": index,
            "text": sentence.text,
            "start": round(sentence.start, 4),
            "end": round(sentence.end, 4),
            "spoken_word_seconds": round(float(sentence.spoken_word_seconds), 4),
            "audio_length_seconds": round(float(max(0.0, sentence.end - sentence.start)), 4),
            "speech_audio_ratio": round(float(sentence.speech_audio_ratio), 4),
            "new_speaker_anchor_words": len(text_content_words(sentence.text)),
            "window_left": round(window_left, 4),
            "window_right": round(window_right, 4),
            "next_left": round(sentence.next_left, 4),
            "words": sentence.words,
            "first_word_start": round_optional(sentence.first_word_start),
            "last_word_end": round_optional(sentence.last_word_end),
            "next_word_start": round_optional(sentence.next_word_start),
            "gap_to_next_word_seconds": round_optional(sentence.gap_to_next_word_seconds),
            "boundary_strategy": sentence.boundary_strategy,
            "sentence_boundary_pre_padding_seconds": round(float(sentence.sentence_boundary_pre_padding_seconds), 4),
            "sentence_boundary_post_padding_seconds": round(float(sentence.sentence_boundary_post_padding_seconds), 4),
            "sentence_boundary_gap_ratio": round(float(sentence.sentence_boundary_gap_ratio), 4),
            "unknown_permanent": False,
        }
        self.bus.emit("sentence", {
            **base_payload,
            "pending": True,
            "assigned_speaker": None,
            **self._speaker_info_for_payload(None),
            "created_speaker": False,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "top_similarity": None,
            "margin": None,
        })
        if sentence.speech_audio_ratio < self.args.min_speech_audio_ratio:
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Marking sentence {index} permanently unknown: "
                        f"speech/audio ratio {sentence.speech_audio_ratio:.2f} "
                        f"below {self.args.min_speech_audio_ratio:.2f}."
                    )
                },
            )
            self.bus.emit("sentence", {
                **base_payload,
                "pending": False,
                "unknown_permanent": True,
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "quality": None,
                "assignment_source": "unknown_permanent",
            })
            return
        if not is_embedding_candidate_text(sentence.text):
            self.bus.emit("status", {"message": f"Skipping non-speech/vocable sentence {index}: {sentence.text[:72]}"})
            self.bus.emit("sentence", {
                **base_payload,
                "pending": False,
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
            })
            return
        duration_seconds = max(0.0, sentence.end - sentence.start)
        audio, sample_rate = self._audio_window_copy(sentence.start, sentence.end)
        job = EmbeddingSentenceJob(
            index=index,
            base_payload=base_payload,
            text=sentence.text,
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
        jobs = self._embedding_jobs
        if jobs is None:
            self._process_sentence_embedding(job)
            return
        jobs.put(job)
        self.bus.emit("status", {"message": f"Queued speaker embedding for sentence {index}: {sentence.text[:72]}"})

    def _process_sentence_embedding(self, job: EmbeddingSentenceJob) -> None:
        base_payload = job.base_payload
        index = job.index
        chunk = pad_audio(
            trim_silence(job.audio, job.sample_rate),
            self.args.min_embed_seconds,
            job.sample_rate,
        )
        duration_seconds = job.duration_seconds
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".sentence.wav", prefix="window-diarize-", dir=str(self.args.output_dir), delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            self.bus.emit("status", {"message": f"Embedding sentence {index}: {job.text[:72]}"})
            embed_started = time.monotonic()
            write_wav(wav_path, chunk, job.sample_rate)
            embedding = self.embedding.embed_wav(wav_path)
            decision = self.memory.classify(
                embedding,
                duration_seconds,
                allow_new_speaker=(
                    len(text_content_words(job.text)) >= self.args.min_new_speaker_words
                ),
            )
            self.bus.emit("status", {
                "message": (
                    f"Embedded sentence {index} in {time.monotonic() - embed_started:.2f}s; "
                    f"speaker={decision.assigned_speaker or 'UNKNOWN'} "
                    f"new={int(bool(decision.created_speaker))} "
                    f"unk={decision.unknown_probability} "
                    f"top={decision.top_similarity} "
                    f"margin={decision.margin}."
                )
            })
        except Exception as exc:
            self.bus.emit("status", {"message": f"Embedding failed for sentence {index}: {type(exc).__name__}: {exc}"})
            self.bus.emit("sentence", {
                **base_payload,
                "pending": False,
                "error": f"{type(exc).__name__}: {exc}",
                "assigned_speaker": None,
                **self._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
            })
            return
        finally:
            if not self.args.keep_segment_audio:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
        self._ensure_speaker_metadata(decision.assigned_speaker)
        self.bus.emit("sentence", {
            **base_payload,
            "pending": False,
            "assigned_speaker": decision.assigned_speaker,
            **self._speaker_info_for_payload(decision.assigned_speaker),
            "created_speaker": decision.created_speaker,
            "probabilities": decision.probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            "assignment_source": decision.assignment_source,
        })
        if decision.assigned_speaker is None:
            self._remember_unknown_sentence(index, base_payload, embedding, duration_seconds)
        elif decision.created_speaker:
            self.emit_speaker_state()
            self._revisit_unknown_sentences()


