"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
from collections import deque
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
)
from window.window_config import (
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
from window.window_media import resolve_browser_stream_id
from window.window_preview import (
    KrokoRealtimePreviewTranscriber,
    KrokoSubprocessPreviewTranscriber,
    MockRealtimePreviewTranscriber,
    RealtimePreviewTranscriber,
)
from window.window_remote_asr import RemoteWindowAsrClient
from window.window_text import (
    is_embedding_candidate_text,
    round_optional,
    split_words_with_stream2sentence,
    text_content_words,
    word_attr,
)
from window.window_speaker_refinement import (
    SpeakerRefinementConfig,
    find_speaker_prototype_revisions,
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
        self.embedding = self._new_embedding_client(args)
        self.memory = self._new_memory()
        self.live_embedding = self._new_live_embedding_client(args)
        self._live_embedding_separate = self.live_embedding is not self.embedding
        self.live_memory = self._new_memory() if self._live_embedding_separate else self.memory
        self._live_probability_history: deque[tuple[float, dict[str, float]]] = deque(
            maxlen=max(1, int(getattr(args, "live_speaker_ema_count", 3)))
        )
        self.speaker_library_dir = Path(getattr(args, "speaker_library_dir", DEFAULT_SPEAKER_LIBRARY_DIR))
        self._speaker_lock = threading.Lock()
        self._speaker_group_name = ""
        self._speaker_metadata: dict[str, dict[str, Any]] = {}
        self._seed_profiles: list[dict[str, Any]] = []
        self._seed_live_profiles: list[dict[str, Any]] = []
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
        self._recent_unknown_pair_candidates: deque[PendingUnknownSentence] = deque(maxlen=24)
        self._sentence_refinement_lock = threading.Lock()
        self._sentence_refinement_records: dict[int, dict[str, Any]] = {}
        self._sentence_refinement_run_lock = threading.Lock()
        self._speaker_last_media_end: dict[str, float] = {}
        self._embedding_jobs: "queue.Queue[EmbeddingSentenceJob | None] | None" = None
        self._embedding_thread: threading.Thread | None = None
        self._live_memory_update_jobs: "queue.Queue[LiveSpeakerMemoryUpdateJob | None] | None" = None
        self._live_memory_update_thread: threading.Thread | None = None
        self._live_memory_update_lock = threading.Lock()
        self._preview_thread: threading.Thread | None = None
        self._live_probe_thread: threading.Thread | None = None
        self._preview_transcriber: RealtimePreviewTranscriber | None = None
        self._preview_lock = threading.Lock()
        self._preview_left = 0.0
        self._preview_generation = 0
        self._preview_paused = False
        self._live_speaker_embedding_throttle_lock = threading.Lock()
        self._live_speaker_embedding_next_at = 0.0
        self._live_speaker_embedding_latency_ewma: float | None = None
        self._live_speaker_embedding_last_status_at = 0.0
        self._live_speaker_verify_lock = threading.Lock()
        self._live_speaker_verify_next_at = 0.0
        self._live_speaker_verify_last_status_at = 0.0
        self._vad_model: Any = None
        self._vad_model_backend = ""
        self._vad_model_error: str | None = None
        self._vad_model_lock = threading.Lock()
        self._sentence_splitter_warmed = False
        self._embedding_warmed = False
        self._asr_probe_warmed = False
        self._embedding_warmed_at: float | None = None
        self._asr_probe_warmed_at: float | None = None
        self._speaker_generation = 0

    def _new_embedding_client(self, args: argparse.Namespace, provider: str | None = None) -> Any:
        embeddings_backend = str(getattr(args, "embeddings_backend", "local") or "local").strip().lower().replace("-", "_")
        embedding_provider = str(provider or args.embedding_provider)
        if embeddings_backend == "remote":
            return RemoteEmbeddingClient(
                base_url=args.remote_embeddings_url,
                provider=embedding_provider,
                device=getattr(args, "remote_embeddings_device", "auto"),
                timeout_seconds=getattr(args, "remote_embeddings_timeout_seconds", 600.0),
            )
        return EmbeddingSubprocessClient(
            args.embedding_python,
            embedding_provider,
            args.embedding_device,
            response_timeout_seconds=getattr(args, "embedding_helper_response_timeout_seconds", 600.0),
        )

    def _new_live_embedding_client(self, args: argparse.Namespace) -> Any:
        provider = str(getattr(args, "live_speaker_embedding_provider", "") or "").strip()
        if not provider or provider == str(args.embedding_provider):
            return self.embedding
        return self._new_embedding_client(args, provider=provider)

    def _current_live_embedding_provider(self) -> str:
        provider = str(getattr(self.args, "live_speaker_embedding_provider", "") or "").strip()
        return provider or str(self.args.embedding_provider)

    def _serialized_live_speaker_profiles(
        self,
        main_profiles: list[dict[str, Any]],
        *,
        portable: bool,
    ) -> list[dict[str, Any]]:
        if not getattr(self, "_live_embedding_separate", False):
            return []
        live_memory = getattr(self, "live_memory", None)
        export_profiles = getattr(live_memory, "export_profiles", None)
        if not callable(export_profiles):
            return []
        main_labels = {str(profile.get("label") or "") for profile in main_profiles}
        live_profiles = [
            dict(profile)
            for profile in export_profiles()
            if str(profile.get("label") or "") in main_labels
        ]
        serialized: list[dict[str, Any]] = []
        for profile in live_profiles:
            item = {
                "label": str(profile.get("label") or ""),
                "sentence_count": int(profile.get("sentence_count") or 1),
                "speech_seconds": float(profile.get("speech_seconds") or 0.0),
                "created_at": float(profile.get("created_at") or time.time()),
                "last_seen_at": float(profile.get("last_seen_at") or time.time()),
                "locked": bool(profile.get("locked")),
            }
            if portable:
                item.update(self._centroid_payload(profile["centroid"]))
            else:
                item["centroid"] = np.asarray(profile["centroid"], dtype=np.float32).astype(float).tolist()
            serialized.append(item)
        return serialized

    def _live_seed_profiles_from_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        if not getattr(self, "_live_embedding_separate", False):
            return []
        live_provider = str(manifest.get("live_embedding_provider") or "")
        if not live_provider or live_provider != self._current_live_embedding_provider():
            return []
        raw_profiles = manifest.get("live_speakers")
        if not isinstance(raw_profiles, list):
            return []
        seed_profiles: list[dict[str, Any]] = []
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            seed_profiles.append({
                "label": str(item.get("label") or ""),
                "centroid": self._centroid_from_payload(item),
                "sentence_count": max(1, int(item.get("sentence_count") or 1)),
                "speech_seconds": float(item.get("speech_seconds") or 0.0),
                "locked": bool(item.get("locked")),
            })
        return seed_profiles

    def prepare_before_browser_release(self) -> None:
        self.bus.emit(
            "status",
            {"message": "Preparing ASR, embeddings, and VAD before publishing the browser URL."},
        )
        self._prepare_model_dependencies(include_asr_probe=True)
        self.bus.emit("status", {"message": "Startup model warmup complete; browser GUI can be opened."})

    def start(self) -> dict[str, Any]:
        self.bus.emit("status", {"message": "Start requested; preparing models before playback."})
        self.stop()
        refresh_runtime_warmup = self._should_refresh_start_runtime_warmup()
        self._prepare_model_dependencies(
            include_asr_probe=refresh_runtime_warmup,
            force_runtime_warmup=refresh_runtime_warmup,
        )
        self.bus.emit("status", {"message": "Loading realtime preview engine."})
        self._load_realtime_preview()
        speaker_state = self._reset_runtime_session_state()
        self._stop = threading.Event()
        self._session_id = uuid.uuid4().hex
        self.set_playback_time(0.0, reset=True)
        self._playback_clock_started_at = time.monotonic()
        self._last_playback_jump_warning_at = 0.0
        self._start_embedding_worker()
        self._start_live_memory_update_worker()
        self._thread = threading.Thread(target=self._run, name="WindowDiarizer", daemon=True)
        self._thread.start()
        self.bus.emit("status", {"message": "Diarization worker started; synchronized playback can begin."})
        if self._preview_transcriber is not None:
            self._preview_thread = threading.Thread(target=self._run_realtime_preview, name="RealtimePreview", daemon=True)
            self._preview_thread.start()
        self._live_probe_thread = threading.Thread(target=self._run_live_speaker_probe, name="LiveSpeakerProbe", daemon=True)
        self._live_probe_thread.start()
        return speaker_state

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
            known_speaker_min_similarity=self.args.known_speaker_min_similarity,
            known_speaker_gray_zone_min_unknown_probability=(
                self.args.known_speaker_gray_zone_min_unknown_probability
            ),
            profile_update_min_similarity=self.args.profile_update_min_similarity,
            profile_update_min_margin=self.args.profile_update_min_margin,
            low_similarity_unknown_floor_similarity=self.args.low_similarity_unknown_floor_similarity,
            low_similarity_unknown_floor_probability=self.args.low_similarity_unknown_floor_probability,
            gray_zone_promote_max_similarity=getattr(
                self.args,
                "gray_zone_promote_max_similarity",
                1.0,
            ),
        )

    def _reset_live_speaker_memory(self) -> None:
        if not hasattr(self, "_live_embedding_separate"):
            self._live_embedding_separate = False
        self.live_memory = self._new_memory() if self._live_embedding_separate else self.memory
        history = getattr(self, "_live_probability_history", None)
        if history is None:
            history = deque(maxlen=max(1, int(getattr(self.args, "live_speaker_ema_count", 3))))
            self._live_probability_history = history
        history.clear()
        self._live_speaker_verify_next_at = 0.0

    def _live_memory_update_lock_obj(self) -> threading.Lock:
        lock = getattr(self, "_live_memory_update_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._live_memory_update_lock = lock
        return lock

    def _cancel_pending_live_memory_update_jobs(self) -> None:
        live_jobs = getattr(self, "_live_memory_update_jobs", None)
        if live_jobs is not None:
            self._cancel_pending_embedding_jobs(live_jobs)

    def _recent_unknown_pair_queue(self) -> deque[PendingUnknownSentence]:
        queue = getattr(self, "_recent_unknown_pair_candidates", None)
        if queue is None:
            queue = deque(maxlen=24)
            self._recent_unknown_pair_candidates = queue
        return queue

    def _clear_unknown_sentence_state_locked(self) -> None:
        self._unknown_sentences = []
        self._recent_unknown_pair_queue().clear()

    def _rehydrate_seed_profiles(self) -> None:
        with self._speaker_lock:
            seed_profiles = [dict(item) for item in self._seed_profiles]
            seed_live_profiles = [dict(item) for item in getattr(self, "_seed_live_profiles", [])]
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
        self._reset_live_speaker_memory()
        if self._live_embedding_separate and seed_live_profiles:
            for index, profile in enumerate(seed_live_profiles, 1):
                label = str(profile.get("label") or f"S{index}")
                self.live_memory.upsert_profile(
                    label,
                    profile["centroid"],
                    duration_seconds=float(profile.get("speech_seconds") or 0.0),
                    sentence_count=max(1, int(profile.get("sentence_count") or 1)),
                    locked=bool(profile.get("locked")),
                )

    def _reset_runtime_session_state(self, *, emit: bool = True) -> dict[str, Any]:
        self.memory = self._new_memory()
        self._rehydrate_seed_profiles()
        with self._unknown_lock:
            self._clear_unknown_sentence_state_locked()
        self._clear_sentence_refinement_records()
        self._speaker_last_media_end = {}
        self._reset_realtime_preview_state()
        if emit:
            return self.emit_speaker_state()
        return self.speaker_state()

    def is_running(self) -> bool:
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._thread, self._preview_thread, self._live_probe_thread)
        )

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
            "live_speakers": saved_live_profiles,
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

    def speaker_refinement_settings(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.args, "speaker_refinement", True)),
            "allow_reassignment": bool(getattr(self.args, "allow_speaker_reassignment", False)),
        }

    def set_allow_speaker_reassignment(self, enabled: Any) -> dict[str, Any]:
        value = bool(enabled)
        setattr(self.args, "allow_speaker_reassignment", value)
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
        self._drain_live_memory_update_jobs()
        self._stop_live_memory_update_worker()

    def set_media(self, media: MediaFiles) -> None:
        self.stop()
        self.media = media
        with self._audio_lock:
            self._streaming_audio = False
            self._stream_audio_chunks = []
            self._stream_audio_samples = 0
            self.audio, self.sample_rate = load_audio_file(media.audio_file)
            self.duration = len(self.audio) / float(self.sample_rate)
        self._reset_runtime_session_state()
        self.set_playback_time(0.0, reset=True)

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
        self._reset_runtime_session_state()
        self.set_playback_time(0.0, reset=True)
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
        if self.live_embedding is not self.embedding:
            self.live_embedding.shutdown()

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
        jobs = getattr(self, "_embedding_jobs", None)
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

    def _start_live_memory_update_worker(self) -> None:
        self._stop_live_memory_update_worker()
        if not self._live_embedding_separate:
            self._live_memory_update_jobs = None
            self._live_memory_update_thread = None
            return
        try:
            queue_size = int(getattr(self.args, "live_speaker_memory_update_queue_size", 64))
        except (TypeError, ValueError):
            queue_size = 64
        self._live_memory_update_jobs = queue.Queue(maxsize=max(1, queue_size))
        self._live_memory_update_thread = threading.Thread(
            target=self._run_live_memory_update_jobs,
            name="LiveSpeakerMemoryUpdate",
            daemon=True,
        )
        self._live_memory_update_thread.start()

    def _drain_live_memory_update_jobs(self, timeout_seconds: float = 10.0) -> bool:
        jobs = getattr(self, "_live_memory_update_jobs", None)
        if jobs is None:
            return True
        if getattr(jobs, "unfinished_tasks", 0) > 0:
            self.bus.emit("status", {"message": "Draining queued live speaker profile updates."})
        drained = self._wait_for_embedding_jobs(jobs, timeout_seconds)
        if not drained:
            self.bus.emit(
                "status",
                {"message": "Timed out draining queued live speaker profile updates; cancelling pending updates."},
            )
            self._cancel_pending_embedding_jobs(jobs)
        return drained

    def _stop_live_memory_update_worker(self) -> None:
        jobs = self._live_memory_update_jobs
        thread = self._live_memory_update_thread
        if jobs is not None and thread is not None and thread.is_alive():
            try:
                jobs.put(None, timeout=1.0)
            except queue.Full:
                self._cancel_pending_embedding_jobs(jobs)
                jobs.put(None)
            self._wait_for_embedding_jobs(jobs, 5.0)
            thread.join(timeout=5.0)
            if thread.is_alive():
                self.bus.emit("status", {"message": "Live speaker profile update worker did not stop before timeout."})
        self._live_memory_update_jobs = None
        self._live_memory_update_thread = None

    def _run_live_memory_update_jobs(self) -> None:
        jobs = self._live_memory_update_jobs
        if jobs is None:
            return
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                self._process_live_speaker_memory_update(job)
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
    def _wait_for_embedding_jobs(jobs: "queue.Queue[Any]", timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while getattr(jobs, "unfinished_tasks", 0) > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    @staticmethod
    def _cancel_pending_embedding_jobs(jobs: "queue.Queue[Any]") -> None:
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

    def _audio_has_live_probe_speech(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        backend = str(getattr(self.args, "live_speaker_probe_speech_backend", "rms") or "rms").lower()
        if backend != "vad":
            return self._audio_has_rms_speech(audio, sample_rate)
        if audio.size <= 0 or sample_rate <= 0 or right <= left:
            return False
        if getattr(self.args, "vad_backend", "silero") == "rms":
            return self._rms_vad_window_state(left, right, audio, sample_rate).has_speech
        return self._silero_vad_window_state(left, right, audio, sample_rate).has_speech

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
            self._recent_unknown_pair_queue().append(pending)

    def _remove_unknown_sentence(self, index: int) -> bool:
        with self._unknown_lock:
            old_count = len(self._unknown_sentences)
            self._unknown_sentences = [
                item for item in self._unknown_sentences
                if item.index != index
            ]
            return len(self._unknown_sentences) != old_count

    def _clear_sentence_refinement_records(self) -> None:
        with self._sentence_refinement_lock:
            self._sentence_refinement_records = {}

    def _speaker_refinement_config(self) -> SpeakerRefinementConfig:
        return SpeakerRefinementConfig(
            max_per_profile=int(getattr(self.args, "speaker_refinement_max_per_profile", 32)),
            prototype_min_duration=float(getattr(self.args, "speaker_refinement_min_duration", 0.15)),
            prototype_max_unknown=float(getattr(self.args, "speaker_refinement_max_unknown", 1.0)),
            top_k=int(getattr(self.args, "speaker_refinement_top_k", 12)),
            centroid_blend=float(getattr(self.args, "speaker_refinement_centroid_blend", 0.555)),
            unknown_min_similarity=float(getattr(self.args, "speaker_refinement_unknown_min_similarity", 0.20)),
            unknown_min_margin=float(getattr(self.args, "speaker_refinement_unknown_min_margin", 0.0)),
            unknown_min_later_rows=int(getattr(self.args, "speaker_refinement_unknown_min_later_rows", 5)),
            known_max_duration=float(getattr(self.args, "speaker_refinement_known_max_duration", 8.0)),
            known_min_similarity=float(getattr(self.args, "speaker_refinement_known_min_similarity", -0.039)),
            known_min_delta=float(getattr(self.args, "speaker_refinement_known_min_delta", 0.108)),
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
        with self._sentence_refinement_lock:
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

    def _unknown_pair_new_speaker_decision(
        self,
        embedding: np.ndarray,
        duration_seconds: float,
        base_payload: dict[str, Any],
        existing_decision: SpeakerDecision,
        *,
        allow_new_speaker: bool,
    ) -> SpeakerDecision | None:
        if not allow_new_speaker or not bool(getattr(self.args, "unknown_pair_new_speaker", False)):
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
        return SpeakerDecision(
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
        )

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
        with self._sentence_refinement_lock:
            record = self._sentence_refinement_records.get(int(revision.index))
            if record is None:
                return False
            if record.get("assigned_speaker") != revision.previous_speaker:
                return False
            record["assigned_speaker"] = revision.assigned_speaker
            record["created_speaker"] = False
            record["probabilities"] = self._prototype_probabilities(
                revision.assigned_speaker,
                revision.prototype_scores,
            )
            record["similarities"] = dict(revision.prototype_scores)
            record["unknown_probability"] = 0.0
            record["top_similarity"] = revision.prototype_score
            record["margin"] = revision.prototype_margin
            record["assignment_source"] = revision.assignment_source
            base_payload = dict(record["base_payload"])
            probabilities = dict(record["probabilities"])
            similarities = dict(record["similarities"])
            quality = record.get("quality")

        self._ensure_speaker_metadata(revision.assigned_speaker)
        if revision.previous_speaker is None:
            self._remove_unknown_sentence(int(revision.index))
        self.bus.emit("sentence", {
            **base_payload,
            "pending": False,
            "revision": True,
            "prototype_reassigned": True,
            "prototype_reassigned_from": revision.previous_speaker or "UNKNOWN",
            "revision_from": revision.previous_speaker or "UNKNOWN",
            "revision_to": revision.assigned_speaker,
            "assigned_speaker": revision.assigned_speaker,
            **self._speaker_info_for_payload(revision.assigned_speaker),
            "created_speaker": False,
            "probabilities": probabilities,
            "similarities": similarities,
            "unknown_probability": 0.0,
            "top_similarity": revision.prototype_score,
            "margin": revision.prototype_margin,
            "quality": quality,
            "assignment_source": revision.assignment_source,
            "prototype_score": revision.prototype_score,
            "prototype_margin": revision.prototype_margin,
            "prototype_delta": revision.prototype_delta,
        })
        return True

    def _refine_speaker_assignments(self) -> None:
        if not bool(getattr(self.args, "speaker_refinement", True)):
            return
        if not self._sentence_refinement_run_lock.acquire(blocking=False):
            return
        try:
            with self._sentence_refinement_lock:
                records = [
                    dict(record)
                    for _, record in sorted(self._sentence_refinement_records.items())
                ]
            if len(records) < 2:
                return
            allow_known = bool(getattr(self.args, "allow_speaker_reassignment", False))
            revisions = find_speaker_prototype_revisions(
                records,
                self._speaker_refinement_config(),
                allow_known_reassignment=allow_known,
            )
            applied = 0
            known_revisions = 0
            for revision in revisions:
                if not allow_known and revision.previous_speaker is not None:
                    continue
                if self._apply_prototype_revision(revision):
                    applied += 1
                    if revision.previous_speaker is not None:
                        known_revisions += 1
            if applied:
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Prototype speaker refinement applied {applied} revision(s)"
                            f"{' including ' + str(known_revisions) + ' known-speaker change(s)' if known_revisions else ''}."
                        )
                    },
                )
        finally:
            self._sentence_refinement_run_lock.release()

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
            payload = {
                **candidate.base_payload,
                "pending": False,
                "revision": True,
                "retro_reassigned": True,
                "retro_reassigned_from": "UNKNOWN",
                "revision_from": "UNKNOWN",
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
            self.bus.emit("sentence", payload)
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

    def _embed_audio_chunk_with_client(self, client: Any, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        embed_audio = getattr(client, "embed_audio", None)
        if callable(embed_audio) and not self.args.keep_segment_audio:
            return embed_audio(audio, sample_rate)

        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="window-diarize-", dir=str(self.args.output_dir), delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            write_wav(wav_path, audio, sample_rate)
            return client.embed_wav(wav_path)
        finally:
            if not self.args.keep_segment_audio:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _embed_audio_chunk(self, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        return self._embed_audio_chunk_with_client(self.embedding, audio, sample_rate, suffix)

    def _embed_live_audio_chunk(self, audio: np.ndarray, sample_rate: int, suffix: str) -> np.ndarray:
        return self._embed_audio_chunk_with_client(self.live_embedding, audio, sample_rate, suffix)

    def _warm_embedding(self, force: bool = False) -> None:
        if self._embedding_warmed and not force:
            self.bus.emit("status", {"message": "Speaker embedding model already warm."})
            return
        warmup_label = "Refreshing" if self._embedding_warmed else "Warming"
        self.bus.emit("status", {"message": f"{warmup_label} speaker embedding model before playback."})
        if isinstance(self.embedding, RemoteEmbeddingClient):
            self.bus.emit("status", {"message": f"Checking remote embeddings server at {self.embedding.base_url}."})
            health = self.embedding.health()
            health_status = health.get("status") or health.get("service") or health.get("raw") or "ok"
            self.bus.emit("status", {"message": f"Remote embeddings server ready at {self.embedding.base_url} (health={health_status})."})
        started = time.monotonic()
        self._embed_audio_chunk(np.zeros(int(self.sample_rate * 0.6), dtype=np.float32), self.sample_rate, ".warm.wav")
        if self._live_embedding_separate:
            self._embed_live_audio_chunk(
                np.zeros(int(self.sample_rate * 0.6), dtype=np.float32),
                self.sample_rate,
                ".live-warm.wav",
            )
        self._embedding_warmed = True
        self._embedding_warmed_at = time.monotonic()
        self.bus.emit("status", {"message": f"Speaker embedding model ready in {self._embedding_warmed_at - started:.2f}s."})

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

    def _try_reserve_live_speaker_embedding(self) -> bool:
        now = time.monotonic()
        with self._live_speaker_embedding_lock():
            next_at = float(getattr(self, "_live_speaker_embedding_next_at", 0.0))
            if now < next_at:
                return False
            self._live_speaker_embedding_next_at = now + self._live_speaker_embedding_min_interval_seconds()
            return True

    def _record_live_speaker_embedding_latency(self, elapsed_seconds: float) -> None:
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
    ) -> dict[str, Any]:
        if min_audio_seconds is None:
            min_audio_seconds = float(self.args.realtime_preview_diarize_min_audio_seconds)
        if duration_seconds < max(0.0, float(min_audio_seconds)):
            return self._realtime_unknown_speaker_payload()
        memory = self.live_memory
        if memory.profile_count() <= 0:
            return self._realtime_unknown_speaker_payload()

        chunk = pad_audio(trim_silence(audio, self.sample_rate), self.args.min_embed_seconds, self.sample_rate)
        try:
            embed_started = time.monotonic()
            embedding = self._embed_live_audio_chunk(chunk, self.sample_rate, ".live.wav")
            self._record_live_speaker_embedding_latency(time.monotonic() - embed_started)
            decision = memory.score_existing(
                embedding,
                duration_seconds,
                min_similarity=self.args.realtime_preview_diarize_min_similarity,
                min_margin=self.args.realtime_preview_diarize_min_margin,
            )
        except Exception as exc:
            self.bus.emit("status", {"message": f"Realtime preview speaker scoring error: {type(exc).__name__}: {exc}"})
            return self._realtime_unknown_speaker_payload()

        raw_probabilities = dict(decision.probabilities)
        smoothed_probabilities = self._live_speaker_ema_probabilities(raw_probabilities)
        assigned_speaker = self._assign_live_speaker_from_probabilities(
            smoothed_probabilities,
            decision.assigned_speaker,
        )
        assist_payload: dict[str, Any] = {}
        if not assigned_speaker:
            assigned_speaker, assist_payload = self._maybe_assign_weak_profile_live_speaker(decision)
        self._ensure_speaker_metadata(assigned_speaker)

        return {
            "assigned_speaker": assigned_speaker,
            **self._speaker_info_for_payload(assigned_speaker),
            "created_speaker": False,
            "probabilities": smoothed_probabilities,
            "raw_probabilities": raw_probabilities,
            "similarities": decision.similarities,
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            **assist_payload,
            "assignment_source": (
                "live_weak_profile_assist"
                if assist_payload
                else (
                    "live_fast_embedding_ema"
                    if self._live_embedding_separate
                    else "realtime_preview_embedding_ema"
                )
            ),
        }

    def _update_live_speaker_memory(
        self,
        speaker_id: str | None,
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
        suffix: str = ".live-profile.wav",
        speaker_generation: int | None = None,
    ) -> None:
        if not getattr(self, "_live_embedding_separate", False) or not speaker_id:
            return
        job = LiveSpeakerMemoryUpdateJob(
            speaker_id=str(speaker_id),
            audio=np.asarray(audio, dtype=np.float32).copy(),
            sample_rate=int(sample_rate),
            duration_seconds=float(duration_seconds),
            suffix=suffix,
            speaker_generation=(
                int(getattr(self, "_speaker_generation", 0))
                if speaker_generation is None
                else int(speaker_generation)
            ),
        )
        jobs = getattr(self, "_live_memory_update_jobs", None)
        if jobs is None:
            self._process_live_speaker_memory_update(job)
            return
        try:
            jobs.put_nowait(job)
        except queue.Full:
            try:
                jobs.get_nowait()
            except queue.Empty:
                pass
            else:
                jobs.task_done()
                self.bus.emit(
                    "status",
                    {"message": "Dropped stale queued live speaker profile update because the queue is full."},
                )
            try:
                jobs.put_nowait(job)
            except queue.Full:
                self.bus.emit(
                    "status",
                    {"message": "Skipped live speaker profile update because the queue is still full."},
                )

    def _process_live_speaker_memory_update(self, job: LiveSpeakerMemoryUpdateJob) -> None:
        try:
            if job.speaker_generation != getattr(self, "_speaker_generation", 0):
                return
            embedding = self._embed_live_audio_chunk(job.audio, job.sample_rate, job.suffix)
            with self._live_memory_update_lock_obj():
                if job.speaker_generation != getattr(self, "_speaker_generation", 0):
                    return
                self.live_memory.upsert_profile(
                    job.speaker_id,
                    embedding,
                    duration_seconds=job.duration_seconds,
                    sentence_count=1,
                )
        except Exception as exc:
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Live speaker profile update failed "
                        f"for {job.speaker_id}: {type(exc).__name__}: {exc}"
                    )
                },
            )

    def _run_live_speaker_probe(self) -> None:
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
            if self._stop.wait(max(0.05, wait_seconds)):
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
            if self._stop.is_set():
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
                if self.memory.profile_count() > 0 and self._try_reserve_live_speaker_embedding():
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
                    if bool(getattr(self.args, "live_speaker_clear_on_vad_split", False)):
                        self.bus.emit(
                            "live_speaker_clear",
                            {
                                "live": False,
                                "fallback": True,
                                "start": round(float(speech_end), 4),
                                "end": round(float(right), 4),
                                "reason": "vad_silence_split",
                                "assignment_source": "main_vad_silence_split_clear",
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
                    if interval_seconds > 0.0 and not media_final_flush:
                        next_tick = time.monotonic() + interval_seconds
                self.bus.emit("status", {"message": f"Window left={left:.2f}s right={right:.2f}s sentences={len(transcript.sentences)}."})
                if media_final_flush:
                    break
        except Exception as exc:
            self.bus.emit("status", {"message": f"Window diarization error: {type(exc).__name__}: {exc}"})
        finally:
            self._pause_realtime_preview()
            self._drain_embedding_jobs()
            self._revisit_unknown_sentences()
            self._drain_live_memory_update_jobs()
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
            speaker_generation=self._speaker_generation,
        )
        jobs = self._embedding_jobs
        if jobs is None:
            self._process_sentence_embedding(job)
            return
        jobs.put(job)
        self.bus.emit("status", {"message": f"Queued speaker embedding for sentence {index}: {sentence.text[:72]}"})

    def _maybe_emit_sentence_live_speaker_hint(
        self,
        sentence_payload: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        if not bool(getattr(self.args, "live_speaker_sentence_hint", True)):
            return
        try:
            min_duration = max(
                0.0,
                float(getattr(self.args, "live_speaker_sentence_hint_min_duration_seconds", 0.0)),
            )
        except (TypeError, ValueError):
            min_duration = 0.0
        if duration_seconds < min_duration:
            return
        speaker_id = str(sentence_payload.get("assigned_speaker") or "")
        if not speaker_id or speaker_id == "UNKNOWN":
            return
        try:
            end = float(sentence_payload.get("end") or 0.0)
            playback_time = float(self.playback_time())
        except (TypeError, ValueError):
            return
        try:
            max_lag = max(0.0, float(getattr(self.args, "live_speaker_sentence_hint_max_lag_seconds", 1.25)))
        except (TypeError, ValueError):
            max_lag = 1.25
        created_speaker = bool(sentence_payload.get("created_speaker"))
        if created_speaker:
            try:
                max_lag = max(
                    max_lag,
                    max(0.0, float(getattr(
                        self.args,
                        "live_speaker_sentence_hint_new_speaker_max_lag_seconds",
                        1.25,
                    ))),
                )
            except (TypeError, ValueError):
                max_lag = max(max_lag, 1.25)
            try:
                max_top_similarity = float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_new_speaker_max_top_similarity",
                    1.0,
                ))
            except (TypeError, ValueError):
                max_top_similarity = 1.0
            try:
                top_similarity = float(sentence_payload.get("top_similarity"))
            except (TypeError, ValueError):
                top_similarity = 1.0
            if top_similarity > max_top_similarity:
                return
        lag_seconds = playback_time - end
        if lag_seconds > max_lag:
            return
        try:
            hold_seconds = max(
                0.0,
                float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_hold_seconds",
                    getattr(self.args, "live_speaker_probe_hold_seconds", 1.0),
                )),
            )
        except (TypeError, ValueError):
            hold_seconds = 1.0
        if created_speaker:
            try:
                new_speaker_hold = float(getattr(
                    self.args,
                    "live_speaker_sentence_hint_new_speaker_hold_seconds",
                    -1.0,
                ))
            except (TypeError, ValueError):
                new_speaker_hold = -1.0
            if new_speaker_hold >= 0.0:
                hold_seconds = max(hold_seconds, new_speaker_hold)
        if bool(getattr(self.args, "live_speaker_sentence_hint_hold_through_sentence", False)):
            hold_seconds = max(hold_seconds, max(0.0, end - playback_time) + hold_seconds)
        if hold_seconds <= 0.0:
            return
        self.bus.emit(
            "live_speaker",
            {
                **sentence_payload,
                "speaker_id": speaker_id,
                "live": True,
                "fallback": True,
                "sentence_hint": True,
                "only_if_no_live_speaker": not bool(
                    getattr(self.args, "live_speaker_sentence_hint_override", False)
                ),
                "start": sentence_payload.get("start"),
                "end": sentence_payload.get("end"),
                "audio_length_seconds": round(float(max(0.0, duration_seconds)), 4),
                "hold_seconds": round(float(hold_seconds), 4),
                "playback_time": round(float(playback_time), 4),
                "live_hint_lag_seconds": round(float(lag_seconds), 4),
                "assignment_source": "final_sentence_live_hint",
            },
        )

    def _process_sentence_embedding(self, job: EmbeddingSentenceJob) -> None:
        base_payload = job.base_payload
        index = job.index
        if job.speaker_generation != self._speaker_generation:
            self.bus.emit("status", {"message": f"Skipped stale speaker embedding for sentence {index}."})
            return
        chunk = pad_audio(
            trim_silence(job.audio, job.sample_rate),
            self.args.min_embed_seconds,
            job.sample_rate,
        )
        duration_seconds = job.duration_seconds
        try:
            self.bus.emit("status", {"message": f"Embedding sentence {index}: {job.text[:72]}"})
            embed_started = time.monotonic()
            embedding = self._embed_audio_chunk(chunk, job.sample_rate, ".sentence.wav")
            if job.speaker_generation != self._speaker_generation:
                self.bus.emit("status", {"message": f"Discarded stale speaker embedding for sentence {index}."})
                return
            allow_new_speaker = len(text_content_words(job.text)) >= self.args.min_new_speaker_words
            decision = self._section_gap_new_speaker_decision(
                embedding,
                duration_seconds,
                base_payload,
                allow_new_speaker=allow_new_speaker,
            )
            if decision is None:
                decision = self.memory.classify(
                    embedding,
                    duration_seconds,
                    allow_new_speaker=allow_new_speaker,
                )
                pair_decision = self._unknown_pair_new_speaker_decision(
                    embedding,
                    duration_seconds,
                    base_payload,
                    decision,
                    allow_new_speaker=allow_new_speaker,
                )
                if pair_decision is not None:
                    decision = pair_decision
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
        self._ensure_speaker_metadata(decision.assigned_speaker)
        sentence_payload = {
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
        }
        self._record_sentence_assignment(
            index,
            base_payload,
            embedding,
            duration_seconds,
            sentence_payload,
        )
        self.bus.emit("sentence", sentence_payload)
        self._maybe_emit_sentence_live_speaker_hint(sentence_payload, duration_seconds)
        self._update_live_speaker_memory(
            decision.assigned_speaker,
            chunk,
            job.sample_rate,
            duration_seconds,
            ".live-sentence.wav",
            speaker_generation=job.speaker_generation,
        )
        if decision.assigned_speaker is None:
            self._remember_unknown_sentence(index, base_payload, embedding, duration_seconds)
        elif decision.created_speaker:
            self.emit_speaker_state()
            self._revisit_unknown_sentences()
        self._refine_speaker_assignments()


