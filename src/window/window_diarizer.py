"""Main growing-window diarization controller."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
from collections import Counter, deque
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
    SILERO_VAD_CHUNK_SAMPLES,
    SILERO_VAD_SAMPLE_RATE,
    apply_new_speaker_sensitivity,
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
        self._session_started_at = ""
        self._session_source_title = ""
        self._next_session_id = ""
        self._playback_lock = threading.Lock()
        self._playback_time = 0.0
        self._playback_clock_started_at: float | None = None
        self._last_playback_jump_warning_at = 0.0
        self._unknown_lock = threading.Lock()
        self._unknown_sentences: list[PendingUnknownSentence] = []
        self._recent_unknown_pair_candidates: deque[PendingUnknownSentence] = deque(maxlen=24)
        self._sentence_refinement_lock = threading.Lock()
        self._sentence_refinement_records: dict[int, dict[str, Any]] = {}
        self._correction_history: list[dict[str, Any]] = []
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
        self._webrtc_vad_error: str | None = None
        self._sentence_splitter_warmed = False
        self._embedding_warmed = False
        self._asr_probe_warmed = False
        self._embedding_warmed_at: float | None = None
        self._asr_probe_warmed_at: float | None = None
        self._speaker_generation = 0
        self._speaker_label_generations: dict[str, int] = {}
        self._final_sentence_count = 0
        self._last_final_sentence_ended_strong = True

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
        if not bool(getattr(args, "live_speaker_assignment", True)):
            return self.embedding
        provider = str(getattr(args, "live_speaker_embedding_provider", "") or "").strip()
        if not provider or provider == str(args.embedding_provider):
            return self.embedding
        return self._new_embedding_client(args, provider=provider)

    def _current_live_embedding_provider(self) -> str:
        if not bool(getattr(self.args, "live_speaker_assignment", True)):
            return str(self.args.embedding_provider)
        provider = str(getattr(self.args, "live_speaker_embedding_provider", "") or "").strip()
        return provider or str(self.args.embedding_provider)

    def _live_speaker_assignment_enabled(self) -> bool:
        return bool(getattr(self.args, "live_speaker_assignment", True))

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
        self._session_id = self._next_session_id or uuid.uuid4().hex
        self._next_session_id = ""
        self._session_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
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
        if self._live_speaker_assignment_enabled() and bool(getattr(self.args, "live_speaker_probe", True)):
            self._live_probe_thread = threading.Thread(target=self._run_live_speaker_probe, name="LiveSpeakerProbe", daemon=True)
            self._live_probe_thread.start()
        return speaker_state

    def set_session_source_title(self, title: str) -> None:
        self._session_source_title = " ".join(str(title or "").strip().split())[:120]

    def set_next_session_id(self, session_id: str) -> None:
        normalized = str(session_id or "").strip()
        if normalized and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", normalized):
            raise ValueError("Invalid session id.")
        self._next_session_id = normalized

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
            first_speaker_immediate_min_seconds=getattr(
                self.args,
                "first_speaker_immediate_min_seconds",
                self.args.min_first_speaker_seconds,
            ),
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
            str(item.get("label") or f"S{index}"): dict(item.get("metadata") or {})
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
        self._final_sentence_count = 0
        self._last_final_sentence_ended_strong = True
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
        rows, embeddings = self._session_transcript_rows_and_embeddings()
        speaker_state = self.speaker_state()
        source = self._session_source_metadata()
        return {
            "id": str(self._session_id or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": float(source.get("duration_seconds") or self.duration),
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
        with self._audio_lock:
            if not self._streaming_audio or not self._stream_audio_chunks:
                return False
            audio = np.concatenate([chunk.astype(np.float32, copy=True) for chunk in self._stream_audio_chunks])
            sample_rate = int(self.sample_rate)
        write_wav(path, audio, sample_rate)
        return True

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
        with self._speaker_lock:
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
            "unknown_tentative": bool(getattr(self.args, "speaker_refinement_unknown_tentative", True)),
            "unknown_commit": bool(getattr(self.args, "speaker_refinement_unknown_commit", True)),
            "allow_reassignment": bool(getattr(self.args, "allow_speaker_reassignment", False)),
        }

    def set_speaker_refinement_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        if "speaker_refinement_unknown_tentative" in updates:
            setattr(
                self.args,
                "speaker_refinement_unknown_tentative",
                bool(updates.get("speaker_refinement_unknown_tentative")),
            )
        if "speaker_refinement_unknown_commit" in updates:
            setattr(
                self.args,
                "speaker_refinement_unknown_commit",
                bool(updates.get("speaker_refinement_unknown_commit")),
            )
        if "allow_speaker_reassignment" in updates:
            setattr(
                self.args,
                "allow_speaker_reassignment",
                bool(updates.get("allow_speaker_reassignment")),
            )
        settings = self.speaker_refinement_settings()
        self.bus.emit("status", {"message": "Speaker refinement settings updated."})
        if settings["unknown_commit"]:
            self._revisit_unknown_sentences()
        if settings["unknown_tentative"] or settings["allow_reassignment"]:
            self._refine_speaker_assignments()
        return settings

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

    def _ensure_realtime_preview_model(self) -> None:
        engine = normalize_preview_engine(getattr(self.args, "realtime_preview_engine", "off"))
        if engine in {"off", "mock"}:
            return
        if engine == "sherpa_onnx":
            model_dir = getattr(self.args, "realtime_preview_model_dir", None)
            if model_dir is None:
                raise RuntimeError("Nemotron realtime preview requires a model directory.")
            try:
                self.args.realtime_preview_model_dir = validate_sherpa_onnx_model_dir(Path(model_dir))
                return
            except RuntimeError:
                if not bool(getattr(self.args, "realtime_preview_auto_download", True)):
                    raise
            preset = str(getattr(self.args, "realtime_preview_model_preset", "") or "")
            self.bus.emit(
                "status",
                {"message": f"Nemotron preview model {preset} not found locally; downloading verified upstream archive."},
            )
            self.args.realtime_preview_model_dir = ensure_sherpa_onnx_model(preset, target_dir=Path(model_dir))
            self.bus.emit("status", {"message": f"Nemotron preview model ready: {self.args.realtime_preview_model_dir}."})
            return

        model_path = getattr(self.args, "realtime_preview_model_path", None)
        if model_path is not None:
            if Path(model_path).is_file():
                return
            raise RuntimeError(f"Kroko preview model path does not exist: {model_path}")

        if not bool(getattr(self.args, "realtime_preview_auto_download", DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD)):
            return

        model_name = str(getattr(self.args, "realtime_preview_model", "") or "")
        self.bus.emit(
            "status",
            {"message": f"Kroko preview model {model_name} not found locally; downloading from Hugging Face."},
        )
        model_path = download_kroko_preview_model(model_name)
        self.args.realtime_preview_model_path = model_path
        self.bus.emit("status", {"message": f"Kroko preview model ready: {model_path}."})

    def _load_realtime_preview(self) -> None:
        self._preview_transcriber = None
        engine = normalize_preview_engine(self.args.realtime_preview_engine)
        if engine == "off":
            self.bus.emit("status", {"message": "Realtime preview disabled."})
            return
        started = time.monotonic()
        try:
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Loading realtime preview engine {self.args.realtime_preview_engine} "
                        f"on {self.args.realtime_preview_provider} before playback."
                    )
                },
            )
            if engine != "mock":
                self._ensure_realtime_preview_model()
            self._preview_transcriber = create_realtime_preview_transcriber(self.args)
            location = getattr(self.args, "realtime_preview_model_dir", None) or getattr(
                self.args, "realtime_preview_model_path", None
            )
            self.bus.emit(
                "status",
                {
                    "message": (
                        f"Realtime preview ready in {time.monotonic() - started:.2f}s "
                        f"({engine}, {self.args.realtime_preview_model_preset}, "
                        f"{self.args.realtime_preview_language}, CPU x{self.args.realtime_preview_num_threads}"
                        f"{', ' + str(location) if location else ''})."
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

    def _format_realtime_preview_text(self, text: str, left: float) -> str:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return ""
        should_uppercase = (
            int(getattr(self, "_final_sentence_count", 0)) <= 0
            or bool(getattr(self, "_last_final_sentence_ended_strong", True))
            or float(left) <= 0.001
        )
        if should_uppercase:
            return sentence_initial_uppercase_after_strong_boundary(normalized)
        return normalized

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
        spans: list[tuple[float, float]] = []
        index = 0
        while index < len(flags):
            if not flags[index]:
                index += 1
                continue
            span_start_index = index
            while index < len(flags) and flags[index]:
                index += 1
            span_end_index = index - 1
            span_start = left + (starts[span_start_index] / sample_rate)
            span_end = left + (min(audio_size, starts[span_end_index] + frame_samples) / sample_rate)
            if span_end <= span_start:
                continue
            spans.append((round(float(span_start), 4), round(float(span_end), 4)))
        return self._vad_state_from_spans(left, right, spans, backend=backend)

    def _vad_state_from_spans(
        self,
        left: float,
        right: float,
        spans: list[tuple[float, float]],
        *,
        backend: str,
        min_speech_seconds: float | None = None,
    ) -> VadWindowState:
        spans = [
            (round(max(left, float(start)), 4), round(min(right, float(end)), 4))
            for start, end in sorted(spans)
            if min(right, float(end)) > max(left, float(start))
        ]
        if not spans:
            return VadWindowState(False, False, backend=backend)

        speech_seconds = sum(max(0.0, end - start) for start, end in spans)
        if min_speech_seconds is None:
            min_speech_seconds = float(self.args.vad_min_speech_seconds)
        if speech_seconds < max(0.0, float(min_speech_seconds)):
            return VadWindowState(False, False, backend=backend)

        speech_start = spans[0][0]
        speech_end = spans[-1][1]
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
            speech_spans=spans,
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

    def _webrtc_vad_window_state(
        self,
        left: float,
        right: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> VadWindowState:
        if getattr(self, "_webrtc_vad_error", None):
            return VadWindowState(False, False, backend="webrtc_unavailable")
        try:
            import webrtcvad  # type: ignore[import-not-found]
        except Exception as exc:
            self._webrtc_vad_error = str(exc)
            self.bus.emit(
                "status",
                {"message": f"WebRTC VAD unavailable; using primary VAD gate only: {exc}"},
            )
            return VadWindowState(False, False, backend="webrtc_unavailable")

        vad_audio = self._resample_for_silero_vad(audio, sample_rate)
        if vad_audio.size <= 0:
            return VadWindowState(False, False, backend="webrtc")

        mode = max(0, min(3, int(getattr(self.args, "vad_gate_webrtc_mode", 3))))
        detector = webrtcvad.Vad(mode)
        frame_samples = int(SILERO_VAD_SAMPLE_RATE * 0.03)
        frame_seconds = frame_samples / float(SILERO_VAD_SAMPLE_RATE)
        flags: list[bool] = []
        starts: list[int] = []
        for start in range(0, vad_audio.size, frame_samples):
            end = min(vad_audio.size, start + frame_samples)
            if end - start < max(1, frame_samples // 2):
                break
            chunk = vad_audio[start:end]
            if chunk.size < frame_samples:
                padded = np.zeros(frame_samples, dtype=np.float32)
                padded[:chunk.size] = chunk
                chunk = padded
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            try:
                flags.append(bool(detector.is_speech(pcm16, SILERO_VAD_SAMPLE_RATE)))
            except Exception as exc:
                self._webrtc_vad_error = str(exc)
                self.bus.emit(
                    "status",
                    {"message": f"WebRTC VAD failed; using primary VAD gate only: {exc}"},
                )
                return VadWindowState(False, False, backend="webrtc_unavailable")
            starts.append(start)

        return self._vad_state_from_flags(
            left=left,
            right=right,
            audio_size=vad_audio.size,
            sample_rate=SILERO_VAD_SAMPLE_RATE,
            frame_samples=frame_samples,
            frame_seconds=frame_seconds,
            flags=flags,
            starts=starts,
            backend=f"webrtc{mode}",
        )

    @staticmethod
    def _spans_overlap_seconds(
        source_spans: list[tuple[float, float]],
        start: float,
        end: float,
    ) -> float:
        overlap = 0.0
        for other_start, other_end in source_spans:
            overlap += max(0.0, min(end, other_end) - max(start, other_start))
        return overlap

    def _vad_gate_secondary_backend(self) -> str:
        return str(getattr(self.args, "vad_gate_secondary_backend", "webrtc") or "off").lower()

    def _vad_gate_evidence_spans(
        self,
        primary_state: VadWindowState,
        secondary_state: VadWindowState | None,
    ) -> list[tuple[float, float]]:
        primary_spans = list(primary_state.speech_spans or [])
        if not primary_spans and primary_state.speech_start is not None and primary_state.speech_end is not None:
            primary_spans = [(float(primary_state.speech_start), float(primary_state.speech_end))]
        if not primary_state.has_speech or not primary_spans:
            return []
        if (
            secondary_state is None
            or self._vad_gate_secondary_backend() == "off"
            or str(secondary_state.backend or "").startswith("webrtc_unavailable")
        ):
            return primary_spans

        secondary_spans = list(secondary_state.speech_spans or [])
        if not secondary_spans and secondary_state.speech_start is not None and secondary_state.speech_end is not None:
            secondary_spans = [(float(secondary_state.speech_start), float(secondary_state.speech_end))]
        if not secondary_state.has_speech or not secondary_spans:
            return []

        min_seconds = max(0.0, float(getattr(self.args, "vad_gate_min_consensus_seconds", 0.12)))
        min_ratio = max(0.0, min(1.0, float(getattr(self.args, "vad_gate_min_consensus_ratio", 0.05))))
        validated_primary: list[tuple[float, float]] = []
        for start, end in primary_spans:
            duration = max(0.0, float(end) - float(start))
            if duration <= 0.0:
                continue
            overlap = self._spans_overlap_seconds(secondary_spans, float(start), float(end))
            required = min(duration, max(min_seconds, duration * min_ratio))
            if overlap >= required:
                validated_primary.append((float(start), float(end)))
        if not validated_primary:
            return []

        evidence: list[tuple[float, float]] = []
        for start, end in secondary_spans:
            if self._spans_overlap_seconds(validated_primary, float(start), float(end)) > 0.0:
                evidence.append((float(start), float(end)))
        return evidence

    def _vad_gate_window_state(
        self,
        left: float,
        right: float,
        *,
        force: bool = False,
        primary_state: VadWindowState | None = None,
    ) -> VadWindowState:
        if primary_state is None:
            primary_state = self._vad_window_state(left, right, force=force)
        if self._vad_gate_secondary_backend() == "off" or not primary_state.has_speech:
            return primary_state
        audio, sample_rate = self._audio_window_copy(left, right)
        if audio.size <= 0 or sample_rate <= 0:
            return VadWindowState(False, False, backend=primary_state.backend)
        secondary_state = self._webrtc_vad_window_state(left, right, audio, sample_rate)
        evidence_spans = self._vad_gate_evidence_spans(primary_state, secondary_state)
        if not evidence_spans:
            return VadWindowState(False, False, backend=f"{primary_state.backend}+{secondary_state.backend}")
        min_speech = max(0.0, float(getattr(self.args, "vad_gate_min_consensus_seconds", 0.12)))
        return self._vad_state_from_spans(
            left,
            right,
            evidence_spans,
            backend=f"{primary_state.backend}+{secondary_state.backend}",
            min_speech_seconds=min_speech,
        )

    def _vad_window_state(self, left: float, right: float, *, force: bool = False) -> VadWindowState:
        if not force and not getattr(self.args, "vad_sentence_splitting", True):
            return VadWindowState(False, False)
        if right <= left:
            return VadWindowState(False, False)

        audio, sample_rate = self._audio_window_copy(left, right)
        if audio.size <= 0 or sample_rate <= 0:
            return VadWindowState(False, False)

        if getattr(self.args, "vad_backend", "silero") == "rms":
            return self._rms_vad_window_state(left, right, audio, sample_rate)
        return self._silero_vad_window_state(left, right, audio, sample_rate)

    def _asr_vad_gate_enabled(self) -> bool:
        return bool(getattr(self.args, "asr_vad_gate", True))

    def _asr_vad_gate_spans(
        self,
        left: float,
        right: float,
        vad_state: VadWindowState,
        secondary_vad_state: VadWindowState | None = None,
    ) -> list[tuple[float, float]]:
        if not self._asr_vad_gate_enabled():
            return [(left, right)]
        if right <= left or not vad_state.has_speech:
            return []

        source_spans = self._vad_gate_evidence_spans(vad_state, secondary_vad_state)
        if not source_spans:
            return []

        pre_padding = max(0.0, float(getattr(self.args, "asr_vad_gate_pre_padding_seconds", 0.20)))
        post_padding = max(0.0, float(getattr(self.args, "asr_vad_gate_post_padding_seconds", 0.35)))
        merge_gap = max(0.0, float(getattr(self.args, "asr_vad_gate_merge_gap_seconds", 0.85)))
        min_clip_seconds = max(0.0, float(getattr(self.args, "asr_vad_gate_min_clip_seconds", 0.20)))
        cut_internal_gaps = bool(getattr(self.args, "asr_vad_gate_cut_internal_gaps", False))
        if not cut_internal_gaps:
            speech_start = min(start for start, _end in source_spans)
            speech_end = max(end for _start, end in source_spans)
            span = (max(left, speech_start - pre_padding), min(right, speech_end + post_padding))
            return [span] if span[1] - span[0] >= min_clip_seconds else []

        spans: list[tuple[float, float]] = []
        for start, end in sorted((float(start), float(end)) for start, end in source_spans):
            padded_start = max(left, start - pre_padding)
            padded_end = min(right, end + post_padding)
            if padded_end - padded_start < min_clip_seconds:
                continue
            if spans and padded_start <= spans[-1][1] + merge_gap:
                spans[-1] = (spans[-1][0], max(spans[-1][1], padded_end))
            else:
                spans.append((padded_start, padded_end))
        return spans

    def _warm_sentence_splitter(self) -> None:
        if self._sentence_splitter_warmed:
            self.bus.emit("status", {"message": "stream2sentence tokenizer already warm."})
            return
        sentence_tokenizer = str(getattr(
            self.args,
            "sentence_tokenizer",
            default_sentence_tokenizer(getattr(self.args, "language", "en")),
        ))
        sentence_language = str(getattr(
            self.args,
            "sentence_language",
            default_sentence_language(getattr(self.args, "language", "en")),
        ))
        self.bus.emit(
            "status",
            {"message": f"Initializing stream2sentence {sentence_tokenizer}/{sentence_language} tokenizer before playback."},
        )
        started = time.monotonic()
        init_tokenizer(sentence_tokenizer, language=sentence_language)
        list(generate_sentences(
            list("A warmup sentence vs. a false split. Another sentence."),
            tokenizer=sentence_tokenizer,
            language=sentence_language,
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
        return SpeakerRefinementConfig(
            max_per_profile=int(getattr(self.args, "speaker_refinement_max_per_profile", 32)),
            prototype_min_duration=float(getattr(self.args, "speaker_refinement_min_duration", 0.15)),
            prototype_max_unknown=float(getattr(self.args, "speaker_refinement_max_unknown", 1.0)),
            top_k=int(getattr(self.args, "speaker_refinement_top_k", 12)),
            centroid_blend=float(getattr(self.args, "speaker_refinement_centroid_blend", 0.555)),
            unknown_min_similarity=float(getattr(self.args, "speaker_refinement_unknown_min_similarity", 0.20)),
            unknown_min_margin=float(getattr(self.args, "speaker_refinement_unknown_min_margin", 0.0)),
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
                with self._sentence_refinement_lock:
                    records = [
                        dict(record)
                        for _, record in sorted(self._sentence_refinement_records.items())
                    ]
                if len(records) >= 2:
                    revisions = find_speaker_prototype_revisions(
                        records,
                        self._speaker_refinement_config(),
                        allow_known_reassignment=allow_known,
                    )
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

    def _load_model(self) -> None:
        with self._model_lock:
            if self._model is not None:
                self.bus.emit("status", {"message": "ASR backend already loaded."})
                return
            asr_backend = str(self.args.asr_backend or "local").strip().lower().replace("-", "_")
            if asr_backend == "remote":
                client = RemoteWindowAsrClient(
                    self.args.remote_asr_url,
                    self.args.remote_asr_timeout_seconds,
                    language=getattr(self.args, "language", "en"),
                )
                self.bus.emit("status", {"message": f"Checking remote ASR server at {client.base_url}."})
                health = client.health()
                health_status = health.get("status") or health.get("model") or health.get("raw") or "ok"
                self._model = client
                self.bus.emit(
                    "status",
                    {
                        "message": (
                            f"Remote faster-whisper large-v2 ASR ready at {client.base_url} for {client.language} "
                            f"(health={health_status})."
                        )
                    },
                )
                return
            self.bus.emit("status", {"message": "Importing faster-whisper."})
            from faster_whisper import WhisperModel

            self.bus.emit("status", {"message": f"Loading faster-whisper {self.args.model} for {getattr(self.args, 'language', 'en')} on {self.args.device} before playback."})
            self._model = WhisperModel(
                self.args.model,
                device=self.args.device,
                compute_type=self.args.compute_type,
                download_root=str(self.args.download_root) if self.args.download_root else None,
            )
            self.bus.emit("status", {"message": "faster-whisper ready; starting synchronized playback."})

    def _transcribe_audio_words(self, model: Any, audio: np.ndarray, sample_rate: int) -> tuple[list[TimedWord], int]:
        if isinstance(model, RemoteWindowAsrClient):
            words, segment_count = model.transcribe_window(audio, sample_rate, self.args.beam_size)
            return self._filter_asr_no_speech_words(words), segment_count

        segments, _info = model.transcribe(
            audio,
            language=getattr(self.args, "language", "en"),
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
            segment_index = segment_count - 1
            no_speech_prob = self._optional_float(word_attr(segment, "no_speech_prob", None))
            avg_logprob = self._optional_float(word_attr(segment, "avg_logprob", None))
            compression_ratio = self._optional_float(word_attr(segment, "compression_ratio", None))
            for word in getattr(segment, "words", None) or []:
                text = str(word_attr(word, "word", "") or "")
                if not text.strip():
                    continue
                words.append(
                    TimedWord(
                        text,
                        float(word_attr(word, "start", 0.0)),
                        float(word_attr(word, "end", 0.0)),
                        probability=self._optional_float(word_attr(word, "probability", None)),
                        no_speech_prob=no_speech_prob,
                        avg_logprob=avg_logprob,
                        compression_ratio=compression_ratio,
                        segment_index=segment_index,
                    )
                )
        words.sort(key=lambda item: (item.start, item.end))
        return self._filter_asr_no_speech_words(words), segment_count

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _filter_asr_no_speech_words(self, words: list[TimedWord]) -> list[TimedWord]:
        if not bool(getattr(self.args, "asr_no_speech_filter", True)):
            return words
        threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_prob_threshold", 0.65))))
        hard_threshold = max(0.0, min(1.0, float(getattr(self.args, "asr_no_speech_hard_threshold", 0.85))))
        keep_short_max_words = max(0, int(getattr(self.args, "asr_no_speech_keep_short_max_words", 2)))
        keep_short_max_seconds = max(0.0, float(getattr(self.args, "asr_no_speech_keep_short_max_seconds", 0.45)))
        kept: list[TimedWord] = []
        dropped_words = 0
        dropped_segments = 0
        max_dropped_prob = 0.0

        def segment_key(word: TimedWord, fallback_index: int) -> tuple[object, ...]:
            if word.segment_index is not None:
                return ("segment", int(word.segment_index))
            if word.no_speech_prob is not None:
                return (
                    "metadata",
                    float(word.no_speech_prob),
                    word.avg_logprob,
                    word.compression_ratio,
                )
            return ("word", fallback_index)

        groups: list[list[TimedWord]] = []
        current_group: list[TimedWord] = []
        current_key: tuple[object, ...] | None = None
        for index, word in enumerate(words):
            key = segment_key(word, index)
            if current_group and key != current_key:
                groups.append(current_group)
                current_group = []
            current_group.append(word)
            current_key = key
        if current_group:
            groups.append(current_group)

        for group in groups:
            probability_values = [float(word.no_speech_prob) for word in group if word.no_speech_prob is not None]
            probability = max(probability_values) if probability_values else None
            if probability is None:
                kept.extend(group)
                continue
            start = min(float(word.start) for word in group)
            end = max(float(word.end) for word in group)
            duration = max(0.0, end - start)
            is_short_interjection = (
                probability < hard_threshold
                and len(group) <= keep_short_max_words
                and duration <= keep_short_max_seconds
            )
            if probability >= threshold and not is_short_interjection:
                dropped_words += len(group)
                dropped_segments += 1
                max_dropped_prob = max(max_dropped_prob, probability)
                continue
            kept.extend(group)
        bus = getattr(self, "bus", None)
        if dropped_words and bus is not None:
            bus.emit(
                "status",
                {
                    "message": (
                        f"ASR no-speech filter dropped {dropped_words} word(s) from {dropped_segments} segment(s) "
                        f"(max no_speech_prob={max_dropped_prob:.2f}, threshold={threshold:.2f})."
                    )
                },
            )
        return kept

    def _transcribe_window_audio_words(
        self,
        model: Any,
        left: float,
        right: float,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> tuple[list[TimedWord], int]:
        spans = speech_spans if speech_spans is not None else [(left, right)]
        words: list[TimedWord] = []
        segment_count = 0
        for span_left, span_right in spans:
            span_left = max(left, min(right, float(span_left)))
            span_right = max(span_left, min(right, float(span_right)))
            if span_right <= span_left:
                continue
            window, sample_rate = self._audio_window_copy(span_left, span_right)
            if window.size <= 0:
                continue
            relative_words, relative_segment_count = self._transcribe_audio_words(model, window, sample_rate)
            segment_count += relative_segment_count
            for word in relative_words:
                start = span_left + float(word.start)
                end = span_left + float(word.end)
                if end <= span_left or start >= span_right:
                    continue
                words.append(
                    TimedWord(
                        word.text,
                        max(left, min(right, start)),
                        max(left, min(right, end)),
                        probability=word.probability,
                        no_speech_prob=word.no_speech_prob,
                        avg_logprob=word.avg_logprob,
                        compression_ratio=word.compression_ratio,
                        segment_index=word.segment_index,
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
        try:
            words, segment_count = self._transcribe_audio_words(self._model, probe, sample_rate)
        except RuntimeError as exc:
            asr_backend = str(getattr(self.args, "asr_backend", "local") or "local").strip().lower().replace("-", "_")
            if asr_backend != "remote":
                raise
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Remote ASR warmup failed after server health check; "
                        f"continuing and retrying during transcription ({exc})."
                    )
                },
            )
            return
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
        if not self._live_speaker_assignment_enabled():
            return self._realtime_unknown_speaker_payload()
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
            speaker_label_generation=int(
                getattr(self, "_speaker_label_generations", {}).get(str(speaker_id), 0)
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

    def _live_memory_update_job_is_current(self, job: LiveSpeakerMemoryUpdateJob) -> bool:
        if job.speaker_generation != getattr(self, "_speaker_generation", 0):
            return False
        label_generations = getattr(self, "_speaker_label_generations", {})
        return int(getattr(job, "speaker_label_generation", 0)) == int(
            label_generations.get(job.speaker_id, 0)
        )

    def _process_live_speaker_memory_update(self, job: LiveSpeakerMemoryUpdateJob) -> None:
        try:
            if not self._live_memory_update_job_is_current(job):
                return
            if not self._live_update_speaker_exists(job.speaker_id):
                return
            embedding = self._embed_live_audio_chunk(job.audio, job.sample_rate, job.suffix)
            with self._live_memory_update_lock_obj():
                if not self._live_memory_update_job_is_current(job):
                    return
                if not self._live_update_speaker_exists(job.speaker_id):
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
        if not self._live_speaker_assignment_enabled():
            return
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
        vad_gate = bool(getattr(self.args, "realtime_preview_vad_gate", True))
        gate_open = not vad_gate
        gate_left = 0.0
        gate_search_left = 0.0
        gate_search_window = max(
            2.5,
            min_audio_seconds
            + max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_pre_padding_seconds", 0.35)))
            + max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_close_silence_seconds", 1.1))),
        )
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
                gate_open = not vad_gate
                gate_left = left
                gate_search_left = left
                try:
                    transcriber.reset_preview()
                except Exception as exc:
                    self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    time.sleep(interval_seconds)
                    continue
            active_left = gate_left if gate_open else gate_search_left
            if right - active_left < min_audio_seconds:
                time.sleep(0.05)
                continue
            if vad_gate and not gate_open:
                # Search every audio sample that has not already been proven to be
                # silence.  In particular, do not jump straight to the tail after
                # a slow final-ASR pass: speech may have started while that pass was
                # running, and skipping to the bounded tail cuts its first words.
                search_left = gate_search_left
                vad_state = self._vad_gate_window_state(search_left, right, force=True)
                if not vad_state.has_speech or vad_state.speech_start is None:
                    # Once the complete unseen range is known to contain no speech,
                    # retaining only a bounded tail keeps idle VAD work constant.
                    gate_search_left = max(gate_search_left, right - gate_search_window)
                    time.sleep(0.05)
                    continue
                pre_padding = max(0.0, float(getattr(self.args, "realtime_preview_vad_gate_pre_padding_seconds", 0.35)))
                gate_left = max(gate_search_left, float(vad_state.speech_start) - pre_padding)
                gate_open = True
                last_right = gate_left
                last_decode_right = -1.0
                last_diarized_right = -1.0
                last_speaker_payload = self._realtime_unknown_speaker_payload()
                try:
                    transcriber.reset_preview()
                except Exception as exc:
                    self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    gate_open = False
                    gate_search_left = max(gate_search_left, gate_left)
                    time.sleep(interval_seconds)
                    continue
            feed_limit = right
            close_after_feed = False
            close_search_left = gate_search_left
            if vad_gate and gate_open:
                vad_state = self._vad_gate_window_state(gate_left, right, force=True)
                close_silence = max(
                    0.0,
                    float(getattr(self.args, "realtime_preview_vad_gate_close_silence_seconds", 1.1)),
                )
                if not vad_state.has_speech:
                    close_after_feed = True
                    feed_limit = last_right
                    close_search_left = max(gate_left, right - gate_search_window)
                elif vad_state.trailing_silence_seconds >= close_silence and vad_state.speech_end is not None:
                    post_padding = max(
                        0.0,
                        float(getattr(self.args, "realtime_preview_vad_gate_post_padding_seconds", 0.35)),
                    )
                    speech_end = float(vad_state.speech_end)
                    feed_limit = max(last_right, min(right, speech_end + post_padding))
                    close_after_feed = True
                    close_search_left = max(gate_left, speech_end + post_padding)
            if feed_limit < last_right + feed_chunk_seconds and not close_after_feed:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now < next_at and not close_after_feed:
                time.sleep(min(0.05, next_at - now))
                continue
            try:
                text = ""
                while feed_limit >= last_right + feed_chunk_seconds:
                    feed_right = last_right + feed_chunk_seconds
                    audio, sample_rate = self._audio_window_copy(last_right, feed_right)
                    last_right = feed_right
                    if audio.size <= 0:
                        continue
                    text = " ".join(transcriber.accept_preview_audio(audio, sample_rate).split())
                if close_after_feed:
                    try:
                        transcriber.reset_preview()
                    except Exception as exc:
                        self.bus.emit("status", {"message": f"Realtime preview reset error: {type(exc).__name__}: {exc}"})
                    gate_open = False
                    gate_search_left = max(close_search_left, last_right)
                    gate_left = gate_search_left
                    last_right = gate_search_left
                    last_decode_right = -1.0
                    last_diarized_right = -1.0
                    last_speaker_payload = self._realtime_unknown_speaker_payload()
                    self.bus.emit("realtime_clear", {"generation": generation, "reason": "preview_vad_gate_closed"})
                    continue
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
            text = self._format_realtime_preview_text(text, gate_left)
            if not self._preview_generation_is_current(generation, left):
                continue
            duration_seconds = max(0.0, preview_right - gate_left)
            if self._live_speaker_assignment_enabled() and duration_seconds >= self.args.realtime_preview_diarize_min_audio_seconds and (
                last_diarized_right < 0.0 or preview_right >= last_diarized_right + diarize_min_advance
            ):
                if self.memory.profile_count() > 0 and self._try_reserve_live_speaker_embedding():
                    audio, _sample_rate = self._audio_window_copy(gate_left, preview_right)
                    last_speaker_payload = self._score_realtime_preview_speaker(audio, duration_seconds)
                    last_diarized_right = preview_right
                    if not self._preview_generation_is_current(generation, left):
                        continue
            self.bus.emit("realtime", {
                "index": f"rt-{generation}",
                "realtime": True,
                "realtime_generation": generation,
                "text": text,
                "start": round(gate_left, 4),
                "end": round(preview_right, 4),
                "audio_length_seconds": round(float(max(0.0, preview_right - gate_left)), 4),
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
            previous_emitted_sentence_ended_strong = True
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
                asr_vad_state = vad_state
                if self._asr_vad_gate_enabled():
                    if getattr(self.args, "vad_sentence_splitting", True):
                        asr_vad_state = self._vad_gate_window_state(
                            left,
                            right,
                            primary_state=vad_state,
                        )
                    else:
                        asr_vad_state = self._vad_gate_window_state(left, right, force=True)
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
                speech_spans: list[tuple[float, float]] | None = None
                if self._asr_vad_gate_enabled():
                    speech_spans = self._asr_vad_gate_spans(left, transcribe_right, asr_vad_state)
                    if not speech_spans:
                        self.bus.emit(
                            "status",
                            {
                                "message": (
                                    f"ASR VAD gate skipped non-speech window "
                                    f"left={left:.2f}s right={transcribe_right:.2f}s."
                                )
                            },
                        )
                        if vad_next_left is not None:
                            left = max(left, vad_next_left)
                            self._advance_realtime_preview_after_commit(left)
                        if media_final_flush:
                            break
                        time.sleep(0.05)
                        continue
                    kept_seconds = sum(max(0.0, span_right - span_left) for span_left, span_right in speech_spans)
                    if len(speech_spans) > 1 or kept_seconds < max(0.0, transcribe_right - left) - 0.05:
                        self.bus.emit(
                            "status",
                            {
                                "message": (
                                    f"ASR VAD gate kept {kept_seconds:.2f}s across "
                                    f"{len(speech_spans)} speech clip(s) from "
                                    f"{transcribe_right - left:.2f}s window."
                                )
                            },
                        )
                transcribe_started = time.monotonic()
                transcript = self._transcribe_window(
                    model,
                    left,
                    transcribe_right,
                    final_flush=transcript_final_flush,
                    previous_text_ended_sentence=previous_emitted_sentence_ended_strong,
                    speech_spans=speech_spans,
                )
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
                    previous_emitted_sentence_ended_strong = text_ends_sentence(sentence.text)
                    self._last_final_sentence_ended_strong = previous_emitted_sentence_ended_strong
                    self._final_sentence_count = int(getattr(self, "_final_sentence_count", 0)) + 1
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
            self._finalize_speaker_refinement()
            self._drain_live_memory_update_jobs()
            self.bus.emit("done", {"message": "Window diarization stopped."})

    def _transcribe_window(
        self,
        model: Any,
        left: float,
        right: float,
        final_flush: bool = False,
        previous_text_ended_sentence: bool = False,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> WindowTranscript:
        words, segment_count = self._transcribe_window_audio_words(model, left, right, speech_spans)
        words.sort(key=lambda item: (item.start, item.end))
        parts = split_words_with_stream2sentence(
            words,
            left=left,
            right=right,
            unstable_tail_seconds=self.args.unstable_tail_seconds,
            final_flush=final_flush,
            previous_text_ended_sentence=previous_text_ended_sentence,
            boundary_pre_padding_seconds=self.args.sentence_boundary_pre_padding_seconds,
            boundary_post_padding_seconds=self.args.sentence_boundary_post_padding_seconds,
            boundary_gap_ratio=self.args.sentence_boundary_gap_ratio,
            sentence_tokenizer=getattr(self.args, "sentence_tokenizer", "nltk+rule-based"),
            sentence_language=getattr(self.args, "sentence_language", getattr(self.args, "language", "en")),
        )
        return WindowTranscript(parts, len(words), segment_count)

    @staticmethod
    def _base_payload_from_sentence_part(
        index: int,
        sentence: SentencePart,
        window_left: float,
        window_right: float,
    ) -> dict[str, Any]:
        source_text_hash = hashlib.sha256(sentence.text.encode("utf-8")).hexdigest()
        return {
            "index": index,
            "text": sentence.text,
            "source_text_hash": source_text_hash,
            "source_revision": source_text_hash,
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

    def _emit_sentence(self, index: int, sentence: SentencePart, window_left: float, window_right: float) -> None:
        base_payload = self._base_payload_from_sentence_part(index, sentence, window_left, window_right)
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
            self._emit_transcript_sentence({
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
            payload = {
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
                "assignment_source": "non_embedding_candidate",
            }
            self._emit_transcript_sentence(payload)
            self._record_unknown_refinement_candidate(
                index,
                base_payload,
                max(0.0, sentence.end - sentence.start),
                payload,
            )
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
        if not self._live_speaker_assignment_enabled():
            return
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

    def _apply_sentence_embedding_decision(
        self,
        *,
        index: int,
        base_payload: dict[str, Any],
        text: str,
        embedding: np.ndarray,
        duration_seconds: float,
        live_memory_audio: np.ndarray | None = None,
        live_memory_sample_rate: int | None = None,
        live_memory_suffix: str = ".live-sentence.wav",
        speaker_generation: int | None = None,
        emit_status: bool = True,
        elapsed_seconds: float | None = None,
        run_speaker_refinement: bool = True,
    ) -> dict[str, Any]:
        paired_unknown_revision: tuple[PendingUnknownSentence, float] | None = None
        allow_new_speaker = len(text_content_words(text)) >= self.args.min_new_speaker_words
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
                decision, paired_candidate, pair_similarity = pair_decision
                paired_unknown_revision = (paired_candidate, pair_similarity)
        if emit_status:
            elapsed = 0.0 if elapsed_seconds is None else max(0.0, float(elapsed_seconds))
            self.bus.emit("status", {
                "message": (
                    f"Embedded sentence {index} in {elapsed:.2f}s; "
                    f"speaker={decision.assigned_speaker or 'UNKNOWN'} "
                    f"new={int(bool(decision.created_speaker))} "
                    f"unk={decision.unknown_probability} "
                    f"top={decision.top_similarity} "
                    f"margin={decision.margin}."
                )
            })
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
        sentence_payload = self._emit_transcript_sentence(sentence_payload)
        if paired_unknown_revision is not None:
            paired_candidate, pair_similarity = paired_unknown_revision
            self._emit_unknown_pair_revision(paired_candidate, decision, pair_similarity)
        self._maybe_emit_sentence_live_speaker_hint(sentence_payload, duration_seconds)
        if live_memory_audio is not None and live_memory_sample_rate is not None:
            self._update_live_speaker_memory(
                decision.assigned_speaker,
                live_memory_audio,
                live_memory_sample_rate,
                duration_seconds,
                live_memory_suffix,
                speaker_generation=self._speaker_generation if speaker_generation is None else speaker_generation,
            )
        if decision.assigned_speaker is None:
            self._remember_unknown_sentence(index, base_payload, embedding, duration_seconds)
        elif decision.created_speaker:
            self.emit_speaker_state()
            self._revisit_unknown_sentences()
        if run_speaker_refinement:
            self._refine_speaker_assignments()
        return sentence_payload

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
            self._apply_sentence_embedding_decision(
                index=index,
                base_payload=base_payload,
                text=job.text,
                embedding=embedding,
                duration_seconds=duration_seconds,
                live_memory_audio=chunk,
                live_memory_sample_rate=job.sample_rate,
                live_memory_suffix=".live-sentence.wav",
                speaker_generation=job.speaker_generation,
                emit_status=True,
                elapsed_seconds=time.monotonic() - embed_started,
            )
        except Exception as exc:
            self.bus.emit("status", {"message": f"Embedding failed for sentence {index}: {type(exc).__name__}: {exc}"})
            self._emit_transcript_sentence({
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


