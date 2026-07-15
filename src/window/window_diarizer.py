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
from speakers.person_library import PersonLibrary
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
class StartSessionRequest:
    session_id: str = ""
    source_title: str = ""

    def __post_init__(self) -> None:
        session_id = str(self.session_id or "").strip()
        if session_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
            raise ValueError("Invalid session id.")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "source_title", " ".join(str(self.source_title or "").split())[:120])


@dataclass(frozen=True)
class WindowDiarizerDependencies:
    audio_loader: Any = load_audio_file
    monotonic: Any = time.monotonic
    embedding_factory: Any = None
    live_embedding_factory: Any = None
    memory_factory: Any = None
    thread_factory: Any = threading.Thread


from window.window_diarizer_session_views import WindowSessionViewMixin
from window.window_diarizer_speaker_review import WindowSpeakerReviewMixin
from window.window_diarizer_speaker_state import WindowSpeakerStateMixin
from window.window_diarizer_speaker_library import WindowSpeakerLibraryMixin
from window.window_diarizer_runtime_audio import WindowRuntimeAudioMixin
from window.window_diarizer_assignment import WindowAssignmentDecisionMixin
from window.window_diarizer_refinement_rules import WindowRefinementRulesMixin
from window.window_diarizer_refinement import WindowRefinementMixin
from window.window_diarizer_models import WindowModelRuntimeMixin
from window.window_diarizer_live_scoring import WindowLiveScoringMixin
from window.window_diarizer_live_probe import WindowLiveProbeMixin
from window.window_diarizer_transcription import WindowTranscriptionMixin
from window.window_diarizer_runtime_state import WindowRuntimeStateMixin
from window.window_diarizer_people import WindowPersonIdentityMixin


class WindowDiarizer(WindowSessionViewMixin, WindowPersonIdentityMixin, WindowSpeakerStateMixin, WindowSpeakerReviewMixin, WindowSpeakerLibraryMixin, WindowRuntimeAudioMixin, WindowRuntimeStateMixin, WindowAssignmentDecisionMixin, WindowRefinementRulesMixin, WindowRefinementMixin, WindowModelRuntimeMixin, WindowLiveScoringMixin, WindowLiveProbeMixin, WindowTranscriptionMixin):
    def __init__(
        self,
        args: argparse.Namespace | DiarizationConfig,
        media: MediaFiles,
        bus: EventBus,
        *,
        dependencies: WindowDiarizerDependencies | None = None,
    ) -> None:
        self.dependencies = dependencies or WindowDiarizerDependencies()
        self.args = DiarizationConfig.from_namespace(args)
        self._config_lock = threading.RLock()
        self.bus = bus
        self._audio_timeline = AudioTimeline(
            media,
            audio_loader=self.dependencies.audio_loader,
            monotonic=self.dependencies.monotonic,
        )
        self._audio_lock = self._audio_timeline.lock
        self._sync_audio_aliases(self._audio_timeline.snapshot(copy_audio=False))
        self.embedding = self._new_embedding_client(self.args)
        self.memory = self._new_memory()
        self.live_embedding = self._new_live_embedding_client(self.args)
        self._live_embedding_separate = self.live_embedding is not self.embedding
        self.live_memory = self._new_memory() if self._live_embedding_separate else self.memory
        self._live_probability_history: deque[tuple[float, dict[str, float]]] = deque(
            maxlen=max(1, int(getattr(self.args, "live_speaker_ema_count", 3)))
        )
        self.speaker_library_dir = Path(getattr(args, "speaker_library_dir", DEFAULT_SPEAKER_LIBRARY_DIR))
        self.person_library = PersonLibrary(self.speaker_library_dir / "people.json")
        self._session_state = DiarizationSession()
        self._assignment_engine = SpeakerAssignmentEngine()
        self._speaker_lock = self._session_state.lock
        self._person_learning_lock = threading.RLock()
        self._person_learning_states: dict[str, Any] = {}
        self._person_learning_fallback_session_id = ""
        self._expected_person_ids: set[str] | None = None
        self._speaker_group_name = ""
        self._speaker_metadata: dict[str, dict[str, Any]] = {}
        self._seed_profiles: list[dict[str, Any]] = []
        self._seed_live_profiles: list[dict[str, Any]] = []
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._active_run: DiarizationRun | None = None
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
        self._unknown_lock = self._session_state.lock
        self._unknown_sentences: list[PendingUnknownSentence] = []
        self._recent_unknown_pair_candidates: deque[PendingUnknownSentence] = deque(maxlen=24)
        self._sentence_refinement_lock = self._session_state.lock
        self._sentence_refinement_records: dict[int, dict[str, Any]] = {}
        self._correction_history: list[dict[str, Any]] = []
        self._sentence_refinement_run_lock = self._session_state.lock
        self._speaker_last_media_end: dict[str, float] = {}
        self._embedding_jobs: "queue.Queue[EmbeddingSentenceJob | None] | None" = None
        self._embedding_thread: threading.Thread | None = None
        self._live_memory_update_jobs: "queue.Queue[LiveSpeakerMemoryUpdateJob | None] | None" = None
        self._live_memory_update_thread: threading.Thread | None = None
        self._live_memory_update_lock = self._session_state.lock
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

    def _sync_audio_aliases(self, snapshot: AudioSnapshot) -> None:
        """Expose read-compatible fields while AudioTimeline owns production writes."""

        self.media = snapshot.media
        self.audio = snapshot.audio
        self.sample_rate = snapshot.sample_rate
        self.duration = snapshot.duration
        self._streaming_audio = snapshot.streaming
        self._stream_audio_samples = snapshot.stream_samples
        # Migrated production reads go through AudioTimeline.  Retain the name
        # temporarily for older tests and extensions without duplicating chunks.
        self._stream_audio_chunks = []

    def _update_config(self, **updates: Any) -> None:
        """Atomically replace production config while supporting legacy test doubles."""

        lock = getattr(self, "_config_lock", None)
        if isinstance(self.args, DiarizationConfig):
            if lock is None:
                self.args = self.args.with_updates(**updates)
            else:
                with lock:
                    self.args = self.args.with_updates(**updates)
            return
        for key, value in updates.items():
            setattr(self.args, key, value)

    def _new_embedding_client(self, args: argparse.Namespace, provider: str | None = None) -> Any:
        embeddings_backend = str(getattr(args, "embeddings_backend", "local") or "local").strip().lower().replace("-", "_")
        embedding_provider = str(provider or args.embedding_provider)
        factory = getattr(getattr(self, "dependencies", None), "embedding_factory", None)
        if callable(factory):
            return factory(args, embedding_provider)
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
        factory = getattr(getattr(self, "dependencies", None), "live_embedding_factory", None)
        if callable(factory):
            return factory(args, self.embedding)
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

    def start(self, request: StartSessionRequest | None = None) -> dict[str, Any]:
        request = request or StartSessionRequest(
            session_id=getattr(self, "_next_session_id", ""),
            source_title=getattr(self, "_session_source_title", ""),
        )
        self.bus.emit("status", {"message": "Start requested; preparing models before playback."})
        self.stop()
        with self._lifecycle_lock:
            if self._active_run is not None:
                raise RuntimeError(
                    f"Previous diarization run {self._active_run.run_id} did not stop cleanly: "
                    f"{self._active_run.failure or self._active_run.state.value}"
                )
            run = DiarizationRun()
            self._active_run = run
        refresh_runtime_warmup = self._should_refresh_start_runtime_warmup()
        self._prepare_model_dependencies(
            include_asr_probe=refresh_runtime_warmup,
            force_runtime_warmup=refresh_runtime_warmup,
        )
        self.bus.emit("status", {"message": "Loading realtime preview engine."})
        self._load_realtime_preview()
        speaker_state = self._reset_runtime_session_state()
        self._stop = run.stop_event
        self._session_id = request.session_id or uuid.uuid4().hex
        self._session_source_title = request.source_title
        self._next_session_id = ""
        self._session_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.set_playback_time(0.0, reset=True)
        self._playback_clock_started_at = time.monotonic()
        self._last_playback_jump_warning_at = 0.0
        cancelled_transcriber: RealtimePreviewTranscriber | None = None
        launched = False
        with self._lifecycle_lock:
            if self._active_run is not run or run.stop_event.is_set():
                cancelled_transcriber = self._preview_transcriber
                self._preview_transcriber = None
            else:
                self._start_embedding_worker()
                self._start_live_memory_update_worker()
                run.main_thread = self.dependencies.thread_factory(
                    target=self._run_main_worker,
                    args=(run,),
                    name=f"WindowDiarizer-{run.run_id[:8]}",
                    daemon=True,
                )
                self._thread = run.main_thread
                if self._preview_transcriber is not None:
                    run.preview_thread = self.dependencies.thread_factory(
                        target=self._run_realtime_preview,
                        args=(run.stop_event,),
                        name=f"RealtimePreview-{run.run_id[:8]}",
                        daemon=True,
                    )
                    self._preview_thread = run.preview_thread
                if self._live_speaker_assignment_enabled() and bool(getattr(self.args, "live_speaker_probe", True)):
                    run.live_probe_thread = self.dependencies.thread_factory(
                        target=self._run_live_speaker_probe,
                        args=(run.stop_event,),
                        name=f"LiveSpeakerProbe-{run.run_id[:8]}",
                        daemon=True,
                    )
                    self._live_probe_thread = run.live_probe_thread
                run.mark_running()
                # Auxiliary consumers must exist before the main producer can
                # finish so `done` follows their shutdown.
                if run.preview_thread is not None:
                    run.preview_thread.start()
                if run.live_probe_thread is not None:
                    run.live_probe_thread.start()
                run.main_thread.start()
                launched = True
        if cancelled_transcriber is not None:
            cancelled_transcriber.close()
        if not launched:
            raise RuntimeError("Diarization start was cancelled before worker launch.")
        self.bus.emit("status", {"message": "Diarization worker started; synchronized playback can begin."})
        return speaker_state

    def _run_main_worker(self, run: DiarizationRun) -> None:
        try:
            self._run(run.stop_event)
        except BaseException as exc:
            run.mark_failed(f"{type(exc).__name__}: {exc}")
        finally:
            # Natural media completion owns auxiliary-worker shutdown.  The
            # captured event cannot accidentally be replaced by a later run.
            run.stop_event.set()
            current = threading.current_thread()
            cleanup_deadline = self.dependencies.monotonic() + 15.0
            for thread in (run.preview_thread, run.live_probe_thread):
                if thread is None or thread is current or not thread.is_alive():
                    continue
                thread.join(timeout=max(0.0, cleanup_deadline - self.dependencies.monotonic()))
            alive = [
                thread.name
                for thread in (run.preview_thread, run.live_probe_thread)
                if thread is not None and thread is not current and thread.is_alive()
            ]
            if alive:
                run.mark_failed(f"auxiliary workers missed shutdown deadline: {', '.join(alive)}")
            self._stop_embedding_worker()
            self._stop_live_memory_update_worker()
            if hasattr(self, "person_library"):
                try:
                    self.consolidate_confirmed_people()
                except Exception as exc:
                    self.bus.emit("status", {
                        "message": f"Could not update remembered people from the completed meeting: {exc}"
                    })
            with self._lifecycle_lock:
                preview_transcriber = self._preview_transcriber
                self._preview_transcriber = None
            if preview_transcriber is not None:
                preview_transcriber.close()
            if not run.done_emitted:
                run.done_emitted = True
                self.bus.emit("done", {"message": "Window diarization stopped."})
            if run.state is not DiarizationRunState.FAILED:
                run.mark_idle()
                with self._lifecycle_lock:
                    if self._active_run is run:
                        self._active_run = None

    def set_session_source_title(self, title: str) -> None:
        self._session_source_title = " ".join(str(title or "").strip().split())[:120]

    def set_next_session_id(self, session_id: str) -> None:
        normalized = str(session_id or "").strip()
        if normalized and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", normalized):
            raise ValueError("Invalid session id.")
        self._next_session_id = normalized

    def is_running(self) -> bool:
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._thread, self._preview_thread, self._live_probe_thread)
        )

    def stop(self) -> None:
        lifecycle_lock = getattr(self, "_lifecycle_lock", None)
        run = getattr(self, "_active_run", None)
        if run is not None:
            with lifecycle_lock:
                run.request_stop()
            deadline = self.dependencies.monotonic() + 15.0
            current = threading.current_thread()
            for thread in run.threads():
                if thread is current or not thread.is_alive():
                    continue
                thread.join(timeout=max(0.0, deadline - self.dependencies.monotonic()))
            alive = [thread.name for thread in run.threads() if thread is not current and thread.is_alive()]
            if alive:
                run.mark_failed(f"workers missed shutdown deadline: {', '.join(alive)}")
                return
        else:
            getattr(self, "_stop", threading.Event()).set()
            for thread in (
                getattr(self, "_thread", None),
                getattr(self, "_preview_thread", None),
                getattr(self, "_live_probe_thread", None),
            ):
                if thread is not None and thread is not threading.current_thread() and thread.is_alive():
                    thread.join(timeout=2.0)

        self._thread = None
        self._preview_thread = None
        self._live_probe_thread = None
        if self._preview_transcriber is not None:
            self._preview_transcriber.close()
        self._preview_transcriber = None
        self._playback_clock_started_at = None
        self._drain_embedding_jobs()
        self._stop_embedding_worker()
        self._drain_live_memory_update_jobs()
        self._stop_live_memory_update_worker()
        if run is not None:
            run.mark_idle()
            with lifecycle_lock:
                if self._active_run is run:
                    self._active_run = None

    def set_media(self, media: MediaFiles) -> None:
        self.stop()
        snapshot = self._audio_timeline.replace_file(media, audio_loader=self.dependencies.audio_loader)
        self._sync_audio_aliases(snapshot)
        self._reset_runtime_session_state()
        self.set_playback_time(0.0, reset=True)

    def set_browser_stream(self, url: str) -> MediaFiles:
        self.stop()
        snapshot = self._audio_timeline.begin_stream(url)
        self._sync_audio_aliases(snapshot)
        self._reset_runtime_session_state()
        self.set_playback_time(0.0, reset=True)
        return snapshot.media

    def append_stream_audio(self, audio: np.ndarray, sample_rate: int) -> float:
        timeline = getattr(self, "_audio_timeline", None)
        if timeline is not None:
            duration = timeline.append(audio, sample_rate)
            self._sync_audio_aliases(timeline.snapshot(copy_audio=False))
            return duration

        # Temporary compatibility for legacy partial-object tests.  Production
        # construction always installs AudioTimeline.
        if not self._streaming_audio:
            raise RuntimeError("Browser audio stream is not active.")
        if int(sample_rate) != int(self.sample_rate):
            raise RuntimeError(f"Browser audio sample rate changed from {self.sample_rate} to {sample_rate}.")
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        if chunk.size <= 0:
            return self.duration
        chunk = np.clip(np.nan_to_num(chunk, copy=False), -1.0, 1.0)
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
