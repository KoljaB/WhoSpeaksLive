"""Runtime-model warmup and per-session memory ownership helpers."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

from speakers.speaker_embedding_cluster import SpeakerMemory
from window.window_domain import PendingUnknownSentence


class WindowRuntimeStateMixin:
    def _prepare_model_dependencies(self, include_asr_probe: bool, force_runtime_warmup: bool = False) -> None:
        self._load_speech_enhancement()
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
                {"message": f"Previous runtime warmup is {age:.1f}s old; refreshing ASR and embeddings before playback."},
            )
            return True
        self.bus.emit(
            "status",
            {"message": f"Previous runtime warmup is {age:.1f}s old; skipping refresh before playback."},
        )
        return False

    def _new_memory(self) -> SpeakerMemory:
        factory = getattr(getattr(self, "dependencies", None), "memory_factory", None)
        if callable(factory):
            return factory(self.args)
        return SpeakerMemory(
            same_speaker_similarity=self.args.same_speaker_similarity,
            similarity_temperature=self.args.similarity_temperature,
            speaker_softmax_temperature=self.args.speaker_softmax_temperature,
            new_speaker_threshold=self.args.new_speaker_threshold,
            duplicate_profile_similarity=self.args.duplicate_profile_similarity,
            unknown_short_threshold=self.args.unknown_short_threshold,
            min_first_speaker_seconds=self.args.min_first_speaker_seconds,
            first_speaker_immediate_min_seconds=getattr(
                self.args, "first_speaker_immediate_min_seconds", self.args.min_first_speaker_seconds
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
            known_speaker_gray_zone_min_unknown_probability=self.args.known_speaker_gray_zone_min_unknown_probability,
            profile_update_min_similarity=self.args.profile_update_min_similarity,
            profile_update_min_margin=self.args.profile_update_min_margin,
            low_similarity_unknown_floor_similarity=self.args.low_similarity_unknown_floor_similarity,
            low_similarity_unknown_floor_probability=self.args.low_similarity_unknown_floor_probability,
            gray_zone_promote_max_similarity=getattr(self.args, "gray_zone_promote_max_similarity", 1.0),
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
        candidates = getattr(self, "_recent_unknown_pair_candidates", None)
        if candidates is None:
            candidates = deque(maxlen=24)
            self._recent_unknown_pair_candidates = candidates
        return candidates

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
        self.memory.replace_profiles(seed_profiles or [])
        if seed_profiles:
            with self._speaker_lock:
                self._speaker_metadata = seed_metadata
        self._sync_metadata_with_memory()
        self._reset_live_speaker_memory()
        if self._live_embedding_separate and seed_live_profiles:
            for index, profile in enumerate(seed_live_profiles, 1):
                self.live_memory.upsert_profile(
                    str(profile.get("label") or f"S{index}"),
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
        self._reset_person_learning_state()
        self._expected_person_ids = self.person_library.expected_person_ids()
        return self.emit_speaker_state() if emit else self.speaker_state()
