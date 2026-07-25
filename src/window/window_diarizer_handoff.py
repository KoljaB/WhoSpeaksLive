"""Low-cost integration for speaker handoffs inside completed sentences."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from common.audio_utils import pad_audio, trim_silence
from window.live_speech_gate import rms_speech_features
from window.sentence_speaker_handoff import (
    HandoffConfig,
    LiveEmbeddingEvidence,
    nominate_stable_handoff,
    normalize_embedding,
    select_context_handoff,
    split_sentence_part,
)
from window.window_domain import SentencePart


class WindowSentenceSpeakerHandoffMixin:
    """Reuse live probes to nominate and precisely verify one A-to-B handoff."""

    def _sentence_speaker_handoff_enabled(self) -> bool:
        return bool(
            getattr(self.args, "sentence_speaker_handoff", True)
            and getattr(self.args, "live_speaker_assignment", True)
        )

    def _emit_sentence_handoff_internal(
        self,
        event: str,
        payload: Mapping[str, Any],
    ) -> None:
        emit_internal = getattr(self.bus, "emit_internal", None)
        if callable(emit_internal):
            emit_internal(event, dict(payload))

    def _record_sentence_handoff_live_evidence(
        self,
        *,
        request_source: str,
        run_id: str,
        probe_id: str,
        request_id: str,
        window_start: float,
        window_end: float,
        source_start_sample: int | None,
        source_end_sample: int | None,
        audio: np.ndarray,
        sample_rate: int,
        embedding: np.ndarray,
        visible_speaker: str | None,
        similarities: Mapping[str, float] | None,
        probabilities: Mapping[str, float] | None,
        profile_generations: Mapping[str, int] | None,
        provider: str,
    ) -> None:
        """Retain only fixed short probes; failures never affect live scoring."""

        del (
            probe_id,
            request_id,
            source_start_sample,
            source_end_sample,
            similarities,
            probabilities,
            profile_generations,
        )
        if (
            not self._sentence_speaker_handoff_enabled()
            or str(request_source) != "dedicated_live_probe"
        ):
            return
        try:
            frame_seconds = max(
                0.01,
                float(getattr(self.args, "vad_frame_seconds", 0.03)),
            )
            rms_threshold = max(
                0.0,
                float(getattr(self.args, "vad_speech_rms_threshold", 0.003)),
            )
            _, _, _, voiced_seconds = rms_speech_features(
                audio,
                int(sample_rate),
                frame_seconds=frame_seconds,
                threshold=rms_threshold,
                min_speech_seconds=0.0,
            )
            item = LiveEmbeddingEvidence(
                window_start=float(window_start),
                window_end=float(window_end),
                short_embedding=embedding,
                visible_speaker=visible_speaker,
                # The raw vector is enough to rescore against any later profile
                # generation.  Per-probe score dictionaries make a one-hour
                # history much larger without adding usable acoustic evidence.
                similarities={},
                profile_generations={},
                provider=str(provider or ""),
                voiced_seconds=voiced_seconds,
            )
            lock = self._sentence_handoff_evidence_lock
            evidence = self._sentence_handoff_evidence
            with lock:
                previous_run = str(
                    getattr(self, "_sentence_handoff_evidence_run_id", "") or ""
                )
                previous_provider = str(
                    getattr(self, "_sentence_handoff_evidence_provider", "") or ""
                )
                current_run = str(run_id or "")
                current_provider = str(provider or "")
                if (
                    (previous_run and current_run and previous_run != current_run)
                    or (
                        previous_provider
                        and current_provider
                        and previous_provider != current_provider
                    )
                ):
                    evidence.clear()
                    # Sentence candidates are meaningful only together with the
                    # probe timeline from the same run.
                    self._sentence_handoff_hindsight_candidates = {}
                    self._sentence_handoff_hindsight_run_id = current_run
                    self._sentence_handoff_hindsight_finalized_run_id = None
                self._sentence_handoff_evidence_run_id = current_run
                self._sentence_handoff_evidence_provider = current_provider
                if (
                    evidence
                    and item.window_end < evidence[-1].window_end - 1e-4
                ):
                    # A backwards media seek must not combine two timelines.
                    evidence.clear()
                    self._sentence_handoff_hindsight_candidates = {}
                    self._sentence_handoff_hindsight_run_id = current_run
                    self._sentence_handoff_hindsight_finalized_run_id = None
                if (
                    evidence
                    and abs(item.window_start - evidence[-1].window_start) <= 1e-4
                    and abs(item.window_end - evidence[-1].window_end) <= 1e-4
                ):
                    evidence[-1] = item
                else:
                    evidence.append(item)
                cache_seconds = max(
                    5.0,
                    float(
                        getattr(
                            self.args,
                            "sentence_speaker_handoff_cache_seconds",
                            3600.0,
                        )
                    ),
                )
                cutoff = item.window_end - cache_seconds
                while evidence and evidence[0].window_end < cutoff:
                    evidence.popleft()
        except Exception as exc:
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_cache_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "window_start": round(float(window_start), 6),
                    "window_end": round(float(window_end), 6),
                },
            )

    def _sentence_handoff_config(self) -> HandoffConfig:
        short_window = max(
            0.1,
            float(
                getattr(
                    self.args,
                    "live_speaker_probe_window_seconds",
                    0.7,
                )
            ),
        )
        return HandoffConfig(
            short_window_seconds=short_window,
            short_window_tolerance_seconds=max(0.12, short_window * 0.32),
            min_probes_per_side=2,
            min_live_voiced_seconds_per_side=0.30,
            require_same_provider=True,
            min_words_per_segment=2,
            min_word_support_per_segment=0.0,
            min_gain_over_no_split=0.25,
            min_gain_over_reverse=0.25,
            min_gain_over_runner_up_cut=0.08,
            min_context_similarity=max(
                -1.0,
                min(
                    1.0,
                    float(
                        getattr(
                            self.args,
                            "sentence_speaker_handoff_min_context_similarity",
                            0.40,
                        )
                    ),
                ),
            ),
            min_context_pair_margin=max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "sentence_speaker_handoff_min_context_margin",
                        0.15,
                    )
                ),
            ),
            min_context_runner_up_margin=max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "sentence_speaker_handoff_min_context_runner_up_margin",
                        0.08,
                    )
                ),
            ),
            min_context_separation=max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "sentence_speaker_handoff_min_context_separation",
                        0.50,
                    )
                ),
            ),
        )

    def _snapshot_sentence_handoff_evidence(
        self,
        sentence: SentencePart,
    ) -> tuple[LiveEmbeddingEvidence, ...]:
        lock = getattr(self, "_sentence_handoff_evidence_lock", None)
        evidence = getattr(self, "_sentence_handoff_evidence", None)
        if lock is None or evidence is None:
            return ()
        padding = max(
            0.5,
            float(
                getattr(
                    self.args,
                    "live_speaker_probe_window_seconds",
                    0.7,
                )
            ),
        )
        with lock:
            return tuple(
                item
                for item in evidence
                if item.window_end >= sentence.start - padding
                and item.window_start <= sentence.end + padding
            )

    def _current_sentence_handoff_profile_anchors(
        self,
    ) -> dict[str, np.ndarray]:
        live_memory = getattr(self, "live_memory", None)
        export_profiles = getattr(live_memory, "export_profiles", None)
        if not callable(export_profiles):
            return {}
        profiles = export_profiles()
        anchors: dict[str, np.ndarray] = {}
        for profile in profiles:
            label = str(profile.get("label") or "").strip()
            speech_seconds = max(0.0, float(profile.get("speech_seconds") or 0.0))
            sentence_count = max(0, int(profile.get("sentence_count") or 0))
            if (
                not bool(profile.get("locked"))
                and speech_seconds < 1.0
                and sentence_count < 2
            ):
                continue
            centroid = np.asarray(profile.get("centroid"), dtype=np.float32).reshape(-1)
            if not label or centroid.size <= 0 or not np.all(np.isfinite(centroid)):
                continue
            norm = float(np.linalg.norm(centroid))
            if norm <= 0.0:
                continue
            anchors[label] = np.ascontiguousarray(centroid / norm, dtype=np.float32)
        return anchors

    def _rescore_sentence_handoff_evidence(
        self,
        evidence: Sequence[LiveEmbeddingEvidence],
        anchors: Mapping[str, np.ndarray],
        *,
        provider: str,
    ) -> tuple[LiveEmbeddingEvidence, ...]:
        """Assign cached raw probes against the latest established profiles."""

        min_similarity = float(
            getattr(self.args, "realtime_preview_diarize_min_similarity", 0.45)
        )
        min_margin = max(
            0.0,
            float(getattr(self.args, "realtime_preview_diarize_min_margin", 0.08)),
        )
        rescored: list[LiveEmbeddingEvidence] = []
        for item in evidence:
            if provider and item.provider and item.provider != provider:
                continue
            scores: dict[str, float] = {}
            for label, anchor in anchors.items():
                if item.short_embedding.shape != anchor.shape:
                    continue
                scores[label] = float(np.dot(item.short_embedding, anchor))
            ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            assigned: str | None = None
            if ordered:
                top_label, top_score = ordered[0]
                runner_up = ordered[1][1] if len(ordered) > 1 else -1.0
                if top_score >= min_similarity and (
                    len(ordered) == 1 or top_score - runner_up >= min_margin
                ):
                    assigned = top_label
            rescored.append(
                LiveEmbeddingEvidence(
                    window_start=item.window_start,
                    window_end=item.window_end,
                    short_embedding=item.short_embedding,
                    visible_speaker=assigned,
                    similarities=scores,
                    profile_generations=item.profile_generations,
                    provider=item.provider,
                    voiced_seconds=item.voiced_seconds,
                )
            )
        return tuple(rescored)

    @staticmethod
    def _sentence_handoff_context_audio(
        sentence_audio: np.ndarray,
        sample_rate: int,
        *,
        sentence_start: float,
        context_start: float,
        context_end: float,
    ) -> np.ndarray:
        relative_start = max(0.0, float(context_start) - float(sentence_start))
        relative_end = max(relative_start, float(context_end) - float(sentence_start))
        start_sample = min(
            int(sentence_audio.size),
            max(0, int(np.floor(relative_start * sample_rate))),
        )
        end_sample = min(
            int(sentence_audio.size),
            max(start_sample, int(np.ceil(relative_end * sample_rate))),
        )
        return sentence_audio[start_sample:end_sample].copy()

    def _try_embed_sentence_handoff_live_context(
        self,
        audio: np.ndarray,
        sample_rate: int,
        suffix: str,
    ) -> np.ndarray | None:
        """Embed only when the live provider is idle; never contend with probes."""

        lock_factory = getattr(self, "_live_speaker_inference_lock_obj", None)
        if not callable(lock_factory):
            return self._embed_live_audio_chunk(audio, sample_rate, suffix)
        inference_lock = lock_factory()
        if not inference_lock.acquire(blocking=False):
            return None
        try:
            return self._embed_live_audio_chunk(audio, sample_rate, suffix)
        finally:
            inference_lock.release()

    def _verify_sentence_handoff_context(
        self,
        sentence: SentencePart,
        *,
        boundary_time: float,
        speaker_a: str,
        speaker_b: str,
        speaker_a_anchor: np.ndarray,
        speaker_b_anchor: np.ndarray,
        anchors: Mapping[str, np.ndarray],
        config: HandoffConfig,
    ) -> tuple[Any, int, str]:
        rejected = lambda: select_context_handoff(
            {},
            {},
            speaker_a,
            speaker_b,
            config,
        )
        max_verification_seconds = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "sentence_speaker_handoff_max_verification_seconds",
                    0.8,
                )
            ),
        )
        required_embedding_count = 2
        latency_ewma = getattr(
            self,
            "_live_speaker_embedding_latency_ewma",
            None,
        )
        if (
            max_verification_seconds <= 0.0
            or (
                latency_ewma is not None
                and float(latency_ewma) > 0.0
                and required_embedding_count * float(latency_ewma)
                > max_verification_seconds
            )
        ):
            return rejected(), 0, "context_embedding_budget_too_small"

        sentence_audio, sample_rate = self._audio_window_copy(
            sentence.start,
            sentence.end,
        )
        if sentence_audio.size <= 0 or sample_rate <= 0:
            return rejected(), 0, "sentence_audio_unavailable"

        deadline = time.monotonic() + max_verification_seconds
        live_reservation_marker = float(
            getattr(self, "_live_speaker_embedding_next_at", 0.0)
        )
        minimum_audio = max(
            0.5,
            float(getattr(self.args, "min_embed_seconds", 0.5)),
        )
        context_seconds = max(
            minimum_audio,
            float(
                getattr(
                    self.args,
                    "sentence_speaker_handoff_context_seconds",
                    1.0,
                )
            ),
        )
        min_voiced_seconds = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "sentence_speaker_handoff_min_context_voiced_seconds",
                    0.25,
                )
            ),
        )
        contexts = (
            (
                "left",
                max(float(sentence.start), float(boundary_time) - context_seconds),
                float(boundary_time),
            ),
            (
                "right",
                float(boundary_time),
                min(float(sentence.end), float(boundary_time) + context_seconds),
            ),
        )
        context_scores: dict[str, dict[str, float]] = {}
        embedded_count = 0
        for side, context_start, context_end in contexts:
            if (
                embedded_count
                and float(
                    getattr(self, "_live_speaker_embedding_next_at", 0.0)
                )
                > live_reservation_marker + 1e-6
            ):
                return (
                    rejected(),
                    embedded_count,
                    "live_probe_preempted_verification",
                )
            if time.monotonic() >= deadline:
                return (
                    rejected(),
                    embedded_count,
                    "verification_time_budget_exhausted",
                )
            raw_context_audio = self._sentence_handoff_context_audio(
                sentence_audio,
                int(sample_rate),
                sentence_start=sentence.start,
                context_start=context_start,
                context_end=context_end,
            )
            if raw_context_audio.size <= 0:
                return (
                    rejected(),
                    embedded_count,
                    "incomplete_context_audio",
                )
            _, _, _, voiced_seconds = rms_speech_features(
                raw_context_audio,
                int(sample_rate),
                frame_seconds=max(
                    0.01,
                    float(getattr(self.args, "vad_frame_seconds", 0.03)),
                ),
                threshold=max(
                    0.0,
                    float(getattr(self.args, "vad_speech_rms_threshold", 0.003)),
                ),
                min_speech_seconds=0.0,
            )
            if voiced_seconds < min_voiced_seconds:
                return (
                    rejected(),
                    embedded_count,
                    f"insufficient_{side}_context_speech",
                )
            prepared = pad_audio(
                trim_silence(raw_context_audio, int(sample_rate)),
                minimum_audio,
                int(sample_rate),
            )
            embedding = self._try_embed_sentence_handoff_live_context(
                prepared,
                int(sample_rate),
                f".handoff.{side}-context.wav",
            )
            if embedding is None:
                return (
                    rejected(),
                    embedded_count,
                    "live_provider_busy",
                )
            embedded_count += 1
            vector = normalize_embedding(embedding)
            if (
                vector.shape != speaker_a_anchor.shape
                or vector.shape != speaker_b_anchor.shape
            ):
                return (
                    rejected(),
                    embedded_count,
                    "context_embedding_shape_mismatch",
                )
            scores: dict[str, float] = {}
            for label, anchor in anchors.items():
                if vector.shape == anchor.shape:
                    scores[label] = float(np.dot(vector, anchor))
            context_scores[side] = scores
        if time.monotonic() > deadline:
            return (
                rejected(),
                embedded_count,
                "verification_time_budget_exhausted",
            )
        return (
            select_context_handoff(
                context_scores.get("left", {}),
                context_scores.get("right", {}),
                speaker_a,
                speaker_b,
                config,
            ),
            embedded_count,
            "",
        )

    def _sentence_handoff_run_id(self) -> str:
        run = getattr(self, "_active_run", None)
        return str(getattr(run, "run_id", "") or "")

    def _remember_sentence_handoff_hindsight_candidate(
        self,
        index: int,
        sentence: SentencePart,
        base_payload: Mapping[str, Any],
    ) -> None:
        """Remember one emitted, unsplit sentence for a final mature-profile retry."""

        if not self._sentence_speaker_handoff_enabled():
            return
        config = self._sentence_handoff_config()
        handoff = dict(getattr(sentence, "speaker_handoff", {}) or {})
        if (
            len(sentence.words) < 2 * config.min_words_per_segment
            or bool(handoff.get("detected"))
            or int(getattr(sentence, "semantic_sentence_part_count", 1) or 1) > 1
        ):
            return

        run_id = self._sentence_handoff_run_id()
        candidate = {
            "index": int(index),
            "run_id": run_id,
            "source_revision": str(
                base_payload.get("source_revision")
                or base_payload.get("source_text_hash")
                or ""
            ),
            "text": str(sentence.text),
            "start": float(sentence.start),
            "end": float(sentence.end),
            "sentence": copy.deepcopy(sentence),
        }
        lock = getattr(self, "_sentence_handoff_evidence_lock", None)
        if lock is None:
            return
        with lock:
            stored_run_id = str(
                getattr(self, "_sentence_handoff_hindsight_run_id", "") or ""
            )
            if stored_run_id != run_id:
                self._sentence_handoff_hindsight_candidates = {}
                self._sentence_handoff_hindsight_run_id = run_id
                self._sentence_handoff_hindsight_finalized_run_id = None
            candidates = getattr(
                self,
                "_sentence_handoff_hindsight_candidates",
                None,
            )
            if candidates is None:
                candidates = {}
                self._sentence_handoff_hindsight_candidates = candidates
            candidates[int(index)] = candidate

            cache_seconds = max(
                5.0,
                float(
                    getattr(
                        self.args,
                        "sentence_speaker_handoff_cache_seconds",
                        3600.0,
                    )
                ),
            )
            cutoff = float(sentence.end) - cache_seconds
            for stale_index in [
                candidate_index
                for candidate_index, item in candidates.items()
                if float(item.get("end") or 0.0) < cutoff
            ]:
                candidates.pop(stale_index, None)

    @staticmethod
    def _sentence_handoff_hindsight_record_rejection(
        candidate: Mapping[str, Any],
        record: Mapping[str, Any] | None,
    ) -> str:
        if not isinstance(record, Mapping):
            return "missing_emitted_record"
        if int(record.get("index", -1)) != int(candidate.get("index", -2)):
            return "stale_emitted_index"
        if isinstance(record.get("correction"), Mapping) and record.get("correction"):
            return "user_corrected_record"

        base_payload = record.get("base_payload")
        if not isinstance(base_payload, Mapping):
            return "missing_base_payload"
        current_revision = str(
            base_payload.get("source_revision")
            or base_payload.get("source_text_hash")
            or ""
        )
        if current_revision != str(candidate.get("source_revision") or ""):
            return "stale_source_revision"
        if str(base_payload.get("text") or "") != str(candidate.get("text") or ""):
            return "stale_sentence_text"
        try:
            start_matches = math.isclose(
                float(base_payload.get("start")),
                float(candidate.get("start")),
                rel_tol=0.0,
                abs_tol=1e-3,
            )
            end_matches = math.isclose(
                float(base_payload.get("end")),
                float(candidate.get("end")),
                rel_tol=0.0,
                abs_tol=1e-3,
            )
        except (TypeError, ValueError):
            return "invalid_sentence_timestamps"
        if not start_matches or not end_matches:
            return "stale_sentence_timestamps"

        handoff = base_payload.get("speaker_handoff")
        if isinstance(handoff, Mapping) and bool(handoff.get("detected")):
            return "already_split"
        try:
            part_count = int(base_payload.get("semantic_sentence_part_count", 1) or 1)
        except (TypeError, ValueError):
            part_count = 1
        if part_count > 1:
            return "already_split"
        return ""

    def _embed_sentence_handoff_hindsight_part(
        self,
        sentence: SentencePart,
        *,
        side: str,
    ) -> np.ndarray:
        audio, sample_rate = self._audio_window_copy(sentence.start, sentence.end)
        if audio.size <= 0 or int(sample_rate) <= 0:
            raise ValueError(f"{side} child audio is unavailable")
        prepared = pad_audio(
            trim_silence(audio, int(sample_rate)),
            max(0.5, float(getattr(self.args, "min_embed_seconds", 0.5))),
            int(sample_rate),
        )
        return normalize_embedding(
            self._embed_audio_chunk(
                prepared,
                int(sample_rate),
                f".handoff-hindsight-{side}.wav",
            )
        )

    def _sentence_handoff_hindsight_child_record(
        self,
        *,
        index: int,
        parent_index: int,
        sentence: SentencePart,
        embedding: np.ndarray,
        decision: Any,
        window_left: float,
        window_right: float,
    ) -> dict[str, Any]:
        handoff = dict(getattr(sentence, "speaker_handoff", {}) or {})
        handoff.update(
            {
                "hindsight": True,
                "replaces_sentence_index": int(parent_index),
                "emitted_sentence_index": int(index),
            }
        )
        sentence = replace(sentence, speaker_handoff=handoff)
        base_payload = self._base_payload_from_sentence_part(
            int(index),
            sentence,
            float(window_left),
            float(window_right),
        )
        return {
            "index": int(index),
            "base_payload": base_payload,
            "embedding": np.asarray(embedding, dtype=np.float32).copy(),
            "duration_seconds": max(
                0.0,
                float(sentence.end) - float(sentence.start),
            ),
            "assigned_speaker": decision.assigned_speaker,
            "created_speaker": False,
            "probabilities": dict(decision.probabilities or {}),
            "similarities": dict(decision.similarities or {}),
            "unknown_probability": decision.unknown_probability,
            "top_similarity": decision.top_similarity,
            "margin": decision.margin,
            "quality": decision.quality,
            "assignment_source": "sentence_handoff_hindsight",
            "automatic_assigned_speaker": decision.assigned_speaker,
            "automatic_assignment_source": "sentence_handoff_hindsight",
        }

    @staticmethod
    def _sentence_handoff_record_chronology(
        item: tuple[int, Mapping[str, Any]],
    ) -> tuple[float, float, int]:
        index, record = item
        base_payload = record.get("base_payload")
        if not isinstance(base_payload, Mapping):
            return (float("inf"), float("inf"), int(index))
        try:
            start = float(base_payload.get("start"))
        except (TypeError, ValueError):
            start = float("inf")
        try:
            end = float(base_payload.get("end"))
        except (TypeError, ValueError):
            end = float("inf")
        return (start, end, int(index))

    def _build_sentence_handoff_rebuilt_memory(
        self,
        records: Mapping[int, Mapping[str, Any]],
    ) -> tuple[Any | None, str]:
        """Build a replacement memory without mutating the authoritative one."""

        current_profiles = self.memory.export_profiles()
        current_labels = {
            str(profile.get("label") or "")
            for profile in current_profiles
            if str(profile.get("label") or "")
        }
        locked_labels = {
            str(profile.get("label") or "")
            for profile in current_profiles
            if bool(profile.get("locked"))
        }
        seed_profiles = [
            copy.deepcopy(profile)
            for profile in list(getattr(self, "_seed_profiles", []) or [])
            if isinstance(profile, Mapping)
        ]
        rebuilt = self._new_memory()
        try:
            if seed_profiles:
                rebuilt.replace_profiles(seed_profiles)
            for _index, record in sorted(
                records.items(),
                key=self._sentence_handoff_record_chronology,
            ):
                speaker = str(record.get("assigned_speaker") or "")
                if not speaker:
                    continue
                if speaker not in current_labels:
                    return None, "record_references_missing_profile"
                raw_embedding = record.get("embedding")
                if raw_embedding is None:
                    return None, "assigned_record_missing_embedding"
                embedding = normalize_embedding(raw_embedding)
                rebuilt.upsert_profile(
                    speaker,
                    embedding,
                    duration_seconds=max(
                        0.0,
                        float(record.get("duration_seconds") or 0.0),
                    ),
                    locked=speaker in locked_labels,
                )
            rebuilt_profiles = rebuilt.export_profiles()
        except Exception as exc:
            return None, f"memory_rebuild_failed:{type(exc).__name__}"

        rebuilt_labels = {
            str(profile.get("label") or "")
            for profile in rebuilt_profiles
            if str(profile.get("label") or "")
        }
        if rebuilt_labels != current_labels:
            return None, "memory_rebuild_profile_set_changed"
        return rebuilt, ""

    def _emit_sentence_handoff_hindsight_rejection(
        self,
        candidate: Mapping[str, Any],
        reason: str,
    ) -> None:
        self._emit_sentence_handoff_internal(
            "sentence_speaker_handoff_hindsight",
            {
                "sentence_index": int(candidate.get("index", -1)),
                "sentence_start": round(float(candidate.get("start") or 0.0), 6),
                "sentence_end": round(float(candidate.get("end") or 0.0), 6),
                "accepted": False,
                "reason": str(reason),
            },
        )

    def _finalize_sentence_handoff_hindsight(self) -> int:
        """Retry eligible emitted sentences once, using final mature profiles."""

        if not self._sentence_speaker_handoff_enabled():
            return 0
        run_id = self._sentence_handoff_run_id()
        run_token = run_id or "<manual>"
        lock = getattr(self, "_sentence_handoff_evidence_lock", None)
        if lock is None:
            return 0
        with lock:
            if (
                getattr(
                    self,
                    "_sentence_handoff_hindsight_finalized_run_id",
                    None,
                )
                == run_token
            ):
                return 0
            self._sentence_handoff_hindsight_finalized_run_id = run_token
            stored_run_id = str(
                getattr(self, "_sentence_handoff_hindsight_run_id", "") or ""
            )
            if stored_run_id != run_id:
                return 0
            candidates = [
                copy.deepcopy(item)
                for item in getattr(
                    self,
                    "_sentence_handoff_hindsight_candidates",
                    {},
                ).values()
            ]
        if not candidates:
            return 0

        max_hindsight_seconds = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "sentence_speaker_handoff_max_hindsight_seconds",
                    5.0,
                )
            ),
        )
        if max_hindsight_seconds <= 0.0:
            return 0
        hindsight_deadline = time.monotonic() + max_hindsight_seconds
        proposals: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                float(item.get("start") or 0.0),
                int(item.get("index") or 0),
            ),
        ):
            if time.monotonic() >= hindsight_deadline:
                self._emit_sentence_handoff_internal(
                    "sentence_speaker_handoff_hindsight",
                    {
                        "accepted": False,
                        "reason": "hindsight_time_budget_exhausted",
                        "max_hindsight_seconds": max_hindsight_seconds,
                    },
                )
                break
            with self._sentence_refinement_lock:
                record = copy.deepcopy(
                    self._sentence_refinement_records.get(
                        int(candidate.get("index", -1))
                    )
                )
            rejection = self._sentence_handoff_hindsight_record_rejection(
                candidate,
                record,
            )
            if rejection:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    rejection,
                )
                continue

            sentence = candidate.get("sentence")
            if not isinstance(sentence, SentencePart):
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    "missing_sentence_part",
                )
                continue
            try:
                split = self._split_one_completed_sentence_handoff(
                    sentence,
                    announce=False,
                )
            except Exception as exc:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    f"retry_failed:{type(exc).__name__}",
                )
                continue
            if len(split) != 2:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    "no_verified_handoff",
                )
                continue
            left, right = split
            left_speaker = str(
                (getattr(left, "speaker_handoff", {}) or {}).get("speaker_a")
                or ""
            )
            right_speaker = str(
                (getattr(right, "speaker_handoff", {}) or {}).get("speaker_b")
                or ""
            )
            if not left_speaker or not right_speaker or left_speaker == right_speaker:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    "invalid_split_speakers",
                )
                continue

            if time.monotonic() >= hindsight_deadline:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    "hindsight_time_budget_exhausted",
                )
                break
            try:
                left_embedding = self._embed_sentence_handoff_hindsight_part(
                    left,
                    side="left",
                )
                right_embedding = self._embed_sentence_handoff_hindsight_part(
                    right,
                    side="right",
                )
                left_decision = self.memory.score_existing(
                    left_embedding,
                    max(0.0, float(left.end) - float(left.start)),
                )
                right_decision = self.memory.score_existing(
                    right_embedding,
                    max(0.0, float(right.end) - float(right.start)),
                )
            except Exception as exc:
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    f"child_embedding_failed:{type(exc).__name__}",
                )
                continue
            if (
                left_decision.assigned_speaker != left_speaker
                or right_decision.assigned_speaker != right_speaker
            ):
                self._emit_sentence_handoff_hindsight_rejection(
                    candidate,
                    "final_provider_segment_disagreement",
                )
                continue
            proposals.append(
                {
                    "candidate": candidate,
                    "left": left,
                    "right": right,
                    "left_embedding": left_embedding,
                    "right_embedding": right_embedding,
                    "left_decision": left_decision,
                    "right_decision": right_decision,
                    "previous_speaker": record.get("assigned_speaker"),
                }
            )

        if not proposals:
            return 0

        session_state = getattr(self, "_session_state", None)
        transaction = (
            session_state.transaction(mutate=True)
            if session_state is not None
            else nullcontext()
        )
        committed: list[dict[str, Any]] = []
        memory_rebuilt = False
        memory_rebuild_reason = ""
        with transaction, self._sentence_refinement_lock:
            prospective = dict(self._sentence_refinement_records)
            next_index = max(prospective, default=-1) + 1
            for proposal in proposals:
                candidate = proposal["candidate"]
                parent_index = int(candidate["index"])
                current_record = prospective.get(parent_index)
                rejection = self._sentence_handoff_hindsight_record_rejection(
                    candidate,
                    current_record,
                )
                if rejection:
                    self._emit_sentence_handoff_hindsight_rejection(
                        candidate,
                        rejection,
                    )
                    continue
                base_payload = dict(current_record.get("base_payload") or {})
                window_left = float(base_payload.get("window_left") or 0.0)
                window_right = float(
                    base_payload.get("window_right")
                    or base_payload.get("end")
                    or 0.0
                )
                child_index = next_index
                next_index += 1
                left_record = self._sentence_handoff_hindsight_child_record(
                    index=parent_index,
                    parent_index=parent_index,
                    sentence=proposal["left"],
                    embedding=proposal["left_embedding"],
                    decision=proposal["left_decision"],
                    window_left=window_left,
                    window_right=window_right,
                )
                right_record = self._sentence_handoff_hindsight_child_record(
                    index=child_index,
                    parent_index=parent_index,
                    sentence=proposal["right"],
                    embedding=proposal["right_embedding"],
                    decision=proposal["right_decision"],
                    window_left=window_left,
                    window_right=window_right,
                )
                prospective[parent_index] = left_record
                prospective[child_index] = right_record
                committed.append(
                    {
                        **proposal,
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "left_record": left_record,
                        "right_record": right_record,
                    }
                )

            if not committed:
                return 0
            rebuilt_memory, memory_rebuild_reason = (
                self._build_sentence_handoff_rebuilt_memory(prospective)
            )
            if rebuilt_memory is None:
                for item in committed:
                    self._emit_sentence_handoff_hindsight_rejection(
                        item["candidate"],
                        memory_rebuild_reason or "memory_rebuild_unavailable",
                    )
                return 0
            memory_rebuilt = True
            for item in committed:
                for child_record in (
                    item["left_record"],
                    item["right_record"],
                ):
                    handoff = dict(
                        child_record["base_payload"].get("speaker_handoff")
                        or {}
                    )
                    handoff["memory_rebuilt"] = memory_rebuilt
                    if memory_rebuild_reason:
                        handoff["memory_rebuild_reason"] = memory_rebuild_reason
                    child_record["base_payload"]["speaker_handoff"] = handoff
            speaker_last_media_end: dict[str, float] = {}
            for record in prospective.values():
                speaker = str(record.get("assigned_speaker") or "")
                if not speaker:
                    continue
                try:
                    end = float((record.get("base_payload") or {}).get("end"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(end):
                    speaker_last_media_end[speaker] = max(
                        end,
                        float(speaker_last_media_end.get(speaker, 0.0)),
                    )
            old_memory = self.memory
            shared_live_memory = getattr(self, "live_memory", None) is old_memory
            # All potentially failing rebuild work happened on ``rebuilt_memory``.
            # Publishing is now a set of non-throwing reference assignments, so
            # records and centroids cannot diverge through a partial mutation of
            # the old SpeakerMemory instance.
            self.memory = rebuilt_memory
            if shared_live_memory:
                self.live_memory = rebuilt_memory
            self._sentence_refinement_records = prospective
            self._final_sentence_count = int(
                getattr(self, "_final_sentence_count", 0)
            ) + len(committed)
            self._speaker_last_media_end = speaker_last_media_end

        for item in committed:
            parent_index = int(item["parent_index"])
            self._remove_unknown_sentence(parent_index)
            left_payload = self._record_to_sentence_payload(item["left_record"])
            left_payload.update(
                {
                    "revision": True,
                    "revision_from": item.get("previous_speaker"),
                    "revision_to": item["left_record"].get("assigned_speaker"),
                }
            )
            self._emit_transcript_sentence(left_payload)
            self._emit_transcript_sentence(
                self._record_to_sentence_payload(item["right_record"])
            )
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_hindsight",
                {
                    "sentence_index": parent_index,
                    "child_sentence_index": int(item["child_index"]),
                    "sentence_start": round(
                        float(item["candidate"].get("start") or 0.0),
                        6,
                    ),
                    "sentence_end": round(
                        float(item["candidate"].get("end") or 0.0),
                        6,
                    ),
                    "accepted": True,
                    "memory_rebuilt": memory_rebuilt,
                    "memory_rebuild_reason": memory_rebuild_reason,
                },
            )
        self.emit_speaker_state()
        self.bus.emit(
            "status",
            {
                "message": (
                    "End-of-run hindsight split "
                    f"{len(committed)} completed sentence"
                    f"{'' if len(committed) == 1 else 's'} after final speaker "
                    "profiles became available."
                )
            },
        )
        return len(committed)

    def _split_one_completed_sentence_handoff(
        self,
        sentence: SentencePart,
        *,
        announce: bool = True,
    ) -> tuple[SentencePart, ...]:
        config = self._sentence_handoff_config()
        if (
            len(sentence.words) < 2 * config.min_words_per_segment
            or bool((getattr(sentence, "speaker_handoff", {}) or {}).get("detected"))
        ):
            return (sentence,)

        cached = self._snapshot_sentence_handoff_evidence(sentence)
        if len(cached) < 2 * config.min_probes_per_side:
            return (sentence,)
        anchors = self._current_sentence_handoff_profile_anchors()
        if len(anchors) < 2:
            return (sentence,)
        provider = str(self._current_live_embedding_provider() or "")
        rescored = self._rescore_sentence_handoff_evidence(
            cached,
            anchors,
            provider=provider,
        )
        nomination = nominate_stable_handoff(
            rescored,
            sentence.start,
            sentence.end,
            word_times=sentence.words,
            profile_anchors=anchors,
            config=config,
        )
        if (
            nomination is None
            or nomination.coarse_boundary_time is None
            or nomination.suggested_word_cut_index is None
            or nomination.suggested_word_boundary_time is None
            or nomination.speaker_a not in anchors
            or nomination.speaker_b not in anchors
        ):
            return (sentence,)
        max_boundary_snap = max(
            0.05,
            min(
                0.50,
                float(
                    getattr(
                        self.args,
                        "sentence_speaker_handoff_max_word_snap_seconds",
                        0.25,
                    )
                ),
            ),
        )
        boundary_snap = abs(
            nomination.coarse_boundary_time
            - nomination.suggested_word_boundary_time
        )
        if boundary_snap > max_boundary_snap:
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_verification",
                {
                    "sentence_start": round(float(sentence.start), 6),
                    "sentence_end": round(float(sentence.end), 6),
                    "speaker_a": nomination.speaker_a,
                    "speaker_b": nomination.speaker_b,
                    "accepted": False,
                    "reason": "acoustic_boundary_too_far_from_word_gap",
                    "coarse_boundary_time": round(
                        float(nomination.coarse_boundary_time),
                        6,
                    ),
                    "suggested_word_boundary_time": round(
                        float(nomination.suggested_word_boundary_time),
                        6,
                    ),
                    "boundary_snap_seconds": round(float(boundary_snap), 6),
                },
            )
            return (sentence,)

        selection, embedded_context_count, precheck_reason = (
            self._verify_sentence_handoff_context(
                sentence,
                boundary_time=nomination.suggested_word_boundary_time,
                speaker_a=nomination.speaker_a,
                speaker_b=nomination.speaker_b,
                speaker_a_anchor=anchors[nomination.speaker_a],
                speaker_b_anchor=anchors[nomination.speaker_b],
                anchors=anchors,
                config=config,
            )
        )
        diagnostic = {
            "sentence_start": round(float(sentence.start), 6),
            "sentence_end": round(float(sentence.end), 6),
            "speaker_a": nomination.speaker_a,
            "speaker_b": nomination.speaker_b,
            "live_probe_count": len(nomination.evidence),
            "coarse_boundary_time": (
                None
                if nomination.coarse_boundary_time is None
                else round(float(nomination.coarse_boundary_time), 6)
            ),
            "suggested_word_cut_index": nomination.suggested_word_cut_index,
            "suggested_word_boundary_time": round(
                float(nomination.suggested_word_boundary_time),
                6,
            ),
            "boundary_snap_seconds": round(float(boundary_snap), 6),
            "embedded_context_count": embedded_context_count,
            "accepted": bool(selection.accepted),
            "reason": precheck_reason or selection.reason,
            "left_context_similarity": selection.left_expected_similarity,
            "left_context_pair_margin": selection.left_pair_margin,
            "left_context_runner_up_margin": selection.left_runner_up_margin,
            "right_context_similarity": selection.right_expected_similarity,
            "right_context_pair_margin": selection.right_pair_margin,
            "right_context_runner_up_margin": selection.right_runner_up_margin,
            "context_separation": selection.separation,
        }
        if not selection.accepted:
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_verification",
                diagnostic,
            )
            return (sentence,)

        selected_cut = int(nomination.suggested_word_cut_index)
        selected_left_end = float(sentence.words[selected_cut - 1].get("end", 0.0))
        selected_right_start = float(
            sentence.words[selected_cut].get("start", selected_left_end)
        )
        selected_boundary_time = 0.5 * (
            selected_left_end + selected_right_start
        )
        selected_boundary_snap = abs(
            nomination.coarse_boundary_time - selected_boundary_time
        )
        if selected_boundary_snap > max_boundary_snap:
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_verification",
                {
                    **diagnostic,
                    "accepted": False,
                    "reason": "selected_cut_too_far_from_acoustic_boundary",
                    "selected_word_cut_index": selected_cut,
                    "selected_word_boundary_time": round(
                        selected_boundary_time,
                        6,
                    ),
                    "selected_boundary_snap_seconds": round(
                        selected_boundary_snap,
                        6,
                    ),
                },
            )
            return (sentence,)
        diagnostic.update(
            {
                "selected_word_cut_index": selected_cut,
                "selected_word_boundary_time": round(
                    selected_boundary_time,
                    6,
                ),
                "selected_boundary_snap_seconds": round(
                    selected_boundary_snap,
                    6,
                ),
            }
        )
        self._emit_sentence_handoff_internal(
            "sentence_speaker_handoff_verification",
            diagnostic,
        )

        semantic_digest = hashlib.sha256(
            (
                f"{sentence.start:.6f}|{sentence.end:.6f}|{sentence.text}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        split = split_sentence_part(
            sentence,
            selected_cut,
            boundary_time=selected_boundary_time,
            speaker_a=nomination.speaker_a,
            speaker_b=nomination.speaker_b,
            semantic_group_id=f"sentence:{semantic_digest}",
        )
        metadata = {
            "algorithm": "cached_live_context_one_cut_v2",
            "coarse_boundary_time": nomination.coarse_boundary_time,
            "live_probe_count": len(nomination.evidence),
            "verified_context_embedding_count": embedded_context_count,
        }
        left_handoff = dict(split.left.speaker_handoff)
        left_handoff.update(metadata)
        right_handoff = dict(split.right.speaker_handoff)
        right_handoff.update(metadata)
        left = replace(split.left, speaker_handoff=left_handoff)
        right = replace(split.right, speaker_handoff=right_handoff)
        if announce:
            self.bus.emit(
                "status",
                {
                    "message": (
                        "Detected a stable within-sentence speaker handoff "
                        f"{nomination.speaker_a}\u2192{nomination.speaker_b} at "
                        f"{split.boundary_time:.2f}s."
                    )
                },
            )
        return left, right

    def _split_completed_sentence_handoffs(
        self,
        parts: Sequence[SentencePart],
    ) -> list[SentencePart]:
        """Return ordinary sentence parts; detector failures are conservative no-ops."""

        if (
            not self._sentence_speaker_handoff_enabled()
            or not bool(
                getattr(
                    self.args,
                    "sentence_speaker_handoff_immediate",
                    False,
                )
            )
        ):
            return list(parts)
        split_parts: list[SentencePart] = []
        for sentence in parts:
            try:
                split_parts.extend(self._split_one_completed_sentence_handoff(sentence))
            except Exception as exc:
                self._emit_sentence_handoff_internal(
                    "sentence_speaker_handoff_error",
                    {
                        "sentence_start": round(float(sentence.start), 6),
                        "sentence_end": round(float(sentence.end), 6),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                split_parts.append(sentence)
        return split_parts

    def _emit_sentence(
        self,
        index: int,
        sentence: SentencePart,
        window_left: float,
        window_right: float,
    ) -> None:
        """Attach a stable emitted index before normal sentence processing."""

        try:
            base_payload = self._base_payload_from_sentence_part(
                int(index),
                sentence,
                float(window_left),
                float(window_right),
            )
            self._remember_sentence_handoff_hindsight_candidate(
                int(index),
                sentence,
                base_payload,
            )
        except Exception as exc:
            self._emit_sentence_handoff_internal(
                "sentence_speaker_handoff_hindsight",
                {
                    "sentence_index": int(index),
                    "accepted": False,
                    "reason": f"candidate_registration_failed:{type(exc).__name__}",
                },
            )
        return super()._emit_sentence(index, sentence, window_left, window_right)
