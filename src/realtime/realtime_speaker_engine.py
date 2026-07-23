"""Speaker embedding and assignment engine for realtime diarization."""

from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from common.audio_utils import SAMPLE_RATE, clamp01, normalize_vector, pad_audio, trim_silence, write_wav
from embeddings.embedding_providers import EmbeddingSubprocessClient
from realtime.realtime_cli import RealtimeConfig
from speakers.realtime_speaker_memory import SpeakerDecision, SpeakerMemory

@dataclass
class SentenceJob:
    session_id: str
    index: int
    text: str
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    split_reason: str | None = None
    part_index: int | None = None
    part_count: int | None = None
    video_start_seconds: float | None = None
    video_end_seconds: float | None = None


@dataclass
class ProcessedSentenceRecord:
    session_id: str
    index: int
    text: str
    duration_seconds: float
    embedding: np.ndarray
    decision: SpeakerDecision
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    video_start_seconds: float | None = None
    video_end_seconds: float | None = None
    split_reason: str | None = None
    part_index: int | None = None
    part_count: int | None = None


def is_reassignment_candidate(
    args: RealtimeConfig,
    decision: SpeakerDecision,
    duration_seconds: float,
) -> bool:
    if not getattr(args, "reassign_uncertain_sentences", True):
        return False
    if decision.assignment_source == "context":
        return True
    if decision.created_speaker:
        return False
    if duration_seconds > args.reassign_max_seconds:
        return False
    if decision.assigned_speaker is None:
        return True
    return decision.unknown_probability >= args.reassign_unknown_min


def is_reassignment_accepted(
    args: RealtimeConfig,
    decision: SpeakerDecision,
    duration_seconds: float,
) -> bool:
    if not decision.assigned_speaker:
        return False
    similarity_ok = (
        decision.top_similarity is not None
        and decision.top_similarity >= args.reassign_min_similarity
    )
    short_similarity_ok = (
        duration_seconds <= args.reassign_short_max_seconds
        and decision.top_similarity is not None
        and decision.top_similarity >= args.reassign_short_min_similarity
        and decision.margin is not None
        and decision.margin >= args.reassign_short_min_margin
    )
    unknown_ok = decision.unknown_probability <= args.reassign_unknown_max
    return similarity_ok or short_similarity_ok or unknown_ok


def speaker_probability_key(speaker_label: str | None) -> str | None:
    if not speaker_label:
        return None
    match = re.fullmatch(r"S(\d+)", str(speaker_label).strip())
    if not match:
        return None
    return f"speaker{match.group(1)}"


def context_adjusted_decision(
    decision: SpeakerDecision,
    assigned_speaker: str,
    confidence: float,
) -> SpeakerDecision:
    confidence = clamp01(confidence)
    speaker_key = speaker_probability_key(assigned_speaker)
    probabilities = {"unknown": round(float(1.0 - confidence), 4)}
    if speaker_key is not None:
        probabilities[speaker_key] = round(float(confidence), 4)
    return replace(
        decision,
        assigned_speaker=assigned_speaker,
        created_speaker=False,
        probabilities=probabilities,
        unknown_probability=round(float(1.0 - confidence), 4),
        assignment_source="context",
    )


def is_context_assignment_candidate(
    args: RealtimeConfig,
    decision: SpeakerDecision,
    duration_seconds: float,
) -> bool:
    if not getattr(args, "context_assign_short_fragments", True):
        return False
    if duration_seconds > args.context_assign_max_seconds:
        return False
    if decision.assignment_source == "context":
        return True
    if not decision.assigned_speaker:
        return True
    return decision.unknown_probability >= args.context_assign_candidate_unknown_min


def is_context_anchor(
    args: RealtimeConfig,
    record: ProcessedSentenceRecord,
) -> bool:
    decision = record.decision
    if not decision.assigned_speaker:
        return False
    if decision.assignment_source == "context":
        return False
    if record.duration_seconds < args.context_assign_stable_min_seconds:
        return False
    return decision.unknown_probability <= args.context_assign_stable_unknown_max


class RealtimeSpeakerEngine:
    def __init__(self, args: RealtimeConfig, bus: "EventBus") -> None:
        self.args = args
        self.bus = bus
        self.memory = self._new_memory()
        self.jobs: "queue.Queue[SentenceJob]" = queue.Queue()
        self.client = EmbeddingSubprocessClient(
            python=Path(args.embedding_python),
            provider=args.embedding_provider,
            device=args.embedding_device,
            response_timeout_seconds=float(args.embedding_helper_response_timeout_seconds),
        )
        self._records_by_session: dict[str, dict[int, ProcessedSentenceRecord]] = {}
        self._records_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._session_id: str | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="RealtimeSpeakerEmbeddingWorker",
            daemon=True,
        )
        self._worker.start()

    def start_session(self, session_id: str) -> None:
        self._session_id = session_id
        self.memory = self._new_memory()
        with self._records_lock:
            self._records_by_session = {session_id: {}}
        self._status(session_id, f"Embedding provider: {self.args.embedding_provider}.")
        self._status(session_id, f"Embedding device: {self.args.embedding_device}.")
        self._status(session_id, f"Embedding helper Python: {self.args.embedding_python}.")
        self._status(
            session_id,
            f"RealtimeSTT post-speech silence: {self.args.post_speech_silence_duration:.2f}s.",
        )

    def submit(
        self,
        session_id: str,
        index: int,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        source_start_seconds: float | None = None,
        source_end_seconds: float | None = None,
        split_reason: str | None = None,
        part_index: int | None = None,
        part_count: int | None = None,
        video_start_seconds: float | None = None,
        video_end_seconds: float | None = None,
    ) -> None:
        duration_seconds = float(len(audio)) / float(sample_rate or SAMPLE_RATE)
        pending_payload = self._sentence_payload(
            session_id=session_id,
            index=index,
            text=text,
            duration_seconds=duration_seconds,
            source_start_seconds=source_start_seconds,
            source_end_seconds=source_end_seconds,
            split_reason=split_reason,
            part_index=part_index,
            part_count=part_count,
            video_start_seconds=video_start_seconds,
            video_end_seconds=video_end_seconds,
            pending=True,
            decision=None,
            error=None,
        )
        self.bus.emit("sentence", pending_payload)
        self.jobs.put(
            SentenceJob(
                session_id=session_id,
                index=index,
                text=text,
                audio=audio,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                source_start_seconds=source_start_seconds,
                source_end_seconds=source_end_seconds,
                split_reason=split_reason,
                part_index=part_index,
                part_count=part_count,
                video_start_seconds=video_start_seconds,
                video_end_seconds=video_end_seconds,
            )
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        self.client.shutdown()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._worker.is_alive() and self._session_id:
            self._status(self._session_id, "Speaker embedding worker did not stop before timeout.")

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
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if job.session_id != self._session_id:
                    continue
                audio = trim_silence(job.audio, job.sample_rate)
                audio = pad_audio(audio, self.args.min_embed_seconds, job.sample_rate)
                wav_path = self._write_temp_wav(job, audio)
                try:
                    embedding = self.client.embed_wav(wav_path)
                finally:
                    if not self.args.keep_segment_audio:
                        try:
                            wav_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                decision = self.memory.classify(embedding, job.duration_seconds)
                self._store_record(job, embedding, decision)
                self.bus.emit(
                    "sentence",
                    self._sentence_payload(
                        session_id=job.session_id,
                        index=job.index,
                        text=job.text,
                        duration_seconds=job.duration_seconds,
                        source_start_seconds=job.source_start_seconds,
                        source_end_seconds=job.source_end_seconds,
                        split_reason=job.split_reason,
                        part_index=job.part_index,
                        part_count=job.part_count,
                        video_start_seconds=job.video_start_seconds,
                        video_end_seconds=job.video_end_seconds,
                        pending=False,
                        decision=decision,
                        error=None,
                    ),
                )
                self._revisit_uncertain_records(job.session_id)
            except Exception as exc:
                self.bus.emit(
                    "sentence",
                    self._sentence_payload(
                        session_id=job.session_id,
                        index=job.index,
                        text=job.text,
                        duration_seconds=job.duration_seconds,
                        source_start_seconds=job.source_start_seconds,
                        source_end_seconds=job.source_end_seconds,
                        split_reason=job.split_reason,
                        part_index=job.part_index,
                        part_count=job.part_count,
                        video_start_seconds=job.video_start_seconds,
                        video_end_seconds=job.video_end_seconds,
                        pending=False,
                        decision=None,
                        error=str(exc),
                    ),
                )
                self._error(job.session_id, f"Embedding failed for sentence {job.index}: {exc}")
            finally:
                self.jobs.task_done()

    def _store_record(
        self,
        job: SentenceJob,
        embedding: np.ndarray,
        decision: SpeakerDecision,
    ) -> None:
        with self._records_lock:
            records = self._records_by_session.setdefault(job.session_id, {})
            records[job.index] = ProcessedSentenceRecord(
                session_id=job.session_id,
                index=job.index,
                text=job.text,
                duration_seconds=job.duration_seconds,
                embedding=embedding.astype(np.float32, copy=True),
                decision=decision,
                source_start_seconds=job.source_start_seconds,
                source_end_seconds=job.source_end_seconds,
                video_start_seconds=job.video_start_seconds,
                video_end_seconds=job.video_end_seconds,
                split_reason=job.split_reason,
                part_index=job.part_index,
                part_count=job.part_count,
            )

    def _revisit_uncertain_records(self, session_id: str) -> None:
        with self._records_lock:
            records = list(self._records_by_session.get(session_id, {}).values())

        records = sorted(records, key=lambda item: item.index)
        for record in records:
            old_decision = record.decision
            if old_decision.assigned_speaker:
                continue
            if not is_reassignment_candidate(
                self.args,
                old_decision,
                record.duration_seconds,
            ):
                continue
            candidate = self.memory.score_existing(
                record.embedding,
                record.duration_seconds,
                force_assignment=True,
            )
            if not is_reassignment_accepted(self.args, candidate, record.duration_seconds):
                continue
            if old_decision.assigned_speaker == candidate.assigned_speaker:
                continue

            self._apply_record_revision(session_id, record, old_decision, candidate)

    def _apply_record_revision(
        self,
        session_id: str,
        record: ProcessedSentenceRecord,
        old_decision: SpeakerDecision,
        candidate: SpeakerDecision,
    ) -> None:
        with self._records_lock:
            current = self._records_by_session.get(session_id, {}).get(record.index)
            if current is None or current.decision is not old_decision:
                return
            current.decision = candidate
            record.decision = candidate
        self.bus.emit(
            "sentence",
            self._sentence_payload(
                session_id=session_id,
                index=record.index,
                text=record.text,
                duration_seconds=record.duration_seconds,
                source_start_seconds=record.source_start_seconds,
                source_end_seconds=record.source_end_seconds,
                split_reason=record.split_reason or "revision",
                part_index=record.part_index,
                part_count=record.part_count,
                video_start_seconds=record.video_start_seconds,
                video_end_seconds=record.video_end_seconds,
                pending=False,
                decision=candidate,
                error=None,
                revision=True,
            ),
        )

    def _context_assignment_candidate(
        self,
        record: ProcessedSentenceRecord,
        records: list[ProcessedSentenceRecord],
    ) -> SpeakerDecision | None:
        try:
            current_position = records.index(record)
        except ValueError:
            return None

        previous_speaker = None
        next_speaker = None
        window = max(1, int(self.args.context_assign_window))
        for position in range(current_position - 1, max(-1, current_position - window - 1), -1):
            candidate = records[position]
            if is_context_anchor(self.args, candidate):
                previous_speaker = candidate.decision.assigned_speaker
                break
        for position in range(current_position + 1, min(len(records), current_position + window + 1)):
            candidate = records[position]
            if is_context_anchor(self.args, candidate):
                next_speaker = candidate.decision.assigned_speaker
                break

        embedding_decision = self.memory.score_existing(
            record.embedding,
            record.duration_seconds,
            force_assignment=False,
        )
        chosen_speaker = None
        confidence = 0.0
        if previous_speaker and next_speaker and previous_speaker == next_speaker:
            chosen_speaker = previous_speaker
            confidence = self.args.context_assign_same_speaker_confidence
        elif previous_speaker and next_speaker:
            chosen_speaker = self._choose_disagreeing_context_anchor(
                embedding_decision,
                previous_speaker,
                next_speaker,
            )
            if chosen_speaker:
                confidence = self.args.context_assign_disagree_confidence
        elif self.args.context_assign_one_sided and previous_speaker:
            if self._one_sided_context_allowed(embedding_decision, previous_speaker):
                chosen_speaker = previous_speaker
                confidence = self.args.context_assign_one_sided_confidence
        elif self.args.context_assign_one_sided and next_speaker:
            if self._one_sided_context_allowed(embedding_decision, next_speaker):
                chosen_speaker = next_speaker
                confidence = max(0.0, self.args.context_assign_one_sided_confidence - 0.04)

        if not chosen_speaker:
            if record.decision.assignment_source == "context":
                return embedding_decision
            return None
        return context_adjusted_decision(record.decision, chosen_speaker, confidence)

    def _choose_disagreeing_context_anchor(
        self,
        embedding_decision: SpeakerDecision,
        previous_speaker: str,
        next_speaker: str,
    ) -> str | None:
        similarities = embedding_decision.similarities
        previous_similarity = float(similarities.get(previous_speaker, -1.0))
        next_similarity = float(similarities.get(next_speaker, -1.0))
        best_similarity = max(previous_similarity, next_similarity)
        if best_similarity < self.args.context_assign_disagree_min_similarity:
            return None
        if abs(previous_similarity - next_similarity) < self.args.context_assign_disagree_margin:
            return None
        return previous_speaker if previous_similarity > next_similarity else next_speaker

    def _one_sided_context_allowed(
        self,
        embedding_decision: SpeakerDecision,
        anchor_speaker: str,
    ) -> bool:
        similarities = embedding_decision.similarities
        anchor_similarity = float(similarities.get(anchor_speaker, -1.0))
        other_similarities = [
            float(value)
            for speaker, value in similarities.items()
            if speaker != anchor_speaker
        ]
        if not other_similarities:
            return True
        return max(other_similarities) - anchor_similarity <= self.args.context_assign_one_sided_block_margin

    def _write_temp_wav(self, job: SentenceJob, audio: np.ndarray) -> Path:
        directory = Path(self.args.segment_audio_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_index = f"{job.index:05d}"
        if self.args.keep_segment_audio:
            path = directory / f"{safe_index}.wav"
        else:
            handle = tempfile.NamedTemporaryFile(
                suffix=f".{safe_index}.wav",
                prefix="realtime-speaker-",
                dir=str(directory),
                delete=False,
            )
            handle.close()
            path = Path(handle.name)
        write_wav(path, audio, job.sample_rate)
        return path

    @staticmethod
    def _sentence_payload(
        session_id: str,
        index: int,
        text: str,
        duration_seconds: float,
        source_start_seconds: float | None,
        source_end_seconds: float | None,
        split_reason: str | None,
        part_index: int | None,
        part_count: int | None,
        video_start_seconds: float | None,
        video_end_seconds: float | None,
        pending: bool,
        decision: SpeakerDecision | None,
        error: str | None,
        revision: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "index": index,
            "text": text,
            "duration_seconds": round(float(duration_seconds), 4),
            "source_start_seconds": (
                None if source_start_seconds is None else round(float(source_start_seconds), 4)
            ),
            "source_end_seconds": (
                None if source_end_seconds is None else round(float(source_end_seconds), 4)
            ),
            "split_reason": split_reason,
            "part_index": part_index,
            "part_count": part_count,
            "video_start_seconds": (
                None if video_start_seconds is None else round(float(video_start_seconds), 4)
            ),
            "video_end_seconds": (
                None if video_end_seconds is None else round(float(video_end_seconds), 4)
            ),
            "pending": pending,
            "revision": revision,
            "error": error,
            "assigned_speaker": None,
            "created_speaker": False,
            "probabilities": {"unknown": 1.0} if pending else {},
            "similarities": {},
            "unknown_probability": None,
            "top_similarity": None,
            "margin": None,
            "quality": None,
            "assignment_source": None,
        }
        if decision is not None:
            payload.update({
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
                "quality": decision.quality,
                "assignment_source": decision.assignment_source,
            })
        return payload

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit("error-status", {"session_id": session_id, "message": message})


