"""Offline diarization and timestamp-split validation workflows."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common.audio_utils import load_audio_file, pad_audio, trim_silence, write_wav
from embeddings.embedding_providers import EmbeddingSubprocessClient, configure_embedding_env
from realtime.canonical_transcript import (
    canonical_overlap,
    read_canonical_segments,
    summarize_canonical_gap_groups,
)
from realtime.realtime_speaker_engine import (
    is_reassignment_accepted,
    is_reassignment_candidate,
)
from realtime.realtime_cli import RealtimeConfig
from realtime.realtime_transcript import split_transcript_by_timestamps
from realtime.validation_models import ValidationItem
from speakers.realtime_speaker_memory import SpeakerMemory

def embed_audio_with_client(
    client: EmbeddingSubprocessClient,
    audio: np.ndarray,
    sample_rate: int,
    directory: Path,
    suffix: str,
) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="realtime-speaker-validate-",
        dir=str(directory),
        delete=False,
    ) as handle:
        wav_path = Path(handle.name)
    try:
        write_wav(wav_path, audio, sample_rate)
        return client.embed_wav(wav_path)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass

def validate_cunk(args: RealtimeConfig) -> int:
    configure_embedding_env()
    client = EmbeddingSubprocessClient(
        python=Path(args.embedding_python),
        provider=args.embedding_provider,
        device=args.embedding_device,
    )
    memory = SpeakerMemory(
        same_speaker_similarity=args.same_speaker_similarity,
        similarity_temperature=args.similarity_temperature,
        speaker_softmax_temperature=args.speaker_softmax_temperature,
        new_speaker_threshold=args.new_speaker_threshold,
        duplicate_profile_similarity=args.duplicate_profile_similarity,
        unknown_short_threshold=args.unknown_short_threshold,
        min_first_speaker_seconds=args.min_first_speaker_seconds,
        min_new_speaker_seconds=args.min_new_speaker_seconds,
        late_new_speaker_min_seconds=args.late_new_speaker_min_seconds,
        max_speakers=args.max_speakers,
        min_margin=args.min_margin,
        margin_temperature=args.margin_temperature,
        update_unknown_max=args.update_unknown_max,
    )
    audio, sample_rate = load_audio_file(args.validation_audio)
    segments = read_canonical_segments(args.validation_canonical)

    items: list[ValidationItem] = []
    try:
        for index, segment in enumerate(segments):
            start_sample = max(0, int(float(segment["start"]) * sample_rate))
            end_sample = min(len(audio), int(float(segment["end"]) * sample_rate))
            chunk = trim_silence(audio[start_sample:end_sample], sample_rate)
            chunk = pad_audio(chunk, args.min_embed_seconds, sample_rate)
            embedding = embed_audio_with_client(
                client,
                chunk,
                sample_rate,
                args.validation_output.parent,
                f".canonical-{index:04d}.wav",
            )
            duration_seconds = float(segment["end"]) - float(segment["start"])
            decision = memory.classify(embedding, duration_seconds)
            items.append(
                ValidationItem(
                    session_id="validation",
                    index=index,
                    text=segment.get("text", ""),
                    duration_seconds=duration_seconds,
                    embedding=embedding,
                    decision=decision,
                    row_fields={
                        "index": index,
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration_seconds": round(duration_seconds, 4),
                        "text": segment.get("text", ""),
                        "canonical_speaker": segment["speaker"],
                    },
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for position, item in enumerate(items):
        old_decision = item.decision
        if not is_reassignment_candidate(args, old_decision, item.duration_seconds):
            continue
        candidate = memory.score_existing(
            item.embedding,
            item.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate, item.duration_seconds):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        items[position] = item.with_decision(candidate)
        reassigned_count += 1

    rows = [item.to_row() for item in items]

    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        assigned = row["assigned_speaker"]
        if assigned:
            profile_speaker_durations[assigned][row["canonical_speaker"]] += row["duration_seconds"]
    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }

    correct_count = 0
    unknown_count = 0
    correct_duration = 0.0
    total_duration = 0.0
    for row in rows:
        total_duration += row["duration_seconds"]
        mapped = profile_map.get(row["assigned_speaker"])
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == row["canonical_speaker"])
        if not row["assigned_speaker"]:
            unknown_count += 1
        if row["matches_canonical"]:
            correct_count += 1
            correct_duration += row["duration_seconds"]

    summary = {
        "embedding_provider": args.embedding_provider,
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "segments": len(rows),
        "profiles": len(profile_map),
        "unknown_segments": unknown_count,
        "reassigned_segments": reassigned_count,
        "segment_accuracy": round(correct_count / max(1, len(rows)), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "profile_map": profile_map,
        "rows": rows,
    }

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Validation output: {args.validation_output}", flush=True)
    print(f"Provider: {summary['embedding_provider']}", flush=True)
    print(f"Segments: {summary['segments']}", flush=True)
    print(f"Profiles: {summary['profiles']} ({summary['profile_map']})", flush=True)
    print(f"Unknown short/noisy segments: {summary['unknown_segments']}", flush=True)
    print(f"Reassigned uncertain segments: {summary['reassigned_segments']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0

def transcribe_validation_audio_with_realtimestt(
    args: RealtimeConfig,
    audio: np.ndarray,
) -> tuple[str, dict[str, Any], float]:
    if os.name == "nt":
        try:
            from torchaudio._extension.utils import _init_dll_path

            _init_dll_path()
        except Exception:
            pass

    from RealtimeSTT import AudioToTextRecorder

    recorder = AudioToTextRecorder(
        use_microphone=False,
        input_device_index=None,
        spinner=False,
        model=args.model,
        realtime_model_type=args.rt_model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.download_root,
        enable_realtime_transcription=False,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        no_log_file=True,
        faster_whisper_vad_filter=False,
        final_transcription_word_timestamps=True,
    )
    started = time.perf_counter()
    try:
        text = recorder.perform_final_transcription(
            audio,
            use_prompt=False,
            word_timestamps=True,
        )
        elapsed = time.perf_counter() - started
        metadata = getattr(recorder, "last_transcription_metadata", None) or {}
        return (text or "").strip(), metadata, elapsed
    finally:
        recorder.shutdown()

def validate_cunk_word_splits(args: RealtimeConfig) -> int:
    audio, sample_rate = load_audio_file(args.validation_audio)
    canonical_segments = read_canonical_segments(args.validation_canonical)
    text, metadata, transcription_seconds = transcribe_validation_audio_with_realtimestt(
        args,
        audio,
    )
    parts = split_transcript_by_timestamps(
        text=text,
        audio=audio,
        sample_rate=sample_rate,
        metadata=metadata,
        args=args,
    )

    configure_embedding_env()
    client = EmbeddingSubprocessClient(
        python=Path(args.embedding_python),
        provider=args.embedding_provider,
        device=args.embedding_device,
    )
    memory = SpeakerMemory(
        same_speaker_similarity=args.same_speaker_similarity,
        similarity_temperature=args.similarity_temperature,
        speaker_softmax_temperature=args.speaker_softmax_temperature,
        new_speaker_threshold=args.new_speaker_threshold,
        duplicate_profile_similarity=args.duplicate_profile_similarity,
        unknown_short_threshold=args.unknown_short_threshold,
        min_first_speaker_seconds=args.min_first_speaker_seconds,
        min_new_speaker_seconds=args.min_new_speaker_seconds,
        late_new_speaker_min_seconds=args.late_new_speaker_min_seconds,
        max_speakers=args.max_speakers,
        min_margin=args.min_margin,
        margin_temperature=args.margin_temperature,
        update_unknown_max=args.update_unknown_max,
    )

    items: list[ValidationItem] = []
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, part in enumerate(parts):
            chunk = trim_silence(part.audio, sample_rate)
            chunk = pad_audio(chunk, args.min_embed_seconds, sample_rate)
            embedding = embed_audio_with_client(
                client,
                chunk,
                sample_rate,
                args.validation_output.parent,
                f".word-split-{index:04d}.wav",
            )
            decision = memory.classify(embedding, part.duration_seconds)
            canonical_speaker, dominant_overlap, overlaps, total_overlap = canonical_overlap(
                canonical_segments,
                part.word_start_seconds,
                part.word_end_seconds,
            )
            overlap_speakers = {
                speaker: round(float(seconds), 4)
                for speaker, seconds in overlaps.items()
                if seconds >= args.mixed_overlap_min_seconds
            }
            items.append(
                ValidationItem(
                    session_id="word_split_validation",
                    index=index,
                    text=part.text,
                    duration_seconds=part.duration_seconds,
                    embedding=embedding,
                    decision=decision,
                    row_fields={
                        "index": index,
                        "text": part.text,
                        "audio_start": part.start_seconds,
                        "audio_end": part.end_seconds,
                        "word_start": part.word_start_seconds,
                        "word_end": part.word_end_seconds,
                        "duration_seconds": round(part.duration_seconds, 4),
                        "word_count": part.word_count,
                        "split_reason": part.split_reason,
                        "canonical_speaker": canonical_speaker,
                        "canonical_overlap_seconds": round(
                            float(dominant_overlap), 4
                        ),
                        "canonical_total_overlap_seconds": round(
                            float(total_overlap), 4
                        ),
                        "canonical_speaker_overlaps": overlap_speakers,
                        "mixed_canonical_speakers": len(overlap_speakers) > 1,
                    },
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for position, item in enumerate(items):
        old_decision = item.decision
        if not is_reassignment_candidate(args, old_decision, item.duration_seconds):
            continue
        candidate = memory.score_existing(
            item.embedding,
            item.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate, item.duration_seconds):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        items[position] = item.with_decision(candidate)
        reassigned_count += 1

    rows = [item.to_row() for item in items]

    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        assigned = row["assigned_speaker"]
        canonical_speaker = row["canonical_speaker"]
        if assigned and canonical_speaker:
            profile_speaker_durations[assigned][canonical_speaker] += row[
                "canonical_overlap_seconds"
            ]
    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }

    correct_count = 0
    scored_count = 0
    unknown_count = 0
    correct_duration = 0.0
    total_duration = 0.0
    mixed_count = 0
    mixed_duration = 0.0
    for row in rows:
        if row["assigned_speaker"] is None:
            unknown_count += 1
        mapped = profile_map.get(row["assigned_speaker"])
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == row["canonical_speaker"])
        score_duration = max(
            0.0,
            float(row["word_end"]) - float(row["word_start"]),
        )
        if row["canonical_speaker"]:
            scored_count += 1
            total_duration += score_duration
            if row["matches_canonical"]:
                correct_count += 1
                correct_duration += score_duration
        if row["mixed_canonical_speakers"]:
            mixed_count += 1
            mixed_duration += score_duration

    word_count = len((metadata or {}).get("words") or [])
    summary = {
        "embedding_provider": args.embedding_provider,
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "transcription_model": args.model,
        "transcription_seconds": round(transcription_seconds, 4),
        "transcript_characters": len(text),
        "words": word_count,
        "split_parts": len(parts),
        "average_part_seconds": round(
            sum(part.duration_seconds for part in parts) / max(1, len(parts)),
            4,
        ),
        "profiles": len(profile_map),
        "unknown_segments": unknown_count,
        "reassigned_segments": reassigned_count,
        "mixed_segments": mixed_count,
        "mixed_duration_seconds": round(mixed_duration, 4),
        "segment_accuracy": round(correct_count / max(1, scored_count), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "profile_map": profile_map,
        "canonical_gap_group_baseline": summarize_canonical_gap_groups(canonical_segments),
        "transcript": text,
        "rows": rows,
    }

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Validation output: {args.validation_output}", flush=True)
    print(f"Transcription model: {summary['transcription_model']}", flush=True)
    print(f"Transcription seconds: {summary['transcription_seconds']:.2f}", flush=True)
    print(f"Words with timestamps: {summary['words']}", flush=True)
    print(f"Timestamped split parts: {summary['split_parts']}", flush=True)
    print(f"Average part seconds: {summary['average_part_seconds']:.3f}", flush=True)
    print(f"Mixed canonical-speaker parts: {summary['mixed_segments']}", flush=True)
    print(f"Mixed duration seconds: {summary['mixed_duration_seconds']:.3f}", flush=True)
    print(f"Profiles: {summary['profiles']} ({summary['profile_map']})", flush=True)
    print(f"Unknown short/noisy segments: {summary['unknown_segments']}", flush=True)
    print(f"Reassigned uncertain segments: {summary['reassigned_segments']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0
