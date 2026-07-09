"""Realtime YouTube WASAPI transcription with embedding-only speaker labels.

The browser plays a YouTube iframe, RealtimeSTT captures system audio through
WASAPI loopback, and this script assigns each completed sentence to a stable
incremental speaker memory using one voice-embedding provider.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from common.audio_utils import (
    INT16_MAX_ABS_VALUE,
    SAMPLE_RATE,
    audio_to_float_mono,
    clamp01,
    cosine_similarity,
    json_dumps,
    load_audio_file,
    normalize_vector,
    pad_audio,
    sigmoid,
    softmax,
    trim_silence,
    write_wav,
)
from embeddings.embedding_providers import (
    BENCHMARK_PROVIDER_ALIASES,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_SPEECHBRAIN_MODEL,
    DEFAULT_SPEECHBRAIN_RESNET_MODEL,
    BenchmarkAdapterProvider,
    EmbeddingSubprocessClient,
    PyannoteModelProvider,
    ResemblyzerProvider,
    SpeechBrainProvider,
    StackedEmbeddingProvider,
    canonical_embedding_provider_name,
    choose_torch_device,
    configure_embedding_env,
    create_embedding_provider,
    create_single_embedding_provider,
    default_embedding_python,
    parse_embedding_provider_stack,
    parse_embedding_provider_stack_specs,
    run_embedding_helper,
    sanitize,
)
from speakers.realtime_speaker_memory import SpeakerDecision, SpeakerMemory, SpeakerProfile
from paths import (
    CACHE_DIR,
    CUNK_CANONICAL,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REALTIME_VALIDATION_OUTPUT_DIR,
)
from window.language_config import default_language_code, language_arg


ROOT = PROJECT_ROOT

from realtime.realtime_capture import (
    EventBus,
    TraceLogger,
    YouTubeWasapiController,
    extract_youtube_video_id,
)
from realtime.realtime_server import GuiServer
from realtime.realtime_speaker_engine import (
    ProcessedSentenceRecord,
    is_reassignment_accepted,
    is_reassignment_candidate,
)
from realtime.realtime_transcript import (
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    split_transcript_by_timestamps,
)

def read_canonical_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return [
            {
                "speaker": segment["speaker_id"],
                "start": segment["start_sec"],
                "end": segment["end_sec"],
                "text": segment["text"],
            }
            for segment in data["segments"]
        ]
    raise ValueError(f"Could not read canonical segments from {path}")


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


def validate_cunk(args: argparse.Namespace) -> int:
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

    rows: list[dict[str, Any]] = []
    validation_records: list[ProcessedSentenceRecord] = []
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
            rows.append({
                "index": index,
                "start": segment["start"],
                "end": segment["end"],
                "duration_seconds": round(duration_seconds, 4),
                "text": segment.get("text", ""),
                "canonical_speaker": segment["speaker"],
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "reassigned": False,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
            })
            validation_records.append(
                ProcessedSentenceRecord(
                    session_id="validation",
                    index=index,
                    text=segment.get("text", ""),
                    duration_seconds=duration_seconds,
                    embedding=embedding.astype(np.float32, copy=True),
                    decision=decision,
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for record in validation_records:
        old_decision = record.decision
        if not is_reassignment_candidate(args, old_decision, record.duration_seconds):
            continue
        candidate = memory.score_existing(
            record.embedding,
            record.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate, record.duration_seconds):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        record.decision = candidate
        row = rows[record.index]
        row.update({
            "assigned_speaker": candidate.assigned_speaker,
            "created_speaker": candidate.created_speaker,
            "reassigned": True,
            "probabilities": candidate.probabilities,
            "similarities": candidate.similarities,
            "unknown_probability": candidate.unknown_probability,
            "top_similarity": candidate.top_similarity,
            "margin": candidate.margin,
        })
        reassigned_count += 1

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


def canonical_overlap(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> tuple[str | None, float, Counter[str], float]:
    overlaps: Counter[str] = Counter()
    for segment in segments:
        left = max(start, float(segment["start"]))
        right = min(end, float(segment["end"]))
        if right <= left:
            continue
        overlaps[str(segment["speaker"])] += right - left
    if not overlaps:
        return None, 0.0, overlaps, 0.0
    dominant, dominant_seconds = overlaps.most_common(1)[0]
    total = sum(overlaps.values())
    return dominant, float(dominant_seconds), overlaps, float(total)


def summarize_canonical_gap_groups(
    segments: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.6, 0.4, 0.28, 0.2, 0.12),
) -> list[dict[str, Any]]:
    summaries = []
    if not segments:
        return summaries
    for threshold in thresholds:
        groups = []
        current = [segments[0]]
        for previous, segment in zip(segments, segments[1:]):
            gap = float(segment["start"]) - float(previous["end"])
            if gap <= threshold:
                current.append(segment)
            else:
                groups.append(current)
                current = [segment]
        groups.append(current)
        mixed = [
            group for group in groups
            if len({item["speaker"] for item in group}) > 1
        ]
        summaries.append({
            "threshold_seconds": threshold,
            "groups": len(groups),
            "mixed_groups": len(mixed),
            "mixed_duration_seconds": round(
                sum(float(group[-1]["end"]) - float(group[0]["start"]) for group in mixed),
                4,
            ),
            "max_group_seconds": round(
                max(float(group[-1]["end"]) - float(group[0]["start"]) for group in groups),
                4,
            ),
        })
    return summaries


def transcribe_validation_audio_with_realtimestt(
    args: argparse.Namespace,
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


def validate_cunk_word_splits(args: argparse.Namespace) -> int:
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

    rows: list[dict[str, Any]] = []
    validation_records: list[ProcessedSentenceRecord] = []
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
            rows.append({
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
                "canonical_overlap_seconds": round(float(dominant_overlap), 4),
                "canonical_total_overlap_seconds": round(float(total_overlap), 4),
                "canonical_speaker_overlaps": overlap_speakers,
                "mixed_canonical_speakers": len(overlap_speakers) > 1,
                "assigned_speaker": decision.assigned_speaker,
                "created_speaker": decision.created_speaker,
                "reassigned": False,
                "probabilities": decision.probabilities,
                "similarities": decision.similarities,
                "unknown_probability": decision.unknown_probability,
                "top_similarity": decision.top_similarity,
                "margin": decision.margin,
            })
            validation_records.append(
                ProcessedSentenceRecord(
                    session_id="word_split_validation",
                    index=index,
                    text=part.text,
                    duration_seconds=part.duration_seconds,
                    embedding=embedding.astype(np.float32, copy=True),
                    decision=decision,
                )
            )
    finally:
        client.shutdown()

    reassigned_count = 0
    for record in validation_records:
        old_decision = record.decision
        if not is_reassignment_candidate(args, old_decision, record.duration_seconds):
            continue
        candidate = memory.score_existing(
            record.embedding,
            record.duration_seconds,
            force_assignment=True,
        )
        if not is_reassignment_accepted(args, candidate, record.duration_seconds):
            continue
        if old_decision.assigned_speaker == candidate.assigned_speaker:
            continue
        record.decision = candidate
        row = rows[record.index]
        row.update({
            "assigned_speaker": candidate.assigned_speaker,
            "created_speaker": candidate.created_speaker,
            "reassigned": True,
            "probabilities": candidate.probabilities,
            "similarities": candidate.similarities,
            "unknown_probability": candidate.unknown_probability,
            "top_similarity": candidate.top_similarity,
            "margin": candidate.margin,
        })
        reassigned_count += 1

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


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def normalized_match_text(text: str) -> str:
    return " ".join(text_tokens(text))


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0] * (len(right) + 1)
        for index, right_item in enumerate(right, 1):
            if left_item == right_item:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def lcs_alignment(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    if not left or not right:
        return []

    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index, left_item in enumerate(left, 1):
        row = table[left_index]
        previous_row = table[left_index - 1]
        for right_index, right_item in enumerate(right, 1):
            if left_item == right_item:
                row[right_index] = previous_row[right_index - 1] + 1
            else:
                row[right_index] = max(previous_row[right_index], row[right_index - 1])

    pairs: list[tuple[int, int]] = []
    left_index = len(left)
    right_index = len(right)
    while left_index > 0 and right_index > 0:
        if left[left_index - 1] == right[right_index - 1]:
            pairs.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif table[left_index - 1][right_index] >= table[left_index][right_index - 1]:
            left_index -= 1
        else:
            right_index -= 1
    pairs.reverse()
    return pairs


def lcs_speaker_matches_by_final(
    finals: dict[int, dict[str, Any]],
    canonical_segments: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    live_tokens: list[str] = []
    live_token_final_indices: list[int] = []
    live_final_token_counts: Counter[int] = Counter()
    for final_index, final in sorted(finals.items()):
        tokens = text_tokens(str(final.get("text") or ""))
        live_final_token_counts[final_index] = len(tokens)
        for token in tokens:
            live_tokens.append(token)
            live_token_final_indices.append(final_index)

    canonical_tokens: list[str] = []
    canonical_token_speakers: list[str] = []
    for segment in canonical_segments:
        speaker = str(segment.get("speaker") or "")
        if not speaker:
            continue
        for token in text_tokens(str(segment.get("text") or "")):
            canonical_tokens.append(token)
            canonical_token_speakers.append(speaker)

    matches: dict[int, Counter[str]] = defaultdict(Counter)
    for live_token_index, canonical_token_index in lcs_alignment(live_tokens, canonical_tokens):
        final_index = live_token_final_indices[live_token_index]
        speaker = canonical_token_speakers[canonical_token_index]
        matches[final_index][speaker] += 1

    result: dict[int, dict[str, Any]] = {}
    for final_index, speaker_counts in matches.items():
        if not speaker_counts:
            continue
        speaker, count = speaker_counts.most_common(1)[0]
        token_count = max(1, live_final_token_counts[final_index])
        result[final_index] = {
            "speaker": speaker,
            "score": count / float(token_count),
            "matched_tokens": int(sum(speaker_counts.values())),
            "speaker_token_counts": dict(speaker_counts),
        }
    return result


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(text_tokens(left))
    right_tokens = set(text_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def best_canonical_text_match(
    text: str,
    canonical_segments: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    normalized = normalized_match_text(text)
    if not normalized:
        return 0.0, None
    best_score = 0.0
    best_segment = None
    for segment in canonical_segments:
        segment_text = str(segment.get("text") or "")
        segment_normalized = normalized_match_text(segment_text)
        if not segment_normalized:
            continue
        ratio = difflib.SequenceMatcher(None, normalized, segment_normalized).ratio()
        score = 0.65 * ratio + 0.35 * token_jaccard(text, segment_text)
        if score > best_score:
            best_score = score
            best_segment = segment
    return best_score, best_segment


def analyze_trace_against_canonical(
    records: list[dict[str, Any]],
    canonical_segments: list[dict[str, Any]],
    match_mode: str = "auto",
) -> dict[str, Any]:
    finals: dict[int, dict[str, Any]] = {}
    sentences: dict[int, dict[str, Any]] = {}
    for record in records:
        payload = record.get("payload") or {}
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        if record.get("event") == "final":
            finals[index] = payload
        elif (
            record.get("event") == "sentence"
            and not payload.get("pending")
            and not payload.get("provisional_assignment")
        ):
            sentences[index] = payload

    canonical_text = " ".join(str(segment.get("text") or "") for segment in canonical_segments)
    live_text = " ".join(str(final.get("text") or "") for _, final in sorted(finals.items()))
    canonical_tokens = text_tokens(canonical_text)
    live_tokens = text_tokens(live_text)
    common_tokens = lcs_length(live_tokens, canonical_tokens)

    rows: list[dict[str, Any]] = []
    timestamped_rows = 0
    text_matched_rows = 0
    profile_speaker_durations: dict[str, Counter[str]] = defaultdict(Counter)
    profile_speaker_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lcs_text_matches = (
        {}
        if match_mode == "timestamp"
        else lcs_speaker_matches_by_final(finals, canonical_segments)
    )

    for index, final in sorted(finals.items()):
        sentence = sentences.get(index) or {}
        assigned = sentence.get("assigned_speaker") or final.get("assigned_speaker")
        text = str(final.get("text") or sentence.get("text") or "")
        video_start = final.get("video_start_seconds")
        video_end = final.get("video_end_seconds")
        if video_start is None:
            video_start = sentence.get("video_start_seconds")
        if video_end is None:
            video_end = sentence.get("video_end_seconds")

        canonical_speaker = None
        canonical_overlap_seconds = 0.0
        canonical_total_overlap_seconds = 0.0
        canonical_match_score = None
        row_match_mode = "unmatched"
        duration_seconds = float(sentence.get("duration_seconds") or 0.0)
        try:
            if match_mode != "text" and video_start is not None and video_end is not None:
                video_start = float(video_start)
                video_end = float(video_end)
                if math.isfinite(video_start) and math.isfinite(video_end) and video_end > video_start:
                    canonical_speaker, canonical_overlap_seconds, overlaps, canonical_total_overlap_seconds = (
                        canonical_overlap(canonical_segments, video_start, video_end)
                    )
                    duration_seconds = max(duration_seconds, video_end - video_start)
                    timestamped_rows += 1
                    row_match_mode = "timestamp"
                else:
                    video_start = None
                    video_end = None
        except (TypeError, ValueError):
            video_start = None
            video_end = None

        if canonical_speaker is None and match_mode != "timestamp":
            lcs_match = lcs_text_matches.get(index)
            if lcs_match is not None and float(lcs_match.get("score") or 0.0) >= 0.34:
                canonical_speaker = str(lcs_match["speaker"])
                canonical_match_score = float(lcs_match["score"])
                text_matched_rows += 1
                row_match_mode = "text_lcs"
            else:
                canonical_match_score, segment = best_canonical_text_match(text, canonical_segments)
                if segment is not None and canonical_match_score >= 0.45:
                    canonical_speaker = str(segment["speaker"])
                    text_matched_rows += 1
                    row_match_mode = "text"

        if assigned and canonical_speaker:
            weight = max(0.001, float(duration_seconds))
            profile_speaker_durations[str(assigned)][canonical_speaker] += weight
            profile_speaker_counts[str(assigned)][canonical_speaker] += 1

        rows.append({
            "index": index,
            "text": text,
            "assigned_speaker": assigned,
            "video_start_seconds": video_start,
            "video_end_seconds": video_end,
            "duration_seconds": round(float(duration_seconds), 4),
            "canonical_speaker": canonical_speaker,
            "canonical_overlap_seconds": round(float(canonical_overlap_seconds), 4),
            "canonical_total_overlap_seconds": round(float(canonical_total_overlap_seconds), 4),
            "canonical_text_match_score": canonical_match_score,
            "canonical_text_lcs_match": lcs_text_matches.get(index),
            "match_mode": row_match_mode,
            "probabilities": sentence.get("probabilities") or {},
            "similarities": sentence.get("similarities") or {},
            "assignment_source": sentence.get("assignment_source"),
        })

    profile_map = {
        profile: counter.most_common(1)[0][0]
        for profile, counter in profile_speaker_durations.items()
        if counter
    }
    if not profile_map:
        profile_map = {
            profile: counter.most_common(1)[0][0]
            for profile, counter in profile_speaker_counts.items()
            if counter
        }

    scored_count = 0
    correct_count = 0
    total_duration = 0.0
    correct_duration = 0.0
    unknown_count = 0
    for row in rows:
        assigned = row.get("assigned_speaker")
        if not assigned:
            unknown_count += 1
        canonical_speaker = row.get("canonical_speaker")
        if not canonical_speaker:
            row["mapped_speaker"] = None
            row["matches_canonical"] = False
            continue
        mapped = profile_map.get(str(assigned)) if assigned else None
        row["mapped_speaker"] = mapped
        row["matches_canonical"] = bool(mapped and mapped == canonical_speaker)
        scored_count += 1
        duration = float(row.get("duration_seconds") or 0.0)
        total_duration += duration
        if row["matches_canonical"]:
            correct_count += 1
            correct_duration += duration

    return {
        "match_mode": match_mode,
        "final_segments": len(finals),
        "resolved_segments": len(sentences),
        "timestamped_segments": timestamped_rows,
        "text_matched_segments": text_matched_rows,
        "unknown_segments": unknown_count,
        "canonical_words": len(canonical_tokens),
        "live_final_words": len(live_tokens),
        "lcs_words": common_tokens,
        "text_recall": round(common_tokens / max(1, len(canonical_tokens)), 4),
        "text_precision": round(common_tokens / max(1, len(live_tokens)), 4),
        "profile_map": profile_map,
        "assigned_counts": dict(Counter(str(row.get("assigned_speaker") or "UNKNOWN") for row in rows)),
        "segment_accuracy": round(correct_count / max(1, scored_count), 4),
        "duration_accuracy": round(correct_duration / max(0.0001, total_duration), 4),
        "rows": rows,
    }


def trace_record_session_id(record: dict[str, Any]) -> str | None:
    payload = record.get("payload") or {}
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def trace_session_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        session_id = trace_record_session_id(record)
        if session_id and session_id not in seen:
            ids.append(session_id)
            seen.add(session_id)
    return ids


def filter_trace_records_by_session(
    records: list[dict[str, Any]],
    session_selector: str,
) -> tuple[list[dict[str, Any]], str | None]:
    selector = (session_selector or "latest").strip()
    if not selector or selector.lower() == "all":
        return records, None

    ids = trace_session_ids(records)
    if not ids:
        return records, None
    selected = ids[-1] if selector.lower() == "latest" else selector
    filtered = [
        record
        for record in records
        if trace_record_session_id(record) == selected
    ]
    if not filtered:
        raise ValueError(
            f"Trace session {selected!r} was not found. Available sessions: "
            + ", ".join(ids)
        )
    return filtered, selected


def analyze_trace(
    path: Path,
    canonical_path: Path | None = None,
    output_path: Path | None = None,
    summary_only: bool = False,
    match_mode: str = "auto",
    trace_session: str = "latest",
) -> int:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No trace records found in {path}")
        return 1

    start = records[0].get("time", 0)
    print(f"Trace: {path}")
    print(f"Records: {len(records)}")
    session_ids = trace_session_ids(records)
    if session_ids:
        print(f"Trace sessions: {len(session_ids)}")
    selected_session = None
    if canonical_path is not None and canonical_path.exists():
        records, selected_session = filter_trace_records_by_session(records, trace_session)
        if selected_session is not None:
            print(f"Selected trace session: {selected_session}")
            print(f"Session records: {len(records)}")
        start = records[0].get("time", start)
        canonical_segments = read_canonical_segments(canonical_path)
        summary = analyze_trace_against_canonical(
            records,
            canonical_segments,
            match_mode=match_mode,
        )
        if selected_session is not None:
            summary["trace_session_id"] = selected_session
            summary["trace_sessions"] = session_ids
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Trace analysis output: {output_path}")
        print(f"Match mode: {summary['match_mode']}")
        print(f"Final segments: {summary['final_segments']}")
        print(f"Resolved segments: {summary['resolved_segments']}")
        print(f"Timestamped segments: {summary['timestamped_segments']}")
        print(f"Text-matched fallback segments: {summary['text_matched_segments']}")
        print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}")
        print(f"Text recall/precision by LCS: {summary['text_recall']:.3f} / {summary['text_precision']:.3f}")
        print(f"Assigned counts: {summary['assigned_counts']}")
        print(f"Profile map: {summary['profile_map']}")
        print(f"Unknown segments: {summary['unknown_segments']}")
        print(f"Live segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}")
        print(f"Live duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}")
        if summary_only:
            return 0
    elif summary_only:
        return 0
    for record in records:
        event = record.get("event")
        payload = record.get("payload") or {}
        if event not in {"sentence", "final", "realtime", "capture-ready", "error-status"}:
            continue
        elapsed = float(record.get("time", start) or start) - float(start or 0)
        if event == "sentence":
            video_range = ""
            if payload.get("video_start_seconds") is not None:
                video_range = (
                    f" video={payload.get('video_start_seconds')}-"
                    f"{payload.get('video_end_seconds')}"
                )
            print(
                f"{elapsed:8.3f}s sentence "
                f"idx={payload.get('index')} "
                f"speaker={payload.get('assigned_speaker')} "
                f"new={payload.get('created_speaker')} "
                f"unknown={payload.get('unknown_probability')} "
                f"{video_range} "
                f"text={(payload.get('text') or '')[:120]!r}"
            )
        elif event == "final":
            video_range = ""
            if payload.get("video_start_seconds") is not None:
                video_range = (
                    f" video={payload.get('video_start_seconds')}-"
                    f"{payload.get('video_end_seconds')}"
                )
            print(
                f"{elapsed:8.3f}s final "
                f"idx={payload.get('index')} "
                f"{video_range} "
                f"text={(payload.get('text') or '')[:120]!r}"
            )
        else:
            print(f"{elapsed:8.3f}s {event} {payload}")
    return 0


def validate_cunk_realtime_replay(args: argparse.Namespace) -> int:
    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            OUTPUTS_DIR
            / "realtime-speakerdiarize-traces"
            / f"trace-replay-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    controller = YouTubeWasapiController(args, bus)
    session_id = "replay"
    with controller._lock:
        controller._session_id = session_id
        controller._stop_event = threading.Event()
        controller._video_ids[session_id] = "local-replay"
        controller._sentence_indices[session_id] = 0
    controller.speaker_engine.start_session(session_id)

    audio, sample_rate = load_audio_file(args.validation_audio)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX_ABS_VALUE).astype(np.int16)
    chunk_samples = max(1, int(sample_rate * args.replay_chunk_seconds))
    replay_speed = max(0.1, float(args.replay_speed))
    stop_event = threading.Event()
    recorder = None
    final_thread = None
    started_at = time.monotonic()
    try:
        recorder = controller._create_recorder(session_id, None)
        recorder.start()
        controller._set_video_time(session_id, 0.0)
        final_thread = threading.Thread(
            target=controller._consume_final_text,
            args=(session_id, recorder, stop_event),
            name="ReplayFinalConsumer",
            daemon=True,
        )
        final_thread.start()

        controller._status(
            session_id,
            f"Replay feeding {args.validation_audio} at {replay_speed:.1f}x.",
        )
        for start in range(0, len(audio_int16), chunk_samples):
            end = min(len(audio_int16), start + chunk_samples)
            chunk = audio_int16[start:end]
            recorder.feed_audio(chunk, original_sample_rate=sample_rate)
            controller._set_video_time(session_id, end / float(sample_rate))
            if args.replay_sleep:
                time.sleep((len(chunk) / float(sample_rate)) / replay_speed)

        silence_seconds = max(0.5, float(args.replay_trailing_silence_seconds))
        silence = np.zeros(int(sample_rate * silence_seconds), dtype=np.int16)
        for start in range(0, len(silence), chunk_samples):
            end = min(len(silence), start + chunk_samples)
            chunk = silence[start:end]
            recorder.feed_audio(chunk, original_sample_rate=sample_rate)
            controller._set_video_time(
                session_id,
                (len(audio_int16) + end) / float(sample_rate),
            )
            if args.replay_sleep:
                time.sleep((len(chunk) / float(sample_rate)) / replay_speed)

        controller._status(session_id, "Replay audio feed complete; draining final transcripts.")
        time.sleep(max(0.0, float(args.replay_drain_seconds)))
    finally:
        stop_event.set()
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass
        if final_thread is not None:
            final_thread.join(timeout=3.0)
        drain_deadline = time.monotonic() + max(0.0, float(args.replay_embedding_drain_seconds))
        while (
            getattr(controller.speaker_engine.jobs, "unfinished_tasks", 0) > 0
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.1)
        controller.speaker_engine.shutdown()

    elapsed = time.monotonic() - started_at
    records = []
    with trace.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    canonical_segments = read_canonical_segments(args.validation_canonical)
    analysis_match_mode = "timestamp" if abs(replay_speed - 1.0) < 0.05 else "text"
    summary = analyze_trace_against_canonical(
        records,
        canonical_segments,
        match_mode=analysis_match_mode,
    )
    summary.update({
        "trace": str(trace.path),
        "audio": str(args.validation_audio),
        "canonical": str(args.validation_canonical),
        "elapsed_seconds": round(elapsed, 4),
        "replay_speed": replay_speed,
    })
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Replay trace: {trace.path}", flush=True)
    print(f"Replay analysis output: {args.validation_output}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Match mode: {summary['match_mode']}", flush=True)
    print(f"Final segments: {summary['final_segments']}", flush=True)
    print(f"Resolved segments: {summary['resolved_segments']}", flush=True)
    print(f"Timestamped segments: {summary['timestamped_segments']}", flush=True)
    print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}", flush=True)
    print(
        "Text recall/precision by LCS: "
        f"{summary['text_recall']:.3f} / {summary['text_precision']:.3f}",
        flush=True,
    )
    print(f"Assigned counts: {summary['assigned_counts']}", flush=True)
    print(f"Profile map: {summary['profile_map']}", flush=True)
    print(f"Unknown segments: {summary['unknown_segments']}", flush=True)
    print(f"Live segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Live duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Embedding-only realtime speaker diarization for a YouTube WASAPI capture."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--allow-default-input", action="store_true")

    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--rt-model", default="tiny.en")
    parser.add_argument("--language", type=language_arg, default=default_language_code())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--split-marks", default="off")
    parser.add_argument("--realtime-processing-pause", type=float, default=0.1)
    parser.add_argument("--post-speech-silence-duration", type=float, default=1.25)
    parser.add_argument("--stop-trailing-silence-seconds", type=float, default=2.0)
    parser.add_argument("--stop-drain-seconds", type=float, default=25.0)
    parser.add_argument("--stop-embedding-drain-seconds", type=float, default=10.0)
    parser.add_argument("--final-video-latency-seconds", type=float, default=0.8)
    parser.add_argument("--min-length-of-recording", type=float, default=0.0)
    parser.add_argument("--silero-sensitivity", type=float, default=0.05)
    parser.add_argument("--webrtc-sensitivity", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--beam-size-realtime", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "RealtimeSTT final decoder batch size. 0 uses the regular decoder; "
            "RealtimeSTT's default 16 is faster but dropped words on the Cunk clip."
        ),
    )
    parser.add_argument(
        "--realtime-batch-size",
        type=int,
        default=0,
        help="RealtimeSTT preview decoder batch size. 0 uses the regular decoder.",
    )
    parser.add_argument(
        "--no-final-word-timestamps",
        dest="final_word_timestamps",
        action="store_false",
    )
    parser.add_argument(
        "--no-split-final-transcripts",
        dest="split_final_transcripts",
        action="store_false",
    )
    parser.add_argument("--word-split-gap-seconds", type=float, default=0.0)
    parser.add_argument("--max-timestamp-split-seconds", type=float, default=0.0)
    parser.add_argument("--max-word-timestamp-seconds", type=float, default=1.2)
    parser.add_argument("--min-timestamp-split-words", type=int, default=1)
    parser.add_argument("--split-audio-padding-seconds", type=float, default=0.0)
    parser.add_argument(
        "--sentence-boundary-pre-padding-seconds",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
        help="Audio kept before the next word when cutting between two sentence groups.",
    )
    parser.add_argument(
        "--sentence-boundary-post-padding-seconds",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
        help="Audio kept after the last word when cutting between two sentence groups.",
    )
    parser.add_argument(
        "--sentence-boundary-gap-ratio",
        type=float,
        default=DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
        help="For tight word gaps, fraction of the gap assigned to the previous sentence.",
    )
    parser.add_argument("--split-on-soft-punctuation", action="store_true")
    parser.add_argument(
        "--allow-non-sentence-live-splits",
        dest="strict_realtimestt_sentence_splits",
        action="store_false",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-python", type=Path, default=default_embedding_python())
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--same-speaker-similarity", type=float, default=0.45)
    parser.add_argument("--similarity-temperature", type=float, default=0.07)
    parser.add_argument("--speaker-softmax-temperature", type=float, default=0.075)
    parser.add_argument("--new-speaker-threshold", type=float, default=0.58)
    parser.add_argument("--duplicate-profile-similarity", type=float, default=0.40)
    parser.add_argument("--unknown-short-threshold", type=float, default=0.86)
    parser.add_argument("--min-first-speaker-seconds", type=float, default=1.2)
    parser.add_argument("--min-new-speaker-seconds", type=float, default=2.0)
    parser.add_argument("--late-new-speaker-min-seconds", type=float, default=3.5)
    parser.add_argument("--min-embed-seconds", type=float, default=0.5)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--margin-temperature", type=float, default=0.035)
    parser.add_argument("--update-unknown-max", type=float, default=0.55)
    parser.add_argument(
        "--no-reassign-uncertain-sentences",
        dest="reassign_uncertain_sentences",
        action="store_false",
    )
    parser.add_argument("--reassign-max-seconds", type=float, default=2.2)
    parser.add_argument("--reassign-unknown-min", type=float, default=0.7)
    parser.add_argument("--reassign-unknown-max", type=float, default=0.82)
    parser.add_argument("--reassign-min-similarity", type=float, default=0.42)
    parser.add_argument("--reassign-short-max-seconds", type=float, default=1.2)
    parser.add_argument("--reassign-short-min-similarity", type=float, default=0.30)
    parser.add_argument("--reassign-short-min-margin", type=float, default=0.10)
    parser.add_argument(
        "--no-context-assign-short-fragments",
        dest="context_assign_short_fragments",
        action="store_false",
        help="Disable nearby-speaker context assignment for very short uncertain fragments.",
    )
    parser.add_argument("--context-assign-max-seconds", type=float, default=1.0)
    parser.add_argument("--context-assign-candidate-unknown-min", type=float, default=0.9)
    parser.add_argument("--context-assign-window", type=int, default=4)
    parser.add_argument("--context-assign-stable-unknown-max", type=float, default=0.7)
    parser.add_argument("--context-assign-stable-min-seconds", type=float, default=0.5)
    parser.add_argument("--context-assign-same-speaker-confidence", type=float, default=0.92)
    parser.add_argument("--context-assign-disagree-confidence", type=float, default=0.78)
    parser.add_argument("--context-assign-disagree-min-similarity", type=float, default=0.28)
    parser.add_argument("--context-assign-disagree-margin", type=float, default=0.08)
    parser.add_argument("--context-assign-one-sided-confidence", type=float, default=0.82)
    parser.add_argument("--context-assign-one-sided-block-margin", type=float, default=0.12)
    parser.add_argument(
        "--no-context-assign-one-sided",
        dest="context_assign_one_sided",
        action="store_false",
        help="Only context-assign short fragments when stable speakers on both sides agree.",
    )
    parser.add_argument(
        "--segment-audio-dir",
        type=Path,
        default=CACHE_DIR / "realtime_speakerdiarize_segments",
    )
    parser.add_argument("--keep-segment-audio", action="store_true")

    parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help="Optional JSONL trace path for backend/frontend UI events.",
    )
    parser.add_argument(
        "--analyze-trace",
        type=Path,
        default=None,
        help="Analyze a previously captured JSONL trace and exit.",
    )
    parser.add_argument(
        "--trace-analysis-output",
        type=Path,
        default=None,
        help="Write structured trace-vs-canonical analysis JSON when --analyze-trace is used.",
    )
    parser.add_argument("--trace-summary-only", action="store_true")
    parser.add_argument(
        "--trace-match-mode",
        choices=("auto", "timestamp", "text"),
        default="auto",
        help="How --analyze-trace maps trace rows to canonical speakers.",
    )
    parser.add_argument(
        "--trace-session",
        default="latest",
        help=(
            "Which session to analyze from a multi-session trace: latest, all, "
            "or an explicit session id. Defaults to latest."
        ),
    )
    parser.add_argument("--validate-cunk", action="store_true")
    parser.add_argument("--validate-cunk-word-splits", action="store_true")
    parser.add_argument("--validate-cunk-realtime-replay", action="store_true")
    parser.add_argument(
        "--validation-audio",
        type=Path,
        default=OUTPUTS_DIR / "pyannote-cunk" / "cunk_on_earth_clip.mp3",
    )
    parser.add_argument(
        "--validation-canonical",
        type=Path,
        default=CUNK_CANONICAL,
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=REALTIME_VALIDATION_OUTPUT_DIR / "latest.json",
    )
    parser.add_argument("--mixed-overlap-min-seconds", type=float, default=0.05)
    parser.add_argument("--replay-speed", type=float, default=8.0)
    parser.add_argument("--replay-chunk-seconds", type=float, default=0.1)
    parser.add_argument("--replay-trailing-silence-seconds", type=float, default=2.0)
    parser.add_argument("--replay-drain-seconds", type=float, default=25.0)
    parser.add_argument("--replay-embedding-drain-seconds", type=float, default=15.0)
    parser.add_argument(
        "--no-replay-sleep",
        dest="replay_sleep",
        action="store_false",
        help="Feed replay audio as fast as possible instead of wall-clock pacing.",
    )
    parser.set_defaults(replay_sleep=True)
    parser.add_argument("--embedding-helper", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rt_model_was_explicit = any(
        item == "--rt-model" or item.startswith("--rt-model=")
        for item in raw_argv
    )
    if args.language != "en" and not rt_model_was_explicit and str(args.rt_model).endswith(".en"):
        args.rt_model = str(args.rt_model)[:-3]
    return args


def main() -> int:
    args = parse_args()
    if args.embedding_helper:
        return run_embedding_helper(args)
    if args.analyze_trace is not None:
        return analyze_trace(
            args.analyze_trace,
            canonical_path=args.validation_canonical,
            output_path=args.trace_analysis_output,
            summary_only=args.trace_summary_only,
            match_mode=args.trace_match_mode,
            trace_session=args.trace_session,
        )
    if args.validate_cunk_realtime_replay:
        return validate_cunk_realtime_replay(args)
    if args.validate_cunk_word_splits:
        return validate_cunk_word_splits(args)
    if args.validate_cunk:
        return validate_cunk(args)

    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            OUTPUTS_DIR
            / "realtime-speakerdiarize-traces"
            / f"trace-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    server = GuiServer((args.host, args.port), args, bus, trace)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Serving realtime speaker diarization GUI at {url}", flush=True)
    print(f"Trace log: {trace.path}", flush=True)
    print(f"Embedding helper: {args.embedding_python}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.controller.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
