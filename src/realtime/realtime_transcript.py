"""Transcript timestamp splitting for realtime diarization."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.audio_utils import SAMPLE_RATE

DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS = 0.06
DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS = 0.09
DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO = 0.6


from realtime.realtime_gui_html import HTML



@dataclass
class TranscriptPart:
    text: str
    audio: np.ndarray
    start_seconds: float
    end_seconds: float
    word_start_seconds: float
    word_end_seconds: float
    word_count: int
    words: list[dict[str, Any]]
    split_reason: str
    part_index: int
    part_count: int

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def timed_word_text(words: list[dict[str, Any]]) -> str:
    pieces = [str(word.get("word", "")) for word in words if str(word.get("word", "")).strip()]
    if not pieces:
        return ""
    if any(piece.startswith(" ") for piece in pieces):
        text = "".join(pieces)
    else:
        text = " ".join(piece.strip() for piece in pieces)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def normalize_timed_words(
    metadata: dict[str, Any] | None,
    duration_seconds: float,
    max_word_seconds: float | None = None,
) -> list[dict[str, Any]]:
    words = []
    for item in (metadata or {}).get("words") or []:
        text = str(item.get("word", ""))
        if not text.strip():
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, min(duration_seconds, start))
        end = max(0.0, min(duration_seconds, end))
        if end <= start:
            continue
        words.append({"word": text, "start": start, "end": end})
    words.sort(key=lambda word: (word["start"], word["end"]))
    if max_word_seconds is not None and max_word_seconds > 0.0:
        for index, word in enumerate(words):
            start = float(word["start"])
            max_end = min(duration_seconds, start + max_word_seconds)
            if index + 1 < len(words):
                next_start = float(words[index + 1]["start"])
                if next_start > start:
                    max_end = min(max_end, next_start)
            if float(word["end"]) > max_end:
                word["end"] = max_end
    return words


def _word_ends_sentence(word_text: str) -> bool:
    return bool(re.search(r"[.!?]+[\"')\]]*$", word_text.strip()))


def _word_ends_soft_boundary(word_text: str) -> bool:
    return bool(re.search(r"[,;:]+[\"')\]]*$", word_text.strip()))


def _text_ends_strong_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]+[\"')\]]*$", text.strip()))


def _sentence_initial_uppercase_after_strong_boundary(text: str) -> str:
    for index, character in enumerate(text):
        if not character.isalpha():
            continue
        if not character.islower():
            return text
        if index + 1 < len(text) and text[index + 1].isupper():
            return text
        return f"{text[:index]}{character.upper()}{text[index + 1:]}"
    return text


def sentence_audio_boundary_between_words(
    last_word_end: float,
    next_word_start: float,
    pre_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    post_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    gap_ratio: float = DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
) -> float:
    gap = float(next_word_start - last_word_end)
    if gap <= 0.0:
        return float(last_word_end)
    if last_word_end + post_padding_seconds > next_word_start - pre_padding_seconds:
        return float(last_word_end + gap * gap_ratio)
    return float(min(last_word_end + post_padding_seconds, next_word_start - pre_padding_seconds))


def sentence_audio_clip_start_for_first_word(
    first_word_start: float,
    lower_bound: float,
    pre_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
) -> float:
    return max(float(lower_bound), float(first_word_start) - pre_padding_seconds)


def split_transcript_by_timestamps(
    text: str,
    audio: np.ndarray,
    sample_rate: int,
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[TranscriptPart]:
    """Split one RealtimeSTT final result into sentence-sized audio blocks.

    The realtime acceptance path is intentionally narrow: RealtimeSTT owns the
    final text, word timestamps, and final audio bytes. This function only
    groups those RealtimeSTT words at sentence punctuation; it does not run a
    second transcription pass or invent additional timing windows.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_seconds = float(len(audio)) / float(sample_rate or SAMPLE_RATE)
    fallback = [
        TranscriptPart(
            text=text.strip(),
            audio=audio,
            start_seconds=0.0,
            end_seconds=duration_seconds,
            word_start_seconds=0.0,
            word_end_seconds=duration_seconds,
            word_count=0,
            words=[],
            split_reason="fallback",
            part_index=0,
            part_count=1,
        )
    ]
    if not text.strip() or not getattr(args, "split_final_transcripts", True):
        return fallback

    words = normalize_timed_words(
        metadata,
        duration_seconds,
        max_word_seconds=float(args.max_word_timestamp_seconds),
    )
    if not words:
        return fallback

    groups: list[tuple[list[dict[str, Any]], str]] = []
    current: list[dict[str, Any]] = []
    strict_sentence_splits = bool(getattr(args, "strict_realtimestt_sentence_splits", True))
    max_seconds = (
        0.0
        if strict_sentence_splits
        else max(0.0, float(args.max_timestamp_split_seconds))
    )
    gap_seconds = (
        0.0
        if strict_sentence_splits
        else max(0.0, float(args.word_split_gap_seconds))
    )
    min_split_words = (
        1
        if strict_sentence_splits
        else max(1, int(args.min_timestamp_split_words))
    )
    split_on_soft = (
        False
        if strict_sentence_splits
        else bool(getattr(args, "split_on_soft_punctuation", False))
    )

    for word in words:
        if gap_seconds > 0.0 and current:
            gap = float(word["start"]) - float(current[-1]["end"])
            if gap >= gap_seconds:
                groups.append((current, "gap"))
                current = []

        current.append(word)
        if len(current) < min_split_words:
            continue

        span = float(current[-1]["end"]) - float(current[0]["start"])
        should_split = False
        reason = ""
        if _word_ends_sentence(str(word.get("word", ""))):
            should_split = True
            reason = "sentence"
        elif split_on_soft and _word_ends_soft_boundary(str(word.get("word", ""))):
            should_split = True
            reason = "punctuation"
        elif max_seconds > 0.0 and span >= max_seconds:
            should_split = True
            reason = "max_duration"

        if should_split:
            groups.append((current, reason))
            current = []

    if current:
        groups.append((current, "final"))

    if not groups:
        return fallback

    edge_padding = (
        0.0
        if strict_sentence_splits
        else max(0.0, float(args.split_audio_padding_seconds))
    )
    boundary_pre_padding_seconds = max(
        0.0,
        float(getattr(args, "sentence_boundary_pre_padding_seconds", DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS)),
    )
    boundary_post_padding_seconds = max(
        0.0,
        float(getattr(args, "sentence_boundary_post_padding_seconds", DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS)),
    )
    boundary_gap_ratio = float(getattr(args, "sentence_boundary_gap_ratio", DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO))
    parts: list[TranscriptPart] = []
    raw_bounds = [
        (float(group[0][0]["start"]), float(group[0][-1]["end"]))
        for group in groups
    ]
    previous_emitted_ended_sentence = False
    for index, ((group, reason), (raw_start, raw_end)) in enumerate(zip(groups, raw_bounds)):
        previous_boundary = None
        if index > 0:
            previous_end = raw_bounds[index - 1][1]
            previous_boundary = sentence_audio_boundary_between_words(
                previous_end,
                raw_start,
                pre_padding_seconds=boundary_pre_padding_seconds,
                post_padding_seconds=boundary_post_padding_seconds,
                gap_ratio=boundary_gap_ratio,
            )
            start = sentence_audio_clip_start_for_first_word(
                raw_start,
                previous_boundary,
                pre_padding_seconds=max(edge_padding, boundary_pre_padding_seconds),
            )
        else:
            start = sentence_audio_clip_start_for_first_word(
                raw_start,
                0.0,
                pre_padding_seconds=max(edge_padding, boundary_pre_padding_seconds),
            )
        if index + 1 < len(raw_bounds):
            next_start = raw_bounds[index + 1][0]
            end = sentence_audio_boundary_between_words(
                raw_end,
                next_start,
                pre_padding_seconds=boundary_pre_padding_seconds,
                post_padding_seconds=boundary_post_padding_seconds,
                gap_ratio=boundary_gap_ratio,
            )
        else:
            end = raw_end + max(edge_padding, boundary_post_padding_seconds)
        start = max(0.0, min(duration_seconds, start))
        end = max(start, min(duration_seconds, end))
        start_sample = max(0, min(len(audio), int(start * sample_rate)))
        end_sample = max(start_sample, min(len(audio), int(math.ceil(end * sample_rate))))
        if end_sample <= start_sample:
            continue
        part_text = timed_word_text(group)
        if not part_text:
            continue
        if previous_emitted_ended_sentence:
            part_text = _sentence_initial_uppercase_after_strong_boundary(part_text)
        parts.append(
            TranscriptPart(
                text=part_text,
                audio=audio[start_sample:end_sample].astype(np.float32, copy=False),
                start_seconds=round(float(start_sample) / float(sample_rate), 4),
                end_seconds=round(float(end_sample) / float(sample_rate), 4),
                word_start_seconds=round(float(raw_start), 4),
                word_end_seconds=round(float(raw_end), 4),
                word_count=len(group),
                words=[
                    {
                        "text": str(word.get("word", "")),
                        "start_seconds": round(float(word["start"]), 4),
                        "end_seconds": round(float(word["end"]), 4),
                    }
                    for word in group
                ],
                split_reason=reason,
                part_index=index,
                part_count=0,
            )
        )
        previous_emitted_ended_sentence = _text_ends_strong_sentence(part_text)

    if not parts:
        return fallback
    if len(parts) == 1:
        parts[0].text = text.strip()
    for index, part in enumerate(parts):
        part.part_index = index
        part.part_count = len(parts)
    return parts


