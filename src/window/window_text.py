"""Text, word mapping, and sentence-window helpers for window diarization."""

from __future__ import annotations

import re
from typing import Any

from stream2sentence import generate_sentences, init_tokenizer
from window.window_config import _console_print
from window.window_domain import (
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    MappedWord,
    SentencePart,
    TimedWord,
    WindowTranscript,
)

def word_attr(word: Any, name: str, default: Any = None) -> Any:
    if isinstance(word, dict):
        return word.get(name, default)
    return getattr(word, name, default)


def word_ends_sentence(text: str) -> bool:
    return bool(text.strip().rstrip('"\'”’)]}').endswith((".", "?", "!")))


def words_to_text(words: list[TimedWord]) -> str:
    text = "".join(word.text if word.text.startswith(" ") else " " + word.text for word in words).strip()
    return " ".join(text.split()).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")


def word_ends_sentence(text: str) -> bool:
    return bool(text.strip().rstrip("\"')]}").endswith((".", "?", "!")))


LEFT_ATTACHING_TOKEN_STARTS = set(".,?!;:%)]}…")


def text_ends_sentence(text: str) -> bool:
    return bool(text.strip().rstrip("\"')]}").endswith((".", "?", "!")))


def word_token_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def words_to_mapped_text(words: list[TimedWord]) -> tuple[str, list[MappedWord]]:
    pieces: list[str] = []
    mapped_words: list[MappedWord] = []
    offset = 0
    for word in words:
        token = word_token_text(word.text)
        if not token:
            continue
        separator = "" if not pieces or token[0] in LEFT_ATTACHING_TOKEN_STARTS else " "
        if separator:
            pieces.append(separator)
            offset += len(separator)
        text_start = offset
        pieces.append(token)
        offset += len(token)
        mapped_words.append(MappedWord(word=word, text_start=text_start, text_end=offset))
    return "".join(pieces), mapped_words


def find_sentence_span(text: str, sentence: str, search_start: int) -> tuple[int, int] | None:
    sentence = " ".join(str(sentence or "").split())
    if not sentence:
        return None
    start = text.find(sentence, search_start)
    if start < 0:
        return None
    return start, start + len(sentence)


def word_indexes_for_text_span(mapped_words: list[MappedWord], span_start: int, span_end: int) -> tuple[int, int] | None:
    first_index = None
    last_index = None
    for index, mapped_word in enumerate(mapped_words):
        if mapped_word.text_end <= span_start:
            continue
        if mapped_word.text_start >= span_end:
            break
        if first_index is None:
            first_index = index
        last_index = index
    if first_index is None or last_index is None:
        return None
    return first_index, last_index


def round_optional(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def sentence_boundary_between_words(
    sentence_left: float,
    right: float,
    last_word: TimedWord,
    next_word: TimedWord | None,
    pre_padding_seconds: float,
    post_padding_seconds: float,
    gap_ratio: float,
) -> tuple[float, str, float | None]:
    if next_word is None:
        boundary = max(sentence_left, min(right, last_word.end + post_padding_seconds))
        return boundary, "final_post_padding", None

    gap = float(next_word.start - last_word.end)
    if gap <= 0.0:
        boundary = max(sentence_left, min(right, last_word.end))
        return boundary, "overlap_last_word_end", gap

    padded_boundary = min(last_word.end + post_padding_seconds, next_word.start - pre_padding_seconds)
    padding_zones_overlap = last_word.end + post_padding_seconds > next_word.start - pre_padding_seconds
    if padding_zones_overlap:
        boundary = last_word.end + gap * gap_ratio
        strategy = "gap_ratio"
    else:
        boundary = padded_boundary
        strategy = "post_padding"

    boundary = max(sentence_left, min(right, min(boundary, next_word.start)))
    return boundary, strategy, gap


def sentence_clip_start_for_first_word(
    sentence_left: float,
    right: float,
    first_word: TimedWord,
    pre_padding_seconds: float,
) -> float:
    return max(sentence_left, min(right, first_word.start - pre_padding_seconds))


def split_words_with_stream2sentence(
    words: list[TimedWord],
    left: float,
    right: float,
    unstable_tail_seconds: float,
    final_flush: bool,
    boundary_pre_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    boundary_post_padding_seconds: float = DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    boundary_gap_ratio: float = DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
) -> list[SentencePart]:
    text, mapped_words = words_to_mapped_text(words)
    if not text or not mapped_words:
        return []

    sentence_texts = list(generate_sentences(
        list(text),
        tokenizer="nltk+rule-based",
        language="en",
        auto_context=True,
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
        context_size=12,
        context_size_look_overhead=64,
    ))

    parts: list[SentencePart] = []
    search_start = 0
    sentence_left = left
    for sentence_text in sentence_texts:
        if not final_flush and not text_ends_sentence(sentence_text):
            break
        span = find_sentence_span(text, sentence_text, search_start)
        if span is None:
            continue
        search_start = span[1]
        word_indexes = word_indexes_for_text_span(mapped_words, span[0], span[1])
        if word_indexes is None:
            continue
        _first_word_index, last_word_index = word_indexes
        first_word = mapped_words[_first_word_index].word
        last_word = mapped_words[last_word_index].word
        has_next_word = last_word_index + 1 < len(mapped_words)
        if not final_flush and not has_next_word:
            break
        if not final_flush and right - last_word.end < unstable_tail_seconds:
            break
        next_word = mapped_words[last_word_index + 1].word if has_next_word else None
        if has_next_word:
            boundary, boundary_strategy, gap_to_next_word_seconds = sentence_boundary_between_words(
                sentence_left=sentence_left,
                right=right,
                last_word=last_word,
                next_word=next_word,
                pre_padding_seconds=boundary_pre_padding_seconds,
                post_padding_seconds=boundary_post_padding_seconds,
                gap_ratio=boundary_gap_ratio,
            )
        else:
            boundary, boundary_strategy, gap_to_next_word_seconds = sentence_boundary_between_words(
                sentence_left=sentence_left,
                right=right,
                last_word=last_word,
                next_word=None,
                pre_padding_seconds=boundary_pre_padding_seconds,
                post_padding_seconds=boundary_post_padding_seconds,
                gap_ratio=boundary_gap_ratio,
            )
        clip_start = sentence_clip_start_for_first_word(
            sentence_left=sentence_left,
            right=right,
            first_word=first_word,
            pre_padding_seconds=boundary_pre_padding_seconds,
        )
        if boundary <= clip_start:
            continue
        sentence_words = [
            {
                "text": mapped_words[index].word.text,
                "start": round(float(mapped_words[index].word.start), 4),
                "end": round(float(mapped_words[index].word.end), 4),
                "duration": round(max(0.0, mapped_words[index].word.end - mapped_words[index].word.start), 4),
            }
            for index in range(_first_word_index, last_word_index + 1)
        ]
        spoken_word_seconds = sum(
            max(0.0, mapped_words[index].word.end - mapped_words[index].word.start)
            for index in range(_first_word_index, last_word_index + 1)
        )
        audio_length = max(0.0, boundary - clip_start)
        speech_audio_ratio = spoken_word_seconds / audio_length if audio_length > 0.0 else 0.0
        parts.append(SentencePart(
            sentence_text.strip(),
            clip_start,
            boundary,
            boundary,
            spoken_word_seconds,
            speech_audio_ratio,
            sentence_words,
            first_word.start,
            last_word.end,
            next_word.start if next_word is not None else None,
            gap_to_next_word_seconds,
            boundary_strategy,
            boundary_pre_padding_seconds,
            boundary_post_padding_seconds,
            boundary_gap_ratio,
        ))
        sentence_left = boundary
    return parts


VOCABLE_WORDS = {
    "ah",
    "ba",
    "baa",
    "bah",
    "da",
    "daa",
    "duh",
    "dum",
    "er",
    "erm",
    "hm",
    "hmm",
    "huh",
    "la",
    "mmm",
    "na",
    "oh",
    "pfft",
    "uh",
    "um",
}


def text_content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return [
        word
        for word in words
        if word.strip("'") and word.strip("'") not in VOCABLE_WORDS
    ]


def is_embedding_candidate_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False
    vocables = sum(1 for word in words if word.strip("'") in VOCABLE_WORDS)
    content_words = len(words) - vocables
    if content_words <= 0:
        return False
    has_music_marker = "\u266a" in text or "\u266b" in text
    if has_music_marker and content_words <= 2:
        return False
    if len(words) >= 4 and vocables / max(1, len(words)) >= 0.6:
        return False
    return True


