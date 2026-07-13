"""Pure canonical-transcript parsing and matching primitives."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
