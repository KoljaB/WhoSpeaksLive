"""Pure trace alignment and session-aware analysis."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from realtime.canonical_transcript import (
    best_canonical_text_match,
    canonical_overlap,
    lcs_length,
    lcs_speaker_matches_by_final,
    text_tokens,
)

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
