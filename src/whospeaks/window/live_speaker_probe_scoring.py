"""Score live speaker probe events against a canonical diarization timeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


Interval = tuple[float, float]


def read_canonical_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [
            {
                "speaker": str(segment["speaker"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text") or ""),
            }
            for segment in data
        ]
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return [
            {
                "speaker": str(segment["speaker_id"]),
                "start": float(segment["start_sec"]),
                "end": float(segment["end_sec"]),
                "text": str(segment.get("text") or ""),
            }
            for segment in data["segments"]
        ]
    raise ValueError(f"Could not read canonical segments from {path}")


def read_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def positive_interval(start: Any, end: Any) -> Interval | None:
    try:
        left = float(start)
        right = float(end)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(left) and math.isfinite(right)) or right <= left:
        return None
    return left, right


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_total(intervals: list[Interval]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def overlap_seconds(left: list[Interval], right: list[Interval]) -> float:
    total = 0.0
    i = 0
    j = 0
    left_sorted = merge_intervals(left)
    right_sorted = merge_intervals(right)
    while i < len(left_sorted) and j < len(right_sorted):
        start = max(left_sorted[i][0], right_sorted[j][0])
        end = min(left_sorted[i][1], right_sorted[j][1])
        if end > start:
            total += end - start
        if left_sorted[i][1] <= right_sorted[j][1]:
            i += 1
        else:
            j += 1
    return total


def subtract_intervals(interval: Interval, blockers: list[Interval]) -> list[Interval]:
    remaining = [interval]
    for block_start, block_end in merge_intervals(blockers):
        next_remaining: list[Interval] = []
        for start, end in remaining:
            if block_end <= start or block_start >= end:
                next_remaining.append((start, end))
                continue
            if block_start > start:
                next_remaining.append((start, min(block_start, end)))
            if block_end < end:
                next_remaining.append((max(block_end, start), end))
        remaining = next_remaining
        if not remaining:
            break
    return remaining


def intervals_for_speaker(canonical_segments: list[dict[str, Any]]) -> dict[str, list[Interval]]:
    by_speaker: dict[str, list[Interval]] = defaultdict(list)
    for segment in canonical_segments:
        interval = positive_interval(segment.get("start"), segment.get("end"))
        if interval is None:
            continue
        by_speaker[str(segment["speaker"])].append(interval)
    return {speaker: merge_intervals(intervals) for speaker, intervals in by_speaker.items()}


def live_speaker_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "live_speaker":
            continue
        payload = record.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        speaker = str(payload.get("speaker_id") or payload.get("assigned_speaker") or "")
        if not speaker or speaker == "UNKNOWN":
            continue
        interval = positive_interval(payload.get("start"), payload.get("end"))
        if interval is None:
            continue
        events.append({
            "time": float(record.get("time") or 0.0),
            "speaker": speaker,
            "start": interval[0],
            "end": interval[1],
            "payload": payload,
        })
    events.sort(key=lambda item: (float(item["time"]), float(item["start"]), float(item["end"])))
    return events


def live_speaker_clear_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "live_speaker_clear":
            continue
        payload = record.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        events.append({
            "time": float(record.get("time") or 0.0),
            "speaker": str(payload.get("speaker_id") or ""),
            "reason": str(payload.get("reason") or ""),
            "payload": payload,
        })
    events.sort(key=lambda item: float(item["time"]))
    return events


def sidebar_counted_live_slices(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    last_right: float | None = None
    for event in events:
        start = float(event["start"])
        end = float(event["end"])
        previous_right = start if last_right is None else last_right
        uncovered_start = max(start, previous_right)
        if end > uncovered_start:
            slices.append({
                "speaker": event["speaker"],
                "start": round(uncovered_start, 4),
                "end": round(end, 4),
                "duration_seconds": round(end - uncovered_start, 4),
                "window_start": round(start, 4),
                "window_end": round(end, 4),
                "event_time": event["time"],
                "payload": event["payload"],
            })
        last_right = max(previous_right, end)
    return slices


def active_live_slices(
    live_events: list[dict[str, Any]],
    clear_events: list[dict[str, Any]],
    replay_start: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    if replay_start is None:
        return []
    start_time, replay_speed = replay_start
    timeline: list[dict[str, Any]] = []
    for event in live_events:
        try:
            hold_seconds = max(0.0, float((event.get("payload") or {}).get("hold_seconds") or 0.0))
        except (TypeError, ValueError):
            hold_seconds = 0.0
        event_playback_time = max(0.0, (float(event["time"]) - start_time) * replay_speed)
        timeline.append({
            "kind": "speaker",
            "time": float(event["time"]),
            "playback_time": event_playback_time,
            "speaker": str(event["speaker"]),
            "hold_seconds": hold_seconds,
        })
    for event in clear_events:
        event_playback_time = max(0.0, (float(event["time"]) - start_time) * replay_speed)
        timeline.append({
            "kind": "clear",
            "time": float(event["time"]),
            "playback_time": event_playback_time,
            "speaker": str(event["speaker"]),
            "reason": str(event["reason"]),
        })
    timeline.sort(key=lambda item: (float(item["time"]), 0 if item["kind"] == "clear" else 1))

    slices: list[dict[str, Any]] = []
    active_speaker = ""
    active_start: float | None = None
    active_until = 0.0
    for event in timeline:
        now = float(event["playback_time"])
        if event["kind"] == "clear":
            clear_speaker = str(event.get("speaker") or "")
            if active_speaker and active_start is not None and (
                not clear_speaker or clear_speaker == active_speaker
            ):
                close_at = min(now, active_until)
                if close_at > active_start:
                    slices.append({
                        "speaker": active_speaker,
                        "start": round(active_start, 4),
                        "end": round(close_at, 4),
                        "duration_seconds": round(close_at - active_start, 4),
                    })
                active_speaker = ""
                active_start = None
                active_until = now
            continue
        if active_speaker and active_start is not None:
            close_at = min(now, active_until)
            if close_at > active_start:
                slices.append({
                    "speaker": active_speaker,
                    "start": round(active_start, 4),
                    "end": round(close_at, 4),
                    "duration_seconds": round(close_at - active_start, 4),
                })
        active_speaker = str(event["speaker"])
        active_start = now
        active_until = now + float(event["hold_seconds"])
    if active_speaker and active_start is not None and active_until > active_start:
        slices.append({
            "speaker": active_speaker,
            "start": round(active_start, 4),
            "end": round(active_until, 4),
            "duration_seconds": round(active_until - active_start, 4),
        })
    return [item for item in slices if float(item["end"]) > float(item["start"]) and float(item["start"]) >= 0.0]


def build_profile_map(
    slices: list[dict[str, Any]],
    canonical_by_speaker: dict[str, list[Interval]],
) -> dict[str, str]:
    overlap_by_profile: dict[str, Counter[str]] = defaultdict(Counter)
    for item in slices:
        interval = [(float(item["start"]), float(item["end"]))]
        for canonical_speaker, canonical_intervals in canonical_by_speaker.items():
            overlap = overlap_seconds(interval, canonical_intervals)
            if overlap > 0.0:
                overlap_by_profile[str(item["speaker"])][canonical_speaker] += overlap
    return {
        profile: counter.most_common(1)[0][0]
        for profile, counter in overlap_by_profile.items()
        if counter
    }


def missed_canonical_gaps(
    canonical_segments: list[dict[str, Any]],
    live_intervals: list[Interval],
    limit: int = 20,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    blockers = merge_intervals(live_intervals)
    for segment in canonical_segments:
        interval = positive_interval(segment.get("start"), segment.get("end"))
        if interval is None:
            continue
        for start, end in subtract_intervals(interval, blockers):
            duration = end - start
            if duration <= 0.0:
                continue
            gaps.append({
                "speaker": str(segment["speaker"]),
                "start": round(start, 4),
                "end": round(end, 4),
                "duration_seconds": round(duration, 4),
                "segment_start": round(float(interval[0]), 4),
                "segment_end": round(float(interval[1]), 4),
                "text": str(segment.get("text") or ""),
            })
    gaps.sort(key=lambda item: float(item["duration_seconds"]), reverse=True)
    return gaps[:limit]


def live_assignment_gaps(slices: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    ordered = sorted(slices, key=lambda item: (float(item["start"]), float(item["end"])))
    for previous, current in zip(ordered, ordered[1:]):
        start = float(previous["end"])
        end = float(current["start"])
        if end <= start:
            continue
        gaps.append({
            "start": round(start, 4),
            "end": round(end, 4),
            "duration_seconds": round(end - start, 4),
            "previous_speaker": str(previous["speaker"]),
            "next_speaker": str(current["speaker"]),
        })
    gaps.sort(key=lambda item: float(item["duration_seconds"]), reverse=True)
    return gaps[:limit]


def live_speaker_switch_count(slices: list[dict[str, Any]]) -> int:
    count = 0
    previous = None
    for item in sorted(slices, key=lambda value: (float(value["start"]), float(value["end"]))):
        speaker = str(item["speaker"])
        if previous is not None and speaker != previous:
            count += 1
        previous = speaker
    return count


def replay_start_record(records: list[dict[str, Any]]) -> tuple[float, float] | None:
    for record in records:
        if record.get("event") != "validation_replay_start":
            continue
        payload = record.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        try:
            return float(record["time"]), float(payload.get("replay_speed") or 1.0)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.9)) - 1)
    return {
        "min": round(ordered[0], 4),
        "median": round(float(median(ordered)), 4),
        "p90": round(ordered[p90_index], 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def f1_score(recall: float, precision: float) -> float:
    if recall <= 0.0 or precision <= 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def score_live_speaker_probe(records: list[dict[str, Any]], canonical_segments: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_by_speaker = intervals_for_speaker(canonical_segments)
    canonical_intervals = merge_intervals([
        interval
        for intervals in canonical_by_speaker.values()
        for interval in intervals
    ])
    canonical_total = interval_total(canonical_intervals)
    raw_events = live_speaker_events(records)
    clear_events = live_speaker_clear_events(records)
    slices = sidebar_counted_live_slices(raw_events)
    live_intervals = [(float(item["start"]), float(item["end"])) for item in slices]
    live_total = interval_total(live_intervals)
    speech_overlap = overlap_seconds(live_intervals, canonical_intervals)
    profile_map = build_profile_map(slices, canonical_by_speaker)
    replay_start = replay_start_record(records)
    active_slices = active_live_slices(raw_events, clear_events, replay_start)
    active_intervals = [(float(item["start"]), float(item["end"])) for item in active_slices]
    active_total = interval_total(active_intervals)
    active_overlap = overlap_seconds(active_intervals, canonical_intervals)
    active_profile_map = build_profile_map(active_slices, canonical_by_speaker)
    active_correct_overlap = 0.0
    for item in active_slices:
        mapped = active_profile_map.get(str(item["speaker"]))
        if not mapped:
            continue
        active_correct_overlap += overlap_seconds(
            [(float(item["start"]), float(item["end"]))],
            canonical_by_speaker.get(mapped, []),
        )
    active_correct_coverage = safe_ratio(active_correct_overlap, canonical_total)
    active_correct_precision = safe_ratio(active_correct_overlap, active_total)

    correct_overlap = 0.0
    slice_rows: list[dict[str, Any]] = []
    lag_values: list[float] = []
    for index, item in enumerate(slices):
        interval = [(float(item["start"]), float(item["end"]))]
        canonical_overlaps = {
            speaker: round(overlap_seconds(interval, intervals), 4)
            for speaker, intervals in canonical_by_speaker.items()
        }
        canonical_overlaps = {
            speaker: value
            for speaker, value in canonical_overlaps.items()
            if value > 0.0
        }
        mapped = profile_map.get(str(item["speaker"]))
        correct = overlap_seconds(interval, canonical_by_speaker.get(mapped or "", [])) if mapped else 0.0
        correct_overlap += correct
        row = {
            "index": index,
            "speaker": item["speaker"],
            "mapped_canonical_speaker": mapped,
            "start": item["start"],
            "end": item["end"],
            "duration_seconds": item["duration_seconds"],
            "window_start": item["window_start"],
            "window_end": item["window_end"],
            "canonical_overlaps": canonical_overlaps,
            "correct_overlap_seconds": round(correct, 4),
        }
        if replay_start is not None:
            start_time, replay_speed = replay_start
            playback_time = max(0.0, (float(item["event_time"]) - start_time) * replay_speed)
            lag = playback_time - float(item["window_end"])
            lag_values.append(lag)
            row["event_playback_time"] = round(playback_time, 4)
            row["lag_after_window_end_seconds"] = round(lag, 4)
        slice_rows.append(row)
    correct_coverage = safe_ratio(correct_overlap, canonical_total)
    correct_precision = safe_ratio(correct_overlap, live_total)

    per_speaker: dict[str, Any] = {}
    for canonical_speaker, intervals in sorted(canonical_by_speaker.items()):
        total = interval_total(intervals)
        any_detected = overlap_seconds(intervals, live_intervals)
        correct_detected = 0.0
        for item in slices:
            if profile_map.get(str(item["speaker"])) != canonical_speaker:
                continue
            correct_detected += overlap_seconds([(float(item["start"]), float(item["end"]))], intervals)
        per_speaker[canonical_speaker] = {
            "canonical_seconds": round(total, 4),
            "any_live_detected_seconds": round(any_detected, 4),
            "any_live_coverage": round(safe_ratio(any_detected, total), 6),
            "correct_live_detected_seconds": round(correct_detected, 4),
            "correct_live_coverage": round(safe_ratio(correct_detected, total), 6),
        }

    return {
        "canonical_speech_seconds": round(canonical_total, 4),
        "raw_live_speaker_event_count": len(raw_events),
        "live_speaker_clear_event_count": len(clear_events),
        "sidebar_counted_live_slice_count": len(slices),
        "live_speaker_switch_count": live_speaker_switch_count(slices),
        "sidebar_counted_live_seconds": round(live_total, 4),
        "any_live_detected_canonical_seconds": round(speech_overlap, 4),
        "any_live_speech_coverage": round(safe_ratio(speech_overlap, canonical_total), 6),
        "live_speech_precision": round(safe_ratio(speech_overlap, live_total), 6),
        "correct_live_detected_canonical_seconds": round(correct_overlap, 4),
        "correct_live_speaker_coverage": round(correct_coverage, 6),
        "correct_live_speaker_precision": round(correct_precision, 6),
        "correct_live_speaker_f1": round(f1_score(correct_coverage, correct_precision), 6),
        "correct_when_live_overlaps_speech": round(safe_ratio(correct_overlap, speech_overlap), 6),
        "active_live_seconds": round(active_total, 4),
        "active_any_live_detected_canonical_seconds": round(active_overlap, 4),
        "active_any_live_speech_coverage": round(safe_ratio(active_overlap, canonical_total), 6),
        "active_live_speech_precision": round(safe_ratio(active_overlap, active_total), 6),
        "active_correct_live_detected_canonical_seconds": round(active_correct_overlap, 4),
        "active_correct_live_speaker_coverage": round(active_correct_coverage, 6),
        "active_correct_live_speaker_precision": round(active_correct_precision, 6),
        "active_correct_live_speaker_f1": round(f1_score(active_correct_coverage, active_correct_precision), 6),
        "active_correct_when_live_overlaps_speech": round(safe_ratio(active_correct_overlap, active_overlap), 6),
        "active_wrong_speaker_during_speech_seconds": round(max(0.0, active_overlap - active_correct_overlap), 4),
        "active_profile_map": active_profile_map,
        "missed_canonical_speech_seconds": round(max(0.0, canonical_total - speech_overlap), 4),
        "identity_error_or_wrong_speaker_seconds": round(max(0.0, speech_overlap - correct_overlap), 4),
        "live_assignment_outside_canonical_speech_seconds": round(max(0.0, live_total - speech_overlap), 4),
        "profile_map": profile_map,
        "per_canonical_speaker": per_speaker,
        "lag_after_window_end_seconds": quantiles(lag_values),
        "largest_missed_canonical_speech_gaps": missed_canonical_gaps(canonical_segments, live_intervals),
        "largest_live_assignment_gaps": live_assignment_gaps(slices),
        "active_live_slices": active_slices,
        "live_slices": slice_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score live_speaker probe trace events against canonical diarization.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_trace(args.trace)
    canonical_segments = read_canonical_segments(args.canonical)
    summary = score_live_speaker_probe(records, canonical_segments)
    summary["trace"] = str(args.trace)
    summary["canonical"] = str(args.canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Live speaker probe score: {args.output}")
    print(f"Raw live_speaker events: {summary['raw_live_speaker_event_count']}")
    print(f"Counted live seconds: {summary['sidebar_counted_live_seconds']:.2f}s")
    print(
        "Any-speaker coverage: "
        f"{summary['any_live_detected_canonical_seconds']:.2f}s / "
        f"{summary['canonical_speech_seconds']:.2f}s "
        f"({summary['any_live_speech_coverage']:.3f})"
    )
    print(
        "Correct speaker coverage: "
        f"{summary['correct_live_detected_canonical_seconds']:.2f}s / "
        f"{summary['canonical_speech_seconds']:.2f}s "
        f"({summary['correct_live_speaker_coverage']:.3f})"
    )
    print(
        "Active correct speaker coverage: "
        f"{summary['active_correct_live_detected_canonical_seconds']:.2f}s / "
        f"{summary['canonical_speech_seconds']:.2f}s "
        f"({summary['active_correct_live_speaker_coverage']:.3f})"
    )
    print(f"Correct while live overlaps speech: {summary['correct_when_live_overlaps_speech']:.3f}")
    print(f"Lag after window end: {summary['lag_after_window_end_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
