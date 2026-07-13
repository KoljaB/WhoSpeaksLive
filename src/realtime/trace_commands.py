"""JSONL trace command I/O and human-readable reporting."""

from __future__ import annotations

import json
from pathlib import Path

from realtime.canonical_transcript import read_canonical_segments
from realtime.trace_analysis import (
    analyze_trace_against_canonical,
    filter_trace_records_by_session,
    trace_session_ids,
)


def read_trace_records(path: Path) -> list[dict]:
    """Read valid JSON-object records while tolerating partial trace lines."""

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def analyze_trace(
    path: Path,
    canonical_path: Path | None = None,
    output_path: Path | None = None,
    summary_only: bool = False,
    match_mode: str = "auto",
    trace_session: str = "latest",
) -> int:
    records = read_trace_records(path)

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
