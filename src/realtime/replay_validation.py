"""RealtimeSTT validation driven through the public external-feed API."""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from common.audio_utils import INT16_MAX_ABS_VALUE, load_audio_file
from paths import OUTPUTS_DIR
from realtime.canonical_transcript import read_canonical_segments
from realtime.realtime_capture import EventBus, RealtimeCapture, TraceLogger
from realtime.realtime_cli import RealtimeConfig
from realtime.trace_analysis import analyze_trace_against_canonical
from realtime.trace_commands import read_trace_records


def validate_cunk_realtime_replay(args: RealtimeConfig) -> int:
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
    capture = RealtimeCapture(args, bus)

    audio, sample_rate = load_audio_file(args.validation_audio)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * INT16_MAX_ABS_VALUE).astype(np.int16)
    chunk_samples = max(1, int(sample_rate * args.replay_chunk_seconds))
    replay_speed = max(0.1, float(args.replay_speed))
    started_at = time.monotonic()
    try:
        with capture.external_feed(
            session_id="replay",
            media_id="local-replay",
        ) as feed:
            feed.report_status(
                f"Replay feeding {args.validation_audio} at {replay_speed:.1f}x."
            )
            pacing_started_at = time.monotonic()
            for start in range(0, len(audio_int16), chunk_samples):
                end = min(len(audio_int16), start + chunk_samples)
                chunk = audio_int16[start:end]
                feed.feed_audio(
                    chunk,
                    original_sample_rate=sample_rate,
                    media_end_seconds=end / float(sample_rate),
                )
                if args.replay_sleep:
                    target = pacing_started_at + (end / float(sample_rate)) / replay_speed
                    time.sleep(max(0.0, target - time.monotonic()))

            silence_seconds = max(0.5, float(args.replay_trailing_silence_seconds))
            silence = np.zeros(int(sample_rate * silence_seconds), dtype=np.int16)
            for start in range(0, len(silence), chunk_samples):
                end = min(len(silence), start + chunk_samples)
                chunk = silence[start:end]
                feed.feed_audio(
                    chunk,
                    original_sample_rate=sample_rate,
                    media_end_seconds=(len(audio_int16) + end) / float(sample_rate),
                )
                if args.replay_sleep:
                    target = pacing_started_at + (
                        (len(audio_int16) + end) / float(sample_rate)
                    ) / replay_speed
                    time.sleep(max(0.0, target - time.monotonic()))

            feed.report_status(
                "Replay audio feed complete; draining final transcripts."
            )
            feed.finish(
                transcript_drain_seconds=max(0.0, float(args.replay_drain_seconds)),
                embedding_drain_seconds=max(
                    0.0, float(args.replay_embedding_drain_seconds)
                ),
                emit_status=False,
            )
    finally:
        capture.shutdown()

    elapsed = time.monotonic() - started_at
    records = read_trace_records(trace.path)
    canonical_segments = read_canonical_segments(args.validation_canonical)
    analysis_match_mode = "timestamp" if abs(replay_speed - 1.0) < 0.05 else "text"
    summary = analyze_trace_against_canonical(
        records,
        canonical_segments,
        match_mode=analysis_match_mode,
    )
    summary.update(
        {
            "trace": str(trace.path),
            "audio": str(args.validation_audio),
            "canonical": str(args.validation_canonical),
            "elapsed_seconds": round(elapsed, 4),
            "replay_speed": replay_speed,
        }
    )
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Replay trace: {trace.path}", flush=True)
    print(f"Replay analysis output: {args.validation_output}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Match mode: {summary['match_mode']}", flush=True)
    print(f"Final segments: {summary['final_segments']}", flush=True)
    print(f"Resolved segments: {summary['resolved_segments']}", flush=True)
    print(f"Timestamped segments: {summary['timestamped_segments']}", flush=True)
    print(
        f"Live final words: {summary['live_final_words']} / "
        f"canonical {summary['canonical_words']}",
        flush=True,
    )
    print(
        "Text recall/precision by LCS: "
        f"{summary['text_recall']:.3f} / {summary['text_precision']:.3f}",
        flush=True,
    )
    print(f"Assigned counts: {summary['assigned_counts']}", flush=True)
    print(f"Profile map: {summary['profile_map']}", flush=True)
    print(f"Unknown segments: {summary['unknown_segments']}", flush=True)
    print(
        "Live segment accuracy after profile mapping: "
        f"{summary['segment_accuracy']:.3f}",
        flush=True,
    )
    print(
        "Live duration accuracy after profile mapping: "
        f"{summary['duration_accuracy']:.3f}",
        flush=True,
    )
    return 0
