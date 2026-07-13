"""Compatibility entrypoint for realtime diarization and validation.

Implementation lives in focused modules; this module retains the historical
``python -m`` target and public imports used by launchers and embedding helpers.
"""

from __future__ import annotations

import sys

from common.audio_utils import (
    json_dumps,
    load_audio_file,
    pad_audio,
    trim_silence,
    write_wav,
)
from embeddings.embedding_providers import (
    EmbeddingSubprocessClient,
    default_embedding_python,
)
from realtime.canonical_transcript import (
    best_canonical_text_match,
    canonical_overlap,
    lcs_alignment,
    lcs_length,
    lcs_speaker_matches_by_final,
    normalized_match_text,
    read_canonical_segments,
    summarize_canonical_gap_groups,
    text_tokens,
    token_jaccard,
)
from realtime.offline_validation import (
    embed_audio_with_client,
    transcribe_validation_audio_with_realtimestt,
    validate_cunk,
    validate_cunk_word_splits,
)
from realtime.external_feed import ExternalAudioFeed
from realtime.realtime_capture import (
    EventBus,
    RealtimeCapture,
    TraceLogger,
    YouTubeWasapiController,
    extract_youtube_video_id,
)
from realtime.realtime_cli import RealtimeConfig, parse_args as _parse_args
from realtime.realtime_command import run_realtime_command
from realtime.replay_validation import validate_cunk_realtime_replay
from realtime.trace_analysis import (
    analyze_trace_against_canonical,
    filter_trace_records_by_session,
    trace_record_session_id,
    trace_session_ids,
)
from realtime.trace_commands import analyze_trace, read_trace_records


def parse_args(argv: list[str] | None = None) -> RealtimeConfig:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    return _parse_args(raw_argv)


def main(argv: list[str] | None = None) -> int:
    return run_realtime_command(parse_args(argv))


__all__ = [
    "EmbeddingSubprocessClient",
    "EventBus",
    "ExternalAudioFeed",
    "RealtimeCapture",
    "RealtimeConfig",
    "TraceLogger",
    "YouTubeWasapiController",
    "analyze_trace",
    "analyze_trace_against_canonical",
    "best_canonical_text_match",
    "canonical_overlap",
    "default_embedding_python",
    "embed_audio_with_client",
    "extract_youtube_video_id",
    "filter_trace_records_by_session",
    "json_dumps",
    "lcs_alignment",
    "lcs_length",
    "lcs_speaker_matches_by_final",
    "load_audio_file",
    "main",
    "normalized_match_text",
    "pad_audio",
    "parse_args",
    "read_canonical_segments",
    "read_trace_records",
    "summarize_canonical_gap_groups",
    "text_tokens",
    "token_jaccard",
    "trace_record_session_id",
    "trace_session_ids",
    "transcribe_validation_audio_with_realtimestt",
    "trim_silence",
    "validate_cunk",
    "validate_cunk_realtime_replay",
    "validate_cunk_word_splits",
    "write_wav",
]


if __name__ == "__main__":
    raise SystemExit(main())
