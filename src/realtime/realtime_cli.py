"""Argument parser composition for realtime capture and validation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from embeddings.embedding_providers import (
    DEFAULT_EMBEDDING_PROVIDER,
    default_embedding_python,
)
from paths import (
    CACHE_DIR,
    CUNK_CANONICAL,
    OUTPUTS_DIR,
    REALTIME_VALIDATION_OUTPUT_DIR,
)
from realtime.realtime_transcript import (
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
)
from window.language_config import default_language_code, language_arg


@dataclass(frozen=True)
class RealtimeConfig:
    """Read-only runtime configuration produced at the parser boundary."""

    _values: Mapping[str, Any]

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "RealtimeConfig":
        return cls(MappingProxyType(dict(vars(namespace))))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--allow-default-input", action="store_true")


def add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
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


def add_speaker_arguments(parser: argparse.ArgumentParser) -> None:
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


def add_trace_validation_arguments(parser: argparse.ArgumentParser) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding-only realtime speaker diarization for a YouTube WASAPI capture."
    )
    for add_arguments in (
        add_server_arguments,
        add_transcription_arguments,
        add_speaker_arguments,
        add_trace_validation_arguments,
    ):
        add_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> RealtimeConfig:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    rt_model_was_explicit = any(
        item == "--rt-model" or item.startswith("--rt-model=")
        for item in raw_argv
    )
    if (
        args.language != "en"
        and not rt_model_was_explicit
        and str(args.rt_model).endswith(".en")
    ):
        args.rt_model = str(args.rt_model)[:-3]
    return RealtimeConfig.from_namespace(args)
