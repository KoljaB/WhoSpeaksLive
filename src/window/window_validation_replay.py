"""Cached replay helpers for validating the live window diarizer path."""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
from typing import Any, Sequence

import numpy as np

from window.window_diarizer import WindowDiarizer
from window.window_config import DEFAULT_SPEAKER_LIBRARY_DIR
from window.window_domain import SentencePart
from window.window_text import is_embedding_candidate_text


@dataclass(frozen=True)
class CachedWindowReplayResult:
    records: list[dict[str, Any]]
    analysis_records: list[dict[str, Any]]
    final_payloads: list[dict[str, Any]]


class CachedReplayEventBus:
    """Small event recorder for optimizer replay; it keeps only scorer-relevant events."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.done = threading.Event()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "done":
            self.done.set()
            return
        if event != "sentence":
            return
        self.records.append({"time": 0.0, "event": event, "payload": dict(payload)})


def make_cached_replay_args(
    config: dict[str, Any] | None = None,
    *,
    base_args: argparse.Namespace | None = None,
    speaker_refinement: bool | None = None,
    speaker_refinement_unknown_tentative: bool | None = None,
    speaker_refinement_unknown_commit: bool | None = None,
    allow_speaker_reassignment: bool | None = None,
) -> argparse.Namespace:
    """Create fast cached-replay args from the live GUI defaults plus candidate config."""
    if base_args is None:
        from window.youtube_window_diarize_gui import parse_args

        old_argv = sys.argv
        try:
            sys.argv = ["youtube_window_diarize_gui"]
            args = parse_args()
        finally:
            sys.argv = old_argv
    else:
        args = copy.copy(base_args)
    for key, value in (config or {}).items():
        setattr(args, str(key), value)
    args.min_embed_seconds = 0.0
    if speaker_refinement is not None:
        args.speaker_refinement = bool(speaker_refinement)
    if speaker_refinement_unknown_tentative is not None:
        args.speaker_refinement_unknown_tentative = bool(speaker_refinement_unknown_tentative)
    if speaker_refinement_unknown_commit is not None:
        args.speaker_refinement_unknown_commit = bool(speaker_refinement_unknown_commit)
    if allow_speaker_reassignment is not None:
        args.allow_speaker_reassignment = bool(allow_speaker_reassignment)
    return args


def cached_sentence_part(sentence: dict[str, Any]) -> SentencePart:
    start = float(sentence.get("start") or sentence.get("video_start_seconds") or 0.0)
    end = float(sentence.get("end") or sentence.get("video_end_seconds") or start)
    duration = max(0.0, end - start)
    spoken = sentence.get("spoken_word_seconds")
    if spoken is None:
        spoken = sentence.get("audio_length_seconds", duration)
    ratio = sentence.get("speech_audio_ratio")
    if ratio is None:
        ratio = float(spoken or 0.0) / duration if duration > 0.0 else 0.0
    next_left = float(sentence.get("next_left") or end)
    return SentencePart(
        text=str(sentence.get("text") or ""),
        start=start,
        end=end,
        next_left=next_left,
        spoken_word_seconds=max(0.0, float(spoken or 0.0)),
        speech_audio_ratio=max(0.0, float(ratio or 0.0)),
        words=list(sentence.get("words") or []),
        first_word_start=sentence.get("first_word_start"),
        last_word_end=sentence.get("last_word_end"),
        next_word_start=sentence.get("next_word_start"),
        gap_to_next_word_seconds=sentence.get("gap_to_next_word_seconds"),
        boundary_strategy=str(sentence.get("boundary_strategy") or "cached_replay"),
    )


def cached_sentence_index(sentence: dict[str, Any], fallback: int) -> int:
    try:
        return int(sentence.get("index", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _install_cached_replay_state(
    controller: WindowDiarizer,
    args: argparse.Namespace,
    bus: CachedReplayEventBus,
) -> None:
    if not hasattr(args, "embedding_provider"):
        setattr(args, "embedding_provider", "cached_replay")
    controller.args = args
    controller.bus = bus
    controller.memory = controller._new_memory()
    controller.live_memory = controller.memory
    controller._live_embedding_separate = False
    controller._live_probability_history = deque(maxlen=max(1, int(getattr(args, "live_speaker_ema_count", 3))))
    controller._speaker_generation = 0
    controller._speaker_lock = threading.Lock()
    controller._speaker_group_name = ""
    controller._speaker_metadata = {}
    controller._seed_profiles = []
    controller._seed_live_profiles = []
    controller.speaker_library_dir = Path(getattr(args, "speaker_library_dir", DEFAULT_SPEAKER_LIBRARY_DIR))
    controller._speaker_last_media_end = {}
    controller._unknown_lock = threading.Lock()
    controller._unknown_sentences = []
    controller._recent_unknown_pair_candidates = deque(maxlen=24)
    controller._sentence_refinement_lock = threading.Lock()
    controller._sentence_refinement_records = {}
    controller._sentence_refinement_run_lock = threading.Lock()
    controller._embedding_jobs = None
    controller._live_memory_update_jobs = None
    controller._live_memory_update_lock = threading.Lock()
    controller._preview_paused = True
    controller.emit_speaker_state = lambda: {}
    controller._maybe_emit_sentence_live_speaker_hint = lambda *_args, **_kwargs: None
    controller._update_live_speaker_memory = lambda *_args, **_kwargs: None


def replay_cached_window_diarizer(
    sentences: Sequence[dict[str, Any]],
    embeddings: Sequence[Any],
    args: argparse.Namespace,
    *,
    emit_done: bool = False,
    defer_speaker_refinement: bool = True,
    max_refinement_passes: int = 8,
) -> CachedWindowReplayResult:
    """Replay cached rows through the live assignment logic without ASR or embedding work.

    The fast optimizer path keeps the exact same assignment functions as live
    playback, but defers prototype refinement so each candidate does not pay
    for repeated all-row prototype passes. Set defer_speaker_refinement=False
    for chronological proof runs that mirror live row timing.
    """
    if len(embeddings) < len(sentences):
        raise ValueError("Expected at least one cached embedding per sentence row.")
    from window.youtube_window_diarize_gui import build_window_validation_records

    bus = CachedReplayEventBus()
    controller = WindowDiarizer.__new__(WindowDiarizer)
    _install_cached_replay_state(controller, args, bus)

    for position, sentence in enumerate(sentences):
        part = cached_sentence_part(dict(sentence))
        index = cached_sentence_index(dict(sentence), position)
        window_left = float(sentence.get("window_left") or part.start)
        window_right = float(sentence.get("window_right") or part.end)
        base_payload = WindowDiarizer._base_payload_from_sentence_part(index, part, window_left, window_right)
        if part.speech_audio_ratio < args.min_speech_audio_ratio:
            bus.emit("sentence", {
                **base_payload,
                "pending": False,
                "unknown_permanent": True,
                "assigned_speaker": None,
                **controller._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "quality": None,
                "assignment_source": "unknown_permanent",
            })
            continue
        if not is_embedding_candidate_text(part.text):
            bus.emit("sentence", {
                **base_payload,
                "pending": False,
                "assigned_speaker": None,
                **controller._speaker_info_for_payload(None),
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
            })
            continue
        controller._apply_sentence_embedding_decision(
            index=index,
            base_payload=base_payload,
            text=part.text,
            embedding=np.asarray(embeddings[position], dtype=np.float32),
            duration_seconds=max(0.0, part.end - part.start),
            emit_status=False,
            run_speaker_refinement=not defer_speaker_refinement,
        )

    controller._revisit_unknown_sentences()
    if defer_speaker_refinement and bool(getattr(args, "speaker_refinement", True)):
        for _ in range(max(1, int(max_refinement_passes))):
            committed_before = sum(
                1
                for record in bus.records
                if record.get("event") == "sentence"
                and (record.get("payload") or {}).get("prototype_reassigned")
                and not (record.get("payload") or {}).get("provisional_assignment")
            )
            controller._refine_speaker_assignments()
            committed_after = sum(
                1
                for record in bus.records
                if record.get("event") == "sentence"
                and (record.get("payload") or {}).get("prototype_reassigned")
                and not (record.get("payload") or {}).get("provisional_assignment")
            )
            if committed_after == committed_before:
                break
    if emit_done:
        bus.emit("done", {"message": "Cached window replay stopped."})
    analysis_records, final_payloads = build_window_validation_records(bus.records)
    return CachedWindowReplayResult(
        records=list(bus.records),
        analysis_records=analysis_records,
        final_payloads=final_payloads,
    )


def replay_cached_window_score(
    sentences: Sequence[dict[str, Any]],
    embeddings: Sequence[Any],
    args: argparse.Namespace,
    canonical_segments: list[dict[str, Any]],
    *,
    match_mode: str = "auto",
) -> dict[str, Any]:
    """Replay cached rows through the live path and score the resulting committed events."""
    from realtime.realtime_speakerdiarize import analyze_trace_against_canonical

    replay = replay_cached_window_diarizer(sentences, embeddings, args)
    return analyze_trace_against_canonical(
        replay.analysis_records,
        canonical_segments,
        match_mode=match_mode,
    )
