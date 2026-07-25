"""Cached replay helpers for validating the live window diarizer path."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np

from window.window_diarizer import WindowDiarizer
from window.window_config import DEFAULT_SPEAKER_LIBRARY_DIR
from window.window_domain import SentencePart
from window.window_runtime_config import WindowConfig
from window.window_text import is_embedding_candidate_text


@dataclass(frozen=True)
class CachedWindowReplayResult:
    records: list[dict[str, Any]]
    analysis_records: list[dict[str, Any]]
    final_payloads: list[dict[str, Any]]
    profile_events: list[dict[str, Any]]
    final_profiles: list[dict[str, Any]]


class CachedReplayEventBus:
    """Small event recorder for optimizer replay; it keeps only scorer-relevant events."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.done = threading.Event()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if event == "done":
            self.done.set()
            return
        if event not in {"sentence", "live_speaker_profile_snapshot"}:
            return
        self.records.append({"time": 0.0, "event": event, "payload": dict(payload)})


def make_cached_replay_args(
    config: dict[str, Any] | None = None,
    *,
    base_args: argparse.Namespace | WindowConfig | None = None,
    speaker_refinement: bool | None = None,
    speaker_refinement_unknown_tentative: bool | None = None,
    speaker_refinement_unknown_commit: bool | None = None,
    allow_speaker_reassignment: bool | None = None,
) -> WindowConfig:
    """Create fast cached-replay args from the live GUI defaults plus candidate config."""
    if base_args is None:
        from window.youtube_window_diarize_gui import parse_args

        args = parse_args([])
    else:
        if isinstance(base_args, WindowConfig):
            args = base_args
        elif isinstance(base_args, argparse.Namespace):
            args = WindowConfig.from_namespace(base_args)
        else:
            args = WindowConfig.from_mapping(base_args)
    updates = {str(key): value for key, value in (config or {}).items()}
    updates["min_embed_seconds"] = 0.0
    if speaker_refinement is not None:
        updates["speaker_refinement"] = bool(speaker_refinement)
    if speaker_refinement_unknown_tentative is not None:
        updates["speaker_refinement_unknown_tentative"] = bool(speaker_refinement_unknown_tentative)
    if speaker_refinement_unknown_commit is not None:
        updates["speaker_refinement_unknown_commit"] = bool(speaker_refinement_unknown_commit)
    if allow_speaker_reassignment is not None:
        updates["allow_speaker_reassignment"] = bool(allow_speaker_reassignment)
    return args.with_updates(**updates)


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
        asr_review=dict(sentence.get("asr_review") or {}),
    )


def cached_sentence_index(sentence: dict[str, Any], fallback: int) -> int:
    try:
        return int(sentence.get("index", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _install_cached_replay_state(
    controller: WindowDiarizer,
    args: WindowConfig,
    bus: CachedReplayEventBus,
) -> None:
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
    controller._playback_lock = threading.Lock()
    controller._playback_time = 0.0
    controller.duration = float("inf")
    controller._streaming_audio = True
    controller._playback_clock_started_at = None
    controller._last_playback_jump_warning_at = 0.0
    controller.emit_speaker_state = lambda: {}
    controller._maybe_emit_sentence_live_speaker_hint = lambda *_args, **_kwargs: None
    controller._update_live_speaker_memory = lambda *_args, **_kwargs: None


class CachedReplayDiarizer(WindowDiarizer):
    """Assignment-only replay collaborator with an explicit bounded lifecycle."""

    def __init__(self, args: WindowConfig, bus: CachedReplayEventBus) -> None:
        _install_cached_replay_state(self, args, bus)


def replay_cached_window_diarizer(
    sentences: Sequence[dict[str, Any]],
    embeddings: Sequence[Any],
    args: argparse.Namespace | WindowConfig,
    *,
    emit_done: bool = False,
    defer_speaker_refinement: bool = True,
    max_refinement_passes: int = 8,
    live_profile_embeddings: Sequence[Any] | None = None,
    live_profile_provider: str | None = None,
    profile_availability_lag_seconds: float = 0.0,
) -> CachedWindowReplayResult:
    """Replay cached rows through the live assignment logic without ASR or embedding work.

    The fast optimizer path keeps the exact same assignment functions as live
    playback, but defers prototype refinement so each candidate does not pay
    for repeated all-row prototype passes. Set defer_speaker_refinement=False
    for chronological proof runs that mirror live row timing.
    """
    if len(embeddings) < len(sentences):
        raise ValueError("Expected at least one cached embedding per sentence row.")
    if live_profile_embeddings is not None and len(live_profile_embeddings) < len(sentences):
        raise ValueError("Expected at least one cached live-profile embedding per sentence row.")
    if live_profile_embeddings is not None and not str(live_profile_provider or "").strip():
        raise ValueError("live_profile_provider is required with live_profile_embeddings.")
    from window.youtube_window_diarize_gui import build_window_validation_records

    replay_args = args if isinstance(args, WindowConfig) else WindowConfig.from_namespace(args)
    bus = CachedReplayEventBus()
    controller = CachedReplayDiarizer(replay_args, bus)
    profile_events: list[dict[str, Any]] = []
    if live_profile_embeddings is not None:
        controller._live_embedding_separate = True
        controller.live_memory = controller._new_memory()

    for position, sentence in enumerate(sentences):
        part = cached_sentence_part(dict(sentence))
        available_at = max(
            0.0,
            float(part.end) + max(0.0, float(profile_availability_lag_seconds)),
        )
        with controller._playback_lock:
            controller._playback_time = available_at
        index = cached_sentence_index(dict(sentence), position)
        window_left = float(sentence.get("window_left") or part.start)
        window_right = float(sentence.get("window_right") or part.end)
        base_payload = WindowDiarizer._base_payload_from_sentence_part(index, part, window_left, window_right)
        if part.speech_audio_ratio < replay_args.min_speech_audio_ratio:
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
            payload = {
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
                "assignment_source": "non_embedding_candidate",
            }
            bus.emit("sentence", payload)
            controller._record_unknown_refinement_candidate(
                index,
                base_payload,
                max(0.0, part.end - part.start),
                payload,
            )
            continue
        sentence_payload = controller._apply_sentence_embedding_decision(
            index=index,
            base_payload=base_payload,
            text=part.text,
            embedding=np.asarray(embeddings[position], dtype=np.float32),
            duration_seconds=max(0.0, part.end - part.start),
            emit_status=False,
            run_speaker_refinement=not defer_speaker_refinement,
        )
        if live_profile_embeddings is not None:
            speaker_id = sentence_payload.get("assigned_speaker")
            if speaker_id:
                from window.live_profile_tape import emit_live_profile_snapshot

                controller.live_memory.upsert_profile(
                    str(speaker_id),
                    np.asarray(live_profile_embeddings[position], dtype=np.float32),
                    duration_seconds=max(0.0, part.end - part.start),
                    sentence_count=1,
                )
                event = emit_live_profile_snapshot(
                    controller,
                    controller.live_memory,
                    str(speaker_id),
                    str(live_profile_provider),
                    source="cached_final_sentence_live_profile_replay",
                    sentence_start=part.start,
                    sentence_end=part.end,
                )
                if event is not None:
                    profile_events.append(event)

    controller._revisit_unknown_sentences()
    if defer_speaker_refinement and bool(getattr(replay_args, "speaker_refinement", True)):
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
        controller._apply_delayed_multirow_clustering()
        controller._refine_speaker_assignments()
        controller._merge_tiny_fragmented_speaker_profiles()
        controller._merge_terminal_promotional_outro()
        controller._split_long_low_confidence_retro_assignments()
        controller._fill_unknown_same_speaker_islands()
        controller._fill_unknown_previous_speaker_tails()
        controller._fill_unknown_next_speaker_heads()
    elif bool(getattr(args, "speaker_refinement", True)):
        controller._finalize_speaker_refinement()
    if emit_done:
        bus.emit("done", {"message": "Cached window replay stopped."})
    analysis_records, final_payloads = build_window_validation_records(bus.records)
    return CachedWindowReplayResult(
        records=list(bus.records),
        analysis_records=analysis_records,
        final_payloads=final_payloads,
        profile_events=profile_events,
        final_profiles=controller.memory.export_profiles(),
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
    from realtime.trace_analysis import analyze_trace_against_canonical

    replay = replay_cached_window_diarizer(sentences, embeddings, args)
    return analyze_trace_against_canonical(
        replay.analysis_records,
        canonical_segments,
        match_mode=match_mode,
    )
