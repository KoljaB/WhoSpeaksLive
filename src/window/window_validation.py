"""Browser-synced growing-window diarization experiment.

No RealtimeSTT is used here. The backend periodically transcribes the current
audio window with faster-whisper large-v2, emits confirmed complete sentences,
and clusters one embedding per emitted sentence.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, quote, unquote, urlparse


def _configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def _safe_console_text(text: object) -> str:
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _console_print(text: object) -> None:
    print(_safe_console_text(text), flush=True)


_configure_console_output()

if __name__ == "__main__":
    _console_print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting youtube_window_diarize_gui.py; importing dependencies.",
    )

import numpy as np

from paths import CACHE_DIR, PROJECT_ROOT, VENDOR_DIR

ROOT = PROJECT_ROOT
os.environ.setdefault("NLTK_DATA", str(CACHE_DIR / "nltk"))
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from realtime.realtime_speakerdiarize import (  # noqa: E402
    EmbeddingSubprocessClient,
    default_embedding_python,
    json_dumps,
    load_audio_file,
    pad_audio,
    trim_silence,
    write_wav,
)
from speakers.speaker_embedding_cluster import SpeakerMemory  # noqa: E402
from window.speaker_color_allocation import SpeakerColorAllocator  # noqa: E402
from stream2sentence import generate_sentences, init_tokenizer  # noqa: E402
from replay.youtube_local_filefeed_replay import (  # noqa: E402
    DEFAULT_URL,
    DEFAULT_WORK_DIR,
)
from window.window_domain import (  # noqa: E402
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    EmbeddingSentenceJob,
    MappedWord,
    MediaFiles,
    PendingUnknownSentence,
    SentencePart,
    TimedWord,
    VadWindowState,
    WindowTranscript,
)
from window.window_media import (  # noqa: E402
    media_cache_status,
    resolve_browser_stream_id,
    resolve_media,
    resolve_media_url,
)
from window.window_remote_asr import RemoteWindowAsrClient  # noqa: E402
from window.session_store import DEFAULT_SESSION_DIR, SessionStore  # noqa: E402
from window.session_lease import SessionLease, SessionLeaseError, SessionLeaseStateMachine  # noqa: E402
from window.session_persistence import SessionPersistenceCoordinator  # noqa: E402
from window.media_manager import MediaManager  # noqa: E402
from window.live_translation import LiveTranslationCoordinator  # noqa: E402


from window.window_config import (  # noqa: E402
    DEFAULT_CUNK_CANONICAL,
    DEFAULT_EMBEDDING_HELPER_RESPONSE_TIMEOUT_SECONDS,
    DEFAULT_FAST_WHISPER_CACHE,
    DEFAULT_KROKO_PREVIEW_AUTO_DOWNLOAD,
    DEFAULT_KROKO_PREVIEW_MODEL_PRESET,
    DEFAULT_KROKO_PREVIEW_PYTHON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REALTIMESTT_ROOT,
    DEFAULT_REMOTE_ASR_URL,
    DEFAULT_REMOTE_EMBEDDINGS_TIMEOUT_SECONDS,
    DEFAULT_REMOTE_EMBEDDINGS_URL,
    DEFAULT_SPEAKER_LIBRARY_DIR,
    DEFAULT_VALIDATION_OUTPUT,
    DEFAULT_WINDOW_EMBEDDING_PROVIDER,
    NEW_SPEAKER_SENSITIVITY_PRESETS,
    PRESET_YOUTUBE_VIDEOS,
    SPEAKER_COLORS,
    apply_new_speaker_sensitivity,
    default_faster_whisper_download_root,
    default_kroko_preview_model_path,
    default_kroko_preview_startup_timeout_seconds,
    default_silero_vad_backend,
    default_silero_vad_model_path,
    new_speaker_sensitivity_config,
)
from window.language_config import (  # noqa: E402
    default_language_code,
    default_sentence_language,
    default_sentence_tokenizer,
    get_language_config,
    infer_language_from_kroko_model_name,
    language_arg,
    language_flag_country_code,
    sentence_tokenizer_arg,
)
from window.realtime_preview_backends import (  # noqa: E402
    apply_preview_timing_defaults,
    default_preview_model,
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
)
from window.sherpa_onnx_models import (  # noqa: E402
    DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET,
    default_sherpa_onnx_model_dir,
    sherpa_onnx_model_preset,
)
from window.window_diarizer import StartSessionRequest, WindowDiarizer  # noqa: E402
from window.window_events import EventBus, RecordingEventBus  # noqa: E402
from window.web_assets import (  # noqa: E402
    read_web_asset,
    render_live_index,
    web_asset_content_type,
)
from window.public_events import PublicEventNormalizer  # noqa: E402
from window.browser_live_speaker_scoring import (  # noqa: E402
    DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS,
    DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
    BrowserLiveObservationRecorder,
)

AUDIO_UPLOAD_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}




def build_window_validation_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest_by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "sentence":
            continue
        payload = record.get("payload") or {}
        if payload.get("pending") or payload.get("realtime") or payload.get("provisional_assignment"):
            continue
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        latest_by_index[index] = dict(payload)

    analysis_records: list[dict[str, Any]] = []
    final_payloads: list[dict[str, Any]] = []
    for index in sorted(latest_by_index):
        payload = dict(latest_by_index[index])
        start = float(payload.get("start") or 0.0)
        end = float(payload.get("end") or start)
        payload["video_start_seconds"] = start
        payload["video_end_seconds"] = end
        payload["duration_seconds"] = max(0.0, end - start)
        final_payloads.append(payload)
        analysis_records.append({"time": time.time(), "event": "final", "payload": payload})
        analysis_records.append({"time": time.time(), "event": "sentence", "payload": payload})
    return analysis_records, final_payloads


def ratio_summary(final_payloads: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ratios = [
        float(payload["speech_audio_ratio"])
        for payload in final_payloads
        if payload.get("speech_audio_ratio") is not None
    ]
    if not ratios:
        return {"count": 0, "below_threshold": 0}
    return {
        "count": len(ratios),
        "below_threshold": sum(1 for ratio in ratios if ratio < threshold),
        "min": round(min(ratios), 4),
        "max": round(max(ratios), 4),
        "mean": round(sum(ratios) / len(ratios), 4),
    }


def retranscribe_final_payloads_with_enhancement(
    controller: WindowDiarizer,
    final_payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Re-ASR accepted sentence clips once, after raw growing-window splitting."""

    started = time.monotonic()
    updated_payloads: list[dict[str, Any]] = []
    model = controller._model
    if model is None:
        raise RuntimeError("Final enhanced ASR requested without a loaded ASR backend.")
    total = len(final_payloads)
    for position, original in enumerate(final_payloads, start=1):
        payload = dict(original)
        start = float(payload.get("start") or 0.0)
        end = float(payload.get("end") or start)
        audio, sample_rate = controller._audio_window_copy(start, end)
        raw_text = str(payload.get("text") or "")
        enhanced_text = controller._transcribe_enhanced_final_audio_text(model, audio, sample_rate)
        source_text_hash = hashlib.sha256(enhanced_text.encode("utf-8")).hexdigest()
        payload.update({
            "text": enhanced_text,
            "source_text_hash": source_text_hash,
            "source_revision": source_text_hash,
            "pre_enhancement_asr_text": raw_text,
            "final_asr_enhanced": True,
        })
        updated_payloads.append(payload)
        if position % 25 == 0 or position == total:
            print(f"Enhanced final ASR={position}/{total}", flush=True)

    analysis_records: list[dict[str, Any]] = []
    for payload in updated_payloads:
        analysis_records.append({"time": time.time(), "event": "final", "payload": payload})
        analysis_records.append({"time": time.time(), "event": "sentence", "payload": payload})
    return analysis_records, updated_payloads, time.monotonic() - started


def run_window_replay_validation(args: Any) -> int:
    from realtime.realtime_speakerdiarize import analyze_trace_against_canonical, read_canonical_segments

    media = resolve_media(args)
    if not args.validation_keep_preview:
        args = args.with_updates(realtime_preview_engine="off")
    bus = RecordingEventBus()
    controller = WindowDiarizer(args, media, bus)
    reached_end_at: float | None = None
    started = time.monotonic()
    try:
        controller.start()
        replay_started = time.monotonic()
        bus.emit(
            "validation_replay_start",
            {
                "replay_speed": args.validation_replay_speed,
                "duration_seconds": round(float(controller.duration), 4),
            },
        )
        last_report = -1
        while not bus.done.is_set():
            elapsed = time.monotonic() - replay_started
            playback_seconds = min(controller.duration, elapsed * max(0.01, args.validation_replay_speed))
            # Validation replay owns the synthetic media clock; allow accelerated
            # runs to advance faster than the browser/live wall-clock clamp.
            controller.set_playback_time(
                playback_seconds,
                reset=bool(args.validation_replay_speed > 1.0),
            )
            report_second = int(playback_seconds // 15) * 15
            if report_second != last_report and report_second > 0:
                last_report = report_second
                print(f"Replay playback={playback_seconds:.1f}s/{controller.duration:.1f}s", flush=True)
            if playback_seconds >= controller.duration:
                if reached_end_at is None:
                    reached_end_at = time.monotonic()
                elif time.monotonic() - reached_end_at >= args.validation_final_wait_seconds:
                    print("Timed out waiting for final window flush.", flush=True)
                    break
            bus.done.wait(max(0.02, args.validation_update_interval_seconds))
    finally:
        controller.shutdown()

    analysis_records, final_payloads = build_window_validation_records(bus.records)
    final_asr_retranscription_seconds = 0.0
    if bool(getattr(args, "enhance_asr", False)):
        analysis_records, final_payloads, final_asr_retranscription_seconds = (
            retranscribe_final_payloads_with_enhancement(controller, final_payloads)
        )
    elapsed = time.monotonic() - started
    canonical_segments = read_canonical_segments(args.validation_canonical)
    summary = analyze_trace_against_canonical(
        analysis_records,
        canonical_segments,
        match_mode=args.validation_match_mode,
    )
    summary.update({
        "system": "youtube_window_diarize_gui",
        "media": {
            "url": media.url,
            "video_id": media.video_id,
            "audio_file": str(media.audio_file),
            "video_file": str(media.video_file),
            "duration_sec": round(float(controller.duration), 4),
        },
        "canonical": str(args.validation_canonical),
        "elapsed_seconds": round(elapsed, 4),
        "validation_replay_speed": args.validation_replay_speed,
        "validation_keep_preview": args.validation_keep_preview,
        "final_asr_retranscription_seconds": round(final_asr_retranscription_seconds, 6),
        "speech_enhancement": {
            "url": getattr(args, "speech_enhancement_url", ""),
            "enhance_asr": bool(getattr(args, "enhance_asr", False)),
            "enhance_embeddings": bool(getattr(args, "enhance_embeddings", False)),
            "stats": (
                controller._speech_enhancement_client.stats()
                if controller._speech_enhancement_client is not None
                else {
                    "request_count": 0,
                    "input_seconds": 0.0,
                    "http_seconds": 0.0,
                    "queue_seconds": 0.0,
                    "processing_seconds": 0.0,
                }
            ),
        },
        "embedding_provider": args.embedding_provider,
        "embeddings_backend": args.embeddings_backend,
        "embedding_device": args.embedding_device,
        "embedding_python": str(args.embedding_python),
        "remote_embeddings_url": args.remote_embeddings_url,
        "clustering_args": {
            "same_speaker_similarity": args.same_speaker_similarity,
            "similarity_temperature": args.similarity_temperature,
            "speaker_softmax_temperature": args.speaker_softmax_temperature,
            "new_speaker_threshold": args.new_speaker_threshold,
            "duplicate_profile_similarity": args.duplicate_profile_similarity,
            "unknown_short_threshold": args.unknown_short_threshold,
            "min_first_speaker_seconds": args.min_first_speaker_seconds,
            "first_speaker_immediate_min_seconds": args.first_speaker_immediate_min_seconds,
            "min_new_speaker_seconds": args.min_new_speaker_seconds,
            "late_new_speaker_min_seconds": args.late_new_speaker_min_seconds,
            "max_speakers": args.max_speakers,
            "min_margin": args.min_margin,
            "margin_temperature": args.margin_temperature,
            "update_unknown_max": args.update_unknown_max,
            "new_speaker_confirmation_count": args.new_speaker_confirmation_count,
            "new_speaker_confirmation_similarity": args.new_speaker_confirmation_similarity,
            "max_pending_new_speakers": args.max_pending_new_speakers,
            "known_speaker_min_similarity": args.known_speaker_min_similarity,
            "known_speaker_gray_zone_min_unknown_probability": (
                args.known_speaker_gray_zone_min_unknown_probability
            ),
            "profile_update_min_similarity": args.profile_update_min_similarity,
            "profile_update_min_margin": args.profile_update_min_margin,
            "low_similarity_unknown_floor_similarity": args.low_similarity_unknown_floor_similarity,
            "low_similarity_unknown_floor_probability": args.low_similarity_unknown_floor_probability,
            "gray_zone_promote_max_similarity": args.gray_zone_promote_max_similarity,
            "min_new_speaker_words": args.min_new_speaker_words,
            "min_speech_audio_ratio": args.min_speech_audio_ratio,
            "retro_reassign_min_similarity": args.retro_reassign_min_similarity,
            "retro_reassign_min_margin": args.retro_reassign_min_margin,
            "speaker_refinement": args.speaker_refinement,
            "speaker_refinement_unknown_tentative": args.speaker_refinement_unknown_tentative,
            "speaker_refinement_unknown_commit": args.speaker_refinement_unknown_commit,
            "allow_speaker_reassignment": args.allow_speaker_reassignment,
            "speaker_refinement_max_per_profile": args.speaker_refinement_max_per_profile,
            "speaker_refinement_min_duration": args.speaker_refinement_min_duration,
            "speaker_refinement_max_unknown": args.speaker_refinement_max_unknown,
            "speaker_refinement_top_k": args.speaker_refinement_top_k,
            "speaker_refinement_centroid_blend": args.speaker_refinement_centroid_blend,
            "speaker_refinement_unknown_min_similarity": args.speaker_refinement_unknown_min_similarity,
            "speaker_refinement_unknown_min_margin": args.speaker_refinement_unknown_min_margin,
            "speaker_refinement_known_max_duration": args.speaker_refinement_known_max_duration,
            "speaker_refinement_known_min_similarity": args.speaker_refinement_known_min_similarity,
            "speaker_refinement_known_min_delta": args.speaker_refinement_known_min_delta,
            "speaker_refinement_final_passes": args.speaker_refinement_final_passes,
            "speaker_refinement_small_island_merge": args.speaker_refinement_small_island_merge,
            "speaker_refinement_small_island_max_duration": args.speaker_refinement_small_island_max_duration,
            "speaker_refinement_small_island_max_segments": args.speaker_refinement_small_island_max_segments,
            "speaker_refinement_tiny_fragmented_merge": args.speaker_refinement_tiny_fragmented_merge,
            "speaker_refinement_tiny_fragmented_max_duration": args.speaker_refinement_tiny_fragmented_max_duration,
            "speaker_refinement_tiny_fragmented_max_segments": args.speaker_refinement_tiny_fragmented_max_segments,
            "speaker_refinement_tiny_fragmented_min_islands": args.speaker_refinement_tiny_fragmented_min_islands,
            "speaker_refinement_tiny_fragmented_max_islands": args.speaker_refinement_tiny_fragmented_max_islands,
            "speaker_refinement_tiny_fragmented_min_neighbor_share": (
                args.speaker_refinement_tiny_fragmented_min_neighbor_share
            ),
            "speaker_refinement_terminal_outro_merge": args.speaker_refinement_terminal_outro_merge,
            "speaker_refinement_terminal_outro_max_duration": args.speaker_refinement_terminal_outro_max_duration,
            "speaker_refinement_terminal_outro_lookback_segments": (
                args.speaker_refinement_terminal_outro_lookback_segments
            ),
            "speaker_refinement_terminal_outro_min_target_duration": (
                args.speaker_refinement_terminal_outro_min_target_duration
            ),
            "speaker_refinement_unknown_same_speaker_fill": (
                args.speaker_refinement_unknown_same_speaker_fill
            ),
            "speaker_refinement_unknown_same_speaker_max_duration": (
                args.speaker_refinement_unknown_same_speaker_max_duration
            ),
            "speaker_refinement_unknown_same_speaker_max_segments": (
                args.speaker_refinement_unknown_same_speaker_max_segments
            ),
            "speaker_refinement_unknown_previous_speaker_fill": (
                args.speaker_refinement_unknown_previous_speaker_fill
            ),
            "speaker_refinement_unknown_previous_speaker_max_duration": (
                args.speaker_refinement_unknown_previous_speaker_max_duration
            ),
            "speaker_refinement_unknown_previous_speaker_max_segments": (
                args.speaker_refinement_unknown_previous_speaker_max_segments
            ),
            "speaker_refinement_unknown_previous_speaker_max_previous_gap": (
                args.speaker_refinement_unknown_previous_speaker_max_previous_gap
            ),
            "speaker_refinement_unknown_previous_speaker_min_next_gap": (
                args.speaker_refinement_unknown_previous_speaker_min_next_gap
            ),
            "speaker_refinement_unknown_next_speaker_fill": (
                args.speaker_refinement_unknown_next_speaker_fill
            ),
            "speaker_refinement_unknown_next_speaker_max_duration": (
                args.speaker_refinement_unknown_next_speaker_max_duration
            ),
            "speaker_refinement_unknown_next_speaker_max_segments": (
                args.speaker_refinement_unknown_next_speaker_max_segments
            ),
            "speaker_refinement_unknown_next_speaker_max_next_gap": (
                args.speaker_refinement_unknown_next_speaker_max_next_gap
            ),
            "speaker_refinement_unknown_next_speaker_min_previous_gap": (
                args.speaker_refinement_unknown_next_speaker_min_previous_gap
            ),
            "speaker_refinement_long_low_confidence_retro_split": (
                args.speaker_refinement_long_low_confidence_retro_split
            ),
            "speaker_refinement_long_low_confidence_retro_min_duration": (
                args.speaker_refinement_long_low_confidence_retro_min_duration
            ),
            "speaker_refinement_long_low_confidence_retro_max_similarity": (
                args.speaker_refinement_long_low_confidence_retro_max_similarity
            ),
            "speaker_refinement_long_low_confidence_retro_max_margin": (
                args.speaker_refinement_long_low_confidence_retro_max_margin
            ),
            "speaker_refinement_long_low_confidence_retro_max_splits": (
                args.speaker_refinement_long_low_confidence_retro_max_splits
            ),
            "new_speaker_sensitivity": getattr(args, "new_speaker_sensitivity", 3),
            "new_speaker_sensitivity_label": getattr(args, "new_speaker_sensitivity_label", "Balanced"),
            "vad_sentence_splitting": args.vad_sentence_splitting,
            "vad_backend": args.vad_backend,
            "vad_silero_backend": args.vad_silero_backend,
            "vad_silero_onnx_model_path": str(args.vad_silero_onnx_model_path) if args.vad_silero_onnx_model_path is not None else None,
            "vad_silero_onnx_threads": args.vad_silero_onnx_threads,
            "vad_silero_speech_threshold": args.vad_silero_speech_threshold,
            "vad_silence_seconds": args.vad_silence_seconds,
            "vad_final_window_post_silence_seconds": args.vad_final_window_post_silence_seconds,
            "vad_next_window_start_silence_seconds": args.vad_next_window_start_silence_seconds,
            "vad_speech_rms_threshold": args.vad_speech_rms_threshold,
            "vad_frame_seconds": args.vad_frame_seconds,
            "vad_merge_gap_seconds": args.vad_merge_gap_seconds,
            "vad_min_speech_seconds": args.vad_min_speech_seconds,
            "vad_gate_secondary_backend": args.vad_gate_secondary_backend,
            "vad_gate_webrtc_mode": args.vad_gate_webrtc_mode,
            "vad_gate_min_consensus_seconds": args.vad_gate_min_consensus_seconds,
            "vad_gate_min_consensus_ratio": args.vad_gate_min_consensus_ratio,
            "asr_no_speech_filter": args.asr_no_speech_filter,
            "asr_no_speech_prob_threshold": args.asr_no_speech_prob_threshold,
            "asr_no_speech_hard_threshold": args.asr_no_speech_hard_threshold,
            "asr_no_speech_keep_short_max_words": args.asr_no_speech_keep_short_max_words,
            "asr_no_speech_keep_short_max_seconds": args.asr_no_speech_keep_short_max_seconds,
            "live_speaker_assignment": args.live_speaker_assignment,
            "live_speaker_embedding_provider": args.live_speaker_embedding_provider,
            "live_speaker_embedding_min_interval_seconds": args.live_speaker_embedding_min_interval_seconds,
            "live_speaker_embedding_target_utilization": args.live_speaker_embedding_target_utilization,
            "live_speaker_verify_on_change": args.live_speaker_verify_on_change,
            "live_speaker_verify_min_interval_seconds": args.live_speaker_verify_min_interval_seconds,
            "live_speaker_ema_window_seconds": args.live_speaker_ema_window_seconds,
            "live_speaker_ema_count": args.live_speaker_ema_count,
            "live_speaker_ema_alpha": args.live_speaker_ema_alpha,
            "live_speaker_probe_interval_seconds": args.live_speaker_probe_interval_seconds,
            "live_speaker_probe_attack_interval_seconds": args.live_speaker_probe_attack_interval_seconds,
            "live_speaker_probe_window_seconds": args.live_speaker_probe_window_seconds,
            "live_speaker_probe_hold_seconds": args.live_speaker_probe_hold_seconds,
            "live_speaker_probe_min_advance_seconds": args.live_speaker_probe_min_advance_seconds,
            "live_speaker_probe_attack_min_advance_seconds": args.live_speaker_probe_attack_min_advance_seconds,
            "live_speaker_probe_clear_silence_count": args.live_speaker_probe_clear_silence_count,
            "live_speaker_probe_clear_unknown_count": args.live_speaker_probe_clear_unknown_count,
            "live_speaker_probe_unknown_clear_debounce_seconds": args.live_speaker_probe_unknown_clear_debounce_seconds,
            "live_speaker_probe_unknown_keepalive": args.live_speaker_probe_unknown_keepalive,
            "live_speaker_probe_unknown_release_smoothing": args.live_speaker_probe_unknown_release_smoothing,
            "live_speaker_probe_unknown_release_count": args.live_speaker_probe_unknown_release_count,
            "live_speaker_probe_unknown_release_ema_alpha": args.live_speaker_probe_unknown_release_ema_alpha,
            "live_speaker_probe_unknown_release_margin": args.live_speaker_probe_unknown_release_margin,
            "live_speaker_weak_profile_assist": args.live_speaker_weak_profile_assist,
            "live_speaker_weak_profile_max_speech_seconds": args.live_speaker_weak_profile_max_speech_seconds,
            "live_speaker_weak_profile_min_similarity": args.live_speaker_weak_profile_min_similarity,
            "live_speaker_weak_profile_min_margin": args.live_speaker_weak_profile_min_margin,
            "live_speaker_weak_profile_max_unknown_probability": (
                args.live_speaker_weak_profile_max_unknown_probability
            ),
            "section_gap_new_speaker": args.section_gap_new_speaker,
            "section_gap_new_speaker_min_gap_seconds": args.section_gap_new_speaker_min_gap_seconds,
            "section_gap_new_speaker_min_prior_speech_seconds": (
                args.section_gap_new_speaker_min_prior_speech_seconds
            ),
            "section_gap_new_speaker_min_duration_seconds": (
                args.section_gap_new_speaker_min_duration_seconds
            ),
            "section_gap_new_speaker_min_similarity": args.section_gap_new_speaker_min_similarity,
            "section_gap_new_speaker_max_similarity": args.section_gap_new_speaker_max_similarity,
            "section_gap_new_speaker_min_margin": args.section_gap_new_speaker_min_margin,
            "unknown_pair_new_speaker": args.unknown_pair_new_speaker,
            "unknown_pair_new_speaker_max_gap_seconds": args.unknown_pair_new_speaker_max_gap_seconds,
            "unknown_pair_new_speaker_min_unknown_duration_seconds": (
                args.unknown_pair_new_speaker_min_unknown_duration_seconds
            ),
            "unknown_pair_new_speaker_min_current_duration_seconds": (
                args.unknown_pair_new_speaker_min_current_duration_seconds
            ),
            "unknown_pair_new_speaker_min_pair_similarity": args.unknown_pair_new_speaker_min_pair_similarity,
            "unknown_pair_new_speaker_max_existing_similarity": (
                args.unknown_pair_new_speaker_max_existing_similarity
            ),
            "unknown_pair_new_speaker_max_existing_margin": args.unknown_pair_new_speaker_max_existing_margin,
            "unknown_pair_new_speaker_min_unknown_probability": (
                args.unknown_pair_new_speaker_min_unknown_probability
            ),
            "live_speaker_raw_change_snap": args.live_speaker_raw_change_snap,
            "live_speaker_raw_change_min_probability": args.live_speaker_raw_change_min_probability,
            "live_speaker_raw_change_min_margin": args.live_speaker_raw_change_min_margin,
            "live_speaker_sentence_hint": args.live_speaker_sentence_hint,
            "live_speaker_sentence_hint_max_lag_seconds": args.live_speaker_sentence_hint_max_lag_seconds,
            "live_speaker_sentence_hint_new_speaker_max_lag_seconds": args.live_speaker_sentence_hint_new_speaker_max_lag_seconds,
            "live_speaker_sentence_hint_new_speaker_hold_seconds": args.live_speaker_sentence_hint_new_speaker_hold_seconds,
            "live_speaker_sentence_hint_new_speaker_max_top_similarity": args.live_speaker_sentence_hint_new_speaker_max_top_similarity,
            "live_speaker_sentence_hint_hold_seconds": args.live_speaker_sentence_hint_hold_seconds,
        },
        "min_speech_audio_ratio": args.min_speech_audio_ratio,
        "speech_audio_ratio": ratio_summary(final_payloads, args.min_speech_audio_ratio),
        "unknown_permanent_segments": sum(1 for payload in final_payloads if payload.get("unknown_permanent")),
        "created_speaker_segments": sum(1 for payload in final_payloads if payload.get("created_speaker")),
        "raw_event_counts": dict(Counter(str(record.get("event")) for record in bus.records)),
        "final_payloads": final_payloads,
    })

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.validation_trace_output is not None:
        args.validation_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_trace_output.write_text(
            "\n".join(json_dumps(record) for record in bus.records) + "\n",
            encoding="utf-8",
        )

    print(f"Window validation output: {args.validation_output}", flush=True)
    if args.validation_trace_output is not None:
        print(f"Window validation trace: {args.validation_trace_output}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Final segments: {summary['final_segments']}", flush=True)
    print(f"Resolved segments: {summary['resolved_segments']}", flush=True)
    print(f"Live final words: {summary['live_final_words']} / canonical {summary['canonical_words']}", flush=True)
    print(f"Text recall/precision by LCS: {summary['text_recall']:.3f} / {summary['text_precision']:.3f}", flush=True)
    print(f"Assigned counts: {summary['assigned_counts']}", flush=True)
    print(f"Profile map: {summary['profile_map']}", flush=True)
    print(f"Unknown segments: {summary['unknown_segments']}", flush=True)
    print(f"Unknown permanent segments: {summary['unknown_permanent_segments']}", flush=True)
    print(f"Created speaker segments: {summary['created_speaker_segments']}", flush=True)
    print(f"Speech/audio ratio: {summary['speech_audio_ratio']}", flush=True)
    print(f"Segment accuracy after profile mapping: {summary['segment_accuracy']:.3f}", flush=True)
    print(f"Duration accuracy after profile mapping: {summary['duration_accuracy']:.3f}", flush=True)
    return 0
