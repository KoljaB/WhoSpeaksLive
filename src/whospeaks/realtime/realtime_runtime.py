"""Compatibility facade for the realtime YouTube WASAPI diarization runtime."""

from __future__ import annotations

from whospeaks.realtime.realtime_capture import (
    EventBus,
    TraceLogger,
    VideoClock,
    YouTubeWasapiController,
    choose_wasapi_loopback_device,
    extract_youtube_video_id,
    list_audio_input_devices,
)
from whospeaks.realtime.realtime_server import GuiServer, RequestHandler
from whospeaks.realtime.realtime_speaker_engine import (
    ProcessedSentenceRecord,
    RealtimeSpeakerEngine,
    SentenceJob,
    context_adjusted_decision,
    is_context_anchor,
    is_context_assignment_candidate,
    is_reassignment_accepted,
    is_reassignment_candidate,
    speaker_probability_key,
)
from whospeaks.realtime.realtime_transcript import (
    DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO,
    DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS,
    DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS,
    TranscriptPart,
    normalize_timed_words,
    sentence_audio_boundary_between_words,
    sentence_audio_clip_start_for_first_word,
    split_transcript_by_timestamps,
    timed_word_text,
)

__all__ = [
    "DEFAULT_SENTENCE_BOUNDARY_GAP_RATIO",
    "DEFAULT_SENTENCE_BOUNDARY_POST_PADDING_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_PRE_PADDING_SECONDS",
    "EventBus",
    "GuiServer",
    "ProcessedSentenceRecord",
    "RealtimeSpeakerEngine",
    "RequestHandler",
    "SentenceJob",
    "TraceLogger",
    "TranscriptPart",
    "VideoClock",
    "YouTubeWasapiController",
    "choose_wasapi_loopback_device",
    "context_adjusted_decision",
    "extract_youtube_video_id",
    "is_context_anchor",
    "is_context_assignment_candidate",
    "is_reassignment_accepted",
    "is_reassignment_candidate",
    "list_audio_input_devices",
    "normalize_timed_words",
    "sentence_audio_boundary_between_words",
    "sentence_audio_clip_start_for_first_word",
    "speaker_probability_key",
    "split_transcript_by_timestamps",
    "timed_word_text",
]
