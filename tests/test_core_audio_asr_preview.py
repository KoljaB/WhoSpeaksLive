from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.asr_hallucination_policy import (
    match_asr_hallucination_policy,
    normalize_asr_hallucination_text,
)
from window.window_domain import TimedWord, VadWindowState
from window.window_events import RecordingEventBus
from window.window_preview import KrokoSubprocessPreviewTranscriber



from tests.window_diarizer_support import make_window_diarizer


class WindowAudioAsrTests(unittest.TestCase):
    def test_asr_hallucination_policy_normalizes_unicode_and_uses_word_boundaries(self) -> None:
        self.assertEqual(normalize_asr_hallucination_text("  THANKS—for watching!!! "), "thanks for watching")
        amara_org = match_asr_hallucination_policy(
            "Visit Amara.org",
            base_suspicion_threshold=0.45,
        )
        amara_name = match_asr_hallucination_policy(
            "Amara, what do you think?",
            base_suspicion_threshold=0.45,
        )
        samara = match_asr_hallucination_policy(
            "Samara, what do you think?",
            base_suspicion_threshold=0.45,
        )
        watching_at_start = match_asr_hallucination_policy(
            "Thanks for watching!",
            base_suspicion_threshold=0.45,
            segment_start_seconds=0.0,
            media_duration_seconds=300.0,
        )
        watching_at_end = match_asr_hallucination_policy(
            "Thanks for watching!",
            base_suspicion_threshold=0.45,
            segment_start_seconds=290.0,
            media_duration_seconds=300.0,
        )

        self.assertIsNotNone(amara_org)
        self.assertEqual(amara_org.rule_id, "amara_org")
        self.assertIsNotNone(amara_name)
        self.assertEqual(amara_name.rule_id, "amara_name")
        self.assertIsNone(samara)
        self.assertGreater(watching_at_start.suspicion_threshold, watching_at_end.suspicion_threshold)

        watching_in_context = match_asr_hallucination_policy(
            "Before we go, thank you for watching and please subscribe.",
            base_suspicion_threshold=0.45,
            segment_start_seconds=0.0,
            media_duration_seconds=300.0,
        )
        self.assertIsNotNone(watching_in_context)
        self.assertEqual(watching_in_context.rule_id, "thanks_for_watching_in_context")
        self.assertLess(watching_in_context.risk_score, 70)

    def test_browser_stream_audio_uses_chunks_and_slices_across_boundaries(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.set_browser_stream("https://example.test/watch?v=stream-test")

        first_duration = diarizer.append_stream_audio(
            np.array([0.1, 0.2], dtype=np.float32),
            16_000,
        )
        second_duration = diarizer.append_stream_audio(
            np.array([0.3, 0.4, 0.5], dtype=np.float32),
            16_000,
        )

        self.assertEqual(first_duration, 2 / 16_000)
        self.assertEqual(second_duration, 5 / 16_000)
        self.assertEqual(diarizer.playback_time(), second_duration)
        self.assertEqual(diarizer._stream_audio_samples, 5)
        np.testing.assert_allclose(diarizer.audio, [0.1, 0.2, 0.3, 0.4, 0.5])

        audio, sample_rate = diarizer._audio_window_copy(1 / 16_000, 5 / 16_000)
        self.assertEqual(sample_rate, 16_000)
        np.testing.assert_allclose(audio, np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))

    def test_file_playback_time_rejects_impossible_jump_to_media_end(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = make_window_diarizer()
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = False
        diarizer.duration = 60.0
        diarizer._playback_clock_started_at = time.monotonic() - 1.0
        diarizer._last_playback_jump_warning_at = 0.0
        diarizer.bus = Bus()

        diarizer.set_playback_time(60.0)

        self.assertLess(diarizer.playback_time(), 5.0)
        self.assertTrue(any("Ignored early playback jump" in str(payload.get("message")) for _event, payload in diarizer.bus.events))

    def test_stream_playback_time_is_not_wall_clock_clamped(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = True
        diarizer.duration = 60.0
        diarizer._playback_clock_started_at = time.monotonic()
        diarizer._last_playback_jump_warning_at = 0.0
        diarizer.bus = object()

        diarizer.set_playback_time(60.0)

        self.assertEqual(diarizer.playback_time(), 60.0)

    def test_live_speaker_probe_uses_cheap_rms_speech_gate(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            vad_frame_seconds=0.1,
            vad_speech_rms_threshold=0.003,
            live_speaker_probe_min_speech_seconds=0.2,
        )

        self.assertFalse(diarizer._audio_has_rms_speech(np.zeros(200, dtype=np.float32), 100))
        audio = np.zeros(200, dtype=np.float32)
        audio[50:90] = 0.01

        self.assertTrue(diarizer._audio_has_rms_speech(audio, 100))

    def test_asr_vad_gate_spans_trim_window_edges_without_cutting_internal_gaps(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_vad_gate=True,
            asr_vad_gate_pre_padding_seconds=0.2,
            asr_vad_gate_post_padding_seconds=0.35,
            asr_vad_gate_merge_gap_seconds=0.85,
            asr_vad_gate_min_clip_seconds=0.2,
            asr_vad_gate_cut_internal_gaps=False,
        )
        vad_state = VadWindowState(
            has_speech=True,
            should_flush=False,
            speech_spans=[(1.0, 1.4), (2.0, 2.3), (4.0, 4.4)],
        )

        spans = diarizer._asr_vad_gate_spans(0.0, 5.0, vad_state)

        self.assertEqual(len(spans), 1)
        np.testing.assert_allclose(spans[0], (0.8, 4.75))

    def test_asr_vad_gate_rejects_primary_vad_without_secondary_evidence(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_vad_gate=True,
            asr_vad_gate_pre_padding_seconds=0.2,
            asr_vad_gate_post_padding_seconds=0.35,
            asr_vad_gate_merge_gap_seconds=0.85,
            asr_vad_gate_min_clip_seconds=0.2,
            asr_vad_gate_cut_internal_gaps=False,
            vad_gate_secondary_backend="webrtc",
            vad_gate_min_consensus_seconds=0.1,
            vad_gate_min_consensus_ratio=0.05,
        )
        primary_state = VadWindowState(
            has_speech=True,
            should_flush=False,
            speech_spans=[(11.8, 14.1)],
            backend="silero",
        )
        secondary_state = VadWindowState(False, False, backend="webrtc3")

        self.assertEqual(diarizer._asr_vad_gate_spans(10.0, 15.0, primary_state, secondary_state), [])

    def test_asr_vad_gate_uses_secondary_evidence_for_edges_but_keeps_middle(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_vad_gate=True,
            asr_vad_gate_pre_padding_seconds=0.2,
            asr_vad_gate_post_padding_seconds=0.35,
            asr_vad_gate_merge_gap_seconds=0.85,
            asr_vad_gate_min_clip_seconds=0.2,
            asr_vad_gate_cut_internal_gaps=False,
            vad_gate_secondary_backend="webrtc",
            vad_gate_min_consensus_seconds=0.1,
            vad_gate_min_consensus_ratio=0.05,
        )
        primary_state = VadWindowState(
            has_speech=True,
            should_flush=False,
            speech_spans=[(1.0, 2.0), (3.0, 4.5)],
            backend="silero",
        )
        secondary_state = VadWindowState(
            has_speech=True,
            should_flush=False,
            speech_spans=[(1.05, 1.2), (4.15, 4.35)],
            backend="webrtc3",
        )

        spans = diarizer._asr_vad_gate_spans(0.0, 5.0, primary_state, secondary_state)

        self.assertEqual(len(spans), 1)
        np.testing.assert_allclose(spans[0], (0.85, 4.7))

    def test_silero_vad_reset_and_window_inference_are_atomic_across_threads(self) -> None:
        diarizer = make_window_diarizer()

        class StatefulVadModel:
            backend = "test"

            def __init__(self) -> None:
                self.owner = ""
                self.reset_count = 0
                self.silence_reset = threading.Event()
                self.speech_reset = threading.Event()

            def reset_states(self) -> None:
                self.owner = threading.current_thread().name
                self.reset_count += 1
                if self.owner == "silence":
                    self.silence_reset.set()
                elif self.owner == "speech":
                    self.speech_reset.set()

            def __call__(self, _chunk: np.ndarray, _sample_rate: int) -> float:
                caller = threading.current_thread().name
                if caller == "silence":
                    self.speech_reset.wait(timeout=0.15)
                    return 0.1 if self.owner == "silence" else 0.9
                return 0.9

        model = StatefulVadModel()
        diarizer._vad_model = model
        diarizer._vad_model_backend = "test"
        audio = np.zeros(512, dtype=np.float32)
        errors: list[Exception] = []
        results: dict[str, bool] = {}

        def evaluate(name: str) -> None:
            try:
                state = diarizer._silero_vad_window_state(
                    0.0,
                    audio.size / 16_000.0,
                    audio,
                    16_000,
                    min_speech_seconds=0.0,
                )
                results[name] = state.has_speech
            except Exception as exc:
                errors.append(exc)

        silence_thread = threading.Thread(target=evaluate, args=("silence",), name="silence")
        speech_thread = threading.Thread(target=evaluate, args=("speech",), name="speech")
        silence_thread.start()
        self.assertTrue(model.silence_reset.wait(timeout=1.0))
        speech_thread.start()
        threads = [silence_thread, speech_thread]
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(model.reset_count, 2)
        self.assertEqual(results, {"silence": False, "speech": True})

    def test_asr_no_speech_filter_retains_and_flags_high_no_speech_prob_words(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_no_speech_filter=True,
            asr_no_speech_prob_threshold=0.65,
            asr_no_speech_hard_threshold=0.85,
            asr_no_speech_keep_short_max_words=2,
            asr_no_speech_keep_short_max_seconds=0.45,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Hallo", 0.0, 0.4, no_speech_prob=0.08, segment_index=0),
            TimedWord(" alpha", 1.0, 1.6, no_speech_prob=0.74, segment_index=1),
            TimedWord(" beta", 1.6, 1.9, no_speech_prob=0.74, segment_index=1),
            TimedWord(" gamma", 1.9, 3.2, no_speech_prob=0.74, segment_index=1),
            TimedWord(" Ja.", 3.5, 3.7, no_speech_prob=0.69, segment_index=2),
            TimedWord(" unknown", 4.0, 4.4, no_speech_prob=None),
        ]

        kept = diarizer._filter_asr_no_speech_words(words)

        self.assertEqual(kept, words)
        self.assertFalse(words[0].asr_review)
        self.assertTrue(all(word.asr_review.get("needs_review") for word in words[1:4]))
        self.assertFalse(words[4].asr_review)
        self.assertFalse(words[5].asr_review)
        self.assertTrue(
            any(
                "ASR no-speech check retained 3 word" in item["payload"]["message"]
                for item in diarizer.bus.records
            )
        )

    def test_asr_no_speech_filter_flags_but_keeps_short_segments_above_hard_threshold(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_no_speech_filter=True,
            asr_no_speech_prob_threshold=0.65,
            asr_no_speech_hard_threshold=0.85,
            asr_no_speech_keep_short_max_words=2,
            asr_no_speech_keep_short_max_seconds=0.45,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Ja.", 0.0, 0.2, no_speech_prob=0.90, segment_index=0),
            TimedWord(" Hallo", 1.0, 1.4, no_speech_prob=0.08, segment_index=1),
        ]

        kept = diarizer._filter_asr_no_speech_words(words)

        self.assertEqual(kept, words)
        self.assertTrue(words[0].asr_review.get("needs_review"))

    def test_asr_credit_text_is_not_blacklisted_without_acoustic_verification(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(asr_no_speech_filter=False)
        diarizer.bus = RecordingEventBus()
        credit = [
            TimedWord(" Subtitles", 0.0, 0.3, segment_index=0),
            TimedWord(" by", 0.3, 0.4, segment_index=0),
            TimedWord(" the", 0.4, 0.5, segment_index=0),
            TimedWord(" Amara.org", 0.5, 0.8, segment_index=0),
            TimedWord(" community.", 0.8, 1.1, segment_index=0),
        ]

        kept = diarizer._filter_asr_no_speech_words(credit)

        self.assertEqual(kept, credit)

    def test_asr_hallucination_verification_rejects_unstable_low_evidence_segment(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_no_speech_filter=True,
            asr_no_speech_prob_threshold=0.65,
            asr_no_speech_hard_threshold=0.85,
            asr_no_speech_keep_short_max_words=2,
            asr_no_speech_keep_short_max_seconds=0.45,
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Thanks", 0.0, 0.5, probability=0.0019, no_speech_prob=0.3403, avg_logprob=-0.9448, segment_index=0),
            TimedWord(" for", 0.5, 0.6, probability=0.8003, no_speech_prob=0.3403, avg_logprob=-0.9448, segment_index=0),
            TimedWord(" watching!", 0.6, 1.22, probability=0.7334, no_speech_prob=0.3403, avg_logprob=-0.9448, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=([], 0))  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(object(), np.zeros(22_240, dtype=np.float32), 16_000, words)

        self.assertEqual(kept, [])
        self.assertEqual(len(diarizer._asr_review_candidates), 1)
        self.assertEqual(diarizer._asr_review_candidates[0]["text"], "Thanks for watching!")
        self.assertEqual(diarizer._asr_review_candidates[0]["policy_rule"], "thanks_for_watching")
        self.assertTrue(
            any(
                "ASR review: suppressed" in item["payload"]["message"]
                for item in diarizer.bus.records
            )
        )

    def test_later_retained_view_clears_transient_suppression_warning(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        weak = [
            TimedWord(" Thanks", 0.0, 0.4, probability=0.20, no_speech_prob=0.40, avg_logprob=-0.90, segment_index=0),
            TimedWord(" for", 0.4, 0.6, probability=0.20, no_speech_prob=0.40, avg_logprob=-0.90, segment_index=0),
            TimedWord(" watching", 0.6, 1.1, probability=0.20, no_speech_prob=0.40, avg_logprob=-0.90, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=([], 0))  # type: ignore[method-assign]
        self.assertEqual(
            diarizer._verify_low_evidence_asr_words(
                object(),
                np.zeros(22_000, dtype=np.float32),
                16_000,
                weak,
                media_start_seconds=0.0,
                media_duration_seconds=300.0,
            ),
            [],
        )
        self.assertEqual(len(diarizer._asr_review_candidates), 1)

        strong = [
            TimedWord(" Thanks", 0.05, 0.45, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
            TimedWord(" for", 0.45, 0.65, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
            TimedWord(" watching", 0.65, 1.15, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
        ]
        kept = diarizer._verify_low_evidence_asr_words(
            object(),
            np.zeros(22_000, dtype=np.float32),
            16_000,
            strong,
            media_start_seconds=0.0,
            media_duration_seconds=300.0,
        )

        self.assertEqual(kept, strong)
        self.assertEqual(diarizer._asr_review_candidates, [])
        self.assertTrue(
            any(
                "were cleared" in item["payload"]["message"]
                for item in diarizer.bus.records
            )
        )

    def test_short_retained_fragment_does_not_clear_suppressed_phrase_warning(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._record_suppressed_asr_candidate({
            "text": "Thanks for watching!",
            "start": 0.0,
            "end": 1.1,
            "policy_rule": "thanks_for_watching",
        })

        cleared = diarizer._reconcile_suppressed_asr_candidates(
            [TimedWord(" Thanks", 0.05, 0.45, segment_index=0)],
            media_start_seconds=0.0,
        )

        self.assertEqual(cleared, 0)
        self.assertEqual(len(diarizer._asr_review_candidates), 1)

    def test_exact_high_risk_phrase_without_acoustic_metadata_fails_open(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Thanks", 0.0, 0.4, segment_index=0),
            TimedWord(" for", 0.4, 0.6, segment_index=0),
            TimedWord(" watching", 0.6, 1.1, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock()  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(
            object(),
            np.zeros(22_000, dtype=np.float32),
            16_000,
            words,
            media_start_seconds=0.0,
            media_duration_seconds=300.0,
        )

        self.assertEqual(kept, words)
        self.assertTrue(all(word.asr_review.get("needs_review") for word in words))
        self.assertEqual(diarizer._asr_review_candidates, [])
        diarizer._transcribe_audio_words.assert_not_called()

    def test_asr_hallucination_verification_keeps_stable_low_evidence_speech(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Hello", 0.0, 0.4, probability=0.02, no_speech_prob=0.34, avg_logprob=-0.95, segment_index=0),
            TimedWord(" there", 0.4, 0.9, probability=0.75, no_speech_prob=0.34, avg_logprob=-0.95, segment_index=0),
        ]
        verification = [
            TimedWord(" there", 0.0, 0.5, probability=0.80, no_speech_prob=0.05, avg_logprob=-0.30, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=(verification, 1))  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(object(), np.zeros(18_400, dtype=np.float32), 16_000, words)

        self.assertEqual(kept, words)

    def test_asr_hallucination_policy_rechecks_medium_evidence_watching_phrase(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Thanks", 0.0, 0.4, probability=0.60, no_speech_prob=0.20, avg_logprob=-0.50, segment_index=0),
            TimedWord(" for", 0.4, 0.6, probability=0.60, no_speech_prob=0.20, avg_logprob=-0.50, segment_index=0),
            TimedWord(" watching", 0.6, 1.1, probability=0.60, no_speech_prob=0.20, avg_logprob=-0.50, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=([], 0))  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(
            object(),
            np.zeros(22_000, dtype=np.float32),
            16_000,
            words,
            media_start_seconds=0.0,
            media_duration_seconds=300.0,
        )

        self.assertEqual(kept, [])
        diarizer._transcribe_audio_words.assert_called_once()
        self.assertTrue(any("thanks_for_watching" in item["payload"]["message"] for item in diarizer.bus.records))

    def test_asr_hallucination_policy_keeps_strongly_supported_watching_phrase(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Thanks", 0.0, 0.4, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
            TimedWord(" for", 0.4, 0.6, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
            TimedWord(" watching", 0.6, 1.1, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock()  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(
            object(),
            np.zeros(22_000, dtype=np.float32),
            16_000,
            words,
            media_start_seconds=0.0,
            media_duration_seconds=300.0,
        )

        self.assertEqual(kept, words)
        diarizer._transcribe_audio_words.assert_not_called()

    def test_asr_hallucination_policy_never_deletes_surrounding_real_speech(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        tokens = " Before we go thank you for watching and please subscribe".split(" ")
        words = [
            TimedWord(
                f" {token}",
                index * 0.25,
                (index + 1) * 0.25,
                probability=0.20,
                no_speech_prob=0.40,
                avg_logprob=-0.90,
                segment_index=0,
            )
            for index, token in enumerate(token for token in tokens if token)
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=([], 0))  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(
            object(),
            np.zeros(48_000, dtype=np.float32),
            16_000,
            words,
            media_start_seconds=0.0,
            media_duration_seconds=300.0,
        )

        self.assertEqual(kept, words)
        self.assertTrue(all(word.asr_review.get("needs_review") for word in words))
        self.assertEqual(diarizer._asr_review_candidates, [])

    def test_asr_hallucination_verification_retains_uncertain_ordinary_speech(self) -> None:
        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            asr_hallucination_verification=True,
            asr_hallucination_suspicion_score=0.45,
            asr_hallucination_verification_shift_seconds=0.20,
            asr_hallucination_verification_context_seconds=0.25,
            asr_hallucination_verification_min_text_similarity=0.50,
        )
        diarizer.bus = RecordingEventBus()
        words = [
            TimedWord(" Hello", 0.0, 0.4, probability=0.02, no_speech_prob=0.34, avg_logprob=-0.95, segment_index=0),
            TimedWord(" there", 0.4, 0.9, probability=0.75, no_speech_prob=0.34, avg_logprob=-0.95, segment_index=0),
        ]
        verification = [
            TimedWord(" Completely", 0.0, 0.4, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
            TimedWord(" different", 0.4, 0.8, probability=0.98, no_speech_prob=0.02, avg_logprob=-0.02, segment_index=0),
        ]
        diarizer._transcribe_audio_words = mock.Mock(return_value=(verification, 1))  # type: ignore[method-assign]

        kept = diarizer._verify_low_evidence_asr_words(object(), np.zeros(18_400, dtype=np.float32), 16_000, words)

        self.assertEqual(kept, words)
        self.assertTrue(all(word.asr_review.get("needs_review") for word in words))
        self.assertTrue(
            any(
                "uncertain ordinary-speech" in item["payload"]["message"]
                for item in diarizer.bus.records
            )
        )

    def test_ybj_real_multiword_speech_survives_moderate_no_speech_probability(self) -> None:
        sample_rate = 16_000
        diarizer = make_window_diarizer(
            audio=np.zeros(int(4.3 * sample_rate), dtype=np.float32),
            sample_rate=sample_rate,
        )
        diarizer.bus = RecordingEventBus()
        primary = SimpleNamespace(
            no_speech_prob=0.6982,
            avg_logprob=-0.6263,
            compression_ratio=1.0,
            words=[
                SimpleNamespace(word=" with", start=0.20, end=0.55, probability=0.3032),
                SimpleNamespace(word=" you", start=0.55, end=0.82, probability=0.8892),
                SimpleNamespace(word=" as", start=0.82, end=1.02, probability=0.8242),
                SimpleNamespace(word=" a", start=1.02, end=1.12, probability=0.9888),
                SimpleNamespace(word=" gesture.", start=1.12, end=1.75, probability=0.8760),
            ],
        )
        verification = SimpleNamespace(
            no_speech_prob=0.6147,
            avg_logprob=-0.8008,
            compression_ratio=1.0,
            words=[
                SimpleNamespace(word=" And", start=0.00, end=0.15, probability=0.4294),
                SimpleNamespace(word=" with", start=0.15, end=0.45, probability=0.8901),
                SimpleNamespace(word=" you", start=0.45, end=0.70, probability=0.7280),
                SimpleNamespace(word=" as", start=0.70, end=0.90, probability=0.9888),
                SimpleNamespace(word=" a", start=0.90, end=1.00, probability=0.8804),
                SimpleNamespace(word=" gesture.", start=1.00, end=1.55, probability=0.8804),
            ],
        )

        class ScriptedAsrModel:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe(self, *_args: object, **_kwargs: object) -> tuple[list[object], object]:
                self.calls += 1
                return ([primary] if self.calls == 1 else [verification]), object()

        model = ScriptedAsrModel()
        words, _segment_count = diarizer._transcribe_window_audio_words(
            model,
            0.0,
            float(diarizer.duration),
        )

        self.assertEqual([word.text for word in words], [" with", " you", " as", " a", " gesture."])
        self.assertEqual(model.calls, 2)
        self.assertTrue(all(word.asr_review.get("needs_review") for word in words))
        self.assertTrue(
            any(
                "no text was discarded on this signal alone" in item["payload"]["message"]
                for item in diarizer.bus.records
            )
        )

    def test_transcribe_window_audio_words_maps_speech_clip_times_to_media_time(self) -> None:
        diarizer = make_window_diarizer(
            audio=np.arange(100, dtype=np.float32),
            sample_rate=10,
        )
        calls: list[int] = []

        def fake_transcribe(_model: object, audio: np.ndarray, sample_rate: int) -> tuple[list[TimedWord], int]:
            calls.append(int(audio.size))
            self.assertEqual(sample_rate, 10)
            return [TimedWord(" word", 0.1, 0.2)], 1

        diarizer._transcribe_audio_words = fake_transcribe  # type: ignore[method-assign]

        words, segment_count = diarizer._transcribe_window_audio_words(
            object(),
            0.0,
            10.0,
            [(2.0, 3.0), (6.0, 7.0)],
        )

        self.assertEqual(calls, [10, 10])
        self.assertEqual(segment_count, 2)
        self.assertEqual([word.text for word in words], [" word", " word"])
        np.testing.assert_allclose([word.start for word in words], [2.1, 6.1])
        np.testing.assert_allclose([word.end for word in words], [2.2, 6.2])


class KrokoPreviewStartupTests(unittest.TestCase):
    def test_kroko_preview_reads_license_options_from_environment(self) -> None:
        from window.window_preview import add_kroko_license_options

        with mock.patch.dict(
            os.environ,
            {
                "REALTIMESTT_KROKO_ONNX_KEY": "test-key",
                "KROKO_ONNX_REFERRALCODE": "test-referral",
            },
        ):
            options: dict[str, object] = {}
            add_kroko_license_options(options)

        self.assertEqual(options["key"], "test-key")
        self.assertEqual(options["referralcode"], "test-referral")

    def test_subprocess_preview_uses_worker_script_without_name_error(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO('{"ready":true}\n')
                self.stderr = io.StringIO("")
                self.returncode = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                self.returncode = 0
                return 0

            def kill(self) -> None:
                self.returncode = -9

        args = argparse.Namespace(
            realtime_preview_request_timeout_seconds=0.2,
            realtime_preview_startup_timeout_seconds=0.5,
            realtime_preview_python=Path(sys.executable),
            realtime_preview_engine="kroko_onnx",
            realtime_preview_model="Kroko-EN-Community-64-L-Streaming-001.data",
            language="de",
            realtime_preview_language="de",
            realtime_preview_provider="cpu",
            realtime_preview_num_threads=2,
            realtime_preview_model_path=None,
            realtime_preview_download_root=None,
            download_root=None,
            realtime_preview_engine_options_json="",
            realtime_preview_realtimestt_root=None,
        )

        with mock.patch("window.window_preview.subprocess.Popen", return_value=FakeProcess()) as popen:
            transcriber = KrokoSubprocessPreviewTranscriber(args)
            transcriber.close()

        command = popen.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("workers.kroko_realtime_preview_worker", command)
        self.assertIn("--language", command)
        self.assertIn("de", command)
        self.assertFalse(any(part.endswith("kroko_realtime_preview_worker.py") for part in command))
        env = popen.call_args.kwargs["env"]
        self.assertIn(str(SRC), str(env.get("PYTHONPATH", "")).split(os.pathsep))


class RemoteWindowAsrClientTests(unittest.TestCase):
    def test_remote_asr_client_sends_configured_language(self) -> None:
        from window.window_remote_asr import RemoteWindowAsrClient

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"words":[],"segment_count":0}'

        captured: dict[str, str] = {}

        def fake_urlopen(request: object, timeout: float) -> FakeResponse:
            captured["url"] = str(getattr(request, "full_url"))
            captured["timeout"] = str(timeout)
            return FakeResponse()

        with mock.patch("window.window_remote_asr.urlopen", side_effect=fake_urlopen):
            client = RemoteWindowAsrClient("http://127.0.0.1:8650", 7.0, language="de")
            words, segment_count = client.transcribe_window(
                np.zeros(160, dtype=np.float32),
                16000,
                5,
                batched=True,
                batch_size=12,
            )

        self.assertEqual(words, [])
        self.assertEqual(segment_count, 0)
        self.assertIn("language=de", captured["url"])
        self.assertIn("batched=true", captured["url"])
        self.assertIn("batch_size=12", captured["url"])
        self.assertEqual(captured["timeout"], "7.0")

    def test_remote_asr_client_retries_transient_http_500(self) -> None:
        from urllib.error import HTTPError

        from window.window_remote_asr import RemoteWindowAsrClient

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"words":[{"word":"Hallo","start":0.0,"end":0.2}],"segment_count":1}'

        error = HTTPError(
            "http://127.0.0.1:8650/transcribe-window",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b"transient"),
        )

        with mock.patch("window.window_remote_asr.urlopen", side_effect=[error, FakeResponse()]) as urlopen:
            with mock.patch("window.window_remote_asr.time.sleep"):
                client = RemoteWindowAsrClient("http://127.0.0.1:8650", 7.0, language="de", retry_attempts=1)
                words, segment_count = client.transcribe_window(np.zeros(160, dtype=np.float32), 16000, 5)

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(segment_count, 1)
        self.assertEqual([word.text for word in words], ["Hallo"])

    def test_remote_asr_client_carries_segment_confidence_to_words(self) -> None:
        from window.window_remote_asr import RemoteWindowAsrClient

        client = RemoteWindowAsrClient("http://127.0.0.1:8650", 7.0, language="de")
        words, segment_count = client._timed_words_from_result({
            "segments": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hallo",
                    "avg_logprob": -0.25,
                    "no_speech_prob": 0.08,
                    "compression_ratio": 1.2,
                    "words": [
                        {"word": " Hallo", "start": 0.0, "end": 0.4, "probability": 0.9},
                    ],
                }
            ],
            "segment_count": 1,
        })

        self.assertEqual(segment_count, 1)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, " Hallo")
        self.assertEqual(words[0].probability, 0.9)
        self.assertEqual(words[0].no_speech_prob, 0.08)
        self.assertEqual(words[0].avg_logprob, -0.25)
        self.assertEqual(words[0].compression_ratio, 1.2)
        self.assertEqual(words[0].segment_index, 1)


class WindowDiarizerWarmupTests(unittest.TestCase):
    def test_remote_asr_warmup_failure_does_not_abort_startup(self) -> None:
        controller = make_window_diarizer()
        controller.args = argparse.Namespace(asr_backend="remote")
        controller.bus = RecordingEventBus()
        controller.sample_rate = 16000
        controller._model = object()
        controller._asr_probe_warmed = False
        controller._asr_probe_warmed_at = None
        controller._load_model = lambda: None
        controller._audio_window_copy = lambda _left, _right: (np.zeros(12000, dtype=np.float32), 16000)

        def fail_transcribe(_model: object, _audio: np.ndarray, _sample_rate: int) -> tuple[list[TimedWord], int]:
            raise RuntimeError("Remote ASR HTTP 500: Internal Server Error")

        controller._transcribe_audio_words = fail_transcribe

        controller._warm_asr_transcription()

        self.assertFalse(controller._asr_probe_warmed)
        messages = [str(record["payload"].get("message") or "") for record in controller.bus.records]
        self.assertTrue(any("Remote ASR warmup failed" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
