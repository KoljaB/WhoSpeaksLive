from __future__ import annotations

import argparse
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_domain import TimedWord, VadWindowState
from window.window_events import RecordingEventBus



from tests.window_diarizer_support import make_window_diarizer


class WindowEventBusTests(unittest.TestCase):
    def test_status_event_survives_detached_console(self) -> None:
        bus = RecordingEventBus()
        with mock.patch("builtins.print", side_effect=OSError(22, "Invalid argument")):
            bus.emit("status", {"message": "Loading media"})

        self.assertEqual(bus.records[0]["event"], "status")
        self.assertEqual(bus.records[0]["payload"], {"message": "Loading media"})

    def test_recording_event_bus_records_json_safe_payloads(self) -> None:
        bus = RecordingEventBus()

        bus.emit("validation_replay_start", {"replay_speed": 1.0})

        self.assertEqual(bus.records[0]["event"], "validation_replay_start")
        self.assertEqual(bus.records[0]["payload"], {"replay_speed": 1.0})
        self.assertIsInstance(bus.records[0]["time"], float)


class LanguageConfigTests(unittest.TestCase):
    def test_language_config_maps_discussion_languages_to_runtime_components(self) -> None:
        from window.language_config import (
            SUPPORTED_LANGUAGE_CODES,
            default_sentence_tokenizer,
            kroko_preview_model_name,
            normalize_language_code,
        )

        self.assertEqual(len(SUPPORTED_LANGUAGE_CODES), 60)
        self.assertEqual(normalize_language_code("Deutsch"), "de")
        self.assertEqual(kroko_preview_model_name("de"), "Kroko-DE-Community-64-L-Streaming-001.data")
        self.assertEqual(default_sentence_tokenizer("de"), "nltk+rule-based")

        self.assertEqual(normalize_language_code("iw"), "he")
        self.assertEqual(kroko_preview_model_name("he"), "Kroko-IW-Community-64-L-Streaming-001.data")
        self.assertEqual(default_sentence_tokenizer("he"), "rule-based")

        self.assertEqual(default_sentence_tokenizer("pl"), "nltk+rule-based")
        self.assertEqual(default_sentence_tokenizer("ml"), "nltk+rule-based")
        self.assertEqual(default_sentence_tokenizer("zh"), "stanza")
        self.assertEqual(default_sentence_tokenizer("nn"), "stanza")
        with self.assertRaisesRegex(ValueError, "Kroko realtime preview"):
            kroko_preview_model_name("pl")

    def test_every_supported_language_has_a_bundled_flag_asset(self) -> None:
        from window.language_config import (
            LANGUAGE_FLAG_COUNTRY_CODES,
            SUPPORTED_LANGUAGE_CODES,
            language_flag_country_code,
        )

        self.assertEqual(set(LANGUAGE_FLAG_COUNTRY_CODES), set(SUPPORTED_LANGUAGE_CODES))
        flag_dir = SRC / "window" / "assets" / "flags" / "4x3"
        missing = [
            code for code in SUPPORTED_LANGUAGE_CODES
            if not (flag_dir / f"{language_flag_country_code(code)}.svg").is_file()
        ]
        self.assertEqual(missing, [])


class WindowParserTests(unittest.TestCase):
    def parse_window_args(self, *args: str) -> argparse.Namespace:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["whospeaks-window", "--realtime-preview-engine", "off", *args]):
            return parse_args()

    def test_demo_seat_lease_is_opt_in(self) -> None:
        self.assertFalse(self.parse_window_args().demo_seat_lease)
        self.assertTrue(self.parse_window_args("--demo-seat-lease").demo_seat_lease)


class PublicEventNormalizerTests(unittest.TestCase):
    def test_final_unknown_and_later_assignment_emit_stable_events(self) -> None:
        from window.public_events import PublicEventNormalizer

        normalizer = PublicEventNormalizer(session_id="test-session")

        final_unknown = normalizer.normalize(
            "sentence",
            {
                "index": 7,
                "text": "We should track this.",
                "pending": False,
                "assigned_speaker": None,
                "start": 1.0,
                "end": 2.5,
                "unknown_probability": 1.0,
                "assignment_source": "embedding",
            },
        )

        self.assertEqual([event["type"] for event in final_unknown], ["transcript.final", "transcript.final_unknown"])
        self.assertEqual(final_unknown[0]["session_id"], "test-session")
        self.assertEqual(final_unknown[0]["payload"]["id"], "7")
        self.assertIsNone(final_unknown[0]["payload"]["speaker"])

        revised = normalizer.normalize(
            "sentence",
            {
                "index": 7,
                "text": "We should track this.",
                "pending": False,
                "revision": True,
                "revision_from": "UNKNOWN",
                "revision_to": "S2",
                "assigned_speaker": "S2",
                "start": 1.0,
                "end": 2.5,
                "unknown_probability": 0.0,
                "assignment_source": "retro",
            },
        )

        self.assertEqual([event["type"] for event in revised], ["transcript.speaker_revised", "transcript.speaker_assigned"])
        self.assertEqual(revised[0]["payload"]["previous_speaker"], None)
        self.assertEqual(revised[0]["payload"]["new_speaker"], "S2")

    def test_speaker_events_detect_create_rename_and_state_change(self) -> None:
        from window.public_events import PublicEventNormalizer

        normalizer = PublicEventNormalizer()
        first = normalizer.normalize(
            "speakers",
            {
                "group_name": "",
                "embedding_provider": "espnet",
                "speakers": [
                    {
                        "id": "S1",
                        "name": "",
                        "display_name": "Speaker 1",
                        "source": "detected",
                        "locked": False,
                        "sentence_count": 1,
                        "speech_seconds": 2.5,
                    }
                ],
            },
        )

        self.assertEqual([event["type"] for event in first], ["speaker.created", "speaker.state_changed"])

        renamed = normalizer.normalize(
            "speakers",
            {
                "group_name": "",
                "embedding_provider": "espnet",
                "speakers": [
                    {
                        "id": "S1",
                        "name": "Alice",
                        "display_name": "Alice",
                        "source": "detected",
                        "locked": False,
                        "sentence_count": 1,
                        "speech_seconds": 2.5,
                    }
                ],
            },
        )

        self.assertEqual([event["type"] for event in renamed], ["speaker.renamed", "speaker.state_changed"])
        self.assertEqual(renamed[0]["payload"]["speaker_id"], "S1")
        self.assertEqual(renamed[0]["payload"]["previous_name"], "")
        self.assertEqual(renamed[0]["payload"]["new_name"], "Alice")

    def test_speaker_snapshot_seeds_state_without_created_events(self) -> None:
        from window.public_events import PublicEventNormalizer

        normalizer = PublicEventNormalizer()
        snapshot = normalizer.speaker_snapshot({
            "group_name": "daily",
            "embedding_provider": "espnet",
            "speakers": [{"id": "S1", "name": "Alice", "display_name": "Alice"}],
        })

        self.assertEqual([event["type"] for event in snapshot], ["speaker.snapshot"])
        self.assertEqual(snapshot[0]["payload"]["speakers"][0]["speaker_id"], "S1")

        unchanged = normalizer.normalize("speakers", snapshot[0]["payload"]["raw"])

        self.assertEqual(unchanged, [])


class WindowSentenceTextTests(unittest.TestCase):
    def test_transcribe_window_capitalizes_after_previous_strong_sentence_boundary(self) -> None:
        import window.window_text as window_text

        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            unstable_tail_seconds=0.0,
            sentence_boundary_pre_padding_seconds=0.06,
            sentence_boundary_post_padding_seconds=0.09,
            sentence_boundary_gap_ratio=0.6,
        )
        diarizer._audio_window_copy = mock.Mock(return_value=(np.zeros(160, dtype=np.float32), 16000))
        diarizer._transcribe_audio_words = mock.Mock(return_value=(
            [
                TimedWord("was", 0.0, 0.2),
                TimedWord("Beethoven", 0.25, 0.6),
                TimedWord("good", 0.65, 0.85),
                TimedWord("at", 0.9, 1.0),
                TimedWord("music?", 1.05, 1.25),
            ],
            1,
        ))

        with mock.patch.object(window_text, "generate_sentences", return_value=["was Beethoven good at music?"]):
            transcript = diarizer._transcribe_window(
                object(),
                160.2,
                162.6,
                final_flush=True,
                previous_text_ended_sentence=True,
            )

        self.assertEqual(transcript.sentences[0].text, "Was Beethoven good at music?")

    def test_stream2sentence_result_is_split_at_ellipsis_boundary(self) -> None:
        import window.window_text as window_text

        words = [
            TimedWord("Many", 0.0, 0.2),
            TimedWord("people", 0.25, 0.55),
            TimedWord("do", 0.6, 0.75),
            TimedWord("this...", 0.8, 1.1),
            TimedWord("Mia,", 1.2, 1.35),
            TimedWord("I", 1.4, 1.5),
            TimedWord("must", 1.55, 1.75),
            TimedWord("interrupt.", 1.8, 2.1),
        ]

        with mock.patch.object(
            window_text,
            "generate_sentences",
            return_value=["Many people do this... Mia, I must interrupt."],
        ):
            parts = window_text.split_words_with_stream2sentence(
                words,
                left=0.0,
                right=2.5,
                unstable_tail_seconds=0.0,
                final_flush=True,
            )

        self.assertEqual([part.text for part in parts], ["Many people do this...", "Mia, I must interrupt."])
        self.assertEqual(parts[0].words[-1]["text"], "this...")
        self.assertEqual(parts[1].words[0]["text"], "Mia,")

        with mock.patch.object(
            window_text,
            "generate_sentences",
            return_value=["Many people do this... Mia, I must interrupt"],
        ):
            live_parts = window_text.split_words_with_stream2sentence(
                words,
                left=0.0,
                right=2.5,
                unstable_tail_seconds=0.0,
                final_flush=False,
            )

        self.assertEqual([part.text for part in live_parts], ["Many people do this..."])

    def test_asr_word_review_is_preserved_on_sentence_part_and_payload(self) -> None:
        import window.window_text as window_text

        review = {
            "needs_review": True,
            "reasons": ["conflicting ASR speech evidence"],
            "details": {"no_speech_probability": 0.6982},
        }
        words = [
            TimedWord("Would", 0.0, 0.3, asr_review=review),
            TimedWord("you?", 0.3, 0.7, asr_review=review),
        ]
        with mock.patch.object(window_text, "generate_sentences", return_value=["Would you?"]):
            parts = window_text.split_words_with_stream2sentence(
                words,
                left=0.0,
                right=1.0,
                unstable_tail_seconds=0.0,
                final_flush=True,
            )

        self.assertTrue(parts[0].asr_review["needs_review"])
        self.assertEqual(parts[0].asr_review["reasons"], ["conflicting ASR speech evidence"])
        payload = make_window_diarizer()._base_payload_from_sentence_part(0, parts[0], 0.0, 1.0)
        self.assertEqual(payload["asr_review"], parts[0].asr_review)

    def test_realtime_preview_capitalizes_session_start_and_after_strong_sentence(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._final_sentence_count = 0
        diarizer._last_final_sentence_ended_strong = False

        self.assertEqual(diarizer._format_realtime_preview_text("hello there", 0.0), "Hello there")

        diarizer._final_sentence_count = 1
        diarizer._last_final_sentence_ended_strong = True
        self.assertEqual(diarizer._format_realtime_preview_text("next idea", 8.0), "Next idea")

        diarizer._last_final_sentence_ended_strong = False
        self.assertEqual(diarizer._format_realtime_preview_text("still continuing", 9.0), "still continuing")

    def test_run_treats_first_final_sentence_as_sentence_start(self) -> None:
        diarizer = make_window_diarizer(
            audio=np.zeros(10, dtype=np.float32),
            sample_rate=10,
        )
        diarizer._update_config(
            interval_seconds=0.0,
            min_playback_advance_seconds=0.0,
            min_window_seconds=0.0,
            final_flush_epsilon_seconds=0.01,
            vad_sentence_splitting=False,
            asr_vad_gate=False,
        )
        diarizer._model = object()
        diarizer.set_playback_time(diarizer.duration, reset=True)
        diarizer._vad_window_state = mock.Mock(
            return_value=VadWindowState(has_speech=True, should_flush=False)
        )
        diarizer._asr_vad_gate_enabled = mock.Mock(return_value=False)
        sentence = argparse.Namespace(text="First complete sentence.", next_left=1.0)
        diarizer._transcribe_window = mock.Mock(
            return_value=argparse.Namespace(
                sentences=[sentence],
                segment_count=1,
                word_count=3,
            )
        )
        diarizer._emit_sentence = mock.Mock()
        diarizer._advance_realtime_preview_after_commit = mock.Mock()
        diarizer._pause_realtime_preview = mock.Mock()
        diarizer._drain_embedding_jobs = mock.Mock()
        diarizer._revisit_unknown_sentences = mock.Mock()
        diarizer._finalize_speaker_refinement = mock.Mock()
        diarizer._drain_live_memory_update_jobs = mock.Mock()

        diarizer._run(threading.Event())

        self.assertTrue(
            diarizer._transcribe_window.call_args.kwargs[
                "previous_text_ended_sentence"
            ]
        )
        self.assertEqual(diarizer._final_sentence_count, 1)
        self.assertTrue(diarizer._last_final_sentence_ended_strong)


if __name__ == "__main__":
    unittest.main()
