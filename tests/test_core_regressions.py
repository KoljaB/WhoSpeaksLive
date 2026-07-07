from __future__ import annotations

import argparse
import inspect
import importlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import write_wav
from embeddings.embedding_providers import EmbeddingSubprocessClient, RemoteEmbeddingClient
from speakers.realtime_speaker_memory import SpeakerMemory as RealtimeSpeakerMemory
from speakers.speaker_embedding_cluster import SpeakerMemory as ClusterSpeakerMemory
from window.window_diarizer import WindowDiarizer
from window.window_domain import LiveSpeakerMemoryUpdateJob, TimedWord, VadWindowState
from window.window_events import RecordingEventBus
from window.window_gui_html import HTML
from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_probe_scoring import score_live_speaker_probe
from window.window_preview import KrokoSubprocessPreviewTranscriber


def realtime_memory() -> RealtimeSpeakerMemory:
    return RealtimeSpeakerMemory(
        same_speaker_similarity=0.45,
        similarity_temperature=0.07,
        speaker_softmax_temperature=0.075,
        new_speaker_threshold=0.58,
        duplicate_profile_similarity=0.40,
        unknown_short_threshold=0.86,
        min_first_speaker_seconds=0.1,
        min_new_speaker_seconds=1.0,
        late_new_speaker_min_seconds=3.5,
        max_speakers=10,
        min_margin=0.08,
        margin_temperature=0.05,
        update_unknown_max=0.55,
    )


class SpeakerDecisionContractTests(unittest.TestCase):
    def assert_created_speaker_probability_contract(self, decision: object) -> None:
        self.assertEqual(decision.assigned_speaker, "S1")
        self.assertTrue(decision.created_speaker)
        self.assertEqual(decision.probabilities.get("unknown"), 0.0)
        self.assertEqual(decision.unknown_probability, 0.0)
        self.assertEqual(decision.probabilities.get("speaker1"), 1.0)

    def test_realtime_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        decision = realtime_memory().classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)
        decision = memory.classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_upsert_keeps_explicit_speaker_label(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)

        label = memory.upsert_profile("S3", np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)
        decision = memory.score_existing(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)

        self.assertEqual(label, "S3")
        self.assertEqual(decision.assigned_speaker, "S3")


class WindowEventBusTests(unittest.TestCase):
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

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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

    def test_realtime_preview_capitalizes_session_start_and_after_strong_sentence(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._final_sentence_count = 0
        diarizer._last_final_sentence_ended_strong = False

        self.assertEqual(diarizer._format_realtime_preview_text("hello there", 0.0), "Hello there")

        diarizer._final_sentence_count = 1
        diarizer._last_final_sentence_ended_strong = True
        self.assertEqual(diarizer._format_realtime_preview_text("next idea", 8.0), "Next idea")

        diarizer._last_final_sentence_ended_strong = False
        self.assertEqual(diarizer._format_realtime_preview_text("still continuing", 9.0), "still continuing")

    def test_run_treats_first_final_sentence_as_sentence_start(self) -> None:
        source = inspect.getsource(WindowDiarizer._run)

        self.assertIn("previous_emitted_sentence_ended_strong = True", source)
        self.assertIn("self._last_final_sentence_ended_strong = previous_emitted_sentence_ended_strong", source)
        self.assertIn('self._final_sentence_count = int(getattr(self, "_final_sentence_count", 0)) + 1', source)


class ScoreParityTests(unittest.TestCase):
    def load_current_memory_optimizer(self) -> object:
        module_path = ROOT / "runtime" / "optimization" / "optimize_current_memory_21.py"
        if not module_path.is_file():
            self.skipTest(f"Local optimizer harness is not present: {module_path}")
        spec = importlib.util.spec_from_file_location("score_parity_current_memory_optimizer", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def base_sentence_payload(self) -> dict[str, object]:
        return {
            "index": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "same speaker evidence",
            "speech_audio_ratio": 1.0,
        }

    def unknown_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "assigned_speaker": None,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "assignment_source": "embedding",
        }

    def confirmed_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "retro_reassigned": True,
            "revision_from": "S3",
            "revision_to": "S6",
            "assigned_speaker": "S6",
            "probabilities": {"unknown": 0.0, "speaker6": 1.0},
            "similarities": {"S6": 0.82},
            "unknown_probability": 0.0,
            "assignment_source": "retro",
        }

    def canonical(self) -> list[dict[str, object]]:
        return [
            {
                "speaker": "canonical_speaker",
                "start": 0.0,
                "end": 3.0,
                "text": "same speaker evidence",
            }
        ]

    def test_current_memory_optimizer_matches_live_memory_path_when_live_refinement_is_disabled(self) -> None:
        optimizer = self.load_current_memory_optimizer()
        from window.window_validation_replay import replay_cached_window_diarizer

        config = dict(optimizer.BASE_CONFIG)
        weights = {"espnet_ecapa_wavlm_joint": 1.0}
        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.4,
                "text": "alpha beta gamma delta",
                "spoken_word_seconds": 2.4,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 2.5,
                "end": 4.7,
                "text": "alpha beta gamma delta again",
                "spoken_word_seconds": 2.2,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 5.0,
                "end": 5.8,
                "text": "epsilon zeta eta theta",
                "spoken_word_seconds": 0.8,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 3,
                "start": 6.0,
                "end": 8.7,
                "text": "epsilon zeta eta theta longer",
                "spoken_word_seconds": 2.7,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 4,
                "start": 9.0,
                "end": 11.1,
                "text": "alpha beta gamma returns",
                "spoken_word_seconds": 2.1,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.99, 0.05], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.02], dtype=np.float32),
        ]
        dataset = argparse.Namespace(
            sentences=sentences,
            embeddings={"espnet_ecapa_wavlm_joint": embeddings},
        )

        optimizer_rows = optimizer.replay_current_memory(dataset, weights, config)
        live_args = argparse.Namespace(
            **config,
            min_embed_seconds=0.0,
            section_gap_new_speaker=False,
            unknown_pair_new_speaker=False,
            speaker_refinement=False,
        )
        live_rows = replay_cached_window_diarizer(sentences, embeddings, live_args).final_payloads

        self.assertEqual(len(live_rows), len(optimizer_rows))
        for live, optimized in zip(live_rows, optimizer_rows):
            self.assertEqual(live["index"], optimized["index"])
            self.assertEqual(live.get("assigned_speaker"), optimized.get("assigned_speaker"))
            self.assertEqual(live.get("created_speaker"), optimized.get("created_speaker"))
            self.assertEqual(live.get("assignment_source"), optimized.get("assignment_source"))
            self.assertEqual(live.get("probabilities"), optimized.get("probabilities"))
            self.assertEqual(live.get("similarities"), optimized.get("similarities"))
            self.assertEqual(live.get("unknown_probability"), optimized.get("unknown_probability"))
            self.assertEqual(live.get("top_similarity"), optimized.get("top_similarity"))
            self.assertEqual(live.get("margin"), optimized.get("margin"))

        self.assertEqual([row.get("assigned_speaker") for row in live_rows], ["S1", "S1", "S2", "S2", "S1"])
        self.assertTrue(live_rows[2].get("retro_reassigned"))

    def test_current_memory_optimizer_uses_fast_cached_live_replay(self) -> None:
        optimizer = self.load_current_memory_optimizer()
        from window.window_validation_replay import replay_cached_window_diarizer

        evaluate_source = inspect.getsource(optimizer.evaluate_candidate)
        live_replay_source = inspect.getsource(optimizer.replay_current_live)
        process_source = inspect.getsource(WindowDiarizer._process_sentence_embedding)
        fast_replay_source = inspect.getsource(replay_cached_window_diarizer)

        self.assertIn("replay_current_live", evaluate_source)
        self.assertNotIn("replay_current_memory(prepared.dataset", evaluate_source)
        self.assertIn("replay_cached_window_diarizer", live_replay_source)
        self.assertIn("make_cached_replay_args", live_replay_source)
        self.assertIn("_apply_sentence_embedding_decision", process_source)
        self.assertIn("_apply_sentence_embedding_decision", fast_replay_source)

    def test_cached_live_replay_scores_committed_prototype_reassignment(self) -> None:
        from window.window_validation_replay import replay_cached_window_diarizer, replay_cached_window_score
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()
        args.min_embed_seconds = 0.0
        args.section_gap_new_speaker = False
        args.unknown_pair_new_speaker = False
        args.speaker_refinement = True
        args.allow_speaker_reassignment = True
        args.min_new_speaker_words = 3
        args.known_speaker_gray_zone_min_unknown_probability = 1.1

        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "alpha beta anchor",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 3.2,
                "end": 6.2,
                "text": "gamma delta",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 6.4,
                "end": 9.4,
                "text": "gamma delta epsilon",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(sentences, embeddings, args)
        chronological_replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {payload["index"]: payload for payload in replay.final_payloads}
        chronological_final_by_index = {
            payload["index"]: payload
            for payload in chronological_replay.final_payloads
        }

        self.assertEqual(final_by_index[0]["assigned_speaker"], "S1")
        self.assertEqual(final_by_index[1]["assigned_speaker"], "S2")
        self.assertEqual(final_by_index[1]["assignment_source"], "prototype_reassign")
        self.assertEqual(final_by_index[2]["assigned_speaker"], "S2")
        self.assertEqual(
            [chronological_final_by_index[index]["assigned_speaker"] for index in sorted(chronological_final_by_index)],
            [final_by_index[index]["assigned_speaker"] for index in sorted(final_by_index)],
        )
        self.assertTrue(any(
            record.get("event") == "sentence"
            and (record.get("payload") or {}).get("index") == 1
            and (record.get("payload") or {}).get("prototype_reassigned")
            and not (record.get("payload") or {}).get("provisional_assignment")
            for record in replay.records
        ))

        canonical = [
            {"speaker": "speaker_a", "start": 0.0, "end": 3.0, "text": "alpha beta anchor"},
            {"speaker": "speaker_b", "start": 3.2, "end": 6.2, "text": "gamma delta"},
            {"speaker": "speaker_b", "start": 6.4, "end": 9.4, "text": "gamma delta epsilon"},
        ]
        score = replay_cached_window_score(sentences, embeddings, args, canonical, match_mode="timestamp")

        self.assertEqual(score["rows"][1]["assigned_speaker"], "S2")
        self.assertEqual(score["assigned_counts"], {"S1": 1, "S2": 2})
        self.assertEqual(score["duration_accuracy"], 1.0)

    def test_small_island_refinement_merges_oneoff_flanked_speaker(self) -> None:
        from window.window_validation_replay import replay_cached_window_diarizer
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()
        args.min_embed_seconds = 0.0
        args.min_first_speaker_seconds = 0.1
        args.min_new_speaker_seconds = 0.1
        args.late_new_speaker_min_seconds = 0.1
        args.min_new_speaker_words = 3
        args.speaker_refinement = True
        args.speaker_refinement_unknown_tentative = False
        args.speaker_refinement_unknown_commit = False
        args.allow_speaker_reassignment = False
        args.speaker_refinement_small_island_merge = True
        args.speaker_refinement_small_island_max_duration = 5.0
        args.speaker_refinement_small_island_max_segments = 3

        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "alpha beta anchor",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 2.2,
                "end": 4.2,
                "text": "gamma delta anchor",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 4.4,
                "end": 5.4,
                "text": "brief different island",
                "spoken_word_seconds": 1.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 3,
                "start": 5.6,
                "end": 7.6,
                "text": "gamma delta returns",
                "spoken_word_seconds": 2.0,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.99, 0.01], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {
            payload["index"]: payload
            for payload in replay.final_payloads
        }

        self.assertEqual(final_by_index[2]["assigned_speaker"], "S2")
        self.assertTrue(final_by_index[2].get("small_island_merged"))
        self.assertEqual(final_by_index[2].get("small_island_merged_from"), "S3")
        self.assertEqual(final_by_index[2].get("assignment_source"), "small_island_merge")

    def test_speaker_refinement_split_switches_default_on(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()

        self.assertTrue(args.speaker_refinement_unknown_tentative)
        self.assertTrue(args.speaker_refinement_unknown_commit)
        self.assertTrue(args.allow_speaker_reassignment)
        self.assertEqual(args.speaker_refinement_known_min_delta, 0.04)
        self.assertEqual(args.speaker_refinement_final_passes, 1)
        self.assertTrue(args.speaker_refinement_small_island_merge)
        self.assertTrue(args.speaker_refinement_tiny_fragmented_merge)
        self.assertEqual(args.speaker_refinement_tiny_fragmented_max_islands, 3)
        self.assertTrue(args.speaker_refinement_terminal_outro_merge)
        self.assertEqual(args.speaker_refinement_terminal_outro_max_duration, 12.0)
        self.assertTrue(args.speaker_refinement_unknown_same_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_same_speaker_max_duration, 3.0)
        self.assertEqual(args.speaker_refinement_unknown_same_speaker_max_segments, 1)
        self.assertTrue(args.speaker_refinement_unknown_previous_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_duration, 0.75)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_segments, 1)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_max_previous_gap, 0.35)
        self.assertEqual(args.speaker_refinement_unknown_previous_speaker_min_next_gap, 0.3)
        self.assertTrue(args.speaker_refinement_unknown_next_speaker_fill)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_duration, 1.75)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_segments, 1)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_max_next_gap, 0.05)
        self.assertEqual(args.speaker_refinement_unknown_next_speaker_min_previous_gap, 0.15)
        self.assertTrue(args.speaker_refinement_long_low_confidence_retro_split)
        self.assertEqual(args.speaker_refinement_long_low_confidence_retro_max_similarity, 0.06)

    def test_final_speaker_refinement_runs_configured_passes(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_final_passes=2,
            speaker_refinement_tiny_fragmented_merge=False,
            speaker_refinement_terminal_outro_merge=False,
            speaker_refinement_long_low_confidence_retro_split=False,
            speaker_refinement_unknown_same_speaker_fill=False,
            speaker_refinement_unknown_previous_speaker_fill=False,
            speaker_refinement_unknown_next_speaker_fill=False,
        )
        calls = 0

        def refine() -> None:
            nonlocal calls
            calls += 1

        diarizer._refine_speaker_assignments = refine

        diarizer._finalize_speaker_refinement()

        self.assertEqual(calls, 2)

    def test_tiny_fragmented_refinement_merges_dominant_neighbor_profile(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_tiny_fragmented_merge=True,
            speaker_refinement_tiny_fragmented_max_duration=6.0,
            speaker_refinement_tiny_fragmented_max_segments=8,
            speaker_refinement_tiny_fragmented_min_islands=2,
            speaker_refinement_tiny_fragmented_max_islands=3,
            speaker_refinement_tiny_fragmented_min_neighbor_share=0.5,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, duration: float = 1.0) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.6, "S2": 0.4},
                "unknown_probability": 0.0,
                "top_similarity": 0.6,
                "margin": 0.2,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 2.0),
            1: record(1, "S3"),
            2: record(2, "S2", 2.0),
            3: record(3, "S2", 2.0),
            4: record(4, "S3"),
            5: record(5, "S2", 2.0),
        }

        self.assertEqual(diarizer._merge_tiny_fragmented_speaker_profiles(), 2)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(diarizer._sentence_refinement_records[4]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["tiny_fragmented_profile_merged"])
        self.assertEqual(payload["tiny_fragmented_profile_merged_from"], "S3")
        self.assertEqual(payload["assignment_source"], "tiny_fragmented_profile_merge")

    def test_tiny_fragmented_refinement_skips_too_many_islands(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_tiny_fragmented_merge=True,
            speaker_refinement_tiny_fragmented_max_duration=6.0,
            speaker_refinement_tiny_fragmented_max_segments=8,
            speaker_refinement_tiny_fragmented_min_islands=2,
            speaker_refinement_tiny_fragmented_max_islands=3,
            speaker_refinement_tiny_fragmented_min_neighbor_share=0.5,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, duration: float = 0.5) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.6, "S1": 0.4},
                "unknown_probability": 0.0,
                "top_similarity": 0.6,
                "margin": 0.2,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", 1.0),
            1: record(1, "S3"),
            2: record(2, "S1", 1.0),
            3: record(3, "S3"),
            4: record(4, "S1", 1.0),
            5: record(5, "S3"),
            6: record(6, "S1", 1.0),
            7: record(7, "S3"),
            8: record(8, "S1", 1.0),
        }

        self.assertEqual(diarizer._merge_tiny_fragmented_speaker_profiles(), 0)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[3]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[5]["assigned_speaker"], "S3")
        self.assertEqual(diarizer._sentence_refinement_records[7]["assigned_speaker"], "S3")

    def test_terminal_promotional_outro_merges_to_opening_speaker(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_terminal_outro_merge=True,
            speaker_refinement_terminal_outro_max_duration=12.0,
            speaker_refinement_terminal_outro_lookback_segments=2,
            speaker_refinement_terminal_outro_min_target_duration=5.0,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str, text: str, duration: float = 2.0) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": text,
                    "start": float(index * 10),
                    "end": float(index * 10) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 0.0},
                "similarities": {speaker: 0.7},
                "unknown_probability": 0.0,
                "top_similarity": 0.7,
                "margin": 0.5,
                "quality": 1.0,
                "assignment_source": "embedding",
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", "Opening narration from the host.", 6.0),
            1: record(1, "S2", "Main guest answer.", 5.0),
            2: record(2, "S3", "Be sure to like and subscribe on YouTube.", 7.0),
        }

        self.assertEqual(diarizer._merge_terminal_promotional_outro(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S1")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["terminal_promotional_outro_merged"])
        self.assertEqual(payload["terminal_promotional_outro_merged_from"], "S3")
        self.assertEqual(payload["assignment_source"], "terminal_promotional_outro_merge")

    def test_unknown_same_speaker_fill_assigns_short_flanked_unknown(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_same_speaker_fill=True,
            speaker_refinement_unknown_same_speaker_max_duration=3.0,
            speaker_refinement_unknown_same_speaker_max_segments=1,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(index: int, speaker: str | None, duration: float = 1.0) -> dict[str, object]:
            assigned = speaker if speaker is not None else "UNKNOWN"
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": float(index),
                    "end": float(index) + duration,
                },
                "duration_seconds": duration,
                "assigned_speaker": assigned,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": "embedding" if speaker is not None else None,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 2.0),
            1: record(1, None, 0.8),
            2: record(2, "S2", 2.0),
        }

        self.assertEqual(diarizer._fill_unknown_same_speaker_islands(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_same_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S2")
        self.assertEqual(payload["assignment_source"], "unknown_same_speaker_island_fill")

    def test_unknown_previous_speaker_fill_assigns_short_tail_before_pause(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_previous_speaker_fill=True,
            speaker_refinement_unknown_previous_speaker_max_duration=0.6,
            speaker_refinement_unknown_previous_speaker_max_segments=1,
            speaker_refinement_unknown_previous_speaker_max_previous_gap=0.05,
            speaker_refinement_unknown_previous_speaker_min_next_gap=0.15,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 0.0, 2.0),
            1: record(1, None, 2.0, 2.25, "non_embedding_candidate"),
            2: record(2, "S4", 2.7, 4.0),
        }

        self.assertEqual(diarizer._fill_unknown_previous_speaker_tails(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_previous_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S2")
        self.assertEqual(payload["assignment_source"], "unknown_previous_speaker_tail_fill")

    def test_unknown_previous_speaker_fill_updates_scan_for_chained_tails(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_previous_speaker_fill=True,
            speaker_refinement_unknown_previous_speaker_max_duration=0.75,
            speaker_refinement_unknown_previous_speaker_max_segments=1,
            speaker_refinement_unknown_previous_speaker_max_previous_gap=0.35,
            speaker_refinement_unknown_previous_speaker_min_next_gap=0.3,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S2", 0.0, 2.0),
            1: record(1, None, 2.0, 2.25, "non_embedding_candidate"),
            2: record(2, None, 2.58, 2.97, "non_embedding_candidate"),
            3: record(3, "S4", 3.45, 5.0),
        }

        self.assertEqual(diarizer._fill_unknown_previous_speaker_tails(), 2)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S2")
        self.assertEqual(diarizer._sentence_refinement_records[2]["assigned_speaker"], "S2")
        self.assertEqual(diarizer.bus.records[-1]["payload"]["revision_to"], "S2")

    def test_unknown_next_speaker_fill_assigns_short_head_after_pause(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_unknown_next_speaker_fill=True,
            speaker_refinement_unknown_next_speaker_max_duration=0.75,
            speaker_refinement_unknown_next_speaker_max_segments=1,
            speaker_refinement_unknown_next_speaker_max_next_gap=0.05,
            speaker_refinement_unknown_next_speaker_min_previous_gap=0.15,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        def record(
            index: int,
            speaker: str | None,
            start: float,
            end: float,
            source: str = "embedding",
        ) -> dict[str, object]:
            return {
                "index": index,
                "base_payload": {
                    "index": index,
                    "text": f"sentence {index}",
                    "start": start,
                    "end": end,
                },
                "duration_seconds": end - start,
                "assigned_speaker": speaker,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0} if speaker is None else {"unknown": 0.0},
                "similarities": {},
                "unknown_probability": 1.0 if speaker is None else 0.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": source,
            }

        diarizer._sentence_refinement_records = {
            0: record(0, "S1", 0.0, 2.0),
            1: record(1, None, 2.7, 3.35, "non_embedding_candidate"),
            2: record(2, "S3", 3.35, 5.0),
        }

        self.assertEqual(diarizer._fill_unknown_next_speaker_heads(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[1]["assigned_speaker"], "S3")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["unknown_next_speaker_filled"])
        self.assertEqual(payload["revision_from"], "UNKNOWN")
        self.assertEqual(payload["revision_to"], "S3")
        self.assertEqual(payload["assignment_source"], "unknown_next_speaker_head_fill")

    def test_long_low_confidence_retro_split_creates_final_speaker(self) -> None:
        from speakers.speaker_embedding_cluster import SpeakerMemory

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement_long_low_confidence_retro_split=True,
            speaker_refinement_long_low_confidence_retro_min_duration=4.0,
            speaker_refinement_long_low_confidence_retro_max_similarity=0.06,
            speaker_refinement_long_low_confidence_retro_max_margin=0.04,
            speaker_refinement_long_low_confidence_retro_max_splits=1,
        )
        diarizer.bus = RecordingEventBus()
        diarizer.memory = SpeakerMemory()
        diarizer.memory.upsert_profile("S1", np.array([1.0, 0.0, 0.0], dtype=np.float32), duration_seconds=8.0)
        diarizer.memory.upsert_profile("S2", np.array([0.0, 1.0, 0.0], dtype=np.float32), duration_seconds=8.0)
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()

        diarizer._sentence_refinement_records = {
            0: {
                "index": 0,
                "base_payload": {
                    "index": 0,
                    "text": "uncertain long segment",
                    "start": 0.0,
                    "end": 12.0,
                },
                "embedding": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "duration_seconds": 12.0,
                "assigned_speaker": "S1",
                "created_speaker": False,
                "probabilities": {"unknown": 0.0, "speaker1": 1.0},
                "similarities": {"S1": 0.044, "S2": 0.038},
                "unknown_probability": 0.0,
                "top_similarity": 0.044,
                "margin": 0.006,
                "quality": 1.0,
                "assignment_source": "retro",
            }
        }

        self.assertEqual(diarizer._split_long_low_confidence_retro_assignments(), 1)
        self.assertEqual(diarizer._sentence_refinement_records[0]["assigned_speaker"], "S3")
        payload = diarizer.bus.records[-1]["payload"]
        self.assertTrue(payload["long_low_confidence_retro_split"])
        self.assertEqual(payload["long_low_confidence_retro_split_from"], "S1")
        self.assertEqual(payload["assignment_source"], "long_low_confidence_retro_split")

    def test_speaker_refinement_settings_update_split_switches(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=True,
            speaker_refinement_unknown_commit=True,
            allow_speaker_reassignment=True,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._revisit_unknown_sentences = mock.Mock()
        diarizer._refine_speaker_assignments = mock.Mock()

        result = diarizer.set_speaker_refinement_settings({
            "speaker_refinement_unknown_tentative": False,
            "speaker_refinement_unknown_commit": False,
            "allow_speaker_reassignment": False,
        })

        self.assertEqual(
            result,
            {
                "enabled": True,
                "unknown_tentative": False,
                "unknown_commit": False,
                "allow_reassignment": False,
            },
        )
        diarizer._revisit_unknown_sentences.assert_not_called()
        diarizer._refine_speaker_assignments.assert_not_called()

        result = diarizer.set_speaker_refinement_settings({
            "speaker_refinement_unknown_tentative": True,
            "speaker_refinement_unknown_commit": True,
            "allow_speaker_reassignment": False,
        })

        self.assertTrue(result["unknown_tentative"])
        self.assertTrue(result["unknown_commit"])
        diarizer._revisit_unknown_sentences.assert_called_once()
        diarizer._refine_speaker_assignments.assert_called_once()

    def test_unknown_commit_switch_blocks_retro_unknown_revisit(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(speaker_refinement_unknown_commit=False)
        diarizer.memory = mock.Mock()

        diarizer._revisit_unknown_sentences()

        diarizer.memory.score_existing.assert_not_called()

    def test_tentative_unknown_switch_blocks_prototype_unknown_hints_only(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=False,
            allow_speaker_reassignment=True,
        )
        diarizer.bus = RecordingEventBus()
        diarizer._sentence_refinement_run_lock = threading.Lock()
        diarizer._sentence_refinement_lock = threading.Lock()
        diarizer._sentence_refinement_records = {
            1: {"index": 1},
            2: {"index": 2},
        }
        diarizer._apply_prototype_revision = mock.Mock(return_value=True)
        unknown_revision = argparse.Namespace(
            index=1,
            previous_speaker=None,
            assigned_speaker="S2",
        )
        known_revision = argparse.Namespace(
            index=2,
            previous_speaker="S1",
            assigned_speaker="S2",
        )

        with mock.patch(
            "window.window_diarizer.find_speaker_prototype_revisions",
            return_value=[unknown_revision, known_revision],
        ):
            diarizer._refine_speaker_assignments()

        diarizer._apply_prototype_revision.assert_called_once_with(known_revision)

    def test_prototype_unknown_revision_is_tentative_not_committed(self) -> None:
        from window.youtube_window_diarize_gui import build_window_validation_records

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.bus = RecordingEventBus()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._sentence_refinement_lock = threading.Lock()
        diarizer._sentence_refinement_records = {
            1: {
                "index": 1,
                "base_payload": self.base_sentence_payload(),
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "duration_seconds": 3.0,
                "assigned_speaker": None,
                "created_speaker": False,
                "probabilities": {"unknown": 1.0},
                "similarities": {},
                "unknown_probability": 1.0,
                "top_similarity": None,
                "margin": None,
                "quality": 1.0,
                "assignment_source": "embedding",
            }
        }
        revision = argparse.Namespace(
            index=1,
            previous_speaker=None,
            assigned_speaker="S3",
            prototype_score=0.62,
            prototype_margin=0.21,
            prototype_delta=1.62,
            prototype_scores={"S3": 0.62, "S1": 0.41},
            assignment_source="prototype_unknown_assign",
        )

        self.assertTrue(diarizer._apply_prototype_revision(revision))

        committed = diarizer._sentence_refinement_records[1]
        self.assertIsNone(committed["assigned_speaker"])
        self.assertEqual(committed["provisional_assigned_speaker"], "S3")

        tentative_payload = diarizer.bus.records[-1]["payload"]
        self.assertEqual(tentative_payload["assigned_speaker"], "S3")
        self.assertTrue(tentative_payload["provisional_assignment"])

        records = [
            {"time": 1.0, "event": "sentence", "payload": self.unknown_sentence_payload()},
            *diarizer.bus.records,
        ]
        _analysis_records, final_payloads = build_window_validation_records(records)

        self.assertEqual(len(final_payloads), 1)
        self.assertIsNone(final_payloads[0]["assigned_speaker"])

    def test_score_reducers_follow_committed_live_state_not_tentative_ui_state(self) -> None:
        from realtime.realtime_speakerdiarize import analyze_trace_against_canonical
        from window.youtube_window_diarize_gui import build_window_validation_records

        tentative_payload = {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "provisional_assignment": True,
            "revision_from": "UNKNOWN",
            "revision_to": "S3",
            "assigned_speaker": "S3",
            "probabilities": {"unknown": 0.45, "speaker3": 0.55},
            "assignment_source": "prototype_unknown_tentative",
        }
        live_records = [
            {"time": 1.0, "event": "sentence", "payload": self.unknown_sentence_payload()},
            {"time": 2.0, "event": "sentence", "payload": tentative_payload},
        ]

        analysis_records, final_payloads = build_window_validation_records(live_records)
        summary = analyze_trace_against_canonical(analysis_records, self.canonical(), match_mode="timestamp")

        self.assertIsNone(final_payloads[0]["assigned_speaker"])
        self.assertIsNone(summary["rows"][0]["assigned_speaker"])
        self.assertEqual(summary["unknown_segments"], 1)
        self.assertEqual(summary["assigned_counts"], {"UNKNOWN": 1})

        confirmed_records = [
            *live_records,
            {"time": 3.0, "event": "sentence", "payload": self.confirmed_sentence_payload()},
        ]
        analysis_records, final_payloads = build_window_validation_records(confirmed_records)
        summary = analyze_trace_against_canonical(analysis_records, self.canonical(), match_mode="timestamp")

        self.assertEqual(final_payloads[0]["assigned_speaker"], "S6")
        self.assertEqual(summary["rows"][0]["assigned_speaker"], "S6")
        self.assertEqual(summary["assigned_counts"], {"S6": 1})
        self.assertEqual(summary["duration_accuracy"], 1.0)

    def test_raw_trace_analysis_ignores_tentative_sentence_events(self) -> None:
        from realtime.realtime_speakerdiarize import analyze_trace_against_canonical

        final_payload = {
            **self.unknown_sentence_payload(),
            "video_start_seconds": 0.0,
            "video_end_seconds": 3.0,
            "duration_seconds": 3.0,
        }
        tentative_payload = {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "provisional_assignment": True,
            "assigned_speaker": "S3",
        }
        raw_records = [
            {"time": 1.0, "event": "final", "payload": final_payload},
            {"time": 1.1, "event": "sentence", "payload": self.unknown_sentence_payload()},
            {"time": 1.2, "event": "sentence", "payload": tentative_payload},
        ]

        summary = analyze_trace_against_canonical(raw_records, self.canonical(), match_mode="timestamp")

        self.assertIsNone(summary["rows"][0]["assigned_speaker"])
        self.assertEqual(summary["unknown_segments"], 1)

        summary = analyze_trace_against_canonical(
            [*raw_records, {"time": 1.3, "event": "sentence", "payload": self.confirmed_sentence_payload()}],
            self.canonical(),
            match_mode="timestamp",
        )

        self.assertEqual(summary["rows"][0]["assigned_speaker"], "S6")
        self.assertEqual(summary["duration_accuracy"], 1.0)


class LiveSpeakerProbeScoringTests(unittest.TestCase):
    def test_live_speaker_probe_score_uses_sidebar_counted_slices(self) -> None:
        records = [
            {
                "time": 100.0,
                "event": "validation_replay_start",
                "payload": {"replay_speed": 1.0},
            },
            {
                "time": 101.7,
                "event": "live_speaker",
                "payload": {"speaker_id": "S1", "start": 0.5, "end": 1.5},
            },
            {
                "time": 104.6,
                "event": "live_speaker",
                "payload": {"speaker_id": "S2", "start": 3.5, "end": 4.5},
            },
        ]
        canonical = [
            {"speaker": "A", "start": 0.0, "end": 2.0, "text": "a"},
            {"speaker": "B", "start": 3.0, "end": 5.0, "text": "b"},
        ]

        score = score_live_speaker_probe(records, canonical)

        self.assertEqual(score["raw_live_speaker_event_count"], 2)
        self.assertEqual(score["sidebar_counted_live_seconds"], 2.0)
        self.assertEqual(score["any_live_speech_coverage"], 0.5)
        self.assertEqual(score["correct_live_speaker_coverage"], 0.5)
        self.assertEqual(score["profile_map"], {"S1": "A", "S2": "B"})
        self.assertEqual(score["lag_after_window_end_seconds"]["median"], 0.15)

    def test_live_speaker_probe_clear_events_are_speaker_specific(self) -> None:
        records = [
            {
                "time": 10.0,
                "event": "validation_replay_start",
                "payload": {"replay_speed": 1.0},
            },
            {
                "time": 11.0,
                "event": "live_speaker",
                "payload": {"speaker_id": "S1", "start": 1.0, "end": 2.0, "hold_seconds": 4.0},
            },
            {
                "time": 12.0,
                "event": "live_speaker_clear",
                "payload": {"speaker_id": "S2", "reason": "stale"},
            },
            {
                "time": 13.0,
                "event": "live_speaker_clear",
                "payload": {"speaker_id": "S1", "reason": "silence"},
            },
        ]
        canonical = [{"speaker": "A", "start": 1.0, "end": 3.0, "text": "a"}]

        score = score_live_speaker_probe(records, canonical)

        self.assertEqual(
            score["active_live_slices"],
            [{"speaker": "S1", "start": 1.0, "end": 3.0, "duration_seconds": 2.0}],
        )
        self.assertEqual(score["active_correct_live_speaker_coverage"], 1.0)

    def test_live_speaker_probe_unknown_clear_debounce_extends_visible_active_speaker(self) -> None:
        records = [
            {
                "time": 10.0,
                "event": "validation_replay_start",
                "payload": {"replay_speed": 1.0},
            },
            {
                "time": 11.0,
                "event": "live_speaker",
                "payload": {"speaker_id": "S1", "start": 1.0, "end": 2.0, "hold_seconds": 1.0},
            },
            {
                "time": 11.8,
                "event": "live_speaker_clear",
                "payload": {"speaker_id": "S1", "reason": "unknown"},
            },
        ]
        canonical = [{"speaker": "A", "start": 1.0, "end": 2.3, "text": "a"}]

        score = score_live_speaker_probe(records, canonical, unknown_clear_debounce_seconds=0.5)

        self.assertEqual(score["unknown_clear_debounce_seconds"], 0.5)
        self.assertEqual(
            score["active_live_slices"],
            [{"speaker": "S1", "start": 1.0, "end": 2.3, "duration_seconds": 1.3}],
        )

    def test_live_speaker_probe_latency_score_uses_visible_active_speaker(self) -> None:
        records = [
            {
                "time": 100.0,
                "event": "validation_replay_start",
                "payload": {"replay_speed": 1.0},
            },
            {
                "time": 100.8,
                "event": "live_speaker",
                "payload": {"speaker_id": "S1", "start": 0.0, "end": 0.5, "hold_seconds": 1.0},
            },
            {
                "time": 103.4,
                "event": "live_speaker",
                "payload": {"speaker_id": "S2", "start": 3.0, "end": 3.5, "hold_seconds": 1.0},
            },
        ]
        canonical = [
            {"speaker": "A", "start": 0.0, "end": 3.0, "text": "a"},
            {"speaker": "B", "start": 3.0, "end": 6.0, "text": "b"},
        ]

        score = score_live_speaker_probe(records, canonical)
        latency = score["live_turn_latency"]

        self.assertEqual(latency["turn_count"], 2)
        self.assertEqual(latency["speaker_change_turn_count"], 1)
        self.assertEqual(latency["missed_turn_count"], 0)
        self.assertEqual(latency["first_correct_latency_seconds"]["median"], 0.6)
        self.assertEqual(latency["speaker_change_latency_seconds"]["median"], 0.4)
        self.assertEqual(latency["first_correct_latency_score"], 0.95)
        self.assertEqual(latency["speaker_change_latency_score"], 1.0)
        self.assertIn("latency_weighted_live_speaker_score", score)


class BrowserLiveSpeakerScoringTests(unittest.TestCase):
    def test_browser_live_speaker_score_uses_dom_observed_state(self) -> None:
        samples = [
            {"playback_time": 0.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 1.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 2.0, "dom_live_speaker_ids": []},
            {"playback_time": 3.0, "dom_live_speaker_ids": ["S2"]},
            {"playback_time": 4.0, "dom_live_speaker_ids": ["S2"]},
        ]
        canonical = [
            {"speaker": "A", "start": 0.0, "end": 2.0, "text": "a"},
            {"speaker": "B", "start": 3.0, "end": 4.0, "text": "b"},
        ]

        score = score_browser_live_speaker_samples(samples, canonical, max_sample_gap_seconds=1.0)

        self.assertEqual(score["speaker_map"], {"S1": "A", "S2": "B"})
        self.assertEqual(score["correct_live_seconds"], 3.0)
        self.assertEqual(score["missing_live_speech_seconds"], 0.0)
        self.assertEqual(score["wrong_live_speech_seconds"], 0.0)
        self.assertEqual(score["strict_browser_live_score"], 1.0)

    def test_browser_live_speaker_score_penalizes_wrong_duplicate_speaker_id(self) -> None:
        samples = [
            {"playback_time": 0.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 1.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 2.0, "dom_live_speaker_ids": ["S2"]},
            {"playback_time": 3.0, "dom_live_speaker_ids": ["S2"]},
        ]
        canonical = [{"speaker": "A", "start": 0.0, "end": 3.0, "text": "a"}]

        score = score_browser_live_speaker_samples(samples, canonical, max_sample_gap_seconds=1.0)

        self.assertEqual(score["speaker_map"], {"S1": "A"})
        self.assertEqual(score["correct_live_seconds"], 2.0)
        self.assertEqual(score["wrong_live_speech_seconds"], 1.0)
        self.assertLess(score["strict_browser_live_score"], score["correct_live_speaker_coverage"])

    def test_browser_live_speaker_score_reports_correct_interruption_flicker(self) -> None:
        samples = [
            {"playback_time": 0.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 1.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 1.4, "dom_live_speaker_ids": []},
            {"playback_time": 2.0, "dom_live_speaker_ids": ["S1"]},
            {"playback_time": 3.0, "dom_live_speaker_ids": ["S1"]},
        ]
        canonical = [{"speaker": "A", "start": 0.0, "end": 3.0, "text": "a"}]

        score = score_browser_live_speaker_samples(
            samples,
            canonical,
            max_sample_gap_seconds=1.0,
            flicker_gap_seconds=0.25,
        )

        self.assertEqual(score["flicker"]["correct_interruption_count"], 1)
        self.assertAlmostEqual(score["flicker"]["correct_interruption_seconds"], 0.6)
        self.assertGreater(score["missing_live_speech_seconds"], 0.0)


class WindowHtmlSafetyTests(unittest.TestCase):
    def test_speaker_label_is_inserted_as_text_not_markup(self) -> None:
        self.assertNotIn("${speakerLabel}</span>", HTML)
        self.assertIn("speakerBadge.textContent = speakerLabel;", HTML)
        self.assertIn("row.replaceChildren(top, text);", HTML)

    def test_revised_sentence_refreshes_speaker_counts(self) -> None:
        self.assertIn("let renderedSpeakerSentenceCounts = {};", HTML)
        self.assertIn("let renderedSpeakerSpeakingSeconds = {};", HTML)
        self.assertIn('let currentLiveSpeakerId = "";', HTML)
        self.assertIn('let transcriptLiveSpeakerId = "";', HTML)
        self.assertIn('let fallbackLiveSpeakerId = "";', HTML)
        self.assertIn("let liveSpeakerTimeline = [];", HTML)
        self.assertIn("let fastSpeakerPanelStats = {};", HTML)
        self.assertIn("let hasRenderedFinalSentenceRows = false;", HTML)
        self.assertIn("function dominantRealtimeSpeakerId(start, end)", HTML)
        self.assertIn("function realtimeDominanceScoredEnd(start, end)", HTML)
        self.assertIn("const tailSeconds = Math.min(3, Math.max(2, duration * 0.25));", HTML)
        self.assertIn("const requiredSeconds = Math.max(0.3, scoredSeconds * 0.5);", HTML)
        self.assertIn("function rememberLiveSpeakerEvidence(speakerId, item)", HTML)
        self.assertIn("function realtimeRowHasSpeakerEvidence(start, end)", HTML)
        self.assertIn("liveSpeakerTimeline.push({speakerId: normalizedSpeakerId, start, end});", HTML)
        self.assertIn("liveSpeakerTimeline = [];", HTML)
        self.assertIn("const previousDisplaySpeakerId = item.realtime ? normalizedLiveSpeakerId(row.dataset.speaker) : \"\";", HTML)
        self.assertIn("realtimeRowDisplaySpeakerId(rawSpeakerId, startSeconds, endSeconds, previousDisplaySpeakerId)", HTML)
        self.assertIn('row.dataset.rawSpeaker = item.realtime ? (visualSplit ? displaySpeakerId : rawSpeakerId) : "";', HTML)
        self.assertIn('row.dataset.speaker = displaySpeakerId || "UNKNOWN";', HTML)
        self.assertIn('row.classList.toggle("live-speaker-row", item.realtime && Boolean(displaySpeakerId));', HTML)
        self.assertIn('row.style.setProperty("--live-row-color", color || "#8F9BA8");', HTML)
        self.assertIn("function updateCurrentLiveSpeakerFromRealtimeRows()", HTML)
        self.assertIn("transcriptLiveSpeakerId = realtimeRowTranscriptLiveSpeakerId(activeRow);", HTML)
        self.assertIn("function reconcileLiveSpeakerHighlight()", HTML)
        self.assertIn("updateCurrentLiveSpeakerFromRealtimeRows();", HTML)
        self.assertIn("speakingSeconds[speakerId] = (speakingSeconds[speakerId] || 0) + Math.max(0, end - start);", HTML)
        self.assertIn("renderedSpeakerSpeakingSeconds = speakingSeconds;", HTML)
        self.assertIn("function applyFastSpeakerPanelSignal(item)", HTML)
        self.assertIn('applyFastSpeakerPanelSignal(item);', HTML)
        self.assertIn("if (item.only_if_no_live_speaker && currentLiveSpeakerId) return;", HTML)
        self.assertIn("refreshSpeakerPanelSentenceCounts();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = displaySpeakerId || "UNKNOWN";'),
            HTML.index("refreshSpeakerPanelSentenceCounts();", HTML.index("function renderSentence(item)")),
        )

    def test_realtime_unknown_sentence_ignores_tail_only_known_speaker(self) -> None:
        display_start = HTML.index("function realtimeRowDisplaySpeakerId")
        display_end = HTML.index("function applyRealtimeRowSpeaker")
        display_block = HTML[display_start:display_end]

        self.assertIn("const dominantSpeakerId = dominantRealtimeSpeakerId(start, end);", display_block)
        self.assertIn('if (realtimeRowHasSpeakerEvidence(start, end)) return "";', display_block)
        self.assertIn("if (rowEnd - rowStart > 3) return \"\";", display_block)
        self.assertNotIn("activeFallbackLiveSpeakerId()", display_block)
        self.assertIn("const requiredSeconds = Math.max(0.3, scoredSeconds * 0.5);", HTML)

    def test_realtime_visual_split_uses_tail_speaker_and_punctuation_only(self) -> None:
        self.assertIn(".row.provisional-visual-split", HTML)
        self.assertIn("function realtimeTailSpeakerChange(start, end, currentSpeakerId = \"\")", HTML)
        self.assertIn("function lastPunctuationTextSplit(textValue)", HTML)
        self.assertIn("function provisionalRealtimeVisualSplit(item, displaySpeakerId, start, end)", HTML)
        self.assertIn("const tailChange = realtimeTailSpeakerChange(start, end, displaySpeakerId);", HTML)
        self.assertIn("const textSplit = lastPunctuationTextSplit(item.text);", HTML)
        self.assertIn('if (tailSeconds < 0.4) return;', HTML)
        self.assertIn("const boundaryPattern = /[.!?][\"')\\]]*\\s+/g;", HTML)
        self.assertIn("renderProvisionalRealtimeSplitRow(row, visualSplit);", HTML)
        self.assertIn("clearProvisionalRealtimeSplitsFor(item.index);", HTML)
        self.assertIn("restoreRealtimeRowFullPreview(row);", HTML)
        self.assertIn("applyProvisionalRealtimeVisualSplit(row, visualSplit);", HTML)
        self.assertIn("row.dataset.fullRawSpeaker = item.realtime ? rawSpeakerId : \"\";", HTML)
        self.assertIn("row.dataset.fullEnd = item.realtime ? String(endSeconds) : \"\";", HTML)
        self.assertIn("row.dataset.fullText = item.realtime ? (item.text || \"\") : \"\";", HTML)

    def test_realtime_clear_settles_rows_for_smooth_final_adoption(self) -> None:
        self.assertIn(".row.realtime-settling", HTML)
        self.assertIn(".row.row-removing", HTML)
        self.assertIn("const realtimeSettleRemovalDelayMs = 1400;", HTML)
        self.assertIn("function markRealtimeRowSettling(row, generation)", HTML)
        self.assertIn("function findAdoptableRealtimeRow(item, options = {})", HTML)
        self.assertIn("function removeOverlappingSettlingRealtimeRows(item, keepRow = null)", HTML)
        self.assertIn("function placeSentenceRowChronologically(row)", HTML)
        self.assertIn("function clearSettlingRealtimeState(row)", HTML)
        self.assertIn("markRealtimeRowSettling(row, generation)", HTML)
        self.assertNotIn("forEach(row => row.remove())", HTML)
        self.assertIn("row = findAdoptableRealtimeRow(item, {settlingOnly: true});", HTML)
        self.assertIn("row = findAdoptableRealtimeRow(item);", HTML)
        self.assertIn("clearSettlingRealtimeState(row);", HTML)
        self.assertIn("removeOverlappingSettlingRealtimeRows(item, row);", HTML)
        self.assertIn("if (settlingOnly && row.dataset.realtimeSettling !== \"true\") return;", HTML)
        self.assertIn("if (timeScore >= 0.34 && textScore >= 0.5) {", HTML)
        self.assertIn('.filter(row => row.dataset.realtimeSettling !== "true")', HTML)
        self.assertNotIn("clearAllProvisionalRealtimeSplits", HTML)

    def test_reused_sentence_rows_are_reinserted_chronologically(self) -> None:
        self.assertIn("function rowShouldSortBefore(a, b)", HTML)
        self.assertIn("function rowChronologyKey(row)", HTML)
        self.assertIn("sentences.insertBefore(row, next);", HTML)
        self.assertIn("sentences.appendChild(row);", HTML)

        render_start = HTML.index("function renderSentence(item)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        place_index = render_block.index("placeSentenceRowChronologically(row);")
        split_index = render_block.index("if (visualSplit) {", place_index)
        self.assertLess(render_block.index("row.replaceChildren(top, text);"), place_index)
        self.assertLess(place_index, split_index)

    def test_late_realtime_update_cannot_overwrite_final_sentence_row(self) -> None:
        self.assertIn("function findFinalSentenceRow(index)", HTML)
        self.assertIn("function findRealtimeSentenceRow(index)", HTML)
        self.assertIn('row.dataset.index === key && row.dataset.realtime !== "true"', HTML)
        self.assertIn('row.dataset.index === key && row.dataset.realtime === "true"', HTML)

        render_start = HTML.index("function renderSentence(item)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        guard = 'if (item.realtime && findFinalSentenceRow(item.index)) {'
        self.assertIn(guard, render_block)
        self.assertLess(render_block.index(guard), render_block.index("clearProvisionalRealtimeSplitsFor(item.index);"))
        self.assertIn("let row = item.realtime", render_block)
        self.assertIn("? findRealtimeSentenceRow(item.index)", render_block)
        self.assertIn(": (findFinalSentenceRow(item.index) || findRealtimeSentenceRow(item.index));", render_block)

    def test_speaker_solo_mute_filters_transcript_rows(self) -> None:
        self.assertIn("let soloSpeakerIds = new Set();", HTML)
        self.assertIn("let mutedSpeakerIds = new Set();", HTML)
        self.assertIn("function speakerTranscriptVisible(speakerId)", HTML)
        self.assertIn("if (mutedSpeakerIds.has(speakerId)) return false;", HTML)
        self.assertIn("if (soloSpeakerIds.size > 0) return soloSpeakerIds.has(speakerId);", HTML)
        self.assertIn("row.hidden = !speakerTranscriptVisible(row.dataset.speaker) || !transcriptSearchVisible(row);", HTML)
        self.assertIn("function setSpeakerFilter(speakerId, mode, active)", HTML)
        self.assertIn("function pruneSpeakerFilterState()", HTML)
        self.assertIn("refreshTranscriptVisibility();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = displaySpeakerId || "UNKNOWN";'),
            HTML.index("refreshTranscriptVisibility();", HTML.index("function renderSentence(item)")),
        )

    def test_live_transcript_header_matches_draft_contract(self) -> None:
        self.assertIn('class="transcript-header"', HTML)
        self.assertIn("Live transcript", HTML)
        self.assertIn('id="followLive" type="checkbox" checked', HTML)
        self.assertIn("let followLiveEnabled = true;", HTML)
        self.assertIn("if (!followLiveEnabled) return;", HTML)
        self.assertIn('id="transcriptSearch" type="search" placeholder="Search transcript"', HTML)
        self.assertIn("let transcriptSearchText = \"\";", HTML)
        self.assertIn("function transcriptSearchVisible(row)", HTML)
        self.assertIn("query.split(/\\s+/).every(term => searchable.includes(term));", HTML)
        self.assertIn('id="clearTranscript" class="transcript-icon-button"', HTML)
        self.assertIn('const clearTranscriptButton = document.getElementById("clearTranscript");', HTML)
        self.assertIn("let transcriptClearBeforeSeconds = 0;", HTML)
        self.assertIn("function clearDisplayedTranscript()", HTML)
        self.assertIn("function itemIsBeforeClearedTranscriptBoundary(item)", HTML)
        self.assertIn("clearTranscriptButton.addEventListener(\"click\", clearDisplayedTranscript);", HTML)
        self.assertIn("if (itemIsBeforeClearedTranscriptBoundary(item)) {", HTML)
        self.assertIn('id="copyTranscript" class="transcript-icon-button"', HTML)
        self.assertIn('id="downloadTranscript" class="transcript-icon-button"', HTML)
        self.assertIn("function transcriptExportText(speakerId = null)", HTML)
        self.assertIn("`[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`", HTML)
        self.assertIn("function copyTextToClipboard(text)", HTML)
        self.assertIn("function downloadTranscript(speakerId = null)", HTML)
        self.assertIn('id="transcriptSettings"', HTML)
        self.assertIn('id="transcriptSettingsPanel" class="transcript-settings-panel" hidden', HTML)
        self.assertIn('id="showTranscriptTags" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptTime" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptSpeechRate" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptProbabilities" type="checkbox" checked', HTML)
        self.assertIn(".transcript-panel.hide-tags .badge.new, .transcript-panel.hide-tags .badge.state { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-time .sentence-duration, .transcript-panel.hide-time .sentence-range { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-speech-rate .sentence-speech-rate { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-probabilities .prob { display:none; }", HTML)
        self.assertNotIn("Show low confidence", HTML)
        self.assertNotIn(">Filter<", HTML)

    def test_playback_clock_ignores_early_media_end_jumps(self) -> None:
        self.assertIn("playbackClockStartedAt", HTML)
        self.assertIn("playbackClockSlackSeconds", HTML)
        self.assertIn("Ignoring early audio ended event", HTML)

    def test_live_header_matches_draft_contract(self) -> None:
        self.assertIn("WhoSpeaks Live", HTML)
        self.assertIn("#17B7FE", HTML)
        self.assertIn("#3DC77C", HTML)
        self.assertIn("#BA79EF", HTML)
        self.assertIn("Stop transcription", HTML)
        self.assertIn("background:#981D20", HTML)
        self.assertIn("border-color:#DF3C36", HTML)
        self.assertIn('id="speakerCountNumber" class="speaker-count-number"', HTML)
        self.assertIn('id="speakerCountLabel" class="speaker-count-label"', HTML)
        self.assertIn(".speaker-count-number { position:relative; top:2px; font-size:16px; font-weight:600; line-height:1; color:#FF9F1C;", HTML)
        self.assertIn(".speaker-count-label { font-size:13px; font-weight:400;", HTML)
        self.assertIn(".speaker-summary { flex:0 0 auto; min-height:23px; display:flex; align-items:center; gap:4px;", HTML)
        self.assertIn("#speakerCount { display:inline-flex; align-items:baseline; gap:7px;", HTML)
        self.assertIn("speakerCountNumber.textContent", HTML)
        self.assertIn("speakerCountLabel.textContent", HTML)
        self.assertIn(".live-summary { min-width:0; margin-left:auto;", HTML)
        self.assertIn(".live-summary { width:100%; justify-content:flex-end; }", HTML)
        header_start = HTML.index('<header class="topbar">')
        header_end = HTML.index("</header>", header_start)
        header = HTML[header_start:header_end]
        self.assertEqual(header.count("topbar-divider"), 2)
        self.assertLess(header.index('class="brand"'), header.index('class="live-summary"'))
        status_speaker_divider = header.index("topbar-divider")
        transport_divider = header.index("topbar-divider", status_speaker_divider + 1)
        self.assertLess(header.index('id="state"'), status_speaker_divider)
        self.assertLess(status_speaker_divider, header.index('id="speakerCount"'))
        self.assertLess(header.index('id="speakerCountNumber"'), header.index('id="speakerCountLabel"'))
        self.assertLess(header.index('id="speakerCount"'), transport_divider)
        self.assertLess(transport_divider, header.index('class="transport"'))

    def test_media_area_matches_draft_contract(self) -> None:
        self.assertIn("#0B1015", HTML)
        self.assertIn("#0F161F", HTML)
        self.assertIn("--bg:#0B1015;", HTML)
        self.assertIn("--panel:#0F161F;", HTML)
        self.assertIn("--panel-2:#0F161F;", HTML)
        self.assertIn("--field:#0B1015;", HTML)
        self.assertIn("--line:#1B2B38;", HTML)
        self.assertIn("font:14px/1.35 Arial", HTML)
        self.assertIn(".topbar { min-height:52px;", HTML)
        topbar_css = HTML[HTML.index(".topbar {"):HTML.index("}", HTML.index(".topbar {"))]
        self.assertNotIn("border-bottom", topbar_css)
        self.assertNotIn("inset 0 -1px", topbar_css)
        control_panel_css = HTML[HTML.index(".control-panel {"):HTML.index("}", HTML.index(".control-panel {"))]
        self.assertNotIn("border-left", control_panel_css)
        self.assertIn(".source-strip { min-height:58px;", HTML)
        self.assertIn(".playback-panel { min-height:132px;", HTML)
        self.assertIn("grid-template-columns:minmax(150px, 240px)", HTML)
        self.assertIn(".timeline-bar { position:relative; height:6px; margin-left:8px; margin-right:10px;", HTML)
        self.assertIn(".source-grid { width:100%;", HTML)
        self.assertIn("border:0; border-radius:0; background:transparent;", HTML)
        self.assertIn(".source-row { display:contents; }", HTML)
        self.assertIn("--text:#F1F5F8;", HTML)
        self.assertIn(".dropdown-control { position:relative; min-height:34px; display:flex; align-items:center; border:1px solid var(--line); border-radius:7px; background:#0F161F; color:var(--text);", HTML)
        self.assertIn(".dropdown-control::after { content:\"\"; position:absolute; right:15px; top:50%; width:8px; height:8px; border-right:1.5px solid currentColor; border-bottom:1.5px solid currentColor;", HTML)
        self.assertIn(".select-control select { width:100%; min-width:0; min-height:32px; border:0; border-radius:7px; padding:0 36px 0 12px; background:#0F161F; color:var(--text); color-scheme:dark;", HTML)
        self.assertIn(".select-control select option, .mode option, .speaker-panel select option { background:#0B1015; color:var(--text); }", HTML)
        self.assertIn(".select-control select option:checked, .mode option:checked, .speaker-panel select option:checked { background:#0F161F; color:#FFFFFF; }", HTML)
        self.assertIn('class="source-mode-button dropdown-control"', HTML)
        self.assertIn('class="select-control dropdown-control"><select id="preset"', HTML)
        self.assertNotIn("background-image:linear-gradient", HTML)
        self.assertIn(".media-controls { min-width:0; min-height:100%; display:grid; grid-template-rows:auto minmax(0,1fr) auto;", HTML)
        self.assertIn(".media-expand { width:40px; height:40px; align-self:end; justify-self:start;", HTML)
        self.assertIn('id="mediaCard" class="media-card mode-youtube"', HTML)
        self.assertIn('class="source-strip"', HTML)
        self.assertIn("Change source", HTML)
        self.assertIn('id="sourceModeOptions"', HTML)
        self.assertIn('data-input-mode="youtube"', HTML)
        self.assertIn('data-input-mode="microphone"', HTML)
        self.assertIn('data-input-mode="system"', HTML)
        self.assertIn('id="youtubeSourceControls"', HTML)
        self.assertIn('id="timelineFill"', HTML)
        self.assertIn('id="timelineThumb"', HTML)
        self.assertIn('id="capturePanel"', HTML)
        self.assertIn('id="captureLevelFill"', HTML)
        self.assertIn('id="micGain"', HTML)
        self.assertIn("function updateMediaMode()", HTML)
        self.assertIn("function updateMediaTimeline()", HTML)
        self.assertIn("function setCaptureLevel(value)", HTML)
        self.assertIn("function setSourceModeMenuOpen(open)", HTML)
        self.assertNotIn("source-panel", HTML)
        self.assertNotIn("source-menu", HTML)
        self.assertNotIn("font-weight:700", HTML)
        self.assertNotIn("font-weight:800", HTML)
        self.assertIn("strong, b, h1, h2, h3, h4, h5, h6, summary { font-weight:400; }", HTML)
        self.assertIn(".speaker-name, .speaker-row-title { font-weight:600; }", HTML)
        old_surface_colors = [
            "#090b0d",
            "#151715",
            "#101210",
            "#080a09",
            "#343a36",
            "#080d12",
            "#0d0f0d",
            "#20241f",
            "#123e2d",
            "#102231",
            "#122231",
            "#111923",
            "#1B2732",
            "#0d131a",
            "#59675d",
            "#2f8f68",
            "#65b891",
            "#9ea89f",
        ]
        for color in old_surface_colors:
            self.assertNotIn(color, HTML)
        oversized_layout_tokens = [
            "min-height:68px",
            "min-height:88px",
            "min-height:200px",
            "font-size:20px",
            "grid-template-columns:minmax(220px, 360px)",
            "padding:16px 18px",
        ]
        for token in oversized_layout_tokens:
            self.assertNotIn(token, HTML)

        media_start = HTML.index('<section id="mediaCard"')
        transcript_start = HTML.index('<section class="transcript-panel"', media_start)
        media = HTML[media_start:transcript_start]
        self.assertIn('id="inputMode"', media)
        self.assertIn('id="preset"', media)
        self.assertIn('id="source"', media)
        self.assertIn('id="load"', media)
        self.assertLess(media.index('id="sourceKind"'), media.index('id="inputMode"'))
        self.assertLess(media.index('class="video-frame"'), media.index('id="youtubeSourceControls"'))
        self.assertLess(media.index('id="youtubeSourceControls"'), media.index('class="timeline-row"'))
        self.assertLess(media.index('class="timeline-row"'), media.index('id="expandMedia"'))
        self.assertNotIn("media-subtle-line", media)

        video_start = HTML.index('<video id="video"')
        video_end = HTML.index("</video>", video_start)
        self.assertNotIn("controls", HTML[video_start:video_end])
        audio_start = HTML.index('<audio id="audio"')
        audio_end = HTML.index("</audio>", audio_start)
        self.assertNotIn("controls", HTML[audio_start:audio_end])

    def test_speaker_panel_matches_draft_contract(self) -> None:
        self.assertIn('class="control-card speaker-panel"', HTML)
        self.assertIn('class="speaker-tabs"', HTML)
        self.assertIn('data-speaker-tab="speakers"', HTML)
        self.assertIn('data-speaker-tab="settings"', HTML)
        self.assertIn('id="speakerPanelTitle" class="speaker-panel-title">Detected speakers (0)</h2>', HTML)
        self.assertIn('id="addReferenceSpeaker"', HTML)
        self.assertIn('id="clearSpeakers"', HTML)
        self.assertIn('Clear speakers</button>', HTML)
        self.assertIn('const clearSpeakersButton = document.getElementById("clearSpeakers");', HTML)
        self.assertIn('const result = await post("/api/speakers/clear", {});', HTML)
        self.assertIn("resetTranscriptDisplay();", HTML)
        self.assertNotIn('id="speakerGroupCurrent"', HTML)
        self.assertNotIn("Current:", HTML)
        self.assertIn('class="speaker-file-actions"', HTML)
        self.assertIn(".speaker-file-actions button { min-height:28px; width:auto; padding:0 10px; font-size:12px; }", HTML)
        self.assertIn('id="loadSpeakerGroup" type="button">Load file</button>', HTML)
        self.assertIn('id="saveSpeakerGroup" type="button">Save file</button>', HTML)
        self.assertIn('id="speakerGroupFile" type="file"', HTML)
        self.assertNotIn('id="speakerGroupName"', HTML)
        self.assertNotIn('id="speakerGroupSelect"', HTML)
        self.assertIn('id="manualSpeakerComposer" class="manual-speaker-composer" hidden', HTML)
        self.assertIn('id="manualSpeakerName"', HTML)
        self.assertIn('id="manualSpeakerReferenceDock"', HTML)
        self.assertIn(".speaker-tab.active { color:#E8EEF5; box-shadow:inset 0 -2px 0 #17B7FE;", HTML)
        self.assertIn('class="sensitivity-title">New speaker</span>', HTML)
        self.assertIn('class="sensitivity-row"', HTML)
        self.assertIn(".sensitivity-title { color:var(--text); font-size:13px; line-height:1.25; }", HTML)
        self.assertIn(".sensitivity-row { display:flex; align-items:center; gap:15px;", HTML)
        self.assertIn(".sensitivity input { flex:0 1 50%; max-width:50%; min-width:120px;", HTML)
        self.assertIn(".manual-speaker-composer { display:grid; gap:8px;", HTML)
        self.assertIn(".speaker-item { --speaker-color:transparent;", HTML)
        self.assertIn(".speaker-item.live-speaker { background:color-mix(in srgb, var(--speaker-color) 18%, #0F161F);", HTML)
        self.assertIn(".speaker-item.live-speaker .speaker-item-summary { box-shadow:inset 4px 0 0 var(--speaker-color), inset 7px 0 14px", HTML)
        self.assertIn(".speaker-title-row { min-width:0; display:flex; align-items:center; gap:7px; }", HTML)
        self.assertIn(".speaker-live-indicator { flex:0 0 auto; display:inline-flex; align-items:center; gap:4px; padding:2px 6px;", HTML)
        self.assertIn("animation:livePulse 1s ease-in-out infinite;", HTML)
        self.assertIn("@keyframes livePulse", HTML)
        self.assertIn("@media (prefers-reduced-motion: reduce)", HTML)
        self.assertIn(".speaker-item-summary { width:100%; min-height:60px; display:grid; grid-template-columns:minmax(0,1fr) auto;", HTML)
        self.assertIn("box-shadow:inset 4px 0 0 var(--speaker-color);", HTML)
        self.assertNotIn("speaker-avatar", HTML)
        self.assertIn(".speaker-item.editing { position:relative; z-index:1; border:1px solid var(--speaker-color);", HTML)
        self.assertIn(".speaker-item:not(.editing) .speaker-row-title { color:var(--speaker-color); }", HTML)
        self.assertIn(".speaker-item-tail { align-self:stretch; display:flex; flex-direction:column; align-items:flex-end; justify-content:space-between;", HTML)
        self.assertIn(".speaker-filter-controls, .speaker-transcript-actions { display:flex; align-items:center; gap:4px; }", HTML)
        self.assertIn(".speaker-filter-toggle { min-height:20px; width:39px;", HTML)
        self.assertIn(".speaker-filter-toggle.mute.active", HTML)
        self.assertIn(".transcript-icon-button { min-height:24px; width:28px;", HTML)
        self.assertIn(".row.realtime { background:color-mix(in srgb, var(--live-row-color, #8F9BA8) 10%, #0B1015); }", HTML)
        self.assertIn(".row.realtime.live-speaker-row { background:color-mix(in srgb, var(--live-row-color, #8F9BA8) 18%, #0B1015);", HTML)
        self.assertIn("border-bottom-color:color-mix(in srgb, var(--live-row-color, #8F9BA8) 35%, var(--line));", HTML)
        self.assertNotIn("inset 4px 0 0 var(--live-row-color, #8F9BA8)", HTML)
        self.assertIn("function createSpeakerLiveIndicator()", HTML)
        self.assertIn('indicator.appendChild(document.createTextNode("Live"));', HTML)
        self.assertIn('titleRow.appendChild(createSpeakerLiveIndicator());', HTML)
        self.assertIn('indicator.remove();', HTML)
        self.assertIn("function applyFallbackLiveSpeaker(item)", HTML)
        self.assertIn("function clearFallbackLiveSpeakerFromProbe(item)", HTML)
        self.assertIn("function refreshRealtimeRowsFromLiveSpeaker()", HTML)
        self.assertIn("row.dataset.start", HTML)
        self.assertIn("row.dataset.end", HTML)
        self.assertIn("row.dataset.speaker", HTML)
        self.assertIn("rememberLiveSpeakerEvidence(speakerId, item);", HTML)
        self.assertIn('row.classList.toggle("live-speaker-row", Boolean(normalizedSpeakerId));', HTML)
        self.assertIn('fallbackLiveSpeakerExpiryTimer = setTimeout(refreshRealtimeRowsFromLiveSpeaker, remainingMs + 25);', HTML)
        self.assertIn('if (speakerId && fallbackLiveSpeakerId && speakerId !== fallbackLiveSpeakerId) return;', HTML)
        self.assertIn('es.addEventListener("live_speaker", e => applyFallbackLiveSpeaker(JSON.parse(e.data)));', HTML)
        self.assertIn('es.addEventListener("live_speaker_clear", e => clearFallbackLiveSpeakerFromProbe(JSON.parse(e.data)));', HTML)
        self.assertIn("const holdSeconds = Math.max(0, Number(item.hold_seconds || 2.0));", HTML)
        self.assertIn("fallbackLiveSpeakerUntilMs = performance.now() + holdSeconds * 1000;", HTML)
        self.assertIn("currentLiveSpeakerId = transcriptLiveSpeakerOverrideId", HTML)
        self.assertIn("|| activeFallbackLiveSpeakerId()", HTML)
        self.assertIn("|| (liveSpeakerConfig.highlight_transcript ? transcriptLiveSpeakerId : \"\");", HTML)
        self.assertIn('return "sentence";', HTML)
        self.assertNotIn("fast window", HTML)
        self.assertIn("fastSpeakerPanelStats[speakerId]", HTML)
        self.assertIn('row.classList.toggle("live-speaker", Boolean(currentLiveSpeakerId) && speaker.id === currentLiveSpeakerId);', HTML)
        self.assertNotIn("speaker-editing-badge", HTML)
        self.assertIn(".speaker-row-name-input", HTML)
        self.assertIn("Reference voice added", HTML)
        self.assertNotIn("No reference voice", HTML)
        self.assertIn("function setSpeakerTab(tabName)", HTML)
        self.assertIn("function setEditingSpeaker(speakerId, options = {})", HTML)
        self.assertIn("const collapse = requestedId && editingSpeakerId === requestedId && !options.keepOpen;", HTML)
        self.assertIn("manualSpeakerComposerOpen = false;", HTML)
        self.assertIn("function syncManualSpeakerComposer()", HTML)
        self.assertIn("manualSpeakerReferenceDock.appendChild(referenceSpeakerForm);", HTML)
        self.assertIn("manualSpeakerName.focus();", HTML)
        self.assertIn("manualSpeakerName.select();", HTML)
        self.assertIn('if (editingSpeakerId && !speakerIds.includes(editingSpeakerId))', HTML)
        self.assertNotIn('editingSpeakerId = speakerIds[0]', HTML)
        self.assertIn("pendingSpeakerNameFocusId = editingSpeakerId && options.focusName !== false ? editingSpeakerId : \"\";", HTML)
        self.assertIn("return manualSpeakerName.value.trim();", HTML)
        self.assertIn("function closeManualSpeakerComposerAfterReference()", HTML)
        self.assertIn('addReferenceSpeakerButton.addEventListener("click"', HTML)
        self.assertNotIn("window.prompt", HTML)
        self.assertIn('const name = speakerLibraryState.group_name || "speakers";', HTML)
        self.assertIn('const result = await post("/api/speakers/export", {name});', HTML)
        self.assertIn("downloadJsonFile(speakerGroupFileName(group.name || name), group);", HTML)
        self.assertIn("speakerGroupFile.click();", HTML)
        self.assertIn("const group = JSON.parse(await file.text());", HTML)
        self.assertIn('const result = await post("/api/speakers/import", {group});', HTML)
        self.assertIn("function speakerPanelName(speaker)", HTML)
        self.assertIn("function createSpeakerFilterToggle(speaker, mode)", HTML)
        self.assertIn('filterControls.appendChild(createSpeakerFilterToggle(speaker, "solo"));', HTML)
        self.assertIn('filterControls.appendChild(createSpeakerFilterToggle(speaker, "mute"));', HTML)
        self.assertIn("function createTranscriptActionButton(kind, speaker)", HTML)
        self.assertIn('transcriptActions.appendChild(createTranscriptActionButton("copy", speaker));', HTML)
        self.assertIn('transcriptActions.appendChild(createTranscriptActionButton("download", speaker));', HTML)
        self.assertIn('button.setAttribute("aria-pressed", active ? "true" : "false");', HTML)
        self.assertIn('target.closest(".speaker-row-name-input, .speaker-filter-toggle, .speaker-transcript-action")', HTML)
        self.assertIn("function recomputeRenderedSpeakerSentenceCounts()", HTML)
        self.assertIn("let speakerSessionBaselineSentenceCounts = {};", HTML)
        self.assertIn("function syncSpeakerSessionBaselines(state = speakerLibraryState)", HTML)
        self.assertIn("function hasCurrentSessionSpeakerCounts()", HTML)
        self.assertIn('if (row.dataset.realtime === "true") return;', HTML)
        self.assertIn("function speakerPanelSpeakingSeconds(speaker)", HTML)
        self.assertIn('function speakerSentenceText(count, speakingSeconds = 0, unit = "sentence")', HTML)
        self.assertIn('return `${total} ${unit}${total === 1 ? "" : "s"} · ${speakerSpeakingTimeText(speakingSeconds)}`;', HTML)
        self.assertIn("function refreshSpeakerPanelSentenceCounts()", HTML)
        self.assertIn("speakerPanelSentenceCount(speaker)", HTML)
        self.assertIn("speakerPanelSpeakingSeconds(speaker)", HTML)
        self.assertIn("speakerBaselineSentenceCount(speaker) + speakerCurrentSessionSentenceCount(speakerId)", HTML)
        self.assertIn("speakerBaselineSpeakingSeconds(speaker) + speakerCurrentSessionSpeakingSeconds(speakerId)", HTML)
        self.assertGreaterEqual(HTML.count("if (hasRenderedFinalSentenceRows) return rendered;"), 2)
        self.assertIn("return fast;", HTML)
        self.assertIn("if (!hasCurrentSessionSpeakerCounts()) {", HTML)
        self.assertIn("function clearUnsavedDetectedSpeakerDisplay()", HTML)
        self.assertIn('if (speakerLibraryState.group_name) return;', HTML)
        self.assertIn("if (result.speaker_state) updateSpeakerState(result.speaker_state);", HTML)
        self.assertIn("if (media.speaker_state) updateSpeakerState(media.speaker_state);", HTML)
        self.assertIn("async function commitSpeakerNameInput(speaker, input)", HTML)
        self.assertIn('title.value = speakerPanelName(speaker);', HTML)
        self.assertIn('title.addEventListener("blur"', HTML)
        self.assertIn("title.focus();", HTML)
        self.assertIn("title.select();", HTML)
        self.assertNotIn('id="saveSpeakerName"', HTML)
        self.assertNotIn('id="cancelSpeakerEdit"', HTML)
        self.assertNotIn('id="stopReference"', HTML)
        self.assertIn("Upload audio", HTML)
        self.assertIn("Record from mic", HTML)
        self.assertIn('recordReferenceButtonLabel.textContent = recording ? "Stop and add" : "Record from mic";', HTML)
        self.assertIn('if (referenceRecordStream || referenceRecordPending)', HTML)
        self.assertIn('recordReferenceButton.classList.toggle("recording", recording);', HTML)


class WindowStreamingAudioTests(unittest.TestCase):
    def test_portable_speaker_group_centroid_preserves_float32_payload(self) -> None:
        centroid = np.array([0.125, -0.5, 0.33333334, 1.0], dtype=np.float32)
        payload = WindowDiarizer._centroid_payload(centroid)

        self.assertEqual(payload["centroid_encoding"], "float32-base64-le")
        restored = np.asarray(WindowDiarizer._centroid_from_payload(payload), dtype=np.float32)
        np.testing.assert_array_equal(restored, centroid)

    def test_portable_speaker_group_export_import_round_trips_profiles(self) -> None:
        class FakeBus:
            def emit(self, *_args: object, **_kwargs: object) -> None:
                return None

        class FakeMemory:
            def __init__(self, profiles: list[dict[str, object]] | None = None) -> None:
                self.profiles = profiles or []

            def export_profiles(self) -> list[dict[str, object]]:
                return [dict(profile) for profile in self.profiles]

            def replace_profiles(self, profiles: list[dict[str, object]]) -> None:
                self.profiles = []
                for index, item in enumerate(profiles, 1):
                    self.profiles.append({
                        "label": f"S{index}",
                        "index": index,
                        "centroid": np.asarray(item["centroid"], dtype=np.float32),
                        "sentence_count": int(item.get("sentence_count") or 1),
                        "speech_seconds": float(item.get("speech_seconds") or 0.0),
                        "created_at": time.time(),
                        "last_seen_at": time.time(),
                        "locked": bool(item.get("locked")),
                    })

            def upsert_profile(
                self,
                label: str,
                embedding: np.ndarray,
                duration_seconds: float = 0.0,
                sentence_count: int = 1,
                locked: bool = False,
            ) -> str:
                index = int(label[1:]) if label.startswith("S") and label[1:].isdigit() else len(self.profiles) + 1
                self.profiles.append({
                    "label": label,
                    "index": index,
                    "centroid": np.asarray(embedding, dtype=np.float32),
                    "sentence_count": int(sentence_count),
                    "speech_seconds": float(duration_seconds),
                    "created_at": time.time(),
                    "last_seen_at": time.time(),
                    "locked": bool(locked),
                })
                self.profiles.sort(key=lambda profile: int(profile["index"]))
                return label

        centroid = np.array([0.125, -0.5, 0.33333334, 1.0], dtype=np.float32)
        second_centroid = np.array([0.25, 0.75, -0.125, 0.5], dtype=np.float32)
        live_centroid = np.array([0.0, 0.25, 0.75], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            source = WindowDiarizer.__new__(WindowDiarizer)
            source.args = argparse.Namespace(
                embedding_provider="mock-main",
                embedding_device="cpu",
                live_speaker_embedding_provider="mock-live",
            )
            source.speaker_library_dir = Path(tmp)
            source.memory = FakeMemory([{
                "label": "S1",
                "index": 1,
                "centroid": centroid,
                "sentence_count": 3,
                "speech_seconds": 7.5,
                "created_at": 10.0,
                "last_seen_at": 12.0,
                "locked": True,
            }, {
                "label": "S2",
                "index": 2,
                "centroid": second_centroid,
                "sentence_count": 5,
                "speech_seconds": 12.5,
                "created_at": 10.0,
                "last_seen_at": 12.0,
                "locked": False,
            }])
            source.live_memory = FakeMemory([{
                "label": "S2",
                "index": 2,
                "centroid": live_centroid,
                "sentence_count": 2,
                "speech_seconds": 6.25,
                "created_at": 10.0,
                "last_seen_at": 12.0,
                "locked": False,
            }])
            source._live_embedding_separate = True
            source._embedding_jobs = None
            source._live_memory_update_jobs = None
            source._speaker_lock = threading.Lock()
            source._unknown_lock = threading.Lock()
            source._speaker_metadata = {
                "S1": {"name": "Alice", "source": "reference", "locked": True, "reference_audio": ""},
                "S2": {"name": "Bob", "source": "detected", "locked": False, "reference_audio": ""},
            }
            source._speaker_group_name = ""
            source._seed_profiles = []
            source._seed_live_profiles = []
            source.bus = FakeBus()

            group = source.export_speaker_group_file("Local group")

            created_memories: list[FakeMemory] = []

            def new_memory() -> FakeMemory:
                memory = FakeMemory()
                created_memories.append(memory)
                return memory

            target = WindowDiarizer.__new__(WindowDiarizer)
            target.args = argparse.Namespace(
                embedding_provider="mock-main",
                embedding_device="cpu",
                live_speaker_embedding_provider="mock-live",
            )
            target.speaker_library_dir = Path(tmp)
            target.memory = FakeMemory()
            target.live_memory = FakeMemory()
            target._live_embedding_separate = True
            target._new_memory = new_memory
            target._speaker_lock = threading.Lock()
            target._unknown_lock = threading.Lock()
            target._unknown_sentences = []
            target._speaker_metadata = {}
            target._speaker_group_name = ""
            target._seed_profiles = []
            target._seed_live_profiles = []
            target._embedding_jobs = None
            target._live_memory_update_jobs = None
            target.bus = FakeBus()

            state = target.import_speaker_group_file(group)

        self.assertEqual(group["format"], "whospeaks-speaker-group")
        self.assertEqual(group["live_embedding_provider"], "mock-live")
        self.assertEqual(group["speakers"][0]["centroid_encoding"], "float32-base64-le")
        self.assertEqual(group["live_speakers"][0]["label"], "S2")
        self.assertEqual(group["live_speakers"][0]["centroid_encoding"], "float32-base64-le")
        self.assertEqual(state["group_name"], "Local_group")
        self.assertEqual(state["speakers"][0]["display_name"], "Alice")
        np.testing.assert_array_equal(target.memory.profiles[0]["centroid"], centroid)
        np.testing.assert_array_equal(target.memory.profiles[1]["centroid"], second_centroid)
        self.assertEqual(target.live_memory.profiles[0]["label"], "S2")
        np.testing.assert_array_equal(target.live_memory.profiles[0]["centroid"], live_centroid)

    def test_clear_speakers_resets_memory_metadata_and_pending_unknowns(self) -> None:
        class FakeBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, object]] = []

            def emit(self, event: str, payload: object) -> None:
                self.events.append((event, payload))

        class FakeMemory:
            def __init__(self, profiles: list[dict[str, object]] | None = None) -> None:
                self.profiles = profiles or []

            def export_profiles(self) -> list[dict[str, object]]:
                return [dict(profile) for profile in self.profiles]

        old_memory = FakeMemory([{
            "label": "S1",
            "index": 1,
            "centroid": np.array([1.0, 0.0], dtype=np.float32),
            "sentence_count": 2,
            "speech_seconds": 3.5,
            "created_at": 1.0,
            "last_seen_at": 2.0,
            "locked": False,
        }])
        new_memory = FakeMemory()
        with tempfile.TemporaryDirectory() as tmp:
            diarizer = WindowDiarizer.__new__(WindowDiarizer)
            diarizer.args = argparse.Namespace(embedding_provider="mock")
            diarizer.speaker_library_dir = Path(tmp)
            diarizer.memory = old_memory
            diarizer._new_memory = lambda: new_memory
            diarizer._speaker_lock = threading.Lock()
            diarizer._unknown_lock = threading.Lock()
            diarizer._sentence_refinement_lock = threading.Lock()
            diarizer._unknown_sentences = [object()]
            diarizer._sentence_refinement_records = {1: {"assigned_speaker": "S1"}}
            diarizer._speaker_metadata = {"S1": {"name": "Alice"}}
            diarizer._speaker_group_name = "Loaded"
            diarizer._seed_profiles = [{"centroid": [1.0, 0.0]}]
            diarizer._embedding_jobs = None
            diarizer._speaker_generation = 7
            diarizer.bus = FakeBus()

            state = diarizer.clear_speakers()

        self.assertIs(diarizer.memory, new_memory)
        self.assertEqual(diarizer._speaker_generation, 8)
        self.assertEqual(diarizer._unknown_sentences, [])
        self.assertEqual(diarizer._speaker_metadata, {})
        self.assertEqual(diarizer._seed_profiles, [])
        self.assertEqual(state["group_name"], "")
        self.assertEqual(state["speakers"], [])
        self.assertTrue(any(event == "speakers" for event, _payload in diarizer.bus.events))

    def test_initial_speaker_state_resets_idle_detected_runtime_profiles(self) -> None:
        class FakeBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, object]] = []

            def emit(self, event: str, payload: object) -> None:
                self.events.append((event, payload))

        class FakeMemory:
            def __init__(self, profiles: list[dict[str, object]] | None = None) -> None:
                self.profiles = profiles or []

            def export_profiles(self) -> list[dict[str, object]]:
                return [dict(profile) for profile in self.profiles]

            def replace_profiles(self, profiles: list[dict[str, object]]) -> None:
                self.profiles = [dict(profile) for profile in profiles]

        old_memory = FakeMemory([{
            "label": "S1",
            "index": 1,
            "centroid": np.array([1.0, 0.0], dtype=np.float32),
            "sentence_count": 4,
            "speech_seconds": 10.4,
            "created_at": 1.0,
            "last_seen_at": 2.0,
            "locked": False,
        }])
        new_memory = FakeMemory()
        with tempfile.TemporaryDirectory() as tmp:
            diarizer = WindowDiarizer.__new__(WindowDiarizer)
            diarizer.args = argparse.Namespace(embedding_provider="mock")
            diarizer.speaker_library_dir = Path(tmp)
            diarizer.memory = old_memory
            diarizer._new_memory = lambda: new_memory
            diarizer._speaker_lock = threading.Lock()
            diarizer._unknown_lock = threading.Lock()
            diarizer._sentence_refinement_lock = threading.Lock()
            diarizer._preview_lock = threading.Lock()
            diarizer._thread = None
            diarizer._preview_thread = None
            diarizer._live_probe_thread = None
            diarizer._unknown_sentences = [object()]
            diarizer._sentence_refinement_records = {1: {"assigned_speaker": "S1"}}
            diarizer._speaker_metadata = {"S1": {"name": "Stale", "source": "detected"}}
            diarizer._speaker_group_name = ""
            diarizer._seed_profiles = []
            diarizer._preview_left = 12.0
            diarizer._preview_generation = 2
            diarizer._preview_paused = True
            diarizer.bus = FakeBus()

            state = diarizer.initial_speaker_state()

        self.assertIs(diarizer.memory, new_memory)
        self.assertEqual(diarizer._unknown_sentences, [])
        self.assertEqual(diarizer._sentence_refinement_records, {})
        self.assertEqual(diarizer._speaker_metadata, {})
        self.assertEqual(state["speakers"], [])
        self.assertFalse(any(event == "speakers" for event, _payload in diarizer.bus.events))
        self.assertTrue(any(event == "realtime_clear" for event, _payload in diarizer.bus.events))

    def test_browser_stream_audio_uses_chunks_and_slices_across_boundaries(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._audio_lock = threading.Lock()
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = True
        diarizer.sample_rate = 4
        diarizer.audio = np.zeros(0, dtype=np.float32)
        diarizer._stream_audio_chunks = []
        diarizer._stream_audio_samples = 0
        diarizer.duration = 0.0

        first_duration = diarizer.append_stream_audio(np.array([0.1, 0.2], dtype=np.float32), 4)
        second_duration = diarizer.append_stream_audio(np.array([0.3, 0.4, 0.5], dtype=np.float32), 4)

        self.assertEqual(first_duration, 0.5)
        self.assertEqual(second_duration, 1.25)
        self.assertEqual(len(diarizer._stream_audio_chunks), 2)
        self.assertEqual(len(diarizer.audio), 0)

        audio, sample_rate = diarizer._audio_window_copy(0.25, 1.25)
        self.assertEqual(sample_rate, 4)
        np.testing.assert_allclose(audio, np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))

    def test_file_playback_time_rejects_impossible_jump_to_media_end(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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

    def test_asr_no_speech_filter_drops_high_no_speech_prob_words(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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

        self.assertEqual([word.text for word in kept], [" Hallo", " Ja.", " unknown"])
        self.assertTrue(any("ASR no-speech filter dropped 3 word" in item["payload"]["message"] for item in diarizer.bus.records))

    def test_asr_no_speech_filter_drops_short_segments_above_hard_threshold(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
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

        self.assertEqual([word.text for word in kept], [" Hallo"])

    def test_transcribe_window_audio_words_maps_speech_clip_times_to_media_time(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._audio_lock = threading.Lock()
        diarizer._streaming_audio = False
        diarizer.sample_rate = 10
        diarizer.audio = np.arange(100, dtype=np.float32)
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

    def test_sentence_live_speaker_hint_emits_fresh_assignment(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_sentence_hint=True,
            live_speaker_sentence_hint_max_lag_seconds=1.25,
            live_speaker_sentence_hint_new_speaker_max_lag_seconds=8.0,
            live_speaker_sentence_hint_hold_seconds=1.0,
            live_speaker_probe_hold_seconds=2.0,
        )
        diarizer.bus = Bus()
        diarizer.playback_time = lambda: 10.5

        diarizer._maybe_emit_sentence_live_speaker_hint(
            {
                "assigned_speaker": "S2",
                "start": 8.0,
                "end": 10.0,
                "probabilities": {"speaker2": 0.9},
            },
            2.0,
        )

        self.assertEqual(len(diarizer.bus.events), 1)
        event, payload = diarizer.bus.events[0]
        self.assertEqual(event, "live_speaker")
        self.assertEqual(payload["speaker_id"], "S2")
        self.assertTrue(payload["sentence_hint"])
        self.assertTrue(payload["only_if_no_live_speaker"])
        self.assertEqual(payload["assignment_source"], "final_sentence_live_hint")
        self.assertEqual(payload["live_hint_lag_seconds"], 0.5)

    def test_live_speaker_assignment_master_switch_disables_sentence_hint(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_assignment=False,
            live_speaker_sentence_hint=True,
        )
        diarizer.bus = Bus()

        diarizer._maybe_emit_sentence_live_speaker_hint({"assigned_speaker": "S2"}, 2.0)

        self.assertEqual(diarizer.bus.events, [])

    def test_live_speaker_assignment_off_reuses_main_embedding_provider(self) -> None:
        main_embedding = object()
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.embedding = main_embedding
        diarizer._new_embedding_client = mock.Mock(side_effect=AssertionError("live provider should not load"))
        args = argparse.Namespace(
            live_speaker_assignment=False,
            embedding_provider="espnet_ecapa_wavlm_joint",
            live_speaker_embedding_provider="pyannote_wespeaker_resnet34_lm",
        )

        self.assertIs(diarizer._new_live_embedding_client(args), main_embedding)

    def test_sentence_live_speaker_hint_skips_stale_assignment(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_sentence_hint=True,
            live_speaker_sentence_hint_max_lag_seconds=1.25,
            live_speaker_sentence_hint_new_speaker_max_lag_seconds=8.0,
            live_speaker_sentence_hint_hold_seconds=1.0,
            live_speaker_probe_hold_seconds=2.0,
        )
        diarizer.bus = Bus()
        diarizer.playback_time = lambda: 12.0

        diarizer._maybe_emit_sentence_live_speaker_hint(
            {
                "assigned_speaker": "S2",
                "start": 8.0,
                "end": 10.0,
                "probabilities": {"speaker2": 0.9},
            },
            2.0,
        )

        self.assertEqual(diarizer.bus.events, [])

    def test_sentence_live_speaker_hint_allows_new_speaker_lag(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_sentence_hint=True,
            live_speaker_sentence_hint_max_lag_seconds=1.25,
            live_speaker_sentence_hint_new_speaker_max_lag_seconds=8.0,
            live_speaker_sentence_hint_hold_seconds=1.0,
            live_speaker_probe_hold_seconds=2.0,
        )
        diarizer.bus = Bus()
        diarizer.playback_time = lambda: 15.0

        diarizer._maybe_emit_sentence_live_speaker_hint(
            {
                "assigned_speaker": "S2",
                "created_speaker": True,
                "start": 8.0,
                "end": 10.0,
                "probabilities": {"speaker2": 1.0},
            },
            2.0,
        )

        self.assertEqual(len(diarizer.bus.events), 1)
        _event, payload = diarizer.bus.events[0]
        self.assertEqual(payload["speaker_id"], "S2")
        self.assertEqual(payload["live_hint_lag_seconds"], 5.0)

    def test_raw_live_speaker_change_snap_promotes_strong_unsmoothed_change(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_raw_change_snap=True,
            live_speaker_raw_change_min_probability=0.62,
            live_speaker_raw_change_min_margin=0.18,
            realtime_preview_diarize_min_known_probability=0.5,
        )
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}

        payload = diarizer._maybe_promote_raw_live_speaker_change(
            "S1",
            {
                "assigned_speaker": "S1",
                "speaker_name": "",
                "probabilities": {"speaker1": 0.7, "speaker2": 0.2, "unknown": 0.1},
                "raw_probabilities": {"speaker1": 0.25, "speaker2": 0.78, "unknown": 0.05},
                "assignment_source": "live_fast_embedding_ema",
            },
        )

        self.assertEqual(payload["assigned_speaker"], "S2")
        self.assertEqual(payload["assignment_source"], "live_fast_embedding_raw_change_snap")
        self.assertEqual(payload["raw_change_previous_speaker"], "S1")
        self.assertEqual(payload["smoothed_assigned_speaker"], "S1")

    def test_raw_live_speaker_change_snap_requires_margin(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_raw_change_snap=True,
            live_speaker_raw_change_min_probability=0.62,
            live_speaker_raw_change_min_margin=0.18,
            realtime_preview_diarize_min_known_probability=0.5,
        )

        original = {
            "assigned_speaker": "S1",
            "probabilities": {"speaker1": 0.7, "speaker2": 0.2, "unknown": 0.1},
            "raw_probabilities": {"speaker1": 0.58, "speaker2": 0.7, "unknown": 0.05},
            "assignment_source": "live_fast_embedding_ema",
        }
        payload = diarizer._maybe_promote_raw_live_speaker_change("S1", original)

        self.assertIs(payload, original)

    def test_live_speaker_embedding_throttle_uses_latency_target(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_embedding_min_interval_seconds=0.75,
            live_speaker_embedding_target_utilization=0.25,
        )
        diarizer.bus = Bus()

        self.assertTrue(diarizer._try_reserve_live_speaker_embedding())
        self.assertFalse(diarizer._try_reserve_live_speaker_embedding())

        diarizer._record_live_speaker_embedding_latency(1.0)

        remaining = diarizer._live_speaker_embedding_next_at - time.monotonic()
        self.assertGreaterEqual(remaining, 2.8)
        self.assertLessEqual(remaining, 3.2)
        self.assertTrue(any(event == "status" for event, _payload in diarizer.bus.events))

    def test_live_speaker_memory_update_is_queued_when_worker_exists(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._live_embedding_separate = True
        diarizer._speaker_generation = 4
        diarizer._live_memory_update_jobs = queue.Queue(maxsize=2)
        diarizer.bus = Bus()
        diarizer._embed_live_audio_chunk = mock.Mock(return_value=np.array([1.0, 0.0], dtype=np.float32))

        diarizer._update_live_speaker_memory(
            "S1",
            np.array([0.2, 0.3], dtype=np.float32),
            16000,
            1.25,
            ".live-sentence.wav",
            speaker_generation=4,
        )

        diarizer._embed_live_audio_chunk.assert_not_called()
        job = diarizer._live_memory_update_jobs.get_nowait()
        diarizer._live_memory_update_jobs.task_done()
        self.assertIsInstance(job, LiveSpeakerMemoryUpdateJob)
        assert isinstance(job, LiveSpeakerMemoryUpdateJob)
        self.assertEqual(job.speaker_id, "S1")
        self.assertEqual(job.sample_rate, 16000)
        self.assertEqual(job.duration_seconds, 1.25)
        self.assertEqual(job.suffix, ".live-sentence.wav")
        self.assertEqual(job.speaker_generation, 4)

    def test_live_speaker_memory_update_reembeds_with_live_provider(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        class Memory:
            def __init__(self) -> None:
                self.upserts: list[tuple[str, np.ndarray, float, int]] = []

            def upsert_profile(
                self,
                label: str,
                embedding: np.ndarray,
                duration_seconds: float = 0.0,
                sentence_count: int = 1,
                locked: bool = False,
            ) -> str:
                self.upserts.append((label, embedding, duration_seconds, sentence_count))
                return label

        live_embedding = np.array([0.0, 1.0], dtype=np.float32)
        audio = np.array([0.2, 0.3], dtype=np.float32)
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._speaker_generation = 6
        diarizer.bus = Bus()
        diarizer.live_memory = Memory()
        diarizer._embed_live_audio_chunk = mock.Mock(return_value=live_embedding)

        diarizer._process_live_speaker_memory_update(
            LiveSpeakerMemoryUpdateJob(
                speaker_id="S2",
                audio=audio,
                sample_rate=16000,
                duration_seconds=2.5,
                suffix=".live-sentence.wav",
                speaker_generation=6,
            )
        )

        diarizer._embed_live_audio_chunk.assert_called_once()
        np.testing.assert_array_equal(diarizer._embed_live_audio_chunk.call_args.args[0], audio)
        self.assertEqual(diarizer._embed_live_audio_chunk.call_args.args[1:], (16000, ".live-sentence.wav"))
        self.assertEqual(len(diarizer.live_memory.upserts), 1)
        label, embedding, duration_seconds, sentence_count = diarizer.live_memory.upserts[0]
        self.assertEqual(label, "S2")
        np.testing.assert_array_equal(embedding, live_embedding)
        self.assertEqual(duration_seconds, 2.5)
        self.assertEqual(sentence_count, 1)

    def test_stale_live_speaker_memory_update_does_not_upsert(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        class Memory:
            def upsert_profile(self, *_args: object, **_kwargs: object) -> str:
                raise AssertionError("stale live speaker memory update should not upsert")

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._speaker_generation = 8
        diarizer.bus = Bus()
        diarizer.live_memory = Memory()
        diarizer._embed_live_audio_chunk = mock.Mock(return_value=np.array([1.0, 0.0], dtype=np.float32))

        diarizer._process_live_speaker_memory_update(
            LiveSpeakerMemoryUpdateJob(
                speaker_id="S1",
                audio=np.array([0.2, 0.3], dtype=np.float32),
                sample_rate=16000,
                duration_seconds=1.0,
                speaker_generation=7,
            )
        )

        diarizer._embed_live_audio_chunk.assert_not_called()

    def test_live_speaker_memory_update_rechecks_generation_after_embedding(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        class Memory:
            def upsert_profile(self, *_args: object, **_kwargs: object) -> str:
                raise AssertionError("live speaker memory update should not upsert after generation changes")

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._speaker_generation = 3
        diarizer._live_memory_update_lock = threading.Lock()
        diarizer.bus = Bus()
        diarizer.live_memory = Memory()

        def embed_and_clear(*_args: object) -> np.ndarray:
            diarizer._speaker_generation = 4
            return np.array([1.0, 0.0], dtype=np.float32)

        diarizer._embed_live_audio_chunk = mock.Mock(side_effect=embed_and_clear)

        diarizer._process_live_speaker_memory_update(
            LiveSpeakerMemoryUpdateJob(
                speaker_id="S1",
                audio=np.array([0.2, 0.3], dtype=np.float32),
                sample_rate=16000,
                duration_seconds=1.0,
                speaker_generation=3,
            )
        )

        diarizer._embed_live_audio_chunk.assert_called_once()

    def test_live_speaker_change_verification_uses_full_stack_result(self) -> None:
        class Bus:
            def emit(self, event: str, payload: dict[str, object]) -> None:
                pass

        class Memory:
            def profile_count(self) -> int:
                return 2

            def score_existing(
                self,
                embedding: np.ndarray,
                duration_seconds: float,
                min_similarity: float | None = None,
                min_margin: float | None = None,
            ) -> object:
                return argparse.Namespace(
                    assigned_speaker="S2",
                    probabilities={"unknown": 0.05, "speaker1": 0.05, "speaker2": 0.9},
                    similarities={"S1": 0.2, "S2": 0.8},
                    unknown_probability=0.05,
                    top_similarity=0.8,
                    margin=0.6,
                    quality=1.0,
                )

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            live_speaker_verify_on_change=True,
            live_speaker_verify_min_interval_seconds=0.0,
            realtime_preview_diarize_min_audio_seconds=0.2,
            realtime_preview_diarize_min_similarity=0.45,
            realtime_preview_diarize_min_margin=0.08,
            realtime_preview_diarize_min_known_probability=0.5,
            min_embed_seconds=0.5,
        )
        diarizer.bus = Bus()
        diarizer.memory = Memory()
        diarizer._speaker_lock = threading.Lock()
        diarizer._speaker_metadata = {}
        diarizer._live_embedding_separate = True
        diarizer._live_speaker_verify_lock = threading.Lock()
        diarizer._live_speaker_verify_next_at = 0.0
        diarizer._embed_audio_chunk = mock.Mock(return_value=np.array([1.0, 0.0], dtype=np.float32))

        result = diarizer._verify_live_speaker_change(
            np.ones(20, dtype=np.float32),
            10,
            2.0,
            {"assigned_speaker": "S1", "probabilities": {"speaker1": 0.85, "unknown": 0.15}},
            "S1",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["assigned_speaker"], "S2")
        self.assertEqual(result["assignment_source"], "live_full_stack_change_verify")
        self.assertEqual(result["fast_assigned_speaker"], "S1")


class EmbeddingSubprocessClientTests(unittest.TestCase):
    def test_embed_wav_times_out_and_kills_unresponsive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "silent_embedding_helper.py"
            helper.write_text(
                "import sys, time\n"
                "for _line in sys.stdin:\n"
                "    time.sleep(10)\n",
                encoding="utf-8",
            )
            audio = root / "audio.wav"
            audio.write_bytes(b"")

            client = EmbeddingSubprocessClient(
                python=Path(sys.executable),
                provider="noop",
                device="cpu",
                helper_script=helper,
                response_timeout_seconds=0.2,
            )
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.embed_wav(audio)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 2.0)
            self.assertIsNone(client._process)
            client.shutdown(lock_timeout_seconds=0.1)


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
            words, segment_count = client.transcribe_window(np.zeros(160, dtype=np.float32), 16000, 5)

        self.assertEqual(words, [])
        self.assertEqual(segment_count, 0)
        self.assertIn("language=de", captured["url"])
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
        controller = WindowDiarizer.__new__(WindowDiarizer)
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


class RemoteEmbeddingClientTests(unittest.TestCase):
    def test_remote_embedding_client_posts_pcm16_with_encoded_provider(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = json.dumps(payload).encode("utf-8")

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.payload

        calls: list[tuple[str, bytes | None, float | None]] = []

        def fake_urlopen(request_or_url: object, timeout: float | None = None) -> FakeResponse:
            url = getattr(request_or_url, "full_url", request_or_url)
            data = getattr(request_or_url, "data", None)
            calls.append((str(url), data, timeout))
            if str(url).endswith("/health"):
                return FakeResponse({"ok": True, "service": "embeddings"})
            if "/load?" in str(url):
                return FakeResponse({"ok": True})
            if "/embed-pcm16?" in str(url):
                return FakeResponse({"ok": True, "embedding": [1.0, 2.0, 2.0]})
            raise AssertionError(f"Unexpected URL: {url}")

        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "voice.wav"
            write_wav(wav_path, np.ones(1600, dtype=np.float32) * 0.1, 16000)
            client = RemoteEmbeddingClient(
                "http://127.0.0.1:8660",
                "espnet_ecapa_wavlm_joint=0.725+jungjee_rawnet3=1",
                timeout_seconds=12.0,
            )
            with mock.patch("embeddings.embedding_providers.urlopen", side_effect=fake_urlopen):
                self.assertEqual(client.health()["service"], "embeddings")
                embedding = client.embed_wav(wav_path)

        self.assertTrue(any("/load?" in url for url, _data, _timeout in calls))
        embed_calls = [(url, data) for url, data, _timeout in calls if "/embed-pcm16?" in url]
        self.assertEqual(len(embed_calls), 1)
        embed_url, embed_body = embed_calls[0]
        self.assertIn("%2B", embed_url)
        self.assertIn("encoding=pcm16", embed_url)
        self.assertIsNotNone(embed_body)
        self.assertEqual(len(embed_body or b"") % 2, 0)
        self.assertTrue(np.allclose(embedding, np.array([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], dtype=np.float32)))


class RepositoryStructureTests(unittest.TestCase):
    def test_package_imports_do_not_require_tools_on_sys_path(self) -> None:
        self.assertFalse((ROOT / "tools").exists())
        self.assertEqual(WindowDiarizer.__name__, "WindowDiarizer")

    def test_window_module_entrypoint_prints_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, "-m", "window.youtube_window_diarize_gui", "--help"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Growing-window faster-whisper speaker diarization GUI", completed.stdout)

    def test_runtime_dir_env_redirects_mutable_defaults(self) -> None:
        import paths as paths

        original_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["WHOSPEAKS_RUNTIME_DIR"] = directory
                os.environ.pop("WHOSPEAKS_CACHE_DIR", None)
                os.environ.pop("WHOSPEAKS_MODEL_DIR", None)
                os.environ.pop("WHOSPEAKS_SPEAKER_LIBRARY_DIR", None)
                reloaded = importlib.reload(paths)
                runtime = Path(directory).resolve()
                self.assertEqual(reloaded.RUNTIME_DIR, runtime)
                self.assertEqual(reloaded.CACHE_DIR, runtime / "cache")
                self.assertEqual(reloaded.MODEL_DIR, runtime / "models")
                self.assertEqual(reloaded.SPEAKER_LIBRARY_DIR, runtime / "speakers")
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(paths)

    def test_window_gui_default_embedding_provider_ignores_environment_override(self) -> None:
        import window.window_config as window_config

        original_env = dict(os.environ)
        try:
            os.environ["WHOSPEAKS_WINDOW_EMBEDDING_PROVIDER"] = "speechbrain_ecapa"
            reloaded = importlib.reload(window_config)
            self.assertEqual(
                reloaded.DEFAULT_WINDOW_EMBEDDING_PROVIDER,
                "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37",
            )
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(window_config)

    def test_window_gui_tuned_default_parameters_match_promoted_set(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        expected = {
            "embedding_provider": (
                "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
            ),
            "interval_seconds": 0.7,
            "same_speaker_similarity": 0.43,
            "similarity_temperature": 0.061,
            "speaker_softmax_temperature": 0.0557,
            "new_speaker_threshold": 0.4309,
            "duplicate_profile_similarity": 0.4247,
            "unknown_short_threshold": 0.287,
            "min_first_speaker_seconds": 1.8373,
            "min_new_speaker_seconds": 2.0358,
            "late_new_speaker_min_seconds": 3.1604,
            "max_speakers": 12,
            "min_margin": 0.0372,
            "margin_temperature": 0.0361,
            "update_unknown_max": 0.4289,
            "new_speaker_confirmation_count": 1,
            "new_speaker_confirmation_similarity": 0.5801,
            "max_pending_new_speakers": 6,
            "known_speaker_min_similarity": 0.5563,
            "known_speaker_gray_zone_min_unknown_probability": 0.064,
            "profile_update_min_similarity": 0.5011,
            "profile_update_min_margin": 0.0037,
            "low_similarity_unknown_floor_similarity": 0.56,
            "low_similarity_unknown_floor_probability": 0.1885,
            "gray_zone_promote_max_similarity": 0.55,
            "min_new_speaker_words": 3,
            "retro_reassign_min_similarity": 0.02,
            "retro_reassign_min_margin": 0.0,
            "speaker_refinement_final_passes": 1,
            "speaker_refinement_small_island_merge": True,
            "speaker_refinement_tiny_fragmented_merge": True,
            "speaker_refinement_tiny_fragmented_max_duration": 6.0,
            "speaker_refinement_tiny_fragmented_max_segments": 8,
            "speaker_refinement_tiny_fragmented_min_islands": 2,
            "speaker_refinement_tiny_fragmented_max_islands": 3,
            "speaker_refinement_tiny_fragmented_min_neighbor_share": 0.5,
            "speaker_refinement_terminal_outro_merge": True,
            "speaker_refinement_terminal_outro_max_duration": 12.0,
            "speaker_refinement_terminal_outro_lookback_segments": 2,
            "speaker_refinement_terminal_outro_min_target_duration": 5.0,
            "speaker_refinement_unknown_same_speaker_fill": True,
            "speaker_refinement_unknown_same_speaker_max_duration": 3.0,
            "speaker_refinement_unknown_same_speaker_max_segments": 1,
            "speaker_refinement_unknown_previous_speaker_fill": True,
            "speaker_refinement_unknown_previous_speaker_max_duration": 0.75,
            "speaker_refinement_unknown_previous_speaker_max_segments": 1,
            "speaker_refinement_unknown_previous_speaker_max_previous_gap": 0.35,
            "speaker_refinement_unknown_previous_speaker_min_next_gap": 0.3,
            "speaker_refinement_unknown_next_speaker_fill": True,
            "speaker_refinement_unknown_next_speaker_max_duration": 1.75,
            "speaker_refinement_unknown_next_speaker_max_segments": 1,
            "speaker_refinement_unknown_next_speaker_max_next_gap": 0.05,
            "speaker_refinement_unknown_next_speaker_min_previous_gap": 0.15,
            "speaker_refinement_long_low_confidence_retro_split": True,
            "speaker_refinement_long_low_confidence_retro_min_duration": 4.0,
            "speaker_refinement_long_low_confidence_retro_max_similarity": 0.06,
            "speaker_refinement_long_low_confidence_retro_max_margin": 0.04,
            "speaker_refinement_long_low_confidence_retro_max_splits": 1,
            "min_embed_seconds": 0.5,
            "min_speech_audio_ratio": 0.0,
            "live_speaker_embedding_provider": "jungjee_rawnet3",
            "unstable_tail_seconds": 1.35,
            "vad_silence_seconds": 1.1,
            "vad_final_window_post_silence_seconds": 0.75,
            "sentence_boundary_pre_padding_seconds": 0.06,
            "sentence_boundary_post_padding_seconds": 0.09,
            "sentence_boundary_gap_ratio": 0.6,
            "realtime_preview_model_preset": "community-64l",
            "realtime_preview_model": "Kroko-EN-Community-64-L-Streaming-001.data",
            "realtime_preview_startup_timeout_seconds": 12.0,
            "realtime_preview_diarize_min_audio_seconds": 1.5,
            "realtime_preview_diarize_min_advance_seconds": 0.75,
            "realtime_preview_diarize_min_similarity": 0.45,
            "realtime_preview_diarize_min_margin": 0.08,
            "realtime_preview_diarize_min_known_probability": 0.5,
            "live_speaker_assignment": True,
            "live_speaker_embedding_min_interval_seconds": 0.2,
            "live_speaker_embedding_target_utilization": 1.0,
            "live_speaker_verify_on_change": False,
            "live_speaker_verify_min_interval_seconds": 2.0,
            "live_speaker_ema_window_seconds": 1.0,
            "live_speaker_ema_count": 1,
            "live_speaker_ema_alpha": 0.55,
            "live_speaker_probe_interval_seconds": 0.2,
            "live_speaker_probe_attack_interval_seconds": 0.0,
            "live_speaker_probe_window_seconds": 1.0,
            "live_speaker_probe_hold_seconds": 1.0,
            "live_speaker_probe_min_advance_seconds": 0.2,
            "live_speaker_probe_attack_min_advance_seconds": 0.0,
            "live_speaker_probe_min_speech_seconds": 0.15,
            "live_speaker_probe_clear_on_silence": True,
            "live_speaker_probe_clear_window_seconds": 1.0,
            "live_speaker_probe_clear_silence_count": 1,
            "live_speaker_probe_clear_unknown_count": 2,
            "live_speaker_probe_unknown_clear_debounce_seconds": 0.0,
            "live_speaker_probe_unknown_keepalive": False,
            "live_speaker_probe_unknown_release_smoothing": "none",
            "live_speaker_probe_unknown_release_count": 3,
            "live_speaker_probe_unknown_release_ema_alpha": 0.5,
            "live_speaker_probe_unknown_release_margin": 0.0,
            "live_speaker_raw_change_snap": True,
            "live_speaker_raw_change_min_probability": 0.7,
            "live_speaker_raw_change_min_margin": 0.25,
            "live_speaker_sentence_hint": True,
            "live_speaker_sentence_hint_override": True,
            "live_speaker_sentence_hint_max_lag_seconds": 1.25,
            "live_speaker_sentence_hint_new_speaker_max_lag_seconds": 1.25,
            "live_speaker_sentence_hint_hold_seconds": 0.3,
            "browser_live_observation_output": None,
            "browser_live_observation_interval_seconds": 0.1,
            "browser_live_observation_max_sample_gap_seconds": 0.5,
            "browser_live_observation_flicker_gap_seconds": 0.25,
        }

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui.py"]):
            args = parse_args()

        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)

    def test_window_gui_can_select_kroko_pro_16l_preview_preset(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--realtime-preview-model-preset",
                "pro-16l",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.realtime_preview_model_preset, "pro-16l")
        self.assertEqual(args.realtime_preview_model, "Kroko-EN-Pro-16-L-Streaming-001.data")
        if args.realtime_preview_model_path is not None:
            self.assertEqual(args.realtime_preview_model_path.name, "Kroko-EN-Pro-16-L-Streaming-001.data")
        self.assertEqual(args.realtime_preview_startup_timeout_seconds, 45.0)
        self.assertEqual(args.realtime_preview_interval_seconds, 0.32)
        self.assertEqual(args.realtime_preview_min_audio_seconds, 0.32)
        self.assertEqual(args.realtime_preview_min_advance_seconds, 0.32)
        self.assertEqual(args.realtime_preview_feed_chunk_seconds, 0.32)

    def test_window_gui_language_selects_kroko_and_sentence_tokenizer(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "de",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "de")
        self.assertEqual(args.realtime_preview_language, "de")
        self.assertEqual(args.realtime_preview_model_preset, "community-64l")
        self.assertEqual(args.realtime_preview_model, "Kroko-DE-Community-64-L-Streaming-001.data")
        self.assertEqual(args.sentence_tokenizer, "nltk+rule-based")
        self.assertEqual(args.sentence_language, "de")

    def test_window_gui_env_language_is_not_overridden_by_custom_preview_model(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.dict(os.environ, {"WHOSPEAKS_LANGUAGE": "de"}):
            with mock.patch.object(
                sys,
                "argv",
                [
                    "youtube_window_diarize_gui.py",
                    "--realtime-preview-model",
                    "Kroko-EN-Community-64-L-Streaming-001.data",
                ],
            ):
                args = parse_args()

        self.assertEqual(args.language, "de")
        self.assertEqual(args.realtime_preview_language, "de")
        self.assertEqual(args.realtime_preview_model_preset, "custom")
        self.assertEqual(args.sentence_language, "de")

    def test_window_gui_extended_language_requires_preview_off(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "pl",
            ],
        ):
            with mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_window_gui_extended_language_selects_nltk_when_preview_off(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "pl",
                "--realtime-preview-engine",
                "off",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "pl")
        self.assertEqual(args.sentence_tokenizer, "nltk+rule-based")
        self.assertEqual(args.sentence_language, "pl")
        self.assertEqual(args.realtime_preview_engine, "off")
        self.assertEqual(args.realtime_preview_model, "")
        self.assertIsNone(args.realtime_preview_model_path)

    def test_window_gui_extended_language_uses_stanza_when_needed(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "zh",
                "--realtime-preview-engine",
                "off",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "zh")
        self.assertEqual(args.sentence_tokenizer, "stanza")
        self.assertEqual(args.sentence_language, "zh-hans")

    def test_kroko_preview_model_path_searches_configured_model_dir(self) -> None:
        from window.window_config import default_kroko_preview_model_path

        model_name = "Kroko-EN-Pro-16-L-Streaming-001.data"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / model_name
            model_path.write_bytes(b"")

            with mock.patch.dict(os.environ, {"WHOSPEAKS_KROKO_PREVIEW_MODEL_DIR": directory}):
                resolved = default_kroko_preview_model_path(model_name, use_env=False)

        self.assertEqual(resolved, model_path)

    def test_kroko_preview_community_model_downloads_to_model_dir(self) -> None:
        from window.window_config import download_kroko_preview_model

        model_name = "Kroko-DE-Community-64-L-Streaming-001.data"

        def fake_hf_hub_download(**kwargs: object) -> str:
            target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
            target.write_bytes(b"model")
            return str(target)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download) as download:
                resolved = download_kroko_preview_model(model_name, target_dir=Path(directory))

        self.assertEqual(resolved.name, model_name)
        self.assertTrue(download.called)

    def test_kroko_preview_auto_download_rejects_non_public_model(self) -> None:
        from window.window_config import download_kroko_preview_model

        with self.assertRaisesRegex(RuntimeError, "public Community"):
            download_kroko_preview_model("Kroko-EN-Pro-16-L-Streaming-001.data")

    def test_window_loop_restarts_interval_after_successful_split(self) -> None:
        source = inspect.getsource(WindowDiarizer._run)

        guard = "if transcript.sentences or vad_next_left is not None:"
        guard_at = source.index(guard)
        advance_at = source.index("self._advance_realtime_preview_after_commit(left)", guard_at)
        cooldown_guard_at = source.index("if interval_seconds > 0.0 and not media_final_flush:", advance_at)
        cooldown_at = source.index("next_tick = time.monotonic() + interval_seconds", cooldown_guard_at)

        self.assertLess(guard_at, advance_at)
        self.assertLess(advance_at, cooldown_guard_at)
        self.assertLess(cooldown_guard_at, cooldown_at)

    def test_window_gui_accepts_remote_embeddings_backend_alias(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "-embeddings-backend",
                "remote",
                "--remote-embeddings-url",
                "http://127.0.0.1:8660",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.embeddings_backend, "remote")
        self.assertEqual(args.remote_embeddings_url, "http://127.0.0.1:8660")

    def test_window_gui_can_disable_live_speaker_assignment_with_master_switch(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--no-live-speaker-assignment",
                "--live-speaker-embedding-provider",
                "pyannote_wespeaker_resnet34_lm",
            ],
        ):
            args = parse_args()

        self.assertFalse(args.live_speaker_assignment)
        self.assertFalse(args.live_speaker_probe)
        self.assertFalse(args.live_speaker_sentence_hint)
        self.assertFalse(args.live_speaker_highlight_transcript)
        self.assertFalse(args.live_speaker_verify_on_change)
        self.assertFalse(args.live_speaker_raw_change_snap)

    def test_cunk_canonical_is_a_small_fixture(self) -> None:
        from paths import CUNK_CANONICAL

        self.assertTrue(CUNK_CANONICAL.is_file())
        self.assertIn("tests", CUNK_CANONICAL.parts)
        self.assertIn("fixtures", CUNK_CANONICAL.parts)

    def test_legacy_tools_folder_is_removed(self) -> None:
        self.assertFalse((ROOT / "tools").exists())


if __name__ == "__main__":
    unittest.main()
