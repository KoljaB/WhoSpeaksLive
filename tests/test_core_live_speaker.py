from __future__ import annotations

import argparse
import queue
import sys
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

from embeddings.embedding_providers import (
    EmbeddingComponentResult,
    EmbeddingResult,
    RemoteEmbeddingClient,
)
from embeddings.provider_identity import PROMOTED_PUBLIC_PROVIDER, PUBLIC_PROVIDER
from speakers.speaker_embedding_cluster import SpeakerDecision
from window.window_domain import EmbeddingSentenceJob, LiveSpeakerMemoryUpdateJob
from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_probe_scoring import score_live_speaker_probe



from tests.window_diarizer_support import make_window_diarizer


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


class LiveSpeakerRuntimeTests(unittest.TestCase):
    def test_sentence_live_speaker_hint_emits_fresh_assignment(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
        diarizer.args = argparse.Namespace(
            live_speaker_assignment=False,
            live_speaker_sentence_hint=True,
        )
        diarizer.bus = Bus()

        diarizer._maybe_emit_sentence_live_speaker_hint({"assigned_speaker": "S2"}, 2.0)

        self.assertEqual(diarizer.bus.events, [])

    def test_live_speaker_assignment_off_reuses_main_embedding_provider(self) -> None:
        main_embedding = object()
        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
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
        diarizer = make_window_diarizer()
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
        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
        diarizer._live_embedding_separate = True
        diarizer._speaker_generation = 4
        diarizer._live_memory_update_jobs = queue.Queue(maxsize=2)
        diarizer.bus = Bus()
        diarizer._embed_live_audio_chunk = mock.Mock(return_value=np.array([1.0, 0.0], dtype=np.float32))
        audio = np.array([0.2, 0.3], dtype=np.float32)

        diarizer._update_live_speaker_memory(
            "S1",
            audio,
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
        self.assertEqual(job.speaker_label_generation, 0)
        np.testing.assert_array_equal(job.audio, audio)
        self.assertIsNot(job.audio, audio)

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
        diarizer = make_window_diarizer()
        diarizer._speaker_generation = 6
        diarizer.bus = Bus()
        diarizer.memory = object()
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

    def test_final_public_stacks_reuse_their_unique_live_component(self) -> None:
        component = np.array([0.0, 1.0], dtype=np.float32)
        for final_provider in (PROMOTED_PUBLIC_PROVIDER, PUBLIC_PROVIDER):
            with self.subTest(final_provider=final_provider):
                diarizer = make_window_diarizer()
                diarizer.embedding = RemoteEmbeddingClient(
                    "http://127.0.0.1:8660", final_provider, device="cuda"
                )
                diarizer.live_embedding = RemoteEmbeddingClient(
                    "http://127.0.0.1:8660", "speechbrain_resnet", device="cuda"
                )
                diarizer._live_embedding_separate = True
                diarizer._update_config(enhance_embeddings=False, keep_segment_audio=False)
                result = EmbeddingResult(
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                    components=(EmbeddingComponentResult(
                        provider="speechbrain_resnet",
                        weight=0.38 if final_provider == PUBLIC_PROVIDER else 0.28,
                        embedding=component,
                    ),),
                )

                reused = diarizer._reusable_live_embedding_from_final_result(result)

                np.testing.assert_array_equal(reused, component)

    def test_final_component_reuse_falls_back_for_enhancement_or_ambiguous_provider(self) -> None:
        component = np.array([0.0, 1.0], dtype=np.float32)
        diarizer = make_window_diarizer()
        diarizer.embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660", PROMOTED_PUBLIC_PROVIDER, device="cuda"
        )
        diarizer.live_embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660", "speechbrain_resnet", device="cuda"
        )
        diarizer._live_embedding_separate = True
        diarizer._update_config(keep_segment_audio=False)
        result = EmbeddingResult(
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            components=(
                EmbeddingComponentResult("speechbrain_resnet", 0.28, component),
                EmbeddingComponentResult("speechbrain_resnet", 0.10, component),
            ),
        )

        diarizer._update_config(enhance_embeddings=False)
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(result))
        unique_result = EmbeddingResult(
            embedding=result.embedding,
            components=(result.components[0],),
        )
        mismatched_weight = EmbeddingResult(
            embedding=result.embedding,
            components=(EmbeddingComponentResult(
                "speechbrain_resnet", 0.29, component
            ),),
        )
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(mismatched_weight))
        diarizer._update_config(enhance_embeddings=True)
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(unique_result))

        diarizer._update_config(enhance_embeddings=False, keep_segment_audio=True)
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(unique_result))

        diarizer._update_config(keep_segment_audio=False)
        diarizer.live_embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8661", "speechbrain_resnet", device="cuda"
        )
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(unique_result))

        diarizer.live_embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660", "speechbrain_resnet", device="cpu"
        )
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(unique_result))

        diarizer.live_embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660",
            "speechbrain_resnet=1.0+wespeaker_campplus=0.5",
            device="cuda",
        )
        self.assertIsNone(diarizer._reusable_live_embedding_from_final_result(unique_result))

    def test_live_speaker_memory_update_uses_precomputed_embedding_without_remote_call(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        class Memory:
            def __init__(self) -> None:
                self.upserts: list[tuple[str, np.ndarray]] = []

            def upsert_profile(self, label: str, embedding: np.ndarray, **_kwargs: object) -> str:
                self.upserts.append((label, np.asarray(embedding, dtype=np.float32)))
                return label

        live_embedding = np.array([0.25, 0.75], dtype=np.float32)
        diarizer = make_window_diarizer()
        diarizer._speaker_generation = 6
        diarizer.bus = Bus()
        diarizer.memory = object()
        diarizer.live_memory = Memory()
        diarizer._embed_live_audio_chunk = mock.Mock(side_effect=AssertionError("unexpected remote call"))

        diarizer._process_live_speaker_memory_update(LiveSpeakerMemoryUpdateJob(
            speaker_id="S2",
            audio=np.array([0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            duration_seconds=2.5,
            speaker_generation=6,
            precomputed_embedding=live_embedding,
        ))

        diarizer._embed_live_audio_chunk.assert_not_called()
        self.assertEqual(len(diarizer.live_memory.upserts), 1)
        np.testing.assert_array_equal(diarizer.live_memory.upserts[0][1], live_embedding)

    def test_precomputed_live_memory_job_does_not_copy_sentence_audio(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._live_embedding_separate = True
        diarizer._live_memory_update_jobs = queue.Queue(maxsize=2)
        audio = np.ones(16000, dtype=np.float32)
        live_embedding = np.array([0.25, 0.75], dtype=np.float32)

        diarizer._update_live_speaker_memory(
            "S1",
            audio,
            16000,
            1.0,
            precomputed_embedding=live_embedding,
        )

        job = diarizer._live_memory_update_jobs.get_nowait()
        diarizer._live_memory_update_jobs.task_done()
        self.assertEqual(job.audio.size, 0)
        np.testing.assert_array_equal(job.precomputed_embedding, live_embedding)
        self.assertIsNot(job.precomputed_embedding, live_embedding)

    def test_live_memory_update_uses_post_refinement_sentence_assignment(self) -> None:
        diarizer = make_window_diarizer()
        decision = SpeakerDecision(
            assigned_speaker="S1",
            created_speaker=False,
            probabilities={"speaker1": 1.0},
            similarities={"S1": 1.0},
            unknown_probability=0.0,
            top_similarity=1.0,
            margin=1.0,
            quality=1.0,
        )
        order: list[str] = []
        diarizer._section_gap_new_speaker_decision = mock.Mock(return_value=decision)
        diarizer._emit_transcript_sentence = mock.Mock(side_effect=lambda payload: payload)
        diarizer._maybe_emit_sentence_live_speaker_hint = mock.Mock()
        diarizer._refresh_person_identity_suggestions = mock.Mock(return_value=False)
        diarizer._maybe_checkpoint_confirmed_people = mock.Mock()

        def refine_assignment() -> None:
            order.append("refine")
            with diarizer._sentence_refinement_lock:
                diarizer._sentence_refinement_records[7]["assigned_speaker"] = "S2"

        def enqueue_live_update(*args: object, **kwargs: object) -> None:
            order.append("enqueue")
            self.assertEqual(args[0], "S2")

        diarizer._refine_speaker_assignments = refine_assignment
        diarizer._update_live_speaker_memory = mock.Mock(side_effect=enqueue_live_update)

        with mock.patch("window.window_diarizer_transcription.emit_live_profile_snapshot") as snapshot:
            diarizer._apply_sentence_embedding_decision(
                index=7,
                base_payload={"start": 0.0, "end": 1.0},
                text="A complete sentence.",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                duration_seconds=1.0,
                live_memory_audio=np.ones(16000, dtype=np.float32),
                live_memory_sample_rate=16000,
                live_memory_embedding=np.array([0.0, 1.0], dtype=np.float32),
                run_speaker_refinement=True,
            )

        self.assertEqual(order, ["refine", "enqueue"])
        self.assertEqual(snapshot.call_args.args[2], "S2")

    def test_sentence_processing_carries_reused_component_without_second_remote_call(self) -> None:
        final_embedding = np.array([1.0, 0.0], dtype=np.float32)
        live_embedding = np.array([0.0, 1.0], dtype=np.float32)
        result = EmbeddingResult(
            embedding=final_embedding,
            components=(EmbeddingComponentResult(
                "speechbrain_resnet", 0.28, live_embedding
            ),),
        )
        diarizer = make_window_diarizer()
        diarizer.embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660", PROMOTED_PUBLIC_PROVIDER, device="cuda"
        )
        diarizer.live_embedding = RemoteEmbeddingClient(
            "http://127.0.0.1:8660", "speechbrain_resnet", device="cuda"
        )
        diarizer._live_embedding_separate = True
        diarizer._update_config(enhance_embeddings=False, keep_segment_audio=False)
        diarizer.embedding.embed_audio_result = mock.Mock(return_value=result)
        diarizer.live_embedding.embed_audio = mock.Mock(
            side_effect=AssertionError("unexpected second remote call")
        )
        diarizer._apply_sentence_embedding_decision = mock.Mock()

        diarizer._process_sentence_embedding(EmbeddingSentenceJob(
            index=1,
            base_payload={"start": 0.0, "end": 1.0},
            text="A complete sentence.",
            audio=np.ones(16000, dtype=np.float32) * 0.1,
            sample_rate=16000,
            duration_seconds=1.0,
            speaker_generation=diarizer._speaker_generation,
        ))

        diarizer.embedding.embed_audio_result.assert_called_once()
        diarizer.live_embedding.embed_audio.assert_not_called()
        kwargs = diarizer._apply_sentence_embedding_decision.call_args.kwargs
        np.testing.assert_array_equal(kwargs["embedding"], final_embedding)
        np.testing.assert_array_equal(kwargs["live_memory_embedding"], live_embedding)

    def test_stale_live_speaker_memory_update_does_not_upsert(self) -> None:
        class Bus:
            def emit(self, _event: str, _payload: object) -> None:
                return None

        class Memory:
            def upsert_profile(self, *_args: object, **_kwargs: object) -> str:
                raise AssertionError("stale live speaker memory update should not upsert")

        diarizer = make_window_diarizer()
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

        diarizer = make_window_diarizer()
        diarizer._speaker_generation = 3
        diarizer._live_memory_update_lock = threading.Lock()
        diarizer.bus = Bus()
        diarizer.memory = object()
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

        diarizer = make_window_diarizer()
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


if __name__ == "__main__":
    unittest.main()
