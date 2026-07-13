from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from realtime.external_feed import ExternalFeedState
from realtime.canonical_transcript import read_canonical_segments
from realtime.realtime_capture import RealtimeCapture
from realtime.realtime_cli import parse_args
from realtime.replay_validation import validate_cunk_realtime_replay
from realtime.trace_analysis import filter_trace_records_by_session
from realtime.trace_commands import read_trace_records
from realtime.validation_models import ValidationItem
from speakers.realtime_speaker_memory import SpeakerDecision


class RecordingBus:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def emit(self, event: str, payload: dict) -> None:
        self.records.append((event, payload))


class FakeSpeakerEngine:
    def __init__(self) -> None:
        self.jobs = SimpleNamespace(unfinished_tasks=0)
        self.started_sessions: list[str] = []
        self.shutdown_count = 0

    def start_session(self, session_id: str) -> None:
        self.started_sessions.append(session_id)

    def submit(self, **_kwargs) -> None:
        return

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeRecorder:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.shutdown_count = 0
        self.feed_calls: list[tuple[np.ndarray, int]] = []
        self._shutdown = threading.Event()

    def start(self) -> None:
        self.started += 1

    def text(self) -> str:
        self._shutdown.wait(timeout=0.05)
        return ""

    def feed_audio(self, audio: np.ndarray, *, original_sample_rate: int) -> None:
        self.feed_calls.append((np.asarray(audio).copy(), original_sample_rate))

    def stop(self) -> None:
        self.stopped += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self._shutdown.set()


class BlockingFeedRecorder(FakeRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.feed_entered = threading.Event()
        self.release_feed = threading.Event()

    def feed_audio(self, audio: np.ndarray, *, original_sample_rate: int) -> None:
        self.feed_entered.set()
        self.release_feed.wait(timeout=2.0)
        super().feed_audio(audio, original_sample_rate=original_sample_rate)


def capture_args() -> argparse.Namespace:
    return argparse.Namespace(
        final_word_timestamps=False,
        stop_drain_seconds=0.0,
        stop_embedding_drain_seconds=0.0,
        stop_trailing_silence_seconds=0.0,
    )


class ExternalAudioFeedTests(unittest.TestCase):
    def test_external_feed_owns_recorder_timing_and_idempotent_cleanup(self) -> None:
        recorder = FakeRecorder()
        engine = FakeSpeakerEngine()
        capture = RealtimeCapture(
            capture_args(),
            RecordingBus(),
            speaker_engine=engine,
            recorder_factory=lambda _session, _device: recorder,
        )

        with capture.external_feed(session_id="replay-1", media_id="clip") as feed:
            self.assertEqual(feed.state, ExternalFeedState.ACTIVE)
            feed.feed_audio(
                np.array([1, 2, 3], dtype=np.int16),
                original_sample_rate=16_000,
                media_end_seconds=0.25,
            )
            self.assertEqual(feed.media_time_seconds, 0.25)
            feed.finish()
            feed.finish()

        self.assertEqual(feed.state, ExternalFeedState.CLOSED)
        self.assertEqual(engine.started_sessions, ["replay-1"])
        self.assertEqual(recorder.started, 1)
        self.assertEqual(recorder.stopped, 1)
        self.assertEqual(recorder.shutdown_count, 1)
        self.assertEqual(len(recorder.feed_calls), 1)
        self.assertEqual(recorder.feed_calls[0][1], 16_000)

        capture.shutdown()
        capture.shutdown()
        self.assertEqual(engine.shutdown_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            capture.external_feed()

    def test_context_cleanup_runs_when_producer_raises(self) -> None:
        recorder = FakeRecorder()
        capture = RealtimeCapture(
            capture_args(),
            RecordingBus(),
            speaker_engine=FakeSpeakerEngine(),
            recorder_factory=lambda _session, _device: recorder,
        )

        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            with capture.external_feed() as feed:
                raise RuntimeError("producer failed")

        self.assertTrue(feed.is_closed)
        self.assertEqual(recorder.shutdown_count, 1)

    def test_finish_waits_for_an_inflight_feed_before_closing_recorder(self) -> None:
        recorder = BlockingFeedRecorder()
        capture = RealtimeCapture(
            capture_args(),
            RecordingBus(),
            speaker_engine=FakeSpeakerEngine(),
            recorder_factory=lambda _session, _device: recorder,
        )

        with capture.external_feed() as feed:
            producer = threading.Thread(
                target=feed.feed_audio,
                kwargs={
                    "audio": np.array([1], dtype=np.int16),
                    "original_sample_rate": 16_000,
                },
            )
            producer.start()
            self.assertTrue(recorder.feed_entered.wait(timeout=1.0))

            finished = threading.Event()
            finisher = threading.Thread(
                target=lambda: (feed.finish(), finished.set()),
            )
            finisher.start()
            self.assertFalse(finished.wait(timeout=0.05))
            self.assertEqual(recorder.shutdown_count, 0)

            recorder.release_feed.set()
            producer.join(timeout=1.0)
            finisher.join(timeout=1.0)

        self.assertTrue(finished.is_set())
        self.assertEqual(recorder.shutdown_count, 1)


class ValidationItemTests(unittest.TestCase):
    @staticmethod
    def decision(speaker: str) -> SpeakerDecision:
        return SpeakerDecision(
            assigned_speaker=speaker,
            created_speaker=False,
            probabilities={"unknown": 0.1, "speaker1": 0.9},
            similarities={speaker: 0.8},
            unknown_probability=0.1,
            top_similarity=0.8,
            margin=0.4,
            quality=1.0,
        )

    def test_reassignment_is_copy_on_write_and_embedding_is_detached(self) -> None:
        source_embedding = np.array([1.0, 0.0], dtype=np.float32)
        item = ValidationItem(
            session_id="validation",
            index=0,
            text="hello",
            duration_seconds=1.0,
            embedding=source_embedding,
            decision=self.decision("S1"),
            row_fields={"index": 0, "canonical_speaker": "A"},
        )
        source_embedding[0] = 0.0
        reassigned = item.with_decision(self.decision("S2"))

        self.assertEqual(float(item.embedding[0]), 1.0)
        self.assertFalse(item.embedding.flags.writeable)
        self.assertEqual(item.decision.assigned_speaker, "S1")
        self.assertEqual(reassigned.decision.assigned_speaker, "S2")
        self.assertFalse(item.reassigned)
        self.assertTrue(reassigned.reassigned)
        with self.assertRaises(FrozenInstanceError):
            item.reassigned = True


class RealtimeCliAndTraceTests(unittest.TestCase):
    def test_canonical_reader_accepts_list_and_segment_document_shapes(self) -> None:
        direct = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hi"}]
        document = {
            "segments": [
                {
                    "speaker_id": "B",
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "text": "There",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            list_path = Path(directory) / "list.json"
            document_path = Path(directory) / "document.json"
            list_path.write_text(json.dumps(direct), encoding="utf-8")
            document_path.write_text(
                json.dumps(document), encoding="utf-8"
            )

            self.assertEqual(read_canonical_segments(list_path), direct)
            self.assertEqual(
                read_canonical_segments(document_path),
                [
                    {
                        "speaker": "B",
                        "start": 1.0,
                        "end": 2.0,
                        "text": "There",
                    }
                ],
            )

    def test_parse_args_accepts_explicit_argv_and_adjusts_english_model(self) -> None:
        args = parse_args(["--language", "de"])
        self.assertEqual(args.language, "de")
        self.assertEqual(args.rt_model, "tiny")
        with self.assertRaises(FrozenInstanceError):
            args.rt_model = "other"

        explicit = parse_args(["--language", "de", "--rt-model", "tiny.en"])
        self.assertEqual(explicit.rt_model, "tiny.en")

    def test_trace_session_selection_is_deterministic(self) -> None:
        records = [
            {"time": 1.0, "payload": {"session_id": "old"}},
            {"time": 2.0, "payload": {"session_id": "new"}},
            {"time": 3.0, "payload": {"session_id": "old"}},
        ]

        latest, session_id = filter_trace_records_by_session(records, "latest")
        self.assertEqual(session_id, "new")
        self.assertEqual(len(latest), 1)

        all_records, session_id = filter_trace_records_by_session(records, "all")
        self.assertIsNone(session_id)
        self.assertEqual(all_records, records)

    def test_jsonl_reader_skips_partial_and_non_object_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                '{"event": "first"}\n'
                '{"event":\n'
                '["not", "an", "event"]\n'
                '{"event": "last"}\n',
                encoding="utf-8",
            )
            records = read_trace_records(path)

        self.assertEqual(
            [record["event"] for record in records],
            ["first", "last"],
        )


class ReplayValidationTests(unittest.TestCase):
    def test_replay_uses_external_feed_contract_and_preserves_summary_shape(self) -> None:
        class FakeFeed:
            def __init__(self) -> None:
                self.statuses: list[str] = []
                self.media_times: list[float] = []
                self.finish_kwargs: dict | None = None

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def report_status(self, message: str) -> None:
                self.statuses.append(message)

            def feed_audio(
                self,
                _audio: np.ndarray,
                *,
                original_sample_rate: int,
                media_end_seconds: float,
            ) -> None:
                self.assert_sample_rate = original_sample_rate
                self.media_times.append(media_end_seconds)

            def finish(self, **kwargs) -> None:
                self.finish_kwargs = kwargs

        class FakeCapture:
            latest = None

            def __init__(self, _args, _bus) -> None:
                self.feed = FakeFeed()
                self.shutdown_count = 0
                type(self).latest = self

            def external_feed(self, **_kwargs):
                return self.feed

            def shutdown(self) -> None:
                self.shutdown_count += 1

        analysis = {
            "match_mode": "text",
            "final_segments": 0,
            "resolved_segments": 0,
            "timestamped_segments": 0,
            "live_final_words": 0,
            "canonical_words": 0,
            "text_recall": 0.0,
            "text_precision": 0.0,
            "assigned_counts": {},
            "profile_map": {},
            "unknown_segments": 0,
            "segment_accuracy": 0.0,
            "duration_accuracy": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                trace_log=root / "trace.jsonl",
                validation_audio=root / "audio.wav",
                validation_canonical=root / "canonical.json",
                validation_output=root / "result.json",
                replay_chunk_seconds=0.5,
                replay_speed=8.0,
                replay_sleep=False,
                replay_trailing_silence_seconds=0.5,
                replay_drain_seconds=0.0,
                replay_embedding_drain_seconds=0.0,
            )
            with (
                mock.patch(
                    "realtime.replay_validation.RealtimeCapture",
                    FakeCapture,
                ),
                mock.patch(
                    "realtime.replay_validation.load_audio_file",
                    return_value=(np.zeros(4, dtype=np.float32), 4),
                ),
                mock.patch(
                    "realtime.replay_validation.read_trace_records",
                    return_value=[],
                ),
                mock.patch(
                    "realtime.replay_validation.read_canonical_segments",
                    return_value=[],
                ),
                mock.patch(
                    "realtime.replay_validation.analyze_trace_against_canonical",
                    return_value=dict(analysis),
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(validate_cunk_realtime_replay(args), 0)
            written = json.loads(args.validation_output.read_text(encoding="utf-8"))

        capture = FakeCapture.latest
        self.assertEqual(capture.shutdown_count, 1)
        self.assertEqual(capture.feed.media_times, [0.5, 1.0, 1.5])
        self.assertEqual(capture.feed.assert_sample_rate, 4)
        self.assertEqual(
            capture.feed.statuses[-1],
            "Replay audio feed complete; draining final transcripts.",
        )
        self.assertEqual(capture.feed.finish_kwargs["emit_status"], False)
        self.assertEqual(written["match_mode"], "text")
        self.assertEqual(written["replay_speed"], 8.0)


if __name__ == "__main__":
    unittest.main()
