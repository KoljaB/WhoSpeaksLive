from __future__ import annotations

import argparse
import io
import os
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

from window.window_domain import TimedWord, VadWindowState
from window.window_events import RecordingEventBus
from window.window_preview import KrokoSubprocessPreviewTranscriber



from tests.window_diarizer_support import make_window_diarizer


class WindowAudioAsrTests(unittest.TestCase):
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

    def test_asr_no_speech_filter_drops_high_no_speech_prob_words(self) -> None:
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

        self.assertEqual([word.text for word in kept], [" Hallo", " Ja.", " unknown"])
        self.assertTrue(any("ASR no-speech filter dropped 3 word" in item["payload"]["message"] for item in diarizer.bus.records))

    def test_asr_no_speech_filter_drops_short_segments_above_hard_threshold(self) -> None:
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

        self.assertEqual([word.text for word in kept], [" Hallo"])

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
