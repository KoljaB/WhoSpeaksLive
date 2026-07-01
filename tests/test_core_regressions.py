from __future__ import annotations

import argparse
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from embedding_providers import EmbeddingSubprocessClient
from realtime_speaker_memory import SpeakerMemory as RealtimeSpeakerMemory
from speaker_embedding_cluster import SpeakerMemory as ClusterSpeakerMemory
from window_diarizer import WindowDiarizer
from window_gui_html import HTML
from window_preview import KrokoSubprocessPreviewTranscriber


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


class WindowHtmlSafetyTests(unittest.TestCase):
    def test_speaker_label_is_inserted_as_text_not_markup(self) -> None:
        self.assertNotIn("${speakerLabel}</span>", HTML)
        self.assertIn("speakerBadge.textContent = speakerLabel;", HTML)
        self.assertIn("row.replaceChildren(top, text);", HTML)

    def test_playback_clock_ignores_early_media_end_jumps(self) -> None:
        self.assertIn("playbackClockStartedAt", HTML)
        self.assertIn("playbackClockSlackSeconds", HTML)
        self.assertIn("Ignoring early audio ended event", HTML)


class WindowStreamingAudioTests(unittest.TestCase):
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
            realtime_preview_provider="cpu",
            realtime_preview_num_threads=2,
            realtime_preview_model_path=None,
            realtime_preview_download_root=None,
            download_root=None,
            realtime_preview_engine_options_json="",
            realtime_preview_realtimestt_root=None,
        )

        with mock.patch("window_preview.subprocess.Popen", return_value=FakeProcess()) as popen:
            transcriber = KrokoSubprocessPreviewTranscriber(args)
            transcriber.close()

        command = popen.call_args.args[0]
        self.assertIn(str(TOOLS / "kroko_realtime_preview_worker.py"), command)


if __name__ == "__main__":
    unittest.main()
