from __future__ import annotations

import argparse
import json
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
from window.window_diarizer import WindowDiarizer



from tests.window_diarizer_support import make_window_diarizer


class SpeakerProfilePersistenceTests(unittest.TestCase):
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
            source = make_window_diarizer()
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

            target = make_window_diarizer()
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
            diarizer = make_window_diarizer()
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
            diarizer = make_window_diarizer()
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


if __name__ == "__main__":
    unittest.main()
