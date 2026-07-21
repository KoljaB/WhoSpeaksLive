from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np


try:
    from fastapi import HTTPException
except ModuleNotFoundError:  # The Windows test venv intentionally omits server-only deps.
    fastapi_stub = types.ModuleType("fastapi")
    responses_stub = types.ModuleType("fastapi.responses")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _FastAPI:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @staticmethod
        def _decorator(*_args: object, **_kwargs: object):
            return lambda function: function

        get = _decorator
        post = _decorator
        on_event = _decorator

    class _JSONResponse:
        def __init__(self, content: object) -> None:
            self.content = content

    fastapi_stub.FastAPI = _FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Query = lambda default=None, **_kwargs: default
    fastapi_stub.Request = object
    responses_stub.JSONResponse = _JSONResponse
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.responses"] = responses_stub

try:
    import av as _av  # noqa: F401
except ModuleNotFoundError:
    av_stub = types.ModuleType("av")

    class _FFmpegError(Exception):
        pass

    av_stub.FFmpegError = _FFmpegError
    av_stub.open = lambda *_args, **_kwargs: None
    av_stub.audio = types.SimpleNamespace(
        resampler=types.SimpleNamespace(AudioResampler=object),
    )
    sys.modules["av"] = av_stub


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "vendor" / "remote_servers" / "voice-embeddings-server" / "embeddings_server.py"
SPEC = importlib.util.spec_from_file_location("test_voice_embeddings_server_module", SERVER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"could not load {SERVER_PATH}")
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


def unit(values: list[float]) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    return value / np.linalg.norm(value)


class CoordinatorTestCase(unittest.TestCase):
    def install_coordinator(
        self,
        *,
        workers: int = 4,
        ttl: float = 3.0,
        entries: int = 128,
    ) -> object:
        coordinator = SERVER.EmbeddingCoordinator(workers, ttl, entries)
        previous = SERVER._component_coordinator
        SERVER._component_coordinator = coordinator
        self.addCleanup(coordinator.shutdown)
        self.addCleanup(setattr, SERVER, "_component_coordinator", previous)
        return coordinator

    def test_stack_single_request_coalesce_and_recent_cache(self) -> None:
        coordinator = self.install_coordinator(workers=4)
        audio = np.linspace(-0.25, 0.25, 640, dtype=np.float32)
        started = threading.Event()
        release = threading.Event()
        calls: dict[str, int] = {}
        calls_lock = threading.Lock()
        vectors = {
            "speechbrain_resnet": unit([1.0, 2.0, 3.0]),
            "resemblyzer": unit([4.0, 2.0]),
        }

        def fake(name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            with calls_lock:
                calls[name] = calls.get(name, 0) + 1
            if name == "speechbrain_resnet":
                started.set()
                self.assertTrue(release.wait(2.0))
            return vectors[name], device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            with ThreadPoolExecutor(max_workers=2) as pool:
                stack_future = pool.submit(
                    SERVER.embed_stack,
                    "speechbrain_resnet=0.25+resemblyzer=0.75",
                    "cpu",
                    audio,
                    16000,
                )
                self.assertTrue(started.wait(1.0))
                single_future = pool.submit(
                    SERVER.embed_stack,
                    "speechbrain_resnet",
                    "cpu",
                    audio.copy(),
                    16000,
                )
                time.sleep(0.03)
                release.set()
                _stacked, stack_components, _device = stack_future.result(timeout=2.0)
                single, single_components, _device = single_future.result(timeout=2.0)

            self.assertEqual(calls, {"speechbrain_resnet": 1, "resemblyzer": 1})
            self.assertEqual([item["provider"] for item in stack_components], [
                "speechbrain_resnet",
                "resemblyzer",
            ])
            self.assertEqual(single_components[0]["reuse"], "joined")
            np.testing.assert_allclose(single, vectors["speechbrain_resnet"], atol=1e-7)

            cached, cached_components, _device = SERVER.embed_stack(
                "speechbrain_resnet",
                "cpu",
                audio.copy(),
                16000,
            )
            self.assertEqual(cached_components[0]["reuse"], "cache")
            np.testing.assert_array_equal(cached, single)
            self.assertEqual(calls["speechbrain_resnet"], 1)

        counters = coordinator.snapshot()["counters"]
        self.assertEqual(counters["joined"], 1)
        self.assertEqual(counters["cache_hits"], 1)

    def test_parallel_completion_preserves_stack_order_weights_and_value(self) -> None:
        self.install_coordinator(workers=2, ttl=0.0)
        audio = np.linspace(-1.0, 1.0, 320, dtype=np.float32)
        speechbrain = unit([3.0, 1.0])
        resemblyzer = unit([2.0, 4.0, 1.0])

        def fake(name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            if name == "speechbrain_resnet":
                time.sleep(0.06)
                return speechbrain, device
            time.sleep(0.005)
            return resemblyzer, device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            actual, components, _device = SERVER.embed_stack(
                "speechbrain_resnet=0.2+resemblyzer=0.8",
                "cpu",
                audio,
                16000,
            )

        expected = SERVER.normalize_vector(np.concatenate([
            speechbrain * 0.2,
            resemblyzer * 0.8,
        ]))
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertEqual([item["provider"] for item in components], [
            "speechbrain_resnet",
            "resemblyzer",
        ])
        self.assertEqual([item["weight"] for item in components], [0.2, 0.8])
        self.assertTrue(all(item["reuse"] == "calculated" for item in components))
        self.assertTrue(all(item["compute_seconds"] >= 0.0 for item in components))
        self.assertTrue(all(item["queue_seconds"] >= 0.0 for item in components))
        self.assertTrue(all(item["wait_seconds"] >= 0.0 for item in components))

    def test_key_uses_canonical_provider_exact_audio_rate_and_count(self) -> None:
        coordinator = self.install_coordinator(workers=4)
        calls = 0
        calls_lock = threading.Lock()

        def fake(_name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            nonlocal calls
            with calls_lock:
                calls += 1
            return unit([1.0, 1.0]), device

        first = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float64)
        canonical, digest = SERVER.canonical_audio_fingerprint(first)
        same, same_digest = SERVER.canonical_audio_fingerprint(first.astype(np.float32))
        self.assertEqual(canonical.dtype, np.dtype("float32"))
        self.assertEqual(digest, same_digest)
        changed = same.copy()
        changed[1] = np.nextafter(changed[1], np.float32(1.0))
        self.assertNotEqual(digest, SERVER.canonical_audio_fingerprint(changed)[1])

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            alias = coordinator.submit("speechbrain_spkrec_resnet_voxceleb", "cpu", same, 16000)
            alias.future.result(timeout=1.0)
            canonical_name = coordinator.submit("speechbrain_resnet", "cpu", same.copy(), 16000)
            canonical_name.future.result(timeout=1.0)
            other_rate = coordinator.submit("speechbrain_resnet", "cpu", same.copy(), 8000)
            other_rate.future.result(timeout=1.0)
            other_count = coordinator.submit("speechbrain_resnet", "cpu", same[:-1], 16000)
            other_count.future.result(timeout=1.0)

        self.assertEqual(canonical_name.reuse, "cache")
        self.assertEqual(other_rate.reuse, "calculated")
        self.assertEqual(other_count.reuse, "calculated")
        self.assertEqual(calls, 3)

    def test_provider_serialization_with_cross_provider_parallelism_and_bound(self) -> None:
        # With two workers, a second SpeechBrain job must remain in the
        # provider queue instead of occupying the worker needed by Resemblyzer.
        coordinator = self.install_coordinator(workers=2, ttl=0.0)
        release = threading.Event()
        two_active = threading.Event()
        state_lock = threading.Lock()
        active_by_provider: dict[str, int] = {}
        max_by_provider: dict[str, int] = {}
        active_total = 0
        max_total = 0

        def fake(name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            nonlocal active_total, max_total
            with state_lock:
                active_by_provider[name] = active_by_provider.get(name, 0) + 1
                max_by_provider[name] = max(max_by_provider.get(name, 0), active_by_provider[name])
                active_total += 1
                max_total = max(max_total, active_total)
                if active_total >= 2:
                    two_active.set()
            self.assertTrue(release.wait(2.0))
            with state_lock:
                active_by_provider[name] -= 1
                active_total -= 1
            return unit([1.0, 2.0]), device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            first = coordinator.submit("speechbrain_resnet", "cpu", np.array([0.1], np.float32), 16000)
            second = coordinator.submit("speechbrain_resnet", "cpu", np.array([0.2], np.float32), 16000)
            other = coordinator.submit("resemblyzer", "cpu", np.array([0.3], np.float32), 16000)
            self.assertTrue(two_active.wait(1.0))
            release.set()
            for handle in (first, second, other):
                handle.future.result(timeout=2.0)

        self.assertEqual(max_by_provider["speechbrain_resnet"], 1)
        self.assertGreaterEqual(max_total, 2)
        self.assertLessEqual(max_total, 2)
        self.assertLessEqual(coordinator.snapshot()["peak_active"], 2)

    def test_failure_is_shared_not_cached_and_retry_recomputes(self) -> None:
        coordinator = self.install_coordinator(workers=2)
        audio = np.array([0.1, 0.2], dtype=np.float32)
        started = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def fake(_name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            nonlocal calls
            with calls_lock:
                calls += 1
                attempt = calls
            if attempt == 1:
                started.set()
                self.assertTrue(release.wait(2.0))
                raise HTTPException(status_code=500, detail="synthetic_failure")
            return unit([2.0, 1.0]), device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            first = coordinator.submit("speechbrain_resnet", "cpu", audio, 16000)
            self.assertTrue(started.wait(1.0))
            joined = coordinator.submit("speechbrain_resnet", "cpu", audio.copy(), 16000)
            self.assertEqual(joined.reuse, "joined")
            release.set()
            for handle in (first, joined):
                with self.assertRaises(HTTPException):
                    handle.future.result(timeout=1.0)
            retry = coordinator.submit("speechbrain_resnet", "cpu", audio.copy(), 16000)
            retry.future.result(timeout=1.0)

        self.assertEqual(retry.reuse, "calculated")
        self.assertEqual(calls, 2)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot["inflight"], 0)
        self.assertEqual(snapshot["counters"]["failures"], 1)

    def test_invalidation_detaches_old_job_and_blocks_stale_cache_fill(self) -> None:
        coordinator = self.install_coordinator(workers=2)
        audio = np.array([0.5, -0.5], dtype=np.float32)
        first_started = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def fake(_name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            nonlocal calls
            with calls_lock:
                calls += 1
                attempt = calls
            if attempt == 1:
                first_started.set()
                self.assertTrue(release.wait(2.0))
                return unit([1.0, 0.0]), device
            return unit([0.0, 1.0]), device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            old = coordinator.submit("speechbrain_resnet", "cpu", audio, 16000)
            self.assertTrue(first_started.wait(1.0))
            invalidated = coordinator.invalidate({("speechbrain_resnet", "cpu")})
            self.assertEqual(invalidated["inflight"], 1)
            fresh = coordinator.submit("speechbrain_resnet", "cpu", audio.copy(), 16000)
            self.assertEqual(fresh.reuse, "calculated")
            release.set()
            old.future.result(timeout=2.0)
            fresh_result = fresh.future.result(timeout=2.0)
            cached = coordinator.submit("speechbrain_resnet", "cpu", audio.copy(), 16000)
            cached_result = cached.future.result(timeout=1.0)

        self.assertEqual(calls, 2)
        self.assertEqual(cached.reuse, "cache")
        np.testing.assert_array_equal(cached_result.embedding, fresh_result.embedding)
        np.testing.assert_allclose(cached_result.embedding, unit([0.0, 1.0]))

    def test_unload_blocks_stale_model_loader_from_repopulating_provider_cache(self) -> None:
        coordinator = self.install_coordinator(workers=1)
        audio = np.array([0.25, -0.25], dtype=np.float32)
        load_started = threading.Event()
        release_load = threading.Event()

        class FakeProvider:
            device = "cpu"

            @staticmethod
            def embed(_audio: np.ndarray, _sample_rate: int) -> np.ndarray:
                return unit([1.0, 2.0])

        def fake_create(_name: str, _device: str) -> FakeProvider:
            load_started.set()
            self.assertTrue(release_load.wait(2.0))
            return FakeProvider()

        key = SERVER.provider_key("speechbrain_resnet", "cpu")
        SERVER.unload("speechbrain_resnet", "cpu")
        self.addCleanup(SERVER.unload, "speechbrain_resnet", "cpu")

        with mock.patch.object(SERVER, "create_single_provider", side_effect=fake_create):
            handle = coordinator.submit("speechbrain_resnet", "cpu", audio, 16000)
            self.assertTrue(load_started.wait(1.0))
            invalidated = SERVER.unload("speechbrain_resnet", "cpu")
            self.assertEqual(invalidated["invalidated"]["inflight"], 1)
            release_load.set()
            handle.future.result(timeout=2.0)

        with SERVER._cache_lock:
            self.assertNotIn(key, SERVER._provider_cache)

    def test_unload_epoch_blocks_overlapping_submit_from_filling_result_cache(self) -> None:
        coordinator = self.install_coordinator(workers=1)
        audio = np.array([0.125, -0.125], dtype=np.float32)
        generation_captured = threading.Event()
        release_submit = threading.Event()
        first_inference_started = threading.Event()
        release_first_inference = threading.Event()
        calls = 0
        original_generation = SERVER.provider_cache_generation

        def paused_generation(name: str, device: str) -> int:
            generation = original_generation(name, device)
            generation_captured.set()
            self.assertTrue(release_submit.wait(2.0))
            return generation

        def fake(_name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_inference_started.set()
                self.assertTrue(release_first_inference.wait(2.0))
            return unit([1.0, 3.0]), device

        SERVER.unload("speechbrain_resnet", "cpu")
        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            with mock.patch.object(SERVER, "provider_cache_generation", side_effect=paused_generation):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    submit_future = pool.submit(
                        coordinator.submit,
                        "speechbrain_resnet",
                        "cpu",
                        audio,
                        16000,
                    )
                    self.assertTrue(generation_captured.wait(1.0))
                    SERVER.unload("speechbrain_resnet", "cpu")
                    release_submit.set()
                    first = submit_future.result(timeout=1.0)
            self.assertTrue(first_inference_started.wait(1.0))
            second = coordinator.submit("speechbrain_resnet", "cpu", audio.copy(), 16000)
            self.assertEqual(second.reuse, "calculated")
            release_first_inference.set()
            first.future.result(timeout=1.0)
            second.future.result(timeout=1.0)

        self.assertEqual(first.reuse, "calculated")
        self.assertEqual(calls, 2)


class CoordinatorAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_stack_wait_does_not_block_event_loop(self) -> None:
        coordinator = SERVER.EmbeddingCoordinator(2, 0.0, 0)
        previous = SERVER._component_coordinator
        SERVER._component_coordinator = coordinator
        self.addAsyncCleanup(asyncio.to_thread, coordinator.shutdown)
        self.addCleanup(setattr, SERVER, "_component_coordinator", previous)
        started = threading.Event()
        release = threading.Event()

        def fake(_name: str, device: str, _audio: np.ndarray, _sample_rate: int) -> tuple[np.ndarray, str]:
            started.set()
            self.assertTrue(release.wait(2.0))
            return unit([1.0, 2.0]), device

        with mock.patch.object(SERVER, "_embed_provider_vector_unlocked", side_effect=fake):
            timer = threading.Timer(0.25, release.set)
            timer.start()
            before = time.perf_counter()
            task = asyncio.create_task(SERVER.embed_stack_async(
                "speechbrain_resnet",
                "cpu",
                np.array([0.1, 0.2], dtype=np.float32),
                16000,
            ))
            self.assertTrue(await asyncio.to_thread(started.wait, 1.0))
            await asyncio.sleep(0.02)
            event_loop_delay = time.perf_counter() - before
            self.assertLess(event_loop_delay, 0.15)
            self.assertFalse(task.done())
            await asyncio.wait_for(task, timeout=1.0)
            timer.cancel()


if __name__ == "__main__":
    unittest.main()
