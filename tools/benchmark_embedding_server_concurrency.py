"""Benchmark realistic final/live embedding traffic against the remote server.

The benchmark deliberately calls ``/embed-pcm16`` directly so it can retain the
server's own elapsed/component timings as well as measuring Windows-to-server
wall time.  It uses deterministic, high-energy slices from a real media file
and never unloads models or changes server state beyond the normal ``/load``
warmup calls.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "vendor"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common.audio_utils import load_audio_file, normalize_vector
from embeddings.provider_identity import PROMOTED_PUBLIC_PROVIDER, PUBLIC_PROVIDER


SAMPLE_RATE = 16_000
DEFAULT_AUDIO = ROOT / "runtime" / "media" / "local-filefeed" / "JWS-qfR6K3w.audio.mp3"
INDIVIDUAL_PROVIDERS = (
    "espnet_ecapa_wavlm_joint",
    "wespeaker_campplus",
    "resemblyzer",
    "speechbrain_resnet",
)
STACKS = {
    "promoted_public": PROMOTED_PUBLIC_PROVIDER,
    "public_quality": PUBLIC_PROVIDER,
}
REUSE_STATES = {"calculated", "joined", "cache"}


def _reuse_diagnostics(value: Any, path: str = "response") -> list[dict[str, str]]:
    """Collect optional server work-reuse states without requiring a new schema."""

    diagnostics: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "embedding":
                continue
            diagnostics.extend(_reuse_diagnostics(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            diagnostics.extend(_reuse_diagnostics(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        state = value.strip().lower()
        if state == "cached":
            state = "cache"
        if state in REUSE_STATES:
            diagnostics.append({"path": path, "state": state})
    return diagnostics


def _component_embeddings(result: dict[str, Any], embedding: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct normalized component vectors from a concatenated stack vector."""

    canonical = result.get("canonical_stack")
    components = result.get("components")
    if not isinstance(canonical, list) or not isinstance(components, list):
        return {}
    dims_by_provider: dict[str, list[int]] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        provider = str(component.get("provider") or "")
        try:
            dim = int(component.get("dim") or 0)
        except (TypeError, ValueError):
            dim = 0
        if provider and dim > 0:
            dims_by_provider.setdefault(provider, []).append(dim)
    offset = 0
    reconstructed: dict[str, np.ndarray] = {}
    for spec in canonical:
        if not isinstance(spec, dict):
            return {}
        provider = str(spec.get("provider") or "")
        provider_dims = dims_by_provider.get(provider) or []
        if not provider_dims:
            return {}
        dim = provider_dims.pop(0)
        right = offset + dim
        if right > embedding.size:
            return {}
        try:
            reconstructed[provider] = normalize_vector(embedding[offset:right])
        except ValueError:
            return {}
        offset = right
    return reconstructed if offset == embedding.size else {}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    clean = [number for value in values if (number := _finite_float(value)) is not None]
    if not clean:
        return {"n": 0}
    array = np.asarray(clean, dtype=np.float64)
    return {
        "n": len(clean),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _round_or_none(value: Any, digits: int = 4) -> str:
    number = _finite_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    prepared = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(prepared, -1.0, 1.0) * 32767.0).astype("<i2")
    return np.ascontiguousarray(pcm16).tobytes()


def _post_json(
    endpoint: str,
    path: str,
    query: dict[str, Any],
    payload: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}?{urlencode(query)}"
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON from {path}: {raw[:500]!r}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object from {path}, received {type(result).__name__}")
    if result.get("error") or result.get("detail"):
        raise RuntimeError(str(result.get("error") or result.get("detail")))
    return result


def _get_json(endpoint: str, path: str, timeout_seconds: float) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            raw = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Cannot read {url}: {exc}") from exc
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object from {path}, received {type(result).__name__}")
    return result


class JsonTransport:
    """Small transport wrapper supporting a baseline and persistent HTTP client."""

    def __init__(self, endpoint: str, kind: str, timeout_seconds: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.kind = kind
        self.timeout_seconds = timeout_seconds
        self._client: Any = None
        self._closed = False
        if kind == "httpx":
            try:
                import httpx
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "--http-client httpx requires the 'httpx' package in this Python environment"
                ) from exc
            # httpx.Client is designed to share its connection pool across
            # threads, which lets concurrent benchmark requests reuse sockets.
            self._client = httpx.Client(
                base_url=self.endpoint,
                timeout=timeout_seconds,
                headers={"Content-Type": "application/octet-stream"},
            )
            atexit.register(self.close)

    def get_json(self, path: str) -> dict[str, Any]:
        if self._client is None:
            return _get_json(self.endpoint, path, min(10.0, self.timeout_seconds))
        response = self._client.get(path, timeout=min(10.0, self.timeout_seconds))
        try:
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"HTTPX GET {path} failed ({response.status_code}): {response.text[:1000]}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected JSON object from {path}, received {type(result).__name__}")
        return result

    def post_json(self, path: str, query: dict[str, Any], payload: bytes) -> dict[str, Any]:
        if self._client is None:
            return _post_json(
                self.endpoint,
                path,
                query,
                payload,
                self.timeout_seconds,
            )
        response = self._client.post(path, params=query, content=payload)
        try:
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"HTTPX POST {path} failed ({response.status_code}): {response.text[:1000]}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected JSON object from {path}, received {type(result).__name__}")
        if result.get("error") or result.get("detail"):
            raise RuntimeError(str(result.get("error") or result.get("detail")))
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            self._client.close()


def _slice_audio(audio: np.ndarray, right_sample: int, seconds: float) -> np.ndarray:
    sample_count = max(1, int(round(seconds * SAMPLE_RATE)))
    left = max(0, right_sample - sample_count)
    result = np.asarray(audio[left:right_sample], dtype=np.float32)
    if result.size < sample_count:
        result = np.pad(result, (sample_count - result.size, 0))
    return result


def _window_activity(audio: np.ndarray) -> tuple[float, float, float]:
    if audio.size <= 0:
        return 0.0, 0.0, 0.0
    absolute = np.abs(np.asarray(audio, dtype=np.float32))
    rms = float(np.sqrt(np.mean(np.square(absolute, dtype=np.float64))))
    active_ratio = float(np.mean(absolute >= 0.012))
    peak = float(np.max(absolute))
    return rms, active_ratio, peak


def _choose_speech_anchors(
    audio: np.ndarray,
    count: int,
    sentence_seconds: float,
    short_seconds: float,
    long_seconds: float,
) -> list[dict[str, Any]]:
    """Pick deterministic high-energy endpoints while keeping selections apart."""

    duration = len(audio) / float(SAMPLE_RATE)
    maximum_window = max(sentence_seconds, short_seconds, long_seconds)
    start = maximum_window + min(5.0, max(0.0, duration * 0.03))
    stop = duration - min(5.0, max(0.0, duration * 0.03))
    if stop <= start:
        start, stop = maximum_window, duration
    if stop <= start:
        raise RuntimeError(
            f"Audio is only {duration:.2f}s, shorter than the {maximum_window:.2f}s benchmark window"
        )

    # A 250 ms grid is fine enough to find voiced material while remaining cheap
    # even for hour-long recordings.  The score rewards sustained activity in all
    # three windows, not merely one loud transient.
    candidates: list[dict[str, Any]] = []
    for right_seconds in np.arange(start, stop + 1e-9, 0.25):
        right_sample = min(len(audio), int(round(float(right_seconds) * SAMPLE_RATE)))
        metrics = []
        for seconds in (short_seconds, long_seconds, sentence_seconds):
            metrics.append(_window_activity(_slice_audio(audio, right_sample, seconds)))
        rms_values = [item[0] for item in metrics]
        active_values = [item[1] for item in metrics]
        # Prefer continuous speech-like energy and penalize clipping.  This is a
        # deterministic heuristic, not a VAD or a claim about ground-truth speech.
        clipping_penalty = max(0.0, max(item[2] for item in metrics) - 0.985) * 2.0
        score = min(rms_values) * (0.35 + 0.65 * min(active_values)) - clipping_penalty
        candidates.append({
            "right_seconds": float(right_sample) / SAMPLE_RATE,
            "right_sample": right_sample,
            "selection_score": float(score),
            "sentence_rms": metrics[2][0],
            "sentence_active_ratio": metrics[2][1],
        })

    selected: list[dict[str, Any]] = []
    minimum_gap = max(2.0, sentence_seconds + 0.5)
    for candidate in sorted(candidates, key=lambda item: (-item["selection_score"], item["right_seconds"])):
        if all(abs(candidate["right_seconds"] - item["right_seconds"]) >= minimum_gap for item in selected):
            selected.append(candidate)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Could select only {len(selected)} separated benchmark slices, requested {count}")
    return sorted(selected, key=lambda item: item["right_seconds"])


class BenchmarkRunner:
    def __init__(
        self,
        endpoint: str,
        device: str,
        timeout_seconds: float,
        slices: dict[str, np.ndarray],
        http_client: str = "urllib",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.slices = slices
        self.transport = JsonTransport(self.endpoint, http_client, timeout_seconds)
        self.http_client = http_client
        self.records: list[dict[str, Any]] = []
        self.cycles: list[dict[str, Any]] = []

    def warm(self, provider: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self.transport.post_json(
                "/load",
                {"provider": provider, "device": self.device},
                b"",
            )
            return {
                "provider": provider,
                "wall_seconds": time.perf_counter() - started,
                "server_elapsed_seconds": result.get("elapsed_seconds"),
                "resolved_device": result.get("resolved_device"),
                "warmups": result.get("warmups", []),
                "ok": True,
            }
        except Exception as exc:
            return {
                "provider": provider,
                "wall_seconds": time.perf_counter() - started,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def request(
        self,
        benchmark: str,
        role: str,
        provider: str,
        slice_id: str,
        repetition: int,
        mode: str,
    ) -> dict[str, Any]:
        audio = self.slices[slice_id]
        started = time.perf_counter()
        record: dict[str, Any] = {
            "benchmark": benchmark,
            "mode": mode,
            "role": role,
            "provider": provider,
            "slice_id": slice_id,
            "duration_seconds": len(audio) / float(SAMPLE_RATE),
            "repetition": repetition,
        }
        try:
            result = self.transport.post_json(
                "/embed-pcm16",
                {
                    "provider": provider,
                    "device": self.device,
                    "sample_rate": SAMPLE_RATE,
                    "encoding": "pcm16",
                },
                _pcm16_bytes(audio),
            )
            wall = time.perf_counter() - started
            embedding = normalize_vector(result.get("embedding"))
            component_embeddings = _component_embeddings(result, embedding)
            record.update({
                "ok": True,
                "wall_seconds": wall,
                "server_elapsed_seconds": result.get("elapsed_seconds"),
                "decode_seconds": result.get("decode_seconds"),
                "server_device": result.get("device"),
                "embedding_dim": int(embedding.size),
                "embedding_norm": float(np.linalg.norm(embedding)),
                "embedding_sha256": hashlib.sha256(
                    np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
                ).hexdigest(),
                "canonical_stack": result.get("canonical_stack", []),
                "components": result.get("components", []),
                "reuse_diagnostics": _reuse_diagnostics(result),
                "component_embedding_hashes": {
                    provider_name: hashlib.sha256(
                        np.ascontiguousarray(vector, dtype=np.float32).tobytes()
                    ).hexdigest()
                    for provider_name, vector in component_embeddings.items()
                },
                "_embedding": embedding,
                "_component_embeddings": component_embeddings,
            })
        except Exception as exc:
            record.update({
                "ok": False,
                "wall_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
        self.records.append(record)
        status = "ok" if record["ok"] else "ERROR"
        print(
            f"  {benchmark:<38} {role:<22} rep={repetition + 1:<2} "
            f"wall={record['wall_seconds']:.4f}s server={_round_or_none(record.get('server_elapsed_seconds'))}s {status}",
            flush=True,
        )
        return record

    def cycle(
        self,
        benchmark: str,
        mode: str,
        tasks: list[dict[str, str]],
        repetition: int,
        release_together: bool = False,
    ) -> None:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        if mode == "concurrent" or release_together:
            barrier = threading.Barrier(len(tasks)) if release_together and len(tasks) > 1 else None

            def run_task(task: dict[str, str]) -> dict[str, Any]:
                if barrier is not None:
                    barrier.wait(timeout=min(10.0, self.timeout_seconds))
                return self.request(
                    benchmark,
                    task["role"],
                    task["provider"],
                    task["slice_id"],
                    repetition,
                    mode,
                )

            with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="embedding-bench") as pool:
                futures = [pool.submit(run_task, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for task in tasks:
                results.append(self.request(
                    benchmark,
                    task["role"],
                    task["provider"],
                    task["slice_id"],
                    repetition,
                    mode,
                ))
        self.cycles.append({
            "benchmark": benchmark,
            "mode": mode,
            "repetition": repetition,
            "wall_seconds": time.perf_counter() - started,
            "request_count": len(tasks),
            "ok": all(result.get("ok") for result in results),
        })

    def followup_cycle(
        self,
        benchmark: str,
        mode: str,
        stack_provider: str,
        slice_id: str,
        repetition: int,
        delay_seconds: float,
    ) -> None:
        """Finish a stack request, then issue the matching component request."""

        started = time.perf_counter()
        stack = self.request(
            benchmark,
            "final_stack",
            stack_provider,
            slice_id,
            repetition,
            mode,
        )
        if delay_seconds > 0.0:
            time.sleep(delay_seconds)
        standalone = self.request(
            benchmark,
            "standalone_speechbrain",
            "speechbrain_resnet",
            slice_id,
            repetition,
            mode,
        )
        self.cycles.append({
            "benchmark": benchmark,
            "mode": mode,
            "repetition": repetition,
            "wall_seconds": time.perf_counter() - started,
            "request_count": 2,
            "followup_delay_seconds": delay_seconds,
            "ok": bool(stack.get("ok") and standalone.get("ok")),
        })


def _component_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, list[float]] = {}
    for record in records:
        for component in record.get("components") or []:
            if not isinstance(component, dict):
                continue
            provider = str(component.get("provider") or "unknown")
            elapsed = _finite_float(component.get("elapsed_seconds"))
            if elapsed is not None:
                by_provider.setdefault(provider, []).append(elapsed)
    return {provider: _distribution(values) for provider, values in sorted(by_provider.items())}


def _reuse_state_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {state: 0 for state in sorted(REUSE_STATES)}
    for record in records:
        states = {
            str(item.get("state"))
            for item in record.get("reuse_diagnostics") or []
            if isinstance(item, dict)
        }
        for state in states:
            if state in counts:
                counts[state] += 1
    return counts


def _reuse_event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {state: 0 for state in sorted(REUSE_STATES)}
    for record in records:
        for item in record.get("reuse_diagnostics") or []:
            if not isinstance(item, dict):
                continue
            state = str(item.get("state"))
            if state in counts:
                counts[state] += 1
    return counts


def _summarize(runner: BenchmarkRunner) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    keys = sorted({(str(item["benchmark"]), str(item["mode"])) for item in runner.records})
    for benchmark, mode in keys:
        records = [
            item for item in runner.records
            if item["benchmark"] == benchmark and item["mode"] == mode
        ]
        cycles = [
            item for item in runner.cycles
            if item["benchmark"] == benchmark and item["mode"] == mode
        ]
        good = [item for item in records if item.get("ok")]
        summaries[f"{benchmark}:{mode}"] = {
            "benchmark": benchmark,
            "mode": mode,
            "requests": len(records),
            "errors": len(records) - len(good),
            "request_wall_seconds": _distribution(item.get("wall_seconds") for item in good),
            "server_elapsed_seconds": _distribution(item.get("server_elapsed_seconds") for item in good),
            "server_decode_seconds": _distribution(item.get("decode_seconds") for item in good),
            "cycle_wall_seconds": _distribution(item.get("wall_seconds") for item in cycles if item.get("ok")),
            "component_elapsed_seconds": _component_summary(good),
            "reuse_state_response_counts": _reuse_state_counts(good),
            "reuse_diagnostic_event_counts": _reuse_event_counts(good),
            "error_messages": sorted({str(item.get("error")) for item in records if not item.get("ok")}),
        }

    comparisons: dict[str, Any] = {}
    for benchmark in sorted({str(item["benchmark"]) for item in runner.cycles}):
        sequential = summaries.get(f"{benchmark}:sequential", {}).get("cycle_wall_seconds", {})
        concurrent = summaries.get(f"{benchmark}:concurrent", {}).get("cycle_wall_seconds", {})
        seq_p50 = _finite_float(sequential.get("p50"))
        con_p50 = _finite_float(concurrent.get("p50"))
        if seq_p50 is not None and con_p50 is not None and con_p50 > 0.0:
            comparisons[benchmark] = {
                "sequential_cycle_p50_seconds": seq_p50,
                "concurrent_cycle_p50_seconds": con_p50,
                "p50_speedup": seq_p50 / con_p50,
                "p50_seconds_saved": seq_p50 - con_p50,
            }
    return summaries, comparisons


def _coalesce_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    focused = [item for item in records if str(item.get("benchmark", "")).startswith("coalesce:")]
    if not focused:
        return {}

    parity_rows: list[dict[str, Any]] = []
    cycle_groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for record in focused:
        key = (
            str(record["benchmark"]),
            str(record["mode"]),
            int(record["repetition"]),
            str(record["slice_id"]),
        )
        cycle_groups.setdefault(key, []).append(record)
    for (benchmark, mode, repetition, slice_id), items in sorted(cycle_groups.items()):
        stack = next((item for item in items if item.get("role") == "final_stack" and item.get("ok")), None)
        standalone = next(
            (item for item in items if item.get("role") == "standalone_speechbrain" and item.get("ok")),
            None,
        )
        if stack is None or standalone is None:
            continue
        component_vectors = stack.get("_component_embeddings") or {}
        reconstructed = component_vectors.get("speechbrain_resnet")
        standalone_vector = standalone.get("_embedding")
        if not isinstance(reconstructed, np.ndarray) or not isinstance(standalone_vector, np.ndarray):
            continue
        if reconstructed.shape != standalone_vector.shape:
            cosine = None
            maximum_delta = None
        else:
            cosine = float(np.dot(reconstructed, standalone_vector))
            maximum_delta = float(np.max(np.abs(reconstructed - standalone_vector)))
        parity_rows.append({
            "benchmark": benchmark,
            "mode": mode,
            "repetition": repetition,
            "slice_id": slice_id,
            "dim": int(standalone_vector.size),
            "cosine_stack_slice_vs_standalone": cosine,
            "max_absolute_delta": maximum_delta,
            "stack_reuse_diagnostics": stack.get("reuse_diagnostics", []),
            "standalone_reuse_diagnostics": standalone.get("reuse_diagnostics", []),
        })

    calculated_reference = [
        item for item in focused
        if item.get("role") == "standalone_speechbrain"
        and item.get("ok")
        and any(
            diagnostic.get("state") == "calculated"
            for diagnostic in item.get("reuse_diagnostics") or []
            if isinstance(diagnostic, dict)
        )
    ]
    if not calculated_reference:
        calculated_reference = [
            item for item in focused
            if item.get("benchmark") == "coalesce:control"
            and item.get("role") == "standalone_speechbrain"
            and item.get("ok")
        ]
    if not calculated_reference:
        calculated_reference = [
            item for item in records
            if item.get("benchmark") == "individual:speechbrain_resnet" and item.get("ok")
        ]
    reference = _distribution(item.get("wall_seconds") for item in calculated_reference)
    modes: dict[str, Any] = {}
    for benchmark, mode in sorted({(str(item["benchmark"]), str(item["mode"])) for item in focused}):
        items = [
            item for item in focused
            if item["benchmark"] == benchmark and item["mode"] == mode
        ]
        standalone = [
            item for item in items
            if item.get("role") == "standalone_speechbrain" and item.get("ok")
        ]
        distribution = _distribution(item.get("wall_seconds") for item in standalone)
        mode_result: dict[str, Any] = {
            "standalone_wall_seconds": distribution,
            "reuse_state_response_counts": _reuse_state_counts(standalone),
            "reuse_diagnostic_event_counts": _reuse_event_counts(standalone),
        }
        reference_p50 = _finite_float(reference.get("p50"))
        mode_p50 = _finite_float(distribution.get("p50"))
        if reference_p50 is not None and mode_p50 is not None:
            mode_result["standalone_p50_seconds_saved_vs_calculated"] = reference_p50 - mode_p50
            if mode_p50 > 0.0:
                mode_result["standalone_p50_speedup_vs_calculated"] = reference_p50 / mode_p50
        modes[f"{benchmark}:{mode}"] = mode_result
    return {
        "calculated_standalone_reference_wall_seconds": reference,
        "modes": modes,
        "stack_speechbrain_vector_parity": parity_rows,
        "parity_cosine_distribution": _distribution(
            item.get("cosine_stack_slice_vs_standalone") for item in parity_rows
        ),
        "parity_max_absolute_delta_distribution": _distribution(
            item.get("max_absolute_delta") for item in parity_rows
        ),
    }


def _parity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Same provider + same exact PCM slice is expected to yield the same vector,
    # regardless of request order or whether the client issued it concurrently.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("ok") and isinstance(record.get("_embedding"), np.ndarray):
            groups.setdefault((str(record["provider"]), str(record["slice_id"])), []).append(record)
    output: list[dict[str, Any]] = []
    for (provider, slice_id), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        reference = items[0]["_embedding"]
        cosines: list[float] = []
        max_abs_deltas: list[float] = []
        incompatible = 0
        for item in items[1:]:
            vector = item["_embedding"]
            if vector.shape != reference.shape:
                incompatible += 1
                continue
            cosines.append(float(np.dot(reference, vector)))
            max_abs_deltas.append(float(np.max(np.abs(reference - vector))))
        output.append({
            "provider": provider,
            "slice_id": slice_id,
            "sample_count": len(items),
            "incompatible_shapes": incompatible,
            "cosine_to_first": _distribution(cosines),
            "max_absolute_delta_to_first": _distribution(max_abs_deltas),
            "unique_embedding_hashes": len({str(item["embedding_sha256"]) for item in items}),
        })
    return output


def _clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if not key.startswith("_")} for record in records]


def _parse_sections(value: str) -> set[str]:
    sections = {item.strip().lower() for item in value.split(",") if item.strip()}
    valid = {"individual", "stacks", "live", "providers", "coalesce"}
    unknown = sections - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown section(s): {', '.join(sorted(unknown))}")
    return sections


def _print_summary(summaries: dict[str, Any], comparisons: dict[str, Any]) -> None:
    print("\nSummary (seconds)")
    print(f"{'benchmark:mode':<50} {'n':>4} {'err':>4} {'req p50':>9} {'req p95':>9} {'cycle p50':>10}")
    for key, item in summaries.items():
        request = item["request_wall_seconds"]
        cycle = item["cycle_wall_seconds"]
        print(
            f"{key:<50} {item['requests']:>4} {item['errors']:>4} "
            f"{_round_or_none(request.get('p50')):>9} {_round_or_none(request.get('p95')):>9} "
            f"{_round_or_none(cycle.get('p50')):>10}"
        )
    if comparisons:
        print("\nSequential versus concurrent")
        for benchmark, comparison in comparisons.items():
            print(
                f"  {benchmark}: {comparison['p50_speedup']:.3f}x; "
                f"{comparison['p50_seconds_saved']:+.4f}s saved per cycle"
            )


def _print_coalesce_analysis(result: dict[str, Any]) -> None:
    if not result:
        return
    reference = result.get("calculated_standalone_reference_wall_seconds") or {}
    print(
        "\nCoalescing diagnostics "
        f"(calculated standalone p50={_round_or_none(reference.get('p50'))}s)"
    )
    for name, item in (result.get("modes") or {}).items():
        distribution = item.get("standalone_wall_seconds") or {}
        counts = item.get("reuse_diagnostic_event_counts") or {}
        saving = item.get("standalone_p50_seconds_saved_vs_calculated")
        print(
            f"  {name}: standalone p50={_round_or_none(distribution.get('p50'))}s "
            f"saved={_round_or_none(saving)}s reuse="
            f"calculated:{counts.get('calculated', 0)} joined:{counts.get('joined', 0)} "
            f"cache:{counts.get('cache', 0)}"
        )
    cosine = result.get("parity_cosine_distribution") or {}
    print(
        "  reconstructed SpeechBrain parity: "
        f"n={cosine.get('n', 0)} min_cosine={_round_or_none(cosine.get('min'), 8)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://192.168.178.22:8660", help="Remote embeddings base URL")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Real audio file used for deterministic slices")
    parser.add_argument("--device", default="auto", help="Device query passed exactly as the live client would")
    parser.add_argument("--repetitions", type=int, default=2, help="Repeat every selected slice this many times")
    parser.add_argument("--slice-count", type=int, default=1, help="Number of separated high-energy source locations")
    parser.add_argument("--sentence-seconds", type=float, default=3.0, help="Final sentence request length")
    parser.add_argument("--short-window-seconds", type=float, default=0.7)
    parser.add_argument("--long-window-seconds", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--http-client",
        choices=("urllib", "httpx"),
        default="urllib",
        help="urllib preserves the original baseline; httpx reuses persistent connections",
    )
    parser.add_argument(
        "--scenario-stack",
        choices=("promoted_public", "public_quality", "both"),
        default="promoted_public",
        help="Final stack(s) used in the simultaneous final + live scenario",
    )
    parser.add_argument(
        "--sections",
        type=_parse_sections,
        default=_parse_sections("individual,stacks,live,providers"),
        help="Comma-separated: individual,stacks,live,providers,coalesce",
    )
    parser.add_argument(
        "--coalesce-stack",
        choices=("promoted_public", "public_quality"),
        default="promoted_public",
        help="Final stack used by the focused duplicate-work/coalescing scenario",
    )
    parser.add_argument(
        "--coalesce-followup-delay-seconds",
        type=float,
        default=0.05,
        help="Additional delayed follow-up after the always-tested immediate follow-up",
    )
    parser.add_argument("--skip-warmup", action="store_true", help="Do not call /load for the four component providers")
    parser.add_argument("--output-json", type=Path, help="Optional detailed machine-readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repetitions < 1 or args.slice_count < 1:
        raise SystemExit("--repetitions and --slice-count must be at least 1")
    for label, value in (
        ("sentence", args.sentence_seconds),
        ("short window", args.short_window_seconds),
        ("long window", args.long_window_seconds),
        ("timeout", args.timeout_seconds),
    ):
        if value <= 0.0:
            raise SystemExit(f"{label} must be positive")
    if args.coalesce_followup_delay_seconds < 0.0:
        raise SystemExit("--coalesce-followup-delay-seconds must not be negative")

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")
    print(f"Reading {audio_path}", flush=True)
    audio, sample_rate = load_audio_file(audio_path, SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"Audio loader returned {sample_rate} Hz instead of {SAMPLE_RATE} Hz")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    anchors = _choose_speech_anchors(
        audio,
        args.slice_count,
        args.sentence_seconds,
        args.short_window_seconds,
        args.long_window_seconds,
    )
    slices: dict[str, np.ndarray] = {}
    for index, anchor in enumerate(anchors):
        anchor_id = f"anchor_{index + 1}"
        for label, seconds in (
            ("sentence", args.sentence_seconds),
            ("live_short", args.short_window_seconds),
            ("live_long", args.long_window_seconds),
        ):
            slices[f"{anchor_id}:{label}"] = _slice_audio(audio, anchor["right_sample"], seconds)
        print(
            f"Selected {anchor_id} ending at {anchor['right_seconds']:.2f}s "
            f"(RMS={anchor['sentence_rms']:.4f}, active={anchor['sentence_active_ratio']:.1%})",
            flush=True,
        )

    # Coalescing scenarios use distinct, nearby real-audio sentences so a
    # previous scenario's short result cache cannot contaminate the next one.
    # Within every scenario pair, however, stack and standalone requests use
    # byte-for-byte identical PCM. Each repetition gets a separate 10 ms shift.
    if "coalesce" in args.sections:
        scenario_labels = ("control", "concurrent", "followup_immediate", "followup_delayed")
        for anchor_index, anchor in enumerate(anchors):
            anchor_id = f"anchor_{anchor_index + 1}"
            for repetition in range(args.repetitions):
                for scenario_index, scenario_label in enumerate(scenario_labels):
                    variant_index = repetition * len(scenario_labels) + scenario_index + 1
                    shifted_right = max(
                        int(round(args.sentence_seconds * SAMPLE_RATE)),
                        int(anchor["right_sample"]) - variant_index * int(round(0.01 * SAMPLE_RATE)),
                    )
                    slice_id = f"{anchor_id}:coalesce:{scenario_label}:rep_{repetition + 1}"
                    slices[slice_id] = _slice_audio(audio, shifted_right, args.sentence_seconds)

    endpoint = args.endpoint.rstrip("/")
    runner = BenchmarkRunner(
        endpoint,
        args.device,
        args.timeout_seconds,
        slices,
        http_client=args.http_client,
    )
    health = runner.transport.get_json("/health")
    print(
        f"Server {endpoint}: {health.get('service', 'unknown')} "
        f"default_device={health.get('default_device')} loaded={len(health.get('loaded') or [])}",
        flush=True,
    )
    warmups: list[dict[str, Any]] = []
    if not args.skip_warmup:
        print("\nWarming the four component providers", flush=True)
        for provider in INDIVIDUAL_PROVIDERS:
            warmup = runner.warm(provider)
            warmups.append(warmup)
            print(
                f"  {provider:<30} wall={warmup['wall_seconds']:.3f}s "
                f"server={_round_or_none(warmup.get('server_elapsed_seconds'), 3)}s "
                f"{'ok' if warmup['ok'] else warmup.get('error')}",
                flush=True,
            )
        failed_warmups = [item for item in warmups if not item["ok"]]
        if failed_warmups:
            print("Warmup errors will not abort the benchmark; requests may still expose useful failures.", flush=True)

    anchor_ids = [f"anchor_{index + 1}" for index in range(len(anchors))]
    if "individual" in args.sections:
        print("\nIndividual provider latency", flush=True)
        for provider in INDIVIDUAL_PROVIDERS:
            for repetition in range(args.repetitions):
                for anchor_id in anchor_ids:
                    benchmark = f"individual:{provider}"
                    task = [{
                        "role": provider,
                        "provider": provider,
                        "slice_id": f"{anchor_id}:sentence",
                    }]
                    runner.cycle(benchmark, "single", task, repetition)

    if "stacks" in args.sections:
        print("\nFinal provider stacks", flush=True)
        for stack_name, provider in STACKS.items():
            for repetition in range(args.repetitions):
                for anchor_id in anchor_ids:
                    task = [{
                        "role": "final_sentence",
                        "provider": provider,
                        "slice_id": f"{anchor_id}:sentence",
                    }]
                    runner.cycle(f"stack:{stack_name}", "single", task, repetition)

    scenario_stacks = (
        tuple(STACKS)
        if args.scenario_stack == "both"
        else (args.scenario_stack,)
    )
    if "live" in args.sections:
        print("\nFinal stack plus two live probes: sequential versus simultaneous", flush=True)
        for stack_name in scenario_stacks:
            for mode in ("sequential", "concurrent"):
                for repetition in range(args.repetitions):
                    for anchor_id in anchor_ids:
                        tasks = [
                            {
                                "role": "final_sentence",
                                "provider": STACKS[stack_name],
                                "slice_id": f"{anchor_id}:sentence",
                            },
                            {
                                "role": "live_0.7s",
                                "provider": "speechbrain_resnet",
                                "slice_id": f"{anchor_id}:live_short",
                            },
                            {
                                "role": "live_1.5s",
                                "provider": "speechbrain_resnet",
                                "slice_id": f"{anchor_id}:live_long",
                            },
                        ]
                        runner.cycle(f"final_live:{stack_name}", mode, tasks, repetition)

    if "providers" in args.sections:
        print("\nFour individual providers: sequential versus simultaneous", flush=True)
        for mode in ("sequential", "concurrent"):
            for repetition in range(args.repetitions):
                for anchor_id in anchor_ids:
                    tasks = [
                        {
                            "role": provider,
                            "provider": provider,
                            "slice_id": f"{anchor_id}:sentence",
                        }
                        for provider in INDIVIDUAL_PROVIDERS
                    ]
                    runner.cycle("four_providers", mode, tasks, repetition)

    if "coalesce" in args.sections:
        print("\nFocused identical-PCM stack/component coalescing", flush=True)
        stack_provider = STACKS[args.coalesce_stack]
        for repetition in range(args.repetitions):
            for anchor_id in anchor_ids:
                control_slice = f"{anchor_id}:coalesce:control:rep_{repetition + 1}"
                runner.cycle(
                    "coalesce:control",
                    "standalone_control",
                    [{
                        "role": "standalone_speechbrain",
                        "provider": "speechbrain_resnet",
                        "slice_id": control_slice,
                    }],
                    repetition,
                )

                concurrent_slice = f"{anchor_id}:coalesce:concurrent:rep_{repetition + 1}"
                runner.cycle(
                    f"coalesce:{args.coalesce_stack}",
                    "concurrent_release",
                    [
                        {
                            "role": "final_stack",
                            "provider": stack_provider,
                            "slice_id": concurrent_slice,
                        },
                        {
                            "role": "standalone_speechbrain",
                            "provider": "speechbrain_resnet",
                            "slice_id": concurrent_slice,
                        },
                    ],
                    repetition,
                    release_together=True,
                )

                immediate_slice = f"{anchor_id}:coalesce:followup_immediate:rep_{repetition + 1}"
                runner.followup_cycle(
                    f"coalesce:{args.coalesce_stack}",
                    "followup_immediate",
                    stack_provider,
                    immediate_slice,
                    repetition,
                    0.0,
                )
                if args.coalesce_followup_delay_seconds > 0.0:
                    delayed_slice = f"{anchor_id}:coalesce:followup_delayed:rep_{repetition + 1}"
                    runner.followup_cycle(
                        f"coalesce:{args.coalesce_stack}",
                        f"followup_{args.coalesce_followup_delay_seconds * 1000.0:g}ms",
                        stack_provider,
                        delayed_slice,
                        repetition,
                        args.coalesce_followup_delay_seconds,
                    )

    summaries, comparisons = _summarize(runner)
    parity = _parity(runner.records)
    coalesce_analysis = _coalesce_analysis(runner.records)
    runner.transport.close()
    _print_summary(summaries, comparisons)
    _print_coalesce_analysis(coalesce_analysis)
    low_parity = [
        item for item in parity
        if _finite_float(item["cosine_to_first"].get("min")) is not None
        and float(item["cosine_to_first"]["min"]) < 0.99999
    ]
    print(
        f"\nCosine parity: {len(parity)} repeated provider/slice groups; "
        f"{len(low_parity)} below 0.99999 minimum cosine.",
        flush=True,
    )

    report = {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "client": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "config": {
            "endpoint": endpoint,
            "audio": str(audio_path),
            "device": args.device,
            "repetitions": args.repetitions,
            "slice_count": args.slice_count,
            "sentence_seconds": args.sentence_seconds,
            "short_window_seconds": args.short_window_seconds,
            "long_window_seconds": args.long_window_seconds,
            "timeout_seconds": args.timeout_seconds,
            "http_client": args.http_client,
            "scenario_stack": args.scenario_stack,
            "coalesce_stack": args.coalesce_stack,
            "coalesce_followup_delay_seconds": args.coalesce_followup_delay_seconds,
            "sections": sorted(args.sections),
            "individual_providers": list(INDIVIDUAL_PROVIDERS),
            "stacks": STACKS,
        },
        "server_health_before": health,
        "audio": {
            "duration_seconds": len(audio) / float(SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "anchors": [
                {key: value for key, value in anchor.items() if key != "right_sample"}
                for anchor in anchors
            ],
        },
        "warmups": warmups,
        "summaries": summaries,
        "comparisons": comparisons,
        "coalesce_analysis": coalesce_analysis,
        "cosine_parity": parity,
        "cycles": runner.cycles,
        "requests": _clean_records(runner.records),
    }
    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Detailed JSON written to {output_path}", flush=True)

    errors = [record for record in runner.records if not record.get("ok")]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
