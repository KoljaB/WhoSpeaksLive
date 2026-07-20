"""Resumable dense live-window embedding corpus builder.

The corpus contains causal right-aligned windows from continuous audio.  It is
deliberately independent of transcript sentence boundaries and canonical labels.
One process owns one provider/video job so exiting the process also releases the
provider's CPU and GPU state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable, Sequence

import numpy as np

from common.audio_utils import SAMPLE_RATE, load_audio_file, pad_audio, trim_silence
from embeddings.live_window_experiment_plan import (
    DEFAULT_HOP_SECONDS,
    FULL_WINDOW_UNIVERSE_SECONDS,
    full_window_count,
    seconds_to_samples,
    shared_right_edges,
)


SCHEMA_VERSION = 1
CORPUS_KIND = "continuous_causal_live_windows"
DEFAULT_PREPROCESSING_ID = "live_local_trim_once_pad_v1"
ARRAY_SPECS: dict[str, tuple[str, Any]] = {
    "embeddings": ("embeddings.f32.npy", np.float32),
    "attempted": ("attempted.u1.npy", np.uint8),
    "valid": ("valid.u1.npy", np.uint8),
    "raw_rms": ("raw_rms.f32.npy", np.float32),
    "raw_peak": ("raw_peak.f32.npy", np.float32),
    "trimmed_samples": ("trimmed_samples.i32.npy", np.int32),
    "prepared_samples": ("prepared_samples.i32.npy", np.int32),
    "latency_ms": ("latency_ms.f32.npy", np.float32),
}


class CorpusIdentityError(RuntimeError):
    """Existing artifacts do not belong to the requested corpus job."""


class ControlledStop(RuntimeError):
    """A test or operator-requested checkpoint stop was reached."""


@dataclass(frozen=True)
class JobConfig:
    audio_path: Path
    video_id: str
    provider: str
    output_root: Path
    device: str = "cuda"
    sample_rate: int = SAMPLE_RATE
    hop_seconds: str = str(DEFAULT_HOP_SECONDS)
    window_seconds: tuple[str, ...] = tuple(str(value) for value in FULL_WINDOW_UNIVERSE_SECONDS)
    min_embed_seconds: float = 0.5
    source_start_seconds: float = 0.0
    block_rows: int = 32
    stop_after_embeddings: int = 0
    provider_backend: str = "local"
    provider_endpoint: str = ""
    allow_resume_builder_code_change: bool = False

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if not self.window_seconds:
            raise ValueError("at least one window length is required")
        if self.min_embed_seconds < 0:
            raise ValueError("min_embed_seconds must be non-negative")
        if self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be non-negative")
        if self.block_rows <= 0:
            raise ValueError("block_rows must be greater than zero")
        if self.stop_after_embeddings < 0:
            raise ValueError("stop_after_embeddings must be non-negative")
        if self.provider_backend not in {"local", "server"}:
            raise ValueError("provider_backend must be 'local' or 'server'")
        if self.provider_backend == "server" and not self.provider_endpoint.strip():
            raise ValueError("provider_endpoint is required for the server backend")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "unnamed"


def _sha256_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _identity_hash(value: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_file_hash(value: Any) -> str:
    try:
        path = Path(sys.modules[value.__module__].__file__ or "")
    except (AttributeError, KeyError, TypeError):
        return "unavailable"
    return _sha256_file(path) if path.is_file() else "unavailable"


def _module_sha256(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return "unavailable"
    path = Path(spec.origin)
    return _sha256_file(path) if path.is_file() else "unavailable"


def _provider_model_identity(provider: str) -> dict[str, str]:
    known = {
        "speechbrain_ecapa": "speechbrain/spkrec-ecapa-voxceleb",
        "speechbrain_resnet": "speechbrain/spkrec-resnet-voxceleb",
        "resemblyzer": "resemblyzer",
        "pyannote_embedding": "pyannote/embedding",
        "pyannote_wespeaker_resnet34_lm": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "wespeaker_campplus": "campplus",
        "wespeaker_resnet34_lm_onnx": "hbredin/wespeaker-voxceleb-resnet34-LM",
        "espnet_ecapa_wavlm_joint": "espnet/voxcelebs12_ecapa_wavlm_joint",
    }
    return {
        "provider": provider,
        "model": known.get(provider, "provider-defined"),
    }


def _open_memmap(path: Path, *, dtype: Any, shape: tuple[int, ...], create: bool) -> np.memmap:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value = np.lib.format.open_memmap(path, mode="r+")
    if value.dtype != np.dtype(dtype) or tuple(value.shape) != shape:
        raise CorpusIdentityError(
            f"Partial array mismatch for {path}: expected {shape}/{np.dtype(dtype)}, "
            f"found {tuple(value.shape)}/{value.dtype}"
        )
    return value


def _partial_path(length_dir: Path, final_name: str) -> Path:
    return length_dir / f"{final_name}.partial"


def _final_path(length_dir: Path, final_name: str) -> Path:
    return length_dir / final_name


def _close_arrays(arrays: dict[str, np.memmap]) -> None:
    for value in arrays.values():
        value.flush()
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    arrays.clear()
    gc.collect()


def _initialise_arrays(
    length_dir: Path,
    *,
    tick_count: int,
    embedding_dimension: int,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for key, (filename, dtype) in ARRAY_SPECS.items():
        shape = (tick_count, embedding_dimension) if key == "embeddings" else (tick_count,)
        path = _partial_path(length_dir, filename)
        arrays[key] = _open_memmap(path, dtype=dtype, shape=shape, create=True)
    arrays["embeddings"][:] = np.nan
    arrays["attempted"][:] = 0
    arrays["valid"][:] = 0
    arrays["raw_rms"][:] = np.nan
    arrays["raw_peak"][:] = np.nan
    arrays["trimmed_samples"][:] = -1
    arrays["prepared_samples"][:] = -1
    arrays["latency_ms"][:] = np.nan
    for value in arrays.values():
        value.flush()
    return arrays


def _resume_arrays(
    length_dir: Path,
    *,
    tick_count: int,
    embedding_dimension: int,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for key, (filename, dtype) in ARRAY_SPECS.items():
        shape = (tick_count, embedding_dimension) if key == "embeddings" else (tick_count,)
        partial_path = _partial_path(length_dir, filename)
        final_path = _final_path(length_dir, filename)
        selected_path = partial_path if partial_path.is_file() else final_path
        if not selected_path.is_file():
            raise CorpusIdentityError(
                f"Checkpoint exists but neither partial nor promoted array is present: {partial_path}"
            )
        arrays[key] = _open_memmap(
            selected_path,
            dtype=dtype,
            shape=shape,
            create=False,
        )
    return arrays


def _window_features(raw_audio: np.ndarray, sample_rate: int, min_embed_seconds: float) -> dict[str, Any]:
    raw = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(raw * raw) + 1e-12)) if raw.size else 0.0
    peak = float(np.max(np.abs(raw))) if raw.size else 0.0
    trimmed = trim_silence(raw, sample_rate)
    prepared = pad_audio(trimmed, min_embed_seconds, sample_rate)
    return {
        "raw_rms": rms,
        "raw_peak": peak,
        "trimmed_samples": int(trimmed.size),
        "prepared_samples": int(prepared.size),
        "prepared_audio": np.asarray(prepared, dtype=np.float32).reshape(-1),
    }


def _embed_window(
    provider: Any,
    raw_audio: np.ndarray,
    *,
    sample_rate: int,
    min_embed_seconds: float,
) -> dict[str, Any]:
    features = _window_features(raw_audio, sample_rate, min_embed_seconds)
    started = time.perf_counter()
    try:
        vector = np.asarray(
            provider.embed(features.pop("prepared_audio"), sample_rate),
            dtype=np.float32,
        ).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise RuntimeError("provider returned an empty or non-finite embedding")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("provider returned a zero-norm embedding")
        vector = (vector / norm).astype(np.float32, copy=False)
        error = ""
    except Exception as exc:
        features.pop("prepared_audio", None)
        vector = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        **features,
        "embedding": vector,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "error": error,
    }


def _store_result(
    arrays: dict[str, np.memmap],
    tick_index: int,
    result: dict[str, Any],
    embedding_dimension: int,
) -> bool:
    arrays["attempted"][tick_index] = 1
    arrays["raw_rms"][tick_index] = result["raw_rms"]
    arrays["raw_peak"][tick_index] = result["raw_peak"]
    arrays["trimmed_samples"][tick_index] = result["trimmed_samples"]
    arrays["prepared_samples"][tick_index] = result["prepared_samples"]
    arrays["latency_ms"][tick_index] = result["latency_ms"]
    vector = result["embedding"]
    if vector is None:
        arrays["valid"][tick_index] = 0
        return False
    if int(vector.size) != embedding_dimension:
        raise RuntimeError(
            f"Embedding dimension changed from {embedding_dimension} to {vector.size} at tick {tick_index}"
        )
    arrays["embeddings"][tick_index] = vector
    arrays["valid"][tick_index] = 1
    return True


def _append_error(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _validate_complete_length(
    length_dir: Path,
    *,
    tick_count: int,
    expected_embeddings: int,
    identity_hash: str,
) -> dict[str, Any] | None:
    metadata_path = length_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("job_identity_hash") != identity_hash:
        raise CorpusIdentityError(f"Completed length metadata does not match this job: {metadata_path}")
    if int(metadata.get("tick_count") or -1) != tick_count:
        raise CorpusIdentityError(f"Completed length tick count mismatch: {metadata_path}")
    if int(metadata.get("attempted_embeddings") or -1) != expected_embeddings:
        raise CorpusIdentityError(f"Completed length embedding count mismatch: {metadata_path}")
    for filename, _dtype in ARRAY_SPECS.values():
        if not _final_path(length_dir, filename).is_file():
            raise CorpusIdentityError(f"Completed length is missing {filename}: {length_dir}")
    return metadata


def _promote_partial_arrays(length_dir: Path) -> None:
    for filename, _dtype in ARRAY_SPECS.values():
        partial = _partial_path(length_dir, filename)
        final = _final_path(length_dir, filename)
        if final.is_file() and not partial.is_file():
            continue
        if not partial.is_file():
            raise RuntimeError(f"Cannot complete length; missing partial array {partial}")
        partial.replace(final)


def _aggregate_root_progress(output_root: Path) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for progress_path in sorted(output_root.glob("providers/*/videos/*/progress.json")):
        try:
            value = _read_json(progress_path)
        except Exception:
            continue
        jobs.append(
            {
                "provider": value.get("provider"),
                "video_id": value.get("video_id"),
                "status": value.get("status"),
                "completed_embeddings": int(value.get("completed_embeddings") or 0),
                "expected_embeddings": int(value.get("expected_embeddings") or 0),
                "percent": float(value.get("percent") or 0.0),
                "progress_path": str(progress_path),
            }
        )
    expected = sum(item["expected_embeddings"] for item in jobs)
    completed = sum(item["completed_embeddings"] for item in jobs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_kind": CORPUS_KIND,
        "status": (
            "complete"
            if jobs and all(item["status"] == "complete" for item in jobs)
            else "running"
            if any(item["status"] in {"initializing", "running"} for item in jobs)
            else "partial"
        ),
        "job_count": len(jobs),
        "completed_embeddings": completed,
        "expected_embeddings": expected,
        "percent": round(100.0 * completed / expected, 6) if expected else 0.0,
        "updated_at": _utc_now(),
        "jobs": jobs,
    }
    _write_json_atomic(output_root / "progress.json", payload)
    return payload


def _write_job_progress(
    path: Path,
    *,
    config: JobConfig,
    status: str,
    completed_embeddings: int,
    successful_embeddings: int,
    failed_embeddings: int,
    expected_embeddings: int,
    completed_lengths: int,
    length_count: int,
    current_length_seconds: float | None,
    current_tick_index: int | None,
    tick_count: int,
    elapsed_seconds: float,
    provider_load_seconds: float,
    error: str = "",
) -> dict[str, Any]:
    rate = completed_embeddings / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(0, expected_embeddings - completed_embeddings)
    eta = remaining / rate if rate > 0 else None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_kind": CORPUS_KIND,
        "provider": config.provider,
        "video_id": config.video_id,
        "status": status,
        "completed_embeddings": completed_embeddings,
        "successful_embeddings": successful_embeddings,
        "failed_embeddings": failed_embeddings,
        "expected_embeddings": expected_embeddings,
        "percent": round(100.0 * completed_embeddings / expected_embeddings, 6)
        if expected_embeddings
        else 100.0,
        "completed_lengths": completed_lengths,
        "length_count": length_count,
        "current_length_seconds": current_length_seconds,
        "current_tick_index": current_tick_index,
        "tick_count": tick_count,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "provider_load_seconds": round(provider_load_seconds, 6),
        "embeddings_per_second": round(rate, 6),
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "updated_at": _utc_now(),
        "error": error,
    }
    _write_json_atomic(path, payload)
    _aggregate_root_progress(config.output_root)
    return payload


def _print_progress(payload: dict[str, Any], *, force: bool, state: dict[str, Any]) -> None:
    percent = float(payload["percent"])
    now = time.monotonic()
    if not force and percent < float(state.get("next_percent", 0.0)) and now - float(
        state.get("last_time", 0.0)
    ) < 30.0:
        return
    eta = payload.get("eta_seconds")
    eta_text = f"{float(eta) / 60.0:.1f}m" if eta is not None else "unknown"
    length = payload.get("current_length_seconds")
    length_text = f"{float(length):.1f}s" if length is not None else "-"
    print(
        "[progress] "
        f"{percent:6.2f}% "
        f"{payload['completed_embeddings']}/{payload['expected_embeddings']} "
        f"length={length_text} "
        f"done_lengths={payload['completed_lengths']}/{payload['length_count']} "
        f"rate={float(payload['embeddings_per_second']):.2f}/s eta={eta_text} "
        f"status={payload['status']}",
        flush=True,
    )
    state["next_percent"] = math.floor(percent * 2.0 + 1.0) / 2.0
    state["last_time"] = now


def build_live_window_job(
    config: JobConfig,
    *,
    provider_factory: Callable[[str, str], Any] | None = None,
    audio_loader: Callable[[Path, int], tuple[np.ndarray, int]] = load_audio_file,
    progress_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build or resume one provider/video corpus job."""

    audio_path = config.audio_path.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        output_root / "corpus.json",
        {
            "schema_version": SCHEMA_VERSION,
            "corpus_kind": CORPUS_KIND,
            "alignment": "causal_right_aligned_shared_source_grid",
            "edge_policy": "full_windows_only",
            "embedding_policy": "all_full_windows_including_non_speech",
            "storage_dtype": "float32",
            "progress_file": str(output_root / "progress.json"),
        },
    )

    audio_file_sha256 = _sha256_file(audio_path)
    audio, decoded_sample_rate = audio_loader(audio_path, config.sample_rate)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if decoded_sample_rate != config.sample_rate:
        raise RuntimeError(
            f"Audio loader returned {decoded_sample_rate} Hz, expected {config.sample_rate} Hz"
        )
    if audio.size <= 0:
        raise RuntimeError(f"Decoded audio is empty: {audio_path}")
    decoded_pcm_sha256 = _sha256_bytes(np.ascontiguousarray(audio).tobytes())

    hop_samples = seconds_to_samples(config.hop_seconds, sample_rate=config.sample_rate)
    source_start_samples = seconds_to_samples(
        config.source_start_seconds,
        sample_rate=config.sample_rate,
        allow_zero=True,
        label="source_start_seconds",
    )
    window_samples = tuple(
        sorted(
            {
                seconds_to_samples(value, sample_rate=config.sample_rate, label="window_seconds")
                for value in config.window_seconds
            }
        )
    )
    right_edges = np.asarray(
        tuple(
            shared_right_edges(
                int(audio.size),
                hop_samples,
                source_start_samples=source_start_samples,
            )
        ),
        dtype=np.int64,
    )
    expected_by_length = {
        value: full_window_count(
            total_samples=int(audio.size),
            hop_samples=hop_samples,
            window_samples=value,
            source_start_samples=source_start_samples,
        )
        for value in window_samples
    }
    expected_embeddings = sum(expected_by_length.values())

    identity = {
        "schema_version": SCHEMA_VERSION,
        "corpus_kind": CORPUS_KIND,
        "video_id": config.video_id,
        "audio_filename": audio_path.name,
        "audio_file_sha256": audio_file_sha256,
        "decoded_pcm_sha256": decoded_pcm_sha256,
        "decoded_samples": int(audio.size),
        "sample_rate": config.sample_rate,
        "source_start_samples": source_start_samples,
        "hop_samples": hop_samples,
        "window_samples": list(window_samples),
        "edge_policy": "full_windows_only",
        "embedding_policy": "all_full_windows_including_non_speech",
        "provider": config.provider,
        "provider_backend": config.provider_backend,
        "provider_endpoint": config.provider_endpoint,
        "device": config.device,
        "preprocessing_id": DEFAULT_PREPROCESSING_ID,
        "min_embed_seconds": config.min_embed_seconds,
        "storage_dtype": "float32",
        "builder_code_sha256": _sha256_file(Path(__file__)),
        "audio_utils_code_sha256": _source_file_hash(trim_silence),
        "embedding_provider_code_sha256": _module_sha256("embeddings.embedding_providers"),
        "package_versions": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "torchaudio": _package_version("torchaudio"),
            "pyannote.audio": _package_version("pyannote.audio"),
            "speechbrain": _package_version("speechbrain"),
        },
    }
    job_identity_hash = _identity_hash(identity)

    provider_slug = _safe_slug(config.provider)
    video_slug = _safe_slug(config.video_id)
    video_dir = output_root / "videos" / video_slug
    job_dir = output_root / "providers" / provider_slug / "videos" / video_slug
    lengths_dir = job_dir / "lengths"
    progress_path = job_dir / "progress.json"
    job_path = job_dir / "job.json"

    if job_path.is_file():
        existing = _read_json(job_path)
        if existing.get("job_identity_hash") != job_identity_hash:
            existing_identity = dict(existing.get("identity") or {})
            current_contract = dict(identity)
            existing_identity.pop("builder_code_sha256", None)
            current_contract.pop("builder_code_sha256", None)
            if config.allow_resume_builder_code_change and existing_identity == current_contract:
                job_identity_hash = str(existing["job_identity_hash"])
            else:
                raise CorpusIdentityError(
                    f"Existing job identity differs at {job_path}; use a new output root for a new corpus contract"
                )
    else:
        _write_json_atomic(
            job_path,
            {
                "status": "created",
                "job_identity_hash": job_identity_hash,
                "identity": identity,
                "expected_embeddings": expected_embeddings,
                "expected_by_window_samples": {
                    str(key): value for key, value in expected_by_length.items()
                },
                "created_at": _utc_now(),
            },
        )

    source_path = video_dir / "source.json"
    source_payload = {
        "video_id": config.video_id,
        "audio_filename": audio_path.name,
        "audio_path_at_creation": str(audio_path),
        "audio_file_sha256": audio_file_sha256,
        "decoded_pcm_sha256": decoded_pcm_sha256,
        "decoded_samples": int(audio.size),
        "duration_seconds": audio.size / config.sample_rate,
        "sample_rate": config.sample_rate,
        "source_start_samples": source_start_samples,
    }
    if source_path.is_file():
        existing_source = _read_json(source_path)
        for key in ("video_id", "audio_file_sha256", "decoded_pcm_sha256", "sample_rate"):
            if existing_source.get(key) != source_payload.get(key):
                raise CorpusIdentityError(f"Source metadata mismatch for {key}: {source_path}")
    else:
        _write_json_atomic(source_path, source_payload)

    timeline_dir = video_dir / "timeline"
    timeline_array_path = timeline_dir / "right_edges.i64.npy"
    timeline_metadata_path = timeline_dir / "metadata.json"
    timeline_hash = _sha256_bytes(np.ascontiguousarray(right_edges).tobytes())
    if timeline_metadata_path.is_file():
        metadata = _read_json(timeline_metadata_path)
        if metadata.get("timeline_sha256") != timeline_hash:
            raise CorpusIdentityError(f"Timeline mismatch: {timeline_metadata_path}")
    else:
        _write_npy_atomic(timeline_array_path, right_edges)
        _write_json_atomic(
            timeline_metadata_path,
            {
                "sample_rate": config.sample_rate,
                "hop_samples": hop_samples,
                "hop_seconds": hop_samples / config.sample_rate,
                "source_start_samples": source_start_samples,
                "tick_count": int(right_edges.size),
                "timeline_sha256": timeline_hash,
                "array": str(timeline_array_path),
            },
        )

    prior_progress = _read_json(progress_path) if progress_path.is_file() else {}
    prior_elapsed = float(prior_progress.get("elapsed_seconds") or 0.0)
    session_started = time.perf_counter()
    provider_load_seconds = float(prior_progress.get("provider_load_seconds") or 0.0)
    print_state: dict[str, Any] = {"next_percent": 0.0, "last_time": 0.0}

    def report(
        status: str,
        *,
        completed: int,
        successful: int,
        failed: int,
        completed_lengths: int,
        current_length: float | None,
        current_tick: int | None,
        error: str = "",
        force_print: bool = False,
    ) -> dict[str, Any]:
        elapsed = prior_elapsed + (time.perf_counter() - session_started)
        payload = _write_job_progress(
            progress_path,
            config=config,
            status=status,
            completed_embeddings=completed,
            successful_embeddings=successful,
            failed_embeddings=failed,
            expected_embeddings=expected_embeddings,
            completed_lengths=completed_lengths,
            length_count=len(window_samples),
            current_length_seconds=current_length,
            current_tick_index=current_tick,
            tick_count=int(right_edges.size),
            elapsed_seconds=elapsed,
            provider_load_seconds=provider_load_seconds,
            error=error,
        )
        _print_progress(payload, force=force_print, state=print_state)
        if progress_observer is not None:
            progress_observer(dict(payload))
        return payload

    completed = 0
    successful = 0
    failed = 0
    completed_lengths = 0
    length_metadata: dict[int, dict[str, Any]] = {}
    partial_counts: dict[int, tuple[int, int, int]] = {}
    for length_samples in window_samples:
        length_dir = lengths_dir / f"{round(length_samples / config.sample_rate * 1000):04d}ms"
        metadata = _validate_complete_length(
            length_dir,
            tick_count=int(right_edges.size),
            expected_embeddings=expected_by_length[length_samples],
            identity_hash=job_identity_hash,
        )
        if metadata is None:
            checkpoint_path = length_dir / "checkpoint.json"
            if checkpoint_path.is_file():
                checkpoint = _read_json(checkpoint_path)
                if checkpoint.get("job_identity_hash") != job_identity_hash:
                    raise CorpusIdentityError(f"Checkpoint identity mismatch: {checkpoint_path}")
                attempted = int(checkpoint.get("attempted_embeddings") or 0)
                succeeded = int(checkpoint.get("successful_embeddings") or 0)
                failed_count = int(checkpoint.get("failed_embeddings") or (attempted - succeeded))
                partial_counts[length_samples] = (attempted, succeeded, failed_count)
                completed += attempted
                successful += succeeded
                failed += failed_count
            continue
        length_metadata[length_samples] = metadata
        completed += int(metadata["attempted_embeddings"])
        successful += int(metadata["successful_embeddings"])
        failed += int(metadata["failed_embeddings"])
        completed_lengths += 1

    if completed == expected_embeddings:
        payload = report(
            "complete",
            completed=completed,
            successful=successful,
            failed=failed,
            completed_lengths=completed_lengths,
            current_length=None,
            current_tick=None,
            force_print=True,
        )
        return payload

    report(
        "initializing",
        completed=completed,
        successful=successful,
        failed=failed,
        completed_lengths=completed_lengths,
        current_length=None,
        current_tick=None,
        force_print=True,
    )

    if provider_factory is None:
        from embeddings.embedding_providers import create_embedding_provider

        provider_factory = create_embedding_provider

    provider_started = time.perf_counter()
    provider = provider_factory(config.provider, config.device)
    provider_load_seconds += time.perf_counter() - provider_started
    _write_json_atomic(
        output_root / "providers" / provider_slug / "provider.json",
        {
            **_provider_model_identity(config.provider),
            "provider_class": type(provider).__name__,
            "device": config.device,
            "packages": {
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "torchaudio": _package_version("torchaudio"),
                "pyannote.audio": _package_version("pyannote.audio"),
                "speechbrain": _package_version("speechbrain"),
            },
            "embedding_provider_code_sha256": _source_file_hash(provider),
            "updated_at": _utc_now(),
        },
    )

    session_attempted = 0
    try:
        for length_samples in window_samples:
            if length_samples in length_metadata:
                continue
            length_seconds = length_samples / config.sample_rate
            length_dir = lengths_dir / f"{round(length_seconds * 1000):04d}ms"
            length_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = length_dir / "checkpoint.json"
            errors_path = length_dir / "errors.jsonl"
            first_eligible_index = next(
                (
                    index
                    for index, right_edge in enumerate(right_edges)
                    if int(right_edge) >= length_samples
                ),
                int(right_edges.size),
            )
            checkpoint = _read_json(checkpoint_path) if checkpoint_path.is_file() else None
            arrays: dict[str, np.memmap]
            embedding_dimension: int
            next_tick_index: int

            if checkpoint is not None:
                if checkpoint.get("job_identity_hash") != job_identity_hash:
                    raise CorpusIdentityError(f"Checkpoint identity mismatch: {checkpoint_path}")
                embedding_dimension = int(checkpoint["embedding_dimension"])
                next_tick_index = int(checkpoint["next_tick_index"])
                arrays = _resume_arrays(
                    length_dir,
                    tick_count=int(right_edges.size),
                    embedding_dimension=embedding_dimension,
                )
                prior_length_attempted = int(np.sum(arrays["attempted"]))
                prior_length_successful = int(np.sum(arrays["valid"]))
                expected_partial = partial_counts.get(length_samples, (0, 0, 0))
                actual_partial = (
                    prior_length_attempted,
                    prior_length_successful,
                    prior_length_attempted - prior_length_successful,
                )
                if actual_partial != expected_partial:
                    attempted_indices = np.flatnonzero(np.asarray(arrays["attempted"], dtype=bool))
                    expected_indices = np.arange(
                        first_eligible_index,
                        first_eligible_index + prior_length_attempted,
                        dtype=np.int64,
                    )
                    arrays_are_contiguous = np.array_equal(attempted_indices, expected_indices)
                    checkpoint_is_prefix = (
                        actual_partial[0] >= expected_partial[0]
                        and actual_partial[1] >= expected_partial[1]
                        and actual_partial[2] >= expected_partial[2]
                    )
                    if not arrays_are_contiguous or not checkpoint_is_prefix:
                        raise CorpusIdentityError(
                            f"Checkpoint counters disagree with partial arrays for {length_seconds:.1f}s: "
                            f"checkpoint={expected_partial}, arrays={actual_partial}"
                        )
                    # A hard kill can land after mmap.flush() but before the
                    # atomic JSON checkpoint replace. In that case the
                    # contiguous attempted mask is the durable source of truth.
                    completed += actual_partial[0] - expected_partial[0]
                    successful += actual_partial[1] - expected_partial[1]
                    failed += actual_partial[2] - expected_partial[2]
                    next_tick_index = first_eligible_index + prior_length_attempted
                    _write_json_atomic(
                        checkpoint_path,
                        {
                            **checkpoint,
                            "status": "running",
                            "next_tick_index": next_tick_index,
                            "attempted_embeddings": actual_partial[0],
                            "successful_embeddings": actual_partial[1],
                            "failed_embeddings": actual_partial[2],
                            "recovered_array_ahead_of_checkpoint": True,
                            "updated_at": _utc_now(),
                        },
                    )
            else:
                if first_eligible_index >= right_edges.size:
                    raise RuntimeError(f"No eligible ticks for {length_seconds:.1f}s")
                pending: list[tuple[int, dict[str, Any]]] = []
                successful_probe: np.ndarray | None = None
                probe_index = first_eligible_index
                while probe_index < right_edges.size and len(pending) < 10:
                    right_edge = int(right_edges[probe_index])
                    raw = audio[right_edge - length_samples : right_edge]
                    result = _embed_window(
                        provider,
                        raw,
                        sample_rate=config.sample_rate,
                        min_embed_seconds=config.min_embed_seconds,
                    )
                    pending.append((probe_index, result))
                    if result["embedding"] is not None:
                        successful_probe = result["embedding"]
                        break
                    probe_index += 1
                if successful_probe is None:
                    raise RuntimeError(
                        f"Provider failed to return a valid vector for the first {len(pending)} "
                        f"eligible {length_seconds:.1f}s windows"
                    )
                embedding_dimension = int(successful_probe.size)
                arrays = _initialise_arrays(
                    length_dir,
                    tick_count=int(right_edges.size),
                    embedding_dimension=embedding_dimension,
                )
                for tick_index, result in pending:
                    ok = _store_result(arrays, tick_index, result, embedding_dimension)
                    completed += 1
                    session_attempted += 1
                    successful += int(ok)
                    failed += int(not ok)
                    if result["error"]:
                        _append_error(
                            errors_path,
                            {
                                "tick_index": tick_index,
                                "right_edge_sample": int(right_edges[tick_index]),
                                "error": result["error"],
                            },
                        )
                next_tick_index = pending[-1][0] + 1
                for value in arrays.values():
                    value.flush()
                _write_json_atomic(
                    checkpoint_path,
                    {
                        "status": "running",
                        "job_identity_hash": job_identity_hash,
                        "window_samples": length_samples,
                        "embedding_dimension": embedding_dimension,
                        "first_eligible_tick_index": first_eligible_index,
                        "next_tick_index": next_tick_index,
                        "attempted_embeddings": int(np.sum(arrays["attempted"])),
                        "successful_embeddings": int(np.sum(arrays["valid"])),
                        "updated_at": _utc_now(),
                    },
                )

            if config.stop_after_embeddings and session_attempted >= config.stop_after_embeddings:
                _close_arrays(arrays)
                payload = report(
                    "paused",
                    completed=completed,
                    successful=successful,
                    failed=failed,
                    completed_lengths=completed_lengths,
                    current_length=length_seconds,
                    current_tick=next_tick_index,
                    force_print=True,
                )
                raise ControlledStop(json.dumps(payload))

            while next_tick_index < right_edges.size:
                block_end = min(int(right_edges.size), next_tick_index + config.block_rows)
                for tick_index in range(next_tick_index, block_end):
                    right_edge = int(right_edges[tick_index])
                    if right_edge < length_samples:
                        continue
                    if arrays["attempted"][tick_index]:
                        continue
                    raw = audio[right_edge - length_samples : right_edge]
                    result = _embed_window(
                        provider,
                        raw,
                        sample_rate=config.sample_rate,
                        min_embed_seconds=config.min_embed_seconds,
                    )
                    ok = _store_result(arrays, tick_index, result, embedding_dimension)
                    completed += 1
                    session_attempted += 1
                    successful += int(ok)
                    failed += int(not ok)
                    if result["error"]:
                        _append_error(
                            errors_path,
                            {
                                "tick_index": tick_index,
                                "right_edge_sample": right_edge,
                                "error": result["error"],
                            },
                        )
                    if config.stop_after_embeddings and session_attempted >= config.stop_after_embeddings:
                        next_tick_index = tick_index + 1
                        break
                else:
                    next_tick_index = block_end

                for value in arrays.values():
                    value.flush()
                length_attempted = int(np.sum(arrays["attempted"]))
                length_successful = int(np.sum(arrays["valid"]))
                _write_json_atomic(
                    checkpoint_path,
                    {
                        "status": "running",
                        "job_identity_hash": job_identity_hash,
                        "window_samples": length_samples,
                        "embedding_dimension": embedding_dimension,
                        "first_eligible_tick_index": first_eligible_index,
                        "next_tick_index": next_tick_index,
                        "attempted_embeddings": length_attempted,
                        "successful_embeddings": length_successful,
                        "failed_embeddings": length_attempted - length_successful,
                        "updated_at": _utc_now(),
                    },
                )
                payload = report(
                    "running",
                    completed=completed,
                    successful=successful,
                    failed=failed,
                    completed_lengths=completed_lengths,
                    current_length=length_seconds,
                    current_tick=next_tick_index,
                )
                if config.stop_after_embeddings and session_attempted >= config.stop_after_embeddings:
                    _close_arrays(arrays)
                    payload = report(
                        "paused",
                        completed=completed,
                        successful=successful,
                        failed=failed,
                        completed_lengths=completed_lengths,
                        current_length=length_seconds,
                        current_tick=next_tick_index,
                        force_print=True,
                    )
                    raise ControlledStop(json.dumps(payload))

            length_attempted = int(np.sum(arrays["attempted"]))
            length_successful = int(np.sum(arrays["valid"]))
            expected_length = expected_by_length[length_samples]
            if length_attempted != expected_length:
                raise RuntimeError(
                    f"Length {length_seconds:.1f}s completed {length_attempted}/{expected_length} windows"
                )
            latency_values = np.asarray(arrays["latency_ms"])[np.asarray(arrays["attempted"]).astype(bool)]
            latency_values = latency_values[np.isfinite(latency_values)]
            _close_arrays(arrays)
            _promote_partial_arrays(length_dir)
            metadata = {
                "status": "complete",
                "job_identity_hash": job_identity_hash,
                "window_samples": length_samples,
                "window_seconds": length_seconds,
                "tick_count": int(right_edges.size),
                "first_eligible_tick_index": first_eligible_index,
                "attempted_embeddings": length_attempted,
                "successful_embeddings": length_successful,
                "failed_embeddings": length_attempted - length_successful,
                "embedding_dimension": embedding_dimension,
                "latency_ms_mean": float(np.mean(latency_values)) if latency_values.size else None,
                "latency_ms_p95": float(np.quantile(latency_values, 0.95))
                if latency_values.size
                else None,
                "completed_at": _utc_now(),
                "arrays": {key: filename for key, (filename, _dtype) in ARRAY_SPECS.items()},
            }
            _write_json_atomic(length_dir / "metadata.json", metadata)
            _write_json_atomic(
                checkpoint_path,
                {
                    "status": "complete",
                    "job_identity_hash": job_identity_hash,
                    "window_samples": length_samples,
                    "embedding_dimension": embedding_dimension,
                    "first_eligible_tick_index": first_eligible_index,
                    "next_tick_index": int(right_edges.size),
                    "attempted_embeddings": length_attempted,
                    "successful_embeddings": length_successful,
                    "failed_embeddings": length_attempted - length_successful,
                    "updated_at": _utc_now(),
                },
            )
            completed_lengths += 1
            report(
                "running",
                completed=completed,
                successful=successful,
                failed=failed,
                completed_lengths=completed_lengths,
                current_length=length_seconds,
                current_tick=int(right_edges.size),
                force_print=True,
            )

        payload = report(
            "complete",
            completed=completed,
            successful=successful,
            failed=failed,
            completed_lengths=completed_lengths,
            current_length=None,
            current_tick=None,
            force_print=True,
        )
        _write_json_atomic(
            job_path,
            {
                **_read_json(job_path),
                "status": "complete",
                "completed_at": _utc_now(),
                "successful_embeddings": successful,
                "failed_embeddings": failed,
            },
        )
        return payload
    except ControlledStop:
        raise
    except Exception as exc:
        report(
            "failed",
            completed=completed,
            successful=successful,
            failed=failed,
            completed_lengths=completed_lengths,
            current_length=None,
            current_tick=None,
            error=f"{type(exc).__name__}: {exc}",
            force_print=True,
        )
        _append_error(
            job_dir / "job_errors.jsonl",
            {
                "time": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        try:
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass
        try:
            del provider
        except UnboundLocalError:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _parse_window_seconds(raw: str) -> tuple[str, ...]:
    value = raw.strip()
    if not value or value == "full":
        return tuple(str(item) for item in FULL_WINDOW_UNIVERSE_SECONDS)
    parts = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parts:
        raise argparse.ArgumentTypeError("window list must not be empty")
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a resumable dense causal live-window embedding corpus."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--provider", default="pyannote_wespeaker_resnet34_lm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider-backend", choices=("local", "server"), default="local")
    parser.add_argument("--provider-endpoint", default="")
    parser.add_argument("--allow-resume-builder-code-change", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime") / "optimization" / "live_shifting_windows_v1",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--hop-seconds", default=str(DEFAULT_HOP_SECONDS))
    parser.add_argument("--window-seconds", type=_parse_window_seconds, default="full")
    parser.add_argument("--min-embed-seconds", type=float, default=0.5)
    parser.add_argument("--source-start-seconds", type=float, default=0.0)
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument(
        "--stop-after-embeddings",
        type=int,
        default=0,
        help="Testing hook: checkpoint and exit after this many new embeddings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    video_id = args.video_id.strip() or re.sub(r"\.audio$", "", args.audio.stem)
    config = JobConfig(
        audio_path=args.audio,
        video_id=video_id,
        provider=args.provider,
        output_root=args.output_root,
        device=args.device,
        sample_rate=args.sample_rate,
        hop_seconds=args.hop_seconds,
        window_seconds=args.window_seconds,
        min_embed_seconds=args.min_embed_seconds,
        source_start_seconds=args.source_start_seconds,
        block_rows=args.block_rows,
        stop_after_embeddings=args.stop_after_embeddings,
        provider_backend=args.provider_backend,
        provider_endpoint=args.provider_endpoint,
        allow_resume_builder_code_change=args.allow_resume_builder_code_change,
    )
    provider_factory = None
    if args.provider_backend == "server":
        from embeddings.embedding_providers import RemotePreparedEmbeddingProvider

        provider_factory = lambda provider, device: RemotePreparedEmbeddingProvider(
            args.provider_endpoint,
            provider,
            device,
            timeout_seconds=300.0,
        )
    try:
        build_live_window_job(config, provider_factory=provider_factory)
    except ControlledStop:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
