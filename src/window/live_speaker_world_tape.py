"""Crash-resilient production traces for live-speaker parity replay.

The recorder deliberately observes the real GUI process.  It does not infer
missing timing or rebuild profile availability from sentence end times.
"""

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid
from typing import Any

import numpy as np

from common.audio_utils import json_dumps
from window.live_speaker_e2e_contract import live_runtime_config


WORLD_TAPE_CONTRACT_ID = "whospeaks.live_world_tape.v1"
WORLD_TAPE_VECTOR_REF_KEY = "$world_tape_array"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "media")).strip(".-_")
    return text[:80] or "media"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert the same values accepted by the GUI bus into plain JSON data."""

    return json.loads(json_dumps(value))


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _media_record(media: Any) -> dict[str, Any]:
    audio_path = Path(getattr(media, "audio_file", ""))
    video_path = Path(getattr(media, "video_file", ""))
    result: dict[str, Any] = {
        "video_id": str(getattr(media, "video_id", "") or ""),
        "url": str(getattr(media, "url", "") or ""),
        "audio_path": str(audio_path),
        "video_path": str(video_path),
    }
    if audio_path.is_file():
        result["audio_size_bytes"] = int(audio_path.stat().st_size)
        result["audio_sha256"] = _sha256_file(audio_path)
    if video_path.is_file() and video_path != audio_path:
        result["video_size_bytes"] = int(video_path.stat().st_size)
        result["video_sha256"] = _sha256_file(video_path)
    return result


class LiveSpeakerWorldTapeRecorder:
    """Append production events and numeric arrays without losing prior runs.

    ``output_root`` is a campaign directory.  Every process launch receives a
    unique child directory, so restarting the same command never overwrites a
    partially useful recording.
    """

    def __init__(self, output_root: Path, *, args: Any, media: Any) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = uuid.uuid4().hex
        video_id = _safe_name(getattr(media, "video_id", "media"))
        self.output_dir = self.output_root / f"{video_id}_{stamp}_{self.run_id[:8]}"
        self.output_dir.mkdir(parents=False, exist_ok=False)
        self.events_path = self.output_dir / "events.jsonl"
        self.arrays_path = self.output_dir / "arrays.f32"
        self.arrays_index_path = self.output_dir / "arrays.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self._events = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._arrays = self.arrays_path.open("ab", buffering=0)
        self._arrays_index = self.arrays_index_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._sequence = 0
        self._written_event_count = 0
        self._array_sequence = 0
        self._array_offset_bytes = 0
        self._counts: Counter[str] = Counter()
        self._closed = False
        self._accepting = True
        self._writer_error: str | None = None
        self._final_summary: dict[str, Any] | None = None
        self._started_epoch = time.time()
        self._started_monotonic = time.monotonic()
        self._last_flush_monotonic = self._started_monotonic
        self._sample_rate = 16000
        self._config = live_runtime_config(args)
        self._media_history = [_media_record(media)]
        self._write_manifest(status="recording", reason="started")
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="live-speaker-world-tape-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._written_event_count

    def record_public(self, event: str, payload: dict[str, Any]) -> None:
        self._record("public", str(event), payload)

    def record_internal(self, event: str, payload: dict[str, Any]) -> None:
        self._record("internal", str(event), payload)

    def record_browser_samples(
        self,
        samples: list[dict[str, Any]],
        *,
        batch_sequence: int | None = None,
    ) -> int:
        valid = [dict(item) for item in samples if isinstance(item, dict)]
        if valid:
            self._record(
                "browser",
                "ui_sample_clock",
                {
                    "samples": valid,
                    "sample_count": len(valid),
                    "batch_sequence": batch_sequence,
                },
            )
        return len(valid)

    def update_media(self, media: Any) -> None:
        record = _media_record(media)
        with self._lock:
            if self._closed:
                return
            if self._media_history and record == self._media_history[-1]:
                return
            self._media_history.append(record)
        self._record("metadata", "media_changed", record)
        self._write_manifest(status="recording", reason="media_changed")

    def record_decoded_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        """Bind the exact decoded float32 PCM used by the GUI to this tape."""

        rate = int(sample_rate)
        if rate <= 0:
            raise ValueError("Decoded-audio sample rate must be positive.")
        pcm = np.ascontiguousarray(
            np.asarray(audio, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        raw = pcm.tobytes(order="C")
        record = {
            "decoded_pcm_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_samples": int(pcm.size),
            "sample_rate": rate,
            "duration_seconds": float(pcm.size) / float(rate),
            "dtype": "float32",
            "byte_order": "native",
            "contiguous": True,
        }
        with self._lock:
            if self._closed or not self._accepting:
                return record
            if self._sequence > 0 and rate != self._sample_rate:
                self._mark_writer_error_locked(
                    "decoded audio sample rate changed after event recording began "
                    f"({self._sample_rate} -> {rate})"
                )
            else:
                self._sample_rate = rate
            if not self._media_history:
                self._media_history.append(dict(record))
            else:
                self._media_history[-1] = {**self._media_history[-1], **record}
        self._record("metadata", "decoded_audio_bound", record)
        self._write_manifest(status="recording", reason="decoded_audio_bound")
        return dict(record)

    def checkpoint(self, reason: str = "checkpoint") -> dict[str, Any]:
        with self._lock:
            closed = self._closed
            final_summary = self._final_summary
        if closed:
            if final_summary is not None:
                return copy.deepcopy(final_summary)
            with self._io_lock:
                return self._artifact_summary(status="invalid", reason=reason, include_hashes=True)
        self._write_queue.join()
        with self._io_lock:
            self._flush_locked()
            summary = self._artifact_summary(
                status="recording",
                reason=reason,
                include_hashes=True,
            )
            self._write_manifest_locked(status="recording", reason=reason, artifact=summary)
            return summary

    def close(self, reason: str = "application_close") -> dict[str, Any]:
        with self._lock:
            if self._closed:
                if self._final_summary is not None:
                    return copy.deepcopy(self._final_summary)
                with self._io_lock:
                    return self._artifact_summary(
                        status="invalid",
                        reason=reason,
                        include_hashes=True,
                    )
            self._accepting = False
        self._write_queue.join()
        self._write_queue.put(None)
        self._writer_thread.join(timeout=30.0)
        if self._writer_thread.is_alive():
            self._mark_writer_error("TimeoutError: writer thread did not stop within 30 seconds")
        with self._io_lock:
            self._flush_locked()
            self._events.close()
            self._arrays_index.close()
            self._arrays.close()
            with self._lock:
                self._closed = True
                writer_error = self._writer_error
            final_status = "complete" if writer_error is None else "invalid"
            final_reason = str(reason)
            if writer_error is not None:
                final_reason = f"{final_reason}; writer_error={writer_error}"
            summary = self._artifact_summary(
                status=final_status,
                reason=final_reason,
                include_hashes=True,
            )
            self._write_manifest_locked(
                status=final_status,
                reason=final_reason,
                artifact=summary,
            )
            with self._lock:
                self._final_summary = copy.deepcopy(summary)
            return copy.deepcopy(summary)

    def _record(self, stream: str, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed or not self._accepting:
                return
            self._sequence += 1
            now_monotonic = time.monotonic()
            media_time = self._payload_media_time(payload)
            try:
                # The writer is asynchronous.  Freeze the caller-owned object before
                # returning so a later list/dict/ndarray mutation cannot alter history.
                # deepcopy performs no disk I/O; JSON conversion and array extraction
                # deliberately remain on the writer thread.
                frozen_payload = copy.deepcopy(payload)
            except Exception as exc:
                self._mark_writer_error_locked(
                    f"payload freeze failed for {stream}:{event}: "
                    f"{type(exc).__name__}: {exc}"
                )
                frozen_payload = {
                    "world_tape_payload_freeze_failed": True,
                    "original_type": type(payload).__name__,
                }
            record = {
                "seq": self._sequence,
                "wall_us": int(round((now_monotonic - self._started_monotonic) * 1_000_000.0)),
                "media_sample": (
                    None if media_time is None else int(round(media_time * self._sample_rate))
                ),
                "media_time": media_time,
                "stream": str(stream),
                "event": str(event),
                "payload": frozen_payload,
            }
            self._write_queue.put(record)

    def _writer_loop(self) -> None:
        while True:
            record = self._write_queue.get()
            try:
                if record is None:
                    return
                plain = _jsonable(record.get("payload") or {})
                with self._io_lock:
                    # Array blocks, their index rows, the referencing event, and
                    # the persisted counters form one checkpoint-visible unit.
                    stored = self._externalize_arrays(
                        plain,
                        path=(str(record.get("event") or "event"),),
                    )
                    output = {**record, "payload": stored}
                    self._events.write(
                        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    with self._lock:
                        self._written_event_count += 1
                        self._counts[f"{output['stream']}:{output['event']}"] += 1
                    if time.monotonic() - self._last_flush_monotonic >= 1.0:
                        self._flush_locked()
            except Exception as exc:
                self._mark_writer_error(f"{type(exc).__name__}: {exc}")
            finally:
                self._write_queue.task_done()

    def _mark_writer_error(self, message: str) -> None:
        with self._lock:
            self._mark_writer_error_locked(message)

    def _mark_writer_error_locked(self, message: str) -> None:
        if self._writer_error is None:
            self._writer_error = str(message)

    @staticmethod
    def _payload_media_time(payload: dict[str, Any]) -> float | None:
        for key in ("media_time", "playback_time", "available_at", "end", "right"):
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                return round(value, 6)
        return None

    def _externalize_arrays(self, value: Any, *, path: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._externalize_arrays(item, path=(*path, str(key)))
                for key, item in value.items()
            }
        if isinstance(value, list):
            array = self._numeric_array(value)
            if array is not None and array.size >= 12:
                return self._append_array(array, semantic_path=".".join(path))
            return [
                self._externalize_arrays(item, path=(*path, str(index)))
                for index, item in enumerate(value)
            ]
        return value

    @staticmethod
    def _numeric_array(value: list[Any]) -> np.ndarray | None:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if array.dtype != np.float32 or array.ndim <= 0 or not np.all(np.isfinite(array)):
            return None
        return np.ascontiguousarray(array)

    def _append_array(self, array: np.ndarray, *, semantic_path: str) -> dict[str, Any]:
        raw = array.tobytes(order="C")
        self._array_sequence += 1
        array_id = f"a{self._array_sequence:08d}"
        metadata = {
            "id": array_id,
            "dtype": "float32",
            "shape": list(array.shape),
            "offset_bytes": self._array_offset_bytes,
            "length_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_path": semantic_path,
        }
        self._arrays.write(raw)
        self._arrays_index.write(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._array_offset_bytes += len(raw)
        return {WORLD_TAPE_VECTOR_REF_KEY: metadata}

    def _flush_locked(self) -> None:
        for handle in (self._events, self._arrays_index):
            if not handle.closed:
                handle.flush()
        if not self._arrays.closed:
            self._arrays.flush()
        self._last_flush_monotonic = time.monotonic()

    def _artifact_summary(
        self,
        *,
        status: str,
        reason: str,
        include_hashes: bool,
    ) -> dict[str, Any]:
        with self._lock:
            enqueued_event_count = self._sequence
            written_event_count = self._written_event_count
            array_count = self._array_sequence
            event_counts = dict(sorted(self._counts.items()))
            writer_error = self._writer_error
        summary: dict[str, Any] = {
            "contract_id": WORLD_TAPE_CONTRACT_ID,
            "run_id": self.run_id,
            "status": status,
            "reason": str(reason),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "events_path": str(self.events_path),
            "arrays_path": str(self.arrays_path),
            "arrays_index_path": str(self.arrays_index_path),
            "event_count": written_event_count,
            "enqueued_event_count": enqueued_event_count,
            "array_count": array_count,
            "event_counts": event_counts,
            "writer_error": writer_error,
        }
        if include_hashes:
            for key, path in (
                ("events_sha256", self.events_path),
                ("arrays_sha256", self.arrays_path),
                ("arrays_index_sha256", self.arrays_index_path),
            ):
                if path.is_file():
                    summary[key] = _sha256_file(path)
        return summary

    def _write_manifest(self, *, status: str, reason: str) -> None:
        with self._io_lock:
            self._write_manifest_locked(status=status, reason=reason)

    def _write_manifest_locked(
        self,
        *,
        status: str,
        reason: str,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        manifest = {
            "contract_id": WORLD_TAPE_CONTRACT_ID,
            "run_id": self.run_id,
            "status": status,
            "reason": str(reason),
            "started_at": datetime.fromtimestamp(
                self._started_epoch, timezone.utc
            ).isoformat(),
            "updated_at": _utc_now(),
            "event_timebase": {
                "wall": "integer microseconds since recorder start",
                "media": f"integer samples at {self._sample_rate} Hz plus diagnostic seconds",
                "tie_break": "global seq",
            },
            "runtime_config": self._config,
            "runtime_config_sha256": _stable_sha256(self._config),
            "media_history": list(self._media_history),
            "artifact": artifact or self._artifact_summary(
                status=status,
                reason=reason,
                include_hashes=False,
            ),
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)


def load_world_tape_array(tape_dir: Path, reference: dict[str, Any]) -> np.ndarray:
    """Load and hash-check one array reference from an existing tape."""

    metadata = reference.get(WORLD_TAPE_VECTOR_REF_KEY, reference)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid world-tape array reference.")
    if str(metadata.get("dtype") or "") != "float32":
        raise ValueError(f"Unsupported world-tape array dtype {metadata.get('dtype')!r}.")
    offset_value = metadata.get("offset_bytes")
    length_value = metadata.get("length_bytes")
    shape_value = metadata.get("shape")
    if (
        isinstance(offset_value, bool)
        or not isinstance(offset_value, int)
        or isinstance(length_value, bool)
        or not isinstance(length_value, int)
        or not isinstance(shape_value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape_value)
    ):
        raise ValueError("Malformed world-tape array geometry.")
    try:
        offset = int(offset_value)
        length = int(length_value)
        shape = tuple(int(item) for item in shape_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed world-tape array geometry.") from exc
    if offset < 0 or offset % np.dtype(np.float32).itemsize:
        raise ValueError(f"Invalid world-tape array offset {offset}.")
    if length <= 0 or length % np.dtype(np.float32).itemsize:
        raise ValueError(f"Invalid world-tape array length {length}.")
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Invalid world-tape array shape {shape!r}.")
    expected_length = int(math.prod(shape)) * np.dtype(np.float32).itemsize
    if expected_length != length:
        raise ValueError(
            f"World-tape array shape/length mismatch ({shape!r}, {length} bytes)."
        )
    arrays_path = Path(tape_dir) / "arrays.f32"
    file_size = arrays_path.stat().st_size
    if offset + length > file_size:
        raise ValueError(f"Out-of-bounds world-tape array {metadata.get('id')!r}.")
    with arrays_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"Truncated world-tape array {metadata.get('id')!r}.")
    if hashlib.sha256(raw).hexdigest() != str(metadata.get("sha256") or ""):
        raise ValueError(f"Hash mismatch for world-tape array {metadata.get('id')!r}.")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
