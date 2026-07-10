"""Pinned Nemotron model metadata and secure sherpa-onnx model installation."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from paths import SHERPA_ONNX_MODEL_DIR


MODEL_RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
REQUIRED_MODEL_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
MINIMUM_FREE_SPACE_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class SherpaOnnxModelPreset:
    name: str
    archive_name: str
    model_dir_name: str
    archive_sha256: str
    internal_chunk_ms: int
    recommended_feed_seconds: float
    recommended_interval_seconds: float = 0.10
    recommended_min_audio_seconds: float = 0.56
    startup_timeout_seconds: float = 30.0

    @property
    def archive_url(self) -> str:
        return f"{MODEL_RELEASE_URL}/{self.archive_name}"


SHERPA_ONNX_PREVIEW_MODEL_PRESETS = {
    "nemotron-3.5-160ms-int8": SherpaOnnxModelPreset(
        name="nemotron-3.5-160ms-int8",
        archive_name="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2",
        model_dir_name="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11",
        archive_sha256="a81909a1780d84cff16d73c15e13e67d9d81d8839faf14870d507d8499f7a61a",
        internal_chunk_ms=160,
        recommended_feed_seconds=0.08,
        recommended_min_audio_seconds=0.16,
    ),
    "nemotron-3.5-560ms-int8": SherpaOnnxModelPreset(
        name="nemotron-3.5-560ms-int8",
        archive_name="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
        model_dir_name="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        archive_sha256="c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
        internal_chunk_ms=560,
        recommended_feed_seconds=0.16,
    ),
}
SHERPA_ONNX_PREVIEW_MODEL_PRESET_ALIASES = {
    "160": "nemotron-3.5-160ms-int8",
    "160ms": "nemotron-3.5-160ms-int8",
    "nemotron-160": "nemotron-3.5-160ms-int8",
    "560": "nemotron-3.5-560ms-int8",
    "560ms": "nemotron-3.5-560ms-int8",
    "nemotron": "nemotron-3.5-560ms-int8",
    "nemotron-560": "nemotron-3.5-560ms-int8",
}
DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET = "nemotron-3.5-560ms-int8"


def normalize_sherpa_onnx_preview_model_preset(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    preset = SHERPA_ONNX_PREVIEW_MODEL_PRESET_ALIASES.get(normalized, normalized)
    if preset not in SHERPA_ONNX_PREVIEW_MODEL_PRESETS:
        allowed = ", ".join(SHERPA_ONNX_PREVIEW_MODEL_PRESETS)
        raise ValueError(f"invalid Nemotron preview model preset {value!r}; choose one of: {allowed}")
    return preset


def sherpa_onnx_model_preset(value: object) -> SherpaOnnxModelPreset:
    return SHERPA_ONNX_PREVIEW_MODEL_PRESETS[normalize_sherpa_onnx_preview_model_preset(value)]


def default_sherpa_onnx_model_dir(preset: object = DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET) -> Path:
    return SHERPA_ONNX_MODEL_DIR / sherpa_onnx_model_preset(preset).model_dir_name


def missing_sherpa_onnx_model_files(model_dir: Path) -> list[str]:
    directory = Path(model_dir).expanduser()
    return [name for name in REQUIRED_MODEL_FILES if not (directory / name).is_file()]


def validate_sherpa_onnx_model_dir(model_dir: Path) -> Path:
    directory = Path(model_dir).expanduser().resolve()
    missing = missing_sherpa_onnx_model_files(directory)
    if missing:
        raise RuntimeError(
            f"Nemotron model is incomplete at {directory}: missing {', '.join(missing)}."
        )
    return directory


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Refusing unsafe model archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing linked model archive member: {member.name}")
    return members


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, progress: Callable[[int, int | None], None] | None = None) -> None:
    received = 0
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
        raw_length = response.headers.get("Content-Length")
        total = int(raw_length) if raw_length and raw_length.isdigit() else None
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            received += len(chunk)
            if progress is not None:
                progress(received, total)


def _acquire_lock(lock_path: Path, timeout_seconds: float = 1800.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Nemotron model download lock: {lock_path}")
            time.sleep(0.2)


def ensure_sherpa_onnx_model(
    preset: object,
    *,
    target_dir: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download, verify, and atomically install one pinned Nemotron archive."""

    selected = sherpa_onnx_model_preset(preset)
    final_dir = Path(target_dir).expanduser() if target_dir is not None else default_sherpa_onnx_model_dir(selected.name)
    final_dir = final_dir.resolve()
    try:
        return validate_sherpa_onnx_model_dir(final_dir)
    except RuntimeError:
        pass

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(final_dir.parent).free < MINIMUM_FREE_SPACE_BYTES:
        raise RuntimeError(
            f"Not enough free disk space to install Nemotron at {final_dir.parent}; at least 2 GB is required."
        )

    lock_path = final_dir.parent / f".{selected.model_dir_name}.lock"
    lock_fd = _acquire_lock(lock_path)
    archive_path = final_dir.parent / f"{selected.archive_name}.part"
    try:
        try:
            return validate_sherpa_onnx_model_dir(final_dir)
        except RuntimeError:
            pass
        archive_path.unlink(missing_ok=True)
        _download(selected.archive_url, archive_path, progress)
        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != selected.archive_sha256:
            raise RuntimeError(
                f"Nemotron archive SHA-256 mismatch for {selected.archive_name}: "
                f"expected {selected.archive_sha256}, got {actual_sha256}."
            )
        with tempfile.TemporaryDirectory(prefix=f".{selected.model_dir_name}.", dir=final_dir.parent) as temporary:
            temporary_root = Path(temporary)
            with tarfile.open(archive_path, "r:bz2") as archive:
                archive.extractall(temporary_root, members=_safe_archive_members(archive), filter="data")
            extracted = temporary_root / selected.model_dir_name
            validate_sherpa_onnx_model_dir(extracted)
            if final_dir.exists():
                raise RuntimeError(
                    f"Nemotron model directory already exists but is incomplete: {final_dir}. "
                    "Remove it manually before retrying the download."
                )
            os.replace(extracted, final_dir)
        return validate_sherpa_onnx_model_dir(final_dir)
    finally:
        archive_path.unlink(missing_ok=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
