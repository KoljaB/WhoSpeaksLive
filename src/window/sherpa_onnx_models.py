"""Pinned streaming-ASR metadata and secure sherpa-onnx model installation."""

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
KROKO_REQUIRED_MODEL_FILES = (
    "encoder.onnx",
    "decoder.onnx",
    "joiner.onnx",
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
    family: str = "nemotron"
    required_model_files: tuple[str, ...] = REQUIRED_MODEL_FILES

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

KROKO_SHERPA_MODEL_PRESETS = {
    "en": SherpaOnnxModelPreset(
        name="kroko-community-64l-en",
        archive_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2",
        model_dir_name="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        archive_sha256="c8676e5ff9ac2a85296e53ee0fd4d5fb1db6770e7a7647166eeafe349ade6834",
        internal_chunk_ms=640,
        recommended_feed_seconds=0.16,
        family="kroko",
        required_model_files=KROKO_REQUIRED_MODEL_FILES,
    ),
    "de": SherpaOnnxModelPreset(
        name="kroko-community-64l-de",
        archive_name="sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2",
        model_dir_name="sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06",
        archive_sha256="9e27b783c20e67b0d0f13a258c1861fce199917c969d9176a438bee38df64962",
        internal_chunk_ms=640,
        recommended_feed_seconds=0.16,
        family="kroko",
        required_model_files=KROKO_REQUIRED_MODEL_FILES,
    ),
    "es": SherpaOnnxModelPreset(
        name="kroko-community-64l-es",
        archive_name="sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06.tar.bz2",
        model_dir_name="sherpa-onnx-streaming-zipformer-es-kroko-2025-08-06",
        archive_sha256="31b2230a95d23290b308b393da930015a4b2105cb3abb9367aed35f7fcf29cf1",
        internal_chunk_ms=640,
        recommended_feed_seconds=0.16,
        family="kroko",
        required_model_files=KROKO_REQUIRED_MODEL_FILES,
    ),
    "fr": SherpaOnnxModelPreset(
        name="kroko-community-64l-fr",
        archive_name="sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06.tar.bz2",
        model_dir_name="sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06",
        archive_sha256="e6ffd3dc43725cd6c8137b05c739f15607d0df946b9b90eb141e10059efca024",
        internal_chunk_ms=640,
        recommended_feed_seconds=0.16,
        family="kroko",
        required_model_files=KROKO_REQUIRED_MODEL_FILES,
    ),
}


def normalize_sherpa_onnx_preview_model_preset(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    preset = SHERPA_ONNX_PREVIEW_MODEL_PRESET_ALIASES.get(normalized, normalized)
    if preset not in SHERPA_ONNX_PREVIEW_MODEL_PRESETS:
        allowed = ", ".join(SHERPA_ONNX_PREVIEW_MODEL_PRESETS)
        raise ValueError(f"invalid Nemotron preview model preset {value!r}; choose one of: {allowed}")
    return preset


def sherpa_onnx_model_preset(value: object) -> SherpaOnnxModelPreset:
    if isinstance(value, SherpaOnnxModelPreset):
        return value
    return SHERPA_ONNX_PREVIEW_MODEL_PRESETS[normalize_sherpa_onnx_preview_model_preset(value)]


def kroko_sherpa_model_preset(language: object) -> SherpaOnnxModelPreset:
    code = str(language or "en").strip().lower().replace("_", "-").split("-", 1)[0]
    try:
        return KROKO_SHERPA_MODEL_PRESETS[code]
    except KeyError as exc:
        allowed = ", ".join(KROKO_SHERPA_MODEL_PRESETS)
        raise ValueError(f"official sherpa-onnx Kroko models support: {allowed}; got {language!r}") from exc


def default_sherpa_onnx_model_dir(preset: object = DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET) -> Path:
    return SHERPA_ONNX_MODEL_DIR / sherpa_onnx_model_preset(preset).model_dir_name


def default_kroko_sherpa_model_dir(language: object) -> Path:
    return SHERPA_ONNX_MODEL_DIR / kroko_sherpa_model_preset(language).model_dir_name


def missing_sherpa_onnx_model_files(model_dir: Path) -> list[str]:
    directory = Path(model_dir).expanduser()
    if all((directory / name).is_file() for name in REQUIRED_MODEL_FILES):
        return []
    if all((directory / name).is_file() for name in KROKO_REQUIRED_MODEL_FILES):
        return []
    return [
        "encoder.int8.onnx or encoder.onnx",
        "decoder.int8.onnx or decoder.onnx",
        "joiner.int8.onnx or joiner.onnx",
        "tokens.txt",
    ]


def sherpa_onnx_model_files(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    directory = validate_sherpa_onnx_model_dir(model_dir)
    suffix = ".int8.onnx" if (directory / "encoder.int8.onnx").is_file() else ".onnx"
    return (
        directory / f"encoder{suffix}",
        directory / f"decoder{suffix}",
        directory / f"joiner{suffix}",
        directory / "tokens.txt",
    )


def validate_sherpa_onnx_model_dir(model_dir: Path) -> Path:
    directory = Path(model_dir).expanduser().resolve()
    missing = missing_sherpa_onnx_model_files(directory)
    if missing:
        raise RuntimeError(
            f"sherpa-onnx model is incomplete at {directory}: missing {', '.join(missing)}."
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
    """Download, verify, and atomically install one pinned sherpa-onnx archive."""

    selected = sherpa_onnx_model_preset(preset)
    final_dir = Path(target_dir).expanduser() if target_dir is not None else default_sherpa_onnx_model_dir(selected.name)
    final_dir = final_dir.resolve()
    try:
        return validate_sherpa_onnx_model_dir(final_dir)
    except RuntimeError:
        pass

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(final_dir.parent).free < MINIMUM_FREE_SPACE_BYTES:
        raise RuntimeError(f"Not enough free disk space to install ASR model at {final_dir.parent}; at least 2 GB is required.")

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
                f"ASR archive SHA-256 mismatch for {selected.archive_name}: "
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
                    f"ASR model directory already exists but is incomplete: {final_dir}. "
                    "Remove it manually before retrying the download."
                )
            os.replace(extracted, final_dir)
        return validate_sherpa_onnx_model_dir(final_dir)
    finally:
        archive_path.unlink(missing_ok=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def ensure_kroko_sherpa_model(
    language: object,
    *,
    target_dir: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    selected = kroko_sherpa_model_preset(language)
    return ensure_sherpa_onnx_model(
        selected,
        target_dir=target_dir or default_kroko_sherpa_model_dir(language),
        progress=progress,
    )
