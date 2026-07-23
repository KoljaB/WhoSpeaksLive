"""Fail-closed evidence contract for real GUI live-speaker promotion runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
from collections.abc import Mapping
from typing import Any, Iterable


REAL_GUI_E2E_CONTRACT_ID = "whospeaks.real_gui_live_speaker_e2e.v2"
PRODUCTION_CHAMPION_STATUS = "REAL_GUI_LIVE_E2E_VERIFIED_CHAMPION"
_WORLD_TAPE_CONTRACT_ID = "whospeaks.live_world_tape.v1"
FINAL_TRANSCRIPT_DOM_SNAPSHOT_SCHEMA_VERSION = "final_clustering_dom_snapshot_v1"
FINAL_TRANSCRIPT_DOM_SNAPSHOT_CAPTURE_SURFACE = (
    "visible_chrome_final_transcript_dom_after_done"
)
FINAL_TRANSCRIPT_DOM_SNAPSHOT_BINDINGS_SCHEMA_VERSION = (
    "final_transcript_dom_snapshot_bindings_v1"
)
FINAL_TRANSCRIPT_DOM_SNAPSHOT_ATTESTATION_SCHEMA_VERSION = (
    "final_transcript_dom_snapshot_attestation_v1"
)
_SECRET_NAME_PARTS = {"token", "password", "secret", "cookie", "authorization"}
_EVIDENCE_ONLY_CONFIG_KEYS = {
    "browser_live_e2e_candidate_artifact",
    "browser_live_observation_output",
    "live_speaker_world_tape_output",
    "validation_output",
    "validation_trace_output",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write one JSON artifact atomically in its final directory."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def final_transcript_dom_snapshot_output_path(
    browser_observation_path: Path,
) -> Path:
    """Return the deterministic sibling path used for the final DOM snapshot."""

    observation = Path(browser_observation_path).resolve()
    return observation.with_name(
        f"{observation.stem}.final_transcript_dom_snapshot.json"
    )


def _exact_object_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    errors: list[str],
) -> None:
    observed = {str(key) for key in value}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        errors.append(f"{label} keys differ (missing={missing}, extra={extra})")


def validate_final_transcript_dom_snapshot(
    payload: Any,
    expected_binding: Mapping[str, Any],
) -> list[str]:
    """Validate an exact, browser-originated post-done transcript snapshot."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["final transcript DOM snapshot root is not an object"]
    _exact_object_keys(
        payload,
        {
            "schema_version",
            "world_tape_run_id",
            "capture_surface",
            "captured_after_done",
            "source_tree_sha256",
            "runtime_config_sha256",
            "media",
            "browser",
            "rows",
        },
        "snapshot",
        errors,
    )
    if payload.get("schema_version") != FINAL_TRANSCRIPT_DOM_SNAPSHOT_SCHEMA_VERSION:
        errors.append("final transcript DOM snapshot schema_version is invalid")
    if payload.get("capture_surface") != FINAL_TRANSCRIPT_DOM_SNAPSHOT_CAPTURE_SURFACE:
        errors.append("final transcript DOM snapshot capture_surface is invalid")
    if payload.get("captured_after_done") is not True:
        errors.append("final transcript DOM snapshot was not captured after done")

    expected_run_id = str(expected_binding.get("world_tape_run_id") or "").strip()
    run_id = str(payload.get("world_tape_run_id") or "").strip()
    if not expected_run_id or run_id != expected_run_id:
        errors.append("final transcript DOM snapshot World Tape run id mismatch")
    expected_source_hash = str(expected_binding.get("source_tree_sha256") or "").strip()
    source_hash = str(payload.get("source_tree_sha256") or "").strip()
    if not _is_sha256(expected_source_hash) or source_hash != expected_source_hash:
        errors.append("final transcript DOM snapshot source tree mismatch")
    expected_runtime_hash = str(
        expected_binding.get("runtime_config_sha256") or ""
    ).strip()
    runtime_hash = str(payload.get("runtime_config_sha256") or "").strip()
    if not _is_sha256(expected_runtime_hash) or runtime_hash != expected_runtime_hash:
        errors.append("final transcript DOM snapshot runtime configuration mismatch")

    expected_media = (
        expected_binding.get("media")
        if isinstance(expected_binding.get("media"), Mapping)
        else {}
    )
    media = payload.get("media") if isinstance(payload.get("media"), dict) else {}
    _exact_object_keys(
        media,
        {
            "video_id",
            "source_audio_path",
            "source_audio_sha256",
            "source_audio_size_bytes",
            "audio_sha256",
            "decoded_pcm_sha256",
            "decoded_samples",
            "sample_rate",
            "duration_seconds",
        },
        "snapshot.media",
        errors,
    )
    for key in (
        "video_id",
        "source_audio_path",
        "source_audio_sha256",
        "audio_sha256",
        "decoded_pcm_sha256",
    ):
        actual = str(media.get(key) or "").strip()
        expected = str(expected_media.get(key) or "").strip()
        if not expected or actual != expected:
            errors.append(f"final transcript DOM snapshot media.{key} mismatch")
        if key.endswith("sha256") and not _is_sha256(actual):
            errors.append(f"final transcript DOM snapshot media.{key} is invalid")
    source_audio_size = media.get("source_audio_size_bytes")
    if (
        isinstance(source_audio_size, bool)
        or not isinstance(source_audio_size, int)
        or source_audio_size <= 0
    ):
        errors.append(
            "final transcript DOM snapshot media.source_audio_size_bytes is invalid"
        )
    if source_audio_size != expected_media.get("source_audio_size_bytes"):
        errors.append(
            "final transcript DOM snapshot media.source_audio_size_bytes mismatch"
        )
    if str(media.get("source_audio_sha256") or "") != str(
        media.get("audio_sha256") or ""
    ):
        errors.append(
            "final transcript DOM snapshot source and legacy audio hashes differ"
        )
    snapshot_source_path = Path(str(media.get("source_audio_path") or ""))
    if snapshot_source_path.is_file():
        if file_sha256(snapshot_source_path) != media.get("source_audio_sha256"):
            errors.append(
                "final transcript DOM snapshot source audio bytes do not match"
            )
        if int(snapshot_source_path.stat().st_size) != source_audio_size:
            errors.append(
                "final transcript DOM snapshot source audio size does not match"
            )
    decoded_samples = media.get("decoded_samples")
    sample_rate = media.get("sample_rate")
    duration_seconds = media.get("duration_seconds")
    if (
        isinstance(decoded_samples, bool)
        or not isinstance(decoded_samples, int)
        or decoded_samples < 0
    ):
        errors.append("final transcript DOM snapshot media.decoded_samples is invalid")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        errors.append("final transcript DOM snapshot media.sample_rate is invalid")
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = math.nan
    if not math.isfinite(duration) or duration < 0.0:
        errors.append("final transcript DOM snapshot media.duration_seconds is invalid")
    for key in ("decoded_samples", "sample_rate"):
        if media.get(key) != expected_media.get(key):
            errors.append(f"final transcript DOM snapshot media.{key} mismatch")
    try:
        expected_duration = float(expected_media.get("duration_seconds"))
    except (TypeError, ValueError):
        expected_duration = math.nan
    if (
        not math.isfinite(expected_duration)
        or not math.isfinite(duration)
        or abs(duration - expected_duration) > 1e-9
    ):
        errors.append("final transcript DOM snapshot media.duration_seconds mismatch")
    if (
        isinstance(decoded_samples, int)
        and not isinstance(decoded_samples, bool)
        and isinstance(sample_rate, int)
        and not isinstance(sample_rate, bool)
        and sample_rate > 0
        and math.isfinite(duration)
        and abs(duration - (decoded_samples / sample_rate)) > 1e-9
    ):
        errors.append(
            "final transcript DOM snapshot decoded duration is inconsistent"
        )

    browser = payload.get("browser") if isinstance(payload.get("browser"), dict) else {}
    _exact_object_keys(
        browser,
        {"visibility_state", "has_focus", "webdriver"},
        "snapshot.browser",
        errors,
    )
    if browser.get("visibility_state") != "visible":
        errors.append("Chrome was not visible for the final transcript DOM snapshot")
    if browser.get("has_focus") is not True:
        errors.append("Chrome was not focused for the final transcript DOM snapshot")
    if browser.get("webdriver") is not False:
        errors.append("WebDriver was present for the final transcript DOM snapshot")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        errors.append("final transcript DOM snapshot rows is not a list")
        rows = []
    if len(rows) > 100_000:
        errors.append("final transcript DOM snapshot has too many rows")
    indexes: list[int] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"final transcript DOM snapshot row {position} is not an object")
            continue
        _exact_object_keys(
            row,
            {"index", "text", "assigned_speaker"},
            f"snapshot.rows[{position}]",
            errors,
        )
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            errors.append(f"final transcript DOM snapshot row {position} index is invalid")
        else:
            indexes.append(index)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"final transcript DOM snapshot row {position} text is empty")
        elif len(text) > 1_000_000:
            errors.append(f"final transcript DOM snapshot row {position} text is too long")
        speaker = row.get("assigned_speaker")
        if speaker is not None and (
            not isinstance(speaker, str)
            or not speaker.strip()
            or len(speaker) > 512
        ):
            errors.append(
                f"final transcript DOM snapshot row {position} assigned_speaker is invalid"
            )
    if indexes != list(range(len(rows))):
        errors.append(
            "final transcript DOM snapshot row indexes are not unique, ordered, and contiguous"
        )
    return sorted(set(errors))


def write_final_transcript_dom_snapshot(
    *,
    browser_observation_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write a validated snapshot and return its attestation entry."""

    snapshot_path = final_transcript_dom_snapshot_output_path(browser_observation_path)
    atomic_write_json(snapshot_path, payload)
    snapshot_hash = file_sha256(snapshot_path)
    media = payload.get("media") if isinstance(payload.get("media"), Mapping) else {}
    return {
        "schema_version": FINAL_TRANSCRIPT_DOM_SNAPSHOT_ATTESTATION_SCHEMA_VERSION,
        "snapshot_schema_version": FINAL_TRANSCRIPT_DOM_SNAPSHOT_SCHEMA_VERSION,
        "capture_surface": FINAL_TRANSCRIPT_DOM_SNAPSHOT_CAPTURE_SURFACE,
        "captured_after_done": True,
        "path": str(snapshot_path),
        "sha256": snapshot_hash,
        "world_tape_run_id": str(payload.get("world_tape_run_id") or ""),
        "source_tree_sha256": str(payload.get("source_tree_sha256") or ""),
        "runtime_config_sha256": str(
            payload.get("runtime_config_sha256") or ""
        ),
        "media": {
            "video_id": str(media.get("video_id") or ""),
            "source_audio_path": str(media.get("source_audio_path") or ""),
            "source_audio_sha256": str(media.get("source_audio_sha256") or ""),
            "source_audio_size_bytes": media.get("source_audio_size_bytes"),
            "audio_sha256": str(media.get("audio_sha256") or ""),
            "decoded_pcm_sha256": str(media.get("decoded_pcm_sha256") or ""),
            "decoded_samples": media.get("decoded_samples"),
            "sample_rate": media.get("sample_rate"),
            "duration_seconds": media.get("duration_seconds"),
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    suffixes = {".py", ".js", ".css", ".html"}
    source_roots = [
        root / "src" / "window",
        root / "src" / "embeddings",
        root / "src" / "speakers",
        root / "src" / "common",
        root / "src" / "realtime",
    ]
    files: list[Path] = []
    for source_root in source_roots:
        if source_root.is_dir():
            files.extend(
                path for path in source_root.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            )
    for path in sorted(files, key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def live_runtime_config(args: Any) -> dict[str, Any]:
    """Capture the complete non-secret startup config that can affect a run.

    Evidence destinations are excluded because they identify where a run is
    recorded, not how it behaves.  An explicit deny-list is safer than the old
    live-speaker prefix allow-list: final-sentence, media, queue, browser and
    session settings can all alter the user-visible live result.
    """
    # ``WindowConfig`` is both a dataclass and a Mapping.  Looking at its
    # ``__dict__`` exposes only the private ``_values`` field and used to turn
    # the effective runtime configuration into an empty attestation.  Prefer
    # the public Mapping contract and retain Namespace/object support.
    values = dict(args) if isinstance(args, Mapping) else vars(args)
    config: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        name = str(key)
        lowered = name.lower()
        if name in _EVIDENCE_ONLY_CONFIG_KEYS:
            continue
        name_parts = set(lowered.replace("-", "_").split("_"))
        if name_parts.intersection(_SECRET_NAME_PARTS) or "api_key" in lowered:
            continue
        config[name] = _jsonable(value)
    return config


def _capture_integrity_snapshot(attestation: dict[str, Any]) -> dict[str, Any]:
    canonical = attestation.get("canonical")
    media = attestation.get("media")
    code = attestation.get("code")
    artifact = attestation.get("candidate_artifact")
    return {
        "runtime_config_sha256": attestation.get("runtime_config_sha256"),
        "canonical_sha256": canonical.get("sha256") if isinstance(canonical, dict) else None,
        "canonical_path": canonical.get("path") if isinstance(canonical, dict) else None,
        "audio_sha256": media.get("audio_sha256") if isinstance(media, dict) else None,
        "source_audio_path": media.get("source_audio_path") if isinstance(media, dict) else None,
        "source_audio_size_bytes": (
            media.get("source_audio_size_bytes") if isinstance(media, dict) else None
        ),
        "video_sha256": media.get("video_sha256") if isinstance(media, dict) else None,
        "video_path": media.get("video_path") if isinstance(media, dict) else None,
        "video_id": media.get("video_id") if isinstance(media, dict) else None,
        "source_tree_sha256": code.get("source_tree_sha256") if isinstance(code, dict) else None,
        "candidate_artifact_sha256": (
            artifact.get("sha256") if isinstance(artifact, dict) else None
        ),
    }


def seal_real_gui_e2e_attestation(
    started: dict[str, Any],
    finished: dict[str, Any],
) -> dict[str, Any]:
    """Bind one startup snapshot to its finish snapshot for drift detection."""
    result = dict(started)
    result["capture_integrity"] = {
        "start": _capture_integrity_snapshot(started),
        "finish": _capture_integrity_snapshot(finished),
    }
    return result


def _canonical_max_end(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            media = payload.get("media")
            if isinstance(media, dict) and media.get("duration_sec") is not None:
                return float(media["duration_sec"])
            segments = payload.get("segments")
        else:
            segments = payload
        if not isinstance(segments, list):
            return None
        return max(
            (
                float(item.get("end", item.get("end_sec")))
                for item in segments
                if isinstance(item, dict) and ("end" in item or "end_sec" in item)
            ),
            default=None,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def build_real_gui_e2e_attestation(
    *,
    root: Path,
    args: Any,
    media: Any,
) -> dict[str, Any]:
    canonical_path = Path(args.validation_canonical).resolve()
    audio_path = Path(media.audio_file).resolve()
    video_path = Path(media.video_file).resolve()
    candidate_value = getattr(args, "browser_live_e2e_candidate_artifact", None)
    candidate_path = Path(candidate_value).resolve() if candidate_value else None
    config = live_runtime_config(args)
    source_hash = source_tree_sha256(root)
    source_audio_hash = file_sha256(audio_path)
    source_audio_size = int(audio_path.stat().st_size) if audio_path.is_file() else None
    return {
        "contract_id": REAL_GUI_E2E_CONTRACT_ID,
        "capture_surface": "normal_gui_server_rendered_browser_dom",
        "runtime_config": config,
        "runtime_config_sha256": stable_sha256(config),
        "canonical": {
            "path": str(canonical_path),
            "sha256": file_sha256(canonical_path),
            "max_end_seconds": _canonical_max_end(canonical_path),
        },
        "media": {
            "url": str(media.url),
            "video_id": str(media.video_id),
            "audio_path": str(audio_path),
            "audio_sha256": source_audio_hash,
            "source_audio_path": str(audio_path),
            "source_audio_sha256": source_audio_hash,
            "source_audio_size_bytes": source_audio_size,
            "video_path": str(video_path),
            "video_sha256": file_sha256(video_path),
        },
        "code": {
            "source_tree_sha256": source_hash,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "candidate_artifact": {
            "path": str(candidate_path) if candidate_path else None,
            "sha256": file_sha256(candidate_path),
        },
    }


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def validate_real_gui_e2e_observation(payload: dict[str, Any]) -> list[str]:
    """Return every reason why an observation is not promotion-grade evidence."""
    errors: list[str] = []
    attestation = payload.get("attestation")
    summary = payload.get("summary")
    samples = payload.get("samples")
    if not isinstance(attestation, dict):
        return ["missing server-generated E2E attestation"]
    if attestation.get("contract_id") != REAL_GUI_E2E_CONTRACT_ID:
        errors.append("wrong E2E contract id")
    if attestation.get("capture_surface") != "normal_gui_server_rendered_browser_dom":
        errors.append("capture did not use the normal GUI server DOM surface")
    if not isinstance(summary, dict):
        errors.append("missing score summary")
        summary = {}
    if not isinstance(samples, list) or not samples:
        errors.append("missing browser DOM samples")
        samples = []
    try:
        sample_count = int(summary.get("sample_count") or 0)
    except (TypeError, ValueError):
        sample_count = 0
    if sample_count < 100:
        errors.append("fewer than 100 browser samples")
    if sample_count != len(samples):
        errors.append("summary sample_count does not equal the serialized sample list")
    if any(not isinstance(item, dict) for item in samples):
        errors.append("browser sample list contains non-object entries")
    try:
        primary_score = float(summary.get("strict_browser_live_score"))
    except (TypeError, ValueError):
        primary_score = math.nan
    if not math.isfinite(primary_score) or not 0.0 <= primary_score <= 1.0:
        errors.append("strict browser live score is missing or invalid")

    canonical = attestation.get("canonical") if isinstance(attestation, dict) else {}
    media = attestation.get("media") if isinstance(attestation, dict) else {}
    code = attestation.get("code") if isinstance(attestation, dict) else {}
    if not isinstance(canonical, dict) or not _is_sha256(canonical.get("sha256")):
        errors.append("canonical reference is not hash-bound")
    if not isinstance(media, dict) or not _is_sha256(media.get("audio_sha256")):
        errors.append("source audio is not hash-bound")
    if not isinstance(code, dict) or not _is_sha256(code.get("source_tree_sha256")):
        errors.append("executed source tree is not hash-bound")
    if isinstance(canonical, dict):
        canonical_path = Path(str(canonical.get("path") or ""))
        if not canonical_path.is_file():
            errors.append("canonical reference path does not exist")
        elif file_sha256(canonical_path) != canonical.get("sha256"):
            errors.append("canonical reference bytes do not match the attested hash")
    if isinstance(media, dict):
        source_path = Path(
            str(media.get("source_audio_path") or media.get("audio_path") or "")
        )
        source_hash = str(
            media.get("source_audio_sha256") or media.get("audio_sha256") or ""
        )
        if not source_path.is_file():
            errors.append("source audio path does not exist")
        else:
            if file_sha256(source_path) != source_hash:
                errors.append("source audio bytes do not match the attested hash")
            try:
                source_size = int(media.get("source_audio_size_bytes"))
            except (TypeError, ValueError):
                source_size = -1
            if source_size != int(source_path.stat().st_size):
                errors.append("source audio size does not match the attestation")
        if source_hash != str(media.get("audio_sha256") or ""):
            errors.append("explicit and legacy source audio hashes differ")
        video_path = Path(str(media.get("video_path") or ""))
        video_hash = media.get("video_sha256")
        if video_path.is_file() and file_sha256(video_path) != video_hash:
            errors.append("video bytes do not match the attested hash")
    canonical = canonical if isinstance(canonical, dict) else {}
    media = media if isinstance(media, dict) else {}
    code = code if isinstance(code, dict) else {}

    capture_integrity = attestation.get("capture_integrity")
    if not isinstance(capture_integrity, dict):
        errors.append("missing start/finish capture integrity")
    else:
        started = capture_integrity.get("start")
        finished = capture_integrity.get("finish")
        expected_started = _capture_integrity_snapshot(attestation)
        if not isinstance(started, dict) or started != expected_started:
            errors.append("startup capture integrity does not match attestation")
        if not isinstance(finished, dict):
            errors.append("missing finish capture integrity")
        elif isinstance(started, dict) and finished != started:
            errors.append("runtime, source tree, artifact, or media drifted during capture")

    world_tape = attestation.get("world_tape")
    if not isinstance(world_tape, dict):
        errors.append("missing complete live-speaker World Tape")
    else:
        if world_tape.get("contract_id") != _WORLD_TAPE_CONTRACT_ID:
            errors.append("live-speaker World Tape contract id is invalid")
        if world_tape.get("status") != "complete":
            errors.append("live-speaker World Tape is incomplete")
        if not world_tape.get("run_id"):
            errors.append("live-speaker World Tape run id is missing")
        if world_tape.get("writer_error"):
            errors.append("live-speaker World Tape writer failed")
        try:
            world_tape_event_count = int(world_tape.get("event_count") or 0)
        except (TypeError, ValueError):
            world_tape_event_count = 0
        if world_tape_event_count <= 0:
            errors.append("live-speaker World Tape contains no events")
        try:
            world_tape_enqueued_count = int(
                world_tape.get("enqueued_event_count") or 0
            )
        except (TypeError, ValueError):
            world_tape_enqueued_count = -1
        if world_tape_enqueued_count != world_tape_event_count:
            errors.append("live-speaker World Tape did not persist every enqueued event")
        for key in ("events_sha256", "arrays_sha256", "arrays_index_sha256"):
            if not _is_sha256(world_tape.get(key)):
                errors.append(f"live-speaker World Tape has missing or invalid {key}")
        for path_key, hash_key in (
            ("events_path", "events_sha256"),
            ("arrays_path", "arrays_sha256"),
            ("arrays_index_path", "arrays_index_sha256"),
        ):
            artifact_path = Path(str(world_tape.get(path_key) or ""))
            if not artifact_path.is_file():
                errors.append(f"live-speaker World Tape {path_key} does not exist")
            elif file_sha256(artifact_path) != world_tape.get(hash_key):
                errors.append(f"live-speaker World Tape {hash_key} does not match bytes")
        world_tape_runtime_hash = str(
            world_tape.get("runtime_config_sha256") or ""
        )
        if not _is_sha256(world_tape_runtime_hash):
            errors.append("live-speaker World Tape runtime configuration hash is invalid")
        tape_media = (
            world_tape.get("media")
            if isinstance(world_tape.get("media"), dict)
            else {}
        )
        if not tape_media:
            errors.append("live-speaker World Tape final media binding is missing")
        else:
            for key in (
                "video_id",
                "source_audio_path",
                "source_audio_sha256",
                "source_audio_size_bytes",
                "audio_sha256",
            ):
                if tape_media.get(key) != media.get(key):
                    errors.append(
                        f"live-speaker World Tape media.{key} differs from browser attestation"
                    )
            try:
                decoded_samples = int(tape_media.get("decoded_samples"))
                decoded_rate = int(tape_media.get("sample_rate"))
                decoded_duration = float(tape_media.get("duration_seconds"))
            except (TypeError, ValueError):
                decoded_samples = -1
                decoded_rate = -1
                decoded_duration = math.nan
            if (
                decoded_samples <= 0
                or decoded_rate <= 0
                or not math.isfinite(decoded_duration)
                or abs(decoded_duration - decoded_samples / decoded_rate) > 1e-9
            ):
                errors.append("live-speaker World Tape decoded media duration is inconsistent")

    playback = [float(item.get("playback_time", -1.0)) for item in samples if isinstance(item, dict)]
    wall = [float(item.get("wall_time", 0.0)) for item in samples if isinstance(item, dict)]
    if playback:
        playback_span = max(playback) - min(playback)
        wall_span = max(wall) - min(wall) if wall else 0.0
        canonical_end = float(canonical.get("max_end_seconds") or 0.0) if isinstance(canonical, dict) else 0.0
        if min(playback) > 0.5:
            errors.append("capture did not start near playback time zero")
        if canonical_end > 0.0 and max(playback) < canonical_end - 1.0:
            errors.append("capture did not cover the complete canonical media")
        tape_media_for_coverage = (
            world_tape.get("media")
            if isinstance(world_tape, dict)
            and isinstance(world_tape.get("media"), dict)
            else {}
        )
        try:
            decoded_duration_for_coverage = float(
                tape_media_for_coverage.get("duration_seconds")
            )
        except (TypeError, ValueError):
            decoded_duration_for_coverage = 0.0
        if (
            decoded_duration_for_coverage > 0.0
            and max(playback) < decoded_duration_for_coverage - 1.0
        ):
            errors.append("capture did not cover the complete decoded source media")
        if playback_span > 1.0 and wall_span < playback_span * 0.95:
            errors.append("playback ran faster than wall clock")
        ordered_playback = sorted(set(playback))
        gaps = [right - left for left, right in zip(ordered_playback, ordered_playback[1:]) if right > left]
        if gaps and percentile(gaps, 0.50) > 0.15:
            errors.append("median browser sampling gap exceeds 150 ms")
        if gaps and percentile(gaps, 0.99) > 0.35:
            errors.append("p99 browser sampling gap exceeds 350 ms")

    dict_samples = [item for item in samples if isinstance(item, dict)]
    if dict_samples:
        if any(bool(item.get("browser_webdriver")) for item in dict_samples):
            errors.append("automation/WebDriver browser detected")
        user_agents = {str(item.get("browser_user_agent") or "") for item in dict_samples}
        if not user_agents or any("Chrome/" not in value for value in user_agents):
            errors.append("browser is not attested as Chrome")
        if any(bool(item.get("fast_processing")) for item in dict_samples):
            errors.append("Fast processing was enabled")
        rates = [float(item.get("playback_rate", 0.0)) for item in dict_samples]
        if any(abs(value - 1.0) > 0.01 for value in rates):
            errors.append("playback rate was not 1.0")
        visible = sum(str(item.get("browser_visibility_state")) == "visible" for item in dict_samples)
        if visible / len(dict_samples) < 0.95:
            errors.append("Chrome tab was visible for less than 95% of samples")

    config = attestation.get("runtime_config") if isinstance(attestation, dict) else None
    if not isinstance(config, dict) or not config:
        errors.append("runtime configuration is missing or empty")
        expected_config_hash = None
    else:
        expected_config_hash = stable_sha256(config)
    if not expected_config_hash or attestation.get("runtime_config_sha256") != expected_config_hash:
        errors.append("runtime configuration hash mismatch")
    if isinstance(world_tape, dict) and world_tape.get(
        "runtime_config_sha256"
    ) != attestation.get("runtime_config_sha256"):
        errors.append("World Tape and browser runtime configuration hashes differ")

    dom_seal = attestation.get("final_transcript_dom_snapshot")
    if not isinstance(dom_seal, dict):
        errors.append("missing authenticated post-done final transcript DOM snapshot")
    else:
        if dom_seal.get(
            "schema_version"
        ) != FINAL_TRANSCRIPT_DOM_SNAPSHOT_ATTESTATION_SCHEMA_VERSION:
            errors.append("final transcript DOM seal schema is invalid")
        if dom_seal.get(
            "snapshot_schema_version"
        ) != FINAL_TRANSCRIPT_DOM_SNAPSHOT_SCHEMA_VERSION:
            errors.append("final transcript DOM seal snapshot schema is invalid")
        if dom_seal.get(
            "capture_surface"
        ) != FINAL_TRANSCRIPT_DOM_SNAPSHOT_CAPTURE_SURFACE:
            errors.append("final transcript DOM seal capture surface is invalid")
        if dom_seal.get("captured_after_done") is not True:
            errors.append("final transcript DOM seal is not post-done")
        snapshot_path = Path(str(dom_seal.get("path") or ""))
        if not snapshot_path.is_file():
            errors.append("final transcript DOM snapshot path does not exist")
        elif file_sha256(snapshot_path) != dom_seal.get("sha256"):
            errors.append("final transcript DOM snapshot bytes do not match its seal")
        else:
            try:
                snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                errors.append("final transcript DOM snapshot is unreadable")
            else:
                tape_media_for_dom = (
                    world_tape.get("media")
                    if isinstance(world_tape, dict)
                    and isinstance(world_tape.get("media"), dict)
                    else {}
                )
                errors.extend(
                    validate_final_transcript_dom_snapshot(
                        snapshot_payload,
                        {
                            "world_tape_run_id": (
                                world_tape.get("run_id")
                                if isinstance(world_tape, dict)
                                else None
                            ),
                            "source_tree_sha256": (
                                code.get("source_tree_sha256")
                                if isinstance(code, dict)
                                else None
                            ),
                            "runtime_config_sha256": attestation.get(
                                "runtime_config_sha256"
                            ),
                            "media": tape_media_for_dom,
                        },
                    )
                )
        for key in (
            "world_tape_run_id",
            "source_tree_sha256",
            "runtime_config_sha256",
        ):
            expected = {
                "world_tape_run_id": (
                    world_tape.get("run_id") if isinstance(world_tape, dict) else None
                ),
                "source_tree_sha256": (
                    code.get("source_tree_sha256") if isinstance(code, dict) else None
                ),
                "runtime_config_sha256": attestation.get(
                    "runtime_config_sha256"
                ),
            }[key]
            if dom_seal.get(key) != expected:
                errors.append(f"final transcript DOM seal {key} mismatch")
        seal_media = (
            dom_seal.get("media")
            if isinstance(dom_seal.get("media"), dict)
            else {}
        )
        expected_seal_media = (
            world_tape.get("media")
            if isinstance(world_tape, dict)
            and isinstance(world_tape.get("media"), dict)
            else {}
        )
        for key in (
            "video_id",
            "source_audio_path",
            "source_audio_sha256",
            "source_audio_size_bytes",
            "audio_sha256",
            "decoded_pcm_sha256",
            "decoded_samples",
            "sample_rate",
            "duration_seconds",
        ):
            if seal_media.get(key) != expected_seal_media.get(key):
                errors.append(f"final transcript DOM seal media.{key} mismatch")
    return errors
