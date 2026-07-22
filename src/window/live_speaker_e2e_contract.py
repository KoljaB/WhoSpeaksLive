"""Fail-closed evidence contract for real GUI live-speaker promotion runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from collections.abc import Mapping
from typing import Any, Iterable


REAL_GUI_E2E_CONTRACT_ID = "whospeaks.real_gui_live_speaker_e2e.v2"
PRODUCTION_CHAMPION_STATUS = "REAL_GUI_LIVE_E2E_VERIFIED_CHAMPION"
_WORLD_TAPE_CONTRACT_ID = "whospeaks.live_world_tape.v1"
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    suffixes = {".py", ".js", ".css", ".html"}
    source_roots = [root / "src" / "window", root / "src" / "embeddings"]
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
        "audio_sha256": media.get("audio_sha256") if isinstance(media, dict) else None,
        "video_sha256": media.get("video_sha256") if isinstance(media, dict) else None,
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
            "audio_sha256": file_sha256(audio_path),
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
    try:
        primary_score = float(summary.get("strict_browser_live_score"))
    except (TypeError, ValueError):
        primary_score = math.nan
    if not math.isfinite(primary_score) or not 0.0 <= primary_score <= 1.0:
        errors.append("strict browser live score is missing or invalid")

    canonical = attestation.get("canonical") if isinstance(attestation, dict) else {}
    media = attestation.get("media") if isinstance(attestation, dict) else {}
    code = attestation.get("code") if isinstance(attestation, dict) else {}
    if not isinstance(canonical, dict) or not canonical.get("sha256"):
        errors.append("canonical reference is not hash-bound")
    if not isinstance(media, dict) or not media.get("audio_sha256"):
        errors.append("source audio is not hash-bound")
    if not isinstance(code, dict) or not code.get("source_tree_sha256"):
        errors.append("executed source tree is not hash-bound")

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
        for key in ("events_sha256", "arrays_sha256", "arrays_index_sha256"):
            if not _is_sha256(world_tape.get(key)):
                errors.append(f"live-speaker World Tape has missing or invalid {key}")

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
    return errors
