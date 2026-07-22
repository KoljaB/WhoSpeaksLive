"""Validate and replay the server-core portion of a production World Tape.

This is intentionally only the first parity rung.  A tape is not optimization
eligible until the shared browser state reducer also reproduces its DOM samples.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from window.live_speaker_algorithm import (
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerStep,
)
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, CausalBayesSpeakerTracker
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_world_tape import (
    WORLD_TAPE_CONTRACT_ID,
    WORLD_TAPE_VECTOR_REF_KEY,
    load_world_tape_array,
)


PARITY_REPLAY_ID = "whospeaks.live_world_tape.server_core_parity.v1"
_ARTIFACT_FILES = (
    ("events_sha256", "events.jsonl"),
    ("arrays_sha256", "arrays.f32"),
    ("arrays_index_sha256", "arrays.jsonl"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _exact_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _collect_array_references(value: Any, destination: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if WORLD_TAPE_VECTOR_REF_KEY in value:
            reference = value.get(WORLD_TAPE_VECTOR_REF_KEY)
            destination.append(reference if isinstance(reference, dict) else {})
            return
        for child in value.values():
            _collect_array_references(child, destination)
    elif isinstance(value, list):
        for child in value:
            _collect_array_references(child, destination)


def _read_and_validate_array_index(
    root: Path,
    events: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    arrays_path = root / "arrays.f32"
    index_path = root / "arrays.jsonl"
    if not arrays_path.is_file() or not index_path.is_file():
        return []
    arrays_size = int(arrays_path.stat().st_size)
    entries: list[dict[str, Any]] = []
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    errors.append(f"arrays.jsonl line {line_number} is blank")
                    continue
                try:
                    entry = json.loads(line)
                except Exception as exc:
                    errors.append(
                        f"arrays.jsonl line {line_number} unreadable: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"arrays.jsonl line {line_number} is not an object")
                    continue
                entries.append(entry)
    except Exception as exc:
        errors.append(f"arrays.jsonl unreadable: {type(exc).__name__}: {exc}")
        return []

    indexed: dict[str, dict[str, Any]] = {}
    expected_offset = 0
    with arrays_path.open("rb") as arrays_handle:
        for position, entry in enumerate(entries, 1):
            array_id = str(entry.get("id") or "")
            if not array_id:
                errors.append(f"array index entry {position} has no id")
            elif array_id in indexed:
                errors.append(f"duplicate array id {array_id!r}")
            else:
                indexed[array_id] = entry
            if str(entry.get("dtype") or "") != "float32":
                errors.append(f"array {array_id or position!r} is not float32")
            shape_raw = entry.get("shape")
            shape: list[int] = []
            if (
                not isinstance(shape_raw, list)
                or not shape_raw
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in shape_raw
                )
            ):
                errors.append(f"array {array_id or position!r} has invalid shape")
            else:
                shape = list(shape_raw)
            offset = _exact_nonnegative_int(entry.get("offset_bytes"))
            length = _exact_nonnegative_int(entry.get("length_bytes"))
            if offset is None or offset % np.dtype(np.float32).itemsize:
                errors.append(f"array {array_id or position!r} has invalid offset")
                offset = expected_offset
            if length is None or length <= 0 or length % np.dtype(np.float32).itemsize:
                errors.append(f"array {array_id or position!r} has invalid length")
                length = 0
            if offset != expected_offset:
                errors.append(
                    f"array {array_id or position!r} is not contiguous "
                    f"(offset {offset}, expected {expected_offset})"
                )
            if shape and length != math.prod(shape) * np.dtype(np.float32).itemsize:
                errors.append(f"array {array_id or position!r} shape/length mismatch")
            end = offset + length
            if end > arrays_size:
                errors.append(f"array {array_id or position!r} exceeds arrays.f32 bounds")
            expected_hash = entry.get("sha256")
            if not _is_sha256(expected_hash):
                errors.append(f"array {array_id or position!r} has no valid SHA-256")
            elif end <= arrays_size and length > 0:
                arrays_handle.seek(offset)
                raw = arrays_handle.read(length)
                if hashlib.sha256(raw).hexdigest() != expected_hash:
                    errors.append(f"array {array_id or position!r} hash mismatch")
            if not isinstance(entry.get("semantic_path"), str) or not entry.get("semantic_path"):
                errors.append(f"array {array_id or position!r} has no semantic path")
            expected_offset = end
    if expected_offset != arrays_size:
        errors.append(
            f"arrays.f32 size/index mismatch ({arrays_size} bytes, {expected_offset} indexed)"
        )

    references: list[dict[str, Any]] = []
    for event in events:
        _collect_array_references(event.get("payload"), references)
    reference_counts: Counter[str] = Counter()
    for position, reference in enumerate(references, 1):
        array_id = str(reference.get("id") or "")
        if not array_id:
            errors.append(f"event array reference {position} has no id")
            continue
        reference_counts[array_id] += 1
        entry = indexed.get(array_id)
        if entry is None:
            errors.append(f"event references unindexed array {array_id!r}")
        elif reference != entry:
            errors.append(f"event metadata differs from index for array {array_id!r}")
    for array_id in indexed:
        count = reference_counts.get(array_id, 0)
        if count == 0:
            errors.append(f"indexed array {array_id!r} is not referenced by an event")
        elif count > 1:
            errors.append(f"indexed array {array_id!r} is referenced {count} times")
    return entries


def _resolve_arrays(tape_dir: Path, value: Any) -> Any:
    if isinstance(value, dict):
        if WORLD_TAPE_VECTOR_REF_KEY in value:
            return load_world_tape_array(tape_dir, value)
        return {key: _resolve_arrays(tape_dir, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_arrays(tape_dir, item) for item in value]
    return value


def read_world_tape_events(
    tape_dir: Path,
    *,
    resolve_arrays: bool = False,
) -> list[dict[str, Any]]:
    root = Path(tape_dir)
    records: list[dict[str, Any]] = []
    with (root / "events.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"events.jsonl line {line_number} is blank")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"events.jsonl line {line_number} is not an object")
            if resolve_arrays:
                record["payload"] = _resolve_arrays(root, record.get("payload") or {})
            record["line_number"] = line_number
            records.append(record)
    return records


def validate_world_tape(tape_dir: Path) -> dict[str, Any]:
    root = Path(tape_dir)
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "contract_id": PARITY_REPLAY_ID,
            "valid": False,
            "errors": [f"manifest unreadable: {type(exc).__name__}: {exc}"],
        }
    if manifest.get("contract_id") != WORLD_TAPE_CONTRACT_ID:
        errors.append("unexpected World Tape contract id")
    if manifest.get("status") != "complete":
        errors.append(
            f"World Tape status is {manifest.get('status')!r}, expected 'complete'"
        )
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        errors.append("manifest has no run_id")

    runtime_config = manifest.get("runtime_config")
    runtime_config_hash = manifest.get("runtime_config_sha256")
    if not isinstance(runtime_config, dict):
        errors.append("manifest runtime_config is missing or not an object")
    elif not _is_sha256(runtime_config_hash):
        errors.append("manifest runtime_config_sha256 is missing or invalid")
    elif _stable_json_sha256(runtime_config) != runtime_config_hash:
        errors.append("runtime_config SHA-256 mismatch")

    media_history = manifest.get("media_history")
    if not isinstance(media_history, list) or not media_history:
        errors.append("manifest media_history is empty")
    else:
        latest_media = media_history[-1]
        if not isinstance(latest_media, dict):
            errors.append("latest media_history entry is not an object")
        else:
            decoded_hash = latest_media.get("decoded_pcm_sha256")
            decoded_samples = _exact_nonnegative_int(latest_media.get("decoded_samples"))
            sample_rate = _exact_nonnegative_int(latest_media.get("sample_rate"))
            try:
                duration = float(latest_media.get("duration_seconds"))
            except (TypeError, ValueError):
                duration = float("nan")
            if not _is_sha256(decoded_hash):
                errors.append("latest media entry has no valid decoded_pcm_sha256")
            if decoded_samples is None or decoded_samples <= 0:
                errors.append("latest media entry has no positive decoded sample count")
            if sample_rate is None or sample_rate <= 0:
                errors.append("latest media entry has no positive sample rate")
            if not math.isfinite(duration) or duration <= 0.0:
                errors.append("latest media entry has no positive decoded duration")
            elif decoded_samples and sample_rate and abs(
                duration - (decoded_samples / sample_rate)
            ) > 1e-9:
                errors.append("latest media decoded duration is inconsistent")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        artifact = {}
        errors.append("manifest artifact is missing or not an object")
    if artifact.get("contract_id") != WORLD_TAPE_CONTRACT_ID:
        errors.append("artifact has unexpected World Tape contract id")
    if str(artifact.get("run_id") or "") != run_id:
        errors.append("artifact run_id does not match manifest")
    if artifact.get("status") != "complete":
        errors.append("artifact status is not complete")
    if artifact.get("writer_error") not in (None, ""):
        errors.append(f"artifact records writer error: {artifact.get('writer_error')}")

    for hash_name, filename in _ARTIFACT_FILES:
        expected = artifact.get(hash_name)
        path = root / filename
        if not path.is_file():
            errors.append(f"required artifact file {filename} is missing")
            continue
        if not _is_sha256(expected):
            errors.append(f"required artifact hash {hash_name} is missing or invalid")
            continue
        if _sha256_file(path) != expected:
            errors.append(f"{filename} hash mismatch")

    try:
        events = read_world_tape_events(root)
    except Exception as exc:
        return {
            "contract_id": PARITY_REPLAY_ID,
            "valid": False,
            "errors": [*errors, f"events unreadable: {type(exc).__name__}: {exc}"],
        }
    if not events:
        errors.append("World Tape contains no events")
    expected_sequence = list(range(1, len(events) + 1))
    actual_sequence: list[int] = []
    wall: list[int] = []
    for position, item in enumerate(events, 1):
        sequence = item.get("seq")
        wall_us = item.get("wall_us")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            errors.append(f"event {position} has an invalid seq")
            actual_sequence.append(0)
        else:
            actual_sequence.append(sequence)
        if isinstance(wall_us, bool) or not isinstance(wall_us, int):
            errors.append(f"event {position} has an invalid wall_us")
            wall.append(-1)
        else:
            wall.append(wall_us)
        if not isinstance(item.get("stream"), str) or not item.get("stream"):
            errors.append(f"event {position} has no stream")
        if not isinstance(item.get("event"), str) or not item.get("event"):
            errors.append(f"event {position} has no event name")
        if not isinstance(item.get("payload"), dict):
            errors.append(f"event {position} payload is not an object")
    if actual_sequence != expected_sequence:
        errors.append("event sequence is not contiguous and globally ordered")
    if any(value < 0 for value in wall):
        errors.append("wall_us contains a negative value")
    if any(right < left for left, right in zip(wall, wall[1:])):
        errors.append("wall_us moves backwards")
    counts: dict[str, int] = {}
    for item in events:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('stream')}:{item.get('event')}"
        counts[key] = counts.get(key, 0) + 1

    event_count = _exact_nonnegative_int(artifact.get("event_count"))
    enqueued_event_count = _exact_nonnegative_int(
        artifact.get("enqueued_event_count")
    )
    array_count = _exact_nonnegative_int(artifact.get("array_count"))
    recorded_counts = artifact.get("event_counts")
    if event_count is None:
        errors.append("artifact event_count is missing or invalid")
    elif event_count != len(events):
        errors.append(
            f"artifact event_count mismatch ({event_count}, actual {len(events)})"
        )
    if enqueued_event_count is None:
        errors.append("artifact enqueued_event_count is missing or invalid")
    elif enqueued_event_count != len(events):
        errors.append(
            "artifact enqueued_event_count does not equal persisted event count "
            f"({enqueued_event_count}, actual {len(events)})"
        )
    if array_count is None:
        errors.append("artifact array_count is missing or invalid")
    if not isinstance(recorded_counts, dict):
        errors.append("artifact event_counts is missing or invalid")
    elif recorded_counts != dict(sorted(counts.items())):
        errors.append("artifact event_counts does not match events.jsonl")

    array_entries = _read_and_validate_array_index(root, events, errors)
    if array_count is not None and array_count != len(array_entries):
        errors.append(
            f"artifact array_count mismatch ({array_count}, actual {len(array_entries)})"
        )

    core_inputs = counts.get("internal:live_speaker_core_input", 0)
    core_outputs = counts.get("internal:live_speaker_core_decision", 0)
    if core_inputs != core_outputs:
        errors.append(
            f"core input/output count mismatch ({core_inputs} inputs, {core_outputs} outputs)"
        )
    if core_inputs == 0:
        errors.append("World Tape contains no server-core parity steps")
    if counts.get("public:done", 0) == 0:
        errors.append("World Tape contains no public done event")
    browser_batches = counts.get("browser:ui_sample_clock", 0)
    browser_samples = 0
    browser_batch_sequences: list[int] = []
    browser_sample_sequences: list[int] = []
    for item in events:
        if (
            item.get("stream") == "browser"
            and item.get("event") == "ui_sample_clock"
        ):
            payload = item.get("payload") or {}
            samples = payload.get("samples") if isinstance(payload, dict) else None
            if not isinstance(samples, list):
                errors.append("browser sample batch has no samples list")
                continue
            batch_sequence = _exact_nonnegative_int(payload.get("batch_sequence"))
            if batch_sequence is None or batch_sequence <= 0:
                errors.append("browser sample batch has no positive batch_sequence")
            else:
                browser_batch_sequences.append(batch_sequence)
            browser_samples += len(samples)
            if _exact_nonnegative_int(payload.get("sample_count")) != len(samples):
                errors.append("browser sample_count does not match samples list")
            for sample in samples:
                if not isinstance(sample, dict):
                    errors.append("browser sample is not an object")
                    continue
                sample_sequence = _exact_nonnegative_int(sample.get("sample_sequence"))
                if sample_sequence is None or sample_sequence <= 0:
                    errors.append("browser sample has no positive sample_sequence")
                else:
                    browser_sample_sequences.append(sample_sequence)
    if browser_batches == 0 or browser_samples == 0:
        errors.append("World Tape contains no browser DOM samples")
    if browser_batch_sequences != list(range(1, len(browser_batch_sequences) + 1)):
        errors.append("browser batch_sequence is not contiguous and ordered")
    if browser_sample_sequences != list(range(1, len(browser_sample_sequences) + 1)):
        errors.append("browser sample_sequence is not contiguous and ordered")
    required_runtime_evidence = {
        "metadata:decoded_audio_bound": "decoded-audio binding",
        "internal:live_speaker_probe_observation": "probe lifecycle",
        "internal:live_speaker_gate_observation": "raw live gate",
        "internal:live_speaker_embedding_request_completed": "live embedding request",
        "internal:final_sentence_embedding_completed": "final sentence embedding",
        "internal:live_profile_embedding_completed": "live profile embedding",
    }
    for event_key, label in required_runtime_evidence.items():
        if counts.get(event_key, 0) == 0:
            errors.append(f"World Tape contains no {label} evidence ({event_key})")
    return {
        "contract_id": PARITY_REPLAY_ID,
        "valid": not errors,
        "errors": errors,
        "manifest_status": manifest.get("status"),
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "core_input_count": core_inputs,
        "core_output_count": core_outputs,
        "browser_sample_batch_count": browser_batches,
        "browser_sample_count": browser_samples,
        "array_count": len(array_entries),
        "gate_observation_count": counts.get(
            "internal:live_speaker_gate_observation", 0
        ),
        "final_sentence_embedding_count": counts.get(
            "internal:final_sentence_embedding_completed", 0
        ),
        "live_profile_embedding_count": counts.get(
            "internal:live_profile_embedding_completed", 0
        ),
    }


def _config_kwargs(config_type: type, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name: field for field in fields(config_type)}
    result = {key: value for key, value in raw.items() if key in allowed}
    for name in ("scale_windows", "scale_weights"):
        if name in result and isinstance(result[name], list):
            result[name] = tuple(float(value) for value in result[name])
    return result


def _numeric_map_error(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = set(left) | set(right)
    error = 0.0
    for key in keys:
        try:
            a = float(left.get(key, 0.0))
            b = float(right.get(key, 0.0))
        except (TypeError, ValueError):
            return float("inf")
        if not math.isfinite(a) or not math.isfinite(b):
            return float("inf")
        error = max(error, abs(a - b))
    return error


def replay_server_core(tape_dir: Path) -> dict[str, Any]:
    root = Path(tape_dir)
    events = read_world_tape_events(root, resolve_arrays=True)
    expected = {
        int(item["payload"]["step_id"]): item["payload"]
        for item in events
        if item.get("stream") == "internal"
        and item.get("event") == "live_speaker_core_decision"
    }
    inputs = [
        item["payload"]
        for item in events
        if item.get("stream") == "internal"
        and item.get("event") == "live_speaker_core_input"
    ]
    algorithm: CausalLiveSpeakerAlgorithm | CausalBayesSpeakerTracker | None = None
    signature = ""
    mismatches: list[dict[str, Any]] = []
    maximum_probability_error = 0.0
    maximum_similarity_error = 0.0
    for index, payload in enumerate(inputs):
        kind = str(payload.get("algorithm_type") or "classic")
        raw_config = dict(payload.get("algorithm_config") or {})
        new_signature = json.dumps(
            {"kind": kind, "config": raw_config},
            sort_keys=True,
            separators=(",", ":"),
        )
        if algorithm is None or signature != new_signature:
            if kind == "bayes":
                config = BayesSpeakerTrackerConfig(
                    **_config_kwargs(BayesSpeakerTrackerConfig, raw_config)
                )
                algorithm = CausalBayesSpeakerTracker(config=config)
            else:
                config = LiveSpeakerAlgorithmConfig(
                    **_config_kwargs(LiveSpeakerAlgorithmConfig, raw_config)
                )
                algorithm = CausalLiveSpeakerAlgorithm(config=config)
            signature = new_signature
        profiles = list(payload.get("profiles") or [])
        algorithm.sync_profiles(profiles)
        media_time = float(payload.get("media_time") or 0.0)
        embedding = payload.get("embedding")
        context_embedding = payload.get("context_embedding")
        if kind == "bayes":
            windows = tuple(float(value) for value in algorithm.config.scale_windows)
            evidences: list[MultiScaleEvidence] = []
            if embedding is not None:
                evidences.append(
                    MultiScaleEvidence(
                        windows[0],
                        np.asarray(embedding, dtype=np.float32),
                    )
                )
            if context_embedding is not None:
                evidences.append(
                    MultiScaleEvidence(
                        windows[-1],
                        np.asarray(context_embedding, dtype=np.float32),
                    )
                )
            decision = algorithm.step(
                MultiScaleStep(
                    media_time=media_time,
                    speech=bool(payload.get("speech")),
                    evidences=tuple(evidences),
                    probe_scheduled=bool(payload.get("probe_scheduled")),
                    release_signal=bool(payload.get("release_signal")),
                    skipped_reason=str(payload.get("skipped_reason") or ""),
                )
            )
        else:
            decision = algorithm.step(
                LiveSpeakerStep(
                    media_time=media_time,
                    speech=bool(payload.get("speech")),
                    embedding=(
                        None
                        if embedding is None
                        else np.asarray(embedding, dtype=np.float32)
                    ),
                    duration_seconds=float(payload.get("duration_seconds") or 0.0),
                    probe_scheduled=bool(payload.get("probe_scheduled")),
                    release_signal=bool(payload.get("release_signal")),
                    embedding_latency_seconds=payload.get("embedding_latency_seconds"),
                    skipped_reason=str(payload.get("skipped_reason") or ""),
                )
            )
        actual = decision.trace_record()
        step_id = int(payload.get("step_id") or 0)
        target = expected.get(step_id)
        if target is None:
            mismatches.append({"step_id": step_id, "reason": "missing recorded decision"})
            continue
        probability_error = _numeric_map_error(
            dict(actual.get("probabilities") or {}),
            dict(target.get("probabilities") or {}),
        )
        similarity_error = _numeric_map_error(
            dict(actual.get("similarities") or {}),
            dict(target.get("similarities") or {}),
        )
        maximum_probability_error = max(maximum_probability_error, probability_error)
        maximum_similarity_error = max(maximum_similarity_error, similarity_error)
        fields_match = all(
            actual.get(name) == target.get(name)
            for name in (
                "algorithm_id",
                "visible_speaker",
                "action",
                "reason",
                "candidate_speaker",
                "profile_count",
                "profile_generations",
            )
        ) and abs(float(actual.get("media_time") or 0.0) - float(target.get("media_time") or 0.0)) <= 1e-6
        if not fields_match or probability_error > 1e-6 or similarity_error > 1e-6:
            mismatches.append({
                "index": index,
                "step_id": step_id,
                "probability_max_abs_error": probability_error,
                "similarity_max_abs_error": similarity_error,
                "recorded": target,
                "replayed": actual,
            })
    compared = len(inputs)
    return {
        "contract_id": PARITY_REPLAY_ID,
        "parity_rung": "server_core",
        "exact_match": compared == len(expected) and not mismatches,
        "input_steps": compared,
        "recorded_steps": len(expected),
        "mismatch_count": len(mismatches) + abs(compared - len(expected)),
        "decision_match_ratio": (
            (compared - len(mismatches)) / compared
            if compared
            else float(not expected)
        ),
        "maximum_probability_abs_error": maximum_probability_error,
        "maximum_similarity_abs_error": maximum_similarity_error,
        "first_mismatches": mismatches[:20],
        "optimization_eligible": False,
        "eligibility_reason": (
            "Server-core parity is necessary but browser reducer and predictive-delta "
            "parity have not yet been proven."
        ),
    }


def validate_and_replay_world_tape(tape_dir: Path) -> dict[str, Any]:
    validation = validate_world_tape(tape_dir)
    replay = replay_server_core(tape_dir) if validation.get("valid") else None
    return {
        "contract_id": PARITY_REPLAY_ID,
        "tape_dir": str(Path(tape_dir).resolve()),
        "validation": validation,
        "server_core_replay": replay,
        "optimization_eligible": False,
    }
