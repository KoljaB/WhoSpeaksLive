from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import numpy as np

from window.live_speaker_algorithm import (
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerStep,
)
from window.live_speaker_parity_replay import (
    read_world_tape_events,
    validate_and_replay_world_tape,
    validate_world_tape,
)
import window.live_speaker_world_tape as world_tape_module
from window.live_speaker_world_tape import LiveSpeakerWorldTapeRecorder
from window.window_domain import MediaFiles
from window.window_events import EventBus
from window.window_diarizer_runtime_audio import WindowRuntimeAudioMixin
from window.window_runtime_config import WindowConfig


def test_internal_events_are_not_published_to_browser_subscribers() -> None:
    bus = EventBus()
    subscriber = bus.subscribe()
    captured: list[tuple[str, dict]] = []
    bus.add_internal_listener(lambda event, payload: captured.append((event, payload)))

    bus.emit_internal("private", {"value": 1})

    assert captured == [("private", {"value": 1})]
    assert subscriber.empty()


def test_live_gate_records_raw_rms_features_without_changing_decision() -> None:
    harness = WindowRuntimeAudioMixin()
    harness.args = SimpleNamespace(
        live_speaker_probe_speech_backend="rms",
        vad_frame_seconds=0.1,
        vad_speech_rms_threshold=0.05,
        live_speaker_probe_min_speech_seconds=0.1,
        vad_min_speech_seconds=0.1,
    )
    harness.bus = EventBus()
    captured: list[tuple[str, dict]] = []
    harness.bus.add_internal_listener(
        lambda event, payload: captured.append((event, payload))
    )
    audio = np.concatenate((
        np.zeros(1600, dtype=np.float32),
        np.full(1600, 0.1, dtype=np.float32),
    ))

    assert harness._audio_has_live_probe_speech(0.0, 0.2, audio, 16000) is True
    assert captured[-1][0] == "live_speaker_gate_observation"
    assert captured[-1][1]["has_speech"] is True
    assert np.allclose(captured[-1][1]["rms_values"], [0.0, 0.1])


def _create_valid_world_tape(tmp_path: Path) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "audio.raw"
    audio.write_bytes(b"same production audio")
    media = MediaFiles("test://video", "video", audio, audio)
    args = WindowConfig.from_mapping({
        "embedding_provider": "speechbrain_resnet",
        "live_speaker_embedding_provider": "speechbrain_resnet",
        "harmless_value": 3,
        "hf_token": "must-not-be-recorded",
    })
    recorder = LiveSpeakerWorldTapeRecorder(tmp_path / "tapes", args=args, media=media)
    decoded = np.linspace(-0.25, 0.25, 3200, dtype=np.float32)
    decoded_binding = recorder.record_decoded_audio(decoded, 16000)
    assert decoded_binding["decoded_pcm_sha256"] == hashlib.sha256(
        decoded.tobytes()
    ).hexdigest()
    assert decoded_binding["decoded_samples"] == 3200
    assert decoded_binding["duration_seconds"] == 0.2
    bus = EventBus()
    bus.add_listener(recorder.record_public)
    bus.add_internal_listener(recorder.record_internal)

    config = LiveSpeakerAlgorithmConfig(
        min_similarity=0.1,
        min_margin=0.0,
        min_known_probability=0.0,
    )
    profiles = [{
        "label": "S1",
        "centroid": [1.0] + [0.0] * 15,
        "sentence_count": 1,
        "speech_seconds": 1.0,
    }]
    embedding = np.asarray([1.0] + [0.0] * 15, dtype=np.float32)
    algorithm = CausalLiveSpeakerAlgorithm(config=config)
    algorithm.sync_profiles(profiles)
    decision = algorithm.step(
        LiveSpeakerStep(
            media_time=1.0,
            speech=True,
            embedding=embedding,
            duration_seconds=1.0,
            probe_scheduled=True,
        )
    )
    bus.emit("status", {"message": "recording"})
    bus.emit_internal(
        "live_speaker_core_input",
        {
            "step_id": 1,
            "media_time": 1.0,
            "speech": True,
            "duration_seconds": 1.0,
            "probe_scheduled": True,
            "release_signal": False,
            "embedding_latency_seconds": 0.01,
            "skipped_reason": "",
            "algorithm_type": "classic",
            "algorithm_config": asdict(config),
            "profiles": profiles,
            "embedding": embedding.tolist(),
            "context_embedding": None,
            "context_duration_seconds": None,
        },
    )
    bus.emit_internal(
        "live_speaker_core_decision",
        {"step_id": 1, **decision.trace_record()},
    )
    for event in (
        "live_speaker_probe_observation",
        "live_speaker_gate_observation",
        "live_speaker_embedding_request_completed",
        "final_sentence_embedding_completed",
        "live_profile_embedding_completed",
    ):
        bus.emit_internal(event, {"media_time": 1.0, "test_evidence": True})
    recorder.record_browser_samples([{
        "sample_sequence": 1,
        "playback_time": 1.0,
        "current_live_speaker_id": "S1",
        "dom_live_speaker_ids": ["S1"],
    }], batch_sequence=1)
    bus.emit("done", {"media_time": 1.0})
    summary = recorder.close("test")
    return Path(summary["output_dir"]), summary


def test_world_tape_preserves_arrays_and_replays_shared_core(tmp_path: Path) -> None:
    tape_dir, summary = _create_valid_world_tape(tmp_path)

    manifest = (tape_dir / "manifest.json").read_text(encoding="utf-8")
    assert "must-not-be-recorded" not in manifest
    assert '"harmless_value": 3' in manifest
    assert summary["array_count"] >= 2
    assert summary["event_count"] == summary["enqueued_event_count"]
    report = validate_and_replay_world_tape(tape_dir)
    assert report["validation"]["valid"] is True
    assert report["server_core_replay"]["exact_match"] is True
    assert report["optimization_eligible"] is False


def test_world_tape_freezes_payload_before_async_write(tmp_path: Path) -> None:
    audio = tmp_path / "audio.raw"
    audio.write_bytes(b"audio")
    media = MediaFiles("test://freeze", "freeze", audio, audio)
    recorder = LiveSpeakerWorldTapeRecorder(
        tmp_path / "tapes",
        args=WindowConfig.from_mapping({"harmless": True}),
        media=media,
    )
    original_jsonable = world_tape_module._jsonable
    entered = threading.Event()
    release = threading.Event()

    def gated_jsonable(value):
        if isinstance(value, dict) and value.get("marker") == "freeze-test":
            entered.set()
            assert release.wait(5.0)
        return original_jsonable(value)

    world_tape_module._jsonable = gated_jsonable
    payload = {"marker": "freeze-test", "nested": {"values": [1, 2, 3]}}
    try:
        recorder.record_public("mutable", payload)
        assert entered.wait(5.0)
        payload["nested"]["values"][0] = 99
        payload["nested"]["values"].append(4)
        release.set()
        summary = recorder.close("freeze-test")
    finally:
        release.set()
        world_tape_module._jsonable = original_jsonable
    events = read_world_tape_events(Path(summary["output_dir"]))
    assert events[0]["payload"]["nested"]["values"] == [1, 2, 3]


def test_world_tape_writer_error_cannot_finalize_complete(tmp_path: Path) -> None:
    audio = tmp_path / "audio.raw"
    audio.write_bytes(b"audio")
    media = MediaFiles("test://writer-error", "writer-error", audio, audio)
    recorder = LiveSpeakerWorldTapeRecorder(
        tmp_path / "tapes",
        args=WindowConfig.from_mapping({"harmless": True}),
        media=media,
    )

    def fail_externalization(*args, **kwargs):
        raise RuntimeError("intentional writer failure")

    recorder._externalize_arrays = fail_externalization
    recorder.record_public("broken", {"value": 1})
    summary = recorder.close("writer-error-test")
    manifest = json.loads(
        (Path(summary["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "invalid"
    assert manifest["status"] == "invalid"
    assert "intentional writer failure" in str(summary["writer_error"])
    assert validate_world_tape(Path(summary["output_dir"]))["valid"] is False


def test_world_tape_rejects_noncomplete_empty_and_tampered_artifacts(
    tmp_path: Path,
) -> None:
    tape_dir, _ = _create_valid_world_tape(tmp_path / "valid")

    for status in ("recording", "aborted", "invalid"):
        clone = tmp_path / f"status-{status}"
        shutil.copytree(tape_dir, clone)
        manifest_path = clone / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert validate_world_tape(clone)["valid"] is False

    empty_audio = tmp_path / "empty.audio"
    empty_audio.write_bytes(b"audio")
    empty_recorder = LiveSpeakerWorldTapeRecorder(
        tmp_path / "empty-tapes",
        args=WindowConfig.from_mapping({}),
        media=MediaFiles("test://empty", "empty", empty_audio, empty_audio),
    )
    empty_summary = empty_recorder.close("empty")
    empty_report = validate_world_tape(Path(empty_summary["output_dir"]))
    assert empty_report["valid"] is False
    assert any("no events" in error for error in empty_report["errors"])

    cases = (
        "missing-hash",
        "bad-config-hash",
        "bad-count",
        "bad-array",
        "bad-array-bounds",
    )
    for case in cases:
        clone = tmp_path / case
        shutil.copytree(tape_dir, clone)
        manifest_path = clone / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if case == "missing-hash":
            del manifest["artifact"]["events_sha256"]
        elif case == "bad-config-hash":
            manifest["runtime_config_sha256"] = "0" * 64
        elif case == "bad-count":
            manifest["artifact"]["event_count"] += 1
        elif case == "bad-array":
            arrays_path = clone / "arrays.f32"
            raw = bytearray(arrays_path.read_bytes())
            raw[0] ^= 0x01
            arrays_path.write_bytes(raw)
            # Keep the whole-file binding current so the per-array hash check is
            # independently exercised.
            manifest["artifact"]["arrays_sha256"] = hashlib.sha256(raw).hexdigest()
        else:
            arrays_path = clone / "arrays.f32"
            index_path = clone / "arrays.jsonl"
            events_path = clone / "events.jsonl"
            entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
            entries[0]["offset_bytes"] = arrays_path.stat().st_size + 4
            replacement = entries[0]

            def replace_reference(value):
                if isinstance(value, dict):
                    reference = value.get("$world_tape_array")
                    if isinstance(reference, dict) and reference.get("id") == replacement["id"]:
                        value["$world_tape_array"] = dict(replacement)
                    else:
                        for child in value.values():
                            replace_reference(child)
                elif isinstance(value, list):
                    for child in value:
                        replace_reference(child)

            event_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            for event_row in event_rows:
                replace_reference(event_row.get("payload"))
            index_bytes = (
                "\n".join(json.dumps(entry, separators=(",", ":")) for entry in entries) + "\n"
            ).encode("utf-8")
            event_bytes = (
                "\n".join(json.dumps(event, separators=(",", ":")) for event in event_rows) + "\n"
            ).encode("utf-8")
            index_path.write_bytes(index_bytes)
            events_path.write_bytes(event_bytes)
            manifest["artifact"]["arrays_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
            manifest["artifact"]["events_sha256"] = hashlib.sha256(event_bytes).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = validate_world_tape(clone)
        assert report["valid"] is False
        if case == "bad-array":
            assert any("array" in error and "hash mismatch" in error for error in report["errors"])
        if case == "bad-array-bounds":
            assert any("bounds" in error for error in report["errors"])
