from __future__ import annotations

import hashlib
import json
from pathlib import Path

from window.live_speaker_e2e_contract import (
    REAL_GUI_E2E_CONTRACT_ID,
    live_runtime_config,
    seal_real_gui_e2e_attestation,
    stable_sha256,
    validate_real_gui_e2e_observation,
    write_final_transcript_dom_snapshot,
)
from window.window_runtime_config import WindowConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_payload(tmp_path: Path) -> dict:
    runtime_config = {"live_speaker_probe_interval_seconds": 0.75}
    samples = [
        {
            "wall_time": 1000.0 + index * 0.1,
            "playback_time": index * 0.1,
            "browser_user_agent": "Mozilla/5.0 Chrome/140.0.0.0 Safari/537.36",
            "browser_webdriver": False,
            "browser_visibility_state": "visible",
            "fast_processing": False,
            "playback_rate": 1.0,
        }
        for index in range(101)
    ]
    canonical_path = tmp_path / "canonical.json"
    audio_path = tmp_path / "audio.wav"
    video_path = tmp_path / "video.mp4"
    canonical_path.write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 10.0, "speaker": "A"}]}),
        encoding="utf-8",
    )
    audio_path.write_bytes(b"exact source audio")
    video_path.write_bytes(b"exact source video")
    source_audio_hash = _sha256(audio_path)
    source_hash = "1" * 64
    attestation = {
        "contract_id": REAL_GUI_E2E_CONTRACT_ID,
        "capture_surface": "normal_gui_server_rendered_browser_dom",
        "runtime_config": runtime_config,
        "runtime_config_sha256": stable_sha256(runtime_config),
        "canonical": {
            "path": str(canonical_path),
            "sha256": _sha256(canonical_path),
            "max_end_seconds": 10.0,
        },
        "media": {
            "audio_path": str(audio_path),
            "audio_sha256": source_audio_hash,
            "source_audio_path": str(audio_path),
            "source_audio_sha256": source_audio_hash,
            "source_audio_size_bytes": audio_path.stat().st_size,
            "video_path": str(video_path),
            "video_sha256": _sha256(video_path),
            "video_id": "video",
        },
        "code": {"source_tree_sha256": source_hash},
        "candidate_artifact": {"sha256": "c" * 64},
    }
    attestation = seal_real_gui_e2e_attestation(attestation, attestation)
    tape_media = {
        "video_id": "video",
        "source_audio_path": str(audio_path),
        "source_audio_sha256": source_audio_hash,
        "source_audio_size_bytes": audio_path.stat().st_size,
        "audio_sha256": source_audio_hash,
        "decoded_pcm_sha256": "d" * 64,
        "decoded_samples": 160_000,
        "sample_rate": 16_000,
        "duration_seconds": 10.0,
    }
    tape_files = {}
    for name in ("events", "arrays", "arrays_index"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        tape_files[name] = path
    attestation["world_tape"] = {
        "contract_id": "whospeaks.live_world_tape.v1",
        "status": "complete",
        "run_id": "run-1",
        "writer_error": None,
        "event_count": 1,
        "enqueued_event_count": 1,
        "events_path": str(tape_files["events"]),
        "arrays_path": str(tape_files["arrays"]),
        "arrays_index_path": str(tape_files["arrays_index"]),
        "events_sha256": _sha256(tape_files["events"]),
        "arrays_sha256": _sha256(tape_files["arrays"]),
        "arrays_index_sha256": _sha256(tape_files["arrays_index"]),
        "runtime_config_sha256": stable_sha256(runtime_config),
        "media": tape_media,
    }
    snapshot = {
        "schema_version": "final_clustering_dom_snapshot_v1",
        "world_tape_run_id": "run-1",
        "capture_surface": "visible_chrome_final_transcript_dom_after_done",
        "captured_after_done": True,
        "source_tree_sha256": source_hash,
        "runtime_config_sha256": stable_sha256(runtime_config),
        "media": dict(tape_media),
        "browser": {
            "visibility_state": "visible",
            "has_focus": True,
            "webdriver": False,
        },
        "rows": [{"index": 0, "text": "Final row.", "assigned_speaker": "A"}],
    }
    attestation["final_transcript_dom_snapshot"] = write_final_transcript_dom_snapshot(
        browser_observation_path=tmp_path / "browser.json",
        payload=snapshot,
    )
    return {
        "attestation": attestation,
        "summary": {"sample_count": len(samples), "strict_browser_live_score": 0.5},
        "samples": samples,
    }


def test_valid_real_gui_e2e_evidence_passes(tmp_path: Path) -> None:
    assert validate_real_gui_e2e_observation(_valid_payload(tmp_path)) == []


def test_old_offline_or_unattested_result_cannot_be_promoted(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload.pop("attestation")
    assert validate_real_gui_e2e_observation(payload) == [
        "missing server-generated E2E attestation"
    ]


def test_fast_or_automated_browser_run_cannot_be_promoted(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["samples"][25]["fast_processing"] = True
    payload["samples"][50]["browser_webdriver"] = True
    errors = validate_real_gui_e2e_observation(payload)
    assert "Fast processing was enabled" in errors
    assert "automation/WebDriver browser detected" in errors


def test_claimed_sample_count_must_equal_serialized_samples(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["summary"]["sample_count"] += 1

    assert (
        "summary sample_count does not equal the serialized sample list"
        in validate_real_gui_e2e_observation(payload)
    )


def test_source_audio_is_rehashed_from_the_attested_path(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    Path(payload["attestation"]["media"]["source_audio_path"]).write_bytes(
        b"tampered source"
    )

    assert (
        "source audio bytes do not match the attested hash"
        in validate_real_gui_e2e_observation(payload)
    )


def test_tampered_runtime_config_cannot_be_promoted(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["attestation"]["runtime_config"]["live_speaker_probe_interval_seconds"] = 0.4
    assert "runtime configuration hash mismatch" in validate_real_gui_e2e_observation(payload)


def test_empty_runtime_config_cannot_be_promoted_even_with_matching_hash(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["attestation"]["runtime_config"] = {}
    payload["attestation"]["runtime_config_sha256"] = stable_sha256({})
    errors = validate_real_gui_e2e_observation(payload)
    assert "runtime configuration is missing or empty" in errors


def test_window_config_uses_public_mapping_values_for_runtime_attestation() -> None:
    config = WindowConfig.from_mapping(
        {
            "live_speaker_probe_interval_seconds": 0.4,
            "embedding_provider": "speechbrain_resnet",
            "browser_live_observation_output": "ignored.json",
            "unrelated": "ignored",
        }
    )

    assert live_runtime_config(config) == {
        "embedding_provider": "speechbrain_resnet",
        "live_speaker_probe_interval_seconds": 0.4,
        "unrelated": "ignored",
    }


def test_runtime_attestation_captures_full_config_but_redacts_evidence_and_secrets() -> None:
    config = WindowConfig.from_mapping(
        {
            "source_url": "https://example.test/media",
            "final_sentence_queue_size": 3,
            "sentence_tokenizer": "nltk+rule-based",
            "browser_live_observation_output": "evidence.json",
            "live_speaker_world_tape_output": "world-tape",
            "hf_token": "secret-value",
        }
    )

    assert live_runtime_config(config) == {
        "final_sentence_queue_size": 3,
        "sentence_tokenizer": "nltk+rule-based",
        "source_url": "https://example.test/media",
    }


def test_source_tree_drift_during_capture_cannot_be_promoted(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["attestation"]["capture_integrity"]["finish"]["source_tree_sha256"] = "changed"

    assert (
        "runtime, source tree, artifact, or media drifted during capture"
        in validate_real_gui_e2e_observation(payload)
    )


def test_incomplete_world_tape_cannot_be_promoted(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["attestation"]["world_tape"]["status"] = "recording"

    assert "live-speaker World Tape is incomplete" in validate_real_gui_e2e_observation(payload)
