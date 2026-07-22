from __future__ import annotations

from window.live_speaker_e2e_contract import (
    REAL_GUI_E2E_CONTRACT_ID,
    live_runtime_config,
    seal_real_gui_e2e_attestation,
    stable_sha256,
    validate_real_gui_e2e_observation,
)
from window.window_runtime_config import WindowConfig


def _valid_payload() -> dict:
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
    attestation = {
        "contract_id": REAL_GUI_E2E_CONTRACT_ID,
        "capture_surface": "normal_gui_server_rendered_browser_dom",
        "runtime_config": runtime_config,
        "runtime_config_sha256": stable_sha256(runtime_config),
        "canonical": {"sha256": "canonical", "max_end_seconds": 10.0},
        "media": {
            "audio_sha256": "audio",
            "video_sha256": "video",
            "video_id": "video",
        },
        "code": {"source_tree_sha256": "source"},
        "candidate_artifact": {"sha256": "candidate"},
    }
    attestation = seal_real_gui_e2e_attestation(attestation, attestation)
    attestation["world_tape"] = {
        "contract_id": "whospeaks.live_world_tape.v1",
        "status": "complete",
        "run_id": "run-1",
        "writer_error": None,
        "event_count": 1,
        "events_sha256": "e" * 64,
        "arrays_sha256": "a" * 64,
        "arrays_index_sha256": "f" * 64,
    }
    return {
        "attestation": attestation,
        "summary": {"sample_count": len(samples), "strict_browser_live_score": 0.5},
        "samples": samples,
    }


def test_valid_real_gui_e2e_evidence_passes() -> None:
    assert validate_real_gui_e2e_observation(_valid_payload()) == []


def test_old_offline_or_unattested_result_cannot_be_promoted() -> None:
    payload = _valid_payload()
    payload.pop("attestation")
    assert validate_real_gui_e2e_observation(payload) == [
        "missing server-generated E2E attestation"
    ]


def test_fast_or_automated_browser_run_cannot_be_promoted() -> None:
    payload = _valid_payload()
    payload["samples"][25]["fast_processing"] = True
    payload["samples"][50]["browser_webdriver"] = True
    errors = validate_real_gui_e2e_observation(payload)
    assert "Fast processing was enabled" in errors
    assert "automation/WebDriver browser detected" in errors


def test_tampered_runtime_config_cannot_be_promoted() -> None:
    payload = _valid_payload()
    payload["attestation"]["runtime_config"]["live_speaker_probe_interval_seconds"] = 0.4
    assert "runtime configuration hash mismatch" in validate_real_gui_e2e_observation(payload)


def test_empty_runtime_config_cannot_be_promoted_even_with_matching_hash() -> None:
    payload = _valid_payload()
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


def test_source_tree_drift_during_capture_cannot_be_promoted() -> None:
    payload = _valid_payload()
    payload["attestation"]["capture_integrity"]["finish"]["source_tree_sha256"] = "changed"

    assert (
        "runtime, source tree, artifact, or media drifted during capture"
        in validate_real_gui_e2e_observation(payload)
    )


def test_incomplete_world_tape_cannot_be_promoted() -> None:
    payload = _valid_payload()
    payload["attestation"]["world_tape"]["status"] = "recording"

    assert "live-speaker World Tape is incomplete" in validate_real_gui_e2e_observation(payload)
