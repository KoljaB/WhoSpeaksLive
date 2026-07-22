from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from window.live_speaker_e2e_contract import (
    REAL_GUI_E2E_CONTRACT_ID,
    seal_real_gui_e2e_attestation,
    stable_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "promote_live_speaker_real_gui_e2e",
    ROOT / "tools" / "promote_live_speaker_real_gui_e2e.py",
)
assert SPEC and SPEC.loader
PROMOTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMOTER)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _artifact(runtime_config: dict, source_hash: str) -> dict:
    return {
        "candidate_name": "test",
        "expected_runtime_config": runtime_config,
        "expected_runtime_config_sha256": stable_sha256(runtime_config),
        "expected_source_tree_sha256": source_hash,
    }


def _observation(
    *,
    runtime_config: dict,
    source_hash: str,
    artifact_hash: str,
    score: float,
    run_id: str,
) -> dict:
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
            "video_id": "JWS-qfR6K3w",
        },
        "code": {"source_tree_sha256": source_hash},
        "candidate_artifact": {"sha256": artifact_hash},
    }
    attestation = seal_real_gui_e2e_attestation(attestation, attestation)
    attestation["world_tape"] = {
        "contract_id": "whospeaks.live_world_tape.v1",
        "status": "complete",
        "run_id": run_id,
        "writer_error": None,
        "event_count": 1,
        "events_sha256": "e" * 64,
        "arrays_sha256": "a" * 64,
        "arrays_index_sha256": "f" * 64,
    }
    return {
        "attestation": attestation,
        "summary": {
            "sample_count": len(samples),
            "strict_browser_live_score": score,
        },
        "samples": samples,
    }


def _run_promoter(
    monkeypatch,
    tmp_path: Path,
    *,
    candidate_observed_config: dict | None = None,
    candidate_observed_source_hash: str | None = None,
) -> tuple[int, dict]:
    baseline_config = {"live_speaker_probe_interval_seconds": 0.4}
    candidate_config = {"live_speaker_probe_interval_seconds": 0.5}
    baseline_source = "a" * 64
    candidate_source = "b" * 64
    baseline_artifact = tmp_path / "baseline.json"
    candidate_artifact = tmp_path / "candidate.json"
    _write_json(baseline_artifact, _artifact(baseline_config, baseline_source))
    _write_json(candidate_artifact, _artifact(candidate_config, candidate_source))
    baseline_hash = PROMOTER.file_sha256(baseline_artifact)
    candidate_hash = PROMOTER.file_sha256(candidate_artifact)
    assert baseline_hash and candidate_hash

    baseline_observations = [
        tmp_path / f"baseline_observation_{index}.json" for index in range(1, 4)
    ]
    candidate_observations = [
        tmp_path / f"candidate_observation_{index}.json" for index in range(1, 4)
    ]
    for index, path in enumerate(baseline_observations, 1):
        _write_json(
            path,
            _observation(
                runtime_config=baseline_config,
                source_hash=baseline_source,
                artifact_hash=baseline_hash,
                score=0.4,
                run_id=f"baseline-{index}",
            ),
        )
    for index, path in enumerate(candidate_observations, 1):
        _write_json(
            path,
            _observation(
                runtime_config=candidate_observed_config or candidate_config,
                source_hash=candidate_observed_source_hash or candidate_source,
                artifact_hash=candidate_hash,
                score=0.42,
                run_id=f"candidate-{index}",
            ),
        )

    output = tmp_path / "promotion.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_live_speaker_real_gui_e2e.py",
            "--baseline-artifact",
            str(baseline_artifact),
            "--candidate-artifact",
            str(candidate_artifact),
            "--output",
            str(output),
        ]
        + [
            item
            for path in baseline_observations
            for item in ("--baseline-observation", str(path))
        ]
        + [
            item
            for path in candidate_observations
            for item in ("--candidate-observation", str(path))
        ],
    )
    result = PROMOTER.main()
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_promoter_accepts_exact_artifact_runtime_and_source_bindings(
    monkeypatch, tmp_path: Path
) -> None:
    result, payload = _run_promoter(monkeypatch, tmp_path)

    assert result == 0
    assert payload["production_promotion_eligible"] is True


def test_promoter_rejects_runtime_config_that_differs_from_candidate_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    result, payload = _run_promoter(
        monkeypatch,
        tmp_path,
        candidate_observed_config={"live_speaker_probe_interval_seconds": 0.75},
    )

    assert result == 2
    assert any(
        "effective runtime config does not match artifact" in error
        for error in payload["real_gui_live_e2e_promotion"]["errors"]
    )


def test_promoter_rejects_source_tree_that_differs_from_candidate_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    result, payload = _run_promoter(
        monkeypatch,
        tmp_path,
        candidate_observed_source_hash="c" * 64,
    )

    assert result == 2
    assert any(
        "source tree hash does not match artifact" in error
        for error in payload["real_gui_live_e2e_promotion"]["errors"]
    )


def test_promoter_rejects_legacy_artifact_without_expected_bindings(
    monkeypatch, tmp_path: Path
) -> None:
    expectations, errors = PROMOTER._artifact_expectations(
        "candidate", {"candidate_name": "legacy"}
    )
    assert expectations is None
    assert "candidate artifact: expected_runtime_config is missing or empty" in errors
    assert "candidate artifact: expected_source_tree_sha256 is missing or invalid" in errors


def test_promoter_rejects_duplicate_observation_paths(monkeypatch, tmp_path: Path) -> None:
    result, payload = _run_promoter(monkeypatch, tmp_path)
    assert result == 0
    argv = list(sys.argv)
    duplicate = argv[argv.index("--baseline-observation") + 1]
    second = argv.index("--baseline-observation", argv.index("--baseline-observation") + 1)
    argv[second + 1] = duplicate
    monkeypatch.setattr(sys, "argv", argv)

    result = PROMOTER.main()
    payload = json.loads((tmp_path / "promotion.json").read_text(encoding="utf-8"))

    assert result == 2
    assert (
        "observation paths must be unique across baseline and candidate runs"
        in payload["real_gui_live_e2e_promotion"]["errors"]
    )


def test_promoter_rejects_duplicate_world_tape_run_ids(monkeypatch, tmp_path: Path) -> None:
    result, payload = _run_promoter(monkeypatch, tmp_path)
    assert result == 0
    candidate_flags = [
        index for index, value in enumerate(sys.argv) if value == "--candidate-observation"
    ]
    first_path = Path(sys.argv[candidate_flags[0] + 1])
    second_path = Path(sys.argv[candidate_flags[1] + 1])
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second["attestation"]["world_tape"]["run_id"] = first["attestation"]["world_tape"][
        "run_id"
    ]
    _write_json(second_path, second)

    result = PROMOTER.main()
    payload = json.loads((tmp_path / "promotion.json").read_text(encoding="utf-8"))

    assert result == 2
    assert (
        "live-speaker World Tape run ids must be unique across all runs"
        in payload["real_gui_live_e2e_promotion"]["errors"]
    )


def test_promoter_minimum_flags_cannot_lower_three_by_three_floor(
    monkeypatch, tmp_path: Path
) -> None:
    result, payload = _run_promoter(monkeypatch, tmp_path)
    assert result == 0
    argv = list(sys.argv)
    for flag in ("--baseline-observation", "--candidate-observation"):
        last = len(argv) - 1 - argv[::-1].index(flag)
        del argv[last : last + 2]
    argv.extend(
        [
            "--minimum-baseline-runs",
            "1",
            "--minimum-candidate-runs",
            "1",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)

    result = PROMOTER.main()
    payload = json.loads((tmp_path / "promotion.json").read_text(encoding="utf-8"))

    assert result == 2
    errors = payload["real_gui_live_e2e_promotion"]["errors"]
    assert "baseline: requires at least 3 complete real GUI runs" in errors
    assert "candidate: requires at least 3 complete real GUI runs" in errors
