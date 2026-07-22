from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from window.live_speaker_e2e_contract import live_runtime_config, stable_sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_live_speaker_real_gui_artifact",
    ROOT / "tools" / "build_live_speaker_real_gui_artifact.py",
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builder_preserves_base_and_applies_runtime_and_algorithm_patches(
    tmp_path: Path,
) -> None:
    recorded_config = {
        "language": "en",
        "live_speaker_probe_interval_seconds": 0.4,
        "browser_live_observation_output": "old-evidence.json",
        "retired_live_speaker_option": 123,
    }
    manifest = {
        "contract_id": "whospeaks.live_world_tape.v1",
        "run_id": "recorded-run",
        "status": "complete",
        "runtime_config": recorded_config,
        "runtime_config_sha256": stable_sha256(recorded_config),
    }
    base = {
        "candidate_name": "base",
        "algorithm_config": {"unknown_bias": 0.5, "nested": {"keep": 1}},
        "preserved": [1, 2, 3],
    }
    base_path = tmp_path / "base.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(base_path, base)
    _write_json(manifest_path, manifest)

    artifact = BUILDER.build_artifact(
        base_artifact=base,
        base_artifact_path=base_path,
        world_tape_manifest=manifest,
        world_tape_manifest_path=manifest_path,
        runtime_patches=[
            {
                "live_speaker_probe_interval_seconds": 0.5,
                "live_speaker_probe_clear_unknown_count": 3,
            }
        ],
        algorithm_config_patches=[
            {"unknown_bias": 0.6, "nested": {"added": 2}}
        ],
        root=ROOT,
    )

    assert artifact["candidate_name"] == "base"
    assert artifact["preserved"] == [1, 2, 3]
    assert artifact["algorithm_config"] == {
        "unknown_bias": 0.6,
        "nested": {"keep": 1, "added": 2},
    }
    runtime = artifact["expected_runtime_config"]
    assert runtime["live_speaker_probe_interval_seconds"] == 0.5
    assert runtime["live_speaker_probe_clear_unknown_count"] == 3
    assert "sentence_tokenizer" in runtime
    assert "browser_live_observation_output" not in runtime
    assert "retired_live_speaker_option" not in runtime
    assert artifact["real_gui_e2e_artifact_build"][
        "retired_recorded_runtime_keys_ignored"
    ] == ["retired_live_speaker_option"]
    assert runtime == live_runtime_config(runtime)
    assert artifact["expected_runtime_config_sha256"] == stable_sha256(runtime)
    assert len(artifact["expected_source_tree_sha256"]) == 64


def test_immutable_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    BUILDER._write_immutable_json(output, {"first": True})

    try:
        BUILDER._write_immutable_json(output, {"second": True})
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable artifact was overwritten")

    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}
