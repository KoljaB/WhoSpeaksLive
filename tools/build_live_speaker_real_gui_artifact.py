"""Build an immutable v2 artifact for real-GUI live-speaker evidence.

The prior World Tape supplies the effective launcher values that were already
observed.  Current CLI defaults fill fields added since that tape, and explicit
JSON patches describe the intended candidate delta.  The resulting artifact is
bound to the complete sanitized runtime config and current source tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_e2e_contract import (  # noqa: E402
    live_runtime_config,
    file_sha256,
    source_tree_sha256,
    stable_sha256,
)
from window.window_cli import parse_args  # noqa: E402


BUILDER_ID = "live_speaker_real_gui_artifact_builder_v1"
WORLD_TAPE_CONTRACT_ID = "whospeaks.live_world_tape.v1"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        current = target.get(str(key))
        if isinstance(current, dict) and isinstance(value, dict):
            _merge(current, value)
        else:
            target[str(key)] = copy.deepcopy(value)


def _load_patch_values(inline: list[str], paths: list[Path]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for value in inline:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Inline JSON patch must be an object.")
        patches.append(parsed)
    for path in paths:
        patches.append(_read_json_object(path.resolve(), label="JSON patch"))
    return patches


def build_artifact(
    *,
    base_artifact: dict[str, Any],
    base_artifact_path: Path,
    world_tape_manifest: dict[str, Any],
    world_tape_manifest_path: Path,
    runtime_patches: list[dict[str, Any]],
    algorithm_config_patches: list[dict[str, Any]],
    root: Path = ROOT,
) -> dict[str, Any]:
    if world_tape_manifest.get("contract_id") != WORLD_TAPE_CONTRACT_ID:
        raise ValueError("Input manifest is not a live-speaker World Tape.")
    if world_tape_manifest.get("status") != "complete":
        raise ValueError("Input World Tape must be complete.")
    run_id = str(world_tape_manifest.get("run_id") or "")
    if not run_id:
        raise ValueError("Input World Tape run_id is missing.")
    recorded_config = world_tape_manifest.get("runtime_config")
    if not isinstance(recorded_config, dict) or not recorded_config:
        raise ValueError("Input World Tape runtime_config is missing or empty.")
    recorded_config_hash = world_tape_manifest.get("runtime_config_sha256")
    if recorded_config_hash != stable_sha256(recorded_config):
        raise ValueError("Input World Tape runtime_config hash is invalid.")

    # Parsing with no arguments is intentional: it runs the same normalization
    # as the GUI and adds defaults introduced after the recorded tape.
    merged_runtime = dict(parse_args([]))
    # A World Tape can outlive CLI options that were later removed.  Only
    # values understood by the current parser can describe an executable
    # current GUI run; carrying retired keys into the artifact would make its
    # expected config impossible for the server to attest.
    retired_recorded_keys = sorted(set(recorded_config) - set(merged_runtime))
    current_recorded_config = {
        key: value for key, value in recorded_config.items() if key in merged_runtime
    }
    _merge(merged_runtime, current_recorded_config)
    for patch in runtime_patches:
        _merge(merged_runtime, patch)
    expected_runtime_config = live_runtime_config(merged_runtime)
    if not expected_runtime_config:
        raise ValueError("Resulting expected runtime config is empty.")

    artifact = copy.deepcopy(base_artifact)
    if algorithm_config_patches:
        algorithm_config = artifact.get("algorithm_config")
        if algorithm_config is None:
            algorithm_config = {}
            artifact["algorithm_config"] = algorithm_config
        if not isinstance(algorithm_config, dict):
            raise ValueError("Base artifact algorithm_config must be a JSON object.")
        for patch in algorithm_config_patches:
            _merge(algorithm_config, patch)

    artifact["real_gui_e2e_artifact_schema_version"] = 2
    artifact["expected_runtime_config"] = expected_runtime_config
    artifact["expected_runtime_config_sha256"] = stable_sha256(expected_runtime_config)
    artifact["expected_source_tree_sha256"] = source_tree_sha256(root.resolve())
    artifact["real_gui_e2e_artifact_build"] = {
        "builder_id": BUILDER_ID,
        "base_artifact_sha256": file_sha256(base_artifact_path.resolve()),
        "world_tape_manifest_sha256": file_sha256(world_tape_manifest_path.resolve()),
        "world_tape_run_id": run_id,
        "world_tape_runtime_config_sha256": recorded_config_hash,
        "retired_recorded_runtime_keys_ignored": retired_recorded_keys,
        "runtime_patches": copy.deepcopy(runtime_patches),
        "algorithm_config_patches": copy.deepcopy(algorithm_config_patches),
    }
    return artifact


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the completed bytes atomically and fails if
        # another process created the immutable destination first.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source/config-bound v2 real-GUI candidate artifact."
    )
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--world-tape-manifest", type=Path, required=True)
    parser.add_argument("--runtime-patch-json", action="append", default=[])
    parser.add_argument("--runtime-patch-file", type=Path, action="append", default=[])
    parser.add_argument("--algorithm-config-patch-json", action="append", default=[])
    parser.add_argument(
        "--algorithm-config-patch-file", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    base_path = args.base_artifact.resolve()
    manifest_path = args.world_tape_manifest.resolve()
    runtime_patches = _load_patch_values(
        args.runtime_patch_json,
        args.runtime_patch_file,
    )
    algorithm_patches = _load_patch_values(
        args.algorithm_config_patch_json,
        args.algorithm_config_patch_file,
    )
    artifact = build_artifact(
        base_artifact=_read_json_object(base_path, label="Base artifact"),
        base_artifact_path=base_path,
        world_tape_manifest=_read_json_object(manifest_path, label="World Tape manifest"),
        world_tape_manifest_path=manifest_path,
        runtime_patches=runtime_patches,
        algorithm_config_patches=algorithm_patches,
    )
    output_path = args.output.resolve()
    _write_immutable_json(output_path, artifact)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "artifact_sha256": file_sha256(output_path),
                "expected_runtime_config_sha256": artifact[
                    "expected_runtime_config_sha256"
                ],
                "expected_source_tree_sha256": artifact[
                    "expected_source_tree_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
