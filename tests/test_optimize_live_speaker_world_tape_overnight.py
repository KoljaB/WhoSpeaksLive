from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "optimize_live_speaker_world_tape_overnight.py"
)
SPEC = importlib.util.spec_from_file_location(
    "optimize_live_speaker_world_tape_overnight", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path, *, cadence: float = 0.4) -> dict[str, Path]:
    workspace = tmp_path / "workspace"
    for relative in (
        "src/window/live_speaker_counterfactual.py",
        "src/window/live_speaker_browser_parity.py",
        "src/window/browser_live_speaker_scoring.py",
        "src/window/live_speaker_bayes.py",
        "src/embeddings/placeholder.py",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    runner = workspace / "tools/runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# immutable runner\n", encoding="utf-8")

    campaign = tmp_path / "campaign"
    _write_json(campaign / "campaign.json", {"campaign": "test"})
    canonical = campaign / "references/video-1/canonical_diarization.json"
    _write_json(canonical, {"segments": []})
    tape = campaign / "video-1_run-1"
    tape.mkdir(parents=True)
    (tape / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (tape / "arrays.f32").write_bytes(b"")
    (tape / "arrays.jsonl").write_text("", encoding="utf-8")
    _write_json(
        tape / "manifest.json",
        {
            "run_id": "run-1",
            "artifact": {
                "status": "complete",
                "events_sha256": "1" * 64,
                "arrays_sha256": "2" * 64,
                "arrays_index_sha256": "3" * 64,
            },
            "runtime_config": {
                "live_speaker_embedding_provider": "speechbrain_resnet",
                "live_speaker_probe_window_seconds": 0.7,
                "live_speaker_probe_context_window_seconds": 1.5,
                "live_speaker_probe_interval_seconds": cadence,
                "live_speaker_embedding_min_interval_seconds": 0.4,
            },
        },
    )
    parity = campaign / "baseline_parity_report.json"
    _write_json(
        parity,
        {
            "contract_id": "test.parity.v1",
            "optimization_eligible": False,
            "runs": [
                {
                    "video_id": "video-1",
                    "run_id": "run-1",
                    # Deliberately non-existent host paths exercise portable rebasing.
                    "tape_dir": f"Z:/other-host/{tape.name}",
                    "canonical_path": "Z:/other-host/canonical_diarization.json",
                }
            ],
        },
    )
    base = tmp_path / "base.json"
    _write_json(
        base,
        {
            "provider_spec": "speechbrain_resnet",
            "windows_seconds": [0.7, 1.5],
            "algorithm_config": {
                "scale_windows": [0.7, 1.5],
                "scale_weights": [0.8, 0.2],
                "min_similarity": 0.2,
            },
        },
    )
    return {
        "workspace": workspace,
        "campaign": campaign,
        "parity": parity,
        "base": base,
        "runner": runner,
    }


class OvernightWorldTapeRunnerTests(unittest.TestCase):
    def test_diagnostic_parity_fails_closed_without_explicit_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _inputs(Path(directory))
            with self.assertRaisesRegex(ValueError, "optimization_eligible=false"):
                MODULE.freeze_inputs(
                    paths["workspace"],
                    paths["campaign"],
                    paths["parity"],
                    allow_diagnostic_parity=False,
                    runner_path=paths["runner"],
                )

            frozen = MODULE.freeze_inputs(
                paths["workspace"],
                paths["campaign"],
                paths["parity"],
                allow_diagnostic_parity=True,
                runner_path=paths["runner"],
            )
            self.assertTrue(frozen.diagnostic_parity_exception)
            self.assertFalse(frozen.optimization_eligible)
            self.assertEqual(frozen.tapes[0].tape_dir.name, "video-1_run-1")

    def test_candidate_id_binds_all_immutable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _inputs(Path(directory))
            frozen = MODULE.freeze_inputs(
                paths["workspace"],
                paths["campaign"],
                paths["parity"],
                allow_diagnostic_parity=True,
                runner_path=paths["runner"],
            )
            config, _ = MODULE.load_base_config(paths["base"])
            candidate_id, config_sha, identity = MODULE.candidate_identity(
                config, frozen
            )

            self.assertTrue(candidate_id.startswith("wt-"))
            self.assertEqual(identity["algorithm_config_sha256"], config_sha)
            self.assertEqual(identity["source_tree_sha256"], frozen.source_tree_sha256)
            self.assertEqual(identity["campaign_sha256"], frozen.campaign_sha256)
            self.assertEqual(
                identity["parity_report_sha256"], frozen.parity_report_sha256
            )
            self.assertTrue(identity["tape_manifests"][0]["manifest_sha256"])
            self.assertTrue(
                identity["scorer_reducer_contract"]["selection_score_contract_id"]
            )

            changed = MODULE._with_patch(
                config, {"high_profile_unknown_bias": 0.6}
            )
            changed_id, _, _ = MODULE.candidate_identity(changed, frozen)
            self.assertNotEqual(changed_id, candidate_id)

    def test_runtime_geometry_is_enforced_for_every_tape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _inputs(Path(directory), cadence=0.75)
            with self.assertRaisesRegex(ValueError, "Unexpected probe cadence"):
                MODULE.freeze_inputs(
                    paths["workspace"],
                    paths["campaign"],
                    paths["parity"],
                    allow_diagnostic_parity=True,
                    runner_path=paths["runner"],
                )

    def test_fsync_journal_and_atomic_snapshot_are_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "run/journal.jsonl"
            MODULE.append_fsync_jsonl(
                journal, {"record_type": "candidate_result", "n": 1}
            )
            MODULE.append_fsync_jsonl(
                journal, {"record_type": "candidate_result", "n": 2}
            )
            self.assertEqual(
                [item["n"] for item in MODULE.read_journal(journal)], [1, 2]
            )

            snapshot = root / "run/progress.json"
            MODULE.atomic_write_json(snapshot, {"completed": 2})
            self.assertEqual(
                json.loads(snapshot.read_text(encoding="utf-8")), {"completed": 2}
            )
            self.assertFalse(list(snapshot.parent.glob("*.tmp")))

    def test_writer_lock_rejects_a_second_live_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".writer.lock"
            with MODULE.exclusive_writer_lock(lock):
                self.assertTrue(lock.is_file())
                with self.assertRaisesRegex(RuntimeError, "writer is active"):
                    with MODULE.exclusive_writer_lock(lock):
                        self.fail("second writer unexpectedly acquired the lock")
            self.assertFalse(lock.exists())

    def test_config_rejects_silent_unused_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluator ignores"):
            MODULE.normalize_and_validate_config(
                {
                    "scale_windows": [0.7, 1.5],
                    "scale_weights": [0.8, 0.2],
                    "made_up_parameter": 1,
                }
            )

    def test_search_space_excludes_rejected_unknown_exit_family(self) -> None:
        base = MODULE.normalize_and_validate_config(
            {
                "scale_windows": [0.7, 1.5],
                "scale_weights": [0.8, 0.2],
                "min_similarity": 0.2,
            }
        )
        fixed = MODULE.fixed_open_set_proposals(base)
        serialized = json.dumps(fixed, sort_keys=True)
        self.assertNotIn("unknown_exit", serialized)
        self.assertFalse(
            any(name.startswith("unknown_exit") for name in MODULE._RANDOM_SPACE)
        )
        hold_values = MODULE._RANDOM_SPACE["live_speaker_probe_hold_seconds"]
        self.assertIn(2.25, hold_values)
        self.assertIn(2.5, hold_values)
        self.assertIn(2.75, hold_values)


if __name__ == "__main__":
    unittest.main()
