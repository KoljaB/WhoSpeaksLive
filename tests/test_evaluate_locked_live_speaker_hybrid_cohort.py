import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_locked_live_speaker_hybrid_cohort as cohort
import sweep_live_speaker_hybrid as sweep
from window.live_speaker_hybrid import HYBRID_ALGORITHM_ID, HybridSpeakerTrackerConfig


def _score(value: float, wrong: float) -> dict[str, float]:
    return {
        "strict_browser_live_score": value,
        "wrong_live_speech_ratio": wrong,
    }


def _locked_config() -> HybridSpeakerTrackerConfig:
    return HybridSpeakerTrackerConfig(
        enable_young_profile_confirmation=True,
        enable_young_profile_lease=True,
        young_trusted_min_sentence_count=4,
        young_trusted_min_speech_seconds=8.0,
        young_min_similarity=0.35,
        young_min_margin=0.08,
        young_required_consecutive_probes=1,
        young_independent_scale_count=2,
        young_fast_independent_scale_count=1,
    )


def _write_locked_run(root: Path, *, windows=None, status="cached_holdout_failed") -> None:
    config = _locked_config()
    family = "young_profile_lease"
    winner = {
        "candidate_id": sweep._candidate_id(family, config),
        "family": family,
        "algorithm_id": HYBRID_ALGORITHM_ID,
        "config": config.__dict__,
        "promotion_gates_passed": True,
    }
    (root / "champion.json").write_text(
        json.dumps({"status": status, "winner": winner}), encoding="utf-8"
    )
    (root / "run.json").write_text(
        json.dumps({
            "algorithm_id": HYBRID_ALGORITHM_ID,
            "provider": sweep.DEFAULT_PROVIDER,
            "windows_seconds": windows or [0.8, 2.8],
            "long_weight": 0.25,
            "fresh_live_cost": {
                "fresh_window_requests_per_probe": 2,
                "max_fresh_window_requests_per_probe": 2,
            },
        }),
        encoding="utf-8",
    )


class CohortGateTests(unittest.TestCase):
    def test_requires_aggregate_and_each_video_to_pass(self):
        result = cohort._cohort_gates(
            {"a": _score(0.40, 0.20), "b": _score(0.50, 0.10)},
            {"a": _score(0.42, 0.18), "b": _score(0.494, 0.08)},
            0.45,
            0.46,
            score_tolerance=0.005,
            wrong_tolerance=0.005,
        )
        self.assertTrue(result["aggregate_score_gate_passed"])
        self.assertFalse(result["per_video_score_gate_passed"])
        self.assertFalse(result["cohort_gate_passed"])

    def test_wrong_ratio_regression_rejects_cohort(self):
        result = cohort._cohort_gates(
            {"a": _score(0.40, 0.20)},
            {"a": _score(0.42, 0.206)},
            0.40,
            0.42,
            score_tolerance=0.005,
            wrong_tolerance=0.005,
        )
        self.assertFalse(result["per_video_wrong_ratio_gate_passed"])
        self.assertFalse(result["cohort_gate_passed"])


class LockedCandidateTests(unittest.TestCase):
    def test_loads_exact_candidate_even_after_failed_old_holdout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_locked_run(root)
            locked = cohort._load_locked_candidate(root)
            self.assertEqual(locked.status, "cached_holdout_failed")
            self.assertEqual(locked.candidate_id, sweep._candidate_id(
                "young_profile_lease", _locked_config()
            ))
            self.assertEqual(
                [locked.short_window_seconds, locked.long_window_seconds],
                [0.8, 2.8],
            )

    def test_accepts_explicit_champion_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_locked_run(root)
            locked = cohort._load_locked_candidate(
                locked_champion=root / "champion.json"
            )
            self.assertEqual(locked.run_dir, root.resolve())

    def test_rejects_more_than_two_windows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_locked_run(root, windows=[0.8, 1.5, 2.8])
            with self.assertRaisesRegex(ValueError, "exactly two"):
                cohort._load_locked_candidate(root)

    def test_rejects_tampered_config_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_locked_run(root)
            path = root / "champion.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["winner"]["config"]["young_min_similarity"] = 0.99
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                cohort._load_locked_candidate(root)


class SafetyTests(unittest.TestCase):
    def test_rejects_previously_opened_videos(self):
        with self.assertRaisesRegex(ValueError, "already used"):
            cohort._validate_fresh_video_ids([sweep.SEALED_HOLDOUT])

    def test_rejects_duplicate_video_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cohort._validate_fresh_video_ids(["fresh", "fresh"])

    def test_output_cannot_touch_locked_run(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(ValueError, "outside"):
                cohort._assert_output_outside_locked_run(
                    root / "cohort.json", root
                )

    def test_atomic_json_replaces_without_leaving_temporary_file(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "nested" / "result.json"
            cohort._atomic_json(path, {"value": 1})
            cohort._atomic_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
