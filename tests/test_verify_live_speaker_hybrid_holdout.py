import sys
from pathlib import Path
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_live_speaker_hybrid_holdout import _holdout_gates


def score(value, wrong):
    return {
        "strict_browser_live_score": value,
        "wrong_live_speech_ratio": wrong,
    }


class HybridHoldoutGateTests(unittest.TestCase):
    def test_passes_improved_candidate(self):
        result = _holdout_gates(
            score(0.40, 0.20), score(0.42, 0.18),
            score_tolerance=0.005, wrong_tolerance=0.005,
        )
        self.assertTrue(result["holdout_passed"])

    def test_rejects_score_regression_beyond_tolerance(self):
        result = _holdout_gates(
            score(0.40, 0.20), score(0.394, 0.18),
            score_tolerance=0.005, wrong_tolerance=0.005,
        )
        self.assertFalse(result["score_gate_passed"])
        self.assertFalse(result["holdout_passed"])

    def test_rejects_wrong_ratio_regression_beyond_tolerance(self):
        result = _holdout_gates(
            score(0.40, 0.20), score(0.42, 0.206),
            score_tolerance=0.005, wrong_tolerance=0.005,
        )
        self.assertFalse(result["wrong_ratio_gate_passed"])
        self.assertFalse(result["holdout_passed"])


if __name__ == "__main__":
    unittest.main()
