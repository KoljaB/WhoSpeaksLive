from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_live_speaker_optimization_pilot import stable_candidate_id


class LiveSpeakerOptimizationPilotTests(unittest.TestCase):
    def test_candidate_id_is_order_independent_and_input_sensitive(self) -> None:
        first = stable_candidate_id(
            {"min_margin": 0.05, "ema_count": 3},
            {"weights": {"b": 0.5, "a": 1.0}, "window": 1.0},
            {"search": ["v1"], "validation": ["v2"]},
            "inputs-a",
        )
        reordered = stable_candidate_id(
            {"ema_count": 3, "min_margin": 0.05},
            {"window": 1.0, "weights": {"a": 1.0, "b": 0.5}},
            {"validation": ["v2"], "search": ["v1"]},
            "inputs-a",
        )
        changed = stable_candidate_id(
            {"ema_count": 3, "min_margin": 0.05},
            {"window": 1.0, "weights": {"a": 1.0, "b": 0.5}},
            {"validation": ["v2"], "search": ["v1"]},
            "inputs-b",
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
