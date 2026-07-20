from __future__ import annotations

import unittest


from window.live_profile_tape import emit_live_profile_snapshot


class _Bus:
    def __init__(self) -> None:
        self.events = []

    def emit(self, name, payload) -> None:
        self.events.append((name, payload))


class _Memory:
    def __init__(self) -> None:
        self.speech_seconds = 1.5

    def export_profiles(self):
        return [{
            "label": "S1", "centroid": [1.0, 0.0], "sentence_count": 1,
            "speech_seconds": self.speech_seconds,
        }]


class _Owner:
    def __init__(self) -> None:
        self.bus = _Bus()

    def playback_time(self) -> float:
        return 4.25


class LiveProfileTapeTests(unittest.TestCase):
    def test_emits_only_changed_complete_snapshots_with_real_availability(self) -> None:
        owner = _Owner()
        memory = _Memory()
        first = emit_live_profile_snapshot(
            owner, memory, "S1", "provider=1.0", source="sync",
            sentence_start=1.0, sentence_end=2.0,
        )
        duplicate = emit_live_profile_snapshot(
            owner, memory, "S1", "provider=1.0", source="sync",
            sentence_start=1.0, sentence_end=2.0,
        )
        memory.speech_seconds = 2.0
        second = emit_live_profile_snapshot(
            owner, memory, "S1", "provider=1.0", source="async",
            sentence_start=2.0, sentence_end=3.0,
        )
        self.assertEqual(first["available_at"], 4.25)
        self.assertEqual(first["profile_generation"], 1)
        self.assertIsNone(duplicate)
        self.assertEqual(second["profile_generation"], 2)
        self.assertEqual(len(owner.bus.events), 2)


if __name__ == "__main__":
    unittest.main()
