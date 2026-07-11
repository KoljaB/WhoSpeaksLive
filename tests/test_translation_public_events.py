from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.public_events import PublicEventNormalizer


class TranslationPublicEventTests(unittest.TestCase):
    def test_translation_states_receive_semantic_public_event_names(self) -> None:
        normalizer = PublicEventNormalizer(session_id="meeting")
        base = {
            "segment_id": "8",
            "target_language": "de",
            "source_text_hash": "abc",
        }
        expected = {
            "queued": "translation.queued",
            "translating": "translation.started",
            "complete": "translation.completed",
            "error": "translation.failed",
        }
        for status, event_type in expected.items():
            envelope = normalizer.normalize("translation", {**base, "status": status})[0]
            self.assertEqual(envelope["type"], event_type)
            self.assertEqual(envelope["session_id"], "meeting")
            self.assertEqual(envelope["payload"]["target_language"], "de")


if __name__ == "__main__":
    unittest.main()
