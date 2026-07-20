from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from extract_live_profile_tape import extract_profile_rows


class ExtractLiveProfileTapeTests(unittest.TestCase):
    def test_extracts_only_profile_events_and_preserves_availability(self) -> None:
        profile = {
            "event_id": "live_speaker_profile_snapshot_v1",
            "available_at": 4.25,
            "speaker_id": "S1",
            "centroid": [1.0, 0.0],
            "speech_seconds": 2.0,
            "sentence_count": 1,
            "profile_generation": 2,
            "profile_embedding_provider": "a=1+b=0.5",
            "source": "sync",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                json.dumps({"event": "status", "payload": {}}) + "\n"
                + json.dumps({"event": "live_speaker_profile_snapshot", "payload": profile}) + "\n",
                encoding="utf-8",
            )
            rows = extract_profile_rows(path, "a=1+b=0.5")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["available_at"], 4.25)
        self.assertEqual(rows[0]["profile_generation"], 2)


if __name__ == "__main__":
    unittest.main()
