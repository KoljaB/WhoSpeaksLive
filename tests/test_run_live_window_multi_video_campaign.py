from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_live_window_multi_video_campaign",
    ROOT / "tools" / "run_live_window_multi_video_campaign.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultiVideoCampaignTests(unittest.TestCase):
    def test_parse_videos_preserves_order_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            self.assertEqual(
                MODULE._parse_videos((f"a={first}", f"b={second}")),
                (("a", first.resolve()), ("b", second.resolve())),
            )
            with self.assertRaisesRegex(ValueError, "Duplicate video id"):
                MODULE._parse_videos((f"a={first}", f"a={second}"))

    def test_shared_provider_lease_never_shuts_down_owner(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def embed(self, audio, sample_rate):
                return audio, sample_rate

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        provider = Provider()
        lease = MODULE._SharedProviderLease(provider)
        self.assertEqual(lease.embed("audio", 16000), ("audio", 16000))
        lease.shutdown()
        self.assertEqual(provider.shutdown_calls, 0)
        lease.release()
        with self.assertRaisesRegex(RuntimeError, "released"):
            lease.embed("audio", 16000)


if __name__ == "__main__":
    unittest.main()
