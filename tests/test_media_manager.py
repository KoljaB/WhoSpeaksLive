from __future__ import annotations

import unittest
from pathlib import Path

from window.media_manager import MediaManager
from window.window_domain import MediaFiles


def media(name: str) -> MediaFiles:
    return MediaFiles(f"file://{name}", name, Path(f"{name}.wav"), Path(f"{name}.mp4"))


class MediaManagerTests(unittest.TestCase):
    def test_failed_controller_transition_does_not_publish_media(self) -> None:
        manager = MediaManager(media("first"), initial_version=10)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            manager.replace(media("second"), lambda _media: (_ for _ in ()).throw(RuntimeError("failed")))

        snapshot = manager.snapshot()
        self.assertEqual(snapshot.media.video_id, "first")
        self.assertEqual(snapshot.version, 10)

    def test_successful_transition_commits_controller_result_and_version_together(self) -> None:
        manager = MediaManager(media("first"), initial_version=10)

        snapshot = manager.transition(lambda: media("stream"))

        self.assertEqual(snapshot.media.video_id, "stream")
        self.assertEqual(snapshot.version, 11)
        self.assertEqual(manager.snapshot(), snapshot)


if __name__ == "__main__":
    unittest.main()
