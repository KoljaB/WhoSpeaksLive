from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from window.live_window_server import LiveWindowApplication, WindowServer
from window.window_cli import parse_args
from window.window_domain import MediaFiles
from window.window_events import EventBus


class _Controller:
    def __init__(self) -> None:
        self.stops = 0
        self.media: MediaFiles | None = None
        self.audio = [0.0, 0.25, -0.25, 0.0]
        self.sample_rate = 16000

    def is_running(self) -> bool:
        return False

    def stop(self) -> None:
        self.stops += 1

    def session_snapshot(self) -> dict[str, object]:
        return {"id": "", "rows": [], "speakers": []}

    def current_session_id(self) -> str:
        return ""

    def write_session_audio(self, _path: Path) -> None:
        return None

    def set_media(self, media: MediaFiles) -> None:
        self.media = media

    def set_browser_stream(self, url: str) -> MediaFiles:
        media = MediaFiles(url, "browser", Path("browser.wav"), Path("browser.wav"))
        self.media = media
        return media


class LiveWindowApplicationTests(unittest.TestCase):
    def test_cancelled_media_load_cannot_replace_the_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_args([]).with_updates(
                session_dir=root / "sessions",
                work_dir=root / "work",
                translation_provider="off",
            )
            current = MediaFiles("test://current", "current", root / "current.wav", root / "current.mp4")
            candidate = MediaFiles("test://candidate", "candidate", root / "candidate.wav", root / "candidate.mp4")
            bus = EventBus()
            controller = _Controller()
            application = LiveWindowApplication(config, current, bus, controller)
            try:
                with patch("window.live_window_server.media_cache_status", return_value=("candidate", True, True)), patch(
                    "window.live_window_server.resolve_media_url",
                    side_effect=lambda *_args, **_kwargs: (
                        application.cancel_media_load("request-1"),
                        candidate,
                    )[1],
                ):
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        application.load_media_url(
                            candidate.url,
                            request_id="request-1",
                        )
                self.assertEqual(application.current_media(), current)
                self.assertIsNone(controller.media)
            finally:
                application.close()

    def test_promotion_capture_atomically_seals_browser_originated_final_dom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.wav"
            video_path = root / "video.mp4"
            canonical_path = root / "canonical.json"
            observation_path = root / "evidence" / "browser.json"
            audio_path.write_bytes(b"audio")
            video_path.write_bytes(b"video")
            canonical_path.write_text('{"segments": []}', encoding="utf-8")
            config = parse_args([]).with_updates(
                session_dir=root / "sessions",
                work_dir=root / "work",
                translation_provider="off",
                browser_live_observation_output=observation_path,
                live_speaker_world_tape_output=root / "world-tapes",
                exit_after_browser_live_observation=True,
                validation_canonical=canonical_path,
            )
            media = MediaFiles(
                "test://media",
                "media",
                audio_path,
                video_path,
            )
            bus = EventBus()
            application = LiveWindowApplication(
                config,
                media,
                bus,
                _Controller(),
            )
            try:
                bus.emit("done", {"message": "test done"})
                bindings = application.final_transcript_dom_snapshot_bindings()
                self.assertTrue(bindings["enabled"])
                self.assertTrue(bindings["promotion_grade_requested"])
                self.assertEqual(bindings["errors"], [])
                snapshot = {
                    "schema_version": "final_clustering_dom_snapshot_v1",
                    "world_tape_run_id": bindings["world_tape_run_id"],
                    "capture_surface": "visible_chrome_final_transcript_dom_after_done",
                    "captured_after_done": True,
                    "source_tree_sha256": bindings["source_tree_sha256"],
                    "runtime_config_sha256": bindings["runtime_config_sha256"],
                    "media": dict(bindings["media"]),
                    "browser": {
                        "visibility_state": "visible",
                        "has_focus": True,
                        "webdriver": False,
                    },
                    "rows": [
                        {
                            "index": 0,
                            "text": "Visible final sentence.",
                            "assigned_speaker": "Speaker 1",
                        }
                    ],
                }
                with self.assertRaisesRegex(RuntimeError, "requires a post-done"):
                    application.finish_browser_live_observation("done", None)
                malformed_snapshot = dict(snapshot)
                malformed_snapshot["runtime_config_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    RuntimeError,
                    "runtime configuration mismatch",
                ):
                    application.finish_browser_live_observation(
                        "done",
                        malformed_snapshot,
                    )

                application.record_browser_live_observation(
                    [
                        {
                            "wall_time": 1.0,
                            "playback_time": 0.0,
                            "browser_user_agent": "Chrome/140",
                            "browser_webdriver": False,
                            "browser_visibility_state": "visible",
                            "browser_has_focus": True,
                            "fast_processing": False,
                            "playback_rate": 1.0,
                        }
                    ],
                    batch_sequence=1,
                )
                summary = application.finish_browser_live_observation(
                    "done",
                    snapshot,
                )
                seal = summary["final_transcript_dom_snapshot"]
                snapshot_path = Path(seal["path"])
                self.assertEqual(
                    snapshot_path,
                    observation_path.with_name(
                        "browser.final_transcript_dom_snapshot.json"
                    ).resolve(),
                )
                self.assertNotIn(
                    application.world_tape_recorder.output_dir.resolve(),
                    snapshot_path.parents,
                )
                self.assertEqual(
                    seal["sha256"],
                    hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    json.loads(snapshot_path.read_text(encoding="utf-8")),
                    snapshot,
                )
                observation = json.loads(
                    observation_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    observation["attestation"]["final_transcript_dom_snapshot"],
                    seal,
                )
                self.assertEqual(
                    observation["attestation"]["runtime_config_sha256"],
                    observation["attestation"]["world_tape"][
                        "runtime_config_sha256"
                    ],
                )
                self.assertEqual(
                    observation["attestation"]["world_tape"]["media"][
                        "source_audio_sha256"
                    ],
                    observation["attestation"]["media"]["source_audio_sha256"],
                )
                self.assertEqual(
                    seal["media"]["decoded_samples"],
                    observation["attestation"]["world_tape"]["media"][
                        "decoded_samples"
                    ],
                )
            finally:
                application.close()

    def test_transport_delegates_to_application_and_closes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_args([]).with_updates(
                session_dir=root / "sessions",
                work_dir=root / "work",
                translation_provider="off",
            )
            media = MediaFiles("test://media", "media", root / "audio.wav", root / "video.mp4")
            bus = EventBus()
            controller = _Controller()
            application = LiveWindowApplication(config, media, bus, controller)
            server = WindowServer(
                ("127.0.0.1", 0),
                config,
                media,
                bus,
                controller,
                application=application,
            )

            self.assertIs(server.application, application)
            self.assertEqual(server.current_media(), media)
            self.assertEqual(server.server_address[0], "127.0.0.1")

            with self.assertRaises(OSError):
                WindowServer(
                    ("127.0.0.1", server.server_address[1]),
                    config,
                    media,
                    bus,
                    controller,
                    application=application,
                )

            server.server_close()
            server.server_close()
            application.close()


if __name__ == "__main__":
    unittest.main()
