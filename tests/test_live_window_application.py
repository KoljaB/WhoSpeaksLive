from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from window.live_window_server import LiveWindowApplication, WindowServer
from window.window_cli import parse_args
from window.window_domain import MediaFiles
from window.window_events import EventBus


class _Controller:
    def __init__(self) -> None:
        self.stops = 0
        self.media: MediaFiles | None = None

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

            server.server_close()
            server.server_close()
            application.close()


if __name__ == "__main__":
    unittest.main()
