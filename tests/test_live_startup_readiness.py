from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from window import youtube_window_diarize_gui


class LiveStartupReadinessTests(unittest.TestCase):
    def test_model_warmup_finishes_before_browser_port_is_bound(self) -> None:
        events: list[str] = []
        args = SimpleNamespace(
            url="microphone://default",
            host="127.0.0.1",
            port=8796,
            asr_backend="remote",
            language="en",
            sentence_tokenizer="auto",
            sentence_language="en",
            embeddings_backend="remote",
            embedding_provider="remote",
            embedding_helper_response_timeout_seconds=60.0,
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_model_preset="nemotron-3.5-160ms-int8",
            realtime_preview_model_dir=None,
            realtime_preview_model_path=None,
            realtime_preview_model="nemotron",
            translation_provider="off",
            translation_model_profile="off",
            translation_target_language=[],
            validate_window_replay=False,
            startup_warmup_before_url=True,
            no_browser=True,
        )

        class FakeController:
            def prepare_before_browser_release(self) -> None:
                events.append("warmup")

            def shutdown(self) -> None:
                events.append("shutdown")

        controller = FakeController()

        class FakeServer:
            server_address = ("127.0.0.1", 8796)

            def __init__(self, *_args: object) -> None:
                events.append("bind")

            def serve_forever(self) -> None:
                events.append("serve")
                raise KeyboardInterrupt

            def server_close(self) -> None:
                events.append("close")

        with (
            mock.patch.object(youtube_window_diarize_gui, "parse_args", return_value=args),
            mock.patch.object(youtube_window_diarize_gui, "resolve_media", return_value=object()),
            mock.patch.object(youtube_window_diarize_gui, "EventBus", return_value=object()),
            mock.patch.object(youtube_window_diarize_gui, "WindowDiarizer", return_value=controller),
            mock.patch.object(youtube_window_diarize_gui, "WindowServer", FakeServer),
        ):
            result = youtube_window_diarize_gui.main()

        self.assertEqual(result, 0)
        self.assertLess(events.index("warmup"), events.index("bind"))
        self.assertLess(events.index("bind"), events.index("serve"))


if __name__ == "__main__":
    unittest.main()
