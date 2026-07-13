from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from textual.widgets import Checkbox, Input, Select, Static, TabbedContent


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from whospeaks_cli import main as backend
from whospeaks_cli.tui import WhoSpeaksSetupApp
from whospeaks_cli.tui_state import PendingAction


class TranslationTuiContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_translation_settings_save_and_live_starts_during_sidecar_warmup(self) -> None:
        class FakeServerProcess:
            def poll(self) -> None:
                return None

        calls: list[list[str]] = []

        def popen_factory(command: list[str], **_kwargs: object) -> FakeServerProcess:
            calls.append(command)
            return FakeServerProcess()

        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            try:
                profile = backend.Profile(
                    translation_enabled=True,
                    translation_provider="sidecar",
                    translation_python="translation-python",
                    translation_port=8897,
                )
                app = WhoSpeaksSetupApp(profile, auto_doctor=False, popen_factory=popen_factory)
                translation_ready = False
                app._server_port_accepting = lambda _host, port: translation_ready and port == 8897
                async with app.run_test(size=(120, 36)) as pilot:
                    app.query_one("#main-tabs", TabbedContent).active = "translation-tab"
                    await pilot.pause()
                    app.query_one("#translation-targets-input", Input).value = "de fr,ja"
                    app.query_one("#translation-max-targets-input", Input).value = "2"
                    app.query_one("#translation-model-profile-select", Select).value = "nllb-200-600m"
                    await pilot.click("#save-translation-settings")
                    await pilot.pause()

                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertTrue(payload["translation_enabled"])
                    self.assertEqual(payload["translation_provider"], "sidecar")
                    self.assertEqual(payload["translation_target_languages"], "de,fr")
                    self.assertEqual(payload["translation_model_profile"], "nllb-200-600m")
                    self.assertTrue(app.query_one("#translation-enabled-checkbox", Checkbox).value)

                    app.query_one("#main-tabs", TabbedContent).active = "setup-tab"
                    await pilot.pause()
                    app.action_launch()
                    await pilot.pause()

                    self.assertEqual(len(calls), 2)
                    self.assertEqual(
                        app._coordinator.snapshot.pending_action,
                        PendingAction.NONE,
                    )
                    translation_command, live_command = calls
                    self.assertEqual(translation_command[0], "translation-python")
                    self.assertIn("window.translation_server", translation_command)
                    self.assertIn("--translation-provider", live_command)
                    self.assertLess(calls.index(translation_command), calls.index(live_command))

                    translation_ready = True
                    app.last_server_probe_at = 0.0
                    app._refresh_server_states()
                    await pilot.pause()

                    self.assertEqual(len(calls), 2)
                    self.assertIn(
                        "Translation: running",
                        str(app.query_one("#translation-server-state", Static).content),
                    )
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config


if __name__ == "__main__":
    unittest.main()
