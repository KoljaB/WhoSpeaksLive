from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from textual.widgets import Button, DataTable, Input, RadioSet, Static, Switch, TabbedContent


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from whospeaks_cli import main as backend
from whospeaks_cli.tui import ConfirmInstallScreen, WhoSpeaksSetupApp


class WhoSpeaksTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_app_mounts_at_wide_and_compact_sizes(self) -> None:
        for size in ((120, 40), (80, 32)):
            app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "setup-tab")
                self.assertEqual(app.query_one("#target-select", RadioSet).pressed_button.id, "target-local")
                self.assertIn("Full local installation", str(app.query_one("#plan-summary", Static).content))
                self.assertFalse(app.query_one("#install-button", Button).disabled)

    async def test_server_target_disables_kroko_and_updates_plan(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click("#target-server")
            await pilot.pause()

            kroko = app.query_one("#kroko-switch", Switch)
            summary = str(app.query_one("#plan-summary", Static).content)
            self.assertTrue(kroko.disabled)
            self.assertFalse(kroko.value)
            self.assertIn("ASR and embeddings server packages", summary)
            self.assertIn("Dependency set: server", summary)

    async def test_install_button_opens_confirmation_without_starting_process(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WHOSPEAKS_CONFIG"] = str(Path(directory) / "config.json")
            try:
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.click("#install-button")
                    await pilot.pause()

                    self.assertIsInstance(app.screen, ConfirmInstallScreen)
                    self.assertIsNotNone(app.pending_install_command)
                    self.assertIn("--target", app.pending_install_command)
                    self.assertIn("local", app.pending_install_command)
                    self.assertIn("--with-kroko", app.pending_install_command)

                    await pilot.click("#cancel-install")
                    await pilot.pause()
                    self.assertIsNone(app.pending_install_command)
                    self.assertIsNone(app.install_process)
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_settings_save_updates_profile_and_file(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            try:
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=(120, 40)) as pilot:
                    app.query_one("#main-tabs", TabbedContent).active = "settings-tab"
                    await pilot.pause()
                    app.query_one("#model-input", Input).value = "small"
                    app.query_one("#port-input", Input).value = "8899"
                    await pilot.click("#save-settings")
                    await pilot.pause()

                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertEqual(app.profile.model, "small")
                    self.assertEqual(app.profile.port, 8899)
                    self.assertEqual(payload["model"], "small")
                    self.assertEqual(payload["port"], 8899)
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_unwritable_profile_is_reported_without_crashing(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(120, 40)) as pilot:
            with mock.patch.object(backend, "save_profile", side_effect=PermissionError("read only")):
                saved = app._save_settings()
            await pilot.pause()

            self.assertFalse(saved)
            self.assertIsNone(app.install_process)

    async def test_doctor_report_populates_both_tables(self) -> None:
        report = backend.DoctorReport(
            "local",
            [
                backend.CheckResult("Python", "ok", "CPython 3.11"),
                backend.CheckResult("ffmpeg", "fail", "Not found", "Install ffmpeg."),
            ],
        )
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(120, 40)) as pilot:
            app._render_report(report)
            await pilot.pause()

            self.assertEqual(app.query_one("#component-table", DataTable).row_count, 2)
            self.assertEqual(app.query_one("#doctor-table", DataTable).row_count, 2)
            self.assertIn("1 failed check", str(app.query_one("#readiness-text", Static).content))

    async def test_automatic_doctor_runs_in_worker_and_updates_readiness(self) -> None:
        report = backend.DoctorReport(
            "local",
            [backend.CheckResult("Python", "ok", "CPython 3.11")],
        )

        def doctor_runner(profile: backend.Profile, mode: str, deep: bool = False) -> backend.DoctorReport:
            self.assertEqual(mode, "local")
            self.assertFalse(deep)
            return report

        app = WhoSpeaksSetupApp(backend.Profile(), doctor_runner=doctor_runner)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()

            readiness = str(app.query_one("#readiness-text", Static).content)
            self.assertIn("Ready: no failed or warning checks", readiness)
            self.assertEqual(app.query_one("#component-table", DataTable).row_count, 1)

    async def test_install_worker_streams_process_and_returns_to_ready_state(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        class FakeProcess:
            stdout = iter(["Installing core\n", "Finished\n"])
            pid = 123

            def wait(self) -> int:
                return 0

            def poll(self) -> int | None:
                return 0

        def popen_factory(command: list[str], **kwargs: object) -> FakeProcess:
            calls.append((command, kwargs))
            return FakeProcess()

        report = backend.DoctorReport("local", [backend.CheckResult("Python", "ok", "CPython 3.11")])
        app = WhoSpeaksSetupApp(
            backend.Profile(),
            auto_doctor=False,
            doctor_runner=lambda *_args, **_kwargs: report,
            popen_factory=popen_factory,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            app.start_install_worker(["python", "-m", "whospeaks_cli", "install"])
            await app.workers.wait_for_complete()
            await app.workers.wait_for_complete()
            await pilot.pause()

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0][-1], "install")
            self.assertIs(calls[0][1]["stdout"], subprocess.PIPE)
            self.assertIs(calls[0][1]["stderr"], subprocess.STDOUT)
            self.assertEqual(app.active_operation, "")
            self.assertIsNone(app.install_process)
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "activity-tab")


if __name__ == "__main__":
    unittest.main()
