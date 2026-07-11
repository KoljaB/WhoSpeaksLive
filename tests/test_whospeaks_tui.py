from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from textual.widgets import Button, Checkbox, DataTable, Input, RadioSet, Select, Static, TabbedContent


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from whospeaks_cli import main as backend
from whospeaks_cli.tui import ConfirmInstallScreen, WhoSpeaksSetupApp


class WhoSpeaksTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_layout_fits_measured_terminal_sizes(self) -> None:
        for size in ((80, 28), (100, 32), (140, 32), (140, 42)):
            with self.subTest(size=size):
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()

                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "setup-tab")
                    self.assertEqual(app.query_one("#target-select", RadioSet).pressed_button.id, "target-local")
                    self.assertIn("Full local installation", str(app.query_one("#plan-summary", Static).content))
                    self.assertIn("READY", str(app.query_one("#operation-primary", Static).content))
                    self.assertFalse(app.query_one("#install-button", Button).disabled)
                    self.assertEqual(app.query_one("#title-bar").region.height, 3)
                    self.assertEqual(app.query_one("#status-row").region.height, 2)
                    self.assertEqual(app.query_one("#operation-banner").region.height, 0)
                    self.assertEqual(app.query_one("#setup-options").region.height, 4)
                    self.assertEqual(app.query_one("#setup-actions").region.height, 4)
                    actions = app.query_one("#setup-actions").region
                    self.assertLessEqual(app.query_one("#install-button", Button).region.bottom, actions.bottom)
                    self.assertEqual(app.query_one("#quick-language-select", Select).value, "en")
                    self.assertTrue(app.query_one("#live-speakers-checkbox", Checkbox).value)
                    self.assertGreaterEqual(
                        app.query_one("#language-label").region.x,
                        app.query_one("#target-select").region.right,
                    )
                    self.assertGreaterEqual(
                        app.query_one("#live-speakers-checkbox", Checkbox).region.x,
                        app.query_one("#realtime-select").region.right,
                    )
                    self.assertEqual(str(app.query_one("#live-speakers-checkbox", Checkbox).label), "Live speaker labels")
                    self.assertEqual(app.screen.has_class("compact"), size[0] < 112)
                    self.assertEqual(app.screen.has_class("short"), size[1] < 38)
                    side_is_hidden = app.query_one("#setup-side").styles.display == "none"
                    self.assertEqual(side_is_hidden, size[0] < 112 or size[1] < 38)
                    compact_plan_is_visible = app.query_one("#compact-plan").styles.display != "none"
                    self.assertEqual(compact_plan_is_visible, size[1] < 38)
                    self.assertFalse(app.screen.has_class("narrow"))
                    self.assertEqual(list(app.query("Header")), [])
                    self.assertEqual(list(app.query("LoadingIndicator")), [])

    async def test_short_layout_expands_to_detailed_layout_after_vertical_resize(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause()
            self.assertTrue(app.screen.has_class("short"))
            self.assertEqual(app.query_one("#setup-side").styles.display, "none")
            self.assertNotEqual(app.query_one("#compact-plan").styles.display, "none")

            await pilot.resize_terminal(140, 42)
            await pilot.pause()
            self.assertFalse(app.screen.has_class("short"))
            self.assertNotEqual(app.query_one("#setup-side").styles.display, "none")
            self.assertEqual(app.query_one("#compact-plan").styles.display, "none")

    async def test_server_target_disables_realtime_selection_and_gives_visible_plan_feedback(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.click("#target-server")
            await pilot.pause()

            realtime = app.query_one("#realtime-select", RadioSet)
            summary = str(app.query_one("#plan-summary", Static).content)
            compact = str(app.query_one("#compact-plan", Static).content)
            self.assertTrue(realtime.disabled)
            self.assertEqual(realtime.pressed_button.id, "realtime-off")
            self.assertIn("ASR and embeddings server packages", summary)
            self.assertIn("ASR + embeddings services", compact)
            self.assertNotIn("Dependency set", summary)
            self.assertNotIn("complete,preview", compact)
            self.assertIn("Installation plan updated", str(app.query_one("#operation-primary", Static).content))

    async def test_kroko_profile_ignores_nemotron_preset_selector_on_startup(self) -> None:
        profile = backend.Profile(
            realtime_preview_engine="kroko_onnx",
            realtime_preview_model_preset="nemotron-3.5-560ms-int8",
        )
        app = WhoSpeaksSetupApp(profile, auto_doctor=False)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()

            self.assertEqual(app.query_one("#realtime-select", RadioSet).pressed_button.id, "realtime-kroko")
            self.assertIn("Kroko / Banafo live text", str(app.query_one("#plan-summary", Static).content))

    async def test_main_runtime_controls_drive_saved_launch_settings(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            try:
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=(100, 32)) as pilot:
                    app.query_one("#quick-language-select", Select).value = "de"
                    await pilot.click("#realtime-off")
                    app.query_one("#live-speakers-checkbox", Checkbox).value = False
                    await pilot.pause()

                    self.assertEqual(app.query_one("#language-select", Select).value, "de")
                    self.assertEqual(app.query_one("#realtime-engine-select", Select).value, "off")
                    self.assertTrue(app._save_settings(notify=False))

                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["language"], "de")
                    self.assertEqual(payload["realtime_preview_engine"], "off")
                    self.assertFalse(payload["live_speaker_assignment"])
                    command = backend.build_launch_command(app.profile)
                    self.assertIn("--no-live-speaker-assignment", command)
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_language_selection_prefers_nemotron_then_kroko_then_off(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(140, 32)) as pilot:
            language = app.query_one("#quick-language-select", Select)

            language.value = "he"
            await pilot.pause()
            self.assertEqual(app.query_one("#realtime-select", RadioSet).pressed_button.id, "realtime-kroko")

            language.value = "cy"
            await pilot.pause()
            self.assertEqual(app.query_one("#realtime-select", RadioSet).pressed_button.id, "realtime-off")

            language.value = "de"
            await pilot.pause()
            self.assertEqual(app.query_one("#realtime-select", RadioSet).pressed_button.id, "realtime-nemotron")

    async def test_explicit_incompatible_live_text_keeps_language_warns_and_blocks_actions(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(language="cy", realtime_preview_engine="off"), auto_doctor=False)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.click("#realtime-nemotron")
            await pilot.pause()

            self.assertEqual(app.query_one("#quick-language-select", Select).value, "cy")
            self.assertEqual(app.query_one("#language-select", Select).value, "cy")
            self.assertTrue(app.screen.has_class("preview-incompatible"))
            self.assertIn("Nemotron does not support Welsh", str(app.query_one("#compatibility-note", Static).content))
            self.assertTrue(app.query_one("#launch-button", Button).disabled)
            self.assertTrue(app.query_one("#install-button", Button).disabled)

            await pilot.click("#realtime-off")
            await pilot.pause()
            self.assertFalse(app.screen.has_class("preview-incompatible"))
            self.assertFalse(app.query_one("#launch-button", Button).disabled)
            self.assertFalse(app.query_one("#install-button", Button).disabled)

    async def test_install_confirmation_can_be_cancelled_without_starting_process(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WHOSPEAKS_CONFIG"] = str(Path(directory) / "config.json")
            try:
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=(100, 32)) as pilot:
                    await pilot.click("#install-button")
                    await pilot.pause()

                    self.assertIsInstance(app.screen, ConfirmInstallScreen)
                    self.assertIsNotNone(app.pending_install_command)
                    self.assertIn("--target", app.pending_install_command)
                    self.assertIn("--realtime-preview-engine", app.pending_install_command)
                    self.assertIn("sherpa_onnx", app.pending_install_command)

                    await pilot.click("#cancel-install")
                    await pilot.pause()
                    self.assertIsNone(app.pending_install_command)
                    self.assertIsNone(app.install_process)
                    self.assertIn(
                        "Installation cancelled before start",
                        str(app.query_one("#operation-primary", Static).content),
                    )
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_settings_save_updates_profile_file_and_operation_banner(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            try:
                app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
                async with app.run_test(size=(100, 32)) as pilot:
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
                    self.assertIn("Settings saved", str(app.query_one("#operation-primary", Static).content))
                    self.assertIn(str(config_path), str(app.query_one("#operation-secondary", Static).content))
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_reports_tab_saves_profile_and_starts_report_server(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        calls: list[tuple[list[str], dict[str, object]]] = []

        def popen_factory(command: list[str], **kwargs: object) -> object:
            calls.append((command, kwargs))
            return object()

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            try:
                app = WhoSpeaksSetupApp(
                    backend.Profile(language="es"),
                    auto_doctor=False,
                    popen_factory=popen_factory,
                )
                async with app.run_test(size=(100, 32)) as pilot:
                    app.query_one("#main-tabs", TabbedContent).active = "reports-tab"
                    await pilot.pause()
                    app.query_one("#reports-enabled-checkbox", Checkbox).value = True
                    app.query_one("#report-language-select", Select).value = "es"
                    app.query_one("#report-llm-provider-select", Select).value = "openai"
                    app.query_one("#report-llm-model-input", Input).value = "gpt-4.1-nano"
                    await pilot.click("#save-reports-settings")
                    await pilot.pause()

                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertTrue(payload["reports_enabled"])
                    self.assertEqual(payload["report_language"], "es")
                    self.assertEqual(payload["report_llm_provider"], "openai")

                    await pilot.click("#start-reports-button")
                    await pilot.pause()
                    self.assertEqual(len(calls), 1)
                    command = calls[0][0]
                    self.assertEqual(command[command.index("--report-language") + 1], "es")
                    self.assertEqual(command[command.index("--llm-model") + 1], "gpt-4.1-nano")
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_launcher_stays_open_and_tracks_servers_started_in_new_consoles(self) -> None:
        class FakeServerProcess:
            def __init__(self) -> None:
                self.return_code: int | None = None

            def poll(self) -> int | None:
                return self.return_code

        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        calls: list[tuple[list[str], dict[str, object], FakeServerProcess]] = []

        def popen_factory(command: list[str], **kwargs: object) -> FakeServerProcess:
            process = FakeServerProcess()
            calls.append((command, kwargs, process))
            return process

        with tempfile.TemporaryDirectory() as directory:
            os.environ["WHOSPEAKS_CONFIG"] = str(Path(directory) / "config.json")
            try:
                app = WhoSpeaksSetupApp(
                    backend.Profile(reports_enabled=True),
                    auto_doctor=False,
                    popen_factory=popen_factory,
                )
                app._server_port_accepting = mock.Mock(return_value=False)
                async with app.run_test(size=(100, 32)) as pilot:
                    await pilot.click("#launch-button")
                    await pilot.pause()

                    self.assertEqual(len(calls), 2)
                    reports_call, live_call = calls
                    self.assertIn("--report-language", reports_call[0])
                    self.assertIn("--language", live_call[0])
                    if os.name == "nt":
                        self.assertEqual(reports_call[1]["creationflags"], subprocess.CREATE_NEW_CONSOLE)
                        self.assertEqual(live_call[1]["creationflags"], subprocess.CREATE_NEW_CONSOLE)
                    else:
                        self.assertTrue(reports_call[1]["start_new_session"])
                        self.assertTrue(live_call[1]["start_new_session"])
                    self.assertIn("Live: starting", str(app.query_one("#live-server-state", Static).content))
                    self.assertIn("Reports: starting", str(app.query_one("#reports-server-state", Static).content))
                    self.assertEqual(str(app.query_one("#launch-button", Button).label), "Live running")
                    self.assertTrue(app.query_one("#launch-button", Button).disabled)

                    with mock.patch.object(app, "_server_port_accepting", return_value=True):
                        app.last_server_probe_at = 0.0
                        app._refresh_server_states()
                    self.assertIn("Live: running", str(app.query_one("#live-server-state", Static).content))
                    self.assertIn("Reports: running", str(app.query_one("#reports-server-state", Static).content))

                    live_call[2].return_code = 3
                    app._refresh_server_states()
                    await pilot.pause()

                    self.assertIn("Live: failed", str(app.query_one("#live-server-state", Static).content))
                    self.assertFalse(app.query_one("#launch-button", Button).disabled)
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_unwritable_profile_is_reported_without_crashing(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(100, 32)) as pilot:
            with mock.patch.object(backend, "save_profile", side_effect=PermissionError("read only")):
                saved = app._save_settings()
            await pilot.pause()

            self.assertFalse(saved)
            self.assertIsNone(app.install_process)
            self.assertEqual(app.operation_status, "error")
            self.assertIn("read only", str(app.query_one("#operation-secondary", Static).content))

    async def test_setup_table_keeps_actionable_checks_while_diagnostics_keeps_all(self) -> None:
        report = backend.DoctorReport(
            "local",
            [
                backend.CheckResult("Python", "ok", "CPython 3.12"),
                backend.CheckResult("ffmpeg", "fail", "Not found", "Install ffmpeg."),
                backend.CheckResult("Embedding model caches", "skip", "Deep check only."),
                backend.CheckResult("CUDA visibility", "ok", "CUDA available."),
            ],
        )
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(100, 32)) as pilot:
            app._render_report(report)
            await pilot.pause()

            self.assertEqual(app.query_one("#component-table", DataTable).row_count, 2)
            self.assertEqual(app.query_one("#doctor-table", DataTable).row_count, 4)
            self.assertIn("1 failed check", str(app.query_one("#readiness-text", Static).content))

    async def test_doctor_announces_start_before_blocking_work_and_persists_result(self) -> None:
        started = threading.Event()
        release = threading.Event()
        report = backend.DoctorReport(
            "local",
            [backend.CheckResult("Python", "ok", "CPython 3.12")],
        )

        def doctor_runner(profile: backend.Profile, mode: str, deep: bool = False) -> backend.DoctorReport:
            started.set()
            release.wait(5)
            return report

        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False, doctor_runner=doctor_runner)
        async with app.run_test(size=(80, 28)) as pilot:
            try:
                app.run_doctor_worker(False)
                await pilot.pause()

                self.assertTrue(started.wait(1))
                self.assertEqual(app.active_operation, "doctor")
                self.assertIn("RUNNING", str(app.query_one("#operation-primary", Static).content))
                self.assertIn("Checking system readiness", str(app.query_one("#operation-primary", Static).content))
                self.assertEqual(str(app.query_one("#refresh-button", Button).label), "Checking...")
                self.assertTrue(app.query_one("#install-button", Button).disabled)
            finally:
                release.set()

            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app.active_operation, "")
            self.assertEqual(app.operation_status, "success")
            self.assertIn("Readiness check completed", str(app.query_one("#operation-primary", Static).content))

    async def test_confirmed_install_stays_on_setup_and_shows_live_progress(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple[list[str], dict[str, object]]] = []

        class FakeProcess:
            pid = 123

            @property
            def stdout(self):
                def lines():
                    yield "Collecting faster-whisper\n"
                    started.set()
                    release.wait(5)
                    yield "Successfully installed WhoSpeaks dependencies\n"

                return lines()

            def wait(self) -> int:
                return 0

            def poll(self) -> int | None:
                return 0 if release.is_set() else None

            def terminate(self) -> None:
                release.set()

        def popen_factory(command: list[str], **kwargs: object) -> FakeProcess:
            calls.append((command, kwargs))
            return FakeProcess()

        report = backend.DoctorReport(
            "local",
            [backend.CheckResult("Python", "ok", "CPython 3.12")],
        )
        with tempfile.TemporaryDirectory() as directory:
            os.environ["WHOSPEAKS_CONFIG"] = str(Path(directory) / "config.json")
            app = WhoSpeaksSetupApp(
                backend.Profile(),
                auto_doctor=False,
                doctor_runner=lambda *_args, **_kwargs: report,
                popen_factory=popen_factory,
            )
            try:
                async with app.run_test(size=(80, 28)) as pilot:
                    await pilot.click("#install-button")
                    await pilot.pause()
                    await pilot.click("#confirm-install")
                    await pilot.pause()

                    self.assertTrue(started.wait(1))
                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "setup-tab")
                    self.assertEqual(app.active_operation, "install")
                    self.assertIn("RUNNING", str(app.query_one("#operation-primary", Static).content))
                    self.assertIn("Installing Python packages", str(app.query_one("#operation-primary", Static).content))
                    self.assertIn("Collecting faster-whisper", str(app.query_one("#operation-secondary", Static).content))
                    compact_status = app.query_one("#compact-plan", Static)
                    self.assertIn("INSTALLING", str(compact_status.content))
                    self.assertIn("Collecting faster-whisper", str(compact_status.content))
                    self.assertTrue(compact_status.has_class("status-running"))
                    self.assertEqual(str(app.query_one("#exit-button", Button).label), "Cancel")
                    self.assertEqual(str(app.query_one("#install-button", Button).label), "Installing")
                    self.assertTrue(app.query_one("#install-button", Button).disabled)
                    self.assertTrue(app.query_one("#launch-button", Button).disabled)
                    self.assertFalse(app.query_one("#view-activity-button", Button).disabled)
                    self.assertTrue(app.query_one("#target-select", RadioSet).disabled)
                    self.assertTrue(app.query_one("#realtime-select", RadioSet).disabled)
                    self.assertTrue(app.query_one("#save-settings", Button).disabled)

                    release.set()
                    await app.workers.wait_for_complete()
                    await app.workers.wait_for_complete()
                    await pilot.pause()

                    self.assertEqual(len(calls), 1)
                    self.assertIs(calls[0][1]["stdout"], subprocess.PIPE)
                    self.assertIs(calls[0][1]["stderr"], subprocess.STDOUT)
                    self.assertEqual(app.active_operation, "")
                    self.assertEqual(app.operation_status, "success")
                    self.assertIsNone(app.install_process)
                    self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "setup-tab")
            finally:
                release.set()
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

    async def test_activity_button_and_clear_action_are_visible(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        async with app.run_test(size=(80, 28)) as pilot:
            await pilot.click("#view-activity-button")
            await pilot.pause()
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "activity-tab")

            await pilot.click("#clear-log")
            await pilot.pause()
            self.assertIn("Activity cleared", str(app.query_one("#operation-primary", Static).content))

    def test_installer_output_is_mapped_to_human_readable_phases(self) -> None:
        app = WhoSpeaksSetupApp(backend.Profile(), auto_doctor=False)
        app.operation_step = "Running installer"
        self.assertEqual(app._install_step_for_line("PyTorch install selection: CUDA"), "Installing PyTorch runtime")
        self.assertEqual(app._install_step_for_line("Downloading numpy.whl"), "Installing Python packages")
        self.assertEqual(app._install_step_for_line("Downloading Nemotron model archive"), "Preparing Nemotron realtime ASR")
        self.assertEqual(app._install_step_for_line("Building Kroko native runtime"), "Preparing Kroko realtime ASR")
        self.assertEqual(app._install_step_for_line("Saved profile"), "Saving configuration")


if __name__ == "__main__":
    unittest.main()
