from __future__ import annotations

import dataclasses
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE_AVAILABLE:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractButton,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDialog,
        QFrame,
        QLabel,
        QLineEdit,
        QSizePolicy,
    )

    from whospeaks_gui.demo import DEMO_STATES, DemoLauncherController
    from whospeaks_gui.main import _configure_fonts
    from whospeaks_gui.pages import (
        LanguageTargetSelector,
        PathPicker,
        SettingsHelpDialog,
        _SETTINGS_HELP,
        suitable_openai_llm_models,
    )
    from whospeaks_gui.settings_help import SETTINGS_MORE_HELP
    from whospeaks_gui.tokens import COLORS, COMPACT_RAIL_WIDTH, application_style
    from whospeaks_gui.window import LauncherWindow
    from whospeaks_cli.cli_diagnostics import CheckResult, DoctorReport
    from whospeaks_cli.launcher_controller import LauncherController
    from whospeaks_cli.profiles import PROVIDER_PRESETS, Profile
    from window.language_config import SUPPORTED_LANGUAGE_CONFIGS


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is required for launcher tests")
class WhoSpeaksGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication(["whospeaks-gui-tests"])
        family = _configure_fonts(cls.app)
        cls.app.setStyleSheet(application_style(family))

    def make_window(self, state: str = "ready") -> LauncherWindow:
        window = LauncherWindow(
            DemoLauncherController(state),
            auto_check=False,
            reduced_motion=True,
        )
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)
        return window

    def test_gui_starts_on_overview_and_closes_cleanly(self) -> None:
        window = self.make_window()

        self.assertEqual(window.pages.currentIndex(), 0)
        self.assertEqual(window.overview.header.title.text(), "Ready to launch")
        self.assertEqual(window.accessibleName(), "WhoSpeaks desktop launcher")

        window.close()
        self.app.processEvents()
        self.assertFalse(window.isVisible())

    def test_navigation_and_keyboard_shortcuts_are_functional(self) -> None:
        window = self.make_window()

        QTest.keySequence(window, "Ctrl+,")
        self.app.processEvents()
        self.assertEqual(window.pages.currentIndex(), 2)
        self.assertTrue(window.sidebar.buttons[2].isChecked())

        window.navigate(1)
        self.assertEqual(window.pages.currentIndex(), 1)
        self.assertEqual(window.sidebar.buttons[1].accessibleName(), "Diagnostics")
        self.assertNotEqual(
            window.overview.primary_button.focusPolicy(),
            Qt.FocusPolicy.NoFocus,
        )

    def test_diagnostics_uses_full_width_table_without_horizontal_scrolling(self) -> None:
        window = self.make_window("ready")
        window.resize(1440, 900)
        window.navigate(1)
        self.app.processEvents()

        table = window.diagnostics.table
        self.assertFalse(hasattr(window.diagnostics, "inspector"))
        self.assertEqual(
            table.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(table.horizontalScrollBar().maximum(), 0)
        self.assertEqual(
            window.diagnostics.model.horizontalHeaderItem(3).text(),
            "Recommended action",
        )

    def test_diagnostic_rows_are_read_only_and_refresh_does_not_select_them(self) -> None:
        window = self.make_window("ready")
        page = window.diagnostics
        initial = DoctorReport(
            "remote",
            [
                CheckResult("Python", "ok", "ready"),
                CheckResult("ASR", "ok", "available"),
                CheckResult("Embeddings", "warn", "warming"),
            ],
        )
        page.set_report(initial)
        page.table.selectRow(1)
        self.assertEqual(
            page.table.selectionMode(),
            page.table.SelectionMode.NoSelection,
        )
        self.assertEqual(page.table.selectionModel().selectedRows(), [])

        page.set_report(DoctorReport("remote", list(initial.checks)))
        self.assertEqual(page.table.selectionModel().selectedRows(), [])

    def test_about_page_credits_the_creator(self) -> None:
        window = self.make_window("ready")
        window.navigate(4)
        credit = window.about.creator_credit
        self.assertIn("https://github.com/KoljaB", credit.text())
        self.assertIn("https://github.com/KoljaB/WhoSpeaksLive", credit.text())
        self.assertIn("#49C7B1", credit.text())
        self.assertTrue(credit.openExternalLinks())

    def test_demo_states_are_deterministic_and_have_distinct_terminal_status(self) -> None:
        expected = {
            "first_run": ("Finish setup", "Install components"),
            "ready": ("Ready to launch", "Launch WhoSpeaks"),
            "starting": ("Starting WhoSpeaks", "Preparing live window…"),
            "running": ("WhoSpeaks is running", "Open live window"),
            "partial_failure": ("Running with an issue", "Open live window"),
            "failed": ("WhoSpeaks could not start", "Retry launch"),
        }
        for state, (title, action) in expected.items():
            with self.subTest(state=state):
                window = self.make_window(state)
                self.assertEqual(window.overview.header.title.text(), title)
                self.assertEqual(window.overview.primary_button.text(), action)
                self.assertTrue(window.overview.summary.accessibleName())

    def test_internal_interaction_specimen_is_not_a_product_route(self) -> None:
        window = self.make_window("ready")

        self.assertNotIn("interaction_states", DEMO_STATES)
        self.assertEqual(window.pages.count(), 5)

    def test_partial_failure_keeps_live_service_available_and_explains_recovery(self) -> None:
        window = self.make_window("partial_failure")

        self.assertEqual(window.overview.service_rows["live"].status_mark.status(), "running")
        self.assertEqual(window.overview.service_rows["translation"].status_mark.status(), "failed")
        self.assertTrue(window.overview.side_code.isVisible())
        self.assertIn("config.json", window.overview.side_code.text())
        self.assertTrue(window.overview.stop_button.isVisible())

    def test_open_live_window_uses_the_saved_profile_endpoint(self) -> None:
        window = self.make_window("running")

        with mock.patch(
            "whospeaks_gui.window.QDesktopServices.openUrl",
            return_value=True,
        ) as open_url:
            window.overview.primary_button.click()

        opened = open_url.call_args.args[0]
        self.assertEqual(opened.toString(), "http://127.0.0.1:8796")

    def test_running_interface_endpoints_are_links_and_open_their_urls(self) -> None:
        window = self.make_window("running")
        live = window.overview.service_rows["live"].endpoint
        reports = window.overview.service_rows["reports"].endpoint

        self.assertTrue(live.isEnabled())
        self.assertTrue(reports.isEnabled())
        self.assertFalse(live.icon().isNull())
        self.assertFalse(reports.icon().isNull())
        self.assertIn("Open Live window", live.accessibleName())
        self.assertIn("Open Meeting Intelligence", reports.accessibleName())

        with mock.patch(
            "whospeaks_gui.window.QDesktopServices.openUrl",
            return_value=True,
        ) as open_url:
            reports.click()

        self.assertEqual(open_url.call_args.args[0].toString(), "http://127.0.0.1:8798")

    def test_interface_endpoints_are_disabled_until_the_services_run(self) -> None:
        window = self.make_window("ready")

        for key in ("live", "reports"):
            self.assertFalse(window.overview.service_rows[key].endpoint.isEnabled())
        self.assertFalse(window.overview.profile_labels["browser"].isEnabled())
        self.assertFalse(window.overview.profile_labels["reports"].isEnabled())

    def test_remote_backends_are_visible_with_their_configured_endpoints(self) -> None:
        window = self.make_window("ready")

        self.assertTrue(window.overview.service_rows["macos_asr"].isVisible())
        self.assertTrue(window.overview.service_rows["macos_embeddings"].isVisible())
        self.assertEqual(
            window.overview.service_rows["macos_asr"].endpoint.text(),
            "http://127.0.0.1:8650",
        )
        self.assertEqual(
            window.overview.service_rows["macos_embeddings"].endpoint.text(),
            "http://127.0.0.1:8660",
        )

    def test_local_core_components_are_visible_and_explain_their_shared_process(self) -> None:
        profile = Profile.from_mapping(
            {
                "mode": "local",
                "reports_enabled": False,
                "translation_enabled": False,
                "model": "large-v2",
                "provider_preset": "smoke",
            }
        )
        controller = LauncherController(
            profile,
            profile_saver=lambda _profile: Path("unused.json"),
        )
        controller.report = DoctorReport("local", [CheckResult("Python", "ok", "ready")])
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        asr = window.overview.service_rows["macos_asr"]
        embeddings = window.overview.service_rows["macos_embeddings"]
        self.assertTrue(asr.isVisible())
        self.assertTrue(embeddings.isVisible())
        self.assertEqual(asr.title.text(), "Final ASR")
        self.assertEqual(embeddings.title.text(), "Speaker embeddings")
        self.assertEqual(asr.subtitle.text(), "Runs inside Live window")
        self.assertIn("large-v2", asr.endpoint.text())
        self.assertEqual(asr.status_label.text(), "Starts with Live window")
        self.assertEqual(embeddings.status_label.text(), "Starts with Live window")

    def test_unavailable_remote_backend_blocks_launch_and_explains_recovery(self) -> None:
        profile = Profile.from_mapping(
            {
                "mode": "remote",
                "reports_enabled": False,
                "translation_enabled": False,
                "remote_asr_url": "http://asr.example:8650",
                "remote_embeddings_url": "http://embeddings.example:8660",
            }
        )
        controller = LauncherController(
            profile,
            profile_saver=lambda _profile: Path("unused.json"),
        )
        controller.report = DoctorReport("remote", [CheckResult("Python", "ok", "ready")])
        controller.servers.observe_backend("macos_asr", available=True)
        controller.servers.observe_backend("macos_embeddings", available=False)
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "backend_unavailable")
        self.assertEqual(window.overview.header.title.text(), "Remote backend unavailable")
        self.assertTrue(window.overview.primary_button.isEnabled())
        self.assertEqual(window.overview.primary_button.text(), "Retry remote services")
        self.assertEqual(
            window.overview.service_rows["macos_embeddings"].status_label.text(),
            "Unavailable",
        )
        self.assertIn("embeddings.example", window.overview.side_code.text())

    def test_service_explanations_are_visible_without_disclosure_buttons(self) -> None:
        window = self.make_window("starting")
        for row in window.overview.service_rows.values():
            if row.isVisible():
                self.assertGreaterEqual(row.height(), 112)
                self.assertTrue(row.extra.isVisible())
                self.assertTrue(row.extra.text().strip())
                self.assertFalse(hasattr(row, "disclosure"))

        window.overview.command_button.click()
        self.app.processEvents()

        self.assertEqual(window.overview.operational_state, "ready")
        self.assertEqual(window.overview.primary_button.text(), "Launch WhoSpeaks")

    def test_real_partial_snapshot_drives_recovery_shell(self) -> None:
        class RunningProcess:
            pid = 9911

            @staticmethod
            def poll() -> None:
                return None

        profile = Profile.from_mapping(
            {
                "mode": "remote",
                "reports_enabled": True,
                "translation_enabled": True,
                "translation_provider": "sidecar",
            }
        )
        controller = LauncherController(
            profile,
            profile_saver=lambda _profile: Path("unused.json"),
        )
        for kind in ("live", "reports"):
            controller.servers.begin(kind, RunningProcess())
            controller.servers.observe(kind, listening=True, probe_due=True)
        controller.servers.fail_start("translation")
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "partial_failure")
        self.assertEqual(window.overview.header.title.text(), "Running with an issue")
        self.assertTrue(window.overview.command_button.isVisible())
        self.assertEqual(window.overview.command_button.text(), "Retry Translation")

    def test_live_window_failure_has_consistent_recovery_and_diagnostic_summary(self) -> None:
        class RunningProcess:
            pid = 9912

            @staticmethod
            def poll() -> None:
                return None

        profile = Profile.from_mapping({"reports_enabled": True, "translation_enabled": False})
        controller = LauncherController(
            profile,
            profile_saver=lambda _profile: Path("unused.json"),
        )
        controller.report = DoctorReport("local", [CheckResult("Python", "ok", "ready")])
        controller.servers.fail_start("live")
        controller.servers.begin("reports", RunningProcess())
        controller.servers.observe("reports", listening=True, probe_due=True)
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "partial_failure")
        self.assertIn("Meeting Intelligence is available", window.overview.header.subtitle.text())
        self.assertEqual(window.overview.summary.state_label.text(), "DEGRADED")
        self.assertEqual(window.overview.primary_button.text(), "Retry Live window")
        self.assertIn("Live window stopped", window.overview.failure_headline.text())
        self.assertNotIn("Translation", window.overview.failure_headline.text())

    def test_probe_completion_does_not_reset_starting_button(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.overview.set_operational_state("starting")
        self.addCleanup(window.bridge.close)

        window._worker_completed("probe", object())

        self.assertIn("Preparing", window.overview.primary_button.text())
        self.assertNotIn("Launch WhoSpeaks", window.overview.primary_button.text())

    def test_background_snapshot_preserves_first_run_setup_choices(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        self.addCleanup(window.bridge.close)

        window.overview.setup_target.setCurrentIndex(
            window.overview.setup_target.findData("cpu")
        )
        window.overview.setup_live_text.setCurrentIndex(
            window.overview.setup_live_text.findData("sherpa_onnx")
        )
        selected_target = window.overview.setup_target.currentData()
        selected_engine = window.overview.setup_live_text.currentData()

        window.apply_snapshot(controller.snapshot)

        self.assertEqual(window.overview.setup_target.currentData(), selected_target)
        self.assertEqual(window.overview.setup_live_text.currentData(), selected_engine)

    def test_idle_duplicate_snapshot_does_not_rebuild_overview(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        self.addCleanup(window.bridge.close)

        with mock.patch.object(window.overview, "set_report") as set_report:
            window.apply_snapshot(controller.snapshot)

        set_report.assert_not_called()

    def test_launcher_has_no_continuous_paint_timer(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=False)
        self.addCleanup(window.bridge.close)

        self.assertFalse(window.animation_timer.isActive())

    def test_idle_launcher_does_not_keep_polling(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        original_refresh = controller.refresh_services
        controller.refresh_services = mock.Mock(wraps=original_refresh)
        window = LauncherWindow(controller, auto_check=False, reduced_motion=False)
        self.addCleanup(window.bridge.close)

        QTest.qWait(2200)

        self.assertEqual(controller.refresh_services.call_count, 0)
        self.assertFalse(window.probe_timer.isActive())

    def test_real_terminal_launch_error_drives_complete_failure_shell(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        controller.coordinator.finish_operation(
            "error",
            "WhoSpeaks did not start",
            "Browser port is already in use.",
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "failed")
        self.assertEqual(window.overview.primary_button.text(), "Retry launch")

    def test_existing_external_live_window_is_openable_not_setup_required(self) -> None:
        controller = LauncherController(
            Profile.from_mapping(
                {
                    "mode": "remote",
                    "reports_enabled": False,
                    "translation_enabled": False,
                }
            ),
            profile_saver=lambda _profile: Path("unused.json"),
            remote_backend_probe=lambda _url: True,
        )
        controller.report = DoctorReport(
            "remote",
            [CheckResult("Browser UI port", "ok", "WhoSpeaks is already running")],
        )
        controller.servers.observe_backend("macos_asr", available=True)
        controller.servers.observe_backend("macos_embeddings", available=True)
        controller.servers.observe("live", listening=True, probe_due=True)
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "running_external")
        self.assertEqual(window.overview.header.title.text(), "WhoSpeaks is already running")
        self.assertEqual(window.overview.primary_button.text(), "Open live window")
        self.assertFalse(window.overview.stop_button.isVisible())

    def test_unrelated_port_conflict_opens_settings_not_installation(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        controller.report = DoctorReport(
            "local",
            [
                CheckResult(
                    "Browser UI port",
                    "fail",
                    "127.0.0.1:8796 is used by another process",
                    "Choose another port in Settings.",
                )
            ],
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        self.assertEqual(window.overview.operational_state, "port_conflict")
        self.assertEqual(window.overview.header.title.text(), "Port already in use")
        self.assertEqual(window.overview.primary_button.text(), "Open settings")
        window.overview.primary_button.click()
        self.assertEqual(window.pages.currentIndex(), 2)

    def test_settings_validation_and_persistence_run_through_controller_bridge(self) -> None:
        window = self.make_window("settings")
        host = window.settings.fields["host"]
        self.assertIsInstance(host, QLineEdit)
        assert isinstance(host, QLineEdit)
        host.setText("localhost")
        window.settings.save_button.click()
        for _ in range(100):
            self.app.processEvents()
            if (
                window.controller.profile.host == "localhost"
                and "Saved" in window.settings.status.text()
            ):
                break
            QTest.qWait(10)

        self.assertEqual(window.controller.profile.host, "localhost")
        self.assertIn("Saved", window.settings.status.text())

    def test_real_validation_error_opens_and_marks_the_exact_field(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1200, 760)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)
        host = window.settings.fields["host"]
        assert isinstance(host, QLineEdit)
        host.clear()

        window.settings.save_button.click()
        for _ in range(100):
            self.app.processEvents()
            if host.property("invalid"):
                break
            QTest.qWait(10)

        self.assertEqual(window.pages.currentWidget(), window.settings)
        self.assertTrue(host.property("invalid"))
        self.assertIn("Browser host cannot be empty", host.accessibleDescription())
        self.assertEqual(window.settings.section_list.currentRow(), 0)

    def test_long_path_validation_stays_inside_settings_content_width(self) -> None:
        window = self.make_window("settings")
        window.navigate(2)
        long_path = (
            r"C:\Users\Start\AppData\Roaming\uv\tools\whospeaks\Scripts\python.exe"
        )
        message = f"ProfileValidationError: The configured Python executable does not exist: {long_path}"

        window.settings.show_validation_error(
            "realtime_preview_python",
            message,
            value=long_path,
        )
        self.app.processEvents()

        section = window.settings.sections.currentWidget()
        error = window.settings._validation_rows[-1][1]
        self.assertTrue(error.wordWrap())
        self.assertEqual(
            error.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self.assertLessEqual(section.widget().width(), section.viewport().width())
        error_right = error.mapTo(window, QPoint(error.width(), 0)).x()
        section_right = section.viewport().mapTo(
            window,
            QPoint(section.viewport().width(), 0),
        ).x()
        self.assertLessEqual(error_right, section_right)

    def test_irrelevant_preview_settings_are_hidden_without_losing_values(self) -> None:
        window = self.make_window("settings")
        engine = window.settings.fields["realtime_preview_engine"]
        preset = window.settings.fields["realtime_preview_model_preset"]
        self.assertIsInstance(engine, QComboBox)
        assert isinstance(engine, QComboBox)

        engine.setCurrentIndex(engine.findData("off"))
        self.app.processEvents()

        self.assertFalse(preset.isVisible())
        self.assertTrue(preset.isEnabled())
        self.assertTrue(preset.accessibleName())

    def test_speaker_preset_immediately_updates_both_provider_fields(self) -> None:
        window = self.make_window("settings")
        preset = window.settings.fields["provider_preset"]
        final_provider = window.settings.fields["embedding_provider"]
        live_provider = window.settings.fields["live_speaker_embedding_provider"]
        self.assertIsInstance(preset, QComboBox)
        self.assertIsInstance(final_provider, QLineEdit)
        self.assertIsInstance(live_provider, QLineEdit)
        assert isinstance(preset, QComboBox)

        preset.setCurrentIndex(preset.findData("public_quality"))
        self.app.processEvents()

        expected = PROVIDER_PRESETS["public_quality"]
        self.assertEqual(final_provider.text(), expected.embedding_provider)
        self.assertEqual(live_provider.text(), expected.live_speaker_embedding_provider)

    def test_live_engine_only_offers_compatible_real_model_presets(self) -> None:
        window = self.make_window("settings")
        engine = window.settings.fields["realtime_preview_engine"]
        model = window.settings.fields["realtime_preview_model_preset"]
        assert isinstance(engine, QComboBox) and isinstance(model, QComboBox)

        engine.setCurrentIndex(engine.findData("kroko_onnx"))
        self.app.processEvents()

        self.assertEqual(
            [model.itemData(index) for index in range(model.count())],
            ["community-64l", "pro-16l"],
        )
        self.assertNotIn("nemotron", " ".join(str(model.itemData(i)) for i in range(model.count())))

    def test_translation_targets_use_a_finite_multiselect_and_exclude_source(self) -> None:
        window = self.make_window("settings")
        selector = window.settings.fields["translation_target_languages"]
        self.assertIsInstance(selector, LanguageTargetSelector)
        assert isinstance(selector, LanguageTargetSelector)

        selector.set_source_language("en")
        selector.set_selected_codes("en,de,fr")

        self.assertEqual(set(selector.selected_codes()), {"de", "fr"})
        self.assertEqual(
            set(str(window.settings.values()["translation_target_languages"]).split(",")),
            {"de", "fr"},
        )
        english = next(
            selector._item(row)
            for row in range(selector.count())
            if selector.itemData(row) == "en"
        )
        self.assertFalse(english.isEnabled())

    def test_test_only_providers_and_redundant_overview_buttons_are_absent(self) -> None:
        window = self.make_window("ready")
        translation_provider = window.settings.fields["translation_provider"]
        live_engine = window.settings.fields["realtime_preview_engine"]
        assert isinstance(translation_provider, QComboBox)
        assert isinstance(live_engine, QComboBox)

        self.assertEqual(translation_provider.findData("mock"), -1)
        self.assertEqual(live_engine.findData("mock"), -1)
        visible_texts = {
            button.text()
            for button in window.overview.findChildren(QAbstractButton)
            if button.isVisible()
        }
        self.assertNotIn("View details", visible_texts)
        self.assertNotIn("Edit settings", visible_texts)

    def test_server_profile_hides_controller_only_settings_and_actions(self) -> None:
        window = self.make_window("settings")
        deployment = window.settings.fields["mode"]
        assert isinstance(deployment, QComboBox)

        deployment.setCurrentIndex(deployment.findData("server"))
        self.app.processEvents()

        for index in (1, 2, 4, 5, 6):
            self.assertTrue(window.settings.section_list.item(index).isHidden())
        for field in ("host", "port", "realtime_preview_engine", "reports_enabled", "translation_enabled"):
            self.assertFalse(window.settings.fields[field].isVisible())
        self.assertIn("not started", window.settings.context_help_detail.text())
        self.assertEqual(window.settings.values()["mode"], "server")

    def test_every_settings_control_has_tooltip_and_accessible_help(self) -> None:
        window = self.make_window("settings")

        for field, widget in window.settings.fields.items():
            with self.subTest(field=field):
                self.assertTrue(widget.toolTip().strip())
                self.assertTrue(widget.accessibleDescription().strip())
        self.assertTrue(window.settings.text_embedding_preset.toolTip().strip())
        self.assertTrue(window.settings.section_list.toolTip().strip())
        self.assertTrue(window.settings.save_button.toolTip().strip())
        self.assertTrue(window.settings.discard_button.toolTip().strip())

    def test_focused_control_updates_persistent_context_help(self) -> None:
        window = self.make_window("settings")
        window.navigate(2)
        deployment = window.settings.fields["mode"]
        language = window.settings.fields["language"]
        runtime = window.settings.fields["realtime_preview_model_dir"]
        assert isinstance(deployment, QComboBox)
        assert isinstance(language, QComboBox)
        assert isinstance(runtime, PathPicker)

        deployment.setCurrentIndex(deployment.findData("remote"))
        deployment.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()
        self.assertEqual(window.settings.context_help_title.text(), "Deployment")
        self.assertEqual(window.settings.context_help_value.text(), "Remote client (models run elsewhere)")
        self.assertIn("another machine", window.settings.context_help_detail.text())

        language.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()
        self.assertEqual(window.settings.context_help_title.text(), "Language")
        self.assertIn("transcription", window.settings.context_help_detail.text())

        window.settings.section_list.setCurrentRow(1)
        runtime.edit.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()
        self.assertEqual(window.settings.context_help_title.text(), "Nemotron model folder")
        self.assertEqual(window.settings.context_help_value.text(), "Automatic")

    def test_f1_opens_more_help_for_focused_control(self) -> None:
        window = self.make_window("settings")
        window.navigate(2)
        deployment = window.settings.fields["mode"]
        assert isinstance(deployment, QComboBox)
        deployment.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()

        with mock.patch.object(SettingsHelpDialog, "exec", autospec=True) as show_dialog:
            QTest.keyClick(deployment, Qt.Key.Key_F1)
            self.app.processEvents()

        show_dialog.assert_called_once()
        dialog = window.settings._help_dialog
        assert dialog is not None
        self.assertEqual(dialog.windowTitle(), "Deployment help")
        self.assertIn("Local on NVIDIA GPU starts the browser app", dialog.help_text.toPlainText())
        self.assertIn("allocates no CUDA memory or VRAM", dialog.help_text.toPlainText())
        self.assertIn("trusted LAN", dialog.help_text.toPlainText())

    def test_every_f1_target_has_distinct_expanded_help(self) -> None:
        window = self.make_window("settings")
        special_targets = {
            "settings_sections": window.settings.section_list,
            "save_changes": window.settings.save_button,
            "discard_changes": window.settings.discard_button,
            "text_embedding_preset": window.settings.text_embedding_preset,
        }

        self.assertEqual(set(SETTINGS_MORE_HELP), set(_SETTINGS_HELP))
        for key, (_summary, context) in _SETTINGS_HELP.items():
            with self.subTest(key=key):
                expanded = SETTINGS_MORE_HELP[key]
                self.assertNotEqual(expanded.strip(), context.strip())
                self.assertGreater(len(expanded), len(context))
                self.assertIn("\n\n", expanded)
                widget = window.settings.fields.get(key) or special_targets[key]
                rendered = window.settings._expanded_control_help(key, widget)
                self.assertGreaterEqual(len(rendered), len(expanded))

    def test_complex_f1_help_includes_runtime_specific_details(self) -> None:
        window = self.make_window("settings")
        settings = window.settings

        final_provider = settings.fields["embedding_provider"]
        live_provider = settings.fields["live_speaker_embedding_provider"]
        target_languages = settings.fields["translation_target_languages"]
        final_help = settings._expanded_control_help("embedding_provider", final_provider)
        live_help = settings._expanded_control_help(
            "live_speaker_embedding_provider", live_provider
        )
        language_help = settings._expanded_control_help(
            "translation_target_languages", target_languages
        )

        self.assertIn("committed sentence audio", final_help)
        self.assertIn("weights scale vectors and are not probabilities", final_help)
        self.assertIn("Selected preset:", final_help)
        self.assertIn("provisional speaker names", live_help)
        self.assertIn("Final provider", live_help)
        for code, config in SUPPORTED_LANGUAGE_CONFIGS.items():
            self.assertIn(f"{code} — {config.display_name}", language_help)

        dialog = SettingsHelpDialog(
            "Target languages",
            _SETTINGS_HELP["translation_target_languages"][0],
            language_help,
            current_value="German, French",
            parent=settings,
        )
        self.assertLessEqual(dialog.maximumHeight(), 560)
        self.assertTrue(dialog.help_text.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def test_remote_profile_hides_local_asr_runtime_settings(self) -> None:
        window = self.make_window("settings")
        deployment = window.settings.fields["mode"]
        assert isinstance(deployment, QComboBox)

        deployment.setCurrentIndex(deployment.findData("remote"))
        window.settings.section_list.setCurrentRow(1)
        self.app.processEvents()

        for field in ("model", "device", "compute_type", "embedding_python"):
            self.assertFalse(window.settings.fields[field].isVisible())
        self.assertTrue(window.settings.fields["vad_backend"].isVisible())

    def test_model_and_runtime_paths_use_purpose_built_editors(self) -> None:
        window = self.make_window("settings")
        model = window.settings.fields["model"]
        assert isinstance(model, QComboBox)
        self.assertTrue(model.isEditable())
        self.assertGreaterEqual(model.findData("large-v3"), 0)
        model.setEditText("org/account-specific-whisper")
        self.assertEqual(window.settings.values()["model"], "org/account-specific-whisper")

        for field in (
            "realtime_preview_model_dir",
            "realtime_preview_python",
            "embedding_python",
            "translation_python",
        ):
            picker = window.settings.fields[field]
            self.assertIsInstance(picker, PathPicker)
            self.assertEqual(picker.browse_button.text(), "Browse…")

    def test_translation_provider_switches_to_visible_provider_defaults(self) -> None:
        window = self.make_window("settings")
        enabled = window.settings.fields["translation_enabled"]
        provider = window.settings.fields["translation_provider"]
        key_env = window.settings.fields["translation_api_key_env"]
        assert isinstance(enabled, QCheckBox)
        assert isinstance(provider, QComboBox)
        assert isinstance(key_env, QLineEdit)
        enabled.setChecked(True)

        provider.setCurrentIndex(provider.findData("deepl"))
        self.assertEqual(key_env.text(), "DEEPL_API_KEY")
        provider.setCurrentIndex(provider.findData("openai_compatible"))
        self.assertEqual(key_env.text(), "")

        key_env.setText("INTERNAL_TRANSLATION_KEY")
        provider.setCurrentIndex(provider.findData("azure_translator"))
        self.assertEqual(key_env.text(), "AZURE_TRANSLATOR_KEY")

    def test_saved_custom_translation_endpoint_survives_settings_mount(self) -> None:
        window = self.make_window("settings")
        profile = Profile.from_mapping({
            "translation_enabled": True,
            "translation_provider": "openai_compatible",
            "translation_base_url": "http://127.0.0.1:5000",
            "translation_model": "company/translator",
        })

        window.settings.set_profile(profile)

        base_url = window.settings.fields["translation_base_url"]
        model = window.settings.fields["translation_model"]
        assert isinstance(base_url, QLineEdit) and isinstance(model, QLineEdit)
        self.assertEqual(base_url.text(), "http://127.0.0.1:5000")
        self.assertEqual(model.text(), "company/translator")

    def test_first_run_install_plan_uses_selected_translation_runtime(self) -> None:
        window = self.make_window("first_run")
        translation = window.overview.setup_translation_profile
        translation.setCurrentIndex(translation.findData("nllb-200-600m"))
        self.app.processEvents()

        with mock.patch.object(
            window.controller,
            "install_plan",
            wraps=window.controller.install_plan,
        ) as install_plan, mock.patch(
            "whospeaks_gui.window.InstallConfirmDialog.exec",
            return_value=QDialog.DialogCode.Rejected,
        ):
            window.request_install()

        self.assertEqual(
            install_plan.call_args.kwargs["translation_model_profile"],
            "nllb-200-600m",
        )
        self.assertTrue(
            any(
                row[1].text().startswith("Local translation runtime") and row[0].isVisible()
                for row in window.overview.setup_plan_rows
            )
        )

    def test_server_first_run_does_not_show_or_install_controller_options(self) -> None:
        window = self.make_window("first_run")
        target = window.overview.setup_target
        target.setCurrentIndex(target.findData("server"))
        self.app.processEvents()

        self.assertFalse(window.overview.setup_live_text_row.isVisible())
        self.assertFalse(window.overview.setup_translation_row.isVisible())
        self.assertFalse(window.overview.setup_speakers_row.isVisible())
        self.assertEqual(window.overview.setup_live_text_value(), "off")
        self.assertEqual(window.overview.setup_translation_profile_value(), "off")

    def test_openai_provider_loads_models_without_silently_changing_semantic_search(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            window = self.make_window("settings")
            provider = window.settings.fields["report_llm_provider"]
            model = window.settings.fields["report_llm_model"]
            self.assertIsInstance(provider, QComboBox)
            self.assertIsInstance(model, QComboBox)
            assert isinstance(provider, QComboBox)
            assert isinstance(model, QComboBox)

            with mock.patch.object(window.settings, "_request_openai_models") as request_models:
                provider.setCurrentIndex(provider.findData("openai"))
            self.app.processEvents()

            request_models.assert_called_once_with()
            self.assertTrue(model.isEditable())
            self.assertEqual(
                window.settings.fields["report_llm_base_url"].text(),
                "https://api.openai.com/v1",
            )
            self.assertEqual(window.settings.text_embedding_preset.currentData(), "off")
            self.assertIn("detected", window.settings.openai_key_status.text())
            self.assertNotIn("test-key", str(window.settings.values()))

            returned = suitable_openai_llm_models(
                [
                    "gpt-account-text-a",
                    "gpt-account-text-b",
                    "gpt-realtime-2",
                    "text-embedding-3-small",
                    "whisper-1",
                ]
            )
            window.settings._apply_openai_models(returned)
            self.assertTrue(model.isEditable())
            self.assertEqual(
                [model.itemData(index) for index in range(model.count())],
                ["gpt-account-text-a", "gpt-account-text-b"],
            )
            self.assertIn("2 suitable models", window.settings.openai_key_status.text())
            model.setCurrentIndex(model.findData("gpt-account-text-b"))
            self.assertEqual(window.settings.values()["report_llm_model"], "gpt-account-text-b")

    def test_openai_model_picker_requires_a_real_catalog_or_manual_model_id(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            window = self.make_window("settings")
            provider = window.settings.fields["report_llm_provider"]
            model = window.settings.fields["report_llm_model"]
            assert isinstance(provider, QComboBox)
            assert isinstance(model, QComboBox)

            provider.setCurrentIndex(provider.findData("openai"))
            self.app.processEvents()

            self.assertTrue(model.isEditable())
            self.assertNotIn("known model defaults", window.settings.openai_key_status.text())
            model.setEditText("gpt-account-specific")
            self.assertEqual(window.settings.values()["report_llm_model"], "gpt-account-specific")

            window.settings._apply_openai_models(["gpt-account-text-a"])
            self.assertEqual(window.settings.values()["report_llm_model"], "gpt-account-specific")

    def test_openai_catalog_loads_off_the_gui_thread(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch(
            "whospeaks_gui.pages._fetch_openai_model_ids",
            return_value=(["gpt-account-text-a", "gpt-account-text-b"], ""),
        ) as fetch_models:
            window = self.make_window("settings")
            provider = window.settings.fields["report_llm_provider"]
            model = window.settings.fields["report_llm_model"]
            assert isinstance(provider, QComboBox)
            assert isinstance(model, QComboBox)

            provider.setCurrentIndex(provider.findData("openai"))
            for _ in range(50):
                self.app.processEvents()
                if model.count() == 2:
                    break
                QTest.qWait(10)

            fetch_models.assert_called_once_with("test-key")
            self.assertEqual(
                [model.itemData(index) for index in range(model.count())],
                ["gpt-account-text-a", "gpt-account-text-b"],
            )
            self.assertIn("2 suitable models", window.settings.openai_key_status.text())

    def test_openai_catalog_waits_until_model_popup_closes_before_rebuilding(self) -> None:
        window = self.make_window("settings")
        window.navigate(2)
        provider = window.settings.fields["report_llm_provider"]
        model = window.settings.fields["report_llm_model"]
        assert isinstance(provider, QComboBox)
        assert isinstance(model, QComboBox)

        with mock.patch.object(window.settings, "_request_openai_models"):
            provider.setCurrentIndex(provider.findData("openai"))
        model.addItem("Current model", "gpt-current")
        model.setCurrentIndex(0)
        model.showPopup()
        self.app.processEvents()
        self.assertTrue(model.view().isVisible())

        window.settings._apply_openai_models(["gpt-new"])
        self.assertEqual(model.itemData(0), "gpt-current")
        self.assertEqual(window.settings._pending_openai_models, ["gpt-new"])

        model.hidePopup()
        window.settings._apply_pending_openai_models()
        self.assertEqual(
            [model.itemData(index) for index in range(model.count())],
            ["gpt-current", "gpt-new"],
        )
        self.assertIsNone(window.settings._pending_openai_models)

    def test_provider_catalogs_have_no_invented_installed_models(self) -> None:
        from window.meeting_server_support import LLM_PROVIDER_OPTIONS

        for provider, option in LLM_PROVIDER_OPTIONS.items():
            with self.subTest(provider=provider):
                self.assertEqual(option["models"], [])

    def test_first_run_rows_do_not_overlap_and_show_installer_choice(self) -> None:
        window = self.make_window("first_run")
        window.resize(1600, 1000)
        self.app.processEvents()
        deployment_detail = window.overview.setup_target_help
        language_heading = next(
            label
            for label in window.overview.findChildren(QLabel)
            if label.text() == "Language" and label.isVisible()
        )
        deployment_bottom = deployment_detail.mapTo(window.overview, deployment_detail.rect().bottomLeft()).y()
        language_top = language_heading.mapTo(window.overview, language_heading.rect().topLeft()).y()

        self.assertLess(deployment_bottom, language_top)
        self.assertIn(window.overview.setup_installer_value(), {"uv", "pip"})

    def test_first_run_form_uses_one_consistent_grid_without_icon_tiles(self) -> None:
        window = self.make_window("first_run")
        window.resize(1600, 1000)
        self.app.processEvents()

        overview = window.overview
        language_top = overview.setup_language_row.mapTo(
            overview,
            overview.setup_language_row.rect().topLeft(),
        ).y()
        installer_top = overview.setup_installer.mapTo(
            overview,
            overview.setup_installer.rect().topLeft(),
        ).y()
        translation_top = overview.setup_translation_row.mapTo(
            overview,
            overview.setup_translation_row.rect().topLeft(),
        ).y()
        speakers_top = overview.setup_speakers_row.mapTo(
            overview,
            overview.setup_speakers_row.rect().topLeft(),
        ).y()

        self.assertLessEqual(abs(language_top - installer_top), 32)
        self.assertEqual(translation_top, speakers_top)
        self.assertFalse(
            any(
                bool(label.property("iconTile"))
                for label in overview.first_run_workspace.findChildren(QLabel)
            )
        )

    def test_first_run_translation_install_defaults_off_for_existing_profile(self) -> None:
        window = self.make_window("first_run")
        window.overview.apply_profile(Profile.from_mapping({
            "translation_enabled": True,
            "translation_provider": "sidecar",
            "translation_model_profile": "nllb-200-600m",
        }))

        self.assertEqual(window.overview.setup_translation_profile_value(), "off")

    def test_first_run_workspace_stacks_in_linux_sized_window(self) -> None:
        window = self.make_window("first_run")
        window.resize(1152, 750)
        self.app.processEvents()

        overview = window.overview
        self.assertEqual(
            overview.first_run_layout.direction(),
            QBoxLayout.Direction.TopToBottom,
        )
        self.assertEqual(
            overview.setup_workspace_divider.frameShape(),
            QFrame.Shape.HLine,
        )
        self.assertEqual(
            overview.workspace_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

    def test_overview_scrollbar_is_available_in_wide_windows_when_content_overflows(self) -> None:
        window = self.make_window("first_run")
        window.resize(1600, 700)
        self.app.processEvents()

        self.assertEqual(
            window.overview.workspace_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        self.assertTrue(window.overview.workspace_scroll.verticalScrollBar().isVisible())

    def test_ready_overview_keeps_installer_choice_visible_and_synchronized(self) -> None:
        window = self.make_window("ready")

        self.assertTrue(window.overview.profile_installer.isVisible())
        window.overview.profile_installer.setCurrentIndex(
            window.overview.profile_installer.findData("pip")
        )
        self.app.processEvents()

        self.assertEqual(window.overview.setup_installer_value(), "pip")
        self.assertEqual(window.overview.setup_installer.currentData(), "pip")

    def test_launch_profile_controls_stay_inside_the_right_panel(self) -> None:
        window = self.make_window("ready")

        # 1320 logical pixels reproduces a maximized 1980px-wide Windows
        # desktop at 150% scaling.  Long CPU-ASR details must yield space to
        # the launch profile instead of pushing its controls out of bounds.
        for width in (1200, 1320, 1440):
            with self.subTest(width=width):
                window.resize(width, 900)
                self.app.processEvents()
                panel_right = window.overview.profile_panel.contentsRect().right()
                if width >= 1280:
                    self.assertGreaterEqual(window.overview.profile_panel.width(), 460)
                installer = window.overview.profile_installer
                self.assertLessEqual(
                    installer.mapTo(
                        window.overview.profile_panel,
                        QPoint(installer.rect().right(), 0),
                    ).x(),
                    panel_right,
                )
                self.assertGreaterEqual(installer.width(), 190)
                for value in window.overview.profile_labels.values():
                    if isinstance(value, QLabel):
                        self.assertTrue(value.wordWrap())
                    else:
                        self.assertIsInstance(value, QAbstractButton)
                    self.assertLessEqual(value.geometry().right(), panel_right)
                    if isinstance(value, QAbstractButton):
                        self.assertTrue(value.toolTip().strip())
                        self.assertIn(value.text(), value.accessibleName())
                    else:
                        self.assertEqual(value.toolTip(), value.text())

    def test_cpu_service_text_stays_inside_each_row_at_windows_scaling_width(self) -> None:
        controller = DemoLauncherController("starting")
        controller.profile = Profile.from_mapping(
            {
                "mode": "cpu",
                "language": "de",
                "asr_backend": "cpu",
                "embeddings_backend": "local",
                "embedding_provider": "speechbrain_ecapa",
                "realtime_preview_engine": "kroko_onnx",
                "realtime_preview_model_preset": "community-64l",
                "cpu_alignment_model": "base",
                "reports_enabled": False,
                "translation_enabled": False,
            }
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.resize(1320, 853)
        window.show()
        self.app.processEvents()
        self.addCleanup(window.bridge.close)
        self.addCleanup(window.hide)

        for key in ("macos_asr", "macos_embeddings", "live"):
            row = window.overview.service_rows[key]
            self.assertTrue(row.isVisible())
            for label in (row.title, row.subtitle, row.endpoint, row.extra, row.status_label):
                if not label.isVisible():
                    continue
                top = label.mapTo(row, QPoint(0, 0)).y()
                bottom = top + label.height()
                self.assertGreaterEqual(top, 0, f"{key}: {label.text()}")
                self.assertLessEqual(bottom, row.height(), f"{key}: {label.text()}")

    def test_overview_action_bar_does_not_duplicate_workspace_border(self) -> None:
        window = self.make_window("ready")

        self.assertTrue(window.overview.action_bar.property("seamless"))
        self.assertFalse(bool(window.settings.action_bar.property("seamless")))

    def test_overview_diagnostics_and_activity_use_one_summary_header_pattern(self) -> None:
        window = self.make_window("ready")
        window.resize(1440, 900)
        operation = mock.Mock()
        operation.name = ""
        operation.status = "success"
        operation.title = "Readiness check completed"
        operation.step = ""
        window.activity.set_operation(operation)
        self.app.processEvents()

        summaries = (
            (0, window.overview.summary),
            (1, window.diagnostics.summary),
            (3, window.activity.summary),
        )
        for page_index, summary in summaries:
            window.navigate(page_index)
            self.app.processEvents()
            self.assertEqual(summary.height(), 68)
            self.assertGreaterEqual(summary.state_label.font().weight(), 600)
            self.assertLessEqual(
                abs(summary.mark.geometry().center().y() - summary.state_label.geometry().center().y()),
                1,
            )

        self.assertEqual(window.activity.operation_state.text(), "SUCCESS")
        self.assertEqual(window.activity.operation_label.text(), "Readiness check completed")
        self.assertEqual(window.activity.operation_label.property("role"), "secondary")

        window.navigate(1)
        self.app.processEvents()
        diagnostics_summary_bottom = window.diagnostics.summary.mapTo(
            window.diagnostics, QPoint(0, window.diagnostics.summary.height())
        ).y()
        diagnostics_table_top = window.diagnostics.table.mapTo(
            window.diagnostics, QPoint(0, 0)
        ).y()
        self.assertEqual(diagnostics_table_top - diagnostics_summary_bottom, 12)

        window.navigate(3)
        self.app.processEvents()
        activity_summary_bottom = window.activity.summary.mapTo(
            window.activity, QPoint(0, window.activity.summary.height())
        ).y()
        activity_log_top = window.activity.log.mapTo(window.activity, QPoint(0, 0)).y()
        self.assertEqual(activity_log_top - activity_summary_bottom, 12)

        window.navigate(0)
        self.app.processEvents()
        overview_summary_bottom = window.overview.summary.mapTo(
            window.overview, QPoint(0, window.overview.summary.height())
        ).y()
        overview_workspace_top = window.overview.workspace_scroll.mapTo(
            window.overview, QPoint(0, 0)
        ).y()
        self.assertEqual(overview_workspace_top - overview_summary_bottom, 12)

    def test_interactive_pages_share_one_footer_action_pattern(self) -> None:
        self.assertEqual(COLORS.button_secondary, "#1E2E3C")
        style = application_style("Segoe UI")
        self.assertGreaterEqual(style.count("background: #1E2E3C;"), 3)
        self.assertIn('QPushButton[inputAction="true"]', style)
        self.assertIn("chevron-up.svg", style)
        window = self.make_window("ready")

        footers = (
            (window.overview.action_bar, window.overview.primary_button),
            (window.diagnostics.action_bar, window.diagnostics.complete_button),
            (window.settings.action_bar, window.settings.save_button),
            (window.activity.action_bar, window.activity.copy_button),
        )
        for footer, primary in footers:
            self.assertEqual(footer.height(), 88)
            self.assertIs(footer.actions.itemAt(0).widget(), primary)
            self.assertTrue(primary.property("footerAction"))
            self.assertTrue(primary.property("primary"))
            self.assertEqual(primary.height(), 48)
            self.assertEqual(primary.iconSize().width(), 22)
            self.assertEqual(primary.iconSize().height(), 22)

        first_action_x = set()
        for page_index, (_footer, primary) in enumerate(footers):
            window.navigate(page_index)
            self.app.processEvents()
            first_action_x.add(primary.mapTo(window, QPoint(0, 0)).x())
        self.assertEqual(len(first_action_x), 1)

        self.assertIs(
            window.diagnostics.action_bar.actions.itemAt(1).widget(),
            window.diagnostics.quick_button,
        )
        self.assertIs(
            window.diagnostics.action_bar.actions.itemAt(2).widget(),
            window.diagnostics.copy_button,
        )
        self.assertIsNone(window.diagnostics.action_bar.actions.itemAt(3).widget())
        for secondary in (
            window.overview.refresh_button,
            window.overview.command_button,
            window.diagnostics.quick_button,
            window.diagnostics.copy_button,
            window.settings.discard_button,
            window.activity.clear_button,
        ):
            self.assertTrue(secondary.property("footerAction"))
            self.assertFalse(bool(secondary.property("primary")))
            self.assertEqual(secondary.height(), 48)

        self.assertFalse(window.settings.save_button.isEnabled())
        host = window.settings.fields["host"]
        assert isinstance(host, QLineEdit)
        host.setText("localhost")
        self.assertTrue(window.settings.save_button.isEnabled())
        window.settings.set_profile(window.settings.profile)
        self.assertFalse(window.settings.save_button.isEnabled())

    def test_install_busy_state_replaces_install_action_on_overview(self) -> None:
        controller = LauncherController(
            Profile(),
            profile_saver=lambda _profile: Path("unused.json"),
        )
        window = LauncherWindow(controller, auto_check=False, reduced_motion=True)
        window.overview.set_operational_state("first_run")

        window.overview.set_busy(True, operation="install")

        self.addCleanup(window.bridge.close)
        self.assertEqual(window.overview.header.title.text(), "Installing components")
        self.assertEqual(window.overview.primary_button.text(), "Installing components...")
        self.assertTrue(window.overview.primary_button.isEnabled() is False)
        self.assertEqual(window.overview.refresh_button.text(), "View activity")
        self.assertEqual(window.overview.command_button.text(), "Cancel installation")

    def test_settings_cover_every_persisted_profile_field(self) -> None:
        window = self.make_window("settings")

        expected = {field.name for field in dataclasses.fields(Profile)}
        derived = {"deployment_target", "asr_backend", "embeddings_backend"}

        self.assertEqual(set(window.settings.fields), expected - derived)
        self.assertEqual(set(window.settings.values()), expected)

    def test_every_button_has_an_explicit_accessible_name(self) -> None:
        window = self.make_window("settings")

        unnamed = [
            button.text()
            for button in window.findChildren(QAbstractButton)
            if button.text().strip() and not button.accessibleName().strip()
        ]

        self.assertEqual(unnamed, [])

    def test_minimum_viewport_collapses_navigation_and_keeps_actions_visible(self) -> None:
        window = self.make_window()
        window.resize(960, 640)
        self.app.processEvents()

        self.assertEqual(window.sidebar.width(), COMPACT_RAIL_WIDTH)
        self.assertTrue(window.overview.primary_button.isVisible())
        self.assertGreaterEqual(window.minimumWidth(), 960)
        self.assertGreaterEqual(window.minimumHeight(), 640)

    def test_repeated_ready_renders_have_identical_pixels(self) -> None:
        first = self.make_window("ready")
        second = self.make_window("ready")
        first.resize(1000, 700)
        second.resize(1000, 700)
        self.app.processEvents()

        first_image = first.grab().toImage()
        second_image = second.grab().toImage()

        self.assertEqual(first_image.size(), second_image.size())
        self.assertEqual(first_image.bits().tobytes(), second_image.bits().tobytes())


if __name__ == "__main__":
    unittest.main()
