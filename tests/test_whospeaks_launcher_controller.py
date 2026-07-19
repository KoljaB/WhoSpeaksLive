from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest import mock

from whospeaks_cli import cli_diagnostics
from whospeaks_cli.cli_diagnostics import (
    CheckResult,
    DoctorReport,
    check_import_group,
    check_text_embedding_provider,
)
from whospeaks_cli.launcher_controller import (
    EventKind,
    LauncherController,
    ProfileValidationError,
)
from whospeaks_cli.profiles import Profile


class FakeProcess:
    next_pid = 7000

    def __init__(self, output: str = "", return_code: int | None = None) -> None:
        self.stdout = io.StringIO(output)
        self.return_code = return_code
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return int(self.return_code or 0)


class LauncherControllerTests(unittest.TestCase):
    def make_controller(self, **kwargs: object) -> LauncherController:
        defaults = {
            "profile": Profile.from_mapping(
                {
                    "mode": "remote",
                    "reports_enabled": True,
                    "report_llm_model": "local",
                    "translation_enabled": True,
                    "translation_provider": "sidecar",
                }
            ),
            "profile_saver": lambda _profile: Path("saved-profile.json"),
            "remote_backend_probe": lambda _url: True,
        }
        defaults.update(kwargs)
        return LauncherController(**defaults)

    def test_profile_update_validates_and_publishes_immutable_snapshot(self) -> None:
        controller = self.make_controller()
        events = []
        controller.subscribe(events.append)

        updated = controller.update_profile({"language": "de", "port": "9123"})

        self.assertEqual(updated.language, "de")
        self.assertEqual(updated.port, 9123)
        self.assertEqual(controller.snapshot.profile.port, 9123)
        self.assertTrue(any(event.kind is EventKind.PROFILE for event in events))
        self.assertTrue(any("Saved launch profile" in line for line in controller.snapshot.logs))

    def test_missing_module_recovery_does_not_reference_removed_setup_controls(self) -> None:
        with mock.patch("whospeaks_cli.cli_diagnostics.module_available", return_value=False):
            result = check_import_group("Runtime", [("fastapi", "fastapi")], required=True)

        self.assertNotIn("Setup tab", result.remediation)
        self.assertNotIn("Install / repair", result.remediation)
        self.assertIn("whospeaks install", result.remediation)

    def test_optional_text_embeddings_are_not_reported_as_a_warning(self) -> None:
        result = check_text_embedding_provider(Profile(), deep=False)

        self.assertEqual(result.status, "skip")
        self.assertIn("Optional semantic search", result.detail)

    def test_profile_update_rejects_invalid_ports_and_target_capacity(self) -> None:
        controller = self.make_controller()

        with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
            controller.update_profile({"port": 70000})
        with self.assertRaisesRegex(ValueError, "between 1 and 16"):
            controller.update_profile({"translation_max_targets": 17})

    def test_diagnostics_emits_progress_report_and_terminal_operation(self) -> None:
        report = DoctorReport(
            "remote",
            [
                CheckResult("Python", "ok", "CPython"),
                CheckResult("Remote ASR", "warn", "Slow", "Retry"),
            ],
        )
        controller = self.make_controller(doctor_runner=lambda *_args, **_kwargs: report)
        events = []
        controller.subscribe(events.append)

        result = controller.run_diagnostics(deep=True)

        self.assertIs(result, report)
        self.assertEqual(controller.snapshot.operation.status, "warning")
        self.assertTrue(any(event.kind is EventKind.REPORT for event in events))
        self.assertIn("1 passed, 1 warnings", controller.snapshot.operation.latest)

    def test_diagnostics_failure_is_not_swallowed(self) -> None:
        controller = self.make_controller(
            doctor_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed"))
        )
        events = []
        controller.subscribe(events.append)

        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            controller.run_diagnostics()

        self.assertEqual(controller.snapshot.operation.status, "error")
        self.assertTrue(any(event.kind is EventKind.ERROR for event in events))

    def test_install_command_captures_selected_runtime_and_translation(self) -> None:
        controller = self.make_controller()
        plan = controller.install_plan(
            "core",
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_model_preset="nemotron-3.5-160ms-int8",
            translation_model_profile="nllb-200-600m",
        )

        command = controller.install_command(plan, installer="pip", model_dir=r"C:\models\nemotron")

        self.assertIn("--target", command)
        self.assertIn("core", command)
        self.assertIn("--realtime-preview-model-dir", command)
        self.assertIn(r"C:\models\nemotron", command)
        self.assertEqual(command[-1], "nllb-200-600m")

    def test_installer_default_prefers_uv_and_falls_back_to_pip(self) -> None:
        controller = self.make_controller()
        with mock.patch(
            "whospeaks_cli.launcher_controller.installer_backend_available",
            return_value=True,
        ):
            self.assertEqual(controller.preferred_installer(), "uv")
        with mock.patch(
            "whospeaks_cli.launcher_controller.installer_backend_available",
            return_value=False,
        ):
            self.assertEqual(controller.preferred_installer(), "pip")

    def test_remote_diagnostics_do_not_require_librosa(self) -> None:
        captured_groups: dict[str, list[tuple[str, str]]] = {}

        def imports(
            name: str,
            modules: list[tuple[str, str]],
            *,
            required: bool,
        ) -> CheckResult:
            captured_groups[name] = modules
            return CheckResult(name, "ok" if required else "skip", "captured")

        profile = Profile.from_mapping(
            {
                "mode": "remote",
                "reports_enabled": False,
                "realtime_preview_engine": "off",
            }
        )
        okay = CheckResult("probe", "ok", "ready")
        with (
            mock.patch.object(cli_diagnostics, "command_version", return_value=(True, "ffmpeg")),
            mock.patch.object(cli_diagnostics, "check_import_group", side_effect=imports),
            mock.patch.object(cli_diagnostics, "check_faster_whisper_cache", return_value=okay),
            mock.patch.object(cli_diagnostics, "check_embedding_cache", return_value=okay),
            mock.patch.object(cli_diagnostics, "check_remote_health", return_value=okay),
            mock.patch.object(cli_diagnostics, "check_remote_providers", return_value=okay),
            mock.patch.object(cli_diagnostics, "check_port", return_value=okay),
        ):
            cli_diagnostics.run_doctor(profile)

        controller_modules = {module for module, _package in captured_groups["Controller Python modules"]}
        self.assertNotIn("librosa", controller_modules)

    def test_install_streams_progress_and_finishes_cleanly(self) -> None:
        process = FakeProcess(
            "Collecting whospeaks\nDownloading sherpa model archive\nSaved configuration\n",
            return_code=0,
        )
        controller = self.make_controller(popen_factory=lambda *_args, **_kwargs: process)

        return_code = controller.install(["python", "-m", "installer"], title="Remote controller")

        self.assertEqual(return_code, 0)
        self.assertEqual(controller.snapshot.operation.status, "success")
        self.assertIsNone(controller.install_process)
        self.assertTrue(any("Downloading sherpa" in line for line in controller.snapshot.logs))

    def test_cancel_terminates_the_owned_installer_tree(self) -> None:
        controller = self.make_controller()
        process = FakeProcess(return_code=None)
        controller.install_process = process
        controller.coordinator.start_operation("install", "Install", "Running")

        with mock.patch("whospeaks_cli.launcher_controller.terminate_service_processes") as terminate:
            cancelled = controller.cancel_operation()

        self.assertTrue(cancelled)
        terminate.assert_called_once_with([process])
        self.assertTrue(controller.snapshot.operation.cancel_requested)

    def test_launch_rejects_an_external_optional_service_port(self) -> None:
        controller = self.make_controller()
        with mock.patch.object(controller, "_port_accepting", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "already owned by another process"):
                controller.launch()

        self.assertEqual(controller.servers.state("reports").ownership, "external")
        self.assertEqual(controller.snapshot.operation.status, "error")

    def test_launch_and_shutdown_track_only_owned_processes(self) -> None:
        processes: list[FakeProcess] = []

        def popen(*_args: object, **_kwargs: object) -> FakeProcess:
            process = FakeProcess(return_code=None)
            processes.append(process)
            return process

        controller = self.make_controller(popen_factory=popen)
        with mock.patch.object(controller, "_port_accepting", return_value=False):
            controller.launch()

        self.assertEqual(len(processes), 3)
        self.assertTrue(all(controller.servers.state(kind).ownership == "app" for kind in ("live", "reports", "translation")))
        with mock.patch("whospeaks_cli.launcher_controller.terminate_service_processes") as terminate:
            controller.stop_owned_services()

        terminate.assert_called_once()
        self.assertEqual(set(terminate.call_args.args[0]), set(processes))
        self.assertTrue(
            all(
                controller.servers.state(kind).status == "stopped"
                for kind in ("live", "reports", "translation")
            )
        )

    def test_optional_service_failure_keeps_live_launch_available(self) -> None:
        calls = 0

        def popen(*_args: object, **_kwargs: object) -> FakeProcess:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("translation executable missing")
            return FakeProcess(return_code=None)

        controller = self.make_controller(popen_factory=popen)
        events = []
        controller.subscribe(events.append)
        with mock.patch.object(controller, "_port_accepting", return_value=False):
            controller.launch()

        self.assertEqual(controller.servers.state("live").ownership, "app")
        self.assertEqual(controller.servers.state("translation").status, "failed")
        self.assertEqual(controller.snapshot.operation.name, "launch")
        self.assertEqual(controller.snapshot.operation.status, "running")
        self.assertTrue(any(event.kind is EventKind.ERROR for event in events))

    def test_failed_optional_service_can_retry_independently(self) -> None:
        process = FakeProcess(return_code=None)
        controller = self.make_controller(popen_factory=lambda *_args, **_kwargs: process)
        controller.servers.fail_start("translation")

        with mock.patch.object(controller, "_port_accepting", return_value=False):
            result = controller.retry_service("translation")

        self.assertIs(result, process)
        self.assertEqual(controller.servers.state("translation").status, "starting")
        self.assertEqual(controller.snapshot.operation.status, "success")

    def test_failed_live_window_can_retry_without_restarting_healthy_reports(self) -> None:
        process = FakeProcess(return_code=None)
        controller = self.make_controller(popen_factory=lambda *_args, **_kwargs: process)
        controller.servers.fail_start("live")

        with mock.patch.object(controller, "_port_accepting", return_value=False):
            result = controller.retry_service("live")

        self.assertIs(result, process)
        self.assertEqual(controller.servers.state("live").status, "starting")
        self.assertEqual(controller.snapshot.operation.status, "success")

    def test_launch_cancellation_cleans_process_started_during_race(self) -> None:
        processes: list[FakeProcess] = []
        controller: LauncherController

        def popen(*_args: object, **_kwargs: object) -> FakeProcess:
            process = FakeProcess(return_code=None)
            processes.append(process)
            if len(processes) == 1:
                controller.cancel_operation()
            return process

        controller = self.make_controller(popen_factory=popen)
        with (
            mock.patch.object(controller, "_port_accepting", return_value=False),
            mock.patch("whospeaks_cli.launcher_controller.terminate_service_processes") as terminate,
        ):
            controller.launch()

        self.assertEqual(len(processes), 1)
        terminate.assert_called_with(processes)
        self.assertEqual(controller.snapshot.operation.status, "warning")
        self.assertEqual(controller.servers.state("reports").status, "stopped")

    def test_live_service_requires_http_health_after_port_opens(self) -> None:
        controller = self.make_controller()
        process = FakeProcess(return_code=None)
        controller.servers.begin("live", process)

        with (
            mock.patch.object(controller, "_port_accepting", return_value=True),
            mock.patch.object(controller, "_http_service_ready", return_value=False),
        ):
            controller.refresh_services(force=True)

        self.assertEqual(controller.servers.state("live").status, "starting")

        with (
            mock.patch.object(controller, "_port_accepting", return_value=True),
            mock.patch.object(controller, "_http_service_ready", return_value=True),
        ):
            controller.refresh_services(force=True)

        self.assertEqual(controller.servers.state("live").status, "running")

    def test_launch_stays_active_until_all_health_checks_pass(self) -> None:
        processes: list[FakeProcess] = []

        def popen(*_args: object, **_kwargs: object) -> FakeProcess:
            process = FakeProcess(return_code=None)
            processes.append(process)
            return process

        controller = self.make_controller(popen_factory=popen)
        with mock.patch.object(controller, "_port_accepting", return_value=False):
            controller.launch()

        self.assertEqual(controller.snapshot.operation.name, "launch")
        self.assertTrue(all(controller.servers.state(kind).status == "starting" for kind in ("live", "reports", "translation")))

        with (
            mock.patch.object(controller, "_service_ready", return_value=True),
            mock.patch.object(controller, "_remote_backend_available", return_value=True),
        ):
            controller.refresh_services(force=True)

        self.assertEqual(controller.snapshot.operation.name, "")
        self.assertEqual(controller.snapshot.operation.status, "success")

    def test_cancel_remains_available_while_services_are_warming_up(self) -> None:
        controller = self.make_controller(popen_factory=lambda *_args, **_kwargs: FakeProcess(return_code=None))
        with mock.patch.object(controller, "_port_accepting", return_value=False):
            controller.launch()

        with mock.patch("whospeaks_cli.launcher_controller.terminate_service_processes") as terminate:
            cancelled = controller.cancel_operation()

        self.assertTrue(cancelled)
        terminate.assert_called_once()
        self.assertEqual(controller.snapshot.operation.name, "")
        self.assertEqual(controller.snapshot.operation.status, "warning")

    def test_remote_backend_health_is_a_first_class_service_state(self) -> None:
        controller = self.make_controller()

        with (
            mock.patch.object(controller, "_port_accepting", return_value=False),
            mock.patch.object(
                controller,
                "_remote_backend_probe",
                side_effect=lambda url: url == controller.profile.remote_asr_url,
            ),
        ):
            controller.refresh_services(force=True)

        asr = controller.servers.state("macos_asr")
        embeddings = controller.servers.state("macos_embeddings")
        self.assertEqual((asr.status, asr.ownership), ("running", "external"))
        self.assertEqual((embeddings.status, embeddings.ownership), ("unavailable", "external"))

    def test_local_asr_and_embeddings_mirror_the_live_process_lifecycle(self) -> None:
        profile = Profile.from_mapping(
            {
                "mode": "local",
                "reports_enabled": False,
                "translation_enabled": False,
            }
        )
        controller = self.make_controller(profile=profile)
        controller.servers.begin("live", FakeProcess(return_code=None))

        with mock.patch.object(controller, "_service_ready", return_value=False):
            controller.refresh_services(force=True)

        self.assertEqual(controller.servers.state("macos_asr").status, "starting")
        self.assertEqual(controller.servers.state("macos_embeddings").status, "starting")
        self.assertIn(profile.model, controller.service_address("macos_asr"))
        self.assertIn("preset", controller.service_address("macos_embeddings").lower())

        with mock.patch.object(controller, "_service_ready", return_value=True):
            controller.refresh_services(force=True)

        self.assertEqual(controller.servers.state("macos_asr").status, "running")
        self.assertEqual(controller.servers.state("macos_embeddings").status, "running")

        with mock.patch("whospeaks_cli.launcher_controller.terminate_service_processes"):
            controller.stop_owned_services()

        self.assertEqual(controller.servers.state("macos_asr").status, "stopped")
        self.assertEqual(controller.servers.state("macos_embeddings").status, "stopped")

    def test_core_launch_fails_before_spawning_when_required_backends_are_offline(self) -> None:
        popen = mock.Mock()
        controller = self.make_controller(
            popen_factory=popen,
            remote_backend_probe=lambda _url: False,
        )

        with self.assertRaisesRegex(RuntimeError, "Final ASR, speaker embeddings"):
            controller.launch()

        popen.assert_not_called()
        self.assertEqual(controller.snapshot.operation.status, "error")
        self.assertEqual(controller.servers.state("macos_asr").status, "unavailable")
        self.assertEqual(controller.servers.state("macos_embeddings").status, "unavailable")

    def test_validation_errors_identify_the_exact_editable_field(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as caught:
            controller.validate_profile_updates({"host": ""})

        self.assertEqual(caught.exception.field, "host")
        self.assertIn("cannot be empty", str(caught.exception))

    def test_remote_deployment_requires_both_service_urls(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as caught:
            controller.validate_profile_updates(
                {
                    "mode": "remote",
                    "asr_backend": "remote",
                    "embeddings_backend": "remote",
                    "remote_asr_url": "",
                    "remote_embeddings_url": "http://127.0.0.1:8660",
                }
            )

        self.assertEqual(caught.exception.field, "remote_asr_url")

    def test_active_local_service_ports_must_be_distinct(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as caught:
            controller.validate_profile_updates(
                {
                    "reports_enabled": True,
                    "port": 8796,
                    "reports_port": 8796,
                }
            )

        self.assertEqual(caught.exception.field, "reports_port")

    def test_meeting_intelligence_requires_an_explicit_provider_model(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as caught:
            controller.validate_profile_updates(
                {
                    "reports_enabled": True,
                    "report_llm_provider": "openai",
                    "report_llm_base_url": "https://api.openai.com/v1",
                    "report_llm_model": "",
                }
            )

        self.assertEqual(caught.exception.field, "report_llm_model")

    def test_launch_revalidates_a_saved_profile_before_spawning(self) -> None:
        controller = self.make_controller(
            profile=Profile.from_mapping(
                {
                    "reports_enabled": True,
                    "report_llm_provider": "openai",
                    "report_llm_model": "",
                }
            )
        )

        with self.assertRaises(ProfileValidationError) as caught:
            controller.launch()

        self.assertEqual(caught.exception.field, "report_llm_model")

    def test_test_only_translation_provider_cannot_be_saved_by_launcher(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as caught:
            controller.validate_profile_updates({"translation_provider": "mock"})

        self.assertEqual(caught.exception.field, "translation_provider")

    def test_openai_compatible_translation_requires_a_real_endpoint_and_model(self) -> None:
        controller = self.make_controller()

        with self.assertRaises(ProfileValidationError) as missing_endpoint:
            controller.validate_profile_updates({
                "translation_enabled": True,
                "translation_provider": "openai_compatible",
                "translation_base_url": "",
                "translation_model": "",
            })
        self.assertEqual(missing_endpoint.exception.field, "translation_base_url")

        with self.assertRaises(ProfileValidationError) as missing_model:
            controller.validate_profile_updates({
                "translation_enabled": True,
                "translation_provider": "openai_compatible",
                "translation_base_url": "http://translator.example/v1",
                "translation_model": "",
            })
        self.assertEqual(missing_model.exception.field, "translation_model")

    def test_configure_for_install_persists_selected_translation_runtime(self) -> None:
        controller = self.make_controller()
        plan = controller.install_plan(
            "core",
            realtime_preview_engine="off",
            translation_model_profile="nllb-200-600m",
        )

        configured = controller.configure_for_install(
            plan,
            language="de",
            live_speaker_assignment=False,
            persist=False,
        )

        self.assertTrue(configured.translation_enabled)
        self.assertEqual(configured.translation_provider, "sidecar")
        self.assertEqual(configured.translation_model_profile, "nllb-200-600m")
        self.assertEqual(configured.language, "de")

    def test_server_profile_cannot_accidentally_launch_browser_controller(self) -> None:
        controller = self.make_controller(profile=Profile.from_mapping({"mode": "server"}))

        with self.assertRaisesRegex(RuntimeError, "does not launch the browser controller"):
            controller.launch()


if __name__ == "__main__":
    unittest.main()
