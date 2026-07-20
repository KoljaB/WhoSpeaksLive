from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
import zipfile
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whospeaks_cli import main as cli
from whospeaks_cli import profiles as profiles_module
from whospeaks_cli.profiles import provider_preset_label


class WhoSpeaksCliTests(unittest.TestCase):
    def test_profile_save_failure_is_reported_without_writing_a_second_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "global" / "config.json"
            fallback = Path(directory) / ".whospeaks" / "config.json"
            with (
                mock.patch.object(profiles_module, "config_path", return_value=primary),
                mock.patch.object(profiles_module, "local_config_path", return_value=fallback),
                mock.patch.object(Path, "write_text", side_effect=PermissionError("read only")) as write,
            ):
                with self.assertRaisesRegex(PermissionError, "read only"):
                    profiles_module.save_profile(profiles_module.Profile())

        self.assertEqual(write.call_count, 1)
        self.assertFalse(fallback.exists())

    def test_speechbrain_encoder_initialization_supports_current_and_older_pretrained_locations(self) -> None:
        module_path = (
            ROOT
            / "vendor"
            / "remote_servers"
            / "voice-embeddings-server"
            / "speechbrain_compat.py"
        )
        spec = importlib.util.spec_from_file_location("test_speechbrain_compat", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        compatibility = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compatibility)

        class EncoderClassifier:
            from_hparams = mock.Mock(return_value=object())

        class CurrentPretrained:
            pass

        speechbrain = types.ModuleType("speechbrain")
        inference = types.ModuleType("speechbrain.inference")
        inference.__path__ = []
        speaker = types.ModuleType("speechbrain.inference.speaker")
        speaker.EncoderClassifier = EncoderClassifier
        interfaces = types.ModuleType("speechbrain.inference.interfaces")
        interfaces.Pretrained = CurrentPretrained
        modules = {
            "speechbrain": speechbrain,
            "speechbrain.inference": inference,
            "speechbrain.inference.speaker": speaker,
            "speechbrain.inference.interfaces": interfaces,
        }
        with mock.patch.dict(sys.modules, modules):
            model = compatibility.load_speechbrain_encoder("model-id", "/cache", "mps")

        self.assertIs(model, EncoderClassifier.from_hparams.return_value)
        EncoderClassifier.from_hparams.assert_called_once_with(
            source="model-id",
            savedir="/cache",
            run_opts={"device": "mps"},
        )
        self.assertEqual(CurrentPretrained.device_type, "cpu")

        class OlderPretrained:
            pass

        speaker.Pretrained = OlderPretrained
        EncoderClassifier.from_hparams.reset_mock()
        older_modules = dict(modules)
        older_modules.pop("speechbrain.inference.interfaces")
        with mock.patch.dict(sys.modules, older_modules, clear=False):
            sys.modules.pop("speechbrain.inference.interfaces", None)
            compatibility.load_speechbrain_encoder("older-id", "/older-cache", "cpu")

        EncoderClassifier.from_hparams.assert_called_once()
        self.assertEqual(OlderPretrained.device_type, "cpu")

    def test_macos_runtime_root_is_stable_across_working_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            with mock.patch.dict(os.environ, {"WHOSPEAKS_MACOS_RUNTIME_ROOT": str(Path(directory) / "runtime")}):
                with mock.patch.object(Path, "cwd", return_value=first):
                    first_root = cli.default_macos_runtime_root()
                with mock.patch.object(Path, "cwd", return_value=second):
                    second_root = cli.default_macos_runtime_root()

        self.assertEqual(first_root, second_root)

    def test_macos_install_plan_uses_current_editable_source_and_packaged_service_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "checkout"
            source.mkdir()
            with mock.patch.object(cli, "installed_package_source", return_value=(source, True)):
                commands = cli.build_macos_install_commands(Path(directory) / "runtime")

        rendered = [" ".join(command) for command in commands]
        self.assertIn(f"-e {source}[controller]", rendered[0])
        service_installs = [item for item in rendered if "--no-deps" in item]
        self.assertEqual(len(service_installs), 2)
        self.assertTrue(all(f"-e {source}" in item for item in service_installs))
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        self.assertTrue(all("remote_servers.launcher" in spec.command for spec in cli.build_macos_service_specs(profile)))

    def test_wheel_contains_managed_service_scripts_and_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            source = temporary_root / "source"
            wheel_directory = temporary_root / "wheel"
            source.mkdir()
            wheel_directory.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
                shutil.copy2(ROOT / name, source / name)
            clean_copy = shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", "*.pyo")
            shutil.copytree(ROOT / "src", source / "src", ignore=clean_copy)
            shutil.copytree(ROOT / "vendor", source / "vendor", ignore=clean_copy)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "-w",
                    str(wheel_directory),
                    ".",
                ],
                cwd=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            wheel = next(wheel_directory.glob("*.whl"))
            installed = temporary_root / "installed"
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                archive.extractall(installed)

            smoke = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        "import json,sys; "
                        f"sys.path.insert(0, {str(installed)!r}); "
                        "from whospeaks_cli.cli_classic import build_server_launch_lines; "
                        "print(json.dumps(build_server_launch_lines()))"
                    ),
                ],
                cwd=temporary_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(smoke.returncode, 0, smoke.stderr)
            launch_lines = json.loads(smoke.stdout)

        self.assertIn("remote_servers/launcher.py", names)
        self.assertIn("remote_servers/faster-whisper-asr/asr_server.py", names)
        self.assertIn("remote_servers/faster-whisper-asr/mlx_asr_server.py", names)
        self.assertIn("remote_servers/faster-whisper-asr/requirements.txt", names)
        self.assertIn("remote_servers/voice-embeddings-server/embeddings_server.py", names)
        self.assertIn("remote_servers/voice-embeddings-server/requirements-macos.txt", names)
        self.assertIn("remote_servers/voice-embeddings-server/requirements.txt", names)
        self.assertNotIn("remote_servers/faster-whisper-asr/test_parent_watchdog.py", names)
        self.assertNotIn("remote_servers/voice-embeddings-server/tools/benchmark_voice_embeddings.py", names)
        self.assertEqual(len(launch_lines), 2)
        self.assertIn(str(installed / "remote_servers" / "faster-whisper-asr"), launch_lines[0])
        self.assertIn(str(installed / "remote_servers" / "voice-embeddings-server"), launch_lines[1])

    def test_macos_install_profile_keeps_remote_backends_and_managed_marker(self) -> None:
        with (
            mock.patch("whospeaks_cli.planning.platform.system", return_value="Darwin"),
            mock.patch("whospeaks_cli.planning.platform.machine", return_value="arm64"),
        ):
            plan = cli.install_plan_for_target("macos")
        profile = cli.profile_for_install(cli.Profile(), plan)

        self.assertEqual(plan.mode, "remote")
        self.assertEqual(plan.realtime_preview_engine, "off")
        self.assertEqual(profile.deployment_target, "macos")
        self.assertEqual(profile.mode, "remote")
        self.assertEqual(profile.asr_backend, "remote")
        self.assertEqual(profile.embeddings_backend, "remote")
        self.assertEqual(profile.remote_asr_url, "http://127.0.0.1:8651")
        self.assertEqual(profile.remote_embeddings_url, "http://127.0.0.1:8660")
        self.assertEqual(profile.embedding_provider, "speechbrain_ecapa")
        self.assertEqual(profile.realtime_preview_engine, "off")
        self.assertEqual(profile.realtime_preview_model_preset, "")

    def test_macos_target_rejects_unsupported_platform(self) -> None:
        with (
            mock.patch("whospeaks_cli.planning.platform.system", return_value="Darwin"),
            mock.patch("whospeaks_cli.planning.platform.machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(SystemExit, "Apple Silicon"):
                cli.install_plan_for_target("macos")

    def test_macos_install_commands_create_isolated_service_venvs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".whospeaks" / "macos"
            commands = cli.build_macos_install_commands(root)

        rendered = [" ".join(command) for command in commands]
        self.assertIn("[controller]", rendered[0])
        self.assertTrue(any(f"-m venv {root / 'mlx-asr'}" in item for item in rendered))
        self.assertTrue(any("mlx-whisper" in item for item in rendered))
        self.assertTrue(any(f"-m venv {root / 'embeddings'}" in item for item in rendered))
        requirements = next(item for item in rendered if "requirements-macos.txt" in item)
        self.assertNotIn("pyannote", requirements)

    def test_macos_launch_plan_has_immutable_http_service_specs(self) -> None:
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        plan = cli.build_launch_plan(profile)

        self.assertEqual([spec.name for spec in plan.services], ["MLX ASR", "MPS embeddings"])
        asr, embeddings = plan.services
        self.assertIsInstance(asr.command, tuple)
        self.assertEqual(asr.health_url, "http://127.0.0.1:8651/health")
        self.assertIn("ASR_PORT", dict(asr.env))
        self.assertEqual(dict(embeddings.env)["EMBEDDINGS_DEVICE"], "auto")
        self.assertEqual(dict(embeddings.env)["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
        self.assertEqual(dict(asr.expected_health), {"service": "mlx-whisper-asr"})
        self.assertEqual(dict(embeddings.expected_health), {"service": "voice-embeddings-server"})

    def test_macos_profile_normalizes_custom_service_urls_to_fixed_loopback(self) -> None:
        profile = cli.Profile.from_mapping({
            "deployment_target": "macos",
            "mode": "remote",
            "remote_asr_url": "http://example.test:9999",
            "remote_embeddings_url": "http://0.0.0.0:1234",
        })

        self.assertEqual(profile.remote_asr_url, "http://127.0.0.1:8651")
        self.assertEqual(profile.remote_embeddings_url, "http://127.0.0.1:8660")
        specs = cli.build_macos_service_specs(profile)
        self.assertEqual(specs[0].health_url, "http://127.0.0.1:8651/health")
        self.assertEqual(dict(specs[0].env)["ASR_PORT"], "8651")

    def test_macos_profile_mapping_forces_remote_topology(self) -> None:
        profile = cli.Profile.from_mapping({
            "deployment_target": "macos",
            "mode": "local",
            "asr_backend": "local",
            "embeddings_backend": "local",
            "remote_asr_url": "http://example.test:9999",
            "remote_embeddings_url": "http://example.test:9998",
        })

        self.assertEqual(profile.mode, "remote")
        self.assertEqual(profile.asr_backend, "remote")
        self.assertEqual(profile.embeddings_backend, "remote")
        self.assertEqual(profile.remote_asr_url, "http://127.0.0.1:8651")
        self.assertEqual(profile.remote_embeddings_url, "http://127.0.0.1:8660")

    def test_service_health_uses_spec_identity_not_url_port(self) -> None:
        spec = cli.ServiceProcessSpec(
            name="test",
            command=("python",),
            cwd=".",
            env=(),
            health_url="http://127.0.0.1:9999/health",
            readiness_timeout=1,
            expected_health=(("service", "expected"),),
        )
        with mock.patch("whospeaks_cli.service_processes.read_json_url", return_value=(True, "ok", {"ok": True, "service": "other"})):
            self.assertFalse(cli.service_health_ready(spec))
        with mock.patch("whospeaks_cli.service_processes.read_json_url", return_value=(True, "ok", {"ok": True, "service": "expected"})):
            self.assertTrue(cli.service_health_ready(spec))

    def test_switching_mode_clears_managed_macos_marker(self) -> None:
        managed = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})

        for mode in ("local", "remote", "server"):
            with self.subTest(mode=mode):
                self.assertEqual(cli.profile_for_mode(managed, mode).deployment_target, "")

    def test_macos_cli_launch_waits_in_order_and_cleans_owned_services(self) -> None:
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        events: list[str] = []

        class FakeProcess:
            pid = 42

            def poll(self) -> None:
                return None

        def fake_start(spec: object) -> FakeProcess:
            events.append(f"start:{spec.name}")
            return FakeProcess()

        def fake_wait(spec: object, _process: object = None) -> None:
            events.append(f"ready:{spec.name}")

        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch.object(cli, "require_apple_silicon_macos"),
            mock.patch(
                "whospeaks_cli.cli_commands.check_port",
                return_value=cli.CheckResult("Browser UI port", "ok", "available"),
            ),
            mock.patch.object(cli, "service_health_ready", return_value=False),
            mock.patch.object(cli, "start_service_process", side_effect=fake_start),
            mock.patch.object(cli, "wait_for_service_health", side_effect=fake_wait),
            mock.patch.object(cli, "terminate_service_processes") as terminate,
            mock.patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
        ):
            code = cli.main(["launch"])

        self.assertEqual(code, 0)
        self.assertEqual(events, ["start:MLX ASR", "ready:MLX ASR", "start:MPS embeddings", "ready:MPS embeddings"])
        run.assert_called_once()
        self.assertEqual(len(terminate.call_args.args[0]), 2)

    def test_macos_cli_launch_preserves_external_service_and_cleans_after_health_failure(self) -> None:
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        process = mock.Mock(pid=42)
        process.poll.return_value = None
        health_results = iter((True, False))
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch.object(cli, "require_apple_silicon_macos"),
            mock.patch(
                "whospeaks_cli.cli_commands.check_port",
                return_value=cli.CheckResult("Browser UI port", "ok", "available"),
            ),
            mock.patch.object(cli, "service_health_ready", side_effect=lambda _url: next(health_results)),
            mock.patch.object(cli, "start_service_process", return_value=process) as start,
            mock.patch.object(cli, "wait_for_service_health", side_effect=RuntimeError("unhealthy")),
            mock.patch.object(cli, "terminate_service_processes") as terminate,
            mock.patch.object(cli.subprocess, "run") as run,
        ):
            code = cli.main(["launch"])

        self.assertEqual(code, 1)
        self.assertEqual(start.call_count, 1)
        terminate.assert_called_once_with([process])
        run.assert_not_called()

    def test_macos_cli_checks_reports_port_before_starting_managed_services(self) -> None:
        profile = cli.Profile.from_mapping(
            {
                "deployment_target": "macos",
                "mode": "remote",
                "reports_enabled": True,
                "report_llm_model": "test-report-model",
            }
        )
        browser_ok = cli.CheckResult("Browser UI port", "ok", "available")
        port_failure = cli.CheckResult("Port 8898", "fail", "already in use")
        with (
            mock.patch.dict(
                os.environ,
                {"WHOSPEAKS_MACOS_RUNTIME_ROOT": tempfile.gettempdir()},
            ),
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch.object(cli, "require_apple_silicon_macos"),
            mock.patch("whospeaks_cli.cli_commands.check_port", side_effect=(browser_ok, port_failure)),
            mock.patch.object(cli, "start_service_process") as start_service,
            mock.patch.object(cli.subprocess, "Popen") as popen,
            mock.patch.object(cli.subprocess, "run") as run,
        ):
            code = cli.main(["launch"])

        self.assertEqual(code, 2)
        start_service.assert_not_called()
        popen.assert_not_called()
        run.assert_not_called()

    def test_cli_checks_browser_port_before_starting_any_process(self) -> None:
        profile = cli.Profile.from_mapping({"mode": "remote"})
        port_failure = cli.CheckResult("Browser UI port", "fail", "already in use")
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch("whospeaks_cli.cli_commands.check_port", return_value=port_failure),
            mock.patch.object(cli.subprocess, "Popen") as popen,
            mock.patch.object(cli.subprocess, "run") as run,
        ):
            code = cli.main(["launch"])

        self.assertEqual(code, 2)
        popen.assert_not_called()
        run.assert_not_called()

    def test_macos_launch_rejects_unsupported_saved_platform(self) -> None:
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch("whospeaks_cli.planning.platform.system", return_value="Darwin"),
            mock.patch("whospeaks_cli.planning.platform.machine", return_value="x86_64"),
        ):
            with self.assertRaisesRegex(SystemExit, "Apple Silicon"):
                cli.main(["launch"])

    def test_owned_process_cleanup_waits_then_escalates(self) -> None:
        process = mock.Mock(pid=42)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["service"], 5), 0]
        with (
            mock.patch("whospeaks_cli.service_processes._is_windows", return_value=False),
            mock.patch("whospeaks_cli.service_processes.os.getpgid", return_value=99, create=True),
            mock.patch("whospeaks_cli.service_processes.os.killpg", create=True) as killpg,
            mock.patch("whospeaks_cli.service_processes.signal.SIGKILL", 9, create=True),
        ):
            cli.terminate_service_processes([process])

        self.assertEqual(killpg.call_args_list, [mock.call(99, signal.SIGTERM), mock.call(99, 9)])
        self.assertEqual(process.wait.call_count, 2)

    def test_posix_cleanup_ignores_exit_between_poll_and_getpgid(self) -> None:
        process = mock.Mock(pid=42)
        process.poll.return_value = None
        process.wait.return_value = 0
        with (
            mock.patch("whospeaks_cli.service_processes._is_windows", return_value=False),
            mock.patch(
                "whospeaks_cli.service_processes.os.getpgid",
                side_effect=ProcessLookupError,
                create=True,
            ),
        ):
            cli.terminate_service_processes([process])

        process.wait.assert_called_once_with(timeout=5)

    def test_windows_cleanup_targets_descendant_tree_before_root_wait(self) -> None:
        process = mock.Mock(pid=42)
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            mock.patch("whospeaks_cli.service_processes._is_windows", return_value=True),
            mock.patch("whospeaks_cli.service_processes.subprocess.run") as run,
        ):
            cli.terminate_service_processes([process])

        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "42", "/T"])
        process.wait.assert_called_once_with(timeout=5)

    def test_macos_doctor_distinguishes_installed_stopped_services(self) -> None:
        profile = cli.Profile.from_mapping({"deployment_target": "macos", "mode": "remote"})
        installed = cli.CheckResult("runtime", "ok", "installed")
        with (
            mock.patch("whospeaks_cli.cli_diagnostics.platform.system", return_value="Darwin"),
            mock.patch("whospeaks_cli.cli_diagnostics.platform.machine", return_value="arm64"),
            mock.patch("whospeaks_cli.cli_diagnostics.check_macos_service_runtime", return_value=installed),
            mock.patch("whospeaks_cli.cli_diagnostics.read_json_url", return_value=(False, "Connection failed: Connection refused", None)),
            mock.patch.object(cli, "command_version", return_value=(True, "ffmpeg")),
            mock.patch.object(cli, "check_import_group", return_value=cli.CheckResult("imports", "ok", "ok")),
        ):
            report = cli.run_doctor(profile)

        asr = next(check for check in report.checks if check.name == "Managed MLX ASR health")
        embeddings = next(check for check in report.checks if check.name == "Managed embeddings health")
        self.assertEqual(asr.status, "warn")
        self.assertEqual(embeddings.status, "warn")
        self.assertIn("installed but stopped", asr.detail)
        self.assertFalse(any(check.name == "CUDA visibility" and check.status == "fail" for check in report.checks))

    def test_profile_and_install_planners_are_copy_on_write(self) -> None:
        profile = cli.Profile(model="large-v2", translation_enabled=False)
        updated = profile.with_updates(model="small")
        plan = cli.install_plan_for_target(
            "local",
            realtime_preview_engine="off",
            translation_model_profile="nllb-200-600m",
        )
        install_profile = cli.profile_for_install(updated, plan)

        self.assertEqual(profile.model, "large-v2")
        self.assertFalse(profile.translation_enabled)
        self.assertEqual(updated.model, "small")
        self.assertTrue(install_profile.translation_enabled)
        self.assertEqual(install_profile.translation_model_profile, "nllb-200-600m")

    def test_launch_plan_captures_detached_commands(self) -> None:
        profile = cli.Profile(
            reports_enabled=True,
            report_llm_model="test-report-model",
            translation_enabled=True,
            translation_provider="sidecar",
        )
        plan = cli.build_launch_plan(profile)

        self.assertIsInstance(plan.live, tuple)
        self.assertIsInstance(plan.reports, tuple)
        self.assertIsInstance(plan.translation, tuple)
        self.assertIn("--translation-provider", plan.live)

    def test_speaker_model_preset_documentation_matches_launcher_definitions(self) -> None:
        documentation = (ROOT / "docs" / "speaker-model-presets.md").read_text(encoding="utf-8")
        for preset_id, preset in cli.PROVIDER_PRESETS.items():
            row_prefix = (
                f"| **{provider_preset_label(preset_id, preset)}** | `{preset_id}` | "
                f"`{preset.embedding_provider}` | `{preset.live_speaker_embedding_provider}` |"
            )
            self.assertIn(row_prefix, documentation)

    def test_recommended_preview_engine_prefers_nemotron_then_kroko_then_off(self) -> None:
        self.assertEqual(cli.recommended_preview_engine("en"), "sherpa_onnx")
        self.assertEqual(cli.recommended_preview_engine("he"), "kroko_onnx")
        self.assertEqual(cli.recommended_preview_engine("cy"), "off")

    def test_install_command_uses_local_preview_extra(self) -> None:
        with mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1"):
            command = cli.build_install_command(installer_backend="pip")

        self.assertEqual(command[:3], [sys.executable, "-m", "pip"])
        self.assertEqual(command[-2:], ["install", "whospeaks[complete,preview]==0.0.1"])
        self.assertIn("whospeaks[complete,preview]==0.0.1", cli.format_command(command))

    def test_uv_install_command_targets_the_requested_python(self) -> None:
        target_python = Path("runtime") / "Scripts" / "python.exe"
        with (
            mock.patch.object(cli.shutil, "which", return_value=r"C:\tools\uv.exe"),
            mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1"),
        ):
            command = cli.build_install_command(
                "server",
                python_executable=target_python,
                installer_backend="uv",
            )

        self.assertEqual(
            command[:5],
            [r"C:\tools\uv.exe", "pip", "install", "--python", str(target_python)],
        )
        self.assertEqual(command[-1], "whospeaks[server]==0.0.1")

    def test_uv_backend_fails_clearly_when_uv_is_unavailable(self) -> None:
        with mock.patch.object(cli.shutil, "which", return_value=None):
            with self.assertRaisesRegex(SystemExit, "uv.*not found"):
                cli.build_install_command("server", installer_backend="uv")

    def test_default_installer_prefers_uv_only_when_it_is_available(self) -> None:
        with mock.patch.object(cli.shutil, "which", return_value=r"C:\tools\uv.exe"):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(cli.INSTALLER_BACKEND_ENV, None)
                self.assertEqual(cli.normalize_installer_backend(None), "uv")
        with mock.patch.object(cli.shutil, "which", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(cli.INSTALLER_BACKEND_ENV, None)
                self.assertEqual(cli.normalize_installer_backend(None), "pip")

    def test_install_command_for_dev_build_keeps_testpypi_available(self) -> None:
        with mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev18"):
            command = cli.build_install_command(installer_backend="pip")

        self.assertIn("--extra-index-url", command)
        self.assertIn(cli.TESTPYPI_SIMPLE_URL, command)
        self.assertIn("whospeaks[complete,preview]==0.0.1.dev18", command)
        self.assertNotIn("--index-strategy", command)

    def test_uv_dev_build_prefers_pypi_files_before_falling_through_to_testpypi(self) -> None:
        with (
            mock.patch.object(cli.shutil, "which", return_value="uv"),
            mock.patch.object(cli, "installed_distribution_version", return_value="0.0.4.dev3"),
        ):
            command = cli.build_install_command("server", installer_backend="uv")

        strategy_index = command.index("--index-strategy")
        self.assertEqual(command[strategy_index + 1], "unsafe-first-match")
        first_index = command.index("--index")
        second_index = command.index("--index", first_index + 1)
        self.assertEqual(command[first_index + 1], cli.PYPI_SIMPLE_URL)
        self.assertEqual(command[second_index + 1], cli.TESTPYPI_SIMPLE_URL)
        self.assertNotIn("--extra-index-url", command)
        self.assertIn("whospeaks[server]==0.0.4.dev3", command)

    def test_torch_auto_installs_cuda_when_nvidia_driver_is_visible(self) -> None:
        completed = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            stdout="NVIDIA GeForce RTX 4090, 555.99\n",
            stderr="",
        )
        with (
            mock.patch.object(cli.shutil, "which", return_value="nvidia-smi"),
            mock.patch.object(cli.subprocess, "run", return_value=completed),
            mock.patch.object(cli.platform, "system", return_value="Windows"),
        ):
            command, selection = cli.build_torch_install_command("auto")

        self.assertEqual(selection.mode, "cuda")
        self.assertEqual(selection.build, "cu128")
        self.assertIn(cli.PYTORCH_CUDA_INDEX_URLS["cu128"], command)
        self.assertIn("torch>=2.2", command)
        self.assertIn("torchaudio>=2.2", command)
        self.assertIn('"torch>=2.2"', cli.format_command(command))

    def test_torch_auto_falls_back_to_cpu_without_nvidia_smi(self) -> None:
        with (
            mock.patch.object(cli.shutil, "which", return_value=None),
            mock.patch.object(cli.platform, "system", return_value="Windows"),
        ):
            command, selection = cli.build_torch_install_command("auto")

        self.assertEqual(selection.mode, "cpu")
        self.assertIn(cli.PYTORCH_CPU_INDEX_URL, command)
        self.assertIn("nvidia-smi was not found", selection.reason)

    def test_torch_skip_policy_returns_no_command(self) -> None:
        command, selection = cli.build_torch_install_command("skip")

        self.assertEqual(command, [])
        self.assertEqual(selection.mode, "skip")

    def test_uv_torch_command_targets_sidecar_python(self) -> None:
        with mock.patch.object(cli.shutil, "which", return_value="uv"):
            command, selection = cli.build_torch_install_command(
                "cpu",
                python_executable="sidecar-python",
                installer_backend="uv",
            )

        self.assertEqual(command[:5], ["uv", "pip", "install", "--python", "sidecar-python"])
        self.assertEqual(selection.mode, "cpu")

    def test_uv_translation_commands_target_the_isolated_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "translation-venv"
            with mock.patch.object(cli.shutil, "which", return_value="uv"):
                commands, python_executable, _model_path, _selection = cli.build_translation_install_commands(
                    "nllb-200-600m",
                    venv_dir=environment,
                    torch_policy="skip",
                    download_model=False,
                    installer_backend="uv",
                )

        uv_commands = [command for command in commands if command[:3] == ["uv", "pip", "install"]]
        self.assertGreaterEqual(len(uv_commands), 2)
        self.assertTrue(
            all(command[3:5] == ["--python", str(python_executable)] for command in uv_commands)
        )

    def test_install_extra_runs_torch_preinstall_before_whospeaks_extra(self) -> None:
        torch_command = ["python", "-m", "pip", "install", "torch"]
        package_command = ["python", "-m", "pip", "install", "whospeaks[complete]"]
        selection = cli.TorchInstallSelection("cuda", cli.PYTORCH_CUDA_INDEX_URLS["cu128"], "CUDA test", "cu128")
        calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if command[1:2] == ["-c"]:
                return subprocess.CompletedProcess(command, 0, stdout='{"cuda_available": true}\n', stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with (
            mock.patch.object(cli, "build_torch_install_command", return_value=(torch_command, selection)),
            mock.patch.object(cli, "build_install_command", return_value=package_command),
            mock.patch.object(cli.subprocess, "run", side_effect=fake_run),
        ):
            code = cli.install_extra(
                "complete",
                assume_yes=True,
                torch_policy="cuda",
                installer_backend="pip",
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            calls[0],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools>=68,<82",
            ],
        )
        self.assertEqual(calls[1], torch_command)
        self.assertEqual(calls[3], package_command)

    def test_kroko_install_command_uses_realtimestt_builder(self) -> None:
        command = cli.build_kroko_install_command("python", variant="free", work_dir=Path("kroko-work"))

        self.assertEqual(command, [
            "python",
            "-m",
            "RealtimeSTT.install_kroko",
            "--build",
            "--variant",
            "free",
            "--work-dir",
            "kroko-work",
        ])

    def test_kroko_sidecar_dry_run_prints_config_step(self) -> None:
        original_venv = os.environ.get(cli.KROKO_PREVIEW_VENV_ENV)
        with tempfile.TemporaryDirectory() as directory:
            os.environ[cli.KROKO_PREVIEW_VENV_ENV] = str(Path(directory) / "preview-venv")
            stdout = io.StringIO()
            try:
                with mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev19"):
                    with contextlib.redirect_stdout(stdout):
                        code = cli.install_kroko_sidecar(
                            cli.Profile(),
                            ["py", "-3.12"],
                            assume_yes=True,
                            dry_run=True,
                        )
            finally:
                if original_venv is None:
                    os.environ.pop(cli.KROKO_PREVIEW_VENV_ENV, None)
                else:
                    os.environ[cli.KROKO_PREVIEW_VENV_ENV] = original_venv

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Kroko native runtime setup", output)
        self.assertIn("RealtimeSTT.install_kroko", output)
        self.assertIn(cli.TESTPYPI_SIMPLE_URL, output)
        self.assertIn("whospeaks[preview]==0.0.1.dev19", output)
        self.assertIn("config", output)
        self.assertIn("realtime-preview-python", output)

    def test_uv_kroko_sidecar_uses_seeded_python312_environment(self) -> None:
        original_venv = os.environ.get(cli.KROKO_PREVIEW_VENV_ENV)
        with tempfile.TemporaryDirectory() as directory:
            os.environ[cli.KROKO_PREVIEW_VENV_ENV] = str(Path(directory) / "preview-venv")
            stdout = io.StringIO()
            try:
                with (
                    mock.patch.object(cli.shutil, "which", return_value="uv"),
                    contextlib.redirect_stdout(stdout),
                ):
                    code = cli.install_kroko_sidecar(
                        cli.Profile(),
                        None,
                        assume_yes=True,
                        dry_run=True,
                        installer_backend="uv",
                    )
            finally:
                if original_venv is None:
                    os.environ.pop(cli.KROKO_PREVIEW_VENV_ENV, None)
                else:
                    os.environ[cli.KROKO_PREVIEW_VENV_ENV] = original_venv

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("uv venv --python 3.12 --seed", output)
        self.assertIn("uv pip install --python", output)

    def test_windows_python312_command_falls_back_to_common_install_path(self) -> None:
        fake_path = Path(r"C:\Python\Python312\python.exe")

        def fake_info(command: list[str]) -> dict[str, object] | None:
            if command == [str(fake_path)]:
                return {"version": [3, 12, 4], "bits": 64, "machine": "AMD64", "executable": str(fake_path)}
            return None

        with (
            mock.patch.object(cli.shutil, "which", return_value=None),
            mock.patch.object(cli.Path, "is_file", lambda self: str(self) == str(fake_path)),
            mock.patch.object(cli, "query_python_command_info", side_effect=fake_info),
        ):
            command = cli.windows_python312_command()

        self.assertEqual(command, [str(fake_path)])

    def test_local_setup_dry_run_offers_complete_nemotron_install(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with mock.patch.object(cli, "windows_python312_command", return_value=None):
                    with contextlib.redirect_stdout(stdout):
                        code = cli.main(["setup", "--mode", "local", "--install", "--dry-run", "--yes"])
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Dry run: would save local profile", output)
        self.assertIn("whospeaks[complete]", output)
        self.assertNotIn("whospeaks[complete,preview]", output)
        self.assertFalse(config_path.exists())

    def test_install_local_without_kroko_uses_complete_plan(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with (
                    mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev21"),
                    mock.patch.object(cli, "run_doctor", return_value=cli.DoctorReport("local", [])),
                    contextlib.redirect_stdout(stdout),
                ):
                    code = cli.main([
                        "install",
                        "--target",
                        "local",
                        "--without-kroko",
                        "--dry-run",
                        "--yes",
                    ])
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Full local installation", output)
        self.assertIn("Realtime text: disabled", output)
        self.assertIn("Run the installer again and choose Kroko", output)
        self.assertNotIn("Run with --with-kroko", output)
        self.assertIn("whospeaks[complete]==0.0.1.dev21", output)
        self.assertNotIn("whospeaks[complete,preview]==0.0.1.dev21", output)
        self.assertFalse(config_path.exists())

    def test_install_server_uv_dry_run_propagates_installer_backend(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with (
                    mock.patch.object(cli.shutil, "which", return_value="uv"),
                    mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1"),
                    mock.patch.object(cli, "run_doctor", return_value=cli.DoctorReport("server", [])),
                    contextlib.redirect_stdout(stdout),
                ):
                    code = cli.main([
                        "install",
                        "--target",
                        "server",
                        "--installer",
                        "uv",
                        "--torch",
                        "skip",
                        "--dry-run",
                        "--yes",
                    ])
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Python package installer: uv", output)
        self.assertIn("uv pip install --python", output)
        self.assertIn("whospeaks[server]==0.0.1", output)
        self.assertFalse(config_path.exists())

    def test_install_local_with_nemotron_dry_run_uses_complete_and_model_preset(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with (
                    mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev27"),
                    mock.patch.object(cli, "run_doctor", return_value=cli.DoctorReport("local", [])),
                    contextlib.redirect_stdout(stdout),
                ):
                    code = cli.main([
                        "install",
                        "--target",
                        "local",
                        "--language",
                        "de",
                        "--realtime-preview-engine",
                        "sherpa_onnx",
                        "--realtime-preview-model-preset",
                        "nemotron-3.5-160ms-int8",
                        "--realtime-preview-model-dir",
                        str(Path(directory) / "nemotron"),
                        "--dry-run",
                        "--yes",
                    ])
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Nemotron 3.5 (nemotron-3.5-160ms-int8; lower latency)", output)
        self.assertIn("verified model downloads on first launch", output)
        self.assertIn("whospeaks[complete]==0.0.1.dev27", output)
        self.assertNotIn("whospeaks[complete,preview]==0.0.1.dev27", output)
        self.assertFalse(config_path.exists())

    def test_install_local_with_kroko_prints_sidecar_plan(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        report = cli.DoctorReport(
            "local",
            [cli.CheckResult("Kroko ONNX runtime", "warn", "missing")],
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with (
                    mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev21"),
                    mock.patch.object(cli, "run_doctor", return_value=report),
                    mock.patch.object(cli, "windows_python312_command", return_value=["py", "-3.12"]),
                    contextlib.redirect_stdout(stdout),
                ):
                    code = cli.main([
                        "install",
                        "--target",
                        "local",
                        "--with-kroko",
                        "--dry-run",
                        "--yes",
                    ])
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("whospeaks[complete,preview]==0.0.1.dev21", output)
        self.assertIn("whospeaks[preview]==0.0.1.dev21", output)
        self.assertIn("RealtimeSTT.install_kroko", output)
        self.assertFalse(config_path.exists())

    def test_install_core_without_kroko_uses_controller_plan(self) -> None:
        plan = cli.install_plan_for_target("core", install_kroko=False)

        self.assertEqual(plan.mode, "remote")
        self.assertEqual(plan.extra, cli.CONTROLLER_EXTRA)
        self.assertFalse(plan.install_kroko)

    def test_install_local_with_nemotron_uses_complete_plan_without_kroko_sidecar(self) -> None:
        plan = cli.install_plan_for_target(
            "local",
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_model_preset="nemotron-3.5-160ms-int8",
        )

        self.assertEqual(plan.extra, cli.COMPLETE_EXTRA)
        self.assertEqual(plan.realtime_preview_engine, "sherpa_onnx")
        self.assertEqual(plan.realtime_preview_model_preset, "nemotron-3.5-160ms-int8")
        self.assertFalse(plan.install_kroko)

    def test_profile_migrates_kroko_engine_to_its_own_default_preset(self) -> None:
        profile = cli.Profile.from_mapping(
            {
                "realtime_preview_engine": "kroko_onnx",
                "realtime_preview_model_preset": "nemotron-3.5-560ms-int8",
            }
        )

        self.assertEqual(profile.realtime_preview_engine, "kroko_onnx")
        self.assertEqual(profile.realtime_preview_model_preset, "community-64l")

    def test_install_server_profile_disables_realtime_preview(self) -> None:
        plan = cli.install_plan_for_target(
            "server",
            install_kroko=True,
            translation_model_profile="nllb-200-600m",
        )
        profile = cli.configure_profile_for_install(cli.Profile(), plan)

        self.assertEqual(plan.extra, cli.SERVER_EXTRA)
        self.assertFalse(plan.install_kroko)
        self.assertEqual(profile.mode, "server")
        self.assertEqual(profile.realtime_preview_engine, "off")
        self.assertEqual(plan.translation_model_profile, "off")
        self.assertFalse(profile.translation_enabled)

    def test_remote_launch_command_includes_remote_urls(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(), "remote")
        profile = profile.with_updates(
            remote_asr_url="http://gpu.example:8650",
            remote_embeddings_url="http://gpu.example:8660",
        )

        command = cli.build_launch_command(profile)

        self.assertIn("--asr-backend", command)
        self.assertIn("remote", command)
        self.assertIn("--model", command)
        self.assertIn("large-v2", command)
        self.assertIn("--device", command)
        self.assertIn("auto", command)
        self.assertIn("--remote-asr-url", command)
        self.assertIn("http://gpu.example:8650", command)
        self.assertIn("--remote-embeddings-url", command)
        self.assertIn("http://gpu.example:8660", command)
        self.assertEqual(command[:3], [sys.executable, "-m", "window.youtube_window_diarize_gui"])

    def test_reports_command_inherits_the_live_profile_language(self) -> None:
        profile = cli.Profile(
            language="de",
            host="127.0.0.1",
            text_embedding_base_url="http://embeddings.example/v1",
            text_embedding_model="multilingual-e5",
            text_embedding_api_key_env="EMBEDDING_API_KEY",
        )

        command = cli.build_reports_command(
            profile,
            llm_provider="openai",
            llm_model="gpt-4.1-nano",
        )

        self.assertIn("--report-language", command)
        self.assertEqual(command[command.index("--report-language") + 1], "de")
        self.assertIn("--auto-generate", command)
        self.assertEqual(command[command.index("--llm-provider") + 1], "openai")
        self.assertEqual(command[command.index("--text-embedding-base-url") + 1], "http://embeddings.example/v1")
        self.assertEqual(command[command.index("--text-embedding-model") + 1], "multilingual-e5")
        self.assertEqual(command[command.index("--text-embedding-api-key-env") + 1], "EMBEDDING_API_KEY")
        self.assertEqual(command[:3], [sys.executable, "-m", "window.meeting_intelligence_server"])

    def test_preferred_meeting_intelligence_flag_configures_live_proxy(self) -> None:
        profile = cli.Profile(language="en", host="0.0.0.0", reports_port=8898)
        stdout = io.StringIO()
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main([
                "launch", "--with-meeting-intelligence", "--reports-port", "8899",
                "--report-llm-model", "test-report-model", "--print",
            ])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Meeting Intelligence — Reports + Ask command:", output)
        self.assertIn("--meeting-intelligence-url http://127.0.0.1:8899", output)
        self.assertIn("--port 8899", output)

    def test_launch_with_reports_prints_both_commands(self) -> None:
        profile = cli.Profile(language="es")
        stdout = io.StringIO()
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main([
                "launch",
                "--with-reports",
                "--report-llm-provider",
                "openai",
                "--report-llm-model",
                "gpt-4.1-nano",
                "--print",
            ])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Meeting Intelligence — Reports + Ask command:", output)
        self.assertIn("--report-language es", output)
        self.assertIn("--llm-model gpt-4.1-nano", output)
        self.assertIn("Live window command:", output)

    def test_launch_with_reports_refuses_to_assume_an_llm_model(self) -> None:
        profile = cli.Profile(language="en")
        with mock.patch.object(cli, "load_profile", return_value=profile):
            with self.assertRaisesRegex(SystemExit, "explicit model ID"):
                cli.main(["launch", "--with-reports", "--print"])

    def test_launch_with_reports_and_translation_prints_one_live_header(self) -> None:
        profile = cli.Profile(
            reports_enabled=True,
            report_llm_model="test-report-model",
            translation_enabled=True,
            translation_provider="sidecar",
        )
        stdout = io.StringIO()
        with (
            mock.patch.object(cli, "load_profile", return_value=profile),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(["launch", "--with-reports", "--print"])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertEqual(output.count("Live window command:"), 1)
        self.assertLess(
            output.index("Meeting Intelligence — Reports + Ask command:"),
            output.index("Translation sidecar command:"),
        )
        self.assertLess(
            output.index("Translation sidecar command:"),
            output.index("Live window command:"),
        )

    def test_local_profile_uses_auto_device_by_default(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(device="cuda"), "local")

        self.assertEqual(profile.device, "auto")

    def test_local_profile_enables_nemotron_preview_by_default(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(realtime_preview_engine="off"), "local")

        self.assertEqual(profile.realtime_preview_engine, "sherpa_onnx")
        self.assertEqual(profile.realtime_preview_model_preset, "nemotron-3.5-560ms-int8")
        command = cli.build_launch_command(profile)
        self.assertIn("--realtime-preview-engine", command)
        self.assertIn("sherpa_onnx", command)
        self.assertIn("--realtime-preview-model-preset", command)
        self.assertIn("nemotron-3.5-560ms-int8", command)
        self.assertIn("--realtime-preview-python", command)
        self.assertEqual(command[command.index("--realtime-preview-python") + 1], sys.executable)

    def test_local_launch_command_uses_current_python_for_embedding_helper_by_default(self) -> None:
        profile = cli.Profile()

        command = cli.build_launch_command(profile)

        self.assertIn("--embedding-python", command)
        self.assertEqual(command[command.index("--embedding-python") + 1], sys.executable)

    def test_local_launch_command_includes_saved_embedding_python(self) -> None:
        profile = cli.Profile(embedding_python="/opt/whospeaks/bin/python")

        command = cli.build_launch_command(profile)

        self.assertIn("--embedding-python", command)
        self.assertIn("/opt/whospeaks/bin/python", command)

    def test_remote_launch_command_does_not_force_embedding_python(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(embedding_python="/opt/whospeaks/bin/python"), "remote")

        command = cli.build_launch_command(profile)

        self.assertNotIn("--embedding-python", command)

    def test_kroko_launch_command_includes_saved_realtime_preview_python(self) -> None:
        profile = cli.Profile(
            realtime_preview_engine="kroko_onnx",
            realtime_preview_model_preset="community-64l",
            realtime_preview_python=r"C:\Python\Python312\python.exe",
        )

        command = cli.build_launch_command(profile)

        self.assertIn("--realtime-preview-python", command)
        self.assertIn(r"C:\Python\Python312\python.exe", command)

    def test_nemotron_launch_ignores_saved_kroko_preview_python(self) -> None:
        stale_kroko_python = r"C:\WhoSpeaks\kroko-preview-py312\Scripts\python.exe"
        profile = cli.Profile(
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_python=stale_kroko_python,
        )

        command = cli.build_launch_command(profile)

        self.assertEqual(command[command.index("--realtime-preview-python") + 1], sys.executable)
        self.assertNotIn(stale_kroko_python, command)
        self.assertEqual(profile.realtime_preview_python, stale_kroko_python)

    def test_nemotron_doctor_rejects_importable_runtime_without_online_recognizer(self) -> None:
        with mock.patch.object(cli.importlib, "import_module", return_value=object()):
            check = cli.check_sherpa_onnx_runtime()

        self.assertEqual(check.status, "warn")
        self.assertIn("sherpa_onnx.OnlineRecognizer is missing", check.detail)
        self.assertIn(sys.executable, check.detail)

    def test_nemotron_doctor_accepts_online_recognizer_runtime(self) -> None:
        runtime = mock.Mock(OnlineRecognizer=object())
        with mock.patch.object(cli.importlib, "import_module", return_value=runtime):
            check = cli.check_sherpa_onnx_runtime()

        self.assertEqual(check.status, "ok")
        self.assertIn("sherpa_onnx.OnlineRecognizer", check.detail)

    def test_nemotron_doctor_ignores_saved_kroko_preview_python(self) -> None:
        profile = cli.Profile(
            realtime_preview_engine="sherpa_onnx",
            realtime_preview_python=r"C:\WhoSpeaks\kroko-preview-py312\Scripts\python.exe",
        )
        runtime_check = cli.CheckResult("Nemotron sherpa-onnx runtime", "ok", "current runtime")
        with (
            mock.patch.object(cli, "check_sherpa_onnx_runtime", return_value=runtime_check) as check_current,
            mock.patch.object(cli, "check_python_imports") as check_sidecar,
        ):
            report = cli.run_doctor(profile, mode="server")

        check_current.assert_called_once_with()
        check_sidecar.assert_not_called()
        self.assertIn(runtime_check, report.checks)

    def test_window_parser_preserves_python_executable_symlinks(self) -> None:
        module_path = ROOT / "src" / "window" / "window_cli.py"
        source = module_path.read_text(encoding="utf-8")

        self.assertIn("_absolute_path_preserving_symlinks(args.embedding_python)", source)
        self.assertIn("_absolute_path_preserving_symlinks(args.realtime_preview_python)", source)
        self.assertNotIn("args.embedding_python = args.embedding_python.resolve()", source)
        self.assertNotIn("args.realtime_preview_python = args.realtime_preview_python.resolve()", source)

    def test_server_launch_lines_start_both_services(self) -> None:
        lines = cli.build_server_launch_lines()

        self.assertEqual(len(lines), 2)
        self.assertTrue(any("asr_server:app" in line for line in lines))
        self.assertTrue(any("embeddings_server:app" in line for line in lines))
        self.assertFalse(any("whospeaks-window" in line for line in lines))

    def test_doctor_json_is_machine_readable_without_strict_failures(self) -> None:
        stdout = io.StringIO()
        report = cli.DoctorReport(
            "local",
            [cli.CheckResult("Python", "ok", "test runtime")],
        )
        with (
            mock.patch.object(cli, "run_doctor", return_value=report),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(["doctor", "--mode", "local", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "local")
        self.assertTrue(any(item["name"] == "Python" for item in payload["checks"]))

    def test_doctor_auto_mode_uses_saved_remote_profile(self) -> None:
        saved_profile = cli.Profile(
            mode="remote",
            asr_backend="remote",
            embeddings_backend="remote",
        )

        def doctor_runner(profile: cli.Profile, mode: str, *, deep: bool) -> cli.DoctorReport:
            selected_mode = profile.mode if mode == "auto" else mode
            remote_status = "ok" if selected_mode == "remote" else "skip"
            return cli.DoctorReport(
                selected_mode,
                [cli.CheckResult("Remote ASR health", remote_status, selected_mode)],
            )

        stdout = io.StringIO()
        with (
            mock.patch.object(cli, "load_profile", return_value=saved_profile),
            mock.patch.object(cli, "run_doctor", side_effect=doctor_runner) as run_doctor,
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(["doctor", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "remote")
        remote_health = next(
            check for check in payload["checks"] if check["name"] == "Remote ASR health"
        )
        self.assertEqual(remote_health["status"], "ok")
        called_profile, called_mode = run_doctor.call_args.args
        self.assertEqual(called_profile.mode, "remote")
        self.assertEqual(called_mode, "auto")

    def test_remote_doctor_skips_unneeded_local_runtime_packages(self) -> None:
        profile = cli.Profile(
            mode="remote",
            asr_backend="remote",
            embeddings_backend="remote",
            realtime_preview_engine="off",
        )
        with (
            mock.patch(
                "whospeaks_cli.cli_diagnostics.command_version",
                return_value=(True, "ffmpeg"),
            ),
            mock.patch(
                "whospeaks_cli.cli_diagnostics.check_import_group",
                side_effect=lambda name, *_args, **_kwargs: cli.CheckResult(name, "ok", "available"),
            ),
            mock.patch.object(
                cli,
                "check_remote_health",
                side_effect=lambda name, *_args, **_kwargs: cli.CheckResult(name, "ok", "reachable"),
            ),
            mock.patch(
                "whospeaks_cli.cli_diagnostics.check_remote_providers",
                return_value=cli.CheckResult("Remote embeddings providers", "ok", "available"),
            ),
        ):
            report = cli.run_doctor(profile)

        local_asr = next(check for check in report.checks if check.name == "Local ASR modules")
        local_embeddings = next(
            check for check in report.checks if check.name == "Local embedding modules"
        )
        self.assertEqual(local_asr.status, "skip")
        self.assertEqual(local_embeddings.status, "skip")
        self.assertIn("external ASR service", local_asr.detail)
        self.assertIn("external embeddings service", local_embeddings.detail)

    def test_module_entrypoint_prints_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, "-m", "whospeaks_cli", "--help"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("WhoSpeaks setup, doctor, and launcher", completed.stdout)

    def test_plain_whospeaks_opens_desktop_launcher(self) -> None:
        with (
            mock.patch.object(cli.sys.stdin, "isatty", return_value=True),
            mock.patch.object(cli, "desktop_session_available", return_value=True),
            mock.patch.object(cli, "run_desktop_dashboard", return_value=0) as desktop,
        ):
            code = cli.main([])

        self.assertEqual(code, 0)
        desktop.assert_called_once_with()

    def test_gui_flag_can_open_desktop_launcher_from_noninteractive_shell(self) -> None:
        with (
            mock.patch.object(cli.sys.stdin, "isatty", return_value=False),
            mock.patch.object(cli, "desktop_session_available", return_value=True),
            mock.patch.object(cli, "run_desktop_dashboard", return_value=0) as desktop,
        ):
            code = cli.main(["--gui"])

        self.assertEqual(code, 0)
        desktop.assert_called_once_with()

    def test_removed_terminal_launcher_flags_are_rejected(self) -> None:
        for flag in ("--tui", "--classic"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                cli.build_parser().parse_args([flag])

    def test_full_profile_editor_lists_every_saved_field_once(self) -> None:
        output = cli.full_profile_editor_text(cli.Profile())

        self.assertEqual(len(cli.EDITABLE_PROFILE_FIELDS), len(cli.profile_field_names()))
        for index, (field, label, _help_text) in enumerate(cli.EDITABLE_PROFILE_FIELDS, start=1):
            self.assertIn(f"{index:>2}. {label:<25}", output, field)
        self.assertIn("Embedding helper Python", output)
        self.assertIn("Realtime preview Python", output)
        self.assertIn("Advanced args", output)

    def test_provider_presets_include_validation_notes(self) -> None:
        self.assertIn("Baseline smoke setting", cli.PROVIDER_PRESETS["smoke"].score_note)
        self.assertIn("until validation decides", cli.PROVIDER_PRESETS["promoted_public"].score_note)
        self.assertNotIn("tuned_private", cli.PROVIDER_PRESETS)
        self.assertEqual(cli.normalize_provider_preset_id("tuned_private"), "custom")

    def test_dashboard_summarizes_problems_without_full_ok_list(self) -> None:
        profile = cli.Profile()
        report = cli.DoctorReport(
            "local",
            [
                cli.CheckResult("Python", "ok", "CPython 3.11"),
                cli.CheckResult("ffmpeg", "fail", "missing", "Install ffmpeg."),
                cli.CheckResult("Remote ASR", "skip", "not needed"),
            ],
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.render_dashboard(profile, report)

        output = stdout.getvalue()
        self.assertIn("Readiness: Action needed: 1 failed check; 1 skipped", output)
        self.assertIn("FAIL  ffmpeg", output)
        self.assertNotIn("OK    Python", output)

    def test_provider_preset_expands_to_exact_launch_providers(self) -> None:
        profile = cli.Profile()

        profile = cli.apply_provider_preset(profile, "public_quality")
        command = cli.build_launch_command(profile)

        self.assertEqual(profile.provider_preset, "public_quality")
        self.assertIn("--embedding-provider", command)
        self.assertIn(cli.PUBLIC_PROVIDER, command)
        self.assertIn("--live-speaker-embedding-provider", command)
        self.assertIn(cli.FAST_LIVE_PROVIDER, command)

    def test_existing_corrupt_profile_is_reported_instead_of_silently_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(cli.ProfileLoadError, "could not use the saved profile"):
                cli.load_profile(config_path)

    def test_existing_invalid_profile_value_is_not_silently_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            payload = cli.Profile().as_dict()
            payload["translation_provider"] = "mock"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(cli.ProfileLoadError, "translation_provider"):
                cli.load_profile(config_path)

    def test_provider_preset_is_custom_when_provider_strings_do_not_match(self) -> None:
        profile = cli.Profile.from_mapping({
            "provider_preset": "public_quality",
            "embedding_provider": "custom_final_provider",
            "live_speaker_embedding_provider": cli.FAST_LIVE_PROVIDER,
        })

        self.assertEqual(profile.provider_preset, "custom")

    def test_config_set_provider_preset_saves_expanded_provider_strings(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(["config", "--set", "provider_preset=smoke_fast_live"])
                profile = cli.load_profile()
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        self.assertEqual(profile.provider_preset, "smoke_fast_live")
        self.assertEqual(profile.embedding_provider, cli.SMOKE_PROVIDER)
        self.assertEqual(profile.live_speaker_embedding_provider, cli.FAST_LIVE_PROVIDER)

    def test_config_language_flag_saves_profile_language(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(["config", "--language", "de", "--json"])
                payload = json.loads(stdout.getvalue())
                profile = cli.load_profile()
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        self.assertEqual(payload["language"], "de")
        self.assertEqual(profile.language, "de")

    def test_config_realtime_preview_python_saves_profile(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main([
                        "config",
                        "--realtime-preview-python",
                        r"C:\Python\Python312\python.exe",
                        "--json",
                    ])
                payload = json.loads(stdout.getvalue())
                profile = cli.load_profile()
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        self.assertEqual(payload["realtime_preview_python"], r"C:\Python\Python312\python.exe")
        self.assertEqual(profile.realtime_preview_python, r"C:\Python\Python312\python.exe")

    def test_config_embedding_python_saves_profile(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main([
                        "config",
                        "--embedding-python",
                        sys.executable,
                        "--json",
                    ])
                payload = json.loads(stdout.getvalue())
                profile = cli.load_profile()
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        self.assertEqual(payload["embedding_python"], sys.executable)
        self.assertEqual(profile.embedding_python, sys.executable)

    def test_launch_language_flag_overrides_without_saving(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            cli.save_profile(cli.Profile(language="en"))
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(["launch", "--language", "de", "--print"])
                profile = cli.load_profile()
            finally:
                if original_config is None:
                    os.environ.pop("WHOSPEAKS_CONFIG", None)
                else:
                    os.environ["WHOSPEAKS_CONFIG"] = original_config

        self.assertEqual(code, 0)
        command = stdout.getvalue()
        self.assertIn("--language", command)
        self.assertIn("de", command)
        self.assertEqual(profile.language, "en")

    def test_realtime_helper_imports_sys_for_parse_args(self) -> None:
        module_path = ROOT / "src" / "realtime" / "realtime_speakerdiarize.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imports_sys = any(
            isinstance(node, ast.Import) and any(alias.name == "sys" for alias in node.names)
            for node in tree.body
        )
        uses_sys_argv = any(
            isinstance(node, ast.Attribute)
            and node.attr == "argv"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            for node in ast.walk(tree)
        )

        self.assertTrue(uses_sys_argv)
        self.assertTrue(imports_sys)

    def test_pyproject_declares_top_level_script_and_extras(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(
            pyproject["project"]["scripts"]["whospeaks"],
            "whospeaks_cli.main:main",
        )
        self.assertEqual(
            pyproject["project"]["scripts"]["whospeaks-gui"],
            "whospeaks_gui.main:main",
        )
        extras = pyproject["project"]["optional-dependencies"]
        self.assertEqual(extras["gui"], [])
        self.assertIn("controller", extras)
        self.assertNotIn("librosa>=0.10.1", extras["controller"])
        self.assertIn("server", extras)
        self.assertIn("complete", extras)
        self.assertIn("all", extras)
        self.assertIn("faster-whisper>=1.2.1", extras["complete"])
        self.assertIn("espnet==202511", extras["complete"])
        self.assertIn("numpy>=2,<3", extras["complete"])
        self.assertNotIn("RealTimeSTT==0.1.13", extras["complete"])
        self.assertNotIn("RealTimeSTT==0.1.13", extras["preview"])
        self.assertNotIn("RealTimeSTT==0.1.13", extras["all"])
        self.assertIn("numpy>=2,<3", extras["preview"])
        self.assertIn("webrtcvad==2.0.10", extras["preview"])
        self.assertIn("PySide6>=6.8,<7", pyproject["project"]["dependencies"])
        self.assertFalse(
            any(dependency.startswith("textual") for dependency in pyproject["project"]["dependencies"])
        )


if __name__ == "__main__":
    unittest.main()
