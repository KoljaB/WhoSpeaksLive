from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whospeaks_cli import main as cli
from whospeaks_cli.tui import provider_preset_label


class WhoSpeaksCliTests(unittest.TestCase):
    def test_clean_wheel_contains_server_resources_and_uses_packaged_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            source = temporary_root / "source"
            wheel_directory = temporary_root / "wheel"
            source.mkdir()
            wheel_directory.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
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

        self.assertIn("remote_servers/faster-whisper-asr/asr_server.py", names)
        self.assertIn("remote_servers/faster-whisper-asr/requirements.txt", names)
        self.assertIn("remote_servers/voice-embeddings-server/embeddings_server.py", names)
        self.assertIn("remote_servers/voice-embeddings-server/requirements.txt", names)
        self.assertNotIn("remote_servers/faster-whisper-asr/test_parent_watchdog.py", names)
        self.assertNotIn("remote_servers/voice-embeddings-server/tools/benchmark_voice_embeddings.py", names)
        self.assertEqual(len(launch_lines), 2)
        self.assertIn(str(installed / "remote_servers" / "faster-whisper-asr"), launch_lines[0])
        self.assertIn(str(installed / "remote_servers" / "voice-embeddings-server"), launch_lines[1])

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
            command = cli.build_install_command()

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

    def test_install_command_for_dev_build_keeps_testpypi_available(self) -> None:
        with mock.patch.object(cli, "installed_distribution_version", return_value="0.0.1.dev18"):
            command = cli.build_install_command()

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
            code = cli.install_extra("complete", assume_yes=True, torch_policy="cuda")

        self.assertEqual(code, 0)
        self.assertEqual(calls[0], torch_command)
        self.assertEqual(calls[2], package_command)

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
        plan = cli.install_plan_for_target("server", install_kroko=True)
        profile = cli.configure_profile_for_install(cli.Profile(), plan)

        self.assertEqual(plan.extra, cli.SERVER_EXTRA)
        self.assertFalse(plan.install_kroko)
        self.assertEqual(profile.mode, "server")
        self.assertEqual(profile.realtime_preview_engine, "off")

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

    def test_reports_command_inherits_the_live_profile_language(self) -> None:
        profile = cli.Profile(language="de", host="127.0.0.1")

        command = cli.build_reports_command(
            profile,
            llm_provider="openai",
            llm_model="gpt-4.1-nano",
        )

        self.assertIn("--report-language", command)
        self.assertEqual(command[command.index("--report-language") + 1], "de")
        self.assertIn("--auto-generate", command)
        self.assertEqual(command[command.index("--llm-provider") + 1], "openai")

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
        self.assertIn("Meeting reports command:", output)
        self.assertIn("--report-language es", output)
        self.assertIn("--llm-model gpt-4.1-nano", output)
        self.assertIn("Live window command:", output)

    def test_launch_with_reports_and_translation_prints_one_live_header(self) -> None:
        profile = cli.Profile(
            reports_enabled=True,
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
            output.index("Meeting reports command:"),
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
        with contextlib.redirect_stdout(stdout):
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

    def test_default_dashboard_uses_simple_menu_labels(self) -> None:
        output = cli.main_menu_text()

        self.assertIn("1. Install or repair WhoSpeaks", output)
        self.assertIn("2. Launch browser UI", output)
        self.assertIn("3. Doctor / complete diagnostics", output)
        self.assertIn("4. Language and realtime text", output)
        self.assertIn("5. Speaker provider quality", output)
        self.assertIn("p. Print exact launch command", output)
        self.assertIn("r. Remote/server profiles", output)
        self.assertEqual(output.count("Install or repair WhoSpeaks"), 1)
        self.assertNotIn("Guided install", output)
        self.assertNotIn("Controller + remote GPU services profile", output)

    def test_classic_whospeaks_menu_starts_installer_from_option_one(self) -> None:
        profile = cli.Profile()
        report = cli.DoctorReport("local", [])
        stdout = io.StringIO()

        with (
            mock.patch.object(cli.sys.stdin, "isatty", return_value=True),
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch.object(cli, "run_doctor", return_value=report),
            mock.patch.object(cli, "render_dashboard"),
            mock.patch.object(cli, "recommended_install_extra", return_value=None),
            mock.patch.object(cli, "install_components_interactively", return_value=None) as installer,
            mock.patch.object(cli, "read_input", side_effect=["1", "q"]),
            contextlib.redirect_stdout(stdout),
        ):
            code = cli.main(["--classic"])

        self.assertEqual(code, 0)
        installer.assert_called_once_with(profile)
        self.assertIn("1. Install or repair WhoSpeaks", stdout.getvalue())

    def test_plain_whospeaks_starts_textual_dashboard(self) -> None:
        profile = cli.Profile()

        with (
            mock.patch.object(cli.sys.stdin, "isatty", return_value=True),
            mock.patch.object(cli, "load_profile", return_value=profile),
            mock.patch.object(cli, "run_textual_dashboard", return_value=0) as dashboard,
        ):
            code = cli.main([])

        self.assertEqual(code, 0)
        dashboard.assert_called_once_with(profile)

    def test_configuration_menu_exposes_important_launcher_parameters(self) -> None:
        output = cli.configuration_menu_text(cli.Profile(language="de"))

        self.assertIn("Language and realtime text", output)
        self.assertIn("Speaker provider quality", output)
        self.assertIn("Backends and remote URLs", output)
        self.assertIn("ASR model, device, and compute", output)
        self.assertIn("Browser host and port", output)
        self.assertIn("All saved profile fields", output)
        self.assertIn("German (de", output)

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
                        "/opt/whospeaks/bin/python",
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
        self.assertEqual(payload["embedding_python"], "/opt/whospeaks/bin/python")
        self.assertEqual(profile.embedding_python, "/opt/whospeaks/bin/python")

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
        extras = pyproject["project"]["optional-dependencies"]
        self.assertIn("controller", extras)
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
        self.assertIn("textual>=8.2,<9", pyproject["project"]["dependencies"])


if __name__ == "__main__":
    unittest.main()
