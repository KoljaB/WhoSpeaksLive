from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whospeaks_cli import main as cli


class WhoSpeaksCliTests(unittest.TestCase):
    def test_install_command_uses_local_preview_extra(self) -> None:
        command = cli.build_install_command()

        self.assertEqual(command[:3], [sys.executable, "-m", "pip"])
        self.assertEqual(command[-2:], ["install", "whospeaks[complete,preview]"])
        self.assertIn("whospeaks[complete,preview]", cli.format_command(command))

    def test_local_setup_dry_run_offers_complete_preview_install(self) -> None:
        original_config = os.environ.get("WHOSPEAKS_CONFIG")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            os.environ["WHOSPEAKS_CONFIG"] = str(config_path)
            stdout = io.StringIO()
            try:
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
        self.assertIn("whospeaks[complete,preview]", output)
        self.assertFalse(config_path.exists())

    def test_remote_launch_command_includes_remote_urls(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(), "remote")
        profile.remote_asr_url = "http://gpu.example:8650"
        profile.remote_embeddings_url = "http://gpu.example:8660"

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

    def test_local_profile_uses_auto_device_by_default(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(device="cuda"), "local")

        self.assertEqual(profile.device, "auto")

    def test_local_profile_enables_kroko_preview_by_default(self) -> None:
        profile = cli.configure_profile_for_mode(cli.Profile(realtime_preview_engine="off"), "local")

        self.assertEqual(profile.realtime_preview_engine, "kroko_onnx")
        command = cli.build_launch_command(profile)
        self.assertIn("--realtime-preview-engine", command)
        self.assertIn("kroko_onnx", command)
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

    def test_launch_command_includes_saved_realtime_preview_python(self) -> None:
        profile = cli.Profile(realtime_preview_python=r"C:\Python\Python312\python.exe")

        command = cli.build_launch_command(profile)

        self.assertIn("--realtime-preview-python", command)
        self.assertIn(r"C:\Python\Python312\python.exe", command)

    def test_window_parser_preserves_python_executable_symlinks(self) -> None:
        module_path = ROOT / "src" / "window" / "youtube_window_diarize_gui.py"
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

        self.assertIn("1. Launch browser UI", output)
        self.assertIn("3. Install recommended dependency group", output)
        self.assertIn("4. Language and realtime text", output)
        self.assertIn("5. Speaker provider quality", output)
        self.assertIn("p. Print exact launch command", output)
        self.assertIn("s. First-time full local setup", output)
        self.assertIn("r. Remote/server profiles", output)
        self.assertNotIn("Controller + remote GPU services profile", output)

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

        cli.apply_provider_preset(profile, "public_quality")
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
        self.assertIn("webrtcvad==2.0.10", extras["preview"])


if __name__ == "__main__":
    unittest.main()
