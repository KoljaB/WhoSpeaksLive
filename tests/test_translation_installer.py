from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whospeaks_cli import main as cli
from window.translation_installer import prepare_model


class TranslationInstallerTests(unittest.TestCase):
    def test_every_local_profile_builds_an_isolated_cross_platform_install_plan(self) -> None:
        for model_profile in cli.TRANSLATION_INSTALL_PROFILE_CHOICES[1:]:
            with self.subTest(model_profile=model_profile):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    commands, python_executable, model_dir, selection = cli.build_translation_install_commands(
                        model_profile,
                        venv_dir=root / "venv",
                        model_dir=root / "model",
                        torch_policy="skip",
                        download_model=False,
                    )
                self.assertEqual(commands[0][1:3], ["-m", "venv"])
                self.assertEqual(python_executable, cli.venv_python_path(root / "venv"))
                self.assertEqual(model_dir, (root / "model").resolve())
                self.assertEqual(selection.mode, "skip")
                self.assertTrue(any("window.translation_installer" in command for command in commands))
                self.assertIn("--verify-only", commands[-1])

    def test_model_verifier_accepts_an_existing_snapshot_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                payload = prepare_model("nllb-200-600m", model_dir, download=False)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model_profile"], "nllb-200-600m")
        self.assertEqual(json.loads(output.getvalue())["family"], "nllb")

    def test_install_translation_dry_run_does_not_mutate_profile(self) -> None:
        parsed = cli.build_parser().parse_args([
            "install-translation",
            "--model-profile", "nllb-200-600m",
            "--torch", "skip",
            "--skip-model-download",
            "--dry-run",
        ])
        profile = cli.Profile()
        with mock.patch.object(cli, "load_profile", return_value=profile):
            self.assertEqual(cli.cmd_install_translation(parsed), 0)
        self.assertFalse(profile.translation_enabled)

    def test_controller_install_plan_persists_selected_translation_sidecar(self) -> None:
        plan = cli.install_plan_for_target("core", translation_model_profile="madlad-400-3b")
        profile = cli.configure_profile_for_install(cli.Profile(), plan)
        self.assertTrue(profile.translation_enabled)
        self.assertEqual(profile.translation_provider, "sidecar")
        self.assertEqual(profile.translation_model_profile, "madlad-400-3b")

    def test_server_install_plan_does_not_enable_controller_translation(self) -> None:
        plan = cli.install_plan_for_target("server", translation_model_profile="madlad-400-3b")
        profile = cli.configure_profile_for_install(cli.Profile(), plan)
        self.assertFalse(profile.translation_enabled)
        self.assertEqual(plan.translation_model_profile, "off")


if __name__ == "__main__":
    unittest.main()
