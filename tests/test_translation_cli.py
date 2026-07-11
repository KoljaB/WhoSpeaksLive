from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from whospeaks_cli import main as cli
from whospeaks_cli.main import Profile, build_launch_command, build_translation_command


class TranslationCliTests(unittest.TestCase):
    def test_sidecar_profile_expands_to_live_multi_target_flags(self) -> None:
        profile = Profile.from_mapping(Profile(
            language="es",
            translation_enabled=True,
            translation_provider="sidecar",
            translation_port=8799,
            translation_target_languages="English, de, es, de",
            translation_max_targets=4,
            translation_model_profile="nllb-200-600m",
        ).as_dict())
        command = build_launch_command(profile)
        self.assertEqual(profile.translation_target_languages, "en,de")
        self.assertIn("--translation-provider", command)
        self.assertEqual(command[command.index("--translation-provider") + 1], "sidecar")
        self.assertIn("http://127.0.0.1:8799", command)
        self.assertEqual(command.count("--translation-target-language"), 2)
        self.assertIn("nllb-200-600m", command)

    def test_translation_sidecar_can_use_an_isolated_python(self) -> None:
        profile = Profile(
            translation_python=r"D:\translation\Scripts\python.exe",
            translation_model_profile="translate-gemma-4b",
            translation_device="cuda",
        )
        command = build_translation_command(profile)
        self.assertEqual(command[:3], [profile.translation_python, "-m", "window.translation_server"])
        self.assertIn("translate-gemma-4b", command)
        self.assertIn("cuda", command)

    def test_reports_llm_translation_reuses_saved_openai_compatible_settings(self) -> None:
        profile = Profile(
            translation_enabled=True,
            translation_provider="reports_llm",
            report_llm_provider="ollama",
            report_llm_base_url="http://reports.local:11434/v1",
            report_llm_model="gemma-report",
        )
        command = build_launch_command(profile)
        provider_index = command.index("--translation-provider")
        self.assertEqual(command[provider_index + 1], "openai_compatible")
        self.assertIn("http://reports.local:11434/v1", command)
        self.assertIn("gemma-report", command)

    def test_launch_translation_flag_temporarily_enables_disabled_saved_profile(self) -> None:
        parsed = cli.build_parser().parse_args(["launch", "--translation", "--print"])
        output = StringIO()
        with mock.patch.object(cli, "load_profile", return_value=Profile(translation_enabled=False)):
            with redirect_stdout(output):
                self.assertEqual(cli.cmd_launch(parsed), 0)
        rendered = output.getvalue()
        self.assertIn("Translation sidecar command:", rendered)
        self.assertIn("--translation-provider sidecar", rendered)

    def test_saved_translation_targets_reject_unknown_language_codes(self) -> None:
        with self.assertRaises(SystemExit):
            cli.apply_profile_updates(Profile(), [("translation_target_languages", "de,not-a-language")])


if __name__ == "__main__":
    unittest.main()
