"""Argument parser composition for the WhoSpeaks CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .cli_commands import (
    cmd_config,
    cmd_doctor,
    cmd_install,
    cmd_install_kroko,
    cmd_install_translation,
    cmd_launch,
    cmd_reports,
    cmd_setup,
    cmd_translation,
)
from .planning import INSTALL_TARGET_CHOICES, TRANSLATION_INSTALL_PROFILE_CHOICES
from .profiles import PROVIDER_PRESET_CHOICES
from .runtime_constants import INSTALLER_BACKEND_CHOICES, TORCH_INSTALL_POLICY_CHOICES


def _add_installer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--installer",
        choices=INSTALLER_BACKEND_CHOICES,
        default=None,
        help=(
            "Python package installer: pip (compatibility default) or uv. "
            "Defaults to WHOSPEAKS_INSTALLER, then pip."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whospeaks",
        description="WhoSpeaks setup, doctor, and launcher.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print the dashboard once and exit instead of opening the interactive starter CLI.",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Open the classic numbered interface instead of the Textual setup application.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Run setup and component checks.")
    doctor.add_argument("--mode", choices=("auto", "local", "remote", "server"), default="auto")
    doctor.add_argument("--language", default="", help="Temporarily check a language profile without saving it.")
    doctor.add_argument("--remote-asr-url", default="")
    doctor.add_argument("--remote-embeddings-url", default="")
    doctor.add_argument("--port", type=int, default=None)
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks such as remote /load.")
    doctor.add_argument("--strict", action="store_true", help="Return non-zero when required checks fail.")
    doctor.add_argument("--fix", action="store_true", help="Offer the recommended package install action after checks.")
    doctor.add_argument("--yes", action="store_true", help="Do not prompt before running the package install action.")
    doctor.add_argument("--dry-run", action="store_true", help="Print installer commands without running them.")
    doctor.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy for --fix: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    _add_installer_argument(doctor)
    doctor.set_defaults(func=cmd_doctor)

    install = subparsers.add_parser(
        "install",
        help="Guided installer for full local, Apple Silicon, core/controller, or server packages.",
    )
    install.add_argument(
        "--target",
        choices=INSTALL_TARGET_CHOICES,
        default="",
        help="Install target: local, macos, core, or server. Omit for an interactive choice.",
    )
    install.add_argument("--language", default="", help="Save the setup profile with this language code, for example de.")
    install.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    kroko_group = install.add_mutually_exclusive_group()
    kroko_group.add_argument(
        "--with-kroko",
        action="store_true",
        help="Enable realtime preview text and run the native Kroko setup after Python packages.",
    )
    kroko_group.add_argument(
        "--without-kroko",
        action="store_true",
        help="Disable realtime preview text for this install.",
    )
    install.add_argument(
        "--realtime-preview-engine",
        default="",
        help="Realtime text engine: sherpa_onnx (Nemotron), kroko_onnx, or off.",
    )
    install.add_argument(
        "--realtime-preview-model-preset",
        default="",
        help="Realtime model preset, for example nemotron-3.5-560ms-int8.",
    )
    install.add_argument(
        "--realtime-preview-model-dir",
        type=Path,
        default=None,
        help="Optional directory containing an unpacked Nemotron model.",
    )
    install.add_argument(
        "--translation-model-profile",
        choices=TRANSLATION_INSTALL_PROFILE_CHOICES,
        default="",
        help="Install an isolated local translation sidecar: nllb-200-600m, translate-gemma-4b, madlad-400-3b, or off.",
    )
    install.add_argument("--translation-venv", type=Path, default=None, help="Override the model-specific translation virtual environment.")
    install.add_argument("--translation-model-dir", type=Path, default=None, help="Override the local translation model directory.")
    install.add_argument("--skip-translation-model-download", action="store_true", help="Install the runtime but only verify already present model files.")
    install.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks after installation.")
    install.add_argument("--yes", action="store_true", help="Do not prompt before running installer actions.")
    install.add_argument("--dry-run", action="store_true", help="Print installer actions without running them.")
    install.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    _add_installer_argument(install)
    install.set_defaults(func=cmd_install)

    install_translation = subparsers.add_parser(
        "install-translation",
        help="Install one isolated local translation server and optionally download its model.",
    )
    install_translation.add_argument("--model-profile", choices=TRANSLATION_INSTALL_PROFILE_CHOICES[1:], required=True)
    install_translation.add_argument("--venv", type=Path, default=None)
    install_translation.add_argument("--model-dir", type=Path, default=None)
    install_translation.add_argument("--skip-model-download", action="store_true")
    install_translation.add_argument("--yes", action="store_true")
    install_translation.add_argument("--dry-run", action="store_true")
    install_translation.add_argument("--torch", choices=TORCH_INSTALL_POLICY_CHOICES, default="auto")
    _add_installer_argument(install_translation)
    install_translation.set_defaults(func=cmd_install_translation)

    setup = subparsers.add_parser("setup", help="Choose a setup mode and optionally install dependencies.")
    setup.add_argument("--mode", choices=("local", "remote", "server"), default="local")
    setup.add_argument("--language", default="", help="Save the setup profile with this language code, for example de.")
    setup.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    setup.add_argument("--install", action="store_true", help="Run the recommended package installer for this mode.")
    setup.add_argument(
        "--skip-kroko",
        action="store_true",
        help="Do not offer/build the native Kroko realtime preview runtime after installing extras.",
    )
    setup.add_argument(
        "--realtime-preview-engine",
        default="",
        help="Realtime text engine: sherpa_onnx (Nemotron), kroko_onnx, or off.",
    )
    setup.add_argument(
        "--realtime-preview-model-preset",
        default="",
        help="Realtime model preset, for example nemotron-3.5-560ms-int8.",
    )
    setup.add_argument(
        "--realtime-preview-model-dir",
        type=Path,
        default=None,
        help="Optional directory containing an unpacked Nemotron model.",
    )
    setup.add_argument("--deep", action="store_true", help="Run expensive provider/cache checks during setup.")
    setup.add_argument("--yes", action="store_true", help="Do not prompt before running installer actions.")
    setup.add_argument("--dry-run", action="store_true", help="Print installer actions without running them.")
    setup.add_argument(
        "--torch",
        choices=TORCH_INSTALL_POLICY_CHOICES,
        default="auto",
        help="Torch wheel policy: auto-detect CUDA, force cuda/cpu, or skip Torch preinstall.",
    )
    _add_installer_argument(setup)
    setup.set_defaults(func=cmd_setup)

    install_kroko = subparsers.add_parser(
        "install-kroko",
        help="Build/install the native Kroko realtime preview runtime.",
    )
    install_kroko.add_argument("--python", default="", help="Python executable that should receive kroko_onnx.")
    install_kroko.add_argument(
        "--engine",
        default="kroko_onnx",
        help="Realtime preview engine to enable while installing. Use off to skip.",
    )
    install_kroko.add_argument("--variant", choices=("free", "pro"), default="free")
    install_kroko.add_argument("--work-dir", type=Path, default=None)
    install_kroko.add_argument("--yes", action="store_true", help="Do not prompt before building/installing Kroko.")
    install_kroko.add_argument("--dry-run", action="store_true", help="Print Kroko installer commands without running them.")
    _add_installer_argument(install_kroko)
    install_kroko.set_defaults(func=cmd_install_kroko)

    launch = subparsers.add_parser("launch", help="Print or run the current whospeaks-window launch command.")
    launch.add_argument("--print", dest="print_only", action="store_true", help="Print the launch command and exit.")
    launch.add_argument("--dry-run", action="store_true", help="Alias for --print.")
    launch.add_argument("--language", default="", help="Temporarily override the saved language for this launch.")
    launch.add_argument("--port", type=int, default=None, help="Temporarily override the saved browser UI port.")
    launch.add_argument("--provider-preset", choices=PROVIDER_PRESET_CHOICES, default="")
    launch.add_argument("--extra-args", default="", help="Additional whospeaks-window arguments appended to the profile.")
    launch.add_argument(
        "--with-meeting-intelligence", "--with-reports",
        dest="with_reports", action="store_true",
        help="Also start Meeting Intelligence (Reports + Ask) in a new console.",
    )
    launch.add_argument(
        "--translation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Temporarily enable or disable the saved translation configuration.",
    )
    launch.add_argument("--reports-port", type=int, default=None, help="Meeting-intelligence report server port.")
    launch.add_argument("--report-language", default=None, help="Report language; defaults to the saved reports setting or live profile language.")
    launch.add_argument(
        "--report-llm-provider",
        choices=("llama_cpp", "ollama", "lm_studio", "openai_compatible", "openai", "openrouter"),
        default=None,
        help="LLM provider used by --with-meeting-intelligence (or its --with-reports compatibility alias).",
    )
    launch.add_argument("--report-llm-base-url", default=None, help="Optional report LLM base URL.")
    launch.add_argument("--report-llm-model", default=None, help="Optional report LLM model id.")
    launch.add_argument("--no-report-auto-generate", dest="report_auto_generate", action="store_false", default=None, help="Do not automatically generate reports for newly saved sessions.")
    launch.set_defaults(func=cmd_launch)

    reports = subparsers.add_parser("reports", help="Print or run Meeting Intelligence (Reports + Ask) from the saved live profile.")
    reports.add_argument("--print", dest="print_only", action="store_true", help="Print the report-server command and exit.")
    reports.add_argument("--dry-run", action="store_true", help="Alias for --print.")
    reports.add_argument("--port", dest="reports_port", type=int, default=None, help="Meeting-intelligence report server port.")
    reports.add_argument("--report-language", default=None, help="Report language; defaults to the saved reports setting or live profile language.")
    reports.add_argument(
        "--llm-provider",
        choices=("llama_cpp", "ollama", "lm_studio", "openai_compatible", "openai", "openrouter"),
        dest="report_llm_provider",
        default=None,
    )
    reports.add_argument("--llm-base-url", dest="report_llm_base_url", default=None)
    reports.add_argument("--llm-model", dest="report_llm_model", default=None)
    reports.add_argument("--no-auto-generate", dest="report_auto_generate", action="store_false", default=None, help="Do not automatically generate reports for newly saved sessions.")
    reports.set_defaults(func=cmd_reports)

    translation = subparsers.add_parser("translation", help="Print or run the optional local translation sidecar.")
    translation.add_argument("--print", dest="print_only", action="store_true")
    translation.add_argument("--dry-run", action="store_true")
    translation.add_argument("--port", dest="translation_port", type=int, default=None)
    translation.add_argument("--model-profile", dest="translation_model_profile", choices=("translate-gemma-4b", "nllb-200-600m", "madlad-400-3b"), default=None)
    translation.add_argument("--model", dest="translation_model", default=None)
    translation.add_argument("--python", dest="translation_python", default=None)
    translation.add_argument("--device", dest="translation_device", choices=("auto", "cuda", "cpu"), default=None)
    translation.set_defaults(func=cmd_translation)

    config = subparsers.add_parser("config", help="Show or update the saved starter profile.")
    config.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="Set any saved profile field.")
    config.add_argument("--reset", action="store_true", help="Reset the saved profile before applying other changes.")
    config.add_argument("--edit", action="store_true", help="Open the interactive full profile editor.")
    config.add_argument("--json", action="store_true", help="Print the profile as JSON.")
    config.add_argument("--mode", choices=("local", "remote", "server"), default=None)
    config.add_argument("--host", default=None)
    config.add_argument("--port", type=int, default=None)
    config.add_argument("--language", default=None, help="Set the saved language code, for example de.")
    config.add_argument("--model", default=None)
    config.add_argument("--device", default=None)
    config.add_argument("--compute-type", dest="compute_type", default=None)
    config.add_argument("--asr-backend", dest="asr_backend", choices=("local", "remote"), default=None)
    config.add_argument("--embeddings-backend", dest="embeddings_backend", choices=("local", "remote"), default=None)
    config.add_argument("--provider-preset", dest="provider_preset", choices=PROVIDER_PRESET_CHOICES, default=None)
    config.add_argument("--remote-asr-url", dest="remote_asr_url", default=None)
    config.add_argument("--remote-embeddings-url", dest="remote_embeddings_url", default=None)
    config.add_argument("--embedding-provider", dest="embedding_provider", default=None)
    config.add_argument("--live-speaker-embedding-provider", dest="live_speaker_embedding_provider", default=None)
    config.add_argument(
        "--live-speaker-assignment",
        dest="live_speaker_assignment",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    config.add_argument("--embedding-python", dest="embedding_python", default=None)
    config.add_argument("--vad-backend", dest="vad_backend", default=None)
    config.add_argument("--realtime-preview-engine", dest="realtime_preview_engine", default=None)
    config.add_argument("--realtime-preview-model-preset", dest="realtime_preview_model_preset", default=None)
    config.add_argument("--realtime-preview-model-dir", dest="realtime_preview_model_dir", type=Path, default=None)
    config.add_argument("--realtime-preview-python", dest="realtime_preview_python", default=None)
    config.add_argument("--reports-enabled", dest="reports_enabled", action="store_true", default=None)
    config.add_argument("--reports-port", dest="reports_port", type=int, default=None)
    config.add_argument("--report-language", dest="report_language", default=None)
    config.add_argument("--report-llm-provider", dest="report_llm_provider", choices=("llama_cpp", "ollama", "lm_studio", "openai_compatible", "openai", "openrouter"), default=None)
    config.add_argument("--report-llm-base-url", dest="report_llm_base_url", default=None)
    config.add_argument("--report-llm-model", dest="report_llm_model", default=None)
    config.add_argument("--text-embedding-base-url", dest="text_embedding_base_url", default=None)
    config.add_argument("--text-embedding-model", dest="text_embedding_model", default=None)
    config.add_argument("--text-embedding-api-key-env", dest="text_embedding_api_key_env", default=None)
    config.add_argument("--report-auto-generate", dest="report_auto_generate", action="store_true", default=None)
    config.add_argument("--translation-enabled", dest="translation_enabled", action=argparse.BooleanOptionalAction, default=None)
    config.add_argument(
        "--translation-browser-preferred",
        dest="translation_browser_preferred",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    config.add_argument(
        "--translation-provider",
        dest="translation_provider",
        choices=(
            "sidecar",
            "transformers",
            "reports_llm",
            "openai_compatible",
            "deepl",
            "google_cloud",
            "azure_translator",
            "libretranslate",
            "mock",
        ),
        default=None,
    )
    config.add_argument("--translation-port", dest="translation_port", type=int, default=None)
    config.add_argument("--translation-target-languages", dest="translation_target_languages", default=None)
    config.add_argument("--translation-max-targets", dest="translation_max_targets", type=int, default=None)
    config.add_argument("--translation-model-profile", dest="translation_model_profile", choices=("translate-gemma-4b", "nllb-200-600m", "madlad-400-3b"), default=None)
    config.add_argument("--translation-model", dest="translation_model", default=None)
    config.add_argument("--translation-base-url", dest="translation_base_url", default=None)
    config.add_argument("--translation-api-key-env", dest="translation_api_key_env", default=None)
    config.add_argument("--translation-region", dest="translation_region", default=None)
    config.add_argument("--translation-python", dest="translation_python", default=None)
    config.add_argument("--translation-device", dest="translation_device", choices=("auto", "cuda", "cpu"), default=None)
    config.add_argument("--advanced-args", dest="advanced_args", default=None)
    config.set_defaults(func=cmd_config)

    return parser
