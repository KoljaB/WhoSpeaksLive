"""Lightweight setup, doctor, and launcher CLI for WhoSpeaks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

from . import __version__
from .profiles import (
    DEFAULT_REMOTE_ASR_URL,
    DEFAULT_MACOS_ASR_URL,
    DEFAULT_REMOTE_EMBEDDINGS_URL,
    EDITABLE_PROFILE_FIELDS,
    FAST_LIVE_PROVIDER,
    PROMOTED_LIVE_PROVIDER,
    PROMOTED_PUBLIC_PROVIDER,
    PROVIDER_PRESET_CHOICES,
    PROVIDER_PRESETS,
    PUBLIC_PROVIDER,
    SINGLE_ESPNET_PROVIDER,
    SMOKE_PROVIDER,
    Profile,
    ProfileLoadError,
    ProviderPreset,
    apply_provider_preset,
    config_path,
    config_read_candidates,
    infer_provider_preset_id,
    load_profile,
    local_config_path,
    normalize_mode,
    normalize_provider_preset_id,
    profile_with_provider_preset,
    save_profile,
    selected_provider_preset,
    update_profile_in_place,
)
from .planning import (
    COMPLETE_EXTRA,
    CONTROLLER_EXTRA,
    INSTALL_TARGET_CHOICES,
    LOCAL_EXTRA,
    PREVIEW_EXTRA,
    SERVER_EXTRA,
    TRANSLATION_INSTALL_PROFILE_CHOICES,
    InstallPlan,
    LaunchPlan,
    ServiceProcessSpec,
    build_launch_command,
    build_launch_plan,
    build_macos_service_specs,
    build_reports_command,
    build_translation_command,
    install_plan_for_target,
    normalize_install_target,
    profile_for_install,
    profile_for_mode,
    default_macos_runtime_root,
    health_payload_matches,
    require_apple_silicon_macos,
    service_resource_path,
)
from .cli_console import (
    color_enabled,
    detail_text,
    label_text,
    language_summary,
    primary_text,
    print_wrapped,
    read_input,
    style_text,
    wrap_styled_lines,
)
from .runtime_constants import (
    DEFAULT_PYTORCH_CUDA_BUILD, INSTALLER_BACKEND_CHOICES, INSTALLER_BACKEND_ENV,
    KROKO_INSTALL_MODULE, KROKO_LANGUAGE_MENU_CODES,
    KROKO_PREVIEW_VENV_ENV, PACKAGE_NAME, PIP_EXTRA_INDEX_URL_ENV, PIP_FIND_LINKS_ENV,
    PIP_INDEX_URL_ENV, PYPI_SIMPLE_URL, PYTORCH_CPU_INDEX_URL, PYTORCH_CPU_INDEX_URL_ENV, PYTORCH_CUDA_BUILD_ENV,
    PYTORCH_CUDA_INDEX_URLS, PYTORCH_CUDA_INDEX_URL_ENV, STATUS_LABEL, STATUS_ORDER,
    TESTPYPI_SIMPLE_URL, TORCH_INSTALL_POLICY_CHOICES, TORCH_INSTALL_POLICY_ENV,
    TORCH_PACKAGE_SPECS, TRANSLATION_MODEL_ROOT_ENV, TRANSLATION_VENV_ROOT_ENV,
)
from .cli_diagnostics import (
    CheckResult,
    DoctorReport,
    TorchInstallSelection,
    check_embedding_cache,
    check_faster_whisper_cache,
    check_import_group,
    check_local_provider_syntax,
    check_macos_audio_capture,
    check_macos_mps,
    check_macos_service_runtime,
    check_port,
    check_python_imports,
    check_remote_health,
    check_remote_provider_load,
    check_remote_providers,
    check_sherpa_onnx_runtime,
    command_version,
    detect_torch_cuda,
    installed_distribution_version,
    module_available,
    post_json_url,
    print_report,
    read_json_url,
    report_to_dict,
    run_doctor,
    runtime_cache_dir,
    subprocess_pythonpath_entries,
)
from .cli_installation import *  # noqa: F403 - compatibility facade
from .cli_classic import *  # noqa: F403 - compatibility facade
from .cli_commands import *  # noqa: F403 - compatibility facade
from .service_processes import (
    service_health_ready,
    start_service_process,
    terminate_service_processes,
    wait_for_service_health,
)
from .cli_parser import build_parser
from window.language_config import SUPPORTED_LANGUAGE_CONFIGS, get_language_config, normalize_language_code
from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
    recommended_preview_engine,
)
from window.sherpa_onnx_models import (
    default_sherpa_onnx_model_dir,
    missing_sherpa_onnx_model_files,
)


def desktop_session_available() -> bool:
    return platform.system() in {"Windows", "Darwin"} or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def run_desktop_dashboard() -> int:
    try:
        from whospeaks_gui.main import main as run_desktop_gui
    except ModuleNotFoundError as exc:
        if exc.name not in {"PySide6", "whospeaks_gui"}:
            raise
        raise RuntimeError(
            "The WhoSpeaks desktop launcher is missing from this installation. "
            "Reinstall the base package with `pip install --force-reinstall whospeaks`."
        ) from exc

    return int(run_desktop_gui([]))

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        try:
            return int(args.func(args))
        except ProfileLoadError as exc:
            parser.error(str(exc))
    desktop_session = desktop_session_available()
    if args.gui or (
        desktop_session
        and sys.stdin.isatty()
        and not args.no_interactive
    ):
        if not desktop_session:
            parser.error(
                "No graphical desktop session is available. Run a command such as "
                "`whospeaks doctor`, `whospeaks install`, or `whospeaks launch` instead."
            )
        try:
            return run_desktop_dashboard()
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        profile = load_profile()
    except ProfileLoadError as exc:
        parser.error(str(exc))
    if args.no_interactive or not sys.stdin.isatty() or not desktop_session:
        report = run_doctor(profile)
        render_dashboard(profile, report)
        if desktop_session:
            print("Run `whospeaks` in an interactive terminal to open the desktop launcher.")
        else:
            print("No graphical desktop session was detected; use the scriptable subcommands below.")
        print("For automation, run `whospeaks install --target local --without-kroko --yes`.")
        print("Run `whospeaks launch --print` to see the exact browser command.")
        return 0
    return 0
