"""Classic menus, profile editing, and launch presentation for WhoSpeaks."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from window.language_config import SUPPORTED_LANGUAGE_CONFIGS, get_language_config, normalize_language_code
from window.realtime_preview_backends import preview_language_error

from .cli_console import (
    detail_text,
    label_text,
    language_summary,
    primary_text,
    print_wrapped,
    read_input,
)
from .cli_diagnostics import (
    CheckResult,
    DoctorReport,
    print_report,
    run_doctor,
)
from .cli_installation import *  # noqa: F403 - cohesive installation API
from .planning import (
    InstallPlan,
    build_launch_command,
    build_reports_command,
    build_translation_command,
    install_plan_for_target,
    service_resource_path,
)
from .profiles import (
    EDITABLE_PROFILE_FIELDS,
    PROVIDER_PRESETS,
    Profile,
    apply_provider_preset,
    config_path,
    infer_provider_preset_id,
    load_profile,
    normalize_mode,
    save_profile,
    selected_provider_preset,
    update_profile_in_place,
)
from .runtime_constants import KROKO_LANGUAGE_MENU_CODES, STATUS_LABEL

__all__ = [
    "report_launch_values", "launch_profile_with_reports", "build_server_launch_lines",
    "print_profile", "print_provider_summary", "shorten_value", "profile_field_metadata",
    "profile_field_label", "profile_field_help", "print_launch_command",
    "full_profile_editor_text", "profile_field_names", "coerce_profile_value",
    "apply_profile_updates", "save_profile_updates", "try_save_profile_updates",
    "profile_summary_lines", "report_status_counts", "report_readiness_line", "problem_checks",
    "render_dashboard", "prompt_value", "edit_profile", "language_menu", "backend_menu",
    "asr_runtime_menu", "browser_menu", "configuration_menu_text", "configuration_menu",
    "provider_preset_menu", "select_profile_interactively",
    "install_missing_group_interactively", "install_components_interactively",
    "advanced_setup_menu", "main_menu_text", "launch_profile", "interactive_dashboard",
    "run_textual_dashboard",
]


def _facade_callable(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("whospeaks_cli.main")
    return getattr(facade, name, fallback) if facade is not None else fallback


def report_launch_values(profile: Profile, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve CLI overrides while using the saved reports configuration by default."""

    return {
        "port": args.reports_port if getattr(args, "reports_port", None) is not None else profile.reports_port,
        "report_language": getattr(args, "report_language", None) if getattr(args, "report_language", None) is not None else profile.report_language,
        "llm_provider": getattr(args, "report_llm_provider", None) or profile.report_llm_provider,
        "llm_base_url": getattr(args, "report_llm_base_url", None) if getattr(args, "report_llm_base_url", None) is not None else profile.report_llm_base_url,
        "llm_model": getattr(args, "report_llm_model", None) if getattr(args, "report_llm_model", None) is not None else profile.report_llm_model,
        "auto_generate": getattr(args, "report_auto_generate", None) if getattr(args, "report_auto_generate", None) is not None else profile.report_auto_generate,
    }


def launch_profile_with_reports(profile: Profile) -> int:
    """Start optional sidecars, then keep the live window in this terminal."""

    reports_command = build_reports_command(
        profile,
        port=profile.reports_port,
        report_language=profile.report_language,
        llm_provider=profile.report_llm_provider,
        llm_base_url=profile.report_llm_base_url,
        llm_model=profile.report_llm_model,
        auto_generate=profile.report_auto_generate,
    )
    live_command = build_launch_command(profile)
    print("Meeting Intelligence — Reports + Ask command:")
    print(format_command(reports_command))
    print("Live window command:")
    print(format_command(live_command))
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(reports_command, **popen_kwargs)
    if profile.translation_enabled and profile.translation_provider == "sidecar":
        translation_command = build_translation_command(profile)
        print("Translation sidecar command:")
        print(format_command(translation_command))
        subprocess.Popen(translation_command, **popen_kwargs)
    return int(subprocess.run(live_command, check=False).returncode)


def build_server_launch_lines() -> list[str]:
    asr_dir = service_resource_path("faster-whisper-asr", "asr_server.py").parent
    embeddings_dir = service_resource_path("voice-embeddings-server", "embeddings_server.py").parent

    def line(directory: Path, app: str, port: int) -> str:
        command = format_command([sys.executable, "-m", "uvicorn", app, "--host", "0.0.0.0", "--port", str(port)])
        if os.name == "nt":
            return f'cd /d "{directory}" && {command}'
        return f"cd {shlex.quote(str(directory))} && {command}"

    return [
        line(asr_dir, "asr_server:app", 8650),
        line(embeddings_dir, "embeddings_server:app", 8660),
    ]


def print_profile(profile: Profile) -> None:
    print("Current starter profile")
    print("=" * 72)
    print(f"Saved config: {config_path()}")
    print("-" * 72)
    for key, label, _help_text in EDITABLE_PROFILE_FIELDS:
        print(f"{label:<28} {getattr(profile, key)}")
    print("=" * 72)
    print_launch_command(profile)


def print_provider_summary(profile: Profile, indent: str = "") -> None:
    preset = selected_provider_preset(profile)
    if preset is None:
        print(f"{indent}Provider preset: {primary_text('Custom')}")
        print(f"{indent}  {label_text('Simple:')} {primary_text('Manual provider strings. The launcher will use the exact values below.')}")
    else:
        print(f"{indent}Provider preset: {primary_text(preset.name)} ({preset.id})")
        print(f"{indent}  {label_text('Simple:')} {primary_text(preset.summary)}")
        print_wrapped(
            f"Deep: {preset.details}",
            initial_indent=f"{indent}  ",
            subsequent_indent=f"{indent}        ",
            style=detail_text,
        )
        if preset.score_note:
            print_wrapped(
                f"Validation note: {preset.score_note}",
                initial_indent=f"{indent}  ",
                subsequent_indent=f"{indent}        ",
                style=detail_text,
            )
        if preset.requirements:
            print_wrapped(
                f"Requirement: {preset.requirements}",
                initial_indent=f"{indent}  ",
                subsequent_indent=f"{indent}        ",
                style=detail_text,
            )
    print_wrapped(
        "Exact final provider: " + str(profile.embedding_provider),
        initial_indent=f"{indent}  ",
        subsequent_indent=f"{indent}        ",
        style=detail_text,
    )
    print_wrapped(
        "Exact live provider:  " + str(profile.live_speaker_embedding_provider),
        initial_indent=f"{indent}  ",
        subsequent_indent=f"{indent}        ",
        style=detail_text,
    )


def shorten_value(value: Any, width: int = 58) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def profile_field_metadata() -> dict[str, tuple[str, str]]:
    return {key: (label, help_text) for key, label, help_text in EDITABLE_PROFILE_FIELDS}


def profile_field_label(key: str) -> str:
    return profile_field_metadata().get(key, (key.replace("_", " ").title(), ""))[0]


def profile_field_help(key: str) -> str:
    return profile_field_metadata().get(key, ("", ""))[1]


def print_launch_command(profile: Profile, extra_args: str = "") -> None:
    if profile.mode == "server":
        print("Server profile service commands:")
        for line in build_server_launch_lines():
            print(f"  {line}")
        return
    print("Launch command:")
    command = format_command(build_launch_command(profile, extra_args))
    print_wrapped(command, width=100, initial_indent="  ", subsequent_indent="  ", style=primary_text)


def full_profile_editor_text(profile: Profile) -> str:
    lines = [
        "All Saved Profile Fields",
        "=" * 72,
        "Choose one field to edit. Press Enter at a prompt to keep the current value.",
        "-" * 72,
    ]
    for index, (key, label, help_text) in enumerate(EDITABLE_PROFILE_FIELDS, start=1):
        value = shorten_value(getattr(profile, key), 46)
        lines.append(f"{index:>2}. {label:<25} {value}")
        lines.append(detail_text(f"    {help_text}"))
    lines.append("b. Back")
    return "\n".join(lines)


def profile_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(Profile)}


def coerce_profile_value(profile: Profile, key: str, value: str) -> Any:
    current = getattr(profile, key)
    if key == "language":
        return normalize_language_code(value)
    if key == "report_language":
        return normalize_language_code(value) if str(value).strip() else ""
    if key == "translation_target_languages":
        targets: list[str] = []
        for raw_target in re.split(r"[,;\s]+", str(value or "")):
            if not raw_target:
                continue
            target = normalize_language_code(raw_target)
            if target != profile.language and target not in targets:
                targets.append(target)
        return ",".join(targets)
    if isinstance(current, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    return value


def apply_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    candidate = Profile.from_mapping(profile.as_dict())
    explicit_provider_preset: str | None = None
    validation_updates: dict[str, object] = {}
    fields = profile_field_names()
    for raw_key, raw_value in updates:
        key = str(raw_key).strip().replace("-", "_")
        value = str(raw_value)
        if key not in fields:
            allowed = ", ".join(sorted(fields))
            raise SystemExit(f"Unknown profile field {key!r}. Known fields: {allowed}.")
        if key == "mode":
            validation_updates[key] = value
            candidate = configure_profile_for_mode(candidate, value)
            continue
        if key == "provider_preset":
            explicit_provider_preset = value
            validation_updates[key] = value
            continue
        try:
            coerced = coerce_profile_value(candidate, key, value)
            validation_updates[key] = coerced
            candidate = candidate.with_updates(
                **{key: coerced}
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    from .launcher_controller import LauncherController, ProfileValidationError

    try:
        LauncherController(profile).validate_profile_updates(validation_updates)
    except ProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc
    candidate = Profile.from_mapping(candidate.as_dict())
    if explicit_provider_preset is not None:
        candidate = apply_provider_preset(candidate, explicit_provider_preset)
    else:
        candidate = candidate.with_updates(
            provider_preset=infer_provider_preset_id(
                candidate.provider_preset,
                candidate.embedding_provider,
                candidate.live_speaker_embedding_provider,
            )
        )
    return candidate




def save_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    updated = apply_profile_updates(profile, updates)
    save_path = save_profile(updated)
    update_profile_in_place(profile, updated)
    changed = ", ".join(f"{key}={getattr(updated, str(key).replace('-', '_'))}" for key, _value in updates)
    print(f"Saved {changed} to {save_path}.")
    return profile


def try_save_profile_updates(profile: Profile, updates: list[tuple[str, Any]]) -> Profile:
    try:
        return save_profile_updates(profile, updates)
    except SystemExit as exc:
        print(str(exc))
        return profile




def profile_summary_lines(profile: Profile) -> list[str]:
    preset = selected_provider_preset(profile)
    provider = f"{preset.name} ({preset.id})" if preset is not None else "Custom"
    return [
        f"Mode: {profile.mode}    ASR: {profile.asr_backend}    Embeddings: {profile.embeddings_backend}",
        f"Language: {language_summary(profile.language)}",
        f"Provider: {provider}",
        f"Final provider: {profile.embedding_provider}",
        f"Live provider:  {profile.live_speaker_embedding_provider}",
        f"Embedding Python: {profile.embedding_python or 'auto/current'}",
        f"ASR model/device: {profile.model} / {profile.device} / {profile.compute_type}",
        f"Realtime text: {profile.realtime_preview_engine} / {profile.realtime_preview_model_preset or 'default'}",
        f"Realtime model folder: {profile.realtime_preview_model_dir or 'automatic'}",
        f"Realtime Python: {profile.realtime_preview_python or 'auto/default'}",
        f"Browser: {profile.host}:{profile.port}",
    ]


def report_status_counts(report: DoctorReport) -> dict[str, int]:
    return {
        status: sum(1 for check in report.checks if check.status == status)
        for status in ("fail", "warn", "skip", "ok")
    }


def report_readiness_line(report: DoctorReport) -> str:
    counts = report_status_counts(report)
    if counts["fail"]:
        state = f"Action needed: {counts['fail']} failed check"
        if counts["fail"] != 1:
            state += "s"
    elif counts["warn"]:
        state = f"Usable with {counts['warn']} warning"
        if counts["warn"] != 1:
            state += "s"
    else:
        state = "Ready: no failed or warning checks"
    if counts["skip"]:
        state += f"; {counts['skip']} skipped"
    return state


def problem_checks(report: DoctorReport) -> list[CheckResult]:
    return [check for check in report.checks if check.is_problem()]


def render_dashboard(profile: Profile, report: DoctorReport) -> None:
    print()
    print("WhoSpeaks")
    print("=" * 72)
    print(f"Profile: {primary_text(profile.mode)}  ASR: {profile.asr_backend}  Embeddings: {profile.embeddings_backend}")
    print(f"Browser: {profile.host}:{profile.port}  Language: {language_summary(profile.language)}")
    if profile.embeddings_backend == "local":
        print(f"Embedding Python: {profile.embedding_python or 'auto/current'}")
    print(f"Realtime text: {profile.realtime_preview_engine}  Python: {profile.realtime_preview_python or 'auto/default'}")
    print_provider_summary(profile)
    if profile.embeddings_backend == "remote":
        print(f"Embeddings URL: {profile.remote_embeddings_url}")
    if profile.asr_backend == "remote":
        print(f"ASR URL: {profile.remote_asr_url}")
    print("-" * 72)
    print(f"Readiness: {report_readiness_line(report)}")
    problems = problem_checks(report)
    if problems:
        width = max([len(check.name) for check in problems] + [10])
        for check in problems:
            print(f"{STATUS_LABEL[check.status]:<5} {check.name:<{width}} {check.detail}")
            if check.remediation:
                print_wrapped(
                    "Fix: " + check.remediation,
                    initial_indent=f"{'':<6}{'':<{width}} ",
                    subsequent_indent=f"{'':<6}{'':<{width}} ",
                    style=detail_text,
                )
        print(detail_text("Run doctor for the complete component list."))
    else:
        print(primary_text("No actionable setup problems detected by the quick doctor pass."))
    print("-" * 72)
    print("Direct controls: language, realtime text, providers, backend URLs, ASR runtime, browser port.")
    print("Validation controls: doctor, install recommendation, print exact launch command, launch browser UI.")
    print("=" * 72)




def prompt_value(label: str, current: Any) -> str:
    value = read_input(f"{label} [{current}]: ").strip()
    return value if value else str(current)


def edit_profile(profile: Profile) -> Profile:
    while True:
        print()
        print(full_profile_editor_text(profile))
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return profile
        try:
            selected = int(choice)
        except ValueError:
            print("Choose one field number or b.")
            continue
        fields = list(EDITABLE_PROFILE_FIELDS)
        if not 1 <= selected <= len(fields):
            print("Choose one of the listed field numbers.")
            continue
        key, label, help_text = fields[selected - 1]
        print_wrapped(help_text, initial_indent="  ", subsequent_indent="  ", style=detail_text)
        value = prompt_value(label, getattr(profile, key))
        try_save_profile_updates(profile, [(key, value)])
    return profile


def language_menu(profile: Profile) -> None:
    while True:
        print()
        print("Language And Realtime Text")
        print("=" * 72)
        print(f"Current: {language_summary(profile.language)}")
        print(f"Realtime preview engine: {profile.realtime_preview_engine}")
        print(f"Realtime preview Python: {profile.realtime_preview_python or 'auto/default'}")
        print("-" * 72)
        for index, code in enumerate(KROKO_LANGUAGE_MENU_CODES, start=1):
            config = SUPPORTED_LANGUAGE_CONFIGS[code]
            marker = " *" if profile.language == code else ""
            print(f"{index}. {config.display_name:<12} {config.code:<3} Kroko {config.kroko_code}{marker}")
        print("c. Custom language code")
        print("o. Turn realtime text off")
        print("e. Realtime text engine")
        print("p. Realtime preview Python")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice in {"o", "off"}:
            try_save_profile_updates(profile, [("realtime_preview_engine", "off")])
            continue
        if choice in {"e", "engine"}:
            print("1. Nemotron 3.5")
            print("2. Kroko / Banafo")
            print("3. Off")
            engine_choice = read_input("Realtime text engine> ", "1").strip().lower()
            engine = {
                "1": "sherpa_onnx",
                "nemotron": "sherpa_onnx",
                "2": "kroko_onnx",
                "kroko": "kroko_onnx",
                "3": "off",
                "off": "off",
            }.get(engine_choice)
            if engine is None:
                print("Choose 1, 2, or 3.")
                continue
            try_save_profile_updates(profile, [("realtime_preview_engine", engine)])
            continue
        if choice in {"p", "python"}:
            try_save_profile_updates(
                profile,
                [("realtime_preview_python", prompt_value("Realtime preview Python path", profile.realtime_preview_python))],
            )
            continue
        if choice in {"c", "custom"}:
            value = prompt_value("Language code or name", profile.language)
            try:
                normalized = normalize_language_code(value)
            except ValueError as exc:
                print(str(exc))
                continue
            updates: list[tuple[str, Any]] = [("language", normalized)]
            if not get_language_config(normalized).kroko_code and profile.realtime_preview_engine != "off":
                answer = read_input("This language has no Kroko live-text model. Turn realtime text off? [Y/n] ", "y").strip().lower()
                if answer not in {"n", "no"}:
                    updates.append(("realtime_preview_engine", "off"))
            try_save_profile_updates(profile, updates)
            continue
        try:
            selected = int(choice)
        except ValueError:
            print("Choose a language number, c, o, or b.")
            continue
        if not 1 <= selected <= len(KROKO_LANGUAGE_MENU_CODES):
            print("Choose one of the listed language numbers.")
            continue
        code = KROKO_LANGUAGE_MENU_CODES[selected - 1]
        updates = [("language", code)]
        if profile.realtime_preview_engine in {"", "off", "none", "false"}:
            answer = read_input("Enable Kroko realtime text for this language? [Y/n] ", "y").strip().lower()
            if answer not in {"n", "no"}:
                updates.append(("realtime_preview_engine", "kroko_onnx"))
        try_save_profile_updates(profile, updates)


def backend_menu(profile: Profile) -> None:
    while True:
        print()
        print("Backends And URLs")
        print("=" * 72)
        print(f"Mode: {profile.mode}")
        print(f"ASR: {profile.asr_backend}    {profile.remote_asr_url}")
        print(f"Embeddings: {profile.embeddings_backend}    {profile.remote_embeddings_url}")
        print(f"Embedding helper Python: {profile.embedding_python or 'auto/current'}")
        print("-" * 72)
        print("1. Full local ASR and embeddings")
        print("2. Controller with remote ASR and embeddings")
        print("3. Remote ASR URL")
        print("4. Remote embeddings URL")
        print("5. Embedding helper Python")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            try_save_profile_updates(
                profile,
                [
                    ("mode", "local"),
                    ("asr_backend", "local"),
                    ("embeddings_backend", "local"),
                    ("device", "auto"),
                ],
            )
        elif choice == "2":
            try_save_profile_updates(
                profile,
                [
                    ("mode", "remote"),
                    ("asr_backend", "remote"),
                    ("embeddings_backend", "remote"),
                    ("device", "auto"),
                ],
            )
        elif choice == "3":
            try_save_profile_updates(profile, [("remote_asr_url", prompt_value("Remote ASR URL", profile.remote_asr_url))])
        elif choice == "4":
            try_save_profile_updates(
                profile,
                [("remote_embeddings_url", prompt_value("Remote embeddings URL", profile.remote_embeddings_url))],
            )
        elif choice == "5":
            try_save_profile_updates(
                profile,
                [("embedding_python", prompt_value("Embedding helper Python path", profile.embedding_python))],
            )
        else:
            print("Choose one of the listed options.")


def asr_runtime_menu(profile: Profile) -> None:
    while True:
        print()
        print("ASR Runtime")
        print("=" * 72)
        print(f"Model: {profile.model}")
        print(f"Device: {profile.device}")
        print(f"Compute type: {profile.compute_type}")
        print(f"Voice activity detector: {profile.vad_backend}")
        print("-" * 72)
        print("1. ASR model")
        print("2. Device")
        print("3. Compute type")
        print("4. Voice activity detector")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            try_save_profile_updates(profile, [("model", prompt_value("ASR model", profile.model))])
        elif choice == "2":
            print("1. Auto")
            print("2. CUDA")
            print("3. CPU")
            device_choice = read_input("Device> ", "1").strip().lower()
            device = {
                "1": "auto",
                "auto": "auto",
                "2": "cuda",
                "cuda": "cuda",
                "3": "cpu",
                "cpu": "cpu",
            }.get(device_choice)
            if device is None:
                print("Choose 1, 2, or 3.")
                continue
            try_save_profile_updates(profile, [("device", device)])
        elif choice == "3":
            try_save_profile_updates(profile, [("compute_type", prompt_value("Compute type", profile.compute_type))])
        elif choice == "4":
            print("1. RMS (lightweight)")
            print("2. Silero")
            vad_choice = read_input("Voice activity detector> ", "1").strip().lower()
            vad = {
                "1": "rms",
                "rms": "rms",
                "2": "silero",
                "silero": "silero",
            }.get(vad_choice)
            if vad is None:
                print("Choose 1 or 2.")
                continue
            try_save_profile_updates(profile, [("vad_backend", vad)])
        else:
            print("Choose one of the listed options.")


def browser_menu(profile: Profile) -> None:
    while True:
        print()
        print("Browser UI")
        print("=" * 72)
        print(f"Host: {profile.host}")
        print(f"Port: {profile.port}")
        print("-" * 72)
        print("1. Host")
        print("2. Port")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice == "1":
            try_save_profile_updates(profile, [("host", prompt_value("Host", profile.host))])
        elif choice == "2":
            try_save_profile_updates(profile, [("port", prompt_value("Port", profile.port))])
        else:
            print("Choose one of the listed options.")


def configuration_menu_text(profile: Profile) -> str:
    summary = "\n".join(f"  {line}" for line in profile_summary_lines(profile))
    return textwrap.dedent(
        f"""
        Configure WhoSpeaks
        ============================================================================
        {summary}
        ----------------------------------------------------------------------------
          1. Language and realtime text
          2. Speaker provider quality
          3. Backends and remote URLs
          4. ASR model, device, and compute
          5. Browser host and port
          6. Advanced launch arguments
          7. All saved profile fields
          b. Back
        """
    ).strip()


def configuration_menu(profile: Profile) -> int | None:
    while True:
        print()
        print(configuration_menu_text(profile))
        choice = read_input("> ", "b").strip().lower()
        if choice == "1":
            language_menu(profile)
        elif choice == "2":
            provider_preset_menu(profile)
        elif choice == "3":
            backend_menu(profile)
        elif choice == "4":
            asr_runtime_menu(profile)
        elif choice == "5":
            browser_menu(profile)
        elif choice == "6":
            try_save_profile_updates(profile, [("advanced_args", prompt_value("Advanced whospeaks-window args", profile.advanced_args))])
        elif choice == "7":
            updated = edit_profile(profile)
            update_profile_in_place(profile, Profile.from_mapping(updated.as_dict()))
            save_profile(profile)
        elif choice in {"b", "back", "q", "quit"}:
            return None
        else:
            print("Choose one of the listed options.")


def provider_preset_menu(profile: Profile) -> None:
    while True:
        selected_provider_preset(profile)
        print()
        print("Provider Presets")
        print("=" * 72)
        print_provider_summary(profile)
        print("-" * 72)
        for index, preset in enumerate(PROVIDER_PRESETS.values(), start=1):
            marker = " *" if profile.provider_preset == preset.id else ""
            print(f"{index}. {primary_text(preset.name)} ({preset.id}){marker}")
            print(f"   {label_text('Simple:')} {primary_text(preset.summary)}")
            print_wrapped(
                f"Deep: {preset.details}",
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
            if preset.score_note:
                print_wrapped(
                    f"Validation note: {preset.score_note}",
                    initial_indent="   ",
                    subsequent_indent="         ",
                    style=detail_text,
                )
            if preset.requirements:
                print_wrapped(
                    f"Requirement: {preset.requirements}",
                    initial_indent="   ",
                    subsequent_indent="         ",
                    style=detail_text,
                )
            print_wrapped(
                "Exact final: " + preset.embedding_provider,
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
            print_wrapped(
                "Exact live:  " + preset.live_speaker_embedding_provider,
                initial_indent="   ",
                subsequent_indent="         ",
                style=detail_text,
            )
        print("c. Custom provider strings")
        print("b. Back")
        choice = read_input("> ", "b").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return
        if choice in {"c", "custom"}:
            final_provider = prompt_value(
                "Final embedding provider", profile.embedding_provider
            )
            live_provider = prompt_value(
                "Live embedding provider",
                profile.live_speaker_embedding_provider,
            )
            try_save_profile_updates(
                profile,
                [
                    ("provider_preset", "custom"),
                    ("embedding_provider", final_provider),
                    ("live_speaker_embedding_provider", live_provider),
                ],
            )
            continue
        try:
            selected = int(choice)
        except ValueError:
            print("Choose a preset number, c, or b.")
            continue
        presets = list(PROVIDER_PRESETS.values())
        if not 1 <= selected <= len(presets):
            print("Choose one of the listed preset numbers.")
            continue
        preset = presets[selected - 1]
        try_save_profile_updates(profile, [("provider_preset", preset.id)])


def select_profile_interactively(profile: Profile, mode: str) -> int | None:
    configure_profile_for_mode(profile, mode)
    save_path = save_profile(profile)
    label = {
        "local": "full local installation",
        "remote": "controller + remote GPU services",
        "server": "GPU server",
    }.get(profile.mode, profile.mode)
    print(f"Selected {label} profile and saved it to {save_path}.")
    report = run_doctor(profile)
    extra = recommended_install_extra(profile, report)
    if extra is None:
        print("No Python package install action is missing for this profile.")
        return None
    installer_backend = normalize_installer_backend(None)
    print(
        "Next installer action: "
        f"{format_command(build_install_command(extra, installer_backend=installer_backend))}"
    )
    answer = read_input("Install the required Python packages now? [y/N] ", "n").strip().lower()
    if answer in {"y", "yes"}:
        installer_backend = prompt_installer_backend()
        return install_extra_and_maybe_kroko(
            profile,
            extra,
            assume_yes=True,
            kroko_assume_yes=False,
            installer_backend=installer_backend,
        )
    print("Install skipped. Choose the install action later to run it.")
    return None


def install_missing_group_interactively(profile: Profile, report: DoctorReport | None = None) -> int | None:
    current_report = report or run_doctor(profile)
    extra = recommended_install_extra(profile, current_report)
    if extra is None:
        print("No Python package install action is missing for this profile.")
        return None
    return install_extra_and_maybe_kroko(
        profile,
        extra,
        installer_backend=prompt_installer_backend(),
    )


def install_components_interactively(profile: Profile) -> int | None:
    target = prompt_install_target()
    preview_engine, preview_preset = prompt_realtime_preview(target)
    installer_backend = prompt_installer_backend()
    plan = install_plan_for_target(
        target,
        realtime_preview_engine=preview_engine,
        realtime_preview_model_preset=preview_preset,
    )
    configure_profile_for_install(profile, plan)
    validate_realtime_preview_language(profile)
    save_path = save_profile(profile)
    print(f"Saved {profile.mode} profile to {save_path}")
    print_install_plan(plan, profile, installer_backend=installer_backend)
    if not confirm_install_start(False, False):
        print("Install skipped.")
        return None
    return install_extra_and_maybe_kroko(
        profile,
        plan.extra,
        assume_yes=True,
        install_kroko=plan.install_kroko,
        kroko_assume_yes=True if plan.install_kroko else False,
        installer_backend=installer_backend,
    )


def advanced_setup_menu(profile: Profile) -> int | None:
    while True:
        print(textwrap.dedent(
            """
            Advanced setup
              1. Controller + remote GPU services profile
              2. This machine as a GPU server profile
              3. Edit profile settings
              4. Print exact launch command
              b. Back
            """
        ).strip())
        choice = read_input("> ", "b").strip().lower()
        if choice == "1":
            code = select_profile_interactively(profile, "remote")
            if code:
                return code
        elif choice == "2":
            code = select_profile_interactively(profile, "server")
            if code:
                return code
        elif choice == "3":
            edit_profile(profile)
            save_profile(profile)
        elif choice == "4":
            print_launch_command(profile)
        elif choice in {"b", "back", "q", "quit"}:
            return None


def main_menu_text() -> str:
    return textwrap.dedent(
        """
        Actions
          1. Install or repair WhoSpeaks
          2. Launch browser UI
          3. Doctor / complete diagnostics
          4. Language and realtime text
          5. Speaker provider quality
          6. Backends and remote URLs
          7. ASR model, device, and compute
          8. Browser host and port
          9. All configuration fields
          p. Print exact launch command
          r. Remote/server profiles
          q. Quit
        """
    ).strip()


def launch_profile(profile: Profile) -> int:
    if profile.mode == "server":
        print("Start each server command in its own shell:")
        for line in build_server_launch_lines():
            print(f"  {line}")
        return 0
    command = build_launch_command(profile)
    if profile.translation_enabled and profile.translation_provider == "sidecar":
        translation_command = build_translation_command(profile)
        print(f"Translation sidecar: {format_command(translation_command)}")
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(translation_command, **popen_kwargs)
    print(format_command(command))
    return int(subprocess.run(command, check=False).returncode)


def interactive_dashboard(profile: Profile) -> int:
    while True:
        report = _facade_callable("run_doctor", run_doctor)(profile)
        _facade_callable("render_dashboard", render_dashboard)(profile, report)
        extra = _facade_callable("recommended_install_extra", recommended_install_extra)(profile, report)
        if extra:
            installer_backend = normalize_installer_backend(None)
            print(
                "Recommended package action: "
                f"{format_command(build_install_command(extra, installer_backend=installer_backend))}"
            )
        print(main_menu_text())
        choice = _facade_callable("read_input", read_input)("> ", "q").strip().lower()
        if choice in {"1", "i", "install", "s", "setup"}:
            code = _facade_callable(
                "install_components_interactively", install_components_interactively
            )(profile)
            if code:
                return code
        elif choice == "2":
            return launch_profile(profile)
        elif choice == "3":
            print_report(_facade_callable("run_doctor", run_doctor)(profile, deep=True))
        elif choice == "4":
            language_menu(profile)
        elif choice == "5":
            provider_preset_menu(profile)
        elif choice == "6":
            backend_menu(profile)
        elif choice == "7":
            asr_runtime_menu(profile)
        elif choice == "8":
            browser_menu(profile)
        elif choice == "9":
            configuration_menu(profile)
        elif choice in {"p", "print"}:
            print_launch_command(profile)
        elif choice in {"r", "remote", "server"}:
            code = advanced_setup_menu(profile)
            if code:
                return code
        elif choice in {"q", "quit", "exit"}:
            return 0


def run_textual_dashboard(profile: Profile) -> int:
    try:
        from .tui import run_setup_app
    except ImportError as exc:
        if not str(exc.name or "").startswith("textual"):
            raise
        print("Textual is unavailable; opening the classic terminal interface.")
        return interactive_dashboard(profile)

    result = run_setup_app(profile)
    if result == "launch":
        return launch_profile(load_profile())
    if result == "launch_with_reports":
        return launch_profile_with_reports(load_profile())
    return 0
