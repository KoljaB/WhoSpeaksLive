"""Shared text rendering and launch helpers for the WhoSpeaks command line."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from window.language_config import normalize_language_code

from .cli_console import (
    detail_text,
    label_text,
    language_summary,
    primary_text,
    print_wrapped,
)
from .cli_diagnostics import (
    CheckResult,
    DoctorReport,
)
from .cli_installation import *  # noqa: F403 - cohesive installation API
from .planning import (
    build_launch_command,
    build_reports_command,
    build_translation_command,
    service_resource_path,
)
from .profiles import (
    EDITABLE_PROFILE_FIELDS,
    Profile,
    apply_provider_preset,
    config_path,
    infer_provider_preset_id,
    save_profile,
    selected_provider_preset,
    update_profile_in_place,
)
from .runtime_constants import STATUS_LABEL

__all__ = [
    "report_launch_values", "launch_profile_with_reports", "build_server_launch_lines",
    "print_profile", "print_provider_summary", "shorten_value", "profile_field_metadata",
    "profile_field_label", "profile_field_help", "print_launch_command",
    "full_profile_editor_text", "profile_field_names", "coerce_profile_value",
    "apply_profile_updates", "save_profile_updates", "try_save_profile_updates",
    "profile_summary_lines", "report_status_counts", "report_readiness_line", "problem_checks",
    "render_dashboard",
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
