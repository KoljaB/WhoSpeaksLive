"""Command handlers for the WhoSpeaks CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
)

from .cli_classic import *  # noqa: F403 - command-facing classic API
from .cli_diagnostics import DoctorReport, print_report, report_to_dict, run_doctor
from .cli_installation import *  # noqa: F403 - command-facing installation API
from .planning import (
    build_launch_plan,
    build_launch_command,
    build_reports_command,
    build_translation_command,
    install_plan_for_target,
    normalize_install_target,
    require_apple_silicon_macos,
)
from .service_processes import (
    service_health_ready,
    start_service_process,
    terminate_service_processes,
    wait_for_service_health,
)
from .profiles import Profile, apply_provider_preset, config_path, load_profile, save_profile

__all__ = [
    "cmd_install", "cmd_install_translation", "cmd_doctor", "cmd_setup",
    "cmd_install_kroko", "cmd_launch", "cmd_reports", "cmd_translation", "cmd_config",
]


def _facade_callable(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("whospeaks_cli.main")
    return getattr(facade, name, fallback) if facade is not None else fallback


def _load_profile() -> Profile:
    return _facade_callable("load_profile", load_profile)()


def _run_doctor(*args: Any, **kwargs: Any) -> DoctorReport:
    return _facade_callable("run_doctor", run_doctor)(*args, **kwargs)


def _save_profile(profile: Profile) -> Any:
    return _facade_callable("save_profile", save_profile)(profile)


def cmd_install(args: argparse.Namespace) -> int:
    if args.with_kroko and args.without_kroko:
        raise SystemExit("Choose either --with-kroko or --without-kroko, not both.")

    if args.target:
        target = normalize_install_target(args.target)
    elif sys.stdin.isatty():
        target = prompt_install_target()
    else:
        target = "local"
        print("No install target was supplied; defaulting to full local installation.")

    requested_engine = str(getattr(args, "realtime_preview_engine", "") or "").strip()
    requested_preset = str(getattr(args, "realtime_preview_model_preset", "") or "").strip()
    requested_model_dir = getattr(args, "realtime_preview_model_dir", None)
    if requested_engine and (args.with_kroko or args.without_kroko):
        raise SystemExit("Choose --realtime-preview-engine or the legacy Kroko switches, not both.")
    if requested_preset and not requested_engine:
        raise SystemExit("--realtime-preview-model-preset requires --realtime-preview-engine.")
    if requested_model_dir is not None and requested_engine not in {"sherpa_onnx", "sherpa-onnx", "nemotron", "sherpa"}:
        raise SystemExit("--realtime-preview-model-dir is only valid with --realtime-preview-engine sherpa_onnx.")

    if target == "server":
        preview_engine, preview_preset = "off", ""
        if requested_engine or args.with_kroko:
            print("Realtime preview text is not installed on server-only targets.")
    elif requested_engine:
        preview_engine, preview_preset = requested_engine, requested_preset
    elif args.with_kroko:
        preview_engine, preview_preset = "kroko_onnx", "community-64l"
    elif args.without_kroko:
        preview_engine, preview_preset = "off", ""
    elif sys.stdin.isatty() and not args.yes:
        preview_engine, preview_preset = prompt_realtime_preview(target)
    else:
        preview_engine, preview_preset = "off", ""
        print("Realtime preview text is not selected. Pass --realtime-preview-engine sherpa_onnx to include Nemotron.")

    requested_translation = str(getattr(args, "translation_model_profile", "") or "").strip()
    if requested_translation:
        translation_profile = requested_translation
    elif sys.stdin.isatty() and not args.yes:
        translation_profile = prompt_translation_model()
    else:
        translation_profile = "off"
        print("Local translation is not selected. Pass --translation-model-profile nllb-200-600m to include it.")

    plan = install_plan_for_target(
        target,
        realtime_preview_engine=preview_engine,
        realtime_preview_model_preset=preview_preset,
        translation_model_profile=translation_profile,
    )
    profile = _load_profile()
    profile = configure_profile_for_install(profile, plan)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.provider_preset:
        profile = apply_provider_preset(profile, args.provider_preset)
    if requested_model_dir is not None:
        profile = profile.with_updates(realtime_preview_model_dir=str(requested_model_dir))
    validate_realtime_preview_language(profile)

    if args.dry_run:
        print(f"Dry run: would save {profile.mode} profile to {config_path()}")
    else:
        save_path = _save_profile(profile)
        print(f"Saved {profile.mode} profile to {save_path}")

    print_install_plan(plan, profile)
    if not confirm_install_start(args.yes, args.dry_run):
        print("Install skipped.")
        return 0

    if plan.target == "macos":
        code = install_macos_runtime(assume_yes=True, dry_run=args.dry_run)
    else:
        code = install_extra_and_maybe_kroko(
            profile,
            plan.extra,
            assume_yes=True,
            dry_run=args.dry_run,
            install_kroko=plan.install_kroko,
            kroko_assume_yes=True if plan.install_kroko else False,
            torch_policy=getattr(args, "torch", None),
        )
    if code:
        return code

    if plan.translation_model_profile != "off":
        code = install_translation_runtime(
            profile,
            plan.translation_model_profile,
            assume_yes=True,
            dry_run=args.dry_run,
            venv_dir=getattr(args, "translation_venv", None),
            model_dir=getattr(args, "translation_model_dir", None),
            torch_policy=getattr(args, "torch", None),
            download_model=not bool(getattr(args, "skip_translation_model_download", False)),
        )
        if code:
            return code

    report = _run_doctor(profile, profile.mode, deep=args.deep)
    print_report(report)
    return 0


def cmd_install_translation(args: argparse.Namespace) -> int:
    profile = _load_profile()
    return install_translation_runtime(
        profile,
        args.model_profile,
        assume_yes=args.yes,
        dry_run=args.dry_run,
        venv_dir=args.venv,
        model_dir=args.model_dir,
        torch_policy=args.torch,
        download_model=not args.skip_model_download,
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    profile = _load_profile()
    if args.mode and args.mode != "auto":
        profile = configure_profile_for_mode(profile, args.mode)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    overrides = {
        "remote_asr_url": args.remote_asr_url,
        "remote_embeddings_url": args.remote_embeddings_url,
        "port": args.port,
    }
    profile = profile.with_updates(**{
        key: value for key, value in overrides.items() if value is not None and value != ""
    })
    report = _run_doctor(profile, args.mode or "auto", deep=args.deep)
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        print_report(report)
    if args.fix:
        extra = recommended_install_extra(profile, report)
        if extra:
            code = install_extra(
                extra,
                assume_yes=args.yes,
                dry_run=args.dry_run,
                torch_policy=getattr(args, "torch", None),
            )
            if code:
                return code
        else:
            print("No Python package install action was recommended for the current failures.")
    return 1 if args.strict and report.has_failures else 0


def cmd_setup(args: argparse.Namespace) -> int:
    profile = _load_profile()
    profile = configure_profile_for_mode(profile, args.mode)
    if args.language:
        profile = apply_profile_updates(profile, [("language", args.language)])
    if args.provider_preset:
        profile = apply_provider_preset(profile, args.provider_preset)
    if getattr(args, "realtime_preview_engine", ""):
        preview_engine = normalize_preview_engine(args.realtime_preview_engine)
        preview_model_dir = profile.realtime_preview_model_dir
        if preview_engine in {"kroko_onnx", "sherpa_onnx"}:
            default_preset = get_preview_backend_spec(preview_engine).default_preset or ""
            try:
                preview_preset = normalize_preview_model_preset(
                    preview_engine,
                    profile.realtime_preview_model_preset or default_preset,
                )
            except (ValueError, argparse.ArgumentTypeError):
                preview_preset = default_preset
        else:
            preview_preset = ""
        if preview_engine != "sherpa_onnx":
            preview_model_dir = ""
        profile = profile.with_updates(
            realtime_preview_engine=preview_engine,
            realtime_preview_model_preset=preview_preset,
            realtime_preview_model_dir=preview_model_dir,
        )
    if getattr(args, "realtime_preview_model_preset", ""):
        profile = profile.with_updates(
            realtime_preview_model_preset=normalize_preview_model_preset(
                profile.realtime_preview_engine,
                args.realtime_preview_model_preset,
            )
        )
    if getattr(args, "realtime_preview_model_dir", None) is not None:
        if profile.realtime_preview_engine != "sherpa_onnx":
            raise SystemExit("--realtime-preview-model-dir is only valid with sherpa_onnx.")
        profile = profile.with_updates(
            realtime_preview_model_dir=str(args.realtime_preview_model_dir)
        )
    validate_realtime_preview_language(profile)
    if args.dry_run:
        save_path = config_path()
        print(f"Dry run: would save {profile.mode} profile to {save_path}")
    else:
        save_path = _save_profile(profile)
        print(f"Saved {profile.mode} profile to {save_path}")
    report = _run_doctor(profile, profile.mode, deep=args.deep)
    print_report(report)
    if args.install:
        extra = recommended_install_extra(profile, report)
        if profile.mode == "local":
            extra = install_plan_for_target(
                "local",
                realtime_preview_engine=profile.realtime_preview_engine,
                realtime_preview_model_preset=profile.realtime_preview_model_preset,
            ).extra
        if extra is None and profile.mode == "server":
            extra = "server"
        if extra is None and profile.mode == "remote":
            extra = "controller"
        if extra is not None:
            return install_extra_and_maybe_kroko(
                profile,
                extra,
                assume_yes=args.yes,
                dry_run=args.dry_run,
                install_kroko=preview_engine_uses_kroko(profile) and not args.skip_kroko,
                torch_policy=getattr(args, "torch", None),
            )
    print("Launch command:")
    print(f"  {format_command(build_launch_command(profile))}")
    return 0


def cmd_install_kroko(args: argparse.Namespace) -> int:
    profile = _load_profile()
    profile = profile.with_updates(**{
        key: value
        for key, value in {
            "realtime_preview_python": args.python,
            "realtime_preview_engine": args.engine,
        }.items()
        if value
    })
    return install_kroko_runtime(
        profile,
        assume_yes=args.yes,
        dry_run=args.dry_run,
        variant=args.variant,
        work_dir=args.work_dir,
        soft_fail=False,
    )


def cmd_launch(args: argparse.Namespace) -> int:
    profile = _load_profile()
    if profile.deployment_target == "macos":
        _facade_callable("require_apple_silicon_macos", require_apple_silicon_macos)()
    updates: list[tuple[str, Any]] = []
    if args.language:
        updates.append(("language", args.language))
    if args.port is not None:
        updates.append(("port", args.port))
    if updates:
        profile = apply_profile_updates(profile, updates)
    if args.provider_preset:
        profile = apply_provider_preset(profile, args.provider_preset)
    if getattr(args, "translation", None) is not None:
        profile = profile.with_updates(translation_enabled=bool(args.translation))
    if profile.mode == "server":
        print("Server profile service commands:")
        for line in build_server_launch_lines():
            print(line)
        if not args.print_only and not args.dry_run:
            print("Start each command in a separate shell so both services stay running.")
        return 0
    launch_plan = build_launch_plan(profile, args.extra_args or "")
    command = list(launch_plan.live)
    reports_command: list[str] | None = None
    translation_command: list[str] | None = None
    if args.with_reports:
        reports_command = build_reports_command(profile, **report_launch_values(profile, args))
        print("Meeting reports command:")
        print(format_command(reports_command))
    if profile.translation_enabled and profile.translation_provider == "sidecar":
        translation_command = build_translation_command(profile)
        print("Translation sidecar command:")
        print(format_command(translation_command))
    if reports_command is not None or translation_command is not None:
        print("Live window command:")
    print(format_command(command))
    for spec in launch_plan.services:
        print(f"Managed {spec.name}: {format_command(list(spec.command))}")
    if args.print_only or args.dry_run:
        return 0
    owned_services: list[object] = []
    try:
        try:
            for spec in launch_plan.services:
                if _facade_callable("service_health_ready", service_health_ready)(spec):
                    print(f"Using already-running compatible {spec.name} at {spec.health_url}")
                    continue
                print(f"Starting {spec.name}...")
                process = _facade_callable("start_service_process", start_service_process)(spec)
                owned_services.append(process)
                _facade_callable("wait_for_service_health", wait_for_service_health)(spec, process)
                print(f"{spec.name} is healthy at {spec.health_url}")
        except (OSError, RuntimeError) as exc:
            print(f"Managed service startup failed: {exc}")
            return 1
        if reports_command is not None:
            popen_kwargs: dict[str, Any] = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(reports_command, **popen_kwargs)
        if translation_command is not None:
            popen_kwargs = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(translation_command, **popen_kwargs)
        return subprocess.run(command, check=False).returncode
    finally:
        _facade_callable("terminate_service_processes", terminate_service_processes)(owned_services)


def cmd_reports(args: argparse.Namespace) -> int:
    profile = _load_profile()
    values = report_launch_values(profile, args)
    command = build_reports_command(profile, **values)
    print(format_command(command))
    if args.print_only or args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def cmd_translation(args: argparse.Namespace) -> int:
    profile = _load_profile()
    updates: list[tuple[str, Any]] = []
    for field_name in ("translation_port", "translation_model_profile", "translation_model", "translation_python", "translation_device"):
        value = getattr(args, field_name, None)
        if value is not None and value != "":
            updates.append((field_name, value))
    if updates:
        profile = apply_profile_updates(profile, updates)
    command = build_translation_command(profile)
    print(format_command(command))
    if args.print_only or args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def cmd_config(args: argparse.Namespace) -> int:
    profile = Profile() if args.reset else _load_profile()
    updates: list[tuple[str, Any]] = []
    direct_fields = (
        "mode",
        "host",
        "port",
        "language",
        "model",
        "device",
        "compute_type",
        "asr_backend",
        "embeddings_backend",
        "provider_preset",
        "remote_asr_url",
        "remote_embeddings_url",
        "embedding_provider",
        "live_speaker_embedding_provider",
        "live_speaker_assignment",
        "embedding_python",
        "vad_backend",
        "realtime_preview_engine",
        "realtime_preview_model_preset",
        "realtime_preview_model_dir",
        "realtime_preview_python",
        "reports_enabled",
        "reports_port",
        "report_language",
        "report_llm_provider",
        "report_llm_base_url",
        "report_llm_model",
        "report_auto_generate",
        "translation_enabled",
        "translation_browser_preferred",
        "translation_provider",
        "translation_port",
        "translation_target_languages",
        "translation_max_targets",
        "translation_model_profile",
        "translation_model",
        "translation_base_url",
        "translation_api_key_env",
        "translation_region",
        "translation_python",
        "translation_device",
        "advanced_args",
    )
    for field_name in direct_fields:
        value = getattr(args, field_name, None)
        if value is not None and value != "":
            updates.append((field_name, value))
    if args.set:
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"Invalid --set value {item!r}; use name=value.")
            key, value = item.split("=", 1)
            updates.append((key, value))
    if updates:
        profile = apply_profile_updates(profile, updates)
    if args.edit:
        profile = edit_profile(profile)
        profile = Profile.from_mapping(profile.as_dict())
    if args.reset or updates or args.edit:
        _save_profile(profile)
    if args.json:
        print(json.dumps(profile.as_dict(), indent=2, sort_keys=True))
    else:
        print_profile(profile)
    return 0
