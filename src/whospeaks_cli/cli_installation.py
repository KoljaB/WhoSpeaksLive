"""Installation planning and execution for the WhoSpeaks CLI."""

from __future__ import annotations

import json
import importlib.metadata
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
    preview_language_error,
)

from .cli_console import detail_text, language_summary, print_wrapped, read_input
from .cli_diagnostics import DoctorReport, TorchInstallSelection, installed_distribution_version, run_doctor
from .planning import (
    CONTROLLER_EXTRA,
    LOCAL_EXTRA,
    PREVIEW_EXTRA,
    TRANSLATION_INSTALL_PROFILE_CHOICES,
    InstallPlan,
    build_translation_command,
    default_macos_runtime_root,
    normalize_install_target,
    profile_for_install,
    profile_for_mode,
    service_resource_path,
)
from .profiles import Profile, config_path, normalize_mode, save_profile, update_profile_in_place
from .runtime_constants import (
    DEFAULT_PYTORCH_CUDA_BUILD,
    INSTALLER_BACKEND_CHOICES,
    INSTALLER_BACKEND_ENV,
    KROKO_INSTALL_MODULE,
    KROKO_PREVIEW_VENV_ENV,
    PACKAGE_NAME,
    PIP_EXTRA_INDEX_URL_ENV,
    PIP_FIND_LINKS_ENV,
    PIP_INDEX_URL_ENV,
    PYPI_SIMPLE_URL,
    PYTORCH_CPU_INDEX_URL,
    PYTORCH_CPU_INDEX_URL_ENV,
    PYTORCH_CUDA_BUILD_ENV,
    PYTORCH_CUDA_INDEX_URLS,
    PYTORCH_CUDA_INDEX_URL_ENV,
    TESTPYPI_SIMPLE_URL,
    TORCH_INSTALL_POLICY_CHOICES,
    TORCH_INSTALL_POLICY_ENV,
    TORCH_PACKAGE_SPECS,
    TRANSLATION_MODEL_ROOT_ENV,
    TRANSLATION_VENV_ROOT_ENV,
)


SETUPTOOLS_RUNTIME_SPEC = "setuptools>=68,<82"

__all__ = [
    "normalize_installer_backend", "installer_backend_available", "package_install_prefix",
    "package_extra_spec", "configure_profile_for_install", "version_is_prerelease",
    "pip_index_args_for_installed_package", "package_index_args_for_installed_package", "build_install_command",
    "normalize_torch_install_policy", "normalize_pytorch_cuda_build", "extra_needs_torch",
    "detect_nvidia_cuda", "select_torch_install", "build_torch_install_command",
    "report_torch_runtime", "format_command", "recommended_install_extra", "install_extra",
    "print_install_plan", "prompt_install_target", "prompt_realtime_preview",
    "prompt_translation_model", "prompt_installer_backend", "prompt_kroko_install", "confirm_install_start",
    "preview_engine_is_enabled", "preview_engine_uses_kroko",
    "validate_realtime_preview_language", "build_kroko_install_command",
    "default_kroko_preview_venv_path", "venv_python_path", "default_translation_venv_dir",
    "default_translation_model_dir", "translation_package_install_command",
    "build_translation_install_commands", "install_translation_runtime",
    "installed_package_source", "build_macos_install_commands", "install_macos_runtime",
    "query_python_command_info", "windows_kroko_python_is_compatible", "windows_python312_command",
    "windows_kroko_docker_readiness",
    "report_suggests_kroko_install",
    "run_command_sequence", "install_kroko_in_python", "install_kroko_sidecar",
    "install_kroko_runtime", "install_extra_and_maybe_kroko", "configure_profile_for_mode",
]


def _facade_callable(name: str, fallback: Any) -> Any:
    """Honor replacements made through the stable ``whospeaks_cli.main`` facade."""

    facade = sys.modules.get("whospeaks_cli.main")
    return getattr(facade, name, fallback) if facade is not None else fallback


def normalize_installer_backend(value: str | None = None) -> str:
    """Return an explicit installer, or prefer available uv when none was selected."""

    configured = os.environ.get(INSTALLER_BACKEND_ENV)
    selected = value if value is not None else configured
    if selected is None:
        selected = "uv" if shutil.which("uv") else "pip"
    normalized = str(selected or "pip").strip().lower().replace("-", "_")
    normalized = {"python": "pip", "uv_pip": "uv"}.get(normalized, normalized)
    if normalized not in INSTALLER_BACKEND_CHOICES:
        raise SystemExit(
            "Unknown installer backend {0!r}. Choose one of: {1}.".format(
                selected,
                ", ".join(INSTALLER_BACKEND_CHOICES),
            )
        )
    return normalized


def installer_backend_available(value: str | None = None) -> bool:
    backend = normalize_installer_backend(value)
    return backend == "pip" or bool(shutil.which("uv"))


def package_install_prefix(
    python_executable: str | Path = sys.executable,
    *,
    installer_backend: str | None = None,
) -> list[str]:
    """Build an install prefix that targets one explicit Python environment."""

    backend = normalize_installer_backend(installer_backend)
    python = str(python_executable)
    if backend == "pip":
        return [python, "-m", "pip", "install"]
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "The uv installer was selected, but `uv` was not found on PATH. "
            "Install uv or rerun with --installer pip."
        )
    return [uv, "pip", "install", "--python", python]


def package_extra_spec(extra: str) -> str:
    version = _facade_callable("installed_distribution_version", installed_distribution_version)(PACKAGE_NAME)
    if version:
        return f"{PACKAGE_NAME}[{extra}]=={version}"
    return f"{PACKAGE_NAME}[{extra}]"


def configure_profile_for_install(profile: Profile, plan: InstallPlan) -> Profile:
    """Configure a legacy retained profile from the pure installation planner."""

    return update_profile_in_place(profile, profile_for_install(profile, plan))


def version_is_prerelease(version: str | None) -> bool:
    normalized = str(version or "").lower()
    return any(marker in normalized for marker in ("a", "b", "rc", "dev"))


def pip_index_args_for_installed_package() -> list[str]:
    args: list[str] = []
    index_url = os.environ.get(PIP_INDEX_URL_ENV, "").strip()
    extra_index_url = os.environ.get(PIP_EXTRA_INDEX_URL_ENV, "").strip()
    find_links = os.environ.get(PIP_FIND_LINKS_ENV, "").strip()
    if index_url:
        args.extend(["--index-url", index_url])
    if extra_index_url:
        args.extend(["--extra-index-url", extra_index_url])
    elif version_is_prerelease(
        _facade_callable("installed_distribution_version", installed_distribution_version)(PACKAGE_NAME)
    ):
        args.extend(["--extra-index-url", TESTPYPI_SIMPLE_URL])
    if find_links:
        args.extend(["--find-links", find_links])
    return args


def package_index_args_for_installed_package(
    installer_backend: str | None = None,
) -> list[str]:
    """Return index arguments adjusted for the selected package installer."""

    args = pip_index_args_for_installed_package()
    backend = normalize_installer_backend(installer_backend)
    if backend == "uv" and any("test.pypi.org" in arg.lower() for arg in args):
        # TestPyPI contains placeholders and incomplete file sets for packages
        # that also exist on PyPI. Keep PyPI first so normal dependencies use
        # its complete wheel set, then let the exact WhoSpeaks development
        # version fall through to TestPyPI. Stable PyPI installs retain uv's
        # safer default first-index strategy.
        primary_index = PYPI_SIMPLE_URL
        test_index = TESTPYPI_SIMPLE_URL
        remaining: list[str] = []
        offset = 0
        while offset < len(args):
            flag = args[offset]
            if flag in {"--index-url", "--extra-index-url"} and offset + 1 < len(args):
                value = args[offset + 1]
                if "test.pypi.org" in value.lower():
                    test_index = value
                elif flag == "--index-url":
                    primary_index = value
                else:
                    remaining.extend((flag, value))
                offset += 2
                continue
            remaining.append(flag)
            offset += 1
        return [
            "--index", primary_index,
            "--index", test_index,
            "--index-strategy", "unsafe-first-match",
            *remaining,
        ]
    return args


def build_install_command(
    extra: str = LOCAL_EXTRA,
    *,
    python_executable: str | Path = sys.executable,
    installer_backend: str | None = None,
) -> list[str]:
    extra = str(extra or LOCAL_EXTRA).strip()
    return [
        *package_install_prefix(
            python_executable,
            installer_backend=installer_backend,
        ),
        *package_index_args_for_installed_package(installer_backend),
        package_extra_spec(extra),
    ]


def installed_package_source() -> tuple[Path | None, bool]:
    try:
        direct_url = importlib.metadata.distribution(PACKAGE_NAME).read_text("direct_url.json")
        payload = json.loads(direct_url or "{}")
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, TypeError):
        payload = {}
    parsed = urlparse(str(payload.get("url") or ""))
    if parsed.scheme == "file":
        path = Path(url2pathname(unquote(parsed.path)))
        if path.exists():
            editable = bool((payload.get("dir_info") or {}).get("editable"))
            return path, editable
    spec = importlib.util.find_spec("whospeaks_cli")
    locations = (spec.submodule_search_locations or ()) if spec is not None else ()
    for location in locations:
        for candidate in Path(location).resolve().parents:
            pyproject = candidate / "pyproject.toml"
            try:
                project_name = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["name"]
            except (FileNotFoundError, OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
                continue
            if str(project_name).strip().lower() == PACKAGE_NAME:
                return candidate, True
    return None, False


def macos_package_install_command(
    python_executable: str | Path,
    *,
    extra: str = "",
    no_deps: bool = False,
    installer_backend: str | None = None,
) -> list[str]:
    command = package_install_prefix(
        python_executable,
        installer_backend=installer_backend,
    )
    if no_deps:
        command.append("--no-deps")
    source, editable = _facade_callable("installed_package_source", installed_package_source)()
    if source is not None:
        target = f"{source}[{extra}]" if extra else str(source)
        if editable:
            command.append("-e")
        command.append(target)
        return command
    command.extend(package_index_args_for_installed_package(installer_backend))
    if extra:
        command.append(package_extra_spec(extra))
    else:
        version = _facade_callable("installed_distribution_version", installed_distribution_version)(PACKAGE_NAME)
        command.append(f"{PACKAGE_NAME}=={version}" if version else PACKAGE_NAME)
    return command


def build_macos_install_commands(
    runtime_root: Path | None = None,
    *,
    installer_backend: str | None = None,
) -> list[list[str]]:
    root = (runtime_root or default_macos_runtime_root()).expanduser()
    asr_venv = root / "mlx-asr"
    embeddings_venv = root / "embeddings"
    asr_python = venv_python_path(asr_venv)
    embeddings_python = venv_python_path(embeddings_venv)
    requirements = service_resource_path("voice-embeddings-server", "requirements-macos.txt")
    return [
        macos_package_install_command(
            sys.executable,
            extra=CONTROLLER_EXTRA,
            installer_backend=installer_backend,
        ),
        [sys.executable, "-m", "venv", str(asr_venv)],
        [
            *package_install_prefix(asr_python, installer_backend=installer_backend),
            "--upgrade", "pip", SETUPTOOLS_RUNTIME_SPEC, "wheel",
        ],
        [
            *package_install_prefix(asr_python, installer_backend=installer_backend),
            "mlx-whisper", "fastapi>=0.110", "numpy>=2,<3", "uvicorn[standard]>=0.29",
        ],
        macos_package_install_command(
            asr_python,
            no_deps=True,
            installer_backend=installer_backend,
        ),
        [sys.executable, "-m", "venv", str(embeddings_venv)],
        [
            *package_install_prefix(embeddings_python, installer_backend=installer_backend),
            "--upgrade", "pip", SETUPTOOLS_RUNTIME_SPEC, "wheel",
        ],
        [
            *package_install_prefix(embeddings_python, installer_backend=installer_backend),
            "-r", str(requirements),
        ],
        macos_package_install_command(
            embeddings_python,
            no_deps=True,
            installer_backend=installer_backend,
        ),
    ]


def install_macos_runtime(
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    runtime_root: Path | None = None,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    commands = build_macos_install_commands(
        runtime_root,
        installer_backend=backend,
    )
    print("Apple Silicon managed runtime install commands:")
    print(f"Python package installer: {backend}")
    for command in commands:
        print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Install the controller and managed macOS services now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("macOS runtime installation skipped.")
            return 0
    root = (runtime_root or default_macos_runtime_root()).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return run_command_sequence(commands)


def normalize_torch_install_policy(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "gpu": "cuda",
        "nvidia": "cuda",
        "cu118": "cuda",
        "cu126": "cuda",
        "cu128": "cuda",
        "none": "skip",
        "off": "skip",
        "no": "skip",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TORCH_INSTALL_POLICY_CHOICES:
        raise SystemExit(
            "Unknown Torch install policy {0!r}. Choose one of: {1}.".format(
                value,
                ", ".join(TORCH_INSTALL_POLICY_CHOICES),
            )
        )
    return normalized


def normalize_pytorch_cuda_build(value: str | None) -> str:
    normalized = str(value or DEFAULT_PYTORCH_CUDA_BUILD).strip().lower().replace(".", "")
    aliases = {
        "cuda": DEFAULT_PYTORCH_CUDA_BUILD,
        "gpu": DEFAULT_PYTORCH_CUDA_BUILD,
        "12": DEFAULT_PYTORCH_CUDA_BUILD,
        "128": "cu128",
        "12_8": "cu128",
        "12-8": "cu128",
        "126": "cu126",
        "12_6": "cu126",
        "12-6": "cu126",
        "118": "cu118",
        "11_8": "cu118",
        "11-8": "cu118",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PYTORCH_CUDA_INDEX_URLS:
        raise SystemExit(
            "Unknown PyTorch CUDA build {0!r}. Choose one of: {1}.".format(
                value,
                ", ".join(PYTORCH_CUDA_INDEX_URLS),
            )
        )
    return normalized


def extra_needs_torch(extra: str) -> bool:
    tokens = {item.strip() for item in str(extra or "").split(",") if item.strip()}
    return bool(tokens & {"all", "complete", "local", "server"})


def detect_nvidia_cuda() -> tuple[bool, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi was not found; assuming CPU PyTorch."
    command = [
        nvidia_smi,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return False, f"nvidia-smi could not be executed ({type(exc).__name__}: {exc}); assuming CPU PyTorch."
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            return False, f"nvidia-smi returned {completed.returncode}: {detail}; assuming CPU PyTorch."
        return False, f"nvidia-smi returned {completed.returncode}; assuming CPU PyTorch."
    first_line = next((line.strip() for line in (completed.stdout or "").splitlines() if line.strip()), "")
    if not first_line:
        return False, "nvidia-smi did not report an NVIDIA GPU; assuming CPU PyTorch."
    return True, f"NVIDIA GPU/driver detected: {first_line}."


def select_torch_install(policy: str | None = None) -> TorchInstallSelection:
    selected_policy = normalize_torch_install_policy(
        policy if policy is not None else os.environ.get(TORCH_INSTALL_POLICY_ENV, "auto")
    )
    if selected_policy == "skip":
        return TorchInstallSelection("skip", "", "Torch preinstall disabled.")
    cuda_build = normalize_pytorch_cuda_build(os.environ.get(PYTORCH_CUDA_BUILD_ENV, DEFAULT_PYTORCH_CUDA_BUILD))
    cuda_index = os.environ.get(PYTORCH_CUDA_INDEX_URL_ENV, "").strip() or PYTORCH_CUDA_INDEX_URLS[cuda_build]
    if selected_policy == "cuda":
        return TorchInstallSelection("cuda", cuda_index, f"CUDA PyTorch forced with {cuda_build}.", cuda_build)
    if selected_policy == "cpu":
        cpu_index = os.environ.get(PYTORCH_CPU_INDEX_URL_ENV, "").strip() or (
            "" if platform.system() == "Darwin" else PYTORCH_CPU_INDEX_URL
        )
        return TorchInstallSelection("cpu", cpu_index, "CPU PyTorch forced.")
    has_cuda, detail = detect_nvidia_cuda()
    if has_cuda and platform.system() in {"Windows", "Linux"}:
        return TorchInstallSelection("cuda", cuda_index, f"{detail} Installing PyTorch {cuda_build}.", cuda_build)
    cpu_index = os.environ.get(PYTORCH_CPU_INDEX_URL_ENV, "").strip() or (
        "" if platform.system() == "Darwin" else PYTORCH_CPU_INDEX_URL
    )
    return TorchInstallSelection("cpu", cpu_index, detail)


def build_torch_install_command(
    policy: str | None = None,
    *,
    python_executable: str | Path = sys.executable,
    installer_backend: str | None = None,
) -> tuple[list[str], TorchInstallSelection]:
    selection = select_torch_install(policy)
    if not selection.should_install:
        return [], selection
    command = [
        *package_install_prefix(
            python_executable,
            installer_backend=installer_backend,
        ),
        "--upgrade",
    ]
    if selection.index_url:
        command.extend(["--index-url", selection.index_url])
    command.extend(TORCH_PACKAGE_SPECS)
    return command, selection


def report_torch_runtime(selection: TorchInstallSelection) -> int:
    if not selection.should_install:
        return 0
    script = (
        "import json, torch; "
        "print(json.dumps({"
        "'version': getattr(torch, '__version__', ''), "
        "'cuda_runtime': getattr(getattr(torch, 'version', None), 'cuda', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        print(f"PyTorch verification failed: {detail}")
        return int(completed.returncode)
    detail = (completed.stdout or "").strip().splitlines()[-1]
    print(f"PyTorch verification: {detail}")
    if selection.mode == "cuda":
        try:
            payload = json.loads(detail)
        except Exception:
            payload = {}
        if not payload.get("cuda_available"):
            print("Warning: CUDA PyTorch installed, but torch.cuda.is_available() is false. The launcher can still fall back to CPU.")
    return 0


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        rendered: list[str] = []
        for arg in command:
            if any(token in arg for token in ("[", "]", "<", ">", "&", "|")):
                rendered.append('"' + arg.replace('"', '\\"') + '"')
            else:
                rendered.append(subprocess.list2cmdline([arg]))
        return " ".join(rendered)
    return shlex.join(command)


def recommended_install_extra(profile: Profile, report: DoctorReport) -> str | None:
    if profile.reports_enabled and any(
        check.status == "fail" and check.name == "Meeting Intelligence modules"
        for check in report.checks
    ):
        return "intelligence"
    if report.mode == "server":
        return "server"
    if report.mode in {"local", "cpu"}:
        for check in report.checks:
            if check.status == "fail" and check.name in {
                "Controller Python modules",
                "Local ASR modules",
                "Local embedding modules",
            }:
                return LOCAL_EXTRA
        if report.mode == "local" and any(check.status == "warn" and check.name == "CUDA visibility" for check in report.checks):
            return LOCAL_EXTRA
        if any(check.status == "warn" and check.name == "Realtime preview" for check in report.checks):
            return PREVIEW_EXTRA
    if report.mode == "remote":
        for check in report.checks:
            if check.status == "fail" and check.name == "Controller Python modules":
                return CONTROLLER_EXTRA
        if any(check.status == "warn" and check.name == "Realtime preview" for check in report.checks):
            return PREVIEW_EXTRA
    return None


def install_extra(
    extra: str,
    assume_yes: bool = False,
    dry_run: bool = False,
    *,
    torch_policy: str | None = None,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    pip_bootstrap_command = (
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", SETUPTOOLS_RUNTIME_SPEC]
        if backend == "pip"
        else []
    )
    command = _facade_callable("build_install_command", build_install_command)(
        extra,
        installer_backend=backend,
    )
    torch_command: list[str] = []
    torch_selection = TorchInstallSelection("skip", "", "Torch is not needed for this dependency set.")
    if extra_needs_torch(extra):
        torch_command, torch_selection = _facade_callable(
            "build_torch_install_command", build_torch_install_command
        )(torch_policy, installer_backend=backend)
        print("PyTorch install selection:")
        print(f"  {torch_selection.reason}")
        if torch_command:
            print("PyTorch install command:")
            print(f"  {format_command(torch_command)}")
        else:
            print("PyTorch install command:")
            print("  skipped")
    print(f"Python package installer: {backend}")
    if pip_bootstrap_command:
        print("pip bootstrap command:")
        print(f"  {format_command(pip_bootstrap_command)}")
    print("WhoSpeaks package install command:")
    print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Run these install commands now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Install skipped.")
            return 0
    if pip_bootstrap_command:
        completed = subprocess.run(pip_bootstrap_command, check=False)
        if completed.returncode != 0:
            print("Warning: pip could not be updated; continuing with the installed pip version.")
    if torch_command:
        completed = subprocess.run(torch_command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
        code = report_torch_runtime(torch_selection)
        if code:
            return code
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def print_install_plan(
    plan: InstallPlan,
    profile: Profile,
    *,
    installer_backend: str | None = None,
) -> None:
    backend = normalize_installer_backend(installer_backend)
    print("WhoSpeaks install plan")
    print("=" * 72)
    print(f"Target: {plan.title}")
    print(f"Profile: {plan.mode}")
    print_wrapped(plan.summary, initial_indent="Summary: ", subsequent_indent="         ", style=detail_text)
    if plan.realtime_preview_engine == "sherpa_onnx":
        preset = plan.realtime_preview_model_preset
        latency_note = "higher stability" if "560ms" in preset else "lower latency"
        print(f"Realtime text: Nemotron 3.5 ({preset}; {latency_note}).")
        print("               sherpa-onnx packages install with WhoSpeaks; the verified model downloads on first launch.")
    elif plan.install_kroko:
        print("Realtime text: Kroko through the official sherpa-onnx CPU runtime; Docker is not required.")
        print("               The verified language model downloads automatically on first launch.")
    elif plan.target in {"local", "core", "macos"}:
        print("Realtime text: disabled for this install. Run the installer again and choose Kroko to try native live text.")
    else:
        print("Realtime text: not part of the server package install.")
    if plan.translation_model_profile == "off":
        print("Translation: not installed by this plan.")
    else:
        print(f"Translation: isolated local {plan.translation_model_profile} sidecar and model files.")
    print(f"Language: {language_summary(profile.language)}")
    print(f"Internal dependency set: {plan.extra}")
    print(f"Python package installer: {backend}")
    print("Underlying package command:")
    print(f"  {format_command(build_install_command(plan.extra, installer_backend=backend))}")
    print("=" * 72)


def prompt_install_target() -> str:
    print(textwrap.dedent(
        """
        What do you want to install?
          1. Full local installation
          2. CPU-only local installation
          3. Apple Silicon managed local services
          4. Core/controller for remote ASR and embeddings servers
          5. ASR and embeddings server packages
        """
    ).strip())
    while True:
        choice = read_input("> ", "1").strip().lower()
        if choice in {"1", "local", "full", "full local"}:
            return "local"
        if choice in {"2", "cpu", "cpu only", "cpu-only"}:
            return "cpu"
        if choice in {"3", "macos", "mac", "apple silicon"}:
            return "macos"
        if choice in {"4", "core", "controller", "remote"}:
            return "core"
        if choice in {"5", "server", "gpu", "services"}:
            return "server"
        print("Choose 1, 2, 3, 4, or 5.")


def prompt_realtime_preview(target: str) -> tuple[str, str]:
    if normalize_install_target(target) not in {"local", "core", "macos"}:
        return "off", ""
    print()
    print("Realtime preview text")
    print_wrapped(
        "1. Nemotron 3.5 (recommended on Windows): installs sherpa-onnx packages normally and downloads a verified model on first use. "
        "2. Nemotron 3.5 low latency: 160ms model, less robust. "
        "3. Kroko/Banafo: official sherpa-onnx CPU runtime; no Docker build required. "
        "4. Disable realtime text. Final ASR and speaker diarization work independently.",
        initial_indent="",
        subsequent_indent="",
        style=detail_text,
    )
    while True:
        answer = read_input("Choose realtime preview [1/2/3/4] ", "1").strip().lower()
        if answer in {"1", "nemotron", "nemotron-560", "560"}:
            return "sherpa_onnx", "nemotron-3.5-560ms-int8"
        if answer in {"2", "nemotron-160", "160"}:
            return "sherpa_onnx", "nemotron-3.5-160ms-int8"
        if answer in {"3", "kroko", "banafo"}:
            return "kroko_onnx", "community-64l"
        if answer in {"4", "off", "none", "no"}:
            return "off", ""
        print("Choose 1, 2, 3, or 4.")


def prompt_translation_model() -> str:
    print()
    print("Local translation server")
    print_wrapped(
        "1. NLLB-200 600M: broad language coverage and the safest GPU-memory choice. "
        "2. TranslateGemma 4B: quality-first, but requires accepting the Gemma terms and more VRAM. "
        "3. MADLAD-400 3B: Apache-2.0 weights with higher memory requirements. "
        "4. Do not install local translation.",
        initial_indent="",
        subsequent_indent="",
        style=detail_text,
    )
    while True:
        answer = read_input("Choose translation [1/2/3/4] ", "1").strip().lower()
        if answer in {"1", "nllb", "nllb-200", "nllb-200-600m"}:
            return "nllb-200-600m"
        if answer in {"2", "gemma", "translategemma", "translate-gemma-4b"}:
            return "translate-gemma-4b"
        if answer in {"3", "madlad", "madlad-400", "madlad-400-3b"}:
            return "madlad-400-3b"
        if answer in {"4", "off", "none", "no"}:
            return "off"
        print("Choose 1, 2, 3, or 4.")


def prompt_installer_backend() -> str:
    default = normalize_installer_backend(None)
    uv_note = "available" if installer_backend_available("uv") else "not found on PATH"
    print()
    print("Python package installer")
    print("  1. pip: compatibility option")
    print(f"  2. uv: preferred faster dependency resolution and installation ({uv_note})")
    while True:
        default_choice = "2" if default == "uv" else "1"
        answer = read_input("Choose installer [1/2] ", default_choice).strip().lower()
        if answer in {"1", "pip", "python"}:
            return "pip"
        if answer in {"2", "uv", "uv-pip", "uv_pip"}:
            if installer_backend_available("uv"):
                return "uv"
            print("uv was not found on PATH. Install uv first or choose pip.")
            continue
        print("Choose 1 or 2.")


def prompt_kroko_install(target: str) -> bool:
    """Legacy prompt helper retained for classic callers and integrations."""

    engine, _preset = prompt_realtime_preview(target)
    return engine == "kroko_onnx"


def confirm_install_start(assume_yes: bool, dry_run: bool) -> bool:
    if assume_yes or dry_run:
        return True
    if not sys.stdin.isatty():
        print("Use --yes to run the installer non-interactively.")
        return False
    answer = read_input("Start this install now? [Y/n] ", "y").strip().lower()
    return answer in {"", "y", "yes"}


def preview_engine_is_enabled(profile: Profile) -> bool:
    return normalize_preview_engine(profile.realtime_preview_engine) not in {"off", "mock"}


def preview_engine_uses_kroko(profile: Profile) -> bool:
    return normalize_preview_engine(profile.realtime_preview_engine) == "kroko_onnx"


def validate_realtime_preview_language(profile: Profile) -> None:
    error = preview_language_error(profile.realtime_preview_engine, profile.language)
    if error:
        raise SystemExit(error)


def build_kroko_install_command(
    python_executable: str | Path = sys.executable,
    *,
    variant: str = "free",
    work_dir: str | Path | None = None,
) -> list[str]:
    command = [str(python_executable), "-m", KROKO_INSTALL_MODULE, "--build", "--variant", str(variant)]
    if work_dir:
        command.extend(["--work-dir", str(work_dir)])
    return command


def default_kroko_preview_venv_path() -> Path:
    override = os.environ.get(KROKO_PREVIEW_VENV_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or config_path().parent)
        return root / "WhoSpeaks" / "kroko-preview-py312"
    root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return root / "whospeaks" / "kroko-preview-py312"


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def default_translation_venv_dir(model_profile: str) -> Path:
    override = os.environ.get(TRANSLATION_VENV_ROOT_ENV, "").strip()
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or config_path().parent) / "WhoSpeaks" / "translation"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "whospeaks" / "translation"
    return root / model_profile / "venv"


def default_translation_model_dir(model_profile: str) -> Path:
    override = os.environ.get(TRANSLATION_MODEL_ROOT_ENV, "").strip()
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or config_path().parent) / "WhoSpeaks" / "models" / "translation"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "whospeaks" / "models" / "translation"
    return root / model_profile


def translation_package_install_command(
    python_executable: Path,
    *,
    installer_backend: str | None = None,
) -> list[str]:
    prefix = package_install_prefix(
        python_executable,
        installer_backend=installer_backend,
    )
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return [*prefix, "-e", f"{source_root}[translation]"]
    return [
        *prefix,
        *package_index_args_for_installed_package(installer_backend), package_extra_spec("translation"),
    ]


def build_translation_install_commands(
    model_profile: str,
    *,
    venv_dir: Path | None = None,
    model_dir: Path | None = None,
    torch_policy: str | None = None,
    download_model: bool = True,
    installer_backend: str | None = None,
) -> tuple[list[list[str]], Path, Path, TorchInstallSelection]:
    if model_profile not in TRANSLATION_INSTALL_PROFILE_CHOICES or model_profile == "off":
        raise SystemExit("Choose nllb-200-600m, translate-gemma-4b, or madlad-400-3b.")
    environment = (venv_dir or default_translation_venv_dir(model_profile)).expanduser().resolve()
    model_path = (model_dir or default_translation_model_dir(model_profile)).expanduser().resolve()
    python_executable = venv_python_path(environment)
    backend = normalize_installer_backend(installer_backend)
    torch_command, selection = _facade_callable(
        "build_torch_install_command", build_torch_install_command
    )(
        torch_policy,
        python_executable=python_executable,
        installer_backend=backend,
    )
    commands: list[list[str]] = [
        [sys.executable, "-m", "venv", str(environment)],
        [
            *package_install_prefix(python_executable, installer_backend=backend),
            "--upgrade", "pip", SETUPTOOLS_RUNTIME_SPEC, "wheel",
        ],
    ]
    if torch_command:
        commands.append(torch_command)
    commands.append(
        translation_package_install_command(
            python_executable,
            installer_backend=backend,
        )
    )
    prepare = [
        str(python_executable), "-m", "window.translation_installer",
        "--model-profile", model_profile, "--model-dir", str(model_path),
    ]
    if not download_model:
        prepare.append("--verify-only")
    commands.append(prepare)
    return commands, python_executable, model_path, selection


def install_translation_runtime(
    profile: Profile,
    model_profile: str,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    venv_dir: Path | None = None,
    model_dir: Path | None = None,
    torch_policy: str | None = None,
    download_model: bool = True,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    commands, python_executable, model_path, selection = build_translation_install_commands(
        model_profile,
        venv_dir=venv_dir,
        model_dir=model_dir,
        torch_policy=torch_policy,
        download_model=download_model,
        installer_backend=backend,
    )
    print(f"Translation model: {model_profile}")
    print(f"Translation environment: {python_executable.parent.parent}")
    print(f"Translation model files: {model_path}")
    print(f"Python package installer: {backend}")
    print(f"PyTorch: {selection.reason}")
    for command in commands:
        print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Install this local translation server now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Translation installation skipped.")
            return 0
    code = run_command_sequence(commands)
    if code:
        return code
    translation_device = profile.translation_device
    if selection.mode in {"cuda", "cpu"}:
        translation_device = selection.mode
    profile = profile.with_updates(
        translation_enabled=True,
        translation_provider="sidecar",
        translation_model_profile=model_profile,
        translation_python=str(python_executable),
        translation_model=str(model_path),
        translation_device=translation_device,
    )
    save_path = save_profile(profile)
    print(f"Saved local translation server profile to {save_path}")
    print(f"Start command: {format_command(build_translation_command(profile))}")
    return 0


def query_python_command_info(command: list[str]) -> dict[str, Any] | None:
    script = (
        "import json, platform, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'version': list(sys.version_info[:3]), "
        "'bits': 64 if sys.maxsize > 2**32 else 32, "
        "'machine': platform.machine()"
        "}, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [*command, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads((completed.stdout or "{}").strip().splitlines()[-1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def windows_kroko_python_is_compatible(command: list[str]) -> bool:
    """Return whether ``command`` is a CPython runtime supported by Kroko's Windows builder."""

    info = _facade_callable("query_python_command_info", query_python_command_info)(command)
    if not info:
        return False
    version = info.get("version") or []
    bits = int(info.get("bits") or 0)
    return len(version) >= 2 and version[0] == 3 and version[1] == 12 and bits == 64


def windows_python312_command() -> list[str] | None:
    launcher = shutil.which("py")
    candidates: list[list[str]] = []
    if launcher:
        candidates.append([launcher, "-3.12"])
    for path in (
        Path("C:/Python/Python312/python.exe"),
        Path("C:/Python312/python.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe",
    ):
        if path.is_file():
            candidates.append([str(path)])
    for name in ("python3.12", "python"):
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])
    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        if windows_kroko_python_is_compatible(command):
            return command
    return None


def windows_kroko_docker_readiness() -> tuple[bool, str]:
    """Check the Docker Desktop Linux engine required by Kroko's Windows builder."""

    if os.name != "nt":
        return True, ""
    docker = shutil.which("docker")
    if not docker:
        return False, (
            "Docker Desktop is required to build Kroko on Windows, but the Docker CLI was not found. "
            "Install Docker Desktop and enable Linux containers."
        )
    try:
        completed = subprocess.run(
            [docker, "version", "--format", "{{.Server.Os}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return False, f"Docker Desktop could not be contacted: {exc}"
    server_os = (completed.stdout or "").strip().lower()
    if completed.returncode != 0:
        return False, (
            "Docker Desktop is installed, but its Linux engine is not running or cannot be reached."
        )
    if server_os != "linux":
        return False, (
            "Docker Desktop is running with Windows containers. Kroko requires the Linux container engine."
        )
    return True, ""


def report_suggests_kroko_install(profile: Profile, report: DoctorReport) -> bool:
    if normalize_mode(profile.mode) not in {"local", "cpu"}:
        return False
    if not preview_engine_uses_kroko(profile):
        return False
    relevant_names = {"Realtime preview", "Realtime preview Python", "Kroko ONNX runtime"}
    for check in report.checks:
        if check.name in relevant_names and check.status in {"warn", "fail"}:
            return True
    return False


def run_command_sequence(commands: list[list[str]]) -> int:
    for command in commands:
        print(f"+ {format_command(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def install_kroko_in_python(
    python_executable: str | Path,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
) -> int:
    command = build_kroko_install_command(python_executable, variant=variant, work_dir=work_dir)
    print("Kroko native runtime install command:")
    print(f"  {format_command(command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Build/install Kroko native runtime now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Kroko native runtime install skipped.")
            return 0
    code = run_command_sequence([command])
    if code and soft_fail:
        print("Kroko native runtime install did not complete. Final ASR can still run without realtime preview text.")
        return 0
    return code


def install_kroko_sidecar(
    profile: Profile,
    python312_command: list[str] | None,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    venv_dir = default_kroko_preview_venv_path()
    preview_python = venv_python_path(venv_dir)
    if backend == "uv":
        uv = shutil.which("uv")
        if not uv:
            raise SystemExit(
                "The uv installer was selected, but `uv` was not found on PATH. "
                "Install uv or rerun with --installer pip."
            )
        create_environment = [uv, "venv", "--python", "3.12", "--seed", str(venv_dir)]
    elif python312_command:
        create_environment = [*python312_command, "-m", "venv", str(venv_dir)]
    else:
        raise SystemExit("Python 3.12 is required to create the Kroko sidecar with pip.")
    commands = [
        create_environment,
        [
            *package_install_prefix(preview_python, installer_backend=backend),
            "--upgrade", "pip", SETUPTOOLS_RUNTIME_SPEC, "wheel",
        ],
        [
            *package_install_prefix(preview_python, installer_backend=backend),
            *package_index_args_for_installed_package(backend),
            package_extra_spec(PREVIEW_EXTRA),
        ],
        build_kroko_install_command(preview_python, variant=variant, work_dir=work_dir),
    ]
    config_command = [
        sys.executable,
        "-m",
        "whospeaks_cli",
        "config",
        "--realtime-preview-python",
        str(preview_python),
    ]
    print("Kroko native runtime setup will use a Python 3.12 realtime-preview sidecar:")
    print(f"Python package installer: {backend}")
    for command in commands:
        print(f"  {format_command(command)}")
    print(f"  {format_command(config_command)}")
    if dry_run:
        return 0
    if not assume_yes:
        answer = read_input("Create sidecar and build/install Kroko now? [y/N] ", "n").strip().lower()
        if answer not in {"y", "yes"}:
            print("Kroko native runtime install skipped.")
            return 0
    try:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Could not create sidecar parent directory: {exc}")
        return 0 if soft_fail else 1
    code = run_command_sequence(commands)
    if code:
        if soft_fail:
            print("Kroko sidecar setup did not complete. Final ASR can still run without realtime preview text.")
            return 0
        return code
    profile = profile.with_updates(realtime_preview_python=str(preview_python))
    save_path = save_profile(profile)
    print(f"Saved realtime preview Python to {save_path}.")
    return 0


def install_kroko_runtime(
    profile: Profile,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    variant: str = "free",
    work_dir: str | Path | None = None,
    soft_fail: bool = False,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    if not preview_engine_uses_kroko(profile):
        print("Kroko native runtime install skipped because realtime preview is disabled.")
        return 0
    if os.name == "nt" and not dry_run:
        docker_ready, docker_problem = windows_kroko_docker_readiness()
        if not docker_ready:
            print("Kroko setup cannot continue yet.")
            print(docker_problem)
            print(
                "Required action: start Docker Desktop, wait until its Linux engine reports that it is running, "
                "then return to WhoSpeaks and click Install again."
            )
            print("No Kroko sidecar was created or modified.")
            return 1
    configured_python = str(profile.realtime_preview_python or "").strip()
    if configured_python:
        if os.name != "nt" or windows_kroko_python_is_compatible([configured_python]):
            return install_kroko_in_python(
                configured_python,
                assume_yes=assume_yes,
                dry_run=dry_run,
                variant=variant,
                work_dir=work_dir,
                soft_fail=soft_fail,
            )
        print(
            f"Ignoring incompatible realtime preview Python {configured_python}. "
            "Kroko's Windows builder requires CPython 3.12 x64."
        )
        update_profile_in_place(profile, profile.with_updates(realtime_preview_python=""))
    if os.name == "nt":
        if backend == "uv":
            return install_kroko_sidecar(
                profile,
                None,
                assume_yes=assume_yes,
                dry_run=dry_run,
                variant=variant,
                work_dir=work_dir,
                soft_fail=soft_fail,
                installer_backend=backend,
            )
        python312 = _facade_callable("windows_python312_command", windows_python312_command)()
        if python312:
            return install_kroko_sidecar(
                profile,
                python312,
                assume_yes=assume_yes,
                dry_run=dry_run,
                variant=variant,
                work_dir=work_dir,
                soft_fail=soft_fail,
                installer_backend=backend,
            )
        print(
            "Kroko realtime preview on Windows currently needs CPython 3.12 x64 for the native build path. "
            "Install Python 3.12 x64 or set realtime_preview_python to a prepared Python 3.12 environment, "
            "then open `whospeaks`, select Kroko under Settings > General, and install again from Overview."
        )
        return 0 if (soft_fail or dry_run) else 1
    update_profile_in_place(
        profile,
        profile.with_updates(realtime_preview_python=str(sys.executable)),
    )
    code = install_kroko_in_python(
        sys.executable,
        assume_yes=assume_yes,
        dry_run=dry_run,
        variant=variant,
        work_dir=work_dir,
        soft_fail=soft_fail,
    )
    if not dry_run:
        save_path = save_profile(profile)
        print(f"Saved realtime preview Python to {save_path}.")
    return code


def install_extra_and_maybe_kroko(
    profile: Profile,
    extra: str,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
    install_kroko: bool = True,
    kroko_assume_yes: bool | None = None,
    torch_policy: str | None = None,
    installer_backend: str | None = None,
) -> int:
    backend = normalize_installer_backend(installer_backend)
    code = install_extra(
        extra,
        assume_yes=assume_yes,
        dry_run=dry_run,
        torch_policy=torch_policy,
        installer_backend=backend,
    )
    if code:
        return code
    preview_engine = normalize_preview_engine(profile.realtime_preview_engine)
    if preview_engine == "sherpa_onnx":
        preset = profile.realtime_preview_model_preset or "nemotron-3.5-560ms-int8"
        print(
            f"Nemotron realtime preview is configured with {preset}. "
            "The verified model downloads automatically on first preview start."
        )
        return 0
    if preview_engine == "kroko_onnx":
        print(
            "Kroko realtime preview uses the official sherpa-onnx CPU runtime. "
            "The verified language model downloads automatically on first preview start; Docker is not required."
        )
        return 0
    if not install_kroko:
        return 0
    if preview_engine != "kroko_onnx":
        return 0
    report = _facade_callable("run_doctor", run_doctor)(profile)
    if not report_suggests_kroko_install(profile, report):
        return 0
    print("Kroko native runtime is required for realtime preview text.")
    return install_kroko_runtime(
        profile,
        assume_yes=assume_yes if kroko_assume_yes is None else kroko_assume_yes,
        dry_run=dry_run,
        soft_fail=True,
        installer_backend=backend,
    )


def configure_profile_for_mode(profile: Profile, mode: str) -> Profile:
    """Configure a legacy retained profile from the pure deployment planner."""

    return update_profile_in_place(profile, profile_for_mode(profile, mode))
