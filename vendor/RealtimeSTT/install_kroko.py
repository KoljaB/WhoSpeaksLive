"""
Build and install Kroko-ONNX for the active RealtimeSTT environment.
"""

from __future__ import print_function

import argparse
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_REPO = "https://github.com/kroko-ai/kroko-onnx.git"
DEFAULT_BRANCH = "cross-platform-builds"
SUPPORTED_VARIANTS = ("free", "pro")
KROKO_LICENSE_QUIET_ENV = "KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT"


class KrokoInstallError(RuntimeError):
    """
    Reports Kroko installation failures.
    """

    pass


def parse_args(argv=None):
    """
    Parses command-line arguments for Kroko installation.
    """

    parser = argparse.ArgumentParser(
        prog="stt-install-kroko",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Build and install Kroko-ONNX for the active Python environment. "
            "Windows builds a wheel with Kroko's Docker workflow; Linux installs "
            "from the upstream source checkout."
        ),
        epilog=(
            "Platform note:\n"
            "  RealtimeSTT core supports Python 3.11+.\n"
            "  On Windows, stt-install-kroko --build currently requires "
            "CPython 3.12 x64.\n"
            "  Use the same Python 3.12 x64 environment for the builder and "
            "Kroko runtime."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build/install Kroko-ONNX from the upstream source checkout.",
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="free",
        help="Build the free community runtime or the licensed pro runtime.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Kroko-ONNX git repository URL.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Kroko-ONNX git branch to build.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory used for the Kroko-ONNX checkout and build artifacts. "
            "If omitted and the default cache is not writable, a project-local "
            "kroko-builder-work directory is used."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing builder checkout before cloning again.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Build only; do not install the produced package into this Python.",
    )
    return parser.parse_args(argv)


def quote_cmd(cmd):
    """
    Formats a command for readable logging.
    """

    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in cmd])
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd, cwd=None, env=None):
    """
    Runs a subprocess command and reports failures.
    """

    print("+ " + quote_cmd(cmd))
    try:
        subprocess.check_call(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise KrokoInstallError(
            "Command failed with exit code {0}: {1}".format(
                exc.returncode,
                quote_cmd(cmd),
            )
        )


def ensure_program(name, message):
    """
    Verifies that a required program is available.
    """

    if shutil.which(name) is None:
        raise KrokoInstallError(message)


def default_work_dir():
    """
    Returns the default Kroko builder work directory.
    """

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "RealtimeSTT" / "kroko-builder"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "RealtimeSTT" / "kroko-builder"
    else:
        root = os.environ.get("XDG_CACHE_HOME")
        if root:
            return Path(root) / "realtimestt" / "kroko-builder"
    return Path(tempfile.gettempdir()) / "realtimestt-kroko-builder"


def resolve_work_dir(args):
    """
    Resolves the Kroko builder work directory.
    """

    if args.work_dir is not None:
        return args.work_dir.expanduser().resolve()

    work_dir = default_work_dir().expanduser().resolve()
    try:
        ensure_work_dir_writable(work_dir)
        return work_dir
    except KrokoInstallError as exc:
        fallback = (Path.cwd() / "kroko-builder-work").resolve()
        print(
            "Default Kroko builder cache is not writable; using project-local "
            "work directory instead:\n"
            "    {0}\n"
            "Use --work-dir to choose a different location.\n"
            "Original error: {1}".format(fallback, exc),
            file=sys.stderr,
        )
        ensure_work_dir_writable(fallback)
        return fallback


def ensure_work_dir_writable(work_dir):
    """
    Verifies that the builder work directory is writable.
    """

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        probe = work_dir / ".realtimestt-kroko-write-test"
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("ok")
        probe.unlink()
    except OSError as exc:
        raise KrokoInstallError(
            "Kroko builder work directory is not writable: {0}\n"
            "Choose a writable directory with:\n"
            "    stt-install-kroko --build --work-dir .\\kroko-builder-work\n"
            "Original error: {1}".format(work_dir, exc)
        )


def preflight_build(args):
    """
    Checks host prerequisites before building Kroko.
    """

    ensure_program("git", "Git is required to download Kroko-ONNX.")
    work_dir = resolve_work_dir(args)
    ensure_work_dir_writable(work_dir)

    if os.name == "nt":
        ensure_windows_host()
    elif sys.platform.startswith("linux"):
        ensure_program("cmake", "CMake is required to build Kroko-ONNX from source on Linux.")

    return work_dir


def remove_tree_inside(path, root):
    """
    Removes a directory tree after validating its parent root.
    """

    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise KrokoInstallError("Refusing to remove path outside builder cache: {0}".format(path))

    def clear_readonly(func, failed_path, _exc_info):
        """
        Clears read-only file attributes during tree removal.
        """

        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(str(path), onerror=clear_readonly)


def prepare_checkout(args, work_dir=None):
    """
    Prepares the Kroko source checkout.
    """

    work_dir = work_dir or resolve_work_dir(args)
    repo_dir = work_dir / "kroko-onnx"
    ensure_work_dir_writable(work_dir)

    if args.force and repo_dir.exists():
        print("Removing existing Kroko-ONNX checkout: {0}".format(repo_dir))
        remove_tree_inside(repo_dir, work_dir)

    if not repo_dir.exists():
        run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.repo,
                str(repo_dir),
            ]
        )
    else:
        print("Using existing Kroko-ONNX checkout: {0}".format(repo_dir))
        print("Pass --force to delete and clone it again.")

    return repo_dir


def read_text(path):
    """
    Reads a UTF-8 text file.
    """

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def write_text(path, text):
    """
    Writes a UTF-8 text file.
    """

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def normalize_lf(path):
    """
    Normalizes a text file to LF line endings.
    """

    data = path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    if normalized != data:
        path.write_bytes(normalized)
        print("Normalized LF line endings: {0}".format(path.name))


def sanitize_batch_ascii(path):
    """
    Replaces non-ASCII batch file characters with safe text.
    """

    text = read_text(path)
    sanitized = "".join(char if ord(char) < 128 else "-" for char in text)
    if sanitized != text:
        with path.open("w", encoding="ascii", newline="") as handle:
            handle.write(sanitized)
        print("Normalized build_windows.bat to ASCII for cmd.exe.")


def patch_windows_bat(repo_dir):
    """
    Patches the Kroko Windows build batch file.
    """

    path = repo_dir / "build_windows.bat"
    if not path.exists():
        raise KrokoInstallError("Missing Kroko Windows build script: {0}".format(path))

    text = read_text(path)
    if 'findstr /C:"set(SHERPA_ONNX_VERSION"' in text:
        return
    if "Select-String" not in text or "SHERPA_ONNX_VERSION" not in text:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    start = text.find('REM CMakeLists has:  set(SHERPA_ONNX_VERSION "1.12.9")')
    if start == -1:
        start = text.find("set \"VERSION=\"")
    if start == -1:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    if_pos = text.find('if "%VERSION%"==""', start)
    if if_pos == -1:
        print("Could not identify Kroko version parser in build_windows.bat; leaving it unchanged.")
        return

    block_end = text.rfind("\n", 0, if_pos) + 1
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement = newline.join(
        [
            'REM CMakeLists has:  set(SHERPA_ONNX_VERSION "1.12.9")',
            "REM Keep this pure batch so cmd.exe does not parse nested PowerShell regex",
            "REM parentheses inside a FOR command substitution.",
            'set "VERSION="',
            'for /f "tokens=2 delims= " %%v in (\'findstr /C:"set(SHERPA_ONNX_VERSION" "%ROOT%\\CMakeLists.txt"\') do set "VERSION=%%~v"',
            'set "VERSION=%VERSION:"=%"',
            'set "VERSION=%VERSION:)=%"',
            "",
        ]
    )
    write_text(path, text[:start] + replacement + text[block_end:])
    print("Patched build_windows.bat version parsing for cmd.exe.")


def patch_windows_free_wheel_only_build(repo_dir):
    """
    Makes the Windows free build produce only the Python wheel.
    """

    script_path = repo_dir / "in_windows_container.sh"
    batch_path = repo_dir / "build_windows.bat"
    if not script_path.exists():
        raise KrokoInstallError("Missing Kroko container build script: {0}".format(script_path))
    if not batch_path.exists():
        raise KrokoInstallError("Missing Kroko Windows build script: {0}".format(batch_path))

    script = read_text(script_path).replace("\r\n", "\n")
    script_original = script
    flag_marker = "WhoSpeaks patch: free Windows build is wheel-only."
    if flag_marker not in script:
        needle = (
            'echo "    Variant: $BUILD_VARIANT (KROKO_LICENSE=$KROKO_LICENSE_FLAG)"\n'
            "echo\n"
        )
        insertion = (
            "\n"
            "# {0}\n"
            "KROKO_WHEEL_ONLY=0\n"
            "KROKO_WEBSOCKET_FLAG=ON\n"
            'if [ "$BUILD_VARIANT" = "free" ]; then\n'
            "    KROKO_WHEEL_ONLY=1\n"
            "    KROKO_WEBSOCKET_FLAG=OFF\n"
            "fi\n"
        ).format(flag_marker)
        if needle in script:
            script = script.replace(needle, needle + insertion, 1)
        else:
            print("Could not identify Kroko variant logging in in_windows_container.sh; leaving wheel-only flag unchanged.")

    script = script.replace(
        "-DSHERPA_ONNX_ENABLE_WEBSOCKET=ON \\",
        "-DSHERPA_ONNX_ENABLE_WEBSOCKET=$KROKO_WEBSOCKET_FLAG \\",
    )

    gpu_flag = "    -DSHERPA_ONNX_ENABLE_GPU=OFF \\\n"
    if gpu_flag not in script:
        binary_flag = "    -DSHERPA_ONNX_ENABLE_BINARY=OFF \\\n"
        if binary_flag in script:
            script = script.replace(binary_flag, binary_flag + gpu_flag, 1)
        else:
            print("Could not identify Kroko wheel CMake flags; leaving GPU mode unchanged.")

    dll_marker = "WhoSpeaks patch: stage the wheel build's ONNX Runtime DLL."
    delvewheel_command = "python3 -m delvewheel repair \\\n"
    if dll_marker not in script and delvewheel_command in script:
        dll_staging = (
            "# {0}\n"
            "# The free wheel skips the websocket-server build, so INSTALL_DIR/bin\n"
            "# does not receive onnxruntime.dll through the server install step.\n"
            "# setup.py still downloads the CPU runtime below SRC_RW/build; find it\n"
            "# explicitly because delvewheel's --add-path is not recursive.\n"
            'ONNXRUNTIME_DLL=$(find "$SRC_RW/build" -type f -iname "onnxruntime.dll" | head -n 1)\n'
            'if [ -z "$ONNXRUNTIME_DLL" ]; then\n'
            '    echo "ERROR: onnxruntime.dll missing from the Kroko wheel build tree" >&2\n'
            "    exit 1\n"
            "fi\n"
            'mkdir -p "$INSTALL_DIR/bin"\n'
            'cp -f "$ONNXRUNTIME_DLL" "$INSTALL_DIR/bin/onnxruntime.dll"\n'
            'echo "Staged ONNX Runtime for delvewheel: $ONNXRUNTIME_DLL"\n'
            "\n"
        ).format(dll_marker)
        script = script.replace(delvewheel_command, dll_staging + delvewheel_command, 1)

    wrap_marker = 'if [ "$KROKO_WHEEL_ONLY" = "1" ]; then'
    if wrap_marker not in script:
        start = script.find("# Configure. The websocket server is gated by SHERPA_ONNX_ENABLE_WEBSOCKET")
        missing = 'kroko-onnx-online-websocket-server.exe missing from build output'
        missing_pos = script.find(missing, start)
        end = -1
        if missing_pos != -1:
            end_marker = "\nfi\n\n"
            end = script.find(end_marker, missing_pos)
            if end != -1:
                end += len(end_marker)
        if start != -1 and end != -1:
            server_block = script[start:end]
            replacement = (
                'if [ "$KROKO_WHEEL_ONLY" = "1" ]; then\n'
                '    echo "Skipping Windows websocket-server build for free wheel-only runtime."\n'
                '    mkdir -p "$INSTALL_DIR/bin"\n'
                "else\n"
                + server_block
                + "fi\n\n"
            )
            script = script[:start] + replacement + script[end:]
        else:
            print("Could not identify Kroko websocket-server build block; leaving it unchanged.")

    if script != script_original:
        write_text(script_path, script)
        print("Patched in_windows_container.sh for a free Windows wheel-only build.")

    batch = read_text(batch_path)
    batch = batch.replace("\r\r\n", "\r\n")
    batch_original = batch
    batch_marker = "WhoSpeaks patch: skip NSIS for free wheel-only build"
    batch = batch.replace(
        'call :size_h "!WHEEL!" _sz',
        "REM WhoSpeaks patch: avoid fragile cmd.exe wheel-size arithmetic",
    ).replace(
        "echo     Wheel: !WHEEL! ^(!_sz!^)",
        "echo     Wheel: !WHEEL!",
    )
    if batch_marker not in batch:
        step_marker = "REM -- Step 3: build the NSIS installer ---------------------------------------"
        newline = "\r\n" if "\r\n" in batch else "\n"
        insertion = newline.join(
            [
                "REM {0}".format(batch_marker),
                'if /I "%VARIANT%"=="free" (',
                "    echo.",
                "    echo [3/3] Skipping NSIS installer (free wheel-only build)",
                "    exit /b 0",
                ")",
                "",
            ]
        )
        if step_marker in batch:
            batch = batch.replace(step_marker, insertion + step_marker, 1)
        else:
            print("Could not identify NSIS build step in build_windows.bat; leaving it unchanged.")

    if batch != batch_original:
        write_text(batch_path, batch)
        print("Patched build_windows.bat to skip the free NSIS installer.")


def patch_windows_dockerfile(repo_dir, install_openssl=True):
    """
    Patches the Kroko Windows Dockerfile.
    """

    path = repo_dir / "Dockerfile.windows"
    if not path.exists():
        raise KrokoInstallError("Missing Kroko Windows Dockerfile: {0}".format(path))

    text = read_text(path)
    original = text

    old_lf = (
        "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
        "RUN chmod +x /usr/local/bin/in_windows_container.sh"
    )
    new_lf = (
        "COPY in_windows_container.sh /usr/local/bin/in_windows_container.sh\n"
        "RUN sed -i 's/\\r$//' /usr/local/bin/in_windows_container.sh \\\n"
        " && chmod +x /usr/local/bin/in_windows_container.sh"
    )
    old_crlf = old_lf.replace("\n", "\r\n")
    new_crlf = new_lf.replace("\n", "\r\n")
    if old_lf in text:
        text = text.replace(old_lf, new_lf)
        print("Patched Dockerfile.windows to tolerate CRLF shell scripts.")
    elif old_crlf in text:
        text = text.replace(old_crlf, new_crlf)
        print("Patched Dockerfile.windows to tolerate CRLF shell scripts.")

    if install_openssl:
        patched_text = patch_windows_dockerfile_openssl_extraction(text)
    else:
        patched_text = patch_windows_dockerfile_skip_openssl_install(text)
    if patched_text != text:
        text = patched_text
        if install_openssl:
            print("Patched Dockerfile.windows OpenSSL extraction for current Slproweb MSI layout.")
        else:
            print("Patched Dockerfile.windows to skip OpenSSL download for the free build.")

    if text != original:
        write_text(path, text)


def patch_windows_dockerfile_openssl_extraction(text):
    """
    Replaces the brittle OpenSSL extraction block in Kroko's Windows Dockerfile.
    """

    marker = "WhoSpeaks patch: robust OpenSSL extraction"
    if marker in text:
        return text

    start_marker = (
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        msitools innoextract"
    )
    start = text.find(start_marker)
    if start == -1 and "\r\n" in text:
        start_marker = start_marker.replace("\n", "\r\n")
        start = text.find(start_marker)
    if start == -1:
        print("Could not identify Dockerfile.windows OpenSSL install block; leaving it unchanged.")
        return text

    end_marker = " && rm -rf /tmp/openssl-msi /tmp/openssl-final /tmp/openssl.msi"
    end = text.find(end_marker, start)
    if end == -1:
        print("Could not identify Dockerfile.windows OpenSSL cleanup line; leaving it unchanged.")
        return text
    newline = "\r\n" if "\r\n" in text else "\n"
    line_end = text.find(newline, end)
    if line_end == -1:
        line_end = len(text)
    else:
        line_end += len(newline)

    replacement = newline.join(
        [
            "# WhoSpeaks patch: robust OpenSSL extraction for Slproweb MSI layouts.",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            "        msitools innoextract \\",
            " && rm -rf /var/lib/apt/lists/* \\",
            " && mkdir -p /tmp/openssl-msi /tmp/openssl-final /opt/openssl-win64/app/bin \\",
            "        /opt/openssl-win64/app/lib /opt/openssl-win64/app/include \\",
            " && cd /tmp \\",
            " && for v in 3_6_2 3_6_1 3_5_4 3_5_3; do \\",
            "        if curl -sLf \"https://slproweb.com/download/Win64OpenSSL-${v}.msi\" \\",
            "                -o openssl.msi; then \\",
            "            echo \"Downloaded Win64OpenSSL-${v}.msi\"; \\",
            "            break; \\",
            "        fi; \\",
            "        rm -f openssl.msi; \\",
            "    done \\",
            " && test -s openssl.msi \\",
            " && msiextract -C /tmp/openssl-msi openssl.msi \\",
            " && (find /tmp/openssl-msi -name \"*.exe\" -exec sh -c 'for exe do innoextract -d /tmp/openssl-final \"$exe\" || 7z x -y -o/tmp/openssl-final \"$exe\"; done' sh {} + || true) \\",
            " && for source in /tmp/openssl-msi /tmp/openssl-final /tmp/openssl-final/app; do \\",
            "        if test -d \"$source\"; then \\",
            "            find \"$source\" -iname \"libcrypto*.dll\" -exec cp -v {} /opt/openssl-win64/app/bin/ \\; || true; \\",
            "            find \"$source\" -iname \"libssl*.dll\" -exec cp -v {} /opt/openssl-win64/app/bin/ \\; || true; \\",
            "            find \"$source\" -iname \"libcrypto*.lib\" -exec cp -v {} /opt/openssl-win64/app/lib/ \\; || true; \\",
            "            find \"$source\" -iname \"libssl*.lib\" -exec cp -v {} /opt/openssl-win64/app/lib/ \\; || true; \\",
            "            find \"$source\" -type d -iname \"openssl\" -exec sh -c 'for dir do case \"$dir\" in */include/*|*/Include/*|*/INCLUDE/*) mkdir -p /opt/openssl-win64/app/include/openssl; cp -rv \"$dir\"/* /opt/openssl-win64/app/include/openssl/ ;; esac; done' sh {} + || true; \\",
            "        fi; \\",
            "    done \\",
            " && if test -d /opt/openssl-win64/app/lib/VC/x64/MT; then cp -v /opt/openssl-win64/app/lib/VC/x64/MT/* /opt/openssl-win64/app/lib/; fi \\",
            " && if test ! -f /opt/openssl-win64/app/lib/libcrypto.lib; then first=$(find /opt/openssl-win64/app/lib -maxdepth 1 -iname \"libcrypto*.lib\" | head -n 1); test -n \"$first\" && cp -v \"$first\" /opt/openssl-win64/app/lib/libcrypto.lib; fi \\",
            " && if test ! -f /opt/openssl-win64/app/lib/libssl.lib; then first=$(find /opt/openssl-win64/app/lib -maxdepth 1 -iname \"libssl*.lib\" | head -n 1); test -n \"$first\" && cp -v \"$first\" /opt/openssl-win64/app/lib/libssl.lib; fi \\",
            " && test -f /opt/openssl-win64/app/lib/libcrypto.lib \\",
            " && test -f /opt/openssl-win64/app/lib/libssl.lib \\",
            " && test -f /opt/openssl-win64/app/include/openssl/ssl.h \\",
            " && rm -rf /tmp/openssl-msi /tmp/openssl-final /tmp/openssl.msi",
        ]
    ) + newline
    return text[:start] + replacement + text[line_end:]


def patch_windows_dockerfile_skip_openssl_install(text):
    """
    Replaces the OpenSSL download block with a no-op directory setup.
    """

    marker = "WhoSpeaks patch: free Windows build skips OpenSSL download"
    if marker in text:
        return text

    start_marker = (
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        msitools innoextract"
    )
    start = text.find(start_marker)
    if start == -1 and "\r\n" in text:
        start_marker = start_marker.replace("\n", "\r\n")
        start = text.find(start_marker)
    if start == -1:
        print("Could not identify Dockerfile.windows OpenSSL install block; leaving it unchanged.")
        return text

    end_marker = " && rm -rf /tmp/openssl-msi /tmp/openssl-final /tmp/openssl.msi"
    end = text.find(end_marker, start)
    if end == -1:
        print("Could not identify Dockerfile.windows OpenSSL cleanup line; leaving it unchanged.")
        return text
    newline = "\r\n" if "\r\n" in text else "\n"
    line_end = text.find(newline, end)
    if line_end == -1:
        line_end = len(text)
    else:
        line_end += len(newline)

    replacement = newline.join(
        [
            "# WhoSpeaks patch: free Windows build skips OpenSSL download.",
            "RUN mkdir -p /opt/openssl-win64/app/bin \\",
            "        /opt/openssl-win64/app/lib /opt/openssl-win64/app/include",
        ]
    ) + newline
    return text[:start] + replacement + text[line_end:]


def _insert_after_line(text, line_text, insertion):
    """
    Inserts text after a matching source line.
    """

    lines = text.splitlines(True)
    for index, line in enumerate(lines):
        if line.strip() == line_text:
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            if insertion in text:
                return text
            lines.insert(index + 1, insertion.replace("\n", newline))
            return "".join(lines)
    return text


def _wrap_license_output_line(text, marker):
    """
    Wraps license output behind the quiet-mode environment flag.
    """

    lines = text.splitlines(True)
    changed = False
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        if "std::cout" not in line and "std::cerr" not in line:
            continue
        previous = "".join(lines[max(0, index - 2):index])
        if "KrokoSuppressLicenseOutput" in previous:
            continue

        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        content = line.rstrip("\r\n")
        indent = content[:len(content) - len(content.lstrip())]
        statement = content[len(indent):]
        lines[index] = (
            indent + "if (!KrokoSuppressLicenseOutput()) {" + newline
            + indent + "    " + statement + newline
            + indent + "}" + newline
        )
        changed = True

    if not changed:
        return text
    return "".join(lines)


def patch_license_quiet_env(repo_dir):
    """
    Patches Kroko sources to support quiet license output.
    """

    path = repo_dir / "sherpa-onnx" / "csrc" / "license.h"
    if not path.exists():
        print("Could not find Kroko license client source; native license logs may remain noisy.")
        return

    text = read_text(path)
    original = text

    if "#include <cstdlib>" not in text:
        text = _insert_after_line(text, "#include <chrono>", "#include <cstdlib>\n")
    if "#include <windows.h>" not in text:
        text = _insert_after_line(
            text,
            "#include <cstdlib>",
            "#ifdef _WIN32\n#include <windows.h>\n#endif\n",
        )

    helper = (
        "inline std::string KrokoLicenseQuietEnvValue() {\n"
        "#ifdef _WIN32\n"
        "    char buffer[64];\n"
        "    DWORD size = GetEnvironmentVariableA(\n"
        "        \"" + KROKO_LICENSE_QUIET_ENV + "\",\n"
        "        buffer,\n"
        "        static_cast<DWORD>(sizeof(buffer)));\n"
        "    if (size > 0) {\n"
        "        if (size < sizeof(buffer)) {\n"
        "            return std::string(buffer, size);\n"
        "        }\n"
        "        return \"1\";\n"
        "    }\n"
        "#endif\n"
        "\n"
        "    const char* value = std::getenv(\"" + KROKO_LICENSE_QUIET_ENV + "\");\n"
        "    if (value == nullptr) {\n"
        "        return \"\";\n"
        "    }\n"
        "    return std::string(value);\n"
        "}\n\n"
        "inline bool KrokoSuppressLicenseOutput() {\n"
        "    std::string text = KrokoLicenseQuietEnvValue();\n"
        "    return !text.empty() && text != \"0\" && text != \"false\" && text != \"False\" && text != \"FALSE\";\n"
        "}\n\n"
    )
    helper_start = text.find("inline bool KrokoSuppressLicenseOutput() {")
    if helper_start != -1:
        block_start = text.rfind("\ninline ", 0, helper_start)
        if block_start == -1:
            block_start = helper_start
        else:
            block_start += 1
        block_end = text.find("struct Feature {", helper_start)
        if block_end != -1:
            text = text[:block_start] + helper + text[block_end:]
    elif KROKO_LICENSE_QUIET_ENV not in text:
        text = _insert_after_line(text, "using json = nlohmann::json;", "\n" + helper)

    for marker in (
        "License not allowed:",
        "License accepted. Remaining seconds:",
        "Usage report error:",
        "Remaining seconds updated:",
        "JSON parse error:",
        "Connected to license server.",
        "Connection closed.",
        "Connection failed.",
        "Failed to create connection:",
        "Retrying connection in 3s...",
        "Cannot send usage: license not allowed.",
        "No active WebSocket connection.",
        "Failed to send usage report:",
        "Offline timeout exceeded (",
    ):
        text = _wrap_license_output_line(text, marker)

    if text != original:
        write_text(path, text)
        print("Patched Kroko license client to honor {0}.".format(KROKO_LICENSE_QUIET_ENV))


def patch_linux_free_build_without_openssl_dev(repo_dir):
    """
    Lets the free Linux build avoid OpenSSL development headers.
    """

    cmake_path = repo_dir / "sherpa-onnx" / "csrc" / "CMakeLists.txt"
    model_path = repo_dir / "sherpa-onnx" / "csrc" / "ModelData.cc"
    transducer_path = repo_dir / "sherpa-onnx" / "csrc" / "online-transducer-model.cc"
    extension_path = repo_dir / "cmake" / "cmake_extension.py"

    if cmake_path.exists():
        text = read_text(cmake_path).replace("\r\n", "\n")
        text = text.replace(
            "find_package(OpenSSL REQUIRED)\n",
            "if(KROKO_LICENSE)\n  find_package(OpenSSL REQUIRED)\nendif()\n",
        )
        old_link = (
            "target_link_libraries(kroko-onnx-core\n"
            "  kaldi-native-fbank-core\n"
            "  kaldi-decoder-core\n"
            "  ssentencepiece_core\n"
            "  OpenSSL::SSL\n"
            "  OpenSSL::Crypto\n"
            ")"
        )
        new_link = (
            "target_link_libraries(kroko-onnx-core\n"
            "  kaldi-native-fbank-core\n"
            "  kaldi-decoder-core\n"
            "  ssentencepiece_core\n"
            ")\n"
            "if(KROKO_LICENSE)\n"
            "  target_link_libraries(kroko-onnx-core\n"
            "    OpenSSL::SSL\n"
            "    OpenSSL::Crypto\n"
            "  )\n"
            "endif()"
        )
        text = text.replace(old_link, new_link)
        write_text(cmake_path, text)

    if transducer_path.exists():
        text = read_text(transducer_path).replace("\r\n", "\n")
        text = text.replace(
            '#include "sherpa-onnx/csrc/license.h"\n',
            '#ifdef KROKO_LICENSE\n#include "sherpa-onnx/csrc/license.h"\n#endif\n',
        )
        write_text(transducer_path, text)

    if model_path.exists():
        text = read_text(model_path).replace("\r\n", "\n")
        text = text.replace(
            "#include <openssl/aes.h>\n#include <openssl/evp.h>\n#include <openssl/rand.h>\n",
            "#ifdef KROKO_LICENSE\n"
            "#include <openssl/aes.h>\n#include <openssl/evp.h>\n#include <openssl/rand.h>\n"
            "#endif\n",
        )
        start = text.find("bool ModelData::decryptPayload")
        end = text.find("\nbool ModelData::loadPayload()", start)
        if start != -1 and end != -1:
            block = text[start:end]
            if "#ifdef KROKO_LICENSE" not in block:
                text = (
                    text[:start]
                    + "#ifdef KROKO_LICENSE\n"
                    + block
                    + "#else\n"
                    + "bool ModelData::decryptPayload(const std::string&) {\n"
                    + "    return false;\n"
                    + "}\n"
                    + "#endif\n"
                    + text[end:]
                )
        write_text(model_path, text)

    if extension_path.exists():
        text = read_text(extension_path).replace("\r\n", "\n")
        needle = "    if enable_alsa():\n        binaries += [\n"
        insertion = (
            '    if os.environ.get("SHERPA_ONNX_ENABLE_WEBSOCKET", "ON").upper() in {"0", "OFF", "FALSE", "NO"}:\n'
            "        binaries = [\n"
            "            item for item in binaries\n"
            '            if "websocket" not in item and "kroko-onnx-online-websocket-server" not in item\n'
            "        ]\n\n"
        )
        if insertion not in text and needle in text:
            text = text.replace(needle, insertion + needle)
            write_text(extension_path, text)

    print("Patched Kroko free Linux build to avoid OpenSSL development headers.")


def prepare_windows_checkout(repo_dir, variant="free"):
    """
    Prepares Windows-specific Kroko build files.
    """

    script = repo_dir / "in_windows_container.sh"
    if not script.exists():
        raise KrokoInstallError("Missing Kroko container build script: {0}".format(script))
    normalize_lf(script)
    patch_windows_bat(repo_dir)
    sanitize_batch_ascii(repo_dir / "build_windows.bat")
    if variant == "free":
        patch_linux_free_build_without_openssl_dev(repo_dir)
        patch_windows_free_wheel_only_build(repo_dir)
    patch_windows_dockerfile(repo_dir, install_openssl=(variant != "free"))
    patch_license_quiet_env(repo_dir)


def ensure_windows_host():
    """
    Verifies that the current host is Windows.
    """

    if sys.version_info[:2] != (3, 12):
        raise KrokoInstallError(
            "Kroko's current Windows wheel build targets CPython 3.12 x64.\n"
            "Your active Python is {0}.{1}.{2} ({3}-bit).\n"
            "RealtimeSTT core supports Python 3.11+, but this Kroko Windows "
            "builder path does not.\n"
            "Create and activate a Python 3.12 x64 environment, install "
            "RealtimeSTT[kroko-builder] there, then rerun:\n"
            "    stt-install-kroko --build".format(
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
                64 if sys.maxsize > 2 ** 32 else 32,
            )
        )
    if sys.maxsize <= 2 ** 32:
        raise KrokoInstallError(
            "Kroko's current Windows wheel build targets CPython 3.12 x64.\n"
            "Your active Python is 32-bit.\n"
            "Create and activate a 64-bit Python 3.12 environment, install "
            "RealtimeSTT[kroko-builder] there, then rerun:\n"
            "    stt-install-kroko --build"
        )
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        raise KrokoInstallError(
            "Kroko's current Windows wheel build targets win_amd64; "
            "this machine reports {0}.".format(platform.machine())
        )
    ensure_program(
        "docker",
        "Docker Desktop is required on Windows. Install Docker Desktop, start it "
        "with the WSL2 backend enabled, then retry.",
    )
    try:
        run(["docker", "version"])
    except KrokoInstallError:
        raise KrokoInstallError(
            "Docker Desktop is not running or its Linux engine is unavailable.\n"
            "Start Docker Desktop, wait until it reports that Docker is running, "
            "then retry:\n"
            "    stt-install-kroko --build\n"
            "You can verify it manually with:\n"
            "    docker version"
        )


def find_windows_wheel(repo_dir, variant):
    """
    Finds the built Kroko Windows wheel.
    """

    tag = "cp{0}{1}".format(sys.version_info.major, sys.version_info.minor)
    wheel_dir = repo_dir / "release_artifacts" / "windows"
    patterns = [
        "kroko_onnx-*-1{0}-{1}-{1}-win_amd64.whl".format(variant, tag),
        "kroko_onnx-*-{0}-{1}-{1}-win_amd64.whl".format(variant, tag),
        "kroko_onnx-*-{0}-{0}-win_amd64.whl".format(tag),
    ]
    wheels = []
    for pattern in patterns:
        wheels.extend(wheel_dir.glob(pattern))
    wheels = sorted(set(wheels), key=lambda item: item.stat().st_mtime, reverse=True)
    if not wheels:
        raise KrokoInstallError(
            "Windows build finished, but no Kroko wheel matching {0}/{1} was found in {2}.".format(
                variant,
                tag,
                wheel_dir,
            )
        )
    return wheels[0]


def install_windows(args, repo_dir):
    """
    Builds and installs Kroko on Windows.
    """

    ensure_windows_host()
    prepare_windows_checkout(repo_dir, args.variant)
    run(["cmd.exe", "/c", str(repo_dir / "build_windows.bat"), "--variant", args.variant], cwd=repo_dir)
    wheel = find_windows_wheel(repo_dir, args.variant)
    print("Built Kroko-ONNX wheel: {0}".format(wheel))
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)])


def install_linux(args, repo_dir):
    """
    Installs Kroko from source on Linux.
    """

    ensure_program("cmake", "CMake is required to build Kroko-ONNX from source on Linux.")
    patch_license_quiet_env(repo_dir)
    env = os.environ.copy()
    if args.variant == "pro":
        env["KROKO_LICENSE"] = "ON"
    else:
        patch_linux_free_build_without_openssl_dev(repo_dir)
        env["SHERPA_ONNX_ENABLE_WEBSOCKET"] = "OFF"
        cmake_args = env.get("SHERPA_ONNX_CMAKE_ARGS", "").strip()
        if not cmake_args:
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"
        if "SHERPA_ONNX_ENABLE_WEBSOCKET" not in cmake_args:
            cmake_args = cmake_args + " -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF"
        env["SHERPA_ONNX_CMAKE_ARGS"] = cmake_args

    if args.skip_install:
        wheel_dir = repo_dir / "release_artifacts" / "linux"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
            ],
            cwd=repo_dir,
            env=env,
        )
        return

    run([sys.executable, "-m", "pip", "install", "."], cwd=repo_dir, env=env)


def main(argv=None):
    """
    Runs the Kroko installer command.
    """

    args = parse_args(argv)
    if not args.build:
        raise SystemExit("Pass --build to build and install Kroko-ONNX.")

    try:
        work_dir = preflight_build(args)
        repo_dir = prepare_checkout(args, work_dir)
        if os.name == "nt":
            install_windows(args, repo_dir)
        elif sys.platform.startswith("linux"):
            install_linux(args, repo_dir)
        else:
            raise KrokoInstallError(
                "stt-install-kroko currently supports Windows and Linux. "
                "Use Kroko's upstream macOS build script on macOS."
            )
    except KrokoInstallError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 1

    print("Kroko-ONNX is ready in this Python environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
