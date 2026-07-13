from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Meeting Intelligence from an isolated wheel target."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18977)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    runtime_dir = args.runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    command = [
        sys.executable,
        "-m",
        "window.meeting_intelligence_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--session-dir",
        str(runtime_dir / "sessions"),
        "--cache-dir",
        str(runtime_dir / "cache"),
        "--chat-dir",
        str(runtime_dir / "chat"),
        "--text-index-db",
        str(runtime_dir / "index.sqlite3"),
        "--mock-llm",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=runtime_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    url = f"http://127.0.0.1:{args.port}/api/config"
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urlopen(url, timeout=1.0) as response:
                    payload = json.load(response)
                config = payload["config"]
                print(
                    "SERVICE_READY",
                    f"provider={config['provider']}",
                    "ask_routes=installed",
                    f"embedding_configured={config['text_embedding']['configured']}",
                )
                return 0
            except (OSError, URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.2)
        stdout, stderr = process.communicate(timeout=2.0)
        raise RuntimeError(
            "Installed-wheel service did not become ready.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
