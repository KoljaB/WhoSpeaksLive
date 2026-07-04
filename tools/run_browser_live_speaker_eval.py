"""Run the real browser UI and score the rendered live speaker tag."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "tools" / "youtube_window_diarize_gui.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Start youtube_window_diarize_gui.py, drive the real browser UI, "
            "and write a browser-observed live-speaker score."
        )
    )
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "validation" / "browser-live-speaker-observed.json",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=ROOT / "runtime" / "validation" / "browser-live-speaker-server.log",
    )
    parser.add_argument("--timeout-seconds", type=float, default=480.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_known_args()


def stream_server_output(process: subprocess.Popen[str], log_path: Path, ready: threading.Event, url_holder: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            text = line.strip()
            if text.startswith("Serving growing-window diarization GUI at "):
                url_holder["url"] = text.rsplit(" ", 1)[-1]
                ready.set()


def main() -> int:
    args, passthrough = parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is required for the automated browser evaluator. "
            "Install it in this venv and run `python -m playwright install chromium`.",
            file=sys.stderr,
        )
        return 2

    output = args.output.resolve()
    command = [
        sys.executable,
        str(GUI),
        "--port",
        str(args.port),
        "--no-browser",
        "--browser-live-observation-output",
        str(output),
        *passthrough,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ready = threading.Event()
    url_holder: dict[str, str] = {}
    reader = threading.Thread(
        target=stream_server_output,
        args=(process, args.server_log.resolve(), ready, url_holder),
        daemon=True,
    )
    reader.start()
    try:
        if not ready.wait(timeout=min(180.0, args.timeout_seconds)):
            raise RuntimeError(f"GUI server did not become ready; see {args.server_log.resolve()}")
        page_url = url_holder["url"]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=bool(args.headless))
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(page_url, wait_until="domcontentloaded")
            page.click("#start")
            page.wait_for_function(
                "() => document.getElementById('state')?.textContent === 'Stopped'",
                timeout=args.timeout_seconds * 1000,
            )
            browser.close()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not output.exists():
            time.sleep(0.25)
        if not output.exists():
            raise RuntimeError(f"Browser live observation output was not written: {output}")
        print(f"Browser live speaker observation written to {output}", flush=True)
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
