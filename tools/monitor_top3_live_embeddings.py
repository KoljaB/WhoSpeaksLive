"""Readable terminal monitor for the remote top-three live-window campaign.

Run from Windows with:
    python tools/monitor_top3_live_embeddings.py

The monitor is read-only. It asks the configured remote wrapper for the three
small JSON progress files and redraws a compact dashboard every two minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_WRAPPER = Path(r"D:\Projekte\remote-codex_cli\remote-codex_cli\tools\codex_remote_cli.py")
DEFAULT_SSH_KEY = Path(r"C:\Users\Start\.ssh\linuxremote_codex_ed25519")
DEFAULT_SSH_HOST = "lon@192.168.178.22"
REMOTE_CWD = "/home/lon/speakerdiar_opt/live_window_pilot_20260719"
REMOTE_ROOT = "runtime/optimization/live_shifting_windows_v1/campaigns"
REMOTE_ROOT_ABS = f"{REMOTE_CWD}/{REMOTE_ROOT}"
VIDEOS = ("JWS-qfR6K3w", "DsyfYJ5Ou3g", "20v1OxUXcQY")
FULL_AUDIO_SECONDS = {"JWS-qfR6K3w": 303.891, "DsyfYJ5Ou3g": 212.822, "20v1OxUXcQY": 346.093}
WINDOW_SECONDS = tuple(index / 10 for index in range(7, 31))
HOP_SECONDS = 0.2
TITLES = {
    "JWS-qfR6K3w": "Cunk on Earth",
    "DsyfYJ5Ou3g": "Gordon tries to make Pad Thai",
    "20v1OxUXcQY": "Simon Pegg / Benedict Cumberbatch",
}


@dataclass
class Snapshot:
    video_id: str
    data: dict[str, Any] | None = None
    error: str = ""


def fetch(wrapper: Path, video_id: str, session: str, transport: str, ssh_key: Path, ssh_host: str) -> Snapshot:
    remote_file = f"{REMOTE_ROOT}/top3_{video_id}/progress.json"
    command = f"cat {remote_file}"
    try:
        if transport == "ssh":
            result = subprocess.run(
                ["ssh", "-i", str(ssh_key), "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8",
                 ssh_host, f"cat {REMOTE_ROOT_ABS}/top3_{video_id}/progress.json"], capture_output=True, text=True, timeout=20, check=False)
            if result.returncode != 0:
                return Snapshot(video_id, error=result.stderr.strip() or "ssh command failed")
            return Snapshot(video_id, data=json.loads(result.stdout))
        result = subprocess.run(
            [sys.executable, str(wrapper), "--json", "--session", session,
             "--cwd", REMOTE_CWD, "--reason",
             "Read live-window campaign progress for the local dashboard.", command],
            capture_output=True, text=True, timeout=45, check=False)
        envelope = json.loads(result.stdout)
        if envelope.get("exit_code") != 0:
            return Snapshot(video_id, error=envelope.get("output", "remote command failed"))
        return Snapshot(video_id, data=json.loads(envelope.get("output", "{}")))
    except Exception as exc:  # dashboard should remain alive across transient outages
        return Snapshot(video_id, error=f"{type(exc).__name__}: {exc}")


def fmt_seconds(value: float | int | None) -> str:
    if value is None or value < 0:
        return "—"
    seconds = int(round(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def bar(percent: float, width: int = 24) -> str:
    percent = max(0.0, min(100.0, float(percent or 0.0)))
    filled = int(round(width * percent / 100.0))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def expected_for_video(video_id: str) -> int:
    duration = FULL_AUDIO_SECONDS[video_id]
    return sum(int((duration - length) // HOP_SECONDS) + 1 for length in WINDOW_SECONDS)


def clear_screen() -> None:
    # ANSI works in Windows Terminal and modern PowerShell; fall back gracefully.
    sys.stdout.write("\x1b[2J\x1b[H")


def render(snapshots: list[Snapshot], interval: int) -> None:
    clear_screen()
    terminal_width = shutil.get_terminal_size((120, 30)).columns
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print("TOP-3 LIVE-EMBEDDINGS  |  read-only monitor")
    print(f"Letztes Update: {now}  |  Nächstes Update: {interval}s  |  Ctrl+C beendet nur den Monitor")
    print("-" * min(terminal_width, 120))

    total_done = 0
    total_expected = sum(expected_for_video(video_id) * 15 for video_id in VIDEOS)
    for snap in snapshots:
        if not snap.data:
            if "No such file" in snap.error or "cannot stat" in snap.error or snap.error.lstrip().startswith("cat:"):
                print(f"{snap.video_id:12}  {TITLES.get(snap.video_id, ''):32}  wartet auf Kampagnenstart ...")
            else:
                print(f"{snap.video_id:12}  {TITLES.get(snap.video_id, ''):32}  FEHLER: {snap.error[:terminal_width-52]}")
            continue
        data = snap.data
        done = int(data.get("completed_embeddings", 0) or 0)
        expected = int(data.get("expected_embeddings_all_providers", 0) or 0)
        if expected <= 0:
            expected = expected_for_video(snap.video_id) * 15
        total_done += done
        provider = data.get("current_provider") or "—"
        current = data.get("current_provider_progress") or {}
        provider_pct = float(current.get("percent", 0) or 0)
        decided = data.get("providers_decided", 0)
        count = data.get("provider_count", 15)
        status = data.get("status", "?")
        eta = fmt_seconds(current.get("eta_seconds"))
        rate = current.get("embeddings_per_second")
        rate_text = f"{float(rate):.0f}/s" if rate else "—"
        data_pct = float(data.get("data_percent", 0) or 0)
        print(f"{snap.video_id:12}  {TITLES.get(snap.video_id, '')[:32]:32}  {bar(data_pct)} {data_pct:6.2f}%  {status}")
        print(f"  Provider {decided + (1 if status == 'running' else 0):02d}/{count:02d}: {provider:32} {provider_pct:6.2f}%  {rate_text:>7}  ETA {eta}")

    overall = 100.0 * total_done / total_expected if total_expected else 0.0
    print("-" * min(terminal_width, 120))
    print(f"GESAMT  {bar(overall, 32)} {overall:6.2f}%   {total_done:,}/{total_expected:,} Embeddings".replace(",", "."))
    print("Hinweis: Die Kampagne läuft providerweise; ein fertiger Provider wird entladen, bevor der nächste startet.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=10, help="Abfrageintervall in Sekunden (Standard: 10)")
    parser.add_argument("--once", action="store_true", help="Nur eine Abfrage ausführen und beenden")
    parser.add_argument("--session", default="main", choices=("main", "diagnostic", "recovery"))
    parser.add_argument("--transport", choices=("ssh", "wrapper"), default="ssh", help="Remote-Transport (Standard: ssh)")
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval muss mindestens 5 Sekunden betragen")
    if args.transport == "wrapper" and not args.wrapper.exists():
        parser.error(f"Remote-Wrapper nicht gefunden: {args.wrapper}")
    if args.transport == "ssh" and not args.ssh_key.exists():
        parser.error(f"SSH-Schlüssel nicht gefunden: {args.ssh_key}")

    try:
        while True:
            snapshots = [fetch(args.wrapper, video_id, args.session, args.transport, args.ssh_key, args.ssh_host) for video_id in VIDEOS]
            render(snapshots, args.interval)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor beendet; der Remote-Job läuft unverändert weiter.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
