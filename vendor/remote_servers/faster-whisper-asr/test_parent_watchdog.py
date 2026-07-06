#!/usr/bin/env python3
"""Self-check: an orphaned server process exits via the parent watchdog.

Run with the server venv: .venv/bin/python test_parent_watchdog.py
Spawns a middleman that launches a watchdog-armed child and then dies,
orphaning it; the child must exit within one poll cycle.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

INNER = "import asr_server, time; asr_server.start_parent_watchdog(); time.sleep(60)"
MIDDLE = (
    "import subprocess, sys, time\n"
    f"p = subprocess.Popen([sys.executable, '-c', {INNER!r}])\n"
    "print(p.pid, flush=True)\n"
    "time.sleep(3)\n"
)


def main() -> int:
    mid = subprocess.Popen(
        [sys.executable, "-c", MIDDLE],
        stdout=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    child_pid = int(mid.stdout.readline())
    mid.wait()

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            print("PASS: orphaned process exited via watchdog")
            return 0
        time.sleep(0.5)

    os.kill(child_pid, 9)
    print("FAIL: watchdog did not fire within 10s of orphaning")
    return 1


if __name__ == "__main__":
    sys.exit(main())
