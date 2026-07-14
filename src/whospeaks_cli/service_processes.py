"""Managed localhost HTTP service process helpers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from .cli_diagnostics import read_json_url
from .planning import ServiceProcessSpec, health_payload_matches


def _is_windows() -> bool:
    return os.name == "nt"


def service_health_ready(spec: ServiceProcessSpec) -> bool:
    ok, _detail, payload = read_json_url(spec.health_url, timeout_seconds=0.5)
    return bool(ok and health_payload_matches(spec, payload))


def start_service_process(spec: ServiceProcessSpec) -> subprocess.Popen[Any]:
    env = dict(os.environ)
    env.update(dict(spec.env))
    kwargs: dict[str, Any] = {"cwd": spec.cwd, "env": env}
    if _is_windows():
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(spec.command), **kwargs)


def wait_for_service_health(
    spec: ServiceProcessSpec,
    process: object | None = None,
) -> None:
    deadline = time.monotonic() + spec.readiness_timeout
    last_detail = "no health response"
    while time.monotonic() < deadline:
        poll = getattr(process, "poll", None)
        if callable(poll):
            return_code = poll()
            if return_code is not None:
                raise RuntimeError(f"{spec.name} exited with code {return_code} before becoming healthy")
        ok, detail, payload = read_json_url(spec.health_url, timeout_seconds=1.0)
        if ok and health_payload_matches(spec, payload):
            return
        last_detail = detail if not ok else f"health returned {payload}"
        time.sleep(0.25)
    raise RuntimeError(
        f"{spec.name} did not become healthy within {spec.readiness_timeout:g}s: "
        f"{spec.health_url} ({last_detail})"
    )


def terminate_service_processes(processes: list[object]) -> None:
    for process in reversed(processes):
        poll = getattr(process, "poll", None)
        if callable(poll):
            try:
                if poll() is not None and not _is_windows():
                    continue
            except (OSError, ProcessLookupError):
                continue
        try:
            if _is_windows():
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except (AttributeError, OSError):
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                try:
                    terminate()
                except (OSError, ProcessLookupError):
                    pass
        wait = getattr(process, "wait", None)
        if not callable(wait):
            continue
        try:
            wait(timeout=5)
            continue
        except subprocess.TimeoutExpired:
            pass
        except (OSError, ProcessLookupError):
            continue
        try:
            if _is_windows():
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (AttributeError, OSError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                try:
                    kill()
                except (OSError, ProcessLookupError):
                    pass
        try:
            wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
            pass
