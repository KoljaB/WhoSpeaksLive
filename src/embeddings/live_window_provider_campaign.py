"""Run isolated live-window corpus jobs for a sequence of providers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common.audio_utils import SAMPLE_RATE, load_audio_file
from embeddings.embedding_providers import SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS
from embeddings.live_window_experiment_plan import (
    DEFAULT_HOP_SECONDS,
    FULL_WINDOW_UNIVERSE_SECONDS,
    full_window_count,
    seconds_to_samples,
)


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"complete", "complete_existing", "failed", "timed_out"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_") or "provider"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _job_dir(output_root: Path, provider: str, video_id: str) -> Path:
    return output_root / "providers" / _safe_slug(provider) / "videos" / _safe_slug(video_id)


def _expected_embeddings(audio_path: Path, window_seconds: Sequence[Any], hop_seconds: Any) -> int:
    audio, _sample_rate = load_audio_file(audio_path, sample_rate=SAMPLE_RATE)
    hop_samples = seconds_to_samples(hop_seconds, sample_rate=SAMPLE_RATE)
    return sum(
        full_window_count(
            total_samples=int(audio.size),
            hop_samples=hop_samples,
            window_samples=seconds_to_samples(window, sample_rate=SAMPLE_RATE),
        )
        for window in window_seconds
    )


def _terminate_process_tree(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait(timeout=grace_seconds)


def _server_unload(base_url: str, provider: str, device: str) -> dict[str, Any]:
    query = urlencode({"provider": provider, "device": device})
    request = Request(f"{base_url.rstrip('/')}/unload?{query}", data=b"", method="POST")
    with urlopen(request, timeout=30.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict) or value.get("ok") is False:
        raise RuntimeError(f"provider unload failed: {value!r}")
    return value


def run_provider_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
    on_poll: Callable[[], None] | None = None,
) -> tuple[int | None, float, bool]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        start_new_session=os.name == "posix",
    )
    timed_out = False
    deadline = started + timeout_seconds
    return_code: int | None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _terminate_process_tree(process)
            return_code = None
            break
        try:
            return_code = process.wait(timeout=min(poll_seconds, remaining))
            break
        except subprocess.TimeoutExpired:
            if on_poll is not None:
                on_poll()
    elapsed = time.monotonic() - started
    if process.poll() is None:
        raise RuntimeError("provider process remained alive after wait/termination")
    return return_code, elapsed, timed_out


def _campaign_payload(
    *,
    providers: Sequence[str],
    results: dict[str, dict[str, Any]],
    expected_per_provider: int,
    current_provider: str | None,
    current_job_progress: dict[str, Any] | None,
    timeout_seconds: float,
    started_at: str,
) -> dict[str, Any]:
    decided = sum(1 for provider in providers if results.get(provider, {}).get("status") in TERMINAL_STATUSES)
    completed = sum(
        int((current_job_progress if provider == current_provider else results.get(provider, {})).get("completed_embeddings", 0))
        for provider in providers
    )
    total_expected = expected_per_provider * len(providers)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if decided == len(providers) else "running",
        "started_at": started_at,
        "updated_at": _utc_now(),
        "provider_timeout_seconds": timeout_seconds,
        "provider_count": len(providers),
        "providers_decided": decided,
        "execution_percent": round(100.0 * decided / len(providers), 6),
        "expected_embeddings_per_provider": expected_per_provider,
        "expected_embeddings_all_providers": total_expected,
        "completed_embeddings": completed,
        "data_percent": round(100.0 * completed / total_expected, 6),
        "current_provider": current_provider,
        "current_provider_progress": current_job_progress or {},
        "results": [results.get(provider, {"provider": provider, "status": "pending"}) for provider in providers],
    }


def run_campaign(
    *,
    root: Path,
    audio_path: Path,
    video_id: str,
    output_root: Path,
    providers: Sequence[str],
    python_executable: Path,
    timeout_seconds: float = 300.0,
    device: str = "cuda",
    block_rows: int = 32,
    poll_seconds: float = 1.0,
    process_runner: Callable[..., tuple[int | None, float, bool]] = run_provider_process,
    provider_backend: str = "local",
    provider_endpoint: str = "",
    campaign_name: str = "",
    server_unloader: Callable[[str, str, str], dict[str, Any]] = _server_unload,
    allow_resume_builder_code_change: bool = False,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if not providers:
        raise ValueError("at least one provider is required")
    if provider_backend not in {"local", "server"}:
        raise ValueError("provider_backend must be 'local' or 'server'")
    if provider_backend == "server" and not provider_endpoint.strip():
        raise ValueError("provider_endpoint is required for the server backend")
    providers = tuple(dict.fromkeys(str(value).strip() for value in providers if str(value).strip()))
    expected_per_provider = _expected_embeddings(audio_path, FULL_WINDOW_UNIVERSE_SECONDS, DEFAULT_HOP_SECONDS)
    campaign_dir = output_root / "campaigns" / _safe_slug(campaign_name or video_id)
    progress_path = campaign_dir / "progress.json"
    results_path = campaign_dir / "results.json"
    previous = _read_json(results_path)
    results = {
        str(item["provider"]): dict(item)
        for item in previous.get("results", [])
        if isinstance(item, dict) and item.get("provider")
    }
    started_at = str(previous.get("started_at") or _utc_now())

    def publish(current_provider: str | None = None) -> dict[str, Any]:
        job_progress = _read_json(_job_dir(output_root, current_provider, video_id) / "progress.json") if current_provider else {}
        payload = _campaign_payload(
            providers=providers,
            results=results,
            expected_per_provider=expected_per_provider,
            current_provider=current_provider,
            current_job_progress=job_progress,
            timeout_seconds=timeout_seconds,
            started_at=started_at,
        )
        _write_json_atomic(progress_path, payload)
        print(
            f"[campaign] execution={payload['execution_percent']:.2f}% "
            f"data={payload['data_percent']:.2f}% provider={current_provider or '-'} "
            f"provider_progress={job_progress.get('percent', 0.0):.2f}%",
            flush=True,
        )
        return payload

    publish()
    builder = root / "tools" / "build_live_shifting_window_corpus.py"
    for ordinal, provider in enumerate(providers, start=1):
        job_dir = _job_dir(output_root, provider, video_id)
        existing_job = _read_json(job_dir / "job.json")
        existing_progress = _read_json(job_dir / "progress.json")
        if existing_job.get("status") == "complete" and existing_progress.get("status") == "complete":
            results[provider] = {
                "provider": provider,
                "status": "complete_existing",
                "wall_seconds": float(existing_progress.get("elapsed_seconds", 0.0)),
                "provider_load_seconds": float(existing_progress.get("provider_load_seconds", 0.0)),
                "completed_embeddings": int(existing_progress.get("completed_embeddings", expected_per_provider)),
                "failed_embeddings": int(existing_progress.get("failed_embeddings", 0)),
                "job_progress_path": str((job_dir / "progress.json").relative_to(output_root)),
            }
            print(f"[campaign] provider {ordinal}/{len(providers)} {provider}: already complete", flush=True)
            publish()
            continue
        if results.get(provider, {}).get("status") == "timed_out":
            print(f"[campaign] provider {ordinal}/{len(providers)} {provider}: previously timed out, skipped", flush=True)
            publish()
            continue

        command = [
            str(python_executable), "-B", str(builder),
            "--audio", str(audio_path),
            "--video-id", video_id,
            "--provider", provider,
            "--device", device,
            "--output-root", str(output_root),
            "--block-rows", str(block_rows),
        ]
        if provider_backend == "server":
            command.extend(["--provider-backend", "server", "--provider-endpoint", provider_endpoint])
        if allow_resume_builder_code_change:
            command.append("--allow-resume-builder-code-change")
        print(f"[campaign] provider {ordinal}/{len(providers)} {provider}: starting, timeout={timeout_seconds:.1f}s", flush=True)
        results[provider] = {
            "provider": provider,
            "status": "running",
            "completed_embeddings": int(existing_progress.get("completed_embeddings", 0)),
        }
        publish(provider)
        unload_result: dict[str, Any] = {}
        try:
            return_code, wall_seconds, timed_out = process_runner(
                command,
                cwd=root,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                on_poll=lambda: publish(provider),
            )
        finally:
            if provider_backend == "server":
                unload_result = server_unloader(provider_endpoint, provider, device)
        job_progress = _read_json(job_dir / "progress.json")
        if timed_out:
            status = "timed_out"
            if job_progress:
                job_progress["status"] = "timed_out"
                job_progress["updated_at"] = _utc_now()
                _write_json_atomic(job_dir / "progress.json", job_progress)
        elif return_code == 0 and job_progress.get("status") == "complete":
            status = "complete"
        else:
            status = "failed"
        results[provider] = {
            "provider": provider,
            "status": status,
            "return_code": return_code,
            "wall_seconds": round(wall_seconds, 6),
            "provider_load_seconds": float(job_progress.get("provider_load_seconds", 0.0)),
            "completed_embeddings": int(job_progress.get("completed_embeddings", 0)),
            "failed_embeddings": int(job_progress.get("failed_embeddings", 0)),
            "percent": float(job_progress.get("percent", 0.0)),
            "job_progress_path": str((job_dir / "progress.json").relative_to(output_root)),
            "provider_backend": provider_backend,
            "unload": unload_result,
        }
        print(
            f"[campaign] provider {provider}: {status} wall={wall_seconds:.2f}s "
            f"embeddings={results[provider]['completed_embeddings']}/{expected_per_provider}",
            flush=True,
        )
        publish()

    payload = publish()
    payload["completed_at"] = _utc_now()
    _write_json_atomic(progress_path, payload)
    _write_json_atomic(results_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all live-window providers in isolated timeout-bounded processes.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--providers", default=",".join(SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--provider-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--provider-backend", choices=("local", "server"), default="local")
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--campaign-name", default="")
    parser.add_argument("--allow-resume-builder-code-change", action="store_true")
    args = parser.parse_args(argv)
    providers = [value.strip() for value in args.providers.split(",") if value.strip()]
    result = run_campaign(
        root=Path(__file__).resolve().parents[2],
        audio_path=args.audio.resolve(),
        video_id=args.video_id,
        output_root=args.output_root.resolve(),
        providers=providers,
        # Keep a virtual-environment interpreter symlink intact. Path.resolve()
        # would turn e.g. `.venv/bin/python` into `/usr/bin/python3`, silently
        # discarding the venv and all provider dependencies in child processes.
        python_executable=Path(os.path.abspath(args.python)),
        timeout_seconds=args.provider_timeout_seconds,
        device=args.device,
        block_rows=args.block_rows,
        provider_backend=args.provider_backend,
        provider_endpoint=args.provider_endpoint,
        campaign_name=args.campaign_name,
        allow_resume_builder_code_change=args.allow_resume_builder_code_change,
    )
    failed = any(item.get("status") in {"failed", "timed_out"} for item in result["results"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
