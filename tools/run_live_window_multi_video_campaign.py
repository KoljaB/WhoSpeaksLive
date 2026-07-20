"""Run a resumable provider-major live-window campaign over several videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embeddings.embedding_providers import (  # noqa: E402
    RemotePreparedEmbeddingProvider,
    SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS,
    create_embedding_provider,
)
from embeddings.live_shifting_window_corpus import (  # noqa: E402
    ControlledStop,
    JobConfig,
    build_live_window_job,
)
from embeddings.live_window_provider_campaign import _expected_embeddings  # noqa: E402
from embeddings.live_window_experiment_plan import (  # noqa: E402
    DEFAULT_HOP_SECONDS,
    FULL_WINDOW_UNIVERSE_SECONDS,
)


DEFAULT_SERVER_PROVIDERS = (
    "wespeaker_campplus",
    "wespeaker_resnet34_lm_onnx",
    "espnet_ecapa_wavlm_joint",
)


class CampaignInterrupted(BaseException):
    """Stop after the next durable builder checkpoint without marking a job failed."""


class _SharedProviderLease:
    """Expose one provider to several jobs while making per-job shutdown a no-op."""

    __module__ = "embeddings.embedding_providers"

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def embed(self, audio: Any, sample_rate: int) -> Any:
        if self._provider is None:
            raise RuntimeError("Shared provider lease has been released")
        return self._provider.embed(audio, sample_rate)

    def shutdown(self) -> None:
        return None

    def release(self) -> None:
        self._provider = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_videos(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    videos: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in values:
        video_id, separator, filename = raw.partition("=")
        video_id = video_id.strip()
        if not separator or not video_id or not filename.strip():
            raise ValueError(f"Expected --video VIDEO_ID=AUDIO_PATH, got {raw!r}")
        if video_id in seen:
            raise ValueError(f"Duplicate video id: {video_id}")
        path = Path(filename.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(video_id)
        videos.append((video_id, path))
    if not videos:
        raise ValueError("At least one --video is required")
    return tuple(videos)


def _job_progress(output_root: Path, provider: str, video_id: str) -> dict[str, Any]:
    return _read_json(output_root / "providers" / _slug(provider) / "videos" / _slug(video_id) / "progress.json")


def _job_complete(output_root: Path, provider: str, video_id: str) -> bool:
    job_dir = output_root / "providers" / _slug(provider) / "videos" / _slug(video_id)
    return (
        _read_json(job_dir / "job.json").get("status") == "complete"
        and _read_json(job_dir / "progress.json").get("status") == "complete"
    )


def run_campaign(
    *,
    videos: Sequence[tuple[str, Path]],
    providers: Sequence[str],
    server_providers: Sequence[str],
    output_root: Path,
    campaign_name: str,
    device: str,
    block_rows: int,
    provider_endpoint: str,
    allow_resume_builder_code_change: bool = False,
) -> dict[str, Any]:
    videos = tuple(videos)
    providers = tuple(dict.fromkeys(providers))
    server_providers = frozenset(server_providers)
    unknown = set(providers) - set(SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS)
    if unknown:
        raise ValueError(f"Unsupported providers: {sorted(unknown)}")

    output_root = output_root.resolve()
    campaign_dir = output_root / "campaigns" / _slug(campaign_name)
    progress_path = campaign_dir / "progress.json"
    errors_path = campaign_dir / "errors.jsonl"
    previous = _read_json(progress_path)
    started_at = str(previous.get("started_at") or _utc_now())
    expected_by_video = {
        video_id: _expected_embeddings(audio_path, FULL_WINDOW_UNIVERSE_SECONDS, DEFAULT_HOP_SECONDS)
        for video_id, audio_path in videos
    }
    total_expected = sum(expected_by_video.values()) * len(providers)
    stop_requested = False
    current_provider: str | None = None
    current_video: str | None = None

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)

    def publish(status: str = "running") -> dict[str, Any]:
        completed = 0
        complete_jobs = 0
        jobs: list[dict[str, Any]] = []
        for provider in providers:
            for video_id, _audio_path in videos:
                progress = _job_progress(output_root, provider, video_id)
                expected = expected_by_video[video_id]
                done = min(expected, int(progress.get("completed_embeddings", 0)))
                complete = _job_complete(output_root, provider, video_id)
                completed += expected if complete else done
                complete_jobs += int(complete)
                jobs.append({
                    "provider": provider,
                    "video_id": video_id,
                    "status": "complete" if complete else str(progress.get("status") or "pending"),
                    "completed_embeddings": expected if complete else done,
                    "expected_embeddings": expected,
                    "percent": round(100.0 * (expected if complete else done) / expected, 6),
                })
        payload = {
            "schema_version": 1,
            "campaign_name": campaign_name,
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "provider_major": True,
            "providers": list(providers),
            "server_providers": sorted(server_providers),
            "videos": [{"video_id": video_id, "audio_path": str(path)} for video_id, path in videos],
            "window_seconds": [float(value) for value in FULL_WINDOW_UNIVERSE_SECONDS],
            "hop_seconds": float(DEFAULT_HOP_SECONDS),
            "job_count": len(jobs),
            "complete_jobs": complete_jobs,
            "completed_embeddings": completed,
            "expected_embeddings": total_expected,
            "percent": round(100.0 * completed / total_expected, 6),
            "current_provider": current_provider,
            "current_video": current_video,
            "jobs": jobs,
        }
        _write_json_atomic(progress_path, payload)
        print(
            f"[multi-campaign] {payload['percent']:.3f}% embeddings="
            f"{completed}/{total_expected} jobs={complete_jobs}/{len(jobs)} "
            f"provider={current_provider or '-'} video={current_video or '-'} status={status}",
            flush=True,
        )
        return payload

    publish()
    try:
        for provider_name in providers:
            pending = [item for item in videos if not _job_complete(output_root, provider_name, item[0])]
            if not pending:
                publish()
                continue
            if stop_requested:
                raise CampaignInterrupted()
            current_provider = provider_name
            backend = "server" if provider_name in server_providers else "local"
            provider: Any | None = None
            lease: _SharedProviderLease | None = None
            try:
                if backend == "server":
                    provider = RemotePreparedEmbeddingProvider(provider_endpoint, provider_name, device)
                else:
                    provider = create_embedding_provider(provider_name, device)
                lease = _SharedProviderLease(provider)
                for video_id, audio_path in pending:
                    current_video = video_id
                    publish()

                    def observe(_job_payload: dict[str, Any]) -> None:
                        publish()
                        if stop_requested:
                            raise CampaignInterrupted()

                    config = JobConfig(
                        audio_path=audio_path,
                        video_id=video_id,
                        provider=provider_name,
                        output_root=output_root,
                        window_seconds=FULL_WINDOW_UNIVERSE_SECONDS,
                        hop_seconds=DEFAULT_HOP_SECONDS,
                        device=device,
                        block_rows=block_rows,
                        provider_backend=backend,
                        provider_endpoint=provider_endpoint if backend == "server" else "",
                        allow_resume_builder_code_change=allow_resume_builder_code_change,
                    )
                    try:
                        build_live_window_job(
                            config,
                            provider_factory=lambda _provider, _device: lease,
                            progress_observer=observe,
                        )
                    except (CampaignInterrupted, KeyboardInterrupt):
                        raise CampaignInterrupted()
                    except ControlledStop:
                        raise
                    except Exception as exc:
                        errors_path.parent.mkdir(parents=True, exist_ok=True)
                        with errors_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "time": _utc_now(),
                                "provider": provider_name,
                                "video_id": video_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }, sort_keys=True) + "\n")
                        print(f"[multi-campaign] failed provider={provider_name} video={video_id}: {exc}", flush=True)
                        raise
                    publish()
            finally:
                if provider is not None:
                    try:
                        shutdown = getattr(provider, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                    finally:
                        if lease is not None:
                            lease.release()
                            lease = None
                        del provider
                        gc.collect()
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
            current_video = None
            publish()
        current_provider = None
        provisional = publish()
        if provisional["complete_jobs"] != provisional["job_count"]:
            raise RuntimeError(
                f"Campaign exhausted its job loop with only "
                f"{provisional['complete_jobs']}/{provisional['job_count']} complete jobs"
            )
        final = publish("complete")
        final["completed_at"] = _utc_now()
        _write_json_atomic(progress_path, final)
        return final
    except CampaignInterrupted:
        final = publish("interrupted")
        final["interrupted_at"] = _utc_now()
        _write_json_atomic(progress_path, final)
        return final
    except Exception as exc:
        final = publish("failed")
        final["failed_at"] = _utc_now()
        final["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(progress_path, final)
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, help="VIDEO_ID=AUDIO_PATH; repeat per video")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--providers", default=",".join(SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS))
    parser.add_argument("--server-providers", default=",".join(DEFAULT_SERVER_PROVIDERS))
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--allow-resume-builder-code-change", action="store_true")
    args = parser.parse_args(argv)
    result = run_campaign(
        videos=_parse_videos(args.video),
        providers=tuple(value.strip() for value in args.providers.split(",") if value.strip()),
        server_providers=tuple(value.strip() for value in args.server_providers.split(",") if value.strip()),
        output_root=args.output_root,
        campaign_name=args.campaign_name,
        device=args.device,
        block_rows=args.block_rows,
        provider_endpoint=args.provider_endpoint,
        allow_resume_builder_code_change=args.allow_resume_builder_code_change,
    )
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
