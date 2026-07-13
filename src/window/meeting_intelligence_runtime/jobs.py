"""Bounded fixed-worker ownership for report-generation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import queue
import threading
from typing import Any, Callable
import uuid

from window.meeting_intelligence_runtime.models import ReportGenerationRequest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GenerationQueueFullError(RuntimeError):
    pass


@dataclass
class _GenerationJob:
    job_id: str
    request: ReportGenerationRequest
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued report generation"
    detail: str = ""
    percent: int = 0
    current: int = 0
    total: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    completed_at: str = ""
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


class GenerationJobManager:
    _STOP = object()

    def __init__(
        self,
        runner: Callable[[ReportGenerationRequest, Callable[[dict[str, Any]], None]], dict[str, Any]],
        *,
        worker_count: int = 1,
        max_queue_size: int = 32,
        max_terminal_jobs: int = 128,
        max_events_per_job: int = 80,
    ) -> None:
        self._runner = runner
        self._worker_count = max(1, int(worker_count))
        self._queue: queue.Queue[_GenerationJob | object] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._max_terminal_jobs = max(1, int(max_terminal_jobs))
        self._max_events_per_job = max(1, int(max_events_per_job))
        self._lock = threading.RLock()
        self._jobs: dict[str, _GenerationJob] = {}
        self._active: dict[tuple[str, str], str] = {}
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False

    def submit(self, request: ReportGenerationRequest) -> dict[str, Any]:
        key = (request.session_id, request.template_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("generation job manager is closed")
            active_id = self._active.get(key)
            if active_id is not None and active_id in self._jobs:
                return self._snapshot_locked(self._jobs[active_id])
            self._start_locked()
            job = _GenerationJob(f"mirjob_{uuid.uuid4().hex[:16]}", request)
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                raise GenerationQueueFullError(
                    f"report generation queue reached its {self._queue.maxsize}-job capacity"
                ) from exc
            self._jobs[job.job_id] = job
            self._active[key] = job.job_id
            return self._snapshot_locked(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            if job is None:
                raise ValueError("Generation job not found.")
            return self._snapshot_locked(job)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            threads = tuple(self._threads)
        for _thread in threads:
            self._queue.put(self._STOP)
        for thread in threads:
            thread.join()
        with self._lock:
            self._threads.clear()
            self._started = False

    def _start_locked(self) -> None:
        if self._started:
            return
        self._started = True
        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._worker, name=f"meeting-intelligence-worker-{index + 1}", daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _GenerationJob)
                self._run_job(item)
            finally:
                self._queue.task_done()

    def _run_job(self, job: _GenerationJob) -> None:
        self._update(
            job.job_id, status="running", stage="starting", message="Starting report generation",
            detail="Using captured transcript, template, and provider settings", percent=0,
        )
        try:
            result = self._runner(job.request, lambda event: self._progress(job.job_id, event))
        except Exception as exc:
            self._finish(job, status="failed", detail=str(exc), error=str(exc))
            return
        report = result.get("report") if isinstance(result, dict) else None
        detail = ""
        if isinstance(report, dict):
            detail = f"{len(report.get('evidence_index') or [])} evidence anchors, {len(report.get('sections') or {})} sections"
        self._finish(job, status="succeeded", detail=detail)

    def _finish(self, job: _GenerationJob, *, status: str, detail: str, error: str = "") -> None:
        with self._lock:
            current = self._jobs.get(job.job_id)
            if current is None:
                return
            current.status = status
            current.stage = "completed" if status == "succeeded" else "failed"
            current.message = "Report generated" if status == "succeeded" else "Report generation failed"
            current.detail = detail
            current.error = error
            if status == "succeeded":
                current.percent = 100
            current.completed_at = _now()
            current.updated_at = current.completed_at
            self._active.pop((job.request.session_id, job.request.template_id), None)
            self._prune_locked()

    def _progress(self, job_id: str, event: dict[str, Any]) -> None:
        clean = {
            "stage": str(event.get("stage") or ""), "message": str(event.get("message") or ""),
            "detail": str(event.get("detail") or ""), "percent": int(event.get("percent") or 0),
            "current": int(event.get("current") or 0), "total": int(event.get("total") or 0),
            "at": str(event.get("at") or _now()),
        }
        updates = {key: clean[key] for key in ("stage", "message", "detail", "percent", "current", "total")}
        self._update(job_id, **updates, append_event=clean)

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            event = updates.pop("append_event", None)
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = _now()
            if isinstance(event, dict):
                job.events.append(event)
                del job.events[:-self._max_events_per_job]

    def _prune_locked(self) -> None:
        terminal = [job for job in self._jobs.values() if job.status in {"succeeded", "failed"}]
        terminal.sort(key=lambda job: (job.completed_at, job.created_at, job.job_id))
        for job in terminal[:-self._max_terminal_jobs]:
            self._jobs.pop(job.job_id, None)

    @staticmethod
    def _snapshot_locked(job: _GenerationJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id, "session_id": job.request.session_id,
            "template_id": job.request.template_id, "status": job.status, "stage": job.stage,
            "message": job.message, "detail": job.detail, "percent": job.percent,
            "current": job.current, "total": job.total, "created_at": job.created_at,
            "updated_at": job.updated_at, "completed_at": job.completed_at, "error": job.error,
            "events": list(job.events[-12:]),
        }


__all__ = ["GenerationJobManager", "GenerationQueueFullError"]
