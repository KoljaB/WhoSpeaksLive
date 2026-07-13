"""Bounded revision-safe translation scheduler."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
from typing import Any

from window.translation.contracts import (
    LOGGER,
    TranslationCallback,
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
    TranslationResultStatus,
    _normalize_language_tag,
    translation_source_hash,
)


class TranslationQueueFullError(RuntimeError):
    pass


@dataclass
class _Subscriber:
    future: Future[TranslationResult]
    callback: TranslationCallback | None


@dataclass
class _Job:
    request: TranslationRequest
    target_language: str
    generation: int
    subscribers: list[_Subscriber]
    started_at: float = 0.0


@dataclass(frozen=True)
class _CacheValue:
    translated_text: str


class TranslationSubmission:
    """Futures for every target language in one multi-target submission."""

    def __init__(self, request: TranslationRequest, futures: Mapping[str, Future[TranslationResult]]) -> None:
        self.request = request
        self.futures = dict(futures)

    def done(self) -> bool:
        return all(future.done() for future in self.futures.values())

    def result(self, target_language: str, timeout: float | None = None) -> TranslationResult:
        target = _normalize_language_tag(target_language)
        return self.futures[target].result(timeout=timeout)

    def wait(self, timeout: float | None = None) -> dict[str, TranslationResult]:
        deadline = None if timeout is None else time.monotonic() + timeout
        results: dict[str, TranslationResult] = {}
        for target, future in self.futures.items():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            results[target] = future.result(timeout=remaining)
        return results


class TranslationScheduler:
    """Fan stable sentences out to target languages on background workers.

    A newer call to :meth:`submit` for the same ``segment_id`` and target
    language supersedes queued and in-flight older work.  Superseded futures
    resolve with a provenance-complete result, but stale translations are never
    sent to callbacks.
    """

    _STOP = object()

    def __init__(
        self,
        provider: TranslationProvider,
        *,
        worker_count: int = 1,
        max_queue_size: int = 256,
        cache_size: int = 2048,
        on_result: TranslationCallback | None = None,
        thread_name_prefix: str = "translation",
    ) -> None:
        self.provider = provider
        self.worker_count = max(1, int(worker_count))
        self.max_queue_size = max(1, int(max_queue_size))
        self.cache_size = max(0, int(cache_size))
        self.on_result = on_result
        self.thread_name_prefix = str(thread_name_prefix or "translation")
        self._queue: Queue[_Job | object] = Queue(maxsize=self.max_queue_size)
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._accepting = True
        self._generation = 0
        self._latest: dict[tuple[str, int, str, str], int] = {}
        self._active: dict[tuple[str, int, str, str], _Job] = {}
        self._cache: OrderedDict[tuple[str, str, str, str, str], _CacheValue] = OrderedDict()
        self._unfinished_jobs = 0
        # Provider status discovery is allowed to change observed metadata, but
        # never the cache namespace of this scheduler instance.
        self._cache_identity = str(provider.cache_identity)
        self._provider_closed = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if not self._accepting:
                raise RuntimeError("translation service has been stopped")
            self._started = True
            for index in range(self.worker_count):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"{self.thread_name_prefix}-{index + 1}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def submit(
        self,
        request: TranslationRequest,
        *,
        callback: TranslationCallback | None = None,
    ) -> TranslationSubmission:
        if not isinstance(request, TranslationRequest):
            raise TypeError("request must be a TranslationRequest")
        self.start()
        futures = {target: Future() for target in request.target_languages}
        deliveries: list[tuple[_Subscriber, TranslationResult]] = []
        jobs_to_queue: list[_Job] = []
        with self._lock:
            if not self._accepting:
                raise RuntimeError("translation service has been stopped")
            plans: list[tuple[str, str, Any]] = []
            for target, future in futures.items():
                row_key = self._row_key(request, target)
                current_job = self._active.get(row_key)
                if (
                    current_job is not None
                    and current_job.request.source_hash == request.source_hash
                    and current_job.request.source_revision == request.source_revision
                    and current_job.request.source_language == request.source_language
                    and current_job.request.context == request.context
                ):
                    plans.append((target, "dedupe", current_job))
                    continue
                cache_key = self._cache_key(request, target)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    plans.append((target, "cache", cached))
                else:
                    plans.append((target, "queue", None))
            new_job_count = sum(1 for _, plan, _ in plans if plan == "queue")
            if self._queue.qsize() + new_job_count > self.max_queue_size:
                raise TranslationQueueFullError(
                    f"translation queue cannot accept {new_job_count} jobs; "
                    f"capacity is {self.max_queue_size}"
                )
            for target, plan, value in plans:
                subscriber = _Subscriber(futures[target], callback)
                row_key = self._row_key(request, target)
                if plan == "dedupe":
                    value.subscribers.append(subscriber)
                    continue
                self._generation += 1
                generation = self._generation
                self._latest[row_key] = generation
                if plan == "cache":
                    cache_key = self._cache_key(request, target)
                    self._cache.move_to_end(cache_key)
                    result = self._result(
                        request,
                        target,
                        translated_text=value.translated_text,
                        status="completed",
                        latency_seconds=0.0,
                        cached=True,
                    )
                    deliveries.append((subscriber, result))
                    continue
                job = _Job(request, target, generation, [subscriber])
                self._active[row_key] = job
                self._unfinished_jobs += 1
                jobs_to_queue.append(job)
            for job in jobs_to_queue:
                self._queue.put_nowait(job)
        for subscriber, result in deliveries:
            self._deliver(subscriber, result, publish=True)
        return TranslationSubmission(request, futures)

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._idle:
            while self._unfinished_jobs:
                if deadline is None:
                    self._idle.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                "started": self._started,
                "accepting": self._accepting,
                "queued_jobs": self._queue.qsize(),
                "unfinished_jobs": self._unfinished_jobs,
                "worker_count": self.worker_count,
                "max_queue_size": self.max_queue_size,
                "cache_entries": len(self._cache),
                "cache_size": self.cache_size,
            }
        # Provider status may perform network or model work. Never call it while
        # holding the scheduler state lock.
        snapshot["provider"] = self.provider.status().to_dict()
        return snapshot

    def stop(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        with self._lock:
            if not self._started:
                self._accepting = False
                return
            already_stopping = not self._accepting
            self._accepting = False
            threads = tuple(self._threads)
        if not already_stopping:
            if cancel_pending:
                self._cancel_queued()
            for _ in threads:
                self._queue.put(self._STOP)
        if wait:
            for thread in threads:
                thread.join()
            with self._lock:
                self._threads.clear()
                self._started = False

    def close(self) -> None:
        self.stop(wait=True)
        with self._lock:
            if self._provider_closed:
                return
            self._provider_closed = True
        self.provider.close()

    def __enter__(self) -> TranslationService:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                self._queue.task_done()
                return
            job = item
            assert isinstance(job, _Job)
            try:
                self._run_job(job)
            finally:
                self._queue.task_done()

    def _run_job(self, job: _Job) -> None:
        row_key = self._row_key(job.request, job.target_language)
        with self._lock:
            current = self._latest.get(row_key) == job.generation
        if not current:
            self._complete_superseded(job, "superseded before translation started")
            return
        job.started_at = time.perf_counter()
        try:
            translated = self.provider.translate(
                job.request.source_text,
                job.request.source_language,
                job.target_language,
                context=job.request.context,
            )
            if not isinstance(translated, str):
                raise TypeError("translation provider returned a non-string result")
            elapsed = time.perf_counter() - job.started_at
            with self._lock:
                current = self._latest.get(row_key) == job.generation
                if current and self.cache_size:
                    cache_key = self._cache_key(job.request, job.target_language)
                    self._cache[cache_key] = _CacheValue(translated)
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
            if not current:
                self._complete_superseded(job, "superseded while translation was in flight", elapsed)
                return
            result = self._result(
                job.request,
                job.target_language,
                translated_text=translated,
                status="completed",
                latency_seconds=elapsed,
                cached=False,
            )
            self._complete_job(job, result, publish=True)
        except Exception as exc:
            elapsed = max(0.0, time.perf_counter() - job.started_at)
            with self._lock:
                current = self._latest.get(row_key) == job.generation
            if not current:
                self._complete_superseded(job, "superseded while translation was in flight", elapsed)
                return
            result = self._result(
                job.request,
                job.target_language,
                translated_text="",
                status="error",
                latency_seconds=elapsed,
                cached=False,
                error=str(exc),
            )
            self._complete_job(job, result, publish=True)

    def _complete_superseded(self, job: _Job, reason: str, elapsed: float = 0.0) -> None:
        result = self._result(
            job.request,
            job.target_language,
            translated_text="",
            status="superseded",
            latency_seconds=elapsed,
            cached=False,
            error=reason,
        )
        self._complete_job(job, result, publish=False)

    def _complete_job(self, job: _Job, result: TranslationResult, *, publish: bool) -> None:
        row_key = self._row_key(job.request, job.target_language)
        with self._idle:
            if publish and self._latest.get(row_key) != job.generation:
                result = self._result(
                    job.request,
                    job.target_language,
                    translated_text="",
                    status="superseded",
                    latency_seconds=result.latency_seconds,
                    cached=False,
                    error="superseded before the translation result was published",
                )
                publish = False
            if self._active.get(row_key) is job:
                self._active.pop(row_key, None)
            subscribers = tuple(job.subscribers)
            self._unfinished_jobs = max(0, self._unfinished_jobs - 1)
            self._idle.notify_all()
        for subscriber in subscribers:
            if not subscriber.future.done():
                subscriber.future.set_result(result)
        if publish and self.on_result is not None:
            self._invoke_callback(self.on_result, result)
        for subscriber in subscribers:
            if publish and subscriber.callback is not None and subscriber.callback != self.on_result:
                self._invoke_callback(subscriber.callback, result)

    def _deliver(
        self,
        subscriber: _Subscriber,
        result: TranslationResult,
        *,
        publish: bool,
        include_global: bool = True,
    ) -> None:
        if not subscriber.future.done():
            subscriber.future.set_result(result)
        if not publish:
            return
        callbacks: list[TranslationCallback] = []
        if include_global and self.on_result is not None:
            callbacks.append(self.on_result)
        if subscriber.callback is not None and subscriber.callback not in callbacks:
            callbacks.append(subscriber.callback)
        for callback in callbacks:
            self._invoke_callback(callback, result)

    @staticmethod
    def _invoke_callback(callback: TranslationCallback, result: TranslationResult) -> None:
        try:
            callback(result)
        except Exception:
            LOGGER.exception("translation result callback failed")

    def _cancel_queued(self) -> None:
        cancelled: list[_Job] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if isinstance(item, _Job):
                cancelled.append(item)
            self._queue.task_done()
        for job in cancelled:
            result = self._result(
                job.request,
                job.target_language,
                translated_text="",
                status="cancelled",
                latency_seconds=0.0,
                cached=False,
                error="translation service stopped before the job started",
            )
            self._complete_job(job, result, publish=False)

    def _cache_key(self, request: TranslationRequest, target_language: str) -> tuple[str, str, str, str, str]:
        context_hash = translation_source_hash("\n\x1e\n".join(request.context))
        return (
            self._cache_identity,
            request.source_language,
            target_language,
            request.source_hash,
            context_hash,
        )

    @staticmethod
    def _row_key(request: TranslationRequest, target_language: str) -> tuple[str, int, str, str]:
        return (request.session_id, request.session_epoch, request.segment_id, target_language)

    def _result(
        self,
        request: TranslationRequest,
        target_language: str,
        *,
        translated_text: str,
        status: TranslationResultStatus,
        latency_seconds: float,
        cached: bool,
        error: str = "",
    ) -> TranslationResult:
        return TranslationResult(
            segment_id=request.segment_id,
            session_id=request.session_id,
            source_hash=request.source_hash,
            source_revision=request.source_revision,
            source_language=request.source_language,
            target_language=target_language,
            translated_text=translated_text,
            provider=self.provider.provider_id,
            model=self.provider.model_id,
            status=status,
            latency_seconds=max(0.0, float(latency_seconds)),
            cached=bool(cached),
            session_epoch=request.session_epoch,
            error=str(error or ""),
        )


class TranslationService(TranslationScheduler):
    """Compatibility façade retaining the documented public class name."""

__all__ = [
    "TranslationQueueFullError",
    "TranslationScheduler",
    "TranslationService",
    "TranslationSubmission",
]
