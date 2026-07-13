"""Live-window translation coordination and sidecar client.

The coordinator keeps translation asynchronous and secondary to transcription:
stable source rows are always emitted first, provider work runs on a bounded
background queue, and every result is tied to the exact source-text hash that
created it.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

from window.language_config import (
    SUPPORTED_LANGUAGE_CONFIGS,
    language_flag_country_code,
    normalize_language_code,
)
from window.translation_service import (
    MADLAD_MODEL_ID,
    NLLB_MODEL_ID,
    TRANSLATEGEMMA_MODEL_ID,
    TRANSLATION_MODEL_METADATA,
    ProviderCapabilities,
    ProviderStatus,
    TranslationProvider,
    TranslationProviderConfig,
    TranslationQueueFullError,
    TranslationRequest,
    TranslationResult,
    TranslationService,
    create_translation_provider,
    translation_source_hash,
)


MODEL_PROFILES: dict[str, tuple[str, str]] = {
    "translate-gemma-4b": ("translategemma", TRANSLATEGEMMA_MODEL_ID),
    "nllb-200-600m": ("nllb", NLLB_MODEL_ID),
    "madlad-400-3b": ("madlad", MADLAD_MODEL_ID),
}


@dataclass(frozen=True)
class LiveTranslationConfig:
    language: str = "en"
    provider: str = "off"
    target_languages: tuple[str, ...] = ()
    max_targets: int = 4
    context_sentences: int = 2
    queue_size: int = 256
    browser_preferred: bool = False
    model_profile: str = "translate-gemma-4b"
    model: str = ""
    timeout_seconds: float = 600.0
    base_url: str = ""
    api_key_env: str = ""
    region: str = ""
    device: str = "auto"
    dtype: str = "auto"

    @classmethod
    def from_value(cls, value: Any) -> "LiveTranslationConfig":
        if isinstance(value, cls):
            return value
        targets = getattr(value, "translation_target_language", ())
        if isinstance(targets, str):
            targets = (targets,)
        return cls(
            language=str(getattr(value, "language", "en")),
            provider=str(getattr(value, "translation_provider", "off") or "off"),
            target_languages=tuple(str(target) for target in targets),
            max_targets=int(getattr(value, "translation_max_targets", 4)),
            context_sentences=int(getattr(value, "translation_context_sentences", 2)),
            queue_size=int(getattr(value, "translation_queue_size", 256)),
            browser_preferred=bool(getattr(value, "translation_browser_preferred", False)),
            model_profile=str(getattr(value, "translation_model_profile", "translate-gemma-4b")),
            model=str(getattr(value, "translation_model", "") or ""),
            timeout_seconds=float(getattr(value, "translation_timeout_seconds", 600.0)),
            base_url=str(getattr(value, "translation_base_url", "") or ""),
            api_key_env=str(getattr(value, "translation_api_key_env", "") or ""),
            region=str(getattr(value, "translation_region", "") or ""),
            device=str(getattr(value, "translation_device", "auto")),
            dtype=str(getattr(value, "translation_dtype", "auto")),
        )


class SidecarTranslationProvider(TranslationProvider):
    """Synchronous client used by TranslationService background workers."""

    provider_id = "sidecar"
    display_name = "WhoSpeaks translation sidecar"

    def __init__(
        self,
        *,
        base_url: str,
        model_profile: str,
        model: str = "",
        timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("translation sidecar base URL must not be empty")
        self.model_profile = str(model_profile or "translate-gemma-4b")
        _family, default_model = MODEL_PROFILES.get(
            self.model_profile,
            MODEL_PROFILES["translate-gemma-4b"],
        )
        self.model_id = str(model or default_model)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._configured_model_id = self.model_id
        self._observed_model_id = self.model_id
        self._status_lock = threading.Lock()
        self.capabilities = ProviderCapabilities(
            local=True,
            requires_network=True,
            lazy_loading=True,
            supports_context=True,
            max_parallel_requests=1,
        )

    @property
    def cache_identity(self) -> str:
        return f"{self.provider_id}:{self.base_url}:{self._configured_model_id}"

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        payload = {
            "source_text": str(text),
            "source_language": str(source_language),
            "target_language": str(target_language),
            "context": list(context),
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/translate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise RuntimeError(f"translation sidecar returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"translation sidecar is unavailable: {exc.reason or exc}") from exc
        translated = result.get("translated_text") if isinstance(result, Mapping) else None
        if not isinstance(translated, str) or not translated.strip():
            raise RuntimeError("translation sidecar response has no translated_text")
        if isinstance(result, Mapping):
            with self._status_lock:
                self._observed_model_id = str(result.get("model") or self._observed_model_id)
        return translated.strip()

    def status(self) -> ProviderStatus:
        request = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        available = False
        ready = False
        detail = "translation sidecar is not reachable"
        capabilities = self.capabilities
        try:
            with urllib.request.urlopen(request, timeout=min(0.75, self.timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, Mapping):
                available = bool(payload.get("ok"))
                ready = str(payload.get("readiness") or "") == "ready"
                detail = str(payload.get("detail") or payload.get("readiness") or "configured")
                with self._status_lock:
                    self._observed_model_id = str(payload.get("model") or self._observed_model_id)
                remote_capabilities = payload.get("capabilities")
                if isinstance(remote_capabilities, Mapping):
                    capabilities = ProviderCapabilities(
                        local=True,
                        requires_network=True,
                        lazy_loading=bool(remote_capabilities.get("lazy_loading", True)),
                        supports_context=bool(remote_capabilities.get("supports_context", False)),
                        supports_multi_target=bool(remote_capabilities.get("supports_multi_target", True)),
                        supported_language_codes=(
                            tuple(str(code) for code in remote_capabilities.get("supported_language_codes") or ())
                            or None
                        ),
                        max_parallel_requests=remote_capabilities.get("max_parallel_requests"),
                    )
        except Exception as exc:
            detail = f"translation sidecar is not reachable: {exc}"
        with self._status_lock:
            observed_model_id = self._observed_model_id
        return ProviderStatus(
            provider=self.provider_id,
            display_name=self.display_name,
            model=observed_model_id,
            available=available,
            ready=ready,
            detail=detail,
            capabilities=capabilities,
            model_metadata=TRANSLATION_MODEL_METADATA.get(observed_model_id),
        )


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
    except Exception:
        return str(exc.reason or exc)
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message") or error)[:1000]
    return str(payload)[:1000]


class LiveTranslationCoordinator:
    """Connect stable transcript rows to a revision-safe TranslationService."""

    def __init__(self, args: Any, bus: Any) -> None:
        self.config = LiveTranslationConfig.from_value(args)
        self.bus = bus
        self.source_language = normalize_language_code(self.config.language)
        self.max_targets = min(16, max(1, self.config.max_targets))
        self.context_sentences = max(0, self.config.context_sentences)
        queue_size = max(1, self.config.queue_size)
        self.max_deferred_jobs = min(8192, max(queue_size * 8, self.max_targets * 64))
        self.provider_kind = self.config.provider
        self.browser_preferred = self.config.browser_preferred
        self.model_profile = self.config.model_profile
        self._lock = threading.RLock()
        self._session_id = ""
        self._session_epoch = 0
        self._targets = self._normalize_targets(self.config.target_languages)
        self._rows: dict[str, dict[str, Any]] = {}
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._submitted: set[tuple[str, str, str, str]] = set()
        self._deferred: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._startup_error = ""
        self.provider: TranslationProvider | None = None
        self.service: TranslationService | None = None
        if self.provider_kind != "off":
            try:
                self.provider = self._create_provider()
                self.service = TranslationService(
                    self.provider,
                    worker_count=1,
                    max_queue_size=queue_size,
                    on_result=self._on_result,
                )
            except Exception as exc:
                self._startup_error = f"{type(exc).__name__}: {exc}"

    @property
    def enabled(self) -> bool:
        return self.service is not None

    def _create_provider(self) -> TranslationProvider:
        model_override = self.config.model
        timeout = self.config.timeout_seconds
        if self.provider_kind == "sidecar":
            return SidecarTranslationProvider(
                base_url=str(
                    self.config.base_url
                    or "http://127.0.0.1:8799"
                ),
                model_profile=self.model_profile,
                model=model_override,
                timeout_seconds=timeout,
            )
        if self.provider_kind in {
            "openai_compatible",
            "deepl",
            "google_cloud",
            "azure_translator",
            "libretranslate",
        }:
            default_key_env = {
                "openai_compatible": "OPENAI_API_KEY",
                "deepl": "DEEPL_API_KEY",
                "google_cloud": "GOOGLE_TRANSLATE_API_KEY",
                "azure_translator": "AZURE_TRANSLATOR_KEY",
                "libretranslate": "LIBRETRANSLATE_API_KEY",
            }[self.provider_kind]
            api_key_env = str(
                self.config.api_key_env or default_key_env
            )
            return create_translation_provider(TranslationProviderConfig(
                kind=self.provider_kind,
                base_url=self.config.base_url,
                model=model_override,
                api_key=os.environ.get(api_key_env, ""),
                timeout_seconds=timeout,
                options={
                    "region": self.config.region,
                },
            ))
        if self.provider_kind == "mock":
            return create_translation_provider({"kind": "mock"})
        family, default_model = MODEL_PROFILES.get(
            self.model_profile,
            MODEL_PROFILES["translate-gemma-4b"],
        )
        return create_translation_provider(TranslationProviderConfig(
            kind=family,
            model=model_override or default_model,
            device=self.config.device,
            dtype=self.config.dtype,
        ))

    def begin_session(self, session_id: str) -> None:
        normalized = str(session_id or "")
        with self._lock:
            self._session_epoch += 1
            self._session_id = normalized
            self._rows.clear()
            self._records.clear()
            self._submitted.clear()
            self._deferred.clear()

    def handle_sentence(self, payload: Mapping[str, Any], session_id: str = "") -> None:
        if not self.enabled or payload.get("realtime") or payload.get("provisional_assignment"):
            return
        text = str(payload.get("text") or "")
        segment_id = str(payload.get("index") if payload.get("index") is not None else "").strip()
        if not text.strip() or not segment_id:
            return
        if session_id:
            with self._lock:
                needs_session = str(session_id) != self._session_id
            if needs_session:
                self.begin_session(session_id)
        row = dict(payload)
        row["text"] = text
        row["source_text_hash"] = str(payload.get("source_text_hash") or translation_source_hash(text))
        row["source_revision"] = str(payload.get("source_revision") or row["source_text_hash"])
        with self._lock:
            self._rows[segment_id] = row
        self._submit_row(segment_id, row)

    def backfill(self, rows: Sequence[Mapping[str, Any]], session_id: str = "") -> None:
        if session_id:
            with self._lock:
                needs_session = str(session_id) != self._session_id
            if needs_session:
                self.begin_session(session_id)
        for row in rows:
            if isinstance(row, Mapping):
                self.handle_sentence(row, session_id)

    def configure(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_targets = payload.get("target_languages", ())
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, Sequence):
            raise ValueError("target_languages must be an array of language codes")
        targets = self._normalize_targets(raw_targets)
        with self._lock:
            self._targets = targets
            self._deferred = {
                key: row
                for key, row in self._deferred.items()
                if key[1] in targets
            }
            rows = list(self._rows.items())
        for segment_id, row in rows:
            self._submit_row(segment_id, row)
        return self.public_config(refresh_status=False)

    def _normalize_targets(self, values: Sequence[Any]) -> list[str]:
        targets: list[str] = []
        for value in values:
            target = normalize_language_code(value)
            if target != self.source_language and target not in targets:
                targets.append(target)
        if len(targets) > self.max_targets:
            raise ValueError(f"Choose at most {self.max_targets} translation target languages.")
        return targets

    def _submit_row(
        self,
        segment_id: str,
        row: Mapping[str, Any],
        *,
        only_targets: Sequence[str] | None = None,
        emit_queued: bool = True,
        force_backend: bool = False,
    ) -> None:
        service = self.service
        if service is None:
            return
        source_hash = str(row.get("source_text_hash") or translation_source_hash(str(row.get("text") or "")))
        source_revision = str(row.get("source_revision") or source_hash)
        with self._lock:
            current_row = self._rows.get(segment_id)
            if current_row is None:
                return
            current_hash = str(
                current_row.get("source_text_hash")
                or translation_source_hash(str(current_row.get("text") or ""))
            )
            current_revision = str(current_row.get("source_revision") or current_hash)
            if current_hash != source_hash or current_revision != source_revision:
                return
            for key in tuple(self._deferred):
                if key[0] == segment_id and (key[2] != source_hash or key[3] != source_revision):
                    self._deferred.pop(key, None)
            requested_targets = set(only_targets) if only_targets is not None else None
            targets = [
                target
                for target in self._targets
                if (requested_targets is None or target in requested_targets)
                if (segment_id, target, source_hash, source_revision) not in self._submitted
                and not (
                    (record := self._records.get((segment_id, target)))
                    and record.get("source_text_hash") == source_hash
                    and record.get("status") == "complete"
                )
            ]
            if not targets:
                return
            context = self._context_for_locked(segment_id)
            for target in targets:
                self._submitted.add((segment_id, target, source_hash, source_revision))
            session_id = self._session_id
            session_epoch = self._session_epoch
        if emit_queued:
            for target in targets:
                self.bus.emit("translation", self._event_payload(
                    segment_id=segment_id,
                    target=target,
                    source_hash=source_hash,
                    source_revision=source_revision,
                    status="queued",
                    session_id=session_id,
                ))
        if self.browser_preferred and not force_backend:
            return
        for target in targets:
            key = (segment_id, target, source_hash, source_revision)
            try:
                service.submit(TranslationRequest(
                    segment_id=segment_id,
                    session_id=session_id,
                    session_epoch=session_epoch,
                    source_text=str(row.get("text") or ""),
                    source_language=self.source_language,
                    target_languages=(target,),
                    source_revision=source_revision,
                    context=context,
                ))
                with self._lock:
                    self._deferred.pop(key, None)
            except TranslationQueueFullError:
                deferred = False
                with self._lock:
                    self._submitted.discard(key)
                    if key in self._deferred or len(self._deferred) < self.max_deferred_jobs:
                        self._deferred[key] = dict(row)
                        deferred = True
                if not deferred:
                    self.bus.emit("translation", self._event_payload(
                        segment_id=segment_id,
                        target=target,
                        source_hash=source_hash,
                        source_revision=source_revision,
                        status="error",
                        session_id=session_id,
                        error=f"translation backlog reached its {self.max_deferred_jobs}-job capacity",
                    ))
            except Exception as exc:
                with self._lock:
                    self._submitted.discard(key)
                    self._deferred.pop(key, None)
                self.bus.emit("translation", self._event_payload(
                    segment_id=segment_id,
                    target=target,
                    source_hash=source_hash,
                    source_revision=source_revision,
                    status="error",
                    session_id=session_id,
                    error=str(exc),
                ))

    def accept_browser_result(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.browser_preferred:
            raise ValueError("browser translation is not enabled")
        segment_id, target, row, source_hash, source_revision, session_id, session_epoch = self._browser_request(payload)
        translated_text = str(payload.get("text") or payload.get("translated_text") or "").strip()
        if not translated_text:
            raise ValueError("translated browser text must not be empty")
        event = self._event_payload(
            segment_id=segment_id,
            target=target,
            source_hash=source_hash,
            source_revision=source_revision,
            status="complete",
            text=translated_text,
            provider="chrome_translator",
            model="Chrome on-device Translator API",
            latency_seconds=float(payload.get("latency_seconds") or 0.0),
            session_id=session_id,
        )
        with self._lock:
            if not self._browser_commit_is_current_locked(
                segment_id, target, source_hash, source_revision, session_id, session_epoch
            ):
                raise ValueError("browser translation result is stale")
            self._records[(segment_id, target)] = dict(event)
        self.bus.emit("translation", event)
        return event

    def request_browser_fallback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.browser_preferred:
            raise ValueError("browser translation is not enabled")
        segment_id, target, row, source_hash, source_revision, session_id, session_epoch = self._browser_request(payload)
        key = (segment_id, target, source_hash, source_revision)
        with self._lock:
            self._submitted.discard(key)
        self._submit_row(
            segment_id,
            row,
            only_targets=(target,),
            emit_queued=False,
            force_backend=True,
        )
        return self._event_payload(
            segment_id=segment_id,
            target=target,
            source_hash=source_hash,
            source_revision=source_revision,
            status="queued",
            provider=self.provider.provider_id if self.provider else self.provider_kind,
            model=self.provider.model_id if self.provider else "",
            session_id=session_id,
        )

    def _browser_request(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str, str, dict[str, Any], str, str, str, int]:
        segment_id = str(
            payload.get("segment_id")
            if payload.get("segment_id") is not None
            else payload.get("sentence_index")
        ).strip()
        target = normalize_language_code(payload.get("target_language"))
        if not segment_id:
            raise ValueError("browser translation segment_id must not be empty")
        with self._lock:
            if target not in self._targets:
                raise ValueError("browser translation target is not selected")
            row = dict(self._rows.get(segment_id) or {})
            session_id = self._session_id
            session_epoch = self._session_epoch
        if not row:
            raise ValueError("browser translation sentence is no longer available")
        source_hash = str(row.get("source_text_hash") or translation_source_hash(str(row.get("text") or "")))
        source_revision = str(row.get("source_revision") or source_hash)
        supplied_hash = str(payload.get("source_text_hash") or payload.get("source_hash") or "")
        supplied_revision = str(payload.get("source_revision") or "")
        if supplied_hash and supplied_hash != source_hash:
            raise ValueError("browser translation source hash is stale")
        if supplied_revision and supplied_revision != source_revision:
            raise ValueError("browser translation source revision is stale")
        return segment_id, target, row, source_hash, source_revision, session_id, session_epoch

    def _browser_commit_is_current_locked(
        self,
        segment_id: str,
        target: str,
        source_hash: str,
        source_revision: str,
        session_id: str,
        session_epoch: int,
    ) -> bool:
        if self._session_id != session_id or self._session_epoch != session_epoch or target not in self._targets:
            return False
        row = self._rows.get(segment_id)
        if row is None:
            return False
        current_hash = str(row.get("source_text_hash") or translation_source_hash(str(row.get("text") or "")))
        current_revision = str(row.get("source_revision") or current_hash)
        return current_hash == source_hash and current_revision == source_revision

    def _context_for_locked(self, segment_id: str) -> tuple[str, ...]:
        if self.context_sentences <= 0:
            return ()
        current = self._rows.get(segment_id, {})
        current_start = float(current.get("start") or 0.0)
        previous = [
            (float(row.get("start") or 0.0), str(row.get("text") or ""))
            for key, row in self._rows.items()
            if key != segment_id and float(row.get("start") or 0.0) <= current_start
        ]
        previous.sort(key=lambda item: item[0])
        return tuple(text for _start, text in previous[-self.context_sentences:] if text)

    def _on_result(self, result: TranslationResult) -> None:
        with self._lock:
            if (
                result.session_id != self._session_id
                or result.session_epoch != self._session_epoch
                or not self._browser_commit_is_current_locked(
                    result.segment_id,
                    result.target_language,
                    result.source_hash,
                    result.source_revision,
                    result.session_id,
                    result.session_epoch,
                )
            ):
                return
        if result.status not in {"completed", "error"}:
            return
        status = "complete" if result.status == "completed" else "error"
        payload = self._event_payload(
            segment_id=result.segment_id,
            target=result.target_language,
            source_hash=result.source_hash,
            source_revision=result.source_revision,
            status=status,
            text=result.translated_text,
            error=result.error,
            provider=result.provider,
            model=result.model,
            latency_seconds=result.latency_seconds,
            cached=result.cached,
            session_id=result.session_id,
        )
        with self._lock:
            if (
                result.session_id != self._session_id
                or result.session_epoch != self._session_epoch
                or not self._browser_commit_is_current_locked(
                    result.segment_id,
                    result.target_language,
                    result.source_hash,
                    result.source_revision,
                    result.session_id,
                    result.session_epoch,
                )
            ):
                return
            if status == "error":
                self._submitted.discard((
                    result.segment_id,
                    result.target_language,
                    result.source_hash,
                    result.source_revision,
                ))
            self._records[(result.segment_id, result.target_language)] = dict(payload)
        self.bus.emit("translation", payload)
        self._drain_deferred()

    def _drain_deferred(self) -> None:
        with self._lock:
            candidate: tuple[tuple[str, str, str, str], dict[str, Any]] | None = None
            for key, row in self._deferred.items():
                if key[1] in self._targets:
                    candidate = (key, dict(row))
                    break
            if candidate is None:
                return
        key, row = candidate
        self._submit_row(key[0], row, only_targets=(key[1],), emit_queued=False)

    def _event_payload(
        self,
        *,
        segment_id: str,
        target: str,
        source_hash: str,
        source_revision: str,
        status: str,
        text: str = "",
        error: str = "",
        provider: str = "",
        model: str = "",
        latency_seconds: float = 0.0,
        cached: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            sentence_index: int | str = int(segment_id)
        except ValueError:
            sentence_index = segment_id
        language = SUPPORTED_LANGUAGE_CONFIGS.get(target)
        return {
            "segment_id": segment_id,
            "sentence_index": sentence_index,
            "session_id": self._session_id if session_id is None else session_id,
            "source_language": self.source_language,
            "source_text_hash": source_hash,
            "source_hash": source_hash,
            "source_revision": source_revision,
            "target_language": target,
            "target_language_name": language.display_name if language else target,
            "status": status,
            "text": text,
            "translated_text": text,
            "error": str(error or "")[:1000],
            "provider": provider or (self.provider.provider_id if self.provider else self.provider_kind),
            "model": model or (self.provider.model_id if self.provider else ""),
            "latency_seconds": round(max(0.0, float(latency_seconds)), 4),
            "cached": bool(cached),
        }

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            records = [
                dict(record)
                for record in self._records.values()
                if str(record.get("status") or "") == "complete"
            ]
        return sorted(records, key=lambda item: (str(item.get("segment_id") or ""), str(item.get("target_language") or "")))

    def public_config(self, *, refresh_status: bool = True) -> dict[str, Any]:
        with self._lock:
            targets = list(self._targets)
        service_status: dict[str, Any] = {}
        if self.service is not None:
            if refresh_status:
                service_status = self.service.status()
                with self._lock:
                    service_status["deferred_jobs"] = len(self._deferred)
                    service_status["max_deferred_jobs"] = self.max_deferred_jobs
            else:
                service_status = {
                    "provider": {
                        "provider": self.provider.provider_id if self.provider else self.provider_kind,
                        "display_name": self.provider.display_name if self.provider else self.provider_kind,
                        "model": self.provider.model_id if self.provider else "",
                    }
                }
        provider_status = service_status.get("provider") if isinstance(service_status, Mapping) else {}
        if not isinstance(provider_status, Mapping):
            provider_status = {}
        model_metadata = provider_status.get("model_metadata")
        if not isinstance(model_metadata, Mapping) and self.provider is not None:
            metadata = TRANSLATION_MODEL_METADATA.get(self.provider.model_id)
            model_metadata = metadata.to_dict() if metadata else None
        provider_payload = {
            "id": str(provider_status.get("provider") or self.provider_kind),
            "label": str(provider_status.get("display_name") or self.provider_kind.replace("_", " ").title()),
            "model": str(provider_status.get("model") or (self.provider.model_id if self.provider else "")),
            "available": bool(provider_status.get("available", self.service is not None)),
            "ready": bool(provider_status.get("ready", False)),
            "detail": str(provider_status.get("detail") or self._startup_error),
            "model_metadata": model_metadata,
        }
        capabilities_payload = provider_status.get("capabilities")
        supported_codes: set[str] | None = None
        if isinstance(capabilities_payload, Mapping):
            configured_codes = capabilities_payload.get("supported_language_codes")
            if isinstance(configured_codes, Sequence) and not isinstance(configured_codes, (str, bytes)):
                supported_codes = {str(code).split("-", 1)[0] for code in configured_codes}
        languages = [
            {
                "code": code,
                "name": config.display_name,
                "flag_url": f"/assets/flags/4x3/{language_flag_country_code(code)}.svg",
            }
            for code, config in sorted(
                SUPPORTED_LANGUAGE_CONFIGS.items(),
                key=lambda item: item[1].display_name.casefold(),
            )
            if code != self.source_language and (supported_codes is None or code in supported_codes)
        ]
        return {
            "available": self.service is not None,
            "enabled": self.service is not None,
            "provider": provider_payload,
            "browser_preferred": self.browser_preferred,
            "source_language": self.source_language,
            "languages": languages,
            "selected_targets": targets,
            "primary_target": targets[0] if targets else "",
            "display_mode": "single" if targets else "original",
            "max_targets": self.max_targets,
            "service": service_status,
            "error": self._startup_error,
        }

    def shutdown(self) -> None:
        if self.service is not None:
            self.service.close()


__all__ = ["LiveTranslationConfig", "LiveTranslationCoordinator", "MODEL_PROFILES", "SidecarTranslationProvider"]
