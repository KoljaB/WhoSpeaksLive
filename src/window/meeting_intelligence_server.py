"""Standalone browser server for LLM-based meeting intelligence reports."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.parse
import urllib.request

from paths import RUNTIME_DIR
from window.meeting_intelligence import transcript_revision_id
from window.meeting_intelligence_pipeline import (
    MeetingLLMConfig,
    MockMeetingLLMClient,
    MultiPassMeetingIntelligencePipeline,
    OpenAICompatibleMeetingClient,
    StructuredChatClient,
    default_llm_config,
    normalize_report_language,
    sanitize_report_output,
    stable_hash,
)
from window.meeting_chat import (
    MeetingChatEngine,
    MeetingChatJobManager,
    MeetingChatStore,
    MeetingTextIndex,
    MockTextEmbeddingClient,
    TextEmbeddingConfig,
)
from window.meeting_intelligence_runtime import (
    AutoGenerationMonitor,
    AutoGenerationTracker,
    GenerationJobManager,
    GenerationQueueFullError,
    MeetingSettingsStore,
    ReportCache,
    ReportGenerationRequest,
)
from window.meeting_intelligence_runtime.models import immutable_json
from window.meeting_server_support import (
    DEFAULT_ENV_FILE,
    LLM_PROVIDER_OPTIONS,
    extract_model_ids,
    load_env_file,
    normalize_provider,
    parse_timecode,
    parse_whospeakslive_transcript,
    provider_options_payload,
    sort_model_ids,
    speaker_id_from_name,
    strip_env_value,
    unique_strings,
    is_likely_text_generation_model,
)
from window.report_templates import (
    STANDARD_TEMPLATE_ID,
    ReportTemplateStore,
    builtin_report_templates,
    get_builtin_report_template,
    validate_report_template,
)
from window.session_store import DEFAULT_SESSION_DIR, SessionStore
from window.web_assets import read_web_asset, web_asset_content_type


DEMO_SESSION_ID = "demo-whospeakslive-transcript"
DEFAULT_CACHE_DIR = RUNTIME_DIR / "meeting_intelligence_reports"
DEFAULT_TEMPLATE_DIR = RUNTIME_DIR / "report_templates"
DEFAULT_CHAT_DIR = RUNTIME_DIR / "meeting_chats"
DEFAULT_TEXT_INDEX_DB = RUNTIME_DIR / "meeting_text_index.sqlite3"


@dataclass(frozen=True)
class MeetingIntelligenceServerConfig:
    session_dir: Path = DEFAULT_SESSION_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    template_dir: Path = DEFAULT_TEMPLATE_DIR
    chat_dir: Path = DEFAULT_CHAT_DIR
    text_index_db: Path = DEFAULT_TEXT_INDEX_DB
    demo_transcript: Path | None = None
    llm_config: MeetingLLMConfig = field(default_factory=default_llm_config)
    mock_llm: bool = False
    max_segment_rows: int = 80
    auto_generate: bool = False
    auto_generate_poll_seconds: float = 10.0
    report_language: str = "en"
    text_embedding: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    text_index_poll_seconds: float = 10.0


@dataclass(frozen=True)
class MeetingIntelligenceRuntimeConfig:
    host: str
    port: int
    meeting: MeetingIntelligenceServerConfig


class MeetingIntelligenceService:
    def __init__(
        self,
        config: MeetingIntelligenceServerConfig,
        *,
        client_factory: Callable[[], StructuredChatClient] | None = None,
    ) -> None:
        self.config = config
        self.report_language, self.report_language_label = normalize_report_language(config.report_language)
        self.store = SessionStore(config.session_dir)
        self.template_store = ReportTemplateStore(config.template_dir)
        self.client_factory = client_factory
        self._settings = MeetingSettingsStore(config.llm_config)
        self._report_cache = ReportCache(config.cache_dir, hash_fn=stable_hash)
        self._job_manager = GenerationJobManager(self._generate_captured_request)
        self._auto_generation = AutoGenerationTracker()
        embedding_factory = (lambda: MockTextEmbeddingClient()) if config.mock_llm else None
        text_index_db = config.text_index_db
        chat_dir = config.chat_dir
        if config.cache_dir != DEFAULT_CACHE_DIR:
            if text_index_db == DEFAULT_TEXT_INDEX_DB:
                text_index_db = config.cache_dir.parent / DEFAULT_TEXT_INDEX_DB.name
            if chat_dir == DEFAULT_CHAT_DIR:
                chat_dir = config.cache_dir.parent / DEFAULT_CHAT_DIR.name
        self._text_index = MeetingTextIndex(
            text_index_db,
            config.text_embedding,
            client_factory=embedding_factory,
        )
        self._chat_store = MeetingChatStore(chat_dir)
        self._chat_engine = MeetingChatEngine(
            self._text_index,
            self._chat_store,
            session_loader=self.load_session,
            llm_client_factory=self._new_client,
            report_loader=self._chat_report_context,
        )
        self._chat_jobs = MeetingChatJobManager(self._chat_engine)
        self._index_scan_lock = threading.Lock()
        self._index_scan_baseline = False
        self._index_seen: dict[str, str] = {}
        self._scope_indexing: set[str] = set()

    def public_config(self) -> dict[str, Any]:
        llm = self._current_llm_config()
        return {
            "provider": llm.provider,
            "base_url": llm.base_url,
            "model": llm.model,
            "schema_mode": llm.schema_mode,
            "mock_llm": self.config.mock_llm,
            "auto_generate": self.config.auto_generate,
            "auto_generate_poll_seconds": self.config.auto_generate_poll_seconds,
            "max_segment_rows": self.config.max_segment_rows,
            "report_language": self.report_language,
            "standard_template_id": STANDARD_TEMPLATE_ID,
            "expected_report_provider": self.expected_report_provider(),
            "api_key_configured": self._provider_api_key_configured(llm),
            "api_key_env_var": self._provider_api_key_env_var(llm.provider),
            "providers": provider_options_payload(),
            "text_embedding": self.config.text_embedding.public(),
            "text_index_poll_seconds": self.config.text_index_poll_seconds,
        }

    def chat_scope(self, session_ids: list[str]) -> dict[str, Any]:
        result = self._chat_engine.scope(session_ids)
        if result.get("requires_index") and self.config.text_embedding.configured:
            self._queue_scope_index(result["scope_id"], result["session_ids"])
        return result

    def start_chat(self, session_ids: list[str], question: str, *, provisional: bool = False) -> dict[str, Any]:
        return self._chat_jobs.submit(session_ids, question, provisional=provisional)

    def get_chat_job(self, job_id: str) -> dict[str, Any]:
        return self._chat_jobs.get(job_id)

    def clear_chat(self, session_ids: list[str]) -> dict[str, Any]:
        return self._chat_store.clear(session_ids)

    def _queue_scope_index(self, scope_id: str, session_ids: list[str]) -> None:
        with self._index_scan_lock:
            if scope_id in self._scope_indexing:
                return
            self._scope_indexing.add(scope_id)

        def run() -> None:
            try:
                self._chat_engine.ensure_index(session_ids)
            except Exception:
                # The status remains not-current and the foreground chat job
                # returns the actionable provider error to the browser.
                pass
            finally:
                with self._index_scan_lock:
                    self._scope_indexing.discard(scope_id)

        threading.Thread(target=run, name=f"meeting-index-{scope_id[-8:]}", daemon=True).start()

    def auto_index_changed_sessions(self) -> list[str]:
        """Index sessions changed after service start without scanning the historical archive."""

        summaries = self.store.list_sessions("all")
        current = {
            str(summary.get("id") or ""): str(summary.get("updated_at") or "")
            for summary in summaries
            if summary.get("id") and summary.get("has_transcript")
        }
        with self._index_scan_lock:
            if not self._index_scan_baseline:
                self._index_seen = current
                self._index_scan_baseline = True
                return []
            changed = [session_id for session_id, updated_at in current.items() if self._index_seen.get(session_id) != updated_at]
            removed = (set(self._index_seen) | self._text_index.session_ids()) - set(current)
            self._index_seen = current
        for session_id in removed:
            self._text_index.delete_session(session_id)
            self._chat_store.delete_scopes_containing(session_id)
        if changed and (self.config.text_embedding.configured or self.config.mock_llm):
            try:
                sessions = self._chat_engine.capture_sessions(changed)
                self._text_index.ensure_sessions(sessions)
            except (FileNotFoundError, ValueError, OSError, RuntimeError):
                return []
        return changed

    def list_report_templates(self) -> list[dict[str, Any]]:
        """Return built-in presets and saved custom templates in one inspectable list."""

        templates = [self._public_template(template) for template in self.template_store.list_templates()]
        templates.sort(key=lambda item: (not bool(item.get("builtin")), str(item.get("name") or "").casefold()))
        return templates

    def get_report_template(self, template_id: str) -> dict[str, Any]:
        template = self._resolve_template(template_id)
        return self._public_template(template)

    def save_report_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("template") if isinstance(payload.get("template"), dict) else payload
        if bool(raw.get("builtin")):
            raise ValueError("Predefined report templates are read-only. Clone the preset before editing it.")
        return self._public_template(self.template_store.save_template(raw))

    def delete_report_template(self, template_id: str) -> bool:
        template_id = str(template_id or "").strip()
        if get_builtin_report_template(template_id) is not None:
            raise ValueError("Predefined report templates cannot be deleted.")
        return bool(self.template_store.delete_template(template_id))

    def clone_report_template(self, template_id: str, name: str) -> dict[str, Any]:
        return self._public_template(self.template_store.clone_template(str(template_id or "").strip(), str(name or "").strip()))

    def _resolve_template(self, template_id: str | None) -> dict[str, Any]:
        normalized_id = str(template_id or STANDARD_TEMPLATE_ID).strip() or STANDARD_TEMPLATE_ID
        builtin = get_builtin_report_template(normalized_id)
        if builtin is not None:
            return validate_report_template(builtin, allow_builtin=True)
        custom = self.template_store.get_template(normalized_id)
        if custom is None:
            raise ValueError(f"Unknown report template: {normalized_id}")
        return validate_report_template(custom)

    @staticmethod
    def _public_template(template: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(template)
        payload["source_kind"] = "predefined" if bool(payload.get("builtin")) else "custom"
        payload["read_only"] = bool(payload.get("builtin"))
        return payload

    def _template_report_language(self, template: dict[str, Any]) -> str:
        language_mode = str(template.get("language_mode") or "inherit").strip().lower()
        if language_mode == "inherit":
            return self.report_language
        return normalize_report_language(language_mode)[0]

    @staticmethod
    def _enforce_template_privacy(template: dict[str, Any], llm: MeetingLLMConfig, *, mock_llm: bool) -> None:
        privacy_policy = str(template.get("privacy_policy") or "allow_remote").strip().lower()
        if privacy_policy == "local_only" and not mock_llm and llm.provider in {"openai", "openrouter"}:
            raise ValueError(
                f"Template '{template.get('name') or template.get('template_id')}' is local-only and cannot use "
                f"the public remote provider '{llm.provider}'. Choose llama.cpp, Ollama, or LM Studio."
            )

    def expected_report_provider(self) -> str:
        if self.config.mock_llm:
            return MockMeetingLLMClient.name
        llm = self._current_llm_config()
        return f"{llm.provider}:{llm.model}"

    def update_llm_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.mock_llm:
            raise ValueError("Provider switching is disabled while mock LLM mode is active.")
        provider = normalize_provider(payload.get("provider"))
        model = str(payload.get("model") or "").strip()
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        if not model:
            raise ValueError("Model is required.")
        current = self._settings.snapshot()
        overrides: dict[str, Any] = {
            "model": model,
            "timeout_seconds": current.timeout_seconds,
            "max_tokens": current.max_tokens,
            "section_max_tokens": current.section_max_tokens,
            "temperature": current.temperature,
            "client_name": current.client_name,
            "lane": current.lane,
            "enable_thinking": current.enable_thinking,
        }
        if base_url:
            overrides["base_url"] = base_url
        if provider == current.provider and current.api_key:
            overrides["api_key"] = current.api_key
        self._settings.replace(default_llm_config(provider, **overrides))
        return self.public_config()

    def list_provider_models(self, provider_value: Any, base_url_value: Any = "") -> dict[str, Any]:
        provider = normalize_provider(provider_value)
        base_url = str(base_url_value or "").strip().rstrip("/")
        current = self._current_llm_config()
        overrides: dict[str, Any] = {
            "timeout_seconds": min(float(current.timeout_seconds or 30.0), 30.0),
            "max_tokens": current.max_tokens,
            "section_max_tokens": current.section_max_tokens,
        }
        if base_url:
            overrides["base_url"] = base_url
        if provider == current.provider and current.api_key:
            overrides["api_key"] = current.api_key
        llm = default_llm_config(provider, **overrides)
        if self._provider_requires_api_key(llm.provider) and not llm.api_key:
            env_var = self._provider_api_key_env_var(llm.provider)
            raise ValueError(f"{llm.provider} requires {env_var} in .env or the server environment.")
        request = urllib.request.Request(
            f"{llm.base_url}/models",
            headers=self._model_list_headers(llm),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(llm.timeout_seconds, 30.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Model list request failed: {exc}") from exc
        model_ids = extract_model_ids(payload)
        return {
            "provider": provider,
            "base_url": llm.base_url,
            "models": model_ids,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if self.config.demo_transcript and self.config.demo_transcript.is_file():
            demo = self._load_demo_session()
            sessions.append({
                **demo["summary"],
                "source_kind": "demo_transcript",
                "has_cached_report": bool(self._read_runtime_cached_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)),
                "report_template_ids": self._cached_template_ids(DEMO_SESSION_ID),
            })
        for session in self.store.list_sessions("all"):
            item = dict(session)
            item["source_kind"] = "saved_session"
            session_id = str(item.get("id") or "")
            item["has_cached_report"] = bool(self._read_runtime_cached_report(session_id, STANDARD_TEMPLATE_ID))
            item["report_template_ids"] = self._cached_template_ids(session_id)
            sessions.append(item)
        return sessions

    def load_session(self, session_id: str) -> dict[str, Any]:
        if session_id == DEMO_SESSION_ID:
            if not self.config.demo_transcript or not self.config.demo_transcript.is_file():
                raise ValueError("Demo transcript is not configured.")
            return self._load_demo_session()
        return self.store.open_session(session_id)

    def get_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any]:
        template = self._resolve_template(template_id)
        template_id = str(template["template_id"])
        report_language = self._template_report_language(template)
        session = self.load_session(session_id)
        rows = [dict(row) for row in session.get("transcript_rows") or []]
        speaker_state = session.get("speaker_state") if isinstance(session.get("speaker_state"), dict) else {}
        revision_id = transcript_revision_id(rows, speaker_state)
        report = self._read_cached_report(session_id, template_id)
        available = (
            isinstance(report, dict)
            and report.get("transcript_revision_id") == revision_id
            and report.get("provider") == self.expected_report_provider()
            and report.get("report_language") == report_language
            and report.get("template_id") == template_id
            and report.get("template_revision") == template.get("revision_hash")
        )
        return {
            "available": available,
            "stale": bool(report and not available),
            "session": session.get("summary") or {},
            "report": report if available else None,
            "transcript_rows": rows,
            "template": self._public_template(template),
            "template_id": template_id,
            "report_language": report_language,
        }

    def generate_report(
        self,
        session_id: str,
        *,
        template_id: str = STANDARD_TEMPLATE_ID,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        request = self._capture_generation_request(session_id, template_id)
        return self._generate_captured_request(request, progress_callback or (lambda _event: None))

    def _capture_generation_request(
        self,
        session_id: str,
        template_id: str = STANDARD_TEMPLATE_ID,
    ) -> ReportGenerationRequest:
        template = self._resolve_template(template_id)
        template_id = str(template["template_id"])
        report_language = self._template_report_language(template)
        session = self.load_session(session_id)
        rows = [dict(row) for row in session.get("transcript_rows") or []]
        if not rows:
            raise ValueError("Selected session has no transcript rows.")
        speaker_state = session.get("speaker_state") if isinstance(session.get("speaker_state"), dict) else {}
        summary = session.get("summary") if isinstance(session.get("summary"), dict) else {}
        title = str(summary.get("title") or session_id)
        llm_config = self._current_llm_config()
        self._enforce_template_privacy(template, llm_config, mock_llm=self.config.mock_llm)
        if not self.config.mock_llm and self._provider_requires_api_key(llm_config.provider) and not llm_config.api_key:
            env_var = self._provider_api_key_env_var(llm_config.provider)
            raise ValueError(f"{llm_config.provider} requires {env_var} in .env or the server environment.")
        return ReportGenerationRequest(
            session_id=str(session_id),
            template_id=template_id,
            title=title,
            transcript_revision_id=transcript_revision_id(rows, speaker_state),
            report_language=report_language,
            transcript_rows_json=immutable_json(rows),
            speaker_state_json=immutable_json(speaker_state),
            summary_json=immutable_json(summary),
            template_json=immutable_json(template),
            llm_config=llm_config,
            mock_llm=bool(self.config.mock_llm),
            max_segment_rows=int(self.config.max_segment_rows),
            client_factory=self.client_factory,
        )

    def _generate_captured_request(
        self,
        request: ReportGenerationRequest,
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        rows = request.transcript_rows()
        speaker_state = request.speaker_state()
        summary = request.summary()
        template = request.template()
        client = self._new_client(
            request.llm_config,
            client_factory=request.client_factory,
            mock_llm=request.mock_llm,
        )
        pipeline = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=request.max_segment_rows,
            evidence_max_tokens=request.llm_config.max_tokens,
            section_max_tokens=request.llm_config.section_max_tokens,
            report_language=request.report_language,
            report_template=template,
            progress_callback=progress_callback,
        )
        report = pipeline.generate(
            session_id=request.session_id,
            transcript_rows=rows,
            speaker_state=speaker_state,
            title=request.title,
        )
        # This guard also catches accidental future pipeline changes that read
        # mutable live state instead of the captured request.
        if report.get("transcript_revision_id") != request.transcript_revision_id:
            raise RuntimeError("generated report provenance does not match captured transcript revision")
        self._write_cached_report(request.session_id, request.template_id, report)
        return {
            "available": True,
            "stale": False,
            "session": summary,
            "report": report,
            "transcript_rows": rows,
            "template": self._public_template(template),
            "template_id": request.template_id,
            "report_language": request.report_language,
        }

    def delete_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("Session id is required.")
        session = self.load_session(session_id)
        template = self._resolve_template(template_id)
        template_id = str(template["template_id"])
        deleted = self._report_cache.delete(
            session_id, template_id, legacy_template_id=STANDARD_TEMPLATE_ID
        )
        return {
            "deleted": deleted,
            "session": session.get("summary") or {},
            "report": None,
            "available": False,
            "stale": False,
            "transcript_rows": [dict(row) for row in session.get("transcript_rows") or []],
            "template": self._public_template(template),
            "template_id": template_id,
        }

    def start_generate_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("Session id is required.")
        request = self._capture_generation_request(session_id, template_id)
        return self._job_manager.submit(request)

    def auto_generate_ready_sessions(self) -> list[dict[str, Any]]:
        """Queue reports for newly finalized saved sessions that have no current report yet."""

        if not self.config.auto_generate:
            return []
        summaries = self.store.list_sessions("all")
        queued: list[dict[str, Any]] = []
        for session_id in self._auto_generation.claim_new_saved_sessions(summaries):
            try:
                if self.get_report(session_id, STANDARD_TEMPLATE_ID)["available"]:
                    continue
                queued.append(self.start_generate_report(session_id, STANDARD_TEMPLATE_ID))
            except (FileNotFoundError, ValueError, OSError):
                # A session may be edited or removed while the monitor scans it.
                self._auto_generation.release(session_id)
                continue
        return queued

    def get_generation_job(self, job_id: str) -> dict[str, Any]:
        return self._job_manager.get(job_id)

    def _current_llm_config(self) -> MeetingLLMConfig:
        return self._settings.snapshot()

    def _new_client(
        self,
        llm_config: MeetingLLMConfig | None = None,
        *,
        client_factory: Callable[[], StructuredChatClient] | None = None,
        mock_llm: bool | None = None,
    ) -> StructuredChatClient:
        factory = self.client_factory if client_factory is None else client_factory
        if factory is not None:
            return factory()
        use_mock = self.config.mock_llm if mock_llm is None else bool(mock_llm)
        if use_mock:
            return MockMeetingLLMClient()
        return OpenAICompatibleMeetingClient(llm_config or self._current_llm_config())

    def _chat_report_context(self, session_id: str, revision_id: str) -> dict[str, Any] | None:
        """Return compact current report context; transcript rows remain the only evidence."""

        result = self.get_report(session_id, STANDARD_TEMPLATE_ID)
        report = result.get("report") if result.get("available") else None
        if not isinstance(report, dict) or str(report.get("transcript_revision_id") or "") != revision_id:
            return None
        sections = report.get("sections") if isinstance(report.get("sections"), dict) else {}
        compact_sections: dict[str, Any] = {}
        for key, value in list(sections.items())[:12]:
            if not isinstance(value, dict):
                continue
            compact_sections[str(key)] = {
                "summary": str(value.get("summary") or ""),
                "items": [
                    {
                        "title": str(item.get("title") or ""),
                        "body": str(item.get("body") or ""),
                    }
                    for item in (value.get("items") or [])[:8]
                    if isinstance(item, dict)
                ],
            }
        return {
            "meeting_id": session_id,
            "title": str(report.get("title") or (result.get("session") or {}).get("title") or session_id),
            "template_id": str(report.get("template_id") or STANDARD_TEMPLATE_ID),
            "sections": compact_sections,
        }

    def close(self) -> None:
        self._chat_jobs.close()
        self._job_manager.close()

    @staticmethod
    def _model_list_headers(llm: MeetingLLMConfig) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if llm.api_key:
            headers["Authorization"] = f"Bearer {llm.api_key}"
        return headers

    @staticmethod
    def _provider_api_key_env_var(provider: str) -> str:
        option = LLM_PROVIDER_OPTIONS.get(provider) or {}
        return str(option.get("api_key_env_var") or "")

    @classmethod
    def _provider_requires_api_key(cls, provider: str) -> bool:
        return bool(cls._provider_api_key_env_var(provider))

    @classmethod
    def _provider_api_key_configured(cls, llm: MeetingLLMConfig) -> bool:
        if not cls._provider_requires_api_key(llm.provider):
            return True
        return bool(llm.api_key)

    def _legacy_cache_path(self, session_id: str) -> Path:
        return self._report_cache.legacy_path(session_id)

    def _cache_path(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> Path:
        return self._report_cache.path(session_id, template_id)

    def _read_cached_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any] | None:
        report = self._report_cache.read(
            session_id, template_id, legacy_template_id=STANDARD_TEMPLATE_ID
        )
        return sanitize_report_output(report) if isinstance(report, dict) else None

    def _read_runtime_cached_report(
        self,
        session_id: str,
        template_id: str = STANDARD_TEMPLATE_ID,
    ) -> dict[str, Any] | None:
        template = self._resolve_template(template_id)
        report = self._read_cached_report(session_id, template_id)
        if not isinstance(report, dict):
            return None
        if report.get("provider") != self.expected_report_provider():
            return None
        if report.get("report_language") != self._template_report_language(template):
            return None
        if report.get("template_id") != template.get("template_id"):
            return None
        if report.get("template_revision") != template.get("revision_hash"):
            return None
        return report

    def _cached_template_ids(self, session_id: str) -> list[str]:
        result: list[str] = []
        for template in self.list_report_templates():
            template_id = str(template.get("template_id") or "")
            if template_id and self._read_runtime_cached_report(session_id, template_id) is not None:
                result.append(template_id)
        return result

    def _write_cached_report(self, session_id: str, template_id: str, report: dict[str, Any]) -> None:
        self._report_cache.write(session_id, template_id, report)

    def _load_demo_session(self) -> dict[str, Any]:
        path = self.config.demo_transcript
        if path is None:
            raise ValueError("Demo transcript is not configured.")
        rows = parse_whospeakslive_transcript(path)
        speaker_names = sorted({str(row["speaker_name"]) for row in rows})
        speakers = [
            {
                "id": speaker_id_from_name(name),
                "name": name,
                "display_name": name,
                "source": "transcript",
            }
            for name in speaker_names
        ]
        duration = max((float(row.get("end") or 0.0) for row in rows), default=0.0)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        cached_report = self._read_runtime_cached_report(DEMO_SESSION_ID, STANDARD_TEMPLATE_ID)
        summary = {
            "id": DEMO_SESSION_ID,
            "title": "WhoSpeaksLive demo meeting",
            "created_at": updated_at,
            "updated_at": updated_at,
            "started_at": updated_at,
            "ended_at": updated_at,
            "archived": False,
            "duration_seconds": round(duration, 4),
            "source": {"kind": "demo_transcript", "path": str(path)},
            "speaker_count": len(speakers),
            "speaker_names": speaker_names,
            "transcript_rows": len(rows),
            "status_label": "Demo",
            "has_transcript": True,
            "has_speakers": True,
            "has_meeting_intelligence": bool(cached_report),
        }
        return {
            "summary": summary,
            "manifest": summary,
            "transcript_rows": rows,
            "speaker_state": {"speakers": speakers},
            "speaker_profiles": [],
            "live_speaker_profiles": [],
            "embedding_count": 0,
            "embeddings_available": False,
            "meeting_intelligence": {
                "available": bool(cached_report),
                "report": cached_report,
            },
        }


def make_handler(service: MeetingIntelligenceService) -> type[BaseHTTPRequestHandler]:
    class MeetingIntelligenceHandler(BaseHTTPRequestHandler):
        server_version = "WhoSpeaksMeetingIntelligence/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"", "/"}:
                    self._send_html(read_web_asset("reports/index.html").decode("utf-8"))
                    return
                if parsed.path == "/health":
                    self._send_json({"ok": True, "ready": True, "service": "meeting-intelligence"})
                    return
                report_asset = {
                    "/assets/web/reports/styles-base.css": "reports/styles-base.css",
                    "/assets/web/reports/styles-components.css": "reports/styles-components.css",
                    "/assets/web/reports/report_builder.js": "reports/report_builder.js",
                    "/assets/web/reports/app.js": "reports/app.js",
                }.get(parsed.path)
                if report_asset:
                    self._send_asset(report_asset)
                    return
                if parsed.path == "/api/config":
                    self._send_json({"config": service.public_config()})
                    return
                if parsed.path == "/api/llm-models":
                    query = parse_qs(parsed.query)
                    provider = str((query.get("provider") or [""])[0])
                    base_url = str((query.get("base_url") or [""])[0])
                    self._send_json(service.list_provider_models(provider, urllib.parse.unquote(base_url)))
                    return
                if parsed.path == "/api/sessions":
                    self._send_json({"sessions": service.list_sessions()})
                    return
                if parsed.path == "/api/templates":
                    self._send_json({
                        "templates": service.list_report_templates(),
                        "standard_template_id": STANDARD_TEMPLATE_ID,
                    })
                    return
                if parsed.path == "/api/template":
                    template_id = single_query_value(parsed.query, "template_id")
                    self._send_json({"template": service.get_report_template(template_id)})
                    return
                if parsed.path == "/api/report":
                    session_id = single_query_value(parsed.query, "session_id")
                    template_id = optional_query_value(parsed.query, "template_id", STANDARD_TEMPLATE_ID)
                    self._send_json(service.get_report(session_id, template_id))
                    return
                if parsed.path == "/api/generate-status":
                    job_id = single_query_value(parsed.query, "job_id")
                    self._send_json({"job": service.get_generation_job(job_id)})
                    return
                if parsed.path == "/api/chat/job":
                    job_id = single_query_value(parsed.query, "job_id")
                    self._send_json({"job": service.get_chat_job(job_id)})
                    return
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path not in {
                    "/api/generate", "/api/generate-async", "/api/delete-report", "/api/llm-config",
                    "/api/templates/save", "/api/templates/delete", "/api/templates/clone",
                    "/api/chat/scope", "/api/chat/ask-async", "/api/chat/clear",
                }:
                    self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json_body()
                if parsed.path == "/api/chat/scope":
                    self._send_json(service.chat_scope(_session_ids(payload)))
                    return
                if parsed.path == "/api/chat/ask-async":
                    self._send_json({"job": service.start_chat(
                        _session_ids(payload),
                        str(payload.get("question") or ""),
                        provisional=bool(payload.get("provisional")),
                    )})
                    return
                if parsed.path == "/api/chat/clear":
                    self._send_json(service.clear_chat(_session_ids(payload)))
                    return
                if parsed.path == "/api/llm-config":
                    self._send_json({"config": service.update_llm_config(payload)})
                    return
                if parsed.path == "/api/templates/save":
                    self._send_json({"template": service.save_report_template(payload)})
                    return
                if parsed.path == "/api/templates/delete":
                    deleted = service.delete_report_template(str(payload.get("template_id") or ""))
                    self._send_json({"deleted": deleted})
                    return
                if parsed.path == "/api/templates/clone":
                    template = service.clone_report_template(
                        str(payload.get("template_id") or ""),
                        str(payload.get("name") or ""),
                    )
                    self._send_json({"template": template})
                    return
                session_id = str(payload.get("session_id") or "").strip()
                template_id = str(payload.get("template_id") or STANDARD_TEMPLATE_ID).strip()
                if parsed.path == "/api/delete-report":
                    self._send_json(service.delete_report(session_id, template_id))
                    return
                if parsed.path == "/api/generate-async":
                    self._send_json({"job": service.start_generate_report(session_id, template_id)})
                    return
                self._send_json(service.generate_report(session_id, template_id=template_id))
            except GenerationQueueFullError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.TOO_MANY_REQUESTS)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            raw = self.rfile.read(max(0, length))
            if not raw:
                return {}
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else {}

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_text: str) -> None:
            body = html_text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, name: str) -> None:
            body = read_web_asset(name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", web_asset_content_type(name))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MeetingIntelligenceHandler


def single_query_value(query: str, key: str) -> str:
    values = parse_qs(query).get(key) or []
    if not values or not str(values[0]).strip():
        raise ValueError(f"Missing query parameter: {key}")
    return str(values[0]).strip()


def optional_query_value(query: str, key: str, default: str = "") -> str:
    values = parse_qs(query).get(key) or []
    if not values or not str(values[0]).strip():
        return default
    return str(values[0]).strip()


def _session_ids(payload: dict[str, Any]) -> list[str]:
    values = payload.get("session_ids")
    if not isinstance(values, list):
        raise ValueError("session_ids must be an array.")
    return [str(value or "").strip() for value in values if str(value or "").strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve WhoSpeaks Meeting Intelligence — Reports + Ask.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--chat-dir", type=Path, default=DEFAULT_CHAT_DIR)
    parser.add_argument("--text-index-db", type=Path, default=DEFAULT_TEXT_INDEX_DB)
    parser.add_argument("--demo-transcript", type=Path)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--llm-provider",
        default="llama_cpp",
        choices=("llama_cpp", "ollama", "lm_studio", "openai_compatible", "openai", "openrouter"),
    )
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--section-max-tokens", type=int, default=4096)
    parser.add_argument("--max-segment-rows", type=int, default=80)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--text-embedding-base-url", default="")
    parser.add_argument("--text-embedding-model", default="")
    parser.add_argument("--text-embedding-api-key-env", default="")
    parser.add_argument("--text-embedding-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--text-index-poll-seconds", type=float, default=10.0)
    parser.add_argument("--auto-generate", action="store_true")
    parser.add_argument(
        "--auto-generate-poll-seconds",
        type=float,
        default=10.0,
        help="How often --auto-generate checks for newly saved sessions.",
    )
    parser.add_argument(
        "--report-language",
        default="en",
        help="Language for generated report content; accepts every WhoSpeaks language code, for example es or de.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MeetingIntelligenceServerConfig:
    env_file = getattr(args, "env_file", None)
    if env_file:
        load_env_file(Path(env_file))
    overrides: dict[str, Any] = {
        "timeout_seconds": args.timeout_seconds,
        "max_tokens": args.max_tokens,
        "section_max_tokens": args.section_max_tokens,
    }
    if args.llm_base_url:
        overrides["base_url"] = args.llm_base_url
    if args.llm_model:
        overrides["model"] = args.llm_model
    if args.llm_api_key:
        overrides["api_key"] = args.llm_api_key
    return MeetingIntelligenceServerConfig(
        session_dir=args.session_dir.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve(),
        template_dir=args.template_dir.expanduser().resolve(),
        chat_dir=args.chat_dir.expanduser().resolve(),
        text_index_db=args.text_index_db.expanduser().resolve(),
        demo_transcript=args.demo_transcript.expanduser().resolve() if args.demo_transcript else None,
        llm_config=default_llm_config(args.llm_provider, **overrides),
        mock_llm=bool(args.mock_llm),
        max_segment_rows=max(12, int(args.max_segment_rows)),
        auto_generate=bool(args.auto_generate),
        auto_generate_poll_seconds=max(1.0, float(args.auto_generate_poll_seconds)),
        report_language=args.report_language,
        text_embedding=TextEmbeddingConfig(
            base_url=str(args.text_embedding_base_url or "").strip(),
            model=str(args.text_embedding_model or "").strip(),
            api_key_env=str(args.text_embedding_api_key_env or "").strip(),
            timeout_seconds=max(1.0, float(args.text_embedding_timeout_seconds)),
        ),
        text_index_poll_seconds=max(1.0, float(args.text_index_poll_seconds)),
    )


def runtime_config_from_args(args: argparse.Namespace) -> MeetingIntelligenceRuntimeConfig:
    meeting = config_from_args(args)
    if not meeting.mock_llm and not meeting.llm_config.model.strip():
        raise ValueError(
            "Meeting Intelligence requires an explicit --llm-model (or "
            "WHOSPEAKS_MI_LLM_MODEL). No installed model is assumed."
        )
    return MeetingIntelligenceRuntimeConfig(
        host=str(args.host),
        port=int(args.port),
        meeting=meeting,
    )


def run_server(runtime: MeetingIntelligenceRuntimeConfig) -> None:
    config = runtime.meeting
    service = MeetingIntelligenceService(config)
    handler = make_handler(service)
    httpd = ThreadingHTTPServer((runtime.host, runtime.port), handler)
    print(f"Meeting intelligence server: http://{runtime.host}:{runtime.port}/", flush=True)
    monitor: AutoGenerationMonitor | None = None
    index_monitor: AutoGenerationMonitor | None = None
    try:
        if config.auto_generate:
            monitor = AutoGenerationMonitor(
                service.auto_generate_ready_sessions,
                interval_seconds=config.auto_generate_poll_seconds,
            )
            monitor.start()
        index_monitor = AutoGenerationMonitor(
            service.auto_index_changed_sessions,
            interval_seconds=config.text_index_poll_seconds,
        )
        index_monitor.start()
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        failures: list[str] = []
        for close in (
            monitor.close if monitor is not None else None,
            index_monitor.close if index_monitor is not None else None,
            httpd.server_close,
            service.close,
        ):
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("meeting intelligence shutdown failed: " + "; ".join(failures))


def _run_auto_generation_monitor(service: MeetingIntelligenceService, stop: threading.Event) -> None:
    """Compatibility wrapper for callers that still provide their own stop event."""

    interval = max(1.0, float(service.config.auto_generate_poll_seconds))
    while not stop.is_set():
        service.auto_generate_ready_sessions()
        stop.wait(interval)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_server(runtime_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
