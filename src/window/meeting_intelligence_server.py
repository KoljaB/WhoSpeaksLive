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
import uuid

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
from window.report_templates import (
    STANDARD_TEMPLATE_ID,
    ReportTemplateStore,
    builtin_report_templates,
    get_builtin_report_template,
    validate_report_template,
)
from window.session_store import DEFAULT_SESSION_DIR, SessionStore


DEMO_SESSION_ID = "demo-whospeakslive-transcript"
DEFAULT_CACHE_DIR = RUNTIME_DIR / "meeting_intelligence_reports"
DEFAULT_TEMPLATE_DIR = RUNTIME_DIR / "report_templates"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
LLM_PROVIDER_OPTIONS = {
    "llama_cpp": {
        "label": "llama.cpp",
        "default_base_url": "http://127.0.0.1:8081/v1",
        "models": ["gemma-4-12b-it-Q6_K.gguf", "local"],
        "requires_api_key": False,
        "api_key_env_var": "",
    },
    "ollama": {
        "label": "Ollama",
        "default_base_url": "http://127.0.0.1:11434/v1",
        "models": ["gemma3", "llama3.1"],
        "requires_api_key": False,
        "api_key_env_var": "",
    },
    "lm_studio": {
        "label": "LM Studio",
        "default_base_url": "http://127.0.0.1:1234/v1",
        "models": ["local-model"],
        "requires_api_key": False,
        "api_key_env_var": "",
    },
    "openai": {
        "label": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
        "requires_api_key": True,
        "api_key_env_var": "OPENAI_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter",
        "default_base_url": "https://openrouter.ai/api/v1",
        "models": ["google/gemma-3-12b-it"],
        "requires_api_key": True,
        "api_key_env_var": "OPENROUTER_API_KEY",
    },
}
TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?::\d+){1,2}(?:\.\d+)?)\s+-\s+"
    r"(?P<end>\d+(?::\d+){1,2}(?:\.\d+)?)\]\s+"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


@dataclass(frozen=True)
class MeetingIntelligenceServerConfig:
    session_dir: Path = DEFAULT_SESSION_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    template_dir: Path = DEFAULT_TEMPLATE_DIR
    demo_transcript: Path | None = None
    llm_config: MeetingLLMConfig = field(default_factory=default_llm_config)
    mock_llm: bool = False
    max_segment_rows: int = 80
    auto_generate: bool = False
    auto_generate_poll_seconds: float = 10.0
    report_language: str = "en"


@dataclass
class GenerationJob:
    job_id: str
    session_id: str
    template_id: str = STANDARD_TEMPLATE_ID
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued"
    detail: str = ""
    percent: int = 0
    current: int = 0
    total: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    completed_at: str = ""
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


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
        self._llm_config = config.llm_config
        self._settings_lock = threading.Lock()
        self._jobs: dict[str, GenerationJob] = {}
        self._jobs_lock = threading.Lock()
        self._auto_generation_baseline_complete = False
        self._auto_generation_existing_saved_ids: set[str] = set()

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
        }

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
        with self._settings_lock:
            current = self._llm_config
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
            self._llm_config = default_llm_config(provider, **overrides)
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
        client = self._new_client(llm_config)
        pipeline = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=self.config.max_segment_rows,
            evidence_max_tokens=llm_config.max_tokens,
            section_max_tokens=llm_config.section_max_tokens,
            report_language=report_language,
            report_template=template,
            progress_callback=progress_callback,
        )
        report = pipeline.generate(
            session_id=session_id,
            transcript_rows=rows,
            speaker_state=speaker_state,
            title=title,
        )
        self._write_cached_report(session_id, template_id, report)
        return {
            "available": True,
            "stale": False,
            "session": summary,
            "report": report,
            "transcript_rows": rows,
            "template": self._public_template(template),
            "template_id": template_id,
            "report_language": report_language,
        }

    def delete_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("Session id is required.")
        session = self.load_session(session_id)
        template = self._resolve_template(template_id)
        template_id = str(template["template_id"])
        path = self._cache_path(session_id, template_id)
        deleted = False
        if path.is_file():
            path.unlink()
            deleted = True
        legacy_path = self._legacy_cache_path(session_id)
        if template_id == STANDARD_TEMPLATE_ID and legacy_path.is_file():
            legacy_path.unlink()
            deleted = True
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
        template = self._resolve_template(template_id)
        template_id = str(template["template_id"])
        with self._jobs_lock:
            for job in self._jobs.values():
                if job.session_id == session_id and job.template_id == template_id and job.status in {"queued", "running"}:
                    return self._job_snapshot(job)
            job = GenerationJob(
                job_id=f"mirjob_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                template_id=template_id,
                message="Queued report generation",
            )
            self._jobs[job.job_id] = job
            snapshot = self._job_snapshot(job)
        thread = threading.Thread(
            target=self._run_generation_job,
            args=(job.job_id, session_id, template_id),
            name=f"meeting-intelligence-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return snapshot

    def auto_generate_ready_sessions(self) -> list[dict[str, Any]]:
        """Queue reports for newly finalized saved sessions that have no current report yet."""

        if not self.config.auto_generate:
            return []
        summaries = self.store.list_sessions("all")
        if not self._auto_generation_baseline_complete:
            self._auto_generation_existing_saved_ids = {
                str(summary.get("id") or "").strip()
                for summary in summaries
                if str(summary.get("status_label") or "").casefold() == "saved"
            }
            self._auto_generation_baseline_complete = True
            return []
        queued: list[dict[str, Any]] = []
        for summary in summaries:
            if str(summary.get("status_label") or "").casefold() != "saved":
                continue
            if not bool(summary.get("has_transcript")):
                continue
            session_id = str(summary.get("id") or "").strip()
            if not session_id or session_id in self._auto_generation_existing_saved_ids:
                continue
            try:
                if self.get_report(session_id, STANDARD_TEMPLATE_ID)["available"]:
                    self._auto_generation_existing_saved_ids.add(session_id)
                    continue
                queued.append(self.start_generate_report(session_id, STANDARD_TEMPLATE_ID))
                self._auto_generation_existing_saved_ids.add(session_id)
            except (FileNotFoundError, ValueError, OSError):
                # A session may be edited or removed while the monitor scans it.
                continue
        return queued

    def get_generation_job(self, job_id: str) -> dict[str, Any]:
        job_id = str(job_id or "").strip()
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Generation job not found.")
            return self._job_snapshot(job)

    def _run_generation_job(self, job_id: str, session_id: str, template_id: str) -> None:
        self._update_job(
            job_id,
            status="running",
            stage="starting",
            message="Starting report generation",
            detail="Loading transcript and speaker state",
            percent=0,
        )
        try:
            result = self.generate_report(
                session_id,
                template_id=template_id,
                progress_callback=lambda event: self._apply_progress_event(job_id, event),
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                message="Report generation failed",
                detail=str(exc),
                error=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            return
        report = result.get("report") if isinstance(result, dict) else None
        detail = ""
        if isinstance(report, dict):
            detail = (
                f"{len(report.get('evidence_index') or [])} evidence anchors, "
                f"{len(report.get('sections') or {})} sections"
            )
        self._update_job(
            job_id,
            status="succeeded",
            stage="completed",
            message="Report generated",
            detail=detail,
            percent=100,
            completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _apply_progress_event(self, job_id: str, event: dict[str, Any]) -> None:
        clean = {
            "stage": str(event.get("stage") or ""),
            "message": str(event.get("message") or ""),
            "detail": str(event.get("detail") or ""),
            "percent": int(event.get("percent") or 0),
            "current": int(event.get("current") or 0),
            "total": int(event.get("total") or 0),
            "at": str(event.get("at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        }
        self._update_job(
            job_id,
            stage=clean["stage"],
            message=clean["message"],
            detail=clean["detail"],
            percent=clean["percent"],
            current=clean["current"],
            total=clean["total"],
            append_event=clean,
        )

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            append_event = updates.pop("append_event", None)
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if isinstance(append_event, dict):
                job.events.append(append_event)
                if len(job.events) > 80:
                    del job.events[:-80]

    @staticmethod
    def _job_snapshot(job: GenerationJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "template_id": job.template_id,
            "status": job.status,
            "stage": job.stage,
            "message": job.message,
            "detail": job.detail,
            "percent": job.percent,
            "current": job.current,
            "total": job.total,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "events": list(job.events[-12:]),
        }

    def _current_llm_config(self) -> MeetingLLMConfig:
        with self._settings_lock:
            return self._llm_config

    def _new_client(self, llm_config: MeetingLLMConfig | None = None) -> StructuredChatClient:
        if self.client_factory is not None:
            return self.client_factory()
        if self.config.mock_llm:
            return MockMeetingLLMClient()
        return OpenAICompatibleMeetingClient(llm_config or self._current_llm_config())

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
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-")
        if not clean:
            clean = "session"
        suffix = stable_hash(session_id, length=10)
        return self.config.cache_dir / f"{clean[:80]}-{suffix}.json"

    def _cache_path(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> Path:
        clean_session = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-") or "session"
        clean_template = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(template_id or "").strip()).strip("-") or "template"
        session_suffix = stable_hash(session_id, length=10)
        template_suffix = stable_hash(template_id, length=10)
        return self.config.cache_dir / (
            f"{clean_session[:64]}-{session_suffix}--{clean_template[:64]}-{template_suffix}.json"
        )

    def _read_cached_report(self, session_id: str, template_id: str = STANDARD_TEMPLATE_ID) -> dict[str, Any] | None:
        path = self._cache_path(session_id, template_id)
        if not path.is_file() and template_id == STANDARD_TEMPLATE_ID:
            path = self._legacy_cache_path(session_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        report = payload.get("report") if isinstance(payload, dict) else None
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
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "template_id": template_id,
            "report": report,
        }
        self._cache_path(session_id, template_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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


def parse_whospeakslive_transcript(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TRANSCRIPT_LINE_RE.match(line.strip())
        if not match:
            continue
        speaker_name = " ".join(match.group("speaker").split())
        text = " ".join(match.group("text").split())
        if not text:
            continue
        index = len(rows) + 1
        rows.append({
            "index": index,
            "row_id": f"demo_row_{index:04d}",
            "start": parse_timecode(match.group("start")),
            "end": parse_timecode(match.group("end")),
            "text": text,
            "assigned_speaker": speaker_id_from_name(speaker_name),
            "speaker_name": speaker_name,
            "source_line": line_number,
        })
    return rows


def parse_timecode(value: str) -> float:
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return round(minutes * 60.0 + seconds, 4)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return round(hours * 3600.0 + minutes * 60.0 + seconds, 4)
    raise ValueError(f"Unsupported transcript timecode: {value}")


def speaker_id_from_name(name: str) -> str:
    match = re.search(r"(\d+)$", str(name or "").strip())
    if match:
        return f"S{int(match.group(1))}"
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "speaker").strip()).strip("_")
    return clean[:40] or "speaker"


def normalize_provider(value: Any) -> str:
    provider = str(value or "llama_cpp").strip().lower().replace("-", "_")
    if provider not in LLM_PROVIDER_OPTIONS:
        raise ValueError(f"Unsupported meeting LLM provider: {value}")
    return provider


def provider_options_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": provider,
            "label": str(option["label"]),
            "default_base_url": str(option["default_base_url"]),
            "models": list(option["models"]),
            "requires_api_key": bool(option["requires_api_key"]),
            "api_key_env_var": str(option["api_key_env_var"]),
        }
        for provider, option in LLM_PROVIDER_OPTIONS.items()
    ]


def extract_model_ids(payload: dict[str, Any]) -> list[str]:
    items = payload.get("data") if isinstance(payload, dict) else []
    raw_ids: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                raw_ids.append(str(item.get("id") or "").strip())
            else:
                raw_ids.append(str(item or "").strip())
    return sort_model_ids([
        model_id
        for model_id in unique_strings(raw_ids)
        if is_likely_text_generation_model(model_id)
    ])


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def is_likely_text_generation_model(model_id: str) -> bool:
    text = model_id.lower()
    if not text:
        return False
    excluded = (
        "audio",
        "dall",
        "embedding",
        "image",
        "moderation",
        "realtime",
        "sora",
        "speech",
        "transcribe",
        "tts",
        "whisper",
    )
    if any(part in text for part in excluded):
        return False
    return text.startswith(("gpt-", "o1", "o3", "o4"))


def sort_model_ids(model_ids: list[str]) -> list[str]:
    def sort_key(model_id: str) -> tuple[int, str]:
        text = model_id.lower()
        if "nano" in text:
            return (0, text)
        if "mini" in text:
            return (1, text)
        if "luna" in text:
            return (2, text)
        if "terra" in text:
            return (3, text)
        if "sol" in text:
            return (4, text)
        return (5, text)

    return sorted(model_ids, key=sort_key)


def load_env_file(path: Path | None = None) -> bool:
    env_path = (path or DEFAULT_ENV_FILE).expanduser()
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = strip_env_value(value.strip())
        os.environ.setdefault(key, value)
    return True


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def make_handler(service: MeetingIntelligenceService) -> type[BaseHTTPRequestHandler]:
    class MeetingIntelligenceHandler(BaseHTTPRequestHandler):
        server_version = "WhoSpeaksMeetingIntelligence/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"", "/"}:
                    self._send_html(PAGE_HTML)
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
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path not in {
                    "/api/generate", "/api/generate-async", "/api/delete-report", "/api/llm-config",
                    "/api/templates/save", "/api/templates/delete", "/api/templates/clone",
                }:
                    self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json_body()
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve WhoSpeaks meeting intelligence reports.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--demo-transcript", type=Path)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--llm-provider",
        default="llama_cpp",
        choices=("llama_cpp", "ollama", "lm_studio", "openai", "openrouter"),
    )
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--section-max-tokens", type=int, default=4096)
    parser.add_argument("--max-segment-rows", type=int, default=80)
    parser.add_argument("--mock-llm", action="store_true")
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
        demo_transcript=args.demo_transcript.expanduser().resolve() if args.demo_transcript else None,
        llm_config=default_llm_config(args.llm_provider, **overrides),
        mock_llm=bool(args.mock_llm),
        max_segment_rows=max(12, int(args.max_segment_rows)),
        auto_generate=bool(args.auto_generate),
        auto_generate_poll_seconds=max(1.0, float(args.auto_generate_poll_seconds)),
        report_language=args.report_language,
    )


def run_server(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    service = MeetingIntelligenceService(config)
    handler = make_handler(service)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Meeting intelligence server: http://{args.host}:{args.port}/", flush=True)
    monitor_stop = threading.Event()
    monitor: threading.Thread | None = None
    if config.auto_generate:
        monitor = threading.Thread(
            target=_run_auto_generation_monitor,
            args=(service, monitor_stop),
            name="meeting-intelligence-auto-generate",
            daemon=True,
        )
        monitor.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
        httpd.server_close()


def _run_auto_generation_monitor(service: MeetingIntelligenceService, stop: threading.Event) -> None:
    """Watch saved sessions so a report starts as soon as the live run is finalized."""

    interval = max(1.0, float(service.config.auto_generate_poll_seconds))
    while not stop.is_set():
        try:
            service.auto_generate_ready_sessions()
        except Exception as exc:
            print(f"Meeting intelligence auto-generation scan failed: {exc}", flush=True)
        stop.wait(interval)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_server(args)
    return 0


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WhoSpeaks Meeting Intelligence</title>
  <style>
    :root {
      --paper: #f3f5f2;
      --panel: #ffffff;
      --panel-soft: #f8faf8;
      --ink: #111827;
      --muted: #64706b;
      --line: #d7ddd8;
      --line-strong: #bdc9c2;
      --accent: #0f766e;
      --accent-strong: #0b5d55;
      --accent-quiet: #e5f3ef;
      --slate: #334155;
      --amber: #b45309;
      --red: #b91c1c;
      --soft: #eef6f3;
      --shadow: 0 10px 26px rgba(17, 24, 39, 0.08);
      --shadow-soft: 0 1px 2px rgba(17, 24, 39, 0.06);
    }

    * {
      box-sizing: border-box;
    }

    [hidden] { display: none !important; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #f7f9f7 0%, var(--paper) 280px),
        var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button,
    input,
    select,
    textarea {
      font: inherit;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
    }

    .sidebar {
      border-right: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      backdrop-filter: blur(12px);
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr);
      min-height: 100vh;
      min-width: 0;
      overflow: hidden;
    }

    .brand {
      padding: 20px 20px 16px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }

    .brand-mark {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 12px;
      align-items: center;
    }

    .logo {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(135deg, var(--accent), #2563eb);
      box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22);
      font-weight: 800;
      letter-spacing: 0;
    }

    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.15;
      font-weight: 760;
    }

    .provider {
      margin-top: 16px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 4px 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 10px 11px;
    }

    .provider strong {
      color: var(--slate);
    }

    .provider-kicker {
      color: var(--slate);
      font-size: 11px;
      font-weight: 780;
    }

    .provider-model,
    .provider-url {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .provider-url {
      grid-column: 1 / -1;
    }

    .provider-connected {
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      gap: 6px;
      color: var(--accent-strong);
      font-size: 11px;
      font-weight: 720;
    }

    .provider-connected.warn {
      color: var(--amber);
    }

    .provider-health {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }

    .provider-health.warn {
      background: var(--amber);
      box-shadow: 0 0 0 4px rgba(180, 83, 9, 0.12);
    }

    .provider-controls {
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }

    .field {
      display: grid;
      gap: 5px;
      color: var(--slate);
      font-size: 11px;
      line-height: 1.25;
      font-weight: 760;
    }

    .field input,
    .field select {
      min-width: 0;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 8px 9px;
      outline: none;
      box-shadow: var(--shadow-soft);
      font-size: 12px;
      font-weight: 560;
    }

    .field input:focus,
    .field select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }

    .provider-control-grid {
      display: grid;
      grid-template-columns: minmax(0, 0.86fr) minmax(0, 1.14fr);
      gap: 8px;
      min-width: 0;
    }

    .provider-action-row {
      display: grid;
      grid-template-columns: auto auto minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }

    .provider-action-row .btn {
      min-height: 34px;
      padding: 7px 10px;
      font-size: 12px;
    }

    .provider-hint {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .toolbar {
      padding: 14px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 12px;
      min-width: 0;
    }

    .search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px 10px 34px;
      background: var(--panel);
      color: var(--ink);
      outline: none;
      box-shadow: var(--shadow-soft);
      background-image: url("data:image/svg+xml,%3Csvg width='16' height='16' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='m21 21-4.3-4.3M10.8 18a7.2 7.2 0 1 1 0-14.4 7.2 7.2 0 0 1 0 14.4Z' stroke='%2364706b' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: 12px 50%;
    }

    .search:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }

    .button-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto 42px;
      gap: 8px;
      min-width: 0;
    }

    .btn {
      min-height: 38px;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 8px 12px;
      cursor: pointer;
      white-space: nowrap;
      box-shadow: var(--shadow-soft);
      font-weight: 680;
      transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }

    .btn span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .btn svg {
      width: 17px;
      height: 17px;
      flex: 0 0 auto;
    }

    .btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .btn:hover {
      border-color: var(--accent);
      box-shadow: 0 5px 14px rgba(15, 118, 110, 0.12);
      transform: translateY(-1px);
    }

    .btn.primary:hover {
      background: var(--accent-strong);
    }

    .btn.danger {
      color: #991b1b;
      border-color: rgba(153, 27, 27, 0.25);
      background: #fff1f2;
    }

    .btn.danger:hover {
      border-color: rgba(153, 27, 27, 0.55);
      background: #ffe4e6;
    }

    .btn.danger.confirming {
      color: #7f1d1d;
      border-color: rgba(153, 27, 27, 0.65);
      background: #fecaca;
    }

    .btn:disabled {
      cursor: wait;
      opacity: 0.7;
    }

    .sessions {
      overflow: auto;
      padding: 10px 10px 18px;
      display: grid;
      align-content: start;
      gap: 7px;
      min-width: 0;
    }

    .sessions-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--slate);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 8px 4px 4px;
    }

    .session {
      width: 100%;
      max-width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: var(--ink);
      padding: 12px 12px 12px 14px;
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 10px;
      cursor: pointer;
      position: relative;
      transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }

    .session::before {
      content: "";
      position: absolute;
      left: 0;
      top: 10px;
      bottom: 10px;
      width: 3px;
      border-radius: 999px;
      background: transparent;
    }

    .session:hover,
    .session.active {
      background: var(--panel);
      border-color: var(--line);
      box-shadow: 0 6px 18px rgba(17, 24, 39, 0.07);
    }

    .session.active {
      border-color: rgba(15, 118, 110, 0.42);
      background: linear-gradient(90deg, #ffffff 0%, #fbfdfc 100%);
    }

    .session.active::before {
      background: var(--accent);
    }

    .session-title {
      font-weight: 740;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }

    .session-title-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: start;
      gap: 8px;
    }

    .session-time {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      text-align: right;
      white-space: nowrap;
      max-width: 78px;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-top: 1px;
    }

    .session-icon {
      width: 34px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--accent);
      background: #fbfdfc;
    }

    .session-icon svg {
      width: 18px;
      height: 18px;
    }

    .session-main {
      min-width: 0;
      display: grid;
      gap: 6px;
    }

    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      background: #f6f8f6;
      font-size: 12px;
      line-height: 1.2;
      font-weight: 620;
    }

    .badge.hot {
      color: var(--accent-strong);
      border-color: rgba(15, 118, 110, 0.35);
      background: var(--accent-quiet);
    }

    .badge.warn {
      color: var(--amber);
      border-color: rgba(180, 83, 9, 0.35);
      background: #fff7ed;
    }

    .badge.evidence-link {
      appearance: none;
      display: inline-flex;
      align-items: center;
      border-style: solid;
      font: inherit;
      font-weight: 680;
      text-decoration: none;
      cursor: pointer;
    }

    .badge.evidence-link:hover,
    .badge.evidence-link:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
      outline: none;
      background: #dff1ec;
    }

    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-height: 100vh;
      background: #eef2ef;
    }

    .topbar {
      padding: 20px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(10px);
      display: grid;
      gap: 11px;
      position: sticky;
      top: 0;
      z-index: 5;
    }

    .title-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    .topbar h2 {
      margin: 0;
      font-size: 25px;
      line-height: 1.2;
      font-weight: 780;
      overflow-wrap: anywhere;
    }

    .report-meta-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
      min-width: 0;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      min-height: 18px;
      font-weight: 560;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 7px;
    }

    .status-state {
      color: var(--accent-strong);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 680;
    }

    .status-dot {
      width: 16px;
      height: 16px;
      border-radius: 999px;
      display: inline-grid;
      place-items: center;
      background: var(--accent);
      color: white;
      font-size: 11px;
      line-height: 1;
      flex: 0 0 auto;
    }

    .status-separator {
      color: #9aa6a0;
    }

    .status-muted {
      color: var(--muted);
    }

    .header-actions {
      display: flex;
      gap: 7px;
      align-items: center;
      justify-content: flex-end;
    }

    .icon-btn {
      width: 34px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--slate);
      display: inline-grid;
      place-items: center;
      cursor: pointer;
      box-shadow: var(--shadow-soft);
      transition: color 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }

    .icon-btn:hover,
    .icon-btn:focus {
      color: var(--accent-strong);
      border-color: rgba(15, 118, 110, 0.42);
      box-shadow: 0 5px 14px rgba(15, 118, 110, 0.12);
      outline: none;
      transform: translateY(-1px);
    }

    .icon-btn svg {
      width: 17px;
      height: 17px;
    }

    .icon-btn:disabled {
      cursor: default;
      opacity: 0.48;
      transform: none;
      box-shadow: var(--shadow-soft);
    }

    .report-stats {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      min-width: 0;
    }

    .report-stats[hidden] {
      display: none;
    }

    .stat-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 8px 10px;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      min-width: 0;
      box-shadow: var(--shadow-soft);
    }

    .stat-card svg {
      width: 16px;
      height: 16px;
      color: var(--accent);
    }

    .stat-card strong,
    .stat-card span {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .stat-card strong {
      color: var(--ink);
      font-size: 13px;
      line-height: 1.2;
      font-weight: 760;
    }

    .stat-card span {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      font-weight: 620;
    }

    body.progress-modal-open {
      overflow: hidden;
    }

    .progress-backdrop {
      position: fixed;
      inset: 0;
      z-index: 1200;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(20, 35, 31, 0.46);
      backdrop-filter: blur(4px);
    }

    .progress-backdrop[hidden] {
      display: none;
    }

    .progress-dialog {
      width: min(780px, 100%);
      max-height: calc(100vh - 48px);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      overflow: hidden;
      border: 1px solid var(--line-strong);
      border-radius: 16px;
      background: var(--panel);
      box-shadow: 0 24px 70px rgba(16, 34, 29, 0.28);
    }

    .progress-dialog-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 19px 22px 15px;
      border-bottom: 1px solid var(--line);
    }

    .progress-dialog-header h2 {
      margin: 3px 0 0;
      font-size: 20px;
      line-height: 1.25;
    }

    .progress-dialog-kicker {
      color: var(--accent-strong);
      font-size: 11px;
      font-weight: 760;
      letter-spacing: .06em;
      text-transform: uppercase;
    }

    .progress-panel {
      padding: 20px 22px;
      display: grid;
      gap: 12px;
      overflow: auto;
      background: var(--panel-soft);
    }

    .progress-dialog-footer {
      display: flex;
      justify-content: flex-end;
      padding: 14px 22px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    .progress-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.35;
      font-weight: 650;
    }

    .progress-detail {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .progress-track {
      height: 8px;
      border-radius: 999px;
      background: #dce4df;
      overflow: hidden;
    }

    .progress-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      transition: width 240ms ease;
    }

    .progress-dialog.failed .progress-fill {
      background: var(--red);
    }

    .progress-log {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .progress-event {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
    }

    .progress-event strong {
      color: var(--accent-strong);
      font-weight: 720;
      overflow-wrap: anywhere;
    }

    @media (max-width: 640px) {
      .progress-backdrop { padding: 0; }
      .progress-dialog { width: 100%; max-height: 100vh; border-radius: 0; }
      .progress-event { grid-template-columns: 1fr; gap: 2px; }
    }

    .tabbar {
      padding: 9px 28px 0;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 6px;
      overflow-x: auto;
      scrollbar-width: none;
      background: rgba(255, 255, 255, 0.84);
      position: sticky;
      top: 172px;
      z-index: 4;
    }

    .tabbar::-webkit-scrollbar {
      display: none;
    }

    .tab {
      border: 1px solid transparent;
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      background: transparent;
      color: var(--muted);
      padding: 10px 13px 11px;
      cursor: pointer;
      font-weight: 650;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
    }

    .tab svg {
      width: 17px;
      height: 17px;
      flex: 0 0 auto;
    }

    .tab.active {
      color: var(--ink);
      border-color: var(--line);
      background: #eef2ef;
      box-shadow: inset 0 3px 0 var(--accent);
    }

    .content {
      min-height: 0;
      overflow: auto;
      padding: 22px 28px 38px;
    }

    .section-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }

    .section-block {
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .section-heading {
      margin: 0;
      font-size: 12px;
      line-height: 1.25;
      text-transform: uppercase;
      color: var(--slate);
      font-weight: 800;
      letter-spacing: 0.06em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .section-heading svg {
      width: 16px;
      height: 16px;
      color: var(--accent);
      flex: 0 0 auto;
    }

    .section-count {
      margin-left: auto;
      color: var(--muted);
      font-size: 11px;
      font-weight: 720;
      letter-spacing: 0;
      text-transform: none;
    }

    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 2px 8px rgba(17, 24, 39, 0.05);
      padding: 16px;
      display: grid;
      gap: 8px;
      min-width: 0;
      transition: border-color 140ms ease, box-shadow 140ms ease;
    }

    .item:hover {
      border-color: var(--line-strong);
      box-shadow: var(--shadow);
    }

    .item h3 {
      margin: 0;
      font-size: 15px;
      line-height: 1.3;
      color: #111827;
      font-weight: 760;
      overflow-wrap: anywhere;
    }

    .item-title-row {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }

    .item-title-row svg {
      width: 17px;
      height: 17px;
      color: var(--accent);
      flex: 0 0 auto;
    }

    .item-title-row h3 {
      min-width: 0;
    }

    .item p {
      margin: 0;
      color: #26332f;
      line-height: 1.48;
      overflow-wrap: anywhere;
    }

    .item-footer {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      padding-top: 2px;
    }

    .evidence-list,
    .transcript-list {
      display: grid;
      gap: 10px;
      max-width: 1080px;
    }

    .transcript-focus {
      max-width: 1080px;
      margin-bottom: 12px;
      border: 1px solid rgba(15, 118, 110, 0.28);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .transcript-focus-title {
      display: grid;
      gap: 3px;
      min-width: 0;
    }

    .transcript-focus-title strong,
    .transcript-focus-title span {
      overflow-wrap: anywhere;
    }

    .transcript-row {
      scroll-margin-top: 120px;
      transition: background 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    .transcript-row.evidence-hit {
      border-color: rgba(15, 118, 110, 0.58);
      background: #e8f4ef;
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }

    .transcript-row:focus {
      outline: 3px solid rgba(15, 118, 110, 0.24);
      outline-offset: 2px;
    }

    .quote {
      color: #2f3a37;
      border-left: 3px solid var(--accent);
      padding-left: 10px;
      background: #f8faf8;
      border-radius: 0 8px 8px 0;
      padding-top: 8px;
      padding-bottom: 8px;
    }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 20px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.72);
    }

    @media (max-width: 880px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        min-height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .sessions {
        max-height: 260px;
      }

      main {
        min-height: 720px;
      }

      .section-grid {
        grid-template-columns: 1fr;
      }

      .title-line {
        align-items: flex-start;
        flex-direction: column;
      }

      .report-meta-row {
        grid-template-columns: 1fr;
      }

      .header-actions {
        justify-content: flex-start;
      }

      .report-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    .template-panel {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #f7fbf9, #eef6f3);
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .template-panel-title,
    .builder-heading-row,
    .builder-section-heading,
    .builder-field-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .template-panel-title strong { font-size: 13px; }
    .template-panel select { width: 100%; }
    .template-summary { color: var(--muted); font-size: 12px; line-height: 1.45; min-height: 34px; }
    .template-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; }
    .template-actions .btn { min-height: 30px; padding: 7px 9px; font-size: 12px; flex: 1 1 auto; }
    .template-origin { display: flex; flex-wrap: wrap; gap: 6px; }

    .builder-backdrop {
      position: fixed;
      inset: 0;
      z-index: 1000;
      display: grid;
      place-items: center;
      padding: 22px;
      background: rgba(15, 23, 42, 0.58);
      backdrop-filter: blur(5px);
    }

    .builder-backdrop[hidden] { display: none; }

    .builder-dialog {
      width: min(1180px, 100%);
      max-height: calc(100vh - 44px);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: var(--panel);
      border: 1px solid var(--line-strong);
      border-radius: 18px;
      box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
    }

    .builder-header,
    .builder-footer { padding: 16px 20px; border-bottom: 1px solid var(--line); }
    .builder-footer { border-top: 1px solid var(--line); border-bottom: 0; display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .builder-header h2 { margin: 0; font-size: 20px; }
    .builder-header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
    .builder-body { padding: 18px 20px 28px; overflow: auto; display: grid; gap: 18px; background: var(--panel-soft); }
    .builder-card { border: 1px solid var(--line); background: var(--panel); border-radius: 13px; padding: 15px; box-shadow: var(--shadow-soft); }
    .builder-meta-grid { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 12px; }
    .builder-meta-grid .wide { grid-column: 1 / -1; }
    .builder-card label { display: grid; gap: 6px; color: var(--slate); font-size: 12px; font-weight: 650; }
    .builder-card input,
    .builder-card select,
    .builder-card textarea,
    .template-panel select {
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      min-width: 0;
    }
    .builder-card textarea { min-height: 76px; resize: vertical; line-height: 1.4; }
    .builder-section-list { display: grid; gap: 14px; }
    .builder-section-card { border-left: 4px solid var(--accent); }
    .builder-section-heading h3 { margin: 0; font-size: 16px; }
    .builder-section-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 11px; margin-top: 13px; }
    .builder-section-grid .span-2 { grid-column: span 2; }
    .builder-section-grid .wide { grid-column: 1 / -1; }
    .builder-inline-actions { display: flex; gap: 6px; flex-wrap: wrap; }
    .builder-inline-actions .btn { padding: 6px 8px; font-size: 11px; }
    .builder-fields { grid-column: 1 / -1; border-top: 1px solid var(--line); padding-top: 12px; display: grid; gap: 9px; }
    .builder-field-row { display: grid; grid-template-columns: 0.75fr 0.9fr 0.7fr 1.25fr 1fr auto; gap: 8px; align-items: end; }
    .builder-field-row .btn { margin-bottom: 1px; }
    .builder-readonly-note { border: 1px solid #99c8b9; background: var(--accent-quiet); color: var(--accent-strong); border-radius: 9px; padding: 10px 12px; font-size: 13px; }
    .builder-validation { color: var(--red); font-size: 12px; min-height: 18px; }
    .builder-footer-actions { display: flex; gap: 8px; }

    .attribute-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .attribute-chip { border-radius: 7px; padding: 5px 7px; background: var(--panel-soft); border: 1px solid var(--line); font-size: 12px; }
    .grounding-warning { color: var(--red); background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 7px 9px; font-size: 12px; }
    .report-table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 11px; background: #fff; }
    .report-table { width: 100%; border-collapse: collapse; min-width: 680px; }
    .report-table th,
    .report-table td { padding: 10px 11px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: start; font-size: 13px; }
    .report-table th { color: var(--muted); background: var(--panel-soft); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .report-table tr:last-child td { border-bottom: 0; }
    .timeline-list { display: grid; gap: 0; padding-left: 17px; border-left: 2px solid #a7d3c7; }
    .timeline-list .item { position: relative; margin: 0 0 12px 12px; }
    .timeline-list .item::before { content: ""; position: absolute; left: -21px; top: 17px; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); border: 2px solid #fff; box-shadow: 0 0 0 1px var(--accent); }
    .quote-layout { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
    .quote-layout .item p { font-size: 16px; line-height: 1.5; font-style: italic; }

    @media (max-width: 900px) {
      .builder-meta-grid,
      .builder-section-grid,
      .builder-field-row { grid-template-columns: 1fr; }
      .builder-section-grid .span-2 { grid-column: auto; }
      .builder-backdrop { padding: 0; }
      .builder-dialog { width: 100%; max-height: 100vh; border-radius: 0; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">
          <div class="logo" aria-hidden="true">WS</div>
          <h1>WhoSpeaks Meeting Intelligence</h1>
        </div>
        <div class="provider" id="provider"></div>
        <div class="provider-controls" id="providerControls">
          <div class="provider-control-grid">
            <label class="field">
              Provider
              <select id="llmProviderSelect"></select>
            </label>
            <label class="field">
              Model
              <select id="llmModelSelect"></select>
              <input id="llmModelInput" autocomplete="off" placeholder="Custom model id" hidden>
            </label>
          </div>
          <label class="field">
            Base URL
            <input id="llmBaseUrlInput" autocomplete="off">
          </label>
          <div class="provider-action-row">
            <button class="btn" id="applyProviderBtn" type="button">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M20 6 9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              <span>Apply</span>
            </button>
            <button class="btn" id="loadModelsBtn" type="button" title="Load available models">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M4 6h16M4 12h16M4 18h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
              <span>Models</span>
            </button>
            <div class="provider-hint" id="providerHint"></div>
          </div>
        </div>
      </div>
      <section class="template-panel" aria-label="Report template">
        <div class="template-panel-title">
          <strong>Report template</strong>
          <span class="badge hot">Generator</span>
        </div>
        <select id="templateSelect" aria-label="Choose report template"></select>
        <div class="template-origin" id="templateBadges"></div>
        <div class="template-summary" id="templateSummary" dir="auto">Loading report templates...</div>
        <div class="template-actions">
          <button class="btn primary" id="newTemplateBtn" type="button">New</button>
          <button class="btn" id="inspectTemplateBtn" type="button">Inspect</button>
          <button class="btn" id="cloneTemplateBtn" type="button">Clone</button>
          <button class="btn danger" id="deleteTemplateBtn" type="button">Delete</button>
        </div>
      </section>
      <div class="toolbar">
        <input class="search" id="sessionSearch" type="search" placeholder="Filter sessions">
        <div class="button-row">
          <button class="btn primary" id="generateBtn" type="button">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 3v18M5 10l7-7 7 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Generate report</span>
          </button>
          <button class="btn danger" id="deleteReportBtn" type="button" title="Delete cached report" aria-label="Delete cached report">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M3 6h18M8 6V4h8v2M6 6l1 15h10l1-15M10 11v6M14 11v6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span id="deleteReportLabel">Delete</span>
          </button>
          <button class="btn" id="refreshBtn" type="button" title="Refresh sessions">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v6h-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="sessions" id="sessions"></div>
    </aside>
    <main>
      <header class="topbar">
        <div class="title-line">
          <h2 id="reportTitle" dir="auto">Select a session</h2>
          <div class="badge-row" id="reportBadges"></div>
        </div>
        <div class="report-meta-row">
          <div class="status" id="status"></div>
          <div class="header-actions" aria-label="Report shortcuts">
            <button class="icon-btn" id="headerGenerateBtn" type="button" title="Regenerate report" aria-label="Regenerate report">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3ZM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
              </svg>
            </button>
            <button class="icon-btn" id="headerTranscriptBtn" type="button" title="Open transcript" aria-label="Open transcript">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M4 10v4M8 7v10M12 5v14M16 8v8M20 11v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
            </button>
            <button class="icon-btn" id="headerRefreshBtn" type="button" title="Refresh report" aria-label="Refresh report">
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v6h-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="report-stats" id="reportStats" hidden></div>
      </header>
      <nav class="tabbar" id="tabs"></nav>
      <section class="content" id="content"></section>
    </main>
  </div>
  <div class="progress-backdrop" id="progressOverlay" hidden>
    <section class="progress-dialog" id="progressDialog" role="dialog" aria-modal="true" aria-labelledby="progressDialogTitle" aria-describedby="progressDetail">
      <header class="progress-dialog-header">
        <div>
          <div class="progress-dialog-kicker">Report generator</div>
          <h2 id="progressDialogTitle">Generating report</h2>
        </div>
        <button class="icon-btn" id="closeProgressBtn" type="button" aria-label="Close generation status" disabled>&times;</button>
      </header>
      <div class="progress-panel" id="progressPanel" aria-live="polite">
          <div class="progress-line">
            <span id="progressLabel">Queued</span>
            <span id="progressPercent">0%</span>
          </div>
          <div class="progress-track" aria-hidden="true">
            <div class="progress-fill" id="progressFill"></div>
          </div>
          <div class="progress-detail" id="progressDetail"></div>
          <div class="progress-log" id="progressLog"></div>
      </div>
      <footer class="progress-dialog-footer">
        <button class="btn primary" id="closeProgressFooterBtn" type="button" disabled>Generating...</button>
      </footer>
    </section>
  </div>
  <div class="builder-backdrop" id="templateBuilder" hidden>
    <section class="builder-dialog" role="dialog" aria-modal="true" aria-labelledby="builderTitle">
      <header class="builder-header">
        <div class="builder-heading-row">
          <div>
            <h2 id="builderTitle">Report builder</h2>
            <p id="builderSubtitle">Create a reusable report from flat, evidence-grounded sections.</p>
          </div>
          <button class="icon-btn" id="closeBuilderBtn" type="button" aria-label="Close report builder">×</button>
        </div>
      </header>
      <div class="builder-body" id="builderBody"></div>
      <footer class="builder-footer">
        <div class="builder-validation" id="builderValidation"></div>
        <div class="builder-footer-actions">
          <button class="btn" id="cancelBuilderBtn" type="button">Cancel</button>
          <button class="btn primary" id="saveTemplateBtn" type="button">Save report template</button>
        </div>
      </footer>
    </section>
  </div>
  <script>
    const state = {
      config: {},
      sessions: [],
      templates: [],
      templateId: "",
      sessionId: "",
      report: null,
      reportAvailable: false,
      transcriptRows: [],
      activeTab: "",
      generating: false,
      generationJob: null,
      activeEvidenceId: "",
      highlightRowIds: [],
      evidenceReturnTab: "",
      confirmDelete: false,
      providerDraft: {provider: "", model: "", base_url: ""},
      providerModels: {},
      loadingModels: false,
      applyingProvider: false,
      status: "",
      builderTemplate: null,
      builderReadOnly: false,
      builderMode: "new"
    };

    const el = (id) => document.getElementById(id);

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {})
        }
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function domId(value) {
      return String(value ?? "").replace(/[^A-Za-z0-9_-]/g, "_") || "row";
    }

    function transcriptRowDomId(rowId) {
      return `transcript-row-${domId(rowId)}`;
    }

    function uniqueValues(values) {
      const result = [];
      const seen = new Set();
      values.forEach((value) => {
        const text = String(value ?? "").trim();
        if (!text || seen.has(text)) return;
        seen.add(text);
        result.push(text);
      });
      return result;
    }

    function normalizedFallbackRowId(row, index) {
      const rawIndex = Number(row?.index);
      const normalizedIndex = Number.isFinite(rawIndex) && rawIndex !== 0
        ? Math.trunc(rawIndex)
        : index + 1;
      return `row_${normalizedIndex}`;
    }

    function transcriptRowAliases(row, index) {
      return uniqueValues([
        row?.row_id,
        row?.id,
        normalizedFallbackRowId(row, index),
        `row-${index + 1}`,
      ]);
    }

    function transcriptRowElementId(row, index) {
      const primary = transcriptRowAliases(row, index)[0] || `row_${index + 1}`;
      return `${transcriptRowDomId(primary)}-${index + 1}`;
    }

    function encodedRowAliases(row, index) {
      return transcriptRowAliases(row, index).map((value) => encodeURIComponent(value)).join(" ");
    }

    function evidenceRowIds(item) {
      return Array.isArray(item?.row_ids) ? item.row_ids.map((value) => String(value)) : [];
    }

    function evidenceRowLabel(item) {
      const count = evidenceRowIds(item).length;
      if (!count) return "Transcript";
      return `${count} transcript row${count === 1 ? "" : "s"}`;
    }

    function clearEvidenceFocus() {
      state.activeEvidenceId = "";
      state.highlightRowIds = [];
      state.evidenceReturnTab = "";
    }

    function clearDeleteConfirm() {
      state.confirmDelete = false;
    }

    function setStatus(value) {
      state.status = value || "";
      el("status").innerHTML = reportStatusHtml();
    }

    function formatDateTime(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return "";
      return date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      });
    }

    function reportStatusHtml() {
      const session = state.sessions.find((item) => item.id === state.sessionId) || {};
      const report = state.report || {};
      const status = state.status || (state.sessionId ? "Ready" : "Select a session");
      const parts = [
        `<span class="status-state"><span class="status-dot" aria-hidden="true">&#10003;</span>${escapeHtml(status)}</span>`
      ];
      const generated = formatDateTime(report.generated_at || report.updated_at);
      if (generated) {
        parts.push(`<span class="status-separator">/</span><span class="status-muted">Generated ${escapeHtml(generated)}</span>`);
      }
      const rows = Number(state.transcriptRows?.length || session.transcript_rows || 0);
      const speakers = Number(session.speaker_count || 0);
      if (rows || speakers) {
        const label = [
          rows ? `${rows} transcript rows` : "",
          speakers ? `${speakers} speakers` : ""
        ].filter(Boolean).join(" / ");
        parts.push(`<span class="status-separator">/</span><span class="status-muted">${escapeHtml(label)}</span>`);
      }
      return parts.join("");
    }

    function providerOptions() {
      return Array.isArray(state.config.providers) ? state.config.providers : [];
    }

    function providerOption(provider) {
      return providerOptions().find((option) => option.id === provider) || providerOptions()[0] || {};
    }

    function syncProviderDraftFromConfig() {
      const cfg = state.config || {};
      state.providerDraft = {
        provider: cfg.provider || providerOptions()[0]?.id || "llama_cpp",
        model: cfg.model || "",
        base_url: cfg.base_url || ""
      };
    }

    function selectedProviderOption() {
      return providerOption(state.providerDraft.provider || state.config.provider);
    }

    function providerModelOptions(option) {
      const provider = option.id || state.providerDraft.provider || state.config.provider || "";
      return uniqueValues([
        ...(state.providerModels[provider] || []),
        ...(option.models || []),
        state.providerDraft.model,
      ]);
    }

    function renderProviderControls() {
      const options = providerOptions();
      const providerSelect = el("llmProviderSelect");
      providerSelect.innerHTML = options.map((option) => `
        <option value="${escapeHtml(option.id)}">${escapeHtml(option.label || option.id)}</option>
      `).join("");
      providerSelect.value = state.providerDraft.provider || state.config.provider || options[0]?.id || "";
      const option = selectedProviderOption();
      const models = providerModelOptions(option);
      const modelSelect = el("llmModelSelect");
      const modelIsPreset = models.includes(state.providerDraft.model);
      modelSelect.innerHTML = models.map((model) => `
        <option value="${escapeHtml(model)}">${escapeHtml(model)}</option>
      `).join("") + '<option value="__custom__">Custom model...</option>';
      modelSelect.value = modelIsPreset ? state.providerDraft.model : "__custom__";
      const modelInput = el("llmModelInput");
      modelInput.hidden = modelIsPreset;
      modelInput.value = modelIsPreset ? "" : (state.providerDraft.model || "");
      el("llmBaseUrlInput").value = state.providerDraft.base_url || option.default_base_url || "";
      const requiresKey = Boolean(option.requires_api_key);
      const keyOk = !requiresKey || Boolean(state.config.api_key_configured);
      const modelCount = state.providerModels[option.id]?.length || 0;
      const keyText = state.loadingModels
        ? "Loading models..."
        : requiresKey
        ? `${escapeHtml(option.api_key_env_var || "API key")}: ${keyOk ? "configured" : "missing"}`
        : "No API key required";
      el("providerHint").innerHTML = `${keyText}${modelCount ? ` / ${modelCount} loaded` : ""}`;
      el("loadModelsBtn").disabled = state.loadingModels || state.generating || state.applyingProvider;
      el("applyProviderBtn").disabled = state.generating || state.applyingProvider || !state.providerDraft.model.trim();
    }

    async function loadProviderModels() {
      if (state.loadingModels) return;
      const draft = state.providerDraft;
      const previousModel = draft.model;
      const previousDefault = selectedProviderOption().models?.[0] || "";
      state.loadingModels = true;
      setStatus("Loading models");
      render();
      try {
        const query = new URLSearchParams({
          provider: draft.provider,
          base_url: draft.base_url || selectedProviderOption().default_base_url || ""
        });
        const data = await api(`/api/llm-models?${query.toString()}`);
        state.providerModels[data.provider] = data.models || [];
        if (
          state.providerModels[data.provider]?.length
          && (!previousModel || previousModel === previousDefault)
        ) {
          state.providerDraft.model = state.providerModels[data.provider][0];
        }
        setStatus(state.providerModels[data.provider]?.length ? "Models loaded" : "No models returned");
      } catch (error) {
        setStatus(error.message);
      } finally {
        state.loadingModels = false;
        render();
      }
    }

    function providerLabel() {
      const cfg = state.config || {};
      const mode = cfg.mock_llm ? "mock" : cfg.provider;
      const option = providerOption(cfg.provider);
      const requiresKey = Boolean(option.requires_api_key);
      const keyOk = !requiresKey || Boolean(cfg.api_key_configured);
      const readyText = cfg.mock_llm ? "Mock" : keyOk ? "Ready" : "Missing key";
      const warnClass = keyOk ? "" : " warn";
      return `
        <div class="provider-kicker">Provider / Model</div>
        <div class="provider-connected${warnClass}"><span class="provider-health${warnClass}" aria-hidden="true"></span>${escapeHtml(readyText)}</div>
        <div class="provider-model"><strong>${escapeHtml(mode)}</strong> / ${escapeHtml(cfg.model || "local")}</div>
        <div class="provider-url">${escapeHtml(cfg.base_url || "no base URL")}</div>
      `;
    }

    function sessionIcon(kind) {
      if (kind === "demo_transcript") {
        return '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3h7l4 4v14H7V3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 3v5h5M10 12h6M10 16h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
      }
      return '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 5h10a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3Z" stroke="currentColor" stroke-width="2"/><path d="m10 9 5 3-5 3V9Z" fill="currentColor"/></svg>';
    }

    function tabIcon(name) {
      const icons = {
        list: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        check: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 12l2 2 4-5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        tasks: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 6h11M9 12h11M9 18h11M4 6l1 1 2-2M4 12l1 1 2-2M4 18l1 1 2-2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        question: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9.1 9a3 3 0 1 1 4.8 2.4c-.9.6-1.4 1.1-1.4 2.1M12 17h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        shield: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3 20 6v5c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10V6l8-3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8v5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 16h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        file: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3h7l4 4v14H7V3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 3v5h5M10 13h6M10 17h4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        link: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.1.1l2-2a5 5 0 0 0-7.1-7.1l-1.1 1.1M14 11a5 5 0 0 0-7.1-.1l-2 2A5 5 0 0 0 12 20l1.1-1.1" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        wave: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 10v4M8 7v10M12 5v14M16 8v8M20 11v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
      };
      return icons[name] || "";
    }

    function sectionIcon(name) {
      const groups = {
        executive_summary: "spark",
        structured_brief: "file",
        speaker_map: "users",
        speaker_participation: "wave",
        discussion_threads: "list",
        decisions: "check",
        deadlines: "calendar",
        disagreements: "alert",
        action_items: "tasks",
        open_questions: "question",
        ask_this_meeting: "question",
        risks: "shield"
      };
      const icons = {
        alert: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 4.4 2.7 18a2 2 0 0 0 1.7 3h15.2a2 2 0 0 0 1.7-3L13.7 4.4a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        calendar: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3v4M17 3v4M4 9h16M6 5h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
        spark: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3ZM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
        users: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2M9.5 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
      };
      return icons[groups[name]] || tabIcon(groups[name] || name) || tabIcon("file");
    }

    function sectionLabel(name) {
      return name.replaceAll("_", " ");
    }

    function pluralLabel(count, singular, plural = `${singular}s`) {
      return `${count} ${count === 1 ? singular : plural}`;
    }

    function deepCopy(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function currentTemplate() {
      return state.templates.find((item) => item.template_id === state.templateId) || null;
    }

    function effectiveReportTemplate() {
      return state.report?.report_template || currentTemplate() || {sections: []};
    }

    function reportSectionDefinitions() {
      const definitions = effectiveReportTemplate()?.sections;
      return Array.isArray(definitions) ? definitions : [];
    }

    function sectionTabId(key) {
      return `section:${key}`;
    }

    function sectionIconName(definition) {
      const kind = String(definition?.render_kind || "cards");
      return kind === "timeline" ? "list" : kind === "quotes" ? "question" : kind === "table" ? "tasks" : "file";
    }

    function reportTabs() {
      const sectionTabs = reportSectionDefinitions().map((definition) => [
        sectionTabId(definition.key),
        definition.title || sectionLabel(definition.key),
        sectionIconName(definition),
      ]);
      return [...sectionTabs, ["evidence", "Evidence", "link"], ["transcript", "Transcript", "wave"]];
    }

    function ensureActiveTab() {
      const available = reportTabs();
      if (!available.some(([id]) => id === state.activeTab)) {
        state.activeTab = available[0]?.[0] || "transcript";
      }
    }

    function normalizeBuilderKey(value, fallback = "section") {
      const key = String(value || "").toLowerCase().normalize("NFKD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 64);
      return key || fallback;
    }

    function defaultOutputField(index = 1) {
      return {key: `field_${index}`, label: `Field ${index}`, type: "text", description: "", options: []};
    }

    function defaultSection(index = 1) {
      return {
        key: `section_${index}`,
        title: `Section ${index}`,
        objective: "Describe exactly what this section should find in the final speaker-labelled transcript.",
        max_items: 6,
        evidence_required: true,
        render_kind: "cards",
        sort_order: "relevance",
        output_fields: [],
      };
    }

    function newTemplateDraft() {
      return {
        schema_version: "report_template_v1",
        name: "Untitled custom report",
        description: "A reusable evidence-grounded report.",
        version: 1,
        builtin: false,
        language_mode: "inherit",
        privacy_policy: "inherit",
        sections: [defaultSection(1)],
      };
    }

    function cleanBuilderTemplate(template) {
      const clean = deepCopy(template || {});
      delete clean.read_only;
      delete clean.source_kind;
      delete clean.revision_hash;
      clean.builtin = false;
      clean.version = Number(clean.version || 1);
      clean.sections = (clean.sections || []).map((section) => ({
        key: normalizeBuilderKey(section.key, "section"),
        title: String(section.title || "").trim(),
        objective: String(section.objective || "").trim(),
        max_items: Math.max(1, Math.min(20, Number(section.max_items || 6))),
        evidence_required: Boolean(section.evidence_required),
        render_kind: section.render_kind || "cards",
        sort_order: section.sort_order || "relevance",
        output_fields: (section.output_fields || []).map((field) => ({
          key: normalizeBuilderKey(field.key, "field"),
          label: String(field.label || "").trim(),
          type: field.type || "text",
          description: String(field.description || "").trim(),
          options: Array.isArray(field.options) ? field.options.map((item) => String(item).trim()).filter(Boolean) : [],
        })),
      }));
      return clean;
    }

    function renderTemplateControls() {
      const select = el("templateSelect");
      const predefined = state.templates.filter((item) => item.builtin);
      const custom = state.templates.filter((item) => !item.builtin);
      const optionHtml = (item) => `<option value="${escapeHtml(item.template_id)}" ${item.template_id === state.templateId ? "selected" : ""}>${escapeHtml(item.name)}</option>`;
      select.innerHTML = [
        `<optgroup label="Predefined reports">${predefined.map(optionHtml).join("")}</optgroup>`,
        custom.length ? `<optgroup label="Custom reports">${custom.map(optionHtml).join("")}</optgroup>` : "",
      ].join("");
      const template = currentTemplate();
      el("templateSummary").textContent = template?.description || "Choose or create a report template.";
      el("templateBadges").innerHTML = template ? [
        `<span class="badge ${template.builtin ? "hot" : ""}">${template.builtin ? "Predefined" : "Custom"}</span>`,
        `<span class="badge">${escapeHtml(template.language_mode === "inherit" ? "Follow server language" : template.language_mode)}</span>`,
        template.privacy_policy === "local_only" ? '<span class="badge warn">Local only</span>' : "",
        `<span class="badge">${template.sections?.length || 0} sections</span>`,
      ].join("") : "";
      el("inspectTemplateBtn").disabled = !template;
      el("cloneTemplateBtn").disabled = !template;
      el("deleteTemplateBtn").disabled = !template || Boolean(template.builtin);
      el("inspectTemplateBtn").textContent = template?.builtin ? "Inspect" : "Edit";
    }

    function builderValidationMessage(template) {
      if (!String(template?.name || "").trim()) return "Give the report a name.";
      if (!Array.isArray(template?.sections) || !template.sections.length) return "Add at least one section.";
      if (template.sections.length > 16) return "A report can contain at most 16 sections.";
      const keys = new Set();
      for (const section of template.sections) {
        const key = normalizeBuilderKey(section.key, "");
        if (!key) return "Every section needs a key.";
        if (keys.has(key)) return `Section key '${key}' is duplicated.`;
        keys.add(key);
        if (!String(section.title || "").trim()) return `Section '${key}' needs a title.`;
        if (!String(section.objective || "").trim()) return `Section '${key}' needs an objective.`;
        const fieldKeys = new Set();
        for (const field of section.output_fields || []) {
          const fieldKey = normalizeBuilderKey(field.key, "");
          if (!fieldKey || !String(field.label || "").trim()) return `Every output field in '${key}' needs a key and label.`;
          if (fieldKeys.has(fieldKey)) return `Output field '${fieldKey}' is duplicated in '${key}'.`;
          fieldKeys.add(fieldKey);
          if (field.type === "enum" && !(field.options || []).length) return `Enum field '${fieldKey}' needs at least one option.`;
        }
      }
      return "";
    }

    function openTemplateBuilder(mode = "inspect") {
      const source = currentTemplate();
      state.builderMode = mode;
      if (mode === "new") {
        state.builderTemplate = newTemplateDraft();
        state.builderReadOnly = false;
      } else if (mode === "clone") {
        state.builderTemplate = cleanBuilderTemplate(source || newTemplateDraft());
        delete state.builderTemplate.template_id;
        state.builderTemplate.name = `${source?.name || "Report"} copy`;
        state.builderTemplate.version = 1;
        state.builderReadOnly = false;
      } else {
        state.builderTemplate = deepCopy(source || newTemplateDraft());
        state.builderReadOnly = Boolean(source?.builtin);
      }
      el("templateBuilder").hidden = false;
      document.body.style.overflow = "hidden";
      renderTemplateBuilder();
    }

    function closeTemplateBuilder() {
      el("templateBuilder").hidden = true;
      document.body.style.overflow = "";
      state.builderTemplate = null;
      el("builderValidation").textContent = "";
    }

    function renderTemplateBuilder() {
      const template = state.builderTemplate;
      if (!template) return;
      const disabled = state.builderReadOnly ? "disabled" : "";
      el("builderTitle").textContent = state.builderReadOnly ? "Inspect predefined report" : state.builderMode === "new" ? "Create custom report" : state.builderMode === "clone" ? "Clone report" : "Edit custom report";
      el("builderSubtitle").textContent = state.builderReadOnly
        ? "This is an ordinary report-template document. Clone it to make an editable copy."
        : "Configure flat top-level sections, their output fields, layout, ordering, and evidence rules.";
      const languageCodes = "inherit,af,ar,be,bg,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,he,hi,hr,hu,hy,id,is,it,ja,ka,kk,ko,la,lt,lv,ml,mr,mt,my,nl,nn,no,pl,pt,ro,ru,sa,sd,sk,sl,sq,sr,sv,ta,te,th,tr,uk,ur,vi,zh".split(",");
      const sections = (template.sections || []).map((section, sectionIndex) => {
        const fields = (section.output_fields || []).map((field, fieldIndex) => `
          <div class="builder-field-row" data-field-row="${fieldIndex}">
            <label>Key<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="key" value="${escapeHtml(field.key || "")}"></label>
            <label>Label<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="label" value="${escapeHtml(field.label || "")}"></label>
            <label>Type<select ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="type">${["text","enum","speaker","date","timestamp","boolean","number"].map((type) => `<option ${field.type === type ? "selected" : ""}>${type}</option>`).join("")}</select></label>
            <label>Description<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="description" value="${escapeHtml(field.description || "")}"></label>
            <label>Enum options<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="options" value="${escapeHtml((field.options || []).join(", "))}" placeholder="open, closed"></label>
            <button class="btn danger" type="button" ${disabled} data-field-action="remove" data-section-index="${sectionIndex}" data-field-index="${fieldIndex}">Remove</button>
          </div>
        `).join("");
        return `
          <article class="builder-card builder-section-card" data-section-card="${sectionIndex}">
            <div class="builder-section-heading">
              <h3 dir="auto">${escapeHtml(section.title || `Section ${sectionIndex + 1}`)}</h3>
              <div class="builder-inline-actions">
                <button class="btn" type="button" ${disabled} data-section-action="up" data-section-index="${sectionIndex}">↑ Up</button>
                <button class="btn" type="button" ${disabled} data-section-action="down" data-section-index="${sectionIndex}">↓ Down</button>
                <button class="btn danger" type="button" ${disabled} data-section-action="remove" data-section-index="${sectionIndex}">Remove</button>
              </div>
            </div>
            <div class="builder-section-grid">
              <label class="span-2">Title<input ${disabled} data-section-index="${sectionIndex}" data-section-prop="title" value="${escapeHtml(section.title || "")}"></label>
              <label class="span-2">Key<input ${disabled} data-section-index="${sectionIndex}" data-section-prop="key" value="${escapeHtml(section.key || "")}"></label>
              <label class="wide">What should this section find?<textarea ${disabled} data-section-index="${sectionIndex}" data-section-prop="objective">${escapeHtml(section.objective || "")}</textarea></label>
              <label>Maximum items<input ${disabled} type="number" min="1" max="20" data-section-index="${sectionIndex}" data-section-prop="max_items" value="${Number(section.max_items || 6)}"></label>
              <label>Layout<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="render_kind">${["cards","table","timeline","quotes"].map((kind) => `<option ${section.render_kind === kind ? "selected" : ""}>${kind}</option>`).join("")}</select></label>
              <label>Order<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="sort_order">${["relevance","chronological","severity"].map((kind) => `<option ${section.sort_order === kind ? "selected" : ""}>${kind}</option>`).join("")}</select></label>
              <label>Evidence<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="evidence_required"><option value="true" ${section.evidence_required ? "selected" : ""}>Required</option><option value="false" ${!section.evidence_required ? "selected" : ""}>Optional</option></select></label>
              <div class="builder-fields">
                <div class="builder-field-heading"><strong>Configurable output fields</strong><button class="btn" type="button" ${disabled} data-section-action="add-field" data-section-index="${sectionIndex}">+ Add field</button></div>
                ${fields || '<div class="meta">No custom fields. Items still include a title, body, status, confidence, and evidence tags.</div>'}
              </div>
            </div>
          </article>
        `;
      }).join("");
      el("builderBody").innerHTML = `
        ${state.builderReadOnly ? '<div class="builder-readonly-note">Predefined reports use the same template format and generator as custom reports. Inspect every setting here, then Clone to customize it.</div>' : ""}
        <section class="builder-card">
          <div class="builder-meta-grid">
            <label>Report name<input ${disabled} data-template-prop="name" value="${escapeHtml(template.name || "")}"></label>
            <label>Report language<select ${disabled} data-template-prop="language_mode">${languageCodes.map((code) => `<option value="${code}" ${template.language_mode === code ? "selected" : ""}>${code === "inherit" ? "Follow report server" : code}</option>`).join("")}</select></label>
            <label>Privacy<select ${disabled} data-template-prop="privacy_policy"><option value="inherit" ${template.privacy_policy === "inherit" ? "selected" : ""}>Follow provider settings</option><option value="local_only" ${template.privacy_policy === "local_only" ? "selected" : ""}>Local models only</option></select></label>
            <label class="wide">Description<textarea ${disabled} data-template-prop="description">${escapeHtml(template.description || "")}</textarea></label>
          </div>
        </section>
        <div class="builder-section-list">${sections}</div>
        <button class="btn primary" type="button" ${disabled} id="addSectionBtn">+ Add top-level section</button>
      `;
      el("saveTemplateBtn").hidden = state.builderReadOnly;
      el("builderValidation").textContent = state.builderReadOnly ? "" : builderValidationMessage(template);
      attachBuilderHandlers();
    }

    function attachBuilderHandlers() {
      const template = state.builderTemplate;
      if (!template || state.builderReadOnly) return;
      el("builderBody").querySelectorAll("[data-template-prop]").forEach((node) => {
        node.addEventListener("input", () => { template[node.dataset.templateProp] = node.value; el("builderValidation").textContent = builderValidationMessage(template); });
        node.addEventListener("change", () => { template[node.dataset.templateProp] = node.value; renderTemplateBuilder(); });
      });
      el("builderBody").querySelectorAll("[data-section-prop]").forEach((node) => {
        const update = () => {
          const section = template.sections[Number(node.dataset.sectionIndex)];
          const prop = node.dataset.sectionProp;
          let value = node.value;
          if (prop === "max_items") value = Number(value);
          if (prop === "evidence_required") value = value === "true";
          if (prop === "key") value = normalizeBuilderKey(value, value);
          section[prop] = value;
          el("builderValidation").textContent = builderValidationMessage(template);
        };
        node.addEventListener("input", update);
        node.addEventListener("change", () => { update(); renderTemplateBuilder(); });
      });
      el("builderBody").querySelectorAll("[data-field-prop]").forEach((node) => {
        const update = () => {
          const field = template.sections[Number(node.dataset.sectionIndex)].output_fields[Number(node.dataset.fieldIndex)];
          const prop = node.dataset.fieldProp;
          field[prop] = prop === "options" ? node.value.split(",").map((item) => item.trim()).filter(Boolean) : prop === "key" ? normalizeBuilderKey(node.value, node.value) : node.value;
          el("builderValidation").textContent = builderValidationMessage(template);
        };
        node.addEventListener("input", update);
        node.addEventListener("change", () => { update(); renderTemplateBuilder(); });
      });
      el("builderBody").querySelectorAll("[data-section-action]").forEach((button) => button.addEventListener("click", () => {
        const index = Number(button.dataset.sectionIndex);
        const action = button.dataset.sectionAction;
        if (action === "remove") template.sections.splice(index, 1);
        if (action === "up" && index > 0) [template.sections[index - 1], template.sections[index]] = [template.sections[index], template.sections[index - 1]];
        if (action === "down" && index < template.sections.length - 1) [template.sections[index + 1], template.sections[index]] = [template.sections[index], template.sections[index + 1]];
        if (action === "add-field") template.sections[index].output_fields.push(defaultOutputField(template.sections[index].output_fields.length + 1));
        renderTemplateBuilder();
      }));
      el("builderBody").querySelectorAll("[data-field-action='remove']").forEach((button) => button.addEventListener("click", () => {
        template.sections[Number(button.dataset.sectionIndex)].output_fields.splice(Number(button.dataset.fieldIndex), 1);
        renderTemplateBuilder();
      }));
      el("addSectionBtn")?.addEventListener("click", () => {
        if (template.sections.length >= 16) return;
        template.sections.push(defaultSection(template.sections.length + 1));
        renderTemplateBuilder();
        el("builderBody").lastElementChild?.scrollIntoView({behavior: "smooth", block: "end"});
      });
    }

    async function saveTemplateBuilder() {
      if (!state.builderTemplate || state.builderReadOnly) return;
      const error = builderValidationMessage(state.builderTemplate);
      el("builderValidation").textContent = error;
      if (error) return;
      const payload = cleanBuilderTemplate(state.builderTemplate);
      if (state.builderMode === "new" || state.builderMode === "clone") delete payload.template_id;
      try {
        const data = await api("/api/templates/save", {method: "POST", body: JSON.stringify({template: payload})});
        closeTemplateBuilder();
        await loadTemplates(data.template?.template_id || "");
        setStatus("Report template saved");
        if (state.sessionId) await selectSession(state.sessionId);
      } catch (error) {
        el("builderValidation").textContent = error.message;
      }
    }

    async function deleteSelectedTemplate() {
      const template = currentTemplate();
      if (!template || template.builtin) return;
      if (!confirm(`Delete custom report template '${template.name}'? Cached reports are left untouched.`)) return;
      await api("/api/templates/delete", {method: "POST", body: JSON.stringify({template_id: template.template_id})});
      await loadTemplates(state.config.standard_template_id || "");
      state.report = null;
      state.reportAvailable = false;
      if (state.sessionId) await selectSession(state.sessionId);
      setStatus("Custom report template deleted");
    }

    async function loadTemplates(preferredTemplateId = "") {
      const data = await api("/api/templates");
      state.templates = data.templates || [];
      const candidate = preferredTemplateId || state.templateId || data.standard_template_id || state.config.standard_template_id;
      state.templateId = state.templates.some((item) => item.template_id === candidate)
        ? candidate
        : state.templates[0]?.template_id || "";
      ensureActiveTab();
      renderTemplateControls();
    }

    function renderSessions() {
      const query = el("sessionSearch").value.trim().toLowerCase();
      const filtered = state.sessions.filter((session) => {
        const text = `${session.title || ""} ${(session.speaker_names || []).join(" ")}`.toLowerCase();
        return !query || text.includes(query);
      });
      const heading = `<div class="sessions-heading"><span>Sessions</span><span>${filtered.length}</span></div>`;
      el("sessions").innerHTML = heading + (filtered.length ? filtered.map((session) => {
        const active = session.id === state.sessionId ? " active" : "";
        const rows = Number(session.transcript_rows || 0);
        const speakers = Number(session.speaker_count || 0);
        const source = session.source_kind === "demo_transcript" ? "Demo" : "Saved";
        const cached = Array.isArray(session.report_template_ids)
          ? session.report_template_ids.includes(state.templateId)
          : session.has_cached_report || session.has_meeting_intelligence;
        const sessionTime = formatDateTime(session.updated_at || session.created_at);
        return `
          <button class="session${active}" type="button" data-session-id="${escapeHtml(session.id)}">
            <span class="session-icon">${sessionIcon(session.source_kind)}</span>
            <span class="session-main">
              <span class="session-title-row">
                <span class="session-title">${escapeHtml(session.title || session.id)}</span>
                ${sessionTime ? `<span class="session-time">${escapeHtml(sessionTime)}</span>` : ""}
              </span>
              <span class="meta">${rows} rows / ${speakers} speakers</span>
              <span class="badge-row">
                <span class="badge">${escapeHtml(source)}</span>
                ${cached ? '<span class="badge hot">Report</span>' : '<span class="badge warn">No report</span>'}
              </span>
            </span>
          </button>
        `;
      }).join("") : '<div class="empty">No sessions found.</div>');
      document.querySelectorAll(".session").forEach((button) => {
        button.addEventListener("click", () => selectSession(button.dataset.sessionId));
      });
    }

    function renderTabs() {
      ensureActiveTab();
      el("tabs").innerHTML = reportTabs().map(([id, label, icon]) => `
        <button class="tab ${id === state.activeTab ? "active" : ""}" type="button" data-tab="${id}">
          ${tabIcon(icon)}
          <span dir="auto">${escapeHtml(label)}</span>
        </button>
      `).join("");
      document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => {
          const tab = button.dataset.tab;
          if (tab !== "transcript") {
            clearEvidenceFocus();
          }
          state.activeTab = tab;
          render();
        });
      });
    }

    function sectionItemCount(names) {
      return names.reduce((total, name) => {
        const items = getSection(name).items;
        return total + (Array.isArray(items) ? items.length : 0);
      }, 0);
    }

    function statCard(icon, value, label) {
      return `
        <div class="stat-card">
          ${tabIcon(icon)}
          <div>
            <strong>${escapeHtml(value)}</strong>
            <span>${escapeHtml(label)}</span>
          </div>
        </div>
      `;
    }

    function renderReportStats() {
      const stats = el("reportStats");
      if (!state.report) {
        stats.hidden = true;
        stats.innerHTML = "";
        return;
      }
      const evidenceCount = Array.isArray(state.report.evidence_index) ? state.report.evidence_index.length : 0;
      const rows = Number(state.transcriptRows?.length || 0);
      const definitions = reportSectionDefinitions();
      const totalItems = definitions.reduce((total, definition) => total + sectionItemCount([definition.key]), 0);
      const groundedItems = definitions.reduce((total, definition) => total + (getSection(definition.key).items || []).filter((item) => item.grounding_status !== "missing_required_evidence").length, 0);
      stats.hidden = false;
      stats.innerHTML = [
        statCard("file", definitions.length, "Report sections"),
        statCard("tasks", totalItems, "Generated items"),
        statCard("check", groundedItems, "Evidence-grounded"),
        statCard("link", evidenceCount, "Evidence links"),
        statCard("wave", rows, "Transcript rows")
      ].join("");
    }

    function renderHeader() {
      const session = state.sessions.find((item) => item.id === state.sessionId) || {};
      const report = state.report || {};
      el("reportTitle").textContent = report.title || session.title || "Select a session";
      const badges = [];
      if (report.pipeline) badges.push(`<span class="badge hot">${escapeHtml(report.pipeline.mode || "pipeline")}</span>`);
      const template = effectiveReportTemplate();
      if (template.name) badges.push(`<span class="badge">${escapeHtml(template.name)}</span>`);
      if (template.privacy_policy === "local_only") badges.push('<span class="badge warn">Local only</span>');
      if (report.provider) badges.push(`<span class="badge">${escapeHtml(report.provider)}</span>`);
      if (report.report_language) badges.push(`<span class="badge">${escapeHtml(report.report_language)}</span>`);
      if (report.pipeline?.segments) badges.push(`<span class="badge">${report.pipeline.segments} segments</span>`);
      el("reportBadges").innerHTML = badges.join("");
      renderReportStats();
    }

    function generationIsTerminal(job = state.generationJob) {
      return Boolean(job && ["succeeded", "failed"].includes(job.status));
    }

    function openProgressOverlay() {
      el("progressOverlay").hidden = false;
      document.body.classList.add("progress-modal-open");
    }

    function closeProgressOverlay() {
      if (state.generating || !generationIsTerminal()) return;
      el("progressOverlay").hidden = true;
      document.body.classList.remove("progress-modal-open");
    }

    function renderProgress() {
      const overlay = el("progressOverlay");
      const dialog = el("progressDialog");
      const job = state.generationJob;
      if (!job) {
        overlay.hidden = true;
        document.body.classList.remove("progress-modal-open");
        return;
      }
      const terminal = generationIsTerminal(job);
      const canClose = terminal && !state.generating;
      if (state.generating || !terminal) openProgressOverlay();
      const percent = Math.max(0, Math.min(100, Number(job.percent || 0)));
      el("progressLabel").textContent = job.message || "Generating report";
      el("progressPercent").textContent = `${percent}%`;
      el("progressFill").style.width = `${percent}%`;
      const step = job.total ? `Step ${job.current || 0} of ${job.total}` : "";
      const detail = [job.stage, step, job.detail].filter(Boolean).join(" / ");
      el("progressDetail").textContent = detail;
      const events = Array.isArray(job.events) ? job.events.slice(-6) : [];
      el("progressLog").innerHTML = events.map((event) => `
        <div class="progress-event">
          <strong>${escapeHtml(event.stage || "")}</strong>
          <span>${escapeHtml(event.message || "")}${event.detail ? `: ${escapeHtml(event.detail)}` : ""}</span>
        </div>
      `).join("");
      dialog.classList.toggle("failed", job.status === "failed");
      dialog.classList.toggle("succeeded", job.status === "succeeded");
      el("progressDialogTitle").textContent = job.status === "succeeded"
        ? "Report generated"
        : job.status === "failed"
          ? "Report generation failed"
          : "Generating report";
      el("closeProgressBtn").disabled = !canClose;
      const footerButton = el("closeProgressFooterBtn");
      footerButton.disabled = !canClose;
      footerButton.textContent = job.status === "succeeded"
        ? "View report"
        : job.status === "failed"
          ? "Close"
          : "Generating...";
    }

    function getSection(name) {
      return state.report?.sections?.[name] || {items: [], summary: ""};
    }

    function evidenceLookup() {
      const result = new Map();
      (state.report?.evidence_index || []).forEach((item) => result.set(item.id, item));
      return result;
    }

    function scrollToTranscriptRows(rowIds) {
      const expected = new Set(rowIds.map((rowId) => encodeURIComponent(String(rowId))));
      const target = Array.from(document.querySelectorAll(".transcript-row")).find((node) => {
        const aliases = String(node.dataset.rowAliases || "").split(" ").filter(Boolean);
        return aliases.some((alias) => expected.has(alias));
      });
      if (!target) return;
      target.scrollIntoView({block: "center", behavior: "smooth"});
      target.focus({preventScroll: true});
    }

    function openEvidenceInTranscript(evidenceId) {
      const evidence = evidenceLookup();
      const item = evidence.get(evidenceId);
      const rowIds = evidenceRowIds(item);
      if (!item || !rowIds.length) return;
      const hadEvidenceFocus = Boolean(state.activeEvidenceId);
      state.evidenceReturnTab = state.activeTab === "transcript"
        ? state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript"
        : state.activeTab;
      state.activeEvidenceId = evidenceId;
      state.highlightRowIds = rowIds;
      state.activeTab = "transcript";
      try {
        const hash = `#evidence-${encodeURIComponent(evidenceId)}`;
        if (hadEvidenceFocus) {
          history.replaceState({meetingEvidence: true, evidenceId}, "", hash);
        } else {
          history.pushState({meetingEvidence: true, evidenceId}, "", hash);
        }
      } catch (error) {
        // Navigation history is a convenience; transcript focus still works without it.
      }
      render();
      window.requestAnimationFrame(() => scrollToTranscriptRows(rowIds));
    }

    function returnFromEvidence() {
      const tab = state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript";
      clearEvidenceFocus();
      state.activeTab = tab;
      try {
        if (location.hash.startsWith("#evidence-")) {
          history.replaceState(history.state, "", `${location.pathname}${location.search}`);
        }
      } catch (error) {
      }
      render();
    }

    function attachContentHandlers() {
      const content = el("content");
      content.querySelectorAll("[data-evidence-id]").forEach((node) => {
        node.addEventListener("click", (event) => {
          event.preventDefault();
          openEvidenceInTranscript(node.dataset.evidenceId);
        });
      });
      const back = content.querySelector("[data-evidence-back]");
      if (back) {
        back.addEventListener("click", returnFromEvidence);
      }
    }

    function evidenceChipsHtml(item, evidence) {
      const evidenceIds = Array.isArray(item.evidence_ids) ? item.evidence_ids : [];
      return evidenceIds.map((id) => {
        const ev = evidence.get(id);
        const label = ev ? `${id} ${ev.start || ""}` : id;
        const rowLabel = ev ? ` / ${evidenceRowLabel(ev)}` : "";
        return `<a class="badge hot evidence-link" href="#evidence-${encodeURIComponent(id)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(label + rowLabel)}</a>`;
      }).join("");
    }

    function itemAttributeMap(item) {
      const result = new Map();
      (Array.isArray(item?.attributes) ? item.attributes : []).forEach((attribute) => {
        const key = String(attribute?.key || "");
        if (key) result.set(key, String(attribute?.value || ""));
      });
      return result;
    }

    function itemHtml(item, evidence, sectionName = "") {
      const chips = evidenceChipsHtml(item, evidence);
      const meta = [
        item.status ? `Status: ${item.status}` : "",
        item.owner ? `Owner: ${item.owner}` : "",
        item.due ? `Due: ${item.due}` : "",
        item.confidence ? `Confidence: ${item.confidence}` : ""
      ].filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
      const attributes = Array.from(itemAttributeMap(item).entries()).map(([key, value]) => `<span class="attribute-chip"><strong>${escapeHtml(key)}:</strong> ${escapeHtml(value)}</span>`).join("");
      const grounding = item.grounding_status === "missing_required_evidence"
        ? '<div class="grounding-warning">Required transcript evidence is missing. Review this item before use.</div>'
        : "";
      return `
        <article class="item" dir="auto">
          <div class="item-title-row">
            ${sectionIcon(sectionName)}
            <h3>${escapeHtml(item.title || "Untitled")}</h3>
          </div>
          <p>${escapeHtml(item.body || item.summary || "")}</p>
          ${attributes ? `<div class="attribute-list">${attributes}</div>` : ""}
          ${grounding}
          <div class="item-footer">${meta}${chips}</div>
        </article>
      `;
    }

    function tableSectionItems(section, definition, evidence) {
      const items = Array.isArray(section.items) ? section.items : [];
      const fields = Array.isArray(definition.output_fields) ? definition.output_fields : [];
      const headings = fields.map((field) => `<th>${escapeHtml(field.label || field.key)}</th>`).join("");
      const rows = items.map((item) => {
        const attributes = itemAttributeMap(item);
        return `<tr dir="auto">
          <td><strong>${escapeHtml(item.title || "Untitled")}</strong><br><span class="meta">${escapeHtml(item.body || "")}</span>${item.grounding_status === "missing_required_evidence" ? '<div class="grounding-warning">Evidence missing</div>' : ""}</td>
          ${fields.map((field) => `<td>${escapeHtml(attributes.get(field.key) || "—")}</td>`).join("")}
          <td><div class="item-footer">${evidenceChipsHtml(item, evidence) || '<span class="badge warn">No evidence</span>'}</div></td>
        </tr>`;
      }).join("");
      return `<div class="report-table-wrap"><table class="report-table"><thead><tr><th>Item</th>${headings}<th>Evidence</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }

    function renderedSectionItems(section, definition, evidence) {
      const items = Array.isArray(section.items) ? section.items : [];
      if (!items.length) return `<div class="empty">No ${escapeHtml(definition.title || section.section || "items")} extracted.</div>`;
      const kind = definition.render_kind || "cards";
      if (kind === "table") return tableSectionItems(section, definition, evidence);
      const cards = items.map((item) => itemHtml(item, evidence, definition.key)).join("");
      if (kind === "timeline") return `<div class="timeline-list">${cards}</div>`;
      if (kind === "quotes") return `<div class="quote-layout">${cards}</div>`;
      return cards;
    }

    function sectionHtml(title, sectionNames) {
      const evidence = evidenceLookup();
      const blocks = sectionNames.map((name) => {
        const section = getSection(name);
        const items = Array.isArray(section.items) ? section.items : [];
        const definition = section.definition || reportSectionDefinitions().find((item) => item.key === name) || {key: name, title: sectionLabel(name), render_kind: "cards", output_fields: []};
        const body = renderedSectionItems(section, definition, evidence);
        return `
          <div class="section-block">
            <h3 class="section-heading">
              ${sectionIcon(name)}
              <span dir="auto">${escapeHtml(definition.title || sectionLabel(name))}</span>
              <span class="section-count">${escapeHtml(pluralLabel(items.length, "item"))}</span>
            </h3>
            ${section.summary ? `<div class="item" dir="auto"><p>${escapeHtml(section.summary)}</p></div>` : ""}
            ${body}
          </div>
        `;
      }).join("");
      return `<div class="section-grid" aria-label="${escapeHtml(title)}">${blocks}</div>`;
    }

    function evidenceHtml() {
      const items = state.report?.evidence_index || [];
      if (!items.length) return '<div class="empty">No evidence index available.</div>';
      return `<div class="evidence-list">${items.map((item) => `
        <article class="item" data-evidence-card="${escapeHtml(item.id)}">
          <h3>${escapeHtml(item.id)}: ${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.summary)}</p>
          <p class="quote">${escapeHtml(item.quote_excerpt || "")}</p>
          <div class="item-footer">
            <span class="badge">${escapeHtml(item.start || "")} - ${escapeHtml(item.end || "")}</span>
            <span class="badge">${escapeHtml((item.speakers || []).join(", "))}</span>
            <span class="badge">${escapeHtml(item.confidence || "")}</span>
            <button class="badge hot evidence-link" type="button" data-evidence-id="${escapeHtml(item.id)}">${escapeHtml(evidenceRowLabel(item))}</button>
          </div>
        </article>
      `).join("")}</div>`;
    }

    function transcriptHtml() {
      const rows = state.transcriptRows || [];
      if (!rows.length) return '<div class="empty">No transcript rows available.</div>';
      const evidence = state.activeEvidenceId ? evidenceLookup().get(state.activeEvidenceId) : null;
      const highlighted = new Set(state.highlightRowIds || []);
      const focus = evidence ? `
        <div class="transcript-focus">
          <div class="transcript-focus-title">
            <strong>${escapeHtml(state.activeEvidenceId)}: ${escapeHtml(evidence.title || "")}</strong>
            <span class="meta">${escapeHtml(evidence.summary || evidence.quote_excerpt || "")}</span>
          </div>
          <button class="btn" type="button" data-evidence-back="1">Back</button>
        </div>
      ` : "";
      return `${focus}<div class="transcript-list">${rows.map((row, index) => {
        const aliases = transcriptRowAliases(row, index);
        const isHighlighted = aliases.some((rowId) => highlighted.has(rowId));
        return `
        <article class="item transcript-row${isHighlighted ? " evidence-hit" : ""}" id="${escapeHtml(transcriptRowElementId(row, index))}" data-row-id="${escapeHtml(aliases[0] || "")}" data-row-aliases="${escapeHtml(encodedRowAliases(row, index))}" tabindex="-1">
          <h3>${escapeHtml(row.speaker_name || row.assigned_speaker || "Speaker")} <span class="meta">${Number(row.start || 0).toFixed(1)}s - ${Number(row.end || 0).toFixed(1)}s</span></h3>
          <p>${escapeHtml(row.text || "")}</p>
          ${isHighlighted ? `<div class="item-footer"><span class="badge hot">${escapeHtml(state.activeEvidenceId)}</span></div>` : ""}
        </article>
      `;
      }).join("")}</div>`;
    }

    function renderContent() {
      if (!state.sessionId) {
        el("content").innerHTML = '<div class="empty">Select a session.</div>';
        return;
      }
      if (!state.report) {
        el("content").innerHTML = '<div class="empty">No current report for this transcript revision.</div>';
        return;
      }
      const tab = state.activeTab;
      if (tab.startsWith("section:")) {
        const sectionKey = tab.slice("section:".length);
        const definition = reportSectionDefinitions().find((item) => item.key === sectionKey) || {title: sectionLabel(sectionKey)};
        el("content").innerHTML = sectionHtml(definition.title || sectionLabel(sectionKey), [sectionKey]);
      } else if (tab === "evidence") {
        el("content").innerHTML = evidenceHtml();
      } else if (tab === "transcript") {
        el("content").innerHTML = transcriptHtml();
      }
    }

    function render() {
      el("provider").innerHTML = providerLabel();
      renderProviderControls();
      el("generateBtn").disabled = !state.sessionId || state.generating;
      el("headerGenerateBtn").disabled = !state.sessionId || state.generating;
      el("headerTranscriptBtn").disabled = !state.sessionId || !state.transcriptRows.length;
      el("headerRefreshBtn").disabled = !state.sessionId || state.generating;
      const deleteButton = el("deleteReportBtn");
      deleteButton.disabled = !state.sessionId || !state.reportAvailable || state.generating;
      deleteButton.classList.toggle("confirming", state.confirmDelete && !deleteButton.disabled);
      const deleteLabel = state.confirmDelete && !deleteButton.disabled ? "Confirm" : "Delete";
      el("deleteReportLabel").textContent = deleteLabel;
      deleteButton.title = state.confirmDelete ? "Confirm cached report deletion" : "Delete cached report";
      deleteButton.setAttribute("aria-label", deleteButton.title);
      renderTemplateControls();
      renderSessions();
      renderTabs();
      renderHeader();
      renderProgress();
      renderContent();
      attachContentHandlers();
      setStatus(state.status);
    }

    async function refreshSelectedReport() {
      if (!state.sessionId || state.generating) return;
      await loadSessions(false);
      await selectSession(state.sessionId);
    }

    function openTranscriptTab() {
      if (!state.sessionId || !state.transcriptRows.length) return;
      clearEvidenceFocus();
      state.activeTab = "transcript";
      render();
    }

    async function applyProviderConfig() {
      if (state.generating || state.applyingProvider) return;
      const draft = state.providerDraft;
      const model = String(draft.model || "").trim();
      if (!model) {
        setStatus("Model is required");
        render();
        return;
      }
      state.applyingProvider = true;
      clearDeleteConfirm();
      setStatus("Switching provider");
      render();
      try {
        const data = await api("/api/llm-config", {
          method: "POST",
          body: JSON.stringify({
            provider: draft.provider,
            model,
            base_url: String(draft.base_url || "").trim()
          })
        });
        state.config = data.config || state.config;
        syncProviderDraftFromConfig();
        await loadSessions(false);
        if (state.sessionId) {
          await selectSession(state.sessionId);
        } else {
          setStatus("Provider switched");
        }
      } catch (error) {
        setStatus(error.message);
      } finally {
        state.applyingProvider = false;
        render();
      }
    }

    async function selectSession(sessionId) {
      state.sessionId = sessionId;
      state.report = null;
      state.reportAvailable = false;
      state.transcriptRows = [];
      clearEvidenceFocus();
      clearDeleteConfirm();
      if (!state.generating) {
        state.generationJob = null;
      }
      setStatus("Loading report");
      render();
      try {
        const data = await api(`/api/report?session_id=${encodeURIComponent(sessionId)}&template_id=${encodeURIComponent(state.templateId)}`);
        state.report = data.report;
        state.reportAvailable = Boolean(data.available);
        state.transcriptRows = data.transcript_rows || [];
        ensureActiveTab();
        setStatus(data.available ? "Cached report loaded" : data.stale ? "Cached report is stale" : "Ready");
        render();
        if (!data.available && state.config.auto_generate && state.templateId === state.config.standard_template_id) {
          await generateReport();
        }
      } catch (error) {
        setStatus(error.message);
        render();
      }
    }

    async function generateReport() {
      if (!state.sessionId || state.generating) return;
      state.generating = true;
      clearEvidenceFocus();
      clearDeleteConfirm();
      state.generationJob = {
        status: "queued",
        stage: "queued",
        message: "Queued report generation",
        detail: "",
        percent: 0,
        current: 0,
        total: 0,
        events: []
      };
      setStatus("Generating report");
      render();
      try {
        const data = await api("/api/generate-async", {
          method: "POST",
          body: JSON.stringify({session_id: state.sessionId, template_id: state.templateId})
        });
        state.generationJob = data.job;
        await pollGenerationJob(data.job.job_id);
        const reportData = await api(`/api/report?session_id=${encodeURIComponent(state.sessionId)}&template_id=${encodeURIComponent(state.templateId)}`);
        state.report = reportData.report;
        state.reportAvailable = Boolean(reportData.available);
        state.transcriptRows = reportData.transcript_rows || [];
        setStatus("Report generated");
      } catch (error) {
        state.generationJob = {
          ...(state.generationJob || {}),
          status: "failed",
          stage: "failed",
          message: "Report generation failed",
          detail: error.message,
          error: error.message
        };
        setStatus(error.message);
      } finally {
        state.generating = false;
        await loadSessions(false);
        render();
      }
    }

    async function deleteReport() {
      if (!state.sessionId || !state.reportAvailable || state.generating) return;
      if (!state.confirmDelete) {
        state.confirmDelete = true;
        setStatus("Click Confirm to delete the cached report");
        render();
        return;
      }
      clearEvidenceFocus();
      clearDeleteConfirm();
      state.generationJob = null;
      setStatus("Deleting report");
      render();
      try {
        const data = await api("/api/delete-report", {
          method: "POST",
          body: JSON.stringify({session_id: state.sessionId, template_id: state.templateId})
        });
        state.report = null;
        state.reportAvailable = false;
        state.transcriptRows = data.transcript_rows || state.transcriptRows || [];
        state.activeTab = reportTabs()[0]?.[0] || "transcript";
        setStatus(data.deleted ? "Report deleted" : "No cached report found");
        await loadSessions(false);
      } catch (error) {
        setStatus(error.message);
      } finally {
        render();
      }
    }

    async function pollGenerationJob(jobId) {
      for (;;) {
        const data = await api(`/api/generate-status?job_id=${encodeURIComponent(jobId)}`);
        const job = data.job || {};
        state.generationJob = job;
        setStatus(job.message || "Generating report");
        render();
        if (job.status === "succeeded") {
          return;
        }
        if (job.status === "failed") {
          throw new Error(job.error || job.detail || "Report generation failed");
        }
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    }

    async function loadSessions(selectFirst = true) {
      const data = await api("/api/sessions");
      state.sessions = data.sessions || [];
      if (selectFirst && state.sessions.length && !state.sessionId) {
        await selectSession(state.sessions[0].id);
      }
    }

    async function boot() {
      try {
        state.config = (await api("/api/config")).config || {};
        syncProviderDraftFromConfig();
        await loadTemplates(state.config.standard_template_id || "");
        await loadSessions(true);
        render();
      } catch (error) {
        setStatus(error.message);
        render();
      }
    }

    el("generateBtn").addEventListener("click", generateReport);
    el("deleteReportBtn").addEventListener("click", deleteReport);
    el("headerGenerateBtn").addEventListener("click", generateReport);
    el("headerTranscriptBtn").addEventListener("click", openTranscriptTab);
    el("headerRefreshBtn").addEventListener("click", () => refreshSelectedReport().catch((error) => setStatus(error.message)));
    el("refreshBtn").addEventListener("click", () => loadSessions(false).then(render).catch((error) => setStatus(error.message)));
    el("sessionSearch").addEventListener("input", renderSessions);
    el("templateSelect").addEventListener("change", async () => {
      state.templateId = el("templateSelect").value;
      state.report = null;
      state.reportAvailable = false;
      clearEvidenceFocus();
      state.activeTab = sectionTabId(currentTemplate()?.sections?.[0]?.key || "");
      render();
      if (state.sessionId) await selectSession(state.sessionId);
    });
    el("newTemplateBtn").addEventListener("click", () => openTemplateBuilder("new"));
    el("inspectTemplateBtn").addEventListener("click", () => openTemplateBuilder("inspect"));
    el("cloneTemplateBtn").addEventListener("click", () => openTemplateBuilder("clone"));
    el("deleteTemplateBtn").addEventListener("click", () => deleteSelectedTemplate().catch((error) => setStatus(error.message)));
    el("closeBuilderBtn").addEventListener("click", closeTemplateBuilder);
    el("cancelBuilderBtn").addEventListener("click", closeTemplateBuilder);
    el("saveTemplateBtn").addEventListener("click", saveTemplateBuilder);
    el("templateBuilder").addEventListener("click", (event) => {
      if (event.target === el("templateBuilder")) closeTemplateBuilder();
    });
    el("closeProgressBtn").addEventListener("click", closeProgressOverlay);
    el("closeProgressFooterBtn").addEventListener("click", closeProgressOverlay);
    el("progressOverlay").addEventListener("click", (event) => {
      if (event.target === el("progressOverlay")) closeProgressOverlay();
    });
    el("llmProviderSelect").addEventListener("change", () => {
      const option = providerOption(el("llmProviderSelect").value);
      state.providerDraft.provider = option.id || el("llmProviderSelect").value;
      state.providerDraft.model = option.models?.[0] || "";
      state.providerDraft.base_url = option.default_base_url || "";
      renderProviderControls();
      if (option.id === "openai" || option.id === "openrouter") {
        loadProviderModels();
      }
    });
    el("llmModelSelect").addEventListener("change", () => {
      const value = el("llmModelSelect").value;
      if (value === "__custom__") {
        if (providerModelOptions(selectedProviderOption()).includes(state.providerDraft.model)) {
          state.providerDraft.model = "";
        }
      } else {
        state.providerDraft.model = value;
      }
      renderProviderControls();
      if (value === "__custom__") {
        el("llmModelInput").focus();
      }
    });
    el("llmModelInput").addEventListener("input", () => {
      state.providerDraft.model = el("llmModelInput").value;
      renderProviderControls();
    });
    el("llmBaseUrlInput").addEventListener("input", () => {
      state.providerDraft.base_url = el("llmBaseUrlInput").value;
      renderProviderControls();
    });
    el("applyProviderBtn").addEventListener("click", () => applyProviderConfig());
    el("loadModelsBtn").addEventListener("click", () => loadProviderModels());
    window.addEventListener("popstate", () => {
      if (state.activeEvidenceId) {
        const tab = state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript";
        clearEvidenceFocus();
        state.activeTab = tab;
        render();
      }
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !el("templateBuilder").hidden) closeTemplateBuilder();
      if (event.key === "Escape" && !el("progressOverlay").hidden) closeProgressOverlay();
    });
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
