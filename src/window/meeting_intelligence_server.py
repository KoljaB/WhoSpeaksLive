"""Standalone browser server for LLM-based meeting intelligence reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
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
    stable_hash,
)
from window.session_store import DEFAULT_SESSION_DIR, SessionStore


DEMO_SESSION_ID = "demo-whospeakslive-transcript"
DEFAULT_CACHE_DIR = RUNTIME_DIR / "meeting_intelligence_reports"
TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?::\d+){1,2}(?:\.\d+)?)\s+-\s+"
    r"(?P<end>\d+(?::\d+){1,2}(?:\.\d+)?)\]\s+"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


@dataclass(frozen=True)
class MeetingIntelligenceServerConfig:
    session_dir: Path = DEFAULT_SESSION_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    demo_transcript: Path | None = None
    llm_config: MeetingLLMConfig = default_llm_config()
    mock_llm: bool = False
    max_segment_rows: int = 80
    auto_generate: bool = False


@dataclass
class GenerationJob:
    job_id: str
    session_id: str
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
        self.store = SessionStore(config.session_dir)
        self.client_factory = client_factory
        self._jobs: dict[str, GenerationJob] = {}
        self._jobs_lock = threading.Lock()

    def public_config(self) -> dict[str, Any]:
        llm = self.config.llm_config
        return {
            "provider": llm.provider,
            "base_url": llm.base_url,
            "model": llm.model,
            "schema_mode": llm.schema_mode,
            "mock_llm": self.config.mock_llm,
            "auto_generate": self.config.auto_generate,
            "max_segment_rows": self.config.max_segment_rows,
            "expected_report_provider": self.expected_report_provider(),
        }

    def expected_report_provider(self) -> str:
        if self.config.mock_llm:
            return MockMeetingLLMClient.name
        llm = self.config.llm_config
        return f"{llm.provider}:{llm.model}"

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if self.config.demo_transcript and self.config.demo_transcript.is_file():
            demo = self._load_demo_session()
            sessions.append({
                **demo["summary"],
                "source_kind": "demo_transcript",
                "has_cached_report": bool(self._read_runtime_cached_report(DEMO_SESSION_ID)),
            })
        for session in self.store.list_sessions("all"):
            item = dict(session)
            item["source_kind"] = "saved_session"
            item["has_cached_report"] = bool(self._read_runtime_cached_report(str(item.get("id") or "")))
            sessions.append(item)
        return sessions

    def load_session(self, session_id: str) -> dict[str, Any]:
        if session_id == DEMO_SESSION_ID:
            if not self.config.demo_transcript or not self.config.demo_transcript.is_file():
                raise ValueError("Demo transcript is not configured.")
            return self._load_demo_session()
        return self.store.open_session(session_id)

    def get_report(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        rows = [dict(row) for row in session.get("transcript_rows") or []]
        speaker_state = session.get("speaker_state") if isinstance(session.get("speaker_state"), dict) else {}
        revision_id = transcript_revision_id(rows, speaker_state)
        report = self._read_cached_report(session_id)
        available = (
            isinstance(report, dict)
            and report.get("transcript_revision_id") == revision_id
            and report.get("provider") == self.expected_report_provider()
        )
        return {
            "available": available,
            "stale": bool(report and not available),
            "session": session.get("summary") or {},
            "report": report if available else None,
            "transcript_rows": rows,
        }

    def generate_report(
        self,
        session_id: str,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        rows = [dict(row) for row in session.get("transcript_rows") or []]
        if not rows:
            raise ValueError("Selected session has no transcript rows.")
        speaker_state = session.get("speaker_state") if isinstance(session.get("speaker_state"), dict) else {}
        summary = session.get("summary") if isinstance(session.get("summary"), dict) else {}
        title = str(summary.get("title") or session_id)
        client = self._new_client()
        pipeline = MultiPassMeetingIntelligencePipeline(
            client,
            max_segment_rows=self.config.max_segment_rows,
            evidence_max_tokens=self.config.llm_config.max_tokens,
            section_max_tokens=self.config.llm_config.section_max_tokens,
            progress_callback=progress_callback,
        )
        report = pipeline.generate(
            session_id=session_id,
            transcript_rows=rows,
            speaker_state=speaker_state,
            title=title,
        )
        self._write_cached_report(session_id, report)
        return {
            "available": True,
            "stale": False,
            "session": summary,
            "report": report,
            "transcript_rows": rows,
        }

    def delete_report(self, session_id: str) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("Session id is required.")
        session = self.load_session(session_id)
        path = self._cache_path(session_id)
        deleted = False
        if path.is_file():
            path.unlink()
            deleted = True
        return {
            "deleted": deleted,
            "session": session.get("summary") or {},
            "report": None,
            "available": False,
            "stale": False,
            "transcript_rows": [dict(row) for row in session.get("transcript_rows") or []],
        }

    def start_generate_report(self, session_id: str) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("Session id is required.")
        with self._jobs_lock:
            for job in self._jobs.values():
                if job.session_id == session_id and job.status in {"queued", "running"}:
                    return self._job_snapshot(job)
            job = GenerationJob(
                job_id=f"mirjob_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                message="Queued report generation",
            )
            self._jobs[job.job_id] = job
            snapshot = self._job_snapshot(job)
        thread = threading.Thread(
            target=self._run_generation_job,
            args=(job.job_id, session_id),
            name=f"meeting-intelligence-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return snapshot

    def get_generation_job(self, job_id: str) -> dict[str, Any]:
        job_id = str(job_id or "").strip()
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Generation job not found.")
            return self._job_snapshot(job)

    def _run_generation_job(self, job_id: str, session_id: str) -> None:
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

    def _new_client(self) -> StructuredChatClient:
        if self.client_factory is not None:
            return self.client_factory()
        if self.config.mock_llm:
            return MockMeetingLLMClient()
        return OpenAICompatibleMeetingClient(self.config.llm_config)

    def _cache_path(self, session_id: str) -> Path:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-")
        if not clean:
            clean = "session"
        suffix = stable_hash(session_id, length=10)
        return self.config.cache_dir / f"{clean[:80]}-{suffix}.json"

    def _read_cached_report(self, session_id: str) -> dict[str, Any] | None:
        path = self._cache_path(session_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        report = payload.get("report") if isinstance(payload, dict) else None
        return report if isinstance(report, dict) else None

    def _read_runtime_cached_report(self, session_id: str) -> dict[str, Any] | None:
        report = self._read_cached_report(session_id)
        if not isinstance(report, dict):
            return None
        if report.get("provider") != self.expected_report_provider():
            return None
        return report

    def _write_cached_report(self, session_id: str, report: dict[str, Any]) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "report": report,
        }
        self._cache_path(session_id).write_text(
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
        cached_report = self._read_runtime_cached_report(DEMO_SESSION_ID)
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
                if parsed.path == "/api/sessions":
                    self._send_json({"sessions": service.list_sessions()})
                    return
                if parsed.path == "/api/report":
                    session_id = single_query_value(parsed.query, "session_id")
                    self._send_json(service.get_report(session_id))
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
                if parsed.path not in {"/api/generate", "/api/generate-async", "/api/delete-report"}:
                    self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json_body()
                session_id = str(payload.get("session_id") or "").strip()
                if parsed.path == "/api/delete-report":
                    self._send_json(service.delete_report(session_id))
                    return
                if parsed.path == "/api/generate-async":
                    self._send_json({"job": service.start_generate_report(session_id)})
                    return
                self._send_json(service.generate_report(session_id))
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve WhoSpeaks meeting intelligence reports.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--demo-transcript", type=Path)
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
    return parser


def config_from_args(args: argparse.Namespace) -> MeetingIntelligenceServerConfig:
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
        demo_transcript=args.demo_transcript.expanduser().resolve() if args.demo_transcript else None,
        llm_config=default_llm_config(args.llm_provider, **overrides),
        mock_llm=bool(args.mock_llm),
        max_segment_rows=max(12, int(args.max_segment_rows)),
        auto_generate=bool(args.auto_generate),
    )


def run_server(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    service = MeetingIntelligenceService(config)
    handler = make_handler(service)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Meeting intelligence server: http://{args.host}:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


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
    input {
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
      grid-template-rows: auto auto minmax(0, 1fr);
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
      gap: 4px;
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
      padding: 22px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(10px);
      display: grid;
      gap: 10px;
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
      font-size: 26px;
      line-height: 1.2;
      font-weight: 780;
      overflow-wrap: anywhere;
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

    .progress-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 12px;
      display: grid;
      gap: 9px;
      max-width: 980px;
      box-shadow: var(--shadow-soft);
    }

    .progress-panel[hidden] {
      display: none;
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

    .tabbar {
      padding: 9px 28px 0;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 6px;
      overflow-x: auto;
      scrollbar-width: none;
      background: rgba(255, 255, 255, 0.84);
      position: sticky;
      top: 98px;
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
      padding: 24px 28px 38px;
    }

    .section-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }

    .section-block {
      display: grid;
      gap: 11px;
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

    .section-heading::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 2px;
      background: var(--accent);
    }

    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow-soft);
      padding: 15px;
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
      font-size: 16px;
      line-height: 1.3;
      color: #111827;
      font-weight: 760;
      overflow-wrap: anywhere;
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
      </div>
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
          <h2 id="reportTitle">Select a session</h2>
          <div class="badge-row" id="reportBadges"></div>
        </div>
        <div class="status" id="status"></div>
        <div class="progress-panel" id="progressPanel" hidden>
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
      </header>
      <nav class="tabbar" id="tabs"></nav>
      <section class="content" id="content"></section>
    </main>
  </div>
  <script>
    const tabs = [
      ["summary", "Summary", "list"],
      ["decisions", "Decisions", "check"],
      ["action_items", "Action items", "tasks"],
      ["questions", "Questions", "question"],
      ["risks", "Risks", "shield"],
      ["evidence", "Evidence", "file"],
      ["transcript", "Transcript", "wave"]
    ];
    const state = {
      config: {},
      sessions: [],
      sessionId: "",
      report: null,
      reportAvailable: false,
      transcriptRows: [],
      activeTab: "summary",
      generating: false,
      generationJob: null,
      activeEvidenceId: "",
      highlightRowIds: [],
      evidenceReturnTab: "",
      confirmDelete: false,
      status: ""
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

    function providerLabel() {
      const cfg = state.config || {};
      const mode = cfg.mock_llm ? "mock" : cfg.provider;
      return `
        <div><strong>${escapeHtml(mode)}</strong> / ${escapeHtml(cfg.model || "local")}</div>
        <div>${escapeHtml(cfg.base_url || "no base URL")}</div>
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
        wave: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 10v4M8 7v10M12 5v14M16 8v8M20 11v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
      };
      return icons[name] || "";
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
        const cached = session.has_cached_report || session.has_meeting_intelligence;
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
      el("tabs").innerHTML = tabs.map(([id, label, icon]) => `
        <button class="tab ${id === state.activeTab ? "active" : ""}" type="button" data-tab="${id}">
          ${tabIcon(icon)}
          ${escapeHtml(label)}
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

    function renderHeader() {
      const session = state.sessions.find((item) => item.id === state.sessionId) || {};
      const report = state.report || {};
      el("reportTitle").textContent = report.title || session.title || "Select a session";
      const badges = [];
      if (report.pipeline) badges.push(`<span class="badge hot">${escapeHtml(report.pipeline.mode || "pipeline")}</span>`);
      if (report.provider) badges.push(`<span class="badge">${escapeHtml(report.provider)}</span>`);
      if (report.pipeline?.segments) badges.push(`<span class="badge">${report.pipeline.segments} segments</span>`);
      el("reportBadges").innerHTML = badges.join("");
    }

    function renderProgress() {
      const panel = el("progressPanel");
      const job = state.generationJob;
      if (!job) {
        panel.hidden = true;
        return;
      }
      panel.hidden = false;
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
        ? state.evidenceReturnTab || "summary"
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
      const tab = state.evidenceReturnTab || "summary";
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

    function itemHtml(item, evidence) {
      const evidenceIds = Array.isArray(item.evidence_ids) ? item.evidence_ids : [];
      const chips = evidenceIds.map((id) => {
        const ev = evidence.get(id);
        const label = ev ? `${id} ${ev.start || ""}` : id;
        const rowLabel = ev ? ` / ${evidenceRowLabel(ev)}` : "";
        return `<a class="badge hot evidence-link" href="#evidence-${encodeURIComponent(id)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(label + rowLabel)}</a>`;
      }).join("");
      const meta = [
        item.status ? `Status: ${item.status}` : "",
        item.owner ? `Owner: ${item.owner}` : "",
        item.due ? `Due: ${item.due}` : "",
        item.confidence ? `Confidence: ${item.confidence}` : ""
      ].filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
      return `
        <article class="item">
          <h3>${escapeHtml(item.title || "Untitled")}</h3>
          <p>${escapeHtml(item.body || item.summary || "")}</p>
          <div class="item-footer">${meta}${chips}</div>
        </article>
      `;
    }

    function sectionHtml(title, sectionNames) {
      const evidence = evidenceLookup();
      const blocks = sectionNames.map((name) => {
        const section = getSection(name);
        const items = Array.isArray(section.items) ? section.items : [];
        const body = items.length
          ? items.map((item) => itemHtml(item, evidence)).join("")
          : `<div class="empty">No ${name.replaceAll("_", " ")} extracted.</div>`;
        return `
          <div class="section-block">
            <h3 class="section-heading">${escapeHtml(name.replaceAll("_", " "))}</h3>
            ${section.summary ? `<div class="item"><p>${escapeHtml(section.summary)}</p></div>` : ""}
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
      if (tab === "summary") {
        el("content").innerHTML = sectionHtml("Summary", [
          "executive_summary",
          "structured_brief",
          "speaker_map",
          "speaker_participation",
          "discussion_threads"
        ]);
      } else if (tab === "decisions") {
        el("content").innerHTML = sectionHtml("Decisions", ["decisions", "deadlines", "disagreements"]);
      } else if (tab === "action_items") {
        el("content").innerHTML = sectionHtml("Action items", ["action_items"]);
      } else if (tab === "questions") {
        el("content").innerHTML = sectionHtml("Questions", ["open_questions", "ask_this_meeting"]);
      } else if (tab === "risks") {
        el("content").innerHTML = sectionHtml("Risks", ["risks"]);
      } else if (tab === "evidence") {
        el("content").innerHTML = evidenceHtml();
      } else if (tab === "transcript") {
        el("content").innerHTML = transcriptHtml();
      }
    }

    function render() {
      el("provider").innerHTML = providerLabel();
      el("generateBtn").disabled = !state.sessionId || state.generating;
      const deleteButton = el("deleteReportBtn");
      deleteButton.disabled = !state.sessionId || !state.reportAvailable || state.generating;
      deleteButton.classList.toggle("confirming", state.confirmDelete && !deleteButton.disabled);
      const deleteLabel = state.confirmDelete && !deleteButton.disabled ? "Confirm" : "Delete";
      el("deleteReportLabel").textContent = deleteLabel;
      deleteButton.title = state.confirmDelete ? "Confirm cached report deletion" : "Delete cached report";
      deleteButton.setAttribute("aria-label", deleteButton.title);
      renderSessions();
      renderTabs();
      renderHeader();
      renderProgress();
      renderContent();
      attachContentHandlers();
      setStatus(state.status);
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
        const data = await api(`/api/report?session_id=${encodeURIComponent(sessionId)}`);
        state.report = data.report;
        state.reportAvailable = Boolean(data.available);
        state.transcriptRows = data.transcript_rows || [];
        setStatus(data.available ? "Cached report loaded" : data.stale ? "Cached report is stale" : "Ready");
        render();
        if (!data.available && state.config.auto_generate) {
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
          body: JSON.stringify({session_id: state.sessionId})
        });
        state.generationJob = data.job;
        await pollGenerationJob(data.job.job_id);
        const reportData = await api(`/api/report?session_id=${encodeURIComponent(state.sessionId)}`);
        state.report = reportData.report;
        state.reportAvailable = Boolean(reportData.available);
        state.transcriptRows = reportData.transcript_rows || [];
        setStatus("Report generated");
      } catch (error) {
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
          body: JSON.stringify({session_id: state.sessionId})
        });
        state.report = null;
        state.reportAvailable = false;
        state.transcriptRows = data.transcript_rows || state.transcriptRows || [];
        state.activeTab = "summary";
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
        await loadSessions(true);
        render();
      } catch (error) {
        setStatus(error.message);
        render();
      }
    }

    el("generateBtn").addEventListener("click", generateReport);
    el("deleteReportBtn").addEventListener("click", deleteReport);
    el("refreshBtn").addEventListener("click", () => loadSessions(false).then(render).catch((error) => setStatus(error.message)));
    el("sessionSearch").addEventListener("input", renderSessions);
    window.addEventListener("popstate", () => {
      if (state.activeEvidenceId) {
        const tab = state.evidenceReturnTab || "summary";
        clearEvidenceFocus();
        state.activeTab = tab;
        render();
      }
    });
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
