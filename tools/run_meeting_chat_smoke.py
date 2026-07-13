"""Run a short grounded meeting-chat smoke test against a compatible LLM."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.meeting_intelligence_pipeline import default_llm_config
from window.meeting_intelligence_server import (
    DEMO_SESSION_ID,
    MeetingIntelligenceServerConfig,
    MeetingIntelligenceService,
    make_handler,
)


def request_json(base_url: str, method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=ROOT / "docs-private" / "demo-meeting" / "whospeakslive_transcript.txt",
    )
    parser.add_argument("--question", default="What was decided, and who received a concrete task?")
    parser.add_argument("--max-lines", type=int, default=0, help="Use only the first N transcript lines for a small smoke fixture.")
    parser.add_argument("--session-dir", type=Path, help="Use an existing saved-session directory instead of the demo transcript.")
    parser.add_argument("--session-id", help="Saved meeting ID to ask; requires --session-dir.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if bool(args.session_dir) != bool(args.session_id):
        parser.error("--session-dir and --session-id must be supplied together")

    with tempfile.TemporaryDirectory(prefix="whospeaks-meeting-chat-smoke-") as directory:
        runtime = Path(directory)
        session_id = str(args.session_id or DEMO_SESSION_ID)
        session_dir = args.session_dir.resolve() if args.session_dir else runtime / "sessions"
        transcript: Path | None = None
        if not args.session_dir:
            transcript = args.transcript.resolve()
            if args.max_lines > 0:
                transcript = runtime / "smoke-transcript.txt"
                source_lines = args.transcript.read_text(encoding="utf-8").splitlines()
                transcript.write_text("\n".join(source_lines[:args.max_lines]) + "\n", encoding="utf-8")
        service = MeetingIntelligenceService(MeetingIntelligenceServerConfig(
            session_dir=session_dir,
            cache_dir=runtime / "reports",
            template_dir=runtime / "templates",
            chat_dir=runtime / "chats",
            text_index_db=runtime / "meeting-index.sqlite3",
            demo_transcript=transcript,
            llm_config=default_llm_config(
                "llama_cpp",
                base_url=str(args.llm_base_url).rstrip("/"),
                model=str(args.llm_model),
            ),
        ))
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        try:
            scope = request_json(base_url, "POST", "/api/chat/scope", {"session_ids": [session_id]})
            started = request_json(base_url, "POST", "/api/chat/ask-async", {
                "session_ids": [session_id],
                "question": args.question,
            })
            job_id = str(started["job"]["job_id"])  # type: ignore[index]
            deadline = time.monotonic() + max(1.0, args.timeout_seconds)
            job: dict[str, object] = {}
            while time.monotonic() < deadline:
                job = request_json(base_url, "GET", f"/api/chat/job?job_id={job_id}")["job"]  # type: ignore[assignment]
                if job.get("status") in {"succeeded", "failed"}:
                    break
                time.sleep(0.25)
            if job.get("status") != "succeeded":
                raise RuntimeError(str(job.get("error") or "Meeting chat smoke test timed out."))
            print(json.dumps({"scope": scope, "job": job}, ensure_ascii=False, indent=2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
