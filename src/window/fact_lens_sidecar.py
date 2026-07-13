"""Sidecar dashboard for live LLM-based claim extraction from WhoSpeaksLive."""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from window.evidence_matching import (
    evidence_matches_transcript,
    normalize_evidence_text,
)
from window.fact_lens import (
    ClaimCard,
    ExtractedClaim,
    ExtractionJob,
    ExtractionResult,
    FactLensStore,
    SentenceRevisionToken,
    SidecarState,
    SnapshotPublisher,
    TranscriptSentence,
    claim_card_id,
    coalesce_sentences,
    placeholder_card_id,
    wall_now,
)
from window.fact_lens.runtime import FactLensRuntime
from window.web_assets import read_web_text


DEFAULT_SOURCE_URL = "http://127.0.0.1:8796"
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8890
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LLM_MODEL = "local"
SCHEMA_VERSION = "claim_triage_v1"
CLAIM_STATUSES = {"queued", "checking", "ignored", "needs_context", "unverified", "supported", "contradicted", "mixed"}
CLASSIFICATIONS = {"ignore", "checkable_claim", "needs_context"}
PRIORITIES = {"low", "medium", "high"}


CLAIM_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "sentence_id", "classification", "claims", "rationale"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "sentence_id": {"type": "string"},
        "classification": {"enum": sorted(CLASSIFICATIONS)},
        "rationale": {"type": "string", "maxLength": 240},
        "claims": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "evidence", "priority", "rationale"],
                "properties": {
                    "claim": {"type": "string", "maxLength": 240},
                    "evidence": {"type": "string", "maxLength": 240},
                    "priority": {"enum": sorted(PRIORITIES)},
                    "rationale": {"type": "string", "maxLength": 240},
                },
            },
        },
    },
}


def normalize_text(text: str) -> str:
    return normalize_evidence_text(text)


def strip_json_fences(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class ClaimExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ServerSentEvent:
    event: str
    data: str
    event_id: str = ""


def parse_sse_lines(lines: Iterable[str | bytes]) -> Iterator[ServerSentEvent]:
    event_name = "message"
    event_id = ""
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield ServerSentEvent(event_name, "\n".join(data_lines), event_id)
            event_name = "message"
            event_id = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        yield ServerSentEvent(event_name, "\n".join(data_lines), event_id)


def iter_url_sse(url: str, *, timeout: float | None = None) -> Iterator[ServerSentEvent]:
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        yield from parse_sse_lines(response)


def parse_openai_chat_json(response_data: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClaimExtractionError("missing_chat_content") from exc

    if isinstance(content, list):
        content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    if not isinstance(content, str):
        raise ClaimExtractionError("chat_content_not_string")

    try:
        payload = json.loads(strip_json_fences(content))
    except json.JSONDecodeError as exc:
        raise ClaimExtractionError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise ClaimExtractionError("json_not_object")
    return payload


def validate_extraction_payload(payload: dict[str, Any], sentence: TranscriptSentence) -> ExtractionResult:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ClaimExtractionError("invalid_schema_version")
    sentence_id = str(payload.get("sentence_id") or "")
    if sentence_id and sentence_id != sentence.id:
        raise ClaimExtractionError("sentence_id_mismatch")

    classification = str(payload.get("classification") or "")
    if classification not in CLASSIFICATIONS:
        raise ClaimExtractionError("invalid_classification")

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise ClaimExtractionError("claims_not_list")

    accepted: list[ExtractedClaim] = []
    rejected: list[str] = []
    accepted_keys: set[tuple[str, str]] = set()
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, dict):
            rejected.append(f"claim_{index}:not_object")
            continue
        claim_text = str(raw_claim.get("claim") or "").strip()
        evidence = str(raw_claim.get("evidence") or "").strip()
        priority = str(raw_claim.get("priority") or "medium")
        rationale = str(raw_claim.get("rationale") or "").strip()
        if priority not in PRIORITIES:
            priority = "medium"
        if not claim_text:
            rejected.append(f"claim_{index}:empty_claim")
            continue
        if not evidence:
            rejected.append(f"claim_{index}:missing_evidence")
            continue
        if not evidence_matches_transcript(evidence, sentence.text):
            rejected.append(f"claim_{index}:evidence_mismatch")
            continue
        claim_key = (normalize_text(claim_text), normalize_text(evidence))
        if claim_key in accepted_keys:
            rejected.append(f"claim_{index}:duplicate_claim")
            continue
        accepted_keys.add(claim_key)
        accepted.append(ExtractedClaim(claim=claim_text, evidence=evidence, priority=priority, rationale=rationale))

    if classification != "checkable_claim":
        accepted = []
    classification_for_result = classification
    if classification == "checkable_claim" and not accepted:
        classification_for_result = "needs_context"

    return ExtractionResult(
        sentence_id=sentence.id,
        classification=classification_for_result,
        rationale=str(payload.get("rationale") or "").strip(),
        claims=accepted,
        rejected_claims=rejected,
    )


class OpenAIClaimClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model: str = DEFAULT_LLM_MODEL,
        client_name: str = "whospeaks-fact-lens",
        lane: str = "shared",
        timeout: float = 12.0,
        max_tokens: int = 768,
        schema_mode: str = "both",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client_name = client_name
        self.lane = lane
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.schema_mode = schema_mode

    def extract(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        payload = self.build_payload(sentence, context)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-LLM-Client": self.client_name,
                "X-LLM-Lane": self.lane,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return parse_openai_chat_json(response_data)

    def build_payload(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        prompt_context = {
            "sentence": sentence.to_prompt_dict(),
            "recent_final_sentences": [item.to_prompt_dict() for item in context],
            "verdict_policy": "Do not verify or label true/false here. Extract checkable claims only; unresolved claims stay unverified.",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": build_claim_extraction_prompt()},
                {"role": "user", "content": json.dumps(prompt_context, ensure_ascii=True)},
            ],
        }
        if self.schema_mode in {"json_schema", "both"}:
            payload["json_schema"] = CLAIM_EXTRACTION_SCHEMA
        if self.schema_mode in {"response_format", "both"}:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "who_speaks_claim_triage",
                    "strict": True,
                    "schema": CLAIM_EXTRACTION_SCHEMA,
                },
            }
        return payload


class MockClaimClient:
    def extract(self, sentence: TranscriptSentence, context: Iterable[TranscriptSentence]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sentence_id": sentence.id,
            "classification": "checkable_claim",
            "rationale": "Synthetic offline extraction.",
            "claims": [
                {
                    "claim": sentence.text,
                    "evidence": sentence.text,
                    "priority": "medium",
                    "rationale": "Mock mode uses the full synthetic sentence as evidence.",
                }
            ],
        }


def build_claim_extraction_prompt() -> str:
    return (
        "You classify one finalized speech transcript sentence for live fact-check preparation. "
        "Return JSON only. Use classification ignore for non-factual remarks, pure opinions, "
        "questions, backchannels, or fragments that should not be checked. Use checkable_claim "
        "for externally verifiable factual statements. Use needs_context when the sentence may "
        "contain a claim but cannot stand alone. Extract only atomic claims that were explicitly "
        "said in the sentence. Copy evidence from the sentence exactly or near-exactly; do not "
        "invent evidence or combine separate claims. Do not fact-check, search, or assign true or "
        "false labels. Return at most three claims and never repeat the same claim. If no atomic "
        "checkable claim is present, return an empty claims array."
    )


class ClaimExtractionWorker:
    def __init__(
        self,
        *,
        state: SidecarState,
        client: Any,
        work_queue: "queue.Queue[ExtractionJob]",
        stop_event: threading.Event,
        debounce_seconds: float = 0.35,
        context_size: int = 8,
    ) -> None:
        self.state = state
        self.client = client
        self.work_queue = work_queue
        self.stop_event = stop_event
        self.debounce_seconds = debounce_seconds
        self.context_size = context_size
        self.thread = threading.Thread(target=self._run, name="fact-lens-llm-worker", daemon=True)
        self._start_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self.thread.start()

    def submit(self, sentence: TranscriptSentence, token: SentenceRevisionToken | None = None) -> None:
        token = token or self.state.current_token(sentence.id)
        if token is None:
            return
        job = ExtractionJob(sentence, token)
        try:
            self.work_queue.put_nowait(job)
        except queue.Full:
            dropped: ExtractionJob | None = None
            try:
                dropped = self.work_queue.get_nowait()
                self.work_queue.task_done()
            except queue.Empty:
                pass
            self.state.record_queue_drop(dropped)
            self.work_queue.put_nowait(job)

    def stop(self, *, timeout: float = 30.0) -> None:
        self.stop_event.set()
        with self._start_lock:
            started = self._started
        if not started:
            return
        self.thread.join(timeout=max(0.0, float(timeout)))
        if self.thread.is_alive():
            raise RuntimeError("fact lens extraction worker did not stop")

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                first = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            batch = [first]
            if self.debounce_seconds > 0:
                self.stop_event.wait(self.debounce_seconds)
            while True:
                try:
                    batch.append(self.work_queue.get_nowait())
                except queue.Empty:
                    break

            latest_by_id: OrderedDict[str, ExtractionJob] = OrderedDict()
            for job in batch:
                if job.sentence.id in latest_by_id:
                    del latest_by_id[job.sentence.id]
                latest_by_id[job.sentence.id] = job
            try:
                for job in latest_by_id.values():
                    if self.stop_event.is_set():
                        return
                    self._process(job)
            finally:
                for _job in batch:
                    self.work_queue.task_done()

    def _process(self, job: ExtractionJob) -> None:
        sentence = job.sentence
        if not self.state.mark_checking(sentence, job.token):
            return
        self.state.record_llm_request()
        context = self.state.recent_sentences(self.context_size)
        try:
            payload = self.client.extract(sentence, context)
            result = validate_extraction_payload(payload, sentence)
        except Exception as exc:
            self.state.apply_extraction_failure(sentence, str(exc), job.token)
            return
        self.state.apply_extraction(sentence, result, job.token)


def run_whospeaks_reader(
    *,
    source_url: str,
    state: SidecarState,
    worker: ClaimExtractionWorker | None,
    stop_event: threading.Event,
    reconnect_seconds: float,
    timeout: float | None,
) -> None:
    url = f"{source_url.rstrip('/')}/api/events?snapshot=0"
    while not stop_event.is_set():
        try:
            state.set_source_status("connecting")
            request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                state.set_source_status("connected")
                for sse_event in parse_sse_lines(response):
                    if stop_event.is_set():
                        return
                    if sse_event.event != "transcript.final":
                        continue
                    event = json.loads(sse_event.data)
                    if not isinstance(event, dict):
                        continue
                    sentence = TranscriptSentence.from_public_event(event)
                    token = state.record_sentence(sentence, queue_claim_extraction=worker is not None)
                    if token is not None and worker is not None:
                        worker.submit(sentence, token)
        except Exception as exc:
            state.set_source_status("reconnecting", f"SSE connection failed: {exc}")
            stop_event.wait(reconnect_seconds)


def run_offline_demo(
    *,
    state: SidecarState,
    worker: ClaimExtractionWorker | None,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    examples = [
        ("demo-berlin", "Berlin has more than three million residents.", "Demo Speaker", 0.0, 4.0),
        ("demo-meeting", "The meeting starts at 14:30 in the main conference room.", "Demo Speaker", 4.5, 8.0),
        ("demo-release", "The first fact lens prototype reads transcript final events from WhoSpeaksLive.", "Demo Speaker", 8.5, 13.0),
    ]
    state.set_source_status("offline_demo")
    counter = 0
    while not stop_event.is_set():
        base_id, text, speaker, start, end = examples[counter % len(examples)]
        cycle = counter // len(examples)
        offset = cycle * 15.0
        sentence = TranscriptSentence(
            id=f"{base_id}-{cycle}",
            text=text,
            speaker=speaker,
            start=start + offset,
            end=end + offset,
        )
        token = state.record_sentence(sentence, queue_claim_extraction=worker is not None)
        if token is not None and worker is not None:
            worker.submit(sentence, token)
        counter += 1
        stop_event.wait(interval_seconds)


def make_handler(state: SidecarState, *, quiet: bool = False) -> type[BaseHTTPRequestHandler]:
    class FactLensHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_dashboard_html())
            elif parsed.path == "/api/state":
                self._send_json(state.snapshot())
            elif parsed.path == "/api/health":
                snapshot = state.snapshot()
                self._send_json(
                    {
                        "ok": True,
                        "source_status": snapshot["source_status"],
                        "llm_status": snapshot["llm_status"],
                        "revision": snapshot["revision"],
                    }
                )
            elif parsed.path == "/events":
                self._serve_events()
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            if not quiet:
                super().log_message(format, *args)

        def _serve_events(self) -> None:
            subscriber = state.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    try:
                        payload = subscriber.get(timeout=15)
                        message = f"event: state\ndata: {payload}\n\n"
                    except queue.Empty:
                        message = ": heartbeat\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            finally:
                state.unsubscribe(subscriber)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return FactLensHandler


def render_dashboard_html() -> str:
    return read_web_text("fact_lens/index.html")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live claim extraction sidecar for WhoSpeaksLive transcript.final events.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM claim extraction. Disabled by default; without this, the sidecar only displays final transcript sentences.",
    )
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-client", default="whospeaks-fact-lens")
    parser.add_argument("--llm-lane", default="shared")
    parser.add_argument("--llm-timeout", type=float, default=12.0)
    parser.add_argument("--llm-max-tokens", type=int, default=768)
    parser.add_argument("--schema-mode", choices=["json_schema", "response_format", "both"], default="both")
    parser.add_argument("--debounce-seconds", type=float, default=0.35)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--max-sentences", type=int, default=80)
    parser.add_argument("--max-cards", type=int, default=80)
    parser.add_argument("--sse-timeout", type=float, default=30.0)
    parser.add_argument("--reconnect-seconds", type=float, default=2.0)
    parser.add_argument("--offline-demo", action="store_true")
    parser.add_argument("--offline-interval-seconds", type=float, default=2.0)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stop_event = threading.Event()
    state = SidecarState(max_sentences=args.max_sentences, max_cards=args.max_cards)
    worker: ClaimExtractionWorker | None = None
    extraction_enabled = bool(args.enable_llm or args.mock_llm)
    if extraction_enabled:
        work_queue: queue.Queue[ExtractionJob] = queue.Queue(maxsize=args.queue_size)
        client = (
            MockClaimClient()
            if args.mock_llm
            else OpenAIClaimClient(
                base_url=args.llm_base_url,
                model=args.llm_model,
                client_name=args.llm_client,
                lane=args.llm_lane,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                schema_mode=args.schema_mode,
            )
        )
        worker = ClaimExtractionWorker(
            state=state,
            client=client,
            work_queue=work_queue,
            stop_event=stop_event,
            debounce_seconds=args.debounce_seconds,
            context_size=args.context_size,
        )
    else:
        state.set_llm_status("disabled")

    if args.offline_demo:
        reader = threading.Thread(
            target=run_offline_demo,
            kwargs={
                "state": state,
                "worker": worker,
                "stop_event": stop_event,
                "interval_seconds": args.offline_interval_seconds,
            },
            name="fact-lens-offline-demo",
            daemon=True,
        )
    else:
        reader = threading.Thread(
            target=run_whospeaks_reader,
            kwargs={
                "source_url": args.source_url,
                "state": state,
                "worker": worker,
                "stop_event": stop_event,
                "reconnect_seconds": args.reconnect_seconds,
                "timeout": args.sse_timeout,
            },
            name="fact-lens-whospeaks-reader",
            daemon=True,
        )
    runtime = FactLensRuntime(
        state=state,
        stop_event=stop_event,
        reader=reader,
        worker=worker,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, quiet=args.quiet))
    try:
        runtime.start()
        print(f"Fact Lens dashboard: http://{args.host}:{args.port}/", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close(
            reader_timeout=max(5.0, float(args.sse_timeout or 0.0) + 1.0),
            worker_timeout=max(5.0, float(args.llm_timeout or 0.0) + 1.0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
