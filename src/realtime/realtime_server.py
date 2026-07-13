"""HTTP server for the realtime diarization GUI."""

from __future__ import annotations

import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from common.audio_utils import json_dumps
from realtime.realtime_capture import EventBus, RealtimeCapture, TraceLogger
from realtime.realtime_cli import RealtimeConfig
from realtime.realtime_gui_html import HTML

class RequestHandler(BaseHTTPRequestHandler):
    server: "GuiServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/events":
            self._serve_events()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                url = str(payload.get("url") or "").strip()
                if not url:
                    raise ValueError("Missing YouTube URL.")
                session_id, video_id = self.server.controller.start(url)
                self._send_json({"session_id": session_id, "video_id": video_id})
                return
            if self.path == "/api/video-playing":
                self.server.controller.mark_video_playing(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/video-time":
                self.server.controller.update_video_time(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/debug-log":
                self.server.trace.write(
                    "frontend",
                    str(payload.get("stage") or "debug"),
                    payload,
                )
                self._send_json({"ok": True})
                return
            if self.path == "/api/stop":
                self.server.controller.stop()
                self._send_json({"ok": True})
                return
            self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json_dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        subscriber = self.server.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event, payload = subscriber.get(timeout=10)
                    message = f"event: {event}\ndata: {payload}\n\n"
                except queue.Empty:
                    message = ": heartbeat\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self.server.bus.unsubscribe(subscriber)


class GuiServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        args: RealtimeConfig,
        bus: EventBus,
        trace: TraceLogger,
    ) -> None:
        super().__init__(address, RequestHandler)
        self.args = args
        self.bus = bus
        self.trace = trace
        self.controller = RealtimeCapture(args, bus)


