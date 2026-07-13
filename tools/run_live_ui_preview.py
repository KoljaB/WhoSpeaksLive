"""Serve the packaged live browser UI with small deterministic API responses."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.web_assets import read_web_asset, render_live_index, web_asset_content_type


def make_handler() -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = render_live_index({
                    "source": "UI preview",
                    "language": {"name": "English", "code": "en", "flag_url": ""},
                    "meeting_intelligence": {"enabled": True},
                }).encode("utf-8")
                self._send(body, "text/html; charset=utf-8")
                return
            if path.startswith("/assets/web/"):
                name = path.removeprefix("/assets/web/")
                try:
                    body = read_web_asset(name)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send(body, web_asset_content_type(name))
                return
            if path == "/api/events":
                self._json({"events": [], "cursor": 0})
                return
            if path == "/api/sessions":
                self._json({"ok": True, "sessions": []})
                return
            if path == "/api/meeting-intelligence/status":
                self._json({"ok": True, "enabled": True, "ready": True})
                return
            self._json({"ok": True})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/meeting-intelligence/chat/scope":
                self._json({
                    "ok": True,
                    "scope_id": "meetings_preview",
                    "session_ids": ["preview-meeting"],
                    "history": [],
                    "meetings": [{"id": "preview-meeting", "title": "Quarterly planning"}],
                    "index": {"configured": True, "sessions": [{
                        "session_id": "preview-meeting", "indexed": True,
                        "current_embedding_model": True, "current_revision": True,
                    }]},
                    "requires_index": False,
                    "provisional": False,
                })
                return
            self._json({"ok": True})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, payload: dict[str, object]) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PreviewHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18976)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"Live UI preview at http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
