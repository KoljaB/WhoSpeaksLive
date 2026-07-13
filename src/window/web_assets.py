"""Whitelisted packaged web resources and safe bootstrap rendering."""

from __future__ import annotations

from importlib import resources
import json
import mimetypes
from pathlib import PurePosixPath
from typing import Any


_WEB_ROOT = ("assets", "web")
_ALLOWED_ASSETS = frozenset({
    "live/index.html",
    "live/styles.css",
    "live/app.js",
    "live/app_store.js",
    "live/live_context.js",
    "live/media_capture.js",
    "live/session_transport.js",
    "live/saved_reports.js",
    "live/meeting_chat.js",
    "live/transcript_translation.js",
    "live/transcript_review.js",
    "live/speaker_panel.js",
    "live/transcript_render.js",
    "live/live_bindings.js",
    "reports/index.html",
    "reports/styles-base.css",
    "reports/styles-components.css",
    "reports/report_builder.js",
    "reports/app.js",
    "fact_lens/index.html",
})


def read_web_asset(name: str) -> bytes:
    normalized = PurePosixPath(str(name).replace("\\", "/")).as_posix().lstrip("/")
    if normalized not in _ALLOWED_ASSETS:
        raise FileNotFoundError(normalized)
    resource = resources.files("window")
    for part in (*_WEB_ROOT, *PurePosixPath(normalized).parts):
        resource = resource.joinpath(part)
    return resource.read_bytes()


def read_web_text(name: str) -> str:
    return read_web_asset(name).decode("utf-8")


def web_asset_content_type(name: str) -> str:
    if str(name).endswith(".js"):
        return "text/javascript"
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def render_live_index(bootstrap: dict[str, Any]) -> str:
    serialized = json.dumps(bootstrap, ensure_ascii=False, separators=(",", ":"))
    # JSON lives in a non-executable script node; escaping angle brackets
    # prevents data from closing that node and becoming markup.
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return read_web_text("live/index.html").replace("__BOOTSTRAP_JSON__", serialized)
