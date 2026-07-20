"""Helpers for contracts spanning the packaged live-page resources."""

from window.web_assets import read_web_text


LIVE_ASSETS = (
    "live/index.html",
    "live/styles.css",
    "live/app.js",
    "live/app_store.js",
    "live/live_context.js",
    "live/help_system.js",
    "live/media_capture.js",
    "live/session_transport.js",
    "live/saved_reports.js",
    "live/transcript_translation.js",
    "live/transcript_review.js",
    "live/speaker_panel.js",
    "live/transcript_render.js",
    "live/live_bindings.js",
)


HTML = "\n".join(read_web_text(name) for name in LIVE_ASSETS)
