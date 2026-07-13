"""Thin entrypoint for the browser-synced growing-window application."""

from __future__ import annotations

from datetime import datetime
import sys
import webbrowser

from window.live_http_handler import Handler, parse_range_header
from window.live_window_server import WindowServer
from window.window_cli import WindowConfig, parse_args
from window.window_diarizer import WindowDiarizer
from window.window_events import EventBus
from window.window_media import resolve_media
from window.window_validation import (
    build_window_validation_records,
    ratio_summary,
    run_window_replay_validation,
)


def _configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


_configure_console_output()


def main() -> int:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing command line.", flush=True)
    args = parse_args()
    preview_model_location = args.realtime_preview_model_dir or args.realtime_preview_model_path
    preview_model_display = preview_model_location.name if preview_model_location is not None else args.realtime_preview_model
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Startup config: "
        f"url={args.url} port={args.port} asr_backend={args.asr_backend} "
        f"language={args.language} sentence_tokenizer={args.sentence_tokenizer}/{args.sentence_language} "
        f"embeddings_backend={args.embeddings_backend} embedding_provider={args.embedding_provider} "
        f"embedding_timeout={args.embedding_helper_response_timeout_seconds:.0f}s "
        f"realtime_preview={args.realtime_preview_engine} "
        f"preview_model={args.realtime_preview_model_preset}:{preview_model_display} "
        f"translation={args.translation_provider}:{args.translation_model_profile} "
        f"translation_targets={','.join(args.translation_target_language) or '-' }.",
        flush=True,
    )
    if args.validate_window_replay:
        return run_window_replay_validation(args)
    media = resolve_media(args)
    bus = EventBus()
    controller = WindowDiarizer(args, media, bus)
    server: WindowServer | None = None
    try:
        server = WindowServer((args.host, args.port), args, media, bus, controller)
        if args.startup_warmup_before_url:
            controller.prepare_before_browser_release()
        else:
            bus.emit("status", {"message": "Startup model warmup skipped; models will warm before playback."})
        page_url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
        print(f"Serving growing-window diarization GUI at {page_url}", flush=True)
        if not args.no_browser:
            webbrowser.open(page_url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.", flush=True)
    finally:
        controller.shutdown()
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
