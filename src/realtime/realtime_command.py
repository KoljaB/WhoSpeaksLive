"""Command dispatch and GUI startup for realtime diarization."""

from __future__ import annotations

import webbrowser
from datetime import datetime

from embeddings.embedding_providers import run_embedding_helper
from paths import OUTPUTS_DIR
from realtime.offline_validation import validate_cunk, validate_cunk_word_splits
from realtime.realtime_capture import EventBus, TraceLogger
from realtime.realtime_cli import RealtimeConfig, parse_args
from realtime.realtime_server import GuiServer
from realtime.replay_validation import validate_cunk_realtime_replay
from realtime.trace_commands import analyze_trace


def run_realtime_command(args: RealtimeConfig) -> int:
    if args.embedding_helper:
        return run_embedding_helper(args)
    if args.analyze_trace is not None:
        return analyze_trace(
            args.analyze_trace,
            canonical_path=args.validation_canonical,
            output_path=args.trace_analysis_output,
            summary_only=args.trace_summary_only,
            match_mode=args.trace_match_mode,
            trace_session=args.trace_session,
        )
    if args.validate_cunk_realtime_replay:
        return validate_cunk_realtime_replay(args)
    if args.validate_cunk_word_splits:
        return validate_cunk_word_splits(args)
    if args.validate_cunk:
        return validate_cunk(args)

    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            OUTPUTS_DIR
            / "realtime-speakerdiarize-traces"
            / f"trace-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    server = GuiServer((args.host, args.port), args, bus, trace)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Serving realtime speaker diarization GUI at {url}", flush=True)
    print(f"Trace log: {trace.path}", flush=True)
    print(f"Embedding helper: {args.embedding_python}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.controller.shutdown()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_realtime_command(parse_args(argv))
