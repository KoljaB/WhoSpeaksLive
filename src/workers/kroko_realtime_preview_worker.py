"""Subprocess worker for Kroko/Banafo replace-only realtime preview text."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from workers.structured_realtime_result import structured_result_payload


KROKO_KEY_ENV_NAMES = (
    "REALTIMESTT_KROKO_ONNX_KEY",
    "KROKO_ONNX_KEY",
    "KROKO_KEY",
)
KROKO_REFERRAL_ENV_NAMES = (
    "REALTIMESTT_KROKO_ONNX_REFERRALCODE",
    "KROKO_ONNX_REFERRALCODE",
    "KROKO_REFERRALCODE",
)


def write_message(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def first_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="kroko_onnx")
    parser.add_argument("--model", default="Kroko-EN-Community-64-L-Streaming-001.data")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--download-root", default="")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--language", default="en")
    parser.add_argument("--engine-options-json", default="")
    parser.add_argument("--realtimestt-root", default="")
    return parser.parse_args()


def create_sessions(args: argparse.Namespace):
    if args.realtimestt_root:
        root = Path(args.realtimestt_root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
    from RealtimeSTT.transcription_engines.factory import create_transcription_engine

    os.environ["KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT"] = "1"
    options = {
        "provider": args.provider,
        "num_threads": args.num_threads,
        "sample_rate": 16000,
        "language": args.language,
        "suppress_native_output": True,
    }
    if args.model_path:
        options["model_path"] = args.model_path
    options["key"] = first_env_value(KROKO_KEY_ENV_NAMES)
    options["referralcode"] = first_env_value(KROKO_REFERRAL_ENV_NAMES)
    if args.engine_options_json:
        extra_options = json.loads(args.engine_options_json)
        if not isinstance(extra_options, dict):
            raise ValueError("--engine-options-json must be a JSON object.")
        options.update(extra_options)

    model = args.model_path or args.model
    config = TranscriptionEngineConfig(
        model=model,
        download_root=args.download_root or None,
        device=args.provider,
        normalize_audio=False,
        engine_options=options,
    )
    engine = create_transcription_engine(args.engine, config)
    return (
        engine.create_streaming_session(language=args.language, use_prompt=False),
        engine.create_streaming_session(language=args.language, use_prompt=False),
    )


def native_session_result(session: object) -> object | None:
    backend = getattr(session, "backend", None)
    recognizer = getattr(backend, "recognizer", None)
    stream = getattr(session, "stream", None)
    get_result_all = getattr(recognizer, "get_result_all", None)
    if callable(get_result_all) and stream is not None:
        return get_result_all(stream)
    return None


def main() -> int:
    args = parse_args()
    try:
        session, verification_session = create_sessions(args)
    except Exception as exc:
        write_message({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    write_message({"ready": True})
    for line in sys.stdin:
        request: Any = {}
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = str(request.get("command") or "transcribe").strip().lower()
            if command == "reset":
                session.reset()
                write_message({"id": request_id, "ok": True, "text": ""})
                continue
            if command not in {"accept", "transcribe", "transcribe_final", "verify"}:
                raise ValueError(f"Unknown command: {command}")

            audio = np.frombuffer(base64.b64decode(request["audio_b64"]), dtype=np.float32).copy()
            sample_rate = int(request.get("sample_rate") or 16000)
            active_session = verification_session if command == "verify" else session
            if command in {"transcribe", "transcribe_final"}:
                active_session.reset()
            if command == "verify":
                active_session.reset()
            active_session.accept_audio(audio, sample_rate=sample_rate)
            result = active_session.finish() if command == "transcribe_final" else None
            if result is None:
                active_session.decode()
                result = active_session.get_result()
            if command == "transcribe_final":
                native = native_session_result(active_session)
                payload = structured_result_payload(native, fallback_text=str(result.text or ""))
                write_message({"id": request_id, **payload})
            else:
                write_message({"id": request_id, "text": (result.text or "").strip()})
        except Exception as exc:
            write_message({"id": request.get("id") if isinstance(request, dict) else None, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
