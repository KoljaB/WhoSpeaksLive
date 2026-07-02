"""Download/cache YouTube audio and run the current RealtimeSTT file-feed replay.

This is the clean local-audio path: it does not use browser playback or WASAPI.
It feeds the extracted MP3 into AudioToTextRecorder.feed_audio through
whospeaks-realtime --validate-cunk-realtime-replay, so the result
is directly comparable to the ElevenLabs canonical baseline.

For a browser UI with synced local video playback, use whospeaks-window.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import subprocess
import sys
from pathlib import Path

from whospeaks.paths import (
    CACHE_DIR,
    CUNK_CANONICAL,
    EMBEDDING_VENV,
    LOCAL_FILEFEED_MEDIA_DIR,
    MAIN_VENV,
    PROJECT_ROOT,
    REALTIME_VALIDATION_OUTPUT_DIR,
    TOOLS_DIR,
    VENVS_DIR,
)
from whospeaks.realtime.realtime_speakerdiarize import extract_youtube_video_id


DEFAULT_URL = "https://www.youtube.com/watch?v=JWS-qfR6K3w"
DEFAULT_WORK_DIR = LOCAL_FILEFEED_MEDIA_DIR
DEFAULT_OUTPUT_DIR = REALTIME_VALIDATION_OUTPUT_DIR
DEFAULT_CANONICAL = CUNK_CANONICAL
DEFAULT_EMBEDDING_PYTHON = EMBEDDING_VENV / "Scripts" / "python.exe"
DEFAULT_FAST_WHISPER_CACHE = CACHE_DIR / "faster-whisper"
DEFAULT_FAST_WHISPER_LARGE_V2 = (
    DEFAULT_FAST_WHISPER_CACHE
    / "models--Systran--faster-whisper-large-v2"
    / "snapshots"
    / "f0fe81560cb8b68660e564f55dd99207059c092e"
)
DEFAULT_FAST_WHISPER_TINY_EN = (
    DEFAULT_FAST_WHISPER_CACHE
    / "models--Systran--faster-whisper-tiny.en"
    / "snapshots"
    / "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
)
KNOWN_YT_DLP_EXES = [
    MAIN_VENV / "Scripts" / "yt-dlp.exe",
    VENVS_DIR / "install-matrix" / "all" / "Scripts" / "yt-dlp.exe",
]


def default_download_root() -> Path | None:
    return DEFAULT_FAST_WHISPER_CACHE if DEFAULT_FAST_WHISPER_CACHE.exists() else None


def default_realtimestt_model() -> str:
    return str(DEFAULT_FAST_WHISPER_LARGE_V2) if DEFAULT_FAST_WHISPER_LARGE_V2.exists() else "large-v2"


def default_realtimestt_rt_model() -> str:
    return str(DEFAULT_FAST_WHISPER_TINY_EN) if DEFAULT_FAST_WHISPER_TINY_EN.exists() else "tiny.en"


def quote_command_part(value: object) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def format_command(command: list[object]) -> str:
    return " ".join(quote_command_part(part) for part in command)


def newest_matching_file(directory: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def yt_dlp_runners(yt_dlp_path: Path | None) -> list[list[str]]:
    runners: list[list[str]] = []
    if yt_dlp_path is not None:
        runners.append([str(yt_dlp_path)])
    for candidate in KNOWN_YT_DLP_EXES:
        if candidate.exists():
            runners.append([str(candidate)])
    runners.extend([
        [sys.executable, "-m", "yt_dlp"],
        ["yt-dlp"],
    ])
    deduped: list[list[str]] = []
    seen: set[str] = set()
    for runner in runners:
        key = "\0".join(runner).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(runner)
    return deduped


def run_yt_dlp(command_args: list[str], yt_dlp_path: Path | None) -> None:
    runners = yt_dlp_runners(yt_dlp_path)
    errors: list[str] = []
    for runner in runners:
        command = runner + command_args
        print("Running " + format_command(command), flush=True)
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode == 0:
            return
        errors.append(f"{command[0]} exited with {completed.returncode}")
    raise RuntimeError("yt-dlp failed. " + " | ".join(errors))


def download_audio(url: str, video_id: str, work_dir: Path, yt_dlp_path: Path | None) -> Path:
    output_template = str(work_dir / f"{video_id}.audio.%(ext)s")
    run_yt_dlp([
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        url,
    ], yt_dlp_path)
    audio_file = newest_matching_file(work_dir, [f"{video_id}.audio.mp3", f"{video_id}.audio.*"])
    if audio_file is None:
        raise RuntimeError("yt-dlp finished but no MP3 file was found.")
    return audio_file


def download_video(url: str, video_id: str, work_dir: Path, yt_dlp_path: Path | None) -> Path:
    output_template = str(work_dir / f"{video_id}.video.%(ext)s")
    run_yt_dlp([
        "--no-playlist",
        "-f",
        "bestvideo[ext=mp4][vcodec^=avc1]/bestvideo[ext=mp4]/bestvideo",
        "-o",
        output_template,
        url,
    ], yt_dlp_path)
    video_file = newest_matching_file(work_dir, [f"{video_id}.video.*"])
    if video_file is None:
        raise RuntimeError("yt-dlp finished but no video file was found.")
    return video_file


def resolve_video_id(url: str, audio_file: Path | None) -> str:
    try:
        return extract_youtube_video_id(url)
    except Exception:
        if audio_file is not None:
            return audio_file.stem
        return "local-filefeed"


def resolve_audio_file(args: argparse.Namespace, video_id: str) -> Path:
    if args.audio_file is not None:
        audio_file = args.audio_file.resolve()
        if not audio_file.exists():
            raise RuntimeError(f"Audio file does not exist: {audio_file}")
        return audio_file

    cached = newest_matching_file(args.work_dir, [f"{video_id}.audio.mp3", f"{video_id}.audio.*"])
    if cached is not None:
        return cached.resolve()
    if args.skip_download:
        raise RuntimeError("Missing cached MP3. Run without --skip-download or pass --audio-file.")
    return download_audio(args.url, video_id, args.work_dir, args.yt_dlp).resolve()


def maybe_resolve_video_file(args: argparse.Namespace, video_id: str) -> Path | None:
    if args.video_file is not None:
        video_file = args.video_file.resolve()
        if not video_file.exists():
            raise RuntimeError(f"Video file does not exist: {video_file}")
        return video_file
    if not args.download_video:
        return newest_matching_file(args.work_dir, [f"{video_id}.video.*"])
    cached = newest_matching_file(args.work_dir, [f"{video_id}.video.*"])
    if cached is not None:
        return cached.resolve()
    if args.skip_download:
        raise RuntimeError("Missing cached video. Run without --skip-download or omit --download-video.")
    return download_video(args.url, video_id, args.work_dir, args.yt_dlp).resolve()


def build_replay_command(
    args: argparse.Namespace,
    passthrough_args: list[str],
    audio_file: Path,
    trace_log: Path,
    analysis_output: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(TOOLS_DIR / "realtime_speakerdiarize.py"),
        "--validate-cunk-realtime-replay",
        "--model",
        str(args.model),
        "--rt-model",
        str(args.rt_model),
        "--validation-audio",
        str(audio_file),
        "--validation-canonical",
        str(args.validation_canonical),
        "--validation-output",
        str(analysis_output),
        "--trace-log",
        str(trace_log),
        "--replay-speed",
        str(args.replay_speed),
        "--replay-chunk-seconds",
        str(args.replay_chunk_seconds),
        "--replay-trailing-silence-seconds",
        str(args.replay_trailing_silence_seconds),
        "--replay-drain-seconds",
        str(args.replay_drain_seconds),
        "--replay-embedding-drain-seconds",
        str(args.replay_embedding_drain_seconds),
        "--embedding-python",
        str(args.embedding_python),
    ]
    if args.download_root is not None:
        command.extend(["--download-root", str(args.download_root)])
    if args.no_replay_sleep:
        command.append("--no-replay-sleep")
    command.extend(passthrough_args)
    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Download/cache a YouTube MP3 and feed it directly into the current "
            "RealtimeSTT replay validation path."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--video-file", type=Path, default=None)
    parser.add_argument("--download-video", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--yt-dlp", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", default=default_realtimestt_model())
    parser.add_argument("--rt-model", default=default_realtimestt_rt_model())
    parser.add_argument("--download-root", type=Path, default=default_download_root())
    parser.add_argument("--embedding-python", type=Path, default=DEFAULT_EMBEDDING_PYTHON)
    parser.add_argument("--validation-canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--trace-log", type=Path, default=None)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-chunk-seconds", type=float, default=0.1)
    parser.add_argument("--replay-trailing-silence-seconds", type=float, default=2.0)
    parser.add_argument("--replay-drain-seconds", type=float, default=25.0)
    parser.add_argument("--replay-embedding-drain-seconds", type=float, default=15.0)
    parser.add_argument("--no-replay-sleep", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args, passthrough_args = parser.parse_known_args()
    args.work_dir = args.work_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.validation_canonical = args.validation_canonical.resolve()
    args.embedding_python = args.embedding_python.resolve()
    args.python = args.python.resolve()
    if args.download_root is not None:
        args.download_root = args.download_root.resolve()
    if args.yt_dlp is not None:
        args.yt_dlp = args.yt_dlp.resolve()
    return args, passthrough_args


def main() -> int:
    args, passthrough_args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_id = resolve_video_id(args.url, args.audio_file)
    audio_file = resolve_audio_file(args, video_id)
    video_file = maybe_resolve_video_file(args, video_id)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_log = (
        args.trace_log.resolve()
        if args.trace_log is not None
        else args.output_dir / f"local-filefeed-{video_id}-{stamp}.trace.jsonl"
    )
    analysis_output = (
        args.validation_output.resolve()
        if args.validation_output is not None
        else args.output_dir / f"local-filefeed-{video_id}-{stamp}.analysis.json"
    )
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)

    command = build_replay_command(
        args=args,
        passthrough_args=passthrough_args,
        audio_file=audio_file,
        trace_log=trace_log,
        analysis_output=analysis_output,
    )
    print(f"Audio feed source: {audio_file}", flush=True)
    if video_file is not None:
        print(f"Cached video: {video_file}", flush=True)
    print("Replay command:", flush=True)
    print(format_command(command), flush=True)
    if args.no_run:
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
