"""Media cache and URL resolution for the window diarization GUI."""

from __future__ import annotations

import argparse
import re
from urllib.parse import urlparse

from window_domain import MediaFiles
from youtube_local_filefeed_replay import (
    download_audio,
    download_video,
    newest_matching_file,
    resolve_video_id,
)


def media_cache_status(args: argparse.Namespace, url: str) -> tuple[str, bool, bool]:
    video_id = resolve_video_id(url, None)
    cached_audio = newest_matching_file(args.work_dir, [f"{video_id}.audio.mp3", f"{video_id}.audio.*"])
    cached_video = newest_matching_file(args.work_dir, [f"{video_id}.video.*"])
    return video_id, cached_audio is not None, cached_video is not None


def resolve_media(args: argparse.Namespace) -> MediaFiles:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    video_id = resolve_video_id(args.url, args.audio_file)
    if args.audio_file is not None:
        audio_file = args.audio_file.resolve()
    else:
        audio_file = newest_matching_file(args.work_dir, [f"{video_id}.audio.mp3", f"{video_id}.audio.*"])
        if audio_file is None:
            if args.skip_download:
                raise RuntimeError("Missing cached audio. Run without --skip-download or pass --audio-file.")
            audio_file = download_audio(args.url, video_id, args.work_dir, args.yt_dlp)
    if args.video_file is not None:
        video_file = args.video_file.resolve()
    else:
        video_file = newest_matching_file(args.work_dir, [f"{video_id}.video.*"])
        if video_file is None:
            if getattr(args, "validate_window_replay", False):
                video_file = audio_file
            elif args.skip_download:
                raise RuntimeError("Missing cached video. Run without --skip-download or pass --video-file.")
            else:
                video_file = download_video(args.url, video_id, args.work_dir, args.yt_dlp)
    return MediaFiles(args.url, video_id, audio_file.resolve(), video_file.resolve())


def resolve_media_url(args: argparse.Namespace, url: str, skip_download: bool = False) -> MediaFiles:
    media_args = argparse.Namespace(**vars(args))
    media_args.url = url
    media_args.audio_file = None
    media_args.video_file = None
    media_args.skip_download = skip_download
    return resolve_media(media_args)


def resolve_browser_stream_id(url: str) -> str:
    parsed = urlparse(str(url or ""))
    if parsed.scheme in {"microphone", "system-audio"}:
        return parsed.scheme
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raw = parsed.scheme or parsed.netloc or parsed.path or "browser-stream"
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
        return cleaned or "browser-stream"
    try:
        video_id = resolve_video_id(url, None)
        if video_id != "local-filefeed":
            return video_id
    except Exception:
        pass
    raw = parsed.netloc or parsed.path or "browser-stream"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
    return cleaned or "browser-stream"
