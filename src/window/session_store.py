"""Durable saved-session storage for the window diarization UI."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from paths import RUNTIME_DIR


DEFAULT_SESSION_DIR = RUNTIME_DIR / "sessions"
SESSION_FORMAT_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _clean_title(value: Any) -> str:
    title = " ".join(str(value or "").strip().split())
    return title[:120]


def _short_datetime_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f"{parsed.strftime('%b')} {parsed.day} {parsed.strftime('%H:%M')}"


def _with_time_suffix(title: str, time_label: str) -> str:
    if not time_label:
        return _clean_title(title)
    return _clean_title(f"{title} - {time_label}")


def _video_id_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").replace("www.", "", 1).lower()
    if host == "youtu.be":
        return _clean_title(parsed.path.strip("/").split("/")[0])
    if host.endswith("youtube.com"):
        values = parse_qs(parsed.query).get("v") or []
        if values:
            return _clean_title(values[0])
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return _clean_title(parts[1])
    return ""


def _title_from_source(source: dict[str, Any], session_id: str, started_at: str = "") -> str:
    time_label = _short_datetime_label(source.get("started_at") or started_at)
    media_title = _clean_title(source.get("video_title") or source.get("media_title"))
    if media_title:
        return media_title
    explicit = _clean_title(source.get("title"))
    if explicit:
        if explicit in {
            "Microphone recording",
            "Browser audio recording",
            "Computer audio + microphone recording",
        }:
            return _with_time_suffix(explicit, time_label)
        return explicit
    capture_mode = str(source.get("capture_mode") or "").strip().lower()
    url = str(source.get("url") or "").strip()
    video_id = _clean_title(source.get("video_id")) or _video_id_from_url(url)
    if capture_mode == "youtube" and video_id:
        return _with_time_suffix(f"YouTube {video_id}", time_label)
    if url:
        parsed = urlparse(url)
        if parsed.scheme in {"microphone", "browser-stream", "mixed-audio"}:
            base = {
                "microphone": "Microphone recording",
                "browser-stream": "Browser audio recording",
                "mixed-audio": "Computer audio + microphone recording",
            }.get(parsed.scheme, "Audio recording")
            return _with_time_suffix(base, time_label)
        if parsed.hostname:
            return parsed.hostname.replace("www.", "", 1)
        return url[:120]
    if video_id:
        return _with_time_suffix(video_id, time_label)
    return f"WhoSpeaks session {session_id[:8]}"


def _duration_from_rows(rows: list[dict[str, Any]]) -> float:
    duration = 0.0
    for row in rows:
        try:
            duration = max(duration, float(row.get("end") or 0.0))
        except (TypeError, ValueError):
            continue
    return round(duration, 4)


def _speaker_display_name(speaker_id: str, name: str = "") -> str:
    if name:
        return name
    match = re.fullmatch(r"S(\d+)", speaker_id)
    return f"Speaker {int(match.group(1))}" if match else speaker_id


class SessionStore:
    """Stores reopenable session snapshots under one local runtime directory."""

    def __init__(self, root: Path = DEFAULT_SESSION_DIR) -> None:
        self.root = Path(root).expanduser().resolve()

    def _session_dir(self, session_id: str) -> Path:
        normalized = self._validate_session_id(session_id)
        return self.root / normalized

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        normalized = str(session_id or "").strip()
        if not SESSION_ID_PATTERN.fullmatch(normalized):
            raise ValueError("Invalid session id.")
        return normalized

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{datetime.now().timestamp():.6f}.tmp")
        temp_path.write_text(
            json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _read_json(path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        if not path.is_file():
            return dict(fallback or {})
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else dict(fallback or {})

    @staticmethod
    def _embedding_payload(record: dict[str, Any]) -> dict[str, Any] | None:
        embedding = record.get("embedding")
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype="<f4").reshape(-1)
        if vector.size <= 0:
            return None
        payload = {
            "index": int(record.get("index") or 0),
            "duration_seconds": float(record.get("duration_seconds") or 0.0),
            "assigned_speaker": record.get("assigned_speaker"),
            "embedding_encoding": "float32-base64-le",
            "embedding_b64": base64.b64encode(vector.tobytes()).decode("ascii"),
            "embedding_length": int(vector.size),
        }
        return payload

    def _load_manifest(self, session_id: str) -> dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._read_json(session_dir / "manifest.json")
        if not manifest:
            raise FileNotFoundError(f"Session {session_id} does not exist.")
        return manifest

    def create_session(
        self,
        *,
        source: dict[str, Any] | None = None,
        title: str = "",
        session_id: str = "",
        status_label: str = "New",
    ) -> dict[str, Any]:
        if session_id:
            normalized_id = self._validate_session_id(session_id)
            if (self._session_dir(normalized_id) / "manifest.json").is_file():
                raise FileExistsError(f"Session {normalized_id} already exists.")
        else:
            while True:
                normalized_id = uuid.uuid4().hex
                if not (self._session_dir(normalized_id) / "manifest.json").exists():
                    break

        now = _now_iso()
        ready_source = _json_ready(source or {})
        started_at = str(ready_source.get("started_at") or now)
        ready_source["started_at"] = started_at
        clean_title = _clean_title(title) or _title_from_source(ready_source, normalized_id, started_at)
        session_dir = self._session_dir(normalized_id)
        self._write_json(session_dir / "transcript.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "rows": [],
        })
        self._write_json(session_dir / "speakers.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "speaker_state": {"group_name": "", "groups": [], "speakers": []},
            "speaker_profiles": [],
            "live_speaker_profiles": [],
        })
        self._write_json(session_dir / "embeddings.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "embedding_provider": "",
            "live_embedding_provider": "",
            "records": [],
        })
        manifest = {
            "version": SESSION_FORMAT_VERSION,
            "id": normalized_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
            "started_at": started_at,
            "ended_at": started_at,
            "archived": False,
            "status_label": _clean_title(status_label) or "New",
            "source": ready_source,
            "audio": {},
            "audio_error": "",
            "duration_seconds": 0.0,
            "speaker_count": 0,
            "speaker_names": [],
            "transcript_rows": 0,
            "has_audio": False,
            "has_audio_reference": False,
            "has_transcript": False,
            "has_speakers": False,
            "has_embeddings": False,
            "paths": {
                "transcript": "transcript.json",
                "speakers": "speakers.json",
                "embeddings": "embeddings.json",
            },
        }
        self._write_json(session_dir / "manifest.json", manifest)
        return self._summary_from_manifest(manifest)

    def _summary_from_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(manifest.get("id") or ""),
            "title": str(manifest.get("title") or ""),
            "created_at": str(manifest.get("created_at") or ""),
            "updated_at": str(manifest.get("updated_at") or ""),
            "started_at": str(manifest.get("started_at") or manifest.get("created_at") or ""),
            "ended_at": str(manifest.get("ended_at") or manifest.get("updated_at") or ""),
            "archived": bool(manifest.get("archived")),
            "duration_seconds": float(manifest.get("duration_seconds") or 0.0),
            "source": manifest.get("source") if isinstance(manifest.get("source"), dict) else {},
            "speaker_count": int(manifest.get("speaker_count") or 0),
            "speaker_names": list(manifest.get("speaker_names") or []),
            "transcript_rows": int(manifest.get("transcript_rows") or 0),
            "status_label": str(manifest.get("status_label") or "Saved"),
            "has_audio": bool(manifest.get("has_audio")),
            "has_audio_reference": bool(manifest.get("has_audio_reference")),
            "has_transcript": bool(manifest.get("has_transcript")),
            "has_speakers": bool(manifest.get("has_speakers")),
            "has_embeddings": bool(manifest.get("has_embeddings")),
        }

    def list_sessions(self, filter_mode: str = "active", query: str = "") -> list[dict[str, Any]]:
        mode = str(filter_mode or "active").strip().lower()
        if mode not in {"active", "archived", "all"}:
            mode = "active"
        query_terms = [term for term in str(query or "").strip().lower().split() if term]

        sessions: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                manifest = self._read_json(child / "manifest.json")
            except (OSError, json.JSONDecodeError):
                continue
            if not manifest:
                continue
            archived = bool(manifest.get("archived"))
            if mode == "active" and archived:
                continue
            if mode == "archived" and not archived:
                continue
            searchable = " ".join([
                str(manifest.get("title") or ""),
                str(manifest.get("created_at") or ""),
                str(manifest.get("updated_at") or ""),
                " ".join(str(name) for name in manifest.get("speaker_names") or []),
            ]).lower()
            if query_terms and not all(term in searchable for term in query_terms):
                continue
            sessions.append(self._summary_from_manifest(manifest))

        sessions.sort(key=lambda item: (item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return sessions

    def save_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        status_label: str = "Autosaved",
        write_audio: bool = False,
        audio_writer: Callable[[Path], bool] | None = None,
    ) -> dict[str, Any]:
        session_id = self._validate_session_id(str(snapshot.get("id") or snapshot.get("session_id") or ""))
        session_dir = self._session_dir(session_id)
        existing = self._read_json(session_dir / "manifest.json")
        now = _now_iso()

        rows = _json_ready(list(snapshot.get("transcript_rows") or []))
        speaker_state = _json_ready(snapshot.get("speaker_state") or {})
        source = _json_ready(snapshot.get("source") or {})
        speaker_profiles = _json_ready(snapshot.get("speaker_profiles") or [])
        live_speaker_profiles = _json_ready(snapshot.get("live_speaker_profiles") or [])
        embedding_records = [
            payload
            for payload in (self._embedding_payload(dict(record)) for record in (snapshot.get("embedding_records") or []))
            if payload is not None
        ]
        duration_seconds = float(snapshot.get("duration_seconds") or 0.0)
        if duration_seconds <= 0.0:
            duration_seconds = _duration_from_rows(rows)

        self._write_json(session_dir / "transcript.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "rows": rows,
        })
        self._write_json(session_dir / "speakers.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "speaker_state": speaker_state,
            "speaker_profiles": speaker_profiles,
            "live_speaker_profiles": live_speaker_profiles,
        })
        self._write_json(session_dir / "embeddings.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "embedding_provider": str(snapshot.get("embedding_provider") or ""),
            "live_embedding_provider": str(snapshot.get("live_embedding_provider") or ""),
            "records": embedding_records,
        })

        audio = dict(existing.get("audio") or {})
        audio_error = ""
        if write_audio and bool(source.get("streaming_audio")) and audio_writer is not None:
            target = session_dir / "audio.wav"
            try:
                if audio_writer(target):
                    audio = {
                        "kind": "managed_wav",
                        "path": str(target),
                        "relative_path": "audio.wav",
                        "sample_rate": int(source.get("audio_sample_rate") or 16000),
                        "saved": True,
                    }
            except Exception as exc:
                audio_error = f"{type(exc).__name__}: {exc}"
        elif not audio:
            audio_path = str(source.get("audio_path") or "")
            if audio_path:
                audio = {
                    "kind": "reference",
                    "path": audio_path,
                    "saved": Path(audio_path).is_file(),
                }

        speakers = speaker_state.get("speakers") if isinstance(speaker_state, dict) else []
        speaker_names = [
            str(speaker.get("display_name") or speaker.get("name") or speaker.get("id") or "")
            for speaker in speakers
            if isinstance(speaker, dict)
        ]
        created_at = str(existing.get("created_at") or snapshot.get("created_at") or source.get("started_at") or now)
        started_at = str(existing.get("started_at") or source.get("started_at") or snapshot.get("started_at") or created_at)
        ended_at = str(snapshot.get("ended_at") or source.get("ended_at") or now)
        title = _clean_title(existing.get("title")) or _title_from_source(source, session_id, started_at)
        manifest = {
            "version": SESSION_FORMAT_VERSION,
            "id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": now,
            "started_at": started_at,
            "ended_at": ended_at,
            "archived": bool(existing.get("archived")),
            "status_label": _clean_title(status_label) or "Saved",
            "source": source,
            "audio": audio,
            "audio_error": audio_error,
            "duration_seconds": round(float(duration_seconds), 4),
            "speaker_count": len(speaker_names),
            "speaker_names": speaker_names,
            "transcript_rows": len(rows),
            "has_audio": bool(audio.get("saved")),
            "has_audio_reference": bool(audio.get("path")),
            "has_transcript": bool(rows),
            "has_speakers": bool(speaker_names),
            "has_embeddings": bool(embedding_records),
            "paths": {
                "transcript": "transcript.json",
                "speakers": "speakers.json",
                "embeddings": "embeddings.json",
            },
        }
        self._write_json(session_dir / "manifest.json", manifest)
        return self._summary_from_manifest(manifest)

    def open_session(self, session_id: str) -> dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        embeddings = self._read_json(session_dir / "embeddings.json", {"records": []})
        return {
            "summary": self._summary_from_manifest(manifest),
            "manifest": manifest,
            "transcript_rows": list(transcript.get("rows") or []),
            "speaker_state": speakers.get("speaker_state") if isinstance(speakers.get("speaker_state"), dict) else {},
            "speaker_profiles": list(speakers.get("speaker_profiles") or []),
            "live_speaker_profiles": list(speakers.get("live_speaker_profiles") or []),
            "embedding_count": len(embeddings.get("records") or []),
            "embeddings_available": bool(embeddings.get("records")),
        }

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        manifest = self._load_manifest(session_id)
        clean_title = _clean_title(title)
        if not clean_title:
            raise ValueError("Session title must not be empty.")
        manifest["title"] = clean_title
        manifest["updated_at"] = _now_iso()
        manifest["status_label"] = "Saved"
        self._write_json(self._session_dir(session_id) / "manifest.json", manifest)
        return self._summary_from_manifest(manifest)

    def archive_session(self, session_id: str, archived: bool = True) -> dict[str, Any]:
        manifest = self._load_manifest(session_id)
        manifest["archived"] = bool(archived)
        manifest["updated_at"] = _now_iso()
        manifest["status_label"] = "Saved"
        self._write_json(self._session_dir(session_id) / "manifest.json", manifest)
        return self._summary_from_manifest(manifest)

    def restore_session(self, session_id: str) -> dict[str, Any]:
        return self.archive_session(session_id, archived=False)

    def delete_session(self, session_id: str) -> dict[str, Any]:
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session {session_id} does not exist.")
        shutil.rmtree(session_dir)
        return {"id": self._validate_session_id(session_id), "deleted": True}

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> dict[str, Any]:
        speaker_id = str(speaker_id or "").strip()
        if not re.fullmatch(r"S\d+", speaker_id):
            raise ValueError("Invalid speaker id.")
        clean_name = _clean_title(name)
        if not clean_name:
            raise ValueError("Speaker name must not be empty.")

        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        speakers_doc = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        transcript_doc = self._read_json(session_dir / "transcript.json", {"rows": []})
        speaker_state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), dict) else {}
        speakers = speaker_state.get("speakers") if isinstance(speaker_state.get("speakers"), list) else []
        speaker_profiles = speakers_doc.get("speaker_profiles") if isinstance(speakers_doc.get("speaker_profiles"), list) else []
        matched = False
        for speaker in speakers:
            if not isinstance(speaker, dict) or str(speaker.get("id") or "") != speaker_id:
                continue
            speaker["name"] = clean_name
            speaker["display_name"] = _speaker_display_name(speaker_id, clean_name)
            matched = True
        for profile in speaker_profiles:
            if not isinstance(profile, dict) or str(profile.get("label") or "") != speaker_id:
                continue
            profile["name"] = clean_name
            profile["display_name"] = _speaker_display_name(speaker_id, clean_name)
        if not matched:
            raise ValueError(f"Unknown speaker {speaker_id}.")

        for row in transcript_doc.get("rows") or []:
            if isinstance(row, dict) and str(row.get("assigned_speaker") or "") == speaker_id:
                row["speaker_name"] = clean_name

        now = _now_iso()
        speakers_doc["updated_at"] = now
        transcript_doc["updated_at"] = now
        manifest["updated_at"] = now
        manifest["status_label"] = "Saved"
        manifest["speaker_names"] = [
            str(speaker.get("display_name") or speaker.get("name") or speaker.get("id") or "")
            for speaker in speakers
            if isinstance(speaker, dict)
        ]
        self._write_json(session_dir / "speakers.json", speakers_doc)
        self._write_json(session_dir / "transcript.json", transcript_doc)
        self._write_json(session_dir / "manifest.json", manifest)
        return self.open_session(session_id)
