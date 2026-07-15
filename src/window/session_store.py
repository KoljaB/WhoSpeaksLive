"""Durable saved-session storage for the window diarization UI."""

from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from paths import RUNTIME_DIR
from window.meeting_intelligence import (
    generate_meeting_report,
    mark_report_stale_if_needed,
    update_report_object,
)
from window.review_flags import annotate_review


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


def _with_review(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    if bool(payload.get("pending")) or bool(payload.get("realtime")):
        return payload
    payload["review"] = annotate_review(payload)
    return payload


def _empty_meeting_intelligence_doc(updated_at: str) -> dict[str, Any]:
    return {
        "version": SESSION_FORMAT_VERSION,
        "updated_at": updated_at,
        "report": None,
    }


def _speaker_display_name(speaker_id: str, name: str = "") -> str:
    if name:
        return name
    match = re.fullmatch(r"S(\d+)", speaker_id)
    return f"Speaker {int(match.group(1))}" if match else speaker_id


class SessionStore:
    """Stores reopenable session snapshots under one local runtime directory."""

    def __init__(self, root: Path = DEFAULT_SESSION_DIR) -> None:
        self.root = Path(root).expanduser().resolve()
        self._mutation_lock = threading.RLock()

    @property
    def mutation_lock(self) -> threading.RLock:
        """Saved identity transactions acquire this before the People lock."""

        return self._mutation_lock

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

    def _meeting_intelligence_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "meeting_intelligence.json"

    def _translations_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "translations.json"

    @staticmethod
    def _meeting_intelligence_doc(report: dict[str, Any] | None, updated_at: str) -> dict[str, Any]:
        doc = _empty_meeting_intelligence_doc(updated_at)
        doc["report"] = report if isinstance(report, dict) and report else None
        return doc

    @staticmethod
    def _manifest_with_meeting_intelligence(
        manifest: dict[str, Any],
        report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        next_manifest = dict(manifest)
        paths = dict(next_manifest.get("paths") or {})
        paths.setdefault("transcript", "transcript.json")
        paths.setdefault("speakers", "speakers.json")
        paths.setdefault("embeddings", "embeddings.json")
        paths["meeting_intelligence"] = "meeting_intelligence.json"
        next_manifest["paths"] = paths
        has_report = isinstance(report, dict) and bool(report)
        next_manifest["has_meeting_intelligence"] = has_report
        next_manifest["meeting_intelligence_status"] = str(report.get("status") or "") if has_report else ""
        return next_manifest

    @staticmethod
    def _manifest_with_translations(
        manifest: dict[str, Any],
        translations: list[Any],
    ) -> dict[str, Any]:
        next_manifest = dict(manifest)
        paths = dict(next_manifest.get("paths") or {})
        paths.setdefault("transcript", "transcript.json")
        paths.setdefault("speakers", "speakers.json")
        paths.setdefault("embeddings", "embeddings.json")
        paths.setdefault("meeting_intelligence", "meeting_intelligence.json")
        paths["translations"] = "translations.json"
        next_manifest["paths"] = paths
        next_manifest["has_translations"] = bool(translations)
        next_manifest["translation_count"] = len(translations)
        return next_manifest

    def _read_translations(self, session_id: str) -> list[Any]:
        path = self._translations_path(session_id)
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        if isinstance(value, list):
            return value
        if not isinstance(value, dict):
            return []
        translations = value.get("translations")
        return translations if isinstance(translations, list) else []

    def _read_meeting_intelligence_report(self, session_id: str) -> dict[str, Any] | None:
        doc = self._read_json(self._meeting_intelligence_path(session_id), {"report": None})
        report = doc.get("report")
        return report if isinstance(report, dict) and report else None

    def _write_meeting_intelligence_report(
        self,
        session_id: str,
        report: dict[str, Any] | None,
        updated_at: str,
    ) -> None:
        self._write_json(
            self._meeting_intelligence_path(session_id),
            self._meeting_intelligence_doc(report, updated_at),
        )

    def _mark_meeting_intelligence_stale_for_rows(
        self,
        session_id: str,
        rows: list[dict[str, Any]],
        speaker_state: dict[str, Any],
        updated_at: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        report = self._read_meeting_intelligence_report(session_id)
        if report is None:
            if not self._meeting_intelligence_path(session_id).is_file():
                self._write_meeting_intelligence_report(session_id, None, updated_at)
            return None, False
        report, changed = mark_report_stale_if_needed(
            report,
            transcript_rows=rows,
            speaker_state=speaker_state,
            updated_at=updated_at,
        )
        if changed:
            self._write_meeting_intelligence_report(session_id, report, updated_at)
        return report, changed

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
        self._write_json(session_dir / "meeting_intelligence.json", _empty_meeting_intelligence_doc(now))
        self._write_json(session_dir / "translations.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "translations": [],
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
            "has_translations": False,
            "translation_count": 0,
            "has_meeting_intelligence": False,
            "meeting_intelligence_status": "",
            "paths": {
                "transcript": "transcript.json",
                "speakers": "speakers.json",
                "embeddings": "embeddings.json",
                "meeting_intelligence": "meeting_intelligence.json",
                "translations": "translations.json",
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
            "has_translations": bool(manifest.get("has_translations")),
            "translation_count": int(manifest.get("translation_count") or 0),
            "has_meeting_intelligence": bool(manifest.get("has_meeting_intelligence")),
            "meeting_intelligence_status": str(manifest.get("meeting_intelligence_status") or ""),
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
        with self._mutation_lock:
            return self._save_snapshot_locked(
                snapshot,
                status_label=status_label,
                write_audio=write_audio,
                audio_writer=audio_writer,
            )

    def _save_snapshot_locked(
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

        rows = _json_ready([_with_review(dict(row)) for row in (snapshot.get("transcript_rows") or [])])
        speaker_state = _json_ready(snapshot.get("speaker_state") or {})
        source = _json_ready(snapshot.get("source") or {})
        speaker_profiles = _json_ready(snapshot.get("speaker_profiles") or [])
        live_speaker_profiles = _json_ready(snapshot.get("live_speaker_profiles") or [])
        snapshot_translations = snapshot.get("translations")
        translations = _json_ready(snapshot_translations if isinstance(snapshot_translations, list) else [])
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
        self._write_json(session_dir / "translations.json", {
            "version": SESSION_FORMAT_VERSION,
            "updated_at": now,
            "translations": translations,
        })
        meeting_intelligence_report, _meeting_intelligence_changed = self._mark_meeting_intelligence_stale_for_rows(
            session_id,
            rows,
            speaker_state if isinstance(speaker_state, dict) else {},
            now,
        )

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
            "has_translations": bool(translations),
            "translation_count": len(translations),
            "has_meeting_intelligence": bool(meeting_intelligence_report),
            "meeting_intelligence_status": str(meeting_intelligence_report.get("status") or "") if meeting_intelligence_report else "",
            "paths": {
                "transcript": "transcript.json",
                "speakers": "speakers.json",
                "embeddings": "embeddings.json",
                "meeting_intelligence": "meeting_intelligence.json",
                "translations": "translations.json",
            },
        }
        manifest = self._manifest_with_meeting_intelligence(manifest, meeting_intelligence_report)
        manifest = self._manifest_with_translations(manifest, translations)
        self._write_json(session_dir / "manifest.json", manifest)
        return self._summary_from_manifest(manifest)

    def open_session(self, session_id: str) -> dict[str, Any]:
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        embeddings = self._read_json(session_dir / "embeddings.json", {"records": []})
        translations = self._read_translations(session_id)
        rows = [_with_review(dict(row)) for row in (transcript.get("rows") or [])]
        speaker_state = speakers.get("speaker_state") if isinstance(speakers.get("speaker_state"), dict) else {}
        meeting_intelligence_report, meeting_intelligence_changed = self._mark_meeting_intelligence_stale_for_rows(
            session_id,
            rows,
            speaker_state,
            _now_iso(),
        )
        manifest_with_meeting_intelligence = self._manifest_with_meeting_intelligence(
            manifest,
            meeting_intelligence_report,
        )
        manifest_with_translations = self._manifest_with_translations(
            manifest_with_meeting_intelligence,
            translations,
        )
        if meeting_intelligence_changed or manifest_with_translations != manifest:
            self._write_json(session_dir / "manifest.json", manifest_with_translations)
            manifest = manifest_with_translations
        public_manifest = copy.deepcopy(manifest)
        public_source = public_manifest.get("source") if isinstance(public_manifest.get("source"), dict) else {}
        for key in ("audio_path", "video_path", "local_path", "path"):
            public_source.pop(key, None)
        public_audio = public_manifest.get("audio") if isinstance(public_manifest.get("audio"), dict) else {}
        public_audio.pop("path", None)
        public_speaker_state = copy.deepcopy(speaker_state)
        for speaker in public_speaker_state.get("speakers") or []:
            if isinstance(speaker, dict):
                speaker.pop("reference_audio", None)
        public_profiles = []
        for profile in speakers.get("speaker_profiles") or []:
            if not isinstance(profile, dict):
                continue
            public_profiles.append({
                key: copy.deepcopy(value)
                for key, value in profile.items()
                if key not in {"centroid", "centroid_b64", "embedding", "embedding_b64", "reference_audio"}
            })
        public_rows = []
        for row in rows:
            public_rows.append({
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key not in {"embedding", "embedding_b64", "centroid", "centroid_b64", "reference_audio"}
            })
        return {
            "summary": self._summary_from_manifest(manifest),
            "manifest": public_manifest,
            "transcript_rows": public_rows,
            "speaker_state": public_speaker_state,
            "speaker_profiles": public_profiles,
            "live_speaker_profiles": [],
            "embedding_count": len(embeddings.get("records") or []),
            "embeddings_available": bool(embeddings.get("records")),
            "translations": translations,
            "meeting_intelligence": {
                "available": bool(meeting_intelligence_report),
                "report": meeting_intelligence_report,
            },
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
        with self._mutation_lock:
            return self._rename_speaker_locked(session_id, speaker_id, name)

    def _rename_speaker_locked(self, session_id: str, speaker_id: str, name: str) -> dict[str, Any]:
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
        meeting_intelligence_report, _meeting_intelligence_changed = self._mark_meeting_intelligence_stale_for_rows(
            session_id,
            [_with_review(dict(row)) for row in (transcript_doc.get("rows") or [])],
            speaker_state,
            now,
        )
        manifest = self._manifest_with_meeting_intelligence(manifest, meeting_intelligence_report)
        self._write_json(session_dir / "speakers.json", speakers_doc)
        self._write_json(session_dir / "transcript.json", transcript_doc)
        self._write_json(session_dir / "manifest.json", manifest)
        return self.open_session(session_id)

    def reassign_rows(self, session_id: str, indexes: list[int], speaker_id: str) -> dict[str, Any]:
        with self._mutation_lock:
            return self._correct_saved_rows(session_id, indexes, speaker_id=str(speaker_id or "").strip())

    def mark_rows_correct(self, session_id: str, indexes: list[int]) -> dict[str, Any]:
        with self._mutation_lock:
            return self._correct_saved_rows(session_id, indexes, speaker_id=None)

    def _correct_saved_rows(
        self,
        session_id: str,
        indexes: list[int],
        *,
        speaker_id: str | None,
    ) -> dict[str, Any]:
        normalized_indexes = sorted({int(index) for index in indexes})
        if not normalized_indexes:
            raise ValueError("Choose at least one transcript row.")

        session_id = self._validate_session_id(session_id)
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript_doc = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers_doc = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        embeddings_doc = self._read_json(session_dir / "embeddings.json", {"records": []})
        rows = [row for row in (transcript_doc.get("rows") or []) if isinstance(row, dict)]
        rows_by_index: dict[int, dict[str, Any]] = {}
        for position, row in enumerate(rows):
            try:
                row_index = int(row.get("index"))
            except (TypeError, ValueError):
                row_index = position
                row["index"] = row_index
            rows_by_index[row_index] = row
        missing = [index for index in normalized_indexes if index not in rows_by_index]
        if missing:
            raise ValueError(f"Unknown transcript row {missing[0]}.")

        speaker_state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), dict) else {}
        speakers = [speaker for speaker in (speaker_state.get("speakers") or []) if isinstance(speaker, dict)]
        target_name = ""
        if speaker_id is not None:
            target = next((speaker for speaker in speakers if str(speaker.get("id") or "") == speaker_id), None)
            if target is None:
                raise ValueError(f"Unknown speaker {speaker_id}.")
            target_name = str(target.get("display_name") or target.get("name") or _speaker_display_name(speaker_id))

        now = _now_iso()
        for index in normalized_indexes:
            row = rows_by_index[index]
            current_speaker = str(row.get("assigned_speaker") or "")
            if speaker_id is None:
                row["correction"] = {
                    "status": "user_confirmed",
                    "action": "mark_correct",
                    "original_speaker": row.get("automatic_assigned_speaker", current_speaker),
                    "corrected_speaker": current_speaker or None,
                    "corrected_at": now,
                    "updates_memory": False,
                }
            else:
                row.setdefault("automatic_assigned_speaker", current_speaker or None)
                row.setdefault("automatic_assignment_source", str(row.get("assignment_source") or ""))
                rejected = {
                    str(value)
                    for value in (row.get("correction") or {}).get("rejected_speakers", [])
                    if str(value)
                } if isinstance(row.get("correction"), dict) else set()
                if current_speaker and current_speaker != speaker_id:
                    rejected.add(current_speaker)
                rejected.discard(speaker_id)
                row["assigned_speaker"] = speaker_id
                row["speaker_name"] = target_name
                row["assignment_source"] = "user_correction"
                row["correction"] = {
                    "status": "user_corrected",
                    "action": "reassign",
                    "original_speaker": row.get("automatic_assigned_speaker"),
                    "previous_speaker": current_speaker or None,
                    "corrected_speaker": speaker_id,
                    "rejected_speakers": sorted(rejected),
                    "corrected_at": now,
                    "updates_memory": False,
                }
            row["review"] = annotate_review(row)

        if speaker_id is not None:
            for record in embeddings_doc.get("records") or []:
                if not isinstance(record, dict):
                    continue
                try:
                    record_index = int(record.get("index"))
                except (TypeError, ValueError):
                    continue
                if record_index in normalized_indexes:
                    record["assigned_speaker"] = speaker_id

        counts: dict[str, int] = {}
        speaking_seconds: dict[str, float] = {}
        for row in rows:
            assigned = str(row.get("assigned_speaker") or "")
            if not assigned:
                continue
            counts[assigned] = counts.get(assigned, 0) + 1
            try:
                duration = max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
            except (TypeError, ValueError):
                duration = 0.0
            speaking_seconds[assigned] = speaking_seconds.get(assigned, 0.0) + duration
        for speaker in speakers:
            identity = str(speaker.get("id") or "")
            speaker["sentence_count"] = counts.get(identity, 0)
            speaker["speech_seconds"] = round(speaking_seconds.get(identity, 0.0), 4)

        transcript_doc["rows"] = rows
        transcript_doc["updated_at"] = now
        speakers_doc["updated_at"] = now
        embeddings_doc["updated_at"] = now
        manifest["updated_at"] = now
        manifest["status_label"] = "Saved"
        meeting_intelligence_report, _changed = self._mark_meeting_intelligence_stale_for_rows(
            session_id,
            [_with_review(dict(row)) for row in rows],
            speaker_state,
            now,
        )
        manifest = self._manifest_with_meeting_intelligence(manifest, meeting_intelligence_report)
        self._write_json(session_dir / "transcript.json", transcript_doc)
        self._write_json(session_dir / "speakers.json", speakers_doc)
        self._write_json(session_dir / "embeddings.json", embeddings_doc)
        self._write_json(session_dir / "manifest.json", manifest)
        return self.open_session(session_id)

    def meeting_intelligence(self, session_id: str) -> dict[str, Any]:
        session_id = self._validate_session_id(session_id)
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript_doc = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers_doc = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        rows = [_with_review(dict(row)) for row in (transcript_doc.get("rows") or [])]
        speaker_state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), dict) else {}
        report, changed = self._mark_meeting_intelligence_stale_for_rows(session_id, rows, speaker_state, _now_iso())
        manifest_with_meeting_intelligence = self._manifest_with_meeting_intelligence(manifest, report)
        if changed or manifest_with_meeting_intelligence != manifest:
            self._write_json(session_dir / "manifest.json", manifest_with_meeting_intelligence)
        return {
            "available": bool(report),
            "report": report,
        }

    def generate_meeting_intelligence(self, session_id: str) -> dict[str, Any]:
        session_id = self._validate_session_id(session_id)
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript_doc = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers_doc = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        rows = [_with_review(dict(row)) for row in (transcript_doc.get("rows") or [])]
        speaker_state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), dict) else {}
        report = generate_meeting_report(
            session_id=session_id,
            transcript_rows=rows,
            speaker_state=speaker_state,
        )
        now = _now_iso()
        self._write_meeting_intelligence_report(session_id, report, now)
        manifest["updated_at"] = now
        manifest["status_label"] = "Saved"
        manifest = self._manifest_with_meeting_intelligence(manifest, report)
        self._write_json(session_dir / "manifest.json", manifest)
        return {
            "available": True,
            "report": report,
        }

    def update_meeting_intelligence_object(
        self,
        session_id: str,
        object_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> dict[str, Any]:
        session_id = self._validate_session_id(session_id)
        session_dir = self._session_dir(session_id)
        manifest = self._load_manifest(session_id)
        transcript_doc = self._read_json(session_dir / "transcript.json", {"rows": []})
        speakers_doc = self._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        rows = [_with_review(dict(row)) for row in (transcript_doc.get("rows") or [])]
        speaker_state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), dict) else {}
        report, _changed = self._mark_meeting_intelligence_stale_for_rows(session_id, rows, speaker_state, _now_iso())
        if report is None:
            raise ValueError("Generate meeting intelligence before updating objects.")
        report = update_report_object(
            report,
            object_id=object_id,
            status=status,
            title=title,
            body=body,
        )
        now = _now_iso()
        self._write_meeting_intelligence_report(session_id, report, now)
        manifest["updated_at"] = now
        manifest["status_label"] = "Saved"
        manifest = self._manifest_with_meeting_intelligence(manifest, report)
        self._write_json(session_dir / "manifest.json", manifest)
        return {
            "available": True,
            "report": report,
        }
