"""Synchronized atomic persistence for generated report cache entries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable


class ReportCache:
    def __init__(self, directory: Path, *, hash_fn: Callable[..., str]) -> None:
        self.directory = Path(directory)
        self._hash_fn = hash_fn
        self._locks_guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.RLock] = {}

    def legacy_path(self, session_id: str) -> Path:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-") or "session"
        return self.directory / f"{clean[:80]}-{self._hash_fn(session_id, length=10)}.json"

    def path(self, session_id: str, template_id: str) -> Path:
        clean_session = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-") or "session"
        clean_template = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(template_id or "").strip()).strip("-") or "template"
        return self.directory / (
            f"{clean_session[:64]}-{self._hash_fn(session_id, length=10)}--"
            f"{clean_template[:64]}-{self._hash_fn(template_id, length=10)}.json"
        )

    def read(self, session_id: str, template_id: str, *, legacy_template_id: str) -> dict[str, Any] | None:
        with self._lock_for(session_id, template_id):
            path = self.path(session_id, template_id)
            if not path.is_file() and template_id == legacy_template_id:
                path = self.legacy_path(session_id)
            if not path.is_file():
                return None
            try:
                serialized = path.read_text(encoding="utf-8")
            except OSError:
                return None
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError:
            return None
        report = payload.get("report") if isinstance(payload, dict) else None
        return dict(report) if isinstance(report, dict) else None

    def write(self, session_id: str, template_id: str, report: dict[str, Any]) -> None:
        payload = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "template_id": template_id,
            "report": report,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._lock_for(session_id, template_id):
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", prefix=".meeting-report-",
                    suffix=".tmp", dir=self.directory, delete=False,
                ) as temporary:
                    temporary.write(serialized)
                    temporary_name = temporary.name
                os.replace(temporary_name, self.path(session_id, template_id))
                temporary_name = None
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)

    def delete(self, session_id: str, template_id: str, *, legacy_template_id: str) -> bool:
        with self._lock_for(session_id, template_id):
            deleted = False
            path = self.path(session_id, template_id)
            if path.is_file():
                path.unlink()
                deleted = True
            legacy = self.legacy_path(session_id)
            if template_id == legacy_template_id and legacy.is_file():
                legacy.unlink()
                deleted = True
            return deleted

    def _lock_for(self, session_id: str, template_id: str) -> threading.RLock:
        key = (str(session_id), str(template_id))
        with self._locks_guard:
            return self._locks.setdefault(key, threading.RLock())


__all__ = ["ReportCache"]
