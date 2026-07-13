"""Atomic transaction ownership for mutable diarization-session aggregates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Iterator


@dataclass(frozen=True)
class SessionVersion:
    value: int


class DiarizationSession:
    """Own the one re-entrant lock guarding the aggregate session state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._version = 0

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def version(self) -> SessionVersion:
        with self._lock:
            return SessionVersion(self._version)

    @contextmanager
    def transaction(self, *, mutate: bool = False) -> Iterator[SessionVersion]:
        with self._lock:
            before = SessionVersion(self._version)
            yield before
            if mutate:
                self._version += 1

    def is_current(self, expected: SessionVersion) -> bool:
        with self._lock:
            return self._version == expected.value
