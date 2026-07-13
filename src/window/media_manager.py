"""Atomic ownership of the live server's current media and version."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from window.window_domain import MediaFiles


@dataclass(frozen=True)
class MediaSnapshot:
    media: MediaFiles
    version: int


class MediaManager:
    """Serialize whole media transitions and publish immutable snapshots."""

    def __init__(self, media: MediaFiles, *, initial_version: int | None = None) -> None:
        self._transition_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._snapshot = MediaSnapshot(
            media=media,
            version=int(initial_version if initial_version is not None else time.time() * 1000),
        )

    def snapshot(self) -> MediaSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    def replace(self, media: MediaFiles, apply: Callable[[MediaFiles], None]) -> MediaSnapshot:
        with self._transition_lock:
            apply(media)
            return self._commit(media)

    def transition(self, apply: Callable[[], MediaFiles]) -> MediaSnapshot:
        with self._transition_lock:
            media = apply()
            return self._commit(media)

    def _commit(self, media: MediaFiles) -> MediaSnapshot:
        with self._snapshot_lock:
            self._snapshot = MediaSnapshot(media, self._snapshot.version + 1)
            return self._snapshot
