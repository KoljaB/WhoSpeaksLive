"""Session-seat state machine with deterministic, injectable time ownership."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable
import uuid


@dataclass(frozen=True)
class LeaseExpiration:
    reason: str
    was_running: bool


class SessionLeaseError(RuntimeError):
    def __init__(self, message: str, session: dict[str, object], status: int = 409) -> None:
        super().__init__(message)
        self.session = session
        self.status = status


class SessionLeaseStateMachine:
    """The sole writer of browser-seat identity and timeout state."""

    def __init__(
        self,
        idle_timeout_seconds: float = 120.0,
        heartbeat_timeout_seconds: float = 45.0,
        completed_release_delay_seconds: float = 10.0,
        max_run_seconds: float = 900.0,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.idle_timeout_seconds = max(1.0, float(idle_timeout_seconds))
        self.heartbeat_timeout_seconds = max(5.0, float(heartbeat_timeout_seconds))
        self.completed_release_delay_seconds = max(0.0, float(completed_release_delay_seconds))
        self.max_run_seconds = max(30.0, float(max_run_seconds))
        self._clock = monotonic
        self._token_factory = token_factory
        self._lock = threading.Lock()
        self._token = ""
        self._client_id = ""
        self._created_at = 0.0
        self._last_seen_at = 0.0
        self._run_started_at: float | None = None
        self._completed_at: float | None = None
        self._last_release_reason = ""
        self._last_release_at = 0.0
        self._waiting_clients: dict[str, float] = {}

    def status(self, client_id: str = "") -> dict[str, object]:
        self.expire_if_needed()
        now = self._clock()
        with self._lock:
            return self._snapshot_locked(now, client_id)

    def acquire(self, client_id: str) -> dict[str, object]:
        self.expire_if_needed()
        now = self._clock()
        client = str(client_id or "").strip()[:120] or self._token_factory()
        with self._lock:
            if self._token and self._client_id != client:
                self._waiting_clients[client] = now
                return {"ok": False, "acquired": False, "session": self._snapshot_locked(now, client)}
            if not self._token:
                self._token = self._token_factory()
                self._client_id = client
                self._created_at = now
                self._run_started_at = None
                self._completed_at = None
            self._last_seen_at = now
            self._waiting_clients.pop(client, None)
            return {
                "ok": True,
                "acquired": True,
                "session_token": self._token,
                "session": self._snapshot_locked(now, client),
            }

    def authorize(self, token: str, client_id: str = "") -> dict[str, object]:
        expired = self.expire_if_needed()
        now = self._clock()
        supplied = str(token or "").strip()
        client = str(client_id or "").strip()[:120]
        with self._lock:
            if not self._token:
                raise SessionLeaseError("Take the demo seat first.", self._snapshot_locked(now, client))
            if supplied != self._token:
                if client:
                    self._waiting_clients[client] = now
                message = "Session in use. Watching live; controls are disabled until the seat is free."
                if expired:
                    message = "The previous session expired; try taking the seat again."
                raise SessionLeaseError(message, self._snapshot_locked(now, client))
            self._last_seen_at = now
            if client:
                self._client_id = client
                self._waiting_clients.pop(client, None)
            return self._snapshot_locked(now, client)

    def heartbeat(self, token: str, client_id: str = "") -> dict[str, object]:
        self.authorize(token, client_id)
        return {"ok": True, "session": self.status(client_id)}

    def release(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, object]:
        self.expire_if_needed()
        now = self._clock()
        supplied = str(token or "").strip()
        client = str(client_id or "").strip()[:120]
        with self._lock:
            released = bool(self._token and supplied == self._token)
            if released:
                self._release_locked(now, reason or "released")
            elif self._token and client and client != self._client_id:
                self._waiting_clients[client] = now
            return {"ok": True, "released": released, "session": self._snapshot_locked(now, client)}

    def mark_running(self, token: str) -> None:
        now = self._clock()
        with self._lock:
            if self._token and str(token or "").strip() == self._token:
                self._run_started_at = now
                self._completed_at = None
                self._last_seen_at = now

    def mark_completed(self, token: str) -> bool:
        now = self._clock()
        with self._lock:
            if not self._token or str(token or "").strip() != self._token:
                return False
            self._completed_at = now
            return True

    def is_active_token(self, token: str) -> bool:
        with self._lock:
            return bool(self._token) and str(token or "").strip() == self._token

    def expire_if_needed(self) -> dict[str, object] | None:
        now = self._clock()
        with self._lock:
            reason = self._expired_reason_locked(now)
            if not reason:
                return None
            expiration = LeaseExpiration(
                reason=reason,
                was_running=self._run_started_at is not None and self._completed_at is None,
            )
            self._release_locked(now, reason)
            return {"reason": expiration.reason, "was_running": expiration.was_running}

    def _expired_reason_locked(self, now: float) -> str:
        if not self._token:
            return ""
        if now - self._last_seen_at > self.heartbeat_timeout_seconds:
            return "heartbeat timeout"
        if self._completed_at is not None and now - self._completed_at > self.completed_release_delay_seconds:
            return "completed"
        if self._run_started_at is None and now - self._created_at > self.idle_timeout_seconds:
            return "idle timeout"
        if self._run_started_at is not None and now - self._run_started_at > self.max_run_seconds:
            return "time limit"
        return ""

    def _release_locked(self, now: float, reason: str) -> None:
        self._token = ""
        self._client_id = ""
        self._created_at = 0.0
        self._last_seen_at = 0.0
        self._run_started_at = None
        self._completed_at = None
        self._last_release_reason = reason
        self._last_release_at = now

    def _snapshot_locked(self, now: float, client_id: str = "") -> dict[str, object]:
        self._prune_waiters_locked(now)
        active = bool(self._token)
        heartbeat = self.heartbeat_timeout_seconds - (now - self._last_seen_at) if active else None
        idle = completed = hard = expires = None
        release_reason = ""
        if active and self._completed_at is not None:
            completed = self.completed_release_delay_seconds - (now - self._completed_at)
            expires, release_reason = completed, "completed"
        elif active and self._run_started_at is None:
            idle = self.idle_timeout_seconds - (now - self._created_at)
            expires, release_reason = min(idle, heartbeat), "idle"
        elif active:
            hard = self.max_run_seconds - (now - self._run_started_at)
            expires, release_reason = min(hard, heartbeat), "timeout"

        remaining = lambda value: round(max(0.0, value), 1) if value is not None else None
        return {
            "active": active,
            "is_owner": active and bool(client_id) and client_id == self._client_id,
            "running": active and self._run_started_at is not None and self._completed_at is None,
            "completed": active and self._completed_at is not None,
            "client_id": self._client_id if active else "",
            "waiter_count": len(self._waiting_clients),
            "expires_in_seconds": remaining(expires),
            "heartbeat_expires_in_seconds": remaining(heartbeat),
            "idle_expires_in_seconds": remaining(idle),
            "completed_expires_in_seconds": remaining(completed),
            "hard_expires_in_seconds": remaining(hard),
            "release_reason": release_reason,
            "last_release_reason": self._last_release_reason,
            "last_release_at": round(self._last_release_at, 3) if self._last_release_at else None,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "completed_release_delay_seconds": self.completed_release_delay_seconds,
            "max_run_seconds": self.max_run_seconds,
        }

    def _prune_waiters_locked(self, now: float) -> None:
        stale_before = now - 120.0
        self._waiting_clients = {
            client_id: seen_at
            for client_id, seen_at in self._waiting_clients.items()
            if seen_at >= stale_before
        }


# Stable name used by the HTTP application and existing imports.
SessionLease = SessionLeaseStateMachine
