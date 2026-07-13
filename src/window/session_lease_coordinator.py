"""Thread and side-effect ownership for live session leases."""

from __future__ import annotations

import threading
from typing import Any, Callable

from window.session_lease import SessionLeaseStateMachine


class SessionLeaseCoordinator:
    """Translate lease transitions into exactly-once controller actions."""

    def __init__(
        self,
        lease: SessionLeaseStateMachine,
        *,
        controller_is_running: Callable[[], bool],
        controller_stop: Callable[[], None],
        publish_status: Callable[[str], None],
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.lease = lease
        self._controller_is_running = controller_is_running
        self._controller_stop = controller_stop
        self._publish_status = publish_status
        self._poll_interval = max(0.01, float(poll_interval_seconds))
        self._lifecycle_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_token = ""
        self._monitor_thread: threading.Thread | None = None
        self._closed = False

    def enforce_timeouts(self) -> dict[str, object] | None:
        expired = self.lease.expire_if_needed()
        if not expired:
            return None
        reason = str(expired.get("reason") or "expired")
        self._publish_status(f"Demo seat released ({reason}).")
        if expired.get("was_running"):
            self._controller_stop()
        return expired

    def status(self, client_id: str = "") -> dict[str, object]:
        self.enforce_timeouts()
        return self.lease.status(client_id)

    def acquire(self, client_id: str) -> dict[str, object]:
        self.enforce_timeouts()
        result = self.lease.acquire(client_id)
        if result.get("acquired"):
            self._publish_status("Demo seat acquired.")
        return result

    def authorize(self, token: str, client_id: str = "") -> dict[str, object]:
        self.enforce_timeouts()
        return self.lease.authorize(token, client_id)

    def heartbeat(self, token: str, client_id: str = "") -> dict[str, object]:
        self.enforce_timeouts()
        return self.lease.heartbeat(token, client_id)

    def release(self, token: str, reason: str = "released", client_id: str = "") -> dict[str, object]:
        was_running = self.lease.is_active_token(token) and self._controller_is_running()
        result = self.lease.release(token, reason, client_id)
        if result.get("released"):
            self._publish_status(f"Demo seat released ({reason}).")
            if was_running:
                self._controller_stop()
        return result

    def mark_running(self, token: str) -> None:
        self.lease.mark_running(token)
        self._ensure_monitor(token)

    def _ensure_monitor(self, token: str) -> None:
        clean_token = str(token or "").strip()
        with self._lifecycle_lock:
            if self._closed:
                return
            current = self._monitor_thread
            if current is not None and current.is_alive() and self._monitor_token == clean_token:
                return
            self._monitor_token = clean_token
            self._monitor_stop.clear()
            thread = threading.Thread(
                target=self._monitor,
                args=(clean_token, self._monitor_stop),
                name="SessionLeaseMonitor",
                daemon=True,
            )
            self._monitor_thread = thread
            thread.start()

    def _monitor(self, token: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(self._poll_interval):
            if not self.lease.is_active_token(token):
                return
            if self.enforce_timeouts():
                return
            if self._controller_is_running():
                continue
            if self.lease.mark_completed(token):
                delay = self.lease.completed_release_delay_seconds
                self._publish_status(f"Run finished; demo seat will release in {delay:.0f}s.")
            while self.lease.is_active_token(token) and not stop_event.wait(self._poll_interval):
                if self.enforce_timeouts():
                    return
            return

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._monitor_stop.set()
            thread = self._monitor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
