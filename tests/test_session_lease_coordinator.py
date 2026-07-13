from __future__ import annotations

import threading
import time
import unittest

from window.session_lease import SessionLeaseStateMachine
from window.session_lease_coordinator import SessionLeaseCoordinator


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class SessionLeaseCoordinatorTests(unittest.TestCase):
    def test_expired_running_lease_stops_controller_exactly_once(self) -> None:
        clock = _Clock()
        lease = SessionLeaseStateMachine(monotonic=clock, token_factory=lambda: "token")
        stops: list[str] = []
        coordinator = SessionLeaseCoordinator(
            lease,
            controller_is_running=lambda: True,
            controller_stop=lambda: stops.append("stop"),
            publish_status=lambda _message: None,
        )
        lease.acquire("client")
        lease.mark_running("token")
        clock.value = 901.0

        coordinator.enforce_timeouts()
        coordinator.enforce_timeouts()

        self.assertEqual(stops, ["stop"])
        coordinator.close()

    def test_natural_completion_is_marked_and_coordinator_closes_idempotently(self) -> None:
        running = True
        statuses: list[str] = []
        lease = SessionLeaseStateMachine(
            completed_release_delay_seconds=60.0,
            token_factory=lambda: "token",
        )
        coordinator = SessionLeaseCoordinator(
            lease,
            controller_is_running=lambda: running,
            controller_stop=lambda: self.fail("natural completion must not stop twice"),
            publish_status=statuses.append,
            poll_interval_seconds=0.01,
        )
        lease.acquire("client")
        coordinator.mark_running("token")
        running = False

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not lease.status("client")["completed"]:
            time.sleep(0.01)

        self.assertTrue(lease.status("client")["completed"])
        self.assertTrue(any("Run finished" in message for message in statuses))
        coordinator.close()
        coordinator.close()


if __name__ == "__main__":
    unittest.main()
