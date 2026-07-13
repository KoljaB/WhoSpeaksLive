from __future__ import annotations

import unittest

from window.session_lease import SessionLeaseError, SessionLeaseStateMachine


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SessionLeaseStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        tokens = iter(("client-generated", "lease-token"))
        self.lease = SessionLeaseStateMachine(
            idle_timeout_seconds=10,
            heartbeat_timeout_seconds=5,
            completed_release_delay_seconds=2,
            max_run_seconds=30,
            monotonic=self.clock,
            token_factory=lambda: next(tokens),
        )

    def test_only_owner_can_authorize_and_waiter_is_recorded(self) -> None:
        acquired = self.lease.acquire("owner")
        waiting = self.lease.acquire("viewer")

        self.assertEqual(acquired["session_token"], "client-generated")
        self.assertFalse(waiting["acquired"])
        self.assertEqual(waiting["session"]["waiter_count"], 1)
        with self.assertRaises(SessionLeaseError):
            self.lease.authorize("wrong", "viewer")

    def test_heartbeat_timeout_reports_whether_controller_stop_is_required(self) -> None:
        token = str(self.lease.acquire("owner")["session_token"])
        self.lease.mark_running(token)
        self.clock.advance(6)

        expiration = self.lease.expire_if_needed()

        self.assertEqual(expiration, {"reason": "heartbeat timeout", "was_running": True})
        self.assertFalse(self.lease.status("owner")["active"])

    def test_completed_run_releases_after_delay_without_running_stop(self) -> None:
        token = str(self.lease.acquire("owner")["session_token"])
        self.lease.mark_running(token)
        self.assertTrue(self.lease.mark_completed(token))
        self.clock.advance(3)

        expiration = self.lease.expire_if_needed()

        self.assertEqual(expiration, {"reason": "completed", "was_running": False})


if __name__ == "__main__":
    unittest.main()
