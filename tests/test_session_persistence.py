from __future__ import annotations

import threading
import time
import unittest

from window.session_persistence import SessionPersistenceCoordinator
from window.window_events import EventBus


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def save_snapshot(self, _snapshot, *, status_label, write_audio, audio_writer):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self.lock:
            self.calls.append((status_label, write_audio))
            self.active -= 1
        return {"status_label": status_label}


class SessionPersistenceCoordinatorTests(unittest.TestCase):
    def make_coordinator(self):
        bus = EventBus()
        store = RecordingStore()
        translated: list[str] = []
        coordinator = SessionPersistenceCoordinator(
            bus=bus,
            store=store,  # type: ignore[arg-type]
            session_snapshot=lambda: {"id": "session-1", "transcript_rows": []},
            session_id=lambda: "session-1",
            write_audio=lambda _path: True,
            translation_snapshot=lambda: [{"segment_id": "1"}],
            handle_sentence_translation=lambda _payload, session_id: translated.append(session_id),
            debounce_seconds=60,
        )
        return bus, store, translated, coordinator

    def test_done_cancels_autosave_and_performs_one_final_audio_save(self) -> None:
        bus, store, translated, coordinator = self.make_coordinator()
        bus.emit("sentence", {"index": 1, "text": "Hello"})
        bus.emit("done", {"message": "done"})

        self.assertEqual(translated, ["session-1"])
        self.assertEqual(store.calls, [("Saved", True)])
        coordinator.close(flush=False)

    def test_concurrent_save_requests_are_serialized(self) -> None:
        _bus, store, _translated, coordinator = self.make_coordinator()
        first = threading.Thread(target=coordinator.save, kwargs={"status_label": "A", "write_audio": False})
        second = threading.Thread(target=coordinator.save, kwargs={"status_label": "B", "write_audio": False})
        first.start()
        second.start()
        first.join()
        second.join()

        self.assertEqual(store.max_active, 1)
        self.assertEqual({label for label, _ in store.calls}, {"A", "B"})
        coordinator.close(flush=False)

    def test_close_unregisters_listener_and_is_idempotent(self) -> None:
        bus, store, translated, coordinator = self.make_coordinator()
        coordinator.close(flush=False)
        coordinator.close(flush=False)
        bus.emit("sentence", {"index": 1, "text": "ignored"})

        self.assertEqual(translated, [])
        self.assertEqual(store.calls, [])


if __name__ == "__main__":
    unittest.main()
