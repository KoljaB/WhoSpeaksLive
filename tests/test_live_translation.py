from __future__ import annotations

from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_translation import LiveTranslationCoordinator
from window.translation_service import translation_source_hash


class RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.lock = threading.Lock()

    def emit(self, event: str, payload: dict) -> None:
        with self.lock:
            self.events.append((event, dict(payload)))


def args(**overrides: object) -> SimpleNamespace:
    values = {
        "language": "es",
        "translation_provider": "mock",
        "translation_target_language": ["en", "de"],
        "translation_max_targets": 4,
        "translation_context_sentences": 2,
        "translation_model_profile": "translate-gemma-4b",
        "translation_model": "",
        "translation_timeout_seconds": 5.0,
        "translation_queue_size": 16,
        "translation_base_url": "http://127.0.0.1:8799",
        "translation_device": "cpu",
        "translation_dtype": "auto",
        "translation_api_key_env": "OPENAI_API_KEY",
        "translation_browser_preferred": False,
        "translation_region": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LiveTranslationCoordinatorTests(unittest.TestCase):
    def test_stable_sentence_fans_out_without_replacing_source(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(args(), bus)
        source = " Buenos días. "
        source_hash = translation_source_hash(source)
        try:
            coordinator.begin_session("meeting-1")
            coordinator.handle_sentence(
                {
                    "index": 3,
                    "text": source,
                    "start": 1.0,
                    "pending": True,
                    "source_text_hash": source_hash,
                    "source_revision": source_hash,
                },
                "meeting-1",
            )
            self.assertTrue(coordinator.service.wait_for_idle(timeout=2.0))

            records = coordinator.snapshot()
            self.assertEqual({item["target_language"] for item in records}, {"en", "de"})
            self.assertTrue(all(item["source_text_hash"] == source_hash for item in records))
            self.assertTrue(all(item["status"] == "complete" for item in records))
            self.assertEqual(source, " Buenos días. ")
            statuses = [payload["status"] for event, payload in bus.events if event == "translation"]
            self.assertEqual(statuses.count("queued"), 2)
            self.assertEqual(statuses.count("complete"), 2)
        finally:
            coordinator.shutdown()

    def test_browser_target_change_backfills_known_rows_and_enforces_capacity(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(
            args(translation_target_language=[], translation_max_targets=2),
            bus,
        )
        try:
            coordinator.begin_session("meeting-2")
            coordinator.handle_sentence({"index": 1, "text": "Hola.", "start": 0.0}, "meeting-2")
            response = coordinator.configure({"target_languages": ["en", "fr"]})
            self.assertEqual(response["selected_targets"], ["en", "fr"])
            self.assertTrue(coordinator.service.wait_for_idle(timeout=2.0))
            self.assertEqual(len(coordinator.snapshot()), 2)
            with self.assertRaisesRegex(ValueError, "at most 2"):
                coordinator.configure({"target_languages": ["en", "fr", "de"]})
        finally:
            coordinator.shutdown()

    def test_off_provider_is_a_noop_with_original_only_config(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(
            args(translation_provider="off", translation_target_language=[]),
            bus,
        )
        coordinator.handle_sentence({"index": 1, "text": "Hola."}, "meeting-3")
        config = coordinator.public_config()
        self.assertFalse(config["available"])
        self.assertEqual(config["display_mode"], "original")
        self.assertEqual(bus.events, [])

    def test_managed_provider_reads_only_its_configured_environment_variable(self) -> None:
        bus = RecordingBus()
        with (
            mock.patch.dict("os.environ", {"CUSTOM_DEEPL_KEY": "secret-value"}, clear=False),
            mock.patch("window.live_translation.create_translation_provider") as factory,
        ):
            factory.return_value = mock.Mock()
            coordinator = LiveTranslationCoordinator(args(
                translation_provider="deepl",
                translation_api_key_env="CUSTOM_DEEPL_KEY",
                translation_base_url="",
                translation_region="",
            ), bus)
        try:
            config = factory.call_args.args[0]
            self.assertEqual(config.kind, "deepl")
            self.assertEqual(config.api_key, "secret-value")
            self.assertEqual(config.base_url, "")
        finally:
            coordinator.shutdown()

    def test_browser_result_is_canonical_and_stale_results_are_rejected(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(args(
            translation_browser_preferred=True,
            translation_target_language=["en"],
        ), bus)
        try:
            coordinator.begin_session("browser-meeting")
            coordinator.handle_sentence(
                {"index": 7, "text": "Buenos dÃ­as.", "start": 0.0},
                "browser-meeting",
            )
            source_hash = translation_source_hash("Buenos dÃ­as.")
            self.assertEqual(coordinator.snapshot(), [])
            event = coordinator.accept_browser_result({
                "segment_id": 7,
                "target_language": "en",
                "source_text_hash": source_hash,
                "source_revision": source_hash,
                "text": "Good morning.",
                "latency_seconds": 0.12,
            })
            self.assertEqual(event["provider"], "chrome_translator")
            self.assertEqual(coordinator.snapshot()[0]["text"], "Good morning.")
            with self.assertRaisesRegex(ValueError, "stale"):
                coordinator.accept_browser_result({
                    "segment_id": 7,
                    "target_language": "en",
                    "source_text_hash": "old-hash",
                    "text": "Stale text",
                })
        finally:
            coordinator.shutdown()

    def test_browser_unavailable_pair_runs_selected_backend_fallback(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(args(
            translation_browser_preferred=True,
            translation_target_language=["de"],
        ), bus)
        try:
            coordinator.begin_session("fallback-meeting")
            coordinator.handle_sentence(
                {"index": 2, "text": "Hola.", "start": 0.0},
                "fallback-meeting",
            )
            source_hash = translation_source_hash("Hola.")
            coordinator.request_browser_fallback({
                "segment_id": 2,
                "target_language": "de",
                "source_text_hash": source_hash,
            })
            self.assertTrue(coordinator.service.wait_for_idle(timeout=2.0))
            records = coordinator.snapshot()
            self.assertEqual(records[0]["provider"], "mock")
            self.assertEqual(records[0]["target_language"], "de")
        finally:
            coordinator.shutdown()

    def test_failed_target_can_be_retried_after_provider_recovers(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(args(translation_target_language=["de"]), bus)
        try:
            coordinator.provider.fail_targets.add("de")
            coordinator.begin_session("meeting-4")
            coordinator.handle_sentence({"index": 1, "text": "Hola.", "start": 0.0}, "meeting-4")
            self.assertTrue(coordinator.service.wait_for_idle(timeout=2.0))
            self.assertEqual(coordinator.snapshot(), [])
            self.assertTrue(any(
                event == "translation" and payload.get("status") == "error"
                for event, payload in bus.events
            ))

            coordinator.provider.fail_targets.clear()
            coordinator.configure({"target_languages": ["de"]})
            self.assertTrue(coordinator.service.wait_for_idle(timeout=2.0))
            self.assertEqual(coordinator.snapshot()[0]["status"], "complete")
        finally:
            coordinator.shutdown()

    def test_large_backfill_is_deferred_instead_of_dropped_when_queue_is_full(self) -> None:
        bus = RecordingBus()
        coordinator = LiveTranslationCoordinator(
            args(translation_queue_size=1, translation_target_language=["en", "de"]),
            bus,
        )
        try:
            coordinator.provider.delay_seconds = 0.005
            coordinator.begin_session("meeting-5")
            for index in range(12):
                coordinator.handle_sentence(
                    {"index": index, "text": f"Frase {index}.", "start": float(index)},
                    "meeting-5",
                )
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and len(coordinator.snapshot()) < 24:
                coordinator.service.wait_for_idle(timeout=0.05)
                time.sleep(0.005)
            self.assertEqual(len(coordinator.snapshot()), 24)
            self.assertEqual(coordinator.public_config()["service"]["deferred_jobs"], 0)
            self.assertFalse(any(
                event == "translation" and payload.get("status") == "error"
                for event, payload in bus.events
            ))
        finally:
            coordinator.shutdown()


if __name__ == "__main__":
    unittest.main()
