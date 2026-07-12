from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.translation_server import (
    LOCAL_MODEL_PROFILES,
    TranslationServerConfig,
    TranslationSidecar,
    build_arg_parser,
    config_from_args,
    make_handler,
)
from window.translation_service import (
    NLLB_MODEL_ID,
    ProviderCapabilities,
    ProviderStatus,
    TRANSLATION_MODEL_METADATA,
    TranslationProvider,
)


class FakeProvider(TranslationProvider):
    provider_id = "fake-local"
    display_name = "Fake local translation"
    model_id = NLLB_MODEL_ID
    capabilities = ProviderCapabilities(
        local=True,
        requires_network=False,
        lazy_loading=True,
        supports_context=True,
        max_parallel_requests=1,
    )

    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.warmup_calls = 0

    def warmup(self) -> None:
        self.warmup_calls += 1

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider_id,
            display_name=self.display_name,
            model=self.model_id,
            available=True,
            ready=True,
            detail="ready for tests",
            capabilities=self.capabilities,
            model_metadata=TRANSLATION_MODEL_METADATA[self.model_id],
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        with self.lock:
            self.calls.append((text, source_language, target_language, tuple(context)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return f"[{target_language}] {text}"
        finally:
            with self.lock:
                self.active -= 1


class TranslationServerTests(unittest.TestCase):
    def _start_server(self, provider: TranslationProvider) -> str:
        sidecar = TranslationSidecar(TranslationServerConfig(), provider=provider)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sidecar))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop() -> None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
            sidecar.close()

        self.addCleanup(stop)
        host, port = server.server_address
        return f"http://{host}:{port}"

    @staticmethod
    def _get_json(url: str) -> tuple[int, dict[str, object]]:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    @staticmethod
    def _post_json(url: str, payload: object) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_exposes_provider_model_capabilities_license_and_readiness(self) -> None:
        base_url = self._start_server(FakeProvider())

        status, payload = self._get_json(f"{base_url}/health")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["readiness"], "ready")
        self.assertEqual(payload["provider"], "fake-local")
        self.assertEqual(payload["model"], NLLB_MODEL_ID)
        self.assertTrue(payload["capabilities"]["local"])
        self.assertEqual(payload["license"]["identifier"], "CC-BY-NC-4.0")

    def test_sidecar_warmup_prepares_provider_before_serving(self) -> None:
        provider = FakeProvider()
        sidecar = TranslationSidecar(TranslationServerConfig(), provider=provider)
        self.addCleanup(sidecar.close)

        sidecar.warmup()

        self.assertEqual(provider.warmup_calls, 1)

    def test_translate_accepts_source_text_or_text_and_preserves_context(self) -> None:
        provider = FakeProvider()
        base_url = self._start_server(provider)

        first_status, first = self._post_json(
            f"{base_url}/v1/translate",
            {
                "source_text": "Hola",
                "source_language": "es",
                "target_language": "de",
                "context": ["Buenos días"],
            },
        )
        second_status, second = self._post_json(
            f"{base_url}/v1/translate",
            {"text": "Adiós", "source_language": "es", "target_language": "en"},
        )

        self.assertEqual((first_status, first["translated_text"]), (200, "[de] Hola"))
        self.assertEqual((second_status, second["translated_text"]), (200, "[en] Adiós"))
        self.assertEqual(provider.calls[0], ("Hola", "es", "de", ("Buenos días",)))
        self.assertIn("latency_seconds", first)

    def test_threaded_http_requests_are_serialized_around_provider(self) -> None:
        provider = FakeProvider(delay_seconds=0.06)
        base_url = self._start_server(provider)

        def request(target: str) -> tuple[int, dict[str, object]]:
            return self._post_json(
                f"{base_url}/v1/translate",
                {"text": "hello", "source_language": "en", "target_language": target},
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(request, ("de", "fr")))

        self.assertEqual([status for status, _ in responses], [200, 200])
        self.assertEqual(provider.max_active, 1)

    def test_bad_requests_have_structured_errors_and_useful_status(self) -> None:
        base_url = self._start_server(FakeProvider())

        status, payload = self._post_json(
            f"{base_url}/v1/translate",
            {"source_language": "es", "target_language": "de"},
        )
        missing_status, missing = self._get_json(f"{base_url}/missing")

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_source_text")
        self.assertEqual(missing_status, 404)
        self.assertEqual(missing["error"]["code"], "not_found")

    def test_cli_profiles_include_all_local_choices_and_configure_override(self) -> None:
        self.assertEqual(
            set(LOCAL_MODEL_PROFILES),
            {"translate-gemma-4b", "nllb-200-600m", "madlad-400-3b"},
        )
        args = build_arg_parser().parse_args(
            [
                "--model-profile",
                "nllb-200-600m",
                "--model",
                "example/custom-nllb",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        config = config_from_args(args)
        provider_config = config.provider_config()

        self.assertEqual(provider_config.kind, "nllb")
        self.assertEqual(provider_config.model, "example/custom-nllb")
        self.assertEqual(provider_config.device, "cpu")
        self.assertEqual(provider_config.dtype, "float32")


if __name__ == "__main__":
    unittest.main()
