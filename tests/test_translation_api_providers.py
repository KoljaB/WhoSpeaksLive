from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.translation_service import (
    AzureTranslatorProvider,
    DeepLTranslationProvider,
    GoogleCloudTranslationProvider,
    LibreTranslateProvider,
)


class ProviderProtocolServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ProviderProtocolHandler)
        self.requests: list[dict[str, object]] = []


class ProviderProtocolHandler(BaseHTTPRequestHandler):
    server: ProviderProtocolServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append({
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        })
        if self.path == "/v2/translate":
            response: object = {"translations": [{"text": "DeepL result"}]}
        elif self.path == "/google":
            response = {"data": {"translations": [{"translatedText": "Google result"}]}}
        elif self.path.startswith("/translate?api-version=3.0"):
            response = [{"translations": [{"text": "Azure result", "to": "en"}]}]
        elif self.path == "/libre/translate":
            response = {"translatedText": "Libre result"}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class TranslationApiProviderRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ProviderProtocolServer()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def setUp(self) -> None:
        self.server.requests.clear()

    def test_all_dedicated_providers_complete_real_http_round_trips(self) -> None:
        providers = [
            DeepLTranslationProvider(base_url=f"{self.base_url}/v2", api_key="deepl-key"),
            GoogleCloudTranslationProvider(base_url=f"{self.base_url}/google", api_key="google-key"),
            AzureTranslatorProvider(
                base_url=self.base_url,
                api_key="azure-key",
                region="westeurope",
            ),
            LibreTranslateProvider(base_url=f"{self.base_url}/libre", api_key="libre-key"),
        ]

        results = [
            provider.translate("Guten Morgen.", "de", "en", context=("BegrÃ¼ÃŸung",))
            for provider in providers
        ]

        self.assertEqual(
            results,
            ["DeepL result", "Google result", "Azure result", "Libre result"],
        )
        self.assertEqual(len(self.server.requests), 4)
        requests_by_path = {str(item["path"]).split("?", 1)[0]: item for item in self.server.requests}
        self.assertEqual(
            requests_by_path["/v2/translate"]["headers"]["authorization"],
            "DeepL-Auth-Key deepl-key",
        )
        self.assertEqual(
            requests_by_path["/google"]["headers"]["x-goog-api-key"],
            "google-key",
        )
        self.assertEqual(
            requests_by_path["/translate"]["headers"]["ocp-apim-subscription-region"],
            "westeurope",
        )
        self.assertEqual(
            requests_by_path["/libre/translate"]["body"]["api_key"],
            "libre-key",
        )


if __name__ == "__main__":
    unittest.main()
