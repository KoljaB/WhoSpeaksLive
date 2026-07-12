from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TranslationMetadataTests(unittest.TestCase):
    def test_model_metadata_keeps_license_constraints_machine_readable(self) -> None:
        from window.translation_service import (
            MADLAD_MODEL_ID,
            NLLB_MODEL_ID,
            TRANSLATEGEMMA_MODEL_ID,
            TRANSLATION_MODEL_METADATA,
        )

        nllb = TRANSLATION_MODEL_METADATA[NLLB_MODEL_ID].to_dict()
        gemma = TRANSLATION_MODEL_METADATA[TRANSLATEGEMMA_MODEL_ID].to_dict()
        madlad = TRANSLATION_MODEL_METADATA[MADLAD_MODEL_ID].to_dict()

        self.assertEqual(nllb["license"]["identifier"], "CC-BY-NC-4.0")
        self.assertFalse(nllb["license"]["commercial_use"])
        self.assertTrue(gemma["license"]["acceptance_required"])
        self.assertEqual(madlad["license"]["identifier"], "Apache-2.0")
        json.dumps({"nllb": nllb, "gemma": gemma, "madlad": madlad})

    def test_request_normalizes_and_deduplicates_targets(self) -> None:
        from window.translation_service import TranslationRequest, translation_source_hash

        request = TranslationRequest(
            segment_id=" row-7 ",
            session_id="session-a",
            source_text="Hola",
            source_language="ES_es",
            target_languages=["DE_de", "de-DE", "FR"],
            source_revision="source-hash-v2",
        )

        self.assertEqual(request.segment_id, "row-7")
        self.assertEqual(request.source_language, "es-ES")
        self.assertEqual(request.target_languages, ("de-DE", "fr"))
        self.assertEqual(request.source_hash, translation_source_hash("Hola"))

    def test_factory_is_lazy_and_status_is_json_serializable(self) -> None:
        from window.translation_service import (
            TRANSLATEGEMMA_MODEL_ID,
            TranslationProviderConfig,
            create_translation_provider,
        )

        with mock.patch("window.translation_service.importlib.import_module") as import_module:
            provider = create_translation_provider(
                TranslationProviderConfig(kind="translategemma", local_files_only=True)
            )
            self.assertEqual(provider.model_id, TRANSLATEGEMMA_MODEL_ID)
            self.assertFalse(provider.status().ready)
        import_module.assert_not_called()
        json.dumps(provider.status().to_dict())


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_chat_request_contains_languages_context_and_no_key_in_status(self) -> None:
        from window.translation_service import OpenAICompatibleTranslationProvider

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": [{"text": "Good morning"}]}}]}
                ).encode("utf-8")

        provider = OpenAICompatibleTranslationProvider(
            base_url="http://127.0.0.1:8081/v1/",
            model="local-report-model",
            api_key="top-secret",
            timeout_seconds=3,
        )
        with mock.patch(
            "window.translation_service.urllib.request.urlopen", return_value=FakeResponse()
        ) as urlopen:
            translated = provider.translate(
                "Buenos días",
                "es",
                "en",
                context=("La reunión acaba de comenzar.",),
            )

        self.assertEqual(translated, "Good morning")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(request.full_url, "http://127.0.0.1:8081/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer top-secret")
        self.assertEqual(user_payload["source_language"], "es")
        self.assertEqual(user_payload["target_language"], "en")
        self.assertEqual(user_payload["context"], ["La reunión acaba de comenzar."])
        self.assertNotIn("top-secret", json.dumps(provider.status().to_dict()))


class DedicatedApiProviderTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def _translate(self, provider, response: object, **kwargs: object):
        with mock.patch(
            "window.translation_service.urllib.request.urlopen",
            return_value=self.FakeResponse(response),
        ) as urlopen:
            result = provider.translate("Guten Morgen & willkommen.", "de", "en", **kwargs)
        return result, urlopen.call_args.args[0]

    def test_deepl_protocol_uses_auth_header_and_unbilled_context(self) -> None:
        from window.translation_service import DeepLTranslationProvider

        provider = DeepLTranslationProvider(api_key="deepl-secret")
        result, request = self._translate(
            provider,
            {"translations": [{"text": "Good morning & welcome."}]},
            context=("Die Besprechung beginnt.",),
        )

        self.assertEqual(result, "Good morning & welcome.")
        self.assertEqual(request.full_url, "https://api-free.deepl.com/v2/translate")
        self.assertEqual(request.get_header("Authorization"), "DeepL-Auth-Key deepl-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["text"], ["Guten Morgen & willkommen."])
        self.assertEqual(payload["source_lang"], "DE")
        self.assertEqual(payload["target_lang"], "EN")
        self.assertEqual(payload["context"], "Die Besprechung beginnt.")
        self.assertNotIn("deepl-secret", json.dumps(provider.status().to_dict()))

    def test_google_cloud_protocol_keeps_key_out_of_url_and_decodes_entities(self) -> None:
        from window.translation_service import GoogleCloudTranslationProvider

        provider = GoogleCloudTranslationProvider(api_key="google-secret")
        result, request = self._translate(
            provider,
            {"data": {"translations": [{"translatedText": "Good morning &amp; welcome."}]}},
        )

        self.assertEqual(result, "Good morning & welcome.")
        self.assertNotIn("google-secret", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "google-secret")
        payload = json.loads(request.data)
        self.assertEqual(payload["source"], "de")
        self.assertEqual(payload["target"], "en")
        self.assertEqual(payload["model"], "nmt")

    def test_azure_protocol_sets_region_and_category(self) -> None:
        from window.translation_service import AzureTranslatorProvider

        provider = AzureTranslatorProvider(
            api_key="azure-secret",
            region="westeurope",
            model="general",
        )
        result, request = self._translate(
            provider,
            [{"translations": [{"text": "Good morning & welcome.", "to": "en"}]}],
        )

        self.assertEqual(result, "Good morning & welcome.")
        self.assertIn("api-version=3.0", request.full_url)
        self.assertIn("from=de", request.full_url)
        self.assertIn("to=en", request.full_url)
        self.assertIn("category=general", request.full_url)
        self.assertEqual(request.get_header("Ocp-apim-subscription-key"), "azure-secret")
        self.assertEqual(request.get_header("Ocp-apim-subscription-region"), "westeurope")
        self.assertEqual(json.loads(request.data), [{"Text": "Guten Morgen & willkommen."}])

    def test_libretranslate_protocol_supports_keyless_custom_endpoint(self) -> None:
        from window.translation_service import LibreTranslateProvider

        provider = LibreTranslateProvider(base_url="http://translate.internal:5000/")
        result, request = self._translate(
            provider,
            {"translatedText": "Good morning & welcome."},
        )

        self.assertEqual(result, "Good morning & welcome.")
        self.assertEqual(request.full_url, "http://translate.internal:5000/translate")
        self.assertNotIn("api_key", json.loads(request.data))
        self.assertTrue(provider.status().ready)

    def test_factory_exposes_all_dedicated_api_providers(self) -> None:
        from window.translation_service import TranslationProviderConfig, create_translation_provider

        providers = {
            "deepl": "deepl",
            "google_cloud": "google_cloud",
            "azure_translator": "azure_translator",
            "libretranslate": "libretranslate",
        }
        for kind, provider_id in providers.items():
            with self.subTest(kind=kind):
                provider = create_translation_provider(TranslationProviderConfig(
                    kind=kind,
                    api_key="secret",
                    options={"region": "westeurope"},
                ))
                self.assertEqual(provider.provider_id, provider_id)


class FakeBatch(dict):
    def __init__(self, input_ids=None) -> None:
        super().__init__(input_ids=input_ids or [[10, 11, 12]])
        self.moves: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeModel:
    def __init__(self, output) -> None:
        self.output = output
        self.device = "cpu"
        self.to_device = ""
        self.generate_kwargs = None

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.output


class TransformersProviderTests(unittest.TestCase):
    def _module_pair(self, processor, model, *, gemma: bool = False):
        class Loader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                return processor

        class ModelLoader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                return model

        transformers = type(
            "FakeTransformers",
            (),
            {
                "AutoProcessor": Loader,
                "AutoTokenizer": Loader,
                "AutoModelForSeq2SeqLM": ModelLoader,
                "AutoModelForImageTextToText": ModelLoader,
            },
        )()
        return lambda name: FakeTorch if name == "torch" else transformers

    def test_nllb_uses_flores_language_tokens(self) -> None:
        from window.translation_service import NLLB_MODEL_ID, TransformersTranslationProvider

        class Tokenizer:
            src_lang = ""

            def __init__(self) -> None:
                self.target_token = ""

            def __call__(self, text, **kwargs):
                self.text = text
                return FakeBatch()

            def convert_tokens_to_ids(self, token):
                self.target_token = token
                return 42

            def batch_decode(self, output, skip_special_tokens):
                return ["Guten Morgen"]

        tokenizer = Tokenizer()
        model = FakeModel([[1, 2]])
        provider = TransformersTranslationProvider(model=NLLB_MODEL_ID, device="cpu")
        with mock.patch(
            "window.translation_service.importlib.import_module",
            side_effect=self._module_pair(tokenizer, model),
        ):
            translated = provider.translate("Good morning", "en", "de")

        self.assertEqual(translated, "Guten Morgen")
        self.assertEqual(tokenizer.src_lang, "eng_Latn")
        self.assertEqual(tokenizer.target_token, "deu_Latn")
        self.assertEqual(model.generate_kwargs["forced_bos_token_id"], 42)
        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertFalse(provider.supports_language_pair("la", "de"))

    def test_translategemma_uses_official_structured_chat_template(self) -> None:
        from window.translation_service import (
            TRANSLATEGEMMA_MODEL_ID,
            TransformersTranslationProvider,
        )

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                self.batch = FakeBatch([[1, 2, 3]])
                return self.batch

            def decode(self, generated, skip_special_tokens):
                self.generated = generated
                return "Im schlimmsten Fall."

        processor = Processor()
        model = FakeModel([[1, 2, 3, 90, 91]])
        provider = TransformersTranslationProvider(model=TRANSLATEGEMMA_MODEL_ID, device="cpu")
        with mock.patch(
            "window.translation_service.importlib.import_module",
            side_effect=self._module_pair(processor, model, gemma=True),
        ):
            translated = provider.translate("V nejhorším případě.", "cs", "de-DE")

        item = processor.messages[0]["content"][0]
        self.assertEqual(translated, "Im schlimmsten Fall.")
        self.assertEqual(item["type"], "text")
        self.assertEqual(item["source_lang_code"], "cs")
        self.assertEqual(item["target_lang_code"], "de-DE")
        self.assertTrue(processor.template_kwargs["add_generation_prompt"])
        self.assertEqual(processor.generated, [90, 91])
        self.assertFalse(model.generate_kwargs["do_sample"])

    def test_madlad_uses_target_control_token(self) -> None:
        from window.translation_service import MADLAD_MODEL_ID, TransformersTranslationProvider

        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.text = text
                return FakeBatch()

            def batch_decode(self, output, skip_special_tokens):
                return ["Bonjour"]

        tokenizer = Tokenizer()
        model = FakeModel([[1, 2]])
        provider = TransformersTranslationProvider(model=MADLAD_MODEL_ID, device="cpu")
        with mock.patch(
            "window.translation_service.importlib.import_module",
            side_effect=self._module_pair(tokenizer, model),
        ):
            translated = provider.translate("Hello", "en", "fr-FR")

        self.assertEqual(translated, "Bonjour")
        self.assertEqual(tokenizer.text, "<2fr> Hello")


class TranslationServiceTests(unittest.TestCase):
    def test_multi_target_fanout_caching_and_provenance(self) -> None:
        from window.translation_service import (
            MockTranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        provider = MockTranslationProvider(
            {
                ("de", "Hola"): "Hallo",
                ("fr", "Hola"): "Bonjour",
            }
        )
        observed = []
        service = TranslationService(provider, worker_count=2, on_result=observed.append)
        try:
            first_request = TranslationRequest(
                segment_id="sentence-1",
                session_id="meeting-9",
                source_text="Hola",
                source_language="es",
                target_languages=("de", "fr"),
                source_revision="r1",
            )
            first = service.submit(first_request).wait(timeout=2)
            second_request = TranslationRequest(
                segment_id="sentence-2",
                session_id="meeting-9",
                source_text="Hola",
                source_language="es",
                target_languages=("de", "fr"),
                source_revision="r1-other-row",
            )
            second = service.submit(second_request).wait(timeout=2)
        finally:
            service.close()

        self.assertEqual(first["de"].translated_text, "Hallo")
        self.assertEqual(first["fr"].translated_text, "Bonjour")
        self.assertFalse(first["de"].cached)
        self.assertTrue(second["de"].cached)
        self.assertTrue(second["fr"].cached)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(second["de"].source_revision, "r1-other-row")
        self.assertEqual(second["de"].session_id, "meeting-9")
        self.assertEqual(second["de"].provider, "mock")
        self.assertEqual(second["de"].model, "mock")
        self.assertGreaterEqual(second["de"].latency_seconds, 0.0)
        self.assertEqual(len(observed), 4)

    def test_new_revision_suppresses_inflight_old_result(self) -> None:
        from window.translation_service import (
            ProviderCapabilities,
            TranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        class BlockingProvider(TranslationProvider):
            provider_id = "blocking"
            display_name = "Blocking"
            model_id = "blocking-v1"
            capabilities = ProviderCapabilities(True, False, False, False)

            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()
                self.calls = []

            def translate(self, text, source_language, target_language, *, context=()):
                self.calls.append(text)
                if text == "old text":
                    self.started.set()
                    self.release.wait(2)
                return f"translated: {text}"

        provider = BlockingProvider()
        observed = []
        service = TranslationService(provider, on_result=observed.append)
        try:
            old = service.submit(
                TranslationRequest("row", "old text", "en", ("de",), source_revision="old")
            )
            self.assertTrue(provider.started.wait(1))
            new = service.submit(
                TranslationRequest("row", "new text", "en", ("de",), source_revision="new")
            )
            provider.release.set()
            old_result = old.result("de", timeout=2)
            new_result = new.result("de", timeout=2)
        finally:
            provider.release.set()
            service.close()

        self.assertEqual(old_result.status, "superseded")
        self.assertEqual(old_result.source_revision, "old")
        self.assertEqual(old_result.translated_text, "")
        self.assertEqual(new_result.status, "completed")
        self.assertEqual(new_result.translated_text, "translated: new text")
        self.assertEqual([result.source_revision for result in observed], ["new"])

    def test_queued_stale_revision_is_skipped_without_calling_provider(self) -> None:
        from window.translation_service import (
            ProviderCapabilities,
            TranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        class GateProvider(TranslationProvider):
            provider_id = "gate"
            display_name = "Gate"
            model_id = "gate-v1"
            capabilities = ProviderCapabilities(True, False, False, False)

            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()
                self.calls = []

            def translate(self, text, source_language, target_language, *, context=()):
                self.calls.append(text)
                if text == "occupy worker":
                    self.started.set()
                    self.release.wait(2)
                return text.upper()

        provider = GateProvider()
        service = TranslationService(provider, worker_count=1)
        try:
            occupying = service.submit(TranslationRequest("a", "occupy worker", "en", ("de",)))
            self.assertTrue(provider.started.wait(1))
            old = service.submit(
                TranslationRequest("b", "obsolete", "en", ("de",), source_revision="1")
            )
            new = service.submit(
                TranslationRequest("b", "current", "en", ("de",), source_revision="2")
            )
            provider.release.set()
            occupying.wait(timeout=2)
            old_result = old.result("de", timeout=2)
            new_result = new.result("de", timeout=2)
        finally:
            provider.release.set()
            service.close()

        self.assertEqual(old_result.status, "superseded")
        self.assertEqual(new_result.translated_text, "CURRENT")
        self.assertEqual(provider.calls, ["occupy worker", "current"])

    def test_identical_inflight_submissions_are_deduplicated(self) -> None:
        from window.translation_service import (
            MockTranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        provider = MockTranslationProvider(delay_seconds=0.08)
        globally_observed = []
        per_submission = []
        service = TranslationService(provider, on_result=globally_observed.append)
        request = TranslationRequest("row", "Hello", "en", ("de",), source_revision="same")
        try:
            first = service.submit(request, callback=per_submission.append)
            second = service.submit(request, callback=per_submission.append)
            first_result = first.result("de", timeout=2)
            second_result = second.result("de", timeout=2)
        finally:
            service.close()

        self.assertEqual(first_result, second_result)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(globally_observed), 1)
        self.assertEqual(len(per_submission), 2)

    def test_same_segment_id_in_different_sessions_does_not_supersede(self) -> None:
        from window.translation_service import (
            MockTranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        provider = MockTranslationProvider(delay_seconds=0.04)
        service = TranslationService(provider, worker_count=2, cache_size=0)
        try:
            first = service.submit(
                TranslationRequest(
                    "row-1", "First", "en", ("de",), source_revision="1", session_id="a"
                )
            )
            second = service.submit(
                TranslationRequest(
                    "row-1", "Second", "en", ("de",), source_revision="1", session_id="b"
                )
            )
            first_result = first.result("de", timeout=2)
            second_result = second.result("de", timeout=2)
        finally:
            service.close()

        self.assertEqual(first_result.status, "completed")
        self.assertEqual(second_result.status, "completed")
        self.assertEqual({call[0] for call in provider.calls}, {"First", "Second"})

    def test_provider_failure_is_a_provenance_complete_result(self) -> None:
        from window.translation_service import (
            MockTranslationProvider,
            TranslationRequest,
            TranslationService,
        )

        provider = MockTranslationProvider(fail_targets=("de",))
        service = TranslationService(provider)
        try:
            request = TranslationRequest(
                "row-error",
                "Hello",
                "en",
                ("de",),
                source_revision="hash-token",
                session_id="session-z",
            )
            result = service.submit(request).result("de", timeout=2)
            status = service.status()
        finally:
            service.close()

        self.assertEqual(result.status, "error")
        self.assertEqual(result.source_revision, "hash-token")
        self.assertEqual(result.source_hash, request.source_hash)
        self.assertEqual(result.target_language, "de")
        self.assertIn("mock translation failure", result.error)
        self.assertFalse(result.cached)
        json.dumps(result.to_dict())
        json.dumps(status)


if __name__ == "__main__":
    unittest.main()
