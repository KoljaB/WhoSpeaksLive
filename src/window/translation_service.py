"""Revision-safe background translation providers for live transcript rows.

The module deliberately has no import-time dependency on PyTorch or
Transformers.  A local model is imported and loaded only when its provider
receives the first translation request, so merely enabling the GUI or querying
provider status does not allocate model memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
import html
import importlib
import importlib.util
import json
import logging
from queue import Empty, Queue
import threading
import time
from typing import Any, Literal
import urllib.error
import urllib.parse
import urllib.request


from window.translation.contracts import (
    LOGGER,
    MADLAD_MODEL_ID,
    NLLB_MODEL_ID,
    TRANSLATEGEMMA_MODEL_ID,
    TRANSLATION_MODEL_METADATA,
    TRANSLATION_RESULT_STATUSES,
    ModelLicenseMetadata,
    ProviderCapabilities,
    ProviderStatus,
    TranslationCallback,
    TranslationModelMetadata,
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
    TranslationResultStatus,
    _normalize_language_tag,
    translation_source_hash,
)
from window.translation.local_runtime import LocalModelState

class IdentityTranslationProvider(TranslationProvider):
    provider_id = "identity"
    display_name = "Identity (no translation)"
    model_id = "identity"
    capabilities = ProviderCapabilities(
        local=True,
        requires_network=False,
        lazy_loading=False,
        supports_context=False,
        max_parallel_requests=None,
    )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        return str(text)


class MockTranslationProvider(TranslationProvider):
    """Deterministic provider for tests, demos, and browser integration work."""

    provider_id = "mock"
    display_name = "Mock translation"
    model_id = "mock"
    capabilities = ProviderCapabilities(
        local=True,
        requires_network=False,
        lazy_loading=False,
        supports_context=True,
    )

    def __init__(
        self,
        translations: Mapping[object, str] | None = None,
        *,
        prefix: str = "",
        delay_seconds: float = 0.0,
        fail_targets: Iterable[str] = (),
        translator: Callable[[str, str, str, Sequence[str]], str] | None = None,
    ) -> None:
        self.translations = dict(translations or {})
        self.prefix = str(prefix)
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.fail_targets = {_normalize_language_tag(value) for value in fail_targets}
        self.translator = translator
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        source = _normalize_language_tag(source_language)
        target = _normalize_language_tag(target_language)
        context_tuple = tuple(context)
        with self._lock:
            self.calls.append((text, source, target, context_tuple))
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if target in self.fail_targets:
            raise RuntimeError(f"mock translation failure for {target}")
        if self.translator is not None:
            return str(self.translator(text, source, target, context_tuple))
        for key in ((source, target, text), (target, text), target, text):
            if key in self.translations:
                return str(self.translations[key])
        return f"{self.prefix}[{target}] {text}"


from window.translation.api_providers import (
    AzureTranslatorProvider,
    DeepLTranslationProvider,
    GoogleCloudTranslationProvider,
    HttpTranslationProvider,
    LibreTranslateProvider,
    OpenAICompatibleTranslationProvider,
)


# ISO-639 aliases used by the app mapped to the FLORES-200 tags expected by NLLB.
NLLB_LANGUAGE_CODES: dict[str, str] = {
    "af": "afr_Latn", "ar": "arb_Arab", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fo": "fao_Latn", "fr": "fra_Latn", "gl": "glg_Latn", "he": "heb_Hebr",
    "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "ml": "mal_Mlym", "mr": "mar_Deva",
    "mt": "mlt_Latn", "my": "mya_Mymr", "nl": "nld_Latn", "nn": "nno_Latn",
    "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sa": "san_Deva", "sd": "snd_Arab", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn", "zh": "zho_Hans",
}


@dataclass
class _TransformersRuntime:
    torch: Any
    processor: Any
    model: Any
    device: str
    dtype: Any


class TransformersTranslationProvider(TranslationProvider):
    """Lazy local backend for NLLB, TranslateGemma, and MADLAD models."""

    provider_id = "transformers"
    display_name = "Local Transformers model"

    def __init__(
        self,
        *,
        model: str = TRANSLATEGEMMA_MODEL_ID,
        family: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        max_input_tokens: int = 512,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = str(model or "").strip()
        if not self.model_id:
            raise ValueError("model must not be empty")
        self.family = self._normalize_family(family, self.model_id)
        self.device = str(device or "auto").strip().lower()
        self.dtype = str(dtype or "auto").strip().lower()
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.max_input_tokens = max(8, int(max_input_tokens))
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        supported = tuple(sorted(NLLB_LANGUAGE_CODES)) if self.family == "nllb" else None
        self.capabilities = ProviderCapabilities(
            local=True,
            requires_network=not self.local_files_only,
            lazy_loading=True,
            supports_context=False,
            supported_language_codes=supported,
            max_parallel_requests=1,
        )
        self._runtime: _TransformersRuntime | None = None
        self._runtime_state = LocalModelState.UNLOADED
        self._load_error = ""
        self._load_condition = threading.Condition(threading.Lock())
        self._inference_lock = threading.Lock()

    @staticmethod
    def _normalize_family(family: str, model: str) -> str:
        normalized = str(family or "auto").strip().lower().replace("-", "")
        if normalized == "auto":
            lower_model = model.lower()
            if "nllb" in lower_model:
                return "nllb"
            if "translategemma" in lower_model or "translate-gemma" in lower_model:
                return "translategemma"
            if "madlad" in lower_model:
                return "madlad"
            raise ValueError("cannot infer Transformers translation family; choose nllb, translategemma, or madlad")
        aliases = {"translate_gemma": "translategemma", "gemma": "translategemma"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"nllb", "translategemma", "madlad"}:
            raise ValueError("family must be nllb, translategemma, madlad, or auto")
        return normalized

    @property
    def cache_identity(self) -> str:
        return f"{self.provider_id}:{self.family}:{self.model_id}"

    def status(self) -> ProviderStatus:
        with self._load_condition:
            runtime = self._runtime
            runtime_state = self._runtime_state
            load_error = self._load_error
        dependencies = ("torch", "transformers")
        missing = [name for name in dependencies if not _module_available(name)]
        available = not missing
        if runtime is not None and runtime_state is LocalModelState.READY:
            detail = f"loaded on {runtime.device}"
        elif runtime_state is LocalModelState.CLOSED:
            detail = "local model runtime is closed"
        elif runtime_state is LocalModelState.LOADING:
            detail = "local model runtime is loading"
        elif load_error:
            detail = load_error
        elif missing:
            detail = f"missing optional dependencies: {', '.join(missing)}"
        else:
            detail = "available; model loads on the first translation request"
        return ProviderStatus(
            provider=self.provider_id,
            display_name=self.display_name,
            model=self.model_id,
            available=available,
            ready=runtime is not None and runtime_state is LocalModelState.READY,
            detail=detail,
            capabilities=self.capabilities,
            model_metadata=TRANSLATION_MODEL_METADATA.get(self.model_id),
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        with self._inference_lock:
            runtime = self._ensure_runtime()
            if self.family == "nllb":
                return self._translate_nllb(runtime, text, source_language, target_language)
            if self.family == "translategemma":
                return self._translate_gemma(runtime, text, source_language, target_language)
            return self._translate_madlad(runtime, text, target_language)

    def warmup(self) -> None:
        self._ensure_runtime()

    def _ensure_runtime(self) -> _TransformersRuntime:
        with self._load_condition:
            while self._runtime_state is LocalModelState.LOADING:
                self._load_condition.wait()
            if self._runtime_state is LocalModelState.READY and self._runtime is not None:
                return self._runtime
            if self._runtime_state is LocalModelState.CLOSED:
                raise RuntimeError("local translation model runtime is closed")
            self._runtime_state = LocalModelState.LOADING
        try:
            runtime = self._load_runtime()
        except Exception as exc:
            with self._load_condition:
                if self._runtime_state is not LocalModelState.CLOSED:
                    self._load_error = f"model load failed: {exc}"
                    self._runtime_state = LocalModelState.FAILED
                self._load_condition.notify_all()
                detail = self._load_error or "local translation model runtime closed while loading"
            raise RuntimeError(detail) from exc
        with self._load_condition:
            if self._runtime_state is LocalModelState.CLOSED:
                self._load_condition.notify_all()
                raise RuntimeError("local translation model runtime closed while loading")
            self._runtime = runtime
            self._runtime_state = LocalModelState.READY
            self._load_error = ""
            self._load_condition.notify_all()
            return runtime

    def _load_runtime(self) -> _TransformersRuntime:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "local translation requires optional dependencies `torch` and `transformers`"
            ) from exc
        device = self._resolved_device(torch)
        dtype = self._resolved_dtype(torch, device)
        common = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        model_kwargs = dict(common)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if self.family == "translategemma":
            processor = transformers.AutoProcessor.from_pretrained(self.model_id, **common)
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_class is None:
                raise RuntimeError(
                    "the installed transformers version lacks AutoModelForImageTextToText; "
                    "upgrade transformers for TranslateGemma support"
                )
            model = model_class.from_pretrained(self.model_id, **model_kwargs)
        else:
            processor = transformers.AutoTokenizer.from_pretrained(self.model_id, **common)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **model_kwargs)
        if device != "auto" and hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return _TransformersRuntime(torch=torch, processor=processor, model=model, device=device, dtype=dtype)

    def _resolved_device(self, torch: Any) -> str:
        if self.device != "auto":
            return self.device
        cuda = getattr(torch, "cuda", None)
        return "cuda" if cuda is not None and bool(cuda.is_available()) else "cpu"

    def _resolved_dtype(self, torch: Any, device: str) -> Any:
        if self.dtype in {"", "auto"}:
            if device.startswith("cuda"):
                cuda = getattr(torch, "cuda", None)
                bf16_supported = bool(
                    cuda is not None
                    and callable(getattr(cuda, "is_bf16_supported", None))
                    and cuda.is_bf16_supported()
                )
                return getattr(torch, "bfloat16" if bf16_supported else "float16", None)
            return getattr(torch, "float32", None)
        aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}
        name = aliases.get(self.dtype, self.dtype)
        dtype = getattr(torch, name, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype {self.dtype!r}")
        return dtype

    def _translate_nllb(
        self,
        runtime: _TransformersRuntime,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        source_code = _nllb_code(source_language)
        target_code = _nllb_code(target_language)
        tokenizer = runtime.processor
        tokenizer.src_lang = source_code
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = _move_batch(inputs, runtime.device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise ValueError(f"NLLB tokenizer does not support target language {target_language!r}")
        with runtime.torch.inference_mode():
            output = runtime.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        return str(tokenizer.batch_decode(output, skip_special_tokens=True)[0]).strip()

    def _translate_gemma(
        self,
        runtime: _TransformersRuntime,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": _normalize_language_tag(source_language),
                        "target_lang_code": _normalize_language_tag(target_language),
                        "text": str(text),
                    }
                ],
            }
        ]
        inputs = runtime.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_batch(inputs, runtime.device, runtime.dtype)
        input_length = len(inputs["input_ids"][0])
        with runtime.torch.inference_mode():
            output = runtime.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        generated = output[0][input_length:]
        return str(runtime.processor.decode(generated, skip_special_tokens=True)).strip()

    def _translate_madlad(
        self,
        runtime: _TransformersRuntime,
        text: str,
        target_language: str,
    ) -> str:
        target = _normalize_language_tag(target_language).split("-", 1)[0]
        inputs = runtime.processor(
            f"<2{target}> {text}",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = _move_batch(inputs, runtime.device)
        with runtime.torch.inference_mode():
            output = runtime.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        return str(runtime.processor.batch_decode(output, skip_special_tokens=True)[0]).strip()

    def close(self) -> None:
        with self._inference_lock:
            with self._load_condition:
                self._runtime_state = LocalModelState.CLOSED
                self._runtime = None
                self._load_condition.notify_all()


def _nllb_code(language: str) -> str:
    value = str(language).strip()
    if "_" in value and len(value.rsplit("_", 1)[-1]) == 4:
        return value
    normalized = _normalize_language_tag(value).split("-", 1)[0]
    try:
        return NLLB_LANGUAGE_CODES[normalized]
    except KeyError as exc:
        raise ValueError(f"NLLB language mapping is not configured for {language!r}") from exc


def _move_batch(batch: Any, device: str, dtype: Any = None) -> Any:
    if not hasattr(batch, "to"):
        return batch
    if dtype is not None:
        try:
            return batch.to(device, dtype=dtype)
        except TypeError:
            pass
    return batch.to(device)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class TranslationProviderConfig:
    kind: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0
    device: str = "auto"
    dtype: str = "auto"
    family: str = "auto"
    max_new_tokens: int = 256
    max_input_tokens: int = 512
    trust_remote_code: bool = False
    local_files_only: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


def create_translation_provider(
    config: TranslationProviderConfig | Mapping[str, Any],
) -> TranslationProvider:
    """Create one provider without importing optional heavyweight libraries."""

    if isinstance(config, Mapping):
        config = TranslationProviderConfig(**dict(config))
    kind = str(config.kind or "").strip().lower().replace("-", "_")
    options = dict(config.options)
    if kind in {"identity", "none", "off"}:
        return IdentityTranslationProvider()
    if kind == "mock":
        return MockTranslationProvider(**options)
    if kind in {"openai", "openai_compatible", "llm"}:
        return OpenAICompatibleTranslationProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            extra_headers=config.extra_headers,
        )
    if kind == "deepl":
        return DeepLTranslationProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
    if kind in {"google", "google_cloud", "google_translate"}:
        return GoogleCloudTranslationProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
    if kind in {"azure", "azure_translator"}:
        return AzureTranslatorProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            region=str(options.get("region") or ""),
        )
    if kind in {"libretranslate", "libre_translate"}:
        return LibreTranslateProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
    if kind in {"nllb", "translategemma", "translate_gemma", "madlad", "transformers", "local"}:
        default_models = {
            "nllb": NLLB_MODEL_ID,
            "translategemma": TRANSLATEGEMMA_MODEL_ID,
            "translate_gemma": TRANSLATEGEMMA_MODEL_ID,
            "madlad": MADLAD_MODEL_ID,
            "transformers": TRANSLATEGEMMA_MODEL_ID,
            "local": TRANSLATEGEMMA_MODEL_ID,
        }
        family = config.family if kind in {"transformers", "local"} else kind
        return TransformersTranslationProvider(
            model=config.model or default_models[kind],
            family=family,
            device=config.device,
            dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            max_input_tokens=config.max_input_tokens,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
    raise ValueError(f"unsupported translation provider kind: {config.kind!r}")


from window.translation.scheduler import (
    TranslationQueueFullError,
    TranslationScheduler,
    TranslationService,
    TranslationSubmission,
)


__all__ = [
    "AzureTranslatorProvider",
    "DeepLTranslationProvider",
    "GoogleCloudTranslationProvider",
    "IdentityTranslationProvider",
    "LibreTranslateProvider",
    "LocalModelState",
    "MADLAD_MODEL_ID",
    "MockTranslationProvider",
    "ModelLicenseMetadata",
    "NLLB_LANGUAGE_CODES",
    "NLLB_MODEL_ID",
    "OpenAICompatibleTranslationProvider",
    "ProviderCapabilities",
    "ProviderStatus",
    "TRANSLATEGEMMA_MODEL_ID",
    "TRANSLATION_MODEL_METADATA",
    "TRANSLATION_RESULT_STATUSES",
    "TransformersTranslationProvider",
    "TranslationModelMetadata",
    "TranslationProvider",
    "TranslationProviderConfig",
    "TranslationQueueFullError",
    "TranslationRequest",
    "TranslationResult",
    "TranslationService",
    "TranslationScheduler",
    "TranslationSubmission",
    "create_translation_provider",
    "translation_source_hash",
]
