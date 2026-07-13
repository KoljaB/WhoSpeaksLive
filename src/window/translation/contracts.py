"""Immutable translation contracts, metadata, and provider protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Literal


LOGGER = logging.getLogger(__name__)

TRANSLATION_RESULT_STATUSES = ("completed", "error", "superseded", "cancelled")
TranslationResultStatus = Literal["completed", "error", "superseded", "cancelled"]


def _normalize_language_tag(value: object) -> str:
    tag = str(value or "").strip().replace("_", "-")
    if not tag:
        raise ValueError("language must not be empty")
    parts = tag.split("-")
    normalized = [parts[0].lower()]
    normalized.extend(part.upper() if len(part) == 2 else part for part in parts[1:])
    return "-".join(normalized)


def translation_source_hash(text: str) -> str:
    """Return the stable content hash echoed by every translation result."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelLicenseMetadata:
    identifier: str
    display_name: str
    url: str
    commercial_use: bool | None
    acceptance_required: bool = False
    notice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "url": self.url,
            "commercial_use": self.commercial_use,
            "acceptance_required": self.acceptance_required,
            "notice": self.notice,
        }


@dataclass(frozen=True)
class TranslationModelMetadata:
    model_id: str
    display_name: str
    family: str
    license: ModelLicenseMetadata
    model_card_url: str
    language_coverage: str
    intended_use_notice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "license": self.license.to_dict(),
            "model_card_url": self.model_card_url,
            "language_coverage": self.language_coverage,
            "intended_use_notice": self.intended_use_notice,
        }


NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
TRANSLATEGEMMA_MODEL_ID = "google/translategemma-4b-it"
MADLAD_MODEL_ID = "google/madlad400-3b-mt"

TRANSLATION_MODEL_METADATA: dict[str, TranslationModelMetadata] = {
    NLLB_MODEL_ID: TranslationModelMetadata(
        model_id=NLLB_MODEL_ID,
        display_name="NLLB-200 distilled 600M",
        family="nllb",
        license=ModelLicenseMetadata(
            identifier="CC-BY-NC-4.0",
            display_name="Creative Commons Attribution-NonCommercial 4.0",
            url="https://creativecommons.org/licenses/by-nc/4.0/",
            commercial_use=False,
            notice=(
                "Keep attribution and the license notice with the optional model. "
                "The weights are licensed for non-commercial use."
            ),
        ),
        model_card_url=f"https://huggingface.co/{NLLB_MODEL_ID}",
        language_coverage="196 FLORES-200 language variants",
        intended_use_notice=(
            "The model card describes NLLB as a research model for single-sentence "
            "translation and says it was not released for production deployment."
        ),
    ),
    TRANSLATEGEMMA_MODEL_ID: TranslationModelMetadata(
        model_id=TRANSLATEGEMMA_MODEL_ID,
        display_name="TranslateGemma 4B IT",
        family="translategemma",
        license=ModelLicenseMetadata(
            identifier="gemma",
            display_name="Gemma Terms of Use",
            url="https://ai.google.dev/gemma/terms",
            commercial_use=True,
            acceptance_required=True,
            notice=(
                "The user must review and accept the Gemma terms and prohibited-use "
                "policy before downloading the gated Hugging Face weights."
            ),
        ),
        model_card_url=f"https://huggingface.co/{TRANSLATEGEMMA_MODEL_ID}",
        language_coverage="55 languages; the model chat template validates each language tag",
    ),
    MADLAD_MODEL_ID: TranslationModelMetadata(
        model_id=MADLAD_MODEL_ID,
        display_name="MADLAD-400 3B MT",
        family="madlad",
        license=ModelLicenseMetadata(
            identifier="Apache-2.0",
            display_name="Apache License 2.0",
            url="https://www.apache.org/licenses/LICENSE-2.0",
            commercial_use=True,
            notice="Keep the Apache-2.0 license and required notices with the model.",
        ),
        model_card_url=f"https://huggingface.co/{MADLAD_MODEL_ID}",
        language_coverage="419 language tags in training; the model card reports evaluation on 204 languages",
    ),
}


@dataclass(frozen=True)
class ProviderCapabilities:
    local: bool
    requires_network: bool
    lazy_loading: bool
    supports_context: bool
    supports_multi_target: bool = True
    supported_language_codes: tuple[str, ...] | None = None
    max_parallel_requests: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local,
            "requires_network": self.requires_network,
            "lazy_loading": self.lazy_loading,
            "supports_context": self.supports_context,
            "supports_multi_target": self.supports_multi_target,
            "supported_language_codes": (
                None if self.supported_language_codes is None else list(self.supported_language_codes)
            ),
            "max_parallel_requests": self.max_parallel_requests,
        }


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    display_name: str
    model: str
    available: bool
    ready: bool
    detail: str
    capabilities: ProviderCapabilities
    model_metadata: TranslationModelMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "model": self.model,
            "available": self.available,
            "ready": self.ready,
            "detail": self.detail,
            "capabilities": self.capabilities.to_dict(),
            "model_metadata": None if self.model_metadata is None else self.model_metadata.to_dict(),
        }


@dataclass(frozen=True)
class TranslationRequest:
    segment_id: str
    source_text: str
    source_language: str
    target_languages: tuple[str, ...]
    source_revision: str = ""
    context: tuple[str, ...] = ()
    session_id: str = ""
    # Internal lifecycle discriminator. It is intentionally omitted from the
    # public JSON result shape so existing HTTP and persistence schemas remain
    # stable while same-ID session restarts can reject stale work.
    session_epoch: int = 0

    def __post_init__(self) -> None:
        segment_id = str(self.segment_id or "").strip()
        if not segment_id:
            raise ValueError("segment_id must not be empty")
        source_text = str(self.source_text)
        source_language = _normalize_language_tag(self.source_language)
        targets: list[str] = []
        seen: set[str] = set()
        target_values: Iterable[object]
        if isinstance(self.target_languages, str):
            target_values = (self.target_languages,)
        else:
            target_values = self.target_languages
        for value in target_values:
            target = _normalize_language_tag(value)
            if target not in seen:
                seen.add(target)
                targets.append(target)
        if not targets:
            raise ValueError("target_languages must contain at least one language")
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "source_language", source_language)
        object.__setattr__(self, "target_languages", tuple(targets))
        object.__setattr__(self, "source_revision", str(self.source_revision or ""))
        context_values = (self.context,) if isinstance(self.context, str) else self.context
        object.__setattr__(self, "context", tuple(str(item) for item in context_values if str(item)))
        object.__setattr__(self, "session_id", str(self.session_id or ""))
        object.__setattr__(self, "session_epoch", max(0, int(self.session_epoch)))

    @property
    def source_hash(self) -> str:
        return translation_source_hash(self.source_text)


@dataclass(frozen=True)
class TranslationResult:
    segment_id: str
    session_id: str
    source_hash: str
    source_revision: str
    source_language: str
    target_language: str
    translated_text: str
    provider: str
    model: str
    status: TranslationResultStatus
    latency_seconds: float
    cached: bool
    session_epoch: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "source_hash": self.source_hash,
            "source_revision": self.source_revision,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "translated_text": self.translated_text,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "cached": self.cached,
            "error": self.error,
        }


TranslationCallback = Callable[[TranslationResult], None]


class TranslationProvider(ABC):
    """Synchronous provider interface executed by ``TranslationService`` workers."""

    provider_id = "provider"
    display_name = "Translation provider"
    model_id = ""
    capabilities = ProviderCapabilities(
        local=False,
        requires_network=False,
        lazy_loading=False,
        supports_context=False,
    )

    @property
    def cache_identity(self) -> str:
        return f"{self.provider_id}:{self.model_id}"

    def supports_language_pair(self, source_language: str, target_language: str) -> bool:
        supported = self.capabilities.supported_language_codes
        if supported is None:
            return True
        source = _normalize_language_tag(source_language).split("-", 1)[0]
        target = _normalize_language_tag(target_language).split("-", 1)[0]
        return source in supported and target in supported

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider_id,
            display_name=self.display_name,
            model=self.model_id,
            available=True,
            ready=True,
            detail="ready",
            capabilities=self.capabilities,
            model_metadata=TRANSLATION_MODEL_METADATA.get(self.model_id),
        )

    def warmup(self) -> None:
        """Prepare provider resources before the provider is advertised as ready."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        context: Sequence[str] = (),
    ) -> str:
        """Translate one stable source sentence and return only target text."""

    def close(self) -> None:
        """Release provider resources, if any."""
