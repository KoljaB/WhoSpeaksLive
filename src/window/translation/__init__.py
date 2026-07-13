"""Translation domain package with lazy compatibility exports."""

from __future__ import annotations

from typing import Any


_PUBLIC_NAMES = frozenset({
    "TranslationProvider",
    "TranslationRequest",
    "TranslationResult",
    "TranslationService",
    "TranslationScheduler",
    "TranslationSubmission",
    "TranslationQueueFullError",
    "create_translation_provider",
    "LocalModelState",
})
__all__ = sorted(_PUBLIC_NAMES)


def __getattr__(name: str) -> Any:
    if name == "LocalModelState":
        from window.translation.local_runtime import LocalModelState
        return LocalModelState
    if name not in _PUBLIC_NAMES:
        raise AttributeError(name)
    from window import translation_service
    return getattr(translation_service, name)
