"""Backend capabilities and shared argument defaults for realtime preview text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from window.language_config import is_kroko_preview_language, kroko_preview_model_name
from window.sherpa_onnx_models import (
    DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET,
    normalize_sherpa_onnx_preview_model_preset,
    sherpa_onnx_model_preset,
)


@dataclass(frozen=True)
class PreviewLanguageSupport:
    stream_code: str
    locale: str
    tier: str


NEMOTRON_LANGUAGE_SUPPORT = {
    "en": PreviewLanguageSupport("en", "en-US", "transcription-ready"),
    "de": PreviewLanguageSupport("de", "de-DE", "transcription-ready"),
    "es": PreviewLanguageSupport("es", "es-ES", "transcription-ready"),
    "fr": PreviewLanguageSupport("fr", "fr-FR", "transcription-ready"),
    "it": PreviewLanguageSupport("it", "it-IT", "transcription-ready"),
    "nl": PreviewLanguageSupport("nl", "nl-NL", "transcription-ready"),
    "pt": PreviewLanguageSupport("pt", "pt-PT", "transcription-ready"),
    "tr": PreviewLanguageSupport("tr", "tr-TR", "transcription-ready"),
    "sv": PreviewLanguageSupport("sv", "sv-SE", "broad-coverage"),
}


@dataclass(frozen=True)
class PreviewBackendSpec:
    engine: str
    display_name: str
    model_location: str
    default_preset: str | None


PREVIEW_BACKENDS = {
    "kroko_onnx": PreviewBackendSpec("kroko_onnx", "Kroko/Banafo via sherpa-onnx", "directory", "community-64l"),
    "sherpa_onnx": PreviewBackendSpec(
        "sherpa_onnx",
        "Nemotron 3.5 via sherpa-onnx",
        "directory",
        DEFAULT_SHERPA_ONNX_PREVIEW_MODEL_PRESET,
    ),
}
PREVIEW_ENGINE_ALIASES = {
    "": "off",
    "none": "off",
    "false": "off",
    "kroko": "kroko_onnx",
    "nemotron": "sherpa_onnx",
    "nemotron_3_5": "sherpa_onnx",
    "sherpa": "sherpa_onnx",
}


def normalize_preview_engine(value: object) -> str:
    engine = str(value or "off").strip().lower().replace("-", "_")
    engine = PREVIEW_ENGINE_ALIASES.get(engine, engine)
    if engine not in {"off", "mock", *PREVIEW_BACKENDS}:
        allowed = ", ".join(("off", "mock", *PREVIEW_BACKENDS))
        raise ValueError(f"invalid realtime preview engine {value!r}; choose one of: {allowed}")
    return engine


def get_preview_backend_spec(value: object) -> PreviewBackendSpec:
    engine = normalize_preview_engine(value)
    if engine not in PREVIEW_BACKENDS:
        raise ValueError(f"{engine!r} has no model-backed preview backend specification")
    return PREVIEW_BACKENDS[engine]


def preview_language_support(engine: object, language: object) -> PreviewLanguageSupport | None:
    normalized = normalize_preview_engine(engine)
    code = str(language or "").strip().lower()
    if normalized == "sherpa_onnx":
        return NEMOTRON_LANGUAGE_SUPPORT.get(code)
    if normalized == "kroko_onnx" and is_kroko_preview_language(code):
        return PreviewLanguageSupport(code, code, "supported")
    return None


def preview_language_error(engine: object, language: object) -> str | None:
    normalized = normalize_preview_engine(engine)
    if normalized in {"off", "mock"}:
        return None
    if preview_language_support(normalized, language) is not None:
        return None
    if normalized == "sherpa_onnx":
        return (
            f"{language!r} is not supported by the Nemotron 3.5 realtime preview backend; "
            "choose Kroko, disable preview, or select en, de, es, fr, it, nl, pt, tr, or sv."
        )
    return (
        f"{language!r} is supported for final ASR and sentence splitting, but not for Kroko realtime preview; "
        "use --realtime-preview-engine off or choose a Kroko preview language."
    )


def recommended_preview_engine(language: object) -> str:
    """Return the preferred realtime engine for a language, or off when none supports it."""

    for engine in ("sherpa_onnx", "kroko_onnx"):
        if preview_language_support(engine, language) is not None:
            return engine
    return "off"


def normalize_preview_model_preset(engine: object, value: object) -> str:
    normalized = normalize_preview_engine(engine)
    if normalized == "sherpa_onnx":
        return normalize_sherpa_onnx_preview_model_preset(value)
    if normalized == "kroko_onnx":
        from window.window_config import normalize_kroko_preview_model_preset

        return normalize_kroko_preview_model_preset(value)
    return ""


def default_preview_model(engine: object, language: object, preset: object) -> str:
    normalized = normalize_preview_engine(engine)
    if normalized == "sherpa_onnx":
        return normalize_sherpa_onnx_preview_model_preset(preset)
    if normalized == "kroko_onnx":
        return kroko_preview_model_name(str(language), str(preset))
    return ""


def apply_preview_timing_defaults(args: Any) -> None:
    engine = normalize_preview_engine(args.realtime_preview_engine)
    if engine == "sherpa_onnx":
        model = sherpa_onnx_model_preset(args.realtime_preview_model_preset)
        defaults = {
            "realtime_preview_interval_seconds": model.recommended_interval_seconds,
            "realtime_preview_min_audio_seconds": model.recommended_min_audio_seconds,
            "realtime_preview_min_advance_seconds": model.recommended_feed_seconds,
            "realtime_preview_feed_chunk_seconds": model.recommended_feed_seconds,
        }
    else:
        from window.window_preview import infer_kroko_preview_chunk_seconds

        chunk = infer_kroko_preview_chunk_seconds(
            getattr(args, "realtime_preview_model_path", None) or args.realtime_preview_model
        )
        defaults = {
            "realtime_preview_interval_seconds": chunk,
            "realtime_preview_min_audio_seconds": chunk,
            "realtime_preview_min_advance_seconds": chunk,
            "realtime_preview_feed_chunk_seconds": chunk,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
