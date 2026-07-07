"""Language normalization and realtime ASR/sentence-tokenizer defaults."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
from typing import Any


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    display_name: str
    kroko_code: str
    sentence_tokenizer: str
    sentence_language: str
    nltk_language: str | None
    aliases: tuple[str, ...] = ()


SUPPORTED_LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "af": LanguageConfig("af", "Afrikaans", "", "stanza", "af", None),
    "ar": LanguageConfig("ar", "Arabic", "", "stanza", "ar", None),
    "be": LanguageConfig("be", "Belarusian", "", "stanza", "be", None),
    "bg": LanguageConfig("bg", "Bulgarian", "", "stanza", "bg", None),
    "ca": LanguageConfig("ca", "Catalan", "", "stanza", "ca", None),
    "cs": LanguageConfig("cs", "Czech", "", "nltk+rule-based", "cs", "czech"),
    "cy": LanguageConfig("cy", "Welsh", "", "stanza", "cy", None),
    "da": LanguageConfig("da", "Danish", "", "nltk+rule-based", "da", "danish"),
    "de": LanguageConfig(
        "de", "German", "DE", "nltk+rule-based", "de", "german",
        ("deu", "ger", "german", "deutsch"),
    ),
    "el": LanguageConfig("el", "Greek", "", "nltk+rule-based", "el", "greek"),
    "en": LanguageConfig(
        "en", "English", "EN", "nltk+rule-based", "en", "english",
        ("eng", "english"),
    ),
    "es": LanguageConfig(
        "es", "Spanish", "ES", "nltk+rule-based", "es", "spanish",
        ("spa", "spanish", "espanol"),
    ),
    "et": LanguageConfig("et", "Estonian", "", "nltk+rule-based", "et", "estonian"),
    "eu": LanguageConfig("eu", "Basque", "", "stanza", "eu", None),
    "fa": LanguageConfig("fa", "Persian", "", "stanza", "fa", None, ("farsi",)),
    "fi": LanguageConfig("fi", "Finnish", "", "nltk+rule-based", "fi", "finnish"),
    "fo": LanguageConfig("fo", "Faroese", "", "stanza", "fo", None),
    "fr": LanguageConfig(
        "fr", "French", "FR", "nltk+rule-based", "fr", "french",
        ("fra", "fre", "french", "francais"),
    ),
    "gl": LanguageConfig("gl", "Galician", "", "stanza", "gl", None),
    "he": LanguageConfig(
        "he", "Hebrew", "IW", "rule-based", "he", None,
        ("iw", "heb", "hebrew", "hebraisch"),
    ),
    "hi": LanguageConfig("hi", "Hindi", "", "stanza", "hi", None),
    "hr": LanguageConfig("hr", "Croatian", "", "stanza", "hr", None),
    "hu": LanguageConfig("hu", "Hungarian", "", "stanza", "hu", None),
    "hy": LanguageConfig("hy", "Armenian", "", "stanza", "hy", None),
    "id": LanguageConfig("id", "Indonesian", "", "stanza", "id", None, ("bahasa",)),
    "is": LanguageConfig("is", "Icelandic", "", "stanza", "is", None),
    "it": LanguageConfig(
        "it", "Italian", "IT", "nltk+rule-based", "it", "italian",
        ("ita", "italian", "italiano"),
    ),
    "ja": LanguageConfig("ja", "Japanese", "", "stanza", "ja", None),
    "ka": LanguageConfig("ka", "Georgian", "", "stanza", "ka", None),
    "kk": LanguageConfig("kk", "Kazakh", "", "stanza", "kk", None),
    "ko": LanguageConfig("ko", "Korean", "", "stanza", "ko", None),
    "la": LanguageConfig("la", "Latin", "", "stanza", "la", None),
    "lt": LanguageConfig("lt", "Lithuanian", "", "stanza", "lt", None),
    "lv": LanguageConfig("lv", "Latvian", "", "stanza", "lv", None),
    "ml": LanguageConfig("ml", "Malayalam", "", "nltk+rule-based", "ml", "malayalam"),
    "mr": LanguageConfig("mr", "Marathi", "", "stanza", "mr", None),
    "mt": LanguageConfig("mt", "Maltese", "", "stanza", "mt", None),
    "my": LanguageConfig("my", "Myanmar/Burmese", "", "stanza", "my", None, ("burmese", "myanmar")),
    "nl": LanguageConfig(
        "nl", "Dutch", "NL", "nltk+rule-based", "nl", "dutch",
        ("nld", "dut", "dutch", "nederlands"),
    ),
    "nn": LanguageConfig("nn", "Norwegian Nynorsk", "", "stanza", "nn", None, ("nynorsk",)),
    "no": LanguageConfig("no", "Norwegian", "", "nltk+rule-based", "no", "norwegian", ("nb", "bokmal")),
    "pl": LanguageConfig("pl", "Polish", "", "nltk+rule-based", "pl", "polish"),
    "pt": LanguageConfig(
        "pt", "Portuguese", "PT", "nltk+rule-based", "pt", "portuguese",
        ("por", "portuguese", "portugues"),
    ),
    "ro": LanguageConfig("ro", "Romanian", "", "stanza", "ro", None),
    "ru": LanguageConfig("ru", "Russian", "", "nltk+rule-based", "ru", "russian"),
    "sa": LanguageConfig("sa", "Sanskrit", "", "stanza", "sa", None),
    "sd": LanguageConfig("sd", "Sindhi", "", "stanza", "sd", None),
    "sk": LanguageConfig("sk", "Slovak", "", "stanza", "sk", None),
    "sl": LanguageConfig("sl", "Slovenian", "", "nltk+rule-based", "sl", "slovene", ("slovene",)),
    "sq": LanguageConfig("sq", "Albanian", "", "stanza", "sq", None),
    "sr": LanguageConfig("sr", "Serbian", "", "stanza", "sr", None),
    "sv": LanguageConfig(
        "sv", "Swedish", "SV", "nltk+rule-based", "sv", "swedish",
        ("swe", "swedish", "svenska"),
    ),
    "ta": LanguageConfig("ta", "Tamil", "", "stanza", "ta", None),
    "te": LanguageConfig("te", "Telugu", "", "stanza", "te", None),
    "th": LanguageConfig("th", "Thai", "", "stanza", "th", None),
    "tr": LanguageConfig(
        "tr", "Turkish", "TR", "nltk+rule-based", "tr", "turkish",
        ("tur", "turkish", "turkisch"),
    ),
    "uk": LanguageConfig("uk", "Ukrainian", "", "stanza", "uk", None),
    "ur": LanguageConfig("ur", "Urdu", "", "stanza", "ur", None),
    "vi": LanguageConfig("vi", "Vietnamese", "", "stanza", "vi", None),
    "zh": LanguageConfig("zh", "Chinese", "", "stanza", "zh-hans", None, ("chinese", "mandarin")),
}

SUPPORTED_LANGUAGE_CODES = tuple(SUPPORTED_LANGUAGE_CONFIGS)
SUPPORTED_KROKO_CODES = tuple(config.kroko_code for config in SUPPORTED_LANGUAGE_CONFIGS.values() if config.kroko_code)
SUPPORTED_SENTENCE_TOKENIZERS = ("auto", "nltk", "nltk+rule-based", "rule-based", "stanza")

_ALIAS_TO_CODE: dict[str, str] = {}
for _code, _config in SUPPORTED_LANGUAGE_CONFIGS.items():
    _ALIAS_TO_CODE[_code] = _code
    if _config.kroko_code:
        _ALIAS_TO_CODE[_config.kroko_code.lower()] = _code
    _ALIAS_TO_CODE[_config.display_name.lower()] = _code
    for _alias in _config.aliases:
        _ALIAS_TO_CODE[_alias.lower()] = _code


def _language_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    return text


def normalize_language_code(value: Any) -> str:
    key = _language_key(value)
    if not key:
        raise ValueError("language must not be empty")
    code = _ALIAS_TO_CODE.get(key)
    if code is None:
        regional_match = re.match(r"^([a-z]{2,3})-[a-z0-9-]+$", key)
        if regional_match:
            code = _ALIAS_TO_CODE.get(regional_match.group(1))
    if code is None:
        allowed = ", ".join(SUPPORTED_LANGUAGE_CODES)
        raise ValueError(f"unsupported language {value!r}; choose one of: {allowed}")
    return code


def language_arg(value: Any) -> str:
    try:
        return normalize_language_code(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def default_language_code() -> str:
    for name in ("WHOSPEAKS_LANGUAGE", "WHOSPEAKS_ASR_LANGUAGE"):
        value = os.environ.get(name)
        if value:
            try:
                return normalize_language_code(value)
            except ValueError:
                return "en"
    return "en"


def get_language_config(value: Any) -> LanguageConfig:
    return SUPPORTED_LANGUAGE_CONFIGS[normalize_language_code(value)]


def is_kroko_preview_language(value: Any) -> bool:
    return bool(get_language_config(value).kroko_code)


def normalize_sentence_tokenizer(value: Any) -> str:
    tokenizer = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "rules": "rule-based",
        "nltk-rules": "nltk+rule-based",
        "nltk+rules": "nltk+rule-based",
        "nltk-rule-based": "nltk+rule-based",
        "consensus": "nltk+rule-based",
    }
    tokenizer = aliases.get(tokenizer, tokenizer)
    if tokenizer not in SUPPORTED_SENTENCE_TOKENIZERS:
        allowed = ", ".join(SUPPORTED_SENTENCE_TOKENIZERS)
        raise ValueError(f"unsupported sentence tokenizer {value!r}; choose one of: {allowed}")
    return tokenizer


def sentence_tokenizer_arg(value: Any) -> str:
    try:
        return normalize_sentence_tokenizer(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def default_sentence_tokenizer(language: Any, requested: Any = None) -> str:
    tokenizer = normalize_sentence_tokenizer(
        requested or os.environ.get("WHOSPEAKS_SENTENCE_TOKENIZER", "auto")
    )
    if tokenizer != "auto":
        return tokenizer
    return get_language_config(language).sentence_tokenizer


def default_sentence_language(language: Any) -> str:
    return get_language_config(language).sentence_language


def kroko_preview_model_name(language: Any, preset: str = "community-64l") -> str:
    config = get_language_config(language)
    normalized_preset = str(preset or "community-64l").strip().lower().replace("_", "-")
    if not config.kroko_code:
        raise ValueError(
            f"{config.display_name} ({config.code}) does not have a configured Kroko realtime preview model"
        )
    if normalized_preset == "community-64l":
        return f"Kroko-{config.kroko_code}-Community-64-L-Streaming-001.data"
    if normalized_preset == "pro-16l":
        if config.code != "en":
            raise ValueError("Kroko pro-16l preview preset is only configured for English.")
        return "Kroko-EN-Pro-16-L-Streaming-001.data"
    raise ValueError(f"unsupported Kroko preview preset {preset!r}")


def infer_language_from_kroko_model_name(model: Any) -> str | None:
    match = re.search(r"Kroko-([A-Z]{2})-", str(model or ""), re.IGNORECASE)
    if not match:
        return None
    try:
        return normalize_language_code(match.group(1))
    except ValueError:
        return None
