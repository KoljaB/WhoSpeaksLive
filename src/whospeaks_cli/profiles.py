"""Profile model and persistence for the WhoSpeaks command-line applications.

This module is intentionally independent from command execution and user-interface
toolkits. A profile is the persisted launch configuration; callers should construct
a new validated value with :meth:`Profile.with_updates` and only then replace their
current snapshot.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

from embeddings.provider_identity import (
    FAST_LIVE_PROVIDER,
    PROMOTED_LIVE_PROVIDER,
    PROMOTED_PUBLIC_PROVIDER,
    PUBLIC_PROVIDER,
    SINGLE_ESPNET_PROVIDER,
    SPEECHBRAIN_ECAPA_PROVIDER,
)
from window.language_config import normalize_language_code
from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
)


DEFAULT_REMOTE_ASR_URL = "http://127.0.0.1:8650"
DEFAULT_MACOS_ASR_URL = "http://127.0.0.1:8651"
DEFAULT_REMOTE_EMBEDDINGS_URL = "http://127.0.0.1:8660"
SMOKE_PROVIDER = SPEECHBRAIN_ECAPA_PROVIDER

TRANSLATION_PROVIDER_OPTIONS: dict[str, dict[str, str]] = {
    "sidecar": {"label": "Local sidecar", "default_base_url": "", "default_api_key_env": ""},
    "transformers": {"label": "Local in live process", "default_base_url": "", "default_api_key_env": ""},
    "deepl": {"label": "DeepL", "default_base_url": "https://api-free.deepl.com/v2", "default_api_key_env": "DEEPL_API_KEY"},
    "google_cloud": {"label": "Google Cloud", "default_base_url": "https://translation.googleapis.com/language/translate/v2", "default_api_key_env": "GOOGLE_TRANSLATE_API_KEY"},
    "azure_translator": {"label": "Azure Translator", "default_base_url": "https://api.cognitive.microsofttranslator.com", "default_api_key_env": "AZURE_TRANSLATOR_KEY"},
    "libretranslate": {"label": "LibreTranslate", "default_base_url": "http://127.0.0.1:5000", "default_api_key_env": "LIBRETRANSLATE_API_KEY"},
    "reports_llm": {"label": "Meeting Intelligence LLM", "default_base_url": "", "default_api_key_env": ""},
    "openai_compatible": {"label": "OpenAI-compatible", "default_base_url": "", "default_api_key_env": ""},
}


def provider_preset_label(preset_id: str, preset: "ProviderPreset") -> str:
    """Describe an embedding stack by its user-visible tradeoff."""

    return {
        "smoke": "Low VRAM - SpeechBrain ECAPA",
        "single_espnet": "Single model - ESPnet ECAPA",
        "smoke_fast_live": "Low VRAM final + fast live",
        "public_quality": "High quality - public ensemble",
        "promoted_public": "Recommended - public ensemble",
    }.get(preset_id, preset.name)


EDITABLE_PROFILE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("mode", "Profile mode", "local GPU, CPU-only, remote, or server. Mode also aligns the ASR and embeddings backends."),
    ("deployment_target", "Managed deployment", "Blank for standard profiles or macos for managed Apple Silicon services."),
    ("language", "Language", "Shared by final ASR, realtime preview model selection, and sentence splitting."),
    ("provider_preset", "Provider preset", "Named final/live speaker embedding stack, or custom."),
    ("embedding_provider", "Final provider", "Exact provider string used for committed speaker assignment."),
    ("live_speaker_embedding_provider", "Live provider", "Exact provider string used for live speaker feedback."),
    ("live_speaker_assignment", "Live speaker labels", "Show provisional speaker labels while speech is in progress."),
    ("embedding_python", "Embedding helper Python", "Optional Python executable for local speaker-embedding helper subprocesses."),
    ("embedding_device", "Embedding device", "Device used by the local speaker-embedding helper."),
    ("realtime_preview_engine", "Realtime text engine", "Use sherpa_onnx for Nemotron, kroko_onnx for Kroko/Banafo, or off."),
    ("realtime_preview_model_preset", "Realtime model preset", "Nemotron: 560ms stable or 160ms low-latency. Kroko: a Kroko model preset."),
    ("realtime_preview_model_dir", "Nemotron model folder", "Optional explicit folder for the unpacked sherpa-onnx/Nemotron model."),
    ("cpu_alignment_model", "CPU alignment model", "Whisper model used only to align the fixed Kroko/Nemotron transcript."),
    ("cpu_alignment_threads", "CPU alignment threads", "Worker threads reserved for final word alignment."),
    ("realtime_preview_python", "Realtime preview Python", "Optional Python executable for the Kroko realtime worker. Nemotron always uses the current WhoSpeaks environment."),
    ("reports_enabled", "Start Meeting Intelligence", "Open the Reports + Ask service whenever the live window launches."),
    ("reports_port", "Meeting Intelligence port", "Port for Reports, hybrid search, and session chat."),
    ("report_language", "Report language", "Blank follows the live transcription language; otherwise use a WhoSpeaks language code."),
    ("report_llm_provider", "Meeting Intelligence LLM provider", "Shared by reports and session chat: llama_cpp, ollama, lm_studio, openai_compatible, openai, or openrouter."),
    ("report_llm_base_url", "Meeting Intelligence LLM base URL", "Optional OpenAI-compatible endpoint override shared by reports and session chat."),
    ("report_llm_model", "Meeting Intelligence LLM model", "Model ID shared by report generation, evidence selection, and grounded answers."),
    ("text_embedding_base_url", "Text embedding base URL", "OpenAI-compatible endpoint used to index transcript text."),
    ("text_embedding_model", "Text embedding model", "Embedding-capable model used for semantic session search."),
    ("text_embedding_api_key_env", "Text embedding API-key variable", "Environment-variable name containing the embedding provider secret."),
    ("report_auto_generate", "Auto-generate reports", "Generate a report automatically when a newly saved meeting session is finalized."),
    ("translation_enabled", "Enable translation", "Translate stable transcript sentences without changing the original transcript."),
    ("translation_browser_preferred", "Prefer Chrome translation", "Use Chrome's on-device Translator API first and the selected provider as fallback."),
    ("translation_provider", "Translation provider", "Local, managed API, LibreTranslate, reports LLM, or OpenAI-compatible backend."),
    ("translation_port", "Translation server port", "Port used by the optional local translation sidecar."),
    ("translation_target_languages", "Translation targets", "Comma-separated WhoSpeaks language codes; the browser can change these while running."),
    ("translation_max_targets", "Maximum translation targets", "Capacity guard for simultaneous target languages."),
    ("translation_model_profile", "Translation model profile", "translate-gemma-4b, nllb-200-600m, or madlad-400-3b."),
    ("translation_model", "Translation model override", "Optional Hugging Face or OpenAI-compatible model ID override."),
    ("translation_base_url", "Translation base URL", "Optional provider endpoint override."),
    ("translation_api_key_env", "Translation API-key variable", "Environment-variable name containing the selected provider secret."),
    ("translation_region", "Translation provider region", "Optional region, currently used by Azure Translator."),
    ("translation_python", "Translation sidecar Python", "Optional Python executable for an isolated translation environment."),
    ("translation_device", "Translation device", "auto, cuda, or cpu for an in-process/sidecar local model."),
    ("asr_backend", "ASR backend", "local faster-whisper, remote service, or CPU streaming ASR."),
    ("embeddings_backend", "Embeddings backend", "local or remote."),
    ("remote_asr_url", "Remote ASR URL", "Base URL for a remote faster-whisper service."),
    ("remote_embeddings_url", "Remote embeddings URL", "Base URL for a remote voice embeddings service."),
    ("model", "ASR model", "Final ASR model name, for example large-v2."),
    ("device", "Device", "auto, cuda, or cpu."),
    ("compute_type", "Compute type", "faster-whisper compute type, for example float16 or int8."),
    ("vad_backend", "VAD backend", "Voice activity detector used for sentence-window finalization."),
    ("host", "Browser host", "Interface for the browser UI server."),
    ("port", "Browser port", "Port for the browser UI server."),
    ("advanced_args", "Advanced args", "Extra whospeaks-window flags appended after the saved profile."),
)


@dataclasses.dataclass(frozen=True)
class ProviderPreset:
    id: str
    name: str
    summary: str
    details: str
    embedding_provider: str
    live_speaker_embedding_provider: str
    requirements: str = ""
    score_note: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "smoke": ProviderPreset(
        id="smoke",
        name="First start",
        summary="Simple setup check. Fastest to try, not the highest-accuracy setting.",
        details="Uses the SpeechBrain ECAPA provider for both final sentence assignment and live speaker feedback.",
        embedding_provider=SMOKE_PROVIDER,
        live_speaker_embedding_provider=SMOKE_PROVIDER,
        score_note="Baseline smoke setting. Use it to prove installation and media flow before comparing quality.",
    ),
    "single_espnet": ProviderPreset(
        id="single_espnet",
        name="Single ESPnet",
        summary="One ESPnet embedding provider for both final and live speaker assignment.",
        details="Useful when validating one provider in isolation. It does not use the weighted multi-provider stacks.",
        embedding_provider=SINGLE_ESPNET_PROVIDER,
        live_speaker_embedding_provider=SINGLE_ESPNET_PROVIDER,
        score_note="Single-provider option. Keep score claims separate from the mixed-provider stacks.",
    ),
    "smoke_fast_live": ProviderPreset(
        id="smoke_fast_live",
        name="Smoke final + fast live",
        summary="Keeps the simple final provider and uses the faster live speaker stack.",
        details=(
            "Final assignment stays on SpeechBrain ECAPA. Live feedback uses the pyannote/wespeaker ONNX "
            "stack recommended for responsive live speaker tags."
        ),
        embedding_provider=SMOKE_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Useful when final quality is not the test target and live feedback latency is.",
    ),
    "public_quality": ProviderPreset(
        id="public_quality",
        name="Public high quality",
        summary="Public multi-provider final stack plus fast live speaker feedback.",
        details=(
            "Uses the documented public stack with ESPnet, WeSpeaker CAM++, SpeechBrain ResNet, and "
            "Resemblyzer. All providers are available through the public setup path."
        ),
        embedding_provider=PUBLIC_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Public quality candidate for reproducible comparisons.",
    ),
    "promoted_public": ProviderPreset(
        id="promoted_public",
        name="Promoted public stack",
        summary="Current promoted public final stack plus the promoted live provider.",
        details=(
            "Matches the current whospeaks-window default final provider stack and uses SpeechBrain ResNet "
            "for the live profiles and shifting-window probes."
        ),
        embedding_provider=PROMOTED_PUBLIC_PROVIDER,
        live_speaker_embedding_provider=PROMOTED_LIVE_PROVIDER,
        score_note="Current promoted public default. Keep this and public_quality visible until validation decides the winner.",
    ),
}
PROVIDER_PRESET_CHOICES = tuple(PROVIDER_PRESETS.keys()) + ("custom",)


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "local").strip().lower().replace("-", "_")
    value = {
        "all_in_one": "local",
        "full_local": "local",
        "cpu_only": "cpu",
        "local_cpu": "cpu",
        "controller_remote": "remote",
        "gpu_server": "server",
    }.get(value, value)
    return value if value in {"auto", "local", "cpu", "remote", "server"} else "local"


def normalize_provider_preset_id(value: str | None) -> str:
    normalized = str(value or "custom").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = {
        "first_start": "smoke",
        "speechbrain": "smoke",
        "smoke_provider": "smoke",
        "espnet": "single_espnet",
        "single_espnet_provider": "single_espnet",
        "fast_live": "smoke_fast_live",
        "public": "public_quality",
        "public_high_quality": "public_quality",
        "high_quality": "public_quality",
        "promoted": "promoted_public",
    }.get(normalized, normalized)
    return normalized if normalized in PROVIDER_PRESETS or normalized == "custom" else "custom"


def infer_provider_preset_id(current: str | None, embedding_provider: str, live_provider: str) -> str:
    normalized = normalize_provider_preset_id(current)
    if normalized in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[normalized]
        if embedding_provider == preset.embedding_provider and live_provider == preset.live_speaker_embedding_provider:
            return normalized
    for preset_id, preset in PROVIDER_PRESETS.items():
        if embedding_provider == preset.embedding_provider and live_provider == preset.live_speaker_embedding_provider:
            return preset_id
    return "custom"


@dataclasses.dataclass
class Profile:
    """Validated persisted settings.

    The class remains assignable for compatibility with existing integrations,
    but application code should treat instances as snapshots and use
    :meth:`with_updates` instead of changing fields in place.
    """

    mode: str = "local"
    deployment_target: str = ""
    host: str = "127.0.0.1"
    port: int = 8796
    language: str = "en"
    model: str = "large-v2"
    device: str = "auto"
    compute_type: str = "float16"
    asr_backend: str = "local"
    embeddings_backend: str = "local"
    provider_preset: str = "smoke"
    remote_asr_url: str = DEFAULT_REMOTE_ASR_URL
    remote_embeddings_url: str = DEFAULT_REMOTE_EMBEDDINGS_URL
    embedding_provider: str = SMOKE_PROVIDER
    live_speaker_embedding_provider: str = SMOKE_PROVIDER
    live_speaker_assignment: bool = True
    embedding_python: str = ""
    embedding_device: str = "cuda"
    vad_backend: str = "silero"
    realtime_preview_engine: str = "sherpa_onnx"
    realtime_preview_model_preset: str = "nemotron-3.5-560ms-int8"
    realtime_preview_model_dir: str = ""
    cpu_alignment_model: str = "base"
    cpu_alignment_threads: int = 2
    realtime_preview_python: str = ""
    reports_enabled: bool = False
    reports_port: int = 8798
    report_language: str = ""
    report_llm_provider: str = "llama_cpp"
    report_llm_base_url: str = ""
    report_llm_model: str = ""
    text_embedding_base_url: str = ""
    text_embedding_model: str = ""
    text_embedding_api_key_env: str = ""
    report_auto_generate: bool = True
    translation_enabled: bool = False
    translation_browser_preferred: bool = False
    translation_provider: str = "sidecar"
    translation_port: int = 8799
    translation_target_languages: str = ""
    translation_max_targets: int = 4
    translation_model_profile: str = "translate-gemma-4b"
    translation_model: str = ""
    translation_base_url: str = ""
    translation_api_key_env: str = ""
    translation_region: str = ""
    translation_python: str = ""
    translation_device: str = "auto"
    advanced_args: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Profile":
        allowed = {field.name for field in dataclasses.fields(cls)}
        profile = cls(**{key: item for key, item in value.items() if key in allowed})
        object.__setattr__(profile, "port", int(profile.port))
        object.__setattr__(profile, "reports_port", int(profile.reports_port))
        object.__setattr__(profile, "translation_port", int(profile.translation_port))
        object.__setattr__(
            profile,
            "translation_max_targets",
            min(16, max(1, int(profile.translation_max_targets))),
        )
        object.__setattr__(profile, "mode", normalize_mode(profile.mode))
        object.__setattr__(
            profile,
            "deployment_target",
            "macos" if str(profile.deployment_target).strip().lower() == "macos" else "",
        )
        if profile.deployment_target == "macos":
            object.__setattr__(profile, "mode", "remote")
            object.__setattr__(profile, "asr_backend", "remote")
            object.__setattr__(profile, "embeddings_backend", "remote")
            object.__setattr__(profile, "remote_asr_url", DEFAULT_MACOS_ASR_URL)
            object.__setattr__(profile, "remote_embeddings_url", DEFAULT_REMOTE_EMBEDDINGS_URL)
        try:
            object.__setattr__(profile, "language", normalize_language_code(profile.language))
        except ValueError:
            object.__setattr__(profile, "language", "en")
        if profile.mode == "cpu":
            object.__setattr__(profile, "asr_backend", "cpu")
            object.__setattr__(profile, "embeddings_backend", "local")
            object.__setattr__(profile, "device", "cpu")
            object.__setattr__(profile, "compute_type", "int8")
            object.__setattr__(profile, "embedding_device", "cpu")
            object.__setattr__(profile, "provider_preset", "smoke")
            object.__setattr__(profile, "embedding_provider", SMOKE_PROVIDER)
            object.__setattr__(profile, "live_speaker_embedding_provider", SMOKE_PROVIDER)
        if (
            normalize_provider_preset_id(profile.provider_preset) == "promoted_public"
            and profile.embedding_provider == PROMOTED_PUBLIC_PROVIDER
            and profile.live_speaker_embedding_provider == FAST_LIVE_PROVIDER
        ):
            object.__setattr__(profile, "live_speaker_embedding_provider", PROMOTED_LIVE_PROVIDER)
        object.__setattr__(
            profile,
            "provider_preset",
            infer_provider_preset_id(
                profile.provider_preset,
                profile.embedding_provider,
                profile.live_speaker_embedding_provider,
            ),
        )
        if profile.mode in {"remote", "local"}:
            object.__setattr__(profile, "asr_backend", profile.mode)
            object.__setattr__(profile, "embeddings_backend", profile.mode)
        if profile.embedding_device not in {"auto", "cuda", "cpu"}:
            object.__setattr__(profile, "embedding_device", "cuda")
        object.__setattr__(profile, "cpu_alignment_threads", max(1, min(4, int(profile.cpu_alignment_threads))))
        try:
            object.__setattr__(
                profile,
                "realtime_preview_engine",
                normalize_preview_engine(profile.realtime_preview_engine),
            )
        except ValueError:
            object.__setattr__(profile, "realtime_preview_engine", "off")
        if profile.realtime_preview_engine in {"kroko_onnx", "sherpa_onnx"}:
            default_preset = get_preview_backend_spec(profile.realtime_preview_engine).default_preset or ""
            try:
                object.__setattr__(
                    profile,
                    "realtime_preview_model_preset",
                    normalize_preview_model_preset(
                        profile.realtime_preview_engine,
                        profile.realtime_preview_model_preset or default_preset,
                    ),
                )
            except (ValueError, argparse.ArgumentTypeError):
                object.__setattr__(profile, "realtime_preview_model_preset", default_preset)
        else:
            object.__setattr__(profile, "realtime_preview_model_preset", "")
        if profile.realtime_preview_engine != "sherpa_onnx":
            object.__setattr__(profile, "realtime_preview_model_dir", "")
        object.__setattr__(
            profile,
            "report_language",
            normalize_language_code(profile.report_language) if profile.report_language else "",
        )
        object.__setattr__(
            profile,
            "report_llm_provider",
            str(profile.report_llm_provider or "llama_cpp").strip().lower().replace("-", "_"),
        )
        if profile.report_llm_provider not in {"llama_cpp", "ollama", "lm_studio", "openai_compatible", "openai", "openrouter"}:
            object.__setattr__(profile, "report_llm_provider", "llama_cpp")
        object.__setattr__(
            profile,
            "translation_provider",
            str(profile.translation_provider or "sidecar").strip().lower().replace("-", "_"),
        )
        if profile.translation_provider not in {
            "sidecar", "transformers", "reports_llm", "openai_compatible", "deepl", "google_cloud",
            "azure_translator", "libretranslate",
        }:
            object.__setattr__(profile, "translation_provider", "sidecar")
        if profile.translation_model_profile not in {"translate-gemma-4b", "nllb-200-600m", "madlad-400-3b"}:
            object.__setattr__(profile, "translation_model_profile", "translate-gemma-4b")
        if profile.translation_device not in {"auto", "cuda", "cpu"}:
            object.__setattr__(profile, "translation_device", "auto")
        normalized_targets: list[str] = []
        for raw_target in re.split(r"[,;\s]+", str(profile.translation_target_languages or "")):
            if not raw_target:
                continue
            try:
                target = normalize_language_code(raw_target)
            except ValueError:
                continue
            if target != profile.language and target not in normalized_targets:
                normalized_targets.append(target)
        object.__setattr__(
            profile,
            "translation_target_languages",
            ",".join(normalized_targets[:profile.translation_max_targets]),
        )
        return profile

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def with_updates(self, **updates: Any) -> "Profile":
        """Return a validated profile snapshot with ``updates`` applied."""

        return type(self).from_mapping({**self.as_dict(), **updates})


class ProfileLoadError(ValueError):
    """An existing profile could not be loaded without changing its meaning."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"WhoSpeaks could not use the saved profile at {path}: {detail} "
            "Fix that file or explicitly reset it with `whospeaks config --reset`."
        )


_INTEGER_PROFILE_FIELDS = {
    "port",
    "reports_port",
    "translation_port",
    "translation_max_targets",
    "cpu_alignment_threads",
}
_BOOLEAN_PROFILE_FIELDS = {
    "live_speaker_assignment",
    "reports_enabled",
    "report_auto_generate",
    "translation_enabled",
    "translation_browser_preferred",
}


def _profile_from_saved_mapping(value: dict[str, Any], path: Path) -> Profile:
    """Load a persisted profile only when normalization would preserve every saved value."""

    fields = {field.name for field in dataclasses.fields(Profile)}
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ProfileLoadError(path, f"unknown setting(s): {', '.join(unknown)}")
    for field, raw_value in value.items():
        if field in _INTEGER_PROFILE_FIELDS:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ProfileLoadError(path, f"{field} must be stored as an integer")
        elif field in _BOOLEAN_PROFILE_FIELDS:
            if not isinstance(raw_value, bool):
                raise ProfileLoadError(path, f"{field} must be stored as true or false")
        elif not isinstance(raw_value, str):
            raise ProfileLoadError(path, f"{field} must be stored as text")
    migrated_value = dict(value)
    if (
        normalize_provider_preset_id(str(migrated_value.get("provider_preset", ""))) == "promoted_public"
        and migrated_value.get("embedding_provider") == PROMOTED_PUBLIC_PROVIDER
        and migrated_value.get("live_speaker_embedding_provider") == FAST_LIVE_PROVIDER
    ):
        migrated_value["live_speaker_embedding_provider"] = PROMOTED_LIVE_PROVIDER
    try:
        profile = Profile.from_mapping(migrated_value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ProfileLoadError(path, str(exc)) from exc
    changed = [
        field
        for field, raw_value in migrated_value.items()
        if getattr(profile, field) != raw_value
    ]
    if changed:
        detail = ", ".join(
            f"{field}={migrated_value[field]!r} is not valid (would become {getattr(profile, field)!r})"
            for field in changed
        )
        raise ProfileLoadError(path, detail)
    return profile


def profile_with_provider_preset(profile: Profile, preset_id: str) -> Profile:
    normalized = normalize_provider_preset_id(preset_id)
    if normalized == "custom":
        return profile.with_updates(provider_preset="custom")
    preset = PROVIDER_PRESETS[normalized]
    return profile.with_updates(
        provider_preset=preset.id,
        embedding_provider=preset.embedding_provider,
        live_speaker_embedding_provider=preset.live_speaker_embedding_provider,
    )


def apply_provider_preset(profile: Profile, preset_id: str) -> Profile:
    """Compatibility wrapper that updates ``profile`` while returning it.

    New code should use :func:`profile_with_provider_preset` and replace its
    current snapshot.
    """

    updated = profile_with_provider_preset(profile, preset_id)
    for field in dataclasses.fields(Profile):
        setattr(profile, field.name, getattr(updated, field.name))
    return profile


def selected_provider_preset(profile: Profile) -> ProviderPreset | None:
    return PROVIDER_PRESETS.get(
        infer_provider_preset_id(
            profile.provider_preset,
            profile.embedding_provider,
            profile.live_speaker_embedding_provider,
        )
    )


def config_path() -> Path:
    override = os.environ.get("WHOSPEAKS_CONFIG")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return root / "WhoSpeaks" / "config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "whospeaks" / "config.json"


def local_config_path() -> Path:
    return Path.cwd() / ".whospeaks" / "config.json"


def config_read_candidates() -> list[Path]:
    if os.environ.get("WHOSPEAKS_CONFIG"):
        return [config_path()]
    primary = config_path()
    fallback = local_config_path()
    return [primary] if primary == fallback else [primary, fallback]


def load_profile(path: Path | None = None) -> Profile:
    for selected in ([path] if path is not None else config_read_candidates()):
        if selected is None:
            continue
        try:
            data = json.loads(selected.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError) as exc:
            raise ProfileLoadError(selected, str(exc)) from exc
        if not isinstance(data, dict):
            raise ProfileLoadError(selected, "the JSON root must be an object")
        return _profile_from_saved_mapping(data, selected)
    return Profile()


def save_profile(profile: Profile, path: Path | None = None) -> Path:
    selected = path or config_path()
    payload = json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n"
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(payload, encoding="utf-8")
    return selected


def update_profile_in_place(profile: Profile, updated: Profile) -> Profile:
    """Compatibility adapter for classic menus that retain a profile object."""

    for field in dataclasses.fields(Profile):
        setattr(profile, field.name, getattr(updated, field.name))
    return profile
