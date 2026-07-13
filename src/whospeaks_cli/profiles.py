"""Profile model and persistence for the WhoSpeaks command-line applications.

This module is intentionally independent from command execution and Textual.  A
profile is the persisted launch configuration; callers should construct a new
validated value with :meth:`Profile.with_updates` and only then replace their
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

from window.language_config import normalize_language_code
from window.realtime_preview_backends import (
    get_preview_backend_spec,
    normalize_preview_engine,
    normalize_preview_model_preset,
)


DEFAULT_REMOTE_ASR_URL = "http://127.0.0.1:8650"
DEFAULT_REMOTE_EMBEDDINGS_URL = "http://127.0.0.1:8660"
SMOKE_PROVIDER = "speechbrain_ecapa"
SINGLE_ESPNET_PROVIDER = "espnet_ecapa_wavlm_joint"
PUBLIC_PROVIDER = (
    "espnet_ecapa_wavlm_joint=0.74+wespeaker_campplus=0.34+"
    "speechbrain_resnet=0.38+resemblyzer=0.12"
)
PROMOTED_PUBLIC_PROVIDER = (
    "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
)
FAST_LIVE_PROVIDER = "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50"


EDITABLE_PROFILE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("mode", "Profile mode", "local, remote, or server. Mode also aligns the ASR and embeddings backends."),
    ("language", "Language", "Shared by final ASR, realtime preview model selection, and sentence splitting."),
    ("provider_preset", "Provider preset", "Named final/live speaker embedding stack, or custom."),
    ("embedding_provider", "Final provider", "Exact provider string used for committed speaker assignment."),
    ("live_speaker_embedding_provider", "Live provider", "Exact provider string used for live speaker feedback."),
    ("live_speaker_assignment", "Live speaker labels", "Show provisional speaker labels while speech is in progress."),
    ("embedding_python", "Embedding helper Python", "Optional Python executable for local speaker-embedding helper subprocesses."),
    ("realtime_preview_engine", "Realtime text engine", "Use sherpa_onnx for Nemotron, kroko_onnx for Kroko/Banafo, or off."),
    ("realtime_preview_model_preset", "Realtime model preset", "Nemotron: 560ms stable or 160ms low-latency. Kroko: a Kroko model preset."),
    ("realtime_preview_model_dir", "Nemotron model folder", "Optional explicit folder for the unpacked sherpa-onnx/Nemotron model."),
    ("realtime_preview_python", "Realtime preview Python", "Optional Python executable for the Kroko realtime worker. Nemotron always uses the current WhoSpeaks environment."),
    ("reports_enabled", "Start reports with live window", "Open the meeting-intelligence server in a second terminal whenever the live window launches."),
    ("reports_port", "Reports browser port", "Port for the meeting-intelligence browser UI."),
    ("report_language", "Report language", "Blank follows the live transcription language; otherwise use a WhoSpeaks language code."),
    ("report_llm_provider", "Reports LLM provider", "llama_cpp, ollama, lm_studio, openai, or openrouter."),
    ("report_llm_base_url", "Reports LLM base URL", "Optional override for the report LLM OpenAI-compatible base URL."),
    ("report_llm_model", "Reports LLM model", "Optional model ID used for report generation."),
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
    ("asr_backend", "ASR backend", "local or remote."),
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
        summary="Current promoted public final stack plus fast live speaker feedback.",
        details=(
            "Matches the current whospeaks-window default final provider stack. Keep it separate from "
            "public_quality until validation confirms which public stack scores higher for the target data."
        ),
        embedding_provider=PROMOTED_PUBLIC_PROVIDER,
        live_speaker_embedding_provider=FAST_LIVE_PROVIDER,
        score_note="Current promoted public default. Keep this and public_quality visible until validation decides the winner.",
    ),
}
PROVIDER_PRESET_CHOICES = tuple(PROVIDER_PRESETS.keys()) + ("custom",)


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "local").strip().lower().replace("-", "_")
    value = {
        "all_in_one": "local",
        "full_local": "local",
        "controller_remote": "remote",
        "gpu_server": "server",
    }.get(value, value)
    return value if value in {"auto", "local", "remote", "server"} else "local"


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
    vad_backend: str = "rms"
    realtime_preview_engine: str = "sherpa_onnx"
    realtime_preview_model_preset: str = "nemotron-3.5-560ms-int8"
    realtime_preview_model_dir: str = ""
    realtime_preview_python: str = ""
    reports_enabled: bool = False
    reports_port: int = 8798
    report_language: str = ""
    report_llm_provider: str = "llama_cpp"
    report_llm_base_url: str = ""
    report_llm_model: str = ""
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
        try:
            object.__setattr__(profile, "language", normalize_language_code(profile.language))
        except ValueError:
            object.__setattr__(profile, "language", "en")
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
        if profile.report_llm_provider not in {"llama_cpp", "ollama", "lm_studio", "openai", "openrouter"}:
            object.__setattr__(profile, "report_llm_provider", "llama_cpp")
        object.__setattr__(
            profile,
            "translation_provider",
            str(profile.translation_provider or "sidecar").strip().lower().replace("-", "_"),
        )
        if profile.translation_provider not in {
            "sidecar", "transformers", "reports_llm", "openai_compatible", "deepl", "google_cloud",
            "azure_translator", "libretranslate", "mock",
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
        except (FileNotFoundError, OSError, ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return Profile.from_mapping(data)
    return Profile()


def save_profile(profile: Profile, path: Path | None = None) -> Path:
    selected = path or config_path()
    payload = json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n"
    try:
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(payload, encoding="utf-8")
        return selected
    except OSError:
        if path is not None or os.environ.get("WHOSPEAKS_CONFIG"):
            raise
    fallback = local_config_path()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(payload, encoding="utf-8")
    return fallback


def update_profile_in_place(profile: Profile, updated: Profile) -> Profile:
    """Compatibility adapter for classic menus that retain a profile object."""

    for field in dataclasses.fields(Profile):
        setattr(profile, field.name, getattr(updated, field.name))
    return profile
