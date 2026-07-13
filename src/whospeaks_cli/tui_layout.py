"""Widget composition for the WhoSpeaks Textual application."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from . import main as backend


def provider_preset_label(preset_id: str, preset: backend.ProviderPreset) -> str:
    """Describe an embedding stack by its user-visible tradeoff."""

    return {
        "smoke": "Low VRAM - SpeechBrain ECAPA",
        "single_espnet": "Single model - ESPnet ECAPA",
        "smoke_fast_live": "Low VRAM final + fast live",
        "public_quality": "High quality - public ensemble",
        "promoted_public": "Recommended - public ensemble",
    }.get(preset_id, preset.name)


def compose_setup_app(app: Any) -> ComposeResult:
    target = {"local": "local", "remote": "core", "server": "server"}.get(app.profile.mode, "local")
    preview_engine = backend.normalize_preview_engine(app.profile.realtime_preview_engine)
    preview_preset = app.profile.realtime_preview_model_preset
    if preview_preset not in {"nemotron-3.5-560ms-int8", "nemotron-3.5-160ms-int8"}:
        preview_preset = "nemotron-3.5-560ms-int8"
    language_options = sorted(
        (
            (config.display_name, code)
            for code, config in backend.SUPPORTED_LANGUAGE_CONFIGS.items()
        ),
        key=lambda option: option[0].casefold(),
    )
    provider_options = [
        (provider_preset_label(preset_id, preset), preset_id)
        for preset_id, preset in backend.PROVIDER_PRESETS.items()
    ]
    report_language_options = [("Follow live language", ""), *language_options]
    report_provider_options = [
        ("llama.cpp", "llama_cpp"),
        ("Ollama", "ollama"),
        ("LM Studio", "lm_studio"),
        ("OpenAI-compatible", "openai_compatible"),
        ("OpenAI", "openai"),
        ("OpenRouter", "openrouter"),
    ]
    translation_provider_options = [
        ("Local sidecar (recommended)", "sidecar"),
        ("Local model in live process", "transformers"),
        ("DeepL API", "deepl"),
        ("Google Cloud Translation", "google_cloud"),
        ("Azure Translator", "azure_translator"),
        ("LibreTranslate endpoint", "libretranslate"),
        ("Reuse Meeting Intelligence LLM", "reports_llm"),
        ("OpenAI-compatible API", "openai_compatible"),
        ("Mock (testing)", "mock"),
    ]
    translation_model_options = [
        ("TranslateGemma 4B (recommended)", "translate-gemma-4b"),
        ("NLLB-200 600M (CC-BY-NC)", "nllb-200-600m"),
        ("MADLAD-400 3B (Apache-2.0)", "madlad-400-3b"),
    ]

    with Vertical(id="app-body"):
        with Horizontal(id="title-bar"):
            yield Static("WhoSpeaks Setup", id="app-title", markup=False)
            yield Static(f"v{backend.__version__}", id="app-meta", markup=False)
        with Horizontal(id="status-row"):
            yield Static("full local", id="mode-pill", markup=False)
            yield Static("not checked", id="readiness-text", markup=False)
            yield Static("", id="server-status-spacer", markup=False)
            yield Static("Live: stopped", id="live-server-state", classes="server-state", markup=False)
            yield Static("Reports: stopped", id="reports-server-state", classes="server-state", markup=False)
            yield Static("Translation: stopped", id="translation-server-state", classes="server-state", markup=False)
        with Vertical(id="operation-banner", classes="status-idle"):
            yield Static(id="operation-primary", markup=False)
            yield Static(id="operation-secondary", markup=False)
        with TabbedContent(initial="setup-tab", id="main-tabs"):
            with TabPane("Setup", id="setup-tab"):
                with Vertical(id="setup-options"):
                    with Horizontal(id="target-row"):
                        yield Label("Deployment:", id="target-label")
                        with RadioSet(
                            RadioButton("Full local", id="target-local", value=target == "local", compact=True),
                            RadioButton("Remote core", id="target-core", value=target == "core", compact=True),
                            RadioButton("Server", id="target-server", value=target == "server", compact=True),
                            id="target-select",
                        ):
                            pass
                        yield Label("Language:", id="language-label")
                        yield Select(
                            language_options,
                            value=app.profile.language,
                            allow_blank=False,
                            id="quick-language-select",
                        )
                    with Horizontal(id="realtime-row"):
                        yield Label("Live text:", id="realtime-label")
                        with RadioSet(
                            RadioButton("Nemotron", id="realtime-nemotron", value=preview_engine == "sherpa_onnx", compact=True),
                            RadioButton("Kroko", id="realtime-kroko", value=preview_engine == "kroko_onnx", compact=True),
                            RadioButton("Off", id="realtime-off", value=preview_engine not in {"sherpa_onnx", "kroko_onnx"}, compact=True),
                            id="realtime-select",
                        ):
                            pass
                        yield Checkbox(
                            "Live speaker labels",
                            value=app.profile.live_speaker_assignment,
                            id="live-speakers-checkbox",
                            compact=True,
                        )
                    with Horizontal(id="translation-install-row"):
                        yield Label("Translation:", id="translation-install-label")
                        yield Select(
                            [
                                ("NLLB-200 600M (recommended)", "nllb-200-600m"),
                                ("TranslateGemma 4B", "translate-gemma-4b"),
                                ("MADLAD-400 3B", "madlad-400-3b"),
                                ("Off", "off"),
                            ],
                            value=app.profile.translation_model_profile if app.profile.translation_enabled else "off",
                            allow_blank=False,
                            id="translation-install-select",
                        )
                    yield Static("", id="compatibility-note", markup=False)
                yield Static(id="compact-plan", markup=False)
                with Horizontal(id="setup-workspace"):
                    with Vertical(id="setup-state"):
                        yield Label("Component readiness", classes="section-title")
                        yield DataTable(id="component-table", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="setup-side"):
                        yield Label("Installation plan", classes="section-title")
                        yield Static(id="plan-summary", markup=False)
                        yield Label("Current operation", classes="section-title")
                        yield Static(id="operation-summary", markup=False)
                with Horizontal(id="setup-actions"):
                    yield Button("Exit", id="exit-button")
                    yield Button("Refresh", id="refresh-button")
                    yield Button("Launch", id="launch-button")
                    yield Button("Activity", id="view-activity-button")
                    yield Button("Install", id="install-button", variant="primary")
            with TabPane("Diagnostics", id="diagnostics-tab"):
                yield DataTable(id="doctor-table", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="doctor-actions"):
                    yield Button("Quick check", id="quick-doctor")
                    yield Button("Complete check", id="deep-doctor", variant="primary")
            with TabPane("Settings", id="settings-tab"):
                with VerticalScroll(id="settings-scroll"):
                    with Vertical(id="settings-grid"):
                        with Vertical(classes="field"):
                            yield Label("Language")
                            yield Select(
                                language_options,
                                value=app.profile.language,
                                allow_blank=False,
                                id="language-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Realtime text")
                            yield Select(
                                [
                                    ("Nemotron 3.5 (recommended)", "sherpa_onnx"),
                                    ("Kroko / Banafo", "kroko_onnx"),
                                    ("Disabled", "off"),
                                ],
                                value=preview_engine if preview_engine in {"sherpa_onnx", "kroko_onnx", "off"} else "off",
                                allow_blank=False,
                                id="realtime-engine-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Nemotron model")
                            yield Select(
                                [
                                    ("560 ms: stable", "nemotron-3.5-560ms-int8"),
                                    ("160 ms: lower latency", "nemotron-3.5-160ms-int8"),
                                ],
                                value=preview_preset,
                                allow_blank=False,
                                id="realtime-preset-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Nemotron model folder")
                            yield Input(
                                app.profile.realtime_preview_model_dir,
                                placeholder="Automatic download on first use",
                                id="realtime-model-dir-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Speaker model preset")
                            yield Select(
                                provider_options,
                                value=app.profile.provider_preset if app.profile.provider_preset in backend.PROVIDER_PRESETS else "smoke",
                                allow_blank=False,
                                id="provider-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("ASR model")
                            yield Input(app.profile.model, id="model-input")
                        with Vertical(classes="field"):
                            yield Label("Device")
                            yield Select(
                                [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                                value=app.profile.device,
                                allow_blank=False,
                                id="device-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Compute type")
                            yield Input(app.profile.compute_type, id="compute-input")
                        with Vertical(classes="field"):
                            yield Label("Browser host")
                            yield Input(app.profile.host, id="host-input")
                        with Vertical(classes="field"):
                            yield Label("Browser port")
                            yield Input(str(app.profile.port), type="integer", id="port-input")
                        with Vertical(classes="field"):
                            yield Label("Remote ASR URL")
                            yield Input(app.profile.remote_asr_url, id="asr-url-input")
                        with Vertical(classes="field"):
                            yield Label("Remote embeddings URL")
                            yield Input(app.profile.remote_embeddings_url, id="embeddings-url-input")
                with Horizontal(id="settings-actions"):
                    yield Button("Save settings", id="save-settings", variant="primary")
            with TabPane("Meeting Intelligence", id="reports-tab"):
                with VerticalScroll(id="reports-scroll"):
                    with Vertical(id="settings-grid"):
                        with Vertical(classes="field"):
                            yield Label("Meeting Intelligence — Reports + Ask")
                            yield Checkbox(
                                "Open Meeting Intelligence automatically",
                                value=app.profile.reports_enabled,
                                id="reports-enabled-checkbox",
                            )
                        with Vertical(classes="field"):
                            yield Label("Report browser port")
                            yield Input(str(app.profile.reports_port), type="integer", id="reports-port-input")
                        with Vertical(classes="field"):
                            yield Label("Report language")
                            yield Select(
                                report_language_options,
                                value=app.profile.report_language,
                                allow_blank=False,
                                id="report-language-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Report LLM provider")
                            yield Select(
                                report_provider_options,
                                value=app.profile.report_llm_provider,
                                allow_blank=False,
                                id="report-llm-provider-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Report LLM base URL")
                            yield Input(app.profile.report_llm_base_url, placeholder="Provider default", id="report-llm-base-url-input")
                        with Vertical(classes="field"):
                            yield Label("Report LLM model")
                            yield Input(app.profile.report_llm_model, placeholder="Provider default", id="report-llm-model-input")
                        with Vertical(classes="field"):
                            yield Label("Text embedding base URL")
                            yield Input(app.profile.text_embedding_base_url, placeholder="OpenAI-compatible /v1", id="text-embedding-base-url-input")
                        with Vertical(classes="field"):
                            yield Label("Text embedding model")
                            yield Input(app.profile.text_embedding_model, placeholder="Embedding-capable model", id="text-embedding-model-input")
                        with Vertical(classes="field"):
                            yield Label("Text embedding API-key variable")
                            yield Input(app.profile.text_embedding_api_key_env, placeholder="Optional environment variable", id="text-embedding-api-key-env-input")
                        with Vertical(classes="field"):
                            yield Label("Automatic reports")
                            yield Checkbox(
                                "Generate when a new meeting is saved",
                                value=app.profile.report_auto_generate,
                                id="report-auto-generate-checkbox",
                            )
                with Horizontal(id="reports-actions"):
                    yield Button("Save Meeting Intelligence", id="save-reports-settings", variant="primary")
                    yield Button("Start Meeting Intelligence", id="start-reports-button")
            with TabPane("Translation", id="translation-tab"):
                with VerticalScroll(id="translation-scroll"):
                    with Vertical(id="settings-grid"):
                        with Vertical(classes="field"):
                            yield Label("Translate stable transcript text")
                            yield Checkbox(
                                "Enable translation with live window",
                                value=app.profile.translation_enabled,
                                id="translation-enabled-checkbox",
                            )
                        with Vertical(classes="field"):
                            yield Label("Chrome on-device translation")
                            yield Checkbox(
                                "Prefer Chrome; use selected provider as fallback",
                                value=app.profile.translation_browser_preferred,
                                id="translation-browser-preferred-checkbox",
                            )
                        with Vertical(classes="field"):
                            yield Label("Translation provider")
                            yield Select(
                                translation_provider_options,
                                value=app.profile.translation_provider,
                                allow_blank=False,
                                id="translation-provider-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Target language codes")
                            yield Input(
                                app.profile.translation_target_languages,
                                placeholder="de, fr, ja",
                                id="translation-targets-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Maximum simultaneous targets")
                            yield Input(
                                str(app.profile.translation_max_targets),
                                type="integer",
                                id="translation-max-targets-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Local model profile")
                            yield Select(
                                translation_model_options,
                                value=app.profile.translation_model_profile,
                                allow_blank=False,
                                id="translation-model-profile-select",
                            )
                        with Vertical(classes="field"):
                            yield Label("Model override (optional)")
                            yield Input(
                                app.profile.translation_model,
                                placeholder="Hugging Face or API model ID",
                                id="translation-model-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("API / sidecar base URL (optional)")
                            yield Input(
                                app.profile.translation_base_url,
                                placeholder="Provider default",
                                id="translation-base-url-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("API-key environment variable")
                            yield Input(
                                app.profile.translation_api_key_env,
                                placeholder="Provider default",
                                id="translation-api-key-env-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Provider region (Azure)")
                            yield Input(
                                app.profile.translation_region,
                                placeholder="For example: westeurope",
                                id="translation-region-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Translation Python (optional)")
                            yield Input(
                                app.profile.translation_python,
                                placeholder="Use the WhoSpeaks Python",
                                id="translation-python-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Local sidecar port")
                            yield Input(
                                str(app.profile.translation_port),
                                type="integer",
                                id="translation-port-input",
                            )
                        with Vertical(classes="field"):
                            yield Label("Local translation device")
                            yield Select(
                                [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                                value=app.profile.translation_device,
                                allow_blank=False,
                                id="translation-device-select",
                            )
                with Horizontal(id="translation-actions"):
                    yield Button("Save translation settings", id="save-translation-settings", variant="primary")
                    yield Button("Start translation now", id="start-translation-button")
            with TabPane("Activity", id="activity-tab"):
                yield RichLog(id="activity-log", wrap=True, markup=False, max_lines=5000)
                with Horizontal(id="activity-actions"):
                    yield Button("Clear", id="clear-log")
                    yield Button("Cancel operation", id="cancel-operation")
