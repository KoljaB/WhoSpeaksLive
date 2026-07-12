"""Textual setup, diagnostics, and launcher interface for WhoSpeaks."""

from __future__ import annotations

import os
import socket
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
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


STATUS_STYLES = {
    "ok": "bold #74c69d",
    "warn": "bold #f4b942",
    "fail": "bold #ff6b6b",
    "skip": "#8d99ae",
}


def realtime_plan_label(plan: backend.InstallPlan) -> str:
    """Return a short, user-facing label for the selected realtime engine."""

    if plan.realtime_preview_engine == "sherpa_onnx":
        if plan.realtime_preview_model_preset == "nemotron-3.5-160ms-int8":
            return "Nemotron 3.5 live text (160 ms)"
        return "Nemotron 3.5 live text (560 ms)"
    if plan.realtime_preview_engine == "kroko_onnx":
        return "Kroko / Banafo live text"
    return "Live text disabled"


def provider_preset_label(preset_id: str, preset: backend.ProviderPreset) -> str:
    """Describe an embedding stack by its user-visible tradeoff, not its raw provider expression."""

    return {
        "smoke": "Low VRAM - SpeechBrain ECAPA",
        "single_espnet": "Single model - ESPnet ECAPA",
        "smoke_fast_live": "Low VRAM final + fast live",
        "public_quality": "High quality - public ensemble",
        "promoted_public": "Recommended - public ensemble",
    }.get(preset_id, preset.name)


class ConfirmInstallScreen(ModalScreen[bool]):
    """Confirm a concrete installation plan before starting subprocesses."""

    BINDINGS = [Binding("escape", "cancel", show=False)]

    def __init__(self, plan: backend.InstallPlan, command: list[str]) -> None:
        super().__init__()
        self.plan = plan
        self.command = command

    def compose(self) -> ComposeResult:
        realtime = realtime_plan_label(self.plan)
        with Vertical(id="confirm-dialog"):
            yield Label("Confirm installation", id="confirm-title")
            yield Static(self.plan.title, classes="confirm-value", markup=False)
            yield Static(self.plan.summary, classes="confirm-summary", markup=False)
            yield Static(f"Realtime text: {realtime}", classes="confirm-value", markup=False)
            yield Static(
                backend.format_command(self.command),
                id="confirm-command",
                markup=False,
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Cancel", id="cancel-install")
                yield Button("Start installation", id="confirm-install", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-install":
            self.dismiss(False)
        elif event.button.id == "confirm-install":
            self.dismiss(True)


class WhoSpeaksSetupApp(App[str]):
    """Interactive setup and operational dashboard."""

    TITLE = "WhoSpeaks Setup"
    SUB_TITLE = "Installation and system readiness"
    BINDINGS = [
        Binding("ctrl+r", "refresh", show=False),
        Binding("ctrl+l", "launch", show=False),
        Binding("ctrl+q", "quit", show=False),
    ]

    CSS = """
    Screen {
        background: #0d1117;
        color: #e9eef0;
    }

    #app-body {
        height: 1fr;
    }

    #title-bar {
        height: 3;
        padding: 1 2 0 2;
        background: #0d1117;
        color: #ffffff;
    }

    #app-title {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    #app-meta {
        width: auto;
        height: 1;
        color: #8292b4;
    }

    #status-row {
        height: 2;
        padding: 0 2 1 2;
        background: #0d1117;
    }

    #mode-pill, #readiness-text, .server-state {
        width: auto;
        height: 1;
        margin-right: 1;
        padding: 0 1;
        color: #64f1c4;
        background: #132b2d;
    }

    #readiness-text {
        color: #ffb845;
        background: #2a2419;
    }

    #server-status-spacer {
        width: 1fr;
        height: 1;
    }

    .server-state {
        color: #91a0ad;
        background: #171d20;
    }

    .server-state.running {
        color: #8dd4a3;
        background: #183527;
    }

    .server-state.starting {
        color: #78b8ff;
        background: #172b42;
    }

    .server-state.failed {
        color: #ff929b;
        background: #3b2024;
    }

    #operation-banner {
        height: 0;
        padding: 0;
        background: #0d1117;
        border: none;
    }

    #operation-primary, #operation-secondary {
        height: 1;
    }

    #operation-primary {
        text-style: bold;
        color: #dce6e8;
    }

    #operation-secondary {
        color: #9faeb4;
    }

    #operation-banner.status-running {
        background: #0d1117;
        border: none;
    }

    #operation-banner.status-running #operation-primary {
        color: #91e0d9;
    }

    #operation-banner.status-success {
        background: #0d1117;
        border: none;
    }

    #operation-banner.status-success #operation-primary {
        color: #8dd4a3;
    }

    #operation-banner.status-warning {
        background: #0d1117;
        border: none;
    }

    #operation-banner.status-warning #operation-primary {
        color: #f2c868;
    }

    #operation-banner.status-error {
        background: #0d1117;
        border: none;
    }

    #operation-banner.status-error #operation-primary {
        color: #ff929b;
    }

    TabbedContent, ContentSwitcher {
        height: 1fr;
    }

    TabPane {
        padding: 0 1;
        height: 1fr;
    }

    Tabs {
        height: 3;
        padding: 0 2;
        background: #0d1117;
        color: #8292b4;
        border-bottom: solid #222a32;
    }

    Tab {
        height: 3;
        padding: 0 2;
        background: #0d1117;
    }

    Tab.-active {
        color: #ffffff;
        background: #0d1117;
        border-bottom: solid #5798f2;
        text-style: bold;
    }

    .section-title {
        height: 1;
        color: #ffffff;
        text-style: bold;
    }

    #setup-options {
        height: 6;
        layout: vertical;
        padding: 0 2;
        background: #0d1117;
    }

    #target-row, #realtime-row, #translation-install-row {
        width: 1fr;
        height: 2;
        align-vertical: middle;
    }

    #target-label, #realtime-label, #translation-install-label {
        width: 12;
        height: 1;
        color: #8f9db8;
    }

    #target-select, #realtime-select {
        width: 1fr;
        height: 2;
        layout: horizontal;
        align-vertical: middle;
        background: transparent;
        border: none;
    }

    #target-select RadioButton, #realtime-select RadioButton {
        width: auto;
        height: 1;
        padding-right: 1;
    }

    #quick-language-select {
        width: 28;
        height: 2;
    }

    #language-label {
        width: 10;
        height: 1;
        color: #8f9db8;
    }

    #quick-language-select SelectCurrent {
        height: 2;
        border: none;
        background: #1b2227;
    }

    #live-speakers-checkbox {
        width: auto;
        height: 1;
    }

    #compatibility-note {
        display: none;
        height: 1;
        padding-left: 12;
        color: #ffb845;
    }

    Screen.preview-incompatible #setup-options {
        height: 7;
    }

    Screen.preview-incompatible #compatibility-note {
        display: block;
    }

    Screen.compact #quick-language-select {
        width: 18;
    }

    #compact-plan {
        display: none;
        height: 6;
        margin: 1 2 1 2;
        padding: 0 2;
        background: #101824;
        color: #9eb0cd;
        border: solid #27466f;
    }

    #compact-plan.status-running {
        display: block;
        background: #101824;
        color: #78b8ff;
        border: solid #315d93;
    }

    #compact-plan.status-success {
        display: block;
        height: 4;
        background: #183527;
        color: #d8f1df;
        border: solid #5eae78;
    }

    #compact-plan.status-warning {
        display: block;
        height: 4;
        background: #3a3019;
        color: #f8e1a9;
        border: solid #d3a642;
    }

    #compact-plan.status-error {
        display: block;
        height: 4;
        background: #3b2024;
        color: #ffd0d5;
        border: solid #cc6570;
    }

    #setup-workspace {
        height: 1fr;
    }

    #setup-state {
        width: 1fr;
        height: 1fr;
        padding: 0 2;
    }

    #setup-side {
        width: 39;
        min-width: 34;
        height: 1fr;
        padding: 0 0 0 1;
        margin-left: 1;
        border-left: solid #3d4a50;
    }

    #plan-summary {
        height: auto;
        max-height: 9;
        margin-bottom: 1;
        padding: 1;
        background: #171d20;
        border: solid #3d4a50;
    }

    #operation-summary {
        height: auto;
        min-height: 10;
        max-height: 12;
        padding: 1;
        background: #171d20;
        border: solid #3d4a50;
        color: #bdc9cd;
    }

    #operation-summary.status-running {
        background: #123d40;
        border: solid #51b9b0;
        color: #d7f5f0;
    }

    #setup-actions, #doctor-actions, #settings-actions, #reports-actions, #translation-actions, #activity-actions {
        dock: bottom;
        height: 4;
        align-horizontal: right;
        padding: 0 2;
        background: #0d1117;
        border-top: solid #222a32;
    }

    #setup-actions Button {
        width: 1fr;
        min-width: 10;
    }

    Button {
        min-width: 13;
        height: 3;
        margin-left: 1;
        border: none;
        content-align: center middle;
    }

    #setup-actions Button:first-of-type {
        margin-left: 0;
    }

    Button.-primary {
        background: #137d73;
        color: #ffffff;
    }

    Button.-primary:hover {
        background: #1b9b8e;
    }

    Button.danger, #cancel-operation {
        background: #9d3f48;
        color: #ffffff;
    }

    #launch-button {
        background: #3b7a57;
        color: #ffffff;
    }

    #cancel-operation {
        display: none;
    }

    DataTable {
        height: 1fr;
        background: #0d1117;
        border: none;
    }

    #settings-scroll {
        height: 1fr;
    }

    #settings-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1 2;
        height: auto;
        padding: 1 1 4 0;
    }

    .field {
        height: 5;
    }

    .field Label {
        height: 2;
        color: #b8c4ca;
    }

    Input {
        height: 3;
        background: #1b2227;
        border: solid #46545d;
    }

    Select {
        height: 3;
        background: transparent;
        border: none;
    }

    SelectCurrent {
        height: 3;
        color: #e9eef0;
        background: #1b2227;
        border: solid #46545d;
    }

    SelectCurrent.-has-value Static#label {
        color: #e9eef0;
    }

    SelectCurrent .arrow {
        color: #65d1c8;
    }

    Select > SelectOverlay {
        color: #e9eef0;
        background: #1b2227;
        border: solid #46545d;
    }

    Input:focus, Select:focus > SelectCurrent {
        border: solid #65d1c8;
    }

    #activity-log {
        height: 1fr;
        margin-top: 1;
        background: #0d1114;
        border: solid #3d4a52;
        padding: 0 1;
    }

    ConfirmInstallScreen {
        align: center middle;
        background: rgba(5, 8, 10, 0.82);
    }

    #confirm-dialog {
        width: 76;
        max-width: 92%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: #1b2227;
        border: solid #65d1c8;
    }

    #confirm-title {
        height: 2;
        text-style: bold;
        color: #ffffff;
    }

    .confirm-value, .confirm-summary {
        height: auto;
        margin-bottom: 1;
    }

    #confirm-command {
        height: auto;
        max-height: 6;
        padding: 1;
        background: #0d1114;
        color: #c7d2d8;
        border: solid #3d4a52;
    }

    .dialog-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }

    Screen.compact #setup-side, Screen.short #setup-side {
        display: none;
    }

    Screen.short #compact-plan {
        margin: 0 2;
    }

    Screen.short #compact-plan.status-idle {
        display: block;
        height: 4;
    }

    Screen.compact #settings-grid {
        grid-size: 1;
        grid-columns: 1fr;
    }

    Screen.narrow #target-row, Screen.narrow #realtime-row, Screen.narrow #translation-install-row {
        width: 1fr;
        height: 2;
    }

    Screen.narrow Tab {
        padding: 0 1;
    }
    """

    def __init__(
        self,
        profile: backend.Profile | None = None,
        *,
        doctor_runner: Callable[..., backend.DoctorReport] = backend.run_doctor,
        auto_doctor: bool = True,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        super().__init__()
        self.profile = profile or backend.load_profile()
        self.doctor_runner = doctor_runner
        self.auto_doctor = auto_doctor
        self.popen_factory = popen_factory
        self.report = backend.DoctorReport(self.profile.mode, [])
        self.install_process: subprocess.Popen[str] | None = None
        self.live_server_process: subprocess.Popen[str] | None = None
        self.reports_server_process: subprocess.Popen[str] | None = None
        self.translation_server_process: subprocess.Popen[str] | None = None
        self.live_server_state = "stopped"
        self.reports_server_state = "stopped"
        self.translation_server_state = "stopped"
        self.last_server_probe_at = 0.0
        self.install_cancelled = False
        self.active_operation = ""
        self.pending_install_command: list[str] | None = None
        self.pending_install_title = ""
        self.operation_status = "idle"
        self.operation_title = "Setup is idle"
        self.operation_step = ""
        self.operation_latest = "Choose a target, review the plan, then select Install."
        self.operation_started_at: float | None = None
        self.spinner_index = 0
        self.language_selection_changed = False

    def compose(self) -> ComposeResult:
        target = {"local": "local", "remote": "core", "server": "server"}.get(self.profile.mode, "local")
        preview_engine = backend.normalize_preview_engine(self.profile.realtime_preview_engine)
        preview_preset = self.profile.realtime_preview_model_preset
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
            ("Reuse reports LLM", "reports_llm"),
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
                                value=self.profile.language,
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
                                value=self.profile.live_speaker_assignment,
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
                                value=self.profile.translation_model_profile if self.profile.translation_enabled else "off",
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
                                    value=self.profile.language,
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
                                    self.profile.realtime_preview_model_dir,
                                    placeholder="Automatic download on first use",
                                    id="realtime-model-dir-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Speaker model preset")
                                yield Select(
                                    provider_options,
                                    value=self.profile.provider_preset if self.profile.provider_preset in backend.PROVIDER_PRESETS else "smoke",
                                    allow_blank=False,
                                    id="provider-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("ASR model")
                                yield Input(self.profile.model, id="model-input")
                            with Vertical(classes="field"):
                                yield Label("Device")
                                yield Select(
                                    [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                                    value=self.profile.device,
                                    allow_blank=False,
                                    id="device-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("Compute type")
                                yield Input(self.profile.compute_type, id="compute-input")
                            with Vertical(classes="field"):
                                yield Label("Browser host")
                                yield Input(self.profile.host, id="host-input")
                            with Vertical(classes="field"):
                                yield Label("Browser port")
                                yield Input(str(self.profile.port), type="integer", id="port-input")
                            with Vertical(classes="field"):
                                yield Label("Remote ASR URL")
                                yield Input(self.profile.remote_asr_url, id="asr-url-input")
                            with Vertical(classes="field"):
                                yield Label("Remote embeddings URL")
                                yield Input(self.profile.remote_embeddings_url, id="embeddings-url-input")
                    with Horizontal(id="settings-actions"):
                        yield Button("Save settings", id="save-settings", variant="primary")
                with TabPane("Reports", id="reports-tab"):
                    with VerticalScroll(id="reports-scroll"):
                        with Vertical(id="settings-grid"):
                            with Vertical(classes="field"):
                                yield Label("Start reports with live window")
                                yield Checkbox(
                                    "Open the report server automatically",
                                    value=self.profile.reports_enabled,
                                    id="reports-enabled-checkbox",
                                )
                            with Vertical(classes="field"):
                                yield Label("Report browser port")
                                yield Input(str(self.profile.reports_port), type="integer", id="reports-port-input")
                            with Vertical(classes="field"):
                                yield Label("Report language")
                                yield Select(
                                    report_language_options,
                                    value=self.profile.report_language,
                                    allow_blank=False,
                                    id="report-language-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("Report LLM provider")
                                yield Select(
                                    report_provider_options,
                                    value=self.profile.report_llm_provider,
                                    allow_blank=False,
                                    id="report-llm-provider-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("Report LLM base URL")
                                yield Input(self.profile.report_llm_base_url, placeholder="Provider default", id="report-llm-base-url-input")
                            with Vertical(classes="field"):
                                yield Label("Report LLM model")
                                yield Input(self.profile.report_llm_model, placeholder="Provider default", id="report-llm-model-input")
                            with Vertical(classes="field"):
                                yield Label("Automatic reports")
                                yield Checkbox(
                                    "Generate when a new meeting is saved",
                                    value=self.profile.report_auto_generate,
                                    id="report-auto-generate-checkbox",
                                )
                    with Horizontal(id="reports-actions"):
                        yield Button("Save report settings", id="save-reports-settings", variant="primary")
                        yield Button("Start reports now", id="start-reports-button")
                with TabPane("Translation", id="translation-tab"):
                    with VerticalScroll(id="translation-scroll"):
                        with Vertical(id="settings-grid"):
                            with Vertical(classes="field"):
                                yield Label("Translate stable transcript text")
                                yield Checkbox(
                                    "Enable translation with live window",
                                    value=self.profile.translation_enabled,
                                    id="translation-enabled-checkbox",
                                )
                            with Vertical(classes="field"):
                                yield Label("Chrome on-device translation")
                                yield Checkbox(
                                    "Prefer Chrome; use selected provider as fallback",
                                    value=self.profile.translation_browser_preferred,
                                    id="translation-browser-preferred-checkbox",
                                )
                            with Vertical(classes="field"):
                                yield Label("Translation provider")
                                yield Select(
                                    translation_provider_options,
                                    value=self.profile.translation_provider,
                                    allow_blank=False,
                                    id="translation-provider-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("Target language codes")
                                yield Input(
                                    self.profile.translation_target_languages,
                                    placeholder="de, fr, ja",
                                    id="translation-targets-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Maximum simultaneous targets")
                                yield Input(
                                    str(self.profile.translation_max_targets),
                                    type="integer",
                                    id="translation-max-targets-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Local model profile")
                                yield Select(
                                    translation_model_options,
                                    value=self.profile.translation_model_profile,
                                    allow_blank=False,
                                    id="translation-model-profile-select",
                                )
                            with Vertical(classes="field"):
                                yield Label("Model override (optional)")
                                yield Input(
                                    self.profile.translation_model,
                                    placeholder="Hugging Face or API model ID",
                                    id="translation-model-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("API / sidecar base URL (optional)")
                                yield Input(
                                    self.profile.translation_base_url,
                                    placeholder="Provider default",
                                    id="translation-base-url-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("API-key environment variable")
                                yield Input(
                                    self.profile.translation_api_key_env,
                                    placeholder="Provider default",
                                    id="translation-api-key-env-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Provider region (Azure)")
                                yield Input(
                                    self.profile.translation_region,
                                    placeholder="For example: westeurope",
                                    id="translation-region-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Translation Python (optional)")
                                yield Input(
                                    self.profile.translation_python,
                                    placeholder="Use the WhoSpeaks Python",
                                    id="translation-python-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Local sidecar port")
                                yield Input(
                                    str(self.profile.translation_port),
                                    type="integer",
                                    id="translation-port-input",
                                )
                            with Vertical(classes="field"):
                                yield Label("Local translation device")
                                yield Select(
                                    [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
                                    value=self.profile.translation_device,
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

    def on_mount(self) -> None:
        self._configure_tables()
        self._apply_size_classes(self.size.width, self.size.height)
        self._sync_realtime_settings()
        self._sync_translation_settings()
        self._update_plan(announce=False)
        self._sync_preview_compatibility()
        self._render_report(self.report)
        self._render_operation()
        self._render_server_states()
        self._sync_action_buttons()
        self.set_interval(0.25, self._tick_operation)
        self._append_log(f"WhoSpeaks {backend.__version__}")
        self._append_log(f"Profile: {backend.config_path()}")
        if self.auto_doctor:
            self.run_doctor_worker(False)
        else:
            self.query_one("#readiness-text", Static).update("Readiness not checked")

    def on_resize(self, event: events.Resize) -> None:
        self._apply_size_classes(event.size.width, event.size.height)
        self._render_operation()

    def _apply_size_classes(self, width: int, height: int) -> None:
        if width < 112:
            self.screen.add_class("compact")
        else:
            self.screen.remove_class("compact")
        if width < 76:
            self.screen.add_class("narrow")
        else:
            self.screen.remove_class("narrow")
        if height < 38:
            self.screen.add_class("short")
        else:
            self.screen.remove_class("short")

    def _configure_tables(self) -> None:
        for table_id in ("#component-table", "#doctor-table"):
            table = self.query_one(table_id, DataTable)
            table.add_column("State", width=8)
            table.add_column("Component", width=28)
            table.add_column("Detail")

    def _selected_target(self) -> str:
        selected = self.query_one("#target-select", RadioSet).pressed_button
        if selected is None:
            return "local"
        return {
            "target-core": "core",
            "target-server": "server",
        }.get(selected.id or "", "local")

    def _selected_realtime_engine(self) -> str:
        selected = self.query_one("#realtime-select", RadioSet).pressed_button
        if selected is None:
            return "sherpa_onnx"
        return {
            "realtime-kroko": "kroko_onnx",
            "realtime-off": "off",
        }.get(selected.id or "", "sherpa_onnx")

    def _select_realtime_engine(self, engine: str) -> None:
        """Synchronize the compact RadioSet with an engine selected elsewhere."""

        radio_set = self.query_one("#realtime-select", RadioSet)
        button_id = {
            "sherpa_onnx": "#realtime-nemotron",
            "kroko_onnx": "#realtime-kroko",
        }.get(engine, "#realtime-off")
        button = self.query_one(button_id, RadioButton)
        if radio_set.pressed_button is button:
            return
        if radio_set.pressed_button is not None:
            radio_set.pressed_button.value = False
        button.value = True
        # Textual exposes the current button as read-only; keep its model state
        # aligned with the programmatic selection used for the server target.
        radio_set._pressed_button = button

    def _language_value(self) -> str:
        return str(self.query_one("#quick-language-select", Select).value or self.profile.language)

    def _preview_compatibility_error(self) -> str | None:
        return backend.preview_language_error(self._selected_realtime_engine(), self._language_value())

    def _sync_preview_compatibility(self) -> str | None:
        error = self._preview_compatibility_error()
        note = self.query_one("#compatibility-note", Static)
        if error:
            language = backend.SUPPORTED_LANGUAGE_CONFIGS[self._language_value()].display_name
            engine = {
                "sherpa_onnx": "Nemotron",
                "kroko_onnx": "Kroko",
            }.get(self._selected_realtime_engine(), "Live text")
            recommendation = backend.recommended_preview_engine(self._language_value())
            alternative = {
                "sherpa_onnx": "Nemotron",
                "kroko_onnx": "Kroko",
            }.get(recommendation, "Off")
            note.update(f"{engine} does not support {language}. Choose {alternative} or Off.")
            self.screen.add_class("preview-incompatible")
        else:
            note.update("")
            self.screen.remove_class("preview-incompatible")
        return error

    def _select_recommended_engine(self, language: str) -> None:
        engine = backend.recommended_preview_engine(language)
        settings_select = self.query_one("#realtime-engine-select", Select)
        if settings_select.value != engine:
            settings_select.value = engine
        self._select_realtime_engine(engine)
        self._sync_realtime_settings()
        self._update_plan()
        self._sync_preview_compatibility()
        self._sync_action_buttons()

    def _selected_plan(self) -> backend.InstallPlan:
        target = self._selected_target()
        engine = "off" if target == "server" else self._selected_realtime_engine()
        preset = self.query_one("#realtime-preset-select", Select).value if engine == "sherpa_onnx" else ""
        return backend.install_plan_for_target(
            target,
            realtime_preview_engine=engine,
            realtime_preview_model_preset=str(preset or ""),
            translation_model_profile=str(self.query_one("#translation-install-select", Select).value or "off"),
        )

    def _update_plan(self, *, announce: bool = True) -> None:
        target = self._selected_target()
        realtime_select = self.query_one("#realtime-select", RadioSet)
        if target == "server":
            self._select_realtime_engine("off")
        realtime_select.disabled = target == "server" or bool(self.active_operation)
        plan = self._selected_plan()
        realtime = realtime_plan_label(plan)
        component_lines, compact_components = self._plan_components(plan)
        summary = "\n".join(
            (
                plan.title,
                "",
                *component_lines,
                "",
                f"Realtime text: {realtime}",
                f"Translation: {plan.translation_model_profile if plan.translation_model_profile != 'off' else 'off'}",
            )
        )
        self.query_one("#mode-pill", Static).update(
            {
                "local": "full local",
                "core": "remote core",
                "server": "server",
            }.get(plan.target, plan.target)
        )
        self.query_one("#plan-summary", Static).update(summary)
        self.query_one("#compact-plan", Static).update(
            f"Plan: {plan.title}\n{realtime} | {compact_components}"
        )
        if announce and not self.active_operation:
            self._set_feedback(
                "idle",
                "Installation plan updated",
                f"{plan.title}; {realtime.lower()}.",
            )

    def _plan_components(self, plan: backend.InstallPlan) -> tuple[tuple[str, ...], str]:
        if plan.target == "local":
            return (
                ("Browser controller", "Local final ASR", "Speaker embeddings"),
                "controller + local ASR + embeddings",
            )
        if plan.target == "core":
            return (
                ("Browser controller", "Remote ASR and embeddings"),
                "controller + remote ASR + embeddings",
            )
        return (
            ("ASR service", "Embeddings service"),
            "ASR + embeddings services",
        )

    def _render_report(self, report: backend.DoctorReport) -> None:
        self.report = report
        readiness = backend.report_readiness_line(report)
        self.query_one("#readiness-text", Static).update(readiness)
        checks = report.checks or [backend.CheckResult("System", "skip", "Run a readiness check.")]
        self._render_table("#component-table", self._setup_checks(checks))
        self._render_table("#doctor-table", checks)

    def _setup_checks(self, checks: list[backend.CheckResult]) -> list[backend.CheckResult]:
        key_components = {
            "Python",
            "WhoSpeaks package",
            "ffmpeg",
            "Controller Python modules",
            "Local ASR modules",
            "Local embedding modules",
            "Server Python modules",
            "Remote ASR health",
            "Remote embeddings health",
            "Realtime preview",
            "Realtime preview Python",
            "Kroko ONNX runtime",
            "Nemotron sherpa-onnx runtime",
            "Nemotron model folder",
            "Browser UI port",
            "Launch profile",
        }
        selected = [
            check
            for check in checks
            if check.status in {"fail", "warn"}
            or (check.status != "skip" and check.name in key_components)
        ]
        return selected or [backend.CheckResult("System", "skip", "No actionable checks to show.")]

    def _render_table(self, table_id: str, checks: list[backend.CheckResult]) -> None:
        table = self.query_one(table_id, DataTable)
        table.clear()
        for check in checks:
            label = backend.STATUS_LABEL.get(check.status, check.status.upper())
            style = STATUS_STYLES.get(check.status, "#edf2f4")
            detail = check.detail
            if check.remediation:
                detail = f"{detail}  Fix: {check.remediation}"
            table.add_row(Text(label, style=style), check.name, detail)

    def _start_operation(self, name: str, title: str, step: str) -> None:
        self.active_operation = name
        self.operation_status = "running"
        self.operation_title = title
        self.operation_step = step
        self.operation_latest = "The operation has started."
        self.operation_started_at = time.monotonic()
        self.spinner_index = 0
        self._render_operation()
        self._sync_action_buttons()

    def _finish_operation(self, status: str, title: str, detail: str) -> None:
        self.active_operation = ""
        self.operation_status = status
        self.operation_title = title
        self.operation_step = ""
        self.operation_latest = detail
        self.operation_started_at = None
        self._render_operation()
        self._sync_action_buttons()

    def _set_feedback(self, status: str, title: str, detail: str) -> None:
        if self.active_operation:
            return
        self.operation_status = status
        self.operation_title = title
        self.operation_step = ""
        self.operation_latest = detail
        self._render_operation()

    def _tick_operation(self) -> None:
        if self.active_operation:
            self.spinner_index += 1
            self._render_operation()
        self._refresh_server_states()

    @staticmethod
    def _process_return_code(process: object | None) -> int | None:
        if process is None:
            return None
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return None
        return poll()

    def _process_is_running(self, process: object | None) -> bool:
        return process is not None and self._process_return_code(process) is None

    def _render_server_state(self, selector: str, label: str, state: str) -> None:
        matches = list(self.query(selector))
        if not matches:
            return
        widget = matches[0]
        for state_class in ("running", "starting", "failed"):
            widget.remove_class(state_class)
        if state in {"running", "starting", "failed"}:
            widget.add_class(state)
        widget.update(f"{label}: {state}")

    def _render_server_states(self) -> None:
        self._render_server_state("#live-server-state", "Live", self.live_server_state)
        self._render_server_state("#reports-server-state", "Reports", self.reports_server_state)
        self._render_server_state("#translation-server-state", "Translation", self.translation_server_state)

    def _refresh_server_states(self) -> None:
        if not list(self.query("#live-server-state")):
            return
        changed = False
        now = time.monotonic()
        probe_due = now - self.last_server_probe_at >= 0.75
        listening: dict[str, bool] = {}
        if probe_due:
            self.last_server_probe_at = now
            translation_should_probe = (
                self._process_is_running(self.translation_server_process)
                or (
                    self.profile.translation_enabled
                    and self.profile.translation_provider == "sidecar"
                )
            )
            listening = {
                "live": self._server_port_accepting(self.profile.host, self.profile.port),
                "reports": self._server_port_accepting(self.profile.host, self.profile.reports_port),
                "translation": (
                    translation_should_probe
                    and self._server_port_accepting(self.profile.host, self.profile.translation_port)
                ),
            }
        for kind in ("live", "reports", "translation"):
            process_attr = f"{kind}_server_process"
            state_attr = f"{kind}_server_state"
            process = getattr(self, process_attr)
            return_code = self._process_return_code(process)
            if process is not None and return_code is None:
                if probe_due:
                    next_state = "running" if listening[kind] else "starting"
                else:
                    next_state = getattr(self, state_attr)
            elif process is None:
                if not probe_due:
                    continue
                current_state = getattr(self, state_attr)
                next_state = "running" if listening[kind] else ("failed" if current_state == "failed" else "stopped")
            else:
                next_state = "stopped" if return_code == 0 else "failed"
                setattr(self, process_attr, None)
                label = self._server_label(kind)
                self._append_log(f"{label} exited with code {return_code}.")
            if getattr(self, state_attr) != next_state:
                setattr(self, state_attr, next_state)
                changed = True
        if changed:
            self._render_server_states()
            self._sync_action_buttons()

    @staticmethod
    def _server_port_accepting(host: str, port: int) -> bool:
        probe_host = str(host or "127.0.0.1").strip()
        if probe_host in {"0.0.0.0", "::", "[::]"}:
            probe_host = "127.0.0.1"
        try:
            with socket.create_connection((probe_host, int(port)), timeout=0.08):
                return True
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _new_server_console_kwargs() -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_CONSOLE}
        return {"start_new_session": True}

    def _start_server_process(self, kind: str, command: list[str]) -> bool:
        process_attr = f"{kind}_server_process"
        state_attr = f"{kind}_server_state"
        if self._process_is_running(getattr(self, process_attr)) or getattr(self, state_attr) == "running":
            label = self._server_label(kind)
            self.notify(f"{label} is already running", severity="warning")
            return False
        label = self._server_label(kind)
        try:
            process = self.popen_factory(command, **self._new_server_console_kwargs())
        except OSError as exc:
            setattr(self, state_attr, "failed")
            self._render_server_states()
            self._append_log(f"Could not start {label.lower()}: {exc}")
            self._set_feedback("error", f"{label} failed to start", str(exc))
            return False
        setattr(self, process_attr, process)
        setattr(self, state_attr, "starting")
        self._render_server_states()
        self._sync_action_buttons()
        self._append_log(f"Started {label.lower()}: {backend.format_command(command)}")
        return True

    @staticmethod
    def _server_label(kind: str) -> str:
        return {
            "live": "Live server",
            "reports": "Reports server",
            "translation": "Translation server",
        }.get(kind, f"{kind.title()} server")

    def _elapsed_text(self) -> str:
        if self.operation_started_at is None:
            return ""
        seconds = max(0, int(time.monotonic() - self.operation_started_at))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _fit_line(self, value: str) -> str:
        limit = max(24, self.size.width - 4)
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def _progress_bar(self) -> str:
        width = min(64, max(24, self.size.width - 14))
        filled = width // 3
        offset = self.spinner_index % max(1, width - filled)
        cells = ["-"] * width
        for index in range(offset, min(width, offset + filled)):
            cells[index] = "="
        return "[" + "".join(cells) + "]"

    def _render_operation(self) -> None:
        banner = self.query_one("#operation-banner", Vertical)
        compact_status = self.query_one("#compact-plan", Static)
        operation_summary = self.query_one("#operation-summary", Static)
        for state in ("idle", "running", "success", "warning", "error"):
            banner.remove_class(f"status-{state}")
            compact_status.remove_class(f"status-{state}")
            operation_summary.remove_class(f"status-{state}")
        banner.add_class(f"status-{self.operation_status}")
        compact_status.add_class(f"status-{self.operation_status}")
        operation_summary.add_class(f"status-{self.operation_status}")

        prefixes = {
            "idle": "READY",
            "success": "DONE",
            "warning": "CHECK",
            "error": "FAILED",
        }
        if self.operation_status == "running":
            frame = ("|", "/", "-", "\\")[self.spinner_index % 4]
            elapsed = self._elapsed_text()
            primary = f"{frame} RUNNING  {self.operation_title}"
            if self.operation_step:
                primary += f" | {self.operation_step}"
            if elapsed:
                primary += f" | {elapsed}"
        else:
            primary = f"{prefixes.get(self.operation_status, 'READY')}  {self.operation_title}"
        self.query_one("#operation-primary", Static).update(self._fit_line(primary))
        self.query_one("#operation-secondary", Static).update(self._fit_line(self.operation_latest))

        elapsed = self._elapsed_text() or "--:--"
        summary = "\n".join(
            (
                f"Status: {self.operation_status}",
                f"Task: {self.operation_title}",
                f"Step: {self.operation_step or 'None'}",
                f"Elapsed: {elapsed}",
                "",
                "Latest:",
                self.operation_latest,
            )
        )
        operation_summary.update(summary)
        if self.operation_status != "idle":
            compact_status.update(self._compact_operation_status())

    def _compact_operation_status(self) -> str:
        """Render durable, main-screen progress in compact terminal layouts."""

        if self.operation_status == "running":
            elapsed = self._elapsed_text() or "00:00"
            step = self.operation_step or "Starting installer"
            return "\n".join(
                (
                    self._fit_line(f"INSTALLING | Installing python packages  {elapsed}"),
                    self._progress_bar(),
                    self._fit_line(step),
                    self._fit_line(self.operation_latest),
                )
            )
        prefix = {
            "success": "DONE",
            "warning": "CHECK",
            "error": "FAILED",
        }.get(self.operation_status, "READY")
        return "\n".join(
            (
                self._fit_line(f"{prefix}  {self.operation_title}"),
                self._fit_line(self.operation_latest),
            )
        )

    def _sync_action_buttons(self) -> None:
        operation = self.active_operation
        installing = operation == "install"
        checking = operation == "doctor"

        exit_button = self.query_one("#exit-button", Button)
        exit_button.label = "Cancel" if installing else "Exit"
        exit_button.set_class(installing, "danger")

        refresh = self.query_one("#refresh-button", Button)
        refresh.label = "Checking..." if checking else "Refresh"
        refresh.disabled = bool(operation)
        launch = self.query_one("#launch-button", Button)
        reports_enabled = self.query_one("#reports-enabled-checkbox", Checkbox).value
        translation_enabled = self.query_one("#translation-enabled-checkbox", Checkbox).value
        translation_provider = str(self.query_one("#translation-provider-select", Select).value or "sidecar")
        translation_sidecar_enabled = translation_enabled and translation_provider == "sidecar"
        live_running = self._process_is_running(self.live_server_process) or self.live_server_state == "running"
        reports_running = self._process_is_running(self.reports_server_process) or self.reports_server_state == "running"
        translation_running = (
            self._process_is_running(self.translation_server_process)
            or self.translation_server_state == "running"
        )
        incompatible = self._preview_compatibility_error() is not None
        if live_running:
            launch.label = "Live running"
        elif reports_enabled and translation_sidecar_enabled:
            launch.label = "Launch + services"
        elif reports_enabled:
            launch.label = "Launch + reports"
        elif translation_sidecar_enabled:
            launch.label = "Launch + translation"
        else:
            launch.label = "Launch"
        launch.disabled = bool(operation) or live_running or incompatible
        self.query_one("#view-activity-button", Button).disabled = False

        install = self.query_one("#install-button", Button)
        install.label = "Installing" if installing else "Install"
        install.disabled = bool(operation) or incompatible

        self.query_one("#target-select", RadioSet).disabled = bool(operation)
        self.query_one("#realtime-select", RadioSet).disabled = bool(operation) or self._selected_target() == "server"
        self.query_one("#translation-install-select", Select).disabled = bool(operation)
        self.query_one("#quick-language-select", Select).disabled = bool(operation)
        self.query_one("#live-speakers-checkbox", Checkbox).disabled = bool(operation)
        self.query_one("#save-settings", Button).disabled = bool(operation)
        self.query_one("#save-reports-settings", Button).disabled = bool(operation)
        self.query_one("#save-translation-settings", Button).disabled = bool(operation)
        reports_button = self.query_one("#start-reports-button", Button)
        reports_button.label = "Reports running" if reports_running else "Start reports now"
        reports_button.disabled = bool(operation) or reports_running
        translation_button = self.query_one("#start-translation-button", Button)
        translation_button.label = "Translation running" if translation_running else "Start translation now"
        translation_button.disabled = (
            bool(operation)
            or translation_running
            or not translation_sidecar_enabled
        )
        self.query_one("#quick-doctor", Button).disabled = bool(operation)
        self.query_one("#deep-doctor", Button).disabled = bool(operation)
        cancel = self.query_one("#cancel-operation", Button)
        cancel.styles.display = "block" if installing else "none"

    def _append_log(self, line: str) -> None:
        if line:
            self.query_one("#activity-log", RichLog).write(line)
            if self.active_operation == "install" and not line.startswith("> "):
                self.operation_latest = line
                self.operation_step = self._install_step_for_line(line)
                self._render_operation()

    def _install_step_for_line(self, line: str) -> str:
        lowered = line.lower()
        if any(token in lowered for token in ("pytorch", "torch", "torchaudio", "cuda")):
            return "Installing PyTorch runtime"
        if any(token in lowered for token in ("sherpa", "nemotron", "model download", "model archive")):
            return "Preparing Nemotron realtime ASR"
        if any(token in lowered for token in ("kroko", "docker", "cmake", "native runtime")):
            return "Preparing Kroko realtime ASR"
        if any(
            token in lowered
            for token in (
                "collecting ",
                "downloading ",
                "installing collected",
                "building wheel",
                "successfully installed",
                "pip install",
            )
        ):
            return "Installing Python packages"
        if any(token in lowered for token in ("saved ", "configuration", "profile")):
            return "Saving configuration"
        if any(token in lowered for token in ("check", "doctor", "readiness")):
            return "Checking installed components"
        return self.operation_step or "Running installer"

    def _install_command(self, plan: backend.InstallPlan) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "whospeaks_cli",
            "install",
            "--target",
            plan.target,
            "--yes",
        ]
        if plan.target != "server":
            command.extend(["--realtime-preview-engine", plan.realtime_preview_engine])
            if plan.realtime_preview_model_preset:
                command.extend(["--realtime-preview-model-preset", plan.realtime_preview_model_preset])
            if plan.realtime_preview_engine == "sherpa_onnx":
                model_dir = self.query_one("#realtime-model-dir-input", Input).value.strip()
                if model_dir:
                    command.extend(["--realtime-preview-model-dir", model_dir])
        command.extend(["--translation-model-profile", plan.translation_model_profile])
        return command

    def _save_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("language", self.query_one("#language-select", Select).value),
            ("provider_preset", self.query_one("#provider-select", Select).value),
            ("live_speaker_assignment", self.query_one("#live-speakers-checkbox", Checkbox).value),
            ("model", self.query_one("#model-input", Input).value),
            ("device", self.query_one("#device-select", Select).value),
            ("compute_type", self.query_one("#compute-input", Input).value),
            ("host", self.query_one("#host-input", Input).value),
            ("port", self.query_one("#port-input", Input).value),
            ("remote_asr_url", self.query_one("#asr-url-input", Input).value),
            ("remote_embeddings_url", self.query_one("#embeddings-url-input", Input).value),
            ("realtime_preview_engine", self.query_one("#realtime-engine-select", Select).value),
            ("realtime_preview_model_preset", self.query_one("#realtime-preset-select", Select).value),
            ("realtime_preview_model_dir", self.query_one("#realtime-model-dir-input", Input).value),
        ]
        try:
            updated = backend.apply_profile_updates(self.profile, updates)
        except SystemExit as exc:
            self._set_feedback("error", "Settings were not saved", str(exc))
            self.notify(str(exc), title="Invalid settings", severity="error")
            return False
        backend.update_profile_in_place(self.profile, updated)
        try:
            path = backend.save_profile(self.profile)
        except OSError as exc:
            self._append_log(f"Could not save settings: {exc}")
            self._set_feedback("error", "Settings were not saved", str(exc))
            self.notify(str(exc), title="Could not save settings", severity="error")
            return False
        self._append_log(f"Saved settings: {path}")
        if notify:
            self._set_feedback("success", "Settings saved", str(path))
            self.notify("Settings saved", title="WhoSpeaks")
        return True

    def _save_reports_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("reports_enabled", self.query_one("#reports-enabled-checkbox", Checkbox).value),
            ("reports_port", self.query_one("#reports-port-input", Input).value),
            ("report_language", self.query_one("#report-language-select", Select).value),
            ("report_llm_provider", self.query_one("#report-llm-provider-select", Select).value),
            ("report_llm_base_url", self.query_one("#report-llm-base-url-input", Input).value),
            ("report_llm_model", self.query_one("#report-llm-model-input", Input).value),
            ("report_auto_generate", self.query_one("#report-auto-generate-checkbox", Checkbox).value),
        ]
        try:
            updated = backend.apply_profile_updates(self.profile, updates)
        except SystemExit as exc:
            self._set_feedback("error", "Report settings were not saved", str(exc))
            self.notify(str(exc), title="Invalid report settings", severity="error")
            return False
        backend.update_profile_in_place(self.profile, updated)
        try:
            path = backend.save_profile(self.profile)
        except OSError as exc:
            self._append_log(f"Could not save report settings: {exc}")
            self._set_feedback("error", "Report settings were not saved", str(exc))
            self.notify(str(exc), title="Could not save report settings", severity="error")
            return False
        self._append_log(f"Saved report settings: {path}")
        self._sync_action_buttons()
        if notify:
            self._set_feedback("success", "Report settings saved", str(path))
            self.notify("Report settings saved", title="WhoSpeaks")
        return True

    def _save_translation_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("translation_enabled", self.query_one("#translation-enabled-checkbox", Checkbox).value),
            ("translation_browser_preferred", self.query_one("#translation-browser-preferred-checkbox", Checkbox).value),
            ("translation_provider", self.query_one("#translation-provider-select", Select).value),
            ("translation_target_languages", self.query_one("#translation-targets-input", Input).value),
            ("translation_max_targets", self.query_one("#translation-max-targets-input", Input).value),
            ("translation_model_profile", self.query_one("#translation-model-profile-select", Select).value),
            ("translation_model", self.query_one("#translation-model-input", Input).value),
            ("translation_base_url", self.query_one("#translation-base-url-input", Input).value),
            ("translation_api_key_env", self.query_one("#translation-api-key-env-input", Input).value),
            ("translation_region", self.query_one("#translation-region-input", Input).value),
            ("translation_python", self.query_one("#translation-python-input", Input).value),
            ("translation_port", self.query_one("#translation-port-input", Input).value),
            ("translation_device", self.query_one("#translation-device-select", Select).value),
        ]
        try:
            updated = backend.apply_profile_updates(self.profile, updates)
        except SystemExit as exc:
            self._set_feedback("error", "Translation settings were not saved", str(exc))
            self.notify(str(exc), title="Invalid translation settings", severity="error")
            return False
        backend.update_profile_in_place(self.profile, updated)
        try:
            path = backend.save_profile(self.profile)
        except OSError as exc:
            self._append_log(f"Could not save translation settings: {exc}")
            self._set_feedback("error", "Translation settings were not saved", str(exc))
            self.notify(str(exc), title="Could not save translation settings", severity="error")
            return False
        self.query_one("#translation-targets-input", Input).value = self.profile.translation_target_languages
        self.query_one("#translation-max-targets-input", Input).value = str(self.profile.translation_max_targets)
        self._append_log(f"Saved translation settings: {path}")
        self._sync_translation_settings()
        self._sync_action_buttons()
        if notify:
            self._set_feedback("success", "Translation settings saved", str(path))
            self.notify("Translation settings saved", title="WhoSpeaks")
        return True

    def _start_reports_server(self, *, save_settings: bool = True) -> None:
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return
        if save_settings and not self._save_reports_settings(notify=False):
            return
        command = backend.build_reports_command(
            self.profile,
            port=self.profile.reports_port,
            report_language=self.profile.report_language,
            llm_provider=self.profile.report_llm_provider,
            llm_base_url=self.profile.report_llm_base_url,
            llm_model=self.profile.report_llm_model,
            auto_generate=self.profile.report_auto_generate,
        )
        if self._start_server_process("reports", command):
            self._set_feedback(
                "success",
                "Reports server starting in another window",
                f"Open http://{self.profile.host}:{self.profile.reports_port}/",
            )

    def _start_translation_server(self, *, save_settings: bool = True) -> bool:
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return False
        if save_settings and not self._save_translation_settings(notify=False):
            return False
        if not self.profile.translation_enabled:
            self.notify("Enable translation before starting its local server", severity="warning")
            return False
        if self.profile.translation_provider != "sidecar":
            self.notify("The selected provider runs through the live server and has no sidecar", severity="warning")
            return False
        command = backend.build_translation_command(self.profile)
        if not self._start_server_process("translation", command):
            return False
        self._set_feedback(
            "success",
            "Translation server starting in another window",
            f"Translation API on http://{self.profile.host}:{self.profile.translation_port}/",
        )
        return True

    def _start_live_server(self) -> bool:
        command = backend.build_launch_command(self.profile)
        if not self._start_server_process("live", command):
            return False
        self._set_feedback(
            "success",
            "Live server starting in another window",
            f"Open http://{self.profile.host}:{self.profile.port}/",
        )
        return True

    @on(RadioSet.Changed, "#target-select")
    def target_changed(self) -> None:
        self._update_plan()

    @on(RadioSet.Changed, "#realtime-select")
    def realtime_changed(self) -> None:
        engine = self._selected_realtime_engine()
        settings_select = self.query_one("#realtime-engine-select", Select)
        if settings_select.value != engine:
            settings_select.value = engine
        self._update_plan()
        self._sync_preview_compatibility()
        self._sync_action_buttons()

    @on(Select.Changed, "#translation-install-select")
    def translation_install_changed(self) -> None:
        self._update_plan()

    @on(Select.Changed, "#quick-language-select")
    def quick_language_changed(self, event: Select.Changed) -> None:
        language = str(event.value or self.profile.language)
        current = str(self.query_one("#quick-language-select", Select).value or self.profile.language)
        if language != current:
            return
        settings_select = self.query_one("#language-select", Select)
        if settings_select.value != language:
            settings_select.value = language
        if self.language_selection_changed or language != self.profile.language:
            self.language_selection_changed = True
            self._select_recommended_engine(language)

    @on(Select.Changed, "#language-select")
    def settings_language_changed(self, event: Select.Changed) -> None:
        language = str(event.value or self.profile.language)
        current = str(self.query_one("#language-select", Select).value or self.profile.language)
        if language != current:
            return
        quick_select = self.query_one("#quick-language-select", Select)
        if quick_select.value != language:
            quick_select.value = language
        if self.language_selection_changed or language != self.profile.language:
            self.language_selection_changed = True
            self._select_recommended_engine(language)

    @on(Select.Changed, "#realtime-engine-select")
    def realtime_engine_changed(self, event: Select.Changed) -> None:
        engine = str(event.value or "off")
        # Programmatic language recommendations can enqueue several Select
        # events. Ignore a superseded event instead of letting an older
        # recommendation overwrite the currently visible selection.
        current = str(self.query_one("#realtime-engine-select", Select).value or "off")
        if engine != current:
            return
        self._sync_realtime_settings()
        self._select_realtime_engine(engine)
        self._update_plan()
        self._sync_preview_compatibility()
        self._sync_action_buttons()

    @on(Checkbox.Changed, "#reports-enabled-checkbox")
    def reports_enabled_changed(self) -> None:
        self._sync_action_buttons()

    @on(Checkbox.Changed, "#translation-enabled-checkbox")
    def translation_enabled_changed(self) -> None:
        self._sync_translation_settings()
        self._sync_action_buttons()

    @on(Select.Changed, "#translation-provider-select")
    def translation_provider_changed(self) -> None:
        self._sync_translation_settings()
        self._sync_action_buttons()

    def _sync_realtime_settings(self) -> None:
        engine = str(self.query_one("#realtime-engine-select", Select).value or "off")
        enabled = engine == "sherpa_onnx"
        self.query_one("#realtime-preset-select", Select).disabled = not enabled
        self.query_one("#realtime-model-dir-input", Input).disabled = not enabled

    def _sync_translation_settings(self) -> None:
        provider = str(self.query_one("#translation-provider-select", Select).value or "sidecar")
        local_model = provider in {"sidecar", "transformers"}
        sidecar = provider == "sidecar"
        self.query_one("#translation-model-profile-select", Select).disabled = not local_model
        self.query_one("#translation-device-select", Select).disabled = not local_model
        self.query_one("#translation-port-input", Input).disabled = not sidecar
        self.query_one("#translation-python-input", Input).disabled = not sidecar
        self.query_one("#translation-base-url-input", Input).disabled = provider in {"transformers", "mock"}
        self.query_one("#translation-api-key-env-input", Input).disabled = provider in {
            "sidecar",
            "transformers",
            "reports_llm",
            "mock",
        }
        self.query_one("#translation-region-input", Input).disabled = provider != "azure_translator"
        self.query_one("#translation-model-input", Input).disabled = provider == "mock"

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id in {"refresh-button", "quick-doctor"}:
            self.run_doctor_worker(False)
        elif button_id == "deep-doctor":
            self.run_doctor_worker(True)
        elif button_id == "save-settings":
            self._save_settings()
        elif button_id == "save-reports-settings":
            self._save_reports_settings()
        elif button_id == "save-translation-settings":
            self._save_translation_settings()
        elif button_id == "start-reports-button":
            self._start_reports_server()
        elif button_id == "start-translation-button":
            self._start_translation_server()
        elif button_id == "install-button":
            self._request_install()
        elif button_id == "launch-button":
            self.action_launch()
        elif button_id == "exit-button":
            if self.active_operation == "install":
                self._request_cancel()
            else:
                self.exit()
        elif button_id == "view-activity-button":
            self._show_activity()
        elif button_id == "cancel-operation":
            self._request_cancel()
        elif button_id == "clear-log":
            self.query_one("#activity-log", RichLog).clear()
            self._set_feedback("idle", "Activity cleared", "The activity log is now empty.")

    def _request_install(self) -> None:
        compatibility_error = self._sync_preview_compatibility()
        if compatibility_error:
            self.notify(compatibility_error, title="Unsupported live text language", severity="warning")
            return
        if self.active_operation:
            self.notify("Another operation is already running", severity="warning")
            return
        if not self._save_settings(notify=False):
            return
        plan = self._selected_plan()
        candidate = backend.Profile.from_mapping(self.profile.as_dict())
        backend.configure_profile_for_install(candidate, plan)
        backend.update_profile_in_place(self.profile, candidate)
        try:
            backend.save_profile(self.profile)
        except OSError as exc:
            self._append_log(f"Could not save install profile: {exc}")
            self._set_feedback("error", "Installation could not start", str(exc))
            self.notify(str(exc), title="Could not start installation", severity="error")
            return
        command = self._install_command(plan)
        self.pending_install_command = command
        self.pending_install_title = plan.title
        self.push_screen(
            ConfirmInstallScreen(plan, command),
            self._install_confirmed,
        )

    def _install_confirmed(self, confirmed: bool | None) -> None:
        if confirmed and self.pending_install_command:
            command = self.pending_install_command
            title = self.pending_install_title
            self.pending_install_command = None
            self.pending_install_title = ""
            self.start_install_worker(command, title=title)
        else:
            self.pending_install_command = None
            self.pending_install_title = ""
            self._set_feedback(
                "warning",
                "Installation cancelled before start",
                "No installation command was run.",
            )

    def action_refresh(self) -> None:
        self.run_doctor_worker(False)

    def action_launch(self) -> None:
        compatibility_error = self._sync_preview_compatibility()
        if compatibility_error:
            self.notify(compatibility_error, title="Unsupported live text language", severity="warning")
            return
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return
        if not (
            self._save_settings(notify=False)
            and self._save_reports_settings(notify=False)
            and self._save_translation_settings(notify=False)
        ):
            return
        if self.profile.reports_enabled and not self._process_is_running(self.reports_server_process):
            self._start_reports_server(save_settings=False)
        if (
            self.profile.translation_enabled
            and self.profile.translation_provider == "sidecar"
            and not self._process_is_running(self.translation_server_process)
        ):
            self._start_translation_server(save_settings=False)
        self._start_live_server()

    def _show_activity(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "activity-tab"
        self.call_after_refresh(self.query_one("#clear-log", Button).focus)

    def run_doctor_worker(self, deep: bool) -> None:
        if self.active_operation:
            self.notify("Another operation is already running", severity="warning")
            return
        title = "Running complete diagnostics" if deep else "Checking system readiness"
        self._start_operation("doctor", title, "Inspecting installed components")
        self._append_log("Starting complete diagnostics..." if deep else "Starting readiness check...")
        self._run_doctor_worker(deep)

    @work(thread=True, exclusive=True, group="doctor")
    def _run_doctor_worker(self, deep: bool) -> None:
        try:
            report = self.doctor_runner(self.profile, self.profile.mode, deep=deep)
        except Exception as exc:
            self.call_from_thread(self._append_log, f"Diagnostics failed: {type(exc).__name__}: {exc}")
            self.call_from_thread(self.notify, str(exc), title="Diagnostics failed", severity="error")
            self.call_from_thread(
                self._finish_operation,
                "error",
                "Diagnostics failed",
                f"{type(exc).__name__}: {exc}",
            )
        else:
            self.call_from_thread(self._render_report, report)
            readiness = backend.report_readiness_line(report)
            self.call_from_thread(self._append_log, readiness)
            statuses = {check.status for check in report.checks}
            if "fail" in statuses:
                status, result_title = "error", "Readiness check found required fixes"
            elif "warn" in statuses:
                status, result_title = "warning", "Readiness check found warnings"
            else:
                status, result_title = "success", "Readiness check completed"
            self.call_from_thread(self._finish_operation, status, result_title, readiness)

    def start_install_worker(self, command: list[str], *, title: str = "Selected setup") -> None:
        if self.active_operation:
            self.notify("Another operation is already running", severity="warning")
            return
        self.install_cancelled = False
        self._start_operation("install", f"Install: {title}", "Starting installer")
        self._run_install_worker(command)

    @work(thread=True, exclusive=True, group="install")
    def _run_install_worker(self, command: list[str]) -> None:
        self.call_from_thread(self._append_log, "")
        self.call_from_thread(self._append_log, f"> {backend.format_command(command)}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            process = self.popen_factory(command, **kwargs)
            self.install_process = process
            if self.install_cancelled and process.poll() is None:
                self._terminate_process_tree(process)
            if process.stdout is not None:
                for line in process.stdout:
                    self.call_from_thread(self._append_log, line.rstrip())
            return_code = int(process.wait())
        except Exception as exc:
            return_code = 1
            self.call_from_thread(self._append_log, f"Installer failed: {type(exc).__name__}: {exc}")
        finally:
            self.install_process = None

        if self.install_cancelled:
            self.call_from_thread(self._append_log, "Installation cancelled.")
            self.call_from_thread(self.notify, "Installation cancelled", severity="warning")
            self.call_from_thread(
                self._finish_operation,
                "warning",
                "Installation cancelled",
                "The running installer was stopped.",
            )
        elif return_code == 0:
            self.call_from_thread(self._append_log, "Installation completed.")
            self.call_from_thread(self.notify, "Installation completed", title="WhoSpeaks")
            self.profile = backend.load_profile()
            self.call_from_thread(
                self._finish_operation,
                "success",
                "Installation completed",
                "Packages were installed. Verifying system readiness next.",
            )
        else:
            self.call_from_thread(self._append_log, f"Installation stopped with exit code {return_code}.")
            self.call_from_thread(self.notify, f"Installer exit code {return_code}", title="Installation failed", severity="error")
            self.call_from_thread(
                self._finish_operation,
                "error",
                "Installation failed",
                f"Installer stopped with exit code {return_code}. Open Activity for details.",
            )
        if not self.install_cancelled and return_code == 0:
            self.call_from_thread(self.run_doctor_worker, False)

    def _request_cancel(self) -> None:
        if self.active_operation != "install":
            return
        self.install_cancelled = True
        self.operation_step = "Cancelling installation"
        self.operation_latest = "Stopping the installer and its child processes..."
        self._render_operation()
        self._append_log("Cancelling installation...")
        self.cancel_install_worker()

    @work(thread=True, exclusive=True, group="cancel")
    def cancel_install_worker(self) -> None:
        process = self.install_process
        if process is None or process.poll() is not None:
            return
        self._terminate_process_tree(process)

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as exc:
            self.call_from_thread(self._append_log, f"Cancellation failed: {exc}")
            process.terminate()


def run_setup_app(profile: backend.Profile | None = None) -> str | None:
    """Run the Textual application and return its requested follow-up action."""

    return WhoSpeaksSetupApp(profile).run()
