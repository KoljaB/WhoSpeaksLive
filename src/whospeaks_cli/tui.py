"""Textual setup, diagnostics, and launcher interface for WhoSpeaks."""

from __future__ import annotations

import os
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


class ConfirmInstallScreen(ModalScreen[bool]):
    """Confirm a concrete installation plan before starting subprocesses."""

    BINDINGS = [Binding("escape", "cancel", show=False)]

    def __init__(self, plan: backend.InstallPlan, command: list[str]) -> None:
        super().__init__()
        self.plan = plan
        self.command = command

    def compose(self) -> ComposeResult:
        realtime = "Enabled with Kroko" if self.plan.install_kroko else "Disabled"
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
        background: #101416;
        color: #e9eef0;
    }

    #app-body {
        height: 1fr;
    }

    #title-bar {
        height: 1;
        padding: 0 1;
        background: #153f42;
        color: #ffffff;
    }

    #app-title {
        width: 20;
        height: 1;
        text-style: bold;
    }

    #readiness-text {
        width: 1fr;
        height: 1;
        color: #d3e2e2;
        text-align: center;
    }

    #app-meta {
        width: auto;
        height: 1;
        color: #b9d6d5;
    }

    #operation-banner {
        height: 3;
        padding: 0 1;
        background: #1a2226;
        border-bottom: solid #3d4a50;
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
        background: #123d40;
        border-bottom: solid #51b9b0;
    }

    #operation-banner.status-running #operation-primary {
        color: #91e0d9;
    }

    #operation-banner.status-success {
        background: #183527;
        border-bottom: solid #5eae78;
    }

    #operation-banner.status-success #operation-primary {
        color: #8dd4a3;
    }

    #operation-banner.status-warning {
        background: #3a3019;
        border-bottom: solid #d3a642;
    }

    #operation-banner.status-warning #operation-primary {
        color: #f2c868;
    }

    #operation-banner.status-error {
        background: #3b2024;
        border-bottom: solid #cc6570;
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
        height: 2;
        background: #151b1e;
        color: #aab7bc;
    }

    Tab {
        height: 2;
        padding: 0 2;
    }

    Tab.-active {
        color: #ffffff;
        background: #246066;
        text-style: bold;
    }

    .section-title {
        height: 1;
        color: #ffffff;
        text-style: bold;
    }

    #setup-options {
        height: 2;
        background: #171d20;
        border-bottom: solid #3d4a50;
    }

    #target-row {
        width: 1fr;
        height: 2;
        align-vertical: middle;
    }

    #target-label {
        width: 9;
        height: 1;
        margin-left: 1;
        color: #b9c5c9;
    }

    #target-select {
        width: 1fr;
        height: 2;
        layout: horizontal;
        align-vertical: middle;
        background: transparent;
        border: none;
    }

    #target-select RadioButton {
        width: auto;
        height: 1;
        padding-right: 1;
    }

    #kroko-row {
        width: 20;
        height: 2;
        padding: 0 1;
        background: #20282c;
        border-left: solid #3d4a50;
        align-vertical: middle;
    }

    #kroko-switch {
        width: 1fr;
        height: 1;
        margin-left: 1;
    }

    #compact-plan {
        display: none;
        height: 3;
        padding: 0 1;
        background: #141a1d;
        color: #bdc9cd;
        border-bottom: solid #3d4a50;
    }

    #setup-workspace {
        height: 1fr;
    }

    #setup-state {
        width: 1fr;
        height: 1fr;
        padding-top: 1;
    }

    #setup-side {
        width: 39;
        min-width: 34;
        height: 1fr;
        padding: 1 0 0 1;
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

    #setup-actions, #doctor-actions, #settings-actions, #activity-actions {
        dock: bottom;
        height: 3;
        align-horizontal: right;
        background: #101416;
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
        background: #13191c;
        border: solid #3d4a50;
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
        padding-top: 1;
        padding-bottom: 4;
    }

    .field {
        height: 5;
    }

    .field Label {
        height: 2;
        color: #b8c4ca;
    }

    Input, Select {
        height: 3;
        background: #1b2227;
        border: solid #46545d;
    }

    Input:focus, Select:focus {
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

    Screen.compact #setup-side {
        display: none;
    }

    Screen.compact #compact-plan {
        display: block;
    }

    Screen.compact #settings-grid {
        grid-size: 1;
        grid-columns: 1fr;
    }

    Screen.narrow #setup-options {
        layout: vertical;
        height: 4;
    }

    Screen.narrow #target-row, Screen.narrow #kroko-row {
        width: 1fr;
        height: 2;
    }

    Screen.narrow #kroko-row {
        border-left: none;
        border-top: solid #3d4a50;
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

    def compose(self) -> ComposeResult:
        target = {"local": "local", "remote": "core", "server": "server"}.get(self.profile.mode, "local")
        preview_enabled = backend.preview_engine_uses_kroko(self.profile)
        language_options = [
            (config.display_name, code)
            for code, config in backend.SUPPORTED_LANGUAGE_CONFIGS.items()
        ]
        provider_options = [
            (preset.name, preset_id)
            for preset_id, preset in backend.PROVIDER_PRESETS.items()
        ]

        with Vertical(id="app-body"):
            with Horizontal(id="title-bar"):
                yield Static("WhoSpeaks Setup", id="app-title", markup=False)
                yield Static("Readiness not checked", id="readiness-text", markup=False)
                yield Static(f"v{backend.__version__}", id="app-meta", markup=False)
            with Vertical(id="operation-banner", classes="status-idle"):
                yield Static(id="operation-primary", markup=False)
                yield Static(id="operation-secondary", markup=False)
            with TabbedContent(initial="setup-tab", id="main-tabs"):
                with TabPane("Setup", id="setup-tab"):
                    with Horizontal(id="setup-options"):
                        with Horizontal(id="target-row"):
                            yield Label("Install:", id="target-label")
                            with RadioSet(
                                RadioButton("Full local", id="target-local", value=target == "local", compact=True),
                                RadioButton("Remote core", id="target-core", value=target == "core", compact=True),
                                RadioButton("Server", id="target-server", value=target == "server", compact=True),
                                id="target-select",
                            ):
                                pass
                        with Horizontal(id="kroko-row"):
                            yield Checkbox(
                                "Kroko text",
                                preview_enabled,
                                id="kroko-switch",
                                compact=True,
                            )
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
                                yield Label("Speaker provider")
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
                with TabPane("Activity", id="activity-tab"):
                    yield RichLog(id="activity-log", wrap=True, markup=False, max_lines=5000)
                    with Horizontal(id="activity-actions"):
                        yield Button("Clear", id="clear-log")
                        yield Button("Cancel operation", id="cancel-operation")

    def on_mount(self) -> None:
        self._configure_tables()
        self._apply_size_classes(self.size.width)
        self._update_plan(announce=False)
        self._render_report(self.report)
        self._render_operation()
        self._sync_action_buttons()
        self.set_interval(0.25, self._tick_operation)
        self._append_log(f"WhoSpeaks {backend.__version__}")
        self._append_log(f"Profile: {backend.config_path()}")
        if self.auto_doctor:
            self.run_doctor_worker(False)
        else:
            self.query_one("#readiness-text", Static).update("Readiness not checked")

    def on_resize(self, event: events.Resize) -> None:
        self._apply_size_classes(event.size.width)
        self._render_operation()

    def _apply_size_classes(self, width: int) -> None:
        if width < 112:
            self.screen.add_class("compact")
        else:
            self.screen.remove_class("compact")
        if width < 76:
            self.screen.add_class("narrow")
        else:
            self.screen.remove_class("narrow")

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

    def _selected_plan(self) -> backend.InstallPlan:
        target = self._selected_target()
        install_kroko = self.query_one("#kroko-switch", Checkbox).value and target != "server"
        return backend.install_plan_for_target(target, install_kroko)

    def _update_plan(self, *, announce: bool = True) -> None:
        target = self._selected_target()
        kroko_switch = self.query_one("#kroko-switch", Checkbox)
        kroko_switch.disabled = target == "server"
        if target == "server" and kroko_switch.value:
            kroko_switch.value = False
        plan = self._selected_plan()
        realtime = "Kroko live text enabled" if plan.install_kroko else "Live text disabled"
        component_lines, compact_components = self._plan_components(plan)
        summary = "\n".join(
            (
                plan.title,
                "",
                *component_lines,
                "",
                f"Realtime text: {'Kroko enabled' if plan.install_kroko else 'disabled'}",
            )
        )
        self.query_one("#plan-summary", Static).update(summary)
        self.query_one("#compact-plan", Static).update(
            f"Plan: {plan.title}\n{'Kroko text on' if plan.install_kroko else 'Kroko text off'} | {compact_components}"
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

    def _render_operation(self) -> None:
        banner = self.query_one("#operation-banner", Vertical)
        for state in ("idle", "running", "success", "warning", "error"):
            banner.remove_class(f"status-{state}")
        banner.add_class(f"status-{self.operation_status}")

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
        self.query_one("#operation-summary", Static).update(summary)

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
        self.query_one("#launch-button", Button).disabled = bool(operation)
        self.query_one("#view-activity-button", Button).disabled = False

        install = self.query_one("#install-button", Button)
        install.label = "Installing..." if installing else "Install"
        install.disabled = bool(operation)

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
            command.append("--with-kroko" if plan.install_kroko else "--without-kroko")
        return command

    def _save_settings(self, *, notify: bool = True) -> bool:
        updates: list[tuple[str, Any]] = [
            ("language", self.query_one("#language-select", Select).value),
            ("provider_preset", self.query_one("#provider-select", Select).value),
            ("model", self.query_one("#model-input", Input).value),
            ("device", self.query_one("#device-select", Select).value),
            ("compute_type", self.query_one("#compute-input", Input).value),
            ("host", self.query_one("#host-input", Input).value),
            ("port", self.query_one("#port-input", Input).value),
            ("remote_asr_url", self.query_one("#asr-url-input", Input).value),
            ("remote_embeddings_url", self.query_one("#embeddings-url-input", Input).value),
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

    @on(RadioSet.Changed, "#target-select")
    def target_changed(self) -> None:
        self._update_plan()

    @on(Checkbox.Changed, "#kroko-switch")
    def kroko_changed(self) -> None:
        self._update_plan()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id in {"refresh-button", "quick-doctor"}:
            self.run_doctor_worker(False)
        elif button_id == "deep-doctor":
            self.run_doctor_worker(True)
        elif button_id == "save-settings":
            self._save_settings()
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
        if self.active_operation:
            self.notify("Wait for the current operation or cancel it", severity="warning")
            return
        if self._save_settings(notify=False):
            self.exit("launch")

    def _show_activity(self) -> None:
        self.query_one("#main-tabs", TabbedContent).active = "activity-tab"

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
                process.terminate()
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
