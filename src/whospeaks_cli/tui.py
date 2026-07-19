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
from window.meeting_server_support import LLM_PROVIDER_OPTIONS

from . import main as backend
from .tui_actions import ProfileActionsMixin
from .tui_layout import compose_setup_app, provider_preset_label
from .profiles import TRANSLATION_PROVIDER_OPTIONS
from .tui_servers import ServerLifecycleMixin
from .tui_styles import APP_CSS
from .tui_state import PendingAction, ServerSupervisor, SetupCoordinator
from .tui_workers import SetupWorkersMixin


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
    return "Live text off"


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


class WhoSpeaksSetupApp(
    ServerLifecycleMixin,
    ProfileActionsMixin,
    SetupWorkersMixin,
    App[str],
):
    """Interactive setup and operational dashboard."""

    TITLE = "WhoSpeaks Setup"
    SUB_TITLE = "Installation and system readiness"
    BINDINGS = [
        Binding("ctrl+r", "refresh", show=False),
        Binding("ctrl+l", "launch", show=False),
        Binding("ctrl+q", "quit", show=False),
    ]

    CSS = APP_CSS

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
        self._coordinator = SetupCoordinator()
        self._servers = ServerSupervisor()
        self._managed_service_started_at: dict[str, float] = {}
        self.install_process: subprocess.Popen[str] | None = None
        self.last_server_probe_at = 0.0
        self.language_selection_changed = False
        self._report_provider_value = self.profile.report_llm_provider
        self._translation_provider_value = self.profile.translation_provider

    @property
    def active_operation(self) -> str:
        return self._coordinator.snapshot.operation.name

    @property
    def operation_status(self) -> str:
        return self._coordinator.snapshot.operation.status

    @property
    def operation_title(self) -> str:
        return self._coordinator.snapshot.operation.title

    @property
    def operation_step(self) -> str:
        return self._coordinator.snapshot.operation.step

    @operation_step.setter
    def operation_step(self, value: str) -> None:
        """Compatibility seam; the coordinator remains the state writer."""

        self._coordinator.update_progress(step=value)

    @property
    def operation_latest(self) -> str:
        return self._coordinator.snapshot.operation.latest

    @property
    def operation_started_at(self) -> float | None:
        return self._coordinator.snapshot.operation.started_at

    @property
    def spinner_index(self) -> int:
        return self._coordinator.snapshot.operation.spinner_index

    @property
    def install_cancelled(self) -> bool:
        return self._coordinator.snapshot.operation.cancel_requested

    @property
    def pending_install_command(self) -> list[str] | None:
        pending = self._coordinator.snapshot.pending_install
        return list(pending.command) if pending is not None else None

    @property
    def pending_install_title(self) -> str:
        pending = self._coordinator.snapshot.pending_install
        return pending.title if pending is not None else ""

    @property
    def live_server_process(self) -> object | None:
        return self._servers.process("live")

    @property
    def reports_server_process(self) -> object | None:
        return self._servers.process("reports")

    @property
    def translation_server_process(self) -> object | None:
        return self._servers.process("translation")

    @property
    def live_server_state(self) -> str:
        return self._servers.state("live").status

    @property
    def reports_server_state(self) -> str:
        return self._servers.state("reports").status

    @property
    def translation_server_state(self) -> str:
        return self._servers.state("translation").status

    def compose(self) -> ComposeResult:
        yield from compose_setup_app(self)

    def on_mount(self) -> None:
        self._configure_tables()
        self._apply_size_classes(self.size.width, self.size.height)
        self._sync_deployment_controls()
        self._sync_realtime_settings()
        self._sync_speaker_provider_settings()
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

    def on_unmount(self) -> None:
        owned: list[object] = []
        for kind in ("macos_asr", "macos_embeddings", "translation", "reports", "live"):
            if self._servers.state(kind).ownership != "app":
                continue
            process = self._servers.process(kind)
            if self._servers.return_code(process) is None and process is not None:
                owned.append(process)
        backend.terminate_service_processes(owned)

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
            "target-macos": "macos",
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
        installer_backend = self._selected_installer_backend()
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
                f"Package installer: {installer_backend}",
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
        if plan.target == "macos":
            return (
                ("Browser controller", "Managed MLX ASR", "Managed MPS embeddings"),
                "controller + MLX ASR + MPS embeddings",
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
        self._coordinator.start_operation(name, title, step)
        self._render_operation()
        self._sync_action_buttons()

    def _finish_operation(self, status: str, title: str, detail: str) -> None:
        self._coordinator.finish_operation(status, title, detail)
        self._render_operation()
        self._sync_action_buttons()

    def _set_feedback(self, status: str, title: str, detail: str) -> None:
        if not self._coordinator.set_feedback(status, title, detail):
            return
        self._render_operation()

    def _tick_operation(self) -> None:
        if self.active_operation:
            self._coordinator.tick()
            self._render_operation()
        self._refresh_server_states()


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
        server_target = self._selected_target() == "server"
        if server_target:
            launch.label = "Server install only"
        elif live_running:
            launch.label = "Live running"
        elif reports_enabled and translation_sidecar_enabled:
            launch.label = "Launch + services"
        elif reports_enabled:
            launch.label = "Launch + intelligence"
        elif translation_sidecar_enabled:
            launch.label = "Launch + translation"
        else:
            launch.label = "Launch"
        launch.disabled = bool(operation) or live_running or incompatible or server_target
        install = self.query_one("#install-button", Button)
        install.label = "Installing" if installing else "Install"
        install.disabled = bool(operation) or incompatible

        self.query_one("#target-select", RadioSet).disabled = bool(operation)
        self.query_one("#realtime-select", RadioSet).disabled = bool(operation) or self._selected_target() == "server"
        self.query_one("#translation-install-select", Select).disabled = bool(operation)
        self.query_one("#installer-select", Select).disabled = bool(operation)
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
                self._coordinator.update_progress(
                    latest=line,
                    step=self._install_step_for_line(line),
                )
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
                "uv pip",
                "resolved ",
                "prepared ",
                "installed ",
            )
        ):
            return "Installing Python packages"
        if any(token in lowered for token in ("saved ", "configuration", "profile")):
            return "Saving configuration"
        if any(token in lowered for token in ("check", "doctor", "readiness")):
            return "Checking installed components"
        return self.operation_step or "Running installer"


    @on(RadioSet.Changed, "#target-select")
    def target_changed(self) -> None:
        if self._selected_target() == "macos":
            self._select_realtime_engine("off")
        self._sync_deployment_controls()
        self._sync_speaker_provider_settings()
        self._update_plan()
        self._sync_action_buttons()

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

    @on(Select.Changed, "#installer-select")
    def installer_changed(self) -> None:
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

    @on(Select.Changed, "#report-llm-provider-select")
    def report_llm_provider_changed(self) -> None:
        provider = str(self.query_one("#report-llm-provider-select", Select).value or "llama_cpp")
        if provider == self._report_provider_value:
            self._sync_action_buttons()
            return
        self._report_provider_value = provider
        option = LLM_PROVIDER_OPTIONS[provider]
        base_url = self.query_one("#report-llm-base-url-input", Input)
        model = self.query_one("#report-llm-model-input", Input)
        base_url.value = str(option["default_base_url"])
        models = [str(item) for item in option.get("models") or []]
        model.value = models[0] if models else ""
        self._sync_action_buttons()

    @on(Select.Changed, "#provider-select")
    def speaker_provider_changed(self) -> None:
        self._sync_speaker_provider_settings()

    def _sync_speaker_provider_settings(self) -> None:
        selected = str(self.query_one("#provider-select", Select).value or "custom")
        final_provider = self.query_one("#embedding-provider-input", Input)
        live_provider = self.query_one("#live-embedding-provider-input", Input)
        custom = selected == "custom"
        preset = backend.PROVIDER_PRESETS.get(selected)
        if preset is not None:
            final_provider.value = preset.embedding_provider
            live_provider.value = preset.live_speaker_embedding_provider
        self.query_one("#embedding-provider-summary", Static).update(final_provider.value)
        self.query_one("#live-embedding-provider-summary", Static).update(live_provider.value)
        server = self._selected_target() == "server"
        self.query_one("#embedding-provider-edit-row").styles.display = (
            "block" if custom and not server else "none"
        )
        self.query_one("#live-embedding-provider-edit-row").styles.display = (
            "block" if custom and not server else "none"
        )
        self.query_one("#embedding-provider-summary-row").styles.display = (
            "block" if not custom and not server else "none"
        )
        self.query_one("#live-embedding-provider-summary-row").styles.display = (
            "block" if not custom and not server else "none"
        )

    def _sync_deployment_controls(self) -> None:
        target = self._selected_target()
        server = target == "server"
        local = target == "local"
        remote = target == "core"
        self.query_one("#translation-install-row").styles.display = (
            "none" if server else "block"
        )
        self.query_one("#realtime-row").styles.display = "none" if server else "block"
        self.query_one("#language-label").styles.display = "none" if server else "block"
        self.query_one("#quick-language-select").styles.display = (
            "none" if server else "block"
        )
        field_visibility = {
            "#language-select": not server,
            "#realtime-engine-select": not server,
            "#embedding-python-input": local,
            "#provider-select": not server,
            "#embedding-provider-input": not server,
            "#live-embedding-provider-input": not server,
            "#model-input": local,
            "#device-select": local,
            "#compute-input": local,
            "#vad-backend-select": not server,
            "#host-input": not server,
            "#port-input": not server,
            "#asr-url-input": remote,
            "#embeddings-url-input": remote,
            "#advanced-args-input": not server,
        }
        for selector, visible in field_visibility.items():
            widget = self.query_one(selector)
            if widget.parent is not None:
                widget.parent.styles.display = "block" if visible else "none"
        tabs = self.query_one("#main-tabs", TabbedContent)
        for tab_id in ("settings-tab", "reports-tab", "translation-tab"):
            if server:
                tabs.hide_tab(tab_id)
            else:
                tabs.show_tab(tab_id)

    @on(Checkbox.Changed, "#translation-enabled-checkbox")
    def translation_enabled_changed(self) -> None:
        self._sync_translation_settings()
        self._sync_action_buttons()

    @on(Select.Changed, "#translation-provider-select")
    def translation_provider_changed(self) -> None:
        provider = str(self.query_one("#translation-provider-select", Select).value or "sidecar")
        if provider != self._translation_provider_value:
            self._translation_provider_value = provider
            option = TRANSLATION_PROVIDER_OPTIONS[provider]
            base_url = self.query_one("#translation-base-url-input", Input)
            key_env = self.query_one("#translation-api-key-env-input", Input)
            base_url.value = str(option["default_base_url"])
            key_env.value = str(option["default_api_key_env"])
        self._sync_translation_settings()
        self._sync_action_buttons()

    def _sync_realtime_settings(self) -> None:
        if getattr(self, "_syncing_realtime_settings", False):
            return
        self._syncing_realtime_settings = True
        try:
            self._sync_realtime_settings_inner()
        finally:
            self._syncing_realtime_settings = False

    def _sync_realtime_settings_inner(self) -> None:
        engine = str(self.query_one("#realtime-engine-select", Select).value or "off")
        preset = self.query_one("#realtime-preset-select", Select)
        options = {
            "sherpa_onnx": [
                ("560 ms: stable", "nemotron-3.5-560ms-int8"),
                ("160 ms: lower latency", "nemotron-3.5-160ms-int8"),
            ],
            "kroko_onnx": [
                ("Community 64L", "community-64l"),
                ("Pro 16L", "pro-16l"),
            ],
            "off": [("No live model", "")],
        }[engine]
        allowed = {value for _label, value in options}
        current = str(preset.value or "")
        selected = current if current in allowed else options[0][1]
        preset.set_options(options)
        preset.value = selected
        visibility = {
            "#realtime-preset-select": engine != "off",
            "#realtime-model-dir-input": engine == "sherpa_onnx",
            "#realtime-python-input": engine == "kroko_onnx",
        }
        for selector, visible in visibility.items():
            widget = self.query_one(selector)
            widget.disabled = False
            if widget.parent is not None:
                widget.parent.styles.display = "block" if visible else "none"

    def _sync_translation_settings(self) -> None:
        provider = str(self.query_one("#translation-provider-select", Select).value or "sidecar")
        local_model = provider in {"sidecar", "transformers"}
        sidecar = provider == "sidecar"
        model_provider = provider in {
            "sidecar",
            "transformers",
            "reports_llm",
            "openai_compatible",
        }
        endpoint_provider = provider not in {"sidecar", "transformers", "reports_llm"}
        key_provider = provider in {
            "deepl",
            "google_cloud",
            "azure_translator",
            "libretranslate",
            "openai_compatible",
        }
        visibility = {
            "#translation-model-profile-select": local_model,
            "#translation-device-select": local_model,
            "#translation-model-input": model_provider,
            "#translation-port-input": sidecar,
            "#translation-python-input": sidecar,
            "#translation-base-url-input": endpoint_provider,
            "#translation-api-key-env-input": key_provider,
            "#translation-region-input": provider == "azure_translator",
        }
        for selector, visible in visibility.items():
            widget = self.query_one(selector)
            widget.disabled = False
            if widget.parent is not None:
                widget.parent.styles.display = "block" if visible else "none"

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
        installer_backend = self._selected_installer_backend()
        if not backend.installer_backend_available(installer_backend):
            message = "uv was selected, but it was not found on PATH. Install uv or choose pip."
            self._set_feedback("error", "Installer unavailable", message)
            self.notify(message, title="Could not start installation", severity="error")
            return
        if not self._save_settings(notify=False):
            return
        plan = self._selected_plan()
        candidate = backend.profile_for_install(self.profile, plan)
        try:
            backend.save_profile(candidate)
        except OSError as exc:
            self._append_log(f"Could not save install profile: {exc}")
            self._set_feedback("error", "Installation could not start", str(exc))
            self.notify(str(exc), title="Could not start installation", severity="error")
            return
        self.profile = candidate
        command = self._install_command(plan)
        self._coordinator.set_pending_install(command, plan.title)
        self.push_screen(
            ConfirmInstallScreen(plan, command),
            self._install_confirmed,
        )

    def _install_confirmed(self, confirmed: bool | None) -> None:
        pending = self._coordinator.take_pending_install()
        if confirmed and pending is not None:
            self.start_install_worker(list(pending.command), title=pending.title)
        else:
            self._set_feedback(
                "warning",
                "Installation cancelled before start",
                "No installation command was run.",
            )

    def action_refresh(self) -> None:
        self.run_doctor_worker(False)

    def action_launch(self) -> None:
        if self._selected_target() == "server":
            self._set_feedback(
                "warning",
                "Server deployment has no local live window",
                "Use Install to prepare the ASR and embeddings service packages.",
            )
            self.notify("Install the server deployment instead of launching a live window", severity="warning")
            return
        if self.profile.deployment_target == "macos":
            try:
                backend.require_apple_silicon_macos()
            except SystemExit as exc:
                self._set_feedback("error", "Apple Silicon is required", str(exc))
                self.notify(str(exc), severity="error")
                return
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
        # Refresh ownership synchronously so a listener from another process is
        # never mistaken for a sidecar launched by this application.
        self.last_server_probe_at = 0.0
        self._refresh_server_states()
        if self.profile.deployment_target == "macos":
            asr_state = self._servers.state("macos_asr")
            embeddings_state = self._servers.state("macos_embeddings")
            if asr_state.status != "running":
                self._coordinator.set_pending_action(PendingAction.START_MACOS_EMBEDDINGS)
                self._start_macos_service("macos_asr")
                return
            if embeddings_state.status != "running":
                self._start_macos_service("macos_embeddings")
                return
        self._start_configured_services_and_live()

    def _start_configured_services_and_live(self) -> None:
        reports_state = self._servers.state("reports")
        if self.profile.reports_enabled and reports_state.ownership == "external":
            self._set_feedback(
                "warning",
                "Meeting Intelligence port is owned by another process",
                "Stop that process or choose another Meeting Intelligence port before launching.",
            )
            self.notify("The Meeting Intelligence port is already used by another process", severity="warning")
            return
        if self.profile.reports_enabled and reports_state.status != "running":
            self._start_reports_server(save_settings=False)
        translation_required = (
            self.profile.translation_enabled
            and self.profile.translation_provider == "sidecar"
        )
        translation_state = self._servers.state("translation")
        if translation_required and translation_state.ownership == "external":
            self._set_feedback(
                "warning",
                "Translation port is owned by another process",
                "Stop that process or choose another translation port before launching live transcription.",
            )
            self.notify(
                "The translation port is already used by another process",
                severity="warning",
            )
            return
        if translation_required and not (
            translation_state.ownership == "app" and translation_state.status == "running"
        ):
            if not self._servers.process_is_running("translation"):
                if not self._start_translation_server(save_settings=False):
                    return
        # Local model warm-up can take tens of seconds. Keep the live browser
        # usable while the sidecar finishes and report readiness independently.
        self._start_live_server()



def run_setup_app(profile: backend.Profile | None = None) -> str | None:
    """Run the Textual application and return its requested follow-up action."""

    return WhoSpeaksSetupApp(profile).run()
