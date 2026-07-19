"""Main PySide6 window for the WhoSpeaks desktop launcher."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import shlex
import sys
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QSize, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QColor, QDesktopServices, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from whospeaks_cli.cli_diagnostics import DoctorReport
from whospeaks_cli.cli_classic import build_server_launch_lines
from whospeaks_cli.launcher_controller import (
    EventKind,
    LauncherController,
    LauncherEvent,
    LauncherSnapshot,
    ProfileValidationError,
    ServiceSnapshot,
)
from whospeaks_cli.planning import InstallPlan, build_launch_plan
from whospeaks_cli.profiles import Profile, TRANSLATION_PROVIDER_OPTIONS
from window.language_config import SUPPORTED_LANGUAGE_CONFIGS

from .demo import DemoLauncherController
from .icons import line_icon
from .pages import (
    AboutPage,
    ActivityPage,
    DiagnosticsPage,
    SettingsPage,
)
from .tokens import (
    CANONICAL_RAIL_BREAKPOINT,
    COLORS,
    COMPACT_BREAKPOINT,
    COMPACT_RAIL_WIDTH,
    MEDIUM_RAIL_WIDTH,
    MINIMUM_SIZE,
    RAIL_WIDTH,
)
from .widgets import (
    ActionFooter,
    EndpointLink,
    PageHeader,
    ServiceRow,
    StatusMark,
    SummaryStrip,
    section_label,
    separator,
)


class BrandMark(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(42, 38)
        self.setAccessibleName("WhoSpeaks")

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(COLORS.accent), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(10, 15, 10, 23)
        painter.drawLine(20, 8, 20, 30)
        painter.drawLine(30, 12, 30, 26)
        painter.end()


class Sidebar(QWidget):
    navigate = Signal(int)

    ITEMS = (
        ("Overview", "home"),
        ("Diagnostics", "diagnostics"),
        ("Settings", "settings"),
        ("Activity", "activity"),
        ("About", "info"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(RAIL_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 44, 16, 60)
        layout.setSpacing(8)
        brand = QHBoxLayout()
        brand.setContentsMargins(8, 0, 8, 45)
        brand.setSpacing(10)
        self.brand_mark = BrandMark()
        self.brand_text = QLabel("WhoSpeaks")
        self.brand_text.setObjectName("logoText")
        brand.addWidget(self.brand_mark)
        brand.addWidget(self.brand_text)
        brand.addStretch(1)
        layout.addLayout(brand)
        self.buttons: list[QPushButton] = []
        for index, (label, icon_name) in enumerate(self.ITEMS):
            button = QPushButton(label)
            button.setProperty("nav", True)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setIcon(line_icon(icon_name, size=26))
            button.setIconSize(QSize(28, 28))
            button.setAccessibleName(label)
            button.setToolTip(label)
            button.clicked.connect(lambda _checked=False, value=index: self.navigate.emit(value))
            layout.addWidget(button)
            self.buttons.append(button)
        self.buttons[0].setChecked(True)
        layout.addStretch(1)
        footer = QHBoxLayout()
        footer.setContentsMargins(8, 0, 8, 0)
        self.footer_mark = StatusMark("stopped")
        footer_text = QVBoxLayout()
        footer_text.setSpacing(3)
        self.footer_primary = QLabel("System idle")
        self.footer_primary.setProperty("role", "secondary")
        self.footer_secondary = QLabel("All services stopped")
        self.footer_secondary.setProperty("role", "muted")
        self.footer_secondary.setWordWrap(True)
        footer_text.addWidget(self.footer_primary)
        footer_text.addWidget(self.footer_secondary)
        footer.addWidget(self.footer_mark)
        footer.addLayout(footer_text, 1)
        layout.addLayout(footer)
        self._compact = False

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def set_layout_mode(self, *, compact: bool, canonical: bool) -> None:
        target_width = (
            COMPACT_RAIL_WIDTH
            if compact
            else RAIL_WIDTH if canonical else MEDIUM_RAIL_WIDTH
        )
        if compact == self._compact and self.width() == target_width:
            return
        self._compact = compact
        self.setFixedWidth(target_width)
        self.brand_text.setVisible(not compact)
        for button, (label, _icon) in zip(self.buttons, self.ITEMS):
            button.setText("" if compact else label)
            button.setStyleSheet("text-align: center;" if compact else "")
        self.footer_primary.setVisible(not compact)
        self.footer_secondary.setVisible(not compact)

    def set_system_state(self, state: str, detail: str) -> None:
        self.footer_mark.set_status(state)
        label = {
            "running": "System running",
            "starting": "System starting",
            "stopping": "System stopping",
            "disconnected": "Disconnected",
            "cancelled": "Operation cancelled",
            "failed": "Needs attention",
        }.get(state, "System idle")
        self.footer_primary.setText(label)
        self.footer_secondary.setText(detail)


class ControllerBridge(QObject):
    event_received = Signal(object)
    completed = Signal(str, object)
    failed = Signal(str, str, object)

    def __init__(self, controller: LauncherController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="whospeaks-gui")
        self._active: set[str] = set()
        self._closed = False
        self._unsubscribe = controller.subscribe(self.event_received.emit)

    @property
    def active(self) -> frozenset[str]:
        return frozenset(self._active)

    def run(self, name: str, function: Callable[[], object], *, exclusive: bool = True) -> bool:
        if self._closed or name in self._active or (exclusive and self._active):
            return False
        self._active.add(name)
        future = self.executor.submit(function)

        def finished(done: concurrent.futures.Future[object]) -> None:
            self._active.discard(name)
            if self._closed:
                return
            try:
                result = done.result()
            except Exception as exc:
                self.failed.emit(name, f"{type(exc).__name__}: {exc}", exc)
            else:
                self.completed.emit(name, result)

        future.add_done_callback(finished)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        self.executor.shutdown(wait=False, cancel_futures=True)


class CommandDialog(QDialog):
    def __init__(
        self,
        title: str,
        command: str | list[str] | tuple[str, ...],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(720)
        self.setAccessibleName(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        heading = section_label(title)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        rendered = command if isinstance(command, str) else subprocess_list2cmdline(command)
        text.setPlainText(rendered)
        text.setAccessibleName("Exact command")
        text.setMinimumHeight(150)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = actions.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(rendered))
        actions.rejected.connect(self.reject)
        layout.addWidget(heading)
        layout.addWidget(text)
        layout.addWidget(actions)


def subprocess_list2cmdline(command: list[str] | tuple[str, ...]) -> str:
    if sys.platform == "win32":
        import subprocess

        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


class InstallConfirmDialog(QDialog):
    def __init__(self, plan: InstallPlan, command: list[str], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm installation")
        self.setModal(True)
        self.setMinimumWidth(680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(section_label("Confirm installation"))
        plan_details = [plan.title, plan.summary]
        if plan.translation_model_profile != "off":
            plan_details.append(
                f"Local translation runtime: {plan.translation_model_profile}"
            )
        summary = QLabel("\n".join(plan_details))
        summary.setWordWrap(True)
        summary.setProperty("role", "secondary")
        layout.addWidget(summary)
        exact = QPlainTextEdit(subprocess_list2cmdline(command))
        exact.setReadOnly(True)
        exact.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        exact.setAccessibleName("Exact installation command")
        exact.setMinimumHeight(120)
        layout.addWidget(exact)
        buttons = QDialogButtonBox()
        cancel = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        start = buttons.addButton("Start installation", QDialogButtonBox.ButtonRole.AcceptRole)
        start.setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        cancel.setDefault(True)
        cancel.setFocus()
        layout.addWidget(buttons)


class StopServicesDialog(QDialog):
    def __init__(self, running_services: list[str], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stop WhoSpeaks services?")
        self.setModal(True)
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(section_label("Stop WhoSpeaks services?"))
        body = QLabel(
            f"The {len(running_services)} active services started by this launcher will be stopped. "
            "Saved settings and meeting data are not deleted."
        )
        body.setWordWrap(True)
        body.setProperty("role", "secondary")
        layout.addWidget(body)
        list_frame = QFrame()
        list_frame.setProperty("group", True)
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(16, 12, 16, 12)
        list_layout.setSpacing(0)
        list_frame.setFixedHeight(len(running_services) * 50 + max(0, len(running_services) - 1))
        service_icons = {
            "Live window": "video",
            "Meeting Intelligence": "users",
            "Translation sidecar": "globe",
            "MLX ASR": "terminal",
            "MPS embeddings": "users",
        }
        for index, service in enumerate(running_services):
            row_widget = QWidget()
            row_widget.setFixedHeight(50)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            icon = QLabel()
            icon.setPixmap(
                line_icon(service_icons.get(service, "info"), COLORS.text_primary, 22).pixmap(22, 22)
            )
            icon.setFixedSize(32, 32)
            row.addWidget(icon)
            row.addWidget(QLabel(service))
            row.addStretch(1)
            status = QLabel("Active")
            status.setProperty("role", "success")
            row.addWidget(status)
            list_layout.addWidget(row_widget)
            if index < len(running_services) - 1:
                list_layout.addWidget(separator())
        layout.addWidget(list_frame)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        keep = QPushButton("Keep running")
        stop = QPushButton(f"Stop {len(running_services)} services")
        keep.setFixedSize(138, 47)
        stop.setFixedSize(144, 47)
        stop.setProperty("dangerFilled", True)
        keep.clicked.connect(self.reject)
        stop.clicked.connect(self.accept)
        keep.setDefault(True)
        keep.setFocus()
        buttons.addWidget(keep)
        buttons.addWidget(stop)
        layout.addLayout(buttons)


class DialogReviewOverlay(QWidget):
    """Embed an actual dialog over its actual client area for deterministic capture."""

    def __init__(self, dialog: QDialog, parent: QWidget) -> None:
        super().__init__(parent)
        self.dialog = dialog
        self.setGeometry(parent.rect())
        self.setAccessibleName("Modal dialog review overlay")
        dialog.setParent(self)
        dialog.setWindowFlags(Qt.WindowType.Widget)
        dialog.setModal(False)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(dialog, 0, 0, Qt.AlignmentFlag.AlignCenter)
        dialog.show()
        self.show()
        self.raise_()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(2, 7, 10, 184))
        painter.end()


class OverviewPage(QWidget):
    launch_requested = Signal()
    open_requested = Signal()
    refresh_requested = Signal()
    command_requested = Signal()
    activity_requested = Signal()
    install_requested = Signal()
    stop_requested = Signal()
    retry_requested = Signal(str)
    cancel_requested = Signal()
    interface_requested = Signal(str)

    def __init__(
        self,
        profile: Profile,
        *,
        preferred_installer: str = "pip",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.demo_state = ""
        self.busy_operation = ""
        self.preferred_installer = preferred_installer
        self.selected_installer = preferred_installer
        self.installer_combos: list[QComboBox] = []
        self.current_report = DoctorReport(profile.mode, [])
        self.operational_state = "ready"
        self._compact = False
        self._operational_context: dict[str, object] = {}
        self.failed_service_kind = "translation"
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.header = PageHeader("Ready to launch", "Remote controller is ready. Three services will start.")
        header_shell = QWidget()
        header_layout = QVBoxLayout(header_shell)
        self.header_layout = header_layout
        header_layout.setContentsMargins(32, 32, 32, 0)
        header_layout.addWidget(self.header)
        root.addWidget(header_shell)
        self.header_gap = QWidget()
        self.header_gap.setFixedHeight(27)
        root.addWidget(self.header_gap)
        self.summary = SummaryStrip()
        summary_shell = QWidget()
        summary_layout = QHBoxLayout(summary_shell)
        summary_layout.setContentsMargins(10, 0, 18, 0)
        summary_layout.addWidget(self.summary)
        root.addWidget(summary_shell)
        root.addSpacing(12)
        self.workspace_stack = QStackedWidget()
        self.normal_workspace = self._build_normal_workspace()
        self.first_run_workspace = self._build_first_run_workspace()
        self.workspace_stack.addWidget(self.normal_workspace)
        self.workspace_stack.addWidget(self.first_run_workspace)
        self.workspace_stack.setMinimumSize(860, 480)
        workspace_scroll = QScrollArea()
        self.workspace_scroll = workspace_scroll
        workspace_scroll.setWidgetResizable(True)
        workspace_scroll.setFrameShape(QFrame.Shape.NoFrame)
        workspace_scroll.setWidget(self.workspace_stack)
        workspace_scroll.setAccessibleName("Overview workspace")
        workspace_shell = QWidget()
        workspace_layout = QHBoxLayout(workspace_shell)
        workspace_layout.setContentsMargins(10, 0, 18, 0)
        workspace_layout.addWidget(workspace_scroll)
        root.addWidget(workspace_shell, 1)
        self.action_bar = ActionFooter(seamless=True)
        actions = self.action_bar.actions
        self.action_layout = actions
        actions.setContentsMargins(28, 20, 18, 20)
        self.primary_button = QPushButton("Launch WhoSpeaks")
        self.primary_button.setIcon(line_icon("play", COLORS.canvas))
        self.primary_button.setMinimumWidth(366)
        self.refresh_button = QPushButton("Refresh checks")
        self.refresh_button.setMinimumWidth(230)
        self.refresh_button.setIcon(line_icon("refresh"))
        self.command_button = QPushButton("View command")
        self.command_button.setMinimumWidth(240)
        self.command_button.setIcon(line_icon("terminal"))
        self.stop_button = QPushButton("Stop services")
        self.stop_button.setMinimumWidth(260)
        self.stop_button.setProperty("danger", True)
        self.stop_button.setIcon(line_icon("stop", COLORS.error))
        self.stop_button.hide()
        self.action_bar.configure_button(self.primary_button, primary=True)
        self.action_bar.configure_button(self.refresh_button)
        self.action_bar.configure_button(self.command_button)
        self.action_bar.configure_button(self.stop_button)
        self.primary_button.clicked.connect(self._primary_clicked)
        self.refresh_button.clicked.connect(self._refresh_clicked)
        self.command_button.clicked.connect(self._command_clicked)
        self.stop_button.clicked.connect(self.stop_requested)
        actions.addWidget(self.primary_button)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.command_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        root.addWidget(self.action_bar)
        self.apply_profile(profile)

    def _installer_combo(self, accessible_name: str) -> QComboBox:
        combo = QComboBox()
        uv_available = self.preferred_installer == "uv"
        if uv_available:
            combo.addItem("uv (recommended)", "uv")
        combo.addItem("pip", "pip")
        combo.setCurrentIndex(combo.findData(self.selected_installer))
        combo.setAccessibleName(accessible_name)
        combo.currentIndexChanged.connect(
            lambda _index, source=combo: self._installer_selection_changed(source)
        )
        self.installer_combos.append(combo)
        return combo

    def _installer_selection_changed(self, source: QComboBox) -> None:
        self.selected_installer = str(source.currentData() or "pip")
        for combo in self.installer_combos:
            if combo is source:
                continue
            index = combo.findData(self.selected_installer)
            if index < 0 or combo.currentIndex() == index:
                continue
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _build_normal_workspace(self) -> QWidget:
        frame = QFrame()
        frame.setProperty("group", True)
        layout = QHBoxLayout(frame)
        self.normal_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        services = QWidget()
        self.services_panel = services
        services_layout = QVBoxLayout(services)
        services_layout.setContentsMargins(20, 16, 12, 8)
        services_layout.setSpacing(0)
        services_layout.addWidget(section_label("Runtime components"))
        services_layout.addSpacing(6)
        self.service_rows = {
            "macos_asr": ServiceRow(
                "Final ASR backend",
                "Remote final transcription",
                "http://127.0.0.1:8650",
                "terminal",
            ),
            "macos_embeddings": ServiceRow(
                "Speaker embeddings backend",
                "Remote speaker identification",
                "http://127.0.0.1:8660",
                "users",
            ),
            "live": ServiceRow(
                "Live window", "Browser UI", "127.0.0.1:8796", "video", endpoint_link=True
            ),
            "reports": ServiceRow(
                "Meeting Intelligence", "Reports + Ask", "127.0.0.1:8798", "users", endpoint_link=True
            ),
            "translation": ServiceRow("Translation", "Translation sidecar", "127.0.0.1:8799", "globe"),
        }
        self.service_rows["macos_asr"].extra.setText("Required for accurate final transcription")
        self.service_rows["macos_embeddings"].extra.setText("Required for speaker identification")
        self.service_rows["live"].extra.setText("Live speaker labels  On")
        self.service_rows["reports"].extra.setText("Reports and grounded session questions")
        self.service_rows["translation"].extra.setText("Starts only when sidecar translation is enabled")
        self.service_rows["live"].endpoint.clicked.connect(
            lambda: self.interface_requested.emit("live")
        )
        self.service_rows["reports"].endpoint.clicked.connect(
            lambda: self.interface_requested.emit("reports")
        )
        self.service_separators: list[QFrame] = []
        for index, row in enumerate(self.service_rows.values()):
            services_layout.addWidget(row)
            if index < len(self.service_rows) - 1:
                row_separator = separator()
                self.service_separators.append(row_separator)
                services_layout.addWidget(row_separator)
        services_layout.addStretch(1)
        profile = QWidget()
        self.profile_panel = profile
        profile.setMinimumWidth(360)
        profile_layout = QVBoxLayout(profile)
        profile_layout.setContentsMargins(22, 22, 22, 20)
        profile_layout.setSpacing(9)
        title_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.profile_title_layout = title_row
        title_row.setSpacing(12)
        self.profile_heading = section_label("Launch profile")
        self.profile_heading.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        title_row.addWidget(self.profile_heading)
        installer_controls = QWidget()
        self.profile_installer_controls = installer_controls
        installer_controls_layout = QHBoxLayout(installer_controls)
        installer_controls_layout.setContentsMargins(0, 0, 0, 0)
        installer_controls_layout.setSpacing(12)
        installer_label = QLabel("Installer")
        installer_label.setProperty("role", "secondary")
        installer_controls_layout.addWidget(installer_label)
        self.profile_installer = self._installer_combo("Launch profile package installer")
        self.profile_installer.setMinimumWidth(190)
        self.profile_installer.setMaximumWidth(230)
        self.profile_installer.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        installer_controls_layout.addWidget(self.profile_installer, 1)
        title_row.addWidget(installer_controls)
        profile_layout.addLayout(title_row)
        self.profile_grid = QGridLayout()
        self.profile_grid.setHorizontalSpacing(14)
        self.profile_grid.setVerticalSpacing(0)
        self.profile_labels: dict[str, QLabel | EndpointLink] = {}
        self.profile_grid_rows: dict[str, tuple[int, tuple[QWidget, QWidget, QWidget]]] = {}
        self.profile_grid_widgets: list[QWidget] = []
        rows = (
            ("Profile", "mode", "users"),
            ("Language", "language", "globe"),
            ("Live text", "live_text", "terminal"),
            ("Live speaker labels", "speaker_labels", "activity"),
            ("Browser UI", "browser", "video"),
            ("Meeting Intelligence", "reports", "users"),
            ("Translation sidecar", "translation", "globe"),
        )
        for row_index, (label_text, key, icon_name) in enumerate(rows):
            icon = QLabel()
            icon.setPixmap(line_icon(icon_name, COLORS.text_primary, 22).pixmap(22, 22))
            icon.setFixedSize(28, 28)
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label = QLabel(label_text)
            label.setProperty("role", "secondary")
            value: QLabel | EndpointLink
            if key in {"browser", "reports"}:
                value = EndpointLink("")
            else:
                value = QLabel("")
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                value.setWordWrap(True)
            value.setMinimumWidth(0)
            value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.profile_labels[key] = value
            self.profile_grid.addWidget(icon, row_index, 0)
            self.profile_grid.addWidget(label, row_index, 1)
            self.profile_grid.addWidget(value, row_index, 2)
            self.profile_grid_rows[key] = (row_index, (icon, label, value))
            self.profile_grid_widgets.extend((icon, label, value))
            self.profile_grid.setRowMinimumHeight(row_index, 44)
        self.profile_labels["browser"].clicked.connect(
            lambda: self.interface_requested.emit("live")
        )
        self.profile_labels["reports"].clicked.connect(
            lambda: self.interface_requested.emit("reports")
        )
        self.profile_grid.setColumnMinimumWidth(1, 180)
        self.profile_grid.setColumnStretch(2, 1)
        profile_layout.addLayout(self.profile_grid)
        self.profile_separator = separator()
        profile_layout.addWidget(self.profile_separator)
        self.failure_hero = QLabel("!")
        self.failure_hero.setObjectName("failureHero")
        self.failure_hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.failure_hero.setFixedSize(82, 82)
        self.failure_hero.hide()
        profile_layout.addWidget(
            self.failure_hero,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        self.side_title = section_label("Warnings")
        self.side_title.setProperty("role", "warning")
        profile_layout.addWidget(self.side_title)
        profile_layout.removeWidget(self.side_title)
        profile_layout.insertWidget(3, self.side_title)
        self.failure_pre_spacer = QWidget()
        self.failure_pre_spacer.setFixedHeight(56)
        self.failure_pre_spacer.hide()
        profile_layout.insertWidget(4, self.failure_pre_spacer)
        self.failure_post_spacer = QWidget()
        self.failure_post_spacer.setFixedHeight(28)
        self.failure_post_spacer.hide()
        profile_layout.insertWidget(6, self.failure_post_spacer)
        self.side_lines: list[QLabel] = []
        for _index in range(5):
            line = QLabel("")
            line.setWordWrap(True)
            line.setProperty("role", "secondary")
            profile_layout.addWidget(line)
            self.side_lines.append(line)
        self.recovery_detail = QLabel("")
        self.recovery_detail.setWordWrap(True)
        self.recovery_detail.setProperty("role", "secondary")
        self.recovery_detail.hide()
        self.failure_headline = QLabel("")
        self.failure_headline.setProperty("role", "error")
        self.failure_headline.hide()
        profile_layout.addWidget(self.failure_headline)
        profile_layout.addWidget(self.recovery_detail)
        self.code_caption = QLabel("Last useful line")
        self.code_caption.setProperty("role", "secondary")
        self.code_caption.hide()
        profile_layout.addWidget(self.code_caption)
        self.side_code = QLabel("")
        self.side_code.setProperty("role", "code")
        self.side_code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.side_code.hide()
        profile_layout.addWidget(self.side_code)
        profile_layout.addStretch(1)
        layout.addWidget(services, 54)
        divider = QFrame()
        self.workspace_divider = divider
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setProperty("separator", True)
        divider.setFixedWidth(1)
        layout.addWidget(divider)
        layout.addWidget(profile, 46)
        return frame

    def _build_first_run_workspace(self) -> QWidget:
        """Build a compact setup form and an installation preview that remain readable."""
        frame = QFrame()
        frame.setProperty("group", True)
        layout = QHBoxLayout(frame)
        self.first_run_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        def secondary(text: str) -> QLabel:
            label = QLabel(text)
            label.setProperty("role", "secondary")
            label.setWordWrap(True)
            return label

        def field_block(title: str, control: QWidget, detail: str = "") -> QWidget:
            block = QWidget()
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(4)
            block_layout.addWidget(QLabel(title))
            block_layout.addWidget(control)
            if detail:
                block_layout.addWidget(secondary(detail))
            return block

        choices = QWidget()
        choices_layout = QVBoxLayout(choices)
        choices_layout.setContentsMargins(24, 20, 24, 18)
        choices_layout.setSpacing(16)
        choices_layout.addWidget(section_label("Choose this machine's role"))
        choices_layout.addWidget(
            secondary("Select where the controller, final transcription, and speaker recognition will run.")
        )

        self.setup_target = QComboBox()
        self.setup_target.setAccessibleName("Deployment")
        for title, value in (
            ("Full local", "local"),
            ("Remote ASR + embeddings", "core"),
            ("ASR + embeddings server", "server"),
        ):
            self.setup_target.addItem(title, value)
        deployment = field_block("Deployment", self.setup_target)
        self.setup_target_help = secondary("")
        deployment.layout().addWidget(self.setup_target_help)
        choices_layout.addWidget(deployment)

        self.setup_language = QComboBox()
        for code, config in sorted(
            SUPPORTED_LANGUAGE_CONFIGS.items(),
            key=lambda item: item[1].display_name.casefold(),
        ):
            self.setup_language.addItem(config.display_name, code)
        self.setup_language.setAccessibleName("Language")
        self.setup_language_row = field_block("Language", self.setup_language)
        self.setup_language_separator = None
        self.setup_installer = self._installer_combo("Python package installer")
        installer = field_block("Package installer", self.setup_installer)
        basics = QWidget()
        basics_layout = QHBoxLayout(basics)
        basics_layout.setContentsMargins(0, 4, 0, 2)
        basics_layout.setSpacing(16)
        basics_layout.addWidget(self.setup_language_row, 1)
        basics_layout.addWidget(installer, 1)
        choices_layout.addWidget(basics)

        optional_header = QWidget()
        optional_header_layout = QVBoxLayout(optional_header)
        optional_header_layout.setContentsMargins(0, 10, 0, 0)
        optional_header_layout.setSpacing(2)
        optional_header_layout.addWidget(section_label("Optional local features"))
        optional_header_layout.addWidget(
            secondary("Choose low-latency live text, local translation, and speaker labels for the browser window.")
        )
        self.setup_optional_header = optional_header
        choices_layout.addWidget(optional_header)

        self.setup_live_text = QComboBox()
        self.setup_live_text.setAccessibleName("Live text engine")
        for title, value in (
            ("Nemotron 3.5", "sherpa_onnx"),
            ("Kroko / Banafo", "kroko_onnx"),
            ("Off", "off"),
        ):
            self.setup_live_text.addItem(title, value)
        self.setup_preview_model = QComboBox()
        self.setup_preview_model.setAccessibleName("Live model")
        engine = field_block("Engine", self.setup_live_text)
        model = field_block("Model", self.setup_preview_model)
        self.setup_preview_model_row = model
        live_text = QWidget()
        live_text_layout = QVBoxLayout(live_text)
        live_text_layout.setContentsMargins(0, 0, 0, 0)
        live_text_layout.setSpacing(4)
        live_text_layout.addWidget(QLabel("Live text"))
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(12)
        selector_row.addWidget(engine, 1)
        selector_row.addWidget(model, 1)
        live_text_layout.addLayout(selector_row)
        self.setup_live_text_help = secondary("")
        live_text_layout.addWidget(self.setup_live_text_help)
        self.setup_live_text_row = live_text
        self.setup_live_text_separator = None
        choices_layout.addWidget(live_text)

        self.setup_translation_profile = QComboBox()
        self.setup_translation_profile.setAccessibleName("Local translation model")
        for label, value in (
            ("Off", "off"),
            ("TranslateGemma 4B", "translate-gemma-4b"),
            ("NLLB-200 600M", "nllb-200-600m"),
            ("MADLAD-400 3B", "madlad-400-3b"),
        ):
            self.setup_translation_profile.addItem(label, value)
        self.setup_translation_row = field_block(
            "Local translation",
            self.setup_translation_profile,
            "Installs an isolated local sidecar and its model files.",
        )
        self.setup_speakers = QCheckBox("Show live speaker labels")
        self.setup_speakers.setChecked(True)
        self.setup_speakers_row = field_block(
            "Speaker labels",
            self.setup_speakers,
            "Shown in the live browser window.",
        )
        self.setup_translation_separator = None
        self.setup_translation_pair = QWidget()
        optional_pair_layout = QHBoxLayout(self.setup_translation_pair)
        optional_pair_layout.setContentsMargins(0, 2, 0, 0)
        optional_pair_layout.setSpacing(16)
        optional_pair_layout.addWidget(self.setup_translation_row, 58)
        optional_pair_layout.addWidget(self.setup_speakers_row, 42)
        choices_layout.addWidget(self.setup_translation_pair)
        choices_layout.addStretch(1)

        plan = QWidget()
        plan_layout = QVBoxLayout(plan)
        plan_layout.setContentsMargins(24, 20, 24, 18)
        plan_layout.setSpacing(8)
        plan_layout.addWidget(section_label("What will be installed"))
        self.setup_plan_summary = secondary("These components will be installed on this machine.")
        plan_layout.addWidget(self.setup_plan_summary)

        def plan_row(icon_name: str, title: str, detail: str, *, missing: bool = False) -> tuple[QWidget, QLabel, QLabel, QLabel | None]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)
            mark_shell = QWidget()
            mark_shell.setFixedSize(34, 34)
            mark_layout = QHBoxLayout(mark_shell)
            mark_layout.setContentsMargins(0, 0, 0, 0)
            mark_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if missing:
                mark_layout.addWidget(StatusMark("warning"))
                mark = None
            else:
                mark = QLabel()
                mark.setPixmap(line_icon(icon_name, size=23).pixmap(23, 23))
                mark_layout.addWidget(mark)
            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(1)
            title_label = QLabel(title)
            title_label.setWordWrap(True)
            detail_label = secondary(detail)
            text_column.addWidget(title_label)
            text_column.addWidget(detail_label)
            row_layout.addWidget(mark_shell)
            row_layout.addLayout(text_column, 1)
            return row, title_label, detail_label, mark

        self.setup_plan_rows: list[tuple[QWidget, QLabel, QLabel, QLabel | None]] = []
        for _index in range(5):
            components = plan_row("server", "", "")
            self.setup_plan_rows.append(components)
            plan_layout.addWidget(components[0])
        plan_layout.addSpacing(2)
        plan_layout.addWidget(separator())
        self.setup_missing_title = section_label("Detected issues")
        self.setup_missing_title.setProperty("role", "error")
        plan_layout.addWidget(self.setup_missing_title)
        self.setup_findings_note = secondary("Based on the most recent readiness check.")
        plan_layout.addWidget(self.setup_findings_note)
        self.setup_missing_rows: list[tuple[QWidget, QLabel, QLabel, QLabel | None]] = []
        for _index in range(3):
            finding = plan_row("", "", "", missing=True)
            self.setup_missing_rows.append(finding)
            plan_layout.addWidget(finding[0])
        plan_layout.addStretch(1)

        self.setup_workspace_divider = QFrame()
        self.setup_workspace_divider.setFrameShape(QFrame.Shape.VLine)
        self.setup_workspace_divider.setProperty("separator", True)
        self.setup_workspace_divider.setFixedWidth(1)
        layout.addWidget(choices, 55)
        layout.addWidget(self.setup_workspace_divider)
        layout.addWidget(plan, 45)
        self.setup_target.currentIndexChanged.connect(self._update_setup_plan)
        self.setup_live_text.currentIndexChanged.connect(self._update_setup_plan)
        self.setup_preview_model.currentIndexChanged.connect(self._update_setup_plan)
        self.setup_translation_profile.currentIndexChanged.connect(self._update_setup_plan)
        self._update_setup_plan()
        return frame

    def setup_target_value(self) -> str:
        return str(self.setup_target.currentData() or "local")

    def setup_live_text_value(self) -> str:
        if self.setup_target_value() == "server":
            return "off"
        return str(self.setup_live_text.currentData() or "off")

    def setup_preview_model_value(self) -> str:
        if self.setup_target_value() == "server":
            return ""
        return str(self.setup_preview_model.currentData() or "")

    def setup_translation_profile_value(self) -> str:
        if self.setup_target_value() == "server":
            return "off"
        return str(self.setup_translation_profile.currentData() or "off")

    def _update_setup_plan(self, *_args: object) -> None:
        target = self.setup_target_value()
        engine = "off" if target == "server" else self.setup_live_text_value()
        self.setup_target_help.setText(
            {
                "local": "The app, final ASR, and speaker embeddings run on this machine.",
                "core": "The app runs here and connects to remote ASR and speaker-embedding services.",
                "server": "Only the final-ASR and speaker-embedding HTTP services are installed.",
            }[target]
        )
        self.setup_live_text_help.setText(
            {
                "sherpa_onnx": "Higher-quality live transcription.",
                "kroko_onnx": "Lower-resource live transcription.",
                "off": "Live transcription is disabled.",
            }[engine]
        )
        show_preview = target != "server"
        self.setup_optional_header.setVisible(show_preview)
        self.setup_language_row.setVisible(target != "server")
        if self.setup_language_separator is not None:
            self.setup_language_separator.setVisible(target != "server")
        self.setup_live_text_row.setVisible(show_preview)
        if self.setup_live_text_separator is not None:
            self.setup_live_text_separator.setVisible(show_preview)
        self.setup_translation_row.setVisible(target != "server")
        if self.setup_translation_separator is not None:
            self.setup_translation_separator.setVisible(target != "server")
        self.setup_speakers_row.setVisible(target != "server")
        self.setup_translation_pair.setVisible(target != "server")
        translation_profile = (
            self.setup_translation_profile_value() if target != "server" else "off"
        )
        choices = {
            "sherpa_onnx": (
                ("560 ms · stable", "nemotron-3.5-560ms-int8"),
                ("160 ms · lower latency", "nemotron-3.5-160ms-int8"),
            ),
            "kroko_onnx": (
                ("Community 64L", "community-64l"),
                ("Pro 16L", "pro-16l"),
            ),
            "off": (),
        }[engine]
        selected = str(self.setup_preview_model.currentData() or "")
        self.setup_preview_model.blockSignals(True)
        self.setup_preview_model.clear()
        for label, value in choices:
            self.setup_preview_model.addItem(label, value)
        index = self.setup_preview_model.findData(selected)
        self.setup_preview_model.setCurrentIndex(index if index >= 0 else 0)
        self.setup_preview_model_row.setVisible(bool(choices))
        self.setup_preview_model.blockSignals(False)

        if target == "local":
            components = [
                ("server", "Browser controller", "Web UI and local orchestration"),
                ("activity", "Local final ASR", "High-accuracy transcription on this machine"),
                ("users", "Local speaker embeddings", "Speaker identification on this machine"),
            ]
            summary = "Install the complete local controller, final ASR, and speaker-embedding stack."
        elif target == "core":
            components = [
                ("server", "Browser controller", "Web UI and local orchestration"),
                ("terminal", "Remote ASR client", "Connects to an external final-transcription service"),
                ("users", "Remote embeddings client", "Connects to an external speaker-identification service"),
            ]
            summary = "Install the controller that connects to remote ASR and embeddings services."
        else:
            components = [
                ("activity", "Final ASR server", "HTTP service dependencies for final transcription"),
                ("users", "Speaker embeddings server", "HTTP service dependencies for speaker identification"),
            ]
            summary = "Install the two service-side runtimes; no browser controller is launched in this profile."
        if engine == "sherpa_onnx":
            components.append(("terminal", "Nemotron live text", self.setup_preview_model.currentText()))
        elif engine == "kroko_onnx":
            components.append(("terminal", "Kroko / Banafo live text", self.setup_preview_model.currentText()))
        if translation_profile != "off":
            components.append(
                (
                    "globe",
                    "Local translation runtime",
                    self.setup_translation_profile.currentText(),
                )
            )
            summary += " An isolated local translation runtime and model will also be prepared."
        self.setup_plan_summary.setText(summary)
        for row, component in zip(self.setup_plan_rows, components, strict=False):
            widget, title, detail, icon = row
            icon_name, title_text, detail_text = component
            title.setText(title_text)
            detail.setText(detail_text)
            detail.setVisible(bool(detail_text))
            if icon is not None:
                icon.setPixmap(line_icon(icon_name, size=23).pixmap(23, 23))
            widget.show()
        for row in self.setup_plan_rows[len(components):]:
            row[0].hide()
        self._update_setup_findings()

    def _update_setup_findings(self) -> None:
        failures = [check for check in self.current_report.checks if check.status == "fail"]
        selected_mode = {"local": "local", "core": "remote", "server": "server"}[
            self.setup_target_value()
        ]
        selection_differs = selected_mode != self.profile.mode
        self.setup_missing_title.setText(
            f"Detected issues ({len(failures)})"
            if failures
            else "No blocking issues detected"
        )
        if selection_differs:
            saved_name = {
                "local": "Full local",
                "remote": "Remote ASR + embeddings",
                "server": "ASR + embeddings server",
            }.get(self.profile.mode, self.profile.mode)
            self.setup_findings_note.setText(
                f"These results belong to the saved {saved_name} profile. "
                "The installation uses the selection on the left."
            )
        else:
            self.setup_findings_note.setText(
                "Based on the most recent readiness check; run checks again after installation."
            )
        self.setup_missing_title.setProperty("role", "error" if failures else "success")
        self.setup_missing_title.style().unpolish(self.setup_missing_title)
        self.setup_missing_title.style().polish(self.setup_missing_title)
        visible_failures = failures[: len(self.setup_missing_rows)]
        if len(failures) > len(self.setup_missing_rows):
            visible_failures = failures[: len(self.setup_missing_rows) - 1]
        for row, check in zip(self.setup_missing_rows, visible_failures, strict=False):
            widget, title, detail, _icon = row
            title.setText(check.name)
            detail.setText(check.detail)
            detail.setVisible(bool(check.detail))
            widget.show()
        shown_count = len(visible_failures)
        if len(failures) > len(self.setup_missing_rows):
            widget, title, detail, _icon = self.setup_missing_rows[shown_count]
            remaining = len(failures) - shown_count
            title.setText(f"{remaining} more issue{'s' if remaining != 1 else ''}")
            detail.setText("Open Diagnostics for the complete readiness report.")
            detail.show()
            widget.show()
            shown_count += 1
        for row in self.setup_missing_rows[shown_count:]:
            row[0].hide()

    def setup_installer_value(self) -> str:
        return self.selected_installer

    def set_compact_layout(self, compact: bool) -> None:
        self._compact = compact
        direction = (
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self.normal_layout.setDirection(direction)
        self.first_run_layout.setDirection(direction)
        self.setup_workspace_divider.setFrameShape(
            QFrame.Shape.HLine if compact else QFrame.Shape.VLine
        )
        if compact:
            self.setup_workspace_divider.setMinimumSize(0, 1)
            self.setup_workspace_divider.setMaximumSize(16777215, 1)
        else:
            self.setup_workspace_divider.setMinimumSize(1, 0)
            self.setup_workspace_divider.setMaximumSize(1, 16777215)
        self.profile_title_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self.workspace_divider.setVisible(not compact)
        self.profile_panel.setMinimumWidth(0 if compact else 360)
        self.workspace_stack.setMinimumSize(0 if compact else 860, 900 if compact else 480)
        self.workspace_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.header_layout.setContentsMargins(
            24 if compact else 32,
            12 if compact else 32,
            24 if compact else 32,
            0,
        )
        self.header_gap.setFixedHeight(6 if compact else 27)
        self.primary_button.setMinimumWidth(280 if compact else 366)
        self.refresh_button.setMinimumWidth(140 if compact else 230)
        self.command_button.setMinimumWidth(160 if compact else 240)
        self.stop_button.setMinimumWidth(180 if compact else 260)

    def _primary_clicked(self) -> None:
        if self.workspace_stack.currentWidget() is self.first_run_workspace:
            self.install_requested.emit()
        elif self.operational_state in {"backend_unavailable", "disconnected"}:
            self.refresh_requested.emit()
        elif self.operational_state == "partial_failure" and self.failed_service_kind == "live":
            self.retry_requested.emit("live")
        elif self.operational_state in {"running", "partial_failure"}:
            self.open_requested.emit()
        else:
            self.launch_requested.emit()

    def _command_clicked(self) -> None:
        if self.busy_operation == "install":
            self.cancel_requested.emit()
        elif self.operational_state == "starting":
            self.cancel_requested.emit()
        elif self.operational_state == "partial_failure":
            self.retry_requested.emit(self.failed_service_kind)
        else:
            self.command_requested.emit()

    def _refresh_clicked(self) -> None:
        if self.busy_operation == "install":
            self.activity_requested.emit()
        elif self.operational_state in {
            "starting",
            "stopping",
            "running",
            "partial_failure",
            "failed",
            "disconnected",
        }:
            self.activity_requested.emit()
        elif self.operational_state == "backend_unavailable":
            self.refresh_requested.emit()
        else:
            self.refresh_requested.emit()

    def apply_profile(self, profile: Profile) -> None:
        self.profile = profile
        self.profile_labels["mode"].setText(
            {
                "local": "Full local",
                "remote": "Remote ASR + embeddings",
                "server": "ASR + embeddings server",
            }.get(profile.mode, profile.mode)
        )
        self.profile_labels["language"].setText(profile.language.upper())
        live_text = {
            "sherpa_onnx": "Nemotron 3.5 · " + ("160 ms" if "160ms" in profile.realtime_preview_model_preset else "560 ms"),
            "kroko_onnx": "Kroko / Banafo · "
            + ("Pro 16L" if profile.realtime_preview_model_preset == "pro-16l" else "Community 64L"),
            "off": "Off",
        }.get(profile.realtime_preview_engine, profile.realtime_preview_engine)
        self.profile_labels["live_text"].setText(live_text)
        self.profile_labels["speaker_labels"].setText("On" if profile.live_speaker_assignment else "Off")
        self.profile_labels["browser"].setText(f"{profile.host}:{profile.port}")
        self.profile_labels["reports"].setText(f"{profile.host}:{profile.reports_port}" if profile.reports_enabled else "Off")
        if not profile.translation_enabled:
            translation_label = "Off"
        elif profile.translation_provider == "sidecar":
            translation_label = f"Local sidecar · {profile.host}:{profile.translation_port}"
        elif profile.translation_provider == "transformers":
            translation_label = "Local in live process"
        else:
            translation_label = TRANSLATION_PROVIDER_OPTIONS[profile.translation_provider]["label"]
        self.profile_labels["translation"].setText(translation_label)
        for value in self.profile_labels.values():
            value.setToolTip(value.text())
        self.service_rows["live"].endpoint.setText(f"{profile.host}:{profile.port}")
        self.service_rows["reports"].endpoint.setText(f"{profile.host}:{profile.reports_port}")
        self.service_rows["translation"].endpoint.setText(f"{profile.host}:{profile.translation_port}")
        show_core_components = profile.mode in {"local", "remote", "server"}
        if profile.mode == "local":
            asr_title = "Final ASR"
            embeddings_title = "Speaker embeddings"
            self.service_rows["macos_asr"].subtitle.setText("Runs inside Live window")
            self.service_rows["macos_embeddings"].subtitle.setText("Runs inside Live window")
            self.service_rows["macos_asr"].endpoint.setText(f"{profile.model} model")
            preset = profile.provider_preset.replace("_", " ").title()
            self.service_rows["macos_embeddings"].endpoint.setText(f"{preset} preset")
            self.service_rows["macos_asr"].extra.setText("Starts and warms up with the Live window")
            self.service_rows["macos_embeddings"].extra.setText("Starts and warms up with the Live window")
        elif profile.mode == "remote":
            asr_title = "Final ASR server"
            embeddings_title = "Speaker embeddings server"
            self.service_rows["macos_asr"].subtitle.setText("Remote final transcription")
            self.service_rows["macos_embeddings"].subtitle.setText("Remote speaker identification")
            self.service_rows["macos_asr"].endpoint.setText(profile.remote_asr_url)
            self.service_rows["macos_embeddings"].endpoint.setText(profile.remote_embeddings_url)
            self.service_rows["macos_asr"].extra.setText("Required for accurate final transcription")
            self.service_rows["macos_embeddings"].extra.setText("Required for speaker identification")
        else:
            asr_title = "Final ASR server"
            embeddings_title = "Speaker embeddings server"
            self.service_rows["macos_asr"].subtitle.setText("Service package installed on this machine")
            self.service_rows["macos_embeddings"].subtitle.setText("Service package installed on this machine")
            self.service_rows["macos_asr"].endpoint.setText("Port 8650")
            self.service_rows["macos_embeddings"].endpoint.setText("Port 8660")
            self.service_rows["macos_asr"].extra.setText("Start from the generated server command")
            self.service_rows["macos_embeddings"].extra.setText("Start from the generated server command")
        self.service_rows["macos_asr"].title.setText(asr_title)
        self.service_rows["macos_embeddings"].title.setText(embeddings_title)
        self.service_rows["macos_asr"].setVisible(show_core_components)
        self.service_rows["macos_embeddings"].setVisible(show_core_components)
        show_live = profile.mode != "server"
        show_reports = show_live and profile.reports_enabled
        show_translation = (
            show_live
            and profile.translation_enabled
            and profile.translation_provider == "sidecar"
        )
        self.service_rows["live"].setVisible(show_live)
        self.service_rows["reports"].setVisible(show_reports)
        self.service_rows["translation"].setVisible(show_translation)
        visible_rows = [
            show_core_components,
            show_core_components,
            show_live,
            show_reports,
            show_translation,
        ]
        for index, row_separator in enumerate(self.service_separators):
            row_separator.setVisible(visible_rows[index] and any(visible_rows[index + 1 :]))
        self.service_rows["live"].extra.setText(f"Live speaker labels  {'On' if profile.live_speaker_assignment else 'Off'}")

        target = {"local": "local", "remote": "core", "server": "server"}.get(profile.mode, "local")
        self.setup_target.blockSignals(True)
        target_index = self.setup_target.findData(target)
        self.setup_target.setCurrentIndex(target_index if target_index >= 0 else 0)
        self.setup_target.blockSignals(False)
        language_index = self.setup_language.findData(profile.language)
        if language_index >= 0:
            self.setup_language.setCurrentIndex(language_index)
        self.setup_live_text.blockSignals(True)
        live_text_index = self.setup_live_text.findData(profile.realtime_preview_engine)
        self.setup_live_text.setCurrentIndex(live_text_index if live_text_index >= 0 else 0)
        self.setup_live_text.blockSignals(False)
        self.setup_speakers.setChecked(profile.live_speaker_assignment)
        self._update_setup_plan()
        preview_index = self.setup_preview_model.findData(profile.realtime_preview_model_preset)
        if preview_index >= 0:
            self.setup_preview_model.setCurrentIndex(preview_index)
        translation_profile = (
            profile.translation_model_profile
            if profile.translation_enabled and profile.translation_provider == "sidecar"
            else "off"
        )
        translation_index = self.setup_translation_profile.findData(translation_profile)
        self.setup_translation_profile.setCurrentIndex(
            translation_index if translation_index >= 0 else 0
        )
        self._sync_profile_grid_visibility()

    def _sync_profile_grid_visibility(self) -> None:
        server_profile = self.profile.mode == "server"
        for key, (row_index, widgets) in self.profile_grid_rows.items():
            visible = not server_profile or key == "mode"
            for widget in widgets:
                widget.setVisible(visible)
            self.profile_grid.setRowMinimumHeight(row_index, 44 if visible else 0)

    def set_report(self, report: DoctorReport) -> None:
        self.current_report = report
        self._update_setup_findings()
        counts = {name: sum(check.status == name for check in report.checks) for name in ("ok", "warn", "fail", "skip")}
        if not report.checks:
            self.summary.set_summary("Not checked", "Run a readiness check before launch", semantic="muted")
        elif counts["fail"]:
            self.summary.set_summary("Setup required", f"{counts['ok']} checks passed · {counts['fail']} need attention", semantic="error")
        elif counts["warn"]:
            self.summary.set_summary("Ready with warnings", f"{counts['ok']} checks passed · {counts['warn']} warnings", semantic="warning")
        else:
            self.summary.set_summary("Ready", f"{counts['ok']} checks passed", semantic="success")
        if self.operational_state == "ready":
            self._show_report_warnings()

    def _show_report_warnings(self) -> None:
        warnings = [check for check in self.current_report.checks if check.status == "warn"]
        self.side_title.setText("Warnings")
        self.side_title.setProperty("role", "warning")
        self.side_title.setVisible(bool(warnings))
        self.profile_separator.setVisible(bool(warnings))
        visible_warnings = warnings[: len(self.side_lines)]
        if len(warnings) > len(self.side_lines):
            visible_warnings = warnings[: len(self.side_lines) - 1]
        for line, check in zip(self.side_lines, visible_warnings, strict=False):
            line.setText(f"WARN — {check.name}: {check.detail}")
            line.show()
        shown_count = len(visible_warnings)
        if len(warnings) > len(self.side_lines):
            overflow = self.side_lines[shown_count]
            overflow.setText(f"{len(warnings) - shown_count} more warnings — open Diagnostics")
            overflow.show()
            shown_count += 1
        for line in self.side_lines[shown_count:]:
            line.hide()

    def apply_services(self, services: tuple[ServiceSnapshot, ...]) -> None:
        by_kind = {item.kind: item for item in services}
        for kind, row in self.service_rows.items():
            snapshot = by_kind.get(kind)
            if kind in {"live", "reports"}:
                available = snapshot is not None and snapshot.status == "running"
                row.set_endpoint_available(available)
                profile_link = self.profile_labels[
                    "browser" if kind == "live" else "reports"
                ]
                if isinstance(profile_link, EndpointLink):
                    profile_link.set_available(available, interface_name=row.title.text())
            if snapshot is not None:
                label = None
                if snapshot.ownership == "external" and snapshot.status == "running":
                    label = (
                        "Available"
                        if kind in {"macos_asr", "macos_embeddings"}
                        else "External"
                    )
                if snapshot.status == "starting" and snapshot.ownership == "app":
                    label = {
                        "macos_asr": "Warming up…",
                        "macos_embeddings": "Warming up…",
                        "live": "Warming up…",
                        "reports": "Starting…",
                        "translation": "Loading model…",
                    }.get(kind, "Starting…")
                if (
                    self.profile.mode == "local"
                    and kind in {"macos_asr", "macos_embeddings"}
                    and snapshot.status == "stopped"
                ):
                    label = "Starts with Live window"
                row.set_state(snapshot.status, label=label)
        if self.operational_state == "starting":
            self.profile_separator.show()
            self.side_title.setText("Launch progress")
            self.side_title.setProperty("role", "info")
            self.side_title.show()
            relevant_kinds = ["live"]
            if self.profile.mode in {"local", "remote"}:
                relevant_kinds = ["macos_asr", "macos_embeddings", *relevant_kinds]
            if self.profile.reports_enabled:
                relevant_kinds.append("reports")
            if self.profile.translation_enabled and self.profile.translation_provider == "sidecar":
                relevant_kinds.append("translation")
            labels = {
                "macos_asr": "Final ASR backend",
                "macos_embeddings": "Speaker embeddings backend",
                "live": "Live window",
                "reports": "Meeting Intelligence",
                "translation": "Translation",
            }
            for line, kind in zip(self.side_lines, relevant_kinds, strict=False):
                snapshot = by_kind.get(kind)
                status = snapshot.status if snapshot is not None else "stopped"
                if status == "running":
                    text = f"READY — {labels[kind]} ready"
                elif status == "failed":
                    text = f"FAILED — {labels[kind]} did not start"
                elif status == "unavailable":
                    text = f"OFFLINE — {labels[kind]} is unavailable"
                elif kind in {"macos_asr", "macos_embeddings"}:
                    text = f"WAIT — {labels[kind]} is starting"
                elif kind == "live":
                    text = "WARMING — Live window is preparing speech models"
                elif kind == "translation":
                    text = "WAIT — Translation model is loading"
                else:
                    text = f"WAIT — {labels[kind]} is starting"
                line.setText(text)
                line.show()
            for line in self.side_lines[len(relevant_kinds):]:
                line.hide()

    def set_demo_state(self, state: str) -> None:
        self.demo_state = state
        self.set_operational_state(state)

    def set_operational_state(
        self,
        state: str,
        *,
        error_detail: str = "",
        failed_kind: str = "translation",
        available_count: int | None = None,
        service_count: int | None = None,
        backend_available_count: int | None = None,
        progress_step: str = "",
        elapsed_seconds: int | None = None,
    ) -> None:
        self.operational_state = state
        self._operational_context = {
            "error_detail": error_detail,
            "failed_kind": failed_kind,
            "available_count": available_count,
            "service_count": service_count,
            "backend_available_count": backend_available_count,
            "progress_step": progress_step,
            "elapsed_seconds": elapsed_seconds,
        }
        self.failed_service_kind = failed_kind
        if service_count is None:
            service_count = 1 + int(self.profile.reports_enabled) + int(
                self.profile.translation_enabled
                and self.profile.translation_provider == "sidecar"
            )
            if self.profile.mode in {"local", "remote"}:
                service_count += 2
        self.workspace_stack.setCurrentWidget(self.first_run_workspace if state == "first_run" else self.normal_workspace)
        self.stop_button.hide()
        self.command_button.show()
        self.refresh_button.show()
        self.primary_button.setDisabled(False)
        self.command_button.setProperty("danger", False)
        self.side_title.hide()
        self.recovery_detail.hide()
        self.failure_headline.hide()
        self.code_caption.hide()
        self.failure_pre_spacer.hide()
        self.failure_post_spacer.hide()
        self.failure_headline.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.recovery_detail.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.recovery_detail.setContentsMargins(0, 0, 0, 0)
        self.side_code.hide()
        self.profile_heading.show()
        self.profile_separator.hide()
        self.failure_hero.hide()
        for widget in self.profile_grid_widgets:
            widget.show()
        for row_index in range(7):
            self.profile_grid.setRowMinimumHeight(row_index, 44)
        self._sync_profile_grid_visibility()
        for line in self.side_lines:
            line.hide()
        if state == "first_run":
            self.header.set_text("Finish setup", "Choose how WhoSpeaks will run, then install the missing components.")
            self.set_report(self.current_report)
            self.primary_button.setText("Install components")
            self.primary_button.setIcon(line_icon("download", COLORS.canvas))
            self.refresh_button.setText("Run checks")
            self.command_button.setText("View plan")
        elif state == "starting":
            elapsed = max(0, int(elapsed_seconds or 0))
            current_step = progress_step or "Preparing the selected services"
            self.header.set_text(
                "Starting WhoSpeaks",
                f"{current_step}. First launch can take about a minute ({elapsed} s elapsed); "
                "the live window opens only when it is ready.",
            )
            ready = 2 if available_count is None else available_count
            self.summary.set_summary(
                "Starting",
                f"{ready} of {service_count} components ready · {elapsed} s elapsed",
                semantic="info",
            )
            self.primary_button.setText("Preparing live window…")
            self.primary_button.setDisabled(True)
            self.refresh_button.setText("View activity")
            self.command_button.setText("Cancel launch")
            self.command_button.setProperty("danger", True)
        elif state == "stopping":
            self.header.set_text(
                "Stopping WhoSpeaks",
                "Owned services are shutting down safely.",
            )
            stopped = 2 if available_count is None else available_count
            self.summary.set_summary(
                "Stopping",
                f"{stopped} of {service_count} components stopped",
                semantic="info",
            )
            self.primary_button.setText("Stopping…")
            self.primary_button.setIcon(line_icon("stop", COLORS.text_muted))
            self.primary_button.setDisabled(True)
            self.refresh_button.setText("View activity")
            self.command_button.hide()
        elif state == "running":
            subtitle = (
                "Live capture and both optional services are available."
                if service_count == 3
                else "Live capture and selected optional services are available."
                if service_count > 1
                else "Live capture is available."
            )
            self.header.set_text("WhoSpeaks is running", subtitle)
            self.summary.set_summary("Running", f"{service_count} components available", semantic="success")
            self.primary_button.setText("Open live window")
            self.primary_button.setIcon(line_icon("play", COLORS.canvas))
            self.refresh_button.setText("View activity")
            self.command_button.hide()
            self.stop_button.show()
        elif state == "backend_unavailable":
            backend_label = {
                "macos_asr": "Final ASR backend",
                "macos_embeddings": "Speaker embeddings backend",
            }.get(failed_kind, "Required remote backend")
            self.header.set_text(
                "Remote backend unavailable",
                "WhoSpeaks needs both Final ASR and speaker embeddings before the Core/Remote app can launch.",
            )
            ready = 0 if backend_available_count is None else backend_available_count
            self.summary.set_summary(
                "Backend offline",
                f"{ready} of 2 required remote backends available",
                semantic="warning",
            )
            self.primary_button.setText("Retry remote services")
            self.primary_button.setIcon(line_icon("refresh", COLORS.canvas))
            self.refresh_button.hide()
            self.command_button.hide()
            self.side_title.setText("Required before launch")
            self.side_title.setProperty("role", "warning")
            self.side_title.show()
            self.profile_separator.show()
            for line in self.side_lines:
                line.hide()
            self.failure_headline.setText(f"{backend_label} is not responding")
            self.recovery_detail.setText(
                "Start the remote service or correct its URL in Settings, then refresh the checks."
            )
            backend_endpoint = (
                self.service_rows[failed_kind].endpoint.text()
                if failed_kind in {"macos_asr", "macos_embeddings"}
                else ""
            )
            self.side_code.setText(
                error_detail
                or backend_endpoint
                or "One or more required remote health endpoints did not respond."
            )
            self.failure_headline.show()
            self.recovery_detail.show()
            self.side_code.show()
        elif state == "partial_failure":
            for row_index in range(7):
                self.profile_grid.setRowMinimumHeight(row_index, 40)
            failed_label = {
                "macos_asr": "Final ASR backend",
                "macos_embeddings": "Speaker embeddings backend",
                "reports": "Meeting Intelligence",
                "translation": "Translation",
                "live": "Live window",
            }.get(failed_kind, failed_kind.replace("_", " ").title())
            if failed_kind == "live":
                subtitle = "Meeting Intelligence is available, but the Live window did not start."
            elif failed_kind in {"macos_asr", "macos_embeddings"}:
                subtitle = f"The Live window is open, but {failed_label} is unavailable."
            else:
                subtitle = f"Live capture is available, but {failed_label} did not start."
            self.header.set_text("Running with an issue", subtitle)
            ready = max(1, service_count - 1) if available_count is None else available_count
            self.summary.set_summary("Degraded", f"{ready} components available · 1 failed", semantic="warning")
            self.primary_button.setText(
                "Retry Live window" if failed_kind == "live" else "Open live window"
            )
            self.primary_button.setIcon(
                line_icon("refresh" if failed_kind == "live" else "play", COLORS.canvas)
            )
            self.refresh_button.setText("View activity")
            self.command_button.setText(f"Retry {failed_label}")
            self.command_button.setIcon(line_icon("refresh"))
            self.command_button.setVisible(failed_kind not in {"macos_asr", "macos_embeddings", "live"})
            self.stop_button.show()
            self.side_title.setText(
                "Recovery"
            )
            self.side_title.setProperty("role", "sectionTitle")
            self.side_title.show()
            self.profile_separator.show()
            for line in self.side_lines:
                line.hide()
            recovery_text = {
                "macos_asr": "The Live window remains open, but final transcription needs the remote ASR backend. Check its URL or service.",
                "macos_embeddings": "The Live window remains open, but speaker identification needs the remote embeddings backend. Check its URL or service.",
                "live": "Meeting Intelligence remains available. Retry the Live window or review its output in Activity.",
                "reports": "The live window remains usable. Retry Meeting Intelligence or review its output in Activity.",
                "translation": "The live window remains usable. Retry Translation or review its output in Activity.",
            }.get(failed_kind, "Other services remain usable. Retry the failed service or review Activity.")
            self.recovery_detail.setText(recovery_text)
            self.failure_headline.setText(f"{failed_label} stopped before becoming ready")
            self.side_code.setText(
                error_detail or f"{failed_label} exited before its service port became available."
            )
            if failed_kind in {"macos_asr", "macos_embeddings"}:
                self.failure_headline.setText(f"{failed_label} is unavailable")
                self.side_code.setText(self.service_rows[failed_kind].endpoint.text())
            elif self.demo_state and failed_kind == "translation" and not error_detail:
                self.recovery_detail.setText(
                    "The live window remains usable. Check the model path or retry Translation."
                )
                self.failure_headline.setText("Translation model failed to load")
                self.side_code.setText("Model directory does not contain config.json")
            elif error_detail:
                self.failure_headline.setText(f"{failed_label} failed to start")
            self.recovery_detail.show()
            self.failure_headline.show()
            self.code_caption.show()
            self.side_code.show()
        elif state == "failed":
            self.header.set_text(
                "WhoSpeaks could not start",
                "The local app did not become available. Review the error, then retry.",
            )
            ready = 0 if available_count is None else available_count
            failed_count = max(0, service_count - ready)
            self.summary.set_summary(
                "Launch failed",
                f"{ready} components available · {failed_count} failed",
                semantic="error",
            )
            self.primary_button.setText("Retry launch")
            self.primary_button.setIcon(line_icon("refresh", COLORS.canvas))
            self.refresh_button.setText("View activity")
            self.command_button.hide()
            self.profile_heading.hide()
            self.profile_separator.hide()
            for widget in self.profile_grid_widgets:
                widget.hide()
            for row_index in range(7):
                self.profile_grid.setRowMinimumHeight(row_index, 0)
            self.failure_pre_spacer.show()
            self.failure_post_spacer.show()
            self.failure_hero.show()
            self.side_title.setText("Recovery")
            self.side_title.setProperty("role", "sectionTitle")
            self.side_title.show()
            for line in self.side_lines:
                line.hide()
            self.recovery_detail.setText(
                "No owned service is still running. The complete error is available in Activity."
            )
            self.failure_headline.setText(
                f"Browser controller did not open {self.profile.host}:{self.profile.port}"
            )
            self.failure_headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recovery_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recovery_detail.setContentsMargins(0, 16, 0, 16)
            self.side_code.setText(
                error_detail or f"Timed out waiting for {self.profile.host}:{self.profile.port}"
            )
            self.recovery_detail.show()
            self.failure_headline.show()
            self.side_code.show()
        elif state == "disconnected":
            for row_index in range(7):
                self.profile_grid.setRowMinimumHeight(row_index, 43)
            self.header.set_text(
                "Remote service connection lost",
                "One or both required remote AI services stopped responding. Check their hosts, then retry the connection.",
            )
            self.summary.set_summary(
                "Disconnected",
                "A required remote service is unavailable",
                semantic="warning",
            )
            self.service_rows["macos_asr"].set_state("disconnected")
            self.service_rows["macos_embeddings"].set_state("unknown")
            self.primary_button.setText("Retry connection")
            self.primary_button.setIcon(line_icon("refresh", COLORS.canvas))
            self.refresh_button.setText("View activity")
            self.command_button.hide()
            self.side_title.setText("Reconnect")
            self.side_title.setProperty("role", "warning")
            self.side_title.show()
            self.profile_separator.show()
            for line in self.side_lines:
                line.hide()
            self.recovery_detail.setText(
                "WhoSpeaks will keep checking the configured ASR and speaker-embedding services. "
                "Verify their hosts or retry now."
            )
            self.side_code.setText(
                f"ASR: {self.profile.remote_asr_url.rstrip('/')}/health\n"
                f"Embeddings: {self.profile.remote_embeddings_url.rstrip('/')}/health"
            )
            self.recovery_detail.show()
            self.side_code.show()
        else:
            subtitle = (
                f"Remote ASR and speaker embeddings are available. {max(1, service_count - 2)} local services will start."
                if self.profile.mode == "remote"
                else (
                    "Local ASR and speaker embeddings will warm up with the Live window. "
                    f"{service_count} components will start."
                )
                if self.profile.mode == "local"
                else "The server packages are ready; start the two generated service commands."
            )
            self.header.set_text("Server profile ready" if self.profile.mode == "server" else "Ready to launch", subtitle)
            self.primary_button.setText("View server commands" if self.profile.mode == "server" else "Launch WhoSpeaks")
            self.primary_button.setIcon(
                line_icon("terminal" if self.profile.mode == "server" else "play", COLORS.canvas)
            )
            self.primary_button.setDisabled(False)
            self.refresh_button.setText("Refresh checks")
            self.command_button.setText("View server commands" if self.profile.mode == "server" else "View command")
            self.command_button.setVisible(self.profile.mode != "server")
            self.command_button.setProperty("danger", False)
            for row in self.service_rows.values():
                row.set_state("stopped")
            self._show_report_warnings()
        for button in (
            self.primary_button,
            self.refresh_button,
            self.command_button,
            self.stop_button,
        ):
            if button.text().strip():
                button.setAccessibleName(button.text().strip())
        self.side_title.style().unpolish(self.side_title)
        self.side_title.style().polish(self.side_title)
        self.command_button.style().unpolish(self.command_button)
        self.command_button.style().polish(self.command_button)

    def set_busy(self, busy: bool, *, operation: str = "") -> None:
        if self.demo_state:
            return
        self.busy_operation = operation if busy else ""
        self.primary_button.setDisabled(busy)
        self.refresh_button.setDisabled(busy)
        if busy and operation == "install":
            self.header.set_text(
                "Installing components",
                "Installation is running in the background; progress is available in Activity.",
            )
            self.summary.set_summary(
                "Installing",
                "Preparing the selected components",
                semantic="info",
            )
            self.primary_button.setText("Installing components...")
            self.primary_button.setAccessibleName("Installing WhoSpeaks components")
            self.refresh_button.setText("View activity")
            self.refresh_button.setDisabled(False)
            self.command_button.setText("Cancel installation")
            self.command_button.setProperty("danger", True)
            self.command_button.setDisabled(False)
        elif busy and operation == "launch":
            self.primary_button.setText("Starting…")
            self.primary_button.setAccessibleName("Starting WhoSpeaks")
        elif busy and operation == "doctor":
            self.refresh_button.setText("Checking…")
            self.refresh_button.setAccessibleName("Checking system readiness")
        elif not busy:
            self.set_operational_state(
                self.operational_state,
                **self._operational_context,
            )


class LauncherWindow(QMainWindow):
    def __init__(
        self,
        controller: LauncherController,
        *,
        auto_check: bool = True,
        reduced_motion: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.bridge = ControllerBridge(controller, self)
        self.auto_check = auto_check
        self.reduced_motion = reduced_motion
        self._closing = False
        self._last_error = ""
        self._pending_snapshot: LauncherSnapshot | None = None
        self.setWindowTitle("WhoSpeaks")
        self.setMinimumSize(*MINIMUM_SIZE)
        self.resize(1440, 900)
        self.setAccessibleName("WhoSpeaks desktop launcher")
        root = QWidget()
        root.setObjectName("appRoot")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.sidebar = Sidebar()
        self.pages = QStackedWidget()
        self.overview = OverviewPage(
            controller.profile,
            preferred_installer=controller.preferred_installer(),
        )
        self.diagnostics = DiagnosticsPage()
        self.settings = SettingsPage(controller.profile)
        self.activity = ActivityPage()
        self.about = AboutPage()
        for page in (
            self.overview,
            self.diagnostics,
            self.settings,
            self.activity,
            self.about,
        ):
            self.pages.addWidget(page)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self._ensure_accessibility_names()
        self.snapshot_timer = QTimer(self)
        self.snapshot_timer.setSingleShot(True)
        self.snapshot_timer.setInterval(100)
        self.snapshot_timer.timeout.connect(self._apply_pending_snapshot)
        self._connect_signals()
        self._install_shortcuts()
        self.apply_snapshot(controller.snapshot)
        self.activity.set_logs(list(controller.snapshot.logs))
        if isinstance(controller, DemoLauncherController):
            self._apply_demo_view(controller.demo_state)
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(92)
        self.animation_timer.timeout.connect(self._advance_activity_marks)
        if not reduced_motion:
            self.animation_timer.start()
        self.probe_timer = QTimer(self)
        self.probe_timer.setInterval(1000)
        self.probe_timer.timeout.connect(self._probe_services)
        if not isinstance(controller, DemoLauncherController):
            self.probe_timer.start()
        if auto_check and not isinstance(controller, DemoLauncherController):
            QTimer.singleShot(0, self.run_quick_check)

    def _ensure_accessibility_names(self) -> None:
        for button in self.findChildren(QAbstractButton):
            if not button.accessibleName() and button.text().strip():
                button.setAccessibleName(button.text().replace("&", "").strip())

    def _connect_signals(self) -> None:
        self.sidebar.navigate.connect(self.navigate)
        self.overview.activity_requested.connect(lambda: self.navigate(3))
        self.overview.launch_requested.connect(self.launch)
        self.overview.open_requested.connect(self.open_live_window)
        self.overview.interface_requested.connect(self.open_interface)
        self.overview.refresh_requested.connect(self.run_quick_check)
        self.overview.command_requested.connect(self.show_command)
        self.overview.install_requested.connect(self.request_install)
        self.overview.stop_requested.connect(self.confirm_stop_services)
        self.overview.retry_requested.connect(self.retry_service)
        self.overview.cancel_requested.connect(self.cancel_operation)
        self.diagnostics.quick_requested.connect(self.run_quick_check)
        self.diagnostics.deep_requested.connect(self.run_deep_check)
        self.settings.save_requested.connect(self.save_settings)
        self.activity.cancel_requested.connect(self.cancel_operation)
        self.bridge.event_received.connect(self.handle_event)
        self.bridge.completed.connect(self._worker_completed)
        self.bridge.failed.connect(self._worker_failed)

    def _install_shortcuts(self) -> None:
        actions = (
            ("Refresh", QKeySequence("Ctrl+R"), self.run_quick_check),
            ("Launch", QKeySequence("Ctrl+L"), self.launch),
            ("Settings", QKeySequence("Ctrl+,"), lambda: self.navigate(2)),
            ("Exit", QKeySequence("Ctrl+Q"), self.close),
        )
        for text, shortcut, slot in actions:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    @Slot(int)
    def navigate(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        self.sidebar.set_current(index)
        if index == 1 and self.diagnostics.model.rowCount() == 0:
            self.diagnostics.set_report(self.controller.report)

    def _apply_demo_view(self, state: str) -> None:
        page_for_state = {
            "diagnostics": 1,
            "settings": 2,
            "invalid_configuration": 2,
            "activity": 3,
            "cancelled": 3,
            "success": 3,
            "about": 4,
        }
        overview_state = "running" if state == "stop_confirmation" else "ready"
        if state not in page_for_state and state != "stop_confirmation":
            overview_state = state
        self.overview.demo_state = overview_state
        profile = self.controller.profile
        relevant_kinds = ["live"]
        if profile.mode in {"local", "remote"}:
            relevant_kinds = ["macos_asr", "macos_embeddings", *relevant_kinds]
        if profile.reports_enabled:
            relevant_kinds.append("reports")
        if profile.translation_enabled and profile.translation_provider == "sidecar":
            relevant_kinds.append("translation")
        relevant = [
            item
            for item in self.controller.snapshot.services
            if item.kind in relevant_kinds
        ]
        self.overview.set_operational_state(
            overview_state,
            failed_kind="translation",
            available_count=sum(item.status == "running" for item in relevant),
            service_count=len(relevant),
            backend_available_count=sum(
                item.status == "running"
                for item in relevant
                if item.kind in {"macos_asr", "macos_embeddings"}
            ),
            progress_step="Preparing the selected services",
            elapsed_seconds=0,
        )
        self.navigate(page_for_state.get(state, 0))
        if state in {"activity", "cancelled", "success"}:
            operation = dataclasses.replace(
                self.controller.coordinator.snapshot.operation,
                name="install" if state == "activity" else "",
                status={"activity": "running", "cancelled": "cancelled", "success": "success"}[state],
                title={
                    "activity": "Installing",
                    "cancelled": "Installation cancelled safely · 03:21",
                    "success": "Installation complete · 03:24",
                }[state],
                step="Preparing Nemotron realtime ASR" if state == "activity" else "",
            )
            self.activity.set_operation(operation)
            if state == "cancelled":
                self.activity.append_log("14:39:07  WARN  Cancellation requested by user")
                self.activity.append_log("14:39:07  INFO  Installer process exited; no child processes remain")
                self.activity.progress.show()
                self.activity.progress.setRange(0, 100)
                self.activity.progress.setValue(0)
            elif state == "success":
                self.activity.append_log("14:39:10  INFO  Installation complete")
                self.activity.append_log("14:39:10  INFO  All requested components are ready")
                self.activity.progress.show()
                self.activity.progress.setRange(0, 100)
                self.activity.progress.setValue(100)
        if state == "invalid_configuration":
            self.settings.show_validation_error(
                "port",
                "Enter a port from 1 to 65535.",
                value=70000,
            )
        sidebar_states = {
            "starting": ("starting", "Services are starting"),
            "stopping": ("stopping", "2 of 3 services stopped"),
            "running": ("running", "5 components available"),
            "partial_failure": ("failed", "1 component failed"),
            "failed": ("failed", "Local app failed · 2 remote components available"),
            "disconnected": ("disconnected", "A required remote service is unavailable"),
            "stop_confirmation": ("running", "5 components available"),
        }
        if state in sidebar_states:
            semantic, detail = sidebar_states[state]
            self.sidebar.set_system_state(semantic, detail)

    def show_stop_confirmation_review(self) -> None:
        dialog = StopServicesDialog(
            ["Live window", "Meeting Intelligence", "Translation sidecar"],
            self,
        )
        self._review_overlay = DialogReviewOverlay(dialog, self)

    @Slot(object)
    def handle_event(self, event: LauncherEvent) -> None:
        if event.kind is EventKind.LOG and event.message:
            self.activity.append_log(event.message)
        elif event.kind is EventKind.ERROR:
            self._last_error = event.message
        elif event.kind is EventKind.REPORT and isinstance(event.payload, DoctorReport):
            self.diagnostics.set_report(event.payload)
            self.overview.set_report(event.payload)
        elif event.kind is EventKind.PROFILE and isinstance(event.payload, Profile):
            self.settings.set_profile(event.payload)
            self.settings.show_saved("launch profile")
            self.overview.apply_profile(event.payload)
        elif event.kind is EventKind.OPERATION and event.payload is not None:
            self.activity.set_operation(event.payload)
            if getattr(event.payload, "name", "") == "install":
                self.overview.set_busy(True, operation="install")
                self.overview.summary.set_summary(
                    "Installing",
                    str(getattr(event.payload, "step", "") or "Preparing the selected components"),
                    semantic="info",
                )
        elif event.kind is EventKind.SNAPSHOT and isinstance(event.payload, LauncherSnapshot):
            self._queue_snapshot(event.payload)
        elif event.kind is EventKind.SERVICE:
            self._queue_snapshot(self.controller.snapshot)

    def _queue_snapshot(self, snapshot: LauncherSnapshot) -> None:
        self._pending_snapshot = snapshot
        if not self.snapshot_timer.isActive():
            self.snapshot_timer.start()

    def _apply_pending_snapshot(self) -> None:
        snapshot = self._pending_snapshot
        self._pending_snapshot = None
        if snapshot is not None:
            self.apply_snapshot(snapshot)

    def apply_snapshot(self, snapshot: LauncherSnapshot) -> None:
        relevant = [item for item in snapshot.services if item.kind == "live"]
        backends: list[ServiceSnapshot] = []
        if snapshot.profile.mode in {"local", "remote"}:
            backends = [
                item
                for item in snapshot.services
                if item.kind in {"macos_asr", "macos_embeddings"}
            ]
            relevant = (
                [*backends, *relevant]
                if snapshot.profile.mode == "remote"
                else [*relevant, *backends]
            )
        if snapshot.profile.reports_enabled:
            relevant.extend(item for item in snapshot.services if item.kind == "reports")
        if snapshot.profile.translation_enabled and snapshot.profile.translation_provider == "sidecar":
            relevant.extend(item for item in snapshot.services if item.kind == "translation")
        statuses = {item.status for item in relevant}
        failed = next(
            (item.kind for item in relevant if item.status in {"failed", "unavailable"}),
            "translation",
        )
        backend_unavailable = snapshot.profile.mode == "remote" and any(
            item.status in {"failed", "unavailable"}
            for item in backends
        )
        operation = snapshot.operation
        operation_name = str(getattr(operation, "name", ""))
        launch_running = getattr(operation, "name", "") == "launch"
        launch_failed = (
            getattr(operation, "status", "") == "error"
            and "did not start" in str(getattr(operation, "title", "")).lower()
        )
        if launch_failed and backend_unavailable:
            state = "backend_unavailable"
        elif launch_failed:
            state = "failed"
        elif launch_running or "starting" in statuses:
            state = "starting"
        elif statuses & {"failed", "unavailable"}:
            live_is_running = any(
                item.kind == "live" and item.status == "running"
                for item in relevant
            )
            if backend_unavailable and not live_is_running:
                state = "backend_unavailable"
            else:
                state = "partial_failure" if "running" in statuses else "failed"
        elif relevant and all(item.status == "running" for item in relevant):
            state = "running"
        elif any(check.status == "fail" for check in snapshot.report.checks):
            state = "first_run"
        else:
            state = "ready"
        self.overview.apply_profile(snapshot.profile)
        self.overview.set_report(snapshot.report)
        if not isinstance(self.controller, DemoLauncherController):
            started_at = getattr(operation, "started_at", None)
            elapsed_seconds = (
                max(0, int(time.monotonic() - float(started_at)))
                if started_at is not None
                else 0
            )
            self.overview.set_operational_state(
                state,
                error_detail=self._last_error,
                failed_kind=failed,
                available_count=sum(item.status == "running" for item in relevant),
                service_count=len(relevant),
                backend_available_count=sum(item.status == "running" for item in backends),
                progress_step=str(getattr(operation, "step", "")),
                elapsed_seconds=elapsed_seconds,
            )
        self.overview.apply_services(snapshot.services)
        self.diagnostics.set_report(snapshot.report)
        self.activity.set_operation(snapshot.operation)
        if operation_name == "install" or "install" in self.bridge.active:
            self.overview.set_busy(True, operation="install")
        if launch_running or "starting" in statuses:
            self.sidebar.set_system_state("starting", "Services are warming up")
        elif statuses & {"failed", "unavailable"}:
            self.sidebar.set_system_state("failed", "A required service is unavailable")
        elif relevant and all(item.status == "running" for item in relevant):
            self.sidebar.set_system_state("running", f"{len(relevant)} components available")
        elif snapshot.profile.mode == "remote" and backends and all(item.status == "running" for item in backends):
            self.sidebar.set_system_state("stopped", "App stopped · 2 remote backends available")
        else:
            self.sidebar.set_system_state("stopped", "All services stopped")

    def run_quick_check(self) -> None:
        if isinstance(self.controller, DemoLauncherController):
            self.diagnostics.set_report(self.controller.run_diagnostics())
            return
        if self.bridge.run("doctor", lambda: self.controller.run_diagnostics(deep=False)):
            self.overview.set_busy(True, operation="doctor")
            self.diagnostics.set_busy(True, deep=False)

    def run_deep_check(self) -> None:
        if isinstance(self.controller, DemoLauncherController):
            self.diagnostics.set_report(self.controller.run_diagnostics(deep=True))
            return
        if self.bridge.run("doctor", lambda: self.controller.run_diagnostics(deep=True)):
            self.overview.set_busy(True, operation="doctor")
            self.diagnostics.set_busy(True, deep=True)

    def launch(self) -> None:
        if self.controller.profile.mode == "server":
            self.show_command()
            return
        if isinstance(self.controller, DemoLauncherController):
            self.controller.launch()
            self.overview.set_demo_state("starting")
            return
        if self.bridge.run("launch", self.controller.launch):
            self.overview.set_operational_state("starting")
            self.overview.set_busy(True, operation="launch")
            self.navigate(0)

    def open_live_window(self) -> None:
        self.open_interface("live")

    @Slot(str)
    def open_interface(self, kind: str) -> None:
        interface = {
            "live": ("live window", self.controller.profile.port),
            "reports": ("Meeting Intelligence", self.controller.profile.reports_port),
        }.get(kind)
        if interface is None:
            return
        label, port = interface
        url = QUrl(f"http://{self.controller.profile.host}:{port}")
        if not QDesktopServices.openUrl(url):
            self._show_error(f"Could not open {label}", url.toString())

    def retry_service(self, kind: str) -> None:
        if isinstance(self.controller, DemoLauncherController):
            self.overview.set_demo_state("starting")
            return
        if self.bridge.run("retry", lambda: self.controller.retry_service(kind)):
            self.overview.set_operational_state("starting")
            self.navigate(0)

    def save_settings(self, values: dict[str, object]) -> None:
        if self.bridge.run("save", lambda: self.controller.update_profile(values)):
            self.settings.status.setText("Saving…")
            self.settings.save_button.setDisabled(True)

    def request_install(self) -> None:
        try:
            plan = self.controller.install_plan(
                self.overview.setup_target_value(),
                realtime_preview_engine=self.overview.setup_live_text_value(),
                realtime_preview_model_preset=self.overview.setup_preview_model_value(),
                translation_model_profile=self.overview.setup_translation_profile_value(),
            )
            command = self.controller.install_command(
                plan,
                installer=self.overview.setup_installer_value(),
            )
        except (SystemExit, ValueError) as exc:
            self._show_error("Invalid installation plan", str(exc))
            return
        dialog = InstallConfirmDialog(plan, command, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            candidate = self.controller.configure_for_install(
                plan,
                language=str(self.overview.setup_language.currentData()),
                live_speaker_assignment=self.overview.setup_speakers.isChecked(),
                persist=not isinstance(self.controller, DemoLauncherController),
            )
        except ProfileValidationError as exc:
            self.settings.show_validation_error(exc.field, str(exc))
            self.navigate(2)
            return
        except OSError as exc:
            self._show_error("Could not save installation profile", str(exc))
            return
        del candidate
        if isinstance(self.controller, DemoLauncherController):
            self.navigate(3)
            return
        if self.bridge.run("install", lambda: self.controller.install(command, title=plan.title)):
            self.overview.set_busy(True, operation="install")
            self.navigate(3)

    def cancel_operation(self) -> None:
        if isinstance(self.controller, DemoLauncherController):
            self.controller.set_demo_state("ready")
            self.overview.set_demo_state("ready")
            return
        self.bridge.run("cancel", self.controller.cancel_operation, exclusive=False)

    def show_command(self) -> None:
        if self.overview.operational_state == "first_run":
            try:
                install_plan = self.controller.install_plan(
                    self.overview.setup_target_value(),
                    realtime_preview_engine=self.overview.setup_live_text_value(),
                    realtime_preview_model_preset=self.overview.setup_preview_model_value(),
                    translation_model_profile=self.overview.setup_translation_profile_value(),
                )
                command = self.controller.install_command(
                    install_plan,
                    installer=self.overview.setup_installer_value(),
                )
            except (SystemExit, ValueError) as exc:
                self._show_error("Invalid installation plan", str(exc))
                return
            CommandDialog("Exact installation command", command, self).exec()
            return
        if self.controller.profile.mode == "server":
            rendered = "\n\n".join(build_server_launch_lines())
            CommandDialog("ASR and embeddings server commands", rendered, self).exec()
            return
        try:
            self.controller.validate_profile_updates(self.controller.profile.as_dict())
        except ProfileValidationError as exc:
            self.settings.show_validation_error(exc.field, str(exc))
            self.navigate(2)
            return
        plan = build_launch_plan(self.controller.profile)
        commands = [command for command in (plan.reports, plan.translation, plan.live) if command]
        rendered = "\n\n".join(subprocess_list2cmdline(command) for command in commands)
        CommandDialog("Exact launch command", rendered, self).exec()

    def confirm_stop_services(self) -> None:
        running = []
        labels = {
            "live": "Live window",
            "reports": "Meeting Intelligence",
            "translation": "Translation sidecar",
            "macos_asr": "MLX ASR",
            "macos_embeddings": "MPS embeddings",
        }
        for item in self.controller.snapshot.services:
            if item.status in {"running", "starting"} and item.ownership == "app":
                running.append(labels[item.kind])
        if isinstance(self.controller, DemoLauncherController) and not running:
            running = [labels[kind] for kind in ("live", "reports", "translation")]
        if not running:
            return
        if StopServicesDialog(running, self).exec() == QDialog.DialogCode.Accepted:
            self.overview.set_operational_state(
                "stopping",
                service_count=len(running),
            )
            if isinstance(self.controller, DemoLauncherController):
                self.controller.stop_owned_services()
                self.overview.set_demo_state("ready")
            else:
                self.bridge.run("stop", self.controller.stop_owned_services)

    @Slot(str, object)
    def _worker_completed(self, name: str, result: object) -> None:
        if name != "probe":
            self.overview.set_busy(False)
        if name == "doctor" and isinstance(result, DoctorReport):
            self.diagnostics.set_report(result)
        elif name == "save" and isinstance(result, Profile):
            self.settings.set_profile(result)
            self.settings.show_saved("profile")
        elif name == "install":
            self.run_quick_check()
        elif name in {"launch", "retry", "stop", "cancel"}:
            self.apply_snapshot(self.controller.snapshot)
        elif name == "shutdown":
            self.bridge.close()
            QApplication.quit()
        self.settings.save_button.setDisabled(False)

    @Slot(str, str, object)
    def _worker_failed(self, name: str, message: str, _exception: object) -> None:
        self._last_error = message
        self.overview.set_busy(False)
        self.settings.save_button.setDisabled(False)
        if isinstance(_exception, ProfileValidationError) and name in {"save", "launch", "retry"}:
            self.settings.show_validation_error(_exception.field, message)
            self.navigate(2)
            return
        if name == "save":
            self.settings.show_error(message)
        if name in {"launch", "retry", "stop", "cancel"}:
            self.apply_snapshot(self.controller.snapshot)
        self._show_error(f"{name.replace('_', ' ').title()} failed", message)

    def _show_error(self, title: str, detail: str) -> None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Critical)
        message.setWindowTitle(title)
        message.setText(title)
        message.setInformativeText(detail)
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        message.exec()

    def _probe_services(self) -> None:
        if not self.bridge.active:
            self.bridge.run("probe", self.controller.refresh_services, exclusive=False)

    def _advance_activity_marks(self) -> None:
        self.overview.summary.mark.advance()
        self.diagnostics.summary.mark.advance()
        self.sidebar.footer_mark.advance()
        for row in self.overview.service_rows.values():
            row.status_mark.advance()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        compact = self.width() < COMPACT_BREAKPOINT
        self.sidebar.set_layout_mode(
            compact=compact,
            canonical=self.width() >= CANONICAL_RAIL_BREAKPOINT,
        )
        self.overview.set_compact_layout(self.width() < 1280)
        self.settings.set_compact_layout(compact)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        owned = any(
            item.ownership == "app" and item.status in {"running", "starting", "unavailable"}
            for item in self.controller.snapshot.services
        )
        if not owned and self.controller.install_process is None:
            self.bridge.close()
            event.accept()
            return
        event.ignore()
        if owned:
            running = [
                {"live": "Live window", "reports": "Meeting Intelligence", "translation": "Translation sidecar", "macos_asr": "MLX ASR", "macos_embeddings": "MPS embeddings"}.get(item.kind, item.kind)
                for item in self.controller.snapshot.services
                if item.ownership == "app" and item.status in {"running", "starting", "unavailable"}
            ]
            if StopServicesDialog(running, self).exec() != QDialog.DialogCode.Accepted:
                return
        self._closing = True
        self.setEnabled(False)
        self.hide()
        self.bridge.run("shutdown", self.controller.shutdown, exclusive=False)
