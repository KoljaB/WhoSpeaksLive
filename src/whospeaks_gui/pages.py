"""Diagnostics, settings, activity, and about pages for the desktop launcher."""

from __future__ import annotations

import json
import os
import platform
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut, QStandardItem, QStandardItemModel, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from whospeaks_cli import __version__
from whospeaks_cli.cli_diagnostics import CheckResult, DoctorReport
from whospeaks_cli.profiles import PROVIDER_PRESETS, TRANSLATION_PROVIDER_OPTIONS, Profile
from window.language_config import SUPPORTED_LANGUAGE_CONFIGS
from window.meeting_server_support import LLM_PROVIDER_OPTIONS

from .icons import line_icon
from .settings_help import SETTINGS_MORE_HELP
from .tokens import COLORS
from .widgets import ActionFooter, PageHeader, SummaryStrip, section_label


_OPENAI_TEXT_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_OPENAI_NON_LLM_MARKERS = (
    "audio",
    "computer-use",
    "dall-e",
    "embedding",
    "gpt-image",
    "moderation",
    "realtime",
    "search",
    "transcribe",
    "tts",
    "whisper",
)
_PREVIEW_MODEL_CHOICES = {
    "sherpa_onnx": (
        ("560 ms · stable", "nemotron-3.5-560ms-int8"),
        ("160 ms · lower latency", "nemotron-3.5-160ms-int8"),
    ),
    "kroko_onnx": (
        ("Community 64L", "community-64l"),
        ("Pro 16L", "pro-16l"),
    ),
    "off": (("No live model", ""),),
}

_ASR_MODEL_CHOICES = (
    ("Whisper large-v3", "large-v3"),
    ("Whisper large-v2", "large-v2"),
    ("Whisper turbo", "turbo"),
    ("Distil-Whisper large-v3", "distil-large-v3"),
    ("Whisper medium", "medium"),
    ("Whisper small", "small"),
    ("Whisper base", "base"),
    ("Whisper tiny", "tiny"),
)


_SETTINGS_HELP: dict[str, tuple[str, str]] = {
    "settings_sections": ("Choose a settings category.", "The category list changes the visible group without saving or discarding any edits."),
    "save_changes": ("Validate and save the edited launch profile.", "WhoSpeaks checks the complete profile first. Invalid values remain visible and are marked beside the exact control that needs attention."),
    "discard_changes": ("Restore every field to the last saved profile.", "This removes all unsaved edits on every settings category."),
    "mode": (
        "Choose where transcription and speaker recognition run.",
        "Full local runs everything on this computer. Remote controller connects this app to ASR and speaker-embedding servers on another machine. Server exposes those two services for another controller.",
    ),
    "language": ("Choose the primary spoken language.", "This language is used for transcription, speaker labels, reports, and as the source language for translation."),
    "realtime_preview_engine": ("Choose the fast engine that produces live text while speech is still in progress.", "Nemotron is usually easier to install on Windows through its CPU-only sherpa-onnx backend. Kroko/Banafo uses a separate native streaming runtime. Off disables provisional live text without disabling final transcription."),
    "realtime_preview_model_preset": ("Choose the latency or model preset for the selected live-text engine.", "Lower-latency presets update sooner; larger or more stable presets may trade responsiveness for quality."),
    "live_speaker_assignment": ("Show identified speakers beside live transcript text.", "Turn this off to keep live transcription visible without assigning speaker names in the live window."),
    "host": ("Choose the network interface used by the local browser controller.", "Use 127.0.0.1 for access from this computer only. Use a LAN address only when another trusted device must connect."),
    "port": ("Choose the local browser controller port.", "The launcher checks this port before startup. Change it when another application already uses the default."),
    "model": ("Choose the final ASR model.", "Standard faster-whisper models can be selected directly; an advanced compatible Hugging Face model ID can also be entered."),
    "device": ("Choose the processor used for final transcription.", "Automatic selects an available accelerator when possible. Choose CPU or CUDA only when you need to force a specific runtime."),
    "compute_type": ("Choose the numeric precision used by faster-whisper.", "Automatic is safest. Lower-precision modes reduce memory use, while float modes may improve compatibility or accuracy."),
    "vad_backend": ("Choose how WhoSpeaks detects speech and silence.", "RMS is lightweight and predictable. Silero uses a neural voice-activity detector and may handle noisy audio better."),
    "realtime_preview_model_dir": ("Choose a local Nemotron model folder.", "Leave this empty to use automatic model discovery and download. Select a folder only for a manually managed model."),
    "realtime_preview_python": ("Choose the Python runtime for Kroko/Banafo live text.", "Leave this empty to use the managed or current runtime. Set it only when the live-text engine is installed in another environment."),
    "embedding_python": ("Choose the Python runtime for local speaker embeddings.", "Leave this empty to use the current runtime. Set it only when speaker models are installed in a separate environment."),
    "provider_preset": ("Choose a tested speaker-model combination.", "The preset updates both final and live speaker providers together. Choose Custom only when you need to edit provider expressions manually."),
    "embedding_provider": ("Review or customize the provider used for final speaker recognition.", "This value follows Speaker model preset. Select Custom before editing it directly."),
    "live_speaker_embedding_provider": ("Review or customize the provider used for live speaker labels.", "This value follows Speaker model preset. Select Custom before editing it directly."),
    "remote_asr_url": ("Enter the final ASR server address.", "The launcher verifies that this remote service is reachable before starting a remote-controller profile."),
    "remote_embeddings_url": ("Enter the speaker-embeddings server address.", "The launcher verifies that this remote service is reachable before starting a remote-controller profile."),
    "reports_enabled": ("Start Meeting Intelligence with the live window.", "This enables saved meeting reports and grounded Ask. It can be turned off without affecting transcription."),
    "reports_port": ("Choose the local Meeting Intelligence port.", "Change it when another application already uses the default port."),
    "report_language": ("Choose the language used for reports and summaries.", "Follow live language keeps reports aligned with transcription; select another language to translate generated reports."),
    "report_llm_provider": ("Choose the language-model service used for reports and Ask.", "Local providers keep requests on your network. Cloud providers require their corresponding API-key environment variable."),
    "report_llm_base_url": ("Enter the language-model API address.", "Provider defaults are filled automatically. Change this only for a custom or self-hosted compatible endpoint."),
    "report_llm_model": ("Choose the language model used for reports and Ask.", "OpenAI models are loaded from the account linked to OPENAI_API_KEY; compatible providers also accept a model ID entered manually."),
    "text_embedding_preset": ("Choose whether long and cross-session Ask uses semantic search.", "OpenAI presets send transcript chunks to the embeddings API. Not configured keeps short single-session Ask available without external embeddings."),
    "text_embedding_base_url": ("Enter the text-embedding API address.", "This endpoint is used only to index and search meeting text for long or cross-session Ask."),
    "text_embedding_model": ("Choose the text-embedding model.", "The model must be supported by the configured embedding endpoint and should remain stable for an existing index."),
    "text_embedding_api_key_env": ("Enter the name of the environment variable containing the embedding API key.", "Only the variable name is saved; WhoSpeaks never stores the secret in the launch profile."),
    "report_auto_generate": ("Generate reports when a meeting is finalized.", "When off, meetings are still saved and reports can be generated manually later."),
    "translation_enabled": ("Translate stable transcript text.", "Translation runs only after text has stabilized, so provisional live words are not repeatedly translated."),
    "translation_browser_preferred": ("Prefer Chrome's on-device Translator API when it supports the language pair.", "Supported translations remain on the device. The configured provider is retained as the fallback."),
    "translation_provider": ("Choose the translation service or local runtime.", "The fields below adapt to this choice and show only configuration used by the selected provider."),
    "translation_target_languages": ("Choose one or more translation targets.", "The live transcription language cannot also be selected as a target."),
    "translation_max_targets": ("Limit how many translations can run at once.", "A lower limit reduces latency and resource usage when many target languages are selected."),
    "translation_model_profile": ("Choose a tested local translation model.", "The profile supplies a suitable default model and runtime configuration for local translation."),
    "translation_model": ("Enter an optional provider-specific model override.", "Leave this empty to use the selected profile or provider default unless your endpoint requires a particular model ID."),
    "translation_base_url": ("Enter the translation API address.", "Provider defaults are filled automatically. Change this for a custom endpoint or self-hosted service."),
    "translation_api_key_env": ("Enter the name of the environment variable containing the translation API key.", "Only the variable name is saved; the secret remains in the process environment."),
    "translation_region": ("Enter the cloud region required by the translation provider.", "This field is shown only for providers, such as Azure Translator, that require a region."),
    "translation_python": ("Choose the Python runtime for the local translation sidecar.", "Leave this empty to use the managed or current runtime. Set it only for a separately installed environment."),
    "translation_port": ("Choose the local translation-sidecar port.", "Change it when another application already uses the default port."),
    "translation_device": ("Choose the processor used for local translation.", "Automatic selects an available accelerator. Choose CPU or CUDA only when you need to force the runtime."),
    "advanced_args": ("Append advanced command-line arguments to the validated launch profile.", "Use this only for supported options that are not represented elsewhere. These arguments are passed verbatim at launch."),
}


def _supported_language_code_help() -> str:
    entries = [
        f"{code} — {config.display_name}"
        for code, config in sorted(SUPPORTED_LANGUAGE_CONFIGS.items())
    ]
    rows = [", ".join(entries[index : index + 8]) for index in range(0, len(entries), 8)]
    return "Supported WhoSpeaks language codes:\n" + "\n".join(rows)


class SettingsHelpDialog(QDialog):
    """Bounded, scrollable F1 help that remains usable on smaller displays."""

    def __init__(
        self,
        title: str,
        summary: str,
        detail: str,
        *,
        current_value: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} help")
        self.setModal(True)
        self.setMinimumSize(580, 360)
        self.setMaximumSize(820, 560)
        self.resize(720, 500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        heading = section_label(title)
        heading.setAccessibleName(f"{title} help")
        layout.addWidget(heading)
        orientation = QLabel(summary)
        orientation.setWordWrap(True)
        orientation.setProperty("role", "secondary")
        layout.addWidget(orientation)
        if current_value:
            value = QLabel(f"Current value: {current_value}")
            value.setWordWrap(True)
            value.setProperty("role", "code")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(value)

        self.help_text = QTextBrowser()
        self.help_text.setObjectName("settingsHelpText")
        self.help_text.setOpenExternalLinks(False)
        self.help_text.setPlainText(detail)
        self.help_text.setAccessibleName(f"Detailed help for {title}")
        layout.addWidget(self.help_text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class PathPicker(QWidget):
    """Editable path field with a native file or folder chooser."""

    textEdited = Signal(str)

    def __init__(self, *, folder: bool, accessible_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.folder = folder
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setAccessibleName(accessible_name)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.setProperty("inputAction", True)
        self.browse_button.setAccessibleName(f"Browse for {accessible_name.lower()}")
        self.browse_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_button)
        self.setFocusProxy(self.edit)
        self.edit.textEdited.connect(self.textEdited)
        self.browse_button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def clear(self) -> None:
        self.edit.clear()

    def _browse(self) -> None:
        current = Path(self.text()).expanduser() if self.text().strip() else Path.home()
        start = current if current.is_dir() else current.parent
        if self.folder:
            selected = QFileDialog.getExistingDirectory(self, "Choose folder", str(start))
        else:
            selected, _filter = QFileDialog.getOpenFileName(
                self,
                "Choose Python executable",
                str(start),
                "Programs (*.exe *.bat *.cmd);;All files (*)" if platform.system() == "Windows" else "All files (*)",
            )
        if selected:
            self.setText(selected)
            self.textEdited.emit(selected)


class LanguageTargetSelector(QComboBox):
    """A checkable language dropdown that persists language codes, not free text."""

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_language = ""
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select one or more target languages")
        self.setAccessibleName("Translation target languages")
        model = QStandardItemModel(self)
        self.setModel(model)
        for code, config in sorted(
            SUPPORTED_LANGUAGE_CONFIGS.items(),
            key=lambda item: item[1].display_name.casefold(),
        ):
            item = QStandardItem(config.display_name)
            item.setData(code, Qt.ItemDataRole.UserRole)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
            model.appendRow(item)
        self.setCurrentIndex(-1)
        self.view().viewport().installEventFilter(self)
        self.view().installEventFilter(self)
        self._update_text()

    def _item(self, row: int) -> QStandardItem | None:
        model = self.model()
        return model.item(row) if isinstance(model, QStandardItemModel) else None

    def _toggle_row(self, row: int) -> None:
        item = self._item(row)
        if item is None or not item.isEnabled():
            return
        checked = item.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        item.setData(
            Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked,
            Qt.ItemDataRole.CheckStateRole,
        )
        self._update_text()
        self.selectionChanged.emit()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                self._toggle_row(index.row())
                return True
        if watched is self.view() and event.type() == QEvent.Type.KeyPress:
            if event.key() in {Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                index = self.view().currentIndex()
                if index.isValid():
                    self._toggle_row(index.row())
                    return True
        return super().eventFilter(watched, event)

    def selected_codes(self) -> list[str]:
        selected: list[str] = []
        for row in range(self.count()):
            item = self._item(row)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked:
                selected.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def set_selected_codes(self, codes: str | list[str] | tuple[str, ...]) -> None:
        selected = (
            {part for part in re.split(r"[,;\s]+", codes) if part}
            if isinstance(codes, str)
            else {str(part) for part in codes}
        )
        self.blockSignals(True)
        for row in range(self.count()):
            item = self._item(row)
            if item is None:
                continue
            code = str(item.data(Qt.ItemDataRole.UserRole))
            item.setData(
                Qt.CheckState.Checked
                if code in selected and code != self._source_language
                else Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        self._update_text()
        self.blockSignals(False)

    def set_source_language(self, code: str) -> None:
        self._source_language = str(code or "")
        for row in range(self.count()):
            item = self._item(row)
            if item is None:
                continue
            item_code = str(item.data(Qt.ItemDataRole.UserRole))
            is_source = item_code == self._source_language
            item.setEnabled(not is_source)
            if is_source:
                item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
                item.setToolTip("The target language must differ from the live language.")
            else:
                item.setToolTip("")
        self._update_text()

    def _update_text(self) -> None:
        labels: list[str] = []
        for row in range(self.count()):
            item = self._item(row)
            if item is not None and item.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked:
                labels.append(item.text())
        self.lineEdit().setText(", ".join(labels))


def suitable_openai_llm_models(model_ids: list[str]) -> list[str]:
    """Return account-visible model IDs that are suitable for text generation."""

    suitable: set[str] = set()
    for raw_id in model_ids:
        model_id = str(raw_id or "").strip()
        normalized = model_id.casefold()
        candidate = normalized
        if normalized.startswith("ft:"):
            parts = normalized.split(":", 2)
            candidate = parts[1] if len(parts) > 1 else ""
        if not candidate.startswith(_OPENAI_TEXT_MODEL_PREFIXES):
            continue
        if any(marker in candidate for marker in _OPENAI_NON_LLM_MARKERS):
            continue
        suitable.add(model_id)

    def sort_key(model_id: str) -> tuple[bool, str]:
        is_snapshot = bool(
            re.search(r"(?:-\d{4}-\d{2}-\d{2}|-\d{8})$", model_id)
        )
        return is_snapshot, model_id.casefold()

    return sorted(suitable, key=sort_key)


def _fetch_openai_model_ids(api_key: str) -> tuple[list[str], str]:
    """Fetch account-visible OpenAI model IDs without using Qt's network stack."""

    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": f"WhoSpeaks/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return [], f"OpenAI returned HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return [], f"Could not reach OpenAI · {exc.reason}"
    except TimeoutError:
        return [], "OpenAI model loading timed out"
    except OSError as exc:
        return [], f"Could not load OpenAI models · {exc}"
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        return [], f"Could not read OpenAI model catalog · {exc}"

    records = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(record.get("id") or "")
        for record in records
        if isinstance(record, dict)
    ]
    return suitable_openai_llm_models(model_ids), ""


class _OpenAIModelCatalogBridge(QObject):
    completed = Signal(int, object, str)


def _page_root(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout, PageHeader]:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(10, 32, 18, 24)
    layout.setSpacing(28)
    header = PageHeader(title, subtitle)
    header.layout().setContentsMargins(22, 0, 14, 0)
    layout.addWidget(header)
    return page, layout, header


class DiagnosticsPage(QWidget):
    quick_requested = Signal()
    deep_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 32, 18, 0)
        layout.setSpacing(0)
        header = PageHeader("Diagnostics", "Check the launcher, runtimes, models, ports, and configured services.")
        header.layout().setContentsMargins(22, 0, 14, 0)
        layout.addWidget(header)
        layout.addSpacing(28)
        self.summary = SummaryStrip()
        layout.addWidget(self.summary)
        layout.addSpacing(12)
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.table.setAccessibleName("Diagnostic results")
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(64)
        self.model = QStandardItemModel(0, 4, self)
        self.model.setHorizontalHeaderLabels(("State", "Component", "Detail", "Recommended action"))
        self.table.setModel(self.model)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setFixedHeight(62)
        header.setMinimumSectionSize(90)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setIconSize(QSize(28, 28))
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)
        results_layout.addWidget(self.table, 1)
        self.action_bar = ActionFooter()
        actions = self.action_bar.actions
        self.quick_button = QPushButton("Quick check")
        self.quick_button.setIcon(line_icon("refresh"))
        self.complete_button = QPushButton("Complete check")
        self.complete_button.setIcon(line_icon("play", COLORS.canvas))
        self.copy_button = QPushButton("Copy results")
        self.copy_button.setIcon(line_icon("copy"))
        self.action_bar.configure_button(self.complete_button, primary=True)
        self.action_bar.configure_button(self.quick_button)
        self.action_bar.configure_button(self.copy_button)
        self.quick_button.clicked.connect(self.quick_requested)
        self.complete_button.clicked.connect(self.deep_requested)
        self.copy_button.clicked.connect(self.copy_results)
        self.complete_button.setMinimumWidth(270)
        self.quick_button.setMinimumWidth(230)
        self.copy_button.setMinimumWidth(240)
        actions.addWidget(self.complete_button)
        actions.addWidget(self.quick_button)
        actions.addWidget(self.copy_button)
        actions.addStretch(1)
        results_layout.addWidget(self.action_bar)
        layout.addWidget(results_panel, 1)
        self._checks: list[CheckResult] = []

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_whole_diagnostic_rows)

    def _fit_whole_diagnostic_rows(self) -> None:
        viewport_height = self.table.viewport().height()
        if viewport_height <= 0:
            return
        visible_rows = max(1, round(viewport_height / 60))
        row_height = max(48, min(68, viewport_height // visible_rows))
        self.table.verticalHeader().setDefaultSectionSize(row_height)

    def set_busy(self, busy: bool, *, deep: bool = False) -> None:
        self.quick_button.setDisabled(busy)
        self.complete_button.setDisabled(busy)
        if busy:
            self.summary.set_summary(
                "Checking",
                "Running complete diagnostics" if deep else "Inspecting installed components",
                semantic="info",
            )

    def set_report(self, report: DoctorReport) -> None:
        incoming_checks = list(report.checks)
        previous_scroll = self.table.verticalScrollBar().value()
        checks_changed = incoming_checks != self._checks
        self._checks = incoming_checks
        status_colors = {
            "ok": COLORS.success,
            "warn": COLORS.warning,
            "fail": COLORS.error,
            "skip": COLORS.text_muted,
        }
        labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
        status_icons = {
            "ok": "check_circle",
            "warn": "warning",
            "fail": "close",
            "skip": "info",
        }
        if checks_changed:
            self.model.removeRows(0, self.model.rowCount())
            for check in self._checks:
                state = QStandardItem("")
                state.setForeground(QColor(status_colors.get(check.status, COLORS.text_primary)))
                state.setIcon(
                    line_icon(
                        status_icons.get(check.status, "info"),
                        status_colors.get(check.status, COLORS.text_primary),
                        28,
                    )
                )
                state.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                state_font = state.font()
                state_font.setWeight(QFont.Weight.DemiBold)
                state.setFont(state_font)
                component = QStandardItem(check.name)
                detail = QStandardItem(check.detail)
                recovery = QStandardItem(check.remediation or "—")
                for item in (state, component, detail, recovery):
                    item.setEditable(False)
                    item.setToolTip(item.text())
                self.model.appendRow((state, component, detail, recovery))
        counts = {name: sum(check.status == name for check in self._checks) for name in ("ok", "warn", "fail", "skip")}
        if counts["fail"]:
            state, semantic = "Action required", "error"
        elif counts["warn"]:
            state, semantic = "Usable with warnings", "warning"
        else:
            state, semantic = "Ready", "success"
        detail = f"{counts['ok']} passed · {counts['warn']} warnings · {counts['fail']} failed"
        self.summary.set_summary(state, detail, semantic=semantic)
        self.set_busy(False)
        if checks_changed and previous_scroll:
            self.table.verticalScrollBar().setValue(previous_scroll)

    def copy_results(self) -> None:
        lines = ["State\tComponent\tDetail\tRecovery"]
        for check in self._checks:
            lines.append(
                f"{check.status.upper()}\t{check.name}\t{check.detail}\t{check.remediation}"
            )
        QApplication.clipboard().setText("\n".join(lines))


class FormSection(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("formContent")
        self.form = QFormLayout(content)
        self.form.setContentsMargins(20, 24, 20, 32)
        self.form.setHorizontalSpacing(20)
        self.form.setVerticalSpacing(18)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.field_rows: dict[QWidget, tuple[QLabel, QWidget]] = {}
        self.setWidget(content)

    def add_field(self, label: str, widget: QWidget, help_text: str = "") -> None:
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(190)
        widget.setProperty("settingsHelpTitle", label)
        widget.setProperty("settingsHelpSummary", help_text)
        if help_text:
            container = QWidget()
            column = QVBoxLayout(container)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(4)
            column.addWidget(widget)
            help_label = QLabel(help_text)
            help_label.setProperty("role", "muted")
            help_label.setWordWrap(True)
            column.addWidget(help_label)
            self.form.addRow(label_widget, container)
            self.field_rows[widget] = (label_widget, container)
        else:
            self.form.addRow(label_widget, widget)
            self.field_rows[widget] = (label_widget, widget)

    def set_field_visible(self, widget: QWidget, visible: bool) -> None:
        row = self.field_rows.get(widget)
        if row is None:
            return
        row[0].setVisible(visible)
        row[1].setVisible(visible)


class GeneralSettingsSection(QScrollArea):
    """Two-column general settings layout used by the canonical desktop view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("formContent")
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(16, 24, 18, 24)
        self.grid.setHorizontalSpacing(20)
        self.grid.setVerticalSpacing(18)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.field_blocks: dict[QWidget, QVBoxLayout] = {}
        self.errors: list[QLabel] = []
        self.setWidget(content)

    def add_field(
        self,
        row: int,
        column: int,
        label: str,
        widget: QWidget,
        help_text: str = "",
    ) -> None:
        block = QWidget()
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(6)
        block_layout.addWidget(QLabel(label))
        widget.setProperty("settingsHelpTitle", label)
        widget.setProperty("settingsHelpSummary", help_text)
        block_layout.addWidget(widget)
        if help_text:
            help_label = QLabel(help_text)
            help_label.setProperty("role", "muted")
            help_label.setWordWrap(True)
            block_layout.addWidget(help_label)
        self.field_blocks[widget] = block_layout
        self.grid.addWidget(block, row, column)

    def show_error(self, widget: QWidget, field: str, message: str) -> QLabel | None:
        layout = self.field_blocks.get(widget)
        if layout is None:
            return None
        error = QLabel(message)
        error.setProperty("role", "error")
        error.setAccessibleName(f"{field.replace('_', ' ')} error")
        layout.insertWidget(2, error)
        self.errors.append(error)
        return error

    def clear_errors(self) -> None:
        for error in self.errors:
            error.setParent(None)
            error.deleteLater()
        self.errors.clear()

    def set_field_visible(self, widget: QWidget, visible: bool) -> None:
        block = self.field_blocks.get(widget)
        if block is not None and block.parentWidget() is not None:
            block.parentWidget().setVisible(visible)


class SettingsPage(QWidget):
    save_requested = Signal(dict)

    SECTIONS = (
        "General",
        "Transcription",
        "Speaker recognition",
        "Connections",
        "Meeting Intelligence",
        "Translation",
        "Advanced",
    )

    def __init__(self, profile: Profile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.fields: dict[str, QWidget] = {}
        self._field_sections: dict[str, int] = {}
        self._validation_rows: list[tuple[QFormLayout, QLabel]] = []
        self._openai_models: list[str] = []
        self._openai_request_serial = 0
        self._openai_request_running = False
        self._openai_catalog_threads: dict[int, threading.Thread] = {}
        self._pending_openai_models: list[str] | None = None
        self._openai_catalog_status = ""
        self._openai_catalog_role = "secondary"
        self._help_targets: dict[QWidget, tuple[str, QWidget]] = {}
        self._active_help_key = "mode"
        self._active_help_widget: QWidget | None = None
        self._help_dialog: SettingsHelpDialog | None = None
        self._openai_catalog_bridge = _OpenAIModelCatalogBridge(self)
        self._openai_catalog_bridge.completed.connect(self._openai_models_finished)
        self._openai_apply_timer = QTimer(self)
        self._openai_apply_timer.setSingleShot(True)
        self._openai_apply_timer.setInterval(100)
        self._openai_apply_timer.timeout.connect(self._apply_pending_openai_models)
        root = QVBoxLayout(self)
        self.root_layout = root
        root.setContentsMargins(10, 32, 18, 0)
        root.setSpacing(0)
        header = PageHeader("Settings", "Edit the saved launch profile. Changes are validated before they are written.")
        header.layout().setContentsMargins(22, 0, 14, 0)
        root.addWidget(header)
        self.header_gap = QWidget()
        self.header_gap.setFixedHeight(28)
        root.addWidget(self.header_gap)
        body = QFrame()
        body.setProperty("group", True)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        form_region = QWidget()
        form_region_layout = QHBoxLayout(form_region)
        form_region_layout.setContentsMargins(0, 0, 0, 0)
        form_region_layout.setSpacing(0)
        self.section_list = QListWidget()
        self.section_list.setObjectName("settingsSections")
        self.section_list.setFixedWidth(220)
        self.section_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.section_list.addItems(self.SECTIONS)
        self.section_list.setAccessibleName("Settings sections")
        self.sections = QStackedWidget()
        form_region_layout.addWidget(self.section_list)
        form_region_layout.addWidget(self.sections, 1)
        body_layout.addWidget(form_region, 1)
        self.launch_effect = QFrame()
        self.launch_effect.setProperty("group", True)
        self.launch_effect.setMinimumHeight(94)
        effect_layout = QHBoxLayout(self.launch_effect)
        self.launch_effect_layout = effect_layout
        effect_layout.setContentsMargins(24, 16, 24, 16)
        effect_layout.setSpacing(16)
        effect_icon = QLabel()
        effect_icon.setPixmap(line_icon("info", COLORS.info, 26).pixmap(26, 26))
        effect_text = QVBoxLayout()
        effect_text.setSpacing(3)
        self.context_help_title = QLabel("Deployment")
        help_title_font = self.context_help_title.font()
        help_title_font.setWeight(QFont.Weight.DemiBold)
        self.context_help_title.setFont(help_title_font)
        self.context_help_value = QLabel("Remote ASR + embeddings")
        self.context_help_value.setProperty("role", "secondary")
        self.context_help_detail = QLabel("Help for the focused setting appears here.")
        self.launch_effect_detail = self.context_help_detail
        self.context_help_detail.setWordWrap(True)
        self.context_help_detail.setProperty("role", "secondary")
        self.context_help_hint = QLabel("F1  More help")
        self.context_help_hint.setProperty("role", "muted")
        self.context_help_hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.context_help_hint.setAccessibleName("Press F1 for more help about the focused setting")
        effect_text.addWidget(self.context_help_title)
        effect_text.addWidget(self.context_help_value)
        effect_text.addWidget(self.context_help_detail)
        effect_layout.addWidget(effect_icon)
        effect_layout.addLayout(effect_text, 1)
        effect_layout.addWidget(self.context_help_hint)
        effect_shell = QWidget()
        effect_shell_layout = QHBoxLayout(effect_shell)
        self.launch_effect_shell_layout = effect_shell_layout
        effect_shell_layout.setContentsMargins(22, 0, 22, 16)
        effect_shell_layout.addWidget(self.launch_effect)
        body_layout.addWidget(effect_shell)
        root.addWidget(body, 1)
        self.action_bar = ActionFooter()
        actions = self.action_bar.actions
        self.action_layout = actions
        self.save_button = QPushButton("Save changes")
        self.save_button.setMinimumWidth(256)
        self.discard_button = QPushButton("Discard")
        self.action_bar.configure_button(self.save_button, primary=True)
        self.action_bar.configure_button(self.discard_button)
        self.status = QLabel("No unsaved changes")
        self.status.setProperty("role", "success")
        actions.addWidget(self.save_button)
        actions.addWidget(self.discard_button)
        actions.addSpacing(12)
        actions.addWidget(self.status)
        actions.addStretch(1)
        root.addWidget(self.action_bar)
        self._build_sections()
        for field, widget in self.fields.items():
            for index in range(self.sections.count()):
                section = self.sections.widget(index)
                if isinstance(section, GeneralSettingsSection) and widget in section.field_blocks:
                    self._field_sections[field] = index
                    break
                if isinstance(section, FormSection) and widget in section.field_rows:
                    self._field_sections[field] = index
                    break
        self.section_list.currentRowChanged.connect(self.sections.setCurrentIndex)
        self.section_list.setCurrentRow(0)
        self.save_button.clicked.connect(self._emit_save)
        self.discard_button.clicked.connect(lambda: self.set_profile(self.profile))
        self._connect_dirty_signals()
        self._install_control_help()
        self.set_profile(profile)

    def set_compact_layout(self, compact: bool) -> None:
        """Keep complete settings rows visible on smaller supported windows."""
        self.root_layout.setContentsMargins(10, 16 if compact else 32, 18, 0)
        self.header_gap.setFixedHeight(14 if compact else 28)
        self.launch_effect.setMinimumHeight(92 if compact else 110)
        self.launch_effect_layout.setContentsMargins(
            18 if compact else 24,
            10 if compact else 16,
            18 if compact else 24,
            10 if compact else 16,
        )
        self.launch_effect_shell_layout.setContentsMargins(
            22,
            0,
            22,
            8 if compact else 16,
        )

    def _install_control_help(self) -> None:
        for field, widget in self.fields.items():
            self._register_help_target(field, widget)
        self._register_help_target("text_embedding_preset", self.text_embedding_preset)
        self._register_help_target("settings_sections", self.section_list, title="Settings categories")
        self._register_help_target("save_changes", self.save_button, title="Save changes")
        self._register_help_target("discard_changes", self.discard_button, title="Discard changes")
        self.section_list.currentRowChanged.connect(self._refresh_context_help)
        self._help_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self._help_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._help_shortcut.activated.connect(self._show_more_help)

    def _register_help_target(
        self,
        key: str,
        widget: QWidget,
        *,
        title: str = "",
    ) -> None:
        summary, detail = _SETTINGS_HELP[key]
        resolved_title = title or str(widget.property("settingsHelpTitle") or widget.accessibleName() or key.replace("_", " ").title())
        widget.setProperty("settingsHelpTitle", resolved_title)
        widget.setToolTip(summary)
        widget.setAccessibleDescription(detail)
        targets = [widget, *widget.findChildren(QWidget)]
        for target in targets:
            target.setToolTip(summary)
            target.installEventFilter(self)
            self._help_targets[target] = (key, widget)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if isinstance(watched, QWidget) and watched in self._help_targets:
            if event.type() in {QEvent.Type.FocusIn, QEvent.Type.Enter}:
                key, root_widget = self._help_targets[watched]
                self._show_control_help(key, root_widget)
        return super().eventFilter(watched, event)

    def _control_value_text(self, key: str, widget: QWidget) -> str:
        if key == "settings_sections":
            item = self.section_list.currentItem()
            return item.text() if item is not None else ""
        if key in {"save_changes", "discard_changes"}:
            return ""
        if key == "text_embedding_preset":
            return self.text_embedding_preset.currentText()
        if isinstance(widget, QComboBox):
            return widget.currentText().strip() or "Not selected"
        if isinstance(widget, QCheckBox):
            return "On" if widget.isChecked() else "Off"
        if isinstance(widget, QSpinBox):
            return widget.text()
        if isinstance(widget, QLineEdit):
            return widget.text().strip() or "Not set"
        if isinstance(widget, PathPicker):
            return widget.text().strip() or "Automatic"
        return ""

    def _control_help_detail(self, key: str, widget: QWidget) -> str:
        validation_message = str(widget.property("validationMessage") or "").strip()
        if validation_message:
            return f"Needs attention: {validation_message}"
        if key == "mode":
            mode = str(self._value("mode"))
            return {
                "local": "Runs final ASR and speaker embeddings inside the local live-window process on this computer.",
                "remote": "Connects this app to ASR and speaker-embedding servers on another machine. Configure their addresses under Connections.",
                "server": "Starts the final ASR and speaker-embedding services for another WhoSpeaks controller. The browser app is not started by this profile.",
                "macos": "Runs ASR and speaker embeddings as launcher-managed Apple Silicon services on this Mac.",
            }.get(mode, _SETTINGS_HELP[key][1])
        return _SETTINGS_HELP[key][1]

    def _show_control_help(self, key: str, widget: QWidget) -> None:
        self._active_help_key = key
        self._active_help_widget = widget
        title = str(widget.property("settingsHelpTitle") or widget.accessibleName() or key.replace("_", " ").title())
        self.context_help_title.setText(title)
        value = self._control_value_text(key, widget)
        self.context_help_value.setText(value)
        self.context_help_value.setVisible(bool(value))
        self.context_help_detail.setText(self._control_help_detail(key, widget))

    def _refresh_context_help(self, *_args: object) -> None:
        if self._active_help_widget is not None:
            self._show_control_help(self._active_help_key, self._active_help_widget)

    def _expanded_control_help(self, key: str, widget: QWidget) -> str:
        detail = SETTINGS_MORE_HELP[key]
        if key == "translation_target_languages":
            detail = detail.format(language_codes=_supported_language_code_help())
        if key in {
            "provider_preset",
            "embedding_provider",
            "live_speaker_embedding_provider",
        }:
            preset_id = str(self._value("provider_preset"))
            preset = PROVIDER_PRESETS.get(preset_id)
            if preset is not None:
                selected = (
                    f"Selected preset: {preset.name} ({preset.id})\n"
                    f"{preset.summary} {preset.details}\n"
                    f"Operational note: {preset.score_note or 'Validate this stack on representative audio.'}\n"
                    f"Final: {preset.embedding_provider}\n"
                    f"Live: {preset.live_speaker_embedding_provider}"
                )
                detail = f"{detail}\n\n{selected}"
        if key in {"report_llm_provider", "report_llm_base_url", "report_llm_model"}:
            provider_id = str(self._value("report_llm_provider"))
            option = LLM_PROVIDER_OPTIONS.get(provider_id)
            if option is not None:
                credential = str(option.get("api_key_env_var") or "No API key required by the default local configuration")
                selected = (
                    f"Selected provider: {option['label']} ({provider_id})\n"
                    f"Default API root: {option['default_base_url']}\n"
                    f"Credential: {credential}"
                )
                detail = f"{detail}\n\n{selected}"
        if key in {
            "translation_provider",
            "translation_base_url",
            "translation_api_key_env",
        }:
            provider_id = str(self._value("translation_provider"))
            option = TRANSLATION_PROVIDER_OPTIONS.get(provider_id)
            if option is not None:
                endpoint = str(option.get("default_base_url") or "No fixed endpoint")
                credential = str(option.get("default_api_key_env") or "No default key variable")
                selected = (
                    f"Selected provider: {option['label']} ({provider_id})\n"
                    f"Default endpoint: {endpoint}\n"
                    f"Default key variable: {credential}"
                )
                detail = f"{detail}\n\n{selected}"
        return detail

    def _show_more_help(self) -> None:
        widget = self._active_help_widget
        if widget is None:
            widget = self.fields["mode"]
            self._show_control_help("mode", widget)
        title = self.context_help_title.text()
        summary = _SETTINGS_HELP[self._active_help_key][0]
        detail = self._expanded_control_help(self._active_help_key, widget)
        dialog = SettingsHelpDialog(
            title,
            summary,
            detail,
            current_value=self._control_value_text(self._active_help_key, widget),
            parent=self,
        )
        self._help_dialog = dialog
        dialog.exec()

    def _combo(self, field: str, choices: list[tuple[str, object]]) -> QComboBox:
        widget = QComboBox()
        for label, value in choices:
            widget.addItem(label, value)
        widget.setAccessibleName(field.replace("_", " ").title())
        self.fields[field] = widget
        return widget

    def _line(self, field: str, *, password: bool = False) -> QLineEdit:
        widget = QLineEdit()
        if password:
            widget.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        widget.setAccessibleName(field.replace("_", " ").title())
        self.fields[field] = widget
        return widget

    def _model_combo(
        self,
        field: str,
        choices: tuple[tuple[str, object], ...] = (),
    ) -> QComboBox:
        widget = QComboBox()
        widget.setEditable(True)
        widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for label, value in choices:
            widget.addItem(label, value)
        widget.setAccessibleName(field.replace("_", " ").title())
        self.fields[field] = widget
        return widget

    def _spin(self, field: str, minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setAccessibleName(field.replace("_", " ").title())
        self.fields[field] = widget
        return widget

    def _check(self, field: str, label: str) -> QCheckBox:
        widget = QCheckBox(label)
        widget.setAccessibleName(label)
        self.fields[field] = widget
        return widget

    def _path(self, field: str, *, folder: bool) -> PathPicker:
        widget = PathPicker(
            folder=folder,
            accessible_name=field.replace("_", " ").title(),
        )
        self.fields[field] = widget
        return widget

    def _build_sections(self) -> None:
        deployment_choices = [
            ("Full local", "local"),
            ("Remote ASR + embeddings", "remote"),
            ("ASR + embeddings server", "server"),
        ]
        apple_silicon = (
            platform.system() == "Darwin"
            and platform.machine().lower() in {"arm64", "aarch64"}
        )
        if apple_silicon or self.profile.deployment_target == "macos":
            deployment_choices.append(
                (
                    "Apple Silicon managed services"
                    if apple_silicon
                    else "Apple Silicon managed services (unavailable here)",
                    "macos",
                )
            )
        general = GeneralSettingsSection()
        general.add_field(
            0,
            0,
            "Deployment",
            self._combo(
                "mode",
                deployment_choices,
            ),
            "Select where final ASR and speaker embeddings run.",
        )
        general.add_field(
            0,
            1,
            "Language",
            self._combo("language", sorted(((cfg.display_name, code) for code, cfg in SUPPORTED_LANGUAGE_CONFIGS.items()), key=lambda item: item[0].casefold())),
            "Language for transcription, speakers, and summaries.",
        )
        general.add_field(
            1,
            0,
            "Live text",
            self._combo("realtime_preview_engine", [("Nemotron 3.5", "sherpa_onnx"), ("Kroko / Banafo", "kroko_onnx"), ("Off", "off")]),
        )
        general.add_field(
            1,
            1,
            "Live model",
            self._combo(
                "realtime_preview_model_preset",
                list(_PREVIEW_MODEL_CHOICES["sherpa_onnx"]),
            ),
            "Choices follow the selected live-text engine.",
        )
        general.add_field(
            2,
            0,
            "Live speaker labels",
            self._check("live_speaker_assignment", "Show speaker labels in the live window"),
            "Uncheck to hide speaker labels in the live window.",
        )
        general.add_field(2, 1, "Browser host", self._line("host"), "Host where the local browser controller is available.")
        general.add_field(3, 0, "Browser port", self._spin("port", 1, 65535), "Port for the local browser controller.")
        self.sections.addWidget(general)

        transcription = FormSection()
        transcription.add_field(
            "ASR model",
            self._model_combo("model", _ASR_MODEL_CHOICES),
            "Choose a standard faster-whisper model or enter a compatible Hugging Face model ID.",
        )
        transcription.add_field("Device", self._combo("device", [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")]))
        transcription.add_field(
            "Compute type",
            self._model_combo(
                "compute_type",
                (
                    ("Automatic", "auto"),
                    ("float16", "float16"),
                    ("int8 + float16", "int8_float16"),
                    ("int8", "int8"),
                    ("int8 + float32", "int8_float32"),
                    ("float32", "float32"),
                    ("bfloat16", "bfloat16"),
                ),
            ),
            "Choose a common faster-whisper compute type or enter a supported custom value.",
        )
        transcription.add_field(
            "Voice activity detector",
            self._combo(
                "vad_backend",
                [
                    ("RMS · lightweight", "rms"),
                    ("Silero · neural", "silero"),
                ],
            ),
            "Detects speech and trailing silence for sentence finalization.",
        )
        transcription.add_field("Nemotron model folder", self._path("realtime_preview_model_dir", folder=True), "Leave empty for automatic model discovery or download.")
        transcription.add_field("Realtime preview Python", self._path("realtime_preview_python", folder=False))
        transcription.add_field("Embedding helper Python", self._path("embedding_python", folder=False))
        self.sections.addWidget(transcription)

        speaker = FormSection()
        speaker.add_field(
            "Speaker model preset",
            self._combo(
                "provider_preset",
                [(preset.name, key) for key, preset in PROVIDER_PRESETS.items()]
                + [("Custom", "custom")],
            ),
        )
        speaker.add_field("Final provider", self._line("embedding_provider"), "Derived from the preset. Choose Custom to edit the provider expression directly.")
        speaker.add_field("Live provider", self._line("live_speaker_embedding_provider"), "Derived from the preset. Choose Custom to edit the provider expression directly.")
        self.sections.addWidget(speaker)

        connections = FormSection()
        self.connection_summary = QLabel()
        self.connection_summary.setWordWrap(True)
        self.connection_summary.setProperty("role", "secondary")
        connections.add_field("Backend topology", self.connection_summary)
        connections.add_field("Remote ASR URL", self._line("remote_asr_url"))
        connections.add_field("Remote embeddings URL", self._line("remote_embeddings_url"))
        self.sections.addWidget(connections)

        reports = FormSection()
        reports.add_field("Meeting Intelligence", self._check("reports_enabled", "Start Reports + Ask with the live window"))
        reports.add_field("Browser port", self._spin("reports_port", 1, 65535))
        reports.add_field(
            "Report language",
            self._combo(
                "report_language",
                [("Follow live language", "")]
                + sorted(
                    (
                        (cfg.display_name, code)
                        for code, cfg in SUPPORTED_LANGUAGE_CONFIGS.items()
                    ),
                    key=lambda item: item[0].casefold(),
                ),
            ),
            "Controls summaries and reports; the default follows the live language.",
        )
        reports.add_field("LLM provider", self._combo("report_llm_provider", [("llama.cpp", "llama_cpp"), ("Ollama", "ollama"), ("LM Studio", "lm_studio"), ("OpenAI-compatible", "openai_compatible"), ("OpenAI", "openai"), ("OpenRouter", "openrouter")]))
        reports.add_field("LLM base URL", self._line("report_llm_base_url"))
        reports.add_field(
            "LLM model",
            self._model_combo("report_llm_model"),
            "OpenAI models are loaded from the account associated with OPENAI_API_KEY.",
        )
        self.text_embedding_preset = QComboBox()
        self.text_embedding_preset.addItem("Not configured", "off")
        self.text_embedding_preset.addItem("OpenAI · small (recommended)", "openai_small")
        self.text_embedding_preset.addItem("OpenAI · large", "openai_large")
        self.text_embedding_preset.addItem("Custom OpenAI-compatible", "custom")
        self.text_embedding_preset.setAccessibleName("Semantic search preset")
        reports.add_field(
            "Semantic search",
            self.text_embedding_preset,
            "Required only for long or multi-session Ask. Selecting OpenAI sends transcript chunks to its embeddings API.",
        )
        reports.add_field("Text embedding base URL", self._line("text_embedding_base_url"))
        reports.add_field("Text embedding model", self._line("text_embedding_model"))
        reports.add_field("Text embedding key variable", self._line("text_embedding_api_key_env"), "Environment-variable name only; secrets are not stored in the profile.")
        self.openai_key_status = QLabel()
        self.openai_key_status.setAccessibleName("OpenAI API key status")
        reports.add_field(
            "OpenAI credentials",
            self.openai_key_status,
            "WhoSpeaks reads OPENAI_API_KEY from its process environment and never writes the secret to the launch profile.",
        )
        reports.add_field("Automatic reports", self._check("report_auto_generate", "Generate when a newly saved meeting is finalized"))
        self.reports_section = reports
        self.sections.addWidget(reports)

        translation = FormSection()
        translation.add_field("Translation", self._check("translation_enabled", "Translate stable transcript text"))
        translation.add_field(
            "On-device browser translation",
            self._check(
                "translation_browser_preferred",
                "Use Chrome's on-device Translator API when available",
            ),
            "Supported language pairs stay on-device; the selected provider remains the fallback.",
        )
        translation.add_field(
            "Provider",
            self._combo(
                "translation_provider",
                [
                    (option["label"], provider_id)
                    for provider_id, option in TRANSLATION_PROVIDER_OPTIONS.items()
                ],
            ),
        )
        target_languages = LanguageTargetSelector()
        self.fields["translation_target_languages"] = target_languages
        translation.add_field(
            "Target languages",
            target_languages,
            "Choose one or more languages. The live language cannot be selected as a target.",
        )
        translation.add_field("Maximum targets", self._spin("translation_max_targets", 1, 16))
        translation.add_field("Local model profile", self._combo("translation_model_profile", [("TranslateGemma 4B", "translate-gemma-4b"), ("NLLB-200 600M", "nllb-200-600m"), ("MADLAD-400 3B", "madlad-400-3b")]))
        translation.add_field("Model override", self._line("translation_model"))
        translation.add_field("Base URL", self._line("translation_base_url"))
        translation.add_field("API-key variable", self._line("translation_api_key_env"))
        translation.add_field("Region", self._line("translation_region"))
        translation.add_field("Sidecar Python", self._path("translation_python", folder=False))
        translation.add_field("Sidecar port", self._spin("translation_port", 1, 65535))
        translation.add_field("Device", self._combo("translation_device", [("Automatic", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")]))
        self.sections.addWidget(translation)

        advanced = FormSection()
        advanced.add_field("Advanced launch arguments", self._line("advanced_args"), "Appended verbatim after the validated saved launch profile.")
        self.sections.addWidget(advanced)
        self._sync_dependencies()

    def _connect_dirty_signals(self) -> None:
        for widget in self.fields.values():
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._mark_dirty)
            elif isinstance(widget, PathPicker):
                widget.textEdited.connect(self._mark_dirty)
            elif isinstance(widget, LanguageTargetSelector):
                widget.selectionChanged.connect(self._mark_dirty)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._mark_dirty)
                if widget.isEditable():
                    widget.editTextChanged.connect(self._mark_dirty)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._mark_dirty)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._mark_dirty)
        engine = self.fields["realtime_preview_engine"]
        provider = self.fields["translation_provider"]
        enabled = self.fields["translation_enabled"]
        speaker_preset = self.fields["provider_preset"]
        report_provider = self.fields["report_llm_provider"]
        mode = self.fields["mode"]
        reports_enabled = self.fields["reports_enabled"]
        source_language = self.fields["language"]
        assert (
            isinstance(engine, QComboBox)
            and isinstance(provider, QComboBox)
            and isinstance(enabled, QCheckBox)
            and isinstance(speaker_preset, QComboBox)
            and isinstance(report_provider, QComboBox)
            and isinstance(mode, QComboBox)
            and isinstance(reports_enabled, QCheckBox)
            and isinstance(source_language, QComboBox)
        )
        engine.currentIndexChanged.connect(self._sync_dependencies)
        provider.currentIndexChanged.connect(self._sync_translation_provider)
        enabled.toggled.connect(self._sync_dependencies)
        mode.currentIndexChanged.connect(self._sync_dependencies)
        reports_enabled.toggled.connect(self._sync_dependencies)
        source_language.currentIndexChanged.connect(self._sync_dependencies)
        speaker_preset.currentIndexChanged.connect(self._sync_provider_preset)
        report_provider.currentIndexChanged.connect(self._sync_llm_provider)
        self.text_embedding_preset.currentIndexChanged.connect(self._sync_text_embedding_preset)

    def _sync_llm_provider(self, *_args: object) -> None:
        provider = str(self._value("report_llm_provider"))
        option = LLM_PROVIDER_OPTIONS.get(provider)
        if option is None:
            return
        base_url = self.fields["report_llm_base_url"]
        model = self.fields["report_llm_model"]
        assert isinstance(base_url, QLineEdit) and isinstance(model, QComboBox)
        base_url.setText(str(option["default_base_url"]))
        models = list(option.get("models") or [])
        self._configure_llm_model_widget(
            provider,
            selected=str(models[0]) if models else "",
            load_openai=True,
        )
        self._sync_dependencies()

    def _configure_llm_model_widget(
        self,
        provider: str,
        *,
        selected: str = "",
        load_openai: bool,
    ) -> None:
        model = self.fields["report_llm_model"]
        assert isinstance(model, QComboBox)
        option = LLM_PROVIDER_OPTIONS.get(provider) or {}
        fallback_models = [str(item) for item in option.get("models") or []]
        choices = list(self._openai_models) if provider == "openai" else fallback_models
        if selected and selected not in choices:
            choices.insert(0, selected)

        model.blockSignals(True)
        model.clear()
        model.setEditable(provider != "openai" or not self._openai_models)
        model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for model_id in choices:
            model.addItem(model_id, model_id)
        if selected:
            index = model.findData(selected)
            if index >= 0:
                model.setCurrentIndex(index)
            elif model.isEditable():
                model.setEditText(selected)
        model.blockSignals(False)

        if provider == "openai" and load_openai:
            self._request_openai_models()

    def _request_openai_models(self) -> None:
        if self._openai_models:
            self._apply_openai_models(self._openai_models)
            return
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._set_openai_catalog_status(
                "OPENAI_API_KEY not detected · enter a model ID manually or add the key to load your account models",
                "warning",
            )
            return
        if self._openai_request_running:
            return

        self._set_openai_catalog_status(
            "OPENAI_API_KEY detected · loading available models…",
            "info",
        )
        self._openai_request_running = True
        self._openai_request_serial += 1
        serial = self._openai_request_serial
        bridge = self._openai_catalog_bridge

        def load_catalog() -> None:
            models, error = _fetch_openai_model_ids(api_key)
            try:
                bridge.completed.emit(serial, models, error)
            except RuntimeError:
                # The launcher was closed while the request was in flight.
                pass

        thread = threading.Thread(
            target=load_catalog,
            name=f"whospeaks-openai-models-{serial}",
            daemon=True,
        )
        self._openai_catalog_threads[serial] = thread
        thread.start()

    def _openai_models_finished(
        self,
        serial: int,
        returned_models: object,
        error: str,
    ) -> None:
        self._openai_catalog_threads.pop(serial, None)
        if serial != self._openai_request_serial:
            return
        self._openai_request_running = False
        if error:
            self._set_openai_catalog_status(
                error,
                "warning",
            )
            return
        models = [str(model) for model in returned_models] if isinstance(returned_models, list) else []
        if not models:
            self._set_openai_catalog_status(
                "OpenAI returned no suitable text-generation models",
                "warning",
            )
            return
        self._openai_models = models
        if str(self._value("report_llm_provider")) == "openai":
            self._apply_openai_models(models)
        else:
            self._set_openai_catalog_status(
                f"OPENAI_API_KEY detected · {len(models)} suitable models loaded from OpenAI",
                "success",
            )

    def _apply_openai_models(self, models: list[str]) -> None:
        model = self.fields["report_llm_model"]
        assert isinstance(model, QComboBox)
        if model.view().isVisible():
            self._pending_openai_models = list(models)
            self._openai_apply_timer.start()
            self._set_openai_catalog_status(
                f"OPENAI_API_KEY detected · {len(models)} suitable models loaded from OpenAI",
                "success",
            )
            return
        self._pending_openai_models = None
        selected = str(self._value("report_llm_model") or "")
        choices = list(models)
        if selected and selected not in choices:
            choices.insert(0, selected)
        model.blockSignals(True)
        model.clear()
        model.setEditable(True)
        model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for model_id in choices:
            model.addItem(model_id, model_id)
        index = model.findData(selected)
        if index >= 0:
            model.setCurrentIndex(index)
        elif choices:
            model.setCurrentIndex(0)
        model.blockSignals(False)
        self._set_openai_catalog_status(
            f"OPENAI_API_KEY detected · {len(models)} suitable models loaded from OpenAI",
            "success",
        )

    def _apply_pending_openai_models(self) -> None:
        if self._pending_openai_models is None:
            return
        model = self.fields["report_llm_model"]
        assert isinstance(model, QComboBox)
        if model.view().isVisible():
            self._openai_apply_timer.start()
            return
        models = self._pending_openai_models
        self._pending_openai_models = None
        self._apply_openai_models(models)

    def _set_openai_catalog_status(self, text: str, role: str) -> None:
        self._openai_catalog_status = text
        self._openai_catalog_role = role
        self._update_openai_key_status()

    def _update_openai_key_status(self) -> None:
        provider = str(self._value("report_llm_provider"))
        openai_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
        if provider == "openai" and self._openai_catalog_status:
            text = self._openai_catalog_status
            role = self._openai_catalog_role
        else:
            text = (
                "OPENAI_API_KEY detected in the launcher environment"
                if openai_key_present
                else "OPENAI_API_KEY not detected in the launcher environment"
            )
            role = "success" if openai_key_present else "warning"
        self.openai_key_status.setText(text)
        self.openai_key_status.setProperty("role", role)
        self.openai_key_status.style().unpolish(self.openai_key_status)
        self.openai_key_status.style().polish(self.openai_key_status)

    def _sync_text_embedding_preset(self, *_args: object) -> None:
        preset = str(self.text_embedding_preset.currentData() or "custom")
        base_url = self.fields["text_embedding_base_url"]
        model = self.fields["text_embedding_model"]
        key_env = self.fields["text_embedding_api_key_env"]
        assert isinstance(base_url, QLineEdit)
        assert isinstance(model, QLineEdit)
        assert isinstance(key_env, QLineEdit)
        if preset == "off":
            base_url.clear()
            model.clear()
            key_env.clear()
        elif preset in {"openai_small", "openai_large"}:
            base_url.setText("https://api.openai.com/v1")
            model.setText(
                "text-embedding-3-small"
                if preset == "openai_small"
                else "text-embedding-3-large"
            )
            key_env.setText("OPENAI_API_KEY")
        self._sync_dependencies()

    def _sync_translation_provider(self, *_args: object) -> None:
        provider = str(self._value("translation_provider"))
        base_url = self.fields["translation_base_url"]
        key_env = self.fields["translation_api_key_env"]
        assert isinstance(base_url, QLineEdit) and isinstance(key_env, QLineEdit)
        option = TRANSLATION_PROVIDER_OPTIONS[provider]
        user_changed_provider = len(_args) > 0
        if user_changed_provider or not base_url.text().strip():
            base_url.setText(str(option["default_base_url"]))
        if user_changed_provider or not key_env.text().strip():
            key_env.setText(str(option["default_api_key_env"]))
        self._sync_dependencies()

    def _sync_provider_preset(self, *_args: object) -> None:
        preset = PROVIDER_PRESETS.get(str(self._value("provider_preset")))
        final_provider = self.fields["embedding_provider"]
        live_provider = self.fields["live_speaker_embedding_provider"]
        assert isinstance(final_provider, QLineEdit) and isinstance(live_provider, QLineEdit)
        if preset is not None:
            final_provider.setText(preset.embedding_provider)
            live_provider.setText(preset.live_speaker_embedding_provider)
        self._sync_dependencies()

    def _sync_preview_model_choices(self, *, selected: str = "") -> None:
        engine = str(self._value("realtime_preview_engine"))
        model = self.fields["realtime_preview_model_preset"]
        assert isinstance(model, QComboBox)
        choices = _PREVIEW_MODEL_CHOICES.get(engine, _PREVIEW_MODEL_CHOICES["off"])
        current = selected or str(model.currentData() or "")
        model.blockSignals(True)
        model.clear()
        for label, value in choices:
            model.addItem(label, value)
        index = model.findData(current)
        model.setCurrentIndex(index if index >= 0 else 0)
        model.blockSignals(False)

    def _mark_dirty(self, *_args: object) -> None:
        self.save_button.setEnabled(True)
        self.status.setText("Unsaved changes")
        self.status.setProperty("role", "warning")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self._refresh_context_help()

    def _set_field_visible(self, field: str, visible: bool) -> None:
        widget = self.fields[field]
        for index in range(self.sections.count()):
            section = self.sections.widget(index)
            if isinstance(section, (FormSection, GeneralSettingsSection)):
                section.set_field_visible(widget, visible)

    def _sync_dependencies(self, *_args: object) -> None:
        engine = str(self._value("realtime_preview_engine"))
        deployment = str(self._value("mode"))
        mode = "remote" if deployment == "macos" else deployment
        server_profile = mode == "server"
        self._sync_preview_model_choices()
        self._set_field_visible(
            "realtime_preview_engine", not server_profile
        )
        self._set_field_visible(
            "realtime_preview_model_preset", not server_profile and engine != "off"
        )
        self._set_field_visible(
            "realtime_preview_model_dir", not server_profile and engine == "sherpa_onnx"
        )
        self._set_field_visible(
            "realtime_preview_python", not server_profile and engine == "kroko_onnx"
        )
        if deployment == "local":
            topology = "Final ASR and speaker embeddings run inside the local Live window process."
        elif deployment == "remote":
            topology = "The local controller connects to the two remote HTTP services below."
        elif deployment == "macos":
            topology = "The launcher manages local Apple Silicon ASR and embeddings services."
        else:
            topology = "This profile installs and exposes the ASR and embeddings service packages."
        self.connection_summary.setText(topology)
        self._set_field_visible("remote_asr_url", deployment == "remote")
        self._set_field_visible("remote_embeddings_url", deployment == "remote")
        for field in ("language", "live_speaker_assignment", "host", "port"):
            self._set_field_visible(field, not server_profile)
        self._set_field_visible("embedding_python", deployment == "local")
        self._set_field_visible("model", deployment == "local")
        self._set_field_visible("device", deployment == "local")
        self._set_field_visible("compute_type", deployment == "local")
        self._set_field_visible("vad_backend", deployment in {"local", "remote", "macos"})

        custom_speakers = str(self._value("provider_preset")) == "custom"
        for field in ("embedding_provider", "live_speaker_embedding_provider"):
            provider_field = self.fields[field]
            assert isinstance(provider_field, QLineEdit)
            provider_field.setReadOnly(not custom_speakers)
            self._set_field_visible(field, not server_profile)
        self._set_field_visible("provider_preset", not server_profile)

        embedding_preset = str(self.text_embedding_preset.currentData() or "off")
        for field in (
            "text_embedding_base_url",
            "text_embedding_model",
            "text_embedding_api_key_env",
        ):
            embedding_field = self.fields[field]
            assert isinstance(embedding_field, QLineEdit)
            embedding_field.setReadOnly(embedding_preset != "custom")

        translation_enabled = bool(self._value("translation_enabled")) and not server_profile
        provider = str(self._value("translation_provider"))
        target_selector = self.fields["translation_target_languages"]
        assert isinstance(target_selector, LanguageTargetSelector)
        target_selector.set_source_language(str(self._value("language")))
        self._set_field_visible("translation_enabled", not server_profile)
        translation_fields = (
            "translation_browser_preferred",
            "translation_provider",
            "translation_target_languages",
            "translation_max_targets",
            "translation_model_profile",
            "translation_model",
            "translation_base_url",
            "translation_api_key_env",
            "translation_region",
            "translation_python",
            "translation_port",
            "translation_device",
        )
        for field in translation_fields:
            self._set_field_visible(field, translation_enabled)
        local_translation = provider in {"sidecar", "transformers"}
        self._set_field_visible("translation_model_profile", translation_enabled and local_translation)
        self._set_field_visible("translation_device", translation_enabled and local_translation)
        self._set_field_visible(
            "translation_model",
            translation_enabled and provider in {"sidecar", "transformers", "reports_llm", "openai_compatible"},
        )
        self._set_field_visible(
            "translation_base_url",
            translation_enabled and provider not in {"sidecar", "transformers", "reports_llm"},
        )
        self._set_field_visible(
            "translation_api_key_env",
            translation_enabled and provider in {"deepl", "google_cloud", "azure_translator", "libretranslate", "openai_compatible"},
        )
        self._set_field_visible("translation_port", translation_enabled and provider == "sidecar")
        self._set_field_visible("translation_python", translation_enabled and provider == "sidecar")
        self._set_field_visible("translation_region", translation_enabled and provider == "azure_translator")

        reports_enabled = bool(self._value("reports_enabled")) and not server_profile
        needs_report_llm_config = reports_enabled or (
            translation_enabled and provider == "reports_llm"
        )
        self._set_field_visible("reports_enabled", not server_profile)
        for field in ("reports_port", "report_language"):
            self._set_field_visible(field, reports_enabled)
        for field in (
            "report_llm_provider",
            "report_llm_base_url",
            "report_llm_model",
        ):
            self._set_field_visible(field, needs_report_llm_config)
        report_only_fields = (
            "text_embedding_base_url",
            "text_embedding_model",
            "text_embedding_api_key_env",
            "report_auto_generate",
        )
        for field in report_only_fields:
            self._set_field_visible(field, reports_enabled)
        self.reports_section.set_field_visible(self.text_embedding_preset, reports_enabled)
        embedding_is_configured = embedding_preset != "off"
        for field in ("text_embedding_base_url", "text_embedding_model", "text_embedding_api_key_env"):
            self._set_field_visible(field, reports_enabled and embedding_is_configured)
        uses_openai = (
            needs_report_llm_config
            and (
                str(self._value("report_llm_provider")) == "openai"
                or (reports_enabled and embedding_preset in {"openai_small", "openai_large"})
            )
        )
        self.reports_section.set_field_visible(self.openai_key_status, uses_openai)
        self._set_field_visible("advanced_args", not server_profile)
        for index in (1, 2, 4, 5, 6):
            self.section_list.item(index).setHidden(server_profile)
        if server_profile and self.section_list.currentRow() in {1, 2, 4, 5, 6}:
            self.section_list.setCurrentRow(0)
        self._refresh_context_help()
        self._update_openai_key_status()

    def _value(self, field: str) -> object:
        widget = self.fields[field]
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, PathPicker):
            return widget.text()
        if isinstance(widget, LanguageTargetSelector):
            return ",".join(widget.selected_codes())
        if isinstance(widget, QComboBox):
            if widget.isEditable():
                index = widget.currentIndex()
                if index >= 0 and widget.currentText() == widget.itemText(index):
                    return widget.currentData()
                return widget.currentText().strip()
            return widget.currentData()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        raise TypeError(f"Unsupported settings widget for {field}")

    def values(self) -> dict[str, object]:
        values = {field: self._value(field) for field in self.fields}
        deployment = str(values["mode"])
        values["mode"] = "remote" if deployment == "macos" else deployment
        values["deployment_target"] = "macos" if deployment == "macos" else ""
        if values["mode"] in {"local", "remote"}:
            values["asr_backend"] = values["mode"]
            values["embeddings_backend"] = values["mode"]
        return values

    def _emit_save(self) -> None:
        self.save_requested.emit(self.values())

    def set_profile(self, profile: Profile) -> None:
        self.clear_validation()
        self.profile = profile
        values = profile.as_dict()
        for field, widget in self.fields.items():
            if field in {"mode", "report_llm_model", "realtime_preview_model_preset"}:
                continue
            value = values[field]
            widget.blockSignals(True)
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, PathPicker):
                widget.setText(str(value))
            elif isinstance(widget, LanguageTargetSelector):
                widget.set_selected_codes(str(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                elif widget.isEditable():
                    widget.setEditText(str(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            widget.blockSignals(False)
        deployment = "macos" if profile.deployment_target == "macos" else profile.mode
        mode_widget = self.fields["mode"]
        assert isinstance(mode_widget, QComboBox)
        mode_widget.blockSignals(True)
        mode_widget.setCurrentIndex(mode_widget.findData(deployment))
        mode_widget.blockSignals(False)
        embedding_base_url = str(profile.text_embedding_base_url or "").rstrip("/")
        embedding_model = str(profile.text_embedding_model or "")
        if not embedding_base_url and not embedding_model:
            embedding_preset = "off"
        elif embedding_base_url == "https://api.openai.com/v1" and embedding_model == "text-embedding-3-small":
            embedding_preset = "openai_small"
        elif embedding_base_url == "https://api.openai.com/v1" and embedding_model == "text-embedding-3-large":
            embedding_preset = "openai_large"
        else:
            embedding_preset = "custom"
        self.text_embedding_preset.blockSignals(True)
        self.text_embedding_preset.setCurrentIndex(
            self.text_embedding_preset.findData(embedding_preset)
        )
        self.text_embedding_preset.blockSignals(False)
        self._sync_preview_model_choices(
            selected=str(profile.realtime_preview_model_preset or "")
        )
        llm_provider = str(profile.report_llm_provider)
        llm_option = LLM_PROVIDER_OPTIONS.get(llm_provider) or {}
        llm_base_url = self.fields["report_llm_base_url"]
        assert isinstance(llm_base_url, QLineEdit)
        if not llm_base_url.text().strip():
            llm_base_url.setText(str(llm_option.get("default_base_url") or ""))
        llm_models = [str(item) for item in llm_option.get("models") or []]
        selected_llm_model = str(profile.report_llm_model or "")
        if not selected_llm_model and llm_models:
            selected_llm_model = llm_models[0]
        self._configure_llm_model_widget(
            llm_provider,
            selected=selected_llm_model,
            load_openai=bool(
                profile.reports_enabled
                or (
                    profile.translation_enabled
                    and profile.translation_provider == "reports_llm"
                )
            ),
        )
        self._sync_translation_provider()
        self._sync_provider_preset()
        self._sync_dependencies()
        self.status.setText("No unsaved changes")
        self.save_button.setDisabled(True)
        self.status.setProperty("role", "success")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        if self._active_help_widget is None:
            self._show_control_help("mode", mode_widget)
        else:
            self._refresh_context_help()

    def clear_validation(self) -> None:
        for field, widget in self.fields.items():
            widget.setProperty("invalid", False)
            widget.setProperty("validationMessage", "")
            if field in _SETTINGS_HELP:
                widget.setAccessibleDescription(_SETTINGS_HELP[field][1])
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        for form, label in self._validation_rows:
            form.removeWidget(label)
            label.deleteLater()
        self._validation_rows.clear()
        for index in range(self.sections.count()):
            section = self.sections.widget(index)
            if isinstance(section, GeneralSettingsSection):
                section.clear_errors()

    def show_validation_error(
        self,
        field: str,
        message: str,
        *,
        value: object | None = None,
    ) -> None:
        self.clear_validation()
        widget = self.fields.get(field)
        if widget is None:
            self.show_error(message)
            return
        section_index = self._field_sections.get(field)
        if section_index is not None:
            self.section_list.item(section_index).setHidden(False)
            self.section_list.setCurrentRow(section_index)
        if value is not None:
            if isinstance(widget, QSpinBox):
                widget.setValue(
                    min(widget.maximum(), max(widget.minimum(), int(value)))
                )
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, PathPicker):
                widget.setText(str(value))
            elif isinstance(widget, LanguageTargetSelector):
                widget.set_selected_codes(str(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                elif widget.isEditable():
                    widget.setEditText(str(value))
        widget.setProperty("invalid", True)
        widget.setProperty("validationMessage", message)
        widget.setAccessibleDescription(message)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        for index in range(self.sections.count()):
            section = self.sections.widget(index)
            if isinstance(section, GeneralSettingsSection):
                if section.show_error(widget, field, message) is not None:
                    break
            if not isinstance(section, FormSection):
                continue
            row, _role = section.form.getWidgetPosition(widget)
            if row >= 0:
                error = QLabel(message)
                error.setProperty("role", "error")
                error.setAccessibleName(f"{field.replace('_', ' ')} error")
                section.form.insertRow(row + 1, "", error)
                self._validation_rows.append((section.form, error))
                break
        widget.setFocus(Qt.FocusReason.OtherFocusReason)
        self._show_control_help(field, widget)
        self.status.setText("1 setting needs attention")
        self.status.setProperty("role", "error")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def show_saved(self, path: str) -> None:
        self.save_button.setDisabled(True)
        self.status.setText(f"Saved · {path}")
        self.status.setProperty("role", "success")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def show_error(self, message: str) -> None:
        self.save_button.setEnabled(True)
        self.status.setText(message)
        self.status.setProperty("role", "error")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


class ActivityPage(QWidget):
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 32, 18, 0)
        layout.setSpacing(0)
        header = PageHeader("Activity", "Installation, checks, and service events from this launcher session.")
        header.layout().setContentsMargins(22, 0, 14, 0)
        layout.addWidget(header)
        layout.addSpacing(28)
        self.summary = SummaryStrip()
        self.operation_mark = self.summary.mark
        self.operation_state = self.summary.state_label
        self.operation_label = self.summary.detail_label
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("danger", True)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.summary.add_trailing_widget(self.cancel_button)
        self.progress = self.summary.progress
        layout.addWidget(self.summary)
        layout.addSpacing(12)
        self.log = QPlainTextEdit()
        self.log.setObjectName("activityLog")
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setMaximumBlockCount(10_000)
        font = QFont("Cascadia Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(12)
        self.log.setFont(font)
        self.log.setAccessibleName("Launcher activity log")
        layout.addWidget(self.log, 1)
        self.action_bar = ActionFooter()
        tools = self.action_bar.actions
        self.copy_button = QPushButton("Copy all")
        self.copy_button.setIcon(line_icon("copy", COLORS.canvas))
        self.wrap_check = QCheckBox("Wrap long lines")
        self.clear_button = QPushButton("Clear")
        self.action_bar.configure_button(self.copy_button, primary=True)
        self.action_bar.configure_button(self.clear_button)
        self.line_count = QLabel("0 lines")
        self.line_count.setProperty("role", "secondary")
        self.copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        self.wrap_check.toggled.connect(self._toggle_wrap)
        self.clear_button.clicked.connect(self.clear)
        tools.addWidget(self.copy_button)
        tools.addWidget(self.clear_button)
        tools.addWidget(self.wrap_check)
        tools.addStretch(1)
        tools.addWidget(self.line_count)
        layout.addWidget(self.action_bar)

    def _toggle_wrap(self, enabled: bool) -> None:
        self.log.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth if enabled else QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.log.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere if enabled else QTextOption.WrapMode.NoWrap)

    def set_logs(self, lines: list[str] | tuple[str, ...]) -> None:
        self.log.setPlainText("\n".join(lines))
        self._update_count()
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        self._update_count()

    def clear(self) -> None:
        self.log.clear()
        self._update_count()

    def _update_count(self) -> None:
        count = 0 if not self.log.toPlainText() else self.log.document().blockCount()
        self.line_count.setText(f"{count:,} lines")

    def set_operation(self, operation: object) -> None:
        name = str(getattr(operation, "name", ""))
        status = str(getattr(operation, "status", "idle"))
        title = str(getattr(operation, "title", "Setup is idle"))
        step = str(getattr(operation, "step", ""))
        if name:
            state = title
            detail = step
            semantic = "info" if status == "running" else status
        else:
            state = status
            detail = title
            semantic = {
                "success": "success",
                "warning": "warning",
                "cancelled": "warning",
                "error": "error",
            }.get(status, "secondary")
        self.summary.set_summary(state, detail, semantic=semantic)
        running = bool(name)
        self.progress.setVisible(running or status in {"cancelled", "success", "warning", "error"})
        self.progress.setRange(0, 0 if running else 100)
        self.cancel_button.setVisible(name in {"install", "launch"})


class LargeBrandMark(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(176, 176)
        self.setAccessibleName("WhoSpeaks brand mark")

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(COLORS.border_strong), 1.5))
        painter.setBrush(QColor(COLORS.surface_1))
        painter.drawRoundedRect(QRectF(1, 1, 174, 174), 28, 28)
        painter.setPen(
            QPen(
                QColor(COLORS.accent),
                11,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        for x, top, bottom in (
            (47, 74, 105),
            (72, 54, 125),
            (97, 32, 147),
            (122, 70, 109),
        ):
            painter.drawLine(x, top, x, bottom)
        painter.end()


class AboutPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 32, 18, 24)
        layout.setSpacing(32)
        header = PageHeader(
            "About",
            "WhoSpeaks local speaker diarization and realtime voice labeling.",
        )
        header.layout().setContentsMargins(22, 0, 14, 0)
        layout.addWidget(header)
        panel = QFrame()
        panel.setProperty("group", True)
        panel.setMinimumHeight(700)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(50, 36, 40, 24)
        panel_layout.setSpacing(43)

        identity = QHBoxLayout()
        identity.setContentsMargins(8, 0, 0, 0)
        identity.setSpacing(40)
        identity.addWidget(LargeBrandMark())
        identity_text = QVBoxLayout()
        identity_text.setSpacing(10)
        name = QLabel("WhoSpeaks")
        name.setObjectName("pageTitle")
        version = QLabel(f"Version {__version__}")
        version.setProperty("role", "secondary")
        self.creator_credit = QLabel(
            f'Created by <a style="color:{COLORS.accent}" href="https://github.com/KoljaB">Kolja Beigel</a>'
            f' · <a style="color:{COLORS.accent}" href="https://github.com/KoljaB/WhoSpeaksLive">WhoSpeaksLive on GitHub</a>'
        )
        self.creator_credit.setProperty("role", "secondary")
        self.creator_credit.setOpenExternalLinks(True)
        self.creator_credit.setAccessibleName(
            "Created by Kolja Beigel. Open Kolja Beigel or WhoSpeaksLive on GitHub."
        )
        description = QLabel(
            "This desktop launcher manages setup, diagnostics, saved configuration,\n"
            "and the local service launch group."
        )
        description.setWordWrap(True)
        description.setProperty("role", "secondary")
        identity_text.addStretch(1)
        identity_text.addWidget(name)
        identity_text.addWidget(version)
        identity_text.addWidget(self.creator_credit)
        identity_text.addWidget(description)
        identity_text.addStretch(1)
        identity.addLayout(identity_text, 1)
        panel_layout.addLayout(identity)

        divider = QFrame()
        divider.setProperty("separator", True)
        panel_layout.addWidget(divider)
        columns = QHBoxLayout()
        columns.setSpacing(40)
        interfaces = QVBoxLayout()
        interfaces.setSpacing(27)
        interfaces.addWidget(section_label("Interfaces"))
        for icon_name, label in (
            ("video", "Desktop launcher — Primary local experience"),
            ("terminal", "CLI — Automation and headless operation"),
        ):
            row = QHBoxLayout()
            icon = QLabel()
            icon.setPixmap(
                line_icon(icon_name, COLORS.text_primary, 24).pixmap(24, 24)
            )
            icon.setProperty("iconTile", True)
            icon.setFixedSize(54, 54)
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(icon)
            text = QLabel(label)
            text.setProperty("role", "secondary")
            row.addWidget(text, 1)
            interfaces.addLayout(row)
        interfaces.addStretch(1)

        keyboard = QGridLayout()
        keyboard.setHorizontalSpacing(18)
        keyboard.setVerticalSpacing(20)
        keyboard.setAlignment(Qt.AlignmentFlag.AlignTop)
        keyboard.addWidget(section_label("Keyboard"), 0, 0, 1, 2)
        for row, (keys, action) in enumerate(
            (
                ("Ctrl+R", "Refresh checks"),
                ("Ctrl+L", "Launch"),
                ("Ctrl+,", "Open Settings"),
                ("Ctrl+Q", "Exit"),
            ),
            start=1,
        ):
            keycap = QLabel(keys)
            keycap.setProperty("role", "keycap")
            keycap.setFixedSize(106, 52)
            keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            keyboard.addWidget(keycap, row, 0)
            action_label = QLabel(action)
            action_label.setProperty("role", "secondary")
            keyboard.addWidget(action_label, row, 1)

        columns.addLayout(interfaces, 1)
        vertical = QFrame()
        vertical.setFrameShape(QFrame.Shape.VLine)
        vertical.setProperty("separator", True)
        vertical.setFixedWidth(1)
        columns.addWidget(vertical)
        columns.addLayout(keyboard, 1)
        panel_layout.addLayout(columns, 1)
        footer = QLabel("Configuration remains compatible across every interface.")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setProperty("role", "muted")
        panel_layout.addWidget(footer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setAccessibleName("About WhoSpeaks")
        layout.addWidget(scroll, 1)
