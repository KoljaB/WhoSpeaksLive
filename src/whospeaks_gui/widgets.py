"""Reusable native Qt widgets for the WhoSpeaks desktop launcher."""

from __future__ import annotations

from PySide6.QtCore import Property, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .icons import line_icon
from .tokens import COLORS


FOOTER_HEIGHT = 88
FOOTER_ACTION_HEIGHT = 48


class StatusMark(QWidget):
    """Paint a semantic status mark that remains distinct without color."""

    def __init__(self, status: str = "stopped", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = status
        self._phase = 0
        self.setFixedSize(22, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._sync_accessibility()

    def sizeHint(self) -> QSize:
        return QSize(22, 22)

    def status(self) -> str:
        return self._status

    def set_status(self, value: str) -> None:
        normalized = str(value or "stopped").lower()
        if normalized == self._status:
            return
        self._status = normalized
        self._phase = 0
        self._sync_accessibility()
        self.update()

    semanticStatus = Property(str, status, set_status)

    def advance(self) -> None:
        if self._status in {"starting", "loading", "running_check", "stopping"}:
            self._phase = (self._phase + 30) % 360
            self.update()

    def set_phase(self, degrees: int) -> None:
        """Set the activity arc phase for deterministic motion review frames."""

        self._phase = int(degrees) % 360
        self.update()

    def _sync_accessibility(self) -> None:
        self.setAccessibleName(f"Status: {self._status.replace('_', ' ')}")

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(3, 3, 16, 16)
        status = self._status
        color = {
            "running": COLORS.success,
            "ok": COLORS.success,
            "ready": COLORS.success,
            "warning": COLORS.warning,
            "warn": COLORS.warning,
            "degraded": COLORS.warning,
            "failed": COLORS.error,
            "fail": COLORS.error,
            "error": COLORS.error,
            "starting": COLORS.info,
            "loading": COLORS.info,
            "running_check": COLORS.info,
            "stopping": COLORS.info,
            "disconnected": COLORS.warning,
            "unavailable": COLORS.warning,
            "unknown": COLORS.warning,
            "cancelled": COLORS.warning,
            "success": COLORS.success,
        }.get(status, COLORS.text_muted)
        pen = QPen(QColor(color), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if status in {"starting", "loading", "running_check", "stopping"}:
            painter.drawArc(rect, (30 + self._phase) * 16, 265 * 16)
        elif status in {"running", "ok", "ready", "success"}:
            painter.drawEllipse(rect)
            painter.drawLine(7, 11, 10, 14)
            painter.drawLine(10, 14, 16, 8)
        elif status in {"warning", "warn", "degraded", "disconnected", "unavailable", "unknown", "cancelled"}:
            painter.drawLine(11, 3, 20, 19)
            painter.drawLine(20, 19, 2, 19)
            painter.drawLine(2, 19, 11, 3)
            painter.drawLine(11, 8, 11, 13)
            painter.drawPoint(11, 16)
        elif status in {"failed", "fail", "error"}:
            painter.drawEllipse(rect)
            painter.drawLine(8, 8, 14, 14)
            painter.drawLine(14, 8, 8, 14)
        else:
            painter.drawEllipse(rect)
        painter.end()


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.title = QLabel(title)
        self.title.setObjectName("pageTitle")
        self.title.setAccessibleName(title)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("pageSubtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        self.setFixedHeight(76)

    def set_text(self, title: str, subtitle: str) -> None:
        self.title.setText(title)
        self.title.setAccessibleName(title)
        self.subtitle.setText(subtitle)


class SummaryStrip(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("summaryStrip")
        self.setFixedHeight(68)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.row_layout = QHBoxLayout()
        self.row_layout.setContentsMargins(18, 0, 18, 0)
        self.row_layout.setSpacing(12)
        self.mark = StatusMark("stopped")
        self.state_label = QLabel("NOT CHECKED")
        self.state_label.setProperty("role", "muted")
        font = self.state_label.font()
        font.setWeight(QFont.Weight.DemiBold)
        self.state_label.setFont(font)
        self.detail_label = QLabel("Readiness has not been checked")
        self.detail_label.setProperty("role", "secondary")
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.row_layout.addWidget(self.mark)
        self.row_layout.addWidget(self.state_label)
        self.row_layout.addWidget(QLabel("·"))
        self.row_layout.addWidget(self.detail_label)
        self.row_layout.addStretch(1)
        layout.addLayout(self.row_layout, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

    def add_trailing_widget(self, widget: QWidget) -> None:
        self.row_layout.insertWidget(self.row_layout.count() - 1, widget)

    def set_summary(self, state: str, detail: str, *, semantic: str) -> None:
        self.state_label.setText(state.upper())
        self.state_label.setProperty("role", semantic)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.detail_label.setText(detail)
        mark_status = {
            "success": "ready",
            "warning": "warning",
            "unavailable": "warning",
            "error": "failed",
            "info": "starting",
        }.get(semantic, "stopped")
        self.mark.set_status(mark_status)
        self.setAccessibleName(f"{state}. {detail}")


class ActionFooter(QFrame):
    """Shared bottom action area used by all interactive launcher pages."""

    def __init__(self, *, seamless: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.setProperty("seamless", seamless)
        self.setFixedHeight(FOOTER_HEIGHT)
        self.actions = QHBoxLayout(self)
        self.actions.setContentsMargins(18, 20, 18, 20)
        self.actions.setSpacing(16)

    def configure_button(self, button: QPushButton, *, primary: bool = False) -> QPushButton:
        button.setProperty("footerAction", True)
        if primary:
            button.setProperty("primary", True)
        button.setFixedHeight(FOOTER_ACTION_HEIGHT)
        button.setIconSize(QSize(22, 22))
        return button


class EndpointLink(QPushButton):
    """A compact browser link that occupies the same space as endpoint text."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("endpointLink", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(14, 14))
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setEnabled(False)
        self._sync_presentation(False)

    def set_available(self, available: bool, *, interface_name: str) -> None:
        self.setEnabled(available)
        self._sync_presentation(available)
        endpoint = self.text()
        if available:
            self.setAccessibleName(f"Open {interface_name} at {endpoint}")
            self.setToolTip(f"Open {interface_name} in your browser")
        else:
            self.setAccessibleName(f"{interface_name} at {endpoint}; not available yet")
            self.setToolTip(f"{interface_name} can be opened after the service is running")

    def _sync_presentation(self, available: bool) -> None:
        self.setIcon(
            line_icon("external_link", COLORS.accent, 14)
            if available
            else line_icon("external_link", COLORS.text_muted, 14)
        )


class ServiceRow(QWidget):
    def __init__(
        self,
        title: str,
        subtitle: str,
        endpoint: str,
        icon_name: str,
        *,
        endpoint_link: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedHeight(88)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 16, 5)
        layout.setSpacing(24)
        icon = QLabel()
        icon.setPixmap(line_icon(icon_name, size=25).pixmap(25, 25))
        icon.setProperty("iconTile", True)
        icon.setFixedSize(54, 54)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setAccessibleName(f"{title} icon")
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        self.title = QLabel(title)
        self.title.setProperty("role", "serviceTitle")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setProperty("role", "secondary")
        if endpoint_link:
            self.endpoint = EndpointLink(endpoint)
        else:
            self.endpoint = QLabel(endpoint)
            self.endpoint.setProperty("role", "secondary")
            self.endpoint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_row = QHBoxLayout()
        detail_row.setSpacing(14)
        detail_row.addWidget(self.subtitle)
        detail_row.addWidget(self.endpoint)
        detail_row.addStretch(1)
        text_col.addWidget(self.title)
        text_col.addLayout(detail_row)
        self.extra = QLabel("")
        self.extra.setProperty("role", "secondary")
        self.extra.setWordWrap(False)
        text_col.addWidget(self.extra)
        state_box = QHBoxLayout()
        state_box.setSpacing(8)
        self.status_mark = StatusMark("stopped")
        self.status_label = QLabel("Stopped")
        state_box.addWidget(self.status_mark)
        state_box.addWidget(self.status_label)
        layout.addWidget(icon)
        layout.addLayout(text_col, 1)
        layout.addLayout(state_box)
        self.setAccessibleName(f"{title}, stopped, {subtitle}, {endpoint}")

    def set_endpoint_available(self, available: bool) -> None:
        if isinstance(self.endpoint, EndpointLink):
            self.endpoint.set_available(available, interface_name=self.title.text())

    def set_state(self, state: str, *, label: str | None = None) -> None:
        normalized = str(state or "stopped").lower()
        display = label or normalized.replace("_", " ").title()
        self.status_mark.set_status(normalized)
        self.status_label.setText(display)
        role = {
            "running": "success",
            "ready": "success",
            "starting": "info",
            "loading": "info",
            "warning": "warning",
            "failed": "error",
        }.get(normalized, "secondary")
        self.status_label.setProperty("role", role)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.setAccessibleName(
            f"{self.title.text()}, {display}, {self.subtitle.text()}, {self.endpoint.text()}"
        )


def separator() -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.HLine)
    frame.setProperty("separator", True)
    frame.setFixedHeight(1)
    return frame


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "sectionTitle")
    return label
