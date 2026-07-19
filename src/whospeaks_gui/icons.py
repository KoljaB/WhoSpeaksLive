"""Small project-local line icon family rendered from deterministic SVG paths."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .tokens import COLORS


PATHS: dict[str, str] = {
    "home": '<path d="M3 11 12 3l9 8M5 10v11h14V10M9 21v-7h6v7"/>',
    "diagnostics": '<circle cx="12" cy="12" r="9"/><path d="M7 12h3l1.5-4 2.5 8 1.5-4H18"/>',
    "settings": '<path d="M4 7h10M18 7h2M4 12h2M10 12h10M4 17h7M15 17h5"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="17" r="2"/>',
    "activity": '<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
    "refresh": '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 8A7 7 0 0 1 18 7l2 5M18 16a7 7 0 0 1-12 1l-2-5"/>',
    "terminal": '<path d="m5 7 4 5-4 5M11 17h8"/>',
    "play": '<path d="m8 5 11 7-11 7Z"/>',
    "video": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/>',
    "users": '<circle cx="9" cy="9" r="3"/><circle cx="17" cy="10" r="2"/><path d="M3 20c0-4 2-6 6-6s6 2 6 6M15 15c4 0 6 2 6 5"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
    "copy": '<rect x="8" y="8" width="11" height="11" rx="1"/><path d="M16 8V5H5v11h3"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    "download": '<path d="M12 3v12m0 0 5-5m-5 5-5-5M4 19h16"/>',
    "close": '<path d="M6 6l12 12M18 6 6 18"/>',
    "chevron": '<path d="m9 7 5 5-5 5"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
    "warning": '<path d="M12 3 22 21H2L12 3Z"/><path d="M12 9v5M12 17h.01"/>',
    "monitor": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
    "server": '<rect x="3" y="4" width="18" height="7" rx="1"/><rect x="3" y="13" width="18" height="7" rx="1"/><path d="M7 7.5h.01M7 16.5h.01M11 7.5h7M11 16.5h7"/>',
    "message": '<path d="M4 4h16v12H9l-5 4V4Z"/><path d="M9 8h.01M12 8h.01M15 8h.01"/>',
}


@lru_cache(maxsize=128)
def line_icon(name: str, color: str = COLORS.text_primary, size: int = 24) -> QIcon:
    path = PATHS.get(name, PATHS["info"])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'''
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    ratio = 2
    pixmap = QPixmap(QSize(size * ratio, size * ratio))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return QIcon(pixmap)
