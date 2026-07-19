"""Authoritative design-master-v1 tokens for the Qt presentation layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Colors:
    canvas: str = "#081116"
    surface_1: str = "#0C171C"
    surface_2: str = "#101D22"
    surface_selected: str = "#12343A"
    surface_pressed: str = "#0D292E"
    border: str = "#334247"
    border_strong: str = "#526167"
    text_primary: str = "#F1F5F4"
    text_secondary: str = "#B8C1C3"
    text_muted: str = "#8E9A9E"
    accent: str = "#49C7B1"
    accent_hover: str = "#62D2BE"
    accent_pressed: str = "#35A994"
    focus: str = "#6CB6FF"
    success: str = "#61D487"
    warning: str = "#F4B942"
    error: str = "#FF6262"
    info: str = "#64B5F6"


COLORS = Colors()
SPACING = (2, 4, 8, 12, 16, 24, 32, 40, 48, 64)
CANONICAL_SIZE = (1600, 1000)
MINIMUM_SIZE = (960, 640)
RAIL_WIDTH = 264
MEDIUM_RAIL_WIDTH = 224
COMPACT_RAIL_WIDTH = 72
COMPACT_BREAKPOINT = 1120
CANONICAL_RAIL_BREAKPOINT = 1500
STACK_BREAKPOINT = 1060


def application_style(font_family: str = "Segoe UI") -> str:
    c = COLORS
    check_icon = (Path(__file__).with_name("assets") / "check.svg").resolve().as_posix()
    combo_arrow = (Path(__file__).with_name("assets") / "chevron-down.svg").resolve().as_posix()
    return f"""
    * {{
        color: {c.text_primary};
        font-family: "{font_family}";
        font-size: 17px;
        outline: 0;
    }}
    QMainWindow, QWidget#appRoot, QDialog {{ background: {c.canvas}; }}
    QWidget#formContent, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {c.canvas}; }}
    QWidget#sidebar {{ background: {c.canvas}; border-right: 1px solid {c.border}; }}
    QLabel#logoText {{ font-size: 22px; font-weight: 650; }}
    QLabel#pageTitle {{ font-size: 30px; font-weight: 650; }}
    QLabel#pageSubtitle {{ color: {c.text_secondary}; font-size: 16px; }}
    QLabel[role="sectionTitle"] {{ font-size: 19px; font-weight: 600; }}
    QLabel[role="serviceTitle"] {{ font-size: 18px; font-weight: 600; }}
    QLabel[role="secondary"] {{ color: {c.text_secondary}; }}
    QLabel[role="muted"] {{ color: {c.text_muted}; font-size: 14px; }}
    QLabel[role="success"] {{ color: {c.success}; }}
    QLabel[role="warning"] {{ color: {c.warning}; }}
    QLabel[role="error"] {{ color: {c.error}; }}
    QLabel[role="info"] {{ color: {c.info}; }}
    QLabel[role="keycap"] {{
        color: {c.text_secondary};
        background: {c.surface_1};
        border: 1px solid {c.border_strong};
        border-radius: 6px;
        padding: 10px 14px;
        font-family: "Consolas";
        font-size: 16px;
    }}
    QLabel[role="code"] {{
        color: {c.text_primary};
        background: {c.canvas};
        border: 1px solid {c.border_strong};
        border-radius: 4px;
        padding: 10px 12px;
        font-family: "Consolas";
        font-size: 14px;
    }}
    QLabel#failureHero {{
        color: {c.error};
        border: 4px solid {c.error};
        border-radius: 41px;
        font-size: 48px;
        font-weight: 700;
    }}
    QFrame[group="true"], QWidget[group="true"] {{
        background: {c.surface_1};
        border: 1px solid {c.border};
        border-radius: 7px;
    }}
    QFrame#summaryStrip {{
        background: {c.surface_1};
        border: 1px solid {c.border};
        border-radius: 7px;
    }}
    QFrame#actionBar {{
        background: {c.canvas};
        border-top: 1px solid {c.border};
    }}
    QFrame#actionBar[seamless="true"] {{ border-top: 0; }}
    QFrame[separator="true"] {{ background: {c.border}; border: 0; max-height: 1px; }}
    QLabel[iconTile="true"] {{
        background: {c.surface_1};
        border: 1px solid {c.border_strong};
        border-radius: 5px;
    }}
    QPushButton {{
        min-height: 46px;
        padding: 0 16px;
        background: transparent;
        border: 1px solid {c.border};
        border-radius: 6px;
        font-size: 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {c.surface_2}; border-color: {c.border_strong}; }}
    QPushButton:pressed {{ background: {c.surface_pressed}; }}
    QPushButton:focus {{ border: 2px solid {c.focus}; }}
    QPushButton[compactAction="true"] {{ min-height: 32px; max-height: 32px; padding: 0 8px; }}
    QPushButton:disabled {{ color: {c.text_muted}; border-color: {c.border}; background: transparent; }}
    QPushButton[primary="true"] {{
        min-height: 56px;
        color: {c.canvas};
        background: {c.accent};
        border-color: {c.accent};
        font-size: 17px;
        font-weight: 650;
    }}
    QPushButton[primary="true"]:hover {{ background: {c.accent_hover}; border-color: {c.accent_hover}; }}
    QPushButton[primary="true"]:pressed {{ background: {c.accent_pressed}; border-color: {c.accent_pressed}; }}
    QPushButton[primary="true"]:disabled {{ color: {c.text_muted}; background: {c.surface_2}; border-color: {c.surface_2}; }}
    QPushButton[footerAction="true"] {{
        min-height: 45px;
        max-height: 45px;
        padding: 0 18px;
        background: {c.surface_2};
        border: 1px solid {c.border_strong};
        border-bottom: 2px solid {c.border};
        border-radius: 8px;
        font-size: 16px;
        font-weight: 550;
    }}
    QPushButton[footerAction="true"]:hover {{ background: {c.surface_selected}; border-color: {c.focus}; }}
    QPushButton[footerAction="true"]:pressed {{ background: {c.surface_pressed}; border-bottom-width: 1px; }}
    QPushButton[footerAction="true"][primary="true"] {{
        color: {c.canvas};
        background: {c.accent};
        border-color: {c.accent};
        border-bottom-color: {c.accent_pressed};
        font-weight: 650;
    }}
    QPushButton[footerAction="true"][primary="true"]:hover {{ background: {c.accent_hover}; border-color: {c.accent_hover}; }}
    QPushButton[footerAction="true"][primary="true"]:pressed {{ background: {c.accent_pressed}; border-color: {c.accent_pressed}; }}
    QPushButton[footerAction="true"]:disabled {{ color: {c.text_muted}; background: {c.canvas}; border-color: {c.border}; }}
    QPushButton[footerAction="true"][primary="true"]:disabled {{ color: {c.text_muted}; background: {c.surface_2}; border-color: {c.border}; }}
    QPushButton[danger="true"] {{ color: {c.error}; border-color: {c.error}; }}
    QPushButton[danger="true"]:hover {{ background: rgba(255, 98, 98, 0.12); }}
    QPushButton[danger="true"]:pressed {{ background: rgba(255, 98, 98, 0.20); }}
    QPushButton[dangerFilled="true"] {{ color: #ffffff; background: {c.error}; border-color: {c.error}; }}
    QPushButton[dangerFilled="true"]:hover {{ background: #e84f5c; border-color: #e84f5c; }}
    QPushButton[recoveryPrimary="true"] {{ min-height: 46px; max-height: 46px; color: {c.canvas}; background: {c.accent}; border-color: {c.accent}; font-weight: 600; }}
    QPushButton[recoveryPrimary="true"]:hover {{ background: {c.accent_hover}; border-color: {c.accent_hover}; }}
    QPushButton[linkButton="true"] {{ min-height: 36px; padding: 0; border: 0; background: transparent; }}
    QPushButton[linkButton="true"]:hover {{ color: {c.text_primary}; background: transparent; border: 0; }}
    QPushButton[nav="true"] {{
        min-height: 64px;
        text-align: left;
        padding: 0 16px;
        border: 0;
        border-radius: 6px;
        color: {c.text_secondary};
    }}
    QPushButton[nav="true"]:hover {{ background: {c.surface_2}; color: {c.text_primary}; }}
    QPushButton[nav="true"]:checked {{ background: {c.surface_selected}; color: {c.text_primary}; font-weight: 600; }}
    QPushButton[nav="true"]:pressed {{ background: {c.surface_pressed}; }}
    QLineEdit, QComboBox, QSpinBox {{
        min-height: 46px;
        padding: 0 12px;
        background: {c.surface_1};
        border: 1px solid {c.border};
        border-radius: 6px;
        selection-background-color: {c.surface_selected};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {c.border_strong}; }}
    QComboBox[compactChoice="true"] {{ min-height: 34px; max-height: 34px; padding-left: 0; border: 0; background: transparent; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 2px solid {c.focus}; }}
    QLineEdit[invalid="true"], QComboBox[invalid="true"], QSpinBox[invalid="true"] {{ border: 2px solid {c.error}; }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {c.text_muted}; background: {c.canvas}; }}
    QComboBox::drop-down {{ border: 0; width: 32px; }}
    QComboBox::down-arrow {{ image: url("{combo_arrow}"); width: 14px; height: 14px; }}
    QComboBox QAbstractItemView {{
        background: {c.surface_1};
        border: 1px solid {c.border_strong};
        selection-background-color: {c.surface_selected};
        selection-color: {c.text_primary};
        padding: 4px;
    }}
    QCheckBox {{ spacing: 10px; min-height: 36px; }}
    QCheckBox::indicator {{ width: 19px; height: 19px; border: 1px solid {c.border_strong}; border-radius: 3px; background: {c.surface_1}; }}
    QCheckBox::indicator:checked {{ background: {c.success}; border-color: {c.success}; image: url("{check_icon}"); }}
    QCheckBox::indicator:focus {{ border: 2px solid {c.focus}; }}
    QTableView, QListView, QTreeView, QPlainTextEdit, QTextEdit {{
        background: {c.canvas};
        alternate-background-color: {c.surface_1};
        border: 1px solid {c.border};
        border-radius: 6px;
        selection-background-color: #0E2A3A;
        selection-color: {c.text_primary};
        gridline-color: {c.border};
    }}
    QPlainTextEdit#activityLog {{ font-family: "Consolas"; font-size: 15px; }}
    QTableView::item {{ min-height: 50px; padding: 8px; border-bottom: 1px solid {c.border}; }}
    QHeaderView::section {{ background: {c.surface_2}; color: {c.text_primary}; border: 0; border-right: 1px solid {c.border}; border-bottom: 1px solid {c.border}; padding: 12px; font-weight: 600; }}
    QListWidget#settingsSections {{ background: {c.surface_1}; border: 0; border-right: 1px solid {c.border}; border-radius: 0; }}
    QListWidget#settingsSections::item {{ min-height: 48px; padding: 0 14px; border-radius: 5px; }}
    QListWidget#settingsSections::item:hover {{ background: {c.surface_2}; }}
    QListWidget#settingsSections::item:selected {{ background: {c.surface_selected}; color: {c.text_primary}; }}
    QScrollBar:vertical {{ background: transparent; width: 14px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c.border}; min-height: 32px; border-radius: 5px; margin: 2px; }}
    QScrollBar::handle:vertical:hover {{ background: {c.border_strong}; }}
    QScrollBar:horizontal {{ background: transparent; height: 14px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {c.border}; min-width: 32px; border-radius: 5px; margin: 2px; }}
    QScrollBar::handle:horizontal:hover {{ background: {c.border_strong}; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QProgressBar {{ min-height: 5px; max-height: 5px; background: {c.surface_2}; border: 0; border-radius: 2px; }}
    QProgressBar::chunk {{ background: {c.accent}; border-radius: 2px; }}
    QToolTip {{ color: {c.text_primary}; background: {c.surface_2}; border: 1px solid {c.border_strong}; padding: 6px; }}
    """
