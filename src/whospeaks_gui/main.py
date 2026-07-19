"""Entry point and deterministic screenshot mode for the PySide6 launcher."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from whospeaks_cli.launcher_controller import create_launcher_controller
from whospeaks_cli.profiles import ProfileLoadError

from .demo import DEMO_STATES, DemoLauncherController
from .tokens import CANONICAL_SIZE, application_style
from .widgets import StatusMark
from .window import LauncherWindow


def _configure_fonts(app: QApplication) -> str:
    """Load the native Windows UI face explicitly for offscreen determinism."""

    if os.name == "nt":
        for path in (
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\consolab.ttf",
        ):
            if Path(path).is_file():
                QFontDatabase.addApplicationFont(path)
    families = set(QFontDatabase.families())
    preferred = next(
        (name for name in ("Segoe UI", "Inter", "Noto Sans", "DejaVu Sans") if name in families),
        app.font().family(),
    )
    app.setFont(QFont(preferred, 11))
    return preferred


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whospeaks-gui", description="WhoSpeaks desktop launcher")
    parser.add_argument("--demo-state", choices=DEMO_STATES, default="", help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", type=Path, default=None, help="Capture the deterministic client area and exit.")
    parser.add_argument("--width", type=int, default=CANONICAL_SIZE[0])
    parser.add_argument("--height", type=int, default=CANONICAL_SIZE[1])
    parser.add_argument("--no-auto-check", action="store_true", help="Do not run the initial quick diagnostic check.")
    parser.add_argument("--reduced-motion", action="store_true", help="Disable nonessential transitions and animated activity marks.")
    parser.add_argument(
        "--motion-phase",
        type=int,
        choices=range(12),
        default=None,
        help="Set a deterministic 0-11 activity-frame phase for design review screenshots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.screenshot is not None and not args.demo_state:
        raise SystemExit("--screenshot requires --demo-state so capture never touches real services or configuration")
    if args.screenshot is not None and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["whospeaks-gui"])
    app.setApplicationName("WhoSpeaks")
    app.setOrganizationName("WhoSpeaks")
    font_family = _configure_fonts(app)
    app.setStyleSheet(application_style(font_family))
    try:
        controller = DemoLauncherController(args.demo_state) if args.demo_state else create_launcher_controller()
    except ProfileLoadError as exc:
        QMessageBox.critical(
            None,
            "WhoSpeaks configuration needs attention",
            str(exc),
        )
        return 2
    window = LauncherWindow(
        controller,
        auto_check=not args.no_auto_check,
        reduced_motion=args.reduced_motion or bool(args.screenshot),
    )
    if args.screenshot is not None:
        window.resize(args.width, args.height)
    else:
        screen = app.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            window.resize(min(1440, int(available.width() * 0.88)), min(900, int(available.height() * 0.88)))
    window.show()
    if args.demo_state == "stop_confirmation":
        window.show_stop_confirmation_review()
    if args.motion_phase is not None:
        phase = int(args.motion_phase) % 12
        for mark in window.findChildren(StatusMark):
            mark.set_phase(phase * 30)
    if args.screenshot is not None:
        destination = args.screenshot.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            app.processEvents()
            pixmap = window.grab()
            if not pixmap.save(str(destination)):
                print(f"Could not save screenshot: {destination}", file=sys.stderr)
                app.exit(2)
                return
            window.bridge.close()
            window.hide()
            app.exit(0)

        QTimer.singleShot(150, capture)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
