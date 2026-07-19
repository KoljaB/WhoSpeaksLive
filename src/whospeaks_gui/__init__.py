"""Optional PySide6 desktop launcher for WhoSpeaks."""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from .main import main as run

    return run(argv)
