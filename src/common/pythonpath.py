"""Helpers for subprocess PYTHONPATH construction."""

from __future__ import annotations

import os
from pathlib import Path


def is_site_packages_path(path: Path) -> bool:
    return path.name.lower() in {"site-packages", "dist-packages"}


def build_pythonpath(candidates: tuple[Path | str, ...], existing_pythonpath: str | None = None) -> str:
    entries: list[str] = []

    def add_entry(candidate: Path | str) -> None:
        path = Path(candidate)
        try:
            path = path.resolve()
        except OSError:
            pass
        if not path.exists():
            return
        if is_site_packages_path(path):
            return
        rendered = str(path)
        if rendered not in entries:
            entries.append(rendered)

    for candidate in candidates:
        add_entry(candidate)
    if existing_pythonpath:
        for item in existing_pythonpath.split(os.pathsep):
            if item:
                add_entry(item)
    return os.pathsep.join(entries)
