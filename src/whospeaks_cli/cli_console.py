"""Terminal presentation and input helpers for the WhoSpeaks classic CLI."""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any

from window.language_config import get_language_config
from window.realtime_preview_backends import preview_language_error


def color_enabled() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ


def style_text(text: str, code: str) -> str:
    if not color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def primary_text(text: str) -> str:
    return style_text(text, "97")


def detail_text(text: str) -> str:
    return style_text(text, "37")


def label_text(text: str) -> str:
    return style_text(text, "96")


def wrap_styled_lines(
    text: str,
    *,
    width: int = 72,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
    style: Any = detail_text,
) -> list[str]:
    follow = initial_indent if subsequent_indent is None else subsequent_indent
    wrapped = textwrap.wrap(
        str(text),
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=follow,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [initial_indent.rstrip()]
    return [style(line) for line in wrapped]


def print_wrapped(
    text: str,
    *,
    width: int = 72,
    initial_indent: str = "",
    subsequent_indent: str | None = None,
    style: Any = detail_text,
) -> None:
    for line in wrap_styled_lines(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        style=style,
    ):
        print(line)


def language_summary(language: str) -> str:
    try:
        config = get_language_config(language)
    except ValueError:
        return str(language)
    preview_support = []
    if config.kroko_code:
        preview_support.append(f"Kroko {config.kroko_code}")
    if preview_language_error("sherpa_onnx", config.code) is None:
        preview_support.append("Nemotron")
    preview = ", ".join(preview_support) or "no realtime preview"
    return f"{config.display_name} ({config.code}, {preview})"


def read_input(prompt: str, default: str = "") -> str:
    try:
        return input(prompt)
    except EOFError:
        print()
        return default
