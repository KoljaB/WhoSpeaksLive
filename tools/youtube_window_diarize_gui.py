"""Compatibility wrapper for the old tools/youtube_window_diarize_gui.py path."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from whospeaks.window.youtube_gui import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
