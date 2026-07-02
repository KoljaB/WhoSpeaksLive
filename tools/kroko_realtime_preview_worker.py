"""Compatibility wrapper for the old tools/kroko_realtime_preview_worker.py path."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from whospeaks.workers.kroko_realtime_preview_worker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
