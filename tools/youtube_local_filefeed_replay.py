"""Compatibility wrapper for the old tools/youtube_local_filefeed_replay.py path."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from whospeaks.replay.youtube_local_filefeed_replay import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
