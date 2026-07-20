from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embeddings.live_shifting_window_corpus import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
