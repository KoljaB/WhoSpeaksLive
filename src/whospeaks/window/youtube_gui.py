"""Console entrypoint alias for the growing-window GUI."""

from __future__ import annotations

from whospeaks.window.youtube_window_diarize_gui import main


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
