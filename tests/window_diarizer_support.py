"""Real-construction test support for ``WindowDiarizer`` integration tests."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any
from unittest import mock

import numpy as np

from window.window_diarizer import WindowDiarizer, WindowDiarizerDependencies
from window.window_domain import MediaFiles
from window.window_events import RecordingEventBus


class FakeEmbeddingClient:
    """Side-effect-free embedding dependency used unless a test installs its own."""

    def __init__(self) -> None:
        self.shutdown_count = 0

    def embed_audio(self, _audio: np.ndarray, _sample_rate: int) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def shutdown(self) -> None:
        self.shutdown_count += 1


class TestWindowDiarizer(WindowDiarizer):
    """Production controller initialized normally with fake external clients."""

    __test__ = False

    def _new_embedding_client(
        self,
        _args: Any,
        provider: str | None = None,
    ) -> FakeEmbeddingClient:
        del provider
        return FakeEmbeddingClient()


@lru_cache(maxsize=1)
def _default_args() -> argparse.Namespace:
    from window.youtube_window_diarize_gui import parse_args

    argv = [
        "whospeaks-window",
        "--realtime-preview-engine",
        "off",
        "--no-browser",
    ]
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


def make_window_diarizer(
    *,
    args: argparse.Namespace | None = None,
    bus: RecordingEventBus | None = None,
    audio: np.ndarray | None = None,
    sample_rate: int = 16_000,
    **overrides: Any,
) -> TestWindowDiarizer:
    """Build a complete controller while keeping models, files, and subprocesses fake."""

    if args is None:
        template = _default_args()
        values = getattr(template, "_values", None)
        args = argparse.Namespace(
            **(dict(values) if values is not None else vars(template))
        )
    for name, value in overrides.items():
        setattr(args, name, value)
    samples = (
        np.zeros(sample_rate, dtype=np.float32)
        if audio is None
        else np.asarray(audio, dtype=np.float32).copy()
    )
    media = MediaFiles(
        url="file://window-diarizer-test",
        video_id="window-diarizer-test",
        audio_file=Path("window-diarizer-test.wav"),
        video_file=Path("window-diarizer-test.mp4"),
    )
    dependencies = WindowDiarizerDependencies(
        audio_loader=lambda _path: (samples.copy(), int(sample_rate)),
    )
    return TestWindowDiarizer(
        args,
        media,
        bus or RecordingEventBus(),
        dependencies=dependencies,
    )
