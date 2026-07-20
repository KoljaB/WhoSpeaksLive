from __future__ import annotations

from types import SimpleNamespace
import argparse
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_diarizer_live_scoring import WindowLiveScoringMixin
from window.window_cli_live_speaker import add_preview_live_speaker_arguments


class _Memory:
    @staticmethod
    def profile_count() -> int:
        return 1


class _Bus:
    def emit(self, _event: str, _payload: object) -> None:
        pass


class _Harness(WindowLiveScoringMixin):
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            realtime_preview_diarize_min_audio_seconds=0.1,
            min_embed_seconds=0.1,
        )
        self.sample_rate = 10
        self.live_memory = _Memory()
        self.bus = _Bus()
        self.embed_suffixes: list[str] = []
        self.core_arguments: dict[str, object] = {}

    @staticmethod
    def _live_speaker_assignment_enabled() -> bool:
        return True

    def _embed_live_audio_chunk(
        self, _audio: np.ndarray, _sample_rate: int, suffix: str
    ) -> np.ndarray:
        self.embed_suffixes.append(suffix)
        if suffix == ".live.short.wav":
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _record_live_speaker_embedding_latency(_latency: float) -> None:
        pass

    @staticmethod
    def playback_time() -> float:
        return 3.0

    def _shared_live_speaker_step(self, **kwargs: object) -> SimpleNamespace:
        self.core_arguments = kwargs
        return SimpleNamespace(
            raw_probabilities={"speaker1": 0.8, "unknown": 0.2},
            probabilities={"speaker1": 0.8, "unknown": 0.2},
            visible_speaker="S1",
            similarities={"S1": 0.8},
            action="acquire",
            reason="known_speaker",
        )

    @staticmethod
    def _ensure_speaker_metadata(_speaker: str | None) -> None:
        pass

    @staticmethod
    def _speaker_info_for_payload(_speaker: str | None) -> dict[str, object]:
        return {}


class ProductionDualWindowTests(unittest.TestCase):
    def test_locked_champion_cli_values_parse(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)
        args = parser.parse_args([
            "--live-speaker-probe-window-seconds", "0.8",
            "--live-speaker-probe-context-window-seconds", "2.8",
            "--live-speaker-probe-context-weight", "0.25",
            "--realtime-preview-diarize-min-similarity", "0.35",
            "--realtime-preview-diarize-min-margin", "0.08",
            "--live-speaker-probe-clear-silence-count", "2",
        ])
        self.assertEqual(args.live_speaker_probe_window_seconds, 0.8)
        self.assertEqual(args.live_speaker_probe_context_window_seconds, 2.8)
        self.assertEqual(args.live_speaker_probe_context_weight, 0.25)
        self.assertEqual(args.realtime_preview_diarize_min_similarity, 0.35)
        self.assertEqual(args.realtime_preview_diarize_min_margin, 0.08)
        self.assertEqual(args.live_speaker_probe_clear_silence_count, 2)

    def test_scoring_blends_both_live_windows_before_shared_core(self) -> None:
        harness = _Harness()
        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32),
            0.8,
            context_audio=np.ones(28, dtype=np.float32),
            context_duration_seconds=2.8,
            context_weight=0.25,
        )

        expected = np.asarray([0.75, 0.25], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            harness.core_arguments["embedding"], expected, rtol=1e-6, atol=1e-7
        )
        self.assertEqual(harness.embed_suffixes, [".live.short.wav", ".live.context.wav"])
        self.assertEqual(harness.core_arguments["duration_seconds"], 2.8)
        self.assertEqual(payload["live_speaker_context_weight"], 0.25)
        self.assertEqual(payload["live_speaker_short_window_seconds"], 0.8)
        self.assertEqual(payload["live_speaker_context_window_seconds"], 2.8)


if __name__ == "__main__":
    unittest.main()
