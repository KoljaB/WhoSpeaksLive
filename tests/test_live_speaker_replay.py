from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.live_speaker_replay import (
    CachedLiveWindowBlock,
    blend_live_speaker_embeddings,
    load_profile_events_jsonl,
    stack_cached_live_window_blocks,
    stack_embedding_matrices,
)


def _block(provider: str, embeddings: list[list[float]], valid: list[bool]) -> CachedLiveWindowBlock:
    return CachedLiveWindowBlock(
        provider=provider,
        video_id="video",
        window_seconds=1.0,
        media_times=np.array([1.0, 1.2], dtype=np.float64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
        raw_rms=np.array([0.1, 0.2], dtype=np.float32),
        sample_rate=16000,
    )


class CachedStackTests(unittest.TestCase):
    def test_dual_window_blend_is_normalized_and_weighted(self) -> None:
        result = blend_live_speaker_embeddings(
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            0.25,
        )
        expected = np.asarray([0.75, 0.25], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_matrix_stack_matches_production_row_policy(self) -> None:
        first = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        second = np.asarray([[0.0, 2.0], [0.0, 1.0]], dtype=np.float32)
        result = stack_embedding_matrices([first, second], [1.0, 0.5])
        expected = np.asarray([0.6, 0.8, 0.0, 0.5], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6, atol=1e-7)

    def test_matches_normalize_weight_concat_policy_and_intersects_validity(self) -> None:
        first = _block("a", [[3.0, 4.0], [1.0, 0.0]], [True, True])
        second = _block("b", [[0.0, 2.0], [0.0, 1.0]], [True, False])
        result = stack_cached_live_window_blocks([first, second], [1.0, 0.5], provider="a+b")

        expected = np.array([0.6, 0.8, 0.0, 0.5], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result.embeddings[0], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(result.valid.tolist(), [True, False])
        np.testing.assert_array_equal(result.embeddings[1], np.zeros(4, dtype=np.float32))

    def test_profile_loader_accepts_production_profile_generation_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.jsonl"
            path.write_text(
                '{"available_at":1.2,"speaker_id":"S1","centroid":[1,0],'
                '"speech_seconds":2.0,"sentence_count":1,"profile_generation":3}\n',
                encoding="utf-8",
            )
            events = load_profile_events_jsonl(path)
        self.assertEqual(events[0].generation, 3)


if __name__ == "__main__":
    unittest.main()
