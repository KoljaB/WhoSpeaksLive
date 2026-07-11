from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from speakers.speaker_embedding_cluster import normalize_vector
from window.window_speaker_refinement import (
    DelayedClusteringConfig,
    find_delayed_speaker_splits,
)
from window.window_text import text_content_words
from window.window_validation_replay import make_cached_replay_args, replay_cached_window_diarizer


def _row(
    index: int,
    embedding: list[float],
    *,
    start: float,
    duration: float,
    source: str,
    unknown: float,
) -> dict[str, object]:
    return {
        "index": index,
        "base_payload": {"start": start, "end": start + duration},
        "duration_seconds": duration,
        "assigned_speaker": "S2",
        "embedding": normalize_vector(embedding),
        "assignment_source": source,
        "unknown_probability": unknown,
        "similarities": {"S2": 0.2 if source != "embedding" else 0.9},
    }


class DelayedMultirowClusteringTests(unittest.TestCase):
    def test_repeated_uncertain_voice_splits_from_stable_core(self) -> None:
        rows = [
            _row(i, [1.0, 0.02 * i, 0.0], start=i * 3.0, duration=2.1, source="embedding", unknown=0.05)
            for i in range(5)
        ]
        rows.extend(
            _row(
                10 + i,
                [0.05 * ((i % 3) - 1), 1.0, 0.03 * (i % 2)],
                start=30.0 + i * 12.0,
                duration=1.0,
                source="retro",
                unknown=0.92,
            )
            for i in range(8)
        )

        proposals = find_delayed_speaker_splits(rows, DelayedClusteringConfig())

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].previous_speaker, "S2")
        self.assertEqual(proposals[0].indexes, tuple(range(10, 18)))
        self.assertGreaterEqual(proposals[0].speech_seconds, 8.0)

    def test_short_uncertain_pool_is_not_split(self) -> None:
        rows = [
            _row(i, [1.0, 0.01 * i, 0.0], start=i * 3.0, duration=2.1, source="embedding", unknown=0.05)
            for i in range(5)
        ]
        rows.extend(
            _row(10 + i, [0.0, 1.0, 0.02 * i], start=30.0 + i * 12.0, duration=1.0, source="retro", unknown=0.95)
            for i in range(4)
        )

        self.assertEqual(find_delayed_speaker_splits(rows, DelayedClusteringConfig()), [])

    def test_german_content_words_are_unicode_aware(self) -> None:
        self.assertEqual(
            text_content_words("Schön, Sie kennenzulernen."),
            ["schön", "sie", "kennenzulernen"],
        )


class GermanRegressionCorpusTests(unittest.TestCase):
    def test_6buk09swn9s_delayed_split_recovers_three_speakers(self) -> None:
        video_dir = (
            REPO_ROOT
            / "data"
            / "datasets"
            / "elevenlabs_scribe_27"
            / "videos"
            / "6BuK09sWn9s"
        )
        sentence_path = video_dir / "live_window" / "sentences.jsonl"
        if not sentence_path.is_file():
            self.skipTest("German regression corpus entry is not available.")
        sentences = [
            json.loads(line)
            for line in sentence_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        provider_weights = {
            "espnet_ecapa_wavlm_joint": 1.0,
            "speechbrain_resnet": 0.28,
            "wespeaker_campplus": 0.37,
        }
        matrices = {
            provider: np.load(video_dir / "live_window" / "embeddings" / f"{provider}.npz")["embeddings"]
            for provider in provider_weights
        }
        embeddings = []
        for index in range(len(sentences)):
            blocks = [normalize_vector(matrices[provider][index]) * weight for provider, weight in provider_weights.items()]
            embeddings.append(normalize_vector(np.concatenate(blocks)))

        args = make_cached_replay_args(
            {},
            speaker_refinement=True,
            speaker_refinement_unknown_tentative=True,
            speaker_refinement_unknown_commit=True,
            allow_speaker_reassignment=True,
        )
        replay = replay_cached_window_diarizer(sentences, embeddings, args)
        final_by_index = {int(row["index"]): row for row in replay.final_payloads}
        revised = sorted(
            int(row["index"])
            for row in replay.final_payloads
            if row.get("delayed_multirow_split")
        )

        self.assertEqual({row.get("assigned_speaker") for row in replay.final_payloads}, {"S1", "S2", "S3"})
        self.assertEqual(revised, [6, 8, 9, 15, 48, 49, 53, 56, 57])
        self.assertEqual(final_by_index[41]["assigned_speaker"], "S2")
        self.assertEqual(final_by_index[48]["assigned_speaker"], "S3")


if __name__ == "__main__":
    unittest.main()
