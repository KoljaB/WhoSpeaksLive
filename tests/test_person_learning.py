from __future__ import annotations

import unittest

import numpy as np

from speakers.person_learning import PersonLearningPolicy, build_person_learning_candidate


def _record(index: int, embedding, *, corrected: bool = False) -> dict:
    start = float(index * 10)
    record = {
        "index": index,
        "base_payload": {
            "start": start,
            "end": start + 3.0,
            "speech_audio_ratio": 1.0,
        },
        "embedding": np.asarray(embedding, dtype=np.float32),
        "duration_seconds": 3.0,
        "assigned_speaker": "S1",
        "quality": 1.0,
        "unknown_probability": 0.05,
    }
    if corrected:
        record["unknown_probability"] = 0.99
        record["correction"] = {
            "status": "user_corrected",
            "corrected_speaker": "S1",
        }
    return record


class PersonLearningCandidateTests(unittest.TestCase):
    def test_robust_candidate_discards_an_internal_outlier(self) -> None:
        records = [
            _record(1, [1.0, 0.0]),
            _record(2, [0.99, 0.01]),
            _record(3, [0.98, -0.02]),
            _record(4, [0.97, 0.03]),
            _record(5, [0.45, 0.89]),
        ]
        candidate = build_person_learning_candidate(
            records,
            {"S1": [1.0, 0.0]},
            speaker_id="S1",
            seed_centroid=[1.0, 0.0],
            policy=PersonLearningPolicy(),
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.sentence_count, 4)
        self.assertEqual(candidate.outlier_count, 1)
        self.assertGreater(candidate.cohesion, 0.99)
        self.assertGreater(float(candidate.centroid[0]), 0.99)

    def test_user_correction_bypasses_assignment_confidence_not_quality_gates(self) -> None:
        ambiguous = [_record(index, [0.7, 0.7]) for index in range(1, 4)]
        policy = PersonLearningPolicy(max_unknown_probability=0.55)
        self.assertIsNone(build_person_learning_candidate(
            ambiguous,
            {"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
            speaker_id="S1",
            seed_centroid=[1.0, 0.0],
            policy=policy,
        ))

        corrected = [_record(index, [0.7, 0.7], corrected=True) for index in range(1, 4)]
        candidate = build_person_learning_candidate(
            corrected,
            {"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
            speaker_id="S1",
            seed_centroid=[1.0, 0.0],
            policy=policy,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.user_trusted_indexes, frozenset({1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
