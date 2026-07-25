from __future__ import annotations

import unittest

import numpy as np

from window.sentence_speaker_handoff import (
    HandoffConfig,
    LiveEmbeddingEvidence,
    WordSpeakerMargin,
    estimate_coarse_boundary,
    nominate_stable_handoff,
    select_context_handoff,
    select_word_handoff,
    split_sentence_part,
)
from window.window_domain import SentencePart


def evidence(
    start: float,
    end: float,
    embedding: tuple[float, ...],
    speaker: str | None,
    *,
    provider: str = "synthetic",
    voiced_seconds: float | None = 0.5,
) -> LiveEmbeddingEvidence:
    return LiveEmbeddingEvidence(
        window_start=start,
        window_end=end,
        short_embedding=np.asarray(embedding, dtype=np.float32),
        visible_speaker=speaker,
        similarities={"A": 0.5, "B": 0.4},
        profile_generations={"A": 2, "B": 3},
        provider=provider,
        voiced_seconds=voiced_seconds,
    )


class LiveHandoffNominationTests(unittest.TestCase):
    def test_live_evidence_normalizes_and_freezes_short_embedding(self) -> None:
        item = evidence(1.0, 1.7, (3.0, 4.0), " A ", voiced_seconds=2.0)

        np.testing.assert_allclose(item.short_embedding, [0.6, 0.8], atol=1e-6)
        self.assertFalse(item.short_embedding.flags.writeable)
        self.assertEqual(item.visible_speaker, "A")
        self.assertAlmostEqual(item.voiced_seconds or 0.0, 0.7)
        self.assertEqual(item.profile_generations["B"], 3)

    def test_nominates_one_a_to_b_run_while_ignoring_unknown_gap(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.20, 0.90, (0.9, 0.1), "A"),
            evidence(0.40, 1.10, (1.0, 1.0), "UNKNOWN"),
            evidence(0.60, 1.30, (0.1, 0.9), "B"),
            evidence(0.80, 1.50, (0.0, 1.0), "B"),
        ]
        words = [
            {"start": 0.05, "end": 0.25},
            {"start": 0.30, "end": 0.60},
            {"start": 0.80, "end": 1.00},
            {"start": 1.05, "end": 1.30},
        ]

        result = nominate_stable_handoff(
            probes,
            0.0,
            2.0,
            words,
            {"A": np.asarray([1.0, 0.0]), "B": np.asarray([0.0, 1.0])},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((result.speaker_a, result.speaker_b), ("A", "B"))
        self.assertEqual((result.left_probe_count, result.right_probe_count), (2, 2))
        self.assertIsNotNone(result.coarse_boundary)
        self.assertAlmostEqual(result.coarse_boundary_time or 0.0, 0.75, places=6)
        self.assertEqual(result.suggested_word_cut_index, 2)
        self.assertAlmostEqual(result.suggested_word_boundary_time or 0.0, 0.70)

    def test_rejects_voiced_unknown_outside_transition_without_pure_support(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0), "A"),
            evidence(0.40, 1.10, (1.0, 1.0), "UNKNOWN"),
            evidence(0.60, 1.30, (1.0, 0.0), "A"),
            evidence(0.80, 1.50, (0.0, 1.0), "B"),
            evidence(1.00, 1.70, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0]),
                    "B": np.asarray([0.0, 1.0]),
                },
            )
        )

    def test_accepts_pairwise_pure_unknown_runs_within_a_and_b_sides(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.20, 0.90, (0.8, 0.1), "UNKNOWN"),
            evidence(0.40, 1.10, (1.0, 0.0), "A"),
            evidence(0.60, 1.30, (0.0, 1.0), "B"),
            evidence(0.80, 1.50, (0.1, 0.8), "UNKNOWN"),
            evidence(1.00, 1.70, (0.0, 1.0), "B"),
        ]

        self.assertIsNotNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0]),
                    "B": np.asarray([0.0, 1.0]),
                },
            )
        )

    def test_accepts_short_bracketed_run_with_lower_absolute_but_strong_margin(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0, 0.0), "A"),
            evidence(0.20, 0.90, (0.3, 0.0, 0.954), "UNKNOWN"),
            evidence(0.40, 1.10, (1.0, 0.0, 0.0), "A"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]

        self.assertIsNotNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0, 0.0]),
                    "B": np.asarray([0.0, 1.0, 0.0]),
                },
            )
        )

    def test_accepts_short_sentence_edge_run_with_stronger_evidence(self) -> None:
        probes = [
            evidence(0.00, 0.70, (0.4, 0.0, 0.916), "UNKNOWN"),
            evidence(0.20, 0.90, (1.0, 0.0, 0.0), "A"),
            evidence(0.40, 1.10, (1.0, 0.0, 0.0), "A"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]

        self.assertIsNotNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0, 0.0]),
                    "B": np.asarray([0.0, 1.0, 0.0]),
                },
            )
        )

    def test_rejects_bracketed_unknown_third_voice_with_only_small_margin(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0, 0.0), "A"),
            evidence(0.20, 0.90, (-0.1, -0.2, 0.97), "UNKNOWN"),
            evidence(0.40, 1.10, (1.0, 0.0, 0.0), "A"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]

        self.assertIsNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0, 0.0]),
                    "B": np.asarray([0.0, 1.0, 0.0]),
                },
            )
        )

    def test_rejects_wide_unknown_transition_band(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0), "A"),
            evidence(0.40, 1.10, (0.8, 0.2), "UNKNOWN"),
            evidence(0.60, 1.30, (0.5, 0.5), "UNKNOWN"),
            evidence(0.80, 1.50, (0.2, 0.8), "UNKNOWN"),
            evidence(1.00, 1.70, (0.5, 0.5), "UNKNOWN"),
            evidence(1.20, 1.90, (0.0, 1.0), "B"),
            evidence(1.40, 2.10, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(nominate_stable_handoff(probes, 0.0, 2.2))

    def test_rejects_long_transition_even_with_strict_side_geometry(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0), "A"),
            evidence(0.40, 1.10, (0.8, 0.1), "UNKNOWN"),
            evidence(0.60, 1.30, (0.8, 0.1), "UNKNOWN"),
            evidence(0.80, 1.50, (0.8, 0.1), "UNKNOWN"),
            evidence(1.00, 1.70, (0.8, 0.1), "UNKNOWN"),
            evidence(1.20, 1.90, (0.0, 1.0), "B"),
            evidence(1.40, 2.10, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.2,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0]),
                    "B": np.asarray([0.0, 1.0]),
                },
            )
        )

    def test_transition_unknown_can_be_ambiguous_between_a_and_b(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0, 0.0), "A"),
            evidence(0.40, 1.10, (0.70, 0.68, 0.0), "UNKNOWN"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]

        self.assertIsNotNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                profile_anchors={
                    "A": np.asarray([1.0, 0.0, 0.0]),
                    "B": np.asarray([0.0, 1.0, 0.0]),
                },
            )
        )

    def test_transition_unknown_must_match_a_b_union_not_third_voice(self) -> None:
        third_voice = [
            evidence(0.00, 0.70, (1.0, 0.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0, 0.0), "A"),
            evidence(0.40, 1.10, (0.40, 0.40, 0.82), "UNKNOWN"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]
        unmatched_voice = [
            evidence(0.00, 0.70, (1.0, 0.0, 0.0), "A"),
            evidence(0.20, 0.90, (1.0, 0.0, 0.0), "A"),
            evidence(0.40, 1.10, (-0.10, -0.20, 0.97), "UNKNOWN"),
            evidence(0.60, 1.30, (0.0, 1.0, 0.0), "B"),
            evidence(0.80, 1.50, (0.0, 1.0, 0.0), "B"),
        ]
        anchors = {
            "A": np.asarray([1.0, 0.0, 0.0]),
            "B": np.asarray([0.0, 1.0, 0.0]),
            "C": np.asarray([0.0, 0.0, 1.0]),
        }

        self.assertIsNone(
            nominate_stable_handoff(
                third_voice,
                0.0,
                2.0,
                profile_anchors=anchors,
            )
        )
        self.assertIsNone(
            nominate_stable_handoff(
                unmatched_voice,
                0.0,
                2.0,
                profile_anchors={
                    "A": anchors["A"],
                    "B": anchors["B"],
                },
            )
        )

    def test_rejects_missing_temporal_coverage_across_transition(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.40, 1.10, (1.0, 0.0), "A"),
            evidence(4.00, 4.70, (0.0, 1.0), "B"),
            evidence(4.40, 5.10, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(nominate_stable_handoff(probes, 0.0, 5.5))

    def test_rejects_missing_temporal_coverage_inside_a_side(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(3.00, 3.70, (1.0, 0.0), "A"),
            evidence(3.40, 4.10, (0.0, 1.0), "B"),
            evidence(3.80, 4.50, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(nominate_stable_handoff(probes, 0.0, 8.0))

    def test_requires_evidence_near_first_and_last_words(self) -> None:
        probes = [
            evidence(0.00, 0.70, (1.0, 0.0), "A"),
            evidence(0.40, 1.10, (1.0, 0.0), "A"),
            evidence(0.80, 1.50, (0.0, 1.0), "B"),
            evidence(1.20, 1.90, (0.0, 1.0), "B"),
        ]
        words = [
            {"start": 0.05, "end": 0.25},
            {"start": 0.40, "end": 0.70},
            {"start": 7.40, "end": 7.80},
            {"start": 7.85, "end": 7.95},
        ]

        self.assertIsNone(
            nominate_stable_handoff(probes, 0.0, 8.0, word_times=words)
        )

    def test_rejects_third_speaker_and_a_b_a_flicker(self) -> None:
        third_speaker = [
            evidence(0.0, 0.7, (1.0, 0.0), "A"),
            evidence(0.2, 0.9, (1.0, 0.0), "A"),
            evidence(0.4, 1.1, (1.0, 1.0), "C"),
            evidence(0.6, 1.3, (0.0, 1.0), "B"),
            evidence(0.8, 1.5, (0.0, 1.0), "B"),
        ]
        flicker = [
            evidence(0.0, 0.7, (1.0, 0.0), "A"),
            evidence(0.2, 0.9, (1.0, 0.0), "A"),
            evidence(0.4, 1.1, (0.0, 1.0), "B"),
            evidence(0.6, 1.3, (1.0, 0.0), "A"),
            evidence(0.8, 1.5, (0.0, 1.0), "B"),
            evidence(1.0, 1.7, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(nominate_stable_handoff(third_speaker, 0.0, 2.0))
        self.assertIsNone(
            nominate_stable_handoff(
                flicker,
                0.0,
                2.0,
                config=HandoffConfig(min_probes_per_side=1),
            )
        )

    def test_requires_configured_probe_and_voiced_support_on_both_sides(self) -> None:
        probes = [
            evidence(0.0, 0.7, (1.0, 0.0), "A", voiced_seconds=0.2),
            evidence(0.2, 0.9, (1.0, 0.0), "A", voiced_seconds=0.2),
            evidence(0.4, 1.1, (0.0, 1.0), "B", voiced_seconds=0.2),
            evidence(0.6, 1.3, (0.0, 1.0), "B", voiced_seconds=0.2),
        ]

        self.assertIsNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                config=HandoffConfig(min_live_voiced_seconds_per_side=0.5),
            )
        )
        self.assertIsNotNone(
            nominate_stable_handoff(
                probes,
                0.0,
                2.0,
                config=HandoffConfig(
                    min_probes_per_side=2,
                    min_live_voiced_seconds_per_side=0.4,
                ),
            )
        )

    def test_coarse_boundary_interpolates_margin_then_removes_half_window(self) -> None:
        probes = [
            evidence(9.3, 10.0, (1.0, 0.0), "A"),
            evidence(9.5, 10.2, (0.0, 1.0), "B"),
        ]

        result = estimate_coarse_boundary(
            probes,
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.trailing_crossing_time, 10.1)
        self.assertAlmostEqual(result.trailing_window_correction, 0.35)
        self.assertAlmostEqual(result.boundary_time, 9.75)

    def test_coarse_boundary_rejects_raw_margin_flicker(self) -> None:
        probes = [
            evidence(0.0, 0.7, (1.0, 0.0), "A"),
            evidence(0.2, 0.9, (0.0, 1.0), "B"),
            evidence(0.4, 1.1, (1.0, 0.0), "A"),
            evidence(0.6, 1.3, (0.0, 1.0), "B"),
        ]

        self.assertIsNone(
            estimate_coarse_boundary(
                probes,
                np.asarray([1.0, 0.0]),
                np.asarray([0.0, 1.0]),
            )
        )

    def test_nomination_localizes_crossing_between_last_a_and_first_b(self) -> None:
        probes = [
            # An earlier low-energy probe can have a misleading raw sign while
            # remaining unassigned. It must not poison a later stable A->B run.
            evidence(0.0, 0.7, (0.0, 1.0), "UNKNOWN", voiced_seconds=0.05),
            evidence(0.4, 1.1, (1.0, 0.0), "A"),
            evidence(0.8, 1.5, (1.0, 0.0), "A"),
            evidence(1.2, 1.9, (0.0, 1.0), "B"),
            evidence(1.6, 2.3, (0.0, 1.0), "B"),
        ]

        result = nominate_stable_handoff(
            probes,
            0.0,
            2.5,
            profile_anchors={
                "A": np.asarray([1.0, 0.0]),
                "B": np.asarray([0.0, 1.0]),
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNotNone(result.coarse_boundary_time)


class ExactWordCutTests(unittest.TestCase):
    def test_accepts_exact_forward_cut_and_preserves_absolute_word_index(self) -> None:
        margins = [
            WordSpeakerMargin(10, -0.8),
            WordSpeakerMargin(11, -0.5),
            WordSpeakerMargin(12, 0.2),
            WordSpeakerMargin(13, 0.7),
            WordSpeakerMargin(14, 0.8),
        ]

        result = select_word_handoff(margins)

        self.assertTrue(result.accepted)
        self.assertEqual(result.cut_index, 12)
        self.assertEqual(result.reason, "accepted")
        self.assertGreaterEqual(result.gain_over_no_split, 0.25)
        self.assertGreaterEqual(result.gain_over_reverse, 0.25)
        self.assertAlmostEqual(result.gain_over_runner_up_cut, 0.2)

    def test_short_first_b_word_can_select_the_exact_reference_cut(self) -> None:
        result = select_word_handoff(
            [-0.604, -0.570, 0.102, 0.356, 0.418, 0.615]
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.cut_index, 2)
        self.assertAlmostEqual(result.gain_over_runner_up_cut, 0.102)

    def test_rejects_reverse_handoff_and_single_speaker_sequence(self) -> None:
        reverse = select_word_handoff([0.8, 0.7, -0.7, -0.8])
        all_b = select_word_handoff([0.4, 0.5, 0.6, 0.7])

        self.assertFalse(reverse.accepted)
        self.assertLess(reverse.gain_over_reverse, 0.0)
        self.assertFalse(all_b.accepted)
        self.assertEqual(all_b.reason, "forward_does_not_beat_no_split")

    def test_rejects_runner_up_cut_when_one_word_has_ambiguous_margin(self) -> None:
        result = select_word_handoff([-0.8, -0.6, 0.02, 0.7, 0.8])

        self.assertFalse(result.accepted)
        self.assertEqual(result.candidate_cut_index, 2)
        self.assertEqual(result.reason, "ambiguous_forward_cut")
        self.assertAlmostEqual(result.gain_over_runner_up_cut, 0.02)

    def test_rejects_when_configured_segment_support_is_missing(self) -> None:
        margins = [
            WordSpeakerMargin(0, -0.8, 0.1),
            WordSpeakerMargin(1, -0.7, 0.1),
            WordSpeakerMargin(2, 0.7, 0.1),
            WordSpeakerMargin(3, 0.8, 0.1),
        ]

        result = select_word_handoff(
            margins,
            HandoffConfig(min_word_support_per_segment=0.5),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "insufficient_segment_support")


class ContextVerifierTests(unittest.TestCase):
    def test_accepts_strong_a_left_b_right_context(self) -> None:
        result = select_context_handoff(
            {"A": 0.695, "B": 0.074, "C": 0.10},
            {"A": 0.040, "B": 0.700, "C": 0.12},
            "A",
            "B",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertAlmostEqual(result.left_pair_margin, 0.621)
        self.assertAlmostEqual(result.right_pair_margin, 0.660)
        self.assertAlmostEqual(result.separation, 1.281)

    def test_rejects_weak_or_third_speaker_context(self) -> None:
        weak = select_context_handoff(
            {"A": 0.20, "B": 0.05},
            {"A": 0.02, "B": 0.21},
            "A",
            "B",
        )
        third = select_context_handoff(
            {"A": 0.70, "B": 0.10, "C": 0.68},
            {"A": 0.05, "B": 0.72, "C": 0.20},
            "A",
            "B",
        )

        self.assertFalse(weak.accepted)
        self.assertEqual(weak.reason, "weak_context_similarity")
        self.assertFalse(third.accepted)
        self.assertEqual(third.reason, "context_matches_third_speaker")


class SentencePartSplitTests(unittest.TestCase):
    def make_sentence(self) -> SentencePart:
        words = [
            {"text": "The", "start": 0.10, "end": 0.30, "duration": 0.20},
            {"text": "answer", "start": 0.35, "end": 0.70, "duration": 0.35},
            {"text": "continues", "start": 0.75, "end": 1.10, "duration": 0.35},
            {"text": "now.", "start": 1.15, "end": 1.40, "duration": 0.25},
        ]
        return SentencePart(
            text="The answer continues now.",
            start=0.0,
            end=1.5,
            next_left=1.5,
            spoken_word_seconds=1.15,
            speech_audio_ratio=1.15 / 1.5,
            words=words,
            first_word_start=0.10,
            last_word_end=1.40,
            next_word_start=2.0,
            gap_to_next_word_seconds=0.6,
            boundary_strategy="asr_gap",
            asr_review={"status": "clean"},
        )

    def test_splits_only_an_accepted_cut_and_keeps_semantic_group(self) -> None:
        sentence = self.make_sentence()
        selection = select_word_handoff([-0.8, -0.7, 0.6, 0.8])
        self.assertTrue(selection.accepted)

        split = split_sentence_part(
            sentence,
            selection,
            speaker_a="A",
            speaker_b="B",
            semantic_group_id="sentence-42",
        )
        left, right = split

        self.assertEqual((left.text, right.text), ("The answer", "continues now."))
        self.assertAlmostEqual(split.boundary_time, 0.725)
        self.assertAlmostEqual(left.end, right.start)
        self.assertEqual(left.semantic_sentence_id, "sentence-42")
        self.assertEqual(right.semantic_sentence_id, "sentence-42")
        self.assertEqual((left.semantic_sentence_part, right.semantic_sentence_part), (0, 1))
        self.assertEqual(left.semantic_sentence_part_count, 2)
        self.assertEqual(left.speaker_handoff["role"], "from")
        self.assertEqual(right.speaker_handoff["role"], "to")
        self.assertEqual(right.next_word_start, 2.0)
        self.assertEqual(right.gap_to_next_word_seconds, 0.6)

    def test_refuses_to_apply_a_rejected_candidate_cut(self) -> None:
        rejected = select_word_handoff([0.4, 0.5, 0.6, 0.7])

        with self.assertRaisesRegex(ValueError, "accepted"):
            split_sentence_part(self.make_sentence(), rejected)


if __name__ == "__main__":
    unittest.main()
