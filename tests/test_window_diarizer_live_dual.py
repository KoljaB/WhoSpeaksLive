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
from window.live_speech_gate import live_silero_gate_parameters
from window.window_cli_live_speaker import add_preview_live_speaker_arguments


class _Memory:
    @staticmethod
    def profile_count() -> int:
        return 1


class _EmptyMemory:
    @staticmethod
    def profile_count() -> int:
        return 0

    @staticmethod
    def export_profiles() -> list[dict[str, object]]:
        return []


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
    def _record_live_speaker_embedding_latency(
        _latency: float, **_correlation: object
    ) -> None:
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
    def _ensure_speaker_metadata(_speaker: str | None, source: str = "detected") -> None:
        pass

    @staticmethod
    def _speaker_info_for_payload(_speaker: str | None) -> dict[str, object]:
        return {}


class ProductionDualWindowTests(unittest.TestCase):
    def test_bayes_provisional_alias_is_published_as_final_speaker(self) -> None:
        aliases = {"S2": "provisional_1"}
        self.assertEqual(
            WindowLiveScoringMixin._public_live_speaker_label("provisional_1", aliases),
            "S2",
        )
        self.assertEqual(
            WindowLiveScoringMixin._public_live_speaker_values(
                {"unknown": 0.1, "provisional_1": 0.9},
                aliases,
                probability_keys=True,
            ),
            {"unknown": 0.1, "speaker2": 0.9},
        )

    def test_live_payload_reconciles_provisional_card_with_final_speaker(self) -> None:
        harness = _Harness()
        harness.args.live_speaker_tracker = "bayes"
        harness._shared_live_speaker_step = lambda **_kwargs: SimpleNamespace(
            raw_probabilities={"unknown": 0.1, "provisional_1": 0.9},
            probabilities={"unknown": 0.1, "provisional_1": 0.9},
            visible_speaker="provisional_1",
            similarities={"provisional_1": 0.8},
            action="hold",
            reason="known_speaker",
            diagnostics={"profile_aliases": {"S2": "provisional_1"}},
        )

        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32),
            0.8,
        )

        self.assertEqual(payload["assigned_speaker"], "S2")
        self.assertEqual(payload["internal_speaker_id"], "provisional_1")
        self.assertEqual(payload["replaces_speaker_id"], "provisional_1")
        self.assertEqual(payload["probabilities"], {"unknown": 0.1, "speaker2": 0.9})
        self.assertEqual(payload["similarities"], {"S2": 0.8})

    def test_locked_champion_cli_values_parse(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)
        args = parser.parse_args([
            "--live-speaker-probe-window-seconds", "0.8",
            "--live-speaker-probe-context-window-seconds", "2.8",
            "--live-speaker-probe-context-weight", "0.25",
            "--live-speaker-probe-release-interval-seconds", "0.2",
            "--realtime-preview-diarize-min-similarity", "0.35",
            "--realtime-preview-diarize-min-margin", "0.08",
            "--live-speaker-probe-clear-silence-count", "2",
        ])
        self.assertEqual(args.live_speaker_probe_window_seconds, 0.8)
        self.assertEqual(args.live_speaker_probe_release_interval_seconds, 0.2)
        self.assertEqual(args.live_speaker_probe_context_window_seconds, 2.8)
        self.assertEqual(args.live_speaker_probe_context_weight, 0.25)
        self.assertEqual(args.realtime_preview_diarize_min_similarity, 0.35)
        self.assertEqual(args.realtime_preview_diarize_min_margin, 0.08)
        self.assertEqual(args.live_speaker_probe_clear_silence_count, 2)

    def test_versioned_profile_contradiction_tracklet_preset_parses(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)

        args = parser.parse_args([
            "--live-speaker-open-set-tracklets",
            "--live-speaker-open-set-tracklet-preset",
            "short_history_hybrid_v2_profile_contradiction",
        ])

        self.assertTrue(args.live_speaker_open_set_tracklets)
        self.assertEqual(
            "short_history_hybrid_v2_profile_contradiction",
            args.live_speaker_open_set_tracklet_preset,
        )

    def test_bayes_provisional_cli_and_config_parse(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)
        args = parser.parse_args([
            "--live-speaker-tracker", "bayes",
            "--live-speaker-bayes-provisional-profiles",
            "--live-speaker-bayes-provisional-creation-count", "1",
            "--live-speaker-bayes-provisional-later-creation-count", "3",
            "--live-speaker-bayes-provisional-creation-similarity-ceiling", "0.05",
            "--live-speaker-bayes-provisional-boundary-creation-similarity-ceiling", "0.131",
            "--live-speaker-bayes-provisional-boundary-continuity", "0.06",
            "--live-speaker-bayes-provisional-max-finalized-profiles", "4",
            "--live-speaker-bayes-provisional-merge-min-similarity", "0.05",
            "--live-speaker-bayes-provisional-update-alpha", "0.65",
            "--live-speaker-bayes-provisional-update-continuity", "0.1",
            "--live-speaker-bayes-provisional-update-history-size", "3",
            "--live-speaker-bayes-provisional-max-active-count", "4",
            "--live-speaker-bayes-provisional-pool-overflow-update-alpha", "0.5",
            "--live-speaker-bayes-incumbent-continuity", "0.3",
            "--live-speaker-bayes-incumbent-continuity-history-size", "6",
            "--live-speaker-bayes-incumbent-continuity-update-on-hold",
            "--live-speaker-bayes-boundary-short-only-continuity", "0.0625",
            "--live-speaker-bayes-boundary-residual-incumbent-alpha", "0.05",
            "--live-speaker-bayes-short-long-crossover-min-margin", "0.06",
            "--live-speaker-bayes-short-long-crossover-min-similarity", "0.2",
            "--live-speaker-bayes-short-long-crossover-count", "1",
            "--live-speaker-bayes-short-long-differential-candidate-gain", "-0.15",
            "--live-speaker-bayes-short-long-differential-incumbent-loss", "0.15",
        ])
        harness = _Harness()
        harness.args = args
        algorithm = harness._shared_live_speaker_algorithm()
        self.assertTrue(algorithm.config.enable_provisional_profiles)
        self.assertEqual(algorithm.config.provisional_creation_count, 1)
        self.assertEqual(algorithm.config.provisional_later_creation_count, 3)
        self.assertEqual(algorithm.config.provisional_creation_max_finalized_profiles, 4)
        self.assertEqual(
            algorithm.config.provisional_boundary_creation_similarity_ceiling,
            0.131,
        )
        self.assertEqual(
            algorithm.config.provisional_boundary_continuity_max_similarity,
            0.06,
        )
        self.assertEqual(algorithm.config.provisional_update_alpha, 0.65)
        self.assertEqual(algorithm.config.provisional_update_continuity_min_similarity, 0.1)
        self.assertEqual(algorithm.config.provisional_update_history_size, 3)
        self.assertEqual(algorithm.config.provisional_max_active_count, 4)
        self.assertEqual(algorithm.config.provisional_pool_overflow_update_alpha, 0.5)
        self.assertEqual(algorithm.config.incumbent_continuity_min_similarity, 0.3)
        self.assertEqual(algorithm.config.incumbent_continuity_history_size, 6)
        self.assertTrue(algorithm.config.incumbent_continuity_update_on_hold)
        self.assertEqual(algorithm.config.boundary_short_only_max_continuity, 0.0625)
        self.assertEqual(algorithm.config.boundary_residual_incumbent_alpha, 0.05)
        self.assertEqual(algorithm.config.short_long_crossover_min_margin, 0.06)
        self.assertEqual(algorithm.config.short_long_crossover_min_similarity, 0.2)
        self.assertEqual(algorithm.config.short_long_crossover_count, 1)
        self.assertEqual(algorithm.config.short_long_differential_candidate_gain, -0.15)
        self.assertEqual(algorithm.config.short_long_differential_incumbent_loss, 0.15)

    def test_split_silero_live_gate_values_parse(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)
        args = parser.parse_args([
            "--live-speaker-probe-speech-backend", "vad",
            "--live-speaker-probe-silero-speech-threshold", "0.3",
            "--live-speaker-probe-vad-min-speech-seconds", "0.032",
            "--live-speaker-probe-release-silero-speech-threshold", "0.25",
            "--live-speaker-probe-release-vad-min-speech-seconds", "0.032",
            "--live-speaker-probe-fast-release-window-seconds", "0.7",
            "--live-speaker-probe-fast-release-silero-speech-threshold", "0.01",
            "--live-speaker-probe-fast-release-vad-min-speech-seconds", "0.08",
        ])
        self.assertEqual(args.live_speaker_probe_speech_backend, "vad")
        self.assertEqual(args.live_speaker_probe_silero_speech_threshold, 0.3)
        self.assertEqual(args.live_speaker_probe_vad_min_speech_seconds, 0.032)
        self.assertEqual(args.live_speaker_probe_release_silero_speech_threshold, 0.25)
        self.assertEqual(args.live_speaker_probe_release_vad_min_speech_seconds, 0.032)
        self.assertEqual(args.live_speaker_probe_fast_release_window_seconds, 0.7)
        self.assertEqual(args.live_speaker_probe_fast_release_silero_speech_threshold, 0.01)
        self.assertEqual(args.live_speaker_probe_fast_release_vad_min_speech_seconds, 0.08)

    def test_split_silero_gate_resolves_acquire_and_release_values(self) -> None:
        args = SimpleNamespace(
            vad_silero_speech_threshold=0.5,
            vad_min_speech_seconds=0.25,
            live_speaker_probe_silero_speech_threshold=0.3,
            live_speaker_probe_vad_min_speech_seconds=0.032,
            live_speaker_probe_release_silero_speech_threshold=0.25,
            live_speaker_probe_release_vad_min_speech_seconds=0.064,
            live_speaker_probe_fast_release_silero_speech_threshold=0.01,
            live_speaker_probe_fast_release_vad_min_speech_seconds=0.08,
        )
        self.assertEqual(live_silero_gate_parameters(args), (0.3, 0.032))
        self.assertEqual(
            live_silero_gate_parameters(args, release=True),
            (0.25, 0.064),
        )
        self.assertEqual(
            live_silero_gate_parameters(args, release=True, fast_release=True),
            (0.01, 0.08),
        )

    def test_bayes_provisional_scores_before_first_final_profile(self) -> None:
        harness = _Harness()
        harness.live_memory = _EmptyMemory()
        harness.args.live_speaker_tracker = "bayes"
        harness.args.live_speaker_bayes_provisional_profiles = True
        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32), 0.8,
        )
        self.assertEqual(harness.embed_suffixes, [".live.short.wav"])
        self.assertEqual(payload["assigned_speaker"], "S1")

    def test_open_set_tracklets_score_before_first_final_profile(self) -> None:
        harness = _Harness()
        harness.live_memory = _EmptyMemory()
        harness.args.live_speaker_tracker = "bayes"
        harness.args.live_speaker_open_set_tracklets = True
        harness.args.live_speaker_open_set_preprofile = True

        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32), 0.8,
        )

        self.assertEqual(harness.embed_suffixes, [".live.short.wav"])
        self.assertEqual(payload["assigned_speaker"], "S1")

    def test_open_set_tracklets_keep_incumbent_gate_without_preprofile_flag(self) -> None:
        harness = _Harness()
        harness.live_memory = _EmptyMemory()
        harness.args.live_speaker_tracker = "bayes"
        harness.args.live_speaker_open_set_tracklets = True

        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32), 0.8,
        )

        self.assertEqual(harness.embed_suffixes, [])
        self.assertIsNone(payload["assigned_speaker"])

    def test_classic_tracker_still_waits_for_first_final_profile(self) -> None:
        harness = _Harness()
        harness.live_memory = _EmptyMemory()
        harness.args.live_speaker_tracker = "classic"
        harness.args.live_speaker_open_set_tracklets = True

        payload = harness._score_realtime_preview_speaker(
            np.ones(8, dtype=np.float32), 0.8,
        )

        self.assertEqual(harness.embed_suffixes, [])
        self.assertIsNone(payload["assigned_speaker"])

    def test_shared_production_core_creates_provisional_before_first_profile(self) -> None:
        parser = argparse.ArgumentParser()
        add_preview_live_speaker_arguments(parser)
        args = parser.parse_args([
            "--live-speaker-tracker", "bayes",
            "--live-speaker-probe-window-seconds", "0.7",
            "--live-speaker-probe-context-window-seconds", "1.5",
            "--live-speaker-probe-context-weight", "0.2",
            "--no-live-speaker-open-set-tracklets",
            "--live-speaker-bayes-provisional-profiles",
            "--live-speaker-bayes-provisional-creation-count", "1",
            "--live-speaker-bayes-provisional-creation-similarity-ceiling", "0.1",
            "--live-speaker-bayes-provisional-scale-agreement", "0.7",
        ])
        harness = _Harness()
        harness.args = args
        harness.live_memory = _EmptyMemory()

        decision = WindowLiveScoringMixin._shared_live_speaker_step(
            harness,
            media_time=0.8,
            speech=True,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            duration_seconds=0.7,
            probe_scheduled=True,
            context_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            context_duration_seconds=1.5,
        )

        self.assertEqual(decision.visible_speaker, "provisional_1")
        self.assertEqual(decision.reason, "provisional_acquire")

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
