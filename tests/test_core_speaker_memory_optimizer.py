from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speakers.realtime_speaker_memory import SpeakerMemory as RealtimeSpeakerMemory
from speakers.speaker_embedding_cluster import SpeakerMemory as ClusterSpeakerMemory


def realtime_memory() -> RealtimeSpeakerMemory:
    return RealtimeSpeakerMemory(
        same_speaker_similarity=0.45,
        similarity_temperature=0.07,
        speaker_softmax_temperature=0.075,
        new_speaker_threshold=0.58,
        duplicate_profile_similarity=0.40,
        unknown_short_threshold=0.86,
        min_first_speaker_seconds=0.1,
        min_new_speaker_seconds=1.0,
        late_new_speaker_min_seconds=3.5,
        max_speakers=10,
        min_margin=0.08,
        margin_temperature=0.05,
        update_unknown_max=0.55,
    )



class SpeakerDecisionContractTests(unittest.TestCase):
    def assert_created_speaker_probability_contract(self, decision: object) -> None:
        self.assertEqual(decision.assigned_speaker, "S1")
        self.assertTrue(decision.created_speaker)
        self.assertEqual(decision.probabilities.get("unknown"), 0.0)
        self.assertEqual(decision.unknown_probability, 0.0)
        self.assertEqual(decision.probabilities.get("speaker1"), 1.0)

    def test_realtime_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        decision = realtime_memory().classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)
        decision = memory.classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_keeps_short_first_speaker_provisional_until_confirmed(self) -> None:
        memory = ClusterSpeakerMemory(
            min_first_speaker_seconds=1.0,
            first_speaker_immediate_min_seconds=4.0,
            min_new_speaker_seconds=2.0,
            new_speaker_confirmation_count=1,
            new_speaker_confirmation_similarity=0.58,
        )
        voice = np.array([0.4, 0.9, 0.1], dtype=np.float32)
        corroborating_voice = np.array([0.42, 0.88, 0.12], dtype=np.float32)

        provisional = memory.classify(voice, 1.5)
        confirmed = memory.classify(corroborating_voice, 1.5)

        self.assertIsNone(provisional.assigned_speaker)
        self.assertEqual(provisional.assignment_source, "first_speaker_pending")
        self.assertEqual(memory.profile_count(), 1)
        self.assertEqual(confirmed.assigned_speaker, "S1")
        self.assertTrue(confirmed.created_speaker)
        self.assertEqual(memory.export_profiles()[0]["sentence_count"], 2)

    def test_cluster_memory_does_not_anchor_short_startup_outlier_as_speaker_one(self) -> None:
        memory = ClusterSpeakerMemory(
            same_speaker_similarity=0.43,
            min_first_speaker_seconds=1.8373,
            first_speaker_immediate_min_seconds=4.0,
            min_new_speaker_seconds=2.0358,
            new_speaker_confirmation_similarity=0.5801,
        )
        startup_outlier = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        main_voice = np.array([0.43, 0.90, 0.05], dtype=np.float32)
        main_voice_long = np.array([0.44, 0.88, 0.08], dtype=np.float32)
        main_voice_followup = np.array([0.41, 0.91, 0.03], dtype=np.float32)

        first = memory.classify(startup_outlier, 1.89)
        second = memory.classify(main_voice, 1.89)
        anchor = memory.classify(main_voice_long, 4.51)
        followup = memory.classify(main_voice_followup, 4.95)

        self.assertIsNone(first.assigned_speaker)
        self.assertIsNone(second.assigned_speaker)
        self.assertEqual(anchor.assigned_speaker, "S1")
        self.assertTrue(anchor.created_speaker)
        self.assertEqual(followup.assigned_speaker, "S1")
        self.assertFalse(followup.created_speaker)
        self.assertEqual(memory.profile_count(), 1)

    def test_cluster_memory_long_first_sentence_still_creates_speaker_immediately(self) -> None:
        memory = ClusterSpeakerMemory(
            min_first_speaker_seconds=1.0,
            first_speaker_immediate_min_seconds=4.0,
        )

        decision = memory.classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 8.1)

        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_upsert_keeps_explicit_speaker_label(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)

        label = memory.upsert_profile("S3", np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)
        decision = memory.score_existing(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)

        self.assertEqual(label, "S3")
        self.assertEqual(decision.assigned_speaker, "S3")

    def test_cluster_memory_allocates_after_highest_remaining_speaker_label(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)
        memory.upsert_profile("S1", np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)
        memory.upsert_profile("S3", np.array([0.0, 1.0, 0.0], dtype=np.float32), 1.0)
        self.assertEqual(memory.remove_profiles({"S3"}), ["S3"])

        label = memory.add_profile(np.array([0.0, 0.0, 1.0], dtype=np.float32), 1.0)

        self.assertEqual(label, "S4")
        self.assertEqual(
            [profile["label"] for profile in memory.export_profiles()],
            ["S1", "S4"],
        )


class OptimizerParityTests(unittest.TestCase):
    def load_current_memory_optimizer(self) -> object:
        module_path = ROOT / "runtime" / "optimization" / "optimize_current_memory_21.py"
        if not module_path.is_file():
            self.skipTest(f"Local optimizer harness is not present: {module_path}")
        spec = importlib.util.spec_from_file_location("score_parity_current_memory_optimizer", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def base_sentence_payload(self) -> dict[str, object]:
        return {
            "index": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "same speaker evidence",
            "speech_audio_ratio": 1.0,
        }

    def unknown_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "assigned_speaker": None,
            "probabilities": {"unknown": 1.0},
            "similarities": {},
            "unknown_probability": 1.0,
            "assignment_source": "embedding",
        }

    def confirmed_sentence_payload(self) -> dict[str, object]:
        return {
            **self.base_sentence_payload(),
            "pending": False,
            "revision": True,
            "retro_reassigned": True,
            "revision_from": "S3",
            "revision_to": "S6",
            "assigned_speaker": "S6",
            "probabilities": {"unknown": 0.0, "speaker6": 1.0},
            "similarities": {"S6": 0.82},
            "unknown_probability": 0.0,
            "assignment_source": "retro",
        }

    def canonical(self) -> list[dict[str, object]]:
        return [
            {
                "speaker": "canonical_speaker",
                "start": 0.0,
                "end": 3.0,
                "text": "same speaker evidence",
            }
        ]

    def test_current_memory_optimizer_matches_live_memory_path_when_live_refinement_is_disabled(self) -> None:
        optimizer = self.load_current_memory_optimizer()
        from window.window_validation_replay import replay_cached_window_diarizer

        config = dict(optimizer.BASE_CONFIG)
        weights = {"espnet_ecapa_wavlm_joint": 1.0}
        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.4,
                "text": "alpha beta gamma delta",
                "spoken_word_seconds": 2.4,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 2.5,
                "end": 4.7,
                "text": "alpha beta gamma delta again",
                "spoken_word_seconds": 2.2,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 5.0,
                "end": 5.8,
                "text": "epsilon zeta eta theta",
                "spoken_word_seconds": 0.8,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 3,
                "start": 6.0,
                "end": 8.7,
                "text": "epsilon zeta eta theta longer",
                "spoken_word_seconds": 2.7,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 4,
                "start": 9.0,
                "end": 11.1,
                "text": "alpha beta gamma returns",
                "spoken_word_seconds": 2.1,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.99, 0.05], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.02], dtype=np.float32),
        ]
        dataset = argparse.Namespace(
            sentences=sentences,
            embeddings={"espnet_ecapa_wavlm_joint": embeddings},
        )

        optimizer_rows = optimizer.replay_current_memory(dataset, weights, config)
        live_args = argparse.Namespace(
            **config,
            min_embed_seconds=0.0,
            section_gap_new_speaker=False,
            unknown_pair_new_speaker=False,
            speaker_refinement=False,
        )
        live_rows = replay_cached_window_diarizer(sentences, embeddings, live_args).final_payloads

        self.assertEqual(len(live_rows), len(optimizer_rows))
        for live, optimized in zip(live_rows, optimizer_rows):
            self.assertEqual(live["index"], optimized["index"])
            self.assertEqual(live.get("assigned_speaker"), optimized.get("assigned_speaker"))
            self.assertEqual(live.get("created_speaker"), optimized.get("created_speaker"))
            self.assertEqual(live.get("assignment_source"), optimized.get("assignment_source"))
            self.assertEqual(live.get("probabilities"), optimized.get("probabilities"))
            self.assertEqual(live.get("similarities"), optimized.get("similarities"))
            self.assertEqual(live.get("unknown_probability"), optimized.get("unknown_probability"))
            self.assertEqual(live.get("top_similarity"), optimized.get("top_similarity"))
            self.assertEqual(live.get("margin"), optimized.get("margin"))

        self.assertEqual([row.get("assigned_speaker") for row in live_rows], ["S1", "S1", "S2", "S2", "S1"])
        self.assertTrue(live_rows[2].get("retro_reassigned"))

    def test_current_memory_optimizer_uses_fast_cached_live_replay(self) -> None:
        optimizer = self.load_current_memory_optimizer()
        backend = mock.Mock()
        backend.score_rows.return_value = {"score": 1.0}
        backend.global_metrics.return_value = {"global_robust_score": 1.0}
        prepared = argparse.Namespace(name="clip", dataset=object())
        base_args = argparse.Namespace()

        with mock.patch.object(
            optimizer,
            "replay_current_live",
            return_value=[{"index": 0}],
        ) as replay_live:
            result = optimizer.evaluate_candidate(
                backend,
                [prepared],
                {"provider": 1.0},
                {"threshold": 0.4},
                base_args,
                phase="test",
            )

        replay_live.assert_called_once_with(
            prepared.dataset,
            {"provider": 1.0},
            {"threshold": 0.4},
            base_args,
        )
        self.assertEqual(result["per_video"], {"clip": {"score": 1.0}})

        dataset = argparse.Namespace(sentences=[{"index": 0}, {"index": 1}])
        cached_args = argparse.Namespace(cached_replay_defer_speaker_refinement=True)
        replay_result = argparse.Namespace(final_payloads=[{"index": 0}])
        with (
            mock.patch.object(
                optimizer,
                "make_embedding",
                side_effect=[np.array([1.0]), np.array([0.0])],
            ),
            mock.patch.object(
                optimizer,
                "make_cached_replay_args",
                return_value=cached_args,
            ) as make_args,
            mock.patch.object(
                optimizer,
                "replay_cached_window_diarizer",
                return_value=replay_result,
            ) as replay_cached,
        ):
            rows = optimizer.replay_current_live(
                dataset,
                {"provider": 1.0},
                {"threshold": 0.4},
                base_args,
            )

        make_args.assert_called_once()
        replay_cached.assert_called_once_with(
            dataset.sentences,
            mock.ANY,
            cached_args,
            defer_speaker_refinement=True,
        )
        self.assertEqual(rows, replay_result.final_payloads)

    def test_cached_live_replay_scores_committed_prototype_reassignment(self) -> None:
        from window.window_validation_replay import replay_cached_window_diarizer, replay_cached_window_score
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui"]):
            args = parse_args()
        args = args.with_updates(
            min_embed_seconds=0.0,
            first_speaker_immediate_min_seconds=3.0,
            section_gap_new_speaker=False,
            unknown_pair_new_speaker=False,
            speaker_refinement=True,
            allow_speaker_reassignment=True,
            min_new_speaker_words=3,
            known_speaker_gray_zone_min_unknown_probability=1.1,
        )

        sentences = [
            {
                "index": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "alpha beta anchor",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 1,
                "start": 3.2,
                "end": 6.2,
                "text": "gamma delta",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
            {
                "index": 2,
                "start": 6.4,
                "end": 9.4,
                "text": "gamma delta epsilon",
                "spoken_word_seconds": 3.0,
                "speech_audio_ratio": 1.0,
            },
        ]
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        replay = replay_cached_window_diarizer(sentences, embeddings, args)
        chronological_replay = replay_cached_window_diarizer(
            sentences,
            embeddings,
            args,
            defer_speaker_refinement=False,
        )
        final_by_index = {payload["index"]: payload for payload in replay.final_payloads}
        chronological_final_by_index = {
            payload["index"]: payload
            for payload in chronological_replay.final_payloads
        }

        self.assertEqual(final_by_index[0]["assigned_speaker"], "S1")
        self.assertEqual(final_by_index[1]["assigned_speaker"], "S2")
        self.assertEqual(final_by_index[1]["assignment_source"], "prototype_reassign")
        self.assertEqual(final_by_index[2]["assigned_speaker"], "S2")
        self.assertEqual(
            [chronological_final_by_index[index]["assigned_speaker"] for index in sorted(chronological_final_by_index)],
            [final_by_index[index]["assigned_speaker"] for index in sorted(final_by_index)],
        )
        self.assertTrue(any(
            record.get("event") == "sentence"
            and (record.get("payload") or {}).get("index") == 1
            and (record.get("payload") or {}).get("prototype_reassigned")
            and not (record.get("payload") or {}).get("provisional_assignment")
            for record in replay.records
        ))

        canonical = [
            {"speaker": "speaker_a", "start": 0.0, "end": 3.0, "text": "alpha beta anchor"},
            {"speaker": "speaker_b", "start": 3.2, "end": 6.2, "text": "gamma delta"},
            {"speaker": "speaker_b", "start": 6.4, "end": 9.4, "text": "gamma delta epsilon"},
        ]
        score = replay_cached_window_score(sentences, embeddings, args, canonical, match_mode="timestamp")

        self.assertEqual(score["rows"][1]["assigned_speaker"], "S2")
        self.assertEqual(score["assigned_counts"], {"S1": 1, "S2": 2})
        self.assertEqual(score["duration_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
