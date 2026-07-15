from __future__ import annotations

import argparse
import importlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_diarizer import WindowDiarizer
from window.window_domain import VadWindowState



from tests.window_diarizer_support import make_window_diarizer


class RepositoryStructureTests(unittest.TestCase):
    def test_publication_ignores_local_data_and_all_dot_env_files(self) -> None:
        gitignore_lines = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn("/data/", gitignore_lines)
        self.assertIn(".env", gitignore_lines)
        self.assertIn(".env.*", gitignore_lines)
        self.assertNotIn("!.env.example", gitignore_lines)
        self.assertFalse((ROOT / ".env.example").exists())

    def test_realtimestt_warmup_asset_and_vendor_licenses_are_release_inputs(self) -> None:
        warmup_audio = ROOT / "vendor" / "RealtimeSTT" / "assets" / "warmup_audio.wav"
        self.assertTrue(warmup_audio.is_file())
        self.assertGreater(warmup_audio.stat().st_size, 0)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('RealtimeSTT = ["assets/**/*", "LICENSE"]', pyproject)
        self.assertIn('RealtimeSTT_server = ["LICENSE"]', pyproject)
        self.assertIn('stream2sentence = ["data/*.json", "LICENSE"]', pyproject)

        for relative_path in (
            "THIRD_PARTY_NOTICES.md",
            "vendor/RealtimeSTT/LICENSE",
            "vendor/RealtimeSTT_server/LICENSE",
            "vendor/stream2sentence/LICENSE",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_package_imports_do_not_require_tools_on_sys_path(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(SRC), str(ROOT / "vendor")))
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from window.window_diarizer import WindowDiarizer; print(WindowDiarizer.__name__)",
                ],
                cwd=directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "WindowDiarizer")

    def test_window_module_entrypoint_prints_help(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, "-m", "window.youtube_window_diarize_gui", "--help"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Growing-window faster-whisper speaker diarization GUI", completed.stdout)

    def test_runtime_dir_env_redirects_mutable_defaults(self) -> None:
        import paths as paths

        original_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["WHOSPEAKS_RUNTIME_DIR"] = directory
                os.environ.pop("WHOSPEAKS_CACHE_DIR", None)
                os.environ.pop("WHOSPEAKS_MODEL_DIR", None)
                os.environ.pop("WHOSPEAKS_SPEAKER_LIBRARY_DIR", None)
                reloaded = importlib.reload(paths)
                runtime = Path(directory).resolve()
                self.assertEqual(reloaded.RUNTIME_DIR, runtime)
                self.assertEqual(reloaded.CACHE_DIR, runtime / "cache")
                self.assertEqual(reloaded.MODEL_DIR, runtime / "models")
                self.assertEqual(reloaded.SPEAKER_LIBRARY_DIR, runtime / "speakers")
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(paths)

    def test_docker_persists_people_library_inside_data_volume(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("WHOSPEAKS_SPEAKER_LIBRARY_DIR=/data/speakers", dockerfile)
        self.assertIn('"--speaker-library-dir", "/data/speakers"', dockerfile)
        self.assertIn("/data/speakers", dockerfile)

    def test_window_gui_default_embedding_provider_ignores_environment_override(self) -> None:
        import window.window_config as window_config

        original_env = dict(os.environ)
        try:
            os.environ["WHOSPEAKS_WINDOW_EMBEDDING_PROVIDER"] = "speechbrain_ecapa"
            reloaded = importlib.reload(window_config)
            self.assertEqual(
                reloaded.DEFAULT_WINDOW_EMBEDDING_PROVIDER,
                "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37",
            )
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(window_config)

    def test_window_gui_tuned_default_parameters_match_promoted_set(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        expected = {
            "embedding_provider": (
                "espnet_ecapa_wavlm_joint=1.0+speechbrain_resnet=0.28+wespeaker_campplus=0.37"
            ),
            "interval_seconds": 0.7,
            "same_speaker_similarity": 0.43,
            "similarity_temperature": 0.061,
            "speaker_softmax_temperature": 0.0557,
            "new_speaker_threshold": 0.4309,
            "duplicate_profile_similarity": 0.4247,
            "unknown_short_threshold": 0.287,
            "min_first_speaker_seconds": 1.8373,
            "first_speaker_immediate_min_seconds": 4.0,
            "min_new_speaker_seconds": 2.0358,
            "late_new_speaker_min_seconds": 3.1604,
            "max_speakers": 12,
            "min_margin": 0.0372,
            "margin_temperature": 0.0361,
            "update_unknown_max": 0.4289,
            "new_speaker_confirmation_count": 1,
            "new_speaker_confirmation_similarity": 0.5801,
            "max_pending_new_speakers": 6,
            "known_speaker_min_similarity": 0.5563,
            "known_speaker_gray_zone_min_unknown_probability": 0.064,
            "profile_update_min_similarity": 0.5011,
            "profile_update_min_margin": 0.0037,
            "low_similarity_unknown_floor_similarity": 0.56,
            "low_similarity_unknown_floor_probability": 0.1885,
            "gray_zone_promote_max_similarity": 0.55,
            "min_new_speaker_words": 3,
            "retro_reassign_min_similarity": 0.02,
            "retro_reassign_min_margin": 0.0,
            "speaker_refinement_final_passes": 1,
            "speaker_refinement_small_island_merge": True,
            "speaker_refinement_tiny_fragmented_merge": True,
            "speaker_refinement_tiny_fragmented_max_duration": 6.0,
            "speaker_refinement_tiny_fragmented_max_segments": 8,
            "speaker_refinement_tiny_fragmented_min_islands": 2,
            "speaker_refinement_tiny_fragmented_max_islands": 3,
            "speaker_refinement_tiny_fragmented_min_neighbor_share": 0.5,
            "speaker_refinement_terminal_outro_merge": True,
            "speaker_refinement_terminal_outro_max_duration": 12.0,
            "speaker_refinement_terminal_outro_lookback_segments": 2,
            "speaker_refinement_terminal_outro_min_target_duration": 5.0,
            "speaker_refinement_unknown_same_speaker_fill": True,
            "speaker_refinement_unknown_same_speaker_max_duration": 3.0,
            "speaker_refinement_unknown_same_speaker_max_segments": 1,
            "speaker_refinement_unknown_previous_speaker_fill": True,
            "speaker_refinement_unknown_previous_speaker_max_duration": 0.75,
            "speaker_refinement_unknown_previous_speaker_max_segments": 1,
            "speaker_refinement_unknown_previous_speaker_max_previous_gap": 0.35,
            "speaker_refinement_unknown_previous_speaker_min_next_gap": 0.3,
            "speaker_refinement_unknown_next_speaker_fill": True,
            "speaker_refinement_unknown_next_speaker_max_duration": 1.75,
            "speaker_refinement_unknown_next_speaker_max_segments": 1,
            "speaker_refinement_unknown_next_speaker_max_next_gap": 0.05,
            "speaker_refinement_unknown_next_speaker_min_previous_gap": 0.15,
            "speaker_refinement_long_low_confidence_retro_split": True,
            "speaker_refinement_long_low_confidence_retro_min_duration": 4.0,
            "speaker_refinement_long_low_confidence_retro_max_similarity": 0.06,
            "speaker_refinement_long_low_confidence_retro_max_margin": 0.04,
            "speaker_refinement_long_low_confidence_retro_max_splits": 1,
            "min_embed_seconds": 0.5,
            "min_speech_audio_ratio": 0.0,
            "live_speaker_embedding_provider": "pyannote_wespeaker_resnet34_lm=1.0+wespeaker_resnet34_lm_onnx=0.50",
            "unstable_tail_seconds": 1.35,
            "vad_silence_seconds": 1.1,
            "vad_final_window_post_silence_seconds": 0.75,
            "sentence_boundary_pre_padding_seconds": 0.06,
            "sentence_boundary_post_padding_seconds": 0.09,
            "sentence_boundary_gap_ratio": 0.6,
            "realtime_preview_model_preset": "community-64l",
            "realtime_preview_model": "Kroko-EN-Community-64-L-Streaming-001.data",
            "realtime_preview_startup_timeout_seconds": 12.0,
            "realtime_preview_diarize_min_audio_seconds": 1.5,
            "realtime_preview_diarize_min_advance_seconds": 0.75,
            "realtime_preview_diarize_min_similarity": 0.45,
            "realtime_preview_diarize_min_margin": 0.08,
            "realtime_preview_diarize_min_known_probability": 0.5,
            "live_speaker_assignment": True,
            "live_speaker_embedding_min_interval_seconds": 0.75,
            "live_speaker_embedding_target_utilization": 0.25,
            "live_speaker_verify_on_change": False,
            "live_speaker_verify_min_interval_seconds": 2.0,
            "live_speaker_ema_window_seconds": 1.0,
            "live_speaker_ema_count": 1,
            "live_speaker_ema_alpha": 0.55,
            "live_speaker_probe_interval_seconds": 0.75,
            "live_speaker_probe_attack_interval_seconds": 0.0,
            "live_speaker_probe_window_seconds": 1.0,
            "live_speaker_probe_hold_seconds": 1.0,
            "live_speaker_probe_min_advance_seconds": 0.75,
            "live_speaker_probe_attack_min_advance_seconds": 0.0,
            "live_speaker_probe_min_speech_seconds": 0.15,
            "live_speaker_probe_clear_on_silence": True,
            "live_speaker_probe_clear_window_seconds": 1.0,
            "live_speaker_probe_clear_silence_count": 1,
            "live_speaker_probe_clear_unknown_count": 2,
            "live_speaker_probe_unknown_clear_debounce_seconds": 0.0,
            "live_speaker_probe_unknown_keepalive": False,
            "live_speaker_probe_unknown_release_smoothing": "none",
            "live_speaker_probe_unknown_release_count": 3,
            "live_speaker_probe_unknown_release_ema_alpha": 0.5,
            "live_speaker_probe_unknown_release_margin": 0.0,
            "live_speaker_raw_change_snap": True,
            "live_speaker_raw_change_min_probability": 0.7,
            "live_speaker_raw_change_min_margin": 0.25,
            "live_speaker_sentence_hint": True,
            "live_speaker_sentence_hint_override": True,
            "live_speaker_sentence_hint_max_lag_seconds": 1.25,
            "live_speaker_sentence_hint_new_speaker_max_lag_seconds": 1.25,
            "live_speaker_sentence_hint_hold_seconds": 0.3,
            "browser_live_observation_output": None,
            "browser_live_observation_interval_seconds": 0.1,
            "browser_live_observation_max_sample_gap_seconds": 0.5,
            "browser_live_observation_flicker_gap_seconds": 0.25,
        }

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui.py"]):
            args = parse_args()

        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)

    def test_window_gui_can_select_kroko_pro_16l_preview_preset(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--realtime-preview-model-preset",
                "pro-16l",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.realtime_preview_model_preset, "pro-16l")
        self.assertEqual(args.realtime_preview_model, "Kroko-EN-Pro-16-L-Streaming-001.data")
        if args.realtime_preview_model_path is not None:
            self.assertEqual(args.realtime_preview_model_path.name, "Kroko-EN-Pro-16-L-Streaming-001.data")
        self.assertEqual(args.realtime_preview_startup_timeout_seconds, 45.0)
        self.assertEqual(args.realtime_preview_interval_seconds, 0.32)
        self.assertEqual(args.realtime_preview_min_audio_seconds, 0.32)
        self.assertEqual(args.realtime_preview_min_advance_seconds, 0.32)
        self.assertEqual(args.realtime_preview_feed_chunk_seconds, 0.32)

    def test_window_gui_language_selects_kroko_and_sentence_tokenizer(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "de",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "de")
        self.assertEqual(args.realtime_preview_language, "de")
        self.assertEqual(args.realtime_preview_model_preset, "community-64l")
        self.assertEqual(args.realtime_preview_model, "Kroko-DE-Community-64-L-Streaming-001.data")
        self.assertEqual(args.sentence_tokenizer, "nltk+rule-based")
        self.assertEqual(args.sentence_language, "de")

    def test_window_gui_env_language_is_not_overridden_by_custom_preview_model(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.dict(os.environ, {"WHOSPEAKS_LANGUAGE": "de"}):
            with mock.patch.object(
                sys,
                "argv",
                [
                    "youtube_window_diarize_gui.py",
                    "--realtime-preview-model",
                    "Kroko-EN-Community-64-L-Streaming-001.data",
                ],
            ):
                args = parse_args()

        self.assertEqual(args.language, "de")
        self.assertEqual(args.realtime_preview_language, "de")
        self.assertEqual(args.realtime_preview_model_preset, "custom")
        self.assertEqual(args.sentence_language, "de")

    def test_window_gui_extended_language_requires_preview_off(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "pl",
            ],
        ):
            with mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_window_gui_extended_language_selects_nltk_when_preview_off(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "pl",
                "--realtime-preview-engine",
                "off",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "pl")
        self.assertEqual(args.sentence_tokenizer, "nltk+rule-based")
        self.assertEqual(args.sentence_language, "pl")
        self.assertEqual(args.realtime_preview_engine, "off")
        self.assertEqual(args.realtime_preview_model, "")
        self.assertIsNone(args.realtime_preview_model_path)

    def test_window_gui_extended_language_uses_stanza_when_needed(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--language",
                "zh",
                "--realtime-preview-engine",
                "off",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.language, "zh")
        self.assertEqual(args.sentence_tokenizer, "stanza")
        self.assertEqual(args.sentence_language, "zh-hans")

    def test_kroko_preview_model_path_searches_configured_model_dir(self) -> None:
        from window.window_config import default_kroko_preview_model_path

        model_name = "Kroko-EN-Pro-16-L-Streaming-001.data"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / model_name
            model_path.write_bytes(b"")

            with mock.patch.dict(os.environ, {"WHOSPEAKS_KROKO_PREVIEW_MODEL_DIR": directory}):
                resolved = default_kroko_preview_model_path(model_name, use_env=False)

        self.assertEqual(resolved, model_path)

    def test_kroko_preview_community_model_downloads_to_model_dir(self) -> None:
        from window.window_config import download_kroko_preview_model

        model_name = "Kroko-DE-Community-64-L-Streaming-001.data"

        def fake_hf_hub_download(**kwargs: object) -> str:
            target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
            target.write_bytes(b"model")
            return str(target)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download) as download:
                resolved = download_kroko_preview_model(model_name, target_dir=Path(directory))

        self.assertEqual(resolved.name, model_name)
        self.assertTrue(download.called)

    def test_kroko_preview_auto_download_rejects_non_public_model(self) -> None:
        from window.window_config import download_kroko_preview_model

        with self.assertRaisesRegex(RuntimeError, "public Community"):
            download_kroko_preview_model("Kroko-EN-Pro-16-L-Streaming-001.data")

    def test_window_loop_restarts_interval_after_successful_split(self) -> None:
        diarizer = make_window_diarizer()
        diarizer._update_config(
            interval_seconds=1.0,
            min_playback_advance_seconds=0.0,
            min_window_seconds=0.0,
            final_flush_epsilon_seconds=0.01,
            vad_sentence_splitting=False,
            asr_vad_gate=False,
        )
        diarizer._model = object()
        diarizer.duration = 10.0
        diarizer._streaming_audio = True
        clock = argparse.Namespace(value=0.0)
        stop_event = threading.Event()
        diarizer.playback_time = mock.Mock(side_effect=lambda: 2.0 + clock.value)
        diarizer._vad_window_state = mock.Mock(
            return_value=VadWindowState(has_speech=True, should_flush=False)
        )
        diarizer._asr_vad_gate_enabled = mock.Mock(return_value=False)
        first_sentence = argparse.Namespace(text="A complete thought.", next_left=0.5)
        diarizer._transcribe_window = mock.Mock(
            return_value=argparse.Namespace(
                sentences=[first_sentence],
                segment_count=1,
                word_count=3,
            )
        )
        diarizer._emit_sentence = mock.Mock()
        diarizer._advance_realtime_preview_after_commit = mock.Mock()
        diarizer._pause_realtime_preview = mock.Mock()
        diarizer._drain_embedding_jobs = mock.Mock()
        diarizer._revisit_unknown_sentences = mock.Mock()
        diarizer._finalize_speaker_refinement = mock.Mock()
        diarizer._drain_live_memory_update_jobs = mock.Mock()

        def advance_clock(_seconds: float) -> None:
            clock.value += 0.1
            if diarizer._transcribe_window.call_count and clock.value >= 1.5:
                stop_event.set()

        with (
            mock.patch(
                "window.window_diarizer_transcription.time.monotonic",
                side_effect=lambda: clock.value,
            ),
            mock.patch(
                "window.window_diarizer_transcription.time.sleep",
                side_effect=advance_clock,
            ),
        ):
            diarizer._run(stop_event)

        diarizer._transcribe_window.assert_called_once()
        diarizer._advance_realtime_preview_after_commit.assert_called_once_with(0.5)

    def test_window_gui_accepts_remote_embeddings_backend_alias(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "-embeddings-backend",
                "remote",
                "--remote-embeddings-url",
                "http://127.0.0.1:8660",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.embeddings_backend, "remote")
        self.assertEqual(args.remote_embeddings_url, "http://127.0.0.1:8660")

    def test_window_gui_can_disable_live_speaker_assignment_with_master_switch(self) -> None:
        from window.youtube_window_diarize_gui import parse_args

        with mock.patch.object(
            sys,
            "argv",
            [
                "youtube_window_diarize_gui.py",
                "--no-live-speaker-assignment",
                "--live-speaker-embedding-provider",
                "pyannote_wespeaker_resnet34_lm",
            ],
        ):
            args = parse_args()

        self.assertFalse(args.live_speaker_assignment)
        self.assertFalse(args.live_speaker_probe)
        self.assertFalse(args.live_speaker_sentence_hint)
        self.assertFalse(args.live_speaker_highlight_transcript)
        self.assertFalse(args.live_speaker_verify_on_change)
        self.assertFalse(args.live_speaker_raw_change_snap)

    def test_cunk_canonical_is_a_small_fixture(self) -> None:
        from paths import CUNK_CANONICAL

        self.assertTrue(CUNK_CANONICAL.is_file())
        self.assertIn("tests", CUNK_CANONICAL.parts)
        self.assertIn("fixtures", CUNK_CANONICAL.parts)

    def test_release_manifest_prunes_repository_only_folders(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        for relative_path in ("tests", "tests-js", "tools", "docs", "docs-private"):
            self.assertIn(f"prune {relative_path}", manifest)


if __name__ == "__main__":
    unittest.main()
