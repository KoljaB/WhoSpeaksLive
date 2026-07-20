from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

from embeddings.embedding_providers import SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS
from embeddings.live_window_provider_campaign import (
    _expected_embeddings,
    run_campaign,
    run_provider_process,
)


def _write_wav(path: Path, seconds: float = 3.2) -> None:
    samples = np.zeros(int(16_000 * seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


class LiveWindowProviderCampaignTests(unittest.TestCase):
    def test_supported_provider_campaign_contains_all_fifteen_single_providers(self) -> None:
        self.assertEqual(len(SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS), 15)
        self.assertIn("speaker3d_eres2netv2", SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS)
        self.assertIn("espnet_ecapa_wavlm_joint", SUPPORTED_SINGLE_EMBEDDING_PROVIDER_IDS)

    def test_campaign_records_success_and_timeout_and_reaches_execution_100_percent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            (root / "tools" / "build_live_shifting_window_corpus.py").write_text("# fake\n", encoding="utf-8")
            audio = root / "audio.wav"
            _write_wav(audio)
            output_root = root / "output"
            expected = _expected_embeddings(audio, tuple(f"{value / 10:.1f}" for value in range(7, 31)), "0.2")

            def fake_runner(command, *, cwd, timeout_seconds, poll_seconds, on_poll):
                provider = command[command.index("--provider") + 1]
                video_id = command[command.index("--video-id") + 1]
                job_dir = output_root / "providers" / provider / "videos" / video_id
                job_dir.mkdir(parents=True, exist_ok=True)
                if provider == "fast":
                    progress = {
                        "status": "complete",
                        "completed_embeddings": expected,
                        "failed_embeddings": 0,
                        "percent": 100.0,
                        "elapsed_seconds": 1.25,
                        "provider_load_seconds": 0.25,
                    }
                    (job_dir / "job.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
                    (job_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
                    on_poll()
                    return 0, 1.25, False
                progress = {
                    "status": "running",
                    "completed_embeddings": 7,
                    "failed_embeddings": 0,
                    "percent": 7.0,
                }
                (job_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
                on_poll()
                return None, timeout_seconds, True

            result = run_campaign(
                root=root,
                audio_path=audio,
                video_id="video",
                output_root=output_root,
                providers=("fast", "slow"),
                python_executable=Path(sys.executable),
                timeout_seconds=5.0,
                process_runner=fake_runner,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["execution_percent"], 100.0)
            self.assertEqual([item["status"] for item in result["results"]], ["complete", "timed_out"])
            self.assertEqual(result["completed_embeddings"], expected + 7)
            timed_out_progress = json.loads(
                (output_root / "providers" / "slow" / "videos" / "video" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(timed_out_progress["status"], "timed_out")

    def test_real_runner_terminates_a_process_at_the_hard_timeout(self) -> None:
        started = time.monotonic()
        return_code, elapsed, timed_out = run_provider_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=Path.cwd(),
            timeout_seconds=0.2,
            poll_seconds=0.05,
        )
        self.assertTrue(timed_out)
        self.assertIsNone(return_code)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
