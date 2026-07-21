"""Compare cached Silero gate masks with the production runtime implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "tools", ROOT / "vendor" / "RealtimeSTT"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common.audio_utils import load_audio_file
try:
    from realtime_silero_vad import create_silero_vad_model
except ImportError:
    from core.silero_vad import create_silero_vad_model
from window.window_diarizer_runtime_audio import WindowRuntimeAudioMixin


class _RuntimeGate(WindowRuntimeAudioMixin):
    def __init__(
        self,
        model: Any,
        *,
        acquire_threshold: float,
        acquire_minimum: float,
        release_threshold: float,
        release_minimum: float,
        fast_release_threshold: float,
        fast_release_minimum: float,
    ) -> None:
        self.args = SimpleNamespace(
            vad_merge_gap_seconds=0.18,
            vad_silence_seconds=0.8,
            vad_silero_speech_threshold=0.5,
            vad_min_speech_seconds=0.25,
            live_speaker_probe_speech_backend="vad",
            vad_backend="silero",
            live_speaker_probe_silero_speech_threshold=acquire_threshold,
            live_speaker_probe_vad_min_speech_seconds=acquire_minimum,
            live_speaker_probe_release_silero_speech_threshold=release_threshold,
            live_speaker_probe_release_vad_min_speech_seconds=release_minimum,
            live_speaker_probe_fast_release_silero_speech_threshold=fast_release_threshold,
            live_speaker_probe_fast_release_vad_min_speech_seconds=fast_release_minimum,
        )
        self._vad_model = model
        self._vad_model_backend = "raw_onnx_ifless"
        self._vad_model_error = None


def _sample_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    positions = np.linspace(0, indices.size - 1, maximum, dtype=np.int64)
    return indices[positions]


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    try:
        audio, decoded_rate = load_audio_file(path, sample_rate)
        if decoded_rate != sample_rate:
            raise RuntimeError(f"Unexpected decoded sample rate {decoded_rate}")
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    except (ImportError, ModuleNotFoundError):
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--acquire-threshold", type=float, required=True)
    parser.add_argument("--release-threshold", type=float, required=True)
    parser.add_argument("--acquire-min-speech-seconds", type=float, default=0.032)
    parser.add_argument("--release-min-speech-seconds", type=float, default=0.032)
    parser.add_argument("--fast-release-window-seconds", type=float, default=0.0)
    parser.add_argument("--fast-release-threshold", type=float, default=-1.0)
    parser.add_argument("--fast-release-min-speech-seconds", type=float, default=-1.0)
    parser.add_argument("--probe-window-seconds", type=float, default=0.7)
    parser.add_argument("--clear-window-seconds", type=float, default=1.1)
    parser.add_argument("--samples-per-kind", type=int, default=100)
    parser.add_argument("--min-match-ratio", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = create_silero_vad_model(
        backend="raw_onnx_ifless",
        onnx_model_path=str(args.model_path.resolve()),
        onnx_threads=2,
        sample_rate=16_000,
        chunk_samples=512,
    )
    gate = _RuntimeGate(
        model,
        acquire_threshold=args.acquire_threshold,
        acquire_minimum=args.acquire_min_speech_seconds,
        release_threshold=args.release_threshold,
        release_minimum=args.release_min_speech_seconds,
        fast_release_threshold=args.fast_release_threshold,
        fast_release_minimum=args.fast_release_min_speech_seconds,
    )
    report: dict[str, Any] = {"videos": {}}
    total = 0
    matching = 0
    for video_id in args.video_id:
        video_root = args.corpus_root.resolve() / "videos" / video_id
        source = json.loads((video_root / "source.json").read_text(encoding="utf-8-sig"))
        sample_rate = int(source["sample_rate"])
        audio = _load_audio(Path(source["audio_path_at_creation"]), sample_rate)
        right_edges = np.load(
            video_root / "timeline" / "right_edges.i64.npy", allow_pickle=False
        )
        gate_root = args.gate_root.resolve() / video_id
        schedule = np.load(gate_root / "probe_schedule.u1.npy", allow_pickle=False)
        speech = np.load(gate_root / "speech_gate.u1.npy", allow_pickle=False)
        release = np.load(gate_root / "release_gate.u1.npy", allow_pickle=False)
        probe_indices = _sample_indices(np.flatnonzero(schedule), args.samples_per_kind)
        release_indices = _sample_indices(
            np.flatnonzero(right_edges >= round(args.clear_window_seconds * sample_rate)),
            args.samples_per_kind,
        )
        mismatches: list[dict[str, Any]] = []
        for kind, indices, window_seconds in (
            ("probe", probe_indices, args.probe_window_seconds),
            ("release", release_indices, args.clear_window_seconds),
        ):
            samples = round(window_seconds * sample_rate)
            for index in indices:
                right_sample = int(right_edges[index])
                left_sample = max(0, right_sample - samples)
                left = left_sample / float(sample_rate)
                right = right_sample / float(sample_rate)
                has_speech = gate._audio_has_live_probe_speech(
                    left,
                    right,
                    audio[left_sample:right_sample],
                    sample_rate,
                    release=kind == "release",
                )
                expected = bool(speech[index]) if kind == "probe" else bool(release[index])
                if kind == "probe":
                    actual = has_speech
                else:
                    actual = not has_speech
                    if args.fast_release_window_seconds > 0.0 and not actual:
                        fast_samples = round(args.fast_release_window_seconds * sample_rate)
                        fast_left_sample = max(0, right_sample - fast_samples)
                        fast_has_speech = gate._audio_has_live_probe_speech(
                            fast_left_sample / float(sample_rate),
                            right,
                            audio[fast_left_sample:right_sample],
                            sample_rate,
                            release=True,
                            fast_release=True,
                        )
                        actual = not fast_has_speech
                total += 1
                matching += int(actual == expected)
                if actual != expected:
                    mismatches.append({
                        "kind": kind,
                        "index": int(index),
                        "media_time": right,
                        "expected": expected,
                        "actual": actual,
                    })
        report["videos"][video_id] = {
            "probe_samples": int(probe_indices.size),
            "release_samples": int(release_indices.size),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
        }
    report["sample_count"] = total
    report["match_count"] = matching
    report["match_ratio"] = matching / total if total else 0.0
    report["minimum_match_ratio_required"] = float(args.min_match_ratio)
    report["passed"] = total > 0 and report["match_ratio"] >= float(args.min_match_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "sample_count", "match_ratio")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
