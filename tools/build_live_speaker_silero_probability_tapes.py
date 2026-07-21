"""Precompute causal Silero frame probabilities for live-speaker gate sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "vendor", ROOT / "vendor" / "RealtimeSTT"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common.audio_utils import load_audio_file
from build_live_speaker_gate_tapes import (
    _atomic_json,
    _atomic_npy,
    _scheduled_ticks,
    _sha256_bytes,
    _sha256_file,
)


SILERO_RATE = 16_000
SILERO_CHUNK = 512
TAPE_ID = "production_silero_live_gate_probability_tape_v1"


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate == SILERO_RATE:
        return values
    duration = values.size / float(sample_rate)
    target_size = max(1, int(round(duration * SILERO_RATE)))
    source_times = np.arange(values.size, dtype=np.float64) / float(sample_rate)
    target_times = np.arange(target_size, dtype=np.float64) / float(SILERO_RATE)
    return np.interp(target_times, source_times, values).astype(np.float32)


def _window_probabilities(model: Any, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = _resample(audio, sample_rate)
    reset = getattr(model, "reset_states", None)
    if callable(reset):
        reset()
    probabilities: list[float] = []
    for start in range(0, values.size, SILERO_CHUNK):
        end = min(values.size, start + SILERO_CHUNK)
        if end - start < SILERO_CHUNK // 2:
            break
        chunk = values[start:end]
        if chunk.size < SILERO_CHUNK:
            padded = np.zeros(SILERO_CHUNK, dtype=np.float32)
            padded[:chunk.size] = chunk
            chunk = padded
        probability = model(chunk.astype(np.float32, copy=False), SILERO_RATE)
        probabilities.append(float(probability))
    return np.asarray(probabilities, dtype=np.float32)


def build_video(
    corpus_root: Path,
    output_root: Path,
    video_id: str,
    *,
    probe_window_seconds: float,
    clear_window_seconds: float,
    cadence_seconds: float,
    model: Any,
) -> dict[str, Any]:
    video_root = corpus_root / "videos" / video_id
    source = json.loads((video_root / "source.json").read_text(encoding="utf-8-sig"))
    timeline = json.loads(
        (video_root / "timeline" / "metadata.json").read_text(encoding="utf-8-sig")
    )
    sample_rate = int(source["sample_rate"])
    audio_path = Path(str(source["audio_path_at_creation"])).resolve()
    if _sha256_file(audio_path) != source["audio_file_sha256"]:
        raise RuntimeError(f"Compressed source hash mismatch for {video_id}")
    audio, decoded_rate = load_audio_file(audio_path, sample_rate)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if decoded_rate != sample_rate or int(audio.size) != int(source["decoded_samples"]):
        raise RuntimeError(f"Decoded source geometry mismatch for {video_id}")
    if _sha256_bytes(np.ascontiguousarray(audio).tobytes()) != source["decoded_pcm_sha256"]:
        raise RuntimeError(f"Decoded PCM hash mismatch for {video_id}")

    right_edges = np.load(
        video_root / "timeline" / "right_edges.i64.npy", mmap_mode="r", allow_pickle=False
    )
    rows = int(right_edges.shape[0])
    probe_samples = round(probe_window_seconds * sample_rate)
    clear_samples = round(clear_window_seconds * sample_rate)
    cadence_samples = round(cadence_seconds * sample_rate)
    schedule = _scheduled_ticks(right_edges, probe_samples, cadence_samples)
    max_probe_frames = int(np.ceil(probe_window_seconds * SILERO_RATE / SILERO_CHUNK))
    max_clear_frames = int(np.ceil(clear_window_seconds * SILERO_RATE / SILERO_CHUNK))
    probe = np.full((rows, max_probe_frames), -1.0, dtype=np.float32)
    release = np.full((rows, max_clear_frames), -1.0, dtype=np.float32)

    scheduled = np.flatnonzero(schedule)
    for ordinal, index in enumerate(scheduled, start=1):
        right = int(right_edges[index])
        values = _window_probabilities(model, audio[max(0, right - probe_samples):right], sample_rate)
        probe[index, :values.size] = values
        if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(scheduled):
            print(f"[silero-prob] {video_id} probe {ordinal}/{len(scheduled)}", flush=True)

    release_indices = np.flatnonzero(right_edges >= clear_samples)
    for ordinal, index in enumerate(release_indices, start=1):
        right = int(right_edges[index])
        values = _window_probabilities(model, audio[max(0, right - clear_samples):right], sample_rate)
        release[index, :values.size] = values
        if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(release_indices):
            print(f"[silero-prob] {video_id} release {ordinal}/{len(release_indices)}", flush=True)

    target = output_root / video_id
    _atomic_npy(target / "probe_probabilities.f32.npy", probe)
    _atomic_npy(target / "release_probabilities.f32.npy", release)
    _atomic_npy(target / "probe_schedule.u1.npy", schedule)
    metadata = {
        "tape_id": TAPE_ID,
        "video_id": video_id,
        "source_audio_sha256": source["audio_file_sha256"],
        "decoded_pcm_sha256": source["decoded_pcm_sha256"],
        "timeline_sha256": timeline["timeline_sha256"],
        "sample_rate": sample_rate,
        "tick_count": rows,
        "probe_window_seconds": probe_window_seconds,
        "clear_window_seconds": clear_window_seconds,
        "cadence_seconds": cadence_seconds,
        "silero_sample_rate": SILERO_RATE,
        "silero_chunk_samples": SILERO_CHUNK,
        "scheduled_probe_count": int(scheduled.size),
        "release_tick_count": int(release_indices.size),
    }
    _atomic_json(target / "probability_tape.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--probe-window-seconds", type=float, default=0.7)
    parser.add_argument("--clear-window-seconds", type=float, default=1.1)
    parser.add_argument("--cadence-seconds", type=float, default=0.75)
    args = parser.parse_args()
    import silero_vad as silero_package
    try:
        from core.silero_vad import create_silero_vad_model
    except ImportError:
        from realtime_silero_vad import create_silero_vad_model

    model_path = Path(silero_package.__file__).resolve().parent / "data" / "silero_vad_op18_ifless.onnx"
    model = create_silero_vad_model(
        backend="raw_onnx_ifless",
        onnx_model_path=str(model_path),
        onnx_threads=2,
        sample_rate=SILERO_RATE,
        chunk_samples=SILERO_CHUNK,
    )
    results = [
        build_video(
            args.corpus_root.resolve(), args.output_root.resolve(), video_id,
            probe_window_seconds=args.probe_window_seconds,
            clear_window_seconds=args.clear_window_seconds,
            cadence_seconds=args.cadence_seconds,
            model=model,
        )
        for video_id in args.video_id
    ]
    print(json.dumps({"status": "complete", "videos": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
