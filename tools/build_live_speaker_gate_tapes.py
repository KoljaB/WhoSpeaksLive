from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import load_audio_file
from window.live_speech_gate import RMS_GATE_ID, rms_speech_present


TAPE_ID = "production_rms_live_gate_tape_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _scheduled_ticks(right_edges: np.ndarray, first_right: int, cadence_samples: int) -> np.ndarray:
    result = np.zeros(right_edges.shape[0], dtype=np.uint8)
    last_right: int | None = None
    for index, right in enumerate(right_edges):
        value = int(right)
        if value < first_right:
            continue
        if last_right is None or value >= last_right + cadence_samples:
            result[index] = 1
            last_right = value
    return result


def build_video(
    corpus_root: Path,
    media_root: Path,
    output_root: Path,
    video_id: str,
    *,
    probe_window_seconds: float,
    clear_window_seconds: float,
    cadence_seconds: float,
    frame_seconds: float,
    threshold: float,
    min_speech_seconds: float,
    release_every_tick: bool = False,
    release_threshold: float | None = None,
    release_min_speech_seconds: float | None = None,
) -> dict[str, Any]:
    video_root = corpus_root / "videos" / video_id
    source = json.loads((video_root / "source.json").read_text(encoding="utf-8"))
    timeline = json.loads((video_root / "timeline" / "metadata.json").read_text(encoding="utf-8"))
    sample_rate = int(source["sample_rate"])
    if sample_rate != int(timeline["sample_rate"]):
        raise RuntimeError(f"Sample-rate mismatch for {video_id}")
    audio_path = media_root / str(source["audio_filename"])
    if _sha256_file(audio_path) != source["audio_file_sha256"]:
        raise RuntimeError(f"Compressed source hash mismatch for {video_id}")
    audio, decoded_rate = load_audio_file(audio_path, sample_rate)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if decoded_rate != sample_rate or int(audio.size) != int(source["decoded_samples"]):
        raise RuntimeError(f"Decoded source geometry mismatch for {video_id}")
    decoded_hash = _sha256_bytes(np.ascontiguousarray(audio).tobytes())
    if decoded_hash != source["decoded_pcm_sha256"]:
        raise RuntimeError(f"Decoded PCM hash mismatch for {video_id}")

    right_edges_path = video_root / "timeline" / "right_edges.i64.npy"
    right_edges = np.load(right_edges_path, mmap_mode="r", allow_pickle=False)
    if right_edges.ndim != 1 or int(right_edges.shape[0]) != int(timeline["tick_count"]):
        raise RuntimeError(f"Timeline geometry mismatch for {video_id}")
    timeline_hash = _sha256_bytes(np.ascontiguousarray(right_edges).tobytes())
    if timeline_hash != timeline["timeline_sha256"]:
        raise RuntimeError(f"Timeline content hash mismatch for {video_id}")

    probe_samples = round(probe_window_seconds * sample_rate)
    clear_samples = round(clear_window_seconds * sample_rate)
    cadence_samples = round(cadence_seconds * sample_rate)
    schedule = _scheduled_ticks(right_edges, probe_samples, cadence_samples)
    speech = np.zeros(right_edges.shape[0], dtype=np.uint8)
    release = np.zeros(right_edges.shape[0], dtype=np.uint8)
    for index in np.flatnonzero(schedule):
        right = int(right_edges[index])
        probe_left = max(0, right - probe_samples)
        probe_audio = audio[probe_left:right]
        has_probe_speech = rms_speech_present(
            probe_audio,
            sample_rate,
            frame_seconds=frame_seconds,
            threshold=threshold,
            min_speech_seconds=min_speech_seconds,
        )
        speech[index] = int(has_probe_speech)
    release_indices = (
        np.flatnonzero(right_edges >= clear_samples)
        if release_every_tick else np.flatnonzero(schedule)
    )
    for index in release_indices:
        right = int(right_edges[index])
        clear_left = max(0, right - clear_samples)
        clear_audio = audio[clear_left:right]
        has_clear_speech = rms_speech_present(
            clear_audio,
            sample_rate,
            frame_seconds=frame_seconds,
            threshold=threshold if release_threshold is None else release_threshold,
            min_speech_seconds=(
                min_speech_seconds
                if release_min_speech_seconds is None else release_min_speech_seconds
            ),
        )
        release[index] = int(not has_clear_speech)

    target = output_root / video_id
    _atomic_npy(target / "probe_schedule.u1.npy", schedule)
    _atomic_npy(target / "speech_gate.u1.npy", speech)
    _atomic_npy(target / "release_gate.u1.npy", release)
    shared_gate_source = ROOT / "src/window/live_speech_gate.py"
    metadata = {
        "tape_id": TAPE_ID,
        "video_id": video_id,
        "source_audio_sha256": source["audio_file_sha256"],
        "decoded_pcm_sha256": decoded_hash,
        "timeline_sha256": timeline_hash,
        "sample_rate": sample_rate,
        "tick_count": int(right_edges.shape[0]),
        "probe_window_seconds": probe_window_seconds,
        "clear_window_seconds": clear_window_seconds,
        "cadence_seconds": cadence_seconds,
        "cadence_policy": "first_full_window_then_first_0.2s_grid_tick_at_least_cadence_later",
        "speech_backend": "rms",
        "speech_gate_id": RMS_GATE_ID,
        "vad_frame_seconds": frame_seconds,
        "vad_speech_rms_threshold": threshold,
        "live_speaker_probe_min_speech_seconds": min_speech_seconds,
        "release_rms_threshold": threshold if release_threshold is None else release_threshold,
        "release_min_speech_seconds": (
            min_speech_seconds
            if release_min_speech_seconds is None else release_min_speech_seconds
        ),
        "shared_gate_source": str(shared_gate_source),
        "shared_gate_source_sha256": _sha256_file(shared_gate_source),
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "scheduled_probe_count": int(np.count_nonzero(schedule)),
        "speech_probe_count": int(np.count_nonzero(speech)),
        "release_signal_count": int(np.count_nonzero(release)),
        "release_cadence_policy": "every_timeline_tick" if release_every_tick else "scheduled_probe_only",
    }
    _atomic_json(target / "gate_tape.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build causal gate tapes with the production RMS gate")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True)
    parser.add_argument("--probe-window-seconds", type=float, default=1.0)
    parser.add_argument("--clear-window-seconds", type=float, default=1.0)
    parser.add_argument("--cadence-seconds", type=float, default=0.75)
    parser.add_argument("--frame-seconds", type=float, default=0.03)
    parser.add_argument("--rms-threshold", type=float, default=0.003)
    parser.add_argument("--min-speech-seconds", type=float, default=0.15)
    parser.add_argument("--release-every-tick", action="store_true")
    parser.add_argument("--release-rms-threshold", type=float)
    parser.add_argument("--release-min-speech-seconds", type=float)
    args = parser.parse_args()
    results = [
        build_video(
            args.corpus_root.resolve(), args.media_root.resolve(), args.output_root.resolve(), video_id,
            probe_window_seconds=args.probe_window_seconds,
            clear_window_seconds=args.clear_window_seconds,
            cadence_seconds=args.cadence_seconds,
            frame_seconds=args.frame_seconds,
            threshold=args.rms_threshold,
            min_speech_seconds=args.min_speech_seconds,
            release_every_tick=args.release_every_tick,
            release_threshold=args.release_rms_threshold,
            release_min_speech_seconds=args.release_min_speech_seconds,
        )
        for video_id in args.video_id
    ]
    print(json.dumps({"status": "complete", "videos": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
