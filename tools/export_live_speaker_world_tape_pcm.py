"""Export dense-corpus source audio as decoder-independent float32 WAV files.

The dense live-window cache is keyed by the exact float32 PCM produced on the
machine that built it.  Compressed files such as MP3 can decode to slightly
different float samples on another platform even when their file hash is
identical.  This tool freezes the already-validated corpus PCM in IEEE-float
WAV so a real GUI run and an inference-free replay can consume the same bits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import load_audio_file


EXPORT_CONTRACT_ID = "whospeaks.live_world_tape_pcm.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pcm_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validated_export(path: Path, expected_pcm_sha256: str) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    audio, sample_rate = load_audio_file(path)
    actual = _pcm_sha256(audio)
    if actual != expected_pcm_sha256:
        return None
    return {
        "export_path": str(path.resolve()),
        "export_file_sha256": _sha256_file(path),
        "decoded_pcm_sha256": actual,
        "sample_rate": int(sample_rate),
        "samples": int(audio.size),
        "duration_seconds": float(audio.size / float(sample_rate)),
        "reused": True,
    }


def export_video(corpus_root: Path, output_root: Path, video_id: str) -> dict[str, Any]:
    source_path = corpus_root / "videos" / video_id / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    expected_pcm_sha256 = str(source["decoded_pcm_sha256"])
    output = output_root / f"{video_id}.audio.f32.wav"
    reusable = _validated_export(output, expected_pcm_sha256)
    if reusable is not None:
        return {"video_id": video_id, "source": source, **reusable}

    input_path = Path(str(source["audio_path_at_creation"])).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    audio, sample_rate = load_audio_file(input_path)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    actual_pcm_sha256 = _pcm_sha256(audio)
    if actual_pcm_sha256 != expected_pcm_sha256:
        raise RuntimeError(
            f"Decoded PCM drift for {video_id}: expected {expected_pcm_sha256}, "
            f"got {actual_pcm_sha256}."
        )

    temporary = output.with_suffix(output.suffix + ".partial")
    sf.write(str(temporary), audio, int(sample_rate), format="WAV", subtype="FLOAT")
    exported, exported_rate = load_audio_file(temporary)
    exported_pcm_sha256 = _pcm_sha256(exported)
    if int(exported_rate) != int(sample_rate) or exported_pcm_sha256 != expected_pcm_sha256:
        raise RuntimeError(
            f"Float-WAV round-trip drift for {video_id}: expected {expected_pcm_sha256}, "
            f"got {exported_pcm_sha256} at {exported_rate} Hz."
        )
    os.replace(temporary, output)
    return {
        "video_id": video_id,
        "source": source,
        "export_path": str(output.resolve()),
        "export_file_sha256": _sha256_file(output),
        "decoded_pcm_sha256": exported_pcm_sha256,
        "sample_rate": int(exported_rate),
        "samples": int(exported.size),
        "duration_seconds": float(exported.size / float(exported_rate)),
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze dense-corpus PCM as cross-platform float32 WAV files."
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = args.corpus_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos: list[dict[str, Any]] = []
    for index, video_id in enumerate(args.video_id, 1):
        print(f"[{index}/{len(args.video_id)}] exporting {video_id}", flush=True)
        videos.append(export_video(corpus_root, output_root, str(video_id)))
    manifest = {
        "contract_id": EXPORT_CONTRACT_ID,
        "corpus_root": str(corpus_root),
        "output_root": str(output_root),
        "video_count": len(videos),
        "videos": videos,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
