from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import load_audio_file
from embeddings.embedding_providers import RemotePreparedEmbeddingProvider, create_single_embedding_provider
from embeddings.live_shifting_window_corpus import _embed_window
from window.live_speaker_replay import load_cached_live_window_block


PARITY_ID = "fresh_cached_single_provider_vector_parity_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fresh Linux embeddings with frozen dense-cache rows")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider-endpoint", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    source = json.loads((corpus_root / "videos" / args.video_id / "source.json").read_text(encoding="utf-8"))
    audio_path = args.media_root.resolve() / str(source["audio_filename"])
    if _sha256_file(audio_path) != source["audio_file_sha256"]:
        raise RuntimeError("Compressed source hash mismatch")
    audio, sample_rate = load_audio_file(audio_path, int(source["sample_rate"]))
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm_hash = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
    if pcm_hash != source["decoded_pcm_sha256"]:
        raise RuntimeError("Decoded PCM hash mismatch")

    block = load_cached_live_window_block(
        corpus_root, args.provider, args.video_id, args.window_seconds
    )
    candidates = np.flatnonzero(np.asarray(block.valid, dtype=bool))
    if candidates.size < args.sample_count:
        raise RuntimeError("Not enough valid cached rows")
    positions = np.linspace(0, candidates.size - 1, args.sample_count, dtype=np.int64)
    indices = candidates[positions]
    provider_started = time.perf_counter()
    provider = (
        RemotePreparedEmbeddingProvider(args.provider_endpoint, args.provider, args.device)
        if args.provider_endpoint
        else create_single_embedding_provider(args.provider, args.device)
    )
    provider_load_seconds = time.perf_counter() - provider_started
    window_samples = round(args.window_seconds * sample_rate)
    rows: list[dict[str, Any]] = []
    for index in indices:
        right = round(float(block.media_times[index]) * sample_rate) - int(source.get("source_start_samples") or 0)
        result = _embed_window(
            provider,
            audio[right - window_samples:right],
            sample_rate=sample_rate,
            min_embed_seconds=0.5,
        )
        if result["embedding"] is None:
            raise RuntimeError(f"Fresh embedding failed at row {index}: {result['error']}")
        fresh = np.asarray(result["embedding"], dtype=np.float32)
        cached = np.asarray(block.embeddings[index], dtype=np.float32)
        delta = fresh - cached
        rows.append({
            "tick_index": int(index),
            "media_time": float(block.media_times[index]),
            "cosine_similarity": float(np.dot(fresh, cached)),
            "max_abs_difference": float(np.max(np.abs(delta))),
            "mean_abs_difference": float(np.mean(np.abs(delta))),
            "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(fresh, cached, rtol=1e-5, atol=1e-6)),
            "fresh_latency_ms": float(result["latency_ms"]),
        })
    payload = {
        "parity_id": PARITY_ID,
        "video_id": args.video_id,
        "provider": args.provider,
        "provider_backend": "server" if args.provider_endpoint else "local",
        "window_seconds": args.window_seconds,
        "sample_count": len(rows),
        "source_audio_sha256": source["audio_file_sha256"],
        "decoded_pcm_sha256": pcm_hash,
        "provider_load_seconds": provider_load_seconds,
        "vector_allclose": all(row["allclose_rtol_1e-5_atol_1e-6"] for row in rows),
        "minimum_cosine_similarity": min(row["cosine_similarity"] for row in rows),
        "maximum_abs_difference": max(row["max_abs_difference"] for row in rows),
        "decision_exact_match": False,
        "decision_exact_match_reason": "No causal production profile/decision trace exists yet; vector evidence must not satisfy the combined gate.",
        "samples": rows,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["vector_allclose"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
