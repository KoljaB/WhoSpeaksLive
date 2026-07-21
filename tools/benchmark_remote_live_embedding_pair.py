"""Measure Windows-to-server latency for one live short/long embedding pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "vendor"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common.audio_utils import load_audio_file
from embeddings.embedding_providers import RemotePreparedEmbeddingProvider


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--provider", default="speechbrain_resnet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--short-window-seconds", type=float, default=0.7)
    parser.add_argument("--long-window-seconds", type=float, default=1.5)
    parser.add_argument("--interval-seconds", type=float, default=0.4)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audio, sample_rate = load_audio_file(args.audio.resolve(), 16_000)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    long_samples = round(args.long_window_seconds * sample_rate)
    short_samples = round(args.short_window_seconds * sample_rate)
    if audio.size < long_samples:
        raise RuntimeError("Audio is shorter than the long benchmark window")
    provider = RemotePreparedEmbeddingProvider(
        args.endpoint, args.provider, args.device, timeout_seconds=30.0
    )
    short_latencies: list[float] = []
    long_latencies: list[float] = []
    pair_latencies: list[float] = []
    usable = audio.size - long_samples
    for index in range(max(1, args.pairs)):
        right = long_samples + (index * round(args.interval_seconds * sample_rate)) % max(1, usable)
        pair_started = time.perf_counter()
        started = time.perf_counter()
        provider.embed(audio[right - short_samples:right], sample_rate)
        short_latencies.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        provider.embed(audio[right - long_samples:right], sample_rate)
        long_latencies.append((time.perf_counter() - started) * 1000.0)
        pair_latencies.append((time.perf_counter() - pair_started) * 1000.0)
    report = {
        "endpoint": args.endpoint,
        "provider": args.provider,
        "pair_count": len(pair_latencies),
        "interval_budget_ms": args.interval_seconds * 1000.0,
        "short_mean_ms": statistics.fmean(short_latencies),
        "long_mean_ms": statistics.fmean(long_latencies),
        "pair_mean_ms": statistics.fmean(pair_latencies),
        "pair_p95_ms": _percentile(pair_latencies, 95.0),
        "pair_max_ms": max(pair_latencies),
        "pair_p95_budget_fraction": _percentile(pair_latencies, 95.0) / (args.interval_seconds * 1000.0),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
