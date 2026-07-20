from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_FILES = {
    "embeddings": "embeddings.f32.npy",
    "attempted": "attempted.u1.npy",
    "valid": "valid.u1.npy",
    "raw_rms": "raw_rms.f32.npy",
    "raw_peak": "raw_peak.f32.npy",
    "trimmed_samples": "trimmed_samples.i32.npy",
    "prepared_samples": "prepared_samples.i32.npy",
    "latency_ms": "latency_ms.f32.npy",
}


def validate_job(job_dir: Path) -> dict[str, Any]:
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if job.get("status") != "complete":
        raise ValueError(f"job is not complete: {job.get('status')!r}")
    partials = sorted(str(path.relative_to(job_dir)) for path in job_dir.rglob("*.partial"))
    if partials:
        raise ValueError(f"job still contains partial arrays: {partials[:3]}")

    expected = {int(key): int(value) for key, value in job["expected_by_window_samples"].items()}
    length_dirs = sorted((job_dir / "lengths").glob("*ms"))
    if len(length_dirs) != len(expected):
        raise ValueError(f"expected {len(expected)} length directories, found {len(length_dirs)}")

    total_rows = 0
    dimensions: set[int] = set()
    norm_min = float("inf")
    norm_max = float("-inf")
    latency_sum = 0.0
    latency_count = 0
    checked_lengths: list[float] = []

    for length_dir in length_dirs:
        metadata = json.loads((length_dir / "metadata.json").read_text(encoding="utf-8"))
        window_samples = int(metadata["window_samples"])
        rows = expected.pop(window_samples, None)
        if rows is None:
            raise ValueError(f"unexpected window length {window_samples} samples")
        if metadata.get("status") != "complete":
            raise ValueError(f"{length_dir.name} is not complete")

        arrays = {name: np.load(length_dir / filename, mmap_mode="r") for name, filename in ARRAY_FILES.items()}
        embeddings = arrays["embeddings"]
        dimension = int(metadata["embedding_dimension"])
        tick_count = int(metadata["tick_count"])
        if embeddings.shape != (tick_count, dimension) or embeddings.dtype != np.float32:
            raise ValueError(f"invalid embedding array for {length_dir.name}: {embeddings.shape} {embeddings.dtype}")
        dimensions.add(dimension)

        for name, array in arrays.items():
            if name != "embeddings" and array.shape != (tick_count,):
                raise ValueError(f"invalid {name} shape for {length_dir.name}: {array.shape}")
        attempted = np.asarray(arrays["attempted"], dtype=bool)
        valid = np.asarray(arrays["valid"], dtype=bool)
        first_eligible = int(metadata["first_eligible_tick_index"])
        if int(attempted.sum()) != rows or int(valid.sum()) != rows:
            raise ValueError(f"attempted/valid counts disagree for {length_dir.name}")
        if np.any(attempted[:first_eligible]) or not np.all(attempted[first_eligible:]):
            raise ValueError(f"eligible tick mask is not contiguous for {length_dir.name}")
        if np.any(valid & ~attempted):
            raise ValueError(f"valid rows exist without attempts for {length_dir.name}")
        if not np.isfinite(embeddings[valid]).all() or not np.isnan(embeddings[~attempted]).all():
            raise ValueError(f"non-finite embedding values for {length_dir.name}")
        if not np.isfinite(arrays["latency_ms"][attempted]).all() or np.any(arrays["latency_ms"][attempted] < 0):
            raise ValueError(f"invalid latency values for {length_dir.name}")
        if not np.isfinite(arrays["raw_rms"][attempted]).all() or not np.isfinite(arrays["raw_peak"][attempted]).all():
            raise ValueError(f"invalid audio statistics for {length_dir.name}")
        if np.any(arrays["trimmed_samples"][attempted] < 0) or np.any(arrays["prepared_samples"][attempted] < 0):
            raise ValueError(f"invalid preprocessing sample counts for {length_dir.name}")

        norms = np.linalg.norm(embeddings[valid], axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
            raise ValueError(f"embeddings are not unit-normalized for {length_dir.name}")
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        latency_sum += float(np.asarray(arrays["latency_ms"][attempted], dtype=np.float64).sum())
        latency_count += rows
        total_rows += rows
        checked_lengths.append(float(metadata["window_seconds"]))

    if expected:
        raise ValueError(f"missing window lengths: {sorted(expected)}")
    if total_rows != int(job["expected_embeddings"]):
        raise ValueError(f"expected {job['expected_embeddings']} rows, validated {total_rows}")
    if total_rows != int(job["successful_embeddings"]) or int(job["failed_embeddings"]) != 0:
        raise ValueError("job success/failure counters disagree with finalized arrays")

    return {
        "status": "valid",
        "job_dir": str(job_dir.resolve()),
        "length_count": len(checked_lengths),
        "window_seconds": checked_lengths,
        "embedding_rows": total_rows,
        "embedding_dimensions": sorted(dimensions),
        "embedding_norm_min": norm_min,
        "embedding_norm_max": norm_max,
        "latency_ms_mean": latency_sum / latency_count,
        "partial_file_count": 0,
    }


def validate_corpus(corpus_root: Path, video_id: str) -> dict[str, Any]:
    job_dirs = sorted((corpus_root / "providers").glob(f"*/videos/{video_id}"))
    if not job_dirs:
        raise ValueError(f"no provider jobs found for video {video_id!r}")
    jobs = [validate_job(path) for path in job_dirs]
    return {
        "status": "valid",
        "corpus_root": str(corpus_root.resolve()),
        "video_id": video_id,
        "provider_count": len(jobs),
        "embedding_rows": sum(int(job["embedding_rows"]) for job in jobs),
        "length_count_per_provider": sorted({int(job["length_count"]) for job in jobs}),
        "embedding_dimensions": sorted(
            {dimension for job in jobs for dimension in job["embedding_dimensions"]}
        ),
        "embedding_norm_min": min(float(job["embedding_norm_min"]) for job in jobs),
        "embedding_norm_max": max(float(job["embedding_norm_max"]) for job in jobs),
        "partial_file_count": sum(int(job["partial_file_count"]) for job in jobs),
        "providers": [path.parents[1].name for path in job_dirs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one finalized live shifting-window embedding job.")
    parser.add_argument("job_dir", type=Path, nargs="?")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--video-id", default="")
    args = parser.parse_args()
    if args.corpus_root:
        if not args.video_id:
            parser.error("--video-id is required with --corpus-root")
        result = validate_corpus(args.corpus_root, args.video_id)
    elif args.job_dir:
        result = validate_job(args.job_dir)
    else:
        parser.error("provide job_dir or --corpus-root")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
