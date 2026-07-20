from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import load_audio_file
from embeddings.embedding_providers import (
    RemotePreparedEmbeddingProvider,
    parse_embedding_provider_stack_specs,
)
from embeddings.live_shifting_window_corpus import _embed_window
from window.live_speaker_algorithm import (
    ALGORITHM_ID,
    LiveSpeakerAlgorithmConfig,
    compare_decision_traces,
)
from window.live_speaker_benchmark import aggregate_video_scores, score_live_speaker_decisions
from window.live_speaker_probe_scoring import read_canonical_segments
from window.live_speaker_replay import (
    CachedLiveWindowBlock,
    load_cached_live_window_block,
    load_profile_events_jsonl,
    replay_cached_live_windows,
    stack_cached_live_window_blocks,
    stack_embedding_matrices,
)


VERIFIER_ID = "fresh_linux_live_candidate_verifier_v1"


def _decode_audio_ffmpeg(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
            "-ar", str(int(sample_rate)), "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no samples from {path}")
    return audio, int(sample_rate)


def _sha256(path: Path) -> str:
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


def _cached_stack(
    corpus_root: Path,
    provider_specs: list[tuple[str, float]],
    video_id: str,
    window_seconds: float,
    provider_spec: str,
) -> CachedLiveWindowBlock:
    blocks = [
        load_cached_live_window_block(corpus_root, provider, video_id, window_seconds)
        for provider, _weight in provider_specs
    ]
    if len(blocks) == 1:
        return blocks[0]
    return stack_cached_live_window_blocks(
        blocks,
        [weight for _provider, weight in provider_specs],
        provider=provider_spec,
    )


def _fresh_block(
    *,
    corpus_root: Path,
    media_root: Path,
    input_root: Path,
    provider_specs: list[tuple[str, float]],
    providers: dict[str, Any],
    provider_spec: str,
    video_id: str,
    window_seconds: float,
) -> tuple[CachedLiveWindowBlock, dict[str, Any]]:
    cached = _cached_stack(corpus_root, provider_specs, video_id, window_seconds, provider_spec)
    source_path = corpus_root / "videos" / video_id / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    audio_path = media_root / str(source["audio_filename"])
    if _sha256(audio_path) != str(source["audio_file_sha256"]):
        raise RuntimeError(f"Compressed source hash mismatch for {video_id}")
    audio, sample_rate = load_audio_file(audio_path, int(source["sample_rate"]))
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm_hash = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
    if pcm_hash != str(source["decoded_pcm_sha256"]):
        raise RuntimeError(f"Decoded PCM hash mismatch for {video_id}")

    speech = np.asarray(np.load(input_root / video_id / "speech_gate.u1.npy", allow_pickle=False), dtype=bool)
    probes = np.asarray(np.load(input_root / video_id / "probe_schedule.u1.npy", allow_pickle=False), dtype=bool)
    attempted = probes & speech & np.asarray(cached.valid, dtype=bool)
    dimensions = [
        int(load_cached_live_window_block(corpus_root, provider, video_id, window_seconds).embeddings.shape[1])
        for provider, _weight in provider_specs
    ]
    embeddings = np.zeros((cached.media_times.shape[0], sum(dimensions)), dtype=np.float32)
    valid = np.zeros(cached.media_times.shape[0], dtype=bool)
    window_samples = int(round(float(window_seconds) * sample_rate))
    source_start = int(source.get("source_start_samples") or 0)
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    for ordinal, index in enumerate(np.flatnonzero(attempted), 1):
        right = int(round(float(cached.media_times[index]) * sample_rate)) - source_start
        component_rows: list[np.ndarray] = []
        failed = False
        for provider_name, weight in provider_specs:
            result = _embed_window(
                providers[provider_name],
                audio[right - window_samples:right],
                sample_rate=sample_rate,
                min_embed_seconds=0.5,
            )
            latencies.append(float(result["latency_ms"]))
            vector = result["embedding"]
            if vector is None:
                failures.append({
                    "tick_index": int(index),
                    "provider": provider_name,
                    "error": str(result["error"]),
                })
                failed = True
                break
            component_rows.append(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        if not failed:
            embeddings[index] = stack_embedding_matrices(
                component_rows,
                [weight for _provider, weight in provider_specs],
            )[0]
            valid[index] = True
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == int(np.count_nonzero(attempted)):
            print(
                f"[fresh-live] {video_id} {window_seconds:.1f}s "
                f"{ordinal}/{int(np.count_nonzero(attempted))}",
                flush=True,
            )
    block = CachedLiveWindowBlock(
        provider=provider_spec,
        video_id=video_id,
        window_seconds=float(window_seconds),
        media_times=np.asarray(cached.media_times),
        embeddings=embeddings,
        valid=valid,
        raw_rms=np.asarray(cached.raw_rms),
        sample_rate=cached.sample_rate,
    )
    return block, {
        "video_id": video_id,
        "window_seconds": float(window_seconds),
        "scheduled_speech_probe_count": int(np.count_nonzero(attempted)),
        "fresh_embedding_count": int(np.count_nonzero(valid)),
        "failure_count": len(failures),
        "failures": failures[:20],
        "mean_component_latency_ms": float(np.mean(latencies)) if latencies else None,
        "max_component_latency_ms": float(np.max(latencies)) if latencies else None,
        "source_audio_sha256": source["audio_file_sha256"],
        "decoded_pcm_sha256": pcm_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freshly embed every scheduled speech probe on Linux and verify accepted live candidates."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--provider-endpoint", default="http://127.0.0.1:8660")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    champion_path = args.champion.resolve()
    champion = json.loads(champion_path.read_text(encoding="utf-8-sig"))
    provider_spec = "+".join(
        f"{provider}={float(weight):g}"
        for provider, weight in spec["baseline"]["provider_weights"].items()
        if float(weight) > 0.0
    )
    provider_specs = [
        (provider, float(weight))
        for provider, weight in parse_embedding_provider_stack_specs(provider_spec)
        if float(weight) > 0.0
    ]
    videos = list(dict.fromkeys(spec["split"]["search"] + spec["split"]["validation"]))
    baseline = {
        "candidate_id": "baseline",
        "window_seconds": float(spec["baseline"]["probe_window_seconds"]),
        "algorithm_config": dict(spec["baseline"]["algorithm_config"]),
    }
    candidates = [baseline] + [dict(row) for row in champion.get("accepted_steps") or []]
    if champion.get("candidate_id") and all(
        str(row.get("candidate_id")) != str(champion["candidate_id"]) for row in candidates
    ):
        candidates.append({
            "candidate_id": champion["candidate_id"],
            "window_seconds": champion["window_seconds"],
            "algorithm_config": champion["algorithm_config"],
        })
    unique_windows = sorted({float(row["window_seconds"]) for row in candidates})

    providers = {
        provider: RemotePreparedEmbeddingProvider(
            args.provider_endpoint,
            provider,
            args.device,
        )
        for provider, _weight in provider_specs
    }
    fresh_blocks: dict[tuple[str, float], CachedLiveWindowBlock] = {}
    generation_reports: list[dict[str, Any]] = []
    for window in unique_windows:
        for video_id in videos:
            block, report = _fresh_block(
                corpus_root=args.corpus_root.resolve(),
                media_root=args.media_root.resolve(),
                input_root=args.input_root.resolve(),
                provider_specs=provider_specs,
                providers=providers,
                provider_spec=provider_spec,
                video_id=video_id,
                window_seconds=window,
            )
            fresh_blocks[(video_id, window)] = block
            generation_reports.append(report)

    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        window = float(candidate["window_seconds"])
        config = LiveSpeakerAlgorithmConfig(**candidate["algorithm_config"])
        scores: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        for video_id in videos:
            root = args.input_root.resolve() / video_id
            profiles = load_profile_events_jsonl(root / "production_stack.profiles.jsonl")
            speech = np.load(root / "speech_gate.u1.npy", allow_pickle=False)
            probes = np.load(root / "probe_schedule.u1.npy", allow_pickle=False)
            releases = np.load(root / "release_gate.u1.npy", allow_pickle=False)
            fresh_decisions = replay_cached_live_windows(
                fresh_blocks[(video_id, window)],
                profiles,
                speech,
                probes,
                releases,
                config=config,
            )
            cached_decisions = replay_cached_live_windows(
                _cached_stack(
                    args.corpus_root.resolve(), provider_specs, video_id, window, provider_spec
                ),
                profiles,
                speech,
                probes,
                releases,
                config=config,
            )
            scores[video_id] = score_live_speaker_decisions(
                fresh_decisions,
                read_canonical_segments(root / "canonical_diarization.json"),
                profiles,
            )
            parity[video_id] = compare_decision_traces(fresh_decisions, cached_decisions)
        candidate_reports.append({
            "candidate_id": candidate["candidate_id"],
            "window_seconds": window,
            "algorithm_config": asdict(config),
            "aggregate": aggregate_video_scores(scores.values()),
            "per_video": scores,
            "cached_decision_parity": parity,
            "all_cached_decisions_exact": all(row["exact_match"] for row in parity.values()),
        })

    baseline_report = candidate_reports[0]
    final_report = next(
        row for row in reversed(candidate_reports)
        if str(row["candidate_id"]) == str(champion.get("candidate_id"))
    ) if champion.get("candidate_id") else baseline_report
    steps_non_regressing = all(
        float(right["aggregate"]["global_score"]) + 1e-9
        >= float(left["aggregate"]["global_score"])
        for left, right in zip(candidate_reports, candidate_reports[1:])
    )
    improved = (
        float(final_report["aggregate"]["global_score"])
        > float(baseline_report["aggregate"]["global_score"]) + 1e-9
    )
    all_exact = all(row["all_cached_decisions_exact"] for row in candidate_reports)
    passed = improved and steps_non_regressing and all_exact and all(
        int(row["failure_count"]) == 0 for row in generation_reports
    )
    payload = {
        "schema_version": 1,
        "verifier_id": VERIFIER_ID,
        "algorithm_id": ALGORITHM_ID,
        "provider": provider_spec,
        "videos": videos,
        "baseline_fresh_score": baseline_report["aggregate"]["global_score"],
        "champion_fresh_score": final_report["aggregate"]["global_score"],
        "fresh_score_delta": round(
            float(final_report["aggregate"]["global_score"])
            - float(baseline_report["aggregate"]["global_score"]),
            6,
        ),
        "every_accepted_step_non_regressing_live": steps_non_regressing,
        "all_cached_decisions_exact": all_exact,
        "generation_reports": generation_reports,
        "candidate_reports": candidate_reports,
        "passed": passed,
    }
    _atomic_json(args.output.resolve(), payload)
    champion["fresh_live_verified"] = passed
    champion["fresh_live_verification"] = {
        "path": str(args.output.resolve()),
        "baseline_score": payload["baseline_fresh_score"],
        "champion_score": payload["champion_fresh_score"],
        "score_delta": payload["fresh_score_delta"],
        "all_cached_decisions_exact": all_exact,
        "every_accepted_step_non_regressing_live": steps_non_regressing,
    }
    champion["status"] = "LIVE_VERIFIED_CHAMPION" if passed else "REJECTED_BY_FRESH_LIVE"
    _atomic_json(champion_path, champion)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
