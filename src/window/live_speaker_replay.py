"""Replay adapters for the dense continuous-audio live-window corpus."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from window.live_speaker_algorithm import (
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerDecision,
    LiveSpeakerStep,
    SpeakerProfileEvent,
)


STACKED_CACHE_POLICY_ID = "normalized_component_concat_v1"


def blend_live_speaker_embeddings(
    short_embedding: np.ndarray,
    long_embedding: np.ndarray,
    long_weight: float,
) -> np.ndarray:
    """Blend aligned short and long live vectors using the replay policy."""

    short_vector = np.asarray(short_embedding, dtype=np.float32).reshape(-1)
    long_vector = np.asarray(long_embedding, dtype=np.float32).reshape(-1)
    if short_vector.shape != long_vector.shape:
        raise ValueError("Short and long live embeddings must have the same shape")
    weight = max(0.0, min(1.0, float(long_weight)))
    blended = ((1.0 - weight) * short_vector + weight * long_vector).astype(
        np.float32, copy=False
    )
    norm = float(np.linalg.norm(blended))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Blended live embedding must have a finite positive norm")
    return (blended / norm).astype(np.float32, copy=False)


def stack_embedding_matrices(
    matrices: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Apply the production stacked-provider policy to aligned embedding rows."""

    if not matrices or len(matrices) != len(weights):
        raise ValueError("Embedding matrices and weights must be non-empty and aligned")
    positive = [
        (np.asarray(matrix, dtype=np.float32), float(weight))
        for matrix, weight in zip(matrices, weights)
        if float(weight) > 0.0
    ]
    if not positive:
        raise ValueError("At least one embedding weight must be positive")
    row_count = int(positive[0][0].shape[0])
    if any(matrix.ndim != 2 or int(matrix.shape[0]) != row_count for matrix, _ in positive):
        raise ValueError("All embedding matrices must be two-dimensional and row-aligned")
    components: list[np.ndarray] = []
    for matrix, weight in positive:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.where(np.isfinite(norms) & (norms > 0.0), norms, 1.0)
        components.append((matrix / safe).astype(np.float32, copy=False) * weight)
    stacked = np.concatenate(components, axis=1).astype(np.float32, copy=False)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    safe = np.where(np.isfinite(norms) & (norms > 0.0), norms, 1.0)
    return (stacked / safe).astype(np.float32, copy=False)


@dataclass(frozen=True)
class CachedLiveWindowBlock:
    provider: str
    video_id: str
    window_seconds: float
    media_times: np.ndarray
    embeddings: np.ndarray
    valid: np.ndarray
    raw_rms: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        rows = int(self.media_times.shape[0])
        if self.embeddings.shape[0] != rows or self.valid.shape[0] != rows or self.raw_rms.shape[0] != rows:
            raise ValueError("All cached live-window arrays must share the timeline row count")


def load_cached_live_window_block(
    corpus_root: Path,
    provider: str,
    video_id: str,
    window_seconds: float,
) -> CachedLiveWindowBlock:
    root = Path(corpus_root)
    source = json.loads((root / "videos" / video_id / "source.json").read_text(encoding="utf-8"))
    timeline_dir = root / "videos" / video_id / "timeline"
    timeline = json.loads((timeline_dir / "metadata.json").read_text(encoding="utf-8"))
    sample_rate = int(timeline["sample_rate"])
    source_start = int(timeline.get("source_start_samples") or source.get("source_start_samples") or 0)
    right_edges = np.load(timeline_dir / "right_edges.i64.npy", mmap_mode="r")
    length_name = f"{round(float(window_seconds) * 1000):04d}ms"
    length_dir = root / "providers" / provider / "videos" / video_id / "lengths" / length_name
    embeddings = np.load(length_dir / "embeddings.f32.npy", mmap_mode="r")
    valid = np.load(length_dir / "valid.u1.npy", mmap_mode="r").astype(bool, copy=False)
    raw_rms = np.load(length_dir / "raw_rms.f32.npy", mmap_mode="r")
    return CachedLiveWindowBlock(
        provider=provider,
        video_id=video_id,
        window_seconds=float(window_seconds),
        media_times=(right_edges.astype(np.float64) + source_start) / float(sample_rate),
        embeddings=embeddings,
        valid=valid,
        raw_rms=raw_rms,
        sample_rate=sample_rate,
    )


def stack_cached_live_window_blocks(
    blocks: Sequence[CachedLiveWindowBlock],
    weights: Sequence[float],
    *,
    provider: str,
) -> CachedLiveWindowBlock:
    """Reconstruct production ``StackedEmbeddingProvider`` vectors from caches."""

    if len(blocks) < 2:
        raise ValueError("At least two cached provider blocks are required")
    if len(blocks) != len(weights):
        raise ValueError("Cached provider weights must match provider blocks")
    positive = [(block, float(weight)) for block, weight in zip(blocks, weights) if float(weight) > 0.0]
    if len(positive) < 2:
        raise ValueError("At least two cached provider weights must be positive")
    reference = positive[0][0]
    for block, _weight in positive[1:]:
        if block.video_id != reference.video_id or block.window_seconds != reference.window_seconds:
            raise ValueError("Cached provider blocks must describe the same video and window length")
        if block.sample_rate != reference.sample_rate or not np.array_equal(block.media_times, reference.media_times):
            raise ValueError("Cached provider blocks must share the exact timeline")

    valid = np.logical_and.reduce([np.asarray(block.valid, dtype=bool) for block, _weight in positive])
    dimensions = [int(block.embeddings.shape[1]) for block, _weight in positive]
    embeddings = np.zeros((reference.media_times.shape[0], sum(dimensions)), dtype=np.float32)
    for row in np.flatnonzero(valid):
        components: list[np.ndarray] = []
        for block, weight in positive:
            vector = np.asarray(block.embeddings[row], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0.0:
                valid[row] = False
                components = []
                break
            components.append((vector / norm).astype(np.float32, copy=False) * weight)
        if components:
            stacked = np.concatenate(components).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(stacked))
            if not np.isfinite(norm) or norm <= 0.0:
                valid[row] = False
            else:
                embeddings[row] = (stacked / norm).astype(np.float32, copy=False)

    return CachedLiveWindowBlock(
        provider=provider,
        video_id=reference.video_id,
        window_seconds=reference.window_seconds,
        media_times=np.asarray(reference.media_times),
        embeddings=embeddings,
        valid=valid,
        raw_rms=np.asarray(reference.raw_rms),
        sample_rate=reference.sample_rate,
    )


def load_profile_events_jsonl(path: Path) -> list[SpeakerProfileEvent]:
    """Read versioned causal profile snapshots; inline embeddings are intentional."""

    events: list[SpeakerProfileEvent] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        events.append(SpeakerProfileEvent(
            available_at=float(value["available_at"]),
            speaker_id=str(value["speaker_id"]),
            centroid=np.asarray(value["centroid"], dtype=np.float32),
            speech_seconds=float(value.get("speech_seconds") or 0.0),
            sentence_count=int(value.get("sentence_count") or 1),
            generation=int(value.get("profile_generation") or value.get("generation") or 0),
            sentence_start=(
                float(value["sentence_start"])
                if value.get("sentence_start") is not None
                else None
            ),
            sentence_end=(
                float(value["sentence_end"])
                if value.get("sentence_end") is not None
                else None
            ),
        ))
    return events


def replay_cached_live_windows(
    block: CachedLiveWindowBlock,
    profile_events: Iterable[SpeakerProfileEvent],
    speech_mask: Sequence[bool] | np.ndarray,
    probe_mask: Sequence[bool] | np.ndarray,
    release_mask: Sequence[bool] | np.ndarray,
    *,
    config: LiveSpeakerAlgorithmConfig | None = None,
) -> list[LiveSpeakerDecision]:
    """Run cached vectors through the exact source-independent live algorithm.

    ``speech_mask`` is required rather than inferred from canonical labels or RMS.
    It must be produced by the same causal speech gate used in production.  This
    makes accidental oracle scoring impossible and exposes missing VAD parity.
    """

    speech = np.asarray(speech_mask, dtype=bool).reshape(-1)
    probes = np.asarray(probe_mask, dtype=bool).reshape(-1)
    releases = np.asarray(release_mask, dtype=bool).reshape(-1)
    rows = block.media_times.shape[0]
    if speech.shape[0] != rows:
        raise ValueError("speech_mask must have exactly one value per corpus tick")
    if probes.shape[0] != rows:
        raise ValueError("probe_mask must have exactly one value per corpus tick")
    if releases.shape[0] != rows:
        raise ValueError("release_mask must have exactly one value per corpus tick")
    algorithm = CausalLiveSpeakerAlgorithm(config=config, profile_events=profile_events)
    results: list[LiveSpeakerDecision] = []
    for index, media_time in enumerate(block.media_times):
        scheduled = bool(probes[index])
        embedding = block.embeddings[index] if scheduled and bool(block.valid[index]) else None
        results.append(algorithm.step(LiveSpeakerStep(
            media_time=float(media_time),
            speech=bool(speech[index]),
            embedding=embedding,
            duration_seconds=block.window_seconds,
            probe_scheduled=scheduled,
            release_signal=bool(releases[index]),
            skipped_reason=(
                "" if embedding is not None else
                "not_a_scheduled_probe" if not scheduled else
                "cached_embedding_invalid"
            ),
        )))
    return results


def replay_cached_live_windows_dual(
    short_block: CachedLiveWindowBlock,
    long_block: CachedLiveWindowBlock,
    profile_events: Iterable[SpeakerProfileEvent],
    speech_mask: Sequence[bool] | np.ndarray,
    probe_mask: Sequence[bool] | np.ndarray,
    release_mask: Sequence[bool] | np.ndarray,
    *,
    long_weight: float,
    config: LiveSpeakerAlgorithmConfig | None = None,
) -> list[LiveSpeakerDecision]:
    """Use a short acquisition window immediately and blend longer context when available."""

    if short_block.video_id != long_block.video_id:
        raise ValueError("Dual live blocks must describe the same video")
    if short_block.sample_rate != long_block.sample_rate or not np.array_equal(
        short_block.media_times, long_block.media_times
    ):
        raise ValueError("Dual live blocks must share the exact timeline")
    if short_block.embeddings.shape != long_block.embeddings.shape:
        raise ValueError("Dual live blocks must have the same embedding shape")
    weight = max(0.0, min(1.0, float(long_weight)))
    speech = np.asarray(speech_mask, dtype=bool).reshape(-1)
    probes = np.asarray(probe_mask, dtype=bool).reshape(-1)
    releases = np.asarray(release_mask, dtype=bool).reshape(-1)
    rows = int(short_block.media_times.shape[0])
    if any(values.shape[0] != rows for values in (speech, probes, releases)):
        raise ValueError("All dual replay masks must have one value per timeline tick")
    algorithm = CausalLiveSpeakerAlgorithm(config=config, profile_events=profile_events)
    results: list[LiveSpeakerDecision] = []
    for index, media_time in enumerate(short_block.media_times):
        scheduled = bool(probes[index])
        embedding: np.ndarray | None = None
        duration = float(short_block.window_seconds)
        skipped_reason = "not_a_scheduled_probe" if not scheduled else "cached_embedding_invalid"
        if scheduled:
            short_valid = bool(short_block.valid[index])
            long_valid = bool(long_block.valid[index])
            if short_valid and long_valid:
                short_vector = np.asarray(short_block.embeddings[index], dtype=np.float32)
                long_vector = np.asarray(long_block.embeddings[index], dtype=np.float32)
                try:
                    embedding = blend_live_speaker_embeddings(
                        short_vector, long_vector, weight
                    )
                except ValueError:
                    embedding = None
                duration = float(long_block.window_seconds)
                skipped_reason = "" if embedding is not None else "cached_embedding_invalid"
            elif short_valid:
                embedding = np.asarray(short_block.embeddings[index], dtype=np.float32)
                skipped_reason = ""
            elif long_valid:
                embedding = np.asarray(long_block.embeddings[index], dtype=np.float32)
                duration = float(long_block.window_seconds)
                skipped_reason = ""
        results.append(algorithm.step(LiveSpeakerStep(
            media_time=float(media_time),
            speech=bool(speech[index]),
            embedding=embedding,
            duration_seconds=duration,
            probe_scheduled=scheduled,
            release_signal=bool(releases[index]),
            skipped_reason=skipped_reason,
        )))
    return results


def run_live_embedding_steps(
    steps: Iterable[LiveSpeakerStep],
    profile_events: Iterable[SpeakerProfileEvent],
    *,
    config: LiveSpeakerAlgorithmConfig | None = None,
) -> list[LiveSpeakerDecision]:
    """Production-side adapter: fresh embeddings, same chronological core."""

    algorithm = CausalLiveSpeakerAlgorithm(config=config, profile_events=profile_events)
    return [algorithm.step(step) for step in steps]
