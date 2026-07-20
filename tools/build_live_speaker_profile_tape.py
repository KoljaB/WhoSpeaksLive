from __future__ import annotations

import argparse
from collections import Counter
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

from embeddings.embedding_providers import parse_embedding_provider_stack_specs
from window.live_speaker_replay import STACKED_CACHE_POLICY_ID, stack_embedding_matrices
from window.window_validation_replay import make_cached_replay_args, replay_cached_window_diarizer


PROFILE_BUILDER_ID = "causal_final_sentence_to_live_profile_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sentences(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No sentence rows found in {path}")
    return rows


def _load_provider_matrix(cache_dir: Path, provider: str) -> tuple[np.ndarray, Path]:
    path = cache_dir / "embeddings" / f"{provider}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "embeddings" not in archive:
            raise ValueError(f"Missing embeddings array in {path}")
        matrix = np.asarray(archive["embeddings"], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional embedding matrix in {path}")
    return matrix, path


def _load_stack(cache_dir: Path, provider_spec: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    specs = parse_embedding_provider_stack_specs(provider_spec)
    matrices: list[np.ndarray] = []
    weights: list[float] = []
    identities: list[dict[str, Any]] = []
    for provider, weight in specs:
        if float(weight) <= 0.0:
            continue
        matrix, path = _load_provider_matrix(cache_dir, provider)
        matrices.append(matrix)
        weights.append(float(weight))
        identities.append({
            "provider": provider,
            "weight": float(weight),
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "shape": list(matrix.shape),
        })
    return stack_embedding_matrices(matrices, weights), identities


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build chronological live-profile snapshots from real cached final-sentence "
            "boundaries and the production final/live provider stacks."
        )
    )
    parser.add_argument("--sentence-cache", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--final-provider", required=True)
    parser.add_argument("--live-provider", required=True)
    parser.add_argument("--availability-lag-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    cache_dir = args.sentence_cache.resolve()
    sentences_path = cache_dir / "sentences.jsonl"
    sentences = _read_sentences(sentences_path)
    final_embeddings, final_inputs = _load_stack(cache_dir, args.final_provider)
    live_embeddings, live_inputs = _load_stack(cache_dir, args.live_provider)
    if len(sentences) != int(final_embeddings.shape[0]) or len(sentences) != int(live_embeddings.shape[0]):
        raise ValueError("Sentence and embedding row counts do not match")

    replay_args = make_cached_replay_args().with_updates(
        embedding_provider=args.final_provider,
        live_speaker_embedding_provider=args.live_provider,
    )
    replay = replay_cached_window_diarizer(
        sentences,
        final_embeddings,
        replay_args,
        defer_speaker_refinement=False,
        live_profile_embeddings=live_embeddings,
        live_profile_provider=args.live_provider,
        profile_availability_lag_seconds=max(0.0, float(args.availability_lag_seconds)),
    )
    profile_events = sorted(
        replay.profile_events,
        key=lambda row: (
            float(row["available_at"]),
            int(row.get("profile_generation") or 0),
            str(row["speaker_id"]),
        ),
    )
    if not profile_events:
        raise RuntimeError("Final-sentence replay produced no live profile events")
    output = args.output.resolve()
    _write_jsonl_atomic(output, profile_events)

    assigned = Counter(
        str(row.get("assigned_speaker"))
        for row in replay.final_payloads
        if row.get("assigned_speaker")
    )
    summary = {
        "schema_version": 1,
        "builder_id": PROFILE_BUILDER_ID,
        "video_id": args.video_id,
        "sentence_count": len(sentences),
        "profile_event_count": len(profile_events),
        "assigned_sentence_counts": dict(sorted(assigned.items())),
        "first_available_at": float(profile_events[0]["available_at"]),
        "last_available_at": float(profile_events[-1]["available_at"]),
        "profile_provider": args.live_provider,
        "final_provider": args.final_provider,
        "availability_lag_seconds": max(0.0, float(args.availability_lag_seconds)),
        "stack_policy_id": STACKED_CACHE_POLICY_ID,
        "sentence_input": {"path": str(sentences_path), "sha256": _sha256(sentences_path)},
        "final_embedding_inputs": final_inputs,
        "live_embedding_inputs": live_inputs,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    summary_output = args.summary_output.resolve() if args.summary_output else output.with_suffix(".summary.json")
    _write_json_atomic(summary_output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
