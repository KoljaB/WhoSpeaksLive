"""Build sentence-level embedding files for diarized baseline videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import urlopen

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.audio_utils import SAMPLE_RATE, load_audio_file, normalize_vector, pad_audio, trim_silence
from embeddings.embedding_providers import RemoteEmbeddingClient, create_embedding_provider


DEFAULT_INPUT_ROOT = Path("data") / "baselines" / "elevenlabs_scribe"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "_sentence_embeddings"
DEFAULT_REMOTE_EMBEDDINGS_URL = os.environ.get("WHOSPEAKS_REMOTE_EMBEDDINGS_URL", "http://127.0.0.1:8660")
LOCAL_PROVIDER_IDS = [
    "resemblyzer",
    "speechbrain_ecapa",
    "speechbrain_resnet",
    "speechbrain_xvector",
    "wespeaker_campplus",
    "wespeaker_resnet34_lm_onnx",
    "pyannote_wespeaker_resnet34_lm",
    "pyannote_embedding",
    "speaker3d_campplus",
    "speaker3d_eres2netv2",
    "nemo_titanet_large",
    "wavlm_base_sv",
    "jungjee_rawnet3",
    "espnet_rawnet3",
    "espnet_ecapa_wavlm_joint",
]


class LocalEmbeddingClient:
    """Small adapter with the same load/embed shape as RemoteEmbeddingClient."""

    def __init__(self, provider: str, device: str) -> None:
        self.provider = provider
        self.device = device
        self._provider: Any | None = None

    def load(self) -> dict[str, Any]:
        if self._provider is None:
            self._provider = create_embedding_provider(self.provider, self.device)
        return {"ok": True, "backend": "local"}

    def embed_audio(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        provider = self._provider
        if provider is None:
            self.load()
            provider = self._provider
        if provider is None:
            raise RuntimeError("Local embedding provider did not load.")
        prepared = pad_audio(
            trim_silence(np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate),
            0.5,
            sample_rate,
        )
        return normalize_vector(provider.embed(prepared, sample_rate))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _provider_dir_name(provider: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in provider).strip("_") or "provider"


def _load_remote_providers(base_url: str, timeout_seconds: float) -> list[str]:
    with urlopen(f"{base_url.rstrip('/')}/providers", timeout=min(timeout_seconds, 30.0)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    providers = payload.get("providers")
    if not isinstance(providers, list):
        raise RuntimeError("Remote embeddings server did not return a provider list.")
    return [str(item["id"]) for item in providers if isinstance(item, dict) and item.get("id")]


def _canonical_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.glob("*/*.canonical_diarization.json")
        if path.parent.name != "_source_lists"
    )


def _resolve_audio_path(canonical: Path, payload: dict[str, Any]) -> Path:
    raw_path = str((payload.get("media") or {}).get("audio_file") or "")
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            return path
    candidates = [
        *canonical.parent.glob("*.wav"),
        *canonical.parent.glob("*.flac"),
        *canonical.parent.glob("*.mp3"),
        *canonical.parent.glob("*.m4a"),
    ]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No audio file found for {canonical}")


def _segment_audio_bounds(
    segment: dict[str, Any],
    duration_seconds: float,
    *,
    min_slice_seconds: float,
) -> tuple[float, float]:
    start = max(0.0, float(segment.get("start_sec") or 0.0))
    end = max(start, float(segment.get("end_sec") or start))
    min_slice_seconds = max(0.0, float(min_slice_seconds))
    if min_slice_seconds > 0.0 and end - start < min_slice_seconds:
        center = (start + end) / 2.0
        half = min_slice_seconds / 2.0
        start = max(0.0, center - half)
        end = min(duration_seconds, center + half)
        if end - start < min_slice_seconds:
            start = max(0.0, min(start, duration_seconds - min_slice_seconds))
            end = min(duration_seconds, start + min_slice_seconds)
    return start, max(start, end)


def _embed_audio_window(
    *,
    client: RemoteEmbeddingClient,
    audio: np.ndarray,
    sample_rate: int,
    max_chunk_seconds: float,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    max_chunk_seconds = max(0.0, float(max_chunk_seconds))
    if max_chunk_seconds <= 0.0 or len(audio) <= int(round(max_chunk_seconds * sample_rate)):
        return client.embed_audio(audio, sample_rate)

    max_samples = max(1, int(round(max_chunk_seconds * sample_rate)))
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for left in range(0, len(audio), max_samples):
        right = min(len(audio), left + max_samples)
        chunk = audio[left:right]
        if not len(chunk):
            continue
        vectors.append(client.embed_audio(chunk, sample_rate).astype(np.float32, copy=False))
        weights.append(float(len(chunk)))
    if not vectors:
        return client.embed_audio(audio, sample_rate)
    return normalize_vector(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights, dtype=np.float32)))


def _is_complete(npz_path: Path, metadata_path: Path, expected_count: int, provider: str) -> bool:
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with np.load(npz_path) as data:
            embeddings = data["embeddings"]
            if int(embeddings.shape[0]) != expected_count:
                return False
        metadata = _read_json(metadata_path)
    except Exception:
        return False
    return metadata.get("provider") == provider and len(metadata.get("segments") or []) == expected_count


def _embed_video(
    *,
    canonical: Path,
    provider: str,
    client: RemoteEmbeddingClient,
    output_dir: Path,
    min_slice_seconds: float,
    max_embed_chunk_seconds: float,
) -> dict[str, Any]:
    video_key = canonical.parent.name
    payload = _read_json(canonical)
    segments = [item for item in payload.get("segments") or [] if isinstance(item, dict)]
    npz_path = output_dir / f"{video_key}.embeddings.npz"
    metadata_path = output_dir / f"{video_key}.segments.json"
    if _is_complete(npz_path, metadata_path, len(segments), provider):
        return {
            "video": video_key,
            "segments": len(segments),
            "status": "skipped",
            "npz": str(npz_path),
            "metadata": str(metadata_path),
        }

    audio_path = _resolve_audio_path(canonical, payload)
    audio, sample_rate = load_audio_file(audio_path, SAMPLE_RATE)
    duration_seconds = len(audio) / float(sample_rate or SAMPLE_RATE)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    started = time.perf_counter()

    for index, segment in enumerate(segments):
        audio_start, audio_end = _segment_audio_bounds(
            segment,
            duration_seconds,
            min_slice_seconds=min_slice_seconds,
        )
        left = int(round(audio_start * sample_rate))
        right = int(round(audio_end * sample_rate))
        chunk = np.asarray(audio[left:right], dtype=np.float32)
        embedding = _embed_audio_window(
            client=client,
            audio=chunk,
            sample_rate=sample_rate,
            max_chunk_seconds=max_embed_chunk_seconds,
        )
        vectors.append(embedding.astype(np.float32, copy=False))
        rows.append(
            {
                "index": index,
                "segment_id": str(segment.get("segment_id") or f"seg_{index:04d}"),
                "speaker_id": str(segment.get("speaker_id") or ""),
                "start_sec": float(segment.get("start_sec") or 0.0),
                "end_sec": float(segment.get("end_sec") or segment.get("start_sec") or 0.0),
                "duration_sec": float(segment.get("duration_sec") or 0.0),
                "audio_start_sec": audio_start,
                "audio_end_sec": audio_end,
                "text": str(segment.get("text") or ""),
            }
        )

    if vectors:
        embeddings = np.stack(vectors).astype(np.float32, copy=False)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embeddings,
            start_sec=np.asarray([row["start_sec"] for row in rows], dtype=np.float32),
            end_sec=np.asarray([row["end_sec"] for row in rows], dtype=np.float32),
            audio_start_sec=np.asarray([row["audio_start_sec"] for row in rows], dtype=np.float32),
            audio_end_sec=np.asarray([row["audio_end_sec"] for row in rows], dtype=np.float32),
        )
    tmp_npz.replace(npz_path)
    _write_json_atomic(
        metadata_path,
        {
            "provider": provider,
            "video": video_key,
            "canonical": str(canonical),
            "audio_file": str(audio_path),
            "sample_rate": sample_rate,
            "embedding_shape": list(embeddings.shape),
            "elapsed_seconds": time.perf_counter() - started,
            "segments": rows,
        },
    )
    return {
        "video": video_key,
        "segments": len(segments),
        "status": "created",
        "elapsed_seconds": time.perf_counter() - started,
        "npz": str(npz_path),
        "metadata": str(metadata_path),
    }


def build_provider(
    *,
    provider: str,
    canonical_files: list[Path],
    output_root: Path,
    backend: str,
    remote_url: str,
    device: str,
    timeout_seconds: float,
    min_slice_seconds: float,
    max_embed_chunk_seconds: float,
) -> dict[str, Any]:
    provider_dir = output_root / _provider_dir_name(provider)
    manifest_path = provider_dir / "manifest.json"
    if backend == "local":
        client = LocalEmbeddingClient(provider=provider, device=device)
    else:
        client = RemoteEmbeddingClient(
            base_url=remote_url,
            provider=provider,
            device=device,
            timeout_seconds=timeout_seconds,
        )
    started = time.perf_counter()
    print(f"[provider] {provider}: loading", flush=True)
    load_result = client.load()
    print(f"[provider] {provider}: loaded", flush=True)
    results: list[dict[str, Any]] = []
    for ordinal, canonical in enumerate(canonical_files, start=1):
        result = _embed_video(
            canonical=canonical,
            provider=provider,
            client=client,
            output_dir=provider_dir,
            min_slice_seconds=min_slice_seconds,
            max_embed_chunk_seconds=max_embed_chunk_seconds,
        )
        results.append(result)
        print(
            f"[provider] {provider}: {ordinal}/{len(canonical_files)} "
            f"{result['video']} {result['status']} segments={result['segments']}",
            flush=True,
        )
    summary = {
        "status": "ok",
        "provider": provider,
        "backend": backend,
        "remote_url": remote_url,
        "device": device,
        "load_result": load_result,
        "video_count": len(results),
        "segment_count": sum(int(item["segments"]) for item in results),
        "created_video_count": sum(1 for item in results if item["status"] == "created"),
        "skipped_video_count": sum(1 for item in results if item["status"] == "skipped"),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": results,
    }
    _write_json_atomic(manifest_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sentence embeddings for baseline diarization videos.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("local", "remote"), default="remote")
    parser.add_argument("--remote-embeddings-url", default=DEFAULT_REMOTE_EMBEDDINGS_URL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--min-slice-seconds", type=float, default=0.10)
    parser.add_argument(
        "--max-embed-chunk-seconds",
        type=float,
        default=0.0,
        help="Split longer segment audio into fixed windows and average embeddings. Zero disables chunking.",
    )
    parser.add_argument("--providers", default="", help="Comma-separated provider ids. Empty means all remote providers.")
    parser.add_argument("--max-videos", type=int, default=0, help="Optional smoke-test limit. Zero means all videos.")
    parser.add_argument(
        "--stop-on-provider-error",
        action="store_true",
        help="Stop the batch when a provider fails instead of recording the error and continuing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    canonical_files = _canonical_files(input_root)
    if not canonical_files:
        raise RuntimeError(f"No canonical diarization files found under {input_root}")
    if args.max_videos > 0:
        canonical_files = canonical_files[: args.max_videos]
    if args.providers.strip():
        providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    elif args.backend == "local":
        providers = list(LOCAL_PROVIDER_IDS)
    else:
        providers = _load_remote_providers(args.remote_embeddings_url, args.timeout_seconds)
    print(
        f"[corpus] videos={len(canonical_files)} providers={len(providers)} output={output_root}",
        flush=True,
    )
    summaries = []
    for ordinal, provider in enumerate(providers, start=1):
        print(f"[corpus] provider {ordinal}/{len(providers)}: {provider}", flush=True)
        try:
            summaries.append(
                build_provider(
                    provider=provider,
                    canonical_files=canonical_files,
                    output_root=output_root,
                    backend=args.backend,
                    remote_url=args.remote_embeddings_url,
                    device=args.device,
                    timeout_seconds=args.timeout_seconds,
                    min_slice_seconds=args.min_slice_seconds,
                    max_embed_chunk_seconds=args.max_embed_chunk_seconds,
                )
            )
        except Exception as exc:
            summary = {
                "status": "failed",
                "provider": provider,
                "backend": args.backend,
                "remote_url": args.remote_embeddings_url,
                "device": args.device,
                "video_count": 0,
                "segment_count": 0,
                "created_video_count": 0,
                "skipped_video_count": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summaries.append(summary)
            _write_json_atomic(output_root / _provider_dir_name(provider) / "manifest.json", summary)
            print(f"[provider] {provider}: failed {type(exc).__name__}: {exc}", flush=True)
            if args.stop_on_provider_error:
                raise
    failed_provider_count = sum(1 for item in summaries if item.get("status") == "failed")
    _write_json_atomic(
        output_root / "manifest.json",
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "backend": args.backend,
            "remote_url": args.remote_embeddings_url,
            "device": args.device,
            "provider_count": len(summaries),
            "failed_provider_count": failed_provider_count,
            "video_count": len(canonical_files),
            "corpus_segment_count": sum(
                len((_read_json(path).get("segments") or [])) for path in canonical_files
            ),
            "total_embedding_count": sum(int(item["segment_count"]) for item in summaries),
            "providers": summaries,
        },
    )
    return 1 if failed_provider_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
