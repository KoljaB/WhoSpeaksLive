"""Build remote sentence embeddings for live sentence-boundary datasets."""

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

from common.audio_utils import SAMPLE_RATE
from embeddings.embedding_providers import RemoteEmbeddingClient


DEFAULT_INPUT_ROOT = (
    Path("data")
    / "live_sentence_boundaries"
    / "live_window_corpus_60_90_cuda_missing_1x"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "_sentence_embeddings"
DEFAULT_REMOTE_EMBEDDINGS_URL = os.environ.get(
    "WHOSPEAKS_REMOTE_EMBEDDINGS_URL",
    "http://192.168.178.22:8660",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _provider_dir_name(provider: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in provider)
    return safe.strip("_") or "provider"


def _load_remote_providers(base_url: str, timeout_seconds: float) -> list[str]:
    with urlopen(f"{base_url.rstrip('/')}/providers", timeout=min(timeout_seconds, 30.0)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    providers = payload.get("providers")
    if not isinstance(providers, list):
        raise RuntimeError("Remote embeddings server did not return a provider list.")
    return [str(item["id"]) for item in providers if isinstance(item, dict) and item.get("id")]


def _dataset_dirs(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "sentences.jsonl").is_file()
    )


def _read_sentences(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number} is not valid JSON.") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
    return rows


def _video_id_from_folder(folder: Path) -> str:
    suffix = "_livewindow_60_90_cuda"
    if folder.name.endswith(suffix):
        return folder.name[: -len(suffix)]
    return folder.name


def _resolve_sentence_audio(folder: Path, row: dict[str, Any]) -> Path:
    raw_path = str(row.get("audio_file") or "")
    if not raw_path:
        raise RuntimeError(f"{folder.name} sentence {row.get('index')} does not have audio_file.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = folder / path
    if not path.is_file():
        raise FileNotFoundError(f"Sentence audio file does not exist: {path}")
    return path


def _float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key) or 0.0) for row in rows], dtype=np.float32)


def _int_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[int] = []
    for ordinal, row in enumerate(rows):
        value = row.get(key)
        values.append(int(value) if value is not None else ordinal)
    return np.asarray(values, dtype=np.int32)


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
    return metadata.get("provider") == provider and len(metadata.get("sentences") or []) == expected_count


def _update_video_manifest(folder: Path, provider: str, output_npz: Path, output_metadata: Path) -> None:
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    providers = list(manifest.get("providers") or [])
    if provider not in providers:
        providers.append(provider)
    outputs = dict(manifest.get("embedding_outputs") or {})
    outputs[provider] = {
        "npz": str(output_npz),
        "metadata": str(output_metadata),
    }
    manifest["providers"] = providers
    manifest["embedding_outputs"] = outputs
    _write_json_atomic(manifest_path, manifest)


def _embed_video(
    *,
    folder: Path,
    provider: str,
    client: RemoteEmbeddingClient,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = _read_sentences(folder / "sentences.jsonl")
    npz_path = output_dir / f"{folder.name}.embeddings.npz"
    metadata_path = output_dir / f"{folder.name}.sentences.json"
    if _is_complete(npz_path, metadata_path, len(rows), provider):
        _update_video_manifest(folder, provider, npz_path, metadata_path)
        return {
            "video": _video_id_from_folder(folder),
            "folder": folder.name,
            "sentences": len(rows),
            "status": "skipped",
            "npz": str(npz_path),
            "metadata": str(metadata_path),
        }

    vectors: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        audio_path = _resolve_sentence_audio(folder, row)
        embedding = client.embed_wav(audio_path)
        vectors.append(embedding.astype(np.float32, copy=False))
        metadata_rows.append(
            {
                "index": int(row.get("index", ordinal - 1)),
                "text": str(row.get("text") or ""),
                "start": float(row.get("start") or 0.0),
                "end": float(row.get("end") or row.get("start") or 0.0),
                "audio_length_seconds": float(row.get("audio_length_seconds") or 0.0),
                "embedding_audio_seconds": float(row.get("embedding_audio_seconds") or 0.0),
                "audio_file": str(row.get("audio_file") or ""),
                "word_count": len(row.get("words") or []),
            }
        )
        if ordinal == 1 or ordinal % 50 == 0 or ordinal == len(rows):
            print(
                f"[provider] {provider}: {folder.name} sentence {ordinal}/{len(rows)}",
                flush=True,
            )

    embeddings = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.empty((0, 0), dtype=np.float32)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with tmp_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=embeddings,
            sentence_index=_int_array(metadata_rows, "index"),
            start_sec=_float_array(metadata_rows, "start"),
            end_sec=_float_array(metadata_rows, "end"),
            audio_length_seconds=_float_array(metadata_rows, "audio_length_seconds"),
            embedding_audio_seconds=_float_array(metadata_rows, "embedding_audio_seconds"),
        )
    tmp_npz.replace(npz_path)
    elapsed = time.perf_counter() - started
    _write_json_atomic(
        metadata_path,
        {
            "provider": provider,
            "video": _video_id_from_folder(folder),
            "folder": folder.name,
            "sentence_file": str(folder / "sentences.jsonl"),
            "sample_rate": SAMPLE_RATE,
            "embedding_shape": list(embeddings.shape),
            "elapsed_seconds": elapsed,
            "sentences": metadata_rows,
        },
    )
    _update_video_manifest(folder, provider, npz_path, metadata_path)
    return {
        "video": _video_id_from_folder(folder),
        "folder": folder.name,
        "sentences": len(rows),
        "status": "created",
        "elapsed_seconds": elapsed,
        "npz": str(npz_path),
        "metadata": str(metadata_path),
    }


def build_provider(
    *,
    provider: str,
    dataset_dirs: list[Path],
    output_root: Path,
    remote_url: str,
    device: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    provider_dir = output_root / _provider_dir_name(provider)
    started = time.perf_counter()
    client = RemoteEmbeddingClient(
        base_url=remote_url,
        provider=provider,
        device=device,
        timeout_seconds=timeout_seconds,
    )

    _write_json_atomic(
        provider_dir / "manifest.json",
        {
            "status": "in_progress",
            "provider": provider,
            "backend": "remote",
            "remote_url": remote_url,
            "device": device,
            "started_at_epoch": time.time(),
        },
    )
    print(f"[provider] {provider}: loading", flush=True)
    load_result = client.load()
    print(f"[provider] {provider}: loaded", flush=True)

    results: list[dict[str, Any]] = []
    for ordinal, folder in enumerate(dataset_dirs, start=1):
        result = _embed_video(
            folder=folder,
            provider=provider,
            client=client,
            output_dir=provider_dir,
        )
        results.append(result)
        print(
            f"[provider] {provider}: {ordinal}/{len(dataset_dirs)} "
            f"{folder.name} {result['status']} sentences={result['sentences']}",
            flush=True,
        )

    summary = {
        "status": "ok",
        "provider": provider,
        "backend": "remote",
        "remote_url": remote_url,
        "device": device,
        "load_result": load_result,
        "video_count": len(results),
        "sentence_count": sum(int(item["sentences"]) for item in results),
        "created_video_count": sum(1 for item in results if item["status"] == "created"),
        "skipped_video_count": sum(1 for item in results if item["status"] == "skipped"),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": results,
    }
    _write_json_atomic(provider_dir / "manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build remote embeddings for live sentence-boundary sentence WAVs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--remote-embeddings-url", default=DEFAULT_REMOTE_EMBEDDINGS_URL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Per remote request timeout. Failed providers are recorded and skipped.",
    )
    parser.add_argument(
        "--providers",
        default="",
        help="Comma-separated provider ids. Empty means all providers reported by the remote server.",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--max-providers", type=int, default=0, help="Optional smoke-test limit.")
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
    dataset_dirs = _dataset_dirs(input_root)
    if not dataset_dirs:
        raise RuntimeError(f"No live sentence dataset folders found under {input_root}")
    if args.max_videos > 0:
        dataset_dirs = dataset_dirs[: args.max_videos]

    if args.providers.strip():
        providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    else:
        providers = _load_remote_providers(args.remote_embeddings_url, args.timeout_seconds)
    if args.max_providers > 0:
        providers = providers[: args.max_providers]
    if not providers:
        raise RuntimeError("No embedding providers selected.")

    corpus_sentence_count = sum(len(_read_sentences(folder / "sentences.jsonl")) for folder in dataset_dirs)
    print(
        f"[corpus] videos={len(dataset_dirs)} sentences={corpus_sentence_count} "
        f"providers={len(providers)} output={output_root}",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for ordinal, provider in enumerate(providers, start=1):
        print(f"[corpus] provider {ordinal}/{len(providers)}: {provider}", flush=True)
        try:
            summaries.append(
                build_provider(
                    provider=provider,
                    dataset_dirs=dataset_dirs,
                    output_root=output_root,
                    remote_url=args.remote_embeddings_url,
                    device=args.device,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        except Exception as exc:
            summary = {
                "status": "failed",
                "provider": provider,
                "backend": "remote",
                "remote_url": args.remote_embeddings_url,
                "device": args.device,
                "video_count": 0,
                "sentence_count": 0,
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
            "backend": "remote",
            "remote_url": args.remote_embeddings_url,
            "device": args.device,
            "provider_count": len(summaries),
            "failed_provider_count": failed_provider_count,
            "video_count": len(dataset_dirs),
            "corpus_sentence_count": corpus_sentence_count,
            "total_embedding_count": sum(int(item["sentence_count"]) for item in summaries),
            "providers": summaries,
        },
    )
    return 1 if failed_provider_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
