"""Add a saved live session and corrected Markdown reference to the regression corpus.

The saved session contains the exact sentence boundaries and stacked embeddings
seen by the live controller.  This tool splits the stacked vector back into its
normalized provider blocks, so cached regression replay remains identical to
the production provider stack without another ASR or embedding RPC.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "elevenlabs_scribe_27"
HEADING_RE = re.compile(
    r"\*\*SPEAKER_(?P<speaker>\d+) "
    r"\[(?P<start>\d\d:\d\d:\d\d(?:\.\d+)?) - (?P<end>\d\d:\d\d:\d\d(?:\.\d+)?)\]\*\*"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def parse_corrected_markdown(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        raise ValueError(f"No speaker headings found in {path}")
    segments: list[dict[str, Any]] = []
    for ordinal, match in enumerate(matches, start=1):
        body_end = matches[ordinal].start() if ordinal < len(matches) else len(text)
        body = " ".join(text[match.end() : body_end].strip().split())
        start = _seconds(match.group("start"))
        end = _seconds(match.group("end"))
        segments.append(
            {
                "segment_id": f"SEGMENT_{ordinal:04d}",
                "speaker_id": f"SPEAKER_{int(match.group('speaker'))}",
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(max(0.0, end - start), 4),
                "text": body,
            }
        )
    return segments


def _provider_dimensions(dataset_root: Path, providers: list[str]) -> dict[str, int]:
    for video_dir in sorted((dataset_root / "videos").iterdir()):
        embedding_dir = video_dir / "live_window" / "embeddings"
        paths = [embedding_dir / f"{provider}.npz" for provider in providers]
        if not all(path.is_file() for path in paths):
            continue
        return {
            provider: int(np.load(path)["embeddings"].shape[1])
            for provider, path in zip(providers, paths)
        }
    raise FileNotFoundError(f"Could not infer provider dimensions for {providers}")


def _decode_session_embeddings(
    embeddings_payload: dict[str, Any],
    dimensions: dict[str, int],
) -> dict[str, np.ndarray]:
    providers = list(dimensions)
    rows: dict[str, list[np.ndarray]] = {provider: [] for provider in providers}
    for record in sorted(embeddings_payload["records"], key=lambda item: int(item["index"])):
        vector = np.frombuffer(base64.b64decode(record["embedding_b64"]), dtype="<f4").copy()
        if int(vector.size) != sum(dimensions.values()):
            raise ValueError(
                f"Stacked embedding has {vector.size} dimensions, expected {sum(dimensions.values())}."
            )
        offset = 0
        for provider in providers:
            block = vector[offset : offset + dimensions[provider]]
            offset += dimensions[provider]
            norm = float(np.linalg.norm(block))
            if norm <= 0.0:
                raise ValueError(f"Empty {provider} block in sentence {record['index']}")
            rows[provider].append((block / norm).astype(np.float32))
    return {provider: np.stack(vectors) for provider, vectors in rows.items()}


def _sentence_row(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "index",
        "text",
        "start",
        "end",
        "spoken_word_seconds",
        "audio_length_seconds",
        "speech_audio_ratio",
        "window_left",
        "window_right",
        "next_left",
        "words",
        "first_word_start",
        "last_word_end",
        "next_word_start",
        "gap_to_next_word_seconds",
        "boundary_strategy",
        "sentence_boundary_pre_padding_seconds",
        "sentence_boundary_post_padding_seconds",
        "sentence_boundary_gap_ratio",
    }
    sentence = {key: row[key] for key in allowed if key in row}
    sentence["index"] = int(row["index"])
    sentence["audio_file"] = None
    sentence["embedding_audio_seconds"] = float(
        row.get("audio_length_seconds") or float(row.get("end") or 0.0) - float(row.get("start") or 0.0)
    )
    return sentence


def _update_dataset_index(dataset_root: Path, metadata: dict[str, Any]) -> None:
    manifest_path = dataset_root / "manifest.json"
    manifest = _read_json(manifest_path)
    videos = [
        item
        for item in manifest.get("videos") or []
        if item.get("canonical_video_id") != metadata["canonical_video_id"]
    ]
    videos.append(metadata)
    videos.sort(key=lambda item: str(item["canonical_video_id"]).lower())
    provider_count = int(manifest.get("provider_count") or 0)
    blacklist = _read_json(dataset_root / "blacklist.json").get("excluded_videos") or []
    manifest["videos"] = videos
    manifest["video_count"] = len(videos)
    manifest["default_evaluation_video_count"] = len(videos) - len(blacklist)
    manifest["evaluation_blacklist_count"] = len(blacklist)
    manifest["baseline_video_count"] = len(videos)
    manifest["baseline_embedding_complete_video_count"] = sum(
        int(item["baseline"].get("embedding_provider_count") or 0) == provider_count
        for item in videos
    )
    manifest["live_sentence_cache_video_count"] = sum(
        int(item["live_window"].get("sentence_count") or 0) > 0
        for item in videos
    )
    manifest["live_embedding_complete_video_count"] = sum(
        int(item["live_window"].get("embedding_provider_count") or 0) == provider_count
        for item in videos
    )
    manifest["live_embedding_incomplete_video_count"] = sum(
        int(item["live_window"].get("sentence_count") or 0) > 0
        and int(item["live_window"].get("embedding_provider_count") or 0) != provider_count
        for item in videos
    )
    manifest["total_baseline_segments"] = sum(
        int(item["baseline"].get("segment_count") or 0) for item in videos
    )
    manifest["total_live_sentences"] = sum(
        int(item["live_window"].get("sentence_count") or 0) for item in videos
    )
    _write_json(manifest_path, manifest)

    coverage_path = dataset_root / "coverage.csv"
    with coverage_path.open("r", newline="", encoding="utf-8-sig") as handle:
        coverage = list(csv.DictReader(handle))
    row = {
        "video_id": metadata["canonical_video_id"],
        "baseline_key": metadata["baseline_key"],
        "baseline_segments": metadata["baseline"]["segment_count"],
        "baseline_embedding_providers": metadata["baseline"]["embedding_provider_count"],
        "live_sentences": metadata["live_window"]["sentence_count"],
        "live_audio_clips": metadata["live_window"]["audio_clip_count"],
        "live_embedding_providers": metadata["live_window"]["embedding_provider_count"],
        "live_source_kind": metadata["live_window"]["source_kind"],
        "complete_live_embeddings": str(metadata["live_window"]["complete_provider_embeddings"]).lower(),
        "title": metadata.get("title") or "",
    }
    coverage = [item for item in coverage if item.get("video_id") != row["video_id"]]
    coverage.append({key: str(value) for key, value in row.items()})
    coverage.sort(key=lambda item: item["video_id"].lower())
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(coverage)

    readme_path = dataset_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("# ElevenLabs Scribe 27-Video Dataset", "# ElevenLabs Scribe Regression Dataset")
    readme = readme.replace(
        "This is the canonical local dataset root for the 27-video ElevenLabs Scribe baseline corpus.",
        "This is the canonical local regression corpus for ElevenLabs references and live-compatible cached replay.",
    )
    replacements = {
        "Videos": manifest["video_count"],
        "Baseline videos with canonical diarization": manifest["baseline_video_count"],
        "Baseline videos with all provider embeddings": manifest["baseline_embedding_complete_video_count"],
        "Baseline segments": manifest["total_baseline_segments"],
        "Live videos with sentence/audio caches": manifest["live_sentence_cache_video_count"],
        "Live videos with all provider embeddings": manifest["live_embedding_complete_video_count"],
        "Live videos still missing provider embeddings": manifest["live_embedding_incomplete_video_count"],
        "Live sentences": manifest["total_live_sentences"],
        "Default optimizer evaluation videos": manifest["default_evaluation_video_count"],
        "Evaluation blacklist entries": manifest["evaluation_blacklist_count"],
    }
    for label, value in replacements.items():
        readme = re.sub(rf"\| {re.escape(label)} \| [^|]+\|", f"| {label} | {value} |", readme)
    readme = readme.replace(
        "No live embedding gaps remain.",
        (
            "`6BuK09sWn9s` intentionally carries the three providers from the current live stack "
            "(`espnet_ecapa_wavlm_joint`, `speechbrain_resnet`, and `wespeaker_campplus`); "
            "the other historical provider caches are not required for its regression score."
        ),
    )
    readme = re.sub(
        r"\d+ videos currently have complete 15-provider live sentence embeddings\.",
        f"{manifest['live_embedding_complete_video_count']} videos currently have complete 15-provider live sentence embeddings.",
        readme,
    )
    readme_path.write_text(readme, encoding="utf-8")


def add_video(args: argparse.Namespace) -> Path:
    session_dir = args.session_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    video_dir = dataset_root / "videos" / args.video_id
    transcript = _read_json(session_dir / "transcript.json")
    session_embeddings = _read_json(session_dir / "embeddings.json")
    manifest = _read_json(session_dir / "manifest.json")
    providers = [token.rsplit("=", 1)[0] for token in session_embeddings["embedding_provider"].split("+")]
    dimensions = _provider_dimensions(dataset_root, providers)
    provider_embeddings = _decode_session_embeddings(session_embeddings, dimensions)
    rows = sorted(transcript["rows"], key=lambda item: int(item["index"]))
    if any(matrix.shape[0] != len(rows) for matrix in provider_embeddings.values()):
        raise ValueError("Transcript and embedding row counts differ.")

    segments = parse_corrected_markdown(args.reference_md)
    baseline_dir = video_dir / "baseline"
    live_dir = video_dir / "live_window"
    embeddings_dir = live_dir / "embeddings"
    source_dir = baseline_dir / "source_files"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.reference_md, source_dir / args.reference_md.name)
    shutil.copy2(args.reference_md, baseline_dir / "best_diarization.md")
    _write_json(baseline_dir / "best_diarization.json", segments)
    with (baseline_dir / "best_diarization.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segments[0]))
        writer.writeheader()
        writer.writerows(segments)
    canonical = {
        "schema": "whospeaks.canonical_diarization.v1",
        "media": {
            "source_url": args.source_url,
            "title": manifest.get("title"),
            "duration_sec": manifest.get("duration_seconds"),
        },
        "provenance": {
            "kind": "manually_corrected_elevenlabs_reference",
            "source": str(args.reference_md),
            "live_session": str(session_dir),
        },
        "speakers": [
            {"speaker_id": f"SPEAKER_{index}", "display_name": f"Speaker {index}"}
            for index in sorted({int(segment["speaker_id"].split("_")[-1]) for segment in segments})
        ],
        "segments": segments,
        "words": [],
    }
    _write_json(baseline_dir / "canonical_diarization.json", canonical)

    live_dir.mkdir(parents=True, exist_ok=True)
    with (live_dir / "sentences.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_sentence_row(row), ensure_ascii=False) + "\n")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for provider, matrix in provider_embeddings.items():
        np.savez_compressed(embeddings_dir / f"{provider}.npz", embeddings=matrix)
    _write_json(
        live_dir / "manifest.json",
        {
            "name": f"{args.video_id}_livewindow_saved_session",
            "video_id": args.video_id,
            "source_url": args.source_url,
            "sentence_count": len(rows),
            "embedding_provider": session_embeddings["embedding_provider"],
            "embedding_providers": providers,
            "source_session": str(session_dir),
        },
    )
    metadata = {
        "canonical_video_id": args.video_id,
        "baseline_key": args.video_id,
        "source_url": args.source_url,
        "title": manifest.get("title"),
        "duration_sec": manifest.get("duration_seconds"),
        "baseline": {
            "canonical_diarization": "baseline/canonical_diarization.json",
            "source_files": "baseline/source_files",
            "segment_count": len(segments),
            "speaker_count": len(canonical["speakers"]),
            "embedding_provider_count": 0,
            "embedding_providers": [],
        },
        "live_window": {
            "source_kind": "saved_live_session",
            "source_path": str(session_dir),
            "sentences": "live_window/sentences.jsonl",
            "audio_dir": None,
            "sentence_count": len(rows),
            "audio_clip_count": 0,
            "embedding_provider_count": len(providers),
            "embedding_providers": providers,
            "complete_provider_embeddings": False,
        },
    }
    _write_json(video_dir / "metadata.json", metadata)
    _update_dataset_index(dataset_root, metadata)
    return video_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--reference-md", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    output = add_video(parse_args())
    print(output)
