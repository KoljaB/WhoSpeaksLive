from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
OUTPUT_ROOT = DATASETS_ROOT / "elevenlabs_scribe_27"
MARKER = OUTPUT_ROOT / ".generated_by_build_elevenlabs_scribe_27"

BASELINE_ROOT = REPO_ROOT / "data" / "baselines" / "elevenlabs_scribe"
BASELINE_EMBEDDINGS_ROOT = BASELINE_ROOT / "_sentence_embeddings"
LIVE_COMPLETE_ROOT = (
    REPO_ROOT
    / "data"
    / "live_sentence_boundaries"
    / "live_window_corpus_60_90_cuda_complete_23"
)
LIVE_MISSING_1X_ROOT = (
    REPO_ROOT
    / "data"
    / "live_sentence_boundaries"
    / "live_window_corpus_60_90_cuda_missing_1x"
)
LIVE_MORE4_ROOT = (
    REPO_ROOT
    / "data"
    / "live_sentence_boundaries"
    / "live_window_corpus_60_90_cuda_more_4"
)
WHOSPEAKS_MORE_ROOT = Path(
    r"D:\Projekte\WhoSpeaks\tools\.window_diarize_feature_cache\live_window_corpus_60_90_cuda_more"
)
EVALUATION_BLACKLIST = {
    "version": 1,
    "excluded_videos": [
        {
            "video_id": "k1tsGGz-Qw0",
            "title": "Key & Peele - High On Potenuse",
            "source_url": "https://www.youtube.com/watch?v=k1tsGGz-Qw0",
            "scope": "score_search",
            "reason": (
                "Known bad ElevenLabs canonical diarization: the opening repeated "
                "'I wish I were/was high on pot noose' lines are labeled as the same "
                "speaker even though they are different speakers."
            ),
            "added_on": "2026-07-06",
        },
        {
            "video_id": "Dd7FixvoKBw",
            "title": "Substitute Teacher - Key & Peele",
            "source_url": "https://www.youtube.com/watch?v=Dd7FixvoKBw",
            "scope": "score_search",
            "reason": (
                "Filtered-26 tuned score 0.742946 is below the 0.78 review threshold; "
                "exclude from aggregate score/search until manually reviewed."
            ),
            "added_on": "2026-07-06",
        },
        {
            "video_id": "WNZn37Uc700",
            "title": "True Confessions with Matthew McConaughey and Hugh Grant",
            "source_url": "https://www.youtube.com/watch?v=WNZn37Uc700",
            "scope": "score_search",
            "reason": (
                "Filtered-26 tuned score 0.753477 is below the 0.78 review threshold; "
                "exclude from aggregate score/search until manually reviewed."
            ),
            "added_on": "2026-07-06",
        },
        {
            "video_id": "acbnyagl8jo",
            "title": "Margin Call meeting scene",
            "source_url": "https://www.youtube.com/watch?v=acbnyagl8jo",
            "scope": "score_search",
            "reason": (
                "Filtered-26 tuned score 0.760147 is below the 0.78 review threshold; "
                "exclude from aggregate score/search until manually reviewed."
            ),
            "added_on": "2026-07-06",
        },
        {
            "video_id": "gj7BRMuB-n4",
            "title": "True Confessions with Kate McKinnon and John Cena",
            "source_url": "https://www.youtube.com/watch?v=gj7BRMuB-n4",
            "scope": "score_search",
            "reason": (
                "Filtered-26 tuned score 0.765265 is below the 0.78 review threshold; "
                "exclude from aggregate score/search until manually reviewed."
            ),
            "added_on": "2026-07-06",
        },
        {
            "video_id": "bPpcfH_HHH8",
            "title": "NPR’s Delicious Dish: Schweddy Balls - SNL",
            "source_url": "https://www.youtube.com/watch?v=bPpcfH_HHH8",
            "scope": "score_search",
            "reason": (
                "Filtered-26 tuned score 0.789927 is below the 0.79 review threshold; "
                "exclude from aggregate score/search until manually reviewed."
            ),
            "added_on": "2026-07-06",
        }
    ],
}

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def youtube_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def canonical_file_for(baseline_dir: Path) -> Path:
    matches = sorted(baseline_dir.glob("*.canonical_diarization.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one canonical diarization file in {baseline_dir}, found {len(matches)}")
    return matches[0]


def provider_names() -> list[str]:
    return sorted(
        p.name
        for p in BASELINE_EMBEDDINGS_ROOT.iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    )


def live_source_for(video_id: str) -> tuple[Path | None, str | None]:
    candidates = [
        (LIVE_COMPLETE_ROOT / f"{video_id}_livewindow_60_90_cuda", "repo_complete_23"),
        (LIVE_MISSING_1X_ROOT / f"{video_id}_livewindow_60_90_cuda", "repo_missing_1x"),
        (LIVE_MORE4_ROOT / f"{video_id}_livewindow_60_90_cuda", "repo_more_4"),
        (WHOSPEAKS_MORE_ROOT / f"{video_id}_livewindow_60_90_cuda", "whospeaks_more"),
    ]
    for path, source_kind in candidates:
        if path.exists():
            return path, source_kind
    return None, None


def copy_best_exports(baseline_dir: Path, dst: Path) -> None:
    for source in sorted(baseline_dir.glob("*.elevenlabs_best_diarization.*")):
        suffix = "".join(source.suffixes[-1:])
        copy_file(source, dst / f"best_diarization{suffix}")


def reset_output_root() -> None:
    if OUTPUT_ROOT.exists():
        if not MARKER.exists():
            raise RuntimeError(f"Refusing to replace unmarked dataset root: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def build() -> None:
    if not BASELINE_ROOT.exists():
        raise RuntimeError(f"Missing baseline root: {BASELINE_ROOT}")
    if not BASELINE_EMBEDDINGS_ROOT.exists():
        raise RuntimeError(f"Missing baseline embeddings root: {BASELINE_EMBEDDINGS_ROOT}")

    reset_output_root()

    generated_at = datetime.now(timezone.utc).isoformat()
    providers = provider_names()
    baseline_dirs = sorted(
        p for p in BASELINE_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")
    )

    source_lists = BASELINE_ROOT / "_source_lists"
    if source_lists.exists():
        copy_tree(source_lists, OUTPUT_ROOT / "source_lists")

    videos: list[dict[str, Any]] = []
    aliases: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    for baseline_dir in baseline_dirs:
        baseline_key = baseline_dir.name
        canonical_file = canonical_file_for(baseline_dir)
        canonical = read_json(canonical_file)
        media = canonical.get("media") or {}
        source_url = media.get("source_url")
        video_id = youtube_id_from_url(source_url) or baseline_key
        video_dir = OUTPUT_ROOT / "videos" / video_id

        if video_id != baseline_key:
            aliases.append(
                {
                    "canonical_video_id": video_id,
                    "source_baseline_key": baseline_key,
                    "reason": "Baseline source folder used a local key; canonical dataset folder uses the YouTube video id.",
                }
            )

        baseline_dst = video_dir / "baseline"
        copy_tree(baseline_dir, baseline_dst / "source_files")
        copy_file(canonical_file, baseline_dst / "canonical_diarization.json")
        copy_best_exports(baseline_dir, baseline_dst)

        baseline_segment_count = len(canonical.get("segments") or [])
        speaker_count = len(canonical.get("speakers") or [])

        baseline_embedding_dst = video_dir / "baseline_embeddings"
        baseline_embedding_providers: list[str] = []
        for provider in providers:
            provider_dir = BASELINE_EMBEDDINGS_ROOT / provider
            npz = provider_dir / f"{baseline_key}.embeddings.npz"
            segments = provider_dir / f"{baseline_key}.segments.json"
            if npz.exists() and segments.exists():
                copy_file(npz, baseline_embedding_dst / f"{provider}.embeddings.npz")
                copy_file(segments, baseline_embedding_dst / f"{provider}.segments.json")
                baseline_embedding_providers.append(provider)
            else:
                issues.append(
                    {
                        "video_id": video_id,
                        "baseline_key": baseline_key,
                        "issue": f"missing baseline embedding provider {provider}",
                    }
                )

        live_src, live_source_kind = live_source_for(video_id)
        live_dst = video_dir / "live_window"
        live_sentence_count = 0
        live_audio_count = 0
        live_embedding_providers: list[str] = []
        live_embedding_count = 0
        if live_src is not None:
            copy_file(live_src / "manifest.json", live_dst / "manifest.json")
            copy_file(live_src / "sentences.jsonl", live_dst / "sentences.jsonl")
            copy_tree(live_src / "audio", live_dst / "audio")
            copy_tree(live_src / "embeddings", live_dst / "embeddings")
            live_sentence_count = count_jsonl(live_dst / "sentences.jsonl")
            live_audio_count = len(list((live_dst / "audio").glob("sentence_*.wav")))
            live_embedding_providers = sorted(p.stem for p in (live_dst / "embeddings").glob("*.npz"))
            live_embedding_count = len(live_embedding_providers)
        else:
            issues.append(
                {
                    "video_id": video_id,
                    "baseline_key": baseline_key,
                    "issue": "missing live-window sentence/audio cache",
                }
            )

        metadata = {
            "canonical_video_id": video_id,
            "baseline_key": baseline_key,
            "source_url": source_url,
            "title": media.get("title") or canonical.get("title"),
            "duration_sec": media.get("duration_sec"),
            "baseline": {
                "canonical_diarization": "baseline/canonical_diarization.json",
                "source_files": "baseline/source_files",
                "segment_count": baseline_segment_count,
                "speaker_count": speaker_count,
                "embedding_provider_count": len(baseline_embedding_providers),
                "embedding_providers": baseline_embedding_providers,
            },
            "live_window": {
                "source_kind": live_source_kind,
                "source_path": str(live_src) if live_src else None,
                "sentences": "live_window/sentences.jsonl" if live_src else None,
                "audio_dir": "live_window/audio" if live_src else None,
                "sentence_count": live_sentence_count,
                "audio_clip_count": live_audio_count,
                "embedding_provider_count": live_embedding_count,
                "embedding_providers": live_embedding_providers,
                "complete_provider_embeddings": live_embedding_count == len(providers),
            },
        }
        write_json(video_dir / "metadata.json", metadata)
        videos.append(metadata)

    videos.sort(key=lambda item: item["canonical_video_id"].lower())
    complete_live_videos = [
        item for item in videos if item["live_window"]["complete_provider_embeddings"]
    ]
    live_sentence_videos = [
        item for item in videos if item["live_window"]["sentence_count"] > 0
    ]
    missing_live_embeddings = [
        item
        for item in videos
        if item["live_window"]["sentence_count"] > 0
        and not item["live_window"]["complete_provider_embeddings"]
    ]

    manifest = {
        "generated_at_utc": generated_at,
        "root": str(OUTPUT_ROOT),
        "layout": "canonical_youtube_id_per_video",
        "source_roots": {
            "baseline_root": str(BASELINE_ROOT),
            "baseline_embeddings_root": str(BASELINE_EMBEDDINGS_ROOT),
            "live_complete_root": str(LIVE_COMPLETE_ROOT),
            "live_missing_1x_root": str(LIVE_MISSING_1X_ROOT),
            "live_more_4_root": str(LIVE_MORE4_ROOT),
            "whospeaks_more_root": str(WHOSPEAKS_MORE_ROOT),
        },
        "provider_count": len(providers),
        "providers": providers,
        "video_count": len(videos),
        "default_evaluation_video_count": len(videos) - len(EVALUATION_BLACKLIST["excluded_videos"]),
        "evaluation_blacklist_count": len(EVALUATION_BLACKLIST["excluded_videos"]),
        "baseline_video_count": len(videos),
        "baseline_embedding_complete_video_count": sum(
            1 for item in videos if item["baseline"]["embedding_provider_count"] == len(providers)
        ),
        "live_sentence_cache_video_count": len(live_sentence_videos),
        "live_embedding_complete_video_count": len(complete_live_videos),
        "live_embedding_incomplete_video_count": len(missing_live_embeddings),
        "total_baseline_segments": sum(item["baseline"]["segment_count"] for item in videos),
        "total_live_sentences": sum(item["live_window"]["sentence_count"] for item in videos),
        "aliases": aliases,
        "issues": issues,
        "videos": videos,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    write_json(OUTPUT_ROOT / "aliases.json", aliases)
    write_json(OUTPUT_ROOT / "blacklist.json", EVALUATION_BLACKLIST)

    coverage_rows = []
    for item in videos:
        coverage_rows.append(
            {
                "video_id": item["canonical_video_id"],
                "baseline_key": item["baseline_key"],
                "baseline_segments": item["baseline"]["segment_count"],
                "baseline_embedding_providers": item["baseline"]["embedding_provider_count"],
                "live_sentences": item["live_window"]["sentence_count"],
                "live_audio_clips": item["live_window"]["audio_clip_count"],
                "live_embedding_providers": item["live_window"]["embedding_provider_count"],
                "live_source_kind": item["live_window"]["source_kind"] or "",
                "complete_live_embeddings": str(
                    item["live_window"]["complete_provider_embeddings"]
                ).lower(),
                "title": item["title"] or "",
            }
        )
    with (OUTPUT_ROOT / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    readme = build_readme(manifest)
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    MARKER.write_text(generated_at + "\n", encoding="utf-8")


def build_readme(manifest: dict[str, Any]) -> str:
    incomplete = [
        item
        for item in manifest["videos"]
        if item["live_window"]["sentence_count"] > 0
        and not item["live_window"]["complete_provider_embeddings"]
    ]
    complete = [
        item for item in manifest["videos"] if item["live_window"]["complete_provider_embeddings"]
    ]
    lines = [
        "# ElevenLabs Scribe 27-Video Dataset",
        "",
        "This is the canonical local dataset root for the 27-video ElevenLabs Scribe baseline corpus.",
        "",
        "## Quick Facts",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Videos | {manifest['video_count']} |",
        f"| Baseline videos with canonical diarization | {manifest['baseline_video_count']} |",
        f"| Baseline videos with all provider embeddings | {manifest['baseline_embedding_complete_video_count']} |",
        f"| Baseline segments | {manifest['total_baseline_segments']} |",
        f"| Live videos with sentence/audio caches | {manifest['live_sentence_cache_video_count']} |",
        f"| Live videos with all provider embeddings | {manifest['live_embedding_complete_video_count']} |",
        f"| Live videos still missing provider embeddings | {manifest['live_embedding_incomplete_video_count']} |",
        f"| Live sentences | {manifest['total_live_sentences']} |",
        f"| Embedding providers | {manifest['provider_count']} |",
        f"| Default optimizer evaluation videos | {manifest['default_evaluation_video_count']} |",
        f"| Evaluation blacklist entries | {manifest['evaluation_blacklist_count']} |",
        "",
        "## Layout",
        "",
        "```text",
        "elevenlabs_scribe_27/",
        "  README.md",
        "  manifest.json",
        "  aliases.json",
        "  blacklist.json",
        "  coverage.csv",
        "  source_lists/",
        "  videos/",
        "    <canonical_youtube_id>/",
        "      metadata.json",
        "      baseline/",
        "        canonical_diarization.json",
        "        best_diarization.{md,json,csv}",
        "        source_files/",
        "      baseline_embeddings/",
        "        <provider>.embeddings.npz",
        "        <provider>.segments.json",
        "      live_window/",
        "        manifest.json",
        "        sentences.jsonl",
        "        audio/",
        "        embeddings/",
        "```",
        "",
        "Every folder under `videos/` is named by canonical YouTube video ID. Source aliases are recorded in `aliases.json` and each video's `metadata.json`.",
        "",
        "## Alias Policy",
        "",
        "`cunk` is not a canonical video folder in this dataset. The Cunk baseline source key maps to canonical YouTube ID `JWS-qfR6K3w`, so the normalized folder is `videos/JWS-qfR6K3w`.",
        "",
        "## Current Live Embedding Gaps",
        "",
    ]
    if incomplete:
        lines.extend(
            [
                "These videos have live sentence/audio caches, but their live-window `embeddings/` folders contain no provider NPZ files yet:",
                "",
            ]
        )
        for item in incomplete:
            lines.append(
                f"- `{item['canonical_video_id']}`: {item['live_window']['sentence_count']} live sentences from `{item['live_window']['source_kind']}`"
            )
    else:
        lines.append("No live embedding gaps remain.")
    lines.extend(
        [
            "",
            "## Evaluation Blacklist",
            "",
            "`blacklist.json` lists videos that stay in the dataset but are excluded from optimizer scoring/search by default because their canonical baseline is known bad. Use `--include-blacklisted` with `runtime\\optimization\\optimize_canonical_27.py` when intentionally inspecting those videos.",
            "",
            "## Complete Live Embedding Videos",
            "",
            f"{len(complete)} videos currently have complete 15-provider live sentence embeddings.",
            "",
            "## Source Roots",
            "",
        ]
    )
    for name, path in manifest["source_roots"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "Run this from the repo root to regenerate the normalized corpus from the current source folders:",
            "",
            "```powershell",
            "python data\\datasets\\_build_elevenlabs_scribe_27.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    build()
