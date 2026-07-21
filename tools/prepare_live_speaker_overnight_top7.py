from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_live_speaker_gate_tapes import build_video as build_gate_tape


PREPARER_ID = "live_speaker_overnight_top7_preparer_v1"
_STOP = False


def _stop(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_is_complete(output: Path, summary: Path) -> bool:
    if not output.is_file() or output.stat().st_size <= 0 or not summary.is_file():
        return False
    try:
        metadata = json.loads(summary.read_text(encoding="utf-8-sig"))
        return (
            int(metadata.get("profile_event_count") or 0) > 0
            and metadata.get("output_sha256") == _sha256(output)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _gate_is_complete(root: Path, video_id: str, window_seconds: float) -> bool:
    target = root / video_id
    required = (
        target / "gate_tape.json",
        target / "probe_schedule.u1.npy",
        target / "speech_gate.u1.npy",
        target / "release_gate.u1.npy",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        metadata = json.loads(required[0].read_text(encoding="utf-8-sig"))
        return abs(float(metadata["probe_window_seconds"]) - float(window_seconds)) < 1e-9
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _validate_dense_cache(corpus_root: Path, videos: list[str], providers: list[str]) -> dict[str, Any]:
    progress_path = corpus_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8-sig"))
    jobs = {
        (str(row.get("video_id")), str(row.get("provider"))): row
        for row in progress.get("jobs") or []
    }
    missing: list[str] = []
    for video_id in videos:
        for provider in providers:
            row = jobs.get((video_id, provider))
            if (
                row is None
                or row.get("status") != "complete"
                or int(row.get("completed_embeddings") or 0)
                != int(row.get("expected_embeddings") or -1)
            ):
                missing.append(f"{video_id}:{provider}")
    if missing:
        raise RuntimeError(f"Dense cache is incomplete for {len(missing)} jobs: {missing[:8]}")
    return {
        "progress_path": str(progress_path),
        "progress_sha256": _sha256(progress_path),
        "status": progress.get("status"),
        "validated_jobs": len(videos) * len(providers),
        "completed_embeddings_in_shared_corpus": int(progress.get("completed_embeddings") or 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare resumable references, gate tapes, and causal profile tapes for the top-seven overnight run."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile-python",
        type=Path,
        help="Optional Python executable for cached final-sentence profile replay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _stop)
    started = time.monotonic()
    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    corpus_root = args.corpus_root.resolve()
    source_root = args.source_dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos = [str(value) for value in spec["videos"]]
    providers = [str(value) for value in spec["providers"]]
    profile_sets = {str(key): str(value) for key, value in spec["profile_sets"].items()}
    short_windows = [float(value) for value in spec["search"]["short_windows_seconds"]]
    final_provider = str(spec["final_provider"])
    # Do not resolve the virtualenv's Python symlink to /usr/bin/python: the
    # symlink path is what activates the venv-specific site-packages.
    profile_python = args.profile_python.expanduser() if args.profile_python else Path(sys.executable)

    cache_validation = _validate_dense_cache(corpus_root, videos, providers)
    tasks: list[tuple[str, str, str]] = []
    for video_id in videos:
        tasks.append(("reference", video_id, "canonical"))
    for window in short_windows:
        for video_id in videos:
            tasks.append(("gate", video_id, f"{window:.1f}"))
    for profile_name in profile_sets:
        for video_id in videos:
            tasks.append(("profile", video_id, profile_name))

    progress_path = output_root / "preparation_progress.json"
    completed = 0

    def progress(active: str, *, status: str = "running") -> None:
        _atomic_json(progress_path, {
            "schema_version": 1,
            "preparer_id": PREPARER_ID,
            "status": status,
            "active": active,
            "completed_steps": completed,
            "total_steps": len(tasks),
            "percent": round(100.0 * completed / max(1, len(tasks)), 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        print(
            f"[{completed:03d}/{len(tasks):03d} {100.0 * completed / max(1, len(tasks)):6.2f}%] {active}",
            flush=True,
        )

    for kind, video_id, variant in tasks:
        if _STOP:
            progress("interrupted; completed artifacts are preserved", status="interrupted")
            return 130
        progress(f"{kind} {video_id} {variant}")
        if kind == "reference":
            source = source_root / "videos" / video_id / "baseline" / "canonical_diarization.json"
            target = output_root / "references" / video_id / "canonical_diarization.json"
            if not source.is_file():
                raise FileNotFoundError(source)
            # Some historical reference files carry a UTF-8 BOM while the
            # production scorer deliberately reads plain UTF-8. Normalize the
            # serialized copy without changing its JSON content.
            _atomic_json(target, json.loads(source.read_text(encoding="utf-8-sig")))
        elif kind == "gate":
            window = float(variant)
            gate_root = output_root / "gate_sets" / f"{round(window * 1000):04d}ms"
            if not _gate_is_complete(gate_root, video_id, window):
                source = json.loads(
                    (corpus_root / "videos" / video_id / "source.json").read_text(encoding="utf-8-sig")
                )
                media_root = Path(str(source["audio_path_at_creation"])).resolve().parent
                build_gate_tape(
                    corpus_root,
                    media_root,
                    gate_root,
                    video_id,
                    probe_window_seconds=window,
                    clear_window_seconds=window,
                    cadence_seconds=float(spec["baseline"]["probe_interval_seconds"]),
                    frame_seconds=0.03,
                    threshold=0.003,
                    min_speech_seconds=0.15,
                )
        else:
            live_provider = profile_sets[variant]
            target_root = output_root / "profiles" / variant / video_id
            output = target_root / "production_stack.profiles.jsonl"
            summary = target_root / "production_stack.profiles.summary.json"
            if not _profile_is_complete(output, summary):
                sentence_cache = source_root / "videos" / video_id / "live_window"
                command = [
                    str(profile_python),
                    str(TOOLS / "build_live_speaker_profile_tape.py"),
                    "--sentence-cache", str(sentence_cache),
                    "--video-id", video_id,
                    "--final-provider", final_provider,
                    "--live-provider", live_provider,
                    "--output", str(output),
                    "--summary-output", str(summary),
                ]
                target_root.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        completed += 1

    manifest = {
        "schema_version": 1,
        "preparer_id": PREPARER_ID,
        "status": "complete",
        "spec": str(args.spec.resolve()),
        "spec_sha256": _sha256(args.spec.resolve()),
        "corpus_root": str(corpus_root),
        "source_dataset_root": str(source_root),
        "gate_python": str(Path(sys.executable)),
        "profile_python": str(profile_python),
        "videos": videos,
        "providers": providers,
        "profile_sets": profile_sets,
        "short_windows_seconds": short_windows,
        "cache_validation": cache_validation,
        "completed_steps": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _atomic_json(output_root / "preparation_manifest.json", manifest)
    progress("complete", status="complete")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
