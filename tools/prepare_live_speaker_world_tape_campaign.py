"""Prepare and audit authentic GUI World-Tape runs for the Top-7 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.audio_utils import load_audio_file
from window.live_speaker_e2e_contract import validate_real_gui_e2e_observation
from window.live_speaker_parity_replay import validate_world_tape


CAMPAIGN_CONTRACT_ID = "whospeaks.live_world_tape_campaign.v1"
DEFAULT_CAMPAIGN_ROOT = (
    ROOT / "runtime" / "optimization" / "live_speaker_world_tapes_20260721"
)
DEFAULT_PCM_ROOT = (
    ROOT / "runtime" / "media" / "live-speaker-world-tape-top7-v1" / "exact-pcm"
)
DEFAULT_TOP7_ROOT = (
    ROOT / "runtime" / "optimization" / "live_speaker_overnight_top7_20260721"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pcm_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _timeline_pcm_sha256(audio: np.ndarray) -> str:
    """Hash the exact float32 buffer owned by AudioTimeline at runtime."""

    timeline_audio = np.clip(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        -1.0,
        1.0,
    ).astype(np.float32, copy=False)
    return _pcm_sha256(timeline_audio)


def _embedding_pcm16_sha256(audio: np.ndarray) -> str:
    """Hash the transport samples sent to the remote embedding server."""

    pcm16 = (
        np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        * 32767.0
    ).astype(np.int16)
    return hashlib.sha256(np.ascontiguousarray(pcm16).tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_pcm_rows(pcm_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = pcm_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("contract_id") != "whospeaks.live_world_tape_pcm.v1":
        raise RuntimeError(f"Unexpected PCM manifest contract: {manifest_path}")
    rows: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("videos") or []:
        row = dict(raw)
        video_id = str(row["video_id"])
        path = pcm_root / f"{video_id}.audio.f32.wav"
        if not path.is_file():
            raise FileNotFoundError(path)
        file_hash = _sha256_file(path)
        if file_hash != str(row["export_file_sha256"]):
            raise RuntimeError(f"Float-PCM file hash mismatch for {video_id}")
        audio, sample_rate = load_audio_file(path)
        decoded_hash = _pcm_sha256(audio)
        if decoded_hash != str(row["decoded_pcm_sha256"]):
            raise RuntimeError(f"Float-PCM decode hash mismatch for {video_id}")
        if int(sample_rate) != int(row["sample_rate"]):
            raise RuntimeError(f"Float-PCM sample-rate mismatch for {video_id}")
        if int(audio.size) != int(row["samples"]):
            raise RuntimeError(f"Float-PCM sample-count mismatch for {video_id}")
        row.update({
            "local_audio_path": str(path.resolve()),
            "local_audio_file_sha256": file_hash,
            "local_decoded_pcm_sha256": decoded_hash,
            # AudioTimeline deliberately clips loaded audio to [-1, 1].  Keep
            # both identities: the dense corpus owns the raw float32 hash,
            # while a real GUI World Tape records the clipped timeline hash.
            "local_timeline_pcm_sha256": _timeline_pcm_sha256(audio),
            # The remote embedding transport clips to PCM16 too, so cached
            # raw PCM and GUI timeline PCM remain inference-input equivalent.
            "local_embedding_pcm16_sha256": _embedding_pcm16_sha256(audio),
        })
        rows[video_id] = row
    return rows


def _scan_runs(campaign_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for directory in sorted(campaign_root.iterdir() if campaign_root.is_dir() else []):
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            runs.append({
                "path": str(directory.resolve()),
                "video_id": "",
                "status": "unreadable",
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            })
            continue
        media_history = manifest.get("media_history") or []
        video_id = str(media_history[-1].get("video_id") or "") if media_history else ""
        validation = validate_world_tape(directory)
        errors = list(validation.get("errors") or [])
        raw_observation_path = str(
            (manifest.get("runtime_config") or {}).get("browser_live_observation_output") or ""
        )
        observation_path = Path(raw_observation_path) if raw_observation_path else None
        observation_valid = False
        observation_score: float | None = None
        if observation_path is None:
            errors.append("browser observation path is missing from runtime config")
        elif not observation_path.is_file():
            errors.append(f"browser observation is missing: {observation_path}")
        else:
            try:
                observation = json.loads(observation_path.read_text(encoding="utf-8-sig"))
                observation_errors = validate_real_gui_e2e_observation(observation)
                errors.extend(f"browser observation: {message}" for message in observation_errors)
                observation_valid = not observation_errors
                observation_score = float(
                    (observation.get("summary") or {}).get("strict_browser_live_score")
                )
            except Exception as exc:
                errors.append(f"browser observation unreadable: {type(exc).__name__}: {exc}")
        runs.append({
            "path": str(directory.resolve()),
            "run_id": str(manifest.get("run_id") or ""),
            "video_id": video_id,
            "status": str(manifest.get("status") or ""),
            "valid": bool(validation.get("valid")) and observation_valid and not errors,
            "errors": errors,
            "media": media_history[-1] if media_history else {},
            "observation_path": str(observation_path) if observation_path is not None else "",
            "observation_score": observation_score,
            "event_count": int(validation.get("event_count") or 0),
            "browser_sample_batch_count": int(
                validation.get("browser_sample_batch_count") or 0
            ),
        })
    return runs


def _launcher_text(
    *,
    video_id: str,
    audio_path: Path,
    canonical_path: Path,
    campaign_root: Path,
    champion_launcher: Path,
    champion_artifact: Path,
) -> str:
    return f"""@echo off
setlocal
pushd "{ROOT}"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUNSTAMP=%%I"
if not exist "{campaign_root / 'browser_observations'}" mkdir "{campaign_root / 'browser_observations'}"
call "{champion_launcher}" ^
  --url "https://www.youtube.com/watch?v={video_id}" ^
  --audio-file "{audio_path}" ^
  --video-file "{audio_path}" --skip-download ^
  --validation-canonical "{canonical_path}" ^
  --live-speaker-world-tape-output "{campaign_root}" ^
  --browser-live-observation-output "{campaign_root / 'browser_observations' / (video_id + '_%RUNSTAMP%.json')}" ^
  --browser-live-e2e-candidate-artifact "{champion_artifact}" ^
  --exit-after-browser-live-observation %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
"""


def _write_launcher(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\r\n")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    campaign_root = args.campaign_root.resolve()
    pcm_root = args.pcm_root.resolve()
    reference_root = args.reference_root.resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    spec_video_ids = [str(value) for value in spec["videos"]]
    # Establish repeatability on the known Cunk canary before spending the
    # longer wall-clock runs.  The remaining videos retain spec order.
    video_ids = ["JWS-qfR6K3w", *(
        value for value in spec_video_ids if value != "JWS-qfR6K3w"
    )]
    pcm_rows = _load_pcm_rows(pcm_root)
    runs = _scan_runs(campaign_root)
    launchers_root = campaign_root / "launchers"
    rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        if video_id not in pcm_rows:
            raise RuntimeError(f"PCM manifest is missing {video_id}")
        canonical = reference_root / video_id / "canonical_diarization.json"
        if not canonical.is_file():
            raise FileNotFoundError(canonical)
        audio = Path(str(pcm_rows[video_id]["local_audio_path"]))
        launcher = launchers_root / f"launch_{video_id}.cmd"
        _write_launcher(
            launcher,
            _launcher_text(
                video_id=video_id,
                audio_path=audio,
                canonical_path=canonical,
                campaign_root=campaign_root,
                champion_launcher=args.champion_launcher.resolve(),
                champion_artifact=args.champion_artifact.resolve(),
            ),
        )
        required_runs = int(args.cunk_repeats if video_id == "JWS-qfR6K3w" else 1)
        matching: list[dict[str, Any]] = []
        expected_file_hash = str(pcm_rows[video_id]["local_audio_file_sha256"])
        expected_pcm_hash = str(pcm_rows[video_id]["local_timeline_pcm_sha256"])
        for source_run in runs:
            if source_run["video_id"] != video_id:
                continue
            run = dict(source_run)
            run["errors"] = list(source_run.get("errors") or [])
            media_record = run.get("media") if isinstance(run.get("media"), dict) else {}
            if str(media_record.get("audio_sha256") or "") != expected_file_hash:
                run["errors"].append("run audio file hash does not match exact campaign PCM")
            if str(media_record.get("decoded_pcm_sha256") or "") != expected_pcm_hash:
                run["errors"].append(
                    "run decoded PCM hash does not match the clipped GUI timeline"
                )
            run["valid"] = bool(run.get("valid")) and not run["errors"]
            matching.append(run)
        complete = [row for row in matching if row["valid"]]
        rows.append({
            "video_id": video_id,
            "required_valid_runs": required_runs,
            "valid_runs": len(complete),
            "remaining_runs": max(0, required_runs - len(complete)),
            "launcher": str(launcher.resolve()),
            "audio": pcm_rows[video_id],
            "canonical_path": str(canonical.resolve()),
            "canonical_sha256": _sha256_file(canonical),
            "runs": matching,
        })
    next_row = next((row for row in rows if row["remaining_runs"] > 0), None)
    report = {
        "contract_id": CAMPAIGN_CONTRACT_ID,
        "campaign_root": str(campaign_root),
        "top7_spec": str(spec_path),
        "top7_spec_sha256": _sha256_file(spec_path),
        "pcm_manifest": str((pcm_root / "manifest.json").resolve()),
        "pcm_manifest_sha256": _sha256_file(pcm_root / "manifest.json"),
        "reference_root": str(reference_root),
        "required_run_count": sum(row["required_valid_runs"] for row in rows),
        "valid_run_count": sum(row["valid_runs"] for row in rows),
        "remaining_run_count": sum(row["remaining_runs"] for row in rows),
        "next_video_id": None if next_row is None else next_row["video_id"],
        "next_launcher": None if next_row is None else next_row["launcher"],
        "videos": rows,
    }
    _atomic_json(campaign_root / "campaign.json", report)
    if next_row is not None:
        next_launcher = Path(str(next_row["launcher"]))
        _write_launcher(
            campaign_root / "launch_next_world_tape.cmd",
            f"@echo off\r\ncall \"{next_launcher}\" %*\r\nexit /b %errorlevel%\r\n",
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact Top-7 PCM, generate resumable authentic-GUI launchers, and audit runs."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_TOP7_ROOT / "spec.json")
    parser.add_argument("--pcm-root", type=Path, default=DEFAULT_PCM_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT / "references")
    parser.add_argument(
        "--champion-launcher",
        type=Path,
        default=DEFAULT_TOP7_ROOT / "launch_champion_gui_realtime.cmd",
    )
    parser.add_argument(
        "--champion-artifact",
        type=Path,
        default=DEFAULT_TOP7_ROOT / "champion_final.json",
    )
    parser.add_argument("--cunk-repeats", type=int, default=3)
    parser.add_argument("--launch-next", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.launch_next:
        launcher = report.get("next_launcher")
        if not launcher:
            print("All required World-Tape runs are complete.", flush=True)
            return 0
        return subprocess.call(["cmd.exe", "/d", "/c", str(launcher)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
