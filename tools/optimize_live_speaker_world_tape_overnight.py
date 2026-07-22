"""Resumable discovery-only search over immutable live-speaker World Tapes.

This runner is intentionally unable to promote a candidate or edit a launcher.  It
only emits ``REPLAY_ONLY`` nominees which must pass the authentic real-GUI gate.
Every candidate identifier commits to the complete replay input and scoring
environment, so results from different code, tape, or scorer revisions cannot be
silently mixed in one journal.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from statistics import mean
from typing import Any, Iterable
import uuid


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from window.live_speaker_bayes import BayesSpeakerTrackerConfig
from window.live_speaker_browser_parity import BROWSER_PARITY_REPLAY_ID
from window.live_speaker_counterfactual import (
    COUNTERFACTUAL_REPLAY_ID,
    evaluate_counterfactual,
)
from window.live_speaker_e2e_contract import source_tree_sha256


RUNNER_CONTRACT_ID = "whospeaks.live_world_tape.overnight_discovery.v1"
CANDIDATE_ID_CONTRACT = "whospeaks.live_world_tape.bound_candidate.v1"
SEARCH_SPACE_ID = "whospeaks.live_world_tape.open_set_existing_parameters.v2"
SCORER_CONTRACT_ID = (
    "causal_live_speaker_primary_macro_v2:"
    "mean_per_video_strict_browser_live_score"
)
REQUIRED_PROVIDER = "speechbrain_resnet"
REQUIRED_WINDOWS = (0.7, 1.5)
REQUIRED_CADENCE = 0.4
REPLAY_ONLY_STATUS = "REPLAY_ONLY_NOMINEE_REQUIRES_REAL_GUI_VALIDATION"
_ALLOWED_EXTRA_CONFIG_KEYS = {"live_speaker_probe_hold_seconds"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values are not valid immutable input")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write a replaceable snapshot without ever exposing a partial JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def append_fsync_jsonl(path: Path, value: Any) -> None:
    """Append one compact journal record through O_APPEND and fsync it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(target, flags, 0o644)
    try:
        # Journal records are deliberately compact.  Retrying a partial write is
        # safe because this runner has one writer and every write uses O_APPEND.
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Journal append made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_journal(path: Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupt journal record at {target}:{line_number}; refusing "
                    "to guess where a candidate ended"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Journal record {line_number} is not an object")
            records.append(value)
    return records


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        # Failing to inspect another user's process is not evidence that it died.
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


@contextmanager
def exclusive_writer_lock(path: Path, *, recover_stale: bool = False):
    """Prevent two adaptive searches from appending to the same journal."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    token = uuid.uuid4().hex
    value = {
        "contract_id": RUNNER_CONTRACT_ID,
        "pid": os.getpid(),
        "hostname": hostname,
        "token": token,
        "created_at": _utc_now(),
    }
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    while True:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        try:
            descriptor = os.open(target, flags, 0o644)
        except FileExistsError:
            try:
                existing = _load_json(target)
            except Exception:
                existing = {}
            same_host = str(existing.get("hostname") or "") == hostname
            alive = same_host and _pid_is_alive(int(existing.get("pid") or 0))
            if alive:
                raise RuntimeError(
                    f"Another overnight writer is active for this output: {existing}"
                )
            if not (recover_stale or same_host):
                raise RuntimeError(
                    f"Stale or foreign-host writer lock exists: {target}. "
                    "Inspect it, then pass --recover-stale-lock if safe."
                )
            target.unlink()
            continue
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        break
    try:
        yield value
    finally:
        try:
            existing = _load_json(target)
        except Exception:
            existing = {}
        if existing.get("token") == token:
            target.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _same_float(left: Any, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _resolve_tape_dir(raw: Any, campaign_root: Path) -> Path:
    path = Path(str(raw or ""))
    if path.is_dir():
        return path.resolve()
    portable = campaign_root / path.name
    if portable.is_dir():
        return portable.resolve()
    raise FileNotFoundError(f"World Tape directory is missing: {raw}")


def _resolve_canonical_path(raw: Any, campaign_root: Path, video_id: str) -> Path:
    path = Path(str(raw or ""))
    if path.is_file():
        return path.resolve()
    filename = path.name or "canonical_diarization.json"
    portable = campaign_root / "references" / video_id / filename
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(f"Canonical reference is missing: {raw}")


@dataclass(frozen=True)
class TapeInput:
    video_id: str
    run_id: str
    tape_dir: Path
    manifest_path: Path
    manifest_sha256: str
    canonical_path: Path
    canonical_sha256: str

    def identity(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "canonical_sha256": self.canonical_sha256,
        }


@dataclass(frozen=True)
class FrozenInputs:
    workspace: Path
    campaign_root: Path
    campaign_path: Path
    campaign_sha256: str
    parity_report_path: Path
    parity_report_sha256: str
    parity_report_contract_id: str
    optimization_eligible: bool
    diagnostic_parity_exception: bool
    parity_eligibility_reason: str
    source_tree_sha256: str
    runner_sha256: str
    scorer_reducer_contract: dict[str, Any]
    tapes: tuple[TapeInput, ...]

    def candidate_binding(self) -> dict[str, Any]:
        return {
            "runner_contract_id": RUNNER_CONTRACT_ID,
            "runner_sha256": self.runner_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "campaign_sha256": self.campaign_sha256,
            "parity_report_sha256": self.parity_report_sha256,
            "parity_report_contract_id": self.parity_report_contract_id,
            "parity_gate": {
                "optimization_eligible": self.optimization_eligible,
                "diagnostic_parity_exception": self.diagnostic_parity_exception,
                "eligibility_reason": self.parity_eligibility_reason,
            },
            "scorer_reducer_contract": self.scorer_reducer_contract,
            "required_runtime": {
                "provider": REQUIRED_PROVIDER,
                "windows_seconds": list(REQUIRED_WINDOWS),
                "cadence_seconds": REQUIRED_CADENCE,
            },
            "tape_manifests": [item.identity() for item in self.tapes],
        }


def freeze_inputs(
    workspace: Path,
    campaign_root: Path,
    parity_report_path: Path,
    *,
    allow_diagnostic_parity: bool,
    runner_path: Path | None = None,
) -> FrozenInputs:
    workspace = Path(workspace).resolve()
    campaign_root = Path(campaign_root).resolve()
    campaign_path = campaign_root / "campaign.json"
    parity_report_path = Path(parity_report_path).resolve()
    if not campaign_path.is_file():
        raise FileNotFoundError(f"Campaign manifest is missing: {campaign_path}")
    report = _load_json(parity_report_path)
    optimization_eligible = bool(report.get("optimization_eligible"))
    if not optimization_eligible and not allow_diagnostic_parity:
        raise ValueError(
            "Parity report is optimization_eligible=false. This runner fails closed; "
            "pass --allow-diagnostic-parity only for explicitly accepted, replay-only "
            "diagnostic discovery."
        )

    raw_runs = report.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Parity report contains no World Tape runs")
    tapes: list[TapeInput] = []
    seen_run_ids: set[str] = set()
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise ValueError("Parity report run is not an object")
        video_id = str(raw_run.get("video_id") or "").strip()
        run_id = str(raw_run.get("run_id") or "").strip()
        if not video_id or not run_id or run_id in seen_run_ids:
            raise ValueError(f"Invalid or duplicate World Tape run: {run_id!r}")
        seen_run_ids.add(run_id)
        tape_dir = _resolve_tape_dir(raw_run.get("tape_dir"), campaign_root)
        try:
            tape_dir.relative_to(campaign_root)
        except ValueError as exc:
            raise ValueError(f"Tape escapes campaign root: {tape_dir}") from exc
        manifest_path = tape_dir / "manifest.json"
        manifest = _load_json(manifest_path)
        artifact = dict(manifest.get("artifact") or {})
        if str(artifact.get("status") or "") != "complete":
            raise ValueError(f"World Tape is not complete: {tape_dir}")
        for hash_name in (
            "events_sha256",
            "arrays_sha256",
            "arrays_index_sha256",
        ):
            digest = str(artifact.get(hash_name) or "")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()
            ):
                raise ValueError(
                    f"World Tape has no valid {hash_name}: {tape_dir}"
                )
        for filename in ("events.jsonl", "arrays.f32", "arrays.jsonl"):
            if not (tape_dir / filename).is_file():
                raise FileNotFoundError(
                    f"World Tape payload is missing {filename}: {tape_dir}"
                )
        if str(manifest.get("run_id") or "") != run_id:
            raise ValueError(f"Run ID mismatch for {tape_dir}")
        config = dict(manifest.get("runtime_config") or {})
        runtime_values = {
            "provider": config.get("live_speaker_embedding_provider"),
            "short": config.get("live_speaker_probe_window_seconds"),
            "context": config.get("live_speaker_probe_context_window_seconds"),
            "cadence": config.get("live_speaker_probe_interval_seconds"),
            "minimum_interval": config.get(
                "live_speaker_embedding_min_interval_seconds"
            ),
        }
        if runtime_values["provider"] != REQUIRED_PROVIDER:
            raise ValueError(f"Unexpected provider in {tape_dir}: {runtime_values}")
        if not _same_float(runtime_values["short"], REQUIRED_WINDOWS[0]):
            raise ValueError(f"Unexpected short window in {tape_dir}: {runtime_values}")
        if not _same_float(runtime_values["context"], REQUIRED_WINDOWS[1]):
            raise ValueError(f"Unexpected context window in {tape_dir}: {runtime_values}")
        if not _same_float(runtime_values["cadence"], REQUIRED_CADENCE):
            raise ValueError(f"Unexpected probe cadence in {tape_dir}: {runtime_values}")
        if not _same_float(runtime_values["minimum_interval"], REQUIRED_CADENCE):
            raise ValueError(
                f"Unexpected embedding minimum interval in {tape_dir}: {runtime_values}"
            )
        canonical_path = _resolve_canonical_path(
            raw_run.get("canonical_path"), campaign_root, video_id
        )
        tapes.append(
            TapeInput(
                video_id=video_id,
                run_id=run_id,
                tape_dir=tape_dir,
                manifest_path=manifest_path,
                manifest_sha256=file_sha256(manifest_path),
                canonical_path=canonical_path,
                canonical_sha256=file_sha256(canonical_path),
            )
        )

    source_modules = {
        "counterfactual": workspace / "src/window/live_speaker_counterfactual.py",
        "browser_reducer": workspace / "src/window/live_speaker_browser_parity.py",
        "browser_scorer": workspace / "src/window/browser_live_speaker_scoring.py",
        "bayes_tracker": workspace / "src/window/live_speaker_bayes.py",
    }
    scorer_reducer_contract = {
        "selection_score_contract_id": SCORER_CONTRACT_ID,
        "counterfactual_contract_id": COUNTERFACTUAL_REPLAY_ID,
        "browser_reducer_contract_id": BROWSER_PARITY_REPLAY_ID,
        "source_sha256": {
            name: file_sha256(path) for name, path in sorted(source_modules.items())
        },
    }
    actual_runner = Path(runner_path or __file__).resolve()
    return FrozenInputs(
        workspace=workspace,
        campaign_root=campaign_root,
        campaign_path=campaign_path,
        campaign_sha256=file_sha256(campaign_path),
        parity_report_path=parity_report_path,
        parity_report_sha256=file_sha256(parity_report_path),
        parity_report_contract_id=str(report.get("contract_id") or ""),
        optimization_eligible=optimization_eligible,
        diagnostic_parity_exception=(not optimization_eligible),
        parity_eligibility_reason=str(report.get("eligibility_reason") or ""),
        source_tree_sha256=source_tree_sha256(workspace),
        runner_sha256=file_sha256(actual_runner),
        scorer_reducer_contract=scorer_reducer_contract,
        tapes=tuple(tapes),
    )


def assert_inputs_unchanged(frozen: FrozenInputs, *, runner_path: Path | None = None) -> None:
    if file_sha256(frozen.campaign_path) != frozen.campaign_sha256:
        raise RuntimeError("campaign.json changed during the search")
    if file_sha256(frozen.parity_report_path) != frozen.parity_report_sha256:
        raise RuntimeError("parity report changed during the search")
    if source_tree_sha256(frozen.workspace) != frozen.source_tree_sha256:
        raise RuntimeError("source tree changed during the search")
    if file_sha256(Path(runner_path or __file__).resolve()) != frozen.runner_sha256:
        raise RuntimeError("overnight runner changed during the search")
    for tape in frozen.tapes:
        if file_sha256(tape.manifest_path) != tape.manifest_sha256:
            raise RuntimeError(f"World Tape manifest changed: {tape.manifest_path}")
        if file_sha256(tape.canonical_path) != tape.canonical_sha256:
            raise RuntimeError(f"Canonical reference changed: {tape.canonical_path}")


def load_base_config(base_artifact_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _load_json(Path(base_artifact_path).resolve())
    if str(artifact.get("provider_spec") or "") != REQUIRED_PROVIDER:
        raise ValueError(
            f"Base artifact must use exactly {REQUIRED_PROVIDER!r}, got "
            f"{artifact.get('provider_spec')!r}"
        )
    windows = tuple(float(value) for value in artifact.get("windows_seconds") or ())
    if windows != REQUIRED_WINDOWS:
        raise ValueError(f"Base artifact must use exactly {REQUIRED_WINDOWS}, got {windows}")
    if artifact.get("cadence_seconds") is not None and not _same_float(
        artifact.get("cadence_seconds"), REQUIRED_CADENCE
    ):
        raise ValueError(
            f"Base artifact cadence must be {REQUIRED_CADENCE}, got "
            f"{artifact.get('cadence_seconds')!r}"
        )
    raw = artifact.get("algorithm_config")
    if not isinstance(raw, dict):
        raise ValueError("Base artifact has no algorithm_config object")
    config = normalize_and_validate_config(raw)
    return config, artifact


def normalize_and_validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(BayesSpeakerTrackerConfig)}
    unsupported = sorted(set(raw) - allowed - _ALLOWED_EXTRA_CONFIG_KEYS)
    if unsupported:
        raise ValueError(
            "Candidate contains parameters that the World-Tape evaluator ignores: "
            + ", ".join(unsupported)
        )
    values = dict(raw)
    windows = tuple(float(value) for value in values.get("scale_windows") or ())
    if windows != REQUIRED_WINDOWS:
        raise ValueError(f"Candidate must use exactly two windows {REQUIRED_WINDOWS}")
    weights = tuple(float(value) for value in values.get("scale_weights") or ())
    if len(weights) != 2 or any(value < 0.0 for value in weights):
        raise ValueError("Candidate must contain two non-negative scale weights")
    if sum(weights) <= 0.0:
        raise ValueError("Scale weights may not both be zero")
    values["scale_windows"] = list(REQUIRED_WINDOWS)
    values["scale_weights"] = [value / sum(weights) for value in weights]
    config_values = {name: value for name, value in values.items() if name in allowed}
    validated = BayesSpeakerTrackerConfig(**config_values)
    normalized = asdict(validated)
    # Preserve only explicitly supplied tracker parameters.  This keeps artifacts
    # compact while validation still exercises defaults for omitted fields.
    result = {name: normalized[name] for name in values if name in normalized}
    if "live_speaker_probe_hold_seconds" in values:
        hold = float(values["live_speaker_probe_hold_seconds"])
        if not 0.0 <= hold <= 5.0:
            raise ValueError("live_speaker_probe_hold_seconds must be in [0, 5]")
        result["live_speaker_probe_hold_seconds"] = hold
    result["scale_windows"] = list(REQUIRED_WINDOWS)
    result["scale_weights"] = list(values["scale_weights"])
    return _jsonable(result)


def candidate_identity(
    config: dict[str, Any], frozen: FrozenInputs
) -> tuple[str, str, dict[str, Any]]:
    config_sha = stable_sha256(config)
    identity = {
        "contract_id": CANDIDATE_ID_CONTRACT,
        "algorithm_config_sha256": config_sha,
        **frozen.candidate_binding(),
    }
    digest = stable_sha256(identity)
    return f"wt-{digest[:24]}", config_sha, identity


def _with_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for name, value in patch.items():
        if name == "short_weight":
            short = float(value)
            result["scale_weights"] = [short, 1.0 - short]
        else:
            result[name] = value
    return normalize_and_validate_config(result)


def fixed_open_set_proposals(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return deterministic, low-risk one-dimensional and gate-family probes."""

    proposals: list[tuple[str, dict[str, Any]]] = [("incumbent", {})]
    dimensions: list[tuple[str, Iterable[Any]]] = [
        ("short_weight", (0.65, 0.7, 0.75, 0.8, 0.85, 0.9)),
        ("min_similarity", (0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3)),
        ("similarity_temperature", (0.06, 0.07, 0.08, 0.0875, 0.095, 0.11, 0.13)),
        ("unknown_bias", (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5)),
        ("profile_count_bias_threshold", (1, 2, 3, 4, 5)),
        ("low_profile_unknown_bias", (-1.0, -0.75, -0.5, -0.25, 0.0)),
        ("high_profile_unknown_bias", (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)),
        ("profile_count_unknown_bias_slope", (-0.15, 0.0, 0.1, 0.2, 0.3)),
        ("min_known_probability", (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)),
        ("unknown_release_count", (1, 2, 3, 4, 5)),
        ("silence_release_count", (1, 2, 3)),
        (
            "live_speaker_probe_hold_seconds",
            (0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.25, 2.5, 2.75),
        ),
    ]
    for name, values in dimensions:
        for value in values:
            proposals.append((f"coordinate:{name}={value}", {name: value}))

    for count, ceiling in ((1, 0.05), (1, 0.1), (1, 0.15), (2, 0.1), (2, 0.2)):
        proposals.append(
            (
                f"provisional-open-set:{count}:{ceiling}",
                {
                    "enable_provisional_profiles": True,
                    "provisional_creation_count": count,
                    "provisional_creation_similarity_ceiling": ceiling,
                    "provisional_creation_max_finalized_profiles": -1,
                },
            )
        )
    # Normalize now so an invalid search-space entry fails before the long run.
    return [(hypothesis, _with_patch(base, patch)) for hypothesis, patch in proposals]


_RANDOM_SPACE: dict[str, tuple[Any, ...]] = {
    "short_weight": (0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    "min_similarity": (0.14, 0.16, 0.175, 0.19, 0.21, 0.23, 0.26, 0.29),
    "similarity_temperature": (0.055, 0.065, 0.075, 0.0875, 0.10, 0.115, 0.135),
    "profile_count_bias_threshold": (1, 2, 3, 4, 5),
    "low_profile_unknown_bias": (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0),
    "high_profile_unknown_bias": (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0),
    "profile_count_unknown_bias_slope": (-0.15, 0.0, 0.1, 0.2, 0.3),
    "min_known_probability": (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65),
    "unknown_release_count": (1, 2, 3, 4, 5),
    "silence_release_count": (1, 2, 3),
    "live_speaker_probe_hold_seconds": (
        0.6,
        0.8,
        1.0,
        1.2,
        1.6,
        2.0,
        2.25,
        2.5,
        2.75,
    ),
}


def adaptive_proposal(
    best: dict[str, Any], *, seed: int, search_index: int, salt: int = 0
) -> tuple[str, dict[str, Any]]:
    entropy = stable_sha256(
        {
            "seed": int(seed),
            "search_index": int(search_index),
            "salt": int(salt),
            "parent_config_sha256": stable_sha256(best),
            "search_space": SEARCH_SPACE_ID,
        }
    )
    rng = random.Random(int(entropy[:16], 16))
    count = rng.randint(2, 5)
    names = rng.sample(sorted(_RANDOM_SPACE), count)
    patch = {name: rng.choice(_RANDOM_SPACE[name]) for name in names}
    hypothesis = "adaptive-open-set:" + ",".join(
        f"{name}={patch[name]}" for name in sorted(patch)
    )
    return hypothesis, _with_patch(best, patch)


def evaluate_candidate(frozen: FrozenInputs, config: dict[str, Any]) -> dict[str, Any]:
    run_scores: list[dict[str, Any]] = []
    started = time.monotonic()
    for tape in frozen.tapes:
        replay = evaluate_counterfactual(
            tape.tape_dir, config, tape.canonical_path
        )
        score = dict(replay["score"])
        run_scores.append(
            {
                "video_id": tape.video_id,
                "run_id": tape.run_id,
                "strict_browser_live_score": float(
                    replay["strict_browser_live_score"]
                ),
                "projected_live_action_count": int(
                    replay["projected_live_action_count"]
                ),
                "correct_live_speaker_coverage": float(
                    score["correct_live_speaker_coverage"]
                ),
                "wrong_live_speech_ratio": float(
                    score["wrong_live_speech_ratio"]
                ),
                "missing_live_speech_ratio": float(
                    score["missing_live_speech_ratio"]
                ),
                "outside_speech_live_ratio": float(
                    score["outside_speech_live_ratio"]
                ),
                "correct_live_precision_during_speech": float(
                    score["correct_live_precision_during_speech"]
                ),
            }
        )
    per_video_runs: dict[str, list[float]] = {}
    for item in run_scores:
        per_video_runs.setdefault(item["video_id"], []).append(
            item["strict_browser_live_score"]
        )
    per_video = {
        video_id: mean(values)
        for video_id, values in sorted(per_video_runs.items())
    }
    macro = mean(per_video.values())
    if not math.isfinite(macro):
        raise ValueError("Candidate produced a non-finite macro score")
    return {
        "selection_score_contract_id": SCORER_CONTRACT_ID,
        "macro_score": macro,
        "per_video": per_video,
        "runs": run_scores,
        "elapsed_seconds": time.monotonic() - started,
    }


def _successful_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in records
        if item.get("record_type") == "candidate_result"
        and item.get("status") == "evaluated"
    ]


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = _successful_records(records)
    return max(successful, key=lambda item: float(item["macro_score"]), default=None)


def _nominee_artifact(
    result: dict[str, Any],
    incumbent: dict[str, Any],
    frozen: FrozenInputs,
    base_artifact_path: Path,
) -> dict[str, Any]:
    return {
        "status": REPLAY_ONLY_STATUS,
        "contract_id": RUNNER_CONTRACT_ID,
        "candidate_id": result["candidate_id"],
        "candidate_identity": result["candidate_identity"],
        "algorithm_config_sha256": result["algorithm_config_sha256"],
        "algorithm_config": result["algorithm_config"],
        "hypothesis": result["hypothesis"],
        "parent_candidate_id": result.get("parent_candidate_id"),
        "base_artifact": str(Path(base_artifact_path).resolve()),
        "provider_spec": REQUIRED_PROVIDER,
        "windows_seconds": list(REQUIRED_WINDOWS),
        "cadence_seconds": REQUIRED_CADENCE,
        "discovery": {
            "score_contract_id": SCORER_CONTRACT_ID,
            "macro_score": result["macro_score"],
            "delta_vs_incumbent": (
                float(result["macro_score"]) - float(incumbent["macro_score"])
            ),
            "per_video": result["per_video"],
            "run_count": len(result["runs"]),
            "video_count": len(result["per_video"]),
            "diagnostic_parity_exception": frozen.diagnostic_parity_exception,
        },
        "evidence_binding": frozen.candidate_binding(),
        "production_promotion_eligible": False,
        "real_gui_live_e2e_verified": False,
        "promotion": {
            "eligible": False,
            "reason": (
                "Discovery replay only. Every behavior change must beat the incumbent "
                "in authentic visible-Chrome, wall-clock 1x GUI E2E before adoption."
            ),
        },
    }


def _run_search_locked(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    parity_path = Path(args.parity_report or campaign_root / "baseline_parity_report.json")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "journal.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    progress_path = output_dir / "progress.json"

    frozen = freeze_inputs(
        workspace,
        campaign_root,
        parity_path,
        allow_diagnostic_parity=bool(args.allow_diagnostic_parity),
    )
    base_config, _ = load_base_config(Path(args.base_artifact))
    base_artifact_path = Path(args.base_artifact).resolve()
    base_artifact_sha = file_sha256(base_artifact_path)
    base_config_sha = stable_sha256(base_config)
    fixed = fixed_open_set_proposals(base_config)
    immutable_run_identity = {
        "contract_id": RUNNER_CONTRACT_ID,
        "search_space_id": SEARCH_SPACE_ID,
        "seed": int(args.seed),
        "base_artifact_sha256": base_artifact_sha,
        "base_config_sha256": base_config_sha,
        "candidate_binding": frozen.candidate_binding(),
        "diagnostic_parity_exception": frozen.diagnostic_parity_exception,
    }
    immutable_run_identity_sha = stable_sha256(immutable_run_identity)
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("immutable_run_identity_sha256") != immutable_run_identity_sha:
            raise ValueError(
                "Existing output belongs to a different code/data/search identity; "
                "use a new output directory rather than mixing journals"
            )
        created_at = str(manifest.get("created_at") or _utc_now())
    else:
        created_at = _utc_now()
        manifest = {
            "contract_id": RUNNER_CONTRACT_ID,
            "status": "REPLAY_ONLY_DISCOVERY_RUNNING",
            "created_at": created_at,
            "immutable_run_identity_sha256": immutable_run_identity_sha,
            "immutable_run_identity": immutable_run_identity,
            "campaign_root": str(campaign_root),
            "parity_report": str(parity_path.resolve()),
            "base_artifact": str(base_artifact_path),
            "output_dir": str(output_dir),
            "journal": str(journal_path),
            "production_promotion_eligible": False,
            "may_change_launcher": False,
        }
        atomic_write_json(manifest_path, manifest)

    records = read_journal(journal_path)
    for record in records:
        if record.get("immutable_run_identity_sha256") not in {
            None,
            immutable_run_identity_sha,
        }:
            raise ValueError("Journal contains a foreign immutable run identity")
    session_record = {
        "record_type": "session_started" if not records else "session_resumed",
        "timestamp": _utc_now(),
        "pid": os.getpid(),
        "immutable_run_identity_sha256": immutable_run_identity_sha,
        "requested_max_evaluations": int(args.max_evaluations),
        "requested_time_limit_seconds": float(args.time_limit_seconds),
        "target_replay_score": float(args.target_replay_score),
        "status": "REPLAY_ONLY_DISCOVERY",
    }
    append_fsync_jsonl(journal_path, session_record)
    records.append(session_record)

    completed = {
        str(item.get("candidate_id")): item
        for item in records
        if item.get("record_type") in {"candidate_result", "candidate_error"}
    }
    completed_config_hashes = {
        str(item.get("algorithm_config_sha256"))
        for item in completed.values()
        if item.get("algorithm_config_sha256")
    }
    search_indices = [
        int(item["search_index"])
        for item in completed.values()
        if isinstance(item.get("search_index"), int)
    ]
    if completed and not any(
        item.get("record_type") == "candidate_result"
        and item.get("status") == "evaluated"
        and int(item.get("search_index", -1)) == 0
        for item in completed.values()
    ):
        raise ValueError(
            "The journal has no successful immutable incumbent at search_index=0; "
            "refusing to optimize against a substitute baseline"
        )
    next_search_index = max(search_indices, default=-1) + 1
    evaluated_this_session = 0
    session_started = time.monotonic()
    stop_reason = "max_evaluations"

    try:
        while len(completed) < int(args.max_evaluations):
            if (
                float(args.time_limit_seconds) > 0.0
                and time.monotonic() - session_started >= float(args.time_limit_seconds)
            ):
                stop_reason = "time_limit"
                break
            successful = _successful_records(records)
            best = _best_record(records)
            parent_config = dict(best["algorithm_config"]) if best else base_config
            parent_id = str(best["candidate_id"]) if best else None

            salt = 0
            while True:
                if next_search_index < len(fixed):
                    hypothesis, config = fixed[next_search_index]
                else:
                    hypothesis, config = adaptive_proposal(
                        parent_config,
                        seed=int(args.seed),
                        search_index=next_search_index,
                        salt=salt,
                    )
                candidate_id, config_sha, identity = candidate_identity(config, frozen)
                if config_sha not in completed_config_hashes:
                    break
                next_search_index += 1
                salt += 1

            assert_inputs_unchanged(frozen)
            print(
                f"[{len(completed) + 1}/{args.max_evaluations}] "
                f"{candidate_id} {hypothesis}",
                flush=True,
            )
            try:
                evaluation = evaluate_candidate(frozen, config)
            except Exception as exc:
                record = {
                    "record_type": "candidate_error",
                    "status": "evaluation_failed",
                    "timestamp": _utc_now(),
                    "immutable_run_identity_sha256": immutable_run_identity_sha,
                    "search_index": next_search_index,
                    "candidate_id": candidate_id,
                    "candidate_identity": identity,
                    "algorithm_config_sha256": config_sha,
                    "algorithm_config": config,
                    "hypothesis": hypothesis,
                    "parent_candidate_id": parent_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "production_promotion_eligible": False,
                    "result_classification": "REPLAY_ONLY",
                }
                if args.fail_fast or next_search_index == 0:
                    stop_reason = "evaluation_error"
                    append_fsync_jsonl(journal_path, record)
                    raise
            else:
                # If source, scorer, tape, or canonical inputs changed while a
                # candidate ran, abort without journaling a misleading score.
                assert_inputs_unchanged(frozen)
                record = {
                    "record_type": "candidate_result",
                    "status": "evaluated",
                    "timestamp": _utc_now(),
                    "immutable_run_identity_sha256": immutable_run_identity_sha,
                    "search_index": next_search_index,
                    "candidate_id": candidate_id,
                    "candidate_identity": identity,
                    "algorithm_config_sha256": config_sha,
                    "algorithm_config": config,
                    "hypothesis": hypothesis,
                    "parent_candidate_id": parent_id,
                    **evaluation,
                    "production_promotion_eligible": False,
                    "result_classification": "REPLAY_ONLY",
                }

            append_fsync_jsonl(journal_path, record)
            records.append(record)
            completed[candidate_id] = record
            completed_config_hashes.add(config_sha)
            next_search_index += 1
            evaluated_this_session += 1

            successful = _successful_records(records)
            incumbent = next(
                (
                    item
                    for item in successful
                    if int(item.get("search_index", -1)) == 0
                ),
                successful[0] if successful else None,
            )
            best = _best_record(records)
            if incumbent and best:
                nominee = _nominee_artifact(
                    best, incumbent, frozen, base_artifact_path
                )
                nominee_path = output_dir / "nominees" / f"{best['candidate_id']}.json"
                if not nominee_path.exists():
                    atomic_write_json(nominee_path, nominee)
                atomic_write_json(output_dir / "best_replay_only_nominee.json", nominee)
                print(
                    f"  score={float(record.get('macro_score', float('nan'))):.6f} "
                    f"best={float(best['macro_score']):.6f} "
                    f"delta={float(best['macro_score']) - float(incumbent['macro_score']):+.6f}",
                    flush=True,
                )
                if float(best["macro_score"]) >= float(args.target_replay_score):
                    stop_reason = "target_replay_score"
                    break
            atomic_write_json(
                progress_path,
                {
                    "contract_id": RUNNER_CONTRACT_ID,
                    "status": "REPLAY_ONLY_DISCOVERY_RUNNING",
                    "updated_at": _utc_now(),
                    "completed_candidates": len(completed),
                    "successful_candidates": len(successful),
                    "requested_max_evaluations": int(args.max_evaluations),
                    "evaluated_this_session": evaluated_this_session,
                    "best_candidate_id": best.get("candidate_id") if best else None,
                    "best_macro_score": best.get("macro_score") if best else None,
                    "production_promotion_eligible": False,
                },
            )
        else:
            stop_reason = "max_evaluations"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception:
        if stop_reason == "max_evaluations":
            stop_reason = "fatal_error"
        raise
    finally:
        records = read_journal(journal_path)
        best = _best_record(records)
        final_record = {
            "record_type": "session_stopped",
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "immutable_run_identity_sha256": immutable_run_identity_sha,
            "reason": stop_reason,
            "completed_candidates": len(
                {
                    item.get("candidate_id")
                    for item in records
                    if item.get("record_type")
                    in {"candidate_result", "candidate_error"}
                }
            ),
            "best_candidate_id": best.get("candidate_id") if best else None,
            "best_macro_score": best.get("macro_score") if best else None,
            "status": "REPLAY_ONLY_DISCOVERY",
            "production_promotion_eligible": False,
        }
        append_fsync_jsonl(journal_path, final_record)
        manifest["status"] = "REPLAY_ONLY_DISCOVERY_STOPPED"
        manifest["updated_at"] = final_record["timestamp"]
        manifest["stop_reason"] = stop_reason
        manifest["best_candidate_id"] = final_record["best_candidate_id"]
        manifest["best_macro_score"] = final_record["best_macro_score"]
        manifest["production_promotion_eligible"] = False
        atomic_write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "REPLAY_ONLY_DISCOVERY_STOPPED",
                "reason": stop_reason,
                "best_candidate_id": (
                    _best_record(read_journal(journal_path)) or {}
                ).get("candidate_id"),
                "best_macro_score": (
                    _best_record(read_journal(journal_path)) or {}
                ).get("macro_score"),
                "output_dir": str(output_dir),
                "production_promotion_eligible": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 130 if stop_reason == "keyboard_interrupt" else 0


def run_search(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_writer_lock(
        output_dir / ".writer.lock",
        recover_stale=bool(args.recover_stale_lock),
    ):
        return _run_search_locked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable replay-only World-Tape parameter discovery. This command "
            "cannot promote a candidate or modify a launcher."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-evaluations", type=int, default=512)
    parser.add_argument("--time-limit-seconds", type=float, default=0.0)
    parser.add_argument("--target-replay-score", type=float, default=0.9)
    parser.add_argument(
        "--allow-diagnostic-parity",
        action="store_true",
        help=(
            "Explicitly permit optimization_eligible=false input for replay-only "
            "diagnostic discovery. The exception is recorded in every artifact."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="Recover a reviewed stale writer lock from another host.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_evaluations < 1:
        raise ValueError("--max-evaluations must be positive")
    if args.time_limit_seconds < 0.0:
        raise ValueError("--time-limit-seconds must be non-negative")
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
