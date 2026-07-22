"""Counterfactual live-speaker replay over an authentic World Tape.

The recorded ASR, final-sentence/profile world, probe admissions, vectors, and wall-clock
times stay frozen.  Only the shared causal tracker configuration changes.  Candidate
decisions are projected back into the public live/clear stream and then consumed by the
browser-state diagnostic reducer at the original DOM sample timestamps.
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_algorithm import (
    CausalLiveSpeakerAlgorithm,
    LiveSpeakerAlgorithmConfig,
    LiveSpeakerStep,
)
from window.live_speaker_bayes import BayesSpeakerTrackerConfig, CausalBayesSpeakerTracker
from window.live_speaker_browser_parity import replay_browser_state
from window.live_speaker_multiscale import MultiScaleEvidence, MultiScaleStep
from window.live_speaker_open_set_tracklets import (
    OPEN_SET_TRACKLET_PRESET,
    OpenSetTrackletOverlay,
    OpenSetTrackletStep,
    open_set_tracklet_config_for_preset,
)
from window.live_speaker_parity_replay import (
    _resolve_arrays,
    read_world_tape_events,
)
from window.live_speaker_probe_scoring import read_canonical_segments


COUNTERFACTUAL_REPLAY_ID = "whospeaks.live_world_tape.counterfactual_diagnostic.v1"


def _config_values(config_type: type, raw: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(config_type)}
    return {name: value for name, value in raw.items() if name in names}


def _public_probability_key(label: str) -> str:
    value = str(label or "")
    if value.startswith("S") and value[1:].isdigit():
        return f"speaker{int(value[1:])}"
    return value


def _mapped_values(
    values: dict[str, Any],
    aliases: dict[str, str],
    *,
    probability_keys: bool = False,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_label, raw_value in (values or {}).items():
        label = str(raw_label)
        if label != "unknown":
            label = aliases.get(label, label)
            if probability_keys:
                label = _public_probability_key(label)
        result[label] = max(float(raw_value), float(result.get(label, 0.0)))
    return result


def _algorithm(kind: str, raw_config: dict[str, Any]):
    if kind == "bayes":
        return CausalBayesSpeakerTracker(
            BayesSpeakerTrackerConfig(
                **_config_values(BayesSpeakerTrackerConfig, raw_config)
            )
        )
    return CausalLiveSpeakerAlgorithm(
        LiveSpeakerAlgorithmConfig(
            **_config_values(LiveSpeakerAlgorithmConfig, raw_config)
        )
    )


def _step(algorithm: Any, kind: str, payload: dict[str, Any]):
    embedding = payload.get("embedding")
    context_embedding = payload.get("context_embedding")
    media_time = float(payload.get("media_time") or 0.0)
    if kind == "bayes":
        windows = tuple(float(value) for value in algorithm.config.scale_windows)
        evidences: list[MultiScaleEvidence] = []
        if embedding is not None:
            evidences.append(
                MultiScaleEvidence(
                    windows[0], np.asarray(embedding, dtype=np.float32)
                )
            )
        if context_embedding is not None:
            evidences.append(
                MultiScaleEvidence(
                    windows[-1], np.asarray(context_embedding, dtype=np.float32)
                )
            )
        return algorithm.step(
            MultiScaleStep(
                media_time=media_time,
                speech=bool(payload.get("speech")),
                evidences=tuple(evidences),
                probe_scheduled=bool(payload.get("probe_scheduled")),
                release_signal=bool(payload.get("release_signal")),
                skipped_reason=str(payload.get("skipped_reason") or ""),
            )
        )
    return algorithm.step(
        LiveSpeakerStep(
            media_time=media_time,
            speech=bool(payload.get("speech")),
            embedding=(
                None if embedding is None else np.asarray(embedding, dtype=np.float32)
            ),
            duration_seconds=float(payload.get("duration_seconds") or 0.0),
            probe_scheduled=bool(payload.get("probe_scheduled")),
            release_signal=bool(payload.get("release_signal")),
            embedding_latency_seconds=payload.get("embedding_latency_seconds"),
            skipped_reason=str(payload.get("skipped_reason") or ""),
        )
    )


@lru_cache(maxsize=32)
def _cached_counterfactual_tape_inputs(
    root_key: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    float,
]:
    """Resolve immutable tracker inputs and recorded outputs once per World Tape."""

    root = Path(root_key)
    records = read_world_tape_events(root)
    input_records: list[dict[str, Any]] = []
    recorded_decisions: dict[int, dict[str, Any]] = {}
    public_by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        payload = dict(record.get("payload") or {})
        stream = str(record.get("stream") or "")
        event = str(record.get("event") or "")
        step_id = int(payload.get("step_id") or 0)
        if stream == "internal" and event == "live_speaker_core_input":
            resolved = dict(record)
            resolved["payload"] = _resolve_arrays(root, payload)
            input_records.append(resolved)
        elif stream == "internal" and event == "live_speaker_core_decision" and step_id:
            recorded_decisions[step_id] = record
        elif stream == "public" and event in {"live_speaker", "live_speaker_clear"} and step_id:
            public_by_step[step_id] = record
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runtime_config = dict(manifest.get("runtime_config") or {})
    hold_seconds = max(
        0.0, float(runtime_config.get("live_speaker_probe_hold_seconds") or 1.0)
    )
    return tuple(input_records), recorded_decisions, public_by_step, hold_seconds


def project_counterfactual_live_actions(
    tape_dir: Path,
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(tape_dir).resolve()
    (
        input_records,
        recorded_decisions,
        public_by_step,
        hold_seconds,
    ) = _cached_counterfactual_tape_inputs(str(root))
    hold_seconds = max(
        0.0,
        float(algorithm_config.get("live_speaker_probe_hold_seconds", hold_seconds)),
    )

    algorithm = None
    algorithm_kind = ""
    overlay = (
        OpenSetTrackletOverlay(open_set_tracklet_config_for_preset(
            preset=str(
                algorithm_config.get("live_speaker_open_set_tracklet_preset")
                or algorithm_config.get("open_set_tracklet_preset")
                or OPEN_SET_TRACKLET_PRESET
            )
        ))
        if bool(
            algorithm_config.get("live_speaker_open_set_tracklets")
            or algorithm_config.get("enable_open_set_tracklets")
        )
        else None
    )
    active_public_speaker = ""
    aliases: dict[str, str] = {}
    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    decision_rows: list[dict[str, Any]] = []
    for record in input_records:
        payload = dict(record["payload"])
        step_id = int(payload.get("step_id") or 0)
        kind = str(payload.get("algorithm_type") or "classic")
        if algorithm is None or algorithm_kind != kind:
            algorithm = _algorithm(kind, algorithm_config)
            algorithm_kind = kind
        algorithm.sync_profiles(list(payload.get("profiles") or []))
        decision = _step(algorithm, kind, payload)
        trace = decision.trace_record()

        overlay_decision = None
        if overlay is not None:
            overlay_decision = overlay.step(OpenSetTrackletStep(
                media_time=float(payload.get("media_time") or 0.0),
                speech=bool(payload.get("speech")),
                probe_scheduled=bool(str(payload.get("probe_id") or "")),
                release_signal=bool(payload.get("release_signal")),
                short_embedding=(
                    None
                    if payload.get("embedding") is None
                    else np.asarray(payload.get("embedding"), dtype=np.float32)
                ),
                long_embedding=(
                    None
                    if payload.get("context_embedding") is None
                    else np.asarray(payload.get("context_embedding"), dtype=np.float32)
                ),
                profiles=tuple(dict(item) for item in payload.get("profiles") or []),
                base_visible_speaker=decision.visible_speaker,
                base_action=decision.action,
                base_reason=decision.reason,
            ))
            trace = {
                **trace,
                "visible_speaker": overlay_decision.visible_speaker,
                "candidate_speaker": overlay_decision.visible_speaker,
                "action": overlay_decision.action,
                "reason": overlay_decision.reason,
                "diagnostics": {
                    **dict(trace.get("diagnostics") or {}),
                    "open_set_tracklet": dict(overlay_decision.diagnostics),
                },
            }

        recorded_decision = dict(
            (recorded_decisions.get(step_id) or {}).get("payload") or {}
        )
        recorded_public = public_by_step.get(step_id)
        recorded_public_payload = dict(
            (recorded_public or {}).get("payload") or {}
        )
        recorded_internal = str(recorded_decision.get("visible_speaker") or "")
        recorded_external = str(
            recorded_public_payload.get("assigned_speaker")
            or recorded_public_payload.get("speaker_id")
            or ""
        )
        if recorded_internal and recorded_external and recorded_internal != recorded_external:
            aliases[recorded_internal] = recorded_external

        internal_speaker = str(trace.get("visible_speaker") or "")
        public_speaker = aliases.get(internal_speaker, internal_speaker)
        raw_probabilities = _mapped_values(
            dict(trace.get("raw_probabilities") or {}),
            aliases,
            probability_keys=True,
        )
        probabilities = _mapped_values(
            dict(trace.get("probabilities") or {}),
            aliases,
            probability_keys=True,
        )
        similarities = _mapped_values(
            dict(trace.get("similarities") or {}), aliases
        )
        media_time = float(payload.get("media_time") or 0.0)
        duration = max(0.0, float(payload.get("duration_seconds") or 0.0))
        dedicated_probe = bool(str(payload.get("probe_id") or ""))
        release_signal = bool(payload.get("release_signal"))
        start = round(max(0.0, media_time - duration), 4)
        end = round(media_time, 4)
        event_wall = float(
            (recorded_public or recorded_decisions.get(step_id) or record).get("wall_us")
            or 0
        ) / 1_000_000.0
        event_sequence = int(
            (recorded_public or recorded_decisions.get(step_id) or record).get("seq")
            or record.get("seq")
            or 0
        )
        base = recorded_public_payload
        if overlay_decision is not None:
            for alias in overlay_decision.aliases:
                actions.append((
                    event_wall,
                    max(0, event_sequence - 1),
                    "live_speaker_identity_alias",
                    {
                        "step_id": step_id,
                        **alias.payload(),
                        "media_time": media_time,
                        "final_to_public": overlay.final_to_public,
                        "public_to_final": overlay.public_to_final,
                        "assignment_source": "open_set_tracklet_profile_merge",
                    },
                ))
        if dedicated_probe and public_speaker and not release_signal:
            live_payload = {
                **base,
                "step_id": step_id,
                "assigned_speaker": public_speaker,
                "speaker_id": public_speaker,
                "internal_speaker_id": internal_speaker,
                "replaces_speaker_id": (
                    internal_speaker if internal_speaker != public_speaker else None
                ),
                "probabilities": probabilities,
                "raw_probabilities": raw_probabilities,
                "similarities": similarities,
                "unknown_probability": float(probabilities.get("unknown", 1.0)),
                "live_speaker_core_action": trace.get("action"),
                "live_speaker_core_reason": trace.get("reason"),
                "live": True,
                "fallback": True,
                "start": base.get("start", start),
                "end": base.get("end", end),
                "audio_length_seconds": base.get(
                    "audio_length_seconds", round(duration, 4)
                ),
                "hold_seconds": round(hold_seconds, 4),
                "assignment_source": "counterfactual_shared_causal_live_speaker_core",
            }
            actions.append(
                (event_wall, event_sequence, "live_speaker", live_payload)
            )
            active_public_speaker = public_speaker
        elif (
            dedicated_probe
            and str(trace.get("action") or "") == "clear"
            and active_public_speaker
        ):
            reason = (
                "silence"
                if bool(payload.get("release_signal"))
                else str(trace.get("reason") or "unknown")
            )
            clear_payload = {
                **base,
                "step_id": step_id,
                "speaker_id": active_public_speaker,
                "assigned_speaker": None,
                "live": False,
                "fallback": True,
                "start": base.get("start", start),
                "end": base.get("end", end),
                "reason": reason,
                "assignment_source": "counterfactual_shared_causal_live_speaker_core",
            }
            actions.append(
                (event_wall, event_sequence, "live_speaker_clear", clear_payload)
            )
            active_public_speaker = ""
        decision_rows.append(
            {
                "step_id": step_id,
                "media_time": media_time,
                "visible_speaker": public_speaker,
                "internal_speaker": internal_speaker,
                "action": trace.get("action"),
                "reason": trace.get("reason"),
            }
        )
    return {
        "actions": actions,
        "decisions": decision_rows,
        "input_step_count": len(input_records),
    }


def evaluate_counterfactual(
    tape_dir: Path,
    algorithm_config: dict[str, Any],
    canonical_path: Path,
) -> dict[str, Any]:
    projection = project_counterfactual_live_actions(tape_dir, algorithm_config)
    browser = replay_browser_state(
        tape_dir, replacement_live_actions=projection["actions"]
    )
    score = score_browser_live_speaker_samples(
        browser["replayed_samples"], read_canonical_segments(canonical_path)
    )
    return {
        "contract_id": COUNTERFACTUAL_REPLAY_ID,
        "tape_dir": str(Path(tape_dir).resolve()),
        "canonical_path": str(Path(canonical_path).resolve()),
        "input_step_count": projection["input_step_count"],
        "projected_live_action_count": len(projection["actions"]),
        "strict_browser_live_score": score["strict_browser_live_score"],
        "score": score,
        "decisions": projection["decisions"],
    }
