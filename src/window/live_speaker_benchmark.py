"""Versioned scoring and multi-video aggregation for causal live-speaker traces."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from window.browser_live_speaker_scoring import score_browser_live_speaker_samples
from window.live_speaker_algorithm import LiveSpeakerDecision, SpeakerProfileEvent
from window.live_speaker_probe_scoring import canonical_speaker_turns, intervals_for_speaker, merge_intervals


SCORER_ID = "causal_live_speaker_score_v1"
PRIMARY_SCORER_V2_ID = "causal_live_speaker_primary_macro_v2"
EVIDENCE_CLASS = "TRACKER_ONLY_DIAGNOSTIC"


@dataclass(frozen=True)
class LiveSpeakerScoreConfig:
    max_sample_gap_seconds: float = 0.25
    flicker_gap_seconds: float = 0.30
    stable_dwell_seconds: float = 0.30
    release_gap_min_seconds: float = 1.0
    latency_cap_seconds: float = 3.0


def _samples(decisions: Iterable[LiveSpeakerDecision]) -> list[dict[str, Any]]:
    """Create synthetic samples for diagnostics; these are not browser evidence."""

    return [
        {
            "playback_time": item.media_time,
            "current_live_speaker_id": item.visible_speaker or "",
            "dom_live_speaker_ids": [item.visible_speaker] if item.visible_speaker else [],
        }
        for item in decisions
    ]


def _stable_state_at(
    decisions: list[LiveSpeakerDecision],
    start_index: int,
    speaker: str | None,
    dwell_seconds: float,
) -> bool:
    start = decisions[start_index].media_time
    target = start + max(0.0, dwell_seconds)
    for item in decisions[start_index:]:
        if item.visible_speaker != speaker:
            return False
        if item.media_time + 1e-9 >= target:
            return True
    return dwell_seconds <= 0.0


def _release_report(
    decisions: list[LiveSpeakerDecision],
    canonical_segments: list[dict[str, Any]],
    config: LiveSpeakerScoreConfig,
) -> dict[str, Any]:
    speech = merge_intervals([
        interval
        for values in intervals_for_speaker(canonical_segments).values()
        for interval in values
    ])
    gaps = [
        (left[1], right[0])
        for left, right in zip(speech, speech[1:])
        if right[0] - left[1] >= config.release_gap_min_seconds
    ]
    rows: list[dict[str, Any]] = []
    for start, end in gaps:
        released_at = None
        for index, item in enumerate(decisions):
            if item.media_time + 1e-9 < start or item.media_time >= end:
                continue
            if item.visible_speaker is None and _stable_state_at(
                decisions, index, None, config.stable_dwell_seconds
            ):
                released_at = item.media_time
                break
        latency = None if released_at is None else max(0.0, released_at - start)
        rows.append({
            "speech_end": round(start, 4),
            "next_speech_start": round(end, 4),
            "released_at": round(released_at, 4) if released_at is not None else None,
            "latency_seconds": round(latency, 4) if latency is not None else None,
            "missed": released_at is None,
        })
    values = [float(item["latency_seconds"]) for item in rows if item["latency_seconds"] is not None]
    return {
        "eligible_gap_count": len(rows),
        "missed_release_count": sum(bool(item["missed"]) for item in rows),
        "mean_latency_seconds": round(mean(values), 6) if values else None,
        "events": rows,
    }


def _availability_report(
    canonical_segments: list[dict[str, Any]],
    profile_events: Iterable[SpeakerProfileEvent],
    speaker_map: dict[str, str],
) -> dict[str, Any]:
    first_available: dict[str, float] = {}
    for event in profile_events:
        canonical = speaker_map.get(event.speaker_id)
        if canonical:
            first_available[canonical] = min(
                first_available.get(canonical, float("inf")), float(event.available_at)
            )
    categories = {"known_at_turn_start": 0, "becomes_known_during_turn": 0, "never_known_during_turn": 0}
    rows: list[dict[str, Any]] = []
    for turn in canonical_speaker_turns(canonical_segments):
        available = first_available.get(str(turn["speaker"]))
        if available is not None and available <= float(turn["start"]) + 1e-9:
            category = "known_at_turn_start"
        elif available is not None and available < float(turn["end"]):
            category = "becomes_known_during_turn"
        else:
            category = "never_known_during_turn"
        categories[category] += 1
        rows.append({**turn, "profile_available_at": available, "availability": category})
    return {"counts": categories, "turns": rows}


def score_live_speaker_decisions(
    decisions: Iterable[LiveSpeakerDecision],
    canonical_segments: list[dict[str, Any]],
    profile_events: Iterable[SpeakerProfileEvent] = (),
    *,
    config: LiveSpeakerScoreConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LiveSpeakerScoreConfig()
    trace = list(decisions)
    events = list(profile_events)
    base = score_browser_live_speaker_samples(
        _samples(trace),
        canonical_segments,
        max_sample_gap_seconds=cfg.max_sample_gap_seconds,
        flicker_gap_seconds=cfg.flicker_gap_seconds,
    )
    return {
        "scorer_id": SCORER_ID,
        "evidence_class": EVIDENCE_CLASS,
        "promotion_eligible": False,
        **base,
        "release": _release_report(trace, canonical_segments, cfg),
        "profile_availability": _availability_report(
            canonical_segments, events, dict(base.get("speaker_map") or {})
        ),
    }


def aggregate_video_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["strict_browser_live_score"]) for item in scores]
    if not values:
        raise ValueError("At least one video score is required")
    ordered = sorted(values)
    bottom = ordered[: min(3, len(ordered))]
    aggregate = 0.55 * mean(values) + 0.30 * min(values) + 0.15 * mean(bottom)
    return {
        "scorer_id": SCORER_ID,
        "evidence_class": EVIDENCE_CLASS,
        "promotion_eligible": False,
        "video_count": len(values),
        "mean_video_score": round(mean(values), 6),
        "worst_video_score": round(min(values), 6),
        "bottom3_mean_video_score": round(mean(bottom), 6),
        "global_score": round(aggregate, 6),
    }


def aggregate_video_scores_primary_v2(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return the one scalar optimized by the top-seven overnight campaign.

    Selection uses the unweighted macro mean of the existing strict per-video
    score.  Per-video and component values remain visible as diagnostics, but
    they are deliberately not promotion vetoes.
    """

    rows = list(scores)
    if not rows:
        raise ValueError("At least one video score is required")

    def metric(name: str) -> float:
        return mean(float(item.get(name) or 0.0) for item in rows)

    canonical_seconds = sum(float(item.get("canonical_speech_seconds") or 0.0) for item in rows)
    flicker_seconds = sum(
        float((item.get("flicker") or {}).get("correct_interruption_seconds") or 0.0)
        for item in rows
    )
    mean_score = metric("strict_browser_live_score")
    return {
        "scorer_id": PRIMARY_SCORER_V2_ID,
        "component_scorer_id": SCORER_ID,
        "evidence_class": EVIDENCE_CLASS,
        "promotion_eligible": False,
        "video_count": len(rows),
        "primary_score": round(mean_score, 6),
        # Keep global_score as a compatibility alias for generic result viewers.
        "global_score": round(mean_score, 6),
        "mean_video_score": round(mean_score, 6),
        "diagnostics": {
            "mean_correct_live_speaker_coverage": round(
                metric("correct_live_speaker_coverage"), 6
            ),
            "mean_wrong_live_speech_ratio": round(metric("wrong_live_speech_ratio"), 6),
            "mean_outside_speech_live_ratio": round(
                metric("outside_speech_live_ratio"), 6
            ),
            "mean_missing_live_speech_ratio": round(
                metric("missing_live_speech_ratio"), 6
            ),
            "corpus_flicker_ratio": round(
                flicker_seconds / canonical_seconds if canonical_seconds > 0.0 else 0.0,
                6,
            ),
        },
    }
