"""Score browser-observed live speaker UI state against canonical diarization."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from window.live_speaker_probe_scoring import (
    Interval,
    canonical_speaker_turns,
    f1_score,
    interval_total,
    intervals_for_speaker,
    live_turn_latency_report,
    merge_intervals,
    overlap_seconds,
    quantiles,
    read_canonical_segments,
    safe_ratio,
    subtract_intervals,
)


DEFAULT_BROWSER_OBSERVATION_INTERVAL_SECONDS = 0.1
DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS = 0.5
DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS = 0.25


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _speaker_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    result: list[str] = []
    for item in values:
        speaker = str(item or "").strip()
        if speaker and speaker != "UNKNOWN":
            result.append(speaker)
    return result


def visible_live_speaker_id(sample: dict[str, Any]) -> str:
    dom_ids = _speaker_ids(sample.get("dom_live_speaker_ids"))
    current = str(sample.get("current_live_speaker_id") or "").strip()
    visible = str(sample.get("visible_live_speaker_id") or "").strip()
    fallback = str(sample.get("fallback_live_speaker_id") or "").strip()
    transcript = str(sample.get("transcript_live_speaker_id") or "").strip()
    if visible and visible != "UNKNOWN":
        return visible
    if len(dom_ids) == 1:
        return dom_ids[0]
    for candidate in (current, fallback, transcript):
        if candidate and candidate != "UNKNOWN" and (not dom_ids or candidate in dom_ids):
            return candidate
    return dom_ids[0] if dom_ids else ""


def normalize_browser_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            continue
        playback_time = _finite_float(item.get("playback_time"), -1.0)
        if playback_time < 0.0:
            continue
        dom_ids = _speaker_ids(item.get("dom_live_speaker_ids"))
        speaker = visible_live_speaker_id(item)
        normalized.append({
            "index": index,
            "playback_time": playback_time,
            "wall_time": _finite_float(item.get("wall_time"), 0.0),
            "performance_ms": _finite_float(item.get("performance_ms"), 0.0),
            "speaker": speaker,
            "dom_live_speaker_ids": dom_ids,
            "current_live_speaker_id": str(item.get("current_live_speaker_id") or ""),
            "runtime_state": str(item.get("runtime_state") or ""),
        })
    normalized.sort(key=lambda value: (float(value["playback_time"]), int(value["index"])))
    return normalized


def browser_observed_state_slices(
    samples: list[dict[str, Any]],
    *,
    max_sample_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    normalized = normalize_browser_samples(samples)
    max_gap = max(0.01, float(max_sample_gap_seconds))
    slices: list[dict[str, Any]] = []
    for current, nxt in zip(normalized, normalized[1:]):
        start = float(current["playback_time"])
        next_time = float(nxt["playback_time"])
        if next_time <= start:
            continue
        end = min(next_time, start + max_gap)
        if end <= start:
            continue
        speaker = str(current.get("speaker") or "")
        if slices and slices[-1]["speaker"] == speaker and float(slices[-1]["end"]) >= start - 0.001:
            slices[-1]["end"] = round(end, 4)
            slices[-1]["duration_seconds"] = round(float(slices[-1]["end"]) - float(slices[-1]["start"]), 4)
        else:
            slices.append({
                "speaker": speaker,
                "start": round(start, 4),
                "end": round(end, 4),
                "duration_seconds": round(end - start, 4),
            })
    return [item for item in slices if float(item["end"]) > float(item["start"])]


def browser_observed_live_slices(
    samples: list[dict[str, Any]],
    *,
    max_sample_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    return [
        item
        for item in browser_observed_state_slices(samples, max_sample_gap_seconds=max_sample_gap_seconds)
        if str(item.get("speaker") or "")
    ]


def one_to_one_profile_map(
    live_slices: list[dict[str, Any]],
    canonical_by_speaker: dict[str, list[Interval]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    profile_ids = sorted({str(item["speaker"]) for item in live_slices if str(item.get("speaker") or "")})
    matrix: list[dict[str, Any]] = []
    pairs: list[tuple[float, str, str]] = []
    for profile in profile_ids:
        interval = [
            (float(item["start"]), float(item["end"]))
            for item in live_slices
            if str(item.get("speaker") or "") == profile
        ]
        overlaps: dict[str, float] = {}
        for canonical_speaker, canonical_intervals in canonical_by_speaker.items():
            seconds = overlap_seconds(interval, canonical_intervals)
            if seconds > 0.0:
                overlaps[canonical_speaker] = seconds
                pairs.append((seconds, profile, canonical_speaker))
        matrix.append({
            "speaker": profile,
            "canonical_overlaps": {
                speaker: round(seconds, 4)
                for speaker, seconds in sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
            },
        })
    mapping: dict[str, str] = {}
    used_canonical: set[str] = set()
    for seconds, profile, canonical_speaker in sorted(pairs, key=lambda item: item[0], reverse=True):
        if seconds <= 0.0 or profile in mapping or canonical_speaker in used_canonical:
            continue
        mapping[profile] = canonical_speaker
        used_canonical.add(canonical_speaker)
    return mapping, matrix


def _intervals_for_slices(slices: list[dict[str, Any]], speaker: str | None = None) -> list[Interval]:
    return [
        (float(item["start"]), float(item["end"]))
        for item in slices
        if speaker is None or str(item.get("speaker") or "") == speaker
    ]


def browser_state_transition_report(state_slices: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    previous = ""
    initialized = False
    for item in state_slices:
        speaker = str(item.get("speaker") or "")
        if not initialized:
            previous = speaker
            initialized = True
            continue
        if speaker == previous:
            continue
        if previous and speaker:
            transitions["speaker_switch"] += 1
        elif previous and not speaker:
            transitions["live_drop"] += 1
        elif not previous and speaker:
            transitions["live_acquire"] += 1
        previous = speaker
    return {
        "state_transition_count": int(sum(transitions.values())),
        "speaker_switch_count": int(transitions["speaker_switch"]),
        "live_drop_count": int(transitions["live_drop"]),
        "live_acquire_count": int(transitions["live_acquire"]),
    }


def _large_gaps(intervals: list[Interval], min_gap_seconds: float) -> list[Interval]:
    threshold = max(0.0, float(min_gap_seconds))
    return [(start, end) for start, end in intervals if end - start >= threshold]


def browser_flicker_report(
    canonical_segments: list[dict[str, Any]],
    live_slices: list[dict[str, Any]],
    profile_map: dict[str, str],
    *,
    min_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
) -> dict[str, Any]:
    turns = canonical_speaker_turns(canonical_segments, 0.5)
    live_intervals = _intervals_for_slices(live_slices)
    no_live_gaps: list[Interval] = []
    correct_gap_intervals: list[Interval] = []
    correct_interruption_gaps: list[Interval] = []
    for turn in turns:
        turn_interval = (float(turn["start"]), float(turn["end"]))
        no_live_gaps.extend(_large_gaps(subtract_intervals(turn_interval, live_intervals), min_gap_seconds))
        correct_intervals = [
            (float(item["start"]), float(item["end"]))
            for item in live_slices
            if profile_map.get(str(item.get("speaker") or "")) == str(turn["speaker"])
        ]
        correct_gap_intervals.extend(
            _large_gaps(subtract_intervals(turn_interval, correct_intervals), min_gap_seconds)
        )
        clipped_correct = merge_intervals([
            (max(turn_interval[0], start), min(turn_interval[1], end))
            for start, end in correct_intervals
            if end > turn_interval[0] and start < turn_interval[1]
        ])
        for left, right in zip(clipped_correct, clipped_correct[1:]):
            if right[0] > left[1] and right[0] - left[1] >= min_gap_seconds:
                correct_interruption_gaps.append((left[1], right[0]))
    return {
        "min_gap_seconds": round(max(0.0, float(min_gap_seconds)), 4),
        "no_live_gap_count": len(no_live_gaps),
        "no_live_gap_seconds": round(interval_total(no_live_gaps), 4),
        "not_correct_gap_count": len(correct_gap_intervals),
        "not_correct_gap_seconds": round(interval_total(correct_gap_intervals), 4),
        "correct_interruption_count": len(correct_interruption_gaps),
        "correct_interruption_seconds": round(interval_total(correct_interruption_gaps), 4),
        "largest_no_live_gaps": [
            {"start": round(start, 4), "end": round(end, 4), "duration_seconds": round(end - start, 4)}
            for start, end in sorted(no_live_gaps, key=lambda item: item[1] - item[0], reverse=True)[:10]
        ],
        "largest_correct_interruption_gaps": [
            {"start": round(start, 4), "end": round(end, 4), "duration_seconds": round(end - start, 4)}
            for start, end in sorted(correct_interruption_gaps, key=lambda item: item[1] - item[0], reverse=True)[:10]
        ],
    }


def score_browser_live_speaker_samples(
    samples: list[dict[str, Any]],
    canonical_segments: list[dict[str, Any]],
    *,
    max_sample_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
    flicker_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
) -> dict[str, Any]:
    canonical_by_speaker = intervals_for_speaker(canonical_segments)
    canonical_intervals = merge_intervals([
        interval
        for intervals in canonical_by_speaker.values()
        for interval in intervals
    ])
    canonical_total = interval_total(canonical_intervals)
    normalized_samples = normalize_browser_samples(samples)
    state_slices = browser_observed_state_slices(
        samples,
        max_sample_gap_seconds=max_sample_gap_seconds,
    )
    live_slices = [item for item in state_slices if str(item.get("speaker") or "")]
    live_intervals = _intervals_for_slices(live_slices)
    live_total = interval_total(live_intervals)
    live_speech_overlap = overlap_seconds(live_intervals, canonical_intervals)
    profile_map, overlap_matrix = one_to_one_profile_map(live_slices, canonical_by_speaker)
    correct_overlap = 0.0
    for item in live_slices:
        mapped = profile_map.get(str(item.get("speaker") or ""))
        if not mapped:
            continue
        correct_overlap += overlap_seconds(
            [(float(item["start"]), float(item["end"]))],
            canonical_by_speaker.get(mapped, []),
        )
    missing_speech = max(0.0, canonical_total - live_speech_overlap)
    wrong_speech = max(0.0, live_speech_overlap - correct_overlap)
    outside_speech = max(0.0, live_total - live_speech_overlap)
    correct_coverage = safe_ratio(correct_overlap, canonical_total)
    correct_precision_during_speech = safe_ratio(correct_overlap, live_speech_overlap)
    correct_precision_total = safe_ratio(correct_overlap, live_total)
    turn_latency = live_turn_latency_report(
        canonical_segments,
        live_slices,
        profile_map,
    )
    flicker = browser_flicker_report(
        canonical_segments,
        live_slices,
        profile_map,
        min_gap_seconds=flicker_gap_seconds,
    )
    wrong_ratio = safe_ratio(wrong_speech, canonical_total)
    outside_ratio = safe_ratio(outside_speech, canonical_total)
    flicker_ratio = safe_ratio(float(flicker["correct_interruption_seconds"]), canonical_total)
    strict_score = max(
        0.0,
        min(
            1.0,
            correct_coverage
            - wrong_ratio
            - (0.25 * outside_ratio)
            - (0.25 * flicker_ratio),
        ),
    )
    playback_times = [float(item["playback_time"]) for item in normalized_samples]
    return {
        "system": "browser_live_speaker_dom",
        "sample_count": len(samples),
        "usable_sample_count": len(normalized_samples),
        "sampled_playback_seconds": {
            "min": round(min(playback_times), 4) if playback_times else None,
            "max": round(max(playback_times), 4) if playback_times else None,
        },
        "max_sample_gap_seconds": round(max(0.01, float(max_sample_gap_seconds)), 4),
        "canonical_speech_seconds": round(canonical_total, 4),
        "browser_live_seconds": round(live_total, 4),
        "browser_live_speech_overlap_seconds": round(live_speech_overlap, 4),
        "correct_live_seconds": round(correct_overlap, 4),
        "wrong_live_speech_seconds": round(wrong_speech, 4),
        "missing_live_speech_seconds": round(missing_speech, 4),
        "live_assignment_outside_speech_seconds": round(outside_speech, 4),
        "correct_live_speaker_coverage": round(correct_coverage, 6),
        "missing_live_speech_ratio": round(safe_ratio(missing_speech, canonical_total), 6),
        "wrong_live_speech_ratio": round(wrong_ratio, 6),
        "outside_speech_live_ratio": round(outside_ratio, 6),
        "correct_live_precision_during_speech": round(correct_precision_during_speech, 6),
        "correct_live_precision_total": round(correct_precision_total, 6),
        "correct_live_speaker_f1": round(f1_score(correct_coverage, correct_precision_during_speech), 6),
        "strict_browser_live_score": round(strict_score, 6),
        "speaker_map": profile_map,
        "overlap_matrix": overlap_matrix,
        "turn_latency": turn_latency,
        "flicker": flicker,
        "state_transitions": browser_state_transition_report(state_slices),
        "browser_live_slices": live_slices,
    }


def read_browser_observation_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        samples = data.get("samples")
        if isinstance(samples, list):
            return [item for item in samples if isinstance(item, dict)]
    raise ValueError(f"Could not read browser observation samples from {path}")


class BrowserLiveObservationRecorder:
    def __init__(
        self,
        *,
        output_path: Path,
        canonical_path: Path,
        max_sample_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS,
        flicker_gap_seconds: float = DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS,
    ) -> None:
        self.output_path = output_path
        self.canonical_path = canonical_path
        self.max_sample_gap_seconds = max_sample_gap_seconds
        self.flicker_gap_seconds = flicker_gap_seconds
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._finished = False

    @property
    def enabled(self) -> bool:
        return True

    def record(self, samples: list[dict[str, Any]]) -> int:
        clean_samples = [dict(item) for item in samples if isinstance(item, dict)]
        with self._lock:
            if self._finished:
                return len(self._samples)
            self._samples.extend(clean_samples)
            return len(self._samples)

    def finish(self, reason: str = "done") -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            self._finished = True
        canonical_segments = read_canonical_segments(self.canonical_path)
        summary = score_browser_live_speaker_samples(
            samples,
            canonical_segments,
            max_sample_gap_seconds=self.max_sample_gap_seconds,
            flicker_gap_seconds=self.flicker_gap_seconds,
        )
        payload = {
            "summary": {
                **summary,
                "reason": reason,
                "elapsed_wall_seconds": round(time.time() - self._started_at, 4),
                "canonical": str(self.canonical_path),
            },
            "samples": samples,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload["summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score browser-observed live speaker samples.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sample-gap-seconds", type=float, default=DEFAULT_BROWSER_OBSERVATION_MAX_SAMPLE_GAP_SECONDS)
    parser.add_argument("--flicker-gap-seconds", type=float, default=DEFAULT_BROWSER_OBSERVATION_FLICKER_GAP_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_browser_observation_file(args.observations)
    canonical_segments = read_canonical_segments(args.canonical)
    summary = score_browser_live_speaker_samples(
        samples,
        canonical_segments,
        max_sample_gap_seconds=args.max_sample_gap_seconds,
        flicker_gap_seconds=args.flicker_gap_seconds,
    )
    summary["observations"] = str(args.observations)
    summary["canonical"] = str(args.canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Browser live speaker score: {args.output}", flush=True)
    print(f"Strict browser live score: {summary['strict_browser_live_score']:.3f}", flush=True)
    print(
        "Correct visible live speaker coverage: "
        f"{summary['correct_live_seconds']:.2f}s / {summary['canonical_speech_seconds']:.2f}s "
        f"({summary['correct_live_speaker_coverage']:.3f})",
        flush=True,
    )
    print(
        "Missing / wrong speech seconds: "
        f"{summary['missing_live_speech_seconds']:.2f}s / {summary['wrong_live_speech_seconds']:.2f}s",
        flush=True,
    )
    print(f"Flicker: {summary['flicker']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
