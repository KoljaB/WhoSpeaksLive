"""Deterministic replay of the live-speaker browser state recorded in a World Tape.

The production browser combines server ``live_speaker`` events with realtime
transcript rows and wall-clock timers.  This module mirrors that causal reducer
without a browser or model inference so cached candidate decisions can later be
scored at the same observation timestamps as a real GUI run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from window.live_speaker_parity_replay import read_world_tape_events


BROWSER_PARITY_REPLAY_ID = "whospeaks.live_world_tape.browser_reducer_parity.v1"
REALTIME_SETTLE_REMOVAL_SECONDS = 1.4
ROW_FADE_REMOVAL_SECONDS = 0.22


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _speaker(value: Any) -> str:
    result = str(value or "").strip()
    return result if result and result != "UNKNOWN" else ""


def _tokens(value: Any) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _text_adoption_score(left: Any, right: Any) -> float:
    tokens_left = _tokens(left)
    tokens_right = _tokens(right)
    if not tokens_left or not tokens_right:
        return 0.0
    remaining: dict[str, int] = {}
    for token in tokens_left:
        remaining[token] = remaining.get(token, 0) + 1
    shared = 0
    for token in tokens_right:
        count = remaining.get(token, 0)
        if count > 0:
            shared += 1
            remaining[token] = count - 1
    return shared / max(1, min(len(tokens_left), len(tokens_right)))


@dataclass
class _RealtimeRow:
    index: str
    start: float
    end: float
    text: str
    raw_speaker: str
    speaker: str = ""
    settling: bool = False
    clear_generation: str = ""
    remove_at: float | None = None


class BrowserLiveSpeakerReducer:
    """Causal state machine corresponding to the browser's visible-speaker path."""

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        config = dict(runtime_config or {})
        self.highlight_transcript = bool(
            config.get("live_speaker_highlight_transcript", True)
        )
        self.transcript_max_lag_seconds = _finite(
            config.get("live_speaker_highlight_transcript_max_lag_seconds"), -1.0
        )
        self.unknown_clear_debounce_seconds = max(
            0.0,
            _finite(
                config.get("live_speaker_probe_unknown_clear_debounce_seconds"),
                0.0,
            ),
        )
        self.timeline: list[dict[str, Any]] = []
        self.rows: list[_RealtimeRow] = []
        self.current_generation = 0
        self.fallback_speaker = ""
        self.fallback_until = 0.0
        self.fallback_clear_at: float | None = None
        self.transcript_speaker = ""
        self.current_speaker = ""
        self.playback_time = 0.0
        self.now = 0.0
        self.alias_generation = 0
        self.final_to_public: dict[str, str] = {}
        self.public_to_final: dict[str, str] = {}

    def _public_speaker(self, value: Any) -> str:
        speaker_id = _speaker(value)
        return self.final_to_public.get(speaker_id, speaker_id)

    def _project_public_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        for key in ("assigned_speaker", "speaker_id", "replaces_speaker_id"):
            if result.get(key):
                result[key] = self._public_speaker(result[key])
        return result

    def _apply_identity_alias(self, item: dict[str, Any]) -> None:
        generation = int(item.get("alias_generation") or 0)
        final_id = _speaker(item.get("final_internal_speaker_id"))
        public_id = _speaker(item.get("surviving_public_speaker_id"))
        if generation <= self.alias_generation or not final_id or not public_id:
            return
        if item.get("retired"):
            if (
                self.final_to_public.get(final_id) != public_id
                or self.public_to_final.get(public_id) != final_id
            ):
                return
            self.final_to_public.pop(final_id, None)
            self.public_to_final.pop(public_id, None)
            self.alias_generation = generation
            self.timeline = [
                evidence for evidence in self.timeline
                if evidence.get("speaker") != public_id
            ]
            for row in self.rows:
                if row.raw_speaker == public_id:
                    row.raw_speaker = ""
                if row.speaker == public_id:
                    row.speaker = ""
            if self.fallback_speaker == public_id:
                self.fallback_speaker = ""
                self.fallback_until = 0.0
            if self.transcript_speaker == public_id:
                self.transcript_speaker = ""
            if self.current_speaker == public_id:
                self.current_speaker = ""
            self._refresh_rows()
            return
        if (
            final_id == public_id
            or final_id in self.final_to_public
            or public_id in self.public_to_final
            or final_id in self.public_to_final
            or public_id in self.final_to_public
        ):
            return
        self.final_to_public[final_id] = public_id
        self.public_to_final[public_id] = final_id
        self.alias_generation = generation
        for evidence in self.timeline:
            if evidence.get("speaker") == final_id:
                evidence["speaker"] = public_id
        for row in self.rows:
            if row.raw_speaker == final_id:
                row.raw_speaker = public_id
            if row.speaker == final_id:
                row.speaker = public_id
        if self.fallback_speaker == final_id:
            self.fallback_speaker = public_id
        if self.transcript_speaker == final_id:
            self.transcript_speaker = public_id
        if self.current_speaker == final_id:
            self.current_speaker = public_id
        self._refresh_rows()

    def _prune_timeline(self, minimum_end: float) -> None:
        cutoff = max(0.0, minimum_end)
        self.timeline = [
            item for item in self.timeline if _finite(item.get("end")) >= cutoff
        ]

    def _remember_evidence(self, speaker_id: str, item: dict[str, Any]) -> None:
        speaker_id = _speaker(speaker_id)
        if not speaker_id:
            return
        end = _finite(item.get("end"), self.playback_time)
        length = max(0.0, _finite(item.get("audio_length_seconds"), 0.0))
        start = _finite(item.get("start"), max(0.0, end - length))
        if end <= start:
            return
        self.timeline.append({"speaker": speaker_id, "start": start, "end": end})
        self._prune_timeline(end - 90.0)

    @staticmethod
    def _scored_end(start: float, end: float) -> float:
        duration = max(0.0, end - start)
        if duration <= 3.0:
            return end
        tail = min(3.0, max(2.0, duration * 0.25))
        return max(start + 0.1, end - tail)

    def _row_has_evidence(self, start: float, end: float) -> bool:
        if end <= start:
            return False
        for item in self.timeline:
            overlap = min(end, item["end"]) - max(start, item["start"])
            if _speaker(item["speaker"]) and overlap > 0.0:
                return True
        return False

    def _dominant_speaker(self, start: float, end: float) -> str:
        if end <= start:
            return ""
        scored_end = self._scored_end(start, end)
        scored_seconds = max(0.0, scored_end - start)
        if scored_seconds <= 0.0:
            return ""
        weights: dict[str, float] = {}
        for item in self.timeline:
            speaker_id = _speaker(item["speaker"])
            if not speaker_id:
                continue
            seconds = max(
                0.0,
                min(scored_end, item["end"]) - max(start, item["start"]),
            )
            if seconds > 0.0:
                weights[speaker_id] = weights.get(speaker_id, 0.0) + seconds
        if not weights:
            return ""
        speaker_id, seconds = max(weights.items(), key=lambda pair: pair[1])
        required = max(0.3, scored_seconds * 0.5)
        return speaker_id if seconds >= required else ""

    def _display_speaker(
        self,
        raw_speaker: str,
        start: float,
        end: float,
        previous_speaker: str,
    ) -> str:
        dominant = self._dominant_speaker(start, end)
        if dominant:
            return dominant
        previous = _speaker(previous_speaker)
        if previous:
            return previous
        if self._row_has_evidence(start, end):
            return ""
        raw = _speaker(raw_speaker)
        if not raw or end - start > 3.0:
            return ""
        return raw

    def _reconcile(self) -> None:
        fallback = (
            self.fallback_speaker
            if self.fallback_speaker and self.fallback_until > self.now
            else ""
        )
        if not fallback and self.fallback_speaker:
            self.fallback_speaker = ""
            self.fallback_until = 0.0
            self.fallback_clear_at = None
        self.current_speaker = fallback or (
            self.transcript_speaker if self.highlight_transcript else ""
        )

    def _refresh_rows(self) -> None:
        for row in self.rows:
            if not row.settling:
                row.speaker = self._display_speaker(
                    row.raw_speaker,
                    row.start,
                    row.end,
                    row.speaker,
                )
        active = [row for row in self.rows if not row.settling]
        active.sort(key=lambda row: (row.start, row.end, row.index))
        self.transcript_speaker = active[-1].speaker if active else ""
        self._reconcile()

    def advance(self, now: float, playback_time: float | None = None) -> None:
        self.now = max(self.now, float(now))
        if playback_time is not None:
            self.playback_time = max(0.0, float(playback_time))
        if (
            self.fallback_clear_at is not None
            and self.now >= self.fallback_clear_at
        ):
            self.fallback_speaker = ""
            self.fallback_until = 0.0
            self.fallback_clear_at = None
        if self.rows:
            self.rows = [
                row
                for row in self.rows
                if row.remove_at is None or row.remove_at > self.now
            ]
        self._refresh_rows()

    @staticmethod
    def _time_adoption_score(row: _RealtimeRow, start: float, end: float) -> float:
        if row.end <= row.start or end <= start:
            return 0.0
        overlap = max(0.0, min(row.end, end) - max(row.start, start))
        return overlap / max(0.1, min(row.end - row.start, end - start))

    def _adoptable_row(
        self,
        item: dict[str, Any],
        *,
        settling_only: bool = False,
    ) -> _RealtimeRow | None:
        start = _finite(item.get("start"), float("nan"))
        end = _finite(item.get("end"), float("nan"))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            return None
        best: tuple[float, _RealtimeRow] | None = None
        for row in self.rows:
            if row.remove_at is not None or (settling_only and not row.settling):
                continue
            time_score = self._time_adoption_score(row, start, end)
            if time_score <= 0.0:
                continue
            text_score = _text_adoption_score(row.text, item.get("text"))
            if time_score < 0.34 and text_score < 0.5:
                continue
            score = time_score * 0.72 + text_score * 0.28 + (0.08 if row.settling else 0.0)
            if best is None or score > best[0]:
                best = (score, row)
        return best[1] if best else None

    def _render_realtime(self, item: dict[str, Any]) -> None:
        generation = int(item.get("realtime_generation") or 0)
        if generation < self.current_generation:
            return
        self.current_generation = max(self.current_generation, generation)
        index = str(item.get("index") or "")
        row = next((candidate for candidate in self.rows if candidate.index == index), None)
        if row is None:
            row = self._adoptable_row(item, settling_only=True)
        start = _finite(item.get("start"), 0.0)
        end = _finite(item.get("end"), start)
        raw = _speaker(item.get("assigned_speaker"))
        previous = row.speaker if row else ""
        if row is None:
            row = _RealtimeRow(index=index, start=start, end=end, text="", raw_speaker=raw)
            self.rows.append(row)
        row.index = index
        row.start = start
        row.end = end
        row.text = str(item.get("text") or "")
        row.raw_speaker = raw
        row.settling = False
        row.clear_generation = ""
        row.remove_at = None
        row.speaker = self._display_speaker(raw, start, end, previous)
        self._refresh_rows()

    def _render_final(self, item: dict[str, Any]) -> None:
        index = str(item.get("index") or "")
        row = next((candidate for candidate in self.rows if candidate.index == index), None)
        if row is None:
            row = self._adoptable_row(item)
        if row is not None:
            self.rows.remove(row)
        start = _finite(item.get("start"), float("nan"))
        end = _finite(item.get("end"), float("nan"))
        if math.isfinite(start) and math.isfinite(end) and end > start:
            for candidate in self.rows:
                if not candidate.settling:
                    continue
                time_score = self._time_adoption_score(candidate, start, end)
                text_score = _text_adoption_score(candidate.text, item.get("text"))
                if time_score >= 0.34 and text_score >= 0.5:
                    candidate.remove_at = self.now + ROW_FADE_REMOVAL_SECONDS
        self._refresh_rows()

    def apply(self, event: str, item: dict[str, Any], now: float) -> None:
        self.advance(now)
        if event == "live_speaker_identity_alias":
            self._apply_identity_alias(item)
            return
        item = self._project_public_payload(item)
        if event == "live_speaker":
            speaker_id = _speaker(item.get("assigned_speaker") or item.get("speaker_id"))
            if not speaker_id:
                return
            if item.get("only_if_no_live_speaker") and self.current_speaker:
                return
            self._remember_evidence(speaker_id, item)
            hold = max(0.0, _finite(item.get("hold_seconds"), 2.0))
            self.fallback_speaker = speaker_id
            self.fallback_until = self.now + hold
            self.fallback_clear_at = None
            self._refresh_rows()
        elif event == "live_speaker_clear":
            speaker_id = _speaker(item.get("assigned_speaker") or item.get("speaker_id"))
            if speaker_id and self.fallback_speaker and speaker_id != self.fallback_speaker:
                return
            if (
                self.fallback_speaker
                and item.get("reason") == "unknown"
                and self.unknown_clear_debounce_seconds > 0.0
            ):
                self.fallback_until = max(
                    self.fallback_until,
                    self.now + self.unknown_clear_debounce_seconds,
                )
                self.fallback_clear_at = self.now + self.unknown_clear_debounce_seconds
            else:
                self.fallback_speaker = ""
                self.fallback_until = 0.0
                self.fallback_clear_at = None
            self._refresh_rows()
        elif event == "realtime_clear":
            generation = int(item.get("generation") or 0)
            self.current_generation = max(self.current_generation, generation)
            generation_key = str(generation or "")
            for row in self.rows:
                row.settling = True
                row.clear_generation = generation_key
                row.remove_at = self.now + REALTIME_SETTLE_REMOVAL_SECONDS + ROW_FADE_REMOVAL_SECONDS
            self._refresh_rows()
        elif event == "realtime":
            self._render_realtime(item)
        elif event == "sentence":
            self._render_final(item)

    def sample(self, recorded: dict[str, Any], now: float) -> dict[str, Any]:
        self.advance(now, _finite(recorded.get("playback_time"), self.playback_time))
        speaker = self.current_speaker
        result = dict(recorded)
        result["fallback_live_speaker_id"] = (
            self.fallback_speaker if self.fallback_until > self.now else ""
        )
        result["transcript_live_speaker_id"] = self.transcript_speaker
        result["transcript_live_override_speaker_id"] = ""
        result["current_live_speaker_id"] = speaker
        result["visible_live_speaker_id"] = speaker
        result["dom_live_speaker_ids"] = [speaker] if speaker else []
        return result


def _started_epoch_seconds(manifest: dict[str, Any]) -> float:
    value = str(manifest.get("started_at") or "")
    if not value:
        raise ValueError("World Tape manifest has no started_at")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@lru_cache(maxsize=32)
def _cached_browser_tape_inputs(
    root_key: str,
) -> tuple[
    dict[str, Any],
    tuple[tuple[float, int, str, dict[str, Any]], ...],
    tuple[tuple[float, int, dict[str, Any]], ...],
]:
    """Read immutable browser-replay inputs once per World Tape and process."""

    root = Path(root_key)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    started_epoch = _started_epoch_seconds(manifest)
    actions: list[tuple[float, int, str, dict[str, Any]]] = []
    recorded_samples: list[tuple[float, int, dict[str, Any]]] = []
    for record in read_world_tape_events(root):
        wall = _finite(record.get("wall_us"), 0.0) / 1_000_000.0
        sequence = int(record.get("seq") or 0)
        stream = str(record.get("stream") or "")
        event = str(record.get("event") or "")
        payload = dict(record.get("payload") or {})
        if stream == "public" and event in {
            "live_speaker",
            "live_speaker_clear",
            "live_speaker_identity_alias",
            "realtime",
            "realtime_clear",
            "sentence",
        }:
            actions.append((wall, sequence, event, payload))
        elif stream == "browser" and event == "ui_sample_clock":
            for sample in payload.get("samples") or []:
                if not isinstance(sample, dict):
                    continue
                sample_wall = _finite(sample.get("wall_time"), started_epoch) - started_epoch
                sample_sequence = int(sample.get("sample_sequence") or 0)
                recorded_samples.append((sample_wall, sample_sequence, dict(sample)))
    actions.sort(key=lambda item: (item[0], item[1]))
    recorded_samples.sort(key=lambda item: (item[0], item[1]))
    return manifest, tuple(actions), tuple(recorded_samples)


def replay_browser_state(
    tape_dir: Path,
    *,
    replacement_live_actions: list[tuple[float, int, str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Replay public events at the exact recorded browser observation timestamps."""

    root = Path(tape_dir).resolve()
    manifest, cached_actions, recorded_samples = _cached_browser_tape_inputs(str(root))
    reducer = BrowserLiveSpeakerReducer(manifest.get("runtime_config") or {})
    actions = [
        item
        for item in cached_actions
        if replacement_live_actions is None
        or item[2] not in {
            "live_speaker",
            "live_speaker_clear",
            "live_speaker_identity_alias",
        }
    ]
    if replacement_live_actions is not None:
        actions.extend(replacement_live_actions)
    actions.sort(key=lambda item: (item[0], item[1]))
    predicted: list[dict[str, Any]] = []
    action_index = 0
    exact_current = 0
    exact_fallback = 0
    exact_transcript = 0
    active_agreement = 0
    for sample_wall, _sample_sequence, recorded in recorded_samples:
        while action_index < len(actions) and actions[action_index][0] <= sample_wall:
            wall, _sequence, event, payload = actions[action_index]
            reducer.apply(event, payload, wall)
            action_index += 1
        replayed = reducer.sample(recorded, sample_wall)
        predicted.append(replayed)
        current_match = _speaker(recorded.get("current_live_speaker_id")) == _speaker(
            replayed.get("current_live_speaker_id")
        )
        exact_current += int(current_match)
        exact_fallback += int(
            _speaker(recorded.get("fallback_live_speaker_id"))
            == _speaker(replayed.get("fallback_live_speaker_id"))
        )
        exact_transcript += int(
            _speaker(recorded.get("transcript_live_speaker_id"))
            == _speaker(replayed.get("transcript_live_speaker_id"))
        )
        active_agreement += int(
            bool(_speaker(recorded.get("current_live_speaker_id")))
            == bool(_speaker(replayed.get("current_live_speaker_id")))
        )
    count = len(recorded_samples)
    ratio = lambda value: value / count if count else 0.0
    return {
        "contract_id": BROWSER_PARITY_REPLAY_ID,
        "parity_rung": "browser_reducer",
        "tape_dir": str(root.resolve()),
        "sample_count": count,
        "public_action_count": len(actions),
        "current_speaker_exact_ratio": ratio(exact_current),
        "fallback_speaker_exact_ratio": ratio(exact_fallback),
        "transcript_speaker_exact_ratio": ratio(exact_transcript),
        "active_state_exact_ratio": ratio(active_agreement),
        "recorded_samples": [item[2] for item in recorded_samples],
        "replayed_samples": predicted,
    }
