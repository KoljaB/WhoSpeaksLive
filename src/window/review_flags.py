"""Review flagging for speaker-labeled transcript rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewThresholds:
    low_margin: float = 0.08
    weak_similarity: float = 0.55
    high_unknown_probability: float = 0.45
    short_audio_seconds: float = 0.8
    duplicate_profile_similarity: float = 0.86


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _speaker(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() == "UNKNOWN" else text


def _duration_seconds(row: dict[str, Any]) -> float | None:
    duration = _optional_float(row.get("duration_seconds") or row.get("audio_length_seconds"))
    if duration is not None:
        return max(0.0, duration)
    start = _optional_float(row.get("start"))
    end = _optional_float(row.get("end"))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def _duplicate_pairs(
    speaker_profiles: list[dict[str, Any]] | None,
    threshold: float,
) -> list[dict[str, Any]]:
    if not speaker_profiles:
        return []
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(speaker_profiles):
        left_label = _speaker(left.get("label") or left.get("id"))
        if not left_label:
            continue
        left_similarities = left.get("similarities")
        if not isinstance(left_similarities, dict):
            continue
        for right in speaker_profiles[left_index + 1 :]:
            right_label = _speaker(right.get("label") or right.get("id"))
            if not right_label:
                continue
            similarity = _optional_float(left_similarities.get(right_label))
            if similarity is None or similarity < threshold:
                continue
            pairs.append({
                "left": left_label,
                "right": right_label,
                "similarity": round(float(similarity), 4),
            })
    return pairs


def annotate_review(
    row: dict[str, Any],
    *,
    speaker_profiles: list[dict[str, Any]] | None = None,
    thresholds: ReviewThresholds | None = None,
) -> dict[str, Any]:
    """Return a stable review summary for one transcript row.

    The score is a suspicion score, not confidence. Higher values mean the row
    should be reviewed earlier.
    """

    limits = thresholds or ReviewThresholds()
    reasons: list[str] = []
    details: dict[str, Any] = {}
    score = 0.0

    asr_review = row.get("asr_review")
    if isinstance(asr_review, dict) and asr_review.get("needs_review"):
        for reason in asr_review.get("reasons") or []:
            normalized = " ".join(str(reason or "").split())
            if normalized:
                reasons.append(normalized)
        asr_details = asr_review.get("details")
        if isinstance(asr_details, dict) and asr_details:
            details["asr"] = dict(asr_details)
        asr_score = _optional_float(asr_review.get("score"))
        score = max(score, asr_score if asr_score is not None else 0.55)

    correction = row.get("correction")
    if isinstance(correction, dict) and correction.get("status") in {"user_corrected", "user_confirmed"}:
        # A transcript correction confirms only the speaker label.  It must
        # not silently resolve an independent ASR/text warning.
        details["speaker_review_resolved_by_user"] = True
        unique_reasons = list(dict.fromkeys(reasons))
        return {
            "needs_review": bool(unique_reasons),
            "score": round(min(1.0, float(score)), 4),
            "reasons": unique_reasons,
            "details": details,
        }

    speaker = _speaker(row.get("assigned_speaker") or row.get("speaker_id") or row.get("speaker"))
    margin = _optional_float(row.get("margin"))
    if speaker and margin is not None and margin < limits.low_margin:
        reasons.append("low margin")
        details["margin"] = round(float(margin), 4)
        score += min(0.35, (limits.low_margin - margin) / max(limits.low_margin, 0.001) * 0.35)

    similarity = _optional_float(row.get("top_similarity"))
    if speaker and similarity is not None and similarity < limits.weak_similarity:
        reasons.append("weak speaker evidence")
        details["top_similarity"] = round(float(similarity), 4)
        score += min(
            0.25,
            (limits.weak_similarity - similarity) / max(abs(limits.weak_similarity), 0.001) * 0.25,
        )

    unknown_probability = _optional_float(row.get("unknown_probability"))
    if speaker and unknown_probability is not None and unknown_probability > limits.high_unknown_probability:
        reasons.append("possibly unknown")
        details["unknown_probability"] = round(float(unknown_probability), 4)
        score += min(0.25, (unknown_probability - limits.high_unknown_probability) * 0.35)

    duration = _duration_seconds(row)
    if duration is not None and duration < limits.short_audio_seconds:
        reasons.append("short audio")
        details["duration_seconds"] = round(float(duration), 4)
        score += min(
            0.20,
            (limits.short_audio_seconds - duration) / max(limits.short_audio_seconds, 0.001) * 0.20,
        )

    live_speaker = _speaker(row.get("live_speaker") or row.get("live_speaker_id"))
    if speaker and live_speaker and live_speaker != speaker:
        reasons.append("conflicting live/final evidence")
        details["live_speaker"] = live_speaker
        score += 0.30

    duplicate_pairs = _duplicate_pairs(speaker_profiles, limits.duplicate_profile_similarity)
    if duplicate_pairs and speaker and any(speaker in {pair["left"], pair["right"]} for pair in duplicate_pairs):
        reasons.append("possible duplicate profile")
        details["duplicate_profiles"] = duplicate_pairs
        score += 0.15

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "needs_review": bool(unique_reasons),
        "score": round(min(1.0, float(score)), 4),
        "reasons": unique_reasons,
        "details": details,
    }
