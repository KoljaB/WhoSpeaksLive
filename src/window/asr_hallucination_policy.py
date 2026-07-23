"""Small, conservative text priors for known Whisper hallucinations.

The curated phrases come from controlled non-speech experiments and the
official Whisper community reports.  Most published hallucination strings are
deliberately *not* included: plausible speech must continue to be decided by
the acoustic evidence rather than by a large blacklist.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class AsrHallucinationPolicyMatch:
    rule_id: str
    risk_score: int
    action: str
    suspicion_threshold: float
    verification_evidence_threshold: float
    normalized_text: str


_HARD_EXACT_CREDITS: dict[str, str] = {
    "subtitles by the amara org community": "credit_amara_community",
    "transcription by castingwords": "credit_castingwords",
    "transcript emily beynon": "credit_emily_beynon",
    "closed captioning provided by the imperial news network": "credit_imperial_news_network",
    "subtitles by steamteamextra": "credit_steamteamextra",
    "transcription by eso translation by": "credit_eso",
    "transcribed by eso translated by": "credit_eso_variant",
    "closed captioning provided by muhsen": "credit_muhsen",
    "captions by nicosubs": "credit_nicosubs",
    "captions by gettranscribed com": "credit_gettranscribed",
    "captioned by cotter captioning services": "credit_cotter",
    "closed captioning by kris brandhagen com": "credit_kris_brandhagen",
    "subtitles by subtitle workshop": "credit_subtitle_workshop",
    "bf watch tv 2021": "credit_bf_watch_tv",
    "tanya cushman reviewer": "credit_tanya_cushman",
    "tanya cushman reviewer s": "credit_tanya_cushman_variant",
}

_WATCHING_PHRASES: tuple[tuple[str, ...], ...] = (
    ("thanks", "for", "watching"),
    ("thank", "you", "for", "watching"),
)


def normalize_asr_hallucination_text(text: str) -> str:
    """Normalize text while preserving Unicode letters, digits, and boundaries."""

    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in value).split())


def _contains_token_sequence(tokens: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if len(pattern) > len(tokens):
        return False
    return any(tokens[index : index + len(pattern)] == pattern for index in range(len(tokens) - len(pattern) + 1))


def match_asr_hallucination_policy(
    text: str,
    *,
    base_suspicion_threshold: float,
    segment_start_seconds: float = 0.0,
    media_duration_seconds: float | None = None,
) -> AsrHallucinationPolicyMatch | None:
    """Return the most specific known-text policy for one ASR segment."""

    normalized = normalize_asr_hallucination_text(text)
    if not normalized:
        return None
    hard_rule = _HARD_EXACT_CREDITS.get(normalized)
    if hard_rule is not None:
        return AsrHallucinationPolicyMatch(
            rule_id=hard_rule,
            risk_score=100,
            action="require_evidence",
            suspicion_threshold=max(base_suspicion_threshold, 0.90),
            verification_evidence_threshold=max(base_suspicion_threshold, 0.70),
            normalized_text=normalized,
        )

    tokens = tuple(normalized.split())
    if "amara" in tokens and "org" in tokens:
        return AsrHallucinationPolicyMatch(
            rule_id="amara_org",
            # This is deliberately review-only unless the complete segment
            # matched one of the exact credit strings above.  A containing
            # segment may include genuine surrounding speech.
            risk_score=60,
            action="require_evidence",
            suspicion_threshold=max(base_suspicion_threshold, 0.80),
            verification_evidence_threshold=max(base_suspicion_threshold, 0.65),
            normalized_text=normalized,
        )

    watching_phrase = next(
        (phrase for phrase in _WATCHING_PHRASES if _contains_token_sequence(tokens, phrase)),
        None,
    )
    if watching_phrase is not None:
        suspicion_threshold = 0.72
        verification_threshold = 0.56
        if segment_start_seconds <= 3.0:
            suspicion_threshold = 0.82
            verification_threshold = 0.62
        elif (
            media_duration_seconds is not None
            and media_duration_seconds > 0.0
            and segment_start_seconds >= media_duration_seconds * 0.90
        ):
            suspicion_threshold = 0.60
            verification_threshold = 0.48
        return AsrHallucinationPolicyMatch(
            rule_id=(
                "thanks_for_watching"
                if tokens == watching_phrase
                else "thanks_for_watching_in_context"
            ),
            # Only a standalone stock phrase may be suppressed.  When the
            # phrase is part of a longer segment, keep the entire segment and
            # use the policy solely to request review/verification.
            risk_score=85 if tokens == watching_phrase else 60,
            action="require_evidence",
            suspicion_threshold=max(base_suspicion_threshold, suspicion_threshold),
            verification_evidence_threshold=max(base_suspicion_threshold, verification_threshold),
            normalized_text=normalized,
        )

    if "amara" in tokens:
        return AsrHallucinationPolicyMatch(
            rule_id="amara_name",
            risk_score=30,
            action="require_evidence",
            suspicion_threshold=max(base_suspicion_threshold, 0.50),
            verification_evidence_threshold=base_suspicion_threshold,
            normalized_text=normalized,
        )
    return None
