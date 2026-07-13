"""Pure transcript-evidence matching shared by reporting and Fact Lens.

Keeping this module free of server and runtime imports lets domain code validate
evidence without importing an executable sidecar (and all of its mutable state).
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re


def normalize_evidence_text(text: str) -> str:
    """Normalize text for conservative transcript-evidence comparisons."""

    return " ".join(re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE))


def evidence_matches_transcript(evidence: str, transcript: str) -> bool:
    """Return whether evidence is a near-verbatim span of the transcript."""

    evidence_norm = normalize_evidence_text(evidence)
    transcript_norm = normalize_evidence_text(transcript)
    if not evidence_norm or not transcript_norm:
        return False
    if evidence_norm in transcript_norm:
        return True

    evidence_tokens = evidence_norm.split()
    transcript_tokens = transcript_norm.split()
    if len(evidence_tokens) < 3 or not transcript_tokens:
        return False

    low = max(1, len(evidence_tokens) - 2)
    high = min(len(transcript_tokens), len(evidence_tokens) + 2)
    evidence_token_set = set(evidence_tokens)
    for size in range(low, high + 1):
        for index in range(0, len(transcript_tokens) - size + 1):
            window_tokens = transcript_tokens[index : index + size]
            window_text = " ".join(window_tokens)
            ratio = SequenceMatcher(None, evidence_norm, window_text).ratio()
            overlap = len(evidence_token_set.intersection(window_tokens)) / len(evidence_token_set)
            if ratio >= 0.82 or overlap >= 0.75:
                return True
    return False


__all__ = ["evidence_matches_transcript", "normalize_evidence_text"]
