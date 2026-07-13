"""Fact Lens domain package."""

from window.fact_lens.domain import (
    ClaimCard,
    ExtractedClaim,
    ExtractionJob,
    ExtractionResult,
    FactLensStore,
    SentenceRevisionToken,
    SidecarState,
    SnapshotPublisher,
    TranscriptSentence,
    claim_card_id,
    coalesce_sentences,
    placeholder_card_id,
    wall_now,
)

__all__ = [name for name in globals() if not name.startswith("_")]
__all__.append("FactLensRuntime")


def __getattr__(name: str):
    if name == "FactLensRuntime":
        from window.fact_lens.runtime import FactLensRuntime
        return FactLensRuntime
    raise AttributeError(name)
