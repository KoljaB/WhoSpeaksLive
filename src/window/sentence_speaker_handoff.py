"""Conservative, compute-light detection of a speaker handoff within a sentence.

The live pass nominates a sentence from cached short-window evidence.  A later,
focused verifier embeds one speech-bearing context block on each side of the
nearest ASR word gap and requires strong A-on-the-left/B-on-the-right evidence
against every established profile.  The older word-margin selector remains a
pure diagnostic primitive, but the production path does not require every
short function word to carry a usable voice signature.

This module deliberately contains no audio or model-provider code.  Its inputs
are timestamps, speaker labels, and already-computed embeddings or margins, so
all decisions are deterministic and inexpensive to replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from window.window_domain import SentencePart


_DEFAULT_UNKNOWN_LABELS = ("", "unknown", "none", "null", "?")
_NEGATIVE_INFINITY = float("-inf")


def normalize_embedding(value: Any) -> np.ndarray:
    """Return a finite, immutable, unit-length float32 embedding."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Speaker embeddings must contain finite values.")
    norm = float(np.linalg.norm(vector))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("Speaker embeddings must have non-zero length.")
    normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


def cosine_margin(
    embedding: Any,
    speaker_a_embedding: Any,
    speaker_b_embedding: Any,
) -> float:
    """Return cosine(B) - cosine(A); negative favours A and positive favours B."""

    vector = normalize_embedding(embedding)
    speaker_a = normalize_embedding(speaker_a_embedding)
    speaker_b = normalize_embedding(speaker_b_embedding)
    if vector.shape != speaker_a.shape or vector.shape != speaker_b.shape:
        raise ValueError("Evidence and profile embeddings must have the same dimensions.")
    return float(np.dot(vector, speaker_b) - np.dot(vector, speaker_a))


@dataclass(frozen=True)
class LiveEmbeddingEvidence:
    """One cached trailing live probe.

    ``window_start`` and ``window_end`` describe the audio used for the probe.
    The live code reports a trailing window at ``window_end``; its acoustic
    centre is therefore half a window earlier.
    """

    window_start: float
    window_end: float
    short_embedding: np.ndarray = field(repr=False, compare=False)
    visible_speaker: str | None = None
    similarities: Mapping[str, float] = field(default_factory=dict)
    profile_generations: Mapping[str, int] = field(default_factory=dict)
    provider: str = ""
    voiced_seconds: float | None = None

    def __post_init__(self) -> None:
        start = float(self.window_start)
        end = float(self.window_end)
        if not isfinite(start) or not isfinite(end) or end <= start:
            raise ValueError("A live evidence window must have finite, increasing bounds.")
        object.__setattr__(self, "window_start", start)
        object.__setattr__(self, "window_end", end)
        object.__setattr__(self, "short_embedding", normalize_embedding(self.short_embedding))

        speaker = str(self.visible_speaker).strip() if self.visible_speaker is not None else None
        object.__setattr__(self, "visible_speaker", speaker or None)
        object.__setattr__(self, "provider", str(self.provider or "").strip())

        similarities: dict[str, float] = {}
        for label, value in dict(self.similarities or {}).items():
            score = float(value)
            if isfinite(score):
                similarities[str(label)] = score
        generations: dict[str, int] = {}
        for label, value in dict(self.profile_generations or {}).items():
            generations[str(label)] = int(value)
        object.__setattr__(self, "similarities", MappingProxyType(similarities))
        object.__setattr__(self, "profile_generations", MappingProxyType(generations))

        if self.voiced_seconds is not None:
            voiced = float(self.voiced_seconds)
            if not isfinite(voiced) or voiced < 0.0:
                raise ValueError("voiced_seconds must be finite and non-negative.")
            # A few VAD implementations can exceed the nominal window by one
            # frame; clipping keeps the cached support measure well-defined.
            object.__setattr__(self, "voiced_seconds", min(voiced, end - start))

    @property
    def duration(self) -> float:
        return self.window_end - self.window_start

    @property
    def acoustic_center(self) -> float:
        return self.window_end - 0.5 * self.duration


@dataclass(frozen=True)
class HandoffConfig:
    """Thresholds shared by the cheap nomination and focused verifier."""

    short_window_seconds: float = 0.70
    short_window_tolerance_seconds: float = 0.22
    min_probes_per_side: int = 2
    min_live_voiced_seconds_per_side: float = 0.0
    require_same_provider: bool = True
    unknown_speaker_labels: tuple[str, ...] = _DEFAULT_UNKNOWN_LABELS
    min_voiced_unknown_seconds: float = 0.25
    max_unknown_transition_probes: int = 3
    max_transition_evidence_gap_seconds: float = 1.75
    max_adjacent_evidence_gap_seconds: float = 1.25
    max_word_endpoint_evidence_gap_seconds: float = 1.25
    min_unknown_side_similarity: float = 0.45
    min_unknown_side_margin: float = 0.08
    min_bracketed_unknown_similarity: float = 0.28
    min_bracketed_unknown_margin: float = 0.20
    max_bracketed_unknown_probes: int = 3
    min_edge_unknown_similarity: float = 0.35
    min_edge_unknown_margin: float = 0.25
    max_edge_unknown_probes: int = 5
    min_transition_unknown_similarity: float = 0.35
    min_transition_unknown_third_margin: float = 0.08

    min_words_per_segment: int = 2
    min_word_support_per_segment: float = 0.0
    min_gain_over_no_split: float = 0.25
    min_gain_over_reverse: float = 0.25
    min_gain_over_runner_up_cut: float = 0.08
    cosine_zero_tolerance: float = 1e-6
    min_context_similarity: float = 0.35
    min_context_pair_margin: float = 0.15
    min_context_runner_up_margin: float = 0.08
    min_context_separation: float = 0.50

    def __post_init__(self) -> None:
        if self.short_window_seconds <= 0.0:
            raise ValueError("short_window_seconds must be positive.")
        if self.short_window_tolerance_seconds < 0.0:
            raise ValueError("short_window_tolerance_seconds cannot be negative.")
        if self.min_probes_per_side < 1:
            raise ValueError("min_probes_per_side must be at least one.")
        if self.min_live_voiced_seconds_per_side < 0.0:
            raise ValueError("min_live_voiced_seconds_per_side cannot be negative.")
        if self.min_voiced_unknown_seconds < 0.0:
            raise ValueError("min_voiced_unknown_seconds cannot be negative.")
        if self.max_unknown_transition_probes < 0:
            raise ValueError("max_unknown_transition_probes cannot be negative.")
        if self.max_transition_evidence_gap_seconds <= 0.0:
            raise ValueError("max_transition_evidence_gap_seconds must be positive.")
        if self.max_adjacent_evidence_gap_seconds <= 0.0:
            raise ValueError("max_adjacent_evidence_gap_seconds must be positive.")
        if self.max_word_endpoint_evidence_gap_seconds <= 0.0:
            raise ValueError("max_word_endpoint_evidence_gap_seconds must be positive.")
        if not -1.0 <= float(self.min_unknown_side_similarity) <= 1.0:
            raise ValueError("min_unknown_side_similarity must be between -1 and 1.")
        if self.min_unknown_side_margin < 0.0:
            raise ValueError("min_unknown_side_margin cannot be negative.")
        if not -1.0 <= float(self.min_bracketed_unknown_similarity) <= 1.0:
            raise ValueError("min_bracketed_unknown_similarity must be between -1 and 1.")
        if self.min_bracketed_unknown_margin < 0.0:
            raise ValueError("min_bracketed_unknown_margin cannot be negative.")
        if self.max_bracketed_unknown_probes < 1:
            raise ValueError("max_bracketed_unknown_probes must be at least one.")
        if not -1.0 <= float(self.min_edge_unknown_similarity) <= 1.0:
            raise ValueError("min_edge_unknown_similarity must be between -1 and 1.")
        if self.min_edge_unknown_margin < 0.0:
            raise ValueError("min_edge_unknown_margin cannot be negative.")
        if self.max_edge_unknown_probes < 1:
            raise ValueError("max_edge_unknown_probes must be at least one.")
        if not -1.0 <= float(self.min_transition_unknown_similarity) <= 1.0:
            raise ValueError("min_transition_unknown_similarity must be between -1 and 1.")
        if self.min_transition_unknown_third_margin < 0.0:
            raise ValueError("min_transition_unknown_third_margin cannot be negative.")
        if self.min_words_per_segment < 1:
            raise ValueError("min_words_per_segment must be at least one.")
        if self.min_word_support_per_segment < 0.0:
            raise ValueError("min_word_support_per_segment cannot be negative.")
        for name in (
            "min_gain_over_no_split",
            "min_gain_over_reverse",
            "min_gain_over_runner_up_cut",
            "cosine_zero_tolerance",
            "min_context_pair_margin",
            "min_context_runner_up_margin",
            "min_context_separation",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if not -1.0 <= float(self.min_context_similarity) <= 1.0:
            raise ValueError("min_context_similarity must be between -1 and 1.")


@dataclass(frozen=True)
class CoarseBoundaryEstimate:
    """A raw cosine crossing corrected from trailing-window time to audio time."""

    boundary_time: float
    trailing_crossing_time: float
    trailing_window_correction: float
    left_margin: float
    right_margin: float
    left_window_end: float
    right_window_end: float
    margins: tuple[float, ...]


@dataclass(frozen=True)
class HandoffNomination:
    """A stable visible A-run followed by one stable visible B-run."""

    speaker_a: str
    speaker_b: str
    sentence_start: float
    sentence_end: float
    evidence: tuple[LiveEmbeddingEvidence, ...]
    left_probe_count: int
    right_probe_count: int
    left_voiced_seconds: float
    right_voiced_seconds: float
    last_a_evidence_index: int
    first_b_evidence_index: int
    coarse_boundary: CoarseBoundaryEstimate | None = None
    suggested_word_cut_index: int | None = None
    suggested_word_boundary_time: float | None = None

    @property
    def visible_transition_time(self) -> float:
        left = self.evidence[self.last_a_evidence_index].acoustic_center
        right = self.evidence[self.first_b_evidence_index].acoustic_center
        return 0.5 * (left + right)

    @property
    def coarse_boundary_time(self) -> float | None:
        return self.coarse_boundary.boundary_time if self.coarse_boundary is not None else None


@dataclass(frozen=True)
class WordSpeakerMargin:
    """One word's B-minus-A score and optional reliability support."""

    word_index: int
    margin: float
    support: float = 1.0

    def __post_init__(self) -> None:
        if self.word_index < 0:
            raise ValueError("word_index cannot be negative.")
        if not isfinite(float(self.margin)):
            raise ValueError("A word margin must be finite.")
        if not isfinite(float(self.support)) or float(self.support) < 0.0:
            raise ValueError("Word support must be finite and non-negative.")
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(self, "support", float(self.support))


@dataclass(frozen=True)
class WordHandoffSelection:
    """Diagnostic result of comparing all constant and single-cut models."""

    accepted: bool
    cut_index: int | None
    candidate_cut_index: int | None
    reason: str
    a_only_score: float
    b_only_score: float
    no_split_score: float
    forward_score: float
    reverse_score: float
    runner_up_forward_score: float
    gain_over_no_split: float
    gain_over_reverse: float
    gain_over_runner_up_cut: float
    left_word_count: int
    right_word_count: int
    left_support: float
    right_support: float


@dataclass(frozen=True)
class ContextHandoffSelection:
    """Decision from one speech-bearing context window on each side of a gap."""

    accepted: bool
    reason: str
    left_expected_similarity: float
    left_other_similarity: float
    left_runner_up_similarity: float
    left_pair_margin: float
    left_runner_up_margin: float
    right_expected_similarity: float
    right_other_similarity: float
    right_runner_up_similarity: float
    right_pair_margin: float
    right_runner_up_margin: float
    separation: float


@dataclass(frozen=True)
class SentencePartSplit:
    """Two ordinary SentencePart values that retain one semantic sentence id."""

    left: SentencePart
    right: SentencePart
    semantic_sentence_id: str
    cut_index: int
    boundary_time: float
    speaker_a: str | None = None
    speaker_b: str | None = None

    @property
    def parts(self) -> tuple[SentencePart, SentencePart]:
        return self.left, self.right

    def __iter__(self) -> Iterator[SentencePart]:
        return iter(self.parts)


def _speaker_label(value: str | None, config: HandoffConfig) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    unknown = {item.strip().casefold() for item in config.unknown_speaker_labels}
    return None if label.casefold() in unknown else label


def _short_sentence_evidence(
    evidence: Sequence[LiveEmbeddingEvidence],
    sentence_start: float,
    sentence_end: float,
    config: HandoffConfig,
) -> tuple[LiveEmbeddingEvidence, ...]:
    selected = [
        item
        for item in evidence
        if sentence_start <= item.acoustic_center <= sentence_end
        and abs(item.duration - config.short_window_seconds)
        <= config.short_window_tolerance_seconds
    ]
    selected.sort(key=lambda item: (item.window_end, item.window_start))
    return tuple(selected)


def _anchor_embedding(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("embedding", "centroid", "vector"):
            if key in value:
                return value[key]
    return value


def _is_voiced_unknown(
    item: LiveEmbeddingEvidence,
    config: HandoffConfig,
) -> bool:
    """Treat missing VAD support conservatively and ignore only tiny residue."""

    if _speaker_label(item.visible_speaker, config) is not None:
        return False
    if item.voiced_seconds is None:
        return True
    return item.voiced_seconds >= config.min_voiced_unknown_seconds


def _contiguous_unknown_runs(
    evidence: Sequence[LiveEmbeddingEvidence],
    config: HandoffConfig,
) -> tuple[tuple[int, ...], ...]:
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, item in enumerate(evidence):
        if _speaker_label(item.visible_speaker, config) is None:
            current.append(index)
        elif current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def _unknown_run_matches_expected_speaker(
    run: Sequence[LiveEmbeddingEvidence],
    expected_speaker: str,
    profile_anchors: Mapping[str, Any] | None,
    config: HandoffConfig,
    *,
    bracketed_by_expected_speaker: bool,
    at_sentence_edge: bool,
) -> bool:
    """Resolve an unknown run from strong absolute and relative geometry.

    A short run bracketed by the same known speaker may use a lower absolute
    floor because subsecond embeddings are measurably weaker.  Its required
    runner-up margin is correspondingly much stronger.
    """

    voiced = [item.short_embedding for item in run if _is_voiced_unknown(item, config)]
    if not voiced:
        return True
    if profile_anchors is None or expected_speaker not in profile_anchors:
        return False
    try:
        aggregate = normalize_embedding(np.mean(np.stack(voiced), axis=0))
        anchors = {
            str(label): normalize_embedding(_anchor_embedding(value))
            for label, value in profile_anchors.items()
        }
    except (TypeError, ValueError):
        return False
    expected = anchors.get(expected_speaker)
    competitors = [
        anchor
        for label, anchor in anchors.items()
        if label != expected_speaker
    ]
    if expected is None or not competitors:
        return False
    if any(anchor.shape != aggregate.shape for anchor in (expected, *competitors)):
        return False
    expected_score = float(np.dot(aggregate, expected))
    runner_up = max(float(np.dot(aggregate, anchor)) for anchor in competitors)
    margin = expected_score - runner_up
    if (
        expected_score >= config.min_unknown_side_similarity
        and margin >= config.min_unknown_side_margin
    ):
        return True
    if (
        bracketed_by_expected_speaker
        and len(voiced) <= config.max_bracketed_unknown_probes
        and expected_score >= config.min_bracketed_unknown_similarity
        and margin >= config.min_bracketed_unknown_margin
    ):
        return True
    return (
        at_sentence_edge
        and len(voiced) <= config.max_edge_unknown_probes
        and expected_score >= config.min_edge_unknown_similarity
        and margin >= config.min_edge_unknown_margin
    )


def _transition_unknown_matches_speaker_union(
    run: Sequence[LiveEmbeddingEvidence],
    speaker_a: str,
    speaker_b: str,
    profile_anchors: Mapping[str, Any] | None,
    config: HandoffConfig,
) -> bool:
    """Require transition uncertainty to belong to A/B, not a third profile."""

    voiced = [item.short_embedding for item in run if _is_voiced_unknown(item, config)]
    if not voiced:
        return True
    if (
        profile_anchors is None
        or speaker_a not in profile_anchors
        or speaker_b not in profile_anchors
    ):
        return False
    try:
        aggregate = normalize_embedding(np.mean(np.stack(voiced), axis=0))
        anchors = {
            str(label): normalize_embedding(_anchor_embedding(value))
            for label, value in profile_anchors.items()
        }
    except (TypeError, ValueError):
        return False
    if any(anchor.shape != aggregate.shape for anchor in anchors.values()):
        return False
    scores = {
        label: float(np.dot(aggregate, anchor))
        for label, anchor in anchors.items()
    }
    pair_score = max(scores[speaker_a], scores[speaker_b])
    if pair_score < config.min_transition_unknown_similarity:
        return False
    third_scores = [
        score
        for label, score in scores.items()
        if label not in {speaker_a, speaker_b}
    ]
    return (
        not third_scores
        or pair_score - max(third_scores)
        >= config.min_transition_unknown_third_margin
    )


def estimate_coarse_boundary(
    evidence: Sequence[LiveEmbeddingEvidence],
    speaker_a_embedding: Any,
    speaker_b_embedding: Any,
    *,
    zero_tolerance: float = 1e-6,
) -> CoarseBoundaryEstimate | None:
    """Find one clean negative-to-positive raw cosine crossing.

    The crossing is first interpolated on the live probe timestamps
    (``window_end``), then corrected backwards by half the interpolated
    trailing-window duration.  A reverse crossing or sign flicker makes the
    estimate ambiguous and returns ``None``.
    """

    ordered = tuple(sorted(evidence, key=lambda item: (item.window_end, item.window_start)))
    if len(ordered) < 2:
        return None
    speaker_a = normalize_embedding(speaker_a_embedding)
    speaker_b = normalize_embedding(speaker_b_embedding)
    margins: list[float] = []
    for item in ordered:
        vector = item.short_embedding
        if vector.shape != speaker_a.shape or vector.shape != speaker_b.shape:
            raise ValueError("Evidence and profile embeddings must have the same dimensions.")
        margins.append(float(np.dot(vector, speaker_b) - np.dot(vector, speaker_a)))

    tolerance = max(0.0, float(zero_tolerance))
    signs = [-1 if value < -tolerance else 1 if value > tolerance else 0 for value in margins]
    non_zero_signs = [value for value in signs if value]
    if not non_zero_signs or non_zero_signs[0] != -1 or non_zero_signs[-1] != 1:
        return None
    compressed = [non_zero_signs[0]]
    for value in non_zero_signs[1:]:
        if value != compressed[-1]:
            compressed.append(value)
    if compressed != [-1, 1]:
        return None

    last_negative = max(index for index, sign in enumerate(signs) if sign == -1)
    first_positive = min(
        index for index, sign in enumerate(signs) if sign == 1 and index > last_negative
    )
    zero_indexes = [
        index for index in range(last_negative + 1, first_positive) if signs[index] == 0
    ]
    if zero_indexes:
        # A flat zero run is rare; its middle is the least biased crossing.
        index = zero_indexes[len(zero_indexes) // 2]
        item = ordered[index]
        crossing_time = item.window_end
        correction = 0.5 * item.duration
        left_margin = margins[index]
        right_margin = margins[index]
        left_end = item.window_end
        right_end = item.window_end
    else:
        left = ordered[last_negative]
        right = ordered[first_positive]
        left_margin = margins[last_negative]
        right_margin = margins[first_positive]
        denominator = right_margin - left_margin
        if denominator <= 0.0:
            return None
        fraction = min(1.0, max(0.0, -left_margin / denominator))
        crossing_time = left.window_end + fraction * (right.window_end - left.window_end)
        left_half = 0.5 * left.duration
        right_half = 0.5 * right.duration
        correction = left_half + fraction * (right_half - left_half)
        left_end = left.window_end
        right_end = right.window_end

    return CoarseBoundaryEstimate(
        boundary_time=float(crossing_time - correction),
        trailing_crossing_time=float(crossing_time),
        trailing_window_correction=float(correction),
        left_margin=float(left_margin),
        right_margin=float(right_margin),
        left_window_end=float(left_end),
        right_window_end=float(right_end),
        margins=tuple(margins),
    )


def _coerce_word_time(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
    else:
        try:
            start, end = value
        except (TypeError, ValueError) as exc:
            raise ValueError("Word times must be mappings or (start, end) pairs.") from exc
    start_value = float(start)
    end_value = float(end)
    if not isfinite(start_value) or not isfinite(end_value) or end_value < start_value:
        raise ValueError("Word times must have finite, non-decreasing bounds.")
    return start_value, end_value


def _suggest_word_cut(
    word_times: Sequence[Any],
    boundary_time: float,
    min_words_per_segment: int,
) -> tuple[int | None, float | None]:
    times = [_coerce_word_time(value) for value in word_times]
    minimum = max(1, int(min_words_per_segment))
    candidates: list[tuple[float, int, float]] = []
    for cut in range(minimum, len(times) - minimum + 1):
        left_end = times[cut - 1][1]
        right_start = times[cut][0]
        word_boundary = 0.5 * (left_end + right_start)
        candidates.append((abs(word_boundary - boundary_time), cut, word_boundary))
    if not candidates:
        return None, None
    _, cut, word_boundary = min(candidates, key=lambda item: (item[0], item[1]))
    return cut, float(word_boundary)


def nominate_stable_handoff(
    evidence: Sequence[LiveEmbeddingEvidence],
    sentence_start: float,
    sentence_end: float,
    word_times: Sequence[Any] = (),
    profile_anchors: Mapping[str, Any] | None = None,
    config: HandoffConfig | None = None,
) -> HandoffNomination | None:
    """Nominate exactly one stable visible A-to-B run inside a sentence.

    One short contiguous unknown band is allowed at the transition.  Voiced
    unknown runs elsewhere must be pairwise-resolvable to their surrounding
    side from mature profile anchors; otherwise the sentence is rejected.
    Missing temporal coverage across the transition, any known third speaker,
    B-to-A return, or additional A/B flicker also rejects the sentence.
    Profile anchors remain optional for a fully labelled clean transition.
    """

    active = config or HandoffConfig()
    start = float(sentence_start)
    end = float(sentence_end)
    if not isfinite(start) or not isfinite(end) or end <= start:
        raise ValueError("Sentence bounds must be finite and increasing.")
    selected = _short_sentence_evidence(evidence, start, end, active)
    if len(selected) < 2 * active.min_probes_per_side:
        return None
    if active.require_same_provider and len({item.provider for item in selected}) > 1:
        return None
    centers = tuple(item.acoustic_center for item in selected)
    if any(
        right - left > active.max_adjacent_evidence_gap_seconds
        for left, right in zip(centers, centers[1:])
    ):
        return None
    if word_times:
        times = tuple(_coerce_word_time(value) for value in word_times)
        first_word_start = times[0][0]
        last_word_end = times[-1][1]
        endpoint_limit = active.max_word_endpoint_evidence_gap_seconds
        if min(abs(center - first_word_start) for center in centers) > endpoint_limit:
            return None
        if min(abs(center - last_word_end) for center in centers) > endpoint_limit:
            return None

    known = [
        (index, label)
        for index, item in enumerate(selected)
        if (label := _speaker_label(item.visible_speaker, active)) is not None
    ]
    if len(known) < 2 * active.min_probes_per_side:
        return None

    compressed: list[str] = []
    for _, label in known:
        if not compressed or label != compressed[-1]:
            compressed.append(label)
    # Exactly [A, B] rejects a third speaker and every A->B->A/B flicker.
    if len(compressed) != 2 or compressed[0] == compressed[1]:
        return None
    speaker_a, speaker_b = compressed
    first_b_index = min(index for index, label in known if label == speaker_b)
    last_a_index = max(index for index, label in known if label == speaker_a)
    if last_a_index >= first_b_index:
        return None

    unknown_runs = _contiguous_unknown_runs(selected, active)
    transition_run = next(
        (
            run_indexes
            for run_indexes in unknown_runs
            if run_indexes[0] > last_a_index
            and run_indexes[-1] < first_b_index
        ),
        (),
    )
    transition_voiced_indexes = [
        index
        for index in transition_run
        if _is_voiced_unknown(selected[index], active)
    ]
    if transition_voiced_indexes and not _transition_unknown_matches_speaker_union(
        [selected[index] for index in transition_run],
        speaker_a,
        speaker_b,
        profile_anchors,
        active,
    ):
        return None
    transition_gap = (
        selected[first_b_index].acoustic_center
        - selected[last_a_index].acoustic_center
    )
    if (
        len(transition_voiced_indexes) > active.max_unknown_transition_probes
        or transition_gap > active.max_transition_evidence_gap_seconds
    ):
        return None

    for run_indexes in unknown_runs:
        voiced_indexes = [
            index
            for index in run_indexes
            if _is_voiced_unknown(selected[index], active)
        ]
        if not voiced_indexes:
            continue
        inside_transition = (
            run_indexes[0] > last_a_index
            and run_indexes[-1] < first_b_index
        )
        if inside_transition:
            if len(voiced_indexes) > active.max_unknown_transition_probes:
                return None
            continue
        expected_speaker = speaker_a if run_indexes[-1] < first_b_index else speaker_b
        previous_index = run_indexes[0] - 1
        next_index = run_indexes[-1] + 1
        bracketed_by_expected = (
            previous_index >= 0
            and next_index < len(selected)
            and _speaker_label(
                selected[previous_index].visible_speaker,
                active,
            ) == expected_speaker
            and _speaker_label(
                selected[next_index].visible_speaker,
                active,
            ) == expected_speaker
        )
        at_sentence_edge = (
            run_indexes[0] == 0
            or run_indexes[-1] == len(selected) - 1
        )
        if not _unknown_run_matches_expected_speaker(
            [selected[index] for index in run_indexes],
            expected_speaker,
            profile_anchors,
            active,
            bracketed_by_expected_speaker=bracketed_by_expected,
            at_sentence_edge=at_sentence_edge,
        ):
            return None

    left = [item for item in selected[:first_b_index] if _speaker_label(item.visible_speaker, active) == speaker_a]
    right = [item for item in selected[first_b_index:] if _speaker_label(item.visible_speaker, active) == speaker_b]
    if len(left) < active.min_probes_per_side or len(right) < active.min_probes_per_side:
        return None
    left_voiced = sum(float(item.voiced_seconds or 0.0) for item in left)
    right_voiced = sum(float(item.voiced_seconds or 0.0) for item in right)
    if (
        left_voiced < active.min_live_voiced_seconds_per_side
        or right_voiced < active.min_live_voiced_seconds_per_side
    ):
        return None

    coarse: CoarseBoundaryEstimate | None = None
    if profile_anchors is not None and speaker_a in profile_anchors and speaker_b in profile_anchors:
        try:
            # The visible assignments already prove one stable A->B run.
            # Estimate the acoustic crossing only between the last confidently
            # assigned A probe and the first confidently assigned B probe.
            # Scanning the entire sentence lets an earlier low-energy/silent
            # probe create a harmless raw-cosine sign flicker and suppress a
            # genuine later handoff.
            crossing_evidence = (
                selected[last_a_index],
                selected[first_b_index],
            )
            coarse = estimate_coarse_boundary(
                crossing_evidence,
                _anchor_embedding(profile_anchors[speaker_a]),
                _anchor_embedding(profile_anchors[speaker_b]),
                zero_tolerance=active.cosine_zero_tolerance,
            )
        except ValueError:
            # A stale provider/profile with different embedding dimensions must
            # not destroy a valid cheap nomination; it simply cannot refine it.
            coarse = None

    visible_time = 0.5 * (
        selected[last_a_index].acoustic_center + selected[first_b_index].acoustic_center
    )
    target_time = coarse.boundary_time if coarse is not None else visible_time
    suggested_cut, suggested_time = _suggest_word_cut(
        word_times,
        target_time,
        active.min_words_per_segment,
    ) if word_times else (None, None)

    return HandoffNomination(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sentence_start=start,
        sentence_end=end,
        evidence=selected,
        left_probe_count=len(left),
        right_probe_count=len(right),
        left_voiced_seconds=float(left_voiced),
        right_voiced_seconds=float(right_voiced),
        last_a_evidence_index=last_a_index,
        first_b_evidence_index=first_b_index,
        coarse_boundary=coarse,
        suggested_word_cut_index=suggested_cut,
        suggested_word_boundary_time=suggested_time,
    )


def _coerce_word_margins(values: Sequence[Any]) -> tuple[WordSpeakerMargin, ...]:
    coerced: list[WordSpeakerMargin] = []
    for position, value in enumerate(values):
        if isinstance(value, WordSpeakerMargin):
            item = value
        elif isinstance(value, Mapping):
            if "margin" in value:
                margin = value["margin"]
            elif "speaker_margin" in value:
                margin = value["speaker_margin"]
            elif "b_minus_a" in value:
                margin = value["b_minus_a"]
            else:
                raise ValueError("A word margin mapping needs margin, speaker_margin, or b_minus_a.")
            item = WordSpeakerMargin(
                word_index=int(value.get("word_index", value.get("index", position))),
                margin=float(margin),
                support=float(value.get("support", 1.0)),
            )
        elif isinstance(value, (tuple, list)) and len(value) in (2, 3):
            if len(value) == 2:
                item = WordSpeakerMargin(position, float(value[0]), float(value[1]))
            else:
                item = WordSpeakerMargin(int(value[0]), float(value[1]), float(value[2]))
        else:
            item = WordSpeakerMargin(position, float(value), 1.0)
        if coerced and item.word_index <= coerced[-1].word_index:
            raise ValueError("word_index values must be strictly increasing.")
        coerced.append(item)
    return tuple(coerced)


def _strictly_beats(gain: float, required_gain: float) -> bool:
    return gain >= required_gain and gain > 0.0


def _finite_similarity_scores(values: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, value in dict(values or {}).items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(score):
            scores[str(label)] = score
    return scores


def _context_side_scores(
    similarities: Mapping[str, Any],
    expected_speaker: str,
    other_speaker: str,
) -> tuple[float, float, float, float, float] | None:
    scores = _finite_similarity_scores(similarities)
    if expected_speaker not in scores or other_speaker not in scores:
        return None
    expected = scores[expected_speaker]
    other = scores[other_speaker]
    runner_up = max(
        (score for label, score in scores.items() if label != expected_speaker),
        default=-1.0,
    )
    return (
        expected,
        other,
        runner_up,
        expected - other,
        expected - runner_up,
    )


def select_context_handoff(
    left_similarities: Mapping[str, Any],
    right_similarities: Mapping[str, Any],
    speaker_a: str,
    speaker_b: str,
    config: HandoffConfig | None = None,
) -> ContextHandoffSelection:
    """Verify a nominated A->B gap from longer context on its two sides.

    Isolated words are often too short to carry a stable voice signature.  The
    verifier therefore embeds one speech-bearing block ending at the candidate
    gap and one beginning there.  Both expected speakers must beat the other
    candidate and every established third-speaker profile; the two pairwise
    margins must also provide strong combined separation.
    """

    active = config or HandoffConfig()
    left = _context_side_scores(left_similarities, speaker_a, speaker_b)
    right = _context_side_scores(right_similarities, speaker_b, speaker_a)
    if left is None or right is None:
        values = (-1.0, -1.0, -1.0, 0.0, 0.0)
        left = left or values
        right = right or values
        reason = "missing_context_profile_score"
    else:
        left_expected, _, _, left_pair, left_runner = left
        right_expected, _, _, right_pair, right_runner = right
        separation = left_pair + right_pair
        if (
            left_expected < active.min_context_similarity
            or right_expected < active.min_context_similarity
        ):
            reason = "weak_context_similarity"
        elif (
            left_pair < active.min_context_pair_margin
            or right_pair < active.min_context_pair_margin
        ):
            reason = "weak_context_pair_margin"
        elif (
            left_runner < active.min_context_runner_up_margin
            or right_runner < active.min_context_runner_up_margin
        ):
            reason = "context_matches_third_speaker"
        elif separation < active.min_context_separation:
            reason = "weak_context_separation"
        else:
            reason = "accepted"

    separation = left[3] + right[3]
    return ContextHandoffSelection(
        accepted=reason == "accepted",
        reason=reason,
        left_expected_similarity=float(left[0]),
        left_other_similarity=float(left[1]),
        left_runner_up_similarity=float(left[2]),
        left_pair_margin=float(left[3]),
        left_runner_up_margin=float(left[4]),
        right_expected_similarity=float(right[0]),
        right_other_similarity=float(right[1]),
        right_runner_up_similarity=float(right[2]),
        right_pair_margin=float(right[3]),
        right_runner_up_margin=float(right[4]),
        separation=float(separation),
    )


def select_word_handoff(
    word_margins: Sequence[Any],
    config: HandoffConfig | None = None,
) -> WordHandoffSelection:
    """Select an exact A-to-B word cut, returning rejection diagnostics too.

    A margin is ``score(B) - score(A)``.  Because A's contribution can be
    subtracted from every model without changing their ranking, A-only has
    score zero, B-only is the total margin, A->B(k) is the suffix sum after
    ``k``, and B->A(k) is the prefix sum before ``k``.  ``support`` acts as a
    reliability weight and as an optional minimum-evidence gate.
    """

    active = config or HandoffConfig()
    words = _coerce_word_margins(word_margins)
    weighted = [item.margin * item.support for item in words]
    support_prefix = [0.0]
    score_prefix = [0.0]
    for item, score in zip(words, weighted):
        support_prefix.append(support_prefix[-1] + item.support)
        score_prefix.append(score_prefix[-1] + score)
    total_score = score_prefix[-1]
    total_support = support_prefix[-1]
    a_only = 0.0
    b_only = total_score
    no_split = max(a_only, b_only)

    valid: list[tuple[int, int, float, float, float, float]] = []
    minimum_words = active.min_words_per_segment
    for position in range(minimum_words, len(words) - minimum_words + 1):
        left_support = support_prefix[position]
        right_support = total_support - left_support
        if (
            left_support < active.min_word_support_per_segment
            or right_support < active.min_word_support_per_segment
        ):
            continue
        cut_index = words[position].word_index
        forward_score = total_score - score_prefix[position]
        reverse_score = score_prefix[position]
        valid.append(
            (
                position,
                cut_index,
                forward_score,
                reverse_score,
                left_support,
                right_support,
            )
        )

    if not valid:
        return WordHandoffSelection(
            accepted=False,
            cut_index=None,
            candidate_cut_index=None,
            reason="insufficient_segment_support",
            a_only_score=a_only,
            b_only_score=b_only,
            no_split_score=no_split,
            forward_score=_NEGATIVE_INFINITY,
            reverse_score=_NEGATIVE_INFINITY,
            runner_up_forward_score=_NEGATIVE_INFINITY,
            gain_over_no_split=_NEGATIVE_INFINITY,
            gain_over_reverse=_NEGATIVE_INFINITY,
            gain_over_runner_up_cut=_NEGATIVE_INFINITY,
            left_word_count=0,
            right_word_count=0,
            left_support=0.0,
            right_support=0.0,
        )

    ranked_forward = sorted(valid, key=lambda item: (-item[2], item[1]))
    best = ranked_forward[0]
    position, candidate_cut, forward_score, _, left_support, right_support = best
    runner_up = ranked_forward[1][2] if len(ranked_forward) > 1 else _NEGATIVE_INFINITY
    reverse_score = max(item[3] for item in valid)
    gain_no_split = forward_score - no_split
    gain_reverse = forward_score - reverse_score
    gain_runner_up = (
        forward_score - runner_up if runner_up != _NEGATIVE_INFINITY else float("inf")
    )

    if not _strictly_beats(gain_no_split, active.min_gain_over_no_split):
        reason = "forward_does_not_beat_no_split"
    elif not _strictly_beats(gain_reverse, active.min_gain_over_reverse):
        reason = "forward_does_not_beat_reverse"
    elif not _strictly_beats(gain_runner_up, active.min_gain_over_runner_up_cut):
        reason = "ambiguous_forward_cut"
    else:
        reason = "accepted"

    accepted = reason == "accepted"
    return WordHandoffSelection(
        accepted=accepted,
        cut_index=candidate_cut if accepted else None,
        candidate_cut_index=candidate_cut,
        reason=reason,
        a_only_score=float(a_only),
        b_only_score=float(b_only),
        no_split_score=float(no_split),
        forward_score=float(forward_score),
        reverse_score=float(reverse_score),
        runner_up_forward_score=float(runner_up),
        gain_over_no_split=float(gain_no_split),
        gain_over_reverse=float(gain_reverse),
        gain_over_runner_up_cut=float(gain_runner_up),
        left_word_count=position,
        right_word_count=len(words) - position,
        left_support=float(left_support),
        right_support=float(right_support),
    )


def _word_bounds(word: Mapping[str, Any]) -> tuple[float, float]:
    start = float(word.get("start", 0.0))
    end = float(word.get("end", start))
    return start, max(start, end)


def _word_duration(word: Mapping[str, Any]) -> float:
    start, end = _word_bounds(word)
    try:
        explicit = float(word.get("duration", end - start))
    except (TypeError, ValueError):
        explicit = end - start
    return max(0.0, explicit)


def _text_from_words(words: Sequence[Mapping[str, Any]]) -> str:
    chunks = [str(word.get("text") or "") for word in words]
    if any(chunk[:1].isspace() for chunk in chunks):
        return "".join(chunks).strip()
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def _semantic_sentence_id(sentence: SentencePart, requested: str | None) -> str:
    existing = str(getattr(sentence, "semantic_sentence_id", "") or "").strip()
    if requested:
        return str(requested)
    if existing:
        return existing
    return f"speaker-handoff:{sentence.start:.4f}:{sentence.end:.4f}"


def split_sentence_part(
    sentence: SentencePart,
    handoff: WordHandoffSelection | int,
    *,
    boundary_time: float | None = None,
    speaker_a: str | None = None,
    speaker_b: str | None = None,
    semantic_group_id: str | None = None,
) -> SentencePartSplit:
    """Split a SentencePart at an accepted textual cut without losing grouping.

    Passing a rejected ``WordHandoffSelection`` is an error, which prevents a
    caller from accidentally applying its diagnostic runner-up cut.  The two
    returned SentencePart values share ``semantic_sentence_id`` and carry their
    A/B handoff metadata in the fields provided by ``window_domain``.
    """

    if isinstance(handoff, WordHandoffSelection):
        if not handoff.accepted or handoff.cut_index is None:
            raise ValueError("Only an accepted word handoff can split a sentence.")
        cut_index = handoff.cut_index
        selection = handoff
    else:
        cut_index = int(handoff)
        selection = None

    if cut_index <= 0 or cut_index >= len(sentence.words):
        raise ValueError("The handoff cut must leave at least one word on each side.")
    left_words = [dict(word) for word in sentence.words[:cut_index]]
    right_words = [dict(word) for word in sentence.words[cut_index:]]
    left_last_end = _word_bounds(left_words[-1])[1]
    right_first_start = _word_bounds(right_words[0])[0]
    if boundary_time is None:
        split_time = 0.5 * (left_last_end + right_first_start)
    else:
        split_time = float(boundary_time)
        if not isfinite(split_time):
            raise ValueError("boundary_time must be finite.")
    split_time = min(float(sentence.end), max(float(sentence.start), split_time))

    semantic_id = _semantic_sentence_id(sentence, semantic_group_id)
    common_handoff = dict(getattr(sentence, "speaker_handoff", {}) or {})
    common_handoff.update(
        {
            "detected": True,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "word_cut_index": cut_index,
            "boundary_time": split_time,
        }
    )
    if selection is not None:
        common_handoff.update(
            {
                "gain_over_no_split": selection.gain_over_no_split,
                "gain_over_reverse": selection.gain_over_reverse,
                "gain_over_runner_up_cut": selection.gain_over_runner_up_cut,
            }
        )

    left_spoken = sum(_word_duration(word) for word in left_words)
    right_spoken = sum(_word_duration(word) for word in right_words)
    left_duration = max(0.0, split_time - float(sentence.start))
    right_duration = max(0.0, float(sentence.end) - split_time)
    word_gap = max(0.0, right_first_start - left_last_end)

    left_handoff = dict(common_handoff)
    left_handoff.update({"role": "from", "semantic_part": 0})
    right_handoff = dict(common_handoff)
    right_handoff.update({"role": "to", "semantic_part": 1})
    left = replace(
        sentence,
        text=_text_from_words(left_words),
        end=split_time,
        next_left=split_time,
        spoken_word_seconds=float(left_spoken),
        speech_audio_ratio=left_spoken / left_duration if left_duration > 0.0 else 0.0,
        words=left_words,
        first_word_start=_word_bounds(left_words[0])[0],
        last_word_end=left_last_end,
        next_word_start=right_first_start,
        gap_to_next_word_seconds=word_gap,
        boundary_strategy="speaker_handoff_word_cut",
        asr_review=dict(sentence.asr_review),
        semantic_sentence_id=semantic_id,
        semantic_sentence_part=0,
        semantic_sentence_part_count=2,
        speaker_handoff=left_handoff,
    )
    right = replace(
        sentence,
        text=_text_from_words(right_words),
        start=split_time,
        spoken_word_seconds=float(right_spoken),
        speech_audio_ratio=right_spoken / right_duration if right_duration > 0.0 else 0.0,
        words=right_words,
        first_word_start=right_first_start,
        last_word_end=_word_bounds(right_words[-1])[1],
        boundary_strategy="speaker_handoff_word_cut",
        asr_review=dict(sentence.asr_review),
        semantic_sentence_id=semantic_id,
        semantic_sentence_part=1,
        semantic_sentence_part_count=2,
        speaker_handoff=right_handoff,
    )
    return SentencePartSplit(
        left=left,
        right=right,
        semantic_sentence_id=semantic_id,
        cut_index=cut_index,
        boundary_time=split_time,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )


__all__ = [
    "CoarseBoundaryEstimate",
    "ContextHandoffSelection",
    "HandoffConfig",
    "HandoffNomination",
    "LiveEmbeddingEvidence",
    "SentencePartSplit",
    "WordHandoffSelection",
    "WordSpeakerMargin",
    "cosine_margin",
    "estimate_coarse_boundary",
    "nominate_stable_handoff",
    "normalize_embedding",
    "select_context_handoff",
    "select_word_handoff",
    "split_sentence_part",
]
