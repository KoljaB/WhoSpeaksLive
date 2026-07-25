from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np

from window.sentence_speaker_handoff import (
    select_context_handoff,
    split_sentence_part,
)
from window.diarization_run import DiarizationRun
from window.window_diarizer_transcription import WindowTranscriptionMixin
from window.window_domain import SentencePart

from tests.window_diarizer_support import make_window_diarizer


def _sentence() -> SentencePart:
    words = [
        {"text": text, "start": start, "end": end, "duration": end - start}
        for text, start, end in (
            ("This", 0.05, 0.30),
            ("part", 0.35, 0.60),
            ("is", 0.65, 0.90),
            ("hers", 0.95, 1.20),
            ("and", 1.45, 1.70),
            ("this", 1.75, 2.00),
            ("is", 2.05, 2.30),
            ("his.", 2.35, 2.60),
        )
    ]
    spoken = sum(float(word["duration"]) for word in words)
    return SentencePart(
        text="This part is hers and this is his.",
        start=0.0,
        end=2.8,
        next_left=2.8,
        spoken_word_seconds=spoken,
        speech_audio_ratio=spoken / 2.8,
        words=words,
        first_word_start=0.05,
        last_word_end=2.60,
        next_word_start=None,
        gap_to_next_word_seconds=None,
        boundary_strategy="final_flush",
    )


def _install_profiles(diarizer) -> str:
    diarizer.live_memory.replace_profiles(
        [
            {
                "label": "S1",
                "centroid": [1.0, 0.0],
                "sentence_count": 5,
                "speech_seconds": 10.0,
            },
            {
                "label": "S2",
                "centroid": [0.0, 1.0],
                "sentence_count": 5,
                "speech_seconds": 10.0,
            },
        ]
    )
    return diarizer._current_live_embedding_provider()


def _record_probe(
    diarizer,
    provider: str,
    *,
    start: float,
    embedding: tuple[float, float],
    run_id: str = "run-1",
    source: str = "dedicated_live_probe",
) -> None:
    sample_rate = 16_000
    diarizer._record_sentence_handoff_live_evidence(
        request_source=source,
        run_id=run_id,
        probe_id=f"probe-{start}",
        request_id=f"request-{start}",
        window_start=start,
        window_end=start + 0.7,
        source_start_sample=int(start * sample_rate),
        source_end_sample=int((start + 0.7) * sample_rate),
        audio=np.full(int(0.7 * sample_rate), 0.1, dtype=np.float32),
        sample_rate=sample_rate,
        embedding=np.asarray(embedding, dtype=np.float32),
        visible_speaker="stale-live-label",
        similarities={},
        probabilities={},
        profile_generations={},
        provider=provider,
    )


def test_stable_cached_transition_runs_two_context_embeddings_and_splits() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
        sentence_speaker_handoff_context_seconds=1.0,
        sentence_speaker_handoff_max_word_snap_seconds=0.25,
    )
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)

    diarizer._audio_window_copy = lambda _left, _right: (
        np.ones(16_000 * 3, dtype=np.float32),
        16_000,
    )
    context_embeddings = iter(
        [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ]
    )
    calls: list[str] = []

    def embed_context(_audio, _sample_rate, suffix):
        calls.append(suffix)
        return next(context_embeddings)

    diarizer._embed_live_audio_chunk = embed_context

    parts = diarizer._split_completed_sentence_handoffs([_sentence()])

    assert len(parts) == 2
    left, right = parts
    assert left.text == "This part is hers"
    assert right.text == "and this is his."
    assert left.end == right.start
    assert left.semantic_sentence_id == right.semantic_sentence_id
    assert (left.semantic_sentence_part, right.semantic_sentence_part) == (0, 1)
    assert left.speaker_handoff["speaker_a"] == "S1"
    assert right.speaker_handoff["speaker_b"] == "S2"
    assert left.speaker_handoff["algorithm"] == "cached_live_context_one_cut_v2"
    assert calls == [
        ".handoff.left-context.wav",
        ".handoff.right-context.wav",
    ]


def test_default_mode_defers_split_until_final_provider_confirmation() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)
    diarizer._embed_live_audio_chunk = mock.Mock(
        side_effect=AssertionError("default mode must wait for finalization")
    )
    sentence = _sentence()

    assert diarizer._split_completed_sentence_handoffs([sentence]) == [sentence]
    diarizer._embed_live_audio_chunk.assert_not_called()


def test_default_word_snap_gate_rejects_boundary_more_than_250_ms_away() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
    )
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)

    sentence = _sentence()
    words = [dict(word) for word in sentence.words]
    for word in words[4:]:
        word["start"] += 0.65
        word["end"] += 0.65
    sentence = replace(
        sentence,
        end=3.45,
        next_left=3.45,
        words=words,
        last_word_end=3.25,
    )
    diarizer._embed_live_audio_chunk = mock.Mock(
        side_effect=AssertionError("the focused verifier must not run")
    )

    parts = diarizer._split_completed_sentence_handoffs([sentence])

    assert parts == [sentence]
    diarizer._embed_live_audio_chunk.assert_not_called()


def test_cache_ignores_non_dedicated_probes_and_resets_between_runs() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    provider = _install_profiles(diarizer)

    _record_probe(
        diarizer,
        provider,
        start=0.0,
        embedding=(1.0, 0.0),
        source="growing_realtime_preview",
    )
    assert list(diarizer._sentence_handoff_evidence) == []

    _record_probe(diarizer, provider, start=0.0, embedding=(1.0, 0.0))
    _record_probe(
        diarizer,
        provider,
        start=1.0,
        embedding=(0.0, 1.0),
        run_id="run-2",
    )
    assert len(diarizer._sentence_handoff_evidence) == 1
    assert diarizer._sentence_handoff_evidence_run_id == "run-2"

    diarizer._reset_live_speaker_memory()
    assert list(diarizer._sentence_handoff_evidence) == []
    assert diarizer._sentence_handoff_evidence_run_id == ""


def test_default_cache_keeps_five_minutes_and_prunes_after_one_hour() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    provider = _install_profiles(diarizer)
    assert diarizer.args.sentence_speaker_handoff_cache_seconds == 3600.0
    assert not diarizer.args.sentence_speaker_handoff_immediate
    assert diarizer.args.sentence_speaker_handoff_max_word_snap_seconds == 0.25
    assert diarizer.args.sentence_speaker_handoff_max_hindsight_seconds == 5.0

    _record_probe(diarizer, provider, start=0.0, embedding=(1.0, 0.0))
    _record_probe(diarizer, provider, start=300.0, embedding=(1.0, 0.0))

    assert len(diarizer._sentence_handoff_evidence) == 2

    _record_probe(diarizer, provider, start=3601.0, embedding=(1.0, 0.0))

    retained_starts = [
        item.window_start for item in diarizer._sentence_handoff_evidence
    ]
    assert retained_starts == [300.0, 3601.0]


def test_single_speaker_sentence_spends_no_word_embedding_compute() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
    )
    provider = _install_profiles(diarizer)
    for start in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0):
        _record_probe(diarizer, provider, start=start, embedding=(1.0, 0.0))

    diarizer._embed_live_audio_chunk = mock.Mock(
        side_effect=AssertionError("word verifier must not run")
    )
    sentence = _sentence()

    parts = diarizer._split_completed_sentence_handoffs([sentence])

    assert parts == [sentence]
    diarizer._embed_live_audio_chunk.assert_not_called()


def test_slow_live_backend_skips_focused_verification() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
        sentence_speaker_handoff_max_verification_seconds=0.8,
    )
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)
    diarizer._live_speaker_embedding_latency_ewma = 0.5
    diarizer._embed_live_audio_chunk = mock.Mock(
        side_effect=AssertionError("slow backend must not be monopolized")
    )
    sentence = _sentence()

    parts = diarizer._split_completed_sentence_handoffs([sentence])

    assert parts == [sentence]
    diarizer._embed_live_audio_chunk.assert_not_called()


def test_busy_live_provider_skips_focused_verification_without_overlap() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
    )
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)
    diarizer._audio_window_copy = lambda _left, _right: (
        np.ones(16_000 * 3, dtype=np.float32),
        16_000,
    )
    diarizer._embed_live_audio_chunk = mock.Mock(
        side_effect=AssertionError("handoff verification must not overlap")
    )
    inference_lock = diarizer._live_speaker_inference_lock_obj()
    inference_lock.acquire()
    try:
        parts = diarizer._split_completed_sentence_handoffs([_sentence()])
    finally:
        inference_lock.release()

    assert parts == [_sentence()]
    diarizer._embed_live_audio_chunk.assert_not_called()


def test_context_verifier_rejects_a_third_speaker_winner() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        sentence_speaker_handoff_immediate=True,
    )
    provider = _install_profiles(diarizer)
    for start, embedding in (
        (0.0, (1.0, 0.0)),
        (0.4, (1.0, 0.0)),
        (0.8, (1.0, 0.0)),
        (1.2, (0.0, 1.0)),
        (1.6, (0.0, 1.0)),
        (2.0, (0.0, 1.0)),
    ):
        _record_probe(diarizer, provider, start=start, embedding=embedding)
    rejected = select_context_handoff(
        {"S1": 0.70, "S2": 0.10, "S3": 0.69},
        {"S1": 0.05, "S2": 0.75, "S3": 0.10},
        "S1",
        "S2",
    )
    assert not rejected.accepted
    diarizer._verify_sentence_handoff_context = mock.Mock(
        return_value=(rejected, 2, "")
    )
    sentence = _sentence()

    parts = diarizer._split_completed_sentence_handoffs([sentence])

    assert parts == [sentence]


def test_transcription_hook_replaces_one_completed_part_with_split_parts() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    sentence = _sentence()
    expected = [
        replace(sentence, text="left"),
        replace(sentence, text="right"),
    ]
    diarizer._transcribe_window_audio_words = lambda *_args, **_kwargs: ([], 1)
    splitter = mock.Mock(return_value=expected)
    diarizer._split_completed_sentence_handoffs = splitter

    with mock.patch(
        "window.window_diarizer_transcription.split_words_with_stream2sentence",
        return_value=[sentence],
    ):
        transcript = diarizer._transcribe_window(
            model=None,
            left=0.0,
            right=3.0,
            final_flush=True,
        )

    splitter.assert_called_once_with([sentence])
    assert transcript.sentences == expected


def _install_hindsight_profiles(diarizer) -> None:
    profiles = [
        {
            "label": "S1",
            "centroid": [1.0, 0.0],
            "sentence_count": 5,
            "speech_seconds": 10.0,
        },
        {
            "label": "S2",
            "centroid": [0.0, 1.0],
            "sentence_count": 5,
            "speech_seconds": 10.0,
        },
    ]
    diarizer.memory.replace_profiles(profiles)
    if diarizer.live_memory is not diarizer.memory:
        diarizer.live_memory.replace_profiles(profiles)


def _install_emitted_hindsight_parent(
    diarizer,
    *,
    index: int = 4,
) -> dict:
    sentence = _sentence()
    base_payload = diarizer._base_payload_from_sentence_part(
        index,
        sentence,
        0.0,
        2.8,
    )
    diarizer._sentence_refinement_records[index] = {
        "index": index,
        "base_payload": base_payload,
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "duration_seconds": 2.8,
        "assigned_speaker": "S1",
        "created_speaker": False,
        "probabilities": {"unknown": 0.0, "speaker1": 1.0},
        "similarities": {"S1": 1.0, "S2": 0.0},
        "unknown_probability": 0.0,
        "top_similarity": 1.0,
        "margin": 1.0,
        "quality": 1.0,
        "assignment_source": "embedding",
    }
    diarizer._remember_sentence_handoff_hindsight_candidate(
        index,
        sentence,
        base_payload,
    )
    return diarizer._sentence_refinement_records[index]


def _verified_hindsight_split() -> tuple[SentencePart, SentencePart]:
    sentence = _sentence()
    result = split_sentence_part(
        sentence,
        4,
        boundary_time=1.325,
        speaker_a="S1",
        speaker_b="S2",
        semantic_group_id="sentence:hindsight-test",
    )
    return result.left, result.right


def test_hindsight_retry_reuses_parent_index_adds_one_child_and_is_idempotent() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        audio=np.ones(16_000 * 3, dtype=np.float32),
    )
    _install_hindsight_profiles(diarizer)
    diarizer._active_run = SimpleNamespace(run_id="run-1")
    _install_emitted_hindsight_parent(diarizer, index=4)
    diarizer._sentence_refinement_records[8] = {
        "index": 8,
        "base_payload": {
            "index": 8,
            "text": "Later.",
            "source_text_hash": "later",
            "source_revision": "later",
            "start": 10.0,
            "end": 11.0,
        },
        "embedding": None,
        "duration_seconds": 1.0,
        "assigned_speaker": None,
        "created_speaker": False,
        "probabilities": {"unknown": 1.0},
        "similarities": {},
        "unknown_probability": 1.0,
        "top_similarity": None,
        "margin": None,
        "quality": None,
        "assignment_source": "unknown",
    }
    left, right = _verified_hindsight_split()
    diarizer._split_one_completed_sentence_handoff = mock.Mock(
        return_value=(left, right)
    )
    child_embeddings = iter(
        [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ]
    )
    diarizer._embed_audio_chunk = mock.Mock(
        side_effect=lambda *_args: next(child_embeddings)
    )
    old_memory = diarizer.memory
    shared_live_memory = diarizer.live_memory is old_memory

    assert diarizer._finalize_sentence_handoff_hindsight() == 1
    assert diarizer.memory is not old_memory
    if shared_live_memory:
        assert diarizer.live_memory is diarizer.memory
    assert set(diarizer._sentence_refinement_records) == {4, 8, 9}
    assert diarizer._sentence_refinement_records[4]["assigned_speaker"] == "S1"
    assert diarizer._sentence_refinement_records[9]["assigned_speaker"] == "S2"
    assert diarizer._sentence_refinement_records[4]["base_payload"]["text"] == (
        "This part is hers"
    )
    assert diarizer._sentence_refinement_records[9]["base_payload"]["text"] == (
        "and this is his."
    )
    assert diarizer._sentence_refinement_records[4]["base_payload"][
        "speaker_handoff"
    ]["memory_rebuilt"]
    assert diarizer._sentence_refinement_records[9]["base_payload"][
        "speaker_handoff"
    ]["replaces_sentence_index"] == 4

    final_sentence_indexes = [
        item["payload"]["index"]
        for item in diarizer.bus.records
        if item["event"] == "sentence" and not item["payload"].get("pending")
    ]
    assert final_sentence_indexes == [4, 9]
    rows, _embeddings = diarizer._session_transcript_rows_and_embeddings()
    assert [row["index"] for row in rows] == [4, 9, 8]

    assert diarizer._finalize_sentence_handoff_hindsight() == 0
    assert [
        item["payload"]["index"]
        for item in diarizer.bus.records
        if item["event"] == "sentence" and not item["payload"].get("pending")
    ] == [4, 9]


def test_hindsight_retry_rejects_user_corrected_parent_without_compute() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    _install_hindsight_profiles(diarizer)
    diarizer._active_run = SimpleNamespace(run_id="run-1")
    record = _install_emitted_hindsight_parent(diarizer)
    record["correction"] = {
        "corrected": True,
        "source": "user",
    }
    diarizer._split_one_completed_sentence_handoff = mock.Mock(
        side_effect=AssertionError("corrected rows must not be retried")
    )
    before_profiles = diarizer.memory.export_profiles()

    assert diarizer._finalize_sentence_handoff_hindsight() == 0
    diarizer._split_one_completed_sentence_handoff.assert_not_called()
    assert set(diarizer._sentence_refinement_records) == {4}
    assert diarizer.memory.export_profiles() == before_profiles


def test_hindsight_retry_does_not_publish_when_memory_rebuild_is_unsafe() -> None:
    diarizer = make_window_diarizer(
        sentence_speaker_handoff=True,
        audio=np.ones(16_000 * 3, dtype=np.float32),
    )
    _install_hindsight_profiles(diarizer)
    diarizer._active_run = SimpleNamespace(run_id="run-1")
    _install_emitted_hindsight_parent(diarizer)
    diarizer._sentence_refinement_records[7] = {
        "index": 7,
        "base_payload": {
            "index": 7,
            "text": "Corrected without audio evidence.",
            "source_text_hash": "manual",
            "source_revision": "manual",
            "start": 8.0,
            "end": 9.0,
        },
        "embedding": None,
        "duration_seconds": 1.0,
        "assigned_speaker": "S1",
        "created_speaker": False,
        "probabilities": {"unknown": 0.0, "speaker1": 1.0},
        "similarities": {},
        "unknown_probability": 0.0,
        "top_similarity": None,
        "margin": None,
        "quality": None,
        "assignment_source": "user_correction",
    }
    left, right = _verified_hindsight_split()
    diarizer._split_one_completed_sentence_handoff = mock.Mock(
        return_value=(left, right)
    )
    child_embeddings = iter(
        [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ]
    )
    diarizer._embed_audio_chunk = mock.Mock(
        side_effect=lambda *_args: next(child_embeddings)
    )
    before_records = copy.deepcopy(diarizer._sentence_refinement_records)
    before_profiles = diarizer.memory.export_profiles()

    assert diarizer._finalize_sentence_handoff_hindsight() == 0
    assert set(diarizer._sentence_refinement_records) == set(before_records)
    assert diarizer._sentence_refinement_records[4]["base_payload"]["text"] == (
        before_records[4]["base_payload"]["text"]
    )
    assert diarizer.memory.export_profiles() == before_profiles
    assert not [
        item
        for item in diarizer.bus.records
        if item["event"] == "sentence" and not item["payload"].get("pending")
    ]


def test_handoff_emit_hook_remembers_stable_index() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    diarizer._active_run = SimpleNamespace(run_id="run-1")
    sentence = _sentence()
    with mock.patch.object(
        WindowTranscriptionMixin,
        "_emit_sentence",
        autospec=True,
    ) as emit_super:
        diarizer._emit_sentence(12, sentence, 0.0, 2.8)

    emit_super.assert_called_once_with(diarizer, 12, sentence, 0.0, 2.8)
    assert 12 in diarizer._sentence_handoff_hindsight_candidates


def test_main_worker_retries_after_live_probe_quiesces_even_on_explicit_stop() -> None:
    diarizer = make_window_diarizer(sentence_speaker_handoff=True)
    run = DiarizationRun(processing_mode="playback")
    run.request_stop()
    diarizer._active_run = run

    calls = mock.Mock()
    diarizer._run = mock.Mock(return_value=None)
    diarizer._stop_embedding_worker = mock.Mock()
    diarizer._stop_live_memory_update_worker = mock.Mock()
    diarizer._finalize_sentence_handoff_hindsight = mock.Mock(return_value=0)
    diarizer.consolidate_confirmed_people = mock.Mock()
    diarizer.emit_authoritative_final_speaker_memory_state = mock.Mock()
    calls.attach_mock(diarizer._stop_live_memory_update_worker, "stop_live")
    calls.attach_mock(
        diarizer._finalize_sentence_handoff_hindsight,
        "finalize_handoffs",
    )

    diarizer._run_main_worker(run)

    diarizer._run.assert_called_once_with(run.stop_event)
    diarizer._finalize_sentence_handoff_hindsight.assert_called_once_with()
    assert calls.mock_calls.index(mock.call.stop_live()) < calls.mock_calls.index(
        mock.call.finalize_handoffs()
    )
