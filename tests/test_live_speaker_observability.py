from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import numpy as np

from window.window_diarizer_live_scoring import WindowLiveScoringMixin
from window.window_diarizer_runtime_audio import WindowRuntimeAudioMixin
from window.window_domain import EmbeddingSentenceJob, LiveSpeakerMemoryUpdateJob


class _RecordingBus:
    def __init__(self) -> None:
        self.internal: list[tuple[str, dict]] = []
        self.public: list[tuple[str, dict]] = []

    def has_internal_listeners(self) -> bool:
        return True

    def emit_internal(self, event: str, payload: dict) -> None:
        self.internal.append((event, payload))

    def emit(self, event: str, payload: dict) -> None:
        self.public.append((event, payload))


class _ProfileMemory:
    def profile_count(self) -> int:
        return 1

    def export_profiles(self):
        return [{
            "label": "S1",
            "centroid": [1.0, 0.0],
            "sentence_count": 1,
            "speech_seconds": 1.0,
        }]


class _ScoringHarness(WindowLiveScoringMixin):
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            realtime_preview_diarize_min_audio_seconds=0.1,
            live_speaker_tracker="classic",
            live_speaker_bayes_provisional_profiles=False,
            min_embed_seconds=0.1,
        )
        self.sample_rate = 10
        self.live_memory = _ProfileMemory()
        self.bus = _RecordingBus()
        self._vectors = iter(
            (
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            )
        )

    def _live_speaker_assignment_enabled(self) -> bool:
        return True

    def _live_speaker_correlation_run_id(self) -> str:
        return "run-1"

    def playback_time(self) -> float:
        return 2.2

    def _current_live_embedding_provider(self) -> str:
        return "test-provider"

    def _embed_live_audio_chunk(self, *_args, **_kwargs) -> np.ndarray:
        return next(self._vectors)

    def _record_live_speaker_embedding_latency(self, *_args, **_kwargs) -> None:
        return None

    def _shared_live_speaker_step(self, *, correlation_out=None, **_kwargs):
        if correlation_out is not None:
            correlation_out.update(
                run_id="run-1",
                probe_id="probe-1",
                request_id="request-1",
                step_id=17,
            )
        return SimpleNamespace(
            diagnostics={},
            visible_speaker="S1",
            raw_probabilities={"S1": 0.9, "unknown": 0.1},
            probabilities={"S1": 0.9, "unknown": 0.1},
            similarities={"S1": 0.8},
            action="acquire",
            reason="test",
        )

    def _ensure_speaker_metadata(self, *_args, **_kwargs) -> None:
        return None

    def _speaker_info_for_payload(self, speaker_id):
        return {"speaker_name": speaker_id}


def test_live_embedding_observation_keeps_short_context_and_effective_vectors_separate() -> None:
    harness = _ScoringHarness()
    payload = harness._score_realtime_preview_speaker(
        np.ones(10, dtype=np.float32),
        1.0,
        context_audio=np.ones(20, dtype=np.float32),
        context_duration_seconds=2.0,
        context_weight=0.25,
        request_source="dedicated_live_probe",
        request_id="request-1",
        run_id="run-1",
        probe_id="probe-1",
        short_window_start=1.0,
        short_window_end=2.0,
        short_source_start_sample=10,
        short_source_end_sample=20,
        context_window_start=0.0,
        context_window_end=2.0,
        context_source_start_sample=0,
        context_source_end_sample=20,
    )

    completed = next(
        event_payload
        for event, event_payload in harness.bus.internal
        if event == "live_speaker_embedding_request_completed"
    )
    np.testing.assert_allclose(completed["short_embedding"], [1.0, 0.0])
    np.testing.assert_allclose(completed["context_embedding"], [0.0, 1.0])
    assert not np.allclose(completed["effective_embedding"], completed["short_embedding"])
    assert completed["short_source_start_sample"] == 10
    assert completed["short_source_end_sample"] == 20
    assert payload["run_id"] == "run-1"
    assert payload["probe_id"] == "probe-1"
    assert payload["request_id"] == "request-1"
    assert payload["step_id"] == 17
    bound = next(
        event_payload
        for event, event_payload in harness.bus.internal
        if event == "live_speaker_embedding_request_step_bound"
    )
    assert bound["step_id"] == 17
    assert bound["request_id"] == "request-1"


class _CoreHarness(WindowLiveScoringMixin):
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            live_speaker_tracker="classic",
            realtime_preview_diarize_min_similarity=0.1,
            realtime_preview_diarize_min_margin=0.0,
            realtime_preview_diarize_min_known_probability=0.0,
            live_speaker_ema_count=1,
            live_speaker_ema_alpha=1.0,
            live_speaker_acquire_count=1,
            live_speaker_switch_count=1,
            live_speaker_probe_clear_unknown_count=1,
            live_speaker_probe_clear_silence_count=1,
        )
        self.live_memory = _ProfileMemory()
        self.bus = _RecordingBus()

    def _live_speaker_correlation_run_id(self) -> str:
        return "run-core"


def test_shared_core_propagates_all_correlation_ids_to_public_and_internal_events() -> None:
    harness = _CoreHarness()
    correlation: dict = {}
    harness._shared_live_speaker_step(
        media_time=1.0,
        speech=True,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        duration_seconds=1.0,
        probe_scheduled=True,
        run_id="run-core",
        probe_id="probe-core",
        request_id="request-core",
        correlation_out=correlation,
    )

    assert correlation == {
        "run_id": "run-core",
        "probe_id": "probe-core",
        "request_id": "request-core",
        "step_id": 1,
    }
    public = next(
        payload
        for event, payload in harness.bus.public
        if event == "live_speaker_shared_core_decision"
    )
    internal = next(
        payload
        for event, payload in harness.bus.internal
        if event == "live_speaker_core_decision"
    )
    for key, value in correlation.items():
        assert public[key] == value
        assert internal[key] == value


class _SileroModel:
    def reset_states(self) -> None:
        return None

    def __call__(self, _audio: np.ndarray, _sample_rate: int) -> float:
        return 0.9


class _VadHarness(WindowRuntimeAudioMixin):
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            vad_merge_gap_seconds=0.0,
            vad_min_speech_seconds=0.0,
            vad_silence_seconds=0.2,
            vad_silero_speech_threshold=0.5,
        )
        self._vad_model_backend = "silero-test"

    def _load_silero_vad_model(self):
        return _SileroModel()


def test_silero_observation_preserves_resampled_and_per_frame_valid_lengths() -> None:
    harness = _VadHarness()
    diagnostics: dict = {}
    state = harness._silero_vad_window_state(
        0.0,
        0.05625,
        np.ones(450, dtype=np.float32),
        8000,
        diagnostics=diagnostics,
    )

    assert state.has_speech
    assert diagnostics["silero_source_sample_count"] == 450
    assert diagnostics["silero_resampled_sample_count"] == 900
    assert diagnostics["silero_frame_valid_samples"] == [512, 388]
    assert diagnostics["silero_frame_padded_samples"] == [0, 124]
    assert diagnostics["silero_discarded_tail_sample_count"] == 0


class _QueueHarness(WindowRuntimeAudioMixin):
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    def _record_final_sentence_queue_stage(self, job, stage, **details) -> None:
        self.events.append((job.job_id, stage, str(details.get("reason") or "")))

    def _record_live_profile_queue_stage(self, job, stage, **details) -> None:
        self.events.append((job.job_id, stage, str(details.get("reason") or "")))


def test_queue_cancellation_records_job_type_and_reason() -> None:
    jobs: Queue = Queue()
    jobs.put(
        EmbeddingSentenceJob(
            index=1,
            base_payload={"start": 0.0, "end": 1.0},
            text="hello",
            audio=np.ones(10, dtype=np.float32),
            sample_rate=10,
            duration_seconds=1.0,
            job_id="final-1",
        )
    )
    jobs.put(
        LiveSpeakerMemoryUpdateJob(
            speaker_id="S1",
            audio=np.ones(10, dtype=np.float32),
            sample_rate=10,
            duration_seconds=1.0,
            job_id="profile-1",
        )
    )
    harness = _QueueHarness()
    harness._cancel_pending_embedding_jobs(jobs, reason="test_cancel")

    assert harness.events == [
        ("final-1", "cancelled", "test_cancel"),
        ("profile-1", "cancelled", "test_cancel"),
    ]
    assert jobs.unfinished_tasks == 0
