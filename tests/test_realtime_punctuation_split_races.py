from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


from RealtimeSTT.core import lifecycle, realtime
from RealtimeSTT.core.recording_buffers import (
    get_frames_lock,
    get_realtime_state_lock,
    queue_recorded_audio,
    snapshot_frames,
)


def _pcm(value: int, samples: int) -> bytes:
    return np.full(samples, value, dtype=np.int16).tobytes()


class _Stabilizer:
    def __init__(self) -> None:
        self.reset_recording_ids: list[int] = []
        self.finalize_count = 0

    def reset(self, recording_id: int, **_kwargs: object) -> None:
        self.reset_recording_ids.append(recording_id)

    def finalize(self) -> None:
        self.finalize_count += 1


class _Recorder:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames_lock = threading.RLock()
        self.realtime_state_lock = threading.RLock()
        self._realtime_punctuation_split_lock = threading.RLock()
        self.recorded_audio_queue: queue.Queue[dict[str, object]] = queue.Queue()
        self.frames = list(frames)
        self.last_frames: list[bytes] = []
        self.is_recording = True
        self.realtime_recording_id = 7
        self.realtime_recording_generation = 11
        self.recording_start_monotonic = 1.0
        self.recording_start_time = 1.0
        self.text_storage = ["old"]
        self.realtime_transcription_text = "old"
        self.realtime_stabilized_text = "old"
        self.realtime_stabilized_safetext = "old"
        self.realtime_observation_sequence = 4
        self._force_current_recording_lowercase_start = False
        self._last_realtime_punctuation_split_attempt_text = "old"
        self._realtime_punctuation_split_candidate = None
        self.realtime_text_stabilizer = _Stabilizer()


class RealtimePunctuationSplitRaceTests(unittest.TestCase):
    def test_commit_keeps_frame_appended_while_split_is_calculated(self) -> None:
        """A frame arriving during the cut must follow the right-side remainder."""

        first = _pcm(1, 4)
        second = _pcm(2, 4)
        third = _pcm(3, 4)
        appended = _pcm(4, 3)
        recorder = _Recorder([first, second, third])
        split_entered = threading.Event()
        allow_split = threading.Event()
        append_started = threading.Event()
        append_finished = threading.Event()
        result: list[bool] = []
        original_split = realtime._split_frames_at_sample

        def blocked_split(frames: list[bytes], split_sample: int):
            split_entered.set()
            self.assertTrue(allow_split.wait(timeout=2.0))
            return original_split(frames, split_sample)

        def commit() -> None:
            result.append(
                bool(
                    realtime._commit_realtime_punctuation_split(
                        recorder,
                        expected_recording_id=7,
                        expected_generation=11,
                        split_sample=5,
                        punctuation=".",
                        sample_rate=10,
                    )
                )
            )

        def append_audio() -> None:
            append_started.set()
            with get_frames_lock(recorder):
                recorder.frames.append(appended)
            append_finished.set()

        with mock.patch.object(
            realtime,
            "_split_frames_at_sample",
            side_effect=blocked_split,
        ):
            commit_thread = threading.Thread(target=commit)
            commit_thread.start()
            self.assertTrue(split_entered.wait(timeout=2.0))

            append_thread = threading.Thread(target=append_audio)
            append_thread.start()
            self.assertTrue(append_started.wait(timeout=2.0))
            self.assertFalse(append_finished.wait(timeout=0.05))

            allow_split.set()
            commit_thread.join(timeout=2.0)
            append_thread.join(timeout=2.0)

        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(append_thread.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(b"".join(recorder.frames), second[2:] + third + appended)
        queued = recorder.recorded_audio_queue.get_nowait()
        self.assertEqual(b"".join(queued["frames"]), first + second[:2])

    def test_stale_split_is_discarded_after_stop_and_new_start(self) -> None:
        """A delayed split for recording A must not cut recording B."""

        recorder = _Recorder([_pcm(1, 8)])
        request_captured = threading.Event()
        allow_commit = threading.Event()
        result: list[bool] = []

        def delayed_commit() -> None:
            request_captured.set()
            self.assertTrue(allow_commit.wait(timeout=2.0))
            result.append(
                bool(
                    realtime._commit_realtime_punctuation_split(
                        recorder,
                        expected_recording_id=7,
                        expected_generation=11,
                        split_sample=4,
                        punctuation=".",
                        sample_rate=10,
                    )
                )
            )

        split_thread = threading.Thread(target=delayed_commit)
        split_thread.start()
        self.assertTrue(request_captured.wait(timeout=2.0))

        replacement = [_pcm(9, 10), _pcm(8, 5)]
        with get_realtime_state_lock(recorder):
            with get_frames_lock(recorder):
                # Stop A invalidates its generation; starting B advances both
                # the public recording id and the private generation again.
                recorder.is_recording = False
                recorder.realtime_recording_generation += 1
                recorder.frames = []
                recorder.realtime_recording_id += 1
                recorder.realtime_recording_generation += 1
                recorder.frames = list(replacement)
                recorder.is_recording = True

        expected_id = recorder.realtime_recording_id
        expected_generation = recorder.realtime_recording_generation
        allow_commit.set()
        split_thread.join(timeout=2.0)

        self.assertFalse(split_thread.is_alive())
        self.assertEqual(result, [False])
        self.assertEqual(recorder.frames, replacement)
        self.assertEqual(recorder.realtime_recording_id, expected_id)
        self.assertEqual(
            recorder.realtime_recording_generation,
            expected_generation,
        )
        self.assertTrue(recorder.recorded_audio_queue.empty())

    def test_left_segment_is_queued_before_concurrent_stop_of_right_segment(self) -> None:
        """The split transition must reserve FIFO order before VAD can stop again."""

        recorder = _Recorder([_pcm(1, 5), _pcm(2, 5)])
        queue_entered = threading.Event()
        allow_left_queue = threading.Event()
        stop_started = threading.Event()
        stop_finished = threading.Event()
        original_queue = queue_recorded_audio

        def blocked_queue(*args: object, **kwargs: object) -> None:
            queue_entered.set()
            self.assertTrue(allow_left_queue.wait(timeout=2.0))
            original_queue(*args, **kwargs)

        def commit() -> None:
            realtime._commit_realtime_punctuation_split(
                recorder,
                expected_recording_id=7,
                expected_generation=11,
                split_sample=5,
                punctuation=".",
                sample_rate=10,
            )

        def stop_right() -> None:
            stop_started.set()
            with get_realtime_state_lock(recorder):
                with get_frames_lock(recorder):
                    right_frames = list(recorder.frames)
                    recorder.frames = []
                    recorder.is_recording = False
                    recorder.realtime_recording_generation += 1
                original_queue(recorder, right_frames)
            stop_finished.set()

        with mock.patch.object(
            realtime,
            "queue_recorded_audio",
            side_effect=blocked_queue,
        ):
            commit_thread = threading.Thread(target=commit)
            commit_thread.start()
            self.assertTrue(queue_entered.wait(timeout=2.0))

            stop_thread = threading.Thread(target=stop_right)
            stop_thread.start()
            self.assertTrue(stop_started.wait(timeout=2.0))
            self.assertFalse(stop_finished.wait(timeout=0.05))

            allow_left_queue.set()
            commit_thread.join(timeout=2.0)
            stop_thread.join(timeout=2.0)

        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        first_queued = recorder.recorded_audio_queue.get_nowait()
        second_queued = recorder.recorded_audio_queue.get_nowait()
        self.assertEqual(b"".join(first_queued["frames"]), _pcm(1, 5))
        self.assertEqual(b"".join(second_queued["frames"]), _pcm(2, 5))

    def test_completed_decode_from_old_generation_does_not_mutate_or_publish(self) -> None:
        """A realtime decode finishing after a split must be dropped completely."""

        recorder = _Recorder([_pcm(1, 8)])
        decode_started = threading.Event()
        allow_decode = threading.Event()
        published: list[object] = []

        def transcribe(_audio: np.ndarray, **_kwargs: object) -> object:
            decode_started.set()
            self.assertTrue(allow_decode.wait(timeout=2.0))
            return SimpleNamespace(
                text="stale opening words",
                info=SimpleNamespace(language="en", language_probability=1.0),
            )

        recorder.enable_realtime_transcription = True
        recorder.is_running = True
        recorder.awaiting_speech_end = False
        recorder.realtime_processing_pause = 0.001
        recorder.realtime_transcription_use_syllable_boundaries = False
        recorder.sample_rate = 10
        recorder.use_main_model_for_realtime = False
        recorder._uses_external_realtime_transcription_executor = True
        recorder.realtime_transcription_executor = transcribe
        recorder.realtime_transcription_model = None
        recorder.language = "en"
        recorder.realtime_transcription_count = 0
        recorder.realtime_transcription_success_count = 0
        recorder.realtime_transcription_empty_count = 0
        recorder.realtime_transcription_trigger_counts = {}
        recorder.detected_realtime_language = "unchanged"
        recorder.detected_realtime_language_probability = -1.0
        recorder.realtime_text_stabilization_event = "unchanged-event"
        recorder.on_realtime_text_stabilization_update = published.append

        worker = threading.Thread(target=realtime.run_realtime_worker, args=(recorder,))
        worker.start()
        self.assertTrue(decode_started.wait(timeout=2.0))

        with get_realtime_state_lock(recorder):
            with get_frames_lock(recorder):
                recorder.frames = [_pcm(9, 5)]
                recorder.realtime_recording_id += 1
                recorder.realtime_recording_generation += 1
            recorder.realtime_observation_sequence = 0
            recorder.is_running = False

        allow_decode.set()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(recorder.realtime_transcription_count, 0)
        self.assertEqual(recorder.realtime_transcription_success_count, 0)
        self.assertEqual(recorder.realtime_observation_sequence, 0)
        self.assertEqual(recorder.detected_realtime_language, "unchanged")
        self.assertEqual(recorder.detected_realtime_language_probability, -1.0)
        self.assertEqual(recorder.realtime_transcription_text, "old")
        self.assertEqual(recorder.text_storage, ["old"])
        self.assertEqual(recorder.realtime_stabilized_text, "old")
        self.assertEqual(recorder.realtime_text_stabilization_event, "unchanged-event")
        self.assertEqual(recorder.realtime_text_stabilizer.reset_recording_ids, [])
        self.assertEqual(published, [])

    def test_start_and_stop_each_advance_recording_generation(self) -> None:
        """Every lifecycle boundary must invalidate delayed split/decode work."""

        recorder = _Recorder([])
        recorder.state = "inactive"
        recorder.recording_stop_time = 0.0
        recorder.min_gap_between_recordings = 0.0
        recorder.min_length_of_recording = 0.0
        recorder._pending_preroll_selection = None
        recorder.last_preroll_selection = None
        recorder.wakeword_detected = False
        recorder.wake_word_detect_time = 0.0
        recorder.speech_end_silence_candidate_start = 0.0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.on_recording_start = None
        recorder.on_recording_stop = None
        recorder.is_webrtc_speech_active = False
        recorder.silero_check_time = 0.0
        initial_generation = recorder.realtime_recording_generation

        with (
            mock.patch.object(lifecycle, "set_recorder_state"),
            mock.patch.object(lifecycle, "reset_silero_vad_state"),
        ):
            lifecycle.start_recording(recorder, [_pcm(5, 4)])
            self.assertEqual(
                recorder.realtime_recording_generation,
                initial_generation + 1,
            )
            lifecycle.stop_recording(recorder)

        self.assertEqual(
            recorder.realtime_recording_generation,
            initial_generation + 2,
        )
        self.assertFalse(recorder.is_recording)
        self.assertEqual(recorder.realtime_text_stabilizer.finalize_count, 1)

    def test_final_consumer_cleanup_cannot_clear_new_recording_frames(self) -> None:
        """Finishing A must not clear B when B starts during A's queue cleanup."""

        recorder = _Recorder([])
        recorder.is_recording = False
        recorder.state = "inactive"
        recorder.spinner = False
        recorder.halo = None
        recorder.recording_stop_time = 0.0
        recorder.min_gap_between_recordings = 0.0
        recorder._pending_preroll_selection = None
        recorder.last_preroll_selection = None
        recorder.wakeword_detected = False
        recorder.wake_word_detect_time = 0.0
        recorder.speech_end_silence_candidate_start = 0.0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.interrupt_stop_event = threading.Event()
        recorder.on_recording_start = None
        recorder.on_vad_detect_start = None
        recorder.on_vad_detect_stop = None
        recorder.on_wakeword_detection_end = None
        recorder.start_callback_in_new_thread = False
        recorder.listen_start = 1.0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.stop_recording_on_voice_deactivity = True
        recorder.use_wake_words = False
        recorder.is_shut_down = False
        recorder.continuous_listening = True
        queue_recorded_audio(recorder, [_pcm(1, 5)])

        cleanup_waiting_for_frames = threading.Event()
        allow_cleanup = threading.Event()
        start_finished = threading.Event()
        original_get_frames_lock = get_frames_lock

        wait_thread = threading.Thread(target=lifecycle.wait_for_recorded_audio, args=(recorder,))
        new_frames = [_pcm(9, 7), _pcm(8, 3)]

        def blocked_get_frames_lock(target: object) -> object:
            if threading.current_thread() is wait_thread:
                cleanup_waiting_for_frames.set()
                self.assertTrue(allow_cleanup.wait(timeout=2.0))
            return original_get_frames_lock(target)

        def start_next_recording() -> None:
            lifecycle.start_recording(recorder, new_frames)
            start_finished.set()

        start_thread = threading.Thread(target=start_next_recording)
        with (
            mock.patch.object(
                lifecycle,
                "set_audio_from_frames",
                return_value=[],
            ),
            mock.patch.object(
                lifecycle,
                "get_frames_lock",
                side_effect=blocked_get_frames_lock,
            ),
            mock.patch.object(lifecycle, "reset_silero_vad_state"),
        ):
            wait_thread.start()
            self.assertTrue(cleanup_waiting_for_frames.wait(timeout=2.0))
            start_thread.start()
            self.assertFalse(start_finished.wait(timeout=0.05))
            allow_cleanup.set()
            wait_thread.join(timeout=2.0)
            start_thread.join(timeout=2.0)

        self.assertFalse(wait_thread.is_alive())
        self.assertFalse(start_thread.is_alive())
        self.assertTrue(start_finished.is_set())
        self.assertTrue(recorder.is_recording)
        self.assertEqual(recorder.state, "recording")
        self.assertEqual(recorder.frames, new_frames)

    def test_wait_arming_cannot_overwrite_new_recording_state(self) -> None:
        """The idle check and VAD arming must not race a new recording start."""

        recorder = _Recorder([])
        recorder.is_recording = False
        recorder.state = "inactive"
        recorder.spinner = False
        recorder.halo = None
        recorder.recording_stop_time = 0.0
        recorder.min_gap_between_recordings = 0.0
        recorder._pending_preroll_selection = None
        recorder.last_preroll_selection = None
        recorder.wakeword_detected = False
        recorder.wake_word_detect_time = 0.0
        recorder.speech_end_silence_candidate_start = 0.0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.interrupt_stop_event = threading.Event()
        recorder.on_recording_start = None
        recorder.on_vad_detect_start = None
        recorder.on_vad_detect_stop = None
        recorder.on_wakeword_detection_end = None
        recorder.start_callback_in_new_thread = False
        recorder.listen_start = 0.0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.stop_recording_on_voice_deactivity = True
        recorder.start_recording_on_voice_activity = False
        recorder.use_wake_words = False
        recorder.is_shut_down = False
        recorder.continuous_listening = True

        idle_snapshot_captured = threading.Event()
        allow_idle_check = threading.Event()
        start_finished = threading.Event()
        block_first_snapshot = True
        wait_thread = threading.Thread(target=lifecycle.wait_for_recorded_audio, args=(recorder,))
        new_frames = [_pcm(7, 6), _pcm(8, 4)]

        def blocked_snapshot(target: object, attr_name: str = "frames") -> tuple[bytes, ...]:
            nonlocal block_first_snapshot
            result = snapshot_frames(target, attr_name)
            if threading.current_thread() is wait_thread and block_first_snapshot:
                block_first_snapshot = False
                idle_snapshot_captured.set()
                self.assertTrue(allow_idle_check.wait(timeout=2.0))
            return result

        def start_next_recording() -> None:
            lifecycle.start_recording(recorder, new_frames)
            recorder.stop_recording_event.set()
            start_finished.set()

        start_thread = threading.Thread(target=start_next_recording)
        with (
            mock.patch.object(lifecycle, "snapshot_frames", side_effect=blocked_snapshot),
            mock.patch.object(lifecycle, "set_audio_from_frames", return_value=[]),
            mock.patch.object(lifecycle, "reset_silero_vad_state"),
        ):
            wait_thread.start()
            self.assertTrue(idle_snapshot_captured.wait(timeout=2.0))
            start_thread.start()
            started_during_idle_check = start_finished.wait(timeout=0.05)
            allow_idle_check.set()
            wait_thread.join(timeout=2.0)
            start_thread.join(timeout=2.0)

        self.assertFalse(started_during_idle_check)
        self.assertFalse(wait_thread.is_alive())
        self.assertFalse(start_thread.is_alive())
        self.assertTrue(recorder.is_recording)
        self.assertEqual(recorder.state, "recording")
        self.assertEqual(recorder.frames, new_frames)


if __name__ == "__main__":
    unittest.main()
