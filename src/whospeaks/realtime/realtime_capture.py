"""Realtime YouTube capture controller and event plumbing."""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from whospeaks.common.audio_utils import SAMPLE_RATE, audio_to_float_mono, json_dumps
from whospeaks.realtime.realtime_speaker_engine import RealtimeSpeakerEngine
from whospeaks.realtime.realtime_transcript import split_transcript_by_timestamps

def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id

    path_match = re.search(r"/(?:embed|shorts|live)/([^/?#]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    raise ValueError("Could not extract a YouTube video id from the URL.")


class TraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        source: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "time": time.time(),
            "source": source,
            "event": event,
            "payload": payload or {},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


@dataclass
class VideoClock:
    current_time_seconds: float
    monotonic_seconds: float


def list_audio_input_devices() -> list[dict[str, Any]]:
    try:
        import pyaudiowpatch as pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyaudiowpatch is required for WASAPI loopback capture."
        ) from exc

    audio = pyaudio.PyAudio()
    devices: list[dict[str, Any]] = []
    try:
        host_api_names = {}
        for host_index in range(audio.get_host_api_count()):
            try:
                host_info = audio.get_host_api_info_by_index(host_index)
                host_api_names[host_index] = host_info.get("name", "")
            except Exception:
                host_api_names[host_index] = ""

        for index in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(index)
            except Exception:
                continue
            max_input_channels = int(info.get("maxInputChannels") or 0)
            if max_input_channels <= 0:
                continue
            host_api_index = int(info.get("hostApi") or 0)
            devices.append({
                "index": int(info.get("index", index)),
                "name": str(info.get("name", "")),
                "host_api": host_api_names.get(host_api_index, ""),
                "channels": max_input_channels,
                "sample_rate": int(float(info.get("defaultSampleRate") or 0)),
                "is_loopback": bool(info.get("isLoopbackDevice")),
            })
    finally:
        audio.terminate()

    return devices


def choose_wasapi_loopback_device(
    requested_index: int | None,
    allow_default_input: bool,
) -> tuple[int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    devices = list_audio_input_devices()
    if requested_index is not None:
        for device in devices:
            if device["index"] == requested_index:
                return requested_index, device, devices
        raise RuntimeError(f"Input device index {requested_index} is not available.")

    try:
        import pyaudiowpatch as pyaudio

        audio = pyaudio.PyAudio()
        try:
            default_loopback = audio.get_default_wasapi_loopback()
        finally:
            audio.terminate()
        if default_loopback and default_loopback.get("isLoopbackDevice"):
            index = int(default_loopback["index"])
            for device in devices:
                if device["index"] == index:
                    return index, device, devices
            return index, {
                "index": index,
                "name": str(default_loopback.get("name", "")),
                "host_api": "Windows WASAPI",
                "channels": int(default_loopback.get("maxInputChannels") or 2),
                "sample_rate": int(float(default_loopback.get("defaultSampleRate") or 48000)),
                "is_loopback": True,
            }, devices
    except Exception:
        pass

    loopbacks = [device for device in devices if device.get("is_loopback")]
    if loopbacks:
        return loopbacks[0]["index"], loopbacks[0], devices

    if allow_default_input:
        return None, None, devices

    device_lines = [
        f"{item['index']}: {item['name']} [{item['host_api']}, "
        f"{item['channels']}ch, {item['sample_rate']} Hz"
        f"{', loopback' if item.get('is_loopback') else ''}]"
        for item in devices
    ]
    raise RuntimeError(
        "No WASAPI loopback-like input device was detected. "
        "Rerun with --input-device-index N after choosing one of:\n"
        + "\n".join(device_lines)
    )


class EventBus:
    def __init__(self, trace: TraceLogger | None = None) -> None:
        self._subscribers: set[queue.Queue[tuple[str, str]]] = set()
        self._lock = threading.Lock()
        self._trace = trace

    def subscribe(self) -> queue.Queue[tuple[str, str]]:
        subscriber: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=300)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[tuple[str, str]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.write("backend_sse", event, payload)
        message = json_dumps(payload)
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait((event, message))
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait((event, message))
                except queue.Full:
                    pass


class YouTubeWasapiController:
    def __init__(self, args: argparse.Namespace, bus: EventBus) -> None:
        self.args = args
        self.bus = bus
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._recorder: Any = None
        self._audio_interface: Any = None
        self._audio_stream: Any = None
        self._final_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stopping = False
        self._video_ids: dict[str, str] = {}
        self._video_clocks: dict[str, VideoClock] = {}
        self._sentence_indices: dict[str, int] = {}
        self.speaker_engine = RealtimeSpeakerEngine(args, bus)

    def start(self, url: str) -> tuple[str, str]:
        video_id = extract_youtube_video_id(url)
        self.stop(emit=False)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._session_id = session_id
            self._stop_event = threading.Event()
            self._stopping = False
            self._video_ids[session_id] = video_id
            self._video_clocks.pop(session_id, None)
            self._sentence_indices[session_id] = 0
        self.speaker_engine.start_session(session_id)
        self._status(session_id, "Started.")
        worker = threading.Thread(
            target=self._start_capture,
            args=(session_id,),
            name="YouTubeWasapiCaptureStarter",
            daemon=True,
        )
        worker.start()
        return session_id, video_id

    def stop(self, emit: bool = True) -> None:
        with self._lock:
            if emit and self._stopping:
                session_id = self._session_id
                if session_id:
                    self._status(session_id, "Stop already in progress.")
                return
            if emit:
                self._stopping = True
            session_id = self._session_id
            recorder = self._recorder
            stream = self._audio_stream
            audio_interface = self._audio_interface
            stop_event = self._stop_event
            final_thread = self._final_thread

        if emit and session_id:
            self._status(session_id, "Stop requested. Draining final transcripts.")
        if emit and session_id and recorder is not None and not stop_event.is_set():
            self._drain_recorder_before_stop(session_id, recorder)
        stop_event.set()
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
            except Exception:
                pass
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass
        if final_thread is not None and final_thread is not threading.current_thread():
            try:
                final_thread.join(timeout=2.0)
            except Exception:
                pass
        if emit and session_id:
            self._drain_embedding_jobs_after_stop(session_id)
            self._status(session_id, "Stopped.")
        if emit:
            with self._lock:
                self._stopping = False

    def shutdown(self) -> None:
        self.stop(emit=False)
        self.speaker_engine.shutdown()

    def _drain_recorder_before_stop(self, session_id: str, recorder: Any) -> None:
        silence_seconds = max(0.0, float(self.args.stop_trailing_silence_seconds))
        drain_seconds = max(0.0, float(self.args.stop_drain_seconds))
        if silence_seconds > 0.0:
            silence = np.zeros(int(SAMPLE_RATE * silence_seconds), dtype=np.int16)
            chunk_samples = max(1, SAMPLE_RATE // 10)
            for start in range(0, len(silence), chunk_samples):
                end = min(len(silence), start + chunk_samples)
                try:
                    recorder.feed_audio(silence[start:end], original_sample_rate=SAMPLE_RATE)
                except Exception:
                    break
        if drain_seconds > 0.0:
            self._status(session_id, f"Waiting {drain_seconds:.1f}s for final transcription drain.")
            time.sleep(drain_seconds)

    def _drain_embedding_jobs_after_stop(self, session_id: str) -> None:
        deadline = time.monotonic() + max(0.0, float(self.args.stop_embedding_drain_seconds))
        while (
            getattr(self.speaker_engine.jobs, "unfinished_tasks", 0) > 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)

    @staticmethod
    def _coerce_video_time(value: Any) -> float | None:
        try:
            current_time = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(current_time) or current_time < 0:
            return None
        return current_time

    def _set_video_time(self, session_id: str | None, current_time: Any) -> float | None:
        if not session_id:
            return None
        video_time = self._coerce_video_time(current_time)
        if video_time is None:
            return None
        with self._lock:
            if session_id != self._session_id:
                return None
            self._video_clocks[session_id] = VideoClock(
                current_time_seconds=video_time,
                monotonic_seconds=time.monotonic(),
            )
        return video_time

    def _estimate_video_time(self, session_id: str, at_monotonic: float | None = None) -> float | None:
        with self._lock:
            clock = self._video_clocks.get(session_id)
        if clock is None:
            return None
        if at_monotonic is None:
            at_monotonic = time.monotonic()
        elapsed = max(0.0, float(at_monotonic) - clock.monotonic_seconds)
        return max(0.0, clock.current_time_seconds + elapsed)

    def mark_video_playing(self, session_id: str | None, current_time: Any) -> None:
        video_time = self._set_video_time(session_id, current_time)
        if not session_id:
            return
        if video_time is None:
            self._status(session_id, "YouTube playback confirmed.")
        else:
            self._status(session_id, f"YouTube playback confirmed at {video_time:.2f}s.")

    def update_video_time(self, session_id: str | None, current_time: Any) -> None:
        self._set_video_time(session_id, current_time)

    def _is_current(self, session_id: str) -> bool:
        with self._lock:
            return session_id == self._session_id

    def _next_sentence_index(self, session_id: str) -> int:
        with self._lock:
            index = self._sentence_indices.get(session_id, 0)
            self._sentence_indices[session_id] = index + 1
            return index

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit("error-status", {"session_id": session_id, "message": message})

    def _start_capture(self, session_id: str) -> None:
        try:
            self._status(session_id, "Selecting WASAPI loopback input.")
            input_device_index, device, devices = choose_wasapi_loopback_device(
                self.args.input_device_index,
                self.args.allow_default_input,
            )
            if device is not None:
                self._status(
                    session_id,
                    "Using input device "
                    f"{device['index']}: {device['name']} [{device['host_api']}].",
                )
            else:
                self._status(session_id, "Using the default input device.")
            if devices:
                self._status(
                    session_id,
                    "Available input devices: "
                    + "; ".join(
                        f"{item['index']}={item['name']} [{item['host_api']}]"
                        for item in devices
                    ),
                )

            recorder = self._create_recorder(session_id, input_device_index)
            audio_interface, stream = self._open_loopback_stream(
                session_id,
                input_device_index,
                recorder,
                device,
            )
            with self._lock:
                if session_id != self._session_id:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    try:
                        audio_interface.terminate()
                    except Exception:
                        pass
                    recorder.shutdown()
                    return
                self._recorder = recorder
                self._audio_interface = audio_interface
                self._audio_stream = stream

            recorder.start()
            self._status(session_id, "WASAPI capture started.")
            final_thread = threading.Thread(
                target=self._consume_final_text,
                args=(session_id, recorder, self._stop_event),
                name="YouTubeWasapiFinalConsumer",
                daemon=True,
            )
            with self._lock:
                self._final_thread = final_thread
            final_thread.start()
            video_id = self._video_ids.get(session_id)
            if video_id:
                self.bus.emit(
                    "capture-ready",
                    {"session_id": session_id, "video_id": video_id},
                )
        except Exception as exc:
            if self._is_current(session_id):
                self._error(session_id, str(exc))

    def _open_loopback_stream(
        self,
        session_id: str,
        input_device_index: int | None,
        recorder: Any,
        device: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        try:
            import pyaudiowpatch as pyaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyaudiowpatch and numpy are required for WASAPI loopback capture."
            ) from exc

        audio_interface = pyaudio.PyAudio()
        try:
            if input_device_index is None:
                device_info = audio_interface.get_default_wasapi_loopback()
                input_device_index = int(device_info["index"])
            else:
                device_info = audio_interface.get_device_info_by_index(input_device_index)

            if not device_info.get("isLoopbackDevice"):
                raise RuntimeError(f"Device {input_device_index} is not a WASAPI loopback device.")

            sample_rate = int(device_info["defaultSampleRate"])
            channels = int(device_info["maxInputChannels"])
            if channels <= 0:
                raise RuntimeError(f"Loopback device {input_device_index} has no input channels.")

            display_name = (
                device["name"]
                if device is not None
                else str(device_info.get("name", input_device_index))
            )
            self._status(
                session_id,
                f"Opening loopback stream: {display_name}, {channels}ch, {sample_rate} Hz.",
            )

            def feed(data: bytes, _frame_count: int, _time_info: Any, _status: Any):
                audio = np.frombuffer(data, np.int16)
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                recorder.feed_audio(audio, original_sample_rate=sample_rate)
                return (None, pyaudio.paContinue)

            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=max(1, sample_rate // 10),
                stream_callback=feed,
            )
            stream.start_stream()
            return audio_interface, stream
        except Exception:
            try:
                audio_interface.terminate()
            except Exception:
                pass
            raise

    def _create_recorder(self, session_id: str, input_device_index: int | None) -> Any:
        self._status(session_id, "Initializing RealtimeSTT.")

        if os.name == "nt":
            try:
                from torchaudio._extension.utils import _init_dll_path

                _init_dll_path()
            except Exception:
                pass

        from RealtimeSTT import AudioToTextRecorder

        def stabilization_update(event: Any) -> None:
            display_text = getattr(event, "display_text", None)
            if display_text is None:
                display_text = getattr(event, "raw_observation_text", "")
            stable_text = getattr(event, "stable_text", "") or ""
            unstable_text = getattr(event, "unstable_text", "") or ""
            self.bus.emit(
                "realtime",
                {
                    "session_id": session_id,
                    "display_text": display_text or "",
                    "stable_text": stable_text,
                    "unstable_text": unstable_text,
                },
            )

        recorder = AudioToTextRecorder(
            use_microphone=False,
            input_device_index=None,
            spinner=False,
            model=self.args.model,
            realtime_model_type=self.args.rt_model,
            language=self.args.language,
            device=self.args.device,
            compute_type=self.args.compute_type,
            download_root=self.args.download_root,
            enable_realtime_transcription=True,
            realtime_punctuation_split_marks=self.args.split_marks,
            realtime_processing_pause=self.args.realtime_processing_pause,
            init_realtime_after_seconds=0.0,
            on_realtime_transcription_update=None,
            on_realtime_text_stabilization_update=stabilization_update,
            batch_size=self.args.batch_size,
            realtime_batch_size=self.args.realtime_batch_size,
            min_length_of_recording=self.args.min_length_of_recording,
            min_gap_between_recordings=0,
            post_speech_silence_duration=self.args.post_speech_silence_duration,
            silero_sensitivity=self.args.silero_sensitivity,
            webrtc_sensitivity=self.args.webrtc_sensitivity,
            silero_deactivity_detection=True,
            realtime_transcription_use_syllable_boundaries=True,
            realtime_boundary_detector_sensitivity=0.6,
            realtime_boundary_followup_delays=(0.05, 0.2),
            beam_size=self.args.beam_size,
            beam_size_realtime=self.args.beam_size_realtime,
            no_log_file=True,
            faster_whisper_vad_filter=False,
            final_transcription_word_timestamps=self.args.final_word_timestamps,
        )
        self._status(session_id, "RealtimeSTT ready.")
        return recorder

    def _read_final_text(self, recorder: Any) -> str:
        if self.args.final_word_timestamps:
            try:
                return (recorder.text(word_timestamps=True) or "").strip()
            except TypeError:
                self._error(
                    self._session_id,
                    "RealtimeSTT does not expose text(word_timestamps=...). "
                    "Falling back to unsplit final transcripts.",
                )
        return (recorder.text() or "").strip()

    def _consume_final_text(
        self,
        session_id: str,
        recorder: Any,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                text = self._read_final_text(recorder)
            except Exception as exc:
                if not stop_event.is_set():
                    self._error(session_id, f"Final transcription failed: {exc}")
                return
            if not text:
                continue

            audio = audio_to_float_mono(getattr(recorder, "last_transcription_bytes", None))
            if len(audio) == 0:
                self._error(session_id, "No audio samples available for final transcript.")
                continue
            metadata = getattr(recorder, "last_transcription_metadata", None) or {}
            parts = split_transcript_by_timestamps(
                text=text,
                audio=audio,
                sample_rate=SAMPLE_RATE,
                metadata=metadata,
                args=self.args,
            )
            if len(parts) > 1:
                self._status(
                    session_id,
                    f"RealtimeSTT final split into {len(parts)} sentence parts.",
                )

            final_audio_duration = float(len(audio)) / float(SAMPLE_RATE)
            video_current = self._estimate_video_time(session_id)
            video_chunk_end = None
            if video_current is not None:
                video_chunk_end = max(
                    0.0,
                    video_current - max(0.0, float(self.args.final_video_latency_seconds)),
                )

            for part in parts:
                index = self._next_sentence_index(session_id)
                video_start_seconds = None
                video_end_seconds = None
                if video_chunk_end is not None:
                    video_chunk_start = max(0.0, video_chunk_end - final_audio_duration)
                    video_start_seconds = max(0.0, video_chunk_start + part.start_seconds)
                    video_end_seconds = max(video_start_seconds, video_chunk_start + part.end_seconds)
                self.bus.emit(
                    "final",
                    {
                        "session_id": session_id,
                        "index": index,
                        "text": part.text,
                        "source_start_seconds": part.start_seconds,
                        "source_end_seconds": part.end_seconds,
                        "word_start_seconds": part.word_start_seconds,
                        "word_end_seconds": part.word_end_seconds,
                        "split_reason": part.split_reason,
                        "part_index": part.part_index,
                        "part_count": part.part_count,
                        "video_start_seconds": (
                            None
                            if video_start_seconds is None
                            else round(float(video_start_seconds), 4)
                        ),
                        "video_end_seconds": (
                            None
                            if video_end_seconds is None else round(float(video_end_seconds), 4)
                        ),
                    },
                )
                self.speaker_engine.submit(
                    session_id=session_id,
                    index=index,
                    text=part.text,
                    audio=part.audio,
                    sample_rate=SAMPLE_RATE,
                    source_start_seconds=part.start_seconds,
                    source_end_seconds=part.end_seconds,
                    split_reason=part.split_reason,
                    part_index=part.part_index,
                    part_count=part.part_count,
                    video_start_seconds=video_start_seconds,
                    video_end_seconds=video_end_seconds,
                )


