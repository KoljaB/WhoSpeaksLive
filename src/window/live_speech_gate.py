"""Pure causal speech gates shared by production and offline replay inputs."""

from __future__ import annotations

import numpy as np


RMS_GATE_ID = "live_rms_speech_gate_v1"


def live_silero_gate_parameters(
    args: object,
    *,
    release: bool = False,
    fast_release: bool = False,
) -> tuple[float, float]:
    """Resolve live-speaker Silero hysteresis without changing sentence VAD."""

    threshold = float(getattr(args, "live_speaker_probe_silero_speech_threshold", -1.0))
    if threshold < 0.0:
        threshold = float(getattr(args, "vad_silero_speech_threshold", 0.5))
    minimum = float(getattr(args, "live_speaker_probe_vad_min_speech_seconds", -1.0))
    if minimum < 0.0:
        minimum = float(getattr(args, "vad_min_speech_seconds", 0.25))
    if release:
        release_threshold = float(
            getattr(args, "live_speaker_probe_release_silero_speech_threshold", -1.0)
        )
        if release_threshold >= 0.0:
            threshold = release_threshold
        release_minimum = float(
            getattr(args, "live_speaker_probe_release_vad_min_speech_seconds", -1.0)
        )
        if release_minimum >= 0.0:
            minimum = release_minimum
    if fast_release:
        fast_threshold = float(
            getattr(args, "live_speaker_probe_fast_release_silero_speech_threshold", -1.0)
        )
        if fast_threshold >= 0.0:
            threshold = fast_threshold
        fast_minimum = float(
            getattr(args, "live_speaker_probe_fast_release_vad_min_speech_seconds", -1.0)
        )
        if fast_minimum >= 0.0:
            minimum = fast_minimum
    return max(0.0, min(1.0, threshold)), max(0.0, minimum)


def rms_speech_present(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float,
    threshold: float,
    min_speech_seconds: float,
) -> bool:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size <= 0 or sample_rate <= 0:
        return False
    frame_seconds = max(0.01, float(frame_seconds))
    frame_samples = max(1, int(sample_rate * frame_seconds))
    threshold = max(0.0, float(threshold))
    minimum = max(0.0, float(min_speech_seconds))
    speech_seconds = 0.0
    for start in range(0, values.size, frame_samples):
        end = min(values.size, start + frame_samples)
        if end - start < max(1, frame_samples // 2):
            break
        frame = values[start:end]
        rms_value = float(np.sqrt(np.mean(frame * frame)))
        if rms_value >= threshold:
            speech_seconds += (end - start) / float(sample_rate)
            if speech_seconds >= minimum:
                return True
    return False
