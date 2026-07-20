"""Pure causal speech gates shared by production and offline replay inputs."""

from __future__ import annotations

import numpy as np


RMS_GATE_ID = "live_rms_speech_gate_v1"


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
