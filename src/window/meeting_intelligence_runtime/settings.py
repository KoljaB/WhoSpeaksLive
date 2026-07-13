"""Atomic ownership of runtime LLM settings."""

from __future__ import annotations

import threading
from window.meeting_intelligence_pipeline import MeetingLLMConfig


class MeetingSettingsStore:
    def __init__(self, initial: MeetingLLMConfig) -> None:
        self._lock = threading.Lock()
        self._value = initial

    def snapshot(self) -> MeetingLLMConfig:
        with self._lock:
            return self._value

    def replace(self, value: MeetingLLMConfig) -> MeetingLLMConfig:
        if not isinstance(value, MeetingLLMConfig):
            raise TypeError("meeting settings must be MeetingLLMConfig")
        with self._lock:
            self._value = value
            return value


__all__ = ["MeetingSettingsStore"]
