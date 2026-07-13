"""Immutable inputs captured before report work enters a background queue."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from window.meeting_intelligence_pipeline import MeetingLLMConfig, StructuredChatClient


@dataclass(frozen=True)
class ReportGenerationRequest:
    session_id: str
    template_id: str
    title: str
    transcript_revision_id: str
    report_language: str
    transcript_rows_json: str
    speaker_state_json: str
    summary_json: str
    template_json: str
    llm_config: MeetingLLMConfig
    mock_llm: bool
    max_segment_rows: int
    client_factory: Callable[[], StructuredChatClient] | None = None

    @staticmethod
    def _load_object(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _load_array(payload: str) -> list[dict[str, Any]]:
        value = json.loads(payload)
        return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def transcript_rows(self) -> list[dict[str, Any]]:
        return self._load_array(self.transcript_rows_json)

    def speaker_state(self) -> dict[str, Any]:
        return self._load_object(self.speaker_state_json)

    def summary(self) -> dict[str, Any]:
        return self._load_object(self.summary_json)

    def template(self) -> dict[str, Any]:
        return self._load_object(self.template_json)


def immutable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["ReportGenerationRequest", "immutable_json"]
