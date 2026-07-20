"""Serialize native streaming-ASR timing data for the JSON-lines workers."""

from __future__ import annotations

import json
import math
from typing import Any


def _float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _native_json(result: object) -> dict[str, Any]:
    serializer = getattr(result, "as_json_string", None)
    if not callable(serializer):
        return {}
    try:
        value = json.loads(serializer())
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _word_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = next(
            (str(item[key]) for key in ("text", "word", "value", "token") if item.get(key) is not None),
            "",
        ).strip()
        start = next(
            (_float(item.get(key)) for key in ("start", "startedAt", "startTime", "timestamp") if key in item),
            None,
        )
        end = next(
            (_float(item.get(key)) for key in ("end", "endedAt", "endTime") if key in item),
            None,
        )
        if text and start is not None:
            normalized: dict[str, object] = {"text": text, "start": start}
            if end is not None:
                normalized["end"] = end
            output.append(normalized)
    return output


def structured_result_payload(result: object, fallback_text: str = "") -> dict[str, object]:
    """Return only stable JSON-compatible text, tokens, timestamps, and words."""

    native_json = _native_json(result)
    text = str(getattr(result, "text", "") or native_json.get("text") or fallback_text or "").strip()
    raw_tokens = getattr(result, "tokens", None)
    if raw_tokens is None:
        raw_tokens = native_json.get("tokens")
    raw_timestamps = getattr(result, "timestamps", None)
    if raw_timestamps is None:
        raw_timestamps = native_json.get("timestamps")
    tokens = [str(item) for item in raw_tokens] if isinstance(raw_tokens, (list, tuple)) else []
    timestamps = [item for item in (_float(value) for value in raw_timestamps or []) if item is not None]

    elements = native_json.get("elements")
    native_words = elements.get("words") if isinstance(elements, dict) else native_json.get("words")
    words = _word_items(native_words)
    return {
        "text": text,
        "tokens": tokens,
        "timestamps": timestamps,
        "words": words,
    }
