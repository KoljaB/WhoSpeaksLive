"""Pure parsing, environment, and provider helpers for the report server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
LLM_PROVIDER_OPTIONS = {
    "llama_cpp": {"label": "llama.cpp", "default_base_url": "http://127.0.0.1:8081/v1", "models": ["gemma-4-12b-it-Q6_K.gguf", "local"], "requires_api_key": False, "api_key_env_var": ""},
    "ollama": {"label": "Ollama", "default_base_url": "http://127.0.0.1:11434/v1", "models": ["gemma3", "llama3.1"], "requires_api_key": False, "api_key_env_var": ""},
    "lm_studio": {"label": "LM Studio", "default_base_url": "http://127.0.0.1:1234/v1", "models": ["local-model"], "requires_api_key": False, "api_key_env_var": ""},
    "openai_compatible": {"label": "OpenAI-compatible", "default_base_url": "http://127.0.0.1:8000/v1", "models": ["local-model"], "requires_api_key": False, "api_key_env_var": ""},
    "openai": {"label": "OpenAI", "default_base_url": "https://api.openai.com/v1", "models": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"], "requires_api_key": True, "api_key_env_var": "OPENAI_API_KEY"},
    "openrouter": {"label": "OpenRouter", "default_base_url": "https://openrouter.ai/api/v1", "models": ["google/gemma-3-12b-it"], "requires_api_key": True, "api_key_env_var": "OPENROUTER_API_KEY"},
}
TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?::\d+){1,2}(?:\.\d+)?)\s+-\s+"
    r"(?P<end>\d+(?::\d+){1,2}(?:\.\d+)?)\]\s+"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


def parse_whospeakslive_transcript(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TRANSCRIPT_LINE_RE.match(line.strip())
        if not match:
            continue
        speaker_name = " ".join(match.group("speaker").split())
        text = " ".join(match.group("text").split())
        if not text:
            continue
        index = len(rows) + 1
        rows.append({
            "index": index, "row_id": f"demo_row_{index:04d}",
            "start": parse_timecode(match.group("start")), "end": parse_timecode(match.group("end")),
            "text": text, "assigned_speaker": speaker_id_from_name(speaker_name),
            "speaker_name": speaker_name, "source_line": line_number,
        })
    return rows


def parse_timecode(value: str) -> float:
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return round(minutes * 60.0 + seconds, 4)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return round(hours * 3600.0 + minutes * 60.0 + seconds, 4)
    raise ValueError(f"Unsupported transcript timecode: {value}")


def speaker_id_from_name(name: str) -> str:
    match = re.search(r"(\d+)$", str(name or "").strip())
    if match:
        return f"S{int(match.group(1))}"
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "speaker").strip()).strip("_")
    return clean[:40] or "speaker"


def normalize_provider(value: Any) -> str:
    provider = str(value or "llama_cpp").strip().lower().replace("-", "_")
    if provider not in LLM_PROVIDER_OPTIONS:
        raise ValueError(f"Unsupported meeting LLM provider: {value}")
    return provider


def provider_options_payload() -> list[dict[str, Any]]:
    return [{
        "id": provider, "label": str(option["label"]),
        "default_base_url": str(option["default_base_url"]), "models": list(option["models"]),
        "requires_api_key": bool(option["requires_api_key"]), "api_key_env_var": str(option["api_key_env_var"]),
    } for provider, option in LLM_PROVIDER_OPTIONS.items()]


def extract_model_ids(payload: dict[str, Any]) -> list[str]:
    items = payload.get("data") if isinstance(payload, dict) else []
    raw_ids = [str(item.get("id") or "").strip() if isinstance(item, dict) else str(item or "").strip() for item in items] if isinstance(items, list) else []
    return sort_model_ids([model_id for model_id in unique_strings(raw_ids) if is_likely_text_generation_model(model_id)])


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def is_likely_text_generation_model(model_id: str) -> bool:
    text = model_id.lower()
    excluded = ("audio", "dall", "embedding", "image", "moderation", "realtime", "sora", "speech", "transcribe", "tts", "whisper")
    return bool(text) and not any(part in text for part in excluded) and text.startswith(("gpt-", "o1", "o3", "o4"))


def sort_model_ids(model_ids: list[str]) -> list[str]:
    def sort_key(model_id: str) -> tuple[int, str]:
        text = model_id.lower()
        for rank, marker in enumerate(("nano", "mini", "luna", "terra", "sol")):
            if marker in text:
                return rank, text
        return 5, text
    return sorted(model_ids, key=sort_key)


def load_env_file(path: Path | None = None) -> bool:
    env_path = (path or DEFAULT_ENV_FILE).expanduser()
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, strip_env_value(value.strip()))
    return True


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip() if " #" in value else value


__all__ = [name for name in globals() if not name.startswith("_") and name not in {"json", "os", "re"}]
