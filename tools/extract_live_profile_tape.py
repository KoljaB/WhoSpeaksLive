from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


EVENT_NAME = "live_speaker_profile_snapshot"
EVENT_ID = "live_speaker_profile_snapshot_v1"


def extract_profile_rows(trace_path: Path, expected_provider: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_available_at = float("-inf")
    for line_number, raw in enumerate(Path(trace_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        envelope = json.loads(raw)
        if envelope.get("event") == EVENT_NAME:
            value = envelope.get("payload")
        elif envelope.get("event_id") == EVENT_ID:
            value = envelope
        else:
            continue
        if not isinstance(value, dict) or value.get("event_id") != EVENT_ID:
            raise ValueError(f"Malformed profile snapshot at {trace_path}:{line_number}")
        provider = str(value.get("profile_embedding_provider") or "").strip()
        if expected_provider and provider != expected_provider:
            raise ValueError(
                f"Profile provider mismatch at {trace_path}:{line_number}: "
                f"expected {expected_provider!r}, got {provider!r}"
            )
        available_at = float(value["available_at"])
        if available_at + 1e-9 < last_available_at:
            raise ValueError(f"Non-chronological profile snapshot at {trace_path}:{line_number}")
        last_available_at = available_at
        centroid = value.get("centroid")
        if not isinstance(centroid, list) or not centroid:
            raise ValueError(f"Empty profile centroid at {trace_path}:{line_number}")
        rows.append({
            "available_at": available_at,
            "speaker_id": str(value["speaker_id"]),
            "centroid": centroid,
            "speech_seconds": float(value.get("speech_seconds") or 0.0),
            "sentence_count": int(value.get("sentence_count") or 1),
            "profile_generation": int(value.get("profile_generation") or 0),
            "profile_embedding_provider": provider,
            "source": str(value.get("source") or ""),
            "sentence_start": value.get("sentence_start"),
            "sentence_end": value.get("sentence_end"),
        })
    if not rows:
        raise ValueError(f"No {EVENT_NAME!r} events found in {trace_path}")
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract causal profile snapshots from a production event trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-provider")
    args = parser.parse_args()
    rows = extract_profile_rows(args.trace.resolve(), args.expected_provider)
    write_jsonl_atomic(args.output.resolve(), rows)
    print(json.dumps({"status": "complete", "event_count": len(rows), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
