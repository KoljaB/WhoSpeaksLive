"""Causal profile-snapshot events captured at the moment live code can use them."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


PROFILE_TAPE_EVENT_ID = "live_speaker_profile_snapshot_v1"


def emit_live_profile_snapshot(
    owner: Any,
    memory: Any,
    speaker_id: str | None,
    provider: str,
    *,
    source: str,
    sentence_start: float | None = None,
    sentence_end: float | None = None,
) -> dict[str, Any] | None:
    """Emit a complete snapshot once its memory mutation is actually visible."""

    label = str(speaker_id or "").strip()
    provider = str(provider or "").strip()
    if not label or not provider:
        return None
    profiles = {
        str(item.get("label") or ""): item
        for item in memory.export_profiles()
        if isinstance(item, dict)
    }
    profile = profiles.get(label)
    if profile is None:
        return None
    centroid = np.asarray(profile["centroid"], dtype=np.float32).reshape(-1)
    fingerprint = hashlib.sha256()
    fingerprint.update(provider.encode("utf-8"))
    fingerprint.update(label.encode("utf-8"))
    fingerprint.update(np.ascontiguousarray(centroid).tobytes())
    fingerprint.update(str(int(profile.get("sentence_count") or 1)).encode("ascii"))
    fingerprint.update(repr(float(profile.get("speech_seconds") or 0.0)).encode("ascii"))
    key = f"{provider}\0{label}"
    fingerprints = getattr(owner, "_live_profile_snapshot_fingerprints", None)
    if not isinstance(fingerprints, dict):
        fingerprints = {}
        setattr(owner, "_live_profile_snapshot_fingerprints", fingerprints)
    digest = fingerprint.hexdigest()
    if fingerprints.get(key) == digest:
        return None
    fingerprints[key] = digest
    generations = getattr(owner, "_live_profile_snapshot_generations", None)
    if not isinstance(generations, dict):
        generations = {}
        setattr(owner, "_live_profile_snapshot_generations", generations)
    generation = int(generations.get(key, 0)) + 1
    generations[key] = generation
    payload = {
        "event_id": PROFILE_TAPE_EVENT_ID,
        "available_at": max(0.0, float(owner.playback_time())),
        "sentence_start": None if sentence_start is None else float(sentence_start),
        "sentence_end": None if sentence_end is None else float(sentence_end),
        "speaker_id": label,
        "profile_embedding_provider": provider,
        "profile_generation": generation,
        "sentence_count": max(1, int(profile.get("sentence_count") or 1)),
        "speech_seconds": max(0.0, float(profile.get("speech_seconds") or 0.0)),
        "centroid": centroid.astype(float).tolist(),
        "source": str(source),
    }
    owner.bus.emit("live_speaker_profile_snapshot", payload)
    return payload
