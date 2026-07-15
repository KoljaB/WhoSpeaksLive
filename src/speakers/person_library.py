"""Versioned storage and conservative matching for Person-owned Voice samples."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from speakers.speaker_embedding_cluster import cosine_similarity, normalize_vector


PEOPLE_FORMAT = "whospeaks-people"
PEOPLE_VERSION = 2
MAX_LEARNED_SAMPLES_PER_PROVIDER = 8
NEAR_DUPLICATE_SIMILARITY = 0.995
CORROBORATION_SUPPORT_THRESHOLD = 0.62
MAX_CORROBORATION_BONUS = 0.025
MANUAL_SAMPLE = "manual_reference"
MEETING_SAMPLE = "meeting_template"
ACTIVE_SAMPLE = "active"
DISABLED_SAMPLE = "disabled"
QUARANTINED_SAMPLE = "quarantined"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean(value: Any, limit: int = 80) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _vector(value: Any) -> np.ndarray:
    result = normalize_vector(value)
    if result.size <= 0 or not np.all(np.isfinite(result)):
        raise ValueError("Voice representation is empty or invalid.")
    return result.astype(np.float32)


def _policy(value: Any = None) -> dict[str, bool]:
    source = value if isinstance(value, Mapping) else {}
    return {
        "manual_samples": bool(source.get("manual_samples", True)),
        "meeting_samples": bool(source.get("meeting_samples", True)),
        "learn_from_confirmed_meetings": bool(source.get("learn_from_confirmed_meetings", True)),
    }


def _quality(seconds: float, count: int, cohesion: float | None) -> float:
    score = max(0.0, min(1.0, (seconds / 8.0) * min(1.0, count / 3.0)))
    if cohesion is not None:
        score *= max(0.5, max(-1.0, min(1.0, cohesion)))
    return round(score, 4)


@dataclass(frozen=True)
class PersonMatch:
    person_id: str
    name: str
    similarity: float
    margin: float
    template_count: int
    primary_sample_id: str = ""
    corroborating_sample_id: str = ""


class PersonLibrary:
    """Thread-safe JSON library with atomic writes and guarded v1 migration."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.storage_root = self.path.parent
        self._lock = threading.RLock()
        self._migration_pending = False
        self._migration_backup_path: Path | None = None
        self._document = self._load()

    @property
    def mutation_lock(self) -> threading.RLock:
        return self._lock

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"format": PEOPLE_FORMAT, "version": PEOPLE_VERSION, "updated_at": _now_iso(), "people": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read People library {self.path}: {exc}") from exc
        if not isinstance(document, dict) or document.get("format") != PEOPLE_FORMAT:
            raise ValueError(f"Unsupported People library format in {self.path}.")
        try:
            version = int(document.get("version") or 0)
        except (TypeError, ValueError):
            version = 0
        if version == 1:
            self._migration_pending = True
            return self._migrate_v1(document)
        if version != PEOPLE_VERSION:
            raise ValueError(
                f"Unsupported People library version {version} in {self.path}; "
                f"this build supports version {PEOPLE_VERSION} and will not downgrade it."
            )
        people = document.get("people")
        if not isinstance(people, list):
            raise ValueError(f"People library {self.path} has no people list.")
        document["people"] = [self._normalize_person(item) for item in people if isinstance(item, dict)]
        return document

    def _migrate_v1(self, document: Mapping[str, Any]) -> dict[str, Any]:
        migrated = self._empty()
        migrated["updated_at"] = str(document.get("updated_at") or _now_iso())
        for raw in document.get("people") or []:
            if not isinstance(raw, Mapping):
                continue
            person = self._new_person(_clean(raw.get("name")), person_id=str(raw.get("id") or uuid.uuid4().hex))
            person.update({
                "recognition_enabled": bool(raw.get("recognition_enabled", True)),
                "created_at": str(raw.get("created_at") or person["created_at"]),
                "updated_at": str(raw.get("updated_at") or person["updated_at"]),
                "profile_version": int(raw.get("profile_version") or 0),
            })
            for template in raw.get("templates") or []:
                if not isinstance(template, Mapping):
                    continue
                provider = str(template.get("embedding_provider") or "")
                try:
                    centroid = _vector(template.get("centroid"))
                except (TypeError, ValueError):
                    centroid = np.asarray([], dtype=np.float32)
                confirmation = str(template.get("confirmation") or "user")
                when = str(template.get("confirmed_at") or person["created_at"])
                quality = max(0.0, min(1.0, float(template.get("quality") or 0.0)))
                person["voice_samples"].append({
                    "id": str(template.get("id") or uuid.uuid4().hex),
                    "kind": MEETING_SAMPLE,
                    "state": ACTIVE_SAMPLE if centroid.size and provider else QUARANTINED_SAMPLE,
                    "label": _clean(template.get("source_title"), 120) or "Confirmed meeting",
                    "trust": "user_confirmed" if confirmation == "user" else "automatically_derived",
                    "created_at": when,
                    "updated_at": when,
                    "source": {"type": "confirmed_meeting", "session_id": str(template.get("session_id") or ""), "session_title": _clean(template.get("source_title"), 120), "capture_condition": ""},
                    "evidence": {"sentence_count": max(1, int(template.get("sentence_count") or 1)), "speech_seconds": max(0.0, float(template.get("speech_seconds") or 0.0)), "quality": quality, "cohesion": template.get("cohesion"), "outlier_count": max(0, int(template.get("outlier_count") or 0))},
                    "provenance": {"confirmation": confirmation, "anchor_sample_ids": [], "legacy_anchor": bool(template.get("anchor"))},
                    "representations": ([self._representation(centroid, provider, quality, when)] if centroid.size and provider else []),
                })
            migrated["people"].append(person)
        return migrated

    @staticmethod
    def _normalize_person(raw: Mapping[str, Any]) -> dict[str, Any]:
        person = dict(raw)
        person["id"] = str(person.get("id") or uuid.uuid4().hex)
        person["name"] = _clean(person.get("name"))
        person["expected"] = bool(person.get("expected", False))
        person["recognition_enabled"] = bool(person.get("recognition_enabled", True))
        person["recognition_policy"] = _policy(person.get("recognition_policy"))
        person["voice_samples"] = [dict(item) for item in (person.get("voice_samples") or []) if isinstance(item, Mapping)]
        person.pop("templates", None)
        return person

    def _backup_v1_locked(self) -> None:
        if not self._migration_pending or self._migration_backup_path is not None:
            return
        if not self.path.is_file():
            raise ValueError("The v1 People library disappeared before migration could be backed up.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = self.path.with_name(f"{self.path.stem}.v1.{stamp}.bak{self.path.suffix}")
        counter = 1
        while candidate.exists():
            candidate = self.path.with_name(f"{self.path.stem}.v1.{stamp}.{counter}.bak{self.path.suffix}")
            counter += 1
        try:
            shutil.copy2(self.path, candidate)
        except OSError as exc:
            raise ValueError(f"Could not back up v1 People library before migration: {exc}") from exc
        self._migration_backup_path = candidate

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_v1_locked()
        self._document.update({"format": PEOPLE_FORMAT, "version": PEOPLE_VERSION, "updated_at": _now_iso()})
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(self._document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError(f"Could not atomically write People library {self.path}: {exc}") from exc
        self._migration_pending = False

    @staticmethod
    def _new_person(name: str, *, person_id: str = "") -> dict[str, Any]:
        now = _now_iso()
        return {"id": person_id or uuid.uuid4().hex, "name": name, "expected": False, "recognition_enabled": True, "recognition_policy": _policy(), "created_at": now, "updated_at": now, "profile_version": 0, "voice_samples": []}

    def _find_locked(self, person_id: str) -> dict[str, Any]:
        target = str(person_id or "").strip()
        for person in self._document["people"]:
            if str(person.get("id") or "") == target:
                return person
        raise ValueError("Unknown Person.")

    @staticmethod
    def _samples(person: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [item for item in (person.get("voice_samples") or []) if isinstance(item, dict)]

    @staticmethod
    def _representation(centroid: np.ndarray, provider: str, quality: float, when: str) -> dict[str, Any]:
        return {"embedding_provider": provider, "embedding_length": int(centroid.size), "centroid": centroid.astype(float).tolist(), "quality": round(max(0.0, min(1.0, quality)), 4), "created_at": when}

    def get(self, person_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                result = copy.deepcopy(self._find_locked(person_id))
            except ValueError:
                return None
            # Temporary private compatibility for existing confirmation code.
            result["templates"] = self._legacy_templates(result)
            return result

    @staticmethod
    def _legacy_templates(person: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for sample in PersonLibrary._samples(person):
            if sample.get("kind") != MEETING_SAMPLE:
                continue
            source = sample.get("source") if isinstance(sample.get("source"), Mapping) else {}
            evidence = sample.get("evidence") if isinstance(sample.get("evidence"), Mapping) else {}
            provenance = sample.get("provenance") if isinstance(sample.get("provenance"), Mapping) else {}
            for representation in sample.get("representations") or []:
                if isinstance(representation, Mapping):
                    result.append({"id": str(sample.get("id") or ""), "centroid": copy.deepcopy(representation.get("centroid")), "embedding_provider": str(representation.get("embedding_provider") or ""), "session_id": str(source.get("session_id") or ""), "source_title": str(source.get("session_title") or sample.get("label") or ""), "sentence_count": int(evidence.get("sentence_count") or 1), "speech_seconds": float(evidence.get("speech_seconds") or 0.0), "quality": float(evidence.get("quality") or 0.0), "confirmation": str(provenance.get("confirmation") or "user"), "cohesion": evidence.get("cohesion"), "outlier_count": int(evidence.get("outlier_count") or 0), "confirmed_at": str(sample.get("updated_at") or ""), "anchor": bool(provenance.get("legacy_anchor")) or sample.get("trust") == "user_confirmed"})
        return result

    def create_person(self, name: str) -> dict[str, Any]:
        clean = _clean(name)
        if not clean:
            raise ValueError("Enter a name for the Person.")
        person = self._new_person(clean)
        with self._lock:
            self._document["people"].append(person)
            self._save_locked()
        return copy.deepcopy(person)

    def create_or_get(self, name: str) -> dict[str, Any]:
        """Compatibility alias; duplicate display names remain distinct."""
        return self.create_person(name)

    def rename_person(self, person_id: str, name: str) -> dict[str, Any]:
        clean = _clean(name)
        if not clean:
            raise ValueError("Person name must not be empty.")
        with self._lock:
            person = self._find_locked(person_id)
            person.update({"name": clean, "updated_at": _now_iso(), "profile_version": int(person.get("profile_version") or 0) + 1})
            self._save_locked()
            return copy.deepcopy(person)

    def set_recognition_enabled(self, person_id: str, enabled: bool) -> None:
        with self._lock:
            person = self._find_locked(person_id)
            person.update({"recognition_enabled": bool(enabled), "updated_at": _now_iso()})
            self._save_locked()

    def expected_person_ids(self) -> set[str]:
        with self._lock:
            return {
                str(person.get("id") or "")
                for person in self._document["people"]
                if bool(person.get("expected", False)) and str(person.get("id") or "")
            }

    def set_expected_people(self, person_ids: Iterable[str]) -> None:
        requested = {str(person_id or "").strip() for person_id in person_ids}
        requested.discard("")
        with self._lock:
            known = {
                str(person.get("id") or "")
                for person in self._document["people"]
                if str(person.get("id") or "")
            }
            unknown = requested - known
            if unknown:
                raise ValueError("Expected-People roster contains an unknown Person.")
            changed_at = _now_iso()
            changed = False
            for person in self._document["people"]:
                expected = str(person.get("id") or "") in requested
                if bool(person.get("expected", False)) == expected:
                    continue
                person.update({"expected": expected, "updated_at": changed_at})
                changed = True
            if changed:
                self._save_locked()

    def set_recognition_policy(self, person_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            person = self._find_locked(person_id)
            policy = _policy(person.get("recognition_policy"))
            for key in policy:
                if key in updates:
                    policy[key] = bool(updates[key])
            person.update({"recognition_policy": policy, "updated_at": _now_iso(), "profile_version": int(person.get("profile_version") or 0) + 1})
            self._save_locked()
            return copy.deepcopy(policy)

    def add_meeting_sample(self, person_id: str, centroid: Any, *, embedding_provider: str, session_id: str, source_title: str = "", capture_condition: str = "", sentence_count: int = 1, speech_seconds: float = 0.0, confirmation: str = "user", cohesion: float | None = None, outlier_count: int = 0, anchor_sample_ids: Iterable[str] = ()) -> dict[str, Any]:
        vector = _vector(centroid)
        provider = str(embedding_provider or "").strip()
        session_key = str(session_id or "").strip()
        if not provider:
            raise ValueError("Voice sample is missing its embedding provider.")
        if not session_key:
            raise ValueError("Meeting Voice samples require a session id.")
        seconds, count = max(0.0, float(speech_seconds or 0.0)), max(1, int(sentence_count or 1))
        normalized_cohesion = None if cohesion is None else max(-1.0, min(1.0, float(cohesion)))
        condition, when = _clean(capture_condition, 120), _now_iso()
        with self._lock:
            person, existing = self._find_locked(person_id), None
            samples = self._samples(person)
            for sample in samples:
                source = sample.get("source") if isinstance(sample.get("source"), Mapping) else {}
                providers = {str(rep.get("embedding_provider") or "") for rep in sample.get("representations") or [] if isinstance(rep, Mapping)}
                if sample.get("kind") == MEETING_SAMPLE and str(source.get("session_id") or "") == session_key and str(source.get("capture_condition") or "").casefold() == condition.casefold() and provider in providers:
                    existing = sample
                    break
            confirmation_value, score = str(confirmation or "user")[:40], _quality(seconds, count, normalized_cohesion)
            payload = {"id": str((existing or {}).get("id") or uuid.uuid4().hex), "kind": MEETING_SAMPLE, "state": ACTIVE_SAMPLE, "label": _clean(source_title, 120) or "Confirmed meeting", "trust": "user_confirmed" if confirmation_value == "user" else "automatically_derived", "created_at": str((existing or {}).get("created_at") or when), "updated_at": when, "source": {"type": "confirmed_meeting", "session_id": session_key, "session_title": _clean(source_title, 120), "capture_condition": condition}, "evidence": {"sentence_count": count, "speech_seconds": round(seconds, 4), "quality": score, "cohesion": None if normalized_cohesion is None else round(normalized_cohesion, 4), "outlier_count": max(0, int(outlier_count or 0))}, "provenance": {"confirmation": confirmation_value, "anchor_sample_ids": sorted({str(item) for item in anchor_sample_ids if str(item)}), "legacy_anchor": bool(((existing or {}).get("provenance") or {}).get("legacy_anchor"))}, "representations": [self._representation(vector, provider, score, when)]}
            if existing is None:
                samples.append(payload)
            else:
                existing.clear(); existing.update(payload)
            person["voice_samples"] = samples
            self._prune_locked(person, provider)
            person.update({"recognition_enabled": True, "profile_version": int(person.get("profile_version") or 0) + 1, "updated_at": when})
            self._save_locked()
            return copy.deepcopy(payload)

    def add_confirmed_template(self, person_id: str, centroid: Any, **kwargs: Any) -> dict[str, Any]:
        self.add_meeting_sample(person_id, centroid, **kwargs)
        result = self.get(person_id)
        assert result is not None
        return result

    def add_manual_sample(self, person_id: str, centroid: Any, *, embedding_provider: str, raw_audio: bytes, filename: str, label: str = "", source_type: str = "manual_upload", speech_seconds: float = 0.0, sentence_count: int = 1, quality: float = 1.0, cohesion: float | None = None, outlier_count: int = 0) -> dict[str, Any]:
        vector, provider = _vector(centroid), str(embedding_provider or "").strip()
        if not provider:
            raise ValueError("Voice sample is missing its embedding provider.")
        if not raw_audio:
            raise ValueError("Voice sample audio is empty.")
        checksum, when = hashlib.sha256(raw_audio).hexdigest(), _now_iso()
        with self._lock:
            person = self._find_locked(person_id)
            for sample in self._samples(person):
                raw = sample.get("raw_audio") if isinstance(sample.get("raw_audio"), Mapping) else {}
                if sample.get("kind") == MANUAL_SAMPLE and raw.get("sha256") == checksum:
                    raise ValueError("This exact Voice sample is already saved for this Person.")
            sample_id = uuid.uuid4().hex
            suffix = Path(str(filename or "sample.wav")).suffix.lower()
            if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
                suffix = ".wav"
            relative = Path("voice-samples") / str(person["id"]) / f"{sample_id}{suffix}"
            absolute = self.storage_root / relative
            absolute.parent.mkdir(parents=True, exist_ok=True)
            temporary = absolute.with_name(f".{absolute.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(raw_audio); os.replace(temporary, absolute)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Could not retain the manual Voice sample locally: {exc}") from exc
            sample = {"id": sample_id, "kind": MANUAL_SAMPLE, "state": ACTIVE_SAMPLE, "label": _clean(label, 120) or _clean(Path(filename).stem, 120) or "Manual voice sample", "trust": "user_confirmed", "created_at": when, "updated_at": when, "source": {"type": str(source_type or "manual_upload")[:40]}, "evidence": {"sentence_count": max(1, int(sentence_count or 1)), "speech_seconds": round(max(0.0, float(speech_seconds or 0.0)), 4), "quality": round(max(0.0, min(1.0, float(quality))), 4), "cohesion": None if cohesion is None else round(max(-1.0, min(1.0, float(cohesion))), 4), "outlier_count": max(0, int(outlier_count or 0))}, "raw_audio": {"retained": True, "relative_path": relative.as_posix(), "sha256": checksum}, "provenance": {"confirmation": "user", "anchor_sample_ids": []}, "representations": [self._representation(vector, provider, quality, when)]}
            person["voice_samples"] = self._samples(person) + [sample]
            person.update({"recognition_enabled": True, "profile_version": int(person.get("profile_version") or 0) + 1, "updated_at": when})
            try:
                self._save_locked()
            except Exception:
                try: absolute.unlink(missing_ok=True)
                except OSError: pass
                raise
            return copy.deepcopy(sample)

    def _sample_locked(self, person: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
        for sample in self._samples(person):
            if str(sample.get("id") or "") == str(sample_id or ""):
                return sample
        raise ValueError("Unknown Voice sample.")

    def set_sample_state(self, person_id: str, sample_id: str, enabled: bool) -> dict[str, Any]:
        with self._lock:
            person, when = self._find_locked(person_id), _now_iso()
            sample = self._sample_locked(person, sample_id)
            sample.update({"state": ACTIVE_SAMPLE if enabled else DISABLED_SAMPLE, "updated_at": when})
            person.update({"updated_at": when, "profile_version": int(person.get("profile_version") or 0) + 1})
            if not enabled: self._revalidate_locked(person, {str(sample_id)})
            self._save_locked()
            return copy.deepcopy(sample)

    def label_sample(self, person_id: str, sample_id: str, label: str) -> dict[str, Any]:
        clean = _clean(label, 120)
        if not clean: raise ValueError("Voice sample label must not be empty.")
        with self._lock:
            person, when = self._find_locked(person_id), _now_iso()
            sample = self._sample_locked(person, sample_id)
            sample.update({"label": clean, "updated_at": when}); person["updated_at"] = when
            self._save_locked(); return copy.deepcopy(sample)

    def _raw_path(self, sample: Mapping[str, Any]) -> Path | None:
        raw = sample.get("raw_audio") if isinstance(sample.get("raw_audio"), Mapping) else {}
        if not str(raw.get("relative_path") or ""): return None
        root, candidate = self.storage_root.resolve(), (self.storage_root / Path(str(raw["relative_path"]))).resolve()
        try: candidate.relative_to(root)
        except ValueError as exc: raise ValueError("Voice sample has an unsafe retained-audio path.") from exc
        return candidate

    def delete_sample(self, person_id: str, sample_id: str) -> dict[str, Any]:
        with self._lock:
            person, sample = self._find_locked(person_id), self._sample_locked(self._find_locked(person_id), sample_id)
            raw_path = self._raw_path(sample)
            if raw_path is not None and raw_path.exists():
                try: raw_path.unlink()
                except OSError as exc: raise ValueError(f"Could not delete retained Voice sample audio; no library data was changed: {exc}") from exc
            person["voice_samples"] = [item for item in self._samples(person) if str(item.get("id") or "") != str(sample_id)]
            self._revalidate_locked(person, {str(sample_id)})
            person.update({"updated_at": _now_iso(), "profile_version": int(person.get("profile_version") or 0) + 1})
            self._save_locked()
            return {"person_id": str(person["id"]), "sample_id": str(sample_id), "deleted": True}

    def _revalidate_locked(self, person: dict[str, Any], changed: set[str]) -> None:
        active_trusted = {str(sample.get("id") or "") for sample in self._samples(person) if sample.get("state", ACTIVE_SAMPLE) == ACTIVE_SAMPLE and sample.get("trust") == "user_confirmed"}
        for sample in self._samples(person):
            provenance = sample.get("provenance") if isinstance(sample.get("provenance"), Mapping) else {}
            anchors = {str(item) for item in provenance.get("anchor_sample_ids") or [] if str(item)}
            if sample.get("trust") == "automatically_derived" and anchors.intersection(changed) and not anchors.intersection(active_trusted):
                sample.update({"state": QUARANTINED_SAMPLE, "quarantine_reason": "trusted_anchor_unavailable", "updated_at": _now_iso()})

    def _prune_locked(self, person: dict[str, Any], provider: str) -> None:
        learned = [sample for sample in self._samples(person) if sample.get("kind") == MEETING_SAMPLE and sample.get("trust") != "user_confirmed" and any(isinstance(rep, Mapping) and str(rep.get("embedding_provider") or "") == provider for rep in sample.get("representations") or [])]
        overflow = len(learned) - MAX_LEARNED_SAMPLES_PER_PROVIDER
        if overflow > 0:
            remove = {str(item.get("id") or "") for item in sorted(learned, key=lambda item: (float((item.get("evidence") or {}).get("quality") or 0.0), str(item.get("updated_at") or "")))[:overflow]}
            person["voice_samples"] = [item for item in self._samples(person) if str(item.get("id") or "") not in remove]

    def remove_session_template(self, person_id: str, *, session_id: str, embedding_provider: str, capture_condition: str = "") -> bool:
        session_key, provider = str(session_id or "").strip(), str(embedding_provider or "").strip()
        if not session_key or not provider: return False
        with self._lock:
            person, retained, removed = self._find_locked(person_id), [], set()
            for sample in self._samples(person):
                source = sample.get("source") if isinstance(sample.get("source"), Mapping) else {}
                compatible = any(isinstance(rep, Mapping) and str(rep.get("embedding_provider") or "") == provider for rep in sample.get("representations") or [])
                if sample.get("kind") == MEETING_SAMPLE and str(source.get("session_id") or "") == session_key and str(source.get("capture_condition") or "").casefold() == str(capture_condition or "").casefold() and compatible: removed.add(str(sample.get("id") or ""))
                else: retained.append(sample)
            if not removed: return False
            person["voice_samples"] = retained; self._revalidate_locked(person, removed)
            person.update({"profile_version": int(person.get("profile_version") or 0) + 1, "updated_at": _now_iso()}); self._save_locked(); return True

    def forget_voice(self, person_id: str) -> None:
        with self._lock:
            person, failures = self._find_locked(person_id), []
            for sample in self._samples(person):
                try:
                    path = self._raw_path(sample)
                    if path is not None and path.exists(): path.unlink()
                except (OSError, ValueError) as exc: failures.append(str(exc))
            if failures: raise ValueError("Could not remove all retained Voice sample audio; the Person was kept unchanged: " + "; ".join(failures[:3]))
            person["voice_samples"] = []
            person.update({"profile_version": int(person.get("profile_version") or 0) + 1, "updated_at": _now_iso()}); self._save_locked()

    def delete_person(self, person_id: str) -> None:
        """Delete one Person and local Voice data without rewriting transcripts."""

        with self._lock:
            person = self._find_locked(person_id)
            failures = []
            for sample in self._samples(person):
                try:
                    path = self._raw_path(sample)
                    if path is not None and path.exists():
                        path.unlink()
                except (OSError, ValueError) as exc:
                    failures.append(str(exc))
            if failures:
                raise ValueError("Could not remove retained Voice sample audio; Person was kept: " + "; ".join(failures[:3]))
            self._document["people"] = [item for item in self._document["people"] if str(item.get("id") or "") != str(person_id)]
            self._save_locked()

    @staticmethod
    def _category_allowed(person: Mapping[str, Any], sample: Mapping[str, Any]) -> bool:
        policy = _policy(person.get("recognition_policy"))
        return (sample.get("kind") == MANUAL_SAMPLE and policy["manual_samples"]) or (sample.get("kind") == MEETING_SAMPLE and policy["meeting_samples"])

    def match(self, centroid: Any, *, embedding_provider: str, min_similarity: float, min_margin: float, excluded_person_ids: Iterable[str] = (), include_disabled: bool = False, expected_person_ids: Iterable[str] | None = None, eligible_sample_ids: Iterable[str] | None = None) -> PersonMatch | None:
        query, provider = _vector(centroid), str(embedding_provider or "").strip()
        excluded = {str(item or "") for item in excluded_person_ids}
        expected = None if expected_person_ids is None else {str(item) for item in expected_person_ids}
        override = None if eligible_sample_ids is None else {str(item) for item in eligible_sample_ids}
        scored: list[tuple[float, dict[str, Any], list[tuple[float, str]]]] = []
        with self._lock:
            for person in self._document["people"]:
                person_id = str(person.get("id") or "")
                if not person_id or person_id in excluded or (expected is not None and person_id not in expected) or (not include_disabled and not bool(person.get("recognition_enabled", True))): continue
                sample_scores: list[tuple[float, str, np.ndarray]] = []
                for sample in self._samples(person):
                    sample_id = str(sample.get("id") or "")
                    if sample.get("state", ACTIVE_SAMPLE) != ACTIVE_SAMPLE or (override is not None and sample_id not in override) or not self._category_allowed(person, sample): continue
                    rep_scores: list[tuple[float, np.ndarray]] = []
                    for rep in sample.get("representations") or []:
                        if not isinstance(rep, Mapping) or str(rep.get("embedding_provider") or "") != provider or int(rep.get("embedding_length") or 0) != int(query.size): continue
                        try: candidate = _vector(rep.get("centroid"))
                        except (TypeError, ValueError): continue
                        if candidate.shape == query.shape: rep_scores.append((cosine_similarity(query, candidate), candidate))
                    if rep_scores:
                        score, candidate = max(rep_scores, key=lambda item: item[0]); sample_scores.append((score, sample_id, candidate))
                if not sample_scores: continue
                sample_scores.sort(key=lambda item: item[0], reverse=True); primary = sample_scores[0]
                corroborating = next((item for item in sample_scores[1:] if item[0] >= CORROBORATION_SUPPORT_THRESHOLD and cosine_similarity(primary[2], item[2]) < NEAR_DUPLICATE_SIMILARITY), None)
                bonus = 0.0 if corroborating is None else min(MAX_CORROBORATION_BONUS, 0.15 * max(0.0, corroborating[0] - CORROBORATION_SUPPORT_THRESHOLD))
                sources = [(primary[0], primary[1])] + ([(corroborating[0], corroborating[1])] if corroborating is not None else [])
                scored.append((min(1.0, primary[0] + bonus), person, sources))
        if not scored: return None
        scored.sort(key=lambda item: item[0], reverse=True); top_score, top_person, sources = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0; margin = top_score - second if len(scored) > 1 else 1.0
        if not math.isfinite(top_score) or top_score < float(min_similarity) or margin < float(min_margin): return None
        return PersonMatch(str(top_person.get("id") or ""), _clean(top_person.get("name")), round(float(top_score), 4), round(float(margin), 4), len(sources), sources[0][1], sources[1][1] if len(sources) > 1 else "")

    def public_state(self, *, embedding_provider: str = "", embedding_length: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            people = []
            for person in self._document["people"]:
                public_samples = []
                for sample in self._samples(person):
                    source = sample.get("source") if isinstance(sample.get("source"), Mapping) else {}; evidence = sample.get("evidence") if isinstance(sample.get("evidence"), Mapping) else {}
                    reps = [rep for rep in sample.get("representations") or [] if isinstance(rep, Mapping)]
                    compatible = not embedding_provider or any(str(rep.get("embedding_provider") or "") == embedding_provider and (not embedding_length or int(rep.get("embedding_length") or 0) == embedding_length) for rep in reps)
                    state = str(sample.get("state") or ACTIVE_SAMPLE)
                    public_samples.append({"id": str(sample.get("id") or ""), "kind": str(sample.get("kind") or ""), "state": state, "effective_state": state if compatible else "incompatible", "compatibility_reason": "" if compatible else "incompatible_provider_or_dimension", "label": _clean(sample.get("label"), 120), "created_at": str(sample.get("created_at") or ""), "updated_at": str(sample.get("updated_at") or ""), "source_type": str(source.get("type") or ""), "session_id": str(source.get("session_id") or ""), "session_title": _clean(source.get("session_title"), 120), "speech_seconds": round(float(evidence.get("speech_seconds") or 0.0), 1), "sentence_count": int(evidence.get("sentence_count") or 0), "quality": round(float(evidence.get("quality") or 0.0), 2), "raw_audio_retained": bool(isinstance(sample.get("raw_audio"), Mapping) and sample["raw_audio"].get("retained"))})
                active = [item for item in public_samples if item["effective_state"] == ACTIVE_SAMPLE]
                people.append({"id": str(person.get("id") or ""), "name": _clean(person.get("name")), "expected": bool(person.get("expected", False)), "recognition_enabled": bool(person.get("recognition_enabled", True)), "recognition_policy": _policy(person.get("recognition_policy")), "recognition_ready": bool(active), "recognition_unavailable_reason": "" if active else "no_active_compatible_voice_samples", "voice_sample_count": len(public_samples), "active_voice_sample_count": len(active), "manual_sample_count": sum(item["kind"] == MANUAL_SAMPLE for item in public_samples), "meeting_sample_count": sum(item["kind"] == MEETING_SAMPLE for item in public_samples), "template_count": len(public_samples), "profile_version": int(person.get("profile_version") or 0), "last_confirmed_at": max((item["updated_at"] for item in public_samples), default=""), "voice_samples": public_samples})
            return sorted(people, key=lambda item: (item["name"].casefold(), item["id"]))
