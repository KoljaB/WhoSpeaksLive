"""Saved-session-specific Person linking and Voice-sample reconstruction.

Lock order is always SessionStore first, then PersonLibrary. A durable intent
file makes a retry repairable and idempotent if a process stops between writes.
"""

from __future__ import annotations

import base64
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from speakers.person_learning import PersonLearningPolicy, build_person_learning_candidate
from speakers.speaker_embedding_cluster import cosine_similarity, normalize_vector


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SavedPersonIdentityService:
    def __init__(self, store: Any, people: Any) -> None:
        self.store = store
        self.people = people

    @staticmethod
    def _decode_embedding(record: Mapping[str, Any]) -> np.ndarray | None:
        if str(record.get("embedding_encoding") or "") != "float32-base64-le":
            return None
        try:
            raw = base64.b64decode(str(record.get("embedding_b64") or ""), validate=True)
            vector = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            expected = int(record.get("embedding_length") or 0)
        except (ValueError, TypeError):
            return None
        if expected <= 0 or vector.size != expected or not np.all(np.isfinite(vector)):
            return None
        return normalize_vector(vector)

    @staticmethod
    def _profile_vectors(speakers_doc: Mapping[str, Any]) -> dict[str, np.ndarray]:
        profiles: dict[str, np.ndarray] = {}
        for profile in speakers_doc.get("speaker_profiles") or []:
            if not isinstance(profile, Mapping):
                continue
            label = str(profile.get("label") or "")
            try:
                centroid = normalize_vector(profile.get("centroid"))
            except (TypeError, ValueError):
                continue
            if label and centroid.size and np.all(np.isfinite(centroid)):
                profiles[label] = centroid
        return profiles

    def _documents(self, session_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        session_id = self.store._validate_session_id(session_id)
        session_dir = self.store._session_dir(session_id)
        manifest = self.store._load_manifest(session_id)
        transcript = self.store._read_json(session_dir / "transcript.json", {"rows": []})
        speakers = self.store._read_json(session_dir / "speakers.json", {"speaker_state": {}})
        embeddings = self.store._read_json(session_dir / "embeddings.json", {"records": []})
        return session_dir, manifest, transcript, speakers, embeddings

    @staticmethod
    def _intent_path(session_dir: Path) -> Path:
        return session_dir / ".person-identity-transaction.json"

    @staticmethod
    def _clear_speaker_link(speaker: dict[str, Any]) -> None:
        for key in ("person_id", "identity_status", "identity_source", "identity_confirmed_at"):
            speaker.pop(key, None)

    def _recover_pending_intent(self, session_id: str) -> dict[str, Any] | None:
        """Complete an interrupted saved-session Person mutation idempotently."""

        session_id = self.store._validate_session_id(session_id)
        intent_path = self._intent_path(self.store._session_dir(session_id))
        if not intent_path.is_file():
            return None
        with self.store.mutation_lock:
            with self.people.mutation_lock:
                intent = self.store._read_json(intent_path)
                operation = str(intent.get("operation") or "")
                speaker_id = str(intent.get("speaker_id") or "")
                person_id = str(intent.get("person_id") or "")
                session_dir, _manifest, _transcript, speakers, embeddings = self._documents(session_id)
                speaker = self._speaker(speakers, speaker_id)
                if speaker is None:
                    raise ValueError("Could not recover the Person change because the saved Speaker no longer exists.")
                if operation == "link":
                    person = self.people.get(person_id)
                    if person is None:
                        if not bool(intent.get("create_person")):
                            raise ValueError("Could not recover the Person link because the Person no longer exists.")
                        person = self.people.create_person(
                            str(intent.get("person_name") or ""),
                            person_id=person_id,
                        )
                    evidence = self.evidence(session_id, speaker_id, person_id=person_id)
                    if not evidence.get("available"):
                        raise ValueError(
                            "Could not recover the Person link: "
                            + str(evidence.get("explanation") or "saved voice evidence is unavailable.")
                        )
                    candidate = evidence["candidate"]
                    self.people.add_meeting_sample(
                        person_id,
                        candidate.centroid,
                        embedding_provider=str(evidence["provider"]),
                        session_id=session_id,
                        source_title=str(evidence["session_title"]),
                        sentence_count=candidate.sentence_count,
                        speech_seconds=candidate.speech_seconds,
                        confirmation="user",
                        cohesion=candidate.cohesion,
                        outlier_count=candidate.outlier_count,
                        allow_restore=True,
                    )
                    when = _now_iso()
                    speaker.update({
                        "person_id": person_id,
                        "identity_status": "confirmed",
                        "identity_source": "user",
                        "identity_confirmed_at": when,
                    })
                elif operation == "unlink":
                    provider = str(embeddings.get("embedding_provider") or "")
                    if person_id and provider:
                        self.people.remove_session_template(
                            person_id,
                            session_id=session_id,
                            embedding_provider=provider,
                        )
                    self._clear_speaker_link(speaker)
                    when = _now_iso()
                else:
                    raise ValueError("The saved Person transaction has an unsupported operation.")
                speakers["updated_at"] = when
                self.store._write_json(session_dir / "speakers.json", speakers)
                intent_path.unlink(missing_ok=True)
                return dict(intent)

    @staticmethod
    def _speaker(speakers_doc: Mapping[str, Any], speaker_id: str) -> dict[str, Any] | None:
        state = speakers_doc.get("speaker_state") if isinstance(speakers_doc.get("speaker_state"), Mapping) else {}
        return next((
            speaker for speaker in state.get("speakers") or []
            if isinstance(speaker, dict) and str(speaker.get("id") or "") == speaker_id
        ), None)

    def evidence(
        self,
        session_id: str,
        speaker_id: str,
        *,
        person_id: str = "",
    ) -> dict[str, Any]:
        """Return either a private robust candidate or a stable unavailable reason."""

        _session_dir, manifest, transcript, speakers, embeddings = self._documents(session_id)
        speaker_id = str(speaker_id or "").strip()
        if self._speaker(speakers, speaker_id) is None:
            return {"available": False, "reason": "unknown_speaker", "explanation": "This saved Speaker no longer exists."}
        provider = str(embeddings.get("embedding_provider") or "").strip()
        if not provider:
            return {"available": False, "reason": "missing_provider", "explanation": "This session does not record a compatible voice-analysis provider."}
        rows_by_index: dict[int, dict[str, Any]] = {}
        for position, row in enumerate(transcript.get("rows") or []):
            if not isinstance(row, dict):
                continue
            try: index = int(row.get("index"))
            except (TypeError, ValueError): index = position
            rows_by_index[index] = row
        records: list[dict[str, Any]] = []
        for record in embeddings.get("records") or []:
            if not isinstance(record, Mapping):
                continue
            try: index = int(record.get("index"))
            except (TypeError, ValueError): continue
            row = rows_by_index.get(index, {})
            assigned = str(row.get("assigned_speaker") or record.get("assigned_speaker") or "")
            if assigned != speaker_id:
                continue
            vector = self._decode_embedding(record)
            if vector is None:
                continue
            try:
                start = float(row.get("start") or 0.0)
                end = float(row.get("end") or start + float(record.get("duration_seconds") or 0.0))
            except (TypeError, ValueError):
                start, end = 0.0, float(record.get("duration_seconds") or 0.0)
            duration = max(0.0, float(record.get("duration_seconds") or end - start))
            records.append({
                "index": index,
                "embedding": vector,
                "duration_seconds": duration,
                "assigned_speaker": speaker_id,
                "quality": 1.0,
                "unknown_probability": 0.0,
                "base_payload": {"start": start, "end": max(end, start + duration), "speech_audio_ratio": 1.0},
                "correction": {"status": "user_confirmed", "corrected_speaker": speaker_id},
            })
        if not records:
            return {"available": False, "reason": "missing_embeddings", "explanation": "This saved Speaker has no compatible stored voice evidence."}
        profiles = self._profile_vectors(speakers)
        seed = profiles.get(speaker_id)
        if seed is None:
            seed = normalize_vector(np.mean(np.stack([record["embedding"] for record in records]), axis=0))
            profiles[speaker_id] = seed
        candidate = build_person_learning_candidate(
            records,
            profiles,
            speaker_id=speaker_id,
            seed_centroid=seed,
            policy=PersonLearningPolicy(
                min_sentences=1,
                min_speech_seconds=1.5,
                seed_similarity=0.35,
                min_cohesion=0.50,
            ),
        )
        if candidate is None:
            return {"available": False, "reason": "insufficient_evidence", "explanation": "This saved Speaker does not have enough coherent speech for future recognition."}
        other_scores = [
            cosine_similarity(candidate.centroid, value)
            for label, value in profiles.items()
            if label != speaker_id and value.shape == candidate.centroid.shape
        ]
        if other_scores and candidate.seed_similarity - max(other_scores) < 0.04:
            return {"available": False, "reason": "ambiguous_speakers", "explanation": "Stored evidence is too similar to another Speaker in this session."}
        competitor = self.people.match(
            candidate.centroid,
            embedding_provider=provider,
            min_similarity=-1.0,
            min_margin=0.0,
            excluded_person_ids={person_id} if person_id else (),
            include_disabled=True,
        )
        if competitor is not None and candidate.seed_similarity - competitor.similarity < 0.04:
            return {"available": False, "reason": "ambiguous_person", "explanation": "Stored evidence is too close to another Person's saved Voice samples."}
        return {
            "available": True,
            "reason": "",
            "explanation": "Compatible saved voice evidence is available.",
            "provider": provider,
            "candidate": candidate,
            "session_title": str(manifest.get("title") or "Saved meeting"),
        }

    def availability(self, session_id: str, speaker_id: str, *, person_id: str = "") -> dict[str, Any]:
        result = self.evidence(session_id, speaker_id, person_id=person_id)
        return {key: value for key, value in result.items() if key not in {"candidate", "provider"}}

    def decorate_session(self, session: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(session)
        session_id = str((result.get("summary") or {}).get("id") or (result.get("manifest") or {}).get("id") or "")
        if session_id and self._intent_path(self.store._session_dir(session_id)).is_file():
            self._recover_pending_intent(session_id)
            result = copy.deepcopy(self.store.open_session(session_id))
        state = result.get("speaker_state") if isinstance(result.get("speaker_state"), dict) else {}
        state["people"] = self.people.public_state()
        for speaker in state.get("speakers") or []:
            if not isinstance(speaker, dict):
                continue
            person_id = str(speaker.get("person_id") or "")
            speaker["future_recognition"] = self.availability(session_id, str(speaker.get("id") or ""), person_id=person_id)
        return result

    def link(
        self,
        session_id: str,
        speaker_id: str,
        *,
        person_id: str = "",
        person_name: str = "",
        expected_updated_at: str = "",
    ) -> dict[str, Any]:
        session_id, speaker_id = str(session_id or "").strip(), str(speaker_id or "").strip()
        recovered = self._recover_pending_intent(session_id)
        if recovered is not None and str(recovered.get("operation") or "") == "link" and str(recovered.get("speaker_id") or "") == speaker_id:
            return self.decorate_session(self.store.open_session(session_id))
        with self.store.mutation_lock:
            with self.people.mutation_lock:
                session_dir, manifest, _transcript, speakers, _embeddings = self._documents(session_id)
                if expected_updated_at and str(manifest.get("updated_at") or "") != str(expected_updated_at):
                    raise ValueError("The saved session changed; reopen it before linking this Speaker.")
                person = self.people.get(person_id) if person_id else None
                if person_id and person is None:
                    raise ValueError("Unknown Person.")
                clean_person_name = " ".join(str(person_name or "").strip().split())[:80]
                if person is None and not clean_person_name:
                    raise ValueError("Enter a name for the Person.")
                target_id = str(person["id"]) if person is not None else uuid.uuid4().hex
                evidence = self.evidence(session_id, speaker_id, person_id=target_id if person is not None else "")
                if not evidence.get("available"):
                    raise ValueError(str(evidence.get("explanation") or "Saved voice evidence is unavailable."))
                speaker = self._speaker(speakers, speaker_id)
                if speaker is None:
                    raise ValueError("This saved Speaker no longer exists.")
                self.store._write_json(self._intent_path(session_dir), {
                    "version": 1,
                    "operation": "link",
                    "session_id": session_id,
                    "speaker_id": speaker_id,
                    "person_id": target_id,
                    "person_name": str(person.get("name") or "") if person is not None else clean_person_name,
                    "create_person": person is None,
                    "created_at": _now_iso(),
                })
        self._recover_pending_intent(session_id)
        return self.decorate_session(self.store.open_session(session_id))

    def unlink(self, session_id: str, speaker_id: str) -> dict[str, Any]:
        recovered = self._recover_pending_intent(session_id)
        if recovered is not None and str(recovered.get("operation") or "") == "unlink" and str(recovered.get("speaker_id") or "") == str(speaker_id or ""):
            return self.decorate_session(self.store.open_session(session_id))
        with self.store.mutation_lock:
            with self.people.mutation_lock:
                session_dir, _manifest, _transcript, speakers, _embeddings = self._documents(session_id)
                speaker = self._speaker(speakers, speaker_id)
                if speaker is None:
                    raise ValueError("This saved Speaker no longer exists.")
                person_id = str(speaker.get("person_id") or "")
                self.store._write_json(self._intent_path(session_dir), {
                    "version": 1,
                    "operation": "unlink",
                    "session_id": str(session_id or ""),
                    "speaker_id": str(speaker_id or ""),
                    "person_id": person_id,
                    "created_at": _now_iso(),
                })
        self._recover_pending_intent(session_id)
        return self.decorate_session(self.store.open_session(session_id))

    def unlink_person_everywhere(self, person_id: str) -> int:
        """Remove one Person id from every saved session without rewriting names."""

        target = str(person_id or "").strip()
        if not target:
            return 0
        changed_sessions = 0
        for summary in self.store.list_sessions(filter_mode="all"):
            session_id = str(summary.get("id") or "")
            if not session_id:
                continue
            self._recover_pending_intent(session_id)
            with self.store.mutation_lock:
                session_dir, _manifest, _transcript, speakers, _embeddings = self._documents(session_id)
                changed = False
                state = speakers.get("speaker_state") if isinstance(speakers.get("speaker_state"), Mapping) else {}
                for speaker in state.get("speakers") or []:
                    if not isinstance(speaker, dict) or str(speaker.get("person_id") or "") != target:
                        continue
                    self._clear_speaker_link(speaker)
                    changed = True
                if changed:
                    speakers["updated_at"] = _now_iso()
                    self.store._write_json(session_dir / "speakers.json", speakers)
                    changed_sessions += 1
        return changed_sessions

    def remove_session_samples(self, session_id: str) -> int:
        """Remove Person-owned samples whose source meeting is being deleted."""

        return int(self.people.remove_session_samples(session_id))

    def recompute_linked_samples(self, session_id: str) -> None:
        with self.store.mutation_lock:
            _session_dir, _manifest, _transcript, speakers, embeddings = self._documents(session_id)
            state = speakers.get("speaker_state") if isinstance(speakers.get("speaker_state"), Mapping) else {}
            links = [(str(speaker.get("id") or ""), str(speaker.get("person_id") or "")) for speaker in state.get("speakers") or [] if isinstance(speaker, Mapping) and str(speaker.get("person_id") or "")]
        for speaker_id, person_id in links:
            if self.people.get(person_id) is None:
                continue
            evidence = self.evidence(session_id, speaker_id, person_id=person_id)
            if evidence.get("available"):
                candidate = evidence["candidate"]
                self.people.add_meeting_sample(person_id, candidate.centroid, embedding_provider=str(evidence["provider"]), session_id=session_id, source_title=str(evidence["session_title"]), sentence_count=candidate.sentence_count, speech_seconds=candidate.speech_seconds, confirmation="user", cohesion=candidate.cohesion, outlier_count=candidate.outlier_count)
            else:
                self.people.remove_session_template(person_id, session_id=session_id, embedding_provider=str(embeddings.get("embedding_provider") or ""))
