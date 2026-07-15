"""Persistent-person recognition layered beside meeting-local diarization."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading
import uuid
from typing import Any

import numpy as np

from speakers.person_learning import (
    PersonLearningCandidate,
    PersonLearningPolicy,
    build_person_learning_candidate,
)
from speakers.person_sample_ingestion import ingest_manual_voice_sample
from speakers.speaker_embedding_cluster import cosine_similarity, normalize_vector


_LEARNING_CHECKPOINT_NEW_SPEECH_SECONDS = 20.0
_LEARNING_FINAL_NEW_SPEECH_SECONDS = 2.0


@dataclass
class _PersonLearningState:
    person_id: str
    speaker_id: str
    session_id: str
    seed_centroid: np.ndarray
    checkpoint_speech_seconds: float = 0.0
    last_evaluated_profile_speech_seconds: float = 0.0
    checkpoint_record_indexes: frozenset[int] = field(default_factory=frozenset)
    checkpoint_user_trusted_indexes: frozenset[int] = field(default_factory=frozenset)


class WindowPersonIdentityMixin:
    def _person_learning_lock_obj(self) -> threading.RLock:
        lock = getattr(self, "_person_learning_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._person_learning_lock = lock
        return lock

    def _reset_person_learning_state(self) -> None:
        with self._person_learning_lock_obj():
            self._person_learning_states: dict[str, _PersonLearningState] = {}
            self._person_learning_fallback_session_id = ""

    def _person_learning_session_id(self) -> str:
        session_id = str(getattr(self, "_session_id", "") or "").strip()
        if session_id:
            return session_id
        active_run = getattr(self, "_active_run", None)
        run_id = str(getattr(active_run, "run_id", "") or "").strip()
        if run_id:
            return run_id
        with self._person_learning_lock_obj():
            fallback = str(getattr(self, "_person_learning_fallback_session_id", "") or "")
            if not fallback:
                fallback = f"runtime-{uuid.uuid4().hex}"
                self._person_learning_fallback_session_id = fallback
            return fallback

    def _profile_for_person_action(self, speaker_id: str) -> dict[str, Any]:
        label = str(speaker_id or "").strip()
        if not re.fullmatch(r"S\d+", label):
            raise ValueError("Invalid speaker id.")
        for profile in self.memory.export_profiles():
            if str(profile.get("label") or "") == label:
                return dict(profile)
        raise ValueError(f"Unknown speaker {label}.")

    def _person_match_thresholds(self) -> tuple[float, float]:
        same = float(getattr(self.args, "same_speaker_similarity", 0.45))
        margin = float(getattr(self.args, "min_margin", 0.04))
        # Cross-meeting names are more costly than a local clustering miss, so
        # suggestions use a slightly stricter operating point.
        return min(0.95, max(-1.0, same + 0.04)), max(0.04, margin)

    def _refresh_person_identity_suggestions(self, profiles: list[dict[str, Any]]) -> bool:
        library = getattr(self, "person_library", None)
        if library is None:
            return False
        profile_by_label = {str(item.get("label") or ""): item for item in profiles}
        with self._speaker_lock:
            confirmed = {
                str(metadata.get("person_id") or "")
                for metadata in self._speaker_metadata.values()
                if str(metadata.get("identity_status") or "") == "confirmed"
            }
            confirmed.discard("")
            metadata_by_label = {
                label: dict(metadata)
                for label, metadata in self._speaker_metadata.items()
                if label in profile_by_label
            }
        minimum_similarity, minimum_margin = self._person_match_thresholds()
        reserved = set(confirmed)
        public_identity_changed = False
        ordered_profiles = sorted(
            profiles,
            key=lambda item: (
                float(item.get("speech_seconds") or 0.0),
                int(item.get("sentence_count") or 0),
            ),
            reverse=True,
        )
        for profile in ordered_profiles:
            label = str(profile.get("label") or "")
            metadata = metadata_by_label.get(label)
            if metadata is None:
                continue
            status = str(metadata.get("identity_status") or "")
            person_id = str(metadata.get("person_id") or "")
            if status == "confirmed" and person_id and library.get(person_id) is not None:
                continue
            rejected = {
                str(value or "")
                for value in (metadata.get("rejected_person_ids") or [])
            }
            excluded = reserved | rejected
            enough_evidence = (
                int(profile.get("sentence_count") or 0) >= 2
                or float(profile.get("speech_seconds") or 0.0) >= 4.0
            )
            match = None
            if enough_evidence:
                match = library.match(
                    profile.get("centroid"),
                    embedding_provider=str(self.args.embedding_provider),
                    min_similarity=minimum_similarity,
                    min_margin=minimum_margin,
                    excluded_person_ids=excluded,
                    expected_person_ids=getattr(self, "_expected_person_ids", None),
                )
            with self._speaker_lock:
                current = self._speaker_metadata.get(label)
                if current is None or str(current.get("identity_status") or "") == "confirmed":
                    continue
                previous_public_identity = (
                    str(current.get("identity_status") or "unidentified"),
                    str(current.get("suggested_person_id") or ""),
                    str(current.get("suggested_person_name") or ""),
                )
                if match is None:
                    current.pop("suggested_person_id", None)
                    current.pop("suggested_person_name", None)
                    current.pop("identity_similarity", None)
                    current.pop("identity_margin", None)
                    current["identity_status"] = "unidentified"
                else:
                    current.update({
                        "identity_status": "suggested",
                        "suggested_person_id": match.person_id,
                        "suggested_person_name": match.name,
                        # Retained for diagnostics/session provenance; the normal UI
                        # intentionally exposes only the qualitative state.
                        "identity_similarity": match.similarity,
                        "identity_margin": match.margin,
                    })
                    reserved.add(match.person_id)
                current_public_identity = (
                    str(current.get("identity_status") or "unidentified"),
                    str(current.get("suggested_person_id") or ""),
                    str(current.get("suggested_person_name") or ""),
                )
                public_identity_changed = (
                    public_identity_changed
                    or current_public_identity != previous_public_identity
                )
        return public_identity_changed

    def _person_learning_candidate(
        self,
        speaker_id: str,
        seed_centroid: np.ndarray,
    ) -> PersonLearningCandidate | None:
        with self._sentence_refinement_lock:
            records = [dict(record) for record in self._sentence_refinement_records.values()]
        profiles = {
            str(profile.get("label") or ""): profile.get("centroid")
            for profile in self.memory.export_profiles()
            if str(profile.get("label") or "")
        }
        policy = PersonLearningPolicy(
            seed_similarity=max(
                0.35,
                float(getattr(self.args, "same_speaker_similarity", 0.45)) - 0.04,
            ),
            competing_speaker_margin=max(0.04, float(getattr(self.args, "min_margin", 0.04))),
            max_unknown_probability=min(0.55, float(getattr(self.args, "update_unknown_max", 0.55))),
            min_speech_audio_ratio=max(0.0, float(getattr(self.args, "min_speech_audio_ratio", 0.0))),
            min_cohesion=max(0.50, float(getattr(self.args, "same_speaker_similarity", 0.45))),
        )
        return build_person_learning_candidate(
            records,
            profiles,
            speaker_id=speaker_id,
            seed_centroid=seed_centroid,
            policy=policy,
        )

    def _person_learning_candidate_is_safe(
        self,
        state: _PersonLearningState,
        candidate: PersonLearningCandidate,
    ) -> bool:
        required_margin = max(0.04, float(getattr(self.args, "min_margin", 0.04)))
        profiles = {
            str(profile.get("label") or ""): normalize_vector(profile.get("centroid"))
            for profile in self.memory.export_profiles()
            if str(profile.get("label") or "")
        }
        active_competitors = [
            cosine_similarity(candidate.centroid, centroid)
            for label, centroid in profiles.items()
            if label != state.speaker_id and centroid.shape == candidate.centroid.shape
        ]
        if active_competitors and candidate.seed_similarity - max(active_competitors) < required_margin:
            return False
        competitor = self.person_library.match(
            candidate.centroid,
            embedding_provider=str(self.args.embedding_provider),
            min_similarity=-1.0,
            min_margin=0.0,
            excluded_person_ids={state.person_id},
            include_disabled=True,
        )
        if competitor is not None and candidate.seed_similarity - competitor.similarity < required_margin:
            return False
        return True

    def _add_confirmed_person_template(
        self,
        person_id: str,
        profile: dict[str, Any],
        *,
        confirmation: str = "user",
        candidate: PersonLearningCandidate | None = None,
        session_id: str | None = None,
    ) -> bool:
        # Linking an identity and enrolling reusable voice evidence are separate
        # operations. Never fall back to an unvalidated profile centroid when the
        # evidence builder could not produce a coherent learning candidate.
        if candidate is None:
            return False
        anchor_sample_ids: list[str] = []
        if confirmation != "user":
            person = self.person_library.get(person_id) or {}
            anchor_sample_ids = [
                str(sample.get("id") or "")
                for sample in (person.get("voice_samples") or [])
                if isinstance(sample, dict)
                and sample.get("state", "active") == "active"
                and sample.get("trust") == "user_confirmed"
            ]
        result = self.person_library.add_meeting_sample(
            person_id,
            candidate.centroid if candidate is not None else profile.get("centroid"),
            embedding_provider=str(self.args.embedding_provider),
            session_id=session_id or self._person_learning_session_id(),
            source_title=str(getattr(self, "_session_source_title", "") or ""),
            sentence_count=(candidate.sentence_count if candidate is not None else int(profile.get("sentence_count") or 1)),
            speech_seconds=(candidate.speech_seconds if candidate is not None else float(profile.get("speech_seconds") or 0.0)),
            confirmation=confirmation,
            cohesion=(candidate.cohesion if candidate is not None else None),
            outlier_count=(candidate.outlier_count if candidate is not None else 0),
            anchor_sample_ids=anchor_sample_ids,
            allow_restore=confirmation == "user",
        )
        return not bool(result.get("suppressed"))

    def _start_person_learning(
        self,
        speaker_id: str,
        person_id: str,
        profile: dict[str, Any],
        candidate: PersonLearningCandidate | None,
    ) -> None:
        with self._sentence_refinement_lock:
            assigned_indexes = frozenset(
                int(record.get("index"))
                for record in self._sentence_refinement_records.values()
                if str(record.get("assigned_speaker") or "") == speaker_id
                and record.get("index") is not None
            )
        state = _PersonLearningState(
            person_id=person_id,
            speaker_id=speaker_id,
            session_id=self._person_learning_session_id(),
            seed_centroid=normalize_vector(profile.get("centroid")),
            checkpoint_speech_seconds=(candidate.speech_seconds if candidate is not None else 0.0),
            last_evaluated_profile_speech_seconds=float(profile.get("speech_seconds") or 0.0),
            checkpoint_record_indexes=(candidate.record_indexes if candidate is not None else assigned_indexes),
            checkpoint_user_trusted_indexes=(
                candidate.user_trusted_indexes if candidate is not None else frozenset()
            ),
        )
        with self._person_learning_lock_obj():
            self._person_learning_states[speaker_id] = state

    def _discard_person_learning(self, speaker_id: str, *, remove_template: bool) -> None:
        with self._person_learning_lock_obj():
            state = self._person_learning_states.pop(speaker_id, None)
        if remove_template and state is not None:
            self.person_library.remove_session_template(
                state.person_id,
                session_id=state.session_id,
                embedding_provider=str(self.args.embedding_provider),
            )

    def _maybe_checkpoint_confirmed_people(
        self,
        *,
        final: bool = False,
        review_assignments: bool = False,
    ) -> None:
        library = getattr(self, "person_library", None)
        if library is None:
            return
        with self._person_learning_lock_obj():
            with self._speaker_lock:
                confirmed = {
                    label: str(metadata.get("person_id") or "")
                    for label, metadata in self._speaker_metadata.items()
                    if str(metadata.get("identity_status") or "") == "confirmed"
                    and str(metadata.get("person_id") or "")
                }
            stale = [
                label
                for label, state in self._person_learning_states.items()
                if confirmed.get(label) != state.person_id
            ]
            for label in stale:
                state = self._person_learning_states.pop(label)
                library.remove_session_template(
                    state.person_id,
                    session_id=state.session_id,
                    embedding_provider=str(self.args.embedding_provider),
                )
            profiles = {
                str(profile.get("label") or ""): dict(profile)
                for profile in self.memory.export_profiles()
            }
            for speaker_id, person_id in confirmed.items():
                person = library.get(person_id)
                profile = profiles.get(speaker_id)
                policy = (person or {}).get("recognition_policy") or {}
                if person is None or profile is None or not bool(
                    policy.get("learn_from_confirmed_meetings", True)
                ):
                    continue
                state = self._person_learning_states.get(speaker_id)
                if state is None:
                    candidate = self._person_learning_candidate(
                        speaker_id,
                        normalize_vector(profile.get("centroid")),
                    )
                    self._start_person_learning(speaker_id, person_id, profile, candidate)
                    state = self._person_learning_states[speaker_id]
                    if (
                        final
                        and candidate is not None
                        and self._person_learning_candidate_is_safe(state, candidate)
                    ):
                        self._add_confirmed_person_template(
                            person_id,
                            profile,
                            confirmation="automatic_final",
                            candidate=candidate,
                            session_id=state.session_id,
                        )
                    continue
                profile_speech_seconds = float(profile.get("speech_seconds") or 0.0)
                if (
                    not final
                    and not review_assignments
                    and profile_speech_seconds - state.last_evaluated_profile_speech_seconds
                    < _LEARNING_CHECKPOINT_NEW_SPEECH_SECONDS
                ):
                    continue
                state.last_evaluated_profile_speech_seconds = profile_speech_seconds
                candidate = self._person_learning_candidate(speaker_id, state.seed_centroid)
                if candidate is None:
                    with self._sentence_refinement_lock:
                        current_indexes = {
                            int(record.get("index"))
                            for record in self._sentence_refinement_records.values()
                            if str(record.get("assigned_speaker") or "") == speaker_id
                            and record.get("index") is not None
                        }
                    if state.checkpoint_record_indexes.difference(current_indexes):
                        library.remove_session_template(
                            state.person_id,
                            session_id=state.session_id,
                            embedding_provider=str(self.args.embedding_provider),
                        )
                        state.checkpoint_speech_seconds = 0.0
                        state.last_evaluated_profile_speech_seconds = 0.0
                        state.checkpoint_record_indexes = frozenset()
                        state.checkpoint_user_trusted_indexes = frozenset()
                    continue
                if not self._person_learning_candidate_is_safe(state, candidate):
                    continue
                new_seconds = candidate.speech_seconds - state.checkpoint_speech_seconds
                removed_evidence = bool(
                    state.checkpoint_record_indexes
                    and not state.checkpoint_record_indexes.issubset(candidate.record_indexes)
                )
                corrected_evidence = not candidate.user_trusted_indexes.issubset(
                    state.checkpoint_user_trusted_indexes
                )
                enough_new_evidence = new_seconds >= _LEARNING_CHECKPOINT_NEW_SPEECH_SECONDS
                if final:
                    enough_new_evidence = new_seconds >= _LEARNING_FINAL_NEW_SPEECH_SECONDS
                if not (enough_new_evidence or removed_evidence or corrected_evidence):
                    continue
                self._add_confirmed_person_template(
                    person_id,
                    profile,
                    confirmation="automatic_final" if final else "automatic_checkpoint",
                    candidate=candidate,
                    session_id=state.session_id,
                )
                state.checkpoint_speech_seconds = candidate.speech_seconds
                state.checkpoint_record_indexes = candidate.record_indexes
                state.checkpoint_user_trusted_indexes = candidate.user_trusted_indexes

    def remember_speaker_as_person(
        self,
        speaker_id: str,
        name: str = "",
        person_id: str = "",
    ) -> dict[str, Any]:
        profile = self._profile_for_person_action(speaker_id)
        label = str(profile.get("label") or "")
        with self._speaker_lock:
            metadata = self._speaker_metadata.get(label)
            if metadata is None:
                raise ValueError(f"Unknown speaker {label}.")
            clean_name = " ".join(str(name or metadata.get("name") or "").strip().split())[:80]
        if person_id:
            person = self.person_library.get(person_id)
            if person is None:
                raise ValueError("Unknown Person.")
        else:
            person = self.person_library.create_person(clean_name)
        seed = normalize_vector(profile.get("centroid"))
        candidate = self._person_learning_candidate(label, seed)
        provisional = _PersonLearningState(
            person_id=str(person["id"]),
            speaker_id=label,
            session_id=self._person_learning_session_id(),
            seed_centroid=seed,
        )
        if candidate is not None and not self._person_learning_candidate_is_safe(provisional, candidate):
            candidate = None
        sample_saved = self._add_confirmed_person_template(
            str(person["id"]),
            profile,
            candidate=candidate,
            session_id=provisional.session_id,
        )
        self._start_person_learning(label, str(person["id"]), profile, candidate)
        with self._speaker_lock:
            metadata = self._speaker_metadata[label]
            metadata.update({
                "name": str(person.get("name") or clean_name),
                "person_id": str(person["id"]),
                "identity_status": "confirmed",
                "identity_source": "user",
                "source": "remembered",
            })
            metadata.pop("suggested_person_id", None)
            metadata.pop("suggested_person_name", None)
        message = (
            f"Linked {label} to {person['name']} and saved a Voice sample."
            if sample_saved
            else f"Linked {label} to {person['name']}; more clean speech is needed before recognition can be saved."
        )
        self.bus.emit("status", {"message": message})
        return self.emit_speaker_state()

    def confirm_speaker_person(self, speaker_id: str, person_id: str) -> dict[str, Any]:
        profile = self._profile_for_person_action(speaker_id)
        person = self.person_library.get(person_id)
        if person is None:
            raise ValueError("Unknown remembered person.")
        label = str(profile.get("label") or "")
        seed = normalize_vector(profile.get("centroid"))
        candidate = self._person_learning_candidate(label, seed)
        provisional = _PersonLearningState(
            person_id=str(person["id"]),
            speaker_id=label,
            session_id=self._person_learning_session_id(),
            seed_centroid=seed,
        )
        if candidate is not None and not self._person_learning_candidate_is_safe(provisional, candidate):
            candidate = None
        sample_saved = self._add_confirmed_person_template(
            str(person["id"]),
            profile,
            candidate=candidate,
            session_id=provisional.session_id,
        )
        self._start_person_learning(label, str(person["id"]), profile, candidate)
        with self._speaker_lock:
            metadata = self._speaker_metadata.get(label)
            if metadata is None:
                raise ValueError(f"Unknown speaker {label}.")
            metadata.update({
                "name": str(person.get("name") or ""),
                "person_id": str(person["id"]),
                "identity_status": "confirmed",
                "identity_source": "user",
                "source": "remembered",
            })
            metadata.pop("suggested_person_id", None)
            metadata.pop("suggested_person_name", None)
        message = (
            f"Confirmed {person['name']} and saved a Voice sample."
            if sample_saved
            else f"Confirmed {person['name']}; more clean speech is needed before recognition can be saved."
        )
        self.bus.emit("status", {"message": message})
        return self.emit_speaker_state()

    def reject_speaker_person(self, speaker_id: str, person_id: str = "") -> dict[str, Any]:
        profile = self._profile_for_person_action(speaker_id)
        label = str(profile.get("label") or "")
        with self._speaker_lock:
            metadata = self._speaker_metadata.get(label)
            if metadata is None:
                raise ValueError(f"Unknown speaker {label}.")
            rejected_id = str(
                person_id
                or metadata.get("suggested_person_id")
                or metadata.get("person_id")
                or ""
            )
            rejected = {
                str(value or "") for value in (metadata.get("rejected_person_ids") or [])
            }
            if rejected_id:
                rejected.add(rejected_id)
            confirmed_was_rejected = (
                str(metadata.get("identity_status") or "") == "confirmed"
                and str(metadata.get("person_id") or "") == rejected_id
            )
            if confirmed_was_rejected and str(metadata.get("name") or "") == str(
                (self.person_library.get(rejected_id) or {}).get("name") or ""
            ):
                metadata["name"] = ""
            metadata.update({
                "identity_status": "unidentified",
                "person_id": "",
                "identity_source": "",
                "rejected_person_ids": sorted(rejected),
            })
            metadata.pop("suggested_person_id", None)
            metadata.pop("suggested_person_name", None)
        self._discard_person_learning(label, remove_template=True)
        if rejected_id:
            self.person_library.remove_session_template(
                rejected_id,
                session_id=self._person_learning_session_id(),
                embedding_provider=str(self.args.embedding_provider),
            )
        self.bus.emit("status", {"message": f"Kept {label} unidentified."})
        return self.emit_speaker_state()

    def unlink_speaker_person(self, speaker_id: str) -> dict[str, Any]:
        profile = self._profile_for_person_action(speaker_id)
        label = str(profile.get("label") or "")
        with self._speaker_lock:
            metadata = self._speaker_metadata.get(label)
            if metadata is None:
                raise ValueError(f"Unknown speaker {label}.")
            person_id = str(metadata.get("person_id") or "")
            metadata.update({"person_id": "", "identity_status": "unidentified", "identity_source": ""})
            metadata.pop("suggested_person_id", None)
            metadata.pop("suggested_person_name", None)
        self._discard_person_learning(label, remove_template=True)
        if person_id:
            self.person_library.remove_session_template(
                person_id,
                session_id=self._person_learning_session_id(),
                embedding_provider=str(self.args.embedding_provider),
            )
        self.bus.emit("status", {"message": f"Unlinked {label} from the Person; transcript names were kept."})
        return self.emit_speaker_state()

    def set_person_recognition(self, person_id: str, enabled: bool) -> dict[str, Any]:
        self.person_library.set_recognition_enabled(person_id, enabled)
        self.bus.emit("status", {
            "message": "Person added to recognition candidates." if enabled else "Person removed from recognition candidates."
        })
        return self.emit_speaker_state()

    def create_person(self, name: str) -> dict[str, Any]:
        self.person_library.create_person(name)
        return self.emit_speaker_state()

    def rename_person(self, person_id: str, name: str) -> dict[str, Any]:
        self.person_library.rename_person(person_id, name)
        return self.emit_speaker_state()

    def set_person_recognition_policy(
        self,
        person_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        self.person_library.set_recognition_policy(person_id, updates)
        return self.emit_speaker_state()

    def set_voice_sample_enabled(
        self,
        person_id: str,
        sample_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        self.person_library.set_sample_state(person_id, sample_id, enabled)
        return self.emit_speaker_state()

    def label_voice_sample(self, person_id: str, sample_id: str, label: str) -> dict[str, Any]:
        self.person_library.label_sample(person_id, sample_id, label)
        return self.emit_speaker_state()

    def delete_voice_sample(self, person_id: str, sample_id: str) -> dict[str, Any]:
        self.person_library.delete_sample(person_id, sample_id)
        return self.emit_speaker_state()

    def add_manual_voice_sample(
        self,
        person_id: str,
        filename: str,
        audio_b64: str,
        *,
        label: str = "",
        source_type: str = "manual_upload",
    ) -> dict[str, Any]:
        if self.person_library.get(person_id) is None:
            raise ValueError("Choose a Person before adding a Voice sample.")
        ingest_manual_voice_sample(
            self.person_library,
            self.embedding,
            person_id=person_id,
            embedding_provider=str(self.args.embedding_provider),
            filename=filename,
            audio_b64=audio_b64,
            label=label,
            source_type=source_type,
        )
        return self.emit_speaker_state()

    def set_expected_people(self, person_ids: list[str] | None) -> dict[str, Any]:
        requested = {str(value or "").strip() for value in (person_ids or [])}
        requested.discard("")
        self.person_library.set_expected_people(requested)
        self._expected_person_ids = requested
        return self.emit_speaker_state()

    def forget_person_voice(self, person_id: str) -> dict[str, Any]:
        person = self.person_library.get(person_id)
        if person is None:
            raise ValueError("Unknown remembered person.")
        self.person_library.forget_voice(person_id)
        discarded: list[str] = []
        with self._speaker_lock:
            for label, metadata in self._speaker_metadata.items():
                if str(metadata.get("person_id") or "") == person_id:
                    discarded.append(label)
                    metadata.update({
                        "person_id": "",
                        "identity_status": "unidentified",
                        "identity_source": "",
                    })
                if str(metadata.get("suggested_person_id") or "") == person_id:
                    metadata.pop("suggested_person_id", None)
                    metadata.pop("suggested_person_name", None)
                    metadata["identity_status"] = "unidentified"
        for label in discarded:
            self._discard_person_learning(label, remove_template=False)
        self.bus.emit("status", {"message": f"Forgot the saved voice for {person['name']}; transcript names were kept."})
        return self.emit_speaker_state()

    def delete_person(self, person_id: str) -> dict[str, Any]:
        person = self.person_library.get(person_id)
        if person is None:
            raise ValueError("Unknown Person.")
        with self._speaker_lock:
            linked_labels = [
                label
                for label, metadata in self._speaker_metadata.items()
                if str(metadata.get("person_id") or "") == person_id
            ]
        # Drop learning state before deleting the library record. Otherwise the
        # next state emission tries to clean up a sample on a missing Person.
        for label in linked_labels:
            self._discard_person_learning(label, remove_template=False)
        self.person_library.delete_person(person_id)
        self._expected_person_ids.discard(person_id)
        with self._speaker_lock:
            for metadata in self._speaker_metadata.values():
                if str(metadata.get("person_id") or "") == person_id:
                    metadata.update({"person_id": "", "identity_status": "unidentified", "identity_source": ""})
                if str(metadata.get("suggested_person_id") or "") == person_id:
                    metadata.pop("suggested_person_id", None)
                    metadata.pop("suggested_person_name", None)
        return self.emit_speaker_state()

    def consolidate_confirmed_people(self) -> None:
        """Commit any final, independently validated evidence not yet checkpointed."""
        self._maybe_checkpoint_confirmed_people(final=True)
