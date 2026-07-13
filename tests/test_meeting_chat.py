from __future__ import annotations

import tempfile
import time
from pathlib import Path
import unittest

from window.meeting_chat import (
    MAX_CHUNK_WORDS,
    MAX_SCOPE_MEETINGS,
    MeetingChatEngine,
    MeetingChatJobManager,
    MeetingChatStore,
    MeetingTextIndex,
    MockTextEmbeddingClient,
    TextEmbeddingConfig,
    chunk_transcript,
    finalized_rows,
    scope_id_for,
)
from window.meeting_intelligence import transcript_revision_id
from window.meeting_intelligence_pipeline import MockMeetingLLMClient


def row(index: int, text: str, speaker: str = "Alice", speaker_id: str = "S1") -> dict[str, object]:
    return {
        "row_id": f"row-{index}",
        "text": text,
        "speaker_id": speaker_id,
        "speaker_name": speaker,
        "start": float(index * 10),
        "end": float(index * 10 + 8),
    }


def loaded_session(session_id: str, title: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "summary": {"id": session_id, "title": title},
        "transcript_rows": rows,
        "speaker_state": {"speakers": []},
    }


def indexed_session(session_id: str, title: str, rows: list[dict[str, object]]) -> dict[str, object]:
    revision = transcript_revision_id(rows, {"speakers": []})
    return {
        "session_id": session_id,
        "title": title,
        "revision_id": revision,
        "rows": rows,
        "chunks": chunk_transcript(session_id, title, rows, revision),
    }


class CountingEmbeddingClient(MockTextEmbeddingClient):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed(texts)


class InvalidCitationClient(MockMeetingLLMClient):
    def chat_json(self, *, schema_name, schema, system_prompt, user_payload, max_tokens):  # type: ignore[no-untyped-def]
        if schema_name == "meeting_chat_answer":
            return {
                "schema_version": "meeting_chat_answer_v1",
                "status": "answered",
                "answer": "An unsupported claim.",
                "evidence_ids": ["another-meeting::invented-row"],
            }
        return super().chat_json(
            schema_name=schema_name,
            schema=schema,
            system_prompt=system_prompt,
            user_payload=user_payload,
            max_tokens=max_tokens,
        )


class WrongSpeakerCitationClient(MockMeetingLLMClient):
    def chat_json(self, *, schema_name, schema, system_prompt, user_payload, max_tokens):  # type: ignore[no-untyped-def]
        if schema_name == "meeting_chat_answer":
            evidence = user_payload["evidence_rows"][0]
            return {
                "schema_version": "meeting_chat_answer_v1",
                "status": "answered",
                "answer": "Jerome Powell said the unrelated sentence.",
                "evidence_ids": [evidence["evidence_id"]],
            }
        return super().chat_json(
            schema_name=schema_name,
            schema=schema,
            system_prompt=system_prompt,
            user_payload=user_payload,
            max_tokens=max_tokens,
        )


class FalseNegativeClient(MockMeetingLLMClient):
    def __init__(self) -> None:
        self.answer_payloads: list[dict[str, object]] = []

    def chat_json(self, *, schema_name, schema, system_prompt, user_payload, max_tokens):  # type: ignore[no-untyped-def]
        if schema_name == "meeting_chat_answer":
            self.answer_payloads.append(user_payload)
            return {
                "schema_version": "meeting_chat_answer_v1",
                "status": "not_established",
                "answer": "status not_established",
                "evidence_ids": [],
            }
        return super().chat_json(
            schema_name=schema_name,
            schema=schema,
            system_prompt=system_prompt,
            user_payload=user_payload,
            max_tokens=max_tokens,
        )


class MeetingChunkingTests(unittest.TestCase):
    def test_saved_row_index_becomes_navigable_canonical_row_id(self) -> None:
        source = row(1, "Beethoven wrote a symphony.")
        source.pop("row_id")
        source["index"] = 39

        normalized = finalized_rows([source])

        self.assertEqual(normalized[0]["row_id"], "row_39")
        evidence = MeetingChatEngine._evidence_row(
            {"session_id": "meeting", "title": "Music"},
            normalized[0],
        )
        self.assertEqual(evidence["row_index"], 39)
        self.assertEqual(evidence["row_id"], "row_39")

    def test_chunks_overlap_by_whole_rows_and_never_exceed_hard_limit(self) -> None:
        rows = [row(index, " ".join(f"word{index}-{word}" for word in range(55))) for index in range(7)]
        revision = transcript_revision_id(rows, {})

        chunks = chunk_transcript("meeting-a", "Planning", rows, revision)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["word_count"] <= MAX_CHUNK_WORDS for chunk in chunks))
        self.assertTrue(set(chunks[0]["row_ids"]) & set(chunks[1]["row_ids"]))

    def test_an_exceptionally_long_row_is_split_without_losing_citation_identity(self) -> None:
        rows = [row(1, " ".join(f"token-{index}" for index in range(700)))]

        chunks = chunk_transcript("meeting-a", "Long monologue", rows, "revision-a")

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk["word_count"] <= MAX_CHUNK_WORDS for chunk in chunks))
        self.assertTrue(all(chunk["row_ids"] == ["row-1"] for chunk in chunks))

    def test_speaker_correction_and_rename_change_hashes_and_revision(self) -> None:
        original = [row(1, "Order the fabric.", "Unknown", "UNKNOWN")]
        corrected = [row(1, "Order the fabric.", "Ms. Müller", "speaker-mueller")]
        original_revision = transcript_revision_id(original, {"speakers": []})
        corrected_revision = transcript_revision_id(corrected, {
            "speakers": [{"id": "speaker-mueller", "name": "Ms. Müller"}],
        })

        original_chunk = chunk_transcript("m", "Meeting", original, original_revision)[0]
        corrected_chunk = chunk_transcript("m", "Meeting", corrected, corrected_revision)[0]

        self.assertNotEqual(original_revision, corrected_revision)
        self.assertNotEqual(original_chunk["content_hash"], corrected_chunk["content_hash"])
        self.assertEqual(
            corrected_chunk["content_hash"],
            chunk_transcript("m", "Meeting", corrected, corrected_revision)[0]["content_hash"],
        )


class MeetingIndexTests(unittest.TestCase):
    def test_incremental_index_persists_and_model_change_reembeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "index.sqlite3"
            client = CountingEmbeddingClient()
            config = TextEmbeddingConfig("http://embedding.test/v1", "model-a")
            session = indexed_session("meeting-a", "Macro briefing", [row(1, "Inflation is easing.")])
            index = MeetingTextIndex(database, config, client_factory=lambda: client)

            self.assertTrue(index._ensure_session(session))
            first_batches = list(client.batch_sizes)
            self.assertFalse(index._ensure_session(session))
            self.assertEqual(client.batch_sizes, first_batches)

            reopened = MeetingTextIndex(database, config, client_factory=lambda: client)
            state = reopened.public_state(["meeting-a"])["sessions"][0]
            self.assertTrue(state["indexed"])
            self.assertTrue(state["current_embedding_model"])

            changed_model = MeetingTextIndex(
                database,
                TextEmbeddingConfig("http://embedding.test/v1", "model-b"),
                client_factory=lambda: client,
            )
            self.assertTrue(changed_model._ensure_session(session))
            self.assertGreater(len(client.batch_sizes), len(first_batches))

    def test_hybrid_search_is_isolated_and_boosts_a_named_diarized_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = MeetingTextIndex(
                Path(directory) / "index.sqlite3",
                TextEmbeddingConfig("mock://embeddings", "mock"),
                client_factory=MockTextEmbeddingClient,
            )
            powell = indexed_session(
                "fed",
                "Federal Reserve briefing",
                [row(1, "Further tightening may be appropriate.", "Jerome Powell", "trained-powell")],
            )
            support = indexed_session(
                "support",
                "Support call",
                [row(1, "The customer requested a refund.", "Unknown", "UNKNOWN")],
            )
            index.ensure_sessions([powell, support])

            matches = index.search(["fed", "support"], "What did Jerome Powell say?")

            self.assertEqual(matches[0]["session_id"], "fed")
            self.assertIn("Jerome Powell", matches[0]["speaker_names"])
            self.assertEqual({match["session_id"] for match in index.search(["support"], "refund")}, {"support"})

    def test_delete_removes_persisted_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = MeetingTextIndex(
                Path(directory) / "index.sqlite3",
                TextEmbeddingConfig("mock://embeddings", "mock"),
                client_factory=MockTextEmbeddingClient,
            )
            index.ensure_sessions([indexed_session("m", "Meeting", [row(1, "A decision")])])
            index.delete_session("m")
            self.assertEqual(index.session_ids(), set())


class MeetingChatEngineTests(unittest.TestCase):
    def make_engine(self, root: Path, sessions: dict[str, dict[str, object]], client_factory=MockMeetingLLMClient):  # type: ignore[no-untyped-def]
        index = MeetingTextIndex(
            root / "index.sqlite3",
            TextEmbeddingConfig("mock://embeddings", "mock"),
            client_factory=MockTextEmbeddingClient,
        )
        store = MeetingChatStore(root / "chats")
        return MeetingChatEngine(
            index,
            store,
            session_loader=lambda session_id: sessions[session_id],
            llm_client_factory=client_factory,
        )

    def test_scope_id_is_order_independent_and_limited_to_twenty_meetings(self) -> None:
        self.assertEqual(scope_id_for(["b", "a"]), scope_id_for(["a", "b"]))
        with self.assertRaisesRegex(ValueError, "at most"):
            scope_id_for([f"meeting-{index}" for index in range(MAX_SCOPE_MEETINGS + 1)])

    def test_short_single_meeting_uses_full_context_and_restores_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {"m": loaded_session("m", "Design review", [row(1, "Alice owns the rollout plan.")])}
            engine = self.make_engine(Path(directory), sessions)

            result = engine.ask(["m"], "Who owns the rollout plan?")
            reopened = engine.scope(["m"])

            self.assertEqual(result["answer"]["grounding_status"], "answered")
            self.assertEqual(result["answer"]["evidence"][0]["row_id"], "row-1")
            self.assertFalse(reopened["requires_index"])
            self.assertEqual(reopened["history"], result["history"])

    def test_cross_meeting_answer_has_revision_bound_citations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {
                "m1": loaded_session("m1", "Kickoff", [row(1, "Alice will order the fabric.", "Alice", "S1")]),
                "m2": loaded_session("m2", "Follow-up", [row(1, "Bob approved the budget.", "Bob", "S2")]),
            }
            engine = self.make_engine(Path(directory), sessions)

            result = engine.ask(["m2", "m1"], "What did people commit to?")
            answer = result["answer"]

            self.assertEqual(set(answer["meeting_revisions"]), {"m1", "m2"})
            self.assertTrue(answer["evidence"])
            self.assertTrue(all(item["meeting_id"] in {"m1", "m2"} for item in answer["evidence"]))

    def test_invalid_llm_citation_is_rejected_as_not_established(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {"m": loaded_session("m", "Meeting", [row(1, "The deadline is Tuesday.")])}
            engine = self.make_engine(Path(directory), sessions, InvalidCitationClient)

            answer = engine.ask(["m"], "When is the deadline?")["answer"]

            self.assertEqual(answer["grounding_status"], "not_established")
            self.assertEqual(answer["evidence"], [])

    def test_exact_transcript_match_recovers_from_model_false_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {"m": loaded_session("m", "Philomena Cunk", [
                row(1, "The orchestra begins the next section."),
                row(2, "Was Beethoven good at music?"),
                row(3, "Beethoven wrote that song that goes da-da-da-dum."),
                row(4, "The presenter changes the subject."),
            ])}
            client = FalseNegativeClient()
            events: list[dict[str, object]] = []
            engine = self.make_engine(Path(directory), sessions, lambda: client)

            answer = engine.ask(
                ["m"],
                "What is said about Beethoven?",
                progress=events.append,
            )["answer"]

            self.assertEqual(answer["grounding_status"], "answered")
            self.assertIn("Beethoven", answer["text"])
            self.assertTrue(answer["evidence"])
            self.assertTrue(all("Beethoven" in item["quote"] for item in answer["evidence"]))
            self.assertEqual(len(client.answer_payloads), 2)
            self.assertTrue(client.answer_payloads[0]["exact_match_evidence_ids"])
            self.assertIn(88, [event.get("percent") for event in events])
            self.assertIn(94, [event.get("percent") for event in events])

    def test_scope_summary_distinguishes_same_titled_meetings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = loaded_session("m", "Repeated title", [row(1, "A transcript row.")])
            first["summary"].update({"started_at": "2026-07-13T13:57:00+02:00", "speaker_count": 4})  # type: ignore[union-attr]
            engine = self.make_engine(Path(directory), {"m": first})

            meeting = engine.scope(["m"])["meetings"][0]

            self.assertEqual(meeting["started_at"], "2026-07-13T13:57:00+02:00")
            self.assertEqual(meeting["speaker_count"], 4)
            self.assertGreater(meeting["duration_seconds"], 0)

    def test_unknown_speaker_cannot_ground_a_claim_about_a_trained_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {"m": loaded_session("m", "Press conference", [
                row(1, "An unrelated person discusses catering.", "Unknown", "UNKNOWN"),
                row(2, "Rates may remain higher for longer.", "Jerome Powell", "trained-powell"),
            ])}
            engine = self.make_engine(Path(directory), sessions, WrongSpeakerCitationClient)

            answer = engine.ask(["m"], "What did Jerome Powell say?")["answer"]

            self.assertEqual(answer["grounding_status"], "not_established")
            self.assertEqual(answer["evidence"], [])

    def test_async_live_answer_retains_full_transcript_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions = {"m": loaded_session("m", "Live", [
                row(1, "First finalized statement."),
                row(2, "Latest finalized statement."),
            ])}
            manager = MeetingChatJobManager(self.make_engine(Path(directory), sessions))
            try:
                job = manager.submit(["m"], "What was said?", provisional=True)
                deadline = time.monotonic() + 5
                current = manager.get(job["job_id"])
                while current["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
                    time.sleep(0.02)
                    current = manager.get(job["job_id"])
                self.assertEqual(current["status"], "succeeded")
                answer = current["result"]["answer"]
                self.assertTrue(answer["provisional"])
                self.assertEqual(answer["transcript_end_seconds"], 28.0)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
