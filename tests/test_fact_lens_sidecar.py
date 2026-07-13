from __future__ import annotations

import sys
import queue
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.fact_lens_sidecar import (
    SCHEMA_VERSION,
    ClaimExtractionWorker,
    ExtractedClaim,
    ExtractionResult,
    FactLensRuntime,
    SidecarState,
    TranscriptSentence,
    build_arg_parser,
    coalesce_sentences,
    evidence_matches_transcript,
    parse_openai_chat_json,
    parse_sse_lines,
    validate_extraction_payload,
)


class FactLensParsingTests(unittest.TestCase):
    def test_llm_claim_extraction_is_disabled_by_default(self) -> None:
        args = build_arg_parser().parse_args([])

        self.assertFalse(args.enable_llm)
        self.assertFalse(args.mock_llm)

    def test_openai_json_parser_accepts_fenced_json(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"schema_version":"claim_triage_v1","sentence_id":"7",'
                            '"classification":"ignore","claims":[],"rationale":"small talk"}'
                            "\n```"
                        )
                    }
                }
            ]
        }

        payload = parse_openai_chat_json(response)

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["classification"], "ignore")

    def test_validate_extraction_rejects_claim_when_evidence_is_not_in_transcript(self) -> None:
        sentence = TranscriptSentence(
            id="42",
            text="Berlin has more than three million residents.",
            speaker="S1",
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sentence_id": "42",
            "classification": "checkable_claim",
            "rationale": "contains a population claim",
            "claims": [
                {
                    "claim": "Paris has fewer than one million residents.",
                    "evidence": "Paris has fewer than one million residents",
                    "priority": "high",
                    "rationale": "population claim",
                }
            ],
        }

        result = validate_extraction_payload(payload, sentence)

        self.assertEqual(result.classification, "needs_context")
        self.assertEqual(result.claims, [])
        self.assertEqual(result.rejected_claims, ["claim_0:evidence_mismatch"])

    def test_validate_extraction_deduplicates_repeated_claims(self) -> None:
        sentence = TranscriptSentence(
            id="43",
            text="The Eiffel Tower is located in Paris, France.",
            speaker="S1",
        )
        claim = {
            "claim": "The Eiffel Tower is located in Paris, France.",
            "evidence": "The Eiffel Tower is located in Paris, France.",
            "priority": "high",
            "rationale": "location claim",
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sentence_id": "43",
            "classification": "checkable_claim",
            "rationale": "contains a location claim",
            "claims": [dict(claim), dict(claim)],
        }

        result = validate_extraction_payload(payload, sentence)

        self.assertEqual(result.classification, "checkable_claim")
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.rejected_claims, ["claim_1:duplicate_claim"])

    def test_evidence_matching_allows_near_exact_punctuation_differences(self) -> None:
        self.assertTrue(
            evidence_matches_transcript(
                "Berlin has more than three million residents",
                "Berlin has more than three million residents.",
            )
        )
        self.assertFalse(
            evidence_matches_transcript(
                "Paris has fewer than one million residents",
                "Berlin has more than three million residents.",
            )
        )

    def test_coalesce_sentences_keeps_latest_duplicate_in_latest_position(self) -> None:
        first = TranscriptSentence(id="1", text="old")
        second = TranscriptSentence(id="2", text="middle")
        updated = TranscriptSentence(id="1", text="new")

        coalesced = coalesce_sentences([first, second, updated])

        self.assertEqual([(item.id, item.text) for item in coalesced], [("2", "middle"), ("1", "new")])

    def test_record_sentence_can_observe_without_queueing_llm_work(self) -> None:
        state = SidecarState()
        sentence = TranscriptSentence(id="1", text="Berlin has more than three million residents.", speaker="S1")

        queued = state.record_sentence(sentence, queue_claim_extraction=False)
        snapshot = state.snapshot()

        self.assertFalse(queued)
        self.assertEqual(snapshot["stats"]["sentences_seen"], 1)
        self.assertEqual(snapshot["stats"]["sentences_queued"], 0)
        self.assertEqual(snapshot["cards"], [])

    def test_sse_parser_handles_event_id_data_and_heartbeat(self) -> None:
        events = list(
            parse_sse_lines(
                [
                    ": heartbeat\n",
                    "id: 11\n",
                    "event: transcript.final\n",
                    'data: {"type":"transcript.final"}\n',
                    "\n",
                    "event: message\n",
                    "data: first\n",
                    "data: second\n",
                    "\n",
                ]
            )
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_id, "11")
        self.assertEqual(events[0].event, "transcript.final")
        self.assertEqual(events[0].data, '{"type":"transcript.final"}')
        self.assertEqual(events[1].data, "first\nsecond")

    def test_sentence_replacement_removes_old_cards_and_rejects_stale_completion(self) -> None:
        state = SidecarState()
        try:
            old = TranscriptSentence(id="same", text="Berlin has three million residents.", speaker="S1")
            old_token = state.record_sentence(old)
            self.assertIsNotNone(old_token)
            old_result = ExtractionResult(
                sentence_id="same",
                classification="checkable_claim",
                rationale="claim",
                claims=[ExtractedClaim(old.text, old.text)],
            )
            self.assertTrue(state.apply_extraction(old, old_result, old_token))

            current = TranscriptSentence(id="same", text="Berlin has four million residents.", speaker="S1")
            current_token = state.record_sentence(current)
            self.assertIsNotNone(current_token)
            cards = state.snapshot()["cards"]
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0]["transcript_text"], current.text)

            self.assertFalse(state.apply_extraction(old, old_result, old_token))
            self.assertEqual(state.snapshot()["cards"][0]["transcript_text"], current.text)
        finally:
            state.close()

    def test_queue_eviction_marks_dropped_current_sentence_terminal(self) -> None:
        state = SidecarState()
        stop = threading.Event()
        work_queue = queue.Queue(maxsize=1)
        worker = ClaimExtractionWorker(
            state=state,
            client=object(),
            work_queue=work_queue,
            stop_event=stop,
            debounce_seconds=0,
        )
        try:
            first = TranscriptSentence(id="first", text="First checkable sentence.")
            second = TranscriptSentence(id="second", text="Second checkable sentence.")
            first_token = state.record_sentence(first)
            second_token = state.record_sentence(second)
            worker.submit(first, first_token)
            worker.submit(second, second_token)

            first_card = next(card for card in state.snapshot()["cards"] if card["sentence_id"] == "first")
            self.assertEqual(first_card["status"], "needs_context")
            self.assertIn("capacity", first_card["error"])
            self.assertEqual(state.snapshot()["stats"]["queue_drops"], 1)
        finally:
            worker.stop(timeout=0.1)
            state.close()

    def test_snapshot_publisher_coalesces_without_losing_monotonic_revision(self) -> None:
        state = SidecarState()
        subscriber = state.subscribe()
        try:
            initial = subscriber.get(timeout=1)
            state.set_source_status("connecting")
            state.set_source_status("connected")
            state.set_llm_status("ready")
            deadline = time.monotonic() + 2
            payloads = [initial]
            while time.monotonic() < deadline:
                try:
                    payloads.append(subscriber.get(timeout=0.05))
                except queue.Empty:
                    pass
                if '"revision": 3' in payloads[-1]:
                    break
            revisions = [__import__("json").loads(payload)["revision"] for payload in payloads]
            self.assertEqual(revisions, sorted(revisions))
            self.assertEqual(revisions[-1], 3)
        finally:
            state.unsubscribe(subscriber)
            state.close()

    def test_runtime_shutdown_joins_reader_and_is_idempotent(self) -> None:
        state = SidecarState()
        stop = threading.Event()
        reader = threading.Thread(target=stop.wait, name="fact-lens-test-reader")
        runtime = FactLensRuntime(state=state, stop_event=stop, reader=reader, worker=None)

        runtime.start()
        runtime.close(reader_timeout=1)
        runtime.close(reader_timeout=1)

        self.assertFalse(reader.is_alive())


if __name__ == "__main__":
    unittest.main()
