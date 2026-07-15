from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from speakers.person_library import PersonLibrary


class PersonMatchingValidationTests(unittest.TestCase):
    def test_cross_condition_fixture_keeps_precision_first_unknown_contract(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "person_voice_matching_cases.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            for person_data in fixture["people"]:
                person = library.create_person(person_data["name"])
                for index, sample in enumerate(person_data["samples"]):
                    library.add_meeting_sample(
                        person["id"],
                        sample["vector"],
                        embedding_provider=fixture["provider"],
                        session_id=f"{person_data['name']}-{index}",
                        capture_condition=sample["condition"],
                        sentence_count=4,
                        speech_seconds=12.0,
                    )
            for case in fixture["cases"]:
                with self.subTest(case=case["name"]):
                    match = library.match(
                        case["vector"],
                        embedding_provider=fixture["provider"],
                        min_similarity=0.80,
                        min_margin=0.04,
                    )
                    self.assertEqual(None if match is None else match.name, case["expected"])

    def test_similar_people_are_unknown_when_competitor_margin_is_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = PersonLibrary(Path(tmp) / "people.json")
            alice = library.create_person("Alice")
            bob = library.create_person("Bob")
            library.add_meeting_sample(alice["id"], [1.0, 0.0], embedding_provider="mock", session_id="a")
            library.add_meeting_sample(bob["id"], [0.98, 0.2], embedding_provider="mock", session_id="b")
            self.assertIsNone(library.match(
                [0.995, 0.1],
                embedding_provider="mock",
                min_similarity=0.8,
                min_margin=0.04,
            ))


if __name__ == "__main__":
    unittest.main()
