from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from window.report_templates import (
    STANDARD_TEMPLATE_ID,
    TEMPLATE_SCHEMA_VERSION,
    ReportTemplateStore,
    builtin_report_templates,
    get_builtin_report_template,
    slugify_template_id,
    template_revision_hash,
    validate_report_template,
)


def custom_template_payload() -> dict[str, object]:
    return {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "template_id": "custom.daily-sync",
        "name": "Daily sync",
        "description": "A focused daily report.",
        "version": 1,
        "builtin": False,
        "language_mode": "inherit",
        "privacy_policy": "inherit",
        "sections": [
            {
                "key": "action_items",
                "title": "Action items",
                "objective": "Find work that was explicitly assigned.",
                "max_items": 8,
                "evidence_required": True,
                "render_kind": "table",
                "sort_order": "chronological",
                "output_fields": [
                    {
                        "key": "owner",
                        "label": "Owner",
                        "type": "speaker",
                        "description": "The assigned speaker.",
                        "options": [],
                    },
                    {
                        "key": "status",
                        "label": "Status",
                        "type": "enum",
                        "description": "Assignment status.",
                        "options": ["assigned", "proposed"],
                    },
                ],
            }
        ],
    }


class BundledReportTemplateTests(unittest.TestCase):
    def test_discovers_exactly_standard_plus_ten_use_case_presets(self) -> None:
        templates = builtin_report_templates()

        self.assertEqual(len(templates), 11)
        self.assertEqual(len({item["template_id"] for item in templates}), 11)
        self.assertEqual(templates[0]["template_id"], STANDARD_TEMPLATE_ID)
        self.assertTrue(all(item["builtin"] for item in templates))
        self.assertTrue(all(item["schema_version"] == TEMPLATE_SCHEMA_VERSION for item in templates))

    def test_presets_are_json_documents_that_pass_the_public_validator(self) -> None:
        preset_directory = SRC / "window" / "report_template_presets"
        preset_paths = sorted(preset_directory.glob("*.json"))

        self.assertEqual(len(preset_paths), 11)
        for path in preset_paths:
            with self.subTest(path=path.name):
                raw = json.loads(path.read_text(encoding="utf-8"))
                normalized = validate_report_template(raw, allow_builtin=True)
                self.assertEqual(normalized["revision_hash"], template_revision_hash(normalized))

    def test_standard_preset_rebuilds_every_existing_v2_section_as_flat_sections(self) -> None:
        standard = get_builtin_report_template(STANDARD_TEMPLATE_ID)

        self.assertIsNotNone(standard)
        self.assertEqual(
            [section["key"] for section in standard["sections"]],
            [
                "speaker_map",
                "executive_summary",
                "structured_brief",
                "decisions",
                "action_items",
                "open_questions",
                "risks",
                "discussion_threads",
                "disagreements",
                "deadlines",
                "speaker_participation",
                "ask_this_meeting",
            ],
        )
        self.assertFalse(any("sections" in section for section in standard["sections"]))

    def test_use_case_presets_cover_every_requested_report_part(self) -> None:
        required_sections = {
            "builtin.german-works-council": {
                "discussed_issues", "decisions", "unresolved_disagreements",
                "action_items", "responsibilities",
            },
            "builtin.english-podcast-production": {
                "episode_summary", "chapter_markers", "notable_quotes", "fact_check_candidates",
            },
            "builtin.french-medical-case-conference": {
                "discussed_cases", "proposed_next_steps", "open_questions",
                "responsibilities", "decisions",
            },
            "builtin.italian-film-production": {
                "schedule_changes", "scene_specific_decisions", "equipment_requirements",
                "continuity_concerns", "responsibilities", "filming_delay_risks",
            },
            "builtin.hebrew-cybersecurity-incident": {
                "incident_timeline", "observations", "hypotheses", "decisions",
                "assigned_actions", "unresolved_risks",
            },
            "builtin.dutch-business-mediation": {
                "party_positions", "agreed_facts", "disputed_claims", "proposals",
                "concessions", "tentative_agreements", "unresolved_issues",
            },
            "builtin.portuguese-investigative-newsroom": {
                "story_hypotheses", "source_information", "research_assignments",
                "verification_tasks", "legal_ethical_concerns", "deadlines",
                "insufficient_evidence_claims",
            },
            "builtin.swedish-qualitative-research": {
                "recurring_themes", "areas_of_agreement", "areas_of_disagreement",
                "participant_viewpoints", "representative_quotes", "unanswered_questions",
            },
            "builtin.turkish-factory-shift-handover": {
                "urgent_safety_issues", "equipment_problems", "temporary_workarounds",
                "routine_safety_concerns", "unfinished_maintenance",
                "production_interruptions", "follow_up_actions",
            },
            "builtin.spanish-municipal-committee": {
                "agenda_items", "participant_arguments", "motions", "decisions",
                "voting_outcomes", "action_items", "deadlines", "unresolved_questions",
            },
        }

        templates = {item["template_id"]: item for item in builtin_report_templates()}
        self.assertEqual(set(templates) - {STANDARD_TEMPLATE_ID}, set(required_sections))
        for template_id, required in required_sections.items():
            with self.subTest(template_id=template_id):
                actual = {section["key"] for section in templates[template_id]["sections"]}
                self.assertTrue(required <= actual)

    def test_use_case_language_and_local_only_policies_are_explicit(self) -> None:
        templates = {item["template_id"]: item for item in builtin_report_templates()}
        expected_languages = {
            "builtin.german-works-council": "de",
            "builtin.english-podcast-production": "en",
            "builtin.french-medical-case-conference": "fr",
            "builtin.italian-film-production": "it",
            "builtin.hebrew-cybersecurity-incident": "he",
            "builtin.dutch-business-mediation": "nl",
            "builtin.portuguese-investigative-newsroom": "pt",
            "builtin.swedish-qualitative-research": "sv",
            "builtin.turkish-factory-shift-handover": "tr",
            "builtin.spanish-municipal-committee": "es",
        }
        self.assertEqual(
            {template_id: templates[template_id]["language_mode"] for template_id in expected_languages},
            expected_languages,
        )
        for template_id in {
            "builtin.german-works-council",
            "builtin.french-medical-case-conference",
            "builtin.hebrew-cybersecurity-incident",
            "builtin.dutch-business-mediation",
            "builtin.portuguese-investigative-newsroom",
        }:
            self.assertEqual(templates[template_id]["privacy_policy"], "local_only")

    def test_builtin_get_returns_a_fresh_document(self) -> None:
        first = get_builtin_report_template(STANDARD_TEMPLATE_ID)
        first["sections"][0]["title"] = "Changed by caller"

        second = get_builtin_report_template(STANDARD_TEMPLATE_ID)

        self.assertNotEqual(second["sections"][0]["title"], "Changed by caller")
        self.assertIsNone(get_builtin_report_template("builtin.does-not-exist"))


class ReportTemplateValidationTests(unittest.TestCase):
    def test_normalizes_names_keys_language_and_defaults(self) -> None:
        payload = custom_template_payload()
        payload["template_id"] = "Custom.My Template"
        payload["name"] = "  Weekly   Review  "
        payload["language_mode"] = "German"
        payload["privacy_policy"] = "local-only"
        section = payload["sections"][0]
        section["key"] = "Action Items"
        del section["max_items"]
        del section["render_kind"]
        del section["sort_order"]

        normalized = validate_report_template(payload)

        self.assertEqual(normalized["template_id"], "custom.my-template")
        self.assertEqual(normalized["name"], "Weekly Review")
        self.assertEqual(normalized["language_mode"], "de")
        self.assertEqual(normalized["privacy_policy"], "local_only")
        self.assertEqual(normalized["sections"][0]["key"], "action_items")
        self.assertEqual(normalized["sections"][0]["max_items"], 8)
        self.assertEqual(normalized["sections"][0]["render_kind"], "cards")
        self.assertEqual(normalized["sections"][0]["sort_order"], "relevance")
        self.assertEqual(normalized["revision_hash"], template_revision_hash(normalized))

    def test_revision_hash_is_deterministic_and_ignores_supplied_hash(self) -> None:
        first = validate_report_template(custom_template_payload())
        tampered = deepcopy(first)
        tampered["revision_hash"] = "not-the-real-hash"

        second = validate_report_template(tampered)

        self.assertEqual(first["revision_hash"], second["revision_hash"])
        changed = deepcopy(first)
        changed["sections"][0]["objective"] = "A materially different objective."
        self.assertNotEqual(first["revision_hash"], template_revision_hash(changed))

    def test_rejects_invalid_structure_and_reserved_builtin_state(self) -> None:
        invalid_payloads: list[dict[str, object]] = []

        empty_sections = custom_template_payload()
        empty_sections["sections"] = []
        invalid_payloads.append(empty_sections)

        duplicate_sections = custom_template_payload()
        duplicate_sections["sections"].append(deepcopy(duplicate_sections["sections"][0]))
        invalid_payloads.append(duplicate_sections)

        too_many_sections = custom_template_payload()
        original_section = too_many_sections["sections"][0]
        too_many_sections["sections"] = [
            {**deepcopy(original_section), "key": f"section_{index}"} for index in range(17)
        ]
        invalid_payloads.append(too_many_sections)

        bad_max_items = custom_template_payload()
        bad_max_items["sections"][0]["max_items"] = 21
        invalid_payloads.append(bad_max_items)

        enum_without_options = custom_template_payload()
        enum_without_options["sections"][0]["output_fields"][1]["options"] = []
        invalid_payloads.append(enum_without_options)

        unknown_field = custom_template_payload()
        unknown_field["mystery"] = True
        invalid_payloads.append(unknown_field)

        builtin = custom_template_payload()
        builtin["template_id"] = "builtin.custom"
        builtin["builtin"] = True
        invalid_payloads.append(builtin)

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_report_template(payload)

    def test_slugify_template_id_is_stable_and_safe(self) -> None:
        self.assertEqual(slugify_template_id("  My Öperations Report! "), "my-operations-report")
        self.assertEqual(slugify_template_id("日本語"), "report")


class ReportTemplateStoreTests(unittest.TestCase):
    def test_persists_custom_templates_and_increments_versions_on_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReportTemplateStore(Path(directory))
            draft = custom_template_payload()
            draft.pop("template_id")

            saved = store.save_template(draft)

            self.assertEqual(saved["template_id"], "custom.daily-sync")
            self.assertEqual(saved["version"], 1)
            self.assertEqual(len(store.list_templates()), 12)
            self.assertTrue((Path(directory) / "custom.daily-sync.json").is_file())

            update = deepcopy(saved)
            update["description"] = "Updated description."
            updated = store.save_template(update)

            self.assertEqual(updated["version"], 2)
            self.assertNotEqual(updated["revision_hash"], saved["revision_hash"])
            reopened = ReportTemplateStore(Path(directory))
            self.assertEqual(reopened.get_template("custom.daily-sync"), updated)

    def test_clone_builtin_then_delete_custom_without_mutating_preset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReportTemplateStore(Path(directory))

            clone = store.clone_template(STANDARD_TEMPLATE_ID, "My Standard")
            second_clone = store.clone_template(STANDARD_TEMPLATE_ID, "My Standard")

            self.assertEqual(clone["template_id"], "custom.my-standard")
            self.assertEqual(second_clone["template_id"], "custom.my-standard-2")
            self.assertFalse(clone["builtin"])
            self.assertEqual(clone["version"], 1)
            self.assertEqual(len(clone["sections"]), 12)
            self.assertTrue(store.delete_template(clone["template_id"]))
            self.assertFalse(store.delete_template(clone["template_id"]))
            self.assertIsNotNone(store.get_template(STANDARD_TEMPLATE_ID))

            with self.assertRaises(ValueError):
                store.delete_template(STANDARD_TEMPLATE_ID)
            with self.assertRaises(ValueError):
                store.save_template(get_builtin_report_template(STANDARD_TEMPLATE_ID))
            with self.assertRaises(ValueError):
                store.clone_template("custom.missing", "Missing")

    def test_concurrent_updates_serialize_version_assignment_and_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReportTemplateStore(Path(directory))
            initial = store.save_template(custom_template_payload())

            def update(index: int) -> dict[str, object]:
                draft = deepcopy(initial)
                draft["description"] = f"Concurrent update {index}"
                return store.save_template(draft)

            with ThreadPoolExecutor(max_workers=8) as executor:
                saved = list(executor.map(update, range(8)))

            self.assertEqual(sorted(item["version"] for item in saved), list(range(2, 10)))
            self.assertEqual(store.get_template(initial["template_id"])["version"], 9)
            self.assertFalse(list(Path(directory).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
