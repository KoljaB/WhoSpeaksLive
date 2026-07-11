from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.session_store import SessionStore


class TranslationPersistenceTests(unittest.TestCase):
    def test_create_session_initializes_empty_translation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)

            summary = store.create_session(session_id="translation-draft")
            opened = store.open_session("translation-draft")

            translations_doc = json.loads(
                (root / "translation-draft" / "translations.json").read_text(encoding="utf-8")
            )
            self.assertEqual(translations_doc["translations"], [])
            self.assertEqual(opened["translations"], [])
            self.assertEqual(opened["manifest"]["paths"]["translations"], "translations.json")
            self.assertFalse(summary["has_translations"])
            self.assertEqual(summary["translation_count"], 0)

    def test_save_snapshot_round_trips_translation_records(self) -> None:
        translations = [
            {
                "segment_id": "sentence-1",
                "source_revision": "7",
                "source_language": "es",
                "target_language": "de",
                "text": "Guten Morgen.",
                "provider": "translate-gemma",
                "status": "completed",
            },
            {
                "segment_id": "sentence-1",
                "source_revision": "7",
                "source_language": "es",
                "target_language": "ja",
                "text": "おはようございます。",
                "provider": "translate-gemma",
                "status": "completed",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)

            summary = store.save_snapshot(
                {
                    "id": "translated-session",
                    "transcript_rows": [],
                    "speaker_state": {"speakers": []},
                    "translations": translations,
                }
            )
            opened = store.open_session("translated-session")

            self.assertEqual(opened["translations"], translations)
            self.assertTrue(summary["has_translations"])
            self.assertEqual(summary["translation_count"], 2)
            self.assertEqual(opened["manifest"]["translation_count"], 2)
            persisted = json.loads(
                (root / "translated-session" / "translations.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["translations"], translations)

    def test_open_legacy_missing_or_corrupt_translation_artifact_defaults_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)
            store.create_session(session_id="legacy-session")
            session_dir = root / "legacy-session"

            (session_dir / "translations.json").unlink()
            manifest_path = session_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["paths"].pop("translations", None)
            manifest.pop("has_translations", None)
            manifest.pop("translation_count", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            missing = store.open_session("legacy-session")
            self.assertEqual(missing["translations"], [])
            self.assertEqual(missing["manifest"]["paths"]["translations"], "translations.json")
            self.assertFalse(missing["manifest"]["has_translations"])
            self.assertEqual(missing["manifest"]["translation_count"], 0)

            (session_dir / "translations.json").write_text("{not-json", encoding="utf-8")
            corrupt = store.open_session("legacy-session")
            self.assertEqual(corrupt["translations"], [])
            self.assertFalse(corrupt["summary"]["has_translations"])
            self.assertEqual(corrupt["summary"]["translation_count"], 0)


if __name__ == "__main__":
    unittest.main()
