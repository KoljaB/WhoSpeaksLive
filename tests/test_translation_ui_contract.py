from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from window.window_gui_html import HTML


class TranslationUiContractTests(unittest.TestCase):
    def test_translation_controls_are_compact_and_support_all_display_modes(self) -> None:
        self.assertIn('id="translationMenuButton" class="translation-menu-button"', HTML)
        self.assertIn('id="translationDisplayMode"', HTML)
        self.assertIn('<option value="original">Original</option>', HTML)
        self.assertIn('<option value="single">One translation</option>', HTML)
        self.assertIn('<option value="all">All translations</option>', HTML)
        self.assertIn('id="translationIncludeOriginal"', HTML)
        self.assertIn('id="translationTargetList" class="translation-target-list"', HTML)

    def test_translation_state_is_derived_without_replacing_the_original(self) -> None:
        render_start = HTML.index("function renderSentence(item)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        self.assertIn("row.dataset.text = displayText;", render_block)
        self.assertIn("row.dataset.sourceTextHash", render_block)
        self.assertIn("row.appendChild(translationLines);", render_block)
        self.assertIn("showPendingSource", HTML)
        self.assertNotIn("row.dataset.text = state.text", HTML)
        self.assertIn("translationStateMatchesRow(state, row)", HTML)

    def test_translation_events_and_saved_session_backfill_share_one_path(self) -> None:
        self.assertIn('const translationConfig = __TRANSLATION_JSON__;', HTML)
        self.assertIn('es.addEventListener("translation", e => applyTranslationEvent(JSON.parse(e.data)));', HTML)
        self.assertIn("item.sentence_index, item.segment_id, item.index", HTML)
        self.assertIn("applyTranslationCollection(sessionData.translations, {refresh:true});", HTML)
        self.assertIn("source_revision", HTML)
        self.assertIn("source_text_hash", HTML)
        self.assertIn("item.source_text_hash || item.source_hash", HTML)
        self.assertIn("item.translated_text", HTML)
        self.assertIn("This language was not translated during the saved session.", HTML)

    def test_search_and_exports_include_translation_without_losing_source_text(self) -> None:
        self.assertIn("const translatedSearchText = Array.from(translationSelectedTargets)", HTML)
        self.assertIn("row.dataset.searchText = [sourceSearchText, ...translatedSearchText]", HTML)
        self.assertIn("translations: transcriptTranslationExportStates(row)", HTML)
        self.assertIn("translation_display:", HTML)
        self.assertIn("if (effectiveTranslationDisplayMode() === \"original\")", HTML)
        self.assertIn("return rows.map(row => `[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`)", HTML)

    def test_target_changes_use_the_translation_configuration_endpoint(self) -> None:
        self.assertIn('post("/api/translation/configure", {target_languages:targetLanguages})', HTML)
        self.assertIn("translationMaximumTargets()", HTML)
        self.assertIn('!hadTargets && translationDisplayMode === "original"', HTML)
        self.assertIn("translationPrimaryTargetStorageKey", HTML)
        self.assertIn("translationDisplayModeStorageKey", HTML)

    def test_provider_license_metadata_is_visible_in_the_translation_menu(self) -> None:
        self.assertIn("function translationProviderLicense()", HTML)
        self.assertIn("provider.model_metadata", HTML)
        self.assertIn("licenseLabel", HTML)
        self.assertIn("translationProvider.title = translationProviderNotice();", HTML)

    def test_all_completed_translations_use_the_same_text_size_and_show_flags(self) -> None:
        self.assertIn(".translation-line { min-width:0; color:#E8EEF5; font-size:15px;", HTML)
        self.assertIn(".translation-line.translation-additional { color:#D7DEE8; }", HTML)
        self.assertNotIn(".translation-line.translation-additional { color:#D7DEE8; font-size", HTML)
        self.assertIn("flag_url:flagUrl", HTML)
        self.assertIn("function translationLanguageFlagUrl(code)", HTML)
        self.assertIn('flag.className = "translation-language-flag";', HTML)
        self.assertIn('flag.setAttribute("aria-hidden", "true");', HTML)

    def test_chrome_translator_uses_feature_detection_and_backend_fallback(self) -> None:
        self.assertIn("globalThis.Translator", HTML)
        self.assertIn("modern.availability(options)", HTML)
        self.assertIn("globalThis.translation", HTML)
        self.assertIn('post("/api/translation/browser-result", completed)', HTML)
        self.assertIn('post("/api/translation/browser-fallback", {', HTML)
        self.assertIn("browserTranslationQueue", HTML)
        self.assertIn("Chrome and backend translation failed", HTML)

    def test_google_provider_keeps_required_attribution_adjacent_to_controls(self) -> None:
        self.assertIn('id="translationProviderAttribution"', HTML)
        self.assertIn('translationProviderId() === "google_cloud"', HTML)
        self.assertIn("Powered by Google Translate", HTML)


if __name__ == "__main__":
    unittest.main()
