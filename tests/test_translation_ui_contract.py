from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.web_asset_support import HTML


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
        render_start = HTML.index("function renderSentenceImmediate(item, rowOverride = null)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        self.assertIn("row.dataset.text = displayText;", render_block)
        self.assertIn("row.dataset.sourceTextHash", render_block)
        self.assertIn("row.appendChild(translationLines);", render_block)
        self.assertIn("showPendingSource", HTML)
        self.assertNotIn("row.dataset.text = state.text", HTML)
        self.assertIn("translationStateMatchesRow(state, row)", HTML)

    def test_translation_events_and_saved_session_backfill_share_one_path(self) -> None:
        self.assertIn('const translationConfig = bootstrap.translation || {};', HTML)
        self.assertIn('ctx.owners.capture.es.addEventListener("translation", e => applyTranslationEvent(JSON.parse(e.data)));', HTML)
        self.assertIn("item.sentence_index, item.segment_id, item.index", HTML)
        self.assertIn("applyTranslationCollection(sessionData.translations, {refresh:true});", HTML)
        self.assertIn("source_revision", HTML)
        self.assertIn("source_text_hash", HTML)
        self.assertIn("item.source_text_hash || item.source_hash", HTML)
        self.assertIn("item.translated_text", HTML)
        self.assertIn("This language was not translated during the saved session.", HTML)

    def test_search_and_exports_include_translation_without_losing_source_text(self) -> None:
        self.assertIn("const translatedSearchText = Array.from(ctx.owners.translation.translationSelectedTargets)", HTML)
        self.assertIn("row.dataset.searchText = [sourceSearchText, ...translatedSearchText]", HTML)
        self.assertIn("translations: transcriptTranslationExportStates(row)", HTML)
        self.assertIn("translation_display:", HTML)
        self.assertIn("if (effectiveTranslationDisplayMode() === \"original\")", HTML)
        self.assertIn("return rows.map(row => `[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`)", HTML)

    def test_target_changes_use_the_translation_configuration_endpoint(self) -> None:
        self.assertIn('post("/api/translation/configure", {target_languages:targetLanguages})', HTML)
        self.assertIn("translationMaximumTargets()", HTML)
        self.assertIn('!hadTargets && ctx.owners.translation.translationDisplayMode === "original"', HTML)
        self.assertIn("translationPrimaryTargetStorageKey", HTML)
        self.assertIn("translationDisplayModeStorageKey", HTML)

    def test_provider_license_metadata_is_visible_in_the_translation_menu(self) -> None:
        self.assertIn("function translationProviderLicense()", HTML)
        self.assertIn("provider.model_metadata", HTML)
        self.assertIn("licenseLabel", HTML)
        self.assertIn("translationProvider.title = translationProviderNotice();", HTML)

    def test_translation_text_style_stays_stable_across_states(self) -> None:
        self.assertIn(".translation-line { min-width:0; color:#E8EEF5; font-size:15px;", HTML)
        self.assertIn(".translation-line.translation-additional { color:#E8EEF5; }", HTML)
        self.assertIn(".translation-line.translation-pending, .translation-line.translation-error { color:#E8EEF5; font-size:15px; }", HTML)
        self.assertIn(".text { color:#E8EEF5; font-size:15px; line-height:1.34; }", HTML)
        self.assertIn(".text.translation-secondary { color:#E8EEF5; font-size:15px; line-height:1.34;", HTML)
        self.assertNotIn(".text.translation-secondary::before", HTML)

    def test_language_labels_support_flag_name_and_code_combinations(self) -> None:
        self.assertIn('id="translationLanguageLabelMode"', HTML)
        self.assertIn('<option value="flag">Flag only</option>', HTML)
        self.assertIn('<option value="flag_name">Flag + full name</option>', HTML)
        self.assertIn('<option value="name">Full name only</option>', HTML)
        self.assertIn('<option value="flag_code">Flag + code</option>', HTML)
        self.assertIn('<option value="code">Code only</option>', HTML)
        self.assertIn('translationLanguageLabelMode: "flag_name",', HTML)
        self.assertIn("translationLanguageLabelModeStorageKey", HTML)
        self.assertIn("translationLanguageCodeLabel(languageCode)", HTML)
        self.assertIn("source.prepend(createTranslationLanguageLabel(languageConfig.code));", HTML)
        self.assertIn("flag_url:flagUrl", HTML)
        self.assertIn("function translationLanguageFlagUrl(code)", HTML)
        self.assertIn('flag.className = "translation-language-flag";', HTML)
        self.assertIn('flag.setAttribute("aria-hidden", "true");', HTML)
        self.assertIn("normalized === normalizedTranslationLanguageCode(languageConfig.code)", HTML)
        self.assertIn('return String(languageConfig.flag_url || "");', HTML)
        self.assertIn("return String(languageConfig.name || normalized);", HTML)

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
