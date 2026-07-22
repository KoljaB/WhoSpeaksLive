from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.web_asset_support import HTML





class WindowWebBehaviorContractTests(unittest.TestCase):
    def test_live_provisional_speakers_are_transient_and_reconciled(self) -> None:
        self.assertIn("function isLiveProvisionalSpeaker(speaker)", HTML)
        self.assertIn('isLiveProvisionalSpeaker(label) ? "Matching new voice..." : "Unknown"', HTML)
        self.assertIn('? "Comparing with detected speakers..."', HTML)
        self.assertIn(
            'if (internalLabel === label && isLiveProvisionalSpeaker(label)) return "#8F9BA8";',
            HTML,
        )
        self.assertIn("speakers.filter(speaker => !isLiveProvisionalSpeaker(speaker)).length", HTML)
        self.assertIn(".speaker-item.provisional-speaker:not(.live-speaker) { display:none; }", HTML)
        self.assertIn('summary.setAttribute("role", isLiveProvisional ? "status" : "button");', HTML)
        self.assertIn("item.replaces_speaker_id", HTML)

    def test_suggested_identity_keeps_the_canonical_speaker_number(self) -> None:
        self.assertIn('return `Speaker ${Number(match[1])}`;', HTML)
        self.assertNotIn('return `Speaker ${Number(match[1]) + 1}`;', HTML)

    def test_speaker_label_is_inserted_as_text_not_markup(self) -> None:
        self.assertNotIn("${speakerLabel}</span>", HTML)
        self.assertIn("speakerBadge.textContent = speakerLabel;", HTML)
        self.assertIn("row.replaceChildren(top, text);", HTML)

    def test_transcript_turn_grouping_defaults_to_enabled(self) -> None:
        self.assertIn('id="groupTranscriptTurns" type="checkbox" checked', HTML)
        self.assertIn('const transcriptGroupTurnsStorageKey = "whospeaks.demo.group_transcript_turns.v3";', HTML)
        self.assertIn("groupTranscriptTurns.checked = storedBooleanValue(transcriptGroupTurnsStorageKey, true);", HTML)
        self.assertNotIn('groupTranscriptTurns.dispatchEvent(new Event("change"));', HTML)

    def test_saved_sessions_support_bulk_selection_and_actions(self) -> None:
        self.assertIn('id="selectAllSessions"', HTML)
        self.assertIn('id="unselectAllSessions"', HTML)
        self.assertIn('id="archiveSelectedSessions"', HTML)
        self.assertIn('id="restoreSelectedSessions"', HTML)
        self.assertIn('id="deleteSelectedSessions"', HTML)
        self.assertIn('selector.type = "checkbox";', HTML)
        self.assertIn("selectedSavedSessionIds: new Set(),", HTML)
        self.assertIn("async function bulkSavedSessionAction(action)", HTML)
        self.assertIn('bulkSavedSessionAction("archive")', HTML)
        self.assertIn('bulkSavedSessionAction("restore")', HTML)
        self.assertIn('bulkSavedSessionAction("delete")', HTML)

    def test_meeting_intelligence_ui_contract_is_present(self) -> None:
        self.assertIn('data-speaker-tab="intelligence"', HTML)
        self.assertIn('id="meetingIntelligenceGenerate"', HTML)
        self.assertIn("function generateMeetingIntelligenceReport()", HTML)
        self.assertIn('post("/api/meeting-intelligence/generate"', HTML)
        self.assertIn('post("/api/meeting-intelligence/update-object"', HTML)
        self.assertIn("row.classList.add(\"meeting-evidence-row\")", HTML)
        self.assertIn("setMeetingIntelligenceReport(sessionData.meeting_intelligence || null);", HTML)

    def test_meeting_intelligence_api_routes_are_registered(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                SRC / "window" / "live_http_handler.py",
                SRC / "window" / "live_window_server.py",
            )
        )
        self.assertIn('path == "/api/meeting-intelligence/report"', source)
        self.assertIn('path == "/api/meeting-intelligence/generate"', source)
        self.assertIn('path == "/api/meeting-intelligence/update-object"', source)
        self.assertIn("def generate_meeting_intelligence(self, session_id: str)", source)

    def test_revised_sentence_refreshes_speaker_counts(self) -> None:
        self.assertIn("renderedSpeakerSentenceCounts: {},", HTML)
        self.assertIn("renderedSpeakerSpeakingSeconds: {},", HTML)
        self.assertIn('currentLiveSpeakerId: "",', HTML)
        self.assertIn('transcriptLiveSpeakerId: "",', HTML)
        self.assertIn('fallbackLiveSpeakerId: "",', HTML)
        self.assertIn("liveSpeakerTimeline: [],", HTML)
        self.assertIn("fastSpeakerPanelStats: {},", HTML)
        self.assertIn("hasRenderedFinalSentenceRows: false,", HTML)
        self.assertIn("function dominantRealtimeSpeakerId(start, end)", HTML)
        self.assertIn("function realtimeDominanceScoredEnd(start, end)", HTML)
        self.assertIn("const tailSeconds = Math.min(3, Math.max(2, duration * 0.25));", HTML)
        self.assertIn("const requiredSeconds = Math.max(0.3, scoredSeconds * 0.5);", HTML)
        self.assertIn("function rememberLiveSpeakerEvidence(speakerId, item)", HTML)
        self.assertIn("function realtimeRowHasSpeakerEvidence(start, end)", HTML)
        self.assertIn("ctx.owners.transcript.liveSpeakerTimeline.push({speakerId: normalizedSpeakerId, start, end});", HTML)
        self.assertIn("ctx.owners.transcript.liveSpeakerTimeline = [];", HTML)
        self.assertIn("const previousDisplaySpeakerId = item.realtime ? normalizedLiveSpeakerId(row.dataset.speaker) : \"\";", HTML)
        self.assertIn("realtimeRowDisplaySpeakerId(rawSpeakerId, startSeconds, endSeconds, previousDisplaySpeakerId)", HTML)
        self.assertIn('row.dataset.rawSpeaker = item.realtime ? (visualSplit ? displaySpeakerId : rawSpeakerId) : "";', HTML)
        self.assertIn('row.dataset.speaker = displaySpeakerId || "UNKNOWN";', HTML)
        self.assertIn('row.classList.toggle("live-speaker-row", item.realtime && Boolean(displaySpeakerId));', HTML)
        self.assertIn('row.style.setProperty("--live-row-color", color || "#8F9BA8");', HTML)
        self.assertIn("function updateCurrentLiveSpeakerFromRealtimeRows()", HTML)
        self.assertIn("ctx.owners.transcript.transcriptLiveSpeakerId = realtimeRowTranscriptLiveSpeakerId(activeRow);", HTML)
        self.assertIn("function reconcileLiveSpeakerHighlight()", HTML)
        self.assertIn("updateCurrentLiveSpeakerFromRealtimeRows();", HTML)
        self.assertIn("speakingSeconds[speakerId] = (speakingSeconds[speakerId] || 0) + Math.max(0, end - start);", HTML)
        self.assertIn("ctx.owners.speakers.renderedSpeakerSpeakingSeconds = speakingSeconds;", HTML)
        self.assertIn("function applyFastSpeakerPanelSignal(item)", HTML)
        self.assertIn('applyFastSpeakerPanelSignal(item);', HTML)
        self.assertIn("if (item.only_if_no_live_speaker && ctx.owners.transcript.currentLiveSpeakerId) return;", HTML)
        self.assertIn("refreshSpeakerPanelSentenceCounts();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = displaySpeakerId || "UNKNOWN";'),
            HTML.index("refreshSpeakerPanelSentenceCounts();", HTML.index("function renderSentence(item)")),
        )

    def test_realtime_unknown_sentence_ignores_tail_only_known_speaker(self) -> None:
        display_start = HTML.index("function realtimeRowDisplaySpeakerId")
        display_end = HTML.index("function applyRealtimeRowSpeaker")
        display_block = HTML[display_start:display_end]

        self.assertIn("const dominantSpeakerId = dominantRealtimeSpeakerId(start, end);", display_block)
        self.assertIn('if (realtimeRowHasSpeakerEvidence(start, end)) return "";', display_block)
        self.assertIn("if (rowEnd - rowStart > 3) return \"\";", display_block)
        self.assertNotIn("activeFallbackLiveSpeakerId()", display_block)
        self.assertIn("const requiredSeconds = Math.max(0.3, scoredSeconds * 0.5);", HTML)

    def test_realtime_visual_split_uses_tail_speaker_and_punctuation_only(self) -> None:
        self.assertIn(".row.provisional-visual-split", HTML)
        self.assertIn("function realtimeTailSpeakerChange(start, end, currentSpeakerId = \"\")", HTML)
        self.assertIn("function lastPunctuationTextSplit(textValue)", HTML)
        self.assertIn("function provisionalRealtimeVisualSplit(item, displaySpeakerId, start, end)", HTML)
        self.assertIn("const tailChange = realtimeTailSpeakerChange(start, end, displaySpeakerId);", HTML)
        self.assertIn("const textSplit = lastPunctuationTextSplit(item.text);", HTML)
        self.assertIn('if (tailSeconds < 0.4) return;', HTML)
        self.assertIn("const boundaryPattern = /[.!?][\"')\\]]*\\s+/g;", HTML)
        self.assertIn("renderProvisionalRealtimeSplitRow(row, visualSplit);", HTML)
        self.assertIn("clearProvisionalRealtimeSplitsFor(item.index);", HTML)
        self.assertIn("restoreRealtimeRowFullPreview(row);", HTML)
        self.assertIn("applyProvisionalRealtimeVisualSplit(row, visualSplit);", HTML)
        self.assertIn("row.dataset.fullRawSpeaker = item.realtime ? rawSpeakerId : \"\";", HTML)
        self.assertIn("row.dataset.fullEnd = item.realtime ? String(endSeconds) : \"\";", HTML)
        self.assertIn("row.dataset.fullText = item.realtime ? (item.text || \"\") : \"\";", HTML)

    def test_realtime_clear_settles_rows_for_smooth_final_adoption(self) -> None:
        self.assertIn(".row.realtime-settling", HTML)
        self.assertIn(".row.row-removing", HTML)
        self.assertIn("const realtimeSettleRemovalDelayMs = 1400;", HTML)
        self.assertIn("function markRealtimeRowSettling(row, generation)", HTML)
        self.assertIn("function findAdoptableRealtimeRow(item, options = {})", HTML)
        self.assertIn("function removeOverlappingSettlingRealtimeRows(item, keepRow = null)", HTML)
        self.assertIn("function placeSentenceRowChronologically(row)", HTML)
        self.assertIn("function clearSettlingRealtimeState(row)", HTML)
        self.assertIn("markRealtimeRowSettling(row, generation)", HTML)
        self.assertNotIn("forEach(row => row.remove())", HTML)
        self.assertIn("row = findAdoptableRealtimeRow(item, {settlingOnly: true});", HTML)
        self.assertIn("row = findAdoptableRealtimeRow(item);", HTML)
        self.assertIn("clearSettlingRealtimeState(row);", HTML)
        self.assertIn("removeOverlappingSettlingRealtimeRows(item, row);", HTML)
        self.assertIn("if (settlingOnly && row.dataset.realtimeSettling !== \"true\") return;", HTML)
        self.assertIn("if (timeScore >= 0.34 && textScore >= 0.5) {", HTML)
        self.assertIn('.filter(row => row.dataset.realtimeSettling !== "true")', HTML)
        self.assertNotIn("clearAllProvisionalRealtimeSplits", HTML)

    def test_reused_sentence_rows_are_reinserted_chronologically(self) -> None:
        self.assertIn("function rowShouldSortBefore(a, b)", HTML)
        self.assertIn("function rowChronologyKey(row)", HTML)
        self.assertIn("sentences.insertBefore(row, next);", HTML)
        self.assertIn("sentences.appendChild(row);", HTML)

        render_start = HTML.index("function renderSentence(item)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        place_index = render_block.index("placeSentenceRowChronologically(row);")
        split_index = render_block.index("if (visualSplit) {", place_index)
        self.assertLess(render_block.index("row.replaceChildren(top, text);"), place_index)
        self.assertLess(place_index, split_index)

    def test_late_realtime_update_cannot_overwrite_final_sentence_row(self) -> None:
        self.assertIn("function findFinalSentenceRow(index)", HTML)
        self.assertIn("function findRealtimeSentenceRow(index)", HTML)
        self.assertIn('row.dataset.index === key && row.dataset.realtime !== "true"', HTML)
        self.assertIn('row.dataset.index === key && row.dataset.realtime === "true"', HTML)

        render_start = HTML.index("function renderSentence(item)")
        render_end = HTML.index("function connect()", render_start)
        render_block = HTML[render_start:render_end]
        guard = 'if (item.realtime && findFinalSentenceRow(item.index)) {'
        self.assertIn(guard, render_block)
        self.assertLess(render_block.index(guard), render_block.index("clearProvisionalRealtimeSplitsFor(item.index);"))
        self.assertIn("let row = item.realtime", render_block)
        self.assertIn("? findRealtimeSentenceRow(item.index)", render_block)
        self.assertIn(": (findFinalSentenceRow(item.index) || findRealtimeSentenceRow(item.index));", render_block)

    def test_speaker_solo_mute_filters_transcript_rows(self) -> None:
        self.assertIn("soloSpeakerIds: new Set(),", HTML)
        self.assertIn("mutedSpeakerIds: new Set(),", HTML)
        self.assertIn("function speakerTranscriptVisible(speakerId)", HTML)
        self.assertIn("if (ctx.owners.speakers.mutedSpeakerIds.has(speakerId)) return false;", HTML)
        self.assertIn("if (ctx.owners.speakers.soloSpeakerIds.size > 0) return ctx.owners.speakers.soloSpeakerIds.has(speakerId);", HTML)
        self.assertIn(
            "row.hidden = hiddenByGroup || !speakerTranscriptVisible(row.dataset.speaker) || !transcriptSearchVisible(row) || !transcriptReviewVisible(row);",
            HTML,
        )
        self.assertIn("function setSpeakerFilter(speakerId, mode, active)", HTML)
        self.assertIn("function pruneSpeakerFilterState()", HTML)
        self.assertIn("refreshTranscriptVisibility();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = displaySpeakerId || "UNKNOWN";'),
            HTML.index("refreshTranscriptVisibility();", HTML.index("function renderSentence(item)")),
        )

    def test_transcript_corrections_use_selection_toolbar(self) -> None:
        self.assertIn('id="selectionToolbar" class="selection-toolbar" hidden', HTML)
        self.assertIn(".selection-toolbar { position:absolute;", HTML)
        self.assertIn('id="bulkCorrectionSpeaker"', HTML)
        self.assertIn('const createSpeakerOptionValue = "__create_speaker__";', HTML)
        self.assertIn('createOption.textContent = createSpeakerAllowed ? "Create new speaker" : "Create new speaker (select one speaker)";', HTML)
        self.assertIn("selectedTranscriptRowIndexes: new Set(),", HTML)
        self.assertIn("function selectedTranscriptRows()", HTML)
        self.assertIn("function disableFollowLiveForTranscriptSelection()", HTML)
        self.assertIn("ctx.owners.speakers.followLiveEnabled = false;", HTML)
        self.assertIn("followLive.checked = false;", HTML)
        self.assertIn("function reassignSelectedSentences()", HTML)
        self.assertIn("function createSpeakerFromSelectedSentences()", HTML)
        self.assertIn('post("/api/corrections/reassign", {indexes, speaker_id: toInternalSpeakerId(speakerId), update_memory: true})', HTML)
        self.assertIn('post("/api/corrections/mark-correct", {indexes})', HTML)
        self.assertIn('post("/api/sessions/corrections/reassign", {', HTML)
        self.assertIn('post("/api/sessions/corrections/mark-correct", {', HTML)
        self.assertIn('row.dataset.selectable = (!item.realtime && !item.pending) ? "true" : "false";', HTML)
        self.assertIn("refreshTranscriptGrouping();", HTML)
        self.assertIn('post("/api/speakers/split", {speaker_id: toInternalSpeakerId(speakerId), sentence_indices: indexes, update_memory: true})', HTML)
        self.assertIn('bulkReassignButton.addEventListener("click", () => reassignSelectedSentences());', HTML)
        self.assertNotIn('id="bulkSplit"', HTML)
        self.assertNotIn("sentence-select-checkbox", HTML)
        self.assertNotIn("sentence-select-control", HTML)
        self.assertNotIn("createSentenceSelectionControl", HTML)
        self.assertNotIn("createSentenceCorrectionControls", HTML)
        self.assertNotIn('className = "correction-actions"', HTML)
        self.assertNotIn('className = "correction-button"', HTML)


if __name__ == "__main__":
    unittest.main()
