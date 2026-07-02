from __future__ import annotations

import argparse
import importlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

from whospeaks.embeddings.embedding_providers import EmbeddingSubprocessClient
from whospeaks.speakers.realtime_speaker_memory import SpeakerMemory as RealtimeSpeakerMemory
from whospeaks.speakers.speaker_embedding_cluster import SpeakerMemory as ClusterSpeakerMemory
from whospeaks.window.window_diarizer import WindowDiarizer
from whospeaks.window.window_gui_html import HTML
from whospeaks.window.window_preview import KrokoSubprocessPreviewTranscriber


def realtime_memory() -> RealtimeSpeakerMemory:
    return RealtimeSpeakerMemory(
        same_speaker_similarity=0.45,
        similarity_temperature=0.07,
        speaker_softmax_temperature=0.075,
        new_speaker_threshold=0.58,
        duplicate_profile_similarity=0.40,
        unknown_short_threshold=0.86,
        min_first_speaker_seconds=0.1,
        min_new_speaker_seconds=1.0,
        late_new_speaker_min_seconds=3.5,
        max_speakers=10,
        min_margin=0.08,
        margin_temperature=0.05,
        update_unknown_max=0.55,
    )


class SpeakerDecisionContractTests(unittest.TestCase):
    def assert_created_speaker_probability_contract(self, decision: object) -> None:
        self.assertEqual(decision.assigned_speaker, "S1")
        self.assertTrue(decision.created_speaker)
        self.assertEqual(decision.probabilities.get("unknown"), 0.0)
        self.assertEqual(decision.unknown_probability, 0.0)
        self.assertEqual(decision.probabilities.get("speaker1"), 1.0)

    def test_realtime_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        decision = realtime_memory().classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)

    def test_cluster_memory_created_speaker_is_not_reported_as_unknown(self) -> None:
        memory = ClusterSpeakerMemory(min_first_speaker_seconds=0.1)
        decision = memory.classify(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.2)
        self.assert_created_speaker_probability_contract(decision)


class WindowHtmlSafetyTests(unittest.TestCase):
    def test_speaker_label_is_inserted_as_text_not_markup(self) -> None:
        self.assertNotIn("${speakerLabel}</span>", HTML)
        self.assertIn("speakerBadge.textContent = speakerLabel;", HTML)
        self.assertIn("row.replaceChildren(top, text);", HTML)

    def test_revised_sentence_refreshes_speaker_counts(self) -> None:
        self.assertIn("let renderedSpeakerSentenceCounts = {};", HTML)
        self.assertIn("let renderedSpeakerSpeakingSeconds = {};", HTML)
        self.assertIn('let currentLiveSpeakerId = "";', HTML)
        self.assertIn('let transcriptLiveSpeakerId = "";', HTML)
        self.assertIn('let fallbackLiveSpeakerId = "";', HTML)
        self.assertIn("let hasRenderedFinalSentenceRows = false;", HTML)
        self.assertIn("row.dataset.speaker = item.assigned_speaker || \"UNKNOWN\";", HTML)
        self.assertIn('row.style.setProperty("--live-row-color", color || "#8F9BA8");', HTML)
        self.assertIn("function updateCurrentLiveSpeakerFromRealtimeRows()", HTML)
        self.assertIn('transcriptLiveSpeakerId = activeRow && activeRow.dataset.speaker !== "UNKNOWN" ? activeRow.dataset.speaker : "";', HTML)
        self.assertIn("function reconcileLiveSpeakerHighlight()", HTML)
        self.assertIn("updateCurrentLiveSpeakerFromRealtimeRows();", HTML)
        self.assertIn("speakingSeconds[speakerId] = (speakingSeconds[speakerId] || 0) + Math.max(0, end - start);", HTML)
        self.assertIn("renderedSpeakerSpeakingSeconds = speakingSeconds;", HTML)
        self.assertIn("refreshSpeakerPanelSentenceCounts();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = item.assigned_speaker || "UNKNOWN";'),
            HTML.index("refreshSpeakerPanelSentenceCounts();", HTML.index("function renderSentence(item)")),
        )

    def test_speaker_solo_mute_filters_transcript_rows(self) -> None:
        self.assertIn("let soloSpeakerIds = new Set();", HTML)
        self.assertIn("let mutedSpeakerIds = new Set();", HTML)
        self.assertIn("function speakerTranscriptVisible(speakerId)", HTML)
        self.assertIn("if (mutedSpeakerIds.has(speakerId)) return false;", HTML)
        self.assertIn("if (soloSpeakerIds.size > 0) return soloSpeakerIds.has(speakerId);", HTML)
        self.assertIn("row.hidden = !speakerTranscriptVisible(row.dataset.speaker) || !transcriptSearchVisible(row);", HTML)
        self.assertIn("function setSpeakerFilter(speakerId, mode, active)", HTML)
        self.assertIn("function pruneSpeakerFilterState()", HTML)
        self.assertIn("refreshTranscriptVisibility();", HTML)
        self.assertLess(
            HTML.index('row.dataset.speaker = item.assigned_speaker || "UNKNOWN";'),
            HTML.index("refreshTranscriptVisibility();", HTML.index("function renderSentence(item)")),
        )

    def test_live_transcript_header_matches_draft_contract(self) -> None:
        self.assertIn('class="transcript-header"', HTML)
        self.assertIn("Live transcript", HTML)
        self.assertIn('id="followLive" type="checkbox" checked', HTML)
        self.assertIn("let followLiveEnabled = true;", HTML)
        self.assertIn("if (!followLiveEnabled) return;", HTML)
        self.assertIn('id="transcriptSearch" type="search" placeholder="Search transcript"', HTML)
        self.assertIn("let transcriptSearchText = \"\";", HTML)
        self.assertIn("function transcriptSearchVisible(row)", HTML)
        self.assertIn("query.split(/\\s+/).every(term => searchable.includes(term));", HTML)
        self.assertIn('id="copyTranscript" class="transcript-icon-button"', HTML)
        self.assertIn('id="downloadTranscript" class="transcript-icon-button"', HTML)
        self.assertIn("function transcriptExportText(speakerId = null)", HTML)
        self.assertIn("`[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`", HTML)
        self.assertIn("function copyTextToClipboard(text)", HTML)
        self.assertIn("function downloadTranscript(speakerId = null)", HTML)
        self.assertIn('id="transcriptSettings"', HTML)
        self.assertIn('id="transcriptSettingsPanel" class="transcript-settings-panel" hidden', HTML)
        self.assertIn('id="showTranscriptTags" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptTime" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptSpeechRate" type="checkbox" checked', HTML)
        self.assertIn('id="showTranscriptProbabilities" type="checkbox" checked', HTML)
        self.assertIn(".transcript-panel.hide-tags .badge.new, .transcript-panel.hide-tags .badge.state { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-time .sentence-duration, .transcript-panel.hide-time .sentence-range { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-speech-rate .sentence-speech-rate { display:none; }", HTML)
        self.assertIn(".transcript-panel.hide-probabilities .prob { display:none; }", HTML)
        self.assertNotIn("Show low confidence", HTML)
        self.assertNotIn(">Filter<", HTML)

    def test_playback_clock_ignores_early_media_end_jumps(self) -> None:
        self.assertIn("playbackClockStartedAt", HTML)
        self.assertIn("playbackClockSlackSeconds", HTML)
        self.assertIn("Ignoring early audio ended event", HTML)

    def test_live_header_matches_draft_contract(self) -> None:
        self.assertIn("WhoSpeaks Live", HTML)
        self.assertIn("#17B7FE", HTML)
        self.assertIn("#3DC77C", HTML)
        self.assertIn("#BA79EF", HTML)
        self.assertIn("Stop transcription", HTML)
        self.assertIn("background:#981D20", HTML)
        self.assertIn("border-color:#DF3C36", HTML)
        self.assertIn('id="speakerCountNumber" class="speaker-count-number"', HTML)
        self.assertIn('id="speakerCountLabel" class="speaker-count-label"', HTML)
        self.assertIn(".speaker-count-number { position:relative; top:2px; font-size:16px; font-weight:600; line-height:1; color:#FF9F1C;", HTML)
        self.assertIn(".speaker-count-label { font-size:13px; font-weight:400;", HTML)
        self.assertIn(".speaker-summary { flex:0 0 auto; min-height:23px; display:flex; align-items:center; gap:4px;", HTML)
        self.assertIn("#speakerCount { display:inline-flex; align-items:baseline; gap:7px;", HTML)
        self.assertIn("speakerCountNumber.textContent", HTML)
        self.assertIn("speakerCountLabel.textContent", HTML)
        self.assertIn(".live-summary { min-width:0; margin-left:auto;", HTML)
        self.assertIn(".live-summary { width:100%; justify-content:flex-end; }", HTML)
        header_start = HTML.index('<header class="topbar">')
        header_end = HTML.index("</header>", header_start)
        header = HTML[header_start:header_end]
        self.assertEqual(header.count("topbar-divider"), 2)
        self.assertLess(header.index('class="brand"'), header.index('class="live-summary"'))
        status_speaker_divider = header.index("topbar-divider")
        transport_divider = header.index("topbar-divider", status_speaker_divider + 1)
        self.assertLess(header.index('id="state"'), status_speaker_divider)
        self.assertLess(status_speaker_divider, header.index('id="speakerCount"'))
        self.assertLess(header.index('id="speakerCountNumber"'), header.index('id="speakerCountLabel"'))
        self.assertLess(header.index('id="speakerCount"'), transport_divider)
        self.assertLess(transport_divider, header.index('class="transport"'))

    def test_media_area_matches_draft_contract(self) -> None:
        self.assertIn("#0B1015", HTML)
        self.assertIn("#0F161F", HTML)
        self.assertIn("--bg:#0B1015;", HTML)
        self.assertIn("--panel:#0F161F;", HTML)
        self.assertIn("--panel-2:#0F161F;", HTML)
        self.assertIn("--field:#0B1015;", HTML)
        self.assertIn("--line:#1B2B38;", HTML)
        self.assertIn("font:14px/1.35 Arial", HTML)
        self.assertIn(".topbar { min-height:52px;", HTML)
        topbar_css = HTML[HTML.index(".topbar {"):HTML.index("}", HTML.index(".topbar {"))]
        self.assertNotIn("border-bottom", topbar_css)
        self.assertNotIn("inset 0 -1px", topbar_css)
        control_panel_css = HTML[HTML.index(".control-panel {"):HTML.index("}", HTML.index(".control-panel {"))]
        self.assertNotIn("border-left", control_panel_css)
        self.assertIn(".source-strip { min-height:58px;", HTML)
        self.assertIn(".playback-panel { min-height:132px;", HTML)
        self.assertIn("grid-template-columns:minmax(150px, 240px)", HTML)
        self.assertIn(".timeline-bar { position:relative; height:6px; margin-left:8px; margin-right:10px;", HTML)
        self.assertIn(".source-grid { width:100%;", HTML)
        self.assertIn("border:0; border-radius:0; background:transparent;", HTML)
        self.assertIn(".source-row { display:contents; }", HTML)
        self.assertIn("--text:#F1F5F8;", HTML)
        self.assertIn(".dropdown-control { position:relative; min-height:34px; display:flex; align-items:center; border:1px solid var(--line); border-radius:7px; background:#0F161F; color:var(--text);", HTML)
        self.assertIn(".dropdown-control::after { content:\"\"; position:absolute; right:15px; top:50%; width:8px; height:8px; border-right:1.5px solid currentColor; border-bottom:1.5px solid currentColor;", HTML)
        self.assertIn(".select-control select { width:100%; min-width:0; min-height:32px; border:0; border-radius:7px; padding:0 36px 0 12px; background:transparent; color:inherit; font:inherit; appearance:none;", HTML)
        self.assertIn('class="source-mode-button dropdown-control"', HTML)
        self.assertIn('class="select-control dropdown-control"><select id="preset"', HTML)
        self.assertNotIn("background-image:linear-gradient", HTML)
        self.assertIn(".media-controls { min-width:0; min-height:100%; display:grid; grid-template-rows:auto minmax(0,1fr) auto;", HTML)
        self.assertIn(".media-expand { width:40px; height:40px; align-self:end; justify-self:start;", HTML)
        self.assertIn('id="mediaCard" class="media-card mode-youtube"', HTML)
        self.assertIn('class="source-strip"', HTML)
        self.assertIn("Change source", HTML)
        self.assertIn('id="sourceModeOptions"', HTML)
        self.assertIn('data-input-mode="youtube"', HTML)
        self.assertIn('data-input-mode="microphone"', HTML)
        self.assertIn('data-input-mode="system"', HTML)
        self.assertIn('id="youtubeSourceControls"', HTML)
        self.assertIn('id="timelineFill"', HTML)
        self.assertIn('id="timelineThumb"', HTML)
        self.assertIn('id="capturePanel"', HTML)
        self.assertIn('id="captureLevelFill"', HTML)
        self.assertIn('id="micGain"', HTML)
        self.assertIn("function updateMediaMode()", HTML)
        self.assertIn("function updateMediaTimeline()", HTML)
        self.assertIn("function setCaptureLevel(value)", HTML)
        self.assertIn("function setSourceModeMenuOpen(open)", HTML)
        self.assertNotIn("source-panel", HTML)
        self.assertNotIn("source-menu", HTML)
        self.assertNotIn("font-weight:700", HTML)
        self.assertNotIn("font-weight:800", HTML)
        self.assertIn("strong, b, h1, h2, h3, h4, h5, h6, summary { font-weight:400; }", HTML)
        self.assertIn(".speaker-name, .speaker-row-title { font-weight:600; }", HTML)
        old_surface_colors = [
            "#090b0d",
            "#151715",
            "#101210",
            "#080a09",
            "#343a36",
            "#080d12",
            "#0d0f0d",
            "#20241f",
            "#123e2d",
            "#102231",
            "#122231",
            "#111923",
            "#1B2732",
            "#0d131a",
            "#59675d",
            "#2f8f68",
            "#65b891",
            "#9ea89f",
        ]
        for color in old_surface_colors:
            self.assertNotIn(color, HTML)
        oversized_layout_tokens = [
            "min-height:68px",
            "min-height:88px",
            "min-height:200px",
            "font-size:20px",
            "grid-template-columns:minmax(220px, 360px)",
            "padding:16px 18px",
        ]
        for token in oversized_layout_tokens:
            self.assertNotIn(token, HTML)

        media_start = HTML.index('<section id="mediaCard"')
        transcript_start = HTML.index('<section class="transcript-panel"', media_start)
        media = HTML[media_start:transcript_start]
        self.assertIn('id="inputMode"', media)
        self.assertIn('id="preset"', media)
        self.assertIn('id="source"', media)
        self.assertIn('id="load"', media)
        self.assertLess(media.index('id="sourceKind"'), media.index('id="inputMode"'))
        self.assertLess(media.index('class="video-frame"'), media.index('id="youtubeSourceControls"'))
        self.assertLess(media.index('id="youtubeSourceControls"'), media.index('class="timeline-row"'))
        self.assertLess(media.index('class="timeline-row"'), media.index('id="expandMedia"'))
        self.assertNotIn("media-subtle-line", media)

        video_start = HTML.index('<video id="video"')
        video_end = HTML.index("</video>", video_start)
        self.assertNotIn("controls", HTML[video_start:video_end])
        audio_start = HTML.index('<audio id="audio"')
        audio_end = HTML.index("</audio>", audio_start)
        self.assertNotIn("controls", HTML[audio_start:audio_end])

    def test_speaker_panel_matches_draft_contract(self) -> None:
        self.assertIn('class="control-card speaker-panel"', HTML)
        self.assertIn('class="speaker-tabs"', HTML)
        self.assertIn('data-speaker-tab="speakers"', HTML)
        self.assertIn('data-speaker-tab="settings"', HTML)
        self.assertIn('id="speakerPanelTitle" class="speaker-panel-title">Detected speakers (0)</h2>', HTML)
        self.assertIn('id="addReferenceSpeaker"', HTML)
        self.assertNotIn('id="speakerGroupCurrent"', HTML)
        self.assertNotIn("Current:", HTML)
        self.assertIn('class="speaker-file-actions"', HTML)
        self.assertIn(".speaker-file-actions button { min-height:28px; width:auto; padding:0 10px; font-size:12px; }", HTML)
        self.assertIn('id="loadSpeakerGroup" type="button">Load file</button>', HTML)
        self.assertIn('id="saveSpeakerGroup" type="button">Save file</button>', HTML)
        self.assertIn('id="speakerGroupFile" type="file"', HTML)
        self.assertNotIn('id="speakerGroupName"', HTML)
        self.assertNotIn('id="speakerGroupSelect"', HTML)
        self.assertIn('id="manualSpeakerComposer" class="manual-speaker-composer" hidden', HTML)
        self.assertIn('id="manualSpeakerName"', HTML)
        self.assertIn('id="manualSpeakerReferenceDock"', HTML)
        self.assertIn(".speaker-tab.active { color:#E8EEF5; box-shadow:inset 0 -2px 0 #17B7FE;", HTML)
        self.assertIn('class="sensitivity-title">New speaker</span>', HTML)
        self.assertIn('class="sensitivity-row"', HTML)
        self.assertIn(".sensitivity-title { color:var(--text); font-size:13px; line-height:1.25; }", HTML)
        self.assertIn(".sensitivity-row { display:flex; align-items:center; gap:15px;", HTML)
        self.assertIn(".sensitivity input { flex:0 1 50%; max-width:50%; min-width:120px;", HTML)
        self.assertIn(".manual-speaker-composer { display:grid; gap:8px;", HTML)
        self.assertIn(".speaker-item { --speaker-color:transparent;", HTML)
        self.assertIn(".speaker-item.live-speaker { background:color-mix(in srgb, var(--speaker-color) 10%, #0F161F); }", HTML)
        self.assertIn(".speaker-title-row { min-width:0; display:flex; align-items:center; gap:7px; }", HTML)
        self.assertIn(".speaker-live-indicator { flex:0 0 auto; display:inline-flex; align-items:center; gap:4px; color:var(--speaker-color);", HTML)
        self.assertIn(".speaker-item-summary { width:100%; min-height:60px; display:grid; grid-template-columns:minmax(0,1fr) auto;", HTML)
        self.assertIn("box-shadow:inset 4px 0 0 var(--speaker-color);", HTML)
        self.assertNotIn("speaker-avatar", HTML)
        self.assertIn(".speaker-item.editing { position:relative; z-index:1; border:1px solid var(--speaker-color);", HTML)
        self.assertIn(".speaker-item:not(.editing) .speaker-row-title { color:var(--speaker-color); }", HTML)
        self.assertIn(".speaker-item-tail { align-self:stretch; display:flex; flex-direction:column; align-items:flex-end; justify-content:space-between;", HTML)
        self.assertIn(".speaker-filter-controls, .speaker-transcript-actions { display:flex; align-items:center; gap:4px; }", HTML)
        self.assertIn(".speaker-filter-toggle { min-height:20px; width:39px;", HTML)
        self.assertIn(".speaker-filter-toggle.mute.active", HTML)
        self.assertIn(".transcript-icon-button { min-height:24px; width:28px;", HTML)
        self.assertIn(".row.realtime { background:color-mix(in srgb, var(--live-row-color, #8F9BA8) 10%, #0B1015); }", HTML)
        self.assertIn("function createSpeakerLiveIndicator()", HTML)
        self.assertIn('indicator.appendChild(document.createTextNode("Live"));', HTML)
        self.assertIn('titleRow.appendChild(createSpeakerLiveIndicator());', HTML)
        self.assertIn('indicator.remove();', HTML)
        self.assertIn("function applyFallbackLiveSpeaker(item)", HTML)
        self.assertIn('es.addEventListener("live_speaker", e => applyFallbackLiveSpeaker(JSON.parse(e.data)));', HTML)
        self.assertIn('fallbackLiveSpeakerUntilMs = performance.now() + Math.max(0, Number(item.hold_seconds || 2.0)) * 1000;', HTML)
        self.assertIn("currentLiveSpeakerId = transcriptLiveSpeakerId || activeFallbackLiveSpeakerId();", HTML)
        self.assertIn('row.classList.toggle("live-speaker", Boolean(currentLiveSpeakerId) && speaker.id === currentLiveSpeakerId);', HTML)
        self.assertNotIn("speaker-editing-badge", HTML)
        self.assertIn(".speaker-row-name-input", HTML)
        self.assertIn("Reference voice added", HTML)
        self.assertNotIn("No reference voice", HTML)
        self.assertIn("function setSpeakerTab(tabName)", HTML)
        self.assertIn("function setEditingSpeaker(speakerId, options = {})", HTML)
        self.assertIn("const collapse = requestedId && editingSpeakerId === requestedId && !options.keepOpen;", HTML)
        self.assertIn("manualSpeakerComposerOpen = false;", HTML)
        self.assertIn("function syncManualSpeakerComposer()", HTML)
        self.assertIn("manualSpeakerReferenceDock.appendChild(referenceSpeakerForm);", HTML)
        self.assertIn("manualSpeakerName.focus();", HTML)
        self.assertIn("manualSpeakerName.select();", HTML)
        self.assertIn('if (editingSpeakerId && !speakerIds.includes(editingSpeakerId))', HTML)
        self.assertNotIn('editingSpeakerId = speakerIds[0]', HTML)
        self.assertIn("pendingSpeakerNameFocusId = editingSpeakerId && options.focusName !== false ? editingSpeakerId : \"\";", HTML)
        self.assertIn("return manualSpeakerName.value.trim();", HTML)
        self.assertIn("function closeManualSpeakerComposerAfterReference()", HTML)
        self.assertIn('addReferenceSpeakerButton.addEventListener("click"', HTML)
        self.assertNotIn("window.prompt", HTML)
        self.assertIn('const name = speakerLibraryState.group_name || "speakers";', HTML)
        self.assertIn('const result = await post("/api/speakers/export", {name});', HTML)
        self.assertIn("downloadJsonFile(speakerGroupFileName(group.name || name), group);", HTML)
        self.assertIn("speakerGroupFile.click();", HTML)
        self.assertIn("const group = JSON.parse(await file.text());", HTML)
        self.assertIn('const result = await post("/api/speakers/import", {group});', HTML)
        self.assertIn("function speakerPanelName(speaker)", HTML)
        self.assertIn("function createSpeakerFilterToggle(speaker, mode)", HTML)
        self.assertIn('filterControls.appendChild(createSpeakerFilterToggle(speaker, "solo"));', HTML)
        self.assertIn('filterControls.appendChild(createSpeakerFilterToggle(speaker, "mute"));', HTML)
        self.assertIn("function createTranscriptActionButton(kind, speaker)", HTML)
        self.assertIn('transcriptActions.appendChild(createTranscriptActionButton("copy", speaker));', HTML)
        self.assertIn('transcriptActions.appendChild(createTranscriptActionButton("download", speaker));', HTML)
        self.assertIn('button.setAttribute("aria-pressed", active ? "true" : "false");', HTML)
        self.assertIn('target.closest(".speaker-row-name-input, .speaker-filter-toggle, .speaker-transcript-action")', HTML)
        self.assertIn("function recomputeRenderedSpeakerSentenceCounts()", HTML)
        self.assertIn('if (row.dataset.realtime === "true") return;', HTML)
        self.assertIn("function speakerPanelSpeakingSeconds(speaker)", HTML)
        self.assertIn('return renderedSpeakerSpeakingSeconds[speakerId] || 0;', HTML)
        self.assertIn('return `${total} ${total === 1 ? "sentence" : "sentences"} · ${speakerSpeakingTimeText(speakingSeconds)}`;', HTML)
        self.assertIn("function refreshSpeakerPanelSentenceCounts()", HTML)
        self.assertIn("speakerPanelSentenceCount(speaker)", HTML)
        self.assertIn("speakerPanelSpeakingSeconds(speaker)", HTML)
        self.assertIn("async function commitSpeakerNameInput(speaker, input)", HTML)
        self.assertIn('title.value = speakerPanelName(speaker);', HTML)
        self.assertIn('title.addEventListener("blur"', HTML)
        self.assertIn("title.focus();", HTML)
        self.assertIn("title.select();", HTML)
        self.assertNotIn('id="saveSpeakerName"', HTML)
        self.assertNotIn('id="cancelSpeakerEdit"', HTML)
        self.assertNotIn('id="stopReference"', HTML)
        self.assertIn("Upload audio", HTML)
        self.assertIn("Record from mic", HTML)
        self.assertIn('recordReferenceButtonLabel.textContent = recording ? "Stop and add" : "Record from mic";', HTML)
        self.assertIn('if (referenceRecordStream || referenceRecordPending)', HTML)
        self.assertIn('recordReferenceButton.classList.toggle("recording", recording);', HTML)


class WindowStreamingAudioTests(unittest.TestCase):
    def test_portable_speaker_group_centroid_preserves_float32_payload(self) -> None:
        centroid = np.array([0.125, -0.5, 0.33333334, 1.0], dtype=np.float32)
        payload = WindowDiarizer._centroid_payload(centroid)

        self.assertEqual(payload["centroid_encoding"], "float32-base64-le")
        restored = np.asarray(WindowDiarizer._centroid_from_payload(payload), dtype=np.float32)
        np.testing.assert_array_equal(restored, centroid)

    def test_portable_speaker_group_export_import_round_trips_profiles(self) -> None:
        class FakeBus:
            def emit(self, *_args: object, **_kwargs: object) -> None:
                return None

        class FakeMemory:
            def __init__(self, profiles: list[dict[str, object]] | None = None) -> None:
                self.profiles = profiles or []

            def export_profiles(self) -> list[dict[str, object]]:
                return [dict(profile) for profile in self.profiles]

            def replace_profiles(self, profiles: list[dict[str, object]]) -> None:
                self.profiles = []
                for index, item in enumerate(profiles, 1):
                    self.profiles.append({
                        "label": f"S{index}",
                        "index": index,
                        "centroid": np.asarray(item["centroid"], dtype=np.float32),
                        "sentence_count": int(item.get("sentence_count") or 1),
                        "speech_seconds": float(item.get("speech_seconds") or 0.0),
                        "created_at": time.time(),
                        "last_seen_at": time.time(),
                        "locked": bool(item.get("locked")),
                    })

        centroid = np.array([0.125, -0.5, 0.33333334, 1.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            source = WindowDiarizer.__new__(WindowDiarizer)
            source.args = argparse.Namespace(embedding_provider="mock", embedding_device="cpu")
            source.speaker_library_dir = Path(tmp)
            source.memory = FakeMemory([{
                "label": "S1",
                "index": 1,
                "centroid": centroid,
                "sentence_count": 3,
                "speech_seconds": 7.5,
                "created_at": 10.0,
                "last_seen_at": 12.0,
                "locked": True,
            }])
            source._speaker_lock = threading.Lock()
            source._unknown_lock = threading.Lock()
            source._speaker_metadata = {"S1": {"name": "Alice", "source": "reference", "locked": True, "reference_audio": ""}}
            source._speaker_group_name = ""
            source._seed_profiles = []
            source.bus = FakeBus()

            group = source.export_speaker_group_file("Local group")

            target = WindowDiarizer.__new__(WindowDiarizer)
            target.args = argparse.Namespace(embedding_provider="mock", embedding_device="cpu")
            target.speaker_library_dir = Path(tmp)
            target.memory = FakeMemory()
            target._new_memory = lambda: FakeMemory()
            target._speaker_lock = threading.Lock()
            target._unknown_lock = threading.Lock()
            target._unknown_sentences = []
            target._speaker_metadata = {}
            target._speaker_group_name = ""
            target._seed_profiles = []
            target.bus = FakeBus()

            state = target.import_speaker_group_file(group)

        self.assertEqual(group["format"], "whospeaks-speaker-group")
        self.assertEqual(group["speakers"][0]["centroid_encoding"], "float32-base64-le")
        self.assertEqual(state["group_name"], "Local_group")
        self.assertEqual(state["speakers"][0]["display_name"], "Alice")
        np.testing.assert_array_equal(target.memory.profiles[0]["centroid"], centroid)

    def test_browser_stream_audio_uses_chunks_and_slices_across_boundaries(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._audio_lock = threading.Lock()
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = True
        diarizer.sample_rate = 4
        diarizer.audio = np.zeros(0, dtype=np.float32)
        diarizer._stream_audio_chunks = []
        diarizer._stream_audio_samples = 0
        diarizer.duration = 0.0

        first_duration = diarizer.append_stream_audio(np.array([0.1, 0.2], dtype=np.float32), 4)
        second_duration = diarizer.append_stream_audio(np.array([0.3, 0.4, 0.5], dtype=np.float32), 4)

        self.assertEqual(first_duration, 0.5)
        self.assertEqual(second_duration, 1.25)
        self.assertEqual(len(diarizer._stream_audio_chunks), 2)
        self.assertEqual(len(diarizer.audio), 0)

        audio, sample_rate = diarizer._audio_window_copy(0.25, 1.25)
        self.assertEqual(sample_rate, 4)
        np.testing.assert_allclose(audio, np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))

    def test_file_playback_time_rejects_impossible_jump_to_media_end(self) -> None:
        class Bus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict[str, object]]] = []

            def emit(self, event: str, payload: dict[str, object]) -> None:
                self.events.append((event, payload))

        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = False
        diarizer.duration = 60.0
        diarizer._playback_clock_started_at = time.monotonic() - 1.0
        diarizer._last_playback_jump_warning_at = 0.0
        diarizer.bus = Bus()

        diarizer.set_playback_time(60.0)

        self.assertLess(diarizer.playback_time(), 5.0)
        self.assertTrue(any("Ignored early playback jump" in str(payload.get("message")) for _event, payload in diarizer.bus.events))

    def test_stream_playback_time_is_not_wall_clock_clamped(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer._playback_lock = threading.Lock()
        diarizer._playback_time = 0.0
        diarizer._streaming_audio = True
        diarizer.duration = 60.0
        diarizer._playback_clock_started_at = time.monotonic()
        diarizer._last_playback_jump_warning_at = 0.0
        diarizer.bus = object()

        diarizer.set_playback_time(60.0)

        self.assertEqual(diarizer.playback_time(), 60.0)

    def test_live_speaker_probe_uses_cheap_rms_speech_gate(self) -> None:
        diarizer = WindowDiarizer.__new__(WindowDiarizer)
        diarizer.args = argparse.Namespace(
            vad_frame_seconds=0.1,
            vad_speech_rms_threshold=0.003,
            live_speaker_probe_min_speech_seconds=0.2,
        )

        self.assertFalse(diarizer._audio_has_rms_speech(np.zeros(200, dtype=np.float32), 100))
        audio = np.zeros(200, dtype=np.float32)
        audio[50:90] = 0.01

        self.assertTrue(diarizer._audio_has_rms_speech(audio, 100))


class EmbeddingSubprocessClientTests(unittest.TestCase):
    def test_embed_wav_times_out_and_kills_unresponsive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "silent_embedding_helper.py"
            helper.write_text(
                "import sys, time\n"
                "for _line in sys.stdin:\n"
                "    time.sleep(10)\n",
                encoding="utf-8",
            )
            audio = root / "audio.wav"
            audio.write_bytes(b"")

            client = EmbeddingSubprocessClient(
                python=Path(sys.executable),
                provider="noop",
                device="cpu",
                helper_script=helper,
                response_timeout_seconds=0.2,
            )
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                client.embed_wav(audio)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 2.0)
            self.assertIsNone(client._process)
            client.shutdown(lock_timeout_seconds=0.1)


class KrokoPreviewStartupTests(unittest.TestCase):
    def test_subprocess_preview_uses_worker_script_without_name_error(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO('{"ready":true}\n')
                self.stderr = io.StringIO("")
                self.returncode = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                self.returncode = 0
                return 0

            def kill(self) -> None:
                self.returncode = -9

        args = argparse.Namespace(
            realtime_preview_request_timeout_seconds=0.2,
            realtime_preview_startup_timeout_seconds=0.5,
            realtime_preview_python=Path(sys.executable),
            realtime_preview_engine="kroko_onnx",
            realtime_preview_model="Kroko-EN-Community-64-L-Streaming-001.data",
            realtime_preview_provider="cpu",
            realtime_preview_num_threads=2,
            realtime_preview_model_path=None,
            realtime_preview_download_root=None,
            download_root=None,
            realtime_preview_engine_options_json="",
            realtime_preview_realtimestt_root=None,
        )

        with mock.patch("whospeaks.window.window_preview.subprocess.Popen", return_value=FakeProcess()) as popen:
            transcriber = KrokoSubprocessPreviewTranscriber(args)
            transcriber.close()

        command = popen.call_args.args[0]
        self.assertIn(str(TOOLS / "kroko_realtime_preview_worker.py"), command)


class RepositoryStructureTests(unittest.TestCase):
    def test_package_imports_do_not_require_tools_on_sys_path(self) -> None:
        self.assertNotIn(str(TOOLS), sys.path)
        self.assertEqual(WindowDiarizer.__name__, "WindowDiarizer")

    def test_legacy_window_wrapper_still_imports(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOLS / "youtube_window_diarize_gui.py"), "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Growing-window faster-whisper speaker diarization GUI", completed.stdout)

    def test_runtime_dir_env_redirects_mutable_defaults(self) -> None:
        import whospeaks.paths as paths

        original_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["WHOSPEAKS_RUNTIME_DIR"] = directory
                os.environ.pop("WHOSPEAKS_CACHE_DIR", None)
                os.environ.pop("WHOSPEAKS_MODEL_DIR", None)
                os.environ.pop("WHOSPEAKS_SPEAKER_LIBRARY_DIR", None)
                reloaded = importlib.reload(paths)
                runtime = Path(directory).resolve()
                self.assertEqual(reloaded.RUNTIME_DIR, runtime)
                self.assertEqual(reloaded.CACHE_DIR, runtime / "cache")
                self.assertEqual(reloaded.MODEL_DIR, runtime / "models")
                self.assertEqual(reloaded.SPEAKER_LIBRARY_DIR, runtime / "speakers")
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(paths)

    def test_window_gui_default_embedding_provider_matches_promoted_stack(self) -> None:
        import whospeaks.window.window_config as window_config

        original_env = dict(os.environ)
        try:
            os.environ["WHOSPEAKS_WINDOW_EMBEDDING_PROVIDER"] = "speechbrain_ecapa"
            reloaded = importlib.reload(window_config)
            self.assertEqual(
                reloaded.DEFAULT_WINDOW_EMBEDDING_PROVIDER,
                "espnet_ecapa_wavlm_joint=0.725+jungjee_rawnet3=1+wespeaker_campplus=0.35",
            )
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            importlib.reload(window_config)

    def test_window_gui_tuned_default_parameters_match_promoted_set(self) -> None:
        from whospeaks.window.youtube_window_diarize_gui import parse_args

        expected = {
            "embedding_provider": "espnet_ecapa_wavlm_joint=0.725+jungjee_rawnet3=1+wespeaker_campplus=0.35",
            "same_speaker_similarity": 0.37,
            "similarity_temperature": 0.0576,
            "speaker_softmax_temperature": 0.0539,
            "new_speaker_threshold": 0.38,
            "duplicate_profile_similarity": 0.4,
            "unknown_short_threshold": 0.333,
            "min_first_speaker_seconds": 1.3098,
            "min_new_speaker_seconds": 1.6,
            "late_new_speaker_min_seconds": 3.4127,
            "max_speakers": 12,
            "min_margin": 0.0386,
            "margin_temperature": 0.03,
            "update_unknown_max": 0.54,
            "new_speaker_confirmation_count": 1,
            "new_speaker_confirmation_similarity": 0.5033,
            "max_pending_new_speakers": 6,
            "min_new_speaker_words": 3,
            "retro_reassign_min_similarity": 0.05,
            "retro_reassign_min_margin": 0.0,
            "min_embed_seconds": 0.5,
            "min_speech_audio_ratio": 0.0,
            "sentence_boundary_pre_padding_seconds": 0.06,
            "sentence_boundary_post_padding_seconds": 0.09,
            "sentence_boundary_gap_ratio": 0.6,
            "realtime_preview_diarize_min_audio_seconds": 1.5,
            "realtime_preview_diarize_min_advance_seconds": 0.75,
            "realtime_preview_diarize_min_similarity": 0.45,
            "realtime_preview_diarize_min_margin": 0.08,
            "realtime_preview_diarize_min_known_probability": 0.5,
        }

        with mock.patch.object(sys, "argv", ["youtube_window_diarize_gui.py"]):
            args = parse_args()

        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)

    def test_cunk_canonical_is_a_small_fixture(self) -> None:
        from whospeaks.paths import CUNK_CANONICAL

        self.assertTrue(CUNK_CANONICAL.is_file())
        self.assertIn("tests", CUNK_CANONICAL.parts)
        self.assertIn("fixtures", CUNK_CANONICAL.parts)

    def test_tools_contains_only_wrappers_and_ignored_runtime_artifacts(self) -> None:
        wrapper_names = {
            "benchmark_voice_embeddings.py",
            "kroko_realtime_preview_worker.py",
            "realtime_speakerdiarize.py",
            "youtube_local_filefeed_replay.py",
            "youtube_window_diarize_gui.py",
        }
        actual_files = {path.name for path in TOOLS.glob("*.py")}
        self.assertEqual(actual_files, wrapper_names)
        self.assertFalse((TOOLS / ".window_diarize").exists())
        self.assertFalse((TOOLS / ".local_filefeed_media").exists())


if __name__ == "__main__":
    unittest.main()
