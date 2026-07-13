export function installTranscriptReview(ctx) {
  const {bulkCorrectionSpeaker, bulkMarkCorrectButton, bulkReassignButton, clearSelectionButton, createSpeakerOptionValue, followLive, groupTranscriptTurns, liveSpeakerConfig, reviewFilterButtons, selectionCount, selectionToolbar, sentences, showTranscriptProbabilities, showTranscriptReviewHints, showTranscriptSpeechRate, showTranscriptTags, showTranscriptTime, source, speakerList, speakerTabButtons, speakerTabPanels, start, state, transcriptPanel, transcriptSettingsButton, transcriptSettingsPanel, undoCorrectionButton} = ctx;
  const applyProvisionalRealtimeVisualSplit = (...args) => ctx.api.applyProvisionalRealtimeVisualSplit(...args), clearProvisionalRealtimeSplitsFor = (...args) => ctx.api.clearProvisionalRealtimeSplitsFor(...args), createSpeakerLiveIndicator = (...args) => ctx.api.createSpeakerLiveIndicator(...args), fetchSavedSessions = (...args) => ctx.api.fetchSavedSessions(...args), log = (...args) => ctx.api.log(...args), playbackSeconds = (...args) => ctx.api.playbackSeconds(...args), pruneSpeakerFilterState = (...args) => ctx.api.pruneSpeakerFilterState(...args), refreshSpeakerRows = (...args) => ctx.api.refreshSpeakerRows(...args), refreshTranslationMenuStatus = (...args) => ctx.api.refreshTranslationMenuStatus(...args), refreshTranslationRow = (...args) => ctx.api.refreshTranslationRow(...args), renderMeetingIntelligencePanel = (...args) => ctx.api.renderMeetingIntelligencePanel(...args), renderSpeakerPanel = (...args) => ctx.api.renderSpeakerPanel(...args), restoreRealtimeRowFullPreview = (...args) => ctx.api.restoreRealtimeRowFullPreview(...args), rowShouldSortBefore = (...args) => ctx.api.rowShouldSortBefore(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), secondsLabel = (...args) => ctx.api.secondsLabel(...args), sessionControlsLocked = (...args) => ctx.api.sessionControlsLocked(...args), speakerColor = (...args) => ctx.api.speakerColor(...args), speakerDisplayLabel = (...args) => ctx.api.speakerDisplayLabel(...args), speakerPanelName = (...args) => ctx.api.speakerPanelName(...args), speakerProbabilityKey = (...args) => ctx.api.speakerProbabilityKey(...args), speakerSentenceText = (...args) => ctx.api.speakerSentenceText(...args), speakerTranscriptVisible = (...args) => ctx.api.speakerTranscriptVisible(...args), syncSavedSessionsAutoRefresh = (...args) => ctx.api.syncSavedSessionsAutoRefresh(...args), syncSpeakerSessionBaselines = (...args) => ctx.api.syncSpeakerSessionBaselines(...args), updateSpeakerCount = (...args) => ctx.api.updateSpeakerCount(...args);
  function transcriptSearchVisible(row) {
    const query = ctx.owners.speakers.transcriptSearchText.trim().toLowerCase();
    if (!query) return true;
    const searchable = (row.dataset.searchText || "").toLowerCase();
    return query.split(/\s+/).every(term => searchable.includes(term));
  }
  function transcriptReviewVisible(row) {
    if (ctx.owners.speakers.transcriptReviewFilter === "needs-review") return row.dataset.needsReview === "true" || row.dataset.groupNeedsReview === "true";
    if (ctx.owners.speakers.transcriptReviewFilter === "corrected") return row.dataset.corrected === "true" || row.dataset.groupCorrected === "true";
    return true;
  }
  function transcriptGroupTurnsEnabled() {
    return Boolean(groupTranscriptTurns && groupTranscriptTurns.checked);
  }
  function transcriptRowsInDisplayOrder() {
    return Array.from(sentences.querySelectorAll(".row")).sort((a, b) => {
      if (rowShouldSortBefore(a, b)) return -1;
      if (rowShouldSortBefore(b, a)) return 1;
      return 0;
    });
  }
  function cleanedTranscriptGroupText(rows) {
    return rows
      .map(row => row.dataset.text || "")
      .map(text => text.trim())
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }
  function removeTranscriptGroupCount(row) {
    const badge = row && row.querySelector(".group-count");
    if (badge) badge.remove();
  }
  function setTranscriptGroupCount(row, count) {
    const topLeft = row && row.querySelector(".top-left");
    if (!topLeft) return;
    let badge = topLeft.querySelector(".group-count");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "badge group-count";
      topLeft.appendChild(badge);
    }
    badge.textContent = `${count} sentence${count === 1 ? "" : "s"}`;
  }
  function updateSentenceRowVisibleTextRange(row, textValue, startValue, endValue, durationValue = null) {
    if (!row) return;
    const text = String(textValue || "");
    const start = finiteAudioSecond(startValue, 0);
    const end = finiteAudioSecond(endValue, start);
    const duration = durationValue === null ? Math.max(0, end - start) : Math.max(0, Number(durationValue || 0));
    const textNode = row.querySelector(".text");
    if (textNode) textNode.textContent = text;
    const durationNode = row.querySelector(".sentence-duration");
    if (durationNode) durationNode.textContent = secondsLabel(duration);
    const rangeNode = row.querySelector(".sentence-range");
    if (rangeNode) rangeNode.textContent = `(${secondsLabel(start)} - ${secondsLabel(end)})`;
  }
  function resetTranscriptGroupingRows() {
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      row.classList.remove("group-leader", "group-hidden", "group-needs-review", "group-corrected");
      delete row.dataset.groupHidden;
      delete row.dataset.groupLeader;
      delete row.dataset.groupSize;
      delete row.dataset.groupIndexes;
      delete row.dataset.groupText;
      delete row.dataset.groupStart;
      delete row.dataset.groupEnd;
      delete row.dataset.groupDuration;
      delete row.dataset.groupNeedsReview;
      delete row.dataset.groupCorrected;
      row.dataset.searchText = row.dataset.text || "";
      row.title = "";
      removeTranscriptGroupCount(row);
      updateSentenceRowVisibleTextRange(row, row.dataset.text || "", row.dataset.start, row.dataset.end);
    });
  }
  function transcriptRowCanGroup(row) {
    if (!row || row.dataset.realtime === "true") return false;
    if (row.dataset.pending === "true" || row.dataset.provisionalAssignment === "true") return false;
    if (row.dataset.provisionalSplit === "true") return false;
    const speakerId = row.dataset.speaker || "";
    if (!speakerId || speakerId === "UNKNOWN") return false;
    return Boolean((row.dataset.text || "").trim());
  }
  function applyTranscriptGroupRows(rows) {
    if (!Array.isArray(rows) || rows.length < 2) return;
    const leader = rows[0];
    const speakerId = leader.dataset.speaker || "";
    const text = cleanedTranscriptGroupText(rows);
    const start = finiteAudioSecond(leader.dataset.start, 0);
    const end = finiteAudioSecond(rows[rows.length - 1].dataset.end, start);
    const duration = rows.reduce((total, row) => {
      const rowStart = finiteAudioSecond(row.dataset.start, 0);
      const rowEnd = finiteAudioSecond(row.dataset.end, rowStart);
      return total + Math.max(0, rowEnd - rowStart);
    }, 0);
    const indexes = rows.map(row => String(row.dataset.index || "")).filter(Boolean);
    const hasReview = rows.some(row => row.dataset.needsReview === "true");
    const hasCorrection = rows.some(row => row.dataset.corrected === "true");
    leader.classList.add("group-leader");
    leader.classList.toggle("group-needs-review", hasReview);
    leader.classList.toggle("group-corrected", hasCorrection && !hasReview);
    leader.dataset.groupLeader = String(leader.dataset.index || "");
    leader.dataset.groupSize = String(rows.length);
    leader.dataset.groupIndexes = JSON.stringify(indexes);
    leader.dataset.groupText = text;
    leader.dataset.groupStart = String(start);
    leader.dataset.groupEnd = String(end);
    leader.dataset.groupDuration = String(duration);
    leader.dataset.groupNeedsReview = hasReview ? "true" : "false";
    leader.dataset.groupCorrected = hasCorrection ? "true" : "false";
    leader.dataset.searchText = text;
    leader.title = "Grouped display; turn grouping can be switched off in transcript settings.";
    updateSentenceRowVisibleTextRange(leader, text, start, end, duration);
    setTranscriptGroupCount(leader, rows.length);
    rows.slice(1).forEach(row => {
      row.classList.add("group-hidden");
      row.dataset.groupHidden = "true";
      row.dataset.groupLeader = speakerId ? String(leader.dataset.index || "") : "";
    });
  }
  function refreshTranscriptGrouping() {
    resetTranscriptGroupingRows();
    if (transcriptGroupTurnsEnabled()) {
      let currentGroup = [];
      transcriptRowsInDisplayOrder().forEach(row => {
        if (!transcriptRowCanGroup(row)) {
          applyTranscriptGroupRows(currentGroup);
          currentGroup = [];
          return;
        }
        if (!currentGroup.length || currentGroup[0].dataset.speaker === row.dataset.speaker) {
          currentGroup.push(row);
        } else {
          applyTranscriptGroupRows(currentGroup);
          currentGroup = [row];
        }
      });
      applyTranscriptGroupRows(currentGroup);
    }
    Array.from(sentences.querySelectorAll(".row")).forEach(refreshTranslationRow);
    refreshTranslationMenuStatus();
    refreshTranscriptVisibility();
    syncTranscriptSelectionState();
  }
  function refreshTranscriptVisibility() {
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      const hiddenByGroup = row.dataset.groupHidden === "true";
      row.hidden = hiddenByGroup || !speakerTranscriptVisible(row.dataset.speaker) || !transcriptSearchVisible(row) || !transcriptReviewVisible(row);
    });
  }
  function setTranscriptReviewFilter(filter) {
    ctx.owners.speakers.transcriptReviewFilter = ["all", "needs-review", "corrected"].includes(filter) ? filter : "all";
    reviewFilterButtons.forEach(button => {
      const active = button.dataset.reviewFilter === ctx.owners.speakers.transcriptReviewFilter;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    refreshTranscriptVisibility();
  }
  function correctionStatus(item) {
    const correction = item && item.correction;
    return correction && typeof correction === "object" ? String(correction.status || "") : "";
  }
  function rowIsCorrected(item) {
    const status = correctionStatus(item);
    return status === "user_corrected" || status === "user_confirmed";
  }
  function reviewReasonsForItem(item, displaySpeakerId, adoptedLiveSpeakerId) {
    const review = item && item.review && typeof item.review === "object" ? item.review : {};
    const reasons = Array.isArray(review.reasons) ? review.reasons.map(reason => String(reason || "").trim()).filter(Boolean) : [];
    if (displaySpeakerId && adoptedLiveSpeakerId && displaySpeakerId !== adoptedLiveSpeakerId) {
      reasons.push("conflicting live/final evidence");
    }
    if (rowIsCorrected(item)) return [];
    return Array.from(new Set(reasons));
  }
  function syncCorrectionUndoState(enabled = ctx.owners.transcript.hasUndoableCorrection) {
    ctx.owners.transcript.hasUndoableCorrection = Boolean(enabled);
    undoCorrectionButton.disabled = !ctx.owners.transcript.hasUndoableCorrection || sessionControlsLocked() || savedSessionReviewOpen();
  }
  function transcriptRowSelectionKey(row) {
    if (!row || row.dataset.realtime === "true" || row.dataset.selectable !== "true") return "";
    if (row.dataset.groupHidden === "true") return "";
    return String(row.dataset.index || "");
  }
  function selectableTranscriptRows() {
    return Array.from(sentences.querySelectorAll(".row[data-selectable='true']")).filter(row => Boolean(transcriptRowSelectionKey(row)));
  }
  function selectedTranscriptRows() {
    return selectableTranscriptRows().filter(row => ctx.owners.transcript.selectedTranscriptRowIndexes.has(transcriptRowSelectionKey(row)));
  }
  function selectedTranscriptIndexes() {
    return selectedTranscriptRows()
      .map(row => Number(row.dataset.index))
      .filter(index => Number.isFinite(index));
  }
  function pruneTranscriptSelection() {
    const available = new Set(selectableTranscriptRows().map(row => transcriptRowSelectionKey(row)).filter(Boolean));
    ctx.owners.transcript.selectedTranscriptRowIndexes.forEach(key => {
      if (!available.has(key)) ctx.owners.transcript.selectedTranscriptRowIndexes.delete(key);
    });
    if (ctx.owners.transcript.lastSelectedTranscriptRowIndex && !available.has(ctx.owners.transcript.lastSelectedTranscriptRowIndex)) {
      ctx.owners.transcript.lastSelectedTranscriptRowIndex = "";
    }
  }
  function setTranscriptSelectionRange(anchorKey, targetKey, selected) {
    const rows = selectableTranscriptRows();
    const anchorIndex = rows.findIndex(row => transcriptRowSelectionKey(row) === anchorKey);
    const targetIndex = rows.findIndex(row => transcriptRowSelectionKey(row) === targetKey);
    if (anchorIndex < 0 || targetIndex < 0) {
      if (selected) ctx.owners.transcript.selectedTranscriptRowIndexes.add(targetKey);
      else ctx.owners.transcript.selectedTranscriptRowIndexes.delete(targetKey);
      return;
    }
    const startIndex = Math.min(anchorIndex, targetIndex);
    const endIndex = Math.max(anchorIndex, targetIndex);
    rows.slice(startIndex, endIndex + 1).forEach(row => {
      const key = transcriptRowSelectionKey(row);
      if (!key) return;
      if (selected) ctx.owners.transcript.selectedTranscriptRowIndexes.add(key);
      else ctx.owners.transcript.selectedTranscriptRowIndexes.delete(key);
    });
  }
  function disableFollowLiveForTranscriptSelection() {
    if (!ctx.owners.speakers.followLiveEnabled && !followLive.checked) return;
    ctx.owners.speakers.followLiveEnabled = false;
    followLive.checked = false;
  }
  function setTranscriptRowSelected(row, selected, options = {}) {
    const key = transcriptRowSelectionKey(row);
    if (!key) return;
    if (selected) {
      disableFollowLiveForTranscriptSelection();
    }
    if (options.range && ctx.owners.transcript.lastSelectedTranscriptRowIndex) {
      setTranscriptSelectionRange(ctx.owners.transcript.lastSelectedTranscriptRowIndex, key, selected);
    } else if (selected) {
      ctx.owners.transcript.selectedTranscriptRowIndexes.add(key);
    } else {
      ctx.owners.transcript.selectedTranscriptRowIndexes.delete(key);
    }
    ctx.owners.transcript.lastSelectedTranscriptRowIndex = key;
    syncTranscriptSelectionState();
  }
  function clearTranscriptSelection() {
    ctx.owners.transcript.selectedTranscriptRowIndexes.clear();
    ctx.owners.transcript.lastSelectedTranscriptRowIndex = "";
    syncTranscriptSelectionState();
  }
  function commonSelectedSpeakerId(rows) {
    let speakerId = "";
    for (const row of rows) {
      const rowSpeaker = row.dataset.speaker || "";
      if (!rowSpeaker || rowSpeaker === "UNKNOWN") return "";
      if (!speakerId) {
        speakerId = rowSpeaker;
      } else if (speakerId !== rowSpeaker) {
        return "";
      }
    }
    return speakerId;
  }
  function selectedRowsNeedSpeakerChange(rows, speakerId) {
    if (!speakerId) return false;
    return rows.some(row => (row.dataset.speaker || "") !== speakerId);
  }
  function selectedRowsHaveUnconfirmed(rows) {
    return rows.some(row => row.dataset.corrected !== "true");
  }
  function syncBulkCorrectionSpeakerOptions(rows = selectedTranscriptRows()) {
    const speakers = Array.isArray(ctx.owners.speakers.speakerLibraryState.speakers) ? ctx.owners.speakers.speakerLibraryState.speakers : [];
    const createSpeakerAllowed = Boolean(commonSelectedSpeakerId(rows)) && !savedSessionReviewOpen();
    const previousValue = bulkCorrectionSpeaker.value || "";
    bulkCorrectionSpeaker.textContent = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Assign speaker...";
    bulkCorrectionSpeaker.appendChild(placeholder);
    speakers.forEach(speaker => {
      if (!speaker.id) return;
      const option = document.createElement("option");
      option.value = speaker.id;
      option.textContent = speakerPanelName(speaker);
      bulkCorrectionSpeaker.appendChild(option);
    });
    const separator = document.createElement("option");
    separator.value = "";
    separator.disabled = true;
    separator.textContent = "--------";
    bulkCorrectionSpeaker.appendChild(separator);
    const createOption = document.createElement("option");
    createOption.value = createSpeakerOptionValue;
    createOption.textContent = createSpeakerAllowed ? "Create new speaker" : "Create new speaker (select one speaker)";
    createOption.disabled = !createSpeakerAllowed;
    bulkCorrectionSpeaker.appendChild(createOption);
    if (speakers.some(speaker => speaker.id === previousValue)) {
      bulkCorrectionSpeaker.value = previousValue;
    } else if (previousValue === createSpeakerOptionValue && createSpeakerAllowed) {
      bulkCorrectionSpeaker.value = createSpeakerOptionValue;
    } else {
      bulkCorrectionSpeaker.value = "";
    }
  }
  function syncBulkCorrectionToolbar() {
    const rows = selectedTranscriptRows();
    const count = rows.length;
    syncBulkCorrectionSpeakerOptions(rows);
    const locked = sessionControlsLocked();
    const speakers = Array.isArray(ctx.owners.speakers.speakerLibraryState.speakers) ? ctx.owners.speakers.speakerLibraryState.speakers : [];
    const selectedSpeakerId = bulkCorrectionSpeaker.value || "";
    const createSpeakerSelected = selectedSpeakerId === createSpeakerOptionValue;
    const canCreateSpeaker = Boolean(commonSelectedSpeakerId(rows));
    selectionToolbar.hidden = count <= 0;
    selectionCount.textContent = `${count} selected`;
    bulkCorrectionSpeaker.disabled = locked || count <= 0 || (!speakers.length && !canCreateSpeaker);
    bulkReassignButton.textContent = createSpeakerSelected ? "Create speaker" : "Reassign";
    bulkReassignButton.disabled = locked
      || count <= 0
      || !selectedSpeakerId
      || (createSpeakerSelected ? !canCreateSpeaker : !selectedRowsNeedSpeakerChange(rows, selectedSpeakerId));
    bulkReassignButton.title = createSpeakerSelected && !canCreateSpeaker
      ? "Create new speaker requires selected rows from one known speaker."
      : "";
    bulkMarkCorrectButton.disabled = locked || count <= 0 || !selectedRowsHaveUnconfirmed(rows);
    clearSelectionButton.disabled = count <= 0;
  }
  function syncTranscriptSelectionState() {
    pruneTranscriptSelection();
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      const key = transcriptRowSelectionKey(row);
      const selected = Boolean(key && ctx.owners.transcript.selectedTranscriptRowIndexes.has(key));
      row.classList.toggle("selected", selected);
      if (key) {
        configureSentenceRowSelection(row);
      } else {
        row.classList.remove("selectable", "selected");
        row.removeAttribute("role");
        row.removeAttribute("aria-selected");
        row.removeAttribute("tabindex");
        row.onclick = null;
        row.onkeydown = null;
      }
    });
    syncBulkCorrectionToolbar();
  }
  function transcriptRowClickIsControl(target) {
    return target instanceof Element && Boolean(target.closest("button, input, select, textarea, a"));
  }
  function configureSentenceRowSelection(row) {
    if (!row || row.dataset.selectable !== "true") {
      if (row) {
        ctx.owners.transcript.selectedTranscriptRowIndexes.delete(String(row.dataset.index || ""));
        row.onclick = null;
        row.onkeydown = null;
        row.classList.remove("selectable", "selected");
        row.removeAttribute("role");
        row.removeAttribute("aria-selected");
        row.removeAttribute("tabindex");
      }
      return;
    }
    const selected = ctx.owners.transcript.selectedTranscriptRowIndexes.has(transcriptRowSelectionKey(row));
    row.classList.add("selectable");
    row.classList.toggle("selected", selected);
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", selected ? "true" : "false");
    row.tabIndex = 0;
    row.onclick = event => {
      if (transcriptRowClickIsControl(event.target)) return;
      const selection = window.getSelection ? String(window.getSelection() || "") : "";
      if (selection.trim()) return;
      if (Number(row.dataset.groupSize || 1) > 1 && groupTranscriptTurns.checked) {
        groupTranscriptTurns.checked = false;
        groupTranscriptTurns.dispatchEvent(new Event("change"));
      }
      const key = transcriptRowSelectionKey(row);
      setTranscriptRowSelected(row, !ctx.owners.transcript.selectedTranscriptRowIndexes.has(key), {range: event.shiftKey});
    };
    row.onkeydown = event => {
      if (transcriptRowClickIsControl(event.target)) return;
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      if (Number(row.dataset.groupSize || 1) > 1 && groupTranscriptTurns.checked) {
        groupTranscriptTurns.checked = false;
        groupTranscriptTurns.dispatchEvent(new Event("change"));
      }
      const key = transcriptRowSelectionKey(row);
      setTranscriptRowSelected(row, !ctx.owners.transcript.selectedTranscriptRowIndexes.has(key), {range: event.shiftKey});
    };
  }
  function setSpeakerFilter(speakerId, mode, active) {
    if (!speakerId) return;
    const target = mode === "mute" ? ctx.owners.speakers.mutedSpeakerIds : ctx.owners.speakers.soloSpeakerIds;
    if (active) {
      target.add(speakerId);
    } else {
      target.delete(speakerId);
    }
    refreshTranscriptVisibility();
    renderSpeakerPanel();
  }
  function setTranscriptSettingsOpen(open) {
    transcriptSettingsPanel.hidden = !open;
    transcriptSettingsButton.setAttribute("aria-expanded", open ? "true" : "false");
  }
  function applyTranscriptDisplaySettings() {
    transcriptPanel.classList.toggle("hide-tags", !showTranscriptTags.checked);
    transcriptPanel.classList.toggle("hide-time", !showTranscriptTime.checked);
    transcriptPanel.classList.toggle("hide-review-hints", !showTranscriptReviewHints.checked);
    transcriptPanel.classList.toggle("hide-speech-rate", !showTranscriptSpeechRate.checked);
    transcriptPanel.classList.toggle("hide-probabilities", !showTranscriptProbabilities.checked);
  }
  function updateSpeakerState(state) {
    if (!state || typeof state !== "object") return;
    ctx.owners.speakers.speakerLibraryState = {
      group_name: state.group_name || "",
      groups: Array.isArray(state.groups) ? state.groups : [],
      speakers: Array.isArray(state.speakers) ? state.speakers : [],
      embedding_provider: state.embedding_provider || "",
    };
    ctx.owners.speakers.speakerNames = {};
    ctx.owners.speakers.speakerLibraryState.speakers.forEach(speaker => {
      if (speaker.id) {
        ctx.owners.speakers.speakerNames[speaker.id] = speaker.display_name || speaker.name || speakerDisplayLabel(speaker.id);
      }
    });
    pruneSpeakerFilterState();
    refreshTranscriptVisibility();
    recomputeRenderedSpeakerSentenceCounts();
    if (!hasCurrentSessionSpeakerCounts()) {
      syncSpeakerSessionBaselines(ctx.owners.speakers.speakerLibraryState);
    }
    updateSpeakerCount();
    renderSpeakerPanel();
    refreshSpeakerRows();
    syncBulkCorrectionToolbar();
  }
  function setSpeakerTab(tabName) {
    const nextTab = ["settings", "sessions", "ask", "intelligence"].includes(tabName) ? tabName : "speakers";
    speakerTabButtons.forEach(button => {
      const active = button.dataset.speakerTab === nextTab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    speakerTabPanels.forEach(panel => {
      panel.hidden = panel.dataset.speakerPanel !== nextTab;
    });
    if (nextTab === "sessions") {
      fetchSavedSessions().catch(error => log(`Refresh sessions failed: ${error.message}`));
    }
    if (nextTab === "ask" && ctx.api.refreshMeetingChatScope) {
      ctx.api.refreshMeetingChatScope();
    }
    syncSavedSessionsAutoRefresh();
    renderMeetingIntelligencePanel();
  }
  function selectedSpeaker() {
    return ctx.owners.speakers.speakerLibraryState.speakers.find(speaker => speaker.id === ctx.owners.reference.editingSpeakerId) || null;
  }
  function recomputeRenderedSpeakerSentenceCounts() {
    const counts = {};
    const speakingSeconds = {};
    let hasFinalRows = false;
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      if (row.dataset.realtime === "true") return;
      hasFinalRows = true;
      const speakerId = row.dataset.speaker;
      if (!speakerId || speakerId === "UNKNOWN") return;
      counts[speakerId] = (counts[speakerId] || 0) + 1;
      const start = Number(row.dataset.start || 0);
      const end = Number(row.dataset.end || 0);
      speakingSeconds[speakerId] = (speakingSeconds[speakerId] || 0) + Math.max(0, end - start);
    });
    ctx.owners.speakers.renderedSpeakerSentenceCounts = counts;
    ctx.owners.speakers.renderedSpeakerSpeakingSeconds = speakingSeconds;
    ctx.owners.speakers.hasRenderedFinalSentenceRows = hasFinalRows;
  }
  function hasCurrentSessionSpeakerCounts() {
    if (ctx.owners.speakers.hasRenderedFinalSentenceRows) return true;
    return Object.values(ctx.owners.speakers.fastSpeakerPanelStats).some(stats => (
      Number((stats && stats.count) || 0) > 0
      || Number((stats && stats.speakingSeconds) || 0) > 0
    ));
  }
  function ensureSpeakerPanelSpeaker(speakerId) {
    if (!speakerId || speakerId === "UNKNOWN") return;
    if (ctx.owners.speakers.speakerLibraryState.speakers.some(speaker => speaker.id === speakerId)) return;
    const speaker = {
      id: speakerId,
      name: "",
      display_name: speakerDisplayLabel(speakerId),
      source: "detected",
      locked: false,
      sentence_count: 0,
      speech_seconds: 0,
      reference_audio: "",
    };
    ctx.owners.speakers.speakerLibraryState = {
        ...ctx.owners.speakers.speakerLibraryState,
        speakers: [...ctx.owners.speakers.speakerLibraryState.speakers, speaker],
    };
    ctx.owners.speakers.speakerNames[speakerId] = speaker.display_name;
    pruneSpeakerFilterState();
    updateSpeakerCount();
    renderSpeakerPanel();
  }
  function speakerBaselineSentenceCount(speaker) {
    const speakerId = speaker && speaker.id;
    if (!speakerId) return Number((speaker && speaker.sentence_count) || 0);
    if (Object.prototype.hasOwnProperty.call(ctx.owners.speakers.speakerSessionBaselineSentenceCounts, speakerId)) {
      return Number(ctx.owners.speakers.speakerSessionBaselineSentenceCounts[speakerId] || 0);
    }
    return 0;
  }
  function speakerBaselineSpeakingSeconds(speaker) {
    const speakerId = speaker && speaker.id;
    if (!speakerId) return Number((speaker && speaker.speech_seconds) || 0);
    if (Object.prototype.hasOwnProperty.call(ctx.owners.speakers.speakerSessionBaselineSpeakingSeconds, speakerId)) {
      return Number(ctx.owners.speakers.speakerSessionBaselineSpeakingSeconds[speakerId] || 0);
    }
    return 0;
  }
  function speakerCurrentSessionSentenceCount(speakerId) {
    const rendered = Number(ctx.owners.speakers.renderedSpeakerSentenceCounts[speakerId] || 0);
    const fast = Number(((ctx.owners.speakers.fastSpeakerPanelStats[speakerId] || {}).count) || 0);
    if (ctx.owners.speakers.hasRenderedFinalSentenceRows) return rendered;
    return fast;
  }
  function speakerCurrentSessionSpeakingSeconds(speakerId) {
    const rendered = Number(ctx.owners.speakers.renderedSpeakerSpeakingSeconds[speakerId] || 0);
    const fast = Number(((ctx.owners.speakers.fastSpeakerPanelStats[speakerId] || {}).speakingSeconds) || 0);
    if (ctx.owners.speakers.hasRenderedFinalSentenceRows) return rendered;
    return fast;
  }
  function speakerPanelSentenceCount(speaker) {
    const speakerId = speaker && speaker.id;
    if (!speakerId) return Number((speaker && speaker.sentence_count) || 0);
    return speakerBaselineSentenceCount(speaker) + speakerCurrentSessionSentenceCount(speakerId);
  }
  function speakerPanelSpeakingSeconds(speaker) {
    const speakerId = speaker && speaker.id;
    if (!speakerId) return Number((speaker && speaker.speech_seconds) || 0);
    return speakerBaselineSpeakingSeconds(speaker) + speakerCurrentSessionSpeakingSeconds(speakerId);
  }
  function speakerPanelCountUnit() {
    return "sentence";
  }
  function refreshSpeakerPanelSentenceCounts() {
    recomputeRenderedSpeakerSentenceCounts();
    Array.from(speakerList.querySelectorAll(".speaker-item")).forEach(row => {
      const speaker = ctx.owners.speakers.speakerLibraryState.speakers.find(item => item.id === row.dataset.speakerId);
      const count = row.querySelector(".speaker-sentence-count");
      if (speaker && count) {
        count.textContent = speakerSentenceText(
          speakerPanelSentenceCount(speaker),
          speakerPanelSpeakingSeconds(speaker),
          speakerPanelCountUnit(),
        );
      }
    });
  }
  function refreshLiveSpeakerHighlight() {
    Array.from(speakerList.querySelectorAll(".speaker-item")).forEach(row => {
      const active = Boolean(ctx.owners.transcript.currentLiveSpeakerId) && row.dataset.speakerId === ctx.owners.transcript.currentLiveSpeakerId;
      row.classList.toggle("live-speaker", active);
      const indicator = row.querySelector(".speaker-live-indicator");
      if (active && !indicator) {
        const titleRow = row.querySelector(".speaker-title-row");
        if (titleRow) titleRow.appendChild(createSpeakerLiveIndicator());
      } else if (!active && indicator) {
        indicator.remove();
      }
    });
  }
  function clearFallbackLiveSpeaker() {
    ctx.owners.transcript.fallbackLiveSpeakerId = "";
    ctx.owners.transcript.fallbackLiveSpeakerUntilMs = 0;
    if (ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer) {
      clearTimeout(ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer);
      ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer = null;
    }
    if (ctx.owners.transcript.fallbackLiveSpeakerClearTimer) {
      clearTimeout(ctx.owners.transcript.fallbackLiveSpeakerClearTimer);
      ctx.owners.transcript.fallbackLiveSpeakerClearTimer = null;
    }
  }
  function clearTranscriptLiveSpeakerExpiryTimer() {
    if (ctx.owners.transcript.transcriptLiveSpeakerExpiryTimer) {
      clearTimeout(ctx.owners.transcript.transcriptLiveSpeakerExpiryTimer);
      ctx.owners.transcript.transcriptLiveSpeakerExpiryTimer = null;
    }
  }
  function clearLiveSpeakerState() {
    ctx.owners.transcript.currentLiveSpeakerId = "";
    ctx.owners.transcript.transcriptLiveSpeakerId = "";
    ctx.owners.transcript.transcriptLiveSpeakerOverrideId = "";
    ctx.owners.speakers.fastSpeakerPanelStats = {};
    ctx.owners.speakers.fastSpeakerPanelLastRight = null;
    ctx.owners.transcript.liveSpeakerTimeline = [];
    clearFallbackLiveSpeaker();
    clearTranscriptLiveSpeakerExpiryTimer();
    refreshRealtimeRowsFromLiveSpeaker();
  }
  function activeFallbackLiveSpeakerId(nowMs = performance.now()) {
    if (!ctx.owners.transcript.fallbackLiveSpeakerId) return "";
    if (ctx.owners.transcript.fallbackLiveSpeakerUntilMs > nowMs) return ctx.owners.transcript.fallbackLiveSpeakerId;
    clearFallbackLiveSpeaker();
    return "";
  }
  function normalizedLiveSpeakerId(speakerId) {
    const value = String(speakerId || "").trim();
    return value && value !== "UNKNOWN" ? value : "";
  }
  function finiteAudioSecond(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }
  function pruneLiveSpeakerTimeline(minEndSeconds) {
    const cutoff = Math.max(0, finiteAudioSecond(minEndSeconds, 0));
    ctx.owners.transcript.liveSpeakerTimeline = ctx.owners.transcript.liveSpeakerTimeline.filter(item => finiteAudioSecond(item.end, 0) >= cutoff);
  }
  function rememberLiveSpeakerEvidence(speakerId, item) {
    const normalizedSpeakerId = normalizedLiveSpeakerId(speakerId);
    if (!normalizedSpeakerId || !item) return;
    const fallbackEnd = playbackSeconds();
    const end = finiteAudioSecond(item.end, fallbackEnd);
    const audioLength = Math.max(0, finiteAudioSecond(item.audio_length_seconds, 0));
    const fallbackStart = Math.max(0, end - audioLength);
    const start = finiteAudioSecond(item.start, fallbackStart);
    if (!(end > start)) return;
    ctx.owners.transcript.liveSpeakerTimeline.push({speakerId: normalizedSpeakerId, start, end});
    pruneLiveSpeakerTimeline(end - 90);
  }
  function realtimeDominanceScoredEnd(start, end) {
    const duration = Math.max(0, end - start);
    if (duration <= 3) return end;
    const tailSeconds = Math.min(3, Math.max(2, duration * 0.25));
    return Math.max(start + 0.1, end - tailSeconds);
  }
  function realtimeRowHasSpeakerEvidence(start, end) {
    const rowStart = Math.max(0, finiteAudioSecond(start, 0));
    const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
    if (!(rowEnd > rowStart)) return false;
    return ctx.owners.transcript.liveSpeakerTimeline.some(item => {
      const speakerId = normalizedLiveSpeakerId(item.speakerId);
      if (!speakerId) return false;
      const overlapStart = Math.max(rowStart, finiteAudioSecond(item.start, rowStart));
      const overlapEnd = Math.min(rowEnd, finiteAudioSecond(item.end, rowStart));
      return overlapEnd - overlapStart > 0;
    });
  }
  function dominantRealtimeSpeakerId(start, end) {
    const rowStart = Math.max(0, finiteAudioSecond(start, 0));
    const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
    if (!(rowEnd > rowStart)) return "";
    const scoredEnd = realtimeDominanceScoredEnd(rowStart, rowEnd);
    const scoredSeconds = Math.max(0, scoredEnd - rowStart);
    if (scoredSeconds <= 0) return "";
    const weights = {};
    ctx.owners.transcript.liveSpeakerTimeline.forEach(item => {
      const speakerId = normalizedLiveSpeakerId(item.speakerId);
      if (!speakerId) return;
      const overlapStart = Math.max(rowStart, finiteAudioSecond(item.start, rowStart));
      const overlapEnd = Math.min(scoredEnd, finiteAudioSecond(item.end, rowStart));
      const seconds = Math.max(0, overlapEnd - overlapStart);
      if (seconds <= 0) return;
      weights[speakerId] = (weights[speakerId] || 0) + seconds;
    });
    let bestSpeakerId = "";
    let bestSeconds = 0;
    Object.entries(weights).forEach(([speakerId, seconds]) => {
      if (seconds > bestSeconds) {
        bestSpeakerId = speakerId;
        bestSeconds = seconds;
      }
    });
    const requiredSeconds = Math.max(0.3, scoredSeconds * 0.5);
    return bestSeconds >= requiredSeconds ? bestSpeakerId : "";
  }
  function realtimeTailSpeakerChange(start, end, currentSpeakerId = "") {
    const rowStart = Math.max(0, finiteAudioSecond(start, 0));
    const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
    if (rowEnd - rowStart < 4) return null;
    const scoredEnd = realtimeDominanceScoredEnd(rowStart, rowEnd);
    const current = normalizedLiveSpeakerId(currentSpeakerId);
    let best = null;
    ctx.owners.transcript.liveSpeakerTimeline.forEach(item => {
      const speakerId = normalizedLiveSpeakerId(item.speakerId);
      if (!speakerId || speakerId === current) return;
      const evidenceStart = Math.max(rowStart, finiteAudioSecond(item.start, rowStart));
      const evidenceEnd = Math.min(rowEnd, finiteAudioSecond(item.end, rowStart));
      if (!(evidenceEnd > evidenceStart)) return;
      const tailStart = Math.max(evidenceStart, scoredEnd);
      const tailSeconds = Math.max(0, evidenceEnd - tailStart);
      if (tailSeconds < 0.4) return;
      const score = tailSeconds + evidenceEnd * 0.001;
      if (!best || score > best.score) {
        best = {speakerId, start: evidenceStart, end: evidenceEnd, tailSeconds, score};
      }
    });
    return best;
  }
  function realtimeRowDisplaySpeakerId(rawSpeakerId = "", start = 0, end = 0, previousSpeakerId = "") {
    const dominantSpeakerId = dominantRealtimeSpeakerId(start, end);
    if (dominantSpeakerId) return dominantSpeakerId;
    const previousNormalizedSpeakerId = normalizedLiveSpeakerId(previousSpeakerId);
    if (previousNormalizedSpeakerId) return previousNormalizedSpeakerId;
    if (realtimeRowHasSpeakerEvidence(start, end)) return "";
    const rawNormalizedSpeakerId = normalizedLiveSpeakerId(rawSpeakerId);
    if (!rawNormalizedSpeakerId) return "";
    const rowStart = Math.max(0, finiteAudioSecond(start, 0));
    const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
    if (rowEnd - rowStart > 3) return "";
    return rawNormalizedSpeakerId;
  }
  function lastPunctuationTextSplit(textValue) {
    const value = String(textValue || "").replace(/\s+/g, " ").trim();
    if (!value) return null;
    const boundaryPattern = /[.!?]["')\]]*\s+/g;
    const candidates = [];
    let match = null;
    while ((match = boundaryPattern.exec(value)) !== null) {
      const boundary = match.index + match[0].length;
      const prefixText = value.slice(0, boundary).trim();
      const suffixText = value.slice(boundary).trim();
      if (!/[A-Za-z0-9]/.test(prefixText) || !/[A-Za-z0-9]/.test(suffixText)) continue;
      candidates.push({
        prefixText,
        suffixText,
        suffixRatio: suffixText.length / Math.max(1, value.length),
      });
    }
    return candidates.length ? candidates[candidates.length - 1] : null;
  }
  function provisionalRealtimeVisualSplit(item, displaySpeakerId, start, end) {
    if (!item || !item.realtime) return null;
    const tailChange = realtimeTailSpeakerChange(start, end, displaySpeakerId);
    if (!tailChange) return null;
    const textSplit = lastPunctuationTextSplit(item.text);
    if (!textSplit) return null;
    const rowStart = Math.max(0, finiteAudioSecond(start, 0));
    const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
    const duration = rowEnd - rowStart;
    if (duration <= 0) return null;
    const splitStart = Math.min(Math.max(tailChange.start, rowStart + 0.2), rowEnd - 0.2);
    const tailDuration = Math.max(0, rowEnd - splitStart);
    if (tailDuration < 0.5) return null;
    const tailRatio = tailDuration / duration;
    const maxSuffixRatio = Math.max(0.25, Math.min(0.5, tailRatio + 0.25));
    if (textSplit.suffixRatio > maxSuffixRatio) return null;
    return {
      speakerId: tailChange.speakerId,
      prefixText: textSplit.prefixText,
      suffixText: textSplit.suffixText,
      splitStart,
      end: rowEnd,
    };
  }
  function applyRealtimeRowSpeaker(row, speakerId) {
    const normalizedSpeakerId = normalizedLiveSpeakerId(speakerId);
    const color = speakerColor(normalizedSpeakerId);
    row.dataset.speaker = normalizedSpeakerId || "UNKNOWN";
    row.classList.toggle("live-speaker-row", Boolean(normalizedSpeakerId));
    row.style.setProperty("--live-row-color", color || "#8F9BA8");
    const badge = row.querySelector(".speaker-name");
    if (!badge) return;
    badge.className = `${normalizedSpeakerId ? "badge" : "badge unknown"} speaker-name`;
    badge.textContent = speakerDisplayLabel(normalizedSpeakerId);
    if (color) {
      badge.style.color = color;
      badge.style.borderColor = color;
      badge.style.background = "#0B1015";
    } else {
      badge.style.removeProperty("color");
      badge.style.removeProperty("border-color");
      badge.style.removeProperty("background");
    }
  }
  function refreshRealtimeRowsFromLiveSpeaker() {
    Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => {
      if (!row.isConnected) return;
      if (row.dataset.realtimeSettling === "true") return;
      if (row.dataset.provisionalSplit === "true") {
        applyRealtimeRowSpeaker(row, row.dataset.speaker);
        return;
      }
      clearProvisionalRealtimeSplitsFor(row.dataset.index);
      restoreRealtimeRowFullPreview(row);
      const rowStart = row.dataset.start;
      const rowEnd = row.dataset.fullEnd || row.dataset.end;
      const rowRawSpeaker = row.dataset.fullRawSpeaker || row.dataset.rawSpeaker || "";
      applyRealtimeRowSpeaker(
        row,
        realtimeRowDisplaySpeakerId(
          rowRawSpeaker,
          rowStart,
          rowEnd,
          row.dataset.speaker,
        ),
      );
      const visualSplit = provisionalRealtimeVisualSplit(
        {realtime: true, text: row.dataset.fullText || row.dataset.text || ""},
        normalizedLiveSpeakerId(row.dataset.speaker),
        rowStart,
        rowEnd,
      );
      if (visualSplit) {
        applyProvisionalRealtimeVisualSplit(row, visualSplit);
      }
    });
    updateCurrentLiveSpeakerFromRealtimeRows();
    refreshSpeakerPanelSentenceCounts();
    refreshTranscriptVisibility();
  }
  function transcriptHighlightMaxLagSeconds() {
    const value = Number(liveSpeakerConfig.transcript_highlight_max_lag_seconds);
    return Number.isFinite(value) ? value : -1;
  }
  function realtimeRowWithinTranscriptHighlightWindow(row) {
    if (!row) return false;
    const maxLag = transcriptHighlightMaxLagSeconds();
    if (maxLag < 0) return true;
    const start = finiteAudioSecond(row.dataset.start, NaN);
    const end = finiteAudioSecond(row.dataset.end, NaN);
    const playback = playbackSeconds();
    if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
    if (playback < start - 0.25) return false;
    if (playback > end + maxLag) return false;
    return true;
  }
  function probabilityForSpeakerId(probabilities, speakerId) {
    const key = speakerProbabilityKey(speakerId);
    if (!key || !probabilities || typeof probabilities !== "object") return 0;
    const value = Number(probabilities[key]);
    return Number.isFinite(value) ? Math.max(0, value) : 0;
  }
  function probabilityLeadOverUnknown(probabilities, speakerId) {
    const speakerProbability = probabilityForSpeakerId(probabilities, speakerId);
    const unknownProbability = Number(probabilities && probabilities.unknown);
    return speakerProbability - (Number.isFinite(unknownProbability) ? Math.max(0, unknownProbability) : 0);
  }
  function transcriptOverrideCandidate(row) {
    if (!row || !liveSpeakerConfig.highlight_transcript) return "";
    if (!realtimeRowWithinTranscriptHighlightWindow(row)) return "";
    const minProbability = Number(liveSpeakerConfig.transcript_override_min_probability);
    if (!Number.isFinite(minProbability) || minProbability > 1) return "";
    const speakerId = normalizedLiveSpeakerId(row.dataset.rawSpeaker);
    if (!speakerId) return "";
    const probability = Number(row.dataset.rawSpeakerProbability || 0);
    if (!Number.isFinite(probability) || probability < minProbability) return "";
    const minMargin = Number(liveSpeakerConfig.transcript_override_min_margin || 0);
    const margin = Number(row.dataset.rawSpeakerUnknownMargin || 0);
    if (Number.isFinite(minMargin) && margin < minMargin) return "";
    return speakerId;
  }
  function realtimeRowTranscriptLiveSpeakerId(row) {
    if (!row || !liveSpeakerConfig.highlight_transcript) return "";
    const speakerId = row.dataset.speaker !== "UNKNOWN" ? normalizedLiveSpeakerId(row.dataset.speaker) : "";
    if (!speakerId) return "";
    return realtimeRowWithinTranscriptHighlightWindow(row) ? speakerId : "";
  }
  function scheduleTranscriptLiveSpeakerExpiry(row) {
    clearTranscriptLiveSpeakerExpiryTimer();
    const maxLag = transcriptHighlightMaxLagSeconds();
    if (!row || maxLag < 0 || !ctx.owners.transcript.transcriptLiveSpeakerId) return;
    const end = finiteAudioSecond(row.dataset.end, NaN);
    if (!Number.isFinite(end)) return;
    const remainingMs = Math.max(0, (end + maxLag - playbackSeconds()) * 1000);
    ctx.owners.transcript.transcriptLiveSpeakerExpiryTimer = setTimeout(updateCurrentLiveSpeakerFromRealtimeRows, remainingMs + 25);
  }
  function reconcileLiveSpeakerHighlight() {
    ctx.owners.transcript.currentLiveSpeakerId = ctx.owners.transcript.transcriptLiveSpeakerOverrideId
      || activeFallbackLiveSpeakerId()
      || (liveSpeakerConfig.highlight_transcript ? ctx.owners.transcript.transcriptLiveSpeakerId : "");
    refreshLiveSpeakerHighlight();
  }
  function scheduleFallbackLiveSpeakerExpiry() {
    if (ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer) {
      clearTimeout(ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer);
      ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer = null;
    }
    const remainingMs = ctx.owners.transcript.fallbackLiveSpeakerUntilMs - performance.now();
    if (!ctx.owners.transcript.fallbackLiveSpeakerId || remainingMs <= 0) {
      refreshRealtimeRowsFromLiveSpeaker();
      return;
    }
    ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer = setTimeout(refreshRealtimeRowsFromLiveSpeaker, remainingMs + 25);
  }
  function applyFallbackLiveSpeaker(item) {
    const speakerId = normalizedLiveSpeakerId(item && (item.assigned_speaker || item.speaker_id));
    if (!speakerId) return;
    if (item.only_if_no_live_speaker && ctx.owners.transcript.currentLiveSpeakerId) return;
    if (ctx.owners.transcript.fallbackLiveSpeakerClearTimer) {
      clearTimeout(ctx.owners.transcript.fallbackLiveSpeakerClearTimer);
      ctx.owners.transcript.fallbackLiveSpeakerClearTimer = null;
    }
    applyFastSpeakerPanelSignal(item);
    rememberLiveSpeakerEvidence(speakerId, item);
    const holdSeconds = Math.max(0, Number(item.hold_seconds || 2.0));
    ctx.owners.transcript.fallbackLiveSpeakerId = speakerId;
    ctx.owners.transcript.fallbackLiveSpeakerUntilMs = performance.now() + holdSeconds * 1000;
    scheduleFallbackLiveSpeakerExpiry();
    refreshRealtimeRowsFromLiveSpeaker();
  }
  function clearFallbackLiveSpeakerFromProbe(item) {
    const speakerId = normalizedLiveSpeakerId(item && (item.assigned_speaker || item.speaker_id));
    if (speakerId && ctx.owners.transcript.fallbackLiveSpeakerId && speakerId !== ctx.owners.transcript.fallbackLiveSpeakerId) return;
    const debounceSeconds = Math.max(0, Number(liveSpeakerConfig.unknown_clear_debounce_seconds || 0));
    if (ctx.owners.transcript.fallbackLiveSpeakerId && item && item.reason === "unknown" && debounceSeconds > 0) {
      const expectedSpeakerId = ctx.owners.transcript.fallbackLiveSpeakerId;
      const debounceMs = debounceSeconds * 1000;
      if (ctx.owners.transcript.fallbackLiveSpeakerClearTimer) clearTimeout(ctx.owners.transcript.fallbackLiveSpeakerClearTimer);
      ctx.owners.transcript.fallbackLiveSpeakerUntilMs = Math.max(ctx.owners.transcript.fallbackLiveSpeakerUntilMs, performance.now() + debounceMs);
      ctx.owners.transcript.fallbackLiveSpeakerClearTimer = setTimeout(() => {
        ctx.owners.transcript.fallbackLiveSpeakerClearTimer = null;
        if (ctx.owners.transcript.fallbackLiveSpeakerId === expectedSpeakerId) {
          clearFallbackLiveSpeaker();
          refreshRealtimeRowsFromLiveSpeaker();
        }
      }, debounceMs);
      scheduleFallbackLiveSpeakerExpiry();
      refreshRealtimeRowsFromLiveSpeaker();
      return;
    }
    clearFallbackLiveSpeaker();
    refreshRealtimeRowsFromLiveSpeaker();
  }
  function applyFastSpeakerPanelSignal(item) {
    const speakerId = item && (item.assigned_speaker || item.speaker_id);
    if (!speakerId || speakerId === "UNKNOWN") return;
    ensureSpeakerPanelSpeaker(speakerId);
    const start = Number(item.start || 0);
    const end = Number(item.end || start);
    if (!(end > start)) return;
    const previousRight = ctx.owners.speakers.fastSpeakerPanelLastRight === null ? start : ctx.owners.speakers.fastSpeakerPanelLastRight;
    const uncoveredStart = Math.max(start, previousRight);
    const seconds = Math.max(0, end - uncoveredStart);
    const current = ctx.owners.speakers.fastSpeakerPanelStats[speakerId] || {count: 0, speakingSeconds: 0};
    ctx.owners.speakers.fastSpeakerPanelStats = {
      ...ctx.owners.speakers.fastSpeakerPanelStats,
      [speakerId]: {
        count: current.count + 1,
        speakingSeconds: current.speakingSeconds + seconds,
        lastStart: start,
        lastEnd: end,
      },
    };
    ctx.owners.speakers.fastSpeakerPanelLastRight = Math.max(previousRight, end);
    refreshSpeakerPanelSentenceCounts();
  }
  function updateCurrentLiveSpeakerFromRealtimeRows() {
    const realtimeRows = Array.from(sentences.querySelectorAll(".row[data-realtime='true']"))
      .filter(row => row.dataset.realtimeSettling !== "true");
    const activeRow = realtimeRows[realtimeRows.length - 1] || null;
    ctx.owners.transcript.transcriptLiveSpeakerId = realtimeRowTranscriptLiveSpeakerId(activeRow);
    ctx.owners.transcript.transcriptLiveSpeakerOverrideId = transcriptOverrideCandidate(activeRow);
    scheduleTranscriptLiveSpeakerExpiry(activeRow);
    reconcileLiveSpeakerHighlight();
  }

  Object.assign(ctx.api, {activeFallbackLiveSpeakerId, applyFallbackLiveSpeaker, applyFastSpeakerPanelSignal, applyRealtimeRowSpeaker, applyTranscriptDisplaySettings, applyTranscriptGroupRows, cleanedTranscriptGroupText, clearFallbackLiveSpeaker, clearFallbackLiveSpeakerFromProbe, clearLiveSpeakerState, clearTranscriptLiveSpeakerExpiryTimer, clearTranscriptSelection, commonSelectedSpeakerId, configureSentenceRowSelection, correctionStatus, disableFollowLiveForTranscriptSelection, dominantRealtimeSpeakerId, ensureSpeakerPanelSpeaker, finiteAudioSecond, hasCurrentSessionSpeakerCounts, lastPunctuationTextSplit, normalizedLiveSpeakerId, probabilityForSpeakerId, probabilityLeadOverUnknown, provisionalRealtimeVisualSplit, pruneLiveSpeakerTimeline, pruneTranscriptSelection, realtimeDominanceScoredEnd, realtimeRowDisplaySpeakerId, realtimeRowHasSpeakerEvidence, realtimeRowTranscriptLiveSpeakerId, realtimeRowWithinTranscriptHighlightWindow, realtimeTailSpeakerChange, recomputeRenderedSpeakerSentenceCounts, reconcileLiveSpeakerHighlight, refreshLiveSpeakerHighlight, refreshRealtimeRowsFromLiveSpeaker, refreshSpeakerPanelSentenceCounts, refreshTranscriptGrouping, refreshTranscriptVisibility, rememberLiveSpeakerEvidence, removeTranscriptGroupCount, resetTranscriptGroupingRows, reviewReasonsForItem, rowIsCorrected, scheduleFallbackLiveSpeakerExpiry, scheduleTranscriptLiveSpeakerExpiry, selectableTranscriptRows, selectedRowsHaveUnconfirmed, selectedRowsNeedSpeakerChange, selectedSpeaker, selectedTranscriptIndexes, selectedTranscriptRows, setSpeakerFilter, setSpeakerTab, setTranscriptGroupCount, setTranscriptReviewFilter, setTranscriptRowSelected, setTranscriptSelectionRange, setTranscriptSettingsOpen, speakerBaselineSentenceCount, speakerBaselineSpeakingSeconds, speakerCurrentSessionSentenceCount, speakerCurrentSessionSpeakingSeconds, speakerPanelCountUnit, speakerPanelSentenceCount, speakerPanelSpeakingSeconds, syncBulkCorrectionSpeakerOptions, syncBulkCorrectionToolbar, syncCorrectionUndoState, syncTranscriptSelectionState, transcriptGroupTurnsEnabled, transcriptHighlightMaxLagSeconds, transcriptOverrideCandidate, transcriptReviewVisible, transcriptRowCanGroup, transcriptRowClickIsControl, transcriptRowSelectionKey, transcriptRowsInDisplayOrder, transcriptSearchVisible, updateCurrentLiveSpeakerFromRealtimeRows, updateSentenceRowVisibleTextRange, updateSpeakerState});
}
