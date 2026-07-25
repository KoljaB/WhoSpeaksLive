export function installTranscriptRender(ctx) {
  const {audio, languageConfig, realtimeSettleRemovalDelayMs, sentences, source, start, state, translationIncludeOriginalControl} = ctx;
  const applyTranslationCollection = (...args) => ctx.api.applyTranslationCollection(...args), configureSentenceRowSelection = (...args) => ctx.api.configureSentenceRowSelection(...args), correctionStatus = (...args) => ctx.api.correctionStatus(...args), createReviewReasonGroup = (...args) => ctx.api.createReviewReasonGroup(...args), downloadJsonFile = (...args) => ctx.api.downloadJsonFile(...args), effectiveTranslationDisplayMode = (...args) => ctx.api.effectiveTranslationDisplayMode(...args), finiteAudioSecond = (...args) => ctx.api.finiteAudioSecond(...args), forgetSentenceTranslations = (...args) => ctx.api.forgetSentenceTranslations(...args), itemIsBeforeClearedTranscriptBoundary = (...args) => ctx.api.itemIsBeforeClearedTranscriptBoundary(...args), log = (...args) => ctx.api.log(...args), normalizedLiveSpeakerId = (...args) => ctx.api.normalizedLiveSpeakerId(...args), probabilityColor = (...args) => ctx.api.probabilityColor(...args), probabilityDisplayLabel = (...args) => ctx.api.probabilityDisplayLabel(...args), probabilityForSpeakerId = (...args) => ctx.api.probabilityForSpeakerId(...args), probabilityLeadOverUnknown = (...args) => ctx.api.probabilityLeadOverUnknown(...args), provisionalRealtimeVisualSplit = (...args) => ctx.api.provisionalRealtimeVisualSplit(...args), queueBrowserPreferredTranslationsForSource = (...args) => ctx.api.queueBrowserPreferredTranslationsForSource(...args), realtimeRowDisplaySpeakerId = (...args) => ctx.api.realtimeRowDisplaySpeakerId(...args), refreshSpeakerPanelSentenceCounts = (...args) => ctx.api.refreshSpeakerPanelSentenceCounts(...args), refreshTranscriptGrouping = (...args) => ctx.api.refreshTranscriptGrouping(...args), refreshTranscriptVisibility = (...args) => ctx.api.refreshTranscriptVisibility(...args), reviewReasonsForItem = (...args) => ctx.api.reviewReasonsForItem(...args), revisionSpeakerId = (...args) => ctx.api.revisionSpeakerId(...args), rowIsCorrected = (...args) => ctx.api.rowIsCorrected(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), scheduleSavedSessionsRefresh = (...args) => ctx.api.scheduleSavedSessionsRefresh(...args), scrollSentencesToBottom = (...args) => ctx.api.scrollSentencesToBottom(...args), selectedTranslationCodesForDisplay = (...args) => ctx.api.selectedTranslationCodesForDisplay(...args), sentenceRevisionLabel = (...args) => ctx.api.sentenceRevisionLabel(...args), speakerColor = (...args) => ctx.api.speakerColor(...args), speakerDisplayLabel = (...args) => ctx.api.speakerDisplayLabel(...args), speakerProbabilityKey = (...args) => ctx.api.speakerProbabilityKey(...args), syncBulkCorrectionToolbar = (...args) => ctx.api.syncBulkCorrectionToolbar(...args), transcriptGroupTurnsEnabled = (...args) => ctx.api.transcriptGroupTurnsEnabled(...args), translationLanguageName = (...args) => ctx.api.translationLanguageName(...args), translationStateMap = (...args) => ctx.api.translationStateMap(...args), translationStateMatchesRow = (...args) => ctx.api.translationStateMatchesRow(...args), updateCurrentLiveSpeakerFromRealtimeRows = (...args) => ctx.api.updateCurrentLiveSpeakerFromRealtimeRows(...args);
  const toInternalSpeakerId = (...args) => ctx.api.toInternalSpeakerId(...args);
  const realtimeSentenceSplitSpawnMs = 300;
  const realtimeSentenceSplitTransferMs = 600;
  const queuedRealtimeSentenceSplits = [];
  let realtimeSentenceSplitSequence = 0;
  let realtimeSentenceSplitState = null;
  function probabilitySegments(probabilities) {
    const entries = Object.entries(probabilities || {})
      .map(([key, value]) => ({key, value: Math.max(0, Number(value) || 0)}))
      .filter(item => item.value > 0);
    entries.sort((left, right) => {
      if (left.key === "unknown") return -1;
      if (right.key === "unknown") return 1;
      const leftIndex = /^speaker(\d+)$/.exec(left.key);
      const rightIndex = /^speaker(\d+)$/.exec(right.key);
      if (leftIndex && rightIndex) return Number(leftIndex[1]) - Number(rightIndex[1]);
      if (leftIndex) return -1;
      if (rightIndex) return 1;
      return left.key.localeCompare(right.key);
    });
    const total = entries.reduce((sum, item) => sum + item.value, 0);
    return entries.map(item => ({
      ...item,
      width: total > 0 ? (item.value / total) * 100 : 0,
    }));
  }
  function probabilityTooltip(probabilities) {
    return probabilitySegments(probabilities)
      .map(item => `${probabilityDisplayLabel(item.key)} ${Math.round(item.value * 100)}%`)
      .join(" | ");
  }
  function displayProbabilities(item) {
    const assignedKey = speakerProbabilityKey(item.assigned_speaker);
    if (assignedKey && item.created_speaker && !item.pending && !item.error) {
      return {[assignedKey]: 1.0};
    }
    return item.probabilities;
  }
  function secondsLabel(value) {
    return `${Number(value || 0).toFixed(1)}s`;
  }
  function ratioLabel(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2) : "n/a";
  }
  function transcriptTimeLabel(value) {
    const total = Math.max(0, Number(value || 0));
    const minutes = Math.floor(total / 60);
    const seconds = total - (minutes * 60);
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`;
  }
  function optionalNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }
  function transcriptTranslationExportStates(row) {
    const result = {};
    const key = String(row && row.dataset.index || "");
    const states = translationStateMap(key);
    if (!states) return result;
    states.forEach((state, code) => {
      if (!translationStateMatchesRow(state, row)) return;
      result[code] = {
        language: state.target_language_name || translationLanguageName(code),
        status: state.status,
        text: state.text || "",
        error: state.error || "",
        provider: state.provider || "",
        source_revision: state.source_revision === undefined ? null : state.source_revision,
        source_text_hash: state.source_text_hash || "",
      };
    });
    return result;
  }
  function transcriptAtomicExportRows() {
    return Array.from(sentences.querySelectorAll(".row"))
      .filter(row => row.dataset.realtime !== "true")
      .map(row => ({
        index: row.dataset.index || "",
        speaker_id: row.dataset.speaker === "UNKNOWN" ? null : toInternalSpeakerId(row.dataset.speaker),
        speaker: speakerDisplayLabel(row.dataset.speaker === "UNKNOWN" ? null : row.dataset.speaker),
        start: transcriptTimeLabel(row.dataset.start),
        end: transcriptTimeLabel(row.dataset.end),
        start_seconds: optionalNumber(row.dataset.start),
        end_seconds: optionalNumber(row.dataset.end),
        text: row.dataset.text || "",
        source_revision: row.dataset.sourceRevision || "",
        source_text_hash: row.dataset.sourceTextHash || "",
        translations: transcriptTranslationExportStates(row),
        assignment_source: row.dataset.assignmentSource || "",
        top_similarity: optionalNumber(row.dataset.topSimilarity),
        margin: optionalNumber(row.dataset.margin),
        unknown_probability: optionalNumber(row.dataset.unknownProbability),
        pending: row.dataset.pending === "true",
        provisional_assignment: row.dataset.provisionalAssignment === "true",
        needs_review: row.dataset.needsReview === "true",
        review_reasons: (row.dataset.reviewReasons || "").split("|").filter(Boolean),
        correction_status: row.dataset.correctionStatus || "",
      }))
      .filter(row => row.text.trim());
  }
  function transcriptExportRows(speakerId = null) {
    const internalFilter = speakerId ? toInternalSpeakerId(speakerId) : null;
    return transcriptAtomicExportRows().filter(row => !internalFilter || row.speaker_id === internalFilter);
  }
  function transcriptExportRowCanGroup(row) {
    return Boolean(
      row
      && row.speaker_id
      && !row.pending
      && !row.provisional_assignment
      && row.text
      && row.text.trim()
    );
  }
  function mergeTranscriptExportRows(rows) {
    const startRow = rows[0];
    const endRow = rows[rows.length - 1];
    const text = rows.map(row => row.text.trim()).filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
    const reviewReasons = Array.from(new Set(rows.flatMap(row => row.review_reasons || [])));
    const assignmentSources = Array.from(new Set(rows.map(row => row.assignment_source || "").filter(Boolean)));
    const correctionStatuses = Array.from(new Set(rows.map(row => row.correction_status || "").filter(Boolean)));
    const translationCodes = Array.from(new Set(rows.flatMap(row => Object.keys(row.translations || {}))));
    const translations = {};
    translationCodes.forEach(code => {
      const states = rows.map(row => (row.translations || {})[code] || null);
      const completed = states.filter(state => state && state.status === "complete" && state.text).map(state => state.text.trim());
      const complete = states.length === rows.length && states.every(state => state && state.status === "complete" && state.text);
      const failed = states.filter(state => state && state.status === "error");
      translations[code] = {
        language: translationLanguageName(code),
        status: complete ? "complete" : (failed.length && completed.length + failed.length === rows.length ? "error" : "translating"),
        text: completed.join(" ").replace(/\s+/g, " ").trim(),
        error: failed.map(state => state.error || "").filter(Boolean).join(" | "),
      };
    });
    return {
      ...startRow,
      index: rows.map(row => row.index).filter(Boolean).join(","),
      sentence_indexes: rows.map(row => row.index).filter(Boolean),
      sentence_count: rows.length,
      start: startRow.start,
      end: endRow.end,
      start_seconds: startRow.start_seconds,
      end_seconds: endRow.end_seconds,
      text,
      translations,
      assignment_source: assignmentSources.length === 1 ? assignmentSources[0] : (assignmentSources.length ? "mixed" : ""),
      top_similarity: null,
      margin: null,
      unknown_probability: null,
      pending: false,
      provisional_assignment: false,
      needs_review: rows.some(row => row.needs_review),
      review_reasons: reviewReasons,
      correction_status: correctionStatuses.length === 1 ? correctionStatuses[0] : (correctionStatuses.length ? "mixed" : ""),
    };
  }
  function transcriptGroupedExportRows() {
    const result = [];
    let currentGroup = [];
    transcriptAtomicExportRows().forEach(row => {
      if (!transcriptExportRowCanGroup(row)) {
        if (currentGroup.length) {
          result.push(currentGroup.length > 1 ? mergeTranscriptExportRows(currentGroup) : currentGroup[0]);
        }
        currentGroup = [];
        result.push(row);
        return;
      }
      if (!currentGroup.length || currentGroup[0].speaker_id === row.speaker_id) {
        currentGroup.push(row);
      } else {
        result.push(currentGroup.length > 1 ? mergeTranscriptExportRows(currentGroup) : currentGroup[0]);
        currentGroup = [row];
      }
    });
    if (currentGroup.length) {
      result.push(currentGroup.length > 1 ? mergeTranscriptExportRows(currentGroup) : currentGroup[0]);
    }
    return result;
  }
  function transcriptTextExportRows(speakerId = null) {
    const rows = transcriptGroupTurnsEnabled() ? transcriptGroupedExportRows() : transcriptAtomicExportRows();
    return rows.filter(row => !speakerId || row.speaker_id === speakerId);
  }
  function transcriptExportText(speakerId = null) {
    const rows = transcriptTextExportRows(speakerId);
    if (effectiveTranslationDisplayMode() === "original") {
      return rows.map(row => `[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`).join("\n");
    }
    const targetCodes = selectedTranslationCodesForDisplay();
    const includeOriginal = Boolean(translationIncludeOriginalControl.checked);
    return rows.map(row => {
      const translated = targetCodes
        .map(code => ({code, state:(row.translations || {})[code]}))
        .filter(item => item.state && item.state.status === "complete" && String(item.state.text || "").trim());
      if (effectiveTranslationDisplayMode() === "single" && translated.length && !includeOriginal) {
        return `[${row.start} - ${row.end}] ${row.speaker}: ${translated[0].state.text}`;
      }
      if (!translated.length) {
        return `[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`;
      }
      const lines = [];
      if (includeOriginal) lines.push(`[Original · ${String(languageConfig.name || languageConfig.code || "Source")}] ${row.text}`);
      translated.forEach(item => lines.push(`[${translationLanguageName(item.code)}] ${item.state.text}`));
      return `[${row.start} - ${row.end}] ${row.speaker}:\n  ${lines.join("\n  ")}`;
    }).join("\n");
  }
  function transcriptExportFilename(speakerId = null) {
    const suffix = speakerId ? `-${speakerDisplayLabel(speakerId).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}` : "";
    return `whospeaks-transcript${suffix || ""}.txt`;
  }
  async function copyTextToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.focus();
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  function copyTranscript(speakerId = null) {
    const text = transcriptExportText(speakerId);
    if (!text) {
      log("No transcript text to copy.");
      return;
    }
    copyTextToClipboard(text)
      .then(() => log(speakerId ? `Copied ${speakerDisplayLabel(speakerId)} transcript.` : "Copied transcript."))
      .catch(error => log(`Copy failed: ${error.message}`));
  }
  function downloadTranscript(speakerId = null) {
    const text = transcriptExportText(speakerId);
    if (!text) {
      log("No transcript text to download.");
      return;
    }
    const url = URL.createObjectURL(new Blob([text], {type: "text/plain;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = transcriptExportFilename(speakerId);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }
  function downloadTranscriptJson(speakerId = null) {
    const rows = transcriptExportRows(speakerId);
    if (!rows.length) {
      log("No transcript metadata to download.");
      return;
    }
    downloadJsonFile(transcriptExportFilename(speakerId).replace(/\.txt$/, ".json"), {
      format: "whospeaks-transcript",
      version: 1,
      exported_at: new Date().toISOString(),
      translation_display: {
        mode: effectiveTranslationDisplayMode(),
        target_languages: selectedTranslationCodesForDisplay(),
        include_original: Boolean(translationIncludeOriginalControl.checked),
      },
      rows,
    });
  }
  function findSentenceRow(index) {
    const key = String(index);
    return Array.from(sentences.querySelectorAll(".row")).find(row => row.dataset.index === key) || null;
  }
  function findFinalSentenceRow(index) {
    const key = String(index);
    return Array.from(sentences.querySelectorAll(".row")).find(row => (
      row.dataset.index === key && row.dataset.realtime !== "true"
    )) || null;
  }
  function findRealtimeSentenceRow(index) {
    const key = String(index);
    return Array.from(sentences.querySelectorAll(".row")).find(row => (
      row.dataset.index === key && row.dataset.realtime === "true"
    )) || null;
  }
  function rowChronologyKey(row) {
    return {
      start: finiteAudioSecond(row && row.dataset.start, Number.POSITIVE_INFINITY),
      end: finiteAudioSecond(row && row.dataset.end, Number.POSITIVE_INFINITY),
      index: String((row && row.dataset.index) || ""),
    };
  }
  function rowShouldSortBefore(a, b) {
    const left = rowChronologyKey(a);
    const right = rowChronologyKey(b);
    if (left.start !== right.start) return left.start < right.start;
    if (left.end !== right.end) return left.end < right.end;
    return left.index < right.index;
  }
  function placeSentenceRowChronologically(row) {
    if (!row || !row.isConnected) return;
    const next = Array.from(sentences.querySelectorAll(".row")).find(candidate => (
      candidate !== row
      && candidate.isConnected
      && rowShouldSortBefore(row, candidate)
    ));
    if (next && next !== row.nextSibling) {
      sentences.insertBefore(row, next);
    } else if (!next && row.nextSibling) {
      sentences.appendChild(row);
    }
  }
  function realtimeSentenceSplitAnimationsEnabled() {
    return true;
  }
  function realtimeSentenceSplitDelay(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  }
  function nextRealtimeSentenceSplitTick() {
    return realtimeSentenceSplitDelay(16);
  }
  function normalizedRealtimeSentenceSplitText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }
  function realtimeSentenceSplitRemainder(previewValue, finalizedValue) {
    const preview = normalizedRealtimeSentenceSplitText(previewValue);
    const finalized = normalizedRealtimeSentenceSplitText(finalizedValue);
    if (!preview || !finalized || preview.length <= finalized.length) return "";
    if (preview.toLowerCase().startsWith(finalized.toLowerCase())) {
      return preview.slice(finalized.length).trim();
    }
    const finalWords = finalized.toLowerCase().split(" ").filter(Boolean);
    const anchor = finalWords.slice(-Math.min(4, finalWords.length)).join(" ");
    const anchorAt = anchor ? preview.toLowerCase().indexOf(anchor) : -1;
    if (anchorAt < 0) return "";
    return preview.slice(anchorAt + anchor.length).trim();
  }
  function combineRealtimeSentenceSplitText(previousValue, liveValue) {
    const previous = normalizedRealtimeSentenceSplitText(previousValue);
    const live = normalizedRealtimeSentenceSplitText(liveValue);
    if (!previous) return live;
    if (!live || previous.toLowerCase().endsWith(live.toLowerCase())) return previous;
    const left = previous.toLowerCase();
    const right = live.toLowerCase();
    const maxOverlap = Math.min(left.length, right.length);
    for (let overlap = maxOverlap; overlap >= 3; overlap -= 1) {
      if (left.endsWith(right.slice(0, overlap))) {
        return `${previous}${live.slice(overlap)}`;
      }
    }
    return `${previous} ${live}`.trim();
  }
  function clearRealtimeSentenceSplitRowState(row) {
    if (!row) return;
    row.classList.remove("realtime-split-target", "realtime-split-spawning");
    row.style.removeProperty("max-height");
    row.style.removeProperty("overflow");
    row.style.removeProperty("opacity");
    row.style.removeProperty("padding-top");
    row.style.removeProperty("padding-bottom");
    delete row.dataset.realtimeSplitActive;
    delete row.dataset.realtimeSplitToken;
  }
  function queuedRealtimeSentenceSplit(item) {
    const key = String(item && item.index);
    const existing = queuedRealtimeSentenceSplits.findIndex(candidate => String(candidate.index) === key);
    if (existing >= 0) {
      queuedRealtimeSentenceSplits[existing] = {...queuedRealtimeSentenceSplits[existing], ...item};
    } else {
      queuedRealtimeSentenceSplits.push({...item});
    }
  }
  function abortRealtimeSentenceSplit(split) {
    if (!split || realtimeSentenceSplitState !== split) return;
    clearTimeout(split.watchdog);
    clearRealtimeSentenceSplitRowState(split.target);
    realtimeSentenceSplitState = null;
    while (queuedRealtimeSentenceSplits.length) {
      renderSentenceImmediate(queuedRealtimeSentenceSplits.shift());
    }
    refreshTranscriptGrouping();
  }
  function updateRealtimeSentenceSplitVisual(split) {
    if (!split || realtimeSentenceSplitState !== split) return;
    const combinedText = combineRealtimeSentenceSplitText(split.transferText, split.target.dataset.text || "");
    split.combinedText = combinedText;
    const transferLower = split.transferText.toLowerCase();
    const combinedLower = combinedText.toLowerCase();
    const liveTail = combinedLower.startsWith(transferLower)
      ? combinedText.slice(split.transferText.length).trim()
      : "";
    const transferCharacters = Array.from(split.transferText);
    const movedCount = Math.min(transferCharacters.length, Math.max(0, split.movedCount || 0));
    const remainingText = transferCharacters.slice(movedCount).join("");
    const movedText = transferCharacters.slice(0, movedCount).join("");
    const sourceText = split.source.querySelector(".text");
    const targetText = split.target.querySelector(".text");
    if (sourceText) {
      sourceText.textContent = [split.finalizedText, remainingText].filter(Boolean).join(" ");
    }
    if (targetText) {
      targetText.textContent = [movedText, liveTail].filter(Boolean).join(" ");
    }
  }
  async function animateRealtimeSentenceSplit(split) {
    const {source:sourceRow, target:targetRow, token} = split;
    const current = () => (
      realtimeSentenceSplitState === split
      && targetRow.dataset.realtimeSplitToken === token
      && sourceRow.isConnected
      && targetRow.isConnected
    );
    const expandedHeight = Math.max(targetRow.scrollHeight, sourceRow.scrollHeight, 1);
    targetRow.classList.add("realtime-split-spawning");
    targetRow.style.maxHeight = "0px";
    targetRow.style.overflow = "hidden";
    targetRow.style.opacity = "0";
    targetRow.style.paddingTop = "0";
    targetRow.style.paddingBottom = "0";
    updateRealtimeSentenceSplitVisual(split);
    await nextRealtimeSentenceSplitTick();
    if (!current()) return abortRealtimeSentenceSplit(split);
    targetRow.style.maxHeight = `${expandedHeight}px`;
    targetRow.style.opacity = "1";
    targetRow.style.paddingTop = "7px";
    targetRow.style.paddingBottom = "7px";
    await realtimeSentenceSplitDelay(realtimeSentenceSplitSpawnMs);
    if (!current()) return abortRealtimeSentenceSplit(split);
    split.phase = "transfer";
    const transferCharacters = Array.from(split.transferText);
    const transferStart = performance.now();
    await new Promise(resolve => {
      const transfer = () => {
        if (!current()) return resolve();
        const progress = Math.min(1, (performance.now() - transferStart) / realtimeSentenceSplitTransferMs);
        split.movedCount = Math.min(transferCharacters.length, Math.floor(progress * transferCharacters.length));
        updateRealtimeSentenceSplitVisual(split);
        progress < 1 ? setTimeout(transfer, 16) : resolve();
      };
      setTimeout(transfer, 16);
    });
    if (!current()) return abortRealtimeSentenceSplit(split);
    split.movedCount = transferCharacters.length;
    updateRealtimeSentenceSplitVisual(split);
    clearTimeout(split.watchdog);
    clearRealtimeSentenceSplitRowState(targetRow);
    realtimeSentenceSplitState = null;
    while (queuedRealtimeSentenceSplits.length) {
      const nextItem = queuedRealtimeSentenceSplits.shift();
      if (startRealtimeSentenceSplit(nextItem, targetRow, split.combinedText)) return;
      renderSentenceImmediate(nextItem);
    }
    refreshTranscriptGrouping();
    if (targetRow.dataset.realtimeSettling === "true") {
      markRealtimeRowSettling(targetRow, targetRow.dataset.realtimeClearGeneration || "");
    }
  }
  function createRealtimeSentenceSplitTarget(sourceRow, remainderText) {
    clearSettlingRealtimeState(sourceRow);
    const targetRow = sourceRow.cloneNode(true);
    targetRow.classList.remove(
      "group-leader",
      "group-hidden",
      "group-needs-review",
      "group-corrected",
      "group-merge-highlight",
      "group-merge-collapsing",
      "row-removing",
      "realtime-settling",
      "provisional-split-source",
      "provisional-visual-split",
    );
    targetRow.classList.add("realtime-split-target");
    targetRow.hidden = false;
    targetRow.dataset.realtime = "true";
    targetRow.dataset.text = remainderText;
    targetRow.dataset.searchText = remainderText;
    targetRow.dataset.fullText = remainderText;
    targetRow.dataset.start = sourceRow.dataset.end || sourceRow.dataset.start || "0";
    delete targetRow.dataset.groupHidden;
    delete targetRow.dataset.groupLeader;
    delete targetRow.dataset.groupSize;
    delete targetRow.dataset.groupIndexes;
    delete targetRow.dataset.groupText;
    delete targetRow.dataset.realtimeSettling;
    delete targetRow.dataset.realtimeClearGeneration;
    delete targetRow.dataset.provisionalSplit;
    delete targetRow.dataset.provisionalSplitFor;
    const targetText = targetRow.querySelector(".text");
    if (targetText) targetText.textContent = "";
    sentences.insertBefore(targetRow, sourceRow.nextSibling);
    return targetRow;
  }
  function startRealtimeSentenceSplit(item, sourceRow, previewOverride = "") {
    if (!realtimeSentenceSplitAnimationsEnabled() || !sourceRow || !sourceRow.isConnected) return false;
    const previewText = normalizedRealtimeSentenceSplitText(
      previewOverride || sourceRow.dataset.fullText || sourceRow.dataset.text || ""
    );
    const finalizedText = normalizedRealtimeSentenceSplitText(item && item.text);
    const remainderText = realtimeSentenceSplitRemainder(previewText, finalizedText);
    if (!remainderText) return false;
    const targetRow = createRealtimeSentenceSplitTarget(sourceRow, remainderText);
    const token = String(++realtimeSentenceSplitSequence);
    delete sourceRow.dataset.realtimeSplitActive;
    delete sourceRow.dataset.realtimeSplitToken;
    targetRow.dataset.realtimeSplitActive = "true";
    targetRow.dataset.realtimeSplitToken = token;
    const split = {
      source:sourceRow,
      target:targetRow,
      token,
      finalizedText,
      transferText:remainderText,
      combinedText:remainderText,
      movedCount:0,
      phase:"spawn",
    };
    realtimeSentenceSplitState = split;
    split.watchdog = setTimeout(() => abortRealtimeSentenceSplit(split), 2500);
    renderSentenceImmediate(item, sourceRow);
    updateRealtimeSentenceSplitVisual(split);
    void animateRealtimeSentenceSplit(split);
    return true;
  }
  function fadeRemoveRow(row) {
    if (!row || !row.isConnected) return;
    row.classList.add("row-removing");
    setTimeout(() => {
      if (row.isConnected && row.classList.contains("row-removing")) {
        row.remove();
      }
    }, 220);
  }
  function markRealtimeRowSettling(row, generation) {
    if (!row || !row.isConnected || row.dataset.realtime !== "true") return;
    const generationKey = String(generation || "");
    row.dataset.realtimeSettling = "true";
    row.dataset.realtimeClearGeneration = generationKey;
    row.classList.add("realtime-settling");
    setTimeout(() => {
      if (
        row.isConnected
        && row.dataset.realtimeSettling === "true"
        && row.dataset.realtimeClearGeneration === generationKey
        && row.dataset.realtimeSplitActive !== "true"
      ) {
        fadeRemoveRow(row);
      }
    }, realtimeSettleRemovalDelayMs);
  }
  function clearSettlingRealtimeState(row) {
    if (!row) return;
    delete row.dataset.realtimeSettling;
    delete row.dataset.realtimeClearGeneration;
    row.classList.remove("realtime-settling", "row-removing");
  }
  function textAdoptionScore(a, b) {
    const tokensA = String(a || "").toLowerCase().match(/[a-z0-9]+/g) || [];
    const tokensB = String(b || "").toLowerCase().match(/[a-z0-9]+/g) || [];
    if (!tokensA.length || !tokensB.length) return 0;
    const remaining = new Map();
    tokensA.forEach(token => remaining.set(token, (remaining.get(token) || 0) + 1));
    let shared = 0;
    tokensB.forEach(token => {
      const count = remaining.get(token) || 0;
      if (count <= 0) return;
      shared += 1;
      remaining.set(token, count - 1);
    });
    return shared / Math.max(1, Math.min(tokensA.length, tokensB.length));
  }
  function rowTimeAdoptionScore(row, start, end) {
    const rowStart = finiteAudioSecond(row.dataset.start, NaN);
    const rowEnd = finiteAudioSecond(row.dataset.end, NaN);
    if (!Number.isFinite(rowStart) || !Number.isFinite(rowEnd) || !(rowEnd > rowStart) || !(end > start)) return 0;
    const overlap = Math.max(0, Math.min(rowEnd, end) - Math.max(rowStart, start));
    return overlap / Math.max(0.1, Math.min(rowEnd - rowStart, end - start));
  }
  function findAdoptableRealtimeRow(item, options = {}) {
    const start = finiteAudioSecond(item && item.start, NaN);
    const end = finiteAudioSecond(item && item.end, NaN);
    if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return null;
    const settlingOnly = options.settlingOnly === true;
    let best = null;
    Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => {
      if (!row.isConnected || row.classList.contains("row-removing")) return;
      if (settlingOnly && row.dataset.realtimeSettling !== "true") return;
      const timeScore = rowTimeAdoptionScore(row, start, end);
      if (timeScore <= 0) return;
      const textScore = textAdoptionScore(row.dataset.text || row.dataset.fullText || "", item.text || "");
      if (timeScore < 0.34 && textScore < 0.5) return;
      const settlingBonus = row.dataset.realtimeSettling === "true" ? 0.08 : 0;
      const score = timeScore * 0.72 + textScore * 0.28 + settlingBonus;
      if (!best || score > best.score) {
        best = {row, score};
      }
    });
    return best ? best.row : null;
  }
  function removeOverlappingSettlingRealtimeRows(item, keepRow = null) {
    const start = finiteAudioSecond(item && item.start, NaN);
    const end = finiteAudioSecond(item && item.end, NaN);
    if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return;
    Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => {
      if (row === keepRow || !row.isConnected || row.dataset.realtimeSettling !== "true") return;
      const timeScore = rowTimeAdoptionScore(row, start, end);
      const textScore = textAdoptionScore(row.dataset.text || row.dataset.fullText || "", item.text || "");
      if (timeScore >= 0.34 && textScore >= 0.5) {
        fadeRemoveRow(row);
      }
    });
  }
  function clearProvisionalRealtimeSplitsFor(index) {
    const key = String(index || "");
    Array.from(sentences.querySelectorAll(".row[data-provisional-split='true']")).forEach(row => {
      if (!key || row.dataset.provisionalSplitFor === key) {
        row.remove();
      }
    });
  }
  function renderProvisionalRealtimeSplitRow(baseRow, split) {
    const baseIndex = String(baseRow.dataset.index || "");
    const splitIndex = `${baseIndex}:split`;
    let row = findSentenceRow(splitIndex);
    if (!row) {
      row = document.createElement("div");
    }
    const speakerId = normalizedLiveSpeakerId(split.speakerId);
    const color = speakerColor(speakerId);
    row.className = "row realtime provisional-visual-split live-speaker-row";
    row.dataset.index = splitIndex;
    row.dataset.realtime = "true";
    row.dataset.provisionalSplit = "true";
    row.dataset.provisionalSplitFor = baseIndex;
    row.dataset.rawSpeaker = speakerId;
    row.dataset.rawSpeakerProbability = "1";
    row.dataset.rawSpeakerUnknownMargin = "1";
    row.dataset.speaker = speakerId || "UNKNOWN";
    row.dataset.start = String(split.splitStart);
    row.dataset.end = String(split.end);
    row.dataset.text = split.suffixText;
    row.dataset.searchText = split.suffixText;
    row.style.setProperty("--live-row-color", color || "#8F9BA8");

    const top = document.createElement("div");
    top.className = "top";
    const topLeft = document.createElement("div");
    topLeft.className = "top-left";

    const speakerBadge = document.createElement("span");
    speakerBadge.className = `${speakerId ? "badge" : "badge unknown"} speaker-name`;
    if (color) {
      speakerBadge.style.color = color;
      speakerBadge.style.borderColor = color;
      speakerBadge.style.background = "#0B1015";
    }
    speakerBadge.textContent = speakerDisplayLabel(speakerId);
    topLeft.appendChild(speakerBadge);

    const stateBadge = document.createElement("span");
    stateBadge.className = "badge state";
    stateBadge.textContent = "Live";
    topLeft.appendChild(stateBadge);

    const duration = document.createElement("span");
    duration.className = "sentence-duration";
    duration.textContent = secondsLabel(Math.max(0, split.end - split.splitStart));
    topLeft.appendChild(duration);

    const range = document.createElement("span");
    range.className = "sentence-range";
    range.textContent = `(${secondsLabel(split.splitStart)} - ${secondsLabel(split.end)})`;
    topLeft.appendChild(range);

    const prob = document.createElement("div");
    prob.className = "prob";
    if (color) {
      const span = document.createElement("span");
      span.style.flex = "0 0 100%";
      span.style.background = color;
      prob.appendChild(span);
    }
    top.appendChild(topLeft);
    top.appendChild(prob);

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = split.suffixText;
    row.replaceChildren(top, text);
    if (baseRow.nextSibling !== row) {
      sentences.insertBefore(row, baseRow.nextSibling);
    }
  }
  function updateRenderedRealtimeRowTextRange(row, textValue, endValue) {
    const text = String(textValue || "");
    const end = finiteAudioSecond(endValue, finiteAudioSecond(row.dataset.end, 0));
    row.dataset.end = String(end);
    row.dataset.text = text;
    row.dataset.searchText = text;
    const textNode = row.querySelector(".text");
    if (textNode) textNode.textContent = text;
    const start = finiteAudioSecond(row.dataset.start, 0);
    const duration = row.querySelector(".sentence-duration");
    if (duration) duration.textContent = secondsLabel(Math.max(0, end - start));
    const range = row.querySelector(".sentence-range");
    if (range) range.textContent = `(${secondsLabel(start)} - ${secondsLabel(end)})`;
  }
  function restoreRealtimeRowFullPreview(row) {
    const fullText = row.dataset.fullText;
    const fullEnd = row.dataset.fullEnd;
    if (fullText === undefined || fullEnd === undefined) return;
    row.classList.remove("provisional-split-source");
    updateRenderedRealtimeRowTextRange(row, fullText, fullEnd);
    row.dataset.rawSpeaker = row.dataset.fullRawSpeaker || row.dataset.rawSpeaker || "";
  }
  function applyProvisionalRealtimeVisualSplit(row, split) {
    row.classList.add("provisional-split-source");
    updateRenderedRealtimeRowTextRange(row, split.prefixText, split.splitStart);
    row.dataset.rawSpeaker = normalizedLiveSpeakerId(row.dataset.speaker);
    renderProvisionalRealtimeSplitRow(row, split);
  }
  function clearRealtimeRows(generation) {
    ctx.owners.capture.currentRealtimeGeneration = Math.max(ctx.owners.capture.currentRealtimeGeneration, Number(generation || 0));
    Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => markRealtimeRowSettling(row, generation));
    updateCurrentLiveSpeakerFromRealtimeRows();
    refreshSpeakerPanelSentenceCounts();
    refreshTranscriptVisibility();
  }
  function renderSentenceImmediate(item, rowOverride = null) {
    if (item.realtime && Number(item.realtime_generation || 0) < ctx.owners.capture.currentRealtimeGeneration) {
      return;
    }
    if (itemIsBeforeClearedTranscriptBoundary(item)) {
      return;
    }
    if (item.realtime && findFinalSentenceRow(item.index)) {
      return;
    }
    if (item.realtime) {
      clearProvisionalRealtimeSplitsFor(item.index);
    }
    if (item.assigned_speaker && item.speaker_name) {
      ctx.owners.speakers.speakerNames[item.assigned_speaker] = item.speaker_name;
    }
    if (item.realtime) {
      ctx.owners.capture.currentRealtimeGeneration = Math.max(ctx.owners.capture.currentRealtimeGeneration, Number(item.realtime_generation || 0));
    }
    let row = rowOverride || (
      item.realtime
        ? findRealtimeSentenceRow(item.index)
        : (findFinalSentenceRow(item.index) || findRealtimeSentenceRow(item.index))
    );
    if (!row && item.realtime) {
      row = findAdoptableRealtimeRow(item, {settlingOnly: true});
    }
    if (!row && !item.realtime) {
      row = findAdoptableRealtimeRow(item);
    }
    const adoptedLiveSpeakerId = !item.realtime && row && row.dataset.realtime === "true"
      ? normalizedLiveSpeakerId(row.dataset.speaker)
      : "";
    const previousCanonicalText = row && row.dataset.realtime !== "true" ? String(row.dataset.text || "") : "";
    const previousSourceTextHash = row && row.dataset.realtime !== "true" ? String(row.dataset.sourceTextHash || "") : "";
    const isNewRow = !row;
    if (!row) {
      row = document.createElement("div");
      sentences.appendChild(row);
    }
    row.className = item.realtime ? "row realtime" : "row";
    if (row.dataset.realtimeSplitActive === "true") {
      row.classList.add("realtime-split-target");
      if (realtimeSentenceSplitState && realtimeSentenceSplitState.phase === "spawn") {
        row.classList.add("realtime-split-spawning");
      }
    }
    row.dataset.index = item.index;
    row.dataset.realtime = item.realtime ? "true" : "false";
    row.dataset.selectable = (!item.realtime && !item.pending) ? "true" : "false";
    if (row.dataset.selectable !== "true") {
      ctx.owners.transcript.selectedTranscriptRowIndexes.delete(String(row.dataset.index || ""));
    }
    if (item.realtime) {
      clearSettlingRealtimeState(row);
    }
    delete row.dataset.provisionalSplit;
    delete row.dataset.provisionalSplitFor;
    if (!item.realtime) {
      clearSettlingRealtimeState(row);
      delete row.dataset.fullRawSpeaker;
      delete row.dataset.fullEnd;
      delete row.dataset.fullText;
    }
    const startSeconds = Number(item.start || 0);
    const endSeconds = Number(item.end || 0);
    const ratio = Number(item.speech_audio_ratio);
    const rawSpeakerId = normalizedLiveSpeakerId(item.assigned_speaker);
    const rawProbabilities = item.probabilities || {};
    const rawSpeakerProbability = probabilityForSpeakerId(rawProbabilities, rawSpeakerId);
    const rawSpeakerUnknownMargin = probabilityLeadOverUnknown(rawProbabilities, rawSpeakerId);
    const previousRevisionSpeakerId = revisionSpeakerId(row.dataset.speaker);
    const previousDisplaySpeakerId = item.realtime ? normalizedLiveSpeakerId(row.dataset.speaker) : "";
    const displaySpeakerId = item.realtime
      ? realtimeRowDisplaySpeakerId(
          rawSpeakerId,
          startSeconds,
          endSeconds,
          previousDisplaySpeakerId,
          ctx.owners.transcript.lastTranscriptSpeakerId,
        )
      // A finalized row is first rendered as pending while its embedding is
      // in flight.  When it adopts the just-visible realtime row, retain that
      // visual identity until the actual embedding decision arrives.
      : (rawSpeakerId || (item.pending ? adoptedLiveSpeakerId : ""));
    const reviewReasons = reviewReasonsForItem(item, displaySpeakerId, adoptedLiveSpeakerId);
    const corrected = rowIsCorrected(item);
    const visualSplit = provisionalRealtimeVisualSplit(item, displaySpeakerId, startSeconds, endSeconds);
    const displayEndSeconds = visualSplit ? visualSplit.splitStart : endSeconds;
    const displayText = visualSplit ? visualSplit.prefixText : (item.text || "");
    const nextSourceTextHash = String(item.source_text_hash || "");
    if (!item.realtime && (
      (previousCanonicalText && previousCanonicalText !== String(item.text || ""))
      || (previousSourceTextHash && nextSourceTextHash && previousSourceTextHash !== nextSourceTextHash)
    )) {
      forgetSentenceTranslations(item.index);
    }
    const durationSeconds = Math.max(0, displayEndSeconds - startSeconds);
    row.dataset.rawSpeaker = item.realtime ? (visualSplit ? displaySpeakerId : rawSpeakerId) : "";
    row.dataset.rawSpeakerProbability = item.realtime ? String(rawSpeakerProbability) : "";
    row.dataset.rawSpeakerUnknownMargin = item.realtime ? String(rawSpeakerUnknownMargin) : "";
    row.dataset.fullRawSpeaker = item.realtime ? rawSpeakerId : "";
    row.dataset.fullEnd = item.realtime ? String(endSeconds) : "";
    row.dataset.fullText = item.realtime ? (item.text || "") : "";
    row.dataset.speaker = displaySpeakerId || "UNKNOWN";
    row.dataset.pending = item.pending ? "true" : "false";
    row.dataset.provisionalAssignment = item.provisional_assignment ? "true" : "false";
    row.dataset.finalSpeakerAssignment = (
      !item.realtime
      && !item.pending
      && !item.provisional_assignment
      && !item.error
      && Boolean(rawSpeakerId)
    ) ? "true" : "false";
    row.dataset.needsReview = (!item.realtime && !item.pending && reviewReasons.length > 0) ? "true" : "false";
    row.dataset.reviewReasons = reviewReasons.join("|");
    row.dataset.corrected = corrected ? "true" : "false";
    row.dataset.assignmentSource = item.assignment_source || "";
    row.dataset.margin = item.margin === undefined || item.margin === null ? "" : String(item.margin);
    row.dataset.topSimilarity = item.top_similarity === undefined || item.top_similarity === null ? "" : String(item.top_similarity);
    row.dataset.unknownProbability = item.unknown_probability === undefined || item.unknown_probability === null ? "" : String(item.unknown_probability);
    row.dataset.correctionStatus = correctionStatus(item);
    row.dataset.sourceTextHash = item.realtime ? "" : nextSourceTextHash;
    row.dataset.sourceRevision = item.realtime || item.source_revision === undefined || item.source_revision === null
      ? ""
      : String(item.source_revision);
    row.classList.toggle("provisional-assignment", Boolean(item.provisional_assignment));
    row.classList.toggle("provisional-split-source", Boolean(visualSplit));
    row.classList.toggle("live-speaker-row", item.realtime && Boolean(displaySpeakerId));
    row.classList.toggle("needs-review", row.dataset.needsReview === "true");
    row.classList.toggle("user-corrected", corrected);
    configureSentenceRowSelection(row);
    const speakerLabel = speakerDisplayLabel(displaySpeakerId);
    const color = speakerColor(displaySpeakerId);
    const speakerClass = displaySpeakerId ? "badge" : "badge unknown";
    const stateLabel = item.realtime ? "Live" : (
      item.pending ? "Embedding" : (
        item.error ? "Error" : sentenceRevisionLabel(row, item, displaySpeakerId, previousRevisionSpeakerId)
      )
    );
    row.dataset.start = String(startSeconds);
    row.dataset.end = String(displayEndSeconds);
    row.dataset.text = displayText;
    row.dataset.searchText = displayText;
    if (!item.realtime && !item.pending && rawSpeakerId) {
      ctx.owners.transcript.lastTranscriptSpeakerId = rawSpeakerId;
    }
    if (item.realtime) {
      row.style.setProperty("--live-row-color", color || "#8F9BA8");
    } else {
      row.style.removeProperty("--live-row-color");
    }

    const top = document.createElement("div");
    top.className = "top";
    const topLeft = document.createElement("div");
    topLeft.className = "top-left";

    const speakerBadge = document.createElement("span");
    speakerBadge.className = `${speakerClass} speaker-name`;
    if (color) {
      speakerBadge.style.color = color;
      speakerBadge.style.borderColor = color;
      speakerBadge.style.background = "#0B1015";
    }
    speakerBadge.textContent = speakerLabel;
    topLeft.appendChild(speakerBadge);

    if (item.created_speaker) {
      const badge = document.createElement("span");
      badge.className = "badge new";
      badge.textContent = "New";
      topLeft.appendChild(badge);
    }
    if (stateLabel) {
      const badge = document.createElement("span");
      badge.className = "badge state";
      badge.textContent = stateLabel;
      topLeft.appendChild(badge);
    }

    const duration = document.createElement("span");
    duration.className = "sentence-duration";
    duration.textContent = secondsLabel(durationSeconds);
    topLeft.appendChild(duration);

    const range = document.createElement("span");
    range.className = "sentence-range";
    range.textContent = `(${secondsLabel(startSeconds)} - ${secondsLabel(displayEndSeconds)})`;
    topLeft.appendChild(range);

    if (Number.isFinite(ratio)) {
      const ratioSpan = document.createElement("span");
      ratioSpan.className = "sentence-speech-rate";
      ratioSpan.textContent = `speech/audio ${ratioLabel(item.speech_audio_ratio)}`;
      topLeft.appendChild(ratioSpan);
    }
    if ((!item.realtime && !item.pending) && (reviewReasons.length || corrected)) {
      topLeft.appendChild(createReviewReasonGroup(reviewReasons, item));
    }

    const prob = document.createElement("div");
    prob.className = "prob";
    top.appendChild(topLeft);
    top.appendChild(prob);

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = displayText;
    row.replaceChildren(top, text);
    const translationLines = document.createElement("div");
    translationLines.className = "translation-lines";
    translationLines.hidden = true;
    row.appendChild(translationLines);
    placeSentenceRowChronologically(row);
    if (visualSplit) {
      renderProvisionalRealtimeSplitRow(row, visualSplit);
    }

    const probabilities = displayProbabilities(item);
    const segments = probabilitySegments(probabilities);
    const rawTooltip = probabilityTooltip(item.probabilities);
    const displayTooltip = probabilityTooltip(probabilities);
    prob.title = rawTooltip && rawTooltip !== displayTooltip ? `${displayTooltip} (raw: ${rawTooltip})` : displayTooltip;
    segments.forEach(segment => {
      const span = document.createElement("span");
      span.style.flex = `0 0 ${segment.width}%`;
      span.style.background = probabilityColor(segment.key);
      prob.appendChild(span);
    });
    if (item.error) {
      prob.title = item.error;
    }
    if (!item.realtime && item.translations) {
      applyTranslationCollection(item.translations, {sentence_index:item.index, refresh:false});
    }
    if (!item.realtime && !item.provisional_assignment) {
      const source = {
        segment_id:String(item.index),
        text:String(item.text || ""),
        source_text_hash:String(item.source_text_hash || ""),
        source_revision:String(
          item.source_revision === undefined || item.source_revision === null
            ? (item.source_text_hash || "")
            : item.source_revision
        ),
      };
      ctx.owners.translation.browserTranslationSourcesBySentence.set(source.segment_id, source);
      queueBrowserPreferredTranslationsForSource(source);
    }
    removeOverlappingSettlingRealtimeRows(item, row);
    updateCurrentLiveSpeakerFromRealtimeRows();
    refreshSpeakerPanelSentenceCounts();
    refreshTranscriptGrouping();
    if (!item.realtime) {
      syncBulkCorrectionToolbar();
    }
    if (!item.realtime && !item.pending && !item.provisional_assignment) {
      scheduleSavedSessionsRefresh();
    }
    if (isNewRow || item.realtime) {
      scrollSentencesToBottom();
    }
  }

  function renderSentence(item) {
    if (
      realtimeSentenceSplitState
      && (
        !realtimeSentenceSplitState.source.isConnected
        || !realtimeSentenceSplitState.target.isConnected
        || realtimeSentenceSplitState.target.dataset.realtimeSplitToken !== realtimeSentenceSplitState.token
      )
    ) {
      abortRealtimeSentenceSplit(realtimeSentenceSplitState);
    }
    if (item.realtime && realtimeSentenceSplitState && realtimeSentenceSplitState.target.isConnected) {
      renderSentenceImmediate(item, realtimeSentenceSplitState.target);
      updateRealtimeSentenceSplitVisual(realtimeSentenceSplitState);
      return;
    }
    if (!item.realtime && !findFinalSentenceRow(item.index)) {
      if (realtimeSentenceSplitState) {
        queuedRealtimeSentenceSplit(item);
        return;
      }
      const sourceRow = findAdoptableRealtimeRow(item);
      if (startRealtimeSentenceSplit(item, sourceRow)) return;
    }
    renderSentenceImmediate(item);
  }

  Object.assign(ctx.api, {applyProvisionalRealtimeVisualSplit, clearProvisionalRealtimeSplitsFor, clearRealtimeRows, clearSettlingRealtimeState, copyTextToClipboard, copyTranscript, displayProbabilities, downloadTranscript, downloadTranscriptJson, fadeRemoveRow, findAdoptableRealtimeRow, findFinalSentenceRow, findRealtimeSentenceRow, findSentenceRow, markRealtimeRowSettling, mergeTranscriptExportRows, optionalNumber, placeSentenceRowChronologically, probabilitySegments, probabilityTooltip, ratioLabel, removeOverlappingSettlingRealtimeRows, renderProvisionalRealtimeSplitRow, renderSentence, restoreRealtimeRowFullPreview, rowChronologyKey, rowShouldSortBefore, rowTimeAdoptionScore, secondsLabel, textAdoptionScore, transcriptAtomicExportRows, transcriptExportFilename, transcriptExportRowCanGroup, transcriptExportRows, transcriptExportText, transcriptGroupedExportRows, transcriptTextExportRows, transcriptTimeLabel, transcriptTranslationExportStates, updateRenderedRealtimeRowTextRange});
}
