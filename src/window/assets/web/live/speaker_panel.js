export function installSpeakerPanel(ctx) {
  const {addReferenceSpeakerButton, audio, bulkCorrectionSpeaker, clearSpeakersButton, createSpeakerOptionValue, manualSpeakerComposer, manualSpeakerName, manualSpeakerReferenceDock, recordReferenceButton, recordReferenceButtonLabel, referenceRecordSeconds, referenceSpeakerFile, referenceSpeakerForm, sentences, source, speakerEditorDock, speakerList, speakerPanelTitle, stop, svgNamespace, targetCaptureSampleRate, video} = ctx;
  const clearTranscriptSelection = (...args) => ctx.api.clearTranscriptSelection(...args), commonSelectedSpeakerId = (...args) => ctx.api.commonSelectedSpeakerId(...args), connect = (...args) => ctx.api.connect(...args), copyTranscript = (...args) => ctx.api.copyTranscript(...args), correctionStatus = (...args) => ctx.api.correctionStatus(...args), downloadTranscript = (...args) => ctx.api.downloadTranscript(...args), ensureSessionOwner = (...args) => ctx.api.ensureSessionOwner(...args), fetchSavedSessions = (...args) => ctx.api.fetchSavedSessions(...args), loadSavedSessionReview = (...args) => ctx.api.loadSavedSessionReview(...args), log = (...args) => ctx.api.log(...args), post = (...args) => ctx.api.post(...args), renderSentence = (...args) => ctx.api.renderSentence(...args), resampleFloat32 = (...args) => ctx.api.resampleFloat32(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), scheduleSavedSessionsRefresh = (...args) => ctx.api.scheduleSavedSessionsRefresh(...args), selectedSpeaker = (...args) => ctx.api.selectedSpeaker(...args), selectedTranscriptIndexes = (...args) => ctx.api.selectedTranscriptIndexes(...args), selectedTranscriptRows = (...args) => ctx.api.selectedTranscriptRows(...args), sessionControlsLocked = (...args) => ctx.api.sessionControlsLocked(...args), setSpeakerFilter = (...args) => ctx.api.setSpeakerFilter(...args), speakerColor = (...args) => ctx.api.speakerColor(...args), speakerCurrentSessionSentenceCount = (...args) => ctx.api.speakerCurrentSessionSentenceCount(...args), speakerDisplayLabel = (...args) => ctx.api.speakerDisplayLabel(...args), speakerPanelCountUnit = (...args) => ctx.api.speakerPanelCountUnit(...args), speakerPanelName = (...args) => ctx.api.speakerPanelName(...args), speakerPanelSentenceCount = (...args) => ctx.api.speakerPanelSentenceCount(...args), speakerPanelSpeakingSeconds = (...args) => ctx.api.speakerPanelSpeakingSeconds(...args), syncCorrectionUndoState = (...args) => ctx.api.syncCorrectionUndoState(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function speakerSpeakingTimeText(seconds) {
    const totalSeconds = Math.max(0, Number(seconds || 0));
    if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
    const roundedSeconds = Math.round(totalSeconds);
    const minutes = Math.floor(roundedSeconds / 60);
    const remainingSeconds = roundedSeconds % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
  }
  function speakerSentenceText(count, speakingSeconds = 0, unit = "sentence") {
    const total = Number(count || 0);
    return `${total} ${unit}${total === 1 ? "" : "s"} · ${speakerSpeakingTimeText(speakingSeconds)}`;
  }
  function speakerReferenceText(speaker) {
    const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
    if (!hasReference) return "";
    const seconds = Number(speaker.speech_seconds || 0);
    return seconds > 0 ? `Reference voice added (${Math.round(seconds)}s)` : "Reference voice added";
  }
  function appendSvgElement(svg, tagName, attributes) {
    const element = document.createElementNS(svgNamespace, tagName);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
    svg.appendChild(element);
  }
  function createSpeakerFilterIcon(mode) {
    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    if (mode === "mute") {
      appendSvgElement(svg, "path", {d: "M11 5 6 9H3v6h3l5 4V5z"});
      appendSvgElement(svg, "path", {d: "m16 9 5 5"});
      appendSvgElement(svg, "path", {d: "m21 9-5 5"});
    } else {
      appendSvgElement(svg, "path", {d: "M4 14v-2a8 8 0 0 1 16 0v2"});
      appendSvgElement(svg, "path", {d: "M4 14a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2v-2z"});
      appendSvgElement(svg, "path", {d: "M20 14a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2v-2z"});
    }
    return svg;
  }
  function createSpeakerFilterToggle(speaker, mode) {
    const active = mode === "mute" ? ctx.owners.speakers.mutedSpeakerIds.has(speaker.id) : ctx.owners.speakers.soloSpeakerIds.has(speaker.id);
    const label = mode === "mute" ? "Mute" : "Solo";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `speaker-filter-toggle ${mode}${active ? " active" : ""}`;
    button.setAttribute("aria-label", `${label} ${speakerPanelName(speaker)}`);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.title = `${label} ${speakerPanelName(speaker)}`;
    button.addEventListener("click", event => {
      event.stopPropagation();
      setSpeakerFilter(speaker.id || "", mode, !active);
    });
    button.addEventListener("keydown", event => {
      event.stopPropagation();
    });
    const switchTrack = document.createElement("span");
    switchTrack.className = "speaker-filter-switch";
    switchTrack.setAttribute("aria-hidden", "true");
    button.appendChild(createSpeakerFilterIcon(mode));
    button.appendChild(switchTrack);
    return button;
  }
  function createTranscriptActionIcon(kind) {
    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    if (kind === "download") {
      appendSvgElement(svg, "path", {d: "M12 3v12"});
      appendSvgElement(svg, "path", {d: "m7 10 5 5 5-5"});
      appendSvgElement(svg, "path", {d: "M5 21h14"});
    } else {
      appendSvgElement(svg, "rect", {x: "9", y: "9", width: "11", height: "11", rx: "2"});
      appendSvgElement(svg, "path", {d: "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"});
    }
    return svg;
  }
  function createTranscriptActionButton(kind, speaker) {
    const speakerName = speaker ? speakerPanelName(speaker) : "";
    const label = `${kind === "download" ? "Download" : "Copy"} ${speakerName ? `${speakerName} transcript` : "transcript"}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "transcript-icon-button speaker-transcript-action";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.appendChild(createTranscriptActionIcon(kind));
    button.addEventListener("click", event => {
      event.stopPropagation();
      if (kind === "download") {
        downloadTranscript(speaker ? speaker.id : null);
      } else {
        copyTranscript(speaker ? speaker.id : null);
      }
    });
    button.addEventListener("keydown", event => event.stopPropagation());
    return button;
  }
  function applyCorrectionResult(result) {
    if (!result || typeof result !== "object") return;
    if (result.speaker_state) updateSpeakerState(result.speaker_state);
    if (Array.isArray(result.rows)) {
      result.rows.forEach(row => renderSentence({...row, realtime:false, pending:false}));
    }
    ctx.owners.transcript.hasUndoableCorrection = true;
    syncCorrectionUndoState(true);
    scheduleSavedSessionsRefresh();
  }
  async function reassignSelectedSentences() {
    const indexes = selectedTranscriptIndexes();
    const speakerId = bulkCorrectionSpeaker.value || "";
    if (!indexes.length || !speakerId) return;
    if (speakerId === createSpeakerOptionValue) {
      await createSpeakerFromSelectedSentences();
      return;
    }
    try {
      const savedSessionId = ctx.owners.sessions.openedSavedSessionId || "";
      if (savedSessionId) {
        const result = await post("/api/sessions/corrections/reassign", {
          session_id: savedSessionId,
          indexes,
          speaker_id: speakerId,
        });
        if (result.session) loadSavedSessionReview(result.session, {quiet: true});
        clearTranscriptSelection();
        await fetchSavedSessions();
        log(`Reassigned ${indexes.length} saved sentence${indexes.length === 1 ? "" : "s"} to ${speakerDisplayLabel(speakerId)}.`);
        return;
      }
      await ensureSessionOwner("correct speaker labels");
      const result = await post("/api/corrections/reassign", {indexes, speaker_id: speakerId, update_memory: true});
      applyCorrectionResult(result);
      clearTranscriptSelection();
      log(`Reassigned ${indexes.length} sentence${indexes.length === 1 ? "" : "s"} to ${speakerDisplayLabel(speakerId)}.`);
    } catch (error) {
      log(`Reassign failed: ${error.message}`);
    }
  }
  async function markSelectedSentencesCorrect() {
    const indexes = selectedTranscriptIndexes();
    if (!indexes.length) return;
    try {
      const savedSessionId = ctx.owners.sessions.openedSavedSessionId || "";
      if (savedSessionId) {
        const result = await post("/api/sessions/corrections/mark-correct", {
          session_id: savedSessionId,
          indexes,
        });
        if (result.session) loadSavedSessionReview(result.session, {quiet: true});
        clearTranscriptSelection();
        await fetchSavedSessions();
        log(`Marked ${indexes.length} saved sentence${indexes.length === 1 ? "" : "s"} correct.`);
        return;
      }
      await ensureSessionOwner("mark speaker labels correct");
      const result = await post("/api/corrections/mark-correct", {indexes});
      applyCorrectionResult(result);
      clearTranscriptSelection();
      log(`Marked ${indexes.length} sentence${indexes.length === 1 ? "" : "s"} correct.`);
    } catch (error) {
      log(`Mark correct failed: ${error.message}`);
    }
  }
  async function createSpeakerFromSelectedSentences() {
    const rows = selectedTranscriptRows();
    const indexes = selectedTranscriptIndexes();
    const speakerId = commonSelectedSpeakerId(rows);
    if (!indexes.length || !speakerId) return;
    if (savedSessionReviewOpen()) {
      log("Creating a new speaker from saved transcript rows is not available yet; choose an existing speaker.");
      return;
    }
    try {
      await ensureSessionOwner("create speaker profiles");
      const result = await post("/api/speakers/split", {speaker_id: speakerId, sentence_indices: indexes, update_memory: true});
      applyCorrectionResult(result);
      clearTranscriptSelection();
      log(`Created ${speakerDisplayLabel(result.new_speaker_id)} from ${indexes.length} selected sentence${indexes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      log(`Create speaker failed: ${error.message}`);
    }
  }
  async function mergeSpeakerInto(sourceSpeakerId, targetSpeakerId) {
    if (!sourceSpeakerId || !targetSpeakerId || sourceSpeakerId === targetSpeakerId) return;
    try {
      await ensureSessionOwner("merge speaker profiles");
      const result = await post("/api/speakers/merge", {
        source_speaker_id: sourceSpeakerId,
        target_speaker_id: targetSpeakerId,
        update_memory: true,
      });
      applyCorrectionResult(result);
      ctx.owners.reference.editingSpeakerId = targetSpeakerId;
      log(`Merged ${speakerDisplayLabel(sourceSpeakerId)} into ${speakerDisplayLabel(targetSpeakerId)}.`);
    } catch (error) {
      log(`Merge failed: ${error.message}`);
    }
  }
  async function deleteSpeakerProfile(speaker) {
    if (!speaker || !speaker.id) return;
    const speakerId = speaker.id;
    const sentenceTotal = speakerCurrentSessionSentenceCount(speakerId);
    const label = speakerPanelName(speaker);
    const message = sentenceTotal > 0
      ? `Delete ${label} and move ${sentenceTotal} sentence${sentenceTotal === 1 ? "" : "s"} to UNKNOWN?`
      : `Delete empty speaker ${label}?`;
    if (!confirm(message)) return;
    try {
      await ensureSessionOwner("delete speaker profiles");
      const result = await post("/api/speakers/delete", {speaker_id: speakerId, update_memory: true});
      applyCorrectionResult(result);
      ctx.owners.reference.editingSpeakerId = "";
      const movedCount = Array.isArray(result.rows) ? result.rows.length : sentenceTotal;
      log(movedCount > 0
        ? `Deleted ${label} and moved ${movedCount} sentence${movedCount === 1 ? "" : "s"} to UNKNOWN.`
        : `Deleted empty speaker ${label}.`);
    } catch (error) {
      log(`Delete speaker failed: ${error.message}`);
    }
  }
  function createSpeakerMergeControls(speaker) {
    const controls = document.createElement("span");
    controls.className = "speaker-merge-controls speaker-profile-action";
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Merge ${speakerPanelName(speaker)} into`);
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Merge into...";
    select.appendChild(placeholder);
    ctx.owners.speakers.speakerLibraryState.speakers
      .filter(candidate => candidate.id && candidate.id !== speaker.id)
      .forEach(candidate => {
        const option = document.createElement("option");
        option.value = candidate.id;
        option.textContent = speakerPanelName(candidate);
        select.appendChild(option);
      });
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Merge";
    button.disabled = sessionControlsLocked() || savedSessionReviewOpen() || select.options.length <= 1;
    button.addEventListener("click", event => {
      event.stopPropagation();
      mergeSpeakerInto(speaker.id || "", select.value || "");
    });
    select.addEventListener("click", event => event.stopPropagation());
    select.addEventListener("keydown", event => event.stopPropagation());
    controls.appendChild(select);
    controls.appendChild(button);
    return controls;
  }
  function createSpeakerDeleteButton(speaker) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "speaker-delete-button speaker-profile-action";
    button.textContent = "Delete";
    button.disabled = sessionControlsLocked() || savedSessionReviewOpen();
    button.addEventListener("click", event => {
      event.stopPropagation();
      deleteSpeakerProfile(speaker);
    });
    button.addEventListener("keydown", event => event.stopPropagation());
    return button;
  }
  function createReviewReasonGroup(reasons, item) {
    const group = document.createElement("span");
    const status = correctionStatus(item);
    if (status === "user_corrected" || status === "user_confirmed") {
      group.className = "review-reasons correction-hints";
      const chip = document.createElement("span");
      chip.className = "review-chip corrected";
      chip.textContent = status === "user_confirmed" ? "marked correct" : "corrected";
      group.appendChild(chip);
      return group;
    }
    group.className = "review-reasons needs-review-hints";
    reasons.slice(0, 3).forEach(reason => {
      const chip = document.createElement("span");
      chip.className = "review-chip";
      chip.textContent = reason;
      group.appendChild(chip);
    });
    if (reasons.length > 3) {
      const chip = document.createElement("span");
      chip.className = "review-chip";
      chip.textContent = `+${reasons.length - 3}`;
      group.appendChild(chip);
    }
    return group;
  }
  function createSpeakerLiveIndicator() {
    const indicator = document.createElement("span");
    indicator.className = "speaker-live-indicator";
    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.8");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("aria-hidden", "true");
    [["2", "10", "2", "14"], ["5", "7", "5", "14"], ["8", "4", "8", "14"], ["11", "8", "11", "14"], ["14", "11", "14", "14"]].forEach(([x1, y1, x2, y2]) => {
      appendSvgElement(svg, "path", {d: `M${x1} ${y1}v${Number(y2) - Number(y1)}`});
    });
    indicator.appendChild(svg);
    indicator.appendChild(document.createTextNode("Live"));
    return indicator;
  }
  function isSpeakerRowControl(target) {
    return target instanceof Element && target.closest(".speaker-row-name-input, .speaker-filter-toggle, .speaker-transcript-action, .speaker-profile-action");
  }
  function setEditingSpeaker(speakerId, options = {}) {
    const requestedId = speakerId || "";
    const collapse = requestedId && ctx.owners.reference.editingSpeakerId === requestedId && !options.keepOpen;
    ctx.owners.reference.editingSpeakerId = collapse ? "" : requestedId;
    ctx.owners.reference.pendingSpeakerNameFocusId = ctx.owners.reference.editingSpeakerId && options.focusName !== false
      ? ctx.owners.reference.editingSpeakerId
      : "";
    if (ctx.owners.reference.editingSpeakerId) {
      ctx.owners.reference.manualSpeakerComposerOpen = false;
      ctx.owners.reference.pendingManualSpeakerNameFocus = false;
    }
    referenceRecordSeconds.textContent = "0.0s";
    referenceSpeakerFile.value = "";
    renderSpeakerPanel();
  }
  function syncSpeakerEditor(speaker) {
    if (!speaker || savedSessionReviewOpen()) {
      referenceSpeakerForm.hidden = true;
      speakerEditorDock.appendChild(referenceSpeakerForm);
      return;
    }
    referenceSpeakerForm.hidden = false;
  }
  function syncManualSpeakerComposer() {
    manualSpeakerComposer.hidden = !ctx.owners.reference.manualSpeakerComposerOpen;
    addReferenceSpeakerButton.setAttribute("aria-expanded", ctx.owners.reference.manualSpeakerComposerOpen ? "true" : "false");
    if (!ctx.owners.reference.manualSpeakerComposerOpen) return;
    ctx.owners.reference.editingSpeakerId = "";
    referenceSpeakerForm.hidden = false;
    manualSpeakerReferenceDock.appendChild(referenceSpeakerForm);
    if (ctx.owners.reference.pendingManualSpeakerNameFocus) {
      ctx.owners.reference.pendingManualSpeakerNameFocus = false;
      requestAnimationFrame(() => {
        manualSpeakerName.focus();
        manualSpeakerName.select();
      });
    }
  }
  function referenceNameMissingMessage() {
    return ctx.owners.reference.manualSpeakerComposerOpen ? "Enter a speaker name first." : "Choose a speaker first.";
  }
  function closeManualSpeakerComposerAfterReference() {
    if (!ctx.owners.reference.manualSpeakerComposerOpen) return;
    ctx.owners.reference.manualSpeakerComposerOpen = false;
    ctx.owners.reference.pendingManualSpeakerNameFocus = false;
    manualSpeakerName.value = "";
  }
  function selectedSpeakerReferenceName() {
    if (ctx.owners.reference.manualSpeakerComposerOpen) {
      return manualSpeakerName.value.trim();
    }
    const inlineName = speakerList.querySelector(".speaker-item.editing .speaker-row-name-input");
    const inlineValue = inlineName ? inlineName.value.trim() : "";
    if (inlineValue) return inlineValue;
    const speaker = selectedSpeaker();
    return speaker ? speakerPanelName(speaker).trim() : "";
  }
  async function commitSpeakerNameInput(speaker, input) {
    if (!speaker || !input || input.dataset.saving === "1") return;
    const currentName = speakerPanelName(speaker);
    const name = input.value.trim();
    if (!name) {
      input.value = currentName;
      log("Enter a speaker name first.");
      return;
    }
    if (name === currentName) {
      input.value = currentName;
      return;
    }
    input.dataset.saving = "1";
    input.disabled = true;
    if (savedSessionReviewOpen()) {
      try {
        const result = await post("/api/sessions/speakers/rename", {
          session_id: ctx.owners.sessions.openedSavedSessionId,
          speaker_id: speaker.id,
          name,
        });
        if (result.session) {
          loadSavedSessionReview(result.session, {quiet:true});
        }
        await fetchSavedSessions();
      } catch (error) {
        input.disabled = false;
        input.dataset.saving = "";
        input.value = currentName;
        log(`Rename failed: ${error.message}`);
      }
      return;
    }
    try {
      await ensureSessionOwner("rename speakers");
    } catch (error) {
      input.disabled = false;
      input.dataset.saving = "";
      log(error.message);
      return;
    }
    try {
      const result = await post("/api/speakers/rename", {speaker_id: speaker.id, name});
      updateSpeakerState(result.speaker_state);
    } catch (error) {
      input.disabled = false;
      input.dataset.saving = "";
      input.value = currentName;
      log(`Rename failed: ${error.message}`);
    }
  }
  function speakerGroupFileName(name) {
    const safe = String(name || "speakers")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._ -]+/g, "")
      .replace(/\s+/g, "-")
      .replace(/^[._ -]+|[._ -]+$/g, "")
      .slice(0, 80) || "speakers";
    return `${safe}.whospeaks-speakers.json`;
  }
  function downloadJsonFile(filename, payload) {
    const text = JSON.stringify(payload, null, 2);
    const url = URL.createObjectURL(new Blob([text], {type: "application/json;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }
  function renderSpeakerPanel() {
    const controlsLocked = sessionControlsLocked();
    const reviewMode = savedSessionReviewOpen();
    speakerPanelTitle.textContent = `Detected speakers (${ctx.owners.speakers.speakerLibraryState.speakers.length})`;
    clearSpeakersButton.disabled = controlsLocked || reviewMode || !ctx.owners.speakers.speakerLibraryState.speakers.length;
    addReferenceSpeakerButton.disabled = controlsLocked || reviewMode;
    manualSpeakerName.disabled = controlsLocked || reviewMode;
    syncManualSpeakerComposer();
    speakerList.textContent = "";
    if (!ctx.owners.speakers.speakerLibraryState.speakers.length) {
      const empty = document.createElement("div");
      empty.className = "speaker-empty";
      empty.textContent = "No speakers yet";
      speakerList.appendChild(empty);
      if (!ctx.owners.reference.manualSpeakerComposerOpen) {
        syncSpeakerEditor(null);
      }
      return;
    }
    const speakerIds = ctx.owners.speakers.speakerLibraryState.speakers.map(speaker => speaker.id).filter(Boolean);
    if (ctx.owners.reference.editingSpeakerId && !speakerIds.includes(ctx.owners.reference.editingSpeakerId)) {
      ctx.owners.reference.editingSpeakerId = "";
    }
    ctx.owners.speakers.speakerLibraryState.speakers.forEach(speaker => {
      const isEditing = speaker.id === ctx.owners.reference.editingSpeakerId;
      const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
      const row = document.createElement("div");
      row.className = `speaker-item${isEditing ? " editing" : ""}`;
      row.classList.toggle("live-speaker", Boolean(ctx.owners.transcript.currentLiveSpeakerId) && speaker.id === ctx.owners.transcript.currentLiveSpeakerId);
      row.dataset.speakerId = speaker.id || "";
      const color = speakerColor(speaker.id);
      row.style.setProperty("--speaker-color", color || "transparent");
      const summary = document.createElement("div");
      summary.className = "speaker-item-summary";
      summary.setAttribute("role", "button");
      summary.tabIndex = controlsLocked ? -1 : 0;
      summary.setAttribute("aria-expanded", isEditing ? "true" : "false");
      summary.addEventListener("click", event => {
        if (isSpeakerRowControl(event.target)) return;
        if (controlsLocked) return;
        setEditingSpeaker(speaker.id || "");
      });
      summary.addEventListener("keydown", event => {
        if (isSpeakerRowControl(event.target)) return;
        if (controlsLocked) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setEditingSpeaker(speaker.id || "");
        }
      });

      const body = document.createElement("span");
      body.className = "speaker-summary-body";
      let title;
      if (isEditing) {
        title = document.createElement("input");
        title.className = "speaker-row-name-input";
        title.type = "text";
        title.value = speakerPanelName(speaker);
        title.disabled = controlsLocked;
        title.setAttribute("aria-label", "Speaker name");
        title.setAttribute("autocomplete", "off");
        title.addEventListener("click", event => event.stopPropagation());
        title.addEventListener("keydown", event => {
          if (event.key === "Enter") {
            event.preventDefault();
            title.blur();
          }
        });
        title.addEventListener("blur", () => {
          commitSpeakerNameInput(speaker, title);
        });
      } else {
        title = document.createElement("span");
        title.className = "speaker-row-title";
        title.textContent = speakerPanelName(speaker);
        title.addEventListener("click", event => {
          event.stopPropagation();
          if (controlsLocked) return;
          setEditingSpeaker(speaker.id || "", {focusName: true});
        });
      }
      const titleRow = document.createElement("span");
      titleRow.className = "speaker-title-row";
      titleRow.appendChild(title);
      if (ctx.owners.transcript.currentLiveSpeakerId && speaker.id === ctx.owners.transcript.currentLiveSpeakerId) {
        titleRow.appendChild(createSpeakerLiveIndicator());
      }
      const sentenceCount = document.createElement("span");
      sentenceCount.className = "speaker-sentence-count";
      sentenceCount.textContent = speakerSentenceText(
        speakerPanelSentenceCount(speaker),
        speakerPanelSpeakingSeconds(speaker),
        speakerPanelCountUnit(),
      );
      body.appendChild(titleRow);
      body.appendChild(sentenceCount);
      if (hasReference) {
        const referenceStatus = document.createElement("span");
        referenceStatus.className = "speaker-reference-status has-reference";
        const referenceIcon = document.createElement("span");
        referenceIcon.className = "speaker-reference-icon";
        referenceIcon.setAttribute("aria-hidden", "true");
        const referenceText = document.createElement("span");
        referenceText.textContent = speakerReferenceText(speaker);
        referenceStatus.appendChild(referenceIcon);
        referenceStatus.appendChild(referenceText);
        body.appendChild(referenceStatus);
      }
      if (isEditing && !reviewMode) {
        const editControls = document.createElement("span");
        editControls.className = "speaker-edit-controls";
        if (ctx.owners.speakers.speakerLibraryState.speakers.length > 1) {
          editControls.appendChild(createSpeakerMergeControls(speaker));
        }
        editControls.appendChild(createSpeakerDeleteButton(speaker));
        body.appendChild(editControls);
      }

      const tail = document.createElement("span");
      tail.className = "speaker-item-tail";
      const filterControls = document.createElement("span");
      filterControls.className = "speaker-filter-controls";
      filterControls.appendChild(createSpeakerFilterToggle(speaker, "solo"));
      filterControls.appendChild(createSpeakerFilterToggle(speaker, "mute"));
      tail.appendChild(filterControls);
      const transcriptActions = document.createElement("span");
      transcriptActions.className = "speaker-transcript-actions";
      transcriptActions.appendChild(createTranscriptActionButton("copy", speaker));
      transcriptActions.appendChild(createTranscriptActionButton("download", speaker));
      tail.appendChild(transcriptActions);
      const chevron = document.createElement("span");
      chevron.className = "speaker-chevron";
      chevron.setAttribute("aria-hidden", "true");
      tail.appendChild(chevron);

      summary.appendChild(body);
      summary.appendChild(tail);
      row.appendChild(summary);
      if (isEditing) {
        syncSpeakerEditor(speaker);
        row.appendChild(referenceSpeakerForm);
        if (ctx.owners.reference.pendingSpeakerNameFocusId === speaker.id && title instanceof HTMLInputElement) {
          ctx.owners.reference.pendingSpeakerNameFocusId = "";
          requestAnimationFrame(() => {
            title.focus();
            title.select();
          });
        }
      }
      speakerList.appendChild(row);
    });
    if (!ctx.owners.reference.editingSpeakerId && !ctx.owners.reference.manualSpeakerComposerOpen) {
      syncSpeakerEditor(null);
    }
  }
  function refreshSpeakerRows() {
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      const label = row.dataset.speaker === "UNKNOWN" ? null : row.dataset.speaker;
      const badge = row.querySelector(".speaker-name");
      if (badge) badge.textContent = speakerDisplayLabel(label);
    });
  }
  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read reference audio."));
      reader.onload = () => resolve(String(reader.result || ""));
      reader.readAsDataURL(file);
    });
  }
  function updateReferenceRecordingControls(recording) {
    const controlsLocked = sessionControlsLocked();
    recordReferenceButton.disabled = controlsLocked;
    referenceSpeakerFile.disabled = controlsLocked || recording;
    recordReferenceButton.classList.toggle("recording", recording);
    if (recordReferenceButtonLabel) {
      recordReferenceButtonLabel.textContent = recording ? "Stop and add" : "Record from mic";
    }
    recordReferenceButton.setAttribute("aria-label", recording ? "Stop and add reference recording" : "Record reference from microphone");
  }
  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const step = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += step) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + step));
    }
    return btoa(binary);
  }
  function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }
  function flattenFloat32Chunks(chunks, totalSamples) {
    const output = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of chunks) {
      output.set(chunk, offset);
      offset += chunk.length;
    }
    return output;
  }
  function encodeWavDataUrl(samples, sampleRate) {
    const wavRate = targetCaptureSampleRate;
    const resampled = resampleFloat32(samples, sampleRate, wavRate);
    const buffer = new ArrayBuffer(44 + resampled.length * 2);
    const view = new DataView(buffer);
    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + resampled.length * 2, true);
    writeAscii(view, 8, "WAVE");
    writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, wavRate, true);
    view.setUint32(28, wavRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeAscii(view, 36, "data");
    view.setUint32(40, resampled.length * 2, true);
    let offset = 44;
    for (let i = 0; i < resampled.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, resampled[i] || 0));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
    return `data:audio/wav;base64,${arrayBufferToBase64(buffer)}`;
  }
  function stopReferenceRecording() {
    ctx.owners.reference.referenceRecordPending = false;
    const chunks = ctx.owners.reference.referenceRecordChunks;
    const totalSamples = ctx.owners.reference.referenceRecordSamples;
    const sampleRate = ctx.owners.reference.referenceRecordSampleRate || targetCaptureSampleRate;
    if (ctx.owners.reference.referenceRecordTimer) {
      clearInterval(ctx.owners.reference.referenceRecordTimer);
      ctx.owners.reference.referenceRecordTimer = null;
    }
    if (ctx.owners.reference.referenceRecordProcessor) {
      try { ctx.owners.reference.referenceRecordProcessor.disconnect(); } catch (_) {}
      ctx.owners.reference.referenceRecordProcessor.onaudioprocess = null;
      ctx.owners.reference.referenceRecordProcessor = null;
    }
    if (ctx.owners.reference.referenceRecordSource) {
      try { ctx.owners.reference.referenceRecordSource.disconnect(); } catch (_) {}
      ctx.owners.reference.referenceRecordSource = null;
    }
    if (ctx.owners.reference.referenceRecordSilentGain) {
      try { ctx.owners.reference.referenceRecordSilentGain.disconnect(); } catch (_) {}
      ctx.owners.reference.referenceRecordSilentGain = null;
    }
    if (ctx.owners.reference.referenceRecordStream) {
      ctx.owners.reference.referenceRecordStream.getTracks().forEach(track => track.stop());
      ctx.owners.reference.referenceRecordStream = null;
    }
    if (ctx.owners.reference.referenceRecordContext) {
      ctx.owners.reference.referenceRecordContext.close().catch(() => {});
      ctx.owners.reference.referenceRecordContext = null;
    }
    ctx.owners.reference.referenceRecordChunks = [];
    ctx.owners.reference.referenceRecordSamples = 0;
    ctx.owners.reference.referenceRecordStartedAt = 0;
    referenceRecordSeconds.textContent = totalSamples > 0 ? `${(totalSamples / sampleRate).toFixed(1)}s` : "0.0s";
    updateReferenceRecordingControls(false);
    return {
      samples: flattenFloat32Chunks(chunks, totalSamples),
      sampleRate,
      seconds: totalSamples / sampleRate,
    };
  }
  async function startReferenceRecording() {
    const name = selectedSpeakerReferenceName();
    if (!name) {
      log(referenceNameMissingMessage());
      return;
    }
    if (ctx.owners.reference.referenceRecordStream || ctx.owners.reference.referenceRecordPending) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      log("Microphone recording is not available in this browser.");
      return;
    }
    ctx.owners.reference.referenceRecordPending = true;
    ctx.owners.reference.referenceRecordChunks = [];
    ctx.owners.reference.referenceRecordSamples = 0;
    ctx.owners.reference.referenceRecordSampleRate = targetCaptureSampleRate;
    referenceRecordSeconds.textContent = "0.0s";
    recordReferenceButton.disabled = true;
    referenceSpeakerFile.disabled = true;
    try {
      ctx.owners.reference.referenceRecordStream = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      ctx.owners.reference.referenceRecordContext = new AudioContextClass({sampleRate: targetCaptureSampleRate});
      ctx.owners.reference.referenceRecordSampleRate = ctx.owners.reference.referenceRecordContext.sampleRate || targetCaptureSampleRate;
      ctx.owners.reference.referenceRecordSource = ctx.owners.reference.referenceRecordContext.createMediaStreamSource(ctx.owners.reference.referenceRecordStream);
      ctx.owners.reference.referenceRecordProcessor = ctx.owners.reference.referenceRecordContext.createScriptProcessor(4096, 1, 1);
      ctx.owners.reference.referenceRecordSilentGain = ctx.owners.reference.referenceRecordContext.createGain();
      ctx.owners.reference.referenceRecordSilentGain.gain.value = 0;
      ctx.owners.reference.referenceRecordProcessor.onaudioprocess = event => {
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        ctx.owners.reference.referenceRecordChunks.push(copy);
        ctx.owners.reference.referenceRecordSamples += copy.length;
        referenceRecordSeconds.textContent = `${(ctx.owners.reference.referenceRecordSamples / ctx.owners.reference.referenceRecordSampleRate).toFixed(1)}s`;
      };
      ctx.owners.reference.referenceRecordSource.connect(ctx.owners.reference.referenceRecordProcessor);
      ctx.owners.reference.referenceRecordProcessor.connect(ctx.owners.reference.referenceRecordSilentGain);
      ctx.owners.reference.referenceRecordSilentGain.connect(ctx.owners.reference.referenceRecordContext.destination);
      await ctx.owners.reference.referenceRecordContext.resume();
      ctx.owners.reference.referenceRecordStartedAt = performance.now();
      ctx.owners.reference.referenceRecordTimer = setInterval(() => {
        const seconds = ctx.owners.reference.referenceRecordSamples > 0
          ? ctx.owners.reference.referenceRecordSamples / ctx.owners.reference.referenceRecordSampleRate
          : (performance.now() - ctx.owners.reference.referenceRecordStartedAt) / 1000;
        referenceRecordSeconds.textContent = `${seconds.toFixed(1)}s`;
      }, 100);
      ctx.owners.reference.referenceRecordPending = false;
      updateReferenceRecordingControls(true);
      log(`Recording reference clip for ${name}.`);
    } catch (error) {
      ctx.owners.reference.referenceRecordPending = false;
      stopReferenceRecording();
      log(`Reference recording failed: ${error.message}`);
    }
  }
  async function stopAndAddReferenceRecording() {
    await ensureSessionOwner("add reference speakers");
    const name = selectedSpeakerReferenceName();
    const recording = stopReferenceRecording();
    if (!name) {
      log(referenceNameMissingMessage());
      return;
    }
    if (recording.seconds < 0.5) {
      log("Reference clip is too short.");
      return;
    }
    recordReferenceButton.disabled = true;
    try {
      const audio_b64 = encodeWavDataUrl(recording.samples, recording.sampleRate);
      const result = await post("/api/speakers/reference", {name, filename: `${name}.wav`, audio_b64});
      closeManualSpeakerComposerAfterReference();
      updateSpeakerState(result.speaker_state);
      referenceRecordSeconds.textContent = "0.0s";
      log(`Added recorded reference speaker ${name}.`);
    } catch (error) {
      log(`Add recorded reference failed: ${error.message}`);
    } finally {
      updateReferenceRecordingControls(false);
    }
  }

  Object.assign(ctx.api, {appendSvgElement, applyCorrectionResult, arrayBufferToBase64, closeManualSpeakerComposerAfterReference, commitSpeakerNameInput, createReviewReasonGroup, createSpeakerDeleteButton, createSpeakerFilterIcon, createSpeakerFilterToggle, createSpeakerFromSelectedSentences, createSpeakerLiveIndicator, createSpeakerMergeControls, createTranscriptActionButton, createTranscriptActionIcon, deleteSpeakerProfile, downloadJsonFile, encodeWavDataUrl, fileToBase64, flattenFloat32Chunks, isSpeakerRowControl, markSelectedSentencesCorrect, mergeSpeakerInto, reassignSelectedSentences, referenceNameMissingMessage, refreshSpeakerRows, renderSpeakerPanel, selectedSpeakerReferenceName, setEditingSpeaker, speakerGroupFileName, speakerReferenceText, speakerSentenceText, speakerSpeakingTimeText, startReferenceRecording, stopAndAddReferenceRecording, stopReferenceRecording, syncManualSpeakerComposer, syncSpeakerEditor, updateReferenceRecordingControls, writeAscii});
  ctx.activators.push(() => {
    updateSpeakerState(ctx.owners.speakers.speakerLibraryState);
  });
}
