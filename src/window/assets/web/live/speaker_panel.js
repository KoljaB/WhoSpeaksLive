export function installSpeakerPanel(ctx) {
  const {addReferenceSpeakerButton, appResources, audio, autoRemoveEmptySpeakers, autoRemoveEmptySpeakersStorageKey, bulkCorrectionSpeaker, clearSpeakersButton, createSpeakerOptionValue, manualSpeakerComposer, manualSpeakerName, manualSpeakerReferenceDock, peopleList, recordReferenceButton, recordReferenceButtonLabel, referenceRecordSeconds, referenceSpeakerFile, referenceSpeakerForm, sentences, source, speakerEditorDock, speakerList, speakerPanelTitle, stop, svgNamespace, targetCaptureSampleRate, video} = ctx;
  const clearTranscriptSelection = (...args) => ctx.api.clearTranscriptSelection(...args), commonSelectedSpeakerId = (...args) => ctx.api.commonSelectedSpeakerId(...args), connect = (...args) => ctx.api.connect(...args), copyTranscript = (...args) => ctx.api.copyTranscript(...args), correctionStatus = (...args) => ctx.api.correctionStatus(...args), downloadTranscript = (...args) => ctx.api.downloadTranscript(...args), ensureSessionOwner = (...args) => ctx.api.ensureSessionOwner(...args), fetchSavedSessions = (...args) => ctx.api.fetchSavedSessions(...args), isLiveProvisionalSpeaker = (...args) => ctx.api.isLiveProvisionalSpeaker(...args), loadSavedSessionReview = (...args) => ctx.api.loadSavedSessionReview(...args), log = (...args) => ctx.api.log(...args), post = (...args) => ctx.api.post(...args), renderSentence = (...args) => ctx.api.renderSentence(...args), resampleFloat32 = (...args) => ctx.api.resampleFloat32(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), scheduleSavedSessionsRefresh = (...args) => ctx.api.scheduleSavedSessionsRefresh(...args), selectedSpeaker = (...args) => ctx.api.selectedSpeaker(...args), selectedTranscriptIndexes = (...args) => ctx.api.selectedTranscriptIndexes(...args), selectedTranscriptRows = (...args) => ctx.api.selectedTranscriptRows(...args), sessionControlsLocked = (...args) => ctx.api.sessionControlsLocked(...args), setSpeakerFilter = (...args) => ctx.api.setSpeakerFilter(...args), speakerColor = (...args) => ctx.api.speakerColor(...args), speakerCurrentSessionSentenceCount = (...args) => ctx.api.speakerCurrentSessionSentenceCount(...args), speakerDisplayLabel = (...args) => ctx.api.speakerDisplayLabel(...args), speakerPanelCountUnit = (...args) => ctx.api.speakerPanelCountUnit(...args), speakerPanelName = (...args) => ctx.api.speakerPanelName(...args), speakerPanelSentenceCount = (...args) => ctx.api.speakerPanelSentenceCount(...args), speakerPanelSpeakingSeconds = (...args) => ctx.api.speakerPanelSpeakingSeconds(...args), syncCorrectionUndoState = (...args) => ctx.api.syncCorrectionUndoState(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  const storeBooleanValue = (...args) => ctx.api.storeBooleanValue(...args);
  const storedBooleanValue = (...args) => ctx.api.storedBooleanValue(...args);
  const toInternalSpeakerId = (...args) => ctx.api.toInternalSpeakerId(...args);
  const emptySpeakerRemovalGraceMs = 8000;
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
  function speakerMeetingName(speaker) {
    if (speaker.name) return speaker.name;
    if (speaker.identity_status === "suggested" && speaker.suggested_person_id) {
      const match = String(speaker.id || "").match(/(\d+)$/);
      if (match) return `Speaker ${Number(match[1])}`;
    }
    return speakerPanelName(speaker);
  }
  function speakerReferenceText(speaker) {
    const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
    if (!hasReference) return "";
    const seconds = Number(speaker.speech_seconds || 0);
    return seconds > 0 ? `Legacy Speaker-group profile (${Math.round(seconds)}s)` : "Legacy Speaker-group profile";
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
    const visibleLabel = document.createElement("span");
    visibleLabel.className = "speaker-filter-label";
    visibleLabel.textContent = label;
    button.appendChild(visibleLabel);
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
  function createTranscriptActionButton(kind, speaker, options = {}) {
    const speakerName = speaker ? speakerPanelName(speaker) : "";
    const label = `${kind === "download" ? "Download" : "Copy"} ${speakerName ? `${speakerName} transcript` : "transcript"}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "transcript-icon-button speaker-transcript-action";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.appendChild(createTranscriptActionIcon(kind));
    if (options.visibleLabel) {
      const text = document.createElement("span");
      text.textContent = kind === "download" ? "Download transcript" : "Copy transcript";
      button.appendChild(text);
    }
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
      const result = await post("/api/corrections/reassign", {indexes, speaker_id: toInternalSpeakerId(speakerId), update_memory: true});
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
      const result = await post("/api/speakers/split", {speaker_id: toInternalSpeakerId(speakerId), sentence_indices: indexes, update_memory: true});
      applyCorrectionResult(result);
      clearTranscriptSelection();
      log(`Created ${speakerDisplayLabel(result.new_speaker_id)} from ${indexes.length} selected sentence${indexes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      log(`Create speaker failed: ${error.message}`);
    }
  }
  async function mergeSpeakerInto(sourceSpeaker, targetSpeaker) {
    const sourceSpeakerId = String((sourceSpeaker && sourceSpeaker.id) || "");
    const targetSpeakerId = String((targetSpeaker && targetSpeaker.id) || "");
    if (!sourceSpeakerId || !targetSpeakerId || sourceSpeakerId === targetSpeakerId) return;
    const sourceInternalId = String(sourceSpeaker.internal_speaker_id || toInternalSpeakerId(sourceSpeakerId));
    const targetInternalId = String(targetSpeaker.internal_speaker_id || toInternalSpeakerId(targetSpeakerId));
    try {
      await ensureSessionOwner("merge speaker profiles");
      const result = await post("/api/speakers/merge", {
        source_speaker_id: sourceInternalId,
        target_speaker_id: targetInternalId,
        expected_source_sentence_count: speakerCurrentSessionSentenceCount(sourceSpeakerId),
        expected_target_sentence_count: speakerCurrentSessionSentenceCount(targetSpeakerId),
        update_memory: true,
      });
      applyCorrectionResult(result);
      const projectedTarget = (result.speaker_state && result.speaker_state.public_speakers || [])
        .find(candidate => candidate.internal_speaker_id === targetInternalId);
      ctx.owners.reference.editingSpeakerId = projectedTarget ? projectedTarget.id : targetInternalId;
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
      const result = await post("/api/speakers/delete", {
        speaker_id: String(speaker.internal_speaker_id || toInternalSpeakerId(speakerId)),
        expected_sentence_count: sentenceTotal,
        update_memory: true,
      });
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
      .filter(candidate => candidate.id && candidate.id !== speaker.id && !isLiveProvisionalSpeaker(candidate))
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
      const targetSpeaker = ctx.owners.speakers.speakerLibraryState.speakers
        .find(candidate => candidate.id === select.value);
      mergeSpeakerInto(speaker, targetSpeaker);
    });
    select.addEventListener("click", event => event.stopPropagation());
    select.addEventListener("keydown", event => event.stopPropagation());
    controls.appendChild(select);
    controls.appendChild(button);
    return controls;
  }
  function createSpeakerDeleteButton(speaker, label = "Delete Speaker", options = {}) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "speaker-delete-button speaker-profile-action";
    button.textContent = label;
    button.disabled = sessionControlsLocked() || savedSessionReviewOpen();
    button.addEventListener("click", event => {
      event.stopPropagation();
      if (options.emptyOnly) removeEmptySpeakerProfile(speaker);
      else deleteSpeakerProfile(speaker);
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
    indicator.appendChild(document.createTextNode("Speaking now"));
    return indicator;
  }
  function isSpeakerRowControl(target) {
    return target instanceof Element && (
      target.closest("button, input, select, label, form")
      || target.closest(".speaker-identity-controls")
    );
  }
  function setEditingSpeaker(speakerId, options = {}) {
    const requestedId = speakerId || "";
    const collapse = requestedId && ctx.owners.reference.editingSpeakerId === requestedId && !options.keepOpen;
    ctx.owners.reference.editingSpeakerId = collapse ? "" : requestedId;
    ctx.owners.reference.pendingSpeakerNameFocusId = ctx.owners.reference.editingSpeakerId && options.focusName === true
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
    if (!speaker || savedSessionReviewOpen() || !speaker.person_id) {
      referenceSpeakerForm.hidden = true;
      speakerEditorDock.appendChild(referenceSpeakerForm);
      return;
    }
    ctx.owners.reference.voiceSamplePersonId = speaker.person_id;
    const person = personById(speaker.person_id);
    const title = referenceSpeakerForm.querySelector(".speaker-reference-title");
    if (title) title.textContent = `Add voice sample to ${person ? person.name : "Person"}`;
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
    const targetPerson = personById(ctx.owners.reference.voiceSamplePersonId || "");
    if (targetPerson) return targetPerson.name;
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
    if (!speaker || !input || input.dataset.saving === "1") return false;
    const currentName = speakerMeetingName(speaker);
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
        return false;
      }
      return true;
    }
    try {
      await ensureSessionOwner("rename speakers");
    } catch (error) {
      input.disabled = false;
      input.dataset.saving = "";
      log(error.message);
      return false;
    }
    try {
      const result = await post("/api/speakers/rename", {speaker_id: toInternalSpeakerId(speaker.id), name});
      updateSpeakerState(result.speaker_state);
      return true;
    } catch (error) {
      input.disabled = false;
      input.dataset.saving = "";
      input.value = currentName;
      log(`Rename failed: ${error.message}`);
      return false;
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
  function identityActionButton(label, className, action, disabledReason = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `speaker-identity-button ${className || ""}`.trim();
    button.textContent = label;
    button.disabled = Boolean(disabledReason);
    if (disabledReason) button.title = disabledReason;
    button.addEventListener("click", async event => {
      event.stopPropagation();
      button.disabled = true;
      try {
        if (!savedSessionReviewOpen()) await ensureSessionOwner("manage remembered people");
        await action();
      } catch (error) {
        log(error.message || String(error));
      } finally {
        button.disabled = false;
      }
    });
    return button;
  }
  function peopleState() {
    return Array.isArray(ctx.owners.speakers.speakerLibraryState.people)
      ? ctx.owners.speakers.speakerLibraryState.people
      : [];
  }
  function personById(personId) {
    return peopleState().find(person => person.id === personId) || null;
  }
  function choosePersonTarget(speaker) {
    const people = peopleState();
    const choices = people.map((person, index) => `${index + 1}. ${person.name} (${String(person.id).slice(0, 6)})`).join("\n");
    const defaultName = String(speaker.name || "").trim();
    const answer = window.prompt(
      `${choices ? `Choose an existing Person by number:\n${choices}\n\n` : ""}Enter a number, or enter a new Person name:`,
      defaultName,
    );
    if (answer === null) return null;
    const selectedIndex = Number(answer.trim());
    if (Number.isInteger(selectedIndex) && selectedIndex >= 1 && selectedIndex <= people.length) {
      return {person_id: people[selectedIndex - 1].id, person_name: ""};
    }
    const personName = answer.trim();
    return personName ? {person_id: "", person_name: personName} : null;
  }
  function openVoiceSampleForm(person) {
    ctx.owners.reference.voiceSamplePersonId = person.id;
    ctx.owners.reference.manualSpeakerComposerOpen = false;
    ctx.owners.reference.editingSpeakerId = "";
    referenceSpeakerForm.hidden = false;
    const title = referenceSpeakerForm.querySelector(".speaker-reference-title");
    if (title) title.textContent = `Add voice sample to ${person.name}`;
    peopleList.appendChild(referenceSpeakerForm);
  }
  function createSpeakerIdentityControls(speaker) {
    const controls = document.createElement("span");
    controls.className = "speaker-identity-controls";
    const saved = savedSessionReviewOpen();
    if (!saved && speaker.identity_status === "suggested" && speaker.suggested_person_id) {
      const status = document.createElement("span");
      status.className = "speaker-identity-status suggested";
      status.textContent = `Likely ${speaker.suggested_person_name}`;
      controls.appendChild(status);
      controls.appendChild(identityActionButton("Confirm", "confirm", async () => {
        const result = await post("/api/people/confirm", {
          speaker_id: toInternalSpeakerId(speaker.id),
          person_id: speaker.suggested_person_id,
        });
        updateSpeakerState(result.speaker_state);
        log(`Confirmed ${speaker.suggested_person_name}.`);
      }));
      controls.appendChild(identityActionButton(`Not ${speaker.suggested_person_name}`, "reject", async () => {
        const result = await post("/api/people/reject", {
          speaker_id: toInternalSpeakerId(speaker.id),
          person_id: speaker.suggested_person_id,
        });
        updateSpeakerState(result.speaker_state);
        log(`Kept ${speaker.id} unidentified.`);
      }));
      return controls;
    }
    if (speaker.identity_status === "confirmed" && speaker.person_id) {
      const person = personById(speaker.person_id);
      const personName = person ? person.name : (speaker.name || "Person");
      const status = document.createElement("span");
      status.className = "speaker-identity-status confirmed";
      const recognitionState = person
        ? (person.recognition_enabled ? "Recognition active" : "Recognition paused")
        : "Person unavailable";
      status.textContent = `Linked to ${personName} · ${recognitionState}`;
      controls.appendChild(status);
      if (!saved && person) {
        controls.appendChild(identityActionButton("Add voice sample", "sample", async () => openVoiceSampleForm(person)));
      }
      controls.appendChild(identityActionButton("Unlink", "reject", async () => {
        const result = saved
          ? await post("/api/sessions/people/unlink", {session_id: ctx.owners.sessions.openedSavedSessionId, speaker_id: speaker.id})
          : await post("/api/people/unlink", {speaker_id: toInternalSpeakerId(speaker.id)});
        if (saved && result.session) loadSavedSessionReview(result.session, {quiet:true});
        else updateSpeakerState(result.speaker_state);
      }));
      return controls;
    }
    const unavailable = saved && speaker.future_recognition && !speaker.future_recognition.available
      ? (speaker.future_recognition.explanation || "Compatible saved voice evidence is unavailable.")
      : "";
    controls.appendChild(identityActionButton("Link to Person…", "remember", async () => {
      const target = choosePersonTarget(speaker);
      if (!target) return;
      if (saved) {
        const result = await post("/api/sessions/people/link", {
          session_id: ctx.owners.sessions.openedSavedSessionId,
          speaker_id: speaker.id,
          ...target,
        });
        if (result.session) loadSavedSessionReview(result.session, {quiet:true});
      } else {
        const result = await post("/api/people/remember", {
          speaker_id: toInternalSpeakerId(speaker.id),
          name: target.person_name,
          person_id: target.person_id,
        });
        updateSpeakerState(result.speaker_state);
      }
    }, unavailable));
    if (unavailable) {
      const reason = document.createElement("span");
      reason.className = "speaker-identity-unavailable";
      reason.textContent = unavailable;
      controls.appendChild(reason);
    }
    return controls;
  }
  function embeddingStackDescription(stack, options = {}) {
    if (!stack) return "Unknown embedding stack";
    const parts = [stack.label || "Unknown embedding stack"];
    if (Number(stack.dimensions) > 0) parts.push(`${stack.dimensions} dimensions`);
    if (options.sampleCount && Number(stack.sample_count) > 0) {
      const count = Number(stack.sample_count);
      parts.push(`${count} sample${count === 1 ? "" : "s"}`);
    }
    return parts.join(" · ");
  }
  function appendEmbeddingStackDetails(parent, label, stack, options = {}) {
    const row = document.createElement("div");
    row.className = "embedding-stack-detail";
    const heading = document.createElement("strong");
    heading.textContent = label;
    const summary = document.createElement("span");
    summary.textContent = embeddingStackDescription(stack, options);
    const identifier = document.createElement("code");
    identifier.textContent = stack && stack.identifier ? stack.identifier : "Provider identifier unavailable";
    row.appendChild(heading);
    row.appendChild(summary);
    row.appendChild(identifier);
    parent.appendChild(row);
  }
  function createEmbeddingMismatchDetails(person) {
    const details = document.createElement("details");
    details.className = "embedding-mismatch-details";
    const summary = document.createElement("summary");
    summary.textContent = "Saved samples and this session use different embedding stacks.";
    details.appendChild(summary);
    const content = document.createElement("div");
    content.className = "embedding-mismatch-content";
    appendEmbeddingStackDetails(content, "Current session", person.current_embedding_stack);
    (person.active_sample_embedding_stacks || []).forEach((stack, index) => {
      appendEmbeddingStackDetails(
        content,
        index ? `Saved samples ${index + 1}` : "Saved samples",
        stack,
        {sampleCount:true},
      );
    });
    details.appendChild(content);
    return details;
  }
  function recognitionStatusText(person) {
    if (person.recognition_ready) {
      return `${person.active_voice_sample_count} active Voice sample${person.active_voice_sample_count === 1 ? "" : "s"}`;
    }
    if (["embedding_provider_mismatch", "embedding_dimension_mismatch"].includes(person.recognition_unavailable_reason)) {
      return "Recognition unavailable · Speaker embedding mismatch";
    }
    if (person.recognition_unavailable_reason === "no_voice_samples") return "Recognition unavailable · No Voice samples";
    if (person.recognition_unavailable_reason === "no_active_voice_samples") return "Recognition unavailable · No active Voice samples";
    return "Recognition unavailable · No active compatible Voice samples";
  }
  function voiceSampleStateLabel(sample) {
    if (sample.state && sample.state !== "active") return sample.state;
    if (sample.compatibility_reason === "embedding_provider_mismatch") return "different embedding stack";
    if (sample.compatibility_reason === "embedding_dimension_mismatch") return "different embedding dimensions";
    return sample.effective_state;
  }
  function renderPeopleList() {
    if (!peopleList) return;
    peopleList.textContent = "";
    const people = peopleState();
    if (!people.length) {
      const empty = document.createElement("span");
      empty.className = "people-empty";
      empty.textContent = "No people saved yet. Add a person here or link a meeting Speaker.";
      peopleList.appendChild(empty);
      return false;
    }
    const expected = new Set(ctx.owners.speakers.speakerLibraryState.expected_person_ids || []);
    people.forEach(person => {
      const row = document.createElement("div");
      row.className = "person-row";
      const heading = document.createElement("div");
      heading.className = "person-heading";
      const identity = document.createElement("div");
      identity.className = "person-identity";
      const name = document.createElement("strong");
      name.className = "person-name";
      name.textContent = person.name;
      identity.appendChild(name);
      const details = document.createElement("span");
      details.className = "person-details";
      details.textContent = recognitionStatusText(person);
      identity.appendChild(details);
      if (["embedding_provider_mismatch", "embedding_dimension_mismatch"].includes(person.recognition_unavailable_reason)) {
        identity.appendChild(createEmbeddingMismatchDetails(person));
      }
      heading.appendChild(identity);
      const persistentToggles = document.createElement("div");
      persistentToggles.className = "person-persistent-toggles";
      const expectedLabel = document.createElement("label");
      expectedLabel.className = "person-expected person-setting-toggle person-policy-toggle";
      const expectedCheckbox = document.createElement("input");
      expectedCheckbox.type = "checkbox";
      expectedCheckbox.checked = expected.has(person.id);
      expectedCheckbox.disabled = savedSessionReviewOpen();
      expectedCheckbox.addEventListener("change", async () => {
        if (expectedCheckbox.checked) expected.add(person.id); else expected.delete(person.id);
        try {
          await ensureSessionOwner("set expected People");
          const result = await post("/api/people/expected", {person_ids: [...expected]});
          updateSpeakerState(result.speaker_state);
        } catch (error) {
          expectedCheckbox.checked = !expectedCheckbox.checked;
          log(error.message || String(error));
        }
      });
      expectedLabel.appendChild(expectedCheckbox);
      const expectedCopy = document.createElement("span");
      expectedCopy.className = "setting-copy";
      const expectedTitle = document.createElement("strong");
      expectedTitle.textContent = "Include in automatic recognition";
      const expectedHint = document.createElement("span");
      expectedHint.textContent = "Saved until you change it. Keep this roster limited to plausible attendees.";
      expectedCopy.appendChild(expectedTitle);
      expectedCopy.appendChild(expectedHint);
      expectedLabel.appendChild(expectedCopy);
      const recognition = document.createElement("label");
      recognition.className = "person-recognition person-setting-toggle person-policy-toggle";
      const recognitionCheckbox = document.createElement("input");
      recognitionCheckbox.type = "checkbox";
      recognitionCheckbox.checked = Boolean(person.recognition_enabled);
      recognitionCheckbox.addEventListener("change", async () => {
        try {
          await ensureSessionOwner("change Person recognition");
          const result = await post("/api/people/recognition", {person_id: person.id, enabled: recognitionCheckbox.checked});
          updateSpeakerState(result.speaker_state);
        } catch (error) { recognitionCheckbox.checked = !recognitionCheckbox.checked; log(error.message || String(error)); }
      });
      recognition.appendChild(recognitionCheckbox);
      const recognitionCopy = document.createElement("span");
      recognitionCopy.className = "setting-copy";
      const recognitionTitle = document.createElement("strong");
      recognitionTitle.textContent = "Recognition active";
      const recognitionHint = document.createElement("span");
      recognitionHint.textContent = "Allow compatible Voice samples from this Person to participate in matching.";
      recognitionCopy.appendChild(recognitionTitle);
      recognitionCopy.appendChild(recognitionHint);
      recognition.appendChild(recognitionCopy);
      persistentToggles.appendChild(expectedLabel);
      persistentToggles.appendChild(recognition);
      const policy = document.createElement("details");
      policy.className = "person-policy";
      const policySummary = document.createElement("summary");
      policySummary.textContent = "Recognition sources";
      policy.appendChild(policySummary);
      [["manual_samples", "Manually added Voice samples"], ["meeting_samples", "Confirmed meeting samples"], ["learn_from_confirmed_meetings", "Learn from confirmed meetings"]].forEach(([key, label]) => {
        const control = document.createElement("label");
        control.className = "person-policy-option";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = Boolean((person.recognition_policy || {})[key]);
        input.addEventListener("change", async () => {
          try {
            await ensureSessionOwner("customize recognition policy");
            const result = await post("/api/people/policy", {person_id: person.id, recognition_policy: {[key]: input.checked}});
            updateSpeakerState(result.speaker_state);
          } catch (error) { input.checked = !input.checked; log(error.message || String(error)); }
        });
        control.appendChild(input); control.appendChild(document.createTextNode(label)); policy.appendChild(control);
      });
      const actions = document.createElement("span");
      actions.className = "person-actions";
      const addSample = document.createElement("button");
      addSample.type = "button";
      addSample.className = "person-primary-action";
      addSample.textContent = "Add Voice sample";
      addSample.addEventListener("click", () => openVoiceSampleForm(person));
      const forget = document.createElement("button");
      forget.type = "button";
      forget.className = "person-forget-button";
      forget.textContent = "Forget voice data";
      forget.disabled = !person.voice_sample_count;
      forget.addEventListener("click", async () => {
        if (!window.confirm(`Remove all Person-owned Voice samples and retained manual audio for ${person.name}? Saved meeting links will be removed; historical transcript labels and meeting evidence will remain.`)) return;
        forget.disabled = true;
        try {
          await ensureSessionOwner("forget Voice data");
          const result = await post("/api/people/forget-voice", {person_id: person.id});
          updateSpeakerState(result.speaker_state);
          log(`Forgot Person-owned Voice data for ${person.name}; saved identity links were removed and transcript labels were kept.`);
        } catch (error) {
          log(error.message || String(error));
        }
      });
      actions.appendChild(addSample); actions.appendChild(forget);
      const deletePerson = document.createElement("button");
      deletePerson.type = "button";
      deletePerson.className = "person-danger-action";
      deletePerson.textContent = "Delete person";
      deletePerson.addEventListener("click", async () => {
        if (!window.confirm(`Delete ${person.name}, all Person-owned Voice samples, retained manual audio, and saved identity links? Historical transcript labels and meeting evidence will remain.`)) return;
        try {
          await ensureSessionOwner("delete a Person");
          const result = await post("/api/people/delete", {person_id: person.id});
          updateSpeakerState(result.speaker_state);
        } catch (error) { log(error.message || String(error)); }
      });
      actions.appendChild(deletePerson);
      const samples = document.createElement("div");
      samples.className = "voice-sample-list";
      const samplesHeading = document.createElement("div");
      samplesHeading.className = "voice-sample-heading";
      samplesHeading.textContent = person.voice_sample_count
        ? `Voice samples (${person.voice_sample_count})`
        : "No Voice samples yet";
      samples.appendChild(samplesHeading);
      (person.voice_samples || []).forEach(sample => {
        const sampleRow = document.createElement("div");
        sampleRow.className = "voice-sample-row";
        const kind = sample.kind === "manual_reference" ? "Manual" : "Confirmed meeting";
        const sampleCopy = document.createElement("span");
        sampleCopy.className = "voice-sample-copy";
        const sampleLabel = document.createElement("strong");
        sampleLabel.textContent = sample.label;
        const sampleMeta = document.createElement("span");
        sampleMeta.textContent = `${kind} · ${sample.speech_seconds}s · ${voiceSampleStateLabel(sample)}`;
        sampleCopy.appendChild(sampleLabel);
        sampleCopy.appendChild(sampleMeta);
        sampleRow.appendChild(sampleCopy);
        if (sample.raw_audio_retained) sampleRow.title = "Original audio is retained locally for audit and re-embedding.";
        const sampleActions = document.createElement("span");
        sampleActions.className = "voice-sample-actions";
        const toggle = document.createElement("button");
        toggle.type = "button"; toggle.textContent = sample.state === "disabled" ? "Enable" : "Disable";
        toggle.addEventListener("click", async () => {
          try {
            await ensureSessionOwner("manage Voice samples");
            const result = await post("/api/people/sample/state", {person_id: person.id, sample_id: sample.id, enabled: sample.state === "disabled"});
            updateSpeakerState(result.speaker_state);
          } catch (error) { log(error.message || String(error)); }
        });
        const remove = document.createElement("button");
        remove.type = "button"; remove.textContent = "Delete";
        remove.addEventListener("click", async () => {
          if (!window.confirm(`Remove ${sample.label}, including retained audio and derived representations? A meeting-derived sample will stay suppressed during background recalculation; historical transcript labels will not change.`)) return;
          try {
            await ensureSessionOwner("delete a Voice sample");
            const result = await post("/api/people/sample/delete", {person_id: person.id, sample_id: sample.id});
            updateSpeakerState(result.speaker_state);
          } catch (error) { log(error.message || String(error)); }
        });
        sampleActions.appendChild(toggle); sampleActions.appendChild(remove);
        sampleRow.appendChild(sampleActions); samples.appendChild(sampleRow);
      });
      row.appendChild(heading);
      row.appendChild(persistentToggles);
      row.appendChild(policy);
      row.appendChild(actions);
      row.appendChild(samples);
      peopleList.appendChild(row);
    });
  }
  async function removeEmptySpeakerProfile(speaker) {
    if (!speaker || !speaker.id) return;
    const speakerId = speaker.id;
    const internalSpeakerId = String(speaker.internal_speaker_id || toInternalSpeakerId(speakerId));
    try {
      await ensureSessionOwner("remove empty Speakers");
      const result = await post("/api/speakers/remove-empty", {speaker_ids: [internalSpeakerId]});
      updateSpeakerState(result.speaker_state);
      const removed = Array.isArray(result.removed_speaker_ids) ? result.removed_speaker_ids : [];
      if (removed.includes(internalSpeakerId)) {
        ctx.owners.speakers.emptySpeakerFirstSeenAt.delete(speakerId);
        log(`Removed empty Speaker ${speakerPanelName(speaker)}.`);
      } else {
        log(`${speakerPanelName(speaker)} was not removed because the server no longer considers it empty.`);
      }
    } catch (error) {
      log(`Remove empty Speaker failed: ${error.message}`);
    }
  }
  function clearAutoRemoveEmptySpeakerTimer() {
    if (ctx.owners.speakers.autoRemoveEmptySpeakerTimer !== null) {
      clearTimeout(ctx.owners.speakers.autoRemoveEmptySpeakerTimer);
      ctx.owners.speakers.autoRemoveEmptySpeakerTimer = null;
    }
  }
  function isAutoRemovableEmptySpeaker(speaker) {
    if (!speaker || !speaker.id || isLiveProvisionalSpeaker(speaker)) return false;
    if (speaker.id === ctx.owners.transcript.currentLiveSpeakerId) return false;
    if (speaker.id === ctx.owners.reference.editingSpeakerId) return false;
    if (speaker.reference_audio || speaker.locked || speaker.source === "reference") return false;
    if (speaker.identity_status === "confirmed" || speaker.person_id) return false;
    return speakerPanelSentenceCount(speaker) === 0 && speakerPanelSpeakingSeconds(speaker) === 0;
  }
  function autoRemoveEmptySpeakerPlan() {
    const firstSeen = ctx.owners.speakers.emptySpeakerFirstSeenAt;
    if (!autoRemoveEmptySpeakers.checked || savedSessionReviewOpen() || sessionControlsLocked()) {
      firstSeen.clear();
      return {speakerIds: [], delayMs: null};
    }
    const now = performance.now();
    const eligible = ctx.owners.speakers.speakerLibraryState.speakers.filter(isAutoRemovableEmptySpeaker);
    const eligibleIds = new Set(eligible.map(speaker => speaker.id));
    Array.from(firstSeen.keys()).forEach(speakerId => {
      if (!eligibleIds.has(speakerId)) firstSeen.delete(speakerId);
    });
    let nextDelayMs = null;
    const speakerIds = [];
    eligible.forEach(speaker => {
      if (!firstSeen.has(speaker.id)) firstSeen.set(speaker.id, now);
      const remaining = emptySpeakerRemovalGraceMs - (now - firstSeen.get(speaker.id));
      if (remaining <= 0) speakerIds.push(speaker.id);
      else nextDelayMs = nextDelayMs === null ? remaining : Math.min(nextDelayMs, remaining);
    });
    return {speakerIds, delayMs: speakerIds.length ? 50 : nextDelayMs};
  }
  function scheduleAutoRemoveEmptySpeakers() {
    clearAutoRemoveEmptySpeakerTimer();
    if (ctx.owners.speakers.autoRemoveEmptySpeakerRequestPending) return;
    const plan = autoRemoveEmptySpeakerPlan();
    if (plan.delayMs === null) return;
    ctx.owners.speakers.autoRemoveEmptySpeakerTimer = setTimeout(
      runAutoRemoveEmptySpeakers,
      Math.max(50, Math.ceil(plan.delayMs)),
    );
  }
  async function runAutoRemoveEmptySpeakers() {
    ctx.owners.speakers.autoRemoveEmptySpeakerTimer = null;
    if (ctx.owners.speakers.autoRemoveEmptySpeakerRequestPending) return;
    const {speakerIds} = autoRemoveEmptySpeakerPlan();
    if (!speakerIds.length) {
      scheduleAutoRemoveEmptySpeakers();
      return;
    }
    ctx.owners.speakers.autoRemoveEmptySpeakerRequestPending = true;
    try {
      await ensureSessionOwner("remove empty Speakers");
      const result = await post("/api/speakers/remove-empty", {speaker_ids: speakerIds.map(toInternalSpeakerId)});
      const removed = Array.isArray(result.removed_speaker_ids) ? result.removed_speaker_ids : [];
      removed.forEach(speakerId => ctx.owners.speakers.emptySpeakerFirstSeenAt.delete(speakerId));
      if (removed.length) {
        updateSpeakerState(result.speaker_state);
        log(`Automatically removed empty Speaker${removed.length === 1 ? "" : "s"} ${removed.join(", ")}.`);
      } else {
        const now = performance.now();
        speakerIds.forEach(speakerId => ctx.owners.speakers.emptySpeakerFirstSeenAt.set(speakerId, now));
      }
    } catch (error) {
      const now = performance.now();
      speakerIds.forEach(speakerId => ctx.owners.speakers.emptySpeakerFirstSeenAt.set(speakerId, now));
      log(`Automatic empty-Speaker cleanup failed: ${error.message}`);
    } finally {
      ctx.owners.speakers.autoRemoveEmptySpeakerRequestPending = false;
      scheduleAutoRemoveEmptySpeakers();
    }
  }
  function renderSpeakerPanel() {
    const controlsLocked = sessionControlsLocked();
    const reviewMode = savedSessionReviewOpen();
    const allSpeakers = ctx.owners.speakers.speakerLibraryState.speakers;
    const detectedSpeakers = allSpeakers.filter(speaker => !isLiveProvisionalSpeaker(speaker));
    speakerPanelTitle.textContent = `Detected speakers (${detectedSpeakers.length})`;
    clearSpeakersButton.disabled = controlsLocked || reviewMode || !allSpeakers.length;
    addReferenceSpeakerButton.disabled = controlsLocked || reviewMode;
    manualSpeakerName.disabled = controlsLocked || reviewMode;
    syncManualSpeakerComposer();
    renderPeopleList();
    speakerList.textContent = "";
    if (!allSpeakers.length) {
      clearAutoRemoveEmptySpeakerTimer();
      ctx.owners.speakers.emptySpeakerFirstSeenAt.clear();
      const empty = document.createElement("div");
      empty.className = "speaker-empty";
      empty.textContent = "No speakers yet";
      speakerList.appendChild(empty);
      if (!ctx.owners.reference.manualSpeakerComposerOpen) {
        syncSpeakerEditor(null);
      }
      return true;
    }
    const speakerIds = allSpeakers.map(speaker => speaker.id).filter(Boolean);
    if (ctx.owners.reference.editingSpeakerId && !speakerIds.includes(ctx.owners.reference.editingSpeakerId)) {
      ctx.owners.reference.editingSpeakerId = "";
    }
    allSpeakers.forEach(speaker => {
      const isLiveProvisional = isLiveProvisionalSpeaker(speaker);
      const isEditing = !isLiveProvisional && speaker.id === ctx.owners.reference.editingSpeakerId;
      const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
      const row = document.createElement("div");
      row.className = `speaker-item${isEditing ? " editing" : ""}`;
      row.classList.toggle("provisional-speaker", isLiveProvisional);
      row.classList.toggle("live-speaker", Boolean(ctx.owners.transcript.currentLiveSpeakerId) && speaker.id === ctx.owners.transcript.currentLiveSpeakerId);
      row.dataset.speakerId = speaker.id || "";
      const color = speakerColor(speaker.id);
      row.style.setProperty("--speaker-color", color || "#8AA0B5");
      const summary = document.createElement("div");
      summary.className = "speaker-item-summary";
      summary.setAttribute("role", isLiveProvisional ? "status" : "button");
      summary.tabIndex = isLiveProvisional || controlsLocked ? -1 : 0;
      if (!isLiveProvisional) {
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
      }

      const body = document.createElement("span");
      body.className = "speaker-summary-body";
      const panelSentenceCount = speakerPanelSentenceCount(speaker);
      const panelSpeakingSeconds = speakerPanelSpeakingSeconds(speaker);
      const isEmpty = !isLiveProvisional && panelSentenceCount === 0 && panelSpeakingSeconds === 0;
      row.classList.toggle("empty-speaker", isEmpty);
      const title = document.createElement("span");
      title.className = "speaker-row-title";
      title.textContent = isLiveProvisional ? "Matching new voice..." : speakerMeetingName(speaker);
      const titleRow = document.createElement("span");
      titleRow.className = "speaker-title-row";
      titleRow.appendChild(title);
      const statusRow = document.createElement("span");
      statusRow.className = "speaker-status-row";
      if (isEmpty) {
        const emptyBadge = document.createElement("span");
        emptyBadge.className = "speaker-empty-badge";
        emptyBadge.textContent = "Empty Speaker";
        statusRow.appendChild(emptyBadge);
      }
      const sentenceCount = document.createElement("span");
      sentenceCount.className = "speaker-sentence-count";
      sentenceCount.textContent = isLiveProvisional
        ? "Comparing with detected speakers..."
        : isEmpty
        ? "No transcript assigned"
        : speakerSentenceText(panelSentenceCount, panelSpeakingSeconds, speakerPanelCountUnit());
      body.appendChild(statusRow);
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
      const showIdentityControls = !isLiveProvisional && (!isEmpty || speaker.identity_status === "suggested" || speaker.identity_status === "confirmed" || hasReference);
      if (showIdentityControls) {
        const identityControls = createSpeakerIdentityControls(speaker);
        if (identityControls.children.length) body.appendChild(identityControls);
      }
      if (isEmpty && !reviewMode && speaker.identity_status !== "confirmed" && !hasReference) {
        const emptyAction = createSpeakerDeleteButton(speaker, "Remove empty Speaker", {emptyOnly: true});
        emptyAction.classList.add("speaker-empty-remove");
        body.appendChild(emptyAction);
      }

      const tail = document.createElement("span");
      tail.className = "speaker-item-tail";
      if (!isLiveProvisional) {
        const filterControls = document.createElement("span");
        filterControls.className = "speaker-filter-controls";
        filterControls.appendChild(createSpeakerFilterToggle(speaker, "solo"));
        filterControls.appendChild(createSpeakerFilterToggle(speaker, "mute"));
        tail.appendChild(filterControls);
        const chevron = document.createElement("span");
        chevron.className = "speaker-chevron";
        chevron.setAttribute("aria-hidden", "true");
        tail.appendChild(chevron);
      }

      const header = document.createElement("span");
      header.className = "speaker-summary-header";
      header.appendChild(titleRow);
      if (!isLiveProvisional) header.appendChild(tail);
      summary.appendChild(header);
      summary.appendChild(body);
      row.appendChild(summary);
      row.appendChild(createSpeakerLiveIndicator());
      if (isEditing) {
        const expanded = document.createElement("div");
        expanded.className = "speaker-expanded-panel";

        const identitySection = document.createElement("section");
        identitySection.className = "speaker-expanded-section speaker-name-section";
        const identityHeading = document.createElement("h3");
        identityHeading.textContent = "Identity";
        const nameLabel = document.createElement("label");
        nameLabel.className = "speaker-field-label";
        nameLabel.textContent = "Meeting display name";
        const nameEditor = document.createElement("div");
        nameEditor.className = "speaker-name-editor";
        const nameInput = document.createElement("input");
        nameInput.className = "speaker-row-name-input";
        nameInput.type = "text";
        nameInput.value = speakerMeetingName(speaker);
        nameInput.disabled = controlsLocked;
        nameInput.setAttribute("aria-label", "Meeting display name");
        nameInput.setAttribute("autocomplete", "off");
        const saveName = document.createElement("button");
        saveName.type = "button";
        saveName.className = "speaker-name-save";
        saveName.textContent = "Save";
        saveName.disabled = controlsLocked;
        const cancelName = document.createElement("button");
        cancelName.type = "button";
        cancelName.className = "speaker-name-cancel";
        cancelName.textContent = "Cancel";
        saveName.addEventListener("click", async () => {
          if (await commitSpeakerNameInput(speaker, nameInput)) {
            ctx.owners.reference.editingSpeakerId = "";
            renderSpeakerPanel();
          }
        });
        cancelName.addEventListener("click", () => setEditingSpeaker(""));
        nameInput.addEventListener("keydown", event => {
          if (event.key === "Enter") {
            event.preventDefault();
            saveName.click();
          } else if (event.key === "Escape") {
            event.preventDefault();
            cancelName.click();
          }
        });
        nameEditor.appendChild(nameInput);
        nameEditor.appendChild(saveName);
        nameEditor.appendChild(cancelName);
        identitySection.appendChild(identityHeading);
        identitySection.appendChild(nameLabel);
        identitySection.appendChild(nameEditor);
        syncSpeakerEditor(speaker);
        identitySection.appendChild(referenceSpeakerForm);
        expanded.appendChild(identitySection);

        const transcriptSection = document.createElement("section");
        transcriptSection.className = "speaker-expanded-section speaker-transcript-section";
        const transcriptHeading = document.createElement("h3");
        transcriptHeading.textContent = "Transcript";
        const transcriptActions = document.createElement("div");
        transcriptActions.className = "speaker-transcript-actions";
        transcriptActions.appendChild(createTranscriptActionButton("copy", speaker, {visibleLabel:true}));
        transcriptActions.appendChild(createTranscriptActionButton("download", speaker, {visibleLabel:true}));
        transcriptSection.appendChild(transcriptHeading);
        transcriptSection.appendChild(transcriptActions);
        expanded.appendChild(transcriptSection);

        if (!reviewMode && detectedSpeakers.length > 1) {
          const correctionSection = document.createElement("section");
          correctionSection.className = "speaker-expanded-section speaker-correction-section";
          const correctionHeading = document.createElement("h3");
          correctionHeading.textContent = "Speaker correction";
          const correctionHint = document.createElement("p");
          correctionHint.textContent = "Move this Speaker's transcript into another detected Speaker.";
          correctionSection.appendChild(correctionHeading);
          correctionSection.appendChild(correctionHint);
          correctionSection.appendChild(createSpeakerMergeControls(speaker));
          expanded.appendChild(correctionSection);
        }
        if (!reviewMode) {
          const dangerSection = document.createElement("section");
          dangerSection.className = "speaker-expanded-section speaker-danger-section";
          const dangerHeading = document.createElement("h3");
          dangerHeading.textContent = "Danger zone";
          const dangerHint = document.createElement("p");
          dangerHint.textContent = panelSentenceCount > 0
            ? "Delete this Speaker and move its transcript to Unknown."
            : "Delete this empty Speaker.";
          dangerSection.appendChild(dangerHeading);
          dangerSection.appendChild(dangerHint);
          dangerSection.appendChild(createSpeakerDeleteButton(speaker));
          expanded.appendChild(dangerSection);
        }
        row.appendChild(expanded);
        if (ctx.owners.reference.pendingSpeakerNameFocusId === speaker.id) {
          ctx.owners.reference.pendingSpeakerNameFocusId = "";
          requestAnimationFrame(() => {
            nameInput.focus();
            nameInput.select();
          });
        }
      }
      speakerList.appendChild(row);
    });
    if (!ctx.owners.reference.editingSpeakerId && !ctx.owners.reference.manualSpeakerComposerOpen) {
      syncSpeakerEditor(null);
    }
    scheduleAutoRemoveEmptySpeakers();
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
      reader.onerror = () => reject(new Error("Could not read Voice sample audio."));
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
    recordReferenceButton.setAttribute("aria-label", recording ? "Stop and add Voice sample recording" : "Record Voice sample from microphone");
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
      log(`Recording a Voice sample for ${name}.`);
    } catch (error) {
      ctx.owners.reference.referenceRecordPending = false;
      stopReferenceRecording();
      log(`Voice sample recording failed: ${error.message}`);
    }
  }
  async function stopAndAddReferenceRecording() {
    await ensureSessionOwner("add a Voice sample");
    const name = selectedSpeakerReferenceName();
    const recording = stopReferenceRecording();
    if (!name) {
      log(referenceNameMissingMessage());
      return;
    }
    if (recording.seconds < 0.5) {
      log("Voice sample recording is too short.");
      return;
    }
    recordReferenceButton.disabled = true;
    try {
      const audio_b64 = encodeWavDataUrl(recording.samples, recording.sampleRate);
      const personId = ctx.owners.reference.voiceSamplePersonId || "";
      if (!personId) throw new Error("Choose a Person before recording a Voice sample.");
      const result = await post("/api/people/sample/add", {person_id: personId, label: "Microphone recording", source_type: "manual_recording", filename: `${name}.wav`, audio_b64});
      closeManualSpeakerComposerAfterReference();
      updateSpeakerState(result.speaker_state);
      referenceRecordSeconds.textContent = "0.0s";
      log(`Added a recorded Voice sample to ${name}. The original audio is retained locally.`);
    } catch (error) {
      log(`Add recorded Voice sample failed: ${error.message}`);
    } finally {
      updateReferenceRecordingControls(false);
    }
  }

  Object.assign(ctx.api, {appendSvgElement, applyCorrectionResult, arrayBufferToBase64, autoRemoveEmptySpeakerPlan, clearAutoRemoveEmptySpeakerTimer, closeManualSpeakerComposerAfterReference, commitSpeakerNameInput, createReviewReasonGroup, createSpeakerDeleteButton, createSpeakerFilterIcon, createSpeakerFilterToggle, createSpeakerFromSelectedSentences, createSpeakerLiveIndicator, createSpeakerMergeControls, createTranscriptActionButton, createTranscriptActionIcon, deleteSpeakerProfile, downloadJsonFile, encodeWavDataUrl, fileToBase64, flattenFloat32Chunks, isAutoRemovableEmptySpeaker, isSpeakerRowControl, markSelectedSentencesCorrect, mergeSpeakerInto, reassignSelectedSentences, referenceNameMissingMessage, refreshSpeakerRows, renderSpeakerPanel, runAutoRemoveEmptySpeakers, scheduleAutoRemoveEmptySpeakers, selectedSpeakerReferenceName, setEditingSpeaker, speakerGroupFileName, speakerReferenceText, speakerSentenceText, speakerSpeakingTimeText, startReferenceRecording, stopAndAddReferenceRecording, stopReferenceRecording, syncManualSpeakerComposer, syncSpeakerEditor, updateReferenceRecordingControls, writeAscii});
  ctx.activators.push(() => {
    autoRemoveEmptySpeakers.checked = storedBooleanValue(autoRemoveEmptySpeakersStorageKey, true);
    autoRemoveEmptySpeakers.addEventListener("change", () => {
      storeBooleanValue(autoRemoveEmptySpeakersStorageKey, autoRemoveEmptySpeakers.checked);
      if (autoRemoveEmptySpeakers.checked) scheduleAutoRemoveEmptySpeakers();
      else {
        clearAutoRemoveEmptySpeakerTimer();
        ctx.owners.speakers.emptySpeakerFirstSeenAt.clear();
      }
    });
    appResources.own(clearAutoRemoveEmptySpeakerTimer);
    updateSpeakerState(ctx.owners.speakers.speakerLibraryState);
  });
}
