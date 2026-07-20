export function installLiveBindings(ctx) {
  const showTranscriptNewTags = document.getElementById("showTranscriptNewTags");
  const showTranscriptSentenceCount = document.getElementById("showTranscriptSentenceCount");
  const {addReferenceSpeakerButton, allowSpeakerReassignment, appResources, appStore, archiveSelectedSessionsButton, audio, audioFileInput, bulkCorrectionSpeaker, bulkMarkCorrectButton, bulkReassignButton, chooseAudioFileButton, clearSelectionButton, clearSpeakersButton, clearTranscriptButton, copyTranscriptButton, deleteSelectedSessionsButton, downloadTranscriptButton, downloadTranscriptJsonButton, expandMedia, fileDropZone, followLive, groupTranscriptTurns, inputMode, load, loadSpeakerGroupButton, manualSpeakerName, meetingIntelligenceGenerate, micGain, newRunSessionButton, newSpeakerSensitivity, preset, recordReferenceButton, referenceRecordSeconds, referenceSpeakerFile, referenceSpeakerForm, releaseSessionButton, restoreSelectedSessionsButton, reviewFilterButtons, saveCorrectedSpeakerGroupButton, saveSpeakerGroupButton, selectAllSessionsButton, sessionFilterButtons, showTranscriptProbabilities, showTranscriptReviewHints, showTranscriptSpeechRate, showTranscriptTags, showTranscriptTime, source, sourceModeButton, sourceModeMenu, sourceModeOptionButtons, sourceModeOptions, speakerGroupFile, speakerRefinementUnknownCommit, speakerRefinementUnknownTentative, speakerTabButtons, start, stop, transcriptGroupTurnsStorageKey, transcriptReviewHintsStorageKey, transcriptSearch, transcriptSettingsButton, transcriptSettingsPanel, translationDisplayModeControl, translationDisplayModeStorageKey, translationIncludeOriginalControl, translationIncludeOriginalStorageKey, translationLanguageLabelModeControl, translationLanguageLabelModeStorageKey, translationMenuButton, translationMenuPanel, translationPrimaryTargetControl, translationPrimaryTargetStorageKey, undoCorrectionButton, unselectAllSessionsButton, video, youtubeFrame} = ctx;
  const applyFallbackLiveSpeaker = (...args) => ctx.api.applyFallbackLiveSpeaker(...args), applySpeakerRefinementSettings = (...args) => ctx.api.applySpeakerRefinementSettings(...args), applySpeakerSensitivity = (...args) => ctx.api.applySpeakerSensitivity(...args), applySpeakerSensitivityIfDirty = (...args) => ctx.api.applySpeakerSensitivityIfDirty(...args), applyTranscriptDisplaySettings = (...args) => ctx.api.applyTranscriptDisplaySettings(...args), applyTranslationEvent = (...args) => ctx.api.applyTranslationEvent(...args), bulkSavedSessionAction = (...args) => ctx.api.bulkSavedSessionAction(...args), clearDisplayedTranscript = (...args) => ctx.api.clearDisplayedTranscript(...args), clearFallbackLiveSpeakerFromProbe = (...args) => ctx.api.clearFallbackLiveSpeakerFromProbe(...args), clearLiveSpeakerState = (...args) => ctx.api.clearLiveSpeakerState(...args), clearRealtimeRows = (...args) => ctx.api.clearRealtimeRows(...args), clearSavedSessionSelection = (...args) => ctx.api.clearSavedSessionSelection(...args), clearTranscriptSelection = (...args) => ctx.api.clearTranscriptSelection(...args), closeManualSpeakerComposerAfterReference = (...args) => ctx.api.closeManualSpeakerComposerAfterReference(...args), copyTranscript = (...args) => ctx.api.copyTranscript(...args), createNewSession = (...args) => ctx.api.createNewSession(...args), currentStartSessionMetadata = (...args) => ctx.api.currentStartSessionMetadata(...args), downloadJsonFile = (...args) => ctx.api.downloadJsonFile(...args), downloadTranscript = (...args) => ctx.api.downloadTranscript(...args), downloadTranscriptJson = (...args) => ctx.api.downloadTranscriptJson(...args), ensureDraftSavedSession = (...args) => ctx.api.ensureDraftSavedSession(...args), ensureSessionOwner = (...args) => ctx.api.ensureSessionOwner(...args), fetchSavedSessions = (...args) => ctx.api.fetchSavedSessions(...args), fetchSessionStatus = (...args) => ctx.api.fetchSessionStatus(...args), fileToBase64 = (...args) => ctx.api.fileToBase64(...args), flushPlaybackEnd = (...args) => ctx.api.flushPlaybackEnd(...args), generateMeetingIntelligenceReport = (...args) => ctx.api.generateMeetingIntelligenceReport(...args), leaveSavedSessionReview = (...args) => ctx.api.leaveSavedSessionReview(...args), loadAudioFile = (...args) => ctx.api.loadAudioFile(...args), log = (...args) => ctx.api.log(...args), logRejectedPlayback = (...args) => ctx.api.logRejectedPlayback(...args), markSelectedSentencesCorrect = (...args) => ctx.api.markSelectedSentencesCorrect(...args), normalizedTranslationLanguageCode = (...args) => ctx.api.normalizedTranslationLanguageCode(...args), normalizedTranslationLanguageLabelMode = (...args) => ctx.api.normalizedTranslationLanguageLabelMode(...args), post = (...args) => ctx.api.post(...args), prepareBrowserStreamSession = (...args) => ctx.api.prepareBrowserStreamSession(...args), reassignSelectedSentences = (...args) => ctx.api.reassignSelectedSentences(...args), referenceNameMissingMessage = (...args) => ctx.api.referenceNameMissingMessage(...args), reflectRuntimeStatus = (...args) => ctx.api.reflectRuntimeStatus(...args), refreshMediaElements = (...args) => ctx.api.refreshMediaElements(...args), refreshSavedSessionsAfterCompletion = (...args) => ctx.api.refreshSavedSessionsAfterCompletion(...args), refreshTranscriptGrouping = (...args) => ctx.api.refreshTranscriptGrouping(...args), refreshTranscriptVisibility = (...args) => ctx.api.refreshTranscriptVisibility(...args), refreshTranslationPresentation = (...args) => ctx.api.refreshTranslationPresentation(...args), releaseSession = (...args) => ctx.api.releaseSession(...args), renderMeetingIntelligencePanel = (...args) => ctx.api.renderMeetingIntelligencePanel(...args), renderSavedSessions = (...args) => ctx.api.renderSavedSessions(...args), renderSentence = (...args) => ctx.api.renderSentence(...args), renderSpeakerPanel = (...args) => ctx.api.renderSpeakerPanel(...args), resetTranscriptDisplay = (...args) => ctx.api.resetTranscriptDisplay(...args), scheduleCompletedSessionRelease = (...args) => ctx.api.scheduleCompletedSessionRelease(...args), scheduleSavedSessionsRefresh = (...args) => ctx.api.scheduleSavedSessionsRefresh(...args), scrollSentencesToBottom = (...args) => ctx.api.scrollSentencesToBottom(...args), selectAllSavedSessions = (...args) => ctx.api.selectAllSavedSessions(...args), selectedSpeakerReferenceName = (...args) => ctx.api.selectedSpeakerReferenceName(...args), selectedSpeakerSensitivityPreset = (...args) => ctx.api.selectedSpeakerSensitivityPreset(...args), sendSessionReleaseBeacon = (...args) => ctx.api.sendSessionReleaseBeacon(...args), sessionControlsLocked = (...args) => ctx.api.sessionControlsLocked(...args), sessionOwnerActive = (...args) => ctx.api.sessionOwnerActive(...args), setBrowserStreamMode = (...args) => ctx.api.setBrowserStreamMode(...args), setSavedSessionFilter = (...args) => ctx.api.setSavedSessionFilter(...args), setSourceControlsDisabled = (...args) => ctx.api.setSourceControlsDisabled(...args), setSourceModeMenuOpen = (...args) => ctx.api.setSourceModeMenuOpen(...args), setSpeakerTab = (...args) => ctx.api.setSpeakerTab(...args), setState = (...args) => ctx.api.setState(...args), setStreamHint = (...args) => ctx.api.setStreamHint(...args), setTranscriptReviewFilter = (...args) => ctx.api.setTranscriptReviewFilter(...args), setTranscriptSettingsOpen = (...args) => ctx.api.setTranscriptSettingsOpen(...args), setTranslationMenuOpen = (...args) => ctx.api.setTranslationMenuOpen(...args), speakerGroupFileName = (...args) => ctx.api.speakerGroupFileName(...args), startBrowserAudioCapture = (...args) => ctx.api.startBrowserAudioCapture(...args), startReferenceRecording = (...args) => ctx.api.startReferenceRecording(...args), startSessionStatusPolling = (...args) => ctx.api.startSessionStatusPolling(...args), startSynchronizedPlaybackFromGesture = (...args) => ctx.api.startSynchronizedPlaybackFromGesture(...args), stopAndAddReferenceRecording = (...args) => ctx.api.stopAndAddReferenceRecording(...args), stopBrowserAudioCapture = (...args) => ctx.api.stopBrowserAudioCapture(...args), stopBrowserLiveObservation = (...args) => ctx.api.stopBrowserLiveObservation(...args), stopBrowserLiveObservationTimerOnly = (...args) => ctx.api.stopBrowserLiveObservationTimerOnly(...args), stopPlaybackClock = (...args) => ctx.api.stopPlaybackClock(...args), stopReferenceRecording = (...args) => ctx.api.stopReferenceRecording(...args), stopSessionHeartbeat = (...args) => ctx.api.stopSessionHeartbeat(...args), stopSessionStatusPolling = (...args) => ctx.api.stopSessionStatusPolling(...args), storeBooleanValue = (...args) => ctx.api.storeBooleanValue(...args), storeSessionValue = (...args) => ctx.api.storeSessionValue(...args), syncBulkCorrectionToolbar = (...args) => ctx.api.syncBulkCorrectionToolbar(...args), syncCorrectionUndoState = (...args) => ctx.api.syncCorrectionUndoState(...args), syncPresetSelection = (...args) => ctx.api.syncPresetSelection(...args), syncSavedSessionsAutoRefresh = (...args) => ctx.api.syncSavedSessionsAutoRefresh(...args), syncSourceReadyState = (...args) => ctx.api.syncSourceReadyState(...args), unlockPlayback = (...args) => ctx.api.unlockPlayback(...args), updateMediaMode = (...args) => ctx.api.updateMediaMode(...args), updateMediaTimeline = (...args) => ctx.api.updateMediaTimeline(...args), updateMicGainLabel = (...args) => ctx.api.updateMicGainLabel(...args), updateSessionBanner = (...args) => ctx.api.updateSessionBanner(...args), updateSpeakerSensitivityLabel = (...args) => ctx.api.updateSpeakerSensitivityLabel(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function connect() {
    if (ctx.owners.capture.es) ctx.owners.capture.es.close();
    ctx.owners.capture.es = new EventSource("/events");
    ctx.owners.capture.es.addEventListener("status", e => {
      const data = JSON.parse(e.data);
      log(data.message);
      reflectRuntimeStatus(data.message);
    });
    ctx.owners.capture.es.addEventListener("speakers", e => updateSpeakerState(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("sentence", e => renderSentence(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("realtime", e => renderSentence(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("translation", e => applyTranslationEvent(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("live_speaker", e => applyFallbackLiveSpeaker(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("live_speaker_clear", e => clearFallbackLiveSpeakerFromProbe(JSON.parse(e.data)));
    ctx.owners.capture.es.addEventListener("realtime_clear", e => clearRealtimeRows(JSON.parse(e.data).generation));
    ctx.owners.capture.es.addEventListener("done", e => {
      stopPlaybackClock();
      stopBrowserAudioCapture();
      void stopBrowserLiveObservation("done");
      ctx.owners.sessions.draftSavedSessionId = "";
      setState("Stopped");
      start.disabled = false;
      stop.disabled = true;
      setSourceControlsDisabled(false);
      log(JSON.parse(e.data).message);
      scheduleCompletedSessionRelease();
      refreshSavedSessionsAfterCompletion();
    });
  }
  function disposeLiveApp() { appResources.dispose(); }

  Object.assign(ctx.api, {connect, disposeLiveApp});
  ctx.activators.push(() => {
    followLive.addEventListener("change", () => {
      ctx.owners.speakers.followLiveEnabled = followLive.checked;
      if (ctx.owners.speakers.followLiveEnabled) scrollSentencesToBottom();
    });
    transcriptSearch.addEventListener("input", () => {
      ctx.owners.speakers.transcriptSearchText = transcriptSearch.value || "";
      refreshTranscriptVisibility();
    });
    clearTranscriptButton.addEventListener("click", clearDisplayedTranscript);
    copyTranscriptButton.addEventListener("click", () => copyTranscript());
    downloadTranscriptButton.addEventListener("click", () => downloadTranscript());
    downloadTranscriptJsonButton.addEventListener("click", () => downloadTranscriptJson());
    transcriptSettingsButton.addEventListener("click", event => {
      event.stopPropagation();
      setTranscriptSettingsOpen(transcriptSettingsPanel.hidden);
    });
    transcriptSettingsPanel.addEventListener("click", event => event.stopPropagation());
    translationMenuButton.addEventListener("click", event => {
      event.stopPropagation();
      setTranslationMenuOpen(translationMenuPanel.hidden);
    });
    translationMenuPanel.addEventListener("click", event => event.stopPropagation());
    translationDisplayModeControl.addEventListener("change", () => {
      ctx.owners.translation.translationDisplayMode = ["original", "single", "all"].includes(translationDisplayModeControl.value)
        ? translationDisplayModeControl.value
        : "original";
      storeSessionValue(translationDisplayModeStorageKey, ctx.owners.translation.translationDisplayMode);
      refreshTranslationPresentation({menu:true});
    });
    translationPrimaryTargetControl.addEventListener("change", () => {
      const code = normalizedTranslationLanguageCode(translationPrimaryTargetControl.value);
      if (ctx.owners.translation.translationSelectedTargets.has(code)) {
        ctx.owners.translation.translationPrimaryTarget = code;
        storeSessionValue(translationPrimaryTargetStorageKey, ctx.owners.translation.translationPrimaryTarget);
        refreshTranslationPresentation();
      }
    });
    translationLanguageLabelModeControl.addEventListener("change", () => {
      ctx.owners.translation.translationLanguageLabelMode = normalizedTranslationLanguageLabelMode(translationLanguageLabelModeControl.value);
      storeSessionValue(translationLanguageLabelModeStorageKey, ctx.owners.translation.translationLanguageLabelMode);
      refreshTranslationPresentation();
    });
    translationIncludeOriginalControl.addEventListener("change", () => {
      storeBooleanValue(translationIncludeOriginalStorageKey, translationIncludeOriginalControl.checked);
      refreshTranslationPresentation();
    });
    [showTranscriptNewTags, showTranscriptTags, showTranscriptSentenceCount, showTranscriptTime, showTranscriptReviewHints, showTranscriptSpeechRate, showTranscriptProbabilities].forEach(control => {
      control.addEventListener("change", applyTranscriptDisplaySettings);
    });
    showTranscriptReviewHints.addEventListener("change", () => {
      storeBooleanValue(transcriptReviewHintsStorageKey, showTranscriptReviewHints.checked);
    });
    groupTranscriptTurns.addEventListener("change", () => {
      storeBooleanValue(transcriptGroupTurnsStorageKey, groupTranscriptTurns.checked);
      refreshTranscriptGrouping();
    });
    releaseSessionButton.addEventListener("click", () => {
      releaseSession("released").catch(error => log(`Release failed: ${error.message}`));
    });
    reviewFilterButtons.forEach(button => {
      button.addEventListener("click", () => setTranscriptReviewFilter(button.dataset.reviewFilter || "all"));
    });
    undoCorrectionButton.addEventListener("click", async () => {
      try {
        await ensureSessionOwner("undo corrections");
        const result = await post("/api/corrections/undo", {});
        if (result.speaker_state) updateSpeakerState(result.speaker_state);
        if (Array.isArray(result.rows)) {
          result.rows.forEach(row => renderSentence({...row, realtime:false, pending:false}));
        }
        ctx.owners.transcript.hasUndoableCorrection = false;
        syncCorrectionUndoState(false);
        scheduleSavedSessionsRefresh();
        log("Undid last correction.");
      } catch (error) {
        ctx.owners.transcript.hasUndoableCorrection = false;
        syncCorrectionUndoState(false);
        log(`Undo failed: ${error.message}`);
      }
    });
    bulkCorrectionSpeaker.addEventListener("change", syncBulkCorrectionToolbar);
    bulkReassignButton.addEventListener("click", () => reassignSelectedSentences());
    bulkMarkCorrectButton.addEventListener("click", () => markSelectedSentencesCorrect());
    clearSelectionButton.addEventListener("click", clearTranscriptSelection);
    applyTranscriptDisplaySettings();
    start.addEventListener("click", async () => {
      const useFastProcessing = ctx.api.fastProcessingEnabled();
      if (ctx.owners.capture.resumePlaybackPending && !ctx.owners.capture.browserStreamMode && !useFastProcessing) {
        start.disabled = true;
        await startSynchronizedPlaybackFromGesture();
        return;
      }
      leaveSavedSessionReview();
      let playbackUnlockResults = null;
      if (!ctx.owners.capture.browserStreamMode && !useFastProcessing) {
        playbackUnlockResults = await unlockPlayback();
      }
      try {
        await ensureSessionOwner("start a run");
      } catch (error) {
        log(error.message);
        setState("Ready");
        return;
      }
      setState("Creating session");
      try {
        await ensureDraftSavedSession("Started");
      } catch (error) {
        ctx.owners.sessions.draftSavedSessionId = "";
        log(`Create session failed: ${error.message}`);
        setState("Ready");
        return;
      }
      start.disabled = true; stop.disabled = false; setSourceControlsDisabled(true); resetTranscriptDisplay(); setState("Starting"); connect();
      if (ctx.owners.capture.browserStreamMode) {
        try {
          await applySpeakerSensitivityIfDirty();
          setState(ctx.owners.capture.captureSourceKind === "microphone" ? "Requesting mic" : (ctx.owners.capture.captureSourceKind === "mixed" ? "Requesting audio + mic" : "Requesting audio"));
          await prepareBrowserStreamSession();
          await startBrowserAudioCapture();
          setState("Warming backend");
          setStreamHint(ctx.owners.capture.captureSourceKind === "microphone" ? "Microphone capture is armed; warming backend." : (ctx.owners.capture.captureSourceKind === "mixed" ? "Mixed audio capture is armed; warming backend before transcription starts." : "Audio capture is armed; warming backend before transcription starts."));
          const result = await post("/api/start", currentStartSessionMetadata());
          if (result.speaker_state) updateSpeakerState(result.speaker_state);
          if (result.saved_session) fetchSavedSessions().catch(() => {});
        } catch (error) {
          stopBrowserAudioCapture();
          start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
          return;
        }
        setState("Capturing");
        return;
      }
      logRejectedPlayback(playbackUnlockResults || []);
      try {
        await applySpeakerSensitivityIfDirty();
        setState("Warming backend");
        log("Warming backend before playback starts. First Modal starts can take about two minutes.");
        const result = await post("/api/start", currentStartSessionMetadata());
        if (result.speaker_state) updateSpeakerState(result.speaker_state);
        if (result.saved_session) fetchSavedSessions().catch(() => {});
      } catch (error) {
        start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
        return;
      }
      if (useFastProcessing) {
        setState("Processing");
        log("Processing the complete media without real-time playback.");
        return;
      }
      setState("Starting playback");
      await startSynchronizedPlaybackFromGesture();
    });
    stop.addEventListener("click", async () => {
      if (!sessionOwnerActive()) {
        log("Only the active demo seat can stop the shared run.");
        return;
      }
      ctx.owners.capture.resumePlaybackPending = false; start.textContent = "Start transcription"; stop.disabled = true; start.disabled = false; setSourceControlsDisabled(false); setState("Stopping"); stopPlaybackClock(); stopBrowserAudioCapture(); video.pause(); audio.pause(); await stopBrowserLiveObservation("stop"); await post("/api/stop");
    });
    preset.addEventListener("change", () => {
      if (preset.value) {
        source.value = preset.value;
        if (inputMode.value === "system") {
          setBrowserStreamMode(true, source.value.trim(), "display");
        } else if (inputMode.value === "both") {
          setBrowserStreamMode(true, source.value.trim(), "mixed");
        }
      }
      updateMediaMode();
    });
    source.addEventListener("input", () => {
      syncPresetSelection(source.value);
      if (inputMode.value === "system") {
        setBrowserStreamMode(true, source.value.trim(), "display");
      } else if (inputMode.value === "both") {
        setBrowserStreamMode(true, source.value.trim(), "mixed");
      }
      updateMediaMode();
    });
    inputMode.addEventListener("change", () => {
      stopPlaybackClock();
      stopBrowserAudioCapture();
      if (inputMode.value === "microphone") {
        setBrowserStreamMode(true, "", "microphone");
        setState("Ready");
        start.disabled = false;
        log("Microphone mode selected. Press Start and allow microphone access.");
      } else if (inputMode.value === "system") {
        setBrowserStreamMode(true, source.value.trim(), "display");
        setState("Ready");
        start.disabled = false;
        log("Computer/tab audio mode selected. Press Start and share audio from a tab or window.");
      } else if (inputMode.value === "both") {
        setBrowserStreamMode(true, source.value.trim(), "mixed");
        setState("Ready");
        start.disabled = false;
        log("Computer audio + microphone mode selected. Press Start, share audio from a tab or window, and allow microphone access.");
      } else if (inputMode.value === "file") {
        setBrowserStreamMode(false);
        setState("Ready");
        log(source.value.trim().startsWith("local-audio://") ? "Audio file mode selected. Press Start to transcribe." : "Audio file mode selected. Drop or choose an audio file.");
        syncSourceReadyState();
      } else {
        setBrowserStreamMode(false);
        setState("Ready");
      }
      updateMediaMode();
    });
    sourceModeButton.addEventListener("click", event => {
      event.stopPropagation();
      setSourceModeMenuOpen(sourceModeOptions.hidden);
    });
    sourceModeOptionButtons.forEach(button => {
      button.addEventListener("click", event => {
        event.stopPropagation();
        const nextMode = button.dataset.inputMode || "youtube";
        setSourceModeMenuOpen(false);
        if (inputMode.value !== nextMode) {
          inputMode.value = nextMode;
          inputMode.dispatchEvent(new Event("change", {bubbles:true}));
        }
      });
    });
    document.addEventListener("click", event => {
      if (!sourceModeMenu.contains(event.target)) {
        setSourceModeMenuOpen(false);
      }
      if (!transcriptSettingsButton.contains(event.target) && !transcriptSettingsPanel.contains(event.target)) {
        setTranscriptSettingsOpen(false);
      }
      if (!translationMenuButton.contains(event.target) && !translationMenuPanel.contains(event.target)) {
        setTranslationMenuOpen(false);
      }
      if (ctx.owners.sessions.openSessionMenuId && !(event.target instanceof Element && event.target.closest(".session-row-menu, .session-menu-button"))) {
        ctx.owners.sessions.openSessionMenuId = "";
        renderSavedSessions();
      }
    });
    document.addEventListener("visibilitychange", syncSavedSessionsAutoRefresh);
    newSpeakerSensitivity.addEventListener("input", () => {
      ctx.owners.capture.speakerSensitivityDirty = true;
      updateSpeakerSensitivityLabel();
    });
    newSpeakerSensitivity.addEventListener("change", () => {
      applySpeakerSensitivity()
        .then(result => {
          const applied = result.new_speaker_sensitivity || selectedSpeakerSensitivityPreset();
          log(`New speaker sensitivity set to ${applied.level}. ${applied.label}.`);
        })
        .catch(error => log(`Sensitivity update failed: ${error.message}`));
    });
    [
      speakerRefinementUnknownTentative,
      speakerRefinementUnknownCommit,
      allowSpeakerReassignment,
    ].forEach(control => {
      control.addEventListener("change", () => {
        applySpeakerRefinementSettings(control)
        .then(result => {
          const settings = result.speaker_refinement || {};
          const tentative = settings.unknown_tentative !== false ? "on" : "off";
          const commit = settings.unknown_commit !== false ? "on" : "off";
          const reassignment = settings.allow_reassignment ? "on" : "off";
          log(`Speaker refinement updated: tentative UNKNOWN ${tentative}, commit UNKNOWN ${commit}, speaker changes ${reassignment}.`);
        })
        .catch(() => {});
      });
    });
    speakerTabButtons.forEach(button => {
      button.addEventListener("click", () => setSpeakerTab(button.dataset.speakerTab));
    });
    meetingIntelligenceGenerate.addEventListener("click", generateMeetingIntelligenceReport);
    sessionFilterButtons.forEach(button => {
      button.addEventListener("click", () => setSavedSessionFilter(button.dataset.sessionFilter || "active"));
    });
    selectAllSessionsButton.addEventListener("click", selectAllSavedSessions);
    unselectAllSessionsButton.addEventListener("click", clearSavedSessionSelection);
    archiveSelectedSessionsButton.addEventListener("click", () => bulkSavedSessionAction("archive"));
    restoreSelectedSessionsButton.addEventListener("click", () => bulkSavedSessionAction("restore"));
    deleteSelectedSessionsButton.addEventListener("click", () => bulkSavedSessionAction("delete"));
    newRunSessionButton.addEventListener("click", createNewSession);
    addReferenceSpeakerButton.addEventListener("click", async () => {
      const name = window.prompt("Person name:", "");
      if (!name || !name.trim()) return;
      try {
        await ensureSessionOwner("create a Person");
        const result = await post("/api/people/create", {name: name.trim()});
        updateSpeakerState(result.speaker_state);
        log(`Created Person ${name.trim()}.`);
      } catch (error) {
        log(`Add person failed: ${error.message}`);
      }
    });
    clearSpeakersButton.addEventListener("click", async () => {
      if (!ctx.owners.speakers.speakerLibraryState.speakers.length) return;
      const confirmed = window.confirm(
        "Reset live speaker detection? This removes the current detected Speakers and transcript, but keeps saved People and their Voice samples."
      );
      if (!confirmed) return;
      try {
        await ensureSessionOwner("reset live speaker detection");
      } catch (error) {
        log(error.message);
        return;
      }
      clearSpeakersButton.disabled = true;
      try {
        const result = await post("/api/speakers/clear", {});
        ctx.owners.reference.editingSpeakerId = "";
        ctx.owners.reference.pendingSpeakerNameFocusId = "";
        ctx.owners.reference.manualSpeakerComposerOpen = false;
        ctx.owners.reference.pendingManualSpeakerNameFocus = false;
        manualSpeakerName.value = "";
        ctx.owners.speakers.soloSpeakerIds.clear();
        ctx.owners.speakers.mutedSpeakerIds.clear();
        clearLiveSpeakerState();
        resetTranscriptDisplay();
        updateSpeakerState(result.speaker_state);
        clearSpeakersButton.closest("details")?.removeAttribute("open");
        log("Reset live speaker detection.");
      } catch (error) {
        log(`Reset speaker detection failed: ${error.message}`);
      } finally {
        renderSpeakerPanel();
      }
    });
    manualSpeakerName.addEventListener("keydown", event => {
      if (event.key === "Enter") {
        event.preventDefault();
      }
    });
    saveSpeakerGroupButton.addEventListener("click", async () => {
      try {
        await ensureSessionOwner("save speakers");
      } catch (error) {
        log(error.message);
        return;
      }
      const name = ctx.owners.speakers.speakerLibraryState.group_name || "speakers";
      saveSpeakerGroupButton.disabled = true;
      try {
        const result = await post("/api/speakers/export", {name});
        const group = result.group || {};
        downloadJsonFile(speakerGroupFileName(group.name || name), group);
        updateSpeakerState(result.speaker_state);
        log(`Saved speaker group ${group.name || name} to a local file.`);
      } catch (error) {
        log(`Save speakers failed: ${error.message}`);
      } finally {
        saveSpeakerGroupButton.disabled = false;
      }
    });
    saveCorrectedSpeakerGroupButton.addEventListener("click", async () => {
      try {
        await ensureSessionOwner("save corrected speakers");
      } catch (error) {
        log(error.message);
        return;
      }
      const name = ctx.owners.speakers.speakerLibraryState.group_name || "corrected-speakers";
      saveCorrectedSpeakerGroupButton.disabled = true;
      try {
        const result = await post("/api/speakers/export", {name});
        const group = result.group || {};
        group.corrected_export = true;
        downloadJsonFile(speakerGroupFileName(group.name || name), group);
        updateSpeakerState(result.speaker_state);
        log(`Saved corrected speaker group ${group.name || name} to a local file.`);
      } catch (error) {
        log(`Save corrected speakers failed: ${error.message}`);
      } finally {
        saveCorrectedSpeakerGroupButton.disabled = false;
      }
    });
    loadSpeakerGroupButton.addEventListener("click", () => {
      if (sessionControlsLocked()) {
        log("Session in use. Watching live until the seat is free.");
        return;
      }
      speakerGroupFile.value = "";
      speakerGroupFile.click();
    });
    speakerGroupFile.addEventListener("change", async () => {
      const file = speakerGroupFile.files && speakerGroupFile.files[0];
      if (!file) return;
      try {
        await ensureSessionOwner("load speakers");
      } catch (error) {
        log(error.message);
        speakerGroupFile.value = "";
        return;
      }
      loadSpeakerGroupButton.disabled = true;
      try {
        const group = JSON.parse(await file.text());
        const result = await post("/api/speakers/import", {group});
        updateSpeakerState(result.speaker_state);
        log(`Loaded speaker group ${result.speaker_state.group_name}.`);
      } catch (error) {
        log(`Load speakers failed: ${error.message}`);
      } finally {
        loadSpeakerGroupButton.disabled = false;
        speakerGroupFile.value = "";
      }
    });
    referenceSpeakerFile.addEventListener("change", () => {
      if (referenceSpeakerFile.files && referenceSpeakerFile.files[0]) {
        referenceSpeakerForm.requestSubmit();
      }
    });
    referenceSpeakerForm.addEventListener("submit", async event => {
      event.preventDefault();
      try {
        await ensureSessionOwner("add a Voice sample");
      } catch (error) {
        log(error.message);
        return;
      }
      if (ctx.owners.reference.referenceRecordStream || ctx.owners.reference.referenceRecordPending) {
        log("Stop the Voice sample recording first.");
        return;
      }
      const name = selectedSpeakerReferenceName();
      const personId = ctx.owners.reference.voiceSamplePersonId || "";
      const file = referenceSpeakerFile.files && referenceSpeakerFile.files[0];
      if (!personId || !file) {
        log(personId ? "Choose a Voice sample audio file first." : "Choose a Person before adding a Voice sample.");
        return;
      }
      const submit = referenceSpeakerForm.querySelector("button[type='submit']");
      if (submit) submit.disabled = true;
      referenceSpeakerFile.disabled = true;
      try {
        const audio_b64 = await fileToBase64(file);
        const result = await post("/api/people/sample/add", {person_id: personId, label: file.name.replace(/\.[^.]+$/, ""), filename: file.name, audio_b64});
        closeManualSpeakerComposerAfterReference();
        updateSpeakerState(result.speaker_state);
        referenceSpeakerFile.value = "";
        log(`Added a Voice sample to ${name}. The original audio is retained locally.`);
      } catch (error) {
        log(`Add Voice sample failed: ${error.message}`);
      } finally {
        if (submit) submit.disabled = false;
        referenceSpeakerFile.disabled = false;
      }
    });
    recordReferenceButton.addEventListener("click", async () => {
      if (sessionControlsLocked()) {
        log("Session in use. Watching live until the seat is free.");
        return;
      }
      if (ctx.owners.reference.referenceRecordStream || ctx.owners.reference.referenceRecordPending) {
        stopAndAddReferenceRecording().catch(error => log(`Add recorded Voice sample failed: ${error.message}`));
        return;
      }
      try {
        await ensureSessionOwner("record a Voice sample");
      } catch (error) {
        log(error.message);
        return;
      }
      startReferenceRecording().catch(error => log(`Voice sample recording failed: ${error.message}`));
    });
    chooseAudioFileButton.addEventListener("click", event => {
      event.stopPropagation();
      if (sessionControlsLocked() || ctx.owners.capture.audioUploadInProgress) {
        log(ctx.owners.capture.audioUploadInProgress ? "Audio upload already in progress." : "Session in use. Watching live until the seat is free.");
        return;
      }
      audioFileInput.click();
    });
    audioFileInput.addEventListener("change", () => {
      const file = audioFileInput.files && audioFileInput.files[0];
      if (file) loadAudioFile(file).catch(error => log(`Audio file load failed: ${error.message}`));
    });
    fileDropZone.addEventListener("click", event => {
      if (event.target instanceof Element && event.target.closest("button")) return;
      if (sessionControlsLocked() || ctx.owners.capture.audioUploadInProgress) return;
      audioFileInput.click();
    });
    fileDropZone.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      if (sessionControlsLocked() || ctx.owners.capture.audioUploadInProgress) return;
      audioFileInput.click();
    });
    ["dragenter", "dragover"].forEach(eventName => {
      fileDropZone.addEventListener(eventName, event => {
        event.preventDefault();
        if (!sessionControlsLocked() && !ctx.owners.capture.audioUploadInProgress) {
          fileDropZone.classList.add("dragover");
        }
      });
    });
    ["dragleave", "drop"].forEach(eventName => {
      fileDropZone.addEventListener(eventName, event => {
        event.preventDefault();
        fileDropZone.classList.remove("dragover");
      });
    });
    fileDropZone.addEventListener("drop", event => {
      if (sessionControlsLocked() || ctx.owners.capture.audioUploadInProgress) return;
      const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) loadAudioFile(file).catch(error => log(`Audio file load failed: ${error.message}`));
    });
    load.addEventListener("click", async () => {
      leaveSavedSessionReview();
      setSourceModeMenuOpen(false);
      const url = source.value.trim();
      if (inputMode.value === "file") {
        if (sessionControlsLocked()) {
          log("Session in use. Watching live until the seat is free.");
          return;
        }
        audioFileInput.click();
        return;
      }
      if (inputMode.value === "microphone") {
        stopPlaybackClock();
        stopBrowserAudioCapture();
        video.pause();
        audio.pause();
        resetTranscriptDisplay();
        connect();
        setBrowserStreamMode(true, "", "microphone");
        log("Microphone mode ready. Press Start and allow microphone access.");
        setState("Ready");
        start.disabled = false;
        stop.disabled = true;
        updateMediaMode();
        return;
      }
      if (inputMode.value === "system") {
        stopPlaybackClock();
        stopBrowserAudioCapture();
        video.pause();
        audio.pause();
        resetTranscriptDisplay();
        connect();
        setBrowserStreamMode(true, url, "display");
        log("Computer/tab audio mode ready. Press Start and share audio from a tab or window.");
        setState("Ready");
        start.disabled = false;
        stop.disabled = true;
        updateMediaMode();
        return;
      }
      if (inputMode.value === "both") {
        stopPlaybackClock();
        stopBrowserAudioCapture();
        video.pause();
        audio.pause();
        resetTranscriptDisplay();
        connect();
        setBrowserStreamMode(true, url, "mixed");
        log("Computer audio + microphone mode ready. Press Start, share audio from a tab or window, and allow microphone access.");
        setState("Ready");
        start.disabled = false;
        stop.disabled = true;
        updateMediaMode();
        return;
      }
      if (!url) {
        log("Enter a YouTube URL first.");
        return;
      }
      try {
        await ensureSessionOwner("load media");
      } catch (error) {
        log(error.message);
        return;
      }
      stopPlaybackClock();
      stopBrowserAudioCapture();
      video.pause();
      audio.pause();
      start.disabled = true;
      stop.disabled = true;
      setSourceControlsDisabled(true);
      setState("Loading");
      resetTranscriptDisplay();
      connect();
      try {
        const media = await post("/api/load-url", {url});
        if (media.speaker_state) updateSpeakerState(media.speaker_state);
        source.value = media.url;
        syncPresetSelection(media.url);
        refreshMediaElements(media.version);
        updateMediaMode();
        log(`Loaded ${media.video_id}.`);
        setState("Ready");
        start.disabled = false;
      } catch (error) {
        log(`Load failed: ${error.message}`);
        try {
          const fallback = await post("/api/browser-stream", {url});
          if (fallback.speaker_state) updateSpeakerState(fallback.speaker_state);
          source.value = fallback.url;
          syncPresetSelection(fallback.url);
          setBrowserStreamMode(true, fallback.url, "display");
          updateMediaMode();
          log("Switched to browser audio mode. Open/play the embedded YouTube video, press Start, then share this tab with audio.");
          setState("Ready");
          start.disabled = false;
        } catch (fallbackError) {
          log(`Browser audio fallback failed: ${fallbackError.message}`);
          setState("Ready");
          start.disabled = false;
        }
      } finally {
        setSourceControlsDisabled(false);
      }
    });
    ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended"].forEach(eventName => {
      video.addEventListener(eventName, updateMediaTimeline);
      audio.addEventListener(eventName, updateMediaTimeline);
    });
    const mediaTimelineTimer = setInterval(updateMediaTimeline, 250);
    appResources.own(() => clearInterval(mediaTimelineTimer));
    expandMedia.addEventListener("click", () => {
      const youtubeStream = youtubeFrame.parentElement;
      const target = ctx.owners.capture.browserStreamMode && youtubeStream && !youtubeStream.classList.contains("empty") ? youtubeStream : video;
      const request = target.requestFullscreen || target.webkitRequestFullscreen || target.msRequestFullscreen;
      if (request) request.call(target);
    });
    micGain.addEventListener("input", updateMicGainLabel);
    audio.addEventListener("ended", flushPlaybackEnd);
    appResources.own(() => {
      sendSessionReleaseBeacon("tab closed");
      stopSessionHeartbeat();
      stopSessionStatusPolling();
      stopBrowserAudioCapture();
      stopBrowserLiveObservationTimerOnly();
      if (ctx.owners.reference.referenceRecordStream) stopReferenceRecording();
      if (ctx.owners.capture.es) { ctx.owners.capture.es.close(); ctx.owners.capture.es = null; }
      for (const timer of [
        ctx.owners.capture.playbackTimer,
        ctx.owners.transcript.fallbackLiveSpeakerExpiryTimer,
        ctx.owners.transcript.fallbackLiveSpeakerClearTimer,
        ctx.owners.transcript.transcriptLiveSpeakerExpiryTimer,
        ctx.owners.transcript.browserLiveObservationTimer,
        ctx.owners.reference.referenceRecordTimer,
        ctx.owners.lease.sessionCompletionReleaseTimer,
        ctx.owners.sessions.savedSessionRefreshTimer,
        ctx.owners.sessions.savedSessionAutoRefreshTimer,
        ctx.owners.translation.translationConfigureTimer,
      ]) {
        if (timer) clearTimeout(timer);
      }
      appStore.dispose();
    });
    window.addEventListener("pagehide", disposeLiveApp, {once: true});
    window.addEventListener("beforeunload", disposeLiveApp, {once: true});
    updateSessionBanner();
    renderMeetingIntelligencePanel();
    fetchSavedSessions().catch(() => {});
    syncSavedSessionsAutoRefresh();
    fetchSessionStatus().catch(() => {});
    startSessionStatusPolling();
  });
}
