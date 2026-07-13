export function installSessionTransport(ctx) {
  const {addReferenceSpeakerButton, allowSpeakerReassignment, audio, audioFileInput, audioUploadExtensions, chooseAudioFileButton, clearSpeakersButton, fileDropZone, fileUploadStatus, inputMode, load, loadSpeakerGroupButton, manualSpeakerName, newRunSessionButton, newSpeakerSensitivity, preset, recordReferenceButton, referenceSpeakerFile, releaseSessionButton, saveCorrectedSpeakerGroupButton, saveSpeakerGroupButton, sessionBanner, sessionBannerMessage, sessionLeaseEnabled, sessionTokenStorageKey, source, sourceModeButton, sourceModeOptionButtons, speakerRefinementUnknownCommit, speakerRefinementUnknownTentative, start, statusBox, stop, transcriptTitle, video} = ctx;
  const clearLiveSpeakerState = (...args) => ctx.api.clearLiveSpeakerState(...args), clearTranscriptSelection = (...args) => ctx.api.clearTranscriptSelection(...args), connect = (...args) => ctx.api.connect(...args), currentSessionDraftTitle = (...args) => ctx.api.currentSessionDraftTitle(...args), currentSessionSourceMetadata = (...args) => ctx.api.currentSessionSourceMetadata(...args), fetchSavedSessions = (...args) => ctx.api.fetchSavedSessions(...args), refreshMediaElements = (...args) => ctx.api.refreshMediaElements(...args), renderMeetingIntelligencePanel = (...args) => ctx.api.renderMeetingIntelligencePanel(...args), renderSavedSessions = (...args) => ctx.api.renderSavedSessions(...args), renderSpeakerPanel = (...args) => ctx.api.renderSpeakerPanel(...args), resetTranscriptDisplay = (...args) => ctx.api.resetTranscriptDisplay(...args), setBrowserStreamMode = (...args) => ctx.api.setBrowserStreamMode(...args), setMeetingIntelligenceReport = (...args) => ctx.api.setMeetingIntelligenceReport(...args), setSourceControlsDisabled = (...args) => ctx.api.setSourceControlsDisabled(...args), setSourceModeMenuOpen = (...args) => ctx.api.setSourceModeMenuOpen(...args), setState = (...args) => ctx.api.setState(...args), stopBrowserAudioCapture = (...args) => ctx.api.stopBrowserAudioCapture(...args), stopPlaybackClock = (...args) => ctx.api.stopPlaybackClock(...args), storeSessionValue = (...args) => ctx.api.storeSessionValue(...args), syncCorrectionUndoState = (...args) => ctx.api.syncCorrectionUndoState(...args), syncPresetSelection = (...args) => ctx.api.syncPresetSelection(...args), syncSourceReadyState = (...args) => ctx.api.syncSourceReadyState(...args), updateMediaMode = (...args) => ctx.api.updateMediaMode(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function log(text) {
    const div = document.createElement("div");
    div.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    statusBox.appendChild(div);
    statusBox.scrollTop = statusBox.scrollHeight;
  }
  function sessionControlsLocked() {
    if (!sessionLeaseEnabled) return false;
    return Boolean(ctx.owners.lease.sessionState && ctx.owners.lease.sessionState.active && !ctx.owners.lease.sessionState.is_owner);
  }
  function sessionOwnerActive() {
    if (!sessionLeaseEnabled) return true;
    return Boolean(ctx.owners.lease.sessionState && ctx.owners.lease.sessionState.active && ctx.owners.lease.sessionState.is_owner && ctx.owners.lease.sessionToken);
  }
  function savedSessionReviewOpen() {
    return Boolean(ctx.owners.sessions.openedSavedSessionId);
  }
  function runInProgress() {
    return Boolean(ctx.owners.lease.sessionState && ctx.owners.lease.sessionState.running) || (stop && !stop.disabled);
  }
  function updateNewRunButtonState() {
    if (!newRunSessionButton) return;
    const disabled = sessionControlsLocked() || runInProgress();
    newRunSessionButton.disabled = disabled;
    newRunSessionButton.title = disabled
      ? "New session is available when the current run is idle."
      : "Create a new session";
  }
  function setTranscriptTitleLive() {
    if (transcriptTitle) transcriptTitle.textContent = "Live transcript";
  }
  function setTranscriptTitleSaved(title) {
    if (transcriptTitle) transcriptTitle.textContent = title ? `Saved: ${title}` : "Saved transcript";
  }
  function leaveSavedSessionReview() {
    if (!ctx.owners.sessions.openedSavedSessionId) return;
    ctx.owners.sessions.openedSavedSessionId = "";
    clearTranscriptSelection();
    setMeetingIntelligenceReport(null);
    setTranscriptTitleLive();
    renderSavedSessions();
  }
  function clearSessionReviewForNewSession() {
    stopPlaybackClock();
    stopBrowserAudioCapture();
    video.pause();
    audio.pause();
    if (ctx.owners.capture.es) {
      ctx.owners.capture.es.close();
      ctx.owners.capture.es = null;
    }
    ctx.owners.sessions.openedSavedSessionId = "";
    ctx.owners.sessions.openSessionMenuId = "";
    ctx.owners.sessions.editingSessionTitleId = "";
    ctx.owners.sessions.pendingSessionTitleFocusId = "";
    ctx.owners.reference.editingSpeakerId = "";
    ctx.owners.reference.pendingSpeakerNameFocusId = "";
    ctx.owners.reference.manualSpeakerComposerOpen = false;
    ctx.owners.reference.pendingManualSpeakerNameFocus = false;
    ctx.owners.speakers.soloSpeakerIds.clear();
    ctx.owners.speakers.mutedSpeakerIds.clear();
    clearLiveSpeakerState();
    resetTranscriptDisplay();
    setMeetingIntelligenceReport(null);
    setTranscriptTitleLive();
    updateSpeakerState({group_name:"", groups:[], speakers:[]});
    updateMediaMode();
    setSourceControlsDisabled(false);
    start.disabled = false;
    stop.disabled = true;
    setState("Ready");
    renderSpeakerPanel();
    updateNewRunButtonState();
  }
  async function createDraftSavedSession(statusLabel = "New") {
    const startedAt = new Date().toISOString();
    const result = await post("/api/sessions/create", {
      title: currentSessionDraftTitle(),
      status_label: statusLabel,
      source: currentSessionSourceMetadata(startedAt),
    });
    const summary = result.session || {};
    ctx.owners.sessions.draftSavedSessionId = summary.id || "";
    ctx.owners.sessions.savedSessionFilter = "active";
    await fetchSavedSessions();
    renderMeetingIntelligencePanel();
    return ctx.owners.sessions.draftSavedSessionId;
  }
  async function ensureDraftSavedSession(statusLabel = "Started") {
    if (ctx.owners.sessions.draftSavedSessionId) return ctx.owners.sessions.draftSavedSessionId;
    return createDraftSavedSession(statusLabel);
  }
  async function createNewSession() {
    if (sessionControlsLocked()) {
      log("Session in use. You can create a new session when the demo seat is free.");
      return;
    }
    if (runInProgress()) {
      log("Stop the current run before creating a new session.");
      return;
    }
    clearSessionReviewForNewSession();
    try {
      await createDraftSavedSession("New");
      renderSavedSessions();
      log("New session created. Press Start to begin transcription.");
    } catch (error) {
      ctx.owners.sessions.draftSavedSessionId = "";
      log(`Create session failed: ${error.message}`);
    }
  }
  function sessionSecondsText(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value <= 0) return "";
    if (value >= 60) return `${Math.ceil(value / 60)} min`;
    return `${Math.ceil(value)}s`;
  }
  function sessionExpiryText(fieldName="expires_in_seconds") {
    const seconds = Number(ctx.owners.lease.sessionState && ctx.owners.lease.sessionState[fieldName]);
    if (!Number.isFinite(seconds) || seconds <= 0) return "";
    return sessionSecondsText(seconds);
  }
  function updateSessionBanner() {
    if (!sessionBanner || !sessionBannerMessage) return;
    if (!sessionLeaseEnabled) {
      sessionBanner.hidden = true;
      if (releaseSessionButton) releaseSessionButton.hidden = true;
      syncSessionControlLock();
      return;
    }
    sessionBanner.hidden = false;
    sessionBanner.classList.remove("available", "owner", "observer");
    releaseSessionButton.hidden = true;
    if (!ctx.owners.lease.sessionState || !ctx.owners.lease.sessionState.active) {
      sessionBanner.classList.add("available");
      sessionBannerMessage.textContent = "Demo seat available. Press Start or Load to take it.";
    } else if (ctx.owners.lease.sessionState.is_owner) {
      sessionBanner.classList.add("owner");
      releaseSessionButton.hidden = false;
      if (ctx.owners.lease.sessionState.running) {
        const hardLimit = sessionExpiryText("hard_expires_in_seconds");
        const heartbeatGrace = sessionSecondsText(ctx.owners.lease.sessionState.heartbeat_timeout_seconds || 45);
        sessionBannerMessage.textContent = hardLimit
          ? `You control this demo. Hard limit in ${hardLimit}; it releases when the run ends or if this browser stops checking in for ${heartbeatGrace}.`
          : `You control this demo. It releases when the run ends or if this browser stops checking in for ${heartbeatGrace}.`;
      } else if (ctx.owners.lease.sessionState.completed) {
        const expires = sessionExpiryText("completed_expires_in_seconds");
        sessionBannerMessage.textContent = expires
          ? `Run finished. Seat releases in ${expires}.`
          : "Run finished. Seat will release shortly.";
      } else {
        const expires = sessionExpiryText("idle_expires_in_seconds");
        sessionBannerMessage.textContent = expires
          ? `You control this demo seat. Start within ${expires} or release it.`
          : "You control this demo seat. Start when ready or release it.";
      }
    } else {
      sessionBanner.classList.add("observer");
      if (ctx.owners.lease.sessionState.running) {
        const hardLimit = sessionExpiryText("hard_expires_in_seconds");
        sessionBannerMessage.textContent = hardLimit
          ? `Session in use. Watching live; controls unlock when the run ends, the owner leaves, or the hard limit hits in ${hardLimit}.`
          : "Session in use. Watching live; controls unlock when the run ends or the owner leaves.";
      } else if (ctx.owners.lease.sessionState.completed) {
        const expires = sessionExpiryText("completed_expires_in_seconds");
        sessionBannerMessage.textContent = expires
          ? `Run finished. Seat releases in ${expires}.`
          : "Run finished. Seat will release shortly.";
      } else {
        const expires = sessionExpiryText("idle_expires_in_seconds");
        sessionBannerMessage.textContent = expires
          ? `Seat reserved but not running. Watching live; controls unlock if the owner does not start within ${expires}.`
          : "Session in use. Watching live; controls unlock automatically when the seat is free.";
      }
    }
    syncSessionControlLock();
  }
  function applySessionLockedDisabled(element, locked) {
    if (!element) return;
    if (locked) {
      if (!element.disabled) element.dataset.sessionDisabled = "1";
      element.disabled = true;
    } else if (element.dataset.sessionDisabled === "1") {
      element.disabled = false;
      delete element.dataset.sessionDisabled;
    }
  }
  function syncSessionControlLock() {
    const locked = sessionControlsLocked();
    [
      start,
      stop,
      load,
      source,
      preset,
      inputMode,
      sourceModeButton,
      newSpeakerSensitivity,
      speakerRefinementUnknownTentative,
      speakerRefinementUnknownCommit,
      allowSpeakerReassignment,
      clearSpeakersButton,
      addReferenceSpeakerButton,
      loadSpeakerGroupButton,
      saveSpeakerGroupButton,
      saveCorrectedSpeakerGroupButton,
      audioFileInput,
      chooseAudioFileButton,
      manualSpeakerName,
      referenceSpeakerFile,
      recordReferenceButton,
    ].forEach(element => applySessionLockedDisabled(element, locked));
    fileDropZone.classList.toggle("disabled", locked);
    fileDropZone.setAttribute("aria-disabled", locked ? "true" : "false");
    sourceModeOptionButtons.forEach(button => applySessionLockedDisabled(button, locked));
    syncSourceReadyState();
    updateNewRunButtonState();
  }
  function updateSessionState(nextState) {
    if (!nextState || typeof nextState !== "object") return;
    const wasLocked = sessionControlsLocked();
    ctx.owners.lease.sessionState = {...ctx.owners.lease.sessionState, ...nextState};
    if (!ctx.owners.lease.sessionState.active || !ctx.owners.lease.sessionState.is_owner) {
      if (!ctx.owners.lease.sessionState.active || ctx.owners.lease.sessionToken) {
        ctx.owners.lease.sessionToken = "";
        storeSessionValue(sessionTokenStorageKey, "");
      }
      stopSessionHeartbeat();
    } else if (ctx.owners.lease.sessionToken) {
      startSessionHeartbeat();
    }
    updateSessionBanner();
    updateNewRunButtonState();
    syncCorrectionUndoState();
    if (wasLocked !== sessionControlsLocked()) {
      renderSpeakerPanel();
    }
  }
  async function fetchSessionStatus() {
    if (!sessionLeaseEnabled) return {};
    const params = new URLSearchParams({client_id: ctx.owners.lease.sessionClientId});
    const response = await fetch(`/api/session/status?${params.toString()}`, {cache:"no-store"});
    const data = await response.json();
    if (data.session) updateSessionState(data.session);
    return data.session || {};
  }
  async function acquireSession() {
    if (!sessionLeaseEnabled) return true;
    const response = await fetch("/api/session/acquire", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({client_id: ctx.owners.lease.sessionClientId}),
    });
    const data = await response.json();
    if (data.session) updateSessionState(data.session);
    if (!response.ok || !data.acquired) {
      throw new Error(data.error || "Session in use. Watching live until the seat is free.");
    }
    ctx.owners.lease.sessionToken = data.session_token || "";
    storeSessionValue(sessionTokenStorageKey, ctx.owners.lease.sessionToken);
    if (data.session) updateSessionState({...data.session, is_owner:true});
    startSessionHeartbeat();
    return true;
  }
  async function ensureSessionOwner(actionLabel="control this demo") {
    if (!sessionLeaseEnabled) return true;
    if (sessionOwnerActive()) return true;
    await fetchSessionStatus().catch(() => null);
    if (sessionOwnerActive()) return true;
    if (ctx.owners.lease.sessionState && ctx.owners.lease.sessionState.active && !ctx.owners.lease.sessionState.is_owner) {
      throw new Error(`Session in use. You are watching live and cannot ${actionLabel} yet.`);
    }
    await acquireSession();
    log("Demo seat acquired.");
    return true;
  }
  async function heartbeatSession() {
    if (!sessionLeaseEnabled) return;
    if (!ctx.owners.lease.sessionToken) return;
    try {
      const response = await fetch("/api/session/heartbeat", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({client_id: ctx.owners.lease.sessionClientId, session_token: ctx.owners.lease.sessionToken}),
      });
      const data = await response.json();
      if (data.session) updateSessionState(data.session);
      if (!response.ok) {
        ctx.owners.lease.sessionToken = "";
        storeSessionValue(sessionTokenStorageKey, "");
        stopSessionHeartbeat();
      }
    } catch (_) {}
  }
  function startSessionHeartbeat() {
    if (!sessionLeaseEnabled) return;
    if (ctx.owners.lease.sessionHeartbeatTimer || !ctx.owners.lease.sessionToken) return;
    void heartbeatSession();
    ctx.owners.lease.sessionHeartbeatTimer = setInterval(heartbeatSession, 5000);
  }
  function stopSessionHeartbeat() {
    if (ctx.owners.lease.sessionHeartbeatTimer) {
      clearInterval(ctx.owners.lease.sessionHeartbeatTimer);
      ctx.owners.lease.sessionHeartbeatTimer = null;
    }
  }
  async function releaseSession(reason="released") {
    if (!sessionLeaseEnabled) return;
    if (!ctx.owners.lease.sessionToken) {
      await fetchSessionStatus().catch(() => null);
      return;
    }
    const token = ctx.owners.lease.sessionToken;
    ctx.owners.lease.sessionToken = "";
    storeSessionValue(sessionTokenStorageKey, "");
    stopSessionHeartbeat();
    const response = await fetch("/api/session/release", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({client_id: ctx.owners.lease.sessionClientId, session_token: token, reason}),
    });
    const data = await response.json();
    if (data.session) updateSessionState(data.session);
  }
  function sendSessionReleaseBeacon(reason="tab closed") {
    if (!sessionLeaseEnabled) return;
    if (!ctx.owners.lease.sessionToken || !navigator.sendBeacon) return;
    const payload = JSON.stringify({client_id: ctx.owners.lease.sessionClientId, session_token: ctx.owners.lease.sessionToken, reason});
    const body = new Blob([payload], {type:"application/json"});
    navigator.sendBeacon("/api/session/release", body);
    ctx.owners.lease.sessionToken = "";
    storeSessionValue(sessionTokenStorageKey, "");
  }
  function scheduleCompletedSessionRelease() {
    if (!sessionLeaseEnabled) return;
    if (!sessionOwnerActive()) return;
    if (ctx.owners.lease.sessionCompletionReleaseTimer) clearTimeout(ctx.owners.lease.sessionCompletionReleaseTimer);
    const delay = Math.max(1000, Number(ctx.owners.lease.sessionState.completed_release_delay_seconds || 10) * 1000);
    ctx.owners.lease.sessionCompletionReleaseTimer = setTimeout(() => {
      releaseSession("completed").catch(() => {});
    }, delay);
  }
  function startSessionStatusPolling() {
    if (!sessionLeaseEnabled) return;
    if (ctx.owners.lease.sessionStatusTimer) clearInterval(ctx.owners.lease.sessionStatusTimer);
    ctx.owners.lease.sessionStatusTimer = setInterval(() => fetchSessionStatus().catch(() => {}), 10000);
  }
  function stopSessionStatusPolling() {
    if (!ctx.owners.lease.sessionStatusTimer) return;
    clearInterval(ctx.owners.lease.sessionStatusTimer);
    ctx.owners.lease.sessionStatusTimer = null;
  }
  async function post(path, payload={}) {
    const requestPayload = {...payload};
    if (path.startsWith("/api/") && !path.startsWith("/api/session/") && !path.startsWith("/api/sessions/")) {
      requestPayload.client_id = ctx.owners.lease.sessionClientId;
      if (ctx.owners.lease.sessionToken) requestPayload.session_token = ctx.owners.lease.sessionToken;
    }
    const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(requestPayload)});
    const data = await r.json();
    if (data.session) updateSessionState(data.session);
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }
  function audioFileExtension(filename) {
    const match = String(filename || "").toLowerCase().match(/\.([a-z0-9]+)$/);
    return match ? match[1] : "";
  }
  function supportedAudioFile(file) {
    if (!file) return false;
    const extension = audioFileExtension(file.name);
    return audioUploadExtensions.has(extension) || String(file.type || "").startsWith("audio/");
  }
  function fileSizeLabel(bytes) {
    const value = Number(bytes || 0);
    if (!Number.isFinite(value) || value <= 0) return "";
    if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
    if (value >= 1024) return `${Math.round(value / 1024)} KB`;
    return `${value} B`;
  }
  function setFileUploadStatus(text, progress=null) {
    const progressText = progress === null ? "" : ` ${Math.max(0, Math.min(100, Math.round(progress)))}%`;
    fileUploadStatus.textContent = `${text || ""}${progressText}`.trim() || "WAV, MP3, M4A, FLAC, OGG";
  }
  function uploadAudioFileRequest(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/load-audio-file");
      xhr.responseType = "json";
      xhr.setRequestHeader("X-Whospeaks-Client", ctx.owners.lease.sessionClientId);
      if (ctx.owners.lease.sessionToken) xhr.setRequestHeader("X-Whospeaks-Session", ctx.owners.lease.sessionToken);
      xhr.setRequestHeader("X-Whospeaks-Filename", encodeURIComponent(file.name || "audio.wav"));
      if (file.type) xhr.setRequestHeader("Content-Type", file.type);
      xhr.upload.onprogress = event => {
        if (!event.lengthComputable) {
          setFileUploadStatus("Uploading");
          return;
        }
        setFileUploadStatus("Uploading", (event.loaded / event.total) * 100);
      };
      xhr.upload.onload = () => {
        setFileUploadStatus("Upload complete; loading audio");
        setState("Loading audio");
      };
      xhr.onerror = () => reject(new Error("Audio upload failed."));
      xhr.onload = () => {
        const data = xhr.response || {};
        if (data.session) updateSessionState(data.session);
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(new Error(data.error || xhr.statusText || "Audio upload failed."));
          return;
        }
        resolve(data);
      };
      xhr.send(file);
    });
  }
  async function loadAudioFile(file) {
    if (!file) return;
    if (!supportedAudioFile(file)) {
      log("Unsupported audio file. Choose WAV, MP3, M4A, FLAC, OGG, OPUS, WEBM, MP4, AAC, AIFF, or OGA.");
      return;
    }
    if (sessionControlsLocked()) {
      log("Session in use. Watching live until the seat is free.");
      return;
    }
    leaveSavedSessionReview();
    setSourceModeMenuOpen(false);
    try {
      await ensureSessionOwner("load audio files");
    } catch (error) {
      log(error.message);
      return;
    }
    stopPlaybackClock();
    stopBrowserAudioCapture();
    video.pause();
    audio.pause();
    inputMode.value = "file";
    setBrowserStreamMode(false);
    ctx.owners.capture.audioUploadInProgress = true;
    ctx.owners.capture.localAudioFileName = file.name || "Audio file";
    ctx.owners.capture.localAudioFileSize = Number(file.size || 0);
    source.value = "";
    start.disabled = true;
    stop.disabled = true;
    setSourceControlsDisabled(true);
    resetTranscriptDisplay();
    connect();
    setState("Uploading");
    setFileUploadStatus(fileSizeLabel(file.size) ? `Uploading ${fileSizeLabel(file.size)}` : "Uploading");
    updateMediaMode();
    try {
      const result = await uploadAudioFileRequest(file);
      if (result.speaker_state) updateSpeakerState(result.speaker_state);
      ctx.owners.capture.localAudioFileName = result.display_name || file.name || "Audio file";
      ctx.owners.capture.localAudioFileSize = Number(result.size_bytes || file.size || 0);
      source.value = result.url || "";
      syncPresetSelection("");
      refreshMediaElements(result.version);
      updateMediaMode();
      setFileUploadStatus(fileSizeLabel(ctx.owners.capture.localAudioFileSize) || "Ready");
      log(`Loaded audio file ${ctx.owners.capture.localAudioFileName}.`);
      setState("Ready");
    } catch (error) {
      source.value = "";
      setFileUploadStatus("Upload failed");
      log(`Audio file load failed: ${error.message}`);
      setState("Ready");
    } finally {
      ctx.owners.capture.audioUploadInProgress = false;
      setSourceControlsDisabled(false);
      syncSourceReadyState();
      audioFileInput.value = "";
    }
  }

  Object.assign(ctx.api, {acquireSession, applySessionLockedDisabled, audioFileExtension, clearSessionReviewForNewSession, createDraftSavedSession, createNewSession, ensureDraftSavedSession, ensureSessionOwner, fetchSessionStatus, fileSizeLabel, heartbeatSession, leaveSavedSessionReview, loadAudioFile, log, post, releaseSession, runInProgress, savedSessionReviewOpen, scheduleCompletedSessionRelease, sendSessionReleaseBeacon, sessionControlsLocked, sessionExpiryText, sessionOwnerActive, sessionSecondsText, setFileUploadStatus, setTranscriptTitleLive, setTranscriptTitleSaved, startSessionHeartbeat, startSessionStatusPolling, stopSessionHeartbeat, stopSessionStatusPolling, supportedAudioFile, syncSessionControlLock, updateNewRunButtonState, updateSessionBanner, updateSessionState, uploadAudioFileRequest});
}
