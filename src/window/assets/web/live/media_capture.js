export function installMediaCapture(ctx) {
  const {allowSpeakerReassignment, audio, audioFileInput, captureDescription, captureLevelFill, captureLevelText, capturePreRollSeconds, captureStartRmsThreshold, captureTitle, chooseAudioFileButton, fastProcessing, fastProcessingControl, fastProcessingStorageKey, fileDropTitle, fileDropZone, filePreviewName, groupTranscriptTurns, initialSource, inputMode, languageConfig, languageFlag, languageName, languageSummary, load, mediaCard, mediaCurrentTime, mediaDuration, mediaTime, micGain, micGainValue, newSpeakerSensitivity, newSpeakerSensitivityLabel, preset, presetVideos, sessionClientIdStorageKey, sessionTokenStorageKey, showTranscriptReviewHints, source, sourceKind, sourceModeButton, sourceModeMenu, sourceModeOptionButtons, sourceModeOptions, sourceTitle, speakerCountLabel, speakerCountNumber, speakerRefinementConfig, speakerRefinementUnknownCommit, speakerRefinementUnknownTentative, speakerSensitivityConfig, start, state, stop, streamHint, targetCaptureSampleRate, timelineFill, timelineThumb, transcriptGroupTurnsStorageKey, transcriptReviewHintsStorageKey, video, youtubeFrame} = ctx;
  const connect = (...args) => ctx.api.connect(...args), ensureSessionOwner = (...args) => ctx.api.ensureSessionOwner(...args), initializeTranslationControls = (...args) => ctx.api.initializeTranslationControls(...args), log = (...args) => ctx.api.log(...args), mediaSeconds = (...args) => ctx.api.mediaSeconds(...args), post = (...args) => ctx.api.post(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), sessionControlsLocked = (...args) => ctx.api.sessionControlsLocked(...args), syncSessionControlLock = (...args) => ctx.api.syncSessionControlLock(...args), updateNewRunButtonState = (...args) => ctx.api.updateNewRunButtonState(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function randomSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }
  function storedSessionValue(key) {
    try {
      return localStorage.getItem(key) || "";
    } catch (_) {
      return "";
    }
  }
  function storeSessionValue(key, value) {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch (_) {}
  }
  function storedBooleanValue(key, fallback = false) {
    try {
      const value = localStorage.getItem(key);
      if (value === null) return Boolean(fallback);
      return value === "true";
    } catch (_) {
      return Boolean(fallback);
    }
  }
  function storeBooleanValue(key, value) {
    try {
      localStorage.setItem(key, value ? "true" : "false");
    } catch (_) {}
  }
  function initializeSessionIdentity() {
    ctx.owners.lease.sessionClientId = storedSessionValue(sessionClientIdStorageKey);
    if (!ctx.owners.lease.sessionClientId) {
      ctx.owners.lease.sessionClientId = randomSessionId();
      storeSessionValue(sessionClientIdStorageKey, ctx.owners.lease.sessionClientId);
    }
    ctx.owners.lease.sessionToken = storedSessionValue(sessionTokenStorageKey);
  }
  function setState(text) { state.textContent = text; }
  function updateLanguageIndicator() {
    const code = String(languageConfig.code || "").trim();
    const name = String(languageConfig.name || code || "Language").trim();
    languageName.textContent = name;
    languageFlag.src = String(languageConfig.flag_url || "");
    languageFlag.alt = "";
    languageSummary.title = `Transcription language: ${name}${code ? ` (${code})` : ""}`;
    languageSummary.setAttribute("aria-label", languageSummary.title);
  }
  function updateSpeakerCount() {
    const count = Array.isArray(ctx.owners.speakers.speakerLibraryState.speakers) ? ctx.owners.speakers.speakerLibraryState.speakers.length : 0;
    speakerCountNumber.textContent = String(count);
    speakerCountLabel.textContent = `${count === 1 ? "speaker" : "speakers"} found`;
  }
  function setStreamHint(text) {
    if (streamHint) streamHint.textContent = text || "";
  }
  function reflectRuntimeStatus(message) {
    const text = String(message || "");
    const lower = text.toLowerCase();
    let nextState = "";
    if (lower.includes("start requested")) nextState = "Preparing";
    else if (lower.includes("loading transcription model") || lower.includes("importing faster-whisper") || lower.includes("loading faster-whisper") || lower.includes("checking remote asr")) nextState = "Loading ASR";
    else if (lower.includes("loading sentence splitter") || lower.includes("initializing stream2sentence")) nextState = "Loading splitter";
    else if (lower.includes("loading speaker embedding model") || lower.includes("warming speaker embedding model") || lower.includes("refreshing speaker embedding model")) nextState = "Warming embeddings";
    else if (lower.includes("loading silero onnx vad")) nextState = "Loading VAD";
    else if (lower.includes("asr warmup transcription")) nextState = "Warming ASR";
    else if (lower.includes("loading realtime preview")) nextState = "Loading preview";
    else if (lower.includes("fast processing started") || lower.includes("fast processing worker started")) nextState = "Processing";
    else if (lower.includes("fast asr")) nextState = "Transcribing";
    else if (lower.includes("fast processing speaker embeddings") || (lower.includes("queued") && lower.includes("sentence embeddings"))) nextState = "Diarizing";
    else if (lower.includes("synchronized playback can begin") || lower.includes("growing-window transcription started") || lower.includes("realtime preview started")) nextState = ctx.owners.capture.browserStreamMode ? "Capturing" : "Playing";
    else if (lower.includes("transcribing window")) nextState = "Transcribing";
    else if (lower.includes("queued speaker embedding") || lower.includes("embedded sentence")) nextState = "Diarizing";
    if (nextState) setState(nextState);

    if (!ctx.owners.capture.browserStreamMode) return;
    if (lower.includes("waiting for audible input")) {
      if (ctx.owners.capture.captureSourceKind === "microphone") {
        setStreamHint("Microphone capture is armed. Speak into the selected microphone.");
      } else if (ctx.owners.capture.captureSourceKind === "mixed") {
        setStreamHint("Mixed capture is armed. Play the shared source or speak into the microphone.");
      } else {
        setStreamHint("Audio capture is armed. Play the video and make sure the shared source includes audio.");
      }
    } else if (lower.includes("detected audible input")) {
      setStreamHint("Audio detected. Transcription appears after the first completed window.");
    } else if (lower.includes("growing-window transcription started") || lower.includes("realtime preview started")) {
      setStreamHint("Capturing audio and transcribing.");
    }
  }
  function setSourceControlsDisabled(disabled) {
    load.disabled = disabled;
    source.disabled = disabled;
    preset.disabled = disabled;
    inputMode.disabled = disabled;
    sourceModeButton.disabled = disabled;
    audioFileInput.disabled = disabled;
    chooseAudioFileButton.disabled = disabled;
    fastProcessing.disabled = disabled;
    fastProcessingControl.classList.toggle("disabled", disabled);
    fileDropZone.classList.toggle("disabled", disabled);
    fileDropZone.setAttribute("aria-disabled", disabled ? "true" : "false");
    sourceModeOptionButtons.forEach(button => { button.disabled = disabled; });
    if (!disabled) syncSessionControlLock();
    syncFastProcessingControls();
    syncSourceReadyState();
    updateNewRunButtonState();
  }
  function normalizeUrl(url) {
    return String(url || "").trim();
  }
  function syncPresetSelection(url) {
    const normalized = normalizeUrl(url);
    const match = presetVideos.find(item => normalizeUrl(item.url) === normalized);
    preset.value = match ? match.url : "";
  }
  function presetForUrl(url) {
    const normalized = normalizeUrl(url);
    return presetVideos.find(item => normalizeUrl(item.url) === normalized) || null;
  }
  function sourceTitleForUrl(url) {
    const match = presetForUrl(url);
    if (match) return match.title;
    const text = normalizeUrl(url);
    if (!text) return "Custom source";
    if (text.startsWith("local-audio://")) {
      try {
        const parsed = new URL(text);
        const name = decodeURIComponent(parsed.pathname.replace(/^\/+/, ""));
        return name || "Audio file";
      } catch (_) {
        return "Audio file";
      }
    }
    try {
      const parsed = new URL(text);
      return parsed.hostname.replace(/^www\./, "") || "Custom source";
    } catch (_) {
      return text.length > 64 ? `${text.slice(0, 61)}...` : text;
    }
  }
  function currentStartSessionMetadata() {
    const match = presetForUrl(source.value);
    const mode = inputMode.value || "youtube";
    return {
      source_title: mode === "file" ? ctx.owners.capture.localAudioFileName : (match && !ctx.owners.capture.browserStreamMode ? match.title : ""),
      session_id: ctx.owners.sessions.draftSavedSessionId,
      processing_mode: fastProcessingEnabled() ? "fast" : "playback",
    };
  }
  function currentSessionSourceMetadata(startedAt = new Date().toISOString()) {
    const mode = inputMode.value || "youtube";
    const url = source.value.trim();
    const match = presetForUrl(url);
    if (mode === "microphone") {
      return {
        url: "microphone://local",
        video_id: "microphone",
        capture_mode: "microphone",
        streaming_audio: true,
        title: "Microphone recording",
        started_at: startedAt,
      };
    }
    if (mode === "system") {
      return {
        url: url || "browser-stream://display",
        video_id: "browser-stream",
        capture_mode: "browser-stream",
        streaming_audio: true,
        title: "Browser audio recording",
        started_at: startedAt,
      };
    }
    if (mode === "both") {
      return {
        url: url || "mixed-audio://local",
        video_id: "mixed-audio",
        capture_mode: "mixed",
        streaming_audio: true,
        title: "Computer audio + microphone recording",
        started_at: startedAt,
      };
    }
    if (mode === "file") {
      return {
        url: url || "local-audio://pending",
        video_id: "",
        capture_mode: "audio-file",
        title: ctx.owners.capture.localAudioFileName || "Audio file",
        size_bytes: ctx.owners.capture.localAudioFileSize,
        processing_mode: fastProcessingEnabled() ? "fast" : "playback",
        started_at: startedAt,
      };
    }
    return {
      url,
      video_id: "",
      capture_mode: "youtube",
      title: match ? match.title : "",
      processing_mode: fastProcessingEnabled() ? "fast" : "playback",
      started_at: startedAt,
    };
  }
  function currentSessionDraftTitle() {
    const mode = inputMode.value || "youtube";
    if (mode === "youtube") {
      const match = presetForUrl(source.value);
      return match ? match.title : "";
    }
    if (mode === "file") return ctx.owners.capture.localAudioFileName || "Audio file";
    return "";
  }
  function updateMediaMode() {
    const mode = inputMode.value || "youtube";
    mediaCard.classList.toggle("mode-youtube", mode === "youtube");
    mediaCard.classList.toggle("mode-file", mode === "file");
    mediaCard.classList.toggle("mode-microphone", mode === "microphone");
    mediaCard.classList.toggle("mode-system", mode === "system");
    mediaCard.classList.toggle("mode-both", mode === "both");
    sourceModeOptionButtons.forEach(button => {
      const active = button.dataset.inputMode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    if (mode === "microphone") {
      sourceKind.textContent = "Microphone";
      sourceTitle.textContent = "Local microphone input";
      mediaTime.textContent = "Live input";
      captureTitle.textContent = "Microphone input";
      captureDescription.textContent = "Input level appears after capture starts.";
    } else if (mode === "system") {
      sourceKind.textContent = "Computer audio";
      sourceTitle.textContent = "Shared tab or system audio";
      mediaTime.textContent = "Live input";
      captureTitle.textContent = "Computer audio";
      captureDescription.textContent = "Shared audio level appears after capture starts.";
    } else if (mode === "both") {
      sourceKind.textContent = "Computer audio + microphone";
      sourceTitle.textContent = "Shared audio mixed with local microphone";
      mediaTime.textContent = "Live input";
      captureTitle.textContent = "Computer audio + microphone";
      captureDescription.textContent = "Mixed input level appears after capture starts.";
    } else if (mode === "file") {
      const hasFile = source.value.trim().startsWith("local-audio://");
      const name = ctx.owners.capture.localAudioFileName || (hasFile ? sourceTitleForUrl(source.value) : "No audio file selected");
      sourceKind.textContent = "Audio/video file";
      sourceTitle.textContent = name;
      fileDropTitle.textContent = name;
      filePreviewName.textContent = name;
      if (ctx.owners.capture.audioUploadInProgress) {
        mediaTime.textContent = "Uploading";
      } else if (hasFile) {
        updateMediaTimeline();
      } else {
        mediaTime.textContent = "Choose an audio file";
        mediaCurrentTime.textContent = "00:00";
        mediaDuration.textContent = "00:00";
        timelineFill.style.width = "0%";
        timelineThumb.style.left = "0%";
      }
    } else {
      sourceKind.textContent = "YouTube";
      sourceTitle.textContent = sourceTitleForUrl(source.value);
      updateMediaTimeline();
    }
    syncFastProcessingControls();
    syncSourceReadyState();
  }
  function fastProcessingEnabled() {
    const mode = inputMode.value || "youtube";
    return !ctx.owners.capture.browserStreamMode
      && (mode === "youtube" || mode === "file")
      && Boolean(fastProcessing.checked);
  }
  function syncFastProcessingControls() {
    const mode = inputMode.value || "youtube";
    const available = !ctx.owners.capture.browserStreamMode && (mode === "youtube" || mode === "file");
    fastProcessingControl.hidden = !available;
    if (!ctx.owners.capture.resumePlaybackPending && stop.disabled) {
      start.textContent = available && fastProcessing.checked
        ? (mode === "file" ? "Process file" : "Process video")
        : "Start transcription";
    }
  }
  function syncSourceReadyState() {
    if ((inputMode.value || "youtube") !== "file") return;
    const loaded = source.value.trim().startsWith("local-audio://");
    if (stop && !stop.disabled) return;
    start.disabled = ctx.owners.capture.audioUploadInProgress || !loaded || sessionControlsLocked() || savedSessionReviewOpen();
  }
  function setSourceModeMenuOpen(open) {
    sourceModeMenu.classList.toggle("open", open);
    sourceModeOptions.hidden = !open;
    sourceModeButton.setAttribute("aria-expanded", open ? "true" : "false");
  }
  function clockLabel(value) {
    const total = Math.max(0, Math.floor(Number(value || 0)));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  function mediaDurationSeconds() {
    const audioDuration = Number(audio.duration || 0);
    const videoDuration = Number(video.duration || 0);
    if (Number.isFinite(audioDuration) && audioDuration > 0) return audioDuration;
    if (Number.isFinite(videoDuration) && videoDuration > 0) return videoDuration;
    return 0;
  }
  function currentMediaSeconds() {
    return Math.max(mediaSeconds(audio), mediaSeconds(video));
  }
  function updateMediaTimeline() {
    const mode = inputMode.value || "youtube";
    if (mode === "microphone" || mode === "system" || mode === "both") {
      mediaCurrentTime.textContent = "00:00";
      mediaDuration.textContent = "00:00";
      timelineFill.style.width = "0%";
      timelineThumb.style.left = "0%";
      return;
    }
    const duration = mediaDurationSeconds();
    const current = duration > 0 ? Math.min(currentMediaSeconds(), duration) : currentMediaSeconds();
    const percent = duration > 0 ? Math.max(0, Math.min(100, (current / duration) * 100)) : 0;
    const currentLabel = clockLabel(current);
    const durationLabel = clockLabel(duration);
    mediaCurrentTime.textContent = currentLabel;
    mediaDuration.textContent = durationLabel;
    mediaTime.textContent = `${currentLabel} / ${durationLabel}`;
    timelineFill.style.width = `${percent}%`;
    timelineThumb.style.left = `${percent}%`;
  }
  function updateMicGainLabel() {
    const gain = microphoneGainValue();
    micGainValue.textContent = `${gain.toFixed(2)}x`;
    if (ctx.owners.capture.captureMicGainNode) ctx.owners.capture.captureMicGainNode.gain.value = gain;
  }
  function microphoneGainValue() {
    const gain = Number(micGain.value || 1);
    return Number.isFinite(gain) ? Math.max(0, Math.min(2, gain)) : 1;
  }
  function captureGainValue() {
    if ((inputMode.value || "youtube") !== "microphone") return 1;
    return microphoneGainValue();
  }
  function copyCaptureSamples(input) {
    const gain = captureGainValue();
    const copy = new Float32Array(input.length);
    if (gain === 1) {
      copy.set(input);
      return copy;
    }
    for (let index = 0; index < input.length; index += 1) {
      copy[index] = Math.max(-1, Math.min(1, input[index] * gain));
    }
    return copy;
  }
  function setCaptureLevel(value) {
    const rmsValue = Math.max(0, Number(value || 0));
    const level = Math.max(0, Math.min(1, Math.sqrt(rmsValue) * 4.2));
    const percent = Math.round(level * 100);
    captureLevelFill.style.width = `${percent}%`;
    captureLevelText.textContent = `${percent}%`;
  }
  function populatePresetVideos() {
    preset.textContent = "";
    const custom = document.createElement("option");
    custom.value = "";
    custom.textContent = "Custom URL";
    preset.appendChild(custom);
    for (const item of presetVideos) {
      const option = document.createElement("option");
      option.value = item.url;
      option.textContent = item.title;
      preset.appendChild(option);
    }
  }
  function selectedSpeakerSensitivityPreset() {
    const level = Number(newSpeakerSensitivity.value || speakerSensitivityConfig.selected || 3);
    return speakerSensitivityConfig.presets.find(item => Number(item.level) === level) || speakerSensitivityConfig.presets[2];
  }
  function updateSpeakerSensitivityLabel() {
    const preset = selectedSpeakerSensitivityPreset();
    newSpeakerSensitivityLabel.textContent = `${preset.level}. ${preset.label}`;
  }
  async function applySpeakerSensitivity() {
    await ensureSessionOwner("change speaker settings");
    const preset = selectedSpeakerSensitivityPreset();
    const result = await post("/api/settings", {new_speaker_sensitivity: preset.level});
    const applied = result.new_speaker_sensitivity || preset;
    if (applied.level && Number(newSpeakerSensitivity.value) !== Number(applied.level)) {
      newSpeakerSensitivity.value = applied.level;
    }
    ctx.owners.capture.speakerSensitivityDirty = false;
    updateSpeakerSensitivityLabel();
    return result;
  }
  async function applySpeakerSensitivityIfDirty() {
    if (!ctx.owners.capture.speakerSensitivityDirty) return null;
    return applySpeakerSensitivity();
  }
  function syncSpeakerRefinementSettings(settings) {
    if (!settings || typeof settings !== "object") return;
    speakerRefinementUnknownTentative.checked = settings.unknown_tentative !== false;
    speakerRefinementUnknownCommit.checked = settings.unknown_commit !== false;
    allowSpeakerReassignment.checked = Boolean(settings.allow_reassignment);
  }
  function speakerRefinementPayload() {
    return {
      speaker_refinement_unknown_tentative: speakerRefinementUnknownTentative.checked,
      speaker_refinement_unknown_commit: speakerRefinementUnknownCommit.checked,
      allow_speaker_reassignment: allowSpeakerReassignment.checked,
    };
  }
  async function applySpeakerRefinementSettings(changedControl) {
    await ensureSessionOwner("change speaker settings");
    const requested = changedControl ? changedControl.checked : null;
    try {
      const result = await post("/api/settings", speakerRefinementPayload());
      syncSpeakerRefinementSettings(result.speaker_refinement);
      return result;
    } catch (error) {
      if (changedControl) changedControl.checked = !requested;
      log(`Speaker refinement setting failed: ${error.message}`);
      throw error;
    }
  }
  function extractYouTubeId(url) {
    const text = String(url || "");
    try {
      const parsed = new URL(text);
      if (parsed.hostname.includes("youtu.be")) return parsed.pathname.replace(/^\/+/, "").split("/")[0] || "";
      if (parsed.searchParams.get("v")) return parsed.searchParams.get("v") || "";
      const parts = parsed.pathname.split("/").filter(Boolean);
      const marker = parts.findIndex(part => ["embed", "shorts", "live"].includes(part));
      if (marker >= 0 && parts[marker + 1]) return parts[marker + 1];
    } catch (_) {}
    const match = text.match(/[?&]v=([^&]+)/) || text.match(/youtu\.be\/([^?&#/]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }
  function youtubeEmbedUrl(url) {
    const id = extractYouTubeId(url);
    return id ? `https://www.youtube.com/embed/${encodeURIComponent(id)}?enablejsapi=1&rel=0` : "";
  }
  function setBrowserStreamMode(enabled, url="", sourceKind="display") {
    ctx.owners.capture.browserStreamMode = Boolean(enabled);
    ctx.owners.capture.captureSourceKind = sourceKind || "display";
    ctx.owners.capture.browserStreamPrepared = false;
    ctx.owners.capture.browserStreamPreparedUrl = "";
    document.querySelector(".app").classList.toggle("browser-stream", ctx.owners.capture.browserStreamMode);
    if (ctx.owners.capture.browserStreamMode) {
      const embed = youtubeEmbedUrl(url);
      youtubeFrame.src = embed;
      youtubeFrame.parentElement.classList.toggle("empty", !embed);
      if (streamHint) {
        if (ctx.owners.capture.captureSourceKind === "microphone") {
          setStreamHint("Microphone mode. Press Start, allow microphone access, then speak.");
        } else if (ctx.owners.capture.captureSourceKind === "mixed") {
          setStreamHint("Computer audio + microphone mode. Press Start, share a tab or window with audio, then allow microphone access.");
        } else if (embed) {
          setStreamHint("Play the video, press Start, then share this tab with audio.");
        } else {
          setStreamHint("Computer/tab audio mode. Press Start, choose a browser tab or window, and enable audio sharing.");
        }
      }
    } else {
      youtubeFrame.src = "";
      youtubeFrame.parentElement.classList.remove("empty");
      setStreamHint("");
    }
    updateMediaMode();
  }
  function browserStreamSourceUrl() {
    if (ctx.owners.capture.captureSourceKind === "microphone") return "microphone://local";
    if (ctx.owners.capture.captureSourceKind === "mixed") return "mixed-audio://local";
    const url = source.value.trim();
    return url || "system-audio://local";
  }
  async function prepareBrowserStreamSession() {
    await ensureSessionOwner("prepare browser audio");
    const url = browserStreamSourceUrl();
    if (ctx.owners.capture.browserStreamPrepared && ctx.owners.capture.browserStreamPreparedUrl === url) return;
    const result = await post("/api/browser-stream", {url});
    if (result.speaker_state) updateSpeakerState(result.speaker_state);
    ctx.owners.capture.browserStreamPrepared = true;
    ctx.owners.capture.browserStreamPreparedUrl = url;
    ctx.owners.capture.mediaVersion = Number(result.version || ctx.owners.capture.mediaVersion || Date.now());
    log(`Browser audio stream prepared for ${result.video_id}.`);
  }
  function initializeInputModeFromSource() {
    const value = source.value.trim();
    if (value.startsWith("local-audio://")) {
      inputMode.value = "file";
      ctx.owners.capture.localAudioFileName = sourceTitleForUrl(value);
      setBrowserStreamMode(false);
    } else if (value.startsWith("microphone://")) {
      inputMode.value = "microphone";
      setBrowserStreamMode(true, "", "microphone");
    } else if (value.startsWith("system-audio://")) {
      inputMode.value = "system";
      setBrowserStreamMode(true, "", "display");
    } else if (value.startsWith("mixed-audio://")) {
      inputMode.value = "both";
      setBrowserStreamMode(true, "", "mixed");
    }
    updateMediaMode();
  }
  function float32ToBase64(samples) {
    const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    let binary = "";
    const step = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += step) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + step));
    }
    return btoa(binary);
  }
  function resampleFloat32(samples, fromRate, toRate=targetCaptureSampleRate) {
    const sourceRate = Math.max(1, Math.round(Number(fromRate || toRate)));
    const targetRate = Math.max(1, Math.round(Number(toRate || sourceRate)));
    if (sourceRate === targetRate || samples.length <= 1) {
      const copy = new Float32Array(samples.length);
      copy.set(samples);
      return copy;
    }
    const outputLength = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
    const output = new Float32Array(outputLength);
    const ratio = sourceRate / targetRate;
    for (let i = 0; i < outputLength; i += 1) {
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(samples.length - 1, left + 1);
      const fraction = position - left;
      output[i] = samples[left] + (samples[right] - samples[left]) * fraction;
    }
    return output;
  }
  function rms(samples) {
    if (!samples || !samples.length) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const value = samples[i];
      sum += value * value;
    }
    return Math.sqrt(sum / samples.length);
  }
  function rememberCapturePreRoll(samples, sampleRate) {
    const copy = new Float32Array(samples.length);
    copy.set(samples);
    ctx.owners.capture.capturePreRoll.push(copy);
    ctx.owners.capture.capturePreRollSamples += copy.length;
    const maxSamples = Math.max(0, Math.floor(sampleRate * capturePreRollSeconds));
    while (ctx.owners.capture.capturePreRollSamples > maxSamples && ctx.owners.capture.capturePreRoll.length) {
      ctx.owners.capture.capturePreRollSamples -= ctx.owners.capture.capturePreRoll.shift().length;
    }
  }
  function flushCapturePreRoll(sampleRate) {
    if (!ctx.owners.capture.capturePreRollSamples) return;
    const combined = new Float32Array(ctx.owners.capture.capturePreRollSamples);
    let offset = 0;
    for (const chunk of ctx.owners.capture.capturePreRoll) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    ctx.owners.capture.capturePreRoll = [];
    ctx.owners.capture.capturePreRollSamples = 0;
    queueBrowserAudioChunk(combined, sampleRate);
  }
  function queueBrowserAudioChunk(samples, sampleRate) {
    const resampled = resampleFloat32(samples, sampleRate, targetCaptureSampleRate);
    const payload = {
      sample_rate: targetCaptureSampleRate,
      audio_b64: float32ToBase64(resampled),
    };
    ctx.owners.capture.captureSendQueue = ctx.owners.capture.captureSendQueue
      .then(() => post("/api/audio-chunk", payload))
      .catch(error => log(`Audio chunk failed: ${error.message}`));
  }
  function flushBrowserAudio(force=false) {
    const sampleRate = ctx.owners.capture.captureAudioContext ? ctx.owners.capture.captureAudioContext.sampleRate : 16000;
    const targetSamples = Math.max(1600, Math.floor(sampleRate * 0.5));
    if (!force && ctx.owners.capture.capturePendingSamples < targetSamples) return;
    if (ctx.owners.capture.capturePendingSamples <= 0) return;
    const combined = new Float32Array(ctx.owners.capture.capturePendingSamples);
    let offset = 0;
    for (const chunk of ctx.owners.capture.capturePending) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    ctx.owners.capture.capturePending = [];
    ctx.owners.capture.capturePendingSamples = 0;
    queueBrowserAudioChunk(combined, sampleRate);
  }
  function stopCaptureStream(stream) {
    if (!stream) return;
    stream.getTracks().forEach(track => {
      track.onended = null;
      try { track.stop(); } catch (_) {}
    });
  }
  async function requestMicrophoneCapture(audioOptions) {
    if (!navigator.mediaDevices.getUserMedia) {
      throw new Error("Microphone capture is not available in this browser.");
    }
    log("Allow microphone access.");
    const stream = await navigator.mediaDevices.getUserMedia({video: false, audio: audioOptions});
    if (!stream.getAudioTracks().length) {
      stopCaptureStream(stream);
      throw new Error("No microphone audio track was shared.");
    }
    return stream;
  }
  async function requestDisplayAudioCapture(audioOptions) {
    if (!navigator.mediaDevices.getDisplayMedia) {
      throw new Error("Browser tab-audio capture is not available in this browser.");
    }
    log("Choose the YouTube/app tab or window and enable tab/system audio in the share dialog.");
    const stream = await navigator.mediaDevices.getDisplayMedia({video: true, audio: audioOptions});
    if (!stream.getAudioTracks().length) {
      stopCaptureStream(stream);
      throw new Error("No tab or system audio track was shared.");
    }
    return stream;
  }
  function stopBrowserAudioCapture() {
    flushBrowserAudio(true);
    if (ctx.owners.capture.captureProcessor) {
      try { ctx.owners.capture.captureProcessor.disconnect(); } catch (_) {}
      ctx.owners.capture.captureProcessor.onaudioprocess = null;
      ctx.owners.capture.captureProcessor = null;
    }
    if (ctx.owners.capture.captureMicGainNode) {
      try { ctx.owners.capture.captureMicGainNode.disconnect(); } catch (_) {}
      ctx.owners.capture.captureMicGainNode = null;
    }
    for (const node of ctx.owners.capture.captureSourceNodes) {
      try { node.disconnect(); } catch (_) {}
    }
    ctx.owners.capture.captureSourceNodes = [];
    if (ctx.owners.capture.captureSourceNode) {
      try { ctx.owners.capture.captureSourceNode.disconnect(); } catch (_) {}
      ctx.owners.capture.captureSourceNode = null;
    }
    if (ctx.owners.capture.captureSilentGain) {
      try { ctx.owners.capture.captureSilentGain.disconnect(); } catch (_) {}
      ctx.owners.capture.captureSilentGain = null;
    }
    const streams = ctx.owners.capture.captureStreams.length ? ctx.owners.capture.captureStreams : (ctx.owners.capture.captureStream ? [ctx.owners.capture.captureStream] : []);
    streams.forEach(stopCaptureStream);
    ctx.owners.capture.captureStreams = [];
    ctx.owners.capture.captureStream = null;
    if (ctx.owners.capture.captureAudioContext) {
      ctx.owners.capture.captureAudioContext.close().catch(() => {});
      ctx.owners.capture.captureAudioContext = null;
    }
    ctx.owners.capture.capturePending = [];
    ctx.owners.capture.capturePendingSamples = 0;
    ctx.owners.capture.captureAudioStarted = false;
    ctx.owners.capture.capturePreRoll = [];
    ctx.owners.capture.capturePreRollSamples = 0;
    setCaptureLevel(0);
  }
  async function startBrowserAudioCapture() {
    stopBrowserAudioCapture();
    if (!navigator.mediaDevices) {
      throw new Error("Browser audio capture is not available in this browser.");
    }
    const audioOptions = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    };
    let displayStream = null;
    let microphoneStream = null;
    try {
      if (ctx.owners.capture.captureSourceKind === "microphone") {
        microphoneStream = await requestMicrophoneCapture(audioOptions);
        ctx.owners.capture.captureStreams = [microphoneStream];
      } else if (ctx.owners.capture.captureSourceKind === "mixed") {
        displayStream = await requestDisplayAudioCapture(audioOptions);
        microphoneStream = await requestMicrophoneCapture(audioOptions);
        ctx.owners.capture.captureStreams = [displayStream, microphoneStream];
      } else {
        displayStream = await requestDisplayAudioCapture(audioOptions);
        ctx.owners.capture.captureStreams = [displayStream];
      }
    } catch (error) {
      stopCaptureStream(displayStream);
      stopCaptureStream(microphoneStream);
      ctx.owners.capture.captureStreams = [];
      ctx.owners.capture.captureStream = null;
      throw error;
    }
    ctx.owners.capture.captureStream = ctx.owners.capture.captureStreams[0] || null;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    ctx.owners.capture.captureAudioContext = new AudioContextClass({sampleRate: targetCaptureSampleRate});
    ctx.owners.capture.captureProcessor = ctx.owners.capture.captureAudioContext.createScriptProcessor(4096, 1, 1);
    ctx.owners.capture.captureSilentGain = ctx.owners.capture.captureAudioContext.createGain();
    ctx.owners.capture.captureSilentGain.gain.value = 0;
    ctx.owners.capture.captureProcessor.onaudioprocess = event => {
      const input = event.inputBuffer.getChannelData(0);
      const copy = copyCaptureSamples(input);
      const level = rms(copy);
      setCaptureLevel(level);
      if (!ctx.owners.capture.captureAudioStarted) {
        rememberCapturePreRoll(copy, ctx.owners.capture.captureAudioContext.sampleRate);
        if (level < captureStartRmsThreshold) {
          return;
        }
        ctx.owners.capture.captureAudioStarted = true;
        log("Detected audible input; streaming audio to backend.");
        flushCapturePreRoll(ctx.owners.capture.captureAudioContext.sampleRate);
        return;
      }
      ctx.owners.capture.capturePending.push(copy);
      ctx.owners.capture.capturePendingSamples += copy.length;
      flushBrowserAudio(false);
    };
    if (ctx.owners.capture.captureSourceKind === "mixed") {
      const displayAudioStream = new MediaStream(displayStream.getAudioTracks());
      const microphoneAudioStream = new MediaStream(microphoneStream.getAudioTracks());
      const displaySourceNode = ctx.owners.capture.captureAudioContext.createMediaStreamSource(displayAudioStream);
      const microphoneSourceNode = ctx.owners.capture.captureAudioContext.createMediaStreamSource(microphoneAudioStream);
      ctx.owners.capture.captureMicGainNode = ctx.owners.capture.captureAudioContext.createGain();
      ctx.owners.capture.captureMicGainNode.gain.value = microphoneGainValue();
      displaySourceNode.connect(ctx.owners.capture.captureProcessor);
      microphoneSourceNode.connect(ctx.owners.capture.captureMicGainNode);
      ctx.owners.capture.captureMicGainNode.connect(ctx.owners.capture.captureProcessor);
      ctx.owners.capture.captureSourceNodes = [displaySourceNode, microphoneSourceNode];
    } else {
      const audioOnlyStream = new MediaStream(ctx.owners.capture.captureStream.getAudioTracks());
      ctx.owners.capture.captureSourceNode = ctx.owners.capture.captureAudioContext.createMediaStreamSource(audioOnlyStream);
      ctx.owners.capture.captureSourceNode.connect(ctx.owners.capture.captureProcessor);
    }
    ctx.owners.capture.captureProcessor.connect(ctx.owners.capture.captureSilentGain);
    ctx.owners.capture.captureSilentGain.connect(ctx.owners.capture.captureAudioContext.destination);
    await ctx.owners.capture.captureAudioContext.resume();
    ctx.owners.capture.captureStreams.forEach(stream => {
      stream.getTracks().forEach(track => {
        track.onended = () => {
          if (ctx.owners.capture.browserStreamMode) log("Browser audio capture ended.");
          stopBrowserAudioCapture();
        };
      });
    });
    log(`Browser audio capture armed at ${Math.round(ctx.owners.capture.captureAudioContext.sampleRate)} Hz; waiting for audible input.`);
  }

  Object.assign(ctx.api, {applySpeakerRefinementSettings, applySpeakerSensitivity, applySpeakerSensitivityIfDirty, browserStreamSourceUrl, captureGainValue, clockLabel, copyCaptureSamples, currentMediaSeconds, currentSessionDraftTitle, currentSessionSourceMetadata, currentStartSessionMetadata, extractYouTubeId, fastProcessingEnabled, float32ToBase64, flushBrowserAudio, flushCapturePreRoll, initializeInputModeFromSource, initializeSessionIdentity, mediaDurationSeconds, microphoneGainValue, normalizeUrl, populatePresetVideos, prepareBrowserStreamSession, presetForUrl, queueBrowserAudioChunk, randomSessionId, reflectRuntimeStatus, rememberCapturePreRoll, requestDisplayAudioCapture, requestMicrophoneCapture, resampleFloat32, rms, selectedSpeakerSensitivityPreset, setBrowserStreamMode, setCaptureLevel, setSourceControlsDisabled, setSourceModeMenuOpen, setState, setStreamHint, sourceTitleForUrl, speakerRefinementPayload, startBrowserAudioCapture, stopBrowserAudioCapture, stopCaptureStream, storeBooleanValue, storeSessionValue, storedBooleanValue, storedSessionValue, syncFastProcessingControls, syncPresetSelection, syncSourceReadyState, syncSpeakerRefinementSettings, updateLanguageIndicator, updateMediaMode, updateMediaTimeline, updateMicGainLabel, updateSpeakerCount, updateSpeakerSensitivityLabel, youtubeEmbedUrl});
  ctx.activators.push(() => {
    groupTranscriptTurns.checked = storedBooleanValue(transcriptGroupTurnsStorageKey, true);
    fastProcessing.checked = storedBooleanValue(fastProcessingStorageKey, true);
    fastProcessing.addEventListener("change", () => {
      storeBooleanValue(fastProcessingStorageKey, fastProcessing.checked);
      syncFastProcessingControls();
    });
    showTranscriptReviewHints.checked = storedBooleanValue(transcriptReviewHintsStorageKey, false);
    initializeTranslationControls();
    initializeSessionIdentity();
    updateLanguageIndicator();
    populatePresetVideos();
    source.value = initialSource;
    syncPresetSelection(initialSource);
    updateMicGainLabel();
    setCaptureLevel(0);
    updateMediaMode();
    newSpeakerSensitivity.value = speakerSensitivityConfig.selected || 3;
    updateSpeakerSensitivityLabel();
    syncSpeakerRefinementSettings(speakerRefinementConfig);
    initializeInputModeFromSource();
  });
}
