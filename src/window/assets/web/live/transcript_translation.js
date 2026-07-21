export function installTranscriptTranslation(ctx) {
  const {audio, inputMode, languageConfig, languageName, load, sentences, source, speakerColors, start, state, statusBox, stop, translationActivity, translationConfig, translationControls, translationDisplayModeControl, translationDisplayModeStorageKey, translationIncludeOriginalControl, translationIncludeOriginalStorageKey, translationLanguageLabelModeControl, translationLanguageLabelModeStorageKey, translationMenuButton, translationMenuHint, translationMenuPanel, translationMenuSummary, translationPrimaryField, translationPrimaryTargetControl, translationPrimaryTargetStorageKey, translationProvider, translationProviderAttribution, translationTargetList, video} = ctx;
  const clearLiveSpeakerState = (...args) => ctx.api.clearLiveSpeakerState(...args), clearMeetingEvidenceHighlight = (...args) => ctx.api.clearMeetingEvidenceHighlight(...args), clearTranscriptSelection = (...args) => ctx.api.clearTranscriptSelection(...args), findFinalSentenceRow = (...args) => ctx.api.findFinalSentenceRow(...args), finiteAudioSecond = (...args) => ctx.api.finiteAudioSecond(...args), log = (...args) => ctx.api.log(...args), post = (...args) => ctx.api.post(...args), refreshSpeakerPanelSentenceCounts = (...args) => ctx.api.refreshSpeakerPanelSentenceCounts(...args), refreshTranscriptVisibility = (...args) => ctx.api.refreshTranscriptVisibility(...args), savedSessionReviewOpen = (...args) => ctx.api.savedSessionReviewOpen(...args), setBrowserStreamMode = (...args) => ctx.api.setBrowserStreamMode(...args), setState = (...args) => ctx.api.setState(...args), startBrowserLiveObservation = (...args) => ctx.api.startBrowserLiveObservation(...args), startPlaybackClock = (...args) => ctx.api.startPlaybackClock(...args), stopBrowserLiveObservationTimerOnly = (...args) => ctx.api.stopBrowserLiveObservationTimerOnly(...args), storeSessionValue = (...args) => ctx.api.storeSessionValue(...args), storedBooleanValue = (...args) => ctx.api.storedBooleanValue(...args), storedSessionValue = (...args) => ctx.api.storedSessionValue(...args), updateMediaTimeline = (...args) => ctx.api.updateMediaTimeline(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function resetTranscriptDisplay() {
    stopBrowserLiveObservationTimerOnly();
    ctx.owners.transcript.browserLiveObservationStarted = false;
    ctx.owners.transcript.browserLiveObservationBuffer = [];
    sentences.textContent = "";
    ctx.owners.translation.translationStatesBySentence.clear();
    ctx.owners.translation.browserTranslationSourcesBySentence.clear();
    ctx.owners.translation.browserTranslationJobs.clear();
    renderTranslationMenu();
    statusBox.textContent = "";
    clearMeetingEvidenceHighlight();
    clearTranscriptSelection();
    ctx.owners.transcript.transcriptClearBeforeSeconds = 0;
    ctx.owners.capture.currentRealtimeGeneration = 0;
    clearLiveSpeakerState();
    clearUnsavedDetectedSpeakerDisplay();
    ctx.owners.speakers.renderedSpeakerSentenceCounts = {};
    ctx.owners.speakers.renderedSpeakerSpeakingSeconds = {};
    ctx.owners.speakers.hasRenderedFinalSentenceRows = false;
    syncSpeakerSessionBaselines();
    refreshSpeakerPanelSentenceCounts();
  }
  function currentTranscriptClearBoundarySeconds() {
    let boundary = ctx.owners.transcript.transcriptClearBeforeSeconds;
    Array.from(sentences.querySelectorAll(".row")).forEach(row => {
      const end = finiteAudioSecond(row.dataset.end, NaN);
      if (Number.isFinite(end)) boundary = Math.max(boundary, end);
    });
    return boundary;
  }
  function itemIsBeforeClearedTranscriptBoundary(item) {
    if (!(ctx.owners.transcript.transcriptClearBeforeSeconds > 0)) return false;
    const start = finiteAudioSecond(item && item.start, NaN);
    if (Number.isFinite(start)) return start < ctx.owners.transcript.transcriptClearBeforeSeconds;
    const end = finiteAudioSecond(item && item.end, NaN);
    return Number.isFinite(end) && end <= ctx.owners.transcript.transcriptClearBeforeSeconds;
  }
  function clearDisplayedTranscript() {
    if (!sentences.querySelector(".row")) {
      log("Transcript is already clear.");
      return;
    }
    ctx.owners.transcript.transcriptClearBeforeSeconds = currentTranscriptClearBoundarySeconds();
    sentences.textContent = "";
    ctx.owners.translation.translationStatesBySentence.clear();
    ctx.owners.translation.browserTranslationSourcesBySentence.clear();
    ctx.owners.translation.browserTranslationJobs.clear();
    renderTranslationMenu();
    clearTranscriptSelection();
    ctx.owners.speakers.renderedSpeakerSentenceCounts = {};
    ctx.owners.speakers.renderedSpeakerSpeakingSeconds = {};
    ctx.owners.speakers.hasRenderedFinalSentenceRows = false;
    clearLiveSpeakerState();
    refreshSpeakerPanelSentenceCounts();
    refreshTranscriptVisibility();
    log("Cleared transcript.");
  }
  function clearUnsavedDetectedSpeakerDisplay() {
    if (ctx.owners.speakers.speakerLibraryState.group_name) return;
    const retainedSpeakers = ctx.owners.speakers.speakerLibraryState.speakers.filter(speaker => (
      speaker.source === "reference" || speaker.locked || speaker.reference_audio
    ));
    if (retainedSpeakers.length === ctx.owners.speakers.speakerLibraryState.speakers.length) return;
    updateSpeakerState({...ctx.owners.speakers.speakerLibraryState, speakers: retainedSpeakers});
  }
  function syncSpeakerSessionBaselines(state = ctx.owners.speakers.speakerLibraryState) {
    const counts = {};
    const speakingSeconds = {};
    (Array.isArray(state.speakers) ? state.speakers : []).forEach(speaker => {
      const speakerId = speaker && speaker.id;
      if (!speakerId) return;
      counts[speakerId] = Number(speaker.sentence_count || 0);
      speakingSeconds[speakerId] = Number(speaker.speech_seconds || 0);
    });
    ctx.owners.speakers.speakerSessionBaselineSentenceCounts = counts;
    ctx.owners.speakers.speakerSessionBaselineSpeakingSeconds = speakingSeconds;
  }
  function refreshMediaElements(version) {
    ctx.owners.capture.resumePlaybackPending = false;
    setBrowserStreamMode(false);
    ctx.owners.capture.mediaVersion = Number(version || Date.now());
    video.pause();
    audio.pause();
    if ((inputMode.value || "youtube") === "file") {
      video.removeAttribute("src");
    } else {
      video.src = `/media/video?v=${ctx.owners.capture.mediaVersion}`;
    }
    audio.src = `/media/audio?v=${ctx.owners.capture.mediaVersion}`;
    video.load();
    audio.load();
    updateMediaTimeline();
  }
  function currentPlaybackElements() {
    return (inputMode.value || "youtube") === "file" ? [audio] : [video, audio];
  }
  function playbackElementLabel(element, index) {
    if (element === audio) return "audio";
    if (element === video) return "video";
    return `media ${index + 1}`;
  }
  async function unlockPlayback() {
    const elements = currentPlaybackElements();
    elements.forEach(element => {
      element.currentTime = 0;
      if (element === video) element.muted = true;
      if (element === audio) element.volume = 1.0;
    });
    const results = await Promise.allSettled(elements.map(element => element.play()));
    elements.forEach(element => {
      element.pause();
      element.currentTime = 0;
    });
    return results.map((result, index) => ({result, element: elements[index], index}));
  }
  function logRejectedPlayback(results) {
    results.forEach((entry, index) => {
      const result = entry && entry.result ? entry.result : entry;
      if (result.status === "rejected") {
        log(`${playbackElementLabel(entry && entry.element, index)} playback blocked: ${result.reason?.name || result.reason}`);
      }
    });
  }
  async function startSynchronizedPlaybackFromGesture() {
    const playbackElements = currentPlaybackElements();
    playbackElements.forEach(element => {
      element.currentTime = 0;
      if (element === video) element.muted = true;
      if (element === audio) element.volume = 1.0;
    });
    // Invoke play before the first await so a second Start click retains user activation.
    const playbackPromises = playbackElements.map(element => element.play());
    const playbackResults = await Promise.allSettled(playbackPromises);
    const results = playbackResults.map((result, index) => ({result, element: playbackElements[index], index}));
    logRejectedPlayback(results);
    if (results.some(entry => entry.result.status === "rejected")) {
      ctx.owners.capture.resumePlaybackPending = true;
      start.disabled = false;
      stop.disabled = false;
      start.textContent = "Resume playback";
      setState("Playback paused");
      log("Playback needs one direct Resume playback click after backend warmup.");
      return false;
    }
    ctx.owners.capture.resumePlaybackPending = false;
    start.textContent = "Start transcription";
    startPlaybackClock();
    startBrowserLiveObservation();
    setState("Playing");
    return true;
  }
  function scrollSentencesToBottom() {
    if (!ctx.owners.speakers.followLiveEnabled) return;
    requestAnimationFrame(() => {
      if (sentences.scrollHeight > sentences.clientHeight + 4) {
        sentences.scrollTop = sentences.scrollHeight;
      } else {
        window.scrollTo({top: document.documentElement.scrollHeight, behavior:"smooth"});
      }
    });
  }
  function speakerIndex(label) {
    const match = /^S(\d+)$/.exec(String(label || ""));
    return match ? Number(match[1]) : null;
  }
  function isLiveProvisionalSpeaker(speaker) {
    const value = speaker && typeof speaker === "object" ? speaker : {id: speaker};
    const speakerId = String(value.id || "");
    return value.source === "live_provisional"
      || /^provisional_\d+$/.test(speakerId)
      || /^LIVE_NEW_\d+$/.test(speakerId);
  }
  function speakerDisplayLabel(label) {
    if (label && ctx.owners.speakers.speakerNames[label]) return ctx.owners.speakers.speakerNames[label];
    const index = speakerIndex(label);
    if (index !== null) return `Speaker ${index}`;
    return isLiveProvisionalSpeaker(label) ? "Matching new voice..." : "Unknown";
  }
  function revisionSpeakerId(label) {
    const value = String(label || "").trim();
    return value && value !== "UNKNOWN" ? value : "UNKNOWN";
  }
  function revisionSpeakerBadge(label) {
    const speakerId = revisionSpeakerId(label);
    const index = speakerIndex(speakerId);
    if (index !== null) return `S${index}`;
    return speakerId === "UNKNOWN" ? "Unknown" : speakerId;
  }
  function parseRevisionChain(row) {
    if (!row || !row.dataset.revisionChain) return [];
    try {
      const values = JSON.parse(row.dataset.revisionChain);
      if (!Array.isArray(values)) return [];
      return values.map(revisionSpeakerId).filter(Boolean);
    } catch (error) {
      return [];
    }
  }
  function pushRevisionSpeaker(chain, speakerId) {
    const normalized = revisionSpeakerId(speakerId);
    if (!chain.length || chain[chain.length - 1] !== normalized) {
      chain.push(normalized);
    }
  }
  function sentenceRevisionLabel(row, item, nextSpeakerId, previousSpeakerId) {
    if (!item.revision) {
      if (!item.pending && !item.realtime && row) {
        delete row.dataset.revisionChain;
      }
      return "";
    }
    const chain = parseRevisionChain(row);
    const from = item.revision_from || item.prototype_reassigned_from || item.retro_reassigned_from || previousSpeakerId || "UNKNOWN";
    const normalizedFrom = revisionSpeakerId(from);
    if (!chain.length || normalizedFrom !== "UNKNOWN") {
      pushRevisionSpeaker(chain, normalizedFrom);
    }
    pushRevisionSpeaker(chain, nextSpeakerId || item.revision_to || item.assigned_speaker || "UNKNOWN");
    row.dataset.revisionChain = JSON.stringify(chain);
    const prefix = item.provisional_assignment ? "Tentative" : "Revised";
    return `${prefix}: ${chain.map(revisionSpeakerBadge).join(" -> ")}`;
  }
  function speakerPanelName(speaker) {
    return speaker.name || speaker.display_name || speakerDisplayLabel(speaker.id);
  }
  function speakerColor(label) {
    if (isLiveProvisionalSpeaker(label)) return "#8F9BA8";
    const index = speakerIndex(label);
    if (index === null) return null;
    return speakerColors[(index - 1) % speakerColors.length];
  }
  function speakerProbabilityKey(label) {
    const index = speakerIndex(label);
    return index === null ? null : `speaker${index}`;
  }
  function probabilityDisplayLabel(key) {
    const match = /^speaker(\d+)$/.exec(String(key || ""));
    if (match) return speakerDisplayLabel(`S${Number(match[1])}`);
    return key === "unknown" ? "Unknown" : key;
  }
  function probabilityColor(key) {
    if (key === "unknown") return "#7d8997";
    const match = /^speaker(\d+)$/.exec(String(key || ""));
    return match ? speakerColor(`S${match[1]}`) : "#d7dee8";
  }
  function pruneSpeakerFilterState() {
    const validSpeakerIds = new Set(ctx.owners.speakers.speakerLibraryState.speakers.map(speaker => speaker.id).filter(Boolean));
    ctx.owners.speakers.soloSpeakerIds = new Set(Array.from(ctx.owners.speakers.soloSpeakerIds).filter(speakerId => validSpeakerIds.has(speakerId)));
    ctx.owners.speakers.mutedSpeakerIds = new Set(Array.from(ctx.owners.speakers.mutedSpeakerIds).filter(speakerId => validSpeakerIds.has(speakerId)));
  }
  function speakerTranscriptVisible(speakerId) {
    if (ctx.owners.speakers.mutedSpeakerIds.has(speakerId)) return false;
    if (ctx.owners.speakers.soloSpeakerIds.size > 0) return ctx.owners.speakers.soloSpeakerIds.has(speakerId);
    return true;
  }
  function normalizedTranslationLanguageCode(value) {
    const raw = value && typeof value === "object" ? (value.code || value.id || value.language) : value;
    return String(raw || "").trim().toLowerCase();
  }
  function translationLanguageOptions() {
    const candidates = [
      translationConfig && translationConfig.languages,
      translationConfig && translationConfig.supported_languages,
      translationConfig && translationConfig.available_languages,
    ].find(value => Array.isArray(value)) || [];
    const result = [];
    const seen = new Set();
    candidates.forEach(item => {
      const code = normalizedTranslationLanguageCode(item);
      if (!code || seen.has(code)) return;
      seen.add(code);
      const name = item && typeof item === "object"
        ? String(item.name || item.label || item.display_name || code)
        : String(item || code);
      const flagUrl = item && typeof item === "object" ? String(item.flag_url || "") : "";
      result.push({code, name, flag_url:flagUrl});
    });
    return result;
  }
  function normalizedTranslationLanguageLabelMode(value) {
    const mode = String(value || "").trim().toLowerCase();
    return ["flag", "flag_name", "name", "flag_code", "code"].includes(mode) ? mode : "flag_name";
  }
  function translationLanguageCodeLabel(code) {
    return normalizedTranslationLanguageCode(code).toUpperCase();
  }
  function translationConfiguredTargets() {
    const candidates = translationConfig && (
      translationConfig.selected_targets
      || translationConfig.target_languages
    );
    if (!Array.isArray(candidates)) return [];
    return candidates.map(normalizedTranslationLanguageCode).filter(Boolean);
  }
  function translationProviderLabel() {
    const provider = translationConfig && translationConfig.provider;
    let label = "";
    if (provider && typeof provider === "object") {
      label = String(provider.label || provider.display_name || provider.name || provider.id || provider.provider || "Translation");
    } else {
      label = String(provider || (translationConfig && translationConfig.provider_label) || "Translation");
    }
    return translationConfig && translationConfig.browser_preferred
      ? `Chrome on-device -> ${label} fallback`
      : label;
  }
  function translationProviderId() {
    const provider = translationConfig && translationConfig.provider;
    return String(
      provider && typeof provider === "object"
        ? (provider.id || provider.provider || "")
        : (provider || "")
    ).toLowerCase();
  }
  function translationProviderLicense() {
    const provider = translationConfig && translationConfig.provider;
    if (!provider || typeof provider !== "object") return null;
    const metadata = provider.model_metadata && typeof provider.model_metadata === "object"
      ? provider.model_metadata
      : null;
    const license = (metadata && metadata.license) || provider.license;
    return license && typeof license === "object" ? license : null;
  }
  function translationProviderNotice() {
    const provider = translationConfig && translationConfig.provider;
    if (!provider || typeof provider !== "object") return "";
    const metadata = provider.model_metadata && typeof provider.model_metadata === "object"
      ? provider.model_metadata
      : null;
    const license = translationProviderLicense();
    return [
      metadata && metadata.intended_use_notice,
      license && license.notice,
      license && license.url,
    ].filter(Boolean).join(" ");
  }
  function translationLanguageName(code) {
    const normalized = normalizedTranslationLanguageCode(code);
    if (normalized && normalized === normalizedTranslationLanguageCode(languageConfig.code)) {
      return String(languageConfig.name || normalized);
    }
    const option = translationLanguageOptions().find(item => item.code === normalized);
    return option ? option.name : String(code || normalized || "Translation");
  }
  function translationLanguageFlagUrl(code) {
    const normalized = normalizedTranslationLanguageCode(code);
    if (normalized && normalized === normalizedTranslationLanguageCode(languageConfig.code)) {
      return String(languageConfig.flag_url || "");
    }
    const option = translationLanguageOptions().find(item => item.code === normalized);
    return option ? String(option.flag_url || "") : "";
  }
  function translationFeatureAvailable() {
    return Boolean(
      translationConfig
      && translationConfig.available !== false
      && translationLanguageOptions().length
    );
  }
  function browserPreferredTranslationEnabled() {
    return Boolean(
      translationFeatureAvailable()
      && translationConfig
      && translationConfig.browser_preferred
    );
  }
  function chromeTranslationUnavailable(value) {
    return ["no", "unavailable", "unsupported"].includes(String(value || "").toLowerCase());
  }
  async function createChromeTranslator(sourceLanguage, targetLanguage) {
    const options = {sourceLanguage, targetLanguage};
    const modern = globalThis.Translator;
    if (modern && typeof modern.create === "function") {
      if (typeof modern.availability === "function") {
        const availability = await modern.availability(options);
        if (chromeTranslationUnavailable(availability)) return null;
      }
      return modern.create(options);
    }
    const legacy = globalThis.translation;
    if (legacy && typeof legacy.createTranslator === "function") {
      if (typeof legacy.canTranslate === "function") {
        const availability = await legacy.canTranslate(options);
        if (chromeTranslationUnavailable(availability)) return null;
      }
      return legacy.createTranslator(options);
    }
    return null;
  }
  function chromeTranslatorForPair(sourceLanguage, targetLanguage) {
    const key = `${sourceLanguage}>${targetLanguage}`;
    if (!ctx.owners.translation.chromeTranslatorsByPair.has(key)) {
      ctx.owners.translation.chromeTranslatorsByPair.set(
        key,
        Promise.resolve(createChromeTranslator(sourceLanguage, targetLanguage)).catch(() => null),
      );
    }
    return ctx.owners.translation.chromeTranslatorsByPair.get(key);
  }
  async function requestBackendTranslationFallback(source, targetLanguage, reason) {
    await post("/api/translation/browser-fallback", {
      segment_id: source.segment_id,
      target_language: targetLanguage,
      source_text_hash: source.source_text_hash,
      source_revision: source.source_revision,
      reason:String(reason || "Chrome Translator API unavailable"),
    });
  }
  async function runBrowserPreferredTranslation(source, targetLanguage) {
    const startedAt = performance.now();
    const baseEvent = {
      segment_id:source.segment_id,
      sentence_index:source.segment_id,
      source_text_hash:source.source_text_hash,
      source_revision:source.source_revision,
      target_language:targetLanguage,
      provider:"chrome_translator",
    };
    applyTranslationEvent({...baseEvent, status:"translating"});
    try {
      const sourceLanguage = normalizedTranslationLanguageCode(languageConfig.code);
      const translator = await chromeTranslatorForPair(sourceLanguage, targetLanguage);
      if (!translator || typeof translator.translate !== "function") {
        await requestBackendTranslationFallback(source, targetLanguage, "Language pair unavailable in Chrome");
        return;
      }
      const translatedText = String(await translator.translate(source.text) || "").trim();
      if (!translatedText) throw new Error("Chrome returned an empty translation");
      const completed = {
        ...baseEvent,
        status:"complete",
        text:translatedText,
        latency_seconds:Math.max(0, performance.now() - startedAt) / 1000,
      };
      applyTranslationEvent(completed);
      await post("/api/translation/browser-result", completed);
    } catch (error) {
      try {
        await requestBackendTranslationFallback(source, targetLanguage, error && error.message);
      } catch (fallbackError) {
        applyTranslationEvent({
          ...baseEvent,
          status:"error",
          error:`Chrome and backend translation failed: ${fallbackError.message || fallbackError}`,
        });
      }
    }
  }
  function queueBrowserPreferredTranslation(source, targetLanguage) {
    if (!browserPreferredTranslationEnabled() || !source || !source.segment_id) return;
    const target = normalizedTranslationLanguageCode(targetLanguage);
    if (!target || target === normalizedTranslationLanguageCode(languageConfig.code)) return;
    const jobKey = [source.segment_id, target, source.source_text_hash, source.source_revision].join("|");
    const existing = translationStateMap(source.segment_id);
    const state = existing && existing.get(target);
    if (ctx.owners.translation.browserTranslationJobs.has(jobKey) || (state && state.status === "complete")) return;
    ctx.owners.translation.browserTranslationJobs.add(jobKey);
    ctx.owners.translation.browserTranslationQueue = ctx.owners.translation.browserTranslationQueue
      .then(() => runBrowserPreferredTranslation(source, target))
      .finally(() => ctx.owners.translation.browserTranslationJobs.delete(jobKey))
      .catch(error => log(`Browser translation failed: ${error.message || error}`));
  }
  function queueBrowserPreferredTranslationsForSource(source) {
    if (!browserPreferredTranslationEnabled()) return;
    ctx.owners.translation.translationSelectedTargets.forEach(target => queueBrowserPreferredTranslation(source, target));
  }
  function queueBrowserPreferredTranslationsForAllSources() {
    ctx.owners.translation.browserTranslationSourcesBySentence.forEach(queueBrowserPreferredTranslationsForSource);
  }
  function translationMaximumTargets() {
    const configured = Number(translationConfig && translationConfig.max_targets);
    return Number.isFinite(configured) && configured > 0
      ? Math.floor(configured)
      : translationLanguageOptions().length;
  }
  function effectiveTranslationDisplayMode() {
    if (!translationFeatureAvailable() || !ctx.owners.translation.translationSelectedTargets.size) return "original";
    return ["single", "all"].includes(ctx.owners.translation.translationDisplayMode)
      ? ctx.owners.translation.translationDisplayMode
      : "original";
  }
  function selectedTranslationCodesForDisplay() {
    const selected = Array.from(ctx.owners.translation.translationSelectedTargets);
    if (effectiveTranslationDisplayMode() === "single") {
      const primary = selected.includes(ctx.owners.translation.translationPrimaryTarget)
        ? ctx.owners.translation.translationPrimaryTarget
        : (selected[0] || "");
      return primary ? [primary] : [];
    }
    return effectiveTranslationDisplayMode() === "all" ? selected : [];
  }
  function translationSentenceKey(item) {
    if (!item || typeof item !== "object") return "";
    for (const value of [item.sentence_index, item.segment_id, item.index]) {
      if (value !== undefined && value !== null && String(value).trim()) return String(value);
    }
    return "";
  }
  function translationTargetCode(item) {
    if (!item || typeof item !== "object") return "";
    return normalizedTranslationLanguageCode(
      item.target_language || item.target_language_code || item.language || item.language_code
    );
  }
  function normalizedTranslationStatus(value, textValue = "") {
    const status = String(value || "").trim().toLowerCase();
    if (["complete", "completed", "ready", "success", "succeeded"].includes(status)) return "complete";
    if (["error", "failed", "failure", "cancelled", "canceled", "superseded"].includes(status)) return "error";
    if (["translating", "running", "processing", "in_progress"].includes(status)) return "translating";
    if (["queued", "pending", "waiting"].includes(status)) return "queued";
    return String(textValue || "").trim() ? "complete" : "queued";
  }
  function translationStateMatchesRow(state, row) {
    if (!state || !row) return true;
    const eventHash = String(state.source_text_hash || "");
    const rowHash = String(row.dataset.sourceTextHash || "");
    if (eventHash && rowHash && eventHash !== rowHash) return false;
    const eventRevision = String(state.source_revision === undefined || state.source_revision === null ? "" : state.source_revision);
    const rowRevision = String(row.dataset.sourceRevision || "");
    return !(eventRevision && rowRevision && eventRevision !== rowRevision);
  }
  function translationStateMap(sentenceKey, create = false) {
    const key = String(sentenceKey || "");
    if (!key) return null;
    if (!ctx.owners.translation.translationStatesBySentence.has(key) && create) {
      ctx.owners.translation.translationStatesBySentence.set(key, new Map());
    }
    return ctx.owners.translation.translationStatesBySentence.get(key) || null;
  }
  function forgetSentenceTranslations(sentenceKey) {
    ctx.owners.translation.translationStatesBySentence.delete(String(sentenceKey || ""));
  }
  function applyTranslationEvent(item, options = {}) {
    if (!item || typeof item !== "object") return false;
    const sentenceKey = translationSentenceKey(item);
    const targetCode = translationTargetCode(item);
    if (!sentenceKey || !targetCode) return false;
    const row = findFinalSentenceRow(sentenceKey);
    const translatedText = item.text === undefined || item.text === null ? item.translated_text : item.text;
    const state = {
      sentence_index: item.sentence_index,
      segment_id: item.segment_id,
      source_revision: item.source_revision,
      source_text_hash: String(item.source_text_hash || item.source_hash || ""),
      target_language: targetCode,
      target_language_name: String(item.target_language_name || translationLanguageName(targetCode)),
      status: normalizedTranslationStatus(item.status, translatedText),
      text: String(translatedText || ""),
      error: String(item.error || item.message || ""),
      provider: String(item.provider && typeof item.provider === "object" ? (item.provider.id || item.provider.name || "") : (item.provider || "")),
    };
    if (row && !translationStateMatchesRow(state, row)) return false;
    translationStateMap(sentenceKey, true).set(targetCode, state);
    if (options.refresh !== false) refreshTranslationPresentation();
    return true;
  }
  function applyTranslationCollection(collection, options = {}) {
    if (!collection) return;
    const defaultSentenceKey = String(options.sentence_index || options.segment_id || "");
    const events = [];
    if (Array.isArray(collection)) {
      collection.forEach(item => events.push(item));
    } else if (collection && typeof collection === "object") {
      const nested = Array.isArray(collection.items)
        ? collection.items
        : (Array.isArray(collection.translations) ? collection.translations : null);
      if (nested) {
        nested.forEach(item => events.push(item));
      } else if (translationTargetCode(collection)) {
        events.push(collection);
      } else if (defaultSentenceKey) {
        Object.entries(collection).forEach(([languageCode, state]) => {
          if (state && typeof state === "object") {
            events.push({...state, sentence_index:state.sentence_index === undefined ? defaultSentenceKey : state.sentence_index, target_language:state.target_language || languageCode});
          } else if (typeof state === "string") {
            events.push({sentence_index:defaultSentenceKey, target_language:languageCode, status:"complete", text:state});
          }
        });
      } else {
        Object.entries(collection).forEach(([outerKey, outerValue]) => {
          if (Array.isArray(outerValue)) {
            outerValue.forEach(item => events.push({...item, sentence_index:item.sentence_index === undefined ? outerKey : item.sentence_index}));
            return;
          }
          if (!outerValue || typeof outerValue !== "object") return;
          if (translationTargetCode(outerValue)) {
            events.push({...outerValue, sentence_index:outerValue.sentence_index === undefined ? (defaultSentenceKey || outerKey) : outerValue.sentence_index});
            return;
          }
          Object.entries(outerValue).forEach(([languageCode, state]) => {
            if (state && typeof state === "object") {
              events.push({...state, sentence_index:state.sentence_index === undefined ? (defaultSentenceKey || outerKey) : state.sentence_index, target_language:state.target_language || languageCode});
            } else if (typeof state === "string") {
              events.push({sentence_index:defaultSentenceKey || outerKey, target_language:languageCode, status:"complete", text:state});
            }
          });
        });
      }
    }
    events.forEach(item => {
      const withSentence = defaultSentenceKey && translationSentenceKey(item) === ""
        ? {...item, sentence_index:defaultSentenceKey}
        : item;
      applyTranslationEvent(withSentence, {refresh:false});
    });
    if (options.refresh !== false) refreshTranslationPresentation();
  }
  function transcriptRowSentenceKeys(row) {
    if (!row) return [];
    if (row.dataset.groupIndexes) {
      try {
        const values = JSON.parse(row.dataset.groupIndexes);
        if (Array.isArray(values) && values.length) return values.map(String);
      } catch (_) {}
    }
    return row.dataset.index ? [String(row.dataset.index)] : [];
  }
  function translationStateForRow(row, languageCode) {
    const code = normalizedTranslationLanguageCode(languageCode);
    const keys = transcriptRowSentenceKeys(row);
    const states = keys.map(key => {
      const map = translationStateMap(key);
      const state = map && map.get(code);
      const sourceRow = findFinalSentenceRow(key);
      return state && translationStateMatchesRow(state, sourceRow) ? state : null;
    });
    const completedText = states
      .filter(state => state && state.status === "complete" && state.text.trim())
      .map(state => state.text.trim())
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (states.length && states.every(state => state && state.status === "complete" && state.text.trim())) {
      return {status:"complete", text:completedText, error:""};
    }
    const errors = states.filter(state => state && state.status === "error");
    const waiting = states.some(state => !state || state.status === "queued" || state.status === "translating");
    if (savedSessionReviewOpen() && waiting) {
      return {status:"error", text:completedText, error:"This language was not translated during the saved session."};
    }
    if (!waiting && errors.length) {
      return {status:"error", text:completedText, error:errors.map(state => state.error).filter(Boolean).join(" | ")};
    }
    return {status:"translating", text:completedText, error:""};
  }
  function createTranslationLanguageLabel(languageCode) {
    const label = document.createElement("span");
    label.className = "translation-language";
    const languageName = translationLanguageName(languageCode);
    const showFlag = ctx.owners.translation.translationLanguageLabelMode === "flag" || ctx.owners.translation.translationLanguageLabelMode.startsWith("flag_");
    const showName = ctx.owners.translation.translationLanguageLabelMode === "name" || ctx.owners.translation.translationLanguageLabelMode === "flag_name";
    const showCode = ctx.owners.translation.translationLanguageLabelMode === "code" || ctx.owners.translation.translationLanguageLabelMode === "flag_code";
    let hasFlag = false;
    if (showFlag) {
      const flagUrl = translationLanguageFlagUrl(languageCode);
      if (flagUrl) {
        const flag = document.createElement("img");
        flag.className = "translation-language-flag";
        flag.src = flagUrl;
        flag.alt = "";
        flag.setAttribute("aria-hidden", "true");
        label.appendChild(flag);
        hasFlag = true;
      }
    }
    if (showName) label.appendChild(document.createTextNode(languageName));
    if (showCode || (showFlag && !hasFlag && !showName)) {
      label.appendChild(document.createTextNode(translationLanguageCodeLabel(languageCode)));
    }
    label.title = languageName;
    label.setAttribute("aria-label", languageName);
    return label;
  }
  function createTranslationLine(languageCode, state, options = {}) {
    const line = document.createElement("div");
    line.className = "translation-line";
    line.lang = normalizedTranslationLanguageCode(languageCode);
    if (options.additional) line.classList.add("translation-additional");
    const label = createTranslationLanguageLabel(languageCode);
    line.appendChild(label);
    if (state.status === "complete") {
      line.appendChild(document.createTextNode(state.text));
      return line;
    }
    if (state.status === "error") {
      line.classList.add("translation-error");
      line.appendChild(document.createTextNode(state.text || "Translation unavailable"));
      if (state.error) line.title = state.error;
      return line;
    }
    line.classList.add("translation-pending");
    if (state.text) {
      line.appendChild(document.createTextNode(state.text));
      const pending = document.createElement("span");
      pending.className = "translation-inline-state";
      pending.textContent = "Translating…";
      line.appendChild(pending);
    } else {
      line.appendChild(document.createTextNode("Translating…"));
    }
    return line;
  }
  function refreshTranslationRow(row) {
    if (!row) return;
    const source = row.querySelector(".text");
    let lines = row.querySelector(".translation-lines");
    if (!lines) {
      lines = document.createElement("div");
      lines.className = "translation-lines";
      row.appendChild(lines);
    }
    const sourceSearchText = String(row.dataset.groupText || row.dataset.text || "");
    const translatedSearchText = Array.from(ctx.owners.translation.translationSelectedTargets)
      .map(code => translationStateForRow(row, code))
      .filter(state => state.status === "complete")
      .map(state => state.text);
    row.dataset.searchText = [sourceSearchText, ...translatedSearchText].filter(Boolean).join(" ");
    const mode = row.dataset.realtime === "true" ? "original" : effectiveTranslationDisplayMode();
    const targetCodes = mode === "original" ? [] : selectedTranslationCodesForDisplay();
    const includeOriginal = mode !== "original" && Boolean(translationIncludeOriginalControl.checked);
    const displayStates = targetCodes.map(code => translationStateForRow(row, code));
    const showPendingSource = mode === "single"
      && displayStates.length === 1
      && displayStates[0].status === "translating"
      && !displayStates[0].text;
    const showErrorFallback = displayStates.length > 0
      && displayStates.every(state => state.status === "error" && !state.text);
    if (source) {
      source.lang = String(languageConfig.code || "");
      source.hidden = mode !== "original" && !includeOriginal && !showErrorFallback && !showPendingSource;
      source.classList.toggle("translation-secondary", includeOriginal || showErrorFallback);
      const existingLabel = source.querySelector(".translation-language");
      if (existingLabel) existingLabel.remove();
      if (includeOriginal || showErrorFallback) {
        source.prepend(createTranslationLanguageLabel(languageConfig.code));
      }
    }
    lines.replaceChildren();
    if (!targetCodes.length || showPendingSource) {
      lines.hidden = true;
      return;
    }
    targetCodes.forEach((code, index) => {
      lines.appendChild(createTranslationLine(code, displayStates[index], {
        additional:index > 0,
      }));
    });
    lines.hidden = false;
  }
  function translationActivityState() {
    if (!ctx.owners.translation.translationSelectedTargets.size) return "off";
    let pending = false;
    let failed = false;
    Array.from(sentences.querySelectorAll(".row[data-realtime='false']")).forEach(row => {
      ctx.owners.translation.translationSelectedTargets.forEach(code => {
        const state = translationStateForRow(row, code);
        if (state.status === "error") failed = true;
        if (state.status === "translating") pending = true;
      });
    });
    return pending ? "pending" : (failed ? "error" : "active");
  }
  function refreshTranslationPresentation(options = {}) {
    Array.from(sentences.querySelectorAll(".row")).forEach(refreshTranslationRow);
    refreshTranscriptVisibility();
    if (options.menu) renderTranslationMenu();
    else refreshTranslationMenuStatus();
  }
  function translationMenuSummaryText() {
    const mode = effectiveTranslationDisplayMode();
    if (mode === "original") return "Original";
    const selected = selectedTranslationCodesForDisplay();
    const suffix = translationIncludeOriginalControl.checked ? " + original" : "";
    if (mode === "single") return `${translationLanguageName(selected[0])}${suffix}`;
    return `${selected.length} translation${selected.length === 1 ? "" : "s"}${suffix}`;
  }
  function refreshTranslationMenuStatus() {
    if (!translationFeatureAvailable()) return;
    translationMenuSummary.textContent = translationMenuSummaryText();
    translationActivity.dataset.state = translationActivityState();
    translationMenuButton.title = `${translationMenuSummary.textContent} · ${translationProviderLabel()}`;
  }
  function renderTranslationMenu() {
    if (!translationControls) return;
    const available = translationFeatureAvailable();
    translationControls.hidden = !available;
    if (!available) return;
    translationDisplayModeControl.value = ctx.owners.translation.translationDisplayMode;
    translationLanguageLabelModeControl.value = ctx.owners.translation.translationLanguageLabelMode;
    const providerLicense = translationProviderLicense();
    const licenseLabel = providerLicense && (providerLicense.identifier || providerLicense.display_name);
    translationProvider.textContent = [translationProviderLabel(), licenseLabel].filter(Boolean).join(" · ");
    translationProvider.title = translationProviderNotice();
    const googleAttributionRequired = translationProviderId() === "google_cloud";
    translationProviderAttribution.hidden = !googleAttributionRequired;
    translationProviderAttribution.textContent = googleAttributionRequired
      ? "Powered by Google Translate"
      : "";
    refreshTranslationMenuStatus();
    const selectedCodes = Array.from(ctx.owners.translation.translationSelectedTargets);
    translationPrimaryTargetControl.replaceChildren();
    selectedCodes.forEach(code => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = translationLanguageName(code);
      translationPrimaryTargetControl.appendChild(option);
    });
    if (!selectedCodes.includes(ctx.owners.translation.translationPrimaryTarget)) {
      ctx.owners.translation.translationPrimaryTarget = selectedCodes[0] || "";
    }
    translationPrimaryTargetControl.value = ctx.owners.translation.translationPrimaryTarget;
    translationPrimaryField.hidden = effectiveTranslationDisplayMode() !== "single";
    translationIncludeOriginalControl.disabled = effectiveTranslationDisplayMode() === "original";
    const maximum = translationMaximumTargets();
    translationTargetList.replaceChildren();
    translationLanguageOptions().forEach(language => {
      const label = document.createElement("label");
      label.className = "translation-target";
      label.title = language.name;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = language.code;
      input.checked = ctx.owners.translation.translationSelectedTargets.has(language.code);
      input.disabled = !input.checked && ctx.owners.translation.translationSelectedTargets.size >= maximum;
      const name = document.createElement("span");
      name.textContent = language.name;
      label.append(input, name);
      input.addEventListener("change", () => setTranslationTargetSelected(language.code, input.checked));
      translationTargetList.appendChild(label);
    });
    translationMenuHint.textContent = selectedCodes.length
      ? `${selectedCodes.length} of ${maximum} target language${maximum === 1 ? "" : "s"} selected`
      : "Choose at least one target language.";
  }
  function setTranslationMenuOpen(open) {
    if (!translationFeatureAvailable()) return;
    translationMenuPanel.hidden = !open;
    translationMenuButton.setAttribute("aria-expanded", open ? "true" : "false");
  }
  function translationTargetPayloadCodes() {
    return translationLanguageOptions()
      .filter(language => ctx.owners.translation.translationSelectedTargets.has(language.code))
      .map(language => language.code);
  }
  function scheduleTranslationConfiguration() {
    if (ctx.owners.translation.translationConfigureTimer) clearTimeout(ctx.owners.translation.translationConfigureTimer);
    ctx.owners.translation.translationConfigureTimer = setTimeout(() => {
      ctx.owners.translation.translationConfigureTimer = null;
      const targetLanguages = translationTargetPayloadCodes();
      post("/api/translation/configure", {target_languages:targetLanguages})
        .then(() => queueBrowserPreferredTranslationsForAllSources())
        .catch(error => log(`Translation configuration failed: ${error.message}`));
    }, 180);
  }
  function setTranslationTargetSelected(codeValue, selected) {
    const code = normalizedTranslationLanguageCode(codeValue);
    if (!code) return;
    const hadTargets = ctx.owners.translation.translationSelectedTargets.size > 0;
    if (selected) {
      if (ctx.owners.translation.translationSelectedTargets.size >= translationMaximumTargets()) return;
      ctx.owners.translation.translationSelectedTargets.add(code);
    } else {
      ctx.owners.translation.translationSelectedTargets.delete(code);
    }
    if (selected && !hadTargets && ctx.owners.translation.translationDisplayMode === "original") {
      ctx.owners.translation.translationDisplayMode = "single";
      storeSessionValue(translationDisplayModeStorageKey, ctx.owners.translation.translationDisplayMode);
    }
    if (!ctx.owners.translation.translationSelectedTargets.has(ctx.owners.translation.translationPrimaryTarget)) {
      ctx.owners.translation.translationPrimaryTarget = Array.from(ctx.owners.translation.translationSelectedTargets)[0] || "";
      storeSessionValue(translationPrimaryTargetStorageKey, ctx.owners.translation.translationPrimaryTarget);
    }
    if (!ctx.owners.translation.translationSelectedTargets.size && ctx.owners.translation.translationDisplayMode !== "original") {
      ctx.owners.translation.translationDisplayMode = "original";
      storeSessionValue(translationDisplayModeStorageKey, ctx.owners.translation.translationDisplayMode);
    }
    refreshTranslationPresentation({menu:true});
    scheduleTranslationConfiguration();
  }
  function initializeTranslationControls() {
    if (!translationControls) return;
    const validCodes = new Set(translationLanguageOptions().map(language => language.code));
    ctx.owners.translation.translationSelectedTargets = new Set(translationConfiguredTargets().filter(code => validCodes.has(code)));
    const configuredMode = String((translationConfig && translationConfig.display_mode) || "original").toLowerCase();
    const storedMode = storedSessionValue(translationDisplayModeStorageKey);
    ctx.owners.translation.translationDisplayMode = ["original", "single", "all"].includes(storedMode)
      ? storedMode
      : (["original", "single", "all"].includes(configuredMode) ? configuredMode : "original");
    const configuredPrimary = normalizedTranslationLanguageCode(translationConfig && translationConfig.primary_target);
    const storedPrimary = normalizedTranslationLanguageCode(storedSessionValue(translationPrimaryTargetStorageKey));
    ctx.owners.translation.translationPrimaryTarget = ctx.owners.translation.translationSelectedTargets.has(storedPrimary)
      ? storedPrimary
      : (ctx.owners.translation.translationSelectedTargets.has(configuredPrimary) ? configuredPrimary : (Array.from(ctx.owners.translation.translationSelectedTargets)[0] || ""));
    translationIncludeOriginalControl.checked = storedBooleanValue(
      translationIncludeOriginalStorageKey,
      Boolean(translationConfig && (translationConfig.include_original || translationConfig.show_original_with_translations))
    );
    ctx.owners.translation.translationLanguageLabelMode = normalizedTranslationLanguageLabelMode(
      storedSessionValue(translationLanguageLabelModeStorageKey)
    );
    if (!ctx.owners.translation.translationSelectedTargets.size) ctx.owners.translation.translationDisplayMode = "original";
    renderTranslationMenu();
    queueBrowserPreferredTranslationsForAllSources();
  }

  Object.assign(ctx.api, {applyTranslationCollection, applyTranslationEvent, browserPreferredTranslationEnabled, chromeTranslationUnavailable, chromeTranslatorForPair, clearDisplayedTranscript, clearUnsavedDetectedSpeakerDisplay, createChromeTranslator, createTranslationLanguageLabel, createTranslationLine, currentPlaybackElements, currentTranscriptClearBoundarySeconds, effectiveTranslationDisplayMode, forgetSentenceTranslations, initializeTranslationControls, isLiveProvisionalSpeaker, itemIsBeforeClearedTranscriptBoundary, logRejectedPlayback, normalizedTranslationLanguageCode, normalizedTranslationLanguageLabelMode, normalizedTranslationStatus, parseRevisionChain, playbackElementLabel, probabilityColor, probabilityDisplayLabel, pruneSpeakerFilterState, pushRevisionSpeaker, queueBrowserPreferredTranslation, queueBrowserPreferredTranslationsForAllSources, queueBrowserPreferredTranslationsForSource, refreshMediaElements, refreshTranslationMenuStatus, refreshTranslationPresentation, refreshTranslationRow, renderTranslationMenu, requestBackendTranslationFallback, resetTranscriptDisplay, revisionSpeakerBadge, revisionSpeakerId, runBrowserPreferredTranslation, scheduleTranslationConfiguration, scrollSentencesToBottom, selectedTranslationCodesForDisplay, sentenceRevisionLabel, setTranslationMenuOpen, setTranslationTargetSelected, speakerColor, speakerDisplayLabel, speakerIndex, speakerPanelName, speakerProbabilityKey, speakerTranscriptVisible, startSynchronizedPlaybackFromGesture, syncSpeakerSessionBaselines, transcriptRowSentenceKeys, translationActivityState, translationConfiguredTargets, translationFeatureAvailable, translationLanguageCodeLabel, translationLanguageFlagUrl, translationLanguageName, translationLanguageOptions, translationMaximumTargets, translationMenuSummaryText, translationProviderId, translationProviderLabel, translationProviderLicense, translationProviderNotice, translationSentenceKey, translationStateForRow, translationStateMap, translationStateMatchesRow, translationTargetCode, translationTargetPayloadCodes, unlockPlayback});
}
