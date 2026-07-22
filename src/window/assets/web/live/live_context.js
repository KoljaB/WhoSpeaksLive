import {createResourceRegistry, createStore} from "./app_store.js";

export class LiveSpeakerPresentationRegistry {
  constructor() {
    this.reset();
  }

  reset(runKey = "") {
    this.runKey = String(runKey || "");
    this.aliasGeneration = 0;
    this.finalToPublic = new Map();
    this.publicToFinal = new Map();
    this.migratedPairs = new Set();
  }

  toPublic(speakerId) {
    const value = String(speakerId || "").trim();
    return this.finalToPublic.get(value) || value;
  }

  toInternal(speakerId) {
    const value = String(speakerId || "").trim();
    return this.publicToFinal.get(value) || value;
  }

  apply(payload = {}) {
    const generation = Number(payload.alias_generation || 0);
    const finalId = String(payload.final_internal_speaker_id || "").trim();
    const publicId = String(payload.surviving_public_speaker_id || "").trim();
    if (!finalId || !publicId || finalId === publicId) return false;
    if (payload.retired) {
      if (!Number.isInteger(generation) || generation <= this.aliasGeneration) return false;
      if (this.finalToPublic.get(finalId) !== publicId || this.publicToFinal.get(publicId) !== finalId) return false;
      this.finalToPublic.delete(finalId);
      this.publicToFinal.delete(publicId);
      this.aliasGeneration = generation;
      return true;
    }
    const existingPair = this.finalToPublic.get(finalId) === publicId
      && this.publicToFinal.get(publicId) === finalId;
    if (existingPair) {
      this.aliasGeneration = Math.max(this.aliasGeneration, generation);
      return true;
    }
    if (!Number.isInteger(generation) || generation <= this.aliasGeneration) return false;
    if (this.finalToPublic.has(finalId) || this.publicToFinal.has(publicId)) return false;
    if (this.publicToFinal.has(finalId) || this.finalToPublic.has(publicId)) return false;
    this.finalToPublic.set(finalId, publicId);
    this.publicToFinal.set(publicId, finalId);
    this.aliasGeneration = generation;
    return true;
  }

  claimMigration(finalId, publicId, retired = false, generation = this.aliasGeneration) {
    const key = `${String(finalId || "")}\u0000${String(publicId || "")}\u0000${retired ? "retired" : "active"}\u0000${Number(generation || 0)}`;
    if (this.migratedPairs.has(key)) return false;
    this.migratedPairs.add(key);
    return true;
  }

  hydrate(finalToPublic = {}, publicToFinal = {}, generation = 0) {
    const nextFinal = new Map();
    const nextPublic = new Map();
    for (const [rawFinal, rawPublic] of Object.entries(finalToPublic || {})) {
      const finalId = String(rawFinal || "").trim();
      const publicId = String(rawPublic || "").trim();
      if (!finalId || !publicId || finalId === publicId) return false;
      if (nextFinal.has(finalId) || nextPublic.has(publicId)) return false;
      if (nextPublic.has(finalId) || nextFinal.has(publicId)) return false;
      nextFinal.set(finalId, publicId);
      nextPublic.set(publicId, finalId);
    }
    for (const [rawPublic, rawFinal] of Object.entries(publicToFinal || {})) {
      const publicId = String(rawPublic || "").trim();
      const finalId = String(rawFinal || "").trim();
      if (nextFinal.get(finalId) !== publicId || nextPublic.get(publicId) !== finalId) return false;
    }
    this.finalToPublic = nextFinal;
    this.publicToFinal = nextPublic;
    this.aliasGeneration = Math.max(this.aliasGeneration, Number(generation || nextFinal.size));
    return true;
  }

  mergeSnapshot(currentSpeakers, projectedSpeakers) {
    return mergePublicSpeakerSnapshot(currentSpeakers, projectedSpeakers, this);
  }

  stripTemporarySpeakers(speakers) {
    return stripTemporaryPublicSpeakers(speakers);
  }
}

export function mergePublicSpeakerSnapshot(currentSpeakers, projectedSpeakers, registry) {
  const projected = Array.isArray(projectedSpeakers) ? projectedSpeakers : [];
  const projectedIds = new Set(projected.map(speaker => speaker && speaker.id).filter(Boolean));
  const retained = (Array.isArray(currentSpeakers) ? currentSpeakers : []).filter(speaker => (
    String((speaker && speaker.id) || "").startsWith("LIVE_TRACKLET_")
    && speaker.source === "live_provisional"
    && speaker.presentation_aliased !== true
    && !String(speaker.internal_speaker_id || "")
    && !registry.publicToFinal.has(speaker.id)
    && !projectedIds.has(speaker.id)
  ));
  return [...projected, ...retained];
}

export function stripTemporaryPublicSpeakers(speakers) {
  return (Array.isArray(speakers) ? speakers : []).filter(
    speaker => !String((speaker && speaker.id) || "").startsWith("LIVE_TRACKLET_")
  );
}

export function createLiveAppContext() {
  const start = document.getElementById("start");
  const stop = document.getElementById("stop");
  const load = document.getElementById("load");
  const sessionBanner = document.getElementById("sessionBanner");
  const sessionBannerMessage = document.getElementById("sessionBannerMessage");
  const releaseSessionButton = document.getElementById("releaseSession");
  const preset = document.getElementById("preset");
  const state = document.getElementById("state");
  const languageSummary = document.getElementById("languageSummary");
  const languageFlag = document.getElementById("languageFlag");
  const languageName = document.getElementById("languageName");
  const source = document.getElementById("source");
  const mediaCard = document.getElementById("mediaCard");
  const sourceKind = document.getElementById("sourceKind");
  const sourceTitle = document.getElementById("sourceTitle");
  const sourceModeMenu = document.getElementById("sourceModeMenu");
  const sourceModeButton = document.getElementById("sourceModeButton");
  const sourceModeOptions = document.getElementById("sourceModeOptions");
  const sourceModeOptionButtons = Array.from(document.querySelectorAll(".source-mode-option"));
  const fileSourceControls = document.getElementById("fileSourceControls");
  const audioFileInput = document.getElementById("audioFileInput");
  const fileDropZone = document.getElementById("fileDropZone");
  const fileDropTitle = document.getElementById("fileDropTitle");
  const fileUploadStatus = document.getElementById("fileUploadStatus");
  const fastProcessingControl = document.getElementById("fastProcessingControl");
  const fastProcessing = document.getElementById("fastProcessing");
  const chooseAudioFileButton = document.getElementById("chooseAudioFile");
  const filePreviewName = document.getElementById("filePreviewName");
  const mediaTime = document.getElementById("mediaTime");
  const mediaCurrentTime = document.getElementById("mediaCurrentTime");
  const mediaDuration = document.getElementById("mediaDuration");
  const timelineFill = document.getElementById("timelineFill");
  const timelineThumb = document.getElementById("timelineThumb");
  const popoutMedia = document.getElementById("popoutMedia");
  const expandMedia = document.getElementById("expandMedia");
  const captureTitle = document.getElementById("captureTitle");
  const captureDescription = document.getElementById("captureDescription");
  const captureLevelFill = document.getElementById("captureLevelFill");
  const captureLevelText = document.getElementById("captureLevelText");
  const micGain = document.getElementById("micGain");
  const micGainValue = document.getElementById("micGainValue");
  const video = document.getElementById("video");
  const audio = document.getElementById("audio");
  const youtubeFrame = document.getElementById("youtubeFrame");
  const streamHint = document.getElementById("streamHint");
  const statusBox = document.getElementById("status");
  const statusCard = document.querySelector(".status-card");
  const sentences = document.getElementById("sentences");
  const transcriptPanel = document.querySelector(".transcript-panel");
  const transcriptTitle = document.getElementById("transcriptTitle");
  const followLive = document.getElementById("followLive");
  const transcriptSearch = document.getElementById("transcriptSearch");
  const clearTranscriptButton = document.getElementById("clearTranscript");
  const copyTranscriptButton = document.getElementById("copyTranscript");
  const downloadTranscriptButton = document.getElementById("downloadTranscript");
  const downloadTranscriptJsonButton = document.getElementById("downloadTranscriptJson");
  const transcriptSettingsButton = document.getElementById("transcriptSettings");
  const transcriptSettingsPanel = document.getElementById("transcriptSettingsPanel");
  const translationControls = document.getElementById("translationControls");
  const translationMenuButton = document.getElementById("translationMenuButton");
  const translationMenuSummary = document.getElementById("translationMenuSummary");
  const translationActivity = document.getElementById("translationActivity");
  const translationMenuPanel = document.getElementById("translationMenuPanel");
  const translationProvider = document.getElementById("translationProvider");
  const translationProviderAttribution = document.getElementById("translationProviderAttribution");
  const translationDisplayModeControl = document.getElementById("translationDisplayMode");
  const translationPrimaryField = document.getElementById("translationPrimaryField");
  const translationPrimaryTargetControl = document.getElementById("translationPrimaryTarget");
  const translationLanguageLabelModeControl = document.getElementById("translationLanguageLabelMode");
  const translationIncludeOriginalControl = document.getElementById("translationIncludeOriginal");
  const translationTargetList = document.getElementById("translationTargetList");
  const translationMenuHint = document.getElementById("translationMenuHint");
  const showTranscriptTags = document.getElementById("showTranscriptTags");
  const showTranscriptTime = document.getElementById("showTranscriptTime");
  const groupTranscriptTurns = document.getElementById("groupTranscriptTurns");
  const showTranscriptReviewHints = document.getElementById("showTranscriptReviewHints");
  const showTranscriptSpeechRate = document.getElementById("showTranscriptSpeechRate");
  const showTranscriptProbabilities = document.getElementById("showTranscriptProbabilities");
  const undoCorrectionButton = document.getElementById("undoCorrection");
  const selectionToolbar = document.getElementById("selectionToolbar");
  const selectionCount = document.getElementById("selectionCount");
  const bulkCorrectionSpeaker = document.getElementById("bulkCorrectionSpeaker");
  const bulkReassignButton = document.getElementById("bulkReassign");
  const bulkMarkCorrectButton = document.getElementById("bulkMarkCorrect");
  const clearSelectionButton = document.getElementById("clearSelection");
  const reviewFilterButtons = Array.from(document.querySelectorAll(".review-filter-button"));
  const inputMode = document.getElementById("inputMode");
  const newSpeakerSensitivity = document.getElementById("newSpeakerSensitivity");
  const newSpeakerSensitivityLabel = document.getElementById("newSpeakerSensitivityLabel");
  const autoRemoveEmptySpeakers = document.getElementById("autoRemoveEmptySpeakers");
  const speakerRefinementUnknownTentative = document.getElementById("speakerRefinementUnknownTentative");
  const speakerRefinementUnknownCommit = document.getElementById("speakerRefinementUnknownCommit");
  const allowSpeakerReassignment = document.getElementById("allowSpeakerReassignment");
  const loadSpeakerGroupButton = document.getElementById("loadSpeakerGroup");
  const saveSpeakerGroupButton = document.getElementById("saveSpeakerGroup");
  const saveCorrectedSpeakerGroupButton = document.getElementById("saveCorrectedSpeakerGroup");
  const speakerGroupFile = document.getElementById("speakerGroupFile");
  const peopleList = document.getElementById("peopleList");
  const speakerCount = document.getElementById("speakerCount");
  const speakerCountNumber = document.getElementById("speakerCountNumber");
  const speakerCountLabel = document.getElementById("speakerCountLabel");
  const speakerPanelTitle = document.getElementById("speakerPanelTitle");
  const speakerList = document.getElementById("speakerList");
  const speakerEditorDock = document.getElementById("speakerEditorDock");
  const clearSpeakersButton = document.getElementById("clearSpeakers");
  const addReferenceSpeakerButton = document.getElementById("addReferenceSpeaker");
  const manualSpeakerComposer = document.getElementById("manualSpeakerComposer");
  const manualSpeakerName = document.getElementById("manualSpeakerName");
  const manualSpeakerReferenceDock = document.getElementById("manualSpeakerReferenceDock");
  const speakerTabButtons = Array.from(document.querySelectorAll(".speaker-tab"));
  const speakerTabPanels = Array.from(document.querySelectorAll(".speaker-tab-panel"));
  const sessionList = document.getElementById("sessionList");
  const newRunSessionButton = document.getElementById("newRunSession");
  const sessionFilterButtons = Array.from(document.querySelectorAll(".sessions-filter-button"));
  const selectAllSessionsButton = document.getElementById("selectAllSessions");
  const unselectAllSessionsButton = document.getElementById("unselectAllSessions");
  const archiveSelectedSessionsButton = document.getElementById("archiveSelectedSessions");
  const restoreSelectedSessionsButton = document.getElementById("restoreSelectedSessions");
  const deleteSelectedSessionsButton = document.getElementById("deleteSelectedSessions");
  const sessionSelectionStatus = document.getElementById("sessionSelectionStatus");
  const askSelectedMeetingsButton = document.getElementById("askSelectedMeetings");
  const meetingChatTitle = document.getElementById("meetingChatTitle");
  const meetingChatStatus = document.getElementById("meetingChatStatus");
  const meetingChatClear = document.getElementById("meetingChatClear");
  const meetingChatScope = document.getElementById("meetingChatScope");
  const meetingChatMessages = document.getElementById("meetingChatMessages");
  const meetingChatProgress = document.getElementById("meetingChatProgress");
  const meetingChatProgressText = document.getElementById("meetingChatProgressText");
  const meetingChatProgressBar = document.getElementById("meetingChatProgressBar");
  const meetingChatProgressPercent = document.getElementById("meetingChatProgressPercent");
  const meetingChatProgressElapsed = document.getElementById("meetingChatProgressElapsed");
  const meetingChatForm = document.getElementById("meetingChatForm");
  const meetingChatQuestion = document.getElementById("meetingChatQuestion");
  const meetingChatSend = document.getElementById("meetingChatSend");
  const meetingIntelligenceGenerate = document.getElementById("meetingIntelligenceGenerate");
  const meetingIntelligenceStatus = document.getElementById("meetingIntelligenceStatus");
  const meetingIntelligenceSummary = document.getElementById("meetingIntelligenceSummary");
  const meetingIntelligenceStats = document.getElementById("meetingIntelligenceStats");
  const meetingIntelligenceObjects = document.getElementById("meetingIntelligenceObjects");
  const meetingIntelligenceEvidence = document.getElementById("meetingIntelligenceEvidence");
  const referenceSpeakerForm = document.getElementById("referenceSpeakerForm");
  const referenceSpeakerFile = document.getElementById("referenceSpeakerFile");
  const recordReferenceButton = document.getElementById("recordReference");
  const recordReferenceButtonLabel = recordReferenceButton.querySelector("span");
  const referenceRecordSeconds = document.getElementById("referenceRecordSeconds");
  const bootstrapElement = document.getElementById("bootstrap-data");
  const bootstrap = bootstrapElement ? JSON.parse(bootstrapElement.textContent || "{}") : {};
  const speakerColors = bootstrap.speaker_colors || [];
  const initialSource = bootstrap.source || "";
  const presetVideos = bootstrap.preset_videos || [];
  const speakerSensitivityConfig = bootstrap.new_speaker_sensitivity || {};
  const speakerRefinementConfig = bootstrap.speaker_refinement || {};
  const liveSpeakerConfig = bootstrap.live_speaker || {};
  const languageConfig = bootstrap.language || {};
  const translationConfig = bootstrap.translation || {};
  const sessionLeaseEnabled = liveSpeakerConfig.session_lease_enabled !== false;
  const initialSpeakerLibrary = bootstrap.speaker_library || {group_name:"", groups:[], speakers:[]};
  const appStore = createStore({
    run: {status: "idle", generation: 0},
    media: {source: initialSource, version: 0},
    transcript: {rows: [], selected: [], filter: "all"},
    speakers: initialSpeakerLibrary,
    translation: translationConfig,
    sessions: {items: [], selected: [], filter: "active"},
    reports: {current: null, selectedObjectId: "", busy: false},
    chat: {scope: null, busy: false, jobId: ""},
  });
  const appResources = createResourceRegistry();
  const svgNamespace = "http://www.w3.org/2000/svg";
  const createSpeakerOptionValue = "__create_speaker__";
  const targetCaptureSampleRate = 16000;
  const captureStartRmsThreshold = 0.003;
  const capturePreRollSeconds = 0.7;
  const audioUploadExtensions = new Set(["aac", "aif", "aiff", "flac", "m4a", "mp3", "mp4", "oga", "ogg", "opus", "wav", "webm"]);
  const playbackClockSlackSeconds = 3.0;
  const transcriptGroupTurnsStorageKey = "whospeaks.demo.group_transcript_turns.v3";
  const transcriptReviewHintsStorageKey = "whospeaks.demo.show_transcript_review_hints";
  const translationDisplayModeStorageKey = "whospeaks.demo.translation_display_mode.v1";
  const translationPrimaryTargetStorageKey = "whospeaks.demo.translation_primary_target.v1";
  const translationIncludeOriginalStorageKey = "whospeaks.demo.translation_include_original.v1";
  const translationLanguageLabelModeStorageKey = "whospeaks.demo.translation_language_label_mode.v1";
  const realtimeSettleRemovalDelayMs = 1400;
  const sessionClientIdStorageKey = "whospeaks.demo.client_id";
  const sessionTokenStorageKey = "whospeaks.demo.session_token";
  const fastProcessingStorageKey = "whospeaks.demo.fast_processing.v1";
  const autoRemoveEmptySpeakersStorageKey = "whospeaks.demo.auto_remove_empty_speakers.v1";

  const owners = {
    presentation: new LiveSpeakerPresentationRegistry(),
    capture: {
      es: null,
      playbackTimer: null,
      playbackClockStartedAt: null,
      videoSyncTimer: null,
      currentRealtimeGeneration: 0,
      mediaVersion: 0,
      browserStreamMode: false,
      resumePlaybackPending: false,
      browserStreamPrepared: false,
      browserStreamPreparedUrl: "",
      localAudioFileName: "",
      localAudioFileSize: 0,
      audioUploadInProgress: false,
      captureSourceKind: "display",
      captureStream: null,
      captureStreams: [],
      captureAudioContext: null,
      captureSourceNode: null,
      captureSourceNodes: [],
      captureProcessor: null,
      captureSilentGain: null,
      captureMicGainNode: null,
      captureSendQueue: Promise.resolve(),
      capturePending: [],
      capturePendingSamples: 0,
      captureAudioStarted: false,
      capturePreRoll: [],
      capturePreRollSamples: 0,
      speakerSensitivityDirty: false,
    },
    speakers: {
      speakerLibraryState: initialSpeakerLibrary || {group_name:"", groups:[], speakers:[]},
      speakerNames: {},
      speakerSessionBaselineSentenceCounts: {},
      speakerSessionBaselineSpeakingSeconds: {},
      renderedSpeakerSentenceCounts: {},
      renderedSpeakerSpeakingSeconds: {},
      hasRenderedFinalSentenceRows: false,
      fastSpeakerPanelStats: {},
      fastSpeakerPanelLastRight: null,
      autoRemoveEmptySpeakerTimer: null,
      autoRemoveEmptySpeakerRequestPending: false,
      emptySpeakerFirstSeenAt: new Map(),
      soloSpeakerIds: new Set(),
      mutedSpeakerIds: new Set(),
      followLiveEnabled: true,
      transcriptSearchText: "",
      transcriptReviewFilter: "all",
    },
    translation: {
      translationDisplayMode: "original",
      translationPrimaryTarget: "",
      translationLanguageLabelMode: "flag_name",
      translationSelectedTargets: new Set(),
      translationStatesBySentence: new Map(),
      translationConfigureTimer: null,
      browserTranslationQueue: Promise.resolve(),
      browserTranslationJobs: new Set(),
      browserTranslationSourcesBySentence: new Map(),
      chromeTranslatorsByPair: new Map(),
    },
    transcript: {
      hasUndoableCorrection: false,
      selectedTranscriptRowIndexes: new Set(),
      lastSelectedTranscriptRowIndex: "",
      transcriptClearBeforeSeconds: 0,
      currentLiveSpeakerId: "",
      transcriptLiveSpeakerId: "",
      fallbackLiveSpeakerId: "",
      fallbackLiveSpeakerUntilMs: 0,
      fallbackLiveSpeakerExpiryTimer: null,
      fallbackLiveSpeakerClearTimer: null,
      transcriptLiveSpeakerExpiryTimer: null,
      liveSpeakerTimeline: [],
      transcriptLiveSpeakerOverrideId: "",
      browserLiveObservationTimer: null,
      browserLiveObservationBuffer: [],
      browserLiveObservationStarted: false,
      browserLiveObservationPosting: false,
      browserLiveObservationPostChain: Promise.resolve(),
      browserLiveObservationBatchSequence: 0,
      browserLiveObservationSampleSequence: 0,
    },
    reference: {
      referenceRecordStream: null,
      referenceRecordContext: null,
      referenceRecordSource: null,
      referenceRecordProcessor: null,
      referenceRecordSilentGain: null,
      referenceRecordChunks: [],
      referenceRecordSamples: 0,
      referenceRecordSampleRate: targetCaptureSampleRate,
      referenceRecordStartedAt: 0,
      referenceRecordTimer: null,
      referenceRecordPending: false,
      editingSpeakerId: "",
      pendingSpeakerNameFocusId: "",
      manualSpeakerComposerOpen: false,
      pendingManualSpeakerNameFocus: false,
      voiceSamplePersonId: "",
    },
    lease: {
      sessionClientId: "",
      sessionToken: "",
      sessionState: {active:false, is_owner:false, running:false, completed:false},
      sessionHeartbeatTimer: null,
      sessionStatusTimer: null,
      sessionCompletionReleaseTimer: null,
    },
    sessions: {
      savedSessions: [],
      selectedSavedSessionIds: new Set(),
      savedSessionBulkActionBusy: false,
      savedSessionFilter: "active",
      openedSavedSessionId: "",
      draftSavedSessionId: "",
      openSessionMenuId: "",
      editingSessionTitleId: "",
      pendingSessionTitleFocusId: "",
      savedSessionRefreshTimer: null,
      savedSessionAutoRefreshTimer: null,
    },
    reports: {
      meetingIntelligenceReport: null,
      meetingIntelligenceSelectedObjectId: "",
      meetingIntelligenceBusy: false,
    },
    chat: {
      scope: null,
      busy: false,
      jobId: "",
      scopeRefreshTimer: null,
      jobPollTimer: null,
      jobElapsedTimer: null,
      jobStartedAt: 0,
    },
  };

  if (statusCard && window.matchMedia("(max-width: 900px)").matches) {
    statusCard.open = false;
  }

  return {
    owners,
    api: Object.create(null),
    activators: [],
    askSelectedMeetingsButton, meetingChatTitle, meetingChatStatus, meetingChatClear, meetingChatScope, meetingChatMessages, meetingChatProgress, meetingChatProgressText, meetingChatProgressBar, meetingChatProgressPercent, meetingChatProgressElapsed, meetingChatForm, meetingChatQuestion, meetingChatSend,
    start, stop, load, sessionBanner, sessionBannerMessage, releaseSessionButton, preset, state, languageSummary, languageFlag, languageName, source, mediaCard, sourceKind, sourceTitle, sourceModeMenu, sourceModeButton, sourceModeOptions, sourceModeOptionButtons, fileSourceControls, audioFileInput, fileDropZone, fileDropTitle, fileUploadStatus, fastProcessingControl, fastProcessing, chooseAudioFileButton, filePreviewName, mediaTime, mediaCurrentTime, mediaDuration, timelineFill, timelineThumb, popoutMedia, expandMedia, captureTitle, captureDescription, captureLevelFill, captureLevelText, micGain, micGainValue, video, audio, youtubeFrame, streamHint, statusBox, statusCard, sentences, transcriptPanel, transcriptTitle, followLive, transcriptSearch, clearTranscriptButton, copyTranscriptButton, downloadTranscriptButton, downloadTranscriptJsonButton, transcriptSettingsButton, transcriptSettingsPanel, translationControls, translationMenuButton, translationMenuSummary, translationActivity, translationMenuPanel, translationProvider, translationProviderAttribution, translationDisplayModeControl, translationPrimaryField, translationPrimaryTargetControl, translationLanguageLabelModeControl, translationIncludeOriginalControl, translationTargetList, translationMenuHint, showTranscriptTags, showTranscriptTime, groupTranscriptTurns, showTranscriptReviewHints, showTranscriptSpeechRate, showTranscriptProbabilities, undoCorrectionButton, selectionToolbar, selectionCount, bulkCorrectionSpeaker, bulkReassignButton, bulkMarkCorrectButton, clearSelectionButton, reviewFilterButtons, inputMode, newSpeakerSensitivity, newSpeakerSensitivityLabel, autoRemoveEmptySpeakers, speakerRefinementUnknownTentative, speakerRefinementUnknownCommit, allowSpeakerReassignment, loadSpeakerGroupButton, saveSpeakerGroupButton, saveCorrectedSpeakerGroupButton, speakerGroupFile, peopleList, speakerCount, speakerCountNumber, speakerCountLabel, speakerPanelTitle, speakerList, speakerEditorDock, clearSpeakersButton, addReferenceSpeakerButton, manualSpeakerComposer, manualSpeakerName, manualSpeakerReferenceDock, speakerTabButtons, speakerTabPanels, sessionList, newRunSessionButton, sessionFilterButtons, selectAllSessionsButton, unselectAllSessionsButton, archiveSelectedSessionsButton, restoreSelectedSessionsButton, deleteSelectedSessionsButton, sessionSelectionStatus, meetingIntelligenceGenerate, meetingIntelligenceStatus, meetingIntelligenceSummary, meetingIntelligenceStats, meetingIntelligenceObjects, meetingIntelligenceEvidence, referenceSpeakerForm, referenceSpeakerFile, recordReferenceButton, recordReferenceButtonLabel, referenceRecordSeconds, bootstrapElement, bootstrap, speakerColors, initialSource, presetVideos, speakerSensitivityConfig, speakerRefinementConfig, liveSpeakerConfig, languageConfig, translationConfig, sessionLeaseEnabled, initialSpeakerLibrary, appStore, appResources, svgNamespace, createSpeakerOptionValue, targetCaptureSampleRate, captureStartRmsThreshold, capturePreRollSeconds, audioUploadExtensions, playbackClockSlackSeconds, transcriptGroupTurnsStorageKey, transcriptReviewHintsStorageKey, translationDisplayModeStorageKey, translationPrimaryTargetStorageKey, translationIncludeOriginalStorageKey, translationLanguageLabelModeStorageKey, realtimeSettleRemovalDelayMs, sessionClientIdStorageKey, sessionTokenStorageKey, fastProcessingStorageKey, autoRemoveEmptySpeakersStorageKey,
  };
}
