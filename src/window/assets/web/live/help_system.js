const HELP_TOPICS = Object.freeze({
  overview: {
    title: "Help for WhoSpeaks Live",
    summary: "Point at a control for a short explanation, or focus it and press F1 for details.",
    detail: "Start by choosing an audio source, then start transcription. Live text and tentative speaker labels may change until a sentence is finalized. Use the Speakers panel to name voices, Transcript review to correct results, and Sessions to return to saved meetings.",
  },
  runtime: {
    title: "Transcription status",
    summary: "Shows whether WhoSpeaks is ready, preparing audio, transcribing, or finishing a run.",
    detail: "Ready means a source can be loaded or started. While transcription is running, live text is provisional. Stopping finishes pending work before the saved session is complete, so the final state can take a moment to appear.",
  },
  source: {
    title: "Audio source",
    summary: "Choose where the speech comes from before starting transcription.",
    detail: "YouTube and audio files use loaded media. Microphone captures your input device. Computer audio asks the browser to share a tab, window, or screen with audio. Computer audio + microphone combines both sources; Chrome may ask for permission for each source separately.",
  },
  start: {
    title: "Start transcription",
    summary: "Takes control of the current session and begins processing the selected source.",
    detail: "Only one browser can control a live session at a time. Starting may first request microphone or screen-audio permission. Live words and speaker hints are provisional; finalized sentences replace them after the sentence ends.",
  },
  stop: {
    title: "Stop transcription",
    summary: "Stops capture and lets pending transcription work finish.",
    detail: "Stopping does not discard the transcript. WhoSpeaks finalizes pending audio and updates the saved session. Wait for the run to report that it is finished before closing the app if the last sentence matters.",
  },
  capture: {
    title: "Input level and microphone gain",
    summary: "The level meter confirms that audible input is reaching the browser.",
    detail: "A moving meter proves that audio is arriving, but does not guarantee that speech is loud or clear enough to transcribe. Increase microphone gain only for consistently quiet input. If loud speech stays near the maximum, reduce gain to avoid distortion.",
  },
  transcript: {
    title: "Live transcript",
    summary: "Live rows can change until final transcription commits the sentence.",
    detail: "A sentence normally moves from provisional live text to final text after a pause. Final text is authoritative. Speaker assignment can still be revised later when stronger voice evidence arrives. Select rows to reassign a speaker or mark reviewed results as correct.",
  },
  follow_live: {
    title: "Follow live",
    summary: "Keeps the newest transcript row in view while speech arrives.",
    detail: "Turn this off when reviewing older text so new sentences do not pull the view back to the bottom. It changes scrolling only; it does not pause capture or transcription.",
  },
  review: {
    title: "Transcript review",
    summary: "Filter uncertain rows, select them, and correct or confirm their speaker labels.",
    detail: "Needs review surfaces rows that may deserve attention. Reassign changes selected rows to another speaker. Mark correct records that the current assignment was reviewed. Undo restores the most recent supported correction.",
  },
  transcript_settings: {
    title: "Transcript display settings",
    summary: "Changes what is shown without rewriting the underlying transcript.",
    detail: "Tags expose processing states, probabilities show model scores, and review hints call attention to uncertain results. Grouping consecutive same-speaker sentences is display-only: original sentence boundaries remain stored.",
  },
  translation: {
    title: "Translation display",
    summary: "Controls which committed-sentence translations appear beside the original text.",
    detail: "Translations begin after a sentence is finalized, so live draft words are not repeatedly translated. The original remains authoritative. Depending on launcher configuration, Chrome may translate locally or the configured fallback provider may receive the sentence.",
  },
  speakers: {
    title: "Speakers in this session",
    summary: "Speakers are meeting-local voice clusters; a saved Person can be recognized across meetings.",
    detail: "WhoSpeaks groups similar voices into Speakers for the current session. Naming a Speaker does not by itself create reusable voice recognition. Add or link a Person with an active voice sample when the same individual should be suggested in future sessions.",
  },
  add_person: {
    title: "Add a person",
    summary: "Create a reusable identity and attach a clean voice sample for future recognition.",
    detail: "Use speech with one person, little background noise, and no overlapping voices. A longer clean sample is generally more useful than a short noisy one. Automatic suggestions also require the Person to be included and recognition to be active.",
  },
  speaker_detection: {
    title: "Speaker detection",
    summary: "Controls when voice evidence creates a new Speaker or revises an existing label.",
    detail: "Higher new-speaker sensitivity separates similar voices more readily, but can split one person into multiple Speakers. Tentative UNKNOWN hints are previews only. Commit UNKNOWN later permits a final assignment after stronger evidence; reassign labeled speakers permits later evidence to correct an existing label.",
  },
  reset_speakers: {
    title: "Reset live speaker detection",
    summary: "Clears meeting-local speaker detection state and rebuilds it from new evidence.",
    detail: "Use this when the live speaker structure is no longer useful. This is more consequential than hiding a Speaker and can affect current-session assignments; it does not delete reusable People from the library.",
  },
  people: {
    title: "People and voice recognition",
    summary: "People are reusable identities that can be suggested across sessions.",
    detail: "Automatic suggestions need all three conditions: the Person is included, recognition is active, and at least one compatible voice sample is active. A configured Person is never forced onto a weak match.",
  },
  sessions: {
    title: "Saved sessions",
    summary: "Open, archive, restore, or select meetings for reports and questions.",
    detail: "A live control seat allows one browser to change a session at a time. Archiving removes a session from the Active list without deleting it. Delete is permanent. Selecting multiple sessions lets Ask search across their combined transcript evidence.",
  },
  ask: {
    title: "Ask sessions",
    summary: "Answers questions from transcript evidence in the selected session scope.",
    detail: "The scope chips show which sessions can support the answer. Check cited evidence for important claims, especially when an answer is marked Needs review. Processing location, privacy, and possible API cost depend on the provider configured in the launcher.",
  },
  insights: {
    title: "Meeting insights",
    summary: "Generates a structured report from a completed saved session.",
    detail: "Reports can summarize decisions, action items, and other meeting objects. Evidence links are the supporting transcript passages, not a guarantee that every generated interpretation is correct. Review consequential results against the transcript.",
  },
  evidence: {
    title: "Report evidence",
    summary: "Shows transcript passages used to support a generated insight.",
    detail: "Evidence makes generated results auditable. Open the relevant transcript context before relying on a decision, commitment, name, date, or number. Low confidence means the generated item deserves additional review.",
  },
  status_log: {
    title: "Status and technical details",
    summary: "Shows recent operational messages and errors from the live app.",
    detail: "Use the newest message first. Permission errors usually require choosing the audio source again. Connection or provider errors may require checking launcher Diagnostics. Technical messages are useful when reporting a problem, but a failed optional feature does not always stop transcription.",
  },
});

const TOPIC_SELECTORS = Object.freeze({
  runtime: [".status-pill"],
  source: ["#inputMode", "#sourceModeButton", "#sourceModeOptions", "#preset", "#source", "#load", "#fileDropZone", "#chooseAudioFile"],
  start: ["#start"],
  stop: ["#stop"],
  capture: ["#capturePanel", "#micGain"],
  transcript: ["#transcriptTitle", "#sentences", "#transcriptSearch"],
  follow_live: [".follow-live-toggle"],
  review: [".review-filter", "#undoCorrection", "#selectionToolbar", "#bulkCorrectionSpeaker", "#bulkReassign", "#bulkMarkCorrect"],
  transcript_settings: ["#transcriptSettings", "#transcriptSettingsPanel"],
  translation: ["#translationMenuButton", "#translationMenuPanel"],
  speakers: ["[data-speaker-tab='speakers']", "#speakerPanelTitle", "#speakerList"],
  add_person: ["#addReferenceSpeaker", "#referenceSpeakerForm", "#recordReference"],
  reset_speakers: ["#clearSpeakers"],
  sessions: ["[data-speaker-tab='sessions']", "[data-speaker-panel='sessions']", "#newRunSession", "#selectAllSessions", "#unselectAllSessions", "#archiveSelectedSessions", "#restoreSelectedSessions", "#deleteSelectedSessions"],
  ask: ["[data-speaker-tab='ask']", "[data-speaker-panel='ask']", "#askSelectedMeetings", "#meetingChatQuestion", "#meetingChatSend", "#meetingChatClear"],
  insights: ["[data-speaker-tab='intelligence']", "[data-speaker-panel='intelligence']", "#meetingIntelligenceGenerate"],
  speaker_detection: [".detection-settings", "#newSpeakerSensitivity", "#speakerRefinementUnknownTentative", "#speakerRefinementUnknownCommit", "#allowSpeakerReassignment"],
  people: [".people-tools", "#peopleList"],
  evidence: [".meeting-evidence-panel"],
  status_log: [".status-card"],
});

function closestHelpTarget(target) {
  return target && typeof target.closest === "function" ? target.closest("[data-help-topic], [data-help-summary]") : null;
}

function disabledReason(element, context) {
  if (!element || !element.disabled) return "";
  const explicit = String(element.dataset?.disabledHelp || "").trim();
  if (explicit) return explicit;
  switch (element.id) {
    case "start":
      if (context.stop && !context.stop.disabled) return "Transcription is already running.";
      return String(context.sessionBannerMessage?.textContent || "Start is unavailable while the app is preparing or another browser controls this session.");
    case "stop": return "Stop becomes available while transcription is running.";
    case "undoCorrection": return "There is no supported transcript correction to undo yet.";
    case "bulkReassign":
    case "bulkMarkCorrect": return "Select one or more transcript rows first.";
    case "meetingIntelligenceGenerate": return "Open a completed saved session before generating insights.";
    case "meetingChatSend": return "Choose a saved session before asking a question.";
    case "archiveSelectedSessions":
    case "restoreSelectedSessions":
    case "deleteSelectedSessions":
    case "askSelectedMeetings": return "Select one or more saved sessions first.";
    default: return "This action is not available in the current state.";
  }
}

function currentValue(element) {
  if (!element) return "";
  if (element.type === "checkbox") return element.checked ? "On" : "Off";
  if (element.tagName === "SELECT") return element.options?.[element.selectedIndex]?.text || element.value || "";
  if (element.type === "range") return element.value || "";
  return "";
}

export function installHelpSystem(context) {
  const helpButton = document.getElementById("helpButton");
  const drawer = document.getElementById("helpDrawer");
  const backdrop = document.getElementById("helpBackdrop");
  const closeButton = document.getElementById("helpClose");
  const title = document.getElementById("helpTitle");
  const summary = document.getElementById("helpSummary");
  const current = document.getElementById("helpCurrent");
  const detail = document.getElementById("helpDetail");
  const tooltip = document.getElementById("helpTooltip");
  let activeTopic = "overview";
  let activeElement = null;
  let tooltipTimer = null;
  let previousFocus = null;

  for (const [topic, selectors] of Object.entries(TOPIC_SELECTORS)) {
    for (const selector of selectors) {
      for (const element of document.querySelectorAll(selector)) {
        if (!element.dataset.helpTopic) element.dataset.helpTopic = topic;
        if (!element.hasAttribute?.("aria-description")) {
          element.setAttribute("aria-description", HELP_TOPICS[topic].summary);
        }
      }
    }
  }
  for (const element of document.querySelectorAll("[title]")) {
    const nativeHint = String(element.getAttribute?.("title") || "").trim();
    if (!nativeHint) continue;
    element.dataset.helpSummary = nativeHint;
    element.removeAttribute("title");
    if (!element.hasAttribute?.("aria-description")) element.setAttribute("aria-description", nativeHint);
  }
  for (const element of document.querySelectorAll("button[aria-label]")) {
    if (element.dataset.helpTopic || element.dataset.helpSummary) continue;
    element.dataset.helpSummary = String(element.getAttribute?.("aria-label") || "").trim();
  }

  const renderTopic = (topicId, element = null) => {
    const topic = HELP_TOPICS[topicId] || HELP_TOPICS.overview;
    activeTopic = HELP_TOPICS[topicId] ? topicId : "overview";
    activeElement = element;
    if (title) title.textContent = topic.title;
    if (summary) summary.textContent = topic.summary;
    if (detail) detail.textContent = topic.detail;
    if (current) {
      const value = currentValue(element);
      current.textContent = value ? `Current value: ${value}` : "";
      current.hidden = !value;
    }
  };

  const openHelp = (topicId = activeTopic, element = activeElement) => {
    previousFocus = document.activeElement || null;
    renderTopic(topicId, element);
    if (drawer) drawer.hidden = false;
    if (backdrop) backdrop.hidden = false;
    if (helpButton) helpButton.setAttribute("aria-expanded", "true");
    if (closeButton && typeof closeButton.focus === "function") closeButton.focus();
  };

  const closeHelp = () => {
    if (drawer) drawer.hidden = true;
    if (backdrop) backdrop.hidden = true;
    if (helpButton) helpButton.setAttribute("aria-expanded", "false");
    if (previousFocus && typeof previousFocus.focus === "function") previousFocus.focus();
    else if (helpButton && typeof helpButton.focus === "function") helpButton.focus();
  };

  const hideTooltip = () => {
    if (tooltipTimer !== null) clearTimeout(tooltipTimer);
    tooltipTimer = null;
    if (tooltip) tooltip.hidden = true;
  };

  const showTooltip = (element) => {
    if (!tooltip || !element) return;
    const topicId = String(element.dataset?.helpTopic || "overview");
    const topic = HELP_TOPICS[topicId] || HELP_TOPICS.overview;
    const reason = disabledReason(element, context);
    const hint = String(element.dataset?.helpSummary || topic.summary);
    tooltip.textContent = reason ? `${reason} ${hint}` : hint;
    tooltip.dataset.state = reason ? "disabled" : "help";
    tooltip.hidden = false;
    if (typeof element.getBoundingClientRect === "function") {
      const rect = element.getBoundingClientRect();
      const left = Math.max(12, Math.min(rect.left, (window.innerWidth || 1200) - 372));
      const below = rect.bottom + 9;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${below}px`;
    }
  };

  const rememberTarget = (event) => {
    const element = closestHelpTarget(event.target);
    if (!element) return;
    const nativeHint = String(element.getAttribute?.("title") || "").trim();
    if (nativeHint) {
      element.dataset.helpSummary = nativeHint;
      element.removeAttribute("title");
    }
    if (element.dataset.helpTopic) activeTopic = String(element.dataset.helpTopic);
    activeElement = element;
    hideTooltip();
    tooltipTimer = setTimeout(() => showTooltip(element), event.type === "focusin" ? 0 : 350);
  };

  const leaveTarget = (event) => {
    const element = closestHelpTarget(event.target);
    const next = closestHelpTarget(event.relatedTarget);
    if (element && element !== next) hideTooltip();
  };

  const onKeyDown = (event) => {
    if (event.key === "F1") {
      event.preventDefault();
      openHelp(activeTopic, activeElement);
    } else if (event.key === "Escape" && drawer && !drawer.hidden) {
      closeHelp();
    } else if (event.key === "Tab" && drawer && !drawer.hidden) {
      const controls = Array.from(drawer.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"))
        .filter(control => !control.disabled && !control.hidden);
      if (!controls.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  };

  const onHelpOpen = (event) => {
    const trigger = event.target && typeof event.target.closest === "function"
      ? event.target.closest("[data-help-open]")
      : null;
    if (!trigger) return;
    event.preventDefault();
    openHelp(String(trigger.dataset.helpOpen || "overview"), trigger);
  };

  helpButton?.addEventListener("click", () => openHelp(activeTopic, activeElement));
  closeButton?.addEventListener("click", closeHelp);
  backdrop?.addEventListener("click", closeHelp);
  document.addEventListener("mouseover", rememberTarget, true);
  document.addEventListener("mouseout", leaveTarget, true);
  document.addEventListener("focusin", rememberTarget, true);
  document.addEventListener("focusout", leaveTarget, true);
  document.addEventListener("click", onHelpOpen);
  window.addEventListener("keydown", onKeyDown);

  context.appResources.own(() => {
    hideTooltip();
    document.removeEventListener("mouseover", rememberTarget, true);
    document.removeEventListener("mouseout", leaveTarget, true);
    document.removeEventListener("focusin", rememberTarget, true);
    document.removeEventListener("focusout", leaveTarget, true);
    document.removeEventListener("click", onHelpOpen);
    window.removeEventListener("keydown", onKeyDown);
  });

  return {openHelp, closeHelp, renderTopic};
}

export {HELP_TOPICS, TOPIC_SELECTORS, disabledReason};
