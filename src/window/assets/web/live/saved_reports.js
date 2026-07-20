export function installSavedReports(ctx) {
  const {archiveSelectedSessionsButton, audio, deleteSelectedSessionsButton, liveSpeakerConfig, mediaCurrentTime, mediaDuration, mediaTime, meetingIntelligenceEvidence, meetingIntelligenceGenerate, meetingIntelligenceObjects, meetingIntelligenceStats, meetingIntelligenceStatus, meetingIntelligenceSummary, playbackClockSlackSeconds, restoreSelectedSessionsButton, selectAllSessionsButton, sentences, sessionFilterButtons, sessionList, sessionSelectionStatus, source, sourceKind, sourceTitle, speakerCount, speakerList, start, state, stop, svgNamespace, timelineFill, timelineThumb, unselectAllSessionsButton, video} = ctx;
  const applyTranslationCollection = (...args) => ctx.api.applyTranslationCollection(...args), clockLabel = (...args) => ctx.api.clockLabel(...args), findFinalSentenceRow = (...args) => ctx.api.findFinalSentenceRow(...args), log = (...args) => ctx.api.log(...args), post = (...args) => ctx.api.post(...args), recomputeRenderedSpeakerSentenceCounts = (...args) => ctx.api.recomputeRenderedSpeakerSentenceCounts(...args), refreshSpeakerPanelSentenceCounts = (...args) => ctx.api.refreshSpeakerPanelSentenceCounts(...args), refreshTranscriptVisibility = (...args) => ctx.api.refreshTranscriptVisibility(...args), renderSentence = (...args) => ctx.api.renderSentence(...args), renderSpeakerPanel = (...args) => ctx.api.renderSpeakerPanel(...args), resetTranscriptDisplay = (...args) => ctx.api.resetTranscriptDisplay(...args), secondsLabel = (...args) => ctx.api.secondsLabel(...args), setSourceControlsDisabled = (...args) => ctx.api.setSourceControlsDisabled(...args), setState = (...args) => ctx.api.setState(...args), setTranscriptTitleLive = (...args) => ctx.api.setTranscriptTitleLive(...args), setTranscriptTitleSaved = (...args) => ctx.api.setTranscriptTitleSaved(...args), speakerDisplayLabel = (...args) => ctx.api.speakerDisplayLabel(...args), stopBrowserAudioCapture = (...args) => ctx.api.stopBrowserAudioCapture(...args), updateNewRunButtonState = (...args) => ctx.api.updateNewRunButtonState(...args), updateSpeakerState = (...args) => ctx.api.updateSpeakerState(...args);
  function savedSessionTitle(sessionData) {
    const summary = (sessionData && sessionData.summary) || {};
    const manifest = (sessionData && sessionData.manifest) || {};
    return summary.title || manifest.title || "Saved session";
  }
  function savedSessionDisplayTitle(item) {
    return (item && item.title) || "Saved session";
  }
  function savedSessionRows(sessionData) {
    return Array.isArray(sessionData && sessionData.transcript_rows) ? sessionData.transcript_rows : [];
  }
  function savedSessionDurationLabel(seconds) {
    const value = Number(seconds || 0);
    return value > 0 ? clockLabel(value) : "00:00";
  }
  function savedSessionDate(value) {
    const time = Date.parse(value || "");
    return Number.isFinite(time) ? new Date(time) : null;
  }
  function sameLocalDate(left, right) {
    return left && right
      && left.getFullYear() === right.getFullYear()
      && left.getMonth() === right.getMonth()
      && left.getDate() === right.getDate();
  }
  function savedSessionDatePart(date) {
    return date.toLocaleDateString([], {month:"short", day:"numeric"});
  }
  function savedSessionTimePart(date) {
    return date.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  }
  function savedSessionTimeRangeLabel(item) {
    const started = savedSessionDate((item && (item.started_at || item.created_at)) || "");
    if (!started) return "";
    const ended = savedSessionDate((item && (item.ended_at || item.updated_at)) || "");
    const startLabel = `${savedSessionDatePart(started)} ${savedSessionTimePart(started)}`;
    if (!ended || ended < started || Math.abs(ended.getTime() - started.getTime()) < 60000) {
      return startLabel;
    }
    if (sameLocalDate(started, ended)) {
      return `${startLabel}-${savedSessionTimePart(ended)}`;
    }
    return `${startLabel} - ${savedSessionDatePart(ended)} ${savedSessionTimePart(ended)}`;
  }
  function savedSessionUpdatedLabel(value) {
    const time = Date.parse(value || "");
    if (!Number.isFinite(time)) return "";
    const seconds = Math.max(0, (Date.now() - time) / 1000);
    if (seconds < 60) return "Just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    return new Date(time).toLocaleDateString([], {month:"short", day:"numeric"});
  }
  function createSessionMenuIcon() {
    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "currentColor");
    svg.setAttribute("aria-hidden", "true");
    [["12", "5"], ["12", "12"], ["12", "19"]].forEach(([cx, cy]) => {
      const circle = document.createElementNS(svgNamespace, "circle");
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      circle.setAttribute("r", "1.7");
      svg.appendChild(circle);
    });
    return svg;
  }
  function selectedSavedSessions() {
    return ctx.owners.sessions.savedSessions.filter(item => ctx.owners.sessions.selectedSavedSessionIds.has(item.id));
  }
  function syncSavedSessionSelectionControls() {
    const selected = selectedSavedSessions();
    const selectedCount = selected.length;
    const allSelected = ctx.owners.sessions.savedSessions.length > 0 && selectedCount === ctx.owners.sessions.savedSessions.length;
    sessionSelectionStatus.textContent = `${selectedCount} selected`;
    selectAllSessionsButton.disabled = ctx.owners.sessions.savedSessionBulkActionBusy || !ctx.owners.sessions.savedSessions.length || allSelected;
    unselectAllSessionsButton.disabled = ctx.owners.sessions.savedSessionBulkActionBusy || !selectedCount;
    archiveSelectedSessionsButton.disabled = ctx.owners.sessions.savedSessionBulkActionBusy || !selected.some(item => !item.archived);
    restoreSelectedSessionsButton.disabled = ctx.owners.sessions.savedSessionBulkActionBusy || !selected.some(item => item.archived);
    deleteSelectedSessionsButton.disabled = ctx.owners.sessions.savedSessionBulkActionBusy || !selectedCount;
    const busyReason = ctx.owners.sessions.savedSessionBulkActionBusy ? "Wait for the current session action to finish." : "";
    selectAllSessionsButton.dataset.disabledHelp = busyReason || (!ctx.owners.sessions.savedSessions.length ? "There are no sessions in this view." : (allSelected ? "All visible sessions are already selected." : ""));
    unselectAllSessionsButton.dataset.disabledHelp = busyReason || (!selectedCount ? "No sessions are selected." : "");
    archiveSelectedSessionsButton.dataset.disabledHelp = busyReason || (!selected.some(item => !item.archived) ? "Select at least one active session to archive." : "");
    restoreSelectedSessionsButton.dataset.disabledHelp = busyReason || (!selected.some(item => item.archived) ? "Select at least one archived session to restore." : "");
    deleteSelectedSessionsButton.dataset.disabledHelp = busyReason || (!selectedCount ? "Select one or more sessions before deleting." : "");
  }
  function selectAllSavedSessions() {
    ctx.owners.sessions.savedSessions.forEach(item => {
      if (item.id) ctx.owners.sessions.selectedSavedSessionIds.add(item.id);
    });
    renderSavedSessions();
  }
  function clearSavedSessionSelection() {
    ctx.owners.sessions.selectedSavedSessionIds.clear();
    renderSavedSessions();
  }
  function renderSavedSessions() {
    if (!sessionList) return;
    const availableSessionIds = new Set(ctx.owners.sessions.savedSessions.map(item => item.id).filter(Boolean));
    ctx.owners.sessions.selectedSavedSessionIds.forEach(sessionId => {
      if (!availableSessionIds.has(sessionId)) ctx.owners.sessions.selectedSavedSessionIds.delete(sessionId);
    });
    updateNewRunButtonState();
    sessionFilterButtons.forEach(button => {
      const active = button.dataset.sessionFilter === ctx.owners.sessions.savedSessionFilter;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    syncSavedSessionSelectionControls();
    if (ctx.api.scheduleMeetingChatScopeRefresh) ctx.api.scheduleMeetingChatScopeRefresh();
    sessionList.textContent = "";
    if (ctx.owners.sessions.editingSessionTitleId && !ctx.owners.sessions.savedSessions.some(item => item.id === ctx.owners.sessions.editingSessionTitleId)) {
      ctx.owners.sessions.editingSessionTitleId = "";
      ctx.owners.sessions.pendingSessionTitleFocusId = "";
    }
    if (!ctx.owners.sessions.savedSessions.length) {
      const empty = document.createElement("div");
      empty.className = "session-empty";
      empty.textContent = ctx.owners.sessions.savedSessionFilter === "archived" ? "No archived sessions" : "No saved sessions";
      sessionList.appendChild(empty);
      return;
    }
    ctx.owners.sessions.savedSessions.forEach(item => {
      const row = document.createElement("div");
      row.className = "session-row";
      row.classList.toggle("open", item.id === ctx.owners.sessions.openedSavedSessionId);
      row.classList.toggle("archived", Boolean(item.archived));
      row.classList.toggle("selected", ctx.owners.sessions.selectedSavedSessionIds.has(item.id));
      row.classList.toggle("editing", item.id === ctx.owners.sessions.editingSessionTitleId);
      row.dataset.sessionId = item.id || "";

      const main = document.createElement("div");
      main.className = "session-row-main";
      const selector = document.createElement("input");
      selector.type = "checkbox";
      selector.className = "session-row-select";
      selector.checked = ctx.owners.sessions.selectedSavedSessionIds.has(item.id);
      selector.disabled = ctx.owners.sessions.savedSessionBulkActionBusy;
      selector.setAttribute("aria-label", `Select ${savedSessionDisplayTitle(item)}`);
      selector.addEventListener("click", event => event.stopPropagation());
      selector.addEventListener("change", () => {
        if (selector.checked) ctx.owners.sessions.selectedSavedSessionIds.add(item.id);
        else ctx.owners.sessions.selectedSavedSessionIds.delete(item.id);
        renderSavedSessions();
      });
      const copy = document.createElement("div");
      copy.className = "session-row-copy";
      let title;
      if (item.id === ctx.owners.sessions.editingSessionTitleId) {
        title = document.createElement("input");
        title.className = "session-row-title-input";
        title.type = "text";
        title.value = savedSessionDisplayTitle(item);
        title.setAttribute("aria-label", "Session name");
        title.setAttribute("autocomplete", "off");
        title.addEventListener("click", event => event.stopPropagation());
        title.addEventListener("keydown", event => {
          if (event.key === "Enter") {
            event.preventDefault();
            title.blur();
          } else if (event.key === "Escape") {
            event.preventDefault();
            title.dataset.cancelled = "1";
            ctx.owners.sessions.editingSessionTitleId = "";
            ctx.owners.sessions.pendingSessionTitleFocusId = "";
            renderSavedSessions();
          }
        });
        title.addEventListener("blur", () => {
          if (title.dataset.cancelled === "1") return;
          commitSessionTitleInput(item, title);
        });
      } else {
        title = document.createElement("div");
        title.className = "session-row-title";
        title.textContent = savedSessionDisplayTitle(item);
        title.setAttribute("role", "button");
        title.tabIndex = 0;
        title.title = "Rename session";
        title.addEventListener("click", event => {
          event.stopPropagation();
          setEditingSessionTitle(item.id || "");
        });
        title.addEventListener("keydown", event => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            setEditingSessionTitle(item.id || "");
          }
        });
      }
      copy.appendChild(title);

      const meta = document.createElement("div");
      meta.className = "session-row-meta";
      const speakerCount = Number(item.speaker_count || 0);
      const transcriptDuration = Number(item.transcript_duration_seconds ?? item.duration_seconds ?? 0);
      const dateLabel = savedSessionTimeRangeLabel(item);
      const detailParts = [
        `${savedSessionDurationLabel(transcriptDuration)} transcribed`,
        `${speakerCount} ${speakerCount === 1 ? "speaker" : "speakers"}`,
      ].filter(Boolean);
      if (dateLabel) {
        const dateLine = document.createElement("span");
        dateLine.className = "session-row-meta-date";
        dateLine.textContent = dateLabel;
        meta.appendChild(dateLine);
      }
      const detailLine = document.createElement("span");
      detailLine.className = "session-row-meta-details";
      detailLine.textContent = detailParts.join(" - ");
      meta.appendChild(detailLine);
      copy.appendChild(meta);

      const status = document.createElement("div");
      status.className = "session-row-status";
      const dot = document.createElement("span");
      dot.className = `session-row-dot${item.archived ? " muted" : ""}`;
      dot.setAttribute("aria-hidden", "true");
      const statusText = document.createElement("span");
      const updated = savedSessionUpdatedLabel(item.updated_at);
      statusText.textContent = `${item.status_label || "Saved"}${updated ? ` - ${updated}` : ""}`;
      status.appendChild(dot);
      status.appendChild(statusText);
      copy.appendChild(status);

      const menuButton = document.createElement("button");
      menuButton.type = "button";
      menuButton.className = "session-menu-button";
      menuButton.title = "Session menu";
      menuButton.setAttribute("aria-label", `Session menu for ${item.title || "saved session"}`);
      menuButton.setAttribute("aria-expanded", ctx.owners.sessions.openSessionMenuId === item.id ? "true" : "false");
      menuButton.appendChild(createSessionMenuIcon());
      menuButton.addEventListener("click", event => {
        event.stopPropagation();
        ctx.owners.sessions.openSessionMenuId = ctx.owners.sessions.openSessionMenuId === item.id ? "" : item.id;
        renderSavedSessions();
      });
      main.appendChild(selector);
      main.appendChild(copy);
      main.appendChild(menuButton);
      row.appendChild(main);

      const actions = document.createElement("div");
      actions.className = "session-row-actions";
      const openButton = document.createElement("button");
      openButton.type = "button";
      openButton.className = "session-open-button";
      openButton.textContent = "Open";
      openButton.addEventListener("click", () => openSavedSession(item.id));
      actions.appendChild(openButton);

      const archiveButton = document.createElement("button");
      archiveButton.type = "button";
      archiveButton.textContent = item.archived ? "Restore" : "Archive";
      archiveButton.addEventListener("click", () => {
        if (item.archived) {
          restoreSavedSession(item.id);
        } else {
          archiveSavedSession(item.id);
        }
      });
      actions.appendChild(archiveButton);
      row.appendChild(actions);

      const menu = document.createElement("div");
      menu.className = "session-row-menu";
      menu.hidden = ctx.owners.sessions.openSessionMenuId !== item.id;
      const renameButton = document.createElement("button");
      renameButton.type = "button";
      renameButton.textContent = "Rename";
      renameButton.addEventListener("click", () => setEditingSessionTitle(item.id || ""));
      menu.appendChild(renameButton);
      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "danger";
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", () => deleteSavedSession(item));
      menu.appendChild(deleteButton);
      row.appendChild(menu);
      sessionList.appendChild(row);
      if (ctx.owners.sessions.pendingSessionTitleFocusId === item.id && title instanceof HTMLInputElement) {
        ctx.owners.sessions.pendingSessionTitleFocusId = "";
        requestAnimationFrame(() => {
          title.focus();
          title.select();
        });
      }
    });
  }
  function setEditingSessionTitle(sessionId) {
    const requestedId = sessionId || "";
    if (!requestedId) return;
    ctx.owners.sessions.editingSessionTitleId = requestedId;
    ctx.owners.sessions.pendingSessionTitleFocusId = requestedId;
    ctx.owners.sessions.openSessionMenuId = "";
    renderSavedSessions();
  }
  async function fetchSavedSessions() {
    if (ctx.owners.sessions.savedSessionRefreshTimer) {
      clearTimeout(ctx.owners.sessions.savedSessionRefreshTimer);
      ctx.owners.sessions.savedSessionRefreshTimer = null;
    }
    const params = new URLSearchParams({filter: ctx.owners.sessions.savedSessionFilter});
    const response = await fetch(`/api/sessions?${params.toString()}`, {cache:"no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    ctx.owners.sessions.savedSessions = Array.isArray(data.sessions) ? data.sessions : [];
    renderSavedSessions();
    return ctx.owners.sessions.savedSessions;
  }
  function scheduleSavedSessionsRefresh(delayMs = 1600) {
    if (ctx.owners.sessions.savedSessionRefreshTimer) clearTimeout(ctx.owners.sessions.savedSessionRefreshTimer);
    ctx.owners.sessions.savedSessionRefreshTimer = setTimeout(() => {
      ctx.owners.sessions.savedSessionRefreshTimer = null;
      if (ctx.owners.sessions.editingSessionTitleId) {
        scheduleSavedSessionsRefresh(delayMs);
        return;
      }
      fetchSavedSessions().catch(() => {});
    }, Math.max(0, Number(delayMs) || 0));
  }
  function savedSessionsTabVisible() {
    const panel = document.querySelector("[data-speaker-panel='sessions']");
    return Boolean(panel && !panel.hidden && document.visibilityState !== "hidden");
  }
  function syncSavedSessionsAutoRefresh() {
    if (savedSessionsTabVisible()) {
      if (!ctx.owners.sessions.savedSessionAutoRefreshTimer) {
        ctx.owners.sessions.savedSessionAutoRefreshTimer = setInterval(() => {
          if (!ctx.owners.sessions.editingSessionTitleId) fetchSavedSessions().catch(() => {});
        }, 5000);
      }
      return;
    }
    if (ctx.owners.sessions.savedSessionAutoRefreshTimer) {
      clearInterval(ctx.owners.sessions.savedSessionAutoRefreshTimer);
      ctx.owners.sessions.savedSessionAutoRefreshTimer = null;
    }
  }
  function refreshSavedSessionsAfterCompletion() {
    if (!ctx.owners.sessions.editingSessionTitleId) {
      fetchSavedSessions().catch(() => {});
    }
    scheduleSavedSessionsRefresh(1500);
  }
  function setSavedSessionFilter(filter) {
    const nextFilter = ["active", "archived", "all"].includes(filter) ? filter : "active";
    if (ctx.owners.sessions.savedSessionFilter === nextFilter) {
      fetchSavedSessions().catch(error => log(`Refresh sessions failed: ${error.message}`));
      return;
    }
    ctx.owners.sessions.savedSessionFilter = nextFilter;
    ctx.owners.sessions.selectedSavedSessionIds.clear();
    ctx.owners.sessions.openSessionMenuId = "";
    renderSavedSessions();
    fetchSavedSessions().catch(error => log(`Refresh sessions failed: ${error.message}`));
  }
  function currentMeetingIntelligenceSessionId() {
    return ctx.owners.sessions.openedSavedSessionId || ctx.owners.sessions.draftSavedSessionId || "";
  }
  function setMeetingIntelligenceReport(payload) {
    const container = payload && payload.meeting_intelligence ? payload.meeting_intelligence : payload;
    ctx.owners.reports.meetingIntelligenceReport = container && container.report ? container.report : null;
    ctx.owners.reports.meetingIntelligenceSelectedObjectId = "";
    clearMeetingEvidenceHighlight();
    renderMeetingIntelligencePanel();
  }
  function meetingObjects() {
    return ctx.owners.reports.meetingIntelligenceReport && Array.isArray(ctx.owners.reports.meetingIntelligenceReport.objects)
      ? ctx.owners.reports.meetingIntelligenceReport.objects
      : [];
  }
  function meetingObjectKindLabel(type) {
    return {
      summary: "Summary",
      decision: "Decision",
      action_item: "Action",
      open_question: "Question",
      risk: "Risk",
      claim: "Claim",
    }[type] || String(type || "Object");
  }
  function meetingStatusLabel(status) {
    return String(status || "draft").replace(/_/g, " ");
  }
  function meetingChip(text, status = "") {
    const chip = document.createElement("span");
    chip.className = `meeting-chip ${String(status || "").replace(/[^a-z0-9_-]+/gi, "")}`;
    chip.textContent = text;
    return chip;
  }
  function meetingEvidenceSpans(object) {
    return object && Array.isArray(object.evidence_spans) ? object.evidence_spans : [];
  }
  function meetingEvidenceRowIndex(rowRef, rowId = "") {
    if (rowRef && rowRef.index !== undefined && rowRef.index !== null) return String(rowRef.index);
    const value = String(rowId || "");
    const canonical = /^row_(\d+)$/.exec(value);
    if (canonical) return String(Number(canonical[1]));
    const legacyChat = /^ROW-(\d+)$/.exec(value);
    if (legacyChat) return String(Math.max(0, Number(legacyChat[1]) - 1));
    return "";
  }
  function findMeetingEvidenceTranscriptRow(rowRef, rowId = "") {
    const index = meetingEvidenceRowIndex(rowRef, rowId);
    if (!index) return null;
    const row = Array.from(sentences.querySelectorAll(".row")).find(row => (
      row.dataset.index === index && row.dataset.realtime !== "true"
    )) || null;
    if (row && row.dataset.groupHidden === "true" && row.dataset.groupLeader) {
      return findFinalSentenceRow(row.dataset.groupLeader) || row;
    }
    return row;
  }
  function clearMeetingEvidenceHighlight() {
    Array.from(sentences.querySelectorAll(".row.meeting-evidence-row")).forEach(row => {
      row.classList.remove("meeting-evidence-row");
    });
  }
  function meetingEvidenceRows(spans) {
    const rows = [];
    (Array.isArray(spans) ? spans : []).forEach(span => {
      const rowIds = Array.isArray(span.row_ids) ? span.row_ids : [];
      const refs = Array.isArray(span.rows) ? span.rows : [];
      if (refs.length) {
        refs.forEach((ref, index) => {
          const row = findMeetingEvidenceTranscriptRow(ref, rowIds[index] || ref.row_id || "");
          if (row && !rows.includes(row)) rows.push(row);
        });
      } else {
        rowIds.forEach(rowId => {
          const row = findMeetingEvidenceTranscriptRow(null, rowId);
          if (row && !rows.includes(row)) rows.push(row);
        });
      }
    });
    return rows;
  }
  function scrollToMeetingEvidence(spans) {
    clearMeetingEvidenceHighlight();
    const rows = meetingEvidenceRows(spans);
    rows.forEach(row => row.classList.add("meeting-evidence-row"));
    if (rows.length) {
      rows[0].scrollIntoView({block:"center", behavior:"smooth"});
    }
  }
  function selectedMeetingObject() {
    return meetingObjects().find(object => object.id === ctx.owners.reports.meetingIntelligenceSelectedObjectId) || null;
  }
  function selectMeetingIntelligenceObject(objectId) {
    ctx.owners.reports.meetingIntelligenceSelectedObjectId = objectId || "";
    renderMeetingIntelligencePanel();
    const object = selectedMeetingObject();
    if (object) scrollToMeetingEvidence(meetingEvidenceSpans(object));
  }
  function renderMeetingEvidence(object) {
    meetingIntelligenceEvidence.textContent = "";
    if (!object) {
      const empty = document.createElement("div");
      empty.className = "meeting-evidence-item";
      empty.textContent = "Select an object.";
      meetingIntelligenceEvidence.appendChild(empty);
      return;
    }
    const spans = meetingEvidenceSpans(object);
    if (!spans.length) {
      const empty = document.createElement("div");
      empty.className = "meeting-evidence-item";
      empty.textContent = "No evidence spans.";
      meetingIntelligenceEvidence.appendChild(empty);
      return;
    }
    spans.forEach(span => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "meeting-evidence-item";
      const meta = document.createElement("span");
      const speaker = span.speaker_name || speakerDisplayLabel(span.speaker_id);
      meta.textContent = `${speaker} ${secondsLabel(span.start)} - ${secondsLabel(span.end)} ${span.support_type || "direct"}`;
      const quote = document.createElement("span");
      quote.className = "meeting-evidence-quote";
      quote.textContent = span.quote_excerpt || "";
      item.append(meta, quote);
      item.addEventListener("click", () => scrollToMeetingEvidence([span]));
      meetingIntelligenceEvidence.appendChild(item);
    });
  }
  function renderMeetingObjectCard(object) {
    const card = document.createElement("div");
    card.className = "meeting-object-card";
    card.classList.toggle("selected", object.id === ctx.owners.reports.meetingIntelligenceSelectedObjectId);
    card.setAttribute("role", "button");
    card.tabIndex = 0;
    card.addEventListener("click", () => selectMeetingIntelligenceObject(object.id || ""));
    card.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectMeetingIntelligenceObject(object.id || "");
      }
    });

    const top = document.createElement("div");
    top.className = "meeting-object-top";
    top.appendChild(meetingChip(meetingObjectKindLabel(object.type)));
    top.appendChild(meetingChip(meetingStatusLabel(object.status), object.status || "draft"));
    const title = document.createElement("div");
    title.className = "meeting-object-title";
    title.textContent = object.title || meetingObjectKindLabel(object.type);
    top.appendChild(title);

    const body = document.createElement("div");
    body.className = "meeting-object-body";
    body.textContent = object.body || "";

    const meta = document.createElement("div");
    meta.className = "meeting-object-meta";
    const confidence = Number(object.confidence);
    const confidenceLabel = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "n/a";
    meta.textContent = `Confidence ${confidenceLabel}. ${object.confidence_reason || ""}`.trim();

    const actions = document.createElement("div");
    actions.className = "meeting-object-actions";
    ["accepted", "rejected"].forEach(status => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = status === "accepted" ? "accept" : "reject";
      button.textContent = status === "accepted" ? "Accept" : "Reject";
      button.disabled = ctx.owners.reports.meetingIntelligenceBusy || object.status === status || !currentMeetingIntelligenceSessionId();
      button.addEventListener("click", event => {
        event.stopPropagation();
        updateMeetingIntelligenceObjectStatus(object.id || "", status);
      });
      actions.appendChild(button);
    });

    card.append(top, body, meta, actions);
    return card;
  }
  function renderMeetingIntelligencePanel() {
    const sessionId = currentMeetingIntelligenceSessionId();
    meetingIntelligenceGenerate.disabled = ctx.owners.reports.meetingIntelligenceBusy || !sessionId;
    meetingIntelligenceGenerate.dataset.disabledHelp = ctx.owners.reports.meetingIntelligenceBusy
      ? "Wait for the current report generation to finish."
      : (!sessionId ? "Open a completed saved session before generating insights." : "");
    meetingIntelligenceObjects.textContent = "";
    meetingIntelligenceStats.textContent = "";
    if (!ctx.owners.reports.meetingIntelligenceReport) {
      meetingIntelligenceStatus.textContent = sessionId ? "No report" : "No saved session";
      meetingIntelligenceSummary.textContent = sessionId ? "No meeting intelligence report yet." : "Open or create a session.";
      renderMeetingEvidence(null);
      return;
    }
    const reportStatus = ctx.owners.reports.meetingIntelligenceReport.status || "draft";
    meetingIntelligenceStatus.textContent = `Local ${reportStatus}`;
    meetingIntelligenceSummary.textContent = ctx.owners.reports.meetingIntelligenceReport.summary || "";
    const objects = meetingObjects();
    const counts = objects.reduce((acc, object) => {
      const key = object.status || "draft";
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    meetingIntelligenceStats.appendChild(meetingChip(`${objects.length} objects`));
    Object.entries(counts).forEach(([status, count]) => {
      meetingIntelligenceStats.appendChild(meetingChip(`${count} ${meetingStatusLabel(status)}`, status));
    });
    if (ctx.owners.reports.meetingIntelligenceReport.current_transcript_revision_id) {
      meetingIntelligenceStats.appendChild(meetingChip("needs recheck", "stale"));
    }
    objects.forEach(object => {
      meetingIntelligenceObjects.appendChild(renderMeetingObjectCard(object));
    });
    if (!objects.length) {
      const empty = document.createElement("div");
      empty.className = "session-empty";
      empty.textContent = "No objects.";
      meetingIntelligenceObjects.appendChild(empty);
    }
    renderMeetingEvidence(selectedMeetingObject());
  }
  async function refreshMeetingIntelligenceReport() {
    const sessionId = currentMeetingIntelligenceSessionId();
    if (!sessionId) {
      renderMeetingIntelligencePanel();
      return;
    }
    try {
      const result = await post("/api/meeting-intelligence/report", {session_id: sessionId});
      if (result.meeting_intelligence) setMeetingIntelligenceReport(result.meeting_intelligence);
    } catch (error) {
      log(`Load meeting intelligence failed: ${error.message}`);
    }
  }
  async function generateMeetingIntelligenceReport() {
    const sessionId = currentMeetingIntelligenceSessionId();
    if (!sessionId) {
      log("Open or create a saved session first.");
      return;
    }
    ctx.owners.reports.meetingIntelligenceBusy = true;
    renderMeetingIntelligencePanel();
    try {
      const result = await post("/api/meeting-intelligence/generate", {session_id: sessionId});
      if (result.meeting_intelligence) setMeetingIntelligenceReport(result.meeting_intelligence);
      await fetchSavedSessions();
      log("Meeting intelligence report generated.");
    } catch (error) {
      log(`Generate meeting intelligence failed: ${error.message}`);
    } finally {
      ctx.owners.reports.meetingIntelligenceBusy = false;
      renderMeetingIntelligencePanel();
    }
  }
  async function updateMeetingIntelligenceObjectStatus(objectId, status) {
    const sessionId = currentMeetingIntelligenceSessionId();
    if (!sessionId || !objectId) return;
    ctx.owners.reports.meetingIntelligenceBusy = true;
    renderMeetingIntelligencePanel();
    try {
      const result = await post("/api/meeting-intelligence/update-object", {session_id: sessionId, object_id: objectId, status});
      if (result.meeting_intelligence) setMeetingIntelligenceReport(result.meeting_intelligence);
      await fetchSavedSessions();
    } catch (error) {
      log(`Update meeting intelligence failed: ${error.message}`);
    } finally {
      ctx.owners.reports.meetingIntelligenceBusy = false;
      renderMeetingIntelligencePanel();
    }
  }
  function reflectSavedSessionSource(sessionData) {
    const summary = sessionData.summary || {};
    const manifest = sessionData.manifest || {};
    const sourceInfo = manifest.source || summary.source || {};
    const mode = sourceInfo.capture_mode || (sourceInfo.streaming_audio ? "browser-stream" : "youtube");
    const label = {
      microphone: "Microphone",
      mixed: "Computer audio + microphone",
      "browser-stream": "Computer audio",
      "audio-file": "Audio file",
      youtube: "YouTube",
    }[mode] || "Source";
    const duration = Number(sourceInfo.duration_seconds || summary.source_duration_seconds || manifest.source_duration_seconds || summary.duration_seconds || manifest.duration_seconds || 0);
    sourceKind.textContent = label;
    sourceTitle.textContent = savedSessionTitle(sessionData);
    mediaTime.textContent = duration > 0 ? `Saved ${clockLabel(duration)}` : "Saved session";
    mediaCurrentTime.textContent = "00:00";
    mediaDuration.textContent = savedSessionDurationLabel(duration);
    timelineFill.style.width = "0%";
    timelineThumb.style.left = "0%";
  }
  function loadSavedSessionReview(sessionData, options = {}) {
    const summary = sessionData.summary || {};
    ctx.owners.sessions.openedSavedSessionId = summary.id || (sessionData.manifest && sessionData.manifest.id) || "";
    ctx.owners.sessions.draftSavedSessionId = "";
    stopPlaybackClock();
    stopBrowserAudioCapture();
    video.pause();
    audio.pause();
    if (ctx.owners.capture.es) {
      ctx.owners.capture.es.close();
      ctx.owners.capture.es = null;
    }
    resetTranscriptDisplay();
    ctx.owners.reference.editingSpeakerId = "";
    ctx.owners.reference.manualSpeakerComposerOpen = false;
    ctx.owners.reference.pendingManualSpeakerNameFocus = false;
    setTranscriptTitleSaved(savedSessionTitle(sessionData));
    reflectSavedSessionSource(sessionData);
    updateSpeakerState(sessionData.speaker_state || {group_name:"", groups:[], speakers:[]});
    ctx.owners.speakers.speakerSessionBaselineSentenceCounts = {};
    ctx.owners.speakers.speakerSessionBaselineSpeakingSeconds = {};
    const rows = savedSessionRows(sessionData);
    const previousFollowLiveEnabled = ctx.owners.speakers.followLiveEnabled;
    ctx.owners.speakers.followLiveEnabled = false;
    rows.forEach(row => renderSentence({...row, realtime:false, pending:false}));
    applyTranslationCollection(sessionData.translations, {refresh:true});
    ctx.owners.speakers.followLiveEnabled = previousFollowLiveEnabled;
    ctx.owners.speakers.speakerSessionBaselineSentenceCounts = {};
    ctx.owners.speakers.speakerSessionBaselineSpeakingSeconds = {};
    recomputeRenderedSpeakerSentenceCounts();
    refreshSpeakerPanelSentenceCounts();
    refreshTranscriptVisibility();
    setMeetingIntelligenceReport(sessionData.meeting_intelligence || null);
    renderSpeakerPanel();
    renderSavedSessions();
    setSourceControlsDisabled(false);
    start.disabled = false;
    stop.disabled = true;
    updateNewRunButtonState();
    setState("Reviewing");
    if (!options.quiet) {
      log(`Opened saved session ${savedSessionTitle(sessionData)}.`);
    }
  }
  async function openSavedSession(sessionId) {
    if (!sessionId) return;
    try {
      const result = await post("/api/sessions/open", {session_id: sessionId});
      if (result.session) {
        loadSavedSessionReview(result.session);
      }
    } catch (error) {
      log(`Open session failed: ${error.message}`);
    }
  }
  async function archiveSavedSession(sessionId) {
    try {
      await post("/api/sessions/archive", {session_id: sessionId});
      if (ctx.owners.sessions.openedSavedSessionId === sessionId) renderSavedSessions();
      await fetchSavedSessions();
    } catch (error) {
      log(`Archive failed: ${error.message}`);
    }
  }
  async function restoreSavedSession(sessionId) {
    try {
      await post("/api/sessions/restore", {session_id: sessionId});
      await fetchSavedSessions();
    } catch (error) {
      log(`Restore failed: ${error.message}`);
    }
  }
  function reflectDeletedSavedSession(sessionId) {
    ctx.owners.sessions.selectedSavedSessionIds.delete(sessionId);
    if (ctx.owners.sessions.draftSavedSessionId === sessionId) ctx.owners.sessions.draftSavedSessionId = "";
    if (ctx.owners.sessions.openedSavedSessionId === sessionId) {
      ctx.owners.sessions.openedSavedSessionId = "";
      setTranscriptTitleLive();
      resetTranscriptDisplay();
      setState("Ready");
    }
  }
  async function bulkSavedSessionAction(action) {
    if (ctx.owners.sessions.savedSessionBulkActionBusy) return;
    const selected = selectedSavedSessions();
    const targets = selected.filter(item => {
      if (action === "archive") return !item.archived;
      if (action === "restore") return Boolean(item.archived);
      return action === "delete";
    });
    if (!targets.length) return;
    if (action === "delete" && !confirm(`Delete ${targets.length} selected session${targets.length === 1 ? "" : "s"} permanently?`)) {
      return;
    }
    const endpoint = {
      archive: "/api/sessions/archive",
      restore: "/api/sessions/restore",
      delete: "/api/sessions/delete",
    }[action];
    ctx.owners.sessions.savedSessionBulkActionBusy = true;
    renderSavedSessions();
    let completed = 0;
    const failures = [];
    for (const item of targets) {
      try {
        await post(endpoint, {session_id: item.id});
        completed += 1;
        ctx.owners.sessions.selectedSavedSessionIds.delete(item.id);
        if (action === "delete") reflectDeletedSavedSession(item.id);
      } catch (error) {
        failures.push(`${savedSessionDisplayTitle(item)}: ${error.message}`);
      }
    }
    try {
      await fetchSavedSessions();
    } catch (error) {
      failures.push(`Refresh: ${error.message}`);
    } finally {
      ctx.owners.sessions.savedSessionBulkActionBusy = false;
      renderSavedSessions();
    }
    const actionLabel = {archive:"Archived", restore:"Restored", delete:"Deleted"}[action];
    if (completed) log(`${actionLabel} ${completed} session${completed === 1 ? "" : "s"}.`);
    if (failures.length) log(`Some sessions could not be updated: ${failures.join("; ")}`);
  }
  async function commitSessionTitleInput(item, input) {
    if (!item || !input || input.dataset.saving === "1") return;
    const sessionId = item.id || "";
    const currentTitle = savedSessionDisplayTitle(item);
    const cleanTitle = input.value.trim();
    if (!cleanTitle) {
      input.value = currentTitle;
      log("Session title must not be empty.");
      return;
    }
    ctx.owners.sessions.editingSessionTitleId = "";
    ctx.owners.sessions.pendingSessionTitleFocusId = "";
    if (cleanTitle === currentTitle) {
      input.value = currentTitle;
      renderSavedSessions();
      return;
    }
    input.dataset.saving = "1";
    input.disabled = true;
    try {
      const result = await post("/api/sessions/rename", {session_id: sessionId, title: cleanTitle});
      if (ctx.owners.sessions.openedSavedSessionId === sessionId && result.session) {
        setTranscriptTitleSaved(result.session.title || cleanTitle);
        sourceTitle.textContent = result.session.title || cleanTitle;
      }
      await fetchSavedSessions();
    } catch (error) {
      input.disabled = false;
      input.dataset.saving = "";
      input.value = currentTitle;
      log(`Rename session failed: ${error.message}`);
      renderSavedSessions();
    }
  }
  async function deleteSavedSession(item) {
    ctx.owners.sessions.openSessionMenuId = "";
    if (!confirm(`Delete "${item.title || "saved session"}" permanently?`)) {
      renderSavedSessions();
      return;
    }
    try {
      await post("/api/sessions/delete", {session_id: item.id});
      reflectDeletedSavedSession(item.id);
      await fetchSavedSessions();
    } catch (error) {
      log(`Delete session failed: ${error.message}`);
      renderSavedSessions();
    }
  }
  function browserLiveObservationEnabled() {
    return Boolean(liveSpeakerConfig.browser_observation_enabled);
  }
  function browserLiveObservationIntervalMs() {
    return Math.max(20, Number(liveSpeakerConfig.browser_observation_interval_seconds || 0.1) * 1000);
  }
  function mediaSeconds(element) {
    const seconds = Number(element.currentTime || 0);
    return Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  }
  function playbackClockMaxSeconds() {
    if (ctx.owners.capture.playbackClockStartedAt === null) return Number.POSITIVE_INFINITY;
    return Math.max(0, ((performance.now() - ctx.owners.capture.playbackClockStartedAt) / 1000) + playbackClockSlackSeconds);
  }
  function playbackSeconds() {
    const audioSeconds = mediaSeconds(audio);
    const videoSeconds = mediaSeconds(video);
    let seconds = 0;
    if (!audio.paused && !audio.ended) {
      seconds = audioSeconds;
    } else if (!video.paused && !video.ended) {
      seconds = videoSeconds;
    } else {
      seconds = Math.min(audioSeconds || videoSeconds, videoSeconds || audioSeconds);
    }
    return Math.min(seconds, playbackClockMaxSeconds());
  }
  function browserLiveObservationSample() {
    if (!browserLiveObservationEnabled() || !ctx.owners.transcript.browserLiveObservationStarted) return;
    const liveRows = Array.from(speakerList.querySelectorAll(".speaker-item.live-speaker"));
    const domLiveSpeakerIds = liveRows
      .map(row => row.dataset.speakerId || "")
      .filter(Boolean);
    ctx.owners.transcript.browserLiveObservationBuffer.push({
      wall_time: Date.now() / 1000,
      performance_ms: performance.now(),
      playback_time: playbackSeconds(),
      dom_live_speaker_ids: domLiveSpeakerIds,
      visible_live_speaker_id: domLiveSpeakerIds.length === 1 ? domLiveSpeakerIds[0] : "",
      current_live_speaker_id: ctx.owners.transcript.currentLiveSpeakerId || "",
      transcript_live_speaker_id: ctx.owners.transcript.transcriptLiveSpeakerId || "",
      transcript_live_override_speaker_id: ctx.owners.transcript.transcriptLiveSpeakerOverrideId || "",
      fallback_live_speaker_id: ctx.owners.transcript.fallbackLiveSpeakerId || "",
      runtime_state: state.textContent || "",
    });
    if (ctx.owners.transcript.browserLiveObservationBuffer.length >= 10) {
      void flushBrowserLiveObservation(false, "batch");
    }
  }
  async function flushBrowserLiveObservation(finalFlush=false, reason="batch") {
    if (!browserLiveObservationEnabled()) return null;
    if (ctx.owners.transcript.browserLiveObservationPosting && !finalFlush) return null;
    const samples = ctx.owners.transcript.browserLiveObservationBuffer.splice(0);
    if (!samples.length && !finalFlush) return null;
    ctx.owners.transcript.browserLiveObservationPosting = true;
    try {
      const endpoint = finalFlush ? "/api/live-observation-finish" : "/api/live-observation";
      return await post(endpoint, {samples, reason});
    } catch (error) {
      if (samples.length) ctx.owners.transcript.browserLiveObservationBuffer = samples.concat(ctx.owners.transcript.browserLiveObservationBuffer);
      log(`Browser live observation failed: ${error.message}`);
      return null;
    } finally {
      ctx.owners.transcript.browserLiveObservationPosting = false;
    }
  }
  function stopBrowserLiveObservationTimerOnly() {
    if (ctx.owners.transcript.browserLiveObservationTimer) {
      clearInterval(ctx.owners.transcript.browserLiveObservationTimer);
      ctx.owners.transcript.browserLiveObservationTimer = null;
    }
  }
  function startBrowserLiveObservation() {
    if (!browserLiveObservationEnabled()) return;
    stopBrowserLiveObservationTimerOnly();
    ctx.owners.transcript.browserLiveObservationBuffer = [];
    ctx.owners.transcript.browserLiveObservationStarted = true;
    browserLiveObservationSample();
    ctx.owners.transcript.browserLiveObservationTimer = setInterval(browserLiveObservationSample, browserLiveObservationIntervalMs());
  }
  async function stopBrowserLiveObservation(reason="done") {
    if (!browserLiveObservationEnabled() || !ctx.owners.transcript.browserLiveObservationStarted) return null;
    stopBrowserLiveObservationTimerOnly();
    browserLiveObservationSample();
    ctx.owners.transcript.browserLiveObservationStarted = false;
    const result = await flushBrowserLiveObservation(true, reason);
    if (result && result.summary) {
      log(`Browser live score ${Number(result.summary.strict_browser_live_score || 0).toFixed(3)}`);
    }
    return result;
  }
  function startPlaybackClock() {
    if (ctx.owners.capture.playbackTimer) clearInterval(ctx.owners.capture.playbackTimer);
    ctx.owners.capture.playbackClockStartedAt = performance.now();
    const send = () => post("/api/playback", {seconds: playbackSeconds()}).catch(() => {});
    send();
    ctx.owners.capture.playbackTimer = setInterval(send, 250);
  }
  function flushPlaybackEnd() {
    const duration = Number(audio.duration || 0);
    if (!ctx.owners.capture.playbackTimer) return;
    if (duration > 0 && ctx.owners.capture.playbackClockStartedAt !== null) {
      const elapsed = (performance.now() - ctx.owners.capture.playbackClockStartedAt) / 1000;
      if (elapsed + playbackClockSlackSeconds < duration) {
        log(`Ignoring early audio ended event at ${playbackSeconds().toFixed(1)}s of ${duration.toFixed(1)}s.`);
        return;
      }
    }
    post("/api/playback", {seconds: duration > 0 ? duration : playbackSeconds()}).catch(() => {});
  }
  function stopPlaybackClock() {
    if (ctx.owners.capture.playbackTimer) {
      clearInterval(ctx.owners.capture.playbackTimer);
      ctx.owners.capture.playbackTimer = null;
    }
    ctx.owners.capture.playbackClockStartedAt = null;
  }

  Object.assign(ctx.api, {archiveSavedSession, browserLiveObservationEnabled, browserLiveObservationIntervalMs, browserLiveObservationSample, bulkSavedSessionAction, clearMeetingEvidenceHighlight, clearSavedSessionSelection, commitSessionTitleInput, createSessionMenuIcon, currentMeetingIntelligenceSessionId, deleteSavedSession, fetchSavedSessions, findMeetingEvidenceTranscriptRow, flushBrowserLiveObservation, flushPlaybackEnd, generateMeetingIntelligenceReport, loadSavedSessionReview, mediaSeconds, meetingChip, meetingEvidenceRowIndex, meetingEvidenceRows, meetingEvidenceSpans, meetingObjectKindLabel, meetingObjects, meetingStatusLabel, openSavedSession, playbackClockMaxSeconds, playbackSeconds, reflectDeletedSavedSession, reflectSavedSessionSource, refreshMeetingIntelligenceReport, refreshSavedSessionsAfterCompletion, renderMeetingEvidence, renderMeetingIntelligencePanel, renderMeetingObjectCard, renderSavedSessions, restoreSavedSession, sameLocalDate, savedSessionDate, savedSessionDatePart, savedSessionDisplayTitle, savedSessionDurationLabel, savedSessionRows, savedSessionTimePart, savedSessionTimeRangeLabel, savedSessionTitle, savedSessionUpdatedLabel, savedSessionsTabVisible, scheduleSavedSessionsRefresh, scrollToMeetingEvidence, selectAllSavedSessions, selectMeetingIntelligenceObject, selectedMeetingObject, selectedSavedSessions, setEditingSessionTitle, setMeetingIntelligenceReport, setSavedSessionFilter, startBrowserLiveObservation, startPlaybackClock, stopBrowserLiveObservation, stopBrowserLiveObservationTimerOnly, stopPlaybackClock, syncSavedSessionSelectionControls, syncSavedSessionsAutoRefresh, updateMeetingIntelligenceObjectStatus});
}
