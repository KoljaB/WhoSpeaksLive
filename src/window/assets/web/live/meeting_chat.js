export function installMeetingChat(ctx) {
  const {
    appResources,
    askSelectedMeetingsButton,
    meetingChatClear,
    meetingChatForm,
    meetingChatMessages,
    meetingChatProgress,
    meetingChatProgressBar,
    meetingChatProgressElapsed,
    meetingChatProgressPercent,
    meetingChatProgressText,
    meetingChatQuestion,
    meetingChatScope,
    meetingChatSend,
    meetingChatStatus,
    meetingChatTitle,
  } = ctx;

  function selectedSessionIds() {
    const selected = ctx.api.selectedSavedSessions ? ctx.api.selectedSavedSessions() : [];
    if (selected.length) return selected.map(item => item.id).filter(Boolean).sort();
    const current = ctx.api.currentMeetingIntelligenceSessionId
      ? ctx.api.currentMeetingIntelligenceSessionId()
      : "";
    return current ? [current] : [];
  }

  function askPanelVisible() {
    const panel = document.querySelector("[data-speaker-panel='ask']");
    return Boolean(panel && !panel.hidden);
  }

  function setBusy(busy, message = "") {
    ctx.owners.chat.busy = Boolean(busy);
    meetingChatSend.disabled = Boolean(busy);
    meetingChatClear.disabled = Boolean(busy) || !ctx.owners.chat.scope;
    meetingChatQuestion.disabled = Boolean(busy);
    meetingChatProgress.hidden = !busy;
    if (message) meetingChatProgressText.textContent = message;
    if (busy) {
      ctx.owners.chat.jobStartedAt = performance.now();
      meetingChatProgressBar.value = 0;
      meetingChatProgressPercent.textContent = "0%";
      meetingChatProgressElapsed.textContent = "0s";
      if (ctx.owners.chat.jobElapsedTimer) clearInterval(ctx.owners.chat.jobElapsedTimer);
      ctx.owners.chat.jobElapsedTimer = setInterval(() => {
        const elapsed = Math.max(0, Math.floor((performance.now() - ctx.owners.chat.jobStartedAt) / 1000));
        meetingChatProgressElapsed.textContent = `${elapsed}s`;
      }, 1000);
    } else if (ctx.owners.chat.jobElapsedTimer) {
      clearInterval(ctx.owners.chat.jobElapsedTimer);
      ctx.owners.chat.jobElapsedTimer = null;
    }
  }

  function clockLabel(seconds) {
    const value = Math.max(0, Number(seconds || 0));
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);
    return hours > 0
      ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
      : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function renderScope(scope) {
    meetingChatScope.textContent = "";
    const meetings = (scope && scope.meetings) || [];
    meetings.forEach(meeting => {
      const chip = document.createElement("span");
      chip.className = "meeting-chat-scope-chip";
      const dateValue = meeting.started_at || meeting.updated_at || "";
      const parsedDate = dateValue ? new Date(dateValue) : null;
      const dateLabel = parsedDate && !Number.isNaN(parsedDate.getTime())
        ? new Intl.DateTimeFormat(undefined, {
          day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
        }).format(parsedDate)
        : "";
      const duration = Number(meeting.duration_seconds || 0);
      const durationLabel = duration > 0 ? clockLabel(duration) : "";
      const title = meeting.title || meeting.id || "Session";
      chip.textContent = [title, dateLabel, durationLabel].filter(Boolean).join(" · ");
      chip.title = `Session ID: ${meeting.id || "unknown"}`;
      meetingChatScope.appendChild(chip);
    });
    const count = meetings.length;
    meetingChatTitle.textContent = count > 1 ? `Ask ${count} sessions` : "Ask this session";
    if (!scope) {
      meetingChatStatus.textContent = "Choose a session to begin.";
    } else if (scope.provisional) {
      meetingChatStatus.textContent = "Live session · answers use finalized transcript rows.";
    } else if (scope.requires_index && !(scope.index && scope.index.configured)) {
      meetingChatStatus.textContent = "Configure text embeddings to search this scope.";
    } else if (scope.requires_index) {
      const indexed = ((scope.index && scope.index.sessions) || []).filter(
        item => item.indexed && item.current_embedding_model && item.current_revision,
      ).length;
      meetingChatStatus.textContent = indexed === count ? "Search index ready" : `Indexing ${indexed} of ${count} sessions`;
    } else {
      meetingChatStatus.textContent = "Full transcript ready";
    }
  }

  function renderMessages(history) {
    meetingChatMessages.textContent = "";
    const entries = Array.isArray(history) ? history : [];
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "meeting-chat-empty";
      empty.textContent = "Ask about decisions, commitments, risks, or what a specific person said.";
      meetingChatMessages.appendChild(empty);
      return;
    }
    entries.forEach(entry => {
      const question = document.createElement("div");
      question.className = "meeting-chat-message meeting-chat-question";
      question.textContent = entry.question || "";
      meetingChatMessages.appendChild(question);

      const answer = document.createElement("article");
      answer.className = `meeting-chat-message meeting-chat-answer ${entry.status || ""}`;
      const answerStatus = document.createElement("div");
      answerStatus.className = "meeting-chat-answer-status";
      answerStatus.textContent = entry.status === "answered"
        ? "Grounded answer"
        : entry.status === "needs_review"
          ? "Needs review"
          : "Not established from transcript";
      answer.appendChild(answerStatus);
      const body = document.createElement("div");
      body.className = "meeting-chat-answer-body";
      const rawText = entry.text || "";
      body.textContent = /^status\s+not_established$/i.test(rawText.trim())
        ? "The selected transcript did not establish an answer."
        : rawText;
      answer.appendChild(body);
      const citations = document.createElement("div");
      citations.className = "meeting-chat-citations";
      (entry.evidence || []).forEach(evidence => citations.appendChild(evidenceButton(evidence)));
      answer.appendChild(citations);
      if (entry.provisional) {
        const cutoff = document.createElement("div");
        cutoff.className = "meeting-chat-cutoff";
        cutoff.textContent = `Live answer through ${clockLabel(entry.transcript_end_seconds)}`;
        answer.appendChild(cutoff);
      }
      meetingChatMessages.appendChild(answer);
    });
    meetingChatMessages.scrollTop = meetingChatMessages.scrollHeight;
  }

  function evidenceButton(evidence) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "meeting-chat-citation";
    button.textContent = `${evidence.meeting_title || "Session"} · ${evidence.speaker_name || "Unknown"} · ${clockLabel(evidence.start)}`;
    button.title = evidence.quote || "Open transcript evidence";
    button.addEventListener("click", async () => {
      const opened = ctx.owners.sessions.openedSavedSessionId || ctx.owners.sessions.draftSavedSessionId || "";
      if (evidence.meeting_id && evidence.meeting_id !== opened && ctx.api.openSavedSession) {
        await ctx.api.openSavedSession(evidence.meeting_id);
      }
      if (ctx.api.scrollToMeetingEvidence) {
        ctx.api.scrollToMeetingEvidence([{
          row_ids: [evidence.row_id],
          rows: [{row_id: evidence.row_id, index: evidence.row_index}],
        }]);
      }
    });
    return button;
  }

  async function refreshMeetingChatScope() {
    if (!askPanelVisible() || ctx.owners.chat.busy) return;
    const ids = selectedSessionIds();
    if (ids.length > 20) {
      ctx.owners.chat.scope = null;
      renderScope(null);
      meetingChatStatus.textContent = "Select at most 20 sessions.";
      renderMessages([]);
      return;
    }
    try {
      const result = await ctx.api.post("/api/meeting-intelligence/chat/scope", {session_ids: ids});
      ctx.owners.chat.scope = result;
      renderScope(result);
      renderMessages(result.history || []);
      meetingChatClear.disabled = !(result.history || []).length;
    } catch (error) {
      ctx.owners.chat.scope = null;
      renderScope(null);
      meetingChatStatus.textContent = error.message;
      renderMessages([]);
    }
  }

  function scheduleMeetingChatScopeRefresh() {
    const explicitlySelected = ctx.api.selectedSavedSessions ? ctx.api.selectedSavedSessions() : [];
    askSelectedMeetingsButton.disabled = explicitlySelected.length < 1;
    askSelectedMeetingsButton.dataset.disabledHelp = explicitlySelected.length < 1
      ? "Select one or more saved sessions first."
      : "";
    if (!askPanelVisible() || ctx.owners.chat.busy) return;
    if (ctx.owners.chat.scopeRefreshTimer) clearTimeout(ctx.owners.chat.scopeRefreshTimer);
    ctx.owners.chat.scopeRefreshTimer = setTimeout(() => {
      ctx.owners.chat.scopeRefreshTimer = null;
      refreshMeetingChatScope();
    }, 250);
  }

  async function askQuestion() {
    const question = meetingChatQuestion.value.trim();
    if (!question || ctx.owners.chat.busy) return;
    const ids = selectedSessionIds();
    if (ids.length > 20) {
      meetingChatStatus.textContent = "Select at most 20 sessions.";
      return;
    }
    setBusy(true, "Preparing session search");
    try {
      const result = await ctx.api.post("/api/meeting-intelligence/chat/ask-async", {session_ids: ids, question});
      const job = result.job || {};
      ctx.owners.chat.jobId = job.job_id || "";
      if (!ctx.owners.chat.jobId) throw new Error("Meeting Intelligence did not return a chat job.");
      await pollJob(ctx.owners.chat.jobId);
      meetingChatQuestion.value = "";
    } catch (error) {
      meetingChatStatus.textContent = error.message;
    } finally {
      ctx.owners.chat.jobId = "";
      setBusy(false);
    }
  }

  async function pollJob(jobId) {
    while (ctx.owners.chat.jobId === jobId) {
      const response = await fetch(`/api/meeting-intelligence/chat/job?job_id=${encodeURIComponent(jobId)}`, {cache: "no-store"});
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || `Chat status failed (${response.status})`);
      const job = payload.job || {};
      meetingChatProgressText.textContent = job.message || job.stage || "Working";
      const percent = Math.max(0, Math.min(100, Number(job.percent || 0)));
      meetingChatProgressBar.value = percent;
      meetingChatProgressPercent.textContent = `${Math.round(percent)}%`;
      if (job.status === "failed") throw new Error(job.error || "Session chat failed.");
      if (job.status === "succeeded") {
        const result = job.result || {};
        if (ctx.owners.chat.scope) ctx.owners.chat.scope.history = result.history || [];
        renderMessages(result.history || []);
        const completedAnswer = result.answer || {};
        if (completedAnswer.provisional) {
          meetingChatStatus.textContent = `Live answer through ${clockLabel(completedAnswer.transcript_end_seconds)}`;
        } else if (completedAnswer.grounding_status === "answered" || completedAnswer.status === "answered") {
          meetingChatStatus.textContent = "Answer grounded in transcript evidence";
        } else if (completedAnswer.grounding_status === "needs_review" || completedAnswer.status === "needs_review") {
          meetingChatStatus.textContent = "Answer needs review";
        } else {
          meetingChatStatus.textContent = "Not established from the selected transcript";
        }
        return;
      }
      await new Promise(resolve => { ctx.owners.chat.jobPollTimer = setTimeout(resolve, 750); });
    }
  }

  async function clearChat() {
    if (!ctx.owners.chat.scope || ctx.owners.chat.busy) return;
    try {
      const result = await ctx.api.post("/api/meeting-intelligence/chat/clear", {
        session_ids: ctx.owners.chat.scope.session_ids || selectedSessionIds(),
      });
      ctx.owners.chat.scope.history = result.history || [];
      renderMessages([]);
      meetingChatClear.disabled = true;
    } catch (error) {
      meetingChatStatus.textContent = error.message;
    }
  }

  Object.assign(ctx.api, {refreshMeetingChatScope, scheduleMeetingChatScopeRefresh});
  ctx.activators.push(() => {
    askSelectedMeetingsButton.addEventListener("click", () => ctx.api.setSpeakerTab("ask"));
    meetingChatForm.addEventListener("submit", event => {
      event.preventDefault();
      askQuestion();
    });
    meetingChatQuestion.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        askQuestion();
      }
    });
    meetingChatClear.addEventListener("click", clearChat);
    scheduleMeetingChatScopeRefresh();
    appResources.own(() => {
      if (ctx.owners.chat.scopeRefreshTimer) clearTimeout(ctx.owners.chat.scopeRefreshTimer);
      if (ctx.owners.chat.jobPollTimer) clearTimeout(ctx.owners.chat.jobPollTimer);
      if (ctx.owners.chat.jobElapsedTimer) clearInterval(ctx.owners.chat.jobElapsedTimer);
      ctx.owners.chat.jobId = "";
    });
  });
}
