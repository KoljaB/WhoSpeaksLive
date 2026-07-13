import {
  el,
  state,
  api,
  escapeHtml,
  domId,
  transcriptRowDomId,
  uniqueValues,
  normalizedFallbackRowId,
  transcriptRowAliases,
  transcriptRowElementId,
  encodedRowAliases,
  evidenceRowIds,
  evidenceRowLabel,
  clearEvidenceFocus,
  clearDeleteConfirm,
  setStatus,
  formatDateTime,
  reportStatusHtml,
  providerOptions,
  providerOption,
  syncProviderDraftFromConfig,
  selectedProviderOption,
  providerModelOptions,
  renderProviderControls,
  loadProviderModels,
  providerLabel,
  sessionIcon,
  tabIcon,
  sectionIcon,
  sectionLabel,
  pluralLabel,
  deepCopy,
  currentTemplate,
  effectiveReportTemplate,
  reportSectionDefinitions,
  sectionTabId,
  sectionIconName,
  reportTabs,
  ensureActiveTab,
  normalizeBuilderKey,
  defaultOutputField,
  defaultSection,
  newTemplateDraft,
  cleanBuilderTemplate,
  renderTemplateControls,
  builderValidationMessage,
  openTemplateBuilder,
  closeTemplateBuilder,
  renderTemplateBuilder,
  attachBuilderHandlers,
  saveTemplateBuilder,
  deleteSelectedTemplate,
  loadTemplates,
} from "./report_builder.js";

function renderSessions() {
  const query = el("sessionSearch").value.trim().toLowerCase();
  const filtered = state.sessions.filter((session) => {
    const text = `${session.title || ""} ${(session.speaker_names || []).join(" ")}`.toLowerCase();
    return !query || text.includes(query);
  });
  const heading = `<div class="sessions-heading"><span>Sessions</span><span>${filtered.length}</span></div>`;
  el("sessions").innerHTML = heading + (filtered.length ? filtered.map((session) => {
    const active = session.id === state.sessionId ? " active" : "";
    const rows = Number(session.transcript_rows || 0);
    const speakers = Number(session.speaker_count || 0);
    const source = session.source_kind === "demo_transcript" ? "Demo" : "Saved";
    const cached = Array.isArray(session.report_template_ids)
      ? session.report_template_ids.includes(state.templateId)
      : session.has_cached_report || session.has_meeting_intelligence;
    const sessionTime = formatDateTime(session.updated_at || session.created_at);
    return `
      <button class="session${active}" type="button" data-session-id="${escapeHtml(session.id)}">
        <span class="session-icon">${sessionIcon(session.source_kind)}</span>
        <span class="session-main">
          <span class="session-title-row">
            <span class="session-title">${escapeHtml(session.title || session.id)}</span>
            ${sessionTime ? `<span class="session-time">${escapeHtml(sessionTime)}</span>` : ""}
          </span>
          <span class="meta">${rows} rows / ${speakers} speakers</span>
          <span class="badge-row">
            <span class="badge">${escapeHtml(source)}</span>
            ${cached ? '<span class="badge hot">Report</span>' : '<span class="badge warn">No report</span>'}
          </span>
        </span>
      </button>
    `;
  }).join("") : '<div class="empty">No sessions found.</div>');
  document.querySelectorAll(".session").forEach((button) => {
    button.addEventListener("click", () => selectSession(button.dataset.sessionId));
  });
}

function renderTabs() {
  ensureActiveTab();
  el("tabs").innerHTML = reportTabs().map(([id, label, icon]) => `
    <button class="tab ${id === state.activeTab ? "active" : ""}" type="button" data-tab="${id}">
      ${tabIcon(icon)}
      <span dir="auto">${escapeHtml(label)}</span>
    </button>
  `).join("");
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      if (tab !== "transcript") {
        clearEvidenceFocus();
      }
      state.activeTab = tab;
      render();
    });
  });
}

function sectionItemCount(names) {
  return names.reduce((total, name) => {
    const items = getSection(name).items;
    return total + (Array.isArray(items) ? items.length : 0);
  }, 0);
}

function statCard(icon, value, label) {
  return `
    <div class="stat-card">
      ${tabIcon(icon)}
      <div>
        <strong>${escapeHtml(value)}</strong>
        <span>${escapeHtml(label)}</span>
      </div>
    </div>
  `;
}

function renderReportStats() {
  const stats = el("reportStats");
  if (!state.report) {
    stats.hidden = true;
    stats.innerHTML = "";
    return;
  }
  const evidenceCount = Array.isArray(state.report.evidence_index) ? state.report.evidence_index.length : 0;
  const rows = Number(state.transcriptRows?.length || 0);
  const definitions = reportSectionDefinitions();
  const totalItems = definitions.reduce((total, definition) => total + sectionItemCount([definition.key]), 0);
  const groundedItems = definitions.reduce((total, definition) => total + (getSection(definition.key).items || []).filter((item) => item.grounding_status !== "missing_required_evidence").length, 0);
  stats.hidden = false;
  stats.innerHTML = [
    statCard("file", definitions.length, "Report sections"),
    statCard("tasks", totalItems, "Generated items"),
    statCard("check", groundedItems, "Evidence-grounded"),
    statCard("link", evidenceCount, "Evidence links"),
    statCard("wave", rows, "Transcript rows")
  ].join("");
}

function renderHeader() {
  const session = state.sessions.find((item) => item.id === state.sessionId) || {};
  const report = state.report || {};
  el("reportTitle").textContent = report.title || session.title || "Select a session";
  const badges = [];
  if (report.pipeline) badges.push(`<span class="badge hot">${escapeHtml(report.pipeline.mode || "pipeline")}</span>`);
  const template = effectiveReportTemplate();
  if (template.name) badges.push(`<span class="badge">${escapeHtml(template.name)}</span>`);
  if (template.privacy_policy === "local_only") badges.push('<span class="badge warn">Local only</span>');
  if (report.provider) badges.push(`<span class="badge">${escapeHtml(report.provider)}</span>`);
  if (report.report_language) badges.push(`<span class="badge">${escapeHtml(report.report_language)}</span>`);
  if (report.pipeline?.segments) badges.push(`<span class="badge">${report.pipeline.segments} segments</span>`);
  el("reportBadges").innerHTML = badges.join("");
  renderReportStats();
}

function generationIsTerminal(job = state.generationJob) {
  return Boolean(job && ["succeeded", "failed"].includes(job.status));
}

function openProgressOverlay() {
  el("progressOverlay").hidden = false;
  document.body.classList.add("progress-modal-open");
}

function closeProgressOverlay() {
  if (state.generating || !generationIsTerminal()) return;
  el("progressOverlay").hidden = true;
  document.body.classList.remove("progress-modal-open");
}

function renderProgress() {
  const overlay = el("progressOverlay");
  const dialog = el("progressDialog");
  const job = state.generationJob;
  if (!job) {
    overlay.hidden = true;
    document.body.classList.remove("progress-modal-open");
    return;
  }
  const terminal = generationIsTerminal(job);
  const canClose = terminal && !state.generating;
  if (state.generating || !terminal) openProgressOverlay();
  const percent = Math.max(0, Math.min(100, Number(job.percent || 0)));
  el("progressLabel").textContent = job.message || "Generating report";
  el("progressPercent").textContent = `${percent}%`;
  el("progressFill").style.width = `${percent}%`;
  const step = job.total ? `Step ${job.current || 0} of ${job.total}` : "";
  const detail = [job.stage, step, job.detail].filter(Boolean).join(" / ");
  el("progressDetail").textContent = detail;
  const events = Array.isArray(job.events) ? job.events.slice(-6) : [];
  el("progressLog").innerHTML = events.map((event) => `
    <div class="progress-event">
      <strong>${escapeHtml(event.stage || "")}</strong>
      <span>${escapeHtml(event.message || "")}${event.detail ? `: ${escapeHtml(event.detail)}` : ""}</span>
    </div>
  `).join("");
  dialog.classList.toggle("failed", job.status === "failed");
  dialog.classList.toggle("succeeded", job.status === "succeeded");
  el("progressDialogTitle").textContent = job.status === "succeeded"
    ? "Report generated"
    : job.status === "failed"
      ? "Report generation failed"
      : "Generating report";
  el("closeProgressBtn").disabled = !canClose;
  const footerButton = el("closeProgressFooterBtn");
  footerButton.disabled = !canClose;
  footerButton.textContent = job.status === "succeeded"
    ? "View report"
    : job.status === "failed"
      ? "Close"
      : "Generating...";
}

function getSection(name) {
  return state.report?.sections?.[name] || {items: [], summary: ""};
}

function evidenceLookup() {
  const result = new Map();
  (state.report?.evidence_index || []).forEach((item) => result.set(item.id, item));
  return result;
}

function scrollToTranscriptRows(rowIds) {
  const expected = new Set(rowIds.map((rowId) => encodeURIComponent(String(rowId))));
  const target = Array.from(document.querySelectorAll(".transcript-row")).find((node) => {
    const aliases = String(node.dataset.rowAliases || "").split(" ").filter(Boolean);
    return aliases.some((alias) => expected.has(alias));
  });
  if (!target) return;
  target.scrollIntoView({block: "center", behavior: "smooth"});
  target.focus({preventScroll: true});
}

function openEvidenceInTranscript(evidenceId) {
  const evidence = evidenceLookup();
  const item = evidence.get(evidenceId);
  const rowIds = evidenceRowIds(item);
  if (!item || !rowIds.length) return;
  const hadEvidenceFocus = Boolean(state.activeEvidenceId);
  state.evidenceReturnTab = state.activeTab === "transcript"
    ? state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript"
    : state.activeTab;
  state.activeEvidenceId = evidenceId;
  state.highlightRowIds = rowIds;
  state.activeTab = "transcript";
  try {
    const hash = `#evidence-${encodeURIComponent(evidenceId)}`;
    if (hadEvidenceFocus) {
      history.replaceState({meetingEvidence: true, evidenceId}, "", hash);
    } else {
      history.pushState({meetingEvidence: true, evidenceId}, "", hash);
    }
  } catch (error) {
    // Navigation history is a convenience; transcript focus still works without it.
  }
  render();
  window.requestAnimationFrame(() => scrollToTranscriptRows(rowIds));
}

function returnFromEvidence() {
  const tab = state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript";
  clearEvidenceFocus();
  state.activeTab = tab;
  try {
    if (location.hash.startsWith("#evidence-")) {
      history.replaceState(history.state, "", `${location.pathname}${location.search}`);
    }
  } catch (error) {
  }
  render();
}

function attachContentHandlers() {
  const content = el("content");
  content.querySelectorAll("[data-evidence-id]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.preventDefault();
      openEvidenceInTranscript(node.dataset.evidenceId);
    });
  });
  const back = content.querySelector("[data-evidence-back]");
  if (back) {
    back.addEventListener("click", returnFromEvidence);
  }
}

function evidenceChipsHtml(item, evidence) {
  const evidenceIds = Array.isArray(item.evidence_ids) ? item.evidence_ids : [];
  return evidenceIds.map((id) => {
    const ev = evidence.get(id);
    const label = ev ? `${id} ${ev.start || ""}` : id;
    const rowLabel = ev ? ` / ${evidenceRowLabel(ev)}` : "";
    return `<a class="badge hot evidence-link" href="#evidence-${encodeURIComponent(id)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(label + rowLabel)}</a>`;
  }).join("");
}

function itemAttributeMap(item) {
  const result = new Map();
  (Array.isArray(item?.attributes) ? item.attributes : []).forEach((attribute) => {
    const key = String(attribute?.key || "");
    if (key) result.set(key, String(attribute?.value || ""));
  });
  return result;
}

function itemHtml(item, evidence, sectionName = "") {
  const chips = evidenceChipsHtml(item, evidence);
  const meta = [
    item.status ? `Status: ${item.status}` : "",
    item.owner ? `Owner: ${item.owner}` : "",
    item.due ? `Due: ${item.due}` : "",
    item.confidence ? `Confidence: ${item.confidence}` : ""
  ].filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
  const attributes = Array.from(itemAttributeMap(item).entries()).map(([key, value]) => `<span class="attribute-chip"><strong>${escapeHtml(key)}:</strong> ${escapeHtml(value)}</span>`).join("");
  const grounding = item.grounding_status === "missing_required_evidence"
    ? '<div class="grounding-warning">Required transcript evidence is missing. Review this item before use.</div>'
    : "";
  return `
    <article class="item" dir="auto">
      <div class="item-title-row">
        ${sectionIcon(sectionName)}
        <h3>${escapeHtml(item.title || "Untitled")}</h3>
      </div>
      <p>${escapeHtml(item.body || item.summary || "")}</p>
      ${attributes ? `<div class="attribute-list">${attributes}</div>` : ""}
      ${grounding}
      <div class="item-footer">${meta}${chips}</div>
    </article>
  `;
}

function tableSectionItems(section, definition, evidence) {
  const items = Array.isArray(section.items) ? section.items : [];
  const fields = Array.isArray(definition.output_fields) ? definition.output_fields : [];
  const headings = fields.map((field) => `<th>${escapeHtml(field.label || field.key)}</th>`).join("");
  const rows = items.map((item) => {
    const attributes = itemAttributeMap(item);
    return `<tr dir="auto">
      <td><strong>${escapeHtml(item.title || "Untitled")}</strong><br><span class="meta">${escapeHtml(item.body || "")}</span>${item.grounding_status === "missing_required_evidence" ? '<div class="grounding-warning">Evidence missing</div>' : ""}</td>
      ${fields.map((field) => `<td>${escapeHtml(attributes.get(field.key) || "—")}</td>`).join("")}
      <td><div class="item-footer">${evidenceChipsHtml(item, evidence) || '<span class="badge warn">No evidence</span>'}</div></td>
    </tr>`;
  }).join("");
  return `<div class="report-table-wrap"><table class="report-table"><thead><tr><th>Item</th>${headings}<th>Evidence</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderedSectionItems(section, definition, evidence) {
  const items = Array.isArray(section.items) ? section.items : [];
  if (!items.length) return `<div class="empty">No ${escapeHtml(definition.title || section.section || "items")} extracted.</div>`;
  const kind = definition.render_kind || "cards";
  if (kind === "table") return tableSectionItems(section, definition, evidence);
  const cards = items.map((item) => itemHtml(item, evidence, definition.key)).join("");
  if (kind === "timeline") return `<div class="timeline-list">${cards}</div>`;
  if (kind === "quotes") return `<div class="quote-layout">${cards}</div>`;
  return cards;
}

function sectionHtml(title, sectionNames) {
  const evidence = evidenceLookup();
  const blocks = sectionNames.map((name) => {
    const section = getSection(name);
    const items = Array.isArray(section.items) ? section.items : [];
    const definition = section.definition || reportSectionDefinitions().find((item) => item.key === name) || {key: name, title: sectionLabel(name), render_kind: "cards", output_fields: []};
    const body = renderedSectionItems(section, definition, evidence);
    return `
      <div class="section-block">
        <h3 class="section-heading">
          ${sectionIcon(name)}
          <span dir="auto">${escapeHtml(definition.title || sectionLabel(name))}</span>
          <span class="section-count">${escapeHtml(pluralLabel(items.length, "item"))}</span>
        </h3>
        ${section.summary ? `<div class="item" dir="auto"><p>${escapeHtml(section.summary)}</p></div>` : ""}
        ${body}
      </div>
    `;
  }).join("");
  return `<div class="section-grid" aria-label="${escapeHtml(title)}">${blocks}</div>`;
}

function evidenceHtml() {
  const items = state.report?.evidence_index || [];
  if (!items.length) return '<div class="empty">No evidence index available.</div>';
  return `<div class="evidence-list">${items.map((item) => `
    <article class="item" data-evidence-card="${escapeHtml(item.id)}">
      <h3>${escapeHtml(item.id)}: ${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.summary)}</p>
      <p class="quote">${escapeHtml(item.quote_excerpt || "")}</p>
      <div class="item-footer">
        <span class="badge">${escapeHtml(item.start || "")} - ${escapeHtml(item.end || "")}</span>
        <span class="badge">${escapeHtml((item.speakers || []).join(", "))}</span>
        <span class="badge">${escapeHtml(item.confidence || "")}</span>
        <button class="badge hot evidence-link" type="button" data-evidence-id="${escapeHtml(item.id)}">${escapeHtml(evidenceRowLabel(item))}</button>
      </div>
    </article>
  `).join("")}</div>`;
}

function transcriptHtml() {
  const rows = state.transcriptRows || [];
  if (!rows.length) return '<div class="empty">No transcript rows available.</div>';
  const evidence = state.activeEvidenceId ? evidenceLookup().get(state.activeEvidenceId) : null;
  const highlighted = new Set(state.highlightRowIds || []);
  const focus = evidence ? `
    <div class="transcript-focus">
      <div class="transcript-focus-title">
        <strong>${escapeHtml(state.activeEvidenceId)}: ${escapeHtml(evidence.title || "")}</strong>
        <span class="meta">${escapeHtml(evidence.summary || evidence.quote_excerpt || "")}</span>
      </div>
      <button class="btn" type="button" data-evidence-back="1">Back</button>
    </div>
  ` : "";
  return `${focus}<div class="transcript-list">${rows.map((row, index) => {
    const aliases = transcriptRowAliases(row, index);
    const isHighlighted = aliases.some((rowId) => highlighted.has(rowId));
    return `
    <article class="item transcript-row${isHighlighted ? " evidence-hit" : ""}" id="${escapeHtml(transcriptRowElementId(row, index))}" data-row-id="${escapeHtml(aliases[0] || "")}" data-row-aliases="${escapeHtml(encodedRowAliases(row, index))}" tabindex="-1">
      <h3>${escapeHtml(row.speaker_name || row.assigned_speaker || "Speaker")} <span class="meta">${Number(row.start || 0).toFixed(1)}s - ${Number(row.end || 0).toFixed(1)}s</span></h3>
      <p>${escapeHtml(row.text || "")}</p>
      ${isHighlighted ? `<div class="item-footer"><span class="badge hot">${escapeHtml(state.activeEvidenceId)}</span></div>` : ""}
    </article>
  `;
  }).join("")}</div>`;
}

function renderContent() {
  if (!state.sessionId) {
    el("content").innerHTML = '<div class="empty">Select a session.</div>';
    return;
  }
  if (!state.report) {
    el("content").innerHTML = '<div class="empty">No current report for this transcript revision.</div>';
    return;
  }
  const tab = state.activeTab;
  if (tab.startsWith("section:")) {
    const sectionKey = tab.slice("section:".length);
    const definition = reportSectionDefinitions().find((item) => item.key === sectionKey) || {title: sectionLabel(sectionKey)};
    el("content").innerHTML = sectionHtml(definition.title || sectionLabel(sectionKey), [sectionKey]);
  } else if (tab === "evidence") {
    el("content").innerHTML = evidenceHtml();
  } else if (tab === "transcript") {
    el("content").innerHTML = transcriptHtml();
  }
}

function render() {
  el("provider").innerHTML = providerLabel();
  renderProviderControls();
  el("generateBtn").disabled = !state.sessionId || state.generating;
  el("headerGenerateBtn").disabled = !state.sessionId || state.generating;
  el("headerTranscriptBtn").disabled = !state.sessionId || !state.transcriptRows.length;
  el("headerRefreshBtn").disabled = !state.sessionId || state.generating;
  const deleteButton = el("deleteReportBtn");
  deleteButton.disabled = !state.sessionId || !state.reportAvailable || state.generating;
  deleteButton.classList.toggle("confirming", state.confirmDelete && !deleteButton.disabled);
  const deleteLabel = state.confirmDelete && !deleteButton.disabled ? "Confirm" : "Delete";
  el("deleteReportLabel").textContent = deleteLabel;
  deleteButton.title = state.confirmDelete ? "Confirm cached report deletion" : "Delete cached report";
  deleteButton.setAttribute("aria-label", deleteButton.title);
  renderTemplateControls();
  renderSessions();
  renderTabs();
  renderHeader();
  renderProgress();
  renderContent();
  attachContentHandlers();
  setStatus(state.status);
}

async function refreshSelectedReport() {
  if (!state.sessionId || state.generating) return;
  await loadSessions(false);
  await selectSession(state.sessionId);
}

function openTranscriptTab() {
  if (!state.sessionId || !state.transcriptRows.length) return;
  clearEvidenceFocus();
  state.activeTab = "transcript";
  render();
}

async function applyProviderConfig() {
  if (state.generating || state.applyingProvider) return;
  const draft = state.providerDraft;
  const model = String(draft.model || "").trim();
  if (!model) {
    setStatus("Model is required");
    render();
    return;
  }
  state.applyingProvider = true;
  clearDeleteConfirm();
  setStatus("Switching provider");
  render();
  try {
    const data = await api("/api/llm-config", {
      method: "POST",
      body: JSON.stringify({
        provider: draft.provider,
        model,
        base_url: String(draft.base_url || "").trim()
      })
    });
    state.config = data.config || state.config;
    syncProviderDraftFromConfig();
    await loadSessions(false);
    if (state.sessionId) {
      await selectSession(state.sessionId);
    } else {
      setStatus("Provider switched");
    }
  } catch (error) {
    setStatus(error.message);
  } finally {
    state.applyingProvider = false;
    render();
  }
}

async function selectSession(sessionId) {
  state.sessionId = sessionId;
  state.report = null;
  state.reportAvailable = false;
  state.transcriptRows = [];
  clearEvidenceFocus();
  clearDeleteConfirm();
  if (!state.generating) {
    state.generationJob = null;
  }
  setStatus("Loading report");
  render();
  try {
    const data = await api(`/api/report?session_id=${encodeURIComponent(sessionId)}&template_id=${encodeURIComponent(state.templateId)}`);
    state.report = data.report;
    state.reportAvailable = Boolean(data.available);
    state.transcriptRows = data.transcript_rows || [];
    ensureActiveTab();
    setStatus(data.available ? "Cached report loaded" : data.stale ? "Cached report is stale" : "Ready");
    render();
    if (!data.available && state.config.auto_generate && state.templateId === state.config.standard_template_id) {
      await generateReport();
    }
  } catch (error) {
    setStatus(error.message);
    render();
  }
}

async function generateReport() {
  if (!state.sessionId || state.generating) return;
  state.generating = true;
  clearEvidenceFocus();
  clearDeleteConfirm();
  state.generationJob = {
    status: "queued",
    stage: "queued",
    message: "Queued report generation",
    detail: "",
    percent: 0,
    current: 0,
    total: 0,
    events: []
  };
  setStatus("Generating report");
  render();
  try {
    const data = await api("/api/generate-async", {
      method: "POST",
      body: JSON.stringify({session_id: state.sessionId, template_id: state.templateId})
    });
    state.generationJob = data.job;
    await pollGenerationJob(data.job.job_id);
    const reportData = await api(`/api/report?session_id=${encodeURIComponent(state.sessionId)}&template_id=${encodeURIComponent(state.templateId)}`);
    state.report = reportData.report;
    state.reportAvailable = Boolean(reportData.available);
    state.transcriptRows = reportData.transcript_rows || [];
    setStatus("Report generated");
  } catch (error) {
    state.generationJob = {
      ...(state.generationJob || {}),
      status: "failed",
      stage: "failed",
      message: "Report generation failed",
      detail: error.message,
      error: error.message
    };
    setStatus(error.message);
  } finally {
    state.generating = false;
    await loadSessions(false);
    render();
  }
}

async function deleteReport() {
  if (!state.sessionId || !state.reportAvailable || state.generating) return;
  if (!state.confirmDelete) {
    state.confirmDelete = true;
    setStatus("Click Confirm to delete the cached report");
    render();
    return;
  }
  clearEvidenceFocus();
  clearDeleteConfirm();
  state.generationJob = null;
  setStatus("Deleting report");
  render();
  try {
    const data = await api("/api/delete-report", {
      method: "POST",
      body: JSON.stringify({session_id: state.sessionId, template_id: state.templateId})
    });
    state.report = null;
    state.reportAvailable = false;
    state.transcriptRows = data.transcript_rows || state.transcriptRows || [];
    state.activeTab = reportTabs()[0]?.[0] || "transcript";
    setStatus(data.deleted ? "Report deleted" : "No cached report found");
    await loadSessions(false);
  } catch (error) {
    setStatus(error.message);
  } finally {
    render();
  }
}

async function pollGenerationJob(jobId) {
  for (;;) {
    const data = await api(`/api/generate-status?job_id=${encodeURIComponent(jobId)}`);
    const job = data.job || {};
    state.generationJob = job;
    setStatus(job.message || "Generating report");
    render();
    if (job.status === "succeeded") {
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.detail || "Report generation failed");
    }
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

async function loadSessions(selectFirst = true) {
  const data = await api("/api/sessions");
  state.sessions = data.sessions || [];
  if (selectFirst && state.sessions.length && !state.sessionId) {
    await selectSession(state.sessions[0].id);
  }
}

async function boot() {
  try {
    state.config = (await api("/api/config")).config || {};
    syncProviderDraftFromConfig();
    await loadTemplates(state.config.standard_template_id || "");
    await loadSessions(true);
    render();
  } catch (error) {
    setStatus(error.message);
    render();
  }
}

el("generateBtn").addEventListener("click", generateReport);
el("deleteReportBtn").addEventListener("click", deleteReport);
el("headerGenerateBtn").addEventListener("click", generateReport);
el("headerTranscriptBtn").addEventListener("click", openTranscriptTab);
el("headerRefreshBtn").addEventListener("click", () => refreshSelectedReport().catch((error) => setStatus(error.message)));
el("refreshBtn").addEventListener("click", () => loadSessions(false).then(render).catch((error) => setStatus(error.message)));
el("sessionSearch").addEventListener("input", renderSessions);
el("templateSelect").addEventListener("change", async () => {
  state.templateId = el("templateSelect").value;
  state.report = null;
  state.reportAvailable = false;
  clearEvidenceFocus();
  state.activeTab = sectionTabId(currentTemplate()?.sections?.[0]?.key || "");
  render();
  if (state.sessionId) await selectSession(state.sessionId);
});
el("newTemplateBtn").addEventListener("click", () => openTemplateBuilder("new"));
el("inspectTemplateBtn").addEventListener("click", () => openTemplateBuilder("inspect"));
el("cloneTemplateBtn").addEventListener("click", () => openTemplateBuilder("clone"));
el("deleteTemplateBtn").addEventListener("click", () => deleteSelectedTemplate().catch((error) => setStatus(error.message)));
el("closeBuilderBtn").addEventListener("click", closeTemplateBuilder);
el("cancelBuilderBtn").addEventListener("click", closeTemplateBuilder);
el("saveTemplateBtn").addEventListener("click", saveTemplateBuilder);
el("templateBuilder").addEventListener("click", (event) => {
  if (event.target === el("templateBuilder")) closeTemplateBuilder();
});
el("closeProgressBtn").addEventListener("click", closeProgressOverlay);
el("closeProgressFooterBtn").addEventListener("click", closeProgressOverlay);
el("progressOverlay").addEventListener("click", (event) => {
  if (event.target === el("progressOverlay")) closeProgressOverlay();
});
el("llmProviderSelect").addEventListener("change", () => {
  const option = providerOption(el("llmProviderSelect").value);
  state.providerDraft.provider = option.id || el("llmProviderSelect").value;
  state.providerDraft.model = option.models?.[0] || "";
  state.providerDraft.base_url = option.default_base_url || "";
  renderProviderControls();
  if (option.id === "openai" || option.id === "openrouter") {
    loadProviderModels();
  }
});
el("llmModelSelect").addEventListener("change", () => {
  const value = el("llmModelSelect").value;
  if (value === "__custom__") {
    if (providerModelOptions(selectedProviderOption()).includes(state.providerDraft.model)) {
      state.providerDraft.model = "";
    }
  } else {
    state.providerDraft.model = value;
  }
  renderProviderControls();
  if (value === "__custom__") {
    el("llmModelInput").focus();
  }
});
el("llmModelInput").addEventListener("input", () => {
  state.providerDraft.model = el("llmModelInput").value;
  renderProviderControls();
});
el("llmBaseUrlInput").addEventListener("input", () => {
  state.providerDraft.base_url = el("llmBaseUrlInput").value;
  renderProviderControls();
});
el("applyProviderBtn").addEventListener("click", () => applyProviderConfig());
el("loadModelsBtn").addEventListener("click", () => loadProviderModels());
window.addEventListener("popstate", () => {
  if (state.activeEvidenceId) {
    const tab = state.evidenceReturnTab || reportTabs()[0]?.[0] || "transcript";
    clearEvidenceFocus();
    state.activeTab = tab;
    render();
  }
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el("templateBuilder").hidden) closeTemplateBuilder();
  if (event.key === "Escape" && !el("progressOverlay").hidden) closeProgressOverlay();
});
boot();

export { render, selectSession };
