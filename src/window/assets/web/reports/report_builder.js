import { render, selectSession } from "./app.js";

const state = {
  config: {},
  sessions: [],
  templates: [],
  templateId: "",
  sessionId: "",
  report: null,
  reportAvailable: false,
  transcriptRows: [],
  activeTab: "",
  generating: false,
  generationJob: null,
  activeEvidenceId: "",
  highlightRowIds: [],
  evidenceReturnTab: "",
  confirmDelete: false,
  providerDraft: {provider: "", model: "", base_url: ""},
  providerModels: {},
  loadingModels: false,
  applyingProvider: false,
  status: "",
  builderTemplate: null,
  builderReadOnly: false,
  builderMode: "new"
};

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function domId(value) {
  return String(value ?? "").replace(/[^A-Za-z0-9_-]/g, "_") || "row";
}

function transcriptRowDomId(rowId) {
  return `transcript-row-${domId(rowId)}`;
}

function uniqueValues(values) {
  const result = [];
  const seen = new Set();
  values.forEach((value) => {
    const text = String(value ?? "").trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    result.push(text);
  });
  return result;
}

function normalizedFallbackRowId(row, index) {
  const rawIndex = Number(row?.index);
  const normalizedIndex = Number.isFinite(rawIndex) && rawIndex !== 0
    ? Math.trunc(rawIndex)
    : index + 1;
  return `row_${normalizedIndex}`;
}

function transcriptRowAliases(row, index) {
  return uniqueValues([
    row?.row_id,
    row?.id,
    normalizedFallbackRowId(row, index),
    `row-${index + 1}`,
  ]);
}

function transcriptRowElementId(row, index) {
  const primary = transcriptRowAliases(row, index)[0] || `row_${index + 1}`;
  return `${transcriptRowDomId(primary)}-${index + 1}`;
}

function encodedRowAliases(row, index) {
  return transcriptRowAliases(row, index).map((value) => encodeURIComponent(value)).join(" ");
}

function evidenceRowIds(item) {
  return Array.isArray(item?.row_ids) ? item.row_ids.map((value) => String(value)) : [];
}

function evidenceRowLabel(item) {
  const count = evidenceRowIds(item).length;
  if (!count) return "Transcript";
  return `${count} transcript row${count === 1 ? "" : "s"}`;
}

function clearEvidenceFocus() {
  state.activeEvidenceId = "";
  state.highlightRowIds = [];
  state.evidenceReturnTab = "";
}

function clearDeleteConfirm() {
  state.confirmDelete = false;
}

function setStatus(value) {
  state.status = value || "";
  el("status").innerHTML = reportStatusHtml();
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function reportStatusHtml() {
  const session = state.sessions.find((item) => item.id === state.sessionId) || {};
  const report = state.report || {};
  const status = state.status || (state.sessionId ? "Ready" : "Select a session");
  const parts = [
    `<span class="status-state"><span class="status-dot" aria-hidden="true">&#10003;</span>${escapeHtml(status)}</span>`
  ];
  const generated = formatDateTime(report.generated_at || report.updated_at);
  if (generated) {
    parts.push(`<span class="status-separator">/</span><span class="status-muted">Generated ${escapeHtml(generated)}</span>`);
  }
  const rows = Number(state.transcriptRows?.length || session.transcript_rows || 0);
  const speakers = Number(session.speaker_count || 0);
  if (rows || speakers) {
    const label = [
      rows ? `${rows} transcript rows` : "",
      speakers ? `${speakers} speakers` : ""
    ].filter(Boolean).join(" / ");
    parts.push(`<span class="status-separator">/</span><span class="status-muted">${escapeHtml(label)}</span>`);
  }
  return parts.join("");
}

function providerOptions() {
  return Array.isArray(state.config.providers) ? state.config.providers : [];
}

function providerOption(provider) {
  return providerOptions().find((option) => option.id === provider) || providerOptions()[0] || {};
}

function syncProviderDraftFromConfig() {
  const cfg = state.config || {};
  state.providerDraft = {
    provider: cfg.provider || providerOptions()[0]?.id || "llama_cpp",
    model: cfg.model || "",
    base_url: cfg.base_url || ""
  };
}

function selectedProviderOption() {
  return providerOption(state.providerDraft.provider || state.config.provider);
}

function providerModelOptions(option) {
  const provider = option.id || state.providerDraft.provider || state.config.provider || "";
  return uniqueValues([
    ...(state.providerModels[provider] || []),
    ...(option.models || []),
    state.providerDraft.model,
  ]);
}

function renderProviderControls() {
  const options = providerOptions();
  const providerSelect = el("llmProviderSelect");
  providerSelect.innerHTML = options.map((option) => `
    <option value="${escapeHtml(option.id)}">${escapeHtml(option.label || option.id)}</option>
  `).join("");
  providerSelect.value = state.providerDraft.provider || state.config.provider || options[0]?.id || "";
  const option = selectedProviderOption();
  const models = providerModelOptions(option);
  const modelSelect = el("llmModelSelect");
  const modelIsPreset = models.includes(state.providerDraft.model);
  modelSelect.innerHTML = models.map((model) => `
    <option value="${escapeHtml(model)}">${escapeHtml(model)}</option>
  `).join("") + '<option value="__custom__">Custom model...</option>';
  modelSelect.value = modelIsPreset ? state.providerDraft.model : "__custom__";
  const modelInput = el("llmModelInput");
  modelInput.hidden = modelIsPreset;
  modelInput.value = modelIsPreset ? "" : (state.providerDraft.model || "");
  el("llmBaseUrlInput").value = state.providerDraft.base_url || option.default_base_url || "";
  const requiresKey = Boolean(option.requires_api_key);
  const keyOk = !requiresKey || Boolean(state.config.api_key_configured);
  const modelCount = state.providerModels[option.id]?.length || 0;
  const keyText = state.loadingModels
    ? "Loading models..."
    : requiresKey
    ? `${escapeHtml(option.api_key_env_var || "API key")}: ${keyOk ? "configured" : "missing"}`
    : "No API key required";
  el("providerHint").innerHTML = `${keyText}${modelCount ? ` / ${modelCount} loaded` : ""}`;
  el("loadModelsBtn").disabled = state.loadingModels || state.generating || state.applyingProvider;
  el("applyProviderBtn").disabled = state.generating || state.applyingProvider || !state.providerDraft.model.trim();
}

async function loadProviderModels() {
  if (state.loadingModels) return;
  const draft = state.providerDraft;
  const previousModel = draft.model;
  const previousDefault = selectedProviderOption().models?.[0] || "";
  state.loadingModels = true;
  setStatus("Loading models");
  render();
  try {
    const query = new URLSearchParams({
      provider: draft.provider,
      base_url: draft.base_url || selectedProviderOption().default_base_url || ""
    });
    const data = await api(`/api/llm-models?${query.toString()}`);
    state.providerModels[data.provider] = data.models || [];
    if (
      state.providerModels[data.provider]?.length
      && (!previousModel || previousModel === previousDefault)
    ) {
      state.providerDraft.model = state.providerModels[data.provider][0];
    }
    setStatus(state.providerModels[data.provider]?.length ? "Models loaded" : "No models returned");
  } catch (error) {
    setStatus(error.message);
  } finally {
    state.loadingModels = false;
    render();
  }
}

function providerLabel() {
  const cfg = state.config || {};
  const mode = cfg.mock_llm ? "mock" : cfg.provider;
  const option = providerOption(cfg.provider);
  const requiresKey = Boolean(option.requires_api_key);
  const keyOk = !requiresKey || Boolean(cfg.api_key_configured);
  const readyText = cfg.mock_llm ? "Mock" : keyOk ? "Ready" : "Missing key";
  const warnClass = keyOk ? "" : " warn";
  return `
    <div class="provider-kicker">Provider / Model</div>
    <div class="provider-connected${warnClass}"><span class="provider-health${warnClass}" aria-hidden="true"></span>${escapeHtml(readyText)}</div>
    <div class="provider-model"><strong>${escapeHtml(mode)}</strong> / ${escapeHtml(cfg.model || "local")}</div>
    <div class="provider-url">${escapeHtml(cfg.base_url || "no base URL")}</div>
  `;
}

function sessionIcon(kind) {
  if (kind === "demo_transcript") {
    return '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3h7l4 4v14H7V3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 3v5h5M10 12h6M10 16h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 5h10a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3Z" stroke="currentColor" stroke-width="2"/><path d="m10 9 5 3-5 3V9Z" fill="currentColor"/></svg>';
}

function tabIcon(name) {
  const icons = {
    list: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 12l2 2 4-5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    tasks: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9 6h11M9 12h11M9 18h11M4 6l1 1 2-2M4 12l1 1 2-2M4 18l1 1 2-2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    question: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M9.1 9a3 3 0 1 1 4.8 2.4c-.9.6-1.4 1.1-1.4 2.1M12 17h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    shield: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3 20 6v5c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10V6l8-3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8v5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 16h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    file: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3h7l4 4v14H7V3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 3v5h5M10 13h6M10 17h4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.1.1l2-2a5 5 0 0 0-7.1-7.1l-1.1 1.1M14 11a5 5 0 0 0-7.1-.1l-2 2A5 5 0 0 0 12 20l1.1-1.1" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    wave: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 10v4M8 7v10M12 5v14M16 8v8M20 11v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
  };
  return icons[name] || "";
}

function sectionIcon(name) {
  const groups = {
    executive_summary: "spark",
    structured_brief: "file",
    speaker_map: "users",
    speaker_participation: "wave",
    discussion_threads: "list",
    decisions: "check",
    deadlines: "calendar",
    disagreements: "alert",
    action_items: "tasks",
    open_questions: "question",
    ask_this_meeting: "question",
    risks: "shield"
  };
  const icons = {
    alert: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 4.4 2.7 18a2 2 0 0 0 1.7 3h15.2a2 2 0 0 0 1.7-3L13.7 4.4a2 2 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    calendar: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 3v4M17 3v4M4 9h16M6 5h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    spark: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3ZM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
    users: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2M9.5 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
  };
  return icons[groups[name]] || tabIcon(groups[name] || name) || tabIcon("file");
}

function sectionLabel(name) {
  return name.replaceAll("_", " ");
}

function pluralLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function currentTemplate() {
  return state.templates.find((item) => item.template_id === state.templateId) || null;
}

function effectiveReportTemplate() {
  return state.report?.report_template || currentTemplate() || {sections: []};
}

function reportSectionDefinitions() {
  const definitions = effectiveReportTemplate()?.sections;
  return Array.isArray(definitions) ? definitions : [];
}

function sectionTabId(key) {
  return `section:${key}`;
}

function sectionIconName(definition) {
  const kind = String(definition?.render_kind || "cards");
  return kind === "timeline" ? "list" : kind === "quotes" ? "question" : kind === "table" ? "tasks" : "file";
}

function reportTabs() {
  const sectionTabs = reportSectionDefinitions().map((definition) => [
    sectionTabId(definition.key),
    definition.title || sectionLabel(definition.key),
    sectionIconName(definition),
  ]);
  return [...sectionTabs, ["evidence", "Evidence", "link"], ["transcript", "Transcript", "wave"]];
}

function ensureActiveTab() {
  const available = reportTabs();
  if (!available.some(([id]) => id === state.activeTab)) {
    state.activeTab = available[0]?.[0] || "transcript";
  }
}

function normalizeBuilderKey(value, fallback = "section") {
  const key = String(value || "").toLowerCase().normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
  return key || fallback;
}

function defaultOutputField(index = 1) {
  return {key: `field_${index}`, label: `Field ${index}`, type: "text", description: "", options: []};
}

function defaultSection(index = 1) {
  return {
    key: `section_${index}`,
    title: `Section ${index}`,
    objective: "Describe exactly what this section should find in the final speaker-labelled transcript.",
    max_items: 6,
    evidence_required: true,
    render_kind: "cards",
    sort_order: "relevance",
    output_fields: [],
  };
}

function newTemplateDraft() {
  return {
    schema_version: "report_template_v1",
    name: "Untitled custom report",
    description: "A reusable evidence-grounded report.",
    version: 1,
    builtin: false,
    language_mode: "inherit",
    privacy_policy: "inherit",
    sections: [defaultSection(1)],
  };
}

function cleanBuilderTemplate(template) {
  const clean = deepCopy(template || {});
  delete clean.read_only;
  delete clean.source_kind;
  delete clean.revision_hash;
  clean.builtin = false;
  clean.version = Number(clean.version || 1);
  clean.sections = (clean.sections || []).map((section) => ({
    key: normalizeBuilderKey(section.key, "section"),
    title: String(section.title || "").trim(),
    objective: String(section.objective || "").trim(),
    max_items: Math.max(1, Math.min(20, Number(section.max_items || 6))),
    evidence_required: Boolean(section.evidence_required),
    render_kind: section.render_kind || "cards",
    sort_order: section.sort_order || "relevance",
    output_fields: (section.output_fields || []).map((field) => ({
      key: normalizeBuilderKey(field.key, "field"),
      label: String(field.label || "").trim(),
      type: field.type || "text",
      description: String(field.description || "").trim(),
      options: Array.isArray(field.options) ? field.options.map((item) => String(item).trim()).filter(Boolean) : [],
    })),
  }));
  return clean;
}

function renderTemplateControls() {
  const select = el("templateSelect");
  const predefined = state.templates.filter((item) => item.builtin);
  const custom = state.templates.filter((item) => !item.builtin);
  const optionHtml = (item) => `<option value="${escapeHtml(item.template_id)}" ${item.template_id === state.templateId ? "selected" : ""}>${escapeHtml(item.name)}</option>`;
  select.innerHTML = [
    `<optgroup label="Predefined reports">${predefined.map(optionHtml).join("")}</optgroup>`,
    custom.length ? `<optgroup label="Custom reports">${custom.map(optionHtml).join("")}</optgroup>` : "",
  ].join("");
  const template = currentTemplate();
  el("templateSummary").textContent = template?.description || "Choose or create a report template.";
  el("templateBadges").innerHTML = template ? [
    `<span class="badge ${template.builtin ? "hot" : ""}">${template.builtin ? "Predefined" : "Custom"}</span>`,
    `<span class="badge">${escapeHtml(template.language_mode === "inherit" ? "Follow server language" : template.language_mode)}</span>`,
    template.privacy_policy === "local_only" ? '<span class="badge warn">Local only</span>' : "",
    `<span class="badge">${template.sections?.length || 0} sections</span>`,
  ].join("") : "";
  el("inspectTemplateBtn").disabled = !template;
  el("cloneTemplateBtn").disabled = !template;
  el("deleteTemplateBtn").disabled = !template || Boolean(template.builtin);
  el("inspectTemplateBtn").textContent = template?.builtin ? "Inspect" : "Edit";
}

function builderValidationMessage(template) {
  if (!String(template?.name || "").trim()) return "Give the report a name.";
  if (!Array.isArray(template?.sections) || !template.sections.length) return "Add at least one section.";
  if (template.sections.length > 16) return "A report can contain at most 16 sections.";
  const keys = new Set();
  for (const section of template.sections) {
    const key = normalizeBuilderKey(section.key, "");
    if (!key) return "Every section needs a key.";
    if (keys.has(key)) return `Section key '${key}' is duplicated.`;
    keys.add(key);
    if (!String(section.title || "").trim()) return `Section '${key}' needs a title.`;
    if (!String(section.objective || "").trim()) return `Section '${key}' needs an objective.`;
    const fieldKeys = new Set();
    for (const field of section.output_fields || []) {
      const fieldKey = normalizeBuilderKey(field.key, "");
      if (!fieldKey || !String(field.label || "").trim()) return `Every output field in '${key}' needs a key and label.`;
      if (fieldKeys.has(fieldKey)) return `Output field '${fieldKey}' is duplicated in '${key}'.`;
      fieldKeys.add(fieldKey);
      if (field.type === "enum" && !(field.options || []).length) return `Enum field '${fieldKey}' needs at least one option.`;
    }
  }
  return "";
}

function openTemplateBuilder(mode = "inspect") {
  const source = currentTemplate();
  state.builderMode = mode;
  if (mode === "new") {
    state.builderTemplate = newTemplateDraft();
    state.builderReadOnly = false;
  } else if (mode === "clone") {
    state.builderTemplate = cleanBuilderTemplate(source || newTemplateDraft());
    delete state.builderTemplate.template_id;
    state.builderTemplate.name = `${source?.name || "Report"} copy`;
    state.builderTemplate.version = 1;
    state.builderReadOnly = false;
  } else {
    state.builderTemplate = deepCopy(source || newTemplateDraft());
    state.builderReadOnly = Boolean(source?.builtin);
  }
  el("templateBuilder").hidden = false;
  document.body.style.overflow = "hidden";
  renderTemplateBuilder();
}

function closeTemplateBuilder() {
  el("templateBuilder").hidden = true;
  document.body.style.overflow = "";
  state.builderTemplate = null;
  el("builderValidation").textContent = "";
}

function renderTemplateBuilder() {
  const template = state.builderTemplate;
  if (!template) return;
  const disabled = state.builderReadOnly ? "disabled" : "";
  el("builderTitle").textContent = state.builderReadOnly ? "Inspect predefined report" : state.builderMode === "new" ? "Create custom report" : state.builderMode === "clone" ? "Clone report" : "Edit custom report";
  el("builderSubtitle").textContent = state.builderReadOnly
    ? "This is an ordinary report-template document. Clone it to make an editable copy."
    : "Configure flat top-level sections, their output fields, layout, ordering, and evidence rules.";
  const languageCodes = "inherit,af,ar,be,bg,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,he,hi,hr,hu,hy,id,is,it,ja,ka,kk,ko,la,lt,lv,ml,mr,mt,my,nl,nn,no,pl,pt,ro,ru,sa,sd,sk,sl,sq,sr,sv,ta,te,th,tr,uk,ur,vi,zh".split(",");
  const sections = (template.sections || []).map((section, sectionIndex) => {
    const fields = (section.output_fields || []).map((field, fieldIndex) => `
      <div class="builder-field-row" data-field-row="${fieldIndex}">
        <label>Key<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="key" value="${escapeHtml(field.key || "")}"></label>
        <label>Label<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="label" value="${escapeHtml(field.label || "")}"></label>
        <label>Type<select ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="type">${["text","enum","speaker","date","timestamp","boolean","number"].map((type) => `<option ${field.type === type ? "selected" : ""}>${type}</option>`).join("")}</select></label>
        <label>Description<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="description" value="${escapeHtml(field.description || "")}"></label>
        <label>Enum options<input ${disabled} data-section-index="${sectionIndex}" data-field-index="${fieldIndex}" data-field-prop="options" value="${escapeHtml((field.options || []).join(", "))}" placeholder="open, closed"></label>
        <button class="btn danger" type="button" ${disabled} data-field-action="remove" data-section-index="${sectionIndex}" data-field-index="${fieldIndex}">Remove</button>
      </div>
    `).join("");
    return `
      <article class="builder-card builder-section-card" data-section-card="${sectionIndex}">
        <div class="builder-section-heading">
          <h3 dir="auto">${escapeHtml(section.title || `Section ${sectionIndex + 1}`)}</h3>
          <div class="builder-inline-actions">
            <button class="btn" type="button" ${disabled} data-section-action="up" data-section-index="${sectionIndex}">↑ Up</button>
            <button class="btn" type="button" ${disabled} data-section-action="down" data-section-index="${sectionIndex}">↓ Down</button>
            <button class="btn danger" type="button" ${disabled} data-section-action="remove" data-section-index="${sectionIndex}">Remove</button>
          </div>
        </div>
        <div class="builder-section-grid">
          <label class="span-2">Title<input ${disabled} data-section-index="${sectionIndex}" data-section-prop="title" value="${escapeHtml(section.title || "")}"></label>
          <label class="span-2">Key<input ${disabled} data-section-index="${sectionIndex}" data-section-prop="key" value="${escapeHtml(section.key || "")}"></label>
          <label class="wide">What should this section find?<textarea ${disabled} data-section-index="${sectionIndex}" data-section-prop="objective">${escapeHtml(section.objective || "")}</textarea></label>
          <label>Maximum items<input ${disabled} type="number" min="1" max="20" data-section-index="${sectionIndex}" data-section-prop="max_items" value="${Number(section.max_items || 6)}"></label>
          <label>Layout<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="render_kind">${["cards","table","timeline","quotes"].map((kind) => `<option ${section.render_kind === kind ? "selected" : ""}>${kind}</option>`).join("")}</select></label>
          <label>Order<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="sort_order">${["relevance","chronological","severity"].map((kind) => `<option ${section.sort_order === kind ? "selected" : ""}>${kind}</option>`).join("")}</select></label>
          <label>Evidence<select ${disabled} data-section-index="${sectionIndex}" data-section-prop="evidence_required"><option value="true" ${section.evidence_required ? "selected" : ""}>Required</option><option value="false" ${!section.evidence_required ? "selected" : ""}>Optional</option></select></label>
          <div class="builder-fields">
            <div class="builder-field-heading"><strong>Configurable output fields</strong><button class="btn" type="button" ${disabled} data-section-action="add-field" data-section-index="${sectionIndex}">+ Add field</button></div>
            ${fields || '<div class="meta">No custom fields. Items still include a title, body, status, confidence, and evidence tags.</div>'}
          </div>
        </div>
      </article>
    `;
  }).join("");
  el("builderBody").innerHTML = `
    ${state.builderReadOnly ? '<div class="builder-readonly-note">Predefined reports use the same template format and generator as custom reports. Inspect every setting here, then Clone to customize it.</div>' : ""}
    <section class="builder-card">
      <div class="builder-meta-grid">
        <label>Report name<input ${disabled} data-template-prop="name" value="${escapeHtml(template.name || "")}"></label>
        <label>Report language<select ${disabled} data-template-prop="language_mode">${languageCodes.map((code) => `<option value="${code}" ${template.language_mode === code ? "selected" : ""}>${code === "inherit" ? "Follow report server" : code}</option>`).join("")}</select></label>
        <label>Privacy<select ${disabled} data-template-prop="privacy_policy"><option value="inherit" ${template.privacy_policy === "inherit" ? "selected" : ""}>Follow provider settings</option><option value="local_only" ${template.privacy_policy === "local_only" ? "selected" : ""}>Local models only</option></select></label>
        <label class="wide">Description<textarea ${disabled} data-template-prop="description">${escapeHtml(template.description || "")}</textarea></label>
      </div>
    </section>
    <div class="builder-section-list">${sections}</div>
    <button class="btn primary" type="button" ${disabled} id="addSectionBtn">+ Add top-level section</button>
  `;
  el("saveTemplateBtn").hidden = state.builderReadOnly;
  el("builderValidation").textContent = state.builderReadOnly ? "" : builderValidationMessage(template);
  attachBuilderHandlers();
}

function attachBuilderHandlers() {
  const template = state.builderTemplate;
  if (!template || state.builderReadOnly) return;
  el("builderBody").querySelectorAll("[data-template-prop]").forEach((node) => {
    node.addEventListener("input", () => { template[node.dataset.templateProp] = node.value; el("builderValidation").textContent = builderValidationMessage(template); });
    node.addEventListener("change", () => { template[node.dataset.templateProp] = node.value; renderTemplateBuilder(); });
  });
  el("builderBody").querySelectorAll("[data-section-prop]").forEach((node) => {
    const update = () => {
      const section = template.sections[Number(node.dataset.sectionIndex)];
      const prop = node.dataset.sectionProp;
      let value = node.value;
      if (prop === "max_items") value = Number(value);
      if (prop === "evidence_required") value = value === "true";
      if (prop === "key") value = normalizeBuilderKey(value, value);
      section[prop] = value;
      el("builderValidation").textContent = builderValidationMessage(template);
    };
    node.addEventListener("input", update);
    node.addEventListener("change", () => { update(); renderTemplateBuilder(); });
  });
  el("builderBody").querySelectorAll("[data-field-prop]").forEach((node) => {
    const update = () => {
      const field = template.sections[Number(node.dataset.sectionIndex)].output_fields[Number(node.dataset.fieldIndex)];
      const prop = node.dataset.fieldProp;
      field[prop] = prop === "options" ? node.value.split(",").map((item) => item.trim()).filter(Boolean) : prop === "key" ? normalizeBuilderKey(node.value, node.value) : node.value;
      el("builderValidation").textContent = builderValidationMessage(template);
    };
    node.addEventListener("input", update);
    node.addEventListener("change", () => { update(); renderTemplateBuilder(); });
  });
  el("builderBody").querySelectorAll("[data-section-action]").forEach((button) => button.addEventListener("click", () => {
    const index = Number(button.dataset.sectionIndex);
    const action = button.dataset.sectionAction;
    if (action === "remove") template.sections.splice(index, 1);
    if (action === "up" && index > 0) [template.sections[index - 1], template.sections[index]] = [template.sections[index], template.sections[index - 1]];
    if (action === "down" && index < template.sections.length - 1) [template.sections[index + 1], template.sections[index]] = [template.sections[index], template.sections[index + 1]];
    if (action === "add-field") template.sections[index].output_fields.push(defaultOutputField(template.sections[index].output_fields.length + 1));
    renderTemplateBuilder();
  }));
  el("builderBody").querySelectorAll("[data-field-action='remove']").forEach((button) => button.addEventListener("click", () => {
    template.sections[Number(button.dataset.sectionIndex)].output_fields.splice(Number(button.dataset.fieldIndex), 1);
    renderTemplateBuilder();
  }));
  el("addSectionBtn")?.addEventListener("click", () => {
    if (template.sections.length >= 16) return;
    template.sections.push(defaultSection(template.sections.length + 1));
    renderTemplateBuilder();
    el("builderBody").lastElementChild?.scrollIntoView({behavior: "smooth", block: "end"});
  });
}

async function saveTemplateBuilder() {
  if (!state.builderTemplate || state.builderReadOnly) return;
  const error = builderValidationMessage(state.builderTemplate);
  el("builderValidation").textContent = error;
  if (error) return;
  const payload = cleanBuilderTemplate(state.builderTemplate);
  if (state.builderMode === "new" || state.builderMode === "clone") delete payload.template_id;
  try {
    const data = await api("/api/templates/save", {method: "POST", body: JSON.stringify({template: payload})});
    closeTemplateBuilder();
    await loadTemplates(data.template?.template_id || "");
    setStatus("Report template saved");
    if (state.sessionId) await selectSession(state.sessionId);
  } catch (error) {
    el("builderValidation").textContent = error.message;
  }
}

async function deleteSelectedTemplate() {
  const template = currentTemplate();
  if (!template || template.builtin) return;
  if (!confirm(`Delete custom report template '${template.name}'? Cached reports are left untouched.`)) return;
  await api("/api/templates/delete", {method: "POST", body: JSON.stringify({template_id: template.template_id})});
  await loadTemplates(state.config.standard_template_id || "");
  state.report = null;
  state.reportAvailable = false;
  if (state.sessionId) await selectSession(state.sessionId);
  setStatus("Custom report template deleted");
}

async function loadTemplates(preferredTemplateId = "") {
  const data = await api("/api/templates");
  state.templates = data.templates || [];
  const candidate = preferredTemplateId || state.templateId || data.standard_template_id || state.config.standard_template_id;
  state.templateId = state.templates.some((item) => item.template_id === candidate)
    ? candidate
    : state.templates[0]?.template_id || "";
  ensureActiveTab();
  renderTemplateControls();
}


export {
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
};
