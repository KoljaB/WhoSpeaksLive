"""Static HTML for the browser-synced window diarization UI."""

from __future__ import annotations

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Growing Window Speaker Diarization</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:#090b0d;
      --panel:#151715;
      --panel-2:#101210;
      --field:#080a09;
      --line:#343a36;
      --text:#f4f7f4;
      --muted:#9ea89f;
      --accent:#65b891;
      --accent-strong:#2f8f68;
    }
    * { box-sizing: border-box; }
    body { margin:0; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); font-family:Arial, Helvetica, sans-serif; }
    .app { height:100vh; display:grid; grid-template-rows:auto minmax(0,1fr); }
    .topbar { display:flex; gap:12px; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--line); background:var(--panel); }
    .brand { min-width:0; display:flex; align-items:center; gap:10px; }
    .title { font-weight:800; letter-spacing:0; white-space:nowrap; }
    .runtime-state { max-width:30vw; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .transport { flex:0 0 auto; display:flex; gap:8px; align-items:center; }
    button { min-height:36px; border:1px solid #59675d; border-radius:6px; padding:0 14px; font-weight:700; cursor:pointer; background:#20241f; color:var(--text); }
    #start { border-color:#69c99a; background:#1f6f4f; }
    #stop { border-color:#a85656; background:#5d2424; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .layout { min-height:0; display:grid; grid-template-columns:minmax(0,1fr) minmax(330px,380px); }
    .workspace { min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr); }
    .control-panel { min-height:0; overflow:auto; padding:12px; border-left:1px solid var(--line); background:#0d0f0d; display:grid; align-content:start; gap:12px; }
    .control-card, .media-card, .transcript-panel { background:var(--panel-2); border:1px solid var(--line); border-radius:8px; }
    .control-card { padding:10px; }
    .section-title, summary { min-height:30px; display:flex; align-items:center; gap:8px; font-weight:800; color:var(--text); }
    summary { cursor:pointer; list-style-position:outside; }
    .source-grid { display:grid; gap:8px; }
    .source-row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; }
    .mode, .preset, .source, .speaker-panel input, .speaker-panel select { width:100%; min-height:36px; border:1px solid #59675d; border-radius:6px; padding:0 10px; background:var(--field); color:var(--text); }
    .mode:disabled, .preset:disabled, .source:disabled { opacity:.6; }
    .sensitivity { min-height:40px; display:grid; grid-template-columns:auto minmax(0,1fr); align-items:center; gap:6px 10px; color:var(--muted); font-size:12px; padding:8px 0; border-bottom:1px solid var(--line); }
    .sensitivity input { width:100%; accent-color:var(--accent); }
    .sensitivity strong { color:var(--text); font-size:12px; text-align:right; }
    .media-card { margin:12px; padding:12px; display:grid; gap:10px; }
    video { width:100%; max-height:38vh; background:#000; border:1px solid var(--line); }
    audio { width:100%; }
    .youtube-stream { display:none; position:relative; overflow:hidden; width:100%; aspect-ratio:16/9; background:#000; border:1px solid var(--line); }
    .youtube-stream iframe { width:100%; height:100%; border:0; display:block; }
    .youtube-stream.empty iframe { display:none; }
    .stream-hint { display:none; position:absolute; left:10px; right:10px; bottom:10px; min-height:38px; align-items:center; justify-content:center; padding:8px 10px; border:1px solid rgba(244,247,244,.16); border-radius:6px; background:rgba(8,10,9,.84); color:var(--muted); text-align:center; line-height:1.35; font-size:13px; }
    .stream-hint:not(:empty) { display:flex; }
    .youtube-stream.empty .stream-hint { inset:0; height:100%; border:0; border-radius:0; background:transparent; padding:24px; }
    .app.browser-stream video, .app.browser-stream audio { display:none; }
    .app.browser-stream .youtube-stream { display:block; }
    .speaker-panel { display:grid; gap:8px; }
    .speaker-panel[open] { display:grid; }
    .speaker-tools, .speaker-add { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .speaker-tools strong { margin-right:auto; }
    .speaker-panel input[type="file"] { padding:7px 8px; color:var(--muted); }
    .speaker-panel button { min-height:36px; padding:0 10px; }
    .speaker-panel button.recording { border-color:#ef4444; color:#fff; background:#991b1b; }
    .speaker-list { display:grid; gap:6px; max-height:190px; overflow:auto; }
    .speaker-item { display:grid; grid-template-columns:auto minmax(90px,1fr); gap:6px 8px; align-items:center; padding:6px 0; border-bottom:1px solid rgba(52,58,54,.65); }
    .speaker-item input { width:100%; }
    .speaker-source { grid-column:2; color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .status { max-height:230px; overflow:auto; color:var(--muted); font-size:13px; line-height:1.35; padding-top:4px; }
    .transcript-panel { min-height:0; margin:0 12px 12px; display:grid; grid-template-rows:auto minmax(0,1fr); }
    .transcript-panel .section-title { padding:8px 12px; border-bottom:1px solid var(--line); }
    .sentences { min-height:0; overflow:auto; padding:12px; }
    .row { border-bottom:1px solid var(--line); padding:10px 2px; }
    .top { display:flex; gap:10px; align-items:flex-start; justify-content:space-between; margin-bottom:5px; color:var(--muted); font-size:12px; }
    .top-left { min-width:0; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .badge { font-weight:800; border-radius:999px; padding:2px 8px; border:1px solid currentColor; background:rgba(255,255,255,.04); }
    .badge.unknown { color:#c3ccd6; border-color:#7d8997; background:rgba(125,137,151,.13); }
    .badge.new { color:#fff; border-color:#ef4444; background:#b91c1c; text-transform:uppercase; letter-spacing:0; }
    .badge.state { color:#d7dee8; border-color:#59675d; background:rgba(89,103,93,.18); font-weight:700; }
    .text { font-size:17px; line-height:1.38; }
    .row.realtime .text { color:#d7dee8; }
    .prob { flex:0 0 min(200px, 28vw); display:flex; width:min(200px, 28vw); height:8px; overflow:hidden; border:1px solid var(--line); border-radius:4px; background:#0d131a; margin-top:5px; }
    .prob span { display:block; height:100%; min-width:0; }
    @media (max-width: 900px) {
      body { min-height:100dvh; height:auto; overflow:auto; }
      .app { min-height:100dvh; height:auto; display:block; }
      .topbar { position:sticky; top:0; z-index:20; padding:8px 10px; }
      .brand { flex-direction:column; align-items:flex-start; gap:2px; }
      .title { font-size:14px; }
      .runtime-state { max-width:45vw; font-size:12px; }
      .transport button { min-height:44px; padding:0 16px; }
      .layout { min-height:0; display:flex; flex-direction:column; }
      .control-panel { order:1; overflow:visible; border-left:0; border-bottom:1px solid var(--line); padding:10px; }
      .workspace { order:2; display:flex; flex-direction:column; min-height:0; }
      .source-row { grid-template-columns:1fr; }
      .mode, .preset, .source, .speaker-panel input, .speaker-panel select, button { min-height:44px; font-size:16px; }
      .sensitivity { grid-template-columns:1fr; }
      .sensitivity strong { text-align:left; }
      .media-card { margin:10px; padding:10px; }
      video { max-height:28vh; }
      .transcript-panel { margin:0 10px 12px; min-height:45vh; display:block; }
      .sentences { overflow:visible; padding:10px; }
      .status { max-height:150px; }
      .speaker-list { max-height:none; }
      .speaker-tools, .speaker-add { display:grid; grid-template-columns:1fr 1fr; }
      .speaker-tools strong { grid-column:1 / -1; }
      .speaker-tools select, .speaker-tools input, .speaker-add input[type="text"], .speaker-add input[type="file"] { grid-column:1 / -1; max-width:none; }
      .speaker-item { grid-template-columns:1fr; }
      .speaker-source { grid-column:1; white-space:normal; }
      .top { display:grid; grid-template-columns:1fr; gap:8px; }
      .prob { width:100%; flex-basis:auto; }
      .text { font-size:16px; line-height:1.42; }
    }
    @media (max-width: 460px) {
      .topbar { align-items:stretch; }
      .transport { gap:6px; }
      .transport button { padding:0 12px; }
      .control-panel { padding:8px; gap:8px; }
      .control-card, .media-card, .transcript-panel { border-radius:6px; }
      .speaker-tools, .speaker-add { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <div class="title">WhoSpeaks Live</div>
      <div id="state" class="runtime-state">Ready</div>
    </div>
    <div class="transport">
      <button id="start">Start</button>
      <button id="stop" disabled>Stop</button>
    </div>
  </header>
  <main class="layout">
    <section class="workspace">
      <section class="media-card">
      <video id="video" src="/media/video" controls muted playsinline></video>
      <audio id="audio" src="/media/audio" controls></audio>
      <div id="youtubeStream" class="youtube-stream"><iframe id="youtubeFrame" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe><div id="streamHint" class="stream-hint"></div></div>
      </section>
      <section class="transcript-panel">
        <div class="section-title">Transcript</div>
        <section id="sentences" class="sentences"></section>
      </section>
    </section>
    <aside class="control-panel">
      <section class="control-card source-panel">
        <div class="section-title">Source</div>
        <div class="source-grid">
          <select id="inputMode" class="mode" aria-label="Input mode">
            <option value="youtube">YouTube URL</option>
            <option value="microphone">Microphone</option>
            <option value="system">Computer/tab audio</option>
          </select>
          <select id="preset" class="preset" aria-label="Preset video"></select>
          <div class="source-row">
            <input id="source" class="source" type="url" spellcheck="false" autocomplete="off">
            <button id="load">Load</button>
          </div>
        </div>
      </section>
      <details class="control-card speaker-panel" open>
        <summary>Speakers</summary>
        <label class="sensitivity" title="Controls how easily the diarizer creates a new speaker profile.">
          <span>New speaker</span>
          <strong id="newSpeakerSensitivityLabel"></strong>
          <input id="newSpeakerSensitivity" type="range" min="1" max="5" step="1">
        </label>
        <div class="speaker-tools">
          <strong>Groups</strong>
          <select id="speakerGroupSelect" aria-label="Speaker group"></select>
          <button id="loadSpeakerGroup">Load</button>
          <input id="speakerGroupName" type="text" placeholder="Group name" autocomplete="off">
          <button id="saveSpeakerGroup">Save</button>
        </div>
        <div id="speakerList" class="speaker-list"></div>
        <form id="referenceSpeakerForm" class="speaker-add">
          <input id="referenceSpeakerName" type="text" placeholder="Speaker name" autocomplete="off">
          <input id="referenceSpeakerFile" type="file" accept="audio/*">
          <button type="submit">Add reference</button>
          <button id="recordReference" type="button">Record</button>
          <button id="stopReference" type="button" disabled>Stop & add</button>
          <span id="referenceRecordSeconds" class="speaker-source">0.0s</span>
        </form>
      </details>
      <details class="control-card status-card" open>
        <summary>Status</summary>
        <div id="status" class="status"></div>
      </details>
    </aside>
  </main>
</div>
<script>
const start = document.getElementById("start");
const stop = document.getElementById("stop");
const load = document.getElementById("load");
const preset = document.getElementById("preset");
const state = document.getElementById("state");
const source = document.getElementById("source");
const video = document.getElementById("video");
const audio = document.getElementById("audio");
const youtubeFrame = document.getElementById("youtubeFrame");
const streamHint = document.getElementById("streamHint");
const statusBox = document.getElementById("status");
const statusCard = document.querySelector(".status-card");
const sentences = document.getElementById("sentences");
const inputMode = document.getElementById("inputMode");
const newSpeakerSensitivity = document.getElementById("newSpeakerSensitivity");
const newSpeakerSensitivityLabel = document.getElementById("newSpeakerSensitivityLabel");
const speakerGroupSelect = document.getElementById("speakerGroupSelect");
const loadSpeakerGroupButton = document.getElementById("loadSpeakerGroup");
const speakerGroupName = document.getElementById("speakerGroupName");
const saveSpeakerGroupButton = document.getElementById("saveSpeakerGroup");
const speakerList = document.getElementById("speakerList");
const referenceSpeakerForm = document.getElementById("referenceSpeakerForm");
const referenceSpeakerName = document.getElementById("referenceSpeakerName");
const referenceSpeakerFile = document.getElementById("referenceSpeakerFile");
const recordReferenceButton = document.getElementById("recordReference");
const stopReferenceButton = document.getElementById("stopReference");
const referenceRecordSeconds = document.getElementById("referenceRecordSeconds");
const speakerColors = __SPEAKER_COLORS__;
const initialSource = __SOURCE_JSON__;
const presetVideos = __PRESET_VIDEOS__;
const speakerSensitivityConfig = __NEW_SPEAKER_SENSITIVITY_JSON__;
const initialSpeakerLibrary = __SPEAKER_LIBRARY_JSON__;
const targetCaptureSampleRate = 16000;
const captureStartRmsThreshold = 0.003;
const capturePreRollSeconds = 0.7;
let es = null;
let playbackTimer = null;
let playbackClockStartedAt = null;
const playbackClockSlackSeconds = 3.0;
let currentRealtimeGeneration = 0;
let mediaVersion = 0;
let browserStreamMode = false;
let browserStreamPrepared = false;
let browserStreamPreparedUrl = "";
let captureSourceKind = "display";
let captureStream = null;
let captureAudioContext = null;
let captureSourceNode = null;
let captureProcessor = null;
let captureSilentGain = null;
let captureSendQueue = Promise.resolve();
let capturePending = [];
let capturePendingSamples = 0;
let captureAudioStarted = false;
let capturePreRoll = [];
let capturePreRollSamples = 0;
let speakerSensitivityDirty = false;
let speakerLibraryState = initialSpeakerLibrary || {group_name:"", groups:[], speakers:[]};
let speakerNames = {};
let referenceRecordStream = null;
let referenceRecordContext = null;
let referenceRecordSource = null;
let referenceRecordProcessor = null;
let referenceRecordSilentGain = null;
let referenceRecordChunks = [];
let referenceRecordSamples = 0;
let referenceRecordSampleRate = targetCaptureSampleRate;
let referenceRecordStartedAt = 0;
let referenceRecordTimer = null;
let referenceRecordPending = false;
if (statusCard && window.matchMedia("(max-width: 900px)").matches) {
  statusCard.open = false;
}
function setState(text) { state.textContent = text; }
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
  else if (lower.includes("synchronized playback can begin") || lower.includes("growing-window transcription started") || lower.includes("realtime preview started")) nextState = browserStreamMode ? "Capturing" : "Playing";
  else if (lower.includes("transcribing window")) nextState = "Transcribing";
  else if (lower.includes("queued speaker embedding") || lower.includes("embedded sentence")) nextState = "Diarizing";
  if (nextState) setState(nextState);

  if (!browserStreamMode) return;
  if (lower.includes("waiting for audible input")) {
    setStreamHint("Audio capture is armed. Play the video and make sure the shared source includes audio.");
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
}
function normalizeUrl(url) {
  return String(url || "").trim();
}
function syncPresetSelection(url) {
  const normalized = normalizeUrl(url);
  const match = presetVideos.find(item => normalizeUrl(item.url) === normalized);
  preset.value = match ? match.url : "";
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
populatePresetVideos();
source.value = initialSource;
syncPresetSelection(initialSource);
function selectedSpeakerSensitivityPreset() {
  const level = Number(newSpeakerSensitivity.value || speakerSensitivityConfig.selected || 3);
  return speakerSensitivityConfig.presets.find(item => Number(item.level) === level) || speakerSensitivityConfig.presets[2];
}
function updateSpeakerSensitivityLabel() {
  const preset = selectedSpeakerSensitivityPreset();
  newSpeakerSensitivityLabel.textContent = `${preset.level}. ${preset.label}`;
}
async function applySpeakerSensitivity() {
  const preset = selectedSpeakerSensitivityPreset();
  const result = await post("/api/settings", {new_speaker_sensitivity: preset.level});
  const applied = result.new_speaker_sensitivity || preset;
  if (applied.level && Number(newSpeakerSensitivity.value) !== Number(applied.level)) {
    newSpeakerSensitivity.value = applied.level;
  }
  speakerSensitivityDirty = false;
  updateSpeakerSensitivityLabel();
  return result;
}
async function applySpeakerSensitivityIfDirty() {
  if (!speakerSensitivityDirty) return null;
  return applySpeakerSensitivity();
}
newSpeakerSensitivity.value = speakerSensitivityConfig.selected || 3;
updateSpeakerSensitivityLabel();
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
  browserStreamMode = Boolean(enabled);
  captureSourceKind = sourceKind || "display";
  browserStreamPrepared = false;
  browserStreamPreparedUrl = "";
  document.querySelector(".app").classList.toggle("browser-stream", browserStreamMode);
  if (browserStreamMode) {
    const embed = youtubeEmbedUrl(url);
    youtubeFrame.src = embed;
    youtubeFrame.parentElement.classList.toggle("empty", !embed);
    if (streamHint) {
      if (captureSourceKind === "microphone") {
        setStreamHint("Microphone mode. Press Start, allow microphone access, then speak.");
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
}
function browserStreamSourceUrl() {
  if (captureSourceKind === "microphone") return "microphone://local";
  const url = source.value.trim();
  return url || "system-audio://local";
}
async function prepareBrowserStreamSession() {
  const url = browserStreamSourceUrl();
  if (browserStreamPrepared && browserStreamPreparedUrl === url) return;
  const result = await post("/api/browser-stream", {url});
  browserStreamPrepared = true;
  browserStreamPreparedUrl = url;
  mediaVersion = Number(result.version || mediaVersion || Date.now());
  log(`Browser audio stream prepared for ${result.video_id}.`);
}
function initializeInputModeFromSource() {
  const value = source.value.trim();
  if (value.startsWith("microphone://")) {
    inputMode.value = "microphone";
    setBrowserStreamMode(true, "", "microphone");
  } else if (value.startsWith("system-audio://")) {
    inputMode.value = "system";
    setBrowserStreamMode(true, "", "display");
  }
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
  capturePreRoll.push(copy);
  capturePreRollSamples += copy.length;
  const maxSamples = Math.max(0, Math.floor(sampleRate * capturePreRollSeconds));
  while (capturePreRollSamples > maxSamples && capturePreRoll.length) {
    capturePreRollSamples -= capturePreRoll.shift().length;
  }
}
function flushCapturePreRoll(sampleRate) {
  if (!capturePreRollSamples) return;
  const combined = new Float32Array(capturePreRollSamples);
  let offset = 0;
  for (const chunk of capturePreRoll) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }
  capturePreRoll = [];
  capturePreRollSamples = 0;
  queueBrowserAudioChunk(combined, sampleRate);
}
function queueBrowserAudioChunk(samples, sampleRate) {
  const resampled = resampleFloat32(samples, sampleRate, targetCaptureSampleRate);
  const payload = {
    sample_rate: targetCaptureSampleRate,
    audio_b64: float32ToBase64(resampled),
  };
  captureSendQueue = captureSendQueue
    .then(() => post("/api/audio-chunk", payload))
    .catch(error => log(`Audio chunk failed: ${error.message}`));
}
function flushBrowserAudio(force=false) {
  const sampleRate = captureAudioContext ? captureAudioContext.sampleRate : 16000;
  const targetSamples = Math.max(1600, Math.floor(sampleRate * 0.5));
  if (!force && capturePendingSamples < targetSamples) return;
  if (capturePendingSamples <= 0) return;
  const combined = new Float32Array(capturePendingSamples);
  let offset = 0;
  for (const chunk of capturePending) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }
  capturePending = [];
  capturePendingSamples = 0;
  queueBrowserAudioChunk(combined, sampleRate);
}
function stopBrowserAudioCapture() {
  flushBrowserAudio(true);
  if (captureProcessor) {
    try { captureProcessor.disconnect(); } catch (_) {}
    captureProcessor.onaudioprocess = null;
    captureProcessor = null;
  }
  if (captureSourceNode) {
    try { captureSourceNode.disconnect(); } catch (_) {}
    captureSourceNode = null;
  }
  if (captureSilentGain) {
    try { captureSilentGain.disconnect(); } catch (_) {}
    captureSilentGain = null;
  }
  if (captureStream) {
    captureStream.getTracks().forEach(track => track.stop());
    captureStream = null;
  }
  if (captureAudioContext) {
    captureAudioContext.close().catch(() => {});
    captureAudioContext = null;
  }
  capturePending = [];
  capturePendingSamples = 0;
  captureAudioStarted = false;
  capturePreRoll = [];
  capturePreRollSamples = 0;
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
  if (captureSourceKind === "microphone") {
    if (!navigator.mediaDevices.getUserMedia) {
      throw new Error("Microphone capture is not available in this browser.");
    }
    log("Allow microphone access. Capture will start when speech is audible.");
    captureStream = await navigator.mediaDevices.getUserMedia({video: false, audio: audioOptions});
  } else {
    if (!navigator.mediaDevices.getDisplayMedia) {
      throw new Error("Browser tab-audio capture is not available in this browser.");
    }
    log("Choose the YouTube/app tab and enable tab/system audio in the share dialog.");
    captureStream = await navigator.mediaDevices.getDisplayMedia({video: true, audio: audioOptions});
  }
  if (!captureStream.getAudioTracks().length) {
    stopBrowserAudioCapture();
    throw new Error(captureSourceKind === "microphone" ? "No microphone audio track was shared." : "No tab audio track was shared.");
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  captureAudioContext = new AudioContextClass({sampleRate: targetCaptureSampleRate});
  const audioOnlyStream = new MediaStream(captureStream.getAudioTracks());
  captureSourceNode = captureAudioContext.createMediaStreamSource(audioOnlyStream);
  captureProcessor = captureAudioContext.createScriptProcessor(4096, 1, 1);
  captureSilentGain = captureAudioContext.createGain();
  captureSilentGain.gain.value = 0;
  captureProcessor.onaudioprocess = event => {
    const input = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    if (!captureAudioStarted) {
      rememberCapturePreRoll(copy, captureAudioContext.sampleRate);
      if (rms(copy) < captureStartRmsThreshold) {
        return;
      }
      captureAudioStarted = true;
      log("Detected audible input; streaming audio to backend.");
      flushCapturePreRoll(captureAudioContext.sampleRate);
      return;
    }
    capturePending.push(copy);
    capturePendingSamples += copy.length;
    flushBrowserAudio(false);
  };
  captureSourceNode.connect(captureProcessor);
  captureProcessor.connect(captureSilentGain);
  captureSilentGain.connect(captureAudioContext.destination);
  await captureAudioContext.resume();
  captureStream.getTracks().forEach(track => {
    track.onended = () => {
      if (browserStreamMode) log("Browser audio capture ended.");
      stopBrowserAudioCapture();
    };
  });
  log(`Browser audio capture armed at ${Math.round(captureAudioContext.sampleRate)} Hz; waiting for audible input.`);
}
initializeInputModeFromSource();
function log(text) {
  const div = document.createElement("div");
  div.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  statusBox.appendChild(div);
  statusBox.scrollTop = statusBox.scrollHeight;
}
async function post(path, payload={}) {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function mediaSeconds(element) {
  const seconds = Number(element.currentTime || 0);
  return Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
}
function playbackClockMaxSeconds() {
  if (playbackClockStartedAt === null) return Number.POSITIVE_INFINITY;
  return Math.max(0, ((performance.now() - playbackClockStartedAt) / 1000) + playbackClockSlackSeconds);
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
function startPlaybackClock() {
  if (playbackTimer) clearInterval(playbackTimer);
  playbackClockStartedAt = performance.now();
  const send = () => post("/api/playback", {seconds: playbackSeconds()}).catch(() => {});
  send();
  playbackTimer = setInterval(send, 250);
}
function flushPlaybackEnd() {
  const duration = Number(audio.duration || 0);
  if (!playbackTimer) return;
  if (duration > 0 && playbackClockStartedAt !== null) {
    const elapsed = (performance.now() - playbackClockStartedAt) / 1000;
    if (elapsed + playbackClockSlackSeconds < duration) {
      log(`Ignoring early audio ended event at ${playbackSeconds().toFixed(1)}s of ${duration.toFixed(1)}s.`);
      return;
    }
  }
  post("/api/playback", {seconds: duration > 0 ? duration : playbackSeconds()}).catch(() => {});
}
function stopPlaybackClock() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  playbackClockStartedAt = null;
}
function resetTranscriptDisplay() {
  sentences.textContent = "";
  statusBox.textContent = "";
  currentRealtimeGeneration = 0;
}
function refreshMediaElements(version) {
  setBrowserStreamMode(false);
  mediaVersion = Number(version || Date.now());
  video.pause();
  audio.pause();
  video.src = `/media/video?v=${mediaVersion}`;
  audio.src = `/media/audio?v=${mediaVersion}`;
  video.load();
  audio.load();
}
async function unlockPlayback() {
  video.currentTime = 0; audio.currentTime = 0; video.muted = true; audio.volume = 1.0;
  const results = await Promise.allSettled([video.play(), audio.play()]);
  video.pause(); audio.pause(); video.currentTime = 0; audio.currentTime = 0;
  return results;
}
function logRejectedPlayback(results) {
  results.forEach((result, index) => {
    if (result.status === "rejected") {
      log(`${index === 0 ? "video" : "audio"} playback blocked: ${result.reason?.name || result.reason}`);
    }
  });
}
function scrollSentencesToBottom() {
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
function speakerDisplayLabel(label) {
  if (label && speakerNames[label]) return speakerNames[label];
  const index = speakerIndex(label);
  return index === null ? "Unknown" : `Speaker ${index}`;
}
function speakerColor(label) {
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
function updateSpeakerState(state) {
  if (!state || typeof state !== "object") return;
  speakerLibraryState = {
    group_name: state.group_name || "",
    groups: Array.isArray(state.groups) ? state.groups : [],
    speakers: Array.isArray(state.speakers) ? state.speakers : [],
    embedding_provider: state.embedding_provider || "",
  };
  speakerNames = {};
  speakerLibraryState.speakers.forEach(speaker => {
    if (speaker.id) {
      speakerNames[speaker.id] = speaker.display_name || speaker.name || speakerDisplayLabel(speaker.id);
    }
  });
  renderSpeakerPanel();
  refreshSpeakerRows();
}
function renderSpeakerPanel() {
  const selected = speakerGroupSelect.value || speakerLibraryState.group_name || "";
  speakerGroupSelect.textContent = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "No saved group";
  speakerGroupSelect.appendChild(placeholder);
  speakerLibraryState.groups.forEach(group => {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group;
    speakerGroupSelect.appendChild(option);
  });
  const fallback = speakerLibraryState.groups.includes(speakerLibraryState.group_name) ? speakerLibraryState.group_name : "";
  speakerGroupSelect.value = speakerLibraryState.groups.includes(selected) ? selected : fallback;
  if (document.activeElement !== speakerGroupName && speakerLibraryState.group_name && !speakerGroupName.value.trim()) {
    speakerGroupName.value = speakerLibraryState.group_name;
  }

  speakerList.textContent = "";
  if (!speakerLibraryState.speakers.length) {
    const empty = document.createElement("div");
    empty.className = "speaker-source";
    empty.textContent = "No speakers yet";
    speakerList.appendChild(empty);
    return;
  }
  speakerLibraryState.speakers.forEach(speaker => {
    const row = document.createElement("div");
    row.className = "speaker-item";
    row.dataset.speakerId = speaker.id || "";
    const badge = document.createElement("span");
    badge.className = "badge";
    const color = speakerColor(speaker.id);
    if (color) {
      badge.style.color = color;
      badge.style.borderColor = color;
    }
    badge.textContent = speakerDisplayLabel(speaker.id);
    const input = document.createElement("input");
    input.type = "text";
    input.value = speaker.name || speaker.display_name || speakerDisplayLabel(speaker.id);
    input.autocomplete = "off";
    input.addEventListener("change", async () => {
      try {
        const result = await post("/api/speakers/rename", {speaker_id: speaker.id, name: input.value});
        updateSpeakerState(result.speaker_state);
      } catch (error) {
        log(`Rename failed: ${error.message}`);
      }
    });
    const source = document.createElement("span");
    source.className = "speaker-source";
    const count = Number(speaker.sentence_count || 0);
    const sourceLabel = speaker.source === "reference" ? "reference" : "detected";
    source.textContent = `${sourceLabel}${speaker.locked ? ", locked" : ""}${count ? `, ${count} sent.` : ""}`;
    row.appendChild(badge);
    row.appendChild(input);
    row.appendChild(source);
    speakerList.appendChild(row);
  });
}
function refreshSpeakerRows() {
  Array.from(sentences.querySelectorAll(".row")).forEach(row => {
    const label = row.dataset.speaker === "UNKNOWN" ? null : row.dataset.speaker;
    const badge = row.querySelector(".speaker-name");
    if (badge) badge.textContent = speakerDisplayLabel(label);
  });
}
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read reference audio."));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });
}
function updateReferenceRecordingControls(recording) {
  recordReferenceButton.disabled = recording;
  stopReferenceButton.disabled = !recording;
  referenceSpeakerFile.disabled = recording;
  recordReferenceButton.classList.toggle("recording", recording);
}
function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const step = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += step) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + step));
  }
  return btoa(binary);
}
function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}
function flattenFloat32Chunks(chunks, totalSamples) {
  const output = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}
function encodeWavDataUrl(samples, sampleRate) {
  const wavRate = targetCaptureSampleRate;
  const resampled = resampleFloat32(samples, sampleRate, wavRate);
  const buffer = new ArrayBuffer(44 + resampled.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + resampled.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, wavRate, true);
  view.setUint32(28, wavRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, resampled.length * 2, true);
  let offset = 44;
  for (let i = 0; i < resampled.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, resampled[i] || 0));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return `data:audio/wav;base64,${arrayBufferToBase64(buffer)}`;
}
function stopReferenceRecording() {
  referenceRecordPending = false;
  const chunks = referenceRecordChunks;
  const totalSamples = referenceRecordSamples;
  const sampleRate = referenceRecordSampleRate || targetCaptureSampleRate;
  if (referenceRecordTimer) {
    clearInterval(referenceRecordTimer);
    referenceRecordTimer = null;
  }
  if (referenceRecordProcessor) {
    try { referenceRecordProcessor.disconnect(); } catch (_) {}
    referenceRecordProcessor.onaudioprocess = null;
    referenceRecordProcessor = null;
  }
  if (referenceRecordSource) {
    try { referenceRecordSource.disconnect(); } catch (_) {}
    referenceRecordSource = null;
  }
  if (referenceRecordSilentGain) {
    try { referenceRecordSilentGain.disconnect(); } catch (_) {}
    referenceRecordSilentGain = null;
  }
  if (referenceRecordStream) {
    referenceRecordStream.getTracks().forEach(track => track.stop());
    referenceRecordStream = null;
  }
  if (referenceRecordContext) {
    referenceRecordContext.close().catch(() => {});
    referenceRecordContext = null;
  }
  referenceRecordChunks = [];
  referenceRecordSamples = 0;
  referenceRecordStartedAt = 0;
  referenceRecordSeconds.textContent = totalSamples > 0 ? `${(totalSamples / sampleRate).toFixed(1)}s` : "0.0s";
  updateReferenceRecordingControls(false);
  return {
    samples: flattenFloat32Chunks(chunks, totalSamples),
    sampleRate,
    seconds: totalSamples / sampleRate,
  };
}
async function startReferenceRecording() {
  const name = referenceSpeakerName.value.trim();
  if (!name) {
    log("Enter a speaker name first.");
    return;
  }
  if (referenceRecordStream || referenceRecordPending) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    log("Microphone recording is not available in this browser.");
    return;
  }
  referenceRecordPending = true;
  referenceRecordChunks = [];
  referenceRecordSamples = 0;
  referenceRecordSampleRate = targetCaptureSampleRate;
  referenceRecordSeconds.textContent = "0.0s";
  recordReferenceButton.disabled = true;
  stopReferenceButton.disabled = true;
  referenceSpeakerFile.disabled = true;
  try {
    referenceRecordStream = await navigator.mediaDevices.getUserMedia({
      video: false,
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    referenceRecordContext = new AudioContextClass({sampleRate: targetCaptureSampleRate});
    referenceRecordSampleRate = referenceRecordContext.sampleRate || targetCaptureSampleRate;
    referenceRecordSource = referenceRecordContext.createMediaStreamSource(referenceRecordStream);
    referenceRecordProcessor = referenceRecordContext.createScriptProcessor(4096, 1, 1);
    referenceRecordSilentGain = referenceRecordContext.createGain();
    referenceRecordSilentGain.gain.value = 0;
    referenceRecordProcessor.onaudioprocess = event => {
      const input = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      referenceRecordChunks.push(copy);
      referenceRecordSamples += copy.length;
      referenceRecordSeconds.textContent = `${(referenceRecordSamples / referenceRecordSampleRate).toFixed(1)}s`;
    };
    referenceRecordSource.connect(referenceRecordProcessor);
    referenceRecordProcessor.connect(referenceRecordSilentGain);
    referenceRecordSilentGain.connect(referenceRecordContext.destination);
    await referenceRecordContext.resume();
    referenceRecordStartedAt = performance.now();
    referenceRecordTimer = setInterval(() => {
      const seconds = referenceRecordSamples > 0
        ? referenceRecordSamples / referenceRecordSampleRate
        : (performance.now() - referenceRecordStartedAt) / 1000;
      referenceRecordSeconds.textContent = `${seconds.toFixed(1)}s`;
    }, 100);
    referenceRecordPending = false;
    updateReferenceRecordingControls(true);
    log(`Recording reference clip for ${name}.`);
  } catch (error) {
    referenceRecordPending = false;
    stopReferenceRecording();
    log(`Reference recording failed: ${error.message}`);
  }
}
async function stopAndAddReferenceRecording() {
  const name = referenceSpeakerName.value.trim();
  const recording = stopReferenceRecording();
  if (!name) {
    log("Enter a speaker name first.");
    return;
  }
  if (recording.seconds < 0.5) {
    log("Reference clip is too short.");
    return;
  }
  stopReferenceButton.disabled = true;
  recordReferenceButton.disabled = true;
  try {
    const audio_b64 = encodeWavDataUrl(recording.samples, recording.sampleRate);
    const result = await post("/api/speakers/reference", {name, filename: `${name}.wav`, audio_b64});
    updateSpeakerState(result.speaker_state);
    referenceSpeakerName.value = "";
    referenceRecordSeconds.textContent = "0.0s";
    log(`Added recorded reference speaker ${name}.`);
  } catch (error) {
    log(`Add recorded reference failed: ${error.message}`);
  } finally {
    updateReferenceRecordingControls(false);
  }
}
updateSpeakerState(speakerLibraryState);
function probabilitySegments(probabilities) {
  const entries = Object.entries(probabilities || {})
    .map(([key, value]) => ({key, value: Math.max(0, Number(value) || 0)}))
    .filter(item => item.value > 0);
  entries.sort((left, right) => {
    if (left.key === "unknown") return -1;
    if (right.key === "unknown") return 1;
    const leftIndex = /^speaker(\d+)$/.exec(left.key);
    const rightIndex = /^speaker(\d+)$/.exec(right.key);
    if (leftIndex && rightIndex) return Number(leftIndex[1]) - Number(rightIndex[1]);
    if (leftIndex) return -1;
    if (rightIndex) return 1;
    return left.key.localeCompare(right.key);
  });
  const total = entries.reduce((sum, item) => sum + item.value, 0);
  return entries.map(item => ({
    ...item,
    width: total > 0 ? (item.value / total) * 100 : 0,
  }));
}
function probabilityTooltip(probabilities) {
  return probabilitySegments(probabilities)
    .map(item => `${probabilityDisplayLabel(item.key)} ${Math.round(item.value * 100)}%`)
    .join(" | ");
}
function displayProbabilities(item) {
  const assignedKey = speakerProbabilityKey(item.assigned_speaker);
  if (assignedKey && item.created_speaker && !item.pending && !item.error) {
    return {[assignedKey]: 1.0};
  }
  return item.probabilities;
}
function secondsLabel(value) {
  return `${Number(value || 0).toFixed(1)}s`;
}
function ratioLabel(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "n/a";
}
function findSentenceRow(index) {
  const key = String(index);
  return Array.from(sentences.querySelectorAll(".row")).find(row => row.dataset.index === key) || null;
}
function clearRealtimeRows(generation) {
  currentRealtimeGeneration = Math.max(currentRealtimeGeneration, Number(generation || 0));
  Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => row.remove());
}
function renderSentence(item) {
  if (item.realtime && Number(item.realtime_generation || 0) < currentRealtimeGeneration) {
    return;
  }
  if (item.assigned_speaker && item.speaker_name) {
    speakerNames[item.assigned_speaker] = item.speaker_name;
  }
  if (item.realtime) {
    currentRealtimeGeneration = Math.max(currentRealtimeGeneration, Number(item.realtime_generation || 0));
  }
  let row = findSentenceRow(item.index);
  const isNewRow = !row;
  if (!row) {
    row = document.createElement("div");
    sentences.appendChild(row);
  }
  row.className = item.realtime ? "row realtime" : "row";
  row.dataset.index = item.index;
  row.dataset.realtime = item.realtime ? "true" : "false";
  row.dataset.speaker = item.assigned_speaker || "UNKNOWN";
  const speakerLabel = speakerDisplayLabel(item.assigned_speaker);
  const color = speakerColor(item.assigned_speaker);
  const speakerClass = item.assigned_speaker ? "badge" : "badge unknown";
  const stateLabel = item.realtime ? "Live" : (item.pending ? "Embedding" : (item.error ? "Error" : (item.revision ? "Revised" : "")));
  const startSeconds = Number(item.start || 0);
  const endSeconds = Number(item.end || 0);
  const durationSeconds = Math.max(0, endSeconds - startSeconds);
  const ratio = Number(item.speech_audio_ratio);

  const top = document.createElement("div");
  top.className = "top";
  const topLeft = document.createElement("div");
  topLeft.className = "top-left";

  const speakerBadge = document.createElement("span");
  speakerBadge.className = `${speakerClass} speaker-name`;
  if (color) {
    speakerBadge.style.color = color;
    speakerBadge.style.borderColor = color;
    speakerBadge.style.background = "rgba(255,255,255,.04)";
  }
  speakerBadge.textContent = speakerLabel;
  topLeft.appendChild(speakerBadge);

  if (item.created_speaker) {
    const badge = document.createElement("span");
    badge.className = "badge new";
    badge.textContent = "New";
    topLeft.appendChild(badge);
  }
  if (stateLabel) {
    const badge = document.createElement("span");
    badge.className = "badge state";
    badge.textContent = stateLabel;
    topLeft.appendChild(badge);
  }

  const duration = document.createElement("span");
  duration.textContent = secondsLabel(durationSeconds);
  topLeft.appendChild(duration);

  const range = document.createElement("span");
  range.textContent = `(${secondsLabel(startSeconds)} - ${secondsLabel(endSeconds)})`;
  topLeft.appendChild(range);

  if (Number.isFinite(ratio)) {
    const ratioSpan = document.createElement("span");
    ratioSpan.textContent = `speech/audio ${ratioLabel(item.speech_audio_ratio)}`;
    topLeft.appendChild(ratioSpan);
  }

  const prob = document.createElement("div");
  prob.className = "prob";
  top.appendChild(topLeft);
  top.appendChild(prob);

  const text = document.createElement("div");
  text.className = "text";
  text.textContent = item.text || "";
  row.replaceChildren(top, text);

  const probabilities = displayProbabilities(item);
  const segments = probabilitySegments(probabilities);
  const rawTooltip = probabilityTooltip(item.probabilities);
  const displayTooltip = probabilityTooltip(probabilities);
  prob.title = rawTooltip && rawTooltip !== displayTooltip ? `${displayTooltip} (raw: ${rawTooltip})` : displayTooltip;
  segments.forEach(segment => {
    const span = document.createElement("span");
    span.style.flex = `0 0 ${segment.width}%`;
    span.style.background = probabilityColor(segment.key);
    prob.appendChild(span);
  });
  if (item.error) {
    prob.title = item.error;
  }
  if (isNewRow || item.realtime) {
    scrollSentencesToBottom();
  }
}
function connect() {
  if (es) es.close();
  es = new EventSource("/events");
  es.addEventListener("status", e => {
    const data = JSON.parse(e.data);
    log(data.message);
    reflectRuntimeStatus(data.message);
  });
  es.addEventListener("speakers", e => updateSpeakerState(JSON.parse(e.data)));
  es.addEventListener("sentence", e => renderSentence(JSON.parse(e.data)));
  es.addEventListener("realtime", e => renderSentence(JSON.parse(e.data)));
  es.addEventListener("realtime_clear", e => clearRealtimeRows(JSON.parse(e.data).generation));
  es.addEventListener("done", e => { stopPlaybackClock(); stopBrowserAudioCapture(); setState("Stopped"); start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); log(JSON.parse(e.data).message); });
}
start.addEventListener("click", async () => {
  start.disabled = true; stop.disabled = false; setSourceControlsDisabled(true); resetTranscriptDisplay(); setState("Starting"); connect();
  if (browserStreamMode) {
    try {
      await applySpeakerSensitivityIfDirty();
      setState(captureSourceKind === "microphone" ? "Requesting mic" : "Requesting audio");
      await prepareBrowserStreamSession();
      await startBrowserAudioCapture();
      setState("Warming backend");
      setStreamHint(captureSourceKind === "microphone" ? "Microphone capture is armed; warming backend." : "Audio capture is armed; warming backend before transcription starts.");
      await post("/api/start");
    } catch (error) {
      stopBrowserAudioCapture();
      start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
      return;
    }
    setState("Capturing");
    return;
  }
  logRejectedPlayback(await unlockPlayback());
  try {
    await applySpeakerSensitivityIfDirty();
    setState("Warming backend");
    log("Warming backend before playback starts. First Modal starts can take about two minutes.");
    await post("/api/start");
  } catch (error) {
    start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
    return;
  }
  setState("Starting playback");
  video.currentTime = 0; audio.currentTime = 0; video.muted = true; audio.volume = 1.0;
  logRejectedPlayback(await Promise.allSettled([video.play(), audio.play()]));
  startPlaybackClock();
  setState("Playing");
});
stop.addEventListener("click", async () => {
  stop.disabled = true; start.disabled = false; setSourceControlsDisabled(false); setState("Stopping"); stopPlaybackClock(); stopBrowserAudioCapture(); video.pause(); audio.pause(); await post("/api/stop");
});
preset.addEventListener("change", () => {
  if (preset.value) {
    source.value = preset.value;
    if (inputMode.value === "system") {
      setBrowserStreamMode(true, source.value.trim(), "display");
    }
  }
});
source.addEventListener("input", () => {
  syncPresetSelection(source.value);
  if (inputMode.value === "system") {
    setBrowserStreamMode(true, source.value.trim(), "display");
  }
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
  } else {
    setBrowserStreamMode(false);
    setState("Ready");
  }
});
newSpeakerSensitivity.addEventListener("input", () => {
  speakerSensitivityDirty = true;
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
saveSpeakerGroupButton.addEventListener("click", async () => {
  const name = speakerGroupName.value.trim();
  if (!name) {
    log("Enter a speaker group name first.");
    return;
  }
  saveSpeakerGroupButton.disabled = true;
  try {
    const result = await post("/api/speakers/save", {name});
    updateSpeakerState(result.speaker_state);
    log(`Saved speaker group ${result.speaker_state.group_name}.`);
  } catch (error) {
    log(`Save speakers failed: ${error.message}`);
  } finally {
    saveSpeakerGroupButton.disabled = false;
  }
});
loadSpeakerGroupButton.addEventListener("click", async () => {
  const name = speakerGroupSelect.value || speakerGroupName.value.trim();
  if (!name) {
    log("Choose a saved speaker group first.");
    return;
  }
  loadSpeakerGroupButton.disabled = true;
  try {
    const result = await post("/api/speakers/load", {name});
    updateSpeakerState(result.speaker_state);
    speakerGroupName.value = result.speaker_state.group_name || name;
    log(`Loaded speaker group ${result.speaker_state.group_name}.`);
  } catch (error) {
    log(`Load speakers failed: ${error.message}`);
  } finally {
    loadSpeakerGroupButton.disabled = false;
  }
});
referenceSpeakerForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (referenceRecordStream || referenceRecordPending) {
    log("Stop the reference recording first.");
    return;
  }
  const name = referenceSpeakerName.value.trim();
  const file = referenceSpeakerFile.files && referenceSpeakerFile.files[0];
  if (!name || !file) {
    log("Choose a speaker name and reference audio file first.");
    return;
  }
  const submit = referenceSpeakerForm.querySelector("button[type='submit']");
  submit.disabled = true;
  try {
    const audio_b64 = await fileToBase64(file);
    const result = await post("/api/speakers/reference", {name, filename: file.name, audio_b64});
    updateSpeakerState(result.speaker_state);
    referenceSpeakerName.value = "";
    referenceSpeakerFile.value = "";
    log(`Added reference speaker ${name}.`);
  } catch (error) {
    log(`Add reference failed: ${error.message}`);
  } finally {
    submit.disabled = false;
  }
});
recordReferenceButton.addEventListener("click", () => {
  startReferenceRecording().catch(error => log(`Reference recording failed: ${error.message}`));
});
stopReferenceButton.addEventListener("click", () => {
  stopAndAddReferenceRecording().catch(error => log(`Add recorded reference failed: ${error.message}`));
});
load.addEventListener("click", async () => {
  const url = source.value.trim();
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
    log(url ? "Computer/tab audio mode ready with embedded video. Press Start and share this tab with audio." : "Computer/tab audio mode ready. Press Start and share audio from a tab or window.");
    setState("Ready");
    start.disabled = false;
    stop.disabled = true;
    return;
  }
  if (!url) {
    log("Enter a YouTube URL first.");
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
    source.value = media.url;
    syncPresetSelection(media.url);
    refreshMediaElements(media.version);
    log(`Loaded ${media.video_id}.`);
    setState("Ready");
    start.disabled = false;
  } catch (error) {
    log(`Load failed: ${error.message}`);
    try {
      const fallback = await post("/api/browser-stream", {url});
      source.value = fallback.url;
      syncPresetSelection(fallback.url);
      setBrowserStreamMode(true, fallback.url, "display");
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
audio.addEventListener("ended", flushPlaybackEnd);
</script>
</body>
</html>
"""
