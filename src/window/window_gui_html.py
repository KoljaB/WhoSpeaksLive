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
      --bg:#0B1015;
      --panel:#0F161F;
      --panel-2:#0F161F;
      --field:#0B1015;
      --line:#1B2B38;
      --text:#F1F5F8;
      --muted:#9EAAB6;
      --accent:#3DC77C;
      --accent-strong:#3DC77C;
    }
    * { box-sizing: border-box; }
    body { margin:0; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); font:14px/1.35 Arial, Helvetica, sans-serif; }
    strong, b, h1, h2, h3, h4, h5, h6, summary { font-weight:400; }
    .app { height:100vh; display:grid; grid-template-rows:auto auto minmax(0,1fr); }
    .topbar { min-height:52px; display:flex; gap:12px; align-items:center; justify-content:flex-start; padding:8px 12px 8px 16px; background:#0B1015; box-shadow:inset 0 1px 0 rgba(255,255,255,.04); }
    .live-summary { min-width:0; margin-left:auto; display:flex; align-items:center; justify-content:flex-end; gap:12px; }
    .brand { min-width:0; display:flex; align-items:center; gap:8px; }
    .brand-icon { flex:0 0 auto; width:22px; height:22px; display:grid; place-items:center; color:#17B7FE; }
    .brand-icon svg, .speaker-summary svg { width:20px; height:20px; display:block; }
    .title { color:var(--text); font-size:17px; font-weight:400; letter-spacing:0; white-space:nowrap; }
    .topbar-divider { flex:0 0 auto; width:1px; height:24px; background:var(--line); }
    .status-pill { flex:0 0 auto; min-height:23px; display:flex; align-items:center; gap:6px; padding:0 9px; border-radius:999px; background:rgba(61,199,124,.08); color:#3DC77C; font-size:13px; font-weight:400; white-space:nowrap; }
    .status-dot { width:9px; height:9px; border-radius:50%; background:#3DC77C; box-shadow:0 0 14px rgba(61,199,124,.45); }
    .runtime-state { max-width:30vw; color:#3DC77C; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .speaker-summary { flex:0 0 auto; min-height:23px; display:flex; align-items:center; gap:4px; color:var(--text); font-size:13px; white-space:nowrap; }
    .speaker-summary svg { color:#BA79EF; }
    #speakerCount { display:inline-flex; align-items:baseline; gap:7px; color:var(--text); }
    .speaker-count-number { position:relative; top:2px; font-size:16px; font-weight:600; line-height:1; color:#FF9F1C; }
    .speaker-count-label { font-size:13px; font-weight:400; }
    .transport { flex:0 0 auto; display:flex; gap:8px; align-items:center; }
    button { min-height:32px; border:1px solid var(--line); border-radius:7px; padding:0 12px; font-size:13px; font-weight:400; cursor:pointer; background:#0F161F; color:var(--text); }
    #start { border-color:#3DC77C; background:#0F161F; color:#3DC77C; }
    #start:disabled, #stop:disabled { display:none; }
    #stop { display:inline-flex; align-items:center; gap:8px; border-color:#DF3C36; background:#981D20; color:var(--text); box-shadow:inset 0 1px 0 rgba(255,255,255,.18); }
    .stop-icon { width:10px; height:10px; border-radius:2px; background:#FFFFFF; display:inline-block; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .session-banner { min-height:34px; display:flex; align-items:center; justify-content:space-between; gap:12px; padding:6px 12px 6px 16px; border-top:1px solid rgba(255,255,255,.035); border-bottom:1px solid var(--line); background:#0F161F; color:#B7C1CD; font-size:13px; }
    .session-banner.available { color:#3DC77C; background:#0B1015; }
    .session-banner.owner { color:#D7DEE8; background:color-mix(in srgb, #3DC77C 10%, #0B1015); border-bottom-color:color-mix(in srgb, #3DC77C 35%, var(--line)); }
    .session-banner.observer { color:#E5B96F; background:color-mix(in srgb, #FF9F1C 10%, #0B1015); border-bottom-color:color-mix(in srgb, #FF9F1C 35%, var(--line)); }
    .session-banner-message { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .session-banner-actions { flex:0 0 auto; display:flex; align-items:center; gap:8px; }
    .session-banner button { min-height:26px; padding:0 9px; font-size:12px; border-color:#20303E; background:#0B1015; color:#D7DEE8; }
    .session-banner button[hidden] { display:none; }
    .layout { min-height:0; display:grid; grid-template-columns:minmax(0,1fr) minmax(330px,380px); }
    .workspace { min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr); }
    .control-panel { min-height:0; overflow:auto; padding:8px; background:#0B1015; display:grid; align-content:start; gap:8px; }
    .control-card, .media-card, .transcript-panel { background:var(--panel-2); border:1px solid var(--line); border-radius:8px; }
    .control-card { padding:8px; }
    .section-title, summary { min-height:24px; display:flex; align-items:center; gap:6px; font-size:13px; font-weight:400; color:var(--text); }
    summary { cursor:pointer; list-style-position:outside; }
    .source-grid { width:100%; display:grid; grid-template-columns:minmax(150px,220px) minmax(0,1fr) auto; align-items:center; gap:6px; padding:0; border:0; border-radius:0; background:transparent; }
    .source-row { display:contents; }
    .dropdown-control { position:relative; min-height:34px; display:flex; align-items:center; border:1px solid var(--line); border-radius:7px; background:#0F161F; color:var(--text); font-size:13px; font-weight:400; }
    .dropdown-control::after { content:""; position:absolute; right:15px; top:50%; width:8px; height:8px; border-right:1.5px solid currentColor; border-bottom:1.5px solid currentColor; transform:translateY(-65%) rotate(45deg); opacity:.9; pointer-events:none; }
    .select-control { width:100%; min-width:0; }
    .select-control select { width:100%; min-width:0; min-height:32px; border:0; border-radius:7px; padding:0 36px 0 12px; background:#0F161F; color:var(--text); color-scheme:dark; font:inherit; appearance:none; outline:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .select-control select option, .mode option, .speaker-panel select option { background:#0B1015; color:var(--text); }
    .select-control select option:checked, .mode option:checked, .speaker-panel select option:checked { background:#0F161F; color:#FFFFFF; }
    .select-control select option:disabled, .mode option:disabled, .speaker-panel select option:disabled { background:#0B1015; color:#536271; }
    .mode, .speaker-panel select { width:100%; min-height:30px; border:1px solid var(--line); border-radius:6px; padding:0 30px 0 8px; font-size:13px; background-color:#0F161F; color:var(--text); color-scheme:dark; appearance:none; }
    .source, .speaker-panel input { width:100%; min-height:30px; border:1px solid var(--line); border-radius:6px; padding:0 8px; font-size:13px; background:#0B1015; color:var(--text); }
    .mode:disabled, .preset:disabled, .source:disabled { opacity:.6; }
    .sensitivity { min-height:48px; display:grid; gap:6px; color:var(--muted); font-size:12px; padding:5px 0 8px; border-bottom:1px solid var(--line); }
    .sensitivity-title { color:var(--text); font-size:13px; line-height:1.25; }
    .sensitivity-row { display:flex; align-items:center; gap:15px; min-width:0; }
    .sensitivity input { flex:0 1 50%; max-width:50%; min-width:120px; accent-color:var(--accent); }
    .sensitivity strong { align-self:center; color:var(--text); font-size:12px; text-align:left; white-space:nowrap; }
    .media-card { margin:8px; padding:0; display:grid; gap:8px; background:#0B1015; border:0; border-radius:0; }
    .source-strip, .playback-panel, .capture-panel { background:#0F161F; border:1px solid var(--line); border-radius:8px; box-shadow:inset 0 1px 0 rgba(255,255,255,.035); }
    .source-strip { min-height:58px; display:grid; grid-template-columns:auto minmax(0,1fr) auto auto; align-items:center; gap:12px; padding:8px 10px; position:relative; }
    .source-icon { width:36px; height:30px; display:none; align-items:center; justify-content:center; border-radius:7px; color:var(--text); }
    .source-icon svg { width:21px; height:21px; display:block; }
    .source-icon-youtube { display:flex; background:#E5252A; box-shadow:inset 0 1px 0 rgba(255,255,255,.22); }
    .media-card.mode-microphone .source-icon-youtube, .media-card.mode-system .source-icon-youtube { display:none; }
    .media-card.mode-microphone .source-icon-microphone, .media-card.mode-system .source-icon-system { display:flex; background:#0F161F; border:1px solid var(--line); color:#17B7FE; }
    .source-copy { min-width:0; display:flex; align-items:baseline; gap:16px; }
    .source-kind { flex:0 0 auto; color:#D7DEE8; font-size:15px; font-weight:400; white-space:nowrap; }
    .source-title { min-width:0; color:var(--text); font-size:15px; font-weight:400; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .media-time { color:#B7C1CD; font-size:15px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .source-mode-select { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
    .source-mode-menu { position:relative; justify-self:end; }
    .source-mode-button { justify-content:flex-start; padding:0 36px 0 12px; cursor:pointer; }
    .source-mode-menu.open .source-mode-button::after { transform:translateY(-25%) rotate(225deg); }
    .source-mode-options { position:absolute; top:calc(100% + 6px); right:0; z-index:40; width:200px; display:grid; gap:4px; padding:6px; border:1px solid var(--line); border-radius:7px; background:#0F161F; box-shadow:0 18px 46px rgba(0,0,0,.42); }
    .source-mode-options[hidden] { display:none; }
    .source-mode-option { min-height:32px; display:flex; align-items:center; justify-content:flex-start; border-color:var(--line); background:#0B1015; color:#D7DEE8; text-align:left; }
    .source-mode-option.active { border-color:#17B7FE; color:var(--text); background:#0F161F; }
    .playback-panel { min-height:132px; display:grid; grid-template-columns:minmax(150px, 240px) minmax(0,1fr); align-items:stretch; gap:14px; padding:8px; }
    .video-frame { width:100%; aspect-ratio:16/9; overflow:hidden; border:1px solid var(--line); border-radius:7px; background:#0B1015; box-shadow:0 14px 32px rgba(0,0,0,.28); }
    video { width:100%; height:100%; display:block; object-fit:cover; background:#0B1015; }
    audio { display:none; }
    .youtube-stream { display:none; position:relative; overflow:hidden; width:100%; height:100%; background:#0B1015; border:0; }
    .youtube-stream iframe { width:100%; height:100%; border:0; display:block; }
    .youtube-stream.empty iframe { display:none; }
    .stream-hint { display:none; position:absolute; left:10px; right:10px; bottom:10px; min-height:38px; align-items:center; justify-content:center; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:#0F161F; color:var(--muted); text-align:center; line-height:1.35; font-size:13px; }
    .stream-hint:not(:empty) { display:flex; }
    .youtube-stream.empty .stream-hint { inset:0; height:100%; border:0; border-radius:0; background:transparent; padding:24px; }
    .app.browser-stream .media-card.mode-youtube .video-frame video { display:none; }
    .app.browser-stream .media-card.mode-youtube .youtube-stream { display:block; }
    .media-controls { min-width:0; min-height:100%; display:grid; grid-template-rows:auto minmax(0,1fr) auto; align-items:center; gap:8px; }
    .youtube-source-controls { align-self:start; }
    .media-card.mode-microphone .youtube-source-controls, .media-card.mode-system .youtube-source-controls { display:none; }
    .timeline-row { width:100%; min-width:0; align-self:center; display:grid; grid-template-columns:auto minmax(120px,1fr) auto; align-items:center; gap:10px; color:#B7C1CD; font-size:14px; font-variant-numeric:tabular-nums; }
    .timeline-bar { position:relative; height:6px; margin-left:8px; margin-right:10px; border-radius:999px; background:#93A1AF; box-shadow:inset 0 1px 2px rgba(0,0,0,.28); }
    .timeline-fill { position:absolute; inset:0 auto 0 0; width:0%; border-radius:inherit; background:#17B7FE; box-shadow:0 0 16px rgba(23,183,254,.35); }
    .timeline-thumb { position:absolute; top:50%; left:0%; width:18px; height:18px; border-radius:50%; background:#FFFFFF; transform:translate(-50%, -50%); box-shadow:0 2px 10px rgba(0,0,0,.32); }
    .media-expand { width:40px; height:40px; align-self:end; justify-self:start; display:grid; place-items:center; padding:0; border-color:var(--line); background:#0F161F; color:#D7DEE8; }
    .media-expand svg { width:20px; height:20px; }
    .capture-panel { min-height:132px; display:none; grid-template-columns:auto minmax(0,1fr); align-items:center; gap:14px; padding:12px; }
    .media-card.mode-microphone .capture-panel, .media-card.mode-system .capture-panel { display:grid; }
    .media-card.mode-microphone .playback-panel, .media-card.mode-system .playback-panel { display:none; }
    .capture-icon { width:42px; height:42px; display:grid; place-items:center; border:1px solid var(--line); border-radius:8px; background:#0F161F; color:#17B7FE; }
    .capture-icon svg { width:24px; height:24px; display:none; }
    .media-card.mode-microphone .capture-icon-mic, .media-card.mode-system .capture-icon-system { display:block; }
    .capture-body { min-width:0; display:grid; gap:12px; }
    .capture-title { margin:0; color:var(--text); font-size:16px; line-height:1.2; }
    .capture-description { margin:0; color:#9EAAB6; font-size:12px; }
    .level-row, .mic-gain-control { display:grid; grid-template-columns:auto minmax(0,1fr) auto; align-items:center; gap:8px; color:#B7C1CD; font-size:12px; font-weight:400; }
    .level-meter { height:8px; overflow:hidden; border-radius:999px; background:#0B1015; border:1px solid var(--line); }
    .level-fill { width:0%; height:100%; background:#3DC77C; box-shadow:0 0 16px rgba(61,199,124,.32); transition:width .08s linear; }
    .mic-gain-control input { width:100%; accent-color:#17B7FE; }
    .media-card.mode-system .mic-gain-control { display:none; }
    .speaker-panel { padding:0; overflow:hidden; display:grid; gap:0; }
    .speaker-tabs { min-height:38px; display:grid; grid-template-columns:1fr 1fr; border-bottom:1px solid var(--line); }
    .speaker-tab { min-height:38px; border:0; border-radius:0; background:transparent; color:#9EAAB6; font-size:13px; }
    .speaker-tab.active { color:#E8EEF5; box-shadow:inset 0 -2px 0 #17B7FE; }
    .speaker-tab-panel { min-width:0; padding:8px; }
    .speaker-tab-panel[hidden] { display:none; }
    .speaker-panel-header { margin:0 0 7px; display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .speaker-panel-title { margin:0; color:#D7DEE8; font-size:13px; font-weight:400; line-height:1.25; }
    .speaker-panel-actions { flex:0 0 auto; display:flex; align-items:center; gap:7px; }
    .add-speaker-button, .clear-speakers-button { min-height:28px; flex:0 0 auto; border-color:#20303E; background:#121C26; color:#C6D0DC; padding:0 9px; font-size:12px; }
    .clear-speakers-button { color:#E0A0A0; }
    .clear-speakers-button:disabled { opacity:.45; color:#87919C; }
    .manual-speaker-composer { display:grid; gap:8px; margin:0 0 8px; padding:9px 10px 10px 12px; border:1px solid var(--line); border-radius:7px; background:#0F161F; box-shadow:inset 4px 0 0 #17B7FE; }
    .manual-speaker-composer[hidden] { display:none; }
    .speaker-list { display:grid; gap:0; max-height:none; overflow:hidden; border:1px solid var(--line); border-radius:7px; background:#0B1015; }
    .speaker-empty { padding:12px; color:var(--muted); font-size:11px; }
    .speaker-item { --speaker-color:transparent; min-width:0; display:grid; border-bottom:1px solid var(--line); background:#0F161F; }
    .speaker-item:last-child { border-bottom:0; }
    .speaker-item.live-speaker { background:color-mix(in srgb, var(--speaker-color) 18%, #0F161F); box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--speaker-color) 35%, transparent); }
    .speaker-item.editing { position:relative; z-index:1; border:1px solid var(--speaker-color); border-radius:7px; box-shadow:0 0 0 1px color-mix(in srgb, var(--speaker-color) 28%, transparent); }
    .speaker-item-summary { width:100%; min-height:60px; display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:start; gap:8px; padding:9px 10px 9px 12px; border:0; border-radius:0; background:transparent; color:#C6D0DC; text-align:left; box-shadow:inset 4px 0 0 var(--speaker-color); cursor:pointer; }
    .speaker-item.live-speaker .speaker-item-summary { box-shadow:inset 4px 0 0 var(--speaker-color), inset 7px 0 14px color-mix(in srgb, var(--speaker-color) 18%, transparent); }
    .speaker-summary-body { min-width:0; display:grid; gap:2px; }
    .speaker-title-row { min-width:0; display:flex; align-items:center; gap:7px; }
    .speaker-row-title { min-width:0; color:#E8EEF5; font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .speaker-item:not(.editing) .speaker-row-title { color:var(--speaker-color); }
    .speaker-title-row .speaker-row-name-input { flex:1 1 auto; }
    .speaker-live-indicator { flex:0 0 auto; display:inline-flex; align-items:center; gap:4px; padding:2px 6px; border:1px solid color-mix(in srgb, var(--speaker-color) 55%, transparent); border-radius:999px; background:color-mix(in srgb, var(--speaker-color) 18%, #0B1015); color:var(--speaker-color); font-size:12px; line-height:1; box-shadow:0 0 14px color-mix(in srgb, var(--speaker-color) 18%, transparent); }
    .speaker-live-indicator svg { width:13px; height:13px; display:block; animation:livePulse 1s ease-in-out infinite; }
    @keyframes livePulse { 0%, 100% { opacity:.58; transform:scale(.92); } 50% { opacity:1; transform:scale(1.08); } }
    @media (prefers-reduced-motion: reduce) { .speaker-live-indicator svg { animation:none; } }
    .speaker-row-name-input { min-width:0; width:100%; height:24px; margin:-2px -5px; border:1px solid transparent; border-radius:5px; padding:0 4px; background:transparent; color:#E8EEF5; font:inherit; font-size:13px; font-weight:600; outline:none; }
    .speaker-row-name-input:focus { border-color:#20303E; background:#0B1015; }
    .manual-speaker-name { margin:0; width:100%; }
    .speaker-sentence-count { color:#9EAAB6; font-size:11px; }
    .speaker-reference-status { min-width:0; display:flex; align-items:center; gap:6px; color:#8F9BA8; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .speaker-reference-icon { width:12px; height:12px; flex:0 0 auto; border:1px dashed #536271; border-radius:50%; }
    .speaker-reference-status.has-reference { color:#AEB9C5; }
    .speaker-reference-status.has-reference .speaker-reference-icon { border:0; border-radius:0; width:13px; height:8px; border-left:2px solid #3DC77C; border-bottom:2px solid #3DC77C; transform:rotate(-45deg) translateY(-1px); }
    .speaker-item-tail { align-self:stretch; display:flex; flex-direction:column; align-items:flex-end; justify-content:space-between; gap:7px; color:#AEB9C5; }
    .speaker-filter-controls, .speaker-transcript-actions { display:flex; align-items:center; gap:4px; }
    .speaker-filter-toggle { min-height:20px; width:39px; border:1px solid #20303E; border-radius:999px; padding:0 4px; display:inline-flex; align-items:center; justify-content:center; gap:3px; background:#0B1015; color:#8392A2; }
    .speaker-filter-toggle.active { border-color:#1789F2; color:var(--text); background:rgba(23,137,242,.16); }
    .speaker-filter-toggle.mute.active { border-color:#DF3C36; background:rgba(152,29,32,.28); }
    .speaker-filter-toggle svg { width:12px; height:12px; flex:0 0 auto; }
    .speaker-filter-switch { position:relative; flex:0 0 17px; height:10px; border-radius:999px; background:#22313E; }
    .speaker-filter-switch::after { content:""; position:absolute; left:1px; top:1px; width:8px; height:8px; border-radius:50%; background:#AEB9C5; transition:transform .12s ease, background .12s ease; }
    .speaker-filter-toggle.active .speaker-filter-switch { background:#1789F2; }
    .speaker-filter-toggle.mute.active .speaker-filter-switch { background:#981D20; }
    .speaker-filter-toggle.active .speaker-filter-switch::after { transform:translateX(7px); background:#FFFFFF; }
    .speaker-filter-toggle:focus-visible { outline:1px solid #17B7FE; outline-offset:2px; }
    .transcript-icon-button { min-height:24px; width:28px; border:1px solid #20303E; border-radius:6px; padding:0; display:inline-grid; place-items:center; background:#0B1015; color:#AEB9C5; }
    .transcript-icon-button:hover { border-color:#2C4052; color:var(--text); }
    .transcript-icon-button svg { width:13px; height:13px; display:block; }
    .speaker-chevron { width:8px; height:8px; border-right:2px solid currentColor; border-bottom:2px solid currentColor; transform:rotate(-45deg); opacity:.72; }
    .speaker-item.editing .speaker-chevron { transform:rotate(45deg) translateY(-2px); }
    .speaker-editor { display:grid; gap:8px; padding:0 10px 10px; }
    .speaker-editor[hidden] { display:none; }
    .speaker-panel input[type="text"] { width:100%; min-height:32px; border:1px solid var(--line); border-radius:6px; padding:0 9px; background:#0B1015; color:#E8EEF5; font-size:12px; }
    .speaker-reference-actions { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
    .speaker-reference-button, .speaker-panel #recordReference { min-height:30px; border:1px solid #20303E; border-radius:6px; padding:0 10px; display:inline-flex; align-items:center; justify-content:center; gap:6px; background:#121C26; color:#C6D0DC; font-size:12px; white-space:nowrap; }
    .speaker-editor .speaker-reference-button { display:inline-flex; }
    .speaker-reference-title { color:#C6D0DC; font-size:12px; }
    .speaker-reference-button input { display:none; }
    .speaker-reference-button svg, .speaker-panel #recordReference svg { width:14px; height:14px; flex:0 0 auto; }
    .speaker-panel #recordReference.recording { border-color:#DF3C36; color:var(--text); background:#981D20; }
    .speaker-record-seconds { color:#9EAAB6; font-size:11px; }
    .speaker-settings-panel { display:grid; gap:8px; }
    .speaker-setting-toggle { min-height:24px; display:flex; align-items:center; gap:8px; color:#C6D0DC; font-size:12px; line-height:1.25; }
    .speaker-setting-toggle input { width:14px; height:14px; margin:0; accent-color:#1789F2; }
    .speaker-tools { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:center; }
    .speaker-tools strong { grid-column:1 / -1; color:var(--text); font-size:13px; }
    .speaker-file-actions { display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
    .speaker-file-actions button { min-height:28px; width:auto; padding:0 10px; font-size:12px; }
    .speaker-tools .select-control { width:100%; }
    .speaker-tools input { min-height:30px; border:1px solid var(--line); border-radius:6px; padding:0 8px; background:#0B1015; color:var(--text); font-size:13px; }
    .speaker-panel button { font-weight:400; }
    .status { max-height:180px; overflow:auto; color:var(--muted); font-size:12px; line-height:1.3; padding-top:2px; }
    .transcript-panel { min-height:0; margin:0 8px 8px; display:grid; grid-template-rows:auto minmax(0,1fr); position:relative; }
    .transcript-header { display:grid; grid-template-columns:auto minmax(240px,1fr) auto; gap:8px 14px; align-items:center; padding:10px; border-bottom:1px solid var(--line); }
    .transcript-title { grid-column:1 / -1; margin:0; color:#E8EEF5; font-size:15px; line-height:1.2; font-weight:400; }
    .transcript-left-tools, .transcript-right-tools { display:flex; align-items:center; gap:8px; }
    .transcript-right-tools { justify-content:flex-end; }
    .follow-live-toggle { min-height:22px; display:inline-flex; align-items:center; gap:7px; color:#C6D0DC; font-size:12px; }
    .follow-live-toggle input { position:absolute; opacity:0; pointer-events:none; }
    .follow-live-track { position:relative; width:38px; height:20px; border-radius:999px; background:#22313E; box-shadow:inset 0 0 0 1px #20303E; }
    .follow-live-track::after { content:""; position:absolute; left:2px; top:2px; width:16px; height:16px; border-radius:50%; background:#AEB9C5; transition:transform .12s ease, background .12s ease; }
    .follow-live-toggle input:checked + .follow-live-track { background:#1789F2; box-shadow:inset 0 0 0 1px #17B7FE; }
    .follow-live-toggle input:checked + .follow-live-track::after { transform:translateX(18px); background:#FFFFFF; }
    .transcript-search { min-width:0; position:relative; }
    .transcript-search input { width:100%; min-height:34px; border:1px solid var(--line); border-radius:7px; padding:0 34px 0 11px; background:#0B1015; color:#E8EEF5; font-size:13px; outline:none; }
    .transcript-search svg { position:absolute; right:10px; top:50%; width:16px; height:16px; transform:translateY(-50%); color:#8392A2; pointer-events:none; }
    .transcript-settings-menu { position:relative; }
    .transcript-settings-panel { position:absolute; right:0; top:calc(100% + 6px); z-index:30; min-width:190px; display:grid; gap:7px; padding:9px; border:1px solid var(--line); border-radius:7px; background:#0F161F; box-shadow:0 18px 46px rgba(0,0,0,.42); }
    .transcript-settings-panel[hidden] { display:none; }
    .transcript-settings-panel label { display:flex; align-items:center; gap:7px; color:#C6D0DC; font-size:12px; }
    .transcript-settings-panel input { width:14px; height:14px; accent-color:#17B7FE; }
    .sentences { min-height:0; overflow:auto; padding:8px; }
    .row { border-bottom:1px solid var(--line); padding:7px 2px; }
    .top { display:flex; gap:8px; align-items:flex-start; justify-content:space-between; margin-bottom:4px; color:var(--muted); font-size:11px; }
    .top-left { min-width:0; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    .badge { font-weight:400; border-radius:999px; padding:2px 8px; border:1px solid currentColor; background:#0B1015; }
    .badge.unknown { color:#c3ccd6; border-color:#7d8997; background:#0B1015; }
    .badge.new { color:var(--text); border-color:#ef4444; background:#b91c1c; text-transform:uppercase; letter-spacing:0; }
    .badge.state { color:#d7dee8; border-color:var(--line); background:#0B1015; font-weight:400; }
    .transcript-panel.hide-tags .badge.new, .transcript-panel.hide-tags .badge.state { display:none; }
    .transcript-panel.hide-time .sentence-duration, .transcript-panel.hide-time .sentence-range { display:none; }
    .transcript-panel.hide-speech-rate .sentence-speech-rate { display:none; }
    .transcript-panel.hide-probabilities .prob { display:none; }
    .speaker-name, .speaker-row-title { font-weight:600; }
    .text { font-size:15px; line-height:1.34; }
    .row.realtime { background:color-mix(in srgb, var(--live-row-color, #8F9BA8) 10%, #0B1015); }
    .row.realtime.live-speaker-row { background:color-mix(in srgb, var(--live-row-color, #8F9BA8) 18%, #0B1015); border-bottom-color:color-mix(in srgb, var(--live-row-color, #8F9BA8) 35%, var(--line)); box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--live-row-color, #8F9BA8) 35%, transparent), inset 7px 0 14px color-mix(in srgb, var(--live-row-color, #8F9BA8) 18%, transparent); }
    .row.realtime .text { color:#d7dee8; }
    .prob { flex:0 0 min(180px, 24vw); display:flex; width:min(180px, 24vw); height:6px; overflow:hidden; border:1px solid var(--line); border-radius:4px; background:#0B1015; margin-top:4px; }
    .prob span { display:block; height:100%; min-width:0; }
    @media (max-width: 900px) {
      body { min-height:100dvh; height:auto; overflow:auto; }
      .app { min-height:100dvh; height:auto; display:block; }
      .topbar { position:sticky; top:0; z-index:20; min-height:auto; padding:8px; flex-wrap:wrap; }
      .live-summary { width:auto; flex-wrap:wrap; gap:8px 10px; }
      .brand { gap:7px; }
      .brand-icon { width:22px; height:22px; }
      .brand-icon svg, .speaker-summary svg { width:19px; height:19px; }
      .title { font-size:15px; }
      .topbar-divider { height:22px; }
      .runtime-state { max-width:45vw; font-size:12px; }
      .status-pill, .speaker-summary { font-size:12px; }
      .transport button { min-height:38px; padding:0 12px; }
      .layout { min-height:0; display:flex; flex-direction:column; }
      .control-panel { order:1; overflow:visible; padding:8px; }
      .workspace { order:2; display:flex; flex-direction:column; min-height:0; }
      .source-row { grid-template-columns:1fr; }
      .mode, .preset, .source, .speaker-panel input, .speaker-panel select, button { min-height:38px; font-size:14px; }
      .speaker-filter-toggle { min-height:20px; width:39px; font-size:12px; }
      .transcript-icon-button { min-height:24px; width:28px; font-size:13px; }
      .sensitivity-row { display:grid; grid-template-columns:1fr; gap:6px; }
      .sensitivity input { max-width:none; min-width:0; }
      .sensitivity strong { text-align:left; }
      .media-card { margin:8px; gap:8px; }
      .source-strip { grid-template-columns:auto minmax(0,1fr); gap:8px; padding:8px; }
      .source-copy { display:grid; gap:2px; }
      .source-kind, .source-title, .media-time { font-size:14px; }
      .media-time, .source-mode-menu { grid-column:2; justify-self:start; }
      .playback-panel { min-height:0; grid-template-columns:1fr; gap:10px; padding:8px; }
      .video-frame { max-width:300px; }
      .media-expand { width:38px; height:38px; justify-self:start; }
      .capture-panel { min-height:0; padding:10px; grid-template-columns:1fr; }
      .transcript-panel { margin:0 8px 8px; min-height:45vh; display:block; }
      .transcript-header { grid-template-columns:1fr; gap:8px; padding:9px; }
      .transcript-left-tools, .transcript-right-tools { justify-content:flex-start; }
      .sentences { overflow:visible; padding:8px; }
      .status { max-height:130px; }
      .speaker-list { max-height:none; }
      .speaker-item-summary { min-height:58px; grid-template-columns:minmax(0,1fr) auto; padding:8px 9px; }
      .speaker-editor { padding:0 9px 10px; }
      .speaker-tools { grid-template-columns:1fr 1fr; }
      .speaker-tools strong { grid-column:1 / -1; }
      .speaker-tools .select-control, .speaker-tools input { grid-column:1 / -1; max-width:none; }
      .top { display:grid; grid-template-columns:1fr; gap:6px; }
      .prob { width:100%; flex-basis:auto; }
      .text { font-size:15px; line-height:1.36; }
    }
    @media (max-width: 460px) {
      .topbar { align-items:stretch; }
      .live-summary { width:100%; justify-content:flex-end; }
      .topbar > .topbar-divider { display:none; }
      .transport { gap:6px; }
      .transport button { padding:0 10px; }
      .source-strip { grid-template-columns:1fr; }
      .source-icon { width:36px; height:30px; }
      .media-time, .source-mode-menu { grid-column:1; }
      .source-mode-menu, .source-mode-button { width:100%; }
      .source-mode-options { width:100%; left:0; right:auto; }
      .source-grid { grid-template-columns:1fr; }
      .timeline-row { grid-template-columns:1fr; gap:6px; font-size:13px; }
      .timeline-thumb { width:16px; height:16px; }
      .level-row, .mic-gain-control { grid-template-columns:1fr; }
      .control-panel { padding:8px; gap:8px; }
      .control-card, .media-card, .transcript-panel { border-radius:6px; }
      .speaker-editor { padding:0 9px 10px; }
      .speaker-reference-actions, .speaker-tools { display:grid; grid-template-columns:1fr; }
      .speaker-file-actions { display:flex; }
    }
  </style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="brand-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
          <path d="M4 10v4"></path>
          <path d="M8 6v12"></path>
          <path d="M12 3v18"></path>
          <path d="M16 7v10"></path>
          <path d="M20 11v2"></path>
        </svg>
      </span>
      <div class="title">WhoSpeaks Live</div>
    </div>
    <div class="live-summary">
      <div class="status-pill" aria-live="polite">
        <span class="status-dot" aria-hidden="true"></span>
        <span id="state" class="runtime-state">Ready</span>
      </div>
      <span class="topbar-divider" aria-hidden="true"></span>
      <div class="speaker-summary">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="10" cy="7" r="4"></circle>
          <path d="M22 21v-2a4 4 0 0 0-3-3.87"></path>
          <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
        </svg>
        <span id="speakerCount"><span id="speakerCountNumber" class="speaker-count-number">0</span><span id="speakerCountLabel" class="speaker-count-label">speakers found</span></span>
      </div>
    </div>
    <span class="topbar-divider" aria-hidden="true"></span>
    <div class="transport">
      <button id="start">Start transcription</button>
      <button id="stop" disabled><span class="stop-icon" aria-hidden="true"></span>Stop transcription</button>
    </div>
  </header>
  <div id="sessionBanner" class="session-banner available">
    <span id="sessionBannerMessage" class="session-banner-message">Demo seat available. Press Start or Load to take it.</span>
    <span class="session-banner-actions">
      <button id="releaseSession" type="button" hidden>Release seat</button>
    </span>
  </div>
  <main class="layout">
    <section class="workspace">
      <section id="mediaCard" class="media-card mode-youtube">
        <section class="source-strip" aria-label="Media source">
          <span class="source-icon source-icon-youtube" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M9.5 8.2v7.6l6.7-3.8-6.7-3.8z"></path>
            </svg>
          </span>
          <span class="source-icon source-icon-microphone" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <path d="M12 19v3"></path>
            </svg>
          </span>
          <span class="source-icon source-icon-system" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="4" width="18" height="12" rx="2"></rect>
              <path d="M8 20h8"></path>
              <path d="M12 16v4"></path>
            </svg>
          </span>
          <div class="source-copy">
            <span class="source-kind">Source: <span id="sourceKind">YouTube</span></span>
            <strong id="sourceTitle" class="source-title">Loading source</strong>
          </div>
          <div id="mediaTime" class="media-time">00:00 / 00:00</div>
          <select id="inputMode" class="mode source-mode-select" aria-label="Input mode">
            <option value="youtube">YouTube video</option>
            <option value="microphone">Microphone</option>
            <option value="system">Computer audio</option>
          </select>
          <div id="sourceModeMenu" class="source-mode-menu">
            <button id="sourceModeButton" class="source-mode-button dropdown-control" type="button" aria-expanded="false" aria-controls="sourceModeOptions">Change source</button>
            <div id="sourceModeOptions" class="source-mode-options" hidden>
              <button class="source-mode-option" type="button" data-input-mode="youtube">YouTube video</button>
              <button class="source-mode-option" type="button" data-input-mode="microphone">Microphone</button>
              <button class="source-mode-option" type="button" data-input-mode="system">Computer audio</button>
            </div>
          </div>
        </section>
        <section class="playback-panel" aria-label="Video playback">
          <div class="video-frame">
            <video id="video" src="/media/video" muted playsinline></video>
            <div id="youtubeStream" class="youtube-stream"><iframe id="youtubeFrame" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe><div id="streamHint" class="stream-hint"></div></div>
          </div>
          <div class="media-controls">
            <div id="youtubeSourceControls" class="source-grid youtube-source-controls">
              <span class="select-control dropdown-control"><select id="preset" class="preset" aria-label="Preset video"></select></span>
              <div class="source-row">
                <input id="source" class="source" type="url" spellcheck="false" autocomplete="off">
                <button id="load">Load</button>
              </div>
            </div>
            <div class="timeline-row" aria-label="Playback progress">
              <span id="mediaCurrentTime">00:00</span>
              <div id="timelineBar" class="timeline-bar" aria-hidden="true">
                <span id="timelineFill" class="timeline-fill"></span>
                <span id="timelineThumb" class="timeline-thumb"></span>
              </div>
              <span id="mediaDuration">00:00</span>
            </div>
            <button id="expandMedia" class="media-expand" type="button" aria-label="Enlarge video">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M15 3h6v6"></path>
                <path d="M21 3l-7 7"></path>
                <path d="M9 21H3v-6"></path>
                <path d="M3 21l7-7"></path>
              </svg>
            </button>
          </div>
          <audio id="audio" src="/media/audio"></audio>
        </section>
        <section id="capturePanel" class="capture-panel" aria-label="Audio input level">
          <div class="capture-icon" aria-hidden="true">
            <svg class="capture-icon-mic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <path d="M12 19v3"></path>
            </svg>
            <svg class="capture-icon-system" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="4" width="18" height="12" rx="2"></rect>
              <path d="M8 20h8"></path>
              <path d="M12 16v4"></path>
            </svg>
          </div>
          <div class="capture-body">
            <h2 id="captureTitle" class="capture-title">Microphone input</h2>
            <p id="captureDescription" class="capture-description">Live input level appears after capture starts.</p>
            <div class="level-row">
              <span>Volume</span>
              <div class="level-meter"><div id="captureLevelFill" class="level-fill"></div></div>
              <span id="captureLevelText">0%</span>
            </div>
            <label id="micGainControl" class="mic-gain-control">
              <span>Mic gain</span>
              <input id="micGain" type="range" min="0" max="2" step="0.05" value="1">
              <span id="micGainValue">1.00x</span>
            </label>
          </div>
        </section>
      </section>
      <section class="transcript-panel">
        <div class="transcript-header">
          <h2 class="transcript-title">Live transcript</h2>
          <div class="transcript-left-tools">
            <label class="follow-live-toggle">
              <input id="followLive" type="checkbox" checked>
              <span class="follow-live-track" aria-hidden="true"></span>
              <span>Follow live</span>
            </label>
          </div>
          <div class="transcript-search">
            <input id="transcriptSearch" type="search" placeholder="Search transcript" autocomplete="off">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="7"></circle>
              <path d="m20 20-3.5-3.5"></path>
            </svg>
          </div>
          <div class="transcript-right-tools">
            <button id="copyTranscript" class="transcript-icon-button" type="button" title="Copy transcript" aria-label="Copy transcript">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="9" y="9" width="11" height="11" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
            <button id="downloadTranscript" class="transcript-icon-button" type="button" title="Download transcript" aria-label="Download transcript">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M12 3v12"></path>
                <path d="m7 10 5 5 5-5"></path>
                <path d="M5 21h14"></path>
              </svg>
            </button>
            <div class="transcript-settings-menu">
              <button id="transcriptSettings" class="transcript-icon-button" type="button" title="Transcript settings" aria-label="Transcript settings" aria-expanded="false">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <circle cx="12" cy="12" r="3"></circle>
                  <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.6-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1A2 2 0 1 1 7.1 4.2l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.1a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.6 1Z"></path>
                </svg>
              </button>
              <div id="transcriptSettingsPanel" class="transcript-settings-panel" hidden>
                <label><input id="showTranscriptTags" type="checkbox" checked> Show tags</label>
                <label><input id="showTranscriptTime" type="checkbox" checked> Show time information</label>
                <label><input id="showTranscriptSpeechRate" type="checkbox" checked> Show speech/audio rate</label>
                <label><input id="showTranscriptProbabilities" type="checkbox" checked> Show probabilities</label>
              </div>
            </div>
          </div>
        </div>
        <section id="sentences" class="sentences"></section>
      </section>
    </section>
    <aside class="control-panel">
      <section class="control-card speaker-panel" aria-label="Speakers">
        <div class="speaker-tabs" role="tablist" aria-label="Speaker controls">
          <button class="speaker-tab active" type="button" role="tab" aria-selected="true" data-speaker-tab="speakers">Speakers</button>
          <button class="speaker-tab" type="button" role="tab" aria-selected="false" data-speaker-tab="settings">Settings</button>
        </div>
        <div class="speaker-tab-panel" data-speaker-panel="speakers">
          <div class="speaker-panel-header">
            <h2 id="speakerPanelTitle" class="speaker-panel-title">Detected speakers (0)</h2>
            <span class="speaker-panel-actions">
              <button id="clearSpeakers" class="clear-speakers-button" type="button">Clear speakers</button>
              <button id="addReferenceSpeaker" class="add-speaker-button" type="button" aria-expanded="false">Add speaker</button>
            </span>
          </div>
          <div id="manualSpeakerComposer" class="manual-speaker-composer" hidden>
            <input id="manualSpeakerName" class="speaker-row-name-input manual-speaker-name" type="text" placeholder="Speaker name" autocomplete="off" aria-label="New speaker name">
            <div id="manualSpeakerReferenceDock"></div>
          </div>
          <div id="speakerList" class="speaker-list"></div>
          <div id="speakerEditorDock" hidden>
            <form id="referenceSpeakerForm" class="speaker-editor" hidden>
              <div class="speaker-reference-title">Add voice reference</div>
              <div class="speaker-reference-actions">
                <label class="speaker-reference-button">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M12 16V4"></path>
                    <path d="M7 9l5-5 5 5"></path>
                    <path d="M5 20h14"></path>
                  </svg>
                  <span>Upload audio</span>
                  <input id="referenceSpeakerFile" type="file" accept="audio/*">
                </label>
                <button id="recordReference" type="button">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <path d="M12 19v3"></path>
                  </svg>
                  <span>Record from mic</span>
                </button>
                <span id="referenceRecordSeconds" class="speaker-record-seconds">0.0s</span>
              </div>
            </form>
          </div>
        </div>
        <div class="speaker-tab-panel speaker-settings-panel" data-speaker-panel="settings" hidden>
          <label class="sensitivity" title="Controls how easily the diarizer creates a new speaker profile.">
            <span class="sensitivity-title">New speaker</span>
            <span class="sensitivity-row">
              <input id="newSpeakerSensitivity" type="range" min="1" max="5" step="1">
              <strong id="newSpeakerSensitivityLabel"></strong>
            </span>
          </label>
          <label class="speaker-setting-toggle" title="When enabled, later prototype evidence may revise an already labeled transcript row. Disabled still allows UNKNOWN rows to be filled later.">
            <input id="allowSpeakerReassignment" type="checkbox">
            <span>Allow later speaker reassignment</span>
          </label>
          <div class="speaker-tools">
            <strong>Speaker groups</strong>
            <span class="speaker-file-actions">
              <button id="loadSpeakerGroup" type="button">Load file</button>
              <button id="saveSpeakerGroup" type="button">Save file</button>
            </span>
            <input id="speakerGroupFile" type="file" accept=".whospeaks-speakers.json,.json,application/json" hidden>
          </div>
        </div>
      </section>
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
const sessionBanner = document.getElementById("sessionBanner");
const sessionBannerMessage = document.getElementById("sessionBannerMessage");
const releaseSessionButton = document.getElementById("releaseSession");
const preset = document.getElementById("preset");
const state = document.getElementById("state");
const source = document.getElementById("source");
const mediaCard = document.getElementById("mediaCard");
const sourceKind = document.getElementById("sourceKind");
const sourceTitle = document.getElementById("sourceTitle");
const sourceModeMenu = document.getElementById("sourceModeMenu");
const sourceModeButton = document.getElementById("sourceModeButton");
const sourceModeOptions = document.getElementById("sourceModeOptions");
const sourceModeOptionButtons = Array.from(document.querySelectorAll(".source-mode-option"));
const mediaTime = document.getElementById("mediaTime");
const mediaCurrentTime = document.getElementById("mediaCurrentTime");
const mediaDuration = document.getElementById("mediaDuration");
const timelineFill = document.getElementById("timelineFill");
const timelineThumb = document.getElementById("timelineThumb");
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
const followLive = document.getElementById("followLive");
const transcriptSearch = document.getElementById("transcriptSearch");
const copyTranscriptButton = document.getElementById("copyTranscript");
const downloadTranscriptButton = document.getElementById("downloadTranscript");
const transcriptSettingsButton = document.getElementById("transcriptSettings");
const transcriptSettingsPanel = document.getElementById("transcriptSettingsPanel");
const showTranscriptTags = document.getElementById("showTranscriptTags");
const showTranscriptTime = document.getElementById("showTranscriptTime");
const showTranscriptSpeechRate = document.getElementById("showTranscriptSpeechRate");
const showTranscriptProbabilities = document.getElementById("showTranscriptProbabilities");
const inputMode = document.getElementById("inputMode");
const newSpeakerSensitivity = document.getElementById("newSpeakerSensitivity");
const newSpeakerSensitivityLabel = document.getElementById("newSpeakerSensitivityLabel");
const allowSpeakerReassignment = document.getElementById("allowSpeakerReassignment");
const loadSpeakerGroupButton = document.getElementById("loadSpeakerGroup");
const saveSpeakerGroupButton = document.getElementById("saveSpeakerGroup");
const speakerGroupFile = document.getElementById("speakerGroupFile");
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
const referenceSpeakerForm = document.getElementById("referenceSpeakerForm");
const referenceSpeakerFile = document.getElementById("referenceSpeakerFile");
const recordReferenceButton = document.getElementById("recordReference");
const recordReferenceButtonLabel = recordReferenceButton.querySelector("span");
const referenceRecordSeconds = document.getElementById("referenceRecordSeconds");
const speakerColors = __SPEAKER_COLORS__;
const initialSource = __SOURCE_JSON__;
const presetVideos = __PRESET_VIDEOS__;
const speakerSensitivityConfig = __NEW_SPEAKER_SENSITIVITY_JSON__;
const speakerRefinementConfig = __SPEAKER_REFINEMENT_JSON__;
const liveSpeakerConfig = __LIVE_SPEAKER_JSON__;
const sessionLeaseEnabled = liveSpeakerConfig.session_lease_enabled !== false;
const initialSpeakerLibrary = __SPEAKER_LIBRARY_JSON__;
const svgNamespace = "http://www.w3.org/2000/svg";
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
let renderedSpeakerSentenceCounts = {};
let renderedSpeakerSpeakingSeconds = {};
let hasRenderedFinalSentenceRows = false;
let fastSpeakerPanelStats = {};
let fastSpeakerPanelLastRight = null;
let soloSpeakerIds = new Set();
let mutedSpeakerIds = new Set();
let followLiveEnabled = true;
let transcriptSearchText = "";
let currentLiveSpeakerId = "";
let transcriptLiveSpeakerId = "";
let fallbackLiveSpeakerId = "";
let fallbackLiveSpeakerUntilMs = 0;
let fallbackLiveSpeakerExpiryTimer = null;
let fallbackLiveSpeakerClearTimer = null;
let transcriptLiveSpeakerExpiryTimer = null;
let liveSpeakerTimeline = [];
let transcriptLiveSpeakerOverrideId = "";
let browserLiveObservationTimer = null;
let browserLiveObservationBuffer = [];
let browserLiveObservationStarted = false;
let browserLiveObservationPosting = false;
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
let editingSpeakerId = "";
let pendingSpeakerNameFocusId = "";
let manualSpeakerComposerOpen = false;
let pendingManualSpeakerNameFocus = false;
const sessionClientIdStorageKey = "whospeaks.demo.client_id";
const sessionTokenStorageKey = "whospeaks.demo.session_token";
let sessionClientId = "";
let sessionToken = "";
let sessionState = {active:false, is_owner:false, running:false, completed:false};
let sessionHeartbeatTimer = null;
let sessionStatusTimer = null;
let sessionCompletionReleaseTimer = null;
if (statusCard && window.matchMedia("(max-width: 900px)").matches) {
  statusCard.open = false;
}
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
function initializeSessionIdentity() {
  sessionClientId = storedSessionValue(sessionClientIdStorageKey);
  if (!sessionClientId) {
    sessionClientId = randomSessionId();
    storeSessionValue(sessionClientIdStorageKey, sessionClientId);
  }
  sessionToken = storedSessionValue(sessionTokenStorageKey);
}
initializeSessionIdentity();
function setState(text) { state.textContent = text; }
function updateSpeakerCount() {
  const count = Array.isArray(speakerLibraryState.speakers) ? speakerLibraryState.speakers.length : 0;
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
  sourceModeButton.disabled = disabled;
  sourceModeOptionButtons.forEach(button => { button.disabled = disabled; });
  if (!disabled) syncSessionControlLock();
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
  try {
    const parsed = new URL(text);
    return parsed.hostname.replace(/^www\./, "") || "Custom source";
  } catch (_) {
    return text.length > 64 ? `${text.slice(0, 61)}...` : text;
  }
}
function updateMediaMode() {
  const mode = inputMode.value || "youtube";
  mediaCard.classList.toggle("mode-youtube", mode === "youtube");
  mediaCard.classList.toggle("mode-microphone", mode === "microphone");
  mediaCard.classList.toggle("mode-system", mode === "system");
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
  } else {
    sourceKind.textContent = "YouTube";
    sourceTitle.textContent = sourceTitleForUrl(source.value);
    updateMediaTimeline();
  }
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
  if ((inputMode.value || "youtube") !== "youtube") {
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
  const gain = Number(micGain.value || 1);
  micGainValue.textContent = `${gain.toFixed(2)}x`;
}
function captureGainValue() {
  if ((inputMode.value || "youtube") !== "microphone") return 1;
  const gain = Number(micGain.value || 1);
  return Number.isFinite(gain) ? Math.max(0, Math.min(2, gain)) : 1;
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
populatePresetVideos();
source.value = initialSource;
syncPresetSelection(initialSource);
updateMicGainLabel();
setCaptureLevel(0);
updateMediaMode();
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
  speakerSensitivityDirty = false;
  updateSpeakerSensitivityLabel();
  return result;
}
async function applySpeakerSensitivityIfDirty() {
  if (!speakerSensitivityDirty) return null;
  return applySpeakerSensitivity();
}
function syncSpeakerReassignmentSetting(settings) {
  if (!settings || typeof settings !== "object") return;
  allowSpeakerReassignment.checked = Boolean(settings.allow_reassignment);
}
async function applySpeakerReassignmentSetting() {
  await ensureSessionOwner("change speaker settings");
  const requested = allowSpeakerReassignment.checked;
  try {
    const result = await post("/api/settings", {allow_speaker_reassignment: requested});
    syncSpeakerReassignmentSetting(result.speaker_refinement);
    return result;
  } catch (error) {
    allowSpeakerReassignment.checked = !requested;
    log(`Speaker reassignment setting failed: ${error.message}`);
    throw error;
  }
}
newSpeakerSensitivity.value = speakerSensitivityConfig.selected || 3;
updateSpeakerSensitivityLabel();
syncSpeakerReassignmentSetting(speakerRefinementConfig);
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
  updateMediaMode();
}
function browserStreamSourceUrl() {
  if (captureSourceKind === "microphone") return "microphone://local";
  const url = source.value.trim();
  return url || "system-audio://local";
}
async function prepareBrowserStreamSession() {
  await ensureSessionOwner("prepare browser audio");
  const url = browserStreamSourceUrl();
  if (browserStreamPrepared && browserStreamPreparedUrl === url) return;
  const result = await post("/api/browser-stream", {url});
  if (result.speaker_state) updateSpeakerState(result.speaker_state);
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
    const copy = copyCaptureSamples(input);
    const level = rms(copy);
    setCaptureLevel(level);
    if (!captureAudioStarted) {
      rememberCapturePreRoll(copy, captureAudioContext.sampleRate);
      if (level < captureStartRmsThreshold) {
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
function sessionControlsLocked() {
  if (!sessionLeaseEnabled) return false;
  return Boolean(sessionState && sessionState.active && !sessionState.is_owner);
}
function sessionOwnerActive() {
  if (!sessionLeaseEnabled) return true;
  return Boolean(sessionState && sessionState.active && sessionState.is_owner && sessionToken);
}
function sessionSecondsText(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) return "";
  if (value >= 60) return `${Math.ceil(value / 60)} min`;
  return `${Math.ceil(value)}s`;
}
function sessionExpiryText(fieldName="expires_in_seconds") {
  const seconds = Number(sessionState && sessionState[fieldName]);
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
  if (!sessionState || !sessionState.active) {
    sessionBanner.classList.add("available");
    sessionBannerMessage.textContent = "Demo seat available. Press Start or Load to take it.";
  } else if (sessionState.is_owner) {
    sessionBanner.classList.add("owner");
    releaseSessionButton.hidden = false;
    if (sessionState.running) {
      const hardLimit = sessionExpiryText("hard_expires_in_seconds");
      const heartbeatGrace = sessionSecondsText(sessionState.heartbeat_timeout_seconds || 45);
      sessionBannerMessage.textContent = hardLimit
        ? `You control this demo. Hard limit in ${hardLimit}; it releases when the run ends or if this browser stops checking in for ${heartbeatGrace}.`
        : `You control this demo. It releases when the run ends or if this browser stops checking in for ${heartbeatGrace}.`;
    } else if (sessionState.completed) {
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
    if (sessionState.running) {
      const hardLimit = sessionExpiryText("hard_expires_in_seconds");
      sessionBannerMessage.textContent = hardLimit
        ? `Session in use. Watching live; controls unlock when the run ends, the owner leaves, or the hard limit hits in ${hardLimit}.`
        : "Session in use. Watching live; controls unlock when the run ends or the owner leaves.";
    } else if (sessionState.completed) {
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
    allowSpeakerReassignment,
    clearSpeakersButton,
    addReferenceSpeakerButton,
    loadSpeakerGroupButton,
    saveSpeakerGroupButton,
    manualSpeakerName,
    referenceSpeakerFile,
    recordReferenceButton,
  ].forEach(element => applySessionLockedDisabled(element, locked));
  sourceModeOptionButtons.forEach(button => applySessionLockedDisabled(button, locked));
}
function updateSessionState(nextState) {
  if (!nextState || typeof nextState !== "object") return;
  const wasLocked = sessionControlsLocked();
  sessionState = {...sessionState, ...nextState};
  if (!sessionState.active || !sessionState.is_owner) {
    if (!sessionState.active || sessionToken) {
      sessionToken = "";
      storeSessionValue(sessionTokenStorageKey, "");
    }
    stopSessionHeartbeat();
  } else if (sessionToken) {
    startSessionHeartbeat();
  }
  updateSessionBanner();
  if (wasLocked !== sessionControlsLocked()) {
    renderSpeakerPanel();
  }
}
async function fetchSessionStatus() {
  if (!sessionLeaseEnabled) return {};
  const params = new URLSearchParams({client_id: sessionClientId});
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
    body:JSON.stringify({client_id: sessionClientId}),
  });
  const data = await response.json();
  if (data.session) updateSessionState(data.session);
  if (!response.ok || !data.acquired) {
    throw new Error(data.error || "Session in use. Watching live until the seat is free.");
  }
  sessionToken = data.session_token || "";
  storeSessionValue(sessionTokenStorageKey, sessionToken);
  if (data.session) updateSessionState({...data.session, is_owner:true});
  startSessionHeartbeat();
  return true;
}
async function ensureSessionOwner(actionLabel="control this demo") {
  if (!sessionLeaseEnabled) return true;
  if (sessionOwnerActive()) return true;
  await fetchSessionStatus().catch(() => null);
  if (sessionOwnerActive()) return true;
  if (sessionState && sessionState.active && !sessionState.is_owner) {
    throw new Error(`Session in use. You are watching live and cannot ${actionLabel} yet.`);
  }
  await acquireSession();
  log("Demo seat acquired.");
  return true;
}
async function heartbeatSession() {
  if (!sessionLeaseEnabled) return;
  if (!sessionToken) return;
  try {
    const response = await fetch("/api/session/heartbeat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({client_id: sessionClientId, session_token: sessionToken}),
    });
    const data = await response.json();
    if (data.session) updateSessionState(data.session);
    if (!response.ok) {
      sessionToken = "";
      storeSessionValue(sessionTokenStorageKey, "");
      stopSessionHeartbeat();
    }
  } catch (_) {}
}
function startSessionHeartbeat() {
  if (!sessionLeaseEnabled) return;
  if (sessionHeartbeatTimer || !sessionToken) return;
  void heartbeatSession();
  sessionHeartbeatTimer = setInterval(heartbeatSession, 5000);
}
function stopSessionHeartbeat() {
  if (sessionHeartbeatTimer) {
    clearInterval(sessionHeartbeatTimer);
    sessionHeartbeatTimer = null;
  }
}
async function releaseSession(reason="released") {
  if (!sessionLeaseEnabled) return;
  if (!sessionToken) {
    await fetchSessionStatus().catch(() => null);
    return;
  }
  const token = sessionToken;
  sessionToken = "";
  storeSessionValue(sessionTokenStorageKey, "");
  stopSessionHeartbeat();
  const response = await fetch("/api/session/release", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({client_id: sessionClientId, session_token: token, reason}),
  });
  const data = await response.json();
  if (data.session) updateSessionState(data.session);
}
function sendSessionReleaseBeacon(reason="tab closed") {
  if (!sessionLeaseEnabled) return;
  if (!sessionToken || !navigator.sendBeacon) return;
  const payload = JSON.stringify({client_id: sessionClientId, session_token: sessionToken, reason});
  const body = new Blob([payload], {type:"application/json"});
  navigator.sendBeacon("/api/session/release", body);
  sessionToken = "";
  storeSessionValue(sessionTokenStorageKey, "");
}
function scheduleCompletedSessionRelease() {
  if (!sessionLeaseEnabled) return;
  if (!sessionOwnerActive()) return;
  if (sessionCompletionReleaseTimer) clearTimeout(sessionCompletionReleaseTimer);
  const delay = Math.max(1000, Number(sessionState.completed_release_delay_seconds || 10) * 1000);
  sessionCompletionReleaseTimer = setTimeout(() => {
    releaseSession("completed").catch(() => {});
  }, delay);
}
function startSessionStatusPolling() {
  if (!sessionLeaseEnabled) return;
  if (sessionStatusTimer) clearInterval(sessionStatusTimer);
  sessionStatusTimer = setInterval(() => fetchSessionStatus().catch(() => {}), 10000);
}
async function post(path, payload={}) {
  const requestPayload = {...payload};
  if (path.startsWith("/api/") && !path.startsWith("/api/session/")) {
    requestPayload.client_id = sessionClientId;
    if (sessionToken) requestPayload.session_token = sessionToken;
  }
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(requestPayload)});
  const data = await r.json();
  if (data.session) updateSessionState(data.session);
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
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
function browserLiveObservationSample() {
  if (!browserLiveObservationEnabled() || !browserLiveObservationStarted) return;
  const liveRows = Array.from(speakerList.querySelectorAll(".speaker-item.live-speaker"));
  const domLiveSpeakerIds = liveRows
    .map(row => row.dataset.speakerId || "")
    .filter(Boolean);
  browserLiveObservationBuffer.push({
    wall_time: Date.now() / 1000,
    performance_ms: performance.now(),
    playback_time: playbackSeconds(),
    dom_live_speaker_ids: domLiveSpeakerIds,
    visible_live_speaker_id: domLiveSpeakerIds.length === 1 ? domLiveSpeakerIds[0] : "",
    current_live_speaker_id: currentLiveSpeakerId || "",
    transcript_live_speaker_id: transcriptLiveSpeakerId || "",
    transcript_live_override_speaker_id: transcriptLiveSpeakerOverrideId || "",
    fallback_live_speaker_id: fallbackLiveSpeakerId || "",
    runtime_state: state.textContent || "",
  });
  if (browserLiveObservationBuffer.length >= 10) {
    void flushBrowserLiveObservation(false, "batch");
  }
}
async function flushBrowserLiveObservation(finalFlush=false, reason="batch") {
  if (!browserLiveObservationEnabled()) return null;
  if (browserLiveObservationPosting && !finalFlush) return null;
  const samples = browserLiveObservationBuffer.splice(0);
  if (!samples.length && !finalFlush) return null;
  browserLiveObservationPosting = true;
  try {
    const endpoint = finalFlush ? "/api/live-observation-finish" : "/api/live-observation";
    return await post(endpoint, {samples, reason});
  } catch (error) {
    if (samples.length) browserLiveObservationBuffer = samples.concat(browserLiveObservationBuffer);
    log(`Browser live observation failed: ${error.message}`);
    return null;
  } finally {
    browserLiveObservationPosting = false;
  }
}
function stopBrowserLiveObservationTimerOnly() {
  if (browserLiveObservationTimer) {
    clearInterval(browserLiveObservationTimer);
    browserLiveObservationTimer = null;
  }
}
function startBrowserLiveObservation() {
  if (!browserLiveObservationEnabled()) return;
  stopBrowserLiveObservationTimerOnly();
  browserLiveObservationBuffer = [];
  browserLiveObservationStarted = true;
  browserLiveObservationSample();
  browserLiveObservationTimer = setInterval(browserLiveObservationSample, browserLiveObservationIntervalMs());
}
async function stopBrowserLiveObservation(reason="done") {
  if (!browserLiveObservationEnabled() || !browserLiveObservationStarted) return null;
  stopBrowserLiveObservationTimerOnly();
  browserLiveObservationSample();
  browserLiveObservationStarted = false;
  const result = await flushBrowserLiveObservation(true, reason);
  if (result && result.summary) {
    log(`Browser live score ${Number(result.summary.strict_browser_live_score || 0).toFixed(3)}`);
  }
  return result;
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
  stopBrowserLiveObservationTimerOnly();
  browserLiveObservationStarted = false;
  browserLiveObservationBuffer = [];
  sentences.textContent = "";
  statusBox.textContent = "";
  currentRealtimeGeneration = 0;
  clearLiveSpeakerState();
  clearUnsavedDetectedSpeakerDisplay();
  renderedSpeakerSentenceCounts = {};
  renderedSpeakerSpeakingSeconds = {};
  hasRenderedFinalSentenceRows = false;
  refreshSpeakerPanelSentenceCounts();
}
function clearUnsavedDetectedSpeakerDisplay() {
  if (speakerLibraryState.group_name) return;
  const retainedSpeakers = speakerLibraryState.speakers.filter(speaker => (
    speaker.source === "reference" || speaker.locked || speaker.reference_audio
  ));
  if (retainedSpeakers.length === speakerLibraryState.speakers.length) return;
  updateSpeakerState({...speakerLibraryState, speakers: retainedSpeakers});
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
  updateMediaTimeline();
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
  if (!followLiveEnabled) return;
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
function speakerPanelName(speaker) {
  return speaker.name || speaker.display_name || speakerDisplayLabel(speaker.id);
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
function pruneSpeakerFilterState() {
  const validSpeakerIds = new Set(speakerLibraryState.speakers.map(speaker => speaker.id).filter(Boolean));
  soloSpeakerIds = new Set(Array.from(soloSpeakerIds).filter(speakerId => validSpeakerIds.has(speakerId)));
  mutedSpeakerIds = new Set(Array.from(mutedSpeakerIds).filter(speakerId => validSpeakerIds.has(speakerId)));
}
function speakerTranscriptVisible(speakerId) {
  if (mutedSpeakerIds.has(speakerId)) return false;
  if (soloSpeakerIds.size > 0) return soloSpeakerIds.has(speakerId);
  return true;
}
function transcriptSearchVisible(row) {
  const query = transcriptSearchText.trim().toLowerCase();
  if (!query) return true;
  const searchable = (row.dataset.searchText || "").toLowerCase();
  return query.split(/\s+/).every(term => searchable.includes(term));
}
function refreshTranscriptVisibility() {
  Array.from(sentences.querySelectorAll(".row")).forEach(row => {
    row.hidden = !speakerTranscriptVisible(row.dataset.speaker) || !transcriptSearchVisible(row);
  });
}
function setSpeakerFilter(speakerId, mode, active) {
  if (!speakerId) return;
  const target = mode === "mute" ? mutedSpeakerIds : soloSpeakerIds;
  if (active) {
    target.add(speakerId);
  } else {
    target.delete(speakerId);
  }
  refreshTranscriptVisibility();
  renderSpeakerPanel();
}
function setTranscriptSettingsOpen(open) {
  transcriptSettingsPanel.hidden = !open;
  transcriptSettingsButton.setAttribute("aria-expanded", open ? "true" : "false");
}
function applyTranscriptDisplaySettings() {
  transcriptPanel.classList.toggle("hide-tags", !showTranscriptTags.checked);
  transcriptPanel.classList.toggle("hide-time", !showTranscriptTime.checked);
  transcriptPanel.classList.toggle("hide-speech-rate", !showTranscriptSpeechRate.checked);
  transcriptPanel.classList.toggle("hide-probabilities", !showTranscriptProbabilities.checked);
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
  pruneSpeakerFilterState();
  refreshTranscriptVisibility();
  recomputeRenderedSpeakerSentenceCounts();
  updateSpeakerCount();
  renderSpeakerPanel();
  refreshSpeakerRows();
}
function setSpeakerTab(tabName) {
  const nextTab = tabName === "settings" ? "settings" : "speakers";
  speakerTabButtons.forEach(button => {
    const active = button.dataset.speakerTab === nextTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  speakerTabPanels.forEach(panel => {
    panel.hidden = panel.dataset.speakerPanel !== nextTab;
  });
}
function selectedSpeaker() {
  return speakerLibraryState.speakers.find(speaker => speaker.id === editingSpeakerId) || null;
}
function recomputeRenderedSpeakerSentenceCounts() {
  const counts = {};
  const speakingSeconds = {};
  let hasFinalRows = false;
  Array.from(sentences.querySelectorAll(".row")).forEach(row => {
    if (row.dataset.realtime === "true") return;
    hasFinalRows = true;
    const speakerId = row.dataset.speaker;
    if (!speakerId || speakerId === "UNKNOWN") return;
    counts[speakerId] = (counts[speakerId] || 0) + 1;
    const start = Number(row.dataset.start || 0);
    const end = Number(row.dataset.end || 0);
    speakingSeconds[speakerId] = (speakingSeconds[speakerId] || 0) + Math.max(0, end - start);
  });
  renderedSpeakerSentenceCounts = counts;
  renderedSpeakerSpeakingSeconds = speakingSeconds;
  hasRenderedFinalSentenceRows = hasFinalRows;
}
function ensureSpeakerPanelSpeaker(speakerId) {
  if (!speakerId || speakerId === "UNKNOWN") return;
  if (speakerLibraryState.speakers.some(speaker => speaker.id === speakerId)) return;
  const speaker = {
    id: speakerId,
    name: "",
    display_name: speakerDisplayLabel(speakerId),
    source: "detected",
    locked: false,
    sentence_count: 0,
    speech_seconds: 0,
    reference_audio: "",
  };
  speakerLibraryState = {
    ...speakerLibraryState,
    speakers: [...speakerLibraryState.speakers, speaker],
  };
  speakerNames[speakerId] = speaker.display_name;
  pruneSpeakerFilterState();
  updateSpeakerCount();
  renderSpeakerPanel();
}
function speakerPanelSentenceCount(speaker) {
  const speakerId = speaker && speaker.id;
  if (hasRenderedFinalSentenceRows && speakerId) {
    return renderedSpeakerSentenceCounts[speakerId] || 0;
  }
  return Number((speaker && speaker.sentence_count) || 0);
}
function speakerPanelSpeakingSeconds(speaker) {
  const speakerId = speaker && speaker.id;
  if (hasRenderedFinalSentenceRows && speakerId) {
    return renderedSpeakerSpeakingSeconds[speakerId] || 0;
  }
  return Number((speaker && speaker.speech_seconds) || 0);
}
function speakerPanelCountUnit() {
  return "sentence";
}
function refreshSpeakerPanelSentenceCounts() {
  recomputeRenderedSpeakerSentenceCounts();
  Array.from(speakerList.querySelectorAll(".speaker-item")).forEach(row => {
    const speaker = speakerLibraryState.speakers.find(item => item.id === row.dataset.speakerId);
    const count = row.querySelector(".speaker-sentence-count");
    if (speaker && count) {
      count.textContent = speakerSentenceText(
        speakerPanelSentenceCount(speaker),
        speakerPanelSpeakingSeconds(speaker),
        speakerPanelCountUnit(),
      );
    }
  });
}
function refreshLiveSpeakerHighlight() {
  Array.from(speakerList.querySelectorAll(".speaker-item")).forEach(row => {
    const active = Boolean(currentLiveSpeakerId) && row.dataset.speakerId === currentLiveSpeakerId;
    row.classList.toggle("live-speaker", active);
    const indicator = row.querySelector(".speaker-live-indicator");
    if (active && !indicator) {
      const titleRow = row.querySelector(".speaker-title-row");
      if (titleRow) titleRow.appendChild(createSpeakerLiveIndicator());
    } else if (!active && indicator) {
      indicator.remove();
    }
  });
}
function clearFallbackLiveSpeaker() {
  fallbackLiveSpeakerId = "";
  fallbackLiveSpeakerUntilMs = 0;
  if (fallbackLiveSpeakerExpiryTimer) {
    clearTimeout(fallbackLiveSpeakerExpiryTimer);
    fallbackLiveSpeakerExpiryTimer = null;
  }
  if (fallbackLiveSpeakerClearTimer) {
    clearTimeout(fallbackLiveSpeakerClearTimer);
    fallbackLiveSpeakerClearTimer = null;
  }
}
function clearTranscriptLiveSpeakerExpiryTimer() {
  if (transcriptLiveSpeakerExpiryTimer) {
    clearTimeout(transcriptLiveSpeakerExpiryTimer);
    transcriptLiveSpeakerExpiryTimer = null;
  }
}
function clearLiveSpeakerState() {
  currentLiveSpeakerId = "";
  transcriptLiveSpeakerId = "";
  transcriptLiveSpeakerOverrideId = "";
  fastSpeakerPanelStats = {};
  fastSpeakerPanelLastRight = null;
  liveSpeakerTimeline = [];
  clearFallbackLiveSpeaker();
  clearTranscriptLiveSpeakerExpiryTimer();
  refreshRealtimeRowsFromLiveSpeaker();
}
function activeFallbackLiveSpeakerId(nowMs = performance.now()) {
  if (!fallbackLiveSpeakerId) return "";
  if (fallbackLiveSpeakerUntilMs > nowMs) return fallbackLiveSpeakerId;
  clearFallbackLiveSpeaker();
  return "";
}
function normalizedLiveSpeakerId(speakerId) {
  const value = String(speakerId || "").trim();
  return value && value !== "UNKNOWN" ? value : "";
}
function finiteAudioSecond(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
function pruneLiveSpeakerTimeline(minEndSeconds) {
  const cutoff = Math.max(0, finiteAudioSecond(minEndSeconds, 0));
  liveSpeakerTimeline = liveSpeakerTimeline.filter(item => finiteAudioSecond(item.end, 0) >= cutoff);
}
function rememberLiveSpeakerEvidence(speakerId, item) {
  const normalizedSpeakerId = normalizedLiveSpeakerId(speakerId);
  if (!normalizedSpeakerId || !item) return;
  const fallbackEnd = playbackSeconds();
  const end = finiteAudioSecond(item.end, fallbackEnd);
  const audioLength = Math.max(0, finiteAudioSecond(item.audio_length_seconds, 0));
  const fallbackStart = Math.max(0, end - audioLength);
  const start = finiteAudioSecond(item.start, fallbackStart);
  if (!(end > start)) return;
  liveSpeakerTimeline.push({speakerId: normalizedSpeakerId, start, end});
  pruneLiveSpeakerTimeline(end - 90);
}
function realtimeDominanceScoredEnd(start, end) {
  const duration = Math.max(0, end - start);
  if (duration <= 2) return end;
  const tailSeconds = duration >= 8 ? 2 : (duration - 2) * 0.25;
  return Math.max(start + 0.1, end - tailSeconds);
}
function dominantRealtimeSpeakerId(start, end) {
  const rowStart = Math.max(0, finiteAudioSecond(start, 0));
  const rowEnd = Math.max(rowStart, finiteAudioSecond(end, rowStart));
  if (!(rowEnd > rowStart)) return "";
  const scoredEnd = realtimeDominanceScoredEnd(rowStart, rowEnd);
  const weights = {};
  liveSpeakerTimeline.forEach(item => {
    const speakerId = normalizedLiveSpeakerId(item.speakerId);
    if (!speakerId) return;
    const overlapStart = Math.max(rowStart, finiteAudioSecond(item.start, rowStart));
    const overlapEnd = Math.min(scoredEnd, finiteAudioSecond(item.end, rowStart));
    const seconds = Math.max(0, overlapEnd - overlapStart);
    if (seconds <= 0) return;
    weights[speakerId] = (weights[speakerId] || 0) + seconds;
  });
  let bestSpeakerId = "";
  let bestSeconds = 0;
  Object.entries(weights).forEach(([speakerId, seconds]) => {
    if (seconds > bestSeconds) {
      bestSpeakerId = speakerId;
      bestSeconds = seconds;
    }
  });
  return bestSeconds >= 0.1 ? bestSpeakerId : "";
}
function realtimeRowDisplaySpeakerId(rawSpeakerId = "", start = 0, end = 0, previousSpeakerId = "") {
  return (
    dominantRealtimeSpeakerId(start, end)
    || normalizedLiveSpeakerId(previousSpeakerId)
    || activeFallbackLiveSpeakerId()
    || normalizedLiveSpeakerId(rawSpeakerId)
  );
}
function applyRealtimeRowSpeaker(row, speakerId) {
  const normalizedSpeakerId = normalizedLiveSpeakerId(speakerId);
  const color = speakerColor(normalizedSpeakerId);
  row.dataset.speaker = normalizedSpeakerId || "UNKNOWN";
  row.classList.toggle("live-speaker-row", Boolean(normalizedSpeakerId));
  row.style.setProperty("--live-row-color", color || "#8F9BA8");
  const badge = row.querySelector(".speaker-name");
  if (!badge) return;
  badge.className = `${normalizedSpeakerId ? "badge" : "badge unknown"} speaker-name`;
  badge.textContent = speakerDisplayLabel(normalizedSpeakerId);
  if (color) {
    badge.style.color = color;
    badge.style.borderColor = color;
    badge.style.background = "#0B1015";
  } else {
    badge.style.removeProperty("color");
    badge.style.removeProperty("border-color");
    badge.style.removeProperty("background");
  }
}
function refreshRealtimeRowsFromLiveSpeaker() {
  Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => {
    applyRealtimeRowSpeaker(
      row,
      realtimeRowDisplaySpeakerId(
        row.dataset.rawSpeaker || "",
        row.dataset.start,
        row.dataset.end,
        row.dataset.speaker,
      ),
    );
  });
  updateCurrentLiveSpeakerFromRealtimeRows();
  refreshSpeakerPanelSentenceCounts();
  refreshTranscriptVisibility();
}
function transcriptHighlightMaxLagSeconds() {
  const value = Number(liveSpeakerConfig.transcript_highlight_max_lag_seconds);
  return Number.isFinite(value) ? value : -1;
}
function realtimeRowWithinTranscriptHighlightWindow(row) {
  if (!row) return false;
  const maxLag = transcriptHighlightMaxLagSeconds();
  if (maxLag < 0) return true;
  const start = finiteAudioSecond(row.dataset.start, NaN);
  const end = finiteAudioSecond(row.dataset.end, NaN);
  const playback = playbackSeconds();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
  if (playback < start - 0.25) return false;
  if (playback > end + maxLag) return false;
  return true;
}
function probabilityForSpeakerId(probabilities, speakerId) {
  const key = speakerProbabilityKey(speakerId);
  if (!key || !probabilities || typeof probabilities !== "object") return 0;
  const value = Number(probabilities[key]);
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}
function probabilityLeadOverUnknown(probabilities, speakerId) {
  const speakerProbability = probabilityForSpeakerId(probabilities, speakerId);
  const unknownProbability = Number(probabilities && probabilities.unknown);
  return speakerProbability - (Number.isFinite(unknownProbability) ? Math.max(0, unknownProbability) : 0);
}
function transcriptOverrideCandidate(row) {
  if (!row || !liveSpeakerConfig.highlight_transcript) return "";
  if (!realtimeRowWithinTranscriptHighlightWindow(row)) return "";
  const minProbability = Number(liveSpeakerConfig.transcript_override_min_probability);
  if (!Number.isFinite(minProbability) || minProbability > 1) return "";
  const speakerId = normalizedLiveSpeakerId(row.dataset.rawSpeaker);
  if (!speakerId) return "";
  const probability = Number(row.dataset.rawSpeakerProbability || 0);
  if (!Number.isFinite(probability) || probability < minProbability) return "";
  const minMargin = Number(liveSpeakerConfig.transcript_override_min_margin || 0);
  const margin = Number(row.dataset.rawSpeakerUnknownMargin || 0);
  if (Number.isFinite(minMargin) && margin < minMargin) return "";
  return speakerId;
}
function realtimeRowTranscriptLiveSpeakerId(row) {
  if (!row || !liveSpeakerConfig.highlight_transcript) return "";
  const speakerId = row.dataset.speaker !== "UNKNOWN" ? normalizedLiveSpeakerId(row.dataset.speaker) : "";
  if (!speakerId) return "";
  return realtimeRowWithinTranscriptHighlightWindow(row) ? speakerId : "";
}
function scheduleTranscriptLiveSpeakerExpiry(row) {
  clearTranscriptLiveSpeakerExpiryTimer();
  const maxLag = transcriptHighlightMaxLagSeconds();
  if (!row || maxLag < 0 || !transcriptLiveSpeakerId) return;
  const end = finiteAudioSecond(row.dataset.end, NaN);
  if (!Number.isFinite(end)) return;
  const remainingMs = Math.max(0, (end + maxLag - playbackSeconds()) * 1000);
  transcriptLiveSpeakerExpiryTimer = setTimeout(updateCurrentLiveSpeakerFromRealtimeRows, remainingMs + 25);
}
function reconcileLiveSpeakerHighlight() {
  currentLiveSpeakerId = transcriptLiveSpeakerOverrideId
    || activeFallbackLiveSpeakerId()
    || (liveSpeakerConfig.highlight_transcript ? transcriptLiveSpeakerId : "");
  refreshLiveSpeakerHighlight();
}
function scheduleFallbackLiveSpeakerExpiry() {
  if (fallbackLiveSpeakerExpiryTimer) {
    clearTimeout(fallbackLiveSpeakerExpiryTimer);
    fallbackLiveSpeakerExpiryTimer = null;
  }
  const remainingMs = fallbackLiveSpeakerUntilMs - performance.now();
  if (!fallbackLiveSpeakerId || remainingMs <= 0) {
    refreshRealtimeRowsFromLiveSpeaker();
    return;
  }
  fallbackLiveSpeakerExpiryTimer = setTimeout(refreshRealtimeRowsFromLiveSpeaker, remainingMs + 25);
}
function applyFallbackLiveSpeaker(item) {
  const speakerId = normalizedLiveSpeakerId(item && (item.assigned_speaker || item.speaker_id));
  if (!speakerId) return;
  if (item.only_if_no_live_speaker && currentLiveSpeakerId) return;
  if (fallbackLiveSpeakerClearTimer) {
    clearTimeout(fallbackLiveSpeakerClearTimer);
    fallbackLiveSpeakerClearTimer = null;
  }
  applyFastSpeakerPanelSignal(item);
  rememberLiveSpeakerEvidence(speakerId, item);
  const holdSeconds = Math.max(0, Number(item.hold_seconds || 2.0));
  fallbackLiveSpeakerId = speakerId;
  fallbackLiveSpeakerUntilMs = performance.now() + holdSeconds * 1000;
  scheduleFallbackLiveSpeakerExpiry();
  refreshRealtimeRowsFromLiveSpeaker();
}
function clearFallbackLiveSpeakerFromProbe(item) {
  const speakerId = normalizedLiveSpeakerId(item && (item.assigned_speaker || item.speaker_id));
  if (speakerId && fallbackLiveSpeakerId && speakerId !== fallbackLiveSpeakerId) return;
  const debounceSeconds = Math.max(0, Number(liveSpeakerConfig.unknown_clear_debounce_seconds || 0));
  if (fallbackLiveSpeakerId && item && item.reason === "unknown" && debounceSeconds > 0) {
    const expectedSpeakerId = fallbackLiveSpeakerId;
    const debounceMs = debounceSeconds * 1000;
    if (fallbackLiveSpeakerClearTimer) clearTimeout(fallbackLiveSpeakerClearTimer);
    fallbackLiveSpeakerUntilMs = Math.max(fallbackLiveSpeakerUntilMs, performance.now() + debounceMs);
    fallbackLiveSpeakerClearTimer = setTimeout(() => {
      fallbackLiveSpeakerClearTimer = null;
      if (fallbackLiveSpeakerId === expectedSpeakerId) {
        clearFallbackLiveSpeaker();
        refreshRealtimeRowsFromLiveSpeaker();
      }
    }, debounceMs);
    scheduleFallbackLiveSpeakerExpiry();
    refreshRealtimeRowsFromLiveSpeaker();
    return;
  }
  clearFallbackLiveSpeaker();
  refreshRealtimeRowsFromLiveSpeaker();
}
function applyFastSpeakerPanelSignal(item) {
  const speakerId = item && (item.assigned_speaker || item.speaker_id);
  if (!speakerId || speakerId === "UNKNOWN") return;
  ensureSpeakerPanelSpeaker(speakerId);
  const start = Number(item.start || 0);
  const end = Number(item.end || start);
  if (!(end > start)) return;
  const previousRight = fastSpeakerPanelLastRight === null ? start : fastSpeakerPanelLastRight;
  const uncoveredStart = Math.max(start, previousRight);
  const seconds = Math.max(0, end - uncoveredStart);
  const current = fastSpeakerPanelStats[speakerId] || {count: 0, speakingSeconds: 0};
  fastSpeakerPanelStats = {
    ...fastSpeakerPanelStats,
    [speakerId]: {
      count: current.count + 1,
      speakingSeconds: current.speakingSeconds + seconds,
      lastStart: start,
      lastEnd: end,
    },
  };
  fastSpeakerPanelLastRight = Math.max(previousRight, end);
  refreshSpeakerPanelSentenceCounts();
}
function updateCurrentLiveSpeakerFromRealtimeRows() {
  const realtimeRows = Array.from(sentences.querySelectorAll(".row[data-realtime='true']"));
  const activeRow = realtimeRows[realtimeRows.length - 1] || null;
  transcriptLiveSpeakerId = realtimeRowTranscriptLiveSpeakerId(activeRow);
  transcriptLiveSpeakerOverrideId = transcriptOverrideCandidate(activeRow);
  scheduleTranscriptLiveSpeakerExpiry(activeRow);
  reconcileLiveSpeakerHighlight();
}
function speakerSpeakingTimeText(seconds) {
  const totalSeconds = Math.max(0, Number(seconds || 0));
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
  const roundedSeconds = Math.round(totalSeconds);
  const minutes = Math.floor(roundedSeconds / 60);
  const remainingSeconds = roundedSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}
function speakerSentenceText(count, speakingSeconds = 0, unit = "sentence") {
  const total = Number(count || 0);
  return `${total} ${unit}${total === 1 ? "" : "s"} · ${speakerSpeakingTimeText(speakingSeconds)}`;
}
function speakerReferenceText(speaker) {
  const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
  if (!hasReference) return "";
  const seconds = Number(speaker.speech_seconds || 0);
  return seconds > 0 ? `Reference voice added (${Math.round(seconds)}s)` : "Reference voice added";
}
function appendSvgElement(svg, tagName, attributes) {
  const element = document.createElementNS(svgNamespace, tagName);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  svg.appendChild(element);
}
function createSpeakerFilterIcon(mode) {
  const svg = document.createElementNS(svgNamespace, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  if (mode === "mute") {
    appendSvgElement(svg, "path", {d: "M11 5 6 9H3v6h3l5 4V5z"});
    appendSvgElement(svg, "path", {d: "m16 9 5 5"});
    appendSvgElement(svg, "path", {d: "m21 9-5 5"});
  } else {
    appendSvgElement(svg, "path", {d: "M4 14v-2a8 8 0 0 1 16 0v2"});
    appendSvgElement(svg, "path", {d: "M4 14a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2v-2z"});
    appendSvgElement(svg, "path", {d: "M20 14a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2v-2z"});
  }
  return svg;
}
function createSpeakerFilterToggle(speaker, mode) {
  const active = mode === "mute" ? mutedSpeakerIds.has(speaker.id) : soloSpeakerIds.has(speaker.id);
  const label = mode === "mute" ? "Mute" : "Solo";
  const button = document.createElement("button");
  button.type = "button";
  button.className = `speaker-filter-toggle ${mode}${active ? " active" : ""}`;
  button.setAttribute("aria-label", `${label} ${speakerPanelName(speaker)}`);
  button.setAttribute("aria-pressed", active ? "true" : "false");
  button.title = `${label} ${speakerPanelName(speaker)}`;
  button.addEventListener("click", event => {
    event.stopPropagation();
    setSpeakerFilter(speaker.id || "", mode, !active);
  });
  button.addEventListener("keydown", event => {
    event.stopPropagation();
  });
  const switchTrack = document.createElement("span");
  switchTrack.className = "speaker-filter-switch";
  switchTrack.setAttribute("aria-hidden", "true");
  button.appendChild(createSpeakerFilterIcon(mode));
  button.appendChild(switchTrack);
  return button;
}
function createTranscriptActionIcon(kind) {
  const svg = document.createElementNS(svgNamespace, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  if (kind === "download") {
    appendSvgElement(svg, "path", {d: "M12 3v12"});
    appendSvgElement(svg, "path", {d: "m7 10 5 5 5-5"});
    appendSvgElement(svg, "path", {d: "M5 21h14"});
  } else {
    appendSvgElement(svg, "rect", {x: "9", y: "9", width: "11", height: "11", rx: "2"});
    appendSvgElement(svg, "path", {d: "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"});
  }
  return svg;
}
function createTranscriptActionButton(kind, speaker) {
  const speakerName = speaker ? speakerPanelName(speaker) : "";
  const label = `${kind === "download" ? "Download" : "Copy"} ${speakerName ? `${speakerName} transcript` : "transcript"}`;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "transcript-icon-button speaker-transcript-action";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.appendChild(createTranscriptActionIcon(kind));
  button.addEventListener("click", event => {
    event.stopPropagation();
    if (kind === "download") {
      downloadTranscript(speaker ? speaker.id : null);
    } else {
      copyTranscript(speaker ? speaker.id : null);
    }
  });
  button.addEventListener("keydown", event => event.stopPropagation());
  return button;
}
function createSpeakerLiveIndicator() {
  const indicator = document.createElement("span");
  indicator.className = "speaker-live-indicator";
  const svg = document.createElementNS(svgNamespace, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("aria-hidden", "true");
  [["2", "10", "2", "14"], ["5", "7", "5", "14"], ["8", "4", "8", "14"], ["11", "8", "11", "14"], ["14", "11", "14", "14"]].forEach(([x1, y1, x2, y2]) => {
    appendSvgElement(svg, "path", {d: `M${x1} ${y1}v${Number(y2) - Number(y1)}`});
  });
  indicator.appendChild(svg);
  indicator.appendChild(document.createTextNode("Live"));
  return indicator;
}
function isSpeakerRowControl(target) {
  return target instanceof Element && target.closest(".speaker-row-name-input, .speaker-filter-toggle, .speaker-transcript-action");
}
function setEditingSpeaker(speakerId, options = {}) {
  const requestedId = speakerId || "";
  const collapse = requestedId && editingSpeakerId === requestedId && !options.keepOpen;
  editingSpeakerId = collapse ? "" : requestedId;
  pendingSpeakerNameFocusId = editingSpeakerId && options.focusName !== false ? editingSpeakerId : "";
  if (editingSpeakerId) {
    manualSpeakerComposerOpen = false;
    pendingManualSpeakerNameFocus = false;
  }
  referenceRecordSeconds.textContent = "0.0s";
  referenceSpeakerFile.value = "";
  renderSpeakerPanel();
}
function syncSpeakerEditor(speaker) {
  if (!speaker) {
    referenceSpeakerForm.hidden = true;
    speakerEditorDock.appendChild(referenceSpeakerForm);
    return;
  }
  referenceSpeakerForm.hidden = false;
}
function syncManualSpeakerComposer() {
  manualSpeakerComposer.hidden = !manualSpeakerComposerOpen;
  addReferenceSpeakerButton.setAttribute("aria-expanded", manualSpeakerComposerOpen ? "true" : "false");
  if (!manualSpeakerComposerOpen) return;
  editingSpeakerId = "";
  referenceSpeakerForm.hidden = false;
  manualSpeakerReferenceDock.appendChild(referenceSpeakerForm);
  if (pendingManualSpeakerNameFocus) {
    pendingManualSpeakerNameFocus = false;
    requestAnimationFrame(() => {
      manualSpeakerName.focus();
      manualSpeakerName.select();
    });
  }
}
function referenceNameMissingMessage() {
  return manualSpeakerComposerOpen ? "Enter a speaker name first." : "Choose a speaker first.";
}
function closeManualSpeakerComposerAfterReference() {
  if (!manualSpeakerComposerOpen) return;
  manualSpeakerComposerOpen = false;
  pendingManualSpeakerNameFocus = false;
  manualSpeakerName.value = "";
}
function selectedSpeakerReferenceName() {
  if (manualSpeakerComposerOpen) {
    return manualSpeakerName.value.trim();
  }
  const inlineName = speakerList.querySelector(".speaker-item.editing .speaker-row-name-input");
  const inlineValue = inlineName ? inlineName.value.trim() : "";
  if (inlineValue) return inlineValue;
  const speaker = selectedSpeaker();
  return speaker ? speakerPanelName(speaker).trim() : "";
}
async function commitSpeakerNameInput(speaker, input) {
  if (!speaker || !input || input.dataset.saving === "1") return;
  try {
    await ensureSessionOwner("rename speakers");
  } catch (error) {
    log(error.message);
    return;
  }
  const currentName = speakerPanelName(speaker);
  const name = input.value.trim();
  if (!name) {
    input.value = currentName;
    log("Enter a speaker name first.");
    return;
  }
  if (name === currentName) {
    input.value = currentName;
    return;
  }
  input.dataset.saving = "1";
  input.disabled = true;
  try {
    const result = await post("/api/speakers/rename", {speaker_id: speaker.id, name});
    updateSpeakerState(result.speaker_state);
  } catch (error) {
    input.disabled = false;
    input.dataset.saving = "";
    input.value = currentName;
    log(`Rename failed: ${error.message}`);
  }
}
function speakerGroupFileName(name) {
  const safe = String(name || "speakers")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._ -]+/g, "")
    .replace(/\s+/g, "-")
    .replace(/^[._ -]+|[._ -]+$/g, "")
    .slice(0, 80) || "speakers";
  return `${safe}.whospeaks-speakers.json`;
}
function downloadJsonFile(filename, payload) {
  const text = JSON.stringify(payload, null, 2);
  const url = URL.createObjectURL(new Blob([text], {type: "application/json;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
function renderSpeakerPanel() {
  const controlsLocked = sessionControlsLocked();
  speakerPanelTitle.textContent = `Detected speakers (${speakerLibraryState.speakers.length})`;
  clearSpeakersButton.disabled = controlsLocked || !speakerLibraryState.speakers.length;
  addReferenceSpeakerButton.disabled = controlsLocked;
  manualSpeakerName.disabled = controlsLocked;
  syncManualSpeakerComposer();
  speakerList.textContent = "";
  if (!speakerLibraryState.speakers.length) {
    const empty = document.createElement("div");
    empty.className = "speaker-empty";
    empty.textContent = "No speakers yet";
    speakerList.appendChild(empty);
    if (!manualSpeakerComposerOpen) {
      syncSpeakerEditor(null);
    }
    return;
  }
  const speakerIds = speakerLibraryState.speakers.map(speaker => speaker.id).filter(Boolean);
  if (editingSpeakerId && !speakerIds.includes(editingSpeakerId)) {
    editingSpeakerId = "";
  }
  speakerLibraryState.speakers.forEach(speaker => {
    const isEditing = speaker.id === editingSpeakerId;
    const hasReference = Boolean(speaker.reference_audio || speaker.locked || speaker.source === "reference");
    const row = document.createElement("div");
    row.className = `speaker-item${isEditing ? " editing" : ""}`;
    row.classList.toggle("live-speaker", Boolean(currentLiveSpeakerId) && speaker.id === currentLiveSpeakerId);
    row.dataset.speakerId = speaker.id || "";
    const color = speakerColor(speaker.id);
    row.style.setProperty("--speaker-color", color || "transparent");
    const summary = document.createElement("div");
    summary.className = "speaker-item-summary";
    summary.setAttribute("role", "button");
    summary.tabIndex = controlsLocked ? -1 : 0;
    summary.setAttribute("aria-expanded", isEditing ? "true" : "false");
    summary.addEventListener("click", event => {
      if (isSpeakerRowControl(event.target)) return;
      if (controlsLocked) return;
      setEditingSpeaker(speaker.id || "");
    });
    summary.addEventListener("keydown", event => {
      if (isSpeakerRowControl(event.target)) return;
      if (controlsLocked) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setEditingSpeaker(speaker.id || "");
      }
    });

    const body = document.createElement("span");
    body.className = "speaker-summary-body";
    let title;
    if (isEditing) {
      title = document.createElement("input");
      title.className = "speaker-row-name-input";
      title.type = "text";
      title.value = speakerPanelName(speaker);
      title.disabled = controlsLocked;
      title.setAttribute("aria-label", "Speaker name");
      title.setAttribute("autocomplete", "off");
      title.addEventListener("click", event => event.stopPropagation());
      title.addEventListener("keydown", event => {
        if (event.key === "Enter") {
          event.preventDefault();
          title.blur();
        }
      });
      title.addEventListener("blur", () => {
        commitSpeakerNameInput(speaker, title);
      });
    } else {
      title = document.createElement("span");
      title.className = "speaker-row-title";
      title.textContent = speakerPanelName(speaker);
      title.addEventListener("click", event => {
        event.stopPropagation();
        if (controlsLocked) return;
        setEditingSpeaker(speaker.id || "", {focusName: true});
      });
    }
    const titleRow = document.createElement("span");
    titleRow.className = "speaker-title-row";
    titleRow.appendChild(title);
    if (currentLiveSpeakerId && speaker.id === currentLiveSpeakerId) {
      titleRow.appendChild(createSpeakerLiveIndicator());
    }
    const sentenceCount = document.createElement("span");
    sentenceCount.className = "speaker-sentence-count";
    sentenceCount.textContent = speakerSentenceText(
      speakerPanelSentenceCount(speaker),
      speakerPanelSpeakingSeconds(speaker),
      speakerPanelCountUnit(),
    );
    body.appendChild(titleRow);
    body.appendChild(sentenceCount);
    if (hasReference) {
      const referenceStatus = document.createElement("span");
      referenceStatus.className = "speaker-reference-status has-reference";
      const referenceIcon = document.createElement("span");
      referenceIcon.className = "speaker-reference-icon";
      referenceIcon.setAttribute("aria-hidden", "true");
      const referenceText = document.createElement("span");
      referenceText.textContent = speakerReferenceText(speaker);
      referenceStatus.appendChild(referenceIcon);
      referenceStatus.appendChild(referenceText);
      body.appendChild(referenceStatus);
    }

    const tail = document.createElement("span");
    tail.className = "speaker-item-tail";
    const filterControls = document.createElement("span");
    filterControls.className = "speaker-filter-controls";
    filterControls.appendChild(createSpeakerFilterToggle(speaker, "solo"));
    filterControls.appendChild(createSpeakerFilterToggle(speaker, "mute"));
    tail.appendChild(filterControls);
    const transcriptActions = document.createElement("span");
    transcriptActions.className = "speaker-transcript-actions";
    transcriptActions.appendChild(createTranscriptActionButton("copy", speaker));
    transcriptActions.appendChild(createTranscriptActionButton("download", speaker));
    tail.appendChild(transcriptActions);
    const chevron = document.createElement("span");
    chevron.className = "speaker-chevron";
    chevron.setAttribute("aria-hidden", "true");
    tail.appendChild(chevron);

    summary.appendChild(body);
    summary.appendChild(tail);
    row.appendChild(summary);
    if (isEditing) {
      syncSpeakerEditor(speaker);
      row.appendChild(referenceSpeakerForm);
      if (pendingSpeakerNameFocusId === speaker.id && title instanceof HTMLInputElement) {
        pendingSpeakerNameFocusId = "";
        requestAnimationFrame(() => {
          title.focus();
          title.select();
        });
      }
    }
    speakerList.appendChild(row);
  });
  if (!editingSpeakerId && !manualSpeakerComposerOpen) {
    syncSpeakerEditor(null);
  }
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
  const controlsLocked = sessionControlsLocked();
  recordReferenceButton.disabled = controlsLocked;
  referenceSpeakerFile.disabled = controlsLocked || recording;
  recordReferenceButton.classList.toggle("recording", recording);
  if (recordReferenceButtonLabel) {
    recordReferenceButtonLabel.textContent = recording ? "Stop and add" : "Record from mic";
  }
  recordReferenceButton.setAttribute("aria-label", recording ? "Stop and add reference recording" : "Record reference from microphone");
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
  const name = selectedSpeakerReferenceName();
  if (!name) {
    log(referenceNameMissingMessage());
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
  await ensureSessionOwner("add reference speakers");
  const name = selectedSpeakerReferenceName();
  const recording = stopReferenceRecording();
  if (!name) {
    log(referenceNameMissingMessage());
    return;
  }
  if (recording.seconds < 0.5) {
    log("Reference clip is too short.");
    return;
  }
  recordReferenceButton.disabled = true;
  try {
    const audio_b64 = encodeWavDataUrl(recording.samples, recording.sampleRate);
    const result = await post("/api/speakers/reference", {name, filename: `${name}.wav`, audio_b64});
    closeManualSpeakerComposerAfterReference();
    updateSpeakerState(result.speaker_state);
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
function transcriptTimeLabel(value) {
  const total = Math.max(0, Number(value || 0));
  const minutes = Math.floor(total / 60);
  const seconds = total - (minutes * 60);
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`;
}
function transcriptExportRows(speakerId = null) {
  return Array.from(sentences.querySelectorAll(".row"))
    .filter(row => row.dataset.realtime !== "true")
    .filter(row => !speakerId || row.dataset.speaker === speakerId)
    .map(row => ({
      speaker: speakerDisplayLabel(row.dataset.speaker === "UNKNOWN" ? null : row.dataset.speaker),
      start: transcriptTimeLabel(row.dataset.start),
      end: transcriptTimeLabel(row.dataset.end),
      text: row.dataset.text || "",
    }))
    .filter(row => row.text.trim());
}
function transcriptExportText(speakerId = null) {
  const rows = transcriptExportRows(speakerId);
  return rows.map(row => `[${row.start} - ${row.end}] ${row.speaker}: ${row.text}`).join("\n");
}
function transcriptExportFilename(speakerId = null) {
  const suffix = speakerId ? `-${speakerDisplayLabel(speakerId).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}` : "";
  return `whospeaks-transcript${suffix || ""}.txt`;
}
async function copyTextToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.focus();
  area.select();
  document.execCommand("copy");
  area.remove();
}
function copyTranscript(speakerId = null) {
  const text = transcriptExportText(speakerId);
  if (!text) {
    log("No transcript text to copy.");
    return;
  }
  copyTextToClipboard(text)
    .then(() => log(speakerId ? `Copied ${speakerDisplayLabel(speakerId)} transcript.` : "Copied transcript."))
    .catch(error => log(`Copy failed: ${error.message}`));
}
function downloadTranscript(speakerId = null) {
  const text = transcriptExportText(speakerId);
  if (!text) {
    log("No transcript text to download.");
    return;
  }
  const url = URL.createObjectURL(new Blob([text], {type: "text/plain;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = transcriptExportFilename(speakerId);
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
function findSentenceRow(index) {
  const key = String(index);
  return Array.from(sentences.querySelectorAll(".row")).find(row => row.dataset.index === key) || null;
}
function clearRealtimeRows(generation) {
  currentRealtimeGeneration = Math.max(currentRealtimeGeneration, Number(generation || 0));
  Array.from(sentences.querySelectorAll(".row[data-realtime='true']")).forEach(row => row.remove());
  updateCurrentLiveSpeakerFromRealtimeRows();
  refreshSpeakerPanelSentenceCounts();
  refreshTranscriptVisibility();
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
  const startSeconds = Number(item.start || 0);
  const endSeconds = Number(item.end || 0);
  const durationSeconds = Math.max(0, endSeconds - startSeconds);
  const ratio = Number(item.speech_audio_ratio);
  const rawSpeakerId = normalizedLiveSpeakerId(item.assigned_speaker);
  const rawProbabilities = item.probabilities || {};
  const rawSpeakerProbability = probabilityForSpeakerId(rawProbabilities, rawSpeakerId);
  const rawSpeakerUnknownMargin = probabilityLeadOverUnknown(rawProbabilities, rawSpeakerId);
  const previousDisplaySpeakerId = item.realtime ? normalizedLiveSpeakerId(row.dataset.speaker) : "";
  const displaySpeakerId = item.realtime
    ? realtimeRowDisplaySpeakerId(rawSpeakerId, startSeconds, endSeconds, previousDisplaySpeakerId)
    : rawSpeakerId;
  row.dataset.rawSpeaker = item.realtime ? rawSpeakerId : "";
  row.dataset.rawSpeakerProbability = item.realtime ? String(rawSpeakerProbability) : "";
  row.dataset.rawSpeakerUnknownMargin = item.realtime ? String(rawSpeakerUnknownMargin) : "";
  row.dataset.speaker = displaySpeakerId || "UNKNOWN";
  row.classList.toggle("live-speaker-row", item.realtime && Boolean(displaySpeakerId));
  const speakerLabel = speakerDisplayLabel(displaySpeakerId);
  const color = speakerColor(displaySpeakerId);
  const speakerClass = displaySpeakerId ? "badge" : "badge unknown";
  const stateLabel = item.realtime ? "Live" : (item.pending ? "Embedding" : (item.error ? "Error" : (item.revision ? "Revised" : "")));
  row.dataset.start = String(startSeconds);
  row.dataset.end = String(endSeconds);
  row.dataset.text = item.text || "";
  row.dataset.searchText = item.text || "";
  if (item.realtime) {
    row.style.setProperty("--live-row-color", color || "#8F9BA8");
  } else {
    row.style.removeProperty("--live-row-color");
  }

  const top = document.createElement("div");
  top.className = "top";
  const topLeft = document.createElement("div");
  topLeft.className = "top-left";

  const speakerBadge = document.createElement("span");
  speakerBadge.className = `${speakerClass} speaker-name`;
  if (color) {
    speakerBadge.style.color = color;
    speakerBadge.style.borderColor = color;
    speakerBadge.style.background = "#0B1015";
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
  duration.className = "sentence-duration";
  duration.textContent = secondsLabel(durationSeconds);
  topLeft.appendChild(duration);

  const range = document.createElement("span");
  range.className = "sentence-range";
  range.textContent = `(${secondsLabel(startSeconds)} - ${secondsLabel(endSeconds)})`;
  topLeft.appendChild(range);

  if (Number.isFinite(ratio)) {
    const ratioSpan = document.createElement("span");
    ratioSpan.className = "sentence-speech-rate";
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
  updateCurrentLiveSpeakerFromRealtimeRows();
  refreshSpeakerPanelSentenceCounts();
  refreshTranscriptVisibility();
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
  es.addEventListener("live_speaker", e => applyFallbackLiveSpeaker(JSON.parse(e.data)));
  es.addEventListener("live_speaker_clear", e => clearFallbackLiveSpeakerFromProbe(JSON.parse(e.data)));
  es.addEventListener("realtime_clear", e => clearRealtimeRows(JSON.parse(e.data).generation));
  es.addEventListener("done", e => {
    stopPlaybackClock();
    stopBrowserAudioCapture();
    void stopBrowserLiveObservation("done");
    setState("Stopped");
    start.disabled = false;
    stop.disabled = true;
    setSourceControlsDisabled(false);
    log(JSON.parse(e.data).message);
    scheduleCompletedSessionRelease();
  });
}
followLive.addEventListener("change", () => {
  followLiveEnabled = followLive.checked;
  if (followLiveEnabled) scrollSentencesToBottom();
});
transcriptSearch.addEventListener("input", () => {
  transcriptSearchText = transcriptSearch.value || "";
  refreshTranscriptVisibility();
});
copyTranscriptButton.addEventListener("click", () => copyTranscript());
downloadTranscriptButton.addEventListener("click", () => downloadTranscript());
transcriptSettingsButton.addEventListener("click", event => {
  event.stopPropagation();
  setTranscriptSettingsOpen(transcriptSettingsPanel.hidden);
});
transcriptSettingsPanel.addEventListener("click", event => event.stopPropagation());
[showTranscriptTags, showTranscriptTime, showTranscriptSpeechRate, showTranscriptProbabilities].forEach(control => {
  control.addEventListener("change", applyTranscriptDisplaySettings);
});
releaseSessionButton.addEventListener("click", () => {
  releaseSession("released").catch(error => log(`Release failed: ${error.message}`));
});
applyTranscriptDisplaySettings();
start.addEventListener("click", async () => {
  let playbackUnlockResults = null;
  if (!browserStreamMode) {
    playbackUnlockResults = await unlockPlayback();
  }
  try {
    await ensureSessionOwner("start a run");
  } catch (error) {
    log(error.message);
    setState("Ready");
    return;
  }
  start.disabled = true; stop.disabled = false; setSourceControlsDisabled(true); resetTranscriptDisplay(); setState("Starting"); connect();
  if (browserStreamMode) {
    try {
      await applySpeakerSensitivityIfDirty();
      setState(captureSourceKind === "microphone" ? "Requesting mic" : "Requesting audio");
      await prepareBrowserStreamSession();
      await startBrowserAudioCapture();
      setState("Warming backend");
      setStreamHint(captureSourceKind === "microphone" ? "Microphone capture is armed; warming backend." : "Audio capture is armed; warming backend before transcription starts.");
      const result = await post("/api/start");
      if (result.speaker_state) updateSpeakerState(result.speaker_state);
    } catch (error) {
      stopBrowserAudioCapture();
      start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
      return;
    }
    setState("Capturing");
    return;
  }
  logRejectedPlayback(playbackUnlockResults || []);
  try {
    await applySpeakerSensitivityIfDirty();
    setState("Warming backend");
    log("Warming backend before playback starts. First Modal starts can take about two minutes.");
    const result = await post("/api/start");
    if (result.speaker_state) updateSpeakerState(result.speaker_state);
  } catch (error) {
    start.disabled = false; stop.disabled = true; setSourceControlsDisabled(false); setState("Ready"); log(`Start failed: ${error.message}`);
    return;
  }
  setState("Starting playback");
  video.currentTime = 0; audio.currentTime = 0; video.muted = true; audio.volume = 1.0;
  logRejectedPlayback(await Promise.allSettled([video.play(), audio.play()]));
  startPlaybackClock();
  startBrowserLiveObservation();
  setState("Playing");
});
stop.addEventListener("click", async () => {
  if (!sessionOwnerActive()) {
    log("Only the active demo seat can stop the shared run.");
    return;
  }
  stop.disabled = true; start.disabled = false; setSourceControlsDisabled(false); setState("Stopping"); stopPlaybackClock(); stopBrowserAudioCapture(); video.pause(); audio.pause(); await stopBrowserLiveObservation("stop"); await post("/api/stop");
});
preset.addEventListener("change", () => {
  if (preset.value) {
    source.value = preset.value;
    if (inputMode.value === "system") {
      setBrowserStreamMode(true, source.value.trim(), "display");
    }
  }
  updateMediaMode();
});
source.addEventListener("input", () => {
  syncPresetSelection(source.value);
  if (inputMode.value === "system") {
    setBrowserStreamMode(true, source.value.trim(), "display");
  }
  updateMediaMode();
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
  updateMediaMode();
});
sourceModeButton.addEventListener("click", event => {
  event.stopPropagation();
  setSourceModeMenuOpen(sourceModeOptions.hidden);
});
sourceModeOptionButtons.forEach(button => {
  button.addEventListener("click", event => {
    event.stopPropagation();
    const nextMode = button.dataset.inputMode || "youtube";
    setSourceModeMenuOpen(false);
    if (inputMode.value !== nextMode) {
      inputMode.value = nextMode;
      inputMode.dispatchEvent(new Event("change", {bubbles:true}));
    }
  });
});
document.addEventListener("click", event => {
  if (!sourceModeMenu.contains(event.target)) {
    setSourceModeMenuOpen(false);
  }
  if (!transcriptSettingsButton.contains(event.target) && !transcriptSettingsPanel.contains(event.target)) {
    setTranscriptSettingsOpen(false);
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
allowSpeakerReassignment.addEventListener("change", () => {
  applySpeakerReassignmentSetting()
    .then(result => {
      const enabled = Boolean((result.speaker_refinement || {}).allow_reassignment);
      log(enabled ? "Later speaker reassignment enabled." : "Later speaker reassignment disabled.");
    })
    .catch(() => {});
});
speakerTabButtons.forEach(button => {
  button.addEventListener("click", () => setSpeakerTab(button.dataset.speakerTab));
});
addReferenceSpeakerButton.addEventListener("click", () => {
  manualSpeakerComposerOpen = !manualSpeakerComposerOpen;
  if (manualSpeakerComposerOpen) {
    editingSpeakerId = "";
    pendingSpeakerNameFocusId = "";
    pendingManualSpeakerNameFocus = true;
    referenceRecordSeconds.textContent = "0.0s";
    referenceSpeakerFile.value = "";
  }
  renderSpeakerPanel();
});
clearSpeakersButton.addEventListener("click", async () => {
  if (!speakerLibraryState.speakers.length) return;
  try {
    await ensureSessionOwner("clear speakers");
  } catch (error) {
    log(error.message);
    return;
  }
  clearSpeakersButton.disabled = true;
  try {
    const result = await post("/api/speakers/clear", {});
    editingSpeakerId = "";
    pendingSpeakerNameFocusId = "";
    manualSpeakerComposerOpen = false;
    pendingManualSpeakerNameFocus = false;
    manualSpeakerName.value = "";
    soloSpeakerIds.clear();
    mutedSpeakerIds.clear();
    clearLiveSpeakerState();
    resetTranscriptDisplay();
    updateSpeakerState(result.speaker_state);
    log("Cleared speakers.");
  } catch (error) {
    log(`Clear speakers failed: ${error.message}`);
  } finally {
    renderSpeakerPanel();
  }
});
manualSpeakerName.addEventListener("keydown", event => {
  if (event.key === "Enter") {
    event.preventDefault();
  }
});
saveSpeakerGroupButton.addEventListener("click", async () => {
  try {
    await ensureSessionOwner("save speakers");
  } catch (error) {
    log(error.message);
    return;
  }
  const name = speakerLibraryState.group_name || "speakers";
  saveSpeakerGroupButton.disabled = true;
  try {
    const result = await post("/api/speakers/export", {name});
    const group = result.group || {};
    downloadJsonFile(speakerGroupFileName(group.name || name), group);
    updateSpeakerState(result.speaker_state);
    log(`Saved speaker group ${group.name || name} to a local file.`);
  } catch (error) {
    log(`Save speakers failed: ${error.message}`);
  } finally {
    saveSpeakerGroupButton.disabled = false;
  }
});
loadSpeakerGroupButton.addEventListener("click", () => {
  if (sessionControlsLocked()) {
    log("Session in use. Watching live until the seat is free.");
    return;
  }
  speakerGroupFile.value = "";
  speakerGroupFile.click();
});
speakerGroupFile.addEventListener("change", async () => {
  const file = speakerGroupFile.files && speakerGroupFile.files[0];
  if (!file) return;
  try {
    await ensureSessionOwner("load speakers");
  } catch (error) {
    log(error.message);
    speakerGroupFile.value = "";
    return;
  }
  loadSpeakerGroupButton.disabled = true;
  try {
    const group = JSON.parse(await file.text());
    const result = await post("/api/speakers/import", {group});
    updateSpeakerState(result.speaker_state);
    log(`Loaded speaker group ${result.speaker_state.group_name}.`);
  } catch (error) {
    log(`Load speakers failed: ${error.message}`);
  } finally {
    loadSpeakerGroupButton.disabled = false;
    speakerGroupFile.value = "";
  }
});
referenceSpeakerFile.addEventListener("change", () => {
  if (referenceSpeakerFile.files && referenceSpeakerFile.files[0]) {
    referenceSpeakerForm.requestSubmit();
  }
});
referenceSpeakerForm.addEventListener("submit", async event => {
  event.preventDefault();
  try {
    await ensureSessionOwner("add reference speakers");
  } catch (error) {
    log(error.message);
    return;
  }
  if (referenceRecordStream || referenceRecordPending) {
    log("Stop the reference recording first.");
    return;
  }
  const name = selectedSpeakerReferenceName();
  const file = referenceSpeakerFile.files && referenceSpeakerFile.files[0];
  if (!name || !file) {
    log(name ? "Choose a reference audio file first." : referenceNameMissingMessage());
    return;
  }
  const submit = referenceSpeakerForm.querySelector("button[type='submit']");
  if (submit) submit.disabled = true;
  referenceSpeakerFile.disabled = true;
  try {
    const audio_b64 = await fileToBase64(file);
    const result = await post("/api/speakers/reference", {name, filename: file.name, audio_b64});
    closeManualSpeakerComposerAfterReference();
    updateSpeakerState(result.speaker_state);
    referenceSpeakerFile.value = "";
    log(`Added reference speaker ${name}.`);
  } catch (error) {
    log(`Add reference failed: ${error.message}`);
  } finally {
    if (submit) submit.disabled = false;
    referenceSpeakerFile.disabled = false;
  }
});
recordReferenceButton.addEventListener("click", async () => {
  if (sessionControlsLocked()) {
    log("Session in use. Watching live until the seat is free.");
    return;
  }
  if (referenceRecordStream || referenceRecordPending) {
    stopAndAddReferenceRecording().catch(error => log(`Add recorded reference failed: ${error.message}`));
    return;
  }
  try {
    await ensureSessionOwner("record reference speakers");
  } catch (error) {
    log(error.message);
    return;
  }
  startReferenceRecording().catch(error => log(`Reference recording failed: ${error.message}`));
});
load.addEventListener("click", async () => {
  setSourceModeMenuOpen(false);
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
    updateMediaMode();
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
    log("Computer/tab audio mode ready. Press Start and share audio from a tab or window.");
    setState("Ready");
    start.disabled = false;
    stop.disabled = true;
    updateMediaMode();
    return;
  }
  if (!url) {
    log("Enter a YouTube URL first.");
    return;
  }
  try {
    await ensureSessionOwner("load media");
  } catch (error) {
    log(error.message);
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
    if (media.speaker_state) updateSpeakerState(media.speaker_state);
    source.value = media.url;
    syncPresetSelection(media.url);
    refreshMediaElements(media.version);
    updateMediaMode();
    log(`Loaded ${media.video_id}.`);
    setState("Ready");
    start.disabled = false;
  } catch (error) {
    log(`Load failed: ${error.message}`);
    try {
      const fallback = await post("/api/browser-stream", {url});
      if (fallback.speaker_state) updateSpeakerState(fallback.speaker_state);
      source.value = fallback.url;
      syncPresetSelection(fallback.url);
      setBrowserStreamMode(true, fallback.url, "display");
      updateMediaMode();
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
["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended"].forEach(eventName => {
  video.addEventListener(eventName, updateMediaTimeline);
  audio.addEventListener(eventName, updateMediaTimeline);
});
setInterval(updateMediaTimeline, 250);
expandMedia.addEventListener("click", () => {
  const youtubeStream = youtubeFrame.parentElement;
  const target = browserStreamMode && youtubeStream && !youtubeStream.classList.contains("empty") ? youtubeStream : video;
  const request = target.requestFullscreen || target.webkitRequestFullscreen || target.msRequestFullscreen;
  if (request) request.call(target);
});
micGain.addEventListener("input", updateMicGainLabel);
audio.addEventListener("ended", flushPlaybackEnd);
window.addEventListener("pagehide", () => sendSessionReleaseBeacon("tab closed"));
window.addEventListener("beforeunload", () => sendSessionReleaseBeacon("tab closed"));
updateSessionBanner();
fetchSessionStatus().catch(() => {});
startSessionStatusPolling();
</script>
</body>
</html>
"""
