"""Static HTML for the realtime speaker diarization browser UI."""

from __future__ import annotations

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealtimeSTT Speaker Diarization</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07090d;
      --panel: #101620;
      --panel-2: #151b25;
      --panel-3: #0d131c;
      --line: #263242;
      --text: #f4f7fb;
      --muted: #93a0b4;
      --green: #39e58e;
      --cyan: #37d5f7;
      --blue: #65a8ff;
      --yellow: #f2d35f;
      --pink: #ff79a6;
      --red: #ff5b70;
      --orange: #f59a43;
      --violet: #b48cff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }

    .app {
      height: 100vh;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(240px, 36vh) minmax(0, 1fr);
      gap: 14px;
      padding: 18px;
      overflow: hidden;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }

    input, button {
      border: 1px solid var(--line);
      border-radius: 7px;
      font: inherit;
      min-height: 44px;
    }

    input {
      width: 100%;
      background: #0b1018;
      color: var(--text);
      padding: 0 14px;
      outline: none;
    }

    input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 2px rgba(55, 213, 247, 0.16);
    }

    button {
      background: var(--green);
      color: #03120a;
      padding: 0 18px;
      font-weight: 400;
      cursor: pointer;
    }

    button.secondary {
      background: #1c2636;
      color: var(--text);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .main {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(260px, 340px);
      gap: 14px;
      min-height: 0;
    }

    .video, .status, .text-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .video {
      background: #000;
      min-height: 280px;
      position: relative;
    }

    #player, .placeholder {
      width: 100%;
      height: 100%;
      min-height: 280px;
    }

    .placeholder {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 18px;
    }

    .status, .text-panel {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      background: var(--panel);
    }

    .text-panel {
      background: var(--panel-2);
    }

    h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--cyan);
      font-size: 14px;
      font-weight: 400;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    #statusLog {
      margin: 0;
      padding: 12px 14px;
      color: #d5e0f5;
      overflow: auto;
      white-space: pre-wrap;
      font: 13px/1.45 Consolas, "Cascadia Mono", monospace;
    }

    .detected-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      min-height: 0;
    }

    #sentenceList {
      padding: 12px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: var(--panel-3);
    }

    .sentence-item {
      border: 1px solid #2b3849;
      border-radius: 8px;
      padding: 10px 11px;
      background: #111923;
    }

    .sentence-item.pending {
      opacity: 0.72;
    }

    .sentence-top {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      margin-bottom: 7px;
    }

    .speaker-badge {
      flex: 0 0 auto;
      min-width: 42px;
      padding: 3px 7px;
      border-radius: 6px;
      background: #233146;
      color: #fff;
      text-align: center;
      font-size: 12px;
      line-height: 1.35;
    }

    .speaker-new {
      background: #6b4d12;
      color: #fff2c0;
    }

    .speaker-unknown {
      background: #343a46;
      color: #d8dfec;
    }

    .sentence-meta {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .sentence-text {
      color: #f3f6fb;
      font-size: 15px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .probabilities {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .prob-label {
      border: 1px solid #354256;
      border-radius: 6px;
      color: #dce5f5;
      background: #0d1420;
      padding: 2px 6px;
      font-size: 12px;
      line-height: 1.35;
    }

    .prob-label.unknown {
      color: #f3d18c;
      border-color: #654b20;
      background: #21180a;
    }

    #transcript {
      padding: 18px;
      overflow: auto;
      font-size: 24px;
      line-height: 1.35;
      white-space: pre-wrap;
      background: var(--panel-2);
    }

    .final-speaker-1 { color: var(--blue); }
    .final-speaker-2 { color: var(--yellow); }
    .final-speaker-3 { color: var(--pink); }
    .final-speaker-4 { color: var(--green); }
    .final-speaker-5 { color: var(--orange); }
    .final-speaker-6 { color: var(--violet); }
    .final-unknown { color: #a9b3c4; }
    .stable { color: #ffffff; }
    .realtime { color: #8a93a5; }
    .error { color: var(--red); }

    @media (max-width: 900px) {
      body {
        height: auto;
        min-height: 100vh;
        overflow: auto;
      }

      .app {
        height: auto;
        min-height: 100vh;
        overflow: visible;
        grid-template-rows: auto minmax(360px, auto) minmax(520px, 1fr);
      }

      .toolbar, .main, .detected-grid {
        grid-template-columns: 1fr;
      }

      #transcript {
        font-size: 21px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <input id="urlInput" type="url" autocomplete="off" spellcheck="false" placeholder="https://www.youtube.com/watch?v=...">
      <button id="startButton">Start</button>
      <button id="stopButton" class="secondary">Stop</button>
    </div>

    <main class="main">
      <section class="video">
        <div id="player"><div class="placeholder">YouTube video</div></div>
      </section>

      <section class="status">
        <h2>Status</h2>
        <pre id="statusLog"></pre>
      </section>
    </main>

    <section class="detected-grid">
      <section class="text-panel">
        <h2>Detected Sentences</h2>
        <div id="sentenceList"></div>
      </section>
      <section class="text-panel">
        <h2>Detected Text</h2>
        <div id="transcript"></div>
      </section>
    </section>
  </div>

  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    const urlInput = document.getElementById("urlInput");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const statusLog = document.getElementById("statusLog");
    const transcript = document.getElementById("transcript");
    const sentenceList = document.getElementById("sentenceList");

    let player = null;
    let ytReady = false;
    let pendingVideoId = null;
    let activeSessionId = null;
    let timePing = null;
    let finalTexts = [];
    let sentences = [];
    let stableText = "";
    let realtimeText = "";
    let postedFirstPlaying = false;

    function snippet(text, limit = 260) {
      text = text || "";
      return text.length <= limit ? text : `${text.slice(0, limit)}...`;
    }

    function debugLog(stage, payload) {
      const body = JSON.stringify({
        stage,
        session_id: activeSessionId,
        browser_time_ms: performance.now(),
        payload: payload || {}
      });
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([body], {type: "application/json"});
          navigator.sendBeacon("/api/debug-log", blob);
          return;
        }
      } catch (_) {}
      fetch("/api/debug-log", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body,
        keepalive: true
      }).catch(() => {});
    }

    function appendStatus(message, cssClass) {
      const line = document.createElement("div");
      if (cssClass) {
        line.className = cssClass;
      }
      const timestamp = new Date().toLocaleTimeString();
      line.textContent = `[${timestamp}] ${message}`;
      statusLog.appendChild(line);
      statusLog.scrollTop = statusLog.scrollHeight;
    }

    function resetDisplay() {
      statusLog.textContent = "";
      finalTexts = [];
      sentences = [];
      stableText = "";
      realtimeText = "";
      postedFirstPlaying = false;
      stopTimePings();
      renderTranscript();
      renderSentences();
    }

    function speakerClass(speaker) {
      if (!speaker) {
        return "final-unknown";
      }
      const number = Number(String(speaker).replace(/[^\d]/g, "")) || 0;
      if (!number) {
        return "final-unknown";
      }
      return `final-speaker-${((number - 1) % 6) + 1}`;
    }

    function renderTranscript() {
      const fragment = document.createDocumentFragment();
      finalTexts.forEach((item) => {
        const span = document.createElement("span");
        span.className = speakerClass(item.speaker);
        span.textContent = `${item.text.trim()} `;
        fragment.appendChild(span);
      });

      if (stableText) {
        const stable = document.createElement("span");
        stable.className = "stable";
        stable.textContent = stableText;
        fragment.appendChild(stable);
      }

      if (realtimeText) {
        const realtime = document.createElement("span");
        realtime.className = "realtime";
        realtime.textContent = realtimeText;
        fragment.appendChild(realtime);
      }

      transcript.replaceChildren(fragment);
      transcript.scrollTop = transcript.scrollHeight;
      requestAnimationFrame(() => {
        transcript.scrollTop = transcript.scrollHeight;
      });
    }

    function probabilityKeys(probabilities) {
      return Object.keys(probabilities || {}).sort((left, right) => {
        if (left === "unknown") return -1;
        if (right === "unknown") return 1;
        const leftNumber = Number(left.replace(/[^\d]/g, "")) || 0;
        const rightNumber = Number(right.replace(/[^\d]/g, "")) || 0;
        return leftNumber - rightNumber || left.localeCompare(right);
      });
    }

    function formatProb(value) {
      return `${Math.round(Number(value || 0) * 100)}%`;
    }

    function sentenceBadgeText(item) {
      if (item.pending) {
        return "...";
      }
      if (!item.assigned_speaker) {
        return "UNK";
      }
      return item.created_speaker ? `${item.assigned_speaker} new` : item.assigned_speaker;
    }

    function renderSentences() {
      const fragment = document.createDocumentFragment();
      sentences.forEach((item) => {
        const row = document.createElement("article");
        row.className = `sentence-item${item.pending ? " pending" : ""}`;

        const top = document.createElement("div");
        top.className = "sentence-top";

        const badge = document.createElement("div");
        badge.className = "speaker-badge";
        if (!item.assigned_speaker) {
          badge.classList.add("speaker-unknown");
        }
        if (item.created_speaker) {
          badge.classList.add("speaker-new");
        }
        badge.textContent = sentenceBadgeText(item);
        top.appendChild(badge);

        const meta = document.createElement("div");
        meta.className = "sentence-meta";
        if (item.pending) {
          meta.textContent = "embedding pending";
        } else {
          const parts = [];
          if (item.duration_seconds) {
            parts.push(`${Number(item.duration_seconds).toFixed(2)}s`);
          }
          if (item.top_similarity !== null && item.top_similarity !== undefined) {
            parts.push(`sim ${Number(item.top_similarity).toFixed(2)}`);
          }
          if (item.margin !== null && item.margin !== undefined) {
            parts.push(`margin ${Number(item.margin).toFixed(2)}`);
          }
          if (item.assignment_source === "context") {
            parts.push("context");
          }
          meta.textContent = parts.join("  ");
        }
        top.appendChild(meta);
        row.appendChild(top);

        const text = document.createElement("div");
        text.className = "sentence-text";
        text.textContent = item.text || "";
        row.appendChild(text);

        const probs = document.createElement("div");
        probs.className = "probabilities";
        probabilityKeys(item.probabilities).forEach((key) => {
          const label = document.createElement("span");
          label.className = `prob-label${key === "unknown" ? " unknown" : ""}`;
          label.textContent = `${key} ${formatProb(item.probabilities[key])}`;
          probs.appendChild(label);
        });
        row.appendChild(probs);

        fragment.appendChild(row);
      });

      sentenceList.replaceChildren(fragment);
      sentenceList.scrollTop = sentenceList.scrollHeight;
      requestAnimationFrame(() => {
        sentenceList.scrollTop = sentenceList.scrollHeight;
      });
    }

    function upsertFinal(index, text) {
      const existing = finalTexts.find((item) => item.index === index);
      if (existing) {
        existing.text = text;
      } else {
        finalTexts.push({index, text, speaker: null});
      }
      renderTranscript();
    }

    function upsertSentence(data) {
      const existing = sentences.find((item) => item.index === data.index);
      if (existing) {
        Object.assign(existing, data);
      } else {
        sentences.push(data);
      }
      sentences.sort((left, right) => Number(left.index) - Number(right.index));

      const finalItem = finalTexts.find((item) => item.index === data.index);
      if (finalItem && data.assigned_speaker) {
        finalItem.speaker = data.assigned_speaker;
      }
      renderSentences();
      renderTranscript();
    }

    function consumeFinalFromRealtime(finalText) {
      const finalClean = (finalText || "").trim();
      if (!finalClean || (!stableText && !realtimeText)) {
        return;
      }
      const current = `${stableText || ""}${realtimeText || ""}`;
      const leading = current.length - current.trimStart().length;
      const comparable = current.trimStart().toLowerCase();
      if (comparable.startsWith(finalClean.toLowerCase())) {
        stableText = "";
        realtimeText = current.slice(leading + finalClean.length).trimStart();
      }
    }

    function splitRealtime(displayText, stable, unstable) {
      displayText = displayText || "";
      stable = stable || "";
      unstable = unstable || "";
      stableText = stable;
      if (unstable) {
        realtimeText = unstable;
      } else if (stable && displayText.toLowerCase().startsWith(stable.toLowerCase())) {
        realtimeText = displayText.slice(stable.length);
      } else {
        realtimeText = displayText;
      }
      renderTranscript();
      debugLog("realtimeRender", {
        stable: snippet(stableText),
        realtime: snippet(realtimeText)
      });
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function playerCurrentTime() {
      try {
        return player ? player.getCurrentTime() || 0 : 0;
      } catch (_) {
        return 0;
      }
    }

    function sendVideoTime(path) {
      if (!activeSessionId || !player) {
        return;
      }
      postJson(path, {
        session_id: activeSessionId,
        current_time: playerCurrentTime()
      }).catch((error) => appendStatus(error.message, "error"));
    }

    function startTimePings() {
      if (timePing) {
        clearInterval(timePing);
      }
      timePing = setInterval(() => {
        if (player && player.getPlayerState && player.getPlayerState() === YT.PlayerState.PLAYING) {
          sendVideoTime("/api/video-time");
        }
      }, 500);
    }

    function stopTimePings() {
      if (timePing) {
        clearInterval(timePing);
        timePing = null;
      }
    }

    function createPlayer(videoId) {
      pendingVideoId = null;
      if (player && player.destroy) {
        player.destroy();
      }
      player = new YT.Player("player", {
        width: "100%",
        height: "100%",
        videoId,
        playerVars: {
          autoplay: 1,
          controls: 1,
          rel: 0,
          modestbranding: 1,
          playsinline: 1
        },
        events: {
          onReady: (event) => {
            appendStatus("YouTube player ready. Starting playback.");
            event.target.playVideo();
          },
          onStateChange: (event) => {
            if (event.data === YT.PlayerState.PLAYING) {
              if (!postedFirstPlaying) {
                postedFirstPlaying = true;
                appendStatus("YouTube reports PLAYING. WASAPI capture is armed.");
                sendVideoTime("/api/video-playing");
              }
              startTimePings();
            } else if (
              event.data === YT.PlayerState.PAUSED ||
              event.data === YT.PlayerState.ENDED
            ) {
              stopTimePings();
              if (event.data === YT.PlayerState.ENDED) {
                appendStatus("YouTube ended. Draining final transcripts.");
                postJson("/api/stop", {}).catch((error) => appendStatus(error.message, "error"));
              }
            }
          }
        }
      });
    }

    window.onYouTubeIframeAPIReady = function() {
      ytReady = true;
      if (pendingVideoId) {
        createPlayer(pendingVideoId);
      }
    };

    function loadVideo(videoId) {
      if (!ytReady) {
        pendingVideoId = videoId;
        appendStatus("Waiting for YouTube player API.");
        return;
      }
      createPlayer(videoId);
    }

    const events = new EventSource("/events");
    events.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "");
    });
    events.addEventListener("error-status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "Error", "error");
    });
    events.addEventListener("capture-ready", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus("Capture ready. Loading YouTube video.");
      loadVideo(data.video_id);
    });
    events.addEventListener("realtime", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      splitRealtime(
        data.display_text || data.text || "",
        data.stable_text || "",
        data.unstable_text || ""
      );
    });
    events.addEventListener("final", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      const text = (data.text || "").trim();
      if (text) {
        upsertFinal(data.index, text);
        consumeFinalFromRealtime(text);
        renderTranscript();
      }
    });
    events.addEventListener("sentence", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      upsertSentence(data);
    });

    startButton.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (!url) {
        appendStatus("Paste a YouTube URL first.", "error");
        return;
      }

      startButton.disabled = true;
      resetDisplay();
      appendStatus("Started.");

      try {
        const data = await postJson("/api/start", {url});
        activeSessionId = data.session_id;
      } catch (error) {
        appendStatus(error.message, "error");
      } finally {
        startButton.disabled = false;
      }
    });

    stopButton.addEventListener("click", async () => {
      stopTimePings();
      if (player && player.pauseVideo) {
        player.pauseVideo();
      }
      try {
        await postJson("/api/stop", {session_id: activeSessionId});
        appendStatus("Stop requested.");
      } catch (error) {
        appendStatus(error.message, "error");
      }
    });
  </script>
</body>
</html>
"""
