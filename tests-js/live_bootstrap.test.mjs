import assert from "node:assert/strict";
import test from "node:test";


function fakeElement(id = "") {
  const bootstrap = {
    speaker_colors: [], source: "", preset_videos: [],
    new_speaker_sensitivity: {
      selected: 3,
      presets: [
        {level: 1, label: "Conservative"},
        {level: 2, label: "Moderate"},
        {level: 3, label: "Balanced"},
      ],
    },
    speaker_refinement: {},
    live_speaker: {session_lease_enabled: false, highlight_transcript: true},
    language: {code: "en", name: "English", flag_url: ""},
    translation: {
      available: true,
      provider: "mock",
      display_mode: "single",
      primary_target: "de",
      target_languages: ["de"],
      languages: [{code: "de", name: "German"}],
    },
    speaker_library: {group_name: "", groups: [], speakers: []},
  };
  const element = {
    id,
    textContent: id === "bootstrap-data" ? JSON.stringify(bootstrap) : "",
    value: "",
    checked: false,
    disabled: false,
    hidden: false,
    open: false,
    files: [],
    options: [],
    children: [],
    listeners: new Map(),
    attributes: {},
    dataset: {},
    style: {setProperty() {}, removeProperty() {}},
    className: "",
    currentTime: 0,
    duration: 0,
    paused: true,
    ended: false,
    parentElement: null,
    classList: {
      add() {}, remove() {}, toggle() { return false; }, contains() { return false; },
    },
    addEventListener(type, listener) {
      const listeners = this.listeners.get(type) || [];
      listeners.push(listener);
      this.listeners.set(type, listeners);
    },
    removeEventListener(type, listener) {
      const listeners = this.listeners.get(type) || [];
      this.listeners.set(type, listeners.filter(candidate => candidate !== listener));
    },
    async emit(type, event = {}) {
      const payload = {stopPropagation() {}, preventDefault() {}, target: this, ...event};
      for (const listener of this.listeners.get(type) || []) {
        await listener.call(this, payload);
      }
    },
    append(...items) { items.forEach(item => this.appendChild(item)); },
    appendChild(child) {
      this.children.push(child);
      if (child && typeof child === "object") child.parentElement = this;
      return child;
    },
    prepend(...items) {
      this.children.unshift(...items);
      items.forEach(item => { if (item && typeof item === "object") item.parentElement = this; });
    },
    insertBefore(child) { return this.appendChild(child); },
    remove() {},
    replaceChildren(...items) {
      this.children = [];
      this.append(...items);
    },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    removeAttribute(name) { delete this.attributes[name]; },
    contains() { return false; },
    closest() { return null; },
    scrollIntoView() {},
    focus() {},
    click() { return this.emit("click"); },
    load() {},
    pause() {},
    play() { return Promise.resolve(); },
    querySelector() { return fakeElement(); },
    querySelectorAll() { return []; },
  };
  element.parentElement = element;
  return element;
}


test("live ES modules bootstrap and dispose with explicit owners", async () => {
  const elements = new Map();
  const eventSources = [];
  const fetchCalls = [];
  const activeIntervals = new Set();
  const activeTimeouts = new Set();
  let nextTimerId = 1;
  globalThis.document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, fakeElement(id));
      return elements.get(id);
    },
    querySelector(selector) {
      if (selector === "[data-speaker-panel='sessions']") return null;
      return fakeElement();
    },
    querySelectorAll() { return []; },
    createElement() { return fakeElement(); },
    createElementNS() { return fakeElement(); },
    createTextNode(text) { return {textContent: String(text), parentElement: null}; },
    addEventListener() {},
    removeEventListener() {},
    documentElement: {scrollHeight: 0},
    visibilityState: "visible",
  };
  globalThis.window = {
    crypto: {randomUUID: () => "client-id"},
    matchMedia: () => ({matches: false, addEventListener() {}, removeEventListener() {}}),
    addEventListener() {},
    removeEventListener() {},
    scrollTo() {},
    location: {href: "http://localhost/"},
  };
  globalThis.localStorage = {getItem() { return null; }, setItem() {}, removeItem() {}};
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {clipboard: {writeText: async () => {}}, mediaDevices: {}},
  });
  globalThis.performance = {now: () => 0};
  globalThis.EventSource = class {
    constructor(url) {
      this.url = url;
      this.listeners = new Map();
      eventSources.push(this);
    }
    addEventListener(type, listener) {
      const listeners = this.listeners.get(type) || [];
      listeners.push(listener);
      this.listeners.set(type, listeners);
    }
    emit(type, data) {
      for (const listener of this.listeners.get(type) || []) {
        listener({data: JSON.stringify(data)});
      }
    }
    close() {}
  };
  globalThis.fetch = async (url, options = {}) => {
    const path = String(url);
    fetchCalls.push({path, options});
    let payload = {ok: true};
    if (path.startsWith("/api/sessions?")) {
      payload = {ok: true, sessions: []};
    } else if (path === "/api/sessions/create") {
      payload = {ok: true, session: {id: "saved-session-1"}};
    } else if (path === "/api/start") {
      payload = {
        ok: true,
        speaker_state: {group_name: "", groups: [], speakers: []},
        saved_session: {id: "saved-session-1"},
      };
    } else if (path.startsWith("/api/session/status?")) {
      payload = {ok: true, session: {active: false, is_owner: true}};
    }
    return {ok: true, statusText: "OK", json: async () => payload};
  };
  globalThis.setInterval = () => {
    const id = nextTimerId++;
    activeIntervals.add(id);
    return id;
  };
  globalThis.clearInterval = id => { activeIntervals.delete(id); };
  globalThis.setTimeout = () => {
    const id = nextTimerId++;
    activeTimeouts.add(id);
    return id;
  };
  globalThis.clearTimeout = id => {
    activeTimeouts.delete(id);
    activeIntervals.delete(id);
  };
  globalThis.requestAnimationFrame = callback => {
    callback();
    return nextTimerId++;
  };

  const module = await import("../src/window/assets/web/live/app.js");

  assert.equal(typeof module.bootstrapLiveApp, "function");
  assert.equal(typeof module.default, "function");
  assert.ok((elements.get("start").listeners.get("click") || []).length > 0);
  assert.equal(elements.get("translationDisplayMode").value, "single");

  await elements.get("start").emit("click");

  assert.ok(fetchCalls.some(call => call.path === "/api/sessions/create"));
  assert.ok(fetchCalls.some(call => call.path === "/api/start"));
  const startCall = fetchCalls.find(call => call.path === "/api/start");
  assert.equal(JSON.parse(startCall.options.body).processing_mode, "fast");
  assert.equal(eventSources.length, 1);
  assert.equal(elements.get("state").textContent, "Processing");

  const events = eventSources[0];
  events.emit("speakers", {
    group_name: "",
    groups: [],
    speakers: [{id: "SPEAKER_00", display_name: "Alice"}],
  });
  assert.equal(elements.get("speakerCountNumber").textContent, "1");
  events.emit("realtime", {
    realtime: true,
    realtime_generation: 1,
    index: 1,
    start: 0,
    end: 0.8,
    text: "hello",
    assigned_speaker: "SPEAKER_00",
    probabilities: {SPEAKER_00: 0.9, UNKNOWN: 0.1},
  });
  events.emit("sentence", {
    realtime: false,
    index: 1,
    start: 0,
    end: 1,
    text: "hello world",
    assigned_speaker: "SPEAKER_00",
    speaker_name: "Alice",
    probabilities: {SPEAKER_00: 0.95, UNKNOWN: 0.05},
  });
  assert.ok(elements.get("sentences").children.length >= 2);

  module.default();
  assert.equal(activeIntervals.size, 0);
});
