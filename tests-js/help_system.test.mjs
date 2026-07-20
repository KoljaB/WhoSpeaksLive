import assert from "node:assert/strict";
import test from "node:test";

import {HELP_TOPICS, TOPIC_SELECTORS, disabledReason} from "../src/window/assets/web/live/help_system.js";

test("help registry covers the main live workflows", () => {
  for (const topic of ["source", "transcript", "review", "speakers", "speaker_detection", "sessions", "ask", "insights", "status_log"]) {
    assert.ok(HELP_TOPICS[topic], `missing ${topic} help topic`);
    assert.ok(HELP_TOPICS[topic].summary.length > 20);
    assert.ok(HELP_TOPICS[topic].detail.length > HELP_TOPICS[topic].summary.length);
    assert.ok(TOPIC_SELECTORS[topic]?.length, `missing ${topic} selectors`);
  }
});

test("disabled controls explain the next available action", () => {
  const context = {
    stop: {disabled: true},
    sessionBannerMessage: {textContent: "Another browser controls this session."},
  };

  assert.equal(
    disabledReason({id: "bulkReassign", disabled: true, dataset: {}}, context),
    "Select one or more transcript rows first.",
  );
  assert.equal(
    disabledReason({id: "newRunSession", disabled: true, dataset: {disabledHelp: "Finish the current run first."}}, context),
    "Finish the current run first.",
  );
  assert.equal(disabledReason({id: "stop", disabled: false, dataset: {}}, context), "");
});
