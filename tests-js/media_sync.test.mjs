import assert from "node:assert/strict";
import test from "node:test";

import {createAudioMasterVideoSyncPolicy} from "../src/window/assets/web/live/media_sync.js";


test("audio-master sync ignores sub-threshold jitter", () => {
  const policy = createAudioMasterVideoSyncPolicy();
  for (let index = 0; index < 8; index += 1) {
    const action = policy.sample({audioTime: 10 + index, videoTime: 10.08 + index, nowMs: index * 400});
    assert.equal(action.kind, "hold");
    assert.equal(action.playbackRate, 1);
  }
  assert.equal(policy.snapshot().mode, "idle");
});


test("audio-master sync requires stable drift and gently catches video up", () => {
  const policy = createAudioMasterVideoSyncPolicy();
  assert.equal(policy.sample({audioTime: 10, videoTime: 9.86, nowMs: 0}).kind, "hold");
  assert.equal(policy.sample({audioTime: 10.4, videoTime: 10.26, nowMs: 400}).kind, "hold");
  const correction = policy.sample({audioTime: 10.8, videoTime: 10.66, nowMs: 800});
  assert.equal(correction.kind, "slew");
  assert.ok(correction.playbackRate > 1);
  assert.ok(correction.playbackRate <= 1.06);

  const settled = policy.sample({audioTime: 13, videoTime: 12.97, nowMs: 3000});
  assert.equal(settled.kind, "hold");
  assert.equal(settled.playbackRate, 1);
  assert.equal(policy.snapshot().mode, "cooldown");
});


test("audio-master sync performs one video-only seek for large drift", () => {
  const policy = createAudioMasterVideoSyncPolicy();
  const action = policy.sample({audioTime: 42, videoTime: 41.2, nowMs: 5000});
  assert.deepEqual(action, {
    kind: "seek",
    driftSeconds: -0.7999999999999972,
    playbackRate: 1,
    targetTime: 42,
  });

  const cooldown = policy.sample({audioTime: 42.4, videoTime: 42.25, nowMs: 5400});
  assert.equal(cooldown.kind, "hold");
  assert.equal(policy.snapshot().mode, "cooldown");
});
