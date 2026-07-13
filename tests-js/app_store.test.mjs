import test from "node:test";
import assert from "node:assert/strict";

import {createResourceRegistry, createStore} from "../src/window/assets/web/live/app_store.js";

test("store replaces one immutable slice and notifies subscribers", () => {
  const store = createStore({run: {status: "idle"}, media: {version: 0}});
  const observed = [];
  store.subscribe(state => observed.push(state.run.status));

  const next = store.updateSlice("run", run => ({...run, status: "running"}));

  assert.equal(next.run.status, "running");
  assert.deepEqual(observed, ["running"]);
  assert(Object.isFrozen(next));
  assert(Object.isFrozen(next.run));
});

test("resource registry disposes once in reverse ownership order", () => {
  const registry = createResourceRegistry();
  const order = [];
  registry.own(() => order.push("first"));
  registry.own(() => order.push("second"));

  registry.dispose();
  registry.dispose();

  assert.deepEqual(order, ["second", "first"]);
});
