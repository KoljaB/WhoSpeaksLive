/** Create one immutable-snapshot store for serializable browser state. */
export function createStore(initialState) {
  let current = freezeState(initialState);
  const listeners = new Set();
  let disposed = false;

  return Object.freeze({
    getState() { return current; },
    update(updater) {
      if (disposed) return current;
      const next = typeof updater === "function" ? updater(current) : updater;
      if (!next || next === current) return current;
      current = freezeState(next);
      for (const listener of [...listeners]) listener(current);
      return current;
    },
    updateSlice(name, updater) {
      return this.update(state => ({
        ...state,
        [name]: freezeState(typeof updater === "function" ? updater(state[name]) : updater),
      }));
    },
    subscribe(listener) {
      if (disposed) return () => {};
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      listeners.clear();
    },
  });
}

/** Own non-serializable resources such as streams, timers, and AudioContexts. */
export function createResourceRegistry() {
  const disposers = [];
  let disposed = false;
  return Object.freeze({
    own(disposer) {
      if (disposed) {
        disposer();
        return disposer;
      }
      disposers.push(disposer);
      return disposer;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const disposer of disposers.splice(0).reverse()) {
        try { disposer(); } catch (_) {}
      }
    },
  });
}

function freezeState(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  return Object.freeze(value);
}
