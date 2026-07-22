const DEFAULT_VIDEO_SYNC_OPTIONS = Object.freeze({
  enterDriftSeconds: 0.1,
  exitDriftSeconds: 0.04,
  hardSeekSeconds: 0.5,
  stableSamples: 3,
  maxRateAdjustment: 0.06,
  rateGain: 0.35,
  cooldownMs: 1800,
});

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function createAudioMasterVideoSyncPolicy(options = {}) {
  const config = {...DEFAULT_VIDEO_SYNC_OPTIONS, ...options};
  let mode = "idle";
  let stableDirection = 0;
  let stableCount = 0;
  let slewDirection = 0;
  let cooldownUntilMs = 0;

  function reset() {
    mode = "idle";
    stableDirection = 0;
    stableCount = 0;
    slewDirection = 0;
    cooldownUntilMs = 0;
  }

  function hold(driftSeconds, baseRate) {
    return {kind: "hold", driftSeconds, playbackRate: baseRate};
  }

  function sample({audioTime, videoTime, baseRate = 1, nowMs = 0} = {}) {
    const audioSeconds = finiteNumber(audioTime, Number.NaN);
    const videoSeconds = finiteNumber(videoTime, Number.NaN);
    const masterRate = Math.max(0.1, finiteNumber(baseRate, 1));
    const now = finiteNumber(nowMs, 0);
    if (!Number.isFinite(audioSeconds) || !Number.isFinite(videoSeconds)) {
      reset();
      return hold(0, masterRate);
    }

    const driftSeconds = videoSeconds - audioSeconds;
    const absoluteDrift = Math.abs(driftSeconds);
    const direction = Math.sign(driftSeconds);

    if (absoluteDrift >= config.hardSeekSeconds) {
      mode = "cooldown";
      stableDirection = 0;
      stableCount = 0;
      slewDirection = 0;
      cooldownUntilMs = now + config.cooldownMs;
      return {
        kind: "seek",
        driftSeconds,
        playbackRate: masterRate,
        targetTime: audioSeconds,
      };
    }

    if (mode === "cooldown") {
      if (now < cooldownUntilMs) return hold(driftSeconds, masterRate);
      mode = "idle";
    }

    if (mode === "slewing") {
      if (absoluteDrift <= config.exitDriftSeconds) {
        mode = "cooldown";
        slewDirection = 0;
        cooldownUntilMs = now + config.cooldownMs;
        return hold(driftSeconds, masterRate);
      }
      if (direction !== slewDirection) {
        mode = "idle";
        stableDirection = direction;
        stableCount = 1;
        slewDirection = 0;
        return hold(driftSeconds, masterRate);
      }
      const relativeRate = clamp(
        1 - (driftSeconds * config.rateGain),
        1 - config.maxRateAdjustment,
        1 + config.maxRateAdjustment,
      );
      return {kind: "slew", driftSeconds, playbackRate: masterRate * relativeRate};
    }

    if (absoluteDrift < config.enterDriftSeconds) {
      stableDirection = 0;
      stableCount = 0;
      return hold(driftSeconds, masterRate);
    }

    if (direction === stableDirection) stableCount += 1;
    else {
      stableDirection = direction;
      stableCount = 1;
    }
    if (stableCount < config.stableSamples) return hold(driftSeconds, masterRate);

    mode = "slewing";
    slewDirection = direction;
    stableDirection = 0;
    stableCount = 0;
    const relativeRate = clamp(
      1 - (driftSeconds * config.rateGain),
      1 - config.maxRateAdjustment,
      1 + config.maxRateAdjustment,
    );
    return {kind: "slew", driftSeconds, playbackRate: masterRate * relativeRate};
  }

  function snapshot() {
    return {mode, stableDirection, stableCount, slewDirection, cooldownUntilMs};
  }

  return {reset, sample, snapshot};
}

export {DEFAULT_VIDEO_SYNC_OPTIONS};
