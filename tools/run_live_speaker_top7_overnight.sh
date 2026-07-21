#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LIVE_SPEAKER_OPT_PYTHON:-/home/lon/speakerdiar_opt/.venv/bin/python}"
BUDGET_SECONDS="${LIVE_SPEAKER_OPT_BUDGET_SECONDS:-28800}"
RUN_DIR="${LIVE_SPEAKER_OPT_RUN_DIR:-runtime/optimization/live_speaker_runs/20260721_top7_overnight}"

cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PYTHON_BIN" tools/optimize_live_speaker_overnight_top7.py \
  --spec runtime/optimization/live_speaker_overnight_top7_20260721/spec.json \
  --corpus-root runtime/optimization/live_shifting_windows_v1 \
  --input-root runtime/optimization/live_speaker_overnight_top7_20260721/inputs \
  --run-dir "$RUN_DIR" \
  --budget-seconds "$BUDGET_SECONDS" \
  --resume
