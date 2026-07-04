#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="${WHOSPEAKS_EMBEDDINGS_CONTROL_LOG:-"$SCRIPT_DIR/embeddings-desktop-control.log"}"
SERVICE="${WHOSPEAKS_EMBEDDINGS_SERVICE:-voice-embeddings-server.service}"

{
  echo "[$(date -Is)] start requested"
  systemctl --user start "$SERVICE"
  sleep 1
  systemctl --user --no-pager --full status "$SERVICE"
} >>"$LOG" 2>&1 || true

if systemctl --user is-active --quiet "$SERVICE"; then
  HOST="${EMBEDDINGS_HOST:-0.0.0.0}"
  PORT="${EMBEDDINGS_PORT:-8660}"
  notify-send "Embeddings server started" "Listening on http://${HOST}:${PORT}" || true
  exit 0
fi

notify-send "Embeddings server failed to start" "See $LOG" || true
exit 1
