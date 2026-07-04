#!/usr/bin/env bash
set -euo pipefail

LOG=/home/lon/Dev/voice-embeddings-server/embeddings-desktop-control.log
SERVICE=voice-embeddings-server.service

{
  echo "[$(date -Is)] stop requested"
  systemctl --user stop "$SERVICE"
  sleep 1
  systemctl --user --no-pager --full status "$SERVICE" || true
} >>"$LOG" 2>&1 || true

if systemctl --user is-active --quiet "$SERVICE"; then
  notify-send "Embeddings server still running" "See $LOG" || true
  exit 1
fi

notify-send "Embeddings server stopped" "Provider models and any VRAM are released." || true
exit 0
