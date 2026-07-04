#!/usr/bin/env bash
set -euo pipefail

LOG=/home/lon/Dev/voice-embeddings-server/embeddings-desktop-control.log
SERVICE=voice-embeddings-server.service

{
  echo "[$(date -Is)] start requested"
  systemctl --user start "$SERVICE"
  sleep 1
  systemctl --user --no-pager --full status "$SERVICE"
} >>"$LOG" 2>&1 || true

if systemctl --user is-active --quiet "$SERVICE"; then
  notify-send "Embeddings server started" "Listening on http://192.168.178.22:8660" || true
  exit 0
fi

notify-send "Embeddings server failed to start" "See $LOG" || true
exit 1
