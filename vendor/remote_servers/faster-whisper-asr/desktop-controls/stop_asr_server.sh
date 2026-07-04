#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="${WHOSPEAKS_ASR_SERVICE:-faster-whisper-asr.service}"
LOG_FILE="${WHOSPEAKS_ASR_CONTROL_LOG:-"$SCRIPT_DIR/asr-desktop-control.log"}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

notice() {
    local title="$1"
    local body="$2"
    local urgency="${3:-normal}"

    if command -v notify-send >/dev/null 2>&1; then
        notify-send --urgency="$urgency" "$title" "$body" 2>/dev/null || true
    elif [ -x /usr/bin/zenity ]; then
        if [ "$urgency" = "critical" ]; then
            /usr/bin/zenity --error --timeout=8 --title="$title" --text="$body" --width=430 2>/dev/null || true
        else
            /usr/bin/zenity --info --timeout=8 --title="$title" --text="$body" --width=430 2>/dev/null || true
        fi
    fi
}

{
    echo "[$(timestamp)] Stop requested from Linux desktop launcher"
    systemctl --user stop "$SERVICE"
    rc=$?
    echo "[$(timestamp)] systemctl --user stop exit_code=${rc}"
    systemctl --user --no-pager --plain status "$SERVICE" || true
} >>"$LOG_FILE" 2>&1

status_text="$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)"

if [ "${rc}" -eq 0 ]; then
    body="Stopped the ASR faster-whisper service.
Status: ${status_text:-inactive}
Log: $LOG_FILE"
    notice "ASR server stopped" "$body" normal
else
    body="systemctl stop failed.
Log: $LOG_FILE"
    notice "Could not stop ASR server" "$body" critical
    exit "${rc}"
fi
