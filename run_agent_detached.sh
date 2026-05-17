#!/usr/bin/env bash
# run_agent_detached.sh — Launch the LiveKit agent watchdog in a fully detached
# new session (setsid) so it survives shell tool timeouts and SSH disconnects.
# vLLM must already be running on :5000 before calling this script.
#
# Usage: bash /app/run_agent_detached.sh
# Logs:  /tmp/agent.log  (appended)
# PID:   /tmp/agent_watchdog.pid

PIDFILE=/tmp/agent_watchdog.pid

# Guard: abort if a watchdog is already alive
if [[ -f "${PIDFILE}" ]]; then
    OLD_PID=$(cat "${PIDFILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Watchdog already running (PID=${OLD_PID}). Aborting."
        exit 1
    fi
    rm -f "${PIDFILE}"
fi

# Guard: abort if port 8081 is already bound (stale agent)
if ss -tlnp 2>/dev/null | grep -q ':8081 '; then
    echo "Port 8081 already in use. Kill the existing agent first:" >&2
    ss -tlnp | grep ':8081 ' >&2
    exit 1
fi

# Source env from vLLM's /proc/environ to get all pod template variables
eval "$(cat /proc/$(pgrep -f 'vllm serve' | head -1)/environ | tr '\0' '\n' \
    | grep -E '^(LIVEKIT|ULTRAVOX|MINIMAX|MIRA|WHISPER|HF_HOME|USE_REAL|WORKSPACE)=' \
    | grep -v '^$' \
    | sed 's/^/export /')"

export HF_HUB_ENABLE_HF_TRANSFER=1
export AGENT_RESTART_DELAY=5
export MIRA_USE_AUDIO_TOWER=1
export MIRA_RAG_COMPOSE=1

watchdog_loop() {
    echo "[watchdog] Starting at $(date -u +%FT%TZ), PID=$$" >> /tmp/agent.log
    echo $$ > "${PIDFILE}"
    cd /app
    while true; do
        python agent.py start >> /tmp/agent.log 2>&1 &
        AGENT_PID=$!
        echo "[watchdog] Agent started PID=${AGENT_PID} at $(date -u +%FT%TZ)" >> /tmp/agent.log
        wait "${AGENT_PID}"
        EXIT_CODE=$?
        if [[ ${EXIT_CODE} -eq 130 || ${EXIT_CODE} -eq 143 ]]; then
            echo "[watchdog] Agent exited cleanly (code=${EXIT_CODE}). Stopping." >> /tmp/agent.log
            rm -f "${PIDFILE}"
            break
        fi
        echo "[watchdog] Agent exited code=${EXIT_CODE}. Restarting in ${AGENT_RESTART_DELAY}s..." >> /tmp/agent.log
        sleep "${AGENT_RESTART_DELAY}"
    done
}

export -f watchdog_loop
export PIDFILE
setsid bash -c 'watchdog_loop' </dev/null >>/tmp/agent.log 2>&1 &
WATCHDOG_PID=$!
echo "[run_agent_detached] Watchdog launched PID=${WATCHDOG_PID}"
echo "  Logs: tail -f /tmp/agent.log"
