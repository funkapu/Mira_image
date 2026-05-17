#!/usr/bin/env bash
# start_agent.sh — Pod boot entrypoint.
#
# 1. Verifies model weights are on the volume (run setup_runpod.sh first
#    if they're missing).
# 2. Launches vLLM (Ultravox) in the background on :5000.
# 3. Waits until vLLM's /v1/models responds, then runs the LiveKit
#    agent worker in a watchdog loop (auto-restarts on unexpected exit).

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
# merged-full-v2-awq-sym/ is the 66 GB fully-merged Qwen3-32B + Whisper turbo +
# projector checkpoint. The sibling qwen3-lora-whisper-8000/ folder on the
# same HF repo is only a 1.6 GB adapter stub (LoRA + projector + audio_tower);
# loading that as a vLLM model fails with empty/incomplete weights.
ULTRAVOX_DIR="${MODELS_DIR}/ultravox/merged-full-v2-awq-sym"
WHISPER_CT2_DIR="${MODELS_DIR}/whisper-thai-ct2"

# Make caches land on the persistent volume
export HF_HOME="${WORKSPACE}/.hf"
export WHISPER_CACHE_DIR="${WHISPER_CT2_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Tell the agent where vLLM lives (mira_agent_v4 reads ULTRAVOX_URL)
export ULTRAVOX_URL="${ULTRAVOX_URL:-http://localhost:5000}"
export ULTRAVOX_MODEL="${ULTRAVOX_MODEL:-mira-cbt}"
export ULTRAVOX_API_KEY="${ULTRAVOX_API_KEY:-dummy-key}"

# Required for mira_agent_v4 to use the real Ultravox endpoint instead
# of the Cerebras-only fallback. Off=fallback to Cerebras gpt-oss-120b.
export USE_REAL_ULTRAVOX="${USE_REAL_ULTRAVOX:-true}"

# Default to MiniMax for Thai TTS (matches user's spec; cartesia stays
# available as fallback if MIRA_TTS=cartesia is set in the Pod template).
export MIRA_TTS="${MIRA_TTS:-minimax}"

# MiniMax voice - hardcoded (pod template defaults to Thai_Optimistic_girl)
export MINIMAX_VOICE_ID="Portuguese_CuteElf"

# MiniMax voice - hardcoded (pod template defaults to Thai_Optimistic_girl)
export MINIMAX_VOICE_ID="Portuguese_CuteElf"

# Voice agent version. v5 = cascading streaming (default, low-latency therapy).
# Set MIRA_AGENT_VERSION=v4 in the Pod template to roll back to the
# pre-cascading agent if a regression appears.
export MIRA_AGENT_VERSION="${MIRA_AGENT_VERSION:-v5}"

# Stage management toggle. When =1, judge_node short-circuits (no phase
# transitions), counsellor skips per-phase branching, output_filter strips
# but ignores [NEXT_STAGE:] tags, and _load_prompt uses the trimmed
# mira_system_v4_thai_nostage.txt (~500 tokens lighter). LLM (CBT-finetuned)
# drives conversation flow itself. Default 0 = original stage logic active.
export MIRA_STAGE_DISABLED="${MIRA_STAGE_DISABLED:-0}"

# Conversation history sliding-window (last N user turns kept). Default 4.
export MIRA_HISTORY_WINDOW_TURNS="${MIRA_HISTORY_WINDOW_TURNS:-4}"

# Unified single-stage prompt mode. When =1 (default), v5 agent uses
# mira_unified_v1.txt directly and skips supervisor/judge/per-phase logic.
# Set =0 to fall back to the legacy stage-disabled multi-prompt path.
export MIRA_UNIFIED_PROMPT="${MIRA_UNIFIED_PROMPT:-1}"

# RAG compose pipeline. =1 enables RAG (production), =0 legacy fallback.
export MIRA_RAG_COMPOSE="${MIRA_RAG_COMPOSE:-0}"

# ── 1. Pre-flight checks ────────────────────────────────────────────────
echo "[start] Pod boot at $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || { echo "FAIL: nvidia-smi unavailable — is this a GPU Pod?"; exit 1; }

if [[ ! -f "${ULTRAVOX_DIR}/config.json" ]]; then
    echo "FAIL: Ultravox weights not found at ${ULTRAVOX_DIR}"
    echo "      Run setup_runpod.sh first (one-time, downloads from HF Hub)."
    exit 1
fi
if [[ ! -f "${WHISPER_CT2_DIR}/model.bin" ]]; then
    echo "FAIL: Whisper CT2 weights not found at ${WHISPER_CT2_DIR}"
    echo "      Run setup_runpod.sh first."
    exit 1
fi

for var in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET MINIMAX_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        echo "WARN: ${var} is not set — agent may fail to start."
    fi
done

# ── 2. Launch vLLM in the background ────────────────────────────────────
# Notes:
#   --trust-remote-code: Ultravox ships custom modeling files
#   --limit-mm-per-prompt audio=1: only one audio clip per request
#   --gpu-memory-utilization 0.82: leave ~14 GB free for whisper + buffers
#   --enable-prefix-caching: v5 win — system prompt is identical across
#       turns, so caching its prefix saves prompt-prefill time on TTFT.
#   --max-model-len 8192: balances context room for system prompt + ~4
#       turns of history + audio embedding. 8192 was tried but A100 80G
#       at gpu-mem-util 0.82 had insufficient KV cache (1.18 GiB <
#       2.00 GiB needed). 4096 is the practical ceiling at this util.
echo "[start] Launching vLLM..."
vllm serve "${ULTRAVOX_DIR}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.82 \
    --port 5000 --quantization compressed-tensors \
    --served-model-name "${ULTRAVOX_MODEL}" \
    --limit-mm-per-prompt '{"audio": 1}' \
    --enable-prefix-caching \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "[start] vLLM PID=${VLLM_PID} (logs: /tmp/vllm.log)"

# Tail vLLM logs to stdout so RunPod's log viewer shows them
( tail -F /tmp/vllm.log | sed 's/^/[vllm] /' ) &
TAIL_PID=$!

cleanup() {
    echo "[start] shutting down — killing vLLM (${VLLM_PID})"
    [[ -n "${AGENT_PID:-}" ]] && kill -TERM "${AGENT_PID}" 2>/dev/null || true
    kill "${VLLM_PID}" "${TAIL_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 3. Wait for vLLM to become ready ────────────────────────────────────
# Cold load of ~65GB safetensors + audio_tower into A100 takes 60–120s.
echo "[start] Waiting for vLLM /v1/models ..."
for i in $(seq 1 600); do
    if curl -fsS http://localhost:5000/v1/models >/dev/null 2>&1; then
        echo "[start] vLLM ready after ${i}s."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FAIL: vLLM exited early — see /tmp/vllm.log"
        tail -n 80 /tmp/vllm.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS http://localhost:5000/v1/models >/dev/null 2>&1; then
    echo "FAIL: vLLM did not become ready within 600s."
    tail -n 80 /tmp/vllm.log
    exit 1
fi

# ── 4. Start the LiveKit agent (watchdog loop) ──────────────────────────
# Runs agent.py in a restart loop so a transient crash or unexpected exit
# does not kill the pod. The cleanup trap above (EXIT/INT/TERM) still fires
# on the shell process, killing vLLM and the tail when the pod is stopped.
# AGENT_RESTART_DELAY: seconds to wait between restarts (default 5).
AGENT_RESTART_DELAY="${AGENT_RESTART_DELAY:-5}"
AGENT_PID=""

echo "[start] Launching LiveKit agent worker (watchdog, restart_delay=${AGENT_RESTART_DELAY}s)..."
cd /app
while true; do
    python agent.py start &
    AGENT_PID=$!
    wait "${AGENT_PID}"
    EXIT_CODE=$?
    AGENT_PID=""
    # Exit codes 0 (clean shutdown) and 130/143 (SIGINT/SIGTERM) are intentional.
    if [[ ${EXIT_CODE} -eq 0 || ${EXIT_CODE} -eq 130 || ${EXIT_CODE} -eq 143 ]]; then
        echo "[start] Agent exited cleanly (code=${EXIT_CODE}). Shutting down."
        break
    fi
    echo "[start] Agent exited unexpectedly (code=${EXIT_CODE}). Restarting in ${AGENT_RESTART_DELAY}s..."
    sleep "${AGENT_RESTART_DELAY}"
done
