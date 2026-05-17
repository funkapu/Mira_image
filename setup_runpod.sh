#!/bin/bash
set -e
echo "[setup] Recovery starting..."

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set" && exit 1
fi

echo "[setup] Downloading AWQ model..."
huggingface-cli download Funk888/ultravox-thai-qwen35 \
    --include "merged-full-v2-awq-sym/*" \
    --local-dir /workspace/models/ultravox \
    --local-dir-use-symlinks False

echo "[setup] Downloading FP16 (optional, for re-quantization)..."
huggingface-cli download Funk888/ultravox-thai-qwen35 \
    --include "merged-full-v2/*" \
    --local-dir /workspace/models/ultravox \
    --local-dir-use-symlinks False || echo "FP16 skipped"

echo "[setup] Verifying..."
ls /workspace/models/ultravox/merged-full-v2-awq-sym/ | wc -l
echo "[setup] ✓ Recovery complete"
echo "[setup] Run: bash /app/start_agent.sh"
