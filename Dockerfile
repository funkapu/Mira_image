# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────
# MindMirror / Mira voice agent — runtime image
#
# Base: runpod/pytorch with CUDA 12.4 + Python 3.11 (matches the working
# Pod environment: Ubuntu 22.04, Python 3.11.15, A100 80GB).
#
# Models (~71 GB) are NOT baked into the image — they are downloaded at
# runtime by setup_runpod.sh onto the persistent /workspace volume.
# Secrets (HF_TOKEN, LIVEKIT_*, MINIMAX_API_KEY, etc.) are env vars at
# runtime, never built into the image.
# ──────────────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ── System deps ───────────────────────────────────────────────────────
# - git/curl: setup_runpod.sh + huggingface-cli downloads
# - ffmpeg/libsndfile1: livekit-agents audio I/O + librosa/av decoding
# - ca-certificates: TLS for HF + LiveKit Cloud
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────
# Copy requirements first so unrelated code edits don't bust the
# pip-install cache layer.
COPY requirements.runpod.txt /app/requirements.runpod.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.runpod.txt

# ── App code ──────────────────────────────────────────────────────────
# Everything not in .dockerignore comes in here.
COPY . /app

# Make boot scripts executable (they are bash, but be explicit).
RUN chmod +x /app/start_agent.sh /app/setup_runpod.sh 2>/dev/null || true

# vLLM listens on :5000; the LiveKit agent connects out to LiveKit Cloud.
EXPOSE 5000

# start_agent.sh:
#   1. Sanity-checks model weights on /workspace (run setup_runpod.sh
#      first on a fresh pod).
#   2. Launches vLLM on localhost:5000.
#   3. Waits for /v1/models then exec's the LiveKit agent worker.
CMD ["bash", "/app/start_agent.sh"]
