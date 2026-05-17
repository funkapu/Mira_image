# MindMirror / Mira — Pod Recovery Guide

This document is the source of truth for re-bootstrapping the Mira voice agent
on a fresh RunPod GPU pod from the published Docker image.

> **Why this exists.** A force-push to `main` once removed the infrastructure
> files (Dockerfile, workflow, this guide) because they were excluded by
> `.dockerignore` and so were not present inside the running container.
> If you are reading this, you have at least the image. Follow this document
> to recover the rest.

---

## TL;DR

```bash
# 1. Pull the latest image
docker pull ghcr.io/funkapu/mira_image:latest

# 2. On a fresh Pod, attach a persistent volume mounted at /workspace,
#    then ONCE only:
bash /app/setup_runpod.sh         # ~30 min, downloads ~71 GB of weights

# 3. Set required env vars (see below), then boot the agent:
bash /app/start_agent.sh
```

---

## Image

| Field | Value |
|---|---|
| Registry | `ghcr.io` |
| Image | `ghcr.io/funkapu/mira_image` |
| Tags | `latest`, `sha-<commit>`, `nsc-demo-<YYYYMMDD>` |
| Base | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Python | 3.11 |
| CUDA | 12.4 (driver compat ≥12.4) |
| Pulled size | ~7–9 GB compressed |
| Built by | `.github/workflows/docker-build.yml` (auto on push to `main`) |

The image contains **only code + Python deps**. Models live on a persistent
volume; secrets live in env vars.

---

## Hardware requirements

| Resource | Min | Notes |
|---|---|---|
| GPU | 1× A100 80 GB | Qwen3-32B bf16 + audio_tower + KV cache |
| Volume | 100 GB persistent at `/workspace` | Models alone are ~71 GB |
| RAM | 64 GB | vLLM + faster-whisper + LiveKit workers |
| CUDA driver | ≥12.4 | Image base is 12.4; driver can be newer |

---

## Required environment variables

Set these in the Pod template **before** starting the container. Missing any
of them will surface as an early failure in `start_agent.sh`.

| Name | Purpose | Example |
|---|---|---|
| `HF_TOKEN` | Hugging Face access for model download | `hf_xxx` |
| `LIVEKIT_URL` | LiveKit Cloud project URL | `wss://your-proj.livekit.cloud` |
| `LIVEKIT_API_KEY` | LiveKit Cloud API key | `APIxxxx` |
| `LIVEKIT_API_SECRET` | LiveKit Cloud API secret | `secretxxxx` |
| `MINIMAX_API_KEY` | MiniMax TTS (default TTS provider) | `eyJhbG...` |

### Optional / pre-set defaults

| Name | Default | What it controls |
|---|---|---|
| `WORKSPACE` | `/workspace` | Where models live |
| **`ULTRAVOX_URL`** | **`http://localhost:5000`** | vLLM endpoint. **NO `/v1` suffix** — `llm.py` appends `/v1/chat/completions` itself. Setting `http://...:5000/v1` produces double-`/v1` 404s. |
| `ULTRAVOX_MODEL` | `mira-cbt` | Served name passed to vLLM |
| `ULTRAVOX_API_KEY` | `dummy-key` | vLLM is local + open; any non-empty value is accepted |
| `USE_REAL_ULTRAVOX` | `true` | If `false`, falls back to Cerebras gpt-oss-120b |
| `MIRA_TTS` | `minimax` | `minimax` or `cartesia` |
| `MIRA_AGENT_VERSION` | `v4` | `v4` or `v5` (cascading) |
| `CARTESIA_API_KEY` | — | Required only if `MIRA_TTS=cartesia` |
| `CARTESIA_VOICE_ID` | — | Optional override |
| `MINIMAX_VOICE_ID` | `socialmedia_female_2_v1` | MiniMax voice id |

> **Common pitfall.** A pod template inherited from an older deployment
> sometimes has `ULTRAVOX_URL` pointing to a Modal endpoint
> (`https://...modal.run`). That URL is no longer guaranteed to be live and
> will silently fail with `HTTPStatusError 68ms` →  `[UV-STREAM ⚠️]` →
> Mira returns the canned fallback `อืม เล่าต่อได้เลยนะคะ`. Always confirm
> `ULTRAVOX_URL=http://localhost:5000` before reporting an issue.

---

## Boot sequence (fresh pod, first time)

```bash
# A. Pull image (RunPod template usually does this automatically)
docker pull ghcr.io/funkapu/mira_image:latest

# B. Open a shell in the container, then:

# B1. Verify GPU is visible
nvidia-smi

# B2. ONE-TIME model download (skip on subsequent boots — volume retains weights)
bash /app/setup_runpod.sh
# Expect:
#   - ~30 min on a clean pod
#   - ~71 GB written to /workspace/models/
#   - Idempotent: re-running just verifies & skips

# B3. Set env vars (or set them in the Pod template / RunPod UI)
export HF_TOKEN=...
export LIVEKIT_URL=...
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
export MINIMAX_API_KEY=...

# B4. Start the agent (and vLLM) — foreground
bash /app/start_agent.sh
```

`start_agent.sh` will:
1. Sanity-check that `merged-full-v2/config.json` and the Whisper CT2 weights
   exist (fail fast if `setup_runpod.sh` was skipped).
2. `vllm serve …merged-full-v2 --port 5000 --max-model-len 4096 …` in the
   background, logging to `/tmp/vllm.log`.
3. Poll `http://localhost:5000/v1/models` for up to 600 s (cold load is
   ~60–120 s).
4. `exec python agent.py start` — connects to LiveKit Cloud as a worker and
   waits for room dispatches.

---

## Boot sequence (returning pod, weights already on volume)

```bash
bash /app/start_agent.sh
```

That's it. Skip `setup_runpod.sh`. Models persist on the volume across
container restarts — the cold path is `vllm serve` weight loading
(~60–120 s) plus the one-time torch CUDA init.

---

## Expected timings

| Stage | Cold pod | Warm pod (volume reattached) |
|---|---|---|
| Image pull | 2–5 min | 0 s (cached) |
| `setup_runpod.sh` (model dl) | ~30 min | 0 s (skipped) |
| `vllm serve` weight load | 60–120 s | 60–120 s |
| First STT/LLM/TTS round-trip | <4 s end-to-end | <4 s |

---

## Verifying a healthy boot

```bash
# 1. vLLM listening on 5000
curl -fsS http://localhost:5000/v1/models | head -c 200
# Expect JSON containing "mira-cbt"

# 2. Agent connected to LiveKit
grep -E "register|connected" /tmp/agent.log | tail -3
# Expect a "registered worker" line

# 3. Send a short chat completion and confirm < ~3 s round-trip
python3 - <<'PY'
import json, urllib.request, time
req = urllib.request.Request(
    "http://localhost:5000/v1/chat/completions",
    data=json.dumps({
        "model": "mira-cbt",
        "messages": [{"role":"user","content":"สวัสดีค่ะ"}],
        "max_tokens": 30,
    }).encode(),
    headers={"Content-Type":"application/json","Authorization":"Bearer dummy"},
)
t = time.time()
r = urllib.request.urlopen(req, timeout=60).read().decode()
print(f"OK {(time.time()-t)*1000:.0f} ms")
print(r[:300])
PY
```

---

## Known issues + troubleshooting

### Symptom: Mira always replies `"อืม เล่าต่อได้เลยนะคะ"`

The strategy fallback. The LLM call to vLLM failed. Common causes:

1. **`ULTRAVOX_URL` points to Modal / a remote URL** — should be
   `http://localhost:5000`. Check with:
   ```bash
   tr '\0' '\n' < /proc/$(pgrep -f 'agent.py')/environ | grep ULTRAVOX_URL
   ```
2. **vLLM not running.** `pgrep -af 'vllm serve'` should show one process.
3. **vLLM started but model not yet loaded.**
   `curl http://localhost:5000/v1/models` will 5xx until weight load finishes.

### Symptom: vLLM returns HTTP 400 `"maximum context length is 4096 tokens"`

The system prompt + supervisor blocks + history exceeded `--max-model-len`.
Two ways to fix:

- **Recommended.** Trim the system prompt. The English system prompt
  (`mira_system_v4_en_full.txt`-style content) is ~5–6 KB / ~1.5–2 K tokens
  vs the Thai one at ~19 KB / ~3.5–4 K tokens.
- **Alternative.** Bump `--max-model-len` in `start_agent.sh` to e.g. 8192
  and `--gpu-memory-utilization` to 0.90. Costs proportional KV-cache RAM.

### Symptom: vLLM startup `ValueError: ... estimated maximum model length is 4800`

KV cache memory is too tight relative to `--max-model-len`. Either lower
`--max-model-len` or raise `--gpu-memory-utilization`.

### Symptom: `pkill -f "python agent.py"` also kills vLLM

`start_agent.sh` runs vLLM as a child of the boot script and traps `EXIT` to
clean it up. To restart only the agent without bouncing vLLM, run vLLM
detached separately:

```bash
nohup vllm serve … > /tmp/vllm.log 2>&1 &
disown
```

Then `start_agent.sh`'s `trap` won't reach it.

### Symptom: GHCR pull denied

The image is published as a **private** GHCR package by default. Either
make the package public (recommended for demo pods) or supply a PAT:

```bash
echo $GHCR_PAT | docker login ghcr.io -u <your-user> --password-stdin
```

---

## Files in this image

| Path | Role |
|---|---|
| `/app/agent.py` | LiveKit agent entrypoint |
| `/app/mira_graph/` | LangGraph nodes (counsellor, judge, crisis, supervisors) |
| `/app/mira_graph/prompts/` | System & supervisor prompts (`.txt`) |
| `/app/voice/` | LiveKit agent variants (`mira_agent_v4.py`, `mira_agent_v5_cascading.py`) |
| `/app/rag/` | RAG retrieval + technique recommender |
| `/app/start_agent.sh` | Boot script (vLLM + agent) |
| `/app/setup_runpod.sh` | One-time model downloader |
| `/app/requirements.runpod.txt` | Pinned runtime deps |
| `/app/requirements.frozen.txt` | `pip freeze` of last known good pod |
| `/app/Dockerfile` | This image's build recipe |
| `/app/.dockerignore` | What stays out of the image |
| `/app/.github/workflows/docker-build.yml` | Auto-build & push on `main` |
| `/app/RECOVERY.md` | This document |

---

## Rebuild & re-publish

Push to `main` triggers `.github/workflows/docker-build.yml` automatically.
For a manual rebuild:

```bash
gh workflow run docker-build.yml
# or via GitHub UI: Actions → Build & push Docker image to GHCR → Run workflow
```

First build: ~15–20 min (no cache). Subsequent: ~5–8 min thanks to the
GHA build cache.
