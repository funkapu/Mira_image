"""Deploy Mira Voice Agent v4 (LiveKit + local faster-whisper) to Modal A10G.

Usage:
    modal deploy voice/modal_deploy.py

Secrets required:
    livekit-secret        — LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    cartesia-secret       — CARTESIA_API_KEY, CARTESIA_VOICE_ID
    cerebras-secret       — CEREBRAS_API_KEY
    modal-ultravox-url    — ULTRAVOX_URL, ULTRAVOX_MODEL, ULTRAVOX_API_KEY, USE_REAL_ULTRAVOX
"""
import modal

APP_NAME = "mira-voice-agent"

# Volume for CT2 model cache — persists across cold starts
whisper_volume = modal.Volume.from_name("mira-whisper-cache", create_if_missing=True)
CACHE_PATH = "/cache"

agent_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .env({"PYTHONPATH": "/app", "WHISPER_CACHE_DIR": f"{CACHE_PATH}/whisper-th-ct2"})
    .pip_install(
        "livekit-agents>=1.5.0,<2.0.0",
        "livekit-plugins-cartesia>=0.4.0",
        "livekit-plugins-silero>=0.7.0",
        "faster-whisper>=1.0.3",
        "ctranslate2>=4.0.0",
        "transformers>=4.46.0",
        "torch>=2.0.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "langgraph>=0.2.50",
        "langchain-core>=0.3.0",
        "cerebras-cloud-sdk>=1.20.0",
        "groq>=1.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "httpx",
        "huggingface-hub>=0.20.0",
        "openai>=1.50.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
    )
    .add_local_dir("mira_graph", remote_path="/app/mira_graph")
    .add_local_dir("voice", remote_path="/app/voice")
    .add_local_dir("rag", remote_path="/app/rag")
)

app = modal.App(APP_NAME, image=agent_image)


@app.function(
    gpu="A10G",
    volumes={CACHE_PATH: whisper_volume},
    secrets=[
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("cartesia-secret"),
        modal.Secret.from_name("cerebras-secret"),
        modal.Secret.from_name("modal-ultravox-url"),
    ],
    timeout=3600,
    scaledown_window=300,
    min_containers=0,
    max_containers=5,
)
def run_agent():
    import sys
    sys.path.insert(0, "/app")

    from livekit.agents import WorkerOptions, cli
    from voice.mira_agent_v4 import entrypoint, _prewarm
    from voice.stt_fast_whisper import FastWhisperSTT

    # Pre-load faster-whisper CT2 model (converts from HF on first run, ~30s)
    FastWhisperSTT.get(device="cuda")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm))


@app.local_entrypoint()
def main():
    run_agent.remote()
