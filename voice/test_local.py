"""Local smoke test for the voice agent (no LiveKit room needed).

Requirements:
    - .env with LIVEKIT_*, CARTESIA_API_KEY, CEREBRAS_API_KEY
    - GPU available (or change device='cpu' in WhisperTurboThaiSTT)
    - pip install -r voice/requirements.txt -r requirements.txt

Usage:
    # Option A — run as LiveKit dev worker (connects to cloud, shows in playground)
    python -m voice.mira_agent_v3 dev

    # Option B — minimal offline unit tests (no LiveKit)
    python voice/test_local.py

Connect (Option A) via:
    https://agents-playground.livekit.io
    Enter your LIVEKIT_URL and a room token from the LiveKit dashboard.
"""
import asyncio
import io
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def test_whisper_cpu():
    """Verify Whisper loads and transcribes a synthetic sine wave (CPU fallback)."""
    import numpy as np
    from voice.stt_whisper import WhisperTurboThaiSTT

    print("=== Whisper STT smoke test (CPU) ===")
    stt = WhisperTurboThaiSTT(device="cpu")

    # 1 second of silence — should return empty / short string
    silence = np.zeros(16000, dtype=np.float32)
    result = await stt.transcribe(silence, sample_rate=16000)
    print(f"Silence transcription: '{result}'")
    print("OK Whisper loaded and ran\n")


async def test_mira_graph():
    """Verify LangGraph responds to a single Thai turn (requires CEREBRAS_API_KEY)."""
    from dotenv import load_dotenv
    load_dotenv()

    from voice.mira_agent_v3 import MiraAgent
    print("=== Mira LangGraph smoke test ===")
    agent = MiraAgent()
    response = await agent.on_user_message("สวัสดีครับ")
    print(f"Mira: {response}")
    assert response, "Expected non-empty Thai response"
    print("OK LangGraph responded\n")


async def main():
    await test_whisper_cpu()
    await test_mira_graph()
    print("All smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
