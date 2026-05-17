"""Mira Voice Agent v3 — LiveKit 1.x + Whisper Turbo Thai + Mira LangGraph + Cartesia

Run locally (requires LIVEKIT_* + CARTESIA_API_KEY in .env):
    python -m voice.mira_agent_v3 dev

Deploy to Modal:
    modal deploy voice/modal_deploy.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    ModelSettings,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
    stt as lk_stt,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.plugins import cartesia, silero

import re

from mira_graph.graph import mira_graph
from voice.stt_whisper import WhisperTurboThaiSTT

_MD_RE = re.compile(r"(\*{1,3}|_{1,3}|`{1,3}|~~|#{1,6}\s?|>\s?|\[([^\]]*)\]\([^)]*\))")

def _strip_markdown(text: str) -> str:
    """Remove markdown symbols that TTS would read aloud as punctuation."""
    text = _MD_RE.sub(r"\2", text)
    return re.sub(r" {2,}", " ", text).strip()


# ── Dummy LLM — prevents pipeline from skipping llm_node when no real LLM set ─
class _PassthroughLLM(llm.LLM):
    """Placeholder LLM; MiraAgent.llm_node override handles all generation."""

    async def chat(self, chat_ctx, **kwargs):
        raise NotImplementedError("llm_node override should handle this")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("cerebras").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logger = logging.getLogger("mira-v3")

# ── Singleton Whisper instance (loaded once per worker process) ───────────────

_whisper: Optional[WhisperTurboThaiSTT] = None


def get_whisper(device: str | None = None) -> WhisperTurboThaiSTT:
    global _whisper
    if _whisper is None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper = WhisperTurboThaiSTT(device=device)
    return _whisper


# ── LiveKit STT adapter (1.5.x API) ──────────────────────────────────────────

class WhisperThaiSTT(lk_stt.STT):
    """LiveKit STT plugin wrapping WhisperTurboThaiSTT."""

    def __init__(self):
        super().__init__(
            capabilities=lk_stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._whisper = get_whisper()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> lk_stt.SpeechEvent:
        # Normalize to list[AudioFrame]
        frames: list[rtc.AudioFrame] = buffer if isinstance(buffer, list) else [buffer]

        sample_rate = frames[0].sample_rate if frames else 16000
        parts = [
            np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0
            for f in frames
        ]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        # Resample to 16 kHz (Whisper native)
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, 16000, sample_rate)

        text = await self._whisper.transcribe(audio, sample_rate=16000)

        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                lk_stt.SpeechData(language="th", text=text, confidence=1.0)
            ],
        )


# ── Mira state factory ────────────────────────────────────────────────────────

def _initial_state() -> dict:
    return {
        "user_text": None,
        "user_audio_b64": None,
        "messages": [],
        "phase": "S1_RAPPORT",
        "phase_turn_count": 0,
        "total_turns": 0,
        "parallel_results": [],
        "crisis_detected": False,
        "mood_score": None,
        "sentiment": None,
        "topic": None,
        "rag_results": [],
        "user_name": None,
        "main_concern": None,
        "automatic_thought": None,
        "distortion": None,
        "mira_response": None,
        "should_transition": False,
        "next_phase": None,
        "_happy_path": False,
        "_transition_requested": False,
        "_transition_reason": None,
        "_crisis_severity": None,
        "_voice_mode": True,
        "supervisor_empathy": None,
        "supervisor_belief": None,
        "supervisor_reflection": None,
        "supervisor_strategy": None,
        "supervisor_encouragement": None,
    }


# ── Mira LiveKit Agent (1.5.x: override llm_node) ────────────────────────────

class MiraAgent(Agent):
    """Drives Mira's CBT LangGraph behind LiveKit's pipeline via llm_node override."""

    def __init__(self):
        super().__init__(
            instructions="คุณคือพี่มิร่า เพื่อนคู่ใจ AI ที่ใช้ CBT framework",
        )
        self._mira_state = _initial_state()

    async def on_enter(self) -> None:
        await self.session.say("สวัสดีค่ะ พี่มิร่านะคะ วันนี้รู้สึกยังไงบ้างคะ")

    # Override llm_node: receives the current ChatContext after STT, returns Thai text.
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> str:
        # Pull the last user message from LiveKit's chat context
        msgs = chat_ctx.messages()
        user_msg = msgs[-1] if msgs else None
        text = (user_msg.text_content if user_msg else "") or ""

        if len(text.strip()) < 2:
            return ""

        # Session already ended — don't re-run LangGraph
        if self._mira_state.get("phase") == "END":
            logger.info("[Mira] Post-END utterance ignored: %s", text)
            return "วันนี้จบแล้วนะคะ ถ้าอยากคุยใหม่ เริ่ม session ใหม่ได้เลยค่ะ"

        logger.info("[Mira] User: %s", text)

        self._mira_state["user_text"] = text
        self._mira_state["messages"] = list(self._mira_state["messages"]) + [
            {"role": "user", "content": text}
        ]
        self._mira_state["parallel_results"] = []

        new_state = await mira_graph.ainvoke(self._mira_state)
        self._mira_state.update(new_state)

        response: str = _strip_markdown(self._mira_state.get("mira_response") or "")
        logger.info("[Mira] Response: %s", response)

        if self._mira_state.get("crisis_detected"):
            logger.warning("[Mira] Crisis protocol triggered")

        if self._mira_state.get("phase") == "END":
            logger.info("[Mira] Session ended")

        return response

    async def on_user_message(self, text: str) -> str:
        """Convenience wrapper for offline tests — bypasses LiveKit pipeline."""
        self._mira_state["user_text"] = text
        self._mira_state["messages"] = list(self._mira_state["messages"]) + [
            {"role": "user", "content": text}
        ]
        self._mira_state["parallel_results"] = []

        new_state = await mira_graph.ainvoke(self._mira_state)
        self._mira_state.update(new_state)
        return self._mira_state.get("mira_response") or ""


# ── LiveKit entrypoint (one per room) ─────────────────────────────────────────

async def _warm_ultravox():
    """Ping Ultravox endpoint so it's warm before the first user speaks."""
    ultravox_url = os.environ.get("ULTRAVOX_URL", "").rstrip("/")
    if not ultravox_url:
        return
    import httpx
    try:
        logger.info("[WARM] Pinging Ultravox cold start...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{ultravox_url}/v1/chat/completions",
                json={"model": os.environ.get("ULTRAVOX_MODEL", "mira-cbt"),
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 1},
                headers={"Authorization": "Bearer dummy-key"},
            )
        logger.info("[WARM] Ultravox ready (status %s)", r.status_code)
    except Exception as e:
        logger.warning("[WARM] Ultravox ping failed: %s", e)


async def entrypoint(ctx: JobContext):
    logger.info("New session — room: %s", ctx.room.name)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Warm Ultravox in background — don't block greeting
    asyncio.create_task(_warm_ultravox())

    tts = cartesia.TTS(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice=os.environ.get("CARTESIA_VOICE_ID", "ccc7bb22-dcd0-42e4-822e-0731b950972f"),
        language="th",
        model="sonic-3",
        emotion=["Calm", "Sympathetic", "Affectionate", "Peaceful"],
        speed=0.8,
    )

    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=0.9),
        stt=WhisperThaiSTT(),
        llm=_PassthroughLLM(),
        tts=tts,
    )

    mira = MiraAgent()

    await session.start(
        agent=mira,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _prewarm(proc):
    get_whisper()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=_prewarm,
        )
    )
