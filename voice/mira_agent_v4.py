"""Mira Voice Agent v4 — Modal STT + Ultravox native audio counsellor

Pipeline per turn:
  AudioBuffer (post-VAD) → Modal Whisper Thai (HTTP, ~200ms) → text → supervisors (400ms, parallel)
                         → OGG base64 → Ultravox counsellor (1700ms, native audio)

STT: Modal endpoint (WHISPER_STT_URL) — no local model download needed
     deploy: modal deploy voice/modal_stt.py

Run locally:
    python -m voice.mira_agent_v4 dev
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import time
from pathlib import Path

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
from livekit.plugins import cartesia, minimax, silero

from mira_graph.llm import (
    ReasoningStreamFilter,
    apply_voice_mode,
    call_ultravox_audio_stream,
    call_ultravox_pure_stream,
    clean_response,
)
from mira_graph.nodes._phase_base import _load_prompt
from mira_graph.nodes.aggregate import aggregate_node
from mira_graph.nodes.counsellor import _build_supervisor_block, _extract_name
from mira_graph.nodes.at_extract import at_supervisor
from mira_graph.nodes.crisis import crisis_check_node
from mira_graph.nodes.intake import (
    build_intake_directive,
    build_intake_response,
    risk_level_label,
    score_prior_answer,
)
from mira_graph.nodes.judge import judge_node
from mira_graph.nodes.output_filter import _strip_leaks
from mira_graph.nodes.phq9 import phq9_supervisor
from mira_graph.nodes.supervisors import (
    belief_supervisor,
    empathy_supervisor,
    encouragement_supervisor,
    reflection_supervisor,
    strategy_supervisor,
)
from voice.stt_fast_whisper import FastWhisperSTT

_MD_RE = re.compile(r"(\*{1,3}|_{1,3}|`{1,3}|~~|#{1,6}\s?|\[([^\]]*)\]\([^)]*\))")
_BLOCKQUOTE_RE = re.compile(r"(?:^|\n)>\s?")
# α-fix: strip leading "(label)" annotations that thinking mode leaks as
# meta-commentary (e.g. "(ถามสุขภาพจิต) ถ้าให้ตั้งเลข..."). Caps internal
# content at 40 chars to avoid eating real parentheticals like "(เห็นใจ)".
_LEADING_PAREN_RE = re.compile(r"^\s*\([^)]{0,40}\)\s*")


def _strip_markdown(text: str) -> str:
    text = _MD_RE.sub(r"\2", text)
    text = _BLOCKQUOTE_RE.sub("\n", text)
    return re.sub(r" {2,}", " ", text).strip()


def _silence_audio_b64() -> str | None:
    """Generate 1-second silence OGG base64 for audio_tower warmup.
    Without this, the first user audio call costs ~18s for CUDA-graph compile.
    """
    try:
        import soundfile as sf
        silence = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
        buf = io.BytesIO()
        sf.write(buf, silence, 16000, format="OGG", subtype="VORBIS")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logging.getLogger("mira-v4").warning("[WARM-AUDIO] silence gen failed: %s", e)
        return None


def _encode_ogg(frames: list[rtc.AudioFrame]) -> str | None:
    """Encode AudioFrame list to OGG Vorbis base64 string."""
    try:
        import soundfile as sf
        if not frames:
            return None
        sr = frames[0].sample_rate
        parts = [np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0 for f in frames]
        audio = np.concatenate(parts)
        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="OGG", subtype="VORBIS")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logging.getLogger("mira-v4").warning("[STT] OGG encode failed: %s", e)
        return None


class _PassthroughLLM(llm.LLM):
    async def chat(self, chat_ctx, **kwargs):
        raise NotImplementedError("llm_node override handles generation")


load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("cerebras").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logger = logging.getLogger("mira-v4")


class ThaiSTT(lk_stt.STT):
    """LiveKit STT — local faster-whisper on same GPU, no HTTP call."""

    def __init__(self):
        super().__init__(capabilities=lk_stt.STTCapabilities(streaming=False, interim_results=False))
        self.last_audio_b64: str | None = None

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> lk_stt.SpeechEvent:
        frames: list[rtc.AudioFrame] = buffer if isinstance(buffer, list) else [buffer]
        sr = frames[0].sample_rate if frames else 16000

        parts = [np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768.0 for f in frames]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        if sr != 16000:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, 16000, sr).astype(np.float32)

        # Reject silent/echo audio — avoids 20s whisper hallucination on TTS echo
        duration_s = len(audio) / 16000
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
        if duration_s < 0.4 or rms < 0.002:
            logger.info("[STT] skip — dur=%.2fs rms=%.4f", duration_s, rms)
            return lk_stt.SpeechEvent(
                type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lk_stt.SpeechData(language="th", text="", confidence=0.0)],
            )

        # OGG encode + STT in parallel
        # Reject hallucinated repetition e.g. "ครับ ครับ ครับ ครับ"
        # (faster-whisper loops on echo audio even with temperature=0)

        ogg_task = asyncio.create_task(asyncio.to_thread(_encode_ogg, frames))
        import torch
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        whisper = FastWhisperSTT.get(device=_dev)
        text = await whisper.transcribe(audio, 16000)
        self.last_audio_b64 = await ogg_task

        if _should_skip_transcript(text):
            return lk_stt.SpeechEvent(
                type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lk_stt.SpeechData(language="th", text="", confidence=0.0)],
            )

        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[lk_stt.SpeechData(language="th", text=text, confidence=1.0)],
        )


def _should_skip_transcript(text: str) -> bool:
    """Return True if transcript looks like a Whisper hallucination."""
    if not text or len(text.strip()) < 2:
        return True

    words = text.split()
    if not words:
        return True

    # Gibberish — many unique short words (random word soup)
    if len(words) > 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.8:
            logger.info("[STT] skip — gibberish detected: %s", text[:60])
            return True

    # Repetition ratio — single word dominating output
    if len(words) > 3:
        top_freq = max(words.count(w) for w in set(words)) / len(words)
        if top_freq > 0.40:
            logger.info("[STT] skip — repetition %.0f%%: %s", top_freq * 100, text[:60])
            return True

    # Known hallucination substrings
    _HALLUCINATION_PHRASES = [
        "ได้รับการที่จะ",
        "ว่าว่าว่า",
        "อ่ะ อ่ะ อ่ะ",
        "ครับ ครับ ครับ",
        "ค่ะ ค่ะ ค่ะ",
    ]
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in text:
            logger.info("[STT] skip — hallucination phrase: %s", text[:60])
            return True

    return False


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
        "quick_response": None,
        # PHQ-9 supervisor state — sticky across turns to avoid repeat probes
        "phq9_directive": None,
        "phq9_detected_symptom": None,
        "phq9_chosen_probe": None,
        "phq9_probes_asked": [],
        # AT extraction state — confidence for sticky overwrite logic
        "_at_confidence": None,
        "_at_asking_count": 0,
        # Covert Intake (PHQ-9) — DISABLED for NSC demo.
        # State pinned past Q9 with neutral score so any residual gating
        # logic treats intake as already-completed, non-crisis.
        "session_type": "therapy",  # HARDCODED
        "intake_started": False,
        "intake_pending_score": False,
        "intake_smalltalk_count": 0,
        "phq9_step": 9,
        "phq9_scores": {},
        "phq9_total": 0,
        "phq9_q9_score": 0,
    }


class MiraAgent(Agent):
    def __init__(self, stt_plugin: ThaiSTT):
        super().__init__(instructions="คุณคือพี่มิร่า เพื่อนคู่ใจ AI ที่ใช้ CBT framework")
        self._mira_state = _initial_state()
        self._stt = stt_plugin
        # Chunk buffer — collect VAD chunks, wait for silence, process once
        self._chunk_buffer: list[str] = []
        self._buffer_task: asyncio.Task | None = None
        self._buffer_timeout: float = 1.5  # seconds of silence before processing
        self._is_processing: bool = False   # prevent concurrent turns
        self._recent_user_texts: list[str] = []  # safety net: track last 10 user turns
        self._crisis_active: bool = False         # sticky flag — once set, never clears in session

    async def on_enter(self) -> None:
        # Default greeting — first user reply will trigger intake flow
        # which asks small-talk Q (มาจากไหนคะ) on Mira's first response.
        await self.session.say("สวัสดีค่ะ พี่มิร่านะคะ คุณมาจากจังหวัดไหนคะ?")
        # Mark that initial greeting was sent — intake will skip the duplicate greet
        self._mira_state["intake_started"] = True

    def _safe_say(self, text: str) -> bool:
        """Wrapper around session.say() that swallows the RuntimeError raised
        when the agent's activity context has ended (user disconnected mid-stream).
        Returns True on success, False if session is gone (caller should stop streaming).
        """
        try:
            self.session.say(text)
            return True
        except RuntimeError as e:
            if "no activity context" in str(e):
                logger.info("[SAY] session ended — dropping output: %s", text[:40])
                return False
            raise

    def _detect_crisis(self, text: str) -> list[str]:
        """Return list of matched crisis signals in text. Empty list = no crisis."""
        matches = [s for s in _CRISIS_SIGNALS if s in text]
        # F2: catch standalone "ตาย" only with self-harm context.
        if _TAY_CONTEXT_RE.search(text):
            matches.append("ตาย-context")
        return matches

    def _crisis_cumulative_check(self, current_text: str) -> list[str]:
        """Scan current text + last 4 user turns. Catches crises split across chunks."""
        haystack = " ".join(self._recent_user_texts[-4:]) + " " + current_text
        return [s for s in _CRISIS_SIGNALS if s in haystack]

    def _trigger_crisis_override(self, source: str, matched: list[str]) -> None:
        """Sticky-set the flag, log, speak override, clear buffer. Safe to call repeatedly."""
        was_active = self._crisis_active
        self._crisis_active = True
        # Reset phase so we never end during crisis
        if self._mira_state.get("phase") == "END":
            self._mira_state["phase"] = "S1_RAPPORT"
            self._mira_state["crisis_detected"] = True
        logger.warning(
            "[CRISIS 🚨] %s — matched=%s, was_active=%s, recent=%s",
            source, matched[:3], was_active, [t[:30] for t in self._recent_user_texts[-3:]],
        )
        # P-D: redact matched signals from recent_user_texts so the cumulative
        # check cannot re-match them on later benign turns.
        for _sig in matched:
            self._recent_user_texts = [
                t.replace(_sig, "[redacted]") for t in self._recent_user_texts
            ]
        # Drop any pending buffered text — crisis takes priority
        self._chunk_buffer.clear()
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        self._safe_say(_CRISIS_OVERRIDE)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> str:
        msgs = chat_ctx.messages()
        user_msg = msgs[-1] if msgs else None
        text = (user_msg.text_content if user_msg else "") or ""

        if len(text.strip()) < 2:
            return ""

        # ── CRITICAL SAFETY: crisis check IMMEDIATELY, before anything else ──
        # Track every chunk so cumulative check sees the full picture.
        self._recent_user_texts.append(text.strip())
        if len(self._recent_user_texts) > 10:
            self._recent_user_texts = self._recent_user_texts[-10:]

        logger.info(
            "[SAFETY CHECK] _crisis_active=%s recent=%s",
            self._crisis_active, [t[:25] for t in self._recent_user_texts[-3:]],
        )

        matched = self._detect_crisis(text)
        # P-A: skip cumulative scan once crisis is already active. The cumulative
        # check re-matches stale crisis words from _recent_user_texts and locks
        # the conversation in override-mode forever. Fresh per-chunk match still
        # re-fires the 1323 override (intended).
        if not matched and not self._crisis_active:
            matched = self._crisis_cumulative_check(text)

        if matched:
            # Bypass Ultravox entirely. Hardcoded override → TTS.
            self._trigger_crisis_override("llm_node", matched)
            return ""

        # ── Post-END handling: silently re-open session, don't say goodbye ──
        # If user keeps talking after phase=END, just reset to therapy phase
        # and let Mira respond naturally — no canned "วันนี้จบแล้ว" message.
        if self._mira_state.get("phase") == "END":
            if self._crisis_active:
                logger.warning("[SAFETY NET 🚨] Blocked END — crisis active, re-processing")
                self._mira_state["phase"] = "S1_RAPPORT"
            else:
                logger.info("[Mira] Re-opening session after END for user input: %s", text[:40])
                self._mira_state["phase"] = "S4_WRAP"  # gentle, allow continued chat

        # ── Buffer chunk — wait for silence before processing ────────────────
        self._chunk_buffer.append(text.strip())
        logger.info("[BUFFER] +chunk: '%s' (total: %d)", text.strip()[:40], len(self._chunk_buffer))

        # Cancel previous timer and start fresh
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()

        self._buffer_task = asyncio.create_task(self._process_after_silence())

        # Return "" immediately — TTS will fire from _process_after_silence via session.say()
        return ""

    async def _process_after_silence(self) -> None:
        """Wait for silence, combine all buffered chunks, process once."""
        try:
            await asyncio.sleep(self._buffer_timeout)
        except asyncio.CancelledError:
            return  # new chunk arrived — don't process yet

        if not self._chunk_buffer:
            return

        # Prevent concurrent turns
        if self._is_processing:
            logger.info("[BUFFER] Already processing — dropping buffer")
            self._chunk_buffer.clear()
            return

        combined = " ".join(self._chunk_buffer)
        chunk_count = len(self._chunk_buffer)
        self._chunk_buffer.clear()

        logger.info("[BUFFER] Combined %d chunks: '%s'", chunk_count, combined[:80])
        await self._run_mira_turn(combined)

    async def _run_mira_turn(self, text: str) -> None:
        """Core processing — supervisors → counsellor → TTS."""
        self._is_processing = True
        _t_turn = time.perf_counter()
        _t_first_chunk = None
        _t_stream_open = None
        _t_stream_end = None
        _t_say = None
        try:
            logger.info("[Mira] User: %s", text)

            # Defense-in-depth: re-check crisis on combined buffered text.
            # llm_node already checks each chunk, but combined text might
            # surface a signal that spans chunks (e.g. "ไม่อยาก" + "อยู่").
            # P-A: same skip-when-active rule as in llm_node.
            matched = self._detect_crisis(text)
            if not matched and not self._crisis_active:
                matched = self._crisis_cumulative_check(text)
            if matched:
                logger.warning(
                    "[SAFETY CHECK in _run_mira_turn] crisis=%s recent=%s",
                    self._crisis_active, [t[:25] for t in self._recent_user_texts[-3:]],
                )
                self._trigger_crisis_override("_run_mira_turn", matched)
                return  # Skip Ultravox entirely

            self._mira_state["user_text"] = text
            self._mira_state["user_audio_b64"] = self._stt.last_audio_b64
            self._mira_state["messages"] = list(self._mira_state["messages"]) + [
                {"role": "user", "content": text}
            ]
            self._mira_state["parallel_results"] = []
            self._mira_state["quick_response"] = None
            for k in ("supervisor_empathy", "supervisor_belief", "supervisor_reflection",
                      "supervisor_strategy", "supervisor_encouragement"):
                self._mira_state[k] = None

            # ── Phase A: crisis + 5 supervisors + aggregate (~500ms) ─────────
            # NOTE: supervisors disabled for testing — Mira runs on system prompt + history only
            # try:
            #     crisis_result = await crisis_check_node(self._mira_state)
            #     self._mira_state.update(crisis_result)
            #
            #     sup_results = await asyncio.gather(
            #         empathy_supervisor(self._mira_state),
            #         belief_supervisor(self._mira_state),
            #         reflection_supervisor(self._mira_state),
            #         strategy_supervisor(self._mira_state),
            #         encouragement_supervisor(self._mira_state),
            #         return_exceptions=True,
            #     )
            #     for r in sup_results:
            #         if isinstance(r, dict):
            #             self._mira_state.update(r)
            #
            #     agg_result = await aggregate_node(self._mira_state)
            #     self._mira_state.update(agg_result)
            # except Exception as e:
            #     logger.exception("[Mira] supervisor phase FAILED: %s", e)
            #     self.session.say("ขอโทษนะคะ มีปัญหาชั่วคราว ลองพูดใหม่ได้เลยค่ะ")
            #     return

            # ── Goodbye detection — force S4_WRAP if user signals exit ───────
            # User saying "ไปก่อน / ลาก่อน / บาย / พอแค่นี้" should end session
            # gracefully instead of looping in S2 with "ไม่เป็นไรค่ะ" forever.
            if any(kw in text for kw in _GOODBYE_KEYWORDS):
                logger.info("[GOODBYE] detected in user_text=%r → force S4_WRAP", text[:50])
                self._mira_state["phase"] = "S4_WRAP"
                # Also flip session_type so we don't get stuck in intake mid-flow
                if self._mira_state.get("session_type") == "intake":
                    self._mira_state["session_type"] = "therapy"

            # ── Phase A: Intake mode (covert PHQ-9) ──────────────────────────
            # Hardcoded responses, NO Ultravox call. LoRA-baked behavior is too
            # strong to follow directives; we bypass it to guarantee PHQ-9 progression.
            if self._mira_state.get("session_type") == "intake":
                # 1) Score the previous PHQ-9 answer (if Mira asked one last turn)
                score_updates = await score_prior_answer(self._mira_state, text)
                self._mira_state.update(score_updates)

                # 2) Q9 crisis branch — user admitted self-harm thoughts
                if score_updates.get("_intake_q9_crisis"):
                    logger.warning("[INTAKE 🚨] Q9 score >= 1 — triggering crisis override")
                    self._trigger_crisis_override(
                        "intake_Q9", ["intake_Q9_self_harm"]
                    )
                    self._mira_state["session_type"] = "therapy"
                    self._mira_state["phase"] = "S1_RAPPORT"
                    return

                # 3) Generate hardcoded response (deterministic alternation)
                # Snapshot state BEFORE for debug — diagnose any Q-skip issue
                _pre_step = self._mira_state.get("phq9_step") or 0
                _pre_pending = bool(self._mira_state.get("intake_pending_score"))
                _pre_smalltalk = self._mira_state.get("intake_smalltalk_count") or 0
                response, response_updates = build_intake_response(self._mira_state)
                self._mira_state.update(response_updates)

                logger.info(
                    "[INTAKE] before: step=%d pending=%s smalltalk=%d | response=%r | updates=%s",
                    _pre_step, _pre_pending, _pre_smalltalk,
                    response[:80], response_updates,
                )

                # 4) Speak directly — bypass Ultravox entirely
                self._safe_say(response)

                # 5) Pivot to therapy if intake complete
                if response_updates.get("_intake_complete"):
                    total = self._mira_state.get("phq9_total") or 0
                    risk = risk_level_label(total)
                    logger.info("[INTAKE ✅] complete — total=%d risk=%s scores=%s",
                                total, risk, self._mira_state.get("phq9_scores"))
                    self._mira_state["session_type"] = "therapy"
                    self._mira_state["phase"] = "S1_RAPPORT"

                # Update message history for therapy continuity
                self._mira_state["mira_response"] = response
                self._mira_state["messages"] = list(self._mira_state["messages"]) + [
                    {"role": "assistant", "content": response}
                ]
                return  # Skip Ultravox stream entirely
            else:
                # ── Therapy mode: load CBT prompt + therapy-specific guards ──
                _phase_now = self._mira_state.get("phase", "S1_RAPPORT")
                system_prompt = _load_prompt(_phase_now, self._mira_state)
                # α-fix 2: drop the 12-distortion catalog from non-reframing phases.
                # The catalog is ~600 tokens and only matters in S2_EXPLORE / S3_REFRAME
                # where Mira actually names distortions. S1/S4 don't need it.
                if _phase_now not in ("S2_EXPLORE", "S3_REFRAME"):
                    _before = len(system_prompt)
                    system_prompt = re.sub(
                        r"=== 12 Cognitive Distortions[\s\S]*?(?=\n=== TONE:)",
                        "",
                        system_prompt,
                        count=1,
                    )
                    _after = len(system_prompt)
                    if _after < _before:
                        logger.info("[PROMPT-TRIM] phase=%s saved %d bytes (cut distortion catalog)",
                                    _phase_now, _before - _after)
                system_prompt += _build_supervisor_block(self._mira_state)

                # Guard: ห้าม hallucinate ชื่อถ้ายังไม่รู้ชื่อ user
                if not self._mira_state.get("user_name"):
                    system_prompt += (
                        "\n\n[หมายเหตุ: ยังไม่ทราบชื่อผู้ใช้ ห้ามเรียกชื่อหรือสมมติชื่อ "
                        "ใช้คำว่า 'คุณ' แทนจนกว่าผู้ใช้จะบอกชื่อ]"
                    )

                # Anti-loop: explicitly remind Mira what the user has already shared.
                # The chat history is in messages but LoRA-baked behavior often
                # ignores it and re-asks. This block flags it as a hard constraint.
                recent_user_msgs = [
                    (m.get("content") or "").strip()
                    for m in self._mira_state.get("messages", [])[-8:]
                    if isinstance(m, dict) and m.get("role") == "user"
                ]
                recent_user_msgs = [m for m in recent_user_msgs if len(m) > 5]
                if recent_user_msgs:
                    facts = "\n".join(f"  - {m[:140]}" for m in recent_user_msgs[-4:])
                    system_prompt += (
                        "\n\n=== ผู้ใช้บอกไปแล้ว (ห้ามถามซ้ำสิ่งที่ตอบแล้ว) ===\n"
                        f"{facts}\n"
                        "ห้ามถาม 'เกิดอะไรขึ้น' ถ้า user เล่าเหตุการณ์แล้ว\n"
                        "ห้ามถาม 'คิดอะไรอยู่' ถ้า user พูดความคิดแล้ว (เช่น 'ผมโง่' / 'เป็นคนล้มเหลว')\n"
                        "ถ้า user ตอบหงุดหงิด ('ก็บอกแล้ว' / 'ไง') แปลว่ากำลังถามซ้ำ — ขอโทษและไปขั้นถัดไป\n"
                    )

                system_prompt = apply_voice_mode(system_prompt, self._mira_state)

            # F1: fire at_supervisor concurrent with the Ultravox stream so it
            # can populate `automatic_thought` / `distortion` / `_at_confidence`
            # in time for judge_node to unblock S2→S3 progression. Runs in
            # parallel with the 2-3s stream — perceived latency cost = 0.
            _at_task: asyncio.Task | None = None
            try:
                _at_task = asyncio.create_task(at_supervisor(self._mira_state))
            except Exception as _e:
                logger.warning("[AT ⚠️] could not start at_supervisor: %s", _e)

            # A/B test: audio mode RESTORED — biodatlab Thai STT still ran in
            # the LiveKit chain so `text` is available for crisis check, log,
            # and text-mode fallback if audio yields nothing.
            audio_b64 = self._stt.last_audio_b64  # was None (forced text)
            history = list(self._mira_state.get("messages", []))[-4:]  # B1: trim from -6 to keep prompt under vLLM context budget
            self._stt.last_audio_b64 = None
            mode_label = "AUDIO" if audio_b64 else "TEXT"

            all_sentences: list[str] = []
            reason_filter = ReasoningStreamFilter()
            crisis_intercepted = False

            async def _consume_stream(_stream) -> bool:
                """Per-sentence cleanup pipeline. Mutates all_sentences + outer _t_first_chunk.
                Returns True if a crisis END was intercepted (caller should stop)."""
                nonlocal _t_first_chunk
                import re as _re
                async for sentence in _stream:
                    if not sentence.strip():
                        continue
                    sc = _re.sub(r"<think>?[\s\S]*?</think>", "", sentence, flags=_re.DOTALL)
                    sc = _re.sub(r"<think>?[\s\S]*$", "", sc, flags=_re.DOTALL)
                    sc = _re.sub(r"^[\s\S]*?</think>", "", sc, flags=_re.DOTALL)
                    _paren_match = _LEADING_PAREN_RE.match(sc)
                    if _paren_match:
                        logger.warning("[GUARD] stripped leading paren-label: %r", _paren_match.group(0))
                        sc = sc[_paren_match.end():]
                    sc = _strip_markdown(_strip_leaks(sc))
                    logger.info("[DEBUG] Raw sentence: %r", sentence[:100])
                    logger.info("[DEBUG] After strip: %r", sc[:100])
                    spoken = sc.strip()
                    if spoken and spoken.startswith((
                        "Okay,", "Alright,", "Let me", "The user", "First,",
                        "In CBT", "Looking", "I need", "Hmm,", "Right,", "<think",
                    )):
                        logger.warning("[GUARD] dropped English-prefix: %r", spoken[:80])
                        spoken = ""
                    elif spoken and not _re.search(r"[\u0e00-\u0e7f]", spoken):
                        logger.warning("[GUARD] dropped Thai-less output: %r", spoken[:80])
                        spoken = ""
                    logger.info("[DEBUG] After filter (spoken): %r", spoken[:100])
                    if not spoken:
                        continue
                    if _t_first_chunk is None:
                        _t_first_chunk = time.perf_counter()
                    if self._crisis_active and any(p in spoken for p in _END_PHRASES):
                        logger.warning("[SAFETY NET 🚨] Intercepted END in stream — replacing")
                        all_sentences.clear()
                        all_sentences.append(_CRISIS_OVERRIDE)
                        return True
                    all_sentences.append(spoken)
                    logger.info("[Mira] CHUNK: %s", spoken)
                return False

            try:
                _t_stream_open = time.perf_counter()
                if audio_b64:
                    stream = call_ultravox_audio_stream(system_prompt, audio_b64, history)
                else:
                    stream = call_ultravox_pure_stream(system_prompt, text, history)
                crisis_intercepted = await _consume_stream(stream)
                _t_stream_end = time.perf_counter()

                # A/B fallback: if audio yielded zero sentences, retry with text.
                # Audio mode can fail silently on bad mic / noise / empty whisper —
                # text fallback (biodatlab STT result) is the safety net.
                if not all_sentences and audio_b64 and not crisis_intercepted:
                    logger.warning("[AUDIO→TEXT] audio yielded 0 sentences — fallback to text mode (text=%r)", text[:60])
                    mode_label = "TEXT-fallback"
                    _t_first_chunk = None  # reset TTFT for text retry
                    _t_stream_open_2 = time.perf_counter()
                    text_stream = call_ultravox_pure_stream(system_prompt, text, history)
                    crisis_intercepted = await _consume_stream(text_stream)
                    _t_stream_end = time.perf_counter()

                # Combine all sentences into a single utterance and speak once.
                if all_sentences:
                    combined = " ".join(s.strip() for s in all_sentences if s.strip())
                    if combined:
                        _t_say = time.perf_counter()
                        if not self._safe_say(combined):
                            logger.info("[SAY] session ended — dropped combined response")

                # Post-stream: log reasoning block (debug telemetry)
                if reason_filter.reasoning:
                    logger.info(
                        "\n%s\n[MIRA REASONING]\n%s\n[MIRA SPOKEN]\n%s\n%s",
                        "=" * 60,
                        reason_filter.reasoning,
                        " ".join(all_sentences),
                        "=" * 60,
                    )

                # Latency summary — tagged with mode_label for A/B comparison.
                _now = time.perf_counter()
                def _ms(t):
                    return f"{(t - _t_turn) * 1000:.0f}ms" if t else "—"
                ttft = f"{(_t_first_chunk - _t_stream_open) * 1000:.0f}ms" if (_t_first_chunk and _t_stream_open) else "—"
                stream_total = f"{(_t_stream_end - _t_stream_open) * 1000:.0f}ms" if (_t_stream_end and _t_stream_open) else "—"
                logger.info(
                    "[LAT-%s] turn_total=%s setup=%s ttft=%s stream=%s say_at=%s sentences=%d",
                    mode_label, _ms(_now), _ms(_t_stream_open), ttft, stream_total, _ms(_t_say), len(all_sentences),
                )

            except Exception as e:
                logger.exception("[Mira] stream FAILED: %s", e)

            if not all_sentences:
                fallback = "อืม เล่าต่อได้เลยนะคะ"
                all_sentences = [fallback]
                self._safe_say(fallback)

            full_response = " ".join(all_sentences)

            # Name extraction
            if not self._mira_state.get("user_name"):
                name = _extract_name(text)
                if name:
                    self._mira_state["user_name"] = name
                    logger.info("[NAME] %s", name)

            # Update state + judge
            self._mira_state["mira_response"] = full_response
            self._mira_state["messages"] = list(self._mira_state["messages"]) + [
                {"role": "assistant", "content": full_response}
            ]

            # F1: collect at_supervisor result (started concurrently with stream).
            # 2s timeout is generous — typical Cerebras call is 200-600ms and
            # the stream itself took 2-3s so it should already be done.
            if _at_task is not None:
                try:
                    _at_result = await asyncio.wait_for(_at_task, timeout=2.0)
                    if isinstance(_at_result, dict):
                        self._mira_state.update(_at_result)
                        if _at_result.get("automatic_thought"):
                            logger.info(
                                "[AT ✅] thought=%r distortion=%s conf=%s",
                                (_at_result.get("automatic_thought") or "")[:60],
                                _at_result.get("distortion"),
                                _at_result.get("_at_confidence"),
                            )
                except (asyncio.TimeoutError, asyncio.CancelledError) as _e:
                    logger.warning("[AT ⚠️] timed out — automatic_thought stays unset")
                    if not _at_task.done():
                        _at_task.cancel()
                except Exception as _e:
                    logger.warning("[AT ⚠️] %s", _e)

            try:
                judge_result = await judge_node(self._mira_state)
                self._mira_state.update(judge_result)
                # Safety net: block judge from transitioning to END during crisis
                if self._crisis_active and self._mira_state.get("phase") == "END":
                    logger.warning("[SAFETY NET 🚨] Blocked judge → END during crisis")
                    self._mira_state["phase"] = "S1_RAPPORT"
                    self._mira_state["crisis_detected"] = True
            except Exception as e:
                logger.warning("[Mira] judge FAILED: %s", e)

            logger.info("[Mira] Response [%s]: %s", self._mira_state.get("phase"), full_response)

        finally:
            self._is_processing = False


_CRISIS_SIGNALS = [
    # Direct suicidal ideation
    "ทำร้ายตัวเอง", "ฆ่าตัวตาย", "อยากตาย", "จบชีวิต", "ปิดสวิตช์ชีวิต",
    # Methods (broad — "โดด" covers ตึก/น้ำ/คลอง)
    "โดด", "กรีด", "กินยา", "แขวนคอ",
    # Passive ideation
    "ไม่อยากอยู่", "ไม่อยากมีชีวิต", "อยู่ไปทำไม", "เป็นภาระ",
    "อยากหายไป", "ไม่อยากตื่น", "ไม่อยากหายใจ", "ไม่มีความหมาย",
]

# F2: context-aware standalone "ตาย" — fires only when preceded within ~12 chars
# by a self-harm cue. Avoids false positives on Thai figurative use like
# "หิวจะตาย" / "ตายล่ะ" / "ตายแน่". Catches STT-mangled phrases such as
# "กษัตริย์เพื่อให้ได้ตายเลย" via the "เพื่อให้" / "ให้ได้" cues.
_TAY_CONTEXT_RE = re.compile(
    r"(อยาก|ขอ|อยากให้|เพื่อให้|ให้ได้|รู้สึกอยาก|ไม่อยากอยู่|ผมจะ|ฉันจะ|หนูจะ).{0,12}ตาย"
)

_END_PHRASES = ["วันนี้จบแล้ว", "จบแล้วนะคะ", "เริ่ม session ใหม่"]

# User-initiated goodbye signals — when detected, force phase=S4_WRAP so Mira
# wraps up instead of looping in S2 with "ไม่เป็นไรค่ะ ถ้าพร้อมจะค่อยๆ..." forever.
_GOODBYE_KEYWORDS = [
    "ไปก่อน", "ลาก่อน", "บายค่ะ", "บายครับ", "บายๆ",
    "หวัดดีครับ", "หวัดดีค่ะ",  # context: at the end (greeting+goodbye are same word)
    "พอแค่นี้", "จบแค่นี้", "พอก่อน", "ขอตัวก่อน",
    "ขอบคุณนะครับ", "ขอบคุณนะคะ",  # often signals wrap-up intent
    "เจอกันใหม่", "วันหลัง", "ไว้เจอกัน",
]

_CRISIS_OVERRIDE = (
    "พี่เป็นห่วงมากเลยนะคะ คุณไม่ได้อยู่คนเดียว "
    "พี่อยากให้โทรหาสายด่วนสุขภาพจิต 1323 นะคะ "
    "เปิด 24 ชั่วโมง มีคนพร้อมรับฟังและช่วยเหลือค่ะ "
    "ตอนนี้มีใครอยู่ด้วยไหมคะ?"
)


async def _warm_ultravox():
    ultravox_url = os.environ.get("ULTRAVOX_URL", "").rstrip("/")
    if not ultravox_url:
        return
    import httpx
    model_name = os.environ.get("ULTRAVOX_MODEL", "mira-cbt")
    headers = {"Authorization": "Bearer dummy-key"}

    # Stage 1 — text warmup (existing)
    try:
        logger.info("[WARM] Pinging Ultravox text path...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{ultravox_url}/v1/chat/completions",
                json={"model": model_name,
                      "messages": [{"role": "system", "content": "คุณคือพี่มิร่า"},
                                   {"role": "user", "content": "สวัสดีค่ะ"}], "max_tokens": 30},
                headers=headers,
            )
        logger.info("[WARM] Ultravox text ready (status %s)", r.status_code)
    except Exception as e:
        logger.warning("[WARM] text ping failed: %s", e)

    # Stage 2 — A/B: audio path warmup (3 pings of 1s silence).
    # Empirically: first audio call costs ~18s (audio_tower JIT/CUDA-graph
    # compile). Ping 3× so the first real user turn hits warm audio (~340ms).
    audio_b64 = _silence_audio_b64()
    if not audio_b64:
        logger.warning("[WARM-AUDIO] could not generate silence — skipping")
        return
    for i in range(3):
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(
                    f"{ultravox_url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "ตอบสั้น"},
                            {"role": "user", "content": [
                                {"type": "text", "text": "สวัสดี"},
                                {"type": "audio_url", "audio_url": {"url": f"data:audio/ogg;base64,{audio_b64}"}},
                            ]},
                        ],
                        "max_tokens": 5,
                    },
                    headers=headers,
                )
            ms = (time.perf_counter() - t0) * 1000
            logger.info("[WARM-AUDIO] ping #%d: %.0fms (status %s)", i + 1, ms, r.status_code)
        except Exception as e:
            logger.warning("[WARM-AUDIO] ping #%d failed: %s", i + 1, e)


async def entrypoint(ctx: JobContext):
    logger.info("New session — room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    from mira_graph.llm import prewarm_cerebras
    asyncio.create_task(prewarm_cerebras())

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    FastWhisperSTT.get(device=_device)

    # Block until Ultravox is ready — greeting plays only after warm
    await _warm_ultravox()

    stt_plugin = ThaiSTT()

    # TTS provider toggle — set MIRA_TTS env var to "minimax" or "cartesia"
    # MiniMax 2.8 has the best Thai quality but needs paid balance.
    # Cartesia sonic-3 works with overages-enabled $5 plan.
    _tts_provider = os.environ.get("MIRA_TTS", "cartesia").lower()
    if _tts_provider == "minimax":
        tts = minimax.TTS(
            model="speech-2.8-turbo",
            voice=os.environ.get("MINIMAX_VOICE_ID", "socialmedia_female_2_v1"),
            language_boost="Thai",
            speed=1.0,
            emotion="neutral",
            sample_rate=24000,
        )
    else:
        tts = cartesia.TTS(
            api_key=os.environ["CARTESIA_API_KEY"],
            voice=os.environ.get("CARTESIA_VOICE_ID", "ccc7bb22-dcd0-42e4-822e-0731b950972f"),
            language="th",
            model="sonic-3",
            emotion=["Calm", "Sympathetic", "Affectionate", "Peaceful"],
            speed=1.0,
        )

    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=0.3),
        stt=stt_plugin,
        llm=_PassthroughLLM(),
        tts=tts,
    )

    mira = MiraAgent(stt_plugin=stt_plugin)
    await session.start(agent=mira, room=ctx.room, room_input_options=RoomInputOptions())


def _prewarm(proc):
    import torch
    if torch.cuda.is_available():
        FastWhisperSTT.get(device="cuda")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm))
