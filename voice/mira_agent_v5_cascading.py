"""Mira Voice Agent v5 — Cascading Architecture for low-latency therapy mode.

Replaces the v4 buffer-then-say pattern with per-sentence streaming TTS,
adds an explicit Mode dispatcher (voice/stage_manager.py), and emits
per-stage latency telemetry. Pipeline:

    AudioFrame -> silero VAD -> StreamingFastWhisperSTT (interim + final)
              -> Mode dispatch:
                   CRISIS    -> hardcoded TTS override
                   SCREENING -> build_intake_response() hardcoded path
                   THERAPY   -> call_ultravox_pure_stream -> per-sentence TTS

vLLM serves Ultravox (Qwen3-32B + Thai CBT LoRA, fully merged) on :5000;
v5 only ever calls the text-only pure_stream entry point, so the audio
tower stays idle. There is no second LLM and no LoRA extraction (the HF
repo is already fully merged).

Run:
    python -m voice.mira_agent_v5_cascading dev
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time

from dotenv import load_dotenv

# Emotion detection (XLM-EMO-T + crisis override)
try:
    from voice.emotion_detector import detect_emotion_from_user
    _EMOTION_DETECTOR_AVAILABLE = True
except Exception as _e:
    logging.warning(f"emotion_detector not available: {_e}")
    _EMOTION_DETECTOR_AVAILABLE = False
    def detect_emotion_from_user(text):
        return "calm"
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
)
from livekit.plugins import cartesia, minimax, silero

from mira_graph.llm import (
    ReasoningStreamFilter,
    apply_voice_mode,
    call_ultravox_audio_stream,
    call_ultravox_pure_stream,
)
from mira_graph.nodes._phase_base import _load_prompt
from mira_graph.nodes.counsellor import _build_supervisor_block, _extract_name
from mira_graph.nodes.intake import (
    build_intake_response,
    risk_level_label,
    score_prior_answer,
)
from mira_graph.nodes.judge import judge_node
from mira_graph.nodes.output_filter import _strip_leaks
from voice.stage_manager import Mode, resolve_mode
from voice.stt_fast_whisper import FastWhisperSTT, StreamingFastWhisperSTT

# ── RAG integration ───────────────────────────────────────────────────────────
# Feature flag for safe rollback: set MIRA_RAG_COMPOSE=1 to enable RAG pipeline
import sys
sys.path.insert(0, '/app')
try:
    from rag_integration import call_with_rag_compose
    _RAG_AVAILABLE = True
except ImportError as e:
    _RAG_AVAILABLE = False
    logging.warning(f"RAG integration not available: {e}")

USE_RAG = os.environ.get("MIRA_RAG_COMPOSE", "0") == "1" and _RAG_AVAILABLE
USE_AUDIO_TOWER = os.environ.get("MIRA_USE_AUDIO_TOWER", "0") == "1"  # send raw audio to Ultravox audio encoder
if USE_RAG:
    logging.info("[RAG] RAG compose pipeline ENABLED")
else:
    logging.info("[RAG] RAG compose pipeline DISABLED (legacy path)")


# ── Unified prompt mode (single-stage architecture) ──────────────────────────
# When MIRA_UNIFIED_PROMPT=1 (default), _handle_therapy loads mira_unified_v1.txt
# directly and skips all per-phase / supervisor / judge logic. Set to "0" to
# fall back to the legacy stage-disabled multi-prompt path.
import pathlib as _pathlib
_UNIFIED_PROMPT = os.environ.get("MIRA_UNIFIED_PROMPT", "1") == "1"
_UNIFIED_PROMPT_PATH = (
    _pathlib.Path(__file__).parent.parent / "mira_graph" / "prompts" / "mira_unified_v1.txt"
)
_UNIFIED_PROMPT_CACHE: str | None = None


# === SMART BUFFERING CONFIG ===
# Buffer 2 sentences before sending to TTS to reduce bubble fragmentation in
# LiveKit Playground. First sentence still sent immediately for low TTFA.
TTS_BUFFER_SENTENCES = 2  # Send to TTS after this many sentences accumulate
TTS_BUFFER_MAX_WAIT_MS = 800  # OR after this much time waiting (whichever first)
TTS_BUFFER_FIRST_SENTENCE_FAST = True  # First sentence sent immediately for low TTFA; fragment guard in _stream_sentences prevents solo micro-fragments


def _load_unified_prompt() -> str:
    """Load + cache the unified system prompt. Reads file once per process."""
    global _UNIFIED_PROMPT_CACHE
    if _UNIFIED_PROMPT_CACHE is None:
        _UNIFIED_PROMPT_CACHE = _UNIFIED_PROMPT_PATH.read_text(encoding="utf-8")
    return _UNIFIED_PROMPT_CACHE


_MD_RE = re.compile(r"(\*{1,3}|_{1,3}|`{1,3}|~~|#{1,6}\s?|>\s?|\[([^\]]*)\]\([^)]*\))")
_INTERJECTION_RE = re.compile(
    r"\((?:chuckle|laughs|sighs|breath|inhale|exhale|humming|emm)\)",
    re.IGNORECASE,
)
_APPROVED_INTERJECTIONS = {
    "(chuckle)", "(laughs)", "(sighs)", "(breath)",
    "(inhale)", "(exhale)", "(humming)", "(emm)",
}


def _strip_forbidden_interjections(text: str) -> str:
    def _check(m):
        tag = m.group(0).lower()
        return m.group(0) if tag in _APPROVED_INTERJECTIONS else ""
    return re.sub(r"\([a-z\-]+\)", _check, text, flags=re.IGNORECASE)


# ── Interjection normalizer: fixes common LLM formatting mistakes ────────────
# Catches patterns like "humming )" (no `(`), "(humming )" (space inside),
# or Thai descriptions like "ยิ้มเบา)" → converts to correct MiniMax tag.
_INTERJECTION_NORMALIZE = [
    # "chuckle )" / "humming )" / etc. → "(chuckle)" / "(humming)" (missing `(`)
    (re.compile(r'(?<!\()(chuckle|laughs?|sighs?|breath|inhale|exhale|humming|emm)\s*\)', re.IGNORECASE),
     r'(\1)'),
    # "(chuckle )" / "(humming )" → "(chuckle)" / "(humming)" (space inside)
    (re.compile(r'\((chuckle|laughs?|sighs?|breath|inhale|exhale|humming|emm)\s+\)', re.IGNORECASE),
     r'(\1)'),
    # Thai: "ยิ้มเบา)" → "(chuckle)"
    (re.compile(r'ยิ้ม[เบา]*[ๆ]*\s*\)', re.IGNORECASE), '(chuckle)'),
    # Thai: "หัวเราะเบา)" / "หัวเราะ)" → "(laughs)"
    (re.compile(r'หัวเราะ[เบา]*[ๆ]*\s*\)', re.IGNORECASE), '(laughs)'),
    # Thai: "ถอนหายใจเบา)" / "ถอนหายใจ)" → "(sighs)"
    (re.compile(r'ถอนหายใจ[เบา]*[ๆ]*\s*\)', re.IGNORECASE), '(sighs)'),
    # Thai: "สูดหายใจ)" / "สูดลมหายใจ)" / "หายใจ)" → "(breath)" (natural pause)
    (re.compile(r'สูด(?:ลม)?หายใจ[เบา]*[ๆ]*\s*\)', re.IGNORECASE), '(breath)'),
    # Thai+English mixed: "ส่งเสียง sighs เบาๆ)" / "sighs เบาๆ)" → "(sighs)"
    (re.compile(r'(?:ส่งเสียง|เสียง)?\s*(?<!\()(chuckle|laughs?|sighs?|breath|inhale|exhale|humming|emm)\s*[เบา]*[ๆ]*\s*\)', re.IGNORECASE),
     r'(\1)'),
]


def _normalize_interjections(text: str) -> str:
    for pattern, replacement in _INTERJECTION_NORMALIZE:
        text = pattern.sub(replacement, text)
    return text


# ── Catch-all: strip any leftover forbidden word) (where LEAK_HEAD_RE ate the `(`) ──
_FORBIDDEN_INTERJECTION_WORDS = {
    "coughs", "sneezes", "burps", "snorts", "groans", "pant",
    "gasps", "hissing", "sniffs", "clear-throat", "lip-smacking",
}


def _strip_forbidden_interjection_tails(text: str) -> str:
    return re.sub(
        r"(?<!\()(" + "|".join(re.escape(w) for w in _FORBIDDEN_INTERJECTION_WORDS) + r")\s*\)",
        "", text, flags=re.IGNORECASE,
    )


def _strip_markdown(text: str) -> str:
    text = _MD_RE.sub(r"\2", text)
    # Defensive: also strip any [EMOTION: xxx] tags that slipped through
    text = re.sub(r'\[EMOTION:\s*[^\]]*\]', '', text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()


# ── Emotion + NEXT_STAGE tag parsing (TTS dynamic emotion + stage signal) ────
# Model emits [EMOTION: <type>] and [NEXT_STAGE: <stage>] as the first two
# lines of every reply. We strip BOTH from the spoken text. EMOTION mutates
# the TTS instance; NEXT_STAGE is consumed by output_filter_node + judge_node
# in the langgraph pipeline (separate path) — here we only strip it so TTS
# doesn't speak the tag aloud.
# 'fluent' is only supported by speech-2.6-* models — on 2.8-* it falls back
# to neutral at runtime (see _apply_meta_tags in MiraAgent).
_EMOTION_TAG_RE = re.compile(r"^\s*\[EMOTION:\s*([a-zA-Z]+)\s*\]\s*", re.IGNORECASE)
# Tolerant: accepts NEXT_STAGE / NEXTSTAGE / NEXT STAGE
_STAGE_TAG_RE = re.compile(r"^\s*\[NEXT[_\s]?STAGE:\s*([A-Z0-9_\s]+?)\s*\]\s*", re.IGNORECASE)
_STAGE_NORMALIZE_RE = re.compile(r"^(S\d)([A-Z]+)$")


# ── Emotion tag stripping for streaming TTS pipeline ──────────────────────────
# Match [EMOTION: xxx] or [emotion: xxx] anywhere in text (not just anchored to start).
# Captures the emotion label so it can be saved to state.
_EMOTION_TAG_ANYWHERE_RE = re.compile(r'\[EMOTION:\s*([^\]]*)\]', re.IGNORECASE)


def _extract_emotion_tag(text: str) -> tuple[str, str | None]:
    """
    Extract [EMOTION: xxx] tag from text.

    Returns:
        (cleaned_text, emotion_label_or_None)

    Examples:
        "[EMOTION: calm] สวัสดี" → ("สวัสดี", "calm")
        "สวัสดี [EMOTION: warm]" → ("สวัสดี", "warm")
        "สวัสดีค่ะ" → ("สวัสดีค่ะ", None)
        "[EMOTION:happy]hello" → ("hello", "happy")
    """
    match = _EMOTION_TAG_ANYWHERE_RE.search(text)
    emotion = None
    if match:
        emotion = match.group(1).strip().lower()

    # Remove all emotion tags from text
    cleaned = _EMOTION_TAG_ANYWHERE_RE.sub('', text)
    # Collapse multiple spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned, emotion


def _strip_emotion_tag(text: str) -> str:
    """Strip [EMOTION: xxx] tags from text. Used in streaming pipeline."""
    cleaned, _ = _extract_emotion_tag(text)
    return cleaned


def _normalize_stage(raw: str) -> str:
    """Normalize stage names: handles legacy 'S2EXPLORE' → 'EXPLORE'."""
    raw = raw.upper().replace(" ", "")
    m = _STAGE_NORMALIZE_RE.match(raw)
    if m:
        raw = f"{m.group(1)}_{m.group(2)}"
    return raw
_EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "calm": "neutral",        # minimax has no 'calm' — neutral is closest
    "neutral": "neutral",
    "fearful": "fearful",
    "surprised": "surprised",
    "fluent": "fluent",        # may fall back to neutral if model is 2.8-*
    # tolerate model emitting minimax-only values
    "angry": "angry",
    "disgusted": "disgusted",
}
_FLUENT_MODELS = (
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-2.5-hd-preview",
    "speech-2.5-turbo-preview",
)


class _PassthroughLLM(llm.LLM):
    async def chat(self, chat_ctx, **kwargs):
        raise NotImplementedError("llm_node override handles generation")


load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("cerebras").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logger = logging.getLogger("mira-v5")


# ──────────────────────────────────────────────────────────────────────────
# Latency telemetry
# ──────────────────────────────────────────────────────────────────────────


class _LatencyTracker:
    """Per-turn timer anchored at speech_end (when VAD reports end-of-speech).

    Logs `[LATENCY] stage=X event=Y t=Nms total=Mms` for each event, plus a
    single `[LATENCY] e2e speech_end_to_first_audio=Mms` line per turn — the
    one number we optimise against (target <1500 ms).
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self._last_event_t: float | None = None
        self._first_audio_logged = False

    def start(self, t0: float | None = None) -> None:
        self._t0 = t0 if t0 is not None else time.perf_counter()
        self._last_event_t = self._t0
        self._first_audio_logged = False

    def event(self, stage: str, event: str) -> float:
        if self._t0 is None:
            self.start()
        now = time.perf_counter()
        t_ms = (now - (self._last_event_t or now)) * 1000
        total_ms = (now - self._t0) * 1000
        self._last_event_t = now
        logger.info(
            "[LATENCY] stage=%s event=%s t=%dms total=%dms",
            stage, event, int(t_ms), int(total_ms),
        )
        if stage == "tts" and event == "first_audio" and not self._first_audio_logged:
            logger.info("[LATENCY] e2e speech_end_to_first_audio=%dms", int(total_ms))
            self._first_audio_logged = True
        return total_ms


def _initial_state() -> dict:
    return {
        "user_text": None,
        "user_audio_b64": None,
        "messages": [],
        "phase": "CHECKIN",
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
        "phq9_directive": None,
        "phq9_detected_symptom": None,
        "phq9_chosen_probe": None,
        "phq9_probes_asked": [],
        "_at_confidence": None,
        "_at_asking_count": 0,
        # NSC demo: PHQ-9 screening DISABLED. Start in therapy mode.
        "session_type": "therapy",
        "intake_started": True,
        "intake_pending_score": False,
        "intake_smalltalk_count": 0,
        "phq9_step": 9,
        "phq9_scores": {},
        "phq9_total": 0,
        "phq9_q9_score": 0,
    }


# Crisis signals + override (identical to v4 — kept verbatim)
_CRISIS_SIGNALS = [
    "ทำร้ายตัวเอง", "ฆ่าตัวตาย", "อยากตาย", "จบชีวิต", "ปิดสวิตช์ชีวิต",
    "โดด", "กรีด", "กินยา", "แขวนคอ",
    "ไม่อยากอยู่", "ไม่อยากมีชีวิต", "อยู่ไปทำไม", "เป็นภาระ",
    "อยากหายไป", "ไม่อยากตื่น", "ไม่อยากหายใจ", "ไม่มีความหมาย",
]

_END_PHRASES = ["วันนี้จบแล้ว", "จบแล้วนะคะ", "เริ่ม session ใหม่"]

_GOODBYE_KEYWORDS = [
    "ไปก่อน", "ลาก่อน", "บายค่ะ", "บายครับ", "บายๆ",
    "หวัดดีครับ", "หวัดดีค่ะ",
    "พอแค่นี้", "จบแค่นี้", "พอก่อน", "ขอตัวก่อน",
    "ขอบคุณนะครับ", "ขอบคุณนะคะ",
    "เจอกันใหม่", "วันหลัง", "ไว้เจอกัน",
]

_CRISIS_OVERRIDE = (
    "พี่เป็นห่วงมากเลยนะคะ คุณไม่ได้อยู่คนเดียว "
    "พี่อยากให้โทรหาสายด่วนสุขภาพจิต 1323 นะคะ "
    "เปิด 24 ชั่วโมง มีคนพร้อมรับฟังและช่วยเหลือค่ะ "
    "ตอนนี้มีใครอยู่ด้วยไหมคะ?"
)


# ──────────────────────────────────────────────────────────────────────────
# MiraAgent — v5
# ──────────────────────────────────────────────────────────────────────────


class MiraAgent(Agent):
    def __init__(self, stt_plugin: StreamingFastWhisperSTT) -> None:
        super().__init__(instructions="คุณคือพี่มิร่า เพื่อนคู่ใจ AI ที่ใช้ CBT framework")
        self._mira_state = _initial_state()
        self._stt = stt_plugin
        self._chunk_buffer: list[str] = []
        self._buffer_task: asyncio.Task | None = None
        self._buffer_timeout: float = 0.3
        self._is_processing: bool = False
        self._recent_user_texts: list[str] = []
        self._crisis_active: bool = False
        self._hotline_mentioned: bool = False   # 1323 said? avoid repeat
        self._calm_turns_in_row: int = 0        # auto-clear sticky lock

    async def on_enter(self) -> None:
        await self.session.say("สวัสดีค่ะ วันนี้มีอะไรให้ช่วยไหมคะ")
        self._mira_state["intake_started"] = True

    def _set_tts_emotion_from_user(self, user_text: str) -> None:
        """Classify user emotion via XLM-EMO-T and set TTS emotion."""
        if not user_text or not _EMOTION_DETECTOR_AVAILABLE:
            return
        try:
            emotion = detect_emotion_from_user(user_text)
        except Exception as e:
            logger.warning(f"[EMOTION-user] detection failed: {e}")
            return
        mapped = _EMOTION_MAP.get(emotion, "neutral")
        tts = getattr(self.session, "tts", None)
        if tts is not None and hasattr(tts, "_opts") and hasattr(tts._opts, "emotion"):
            model = str(getattr(tts._opts, "model", "") or "")
            if mapped == "fluent" and not model.startswith(_FLUENT_MODELS):
                mapped = "neutral"
            if tts._opts.emotion != mapped:
                logger.info(
                    "[EMOTION-user] %r -> %r (XLM-EMO: %r, text: %r)",
                    tts._opts.emotion, mapped, emotion, user_text[:50]
                )
                tts._opts.emotion = mapped

    def _apply_meta_tags(self, text: str) -> str:
        """Strip leading [EMOTION:] and [NEXT_STAGE:] tags. Mutate TTS emotion.

        Returns the tag-stripped text. If no tags are present, returns text unchanged.
        Empty string is returned if the entire input was just tags.
        NEXT_STAGE is stripped here so TTS doesn't speak it aloud — the actual
        stage suggestion is consumed by output_filter_node in the langgraph.
        """
        if not text:
            return text

        # Strip EMOTION first (and apply to TTS), then NEXT_STAGE — both anchored
        # to the start, so order matters when they appear on consecutive lines.
        em = _EMOTION_TAG_RE.match(text)
        if em:
            raw = em.group(1).lower()
            mapped = _EMOTION_MAP.get(raw, "neutral")
            text = text[em.end():]

            tts = getattr(self.session, "tts", None)
            if tts is not None and hasattr(tts, "_opts") and hasattr(tts._opts, "emotion"):
                model = str(getattr(tts._opts, "model", "") or "")
                if mapped == "fluent" and not model.startswith(_FLUENT_MODELS):
                    logger.warning(
                        "[EMOTION] 'fluent' not supported by model %s — using neutral",
                        model,
                    )
                    mapped = "neutral"
                if tts._opts.emotion != mapped:
                    logger.info(
                        "[EMOTION] %r → %r (tag=%r)", tts._opts.emotion, mapped, raw
                    )
                    tts._opts.emotion = mapped

        # NEXT_STAGE: strip + log normalized value so we can see what Mira chose
        sm = _STAGE_TAG_RE.match(text)
        if sm:
            normalized = _normalize_stage(sm.group(1))
            logger.info("[STAGE-tag] raw=%r → normalized=%r", sm.group(1), normalized)
            text = text[sm.end():]

        return text.strip()

    # Backwards-compatible alias kept in case external callers reference it
    _apply_emotion_tag = _apply_meta_tags

    def _safe_say(self, text: str) -> bool:
        text = self._apply_meta_tags(text)
        # Strip orphan leading punctuation (?, !, ., ,) that streamed in after a
        # prior sentence already flushed at คะ/ค่ะ/ครับ — without this the
        # TTS would speak a standalone "?" then the next sentence, breaking audio.
        text = re.sub(r"^[\s?!.,]+", "", text or "")
        if not text:
            return True  # tag-only output, nothing to speak
        try:
            self.session.say(text)
            # Notify STT to open echo-mute window so Whisper doesn't transcribe
            # Mira's own TTS playback as user speech.
            if self._stt is not None:
                self._stt.notify_tts_done()
            return True
        except RuntimeError as e:
            msg = str(e)
            if "no activity context" in msg or "AgentSession is closing" in msg:
                logger.info("[SAY] session ended — dropping output: %s", text[:40])
                return False
            raise

    async def _send_event(self, payload: dict) -> None:
        """Publish a JSON event over the LiveKit data channel.

        Best-effort: used to let the frontend merge per-sentence transcription
        bubbles into a single coherent message. Schema:
            {"type": "response_chunk",    "text": "<sentence>"}
            {"type": "response_complete", "text": "<full_response>",
             "emotion": "<label_or_null>"}
        Any error is swallowed — frontend events are not critical to the
        voice/text pipeline, so a missing room or stopped participant should
        not break the conversation.
        """
        try:
            import json as _json
            room = getattr(getattr(self, "session", None), "room", None) or getattr(self, "_room", None)
            participant = getattr(room, "local_participant", None) if room else None
            if participant is None:
                return
            data = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
            # Prefer reliable delivery so the frontend definitely sees the
            # "response_complete" marker. Falls back silently if the room is
            # already closed or the API surface differs across livekit-rtc
            # versions.
            try:
                await participant.publish_data(data, reliable=True, topic="mira.response")
            except TypeError:
                # Older livekit-rtc signature
                await participant.publish_data(data)
        except Exception as e:
            logger.debug("[Mira] publish_data skipped: %s", e)

    def _detect_crisis(self, text: str) -> list[str]:
        return [s for s in _CRISIS_SIGNALS if s in text]

    def _crisis_cumulative_check(self, current_text: str) -> list[str]:
        haystack = " ".join(self._recent_user_texts[-2:]) + " " + current_text
        return [s for s in _CRISIS_SIGNALS if s in haystack]

    def _trigger_crisis_override(
        self,
        source: str,
        matched: list[str],
        lat: _LatencyTracker | None = None,
    ) -> bool:
        """Returns True if a hardcoded reply was spoken (caller should stop).
        Returns False if hotline was already mentioned — caller should run
        the normal LLM path with a crisis hint so Mira can follow up
        empathetically without repeating 1323."""
        was_active = self._crisis_active
        self._crisis_active = True
        self._calm_turns_in_row = 0
        if self._mira_state.get("phase") == "END":
            self._mira_state["phase"] = "CHECKIN"
            self._mira_state["crisis_detected"] = True
        logger.warning(
            "[CRISIS 🚨] %s — matched=%s, was_active=%s, hotline_mentioned=%s",
            source, matched[:3], was_active, self._hotline_mentioned,
        )
        if self._hotline_mentioned:
            # Already said 1323 earlier — let LLM continue therapy with hint.
            self._mira_state["crisis_followup"] = True
            return False
        self._chunk_buffer.clear()
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        spoke = self._safe_say(_CRISIS_OVERRIDE)
        self._hotline_mentioned = True
        # Append to messages so the LLM knows what Mira just said in the
        # next turn — otherwise user's reply "มี อยู่" looks context-less.
        self._mira_state["messages"] = list(self._mira_state.get("messages", [])) + [
            {"role": "assistant", "content": _CRISIS_OVERRIDE}
        ]
        self._mira_state["mira_response"] = _CRISIS_OVERRIDE
        if lat:
            if spoke:
                lat.event("tts", "first_audio")
            lat.event("tts", "complete")
        return True

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

        # Earliest crisis interception — every chunk gets checked.
        self._recent_user_texts.append(text.strip())
        if len(self._recent_user_texts) > 10:
            self._recent_user_texts = self._recent_user_texts[-10:]

        # XLM-EMO emotion detection - earliest TTS emotion setup
        self._set_tts_emotion_from_user(text)

        matched = self._detect_crisis(text) or self._crisis_cumulative_check(text)
        if matched:
            handled = self._trigger_crisis_override("llm_node", matched)
            if handled:
                return ""
            # hotline already mentioned earlier → fall through to therapy LLM
            # (crisis_followup hint was set by _trigger_crisis_override)
        else:
            if self._crisis_active:
                self._calm_turns_in_row += 1
                if self._calm_turns_in_row >= 2:
                    logger.info(
                        "[CRISIS] cleared after %d calm turns",
                        self._calm_turns_in_row,
                    )
                    self._crisis_active = False
                    self._calm_turns_in_row = 0
                    self._mira_state["crisis_followup"] = False

        if self._mira_state.get("phase") == "END":
            if self._crisis_active:
                self._mira_state["phase"] = "CHECKIN"
            else:
                logger.info("[Mira] Re-opening session after END for: %s", text[:40])
                self._mira_state["phase"] = "WRAP"

        # Buffer chunks; the 1.5 s silence gate concatenates fragments before
        # we run a single Ultravox call.
        self._chunk_buffer.append(text.strip())
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        self._buffer_task = asyncio.create_task(self._process_after_silence())
        return ""

    async def _process_after_silence(self) -> None:
        try:
            await asyncio.sleep(self._buffer_timeout)
        except asyncio.CancelledError:
            return
        if not self._chunk_buffer:
            return
        if self._is_processing:
            self._chunk_buffer.clear()
            return
        combined = " ".join(self._chunk_buffer)
        self._chunk_buffer.clear()
        await self._run_mira_turn(combined)

    async def _run_mira_turn(self, text: str) -> None:
        self._is_processing = True
        try:
            # Anchor telemetry at VAD speech_end if the streaming STT recorded
            # one this turn; otherwise anchor at "now" which still gives us
            # a useful relative breakdown.
            t0 = self._stt.last_speech_end_t
            self._stt.last_speech_end_t = None
            lat = _LatencyTracker()
            lat.start(t0)
            lat.event("stt", "complete")

            logger.info("[Mira] User: %s", text)

            matched = self._detect_crisis(text) or self._crisis_cumulative_check(text)
            if matched:
                handled = self._trigger_crisis_override("_run_mira_turn", matched, lat=lat)
                if handled:
                    return
                # hotline already mentioned — fall through to therapy LLM
            else:
                if self._crisis_active:
                    self._calm_turns_in_row += 1
                    if self._calm_turns_in_row >= 2:
                        logger.info(
                            "[CRISIS] cleared after %d calm turns",
                            self._calm_turns_in_row,
                        )
                        self._crisis_active = False
                        self._calm_turns_in_row = 0
                        self._mira_state["crisis_followup"] = False

            if any(kw in text for kw in _GOODBYE_KEYWORDS):
                logger.info("[GOODBYE] detected → WRAP")
                self._mira_state["phase"] = "WRAP"
                if self._mira_state.get("session_type") == "intake":
                    self._mira_state["session_type"] = "therapy"

            self._mira_state["user_text"] = text
            # XLM-EMO emotion detection - set TTS emotion early
            self._set_tts_emotion_from_user(text)
            if USE_AUDIO_TOWER and self._stt is not None:
                self._mira_state["user_audio_b64"] = getattr(self._stt, "last_audio_b64", None)
            else:
                self._mira_state["user_audio_b64"] = None
            self._mira_state["messages"] = list(self._mira_state["messages"]) + [
                {"role": "user", "content": text}
            ]
            for k in (
                "supervisor_empathy", "supervisor_belief", "supervisor_reflection",
                "supervisor_strategy", "supervisor_encouragement",
            ):
                self._mira_state[k] = None
            self._mira_state["parallel_results"] = []
            self._mira_state["quick_response"] = None

            mode = resolve_mode(self._mira_state, self._crisis_active)
            logger.info(
                "[MODE] resolved=%s phase=%s session=%s",
                mode.value, self._mira_state.get("phase"),
                self._mira_state.get("session_type"),
            )

            if mode == Mode.CRISIS:
                # Already covered by the crisis-keyword path above; this is a
                # safety fallback for state-driven crisis (e.g. phase=CRISIS).
                handled = self._trigger_crisis_override(
                    "mode=CRISIS", ["mode_crisis"], lat=lat
                )
                if handled:
                    return
                # hotline already said — continue to therapy with crisis hint

            if mode == Mode.SCREENING:
                await self._handle_screening(text, lat)
                return

            await self._handle_therapy(text, lat)
        finally:
            self._is_processing = False

    async def _handle_screening(self, text: str, lat: _LatencyTracker) -> None:
        """Hardcoded PHQ-9 alternation — same logic as v4 intake path,
        kept deterministic to guarantee Q1..Q9 progression.
        """
        score_updates = await score_prior_answer(self._mira_state, text)
        self._mira_state.update(score_updates)

        if score_updates.get("_intake_q9_crisis"):
            logger.warning("[INTAKE 🚨] Q9 score >= 1 — triggering crisis override")
            self._trigger_crisis_override(
                "intake_Q9", ["intake_Q9_self_harm"], lat=lat
            )
            self._mira_state["session_type"] = "therapy"
            self._mira_state["phase"] = "CHECKIN"
            return

        response, response_updates = build_intake_response(self._mira_state)
        self._mira_state.update(response_updates)

        # SCREENING uses no LLM call; fold first_token into "complete" for
        # consistent telemetry shape.
        lat.event("llm", "first_token")
        lat.event("llm", "complete")
        if self._safe_say(response):
            lat.event("tts", "first_audio")
        lat.event("tts", "complete")

        if response_updates.get("_intake_complete"):
            total = self._mira_state.get("phq9_total") or 0
            risk = risk_level_label(total)
            logger.info(
                "[INTAKE ✅] complete — total=%d risk=%s scores=%s",
                total, risk, self._mira_state.get("phq9_scores"),
            )
            self._mira_state["session_type"] = "therapy"
            self._mira_state["phase"] = "CHECKIN"
            logger.info("[MODE] SCREENING → THERAPY (intake complete)")

        self._mira_state["mira_response"] = response
        self._mira_state["messages"] = list(self._mira_state["messages"]) + [
            {"role": "assistant", "content": response}
        ]

    async def _handle_therapy(self, text: str, lat: _LatencyTracker) -> None:
        """Cascading: text -> Ultravox text-only stream -> per-sentence TTS.

        Every sentence yielded by Ultravox is spoken immediately (no buffer).
        This is the v5 fix for the v4 buffer-then-say latency tax.

        When MIRA_UNIFIED_PROMPT=1 (default), uses single unified prompt and
        skips per-phase / supervisor / judge logic. The LLM (CBT-finetuned)
        drives flow itself.
        """
        if _UNIFIED_PROMPT:
            # Single-stage architecture: load unified prompt only.
            system_prompt = _load_unified_prompt()

            # Crisis follow-up addendum (still useful in unified mode for safety).
            if self._mira_state.get("crisis_followup") or self._crisis_active:
                last_mira = ""
                for m in reversed(self._mira_state.get("messages", [])):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        last_mira = (m.get("content") or "")[:200]
                        break
                system_prompt += (
                    "\n\n=== CRISIS FOLLOW-UP ===\n"
                    f"พี่มิร่าเพิ่งพูดว่า: \"{last_mira}\"\n"
                    "User เพิ่งแสดงสัญญาณวิกฤต — แนะนำสายด่วน 1323 ไปแล้ว\n"
                    "- อ่านคำตอบ user ตรงไปตรงมา (ไม่ตีความ metaphor)\n"
                    "- ถ้ามีคนอยู่ → ถาม 'ใครอยู่ด้วยคะ? พอจะคุยกับเขาได้ไหม?'\n"
                    "- ถ้าไม่มีใคร → empathy + ชวนใช้ 1323 (ไม่พูดคำว่า 1323 ซ้ำ)\n"
                    "- ห้าม CBT / advice — แค่ presence + safety\n"
                    "- ตอบสั้น 1-2 ประโยค\n"
                )
        else:
            # Legacy stage-disabled multi-prompt path.
            system_prompt = _load_prompt(
                self._mira_state.get("phase", "CHECKIN"), self._mira_state
            )
            system_prompt += _build_supervisor_block(self._mira_state)

            if self._mira_state.get("crisis_followup") or self._crisis_active:
                last_mira = ""
                for m in reversed(self._mira_state.get("messages", [])):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        last_mira = (m.get("content") or "")[:200]
                        break
                system_prompt += (
                    "\n\n=== [CRISIS FOLLOW-UP — สำคัญ] ===\n"
                    "User เพิ่งแสดงสัญญาณวิกฤต (passive/active ideation) และระบบ\n"
                    "ได้แนะนำสายด่วน 1323 ไปแล้วในเทิร์นก่อนหน้า\n"
                    f"พี่มิร่าเพิ่งพูดว่า: \"{last_mira}\"\n"
                    "กฎการตอบ:\n"
                    "- **อ่านคำตอบล่าสุดของ user อย่างตรงไปตรงมา** เป็นคำตอบของคำถามที่พี่เพิ่งถาม\n"
                    "  เช่น ถ้าพี่ถาม 'มีใครอยู่ด้วยไหมคะ?' แล้ว user ตอบ 'มี' = มีคนอยู่ด้วยจริงๆ\n"
                    "  ห้ามตีความเป็น semantic reflection หรือ metaphor\n"
                    "- ถ้า user ตอบว่ามีคนอยู่ → ถาม 'ใครอยู่ด้วยคะ?' / 'พอจะคุยกับเขาได้ไหมคะ?'\n"
                    "- ถ้า user ตอบว่าไม่มีใคร → รับรู้ด้วย empathy แล้วชวนให้โทร 1323\n"
                    "- ห้ามพูดคำว่า '1323' หรือ 'สายด่วน' ซ้ำในเทิร์นนี้ (พูดไปแล้ว)\n"
                    "- ห้ามถาม 'รู้สึกยังไงคะ' / 'รู้สึกอะไรคะ' (คำถามนี้กว้างเกินสำหรับ crisis)\n"
                    "- ห้ามทำ CBT / ไม่วิเคราะห์ distortion / ไม่ให้คำแนะนำแก้ปัญหา\n"
                    "- ตอบสั้น 1-2 ประโยค โทนอ่อนโยน ใส่ใจ\n"
                )

            if not self._mira_state.get("user_name"):
                system_prompt += (
                    "\n\n[หมายเหตุ: ยังไม่ทราบชื่อผู้ใช้ ห้ามเรียกชื่อหรือสมมติชื่อ "
                    "ใช้คำว่า 'คุณ' แทนจนกว่าผู้ใช้จะบอกชื่อ]"
                )

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

        history = list(self._mira_state.get("messages", []))[-6:]

        # Cache stability debug — system_prompt sent to vLLM (legacy path).
        # When MIRA_RAG_COMPOSE=1 (default), call_with_rag_compose ignores this
        # and builds its own stable prompt; see [CACHE-DEBUG] log there.
        try:
            import hashlib as _hashlib
            _sp_hash = _hashlib.md5(system_prompt[:2000].encode("utf-8")).hexdigest()[:8]
            logger.info(f"[CACHE-DEBUG] legacy system_prompt prefix hash: {_sp_hash} (len={len(system_prompt)})")
        except Exception:
            pass

        all_sentences: list[str] = []
        reason_filter = ReasoningStreamFilter()
        first_token_logged = False
        first_audio_logged = False
        session_dropped = False

        # === SMART BUFFERING STATE ===
        # Accumulate 1-2 sentences before sending to TTS so LiveKit transcription
        # events (= chat bubbles) don't fragment. First sentence still fires
        # immediately to keep TTFA low.
        import time as _time
        tts_buffer: list[str] = []
        buffer_start_time: float | None = None
        first_sentence_sent = False

        async def flush_tts_buffer() -> bool:
            """Send accumulated buffer to TTS as one chunk. Returns False if
            the session dropped mid-say so the outer loop can break cleanly."""
            nonlocal tts_buffer, buffer_start_time, first_audio_logged
            if not tts_buffer:
                return True
            merged = " ".join(tts_buffer)
            n = len(tts_buffer)
            tts_buffer = []
            buffer_start_time = None
            if not self._safe_say(merged):
                return False
            if not first_audio_logged:
                lat.event("tts", "first_audio")
                first_audio_logged = True
            logger.info("[Mira] STREAM_CHUNK (%d sentences): %s", n, merged)
            await self._send_event({"type": "response_chunk", "text": merged})
            return True

        try:
            if USE_RAG:
                audio_b64 = self._mira_state.get("user_audio_b64") if USE_AUDIO_TOWER else None
                stream = call_with_rag_compose(
                    system_prompt, text, history,
                    graph_phase=self._mira_state.get("phase"),
                    audio_b64=audio_b64,
                )
            else:
                audio_b64 = self._mira_state.get("user_audio_b64") if USE_AUDIO_TOWER else None
                if audio_b64:
                    logger.info("[LLM_PATH] audio_tower (b64_len=%d)", len(audio_b64))
                    stream = call_ultravox_audio_stream(system_prompt, audio_b64, history)
                else:
                    logger.info("[LLM_PATH] text_only (use_audio_tower=%s, audio_b64_is_none=%s)", USE_AUDIO_TOWER, audio_b64 is None)
                    stream = call_ultravox_pure_stream(system_prompt, text, history)

            async for sentence in stream:
                if not sentence.strip():
                    continue

                # Extract emotion tag for state (avatar/logs) and strip from TTS text
                sentence, emotion_label = _extract_emotion_tag(sentence)
                if emotion_label:
                    # Store last detected emotion in state for downstream use
                    self._mira_state["last_emotion"] = emotion_label
                    logger.info(f"[EMOTION] Detected: {emotion_label}")

                # Skip if entire sentence was just the emotion tag
                if not sentence:
                    continue

                # Protect approved MiniMax interjection tags from being stripped
                _protected = []
                def _save(m):
                    _protected.append(m.group(0))
                    return f"\x00INJ{len(_protected)-1}\x00"
                sentence = _INTERJECTION_RE.sub(_save, sentence)

                sentence = _strip_markdown(_strip_leaks(sentence))

                for i, interj in enumerate(_protected):
                    sentence = sentence.replace(f"\x00INJ{i}\x00", interj)

                sentence = _strip_forbidden_interjections(sentence)

                sentence = _normalize_interjections(sentence)

                sentence = _strip_forbidden_interjection_tails(sentence)
                if not sentence:
                    continue

                spoken = reason_filter.feed(sentence)
                if not spoken or not spoken.strip():
                    continue

                if not first_token_logged:
                    lat.event("llm", "first_token")
                    first_token_logged = True

                # Mid-stream crisis safety net.
                if self._crisis_active and any(p in spoken for p in _END_PHRASES):
                    logger.warning("[SAFETY NET 🚨] Intercepted END mid-stream — replacing")
                    # Drop any buffered content; crisis override takes over.
                    tts_buffer = []
                    buffer_start_time = None
                    if self._safe_say(_CRISIS_OVERRIDE):
                        if not first_audio_logged:
                            lat.event("tts", "first_audio")
                            first_audio_logged = True
                    all_sentences = [_CRISIS_OVERRIDE]
                    break

                # === SMART BUFFER: accumulate then flush ===
                tts_buffer.append(spoken)
                all_sentences.append(spoken)

                if buffer_start_time is None:
                    buffer_start_time = _time.monotonic()

                # FAST FIRST SENTENCE — keep TTFA low.
                if TTS_BUFFER_FIRST_SENTENCE_FAST and not first_sentence_sent:
                    first_sentence_sent = True
                    if not await flush_tts_buffer():
                        session_dropped = True
                        break
                    continue

                # NORMAL — flush when buffer hits target size.
                if len(tts_buffer) >= TTS_BUFFER_SENTENCES:
                    if not await flush_tts_buffer():
                        session_dropped = True
                        break
                    continue

                # TIMEOUT — flush if waited too long for next sentence.
                elapsed_ms = (_time.monotonic() - buffer_start_time) * 1000
                if elapsed_ms >= TTS_BUFFER_MAX_WAIT_MS:
                    if not await flush_tts_buffer():
                        session_dropped = True
                        break

            # Stream ended — flush any remaining buffered sentences.
            if not session_dropped:
                await flush_tts_buffer()

            lat.event("llm", "complete")

            if reason_filter.reasoning:
                logger.info(
                    "\n%s\n[MIRA REASONING]\n%s\n[MIRA SPOKEN]\n%s\n%s",
                    "=" * 60,
                    reason_filter.reasoning,
                    " ".join(all_sentences),
                    "=" * 60,
                )

        except Exception as e:
            logger.exception("[Mira] stream FAILED: %s", e)

        lat.event("tts", "complete")

        if not all_sentences and not session_dropped:
            fallback = "อืม เล่าต่อได้เลยนะคะ"
            all_sentences = [fallback]
            self._safe_say(fallback)

        full_response = " ".join(all_sentences)

        # Name-correction: detect denial ("ผมไม่ได้ชื่อ X") or affirmation ("ผมชื่อ Y")
        # and overwrite whatever was previously stored (even if wrong due to echo).
        import re as _re
        _DENY_RE = _re.compile(r"ไม่(?:ได้)?ชื่อ|ชื่อ(?:ไม่ใช่)")
        _AFFIRM_RE = _re.compile(r"(?:ผม|ฉัน|หนู|เขา)(?:\s*)ชื่อ\s*([ก-๙a-zA-Z]{2,15})")
        _CORRECT_CALL = _re.compile(r"(?:เรียก(?:ผม)?ว่า|เรียกฉันว่า|ชื่อจริงๆ(?:คือ)?)\s*([ก-๙a-zA-Z]{2,15})")
        _is_correction = bool(_DENY_RE.search(text))
        if _is_correction or not self._mira_state.get("user_name"):
            name = None
            for pat in (_AFFIRM_RE, _CORRECT_CALL):
                m = pat.search(text)
                if m:
                    candidate = m.group(1).strip()
                    # Reject function words that slip through
                    _BAD = {"ไม่", "ได้", "แล้ว", "นะ", "ครับ", "ค่ะ", "คะ", "จริง"}
                    if candidate not in _BAD and 2 <= len(candidate) <= 12:
                        name = candidate
                        break
            if name is None and not _is_correction:
                # Fall back to the counsellor-side extractor (covers "เรียก X ก็ได้" etc.)
                name = _extract_name(text)
            if name:
                prev = self._mira_state.get("user_name")
                self._mira_state["user_name"] = name
                if prev and prev != name:
                    logger.info("[NAME] corrected %s → %s", prev, name)
                else:
                    logger.info("[NAME] %s", name)

        self._mira_state["mira_response"] = full_response
        self._mira_state["messages"] = list(self._mira_state["messages"]) + [
            {"role": "assistant", "content": full_response}
        ]

        # Best-effort completion signal so the frontend can finalize / merge
        # the per-chunk transcription bubbles into a single coherent message.
        await self._send_event({
            "type": "response_complete",
            "text": full_response,
            "emotion": self._mira_state.get("last_emotion"),
        })

        # Track previous phase for transition logging
        _prev_phase = self._mira_state.get("phase")

        # Always run judge_node to advance phase (even in unified mode — the old
        # "single-stage: no judge" comment was the root cause of stuck S1_RAPPORT).
        try:
            judge_result = await judge_node(self._mira_state)
            self._mira_state.update(judge_result)
            if _prev_phase != self._mira_state.get("phase"):
                logger.info("[JUDGE] PHASE TRANSITION: %s -> %s (turn=%s)",
                            _prev_phase, self._mira_state.get("phase"),
                            self._mira_state.get("phase_turn_count", 0))
            if self._crisis_active and self._mira_state.get("phase") == "END":
                logger.warning("[SAFETY NET 🚨] Blocked judge → END during crisis")
                self._mira_state["phase"] = "CHECKIN"
                self._mira_state["crisis_detected"] = True
        except Exception as e:
            logger.warning("[Mira] judge FAILED: %s", e)

        if _UNIFIED_PROMPT:
            logger.info("[Mira] Response [%s]: %s", self._mira_state.get("phase"), full_response)
        else:
            logger.info("[Mira] Response [%s]: %s", self._mira_state.get("phase"), full_response)


# ──────────────────────────────────────────────────────────────────────────
# Warm-up + entrypoint
# ──────────────────────────────────────────────────────────────────────────


async def _warm_ultravox():
    ultravox_url = os.environ.get("ULTRAVOX_URL", "").rstrip("/")
    if not ultravox_url:
        return
    import httpx
    try:
        logger.info("[WARM] Pinging Ultravox...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{ultravox_url}/v1/chat/completions",
                json={
                    "model": os.environ.get("ULTRAVOX_MODEL", "mira-cbt"),
                    "messages": [
                        {"role": "system", "content": "คุณคือพี่มิร่า"},
                        {"role": "user", "content": "สวัสดีค่ะ"},
                    ],
                    "max_tokens": 30,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "stop": ["<|im_end|>"],
                },
                headers={"Authorization": "Bearer dummy-key"},
            )
        logger.info("[WARM] Ultravox ready (status %s)", r.status_code)
    except Exception as e:
        logger.warning("[WARM] Ultravox ping failed: %s", e)


async def entrypoint(ctx: JobContext):
    logger.info("New session (v5 cascading) — room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    from mira_graph.llm import prewarm_cerebras
    asyncio.create_task(prewarm_cerebras())

    _device = "cuda"
    logger.info("[STT] forcing device=cuda in subprocess (bypass fork-unsafe torch.cuda.is_available)")
    FastWhisperSTT.get(device=_device)

    await _warm_ultravox()

    silero_vad = silero.VAD.load(min_silence_duration=0.3)
    stt_plugin = StreamingFastWhisperSTT(vad=silero_vad, device=_device)

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
        vad=silero_vad,
        stt=stt_plugin,
        llm=_PassthroughLLM(),
        tts=tts,
        turn_handling={"endpointing": {"min_delay": 0.0}},
    )
    mira = MiraAgent(stt_plugin=stt_plugin)
    await session.start(agent=mira, room=ctx.room, room_input_options=RoomInputOptions())


def _prewarm(proc):
    # Load CT2 model weights into memory on CPU only.
    # CUDA cannot be initialised in a forked subprocess (fork-unsafe → SIGBUS);
    # the entrypoint will call FastWhisperSTT.get(device="cuda") on first use
    # inside the child process before any CUDA context exists in that process.
    FastWhisperSTT.get(device="cpu")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm))
