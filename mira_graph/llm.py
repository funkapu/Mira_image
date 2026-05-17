"""LLM clients — Cerebras primary, Groq optional fallback.

Provider chain per call:
  1. Modal Ultravox endpoint  — when USE_REAL_ULTRAVOX=true (Phase 2.0a)
  2. Cerebras gpt-oss-120b    — primary text/tool-calling model
  3. Groq llama-3.3-70b       — fallback, optional
  4. raise RuntimeError        — caller uses heuristic
"""
from __future__ import annotations

import asyncio
import os
import re
import time

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from response."""
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()


_UNDERSTAND_PREFIX = re.compile(
    # Match "(พี่)?เข้าใจ(แล้ว|เลย|มาก)?(ค่ะ|นะคะ|นะ|ครับ)+ ..."
    # Polite particle is REQUIRED — leaves "เข้าใจไหม/ถูกไหม/อะไร" intact.
    r"^(?:พี่)?เข้าใจ(?:แล้ว|เลย|มาก)?(?:ค่ะ|นะคะ|นะ|ครับ)+[ ,—\-:]*"
)

# Emoji + symbol stripper. Mira's prompt says "no emoji" but LoRA-baked behavior
# still slips them in. TTS reads "หน้ายิ้ม" aloud which sounds odd.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # Misc symbols + emoticons + transport + supplemental
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F680-\U0001F6FF"   # Transport + map
    "\U0001F1E0-\U0001F1FF"   # Flags
    "☀-➿"            # Misc symbols + dingbats
    "⌀-⏿"            # Misc technical
    "︀-️"            # Variation selectors
    "‍"                   # Zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# ── REASONING/SPOKEN parsing (debug telemetry) ────────────────────────────────
# Mira's prompt instructs the model to wrap output as:
#   [REASONING]...phase/symptoms/crisis_flag/...[/REASONING]
#   [SPOKEN]<thai response>[/SPOKEN]
# The reasoning block is logged for debugging; only [SPOKEN] reaches TTS.

_REASONING_RE = re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL)
_SPOKEN_RE = re.compile(r"\[SPOKEN\](.*?)\[/SPOKEN\]", re.DOTALL)
_TAG_STRIP_RE = re.compile(r"\[/?(?:REASONING|SPOKEN)\]")


def parse_mira_output(raw: str) -> dict:
    """Split Mira's tagged output into reasoning + spoken halves.

    Falls back gracefully when tags are missing (e.g. model regressed):
    treats the raw text as spoken with empty reasoning.
    """
    rm = _REASONING_RE.search(raw)
    sm = _SPOKEN_RE.search(raw)
    reasoning = rm.group(1).strip() if rm else ""
    if sm:
        spoken = sm.group(1).strip()
    else:
        spoken = _TAG_STRIP_RE.sub("", raw).strip()
    return {"reasoning": reasoning, "spoken": spoken, "raw": raw}


class ReasoningStreamFilter:
    """State machine that strips [REASONING] from a streaming response.

    Feed each streamed sentence/chunk via .feed(); it returns ONLY the portion
    that should be sent to TTS (i.e. the [SPOKEN] payload). Reasoning content
    is buffered into .reasoning for post-stream logging.

    State graph:
        pre        -> sees [REASONING] or [SPOKEN]
        reasoning  -> collects until [/REASONING]
        between    -> waits for [SPOKEN] tag (skips whitespace)
        spoken     -> emits to TTS until [/SPOKEN]
        done       -> ignores remaining tokens

    Tolerates malformed output: if no tags at all, falls back to passing
    everything through as spoken (so old pre-format prompts still work).
    """

    def __init__(self):
        self._state = "pre"
        self._reasoning_chunks: list[str] = []

    @property
    def reasoning(self) -> str:
        return "\n".join(self._reasoning_chunks).strip()

    @property
    def state(self) -> str:
        return self._state

    def feed(self, sentence: str) -> str:
        out = ""
        text = sentence
        while text:
            if self._state == "pre":
                if "[REASONING]" in text:
                    _, _, text = text.partition("[REASONING]")
                    self._state = "reasoning"
                elif "[SPOKEN]" in text:
                    _, _, text = text.partition("[SPOKEN]")
                    self._state = "spoken"
                else:
                    # Model didn't follow format — emit as spoken
                    out += text
                    text = ""
            elif self._state == "reasoning":
                if "[/REASONING]" in text:
                    before, _, text = text.partition("[/REASONING]")
                    if before:
                        self._reasoning_chunks.append(before)
                    self._state = "between"
                else:
                    self._reasoning_chunks.append(text)
                    text = ""
            elif self._state == "between":
                if "[SPOKEN]" in text:
                    _, _, text = text.partition("[SPOKEN]")
                    self._state = "spoken"
                else:
                    text = ""  # discard whitespace between blocks
            elif self._state == "spoken":
                if "[/SPOKEN]" in text:
                    before, _, _ = text.partition("[/SPOKEN]")
                    out += before
                    self._state = "done"
                    text = ""
                else:
                    out += text
                    text = ""
            else:  # done
                break
        return out


def clean_response(text: str) -> str:
    """Strip thinking tags, emojis, banned 'เข้าใจ' opener; normalize whitespace.

    Multi-stage clean:
      1. Strip <think>...</think> blocks (Qwen3 thinking mode)
      2. Strip emojis (LoRA still slips them despite prompt ban → TTS reads "หน้ายิ้ม")
      3. Strip leading "(พี่)?เข้าใจ..." opener
      4. Trim to non-empty result, else fallback
    """
    out = strip_thinking(text).strip()
    out = _EMOJI_RE.sub("", out).strip()
    m = _UNDERSTAND_PREFIX.match(out)
    if m:
        rest = out[m.end():].lstrip()
        if len(rest) < 4:
            return "อืมค่ะ"
        return rest
    return out


_VOICE_MODE_INSTRUCTION = "\n\n=== VOICE MODE ===\nตอบสั้น 1-2 ประโยคเท่านั้น ห้ามใช้ markdown (**, *, #, -) เพราะ TTS จะอ่านสัญลักษณ์ออกมา\n"


def apply_voice_mode(system_prompt: str, state: dict) -> str:
    """Append voice-mode instruction when running in voice pipeline."""
    if state.get("_voice_mode"):
        return system_prompt + _VOICE_MODE_INSTRUCTION
    return system_prompt

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

# ── Modal Ultravox endpoint (Phase 2.0a) ──────────────────────────────────────
# When USE_REAL_ULTRAVOX=true and ULTRAVOX_URL is set, call_ultravox_mock()
# routes to the real Modal endpoint instead of Cerebras gpt-oss-120b.
_USE_REAL_ULTRAVOX = os.environ.get("USE_REAL_ULTRAVOX", "").lower() in ("1", "true", "yes")
_ULTRAVOX_URL = os.environ.get("ULTRAVOX_URL", "").rstrip("/")
_ULTRAVOX_MODEL = os.environ.get("ULTRAVOX_MODEL", "mira-cbt")
_ULTRAVOX_API_KEY = os.environ.get("ULTRAVOX_API_KEY", "dummy-key")

if _USE_REAL_ULTRAVOX and _ULTRAVOX_URL:
    print(f"[LLM] Ultravox endpoint: {_ULTRAVOX_URL} (model={_ULTRAVOX_MODEL})")
else:
    print("[LLM] Ultravox: using Cerebras gpt-oss-120b mock")

# ── Cerebras (primary) ────────────────────────────────────────────────────────
_cerebras_client: Cerebras | None = None


def _get_cerebras() -> Cerebras:
    global _cerebras_client
    if _cerebras_client is None:
        key = os.environ.get("CEREBRAS_API_KEY")
        if not key:
            raise RuntimeError("CEREBRAS_API_KEY not set")
        _cerebras_client = Cerebras(api_key=key)
    return _cerebras_client


# ── Groq (fallback, optional) ─────────────────────────────────────────────────
_groq_client = None

try:
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        print("[LLM] Groq configured as fallback")
except ImportError:
    pass  # groq not installed — Cerebras only

# ── Model names (providers use different suffixes) ────────────────────────────
# gpt-oss-120b: 120B quality model for Mira's voice.
# llama3.1-8b: fast 8B for classify nodes (judge, analyze).
_CEREBRAS_QUALITY = "gpt-oss-120b"
_CEREBRAS_FAST = "llama3.1-8b"
_GROQ_QUALITY = "llama-3.3-70b-versatile"
_GROQ_FAST = "llama-3.1-8b-instant"

# ── Timeout / retry constants (exported for node imports) ─────────────────────
_TIMEOUT_SUBAGENT: float = 10.0
_TIMEOUT_JUDGE: float = 5.0
_TIMEOUT_CHAT: float = 20.0
_MAX_RETRIES: int = 2
_BACKOFF_BASE: float = 0.3


# ── Sync inner calls (run via asyncio.to_thread) ──────────────────────────────

def _sync_cerebras(messages, max_tokens, temperature, json_mode, model) -> str:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # gpt-oss-120b on Cerebras supports reasoning_effort. Default "medium" can
    # eat the entire token budget on long Thai prompts and leave content empty.
    # "low" keeps thinking minimal so the structured answer always lands in content.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    resp = _get_cerebras().chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    # Fallback: some gpt-oss responses put the actual answer in reasoning_content
    # when content is empty (model exhausted budget on reasoning then truncated).
    if not content.strip():
        content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    return content


def _sync_groq(messages, max_tokens, temperature, json_mode, model) -> str:
    if _groq_client is None:
        raise RuntimeError("Groq fallback not configured")
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _groq_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ── Core call with fallback chain ─────────────────────────────────────────────

async def call_llm(
    messages: list,
    max_tokens: int = 100,
    temperature: float = 0.0,
    json_mode: bool = False,
    timeout: float = _TIMEOUT_SUBAGENT,
    use_fast: bool = False,
) -> tuple[str, float]:
    """LLM call: Cerebras → Groq → raise.  Returns (text, latency_ms)."""
    c_model = _CEREBRAS_FAST if use_fast else _CEREBRAS_QUALITY
    g_model = _GROQ_FAST if use_fast else _GROQ_QUALITY
    last_exc: Exception | None = None

    # 1 ── Cerebras with retry
    for attempt in range(_MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_sync_cerebras, messages, max_tokens, temperature, json_mode, c_model),
                timeout=timeout + 1,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if attempt:
                print(f"  [CEREBRAS] retry {attempt + 1} succeeded")
            return text, latency_ms
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            last_exc = exc
            tag = "timeout" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
            print(f"  [CEREBRAS ⚠️] {tag} {latency_ms:.0f}ms (attempt {attempt + 1}/{_MAX_RETRIES})")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))

    # 2 ── Groq fallback
    if _groq_client is not None:
        print("  [LLM] Cerebras failed — trying Groq fallback")
        t0 = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_sync_groq, messages, max_tokens, temperature, json_mode, g_model),
                timeout=timeout + 1,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [GROQ ✅] fallback succeeded {latency_ms:.0f}ms")
            return text, latency_ms
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [GROQ ❌] {type(exc).__name__} {latency_ms:.0f}ms")
            last_exc = exc

    # 3 ── All providers exhausted — caller uses heuristic
    raise RuntimeError(f"All LLM providers failed: {last_exc}") from last_exc


async def call_llm_fast(
    messages: list,
    max_tokens: int = 100,
    temperature: float = 0.0,
    json_mode: bool = False,
    timeout: float = 3.0,
) -> tuple[str, float]:
    """Fast 8B model — for classification (judge, analyze)."""
    return await call_llm(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        timeout=timeout,
        use_fast=True,
    )


async def _call_real_ultravox(
    messages: list,
    max_tokens: int,
) -> tuple[str, float]:
    """Call the Modal Ultravox endpoint (OpenAI-compatible API)."""
    import httpx

    t0 = time.perf_counter()
    payload = {
        "model": _ULTRAVOX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.83,
        "chat_template_kwargs": {"enable_thinking": True},
        "stop": ["<|im_end|>"],
    }
    headers = {
        "Authorization": f"Bearer {_ULTRAVOX_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT_CHAT + 10) as client:
        r = await client.post(
            f"{_ULTRAVOX_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        text: str = data["choices"][0]["message"]["content"] or ""
    text = strip_thinking(text)
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"  [ULTRAVOX ✅] {latency_ms:.0f}ms")
    return text, latency_ms


async def call_ultravox_mock(
    system_prompt: str,
    user_text: str,
    history: list,
    max_tokens: int = 200,
) -> tuple[str, float]:
    """Ultravox call — routes to Modal endpoint when USE_REAL_ULTRAVOX=true,
    otherwise falls back to Cerebras gpt-oss-120b as a quality stand-in.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            messages.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            messages.append(msg)
    messages.append({"role": "user", "content": user_text or ""})

    if _USE_REAL_ULTRAVOX and _ULTRAVOX_URL:
        try:
            return await _call_real_ultravox(messages, max_tokens)
        except Exception as exc:
            print(f"  [ULTRAVOX ⚠️] {type(exc).__name__} — falling back to Cerebras")

    return await call_llm(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        timeout=_TIMEOUT_CHAT,
    )


async def prewarm_cerebras() -> None:
    """Pre-warm Cerebras (and Groq if configured)."""
    tasks = [call_llm(messages=[{"role": "user", "content": "hi"}], max_tokens=5, timeout=10.0)]
    if _groq_client is not None:
        tasks.append(
            call_llm(messages=[{"role": "user", "content": "hi"}], max_tokens=5, timeout=10.0, use_fast=True)
        )
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
        print("[PREWARM] Cerebras ready" + (" + Groq" if _groq_client else ""))
    except Exception as e:
        print(f"[PREWARM] non-critical: {e}")


# Alias
call_cerebras = call_llm


# ── Tool-calling (Groq primary, Cerebras llama3.1-8b fallback) ────────────────

def _sync_groq_tools(messages: list, tools: list, max_tokens: int, temperature: float):
    if _groq_client is None:
        raise RuntimeError("Groq not configured")
    import json
    resp = _groq_client.chat.completions.create(
        model=_GROQ_QUALITY,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return content, tool_calls


def _sync_cerebras_tools(messages: list, tools: list, max_tokens: int, temperature: float):
    import json
    resp = _get_cerebras().chat.completions.create(
        model=_CEREBRAS_QUALITY,  # quality model — proper function calling support
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return content, tool_calls


def _build_messages(system_prompt: str, user_text: str, history: list) -> list:
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            msgs.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            msgs.append(msg)
    # Dedup: history already contains the user turn (appended at _run_mira_turn:700).
    # Only append if the last message is NOT already a user role to avoid [user, user].
    if not msgs or msgs[-1].get("role") != "user":
        msgs.append({"role": "user", "content": "/no_think\n" + (user_text or "")})
    return msgs


_JSON_BLOCK_RE = __import__("re").compile(
    r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}|```[\w]*\s*\{[\s\S]*?\}\s*```)", __import__("re").DOTALL
)
_TOOL_KEYWORDS = (
    "track_mood", "track_concern", "track_automatic_thought",
    "transition_phase", "trigger_crisis_protocol",
    '"tool":', '"action":', '"name":', '"arguments":', '"args":', '"parameters":', '"distortion_type":',
    "We need to", "final answer with", "produce final",
    "We will call", "I will call", "I need to call",
)


def _content_is_clean(content: str) -> bool:
    """Return True if content is natural language with no JSON/tool artifacts."""
    stripped = content.strip()
    if not stripped:
        return False
    if _JSON_BLOCK_RE.search(stripped):
        return False
    if any(kw in stripped for kw in _TOOL_KEYWORDS):
        return False
    return True


_TOOL_SECTION_MARKERS = ("=== Available Tools ===", "=== Tool Usage ===")

_LEADING_JUNK_RE = __import__("re").compile(
    r"^[\s\S]*?(?=[฀-๿])",  # skip everything before first Thai char
    __import__("re").DOTALL,
)
_EMBEDDED_JSON_RE = __import__("re").compile(
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}",
    __import__("re").DOTALL,
)


def _strip_tools_section(messages: list) -> list:
    """Remove tool-related sections from system message and add no-JSON directive."""
    result = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content") or ""
            for marker in _TOOL_SECTION_MARKERS:
                if marker in content:
                    content = content.split(marker)[0].rstrip()
            content += "\n\nตอบเป็นประโยคภาษาไทยธรรมชาติเท่านั้น ห้าม JSON ห้าม code block"
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


def _salvage_thai(content: str) -> str:
    """Extract Thai text from JSON-polluted content as a last resort."""
    # Remove embedded JSON blocks (up to 2-level nesting)
    cleaned = _EMBEDDED_JSON_RE.sub("", content)
    # Strip leading non-Thai junk
    m = _LEADING_JUNK_RE.match(cleaned)
    if m and m.end() < len(cleaned):
        cleaned = cleaned[m.end():]
    cleaned = cleaned.strip()
    return cleaned if cleaned else content.strip()


async def call_with_tools(
    system_prompt: str,
    user_text: str,
    history: list,
    tools: list,
    max_tokens: int = 1500,
    timeout: float = 25.0,
) -> tuple[str, float, list]:
    """LLM call with function calling — Cerebras primary, Groq fallback.
    Two-step: (1) tool extraction call, (2) text generation if content was empty.
    Returns (text_response, latency_ms, tool_calls).
    """
    messages = _build_messages(system_prompt, user_text, history)
    last_exc: Exception | None = None
    total_ms: float = 0.0

    # 1 ── Cerebras primary (gpt-oss-120b — proper function calling)
    content: str = ""
    tool_calls: list = []
    for attempt in range(_MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            content, tool_calls = await asyncio.wait_for(
                asyncio.to_thread(_sync_cerebras_tools, messages, tools, max_tokens, 0.7),
                timeout=timeout,
            )
            total_ms += (time.perf_counter() - t0) * 1000
            if attempt:
                print(f"  [CEREBRAS] retry {attempt + 1} succeeded")
            break
        except Exception as exc:
            total_ms += (time.perf_counter() - t0) * 1000
            last_exc = exc
            tag = "timeout" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
            print(f"  [CEREBRAS ⚠️] {tag} {total_ms:.0f}ms (attempt {attempt + 1}/{_MAX_RETRIES})")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
            else:
                # Try Groq fallback
                if _groq_client is not None:
                    print("  [LLM] Cerebras failed — trying Groq tool fallback")
                    t0 = time.perf_counter()
                    try:
                        content, tool_calls = await asyncio.wait_for(
                            asyncio.to_thread(_sync_groq_tools, messages, tools, max_tokens, 0.7),
                            timeout=timeout,
                        )
                        total_ms += (time.perf_counter() - t0) * 1000
                        print(f"  [GROQ FALLBACK ✅] {total_ms:.0f}ms")
                    except Exception as e2:
                        total_ms += (time.perf_counter() - t0) * 1000
                        print(f"  [GROQ FALLBACK ❌] {type(e2).__name__} {total_ms:.0f}ms")
                        raise RuntimeError(f"All providers failed: {e2}") from e2
                else:
                    raise RuntimeError(f"All providers failed: {last_exc}") from last_exc

    # Strip thinking tokens immediately — must happen before content_is_clean check
    content = strip_thinking(content)

    # 2 ── If content is absent or JSON-polluted, regenerate without tool instructions
    #       Include tool context so model knows what was already done
    if not _content_is_clean(content):
        clean_msgs = _strip_tools_section(messages)

        # Append a note about which tools fired so model responds naturally
        if tool_calls:
            tool_ctx = ", ".join(tc["name"] for tc in tool_calls)
            clean_msgs = clean_msgs + [{
                "role": "user",
                "content": (
                    f"[ระบบ: เรียก {tool_ctx} แล้ว] "
                    "ตอนนี้ตอบ user เป็นภาษาไทยธรรมชาติ 1-2 ประโยค ห้าม JSON"
                ),
            }]

        t0 = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(
                    _sync_cerebras, clean_msgs, 400, 0.7, False, _CEREBRAS_QUALITY
                ),
                timeout=15.0,
            )
            total_ms += (time.perf_counter() - t0) * 1000
            text = strip_thinking(text)
            if _content_is_clean(text):
                content = text
                print(f"  [FOLLOW-UP] ok: '{text[:60]}'")
            else:
                salvaged = _salvage_thai(text)
                if _content_is_clean(salvaged) and len(salvaged) > 5:
                    print(f"  [FOLLOW-UP] salvaged Thai from dirty response")
                    content = salvaged
                else:
                    print(f"  [FOLLOW-UP] salvage failed — using empty content")
                    content = ""
        except Exception as exc:
            total_ms += (time.perf_counter() - t0) * 1000
            print(f"  [FOLLOW-UP ⚠️] {type(exc).__name__}")

    return content, total_ms, tool_calls


# ── Ultravox pure — NO tools, just text generation ───────────────────────────

async def call_ultravox_pure(
    system_prompt: str,
    user_text: str,
    history: list,
    max_tokens: int = 500,
    timeout: float = 35.0,
) -> tuple[str, float]:
    """Call Ultravox CBT LoRA — no tools, pure Thai text generation.
    Falls back to Cerebras gpt-oss-120b if Ultravox unavailable.
    Returns (response_text, latency_ms).
    """
    messages = _build_messages(system_prompt, user_text, history)
    t0 = time.perf_counter()

    # Try Ultravox CBT LoRA
    if _USE_REAL_ULTRAVOX and _ULTRAVOX_URL:
        try:
            import httpx
            payload = {
                "model": _ULTRAVOX_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.83,
                "chat_template_kwargs": {"enable_thinking": True},
                "stop": ["<|im_end|>"],
            }
            headers = {
                "Authorization": f"Bearer {_ULTRAVOX_API_KEY}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{_ULTRAVOX_URL}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
            text = clean_response(text)
            ms = (time.perf_counter() - t0) * 1000
            if text.strip():
                print(f"  [UV-PURE] {ms:.0f}ms")
                return text, ms
            print(f"  [UV-PURE ⚠️] Empty response, falling back")
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [UV-PURE ⚠️] {type(exc).__name__} {ms:.0f}ms, falling back")

    # Fallback: Cerebras gpt-oss-120b, no tools
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_cerebras, messages, max_tokens, 0.7, False, _CEREBRAS_QUALITY
            ),
            timeout=timeout,
        )
        text = clean_response(text)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [CEREBRAS-PURE] {ms:.0f}ms")
        return text, ms
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [LLM ⚠️] All failed: {type(exc).__name__}")
        return "", ms


def _is_english(text: str) -> bool:
    """Return True if text is predominantly ASCII (English drift from audio tower)."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return (ascii_chars / total_alpha) > 0.85


# ── Ultravox audio — native audio input, no STT needed ───────────────────────

async def call_ultravox_audio(
    system_prompt: str,
    audio_b64: str,
    history: list,
    max_tokens: int = 120,
    timeout: float = 30.0,
) -> tuple[str, float]:
    """Call Ultravox with native OGG audio — Ultravox audio_tower processes it directly.
    Returns (response_text, latency_ms). Falls back to Cerebras on any error.
    """
    if not (_USE_REAL_ULTRAVOX and _ULTRAVOX_URL):
        return "", 0.0

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            messages.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            messages.append(msg)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "สำคัญมาก: ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ ตอบสั้นๆ 1-2 ประโยค"},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/ogg;base64,{audio_b64}"}},
        ],
    })

    t0 = time.perf_counter()
    try:
        import httpx
        payload = {"model": _ULTRAVOX_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.83, "chat_template_kwargs": {"enable_thinking": True}, "stop": ["<|im_end|>"]}
        headers = {"Authorization": f"Bearer {_ULTRAVOX_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{_ULTRAVOX_URL}/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        text = clean_response(text)
        ms = (time.perf_counter() - t0) * 1000
        if _is_english(text):
            print(f"  [UV-AUDIO ⚠️ LANG-DRIFT] {ms:.0f}ms → dropped English: '{text[:60]}'")
            return "", ms
        print(f"  [UV-AUDIO] {ms:.0f}ms")
        return text, ms
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [UV-AUDIO ⚠️] {type(exc).__name__} {ms:.0f}ms")
        return "", ms


# ── Streaming Ultravox — sentence-by-sentence TTS (Phase 3.2) ────────────────

# Thai sentence endings — checked longest-first to avoid partial matches.
# NOTE: '?' and '!' removed — Thai polite forms (คะ/ค่ะ/ครับ) already trigger
# flush; trailing '?' would arrive as a separate token AFTER flush, causing
# the next sentence to start with a stray '?' and TTS to break audibly.
_SENTENCE_ENDINGS = [
    "ไหมครับ", "ไหมคะ", "มั้ยครับ", "มั้ยคะ",
    "บ้างครับ", "บ้างคะ", "หรือครับ", "หรือคะ",
    "ด้วยครับ", "ด้วยคะ", "ด้วยค่ะ",
    "เลยครับ", "เลยคะ", "เลยค่ะ", "เลยนะคะ",
    "อะครับ", "อะคะ",
    "นะคะ", "นะค่ะ", "นะครับ",
    "ค่ะ", "คะ", "ครับ",
    "นะ",
]


def _build_audio_messages(system_prompt: str, audio_b64: str, history: list, user_prefix: str = "") -> list:
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            msgs.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            msgs.append(msg)
    base_text = "/no_think\nสำคัญมาก: ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ ตอบสั้นๆ 1-2 ประโยค"
    text_payload = (user_prefix + "\n\n" + base_text) if user_prefix else base_text
    msgs.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text_payload},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/ogg;base64,{audio_b64}"}},
        ],
    })
    return msgs

async def _stream_sentences(messages: list, max_tokens: int, timeout: float):
    """Core SSE streaming — yields clean Thai sentences from Ultravox."""
    import json as _json
    import httpx

    payload = {
        "model": _ULTRAVOX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.83,
        "stream": True,
        # Thinking DISABLED — model produces Thai response directly (no <think> block).
        # Lower TTFT: skips ~500-2000 thinking tokens before first spoken char.
        "chat_template_kwargs": {"enable_thinking": True},
        # Stop on </think> as defense-in-depth: if model leaks thinking, cut early.
        "stop": ["<|im_end|>"],
    }
    headers = {
        "Authorization": f"Bearer {_ULTRAVOX_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    buffer = ""
    yielded_any = False

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{_ULTRAVOX_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as r:
                r.raise_for_status()
                async for raw_line in r.aiter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    data = raw_line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data)
                        token = (chunk["choices"][0]["delta"].get("content") or "")
                    except Exception:
                        continue
                    if not token:
                        continue

                    buffer += token

                    # Safety net: strip mid-think if /no_think missed
                    if "<think>" in buffer and "</think>" not in buffer:
                        continue
                    if "</think>" in buffer:
                        import re as _re
                        buffer = _re.sub(r"<think>[\s\S]*?</think>\s*", "", buffer)
                        if not buffer.strip():
                            continue

                    buf = buffer.rstrip()
                    should_flush = any(buf.endswith(e) for e in _SENTENCE_ENDINGS)

                    if should_flush:
                        sentence = clean_response(buffer.strip())
                        if len(sentence) > 3:
                            ms = (time.perf_counter() - t0) * 1000
                            if _is_english(sentence):
                                print(f"  [UV-STREAM ⚠️ LANG-DRIFT] {ms:.0f}ms → dropped English: '{sentence[:60]}'")
                                buffer = ""
                                continue
                            # Hold short fragments (< 15 chars) — likely a continuation
                            # clause (e.g. "ว่า อะไร...") that arrived after a flush;
                            # keep in buffer so it merges with the next sentence.
                            if len(sentence) < 15:
                                print(f"  [UV-STREAM] {ms:.0f}ms → held short fragment: '{sentence[:60]}'")
                                continue
                            print(f"  [UV-STREAM] {ms:.0f}ms → '{sentence[:60]}'")
                            yield sentence
                            yielded_any = True
                            buffer = ""

        # flush remainder — BUT drop buffer if <think> never closed.
        # max_tokens truncating mid-thinking would otherwise leak English reasoning.
        if buffer.strip():
            if "<think" in buffer and "</think>" not in buffer:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  [UV-STREAM ⚠️] {ms:.0f}ms → dropped unclosed-think flush ({len(buffer)} chars)")
            else:
                sentence = clean_response(buffer.strip())
                if len(sentence) > 2:
                    ms = (time.perf_counter() - t0) * 1000
                    if _is_english(sentence):
                        print(f"  [UV-STREAM ⚠️ LANG-DRIFT] {ms:.0f}ms → dropped English flush: '{sentence[:60]}'")
                    else:
                        print(f"  [UV-STREAM] {ms:.0f}ms → '{sentence[:60]}' (flush)")
                        yield sentence
                        yielded_any = True

        ms = (time.perf_counter() - t0) * 1000
        print(f"  [UV-STREAM] total {ms:.0f}ms")

    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [UV-STREAM ⚠️] {type(exc).__name__} {ms:.0f}ms")
        if buffer.strip() and not ("<think" in buffer and "</think>" not in buffer):
            yield clean_response(buffer.strip())
            yielded_any = True

    if not yielded_any:
        yield ""


async def call_ultravox_audio_stream(
    system_prompt: str,
    audio_b64: str,
    history: list,
    max_tokens: int = 500,  # thinking-OFF — 500 fits in 4096 ctx with audio prompts up to ~3500 tokens
    timeout: float = 30.0,
    user_prefix: str = "",
):
    """Stream Ultravox with native audio — yields Thai sentences as they complete."""
    if not (_USE_REAL_ULTRAVOX and _ULTRAVOX_URL):
        # Fallback: non-streaming cerebras, yield full response as one sentence
        messages = _build_audio_messages(system_prompt, audio_b64, history, user_prefix)
        text, _ = await call_llm(messages=messages, max_tokens=max_tokens, temperature=0.7, timeout=timeout)
        if text.strip():
            yield clean_response(text)
        return

    messages = _build_audio_messages(system_prompt, audio_b64, history, user_prefix)
    async for sentence in _stream_sentences(messages, max_tokens, timeout):
        yield sentence
async def call_ultravox_pure_stream(
    system_prompt: str,
    user_text: str,
    history: list,
    max_tokens: int = 500,  # thinking-OFF — 500 fits in 4096 ctx with audio prompts up to ~3500 tokens
    timeout: float = 30.0,
):
    """Stream Ultravox text-only — yields Thai sentences as they complete."""
    if not (_USE_REAL_ULTRAVOX and _ULTRAVOX_URL):
        messages = _build_messages(system_prompt, user_text, history)
        text, _ = await call_llm(messages=messages, max_tokens=max_tokens, temperature=0.7, timeout=timeout)
        if text.strip():
            yield clean_response(text)
        return

    messages = _build_messages(system_prompt, user_text, history)
    async for sentence in _stream_sentences(messages, max_tokens, timeout):
        yield sentence


# ── Supervisor call — same Mira endpoint, low temp, no tools ─────────────────

import random as _random

_QUICK_FILLERS = [
    "อืมมค่ะ",           # Mm-hmm
    "อ่าค่ะ",            # Uh-huh
    "อ่อ เหรอคะ",           # I see
    "ค่ะ",               # Yes
    "เล่าต่อได้เลยค่ะ",  # Go on
    "ใช่ค่ะ",            # Right
    "โอเคค่ะ",           # Okay
]


def get_quick_filler() -> str:
    """Phase 3.1: instant filler — no LLM call, picked from curated list."""
    return _random.choice(_QUICK_FILLERS)


async def call_slm_quick(user_text: str, phase: str, timeout: float = 3.0) -> str:
    """Phase 3.1: kept for API compatibility — now returns a hardcoded filler instantly."""
    filler = get_quick_filler()
    print(f"  [QUICK-FILLER] → '{filler}'")
    return filler


async def call_supervisor(
    system_prompt: str,
    user_text: str,
    context: str,
    max_tokens: int = 150,
    timeout: float = 12.0,
) -> str:
    """Supervisor call — Cerebras gpt-oss-120b, English output, structured classification."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser said:\n{user_text}"},
    ]
    t0 = time.perf_counter()
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_cerebras, messages, max_tokens, 0.1, False, _CEREBRAS_QUALITY
            ),
            timeout=timeout,
        )
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [SUP] {ms:.0f}ms")
        # Only strip <think> blocks — DO NOT run full clean_response here.
        # clean_response strips Thai เข้าใจ-prefix which would corrupt structured
        # English fields like "chosen_symptom: ..." if any field starts with "ค่ะ"/etc.
        return strip_thinking(text).strip()
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [SUP ⚠️] {type(exc).__name__} {ms:.0f}ms")
        return ""
