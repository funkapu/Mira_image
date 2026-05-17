"""Counsellor Agent — Ultravox native audio (Phase 3.0) with text fallback."""
from __future__ import annotations

import re

from mira_graph.llm import apply_voice_mode, call_ultravox_audio, call_ultravox_pure, clean_response
from mira_graph.nodes._phase_base import STAGE_DISABLED, _load_prompt, _sliding_window, _trim_history_to_budget
from mira_graph.state import MiraState

# User text cap — single utterances longer than this are truncated before
# being fed into the LLM. Protects against rare STT bursts (user speaking
# 3+ sentences at once) blowing the 4096-token context window.
_USER_TEXT_MAX_CHARS = 500

_NAME_PATTERNS = [
    re.compile(r"ชื่อ\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียก(?:ผม|ฉัน|หนู)?ว่า\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียกว่า\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียก\s*([ก-๙]{1,12})\s*(?:ก็ได้|นะ|ครับ|ค่ะ|คะ)"),
    re.compile(r"ผม(?!ไม่)(?:ชื่อ)?\s*([ก-๙a-zA-Z]{2,15})"),
]
_STRIP_SUFFIX = re.compile(r"(ก็ได้|ครับ|ค่ะ|คะ|นะคะ|นะครับ|นะ|จ้า|เลย|ด้วย)+$")

_STRATEGY_FALLBACKS = {
    "A": "อืม เล่าต่อได้เลยนะคะ",
    "B": "หนักใจจริงๆ เลยนะคะ",
    "C": "เล่าต่อได้นะคะ พี่อยู่ตรงนี้",
    "D": "ฟังดูหนักใจเลยนะคะ",
    "E": "ที่บอกมาหมายความว่า... ถูกไหมคะ?",  # PARAPHRASE
    "F": "ถ้าเพื่อนเจอแบบนี้ จะบอกเพื่อนว่าอะไรคะ?",
    "G": "แปลว่า... ใช่ไหมคะ?",  # CLARIFY
    "H": "วันนี้พอแค่นี้นะคะ ดูแลตัวเองด้วยนะคะ",
    "I": "อืม เล่าต่อได้เลยนะคะ",
}


def _extract_name(text: str) -> str | None:
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = _STRIP_SUFFIX.sub("", m.group(1)).strip()
            if 1 <= len(name) <= 12:
                return name
    return None


def _build_supervisor_block(state: MiraState) -> str:
    parts = []
    for key, label in [
        ("supervisor_empathy", "Empathy"),
        ("supervisor_belief", "Belief"),
        ("supervisor_reflection", "Reflection"),
        ("supervisor_strategy", "Strategy"),
        ("supervisor_encouragement", "Encouragement"),
    ]:
        if state.get(key):
            # Truncate each supervisor's advice to ~200 chars to cap total
            # supervisor_block size and keep us within Ultravox's 4096-token
            # context window even when all 5 supervisors produce long output.
            advice = str(state[key])
            if len(advice) > 200:
                advice = advice[:200].rstrip() + "..."
            parts.append(f"[{label}]\n{advice}")

    if not parts:
        return ""

    return (
        "\n\n=== Supervisor Advice ===\n"
        "ใช้ข้อมูลด้านล่างประกอบการตอบ:\n\n"
        + "\n\n".join(parts)
        + "\n\nทำตาม Strategy ที่แนะนำ ตอบสั้น 1-2 ประโยค ภาษาไทยเท่านั้น\n"
        "ห้ามถามซ้ำสิ่งที่ Reflection บอกว่า user บอกแล้ว (already_told)\n"
        "ห้ามใส่ JSON, English, code หรือ tool calls ใน response\n"
    )


_QUESTION_MARKERS = ("?", "ไหม", "ยังไง", "อย่างไร", "เหรอ", "หรือเปล่า")


def _fallback(state: MiraState) -> str:
    """Context-aware fallback when LLM returned empty (e.g. <think> overflow).

    Priority:
      1. If AT captured → Socratic challenge (advances therapy, not loop)
      2. If user asked a question → acknowledge + ask to repeat (don't ignore)
      3. Supervisor encouragement / empathy suggest
      4. Supervisor strategy letter
      5. Last resort: "อืม เล่าต่อได้เลยนะคะ"
    """
    captured_at = state.get("automatic_thought")
    if captured_at:
        return (
            f"ลองคิดดูนะคะ — หลักฐานที่สนับสนุนความคิด '{captured_at}' "
            f"คืออะไรคะ? แล้วที่ขัดแย้งล่ะคะ?"
        )

    user_text = state.get("user_text") or ""
    if any(m in user_text for m in _QUESTION_MARKERS):
        return "ขอโทษนะคะ พี่ฟังไม่ค่อยชัด ลองถามอีกครั้งได้ไหมคะ"

    enc = state.get("supervisor_encouragement") or ""
    if "praise: yes" in enc:
        for line in enc.splitlines():
            if line.startswith("suggest:"):
                val = line.split(":", 1)[-1].strip()
                if val and val != "-":
                    return val

    emp = state.get("supervisor_empathy") or ""
    if "validate: yes" in emp:
        for line in emp.splitlines():
            if line.startswith("suggest:"):
                val = line.split(":", 1)[-1].strip()
                if val and val != "-":
                    return val

    strat = state.get("supervisor_strategy") or ""
    for line in strat.splitlines():
        if line.startswith("strategy:"):
            letter = line.split(":", 1)[-1].strip().upper()
            return _STRATEGY_FALLBACKS.get(letter, "อืม เล่าต่อได้เลยนะคะ")

    return "อืม เล่าต่อได้เลยนะคะ"


async def counsellor_node(state: MiraState) -> dict:
    phase = state.get("phase", "CHECKIN")
    system_prompt = _load_prompt(phase, state)
    system_prompt += _build_supervisor_block(state)

    # Phase 3.1: inform counsellor that quick acknowledgment was already sent
    quick = state.get("quick_response") or ""
    if quick:
        system_prompt += (
            f"\n\n[หมายเหตุ: ระบบส่งประโยคตอบรับไปแล้วว่า '{quick}' "
            "ห้ามพูดซ้ำประโยคนั้น ตอบต่อจากนั้นเลยค่ะ]"
        )

    # Inject recent Mira responses to prevent looping
    asked = state.get("asked_questions") or []
    if asked:
        recent = asked[-3:]
        system_prompt += "\n\nMira already said (DO NOT REPEAT OR ASK SAME THING):\n"
        for a in recent:
            system_prompt += f"  - {a}\n"

    system_prompt = apply_voice_mode(system_prompt, state)
    system_prompt += "\nห้ามขึ้นต้นด้วย 'เข้าใจ' ทุกรูปแบบ: เข้าใจค่ะ เข้าใจนะคะ เข้าใจเลยค่ะ พี่เข้าใจ\n"
    # Hard no-think reinforcement — model occasionally slips into Qwen3
    # thinking mode despite the /no_think header, which causes max_tokens
    # to be consumed by an unfinished <think> block, leaving an empty body
    # and triggering the generic fallback. Restate the constraint LAST.
    system_prompt += (
        "\nREPLY DIRECTLY IN THAI. DO NOT emit <think>...</think> blocks. "
        + (
            "Start the response with [EMOTION: ...] tag, "
            "then the Thai reply body. Nothing else.\n"
            if STAGE_DISABLED
            else "Start the response with [EMOTION: ...] and [NEXT_STAGE: ...] tags, "
                 "then the Thai reply body. Nothing else.\n"
        )
    )

    # ── Final instruction block — injected LAST so model attends to it ──────────
    # Placed at the end of system prompt, closest to generation.
    _raw_user_text = state.get("user_text") or ""
    if len(_raw_user_text) > _USER_TEXT_MAX_CHARS:
        user_text = _raw_user_text[-_USER_TEXT_MAX_CHARS:]
        print(f"  [COUNSELLOR ⚠️] user_text truncated {len(_raw_user_text)} → {len(user_text)} chars")
    else:
        user_text = _raw_user_text
    user_name = state.get("user_name") or "คุณ"
    captured_at = state.get("automatic_thought")

    # ── State-aware: count recent reflections & user confirmations to
    # detect reflection loops and force stage advance.
    asked_prev = state.get("asked_questions") or []
    recent_asked = asked_prev[-4:]
    reflection_count = sum(
        1 for q in recent_asked
        if ("ใช่ไหมคะ" in q or "ฟังดูเหมือน" in q)
    )
    # Count consecutive user confirmations in last 4 user messages
    recent_user_msgs = [
        (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        for m in (state.get("messages") or [])
        if (isinstance(m, dict) and m.get("role") == "user")
        or (not isinstance(m, dict) and getattr(m, "type", "") == "human")
    ][-4:]
    _CONFIRM_TOKENS = ("ใช่ครับ", "ใช่ค่ะ", "ใช่", "ครับ", "ค่ะ", "อืม")
    confirm_count = sum(
        1 for msg in recent_user_msgs
        if any(t in msg.strip() for t in _CONFIRM_TOKENS) and len(msg.strip()) <= 20
    )
    force_advance = reflection_count >= 2 and confirm_count >= 2

    if STAGE_DISABLED:
        # Stage-disabled path: universal reflect-or-challenge instruction,
        # no phase branching, no force_advance NEXT_STAGE directive.
        if user_text and len(user_text) > 5:
            preview = user_text[:25]
            if captured_at:
                reflect_example = f'"ฟังดูเหมือนตอนนั้นคุณกำลังคิดว่า \'{captured_at}\' ใช่ไหมคะ?"'
                at_hint = f'\nระบบจับ automatic thought ได้แล้ว: "{captured_at}" — ใช้ตัวนี้สะท้อนกลับ ห้ามเดาใหม่\n'
            else:
                reflect_example = f'"{preview}...เหรอคะ? ฟังดูเหมือนตอนนั้นคุณคิดว่า \'...\' ใช่ไหมคะ?"'
                at_hint = ""

            if reflection_count >= 2 and confirm_count >= 2:
                at_for_challenge = captured_at or "สิ่งที่เกิดขึ้น"
                system_prompt += f"""

=== CRITICAL: STOP REFLECTING ===
You asked {reflection_count} reflections and {user_name} confirmed {confirm_count} times in a row.
Continuing to reflect = loop (BAD). Move to Socratic challenge NOW. Example:
  "หลักฐานที่สนับสนุนความคิด '{at_for_challenge}' คืออะไรคะ? แล้วที่ขัดแย้งล่ะคะ?"
DO NOT: ask another "ฟังดูเหมือน...ใช่ไหมคะ?" / "รู้สึกยังไง" / "คิดอะไรในหัว".
"""
            else:
                system_prompt += f"""

=== คำสั่งสำคัญที่สุด ===
{user_name} เพิ่งบอกว่า: "{user_text}"
→ ฟัง รับฟัง อนุมาน automatic thought แล้วสะท้อนกลับ ({user_name} ยืนยัน)
ตัวอย่าง: {reflect_example}{at_hint}
ห้าม: "รู้สึกยังไงคะ" / "ตอนนั้นคิดอะไรอยู่" / "คิดยังไงคะ" / ขึ้นต้น "เข้าใจเลยค่ะ"
"""
    elif phase in ("CHECKIN", "EXPLORE") and user_text and len(user_text) > 5:
        preview = user_text[:25]
        if captured_at:
            reflect_example = f'"ฟังดูเหมือนตอนนั้นคุณกำลังคิดว่า \'{captured_at}\' ใช่ไหมคะ?"'
            at_hint = f'\nระบบจับ automatic thought ได้แล้ว: "{captured_at}" — ใช้ตัวนี้สะท้อนกลับ ห้ามเดาใหม่\n'
        else:
            reflect_example = f'"{preview}...เหรอคะ? ฟังดูเหมือนตอนนั้นคุณคิดว่า \'...\' ใช่ไหมคะ?"'
            at_hint = ""

        if force_advance:
            # User confirmed 2+ reflections — STOP reflecting, ADVANCE stage NOW.
            at_for_challenge = captured_at or "สิ่งที่เกิดขึ้น"
            system_prompt += f"""

=== CRITICAL: STOP REFLECTING — ADVANCE NOW ===
You asked {reflection_count} reflections and {user_name} confirmed {confirm_count} times in a row.
Continuing to reflect = reflection loop (BAD).
→ Set [NEXT_STAGE: WORK]
→ Start Socratic challenge IMMEDIATELY. Example:
  "หลักฐานที่สนับสนุนความคิด '{at_for_challenge}' คืออะไรคะ? แล้วที่ขัดแย้งล่ะคะ?"
DO NOT: ask another "ฟังดูเหมือน...ใช่ไหมคะ?" / "รู้สึกยังไง" / "คิดอะไรในหัว" / "คิดยังไง".
"""
        else:
            system_prompt += f"""

=== คำสั่งสำคัญที่สุด ===
{user_name} เพิ่งบอกว่า: "{user_text}"
→ ขั้นตอน 1: พูดซ้ำสิ่งที่ {user_name} บอก ด้วยคำของเรา (paraphrase สั้นๆ)
→ ขั้นตอน 2: อนุมาน automatic thought จากบริบท แล้วสะท้อนกลับให้ {user_name} ยืนยัน (ห้ามถาม)
ตัวอย่าง: {reflect_example}{at_hint}
ห้าม: "รู้สึกยังไงคะ" / "เป็นยังไงบ้างคะ" / "ตอนนั้นคิดอะไรอยู่" / "มีความคิดอะไรผ่านหัว" / "คิดยังไงคะ" / ขึ้นต้น "เข้าใจเลยค่ะ"
"""
