"""Heuristic phase transition logic — goal-based readiness checks per phase."""
from __future__ import annotations

from mira_graph.state import MiraState

# ── Readiness signals per phase ─────────────────────────────────────────────

# Emotional/problem content → CHECKIN goal met: user named something real
_EMOTIONAL_MARKERS = [
    "เครียด", "เหนื่อย", "เศร้า", "ท้อ", "กังวล", "กลัว", "โกรธ",
    "ไม่ไหว", "ไม่อยาก", "เบื่อ", "หนัก", "แย่", "ไม่ดี", "เจ็บ",
    "ทำไม่ได้", "ไม่ชอบ", "หงุดหงิด", "อยากหนี", "ติด", "กดดัน",
]

# Workable automatic thought patterns → EXPLORE goal met: thought is on the table
_THOUGHT_MARKERS = [
    "ผมไม่", "ผมมัน", "ฉันไม่", "ฉันมัน", "ทำไม่ได้", "ไม่ดีพอ",
    "คนอื่น", "ทุกคน", "ไม่มีใคร", "ตลอด", "ไม่เคย", "ต้อง",
    "ควรจะ", "ผิดที่ผม", "เพราะผม", "เพราะฉัน", "ไม่เก่ง",
]

# Cognitive movement patterns → WORK goal met: user shows doubt/reconsideration
_MOVEMENT_MARKERS = [
    "อาจจะ", "บางที", "ก็จริง", "อืม", "ไม่แน่", "หรือว่า",
    "คิดอีกที", "จริงด้วย", "เริ่มเห็น", "ก็ไม่เชิง", "อาจไม่",
]

# Deflecting / disengaged reply patterns — short + vague = user not engaging
_DEFLECT_MARKERS = [
    "ไม่รู้", "ก็ไม่รู้", "เฉยๆ", "เฉยๆ นะ", "ก็ชิล", "ชิลๆ",
    "โอเค", "โอเคครับ", "โอเคค่ะ", "ได้เลย", "ก็ได้",
    "ไม่เป็นไร", "ก็แล้วแต่", "ไม่แน่ใจ",
]

# Short deflection threshold (chars) — replies under this with no movement = disengaged
_SHORT_REPLY_CHARS = 30

# Positive-affect markers — short replies containing these are genuine engagement, not deflection
_POSITIVE_MARKERS = [
    "รู้สึกดี", "ดีขึ้น", "เข้าใจ", "ลองดู", "จะลอง", "ลองทำ",
    "ขอบคุณ", "ดีมาก", "ช่วยได้", "เห็นด้วย", "พยายาม", "ตั้งใจ",
    "จะพยาม", "โอเคแล้ว", "สบายใจ", "ดีใจ", "มีหวัง", "เริ่มดี",
]


def heuristic_judge(state: MiraState) -> dict:
    """
    Return {"transition": True} when the current phase's GOAL is met.

    Each phase has a specific goal. When it's met, the conversation is ready
    to move forward. Transitions are readiness-based with turn ceilings in
    judge.py as safety nets.
    """
    current = state.get("phase", "CHECKIN")
    turn_count = state.get("phase_turn_count", 0)
    user_text = state.get("user_text") or ""

    if current == "CHECKIN":
        # CHECKIN goal: user has named something real (emotion or problem)
        if turn_count >= 1 and _has_emotional_or_problem_content(user_text):
            return {"transition": True}

    elif current == "EXPLORE":
        # EXPLORE goal: a workable automatic thought is on the table
        if turn_count >= 1 and _has_workable_thought(state, user_text):
            return {"transition": True}

    elif current == "WORK":
        # WORK goal: user has engaged — shown movement, doubt, or a balanced view
        if turn_count >= 2 and _shows_cognitive_movement(user_text):
            return {"transition": True}
        # Stuck detector: if user is giving short/deflecting replies with no
        # movement after 3 WORK turns, pivot to WRAP — don't keep drilling
        if turn_count >= 3 and _is_disengaged(user_text):
            print(f"  [JUDGE 🔄] WORK stuck (turn={turn_count}, short/deflect) → WRAP")
            return {"transition": True}

    return {"transition": False}


def _has_emotional_or_problem_content(text: str) -> bool:
    return any(m in text for m in _EMOTIONAL_MARKERS)


def _has_workable_thought(state: MiraState, text: str) -> bool:
    if state.get("automatic_thought"):
        return True
    return any(m in text for m in _THOUGHT_MARKERS)


def _shows_cognitive_movement(text: str) -> bool:
    return any(m in text for m in _MOVEMENT_MARKERS)


def _is_disengaged(text: str) -> bool:
    """True when reply is short AND contains no movement, OR contains a deflect marker."""
    if any(m in text for m in _POSITIVE_MARKERS):
        return False
    if any(m in text for m in _DEFLECT_MARKERS):
        return True
    stripped = text.strip()
    if len(stripped) <= _SHORT_REPLY_CHARS and not _shows_cognitive_movement(stripped):
        return True
    return False
