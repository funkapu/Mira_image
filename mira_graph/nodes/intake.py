"""PHQ-9 Covert Intake — alternates small talk with PHQ-9 questions.

Flow:
    Turn 1: Mira greets + asks small-talk Q
    Turn 2: User replies → Mira asks PHQ-9 Q1
    Turn 3: User replies → score Q1 → Mira asks small-talk Q
    Turn 4: User replies → Mira asks PHQ-9 Q2
    Turn 5: User replies → score Q2 → Mira asks small-talk Q
    ...
    Turn 18: User replies → Mira asks PHQ-9 Q9
    Turn 19: User replies → score Q9 →
              ├─ if Q9 score >= 1 → Crisis protocol (override)
              └─ else → Pivot to therapy ("วันนี้มีอะไรในใจ?")

State machine driven by:
    phq9_step: int  — number of PHQ-9 questions asked so far (0..9)
    intake_pending_score: bool — True if last Mira turn asked a PHQ-9 Q
                                  (so user_text contains the score answer)

Code is deterministic: builds a DIRECTIVE string each turn and lets Mira
(Ultravox) generate the natural-sounding reply around it.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from mira_graph.llm import call_supervisor
from mira_graph.state import MiraState

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_SCORE_PROMPT_PATH = _PROMPT_DIR / "sup_phq9_score.txt"


# ── PHQ-9 Questions in Thai conversational form ──────────────────────────────
# (q_id, conversational_thai, topic_for_scorer)
INTAKE_QUESTIONS: list[tuple[str, str, str]] = [
    ("Q1", "ช่วง 2 อาทิตย์ที่ผ่านมา รู้สึกเบื่อ ไม่อยากทำอะไรบ่อยไหมคะ?",
     "anhedonia"),
    ("Q2", "เศร้า หดหู่ หรือสิ้นหวังบ่อยไหมคะ?",
     "depressed_mood"),
    ("Q3", "การนอนเป็นยังไงบ้างคะ — หลับยาก ตื่นกลางดึก หรือนอนเยอะผิดปกติ?",
     "sleep"),
    ("Q4", "รู้สึกเหนื่อยล้าหรือหมดแรงบ่อยไหมคะ?",
     "fatigue"),
    ("Q5", "เรื่องอาหาร — กินไม่ลง หรือกินเยอะผิดปกติบ้างไหมคะ?",
     "appetite"),
    ("Q6", "รู้สึกแย่กับตัวเอง คิดว่าตัวเองล้มเหลว หรือทำให้คนรอบข้างผิดหวังบ้างไหมคะ?",
     "self_worth"),
    ("Q7", "สมาธิเวลาอ่านหนังสือ ดูทีวี หรือทำงาน เป็นยังไงบ้างคะ?",
     "concentration"),
    ("Q8", "เคลื่อนไหวหรือพูดช้าลงจนคนรอบข้างสังเกต หรือกระสับกระส่ายมากผิดปกติบ้างไหมคะ?",
     "psychomotor"),
    ("Q9", "มีอีกข้อนึงที่อยากถามอ่อนโยนค่ะ ในช่วงที่ผ่านมา เคยมีความคิดอยากทำร้ายตัวเอง หรือไม่อยากอยู่บ้างไหมคะ?",
     "self_harm"),
]


# Small-talk topics rotated between PHQ-9 questions
_SMALLTALK_TOPICS = [
    ("place",   "ถามว่ามาจากจังหวัดไหน หรือ ที่นั่นช่วงนี้อากาศเป็นยังไง"),
    ("food",    "ถามว่าวันนี้กินอะไรมาแล้ว หรือ ของโปรดเป็นอะไร"),
    ("work",    "ถามว่าทำงาน/เรียนสายอะไร"),
    ("hobby",   "ถามงานอดิเรกตอนว่าง"),
    ("media",   "ถามว่าฟังเพลงอะไรอยู่ หรือ ดูซีรีส์อะไรอยู่"),
    ("weather", "ถามว่าอากาศวันนี้เป็นยังไง / ช่วงนี้ฝนตกเยอะไหม"),
    ("weekend", "ถามว่าวันหยุดที่ผ่านมาทำอะไรบ้าง"),
]


# ── Hardcoded Thai response templates ──────────────────────────────────────────
# Bypassing Ultravox for intake — LoRA-baked behavior ignores prompt directives,
# so we generate Mira's response in code with predictable structure.

_INTAKE_ACK_PHRASES = [
    "อืมค่ะ", "เข้าใจค่ะ", "อ๋อ", "ดีค่ะ", "อืม",
    "โอเคค่ะ", "อืมม", "เข้าใจแล้ว", "อ่าค่ะ", "แอ้ก",
]

_SMALLTALK_RESPONSES = [
    "วันนี้กินอะไรมาแล้วคะ?",
    "ช่วงนี้ทำงานหรือเรียนอะไรอยู่คะ?",
    "งานอดิเรกตอนว่างชอบทำอะไรคะ?",
    "ฟังเพลงอะไรอยู่ช่วงนี้คะ?",
    "ที่นั่นอากาศเป็นยังไงคะ?",
    "วันหยุดที่ผ่านมาทำอะไรบ้างคะ?",
    "มีคนรอบข้างให้คุยด้วยบ้างไหมคะ?",
    "วันนี้ตื่นมารู้สึกยังไงบ้างคะ?",
]

_PIVOT_TO_THERAPY = "ขอบคุณที่เล่าให้พี่ฟังนะคะ ตอนนี้มีอะไรอยู่ในใจอยากเล่าไหมคะ?"


def build_intake_response(state: MiraState) -> tuple[str, dict]:
    """Generate Mira's intake response deterministically (no LLM call).

    Returns (response_text, state_updates).
    """
    step = state.get("phq9_step") or 0
    pending = bool(state.get("intake_pending_score"))
    smalltalk_count = state.get("intake_smalltalk_count") or 0
    ack = _INTAKE_ACK_PHRASES[(step + smalltalk_count) % len(_INTAKE_ACK_PHRASES)]

    # User just answered a PHQ-9 Q → either pivot (after Q9) or smalltalk
    if pending and step >= 1:
        if step >= 9:
            # Pivot to therapy
            return (
                _PIVOT_TO_THERAPY,
                {
                    "intake_pending_score": False,
                    "_intake_complete": True,
                },
            )
        # Smalltalk between PHQ-9 Qs
        smalltalk_q = _SMALLTALK_RESPONSES[smalltalk_count % len(_SMALLTALK_RESPONSES)]
        return (
            f"{ack} {smalltalk_q}",
            {
                "intake_pending_score": False,
                "intake_smalltalk_count": smalltalk_count + 1,
            },
        )

    # User answered smalltalk (or initial greeting) → ask next PHQ-9 Q
    next_step = step + 1
    if next_step > 9:
        return (
            _PIVOT_TO_THERAPY,
            {"_intake_complete": True},
        )

    q_id, q_text, _ = INTAKE_QUESTIONS[next_step - 1]
    return (
        f"{ack} {q_text}",
        {
            "phq9_step": next_step,
            "intake_pending_score": True,
        },
    )


# ── Score extraction (Cerebras LLM call) ──────────────────────────────────────

_SCORE_RE = re.compile(r"score\s*:\s*([0-3])", re.IGNORECASE)


async def score_phq9_answer(user_text: str, q_topic: str) -> tuple[int, str]:
    """Cerebras supervisor → score 0-3 + reasoning.

    Falls back to score 1 (mild) if LLM fails or output unparseable —
    conservative: don't miss a real symptom.
    """
    try:
        prompt = _SCORE_PROMPT_PATH.read_text(encoding="utf-8")
        raw = await call_supervisor(prompt, user_text, f"Topic: {q_topic}", max_tokens=80)
    except Exception as e:
        print(f"  [INTAKE:score] error — {e}, defaulting to 1")
        return 1, "score-extractor failed"

    print(f"  [INTAKE:score RAW]\n{raw}\n  [/INTAKE:score RAW]")

    m = _SCORE_RE.search(raw)
    if not m:
        return 1, "unparseable output, default 1"
    score = int(m.group(1))
    reasoning_match = re.search(r"reasoning\s*:\s*(.+)", raw, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip()[:80] if reasoning_match else ""
    return score, reasoning


# ── Directive builder (deterministic state machine) ───────────────────────────

def _format_progress(scores: dict, step: int) -> str:
    if not scores:
        return f"ยังไม่ได้ถาม PHQ-9 (เริ่มต้น)"
    return f"ตอบ PHQ-9 แล้ว {len(scores)}/9 ข้อ (step={step})"


def _smalltalk_directive(turn_count: int) -> str:
    """Pick a small-talk topic — rotate so we don't repeat."""
    topic, hint = _SMALLTALK_TOPICS[turn_count % len(_SMALLTALK_TOPICS)]
    return f"Acknowledge สั้นๆ + small talk เรื่อง: {hint} (1 ประโยค)"


def _phq9_directive(q_text: str) -> str:
    return (
        f"Acknowledge สั้นๆ + ถามคำถามนี้ตรงๆ (จะ paraphrase เล็กน้อยให้ smooth ได้): "
        f"\"{q_text}\""
    )


def build_intake_directive(state: MiraState) -> tuple[str, dict]:
    """Determine what Mira should do this turn. Returns (directive_text, state_updates).

    State machine:
        step=0, pending=False: greet + smalltalk (turn 1)
        step=N (1..9), pending=True: user just answered Q{N} — score + smalltalk
        step=N, pending=False: user just answered smalltalk — ask Q{N+1}
        step=9, after scoring: pivot to therapy
    """
    step = state.get("phq9_step") or 0
    pending = bool(state.get("intake_pending_score"))
    smalltalk_count = state.get("intake_smalltalk_count") or 0

    # First turn ever — greet + smalltalk Q
    if step == 0 and not state.get("intake_started"):
        return (
            "ทักทาย user + ถามว่ามาจากจังหวัดไหนคะ (small talk แรก)",
            {
                "intake_started": True,
                "intake_pending_score": False,
            },
        )

    # User just answered a PHQ-9 Q — score it (handled by caller),
    # next we ask smalltalk (alternate)
    if pending and step >= 1:
        # Pivot moment: just scored Q9
        if step >= 9:
            return (
                "ขอบคุณที่เล่าให้พี่ฟังนะคะ — ถาม transition: "
                "\"ตอนนี้มีอะไรอยู่ในใจอยากเล่าไหมคะ?\"",
                {"_intake_complete": True},
            )
        # Otherwise: smalltalk between Qs
        return (
            _smalltalk_directive(smalltalk_count),
            {
                "intake_pending_score": False,
                "intake_smalltalk_count": smalltalk_count + 1,
            },
        )

    # User just answered smalltalk — ask next PHQ-9 Q
    next_step = step + 1
    if next_step > 9:
        # Shouldn't reach here normally
        return (
            "ขอบคุณที่เล่าให้พี่ฟังนะคะ — ถาม: \"ตอนนี้มีอะไรอยู่ในใจอยากเล่าไหมคะ?\"",
            {"_intake_complete": True},
        )

    q_id, q_text, _ = INTAKE_QUESTIONS[next_step - 1]
    return (
        _phq9_directive(q_text),
        {
            "phq9_step": next_step,
            "intake_pending_score": True,
        },
    )


# ── Score prior PHQ-9 answer (called BEFORE building next directive) ──────────

async def score_prior_answer(state: MiraState, user_text: str) -> dict:
    """If we asked a PHQ-9 Q last turn, score user's reply now.

    Returns state updates including the new score, possibly Q9 crisis flag.
    """
    step = state.get("phq9_step") or 0
    pending = bool(state.get("intake_pending_score"))

    if not pending or step < 1 or step > 9:
        return {}

    q_id, q_text, topic = INTAKE_QUESTIONS[step - 1]
    score, reasoning = await score_phq9_answer(user_text, topic)

    scores = dict(state.get("phq9_scores") or {})
    scores[q_id] = score
    total = sum(scores.values())

    print(f"  [INTAKE] {q_id} ({topic}) → score={score} | {reasoning}")
    print(f"  [INTAKE] running total={total}, scores={scores}")

    updates: dict = {
        "phq9_scores": scores,
        "phq9_total": total,
    }

    # Q9 crisis trigger — caller (mira_agent_v4) handles override
    if step == 9 and score >= 1:
        updates["_intake_q9_crisis"] = True
        updates["phq9_q9_score"] = score

    return updates


# ── Risk level mapping (after Q9) ─────────────────────────────────────────────

def risk_level_label(total: int) -> str:
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately severe"
    return "severe"
