"""
Phase-aware response composition for Mira.
Combines crisis detection, smart retrieval, and CBT phase guidance.
"""

import json
import re
import logging
import sys
sys.path.insert(0, '/app/rag')

logger = logging.getLogger(__name__)

from crisis_detector import detect_crisis
from smart_retriever import get_smart_retriever
from phase_detector import detect_phase, detect_topic


with open("/app/rag/crisis_responses.json") as f:
    CRISIS_RESPONSES = json.load(f)


PHASE_PROMPTS = {
    'rapport': """
PHASE: Check-in (warm open)
═══════════════════════════════════════
This is the START of the conversation. Your job: make it safe, get a read on how they feel.

DO:
✓ Greet warmly, acknowledge what they bring
✓ VALIDATE the feeling — name it, mirror their words
✓ Get a gentle read on their state
✓ Invite what's on their mind: "วันนี้มีอะไรอยากเล่าให้พี่ฟังไหมคะ"

DON'T:
✗ Don't start CBT work yet — no Socratic questions, no challenging
✗ Don't rush — let them land first

This is the ONE phase where pure validation is the right move.
""",

    'assessment': """
PHASE: Explore (find the automatic thought)
═══════════════════════════════════════
They've shared a feeling. Your job NOW: find the THOUGHT behind it.
This is the pivot — validation is DONE, exploration STARTS.

DO:
✓ Briefly acknowledge (ONE short clause — you already validated in check-in)
✓ Then SURFACE the automatic thought — the specific belief that ran through their mind:
  "ตอนนั้น มีความคิดอะไรแวบเข้ามาในหัวบ้างคะ"
  "ฟังดูเหมือนมีความคิดว่า 'ผมไม่ดีพอ' — ใช่ความคิดนั้นไหมคะ"
✓ ASK ONE open question — every reply in this phase MUST contain a question
✓ Get the concrete situation: what happened, when, with whom

DON'T:
✗ DON'T just validate again — you already did that in check-in.
  Validating here instead of asking = being stuck. It is the #1 mistake.
✗ DON'T pile on more emotion words ("หดหู่และเหนื่อยใจมากแน่ๆ") — that is dwelling, not exploring
✗ DON'T collect feelings endlessly — you are hunting for the THOUGHT

A reply in this phase WITHOUT a question is a failed reply.
""",

    'intervention': """
PHASE: Work (evaluate the thought + reframe)
═══════════════════════════════════════
The automatic thought is on the table. Your job: evaluate it WITH them, guide a reframe.
This is where change happens. Do NOT validate the thought — question it.

DO:
✓ Take the specific automatic thought and gently question it — guided discovery:
  "ความคิดนั้น... มันจริงทั้งหมดไหมคะ หรือมีหลักฐานอีกมุม"
  "ถ้าเพื่อนสนิทคิดแบบนี้กับตัวเอง น้องจะบอกเขาว่ายังไง"
  "มีครั้งไหนบ้างที่มันไม่เป็นแบบนั้น"
✓ ONE Socratic question at a time — never a checklist
✓ Guide them to reach a more balanced, kinder thought THEMSELVES
✓ When they show movement, name it: "เมื่อกี้น้องมองมันใหม่ได้เองเลยนะ"

DON'T:
✗ DON'T validate the distorted thought ("เข้าใจว่ารู้สึกแบบนั้น" then move on) — question it
✗ DON'T lecture the reframe — guide them to it with questions
✗ DON'T rush — this is the core work

A reply in this phase WITHOUT a Socratic question is a failed reply.
""",

    'skills': """
PHASE: Wrap (consolidate + one small step)
═══════════════════════════════════════
The work is done. Your job: lock in the shift, give ONE concrete step, close warmly.

DO:
✓ Briefly name the shift they made: "เมื่อกี้น้องมองมันใหม่ได้เองเลยนะ"
✓ Offer ONE small, concrete, specific step tied to their actual situation
✓ Close warmly — leave the door open

DON'T:
✗ DON'T open new threads or start exploring again
✗ DON'T give generic advice ("ดูแลตัวเองนะ") — the step must be specific to their situation
✗ DON'T list multiple steps — ONE
""",
}


def get_mira_response(
    user_text: str,
    history: list = None,
    phase: str | None = None,
):
    """
    Phase-aware CBT response generation.

    Args:
        user_text: User's input text
        history: Conversation history (list of dicts with 'role' and 'content')
        phase: Phase from heuristic_judge graph state (source of truth).
               If None, falls back to stateless detect_phase() (safety net).

    Returns:
        dict with:
            source: 'crisis' | 'rag'
            phase: rapport/assessment/intervention/skills/crisis
            topic: detected topic or None
            severity: crisis severity (for backward compat: crisis_severity)
            matched_keywords: crisis keywords (for backward compat)
            response: For crisis only - direct response text
            reasoning_prompt: For rag - prompt to send LLM
            references_used: List of retrieved refs
    """
    history = history or []

    crisis = detect_crisis(user_text)
    if crisis['is_crisis']:
        severity = crisis['severity']
        keywords = crisis['matched_keywords']

        crisis_response = None
        for template in CRISIS_RESPONSES.get('crisis_responses', []):
            if template.get('trigger_severity') == severity:
                if any(kw in template.get('match_keywords', []) for kw in keywords):
                    crisis_response = template.get('response', '')
                    break

        if not crisis_response:
            crisis_response = CRISIS_RESPONSES.get('default_crisis_response', {}).get('response',
                '[EMOTION: concerned] Mira ห่วงคุณนะ ลองโทร 1323 สายด่วนสุขภาพจิตฟรี 24 ชม.')

        return {
            'source': 'crisis',
            'phase': 'crisis',
            'severity': severity,
            'crisis_severity': severity,
            'matched_keywords': keywords,
            'response': crisis_response,
            'reasoning_prompt': None,
            'references_used': []
        }

    # === PHASE RESOLUTION ===
    if phase is not None:
        logger.info("[PHASE] resolved=%s source=graph", phase)
    else:
        phase = detect_phase(history, user_text)
        logger.info("[PHASE] resolved=%s source=fallback_detect", phase)

    topic = detect_topic(user_text)

    retriever = get_smart_retriever()
    refs = retriever.retrieve(
        query=user_text,
        phase=phase,
        topic_hint=topic,
        k=3
    )

    reasoning_prompt = build_phase_aware_prompt(
        user_text=user_text,
        refs=refs,
        phase=phase,
        topic=topic,
        history=history
    )

    # Also build the cache-friendly split: stable system + dynamic user prefix.
    # Callers that want prefix-cache hits should use these two fields and
    # construct messages = [{role: system, content: stable}, *history,
    # {role: user, content: dynamic_prefix + user_text}].
    stable_system = build_stable_system_prompt(phase)
    dynamic_prefix = build_dynamic_user_prefix(refs=refs, topic=topic)

    return {
        'source': 'rag',
        'phase': phase,
        'topic': topic,
        'reasoning_prompt': reasoning_prompt,
        'stable_system_prompt': stable_system,
        'dynamic_user_prefix': dynamic_prefix,
        'references_used': refs,
    }


def build_phase_aware_prompt(user_text, refs, phase, topic, history):
    """Build comprehensive phase-aware reasoning prompt."""

    phase_guidance = PHASE_PROMPTS.get(phase, PHASE_PROMPTS['rapport'])

    refs_text = ""
    for i, ref in enumerate(refs, 1):
        user_excerpt = ref['user_input'][:200]
        resp_excerpt = ref['counselor_response'][:300]
        refs_text += f"""
[ตัวอย่างที่ {i}] (เทคนิค: {ref['technique']})
ผู้ใช้พูดว่า: {user_excerpt}
นักบำบัด: {resp_excerpt}
"""

    history_text = ""
    if history:
        recent = history[-4:]
        for msg in recent:
            role = "User" if msg.get('role') == 'user' else "Mira"
            content = msg.get('content', '')[:150]
            history_text += f"{role}: {content}\n"

    prompt = f"""You are Mira, a Thai mental health companion using CBT principles.

═══════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════
Phase: {phase}
Topic: {topic or 'general'}

{phase_guidance}

═══════════════════════════════════════
RECENT CONVERSATION
═══════════════════════════════════════
{history_text if history_text else '(beginning of conversation)'}

═══════════════════════════════════════
USER'S CURRENT MESSAGE
═══════════════════════════════════════
"{user_text}"

═══════════════════════════════════════
THERAPIST REFERENCES (from real conversations)
═══════════════════════════════════════
{refs_text}

═══════════════════════════════════════
REASONING PROTOCOL
═══════════════════════════════════════

<analysis>
1. Emotion in user's message:
2. Thought (if expressed):
3. Behavior (if mentioned):
4. Phase-appropriate move:
</analysis>

<reference_review>
Best matching reference (1, 2, or 3):
CBT technique used by counselor:
Why this fits user's situation:
</reference_review>

<intervention_plan>
Based on phase = {phase}:
- Validation needed: yes
- Question to ask: (or "none" if rapport phase)
- Technique to apply: (or "none" if rapport phase)
</intervention_plan>

<draft>
Generate Mira's response that:
1. Speaks AS Mira (counselor perspective, NOT as user)
2. Mirrors user's specific words
3. Follows phase rules above
4. Uses natural Thai
5. Includes [EMOTION: tag] at end
</draft>

<verify>
Final checks before output:
✓ Forbidden opener avoided: NO "พี่เข้าใจ", NO "พี่มิร่าเข้าใจ", NO "ฟังดูเหมือน"
✓ "ใช่ไหมคะ" ending avoided
✓ First-person impersonation avoided (don't say "ผม/ฉัน รู้สึก..." as if user)
✓ Speaks AS Mira (use "Mira" or addresses user as "คุณ"/"น้อง")
✓ Phase rule respected
✓ CBT technique present (if not rapport phase)
</verify>

OUTPUT (Thai response + [EMOTION: tag]):"""

    return prompt


# ── Cache-friendly split: stable system prompt + dynamic user prefix ──────────
# Goal: maximize vLLM prefix-cache hit rate by keeping per-turn changes out of
# the system message. The system message becomes a pure function of `phase` —
# it's identical across every turn that stays in the same phase, so vLLM
# caches the whole 2000-ish token prefix and only has to prefill the short
# dynamic user prefix + user message on each turn.

STABLE_SYSTEM_HEADER = """You are Mira, a Thai mental health companion using CBT principles.

═══════════════════════════════════════
REASONING PROTOCOL (applies every turn)
═══════════════════════════════════════
Before you speak, silently run through:

<analysis>
1. Emotion in user's message
2. Thought (if expressed)
3. Behavior (if mentioned)
4. Phase-appropriate move
</analysis>

<reference_review>
Best matching reference (1, 2, or 3)
CBT technique used by counselor
Why this fits user's situation
</reference_review>

<intervention_plan>
- Validation needed: (phase-dependent — check phase rules above)
- Question to ask: (or "none" if rapport/WRAP phase)
- Technique to apply: (or "none" if rapport/EXPLORE phase)
</intervention_plan>

<draft>
Compose Mira's response that:
1. Speaks AS Mira (counselor perspective, NOT as user)
2. Mirrors user's specific words
3. Follows the phase rules below
4. Uses natural Thai
5. Includes [EMOTION: tag] at the end
</draft>

<verify>
Final checks before output:
✓ No forbidden openers: NO "พี่เข้าใจ", NO "พี่มิร่าเข้าใจ", NO "ฟังดูเหมือน"
✓ No "ใช่ไหมคะ" endings
✓ No first-person impersonation (don't say "ผม/ฉัน รู้สึก..." as if user)
✓ Speaks AS Mira (use "Mira" or address user as "คุณ"/"น้อง")
✓ Phase rule respected
✓ CBT technique present (if not rapport phase)
</verify>

OUTPUT: Thai response followed by [EMOTION: tag] on a new line.
"""


def build_stable_system_prompt(phase: str) -> str:
    """
    Build the STABLE portion of the prompt — pure function of phase.

    Identical across every turn that stays in the same phase, so vLLM's
    prefix cache can reuse the whole thing (~2000 tokens) per turn.

    Do NOT include: history, refs, user_text, timestamps, turn counters,
    or any state that changes per turn.
    """
    phase_guidance = PHASE_PROMPTS.get(phase, PHASE_PROMPTS['rapport'])
    return (
        STABLE_SYSTEM_HEADER
        + "\n═══════════════════════════════════════\n"
        + f"CURRENT PHASE: {phase}\n"
        + "═══════════════════════════════════════\n"
        + phase_guidance
    )


def build_dynamic_user_prefix(refs: list, topic: str | None) -> str:
    """
    Build the DYNAMIC portion that changes per turn — goes into the user
    message, NOT the system prompt.

    Contains the retrieved RAG references and detected topic. These change
    every turn and must stay out of the cached system prefix.

    Returns a short prefix that the caller prepends to the user's message.
    """
    parts = []

    if topic:
        parts.append(f"[หัวข้อ: {topic}]")

    _thai_char_re = re.compile(r"[\u0e00-\u0e7f]")

    def _is_thai_excerpt(text: str) -> bool:
        if not text:
            return False
        return (len(_thai_char_re.findall(text)) / max(len(text), 1)) >= 0.10

    if refs:
        refs_text = ""
        for i, ref in enumerate(refs, 1):
            user_raw = ref['user_input'][:200]
            resp_raw = ref['counselor_response'][:300]
            if _is_thai_excerpt(user_raw + resp_raw):
                refs_text += (
                    f"\n[ตัวอย่างที่ {i}] (เทคนิค: {ref['technique']})\n"
                    f"ผู้ใช้พูดว่า: {user_raw}\n"
                    f"นักบำบัด: {resp_raw}\n"
                )
            else:
                refs_text += f"\n[เทคนิคที่แนะนำ {i}]: {ref['technique']}\n"
        parts.append(
            "═══════════════════════════════════════\n"
            "เทคนิคที่แนะนำสำหรับการสนทนานี้\n"
            "═══════════════════════════════════════"
            + refs_text
        )

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n\n───\nข้อความของผู้ใช้:\n"