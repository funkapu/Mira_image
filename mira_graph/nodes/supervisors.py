"""5 Supervisor Agents — run in parallel, Cerebras 8B, English structured output"""
from pathlib import Path

from mira_graph.llm import call_supervisor
from mira_graph.state import MiraState

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _build_context(state: MiraState) -> str:
    lines = [f"Phase: {state.get('phase', '?')}"]

    name = state.get("user_name")
    lines.append(f"Name: {name if name else 'UNKNOWN'}")

    mood = state.get("mood_score")
    lines.append(f"Mood: {str(mood) + '/10' if mood is not None else 'UNKNOWN'}")

    concern = state.get("main_concern")
    lines.append(f"Concern: {concern if concern else 'UNKNOWN'}")

    at = state.get("automatic_thought")
    dist = state.get("distortion")
    lines.append(f"Automatic thought: {at if at else 'UNKNOWN'}")
    lines.append(f"Distortion: {dist if dist else 'UNKNOWN'}")

    msgs = state.get("messages", [])[-8:]

    mira_msgs = []
    user_msgs = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = (m.get("content") or "")[:80]
        else:
            role = "assistant" if getattr(m, "type", "") == "ai" else "user"
            content = (getattr(m, "content", "") or "")[:80]
        if role == "assistant":
            mira_msgs.append(content)
        elif role == "user":
            user_msgs.append(content)

    if mira_msgs:
        lines.append("\nMira already said:")
        for m in mira_msgs:
            lines.append(f"  - {m}")

    if user_msgs:
        lines.append("\nUser already said:")
        for m in user_msgs:
            lines.append(f"  - {m}")

    return "\n".join(lines)


async def empathy_supervisor(state: MiraState) -> dict:
    prompt = (_PROMPT_DIR / "sup_empathy.txt").read_text(encoding="utf-8")
    advice = await call_supervisor(prompt, state.get("user_text") or "", _build_context(state))
    print(f"  [SUP:Empathy] {advice[:80]}")
    return {"supervisor_empathy": advice}


async def belief_supervisor(state: MiraState) -> dict:
    prompt = (_PROMPT_DIR / "sup_belief.txt").read_text(encoding="utf-8")
    advice = await call_supervisor(prompt, state.get("user_text") or "", _build_context(state))
    print(f"  [SUP:Belief] {advice[:80]}")
    return {"supervisor_belief": advice}


async def reflection_supervisor(state: MiraState) -> dict:
    prompt = (_PROMPT_DIR / "sup_reflection.txt").read_text(encoding="utf-8")
    advice = await call_supervisor(prompt, state.get("user_text") or "", _build_context(state))
    print(f"  [SUP:Reflect] {advice[:80]}")
    return {"supervisor_reflection": advice}


async def strategy_supervisor(state: MiraState) -> dict:
    prompt = (_PROMPT_DIR / "sup_strategy.txt").read_text(encoding="utf-8")
    advice = await call_supervisor(prompt, state.get("user_text") or "", _build_context(state))
    print(f"  [SUP:Strategy] {advice[:80]}")
    return {"supervisor_strategy": advice}


async def encouragement_supervisor(state: MiraState) -> dict:
    prompt = (_PROMPT_DIR / "sup_encourage.txt").read_text(encoding="utf-8")
    advice = await call_supervisor(prompt, state.get("user_text") or "", _build_context(state))
    print(f"  [SUP:Encourage] {advice[:80]}")
    return {"supervisor_encouragement": advice}
