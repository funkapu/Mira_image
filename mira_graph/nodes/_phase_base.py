from __future__ import annotations

import os
from pathlib import Path

from mira_graph.llm import apply_voice_mode, call_with_tools
from mira_graph.state import MiraState
from mira_graph.tools import TOOLS, execute_tool

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "mira_system_v4_thai.txt"
_PROMPT_PATH_NOSTAGE = Path(__file__).parent.parent / "prompts" / "mira_system_v4_thai_nostage.txt"
_PHASE_MARKER = "---PHASE "

_DEFAULT_WINDOW_TURNS = int(os.environ.get("MIRA_HISTORY_WINDOW_TURNS", "4"))
STAGE_DISABLED = os.environ.get("MIRA_STAGE_DISABLED", "0") == "1"


def _approx_tokens(text: str) -> int:
    """Rough token count — Thai ~3 chars/token, English ~4. Use 3 for safety."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def _msg_role(msg) -> str:
    """Normalize role across LangGraph BaseMessage (.type) and dict (.role)."""
    if isinstance(msg, dict):
        return msg.get("role", "") or ""
    t = getattr(msg, "type", "") or ""
    if t == "human":
        return "user"
    if t == "ai":
        return "assistant"
    return t


def _sliding_window(history: list, max_turns: int = _DEFAULT_WINDOW_TURNS) -> list:
    """Keep the last `max_turns` user turns and everything that followed them.

    Walks newest → oldest counting user messages; cuts at the Nth-from-last
    user msg. Preserves trailing assistant replies and a possibly-unpaired
    final user msg. Returns chronological order.
    """
    if not history or max_turns <= 0:
        return list(history or [])

    user_seen = 0
    cut_index = 0
    found = False
    for i in range(len(history) - 1, -1, -1):
        if _msg_role(history[i]) == "user":
            user_seen += 1
            if user_seen >= max_turns:
                cut_index = i
                found = True
                break
    if not found:
        cut_index = 0
    return history[cut_index:]


def _trim_history_to_budget(history: list, budget_tokens: int = 800) -> list:
    """Keep most-recent messages that fit within a token budget.

    Each message costs approx tokens(content) + 10 (for role tag + framing).
    Walks from newest to oldest and stops when the next message would
    exceed the budget. Returns messages in original (chronological) order.
    """
    kept: list = []
    used = 0
    for msg in reversed(history or []):
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        else:
            content = getattr(msg, "content", "") or ""
        cost = _approx_tokens(content) + 10
        if used + cost > budget_tokens:
            break
        kept.insert(0, msg)
        used += cost
    return kept


def _load_prompt(phase: str, state: MiraState) -> str:
    # Stage-disabled: use the trimmed prompt and skip per-phase injection.
    if STAGE_DISABLED:
        raw = _PROMPT_PATH_NOSTAGE.read_text(encoding="utf-8")
        shared_template = raw
        phase_content = ""
    else:
        raw = _PROMPT_PATH.read_text(encoding="utf-8")
        parts = raw.split(_PHASE_MARKER)
        shared_template = parts[0]
        phase_content = ""
        for part in parts[1:]:
            first_line, _, rest = part.partition("\n")
            if first_line.rstrip("-").strip() == phase:
                phase_content = rest.strip()
                break

    def _val(key: str, default: str = "—") -> str:
        v = state.get(key)
        return str(v) if v is not None else default

    prompt = shared_template.replace("{phase_instructions}", phase_content)
    prompt = prompt.replace("{user_name}", _val("user_name"))
    prompt = prompt.replace("{mood_score}", _val("mood_score"))
    prompt = prompt.replace("{main_concern}", _val("main_concern"))
    prompt = prompt.replace("{automatic_thought}", _val("automatic_thought"))
    prompt = prompt.replace("{distortion}", _val("distortion"))

    return prompt


async def run_phase(
    phase: str,
    state: MiraState,
    *,
    history_depth: int = 8,
    max_tokens: int = 1000,
) -> dict:
    system_prompt = _load_prompt(phase, state)
    system_prompt = apply_voice_mode(system_prompt, state)

    history = _trim_history_to_budget(
        _sliding_window(state.get("messages", [])[-history_depth:]),
        budget_tokens=800,
    )

    # Truncate single-utterance user_text (rare STT bursts) to avoid overflow.
    raw_user = state.get("user_text") or ""
    user_text = raw_user[-500:] if len(raw_user) > 500 else raw_user

    response, latency, tool_calls = await call_with_tools(
        system_prompt=system_prompt,
        user_text=user_text,
        history=history,
        tools=TOOLS,
        max_tokens=max_tokens,
    )

    print(f"  [{phase}] {latency:.0f}ms, tools={len(tool_calls)}")

    state_updates: dict = {}
    for tc in tool_calls:
        state_updates.update(execute_tool(tc["name"], tc["args"], state))

    return {
        "mira_response": response,
        "messages": [{"role": "assistant", "content": response}],
        **state_updates,
    }
