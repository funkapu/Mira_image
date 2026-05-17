"""Voice Agent v5 — Stage Manager: explicit mode dispatcher.

Centralises the rules that pick which response handler runs each turn.
Before v5, mode logic was scattered across mira_agent_v4.py:464-565 and the
nodes/intake.py / nodes/judge.py modules; the agent decided "intake vs
therapy" by checking session_type inline. This module makes the decision a
single pure function so the dispatch is testable and so latency telemetry
can log a `mode=...` tag.

Truth table:

    _crisis_active=True      → CRISIS    (hardcoded TTS, bypass LLM)
    session_type=="intake"   → SCREENING (deterministic intake responses)
    session_type=="therapy"  → THERAPY   (Ultravox text-only cascading)

CRISIS takes priority over everything else. Any other unexpected combination
falls back to THERAPY (the safest live conversational path).
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping


class Mode(str, Enum):
    THERAPY = "therapy"
    SCREENING = "screening"
    CRISIS = "crisis"


def resolve_mode(state: Mapping, crisis_active: bool) -> Mode:
    """Pick the response mode for the current turn.

    Args:
        state: MiraState dict (see mira_graph/state.py). Reads `session_type`
            and `phase`.
        crisis_active: agent's sticky `_crisis_active` flag — once set in a
            session it overrides everything until the session ends.
    """
    if crisis_active:
        return Mode.CRISIS

    phase = state.get("phase")
    if phase == "CRISIS":
        return Mode.CRISIS

    session_type = state.get("session_type")
    # NSC demo: PHQ-9 screening DISABLED unconditionally.
    # Crisis detection still runs via _detect_crisis (text regex on user text).
    # if session_type == "intake":
    #     return Mode.SCREENING

    # session_type == "therapy" or any unexpected value → therapy is the
    # default live path. Phase (CHECKIN..WRAP / END) is handled inside the
    # therapy handler via _load_prompt(phase, ...).
    return Mode.THERAPY
