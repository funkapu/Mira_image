"""Phase transition judge — within-session CBT structure.

Priority order:
  1. Crisis → hold phase (untouched)
  2. LLM signal via [NEXT_STAGE: X]
  3. Readiness heuristic (goal-based per phase)
  4. Turn ceiling (safety net)
  5. Stay
"""
from __future__ import annotations

import os

from mira_graph.heuristic_judge import heuristic_judge
from mira_graph.state import MiraState

STAGE_DISABLED = os.environ.get("MIRA_STAGE_DISABLED", "0") == "1"

NEXT_PHASE: dict[str, str] = {
    "CHECKIN": "EXPLORE",
    "EXPLORE": "WORK",
    "WORK": "WRAP",
    "WRAP": "END",
}

_PHASE_ORDER: dict[str, int] = {
    "CHECKIN": 1,
    "EXPLORE": 2,
    "WORK": 3,
    "WRAP": 4,
    "END": 5,
}

_VALID_LLM_TARGETS = {"EXPLORE", "WORK", "WRAP", "END"}

MAX_TURNS: dict[str, int] = {
    "CHECKIN": 3,
    "EXPLORE": 6,
    "WORK": 6,
    "WRAP": 4,
}


def _is_legal_forward(current: str, target: str) -> bool:
    """LLM may only suggest forward jumps (or stay). No backward transitions."""
    return _PHASE_ORDER.get(target, 0) > _PHASE_ORDER.get(current, 0)


async def judge_node(state: MiraState) -> dict:
    current: str = state.get("phase", "CHECKIN")

    if STAGE_DISABLED:
        if state.get("crisis_detected"):
            return {
                "phase_turn_count": state.get("phase_turn_count", 0) + 1,
                "total_turns": state.get("total_turns", 0) + 1,
            }
        return {
            "phase_turn_count": state.get("phase_turn_count", 0) + 1,
            "total_turns": state.get("total_turns", 0) + 1,
            "should_transition": False,
            "next_phase": None,
            "_transition_requested": False,
        }

    if current == "END":
        return {}

    turn_count: int = state.get("phase_turn_count", 0) + 1
    total_turns: int = state.get("total_turns", 0) + 1

    # Crisis — hold phase
    if state.get("crisis_detected"):
        print(f"  [JUDGE 🚨] Crisis active, holding {current}")
        return {"phase_turn_count": turn_count, "total_turns": total_turns}

    next_phase: str | None = None

    # PRIORITY 0: LLM signal — Mira explicitly suggests next phase
    mira_suggest = state.get("_llm_phase_suggestion")
    if mira_suggest and mira_suggest != "STAY":
        if mira_suggest in _VALID_LLM_TARGETS and _is_legal_forward(current, mira_suggest):
            next_phase = mira_suggest
            print(f"  [JUDGE 🤖] LLM suggested: {current} → {next_phase}")
        else:
            print(f"  [JUDGE 🤖] LLM suggestion {mira_suggest!r} ignored (illegal from {current})")

    if next_phase is None:
        # PRIORITY 1: Readiness heuristic
        heuristic_state = dict(state, phase_turn_count=turn_count)
        if heuristic_judge(heuristic_state).get("transition") and current != "WRAP":
            next_phase = NEXT_PHASE.get(current)
            if next_phase:
                print(f"  [JUDGE 🔧] Heuristic: {current} → {next_phase}")

    if next_phase is None:
        # PRIORITY 2: Turn ceiling (safety net)
        if turn_count >= MAX_TURNS.get(current, 6):
            next_phase = NEXT_PHASE.get(current)
            if next_phase:
                print(f"  [JUDGE ⏰] Max turns {turn_count}: {current} → {next_phase}")

    if next_phase:
        return {
            "phase": next_phase,
            "phase_turn_count": 0,
            "total_turns": total_turns,
            "should_transition": True,
            "next_phase": next_phase,
            "_transition_requested": False,
        }

    print(f"  [JUDGE STAY] {current} {turn_count}/{MAX_TURNS.get(current, 6)}")
    return {
        "phase_turn_count": turn_count,
        "total_turns": total_turns,
        "should_transition": False,
        "next_phase": None,
        "_transition_requested": False,
    }
