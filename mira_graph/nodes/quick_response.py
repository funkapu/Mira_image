"""Phase 3.1: quick-response node — runs in parallel with 5 supervisors.

Generates a short empathic acknowledgment via Cerebras 8B (~300ms) so the
agent can fire session.say() immediately while the full graph finishes.
"""
from mira_graph.llm import call_slm_quick
from mira_graph.state import MiraState


async def quick_response_node(state: MiraState) -> dict:
    text = state.get("user_text") or ""
    phase = state.get("phase", "CHECKIN")

    # Skip if input is too short or already at END
    if len(text.strip()) < 2 or phase == "END":
        return {"quick_response": None}

    quick = await call_slm_quick(text, phase)
    return {"quick_response": quick or None}
