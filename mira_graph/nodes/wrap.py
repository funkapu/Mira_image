from mira_graph.nodes._phase_base import run_phase
from mira_graph.state import MiraState


async def wrap_node(state: MiraState) -> dict:
    return await run_phase("WRAP", state, history_depth=8, max_tokens=800)
