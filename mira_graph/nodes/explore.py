from mira_graph.nodes._phase_base import run_phase
from mira_graph.state import MiraState


async def explore_node(state: MiraState) -> dict:
    return await run_phase("EXPLORE", state, max_tokens=1000)
