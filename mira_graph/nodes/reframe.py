from mira_graph.nodes._phase_base import run_phase
from mira_graph.state import MiraState


async def work_node(state: MiraState) -> dict:
    return await run_phase("WORK", state, max_tokens=1000)
