from mira_graph.state import MiraState


def route_by_phase(state: MiraState) -> str:
    if state.get("crisis_detected"):
        # TODO: crisis_handler in Phase 2
        return "checkin"

    return {
        "CHECKIN": "checkin",
        "EXPLORE": "explore",
        "WORK": "work",
        "WRAP": "wrap",
    }.get(state.get("phase", "CHECKIN"), "checkin")
