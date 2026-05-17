"""Mira LangGraph — AutoCBT-inspired Multi-agent Architecture

Inspired by: AutoCBT (Xu et al., 2025, arXiv:2501.09426)

Flow per turn:
  User → crisis_check → 5 supervisors + at_extract (parallel) → aggregate → counsellor → output_filter → judge → END

All agents (supervisors + counsellor) use the same Mira endpoint:
  Primary: Ultravox CBT LoRA (ULTRAVOX_URL)
  Fallback: Cerebras GPT-OSS 120B
"""
from langgraph.graph import END, START, StateGraph

from mira_graph.nodes.aggregate import aggregate_node
from mira_graph.nodes.at_extract import at_supervisor
from mira_graph.nodes.counsellor import counsellor_node
from mira_graph.nodes.crisis import crisis_check_node
from mira_graph.nodes.judge import judge_node
from mira_graph.nodes.output_filter import output_filter_node
from mira_graph.nodes.supervisors import (
    belief_supervisor,
    empathy_supervisor,
    encouragement_supervisor,
    reflection_supervisor,
    strategy_supervisor,
)
from mira_graph.state import MiraState

_SUPERVISORS = [
    "sup_empathy", "sup_belief", "sup_reflection", "sup_strategy", "sup_encourage",
    "sup_at",
]


def build_mira_graph():
    g = StateGraph(MiraState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    g.add_node("crisis_check", crisis_check_node)

    g.add_node("sup_empathy", empathy_supervisor)
    g.add_node("sup_belief", belief_supervisor)
    g.add_node("sup_reflection", reflection_supervisor)
    g.add_node("sup_strategy", strategy_supervisor)
    g.add_node("sup_encourage", encouragement_supervisor)
    g.add_node("sup_at", at_supervisor)

    g.add_node("aggregate", aggregate_node)
    g.add_node("counsellor", counsellor_node)
    g.add_node("output_filter", output_filter_node)
    g.add_node("judge", judge_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    g.add_edge(START, "crisis_check")

    # Fan-out: crisis → supervisors (incl. at_extract) in parallel
    for sup in _SUPERVISORS:
        g.add_edge("crisis_check", sup)

    # Fan-in: all supervisors → aggregate
    for sup in _SUPERVISORS:
        g.add_edge(sup, "aggregate")

    g.add_edge("aggregate", "counsellor")
    g.add_edge("counsellor", "output_filter")
    g.add_edge("output_filter", "judge")
    g.add_edge("judge", END)

    return g.compile()


mira_graph = build_mira_graph()
