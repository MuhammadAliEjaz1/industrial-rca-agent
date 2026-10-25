"""
Phase 5: Wire the nodes into the actual Phase 0 workflow.

gather_context -> run_anomaly_model -> gather_evidence -> generate_hypotheses
    -> investigate_hypotheses --[insufficient, retries<limit]--> gather_more_evidence --> (loop back)
    -> investigate_hypotheses --[sufficient]--> determine_root_cause -> generate_report -> END

No checkpointer: each investigation is a single self-contained run (graph.invoke once),
not a multi-turn conversation, so there's no conversation state to persist across calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from graph.state import InvestigationState
from graph.nodes import (
    gather_context, run_anomaly_model, gather_evidence,
    generate_hypotheses, investigate_hypotheses, route_after_investigation,
    gather_more_evidence, determine_root_cause, generate_report,
)


def build_graph():
    builder = StateGraph(InvestigationState)

    builder.add_node("gather_context", gather_context)
    builder.add_node("run_anomaly_model", run_anomaly_model)
    builder.add_node("gather_evidence", gather_evidence)
    builder.add_node("generate_hypotheses", generate_hypotheses)
    builder.add_node("investigate_hypotheses", investigate_hypotheses)
    builder.add_node("gather_more_evidence", gather_more_evidence)
    builder.add_node("determine_root_cause", determine_root_cause)
    builder.add_node("generate_report", generate_report)

    builder.set_entry_point("gather_context")
    builder.add_edge("gather_context", "run_anomaly_model")
    builder.add_edge("run_anomaly_model", "gather_evidence")
    builder.add_edge("gather_evidence", "generate_hypotheses")
    builder.add_edge("generate_hypotheses", "investigate_hypotheses")

    builder.add_conditional_edges(
        "investigate_hypotheses",
        route_after_investigation,
        {"sufficient": "determine_root_cause", "insufficient": "gather_more_evidence"},
    )
    builder.add_edge("gather_more_evidence", "investigate_hypotheses")  # the loop

    builder.add_edge("determine_root_cause", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()
