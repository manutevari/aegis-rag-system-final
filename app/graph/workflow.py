"""
Merged Workflow — Simple backbone with LangGraph scalability
"""

import logging
from langgraph.graph import StateGraph, END
from app.state import AgentState, to_state
from app.tools import retriever, planner, sql_tool, compute

logger = logging.getLogger(__name__)

# ── Simple node wrappers ──
def planner_run(state: AgentState) -> AgentState:
    state = to_state(state)
    q = state.query.lower()
    if "budget" in q:
        state.route = "sql"
    elif "calculate" in q:
        state.route = "compute"
    else:
        state.route = "retrieval"
    return state

def retrieval_run(state: AgentState) -> AgentState:
    state = to_state(state)
    docs = retriever.run(state) if hasattr(retriever, "run") else ["stub doc"]
    state.retrieval_docs = docs
    state.context = " ".join(docs)
    state.answer = f"Answer from retrieval: {state.context}"
    return state

def sql_run(state: AgentState) -> AgentState:
    state = to_state(state)
    result = sql_tool.run(state) if hasattr(sql_tool, "run") else [{"budget": "1000"}]
    state.sql_result = result
    state.answer = f"SQL result: {state.sql_result}"
    return state

def compute_run(state: AgentState) -> AgentState:
    state = to_state(state)
    result = compute.run(state) if hasattr(compute, "run") else 42.0
    state.compute_result = result
    state.compute_summary = "Computed successfully."
    state.answer = f"Compute result: {state.compute_result}"
    return state

# ── Routing helpers ──
def _route_after_planner(state: AgentState) -> str:
    return state.route

# ── Graph builder ──
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Core nodes
    g.add_node("planner", planner_run)
    g.add_node("retrieval", retrieval_run)
    g.add_node("sql", sql_run)
    g.add_node("compute", compute_run)

    # Entry
    g.set_entry_point("planner")

    # Planner routing
    g.add_conditional_edges(
        "planner", _route_after_planner,
        {
            "retrieval": "retrieval",
            "sql": "sql",
            "compute": "compute",
        },
    )

    # End each branch
    g.add_edge("retrieval", END)
    g.add_edge("sql", END)
    g.add_edge("compute", END)

    return g.compile()

# ── Example usage ──
if __name__ == "__main__":
    graph = build_graph()
    q = "What is the laptop budget for L6 employees?"
    result = graph.invoke(AgentState(query=q))
    print("Query:", q)
    print("Route:", result.route)
    print("Answer:", result.answer)
