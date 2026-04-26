"""
Merged Workflow — Simple backbone with LangGraph scalability
"""

import logging
from langgraph.graph import StateGraph, END
from app.state import AgentState, to_state
from app.core.vector_store import vector_db
from app.core.models import get_chat_model

logger = logging.getLogger(__name__)

# ── Simple node wrappers ──
def planner_run(state: AgentState) -> AgentState:
    """Route query based on keywords."""
    state = to_state(state)
    q = state.query.lower()
    if "budget" in q or "cost" in q:
        state.route = "sql"
    elif "calculate" in q or "compute" in q:
        state.route = "compute"
    else:
        state.route = "retrieval"
    return state

def retrieval_run(state: AgentState) -> AgentState:
    """Fetch from vector store."""
    state = to_state(state)
    docs = vector_db.search(state.query, top_k=5)
    state.retrieval_docs = [d.page_content for d in docs]
    state.context = " ".join(state.retrieval_docs)
    
    # Generate answer
    model = get_chat_model()
    prompt = f"Based on: {state.context}\n\nAnswer: {state.query}"
    response = model.invoke(prompt)
    state.answer = response.content if hasattr(response, 'content') else str(response)
    
    return state

def sql_run(state: AgentState) -> AgentState:
    """Execute SQL (stub)."""
    state = to_state(state)
    state.answer = f"SQL query result for: {state.query}"
    return state

def compute_run(state: AgentState) -> AgentState:
    """Execute computation (stub)."""
    state = to_state(state)
    state.answer = f"Computed result for: {state.query}"
    return state

def _route_after_planner(state: AgentState) -> str:
    """Route decision."""
    return state.route

def build_graph():
    """Build LangGraph workflow."""
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("planner", planner_run)
    g.add_node("retrieval", retrieval_run)
    g.add_node("sql", sql_run)
    g.add_node("compute", compute_run)

    # Entry
    g.set_entry_point("planner")

    # Conditional routing
    g.add_conditional_edges(
        "planner", _route_after_planner,
        {"retrieval": "retrieval", "sql": "sql", "compute": "compute"}
    )

    # End
    g.add_edge("retrieval", END)
    g.add_edge("sql", END)
    g.add_edge("compute", END)

    return g.compile()
