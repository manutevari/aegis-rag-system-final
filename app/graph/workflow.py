# app/graph/workflow.py

from langgraph.graph import StateGraph, END
from app.state import AgentState

# Nodes
from app.nodes import (
    planner,
    retrieval,
    context_assembler,
    token_manager,
    generator,
    verifier,
    router,
    retry_controller,
    hitl,
    trace_node,
)

# -----------------------------
# 🔒 Guard Functions (Critical)
# -----------------------------

def guard_halt(state: AgentState):
    """Global halt guard."""
    if state.get("halted"):
        return "halt"
    return "continue"


def guard_route(state: AgentState):
    """Route decision from router node."""
    route = state.get("route", "rag")
    if route == "rag":
        return "rag"
    if route == "tools":
        return "tools"
    return "direct"


def guard_context(state: AgentState):
    """RAG lock: no context → stop."""
    ctx = (state.get("context") or "").strip()
    if len(ctx) < 50:
        return "no_context"
    return "ok"


def guard_verified(state: AgentState):
    """Verifier gate."""
    if state.get("verified") is True:
        return "verified"
    return "rejected"


def guard_retry(state: AgentState):
    """Retry policy: only if confidence low and retries left."""
    retries = int(state.get("retry_count", 0))
    max_r = int(state.get("max_retries", 1))
    conf = float(state.get("confidence", 0.0))

    # ❌ Never retry if halted or no context
    if state.get("halted"):
        return "no_retry"

    # Retry only on low confidence
    if conf < 0.35 and retries < max_r:
        return "retry"

    return "no_retry"


# -----------------------------
# 🧠 Graph Builder
# -----------------------------

def build_graph():
    graph = StateGraph(AgentState)

    # -------- Nodes --------
    graph.add_node("trace_start", trace_node)
    graph.add_node("planner", planner)
    graph.add_node("router", router)

    graph.add_node("retrieval", retrieval)
    graph.add_node("context_assembler", context_assembler)
    graph.add_node("token_manager", token_manager)

    graph.add_node("generator", generator)
    graph.add_node("verifier", verifier)

    graph.add_node("retry_controller", retry_controller)
    graph.add_node("hitl", hitl)
    graph.add_node("trace_end", trace_node)

    # -------- Entry --------
    graph.set_entry_point("trace_start")

    # -------- Linear Start --------
    graph.add_edge("trace_start", "planner")
    graph.add_edge("planner", "router")

    # -------- Routing --------
    graph.add_conditional_edges(
        "router",
        guard_route,
        {
            "rag": "retrieval",
            "tools": "generator",   # if you have tool executor, place it here
            "direct": "generator",
        },
    )

    # -------- RAG Path --------
    graph.add_edge("retrieval", "context_assembler")
    graph.add_edge("context_assembler", "token_manager")

    # RAG LOCK
    graph.add_conditional_edges(
        "token_manager",
        guard_context,
        {
            "ok": "generator",
            "no_context": "hitl",   # or END if you prefer hard stop
        },
    )

    # -------- Generation --------
    graph.add_edge("generator", "verifier")

    # -------- Verification Gate --------
    graph.add_conditional_edges(
        "verifier",
        guard_verified,
        {
            "verified": "trace_end",
            "rejected": "retry_controller",
        },
    )

    # -------- Retry Loop --------
    graph.add_conditional_edges(
        "retry_controller",
        guard_retry,
        {
            "retry": "generator",
            "no_retry": "hitl",
        },
    )

    # -------- HITL --------
    graph.add_edge("hitl", "trace_end")

    # -------- Global Halt Guards --------
    # Apply halt checks after key nodes
    for node in [
        "planner",
        "router",
        "retrieval",
        "context_assembler",
        "token_manager",
        "generator",
        "verifier",
        "retry_controller",
    ]:
        graph.add_conditional_edges(
            node,
            guard_halt,
            {
                "halt": "trace_end",
                "continue": None,  # fallthrough to existing edges
            },
        )

    # -------- Exit --------
    graph.add_edge("trace_end", END)

    return graph.compile()
