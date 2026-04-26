# app/graph/workflow.py

from langgraph.graph import StateGraph, END
from app.state import AgentState

# Nodes
from app.nodes import (
    planner,
    router,
    chat,
    retrieval,
    context_assembler,
    summarizer,
    token_manager,
    compute,
    generator,
    verifier,
    retry_controller,
    hitl,
    trace_node,
)

# -----------------------------
# 🔒 Guard Functions
# -----------------------------

def guard_halt(state: AgentState):
    if state.get("halted"):
        return "halt"
    return "continue"


def guard_route(state: AgentState):
    intent = state.get("intent", "rag")

    if intent == "chat":
        return "chat"
    if intent == "compute":
        return "compute"
    return "rag"


def guard_context(state: AgentState):
    ctx = (state.get("context") or "").strip()
    if len(ctx) < 50:
        return "no_context"
    return "ok"


def guard_verified(state: AgentState):
    return "verified" if state.get("verified") else "rejected"


def guard_retry(state: AgentState):
    retries = int(state.get("retry_count", 0))
    max_r = int(state.get("max_retries", 1))
    conf = float(state.get("confidence", 0.0))

    if state.get("halted"):
        return "no_retry"

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

    # NEW
    graph.add_node("chat", chat)
    graph.add_node("compute", compute)
    graph.add_node("summarizer", summarizer)

    # EXISTING
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

    # -------- Initial Flow --------
    graph.add_edge("trace_start", "planner")
    graph.add_edge("planner", "router")

    # -------- Routing (UPGRADED) --------
    graph.add_conditional_edges(
        "router",
        guard_route,
        {
            "chat": "chat",
            "compute": "compute",
            "rag": "retrieval",
        },
    )

    # -------- CHAT (direct exit) --------
    graph.add_edge("chat", "trace_end")

    # -------- COMPUTE (direct exit) --------
    graph.add_edge("compute", "trace_end")

    # -------- RAG FLOW --------
    graph.add_edge("retrieval", "context_assembler")
    graph.add_edge("context_assembler", "token_manager")

    # RAG LOCK
    graph.add_conditional_edges(
        "token_manager",
        guard_context,
        {
            "ok": "summarizer",   # NEW insertion
            "no_context": "hitl",
        },
    )

    # NEW: summarization before generation
    graph.add_edge("summarizer", "generator")

    # -------- Generation --------
    graph.add_edge("generator", "verifier")

    # -------- Verification --------
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

    # -------- Global Halt --------
    for node in [
        "planner",
        "router",
        "retrieval",
        "context_assembler",
        "token_manager",
        "summarizer",
        "generator",
        "verifier",
        "retry_controller",
    ]:
        graph.add_conditional_edges(
            node,
            guard_halt,
            {
                "halt": "trace_end",
                "continue": None,
            },
        )

    # -------- Exit --------
    graph.add_edge("trace_end", END)

    return graph.compile()
