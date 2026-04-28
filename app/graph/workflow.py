"""
AEGIS unified graph workflow.

All guard functions use safe state access so they work with either dict state or
AgentState objects. The graph avoids adding catch-all halt edges to nodes that
already have normal outgoing edges, which keeps routing deterministic.
"""

from langgraph.graph import END, StateGraph

from app.core.stability_patch import safe_get
from app.state import AgentState

from app.nodes.chat import run as chat_run
from app.nodes.compute import run as compute_run
from app.nodes.confidence import run as confidence_run
from app.nodes.context_assembler import run as context_run
from app.nodes.generator import run as generator_run
from app.nodes.hitl import run as hitl_run
from app.nodes.planner import run as planner_run
from app.nodes.retrieval import run as retrieval_run
from app.nodes.retry_controller import run as retry_run
from app.nodes.router import run as router_run
from app.nodes.summarizer import run as summarizer_run
from app.nodes.token_manager import run as token_run
from app.nodes.trace_node import run as trace_run
from app.nodes.verifier import run as verifier_run


def guard_route(state):
    intent = safe_get(state, "intent", "rag") or "rag"
    if intent == "chat":
        return "chat"
    if intent == "compute":
        return "compute"
    return "rag"


def guard_context(state):
    context = (safe_get(state, "context", "") or "").strip()
    if len(context) < 50:
        return "no_context"
    return "ok"


def guard_verified(state):
    return "verified" if safe_get(state, "verified", False) else "rejected"


def guard_retry(state):
    retries = int(safe_get(state, "retry_count", 0) or 0)
    max_retries = int(safe_get(state, "max_retries", 1) or 1)
    confidence = float(safe_get(state, "confidence", 0.0) or 0.0)

    if safe_get(state, "halted", False):
        return "no_retry"
    if confidence < 0.35 and retries < max_retries:
        return "retry"
    return "no_retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("trace_start", trace_run)
    graph.add_node("planner", planner_run)
    graph.add_node("router", router_run)
    graph.add_node("chat", chat_run)
    graph.add_node("compute", compute_run)
    graph.add_node("retrieval", retrieval_run)
    graph.add_node("context_assembler", context_run)
    graph.add_node("token_manager", token_run)
    graph.add_node("summarizer", summarizer_run)
    graph.add_node("generator", generator_run)
    graph.add_node("confidence", confidence_run)
    graph.add_node("verifier", verifier_run)
    graph.add_node("retry_controller", retry_run)
    graph.add_node("hitl", hitl_run)
    graph.add_node("trace_end", trace_run)

    graph.set_entry_point("trace_start")

    graph.add_edge("trace_start", "planner")
    graph.add_edge("planner", "router")

    graph.add_conditional_edges(
        "router",
        guard_route,
        {"chat": "chat", "compute": "compute", "rag": "retrieval"},
    )

    graph.add_edge("chat", "trace_end")
    graph.add_edge("compute", "trace_end")

    graph.add_edge("retrieval", "context_assembler")
    graph.add_edge("context_assembler", "token_manager")

    graph.add_conditional_edges(
        "token_manager",
        guard_context,
        {"ok": "summarizer", "no_context": "hitl"},
    )

    graph.add_edge("summarizer", "generator")
    graph.add_edge("generator", "confidence")
    graph.add_edge("confidence", "verifier")

    graph.add_conditional_edges(
        "verifier",
        guard_verified,
        {"verified": "trace_end", "rejected": "retry_controller"},
    )

    graph.add_conditional_edges(
        "retry_controller",
        guard_retry,
        {"retry": "generator", "no_retry": "hitl"},
    )

    graph.add_edge("hitl", "trace_end")
    graph.add_edge("trace_end", END)

    return graph.compile()
