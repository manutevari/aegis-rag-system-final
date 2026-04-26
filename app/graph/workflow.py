"""
LangGraph Workflow — Decision-Grade RAG backbone (UPGRADED)
"""

import logging
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes import (
    planner, retrieval, sql_tool, compute,
    context_assembler, token_manager,
    generator, verifier, hitl,
    encrypt_node, decrypt_node, trace_node,

    # ✅ Tools
    numpy_node,
    pandas_node,
    plot_node,

    # 🔥 NEW (critical)
    retry_controller,
)

logger = logging.getLogger(__name__)


# ── Routing ─────────────────────────────────────────────────────────────

def _route_after_planner(state: AgentState) -> str:
    return state.get("route", "retrieval")


def _route_after_token(state: AgentState) -> str:
    return "summarize" if state.get("token_count", 0) > 3000 else "generate"


def _route_after_retry(state: AgentState) -> str:
    return state.get("route", "generate")


def _route_after_hitl(state: AgentState) -> str:
    decision = state.get("hitl_decision", "approve")
    return "approved" if decision == "approve" else "rejected"


# ── Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    b = StateGraph(AgentState)

    # ── Core Nodes ──
    b.add_node("planner",           planner.run)
    b.add_node("retrieval",         retrieval.run)
    b.add_node("sql",               sql_tool.run)
    b.add_node("compute",           compute.run)

    # ✅ Tool Nodes
    b.add_node("numpy_compute",     numpy_node.run)
    b.add_node("pandas_query",      pandas_node.run)
    b.add_node("plot_chart",        plot_node.run)

    b.add_node("context_assembler", context_assembler.run)
    b.add_node("token_check",       token_manager.run)
    b.add_node("summarize_context", token_manager.summarize)

    b.add_node("generate",          generator.run)
    b.add_node("verify",            verifier.run)

    # 🔥 NEW CONTROL LAYER
    b.add_node("retry_controller",  retry_controller.run)

    b.add_node("hitl",              hitl.run)
    b.add_node("encrypt",           encrypt_node.run)
    b.add_node("decrypt",           decrypt_node.run)
    b.add_node("trace",             trace_node.run)

    # Entry
    b.set_entry_point("planner")

    # ── Planner Routing ──
    b.add_conditional_edges(
        "planner", _route_after_planner,
        {
            "sql": "sql",
            "retrieval": "retrieval",
            "compute": "compute",
            "direct": "context_assembler",

            # tools
            "numpy_compute": "numpy_compute",
            "pandas_query": "pandas_query",
            "plot_chart": "plot_chart",
        },
    )

    # ── Tool Pipelines (UPGRADED) ──

    # Retrieval
    b.add_edge("retrieval", "context_assembler")

    # SQL → Compute → Pandas → Context
    b.add_edge("sql", "compute")
    b.add_edge("compute", "pandas_query")
    b.add_edge("pandas_query", "context_assembler")

    # NumPy direct
    b.add_edge("numpy_compute", "context_assembler")

    # Plot always uses pandas
    b.add_edge("plot_chart", "pandas_query")

    # ── Context Flow ──
    b.add_edge("context_assembler", "token_check")

    b.add_conditional_edges(
        "token_check", _route_after_token,
        {
            "summarize": "summarize_context",
            "generate": "generate",
        },
    )

    b.add_edge("summarize_context", "generate")

    # ── Generation ──
    b.add_edge("generate", "verify")

    # ── Validation Layer ──
    b.add_edge("verify", "retry_controller")

    # 🔥 Retry Logic (SMART)
    b.add_conditional_edges(
        "retry_controller", _route_after_retry,
        {
            "generate": "generate",   # retry with upgraded model
            "trace": "trace",         # accept
            "hitl": "hitl",           # escalate
        },
    )

    # ── HITL ──
    b.add_conditional_edges(
        "hitl", _route_after_hitl,
        {
            "approved": "encrypt",
            "rejected": "generate",   # 🔥 retry with feedback
        },
    )

    # ── Final ──
    b.add_edge("encrypt", "decrypt")
    b.add_edge("decrypt", "trace")
    b.add_edge("trace", END)

    return b.compile()
