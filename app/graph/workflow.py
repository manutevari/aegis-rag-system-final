"""
LangGraph Workflow — Decision-Grade RAG backbone.

Full node map (matches architecture diagram):
  START
    └─► planner ──(route)──► sql ──► compute ──► context_assembler
                         └─► retrieval ──────────► context_assembler
                         └─► direct ─────────────► context_assembler
    context_assembler ──► token_check
       ├─(summarize)──► summarize_context ──► generate
       └─(proceed) ─────────────────────────► generate
    generate ──► verify
       ├─(valid) ──► hitl
       │                ├─(approve/edit) ──► encrypt ──► decrypt ──► trace ──► END
       │                └─(reject) ──────────────────────────────────► planner (retry)
       └─(invalid) ──────────────────────────────────────────────────► planner (retry)
"""

import logging
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes import (
    planner, retrieval, sql_tool, compute,
    context_assembler, token_manager,
    generator, verifier, hitl,
    encrypt_node, decrypt_node, trace_node,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ── Routing lambdas ────────────────────────────────────────────────────────────

def _route_after_planner(state: AgentState) -> str:
    return state.get("route", "retrieval")

def _route_after_token(state: AgentState) -> str:
    return "summarize" if state.get("token_count", 0) > 3000 else "proceed"

def _route_after_verify(state: AgentState) -> str:
    return "valid" if state.get("verified", False) else "invalid"

def _route_after_hitl(state: AgentState) -> str:
    decision = state.get("hitl_decision", "approve")
    if decision == "reject":
        retries = state.get("retry_count", 0)
        return "retry" if retries < MAX_RETRIES else "end_failed"
    return "end_ok"


# ── Builder ────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    b = StateGraph(AgentState)

    # Nodes
    b.add_node("planner",           planner.run)
    b.add_node("retrieval",         retrieval.run)
    b.add_node("sql",               sql_tool.run)
    b.add_node("compute",           compute.run)
    b.add_node("context_assembler", context_assembler.run)
    b.add_node("token_check",       token_manager.run)
    b.add_node("summarize_context", token_manager.summarize)
    b.add_node("generate",          generator.run)
    b.add_node("verify",            verifier.run)
    b.add_node("hitl",              hitl.run)
    b.add_node("encrypt",           encrypt_node.run)
    b.add_node("decrypt",           decrypt_node.run)
    b.add_node("trace",             trace_node.run)

    # Entry
    b.set_entry_point("planner")

    # Planner → tool routing
    b.add_conditional_edges(
        "planner", _route_after_planner,
        {"sql": "sql", "retrieval": "retrieval", "compute": "compute", "direct": "context_assembler"},
    )

    # Tool convergence to context
    b.add_edge("retrieval", "context_assembler")
    b.add_edge("sql",       "compute")          # SQL always feeds compute for numerical grounding
    b.add_edge("compute",   "context_assembler")

    # Context → token gate
    b.add_edge("context_assembler", "token_check")
    b.add_conditional_edges(
        "token_check", _route_after_token,
        {"summarize": "summarize_context", "proceed": "generate"},
    )
    b.add_edge("summarize_context", "generate")

    # Generate → verify
    b.add_edge("generate", "verify")

    # Verify → HITL or retry
    b.add_conditional_edges(
        "verify", _route_after_verify,
        {"valid": "hitl", "invalid": "planner"},
    )

    # HITL → final path or retry
    b.add_conditional_edges(
        "hitl", _route_after_hitl,
        {"end_ok": "encrypt", "retry": "planner", "end_failed": "trace"},
    )

    # Final pipeline
    b.add_edge("encrypt", "decrypt")
    b.add_edge("decrypt", "trace")
    b.add_edge("trace",   END)

    return b.compile()
