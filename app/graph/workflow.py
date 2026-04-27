# app/graph/workflow.py

from langgraph.graph import StateGraph, END
from app.state import AgentState

# ✅ Import actual callables, not modules
from app.nodes.planner import run as planner_run
from app.nodes.retrieval import run as retrieval_run
from app.nodes.compute import run as compute_run
from app.nodes.sql_tool import run as sql_run
from app.nodes.router import run as router_run
from app.nodes.chat import run as chat_run
from app.nodes.context_assembler import run as context_run
from app.nodes.summarizer import run as summarizer_run
from app.nodes.token_manager import run as token_run
from app.nodes.generator import run as generator_run
from app.nodes.verifier import run as verifier_run
from app.nodes.retry_controller import run as retry_run
from app.nodes.hitl import run as hitl_run
from app.nodes.trace_node import run as trace_run

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
    graph.add_node("trace_start", trace_run)
    graph.add_node("planner", planner_run)
    graph.add_node("router", router_run)
    graph.add_node("chat", chat_run)
    graph.add_node("compute", compute_run)
    graph.add_node("summarizer", summarizer_run)
    graph.add_node("retrieval", retrieval_run)
    graph.add_node("context_assembler", context_run)
    graph.add_node("token_manager", token_run)
    graph.add_node("generator", generator_run)
    graph.add_node("verifier", verifier_run)
    graph.add_node("retry_controller", retry_run)
    graph.add_node("hitl", hitl_run)
    graph.add_node("trace_end", trace_run)

    # -------- Entry --------
    graph.set_entry_point("trace_start")

    # -------- Initial Flow --------
    graph.add_edge("trace_start", "planner")
    graph.add_edge("planner", "router")

    # -------- Routing --------
    graph.add_conditional_edges(
        "router",
        guard_route,
        {"chat": "chat", "compute": "compute", "rag": "retrieval"},
    )

    # -------- CHAT --------
    graph.add_edge("chat", "trace_end")

    # -------- COMPUTE --------
    graph.add_edge("compute", "trace_end")

    # -------- RAG FLOW --------
    graph.add_edge("retrieval", "context_assembler")
    graph.add_edge("context_assembler", "token_manager")

    graph.add_conditional_edges(
        "token_manager",
        guard_context,
        {"ok": "summarizer", "no_context": "hitl"},
    )

    graph.add_edge("summarizer", "generator")

    graph.add_edge("generator", "verifier")

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

    # -------- Global Halt --------
    for node in [
        "planner", "router", "retrieval", "context_assembler",
        "token_manager", "summarizer", "generator", "verifier",
        "retry_controller",
    ]:
        graph.add_conditional_edges(
            node,
            guard_halt,
            {"halt": "trace_end", "continue": None},
        )

    graph.add_edge("trace_end", END)
    return graph.compile()
