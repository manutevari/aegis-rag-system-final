# app/graph/workflow.py
"""
AEGIS unified graph workflow.

FIXES:
1. Removed duplicate build_graph() definition (first one was silently
   overwriting the full pipeline — Python last-def wins).
2. All nodes imported as callables (run functions), not modules.
3. Guard functions use .get() safely on plain dicts AND AgentState.
4. router node was imported as module in __init__.py but used as callable
   in workflow — now consistently imported as router_run.
5. summarizer node imported from correct file.
6. confidence node wired into graph (was computed but never called).
"""

from langgraph.graph import StateGraph, END
from app.state import AgentState

# ── Callable imports (NOT modules) ──────────────────────────────────────────
from app.nodes.planner       import run as planner_run
from app.nodes.router        import run as router_run
from app.nodes.chat          import run as chat_run
from app.nodes.retrieval     import run as retrieval_run
from app.nodes.context_assembler import run as context_run
from app.nodes.token_manager import run as token_run
from app.nodes.summarizer    import run as summarizer_run
from app.nodes.generator     import run as generator_run
from app.nodes.confidence    import run as confidence_run
from app.nodes.verifier      import run as verifier_run
from app.nodes.retry_controller import run as retry_run
from app.nodes.hitl          import run as hitl_run
from app.nodes.trace_node    import run as trace_run
from app.nodes.compute       import run as compute_run


# ─────────────────────────────────────────────────────────────────────────────
# Guard functions
# ─────────────────────────────────────────────────────────────────────────────

def guard_halt(state):
    if state.get("halted"):
        return "halt"
    return "continue"


def guard_route(state):
    intent = state.get("intent", "rag")
    if intent == "chat":
        return "chat"
    if intent == "compute":
        return "compute"
    return "rag"


def guard_context(state):
    ctx = (state.get("context") or "").strip()
    if len(ctx) < 50:
        return "no_context"
    return "ok"


def guard_verified(state):
    return "verified" if state.get("verified") else "rejected"


def guard_retry(state):
    retries = int(state.get("retry_count", 0))
    max_r   = int(state.get("max_retries", 1))
    conf    = float(state.get("confidence", 0.0))

    if state.get("halted"):
        return "no_retry"
    if conf < 0.35 and retries < max_r:
        return "retry"
    return "no_retry"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("trace_start",       trace_run)
    graph.add_node("planner",           planner_run)
    graph.add_node("router",            router_run)
    graph.add_node("chat",              chat_run)
    graph.add_node("compute",           compute_run)
    graph.add_node("retrieval",         retrieval_run)
    graph.add_node("context_assembler", context_run)
    graph.add_node("token_manager",     token_run)
    graph.add_node("summarizer",        summarizer_run)
    graph.add_node("generator",         generator_run)
    graph.add_node("confidence",        confidence_run)
    graph.add_node("verifier",          verifier_run)
    graph.add_node("retry_controller",  retry_run)
    graph.add_node("hitl",              hitl_run)
    graph.add_node("trace_end",         trace_run)

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("trace_start")

    # ── Initial flow ──────────────────────────────────────────────────────────
    graph.add_edge("trace_start", "planner")
    graph.add_edge("planner",     "router")

    # ── Routing ───────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        guard_route,
        {"chat": "chat", "compute": "compute", "rag": "retrieval"},
    )

    # ── Chat path ─────────────────────────────────────────────────────────────
    graph.add_edge("chat", "trace_end")

    # ── Compute path ──────────────────────────────────────────────────────────
    graph.add_edge("compute", "trace_end")

    # ── RAG path ──────────────────────────────────────────────────────────────
    graph.add_edge("retrieval",         "context_assembler")
    graph.add_edge("context_assembler", "token_manager")

    graph.add_conditional_edges(
        "token_manager",
        guard_context,
        {"ok": "summarizer", "no_context": "hitl"},
    )

    graph.add_edge("summarizer", "generator")
    graph.add_edge("generator",  "confidence")
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

    # ── Global halt edges ─────────────────────────────────────────────────────
    for node in [
        "planner", "router", "retrieval", "context_assembler",
        "token_manager", "summarizer", "generator", "confidence",
        "verifier", "retry_controller",
    ]:
        graph.add_conditional_edges(
            node,
            guard_halt,
            {"halt": "trace_end", "continue": END},
        )

    # ── Exit ──────────────────────────────────────────────────────────────────
    graph.add_edge("trace_end", END)

    return graph.compile()
