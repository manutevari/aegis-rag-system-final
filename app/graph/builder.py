# app/graph/builder.py
"""
Minimal graph builder used by api.py.

FIXES:
1. Import was `from app.nodes.retriever import retriever_node` — file is
   `retrieval.py` and function is `run`.  Corrected to retrieval.run.
2. State schema changed to AgentState (TypedDict mismatch with AgentState
   Pydantic model caused LangGraph type errors).
3. Removed RunnableLambda wrapping — LangGraph ≥ 0.2 accepts plain callables.
4. generator_node import path fixed (generator.py exposes `run`, not
   `generator_node`).
5. trace node import corrected (`run`, not `trace_node` function name).
"""

from langgraph.graph import StateGraph, END
from app.state import AgentState

from app.nodes.planner   import run as planner_node
from app.nodes.retrieval import run as retriever_node   # FIX: was app.nodes.retriever
from app.nodes.generator import run as generator_node   # FIX: was generator_node (non-existent)
from app.nodes.trace_node import run as trace_node_fn   # FIX: was trace_node (non-existent)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner",   planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("trace",     trace_node_fn)

    graph.set_entry_point("planner")

    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "trace")
    graph.add_edge("trace",     END)

    return graph.compile()
