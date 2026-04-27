# app/graph/builder.py

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Dict, Any, Callable

# ✅ Correct imports (FUNCTIONS ONLY — NOT MODULES)
from app.nodes.planner import planner_node
from app.nodes.retriever import retriever_node
from app.nodes.generator import generator_node
from app.nodes.trace_node import trace_node

# ==============================
# 🔹 STATE DEFINITION
# ==============================

class GraphState(TypedDict, total=False):
    query: str
    context: str
    answer: str
    trace: Dict[str, Any]
    error: str


# ==============================
# 🔹 VALIDATION LAYER (CRITICAL)
# ==============================

def validate_node(name: str, node: Callable):
    if node is None:
        raise ValueError(f"Node '{name}' is None")

    if not callable(node):
        raise TypeError(
            f"Node '{name}' must be callable, got {type(node)}"
        )


def wrap_node(node: Callable):
    """Wrap node into RunnableLambda for LangGraph safety"""
    return RunnableLambda(node)


# ==============================
# 🔹 SAFE NODE REGISTRATION
# ==============================

def get_nodes():
    nodes = {
        "planner": planner_node,
        "retriever": retriever_node,
        "generator": generator_node,
        "trace": trace_node,
    }

    # ✅ Validate ALL nodes before graph build
    for name, node in nodes.items():
        validate_node(name, node)

    # ✅ Wrap for safety
    return {k: wrap_node(v) for k, v in nodes.items()}


# ==============================
# 🔹 GRAPH BUILDER
# ==============================

def build_graph():
    nodes = get_nodes()

    graph = StateGraph(GraphState)

    # Add nodes
    for name, node in nodes.items():
        graph.add_node(name, node)

    # Flow
    graph.set_entry_point("planner")

    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "trace")
    graph.add_edge("trace", END)

    return graph.compile()
