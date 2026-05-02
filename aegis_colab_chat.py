"""Colab-style Python entrypoint for the AEGIS RAG system.

This file combines the single-file feel of ``aegis-ai-rag-system`` with the
production LangGraph pipeline in this repository. It can be run as a Streamlit
chat app with a per-node workflow chart, or as a simple terminal chat loop.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from pydantic import BaseModel, Field
except ImportError:  # Keeps the CLI and chart helpers usable before pip install.
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            return dict(self.__dict__)

    def Field(default_factory=None, default=None, **_: Any):
        return default_factory() if default_factory else default


APP_TITLE = "AEGIS Colab Chat"
DEFAULT_EMPTY_ANSWERS = {
    "",
    "No response generated",
    "I'm sorry, I couldn't find relevant information.",
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    title: str
    group: str
    purpose: str


class TurnMetadata(BaseModel):
    query: str
    route: str = "unknown"
    intent: str = "unknown"
    answer_characters: int = 0
    sources: List[str] = Field(default_factory=list)


NODE_SPECS: Tuple[NodeSpec, ...] = (
    NodeSpec("trace_start", "Trace Start", "entry", "Open a trace for this question."),
    NodeSpec("planner", "Planner", "routing", "Choose the policy route and grade context."),
    NodeSpec("router", "Router", "routing", "Send the question to chat, compute, or RAG."),
    NodeSpec("chat", "Chat", "branch", "Handle greetings and simple conversation."),
    NodeSpec("compute", "Compute", "branch", "Run numerical policy calculations."),
    NodeSpec("retrieval", "Retrieval", "rag", "Expand, filter, retrieve, and rerank policy chunks."),
    NodeSpec("context_assembler", "Context", "rag", "Build grounded answer context."),
    NodeSpec("token_manager", "Tokens", "rag", "Keep context inside the answer budget."),
    NodeSpec("summarizer", "Summarizer", "rag", "Condense long policy context."),
    NodeSpec("generator", "Generator", "answer", "Generate the grounded response."),
    NodeSpec("confidence", "Confidence", "quality", "Score answer confidence."),
    NodeSpec("verifier", "Verifier", "quality", "Check grounding and numerical consistency."),
    NodeSpec("retry_controller", "Retry", "quality", "Retry low-confidence answers when useful."),
    NodeSpec("hitl", "HITL", "quality", "Fall back to a human-review style response."),
    NodeSpec("trace_end", "Trace End", "exit", "Close the trace and return the answer."),
)

NODE_BY_NAME = {node.name: node for node in NODE_SPECS}

WORKFLOW_EDGES: Tuple[Tuple[str, str, str], ...] = (
    ("trace_start", "planner", ""),
    ("planner", "router", ""),
    ("router", "chat", "chat"),
    ("router", "compute", "compute"),
    ("router", "retrieval", "rag"),
    ("chat", "trace_end", ""),
    ("compute", "trace_end", ""),
    ("retrieval", "context_assembler", ""),
    ("context_assembler", "token_manager", ""),
    ("token_manager", "summarizer", "ok"),
    ("token_manager", "hitl", "no context"),
    ("summarizer", "generator", ""),
    ("generator", "confidence", ""),
    ("confidence", "verifier", ""),
    ("verifier", "trace_end", "verified"),
    ("verifier", "retry_controller", "rejected"),
    ("retry_controller", "generator", "retry"),
    ("retry_controller", "hitl", "stop"),
    ("hitl", "trace_end", ""),
)

NODE_POSITIONS: Dict[str, Tuple[float, float]] = {
    "trace_start": (0.08, 0.86),
    "planner": (0.23, 0.86),
    "router": (0.38, 0.86),
    "chat": (0.55, 0.96),
    "compute": (0.55, 0.86),
    "retrieval": (0.55, 0.72),
    "context_assembler": (0.72, 0.72),
    "token_manager": (0.89, 0.72),
    "summarizer": (0.89, 0.50),
    "generator": (0.72, 0.50),
    "confidence": (0.55, 0.50),
    "verifier": (0.38, 0.50),
    "retry_controller": (0.23, 0.50),
    "hitl": (0.23, 0.28),
    "trace_end": (0.55, 0.28),
}

GROUP_COLORS = {
    "entry": "#e9eef2",
    "routing": "#fff2cc",
    "branch": "#dcefe8",
    "rag": "#e4e9fb",
    "answer": "#e7f1ff",
    "quality": "#f6e1e6",
    "exit": "#e8ecef",
}

ACTIVE_COLOR = "#246bfe"
TEXT_COLOR = "#17202a"
EDGE_COLOR = "#6b7280"


def bootstrap_repo() -> Path:
    """Make the repository importable from Streamlit, Colab, and CLI runs."""
    repo_root = Path(__file__).resolve().parent
    for path in (repo_root, repo_root.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return repo_root


def configure_runtime_defaults() -> None:
    """Use safe local defaults unless the caller has selected another provider."""
    os.environ.setdefault("RAG_EMBEDDINGS_PROVIDER", "hash")
    os.environ.setdefault("LLM_PROVIDER", "extractive")
    os.environ.setdefault("AUTO_INGEST", "true")


def is_basic_chat(query: str) -> bool:
    lower = (query or "").strip().lower()
    if not lower:
        return False
    return bool(
        re.search(
            r"\b(hi|hello|hey|thanks|thank you|who are you|how are you|help)\b",
            lower,
        )
    )


def basic_chat_answer(query: str) -> str:
    lower = (query or "").lower()
    if "thank" in lower:
        return "You are welcome. Send me any policy question when you are ready."
    if "who are you" in lower:
        return "I am AEGIS, a policy intelligence assistant for company policy questions, calculations, and source-grounded answers."
    if "how are you" in lower:
        return "I am ready and tracking the policy graph. What would you like to check?"
    if "help" in lower:
        return (
            "Ask about travel, fuel, HR, security, reimbursements, eligibility, "
            "or a quick calculation such as total allowance for a trip."
        )
    return "Hi. Ask me a policy question and I will route it through the AEGIS graph."


def trace_nodes(trace_log: Iterable[Any]) -> List[str]:
    visited: List[str] = []
    seen: Set[str] = set()

    for item in trace_log or []:
        node_name = ""
        if isinstance(item, dict):
            node_name = str(item.get("node") or item.get("name") or "")
        elif isinstance(item, str):
            lowered = item.lower()
            node_name = next((name for name in NODE_BY_NAME if name in lowered), "")

        if node_name in NODE_BY_NAME and node_name not in seen:
            visited.append(node_name)
            seen.add(node_name)

    return visited


def node_rows(trace_log: Iterable[Any]) -> List[Dict[str, str]]:
    visited = set(trace_nodes(trace_log))
    return [
        {
            "node": node.name,
            "title": node.title,
            "group": node.group,
            "status": "visited" if node.name in visited else "pending",
            "purpose": node.purpose,
        }
        for node in NODE_SPECS
    ]


def serialise_sources(raw_sources: Any, documents: Any = None) -> List[str]:
    sources: List[str] = []
    for source in raw_sources or []:
        value = str(source).strip()
        if value and value not in sources:
            sources.append(value)

    for document in documents or []:
        if not isinstance(document, dict):
            continue
        source = str(document.get("source") or "").strip()
        if source and source not in sources:
            sources.append(source)
    return sources


def normalise_result(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    data = dict(result or {})
    answer = str(data.get("answer") or "").strip()
    route = str(data.get("route") or "unknown")
    intent = str(data.get("intent") or "unknown")

    if is_basic_chat(query) and answer in DEFAULT_EMPTY_ANSWERS:
        answer = basic_chat_answer(query)
        route = "chat"
        intent = "chat"
        data.setdefault("trace_log", [])
        data["trace_log"] = list(data.get("trace_log") or []) + [
            {"node": "chat", "data": {"fallback": "basic_chat"}}
        ]

    if not answer:
        answer = "No answer was generated. Try rephrasing the question with the policy topic and employee grade."

    sources = serialise_sources(data.get("sources"), data.get("documents"))
    metadata = TurnMetadata(
        query=query,
        route=route,
        intent=intent,
        answer_characters=len(answer),
        sources=sources,
    )

    data.update(
        {
            "answer": answer,
            "route": metadata.route,
            "intent": metadata.intent,
            "sources": metadata.sources,
            "turn_metadata": metadata.model_dump(),
        }
    )
    return data


def fallback_result(query: str, error: Optional[Exception] = None) -> Dict[str, Any]:
    if is_basic_chat(query):
        answer = basic_chat_answer(query)
        route = "chat"
        intent = "chat"
    else:
        answer = (
            "The graph engine was not available for this run. Install the repo "
            "requirements, build the policy index, and run this file from the "
            "repository root to enable full RAG answers."
        )
        route = "fallback"
        intent = "rag"

    return {
        "answer": answer,
        "route": route,
        "intent": intent,
        "sources": [],
        "trace_log": [
            {"node": "trace_start", "data": {"fallback": True}},
            {"node": route if route in NODE_BY_NAME else "hitl", "data": {"error": str(error or "")}},
            {"node": "trace_end", "data": {}},
        ],
        "error": str(error or ""),
    }


def run_aegis_turn(
    query: str,
    *,
    employee_grade: str = "L3",
    history: Optional[List[Dict[str, str]]] = None,
    memory_context: str = "",
    graph: Any = None,
) -> Dict[str, Any]:
    bootstrap_repo()
    configure_runtime_defaults()

    try:
        from app.core.stability_patch import safe_invoke
        from app.graph.workflow import build_graph

        active_graph = graph or build_graph()
        result = safe_invoke(
            active_graph,
            {
                "query": query,
                "history": history or [],
                "memory_context": memory_context,
                "trace_log": [],
                "employee_grade": employee_grade,
            },
        )
    except Exception as exc:
        result = fallback_result(query, exc)

    return normalise_result(result, query)


def ensure_policy_index() -> Dict[str, Any]:
    bootstrap_repo()
    configure_runtime_defaults()
    try:
        from app.core.vector_store import ensure_vectorstore_ready

        count = ensure_vectorstore_ready(auto_ingest=True)
        return {"ready": True, "chunks": count, "error": ""}
    except Exception as exc:
        return {"ready": False, "chunks": 0, "error": str(exc)}


def draw_workflow_chart(active_nodes: Optional[Sequence[str]] = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    active = set(active_nodes or [])
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 1.02)
    ax.axis("off")

    for source, target, label in WORKFLOW_EDGES:
        x1, y1 = NODE_POSITIONS[source]
        x2, y2 = NODE_POSITIONS[target]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=EDGE_COLOR,
            alpha=0.74,
            connectionstyle="arc3,rad=0.06",
        )
        ax.add_patch(arrow)
        if label:
            lx = (x1 + x2) / 2
            ly = (y1 + y2) / 2
            ax.text(lx, ly + 0.018, label, fontsize=7.5, color="#4b5563", ha="center")

    box_w = 0.13
    box_h = 0.064
    for node in NODE_SPECS:
        x, y = NODE_POSITIONS[node.name]
        is_active = node.name in active
        fill = ACTIVE_COLOR if is_active else GROUP_COLORS.get(node.group, "#eef0f3")
        edge = "#173ea5" if is_active else "#64748b"
        text_color = "white" if is_active else TEXT_COLOR
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.25,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y + 0.006,
            node.title,
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            x,
            y - 0.018,
            node.group,
            ha="center",
            va="center",
            fontsize=6.8,
            color=text_color,
            alpha=0.88,
        )

    ax.set_title("AEGIS LangGraph Node Flow", loc="left", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout()
    return fig


def draw_retrieval_chart(result: Dict[str, Any]):
    import matplotlib.pyplot as plt

    documents = result.get("documents") or []
    if not documents:
        return None

    labels = []
    lengths = []
    scores = []
    for index, document in enumerate(documents[:8], start=1):
        if not isinstance(document, dict):
            continue
        labels.append(str(index))
        lengths.append(len(str(document.get("content") or "")))
        scores.append(float(document.get("rerank_score") or 0.0))

    if not lengths:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(labels, lengths, color="#2a9d8f", label="chunk chars")
    ax.set_xlabel("Retrieved chunk")
    ax.set_ylabel("Characters")
    ax.set_title("Retrieved Chunk Lengths")

    if any(scores):
        twin = ax.twinx()
        twin.plot(labels, scores, color="#e76f51", marker="o", linewidth=2, label="rerank")
        twin.set_ylabel("Rerank score")

    fig.tight_layout()
    return fig


def render_streamlit_app() -> None:
    import pandas as pd
    import streamlit as st

    bootstrap_repo()
    configure_runtime_defaults()

    st.set_page_config(page_title=APP_TITLE, page_icon=":shield:", layout="wide")

    @st.cache_resource(show_spinner=False)
    def cached_graph():
        from app.graph.workflow import build_graph

        return build_graph()

    @st.cache_resource(show_spinner=False)
    def cached_index_status():
        return ensure_policy_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "traces" not in st.session_state:
        st.session_state.traces = []
    if "memory_context" not in st.session_state:
        st.session_state.memory_context = ""

    index_status = cached_index_status()

    with st.sidebar:
        st.title("AEGIS Control")
        employee_grade = st.selectbox("Access Level", ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "VP", "SVP", "CXO"], index=2)
        use_graph = st.toggle("Use Graph Engine", True)
        show_trace = st.toggle("Show Node Trace", True)
        st.caption(f"Policy chunks indexed: {index_status.get('chunks', 0)}")
        if index_status.get("error"):
            st.warning(index_status["error"])
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.traces = []
            st.session_state.memory_context = ""
            st.rerun()

    chat_col, chart_col = st.columns([0.58, 0.42], gap="large")

    with chat_col:
        st.title("AEGIS Policy Chat")
        st.caption("Ask a policy question, a basic greeting, or a quick calculation.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Ask a policy question...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Running AEGIS..."):
                    graph = None
                    graph_error = None
                    if use_graph:
                        try:
                            graph = cached_graph()
                        except Exception as exc:
                            graph_error = exc

                    if graph_error:
                        result = normalise_result(fallback_result(query, graph_error), query)
                    else:
                        result = run_aegis_turn(
                            query,
                            employee_grade=employee_grade,
                            history=st.session_state.messages[-8:],
                            memory_context=st.session_state.memory_context,
                            graph=graph,
                        )

                placeholder = st.empty()
                answer = result["answer"]
                visible = ""
                for word in answer.split(" "):
                    visible += word + " "
                    placeholder.markdown(visible + "|")
                    time.sleep(0.006)
                placeholder.markdown(answer)

                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"][:5]))

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.session_state.traces.append(result)

    latest = st.session_state.traces[-1] if st.session_state.traces else {}
    active_nodes = trace_nodes(latest.get("trace_log", []))

    with chart_col:
        st.subheader("Node Chart")
        st.pyplot(draw_workflow_chart(active_nodes), clear_figure=True)

        doc_chart = draw_retrieval_chart(latest)
        if doc_chart is not None:
            st.pyplot(doc_chart, clear_figure=True)

        if show_trace:
            rows = node_rows(latest.get("trace_log", []))
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if latest.get("turn_metadata"):
            st.json(latest["turn_metadata"], expanded=False)


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_cli_chat(employee_grade: str = "L3") -> None:
    history: List[Dict[str, str]] = []
    print("AEGIS chat is ready. Type 'exit' to stop.")
    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        result = run_aegis_turn(query, employee_grade=employee_grade, history=history)
        print(f"AEGIS: {result['answer']}\n")
        history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": result["answer"]},
            ]
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="Run a terminal chat loop.")
    parser.add_argument("--grade", default="L3", help="Employee grade used for routing.")
    args, _ = parser.parse_known_args(argv)

    if args.cli or not running_under_streamlit():
        run_cli_chat(employee_grade=args.grade)
    else:
        render_streamlit_app()


if __name__ == "__main__":
    main()
