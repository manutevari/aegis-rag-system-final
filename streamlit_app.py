```python
import os, sys, asyncio, time

# ✅ FIXED: deterministic path resolution
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

os.chdir(ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

import streamlit as st

# Backend
from app.core.models import get_embed_model
from app.graph.workflow import build_graph
from app.utils.pickle_cache import PickleCache
from app.utils.encryption import encrypt, decrypt


# -------------------------
# Async Fix
# -------------------------
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# -------------------------
# Streaming Output
# -------------------------
def stream_text(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.015)


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Policy RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.source-tag{background:#e8f4fd;color:#1a73e8;padding:2px 8px;border-radius:12px;font-size:12px;margin-right:4px}
.ok-tag{background:#d4edda;color:#155724;padding:2px 8px;border-radius:12px;font-size:12px}
.fail-tag{background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:12px;font-size:12px}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Session State
# -------------------------
if "messages" not in st.session_state: st.session_state.messages = []
if "traces" not in st.session_state: st.session_state.traces = []
if "cache" not in st.session_state: st.session_state.cache = PickleCache()


@st.cache_resource
def load_graph():
    return build_graph()


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("🏢 Decision-Grade RAG")
    st.caption("LangGraph + Transparent AI")

    st.divider()

    use_cache = st.toggle("Cache answers", True)
    show_context = st.toggle("Show Retrieved Context", False)

    hitl_mode = st.selectbox("HITL Mode", ["auto","queue","cli"])
    os.environ["HITL_MODE"] = hitl_mode

    grade_override = st.text_input("Employee grade", placeholder="L4, VP")

    st.divider()

    stats = st.session_state.cache.stats()
    st.metric("Cached entries", stats["entries"])
    st.metric("Cache size", f"{stats['size_mb']} MB")

    if st.button("🗑️ Clear cache"):
        st.success(f"Cleared {st.session_state.cache.clear()} entries")


# -------------------------
# Layout
# -------------------------
col_chat, col_trace = st.columns([3, 2])


# -------------------------
# Chat Section
# -------------------------
with col_chat:
    st.header("💬 Chat")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

            if m.get("sources"):
                st.markdown(
                    " ".join(f'<span class="source-tag">{s}</span>' for s in m["sources"]),
                    unsafe_allow_html=True
                )

            if "verified" in m:
                cls = "ok-tag" if m["verified"] else "fail-tag"
                lbl = "✓ Verified" if m["verified"] else "⚠ Unverified"
                st.markdown(f'<span class="{cls}">{lbl}</span>', unsafe_allow_html=True)


    query = st.chat_input("Ask about any corporate policy…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing..."):
                start = time.time()

                cached = st.session_state.cache.get(query) if use_cache else None

                if cached:
                    answer = decrypt(cached)
                    sources, verified, route, trace_steps = [], True, "cache", []
                else:
                    graph = load_graph()

                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-8:]
                    ]

                    state = {
                        "query": query,
                        "history": history,
                        "trace_log": [],
                        "retry_count": 0
                    }

                    if grade_override:
                        state["employee_grade"] = grade_override.strip().upper()

                    result = asyncio.run(graph.ainvoke(state))

                    answer = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    verified = result.get("verified", False)
                    route = result.get("route", "?")
                    trace_steps = result.get("trace_log", [])

                    if use_cache and answer:
                        st.session_state.cache.set(query, encrypt(answer))

                    st.session_state.traces.append({
                        "query": query,
                        "route": route,
                        "verified": verified,
                        "steps": trace_steps
                    })

                latency = round(time.time() - start, 2)

            # -------------------------
            # STREAMING ANSWER
            # -------------------------
            st.write_stream(stream_text(answer))

            # -------------------------
            # CONFIDENCE SCORE
            # -------------------------
            confidence = 0.0
            if verified:
                confidence += 0.5
            if sources:
                confidence += min(0.5, len(sources) * 0.1)

            confidence = min(confidence, 1.0)

            if confidence >= 0.8:
                st.success(f"Confidence: {confidence:.2f}")
            elif confidence >= 0.5:
                st.warning(f"Confidence: {confidence:.2f}")
            else:
                st.error(f"Low Confidence: {confidence:.2f}")

            # -------------------------
            # SOURCES
            # -------------------------
            if sources:
                st.markdown(
                    " ".join(f'<span class="source-tag">{s}</span>' for s in sources),
                    unsafe_allow_html=True
                )

            # -------------------------
            # CONTEXT VIEWER
            # -------------------------
            if show_context and trace_steps:
                with st.expander("🔍 Retrieved Context"):
                    for step in trace_steps:
                        if step.get("node") == "retrieval":
                            docs = step.get("data", {}).get("documents", [])
                            for i, d in enumerate(docs[:5], 1):
                                st.markdown(f"**Doc {i}:**")
                                st.write(str(d)[:500])
                                st.divider()

            # -------------------------
            # META INFO
            # -------------------------
            cls = "ok-tag" if verified else "fail-tag"
            lbl = "✓ Verified" if verified else "⚠ Unverified"

            st.markdown(
                f'<span class="{cls}">{lbl}</span> · route: **{route}**',
                unsafe_allow_html=True
            )

            st.caption(f"⏱️ Response time: {latency}s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "verified": verified
        })


# -------------------------
# TRACE PANEL
# -------------------------
with col_trace:
    st.header("🔍 Execution Trace")

    if not st.session_state.traces:
        st.info("Run a query to see trace")
    else:
        latest = st.session_state.traces[-1]

        st.caption(
            f"Route: **{latest.get('route')}** · Verified: **{latest.get('verified')}**"
        )

        ICONS = {
            "planner":"🧭","retrieval":"📄","sql":"🗄️","compute":"🔢",
            "context_assembler":"📋","generate":"✨","verify":"✅"
        }

        for step in latest.get("steps", []):
            node = step.get("node", "?")
            with st.expander(f"{ICONS.get(node,'⚙️')} {node}"):
                data = step.get("data", {})

                if isinstance(data, dict):
                    trimmed = {
                        k: (str(v)[:500] + "...") if len(str(v)) > 500 else v
                        for k, v in data.items()
                    }
                    st.json(trimmed)
                else:
                    st.write(str(data)[:500])
```
