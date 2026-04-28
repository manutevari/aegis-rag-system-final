import os
import sys
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(CURRENT_DIR)
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from app.core.stability_patch import safe_invoke
from app.graph.workflow import build_graph
from app.memory.memory_manager import MemoryManager
from app.utils.encryption import decrypt, encrypt
from app.utils.pickle_cache import PickleCache

st.set_page_config(
    page_title="AEGIS Policy Intelligence",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def ensure_policy_index():
    from app.core.vector_store import ensure_vectorstore_ready

    return ensure_vectorstore_ready(auto_ingest=True)


@st.cache_resource
def load_graph():
    return build_graph()


index_error = None
try:
    indexed_chunks = ensure_policy_index()
except Exception as exc:
    indexed_chunks = 0
    index_error = str(exc)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache" not in st.session_state:
    st.session_state.cache = PickleCache()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "traces" not in st.session_state:
    st.session_state.traces = []

with st.sidebar:
    st.title("🛡️ AEGIS Control")
    st.markdown("---")
    use_cache = st.toggle("Enable Secure Cache", True)
    debug_mode = st.toggle("Show System Trace", True)

    st.caption(f"Policy chunks indexed: {indexed_chunks}")
    if index_error:
        st.warning(f"Policy index not ready: {index_error}")

    grade_override = st.selectbox(
        "Access Level (Grade)",
        ["L1", "L2", "L3", "L4", "Executive"],
        index=2,
    )

    if st.button("🗑️ Clear Cache"):
        st.session_state.cache.clear()
        st.success("Cache Purged")
        st.rerun()

st.title("🏢 Policy Assistant")
st.caption("Cross-linked intelligence for Travel, Fuel, and HR Policies.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a policy question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Knowledge Base..."):
            try:
                cache_key = f"{grade_override}_{query}"
                cached = st.session_state.cache.get(cache_key) if use_cache else None

                if cached:
                    result = {
                        "answer": decrypt(cached),
                        "route": "cache_hit",
                        "intent": "cache_hit",
                        "trace_log": ["Retrieved from encrypted cache"],
                    }
                else:
                    graph = load_graph()
                    result = safe_invoke(
                        graph,
                        {
                            "query": query,
                            "memory_context": st.session_state.memory.get_context(),
                            "history": st.session_state.messages[-6:],
                            "trace_log": [],
                            "employee_grade": grade_override,
                        },
                    )

                    if use_cache and result.get("answer") and result.get("route") != "error":
                        st.session_state.cache.set(cache_key, encrypt(result["answer"]))

                answer = result.get("answer", "I'm sorry, I couldn't find relevant information.")
                placeholder = st.empty()
                full_text = ""

                for chunk in answer.split(" "):
                    full_text += chunk + " "
                    placeholder.markdown(full_text + "▌")
                    time.sleep(0.01)
                placeholder.markdown(full_text)

                st.session_state.messages.append({"role": "assistant", "content": full_text})
                st.session_state.traces.append(result)

                if debug_mode and result.get("error"):
                    with st.expander("Engine Error Details", expanded=False):
                        st.code(result["error"])

            except Exception as e:
                error_trace = traceback.format_exc()
                st.error(f"Engine Error: {str(e)}")
                if debug_mode:
                    st.code(error_trace)
                st.session_state.traces.append(
                    {
                        "route": "error",
                        "intent": "error",
                        "trace_log": [str(e)],
                        "error": error_trace,
                    }
                )

if debug_mode and st.session_state.traces:
    with st.expander("🔍 System Logic Trace", expanded=False):
        latest = st.session_state.traces[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Routing Decision:**")
            st.code(latest.get("route"))
        with col2:
            st.write("**Access Level Used:**")
            st.code(grade_override)

        if latest.get("intent"):
            st.write("**Intent:**")
            st.code(latest.get("intent"))

        st.write("**Processing Steps:**")
        for log in latest.get("trace_log", []):
            st.text(f"→ {log}")

        if latest.get("error"):
            st.write("**Error:**")
            st.code(latest.get("error"))
