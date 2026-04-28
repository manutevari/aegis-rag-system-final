import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# ==============================
# 🔹 PATH SETUP
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(CURRENT_DIR)
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# ==============================
# 🔹 IMPORTS
# ==============================
from app.graph.workflow import build_graph
from app.utils.pickle_cache import PickleCache
from app.utils.encryption import encrypt, decrypt
from app.memory.memory_manager import MemoryManager
from app.core.stability_patch import safe_invoke

# ==============================
# 🔹 APP CONFIG
# ==============================
st.set_page_config(
    page_title="AEGIS Policy Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ==============================
# 🔹 LOAD GRAPH
# ==============================
@st.cache_resource
def load_graph():
    return build_graph()

# ==============================
# 🔹 SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache" not in st.session_state:
    st.session_state.cache = PickleCache()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "traces" not in st.session_state:
    st.session_state.traces = []

# ==============================
# 🔹 SIDEBAR
# ==============================
with st.sidebar:
    st.title("🛡️ AEGIS Control")
    st.markdown("---")
    use_cache = st.toggle("Enable Secure Cache", True)
    debug_mode = st.toggle("Show System Trace", True)
    
    # Grade determines document access (Critical for RAG filtering)
    grade_override = st.selectbox(
        "Access Level (Grade)", 
        ["L1", "L2", "L3", "L4", "Executive"], 
        index=2
    )

    if st.button("🗑️ Clear Cache"):
        st.session_state.cache.clear()
        st.success("Cache Purged")
        st.rerun()

# ==============================
# 🔹 MAIN UI
# ==============================
st.title("🏢 Policy Assistant")
st.caption("Cross-linked intelligence for Travel, Fuel, and HR Policies.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# 🔹 ENTRYPOINT
# ==============================
query = st.chat_input("Ask a policy question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Knowledge Base..."):
            try:
                # FIX: Cache key must include the grade to prevent cross-access leaks
                cache_key = f"{grade_override}_{query}"
                cached = st.session_state.cache.get(cache_key) if use_cache else None

                if cached:
                    result = {
                        "answer": decrypt(cached),
                        "route": "cache_hit",
                        "trace_log": ["Retrieved from encrypted cache"]
                    }
                else:
                    graph = load_graph()
                    # SAFE INVOKE against the RAG Pipeline
                    result = safe_invoke(graph, {
                        "query": query,
                        "memory_context": st.session_state.memory.get_context(),
                        "history": st.session_state.messages[-6:],
                        "trace_log": [],
                        "employee_grade": grade_override
                    })

                    if use_cache and result.get("answer"):
                        st.session_state.cache.set(cache_key, encrypt(result["answer"]))

                # ==============================
                # 🔹 FORMATTED STREAMING
                # ==============================
                answer = result.get("answer", "I'm sorry, I couldn't find relevant information.")
                placeholder = st.empty()
                full_text = ""
                
                # FIX: Use a character or line-based stream to preserve Markdown formatting/tables
                for chunk in answer.split(" "):
                    full_text += chunk + " "
                    placeholder.markdown(full_text + "▌")
                    time.sleep(0.01)
                placeholder.markdown(full_text)

                st.session_state.messages.append({"role": "assistant", "content": full_text})
                st.session_state.traces.append(result)

            except Exception as e:
                st.error(f"Engine Error: {str(e)}")

# ==============================
# 🔹 DEBUG TRACE
# ==============================
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
        
        st.write("**Processing Steps:**")
        for log in latest.get("trace_log", []):
            st.text(f"→ {log}")
