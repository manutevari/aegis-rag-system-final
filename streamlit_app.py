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

# ✅ PATCH IMPORT
from app.core.stability_patch import safe_invoke

# ==============================
# 🔹 APP CONFIG
# ==============================
st.set_page_config(
    page_title="AEGIS Policy Intelligence",
    page_icon="🏢",
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
    st.title("⚙️ Control Panel")

    use_cache = st.toggle("Enable Cache", True)
    debug_mode = st.toggle("Debug Mode", True)
    grade_override = st.text_input("Employee Grade Override")

    if st.button("🗑️ Clear Cache"):
        st.session_state.cache.clear()
        st.rerun()

# ==============================
# 🔹 MAIN UI
# ==============================
st.title("💬 Policy Assistant")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# 🔹 NEW STABLE CHAT ENTRYPOINT
# ==============================
query = st.chat_input("Ask anything")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                # ==============================
                # 🔹 CACHE CHECK
                # ==============================
                cached = st.session_state.cache.get(query) if use_cache else None

                if cached:
                    result = {
                        "answer": decrypt(cached),
                        "route": "cache_hit",
                        "trace_log": []
                    }

                else:
                    graph = load_graph()

                    # ✅ SAFE INVOKE (CORE FIX)
                    result = safe_invoke(graph, {
                        "query": query,
                        "memory_context": st.session_state.memory.get_context(),
                        "history": st.session_state.messages[-6:],
                        "trace_log": [],
                        "employee_grade": grade_override.upper() if grade_override else None
                    })

                    if use_cache and result.get("answer"):
                        st.session_state.cache.set(query, encrypt(result["answer"]))

                # ==============================
                # 🔹 STREAM OUTPUT
                # ==============================
                answer = result.get("answer", "No response")
                full = ""

                placeholder = st.empty()

                for word in answer.split():
                    full += word + " "
                    placeholder.markdown(full + "▌")
                    time.sleep(0.02)

                placeholder.markdown(full)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full
                })

                st.session_state.traces.append(result)

            except Exception as e:
                st.error(f"System Error: {str(e)}")

# ==============================
# 🔹 DEBUG PANEL
# ==============================
if debug_mode and st.session_state.traces:
    st.divider()
    st.subheader("🔍 Debug Trace")

    latest = st.session_state.traces[-1]

    st.json({
        "route": latest.get("route"),
        "trace_log": latest.get("trace_log")
    })

    st.subheader("🧠 Memory")
    st.json(st.session_state.memory.get_context())
