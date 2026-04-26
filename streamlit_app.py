import os
import sys
import asyncio
import time
import streamlit as st
from dotenv import load_dotenv

# ==============================================================================
# 🔹 PATH & ENVIRONMENT CONFIGURATION
# ==============================================================================

# Ensuring the script can find local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(CURRENT_DIR)
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# Custom Imports
from app.graph.workflow import build_graph
from app.utils.pickle_cache import PickleCache
from app.utils.encryption import encrypt, decrypt
from app.memory.memory_manager import MemoryManager

# ==============================================================================
# 🔹 APP CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="AEGIS Policy Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat aesthetics
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stChatFloatingInputContainer { bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🔹 ASYNC EXECUTOR & RESOURCE LOADING
# ==============================================================================

def run_graph(graph, state):
    """Executes the LangGraph workflow in a dedicated event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(graph.ainvoke(state))
    finally:
        loop.close()

@st.cache_resource
def load_graph_instance():
    return build_graph()

# ==============================================================================
# 🔹 SESSION STATE MANAGEMENT
# ==============================================================================

def init_session_state():
    defaults = {
        "messages": [],
        "traces": [],
        "cache": PickleCache(),
        "memory": MemoryManager(),
        "_prefill": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# 🔹 SIDEBAR (CONTROL PANEL)
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")
    
    with st.container(border=True):
        st.subheader("Global Settings")
        use_cache = st.toggle("Enable Cache", value=True)
        debug_mode = st.toggle("Debug Mode", value=True)
        grade_override = st.text_input("Employee Grade Override", placeholder="e.g. L5")

    st.divider()

    # Cache Metrics
    st.subheader("📊 System Stats")
    stats = st.session_state.cache.stats()
    col1, col2 = st.columns(2)
    col1.metric("Entries", stats["entries"])
    col2.metric("Size", f"{stats['size_mb']} MB")
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.cache.clear()
        st.rerun()

    st.divider()

    # Example Queries
    st.subheader("💡 Example Queries")
    EXAMPLES = [
        "Hotel limit for L5 on domestic travel?",
        "International per diem for L4 grade?",
        "Approval needed for ₹3 lakh expense claim?"
    ]
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"btn_{ex}"):
            st.session_state["_prefill"] = ex
            st.rerun()

# ==============================================================================
# 🔹 MAIN UI LAYOUT
# ==============================================================================

col_chat, col_trace = st.columns([1.8, 1.2], gap="medium")

# ------------------------------------------------------------------------------
# 💬 CHAT PANEL
# ------------------------------------------------------------------------------
with col_chat:
    st.title("💬 Policy Assistant")
    st.caption("AI-powered enterprise policy intelligence")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Input
    user_input = st.chat_input("Ask a policy question...")
    query = user_input or st.session_state.pop("_prefill", None)

    if query:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            # Using st.status for a more professional "Thinking" experience
            with st.status("🔍 Processing Intelligence...", expanded=False) as status:
                try:
                    # 1. Cache Lookup
                    cached_data = st.session_state.cache.get(query) if use_cache else None

                    if cached_data:
                        status.update(label="✅ Retrieved from Cache", state="complete")
                        result = {
                            "answer": decrypt(cached_data),
                            "route": "cache_hit",
                            "verified": True,
                            "sources": [],
                            "trace_log": [{"node": "cache_lookup", "data": {"status": "hit"}}]
                        }
                    else:
                        # 2. Graph Execution
                        status.update(label="🧠 Analyzing Policy Graph...")
                        graph = load_graph_instance()
                        
                        state = {
                            "query": query,
                            "history": st.session_state.messages[-6:],
                            "trace_log": []
                        }
                        if grade_override:
                            state["employee_grade"] = grade_override.upper()

                        result = run_graph(graph, state)

                        if use_cache:
                            st.session_state.cache.set(query, encrypt(result["answer"]))
                        
                        status.update(label="✨ Analysis Complete", state="complete")

                    # 3. Handle Streaming Result
                    placeholder = st.empty()
                    answer = result.get("answer", "")
                    full_response = ""
                    
                    # Simulation of word streaming
                    for word in answer.split():
                        full_response += word + " "
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
                    placeholder.markdown(full_response)

                    # Update session data
                    st.session_state.traces.append(result)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })

                except Exception as e:
                    status.update(label="❌ Error Encountered", state="error")
                    st.error(f"System Error: {str(e)}")

# ------------------------------------------------------------------------------
# 🔍 TRACE PANEL
# ------------------------------------------------------------------------------
with col_trace:
    st.title("🔍 Execution Trace")
    
    if not st.session_state.traces:
        st.info("Ask a question to see the logic flow behind the answer.")
    else:
        latest = st.session_state.traces[-1]
        
        # Header Metrics
        t_col1, t_col2 = st.columns(2)
        t_col1.markdown(f"**Route:** `{latest.get('route', 'N/A')}`")
        t_col2.markdown(f"**Verified:** {'✅' if latest.get('verified') else '⚠️'}")
        
        st.divider()

        # Step-by-Step Logs
        for step in latest.get("trace_log", []):
            node_name = step.get("node", "Unknown Node").replace("_", " ").title()
            with st.expander(f"⚙️ {node_name}", expanded=False):
                st.json(step.get("data", {}))

    # Memory Debugger
    if debug_mode:
        st.divider()
        with st.expander("🧠 Active Memory Context", expanded=False):
            st.json(st.session_state.memory.get_context())
