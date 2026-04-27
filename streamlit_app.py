import os
import sys
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
load_dotenv()

# ==============================
# 🔹 IMPORT YOUR SYSTEM
# ==============================
from app.graph.workflow import build_graph
from app.core.stability_patch import safe_invoke   # ✅ CRITICAL

# ==============================
# 🔹 LOAD GRAPH (CACHED)
# ==============================
@st.cache_resource
def load_graph():
    return build_graph()

# ==============================
# 🔹 DEFINE APP (🔥 YOUR MISSING PIECE)
# ==============================
def app(query: str):
    graph = load_graph()

    result = safe_invoke(graph, {
        "query": query,
        "memory_context": [],
        "history": [],
        "trace_log": []
    })

    return result

# ==============================
# 🔹 UI CONFIG
# ==============================
st.set_page_config(page_title="Aegis RAG System", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .main-header { font-size: 2.5rem; color: #2E4053; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🛡️ Aegis RAG System</h1>", unsafe_allow_html=True)

# ==============================
# 🔹 SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# 🔹 DISPLAY CHAT
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# 🔹 CHAT INPUT
# ==============================
if query := st.chat_input("Ask Aegis anything..."):

    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing..."):

                response = app(query)
                answer = response.get("answer", "No answer generated.")

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"System Error: {str(e)}")
