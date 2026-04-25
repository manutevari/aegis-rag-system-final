"""
Streamlit UI — Decision-Grade RAG Chatbot
Features: chat history · execution trace viewer · HITL review panel · cache stats
"""
import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dotenv import load_dotenv; load_dotenv()

import streamlit as st
from app.core.models import get_chat_model, get_embed_model

CHAT_MODEL = get_chat_model()
EMBED_MODEL = get_embed_model()

# Fail fast if invalid
CHAT_MODEL = get_chat_model()
EMBED_MODEL = get_embed_model()
from app.graph.workflow import build_graph
from app.utils.pickle_cache import PickleCache
from app.utils.encryption import encrypt, decrypt

st.set_page_config(page_title="Policy RAG", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.source-tag{background:#e8f4fd;color:#1a73e8;padding:2px 8px;border-radius:12px;font-size:12px;margin-right:4px}
.ok-tag{background:#d4edda;color:#155724;padding:2px 8px;border-radius:12px;font-size:12px}
.fail-tag{background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:12px;font-size:12px}
</style>""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "traces"   not in st.session_state: st.session_state.traces   = []
if "cache"    not in st.session_state: st.session_state.cache    = PickleCache()

@st.cache_resource
def load_graph(): return build_graph()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 Decision-Grade RAG")
    st.caption("Corporate Policy Assistant · LangGraph backbone")
    st.divider()
    use_cache = st.toggle("Cache answers", True)
    hitl_mode = st.selectbox("HITL Mode", ["auto","queue","cli"])
    os.environ["HITL_MODE"] = hitl_mode
    grade_override = st.text_input("Employee grade (optional)", placeholder="e.g. L4, VP")
    st.divider()
    stats = st.session_state.cache.stats()
    st.metric("Cached entries", stats["entries"])
    st.metric("Cache size", f"{stats['size_mb']} MB")
    if st.button("🗑️ Clear cache"): st.success(f"Cleared {st.session_state.cache.clear()} entries")
    st.divider()
    st.subheader("💡 Example queries")
    EXAMPLES = [
        "Hotel limit for L5 on domestic travel?",
        "Calculate VP trip cost: 3 nights domestic",
        "International per diem for L4 grade?",
        "How is EL encashment calculated?",
        "What laptop budget does L6 get?",
        "Approval needed for ₹3 lakh expense claim?",
    ]
    for ex in EXAMPLES:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["_prefill"] = ex

col_chat, col_trace = st.columns([3, 2])

# ── Chat ─────────────────────────────────────────────────────────────────────
with col_chat:
    st.header("💬 Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.markdown(" ".join(f'<span class="source-tag">{s}</span>' for s in m["sources"]), unsafe_allow_html=True)
            if "verified" in m:
                cls, lbl = ("ok-tag","✓ Verified") if m["verified"] else ("fail-tag","⚠ Unverified")
                st.markdown(f'<span class="{cls}">{lbl}</span>', unsafe_allow_html=True)

    prefill = st.session_state.pop("_prefill", "")
    query   = st.chat_input("Ask about any corporate policy…") or prefill
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching policies…"):
                cached_bytes = st.session_state.cache.get(query) if use_cache else None
                if cached_bytes:
                    answer = decrypt(cached_bytes)
                    sources, verified, route, trace_steps = [], True, "cache", []
                else:
                    graph = load_graph()
                    history = [{"role":m["role"],"content":m["content"]} for m in st.session_state.messages[-8:]]
                    init_state = {"query":query,"history":history,"trace_log":[],"retry_count":0}
                    if grade_override: init_state["employee_grade"] = grade_override.strip().upper()
                    result      = asyncio.run(graph.ainvoke(init_state))
                    answer      = result.get("answer","No answer generated.")
                    sources     = result.get("sources",[])
                    verified    = result.get("verified",False)
                    route       = result.get("route","?")
                    trace_steps = result.get("trace_log",[])
                    if use_cache and answer: st.session_state.cache.set(query, encrypt(answer))
                    st.session_state.traces.append({"query":query,"route":route,"verified":verified,"steps":trace_steps})
            st.markdown(answer)
            if sources: st.markdown(" ".join(f'<span class="source-tag">{s}</span>' for s in sources), unsafe_allow_html=True)
            cls,lbl = ("ok-tag","✓ Verified") if verified else ("fail-tag","⚠ Unverified")
            st.markdown(f'<span class="{cls}">{lbl}</span> · route: **{route}**', unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources,"verified":verified})

# ── Trace viewer ─────────────────────────────────────────────────────────────
with col_trace:
    st.header("🔍 Execution Trace")
    if not st.session_state.traces:
        st.info("Run a query to see the trace here.")
    else:
        latest = st.session_state.traces[-1]
        st.caption(f"Route: **{latest.get('route','?')}** · Verified: **{latest.get('verified','?')}**")
        ICONS = {"planner":"🧭","retrieval":"📄","sql":"🗄️","compute":"🔢","context_assembler":"📋",
                 "token_check":"⚖️","summarize_context":"📝","generate":"✨","verify":"✅",
                 "hitl":"👤","encrypt":"🔐","decrypt":"🔓","trace":"📊"}
        for step in latest.get("steps",[]):
            n = step.get("node","?")
            with st.expander(f"{ICONS.get(n,'⚙️')} {n}", expanded=False):
                st.json(step.get("data",{}))
