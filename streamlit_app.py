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

from app.core.runtime_config import apply_runtime_model_config, default_model_for_provider, normalize_provider
from app.core.runtime_config import apply_local_runtime_config, normalize_local_provider
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


if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache" not in st.session_state:
    st.session_state.cache = PickleCache()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "traces" not in st.session_state:
    st.session_state.traces = []

_PROVIDER_LABELS = ["Gemini", "OpenAI", "Extractive"]
_PROVIDER_VALUES = {"Gemini": "gemini", "OpenAI": "openai", "Extractive": "extractive"}
_PROVIDER_INDEX = {"gemini": 0, "openai": 1, "extractive": 2}

initial_provider = normalize_provider(os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "gemini")
runtime_model_config = {"provider": initial_provider, "model": default_model_for_provider(initial_provider)}
_PROVIDER_LABELS = ["Local Auto", "Ollama", "llama.cpp", "Mistral Local", "Extractive"]
_PROVIDER_VALUES = {
    "Local Auto": "local_auto",
    "Ollama": "ollama",
    "llama.cpp": "llama_cpp",
    "Mistral Local": "mistral_local",
    "Extractive": "extractive",
}
_PROVIDER_INDEX = {"local_auto": 0, "ollama": 1, "llama_cpp": 2, "mistral_local": 3, "extractive": 4}

with st.sidebar:
    st.title("🛡️ AEGIS Control")
    st.markdown("---")

    provider_label = st.selectbox(
        "Model Provider",
    initial_provider = normalize_local_provider(os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "local_auto")
    provider_label = st.selectbox(
        "Local LLM Runtime",
        _PROVIDER_LABELS,
        index=_PROVIDER_INDEX.get(initial_provider, 0),
    )
    runtime_provider = _PROVIDER_VALUES[provider_label]
    runtime_model = ""
    runtime_api_key = ""

    if runtime_provider != "extractive":
        model_env_name = "GOOGLE_MODEL" if runtime_provider == "gemini" else "OPENAI_MODEL"
        key_label = "Gemini API Key" if runtime_provider == "gemini" else "OpenAI API Key"
        runtime_model = st.text_input(
            "Model",
            value=os.getenv(model_env_name) or default_model_for_provider(runtime_provider),
            key=f"runtime_model_{runtime_provider}",
        )
        runtime_api_key = st.text_input(
            key_label,
            type="password",
            placeholder=f"Paste {key_label}",
            key=f"runtime_api_key_{runtime_provider}",
        )

    runtime_model_config = apply_runtime_model_config(
        runtime_provider,
        api_key=runtime_api_key,
        model=runtime_model,

    local_orchestration_model = st.text_input(
        "Orchestration Model",
        value=os.getenv("LOCAL_ORCHESTRATION_MODEL", "llama3.1"),
    )
    local_generation_model = st.text_input(
        "Generation Model",
        value=os.getenv("LOCAL_GENERATION_MODEL", "mistral"),
    )

    with st.expander("Local Runtime Endpoints", expanded=False):
        ollama_base_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        ollama_model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "mistral"))
        llama_cpp_base_url = st.text_input("llama.cpp URL", value=os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1"))
        llama_cpp_model = st.text_input("llama.cpp Model", value=os.getenv("LLAMA_CPP_MODEL", "local-model"))
        mistral_local_base_url = st.text_input(
            "Mistral Local URL",
            value=os.getenv("MISTRAL_LOCAL_BASE_URL", "http://localhost:8000/v1"),
        )
        mistral_local_model = st.text_input("Mistral Local Model", value=os.getenv("MISTRAL_LOCAL_MODEL", "mistral"))

    runtime_model_config = apply_local_runtime_config(
        runtime_provider,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        llama_cpp_base_url=llama_cpp_base_url,
        llama_cpp_model=llama_cpp_model,
        mistral_local_base_url=mistral_local_base_url,
        mistral_local_model=mistral_local_model,
        local_orchestration_model=local_orchestration_model,
        local_generation_model=local_generation_model,
    )

    use_cache = st.toggle("Enable Secure Cache", True)
    debug_mode = st.toggle("Show System Trace", True)

    index_error = None
    try:
        indexed_chunks = ensure_policy_index()
    except Exception as exc:
        indexed_chunks = 0
        index_error = str(exc)

    st.caption(f"Policy chunks indexed: {indexed_chunks}")
    st.caption(f"Runtime: {runtime_model_config['provider']}")
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
                runtime_provider = str(runtime_model_config.get("provider") or "default")
                runtime_model = str(runtime_model_config.get("model") or "default")
                cache_key = f"{grade_override}_{runtime_provider}_{runtime_model}_{query}"
                cache_key = f"{grade_override}_{runtime_model_config['provider']}_{query}"
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
                            "model": None if runtime_provider == "extractive" else runtime_model,
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
