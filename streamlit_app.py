import hashlib
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(CURRENT_DIR)
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from app.core.runtime_config import apply_local_runtime_config, normalize_local_provider
from app.core.stability_patch import safe_invoke
from app.core.vector_store import reset_vectorstore_cache
from app.graph.workflow import build_graph
from app.memory.memory_manager import MemoryManager
from app.tools.compute import compute
from app.utils.encryption import decrypt, encrypt
from app.utils.pickle_cache import PickleCache
from policy_ingestion import SUPPORTED_EXTENSIONS, run_ingestion

st.set_page_config(
    page_title="AEGIS Policy Intelligence",
    page_icon="🛡️",
    layout="wide",
)

DATA_DIR = Path(CURRENT_DIR) / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STRICT_NOT_FOUND = "This information is not available in the policy."

_PROVIDER_LABELS = [
    "Auto (sentiment best)",
    "OpenAI",
    "Grok (xAI)",
    "Gemini",
    "Mistral",
    "OpenRouter",
    "Ollama",
    "llama.cpp",
    "Mistral Local",
    "Extractive",
]
_PROVIDER_VALUES = {
    "Auto (sentiment best)": "auto_sentiment",
    "OpenAI": "openai",
    "Grok (xAI)": "grok",
    "Gemini": "gemini",
    "Mistral": "mistral",
    "OpenRouter": "openrouter",
    "Ollama": "ollama",
    "llama.cpp": "llama_cpp",
    "Mistral Local": "mistral_local",
    "Extractive": "extractive",
}
_PROVIDER_INDEX = {value: index for index, value in enumerate(_PROVIDER_VALUES.values())}
_CLOUD_PROVIDERS = {"openai", "grok", "gemini", "mistral", "openrouter"}
_LOCAL_PROVIDERS = {"ollama", "llama_cpp", "mistral_local", "extractive"}

_PROVIDER_MODELS = {
    "auto_sentiment": ["Auto choose"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "custom"],
    "grok": ["grok-4.20-reasoning", "grok-4", "grok-3-mini", "custom"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "custom"],
    "mistral": ["mistral-medium-latest", "mistral-large-latest", "ministral-8b-latest", "custom"],
    "openrouter": ["openrouter/auto", "openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5", "custom"],
    "ollama": ["llama3.1", "llama3", "mistral", "custom"],
    "llama_cpp": ["local-model", "custom"],
    "mistral_local": ["mistral", "custom"],
    "extractive": ["Local retrieval only"],
}

_PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}

_OPENAI_COMPATIBLE_BASE_URLS = {
    "openai": None,
    "grok": "https://api.x.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_POSITIVE_WORDS = {
    "thanks",
    "thank",
    "great",
    "good",
    "happy",
    "excellent",
    "helpful",
    "clear",
    "perfect",
}
_NEGATIVE_WORDS = {
    "angry",
    "bad",
    "confused",
    "denied",
    "frustrated",
    "issue",
    "problem",
    "reject",
    "rejected",
    "urgent",
    "worried",
    "wrong",
}


@st.cache_resource(show_spinner=False)
def ensure_policy_index():
    from app.core.vector_store import ensure_vectorstore_ready

    return ensure_vectorstore_ready(auto_ingest=True)


@st.cache_resource
def load_graph():
    return build_graph()


def _secret_or_env(names: List[str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    try:
        for name in names:
            value = st.secrets.get(name, "")
            if value:
                return value
    except Exception:
        pass
    return ""


def _provider_key(provider: str) -> str:
    session_key = f"{provider}_api_key"
    return st.session_state.get(session_key) or _secret_or_env(_PROVIDER_ENV.get(provider, []))


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).name).strip("_")
    return stem or "policy_document.txt"


def _policy_file_count() -> int:
    if not DATA_DIR.exists():
        return 0
    return sum(
        1
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _payload_fingerprint(files) -> str:
    digest = hashlib.sha256()
    for file in files or []:
        content = file.getvalue()
        digest.update(file.name.encode("utf-8"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def ingest_uploaded_files(uploaded_files) -> Tuple[dict, List[str]]:
    if not uploaded_files:
        return {"status": "empty", "chunks_indexed": 0, "collection_count": 0}, []

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    seen_hashes = set(st.session_state.get("uploaded_hashes", set()))

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash in seen_hashes:
            continue
        safe_name = _safe_filename(uploaded_file.name)
        target = UPLOAD_DIR / f"{file_hash[:12]}_{safe_name}"
        target.write_bytes(content)
        saved_paths.append(str(target))
        seen_hashes.add(file_hash)

    st.session_state.uploaded_hashes = seen_hashes
    if not saved_paths:
        return {"status": "duplicate", "chunks_indexed": 0, "collection_count": ensure_policy_index()}, []

    reset_vectorstore_cache()
    result = run_ingestion(file_paths=saved_paths)
    ensure_policy_index.clear()
    load_graph.clear()
    return result, saved_paths


def analyze_sentiment(text: str) -> Dict[str, object]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    positive = sum(1 for word in words if word in _POSITIVE_WORDS)
    negative = sum(1 for word in words if word in _NEGATIVE_WORDS)
    if negative > positive:
        label = "negative"
        tone = "empathetic, calm, and reassuring"
    elif positive > negative:
        label = "positive"
        tone = "warm, confident, and concise"
    else:
        label = "neutral"
        tone = "clear, professional, and human"
    return {
        "label": label,
        "tone": tone,
        "positive_hits": positive,
        "negative_hits": negative,
    }


def compute_query(query: str) -> Dict[str, object]:
    if not re.search(r"\d", query or ""):
        return {"compute_result": None, "compute_steps": [], "compute_summary": ""}
    return compute({"query": query, "trace_log": []})


def choose_auto_provider(sentiment: Dict[str, object], compute_state: Dict[str, object]) -> str:
    candidates = {
        "negative": ["grok", "openai", "gemini", "openrouter", "mistral"],
        "positive": ["gemini", "openai", "grok", "openrouter", "mistral"],
        "neutral": ["openai", "grok", "gemini", "mistral", "openrouter"],
    }.get(str(sentiment.get("label")), ["openai", "grok", "gemini", "mistral", "openrouter"])

    if compute_state.get("compute_result") is not None:
        candidates = ["openai", "grok", "gemini", "mistral", "openrouter"]

    for provider in candidates:
        if _provider_key(provider):
            return provider
    return "extractive"


def selected_model(provider: str) -> str:
    models = _PROVIDER_MODELS.get(provider, ["custom"])
    choice = st.selectbox("Model", models, key=f"{provider}_model")
    if choice == "custom":
        return st.text_input("Custom model name", key=f"{provider}_custom_model")
    return choice


def render_provider_key(provider: str) -> None:
    if provider not in _CLOUD_PROVIDERS:
        return
    st.text_input(
        "API Key",
        type="password",
        key=f"{provider}_api_key",
        help=f"Uses {' or '.join(_PROVIDER_ENV[provider])} if this field is blank.",
    )


def apply_runtime(provider: str, model: str, local_controls: Dict[str, str]) -> Dict[str, str]:
    if provider == "auto_sentiment" or provider in _CLOUD_PROVIDERS or provider == "extractive":
        provider_for_graph = "extractive" if provider != "auto_sentiment" else "extractive"
    else:
        provider_for_graph = provider

    runtime_model_config = apply_local_runtime_config(
        provider_for_graph,
        ollama_base_url=local_controls["ollama_base_url"],
        ollama_model=local_controls["ollama_model"],
        llama_cpp_base_url=local_controls["llama_cpp_base_url"],
        llama_cpp_model=local_controls["llama_cpp_model"],
        mistral_local_base_url=local_controls["mistral_local_base_url"],
        mistral_local_model=local_controls["mistral_local_model"],
        local_orchestration_model=local_controls["local_orchestration_model"],
        local_generation_model=model if provider in _LOCAL_PROVIDERS and provider != "extractive" else local_controls["local_generation_model"],
    )

    for cloud_provider in _CLOUD_PROVIDERS:
        key = _provider_key(cloud_provider)
        if key:
            for env_name in _PROVIDER_ENV[cloud_provider][:1]:
                os.environ[env_name] = key
    return runtime_model_config


def message_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def call_cloud_llm(provider: str, model: str, prompt: str) -> str:
    api_key = _provider_key(provider)
    if not api_key:
        raise RuntimeError(f"{provider} API key is not configured")

    if provider == "gemini":
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        return "\n".join(part.get("text", "") for part in parts).strip()

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    base_url = _OPENAI_COMPATIBLE_BASE_URLS.get(provider)
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    extra_headers = None
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://github.com/manutevari/aegis-rag-system-final",
            "X-Title": "AEGIS Policy Intelligence",
        }
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        extra_headers=extra_headers,
    )
    return response.choices[0].message.content or ""


def compose_cloud_answer(
    provider: str,
    model: str,
    query: str,
    result: Dict[str, object],
    sentiment: Dict[str, object],
    compute_state: Dict[str, object],
) -> str:
    context = result.get("context") or ""
    draft = result.get("answer") or ""
    compute_summary = compute_state.get("compute_summary") or "No deterministic computation was required."
    prompt = f"""You are the AEGIS policy response composer.

<context>
{context}
</context>

<draft_answer>
{draft}
</draft_answer>

<tool_results>
Sentiment: {sentiment['label']}
Tone: {sentiment['tone']}
Deterministic calculation: {compute_summary}
</tool_results>

Question: {query}

Rules:
- Answer ONLY from context, draft_answer, and verified tool_results.
- If the information is not present, say: "{STRICT_NOT_FOUND}"
- Use a {sentiment['tone']} tone.
- Keep the answer human-like, concise, and well formatted in Markdown.
- Preserve precise numeric values and calculation results exactly.
- Include a short "Reasoning" section that summarizes evidence, not hidden chain-of-thought.
"""
    return call_cloud_llm(provider, model, prompt).strip()


def sources_from_result(result: Dict[str, object]) -> List[str]:
    sources = list(result.get("sources") or [])
    for doc in result.get("documents") or []:
        if isinstance(doc, dict):
            source = doc.get("source") or (doc.get("metadata") or {}).get("source_path")
            if source and source not in sources:
                sources.append(source)
    return sources


def trace_rows(result: Dict[str, object], sentiment: Dict[str, object], provider: str, model: str, compute_state: Dict[str, object]):
    rows = [
        {"node": "sentiment", "status": sentiment["label"], "detail": sentiment["tone"]},
        {"node": "compute_tool", "status": "complete" if compute_state.get("compute_steps") else "skipped", "detail": compute_state.get("compute_summary", "")},
        {"node": "model_router", "status": provider, "detail": model},
    ]
    for item in result.get("trace_log") or []:
        if isinstance(item, dict):
            rows.append({"node": item.get("node", "graph"), "status": "complete", "detail": item.get("data", {})})
        else:
            rows.append({"node": "graph", "status": "log", "detail": str(item)})
    return rows


if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache" not in st.session_state:
    st.session_state.cache = PickleCache()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "traces" not in st.session_state:
    st.session_state.traces = []
if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()

with st.sidebar:
    st.title("🛡️ AEGIS Control")
    st.markdown("---")

    st.subheader("Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload Policy Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one file, many files, or add several batches one by one.",
    )
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected")
    else:
        st.write("0 files selected")

    if st.button("Process Documents", type="primary"):
        try:
            ingest_result, saved_paths = ingest_uploaded_files(uploaded_files)
            if saved_paths:
                st.success(f"Indexed {ingest_result.get('chunks_indexed', 0)} chunks from {len(saved_paths)} uploaded file(s).")
            else:
                st.info("No new files to index.")
        except Exception as exc:
            st.error(f"Document ingestion failed: {exc}")

    st.caption(f"Repo data files available: {_policy_file_count()}")

    st.subheader("Model")
    initial_provider = normalize_local_provider(os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "local_auto")
    initial_index = _PROVIDER_INDEX.get(initial_provider, 0)
    provider_label = st.selectbox("LLM Provider", _PROVIDER_LABELS, index=initial_index)
    selected_provider = _PROVIDER_VALUES[provider_label]

    if selected_provider == "auto_sentiment":
        selected_model_name = "Auto choose"
        with st.expander("Cloud API Keys", expanded=False):
            for cloud_provider in ["openai", "grok", "gemini", "mistral", "openrouter"]:
                st.text_input(
                    f"{cloud_provider.title()} API Key",
                    type="password",
                    key=f"{cloud_provider}_api_key",
                    help=f"Uses {' or '.join(_PROVIDER_ENV[cloud_provider])} if blank.",
                )
    else:
        render_provider_key(selected_provider)
        selected_model_name = selected_model(selected_provider)

    with st.expander("Local Runtime Endpoints", expanded=False):
        local_controls = {
            "local_orchestration_model": st.text_input("Orchestration Model", value=os.getenv("LOCAL_ORCHESTRATION_MODEL", "llama3.1")),
            "local_generation_model": st.text_input("Generation Model", value=os.getenv("LOCAL_GENERATION_MODEL", "mistral")),
            "ollama_base_url": st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            "ollama_model": st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "mistral")),
            "llama_cpp_base_url": st.text_input("llama.cpp URL", value=os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")),
            "llama_cpp_model": st.text_input("llama.cpp Model", value=os.getenv("LLAMA_CPP_MODEL", "local-model")),
            "mistral_local_base_url": st.text_input("Mistral Local URL", value=os.getenv("MISTRAL_LOCAL_BASE_URL", "http://localhost:8000/v1")),
            "mistral_local_model": st.text_input("Mistral Local Model", value=os.getenv("MISTRAL_LOCAL_MODEL", "mistral")),
        }

    runtime_model_config = apply_runtime(selected_provider, selected_model_name, local_controls)

    use_cache = st.toggle("Enable Secure Cache", True)
    debug_mode = st.toggle("Show System Trace", True)

    index_error = None
    try:
        indexed_chunks = ensure_policy_index()
    except Exception as exc:
        indexed_chunks = 0
        index_error = str(exc)

    st.caption(f"Policy chunks indexed: {indexed_chunks}")
    st.caption(f"Graph runtime: {runtime_model_config['provider']}")
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
st.caption("Multi-document policy intelligence with unified retrieval, sentiment-aware responses, and precise calculation tools.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a policy question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running AEGIS tools and nodes..."):
            try:
                sentiment = analyze_sentiment(query)
                compute_state = compute_query(query)
                active_provider = (
                    choose_auto_provider(sentiment, compute_state)
                    if selected_provider == "auto_sentiment"
                    else selected_provider
                )
                active_model = (
                    _PROVIDER_MODELS.get(active_provider, ["Local retrieval only"])[0]
                    if selected_provider == "auto_sentiment"
                    else selected_model_name
                )

                cache_key = f"{grade_override}_{active_provider}_{active_model}_{_payload_fingerprint(uploaded_files)}_{query}"
                cached = st.session_state.cache.get(cache_key) if use_cache else None

                if cached:
                    result = {
                        "answer": decrypt(cached),
                        "route": "cache_hit",
                        "intent": "cache_hit",
                        "trace_log": ["Retrieved from encrypted cache"],
                    }
                    final_answer = result["answer"]
                else:
                    graph = load_graph()
                    initial_state = {
                        "query": query,
                        "memory_context": st.session_state.memory.get_context(),
                        "history": st.session_state.messages[-6:],
                        "trace_log": [],
                        "employee_grade": grade_override,
                        "sentiment_label": sentiment["label"],
                        "sentiment_tone": sentiment["tone"],
                        "compute_result": compute_state.get("compute_result"),
                        "compute_steps": compute_state.get("compute_steps", []),
                        "compute_summary": compute_state.get("compute_summary", ""),
                    }
                    result = safe_invoke(graph, initial_state)
                    final_answer = result.get("answer", "I'm sorry, I couldn't find relevant information.")

                    if active_provider in _CLOUD_PROVIDERS:
                        try:
                            final_answer = compose_cloud_answer(active_provider, active_model, query, result, sentiment, compute_state)
                            result["answer"] = final_answer
                            result["model_provider"] = active_provider
                            result["model"] = active_model
                        except Exception as exc:
                            result.setdefault("trace_log", []).append(f"cloud composer fallback: {exc}")

                    if use_cache and final_answer and result.get("route") != "error":
                        st.session_state.cache.set(cache_key, encrypt(final_answer))

                st.markdown("### Answer")
                placeholder = st.empty()
                full_text = ""
                for chunk in final_answer.split(" "):
                    full_text += chunk + " "
                    placeholder.markdown(full_text + "▌")
                    time.sleep(0.005)
                placeholder.markdown(full_text)

                st.markdown("### 📄 Sources")
                sources = sources_from_result(result)
                if sources:
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.write("- No source metadata returned")

                st.markdown("### Response Context")
                st.info(f"Tone: {sentiment['tone']} | Provider: {active_provider} | Model: {active_model}")
                if compute_state.get("compute_summary"):
                    st.success(compute_state["compute_summary"])

                with st.expander("Tools and Nodes", expanded=debug_mode):
                    st.json(trace_rows(result, sentiment, active_provider, active_model, compute_state))

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
