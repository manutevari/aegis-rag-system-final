import os
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.embeddings import OpenAIEmbeddings
import openai

# ==============================
# 🔹 Model Whitelists
# ==============================

OPENAI_CHAT_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"}
OPENAI_EMBED_MODELS = {"text-embedding-3-small", "text-embedding-3-large"}

# For OpenRouter, allow all free trustworthy models
OPENROUTER_FREE_MODELS = {
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "llama-3.3-70b",
    "qwen-3-next-80b",
    "nemotron-3-super-120b",
    "deepseek-r1",
    "gpt-oss-120b"
}

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🔹 Chat Model (with middleware)
# ==============================

def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if provider == "openai":
        if model_name not in OPENAI_CHAT_MODELS:
            raise ValueError(f"❌ Unauthorized OpenAI chat model: {model_name}")
    elif provider == "openrouter":
        if model_name not in OPENROUTER_FREE_MODELS:
            raise ValueError(f"❌ Unauthorized OpenRouter chat model: {model_name}")

    model = init_chat_model(
        model=model_name,
        model_provider=provider,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        from middleware import orchestrator_middleware
        model.middleware = [orchestrator_middleware]
    except Exception:
        pass

    return model

# ==============================
# 🔹 Embedding Model
# ==============================

def get_embed_model():
    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if provider == "openai":
        if model_name not in OPENAI_EMBED_MODELS:
            raise ValueError(f"❌ Unauthorized OpenAI embedding model: {model_name}")
    elif provider == "openrouter":
        # For OpenRouter, allow OpenAI embeddings or other free embeddings
        if model_name not in OPENAI_EMBED_MODELS:
            raise ValueError(f"❌ Unauthorized OpenRouter embedding model: {model_name}")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ==============================
# 🔹 Audit Logging
# ==============================

audit_store = []

def log_audit(stage: str, data: dict):
    audit_store.append({"stage": stage, "data": data})

# ==============================
# 🔹 Pipeline Helpers
# ==============================

def expand_query(user_query: str) -> list[str]:
    return [user_query, f"Explain {user_query} in simple terms"]

def hyde(user_query: str, enable: bool = False) -> str:
    return f"Hypothetical doc for: {user_query}" if enable else ""

# ✅ Retrieval wired to vector_db
from app.core.vector_store import vector_db
def retrieve(query_embedding, top_k: int = 5) -> list[str]:
    results = vector_db.search(query_embedding, top_k=top_k)
    return [getattr(r, "page_content", str(r)) for r in results]

# ✅ Rerank wired to cross_encoder
from app.core.utils import cross_encoder
def rerank(chunks: list[str], query: str, cutoff: int = 3) -> list[str]:
    scores = cross_encoder.rank(query, chunks)
    return [chunk for chunk, _ in scores[:cutoff]]

# ✅ PII Redaction wired with quota-aware fallback
from app.core.utils import redact
def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Answer based only on context:\n{context}\n\nQ: {query}\nA:"
    chat_model = get_chat_model()

    try:
        resp = chat_model.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return redact(raw)

    except openai.error.RateLimitError:
        # Quota exceeded fallback
        return "⚠️ Quota exceeded. Please use one of the free OpenRouter models: " + ", ".join(OPENROUTER_FREE_MODELS)

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ==============================
# 🔹 Pipeline Runner
# ==============================

def run_pipeline(user_query: str):
    log_audit("start", {"query": user_query})

    queries = expand_query(user_query)
    log_audit("expand_query", {"queries": queries})

    hypo_doc = hyde(user_query, enable=False)
    log_audit("hyde", {"hypo_doc": hypo_doc})

    embed_model = get_embed_model()
    chunks = []
    for q in queries:
        q_emb = embed_model.embed_query(q)
        chunks.extend(retrieve(q_emb, top_k=5))
    log_audit("retriever", {"chunks": chunks})

    top_chunks = rerank(chunks, user_query, cutoff=3)
    log_audit("rerank", {"top_chunks": top_chunks})

    answer = generate_answer(user_query, top_chunks)
    log_audit("generate_answer", {"answer": answer})

    return answer

# ==============================
# 🔹 Streamlit Frontend
# ==============================

if __name__ == "__main__":
    st.title("Aegis RAG System")
    query = st.text_input("Ask a question:")
    if query:
        st.write(run_pipeline(query))
