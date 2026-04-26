import os
from langchain.chat_models import init_chat_model
from langchain.embeddings import OpenAIEmbeddings

# ==============================
# 🔹 Allowed Models
# ==============================

ALLOWED_CHAT_MODELS = {
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4o-mini",
}

ALLOWED_EMBED_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
}

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🔹 Chat Model (with middleware)
# ==============================

def get_chat_model():
    model_name = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)
    if model_name not in ALLOWED_CHAT_MODELS:
        raise ValueError(f"❌ Unauthorized chat model: {model_name}")

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Attach middleware safely
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
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if model_name not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model_name}")

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
    # Optional: persist to file/db

# ==============================
# 🔹 Pipeline Helpers
# ==============================

def expand_query(user_query: str) -> list[str]:
    return [user_query, f"Explain {user_query} in simple terms"]

def hyde(user_query: str, enable: bool = False) -> str:
    if not enable:
        return ""
    return f"Hypothetical doc for: {user_query}"

def retrieve(query_embedding, top_k: int = 5) -> list[str]:
    # Replace with actual vector DB call
    # Example: results = vector_db.search(query_embedding, top_k=top_k)
    results = []  # placeholder
    return [str(r) for r in results]

def rerank(chunks: list[str], query: str, cutoff: int = 3) -> list[str]:
    # Replace with actual cross-encoder rerank
    # Example: scores = cross_encoder.rank(query, chunks)
    scores = [(chunk, i) for i, chunk in enumerate(chunks)]
    return [chunk for chunk, _ in scores[:cutoff]]

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Answer based only on context:\n{context}\n\nQ: {query}\nA:"
    chat_model = get_chat_model()
    resp = chat_model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

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
# 🔹 Example Usage
# ==============================

if __name__ == "__main__":
    query = "Explain quantum entanglement"
    print(run_pipeline(query))
