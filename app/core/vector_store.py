import logging
import os
from typing import List, Any

from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Optional: if you have cross_encoder and trace utilities
# from app.core.utils import cross_encoder
# from app.utils.tracing import trace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==============================
# 🔹 Embedding Model
# ==============================
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ==============================
# 🔹 Vector Store Wrapper
# ==============================
class VectorDB:
    def __init__(self):
        self.embeddings = _get_embeddings()
        self.store: FAISS | None = None

    def load(self, path: str = "vector_store.index"):
        """Load FAISS index from disk."""
        try:
            self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load vector store: %s", e)
            self.store = None

    def save(self, path: str = "vector_store.index"):
        """Persist FAISS index to disk."""
        if self.store:
            self.store.save_local(path)
            logger.info("Vector store saved to %s", path)

    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        """Add new documents to the vector store."""
        if not self.store:
            self.store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.store.add_texts(texts, metadatas=metadatas)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Any]:
        """Search the vector store using a precomputed embedding."""
        if not self.store:
            logger.warning("Vector store is empty, returning []")
            return []
        try:
            results = self.store.similarity_search_by_vector(query_embedding, k=top_k)
            return results
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

# ==============================
# 🔹 Global Instance
# ==============================
vector_db = VectorDB()

# ==============================
# 🔹 One-shot Retrieval Function
# ==============================
def retrieve(query: str, top_k: int = 3):
    """Embed query, search vector DB, and return results."""
    embedder = _get_embeddings()
    q_emb = embedder.embed_query(query)

    results = vector_db.search(q_emb, top_k=top_k)
    docs = [getattr(r, "page_content", str(r)) for r in results]

    # Optional rerank if you have cross_encoder
    # try:
    #     reranked = cross_encoder.rank(query, docs)
    #     docs = [chunk for chunk, _ in reranked[:top_k]]
    # except Exception as e:
    #     logger.warning("Rerank failed: %s", e)

    return docs

# ==============================
# 🔹 Entrypoint
# ==============================
if __name__ == "__main__":
    # Example usage
    vector_db.load()  # load existing FAISS index
    query = "What is the laptop budget for L6 employees?"
    results = retrieve(query, top_k=3)

    print("\nQuery:", query)
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc[:200]}...")
