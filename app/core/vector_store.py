import logging
from typing import List, Any
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ==============================
# 🔹 Embedding Model
# ==============================
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        openai_api_key=None  # will be picked up from env
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
