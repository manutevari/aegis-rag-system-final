"""
Vector Store — Unified FAISS + Chroma backend with persistence.

Features:
- Backend switch via ENV (VECTOR_STORE=faiss|chroma)
- Auto-load + auto-save
- Clean API: add_texts, search, retrieve
- Safe initialization (no dummy embeddings pollution)
"""

import os
import logging
from typing import List, Any, Optional

from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PERSIST_DIR = "vector_store"


# ==============================
# 🔹 Embeddings
# ==============================
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# ==============================
# 🔹 VectorDB (Unified Wrapper)
# ==============================
class VectorDB:
    def __init__(self):
        self.embeddings = _get_embeddings()
        self.backend = os.getenv("VECTOR_STORE", "faiss").lower()
        self.store: Optional[Any] = None

        self._init_store()

    # --------------------------
    # Init / Load
    # --------------------------
    def _init_store(self):
        """Initialize or load persistent store."""
        try:
            if self.backend == "chroma":
                self.store = Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=self.embeddings,
                )
                logger.info("Chroma initialized (persistent)")

            else:
                # FAISS
                if os.path.exists(PERSIST_DIR):
                    self.store = FAISS.load_local(
                        PERSIST_DIR,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("FAISS loaded from disk")
                else:
                    self.store = None
                    logger.info("FAISS initialized (empty)")

        except Exception as e:
            logger.error("Vector store init failed: %s", e)
            self.store = None

    # --------------------------
    # Add Data
    # --------------------------
    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to store."""
        if not texts:
            return

        try:
            if self.store is None:
                # create new store
                if self.backend == "chroma":
                    self.store = Chroma.from_texts(
                        texts,
                        embedding=self.embeddings,
                        persist_directory=PERSIST_DIR,
                        metadatas=metadatas,
                    )
                else:
                    self.store = FAISS.from_texts(
                        texts,
                        self.embeddings,
                        metadatas=metadatas,
                    )
            else:
                self.store.add_texts(texts, metadatas=metadatas)

            logger.info("Added %d documents", len(texts))

        except Exception as e:
            logger.error("Add texts failed: %s", e)

    # --------------------------
    # Search
    # --------------------------
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """Search using query string."""
        if not self.store:
            logger.warning("Vector store empty")
            return []

        try:
            return self.store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    # --------------------------
    # Persist
    # --------------------------
    def persist(self):
        """Persist store safely."""
        if not self.store:
            return

        try:
            if self.backend == "chroma":
                self.store.persist()
                logger.info("Chroma persisted")

            else:
                self.store.save_local(PERSIST_DIR)
                logger.info("FAISS saved")

        except Exception as e:
            logger.error("Persist failed: %s", e)

    # --------------------------
    # Reset
    # --------------------------
    def reset(self):
        """Delete store completely."""
        import shutil

        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            logger.warning("Vector store reset")

        self.store = None


# ==============================
# 🔹 Global Instance
# ==============================
vector_db = VectorDB()


# ==============================
# 🔹 Retrieval API
# ==============================
def retrieve(query: str, top_k: int = 3) -> List[str]:
    """End-to-end retrieval."""
    results = vector_db.search(query, top_k=top_k)

    docs = [
        getattr(r, "page_content", str(r))
        for r in results
    ]

    return docs


# ==============================
# 🔹 Example Usage
# ==============================
if __name__ == "__main__":
    # Example: add + persist
    vector_db.add_texts(
        ["L6 employees get ₹1,50,000 laptop budget"],
        metadatas=[{"source": "policy"}],
    )
    vector_db.persist()

    # Query
    query = "laptop budget for L6"
    results = retrieve(query)

    print("\nQuery:", query)
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc[:200]}...")
