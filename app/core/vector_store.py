"""
Shared vector store factory for ingestion and retrieval.

All policy indexing and query-time retrieval should use this module so the
persist directory, collection name, and embedding model stay aligned.
"""

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "db"
DEFAULT_COLLECTION_NAME = "aegis_policies"

_vectorstore = None


def get_db_path() -> str:
    return os.getenv("CHROMA_DIR") or os.getenv("VECTOR_DB_PATH") or DEFAULT_DB_PATH


def get_collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME)


def get_embedding_function():
    from langchain_openai import OpenAIEmbeddings

    from app.core.models import get_embed_model

    return OpenAIEmbeddings(
        model=get_embed_model(),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vectorstore():
    """Return the singleton persistent Chroma store."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_chroma import Chroma

    db_path = get_db_path()
    Path(db_path).mkdir(parents=True, exist_ok=True)

    _vectorstore = Chroma(
        collection_name=get_collection_name(),
        persist_directory=db_path,
        embedding_function=get_embedding_function(),
    )
    logger.info(
        "Vector store ready: Chroma path=%s collection=%s count=%s",
        db_path,
        get_collection_name(),
        get_collection_count(_vectorstore),
    )
    return _vectorstore


def get_vector_store():
    """Backward-compatible alias for older imports."""
    return get_vectorstore()


def reset_vectorstore_cache() -> None:
    global _vectorstore
    _vectorstore = None


def persist_vectorstore(store: Optional[Any] = None) -> None:
    store = store or get_vectorstore()
    persist = getattr(store, "persist", None)
    if callable(persist):
        persist()


def get_collection_count(store: Optional[Any] = None) -> int:
    try:
        store = store or _vectorstore or get_vectorstore()
        collection = getattr(store, "_collection", None)
        if collection is not None and hasattr(collection, "count"):
            return int(collection.count())
    except Exception as exc:
        logger.debug("Could not read Chroma collection count: %s", exc)
    return 0


def has_persisted_index() -> bool:
    return Path(get_db_path(), "chroma.sqlite3").exists()


def index_documents(documents: Iterable[Any]) -> dict:
    docs = list(documents or [])
    if not docs:
        return {"chunks_indexed": 0, "collection_count": get_collection_count()}

    store = get_vectorstore()
    store.add_documents(docs)
    persist_vectorstore(store)

    return {
        "chunks_indexed": len(docs),
        "collection_count": get_collection_count(store),
        "db_path": get_db_path(),
        "collection": get_collection_name(),
    }


def get_retriever(k: int = 5):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def search_documents(query: str, k: int = 5) -> List[Any]:
    if not query:
        return []

    retriever = get_retriever(k=k)
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query) or [])
    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(query) or [])
    return []


def ensure_vectorstore_ready(auto_ingest: bool = True) -> int:
    count = get_collection_count(get_vectorstore())
    if count or not auto_ingest:
        return count

    from policy_ingestion import run_ingestion

    logger.info("Vector store is empty; running policy ingestion")
    result = run_ingestion()
    return int(result.get("collection_count") or result.get("chunks_indexed") or 0)


class _LazyVectorStoreProxy:
    """Compatibility proxy for legacy code that imported vector_db directly."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_vectorstore(), name)

    def search(self, query: Any, top_k: int = 5, **kwargs: Any) -> List[Any]:
        store = get_vectorstore()
        if isinstance(query, str):
            return store.similarity_search(query, k=top_k, **kwargs)
        if hasattr(store, "similarity_search_by_vector"):
            return store.similarity_search_by_vector(query, k=top_k, **kwargs)
        return []


vector_db = _LazyVectorStoreProxy()
