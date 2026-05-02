"""
Shared vector store factory for ingestion and retrieval.

AEGIS uses OpenAI text-embedding-3-large and Pinecone when credentials are
configured, with Chroma/hash fallbacks for local development and CI.
"""

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "db"
DEFAULT_COLLECTION_NAME = "aegis_policies"
DEFAULT_LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HASH_DIMENSIONS = 384

_vectorstore = None


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embedding fallback with no network dependency."""

    def __init__(self, dimension: Optional[int] = None):
        settings = get_settings()
        self.dimension = dimension or settings.local_hash_embed_dim or DEFAULT_HASH_DIMENSIONS

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower()) or [""]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class OpenAIEmbeddingModel(Embeddings):
    """LangChain-compatible OpenAI embedding adapter."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        if not self.settings.openai_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.settings.openai_key)

    def _kwargs(self, texts: List[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.settings.openai_embedding_model,
            "input": texts,
        }
        if self.settings.openai_embedding_dimensions > 0:
            kwargs["dimensions"] = self.settings.openai_embedding_dimensions
        return kwargs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client().embeddings.create(**self._kwargs(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_db_path() -> str:
    settings = get_settings()
    return settings.chroma_dir or os.getenv("VECTOR_DB_PATH") or DEFAULT_DB_PATH


def get_collection_name() -> str:
    return get_settings().chroma_collection or DEFAULT_COLLECTION_NAME


def _huggingface_embeddings():
    settings = get_settings()
    if not settings.allow_hf_downloads:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_kwargs = {}
    if not settings.allow_hf_downloads:
        model_kwargs["local_files_only"] = True

    return HuggingFaceEmbeddings(
        model_name=settings.local_embed_model or DEFAULT_LOCAL_EMBED_MODEL,
        model_kwargs=model_kwargs,
    )


def get_embeddings() -> Embeddings:
    settings = get_settings()
    provider = settings.rag_embeddings_provider.strip().lower()

    if provider == "openai":
        if not settings.openai_key:
            logger.warning("OPENAI_API_KEY is not set; using hash embeddings")
            return LocalHashEmbeddings()
        try:
            logger.info("Using OpenAI embeddings model %s", settings.openai_embedding_model)
            return OpenAIEmbeddingModel(settings=settings)
        except Exception as exc:
            logger.warning("OpenAI embeddings unavailable; using hash embeddings: %s", exc)
            return LocalHashEmbeddings()

    if provider in {"hash", "local_hash", "offline"}:
        logger.info("Using deterministic local hash embeddings")
        return LocalHashEmbeddings()

    if provider != "local":
        logger.warning("Unknown RAG_EMBEDDINGS_PROVIDER=%s; using hash embeddings", provider)
        return LocalHashEmbeddings()

    try:
        embeddings = _huggingface_embeddings()
        logger.info("Using local Hugging Face embeddings")
        return embeddings
    except Exception as exc:
        logger.warning("Local Hugging Face embeddings unavailable; using hash embeddings: %s", exc)
        return LocalHashEmbeddings()


def get_embedding_function() -> Embeddings:
    return get_embeddings()


def _safe_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return str(value)


def _sanitize_metadata(metadata: Dict[str, Any], content: str) -> Dict[str, Any]:
    clean: Dict[str, Any] = {"content": content}
    for key, value in (metadata or {}).items():
        safe_value = _safe_metadata_value(value)
        if safe_value is not None:
            clean[str(key)] = safe_value
    return clean


def _document_id(doc: Document, index: int) -> str:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = str(metadata.get("source_path") or metadata.get("source") or "document")
    chunk_index = metadata.get("chunk_index", index)
    digest = hashlib.sha1((doc.page_content or "").encode("utf-8")).hexdigest()[:12]
    safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "-", source).strip("-")[:80] or "document"
    return f"{safe_source}-{chunk_index}-{digest}"


class PineconePolicyRetriever:
    def __init__(self, store: "PineconePolicyStore", search_kwargs: Optional[Dict[str, Any]] = None):
        self.store = store
        self.search_kwargs = search_kwargs or {}

    def invoke(self, query: str) -> List[Document]:
        return self.store.similarity_search(
            query,
            k=int(self.search_kwargs.get("k", 5)),
            metadata_filter=self.search_kwargs.get("filter"),
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


class PineconePolicyStore:
    """Small Pinecone wrapper matching the subset of Chroma used by AEGIS."""

    def __init__(self, settings: AppSettings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.index = self._index()

    def _index(self):
        from pinecone import Pinecone, ServerlessSpec

        if not self.settings.pinecone_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone")
        if not self.settings.pinecone_index_name and not self.settings.pinecone_index_host:
            raise ValueError("PINECONE_INDEX_NAME or PINECONE_INDEX_HOST is required")

        pc = Pinecone(api_key=self.settings.pinecone_key)
        if self.settings.pinecone_create_index and self.settings.pinecone_index_name:
            existing = {item.get("name") if isinstance(item, dict) else getattr(item, "name", None) for item in pc.list_indexes()}
            if self.settings.pinecone_index_name not in existing:
                pc.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=self.settings.openai_embedding_dimensions,
                    metric=self.settings.pinecone_metric,
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                    deletion_protection="disabled",
                )

        if self.settings.pinecone_index_host:
            return pc.Index(host=self.settings.pinecone_index_host)
        return pc.Index(self.settings.pinecone_index_name)

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return

        batch_size = max(int(self.settings.pinecone_batch_size or 100), 1)
        namespace = self.settings.pinecone_namespace or None
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            vectors = self.embeddings.embed_documents([doc.page_content for doc in batch])
            records = []
            for offset, (doc, vector) in enumerate(zip(batch, vectors)):
                records.append(
                    {
                        "id": _document_id(doc, start + offset),
                        "values": vector,
                        "metadata": _sanitize_metadata(dict(doc.metadata or {}), doc.page_content),
                    }
                )
            kwargs: Dict[str, Any] = {"vectors": records}
            if namespace:
                kwargs["namespace"] = namespace
            self.index.upsert(**kwargs)

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> PineconePolicyRetriever:
        return PineconePolicyRetriever(self, search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        kwargs: Dict[str, Any] = {
            "vector": vector,
            "top_k": k,
            "include_metadata": True,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        if self.settings.pinecone_namespace:
            kwargs["namespace"] = self.settings.pinecone_namespace

        result = self.index.query(**kwargs)
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        docs: List[Document] = []
        for match in matches or []:
            metadata = dict(match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {})
            score = match.get("score") if isinstance(match, dict) else getattr(match, "score", None)
            content = str(metadata.pop("content", ""))
            metadata["pinecone_score"] = float(score or 0.0)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        if isinstance(stats, dict):
            return int(stats.get("total_vector_count") or 0)
        return int(getattr(stats, "total_vector_count", 0) or 0)


def _get_chroma_store():
    from langchain_chroma import Chroma

    db_path = get_db_path()
    Path(db_path).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=get_collection_name(),
        persist_directory=db_path,
        embedding_function=get_embeddings(),
    )


def get_vectorstore():
    """Return the singleton vector store."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()
    if settings.use_pinecone:
        if settings.pinecone_key and settings.openai_key:
            try:
                _vectorstore = PineconePolicyStore(settings=settings, embeddings=get_embeddings())
                logger.info("Pinecone vector store ready: index=%s namespace=%s", settings.pinecone_index_name, settings.pinecone_namespace)
                return _vectorstore
            except Exception as exc:
                logger.warning("Pinecone unavailable; falling back to Chroma: %s", exc)
        else:
            logger.warning("Pinecone/OpenAI credentials are not fully configured; falling back to Chroma")

    _vectorstore = _get_chroma_store()
    logger.info(
        "Chroma vector store ready: path=%s collection=%s count=%s",
        get_db_path(),
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
        if isinstance(store, PineconePolicyStore):
            return store.count()
        collection = getattr(store, "_collection", None)
        if collection is not None and hasattr(collection, "count"):
            return int(collection.count())
    except Exception as exc:
        logger.debug("Could not read collection count: %s", exc)
    return 0


def has_persisted_index() -> bool:
    settings = get_settings()
    if settings.use_pinecone:
        return bool(settings.pinecone_key and (settings.pinecone_index_name or settings.pinecone_index_host))
    return Path(get_db_path(), "chroma.sqlite3").exists()


def index_documents(documents: Iterable[Any]) -> dict:
    docs = list(documents or [])
    if not docs:
        return {"chunks_indexed": 0, "collection_count": get_collection_count()}

    store = get_vectorstore()
    store.add_documents(docs)
    persist_vectorstore(store)

    settings = get_settings()
    return {
        "chunks_indexed": len(docs),
        "collection_count": get_collection_count(store),
        "db_path": get_db_path() if not isinstance(store, PineconePolicyStore) else None,
        "collection": get_collection_name() if not isinstance(store, PineconePolicyStore) else settings.pinecone_index_name,
        "vector_backend": "pinecone" if isinstance(store, PineconePolicyStore) else "chroma",
    }


def get_retriever(k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None):
    search_kwargs: Dict[str, Any] = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def search_documents(
    query: str,
    k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    if not query:
        return []

    retriever = get_retriever(k=k, metadata_filter=metadata_filter)
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
        if isinstance(store, PineconePolicyStore):
            return store.similarity_search(str(query), k=top_k, metadata_filter=kwargs.get("filter"))
        if isinstance(query, str):
            return store.similarity_search(query, k=top_k, **kwargs)
        if hasattr(store, "similarity_search_by_vector"):
            return store.similarity_search_by_vector(query, k=top_k, **kwargs)
        return []


vector_db = _LazyVectorStoreProxy()
