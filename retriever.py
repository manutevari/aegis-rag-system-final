"""
PolicyRetriever — Vector store over corporate policy documents.
Backend: FAISS (default, local) or Chroma (persistent, VECTOR_STORE=chroma).
"""
import logging, os, pathlib
from typing import List
logger      = logging.getLogger(__name__)
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")
POLICY_DIR   = os.getenv("POLICY_DIR", "data/policies")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "800"))
OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "100"))
CHROMA_DIR   = os.getenv("CHROMA_DIR", "/tmp/dg_rag_chroma")


def _embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL)


def _load_docs():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    p = pathlib.Path(POLICY_DIR)
    if not p.exists():
        from app.tools._sample_policies import get_sample_docs
        logger.info("No policy dir found — loading built-in sample policies")
        return get_sample_docs()
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    docs = []
    for f in p.rglob("*"):
        try:
            loader = PyPDFLoader(str(f)) if f.suffix == ".pdf" else TextLoader(str(f))
            docs.extend(loader.load())
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    chunks = splitter.split_documents(docs)
    logger.info("Loaded %d chunks from %s", len(chunks), POLICY_DIR)
    return chunks


class PolicyRetriever:
    def __init__(self):
        emb   = _embeddings()
        docs  = _load_docs()
        if VECTOR_STORE == "chroma":
            from langchain_chroma import Chroma
            self._store = Chroma.from_documents(docs, emb, persist_directory=CHROMA_DIR, collection_name="dg_rag")
        else:
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.from_documents(docs, emb)
        logger.info("Vector store ready (%s, %d docs)", VECTOR_STORE, len(docs))

    def retrieve(self, query: str, top_k: int = 6) -> List[str]:
        try:
            results = self._store.max_marginal_relevance_search(query, k=top_k, fetch_k=top_k*3, lambda_mult=0.7)
        except Exception:
            results = self._store.similarity_search(query, k=top_k)
        return [d.page_content for d in results]
