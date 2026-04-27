"""
Vector Store Factory

Supports FAISS (default, local) and Chroma (persistent)
"""

import os
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

_vector_store = None


def get_embed_model_instance():
    """Get embedding model instance."""
    from langchain_openai import OpenAIEmbeddings
    from app.core.models import get_embed_model
    
    return OpenAIEmbeddings(
        model=get_embed_model(),
        api_key=os.getenv("OPENAI_API_KEY")
    )


def get_vector_store():
    """
    Get or initialize vector store singleton.
    
    Returns:
        Vector store instance (FAISS or Chroma)
    """
    global _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    vector_store_type = os.getenv("VECTOR_STORE", "faiss")
    
    # Dummy documents for initialization
    from langchain_core.documents import Document
    dummy_docs = [Document(page_content="Sample policy document")]
    
    emb = get_embed_model_instance()
    
    try:
        if vector_store_type == "chroma":
            from langchain_chroma import Chroma
            chroma_dir = os.getenv("CHROMA_DIR", "/tmp/dg_rag_chroma")
            _vector_store = Chroma.from_documents(
                dummy_docs,
                emb,
                persist_directory=chroma_dir,
                collection_name="dg_rag"
            )
            logger.info(f"Vector store initialized: Chroma ({chroma_dir})")
        else:
            from langchain_community.vectorstores import FAISS
            _vector_store = FAISS.from_documents(dummy_docs, emb)
            logger.info("Vector store initialized: FAISS (local)")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise
    
    return _vector_store


# Convenience alias for backwards compatibility
vector_db = get_vector_store()
