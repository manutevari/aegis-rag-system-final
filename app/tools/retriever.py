"""Backward-compatible retriever facade.

The graph now retrieves directly from app.core.vector_store. This class remains
for older imports, but it delegates to the same shared Chroma collection.
"""

import logging
from typing import Any, List

from app.core.vector_store import search_documents

logger = logging.getLogger(__name__)


class PolicyRetriever:
    def __init__(self, default_k: int = 6):
        self.default_k = default_k

    def retrieve_documents(self, query: str, top_k: int = None) -> List[Any]:
        k = top_k or self.default_k
        try:
            return search_documents(query, k=k)
        except Exception as exc:
            logger.error("Policy retrieval failed: %s", exc)
            return []

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        docs = self.retrieve_documents(query, top_k=top_k)
        return [getattr(doc, "page_content", str(doc)) for doc in docs]
