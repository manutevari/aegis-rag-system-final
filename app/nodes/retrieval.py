"""
Retrieval Node — Ultra-Lean 500-Token Optimized

Enhancements:

* History-aware query enrichment (trimmed)
* Long-term memory recall
* Strict top_k control (default=2)
* Deduplication
* Context compression (token-safe)
* Safe fallbacks + tracing
  """

import logging
import os
from typing import List, Any

from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace

logger = logging.getLogger(**name**)
_retriever: PolicyRetriever | None = None

# 🔒 HARD LIMITS (500-token architecture)

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 120
DEFAULT_TOP_K = 2

def _get_retriever() -> PolicyRetriever:
global _retriever
if _retriever is None:
_retriever = PolicyRetriever()
return _retriever

def _deduplicate_docs(docs: List[Any]) -> List[Any]:
seen = set()
unique_docs = []
for d in docs:
content = getattr(d, "page_content", str(d))
if content not in seen:
seen.add(content)
unique_docs.append(d)
return unique_docs

def _trim_text(text: str, max_tokens: int) -> str:
words = text.split()
return " ".join(words[:max_tokens])

def _compress_docs(docs: List[Any], k: int) -> List[str]:
"""Convert docs → compressed text chunks"""
texts = []
for d in docs[:k]:
content = getattr(d, "page_content", str(d))
texts.append(_trim_text(content, MAX_CHUNK_TOKENS))
return texts

def run(state: AgentState) -> AgentState:
query = state.get("query", "")
grade = state.get("employee_grade")
history = state.get("history", "")
vector_memory = state.get("vector_memory")

```
# 🔒 STRICT TOP-K CONTROL
top_k = int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K))
top_k = min(top_k, 3)  # hard cap

# --- STEP 1: Grade-boosted query ---
base_query = f"[Grade: {grade}] {query}" if grade else query

# --- STEP 2: History-aware enrichment (ULTRA-TRIMMED) ---
if history:
    history_trimmed = history[-MAX_HISTORY_CHARS:]
    enhanced_query = f"{base_query}\nContext:\n{history_trimmed}"
else:
    enhanced_query = base_query

logger.info(
    "Retrieval — query_len=%d top_k=%d",
    len(enhanced_query),
    top_k,
)

# --- STEP 3: Primary retrieval ---
try:
    docs = _get_retriever().retrieve(enhanced_query, top_k=top_k)
except Exception as e:
    logger.error("Retrieval error: %s", e)
    docs = []

# --- STEP 4: Memory augmentation (LIMITED) ---
if vector_memory:
    try:
        memory_docs = vector_memory.search(query, k=1)  # ultra-lean
        docs.extend(memory_docs)
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)

# --- STEP 5: Deduplicate ---
docs = _deduplicate_docs(docs)

# --- STEP 6: COMPRESS CONTEXT (CRITICAL) ---
compressed_chunks = _compress_docs(docs, k=top_k)

# --- STEP 7: TRACE ---
return trace(
    {
        **state,
        "retrieval_docs": docs,
        "retrieval_text": compressed_chunks,  # 🔥 USE THIS DOWNSTREAM
    },
    node="retrieval",
    data={
        "chunks": len(docs),
        "compressed_chunks": len(compressed_chunks),
        "top_k": top_k,
        "used_history": bool(history),
        "used_memory": bool(vector_memory),
    },
)
```

# -------------------------

# SIMPLE RETRIEVE FUNCTION (FOR DIRECT USE)

# -------------------------

def retrieve(query, retriever, k=2):
docs = retriever.get_relevant_documents(query)
texts = [d.page_content for d in docs[:k]]
return texts
