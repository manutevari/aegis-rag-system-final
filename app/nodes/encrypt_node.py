"""
Ultra-Lean 500-Token Pipeline — Retrieval + Generator + Encrypt

Includes:

* Optimized Retrieval Node (top_k=2, compression, dedup, history-aware)
* Generator (strict prompt, ≤100 tokens output)
* Encrypt Node
  """

import logging
import os
from typing import List, Any

from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace
from app.utils.encryption import encrypt
from app.core.models import get_llm
from app.utils.token_budget import build_prompt

logger = logging.getLogger(**name**)
_retriever: PolicyRetriever | None = None

# -------------------------

# CONSTANTS (ULTRA-LEAN)

# -------------------------

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 120
TOP_K = 2

SYSTEM = (
"Answer concisely using ONLY the context. "
"If insufficient context, say: 'Insufficient data'. "
"Max 100 tokens. No fluff."
)

# -------------------------

# RETRIEVER

# -------------------------

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
return " ".join(text.split()[:max_tokens])

def _compress_docs(docs: List[Any], k: int) -> List[str]:
texts = []
for d in docs[:k]:
content = getattr(d, "page_content", str(d))
texts.append(_trim_text(content, MAX_CHUNK_TOKENS))
return texts

def retrieval_node(state: AgentState) -> AgentState:
query = state.get("query", "")
grade = state.get("employee_grade")
history = state.get("history", "")
vector_memory = state.get("vector_memory")

```
base_query = f"[Grade: {grade}] {query}" if grade else query

if history:
    history_trimmed = history[-MAX_HISTORY_CHARS:]
    enhanced_query = f"{base_query}\nContext:\n{history_trimmed}"
else:
    enhanced_query = base_query

try:
    docs = _get_retriever().retrieve(enhanced_query, top_k=TOP_K)
except Exception as e:
    logger.error("Retrieval error: %s", e)
    docs = []

if vector_memory:
    try:
        docs.extend(vector_memory.search(query, k=1))
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)

docs = _deduplicate_docs(docs)
compressed_chunks = _compress_docs(docs, TOP_K)

return trace(
    {
        **state,
        "retrieval_docs": docs,
        "retrieval_text": compressed_chunks,
    },
    node="retrieval",
    data={"chunks": len(compressed_chunks)}
)
```

# -------------------------

# GENERATOR

# -------------------------

def generate_node(state: AgentState) -> AgentState:
query = state.get("query", "")
context_chunks = state.get("retrieval_text", [])

```
context = "\n\n".join(context_chunks)

llm = get_llm(model_override="gpt-4o-mini", max_tokens=120, temperature=0)

prompt = build_prompt(SYSTEM, query, context)

try:
    res = llm.invoke(prompt)
    answer = res.content.strip()
except Exception as e:
    logger.error("Generation failed: %s", e)
    answer = "Error generating response."

return trace(
    {**state, "answer": answer},
    node="generator",
    data={"answer_len": len(answer)}
)
```

# -------------------------

# ENCRYPT NODE

# -------------------------

def encrypt_node(state: AgentState) -> AgentState:
try:
enc = encrypt(state.get("answer", ""))
except Exception as e:
logger.error("Encrypt failed: %s", e)
enc = state.get("answer", "").encode()

```
return trace(
    {**state, "_encrypted_answer": enc},
    node="encrypt",
    data={"ok": True}
)
```

# -------------------------

# SIMPLE RETRIEVE FUNCTION

# -------------------------

def retrieve(query, retriever, k=2):
docs = retriever.get_relevant_documents(query)
texts = [d.page_content for d in docs[:k]]
return texts
