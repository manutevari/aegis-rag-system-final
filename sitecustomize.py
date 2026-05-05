"""AEGIS startup guard.

The deployed app should not crash during indexing when an OPENAI_API_KEY exists
but has no embedding quota. By default, keep LangChain OpenAI embeddings
unavailable so app.py uses local hash embeddings for the FAISS index. Set
AEGIS_ALLOW_HOSTED_EMBEDDINGS=1 to opt back into hosted embeddings.
"""
from __future__ import annotations

import os
import sys
import types


if os.getenv("AEGIS_ALLOW_HOSTED_EMBEDDINGS", "").lower() not in {"1", "true", "yes"}:
    if "langchain_openai" not in sys.modules:
        sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")
