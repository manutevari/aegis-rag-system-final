"""
AEGIS UNIVERSAL STABILITY PATCH
Drop-in module to harden entire RAG + LangGraph + Streamlit system
"""

import os
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# ==============================
# 🔧 GLOBAL CONFIG
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AEGIS_PATCH")

MAX_MEMORY = 5
TIMEOUT = 20

VALID_MODELS = {
    "fast": "gpt-4o-mini",
    "strong": "gpt-4o"
}

# ==============================
# 🧠 SCHEMA VALIDATION
# ==============================

class GraphInput(BaseModel):
    query: str
    memory_context: List[str] = []

class GraphOutput(BaseModel):
    answer: str

# ==============================
# 🤖 MODEL MANAGER (SAFE)
# ==============================

def get_safe_llm(model_type="fast"):
    model = VALID_MODELS.get(model_type)

    if not model:
        raise ValueError(f"Invalid model type: {model_type}")

    try:
        return OpenAI(
            model=model,
            timeout=TIMEOUT,
            max_retries=2
        )
    except Exception as e:
        logger.exception("Model initialization failed")
        raise RuntimeError("LLM init failure") from e

# ==============================
# 🧠 MEMORY GUARD
# ==============================

class SafeMemory:
    def __init__(self):
        self.memory = []

    def add(self, message: str):
        self.memory.append(message)
        self.memory = self.memory[-MAX_MEMORY:]

    def get(self) -> List[str]:
        return self.memory[-MAX_MEMORY:]

# ==============================
# 📚 RETRIEVER GUARD
# ==============================

def safe_retrieve(retriever, query: str):
    try:
        docs = retriever.invoke(query)

        if not docs:
            return ["No relevant documents found"]

        return docs

    except Exception as e:
        logger.exception("Retriever failure")
        return ["Retriever error"]

# ==============================
# 🔁 SAFE GRAPH INVOKE
# ==============================

def safe_invoke(graph, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validate input
        validated_input = GraphInput(**payload)

        result = graph.invoke(validated_input.dict())

        if not result:
            return {"answer": "System error: empty response"}

        # Validate output
        validated_output = GraphOutput(**result)

        return validated_output.dict()

    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return {"answer": "Invalid input format"}

    except Exception as e:
        logger.exception("Graph execution failure")
        return {"answer": f"System failure: {str(e)}"}

# ==============================
# 🧩 SAFE NODE DECORATOR
# ==============================

def safe_node(fn):
    def wrapper(state):
        try:
            return fn(state)
        except Exception as e:
            logger.exception(f"Node failure: {fn.__name__}")
            return {"error": str(e)}
    return wrapper

# ==============================
# 🌐 STREAMLIT SAFE RUNNER
# ==============================

def run_streamlit_safe(graph, query: str, memory: SafeMemory):
    try:
        if not query or not query.strip():
            return {"answer": "Empty query"}

        memory_context = memory.get()

        result = safe_invoke(graph, {
            "query": query,
            "memory_context": memory_context
        })

        memory.add(query)
        memory.add(result.get("answer", ""))

        return result

    except Exception as e:
        logger.exception("Streamlit execution failed")
        return {"answer": "UI failure"}

# ==============================
# 🔒 PROMPT GUARD
# ==============================

SYSTEM_PROMPT = (
    "Answer strictly from provided context.\n"
    "If insufficient data, say: 'Insufficient data'.\n"
    "Max 100 tokens. No hallucination."
)

# ==============================
# 🧯 FAILSAFE ENTRYPOINT
# ==============================

def guarded_app(graph, retriever=None):
    """
    Universal entry wrapper
    """

    memory = SafeMemory()

    def handle(query: str):
        try:
            # Retrieve context safely
            context = []
            if retriever:
                context = safe_retrieve(retriever, query)

            result = run_streamlit_safe(graph, query, memory)

            if not result or "answer" not in result:
                return {"answer": "Unexpected system failure"}

            return result

        except Exception as e:
            logger.exception("Critical failure")
            return {"answer": "Fatal system error"}

    return handle
