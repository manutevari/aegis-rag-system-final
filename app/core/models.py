"""
Core LLM and Embedding Model Factory
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_embed_model() -> str:
    """Get embedding model name from env or default."""
    return os.getenv("EMBED_MODEL", "text-embedding-3-small")


def get_llm(model_override: Optional[str] = None, temperature: float = 0.7, max_tokens: Optional[int] = None):
    """
    Factory for LLM instances.
    
    Args:
        model_override: Override default model (gpt-4o-mini)
        temperature: LLM temperature (0-1)
        max_tokens: Max output tokens
    
    Returns:
        Initialized ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    model = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def invoke_llm(messages: list, model_override: Optional[str] = None, temperature: float = 0):
    """
    Invoke LLM with messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_override: Override default model
        temperature: LLM temperature
    
    Returns:
        LLM response object
    """
    llm = get_llm(model_override=model_override, temperature=temperature)
    return llm.invoke(messages)
