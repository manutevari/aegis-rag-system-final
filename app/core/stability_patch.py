"""
Stability Patch for Safe Graph Invocation

Provides error-safe wrapper around LangGraph execution.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def safe_invoke(graph, initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely invoke graph with error handling and fallback responses.
    
    Args:
        graph: Compiled LangGraph workflow
        initial_state: Initial state dict for the graph
    
    Returns:
        Result dict with answer and metadata
    """
    try:
        # Sanitize initial state
        sanitized_state = _sanitize_state(initial_state)
        
        # Invoke the graph
        result = graph.invoke(sanitized_state)
        
        # Validate and extract answer
        answer = result.get("answer", "No response generated")
        
        return {
            "answer": answer,
            "route": result.get("route", "unknown"),
            "sources": result.get("sources", []),
            "trace_log": result.get("trace_log", []),
        }
    
    except KeyError as e:
        logger.error(f"Missing required state key: {e}")
        return {
            "answer": f"⚠️ Configuration error: {str(e)}",
            "route": "error",
            "sources": [],
            "trace_log": [str(e)],
        }
    
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        return {
            "answer": "⚠️ System error. Please try again.",
            "route": "error",
            "sources": [],
            "trace_log": [str(e)],
        }


def _sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize state dict for graph invocation.
    
    - Ensures required fields exist
    - Converts None values to defaults
    - Validates list/dict types
    
    Args:
        state: Raw state dict
    
    Returns:
        Sanitized state dict
    """
    sanitized = {
        "query": state.get("query", "").strip(),
        "history": state.get("history") or [],
        "memory_context": state.get("memory_context") or "",
        "trace_log": state.get("trace_log") or [],
        "employee_grade": state.get("employee_grade"),
    }
    
    # Validate list fields
    if not isinstance(sanitized["history"], list):
        sanitized["history"] = []
    
    if not isinstance(sanitized["trace_log"], list):
        sanitized["trace_log"] = []
    
    # Validate string fields
    if not isinstance(sanitized["query"], str):
        sanitized["query"] = ""
    
    if not isinstance(sanitized["memory_context"], str):
        sanitized["memory_context"] = ""
    
    # Ensure query is not empty
    if not sanitized["query"]:
        raise ValueError("Query cannot be empty")
    
    return sanitized
