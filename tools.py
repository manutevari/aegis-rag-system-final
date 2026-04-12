# =============================================================================
# AEGIS — LANGCHAIN TOOL SCHEMAS
# tools.py
#
# Lecture points addressed:
#   "LLMs generate text only; they signal tool usage by returning a tool
#    name and arguments." (Tool Calling #1)
#   "Execution loop: LLM → tool call → tool execution → tool output
#    appended to history → LLM final answer." (Tool Calling #2)
#   "LangChain workflow: bind tools, inspect ai_message.tool_calls,
#    invoke the tool, append results, re-invoke LLM." (Tool Calling #4)
#   "Tools require clear schemas; Pydantic is recommended." (Tool Calling #5)
#
# What this module provides:
#   1. PolicySearchInput  — Pydantic input schema for the search tool
#   2. PolicySummariseInput — Pydantic input schema for the summarise tool
#   3. search_policy_tool — @tool decorated LangChain tool
#   4. summarise_chunks_tool — @tool for cheap-model summarisation
#   5. run_tool_loop()    — explicit tool calling execution loop that:
#         a) binds tools to LLM
#         b) calls LLM with user query
#         c) inspects ai_message.tool_calls
#         d) executes each called tool
#         e) appends ToolMessage results to history
#         f) re-invokes LLM for final answer
#
# All tool inputs validated through Pydantic. Tool outputs appended to
# AegisState.messages as ToolMessage for full audit trail.
# =============================================================================

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from graph_state import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    VALID_CATEGORIES,
    tool_log,
)

__all__ = [
    "PolicySearchInput",
    "PolicySummariseInput",
    "search_policy_tool",
    "summarise_chunks_tool",
    "run_tool_loop",
    "AEGIS_TOOLS",
]

# ---------------------------------------------------------------------------
# Lazy LLM clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _llm_smart():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                      api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _llm_cheap():
    """Cheaper model for summarisation — reduces token cost."""
    from langchain_openai import ChatOpenAI
    # gpt-3.5-turbo is ~10× cheaper than gpt-4o for summarisation tasks
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                      api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------------------------------------------------------------------------
# Pydantic input schemas (lecture: "Pydantic is recommended for validation")
# ---------------------------------------------------------------------------

class PolicySearchInput(BaseModel):
    """Input schema for the policy search tool."""

    query:        str           = Field(..., min_length=3,
                                        description="User's natural language question")
    category:     Optional[str] = Field(default=None,
                                        description="Optional policy category filter")
    top_k:        int           = Field(default=5, ge=1, le=25,
                                        description="Number of chunks to return")
    num_queries:  int           = Field(default=4, ge=1, le=8,
                                        description="Number of query expansion variants")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v if v in VALID_CATEGORIES else None

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class PolicySummariseInput(BaseModel):
    """Input schema for the chunk summarisation tool."""

    chunks:       list[str] = Field(..., min_length=1,
                                    description="List of chunk texts to summarise")
    query:        str        = Field(..., min_length=3,
                                    description="User question to guide summarisation")
    max_tokens:   int        = Field(default=300, ge=50, le=800,
                                    description="Target token budget for summary")

    @field_validator("chunks")
    @classmethod
    def at_least_one_chunk(cls, v: list[str]) -> list[str]:
        non_empty = [c.strip() for c in v if c.strip()]
        if not non_empty:
            raise ValueError("At least one non-empty chunk is required")
        return non_empty


# ---------------------------------------------------------------------------
# LangChain @tool definitions
# (lecture: "Convert a function into a LangChain tool with schema and full
#  invocation loop")
# ---------------------------------------------------------------------------

@tool("search_policy", args_schema=PolicySearchInput)
def search_policy_tool(
    query: str,
    category: Optional[str] = None,
    top_k: int = 5,
    num_queries: int = 4,
) -> dict:
    """
    Search the corporate policy vector store for relevant policy chunks.

    Use this tool when you need to retrieve specific policy information.
    Returns a list of relevant chunks with metadata.

    Args:
        query:       The user's natural language question.
        category:    Optional policy category (Travel/HR/Finance/Legal/IT/General).
        top_k:       How many top chunks to return (1-25, default 5).
        num_queries: Number of query expansion variants (1-8, default 4).
    """
    from graph import run_query
    result = run_query(query)
    sources = result.get("sources", [])[:top_k]
    return {
        "chunks": [
            {
                "text":       s.get("chunk_text", "")[:500],
                "document_id": s.get("document_id", ""),
                "category":    s.get("policy_category", ""),
                "h1":          s.get("h1_header", ""),
                "h2":          s.get("h2_header", ""),
                "score":       round(s.get("rerank_score", 0.0), 4),
            }
            for s in sources
        ],
        "category_detected": result.get("category"),
        "total_retrieved":   result.get("retrieved", 0),
    }


@tool("summarise_chunks", args_schema=PolicySummariseInput)
def summarise_chunks_tool(
    chunks: list[str],
    query: str,
    max_tokens: int = 300,
) -> str:
    """
    Summarise a list of policy chunks relevant to a user query.

    Use this tool to compress large context before final answer generation.
    Uses a cheaper model to reduce token costs.

    Args:
        chunks:     List of raw chunk texts to summarise.
        query:      The user question to guide what to include.
        max_tokens: Target length of the summary in tokens.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    context = "\n\n---\n\n".join(chunks[:10])   # hard cap at 10 chunks
    prompt  = ChatPromptTemplate.from_messages([
        ("system",
         "You are a concise policy summariser. Given policy excerpts and a question, "
         f"write a summary of at most {max_tokens} tokens that captures all facts "
         "needed to answer the question. Be precise. Preserve numbers, dates, and limits."),
        ("human", f"Question: {query}\n\nPolicy excerpts:\n{context}"),
    ])
    chain = prompt | _llm_cheap() | StrOutputParser()
    try:
        return chain.invoke({})
    except Exception as exc:
        return f"Summarisation failed: {exc}"


# ---------------------------------------------------------------------------
# Tool registry — expose as a list for bind_tools()
# ---------------------------------------------------------------------------

AEGIS_TOOLS = [search_policy_tool, summarise_chunks_tool]


# ---------------------------------------------------------------------------
# Tool execution loop
# (lecture: "bind tools, inspect ai_message.tool_calls, invoke the tool,
#  append results, re-invoke LLM")
# ---------------------------------------------------------------------------

_TOOL_LOOP_SYSTEM = (
    "You are an authoritative corporate policy assistant with access to tools. "
    "Use the search_policy tool to find relevant policy information, then "
    "optionally use summarise_chunks to compress large results before answering. "
    "Always ground your final answer in the retrieved policy context."
)

# Map tool name → callable for dispatch
_TOOL_DISPATCH: dict[str, callable] = {
    "search_policy":  search_policy_tool,
    "summarise_chunks": summarise_chunks_tool,
}


def run_tool_loop(query: str, max_iterations: int = 3) -> dict:
    """
    Explicit LangChain tool calling execution loop.

    Lecture reference (Tool Calling #2 & #4):
      LLM → tool call → tool execution → tool output appended to history
      → LLM final answer

    Step 1: Build message history with SystemMessage + HumanMessage
    Step 2: Bind AEGIS_TOOLS to LLM; invoke with current history
    Step 3: If AI returns tool_calls:
              • Execute each tool via _TOOL_DISPATCH
              • Append AIMessage (with tool_calls) to history
              • Append ToolMessage (tool result) to history
              • Loop back to Step 2
    Step 4: No more tool_calls → return final AIMessage answer

    Returns:
        dict with keys: answer, messages (full history), iterations
    """
    from langchain_core.messages import BaseMessage

    llm_with_tools = _llm_smart().bind_tools(AEGIS_TOOLS)

    messages: list[BaseMessage] = [
        SystemMessage(content=_TOOL_LOOP_SYSTEM),
        HumanMessage(content=query),
    ]

    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # No tool calls → final answer
        if not getattr(ai_msg, "tool_calls", None):
            break

        # Execute each tool call and append results
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            call_id   = tc.get("id", f"{tool_name}_call")

            tool_fn = _TOOL_DISPATCH.get(tool_name)
            if tool_fn is None:
                result_content = f"Unknown tool: {tool_name}"
            else:
                try:
                    raw_result    = tool_fn.invoke(tool_args)
                    result_content = str(raw_result)
                except Exception as exc:
                    result_content = f"Tool error: {exc}"

            # Append ToolMessage — lecture: "tool output appended to history"
            messages.append(
                ToolMessage(content=result_content, tool_call_id=call_id)
            )

            # Also log a structured audit entry
            messages.append(
                tool_log(
                    tool_name=tool_name,
                    reason=f"LLM requested tool '{tool_name}' to answer: {query[:80]}",
                    inputs=tool_args,
                    outputs=result_content[:300],
                    tool_call_id=f"audit_{call_id}",
                )
            )

    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
            break

    return {
        "answer":     final_answer,
        "messages":   messages,
        "iterations": iterations,
    }
