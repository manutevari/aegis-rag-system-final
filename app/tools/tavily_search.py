from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

TAVILY_ENV = ["TAVILY_API_KEY", "TAVILY_KEY"]
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def tavily_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key:
        return explicit_key
    for name in TAVILY_ENV:
        value = os.getenv(name)
        if value:
            return value
    return ""


def tavily_search(
    query: str,
    *,
    api_key: Optional[str] = None,
    search_depth: str = "basic",
    max_results: int = 3,
    include_answer: str = "basic",
) -> Dict[str, object]:
    key = tavily_key(api_key)
    if not key:
        return {
            "enabled": False,
            "status": "missing_key",
            "answer": "",
            "results": [],
            "sources": [],
            "error": "Tavily API key is not configured",
        }

    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": max(1, min(int(max_results or 3), 10)),
        "include_answer": include_answer,
        "include_raw_content": False,
        "include_images": False,
    }
    response = requests.post(
        TAVILY_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()

    results: List[Dict[str, object]] = []
    for item in data.get("results", [])[: payload["max_results"]]:
        url = item.get("url", "")
        results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "content": item.get("content", ""),
                "score": item.get("score", ""),
                "published_date": item.get("published_date", ""),
            }
        )

    return {
        "enabled": True,
        "status": "success",
        "answer": data.get("answer", ""),
        "results": results,
        "sources": [item["url"] for item in results if item.get("url")],
        "response_time": data.get("response_time"),
        "usage": data.get("usage", {}),
    }


def tavily_context_block(payload: Optional[Dict[str, object]]) -> str:
    if not payload or not payload.get("enabled"):
        return "[]"
    compact = {
        "answer": payload.get("answer", ""),
        "results": payload.get("results", []),
        "sources": payload.get("sources", []),
    }
    return json.dumps(compact, ensure_ascii=True, indent=2, default=str)
