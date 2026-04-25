from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from app.graph.workflow import build_graph

app = FastAPI()
graph = build_graph()

class Query(BaseModel):
    query: str
    history: list = []

@app.post("/chat")
async def chat(q: Query):
    state = {
        "query": q.query,
        "history": q.history,
        "trace_log": [],
        "retry_count": 0
    }

    result = await graph.ainvoke(state)

    return {
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "verified": result.get("verified"),
        "trace": result.get("trace_log")
    }
