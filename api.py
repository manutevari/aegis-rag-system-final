from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio

from app.graph.workflow import build_graph


# ==============================
# 🔹 Initialize App
# ==============================
app = FastAPI()


# ==============================
# 🔹 CORS (REQUIRED for React)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# 🔹 Load Graph (once)
# ==============================
graph = build_graph()


# ==============================
# 🔹 Request Schema
# ==============================
class Query(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []


# ==============================
# 🔹 Health Check (optional but useful)
# ==============================
@app.get("/")
def root():
    return {"status": "API is running"}


# ==============================
# 🔹 Chat Endpoint
# ==============================
@app.post("/chat")
async def chat(q: Query):
    try:
        # Basic validation
        if not q.query.strip():
            return {"answer": "Query cannot be empty."}

        state = {
            "query": q.query,
            "history": q.history,
            "trace_log": [],
            "retry_count": 0
        }

        result = await graph.ainvoke(state)

        return {
            "answer": result.get("answer", "No answer generated."),
            "sources": result.get("sources", []),
            "verified": result.get("verified", False),
            "trace": result.get("trace_log", [])
        }

    except Exception as e:
        return {
            "answer": "Internal server error.",
            "error": str(e)
        }
