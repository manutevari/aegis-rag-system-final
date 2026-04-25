"""FastAPI REST API — /chat, /health, /cache, /hitl/review, /traces"""
import asyncio, json, logging, os, pathlib, time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graph.workflow import build_graph
from app.utils.pickle_cache import PickleCache
from app.utils.encryption import encrypt, decrypt

app = FastAPI(title="Decision-Grade Policy RAG", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
logger = logging.getLogger(__name__)
_graph = None
_cache = PickleCache()

def get_graph():
    global _graph
    if _graph is None: _graph = build_graph()
    return _graph

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []
    employee_grade: Optional[str] = None
    use_cache: bool = True

class ChatResponse(BaseModel):
    answer: str; sources: List[str] = []; route: Optional[str] = None
    verified: Optional[bool] = None; cached: bool = False; latency_ms: float = 0

class HITLDecision(BaseModel):
    review_id: str; decision: str; edited_answer: Optional[str] = None

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()
    if req.use_cache:
        b = _cache.get(req.query)
        if b: return ChatResponse(answer=decrypt(b), cached=True, latency_ms=round((time.time()-t0)*1000,1))
    try:
        init = {"query": req.query, "history": req.history, "trace_log": [], "retry_count": 0}
        if req.employee_grade: init["employee_grade"] = req.employee_grade.upper()
        result = await get_graph().ainvoke(init)
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    answer = result.get("answer","")
    if req.use_cache and answer: _cache.set(req.query, encrypt(answer))
    return ChatResponse(answer=answer, sources=result.get("sources",[]),
                        route=result.get("route"), verified=result.get("verified"),
                        latency_ms=round((time.time()-t0)*1000,1))

@app.get("/cache/stats")
def cache_stats(): return _cache.stats()

@app.post("/cache/clear")
def cache_clear(): return {"deleted": _cache.clear()}

@app.post("/hitl/review")
def hitl_review(d: HITLDecision):
    q = pathlib.Path(os.getenv("HITL_QUEUE_DIR","/tmp/hitl_queue"))
    p = q / f"{d.review_id}.json"
    if not p.exists(): raise HTTPException(404, "Review item not found")
    data = json.loads(p.read_text())
    data.update({"status": d.decision, "edited_answer": d.edited_answer})
    p.write_text(json.dumps(data, indent=2))
    return {"ok": True}

@app.get("/traces")
def list_traces(limit: int = 20):
    td = pathlib.Path(os.getenv("TRACE_DIR","/tmp/dg_rag_traces"))
    if not td.exists(): return []
    files = sorted(td.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [json.loads(f.read_text()) for f in files[:limit]]
