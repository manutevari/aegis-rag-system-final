import ast
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.metadata import VALID_CATEGORIES, extract_metadata

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


APP_TITLE = "Multi-Document Policy Intelligence Engine"
DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
STRICT_NOT_FOUND = "This information is not available in the policy."

CHUNK_TARGET_CHARS = 1000
CHUNK_OVERLAP_RATIO = 0.12
EXPANSION_COUNT = 4
BROAD_RETRIEVAL_K = 25
RERANK_TOP_N = 5

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_COHERE_RERANK_MODEL = "rerank-v4.0-pro"
TAVILY_URL = "https://api.tavily.com/search"

getcontext().prec = 28


PROVIDERS: Dict[str, dict] = {
    "Auto": {"kind": "auto", "models": ["Auto choose"], "env": []},
    "Extractive": {"kind": "extractive", "models": ["Local retrieval only"], "env": []},
    "OpenAI": {"kind": "openai", "models": ["gpt-4o-mini", "gpt-4o", "custom"], "env": ["OPENAI_API_KEY"]},
    "Grok": {"kind": "openai", "base_url": "https://api.x.ai/v1", "models": ["grok-4", "grok-3-mini", "custom"], "env": ["XAI_API_KEY", "GROK_API_KEY"]},
    "Gemini": {"kind": "gemini", "models": ["gemini-2.5-flash", "gemini-2.5-pro", "custom"], "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]},
    "Mistral": {"kind": "openai", "base_url": "https://api.mistral.ai/v1", "models": ["mistral-medium-latest", "mistral-large-latest", "custom"], "env": ["MISTRAL_API_KEY"]},
    "OpenRouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "models": ["openrouter/auto", "openai/gpt-4o-mini", "custom"], "env": ["OPENROUTER_API_KEY"]},
    "Hugging Face": {"kind": "openai", "base_url": "https://router.huggingface.co/v1", "models": ["meta-llama/Llama-3.1-8B-Instruct:fastest", "mistralai/Mistral-7B-Instruct-v0.3:fastest", "custom"], "env": ["HF_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]},
    "Ollama": {"kind": "ollama", "models": ["llama3.1", "llama3", "mistral", "custom"], "env": []},
}

POSITIVE_WORDS = {"thanks", "thank", "great", "good", "happy", "excellent", "clear", "perfect"}
NEGATIVE_WORDS = {"angry", "bad", "confused", "denied", "frustrated", "issue", "problem", "urgent", "wrong"}
STOPWORDS = {"about", "after", "also", "answer", "because", "before", "could", "does", "from", "have", "into", "only", "policy", "question", "should", "that", "their", "there", "this", "what", "when", "where", "which", "with", "would"}


class HashEmbeddings(Embeddings):
    """Tiny local fallback so the app still indexes without an embedding key."""

    def __init__(self, size: int = 768):
        self.size = size

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        for token in re.findall(r"\w+", (text or "").lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@dataclass
class AppIndex:
    vectorstore: FAISS
    category_indexes: Dict[str, FAISS]
    sources: List[str]
    metadata_rows: List[dict]
    chunk_count: int


def secret_or_env(names: Sequence[str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    try:
        for name in names:
            value = st.secrets.get(name, "")
            if value:
                return value
    except Exception:
        pass
    return ""


def provider_key(provider: str) -> str:
    session_key = f"{provider.lower().replace(' ', '_')}_api_key"
    return st.session_state.get(session_key, "") or secret_or_env(PROVIDERS[provider].get("env", []))


def tool_key(name: str, env_names: Sequence[str]) -> str:
    return st.session_state.get(f"{name}_key", "") or secret_or_env(env_names)


def get_embeddings(use_openai: bool) -> Embeddings:
    key = provider_key("OpenAI")
    if use_openai and key and OpenAIEmbeddings:
        return OpenAIEmbeddings(api_key=key)
    return HashEmbeddings()


def loader_for(path: Path, content_type: str = ""):
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path))
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or path.suffix.lower() == ".docx":
        return Docx2txtLoader(str(path))
    return TextLoader(str(path), encoding="utf-8")


def load_file(path: Path, source_name: str, content_type: str = "", metadata_key: str = "") -> Optional[Document]:
    docs = loader_for(path, content_type).load()
    text = "\n\n".join(doc.page_content for doc in docs if doc.page_content).strip()
    if not text:
        return None
    metadata = {"source": source_name, **extract_metadata(text, api_key=metadata_key)}
    return Document(page_content=text, metadata=metadata)


def load_documents(upload_payload: Tuple[Tuple[str, str, bytes], ...], metadata_key: str = "") -> List[Document]:
    if upload_payload:
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_name, content_type, content in upload_payload:
                suffix = Path(file_name).suffix or ".txt"
                path = Path(temp_dir) / f"{hashlib.sha256(content).hexdigest()}{suffix}"
                path.write_bytes(content)
                doc = load_file(path, file_name, content_type, metadata_key)
                if doc:
                    documents.append(doc)
        return documents

    if not DATA_DIR.exists():
        return []
    documents = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            doc = load_file(path, path.name, metadata_key=metadata_key)
            if doc:
                documents.append(doc)
    return documents


HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def markdown_sections(text: str) -> List[Tuple[str, str]]:
    lines = (text or "").splitlines()
    sections, stack, current_header, current = [], [], "Document", []
    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            if "\n".join(current).strip():
                sections.append((current_header, "\n".join(current).strip()))
            level, title = len(match.group(1)), match.group(2).strip()
            stack = stack[: level - 1] + [title]
            current_header = " > ".join(stack)
            current = [line]
        else:
            current.append(line)
    if "\n".join(current).strip():
        sections.append((current_header, "\n".join(current).strip()))
    return sections


def is_table_start(lines: List[str], i: int) -> bool:
    return i + 1 < len(lines) and "|" in lines[i] and bool(TABLE_SEPARATOR_RE.match(lines[i + 1]))


def blocks(section: str) -> List[Tuple[str, str]]:
    lines, out, i = section.splitlines(), [], 0
    while i < len(lines):
        if is_table_start(lines, i):
            table = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                table.append(lines[i])
                i += 1
            out.append(("table", "\n".join(table).strip()))
        else:
            plain = []
            while i < len(lines) and not is_table_start(lines, i):
                plain.append(lines[i])
                i += 1
            text = "\n".join(plain).strip()
            if text:
                out.append(("text", text))
    return out


def split_words(text: str, max_chars: int) -> List[str]:
    pieces, current, size = [], [], 0
    for word in re.findall(r"\S+", text):
        projected = size + len(word) + (1 if current else 0)
        if current and projected > max_chars:
            pieces.append(" ".join(current))
            current, size = [word], len(word)
        else:
            current.append(word)
            size = projected
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_table(table: str) -> List[Tuple[str, str]]:
    if len(table) <= CHUNK_TARGET_CHARS:
        return [("table", table)]
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) <= 2:
        return [("table", table)]
    header, rows, chunks, current = lines[:2], lines[2:], [], []
    for row in rows:
        candidate = "\n".join(header + current + [row])
        if current and len(candidate) > CHUNK_TARGET_CHARS:
            chunks.append(("table_row_chunk", "\n".join(header + current)))
            current = [row]
        else:
            current.append(row)
    if current:
        chunks.append(("table_row_chunk", "\n".join(header + current)))
    return chunks


def overlap(previous: str) -> Tuple[str, int]:
    tokens = re.findall(r"\S+", previous or "")
    if len(tokens) < 8:
        return "", 0
    count = max(1, min(80, int(len(tokens) * CHUNK_OVERLAP_RATIO)))
    return " ".join(tokens[-count:]), count


def split_documents(documents: List[Document]) -> List[Document]:
    chunks = []
    for doc in documents:
        raw = []
        for section_header, section_text in markdown_sections(doc.page_content):
            for kind, text in blocks(section_text):
                if kind == "table":
                    raw.extend((section_header, table_kind, table_text) for table_kind, table_text in split_table(text))
                else:
                    raw.extend((section_header, "text", piece) for piece in split_words(text, CHUNK_TARGET_CHARS))

        previous = ""
        for index, (section_header, kind, text) in enumerate(raw):
            prefix, overlap_tokens = overlap(previous)
            page = f"[Overlap from previous chunk]\n{prefix}\n\n{text}" if prefix else text
            metadata = {
                **doc.metadata,
                "chunk_index": index,
                "section_header": section_header,
                "block_type": kind,
                "chunking_strategy": "markdown_header_table_overlap",
                "chunk_overlap_ratio": CHUNK_OVERLAP_RATIO,
                "chunk_overlap_tokens": overlap_tokens,
            }
            chunks.append(Document(page_content=page, metadata=metadata))
            previous = text
    return chunks


def metadata_rows(documents: List[Document]) -> List[dict]:
    keys = ["source", "document_id", "policy_category", "policy_owner", "effective_date", "h1_header", "h2_header"]
    return [{key: doc.metadata.get(key, "") for key in keys} for doc in documents]


def data_fingerprint() -> Tuple[Tuple[str, int, int], ...]:
    if not DATA_DIR.exists():
        return tuple()
    return tuple(sorted((str(p.relative_to(DATA_DIR)), p.stat().st_size, p.stat().st_mtime_ns) for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS))


@st.cache_resource(show_spinner=False)
def build_index(upload_payload, repo_fingerprint, use_openai_embeddings, metadata_key_hash, metadata_key) -> Optional[AppIndex]:
    del repo_fingerprint, metadata_key_hash
    documents = load_documents(upload_payload, metadata_key=metadata_key)
    if not documents:
        return None
    chunks = split_documents(documents)
    if not chunks:
        return None
    embeddings = get_embeddings(use_openai_embeddings)
    main_index = FAISS.from_documents(chunks, embeddings)
    category_indexes = {}
    for category in sorted({chunk.metadata.get("policy_category", "General") for chunk in chunks}):
        cat_chunks = [chunk for chunk in chunks if chunk.metadata.get("policy_category", "General") == category]
        category_indexes[category] = FAISS.from_documents(cat_chunks, embeddings)
    return AppIndex(
        vectorstore=main_index,
        category_indexes=category_indexes,
        sources=sorted({doc.metadata.get("source", "Unknown") for doc in documents}),
        metadata_rows=metadata_rows(documents),
        chunk_count=len(chunks),
    )


def content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", (text or "").lower()) if w not in STOPWORDS}


def choose_category(query: str, available: Sequence[str]) -> dict:
    terms = {
        "Travel": ["travel", "taxi", "cab", "uber", "lyft", "rideshare", "flight", "hotel", "per diem", "ground transportation"],
        "HR": ["hr", "maternity", "paternity", "leave", "pto", "employee", "benefit", "sabbatical", "payroll"],
        "Finance": ["finance", "budget", "invoice", "reimbursement", "expense report", "approval limit"],
        "Legal": ["legal", "contract", "compliance", "privacy", "regulatory"],
        "IT": ["it policy", "information technology", "security", "vpn", "password", "laptop", "device", "software"],
    }
    lower = (query or "").lower()

    def has(term: str) -> bool:
        escaped = re.escape(term.lower())
        return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lower))

    scores = {cat: sum(1 for term in cat_terms if has(term)) for cat, cat_terms in terms.items()}
    category, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return {"category": "", "confidence": 0.0, "reason": "no metadata category signal", "applied": False}
    return {
        "category": category,
        "confidence": min(0.95, 0.55 + score * 0.15),
        "reason": "metadata category signal",
        "applied": True,
        "available_categories": list(available),
    }


def expansion_fallback(query: str) -> List[str]:
    lower = query.lower()
    expansions = [
        f"Corporate policy guidance for {query}",
        f"Eligibility rules, limits, reimbursement terms, approvals, and exceptions for {query}",
    ]
    if "allowance" in lower:
        expansions += ["Travel allowance per diem reimbursement limits", "Leave PTO sabbatical benefit allowance"]
    if any(w in lower for w in ["taxi", "cab", "uber", "lyft", "rideshare"]):
        expansions += ["Ground transportation reimbursement policy", "Rideshare taxi cab fare corporate travel"]
    seen, out = set(), []
    for item in expansions:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
        if len(out) >= EXPANSION_COUNT:
            break
    return out


def query_pack(query: str, llm) -> dict:
    expansions = expansion_fallback(query)
    hyde = f"A corporate policy section answering: {query}. It states eligibility, limits, approvals, exceptions, effective date, and reimbursement or benefit rules."
    if llm:
        try:
            raw = llm.invoke(f"Return only a JSON array with {EXPANSION_COUNT} alternative search queries for this policy question: {query}")
            data = json.loads(raw)
            if isinstance(data, list):
                expansions = [str(x) for x in data[:EXPANSION_COUNT] if str(x).strip()]
        except Exception:
            pass
        try:
            hyde = llm.invoke(f"Write one short hypothetical policy paragraph that would answer this question. This is only for retrieval: {query}")
        except Exception:
            pass
    return {"raw": query, "expansions": expansions, "hyde": hyde}


def effective_date(doc: Document) -> str:
    date = str(doc.metadata.get("effective_date", ""))
    return date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else "0000-00-00"


def family(doc: Document) -> Tuple[str, str]:
    meta = doc.metadata
    doc_id = re.sub(r"[-_]?V\d+\b", "", str(meta.get("document_id", "")), flags=re.IGNORECASE)
    title = str(meta.get("h1_header") or meta.get("source") or "").lower()
    return str(meta.get("policy_category", "General")), doc_id or title


def keep_latest(docs: List[Document]) -> Tuple[List[Document], dict]:
    latest = {}
    for doc in docs:
        latest[family(doc)] = max(latest.get(family(doc), "0000-00-00"), effective_date(doc))
    kept, dropped = [], 0
    for doc in docs:
        if effective_date(doc) == latest[family(doc)]:
            kept.append(doc)
        else:
            dropped += 1
    return kept, {"dropped_older_versions": dropped, "latest_by_family": {" | ".join(k): v for k, v in latest.items()}}


def doc_key(doc: Document) -> Tuple[str, str, str]:
    return str(doc.metadata.get("source", "")), str(doc.metadata.get("chunk_index", "")), hashlib.sha256(doc.page_content.encode()).hexdigest()


def rerank_text(doc: Document) -> str:
    meta = doc.metadata
    return "\n".join([
        f"Source: {meta.get('source')}",
        f"Category: {meta.get('policy_category')}",
        f"Effective date: {meta.get('effective_date')}",
        f"Section: {meta.get('section_header')}",
        "",
        doc.page_content,
    ])


def cohere_rerank(query: str, docs: List[Document], api_key: str, model: str) -> Tuple[List[Document], dict]:
    response = requests.post(
        COHERE_RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model or DEFAULT_COHERE_RERANK_MODEL, "query": query, "documents": [rerank_text(d) for d in docs], "top_n": min(RERANK_TOP_N, len(docs)), "max_tokens_per_doc": 1200},
        timeout=45,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    scores, ranked = [], []
    for rank, item in enumerate(results, start=1):
        index = int(item.get("index", -1))
        if 0 <= index < len(docs):
            doc = docs[index]
            score = float(item.get("relevance_score", 0.0))
            ranked.append(doc)
            scores.append({"rank": rank, "score": round(score, 4), "source": doc.metadata.get("source"), "section": doc.metadata.get("section_header")})
    return ranked, {"provider": "Cohere ReRank", "scores": scores}


def lexical_rerank(query: str, docs: List[Document]) -> Tuple[List[Document], dict]:
    q = content_tokens(query)
    ranked = []
    for doc in docs:
        d = content_tokens(rerank_text(doc))
        score = len(q & d) / max(1, len(q))
        ranked.append((doc, round(score, 4)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    kept = ranked[:RERANK_TOP_N]
    return [doc for doc, _ in kept], {
        "provider": "Lexical fallback",
        "scores": [{"rank": i + 1, "score": score, "source": doc.metadata.get("source"), "section": doc.metadata.get("section_header")} for i, (doc, score) in enumerate(kept)],
    }


def rerank(query: str, docs: List[Document], provider: str, cohere_key: str, cohere_model: str) -> Tuple[List[Document], dict]:
    if not docs:
        return [], {"provider": provider, "input_chunks": 0, "output_chunks": 0, "scores": []}
    if provider in {"Auto", "Cohere ReRank"} and cohere_key:
        try:
            ranked, trace = cohere_rerank(query, docs, cohere_key, cohere_model)
        except Exception as exc:
            ranked, trace = lexical_rerank(query, docs)
            trace["fallback_reason"] = str(exc)
    else:
        ranked, trace = lexical_rerank(query, docs)
    trace.update({"input_chunks": len(docs), "output_chunks": len(ranked), "top_n": RERANK_TOP_N})
    return ranked, trace


def header(meta: dict) -> str:
    return "[Source: {source} | Category: {policy_category} | Owner: {policy_owner} | Effective: {effective_date} | Section: {section_header}]".format(
        source=meta.get("source", "Unknown"),
        policy_category=meta.get("policy_category", "General"),
        policy_owner=meta.get("policy_owner", "Unknown"),
        effective_date=meta.get("effective_date", ""),
        section_header=meta.get("section_header") or meta.get("h1_header") or "",
    )


def retrieve(index: AppIndex, query: str, llm, reranker_provider: str, cohere_key: str, cohere_model: str) -> Tuple[str, List[str], dict]:
    pack = query_pack(query, llm)
    route = choose_category(query, index.category_indexes.keys())
    if route["applied"] and route["category"] not in index.category_indexes:
        return "", [], {"query_pack": pack, "metadata_prefilter": route, "unique_chunks": 0, "reranker": {"provider": "skipped", "input_chunks": 0, "output_chunks": 0}, "context_chunks": 0}

    vectorstore = index.category_indexes.get(route["category"], index.vectorstore) if route["applied"] else index.vectorstore
    seen, pooled, runs = set(), [], []
    for kind, search_text in [("raw", pack["raw"]), *[(f"expansion_{i + 1}", q) for i, q in enumerate(pack["expansions"])], ("hyde", pack["hyde"])]:
        docs = vectorstore.similarity_search(search_text, k=BROAD_RETRIEVAL_K)
        runs.append({"type": kind, "retrieved": len(docs), "sources": sorted({d.metadata.get("source") for d in docs})})
        for doc in docs:
            key = doc_key(doc)
            if key not in seen:
                pooled.append(doc)
                seen.add(key)

    latest_docs, date_trace = keep_latest(pooled)
    broad = latest_docs[:BROAD_RETRIEVAL_K]
    final_docs, rerank_trace = rerank(query, broad, reranker_provider, cohere_key, cohere_model)
    context = "\n\n".join(f"{header(doc.metadata)}\n{doc.page_content}" for doc in final_docs)
    sources = sorted({doc.metadata.get("source", "Unknown") for doc in final_docs})
    trace = {
        "query_pack": pack,
        "metadata_prefilter": {**route, "matched_chunks": len(pooled)},
        "date_post_filter": date_trace,
        "broad_k": BROAD_RETRIEVAL_K,
        "unique_chunks": len(pooled),
        "rerank_candidates": len(broad),
        "reranker": rerank_trace,
        "context_chunks": len(final_docs),
        "runs": runs,
    }
    return context, sources, trace


def sentiment(text: str) -> dict:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    if neg > pos:
        return {"label": "negative", "tone": "empathetic, calm, and reassuring"}
    if pos > neg:
        return {"label": "positive", "tone": "warm, confident, and concise"}
    return {"label": "neutral", "tone": "clear, professional, and human"}


def safe_decimal_eval(expression: str) -> Decimal:
    operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** int(b),
    }

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Unsupported calculation.")

    return walk(ast.parse(expression, mode="eval"))


def calculation(query: str) -> Optional[dict]:
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", query, re.I)
    expression = f"({percent.group(1)}/100)*{percent.group(2)}" if percent else ""
    if not expression:
        pieces = re.findall(r"[\d\s\.\+\-\*\/\(\)%]{3,}", query)
        expression = max(pieces, key=len).strip() if pieces else ""
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression).replace("x", "*")
    expression = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expression).strip()
    if not expression or not any(op in expression for op in "+-*/"):
        return None
    try:
        value = safe_decimal_eval(expression)
    except (SyntaxError, ValueError, InvalidOperation, DivisionByZero, OverflowError):
        return None
    return {"expression": expression, "result": format(value.normalize(), "f")}


class LLM:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def invoke(self, prompt: str) -> str:
        config = PROVIDERS[self.provider]
        kind = config["kind"]
        if kind == "extractive":
            raise RuntimeError("Extractive provider has no hosted LLM.")
        if kind == "ollama":
            response = requests.post("http://localhost:11434/api/chat", json={"model": self.model, "stream": False, "messages": [{"role": "user", "content": prompt}]}, timeout=45)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        if kind == "gemini":
            key = provider_key(self.provider)
            response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", params={"key": key}, json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}, timeout=90)
            response.raise_for_status()
            parts = response.json()["candidates"][0]["content"].get("parts", [])
            return "\n".join(part.get("text", "") for part in parts).strip()
        if not OpenAI:
            raise RuntimeError("openai package is required for this provider.")
        key = provider_key(self.provider)
        kwargs = {"api_key": key, "timeout": 60.0, "max_retries": 0}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(model=self.model, temperature=0, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""


def provider_available(provider: str) -> bool:
    if provider in {"Extractive", "Ollama"}:
        return True
    return bool(provider_key(provider))


def choose_provider(question: str, calc: Optional[dict], mood: dict) -> str:
    order = ["OpenAI", "Grok", "Gemini", "Mistral", "OpenRouter", "Hugging Face"] if calc else ["OpenAI", "Gemini", "Grok", "Hugging Face", "Mistral", "OpenRouter"]
    if mood["label"] == "negative":
        order = ["Grok", "OpenAI", "Gemini", "OpenRouter", "Hugging Face"]
    return next((p for p in order if provider_available(p)), "Extractive")


def model_for(provider: str) -> str:
    return next(model for model in PROVIDERS[provider]["models"] if model not in {"custom", "Auto choose"})


def extractive_answer(context: str, query: str, calc: Optional[dict]) -> str:
    if not context.strip():
        return STRICT_NOT_FOUND
    q = content_tokens(query)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    scored = sorted(((len(q & content_tokens(s)), s.strip()) for s in sentences if s.strip()), reverse=True)
    useful = [s for score, s in scored if score][:4]
    if not useful and not calc:
        return STRICT_NOT_FOUND
    lines = ["Based on the policy:"]
    lines += [f"- {line}" for line in useful]
    if calc:
        lines.append(f"- Precise calculation: {calc['expression']} = {calc['result']}")
    return "\n".join(lines)


def answer_prompt(context: str, tavily_context: str, query: str, mood: dict, calc: Optional[dict]) -> str:
    calc_text = f"{calc['expression']} = {calc['result']}" if calc else "not required"
    return f"""<context>
{context}
</context>

<tavily_context>
{tavily_context}
</tavily_context>

Question: {query}

Rules:
- Answer ONLY from context, tavily_context, and verified calculation.
- Prefer policy context over tavily_context.
- If not found, say exactly: "{STRICT_NOT_FOUND}"
- Use a {mood['tone']} tone.
- Precise calculation: {calc_text}
"""


def critic_prompt(context: str, query: str, answer: str) -> str:
    return f"""Return strict JSON with keys faithfulness, hallucination, missing_info, feedback.

<context>
{context}
</context>

Question: {query}
Answer: {answer}
"""


def validated_answer(llm: Optional[LLM], context: str, tavily_context: str, query: str, mood: dict, calc: Optional[dict]) -> Tuple[str, dict]:
    if not context.strip() and not tavily_context.strip() and not calc:
        return STRICT_NOT_FOUND, {"faithfulness": True, "hallucination": False, "missing_info": False, "feedback": "No context."}
    if not llm:
        return extractive_answer(context, query, calc), {"faithfulness": True, "hallucination": False, "missing_info": False, "feedback": "Extractive answer."}

    answer = llm.invoke(answer_prompt(context, tavily_context, query, mood, calc))
    for _ in range(2):
        try:
            review = json.loads(llm.invoke(critic_prompt(context + "\n" + tavily_context, query, answer)).strip().strip("`").replace("json", "", 1))
        except Exception:
            review = {"faithfulness": False, "hallucination": True, "missing_info": False, "feedback": "Use only supplied context."}
        if review.get("faithfulness") and not review.get("hallucination") and not review.get("missing_info"):
            return answer, review
        answer = llm.invoke(answer_prompt(context, tavily_context, f"{query}\nCritic feedback: {review.get('feedback', '')}", mood, calc))
    return answer, review


def tavily_search(query: str, enabled: bool, api_key: str) -> Tuple[str, List[str], dict]:
    if not enabled:
        return "", [], {"status": "disabled"}
    if not api_key:
        return "", [], {"status": "missing_key"}
    try:
        response = requests.post(TAVILY_URL, json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 3, "include_answer": True}, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        context = "\n\n".join([data.get("answer", "")] + [r.get("content", "") for r in results if r.get("content")]).strip()
        sources = [r.get("url", "") for r in results if r.get("url")]
        return context, sources, {"status": "complete", "results": len(results)}
    except Exception as exc:
        return "", [], {"status": "error", "error": str(exc)}


def payload_hash(items: Tuple[Tuple[str, str, bytes], ...]) -> str:
    h = hashlib.sha256()
    for name, content_type, content in items:
        h.update(name.encode())
        h.update(content_type.encode())
        h.update(hashlib.sha256(content).digest())
    return h.hexdigest()


def add_uploads(files) -> None:
    st.session_state.setdefault("uploads", [])
    existing = {(name, hashlib.sha256(content).hexdigest()) for name, _typ, content in st.session_state.uploads}
    for file in files or []:
        content = file.getvalue()
        key = (file.name, hashlib.sha256(content).hexdigest())
        if key not in existing:
            st.session_state.uploads.append((file.name, file.type, content))


def render_key(provider: str) -> None:
    if provider in {"Auto", "Extractive", "Ollama"}:
        return
    st.text_input(f"{provider} API Key", type="password", key=f"{provider.lower().replace(' ', '_')}_api_key")


def select_model(provider: str) -> str:
    choice = st.selectbox("Model", PROVIDERS[provider]["models"], key=f"{provider}_model")
    return st.text_input("Custom model", key=f"{provider}_custom") if choice == "custom" else choice


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)
    st.caption("Simple policy RAG: upload, index, filter, rerank, answer.")

    st.session_state.setdefault("uploads", [])

    with st.sidebar:
        st.header("Documents")
        incoming = st.file_uploader("Upload Policy Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
        if st.button("Add Selected Files"):
            add_uploads(incoming)
        if st.button("Clear Uploaded Files"):
            st.session_state.uploads = []
            st.session_state.pop("index_key", None)
        st.write(f"{len(st.session_state.uploads)} files uploaded")

        st.header("LLM")
        provider_choice = st.selectbox("Provider", list(PROVIDERS), index=0)
        if provider_choice == "Auto":
            with st.expander("API Keys"):
                for provider in ["OpenAI", "Grok", "Gemini", "Mistral", "OpenRouter", "Hugging Face"]:
                    render_key(provider)
            model_choice = "Auto choose"
        else:
            render_key(provider_choice)
            model_choice = select_model(provider_choice)

        use_openai_embeddings = st.checkbox("Use OpenAI embeddings", value=bool(provider_key("OpenAI")) and OpenAIEmbeddings is not None)

        st.header("Reranker")
        reranker_provider = st.selectbox("Reranker", ["Auto", "Cohere ReRank", "Lexical fallback"], index=0)
        st.text_input("Cohere API Key", type="password", key="cohere_key")
        cohere_model = st.text_input("Cohere model", value=DEFAULT_COHERE_RERANK_MODEL)

        st.header("Tavily")
        tavily_enabled = st.toggle("Use Tavily", value=False)
        st.text_input("tavily_key", type="password", key="tavily_key")

        process = st.button("Process Documents", type="primary")

    if process and incoming:
        add_uploads(incoming)

    uploads = tuple(st.session_state.uploads)
    metadata_key = provider_key("OpenAI")
    index_key = (payload_hash(uploads), data_fingerprint(), use_openai_embeddings, hashlib.sha256(metadata_key.encode()).hexdigest())

    if process or st.session_state.get("index_key") != index_key:
        with st.spinner("Indexing documents..."):
            index = build_index(uploads, data_fingerprint(), use_openai_embeddings, index_key[-1], metadata_key)
            st.session_state.index = index
            st.session_state.index_key = index_key

    index: Optional[AppIndex] = st.session_state.get("index")
    if not index:
        st.warning("No policy documents available. Please upload files or add PDF, DOCX, TXT, or MD files to ./data.")
        return

    with st.sidebar:
        st.markdown("### Indexed")
        st.write(f"Sources: {len(index.sources)}")
        st.write(f"Chunks: {index.chunk_count}")
        st.caption(f"Categories: {', '.join(sorted(index.category_indexes))}")
        with st.expander("Source Metadata"):
            st.dataframe(index.metadata_rows, use_container_width=True)

    query = st.chat_input("Ask a question across all policy documents")
    if not query:
        st.info("Ask a policy question when ready.")
        return

    st.chat_message("user").write(query)

    mood = sentiment(query)
    calc = calculation(query)
    active_provider = choose_provider(query, calc, mood) if provider_choice == "Auto" else provider_choice
    active_model = model_for(active_provider) if provider_choice == "Auto" else model_choice
    llm = None if active_provider == "Extractive" else LLM(active_provider, active_model)

    with st.spinner("Retrieving and reranking..."):
        context, sources, retrieval_trace = retrieve(
            index=index,
            query=query,
            llm=llm,
            reranker_provider=reranker_provider,
            cohere_key=tool_key("cohere", ["COHERE_API_KEY", "COHERE_KEY"]),
            cohere_model=cohere_model,
        )
        tavily_context, tavily_sources, tavily_trace = tavily_search(query, tavily_enabled, tool_key("tavily", ["TAVILY_KEY", "TAVILY_API_KEY", "tavily_key"]))

    with st.spinner("Generating answer..."):
        try:
            answer, critic = validated_answer(llm, context, tavily_context, query, mood, calc)
        except Exception as exc:
            answer = extractive_answer(context, query, calc)
            critic = {"faithfulness": True, "hallucination": False, "missing_info": False, "feedback": f"LLM failed, used extractive fallback: {exc}"}

    trace = [
        {"node": "Ingestion", "detail": f"{len(index.sources)} source(s), {index.chunk_count} chunk(s)"},
        {"node": "Metadata Pre-Filter", "detail": retrieval_trace["metadata_prefilter"]},
        {"node": "Query Expansion + HyDE", "detail": retrieval_trace["query_pack"]},
        {"node": "Broad Retrieval", "detail": f"Top {BROAD_RETRIEVAL_K}, pooled {retrieval_trace['unique_chunks']} unique chunk(s)"},
        {"node": "Date Post-Filter", "detail": retrieval_trace.get("date_post_filter", {})},
        {"node": "Reranker", "detail": retrieval_trace["reranker"]},
        {"node": "Answer Model", "detail": f"{active_provider} / {active_model}"},
        {"node": "Tavily", "detail": tavily_trace},
    ]

    with st.chat_message("assistant"):
        st.markdown("### Answer")
        st.write(answer)
        st.info(f"Tone: {mood['tone']} | Provider: {active_provider} | Model: {active_model}")
        if calc:
            st.success(f"Precise calculation: {calc['expression']} = {calc['result']}")

        st.markdown("### Sources")
        for source in sources:
            st.write(f"- {source}")
        if tavily_sources:
            st.markdown("### Tavily Sources")
            for source in tavily_sources:
                st.write(f"- {source}")

        with st.expander("Query, Filters, Reranker"):
            st.json(retrieval_trace)
        with st.expander("Tools and Nodes"):
            st.json(trace)
        with st.expander("Critic"):
            st.json(critic)


if __name__ == "__main__":
    main()
