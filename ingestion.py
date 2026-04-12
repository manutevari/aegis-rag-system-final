# =============================================================================
# AEGIS — INGESTION ENGINE (LangChain + LangGraph Edition)
# ingestion.py
#
# LangChain components:
#   ChatOpenAI + StructuredOutputParser  — typed metadata extraction
#   OpenAIEmbeddings                     — text-embedding-3-large
#   MarkdownHeaderTextSplitter           — H1/H2/H3 boundary detection
#   RecursiveCharacterTextSplitter       — token-budget + 12.5% overlap
#   PineconeVectorStore                  — upsert via add_documents()
#
# Message types (imported from graph_state):
#   SystemMessage  — extraction prompt description
#   HumanMessage   — document snippet fed to LLM
#   AIMessage      — raw LLM extraction reply
#   ToolMessage    — audit log entry (tool_log helper)
#
# Pydantic enforcement:
#   ChunkMetadata validates every chunk before embedding.
#   extract_metadata returns a validated dict or safe defaults — never None fields.
# =============================================================================

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator

from graph_state import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    VALID_CATEGORIES,
    tool_log,
)

# ---------------------------------------------------------------------------
# Lazy LangChain clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                      api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL,
                            api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _vector_store():
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return PineconeVectorStore(
        index_name=INDEX_NAME, embedding=_embeddings(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL     = "text-embedding-3-large"
EMBED_DIM       = 3072
INDEX_NAME      = "aegis-index"
BATCH_SIZE      = 100
OVERLAP_RATIO   = 0.125
MAX_CHUNK_CHARS = 1600      # ~400 tokens × 4 chars/token

# ---------------------------------------------------------------------------
# Pydantic chunk model
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    document_id:     str  = Field(..., description="Unique document identifier")
    policy_category: str  = Field(..., description="Travel|HR|Finance|Legal|IT|General")
    policy_owner:    str  = Field(default="Unknown")
    effective_date:  str  = Field(default="")
    h1_header:       str  = Field(default="")
    h2_header:       str  = Field(default="")
    chunk_text:      str  = Field(..., min_length=1)
    is_table:        bool = Field(default=False)

    @field_validator("policy_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        return v if v in VALID_CATEGORIES else "General"

    @field_validator("chunk_text")
    @classmethod
    def strip_chunk(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("chunk_text must not be blank")
        return stripped

# ---------------------------------------------------------------------------
# Metadata extraction — LangChain StructuredOutputParser chain
# Returns (metadata_dict, list[BaseMessage]) so callers can log messages
# ---------------------------------------------------------------------------

_META_SYSTEM_TEXT = (
    "You are a metadata extraction assistant for corporate policy documents.\n"
    "Extract the requested fields from the document snippet.\n"
    "{format_instructions}"
)

def _build_meta_chain():
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    schemas = [
        ResponseSchema(name="document_id",
                       description='Short code like "TRV-POL-2025-V1"'),
        ResponseSchema(name="policy_category",
                       description="One of: Travel|HR|Finance|Legal|IT|General"),
        ResponseSchema(name="policy_owner",
                       description='Responsible department, e.g. "GCT-RM"'),
        ResponseSchema(name="effective_date",
                       description="ISO date YYYY-MM-DD; today if not found"),
        ResponseSchema(name="h1_header",  description="Top-level document title"),
        ResponseSchema(name="h2_header",  description="Most prominent sub-section heading"),
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    prompt = (
        ChatPromptTemplate.from_messages([
            ("system", _META_SYSTEM_TEXT),
            ("human",  "{snippet}"),
        ])
        .partial(format_instructions=parser.get_format_instructions())
    )
    return prompt | _llm() | parser


def extract_metadata(text: str) -> tuple[dict, list]:
    """
    Run LangChain StructuredOutputParser chain to extract metadata.

    Returns:
        (metadata_dict, messages)
        where messages is [SystemMessage, HumanMessage, AIMessage, ToolMessage]
        ready to be appended to AegisState.messages.
    """
    snippet = text[:3000]

    sys_msg  = SystemMessage(content=_META_SYSTEM_TEXT.split("{format_instructions}")[0].strip())
    user_msg = HumanMessage(content=f"[Document snippet — {len(snippet)} chars]\n{snippet[:200]}…")

    try:
        chain  = _build_meta_chain()
        data   = chain.invoke({"snippet": snippet})
        ai_msg = AIMessage(content=str(data))
        log    = tool_log(
            tool_name="extract_metadata",
            reason="Extract structured metadata (document_id, category, owner, date, headers) "
                   "from the document's opening text before chunking.",
            inputs={"snippet_chars": len(snippet)},
            outputs=data,
        )
        success = True
    except Exception as exc:
        data   = {}
        ai_msg = AIMessage(content=f"extraction failed: {exc}")
        log    = tool_log(
            tool_name="extract_metadata",
            reason="Metadata extraction attempted; fell back to defaults.",
            inputs={"snippet_chars": len(snippet)},
            outputs={"error": str(exc)},
        )
        success = False

    # Build validated metadata dict
    cat = data.get("policy_category", "General")
    if cat not in VALID_CATEGORIES:
        cat = "General"

    meta = {
        "document_id":     data.get("document_id",    f"DOC-{uuid.uuid4().hex[:6].upper()}"),
        "policy_category": cat,
        "policy_owner":    data.get("policy_owner",   "Unknown"),
        "effective_date":  data.get("effective_date", str(datetime.now().date())),
        "h1_header":       data.get("h1_header",      ""),
        "h2_header":       data.get("h2_header",      ""),
    }

    return meta, [sys_msg, user_msg, ai_msg, log]

# ---------------------------------------------------------------------------
# Markdown-Aware Chunking (MarkdownHeaderTextSplitter + RecursiveChar)
# ---------------------------------------------------------------------------

def _build_splitters():
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1_header"),
            ("##",  "h2_header"),
            ("###", "h2_header"),
        ],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=int(MAX_CHUNK_CHARS * OVERLAP_RATIO),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return header_splitter, char_splitter


def _extract_tables(text: str) -> tuple[str, list[tuple[str, str]]]:
    pattern = re.compile(r"(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)*)", re.MULTILINE)
    tables: list[tuple[str, str]] = []
    result = text
    for i, m in enumerate(pattern.finditer(text)):
        ph = f"__TABLE_{i}__"
        tables.append((ph, m.group(0)))
        result = result.replace(m.group(0), ph + "\n")
    return result, tables


def _split_table_by_rows(header_row: str, data_rows: list[str],
                          max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for row in data_rows:
        candidate = header_row + "\n" + "\n".join(current + [row])
        if len(candidate) > max_chars and current:
            chunks.append(header_row + "\n" + "\n".join(current))
            current = [row]
        else:
            current.append(row)
    if current:
        chunks.append(header_row + "\n" + "\n".join(current))
    return chunks


def chunk_document(text: str, base_meta: dict) -> list[ChunkMetadata]:
    """
    Phase 1: MarkdownHeaderTextSplitter → section boundaries
    Phase 2: RecursiveCharacterTextSplitter → token budget + overlap
    Phase 3: Table preservation (atomic; row-split if oversized)
    All chunks validated through ChunkMetadata Pydantic model.
    """
    text_no_tables, tables = _extract_tables(text)
    table_map = dict(tables)
    header_splitter, char_splitter = _build_splitters()

    try:
        header_docs = header_splitter.split_text(text_no_tables)
    except Exception:
        from langchain_core.documents import Document
        header_docs = [Document(page_content=text_no_tables, metadata={})]

    base_fields = {k: v for k, v in base_meta.items()
                   if k not in ("h1_header", "h2_header")}
    result: list[ChunkMetadata] = []

    for hdoc in header_docs:
        body = hdoc.page_content.strip()
        if not body:
            continue
        h1  = hdoc.metadata.get("h1_header", base_meta.get("h1_header", ""))
        h2  = hdoc.metadata.get("h2_header", "")
        ctx = (f"[{h1}] [{h2}]\n" if h2 else f"[{h1}]\n") if (h1 or h2) else ""

        # ── Tables ──────────────────────────────────────────────────────
        for ph, table_text in table_map.items():
            if ph not in body:
                continue
            full = ctx + table_text
            if len(full) <= MAX_CHUNK_CHARS:
                try:
                    result.append(ChunkMetadata(**base_fields, h1_header=h1,
                                                h2_header=h2, chunk_text=full,
                                                is_table=True))
                except Exception:
                    pass
            else:
                lines = table_text.strip().split("\n")
                if len(lines) >= 2:
                    hdr = lines[0] + "\n" + lines[1]
                    for sub in _split_table_by_rows(ctx + hdr, lines[2:], MAX_CHUNK_CHARS):
                        try:
                            result.append(ChunkMetadata(**base_fields, h1_header=h1,
                                                        h2_header=h2, chunk_text=sub,
                                                        is_table=True))
                        except Exception:
                            pass
            body = body.replace(ph, "").strip()

        if not body:
            continue

        # ── Prose ────────────────────────────────────────────────────────
        for chunk_text in char_splitter.split_text(ctx + body if ctx else body):
            try:
                result.append(ChunkMetadata(**base_fields, h1_header=h1,
                                            h2_header=h2,
                                            chunk_text=chunk_text.strip(),
                                            is_table=False))
            except Exception:
                pass

    return result

# ---------------------------------------------------------------------------
# Embed + Upsert via LangChain PineconeVectorStore
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _embeddings().embed_documents(texts)


def embed_and_store(chunks: list[ChunkMetadata]) -> tuple[int, ToolMessage]:
    """
    Embed via OpenAIEmbeddings and upsert via PineconeVectorStore.
    Returns (total_upserted, ToolMessage audit log).
    """
    from langchain_core.documents import Document
    if not chunks:
        log = tool_log("embed_and_store", "No chunks to store.", {}, {"upserted": 0})
        return 0, log

    lc_docs = [Document(page_content=c.chunk_text, metadata=c.model_dump())
               for c in chunks]
    ids     = [f"{c.document_id}_{i}" for i, c in enumerate(chunks)]
    vs      = _vector_store()
    total   = 0

    for start in range(0, len(lc_docs), BATCH_SIZE):
        vs.add_documents(documents=lc_docs[start: start + BATCH_SIZE],
                         ids=ids[start: start + BATCH_SIZE])
        total += len(lc_docs[start: start + BATCH_SIZE])

    log = tool_log(
        tool_name="embed_and_store",
        reason=f"Embed {len(chunks)} validated chunks with {EMBED_MODEL} "
               f"and upsert to Pinecone index '{INDEX_NAME}' in batches of {BATCH_SIZE}.",
        inputs={"chunk_count": len(chunks), "index": INDEX_NAME},
        outputs={"upserted": total, "batch_size": BATCH_SIZE},
    )
    return total, log


upsert_chunks = embed_and_store   # backwards-compat alias

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_document(text: str) -> dict:
    """
    Full ingestion pipeline. Returns summary dict + messages for audit log.
    """
    base_meta, meta_messages = extract_metadata(text)
    chunks  = chunk_document(text, base_meta)
    upserted, store_log = embed_and_store(chunks)
    return {
        "document_id": base_meta["document_id"],
        "category":    base_meta["policy_category"],
        "chunks":      len(chunks),
        "upserted":    upserted,
        "messages":    meta_messages + [store_log],
    }
