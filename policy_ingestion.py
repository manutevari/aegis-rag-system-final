"""Canonical policy ingestion entrypoint.

The ingestion engine follows the Project Aegis guidelines: it parses policy
files by Markdown section, preserves tables as atomic blocks, applies a small
sequential overlap, enriches every chunk with structured policy metadata, and
indexes the verified chunks through the shared vector store.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.core.vector_store import get_vectorstore, index_documents

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("POLICY_DIR", "data")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_CHUNK_TOKENS = 450
DEFAULT_TABLE_ROW_CHUNK = 18
DEFAULT_OVERLAP_RATIO = 0.12

CATEGORY_KEYWORDS = {
    "travel": ["travel", "fuel", "mileage", "vehicle", "hotel", "flight", "rental", "taxi", "rideshare"],
    "security": ["security", "privacy", "data", "vpn", "incident", "cyber"],
    "hr": ["hr", "conduct", "leave", "absence", "discipline", "work policies", "maternity"],
    "training": ["training", "learning", "tuition", "certification"],
    "compensation": ["compensation", "salary", "pay", "hra", "bonus", "performance"],
}

GRADE_MAP = {
    "l1": 1,
    "l2": 2,
    "l3": 3,
    "l4": 4,
    "l5": 5,
    "l6": 6,
    "l7": 7,
    "executive": 8,
    "vp": 8,
    "svp": 9,
    "cxo": 10,
}

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
FIELD_RE = re.compile(r"^\s*\*\*(?P<name>[^:*]+):\*\*\s*(?P<value>.+?)\s*$")
TOKEN_RE = re.compile(r"\S+")

REQUIRED_METADATA = [
    "document_id",
    "policy_category",
    "policy_owner",
    "effective_date",
    "h1_header",
    "h2_header",
    "source_path",
]


def _normalise_signal(path: Path, text: str = "") -> str:
    return f"{path.as_posix()} {text[:2000]}".lower()


def detect_category(path: Path, text: str = "") -> str:
    signal = _normalise_signal(path, text)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in signal for keyword in keywords):
            return category
    return "general"


def detect_grade(path: Path, text: str = "") -> int:
    signal = _normalise_signal(path, text)
    for label, grade in GRADE_MAP.items():
        if re.search(rf"\b{re.escape(label)}\b", signal):
            return grade
    return 3


def iter_policy_files(
    data_path: str = DATA_PATH,
    file_paths: Optional[Iterable[str]] = None,
) -> List[Path]:
    if file_paths is not None:
        return [Path(path) for path in file_paths if Path(path).exists()]

    root = Path(data_path)
    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_text(path: Path) -> List[Document]:
    try:
        return TextLoader(str(path), encoding="utf-8").load()
    except UnicodeDecodeError:
        return TextLoader(str(path), encoding="latin-1").load()


def _load_file(path: Path) -> List[Document]:
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    return _load_text(path)


def _normalise_date(value: str) -> str:
    clean = (value or "").strip().rstrip(".")
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            pass
    return clean


def _document_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in text.splitlines()[:40]:
        match = FIELD_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip().lower().replace(" ", "_")
        value = match.group("value").strip()
        if name in {"effective_date", "last_revised"}:
            value = _normalise_date(value)
        fields[name] = value
    return fields


def _first_header(text: str, level: int) -> str:
    prefix = "#" * level
    for line in text.splitlines():
        match = HEADER_RE.match(line)
        if match and match.group(1) == prefix:
            return match.group(2).strip()
    return ""


def _base_metadata(path: Path, root: Path, text: str) -> Dict[str, Any]:
    fields = _document_fields(text)
    source_path = path.as_posix()
    try:
        source_path = path.relative_to(root).as_posix()
    except ValueError:
        pass

    h1_header = _first_header(text, 1) or path.stem
    return {
        "source": path.name,
        "source_path": source_path,
        "document_id": fields.get("document_id") or path.stem,
        "policy_category": detect_category(path, text),
        "policy_owner": fields.get("policy_owner") or "unknown",
        "effective_date": fields.get("effective_date") or "unknown",
        "last_revised": fields.get("last_revised") or "unknown",
        "applies_to": fields.get("applies_to") or "unknown",
        "h1_header": h1_header,
        "grade_level": detect_grade(path, text),
    }


def load_documents(
    data_path: str = DATA_PATH,
    file_paths: Optional[Iterable[str]] = None,
) -> List[Document]:
    root = Path(data_path)
    documents: List[Document] = []

    for path in iter_policy_files(data_path=data_path, file_paths=file_paths):
        try:
            docs = _load_file(path)
            raw_text = "\n\n".join(doc.page_content for doc in docs)
            metadata = _base_metadata(path, root, raw_text)
            combined = Document(page_content=raw_text, metadata=metadata)
            documents.append(combined)
        except Exception as exc:
            logger.warning("Could not load policy file %s: %s", path, exc)

    if not documents:
        from app.tools._sample_policies import get_sample_docs

        logger.warning("No policy files loaded from %s; using built-in sample policies", data_path)
        for index, doc in enumerate(get_sample_docs()):
            text = getattr(doc, "page_content", str(doc))
            metadata = dict(getattr(doc, "metadata", {}) or {})
            source = metadata.get("source", f"sample-policy-{index + 1}")
            metadata.setdefault("source", source)
            metadata.setdefault("source_path", source)
            metadata.setdefault("document_id", metadata.get("policy_code") or source)
            metadata.setdefault("policy_category", detect_category(Path(source), text))
            metadata.setdefault("policy_owner", "sample")
            metadata.setdefault("effective_date", "unknown")
            metadata.setdefault("last_revised", "unknown")
            metadata.setdefault("applies_to", "all")
            metadata.setdefault("h1_header", metadata.get("document_id", source))
            metadata.setdefault("grade_level", 3)
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def _tail_tokens(text: str, count: int) -> str:
    tokens = TOKEN_RE.findall(text or "")
    return " ".join(tokens[-count:])


def _is_table_start(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(TABLE_SEPARATOR_RE.match(lines[index + 1]))


def _split_section_blocks(lines: List[str]) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if _is_table_start(lines, i):
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
            continue

        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_table_start(lines, i):
            para_lines.append(lines[i])
            i += 1
        blocks.append(("text", "\n".join(para_lines)))

    return blocks


def _markdown_sections(text: str, default_h1: str) -> List[Tuple[Dict[str, str], List[Tuple[str, str]]]]:
    headers = {"h1_header": default_h1, "h2_header": "", "h3_header": ""}
    sections: List[Tuple[Dict[str, str], List[Tuple[str, str]]]] = []
    section_lines: List[str] = []

    def flush() -> None:
        nonlocal section_lines
        blocks = _split_section_blocks(section_lines)
        if blocks:
            sections.append((dict(headers), blocks))
        section_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = HEADER_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            value = match.group(2).strip()
            if level == 1:
                headers["h1_header"] = value
                headers["h2_header"] = ""
                headers["h3_header"] = ""
            elif level == 2:
                headers["h2_header"] = value
                headers["h3_header"] = ""
            elif level == 3:
                headers["h3_header"] = value
            section_lines = [line]
        else:
            section_lines.append(line)
    flush()

    if not sections and text.strip():
        sections.append((dict(headers), _split_section_blocks(text.splitlines())))
    return sections


def _split_long_text(text: str, max_tokens: int) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= max_tokens:
        return [text]
    return [" ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _split_large_table(table: str, row_limit: int) -> List[str]:
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) <= 2 + row_limit:
        return [table]

    header = lines[:2]
    rows = lines[2:]
    chunks = []
    for index in range(0, len(rows), row_limit):
        chunks.append("\n".join(header + rows[index : index + row_limit]))
    return chunks


def _build_section_chunks(
    blocks: List[Tuple[str, str]],
    max_tokens: int,
    table_row_limit: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    chunks: List[Tuple[str, Dict[str, Any]]] = []
    current: List[str] = []
    current_tables = 0

    def flush() -> None:
        nonlocal current, current_tables
        text = "\n\n".join(part for part in current if part.strip()).strip()
        if text:
            chunks.append((text, {"contains_table": current_tables > 0, "table_count": current_tables}))
        current = []
        current_tables = 0

    for block_type, block_text in blocks:
        block_parts = [block_text]
        if block_type == "table":
            block_parts = _split_large_table(block_text, table_row_limit)
        elif _token_count(block_text) > max_tokens:
            block_parts = _split_long_text(block_text, max_tokens)

        for part in block_parts:
            part_tokens = _token_count(part)
            current_tokens = _token_count("\n\n".join(current))
            if current and current_tokens + part_tokens > max_tokens:
                flush()
            current.append(part)
            if block_type == "table":
                current_tables += 1
            if part_tokens >= max_tokens:
                flush()
    flush()
    return chunks


def _apply_overlap(chunks: List[str], ratio: float, max_tokens: int) -> List[str]:
    if len(chunks) <= 1:
        return chunks

    overlap_tokens = max(20, int(max_tokens * ratio))
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        tail = _tail_tokens(previous, overlap_tokens)
        if tail:
            current = f"Context overlap: {tail}\n\n{current}"
        overlapped.append(current)
    return overlapped


def _section_path(metadata: Dict[str, Any]) -> str:
    return " > ".join(
        str(metadata.get(key, "")).strip()
        for key in ("h1_header", "h2_header", "h3_header")
        if str(metadata.get(key, "")).strip()
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    max_tokens = chunk_size or int(os.getenv("SEMANTIC_CHUNK_TOKENS", str(DEFAULT_CHUNK_TOKENS)))
    table_row_limit = int(os.getenv("TABLE_ROW_CHUNK_LIMIT", str(DEFAULT_TABLE_ROW_CHUNK)))
    ratio = float(os.getenv("CHUNK_OVERLAP_RATIO", str(DEFAULT_OVERLAP_RATIO)))
    if chunk_overlap is not None:
        ratio = max(0.0, min(0.3, chunk_overlap / max(max_tokens, 1)))

    chunks: List[Document] = []
    for document in documents:
        base_metadata = dict(document.metadata or {})
        sections = _markdown_sections(document.page_content, str(base_metadata.get("h1_header") or "Policy"))
        raw_chunks: List[Tuple[str, Dict[str, Any]]] = []

        for headers, blocks in sections:
            section_chunks = _build_section_chunks(blocks, max_tokens=max_tokens, table_row_limit=table_row_limit)
            for text, chunk_flags in section_chunks:
                metadata = {**base_metadata, **headers, **chunk_flags}
                metadata["section_path"] = _section_path(metadata)
                raw_chunks.append((text, metadata))

        overlapped_texts = _apply_overlap([text for text, _ in raw_chunks], ratio=ratio, max_tokens=max_tokens)
        for index, ((_, metadata), text) in enumerate(zip(raw_chunks, overlapped_texts)):
            metadata.update(
                {
                    "chunk_index": index,
                    "content_length": len(text),
                    "token_count": _token_count(text),
                    "chunking_strategy": "markdown_semantic_table_preserving",
                    "overlap_ratio": ratio,
                }
            )
            chunks.append(Document(page_content=text, metadata=metadata))

    return chunks


def verify_ingestion_chunks(chunks: List[Document]) -> List[str]:
    issues: List[str] = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
        if missing:
            issues.append(f"chunk {index} missing metadata: {', '.join(missing)}")
        if metadata.get("contains_table") and "|" not in chunk.page_content:
            issues.append(f"chunk {index} marked as table chunk but has no markdown table")
        if _token_count(chunk.page_content) == 0:
            issues.append(f"chunk {index} is empty")
    return issues


def run_ingestion(data_path: str = DATA_PATH, file_paths: Optional[Iterable[str]] = None) -> dict:
    documents = load_documents(data_path=data_path, file_paths=file_paths)
    chunks = split_documents(documents)
    issues = verify_ingestion_chunks(chunks)
    if issues:
        preview = "; ".join(issues[:5])
        raise ValueError(f"Ingestion verification failed: {preview}")

    result = index_documents(chunks)
    payload = {
        "status": "success" if chunks else "empty",
        "documents_loaded": len(documents),
        "chunks_indexed": result.get("chunks_indexed", 0),
        "collection_count": result.get("collection_count", 0),
        "db_path": result.get("db_path"),
        "collection": result.get("collection"),
        "chunking_strategy": "markdown_semantic_table_preserving",
        "metadata_fields": REQUIRED_METADATA,
    }
    print(f"Indexed {payload['chunks_indexed']} policy chunks")
    return payload


def ingest_policies_incremental(policy_dir: str = DATA_PATH) -> dict:
    return run_ingestion(data_path=policy_dir)


def get_vector_store() -> Any:
    return get_vectorstore()


def main() -> dict:
    return run_ingestion()


if __name__ == "__main__":
    results = main()
    raise SystemExit(0 if results.get("status") == "success" else 1)
