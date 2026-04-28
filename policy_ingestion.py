"""Canonical policy ingestion entrypoint.

This module is the one ingestion path for the app. It loads policy files from
``data/``, splits them into chunks, and writes them through
``app.core.vector_store`` so retrieval reads the same Chroma collection.
"""

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.core.vector_store import get_vectorstore, index_documents

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("POLICY_DIR", "data")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CATEGORY_KEYWORDS = {
    "travel": ["travel", "fuel", "mileage", "vehicle", "hotel", "flight", "rental"],
    "security": ["security", "privacy", "data", "vpn", "incident"],
    "hr": ["hr", "conduct", "leave", "discipline", "work policies"],
    "training": ["training", "learning", "tuition", "certification"],
    "compensation": ["compensation", "salary", "pay", "hra"],
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
}


def _normalise_signal(path: Path, text: str = "") -> str:
    return f"{path.as_posix()} {text[:1000]}".lower()


def detect_category(path: Path, text: str = "") -> str:
    signal = _normalise_signal(path, text)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in signal for keyword in keywords):
            return category
    return "general"


def detect_grade(path: Path, text: str = "") -> int:
    signal = _normalise_signal(path, text)
    for label, grade in GRADE_MAP.items():
        if label in signal:
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


def load_documents(
    data_path: str = DATA_PATH,
    file_paths: Optional[Iterable[str]] = None,
) -> List[Document]:
    root = Path(data_path)
    documents: List[Document] = []

    for path in iter_policy_files(data_path=data_path, file_paths=file_paths):
        try:
            docs = _load_file(path)
            preview = "\n".join(doc.page_content for doc in docs[:1])
            category = detect_category(path, preview)
            grade_level = detect_grade(path, preview)
            source_path = path.as_posix()
            try:
                source_path = path.relative_to(root).as_posix()
            except ValueError:
                pass

            for doc in docs:
                doc.metadata.update(
                    {
                        "source": path.name,
                        "source_path": source_path,
                        "policy_category": category,
                        "grade_level": grade_level,
                    }
                )
            documents.extend(docs)
        except Exception as exc:
            logger.warning("Could not load policy file %s: %s", path, exc)

    if not documents:
        from app.tools._sample_policies import get_sample_docs

        logger.warning("No policy files loaded from %s; using built-in sample policies", data_path)
        documents = get_sample_docs()

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50")),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_index", index)
        chunk.metadata.setdefault("content_length", len(chunk.page_content))

    return chunks


def run_ingestion(data_path: str = DATA_PATH, file_paths: Optional[Iterable[str]] = None) -> dict:
    documents = load_documents(data_path=data_path, file_paths=file_paths)
    chunks = split_documents(documents)
    result = index_documents(chunks)

    payload = {
        "status": "success" if chunks else "empty",
        "documents_loaded": len(documents),
        "chunks_indexed": result.get("chunks_indexed", 0),
        "collection_count": result.get("collection_count", 0),
        "db_path": result.get("db_path"),
        "collection": result.get("collection"),
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
