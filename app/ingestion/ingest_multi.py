"""Multi-file ingestion adapter.

The upload/admin path delegates to policy_ingestion so uploaded files land in the
same Chroma collection used by runtime retrieval.
"""

from typing import Dict, List

from app.core.vector_store import index_documents
from policy_ingestion import load_documents, split_documents


def ingest_files(file_paths: List[str], progress_callback=None) -> Dict:
    documents = load_documents(file_paths=file_paths)
    chunks = split_documents(documents)
    result = index_documents(chunks)

    if progress_callback:
        progress_callback(1.0)

    return {
        "files": len(file_paths),
        "total_chunks": len(chunks),
        "collection_count": result.get("collection_count", 0),
        "db_path": result.get("db_path"),
        "collection": result.get("collection"),
        "details": [
            {"file": path, "status": "indexed"}
            for path in file_paths
        ],
    }
