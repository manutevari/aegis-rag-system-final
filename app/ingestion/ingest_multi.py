import os
from typing import List, Dict
from app.ingestion.loader import load_file
from app.ingestion.chunker import chunk_docs
from app.core.vector_store import get_vector_store


def ingest_files(file_paths: List[str], progress_callback=None) -> Dict:
    vs = get_vector_store()

    total_chunks = 0
    results = []

    for i, path in enumerate(file_paths):
        try:
            docs = load_file(path)
            chunks = chunk_docs(docs)

            # attach metadata
            for c in chunks:
                c.metadata["source"] = os.path.basename(path)

            vs.add_documents(chunks)

            total_chunks += len(chunks)
            results.append({"file": path, "chunks": len(chunks), "status": "ok"})

        except Exception as e:
            results.append({"file": path, "error": str(e), "status": "failed"})

        if progress_callback:
            progress_callback((i + 1) / len(file_paths))

    return {
        "files": len(file_paths),
        "total_chunks": total_chunks,
        "details": results,
    }
