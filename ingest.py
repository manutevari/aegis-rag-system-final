# ingest.py

import os
import patoolib
from pathlib import Path
import re
from typing import List, Dict

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# CONFIG
# =========================
RAR_FILE = "data.rar"
EXTRACT_DIR = "data"
COLLECTION_NAME = "aegis_policies"

# Grade mapping (customize if needed)
GRADE_MAP = {
    "l1": 1,
    "l2": 2,
    "l3": 3,
    "l4": 4
}

# =========================
# STEP 1: EXTRACT
# =========================
def extract_rar():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)

    print("📦 Extracting RAR...")
    patoolib.extract_archive(RAR_FILE, outdir=EXTRACT_DIR)
    print("✅ Extraction complete")


# =========================
# STEP 2: METADATA PARSER
# =========================
def detect_grade(filename: str) -> int:
    filename = filename.lower()
    for key, val in GRADE_MAP.items():
        if key in filename:
            return val
    return 4  # default = highest access


def detect_category(filepath: str) -> str:
    if "travel" in filepath.lower():
        return "travel"
    return "general"


# =========================
# STEP 3: MARKDOWN CHUNKING
# =========================
def chunk_markdown(text: str) -> List[Dict]:
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    docs = splitter.split_text(text)

    return docs


# =========================
# STEP 4: LOAD FILES
# =========================
def load_documents() -> List[Document]:
    all_docs = []

    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                filepath = os.path.join(root, file)

                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_markdown(text)

                for chunk in chunks:
                    metadata = {
                        "source": file,
                        "grade_level": detect_grade(file),
                        "policy_category": detect_category(filepath),
                        "section": chunk.metadata
                    }

                    doc = Document(
                        page_content=chunk.page_content,
                        metadata=metadata
                    )

                    all_docs.append(doc)

    print(f"📄 Total chunks created: {len(all_docs)}")
    return all_docs


# =========================
# STEP 5: STORE IN VECTOR DB
# =========================
def store_in_chroma(docs: List[Document]):
    print("🔗 Generating embeddings + storing...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory="./chroma_db"
    )

    db.persist()
    print("✅ Stored in ChromaDB")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    extract_rar()
    docs = load_documents()
    store_in_chroma(docs)

    print("🚀 Ingestion complete. Your brain is ready.")
