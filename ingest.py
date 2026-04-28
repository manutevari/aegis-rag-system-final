import os
import patoolib
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import

# =========================
# CONFIG
# =========================
RAR_FILE = "data.rar"
EXTRACT_DIR = "data"
COLLECTION_NAME = "aegis_policies"

GRADE_MAP = {
    "l1": 1,
    "l2": 2,
    "l3": 3,
    "l4": 4,
    "executive": 5
}

# =========================
# STEP 1: EXTRACT
# =========================
def extract_rar():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
    
    print("📦 Extracting RAR...")
    try:
        patoolib.extract_archive(RAR_FILE, outdir=EXTRACT_DIR)
        print("✅ Extraction complete")
    except Exception as e:
        print(f"❌ Extraction failed: {e}. Ensure a RAR extractor is installed on your OS.")

# =========================
# STEP 2: METADATA HELPERS
# =========================
def detect_grade(filename: str) -> int:
    filename = filename.lower()
    for key, val in GRADE_MAP.items():
        if key in filename:
            return val
    return 3  # Default to L3 as per Aegis baseline

def detect_category(filepath: str) -> str:
    path_str = filepath.lower()
    if "travel" in path_str: return "travel"
    if "fuel" in path_str: return "logistics"
    if "hr" in path_str: return "human_resources"
    return "general"

# =========================
# STEP 3: DOCUMENT PROCESSING
# =========================
def load_and_process() -> List[Document]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_docs = []

    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.lower().endswith((".md", ".txt")):
                filepath = os.path.join(root, file)
                
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                # Perform Header-Based Splitting
                chunks = splitter.split_text(text)

                for chunk in chunks:
                    # Flatten Metadata for Vector Store Compatibility
                    metadata = {
                        "source": file,
                        "grade_level": detect_grade(file),
                        "policy_category": detect_category(filepath),
                    }
                    
                    # Add section info from the splitter to the top-level metadata
                    metadata.update(chunk.metadata)

                    doc = Document(
                        page_content=chunk.page_content,
                        metadata=metadata
                    )
                    all_docs.append(doc)

    print(f"📄 Total Aegis Chunks: {len(all_docs)}")
    return all_docs

# =========================
# STEP 4: VECTOR STORAGE
# =========================
def store_in_chroma(docs: List[Document]):
    print("🔗 Generating embeddings + indexing...")

    # Using text-embedding-3-small as requested in Aegis specs
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Stored in ChromaDB at ./chroma_db")
    return vectorstore

if __name__ == "__main__":
    extract_rar()
    processed_docs = load_and_process()
    if processed_docs:
        store_in_chroma(processed_docs)
        print("🚀 Ingestion complete. Vector Store is live.")
