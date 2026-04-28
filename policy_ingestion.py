"""
================================================================================
COMPREHENSIVE POLICY INGESTION & INDEXING TOOL
================================================================================

Complete one-shot implementation for:
  • Multi-format policy document loading (TXT, PDF, DOCX)
  • Intelligent chunking with metadata extraction
  • Semantic indexing with vector store (FAISS/Chroma)
  • Policy category classification and grounding
  • Hierarchical document structure preservation
  • Duplicate detection and deduplication
  • Progress tracking and error handling
  • Integration with existing RAG pipeline

Author: Claude
Date: 2026-04-28
"""

import os
import json
import logging
import hashlib
import pathlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import re

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
)
from langchain_openai import OpenAIEmbeddings

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATA MODELS & CONFIGURATION
# ============================================================================

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each document chunk"""
    source_file: str
    policy_category: str
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    word_count: int = 0
    has_list: bool = False
    has_table: bool = False
    document_id: str = ""
    effective_date: Optional[str] = None
    version: Optional[str] = None
    last_revised: Optional[str] = None
    document_hash: str = ""
    ingestion_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyIngestionConfig:
    """Configuration for policy ingestion pipeline"""
    # Paths
    policy_dir: str = "data"
    vector_db_dir: str = "/tmp/dg_rag_chroma"
    metadata_cache_dir: str = "data/.metadata_cache"
    
    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 100
    
    # Vector store
    vector_store_type: str = "faiss"  # "faiss" or "chroma"
    embedding_model: str = "text-embedding-3-small"
    
    # Processing
    batch_size: int = 10
    enable_deduplication: bool = True
    extract_metadata: bool = True
    
    # File types
    supported_formats: List[str] = field(default_factory=lambda: ['.txt', '.pdf', '.docx'])
    
    # Category mapping
    category_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "Work Policies": ["conduct", "code of", "discipline", "leave", "absence", "compensation", "performance"],
        "Security": ["security", "privacy", "data", "protection", "encryption", "access control"],
        "Travel": ["travel", "hotel", "flight", "mileage", "fuel", "international", "expense"],
        "Training": ["training", "learning", "tuition", "development", "certification", "course"],
        "Benefits": ["health", "insurance", "retirement", "pension", "benefits", "pto"],
    })


# ============================================================================
# 2. POLICY METADATA EXTRACTION
# ============================================================================

class PolicyMetadataExtractor:
    """Extract structured metadata from policy documents"""
    
    # Regex patterns for common policy document fields
    PATTERNS = {
        'document_id': r'(?:Document ID|Doc ID|Policy ID|ID)[:\s]+([A-Z0-9\-]+)',
        'effective_date': r'(?:Effective Date|Effective|Effective from)[:\s]+([A-Za-z\d\s,]+)',
        'last_revised': r'(?:Last Revised|Revised|Last Updated)[:\s]+([A-Za-z\d\s,]+)',
        'version': r'(?:Version|Ver)[:\s]+([\d.v]+)',
        'policy_owner': r'(?:Policy Owner|Owner|Authority)[:\s]+([^\n]+)',
        'section_title': r'^#+\s+(\d+\.?\d*[\s\.\-]*)?(.+?)$',
    }
    
    @staticmethod
    def extract_document_metadata(text: str, filename: str) -> Dict[str, Optional[str]]:
        """Extract top-level document metadata"""
        metadata = {
            'document_id': None,
            'effective_date': None,
            'last_revised': None,
            'version': None,
            'policy_owner': None,
        }
        
        # Search only first 1000 characters for efficiency
        header = text[:1000]
        
        for key, pattern in PolicyMetadataExtractor.PATTERNS.items():
            if key.startswith('section'):
                continue
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip() if match.lastindex >= 1 else match.group(0)
        
        return metadata
    
    @staticmethod
    def extract_section_info(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract section title and number from chunk"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            match = re.match(r'^#+\s+(\d+\.?\d*[\s\.\-]*)?(.+?)$', line)
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                return section_num, section_title
        return None, None
    
    @staticmethod
    def detect_list_and_table(text: str) -> Tuple[bool, bool]:
        """Detect if chunk contains lists or tables"""
        has_list = bool(re.search(r'^\s*[\*\-\+]\s+|\n\s*[\d]+\.\s+', text, re.MULTILINE))
        has_table = bool(re.search(r'\|.*\|', text)) or bool(re.search(r'^\s*\|', text, re.MULTILINE))
        return has_list, has_table


# ============================================================================
# 3. INTELLIGENT DOCUMENT CHUNKER
# ============================================================================

class IntelligentPolicyChunker:
    """Smart chunking that preserves policy structure and context"""
    
    def __init__(self, config: PolicyIngestionConfig):
        self.config = config
        self.metadata_extractor = PolicyMetadataExtractor()
        
        # Use recursive splitter for better semantic coherence
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        # Secondary splitter for fallback
        self.char_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n",
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents into chunks with enhanced metadata
        
        Args:
            documents: List of loaded Document objects
            
        Returns:
            List of chunked documents with comprehensive metadata
        """
        chunked_docs = []
        document_hashes = defaultdict(str)
        
        for doc in documents:
            try:
                # Calculate document hash for duplicate detection
                doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                document_hashes[doc.metadata.get('source', 'unknown')] = doc_hash
                
                # Extract document-level metadata
                doc_metadata = self.metadata_extractor.extract_document_metadata(
                    doc.page_content,
                    doc.metadata.get('source', 'unknown')
                )
                
                # Perform chunking
                try:
                    chunks = self.splitter.split_documents([doc])
                except Exception as e:
                    logger.warning(f"Recursive split failed for {doc.metadata.get('source')}, using fallback: {e}")
                    chunks = self.char_splitter.split_documents([doc])
                
                # Enrich each chunk with metadata
                for chunk_idx, chunk in enumerate(chunks):
                    # Extract section info from chunk
                    section_num, section_title = self.metadata_extractor.extract_section_info(
                        chunk.page_content
                    )
                    has_list, has_table = self.metadata_extractor.detect_list_and_table(
                        chunk.page_content
                    )
                    
                    # Determine policy category
                    category = self._categorize_policy(
                        chunk.page_content,
                        doc.metadata.get('source', ''),
                        self.config.category_keywords
                    )
                    
                    # Build enhanced metadata
                    chunk_meta = ChunkMetadata(
                        source_file=doc.metadata.get('source', 'unknown'),
                        policy_category=category,
                        section_title=section_title,
                        section_number=section_num,
                        chunk_index=chunk_idx,
                        total_chunks=len(chunks),
                        char_count=len(chunk.page_content),
                        word_count=len(chunk.page_content.split()),
                        has_list=has_list,
                        has_table=has_table,
                        document_id=doc_metadata.get('document_id', ''),
                        effective_date=doc_metadata.get('effective_date'),
                        version=doc_metadata.get('version'),
                        last_revised=doc_metadata.get('last_revised'),
                        document_hash=doc_hash,
                    )
                    
                    # Merge with existing metadata
                    chunk.metadata.update(chunk_meta.to_dict())
                    chunked_docs.append(chunk)
                    
                logger.info(
                    f"✓ Chunked '{doc.metadata.get('source')}' into {len(chunks)} chunks "
                    f"[hash: {doc_hash}]"
                )
                
            except Exception as e:
                logger.error(f"✗ Error chunking {doc.metadata.get('source')}: {e}")
                continue
        
        return chunked_docs
    
    @staticmethod
    def _categorize_policy(text: str, filename: str, category_keywords: Dict[str, List[str]]) -> str:
        """Categorize policy based on content and filename"""
        text_lower = (text[:500] + filename).lower()  # Use header + filename
        
        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return "General"


# ============================================================================
# 4. DOCUMENT LOADER
# ============================================================================

class PolicyDocumentLoader:
    """Load policies from multiple formats"""
    
    def __init__(self, config: PolicyIngestionConfig):
        self.config = config
    
    def load_policies(self, policy_dir: Optional[str] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Load all policy documents from directory
        
        Args:
            policy_dir: Directory containing policies (uses config if None)
            
        Returns:
            Tuple of (documents, load_stats)
        """
        policy_dir = policy_dir or self.config.policy_dir
        docs = []
        stats = {
            'total_files': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'unsupported_formats': 0,
            'errors': []
        }
        
        policy_path = pathlib.Path(policy_dir)
        
        if not policy_path.exists():
            logger.error(f"Policy directory not found: {policy_dir}")
            return [], stats
        
        # Find all supported files
        files_to_load = []
        for format_ext in self.config.supported_formats:
            files_to_load.extend(policy_path.rglob(f'*{format_ext}'))
        
        logger.info(f"Found {len(files_to_load)} policy files to load")
        
        for file_path in sorted(files_to_load):
            stats['total_files'] += 1
            
            try:
                logger.info(f"Loading: {file_path.relative_to(policy_path)}")
                
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() == '.docx':
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                else:  # .txt and others
                    loader = TextLoader(str(file_path), encoding='utf-8')
                
                file_docs = loader.load()
                
                # Add source metadata
                for doc in file_docs:
                    doc.metadata['source'] = file_path.name
                    doc.metadata['source_path'] = str(file_path)
                
                docs.extend(file_docs)
                stats['successful_loads'] += 1
                logger.info(f"  ✓ Loaded {len(file_docs)} pages")
                
            except Exception as e:
                stats['failed_loads'] += 1
                error_msg = f"Failed to load {file_path.name}: {str(e)}"
                stats['errors'].append(error_msg)
                logger.error(f"  ✗ {error_msg}")
        
        logger.info(
            f"Load Summary: {stats['successful_loads']} successful, "
            f"{stats['failed_loads']} failed, {len(docs)} total pages"
        )
        
        return docs, stats


# ============================================================================
# 5. DEDUPLICATION ENGINE
# ============================================================================

class DeduplicationEngine:
    """Detect and handle duplicate/near-duplicate chunks"""
    
    SIMILARITY_THRESHOLD = 0.85
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def deduplicate_chunks(chunks: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Remove exact and near-duplicate chunks
        
        Returns:
            Tuple of (deduplicated_chunks, dedup_stats)
        """
        seen_hashes = set()
        dedup_chunks = []
        dedup_stats = {
            'total_chunks': len(chunks),
            'exact_duplicates_removed': 0,
            'near_duplicates_removed': 0,
            'unique_chunks': 0
        }
        
        # First pass: exact hash matching
        for chunk in chunks:
            chunk_hash = DeduplicationEngine.compute_hash(chunk.page_content)
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                dedup_chunks.append(chunk)
            else:
                dedup_stats['exact_duplicates_removed'] += 1
        
        # Second pass: near-duplicate detection (optional, slower)
        # Using token-based Jaccard similarity
        final_chunks = []
        seen_tokens = []
        
        for chunk in dedup_chunks:
            tokens = set(chunk.page_content.split())
            is_duplicate = False
            
            for seen_token_set in seen_tokens:
                similarity = DeduplicationEngine.jaccard_similarity(tokens, seen_token_set)
                if similarity > DeduplicationEngine.SIMILARITY_THRESHOLD:
                    dedup_stats['near_duplicates_removed'] += 1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_tokens.append(tokens)
                final_chunks.append(chunk)
        
        dedup_stats['unique_chunks'] = len(final_chunks)
        logger.info(
            f"Deduplication: {dedup_stats['exact_duplicates_removed']} exact + "
            f"{dedup_stats['near_duplicates_removed']} near duplicates removed. "
            f"Remaining: {dedup_stats['unique_chunks']} unique chunks"
        )
        
        return final_chunks, dedup_stats


# ============================================================================
# 6. VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """Manage vector store initialization and indexing"""
    
    def __init__(self, config: PolicyIngestionConfig):
        self.config = config
        self._store = None
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Initialize embedding model"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return OpenAIEmbeddings(
            model=self.config.embedding_model,
            api_key=api_key
        )
    
    def initialize_store(self, documents: List[Document]) -> Any:
        """Initialize vector store with documents"""
        if not documents:
            logger.warning("No documents provided for vector store initialization")
            return None
        
        embeddings = self._get_embeddings()
        
        try:
            if self.config.vector_store_type.lower() == "chroma":
                from langchain_chroma import Chroma
                self._store = Chroma.from_documents(
                    documents,
                    embeddings,
                    persist_directory=self.config.vector_db_dir,
                    collection_name="dg_rag_policies"
                )
                logger.info(f"✓ Chroma vector store initialized at {self.config.vector_db_dir}")
            else:  # FAISS
                from langchain_community.vectorstores import FAISS
                self._store = FAISS.from_documents(documents, embeddings)
                logger.info(f"✓ FAISS vector store initialized with {len(documents)} documents")
        
        except Exception as e:
            logger.error(f"✗ Failed to initialize vector store: {e}")
            raise
        
        return self._store
    
    def get_store(self) -> Any:
        """Return initialized vector store"""
        return self._store
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to existing vector store"""
        if self._store is None:
            return {"status": "error", "message": "Vector store not initialized"}
        
        try:
            self._store.add_documents(documents)
            return {
                "status": "success",
                "documents_added": len(documents),
                "message": f"Added {len(documents)} documents to vector store"
            }
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}


# ============================================================================
# 7. METADATA CACHE
# ============================================================================

class MetadataCache:
    """Cache chunk metadata for efficient retrieval"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "chunks_metadata.jsonl"
    
    def save_metadata(self, chunks: List[Document]) -> None:
        """Save chunk metadata to JSONL cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    metadata = {
                        'content_hash': hashlib.md5(chunk.page_content.encode()).hexdigest()[:16],
                        'content_preview': chunk.page_content[:100],
                        'metadata': chunk.metadata,
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(json.dumps(metadata) + '\n')
            
            logger.info(f"✓ Saved metadata for {len(chunks)} chunks to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load cached metadata"""
        metadata_index = defaultdict(list)
        
        if not self.cache_file.exists():
            return metadata_index
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line)
                        category = meta['metadata'].get('policy_category', 'unknown')
                        metadata_index[category].append(meta)
            
            logger.info(f"✓ Loaded metadata from {self.cache_file}")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")
        
        return metadata_index


# ============================================================================
# 8. ORCHESTRATOR: MAIN POLICY INGESTION PIPELINE
# ============================================================================

class PolicyIngestionPipeline:
    """
    End-to-end policy ingestion orchestrator
    
    Workflow:
      1. Load policy documents from directory
      2. Intelligently chunk with metadata extraction
      3. Deduplicate chunks
      4. Build vector index
      5. Cache metadata for retrieval
    """
    
    def __init__(self, config: Optional[PolicyIngestionConfig] = None):
        self.config = config or PolicyIngestionConfig()
        self.loader = PolicyDocumentLoader(self.config)
        self.chunker = IntelligentPolicyChunker(self.config)
        self.deduplicator = DeduplicationEngine()
        self.vector_manager = VectorStoreManager(self.config)
        self.cache = MetadataCache(self.config.metadata_cache_dir)
        
        self.ingestion_results = {
            'status': 'pending',
            'load_stats': {},
            'chunking_stats': {},
            'dedup_stats': {},
            'vector_store_stats': {},
            'total_chunks': 0,
            'total_unique_chunks': 0,
            'ingestion_time': 0,
            'errors': []
        }
    
    def ingest(self, policy_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline
        
        Args:
            policy_dir: Override default policy directory
            
        Returns:
            Comprehensive ingestion results and statistics
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("=" * 80)
            logger.info("STARTING POLICY INGESTION PIPELINE")
            logger.info("=" * 80)
            
            # ===== STEP 1: LOAD =====
            logger.info("\n[STEP 1/5] Loading policy documents...")
            docs, load_stats = self.loader.load_policies(policy_dir)
            self.ingestion_results['load_stats'] = load_stats
            
            if not docs:
                raise ValueError("No documents loaded. Check policy directory and file formats.")
            
            # ===== STEP 2: CHUNK =====
            logger.info(f"\n[STEP 2/5] Intelligent chunking ({len(docs)} pages)...")
            chunks = self.chunker.chunk_documents(docs)
            self.ingestion_results['chunking_stats'] = {
                'input_pages': len(docs),
                'output_chunks': len(chunks),
                'avg_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
            }
            
            # ===== STEP 3: DEDUPLICATE =====
            logger.info(f"\n[STEP 3/5] Deduplicating chunks...")
            unique_chunks, dedup_stats = self.deduplicator.deduplicate_chunks(chunks)
            self.ingestion_results['dedup_stats'] = dedup_stats
            self.ingestion_results['total_unique_chunks'] = len(unique_chunks)
            
            # ===== STEP 4: INDEX =====
            logger.info(f"\n[STEP 4/5] Building vector index ({len(unique_chunks)} chunks)...")
            vs = self.vector_manager.initialize_store(unique_chunks)
            self.ingestion_results['vector_store_stats'] = {
                'type': self.config.vector_store_type,
                'embedding_model': self.config.embedding_model,
                'chunks_indexed': len(unique_chunks)
            }
            
            # ===== STEP 5: CACHE =====
            logger.info(f"\n[STEP 5/5] Caching metadata...")
            self.cache.save_metadata(unique_chunks)
            
            # ===== SUMMARY =====
            elapsed = time.time() - start_time
            self.ingestion_results['status'] = 'success'
            self.ingestion_results['ingestion_time'] = elapsed
            self.ingestion_results['total_chunks'] = len(chunks)
            
            self._log_summary()
            
            return self.ingestion_results
        
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
            self.ingestion_results['status'] = 'failed'
            self.ingestion_results['errors'].append(str(e))
            return self.ingestion_results
    
    def _log_summary(self) -> None:
        """Log comprehensive ingestion summary"""
        results = self.ingestion_results
        logger.info("\n" + "=" * 80)
        logger.info("INGESTION COMPLETE - SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info(f"Status: {results['status'].upper()}")
        logger.info(f"Total Time: {results['ingestion_time']:.2f}s")
        logger.info(f"\nDocuments Loaded: {results['load_stats'].get('total_files', 0)}")
        logger.info(f"  ├─ Successful: {results['load_stats'].get('successful_loads', 0)}")
        logger.info(f"  └─ Failed: {results['load_stats'].get('failed_loads', 0)}")
        logger.info(f"\nChunking:")
        logger.info(f"  ├─ Input: {results['chunking_stats'].get('input_pages', 0)} pages")
        logger.info(f"  ├─ Output: {results['chunking_stats'].get('output_chunks', 0)} chunks")
        logger.info(f"  └─ Avg Size: {results['chunking_stats'].get('avg_chunk_size', 0):.0f} chars")
        logger.info(f"\nDeduplication:")
        logger.info(f"  ├─ Exact Duplicates: {results['dedup_stats'].get('exact_duplicates_removed', 0)}")
        logger.info(f"  ├─ Near Duplicates: {results['dedup_stats'].get('near_duplicates_removed', 0)}")
        logger.info(f"  └─ Unique: {results['dedup_stats'].get('unique_chunks', 0)}")
        logger.info(f"\nVector Store:")
        logger.info(f"  ├─ Type: {results['vector_store_stats'].get('type', 'unknown')}")
        logger.info(f"  ├─ Model: {results['vector_store_stats'].get('embedding_model', 'unknown')}")
        logger.info(f"  └─ Indexed: {results['vector_store_stats'].get('chunks_indexed', 0)} chunks")
        logger.info("=" * 80 + "\n")


# ============================================================================
# 9. USAGE EXAMPLE & MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point demonstrating complete ingestion workflow
    """
    # Initialize with default config
    config = PolicyIngestionConfig(
        policy_dir="data",
        vector_store_type="faiss",
        chunk_size=800,
        chunk_overlap=100,
        enable_deduplication=True,
        extract_metadata=True,
    )
    
    # Create and run pipeline
    pipeline = PolicyIngestionPipeline(config)
    results = pipeline.ingest()
    
    # Print results
    print("\n" + "=" * 80)
    print("INGESTION RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))
    print("=" * 80)
    
    return results


def ingest_policies_incremental(policy_dir: str = "data") -> Dict[str, Any]:
    """
    Simplified function for incremental policy ingestion
    
    Args:
        policy_dir: Directory containing policies
        
    Returns:
        Ingestion results
    """
    config = PolicyIngestionConfig(policy_dir=policy_dir)
    pipeline = PolicyIngestionPipeline(config)
    return pipeline.ingest(policy_dir)


def get_vector_store():
    """Get initialized vector store (integration hook for RAG pipeline)"""
    config = PolicyIngestionConfig()
    pipeline = PolicyIngestionPipeline(config)
    pipeline.ingest()
    return pipeline.vector_manager.get_store()


if __name__ == "__main__":
    results = main()
    exit(0 if results['status'] == 'success' else 1)
