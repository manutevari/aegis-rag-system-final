"""Typed runtime settings for AEGIS provider integrations."""

from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Generation
    llm_provider: str = Field(default="openai")
    model_provider: Optional[str] = None
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.1)
    openai_max_output_tokens: int = Field(default=1024)
    google_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    google_model: str = Field(default="gemini-2.5-flash")
    google_temperature: float = Field(default=0.1)
    google_max_output_tokens: int = Field(default=1024)

    # Embeddings
    rag_embeddings_provider: str = Field(default="openai")
    openai_embedding_model: str = Field(default="text-embedding-3-large")
    openai_embedding_dimensions: int = Field(default=3072)
    google_embedding_model: str = Field(default="gemini-embedding-001")
    google_embedding_dimensions: int = Field(default=3072)
    local_hash_embed_dim: int = Field(default=384)
    local_embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    allow_hf_downloads: bool = Field(default=False)

    # Vector database
    vector_backend: str = Field(default="pinecone")
    chroma_dir: str = Field(default="db")
    chroma_collection: str = Field(default="aegis_policies")
    pinecone_api_key: Optional[SecretStr] = None
    pinecone_index_name: str = Field(default="aegis-policies")
    pinecone_index_host: Optional[str] = None
    pinecone_namespace: str = Field(default="default")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")
    pinecone_metric: str = Field(default="cosine")
    pinecone_create_index: bool = Field(default=False)
    pinecone_batch_size: int = Field(default=100)

    # Reranking
    rerank_provider: str = Field(default="cohere")
    cohere_api_key: Optional[SecretStr] = None
    cohere_rerank_model: str = Field(default="rerank-v3.5")
    rerank_model: str = Field(default="rerank-v3.5")
    rerank_top_k: int = Field(default=5)
    retrieval_broad_k: int = Field(default=25)

    # Ollama fallback path
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_host: Optional[str] = None
    ollama_model: str = Field(default="llama3")
    ollama_health_timeout: float = Field(default=0.5)

    @property
    def active_llm_provider(self) -> str:
        return (self.model_provider or self.llm_provider or "openai").strip().lower()

    @property
    def active_embeddings_provider(self) -> str:
        return (self.rag_embeddings_provider or "openai").strip().lower()

    @property
    def active_embedding_dimensions(self) -> int:
        provider = self.active_embeddings_provider
        if provider in {"google", "gemini", "google-gemini"}:
            return self.google_embedding_dimensions
        if provider == "openai":
            return self.openai_embedding_dimensions
        return self.local_hash_embed_dim

    @property
    def openai_key(self) -> Optional[str]:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def google_key(self) -> Optional[str]:
        return self.google_api_key.get_secret_value() if self.google_api_key else None

    @property
    def cohere_key(self) -> Optional[str]:
        return self.cohere_api_key.get_secret_value() if self.cohere_api_key else None

    @property
    def pinecone_key(self) -> Optional[str]:
        return self.pinecone_api_key.get_secret_value() if self.pinecone_api_key else None

    @property
    def use_pinecone(self) -> bool:
        return self.vector_backend.strip().lower() == "pinecone"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
