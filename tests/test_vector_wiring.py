import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.documents import Document

from app.state import AgentState


def _clear_settings_cache():
    from app.core.settings import get_settings

    get_settings.cache_clear()


def test_hash_embeddings_are_deterministic_without_network():
    from app.core.vector_store import LocalHashEmbeddings

    embeddings = LocalHashEmbeddings(dimension=32)
    first = embeddings.embed_query("fuel reimbursement policy")
    second = embeddings.embed_query("fuel reimbursement policy")

    assert first == second
    assert len(first) == 32
    assert any(value != 0 for value in first)


def test_get_embeddings_defaults_to_hash_provider_without_openai_key(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_EMBEDDINGS_PROVIDER", raising=False)
    _clear_settings_cache()

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.LocalHashEmbeddings)


def test_get_embeddings_can_force_hash_provider(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "hash")
    _clear_settings_cache()

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.LocalHashEmbeddings)


def test_get_embeddings_can_use_google_provider(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    _clear_settings_cache()

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.GoogleEmbeddingModel)


def test_local_embeddings_fall_back_to_hash_when_model_unavailable(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setattr(
        vector_store,
        "_huggingface_embeddings",
        lambda: (_ for _ in ()).throw(RuntimeError("model not cached")),
    )
    _clear_settings_cache()

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.LocalHashEmbeddings)


def test_semantic_chunking_preserves_tables_and_metadata():
    import policy_ingestion

    document = Document(
        page_content=(
            "# Travel Policy\n"
            "**Document ID:** TRV-POL-9999-V1\n"
            "**Effective Date:** January 1, 2026\n"
            "**Policy Owner:** Travel Team\n\n"
            "## Ground Transportation\n"
            "Licensed taxis are reimbursable for airport transfers.\n\n"
            "| Mode | Rule |\n"
            "| --- | --- |\n"
            "| Taxi | Standard licensed taxis are allowed. |\n"
            "| Premium car | Not allowed unless with a client. |\n"
        ),
        metadata={
            "source": "Travel Policy.txt",
            "source_path": "travel/Travel Policy.txt",
            "document_id": "TRV-POL-9999-V1",
            "policy_category": "travel",
            "policy_owner": "Travel Team",
            "effective_date": "2026-01-01",
            "h1_header": "Travel Policy",
            "grade_level": 3,
        },
    )

    chunks = policy_ingestion.split_documents([document], chunk_size=80)
    issues = policy_ingestion.verify_ingestion_chunks(chunks)

    assert issues == []
    assert all(chunk.metadata["h2_header"] for chunk in chunks)
    assert any(chunk.metadata["contains_table"] for chunk in chunks)
    table_chunk = next(chunk for chunk in chunks if chunk.metadata["contains_table"])
    assert "| Mode | Rule |" in table_chunk.page_content
    assert table_chunk.metadata["section_path"] == "Travel Policy > Ground Transportation"


def test_retrieval_expands_taxi_query_and_adds_travel_filter():
    import app.nodes.retrieval as retrieval

    expansions = retrieval._expand_query("Can I expense a taxi?")
    metadata_filter = retrieval._metadata_filter("Can I expense a taxi?")

    assert any("licensed taxi" in expansion for expansion in expansions)
    assert any("ground transportation" in expansion for expansion in expansions)
    assert metadata_filter == {"policy_category": "travel"}


def test_retrieval_node_uses_shared_vector_store(monkeypatch):
    import app.nodes.retrieval as retrieval

    class FakeRetriever:
        def invoke(self, query):
            assert "fuel reimbursement" in query
            return [
                Document(
                    page_content="Fuel receipts are reimbursable for rental cars only.",
                    metadata={"source": "Fuel and Mileage Policy.txt", "policy_category": "travel"},
                )
            ]

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(retrieval, "ensure_vectorstore_ready", lambda auto_ingest=True: 1)
    monkeypatch.setattr(retrieval, "get_retriever", lambda k=5, metadata_filter=None: FakeRetriever())
    _clear_settings_cache()

    result = retrieval.run(AgentState(query="fuel reimbursement", trace_log=[]))

    assert result["documents"][0]["source"] == "Fuel and Mileage Policy.txt"
    assert result["retrieval_docs"]
    assert "Fuel receipts" in result["context"]
    assert result["metadata_filter"] == {"policy_category": "travel"}
    assert result["trace_log"][-1]["data"]["chunks"] == 1


def test_policy_retriever_delegates_to_shared_search(monkeypatch):
    import app.tools.retriever as retriever

    monkeypatch.setattr(
        retriever,
        "search_documents",
        lambda query, k=6: [Document(page_content="Travel policy chunk", metadata={})],
    )

    assert retriever.PolicyRetriever().retrieve("travel", top_k=1) == ["Travel policy chunk"]


def test_generator_guard_skips_llm_when_no_documents(monkeypatch):
    import app.nodes.generator as generator

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called without retrieved policy documents")

    monkeypatch.setattr(generator, "safe_invoke_llm", fail_if_called)

    result = generator.run(AgentState(query="fuel policy", documents=[], retrieval_docs=[], trace_log=[]))

    assert "No policy data found" in result["answer"]
    assert result["response"] == result["answer"]
    assert result["trace_log"][-1]["data"]["guard"] == "no_documents"


def test_policy_ingestion_indexes_through_shared_store(tmp_path, monkeypatch):
    import policy_ingestion

    policy_file = tmp_path / "Fuel Policy.txt"
    policy_file.write_text(
        "# Fuel Policy\n\n"
        "**Document ID:** TRV-POL-3012-V2\n"
        "**Effective Date:** April 1, 2026\n"
        "**Policy Owner:** Fleet Team\n\n"
        "## Fuel Reimbursement\n"
        "Fuel reimbursement applies to rental vehicles. Personal mileage claims cannot include fuel receipts.",
        encoding="utf-8",
    )

    calls = {}

    def fake_index_documents(chunks):
        calls["chunks"] = list(chunks)
        return {
            "chunks_indexed": len(calls["chunks"]),
            "collection_count": len(calls["chunks"]),
            "db_path": "db",
            "collection": "aegis_policies",
        }

    monkeypatch.setattr(policy_ingestion, "index_documents", fake_index_documents)

    result = policy_ingestion.run_ingestion(data_path=str(tmp_path))

    assert result["chunks_indexed"] > 0
    assert calls["chunks"]
    assert calls["chunks"][0].metadata["policy_category"] == "travel"
    assert calls["chunks"][0].metadata["document_id"] == "TRV-POL-3012-V2"
