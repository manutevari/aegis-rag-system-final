import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.documents import Document

from app.state import AgentState


def test_hash_embeddings_are_deterministic_without_network():
    from app.core.vector_store import LocalHashEmbeddings

    embeddings = LocalHashEmbeddings(dimension=32)
    first = embeddings.embed_query("fuel reimbursement policy")
    second = embeddings.embed_query("fuel reimbursement policy")

    assert first == second
    assert len(first) == 32
    assert any(value != 0 for value in first)


def test_get_embeddings_can_force_hash_provider(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "hash")

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.LocalHashEmbeddings)


def test_local_embeddings_fall_back_to_hash_when_model_unavailable(monkeypatch):
    import app.core.vector_store as vector_store

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setattr(
        vector_store,
        "_huggingface_embeddings",
        lambda: (_ for _ in ()).throw(RuntimeError("model not cached")),
    )

    embeddings = vector_store.get_embeddings()

    assert isinstance(embeddings, vector_store.LocalHashEmbeddings)


def test_retrieval_node_uses_shared_vector_store(monkeypatch):
    import app.nodes.retrieval as retrieval

    class FakeRetriever:
        def invoke(self, query):
            assert "fuel reimbursement" in query
            return [
                Document(
                    page_content="Fuel receipts are reimbursable for rental cars only.",
                    metadata={"source": "Fuel and Mileage Policy.txt"},
                )
            ]

    monkeypatch.setattr(retrieval, "ensure_vectorstore_ready", lambda auto_ingest=True: 1)
    monkeypatch.setattr(retrieval, "get_retriever", lambda k=5: FakeRetriever())

    result = retrieval.run(AgentState(query="fuel reimbursement", trace_log=[]))

    assert result["documents"][0]["source"] == "Fuel and Mileage Policy.txt"
    assert result["retrieval_docs"]
    assert "Fuel receipts" in result["context"]
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
