"""Tests for choosing the vector store that matches the configured retrieval mode."""

import pytest
from qdrant_client import QdrantClient

from rag import store_factory
from rag.chunking import Chunk
from rag.hybrid_store import HybridVectorStore
from rag.qdrant_store import QdrantVectorStore
from rag.retrieval_settings import RetrievalMode, RetrievalSettings
from rag.store_factory import build_vector_store
from rag.vector_store import TfidfVectorStore
from tests.fakes import HashingEmbedder


class CountingEmbedderFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self) -> HashingEmbedder:
        self.calls += 1
        return HashingEmbedder()


@pytest.fixture
def qdrant_connections(monkeypatch) -> list[dict]:
    connections: list[dict] = []

    def connect(**connection_arguments) -> QdrantClient:
        connections.append(connection_arguments)
        return QdrantClient(location=":memory:")

    monkeypatch.setattr(store_factory, "QdrantClient", connect)
    return connections


def test_tfidf_mode_builds_the_lexical_store():
    settings = RetrievalSettings(mode=RetrievalMode.TFIDF)

    assert isinstance(build_vector_store(settings, CountingEmbedderFactory()), TfidfVectorStore)


def test_tfidf_mode_never_asks_for_an_embedding_model():
    embedder_factory = CountingEmbedderFactory()

    build_vector_store(RetrievalSettings(mode=RetrievalMode.TFIDF), embedder_factory)

    assert embedder_factory.calls == 0


def test_semantic_mode_builds_the_qdrant_store():
    embedder_factory = CountingEmbedderFactory()

    store = build_vector_store(RetrievalSettings(mode=RetrievalMode.SEMANTIC), embedder_factory)

    assert isinstance(store, QdrantVectorStore)
    assert embedder_factory.calls == 1


def test_hybrid_mode_builds_the_hybrid_combiner():
    embedder_factory = CountingEmbedderFactory()

    store = build_vector_store(RetrievalSettings(mode=RetrievalMode.HYBRID), embedder_factory)

    assert isinstance(store, HybridVectorStore)
    assert embedder_factory.calls == 1


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_every_mode_indexes_and_finds_a_document_through_the_common_interface(mode):
    store = build_vector_store(RetrievalSettings(mode=mode), HashingEmbedder)
    store.add_documents(
        [
            Chunk(document_id="cats", index=0, text="Cats are small feline animals."),
            Chunk(document_id="rockets", index=0, text="Rockets reach orbit."),
        ]
    )

    assert store.search("feline animals")[0].chunk.document_id == "cats"


def test_a_configured_qdrant_url_is_used_to_connect(qdrant_connections):
    settings = RetrievalSettings(
        mode=RetrievalMode.SEMANTIC, qdrant_url="http://qdrant.internal:6333"
    )

    build_vector_store(settings, HashingEmbedder)

    assert qdrant_connections == [{"url": "http://qdrant.internal:6333"}]


def test_without_a_qdrant_url_the_in_memory_mode_is_used(qdrant_connections):
    build_vector_store(RetrievalSettings(mode=RetrievalMode.SEMANTIC), HashingEmbedder)

    assert qdrant_connections == [{"location": ":memory:"}]


def test_tfidf_mode_never_connects_to_qdrant(qdrant_connections):
    build_vector_store(RetrievalSettings(mode=RetrievalMode.TFIDF), HashingEmbedder)

    assert qdrant_connections == []
