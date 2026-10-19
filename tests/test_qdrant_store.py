"""Tests for the Qdrant-backed semantic store, running Qdrant in memory with a fake embedder."""

import pytest
from qdrant_client import QdrantClient

from rag.chunking import Chunk
from rag.qdrant_store import COLLECTION_NAME, UPSERT_BATCH_SIZE, QdrantVectorStore
from tests.fakes import HashingEmbedder


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(location=":memory:")


@pytest.fixture
def store(client) -> QdrantVectorStore:
    return QdrantVectorStore(embedder=HashingEmbedder(), client=client)


def test_search_on_an_empty_store_returns_nothing(store):
    assert store.search("anything") == []


def test_search_ranks_the_more_relevant_chunk_first(store):
    store.add_documents(
        [
            Chunk(document_id="cats", index=0, text="Cats are small domesticated feline animals."),
            Chunk(document_id="rockets", index=0, text="Rockets use combustion to reach orbit."),
        ]
    )

    results = store.search("feline animals")

    assert results[0].chunk.document_id == "cats"
    assert results[0].score > results[1].score


def test_search_returns_the_stored_chunk_unchanged(store):
    stored_chunk = Chunk(document_id="cats", index=3, text="Cats purr when they are content.")
    store.add_documents([stored_chunk])

    results = store.search("cats purr")

    assert results[0].chunk == stored_chunk


def test_search_respects_top_k(store):
    store.add_documents(
        [
            Chunk(document_id="doc", index=position, text="python programming")
            for position in range(5)
        ]
    )

    assert len(store.search("python programming", top_k=2)) == 2


def test_a_top_k_below_one_returns_nothing_instead_of_failing(store):
    store.add_documents([Chunk(document_id="doc", index=0, text="python programming")])

    assert store.search("python programming", top_k=0) == []


def test_documents_added_later_are_still_searchable(store):
    store.add_documents([Chunk(document_id="a", index=0, text="apples and oranges")])
    store.add_documents([Chunk(document_id="b", index=0, text="bicycles and cars")])

    assert store.search("bicycles")[0].chunk.document_id == "b"


def test_adding_the_same_chunk_twice_stores_it_once(store, client):
    chunk = Chunk(document_id="a", index=0, text="apples and oranges")

    store.add_documents([chunk])
    store.add_documents([chunk])

    assert client.count(COLLECTION_NAME).count == 1


def test_more_chunks_than_one_batch_are_all_stored(store, client):
    chunk_count = UPSERT_BATCH_SIZE * 2 + 5
    store.add_documents(
        [
            Chunk(document_id="doc", index=position, text=f"note {position}")
            for position in range(chunk_count)
        ]
    )

    assert client.count(COLLECTION_NAME).count == chunk_count


def test_adding_no_chunks_changes_nothing(store, client):
    store.add_documents([])

    assert client.count(COLLECTION_NAME).count == 0


def test_a_new_store_starts_from_an_empty_collection(client):
    first_store = QdrantVectorStore(embedder=HashingEmbedder(), client=client)
    first_store.add_documents([Chunk(document_id="old", index=0, text="stale content")])

    second_store = QdrantVectorStore(embedder=HashingEmbedder(), client=client)

    assert second_store.search("stale content") == []
