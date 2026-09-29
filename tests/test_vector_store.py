"""Tests for the TF-IDF backed vector store."""

from rag.chunking import Chunk
from rag.vector_store import TfidfVectorStore


def test_search_on_an_empty_store_returns_nothing():
    store = TfidfVectorStore()

    assert store.search("anything") == []


def test_search_ranks_the_more_relevant_chunk_first():
    store = TfidfVectorStore()
    store.add_documents(
        [
            Chunk(document_id="cats", index=0, text="Cats are small domesticated feline animals."),
            Chunk(document_id="rockets", index=0, text="Rockets use combustion to reach orbit."),
        ]
    )

    results = store.search("Tell me about feline pets")

    assert results[0].chunk.document_id == "cats"


def test_search_respects_top_k():
    store = TfidfVectorStore()
    store.add_documents(
        [
            Chunk(document_id="doc", index=0, text="python programming language"),
            Chunk(document_id="doc", index=1, text="python programming tutorial"),
            Chunk(document_id="doc", index=2, text="python programming guide"),
        ]
    )

    results = store.search("python programming", top_k=2)

    assert len(results) == 2


def test_documents_added_later_are_still_searchable():
    store = TfidfVectorStore()
    store.add_documents([Chunk(document_id="a", index=0, text="apples and oranges")])
    store.add_documents([Chunk(document_id="b", index=0, text="bicycles and cars")])

    results = store.search("bicycles")

    assert results[0].chunk.document_id == "b"


def test_a_query_with_no_matching_terms_returns_no_results():
    store = TfidfVectorStore()
    store.add_documents([Chunk(document_id="a", index=0, text="apples and oranges")])

    assert store.search("submarine reactor") == []
