"""Tests for the hybrid store that fuses a lexical and a semantic ranking."""

import pytest
from qdrant_client import QdrantClient

from rag.chunking import Chunk
from rag.fusion import RRF_RANK_CONSTANT
from rag.hybrid_store import CANDIDATES_PER_RANKER, HybridVectorStore
from rag.qdrant_store import QdrantVectorStore
from rag.vector_store import TfidfVectorStore
from tests.fakes import HashingEmbedder, PresetRankingStore, scored_chunk


def document_order(results) -> list[str]:
    return [result.chunk.document_id for result in results]


def test_added_chunks_reach_both_stores():
    lexical_store, semantic_store = PresetRankingStore(), PresetRankingStore()
    hybrid_store = HybridVectorStore(lexical_store, semantic_store)
    chunks = [Chunk(document_id="doc", index=0, text="some text")]

    hybrid_store.add_documents(chunks)

    assert lexical_store.added_chunks == chunks
    assert semantic_store.added_chunks == chunks


def test_a_chunk_both_stores_agree_on_is_ranked_first():
    lexical_store = PresetRankingStore([scored_chunk("lexical-pick"), scored_chunk("agreed")])
    semantic_store = PresetRankingStore([scored_chunk("semantic-pick"), scored_chunk("agreed")])

    results = HybridVectorStore(lexical_store, semantic_store).search("question")

    assert document_order(results)[0] == "agreed"


def test_the_semantic_ranking_alone_is_used_when_the_lexical_store_finds_nothing():
    lexical_store = PresetRankingStore([])
    semantic_store = PresetRankingStore([scored_chunk("paraphrased-match")])

    results = HybridVectorStore(lexical_store, semantic_store).search("question")

    assert document_order(results) == ["paraphrased-match"]


def test_the_lexical_ranking_alone_is_used_when_the_semantic_store_finds_nothing():
    lexical_store = PresetRankingStore([scored_chunk("exact-term-match")])
    semantic_store = PresetRankingStore([])

    results = HybridVectorStore(lexical_store, semantic_store).search("question")

    assert document_order(results) == ["exact-term-match"]


def test_results_carry_the_fused_score_not_a_store_score():
    lexical_store = PresetRankingStore([scored_chunk("a", score=0.87)])
    semantic_store = PresetRankingStore([scored_chunk("a", score=0.41)])

    results = HybridVectorStore(lexical_store, semantic_store).search("question")

    assert results[0].score == pytest.approx(2 / (RRF_RANK_CONSTANT + 1))


def test_top_k_limits_the_fused_results():
    ranking = [scored_chunk(f"doc-{position}") for position in range(6)]
    hybrid_store = HybridVectorStore(PresetRankingStore(ranking), PresetRankingStore(ranking))

    assert len(hybrid_store.search("question", top_k=2)) == 2


def test_each_store_is_asked_for_more_candidates_than_top_k():
    lexical_store, semantic_store = PresetRankingStore(), PresetRankingStore()

    HybridVectorStore(lexical_store, semantic_store).search("question", top_k=3)

    assert lexical_store.requested_top_k == [CANDIDATES_PER_RANKER]
    assert semantic_store.requested_top_k == [CANDIDATES_PER_RANKER]


def test_a_top_k_larger_than_the_candidate_pool_is_still_honored():
    lexical_store, semantic_store = PresetRankingStore(), PresetRankingStore()

    HybridVectorStore(lexical_store, semantic_store).search(
        "question", top_k=CANDIDATES_PER_RANKER + 5
    )

    assert lexical_store.requested_top_k == [CANDIDATES_PER_RANKER + 5]


def test_a_chunk_ranked_deep_by_both_stores_can_beat_one_ranked_first_by_one():
    deep_ranking = [scored_chunk(f"filler-{position}") for position in range(4)]
    lexical_store = PresetRankingStore([scored_chunk("lexical-only"), *deep_ranking])
    semantic_store = PresetRankingStore([scored_chunk("semantic-only"), *deep_ranking])

    results = HybridVectorStore(lexical_store, semantic_store).search("question", top_k=3)

    assert "lexical-only" not in document_order(results)[:1]
    assert document_order(results)[0] == "filler-0"


def test_hybrid_over_real_stores_finds_documents_the_words_point_to():
    hybrid_store = HybridVectorStore(
        lexical_store=TfidfVectorStore(),
        semantic_store=QdrantVectorStore(
            embedder=HashingEmbedder(), client=QdrantClient(location=":memory:")
        ),
    )
    hybrid_store.add_documents(
        [
            Chunk(document_id="cats", index=0, text="Cats are small domesticated feline animals."),
            Chunk(document_id="rockets", index=0, text="Rockets use combustion to reach orbit."),
        ]
    )

    assert hybrid_store.search("feline animals")[0].chunk.document_id == "cats"
    assert hybrid_store.search("orbit combustion")[0].chunk.document_id == "rockets"
