"""Tests for Reciprocal Rank Fusion, including the arithmetic of the fused scores."""

import pytest

from rag.fusion import RRF_RANK_CONSTANT, reciprocal_rank_fusion
from tests.fakes import scored_chunk


def document_order(results) -> list[str]:
    return [result.chunk.document_id for result in results]


def test_a_chunk_ranked_first_in_two_rankings_scores_twice_the_single_rank_score():
    results = reciprocal_rank_fusion([[scored_chunk("a")], [scored_chunk("a")]], top_k=1)

    assert results[0].score == pytest.approx(2 / (RRF_RANK_CONSTANT + 1))


def test_each_rank_contributes_one_over_the_rank_constant_plus_the_rank():
    ranking = [scored_chunk("a"), scored_chunk("b"), scored_chunk("c")]

    results = reciprocal_rank_fusion([ranking], top_k=3)

    assert [result.score for result in results] == pytest.approx(
        [1 / (RRF_RANK_CONSTANT + rank) for rank in (1, 2, 3)]
    )


def test_a_chunk_found_by_both_rankings_beats_chunks_found_by_only_one():
    lexical = [scored_chunk("only-lexical"), scored_chunk("shared")]
    semantic = [scored_chunk("only-semantic"), scored_chunk("shared")]

    results = reciprocal_rank_fusion([lexical, semantic], top_k=3)

    assert document_order(results)[0] == "shared"


def test_the_raw_scores_of_the_rankings_are_ignored():
    small_scale = [scored_chunk("a", score=0.9), scored_chunk("b", score=0.1)]
    huge_scale = [scored_chunk("b", score=9000.0), scored_chunk("a", score=1.0)]

    results = reciprocal_rank_fusion([small_scale, huge_scale], top_k=2)

    assert [result.score for result in results] == pytest.approx(
        [1 / (RRF_RANK_CONSTANT + 1) + 1 / (RRF_RANK_CONSTANT + 2)] * 2
    )


def test_top_k_limits_the_number_of_fused_results():
    ranking = [scored_chunk("a"), scored_chunk("b"), scored_chunk("c")]

    assert document_order(reciprocal_rank_fusion([ranking], top_k=2)) == ["a", "b"]


def test_equal_scores_keep_the_order_in_which_chunks_were_first_seen():
    results = reciprocal_rank_fusion([[scored_chunk("first")], [scored_chunk("second")]], top_k=2)

    assert document_order(results) == ["first", "second"]


def test_chunks_of_the_same_document_are_told_apart_by_their_index():
    ranking = [scored_chunk("doc", index=0), scored_chunk("doc", index=1)]

    results = reciprocal_rank_fusion([ranking], top_k=2)

    assert [result.chunk.index for result in results] == [0, 1]


def test_a_chunk_repeated_within_one_ranking_counts_once_at_its_best_position():
    duplicated = [scored_chunk("a"), scored_chunk("a"), scored_chunk("b")]

    results = reciprocal_rank_fusion([duplicated], top_k=2)

    assert document_order(results) == ["a", "b"]
    assert results[0].score == pytest.approx(1 / (RRF_RANK_CONSTANT + 1))
    assert results[1].score == pytest.approx(1 / (RRF_RANK_CONSTANT + 2))


def test_fusing_a_single_ranking_keeps_its_order():
    ranking = [scored_chunk("a"), scored_chunk("b"), scored_chunk("c")]

    assert document_order(reciprocal_rank_fusion([ranking], top_k=3)) == ["a", "b", "c"]


def test_empty_rankings_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []], top_k=3) == []
    assert reciprocal_rank_fusion([], top_k=3) == []


def test_an_empty_ranking_does_not_disturb_the_other_one():
    ranking = [scored_chunk("a"), scored_chunk("b")]

    assert document_order(reciprocal_rank_fusion([[], ranking], top_k=2)) == ["a", "b"]
