"""Tests for the retrieval metrics, checked against values worked out by hand."""

import pytest

from evaluation.metrics import RetrievalOutcome, hit_at_k, reciprocal_rank, score_outcomes


def outcome(ranked_document_ids: list[str], relevant_documents: set[str]) -> RetrievalOutcome:
    return RetrievalOutcome(
        ranked_document_ids=tuple(ranked_document_ids),
        relevant_documents=frozenset(relevant_documents),
    )


def test_a_relevant_document_at_the_cutoff_position_is_a_hit():
    assert hit_at_k(outcome(["x", "y", "a"], {"a"}), k=3)


def test_a_relevant_document_just_past_the_cutoff_is_a_miss():
    assert not hit_at_k(outcome(["x", "y", "z", "a"], {"a"}), k=3)


def test_any_of_several_relevant_documents_counts_as_a_hit():
    assert hit_at_k(outcome(["x", "b", "y"], {"a", "b"}), k=3)


def test_chunks_of_one_document_fill_the_cutoff_like_any_other_chunks():
    ranking = ["x", "x", "x", "a"]

    assert not hit_at_k(outcome(ranking, {"a"}), k=3)
    assert hit_at_k(outcome(ranking, {"a"}), k=4)


def test_an_empty_ranking_is_a_miss():
    assert not hit_at_k(outcome([], {"a"}), k=3)


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 1 / 2),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank_is_one_over_the_position_of_the_first_relevant_chunk(ranking, expected):
    assert reciprocal_rank(outcome(ranking, {"a"})) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_earliest_of_several_relevant_documents():
    assert reciprocal_rank(outcome(["x", "b", "a"], {"a", "b"})) == pytest.approx(1 / 2)


def test_reciprocal_rank_counts_every_chunk_that_comes_before_the_answer():
    assert reciprocal_rank(outcome(["x", "x", "a"], {"a"})) == pytest.approx(1 / 3)


def test_scores_aggregate_hits_and_the_mean_reciprocal_rank():
    outcomes = [
        outcome(["a", "x"], {"a"}),
        outcome(["x", "y", "a"], {"a"}),
        outcome(["x", "y", "z"], {"a"}),
    ]

    scores = score_outcomes(outcomes, k=3)

    assert scores.hits == 2
    assert scores.questions == 3
    assert scores.hit_rate == pytest.approx(2 / 3)
    assert scores.mean_reciprocal_rank == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_scores_honor_the_requested_cutoff():
    outcomes = [outcome(["x", "a"], {"a"})]

    assert score_outcomes(outcomes, k=1).hits == 0
    assert score_outcomes(outcomes, k=2).hits == 1


def test_the_default_cutoff_is_three():
    assert score_outcomes([outcome(["x", "y", "a"], {"a"})]).hits == 1
    assert score_outcomes([outcome(["x", "y", "z", "a"], {"a"})]).hits == 0


def test_scoring_zero_questions_is_an_error():
    with pytest.raises(ValueError, match="zero questions"):
        score_outcomes([])
