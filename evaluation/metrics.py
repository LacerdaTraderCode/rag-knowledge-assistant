"""Retrieval metrics computed from ranked document ids and the labeled relevant documents."""

from collections.abc import Sequence
from dataclasses import dataclass

HIT_RATE_CUTOFF = 3
RANKING_DEPTH = 10


@dataclass(frozen=True)
class RetrievalOutcome:
    """What one store returned for one question, next to the documents that answer it.

    `ranked_document_ids` holds one entry per retrieved chunk, best first, so a
    document can appear more than once.
    """

    ranked_document_ids: tuple[str, ...]
    relevant_documents: frozenset[str]


@dataclass(frozen=True)
class RetrievalScores:
    hits: int
    questions: int
    mean_reciprocal_rank: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.questions


def hit_at_k(outcome: RetrievalOutcome, k: int) -> bool:
    """Whether a relevant document appears among the first `k` retrieved chunks."""
    return any(
        document_id in outcome.relevant_documents for document_id in outcome.ranked_document_ids[:k]
    )


def reciprocal_rank(outcome: RetrievalOutcome) -> float:
    """1 divided by the position of the first relevant chunk, or 0 if none was retrieved.

    Rankings are cut at `RANKING_DEPTH` chunks, so a relevant document ranked
    deeper than that scores 0.
    """
    for position, document_id in enumerate(outcome.ranked_document_ids, start=1):
        if document_id in outcome.relevant_documents:
            return 1.0 / position
    return 0.0


def score_outcomes(
    outcomes: Sequence[RetrievalOutcome], k: int = HIT_RATE_CUTOFF
) -> RetrievalScores:
    if not outcomes:
        raise ValueError("Cannot score zero questions")
    return RetrievalScores(
        hits=sum(hit_at_k(outcome, k) for outcome in outcomes),
        questions=len(outcomes),
        mean_reciprocal_rank=sum(reciprocal_rank(outcome) for outcome in outcomes) / len(outcomes),
    )
