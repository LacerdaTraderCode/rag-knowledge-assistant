"""Reciprocal Rank Fusion: merges several rankings without comparing their scores."""

from collections.abc import Sequence

from rag.chunking import Chunk
from rag.vector_store import SearchResult

RRF_RANK_CONSTANT = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], top_k: int
) -> list[SearchResult]:
    """Fuse `rankings` into one list of at most `top_k` results, best first.

    A chunk's fused score is the sum of 1 / (RRF_RANK_CONSTANT + rank) over every
    ranking it appears in. Only ranks are used, never the raw scores, which is
    what makes rankings from different scorers (TF-IDF cosine, embedding cosine)
    safe to combine. A chunk listed twice in one ranking counts once, at its best
    position. Equal fused scores keep the order in which the chunks were first
    seen, so earlier rankings win ties.
    """
    fused_scores: dict[tuple[str, int], float] = {}
    chunks_by_key: dict[tuple[str, int], Chunk] = {}

    for ranking in rankings:
        for rank, scored_chunk in enumerate(_first_occurrences(ranking), start=1):
            key = _chunk_key(scored_chunk.chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)
            chunks_by_key.setdefault(key, scored_chunk.chunk)

    best_first = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
    return [
        SearchResult(chunk=chunks_by_key[key], score=fused_scores[key])
        for key in best_first[:top_k]
    ]


def _first_occurrences(ranking: Sequence[SearchResult]) -> list[SearchResult]:
    seen: set[tuple[str, int]] = set()
    distinct: list[SearchResult] = []
    for scored_chunk in ranking:
        key = _chunk_key(scored_chunk.chunk)
        if key not in seen:
            seen.add(key)
            distinct.append(scored_chunk)
    return distinct


def _chunk_key(chunk: Chunk) -> tuple[str, int]:
    return (chunk.document_id, chunk.index)
