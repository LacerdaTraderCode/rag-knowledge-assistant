"""Hybrid retrieval: fuses a lexical and a semantic ranking with Reciprocal Rank Fusion."""

from rag.chunking import Chunk
from rag.fusion import reciprocal_rank_fusion
from rag.vector_store import DEFAULT_TOP_K, SearchResult, VectorStore

CANDIDATES_PER_RANKER = 20


class HybridVectorStore:
    """Combines two stores behind the common `VectorStore` contract.

    Each store ranks the query on its own and only the ranks are fused, so the
    two scoring scales never have to be reconciled.
    """

    def __init__(self, lexical_store: VectorStore, semantic_store: VectorStore):
        self._lexical_store = lexical_store
        self._semantic_store = semantic_store

    def add_documents(self, chunks: list[Chunk]) -> None:
        self._lexical_store.add_documents(chunks)
        self._semantic_store.add_documents(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        # Each store is asked for more than `top_k` candidates: a chunk ranked
        # fifth by both stores can beat one ranked first by only one of them.
        candidate_count = max(top_k, CANDIDATES_PER_RANKER)
        lexical_ranking = self._lexical_store.search(query, top_k=candidate_count)
        semantic_ranking = self._semantic_store.search(query, top_k=candidate_count)
        return reciprocal_rank_fusion([lexical_ranking, semantic_ranking], top_k=top_k)
