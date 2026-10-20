"""The retrieval contract shared by every store, plus the TF-IDF store.

TF-IDF is the lexical baseline and a first-class retrieval mode: it needs no
model download and no API key, so it runs fully offline. The semantic store
(`rag.qdrant_store`) and the hybrid combiner (`rag.hybrid_store`) implement the
same `VectorStore` contract, which is why the pipeline never depends on one
concrete store.
"""

from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunking import Chunk

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """Indexes chunks and ranks them against a query, best match first.

    Every store scores on its own scale, so scores are only comparable within
    the results of a single `search` call.
    """

    def add_documents(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]: ...


class TfidfVectorStore:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._chunks: list[Chunk] = []
        self._matrix = None

    def add_documents(self, chunks: list[Chunk]) -> None:
        # Refitting on the full accumulated set keeps every chunk comparable in
        # the same vector space; for a store this size that costs far less than
        # the complexity of an incremental TF-IDF update would.
        self._chunks.extend(chunks)
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in self._chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if not self._chunks:
            return []
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._matrix)[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]
        return [
            SearchResult(chunk=self._chunks[i], score=float(similarities[i]))
            for i in ranked_indices
            if similarities[i] > 0
        ]
