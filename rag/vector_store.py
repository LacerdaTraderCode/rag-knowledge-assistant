"""A small TF-IDF backed vector store for similarity search over chunks.

TF-IDF is the deliberate baseline here: it needs no model download and no API
key, so the whole pipeline runs and is fully tested offline. The public
interface below (`add_documents`, `search`) does not expose anything
TF-IDF-specific, so swapping in a dense embedding model later only touches
this file.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunking import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


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

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
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
