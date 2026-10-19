"""Deterministic test doubles shared by the retrieval tests."""

import hashlib
import re
from collections.abc import Sequence

from rag.chunking import Chunk
from rag.vector_store import DEFAULT_TOP_K, SearchResult

HASHING_EMBEDDER_DIMENSION = 256
WORD_HASH_BYTES = 4


class HashingEmbedder:
    """Bag-of-words embedder: every word hashes into one bucket of the vector.

    Texts that share words end up close together, which is enough to exercise the
    stores without a model, and the vectors are identical on every run.
    """

    model_name = "hashing-bag-of-words"
    dimension = HASHING_EMBEDDER_DIMENSION

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=WORD_HASH_BYTES).digest()
            vector[int.from_bytes(digest, "big") % self.dimension] += 1.0
        return vector


class PresetRankingStore:
    """A store that ignores the query and answers with a fixed ranking.

    It also remembers what it was given and how many results it was asked for.
    """

    def __init__(self, ranking: list[SearchResult] | None = None):
        self._ranking = ranking or []
        self.added_chunks: list[Chunk] = []
        self.requested_top_k: list[int] = []

    def add_documents(self, chunks: list[Chunk]) -> None:
        self.added_chunks.extend(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        self.requested_top_k.append(top_k)
        return self._ranking[:top_k]


def scored_chunk(document_id: str, index: int = 0, score: float = 1.0) -> SearchResult:
    return SearchResult(chunk=Chunk(document_id=document_id, index=index, text="..."), score=score)
