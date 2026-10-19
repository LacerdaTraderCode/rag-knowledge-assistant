"""Semantic vector store: chunk embeddings indexed and searched in Qdrant."""

import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from rag.chunking import Chunk
from rag.embeddings import Embedder
from rag.vector_store import DEFAULT_TOP_K, SearchResult

COLLECTION_NAME = "knowledge_chunks"
UPSERT_BATCH_SIZE = 64
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "rag-knowledge-assistant")


class QdrantVectorStore:
    """Ranks chunks by cosine similarity between their embeddings and the query's.

    The client decides where Qdrant runs: `QdrantClient(location=":memory:")` for
    tests and quick starts, `QdrantClient(url=...)` for a real server.
    """

    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
    ):
        self._embedder = embedder
        self._client = client
        self._collection_name = collection_name
        self._recreate_collection()

    def add_documents(self, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            self._upsert_batch(chunks[start : start + UPSERT_BATCH_SIZE])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        # Qdrant rejects a limit below 1 while the TF-IDF store answers with an
        # empty list, so the stores stay interchangeable only if this one does too.
        if top_k < 1:
            return []
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=self._embedder.embed_query(query),
            limit=top_k,
        )
        return [_to_search_result(point) for point in response.points]

    def _recreate_collection(self) -> None:
        # Starting empty keeps the lexical and semantic indexes describing the
        # same chunks after a restart; the collection is an index, not the
        # system of record.
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._embedder.dimension, distance=Distance.COSINE),
        )

    def _upsert_batch(self, chunks: list[Chunk]) -> None:
        vectors = self._embedder.embed_passages([chunk.text for chunk in chunks])
        points = [
            PointStruct(id=_point_id(chunk), vector=vector, payload=asdict(chunk))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)


def _point_id(chunk: Chunk) -> str:
    # Deterministic ids make re-ingesting the same chunk overwrite it instead of
    # storing a duplicate.
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{chunk.document_id}#{chunk.index}"))


def _to_search_result(point: ScoredPoint) -> SearchResult:
    return SearchResult(chunk=Chunk(**point.payload), score=point.score)
