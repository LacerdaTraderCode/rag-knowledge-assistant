"""Assembles the vector store that matches the configured retrieval mode."""

from collections.abc import Callable

from qdrant_client import QdrantClient

from rag.embeddings import Embedder, FastEmbedEmbedder
from rag.hybrid_store import HybridVectorStore
from rag.qdrant_store import QdrantVectorStore
from rag.retrieval_settings import RetrievalMode, RetrievalSettings
from rag.vector_store import TfidfVectorStore, VectorStore

IN_MEMORY_QDRANT_LOCATION = ":memory:"


def build_vector_store(
    settings: RetrievalSettings,
    embedder_factory: Callable[[], Embedder] = FastEmbedEmbedder,
) -> VectorStore:
    """Build the store for `settings.mode`.

    `embedder_factory` is only called for the semantic and hybrid modes, so the
    `tfidf` mode never loads, or even downloads, an embedding model.
    """
    if settings.mode is RetrievalMode.TFIDF:
        return TfidfVectorStore()

    semantic_store = QdrantVectorStore(
        embedder=embedder_factory(), client=_connect_to_qdrant(settings)
    )
    if settings.mode is RetrievalMode.SEMANTIC:
        return semantic_store
    return HybridVectorStore(lexical_store=TfidfVectorStore(), semantic_store=semantic_store)


def _connect_to_qdrant(settings: RetrievalSettings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(location=IN_MEMORY_QDRANT_LOCATION)
