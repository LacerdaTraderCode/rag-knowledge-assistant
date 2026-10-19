"""Turns text into vectors behind a small interface, so no store depends on one model."""

from collections.abc import Sequence
from typing import Protocol

from fastembed import TextEmbedding

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    """Maps text to fixed-size vectors in which closer vectors mean closer meaning."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """ONNX embeddings through fastembed, which keeps PyTorch out of the install.

    Constructing it downloads the model on first use and loads it into memory, so
    it must be built at application startup and never at import time.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, cache_dir: str | None = None):
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._dimension = TextEmbedding.get_embedding_size(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()
