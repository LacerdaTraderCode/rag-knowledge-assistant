"""Tests for the fastembed adapter, using a stand-in for the model so nothing is downloaded."""

import numpy as np
import pytest

from rag import embeddings
from rag.embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedEmbedder

STUB_DIMENSION = 3


class StubTextEmbedding:
    """Mimics the parts of fastembed's TextEmbedding that the adapter uses."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        return STUB_DIMENSION

    def passage_embed(self, texts):
        self.passage_calls.append(list(texts))
        return (
            np.full(STUB_DIMENSION, float(index), dtype=np.float32) for index, _ in enumerate(texts)
        )

    def query_embed(self, query):
        self.query_calls.append(query)
        return iter([np.array([0.5, 0.25, 0.125], dtype=np.float32)])


@pytest.fixture
def embedder(monkeypatch) -> FastEmbedEmbedder:
    monkeypatch.setattr(embeddings, "TextEmbedding", StubTextEmbedding)
    return FastEmbedEmbedder()


def test_the_default_model_is_a_multilingual_one(embedder):
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL
    assert "multilingual" in embedder.model_name


def test_the_dimension_comes_from_the_model_registry(embedder):
    assert embedder.dimension == STUB_DIMENSION


def test_passages_become_plain_python_float_lists(embedder):
    vectors = embedder.embed_passages(["first", "second"])

    assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert all(isinstance(number, float) for vector in vectors for number in vector)


def test_passages_use_the_passage_side_of_the_model(embedder):
    embedder.embed_passages(["first", "second"])

    assert embedder._model.passage_calls == [["first", "second"]]
    assert embedder._model.query_calls == []


def test_a_query_becomes_a_single_float_list_through_the_query_side(embedder):
    vector = embedder.embed_query("what is a cache?")

    assert vector == [0.5, 0.25, 0.125]
    assert embedder._model.query_calls == ["what is a cache?"]


def test_the_model_name_and_cache_directory_reach_the_model(monkeypatch):
    monkeypatch.setattr(embeddings, "TextEmbedding", StubTextEmbedding)

    embedder = FastEmbedEmbedder(model_name="some/model", cache_dir="/tmp/models")

    assert embedder.model_name == "some/model"
    assert embedder._model.cache_dir == "/tmp/models"
