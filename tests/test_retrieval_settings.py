"""Tests for reading the retrieval mode and the Qdrant location from the environment."""

import pytest

from rag.retrieval_settings import (
    DEFAULT_RETRIEVAL_MODE,
    InvalidRetrievalModeError,
    RetrievalMode,
    load_retrieval_settings,
)


def test_hybrid_is_the_default_mode():
    assert DEFAULT_RETRIEVAL_MODE is RetrievalMode.HYBRID
    assert load_retrieval_settings({}).mode is RetrievalMode.HYBRID


@pytest.mark.parametrize("name", ["tfidf", "semantic", "hybrid"])
def test_every_documented_mode_can_be_selected(name):
    settings = load_retrieval_settings({"RETRIEVAL_MODE": name})

    assert settings.mode is RetrievalMode(name)


def test_the_mode_name_ignores_case_and_surrounding_spaces():
    assert load_retrieval_settings({"RETRIEVAL_MODE": "  TfIdf "}).mode is RetrievalMode.TFIDF


def test_a_blank_mode_falls_back_to_the_default():
    assert load_retrieval_settings({"RETRIEVAL_MODE": "  "}).mode is DEFAULT_RETRIEVAL_MODE


def test_an_unknown_mode_is_rejected_and_the_valid_ones_are_listed():
    with pytest.raises(InvalidRetrievalModeError) as error:
        load_retrieval_settings({"RETRIEVAL_MODE": "bm25"})

    message = str(error.value)
    assert "bm25" in message
    assert all(mode.value in message for mode in RetrievalMode)


def test_without_a_qdrant_url_the_in_memory_mode_is_implied():
    assert load_retrieval_settings({}).qdrant_url is None
    assert load_retrieval_settings({"QDRANT_URL": "  "}).qdrant_url is None


def test_a_qdrant_url_is_passed_through():
    settings = load_retrieval_settings({"QDRANT_URL": "http://localhost:6333"})

    assert settings.qdrant_url == "http://localhost:6333"
