"""The HTTP contract must not change with the retrieval mode: same routes, same response shapes."""

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline
from rag.retrieval_settings import RetrievalMode, RetrievalSettings
from rag.store_factory import build_vector_store
from tests.fakes import HashingEmbedder

SOURCE_FIELDS = {"document_id", "chunk_index", "text", "score"}


def build_client(mode: RetrievalMode) -> TestClient:
    def answer_with_a_fixed_reply(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"content": [{"type": "text", "text": "Cats are mammals [source 1]."}]}
        )

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(answer_with_a_fixed_reply))
    generator = ClaudeAnswerGenerator(api_key="test-key", model="test-model", client=http_client)
    store = build_vector_store(RetrievalSettings(mode=mode), HashingEmbedder)
    return TestClient(create_app(pipeline=RagPipeline(generator=generator, store=store)))


@pytest.fixture(params=list(RetrievalMode), ids=lambda mode: mode.value)
def client(request) -> TestClient:
    return build_client(request.param)


def test_ingesting_reports_the_document_and_chunk_count(client):
    response = client.post(
        "/documents", json={"document_id": "animals", "text": "Cats are small mammals."}
    )

    assert response.status_code == 200
    assert response.json() == {"document_id": "animals", "chunks_created": 1}


def test_querying_an_empty_index_returns_the_same_no_match_answer(client):
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "No indexed documents match this question.",
        "sources": [],
    }


def test_a_query_returns_the_answer_with_sources_in_the_original_shape(client):
    client.post(
        "/documents",
        json={"document_id": "animals", "text": "Cats are small domesticated mammals."},
    )

    payload = client.post("/query", json={"question": "Are cats mammals?"}).json()

    assert set(payload) == {"answer", "sources"}
    assert payload["answer"] == "Cats are mammals [source 1]."
    assert payload["sources"][0]["document_id"] == "animals"
    assert set(payload["sources"][0]) == SOURCE_FIELDS
    assert isinstance(payload["sources"][0]["score"], float)


def test_top_k_still_limits_the_number_of_sources(client):
    for position in range(4):
        client.post(
            "/documents",
            json={"document_id": f"note-{position}", "text": "Cats are small mammals."},
        )

    payload = client.post("/query", json={"question": "cats mammals", "top_k": 2}).json()

    assert len(payload["sources"]) == 2


def test_a_top_k_of_zero_still_answers_instead_of_failing(client):
    client.post("/documents", json={"document_id": "animals", "text": "Cats are mammals."})

    response = client.post("/query", json={"question": "cats", "top_k": 0})

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_the_published_routes_and_request_fields_are_unchanged():
    schema = build_client(RetrievalMode.TFIDF).get("/openapi.json").json()

    assert set(schema["paths"]) == {"/documents", "/query"}
    schemas = schema["components"]["schemas"]
    assert set(schemas["IngestRequest"]["properties"]) == {"document_id", "text"}
    assert set(schemas["QueryRequest"]["properties"]) == {"question", "top_k"}
    assert schemas["QueryRequest"]["properties"]["top_k"]["default"] == 3
    assert set(schemas["QueryResponse"]["properties"]) == {"answer", "sources"}
    assert set(schemas["SourceOut"]["properties"]) == SOURCE_FIELDS
