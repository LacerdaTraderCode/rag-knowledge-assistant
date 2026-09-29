"""Integration tests for the FastAPI endpoints, using an injected pipeline."""

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline


@pytest.fixture
def client():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"content": [{"type": "text", "text": "Cats are mammals [source 1]."}]}
        )

    mock_http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    generator = ClaudeAnswerGenerator(api_key="test-key", model="test-model", client=mock_http_client)
    pipeline = RagPipeline(generator=generator)
    app = create_app(pipeline=pipeline)
    return TestClient(app)


def test_ingesting_a_document_reports_the_chunk_count(client):
    response = client.post(
        "/documents", json={"document_id": "animals", "text": "Cats are small mammals."}
    )

    assert response.status_code == 200
    assert response.json() == {"document_id": "animals", "chunks_created": 1}


def test_querying_before_any_document_is_ingested_returns_no_sources(client):
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_querying_after_ingesting_returns_a_cited_answer(client):
    client.post("/documents", json={"document_id": "animals", "text": "Cats are small domesticated mammals."})

    response = client.post("/query", json={"question": "Are cats mammals?"})

    payload = response.json()
    assert payload["answer"] == "Cats are mammals [source 1]."
    assert payload["sources"][0]["document_id"] == "animals"
