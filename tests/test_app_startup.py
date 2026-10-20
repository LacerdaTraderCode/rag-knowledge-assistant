"""Tests that the default pipeline is built at server startup, never at import time."""

import httpx2
from fastapi.testclient import TestClient

from app import main
from app.main import create_app
from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline
from rag.vector_store import TfidfVectorStore


class PipelineBuilderSpy:
    def __init__(self, pipeline: RagPipeline):
        self._pipeline = pipeline
        self.calls = 0

    def __call__(self) -> RagPipeline:
        self.calls += 1
        return self._pipeline


def build_offline_pipeline() -> RagPipeline:
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200))
    )
    generator = ClaudeAnswerGenerator(api_key="test-key", model="test-model", client=client)
    return RagPipeline(generator=generator, store=TfidfVectorStore())


def test_creating_the_app_builds_no_pipeline(monkeypatch):
    spy = PipelineBuilderSpy(build_offline_pipeline())
    monkeypatch.setattr(main, "build_default_pipeline", spy)

    create_app()

    assert spy.calls == 0


def test_starting_the_server_builds_the_pipeline_once(monkeypatch):
    spy = PipelineBuilderSpy(build_offline_pipeline())
    monkeypatch.setattr(main, "build_default_pipeline", spy)

    with TestClient(create_app()) as client:
        client.post("/documents", json={"document_id": "notes", "text": "Cats purr."})
        client.post("/documents", json={"document_id": "more", "text": "Dogs bark."})

    assert spy.calls == 1


def test_the_pipeline_built_at_startup_serves_requests(monkeypatch):
    monkeypatch.setattr(main, "build_default_pipeline", build_offline_pipeline)

    with TestClient(create_app()) as client:
        response = client.post("/documents", json={"document_id": "notes", "text": "Cats purr."})

    assert response.json() == {"document_id": "notes", "chunks_created": 1}


def test_an_injected_pipeline_is_used_and_nothing_is_built_at_startup(monkeypatch):
    spy = PipelineBuilderSpy(build_offline_pipeline())
    monkeypatch.setattr(main, "build_default_pipeline", spy)
    injected_pipeline = build_offline_pipeline()

    with TestClient(create_app(pipeline=injected_pipeline)) as client:
        client.post("/documents", json={"document_id": "notes", "text": "Cats purr."})

    assert spy.calls == 0
    assert injected_pipeline._store.search("cats")[0].chunk.document_id == "notes"
