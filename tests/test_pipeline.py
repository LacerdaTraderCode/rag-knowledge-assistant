"""Tests for the end-to-end pipeline: ingest, retrieve, and generate a cited answer."""

import httpx2

from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline


def _pipeline_with_generator(handler) -> RagPipeline:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    generator = ClaudeAnswerGenerator(api_key="test-key", model="test-model", client=client)
    return RagPipeline(generator=generator)


async def test_asking_before_ingesting_anything_returns_no_results():
    pipeline = _pipeline_with_generator(lambda request: httpx2.Response(200, json={}))

    answer = await pipeline.ask("anything")

    assert answer.sources == []
    assert "No indexed documents" in answer.text


async def test_ingest_reports_how_many_chunks_it_created():
    pipeline = _pipeline_with_generator(lambda request: httpx2.Response(200, json={}))

    chunk_count = pipeline.ingest("doc-a", "one two three")

    assert chunk_count == 1


async def test_ask_returns_the_generated_answer_with_matching_sources():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"content": [{"type": "text", "text": "Cats are mammals."}]})

    pipeline = _pipeline_with_generator(handler)
    pipeline.ingest("animals", "Cats are small domesticated feline mammals.")
    pipeline.ingest("space", "Rockets use combustion to reach orbit.")

    answer = await pipeline.ask("Are cats mammals?")

    assert answer.text == "Cats are mammals."
    assert answer.sources[0].document_id == "animals"
