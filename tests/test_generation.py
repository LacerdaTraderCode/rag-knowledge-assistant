"""Tests for building the grounded prompt and calling the Claude API."""

import json

import httpx2

from rag.chunking import Chunk
from rag.generation import ClaudeAnswerGenerator, build_grounded_prompt
from rag.vector_store import SearchResult


def test_grounded_prompt_includes_every_source_and_the_question():
    results = [
        SearchResult(chunk=Chunk(document_id="doc-a", index=0, text="Cats are mammals."), score=0.9),
        SearchResult(chunk=Chunk(document_id="doc-b", index=1, text="Dogs are mammals too."), score=0.5),
    ]

    prompt = build_grounded_prompt("Are cats mammals?", results)

    assert "Cats are mammals." in prompt
    assert "Dogs are mammals too." in prompt
    assert "doc-a#0" in prompt
    assert "doc-b#1" in prompt
    assert "Are cats mammals?" in prompt


async def test_generator_sends_the_grounded_prompt_and_returns_the_answer(build_mock_client):
    captured = {}
    results = [SearchResult(chunk=Chunk(document_id="doc-a", index=0, text="Cats are mammals."), score=0.9)]

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={"content": [{"type": "text", "text": "Yes, cats are mammals [source 1]."}]})

    generator = ClaudeAnswerGenerator(
        api_key="test-key", model="test-model", client=build_mock_client(handler)
    )

    answer = await generator.generate("Are cats mammals?", results)

    assert answer == "Yes, cats are mammals [source 1]."
    assert captured["headers"]["x-api-key"] == "test-key"
    assert "Cats are mammals." in captured["body"]["messages"][0]["content"]
