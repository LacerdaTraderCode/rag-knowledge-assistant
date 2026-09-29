"""Turns retrieved chunks and a question into a grounded, cited answer."""

import httpx2

from rag.vector_store import SearchResult

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def build_grounded_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = "\n\n".join(
        f"[source {i + 1}: {result.chunk.document_id}#{result.chunk.index}]\n{result.chunk.text}"
        for i, result in enumerate(results)
    )
    return (
        "Answer the question using only the sources below. "
        "Cite each source you use as [source N]. "
        "If the sources do not contain the answer, say so instead of guessing.\n\n"
        f"Sources:\n{context_blocks}\n\nQuestion: {question}"
    )


class ClaudeAnswerGenerator:
    def __init__(self, api_key: str, model: str, client: httpx2.AsyncClient | None = None):
        self._api_key = api_key
        self._model = model
        self._client = client or httpx2.AsyncClient()

    async def generate(self, question: str, results: list[SearchResult]) -> str:
        prompt = build_grounded_prompt(question, results)
        response = await self._client.post(
            CLAUDE_API_URL,
            headers={"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION},
            json={
                "model": self._model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
