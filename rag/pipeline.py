"""Wires chunking, the vector store, and answer generation into one pipeline."""

from dataclasses import dataclass

from rag.chunking import split_into_chunks
from rag.generation import ClaudeAnswerGenerator
from rag.vector_store import TfidfVectorStore


@dataclass(frozen=True)
class Source:
    document_id: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


class RagPipeline:
    def __init__(self, generator: ClaudeAnswerGenerator, store: TfidfVectorStore | None = None):
        self._store = store or TfidfVectorStore()
        self._generator = generator

    def ingest(self, document_id: str, text: str) -> int:
        """Chunk and index `text`, returning the number of chunks created."""
        chunks = split_into_chunks(document_id, text)
        self._store.add_documents(chunks)
        return len(chunks)

    async def ask(self, question: str, top_k: int = 3) -> Answer:
        results = self._store.search(question, top_k=top_k)
        if not results:
            return Answer(text="No indexed documents match this question.", sources=[])

        answer_text = await self._generator.generate(question, results)
        sources = [
            Source(
                document_id=result.chunk.document_id,
                chunk_index=result.chunk.index,
                text=result.chunk.text,
                score=result.score,
            )
            for result in results
        ]
        return Answer(text=answer_text, sources=sources)
