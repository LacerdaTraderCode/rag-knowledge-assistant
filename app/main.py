"""FastAPI application exposing the RAG pipeline: ingest documents, then ask questions."""

import os

from fastapi import FastAPI

from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceOut
from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline
from rag.sample_corpus import load_sample_documents


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

    if pipeline is None:
        generator = ClaudeAnswerGenerator(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        )
        pipeline = RagPipeline(generator=generator)
        for document_id, text in load_sample_documents().items():
            pipeline.ingest(document_id, text)

    app.state.pipeline = pipeline

    @app.post("/documents", response_model=IngestResponse)
    def ingest_document(payload: IngestRequest) -> IngestResponse:
        chunks_created = app.state.pipeline.ingest(payload.document_id, payload.text)
        return IngestResponse(document_id=payload.document_id, chunks_created=chunks_created)

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
        answer = await app.state.pipeline.ask(payload.question, top_k=payload.top_k)
        return QueryResponse(
            answer=answer.text,
            sources=[
                SourceOut(
                    document_id=source.document_id,
                    chunk_index=source.chunk_index,
                    text=source.text,
                    score=source.score,
                )
                for source in answer.sources
            ],
        )

    return app


app = create_app()
