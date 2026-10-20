"""FastAPI application exposing the RAG pipeline: ingest documents, then ask questions."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.pipeline_factory import build_default_pipeline
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceOut
from rag.pipeline import RagPipeline

APP_VERSION = "1.1.0"


@asynccontextmanager
async def _build_pipeline_on_startup(app: FastAPI) -> AsyncIterator[None]:
    # Building the pipeline loads the embedding model and connects to Qdrant, so
    # it happens here and never at import time.
    if app.state.pipeline is None:
        app.state.pipeline = build_default_pipeline()
    yield


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    """Create the API around `pipeline`, or around the environment-configured one at startup."""
    app = FastAPI(
        title="RAG Knowledge Assistant", version=APP_VERSION, lifespan=_build_pipeline_on_startup
    )
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
