"""Pydantic schemas for the RAG API."""

from pydantic import BaseModel


class IngestRequest(BaseModel):
    document_id: str
    text: str


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class SourceOut(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
