"""Builds the default pipeline from environment variables when the server starts."""

import logging
import os
from collections.abc import Mapping

from rag.generation import ClaudeAnswerGenerator
from rag.pipeline import RagPipeline
from rag.retrieval_settings import load_retrieval_settings
from rag.sample_corpus import load_sample_documents
from rag.store_factory import build_vector_store

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"

logger = logging.getLogger(__name__)


def build_default_pipeline(environ: Mapping[str, str] = os.environ) -> RagPipeline:
    """Create the pipeline, load the retrieval store it needs, and ingest the sample corpus."""
    retrieval_settings = load_retrieval_settings(environ)
    logger.info("Starting with retrieval mode %s", retrieval_settings.mode)

    generator = ClaudeAnswerGenerator(
        api_key=environ.get("ANTHROPIC_API_KEY", ""),
        model=environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
    )
    pipeline = RagPipeline(generator=generator, store=build_vector_store(retrieval_settings))
    for document_id, text in load_sample_documents().items():
        pipeline.ingest(document_id, text)
    return pipeline
