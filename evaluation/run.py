"""Evaluates every retrieval mode on the golden set.

Run `python -m evaluation.run` to print the results. With `--enforce-gate` the
process also exits with status 1 when hybrid retrieval scores a lower hit rate
than the TF-IDF baseline.
"""

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import cache

from evaluation.golden_set import (
    GoldenQuestion,
    GoldenSet,
    QuestionKind,
    ensure_labels_exist,
    load_golden_set,
)
from evaluation.metrics import (
    HIT_RATE_CUTOFF,
    RANKING_DEPTH,
    RetrievalOutcome,
    RetrievalScores,
    hit_at_k,
    score_outcomes,
)
from evaluation.report import EvaluationReport, ModeReport, render_markdown
from rag.chunking import Chunk, split_into_chunks
from rag.embeddings import Embedder, FastEmbedEmbedder
from rag.retrieval_settings import RetrievalMode, RetrievalSettings
from rag.sample_corpus import load_sample_documents
from rag.store_factory import build_vector_store
from rag.vector_store import VectorStore

EXIT_OK = 0
EXIT_QUALITY_GATE_FAILED = 1

logger = logging.getLogger(__name__)


def run_evaluation(
    golden_set: GoldenSet,
    documents: Mapping[str, str],
    embedder_factory: Callable[[], Embedder],
) -> EvaluationReport:
    ensure_labels_exist(golden_set, documents.keys())
    chunks = _chunk_corpus(documents)
    # The semantic and hybrid modes share one embedder so the model loads once.
    shared_embedder_factory = cache(embedder_factory)
    mode_reports = [
        _evaluate_mode(mode, chunks, golden_set.questions, shared_embedder_factory)
        for mode in RetrievalMode
    ]
    return EvaluationReport(
        golden_set_version=golden_set.version,
        document_count=len(documents),
        chunk_count=len(chunks),
        embedding_model=shared_embedder_factory().model_name,
        modes=mode_reports,
    )


def hybrid_meets_tfidf_baseline(report: EvaluationReport) -> bool:
    hit_rates = {mode_report.mode: mode_report.overall.hit_rate for mode_report in report.modes}
    return hit_rates[RetrievalMode.HYBRID] >= hit_rates[RetrievalMode.TFIDF]


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: Callable[[], Embedder] = FastEmbedEmbedder,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = _parse_arguments(argv)

    report = run_evaluation(load_golden_set(), load_sample_documents(), embedder_factory)
    sys.stdout.write(render_markdown(report) + "\n")

    if arguments.enforce_gate and not hybrid_meets_tfidf_baseline(report):
        logger.error(
            "Quality gate failed: hybrid hit rate@%d is below the TF-IDF baseline", HIT_RATE_CUTOFF
        )
        return EXIT_QUALITY_GATE_FAILED
    return EXIT_OK


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes on the golden set.")
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="exit with status 1 if hybrid hit rate falls below the TF-IDF baseline",
    )
    return parser.parse_args(argv)


def _chunk_corpus(documents: Mapping[str, str]) -> list[Chunk]:
    return [
        chunk
        for document_id, text in documents.items()
        for chunk in split_into_chunks(document_id, text)
    ]


def _evaluate_mode(
    mode: RetrievalMode,
    chunks: list[Chunk],
    questions: list[GoldenQuestion],
    embedder_factory: Callable[[], Embedder],
) -> ModeReport:
    store = build_vector_store(RetrievalSettings(mode=mode), embedder_factory)
    store.add_documents(chunks)

    outcomes = [
        RetrievalOutcome(_rank_documents(store, question.text), question.relevant_documents)
        for question in questions
    ]
    _log_misses(mode, questions, outcomes)
    return ModeReport(
        mode=mode,
        overall=score_outcomes(outcomes),
        by_kind=_score_by_kind(questions, outcomes),
    )


def _rank_documents(store: VectorStore, question_text: str) -> tuple[str, ...]:
    matches = store.search(question_text, top_k=RANKING_DEPTH)
    return tuple(match.chunk.document_id for match in matches)


def _score_by_kind(
    questions: list[GoldenQuestion], outcomes: list[RetrievalOutcome]
) -> dict[QuestionKind, RetrievalScores]:
    outcomes_by_kind: dict[QuestionKind, list[RetrievalOutcome]] = defaultdict(list)
    for question, outcome in zip(questions, outcomes, strict=True):
        outcomes_by_kind[question.kind].append(outcome)
    return {kind: score_outcomes(kind_outcomes) for kind, kind_outcomes in outcomes_by_kind.items()}


def _log_misses(
    mode: RetrievalMode, questions: list[GoldenQuestion], outcomes: list[RetrievalOutcome]
) -> None:
    for question, outcome in zip(questions, outcomes, strict=True):
        if not hit_at_k(outcome, HIT_RATE_CUTOFF):
            logger.info(
                "[%s] miss at %d: %s %r -> %s",
                mode.value,
                HIT_RATE_CUTOFF,
                question.question_id,
                question.text,
                outcome.ranked_document_ids[:HIT_RATE_CUTOFF],
            )


if __name__ == "__main__":
    raise SystemExit(main())
