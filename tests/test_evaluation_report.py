"""Tests for the Markdown rendering of an evaluation report."""

from evaluation.golden_set import QuestionKind
from evaluation.metrics import RetrievalScores
from evaluation.report import EvaluationReport, ModeReport, render_markdown
from rag.retrieval_settings import RetrievalMode


def build_report() -> EvaluationReport:
    def mode_report(mode: RetrievalMode, hits: int, mrr: float) -> ModeReport:
        return ModeReport(
            mode=mode,
            overall=RetrievalScores(hits=hits, questions=4, mean_reciprocal_rank=mrr),
            by_kind={
                QuestionKind.LEXICAL_OVERLAP: RetrievalScores(
                    hits=hits - 1, questions=3, mean_reciprocal_rank=mrr
                ),
                QuestionKind.EXACT_TERM: RetrievalScores(
                    hits=1, questions=1, mean_reciprocal_rank=1.0
                ),
            },
        )

    return EvaluationReport(
        golden_set_version=2,
        document_count=18,
        chunk_count=36,
        embedding_model="some/model",
        modes=[
            mode_report(RetrievalMode.TFIDF, 3, 0.75),
            mode_report(RetrievalMode.SEMANTIC, 2, 0.5),
            mode_report(RetrievalMode.HYBRID, 4, 0.875),
        ],
    )


def test_the_report_opens_with_the_corpus_and_model_it_describes():
    lines = render_markdown(build_report()).splitlines()

    assert lines[0] == (
        "Golden set v2: 4 questions over 18 notes (36 chunks). Embedding model: some/model."
    )


def test_the_overall_table_shows_hit_rate_with_counts_and_mrr_per_mode():
    rendered = render_markdown(build_report())

    assert "| Mode | Hit rate@3 | MRR |" in rendered
    assert "| tfidf | 0.750 (3/4) | 0.750 |" in rendered
    assert "| semantic | 0.500 (2/4) | 0.500 |" in rendered
    assert "| hybrid | 1.000 (4/4) | 0.875 |" in rendered


def test_the_per_type_tables_list_only_the_types_that_were_asked():
    rendered = render_markdown(build_report())

    assert "| Mode | Lexical overlap (3) | Exact term (1) |" in rendered
    assert "Paraphrase" not in rendered


def test_the_per_type_tables_hold_hit_rate_and_mrr_for_each_mode():
    rendered = render_markdown(build_report())

    assert "| tfidf | 0.667 (2/3) | 1.000 (1/1) |" in rendered
    assert "| hybrid | 1.000 (3/3) | 1.000 (1/1) |" in rendered
    assert "| tfidf | 0.750 | 1.000 |" in rendered


def test_every_table_has_a_header_separator():
    rendered = render_markdown(build_report())

    assert rendered.count("| --- | --- | --- |") == 3
