"""Tests for running every retrieval mode against a golden set and gating on the result."""

import logging

import pytest

from evaluation import run
from evaluation.golden_set import GoldenQuestion, GoldenSet, GoldenSetError, QuestionKind
from evaluation.metrics import RetrievalScores
from evaluation.report import EvaluationReport, ModeReport
from evaluation.run import (
    EXIT_OK,
    EXIT_QUALITY_GATE_FAILED,
    hybrid_meets_tfidf_baseline,
    main,
    run_evaluation,
)
from rag.retrieval_settings import RetrievalMode
from tests.fakes import HashingEmbedder

DOCUMENTS = {
    "cats": "Cats are small domesticated feline animals that purr.",
    "rockets": "Rockets use combustion to reach orbit around the planet.",
    "bread": "Sourdough bread rises slowly during a long cold fermentation.",
    "bridges": "Suspension bridges hang their deck from cables anchored at both ends.",
    "glaciers": "Glaciers are slow rivers of ice that carve valleys over centuries.",
    "tea": "Green tea is steeped briefly in water that is hot but not boiling.",
}

GOLDEN_SET = GoldenSet(
    version=7,
    questions=[
        GoldenQuestion(
            "q1", QuestionKind.LEXICAL_OVERLAP, "Which animals purr?", frozenset({"cats"})
        ),
        GoldenQuestion("q2", QuestionKind.EXACT_TERM, "combustion", frozenset({"rockets"})),
        GoldenQuestion("q3", QuestionKind.PARAPHRASE, "zzz qqq", frozenset({"bread"})),
    ],
)


class CountingEmbedderFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self) -> HashingEmbedder:
        self.calls += 1
        return HashingEmbedder()


def scores(hits: int, questions: int = 10) -> RetrievalScores:
    return RetrievalScores(hits=hits, questions=questions, mean_reciprocal_rank=0.5)


def report_with_hits(tfidf_hits: int, hybrid_hits: int) -> EvaluationReport:
    def mode_report(mode: RetrievalMode, hits: int) -> ModeReport:
        return ModeReport(
            mode=mode, overall=scores(hits), by_kind={QuestionKind.PARAPHRASE: scores(hits)}
        )

    return EvaluationReport(
        golden_set_version=1,
        document_count=1,
        chunk_count=1,
        embedding_model="stub",
        modes=[
            mode_report(RetrievalMode.TFIDF, tfidf_hits),
            mode_report(RetrievalMode.SEMANTIC, 0),
            mode_report(RetrievalMode.HYBRID, hybrid_hits),
        ],
    )


def test_every_retrieval_mode_is_evaluated_in_a_fixed_order():
    report = run_evaluation(GOLDEN_SET, DOCUMENTS, HashingEmbedder)

    assert [mode_report.mode for mode_report in report.modes] == list(RetrievalMode)


def test_the_report_describes_the_corpus_and_the_model_it_ran_on():
    report = run_evaluation(GOLDEN_SET, DOCUMENTS, HashingEmbedder)

    assert report.golden_set_version == 7
    assert report.document_count == len(DOCUMENTS)
    assert report.chunk_count == len(DOCUMENTS)
    assert report.embedding_model == "hashing-bag-of-words"


def test_tfidf_is_scored_overall_and_per_question_type():
    report = run_evaluation(GOLDEN_SET, DOCUMENTS, HashingEmbedder)
    tfidf = report.modes[0]

    assert (tfidf.overall.hits, tfidf.overall.questions) == (2, 3)
    assert tfidf.by_kind[QuestionKind.LEXICAL_OVERLAP].hits == 1
    assert tfidf.by_kind[QuestionKind.EXACT_TERM].hits == 1
    assert tfidf.by_kind[QuestionKind.PARAPHRASE].hits == 0
    assert tfidf.overall.mean_reciprocal_rank == pytest.approx(2 / 3)


def test_the_embedding_model_is_loaded_once_for_all_modes():
    embedder_factory = CountingEmbedderFactory()

    run_evaluation(GOLDEN_SET, DOCUMENTS, embedder_factory)

    assert embedder_factory.calls == 1


def test_a_label_for_a_missing_note_stops_the_run_before_any_model_loads():
    mislabeled = GoldenSet(
        version=1,
        questions=[GoldenQuestion("q1", QuestionKind.PARAPHRASE, "?", frozenset({"ghost"}))],
    )
    embedder_factory = CountingEmbedderFactory()

    with pytest.raises(GoldenSetError, match="ghost"):
        run_evaluation(mislabeled, DOCUMENTS, embedder_factory)

    assert embedder_factory.calls == 0


def test_missed_questions_are_logged_with_the_documents_that_were_returned(caplog):
    with caplog.at_level(logging.INFO, logger="evaluation.run"):
        run_evaluation(GOLDEN_SET, DOCUMENTS, HashingEmbedder)

    tfidf_misses = [
        record.getMessage() for record in caplog.records if "[tfidf]" in record.getMessage()
    ]
    assert len(tfidf_misses) == 1
    assert "q3" in tfidf_misses[0]


@pytest.mark.parametrize(
    ("tfidf_hits", "hybrid_hits", "passes"),
    [(8, 9, True), (8, 8, True), (8, 7, False)],
)
def test_the_gate_fails_only_when_hybrid_hits_fewer_questions_than_tfidf(
    tfidf_hits, hybrid_hits, passes
):
    assert hybrid_meets_tfidf_baseline(report_with_hits(tfidf_hits, hybrid_hits)) is passes


def test_the_command_prints_the_results_table_for_every_mode(capsys):
    exit_status = main([], embedder_factory=HashingEmbedder)

    output = capsys.readouterr().out
    assert exit_status == EXIT_OK
    for mode in RetrievalMode:
        assert f"| {mode.value} |" in output


def test_the_gate_flag_returns_a_failure_status_when_hybrid_regresses(monkeypatch, caplog):
    monkeypatch.setattr(run, "run_evaluation", lambda *arguments: report_with_hits(8, 5))

    with caplog.at_level(logging.ERROR, logger="evaluation.run"):
        exit_status = main(["--enforce-gate"], embedder_factory=HashingEmbedder)

    assert exit_status == EXIT_QUALITY_GATE_FAILED
    assert "Quality gate failed" in caplog.text


def test_without_the_gate_flag_a_regression_does_not_fail_the_command(monkeypatch):
    monkeypatch.setattr(run, "run_evaluation", lambda *arguments: report_with_hits(8, 5))

    assert main([], embedder_factory=HashingEmbedder) == EXIT_OK


def test_the_gate_flag_returns_success_when_hybrid_holds_the_baseline(monkeypatch):
    monkeypatch.setattr(run, "run_evaluation", lambda *arguments: report_with_hits(8, 8))

    assert main(["--enforce-gate"], embedder_factory=HashingEmbedder) == EXIT_OK
