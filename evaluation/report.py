"""The shape of an evaluation report and its Markdown rendering."""

from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.golden_set import QuestionKind
from evaluation.metrics import HIT_RATE_CUTOFF, RetrievalScores
from rag.retrieval_settings import RetrievalMode


@dataclass(frozen=True)
class ModeReport:
    mode: RetrievalMode
    overall: RetrievalScores
    by_kind: dict[QuestionKind, RetrievalScores]


@dataclass(frozen=True)
class EvaluationReport:
    golden_set_version: int
    document_count: int
    chunk_count: int
    embedding_model: str
    modes: list[ModeReport]


def render_markdown(report: EvaluationReport) -> str:
    question_count = report.modes[0].overall.questions
    description = (
        f"Golden set v{report.golden_set_version}: {question_count} questions over "
        f"{report.document_count} notes ({report.chunk_count} chunks). "
        f"Embedding model: {report.embedding_model}."
    )
    hit_rate_label = f"Hit rate@{HIT_RATE_CUTOFF}"
    kinds = [kind for kind in QuestionKind if kind in report.modes[0].by_kind]
    kind_headers = [
        f"{_kind_label(kind)} ({report.modes[0].by_kind[kind].questions})" for kind in kinds
    ]

    overall_table = _markdown_table(
        ["Mode", hit_rate_label, "MRR"],
        [
            [
                mode_report.mode.value,
                _format_hit_rate(mode_report.overall),
                f"{mode_report.overall.mean_reciprocal_rank:.3f}",
            ]
            for mode_report in report.modes
        ],
    )
    hit_rate_by_kind_table = _markdown_table(
        ["Mode", *kind_headers],
        [
            [mode_report.mode.value, *(_format_hit_rate(mode_report.by_kind[k]) for k in kinds)]
            for mode_report in report.modes
        ],
    )
    mrr_by_kind_table = _markdown_table(
        ["Mode", *kind_headers],
        [
            [
                mode_report.mode.value,
                *(f"{mode_report.by_kind[k].mean_reciprocal_rank:.3f}" for k in kinds),
            ]
            for mode_report in report.modes
        ],
    )
    return "\n\n".join(
        [
            description,
            overall_table,
            f"{hit_rate_label} by question type:",
            hit_rate_by_kind_table,
            "MRR by question type:",
            mrr_by_kind_table,
        ]
    )


def _format_hit_rate(scores: RetrievalScores) -> str:
    return f"{scores.hit_rate:.3f} ({scores.hits}/{scores.questions})"


def _kind_label(kind: QuestionKind) -> str:
    return kind.value.replace("_", " ").capitalize()


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    separator = ["---"] * len(header)
    return "\n".join(
        "| " + " | ".join(cells) + " |" for cells in [list(header), separator, *map(list, rows)]
    )
