"""The retrieval golden set: questions labeled with the documents that answer them."""

import json
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
REQUIRED_FIELDS = ("id", "kind", "question", "relevant_documents")


class QuestionKind(StrEnum):
    LEXICAL_OVERLAP = "lexical_overlap"
    PARAPHRASE = "paraphrase"
    EXACT_TERM = "exact_term"


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    kind: QuestionKind
    text: str
    relevant_documents: frozenset[str]


@dataclass(frozen=True)
class GoldenSet:
    version: int
    questions: list[GoldenQuestion]


class GoldenSetError(ValueError):
    """Raised when the golden set file is malformed or labels a document that does not exist."""


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> GoldenSet:
    contents = json.loads(path.read_text(encoding="utf-8"))
    if "version" not in contents or "questions" not in contents:
        raise GoldenSetError(f"{path.name} needs a 'version' and a 'questions' list")
    questions = [_parse_question(entry) for entry in contents["questions"]]
    _reject_duplicate_ids(questions)
    return GoldenSet(version=contents["version"], questions=questions)


def ensure_labels_exist(golden_set: GoldenSet, document_ids: Collection[str]) -> None:
    """Fail loudly when a question is labeled with a document that is not in the corpus."""
    for question in golden_set.questions:
        unknown_documents = sorted(question.relevant_documents - set(document_ids))
        if unknown_documents:
            raise GoldenSetError(
                f"Question {question.question_id} is labeled with unknown documents: "
                f"{unknown_documents}"
            )


def _parse_question(entry: Mapping[str, object]) -> GoldenQuestion:
    missing_fields = [name for name in REQUIRED_FIELDS if name not in entry]
    if missing_fields:
        raise GoldenSetError(f"Golden set entry {dict(entry)} is missing {missing_fields}")

    question_id = str(entry["id"])
    try:
        kind = QuestionKind(entry["kind"])
    except ValueError:
        raise GoldenSetError(f"Question {question_id} has unknown kind {entry['kind']!r}") from None

    relevant_documents = frozenset(entry["relevant_documents"])
    if not relevant_documents:
        raise GoldenSetError(f"Question {question_id} has no relevant document")

    return GoldenQuestion(
        question_id=question_id,
        kind=kind,
        text=str(entry["question"]),
        relevant_documents=relevant_documents,
    )


def _reject_duplicate_ids(questions: list[GoldenQuestion]) -> None:
    id_counts = Counter(question.question_id for question in questions)
    repeated_ids = sorted(question_id for question_id, count in id_counts.items() if count > 1)
    if repeated_ids:
        raise GoldenSetError(f"Duplicate question ids: {repeated_ids}")
