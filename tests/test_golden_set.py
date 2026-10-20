"""Tests that the golden set file is well formed and that its question types mean what they say."""

import json
import re

import pytest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from evaluation.golden_set import (
    GoldenQuestion,
    GoldenSetError,
    QuestionKind,
    ensure_labels_exist,
    load_golden_set,
)
from rag.sample_corpus import load_sample_documents

MIN_QUESTIONS = 40
MIN_QUESTIONS_PER_KIND = 10
MIN_LEXICAL_OVERLAP = 0.5
MAX_PARAPHRASE_OVERLAP = 0.45
COPIED_PHRASE_WORDS = 5


@pytest.fixture(scope="module")
def golden_set():
    return load_golden_set()


@pytest.fixture(scope="module")
def documents() -> dict[str, str]:
    return load_sample_documents()


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def content_words(text: str) -> set[str]:
    return {word for word in words_of(text) if word not in ENGLISH_STOP_WORDS and len(word) > 1}


def overlap_with_notes(question: GoldenQuestion, documents: dict[str, str]) -> float:
    question_words = content_words(question.text)
    note_words = set().union(
        *(content_words(documents[document_id]) for document_id in question.relevant_documents)
    )
    return len(question_words & note_words) / len(question_words)


def phrases_of(text: str, length: int) -> set[tuple[str, ...]]:
    words = words_of(text)
    return {tuple(words[start : start + length]) for start in range(len(words) - length + 1)}


def write_golden_set(path, questions: list[dict], version: int = 1):
    path.write_text(json.dumps({"version": version, "questions": questions}))
    return path


def valid_entry(**overrides) -> dict:
    entry = {
        "id": "q-1",
        "kind": "paraphrase",
        "question": "Why?",
        "relevant_documents": ["notes"],
    }
    return entry | overrides


def test_the_committed_golden_set_is_versioned_and_large_enough(golden_set):
    assert golden_set.version >= 1
    assert len(golden_set.questions) >= MIN_QUESTIONS


def test_every_question_type_has_enough_questions(golden_set):
    for kind in QuestionKind:
        count = sum(question.kind is kind for question in golden_set.questions)
        assert count >= MIN_QUESTIONS_PER_KIND, kind


def test_every_label_points_at_a_note_that_exists(golden_set, documents):
    ensure_labels_exist(golden_set, documents.keys())


def test_every_note_is_the_answer_to_at_least_one_question(golden_set, documents):
    labeled_documents = {
        document_id
        for question in golden_set.questions
        for document_id in question.relevant_documents
    }

    assert labeled_documents == set(documents)


def test_no_question_repeats_a_phrase_of_the_notes_verbatim(golden_set, documents):
    note_phrases = set().union(
        *(phrases_of(text, COPIED_PHRASE_WORDS) for text in documents.values())
    )

    for question in golden_set.questions:
        copied = phrases_of(question.text, COPIED_PHRASE_WORDS) & note_phrases
        assert not copied, (question.question_id, copied)


def test_lexical_overlap_questions_reuse_the_words_of_their_notes(golden_set, documents):
    for question in golden_set.questions:
        if question.kind is QuestionKind.LEXICAL_OVERLAP:
            overlap = overlap_with_notes(question, documents)
            assert overlap >= MIN_LEXICAL_OVERLAP, (question.question_id, overlap)


def test_paraphrase_questions_share_few_words_with_their_notes(golden_set, documents):
    for question in golden_set.questions:
        if question.kind is QuestionKind.PARAPHRASE:
            overlap = overlap_with_notes(question, documents)
            assert overlap <= MAX_PARAPHRASE_OVERLAP, (question.question_id, overlap)


def test_exact_term_questions_are_short_terms_found_in_their_notes(golden_set, documents):
    for question in golden_set.questions:
        if question.kind is QuestionKind.EXACT_TERM:
            assert len(question.text.split()) <= 4, question.question_id
            assert overlap_with_notes(question, documents) == 1.0, question.question_id


def test_a_valid_file_loads_with_its_labels(tmp_path):
    path = write_golden_set(tmp_path / "golden.json", [valid_entry()], version=3)

    loaded = load_golden_set(path)

    assert loaded.version == 3
    assert loaded.questions == [
        GoldenQuestion(
            question_id="q-1",
            kind=QuestionKind.PARAPHRASE,
            text="Why?",
            relevant_documents=frozenset({"notes"}),
        )
    ]


@pytest.mark.parametrize("missing_field", ["id", "kind", "question", "relevant_documents"])
def test_an_entry_missing_a_field_is_rejected(tmp_path, missing_field):
    entry = valid_entry()
    del entry[missing_field]
    path = write_golden_set(tmp_path / "golden.json", [entry])

    with pytest.raises(GoldenSetError, match=missing_field):
        load_golden_set(path)


def test_an_unknown_question_kind_is_rejected(tmp_path):
    path = write_golden_set(tmp_path / "golden.json", [valid_entry(kind="trick_question")])

    with pytest.raises(GoldenSetError, match="trick_question"):
        load_golden_set(path)


def test_a_question_without_relevant_documents_is_rejected(tmp_path):
    path = write_golden_set(tmp_path / "golden.json", [valid_entry(relevant_documents=[])])

    with pytest.raises(GoldenSetError, match="no relevant document"):
        load_golden_set(path)


def test_duplicate_question_ids_are_rejected(tmp_path):
    path = write_golden_set(tmp_path / "golden.json", [valid_entry(), valid_entry()])

    with pytest.raises(GoldenSetError, match="q-1"):
        load_golden_set(path)


def test_a_file_without_a_version_is_rejected(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"questions": [valid_entry()]}))

    with pytest.raises(GoldenSetError, match="version"):
        load_golden_set(path)


def test_a_label_for_a_missing_note_is_reported_with_the_question(tmp_path):
    path = write_golden_set(
        tmp_path / "golden.json", [valid_entry(relevant_documents=["notes", "ghost"])]
    )

    with pytest.raises(GoldenSetError, match=r"q-1.*ghost"):
        ensure_labels_exist(load_golden_set(path), {"notes"})
