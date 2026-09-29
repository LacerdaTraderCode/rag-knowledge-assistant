"""Tests for splitting text into overlapping chunks."""

from rag.chunking import split_into_chunks


def test_short_text_becomes_a_single_chunk():
    chunks = split_into_chunks("doc-a", "one two three")

    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].document_id == "doc-a"
    assert chunks[0].index == 0


def test_empty_text_produces_no_chunks():
    assert split_into_chunks("doc-a", "") == []
    assert split_into_chunks("doc-a", "   ") == []


def test_long_text_is_split_into_multiple_chunks():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)

    chunks = split_into_chunks("doc-a", text, words_per_chunk=120, overlap_words=20)

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-a" for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)

    chunks = split_into_chunks("doc-a", text, words_per_chunk=100, overlap_words=20)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-20:] == second_words[:20]


def test_no_word_from_the_original_text_is_dropped():
    words = [f"word{i}" for i in range(250)]
    text = " ".join(words)

    chunks = split_into_chunks("doc-a", text, words_per_chunk=100, overlap_words=15)

    covered_words: set[str] = set()
    for chunk in chunks:
        covered_words.update(chunk.text.split())
    assert covered_words == set(words)
