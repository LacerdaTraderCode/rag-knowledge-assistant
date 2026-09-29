"""Splits raw text into overlapping chunks suitable for indexing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    document_id: str
    index: int
    text: str


def split_into_chunks(
    document_id: str, text: str, words_per_chunk: int = 120, overlap_words: int = 20
) -> list[Chunk]:
    """Split `text` into overlapping chunks of `words_per_chunk` words.

    Consecutive chunks share `overlap_words` words so an answer sitting near a
    chunk boundary is not split away from the context it needs.
    """
    words = text.split()
    if not words:
        return []

    step = words_per_chunk - overlap_words
    chunks: list[Chunk] = []
    index = 0
    start = 0

    while start < len(words):
        piece = words[start : start + words_per_chunk]
        chunks.append(Chunk(document_id=document_id, index=index, text=" ".join(piece)))
        index += 1
        if start + words_per_chunk >= len(words):
            break
        start += step

    return chunks
