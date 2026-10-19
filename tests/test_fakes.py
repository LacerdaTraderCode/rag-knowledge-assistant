"""The retrieval tests lean on the fake embedder, so its own guarantees are checked here."""

from tests.fakes import HashingEmbedder


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot_product / (left_norm * right_norm)


def test_the_same_text_always_produces_the_same_vector():
    embedder = HashingEmbedder()

    assert embedder.embed_query("cats purr") == embedder.embed_query("cats purr")


def test_vectors_have_the_declared_dimension():
    embedder = HashingEmbedder()

    assert len(embedder.embed_query("cats purr")) == embedder.dimension


def test_passages_and_queries_share_one_vector_space():
    embedder = HashingEmbedder()

    assert embedder.embed_passages(["cats purr"])[0] == embedder.embed_query("cats purr")


def test_texts_sharing_words_are_closer_than_unrelated_texts():
    embedder = HashingEmbedder()
    query = embedder.embed_query("cats purr loudly")

    related = cosine_similarity(query, embedder.embed_query("cats purr when happy"))
    unrelated = cosine_similarity(query, embedder.embed_query("rockets reach orbit"))

    assert related > unrelated
